import time
import argparse
from collections import deque
from datetime import datetime

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


class RealTimeFBCCA:
    def __init__(
        self,
        serial_port="COM3",
        freqs=(8, 10, 12, 15),
        win_sec=3.0,
        step_sec=0.5,
        score_threshold=0.02,
        conf_ratio_th=1.10,
        history_len=5,
        board_id=BoardIds.CYTON_BOARD.value,
        notch_freq=50.0,
        notch_q=30.0,
        verbose=True,
    ):
        # ===== 设备参数 =====
        self.serial_port = serial_port
        self.board_id = board_id

        # ===== SSVEP / FBCCA 参数 =====
        self.freqs = list(freqs)
        self.Nh = 3
        self.subbands = [(6, 50), (10, 50), (14, 50), (18, 50), (22, 50)]
        self.K = len(self.subbands)

        a_w = 1.25
        b_w = 0.25
        self.weights = np.array([(k + 1) ** (-a_w) + b_w for k in range(self.K)], dtype=float)
        self.weights = self.weights / self.weights.sum()

        # ===== 实时参数 =====
        self.win_sec = float(win_sec)          # 3秒视窗
        self.step_sec = float(step_sec)        # 0.5秒更新一次
        self.score_threshold = float(score_threshold)   # 绝对阈值：全低于它 -> 输出“无”
        self.conf_ratio_th = float(conf_ratio_th)       # 相对阈值：第一名/第二名太接近 -> “不确定”
        self.notch_freq = float(notch_freq)
        self.notch_q = float(notch_q)
        self.verbose = verbose

        # 历史平滑
        self.pred_history = deque(maxlen=int(history_len))

        # 运行时变量
        self.fs = None
        self.eeg_channels = None
        self.win_samples = None
        self.Y_refs = None

    # ==========================================================
    # 1) 构造参考矩阵
    # ==========================================================
    @staticmethod
    def build_ref_matrix(fs, T, f, Nh=3):
        """
        返回形状: (T, 2*Nh)
        """
        t = np.arange(T) / fs
        cols = []
        for h in range(1, Nh + 1):
            cols.append(np.sin(2 * np.pi * h * f * t))
            cols.append(np.cos(2 * np.pi * h * f * t))
        Y = np.stack(cols, axis=1)
        Y = Y - Y.mean(axis=0, keepdims=True)
        return Y

    # ==========================================================
    # 2) 预处理：去基线 + 50Hz 陷波
    # ==========================================================
    def detrend_and_notch(self, x_raw_1d):
        # 用低通估计慢漂移基线，再相减
        b_lp, a_lp = butter(1, 3 / (self.fs / 2), btype="low")
        base = filtfilt(b_lp, a_lp, x_raw_1d)
        x1 = x_raw_1d - base

        # 50 Hz 陷波
        b_notch, a_notch = iirnotch(self.notch_freq, self.notch_q, self.fs)
        x2 = filtfilt(b_notch, a_notch, x1)
        return x2

    def preprocess_window(self, X_raw):
        """
        X_raw: (T, C)
        """
        X0 = np.zeros_like(X_raw, dtype=float)
        for ci in range(X_raw.shape[1]):
            X0[:, ci] = self.detrend_and_notch(X_raw[:, ci])

        X0 = X0 - X0.mean(axis=0, keepdims=True)
        return X0

    # ==========================================================
    # 3) 多通道 CCA（SVD 版）
    # ==========================================================
    @staticmethod
    def cca_multi_channel_svd(X, Y, reg=1e-8):
        """
        X: (T, C)
        Y: (T, 2Nh)
        """
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)

        denom = max(X.shape[0] - 1, 1)
        Sxx = X.T @ X / denom
        Syy = Y.T @ Y / denom
        Sxy = X.T @ Y / denom

        Sxx += reg * np.eye(Sxx.shape[0])
        Syy += reg * np.eye(Syy.shape[0])

        ex, vx = np.linalg.eigh(Sxx)
        ey, vy = np.linalg.eigh(Syy)
        ex = np.maximum(ex, reg)
        ey = np.maximum(ey, reg)

        Sxx_inv_sqrt = vx @ np.diag(1.0 / np.sqrt(ex)) @ vx.T
        Syy_inv_sqrt = vy @ np.diag(1.0 / np.sqrt(ey)) @ vy.T

        Tmat = Sxx_inv_sqrt @ Sxy @ Syy_inv_sqrt
        s = np.linalg.svd(Tmat, compute_uv=False)
        return float(np.max(s))

    # ==========================================================
    # 4) 子带滤波
    # ==========================================================
    def bandpass_filter_multichannel(self, X_in, fl, fh):
        """
        X_in: (T, C)
        """
        b, a = butter(2, [fl / (self.fs / 2), fh / (self.fs / 2)], btype="band")
        X_out = np.zeros_like(X_in, dtype=float)
        for ci in range(X_in.shape[1]):
            X_out[:, ci] = filtfilt(b, a, X_in[:, ci])

        X_out = X_out - X_out.mean(axis=0, keepdims=True)
        return X_out

    # ==========================================================
    # 5) 单窗口 FBCCA 分类
    # ==========================================================
    def classify_window(self, X_window):
        """
        X_window: (T, C)

        返回:
        {
            "decision": "none" / "uncertain" / "freq",
            "pred_f": None 或 某个频率,
            "scores": ndarray,
            "max_score": float,
            "second_score": float,
            "ratio": float
        }
        """
        # 预处理
        X0 = self.preprocess_window(X_window)

        # 先把所有子带结果算好，避免对每个频率重复滤波
        X_subbands = []
        for fl, fh in self.subbands:
            Xk = self.bandpass_filter_multichannel(X0, fl, fh)
            X_subbands.append(Xk)

        # 对每个候选频率打分
        scores = np.zeros(len(self.freqs), dtype=float)

        for fi, f in enumerate(self.freqs):
            Yf = self.Y_refs[f]
            score_f = 0.0
            for k, Xk in enumerate(X_subbands):
                rho_k = self.cca_multi_channel_svd(Xk, Yf)
                score_f += self.weights[k] * (rho_k ** 2)
            scores[fi] = score_f

        # 排名
        best_idx = int(np.argmax(scores))
        max_score = float(scores[best_idx])
        pred_f = self.freqs[best_idx]

        scores_sorted = np.sort(scores)[::-1]
        second_score = float(scores_sorted[1]) if len(scores_sorted) >= 2 else 0.0
        ratio = max_score / (second_score + 1e-12) if second_score > 0 else np.inf

        # 判决逻辑
        # 1) 四个候选频率分数都低于阈值 -> 输出“无”
        if max_score < self.score_threshold:
            decision = "none"
            pred_f = None

        # 2) 最大分数虽然过了绝对阈值，但第一名和第二名太接近 -> 不确定
        elif ratio < self.conf_ratio_th:
            decision = "uncertain"

        # 3) 否则输出该频率
        else:
            decision = "freq"

        return {
            "decision": decision,
            "pred_f": pred_f,
            "scores": scores,
            "max_score": max_score,
            "second_score": second_score,
            "ratio": float(ratio),
        }

    # ==========================================================
    # 6) 简单历史平滑
    # ==========================================================
    def smooth_prediction(self, pred_f):
        self.pred_history.append(pred_f)
        vals = [x for x in self.pred_history if x is not None]
        if not vals:
            return None
        return max(set(vals), key=vals.count)

    # ==========================================================
    # 7) 主循环
    # ==========================================================
    def run(self):
        BoardShim.enable_dev_board_logger()

        params = BrainFlowInputParams()
        params.serial_port = self.serial_port

        board = None

        try:
            # ---------- 连接设备 ----------
            board = BoardShim(self.board_id, params)
            board.prepare_session()

            self.fs = BoardShim.get_sampling_rate(self.board_id)
            self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
            self.win_samples = int(round(self.win_sec * self.fs))

            if self.win_samples < 64:
                raise ValueError("窗口太短，至少保证几十个采样点。")

            # 参考矩阵只需按窗口长度建一次
            self.Y_refs = {
                f: self.build_ref_matrix(self.fs, self.win_samples, f, self.Nh)
                for f in self.freqs
            }

            if self.verbose:
                print("=" * 70)
                print("实时 SSVEP-FBCCA 分类器")
                print("=" * 70)
                print(f"串口: {self.serial_port}")
                print(f"Board ID: {self.board_id}")
                print(f"采样率: {self.fs} Hz")
                print(f"EEG通道索引: {self.eeg_channels}")
                print(f"目标频率: {self.freqs}")
                print(f"窗口: {self.win_sec:.2f} s ({self.win_samples} 点)")
                print(f"步长: {self.step_sec:.2f} s")
                print(f"绝对阈值 score_threshold: {self.score_threshold:.4f}")
                print(f"相对阈值 conf_ratio_th: {self.conf_ratio_th:.4f}")
                print("开始采集...\n")

            # ---------- 开始流 ----------
            board.start_stream(450000)

            # 等缓冲区攒够一个完整窗口
            while board.get_board_data_count() < self.win_samples:
                cnt = board.get_board_data_count()
                print(f"等待缓冲区数据... {cnt}/{self.win_samples}", end="\r", flush=True)
                time.sleep(0.1)
            print(" " * 80, end="\r")

            # 定时更新
            next_update = time.time()

            while True:
                now = time.time()
                sleep_time = next_update - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # 如果计算偶尔超时，就把下次更新点重置到当前时间，避免越拖越慢
                    next_update = now
                next_update += self.step_sec

                available = board.get_board_data_count()
                if available < self.win_samples:
                    print(f"[WARN] 当前可用点数不足: {available}/{self.win_samples}")
                    continue

                # 取最近一个窗口的数据
                data = board.get_current_board_data(self.win_samples)
                if data.shape[1] < self.win_samples:
                    print(f"[WARN] 实际取到点数不足: {data.shape[1]}/{self.win_samples}")
                    continue

                # BrainFlow 返回 [channels x datapoints]
                # 这里只取 EEG 行，再转成 (T, C)
                X_window = data[self.eeg_channels, -self.win_samples:].T

                # 分类
                result = self.classify_window(X_window)
                pred_f = result["pred_f"]
                scores = result["scores"]
                max_score = result["max_score"]
                ratio = result["ratio"]
                decision = result["decision"]

                score_str = " | ".join(
                    [f"{f}Hz:{s:.4f}" for f, s in zip(self.freqs, scores)]
                )
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                if decision == "none":
                    self.pred_history.append(None)
                    print(
                        f"[{ts}] 输出: 无 | max_score={max_score:.4f} < "
                        f"threshold={self.score_threshold:.4f} | {score_str}"
                    )

                elif decision == "uncertain":
                    self.pred_history.append(None)
                    print(
                        f"[{ts}] 输出: 不确定 | raw={pred_f}Hz | "
                        f"max_score={max_score:.4f}, ratio={ratio:.3f} < "
                        f"{self.conf_ratio_th:.3f} | {score_str}"
                    )

                else:
                    smooth_pred = self.smooth_prediction(pred_f)
                    print(
                        f"[{ts}] 输出: {pred_f}Hz | 平滑: {smooth_pred}Hz | "
                        f"max_score={max_score:.4f}, ratio={ratio:.3f} | {score_str}"
                    )

        except KeyboardInterrupt:
            print("\n手动停止。")

        except Exception as e:
            print(f"\n错误: {e}")

        finally:
            if board is not None:
                try:
                    board.stop_stream()
                except Exception:
                    pass
                try:
                    board.release_session()
                except Exception:
                    pass
            print("设备已释放。")


def parse_args():
    parser = argparse.ArgumentParser(description="实时 SSVEP-FBCCA 分类器")
    parser.add_argument("--port", type=str, default="COM3", help="串口号，例如 COM3")
    parser.add_argument("--win-sec", type=float, default=3.0, help="分类窗口长度（秒）")
    parser.add_argument("--step-sec", type=float, default=0.5, help="更新步长（秒）")
    parser.add_argument("--score-th", type=float, default=0.02, help="绝对分数阈值，小于它则输出无")
    parser.add_argument("--ratio-th", type=float, default=1.10, help="第一名/第二名分数比阈值，小于它则输出不确定")
    parser.add_argument("--freqs", type=str, default="8,10,12,15", help="目标频率列表，逗号分隔")
    return parser.parse_args()


def main():
    args = parse_args()
    freqs = [float(x) for x in args.freqs.split(",") if x.strip()]

    clf = RealTimeFBCCA(
        serial_port=args.port,
        freqs=freqs,
        win_sec=args.win_sec,
        step_sec=args.step_sec,
        score_threshold=args.score_th,
        conf_ratio_th=args.ratio_th,
    )
    clf.run()


if __name__ == "__main__":
    main()