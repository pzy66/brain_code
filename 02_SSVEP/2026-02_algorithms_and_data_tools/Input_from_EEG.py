import argparse
import time
import numpy as np
from collections import deque
from scipy.signal import butter, iirnotch, lfilter
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


# ====== 1. 初始化滤波器 ======
def init_filters(fs):
    notch_b, notch_a = iirnotch(50, 30, fs)   ##去除工频干扰 50Hz陷波滤波
    band_b, band_a = butter(2, [4/(fs/2), 50/(fs/2)], btype='band')   ##带通滤波4-50Hz
    return (notch_b, notch_a, band_b, band_a)


# ====== 2. 滑动滤波（带状态） ======
def sliding_filter(new_data, notch, band, zi_notch, zi_band):
    y_notch, zi_notch = lfilter(notch[0], notch[1], new_data, zi=zi_notch)
    y_band, zi_band = lfilter(band[0], band[1], y_notch, zi=zi_band)
    return y_band, zi_notch, zi_band


# ====== 3. 构造参考信号矩阵 ======
def build_ref_matrix(fs, L, f, Nh=3):
    T = int(fs * L)
    t = np.arange(1, T + 1) / fs
    cols = []
    for h in range(1, Nh + 1):
        cols.append(np.sin(2 * np.pi * h * f * t))
        cols.append(np.cos(2 * np.pi * h * f * t))
    return np.stack(cols, axis=1)


# ====== 4. CCA识别 ======
def cca_rho_regression(X, Y):
    X = X - np.mean(X)
    Y = Y - np.mean(Y, axis=0)
    G = Y.T @ Y + 1e-8 * np.eye(Y.shape[1])
    beta = np.linalg.solve(G, Y.T @ X)
    x_hat = Y @ beta
    rho = np.linalg.norm(x_hat) / np.linalg.norm(X)
    return float(rho)


def classify_ssvep(sig, Y):
    X = sig[:, np.newaxis]
    rhos = [cca_rho_regression(X, Yf) for Yf in Y.values()]
    best_f = list(Y.keys())[int(np.argmax(rhos))]
    return best_f, rhos


# ====== 5. 主函数 ======
def main():
    BoardShim.enable_dev_board_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial-port', type=str, default='COM4')
    parser.add_argument('--board-id', type=int, default=BoardIds.SYNTHETIC_BOARD.value)
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.serial_port = args.serial_port

    fs = 256                # 采样率（根据你的设备调整）
    window_length = 3.0     # 滑动窗口长度（秒）
    step = 0.5              # 每0.5秒更新一次识别
    freqs = [9, 9.6, 9.8, 10, 10.2, 10.4, 10.6]  # 参考频率
    Nh = 3

    # 初始化滤波器与缓存
    notch_b, notch_a, band_b, band_a = init_filters(fs)
    zi_notch = np.zeros(max(len(notch_a), len(notch_b)) - 1)
    zi_band = np.zeros(max(len(band_a), len(band_b)) - 1)
    buffer = deque(maxlen=int(fs * window_length))

    # 构造参考矩阵
    Y = {f: build_ref_matrix(fs, window_length, f, Nh) for f in freqs}

    try:
        board = BoardShim(args.board_id, params)
        board.prepare_session()
        board.start_stream()
        print("🧠 开始实时采集与滑动识别... (Ctrl+C退出)")

        last_time = time.time()
        while True:
            data = board.get_current_board_data(int(fs * step))  # 每次取0.5秒的新数据
            if data.shape[1] == 0:
                continue

            # 取EEG通道（前8个通道）
            eeg_chunk = np.mean(data[1:9, :], axis=0)  # 取8通道平均或单独选择某通道

            # 滑动滤波
            y_notch, zi_notch = lfilter(notch_b, notch_a, eeg_chunk, zi=zi_notch)
            y_band, zi_band = lfilter(band_b, band_a, y_notch, zi=zi_band)

            buffer.extend(y_band)

            # 当积满窗口时识别
            if len(buffer) == int(fs * window_length):
                sig = np.array(buffer)
                freq, rhos = classify_ssvep(sig, Y)
                print(f"预测频率：{freq:.2f} Hz, ρ_max = {max(rhos):.3f}")

            # 控制循环速率
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("🛑 停止采集")
        board.stop_stream()
        board.release_session()

    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    main()
