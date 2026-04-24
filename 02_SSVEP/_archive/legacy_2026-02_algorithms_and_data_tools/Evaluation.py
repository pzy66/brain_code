import scipy.io as sio
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

# ==========================================================
# 全局参数
# ==========================================================
fs = 250
Nh = 3

# 你的通道（枕区 8 个）: PO7, PO3, POz, PO4, PO8, O1, Oz, O2
channels = [53, 55, 56, 57, 59, 61, 62, 63]

# 你的频率集合（和你原来一致）
unique_freqs = [round(8 + 0.2 * i, 1) for i in range(int((15.8 - 8) / 0.2) + 1)]

# ==========================================================
# 滤波函数
# ==========================================================
def bandpass_notch_detrend(x_raw):
    # 去趋势：3 Hz 低通当基线
    b, a = butter(1, 3/(fs/2), btype='low')
    base = filtfilt(b, a, x_raw)
    x1 = x_raw - base

    # 50 Hz 陷波
    b, a = iirnotch(50, 30, fs)
    x2 = filtfilt(b, a, x1)

    # 4-50 Hz 带通
    b, a = butter(2, [4/(fs/2), 50/(fs/2)], btype='band')
    x3 = filtfilt(b, a, x2)

    return x3

# ==========================================================
# 构造参考矩阵（按点数 T 构造，避免 750/1000 不一致）
# ==========================================================
def build_ref_matrix_by_T(fs, T, f, Nh=3):
    t = np.arange(1, T + 1) / fs
    cols = []
    for h in range(1, Nh + 1):
        cols.append(np.sin(2*np.pi*h*f*t))
        cols.append(np.cos(2*np.pi*h*f*t))
    Y = np.stack(cols, axis=1)  # (T, 2*Nh)
    Y -= Y.mean(axis=0, keepdims=True)
    return Y

# ==========================================================
# 多通道CCA（保持你原思路）
# ==========================================================
def cca_multi_channel(X, Y):
    # X: (T, C), Y: (T, 2Nh)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    Sxx = X.T @ X
    Syy = Y.T @ Y
    Sxy = X.T @ Y
    Syx = Sxy.T

    reg = 1e-6
    Sxx = Sxx + reg * np.eye(Sxx.shape[0])
    Syy = Syy + reg * np.eye(Syy.shape[0])

    M = np.block([[np.zeros_like(Sxx), Sxy],
                  [Syx, np.zeros_like(Syy)]])
    N = np.block([[Sxx, np.zeros_like(Sxy)],
                  [np.zeros_like(Syx), Syy]])

    eigvals = np.linalg.eigvals(np.linalg.solve(N, M))
    rho = np.max(np.abs(eigvals))
    return float(rho)

# ==========================================================
# 预生成两套参考矩阵（只做一次）
# ==========================================================
print("预生成参考矩阵：T=750 和 T=1000 ...")
Y_refs_750  = {f: build_ref_matrix_by_T(fs, 750,  f, Nh) for f in unique_freqs}
Y_refs_1000 = {f: build_ref_matrix_by_T(fs, 1000, f, Nh) for f in unique_freqs}
freq_list = list(unique_freqs)
print("参考矩阵完成。")

# ==========================================================
# 整体统计
# ==========================================================
total_correct = 0
total_trials = 0
subject_accuracies = []

print("\n==========  开始 S1 ～ S70 多通道 + 多block 计算  ==========")

for sid in range(1, 71):
    filename = fr"C:\Users\P1233\Desktop\脑电\数据集3\12264401 (1)\S{sid}.mat"
    print(f"\n====== 处理 S{sid}.mat ======")

    mat = sio.loadmat(filename)
    data_struct = mat['data'][0, 0]
    eeg_data = data_struct['EEG']  # (64, T, 4, 40)

    # 取真实频率标签
    true_freqs = data_struct['suppl_info'][0, 0]['freqs'][0]  # (40,)

    # 根据你的规则选参考库 + 校验点数
    T = eeg_data.shape[1]
    if 1 <= sid <= 15:
        if T != 750:
            raise ValueError(f"S{sid} 期望750点，但实际是 {T}")
        Y_refs = Y_refs_750
    elif 16 <= sid <= 70:
        if T != 1000:
            raise ValueError(f"S{sid} 期望1000点，但实际是 {T}")
        Y_refs = Y_refs_1000
    else:
        raise ValueError(f"sid={sid} 不在 1~70 范围内")

    correct = 0
    total = 0  # 40 cond × 4 block = 160

    # 遍历 block 0–3
    for block in range(4):
        # 遍历 40 个 condition
        for cond in range(40):
            # (T, C)
            X_raw = eeg_data[channels, :, block, cond].T

            # 滤波（逐通道）
            X = np.zeros_like(X_raw)
            for ci in range(len(channels)):
                X[:, ci] = bandpass_notch_detrend(X_raw[:, ci])
            X -= X.mean(axis=0, keepdims=True)

            true_f = float(true_freqs[cond])

            # CCA 分类
            rhos = []
            for f in freq_list:
                rho = cca_multi_channel(X, Y_refs[f])
                rhos.append(rho)

            pred_f = freq_list[int(np.argmax(rhos))]

            if abs(pred_f - true_f) < 1e-8:
                correct += 1
            total += 1

    acc = correct / total * 100
    subject_accuracies.append(acc)

    print(f"S{sid} 准确率：{acc:.2f}%   ({correct}/{total})")

    total_correct += correct
    total_trials += total

# ==========================================================
# 最终总体准确率
# ==========================================================
overall_acc = total_correct / total_trials * 100
print("\n================ 最终结果 ================")
print(f"总正确数：{total_correct}/{total_trials}")
print(f"总体准确率：{overall_acc:.2f}%")
print("每个被试的准确率：")
print(subject_accuracies)
