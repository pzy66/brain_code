import scipy.io as sio
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

# ==========================================================
# 全局参数
# ==========================================================
fs = 250
Nh = 3

# 通道（枕区 8 个）
channels = [53, 55, 56, 57, 59, 61, 62, 63]  # PO7 PO3 POz PO4 PO8 O1 Oz O2

# 候选频率集合（与你一致）
freqs = [round(8 + 0.2 * i, 1) for i in range(int((15.8 - 8) / 0.2) + 1)]
freq_list = list(freqs)

# ==========================================================
# 公共预处理：去基线 + 50Hz 陷波
# ==========================================================
def detrend_and_notch(x_raw_1d):
    # 去基线（低通估计基线再相减）
    b, a = butter(1, 3/(fs/2), btype='low')
    base = filtfilt(b, a, x_raw_1d)
    x1 = x_raw_1d - base

    # 50Hz 陷波
    b, a = iirnotch(50, 30, fs)
    x2 = filtfilt(b, a, x1)
    return x2

# ==========================================================
# 带通滤波（多通道），与你示例一致：逐通道 filtfilt + 去均值
# ==========================================================
def bandpass_filter_multichannel(X_in, fl, fh, fs):
    """X_in: (T,C) -> bandpass -> (T,C)"""
    b, a = butter(2, [fl/(fs/2), fh/(fs/2)], btype='band')
    X_out = np.zeros_like(X_in)
    for ci in range(X_in.shape[1]):
        X_out[:, ci] = filtfilt(b, a, X_in[:, ci])
    X_out = X_out - X_out.mean(axis=0, keepdims=True)
    return X_out

# ==========================================================
# 构造参考矩阵：按 T（点数）构造，避免 750/1000 不匹配
# ==========================================================
def build_ref_matrix_by_T(fs, T, f, Nh=3):
    t = np.arange(1, T + 1) / fs
    cols = []
    for h in range(1, Nh + 1):
        cols.append(np.sin(2 * np.pi * h * f * t))
        cols.append(np.cos(2 * np.pi * h * f * t))
    Y = np.stack(cols, axis=1)            # (T, 2Nh)
    Y -= Y.mean(axis=0, keepdims=True)
    return Y

# ==========================================================
# 多通道 CCA（不变）
# ==========================================================
def cca_multi_channel(X, Y):
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
# CCA 预测（单一带通：4-50Hz，和你之前 CCA 代码一致）
# ==========================================================
def predict_cca(X_dn, Y_refs, freq_list):
    # 这里做一次 4-50Hz 带通，作为 CCA baseline
    X_cca = bandpass_filter_multichannel(X_dn, 4, 50, fs)

    rhos = []
    for f in freq_list:
        rhos.append(cca_multi_channel(X_cca, Y_refs[f]))
    return freq_list[int(np.argmax(rhos))]

# ==========================================================
# FBCCA：按你给的范式一（固定高截止 50，低截止递增）
# ==========================================================
subbands = [(6, 50), (10, 50), (14, 50), (18, 50), (22, 50)]
K = len(subbands)

a_w = 1.25
b_w = 0.25
weights = np.array([(k+1)**(-a_w) + b_w for k in range(K)], dtype=float)
weights = weights / weights.sum()  # 归一化（不影响argmax）

def predict_fbcca(X_dn, Y_refs, freq_list):
    """
    X_dn: (T,C) 已完成 去基线+陷波+去均值 的数据
    FBCCA score(f) = sum_k w_k * rho_k(f)^2
    重要优化：每个 trial 的子带滤波只做一次
    """
    # 1) 先把每个子带的 Xk 都算出来（只算一次）
    Xk_list = []
    for (fl, fh) in subbands:
        Xk_list.append(bandpass_filter_multichannel(X_dn, fl, fh, fs))

    # 2) 对每个频率算加权分数
    scores = np.zeros(len(freq_list), dtype=float)
    for fi, f in enumerate(freq_list):
        Yf = Y_refs[f]
        s = 0.0
        for k in range(K):
            rho_k = cca_multi_channel(Xk_list[k], Yf)
            s += weights[k] * (rho_k ** 2)
        scores[fi] = s

    return freq_list[int(np.argmax(scores))]

# ==========================================================
# 预生成两套参考矩阵（只做一次）
# ==========================================================
print("预生成参考矩阵：T=750 和 T=1000 ...")
Y_refs_750  = {f: build_ref_matrix_by_T(fs, 750,  f, Nh) for f in freq_list}
Y_refs_1000 = {f: build_ref_matrix_by_T(fs, 1000, f, Nh) for f in freq_list}
print("参考矩阵完成。")

# ==========================================================
# 主循环：对比 CCA vs FBCCA
# ==========================================================
cca_subject_acc = []
fbcca_subject_acc = []

total_correct_cca = 0
total_correct_fbcca = 0
total_trials = 0

print("\n==========  开始 S1 ～ S70：CCA vs FBCCA  ==========")

for sid in range(1, 71):
    filename = fr"C:\Users\P1233\Desktop\脑电\数据集3\12264401 (1)\S{sid}.mat"
    print(f"\n====== 处理 S{sid}.mat ======")

    mat = sio.loadmat(filename)
    data_struct = mat['data'][0, 0]
    eeg_data = data_struct['EEG']  # (64, T, 4, 40)
    true_freqs = data_struct['suppl_info'][0, 0]['freqs'][0]  # (40,)

    T = eeg_data.shape[1]
    if 1 <= sid <= 15:
        if T != 750:
            raise ValueError(f"S{sid} 期望750点，但实际是 {T}")
        Y_refs = Y_refs_750
    else:
        if T != 1000:
            raise ValueError(f"S{sid} 期望1000点，但实际是 {T}")
        Y_refs = Y_refs_1000

    correct_cca = 0
    correct_fbcca = 0
    total = 0

    for block in range(4):
        for cond in range(40):
            # 取多通道 (T,C)
            X_raw = eeg_data[channels, :, block, cond].T

            # 公共预处理：去基线 + 陷波（逐通道）
            X_dn = np.zeros_like(X_raw)
            for ci in range(X_raw.shape[1]):
                X_dn[:, ci] = detrend_and_notch(X_raw[:, ci])

            # 去均值（与你示例一致）
            X_dn = X_dn - X_dn.mean(axis=0, keepdims=True)

            true_f = float(true_freqs[cond])

            # 1) CCA
            pred_f_cca = predict_cca(X_dn, Y_refs, freq_list)

            # 2) FBCCA（按你给的范式一）
            pred_f_fbcca = predict_fbcca(X_dn, Y_refs, freq_list)

            if abs(pred_f_cca - true_f) < 1e-8:
                correct_cca += 1
            if abs(pred_f_fbcca - true_f) < 1e-8:
                correct_fbcca += 1

            total += 1

    acc_cca = correct_cca / total * 100
    acc_fbcca = correct_fbcca / total * 100

    cca_subject_acc.append(acc_cca)
    fbcca_subject_acc.append(acc_fbcca)

    total_correct_cca += correct_cca
    total_correct_fbcca += correct_fbcca
    total_trials += total

    print(f"S{sid}  CCA={acc_cca:.2f}%  FBCCA={acc_fbcca:.2f}%   "
          f"(CCA {correct_cca}/{total}, FBCCA {correct_fbcca}/{total})")

# ==========================================================
# 总体结果
# ==========================================================
overall_cca = total_correct_cca / total_trials * 100
overall_fbcca = total_correct_fbcca / total_trials * 100
diff = [fb - cc for fb, cc in zip(fbcca_subject_acc, cca_subject_acc)]

print("\n================ 最终对比结果 ================")
print(f"CCA   总正确数：{total_correct_cca}/{total_trials}  总体准确率：{overall_cca:.2f}%")
print(f"FBCCA 总正确数：{total_correct_fbcca}/{total_trials}  总体准确率：{overall_fbcca:.2f}%")

print("\n每个被试准确率（CCA）：")
print(cca_subject_acc)

print("\n每个被试准确率（FBCCA）：")
print(fbcca_subject_acc)

print("\n每个被试差值（FBCCA-CCA，单位%）：")
print(diff)
