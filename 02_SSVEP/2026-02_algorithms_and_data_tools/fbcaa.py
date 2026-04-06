import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

# ==========================================================
# 1. 读取 EEG 数据
# ==========================================================
mat = sio.loadmat(r'C:\Users\P1233\Desktop\脑电\数据集3\12264401 (1)\S1.mat')

data_struct = mat['data'][0, 0]
eeg_data = data_struct['EEG']     # 64 × 750 × 4 × 40
print("EEG shape:", eeg_data.shape)

fs = 250
block = 1
condition = 10

# ==========================================================
# 2. 选择多通道并取数据 (T × C)
# ==========================================================
channels = [53, 55, 56, 57, 59, 61, 62, 63]  # PO7 PO3 POz PO4 PO8 O1 Oz O2(按你数据集编号)

X_raw = eeg_data[channels, :, block, condition]  # (C, T)
X_raw = X_raw.T                                  # (T, C)

T = X_raw.shape[0]
time_points = np.arange(T) / fs

# ==========================================================
# 3. 公共预处理（去基线 + 50Hz 陷波）——注意：FBCCA 子带带通放后面做
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

X0 = np.zeros_like(X_raw)
for ci in range(len(channels)):
    X0[:, ci] = detrend_and_notch(X_raw[:, ci])

# 去均值
X0 = X0 - X0.mean(axis=0, keepdims=True)

# ==========================================================
# 4. 构造 CCA 参考矩阵（不变）
# ==========================================================
def build_ref_matrix(fs, L, f, Nh=3):
    T = int(fs * L)
    t = np.arange(1, T + 1) / fs
    cols = []
    for h in range(1, Nh + 1):
        cols.append(np.sin(2 * np.pi * h * f * t))
        cols.append(np.cos(2 * np.pi * h * f * t))
    Y = np.stack(cols, axis=1)
    Y -= Y.mean(axis=0, keepdims=True)
    return Y

L = 3
freqs = [round(8 + 0.2*i, 1) for i in range(int((15.8 - 8) / 0.2) + 1)]
Nh = 3
Y_refs = {f: build_ref_matrix(fs, L, f, Nh) for f in freqs}

# ==========================================================
# 5. 多通道 CCA（不变）
# ==========================================================
def cca_multi_channel(X, Y):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    Sxx = X.T @ X
    Syy = Y.T @ Y
    Sxy = X.T @ Y
    Syx = Sxy.T

    reg = 1e-6
    Sxx += reg * np.eye(Sxx.shape[0])
    Syy += reg * np.eye(Syy.shape[0])

    M = np.block([[np.zeros_like(Sxx), Sxy],
                  [Syx, np.zeros_like(Syy)]])
    N = np.block([[Sxx, np.zeros_like(Sxy)],
                  [np.zeros_like(Syx), Syy]])

    eigvals = np.linalg.eigvals(np.linalg.solve(N, M))
    rho = np.max(np.abs(eigvals))
    return float(rho)

# ==========================================================
# 6. FBCCA：定义子带 + 权重
# ==========================================================

subbands = [(6, 50), (10, 50), (14, 50), (18, 50), (22, 50)]
K = len(subbands)

# 权重：w_k = k^{-a} + b（k从1开始）
a_w = 1.25
b_w = 0.25
weights = np.array([(k+1)**(-a_w) + b_w for k in range(K)], dtype=float)
# 可选：归一化让数值更稳定（不影响argmax）
weights = weights / weights.sum()

def bandpass_filter_multichannel(X_in, fl, fh, fs):
    """X_in: (T,C) -> bandpass -> (T,C)"""
    b, a = butter(2, [fl/(fs/2), fh/(fs/2)], btype='band')
    X_out = np.zeros_like(X_in)
    for ci in range(X_in.shape[1]):
        X_out[:, ci] = filtfilt(b, a, X_in[:, ci])
    X_out = X_out - X_out.mean(axis=0, keepdims=True)
    return X_out

# ==========================================================
# 7. 对每个频率计算 FBCCA score(f)
# ==========================================================
scores = []
# （可选）保存每个子带的rho，便于你调试/看贡献
rho_bank = np.zeros((K, len(freqs)), dtype=float)

for fi, f in enumerate(freqs):
    Yf = Y_refs[f]
    score_f = 0.0
    for k, (fl, fh) in enumerate(subbands):
        Xk = bandpass_filter_multichannel(X0, fl, fh, fs)
        rho_k = cca_multi_channel(Xk, Yf)
        rho_bank[k, fi] = rho_k
        score_f += weights[k] * (rho_k ** 2)
    scores.append(score_f)
    print(f"{f} Hz → score = {score_f:.6f}")

scores = np.array(scores)

pred_f = freqs[int(np.argmax(scores))]
print("\nFBCCA 预测目标频率：", pred_f, "Hz")

# ==========================================================
# 8. 可视化 score
# ==========================================================
fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
y_pos = np.arange(len(freqs))
ax.barh(y_pos, scores, edgecolor='black')
ax.barh(np.argmax(scores), scores.max(), edgecolor='black')

for i, v in enumerate(scores):
    ax.text(v + 0.0005, i, f"{v:.4f}", va='center')

ax.set_yticks(y_pos)
ax.set_yticklabels([f"{f} Hz" for f in freqs])
ax.set_xlabel("FBCCA score")
ax.set_title("FBCCA 结果（固定高截止 + 低截止递增）")
ax.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
