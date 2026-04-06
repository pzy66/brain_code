import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

# ==========================================================
# 0) 参数
# ==========================================================
csv_path = r"C:\Users\P1233\Desktop\脑电\大创\LTM-12_4.csv"
fs = 250

channels = ["EEG_Channel_1","EEG_Channel_2","EEG_Channel_3","EEG_Channel_4",
            "EEG_Channel_5","EEG_Channel_6","EEG_Channel_7","EEG_Channel_8"]

freqs = [8,10,12,15]
Nh = 3

subbands = [(6, 50), (10, 50), (14, 50), (18, 50), (22, 50)]
K = len(subbands)

a_w = 1.25
b_w = 0.25
weights = np.array([(k+1)**(-a_w) + b_w for k in range(K)], dtype=float)
weights = weights / weights.sum()

# ==========================================================
# 1) 读取 CSV -> (T, C)，并自动用实际长度 T
# ==========================================================
df = pd.read_csv(csv_path)

# 从第3行开始：跳过第2行全0（索引=1）
X_raw = df.loc[1:, channels].astype(float).values  # (T_total, 8)

T = X_raw.shape[0]          # 这里就是 702
L_sec = T / fs              # 实际时长（秒）
print(f"实际可用点数 T={T}, 时长 L_sec={L_sec:.3f} s")

time_points = np.arange(T) / fs

# ==========================================================
# 2) 公共预处理：3Hz 去基线 + 50Hz 陷波
# ==========================================================
def detrend_and_notch(x_raw_1d, fs):
    b, a = butter(1, 3/(fs/2), btype='low')
    base = filtfilt(b, a, x_raw_1d)
    x1 = x_raw_1d - base

    b, a = iirnotch(50, 30, fs)
    x2 = filtfilt(b, a, x1)
    return x2

X0 = np.zeros_like(X_raw)
for ci in range(X_raw.shape[1]):
    X0[:, ci] = detrend_and_notch(X_raw[:, ci], fs)

X0 = X0 - X0.mean(axis=0, keepdims=True)

# ==========================================================
# 3) 构造参考矩阵：用“实际 T”点数（702点）！
# ==========================================================
def build_ref_matrix(fs, T, f, Nh=3):
    t = np.arange(1, T + 1) / fs
    cols = []
    for h in range(1, Nh + 1):
        cols.append(np.sin(2 * np.pi * h * f * t))
        cols.append(np.cos(2 * np.pi * h * f * t))
    Y = np.stack(cols, axis=1)     # (T, 2Nh)
    Y = Y - Y.mean(axis=0, keepdims=True)
    return Y

Y_refs = {f: build_ref_matrix(fs, T, f, Nh) for f in freqs}

# ==========================================================
# 4) 多通道 CCA（SVD版，稳定）
# ==========================================================
def cca_multi_channel_svd(X, Y, reg=1e-8):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    Sxx = X.T @ X / (X.shape[0] - 1)
    Syy = Y.T @ Y / (Y.shape[0] - 1)
    Sxy = X.T @ Y / (X.shape[0] - 1)

    Sxx += reg * np.eye(Sxx.shape[0])
    Syy += reg * np.eye(Syy.shape[0])

    ex, vx = np.linalg.eigh(Sxx)
    ey, vy = np.linalg.eigh(Syy)
    ex = np.maximum(ex, reg)
    ey = np.maximum(ey, reg)

    Sxx_inv_sqrt = vx @ np.diag(1.0/np.sqrt(ex)) @ vx.T
    Syy_inv_sqrt = vy @ np.diag(1.0/np.sqrt(ey)) @ vy.T

    Tmat = Sxx_inv_sqrt @ Sxy @ Syy_inv_sqrt
    s = np.linalg.svd(Tmat, compute_uv=False)
    return float(np.max(s))

# ==========================================================
# 5) 子带带通
# ==========================================================
def bandpass_filter_multichannel(X_in, fl, fh, fs):
    b, a = butter(2, [fl/(fs/2), fh/(fs/2)], btype='band')
    X_out = np.zeros_like(X_in)
    for ci in range(X_in.shape[1]):
        X_out[:, ci] = filtfilt(b, a, X_in[:, ci])
    X_out = X_out - X_out.mean(axis=0, keepdims=True)
    return X_out

# ==========================================================
# 6) FBCCA 计算
# ==========================================================
scores = np.zeros(len(freqs), dtype=float)

for fi, f in enumerate(freqs):
    Yf = Y_refs[f]
    score_f = 0.0
    for k, (fl, fh) in enumerate(subbands):
        Xk = bandpass_filter_multichannel(X0, fl, fh, fs)
        rho_k = cca_multi_channel_svd(Xk, Yf)
        score_f += weights[k] * (rho_k ** 2)
    scores[fi] = score_f
    print(f"{f:.2f} Hz -> score = {score_f:.6f}")

pred_f = freqs[int(np.argmax(scores))]
print("\nFBCCA 预测目标频率：", pred_f, "Hz")

# ==========================================================
# 7) 可视化
# ==========================================================
plt.figure(figsize=(8, 5), dpi=200)
y_pos = np.arange(len(freqs))
plt.barh(y_pos, scores, edgecolor='black')
plt.barh(np.argmax(scores), scores.max(), edgecolor='black')
plt.yticks(y_pos, [f"{f:.2f} Hz" for f in freqs])
plt.xlabel("FBCCA score")
plt.title("FBCCA 结果（使用702点参考矩阵）")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
