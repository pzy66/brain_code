import socket
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

# ==========================================================
#网络参数
# ==========================================================
JETSON_IP = "192.168.149.1"
PORT = 8888


freq2cmd = {8: b'F', 10: b'B', 12: b'L', 15: b'R'}

def send_cmd(cmd_byte: bytes):
    """发送单字节指令给 Jetson"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3)
    sock.connect((JETSON_IP, PORT))
    sock.sendall(cmd_byte)   # 单字节
    sock.close()

# ==========================================================

csv_path = r"C:\Users\P1233\Desktop\脑电\大创\LTM-10_4.csv"
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
# 2) 读取 CSV -> (T, C)，并自动用实际长度 T
# ==========================================================
df = pd.read_csv(csv_path)
X_raw = df.loc[1:, channels].astype(float).values  # (T, 8)

T = X_raw.shape[0]
print(f"实际可用点数 T={T}, 时长 L_sec={T/fs:.3f} s")

# ==========================================================
# 3) 公共预处理：3Hz 去基线 + 50Hz 陷波
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
# 4) 构造参考矩阵（用实际 T 点数）
# ==========================================================
def build_ref_matrix(fs, T, f, Nh=3):
    t = np.arange(1, T + 1) / fs
    cols = []
    for h in range(1, Nh + 1):
        cols.append(np.sin(2 * np.pi * h * f * t))
        cols.append(np.cos(2 * np.pi * h * f * t))
    Y = np.stack(cols, axis=1)
    Y = Y - Y.mean(axis=0, keepdims=True)
    return Y

Y_refs = {f: build_ref_matrix(fs, T, f, Nh) for f in freqs}

# ==========================================================
# 5) 多通道 CCA（SVD版）
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
# 6) 子带带通
# ==========================================================
def bandpass_filter_multichannel(X_in, fl, fh, fs):
    b, a = butter(2, [fl/(fs/2), fh/(fs/2)], btype='band')
    X_out = np.zeros_like(X_in)
    for ci in range(X_in.shape[1]):
        X_out[:, ci] = filtfilt(b, a, X_in[:, ci])
    X_out = X_out - X_out.mean(axis=0, keepdims=True)
    return X_out

# ==========================================================
# 7) FBCCA 计算 -> pred_f
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
# 8) 频率 -> 指令 -> 发送
# ==========================================================
cmd = freq2cmd[int(pred_f)]
print("映射指令：", cmd.decode("ascii"))

try:
    send_cmd(cmd)
    print(f"✅ 已发送到 Jetson {JETSON_IP}:{PORT}")
except Exception as e:
    print("❌ 发送失败：", e)

# ==========================================================
# 9) 可视化（可选）
# ==========================================================
plt.figure(figsize=(8, 5), dpi=200)
y_pos = np.arange(len(freqs))
plt.barh(y_pos, scores, edgecolor='black')
plt.barh(np.argmax(scores), scores.max(), edgecolor='black')
plt.yticks(y_pos, [f"{f:.2f} Hz" for f in freqs])
plt.xlabel("FBCCA score")
plt.title("FBCCA 结果")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
