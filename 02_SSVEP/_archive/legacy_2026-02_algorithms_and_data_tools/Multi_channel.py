import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

# ==========================================================
# 1. 读取 EEG 数据
# ==========================================================
mat = sio.loadmat(r'C:\Users\P1233\Desktop\脑电\数据集3\12264401 (1)\S1.mat')

data_struct = mat['data'][0, 0]   # 取结构体
eeg_data = data_struct['EEG']     # 形状：64 × 750 × 4 × 40
print("EEG shape:", eeg_data.shape)

fs = 250
block = 1
condition = 10

# ==========================================================
# 2. 选择多通道并取数据 (T × C)
# ==========================================================

# channels = list(range(64))       # 64通道
channels = [ 53,55,56,57,59,61, 62, 63]        # 脑区名称在数据集里，选取8个最相关的 Oz O1 O2 POz PO3 PO4 PO7 PO8
                                               # 八个脑区在这个数据集中相关的编号是：62  61 63 56  55  57  53  59

# eeg_data[channel, time, block, condition]
X_raw = eeg_data[channels, :, block, condition]   # (C, T)
X_raw = X_raw.T                                   # (T, C)

T = X_raw.shape[0]
time_points = np.arange(T) / fs

# ==========================================================
# 3. 滤波处理（逐通道）
# ==========================================================

def bandpass_notch_detrend(x_raw):
    # 去基线
    b, a = butter(1, 3/(fs/2), btype='low')
    base = filtfilt(b, a, x_raw)
    x1 = x_raw - base






、
    # 陷波 50Hz
    b, a = iirnotch(50, 30, fs)
    x2 = filtfilt(b, a, x1)

    # 4–30Hz 带通
    b, a = butter(2, [3/(fs/2), 30/(fs/2)], btype='band')
    x3 = filtfilt(b, a, x2)

    return x3

# 对所有通道滤波
X = np.zeros_like(X_raw)
for ci in range(len(channels)):
    X[:, ci] = bandpass_notch_detrend(X_raw[:, ci])

# 去均值
X = X - X.mean(axis=0, keepdims=True)

# ==========================================================
# 4. 构造 CCA 参考矩阵
# ==========================================================

def build_ref_matrix(fs, L, f, Nh=3):
    """
    构造参考信号矩阵 Y_f (T × 2Nh)
    """
    T = int(fs * L)
    t = np.arange(1, T + 1) / fs
    cols = []
    for h in range(1, Nh + 1):
        cols.append(np.sin(2 * np.pi * h * f * t))
        cols.append(np.cos(2 * np.pi * h * f * t))
    Y = np.stack(cols, axis=1)
    Y -= Y.mean(axis=0, keepdims=True)
    return Y

L = 3                      # 3 秒窗口
freqs = [round(8 + 0.2*i, 1) for i in range(int((15.8 - 8) / 0.2) + 1)]

Nh = 3

Y_refs = {f: build_ref_matrix(fs, L, f, Nh) for f in freqs}

# ==========================================================
# 5. 多通道 CCA（广义特征值求解）
# ==========================================================

def cca_multi_channel(X, Y):
    """
    多通道EEG: X (T,C)
    参考信号: Y (T,2Nh)
    返回最大相关系数 rho
    """
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
# 6. 计算每个频率的 ρ
# ==========================================================

rhos = []
for f, Yf in Y_refs.items():
    rho = cca_multi_channel(X, Yf)
    rhos.append(rho)
    print(f"{f} Hz → rho = {rho:.4f}")

rhos = np.array(rhos)

# ==========================================================
# 7. 预测频率
# ==========================================================
pred_f = freqs[int(np.argmax(rhos))]
print("\n预测目标频率：", pred_f, "Hz")

# ==========================================================
# 8. 可视化 Bar 图
# ==========================================================

fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

y_pos = np.arange(len(freqs))
ax.barh(y_pos, rhos, color='skyblue', edgecolor='black')
ax.barh(np.argmax(rhos), rhos.max(), color='crimson')

for i, v in enumerate(rhos):
    ax.text(v + 0.002, i, f"{v:.3f}", va='center')

ax.set_yticks(y_pos)
ax.set_yticklabels([f"{f} Hz" for f in freqs])
ax.set_xlabel("ρ")
ax.set_title("多通道 CCA 结果")
ax.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
