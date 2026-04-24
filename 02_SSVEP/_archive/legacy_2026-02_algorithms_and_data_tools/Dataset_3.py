import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch ,find_peaks

# 读取 .mat 文件
mat = sio.loadmat(r'C:\Users\P1233\Desktop\脑电\数据集3\S1.mat')

# 取出 data 结构体（它是个 numpy.ndarray）
data_struct = mat['data'][0, 0]

# 访问 EEG（4维数组）
eeg_data = data_struct['EEG']            # 这里的 EEG 就是 64×750×4×40 的 double 数组
print("EEG数据维度：", eeg_data.shape)   # 应该输出 (64, 750, 4, 40)

fs = 250  # 采样率（你也可以从 suppl_info 中读取）
channel = 61
block =1
condition = 7
signal = eeg_data[channel, :, block, condition]

time_points = np.arange(signal.shape[0]) / fs
plt.figure(figsize=(20, 5),dpi=300)
plt.plot(time_points, signal)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.title(f"Channel {channel+1}, Block {block+1}, Condition {condition+1}")
plt.grid(True)
plt.show()

# ##==============没处理过的频谱画图======================
# n = len(signal)
# freqs = np.fft.fftfreq(n, d=1/fs)
# fft_values = np.fft.fft(signal)
# amplitude = np.abs(fft_values) / n * 2
#
# # 只取正频率部分
# pos_mask = freqs > 0
# freqs = freqs[pos_mask]
# amplitude = amplitude[pos_mask]
#
# plt.figure(figsize=(20,6), dpi=300)
# plt.plot(freqs, amplitude, color='crimson')
# plt.title('FFT  (Oz, 0-3s)')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.xlim(0, 80)
# plt.grid(True)
# plt.show()
#

# ========== 4. 滤波参数 ==========

cutoff = 3.0   # 低通频率，用于提取基线趋势
f0 = 50.0      # 陷波中心频率
Q = 30.0       # 品质因数
lowcut = 4.0   # 带通下限
highcut = 50.0 # 带通上限
# ========== 5. 基线漂移去除 ==========
b, a = butter(1, cutoff/(fs/2), btype='low')
data_3Hz = filtfilt(b, a, signal)
sig_after_detrend = signal - data_3Hz

plt.figure(figsize=(20,5), dpi=300)
plt.plot(time_points, sig_after_detrend, color='orange')
plt.title('1')
plt.xlabel('time (s)')
plt.ylabel('voltage(μV)')
plt.grid(True)
plt.show()

# ========== 6. 50Hz 陷波滤波 ==========
b, a = iirnotch(f0, Q, fs)
sig_notch = filtfilt(b, a, sig_after_detrend)

plt.figure(figsize=(20,5), dpi=300)
plt.plot(time_points, sig_notch, color='green')
plt.title('2')
plt.xlabel('time (s)')
plt.ylabel('voltage(μV)')
plt.grid(True)
plt.show()

# ========== 7. 4–90Hz 带通滤波 ==========
b, a = butter(2, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
sig_bandpass = filtfilt(b, a, sig_notch)

plt.figure(figsize=(20,5), dpi=300)
plt.plot(time_points, sig_bandpass, color='purple')
plt.title('3')
plt.xlabel('time (s)')
plt.ylabel('voltage(μV)')
plt.grid(True)
plt.show()

# ========== 8. FFT 频谱分析 ==========
n = len(sig_bandpass)
freqs = np.fft.fftfreq(n, d=1/fs)
fft_values = np.fft.fft(sig_bandpass)
amplitude = np.abs(fft_values) / n * 2

# 只取正频率部分
pos_mask = freqs > 0
freqs = freqs[pos_mask]
amplitude = amplitude[pos_mask]

plt.figure(figsize=(20,6), dpi=300)
plt.plot(freqs, amplitude, color='crimson')
plt.title('FFT  (Oz, 0-3s)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 80)
plt.grid(True)
plt.show()

#=====================================以下为分类模块======================================================

#构造参考矩阵
def build_ref_matrix(fs, L, f, Nh=3):
    """
    构造参考信号矩阵 Y_f
    fs: 采样率 (Hz)
    L : 窗口长度 (秒)
    f : 频率 (Hz)
    Nh: 谐波数 (默认取前三次)
    返回矩阵形状 [T, 2*Nh]
    """
    T = int(fs * L)                   # 总采样点数 = 250*3 = 750
    t = np.arange(1, T + 1) / fs      # 从 1/fs 到 3 秒
    cols = []
    for h in range(1, Nh + 1):
        cols.append(np.sin(2 * np.pi * h * f * t))
        cols.append(np.cos(2 * np.pi * h * f * t))
    Y = np.stack(cols, axis=1)
    # 零均值化（可选，提升数值稳定性）
    Y -= Y.mean(axis=0, keepdims=True)
    return Y

# ===== 参数 =====
L = 3
freqs = [9,9.6,9.8, 10, 10.2,10.4,10.6 ]
Nh = 3

# ===== 构造每个频率的参考矩阵 =====
Y = {f: build_ref_matrix(fs, L, f, Nh) for f in freqs}

# # 检查形状
# for f, Y in Y_refs.items():
#     print(f"{f} Hz → Y.shape = {Y.shape}")
#

X = sig_bandpass[:, np.newaxis]   ##把脑电数据转化为矩阵

X = X - X.mean(axis=0, keepdims=True) ##去均值

# print(sig_bandpass.shape)
# print(X.shape)


# ====== 1. 定义 CCA 的回归公式函数 ======
def cca_rho_regression(X, Y):
    """
    单通道CCA回归/投影公式
    X: (T,1) EEG列向量
    Y: (T,2*Nh) 参考矩阵
    """
    # 零均值化
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # 协方差与正则
    G = Y.T @ Y + 1e-8 * np.eye(Y.shape[1])

    # β = (Y^T Y)^(-1) Y^T X
    beta = np.linalg.solve(G, Y.T @ X)

    # 投影回时域
    x_hat = Y @ beta

    # 计算相关系数
    rho = np.linalg.norm(x_hat) / np.linalg.norm(X)
    return float(rho)


# ====== 2. 对每个参考频率计算 rho ======
rhos = []  # 存储每个频率的ρ
freq_list = []  # 记录频率顺序

for f, Yf in Y.items():  # 遍历字典
    rho = cca_rho_regression(X, Yf)
    rhos.append(rho)
    freq_list.append(f)
    print(f"{f} Hz → ρ = {rho:.4f}")

# ====== 3. 判决：取最大ρ对应的频率 ======
rhos = np.array(rhos)
pred_f = freq_list[int(np.argmax(rhos))]

print(f"\n预测频率：{pred_f} Hz")



# ===== 示例 =====
# freq_list = [9, 9.6, 9.8, 10, 10.2, 10.4, 10.6]
# rhos = np.array(rhos)
# pred_f = freq_list[int(np.argmax(rhos))]

fig, ax = plt.subplots(figsize=(8,5), dpi=200)

# 绘制水平条
y_pos = np.arange(len(freq_list))
ax.barh(y_pos, rhos, color='lightsteelblue', edgecolor='black')

# 高亮最大ρ的频率
max_idx = np.argmax(rhos)
ax.barh(y_pos[max_idx], rhos[max_idx], color='crimson', label=f"frequency prediction {freq_list[max_idx]} Hz")

# 加上数值标注
for i, v in enumerate(rhos):
    ax.text(v + 0.002, y_pos[i], f"{v:.3f}", va='center', fontsize=9)

# 坐标与标题
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{f} Hz" for f in freq_list])
ax.set_xlabel(" ρ")
ax.set_title("CCA")
ax.legend()
ax.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


