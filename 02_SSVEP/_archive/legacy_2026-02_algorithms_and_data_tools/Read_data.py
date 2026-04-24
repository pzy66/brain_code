import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

# ========== 1. 读取数据 ==========
mat = sio.loadmat(r'C:\Users\P1233\Desktop\脑电\数据集\Stest.mat')
data = mat['data']  # 维度 [8,710,2,10,12]
print("数据维度：", data.shape)

sig_all = np.zeros((710, 10))  # N×10矩阵  存储数据的矩阵

# ========== 3. 滤波参数 ==========
fs = 250  # 采样率
cutoff = 3  # 低通频率，用于提取基线趋势
f0 = 50.0  # 陷波中心频率
Q = 30.0  # 品质因数（20~50）
lowcut = 3.0  # 带通下限
highcut = 90.0  # 带通上限

# ========== 2. 提取指定条件 ==========
channel =  5  # Oz通道
electrode = 0 # 0=湿电极, 1=干电极
block = 2
target = 2
signal = data[channel, :, electrode, block, target]
time_points = np.arange(710) / 250  # 采样率 250Hz
# ========== 刺激频率对应关系（Python中 target 从 0 开始） ==========
# target = 0 → 9.25 Hz
# target = 1 → 9.75 Hz
# target = 2 → 10.25 Hz
# target = 3 → 10.75 Hz
# target = 4 → 11.25 Hz
# target = 5 → 11.75 Hz
# target = 6 → 12.25 Hz
# target = 7 → 12.75 Hz
# target = 8 → 13.25 Hz
# target = 9 → 13.75 Hz
# target = 10 → 14.25 Hz
# target = 11 → 14.75 Hz
plt.figure(figsize=(30,6), dpi=300)
plt.plot(time_points, signal)
plt.title('Source')
plt.xlabel('time(s)')
plt.ylabel('voltage(uV)')
plt.grid(True)
plt.show()

# ========== 4. 基线漂移去除 ==========
b, a = butter(1, cutoff/(fs/2), btype='low')
data_3Hz = filtfilt(b, a, signal)
sig_after_detrend = signal - data_3Hz

plt.figure(figsize=(20,6), dpi=300)
plt.plot(time_points, sig_after_detrend)
plt.title('1')
plt.show()

# ========== 5. 50Hz 陷波滤波 ==========
b, a = iirnotch(f0, Q, fs)
sig_notch = filtfilt(b, a, sig_after_detrend)

plt.figure(figsize=(20,6), dpi=300)
plt.plot(time_points, sig_notch)
plt.title('2')
plt.show()

# ========== 6. 4–90Hz 带通滤波 ==========
b, a = butter(2, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
sig_bandpass = filtfilt(b, a, sig_notch)

# plt.figure(figsize=(20,6), dpi=300)
# plt.plot(time_points, sig_bandpass)
# plt.title('3')
# plt.xlabel('time(s)')
# plt.ylabel('voltage(uV)')
# plt.grid(True)
# plt.show()


# ========== 7. FFT 频谱分析 ==========
n = len(sig_bandpass)            # 数据长度
freqs = np.fft.fftfreq(n, d=1/fs)  # 频率轴
fft_values = np.fft.fft(sig_bandpass)  # 进行FFT
amplitude = np.abs(fft_values) / n * 2  # 幅值归一化（双边谱转单边谱）

# 只取正频率部分
pos_mask = freqs > 0
freqs = freqs[pos_mask]
amplitude = amplitude[pos_mask]

plt.figure(figsize=(20,6), dpi=300)
plt.plot(freqs, amplitude)
plt.title('FFT')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.xlim(0, 100)  # 只看0–100Hz范围
plt.show()