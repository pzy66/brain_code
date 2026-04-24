import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch ,find_peaks

# ========== 1. 读取数据 ==========
mat = sio.loadmat(r'C:\Users\P1233\Desktop\脑电\数据集2\EEGSSVEPPart1\EEG-SSVEP-Part1\S001b.mat')  # 改成你的文件名
data = mat['eeg']
print("数据维度：", data.shape)  # 应该是 (257, 117917)

# ========== 2. 参数设置 ==========

fs = 250        # 采样率（Hz）
channel = 126   # Oz通道
start_time = 205  # 开始时间（秒）
duration = 15 # 截取时长（秒）

# 计算采样点范围
start_index = int(start_time * fs)
end_index = int((start_time + duration) * fs)

# ========== 3. 提取信号 ==========
signal = data[channel - 1, start_index:end_index]  # Python索引从0开始
time_points = np.arange(0, duration, 1/fs)

plt.figure(figsize=(20, 5), dpi=300)
plt.plot(time_points, signal, color='steelblue')
plt.title(f'Oz 200s-210s')
plt.xlabel('time (s)')
plt.ylabel('voltage(μV)')
plt.grid(True)
plt.show()

##==============没处理过的频谱画图======================
n = len(signal)
freqs = np.fft.fftfreq(n, d=1/fs)
fft_values = np.fft.fft(signal)
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
plt.axvline(x=6.5, color='green', linestyle='--', linewidth=1.5, label='6.5 Hz')
plt.axvline(x=13, color='dodgerblue', linestyle='--', linewidth=1.5, label='13 Hz')
plt.axvline(x=26, color='green', linestyle='--', linewidth=1.5, label='26 Hz')
plt.title('FFT  (Oz, 200–230s)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 80)
plt.grid(True)
plt.show()

