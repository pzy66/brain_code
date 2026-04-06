import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

# ===============================
# 1) 参数
# ===============================
csv_path = r"C:\Users\P1233\Desktop\脑电\大创\LTM-8.csv"   # 改成你的CSV路径
channel_name = "EEG_Channel_1"                           # 改成你要的通道
fs = 250

# 预处理参数（按你给的代码）
cutoff = 3.0       # 低通提取基线趋势(Hz)
f0 = 50.0          # 陷波中心频率
Q = 30.0           # 陷波品质因数
lowcut = 3.0       # 带通下限
highcut = 90.0     # 带通上限

# ===============================
# 2) 读取CSV：跳过第2行全0；并从第3行开始取到结束
# ===============================
df = pd.read_csv(csv_path)

# 第1行是表头；第2行通常为全0；所以从索引1开始（对应第3行）
signal = df.loc[1:, channel_name].astype(float).values

# 时间轴
t = np.arange(len(signal)) / fs

# ===============================
# 3) 原始波形
# ===============================
plt.figure(figsize=(16, 4), dpi=150)
plt.plot(t, signal, linewidth=0.8)
plt.title("Source (CSV)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================
# 4) 基线漂移去除：3Hz低通得到趋势，再相减
# ===============================
b, a = butter(1, cutoff/(fs/2), btype='low')
baseline = filtfilt(b, a, signal)
sig_after_detrend = signal - baseline

plt.figure(figsize=(16, 4), dpi=150)
plt.plot(t, sig_after_detrend, linewidth=0.8)
plt.title("After Detrend (LP 3Hz baseline removed)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================
# 5) 50Hz 陷波
# ===============================
b, a = iirnotch(f0, Q, fs)
sig_notch = filtfilt(b, a, sig_after_detrend)

plt.figure(figsize=(16, 4), dpi=150)
plt.plot(t, sig_notch, linewidth=0.8)
plt.title("After Notch (50Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================
# 6) 3–90Hz 带通
# ===============================
b, a = butter(2, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
sig_bandpass = filtfilt(b, a, sig_notch)

plt.figure(figsize=(16, 4), dpi=150)
plt.plot(t, sig_bandpass, linewidth=0.8)
plt.title("After Bandpass (3–90Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================
# 7) FFT 频谱
# ===============================
n = len(sig_bandpass)
freqs = np.fft.fftfreq(n, d=1/fs)
fft_values = np.fft.fft(sig_bandpass)
amplitude = np.abs(fft_values) / n * 2   # 双边转单边幅值近似

# 只取正频率
pos_mask = freqs > 0
freqs_pos = freqs[pos_mask]
amp_pos = amplitude[pos_mask]

plt.figure(figsize=(16, 4), dpi=150)
plt.plot(freqs_pos, amp_pos, linewidth=0.8)
plt.title("FFT Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(0, 100)
plt.tight_layout()
plt.show()
