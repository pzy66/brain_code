import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

# ========== 1. 读取数据 ==========
mat = sio.loadmat(r'C:\Users\P1233\Desktop\脑电\数据集\S100.mat')
data = mat['data']  # 维度 [8,710,2,10,12]
print("数据维度：", data.shape)

# ========== 2. 提取指定条件 ==========
channel =  5  # Oz通道
electrode = 0 # 0=湿电极, 1=干电极
block = 0
target = 2
signal = data[channel, :, electrode, block, target]
# time_points = np.arange(710) / 250  # 采样率 250Hz
time_points = np.arange(710)
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