# 运动想象(MI)EEG分类器 - 代码开发大纲

## 📊 数据集信息
- **数据集**: BCI Competition IV Dataset 2a
- **任务**: 4类运动想象分类（左手、右手、脚、舌头）
- **通道**: 22 EEG + 3 EOG
- **采样率**: 250Hz
- **受试者**: 9人
- **文件格式**: GDF

---

## 🗂️ 项目结构

```
brain/
├── MI数据集/
│   └── BCICIV_2a_gdf.zip    # 已下载 ✓
├── data/
│   └── (预处理后的数据)
├── models/
│   └── (训练好的模型)
├── src/
│   ├── data_loader.py        # 数据加载
│   ├── preprocessing.py      # 预处理
│   ├── features.py          # 特征提取
│   ├── models.py            # 分类模型
│   ├── train.py             # 训练脚本
│   └── evaluate.py          # 评估脚本
└── results/
    └── (实验结果)
```

---

## 📝 代码大纲

### 1. 数据加载 (data_loader.py)
```
- load_gdf_file(): 读取GDF格式数据
- load_dataset(): 加载全部受试者数据
- get_subject_data(): 按受试者分割数据
- Dataset类: PyTorch Dataset封装
```

### 2. 预处理 (preprocessing.py)
```
- bandpass_filter(): 带通滤波 (0.5-100Hz)
- notch_filter(): 陷波滤波 (50Hz去除工频)
- apply_car(): 共平均参考(CAR)
- downsample(): 下采样(可选)
- normalize(): 标准化(Z-score)
```

### 3. 特征提取 (features.py)
```
传统方法:
- CSP (Common Spatial Patterns): 空间模式特征
- FFT/PowerSpectrum: 频域特征
- Welch PSD: 功率谱密度

深度学习 (端到端):
- 自动学习特征
```

### 4. 模型 (models.py)
```
传统ML:
- LDA (线性判别分析)
- SVM (支持向量机)
- FBCSP (滤波器组CSP)

深度学习:
- EEGNet: 经典CNN架构
- ShallowConvNet: 浅层卷积网络
- DeepConvNet: 深层卷积网络
- CNN-Transformer: CNN+注意力机制
- RNN/LSTM: 时序建模
```

### 5. 训练 (train.py)
```
- 训练集/验证集/测试集划分
- 交叉验证 (k-fold CV)
- 早停机制
- 学习率调度
- 模型保存
```

### 6. 评估 (evaluate.py)
```
- 准确率 (Accuracy)
- Kappa系数
- 混淆矩阵
- 分类报告
- 可视化 (t-SNE, 通道重要性)
```

---

## 🔬 推荐实验流程

### Phase 1: 基线模型
1. CSP + LDA (传统SOTA)
2. EEGNet (深度学习入门)

### Phase 2: 模型改进
1. FBCSP + SVM
2. CNN-Transformer (如ATCNet)

### Phase 3: 迁移学习
1. 预训练模型微调
2. 域适应

---

## 📚 参考文献

1. Blankertz et al. (2008) - BCI Competition IV
2. Lawhern et al. (2018) - EEGNet
3. Zhang et al. (2020) - FBCSP
4. Altaheri et al. (2023) - ATCNet

---

## 🛠️ 依赖库

```txt
numpy
scipy
mne
scikit-learn
torch
torchvision
tensorboard
matplotlib
seaborn
```

---

*生成时间: 2026-03-17*
