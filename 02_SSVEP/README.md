# 02_SSVEP 目录说明与快速启动

这个目录包含 SSVEP 相关代码与启动入口。当前推荐主线是：

- `2026-04_async_fbcca_idle_decoder/`

该主线支持：

- 异步识别（目标/Idle）
- 多模型评测与排序
- 单模型预训练 + 在线分类

## 1. 目录结构（推荐关注）

- `2026-04_async_fbcca_idle_decoder/`
  - 当前主线（建议优先使用）
- `2026-03_realtime_ui_and_online_decoder/`
  - 较早期 UI + 在线识别实现
- `2026-02_realtime_stimulus_and_classifier_core/`
  - 更早的实时刺激与分类核心
- `2026-02_algorithms_and_data_tools/`
  - 算法与数据处理脚本
- `2026-02_stimulus_variants/`
  - 刺激样式变体
- `2026-02_custom_dataset_scripts/`
  - 自采数据相关脚本

## 2. 一键入口（Python，推荐）

只保留两个入口：

- `START_MODEL_EVALUATION_UI.py`（多模型评测）
- `START_FBCCA_REALTIME_UI.py`（单模型预训练 + 在线分类，默认 FBCCA）

直接运行：

```powershell
python .\START_MODEL_EVALUATION_UI.py --serial-port COM4
python .\START_FBCCA_REALTIME_UI.py --serial-port COM4
```

## 3. PyCharm 直接运行

直接在 PyCharm 里运行以下文件即可：

- `START_MODEL_EVALUATION_UI.py`
- `START_FBCCA_REALTIME_UI.py`

默认参数可直接跑（`serial-port=auto`，`board-id=0`，`freqs=8,10,12,15`）。

如果要固定 COM4，可在 PyCharm 的 Run Configuration 里加参数：

```text
--serial-port COM4
```

## 4. 建议使用顺序

1. 先跑评测，选模型与阈值：
   - `python .\START_MODEL_EVALUATION_UI.py --serial-port COM4`
2. 再做单模型预训练 + 在线验证（默认 FBCCA）：
   - `python .\START_FBCCA_REALTIME_UI.py --serial-port COM4`
3. 需要脚本化时，直接使用：
   - `2026-04_async_fbcca_idle_decoder\async_fbcca_idle_standalone.py`

## 5. 详细文档

完整流程、输出字段、验收与排障，请看：

- `2026-04_async_fbcca_idle_decoder/README.md`
