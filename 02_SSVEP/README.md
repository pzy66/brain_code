# 02 SSVEP

这里集中存放 SSVEP 相关代码。

## 子目录说明

- `2026-03_realtime_ui_and_online_decoder/`
  当前主线，偏实时刺激界面、BrainFlow 采集和在线识别
- `2026-02_realtime_stimulus_and_classifier_core/`
  较早期的实时主流程，包含 `main.py`、`stimulus.py`、`classifier.py`
- `2026-02_algorithms_and_data_tools/`
  早期算法与数据处理脚本，包含 FBCCA、多通道、数据读取和评估脚本
- `2026-02_stimulus_variants/`
  不同刺激脚本和刺激布局变体
- `2026-02_custom_dataset_scripts/`
  自采数据和相关脚本

## 推荐阅读顺序

1. 先看 `2026-03_realtime_ui_and_online_decoder/`
2. 再看 `2026-02_realtime_stimulus_and_classifier_core/`
3. 需要补算法细节时再看 `2026-02_algorithms_and_data_tools/`
