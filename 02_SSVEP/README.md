# 02_SSVEP

本目录存放 SSVEP 相关代码与运行入口。

## 目录说明

- `2026-03_realtime_ui_and_online_decoder/`
  - 较早期实时 UI + 在线识别主线
- `2026-04_async_fbcca_idle_decoder/`
  - 当前异步识别主线（支持多模型 benchmark + 在线门控）
- `2026-02_realtime_stimulus_and_classifier_core/`
  - 更早的实时刺激/分类核心脚本
- `2026-02_algorithms_and_data_tools/`
  - 算法与数据工具脚本（离线分析）
- `2026-02_stimulus_variants/`
  - 各类刺激脚本变体
- `2026-02_custom_dataset_scripts/`
  - 自采数据集相关脚本

## 快速入口

- `START_ASYNC_SSVEP_COM4.cmd`
  - 启动 async SSVEP 校验 UI（COM4，board_id=0）
- `START_ASYNC_SSVEP_BENCHMARK_COM4.cmd`
  - 启动多模型 benchmark（COM4，board_id=0）
- `START_ASYNC_SSVEP_ONLINE_COM4.cmd`
  - 用默认 profile 启动在线识别（COM4，board_id=0）

## 推荐顺序

1. 先看：
   - `2026-04_async_fbcca_idle_decoder/README.md`
2. 再运行：
   - `START_ASYNC_SSVEP_BENCHMARK_COM4.cmd`
3. 然后运行：
   - `START_ASYNC_SSVEP_ONLINE_COM4.cmd`
