# 02_SSVEP 使用入口（当前主线）

当前 SSVEP 主线目录：
- `2026-04_async_fbcca_idle_decoder/`

主线分成 3 个程序：
- 实时识别：`ssvep_realtime_online_ui.py`
- 数据采集：`ssvep_dataset_collection_ui.py`
- 训练评测：`ssvep_training_evaluation_ui.py`

推荐从本目录直接运行启动脚本（适合 PyCharm）：

```powershell
# 1) 实时识别（读 profile 在线解码）
python .\START_FBCCA_REALTIME_UI.py --serial-port COM4 --model fbcca

# 2) 数据采集（多轮采集入口）
python .\START_SSVEP_DATA_COLLECTION_UI.py --serial-port COM4 --subject-id subject001 --session-index 1

# 3) 训练评测（读取采集数据集）
python .\START_MODEL_EVALUATION_UI.py `
  --dataset-manifest .\2026-04_async_fbcca_idle_decoder\profiles\datasets\<session1>\session_manifest.json `
  --dataset-manifest-session2 .\2026-04_async_fbcca_idle_decoder\profiles\datasets\<session2>\session_manifest.json
```

推荐闭环：
1. 先采集 `session1`，建议再采 `session2` 做跨会话外测。
2. 再训练评测并生成报告与 profile。
3. 最后在实时识别程序中加载 profile 验证在线效果。

详细说明见：
- `2026-04_async_fbcca_idle_decoder/README.md`
