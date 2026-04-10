# 02_SSVEP 使用导航

当前 SSVEP 主线目录是：

- `2026-04_async_fbcca_idle_decoder/`

这条主线已经固定为“三程序架构”：

1. 实时识别（读取 profile 在线解码）
2. 数据采集（保存 NPZ + JSON manifest）
3. 训练评测（读取数据集训练并输出报告/profile）

详细说明请直接看：

- `2026-04_async_fbcca_idle_decoder/README.md`

---

## 快速入口（Py 文件）

在本目录运行：

```powershell
# 1) 实时识别
python .\START_FBCCA_REALTIME_UI.py --serial-port COM4 --model fbcca

# 2) 数据采集
python .\START_SSVEP_DATA_COLLECTION_UI.py --serial-port COM4 --subject-id subject001 --session-index 1

# 3) 训练评测
python .\START_MODEL_EVALUATION_UI.py `
  --dataset-manifest .\2026-04_async_fbcca_idle_decoder\profiles\datasets\<session1>\session_manifest.json
```

---

## 推荐闭环

1. 先采集 `session1`（建议再采 `session2` 用于跨会话验证）。
2. 再训练评测，拿到 `offline_train_eval_*.json/.md` 和 profile。
3. 最后实时识别验证在线效果。

---

## 旧目录说明

`2026-02_*`、`2026-03_*` 目录为历史版本或实验分支。默认不作为当前在线主流程入口。
