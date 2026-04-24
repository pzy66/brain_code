# entrypoints/README

本目录是“薄入口”集合。它们的设计目标非常明确：

- 不在这里写业务逻辑
- 只负责把 Python 路径准备好
- 然后转发到 `apps/` 中真正的 UI 程序
- 或者为某个任务附加少量默认参数

如果你日常只是想启动某条固定工作流，优先用这里，而不是直接记住一堆 UI 内部参数。

---

## 1. 文件总览

| 文件 | 作用 |
|---|---|
| `start_collection.py` | 打开数据采集 UI |
| `start_realtime.py` | 打开实时在线 UI |
| `start_training_eval.py` | 打开训练评测 UI |
| `start_tdca_local_opt.py` | 用 TDCA local-opt 预设打开训练评测 UI |
| `start_fbcca_local_opt.py` | 用 FBCCA local-opt 预设打开训练评测 UI |
| `start_fbcca_external_replay.py` | 用 external replay 预设打开训练评测 UI |

---

## 2. 每个入口的真实行为

### `start_collection.py`

- 转发到 `apps.data_collection_ui.main`
- 不额外注入任务参数

### `start_realtime.py`

- 转发到 `apps.realtime_online_ui.main`
- 不额外注入任务参数

### `start_training_eval.py`

- 转发到 `apps.training_evaluation_ui.main`
- 不额外注入任务参数

### `start_tdca_local_opt.py`

会自动附带：

- `--task tdca-local-opt`
- `--remote-mode 0`
- `--enable-local-fallback 1`

含义是：

- 直接进入 TDCA 本地优化模式
- 默认本地跑
- 允许必要时走本地 fallback

### `start_fbcca_local_opt.py`

会自动附带：

- `--task fbcca-local-opt`
- `--remote-mode 0`
- `--enable-local-fallback 1`

### `start_fbcca_external_replay.py`

会自动附带：

- `--task fbcca-external-replay-opt`
- `--remote-mode 0`
- `--enable-local-fallback 1`

---

## 3. 推荐用法

### 直接启动某条主线

```bash
python D:\brain\brain_code\02_SSVEP\entrypoints\start_tdca_local_opt.py
python D:\brain\brain_code\02_SSVEP\entrypoints\start_fbcca_local_opt.py
python D:\brain\brain_code\02_SSVEP\entrypoints\start_fbcca_external_replay.py
```

### 如果你还不确定用哪条

```bash
python D:\brain\brain_code\02_SSVEP\START_SSVEP.py
```

---

## 4. 为什么要保留这些薄入口

原因有三条：

1. 避免每次都手动切 task。
2. 避免 UI 越来越多后，日常入口混乱。
3. 允许把“主界面”与“固定预设”分离。

这比复制一份新 UI 更可维护。

---

## 5. 维护规则

1. 薄入口只做参数预设，不做算法逻辑。
2. 如果一个新任务需要单独入口，优先做成这里的一层包装。
3. 不要把复杂参数分支堆到 entrypoint；复杂逻辑应回到 `apps/` 或 `ssvep_core/`。
