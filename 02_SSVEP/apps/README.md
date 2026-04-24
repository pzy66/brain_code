# apps/README

本目录存放 PyQt 图形界面程序。它们不是算法实现本体，而是对 `ssvep_core/` 的可视化封装、调度封装或观察封装。

如果你想做日常操作，通常先从这里进入；如果你想做批跑或服务器任务，则转到 `tools/`。

---

## 1. 程序总览

| 文件 | 角色 | 典型使用时机 |
|---|---|---|
| `launcher_ui.py` | 总入口 | 想从一个窗口进入所有主功能 |
| `data_collection_ui.py` | 数据采集 UI | 采新的本地 SSVEP session |
| `realtime_online_ui.py` | 实时在线 UI | 加载 deployed profile 做在线识别 |
| `training_evaluation_ui.py` | 训练评测 UI | 发起 benchmark、compare、local-opt、external replay |
| `external_replay_viewer.py` | 外部 replay 时间轴查看器 | 看 held-out session 回放明细 |
| `async_fbcca_validation_ui.py` | 共享刺激/验证组件 | 给采集或实时界面提供刺激显示部件 |

---

## 2. `launcher_ui.py`

### 作用

统一启动器。它本身不做训练和解码，只负责把你带到正确的功能页。

### 当前入口按钮

- 数据采集
- 实时在线解码
- 训练评测
- TDCA 本地异步优化

### 优点

- 不需要记参数
- 适合人工操作
- 适合作为日常默认入口

### 不足

- 不适合批跑
- 不适合写脚本
- 不适合服务器无界面环境

### 推荐用法

```bash
python D:\brain\brain_code\02_SSVEP\START_SSVEP.py
```

---

## 3. `data_collection_ui.py`

### 作用

负责本地 4 目标 SSVEP 数据采集。

它会：

- 连接 BrainFlow 设备
- 按协议组织 prepare / active / rest
- 控制刺激界面
- 记录 trial 片段
- 做基本质量控制
- 保存 `session_manifest.json` 与 `raw_trials.npz`

### 当前重要特性

- 默认目标频率来自主线 4 目标配置
- 预置采集协议：
  - `stable_12m`
  - `enhanced_45m`
  - `custom`
- 支持 `long_idle_sec`
- 支持最小采样质量门槛和 retry

### 主要输入

- 串口或设备连接方式
- `board_id`
- 目标频率
- 采集协议参数
- 输出目录

### 主要输出

- `artifacts/datasets/<session_id>/session_manifest.json`
- `artifacts/datasets/<session_id>/raw_trials.npz`

### 何时使用

- 新采集一组训练数据
- 追加一个 session
- 补足 idle / switch / long idle 数据

### 不适合用它的情况

- 你只是想做离线比较
- 你要跑外部公开数据
- 你要做服务器批跑

---

## 4. `realtime_online_ui.py`

### 作用

这是主实时在线界面。它读取 deployed profile，接入实时 EEG 流，做在线判断。

### 它会做什么

- 读取 `profile.json`
- 可按 profile 恢复 decoder
- 自动或手动选择 compute backend
- 驱动在线窗口打分
- 执行 gate / decision
- 显示当前状态和日志
- 支持 shadow runtime

### 关键运行语义

- 优先相信 deployed profile，而不是临时 run 内 profile
- 如果 profile 是 fallback/default，需要在 UI 和日志里认真核对
- 在线系统关注的是稳定性和可解释性，不只是单窗 top-1

### 主要输入

- profile 路径
- 设备串口
- 模型选择
- compute backend
- GPU 配置
- shadow mode 开关

### 主要输出

- UI 实时状态
- 日志
- 在线 commit / release 决策结果

### 典型风险

- profile 路径指错
- 设备未就绪
- GPU 自动选择不符合预期
- 训练和在线使用的通道顺序不一致

---

## 5. `training_evaluation_ui.py`

### 作用

这是当前最重要的研究主界面。绝大多数离线任务都从这里发起。

### 当前 UI 公开任务

- `fbcca-weights`
- `model-compare`
- `fbcca-weighted-compare`
- `tdca-local-opt`
- `fbcca-local-opt`
- `fbcca-external-replay-opt`

### 它解决的问题

- 选 dataset / manifest
- 组装参数
- 发起本地任务
- 发起远端服务器任务
- 监控进度
- 打开 run 目录、report、profile

### 主要输入

- dataset root 或 external dataset root
- manifest 列表
- task 类型
- 模型、窗长、backend、search preset
- 服务器连接参数

### 主要输出

- `artifacts/runs/local/...`
- `artifacts/runs/remote/...`
- UI 中的进度、日志、报告路径

### 适合它的原因

- 把参数和 run 管理统一了
- 能清楚看到当前任务到底在做哪条主线
- 比手动拼命令更不容易混淆本地/远端/外部 replay

### 不足

- 任务多，参数面板较重
- 研究性选项和稳定主线共存，需要按 task 读语义

---

## 6. `external_replay_viewer.py`

### 作用

读取 `fbcca-external-replay-opt` 的 `report.json`，展示 `replay_timeline_board`。

### 它适合看什么

- 当前 session 时间轴
- 当前 trial 类型
- 当前 top1 频率
- `p_correct`
- `selected_freq`
- gate 开闭状态
- commit 事件

### 它不做什么

- 不重新跑模型
- 不连接硬件
- 不参与选型

### 适用场景

- 你已经有 external replay 的 run
- 想逐点检查某次 held-out session 是怎么被 gate / decision 处理的

---

## 7. `async_fbcca_validation_ui.py`

### 作用

这个文件更像共享 UI 组件，而不是面向最终用户的独立业务程序。

它主要提供：

- 刺激显示控件
- phase 切换常量
- 采集/实时流程共享的视觉组件

### 什么时候看它

- 你要改刺激表现
- 你要查 phase 切换
- 你要调试采集和实时界面共用的视觉状态

### 什么时候不该从它开始

- 你只是想采集数据
- 你只是想跑评测
- 你只是想看实时结果

---

## 8. 启动建议

### 最稳妥

```bash
python D:\brain\brain_code\02_SSVEP\START_SSVEP.py
```

### 直接进训练评测

```bash
python D:\brain\brain_code\02_SSVEP\entrypoints\start_training_eval.py
```

### 直接进 external replay

```bash
python D:\brain\brain_code\02_SSVEP\entrypoints\start_fbcca_external_replay.py
```

---

## 9. 维护建议

1. UI 文件负责参数组织、线程、日志和可视化，不要把核心算法塞回 UI。
2. 新任务优先接入 `training_evaluation_ui.py`，不要再单独复制一套大面板。
3. 共享的刺激控件和 phase 常量应继续留在 `async_fbcca_validation_ui.py` 一类的共享层，不要在多个 UI 里拷贝。
4. 如果某个 UI 只是“预设某个 task 的薄入口”，优先做成 `entrypoints/` 中的包装，而不是再写一份新 UI。
