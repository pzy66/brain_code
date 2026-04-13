# SSVEP 异步识别代码说明

本目录是一套围绕异步 SSVEP 的完整工程链路，目标很明确：

1. 看某个闪烁目标时，输出对应频率。
2. 不看任何目标时，输出 `None`，尽量减少误触发。
3. 把采集、训练评测、实时识别拆开，职责清晰。

当前主线是 3 个程序：

- `ssvep_dataset_collection_ui.py`
  只负责采集数据集。
- `ssvep_training_evaluation_ui.py`
  只负责训练、评测、生成报告和 profile。
- `ssvep_realtime_online_ui.py`
  只负责读取 profile 做实时识别。

另外有一个服务器助手：

- `ssvep_server_train_client.py`
  只负责把本地数据上传到服务器、启动训练、查看进度、下载报告和 profile。

## 1. 目录结构

本目录常用文件如下：

- `async_fbcca_idle_standalone.py`
  核心算法和公共解码逻辑。包含 FBCCA、门控状态机、profile 读写、在线分析等。
- `ssvep_dataset_collection_ui.py`
  数据采集 UI。
- `ssvep_training_evaluation_ui.py`
  训练评测 UI。
- `ssvep_training_evaluation_cli.py`
  训练评测 CLI，UI 和服务器助手最终都调用它。
- `ssvep_realtime_online_ui.py`
  实时识别 UI。
- `ssvep_server_train_client.py`
  服务器训练助手。
- `ssvep_core/`
  训练评测、数据集读写、后端计算等公共模块。
- `profiles/`
  本地 profile、报告、服务器下载产物、采集数据集的默认落点。

父目录中保留了几个便捷启动脚本，方便在 PyCharm 里直接运行：

- `C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\START_SSVEP_DATA_COLLECTION_UI.py`
- `C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\START_MODEL_EVALUATION_UI.py`
- `C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\START_FBCCA_REALTIME_UI.py`
- `C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\START_SSVEP_SERVER_TRAIN.py`

## 2. 当前推荐工作流

推荐按下面顺序使用：

1. 先用采集程序采集一轮或多轮数据。
2. 再用训练评测程序训练 FBCCA 权重，或做多模型评测。
3. 训练得到可用 profile 后，用实时识别程序加载 profile 做在线验证。
4. 如果本地训练太慢，再用服务器助手上传数据到服务器训练。

这套设计的原则是：

- 采集程序不训练。
- 训练程序不直接接设备采集。
- 实时程序不做训练，只读 profile。

## 3. 采集程序

主程序：

- `ssvep_dataset_collection_ui.py`

### 3.1 采集程序做什么

采集程序负责：

- 连接 BrainFlow 设备。
- 按协议展示闪烁刺激。
- 在每个 trial 的 active 段采样 EEG。
- 做基本质量检查。
- 把每轮数据保存成可复用数据集。

### 3.2 当前默认协议

当前默认预设是 `stable_12m`，参数为：

- `prepare_sec = 1.0`
- `active_sec = 5.0`
- `rest_sec = 4.0`
- `target_repeats = 10`
- `idle_repeats = 20`
- `switch_trials = 14`
- `long_idle_sec = 0.0`

也就是说，默认每轮没有额外 long-idle 段。这个设计是为了兼容你当前已经在用的短轮采集流程。

### 3.3 新增的 long-idle 采集

采集 UI 现在支持：

- `Long Idle (sec, 0=off)`

当它大于 0 时，每轮会额外增加 1 个 `long_idle` trial：

- 被试只看中心，不看任何闪烁目标。
- 该 trial 会被单独标记为 `label=long_idle`
- manifest 中会记录 `stage=long_idle`

这个 long-idle 的作用不是提高四分类准确率，而是让异步评测里的 `idle_fp_per_min` 更可信。

### 3.4 质量门槛

采集时每个 trial 都做质量检查：

- `MIN_ACTIVE_SEC_FOR_TRAINING = 1.5`
- `MIN_TRIAL_QUALITY_RATIO = 0.90`
- `MAX_TRIAL_RETRIES = 3`

如果有效样本比例太低，会自动重采该 trial，最多 3 次。

### 3.5 提示音和多轮

采集 UI 支持：

- 计划轮数显示
- 当前轮次显示
- 每次只采一轮，结束后手动开始下一轮
- 每个 trial 的 active 开始和结束提示音

这样做是为了让每轮长度可控，降低疲劳，同时保留多轮采集的可重复性。

### 3.6 采集输出格式

每轮会生成一个独立 session 目录，默认在：

- `profiles/datasets/<session_id>/`

固定文件：

- `session_manifest.json`
- `raw_trials.npz`

其中：

- `raw_trials.npz`
  保存每个 trial 的原始 EEG 段，shape 一般是 `samples x channels`
- `session_manifest.json`
  保存协议参数、trial 元数据、质量信息、split 需要的索引信息

### 3.7 manifest 里的关键字段

训练端会依赖这些字段：

- `protocol_signature`
  用于判断是否是同一协议数据。
- `quality_summary`
  记录有效 trial 数、丢弃数、重采统计。
- `trials`
  记录每个 trial 的 `label / expected_freq / trial_id / npz_key / used_samples / target_samples`
- `protocol_config.long_idle_sec`
  明确本轮是否含 long-idle

## 4. 训练评测程序

主程序：

- `ssvep_training_evaluation_ui.py`
- `ssvep_training_evaluation_cli.py`

### 4.1 训练评测程序做什么

它负责：

- 读取一个或多个 session 数据集
- 做 trial 级切分
- 训练模型或权重
- 输出固定窗分类指标
- 输出异步在线可用性指标
- 生成报告和 profile

### 4.2 当前支持的主要任务

当前主推 4 个任务：

- `fbcca-weights`
  只训练 FBCCA 通道权重和子带权重，目标是给实时端提供可部署 profile。
- `classifier-compare`
  跑所有候选模型的 fixed-window 四分类比较，不做完整异步门控搜索。
- `focused-compare`
  先做全模型快速筛选，再对 Top 模型做完整异步端到端评测。
- `profile-eval`
  读取一个已经训练好的 FBCCA weighted profile，在指定数据集上比较：
  `fbcca_plain_all8`、`fbcca_profile_weighted` 和其他模型。

兼容任务仍然保留：

- `model-compare`
- `fbcca-weighted-compare`

但日常建议优先使用前面 4 个任务。

### 4.3 多 seed 现在的语义

多随机种子不是为了“扩充样本量”，而是为了：

- 让同一批数据做多次不同的 trial-level split
- 估计模型结果稳定性
- 减少单次切分的偶然性

当前默认：

- `classifier-compare`: `multi_seed_count = 10`
- `focused-compare`: `multi_seed_count = 5`

训练结论应该看聚合结果，而不是只看某一个 seed 的最好结果。

### 4.4 训练数据如何切分

训练程序使用 trial 级切分，而不是窗口级打散：

- `train : gate : holdout = 60% : 20% : 20%`

这样做是为了避免窗口泄漏。因为同一 trial 切出的窗口高度相关，不能当独立样本随便打散。

### 4.5 双口径评测

训练程序现在同时输出两套指标。

#### 4.5.1 Classifier-only / paper lens

这套指标对应论文常见写法，主要评估分类器本体：

- `Acc_SSVEP`
- `Macro-F1`
- `Confusion Matrix`
- `Mean Decision Time`
- `ITR`

这回答的是：

- 8/10/12/15 Hz 能不能分清
- 哪些频率之间容易混淆
- 单次固定窗决策有多快

#### 4.5.2 End-to-end async lens

这套指标对应真实异步在线可用性：

- `idle_fp_per_min`
- `control_recall`
- `control_recall_at_2s / 3s`
- `switch_detect_rate`
- `switch_latency_s`
- `release_latency_s`
- `inference_ms`

这回答的是：

- 不看目标时会不会误触发
- 看目标时多久能进入 selected
- 目标切换时多久能跟上
- 目光移开时多久能回到 `None`

### 4.6 当前 FBCCA 权重训练在做什么

当前实时主线仍然是：

- `FBCCA + 8通道权重 + 5个全局子带权重`

含义要区分清楚：

- `channel_weights`
  是 8 个 EEG 通道的连续权重
- `subband_weights`
  是 5 个 filter-bank 子带的全局融合权重

当前不训练：

- 每个通道各自一套子带权重矩阵

原因很简单：你现在的数据量还不够支持 `8 x 5` 这种更高自由度的参数，容易过拟合。

### 4.7 权重训练目标

FBCCA weighted 训练已经不是单纯追求 recall，而是优先控制误触发。

当前思路是：

1. 先满足硬约束
   - `idle_fp_per_min <= 阈值`
   - `control_recall >= 下限`
2. 再在满足约束的候选里比较
   - `control_recall`
   - `control_recall_at_3s`
   - `switch_latency_s`
   - `release_latency_s`
   - `Acc_4class / Macro-F1`

这和“只追求准确率”不同。因为异步 SSVEP 的主要问题通常不是四分类分不开，而是 idle 时误触发太多。

### 4.8 frequency-specific control-state

当前门控支持两类思路：

- `unified`
  所有频率共用一套 enter/exit 阈值
- `frequency-specific-threshold`
  每个频率拟合自己的一套阈值

目前已经真正接入的是：

- `frequency-specific-threshold`

当前还没有完全独立实现的是：

- `frequency-specific-logistic`

现在如果你选 logistic，不应把它理解成已经有完整的频率专属逻辑回归分类器训练流程。当前可靠可用的是 threshold 版本。

### 4.9 同 session 和跨 session

训练评测报告会明确区分：

- `internal_holdout`
- `same_session_profile_eval`
- `cross_session_eval`
- `long_idle_used = true/false`

解释如下：

- `internal_holdout`
  同一轮 session 内部切分出的 holdout，只能说明方向，不是最终泛化结论。
- `same_session_profile_eval`
  同一轮既训练 profile 又拿来复评，会偏乐观。
- `cross_session_eval`
  用另一轮独立 session 评测训练好的 profile，才是更可信的部署依据。

## 5. 实时识别程序

主程序：

- `ssvep_realtime_online_ui.py`

### 5.1 实时程序做什么

它只做一件事：

- 读取 profile，加载对应模型和参数，实时输出 `selected_freq` 或 `None`

它不负责训练，也不应该在这里重新选模型、重新拟合阈值。

### 5.2 实时程序如何依赖 profile

实时程序以 profile 为主，而不是以 UI 下拉框为主。

也就是说：

- UI 上选的模型如果和 profile 内记录的模型冲突，最终还是以 profile 为准
- 这是为了避免“加载的是 weighted FBCCA profile，但 UI 误切到别的模型”这种错误

### 5.3 实时端会校验什么

实时端加载 profile 时会检查：

- 通道权重数量是否等于当前 EEG 通道数
- 子带权重数量是否等于当前 filter-bank 子带数
- profile 中的模型状态是否能正常恢复

如果不匹配，会直接报错，不会悄悄截断。

### 5.4 GPU / CPU 后端

实时端支持：

- `auto`
- `cpu`
- `cuda`

但实时端不会盲目强制 CUDA。启动时会比较实际开销，如果 GPU 对单窗口推理并不更快，会自动回退 CPU。

这点是合理的，因为实时识别是低 batch、低延迟问题，不是大批量吞吐问题。

## 6. 服务器训练助手

主程序：

- `ssvep_server_train_client.py`

### 6.1 服务器侧目录约束

服务器所有写入统一限制在：

- `/data1/zkx/brain/ssvep`

其中主要目录为：

- `/data1/zkx/brain/ssvep/code`
- `/data1/zkx/brain/ssvep/data`
- `/data1/zkx/brain/ssvep/reports`
- `/data1/zkx/brain/ssvep/profiles`
- `/data1/zkx/brain/ssvep/logs`

当前代码里写死的远端根目录也是这个位置，不会往系统路径或 home 目录乱写。

### 6.2 服务器助手当前支持的主要操作

按钮层面，当前已经有：

- `只训练FBCCA权重`
- `读取权重评测`
- `精选模型深度分析`
- `全量分类对比`

也就是说，服务器端已经可以分别承担：

- 只训练 weighted FBCCA profile
- 读取已有 weighted profile 做复评
- 跑全模型 fixed-window 分类榜
- 跑筛选后的异步端到端榜

### 6.3 数据流

本地采集后：

- `profiles/datasets/<session_id>/session_manifest.json`
- `profiles/datasets/<session_id>/raw_trials.npz`

上传到服务器后：

- `/data1/zkx/brain/ssvep/data/<session_id>/session_manifest.json`
- `/data1/zkx/brain/ssvep/data/<session_id>/raw_trials.npz`

服务器训练完成后，本地可下载：

- 报告 JSON
- 报告 Markdown
- 图表
- `profile_best_fbcca_weighted.json`
- 推荐实时 profile

## 7. 输出文件说明

### 7.1 训练报告

本地训练报告默认位于：

- `profiles/reports/train_eval/<date>/<run_id>/`

常见文件：

- `offline_train_eval.json`
- `offline_train_eval.md`
- `run.log`
- `selection_snapshot.json`
- `progress_snapshot.json`
- `figures/*.png`

### 7.2 profile

常见 profile 包括：

- `profile_best_candidate.json`
- `profile_best_fbcca_weighted.json`

其中对实时端最关键的是：

- `profile_best_fbcca_weighted.json`

如果该文件经过独立评测验证，可以手动导入到实时程序使用。

### 7.3 服务器下载产物

服务器训练下载后，通常位于：

- `profiles/server_runs/<run_id>/`
- `profiles/server_profiles/<run_id>__profile_best_fbcca_weighted.json`

## 8. 旧数据和新数据的兼容性

当前代码已经兼容两类数据：

### 8.1 旧数据

旧的 session 没有 `long_idle` 也可以：

- 做 `classifier-compare`
- 做 `focused-compare`
- 做 `fbcca-weights`
- 做 `profile-eval`

但限制是：

- `long_idle_required=1` 时不能通过
- `idle_fp_per_min` 的可信度不如带 long-idle 的新数据

### 8.2 新数据

新协议数据如果带有 `long_idle_sec > 0`，训练评测会更合理：

- idle 误触发统计更稳定
- 更适合调 control-state 门控
- 更适合做跨 session 部署验证

## 9. 当前建议的实际使用方式

### 9.1 如果你已经有旧数据

建议这样做：

1. 先跑 `fbcca-weights`
2. 再跑 `profile-eval`
3. 看 `fbcca_plain_all8` 和 `fbcca_profile_weighted` 的差异
4. 结果只当方向性参考，不直接当最终定型依据

### 9.2 如果你准备重新采数据

建议至少 3 轮：

1. 第 1 轮：训练权重
2. 第 2 轮：拟合门控阈值
3. 第 3 轮：完全外测

更稳妥的是 5 轮：

1. 2 轮训练
2. 1 轮 gate
3. 2 轮 cross-session holdout

并且建议把 `Long Idle` 设为 `60~120s`。

## 10. 快速启动

如果你习惯直接运行 Python 启动脚本，用下面这些：

- 采集：
  `C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\START_SSVEP_DATA_COLLECTION_UI.py`
- 训练评测：
  `C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\START_MODEL_EVALUATION_UI.py`
- 实时识别：
  `C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\START_FBCCA_REALTIME_UI.py`
- 服务器训练助手：
  `C:\Users\P1233\Desktop\brain\brain_code\02_SSVEP\START_SSVEP_SERVER_TRAIN.py`

## 11. 当前已知边界

当前代码可以用，但有几个边界需要明确：

1. `frequency-specific-threshold` 已可用。
2. `frequency-specific-logistic` 还不是完整独立实现，不应把它当作已经正式可用的主线。
3. weighted FBCCA 的训练仍然比普通模型慢，因为它要搜索通道权重和子带权重。
4. 四分类准确率高，不等于异步在线可用；真正部署前必须看 `idle_fp_per_min`、`switch_latency_s`、`release_latency_s`。
5. 同 session 训练再同 session 评测的结果会偏乐观，最终结论应尽量基于 `cross_session_eval`。

## 12. 一句话总结

这套代码现在已经形成了清晰主线：

- 采集程序负责产生结构化数据集
- 训练评测程序负责训练 weighted FBCCA 和做多模型评测
- 实时程序负责读取 profile 做在线识别
- 服务器助手负责把训练搬到服务器上执行

如果你的目标是“训练出能直接用于实时 FBCCA 的权重，并系统比较各模型效果”，当前这套结构已经是对的。后续主要优化点不再是程序职责划分，而是：

- 增加 long-idle 数据
- 做跨 session 验证
- 继续压低异步误触发

