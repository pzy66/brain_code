# MI 采集模块说明

本模块负责三件事：

1. 驱动采集 UI 和会话流程
2. 写入 marker 并维护事件日志
3. 把一次采集保存成可追溯、可训练、可回看的数据包

它不负责在线分类。

## 1. 入口

推荐从项目根目录启动：

```powershell
python run_01_collection_only.py
```

等价入口：

```powershell
python code/collection/run_collection_pycharm.py
```

直接运行采集主程序：

```powershell
python code/collection/mi_data_collector.py --subject-id 001 --session-id 20260401_203000
```

常用参数：

- `--serial-port COM3`
- `--output-root <path>`
- `--subject-id 001`
- `--session-id 20260401_203000`
- `--synthetic`

## 2. 关键代码文件

- [mi_data_collector.py](./mi_data_collector.py)
  采集 UI、会话状态机、显示逻辑、用户操作、设备线程协调。
- [mi_collection.py](../shared/src/mi_collection.py)
  保存导出、事件映射、segments 构建、manifest 写入。

## 3. 开始采集前必须满足

1. 使用当前仓库的 `.venv`，或任意一台机器上你自己准备好的 `brain_code` 环境，并安装 requirements。
2. 非 synthetic 板卡必须选择有效串口。
   - 如果当前已经枚举到可用串口，则所选端口必须出现在列表里。
   - 如果当前完全没有枚举到任何可用串口，仍允许操作员手工输入端口尝试连接一次。
   - Windows 明确标记为 `Problem/Disconnected` 的端口会在开始前直接拒绝。
3. 通道配置必须固定：
   - `channel_names=C3,Cz,C4,PO3,PO4,O1,Oz,O2`
   - `channel_positions=0,1,2,3,4,5,6,7`
4. 当前采集 UI 只开放以下板卡：
   - `Cyton（8 通道）`
   - `Cyton Daisy（仅前 8 通道）`
   - `Synthetic`
   当前流程是固定 8 通道协议，因此不会再暴露 `Ganglion（4 通道）` 这类必然无法满足通道约束的板卡选项。
5. 输出目录可写，默认是 `datasets/custom_mi`。

## 4. 默认会话流程

默认顺序：

1. 连接设备后，操作员先在预览面板检查连接质量。
2. `calibration`
3. `MI run 1`
4. `MI run 2`
5. `continuous block 1`
6. `MI run 3`
7. `continuous block 2`
8. `idle_block`
9. 自动保存

说明：

- 连接设备后，操作员仍应先在预览面板手工观察信号质量；正式流程现在直接从 `calibration` 开始，不再插入单独的 `quality_check` 阶段。
- 连接设备后，控制区会额外显示 `显示完整流程计划` 和 `显示 MI 主任务计划` 两个按钮；它们会按当前界面参数动态列出后续阶段和每段时长，并汇总预计正式计时总长，便于开始前快速核对。
- `practice` 仍然保留；只有当 `practice_sec > 0` 时，才会插入到 `calibration` 和 `MI run 1` 之间。默认关闭。
- 正式采集、实时推理和通道监看现在使用同一套串口校验规则，避免不同入口表现不一致。
- continuous 插入位置由 `run_count` 和 `continuous_block_count` 自动决定。
- 默认 `run_count=3`、`continuous_block_count=2` 时，continuous 插在 MI run 2 和 MI run 3 之后。

默认参数：

- `trials_per_class=10`
- `run_count=3`
- `max_consecutive_same_class=2`
- `baseline/cue/imagery/iti = 2.0/2.0/4.0/2.0s`
- `run_rest_sec=60`
- `long_run_rest_every=2`
- `long_run_rest_sec=120`
- `calibration_open/closed/eye/blink/swallow/jaw/head = 60/60/30/20/20/20/20s`
- `practice_sec=0`（保留功能，默认关闭）
- `idle_block = 2 x 60s`
- `continuous = 2 x 240s`
- `continuous_command = 4-5s`
- `continuous_gap = 2-3s`
- `include_eyes_closed_rest_in_gate_neg=False`
- `use_separate_participant_screen=True`

continuous 约束：

- `continuous_block_count <= run_count`
- `continuous_command_max_sec >= continuous_command_min_sec`
- `continuous_gap_max_sec >= continuous_gap_min_sec`
- `continuous_block_sec >= continuous_command_min_sec`
- 计划弹窗中的“总时间”按上述正式计时配置直接累加，只展示流程名、各段时长与总时间，不包含语音播报、提示音、人工暂停、切屏与保存耗时。

## 5. 现在到底采集并保存什么

一次保存会同时写出真源层、语义层、会话层和派生层。

### 5.1 真源层

这层用于长期保留和后续追溯。

- `*_board_data.npy`
  保存裁剪后的完整 BrainFlow 二维矩阵，`dtype=float32`。
  这不是只含 EEG 的数组，而是包含保存时保留的所有 board 行。
- `*_board_map.json`
  说明 `board_data.npy` 每一行的含义，包括：
  - `selected_eeg_rows`
  - `marker_row`
  - `timestamp_row`
  - `package_num_row`
  - `channel_rows`
  - `board_descr`
  - session crop 范围
- `*_raw.fif`
  这是 MNE 友好的工作文件，只包含：
  - 选中的 EEG 通道
  - 一个 `STI 014` stim 通道
  - 注释化后的事件与区间
  它不是完整 board 真源。
- `*_events.csv`
  这是原子事件流，每个 marker 一行，字段包括：
  - `event_index`
  - `save_index`
  - `event_name`
  - `marker_code`
  - `trial_id`
  - `mi_run_index`
  - `run_trial_index`
  - `block_index`
  - `prompt_index`
  - `class_name`
  - `command_duration_sec`
  - `execution_success`
  - `sample_index`
  - `absolute_sample_index`
  - `elapsed_sec`
  - `iso_time`

### 5.2 语义层

这层把原子事件整理成更容易审查和训练的结构。

- `*_trials.csv`
  每个 MI trial 一行，字段包括：
  - `trial_id`
  - `save_index`
  - `mi_run_index`
  - `run_trial_index`
  - `class_name`
  - `display_name`
  - `accepted`
  - `cue_onset_sample`
  - `imagery_onset_sample`
  - `imagery_offset_sample`
  - `trial_end_sample`
  - `note`
- `*_segments.csv`
  每个语义区间一行，字段包括：
  - `segment_id`
  - `save_index`
  - `segment_type`
  - `label`
  - `start_sample`
  - `end_sample`
  - `duration_sec`
  - `trial_id`
  - `mi_run_index`
  - `run_trial_index`
  - `block_index`
  - `prompt_index`
  - `accepted`
  - `execution_success`
  - `source_start_event`
  - `source_end_event`

当前 `segment_type` 可能包括：

- `trial`
- `baseline`
- `cue`
- `imagery`
- `iti`
- `calibration`
- `practice`
- `run_rest`
- `eyes_open_rest`
- `eyes_closed_rest`
- `idle_block`
- `continuous_block`
- `artifact_block`
- `continuous_prompt`

### 5.3 会话描述层

- `*_session_meta.json`
  保存本次落盘的总体描述，包括：
  - `schema_version`
  - `subject_id`
  - `session_id`
  - `save_index`
  - `run_stem`
  - `session` 参数快照
  - `sampling_rate_hz`
  - `selected_eeg_rows`
  - `marker_row`
  - `timestamp_row`
  - `package_num_row`
  - 样本数量与时长
  - 各派生文件样本数量
  - `raw_preservation_level`
  - `board_data_sha256`
  - `source_alignment_policy`
  - `files` 相对路径索引
- `session_meta_latest.json`
  当前 `ses-*` 目录下最近一次保存的指针文件。
- `*_quality_report.json`
  连续信号的质量统计，主要使用微伏单位。

### 5.4 派生层

这层只服务训练与评估，不替代真源层。

- `*_mi_epochs.npz`
  accepted MI imagery epoch，核心键：
  - `X_mi`
  - `y_mi`
  - `mi_trial_ids`
  - 公共元数据键
- `*_mi_epochs.meta.json`
  写明派生来源和策略，例如：
  - `source_run_stem`
  - `source_save_index`
  - `source_sha256`
  - `source_files`
  - `derivation_name=mi_epochs`
  - `derivation_policy`
- `*_gate_epochs.npz`
  gate 训练数据，核心键：
  - `X_gate_pos`
  - `X_gate_neg`
  - `X_gate_hard_neg`
  - `gate_neg_sources`
  - `gate_hard_neg_sources`
  - 公共元数据键
- `*_artifact_epochs.npz`
  artifact rejector 数据，核心键：
  - `X_artifact`
  - `artifact_labels`
  - 公共元数据键
- `*_continuous.npz`
  continuous online-like 评估数据，核心键：
  - `X_continuous`
  - `continuous_event_labels`
  - `continuous_event_samples`
  - `continuous_event_end_samples`
  - `continuous_block_indices`
  - `continuous_prompt_indices`
  - `continuous_execution_success`
  - `continuous_command_duration_sec`
  - `continuous_block_start_samples`
  - `continuous_block_end_samples`
  - 公共元数据键

所有派生 `npz` 的公共元数据键包括：

- `schema_version`
- `class_names`
- `channel_names`
- `sampling_rate`
- `signal_unit`
- `subject_id`
- `session_id`
- `save_index`
- `run_stem`
- `trials_per_class`
- `mi_run_count`
- `trials_per_run`
- `total_trials`
- `accepted_trials`
- `rejected_trials`
- `created_at`

## 6. 显示和保存的边界

这是最容易误解的地方：

- EEG 预览滤波只用于显示，不会写回落盘文件。
- 阻抗模式显示的是当前阻抗检查信息，不属于训练输入。
- `*_board_data.npy` 保存的是裁剪后的连续 board 数据，最接近保存时的板卡输出。
- `*_raw.fif` 保存的是 EEG + STIM 的工作视图，不是完整 board 真源。
- `*_npz` 保存的是切好的训练窗口，信号单位统一写成 `volt`。
- `*_quality_report.json` 的统计口径基于微伏连续信号。

## 7. 单窗口 / 双窗口

默认 `use_separate_participant_screen=True`：

- 操作员窗口：配置、设备状态、控制按钮、实时预览
- 受试者窗口：全屏提示词和倒计时

实际行为：

- 连接成功后先停留在操作员窗口检查信号
- 点击“开始采集”后，如果启用双屏，则切到受试者界面
- 关闭该选项时，全流程在操作员窗口显示

## 8. 采集中可做的操作

按钮：

- 连接设备
- 开始采集
- 暂停/继续
- 标记坏试次
- 停止并保存
- 断开设备

快捷键：

- `Space`：暂停/继续
- `B`：
  - MI trial 阶段：标记坏试次
  - continuous 阶段：标记当前命令执行失败，写入 `execution_success=0`
- `N`：提前结束当前 phase，默认主要用于 `practice`
- `Esc`：停止并保存

提示音：

- 每段正式计时前都会先播报流程名，然后播报“准备”，最后播放一声开始滴声。
- start marker 写入发生在开始滴声结束之后；也就是说，保存出来的 `baseline`、`cue`、`imagery`、`iti`、校准段、休息段、idle 段和 continuous block 都从滴声结束后的采样点开始。
- 每段正常结束时先写 end marker，再立刻播放结束提示音。结束提示音不放进下一段数据窗口。
- `cue` 现在额外写入 `cue_end` marker，用于把任务提示段和后面的“准备——滴”分开；旧数据如果没有 `cue_end`，保存逻辑仍会回退到 `imagery_start` 作为 cue 结束。
- 这些提示音会让两个 phase 之间出现额外准备时间，但不会改变各 phase 配置的正式采集时长。

## 9. 保存完成后的目录结果

每次保存至少会写入：

- `*_board_data.npy`
- `*_board_map.json`
- `*_raw.fif`
- `*_events.csv`
- `*_trials.csv`
- `*_segments.csv`
- `*_session_meta.json`
- `session_meta_latest.json`
- `*_quality_report.json`
- `*_mi_epochs.npz`
- `*_mi_epochs.meta.json`
- `*_gate_epochs.npz`
- `*_gate_epochs.meta.json`
- `*_artifact_epochs.npz`
- `*_artifact_epochs.meta.json`
- `*_continuous.npz`
- `*_continuous.meta.json`

同时更新：

- `datasets/custom_mi/collection_manifest.csv`

命名规则见 [README_SAVE_NAMING.md](<workspace>/mi_classifier/code/collection/README_SAVE_NAMING.md)。

## 10. 常见问题

### 10.1 无法开始采集

优先检查：

- 串口
- 板卡类型
- 固定通道名
- 固定通道位置

### 10.2 训练找不到数据

确认会话目录里确实存在：

- `*_mi_epochs.npz`
- 如需 gate/artifact/continuous，则对应任务文件也存在

### 10.3 marker 写入失败导致中止

程序会主动终止本次保存，避免生成对不上 marker 的脏数据。应先排查：

- 设备连接
- marker 写入
- board 数据读取是否稳定

### 10.4 `collection_manifest.csv` 报 schema mismatch

当前仓库已经不做旧 manifest 自动迁移。如果你手工留下了旧表头 manifest，需要先删除或整理该文件，再继续采集。
