# SSVEP 模型实现对照与优化矩阵（2026-04）

## 1. 本轮目标
- 主目标：降低异步场景 `idle_fp_per_min`，并保持 `control_recall` 不低于现有基线。
- 本轮范围：问题 `1/2/3/4/6`，明确不处理问题5（全屏显示）。

## 2. 模型实现级别标注
| 模型 | 实现级别 | 说明 |
|---|---|---|
| `cca` | paper-faithful | 标准参考信号 CCA 基线实现。 |
| `fbcca` | paper-faithful | 采用滤波器组与谐波融合的 FBCCA 基线。 |
| `itcca` | engineering-approx | 个体模板 CCA 的工程近似实现。 |
| `ecca` | engineering-approx | 扩展 CCA 的工程近似实现（模板+参考融合）。 |
| `msetcca` | engineering-approx | MsetCCA 思路的工程近似融合版本。 |
| `trca` | paper-faithful | TRCA 空间滤波主干实现。 |
| `trca_r` | engineering-approx | TRCA-R/eTRCA 家族的滤波器组+集成近似。 |
| `sscor` | engineering-approx | SSCOR 思路的多分量相关评分近似。 |
| `tdca` | engineering-approx | TDCA 思路（延迟嵌入+判别投影）近似。 |
| `oacca` | engineering-approx | 在线自适应 CCA 的模板更新近似。 |

## 3. 优化矩阵（问题 -> 证据 -> 改法 -> 风险 -> 验收）
| 问题 | 证据/文献方向 | 本轮改法 | 主要风险 | 验收阈值 |
|---|---|---|---|---|
| 1 通道选择过拟合 | 训练/调参/评估需要隔离，避免信息泄漏 | 通道评分改为嵌套：`fit_segments` 拟合 decoder，`threshold_segments` 拟合门控阈值，`score_segments` 仅用于评分 | 数据量偏小时评分方差变大 | 通道排序在重复会话下稳定，`idle_fp_per_min` 不劣化 |
| 2 方法学一致性 | 异步对比应明确“论文复现度” | 报告中新增 `implementation_level` 与 `method_note`，并输出模型方法映射 | 用户误把近似实现当论文原法 | 报告可直接辨识每模型实现级别 |
| 3 评测状态机不连续 | 异步评测应尽量接近连续在线过程 | 伪在线评测改为连续门控，不再每 trial 强制 `gate.reset()` | 若 trial 间无休息数据，可能出现残留状态 | 连续评测指标趋势与实机在线更一致 |
| 4 切换延迟乐观 | 漏检切换不能被“忽略” | 新增惩罚版 `switch_latency_s`：漏检切换使用 `trial_duration + win_sec` 计入中位数 | 指标数值会较旧版更严格 | 切换指标与人工观察更一致，不再系统性偏乐观 |
| 6 时钟资源浪费 | 非 flicker 阶段不应持续高频刷新 | 刺激时钟仅在 `flicker=True` 运行，其他阶段停线程 | 频繁启停线程的边界同步 | CPU 占用下降，显示逻辑无回归 |

## 3.1 代码落地点（本轮）
| 问题 | 代码落地点 | 关键实现 |
|---|---|---|
| 1 通道选择过拟合 | `async_fbcca_idle_standalone.py::select_auto_eeg_channels_for_model` | 通道评分改为 `fit_segments / threshold_segments / score_segments` 三段分离，避免阈值拟合与评分复用同一段。 |
| 2 方法学一致性 | `async_fbcca_idle_standalone.py::MODEL_IMPLEMENTATION_LEVELS/MODEL_METHOD_NOTES`、`model_method_mapping_payload`、`_benchmark_single_model` | 在每模型结果与总报告中输出 `implementation_level` 与 `method_note`，并固化映射字典。 |
| 3 评测状态机不连续 | `async_fbcca_idle_standalone.py::evaluate_decoder_on_trials` | 门控 `gate.reset()` 仅在整体评测开始时调用，不再每 trial 重置。 |
| 4 切换延迟乐观 | `async_fbcca_idle_standalone.py::evaluate_decoder_on_trials`、`benchmark_metric_definition_payload` | 对漏检切换使用惩罚延迟 `trial_duration_sec + win_sec`，`switch_latency_s` 改为惩罚版中位数。 |
| 6 时钟资源浪费 | `async_fbcca_validation_ui.py::FourArrowStimWidget.apply_phase`、`ssvep_model_evaluation_ui.py::phase_from_benchmark_line` | 非 flicker 阶段显式 `stop_clock()`；模型离线评分阶段 `flicker=False`。 |

## 3.2 数据集导出落地点（评测流程）
| 项目 | 代码落地点 | 说明 |
|---|---|---|
| CLI 参数 | `async_fbcca_idle_standalone.py::build_parser` | benchmark 新增 `--dataset-dir`，默认 `profiles/datasets`。 |
| 数据集写出 | `async_fbcca_idle_standalone.py::save_benchmark_dataset_bundle` | 每次 benchmark 会话写出 `session_manifest.json` 与 `raw_trials.npz`。 |
| 失败即中断 | `async_fbcca_idle_standalone.py::BenchmarkRunner.run` | 数据集写出异常不吞掉，直接让 benchmark 失败退出。 |
| UI 参数传递 | `ssvep_model_evaluation_ui.py::EvalConfig/build_benchmark_command` | UI 的 `Dataset Dir` 传入 benchmark 子进程。 |

## 4. 指标定义（本轮）
- 排序优先级保持：`idle_fp_per_min > control_recall > switch_latency_s > itr_bpm > inference_ms`。
- `switch_latency_s` 定义为惩罚版中位数：
  - 若切换 trial 检出：使用首个正确锁定延迟；
  - 若切换 trial 漏检：使用 `trial_duration_sec + win_sec` 作为惩罚延迟。

## 5. 参考文献（用于方法映射）
- CCA 基线：DOI [10.1088/1741-2560/6/4/046002](https://doi.org/10.1088/1741-2560/6/4/046002)
- FBCCA：PubMed [26035476](https://pubmed.ncbi.nlm.nih.gov/26035476/)
- CCA 变体对比：DOI [10.1371/journal.pone.0140703](https://doi.org/10.1371/journal.pone.0140703)
- TRCA：PubMed [28436836](https://pubmed.ncbi.nlm.nih.gov/28436836/)
- 异步 control-state：PubMed [26246229](https://pubmed.ncbi.nlm.nih.gov/26246229/)
- 伪在线评测框架：DOI [10.1088/1741-2552/ad171a](https://doi.org/10.1088/1741-2552/ad171a)
