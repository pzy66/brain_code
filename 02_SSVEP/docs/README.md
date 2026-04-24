# docs 说明

`docs/` 放的是“为什么这样做”和“怎么判断当前结果是否合理”的文档，不是简单的目录索引。

根目录与子目录 README 的分工如下：

- 根目录 `README.md`
  - 项目总览、主入口、运行方式
- 各子目录 README
  - 该目录下的程序做什么、输入输出是什么
- `docs/`
  - 方法口径、对齐说明、迁移规则、诊断指南

如果你现在接手这套 SSVEP 工程，建议按下面顺序阅读：

1. [../README.md](/D:/brain/brain_code/02_SSVEP/README.md)
2. [../apps/README.md](/D:/brain/brain_code/02_SSVEP/apps/README.md)
3. [../entrypoints/README.md](/D:/brain/brain_code/02_SSVEP/entrypoints/README.md)
4. [../ssvep_core/README.md](/D:/brain/brain_code/02_SSVEP/ssvep_core/README.md)
5. 再回到本目录看专题文档

---

## 1. 当前最重要的专题

### 方法与实现对齐
- `METHOD_ALIGNMENT_AND_OPTIMIZATION_MATRIX.md`
- 用来回答：
  - 某个 decoder 是 `paper-faithful` 还是 `engineering-approx`
  - 哪些结论能直接对外说，哪些只能算研究分支观察

### 迁移与目录规则
- `MIGRATION.md`
- 用来回答：
  - 为什么 run 目录都收敛到 `artifacts/`
  - 历史脚本和当前 run-based 结构怎么对应

### 历史归档索引
- `ARCHIVE_INDEX.md`
- 用来回答：
  - `_archive/` 下面保留了什么
  - 哪些目录已经不在 active import graph 里

---

## 2. external replay 主线现在怎么读

当前 external 8 通道 replay 主线的关键文档不在单独一页，而是在：

- [ssvep_core/README.md](/D:/brain/brain_code/02_SSVEP/ssvep_core/README.md)
- external replay 相关 run 的 `report.json` / `report.md`

读 external replay 时，建议按这个顺序：

1. `status` 与 `status_reasons`
2. `strict_eligible_candidate_count`
3. `diagnostic_best_row`
4. `replay_frequency_breakdown`
5. `decision_bottleneck_summary`
6. `reference_diagnostics_board`

原因是 external replay 当前最容易出问题的地方不是 top-1 分类，而是：

- tune 数据是否泄漏
- confidence 是否把某个频点整体压掉
- decision evidence 的零点是否对齐

---

## 3. 如何判断问题在 decoder、confidence 还是 decision

### 看 `replay_frequency_breakdown`

如果某个频点：

- `raw_correct_rate` 高
- 但 `gate_pass_rate` 很低

优先怀疑 confidence / enter reference，不要先动 decoder。

### 看 `decision_bottleneck_summary`

如果失败主要是：

- `decoder_miss` 高：先看 decoder
- `confidence_reject_miss` 高：先看 calibrator / gate / reference
- `decision_miss` 高：再看 decision 参数和 centered evidence

### 看 `reference_diagnostics_board`

如果某个频点的：

- `enter_reference` 接近或高于 `positive_trial_max_p50 / p75`
- `reference_headroom_p50` 为负

说明当前 enter reference 已经高于该频点正确 trial 的典型置信度峰值。此时继续加严 gate 没意义，应该先修 calibration correctness。

---

## 4. 当前 external replay 的强约束

这几条是当前主线口径，不应随意改：

1. `tune_split` 对 external replay 来说就是 outer-train OOF rows
2. 不能把 `train_full` 当 tuning 数据
3. `simulation_only_profile=true`
4. 不覆盖真实在线 deployed profile
5. `itcca/ecca` 当前只做 diagnostic compare，不进 strict final selection
6. 在 fixed-window confidence 还不稳时，不引入 dynamic stopping

这些约束不是风格问题，而是为了防止：

- 报告口径失真
- 外部数据结果误导真实在线系统
- 研究分支和主线部署分支混在一起

---

## 5. 继续优化时的文档维护规则

以后只要改了下面这些语义，README 和 docs 必须同步更新：

- external replay 的 tune / holdout 定义
- strict selection 规则
- diagnostic row 规则
- profile promotion 规则
- decoder family 的主线 / 诊断分工

最低要求是同时更新：

- [ssvep_core/README.md](/D:/brain/brain_code/02_SSVEP/ssvep_core/README.md)
- 本文件

不要只在提交说明里说“改了语义”，否则几轮之后报告字段就会变成只能靠读代码猜。
