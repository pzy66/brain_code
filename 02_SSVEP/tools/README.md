# tools/README

本目录是脚本化与运维化工具集合。它的定位不是“给最终用户点着用”，而是：

- 纯命令行运行
- 服务器辅助
- 旧产物迁移
- 对外部数据做快速分析

如果你想批跑、写脚本、做服务器任务，优先看这里。

---

## 1. 文件总览

| 文件 | 作用 |
|---|---|
| `training_evaluation_cli.py` | 训练评测纯 CLI 入口 |
| `compare_external_ssvep_fbcca_tdca.py` | 外部 GDF 数据上做固定窗 FBCCA vs TDCA 快速对比 |
| `server_train_client.py` | 远端服务器任务提交、同步、下载结果 |
| `prepare_server_layout.py` | 生成或创建 Linux 服务器目录结构 |
| `migrate_legacy_artifacts.py` | 把旧版产物迁移到当前 `artifacts/` 规范 |

---

## 2. `training_evaluation_cli.py`

### 作用

这是训练评测体系的纯命令行入口。UI 做的很多事，底层最终也会落到这类配置对象和运行函数上。

### 它适合什么

- 服务器
- 批量脚本
- 无图形环境
- 需要把参数固定写进命令

### 当前支持的任务

#### UI 同步公开任务

- `fbcca-weights`
- `model-compare`
- `fbcca-weighted-compare`
- `tdca-local-opt`
- `fbcca-local-opt`
- `fbcca-external-replay-opt`

#### CLI 额外任务

- `focused-compare`
- `classifier-compare`
- `profile-eval`

### 参数风格

它支持三类输入：

1. 本地 manifest 输入
2. external replay 输入
3. 纯参数化 override

### 常用示例

#### 运行本地 TDCA local opt

```bash
python <repo>\02_SSVEP\tools\training_evaluation_cli.py ^
  --task tdca-local-opt ^
  --dataset-manifest <repo>\02_SSVEP\artifacts\datasets\subject001_xxx\session_manifest.json ^
  --search-preset reduced13
```

#### 运行本地 FBCCA local opt

```bash
python <repo>\02_SSVEP\tools\training_evaluation_cli.py ^
  --task fbcca-local-opt ^
  --dataset-manifest <repo>\02_SSVEP\artifacts\datasets\subject001_xxx\session_manifest.json ^
  --search-preset reduced40
```

#### 运行 external replay

```bash
python <repo>\02_SSVEP\tools\training_evaluation_cli.py ^
  --task fbcca-external-replay-opt ^
  --external-dataset-root <repo>\02_SSVEP\artifacts\datasets\external\dataset_ssvep_led_github ^
  --subject s1 ^
  --search-preset smoke8 ^
  --outer-eval loso4
```

#### 运行通用 model compare

```bash
python <repo>\02_SSVEP\tools\training_evaluation_cli.py ^
  --task model-compare ^
  --include-manifests <path>\to\s1\session_manifest.json,<path>\to\s2\session_manifest.json
```

### 使用注意

1. 非 external replay 任务必须提供 `--dataset-manifest` 或 `--include-manifests`。
2. external replay 任务必须提供 `--subject`。
3. 默认会按 run 目录组织结果，建议不要关掉这条规则。
4. UI 里可见的 preset 和 CLI 的 preset 应保持一致，不要手工发明新名字。

---

## 3. `compare_external_ssvep_fbcca_tdca.py`

### 作用

这是一个快速对比脚本，用外部 GDF 数据做固定窗 LOSO 比较，重点是：

- `fbcca_fixed_all8`
- `tdca_like_legacy`

它更像早期 sanity check 或快速基线确认工具，不是完整 async-first 主线。

### 适合场景

- 快速确认“在某个公开数据上 FBCCA 和 TDCA 谁方向更好”
- 不需要整套 gate / confidence / decision / replay timeline

### 不适合场景

- 做最终 external replay 选型
- 做 profile 发布
- 做完整 held-out pseudo-online 诊断

### 使用提醒

如果你已经进入当前 external replay 主线，优先用 `fbcca-external-replay-opt`，这个脚本只保留为轻量对照。

---

## 4. `server_train_client.py`

### 作用

这是远端训练的运维脚本，负责：

- 发现本地 dataset
- 同步代码到远端
- 上传数据
- 在远端发起训练任务
- 轮询状态
- 下载结果
- 可选发布 profile

### 典型使用路径

1. 本地准备代码和数据
2. 同步到 Linux 服务器
3. 在远端运行训练评测 CLI
4. 把 `report.json`、`profile.json` 等拉回本地

### 当前约束

- 远端路径严格限制在 `/data1/zkx/...`
- 远端训练只调用 `tools/training_evaluation_cli.py`，不走 PyQt UI 入口；服务器环境可以不安装 `PyQt5`
- 同步时会跳过：
  - `_archive`
  - `artifacts`
  - `.git`
  - `__pycache__`
  - 大量缓存和二进制输出

### 什么时候用

- 本地算力不够
- 你要跑 CUDA 版本
- 你要把 run 留在统一远端目录

### 什么时候不用

- 你只是在本机 smoke
- 你只是调 UI

---

## 5. `prepare_server_layout.py`

### 作用

给 Linux 服务器创建标准目录结构，避免远端路径随意生长。

### 典型用途

- 初始化新服务器
- 检查目录结构是否符合当前 SSVEP 主线要求

### 输出内容

会围绕 `/data1/zkx/brain/...` 生成或展示类似目录：

- code
- data
- reports
- profiles
- logs
- tmp

### 推荐习惯

先 dry-run，再 `--apply 1`。

---

## 6. `migrate_legacy_artifacts.py`

### 作用

把旧版 `2026-04_async_fbcca_idle_decoder` 一类目录中的历史产物，迁移到当前统一 artifact 结构。

### 迁移对象

- datasets
- smoke 目录
- reports
- server runs
- server profiles
- default profile

### 迁移目标

- `artifacts/datasets/`
- `artifacts/runs/_legacy_imported/`
- `artifacts/deployed_profiles/`

### 适合场景

- 老项目切到新结构
- 保留旧实验结果但不再让新代码 import 老目录

### 不要做的事

- 不要把迁移脚本当成日常运行入口
- 不要边跑新任务边随手迁移旧目录

---

## 7. 推荐分工

### 你是人工使用者

优先：

- `START_SSVEP.py`
- `entrypoints/`
- `apps/`

### 你是批跑/服务器使用者

优先：

- `training_evaluation_cli.py`
- `server_train_client.py`
- `prepare_server_layout.py`

### 你在做历史整理

优先：

- `migrate_legacy_artifacts.py`

---

## 8. 维护建议

1. 新的研究任务，优先先接入 `training_evaluation_cli.py`，再决定是否需要 UI。
2. 服务器逻辑继续放在 `server_train_client.py` 一类脚本里，不要混回 UI 文件。
3. 迁移脚本只负责历史兼容，不要把新逻辑写进迁移链路。
4. 如果一个工具脚本已经被主线任务替代，应在文档里明确它现在是 sanity compare 还是正式主线，避免使用者误解。
