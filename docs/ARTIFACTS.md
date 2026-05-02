# 产物说明

这个仓库会有意识地把较大的实验产物保留在 Git 中，优先保证完整备份和可复现性，而不是追求最小克隆体积。

## 已跟踪的产物类型

- `02_SSVEP/artifacts/runs`：正式本地运行和导入的 SSVEP 运行结果、报告、profile 快照和选择快照。
- `02_SSVEP/artifacts/datasets`：当前实验使用的 SSVEP 采集包和外部回放数据集。
- `02_SSVEP/artifacts/deployed_profiles`：已部署或候选的 SSVEP profile。
- `02_SSVEP/_archive`：保留的历史代码和对应历史输出。
- `hybrid_controller/models`：集成视觉/机械臂流程需要的训练模型。
- `artifacts`：仓库级生成索引和跨流程输出。

## 不应该跟踪的内容

下面这些属于缓存或本地临时产物，应该保持忽略：

- `__pycache__`、`.pytest_cache`、`.pytest_tmp*`、`.tmp*`
- `pytest-cache-files-*`、`pytest_tmp*`、`pytest_temp*`、`tmp_pytest*`
- `02_SSVEP/artifacts/gpu_runtime/cupy_cache`
- `02_SSVEP/artifacts/gpu_runtime/tmp`
- 冒烟测试截图和本地 UI 探测输出

## 诊断

运行：

```powershell
powershell -ExecutionPolicy Bypass -File tools\diagnose_workspace.ps1
```

诊断脚本会汇报已跟踪体积、最大文件、产物类别统计、忽略条目和权限受限的临时目录。
