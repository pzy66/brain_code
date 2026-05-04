# Start Here

`brain_code` 是正式代码仓库。请在这个目录里开发、测试、提交和推送。

## 第一次运行

```powershell
cd C:\Users\P1233\Desktop\brain\brain_code
$py = & .\tools\resolve_brain_python.cmd
& $py -m pip install -e ".[dev,gui,ssvep,mi,hybrid]"
& $py -m brain diagnose
& $py -m brain launch --simulate
```

需要打开统一入口时运行：

```powershell
& $py -m brain
```

## 常用入口

- 统一入口：`python -m brain`
- 环境诊断：`python -m brain diagnose`
- MI/SSVEP 统一采集：`run_unified_collection.py`
- SSVEP 工具启动器：`02_SSVEP/START_SSVEP.py`
- Hybrid Controller：`hybrid_controller/run_real.py`
- Hybrid Controller SSVEP 模式：`hybrid_controller/run_real_ssvep.py`

## 数据位置

默认使用仓库内 `datasets/`：

- `datasets/MI/`
- `datasets/SSVEP/`
- `datasets/vision/`
- `datasets/profiles/`

如果数据放在外部磁盘，设置：

```powershell
$env:BRAIN_DATA_ROOT = "D:\brain_data"
```

真实数据、模型权重、运行报告和日志不会进入 Git。
