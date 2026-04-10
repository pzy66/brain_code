# hybrid_controller 主线说明

`hybrid_controller` 是当前真机主线目录（ROS 主链 + TCP 兼容 + 圆柱坐标控制）。

## 唯一解释器

- `C:\Users\P1233\miniconda3\envs\brain-vision\python.exe`

入口脚本自带解释器守卫：

- `run_real.py`
- `run_real_ssvep.py`

若误用其他解释器（例如 `.venv`），会报 `Interpreter mismatch` 并退出。

## 启动方式

推荐：在 PyCharm 中从仓库根目录 `C:\Users\P1233\Desktop\brain` 打开，并运行共享配置：

- `Hybrid_Controller_Real_GUI`
- `Hybrid_Controller_Real_SSVEP_GUI`

命令行也可直接启动：

```powershell
C:\Users\P1233\miniconda3\envs\brain-vision\python.exe C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\run_real.py
C:\Users\P1233\miniconda3\envs\brain-vision\python.exe C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\run_real_ssvep.py
```

## 统一解释器解析脚本

以下脚本供 Windows 启动入口统一解析解释器：

- `C:\Users\P1233\Desktop\brain\brain_code\tools\resolve_brain_python.cmd`
- `C:\Users\P1233\Desktop\brain\brain_code\tools\resolve_brain_python.ps1`

可选覆盖变量：

- `BRAIN_PYTHON_EXE`（显式指定 `python.exe` 绝对路径）

默认不再回退 `.venv\Scripts\python.exe`。
