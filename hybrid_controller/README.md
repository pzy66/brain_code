# hybrid_controller 主线说明

`hybrid_controller` 是当前真机主线目录（ROS 主链 + TCP 兼容 + 圆柱坐标控制 + 视觉/SSVEP UI）。

## 1. 唯一解释器

- `C:\Users\P1233\miniconda3\envs\brain-vision\python.exe`

入口脚本自带解释器守卫：

- `run_real.py`
- `run_real_ssvep.py`

如果误用其他解释器（如 `.venv`），会直接报 `Interpreter mismatch` 并退出。

## 2. 启动方式

推荐在 PyCharm 里从仓库根目录 `C:\Users\P1233\Desktop\brain` 打开，然后运行：

- `Hybrid_Controller_Real_GUI`
- `Hybrid_Controller_Real_SSVEP_GUI`

命令行启动：

```powershell
C:\Users\P1233\miniconda3\envs\brain-vision\python.exe C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\run_real.py
C:\Users\P1233\miniconda3\envs\brain-vision\python.exe C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\run_real_ssvep.py
```

## 3. 机器人与ROS入口

- JetMax 侧 ROS 运行入口：`robot/run_hybrid_controller_ros_runtime.sh`
- JetMax 侧 TCP 兼容入口：`robot/run_jetmax_robot_runtime.sh`
- 桌面端远程启动工具：`robot/tools/jetmax_start_ros_runtime.py`

## 4. 关键功能状态

- UI 主布局：主相机画面 + 右上角 Robot Pose + 右侧控制区（可滚动）。
- 右上角方位图：已按正视图语义翻转，随窗口大小自动重定位。
- 抓取链路：
  - 视觉主路径默认 `PICK_WORLD`（ROS 主链）。
  - `PICK_WORLD` 与 `PICK_CYL` 共用 `r/theta bias`。
  - 抓后稳定姿态采用 `z_settle = clamp(max(z_carry_floor, z_auto(r)))`。
- 抓放调参：
  - ROS 服务：`/hybrid_controller/get_pick_tuning`、`/hybrid_controller/set_pick_tuning`
  - 落盘：`dataset/robot_pick_tuning/current_pick_tuning.json`

## 5. 维护脚本

- 清理临时文件：`tools/clean_hybrid_temp.ps1`
- 一键检查（编译 + 测试）：`tools/run_hybrid_checks.ps1`

示例：

```powershell
powershell -ExecutionPolicy Bypass -File C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\tools\clean_hybrid_temp.ps1
powershell -ExecutionPolicy Bypass -File C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\tools\run_hybrid_checks.ps1
```

## 6. 文档入口

- 总文档索引：`docs/README.md`
- 真机联动：`docs/JetMax_真机联动说明.md`
- 抓取链路：`docs/JetMax_抓取流程确认说明.md`
- 圆柱坐标：`docs/JetMax_圆柱坐标改造说明.md`
- 历史报告归档：`docs/archive/`

## 7. 当前已知风险

- `app.py` 仍较大，`runtime_info` 兼容层字段较多（正在逐步下沉到 typed state）。
- 目录中可能出现被外部进程占用的临时测试目录（`tmp_pytest*`、`pytest-cache-files-*`）；建议先关闭占用进程，再运行清理脚本。
- `numpy.matrix` 的 PendingDeprecationWarning 来自历史视觉处理路径，功能正常，但建议后续改成 `ndarray`。
