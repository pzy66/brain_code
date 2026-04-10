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

## 机械臂改动留档

机械臂主链改动与落盘路径清单见：

- [JetMax_机械臂改动与落盘清单.md](C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\docs\JetMax_机械臂改动与落盘清单.md)

## UI 方位窗说明

最新 UI 中，`Robot Pose` 已改为相机区域右上角悬浮窗：

- 上下方向已按正视图翻转显示
- 窗口 resize 时自动重定位
- 右侧控制区改为可滚动，避免挤压导致方位图显示不全
- 程序启动默认最大化；按 `F11` 可在全屏/最大化之间切换

## 抓取调参与稳定姿态（本轮新增）

抓放链路已新增“抓后姿态对齐 + 下探/吸盘时序可调”：

- 抓取抬升终点不再固定 `z_carry`，而是：
  - `z_settle = clamp(max(z_carry_floor, z_auto(current_r)))`
- 放置释放优先使用硬件 `release()`，若不可用自动回退：
  - `set_state(False) + release_sec`
- `PICK_WORLD` 与 `PICK_CYL` 都会应用同一套 `r/theta bias`（避免只调到手动入口）。

### UI 调参入口

右侧 `Pick Tuning` 区支持实时微调：

- 深度参数：`±1 mm`
- 时序参数：`±0.05 s`
- 释放模式切换：`mode: release/off`
- 控制按钮：
  - `应用到机器人`（调用 ROS 服务）
  - `恢复默认`
  - `保存配置`

### 调参落盘路径

- 当前调参配置文件：
  - `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\dataset\robot_pick_tuning\current_pick_tuning.json`

### ROS 服务（抓放调参）

- `/hybrid_controller/get_pick_tuning`
- `/hybrid_controller/set_pick_tuning`

`/hybrid_controller/state` 额外返回：

- `pick_tuning`
- `post_pick_settle_z`
- `release_mode_effective`

用于现场核对“UI 参数是否已实际生效”。
