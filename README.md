# Brain Code

`brain_code/` 现在是这个项目的唯一整理后主线目录。

从现在开始，和统一总控相关的代码、脚本、测试、部署说明都以这里为准：

- 主程序：`brain_code/hybrid_controller/`
- 启动脚本：`brain_code/scripts/`
- 测试：`brain_code/tests/`
- 文档：`brain_code/docs/`

外层根目录里如果还有旧的 `hybrid_controller/`、`tests/`、`scripts/` 副本，不再作为主线维护目标；`brain_code/` 才是你后续在 PyCharm 里应该继续开发的那套。

## 推荐工作方式

PyCharm 继续打开整个：

- `C:\Users\P1233\Desktop\brain`

但统一使用下面这个解释器：

- `C:\Users\P1233\miniconda3\envs\brain-vision\python.exe`

运行和调试时，工作目录建议设为：

- `C:\Users\P1233\Desktop\brain\brain_code`

## PyCharm 最简单启动方式

根项目已经补好了 3 个可直接点运行的配置，打开 `brain` 后在右上角运行配置里直接选：

- `Hybrid_Controller_Real_GUI`
- `Hybrid_Controller_Sim_GUI`
- `Hybrid_Controller_Fake_Robot_Server`

对应关系：

- `Hybrid_Controller_Real_GUI`
  连接真机 JetMax，默认 `192.168.149.1:8888`
- `Hybrid_Controller_Sim_GUI`
  连接本地 fake robot，默认 `127.0.0.1:8899`
- `Hybrid_Controller_Fake_Robot_Server`
  启动本地 fake robot server

如果你只想在 PyCharm 里点运行，不想管 PowerShell，优先就用这 3 个。

如果你以后直接打开的是 `brain_code` 子目录，也已经同步带上了同名运行配置。

如果你不想切运行配置，也可以直接在 PyCharm 里右键运行这 3 个文件：

- `run_hybrid_real.py`
- `run_hybrid_sim_gui.py`
- `run_fake_robot_server.py`

## 快速入口

快捷入口现在已经放回各自对应的代码目录，不再集中堆在 `brain_code/` 根目录。

- 环境安装：`scripts/SETUP_HYBRID_ENV.cmd`
- 真机 GUI：`hybrid_controller/START_HYBRID_REAL.cmd`
- 仿真 fake robot：`hybrid_controller/START_HYBRID_SIM.cmd`
- 仿真 GUI：`hybrid_controller/START_HYBRID_SIM_GUI.cmd`
- SSVEP async 校验界面：`02_SSVEP/2026-04_async_fbcca_idle_decoder/START_ASYNC_SSVEP_VALIDATION_UI.cmd`
- 05 视觉识别主程序：`05_Vision_Block_Recognition/2026-03_yolo_camera_detection/START_BLOCK_SSVEP_VIEWER.cmd`
- 06 数据采集器：`06_Data_Collection/2026-04_jetmax_block_dataset_collection/START_BLOCK_DATASET_COLLECTOR.cmd`

更详细的步骤见：

- `START_HERE.md`
- `docs/JetMax_真机联动说明.md`
- `docs/JetMax_圆柱坐标改造说明.md`
- `docs/Hybrid_Controller_启动与部署说明.md`

## 当前主线范围

这套 `hybrid_controller` 已经包含：

- 统一状态机与事件层
- JetMax TCP 执行协议
- fake robot / fake scene 联调
- 固定槽位抓取测试
- JetMax 真机执行端
- 圆柱坐标控制接口
- 自动安全高度 `z_auto(r)` 逻辑

关于“为什么改成圆柱坐标、改到了哪一层、保留了哪些旧接口、和 JetMax 官方/文献方案有什么区别”，看：

- `docs/JetMax_圆柱坐标改造说明.md`

当前控制协议采用“双线路”：

- legacy compatibility：
  `MOVE / PICK / PICK_WORLD / PLACE`
- preferred cylindrical path：
  `MOVE_CYL / MOVE_CYL_AUTO / PICK_CYL`

也就是：

- 旧笛卡尔指令继续保留，不会被删
- 但新的主路径和后续扩展，默认都优先走圆柱坐标线路

当前默认仍然是：

- `move_source=sim`
- `decision_source=sim`

也就是先用键盘把总控和机械臂控制链调稳，再逐步接 MI / SSVEP。

## 目录说明

- `hybrid_controller/`
  统一总控代码。包含 UI、状态机、适配层、调试层、JetMax 运行时。
- `scripts/`
  PowerShell / Python / Shell 启动脚本和部署脚本。
- `tests/`
  当前项目测试入口。包含 hybrid controller 和 async FBCCA 相关测试。
- `docs/`
  启动部署、接口协议、状态机、开发记录。
- `01_MI/` `02_SSVEP/` `03_RobotArm_Control/` `05_Vision_Block_Recognition/`
  保留原始研发分支代码，作为参考和后续集成来源。

## 当前推荐启动方式

### 电脑端

```powershell
cd C:\Users\P1233\Desktop\brain\brain_code
.\scripts\run_desktop_fixed_world_real.ps1 -RobotHost 192.168.149.1 -VisionMode fixed_cyl_slots
```

### JetMax 端

现在机械臂端最小运行集已经单独归档到：

- `jetmax_robot/`

最推荐的放法是直接把整个目录复制到 JetMax：

- 本地：`brain_code/jetmax_robot/`
- JetMax：`~/brain_code/jetmax_robot/`

然后在 JetMax 上运行：

```bash
cd ~/brain_code/jetmax_robot
python3 -m pip install -r requirements-jetmax-robot-python.txt
bash run_jetmax_robot_runtime.sh
```

更详细的真机联动顺序见：

- `docs/JetMax_真机联动说明.md`

## 测试

在本机统一测试入口：

```powershell
cd C:\Users\P1233\Desktop\brain\brain_code
& C:\Users\P1233\miniconda3\envs\brain-vision\python.exe -m pytest tests -q
```

如果只测 hybrid controller 主线：

```powershell
cd C:\Users\P1233\Desktop\brain\brain_code
& C:\Users\P1233\miniconda3\envs\brain-vision\python.exe -m pytest tests\test_cylindrical_control.py tests\test_robot_runtime.py tests\test_simulation_world.py tests\test_state_machine.py tests\test_robot_protocol.py tests\test_event_routing.py -q
```
