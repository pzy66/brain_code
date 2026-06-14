# Codex 新电脑联调交接说明

## 当前目标

在新电脑上联调 `BrainRobotWorkbench.exe`，连接 JetMax/Hiwonder 机械臂和脑电帽，展示完整脑机机械臂流程。表面流程按脑电控制展示；当前演示识别可以由电脑端输入事件代替，但 UI 不显示键盘提示。机械臂控制、摄像头画面、视觉识别和抓取链路需要能真实联调。

## 重要产物

- 便携 exe：`dist/BrainRobotWorkbench.zip`
- 可调试源码：当前 `brain_code` 仓库
- 新电脑说明：`handoff/new_pc_joint_debug/NEW_PC_QUICK_START.md`
- 诊断脚本：`handoff/new_pc_joint_debug/scripts/diagnose_new_pc.ps1`

## 默认硬件参数

- 机械臂 IP：`192.168.149.1`
- SSH：`hiwonder / hiwonder`
- 机械臂端代码目录：`/home/hiwonder/brain_code`
- 上位机控制主链：ROS rosbridge
- rosbridge 端口：`9091`
- legacy TCP 端口：`8888`，非主线
- 摄像头流：`http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80`
- 脑电串口：默认 `auto`

## 机械臂端需要运行什么

在机械臂上运行：

```bash
cd /home/hiwonder/brain_code/hybrid_controller/robot
bash run_hybrid_controller_ros_runtime.sh
```

这个脚本会：

1. 同步 `ros_pkg/hybrid_controller_ros` 到 `~/catkin_ws/src`。
2. 必要时执行 `catkin_make --pkg hybrid_controller_ros`。
3. 启动或复用 `rosbridge_websocket`，默认 `9091`。
4. 运行 `ros_pkg/hybrid_controller_ros/scripts/hybrid_controller_runtime_node.py`。
5. 保持 JetMax 官方摄像头链路，不默认改写或重启 `usb_cam.service`。

## 新电脑 Codex 调试建议

1. 将 Codex 工作目录切到复制后的 `brain_code` 源码根目录。
2. 先阅读 `handoff/new_pc_joint_debug/README.md`。
3. 先运行静态检查：

```powershell
python -m compileall -q robot_workbench hybrid_controller run_integrated_workbench.py packaging
```

如果新电脑没有 Python 环境，先只调试 exe 和网络；源码环境可后续再配。

4. 先用诊断脚本检查硬件链路：

```powershell
.\handoff\new_pc_joint_debug\scripts\diagnose_new_pc.ps1
```

5. 如果 `9091` 不通，但 `22` 通，手动 SSH 到机械臂运行主脚本。

## 当前代码状态提示

这个源码快照包含当前工作区的未提交修改和新文件，不能假设它等同于 GitHub 上的 `main`。与 exe 相关的关键目录包括：

- `robot_workbench/`
- `hybrid_controller/`
- `hybrid_controller/robot/`
- `datasets/vision/models/best.pt`
- `datasets/vision/calibration/current_profile.json`
- `datasets/profiles/hybrid_controller/`
- `packaging/`
- `tools/build_integrated_workbench.ps1`
- `新代码/control`
- `新代码/UI`

## 不要误操作

- 不要只复制 `BrainRobotWorkbench.exe`，必须保留 `_internal`。
- 不要把 `9092` 当主链端口；当前主链是 `9091`。
- 不要优先运行 `run_jetmax_robot_runtime.sh`，它是 legacy TCP 路径。
- 不要默认修改 `/home/hiwonder/ros/autostart/usb_cam.launch`。
- 不要默认重启 `usb_cam.service` 或接管 `web_video_server`。
