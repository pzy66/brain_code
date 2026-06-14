# 新电脑 Codex 联调入口

这个文件夹是给新电脑上的 Codex 读取的交接资料。目标是在新电脑上继续联调脑机机械臂一体化控制台，包括 exe 启动、机械臂连接、摄像头画面、脑电串口和展示流程。

## 新电脑 Codex 先读顺序

1. `README.md`
2. `CODEX_HANDOFF.md`
3. `CURRENT_STATE.md`
4. `COPY_CHECKLIST.md`
5. `GITHUB_SYNC.md`
6. `NEW_PC_QUICK_START.md`
7. `TROUBLESHOOTING.md`

## 推荐给新电脑 Codex 的第一句话

```text
请先阅读 handoff/new_pc_joint_debug/README.md 和 CODEX_HANDOFF.md，然后帮我在这台新电脑上联调 BrainRobotWorkbench.exe、机械臂 rosbridge 9091、摄像头 8080、脑电串口和完整展示流程。回答用中文。
```

## 当前仓库里最重要的入口

- 软件源码入口：`run_integrated_workbench.py`
- UI 主体：`robot_workbench/flow_ui.py`
- exe 打包脚本：`tools/build_integrated_workbench.ps1`
- PyInstaller 配置：`packaging/BrainRobotWorkbench.spec`
- 便携 exe 产物：`dist/BrainRobotWorkbench.zip`
- 机械臂端主脚本：`hybrid_controller/robot/run_hybrid_controller_ros_runtime.sh`
- 机械臂端 ROS runtime node：`hybrid_controller/robot/ros_pkg/hybrid_controller_ros/scripts/hybrid_controller_runtime_node.py`
- 视觉权重：`datasets/vision/models/best.pt`
- 视觉标定：`datasets/vision/calibration/current_profile.json`
- SSVEP 默认配置：`datasets/profiles/hybrid_controller/ssvep_profiles/`

## 新电脑最小动作

1. 复制整个仓库到新电脑。
2. 如果是从 GitHub clone，先切到 `codex/integrated-workbench-robot-debug` 分支。
3. 如果有单独复制的 `dist/BrainRobotWorkbench.zip`，解压并运行 `BrainRobotWorkbench/BrainRobotWorkbench.exe`。
4. 如果没有 exe 包，则先从源码启动或在新电脑重新打包。
5. 如果机械臂自动 SSH 启动失败，按 `ROBOT_MANUAL_START.md` 手动启动机械臂端脚本。
6. 运行 `scripts/diagnose_new_pc.ps1` 检查网络、端口和串口。

## 端口和账号

- 机械臂 IP：`192.168.149.1`
- SSH：`hiwonder / hiwonder`
- rosbridge：`9091`
- 摄像头：`8080`
- legacy TCP：`8888`，不是主链
- 脑电串口：默认 `auto`
