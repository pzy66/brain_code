# 当前状态快照

## 目标

最终目标是拿到一个容易迁移的新电脑联调版本：

- 普通使用者可以直接运行 `dist/BrainRobotWorkbench.zip` 解压后的 exe。
- Codex 可以读取当前仓库和本文件夹，继续定位新电脑上的网络、串口、摄像头、机械臂控制和 UI 问题。

## 当前已生成的 exe

```text
dist/BrainRobotWorkbench.zip
dist/BrainRobotWorkbench/BrainRobotWorkbench.exe
```

注意：不要只复制 `BrainRobotWorkbench.exe`，必须保留同级 `_internal` 文件夹。

注意：`dist/BrainRobotWorkbench.zip` 不进入普通 GitHub 源码提交；从 GitHub clone 到新电脑时，需要单独复制这个 zip，或在新电脑重新打包。

## 已验证过的内容

- 源码 smoke test 通过。
- PyInstaller onedir 打包成功。
- `dist/BrainRobotWorkbench.zip` 解压到仓库外目录后，exe smoke test 退出码为 `0`。
- 打包中已包含视觉模型、视觉标定、SSVEP profile、机械臂抓取参数和机械臂运行脚本。

## 当前主要改动范围

这些文件/目录和新电脑联调最相关：

```text
robot_workbench/
run_integrated_workbench.py
tools/build_integrated_workbench.ps1
packaging/
hybrid_controller/
hybrid_controller/robot/
datasets/vision/models/best.pt
datasets/vision/calibration/current_profile.json
datasets/profiles/hybrid_controller/
START_INTEGRATED_WORKBENCH.cmd
START_UI_DEMO.cmd
新代码/control
新代码/UI
```

## 机械臂主链

当前主链是 ROS + rosbridge：

```text
Windows exe -> ws://192.168.149.1:9091 -> rosbridge -> hybrid_controller_runtime_node.py -> JetMax runtime
```

摄像头链路是官方 web video：

```text
usb_cam.service -> /usb_cam/image_rect_color -> web_video_server:8080 -> Windows exe
```

不要把 `9092` 当主链端口。`8888` 是 legacy TCP 兼容/诊断端口。

## 演示流程约定

UI 表面展示为脑电控制流程。当前识别可以用电脑端输入事件代替，但 UI 不显示键盘提示。

目标展示流程：

1. MI 阶段约 20 秒，展示移动控制。
2. 进入 SSVEP 目标选择，小木块闪烁。
3. 选择目标后停止闪烁。
4. 弹出确认抓取/重新选择。
5. 确认后机械臂抓取。
6. 回到 MI 移动阶段。
7. 到放置确认阶段，选择放下或继续移动。

## 新电脑需要重点验证

1. `BrainRobotWorkbench.exe` 是否能启动。
2. 电脑是否能连上机械臂 Wi-Fi。
3. `192.168.149.1:22` SSH 是否通。
4. `192.168.149.1:9091` rosbridge 是否通。
5. `192.168.149.1:8080` 摄像头是否通。
6. 脑电帽 COM 口是否能看到。
7. 软件内 EEG 波形是否刷新。
8. 视觉识别是否显示最多 4 个目标。
9. 选中目标后，确认抓取是否真的发出抓取动作。
