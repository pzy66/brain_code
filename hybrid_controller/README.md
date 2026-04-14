# Hybrid Controller（真机主线）

本目录是当前唯一主线：JetMax 真机控制 + 视觉 + SSVEP UI。  
主链路为 ROS（`rosbridge`），TCP 仅保留兼容/诊断用途。

---

## 1. 目录定位

- 主入口（电脑端）：
  - `run_real.py`（决策源默认 `sim`）
  - `run_real_ssvep.py`（决策源默认 `ssvep`）
- 机械臂端启动脚本：
  - `robot/run_hybrid_controller_ros_runtime.sh`（推荐，ROS 主链）
  - `robot/run_jetmax_robot_runtime.sh`（legacy，TCP only）
- 电脑端远程启动机械臂工具：
  - `robot/tools/jetmax_start_ros_runtime.py`
- ROS 服务探针：
  - `robot/tools/ros_service_probe.py`
- 详细文档入口：
  - `docs/README.md`

---

## 2. 解释器与环境（必须先确认）

本项目按单解释器收口，默认使用：

`C:\Users\P1233\miniconda3\envs\brain-vision\python.exe`

`run_real.py` 和 `run_real_ssvep.py` 含解释器守卫，若不是 `brain-vision` 会直接退出并提示 `Interpreter mismatch`。

### PyCharm 推荐设置

1. 从仓库根目录打开项目：`C:\Users\P1233\Desktop\brain`
2. `Settings -> Python Interpreter` 选择已有 Conda 环境 `brain-vision`
3. 确认解释器路径是：
   - `C:\Users\P1233\miniconda3\envs\brain-vision\python.exe`
4. 删除或禁用旧的 `.venv` 运行配置，避免误用旧路径

---

## 3. 一键启动（推荐流程）

### 步骤 1：机械臂上电并接入同一局域网

- 机械臂与电脑需互通（示例 IP：`192.168.149.1`）。

### 步骤 2：电脑端远程拉起机械臂 ROS runtime

```powershell
C:\Users\P1233\miniconda3\envs\brain-vision\python.exe C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\tools\jetmax_start_ros_runtime.py --host 192.168.149.1 --user hiwonder --password hiwonder --remote-root /home/hiwonder/brain_code
```

### 步骤 3：启动 GUI（SSVEP 版本）

```powershell
C:\Users\P1233\miniconda3\envs\brain-vision\python.exe C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\run_real_ssvep.py
```

如果只做键盘/视觉调试，不用 SSVEP 识别，可运行：

```powershell
C:\Users\P1233\miniconda3\envs\brain-vision\python.exe C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\run_real.py
```

---

## 4. 手工启动机械臂（在 JetMax 板端执行）

仅在远程脚本不可用时使用。

```bash
cd ~/brain_code/hybrid_controller/robot
python3 -m pip install -r requirements-jetmax-robot-python.txt
bash run_hybrid_controller_ros_runtime.sh
```

该脚本会启动：

- `rosbridge`（默认端口 `9091`）
- `web_video_server`（默认端口 `8080`）
- `hybrid_controller_runtime_node.py`

---

## 5. GUI 关键操作

### 5.1 机械臂控制

- 按住 `W/A/S/D` 连续移动（机器人自身坐标系）：
  - `A`：向左转（`theta` 减小）
  - `D`：向右转（`theta` 增大）
  - `W`：前伸（`radius` 增大）
  - `S`：后收（`radius` 减小）
- 方向键与 `W/A/S/D` 等价。
- `Abort`：立即中止并回收动作。
- `Reset`：从错误态恢复。

注意：

- 主窗口必须有焦点，否则键盘事件不会进入程序。
- 当前状态在 `PICKING / PLACING / ERROR` 时会门控移动指令。

### 5.2 目标与决策（键盘调试）

- `1/2/3/4`：选中槽位目标
- `Enter` 或 `C`：确认
- `Esc` 或 `X`：取消
- `N`：开始任务
- `R`：重置任务状态机

### 5.3 SSVEP 两个开关（独立）

- `开启/关闭SSVEP刺激`：只控制视觉闪烁层。
- `开启/关闭SSVEP识别`：只控制在线识别线程。

补充行为：

- 抓取完成（`PICK_DONE`）后，若刺激还开着，会自动关闭刺激。
- 停止识别后，画面仍保留目标框，只停止闪烁与识别事件流。

---

## 6. SSVEP Profile 说明

- 目录：`dataset/ssvep_profiles/`
- 自动指针：`dataset/ssvep_profiles/current_fbcca_profile.json`
- 预训练产物命名：`ssvep_fbcca_profile_YYYYMMDD_HHMMSS.json`

策略：

- 有 `current_fbcca_profile.json` 时默认优先加载它。
- 没有 current profile 时可 fallback 启动（界面会标注 fallback 来源）。

---

## 7. 视觉与抓取链路（当前语义）

- 视觉主模式：`robot_camera_detection`
- 视觉抓取默认走 `PICK_WORLD x y`（ROS 主链）
- 圆柱坐标作为主显示/主操控语义，抓取执行统一解析到基座世界坐标
- 当前映射模式默认：`delta_servo`

抓取调试入口：

- `Pick r bias`、`Pick theta bias`（步进 ±1）
- 抓放时序/深度调参区（应用/恢复默认/保存配置）

---

## 8. 连通性与健康检查

### 8.1 ROS 服务状态

```powershell
C:\Users\P1233\miniconda3\envs\brain-vision\python.exe C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\tools\ros_service_probe.py --host 192.168.149.1 --port 9091 --action status
```

### 8.2 快速动作探针

```powershell
C:\Users\P1233\miniconda3\envs\brain-vision\python.exe C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\tools\ros_service_probe.py --host 192.168.149.1 --port 9091 --action move_cyl_auto --theta 0 --radius 170
```

---

## 9. 常见问题（先看这里）

### 9.1 WASD 无法控制

按顺序检查：

1. GUI 右侧 `robot connected=True` 且 `health=ok`。
2. `Preflight` 为 `ok=true`。
3. 当前窗口焦点在主程序（不是终端/其他窗口）。
4. `move_source=sim`（当前 MI 仍是占位接口，不会输出真实移动事件）。
5. 状态不在 `PICKING / PLACING / ERROR` 门控阶段。
6. `ros_service_probe --action status` 可连通；若失败先重启机械臂 runtime。

### 9.2 能连上 GUI 但机械臂不动

- 先执行一次 `status` 探针，确认 ROS 服务在线。
- 再执行一次 `move_cyl_auto` 探针，排除 UI 输入问题。
- 若探针也失败，优先重启机械臂端 `run_hybrid_controller_ros_runtime.sh`。

### 9.3 画面有框但不闪烁

- 检查是否关闭了 `SSVEP刺激` 开关（识别可开着，但刺激单独关闭时不会闪）。

---

## 10. 端口与协议

- `9091`：ROS bridge（主链）
- `8080`：相机视频流（`web_video_server`）
- `8888`：TCP legacy 兼容

说明：

- 当前主流程不依赖 `9092`。
- TCP 保留用于兼容和诊断，不作为 GUI 真机主路径。

---

## 11. 开发与清理脚本

- 临时文件清理：
  - `tools/clean_hybrid_temp.ps1`
- 快速检查（编译 + 测试）：
  - `tools/run_hybrid_checks.ps1`

---

## 12. 安全建议（真机调试必读）

- 首次上电或改参数后，先做 `MOVE-only`，确认方向与限位。
- 调整抓取深度/时序时，每次只改一个参数，小步迭代。
- 任何异常先 `Abort`，确认停止后再 `Reset`。
- 不要在未知状态下重复下发抓取命令。

---

## 13. 相关文档

- `docs/README.md`
- `docs/Hybrid_Controller_启动与部署说明.md`
- `docs/JetMax_真机联动说明.md`
- `docs/JetMax_抓取流程确认说明.md`
- `docs/SSVEP_FBCCA_预训练与联调说明.md`
