# JetMax 真机联动说明

## 联动目标

当前主线是：

- JetMax 端运行 ROS runtime
- 电脑端运行 `hybrid_controller` 主界面
- 通过 rosbridge + ROS service 完成控制

## JetMax 端放置目录

- 复制源：`C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller`
- JetMax 目标：`/home/hiwonder/brain_code/hybrid_controller`

## JetMax 端启动

```bash
cd ~/brain_code/hybrid_controller/robot
python3 -m pip install -r requirements-jetmax-robot-python.txt
bash run_hybrid_controller_ros_runtime.sh
```

## 电脑端启动

```bash
cd C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller
python run_real.py
# 或
python run_real_ssvep.py
```

## 主链端口

- `9091`：ROS bridge（必须，默认主链端口）
- `8080`：视频流（必须）
- `8888`：TCP 兼容链路（可选）

## 抓取命令决策（关键）

在 `vision_mode=robot_camera_detection` 下：

- `command_mode=world` -> 下发 `PICK_WORLD x y`（默认视觉主路径）
- `command_mode=cyl` -> 下发 `PICK_CYL theta r`（手动/特定路径）
- `actionable=false` -> 拒绝抓取，不发命令

## 联调顺序（建议）

1. 启动 JetMax 端 runtime
2. 启动电脑端 GUI
3. 点击“连接机器人”，确认 `robot_connected=True`
4. 先测移动（`a/d/w/s`）
5. 再测视觉槽位
6. 最后测抓放（`PICK_WORLD -> PLACE`）
