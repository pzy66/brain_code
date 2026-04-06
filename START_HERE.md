# Start Here

如果你要继续开发这个仓库，建议从这里开始。

## 推荐阅读顺序

1. `docs/主线入口与代码地图.md`
2. `docs/系统架构与主线开发图.md`
3. `docs/系统现状与开发建议报告_2026-04-06.md`

## 当前主线入口

- 总控原型：
  `05_Vision_Block_Recognition/2026-03_yolo_camera_detection/computer/test2.py`
- 机械臂执行端：
  `03_RobotArm_Control/2026-03_jetmax_execution_server/test2_robot.py`
- SSVEP 最新实时识别：
  `02_SSVEP/2026-03_realtime_ui_and_online_decoder/SSVEP/demo.py`
- MI 最新实时识别：
  `01_MI/mi_classifier_latest/code/realtime/mi_realtime_infer_only.py`

## 当前最重要的判断

这个仓库现在最缺的不是新的单独算法脚本，而是统一任务控制层。

也就是说，后续开发最应该优先做的是：

1. 统一总控状态机
2. 统一机械臂通信协议
3. 把真实 SSVEP 接进总控
4. 再补强并接入真实 MI

## 不建议直接作为主线继续扩展的内容

- 单字符 socket 控制脚本
- 早期单独联调脚本
- 依赖键盘和鼠标模拟脑控输入的临时代码

这些代码可以保留参考，但不建议继续作为未来主线入口。
