# JetMax 圆柱坐标改造说明

## 为什么改

JetMax 官方控制主路径是笛卡尔坐标：

- `set_position((x, y, z), t)`

这对机器执行是合适的，但对上层控制不直观。  
当前主线改成了以圆柱坐标为主语义：

- `theta`
- `radius`
- `z`

原因：
- 更符合“左转 / 右转 / 前伸 / 后收 / 抬高”的直觉
- 更适合 teleop 和后续 SSVEP/视觉驱动
- 更容易表达工作空间和安全边界

## 当前实现方式

注意：这不是修改 Hiwonder 官方底层驱动协议。  
当前做法是：

1. 主程序外部接口以圆柱坐标为主
2. 运行时内部也以圆柱语义维护状态和限制
3. 最终调用 JetMax 官方能力时，仍转换为 `x/y/z`

也就是说：
- 主线控制逻辑：圆柱坐标
- 官方底层驱动：仍是 `x/y/z + IK + servo`

## 当前方向定义

- `theta = 0°`：机械臂正前方
- `theta > 0`：机械臂自身左侧
- `theta < 0`：机械臂自身右侧

## 当前范围

默认主线范围：
- `theta ∈ [-120°, 120°]`
- `MOVE_CYL_AUTO radius ∈ [80, 260]`
- `MOVE_CYL radius ∈ [50, 280]`

## 自动高度策略

`MOVE_CYL_AUTO` 不是固定 `z`，而是自动生成 `z_auto(r)`。

当前策略：
- 优先让前臂尽量保持水平
- 其次保证安全
- 中段尽量维持稳定高度
- 前后两端允许按需要调整高度

这套策略是根据实机反复调出来的，目标是让前后移动更符合操作直觉。

## 当前代码位置

- 圆柱换算：
  - [cylindrical.py](C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\cylindrical.py)
- 机器人运行时：
  - [robot/runtime/robot_runtime.py](C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\runtime\robot_runtime.py)
  - [robot/runtime/robot_runtime_py36.py](C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\runtime\robot_runtime_py36.py)
- ROS teleop kernel：
  - [robot/runtime/teleop_kernel.py](C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\runtime\teleop_kernel.py)
