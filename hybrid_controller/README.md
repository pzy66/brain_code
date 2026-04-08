# Hybrid Controller

这是当前统一总控主程序包。

## 目标

这套代码负责把以下内容收敛到同一条主线里：

- 任务状态机
- JetMax 机械臂执行协议
- fake robot / fake scene 联调
- 固定槽位抓取测试
- 真机 JetMax 执行端
- 圆柱坐标控制接口

## 包结构

- `app.py`
  Qt 总装配层。负责把 UI、控制器、robot client、仿真世界和日志串起来。
- `config.py`
  当前运行配置。
- `cylindrical.py`
  `theta / r / z` 与 `x / y / z` 的转换和 `z_auto(r)` profile 生成。
- `controller/`
  纯状态机、事件、上下文。
- `adapters/`
  输入适配层、robot client、固定槽位定义、vision target 结构。
- `debug/`
  fake robot、fake scene、日志和回放。
- `integrations/`
  真机 JetMax 运行时，以及后续 MI / SSVEP / vision 桥接层。
- `ui/`
  PyQt5 主窗口。

## 当前推荐控制模式

默认移动逻辑已经切到圆柱坐标主语义：

- `theta_deg`
  底座方向
  默认理论范围现为 `[-120°, 120°]`，但真实可用角度仍会被当前 `r/z`、`x/y` 工作区和 IK 安全校验进一步裁剪。
- `radius_mm`
  水平半径
- `z_mm`
  高度

其中：

- `MOVE_CYL theta r z`
  显式圆柱坐标移动
- `MOVE_CYL_AUTO theta r`
  自动安全高度模式
- `PICK_CYL theta r`
  圆柱抓取点

当前执行端是双线路：

- `legacy_cartesian`
  保留 `MOVE / PICK / PICK_WORLD / PLACE`
- `cylindrical_kernel`
  优先服务 `MOVE_CYL / MOVE_CYL_AUTO / PICK_CYL`

底层执行仍然统一落到 JetMax 官方 `set_position(x, y, z, t)`，没有直接改 Hiwonder 官方包或 ROS 原生消息定义。

## 真机相关文件

JetMax 上实际运行的是：

- `integrations/robot_runtime_py36.py`

桌面端用于真机通信的是：

- `adapters/robot_client.py`

## 快捷入口

这个目录里现在保留统一总控相关的双击入口：

- `START_HYBRID_REAL.cmd`
- `START_HYBRID_SIM.cmd`
- `START_HYBRID_SIM_GUI.cmd`

它们会调用 `../scripts/` 里的 PowerShell 启动脚本。

## 开发约定

- 电脑端统一从 `brain_code/` 根目录运行
- 真机脚本和部署脚本也都从 `brain_code/scripts/` 调用
- 测试统一放在 `brain_code/tests/`
- 不再维护外层根目录的旧 `hybrid_controller/` 副本作为主线
