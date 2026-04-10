# JetMax 机械臂改动与落盘清单

本文用于回答两个问题：

1. 机械臂相关改动是否已经写清楚。
2. 改动代码是否都保存在 `hybrid_controller` 目录内。

## 结论

- 已完成的机械臂主链改动，代码均在：
  - `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\...`
- 没有依赖外部 `01_MI/02_SSVEP/...` 目录里的机械臂运行时代码。
- ROS 主链抓取可走 `PICK_WORLD`，TCP 仅保留兼容与诊断。

## 本轮关键改动（机械臂相关）

### 1) ROS 增加 `PICK_WORLD` 服务链路

- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\adapters\rosbridge_client.py`
  - 新增 `send_pick_world(x_mm, y_mm)`
- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\app.py`
  - ROS 模式命令路由支持 `PICK_WORLD`
  - `robot_camera_detection` 模式下，手动 pick 不再回退 `slot_catalog` 兜底
- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\ros_pkg\hybrid_controller_ros\srv\PickWorld.srv`
  - 新增 ROS service 定义
- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\ros_pkg\hybrid_controller_ros\scripts\hybrid_controller_runtime_node.py`
  - 运行时节点注册 `pick_world` 服务
- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\ros_pkg\hybrid_controller_ros\CMakeLists.txt`
  - 加入 `PickWorld.srv` 生成配置

### 2) ERROR 态网关门控修正（Py36 runtime）

- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\runtime\robot_runtime_py36.py`
  - `ERROR` 状态下仅允许：`PING/STATUS/ABORT/RESET`

### 3) 启动脚本行为修正（ROS 主链优先）

- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\tools\jetmax_start_ros_runtime.py`
  - 默认只校验 ROS 主链端口可用（`9091` + `8080`）
  - 新增 `--require-tcp-check` 用于可选强制检查 `8888`

### 4) 诊断工具补齐

- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\tools\ros_service_probe.py`
  - 可直接测试：`status / move_cyl / move_cyl_auto / pick_world / pick_cyl / place / abort / reset`

### 5) 可观测性增强

- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\app.py`
  - 新增抓取 trace 字段：`mapping_mode / resolved_base_xy / resolved_cyl / snapshot_age_ms / robot_pose / response`
  - 视觉状态文本增加 `resolved_xy / resolved_cyl`

### 6) 抓取调参与姿态收敛（本轮新增）

- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\runtime\robot_runtime.py`
  - 抓后抬升改为 `z_settle=max(z_carry_floor, z_auto(r))`
  - 放置释放支持 `release` 优先 + `off` 回退
  - 新增 `get_pick_tuning()/set_pick_tuning()`
- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\runtime\robot_runtime_py36.py`
  - 同步实现上述抓放调参与 `z_settle` 行为
  - `snapshot/STATUS` 新增 `pick_tuning/post_pick_settle_z/release_mode_effective`
- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\ros_pkg\hybrid_controller_ros\srv\GetPickTuning.srv`
- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\ros_pkg\hybrid_controller_ros\srv\SetPickTuning.srv`
- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\ros_pkg\hybrid_controller_ros\msg\RobotState.msg`
  - 增加调参字段与 `post_pick_settle_z/release_mode_effective`
- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\adapters\rosbridge_client.py`
  - 新增 `get_pick_tuning/set_pick_tuning` ROS 调用
- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\app.py`
  - 右侧 `Pick Tuning` 微调入口接入 ROS 生效与本地保存
  - `PICK_WORLD` 也应用 `r/theta` 偏置（不再只影响 `PICK_CYL`）
- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\ui\main_window.py`
  - 新增调参按钮与 `应用/恢复默认/保存配置` 操作

## 启动与验证建议（最小闭环）

1. JetMax 端启动：
   - `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\run_hybrid_controller_ros_runtime.sh`
2. 桌面端一键拉起（可选）：
   - `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\tools\jetmax_start_ros_runtime.py`
3. 服务探针验证：
   - `status -> pick_world -> place`
4. GUI 验证：
   - `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\run_real_ssvep.py`

## 关联测试文件

- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\tests\test_pick_world_ros_routing.py`
- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\tests\test_robot_runtime_py36_gateway.py`
- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\tests\test_robot_runtime.py`

以上测试用于保证：

- ROS 模式 `PICK_WORLD` 走 ROS，不静默回退 TCP
- `robot_camera_detection` 下手动 pick 不回退 `slot_catalog`
- Py36 网关 ERROR 态门控规则正确
