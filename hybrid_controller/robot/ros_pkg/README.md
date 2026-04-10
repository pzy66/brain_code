# ros_pkg

这里放的是 JetMax 真机主链需要的内置 ROS package。

当前唯一包：

- [hybrid_controller_ros](C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\ros_pkg\hybrid_controller_ros)

它负责提供：
- `/hybrid_controller/teleop_cyl_cmd`
- `/hybrid_controller/state`
- `/hybrid_controller/reset`
- `/hybrid_controller/abort`
- `/hybrid_controller/move_cyl`
- `/hybrid_controller/move_cyl_auto`
- `/hybrid_controller/pick_cyl`
- `/hybrid_controller/place`

JetMax 端启动脚本会从这里把包复制到 `~/catkin_ws/src/` 再编译。  
所以主程序不再依赖外面的 `brain_code/hybrid_controller_ros/` 旧副本。
