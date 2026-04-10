# hybrid_controller_ros

ROS-first transport package for JetMax teleop smoothing.

This package provides:

- `/hybrid_controller/teleop_cyl_cmd` topic for continuous cylindrical teleop input
- `/hybrid_controller/state` topic for continuous robot state streaming
- ROS services for `move_cyl`, `move_cyl_auto`, `pick_cyl`, `pick_world`, `place`, `abort`, `reset`,
  `get_pick_tuning`, and `set_pick_tuning`

The node is designed to run on JetMax after the package is copied into a catkin workspace and built.

Recommended runtime chain:

1. Desktop GUI publishes `/hybrid_controller/teleop_cyl_cmd` through rosbridge
2. JetMax subscribes locally at `20Hz`
3. JetMax computes cylindrical teleop smoothing locally
4. JetMax publishes `/hybrid_controller/state`

The preferred JetMax startup script is:

- `brain_code/hybrid_controller/robot/run_hybrid_controller_ros_runtime.sh`

TCP remains available as a legacy fallback, but GUI teleop should use this ROS path by default.
