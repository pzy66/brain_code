# hybrid_controller_ros

ROS-first transport package for JetMax teleop smoothing.

This package provides:

- `/hybrid_controller/teleop_cyl_cmd` topic for continuous cylindrical teleop input
- `/hybrid_controller/state` topic for continuous robot state streaming
- ROS services for `move_cyl`, `move_cyl_auto`, `pick_cyl`, `pick_world`, `place`, `abort`, `reset`,
  `get_pick_tuning`, and `set_pick_tuning`

The node is designed to run on JetMax after the package is copied into a catkin workspace and built.

## Communication semantics (current)

- Teleop command topic now carries `cmd_seq` and `client_ts` for stale/out-of-order rejection.
- `CylindricalTeleop` carries three velocity axes:
  `theta_rate_deg_s`, `radius_rate_mm_s`, and `z_rate_mm_s`.
- `use_auto_z=true` preserves legacy keyboard teleop behavior where z follows the radius auto-z curve.
  Continuous visual servo must publish `use_auto_z=false` so centering motion does not change height unless
  `z_rate_mm_s` is explicitly nonzero.
- State topic now carries `state_seq`, `robot_ts`, and `last_ack`.
- Service calls (`move_cyl`, `move_cyl_auto`, `pick_world`, `pick_cyl`, `place`) are
  accepted quickly and queued/executed asynchronously; action completion is confirmed by
  state stream (`last_ack` + state/busy fields), not by long blocking service waits.
- `abort` / `reset` preempt by clearing queued actions before execution.

## rosbridge tuning

`run_hybrid_controller_ros_runtime.sh` starts rosbridge with explicit networking options:

- `websocket_ping_interval`
- `websocket_ping_timeout`
- `retry_startup_delay`
- `use_compression` (default `false` for LAN)

Recommended runtime chain:

1. Desktop GUI publishes `/hybrid_controller/teleop_cyl_cmd` through rosbridge
2. JetMax subscribes locally at `20Hz`
3. JetMax computes cylindrical teleop smoothing locally
4. JetMax publishes `/hybrid_controller/state`

The preferred JetMax startup script is:

- `brain_code/hybrid_controller/robot/run_hybrid_controller_ros_runtime.sh`

TCP remains available as a legacy fallback, but GUI teleop should use this ROS path by default.

After changing `CylindricalTeleop.msg`, rebuild and redeploy this catkin package on JetMax before running
desktop continuous-servo code. PC and robot message definitions must stay in lockstep.
