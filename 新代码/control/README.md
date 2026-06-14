# JetMax brain control extract

This folder is the extracted JetMax arm control bundle. It keeps only the code needed for the robot-side workflow:

- camera video display: `hybrid_controller/ui/vision_feed_widget.py`
- block detection and slot/frequency overlay: `hybrid_controller/vision/*`, `hybrid_controller/ui/main_window.py`
- camera centering and continuous visual servo: `hybrid_controller/vision/servo_controller.py`, `hybrid_controller/vision/continuous_servo_controller.py`
- keyboard movement and manual confirm flow: `hybrid_controller/run_real.py`, `hybrid_controller/app.py`
- final automatic pick/place and suction control: `hybrid_controller/app_robot_commands.py`, `hybrid_controller/robot/*`
- JetMax ROS package and services: `hybrid_controller/robot/ros_pkg/hybrid_controller_ros`

Run from this directory with the allowed environment:

```powershell
conda activate brain_robot
python .\run_jetmax_robot.py
```

Useful options:

```powershell
python .\run_jetmax_robot.py --vision-continuous-servo-stop-at-confirm
python .\hybrid_controller\robot\tools\ros_service_probe.py --host 192.168.149.1 --port 9091 --action status
python .\hybrid_controller\tools\debug_vision_grasp_flow.py --slot-id 1 --low-height-centering-check
```

The vision model is included at `datasets/vision/models/best.pt`. The current grasp profile is included at `datasets/profiles/hybrid_controller/vision_grasp/current_grasp_profile.json`; the calibration profile directory is prepared at `datasets/vision/calibration/` for the next step.
