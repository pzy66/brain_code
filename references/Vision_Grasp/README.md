# Vision Grasp References

This collection records the methods used by the JetMax wood-block recognition
and grasp-flow implementation in `hybrid_controller`.

## Method Alignment

- Ultralytics YOLO instance segmentation: keep YOLO segmentation as the primary
  detector because masks expose object shape, not only bounding boxes. The code
  uses `result.masks.data` when available, then falls back to bbox geometry and
  finally to color/shape geometry for debug continuity.
- Ultralytics predict arguments: retain explicit confidence, IoU, image size,
  device, half precision, and `max_det` configuration so live inference remains
  reproducible and latency can be tuned without code edits.
- Chaumette and Hutchinson visual servo control: keep visual feedback in the
  loop by using image/servo error to command incremental robot motion before a
  final PICK.
- Hiwonder JetMax object tracking examples: keep image updates in a small queue
  and use feedback control against the target center instead of issuing only
  one-shot open-loop moves; the operator or task state still supplies the
  explicit target-selection intent.
- Low-height confirm alignment: treat the strict 2 px threshold as a final
  settled measurement, not as an in-motion trigger. Debug and calibration tools
  therefore use stop-settle-measure: wait for robot IDLE, wait a settle delay,
  drain transition MJPEG frames, then measure repeated `geometry_subpixel`
  points from the persistent official stream.
- OpenCV contour moments and rotated rectangles: for the strict low-height
  2 px confirm threshold, keep floating-point centers from `cv2.moments()` and
  `cv2.minAreaRect()` through the PC-side servo path. Rounded integer centers
  are still emitted for display compatibility, but they are too coarse for
  final centering.
- MoveIt Servo and ros2_control joint trajectory controller documentation:
  use them only as interface-level references for streamed velocity commands,
  stale-command handling, bounded motion, and explicit stopping. This project
  still uses its own lightweight ROS teleop topic and JetMax runtime; it does
  not import MoveIt or ros2_control.
- Uncalibrated/model-free visual servoing work: for low-height alignment, do
  not trust the old global eye-in-hand model when live measurements show
  backlash or stage-dependent response. The PC-side calibration tool now
  measures a local image Jacobian around the confirm height using small,
  bounded exploratory moves, then writes that as `stage_models.confirm` so the
  final 2 px centering threshold is based on local response instead of guesswork.
- ROS `web_video_server` multipart MJPEG implementation: read the official
  JetMax stream as a multipart response and prefer each frame's
  `Content-Length` over scanning the raw byte stream for JPEG markers. This
  matches the browser `stream_viewer` path and avoids cross-frame splices in
  the PC-side debug/runtime reader without touching the robot-side sender.
  Debug reports record Content-Length payload counts, rejected-frame counts,
  buffer resets, reopen counts, and frame age so stream problems can be
  separated from motion-window sampling problems.
- GG-CNN closed-loop grasping: prefer feedback-driven correction and stable
  grasp quality over one-shot open-loop detection. The current main program
  defaults to continuous visual servo after an explicit pick intent; automatic
  vision alone never starts motion or suction, and the older discrete
  stop-and-go path is retained as a fallback and comparison mode.
- Dex-Net 3.0 suction grasping: treat suction placement as a local seal-quality
  problem. The current mask geometry searches for an internal, bright/top-face
  suction point and records grasp quality for later correction.
- SAHI small-object detection: keep slicing as a future option only if replay
  shows missed small blocks caused by input resolution. It is not enabled by
  default because it can add latency and duplicate-merging complexity.

## References

- Ultralytics. "Instance Segmentation - Ultralytics YOLO Docs." https://docs.ultralytics.com/tasks/segment/
- Ultralytics. "Model Prediction with Ultralytics YOLO." https://docs.ultralytics.com/modes/predict/
- Hiwonder. "JetMax | AI Vision Robotic Arm Powered by Jetson Nano." https://www.hiwonder.com/products/jetmax
- Hiwonder. "JetMax v1.0 documentation - AI Vision Games Lesson, Object Tracking." https://wiki.hiwonder.com/projects/JetMax/en/latest/docs/3_AI_Vision_Games_Lesson.html
- Hutchinson, S., Hager, G. D., and Corke, P. I. "A tutorial on visual servo control." IEEE Transactions on Robotics and Automation, 12(5), 651-670, 1996. https://doi.org/10.1109/70.538972
- Chaumette, F., and Hutchinson, S. "Visual servo control. I. Basic approaches." IEEE Robotics and Automation Magazine, 13(4), 82-90, 2006. https://doi.org/10.1109/MRA.2006.250573
- MoveIt. "Realtime Servo." https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html
- ros2_control. "joint_trajectory_controller user documentation." https://control.ros.org/rolling/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html
- OpenCV. "Contour Features." https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
- Musić, J., Bonković, M., and Cecić, M. "Comparison of Uncalibrated Model-Free Visual Servoing Methods for Small-Amplitude Movements: A Simulation Study." International Journal of Advanced Robotic Systems, 2014. https://doi.org/10.5772/58822
- Piepmeier, J. A., and Lipkin, H. "Uncalibrated Eye-in-Hand Visual Servoing." International Journal of Robotics Research, 22(10-11), 805-819, 2003.
- Shademan, A., Farahmand, A.-M., and Jägersand, M. "Robust Jacobian Estimation for Uncalibrated Visual Servoing." ICRA 2010.
- ROS `web_video_server`. "multipart_stream.cpp source." https://docs.ros.org/en/kinetic/api/web_video_server/html/multipart__stream_8cpp_source.html
- ROS `web_video_server`. "jpeg_streamers.cpp source." https://docs.ros.org/hydro/api/web_video_server/html/jpeg__streamers_8cpp_source.html
- Morrison, D., Corke, P., and Leitner, J. "Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach." RSS 2018. https://arxiv.org/abs/1804.05172
- Mahler, J., Matl, M., Liu, X., Li, A., Gealy, D., and Goldberg, K. "Dex-Net 3.0: Computing Robust Robot Vacuum Suction Grasp Targets in Point Clouds using a New Analytic Model and Deep Learning." ICRA 2018. https://arxiv.org/abs/1709.06670
- Akyon, F. C., Altinuc, S. O., and Temizel, A. "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection." ICIP 2022. https://arxiv.org/abs/2202.06934
