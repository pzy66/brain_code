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
  explicit target-selection intent. The robot-side tutorial file inspected on
  the JetMax was
  `/home/hiwonder/ros/src/jetmax_buildin_funcs/object_tracking/scripts/object_tracking_main.py`
  (SHA256 `46157b09344dd29147da9cebfecd9c40d608efbb3f50f7ebd7d2cdce3112c494`).
  Its color tracker uses `hiwonder.PID(0.07, 0.01, 0.0015)` for image x and
  `hiwonder.PID(0.08, 0.008, 0.001)` for image y, then sends short
  `jetmax.set_servo(..., duration=0.02)` updates every processed frame. Its face
  tracker similarly uses image-center PID and short `set_servo` / `set_position`
  updates around `0.08 s`. The relevant lesson for this project is the motion
  topology: newest image, small feedback correction, continuous command stream.
  The color segmentation itself is not used for wood-block recognition.
- Continuous descent with gated visual servo: keep the main debug path close to
  classical image-based visual servoing. The robot should descend while the
  image error is being corrected, not stop after every frame. The implementation
  therefore allows a slow z velocity inside a wider center-error band, while
  still stopping descent on stale frames, target jumps, large error, robot
  errors, or near the final 2 px confirm gate. Inside the low-height guard band,
  z velocity is scaled down together with theta/radius correction so the arm
  keeps moving smoothly but does not overshoot the final stopped-frame check.
- Damped image-Jacobian centering: for live high-to-low alignment tests, prefer
  an image-based visual-servo control law over fixed pixel-axis gains. The
  controller treats the measured target offset `e=[u-u0, v-v0]` as image error
  and solves a damped least-squares inverse of a local 2x2 image Jacobian from
  `[theta, radius]` to `[u, v]`. This follows the IBVS/interaction-matrix
  pattern from the visual-servo literature while remaining small enough for the
  existing JetMax teleop velocity topic. Fixed `pixel_axis` remains only a debug
  comparison mode. The default path now fails closed if the IBVS image feature
  or Jacobian is unavailable: it stops teleop and requests calibration or fresh
  vision data instead of silently reverting to old `servo_command_point`
  chasing.
- Confirm-height centering-only mode: the debug tool and main program both have
  an explicit stop-at-confirm mode for the current `z=120` tuning phase. When
  the controller reaches `PICK_READY`, this mode stops teleop, records the
  candidate `PICK_CYL`, and blocks execution. That keeps the validation target
  limited to camera-center alignment and prevents the `+40mm` forward extension
  or suction from being mixed into centering tests.
- Stage-aware Jacobian selection: the continuous controller now records the
  Jacobian source and values in every debug trace. The priority is a temporary
  fitted low-height model, then the profile `stage_models.confirm` response
  model converted from XY to cylindrical theta/radius at the current pose, then
  conservative config constants. This keeps the Hiwonder-style continuous
  feedback loop but avoids pretending one global pixel-to-motion mapping is
  accurate at every height.
- Low-height guardrails for uncalibrated servoing: once the arm enters the
  confirm-height band, keep a local anchor pose and the best image error seen so
  far. If theta/radius drift from that anchor exceeds small bounds, or if the
  target was once close to center and then rebounds, stop and require
  stop-settle-measure/local calibration instead of continuing to chase an
  untrusted image-to-motion mapping. A second guard detects the observed
  `z=120` failure mode: a repeatable 8-30 px residual near the confirm height
  that does not improve for multiple low-height frames. That now stops with
  `low_height_local_model_required` and points the operator to local search or
  calibration instead of wasting time in a smooth but ineffective servo loop.
  This is the conservative engineering version of local-Jacobian/un-calibrated
  visual servo practice.
- Low-height confirm alignment: treat the strict 2 px threshold as a final
  settled measurement, not as an in-motion trigger. Debug and calibration tools
  therefore use stop-settle-measure: wait for robot IDLE, wait a settle delay,
  drain transition MJPEG frames, then measure repeated low-height feature
  points from the persistent official stream. The default profile still starts
  from `geometry_subpixel`, but the confirm/pick stage now has a separate
  low-height measurement-point setting so live data can switch to a more stable
  point such as `top_face_subpixel` or `grasp_subpixel` without changing the
  high/mid-height tracking point. Stop-then-MOVE_CYL stepping is kept as an
  explicit diagnostic fallback for backlash/local-mapping isolation, not as the
  default smooth motion strategy.
- Low-height feature diagnostics: before writing a confirm-stage local model,
  compare `pixel_center_f`, `geometry_center_f`, `top_face_center_f`, and
  `grasp_pixel_f` over a stopped-frame sequence. Rank candidate points by
  repeat spread, median error to the alignment target, and jump count. This
  keeps the final 2 px gate tied to a named, stable visual feature instead of a
  vague "center" that may drift at close range.
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
- JetMax camera failure isolation: keep the Hiwonder sender baseline as
  `/home/hiwonder/ros/autostart/usb_cam.launch` at `640x480 yuyv 20 FPS` with
  `io_method=mmap`, then publish `/usb_cam/image_rect_color` through
  `web_video_server:8080` as MJPEG. Do not keep the hybrid-written
  `/etc/modprobe.d/hiwonder-uvcvideo.conf` during factory restore; if driver
  diagnosis is needed, use fail-closed `quirks=0 nodrop=0 timeout=5000` so
  incomplete UVC frames are dropped instead of published as gray/green frames.
  If a direct V4L2 YUYV read still returns far fewer than `640*480*2` bytes, the
  fault is below ROS/web_video_server, so frame-rate/quality tuning or PC-side
  masking is the wrong fix; reboot/power-cycle/re-seat the USB camera path
  before continuing centering tests.
- Time alignment for visual servo debugging: keep image processing latency
  (`image_age_ms`, derived from MJPEG capture/processing time) separate from
  robot pose/image synchronization (`frame_pose_age_ms`, derived from the
  nearest or post-frame robot-state timestamp). A slow detector frame must not
  be reported as stale robot pose unless the robot state sample is actually far
  from that image capture time.
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
- Hiwonder. "JetMax v1.0 documentation - Deep Learning Lesson." https://wiki.hiwonder.com/projects/JetMax/en/latest/docs/8_Deep_Learning_Lesson.html
- Hutchinson, S., Hager, G. D., and Corke, P. I. "A tutorial on visual servo control." IEEE Transactions on Robotics and Automation, 12(5), 651-670, 1996. https://doi.org/10.1109/70.538972
- Chaumette, F., and Hutchinson, S. "Visual servo control. I. Basic approaches." IEEE Robotics and Automation Magazine, 13(4), 82-90, 2006. https://doi.org/10.1109/MRA.2006.250573
- Chaumette, F., and Hutchinson, S. "Visual Servo Control, Part I: Basic Approaches." PDF mirror. https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf
- MoveIt. "Realtime Servo." https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html
- ros2_control. "joint_trajectory_controller user documentation." https://control.ros.org/rolling/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html
- Haviland, J., Dayoub, F., and Corke, P. "A Holistic Approach to Reactive Mobile Manipulation." arXiv, 2020. https://arxiv.org/abs/2001.05650
- OpenCV. "Contour Features." https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
- Music, J., Bonkovic, M., and Cecic, M. "Comparison of Uncalibrated Model-Free Visual Servoing Methods for Small-Amplitude Movements: A Simulation Study." International Journal of Advanced Robotic Systems, 2014. https://doi.org/10.5772/58822
- Piepmeier, J. A., and Lipkin, H. "Uncalibrated Eye-in-Hand Visual Servoing." International Journal of Robotics Research, 22(10-11), 805-819, 2003.
- Shademan, A., Farahmand, A.-M., and Jagersand, M. "Robust Jacobian Estimation for Uncalibrated Visual Servoing." ICRA 2010.
- Malis, E. "Improving vision-based control using efficient second-order minimization techniques." ICRA 2004. https://doi.org/10.1109/ROBOT.2004.1308092
- ROS `web_video_server`. "multipart_stream.cpp source." https://docs.ros.org/en/kinetic/api/web_video_server/html/multipart__stream_8cpp_source.html
- ROS `web_video_server`. "jpeg_streamers.cpp source." https://docs.ros.org/hydro/api/web_video_server/html/jpeg__streamers_8cpp_source.html
- ROS `web_video_server`. "HTTP streaming README." https://docs.ros.org/en/ros2_packages/rolling/api/web_video_server/index.html
- ROS `usb_cam`. "`usb_cam.cpp` source, select timeout path." https://docs.ros.org/jade/api/usb_cam/html/usb__cam_8cpp_source.html
- Linux kernel documentation. "The Linux USB Video Class driver." https://docs.kernel.org/userspace-api/media/drivers/uvcvideo.html
- Linux kernel media driver source. "uvcvideo driver module parameters." https://github.com/torvalds/linux/blob/master/drivers/media/usb/uvc/uvc_driver.c
- Linux kernel source browser. "`uvcvideo.h`, UVC_QUIRK_FIX_BANDWIDTH." https://codebrowser.dev/linux/linux/drivers/media/usb/uvc/uvcvideo.h.html
- Linux kernel mailing list. "`uvcvideo`: remove nodrop module parameter." https://lkml.iu.edu/2412.2/03131.html
- linux-hardware.org. "icSpring icspring camera USB 32e6:9005." https://linux-hardware.org/?id=usb%3A32e6-9005
- Morrison, D., Corke, P., and Leitner, J. "Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach." RSS 2018. https://arxiv.org/abs/1804.05172
- Mahler, J., Matl, M., Liu, X., Li, A., Gealy, D., and Goldberg, K. "Dex-Net 3.0: Computing Robust Robot Vacuum Suction Grasp Targets in Point Clouds using a New Analytic Model and Deep Learning." ICRA 2018. https://arxiv.org/abs/1709.06670
- Akyon, F. C., Altinuc, S. O., and Temizel, A. "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection." ICIP 2022. https://arxiv.org/abs/2202.06934
