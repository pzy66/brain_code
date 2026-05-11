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
- GG-CNN closed-loop grasping: prefer feedback-driven correction and stable
  grasp quality over one-shot open-loop detection. The current implementation
  keeps continuous servo experimental and leaves discrete servo as the safe
  default.
- Dex-Net 3.0 suction grasping: treat suction placement as a local seal-quality
  problem. The current mask geometry searches for an internal, bright/top-face
  suction point and records grasp quality for later correction.
- SAHI small-object detection: keep slicing as a future option only if replay
  shows missed small blocks caused by input resolution. It is not enabled by
  default because it can add latency and duplicate-merging complexity.

## References

- Ultralytics. "Instance Segmentation - Ultralytics YOLO Docs." https://docs.ultralytics.com/tasks/segment/
- Ultralytics. "Model Prediction with Ultralytics YOLO." https://docs.ultralytics.com/modes/predict/
- Chaumette, F., and Hutchinson, S. "Visual servo control. I. Basic approaches." IEEE Robotics and Automation Magazine, 13(4), 82-90, 2006. https://doi.org/10.1109/MRA.2006.250573
- Morrison, D., Corke, P., and Leitner, J. "Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach." RSS 2018. https://arxiv.org/abs/1804.05172
- Mahler, J., Matl, M., Liu, X., Li, A., Gealy, D., and Goldberg, K. "Dex-Net 3.0: Computing Robust Robot Vacuum Suction Grasp Targets in Point Clouds using a New Analytic Model and Deep Learning." ICRA 2018. https://arxiv.org/abs/1709.06670
- Akyon, F. C., Altinuc, S. O., and Temizel, A. "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection." ICIP 2022. https://arxiv.org/abs/2202.06934
