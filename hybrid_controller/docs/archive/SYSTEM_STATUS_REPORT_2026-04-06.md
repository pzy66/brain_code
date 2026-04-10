# Brain System Status Report

Date: 2026-04-06
Scope: static code review of `brain_code` for MI, SSVEP, robot control, communication, and vision modules
Goal baseline: "MI movement -> SSVEP grab confirm -> MI carrying movement -> SSVEP place confirm"

## 1. Executive Summary

The codebase already contains four usable technical assets:

1. A relatively complete MI data collection, training, and realtime inference pipeline.
2. A usable realtime SSVEP stimulus and FBCCA decoding pipeline.
3. A working robot execution server with `MOVE`, `PICK`, and `PLACE` commands.
4. A vision-driven task prototype that already contains a search/pick/carry/place state skeleton.

The core problem is not "missing everything". The real problem is that the existing modules are not yet unified into one task coordinator, and the MI model quality is not yet strong enough for reliable four-class online control.

The best current integration base is not the MI repo alone. The best base is:

- Vision/task prototype: `05_Vision_Block_Recognition/2026-03_yolo_camera_detection/computer/test2.py`
- Robot backend: `03_RobotArm_Control/2026-03_jetmax_execution_server/test2_robot.py`
- MI decoder: `01_MI/mi_classifier_latest/code/shared/src/realtime_mi.py`
- MI realtime app logic: `01_MI/mi_classifier_latest/code/realtime/mi_realtime_infer_only.py`
- SSVEP decoder/UI: `02_SSVEP/2026-03_realtime_ui_and_online_decoder/SSVEP/demo.py`

The optimal next step is to build a unified coordinator around the existing vision state-machine skeleton, then replace mouse and keyboard simulation with real MI and SSVEP outputs in two stages.

## 2. Current Assets by Part

### 2.1 MI Module

Main folder:

- `01_MI/mi_classifier_latest`

What already exists:

- Four-class MI definition: `left_hand`, `right_hand`, `feet`, `tongue`
- Data collection UI and event schema
- Training pipeline for custom MI dataset
- Realtime inference with rolling-window prediction
- Stable prediction logic
- Confidence and margin thresholds
- Gate model for control-vs-rest
- Artifact rejection and freeze logic
- Continuous mode and guided mode

Key files:

- `01_MI/mi_classifier_latest/README.md`
- `01_MI/mi_classifier_latest/code/collection/mi_data_collector.py`
- `01_MI/mi_classifier_latest/code/shared/src/mi_collection.py`
- `01_MI/mi_classifier_latest/code/shared/src/realtime_mi.py`
- `01_MI/mi_classifier_latest/code/realtime/mi_realtime_infer_only.py`

Current maturity:

- Architecture maturity: high
- Online inference engineering maturity: medium to high
- Control-task readiness: low to medium
- Model readiness for four-class robot control: low

Why it is not ready yet:

- The repo explicitly focuses on MI only and does not include SSVEP.
- The runtime is good at producing stable class decisions, but it is still a classifier/UI, not a robot task controller.
- Current metrics are not sufficient for direct four-direction robot control.

Important evidence:

- Training summary shows `bank_test_acc = 0.375`
- `left_hand = 0.0`
- `right_hand = 1.0`
- `feet = 0.5`
- `tongue = 0.0`

Assessment:

- MI infrastructure is the strongest algorithmic foundation in the codebase.
- MI should be reused as the realtime decoder.
- MI should not yet be trusted as the direct closed-loop movement source until data and model quality improve.

### 2.2 SSVEP Module

Main folder:

- `02_SSVEP`

What already exists:

- Realtime stimulus presentation
- Realtime FBCCA decoding
- Smoothed prediction output
- Command emission signal
- Older CCA-based multiprocessing prototype

Key files:

- `02_SSVEP/2026-03_realtime_ui_and_online_decoder/SSVEP/demo.py`
- `02_SSVEP/2026-02_realtime_stimulus_and_classifier_core/main.py`
- `02_SSVEP/2026-02_realtime_stimulus_and_classifier_core/classifier.py`
- `02_SSVEP/2026-02_realtime_stimulus_and_classifier_core/stimulus.py`

Current maturity:

- Realtime decoding maturity: medium to high
- UI maturity: medium
- Task integration maturity: low

What is already good:

- The analyzer already outputs `decision`, `pred_f`, `smooth_pred`, and `command`.
- The worker already emits `command_detected`.
- The code is close to being wrapped as a reusable decision service.

What is not aligned with your goal:

- The latest GUI is currently hard-coded around four targets.
- The command mapping is still direction-style, not "confirm / cancel".
- The result is only logged in the GUI and does not yet trigger task transitions.

Assessment:

- SSVEP algorithm core is reusable.
- The next job is not "rewrite SSVEP".
- The next job is "split it into target-selection mode and two-choice decision mode".

### 2.3 Robot Arm Control Module

Main folder:

- `03_RobotArm_Control/2026-03_jetmax_execution_server`

What already exists:

- TCP server
- `MOVE x y`
- `PICK pixel_x pixel_y`
- `PLACE`
- Camera-to-world conversion inside pick logic
- Robot-side pick/place execution routines

Key file:

- `03_RobotArm_Control/2026-03_jetmax_execution_server/test2_robot.py`

Current maturity:

- Command execution maturity: high
- Integration maturity: medium
- Safety and robustness maturity: medium

Assessment:

- This is the best existing execution backend.
- Future integration should treat this file as the single source of truth for robot command protocol.

### 2.4 Communication and Integration Module

Main folder:

- `04_Communication_And_Integration`

What already exists:

- Older socket clients
- Older pick command sender
- EEG monitoring/debug script

Key files:

- `04_Communication_And_Integration/2026-02_socket_and_pick_command/choose+pick.py`
- `04_Communication_And_Integration/2026-02_socket_and_pick_command/connect.py`
- `04_Communication_And_Integration/2026-03_signal_monitoring_and_debug/t.py`

Current maturity:

- Historical reference value: medium
- Current production value: low

Main problem:

- Protocol inconsistency

Examples:

- `choose+pick.py` sends `PICK cx cy angle`
- robot server accepts only `PICK x y`
- `connect.py` sends single ASCII direction letters, which no longer match the robot server protocol

Assessment:

- This folder is useful as history.
- It should not be the center of new development.
- A single protocol adapter layer should replace these scattered scripts.

### 2.5 Vision and Block Recognition Module

Main folder:

- `05_Vision_Block_Recognition`

What already exists:

- YOLO-based realtime object detection
- Display-space and raw-image coordinate conversion
- ROI-based candidate filtering
- Frequency assignment to visible targets
- A state skeleton with search/picking/carry/placing
- Direct socket commands to the robot backend

Key files:

- `05_Vision_Block_Recognition/2026-03_yolo_camera_detection/computer/test2.py`
- `05_Vision_Block_Recognition/2026-03_yolo_camera_detection/deeplearning.py`
- `05_Vision_Block_Recognition/2026-02_template_matching_and_camera/template_maching.py`

Current maturity:

- Perception maturity: medium
- Task orchestration maturity: medium
- Brain-control integration maturity: low

Why this part matters most:

- It already contains the closest thing to your final task flow.
- It already knows about objects, target boxes, SSVEP flicker overlays, movement state, pick state, carry state, and place state.

Current limitation:

- Movement is simulated by mouse input.
- Object selection is simulated by keyboard keys `1-4`.
- Place is simulated by `Space`.
- Pick success and carry transition are timer-based assumptions, not decision-based state changes.

Assessment:

- This file should become the main coordinator baseline for the full project.

## 3. Gap Analysis Against Your Final Goal

Your target workflow is:

1. Stage 1: MI-guided movement for 10 seconds
2. Stage 1 decision: `8 Hz` enter stage 2, `15 Hz` continue stage 1
3. Stage 2: SSVEP object confirmation
4. Stage 2 decision: `8 Hz` confirm grab, `15 Hz` cancel and return stage 1
5. Stage 3: MI carrying movement for 10 seconds
6. Stage 3 decision: `8 Hz` confirm place, `15 Hz` continue stage 3

Current status versus target:

- Stage 1 movement timer: missing
- Stage 1 MI-to-motion bridge: missing
- Stage 1 two-choice SSVEP decision box: missing
- Stage 2 object-attention lock: partially present
- Stage 2 confirm/cancel decision layer: missing
- Real grab success feedback loop: weak
- Stage 3 carrying timer: missing
- Stage 3 MI-to-motion bridge: missing
- Stage 3 place confirmation decision box: missing
- Top-level task coordinator: missing

The codebase therefore has all major subsystem ingredients, but lacks the final "task logic layer" that coordinates them.

## 4. What Is Missing

The following pieces are missing if the project is to become a complete brain-controlled workflow:

1. A top-level finite state machine for the full task.
2. A unified event bus or controller API between MI, SSVEP, vision, and robot modules.
3. A timer-driven stage controller for fixed 10-second movement windows.
4. A dedicated two-option SSVEP decision UI for confirm/continue and confirm/cancel interactions.
5. A stable target identity mechanism in the vision module so object-frequency assignment does not drift frame to frame.
6. A proper MI command mapper from stable MI class to robot planar velocity or position increment.
7. Real acknowledgment or completion signals from robot actions.
8. Better MI data quality and model quality for four-class online use.
9. Protocol cleanup so every component talks the same robot command format.

## 5. Best Development Strategy

The best strategy is not to optimize MI first in isolation and not to rewrite everything from scratch.

The best strategy is:

### Step 1. Build the unified coordinator first

Base it on:

- `05_Vision_Block_Recognition/2026-03_yolo_camera_detection/computer/test2.py`

Why:

- It already contains the closest end-to-end task skeleton.
- It already manages state transitions.
- It already sends robot commands.
- It already displays object-linked SSVEP targets.

Target states should be redesigned as:

- `STAGE1_MI_MOVE`
- `STAGE1_SSVEP_DECISION`
- `STAGE2_TARGET_SELECTION`
- `STAGE2_GRAB_DECISION`
- `STAGE2_PICKING`
- `STAGE3_MI_CARRY`
- `STAGE3_SSVEP_DECISION`
- `STAGE3_PLACING`
- `TASK_FINISHED`

### Step 2. Freeze the robot protocol

Make `test2_robot.py` the only command protocol reference:

- `MOVE x y`
- `PICK x y`
- `PLACE`

Then update or retire old communication scripts.

### Step 3. Refactor SSVEP into two modes

Mode A:

- Object-selection mode
- Used when multiple blocks are visible
- Based on the current multi-target logic

Mode B:

- Two-choice decision mode
- Only `8 Hz` and `15 Hz`
- Used for continue/enter, confirm/cancel, confirm/place decisions

### Step 4. Integrate real SSVEP before real MI

Reason:

- SSVEP module is closer to usable task decisions than MI is to usable closed-loop four-direction control.
- This lets you first replace keyboard selection with real SSVEP decisions while still using simulated movement.

### Step 5. Improve MI dataset and online validation

Do this before relying on MI for robot movement.

Required work:

- Collect more sessions
- Balance all four classes
- Measure online stable output, not only offline accuracy
- Examine class confusion, especially `left_hand` and `tongue`
- Recheck thresholds and gate behavior for online use

### Step 6. Replace mouse movement with real MI movement

Map:

- `left_hand -> left`
- `right_hand -> right`
- `feet -> backward`
- `tongue -> forward`

Use only stable MI output after gate/artifact checks.

### Step 7. Add robot feedback and failure recovery

At minimum:

- robot busy state
- pick success/failure
- timeout recovery
- cancel return path

## 6. Optimization Priorities for Existing Code

### 6.1 MI Optimizations

- Increase data volume across multiple sessions
- Improve class balance and subject consistency
- Add explicit online evaluation reports for stable command rate, false activation rate, and command latency
- Separate "offline classifier good" from "online control usable"
- Tune thresholds for four-direction command stability rather than only classification accuracy

### 6.2 SSVEP Optimizations

- Remove the hard requirement that GUI must always be four-target
- Turn decoder output into reusable API events instead of GUI-only logs
- Add a lightweight two-choice decision layout
- Add confidence and hold-time safeguards for confirm/cancel decisions

### 6.3 Vision Optimizations

- Replace frame-by-frame nearest-object frequency reassignment with tracked target IDs
- Keep object-frequency binding stable while the object remains in scene
- Separate "object selection" from "grab confirmation"
- Add clearer visual cues for current stage and current selected target

### 6.4 Robot Control Optimizations

- Add action acknowledgment messages
- Add execution result states such as `PICK_OK`, `PICK_FAIL`, `PLACE_OK`
- Add safe timeout and movement clamp logic
- Log command timestamps for end-to-end debugging

### 6.5 Communication Optimizations

- Remove legacy single-character command assumptions
- Use one shared protocol definition for all modules
- Add heartbeat, error reporting, and reconnect logic
- Avoid one-off socket helper scripts as core architecture

## 7. Recommended Immediate Roadmap

If you want the fastest path to a usable experimental system, the roadmap should be:

1. Refactor `test2.py` into the official task coordinator.
2. Normalize robot communication around `test2_robot.py`.
3. Add explicit three-stage state machine and 10-second timers.
4. Convert SSVEP into "target selection mode" plus "two-choice decision mode".
5. Replace keyboard-based SSVEP simulation with real decoder events.
6. Keep movement simulated until state logic and robot action loop are stable.
7. Improve MI dataset and realtime performance.
8. Replace mouse-based movement with real MI commands.
9. Run full end-to-end closed-loop testing.

## 8. Final Conclusion

The project already has enough code to support a real integrated system. The main missing piece is not another standalone algorithm script. The main missing piece is a unified coordinator that turns existing MI, SSVEP, vision, and robot modules into one controlled task flow.

At this moment:

- The robot backend is ready enough to reuse directly.
- The vision prototype is the best integration starting point.
- The SSVEP decoder is close to usable after mode refactoring.
- The MI decoder architecture is good, but the model quality is not yet strong enough for dependable four-class robot movement.

Therefore, the most efficient path is:

- build the coordinator first,
- integrate real SSVEP second,
- improve and integrate MI third,
- then do full closed-loop validation.
