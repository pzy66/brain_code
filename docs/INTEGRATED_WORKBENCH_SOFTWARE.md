# Integrated Robot Workbench Software

This is the packaged desktop application for the three-stage keyboard-operated BCI robot flow.

## Run From Source

```powershell
cd D:\brain\brain_code
$py = & .\tools\resolve_brain_python.cmd
& $py -m robot_workbench.app
```

Equivalent shortcuts:

```powershell
& $py -m brain
.\START_INTEGRATED_WORKBENCH.cmd
```

## Default Runtime Policy

- The UI starts with a clean device connection gate. The operator must connect the robot arm and the EEG cap before entering the SSVEP, MI, and robot control stages.
- After the connection gate, the UI follows the three-stage flow: SSVEP setup, MI setup, then robot closed-loop control.
- After both device gates are ready, the operator can also enter robot control directly for manual movement and live camera operation.
- The robot control page prioritizes a large camera region. Secondary controls are kept in a narrow side rail and key robot state is overlaid on the camera view.
- MI/SSVEP realtime recognition is not started; keyboard and buttons replace recognition decisions.
- EEG cap connection is still a readiness gate for the keyboard-operated flow, but the side preview can start a lightweight BrainFlow stream and display the incoming raw 8-channel EEG waveform in real time. It does not run classifier inference.
- The EEG preview keeps only the recent display window in memory, so long-running monitoring does not accumulate a full recording buffer.
- ROS robot control is available by default at `192.168.149.1:9091`, but the app does not auto-connect until the operator clicks connect.
- When the operator clicks robot connect, the app now shows staged progress for WiFi/rosbridge checks, `/hybrid_controller/state` readiness, SSH runtime startup, and the final reconnect. Temporary rosbridge failures are treated as in-progress states; only the final failure is shown as a connection failure.
- The remote runtime start uses `--skip-camera-check --skip-tcp-check`; it restores the robot control runtime and does not start, restart, or repair the JetMax camera sender.
- The robot page reads the official JetMax MJPEG camera stream at `http://<robot-host>:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80`.
- The camera display is video-only in this profile; it does not start object detection and can run without `ultralytics/torch/cv2`.
- The robot page uses the camera as the primary operation view, with overlays for phase state, countdown, selected target, pose, gripper state, and keyboard command state.
- The robot page includes a manual drive mode. After the robot is connected, click `手动移动`, hold `W/A/S/D` or arrow keys to move, release the key to stop, and press `Esc` for safe stop/reset.

## UI Design Direction

- Connection first: the first screen is a compact preflight gate, not the full control dashboard.
- Camera first: the robot stage treats the live robot camera as the primary workspace and avoids crowding it with secondary panels.
- Fullscreen friendly: side rails use responsive width tiers while the camera viewport consumes the remaining space during maximized or fullscreen use.
- Keyboard-operated BCI: SSVEP and MI pages remain in the product flow, but decisions are confirmed by keyboard/buttons in this software profile.
- Console layout: robot commands, target selection, pose, gripper state, and logs stay in a narrow left rail; live camera, crosshair, target overlay, and decision prompts stay in the main viewport.

Keyboard mapping:

```text
S = complete SSVEP setup stage
M = complete MI setup stage
W/A/S/D or arrow keys = move the robot during MI-move phases
W/A/S/D or arrow keys = also move the robot while manual drive mode is active
1/2/3/4 = select target block
Enter or Space = confirm current decision
C or Backspace = continue/cancel current decision
G = execute grab when target confirmation is active
P = execute place at the final decision point
Esc = safe stop/reset the current robot task
```

Camera options:

```powershell
& $py -m robot_workbench.app --camera-stream-url "http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80"
& $py -m robot_workbench.app --no-camera-auto-start
& $py -m robot_workbench.app --eeg-serial-port COM4 --eeg-board-id 0
& $py -m robot_workbench.app --eeg-serial-port auto --eeg-board-id -1
& $py -m robot_workbench.app --no-eeg-signal-auto-start
```

For OpenBCI Cyton, use `--eeg-board-id 0` and the receiver serial port such as `COM4`. If the serial field stays as `auto`, the app will try detected serial ports; when no port is detected, the keyboard workflow remains available and the signal panel clearly stays in placeholder mode.

## Build Windows App

```powershell
cd D:\brain\brain_code
.\tools\build_integrated_workbench.ps1
```

Output:

```text
dist\BrainRobotWorkbench\BrainRobotWorkbench.exe
```

The build includes the default model/profile files that are already tracked in the repository. The default software profile launches the new flow UI. To run the previous `hybrid_controller` interface for debugging, pass `--legacy-hybrid-ui`.
