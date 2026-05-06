# 2026-05 Vision Grasp Flow Debug

This is the standalone debug workspace for validating:

```text
camera -> recognition -> target resolution -> MOVE alignment -> optional PICK
```

It intentionally lives outside `hybrid_controller` so the main program does not
become the place where unfinished hardware debugging happens.

## Safety ladder

Run the flow in this order:

```text
dry-run -> camera-only -> resolve-only -> move-only -> execute-move -> allow-pick
```

- `dry-run`: capture frames, build packets, and save artifacts; no robot command.
- `camera-only`: validate camera/model output only; no robot connection required.
- `resolve-only`: connect/read robot state if available and resolve target to a
  robot command; no robot command is sent.
- `move-only`: compute MOVE decisions only; no command is sent.
- `execute-move`: send MOVE commands only when ROS is connected.
- `allow-pick`: allow PICK commands; this is the only mode that can descend and
  turn suction on.

## Entry point

```powershell
$py = & .\tools\resolve_brain_python.cmd
& $py .\04_Communication_And_Integration\2026-05_vision_grasp_flow_debug\debug_vision_grasp_flow.py --mode dry-run
```

Artifacts are written under `artifacts/vision_grasp_flow_debug/` by default.

## Promotion rule

Once this flow is validated with real hardware, copy only the stable minimum
implementation into `hybrid_controller` and record the copy in:

```text
hybrid_controller/VENDORED_SOURCES.md
```
