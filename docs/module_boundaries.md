# brain_code module boundaries

This repository keeps the numbered folders as independent work areas and uses
`hybrid_controller` as the final integration runtime. The default rule is:
debug or tune a subsystem in its own folder first, then copy the validated
minimum code into `hybrid_controller` with source tracking.

## Allowed ownership

- `01_MI`: MI collection, training, realtime inference, and its local tests.
- `02_SSVEP`: SSVEP collection, realtime decoding, offline optimization,
  profiles, and its local tests.
- `03_RobotArm_Control`: JetMax-side execution reference and deployment source.
- `04_Communication_And_Integration`: cross-module debugging, socket/ROS probes,
  and end-to-end recognition-to-grasp validation tools.
- `05_Vision_Block_Recognition`: camera, image collection, visual recognition,
  vision model training, and vision-only validation.
- `06_Data_Collection`: standalone dataset collection helpers that are not MI
  or SSVEP specific.
- `07_Simulation_Lab`: simulation adapters for stable `hybrid_controller` APIs.
- `hybrid_controller`: final runtime UI, local copied integration code, robot
  command flow, vision runtime, and SSVEP/MI adapters used by the main program.

## Import policy

- Numbered sibling folders must not runtime-import each other's private code.
- `hybrid_controller` must not runtime-import from `01_MI`, `02_SSVEP`,
  `04_Communication_And_Integration`, or `05_Vision_Block_Recognition`.
- If the main program needs proven code from a numbered folder, copy the minimum
  required implementation into the appropriate `hybrid_controller` package and
  record it in `hybrid_controller/VENDORED_SOURCES.md`.
- Compatibility launchers may call `brain_workspace.paths.ensure_runtime_import_paths()`
  explicitly, but default package imports and tests should not depend on hidden
  global `sys.path` additions.
- `_archive`, runtime output, logs, local datasets, and generated model folders
  are not valid sources for active imports.

## Debug-to-main promotion flow

1. Build and verify the focused debug tool in its owning numbered folder.
2. Capture a replayable artifact: JSON report, image/overlay, command decision,
   and relevant config paths.
3. Promote only stable behavior into `hybrid_controller`.
4. Update tests and `hybrid_controller/VENDORED_SOURCES.md`.
5. Keep the original numbered-folder tool as a reproducible debugger, not as a
   runtime dependency of the main program.

## Hardware safety gates

Recognition-to-grasp tools must default to no robot motion. The required
escalation is:

```text
dry-run -> camera-only -> resolve-only -> move-only -> execute-move -> allow-pick
```

`PICK_*` commands must require an explicit `--allow-pick` style flag in addition
to any execution flag.
