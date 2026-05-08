# Robot Code Backups

This directory stores manual snapshots of the JetMax/Hiwonder robot-side control code.

Scope:
- Source: `hybrid_controller/robot`
- Purpose: quick recovery or diffing before deploying robot-side control changes
- Camera rule: backups do not change the locked official camera sender chain

The current locked camera chain remains:

```text
usb_cam.service -> usb_cam_node -> /usb_cam/image_rect_color -> web_video_server:8080 -> PC
```

Use these backups only for robot-control code recovery. Do not restore camera sender files from here unless an explicit camera repair task is active.
