# Robot Control Backup 20260508_200424

Source snapshot: `hybrid_controller/robot`

Created before continuing JetMax robot-side control startup safety optimizations.

This backup is for control-code recovery and diffing. It should not be used to modify the official camera sender chain unless a future task explicitly reopens camera repair.

Locked camera chain:

```text
usb_cam.service -> usb_cam_node -> /usb_cam/image_rect_color -> web_video_server:8080 -> PC
```
