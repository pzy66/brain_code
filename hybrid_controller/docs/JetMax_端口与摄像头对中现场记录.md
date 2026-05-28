# JetMax 端口与摄像头对中现场记录

记录时间：2026-05-27

## 端口结论

上位机主流程只需要固定依赖这些端口：

```text
22/tcp    SSH，部署、启动 runtime、现场诊断
9091/tcp  rosbridge，GUI/调试工具调用 ROS service 和订阅状态
8080/tcp  web_video_server，读取 /usb_cam/image_rect_color MJPEG
```

JetMax 现场还会看到：

```text
11311/tcp ROS master，JetMax ROS1 基础服务；主流程不直接连它
80/tcp    Hiwonder/系统 Web 服务；主流程不用
8888/tcp  legacy TCP runtime；当前关闭，只保留兼容/诊断
9092/tcp  不使用，当前关闭
3xxxx-4xxxx/tcp ROS1 节点 XMLRPC 随机端口，不要在上位机硬编码
```

因此项目主链应保持：

```text
PC GUI/debug -> 9091 rosbridge
PC vision    -> 8080 web_video_server
PC deploy    -> 22 SSH
```

不要把 `8888`、`9092` 或 ROS1 随机端口纳入默认健康检查。

## 机械臂现场状态

SSH 登录 `hiwonder@192.168.149.1` 后确认：

```text
ssh.service       active
roscore.service   active
rosbridge.service active
usb_cam.service   active
hybrid runtime    可由 jetmax_start_ros_runtime.py 启动
```

相机发送链路保持：

```text
/dev/usb_cam0 -> usb_cam_node -> /usb_cam/image_rect_color -> web_video_server:8080
640x480 / yuyv / 20 FPS / mmap / MJPEG quality=80
```

这台 JetMax 的 USB 摄像头需要保留：

```text
/etc/modprobe.d/hiwonder-uvcvideo.conf
options uvcvideo quirks=128 nodrop=1 timeout=5000
```

删除这个兼容文件会让 `usb_cam_node` 进入 `select timeout`。它不是多余覆盖。

## 摄像头验证

重启后从上位机直接读取官方 URL：

```text
http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80
```

验证结果：

```text
ROS rostopic hz /usb_cam/image_rect_color: 约 20.0 FPS
PC 读取 MJPEG: 45/45 帧成功，640x480，约 18 FPS
坏帧症状: 未复现整屏绿色、乱码条带或上下坏帧
```

现场截图保存在：

```text
hybrid_controller/logs/camera_probe/
hybrid_controller/logs/vision_debug/
```

## 对中根因

连续对中第一次失败不是摄像头发送问题，而是调试工具把全局 `pixel_to_delta` 标定矩阵当成连续速度控制的 IBVS Jacobian 使用。

现象：

```text
初始中心误差约 88 px
错误 profile_global Jacobian 持续给正 theta
画面中心从 u≈334 先到 320 后继续冲到 u≈236
误差扩大到约 131 px
保护逻辑触发 target_center_jump
```

修复：

```text
debug_vision_grasp_flow.py 只在 stage profile 带明确 z_mm 且当前高度落在 stage band 内时，才用 profile Jacobian。
没有高度绑定的全局 profile 不再覆盖连续 IBVS Jacobian。
```

## 当前调试结果

修复后重新跑低高度对中：

```text
高位 z=205 -> 低位 z≈123.24
中心误差从约 76.7 px 降到最小 0.7 px
最后手动确认位姿：theta=0.58 deg, radius=143.21 mm, z=120.0 mm
stopped-frame 复测中心误差：0.7 px
吸盘保持 sucker_frozen，未执行真实吸取
```

这说明当前相机链路和对中方向已经恢复正常。后续要做真实抓取前，建议保持 `sucker_frozen` 先跑一次完整 `low_height_centering_check`，确认最后 stopped-frame 仍在 `2 px` 内，再解除冻结执行真实 `PICK`。
