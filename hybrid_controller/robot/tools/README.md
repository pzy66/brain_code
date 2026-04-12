# robot/tools（部署与联调工具）

这些工具用于桌面端远程启动和探针，不是机械臂端常驻服务。

## 主线工具（建议使用）

- `jetmax_start_ros_runtime.py`
  - 通过 SSH 一键启动 JetMax ROS runtime
  - 会清理残留进程并停用系统自启旧 rosbridge
  - 默认触发机械臂端强制重启 rosbridge/web_video，确保主链进程和消息定义一致
- `ros_service_probe.py`
  - 通过 rosbridge 调用 ROS 服务，做联通与动作探针
- `jetmax_move_probe.py`
  - TCP 兼容链路的移动探针（可选）
- `jetmax_env_probe.py`
  - JetMax 端运行环境自检（可选）

## legacy 工具（仅兼容）

- `jetmax_start_runtime.py`
- `deploy_jetmax_runtime.py`

说明：这两项是 TCP-only 老路径，不是当前 GUI 的主链。

## 一键启动（ROS 主链）

```powershell
C:\Users\P1233\miniconda3\envs\brain-vision\python.exe C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\tools\jetmax_start_ros_runtime.py --host 192.168.149.1 --user hiwonder --password hiwonder --remote-root /home/hiwonder/brain_code
```

默认检查：

- `9091`（rosbridge）
- `8080`（web_video_server）

可选参数：

- `--require-tcp-check`：额外强制检查 `8888`
- `--skip-tcp-check`：显式跳过 TCP 检查
- `--skip-camera-check`：跳过 `8080` 检查

## ROS 服务探针

```powershell
C:\Users\P1233\miniconda3\envs\brain-vision\python.exe C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\tools\ros_service_probe.py --host 192.168.149.1 --port 9091 --action status
C:\Users\P1233\miniconda3\envs\brain-vision\python.exe C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\tools\ros_service_probe.py --host 192.168.149.1 --port 9091 --action pick_world --x 0 --y -162.94
C:\Users\P1233\miniconda3\envs\brain-vision\python.exe C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\tools\ros_service_probe.py --host 192.168.149.1 --port 9091 --action place
C:\Users\P1233\miniconda3\envs\brain-vision\python.exe C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\robot\tools\ros_service_probe.py --host 192.168.149.1 --port 9091 --action reset
```

支持 `--action`：

- `status`
- `move_cyl`
- `move_cyl_auto`
- `pick_world`
- `pick_cyl`
- `place`
- `abort`
- `reset`
