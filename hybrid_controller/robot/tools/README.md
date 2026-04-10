# robot/tools

这个目录放 JetMax 部署与联调工具，全部服务于主线目录：

- `C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller`

## 主要工具

- `jetmax_start_ros_runtime.py`
  - 通过 SSH 启动 JetMax 端 ROS runtime
- `ros_service_probe.py`
  - 直接调用 ROS service 做动作探针
- `jetmax_env_probe.py`
  - 环境检查
- `jetmax_move_probe.py`
  - 基础移动探针
- `deploy_jetmax_runtime.py`
  - 部署辅助

## jetmax_start_ros_runtime.py

常用启动：

```bash
python robot/tools/jetmax_start_ros_runtime.py --host 192.168.149.1 --user hiwonder --password hiwonder --remote-root /home/hiwonder/brain_code
```

端口检查参数：

- 默认：检查 `9091`（rosbridge）+ `8080`（视频）
- `--require-tcp-check`：额外强制检查 `8888`
- `--skip-tcp-check`：显式跳过 TCP 检查
- `--skip-camera-check`：跳过 `8080` 检查

## ros_service_probe.py

示例：

```bash
python robot/tools/ros_service_probe.py --host 192.168.149.1 --port 9091 --action status
python robot/tools/ros_service_probe.py --host 192.168.149.1 --port 9091 --action pick_world --x 0 --y -162.94
python robot/tools/ros_service_probe.py --host 192.168.149.1 --port 9091 --action place
python robot/tools/ros_service_probe.py --host 192.168.149.1 --port 9091 --action reset
```

可用 `--action`：

- `status`
- `move_cyl`
- `move_cyl_auto`
- `pick_world`
- `pick_cyl`
- `place`
- `abort`
- `reset`
