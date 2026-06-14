# 机械臂端手动启动

如果 exe 里显示 SSH 自动启动机械臂失败，但是电脑能连上机械臂 Wi-Fi，可以手动启动机械臂端代码。

## 1. 从 Windows SSH 到机械臂

```powershell
ssh hiwonder@192.168.149.1
```

默认密码：

```text
hiwonder
```

## 2. 在机械臂上运行主脚本

如果依赖已经安装过，只运行：

```bash
cd /home/hiwonder/brain_code/hybrid_controller/robot
pkill -f hybrid_controller_runtime_node.py || true
bash run_hybrid_controller_ros_runtime.sh
```

这个窗口不要关，保持它运行。

## 3. 展示时后台运行

```bash
cd /home/hiwonder/brain_code/hybrid_controller/robot
pkill -f hybrid_controller_runtime_node.py || true
nohup bash run_hybrid_controller_ros_runtime.sh > ~/hybrid_controller_runtime.log 2>&1 &
tail -f ~/hybrid_controller_runtime.log
```

## 4. 脚本实际做什么

`run_hybrid_controller_ros_runtime.sh` 会：

1. 把 `ros_pkg/hybrid_controller_ros` 同步到 `~/catkin_ws/src`。
2. 必要时编译 ROS package。
3. 启动或复用 rosbridge，默认端口 `9091`。
4. 启动 `hybrid_controller_runtime_node.py`。
5. 保持官方摄像头链路，不默认修改 `usb_cam.service` 或 `web_video_server`。

## 5. 从 Windows 验证

```powershell
Test-NetConnection 192.168.149.1 -Port 9091
```

如果显示：

```text
TcpTestSucceeded : True
```

说明机械臂控制链路起来了，可以回到 exe 重新连接机械臂。

## 6. 如果目录不存在

如果机械臂上没有这个目录：

```text
/home/hiwonder/brain_code/hybrid_controller/robot
```

说明机械臂端代码没有部署过去。需要把当前仓库的：

```text
hybrid_controller/robot
```

复制到机械臂：

```text
/home/hiwonder/brain_code/hybrid_controller/robot
```
