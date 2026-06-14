用中文回答。

这是脑机机械臂一体化控制台的新电脑联调交接包。

优先事项：
1. 先确认 `BrainRobotWorkbench.exe` 能从便携包启动。
2. 再确认电脑已连接 JetMax/Hiwonder 机械臂 Wi-Fi，默认 IP 为 `192.168.149.1`。
3. 真机控制主链走 ROS rosbridge，默认端口 `9091`，不是 `9092`。
4. 机械臂端手动启动脚本是 `/home/hiwonder/brain_code/hybrid_controller/robot/run_hybrid_controller_ros_runtime.sh`。
5. 不要默认修改或重启 JetMax 官方摄像头链路；上位机读取官方 `web_video_server:8080`。
6. 不要只看 exe；需要调试时打开复制后的 `brain_code` 仓库源码。
