# 新电脑联调排查清单

## exe 启动不了

先确认不是只拷贝了 exe。正确结构应该是：

```text
BrainRobotWorkbench/
  BrainRobotWorkbench.exe
  _internal/
  BrainRobotWorkbench_PORTABLE_README.txt
```

运行 smoke test：

```powershell
.\scripts\start_exe_smoke.ps1 -ExePath "C:\path\to\BrainRobotWorkbench\BrainRobotWorkbench.exe"
```

## 连不上机械臂

先检查网络：

```powershell
ping 192.168.149.1
Test-NetConnection 192.168.149.1 -Port 22
Test-NetConnection 192.168.149.1 -Port 9091
```

含义：

- `ping` 不通：电脑还没有正确连接机械臂 Wi-Fi。
- `22` 不通：SSH 不通，自动启动也不可能成功。
- `22` 通但 `9091` 不通：需要手动启动机械臂端脚本。

## SSH 自动启动失败

按 `ROBOT_MANUAL_START.md` 手动启动：

```bash
cd /home/hiwonder/brain_code/hybrid_controller/robot
bash run_hybrid_controller_ros_runtime.sh
```

## 有控制但没摄像头

检查：

```powershell
Test-NetConnection 192.168.149.1 -Port 8080
```

默认摄像头 URL：

```text
http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80
```

不要默认重启 `usb_cam.service`，除非明确确认 JetMax 官方摄像头服务坏了。

## EEG 没波形

先看串口：

```powershell
Get-CimInstance Win32_SerialPort | Select-Object DeviceID,Name,Description
```

如果没有 COM 口：

- 检查脑电帽是否开机。
- 检查 USB/蓝牙/串口驱动。
- 检查设备管理器。

如果有 COM 口但软件没有数据：

- 先在软件里尝试重新连接脑电帽。
- 确认没有其他程序占用该 COM 口。

## 机械臂能连接但不抓取

优先检查这些：

1. `9091` 是否通。
2. 软件是否显示机械臂状态新鲜。
3. 摄像头画面是否正常。
4. 视觉是否识别到目标。
5. 选择目标后是否进入确认抓取阶段。
6. 是否再次确认抓取后才发命令。

当前流程要求：

```text
选中目标 -> 停止闪烁 -> 抓取确认 -> 确认后才抓取
```

## 端口不要混淆

- `9091`：当前主链 rosbridge。
- `8080`：摄像头 web_video_server。
- `8888`：legacy TCP 兼容/诊断。
- `9092`：不要用作当前主链。
