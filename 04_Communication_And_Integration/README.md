# 04 Communication And Integration

这里集中存放通信与跨模块联调脚本。

## 子目录说明

- `2026-02_socket_and_pick_command/`
  socket 通信、抓取命令发送、联调脚本
- `2026-03_signal_monitoring_and_debug/`
  实时信号监视与调试工具

## 核心文件

- `2026-02_socket_and_pick_command/connect.py`
- `2026-02_socket_and_pick_command/choose+pick.py`
- `2026-03_signal_monitoring_and_debug/t.py`

## 说明

- `choose+pick.py` 是跨模块脚本，连接了视觉检测和抓取命令发送
- 这里主要保存“通信/联调”性质代码，不把视觉算法和机械臂执行端混放在一起
