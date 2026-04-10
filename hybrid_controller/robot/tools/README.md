# robot/tools

这里放的是 JetMax 部署和诊断辅助工具。

当前保留：
- `deploy_jetmax_runtime.py`
- `jetmax_start_runtime.py`
- `jetmax_start_ros_runtime.py`
- `jetmax_env_probe.py`
- `jetmax_move_probe.py`

推荐主链启动（ROS + 相机 + TCP兼容）：

```bash
python robot/tools/jetmax_start_ros_runtime.py --host 192.168.149.1
```

这些工具只依赖当前主线目录：

- `brain_code/hybrid_controller`

不再依赖外面的 `jetmax_robot` 或其他旧副本。
