# hybrid_controller_sim

这个目录是 `hybrid_controller` 主线之外的独立仿真实验工程。

用途：
- 保留 fake robot、replay、simulation world
- 为后续仿真调试提供独立空间
- 允许依赖 `brain_code/hybrid_controller` 的稳定接口

约束：
- 主线程序 **不能** 反向依赖这里的代码
- 这里只是实验/仿真环境，不是当前真机主入口

当前包含：
- `simulation_world.py`
- `fake_robot_server.py`
- `fake_vision_source.py`
- `replay_source.py`
- `test_simulation_world.py`
