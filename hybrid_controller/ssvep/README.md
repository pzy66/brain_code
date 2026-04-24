# ssvep

这里是主程序内置的 SSVEP 最小内核。

目标：
- 主程序不再动态加载外部 reference 脚本
- 主程序不再依赖 `02_SSVEP/...` 的运行时代码
- 只保留当前真机主线真正需要的 FBCCA 相关功能

## 当前包含

- [async_fbcca_idle.py](./async_fbcca_idle.py)
  - profile 读写、fallback profile、gate 判别、串口/设备辅助逻辑
- [single_model.py](./single_model.py)
  - 单模型 FBCCA 预训练与在线 worker
- [validation_ui.py](./validation_ui.py)
  - 预训练/验证辅助逻辑
- [backend.py](./backend.py)
  - 后端封装
- [profiles.py](./profiles.py)
  - profile 路径与元数据辅助
- [service.py](./service.py)
  - 主程序可复用 service 层
- [runtime.py](./runtime.py)
  - 主程序运行时桥接

## Profile 目录

主程序默认使用：

- [dataset/ssvep_profiles](../dataset/ssvep_profiles)

其中：
- 历史 profile：`ssvep_fbcca_profile_YYYYMMDD_HHMMSS.json`
- 当前 profile 别名：`current_fbcca_profile.json`
- fallback profile：`fallback_fbcca_profile.json`

## 说明

这套代码来自 `02_SSVEP/2026-04_async_fbcca_idle_decoder` 的最小必要提取版。  
外部实验目录保持原样保留，但主程序运行时只认这里。
