# 版本说明草稿

## V1.0 申报名称

基于混合脑机接口的智能机械臂协同控制软件 V1.0

## V1.0 范围

- 软著演示工作台：六页主导航，总览、数据采集、训练评估、在线控制、视觉机械臂、软著材料。
- MI 接入契约：训练入口、实时推理入口、profile/model 输出入口和统一状态 `ready / missing / training / published / error`。
- SSVEP 复用：只读取既有入口、profile、report 和 async 控制指标，不复制训练逻辑。
- 视觉机械臂复用：只读取 `hybrid_controller` 当前配置、模型和 profile 状态，真实动作保留在原安全门控内。
- 材料工作流：说明书、用户手册、测试报告、源码交存清单、版本说明和引用资料目录。

## 暂不纳入 V1.0 的内容

- 直接在软著 UI 内启动真实抓取、写入 profile 或绕过 `hybrid_controller` 安全门控。
- 将真实 EEG 数据、外部数据集原始文件、论文 PDF、训练输出和临时缓存纳入源码交存。
- 在 MI 分类器正式入库前，承诺具体模型权重或最终离线精度。

## 下一版本候选

- 增加 `brain launch --target softcopyright` 统一入口。
- 补充 MI profile schema 校验工具和训练/实时 smoke test。
- 增加软著冻结清单自动生成脚本。
- 增加只读运行日志汇总和截图索引。
