# JetMax 视觉链路优化手册

## 当前定位

视觉主线已经内置到 `hybrid_controller`，当前目标不是再运行旧的独立视觉 UI，而是：

- 在主界面中显示 JetMax 相机主画面
- 叠加 ROI、检测框和槽位频闪
- 将木块中心转换为机械臂基座圆柱坐标
- 供后续 `PICK_CYL` 直接消费

## 当前主链

- 相机来源：JetMax 相机流
- 检测模型：仓库内权重
  - [models/vision/best.pt](C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\models\vision\best.pt)
- 运行时代码：
  - [vision/runtime.py](C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\vision\runtime.py)
  - [vision/processing.py](C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\vision\processing.py)

## 当前行为

- 主窗口中央以相机画面为主
- 右上角显示圆柱姿态小窗
- ROI 内最多 4 个目标进入槽位
- 槽位目标按 05 方案做频闪叠加
- 每个槽位输出：
  - 像素中心
  - `command_mode="cyl"`
  - `command_point=(theta_deg, radius_mm[, z_mm])`

## 调试顺序

1. 先确认 JetMax 相机流正常
2. 确认主界面能稳定显示相机画面
3. 确认 ROI 和检测框叠加稳定
4. 确认槽位编号和频闪分配稳定
5. 最后确认槽位圆柱坐标是否可用于 `PICK_CYL`

## 当前结论

- 视觉主线现在属于 `hybrid_controller` 自有代码
- 旧的 `05_Vision_Block_Recognition` 目录保留为历史参考
- 主程序运行时不再依赖外部视觉脚本
