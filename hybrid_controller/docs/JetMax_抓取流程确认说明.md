# JetMax 抓取流程确认说明

本文描述当前主线真实执行的抓取链路。

## 默认视觉抓取主路径

当前默认视觉抓取不是 `PICK_CYL`，而是：

1. 相机检测到 ROI 内目标
2. 视觉层输出 `command_mode=world` + `command_point=(x,y)`
3. 控制层校验 `actionable=true`
4. 下发 `PICK_WORLD x y`
5. 执行端按状态机完成抓取
6. 下发 `PLACE` 放置

## 命令分工

- `PICK_WORLD x y`
  - 视觉主路径（机器人基座坐标系）
- `PICK_CYL theta r`
  - 手动调试/圆柱入口
- `PICK u v`
  - 像素抓取兼容入口（非当前默认）

## 抓取前拒绝条件

以下任一成立都会拒绝抓取：

- 目标 `actionable=false`
- 视觉映射无效（标定不可用/快照过旧/超边界）
- 机器人处于 BUSY 或 ERROR 不允许抓取状态

## 执行状态机（抓取阶段）

- `PICK_APPROACH`
- `PICK_SUCTION_ON`
- `PICK_DESCEND`
- `PICK_LIFT`
- `CARRY_READY`

## 当前调试要点

- 先看 UI 里的 `resolved_xy` 与 `resolved_cyl`
- 再看 `pick_trace`（命令、解析结果、响应）
- 若失败，优先区分：
  - 坐标链路问题
  - 状态门控问题
  - 执行端动作失败
