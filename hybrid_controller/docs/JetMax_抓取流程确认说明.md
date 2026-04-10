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

## 抓后姿态与放置策略（本轮关键）

当前不再采用“抓完固定回 `z_carry`”：

- 抓取完成后抬升到：
  - `z_settle = clamp(max(z_carry_floor, z_auto(target_radius)))`
- 该 `z_settle` 会写入状态，下一次 `MOVE_CYL_AUTO` 直接从这个姿态继续，避免高度突变。

放置阶段同样对齐稳定姿态：

- 放置前只在“当前高度低于 `z_carry_floor`”时才补抬高
- 放置完成后抬回当前半径对应的 `z_settle`

## 下探与吸盘时序参数（可实时调）

抓放参数已从硬编码改为运行时配置：

- `pick_approach_z_mm`
- `pick_descend_z_mm`
- `pick_pre_suction_sec`
- `pick_bottom_hold_sec`
- `pick_lift_sec`
- `place_descend_z_mm`
- `place_release_mode` (`release` / `off`)
- `place_release_sec`
- `place_post_release_hold_sec`
- `z_carry_floor_mm`

执行顺序：

1. 抓取：approach -> 开吸盘 -> 预吸附等待 -> 下探 -> 底部保持 -> 抬升到 `z_settle`
2. 放置：下降 -> 释放 -> 释放保持 -> 抬升到 `z_settle`

释放兼容策略：

- 优先调用硬件 `release()`
- 若硬件无该接口，则自动回退为 `set_state(False) + release_sec`

## 当前调试要点

- 先看 UI 里的 `resolved_xy` 与 `resolved_cyl`
- 再看 `pick_trace`（命令、解析结果、响应）
- 若失败，优先区分：
  - 坐标链路问题
  - 状态门控问题
  - 执行端动作失败

## 现场调试建议（按顺序）

1. 先调 `pick_descend_z_mm`（步进 `1 mm`）
2. 再调 `pick_pre_suction_sec` 与 `pick_bottom_hold_sec`（步进 `0.05 s`）
3. 再调放置 `place_release_mode/release_sec/post_release_hold_sec`
4. 最后调 `z_carry_floor_mm`，确保抓后移动不突变

完成后点击：

- `应用到机器人`（立即生效）
- `保存配置`（落盘到 `dataset/robot_pick_tuning/current_pick_tuning.json`）
