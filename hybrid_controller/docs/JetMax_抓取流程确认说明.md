# JetMax 抓取流程确认说明

这份文档只描述当前主线里真实在用的抓取链路。

## 当前主抓取路径

当前默认抓取不是：
- `PICK pixel_x pixel_y`
- `PICK_WORLD x y`

当前主抓取路径是：

1. 相机取流
2. YOLO 检测木块
3. ROI 内目标进入最多 4 个槽位
4. 每个槽位转换成机械臂基座圆柱坐标
5. 用户选中一个槽位
6. 主控下发 `PICK_CYL theta r`
7. JetMax 执行：
   - `PICK_APPROACH`
   - `PICK_SUCTION_ON`
   - `PICK_DESCEND`
   - `PICK_LIFT`
8. 抓起后进入带载移动
9. 用户确认放置后下发 `PLACE`
10. JetMax 在当前位置下放、释放、再抬起

## 当前语义

- 视觉层主输出：圆柱坐标
- 执行端主抓取命令：`PICK_CYL`
- `PLACE` 当前语义：在当前位置放下，不是指定放置点

## 各层职责

### 视觉层

负责：
- 相机帧
- 检测框
- ROI 与槽位
- 槽位频闪
- 像素中心到圆柱坐标的换算

### 控制器

负责：
- 冻结当前槽位列表
- 记录选中的目标
- 在确认抓取时，根据 `command_mode` 选择真正下发的命令

当前真机主线要求：
- 选中的目标必须是 `command_mode="cyl"`
- `command_point` 必须能解析为 `(theta, r)` 或 `(theta, r, z)`

### JetMax 执行端

负责：
- 接收 `PICK_CYL`
- 验证圆柱目标是否在安全范围内
- 转成内部执行目标
- 按既定抓取状态序列执行

## 实机确认前检查

1. GUI 显示：
   - `robot_connected=True`
   - `preflight ok=True`
2. 机械臂状态：
   - `state=IDLE` 或 `CARRY_READY`
   - `busy=False`
3. 相机主画面正常
4. ROI 和槽位稳定
5. 先只确认槽位，不立刻下抓取
6. 再做第一次 `PICK_CYL`
