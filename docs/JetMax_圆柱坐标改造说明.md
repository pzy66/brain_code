# JetMax 圆柱坐标改造说明

## 1. 这次改造解决的是什么问题

JetMax 官方主控制方式更偏向笛卡尔末端位姿控制。典型写法是：

```python
jetmax.set_position((x, y, z), t)
```

官方视觉抓取课程和动作课程里，抓取、抬升、搬运的主路径都是围绕 `x/y/z + duration` 来写的；更底层的舵机课程则直接给出了 `hiwonder.serial_servo.set_position(id, pulse, ms)` 这种脉宽级接口。[JetMax 官方文档](https://wiki.hiwonder.com/projects/JetMax/en/latest/)、[AI Vision Games Lesson](https://wiki.hiwonder.com/projects/JetMax/en/latest/docs/3_AI_Vision_Games_Lesson.html)、[Advanced Program Lessons](https://wiki.hiwonder.com/projects/JetMax/en/latest/docs/6_Advanced_Program_Lessons.html)

这套官方方式本身没有错，但对我们这个项目的上层任务控制并不直观，主要问题是：

- 对操作者来说，`x/y/z` 不如“转多少角、伸多远、抬多高”直观。
- `x/y` 常常要配合正负号记忆，尤其在真机调试时容易搞混。
- 真实可达空间并不是一个规则长方体，直接按 `x/y` 矩形限位会掩盖机械臂本身更自然的工作空间结构。
- 后续接 MI / SSVEP 时，用户意图更接近“转”“伸”“收”“抬”，而不是直接给末端 `x/y/z`。

所以这次改造的目标不是推翻 JetMax 官方底层，而是把**我们的系统主接口**改成更贴近 JetMax 运动直觉的圆柱坐标：

- `theta_deg`：绕底座转角
- `radius_mm`：水平半径
- `z_mm`：高度

当前项目里经过真机验证后，最终采用的符号约定是：

- `theta = 0°`：机械臂正前方
- `theta > 0`：机械臂向自身左侧转
- `theta < 0`：机械臂向自身右侧转

这个约定和 `brain_code/hybrid_controller/cylindrical.py` 里的转换公式是一致的：

- `x = r * sin(theta)`
- `y = -r * cos(theta)`

因为 JetMax 的“正前方”在当前项目里定义为世界坐标 `-y` 方向，所以正角度落到 `+x` 分支，实机上表现为机械臂向自身左侧旋转。这个定义后面应当保持全系统一致，不再混用。

## 2. 这次到底改到了哪一层

这次改造是**新增一条圆柱坐标主线路**，不是把官方底层驱动改写成原生圆柱坐标。

当前分层如下：

### 2.1 用户层 / 任务层

用户、GUI、状态机优先使用圆柱语义：

- `MOVE_CYL theta r z`
- `MOVE_CYL_AUTO theta r`
- `PICK_CYL theta r`

对应代码：

- `brain_code/hybrid_controller/controller/task_controller.py`
- `brain_code/hybrid_controller/app.py`
- `brain_code/hybrid_controller/ui/main_window.py`

其中，默认移动阶段已经优先发 `MOVE_CYL_AUTO`，而不是旧的 `MOVE x y`。

### 2.2 运动学与安全层

我们新增了共享圆柱转换和自动高度 profile：

- `brain_code/hybrid_controller/cylindrical.py`

这里负责：

- `theta/r/z <-> x/y/z` 转换
- `z_auto(r)` profile 生成与插值
- 半径采样、候选高度筛选、姿态优先打分

这是这次改造最核心的一层。因为它决定了：

- 用户输入圆柱坐标后，系统如何内部转换
- `r` 变化时，`z` 是固定、联动还是自适应
- “姿态优先”还是“高度优先”

### 2.3 执行端内核

执行端现在明确拆成两条控制线路：

- `legacy_cartesian`
- `cylindrical_kernel`

对应代码：

- `brain_code/hybrid_controller/integrations/robot_runtime.py`
- `brain_code/jetmax_robot/robot_runtime_py36.py`

这里不是简单地“协议换个名字”，而是：

- 圆柱指令进入执行端后，先在圆柱空间里做验证、规划和状态更新
- `STATUS` 同时返回 `robot_cyl` 和 `robot_xy`
- 执行端内部会记录当前走的是哪条 kernel：`control_kernel`

也就是说，**我们的执行端底层逻辑已经是圆柱主语义**。

### 2.4 官方驱动层

这一层没有被修改，仍然保留官方行为：

- `hiwonder.JetMax.set_position((x, y, z), duration)`
- `hiwonder.serial_servo.set_position(id, pulse, ms)`

所以要说清楚：

- 我们已经把**自己的底层控制内核**改成了圆柱主语义
- 但最末端驱动 JetMax 舵机时，仍然还是落回官方 `x/y/z + IK + servo pulse`

这是这版最稳妥的方案，因为它保留了官方驱动的稳定性和兼容性。

## 3. 具体改了什么

### 3.1 新增了圆柱坐标主接口

当前主接口已经包含：

- `MOVE_CYL theta r z`
- `MOVE_CYL_AUTO theta r`
- `PICK_CYL theta r`

同时保留 legacy 兼容命令：

- `MOVE x y`
- `PICK u v`
- `PICK_WORLD x y`
- `PLACE`

这样改的好处是：

- 老代码、老测试、老协议不需要立刻全部重写
- 新功能和后续 MI/SSVEP 优先走圆柱线路
- 迁移风险更低

### 3.2 引入了圆柱 canonical pose

在 `brain_code/hybrid_controller/cylindrical.py` 里新增了 `CylindricalPose`，让系统有一个统一的圆柱规范状态，而不是把圆柱当作输入皮肤、内部仍然到处散落 `x/y/z` 计算。

这一步的好处是：

- 状态更统一
- 调试时更容易看懂当前姿态
- `STATUS`、日志、GUI 和执行内核可以说同一种“坐标语言”

### 3.3 新增了 `z_auto(r)` 自动高度策略

这是这次改造的第二个关键点。

我们没有把默认移动做成“固定 z”，而是做成：

- 中段半径维持一段较稳定的高度平台
- 再往前或往后，允许高度按规则变化
- 但优先满足“前臂尽量保持水平，其次安全”

它不是手写一个简单公式，而是：

1. 在 `r-z` 空间离线采样候选点
2. 对每个候选点做：
   - 圆柱范围验证
   - 可选的 `x/y` 工作区验证
   - IK 验证
   - 舵机 pulse 安全验证
3. 在合法点里按照以下优先级选点：
   - 前臂姿态尽量接近参考姿态
   - 高度尽量接近平台目标
   - 安全裕量尽量大
   - 相邻半径变化尽量平滑

好处是：

- 不会再出现“半径一收回就一路往下塌”的旧行为
- 机械臂 L 型结构在用户感知上更自然
- 把“机械几何限制”纳入了自动高度策略，而不是只凭经验常数

### 3.4 补上了真实底层舵机安全验证

这次实机调试里，我们发现 JetMax 官方底层还有一层“静默夹紧”：

- `serial_servo.set_position()` 会把 pulse 夹到 `[0, 1000]`
- 而 `JetMax.set_position()` 上层不会把所有越界都显式报出来

所以我们补了执行端自己的验证：

- `servo1` pulse 不能低于 `0` 或高于 `1000`
- `servo2` pulse 不能超出当前安全范围
- `servo3` pulse 不能低于 JetMax 实际安全下界

好处是：

- 目标点不再“看起来能到，实际都被底层夹成同一个位置”
- 避免把物理极限误判成软件控制成功
- 真机联调时能更早、更明确地发现“到底是谁在限制动作”

## 4. 这次优化带来的直接好处

### 4.1 控制语义更直观

原来你要想：

- `x = ?`
- `y = ?`
- `z = ?`

现在更多时候只要想：

- 转多少角
- 伸多远
- 抬到多高

对人来说，这比末端笛卡尔值更接近真实操作习惯。

### 4.2 更适合 JetMax 这种 3 自由度结构

JetMax 不是一个姿态完全独立可控的六轴工业臂。对这类小型 3 自由度机械臂来说：

- `r` 和 `z` 往往并不是完全独立的
- 真实可达空间更像某种受约束的 `r-z` 截面

圆柱主接口 + `z_auto(r)`，比“用户自己硬给 `x/y/z`”更符合它的结构特点。

### 4.3 更适合做脑控和半自动控制

MI / SSVEP 最终更适合表达的通常是：

- 左转 / 右转
- 前伸 / 后收
- 确认抓取 / 确认放置

而不是持续输出一个精确的 `x/y/z` 终点。

所以圆柱坐标主接口更容易和脑控语义对齐。

### 4.4 更容易把安全区描述清楚

以前安全限制主要写成：

- `x_min/x_max`
- `y_min/y_max`
- `z_min/z_max`

现在除了兼容旧限制外，还能直接表达：

- `theta` 允许范围
- `radius` 允许范围
- `z` 允许范围
- 自动高度 profile

这更贴近机械臂实际工作习惯，也更容易解释给后续维护者。

## 5. 和官方 JetMax 控制方式相比，有什么不同

### 官方方式

JetMax 官方资料主要展示的是两类路径：

1. **末端位姿控制**
   - 典型是 `jetmax.set_position((x, y, z), t)`
   - 这是抓取、抬升、搬运示例里的主流方式。[AI Vision Games Lesson](https://wiki.hiwonder.com/projects/JetMax/en/latest/docs/3_AI_Vision_Games_Lesson.html)

2. **舵机/脉宽直接控制**
   - 典型是 `hiwonder.serial_servo.set_position(id, pulse, ms)`
   - 官方高级教程明确给了这条底层路径。[Advanced Program Lessons](https://wiki.hiwonder.com/projects/JetMax/en/latest/docs/6_Advanced_Program_Lessons.html)

### 我们现在的方式

我们没有去修改官方底层包，而是在上面增加了一层：

- 用户和任务层优先用圆柱坐标
- 我们自己的执行端把圆柱目标做成底层控制内核
- 最后再调用官方 `set_position`

所以相比官方示例，我们的区别是：

- 官方更偏“直接写末端 xyz”
- 我们更偏“先用圆柱语义表达，再在系统内部转成 xyz”

这并不是和官方相反，而是**在官方控制栈之上包了一层更适合任务控制的运动学接口**。

## 6. 和网上/文献里的其他控制思路相比，有没有类似的

有，而且是相当常见的思路。

### 6.1 和圆柱/极坐标机器人工作空间表达类似

机器人学课程里，圆柱机器人本来就是用角度、半径和高度来描述工作空间的。对“绕固定底座转 + 水平伸缩 + 高度变化”的结构来说，用圆柱坐标表达工作空间和安全边界，本来就更自然。[IRIS Lab Robotics Notes](https://irislab.tech/course_robotics/lec2-dof/configuration.html)

我们这次并不是把 JetMax 机械结构硬改成“圆柱机器人”，而是借用了这种更自然的工作空间表达方式，让上层接口更符合操作者直觉。

### 6.2 和遥操作里的“位置/速率混合控制”类似

不少遥操作研究都不是让操作者直接给定完整的末端绝对位置，而是把：

- 大范围移动
- 精细定位

拆成不同层次的控制模式，或者用更适合人直觉的中间变量来描述目标。  
[A hybrid position–rate teleoperation system](https://www.sciencedirect.com/science/article/abs/pii/S092188902100066X) 这类工作就强调：当工作空间较大、任务包含粗调和细调时，混合位置/速率控制会比单一位置模式更自然。

我们的 `MOVE_CYL_AUTO` 也属于类似思路：

- 用户只管 `theta + r`
- 系统内部自动补 `z`
- 把复杂的几何约束和姿态约束隐藏在控制内核里

### 6.3 和“直接关节角控制”不同

网上也常见另一种思路：直接给关节角或舵机 pulse。

这种方式的优点是：

- 更接近底层
- 某些标定或诊断场景更直接

但对任务控制来说，它的问题也很明显：

- 不直观
- 难和抓取点、工作区、安全区对应
- 对后续 MI/SSVEP 这类语义输入不友好

所以我们没有把主路径改成关节角控制，而是保留它作为未来可能的辅助调试线路。

## 7. 当前这套方案的边界和限制

这次改造虽然已经把系统主接口圆柱化，但仍有几个边界必须说清楚：

### 7.1 最末端驱动仍然是官方 xyz

所以“底层圆柱化”的准确说法应该是：

- **我们的执行内核已经圆柱化**
- **官方底层驱动没有被替换**

### 7.2 `r-z` 不是完全独立

对 JetMax 这种结构来说：

- `r` 变小时，某些高度就不可达
- `r` 变大时，某些姿态又会逼近舵机极限

所以自动高度不是“可有可无的优化”，而是这类机械臂上层控制里必要的一部分。

### 7.3 圆柱角度上限不等于物理一定能到

我们在实机里已经验证过：

- 软件理论角度可以放很大
- 但真实可达仍然要受舵机 pulse 和机械极限限制

所以当前最合理的做法是：

- 对外保留清晰的 `theta/r/z` 语义
- 内部继续保留 IK 和舵机安全校验

## 8. 当前建议如何继续维护

后续如果继续优化，我建议遵循下面的原则：

1. 圆柱坐标保持主接口，不再回到“业务层直接写 `x/y`”。
2. legacy 笛卡尔命令继续保留，只做兼容，不做主路径。
3. `MOVE_CYL_AUTO` 继续以“姿态优先，其次安全”为主，不要退回到纯高度优先。
4. 真机联调时优先看：
   - `robot_cyl`
   - `control_kernel`
   - `auto_z_current`
   - `validation_error`
5. 如果后面真的需要更低层调试，再单独加 joint/pulse 调试线路，而不是把主控制逻辑降到舵机脉宽层。

## 9. 这份改造的最终定位

这次改造的本质不是“换一个公式”，而是把 JetMax 的主控制接口从：

- 以末端笛卡尔坐标为中心

改成：

- 以操作者直觉和机械几何更一致的圆柱语义为中心

同时又保留了：

- 旧笛卡尔命令兼容
- 官方底层驱动兼容
- 真机安全校验

所以它不是推翻式重写，而是一种**兼容官方控制栈的上层接口重构**。
