# Pretrain UI

面向用户的一键脑机接口预训练前端。

启动方式：

```powershell
.\START_PRETRAIN_UI.cmd
```

或：

```powershell
$py = .\tools\resolve_brain_python.cmd
& $py run_pretrain_ui.py
```

当前版本的预训练仍是前端 dry-run 进度，不接真实训练后端；机械臂控制页已经接入摄像头读取、TCP 机械臂指令和离线模拟降级。后续可以从 `PretrainWindow.start_pretrain()` 和阶段推进逻辑继续接入 SSVEP / MI 采集与训练任务。

预训练完成后会进入机械臂控制页：

- 摄像头默认读取 JetMax / Hiwonder 的 MJPEG 流。
- 点击“连接”后使用默认机械臂地址发送指令。
- `W / A / S / D` 控制机械臂半径与角度微调，用来临时代替 MI。
- `1 / 2 / 3 / 4` 选择小木块，用来临时代替 SSVEP。
- “确认抓取”会向当前选中木块发送 `PICK_CYL` 指令。

没有硬件或测试环境下会自动进入离线模拟，UI 仍可完整走通。
