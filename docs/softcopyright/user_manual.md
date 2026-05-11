# 用户手册草稿

## 启动方式

在仓库根目录运行：

```powershell
$py = & .\tools\resolve_brain_python.cmd
& $py .\08_SoftCopyright_UI\run_softcopyright_workbench.py
```

生成无硬件截图：

```powershell
$py = & .\tools\resolve_brain_python.cmd
& $py .\08_SoftCopyright_UI\run_softcopyright_workbench.py --screenshot .\08_SoftCopyright_UI\artifacts\workbench.png
```

如果 Qt offscreen 字体异常，使用静态预览：

```powershell
C:\Users\P1233\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  .\08_SoftCopyright_UI\tools\render_static_preview.py `
  --output .\08_SoftCopyright_UI\artifacts\workbench_static_preview.png
```

## 页面说明

- 总览：查看 MI、SSVEP、视觉、机械臂和材料的当前状态。
- 数据采集：定位 MI 与 SSVEP 采集入口，确认数据目录和 profile 路径。
- 训练评估：查看模型发布产物、验收指标和发布 gate。
- 在线控制：查看键盘、MI、SSVEP 的仲裁关系和 idle/no-control 规则。
- 视觉机械臂：查看视觉模型、抓取 profile、hybrid_controller 入口和只读启动命令。
- 软著材料：查看软著材料草稿、引用资料、截图目录和冻结命令提示。

## 安全说明

软著 UI 的按钮默认只执行三类动作：打开目录、定位文件、显示命令。该 UI 不直接发送机器人 MOVE、PICK、PLACE 或 ABORT 指令，不写入真实 profile，也不要求硬件在线。

真实机器人执行必须回到 `hybrid_controller` 的安全门控流程，按 dry-run、camera-only、resolve-only、move-only、execute-move、allow-pick 的阶段逐步验证。
