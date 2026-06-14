# GitHub 同步说明

当前联调源码会推到 GitHub 分支：

```text
codex/integrated-workbench-robot-debug
```

新电脑可以拉这个分支继续让 Codex 联调：

```powershell
git clone https://github.com/pzy66/brain_code.git
cd brain_code
git switch codex/integrated-workbench-robot-debug
```

## 注意 exe

普通 GitHub Git 仓库不提交 `dist/BrainRobotWorkbench.zip`，因为当前便携包约 405 MB，超过普通源码仓库适合承载的范围。

所以新电脑有两种方式：

1. 只联调源码和硬件链路：直接 clone 这个分支。
2. 要直接运行 exe：单独复制当前电脑的 `dist/BrainRobotWorkbench.zip`，或者在新电脑装好环境后运行：

```powershell
.\tools\build_integrated_workbench.ps1 -NoInstall
```

## 新电脑 Codex 首读

```text
handoff/new_pc_joint_debug/README.md
handoff/new_pc_joint_debug/CODEX_HANDOFF.md
handoff/new_pc_joint_debug/TROUBLESHOOTING.md
```
