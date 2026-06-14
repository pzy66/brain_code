# 新电脑复制清单

## 推荐方式：复制整个仓库

把当前电脑的这个目录整体复制到新电脑：

```text
D:\brain\brain_code
```

新电脑上可以放在任意位置，例如：

```text
D:\brain\brain_code
C:\Users\<你>\Desktop\brain_code
D:\projects\brain_code
```

然后让新电脑 Codex 打开复制后的 `brain_code` 目录，先读：

```text
handoff/new_pc_joint_debug/README.md
```

## 为什么要复制整个仓库

完整联调不只是运行 exe，还可能需要让 Codex 检查：

- UI 流程代码
- 机械臂连接代码
- 摄像头读取代码
- 视觉识别配置
- EEG 串口显示链路
- PyInstaller 打包配置
- 机械臂端 runtime 脚本
- 你新建的 `新代码/control` 和 `新代码/UI`

这些都在仓库里。

## 如果只想先跑 exe

至少复制：

```text
dist/BrainRobotWorkbench.zip
handoff/new_pc_joint_debug/
```

然后在新电脑解压 `dist/BrainRobotWorkbench.zip`，运行：

```text
BrainRobotWorkbench/BrainRobotWorkbench.exe
```

如果是从 GitHub clone 仓库，默认不会带这个 exe zip；需要单独复制当前电脑的 `dist/BrainRobotWorkbench.zip`，或者在新电脑重新运行打包脚本。

## 不建议复制的东西

如果你手动精简仓库，可以不复制这些临时/缓存目录：

```text
.venv/
build/
dist_4targets/
build_4targets/
tmp/
.pytest_cache/
__pycache__/
```

但是不要删：

```text
dist/BrainRobotWorkbench.zip
handoff/new_pc_joint_debug/
robot_workbench/
hybrid_controller/
datasets/
packaging/
tools/
run_integrated_workbench.py
```

## 机械臂端代码

新电脑联调时，机械臂上也需要有：

```text
/home/hiwonder/brain_code/hybrid_controller/robot
```

如果机械臂端已经部署过，一般不用再复制。若 SSH 登录后发现目录不存在，再把仓库里的：

```text
hybrid_controller/robot
```

复制到机械臂：

```text
/home/hiwonder/brain_code/hybrid_controller/robot
```
