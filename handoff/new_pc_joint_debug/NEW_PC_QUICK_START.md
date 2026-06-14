# 新电脑联调快速启动

## 复制什么到新电脑

推荐直接复制整个当前仓库：

```text
D:\brain\brain_code
```

新电脑上 Codex 先读：

```text
handoff/new_pc_joint_debug/
```

如果只想先跑 exe，至少复制：

```text
dist/BrainRobotWorkbench.zip
handoff/new_pc_joint_debug/
```

## 新电脑运行 exe

1. 解压 `dist/BrainRobotWorkbench.zip`。
2. 进入解压出的 `BrainRobotWorkbench` 文件夹。
3. 双击 `BrainRobotWorkbench.exe`。

必须保留同级 `_internal` 文件夹，不要只复制 exe。

## 机械臂端手动启动

如果软件提示 SSH 自动启动机械臂失败，先手动 SSH：

```powershell
ssh hiwonder@192.168.149.1
```

密码默认：

```text
hiwonder
```

进入机械臂后执行：

```bash
cd /home/hiwonder/brain_code/hybrid_controller/robot
pkill -f hybrid_controller_runtime_node.py || true
bash run_hybrid_controller_ros_runtime.sh
```

展示时建议后台运行：

```bash
cd /home/hiwonder/brain_code/hybrid_controller/robot
pkill -f hybrid_controller_runtime_node.py || true
nohup bash run_hybrid_controller_ros_runtime.sh > ~/hybrid_controller_runtime.log 2>&1 &
tail -f ~/hybrid_controller_runtime.log
```

## 新电脑快速诊断

在 `handoff/new_pc_joint_debug/scripts` 目录打开 PowerShell：

```powershell
.\diagnose_new_pc.ps1
```

重点看：

- `192.168.149.1 ping` 是否成功。
- `22 SSH` 是否通。
- `9091 rosbridge` 是否通。
- `8080 camera` 是否通。
- 是否能看到脑电帽 COM 口。

## Codex 调试入口

在新电脑打开 Codex 时，工作目录指向：

```text
brain_code
```

先让 Codex 阅读：

```text
handoff/new_pc_joint_debug/CODEX_HANDOFF.md
handoff/new_pc_joint_debug/NEW_PC_QUICK_START.md
```

然后再调试 exe、机械臂连接、摄像头和 EEG 串口。
