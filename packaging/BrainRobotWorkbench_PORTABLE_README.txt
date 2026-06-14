脑机机械臂一体化控制工作台 - 便携版说明

运行方式
1. 解压 BrainRobotWorkbench.zip。
2. 进入解压后的 BrainRobotWorkbench 文件夹。
3. 双击 BrainRobotWorkbench.exe。

移植到其他电脑时必须保留
- BrainRobotWorkbench.exe
- _internal 文件夹

不要只拷贝 exe。视觉模型、脑电库、机械臂运行脚本和配置文件都在 _internal 里。

真实设备联调前检查
1. 电脑需要连接机械臂 Wi-Fi。
2. 脑电帽串口驱动需要已安装，并能在设备管理器里看到 COM 口。
3. Windows 防火墙需要允许本程序访问专用网络。
4. 如果要自动拉起机械臂端程序，机械臂默认 SSH 用户名和密码需要保持为 hiwonder。

默认连接参数
- 机械臂地址: 192.168.149.1
- rosbridge 端口: 9091
- 机械臂 TCP 端口: 8888
- 脑电串口: auto

演示说明
- 界面按脑机控制流程展示。
- 当前演示控制可以由电脑端输入事件驱动，但界面不会显示键盘提示。
- 机械臂真实抓取仍依赖摄像头画面、视觉模型、标定文件和机械臂连接状态。
