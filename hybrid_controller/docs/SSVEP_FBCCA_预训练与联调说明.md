# SSVEP FBCCA 预训练与联调说明

## 目标

主界面内置 SSVEP 识别与预训练，满足以下调试需求：

- 启动程序时不依赖脑电设备，不会自动尝试连接串口
- 可单独连接脑电设备
- 可单独开始/停止在线识别
- 没有预训练也能按策略启动（默认 profile 或 fallback）
- Profile 多文件可选择，并支持自动策略
- 键盘 `1/2/3/4 + Enter/c + Esc/x` 与 SSVEP 并行调试

## 主界面按钮

右侧 SSVEP 区固定包含：

- `连接设备`
- `开始预训练`
- `加载选中`
- `打开 Profile 目录`
- `开始 SSVEP 识别`
- `停止 SSVEP 识别`

行为约定：

- 点击 `连接设备` 时，按 auto 串口模式搜索并尝试候选串口
- `停止 SSVEP 识别`：仅停止在线解码，不会关闭视觉检测
- 停止后：画面继续显示检测框和 ROI；频闪叠加关闭

## Profile 目录与命名

统一目录：

- [dataset/ssvep_profiles](C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\dataset\ssvep_profiles)

文件约定：

- 预训练历史：`ssvep_fbcca_profile_YYYYMMDD_HHMMSS.json`
- 当前别名：`current_fbcca_profile.json`
- 默认通用 profile（可选）：`default_fbcca_profile.json`
- fallback：`fallback_fbcca_profile.json`

## 在线识别 profile 选择策略

启动在线识别时，按以下顺序选择 profile：

1. 会话手动选择的 profile（本次会话有效）
2. 最新兼容历史 profile（自动模式）
3. `current_fbcca_profile.json`
4. `default_fbcca_profile.json`（存在且兼容）
5. `fallback_fbcca_profile.json`（仅 `ssvep_allow_fallback_profile=True`）

说明：

- 兼容性检查包含 `model_name` 与 `freqs`
- 若 fallback 被禁用且前四项都不可用，启动会返回错误

## 无预训练时如何处理

推荐顺序：

1. 先点 `连接设备`
2. 若可预训练，优先做一次预训练
3. 若暂时不预训练，可直接开始在线识别，让系统走 `default` 或 `fallback` 路径

工程建议：

- `default_fbcca_profile.json` 可放一份现场验证较稳的“通用基线”
- 首次上机建议仍尽快做个人预训练，替代通用 profile

## 多 profile 情况如何选择

- 下拉框可选：
  - `自动（最新训练）`
  - 历史 profile 列表（按展示名）
- 点 `加载选中` 后生效
- 选择“自动”并加载，会清除会话手动选择，回到自动策略

## 常见联调流程

1. 打开主程序：`python run_real_ssvep.py`
2. 点击 `连接设备`
3. 视情况：
   - 直接 `开始 SSVEP 识别`（default/fallback）
   - 或先 `开始预训练`，完成后再启动识别
4. 联调期间可并行使用键盘事件验证状态机
5. 点击 `停止 SSVEP 识别` 结束在线解码

## 关键配置项

见 [config.py](C:\Users\P1233\Desktop\brain\brain_code\hybrid_controller\config.py)：

- `ssvep_profile_dir`
- `ssvep_current_profile_path`
- `ssvep_default_profile_path`
- `ssvep_auto_use_latest_profile`
- `ssvep_prefer_default_profile`
- `ssvep_allow_fallback_profile`
- `ssvep_keyboard_debug_enabled`
