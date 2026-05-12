# 测试报告草稿

## 测试对象

基于混合脑机接口的智能机械臂协同控制软件 V1.0，当前重点验证 `08_SoftCopyright_UI` 演示工作台、路径状态读取、材料清单和仓库级无硬件启动能力。

本报告当前状态为 **freeze-preview 候选冻结记录**，不是最终申报冻结记录。新 MI 分类器尚未并入，`datasets/profiles/MI/current_mi_profile.json` 和 `datasets/profiles/MI/mi_status.json` 尚未发布。

## 当前测试命令

```powershell
$py = & .\tools\resolve_brain_python.cmd
& $py -m py_compile `
  .\08_SoftCopyright_UI\run_softcopyright_workbench.py `
  .\08_SoftCopyright_UI\softcopyright_workbench\app.py `
  .\08_SoftCopyright_UI\softcopyright_workbench\state.py `
  .\08_SoftCopyright_UI\softcopyright_workbench\mi_contract.py `
  .\08_SoftCopyright_UI\tools\render_static_preview.py

& $py -m json.tool .\08_SoftCopyright_UI\schemas\mi_profile.schema.json > $null
& $py -m json.tool .\docs\softcopyright\source_manifest.draft.json > $null
& $py -m json.tool .\docs\softcopyright\source_manifest.freeze-preview.json > $null

& $py .\08_SoftCopyright_UI\run_softcopyright_workbench.py --screenshot .\08_SoftCopyright_UI\artifacts\workbench.png
& $py .\08_SoftCopyright_UI\tools\render_static_preview.py --output .\08_SoftCopyright_UI\artifacts\workbench_static_preview.png
& $py -m brain diagnose
& $py -m brain launch --simulate
```

## 当前候选专项 smoke

```powershell
$py = & .\tools\resolve_brain_python.cmd
& $py -m pytest .\01_MI\mi_classifier_latest\tests\test_mi_realtime_runtime.py .\01_MI\mi_classifier_latest\tests\test_training_aux_usage.py -q -o addopts=
& $py -m pytest .\02_SSVEP\tests\test_export_classifier_candidate_profile.py .\02_SSVEP\tests\test_backend_atomic_and_roundtrip.py -q -o addopts=
& $py -m pytest .\hybrid_controller\tests\test_low_height_alignment.py .\hybrid_controller\tests\test_low_height_center_search.py .\hybrid_controller\tests\test_continuous_vision_servo_controller.py -q -o addopts=
```

## 冻结前目标测试

```powershell
$py = & .\tools\resolve_brain_python.cmd
& $py -m pytest tests -q -o addopts=
& $py -m pytest .\02_SSVEP\tests -q -o addopts=
& $py -m pytest .\hybrid_controller\tests -q -o addopts=
```

失败项需分类为：UI 无关、硬件依赖、环境缺失、软著 V1.0 必须修复。MI 分类器正式并入后，应补充 MI 训练 smoke test、实时推理 smoke test 和 profile schema 校验。

## 当前实测结果

- UI 语法检查：已通过，覆盖 `run_softcopyright_workbench.py`、`app.py`、`state.py`、`mi_contract.py` 和 `render_static_preview.py`。
- JSON schema/清单检查：已通过，覆盖 `mi_profile.schema.json` 和 `source_manifest.draft.json`。
- Freeze-preview 清单检查：已通过，覆盖 `source_manifest.freeze-preview.json`。
- UI 演示模式截图：已通过，输出 `08_SoftCopyright_UI/artifacts/workbench.png`。
- 静态材料截图：已通过，输出 `08_SoftCopyright_UI/artifacts/workbench_static_preview.png`。
- 仓库级无硬件诊断：已通过，`missing_paths=0`、`missing_modules=0`。
- 仓库级模拟启动：已通过，`launch_target=unified`、`simulate=true`。
- MI 轻量测试：已通过，`9 passed`，覆盖旧入口相关实时推理与训练辅助测试。
- SSVEP profile/export 轻量测试：已通过，`7 passed`，覆盖 classifier candidate profile 导出和 backend roundtrip。
- hybrid_controller 低高度/连续视觉伺服轻量测试：已通过，`52 passed`，覆盖低高度视觉对齐、中心搜索和连续视觉伺服。

## 当前候选结论

- 当前候选可以作为软著 V1.0 freeze-preview 证据包继续整理。
- 当前候选不是最终申报冻结；正式 `softcopyright-v1.0` tag 应等 MI profile/status 发布并补齐最终测试后再创建。
- 软著 UI 仍保持只读，不触发真实机器人动作，不写入 profile。
