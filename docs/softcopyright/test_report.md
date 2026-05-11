# 测试报告草稿

## 测试对象

基于混合脑机接口的智能机械臂协同控制软件 V1.0，当前重点验证 `08_SoftCopyright_UI` 演示工作台、路径状态读取、材料清单和仓库级无硬件启动能力。

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

& $py .\08_SoftCopyright_UI\run_softcopyright_workbench.py --screenshot .\08_SoftCopyright_UI\artifacts\workbench.png
& $py .\08_SoftCopyright_UI\tools\render_static_preview.py --output .\08_SoftCopyright_UI\artifacts\workbench_static_preview.png
& $py -m brain diagnose
& $py -m brain launch --simulate
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
- UI 演示模式截图：已通过，输出 `08_SoftCopyright_UI/artifacts/workbench.png`。
- 静态材料截图：已通过，输出 `08_SoftCopyright_UI/artifacts/workbench_static_preview.png`。
- 仓库级无硬件诊断：已通过，`missing_paths=0`、`missing_modules=0`。
- 仓库级模拟启动：已通过，`launch_target=unified`、`simulate=true`。
