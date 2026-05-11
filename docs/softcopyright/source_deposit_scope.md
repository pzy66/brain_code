# 源码交存清单草稿

## 纳入范围

- 自有源码：`brain_workspace/`、`brain/`、`01_MI/`、`02_SSVEP/`、`08_SoftCopyright_UI/`、`hybrid_controller/` 中属于本项目实现的源码。
- UI 与材料：`08_SoftCopyright_UI/`、`docs/softcopyright/`。
- 配置与 schema：用于演示、profile 读取和软件运行的配置文件、`08_SoftCopyright_UI/schemas/mi_profile.schema.json`。
- Profile 样例或默认 profile：可公开的 schema、默认配置和轻量 profile。真实训练权重默认不进 Git，除非明确作为基线资产。
- 测试：`tests/`、`02_SSVEP/tests/`、`hybrid_controller/tests/` 和后续 MI smoke tests。
- 引用说明：`references/` 下的可公开 README、bibliography、方法说明和资料索引。

## 排除范围

- 真实 EEG 原始数据、个人实验数据、外部公开数据集原始文件。
- 论文 PDF、授权受限资料和不可再分发材料。
- 训练输出、模型中间产物、大体积权重、日志、缓存、临时图片、临时导出包。
- 机器人现场调试日志、摄像头临时截图和非 V1.0 冻结所需的运行痕迹。

## 冻结流程

1. 确认软著 UI、MI 契约、SSVEP profile、视觉抓取 profile 和材料草稿全部可定位。
2. 更新测试报告，记录实际命令、结果、失败分类和截图路径。
3. 更新 `source_manifest.draft.json`，记录提交号、纳入路径、排除路径和测试命令。
4. 创建 `softcopyright-v1.0` tag。
5. 对冻结清单做只读归档，后续开发从新分支继续。
