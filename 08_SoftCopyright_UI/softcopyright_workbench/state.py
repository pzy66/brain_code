from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from brain_workspace.paths import (
    BRAIN_CODE_ROOT,
    DATASETS_ROOT,
    HYBRID_CONTROLLER_DIR,
    MI_DATASET_DIR,
    PROFILE_DATASET_DIR,
    SSVEP_DATASET_DIR,
    SSVEP_PROJECT_DIR,
    VISION_DATASET_DIR,
)

from .mi_contract import MiContractStatus, assess_mi_contract


SOFTCOPYRIGHT_DOC_DIR = BRAIN_CODE_ROOT / "docs" / "softcopyright"


@dataclass(frozen=True)
class StatusCard:
    name: str
    state: str
    detail: str
    level: str


@dataclass(frozen=True)
class ArtifactRow:
    item: str
    status: str
    path: str
    owner: str


@dataclass(frozen=True)
class MaterialStatus:
    title: str
    path: Path
    exists: bool
    owner: str
    freeze_required: bool = True

    @property
    def status(self) -> str:
        if not self.exists:
            return "缺失"
        return "待冻结" if self.freeze_required else "可用"

    @property
    def level(self) -> str:
        if not self.exists:
            return "bad"
        return "warn" if self.freeze_required else "good"


@dataclass(frozen=True)
class WorkbenchState:
    repo_root: Path
    datasets_root: Path
    mi_contract: MiContractStatus
    mi_dataset_dir: Path
    ssvep_entry: Path
    ssvep_entry_exists: bool
    ssvep_dataset_dir: Path
    ssvep_profile: Path
    ssvep_profile_exists: bool
    ssvep_report_dir: Path
    ssvep_report_exists: bool
    vision_model: Path
    vision_model_exists: bool
    vision_profile: Path
    vision_profile_exists: bool
    hybrid_entry: Path
    hybrid_entry_exists: bool
    hybrid_config: Path
    hybrid_config_exists: bool
    profile_root: Path
    materials: tuple[MaterialStatus, ...]

    def status_cards(self) -> list[StatusCard]:
        mi_level = {
            "published": "good",
            "training": "warn",
            "ready": "warn",
            "legacy_detected": "warn",
            "missing": "bad",
            "error": "bad",
        }.get(self.mi_contract.state, "neutral")
        mi_state = {
            "published": "已发布",
            "training": "训练中",
            "ready": "待发布",
            "legacy_detected": "旧版入口",
            "missing": "缺失",
            "error": "错误",
        }.get(self.mi_contract.state, self.mi_contract.state)
        material_count = sum(1 for item in self.materials if item.exists)
        material_level = "good" if material_count == len(self.materials) else "warn"
        return [
            StatusCard("MI", mi_state, self.mi_contract.detail, mi_level),
            StatusCard(
                "SSVEP",
                "可用" if self.ssvep_entry_exists else "缺入口",
                f"{short_path(self.ssvep_entry)}；profile: {'已存在' if self.ssvep_profile_exists else '待发布'}",
                "good" if self.ssvep_entry_exists and self.ssvep_profile_exists else "warn",
            ),
            StatusCard(
                "视觉",
                "有基线" if self.vision_model_exists else "缺模型",
                f"{short_path(self.vision_model)}；抓取 profile: {'已存在' if self.vision_profile_exists else '待补齐'}",
                "good" if self.vision_model_exists and self.vision_profile_exists else "warn",
            ),
            StatusCard(
                "机械臂",
                "可定位" if self.hybrid_entry_exists else "缺入口",
                f"{short_path(self.hybrid_entry)}；真实 MOVE/PICK/PLACE 仍由 hybrid_controller 安全门控执行",
                "good" if self.hybrid_entry_exists else "bad",
            ),
            StatusCard(
                "材料",
                f"{material_count}/{len(self.materials)}",
                "软著说明书、用户手册、测试报告、源码交存边界和版本说明的草稿状态",
                material_level,
            ),
        ]

    def artifact_rows(self) -> list[ArtifactRow]:
        rows = [
            ArtifactRow("MI profile schema", "可用" if self.mi_contract.schema_exists else "缺失", short_path(self.mi_contract.schema_path), "UI/MI"),
            ArtifactRow("MI status file", "可用" if self.mi_contract.status_exists else "可选", short_path(self.mi_contract.status_path), "MI"),
            ArtifactRow("MI smoke test", "可用" if self.mi_contract.smoke_test else "待补齐", short_path(self.mi_contract.smoke_test) if self.mi_contract.smoke_test else "01_MI/mi_classifier_latest/tests/", "MI"),
            ArtifactRow("MI current profile", "已发布" if self.mi_contract.profile_exists else "待发布", short_path(self.mi_contract.profile_path), "MI"),
            ArtifactRow("SSVEP profile", "可用" if self.ssvep_profile_exists else "待发布", short_path(self.ssvep_profile), "SSVEP"),
            ArtifactRow("视觉抓取 profile", "可用" if self.vision_profile_exists else "待补齐", short_path(self.vision_profile), "视觉机械臂"),
            ArtifactRow("默认视觉模型", "可用" if self.vision_model_exists else "缺失", short_path(self.vision_model), "视觉机械臂"),
        ]
        for material in self.materials:
            rows.append(ArtifactRow(material.title, material.status, short_path(material.path), material.owner))
        rows.extend(
            [
                ArtifactRow("MI 引用资料", "可用" if (BRAIN_CODE_ROOT / "references" / "MI" / "README.md").exists() else "待补齐", "references/MI/README.md", "算法"),
                ArtifactRow("SSVEP 引用资料", "可用" if (BRAIN_CODE_ROOT / "references" / "SSVEP").exists() else "待补齐", "references/SSVEP/", "算法"),
                ArtifactRow("视觉抓取引用资料", "可用" if (BRAIN_CODE_ROOT / "references" / "Vision_Grasp").exists() else "待补齐", "references/Vision_Grasp/", "视觉机械臂"),
                ArtifactRow("运行截图", "可生成", "08_SoftCopyright_UI/artifacts/", "UI"),
            ]
        )
        return rows


def short_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(BRAIN_CODE_ROOT))
    except (OSError, ValueError):
        return str(path)


def load_json_summary(path: Path, keys: tuple[str, ...]) -> str:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return ""
    values: list[str] = []
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        values.append(f"{key}={value}")
    return ", ".join(values)


def build_material_statuses() -> tuple[MaterialStatus, ...]:
    items = (
        ("软著规划文档", "softcopyright_planning.md", "工程"),
        ("软件说明书", "software_manual.md", "软著材料"),
        ("用户手册", "user_manual.md", "软著材料"),
        ("测试报告", "test_report.md", "验证"),
        ("源码交存清单", "source_deposit_scope.md", "工程"),
        ("源码冻结规则", "source_manifest.draft.json", "工程"),
        ("版本说明", "version_notes.md", "工程"),
    )
    return tuple(
        MaterialStatus(title=title, path=SOFTCOPYRIGHT_DOC_DIR / filename, exists=(SOFTCOPYRIGHT_DOC_DIR / filename).exists(), owner=owner)
        for title, filename, owner in items
    )


def collect_workbench_state() -> WorkbenchState:
    ssvep_entry = SSVEP_PROJECT_DIR / "START_SSVEP.py"
    ssvep_profile = PROFILE_DATASET_DIR / "hybrid_controller" / "ssvep_profiles" / "current_fbcca_profile.json"
    ssvep_report_dir = SSVEP_PROJECT_DIR / "reports"
    vision_model = VISION_DATASET_DIR / "models" / "best.pt"
    vision_profile = PROFILE_DATASET_DIR / "hybrid_controller" / "vision_grasp" / "current_grasp_profile.json"
    hybrid_entry = HYBRID_CONTROLLER_DIR / "run_real.py"
    hybrid_config = HYBRID_CONTROLLER_DIR / "config.py"
    return WorkbenchState(
        repo_root=BRAIN_CODE_ROOT,
        datasets_root=DATASETS_ROOT,
        mi_contract=assess_mi_contract(),
        mi_dataset_dir=MI_DATASET_DIR,
        ssvep_entry=ssvep_entry,
        ssvep_entry_exists=ssvep_entry.exists(),
        ssvep_dataset_dir=SSVEP_DATASET_DIR,
        ssvep_profile=ssvep_profile,
        ssvep_profile_exists=ssvep_profile.exists(),
        ssvep_report_dir=ssvep_report_dir,
        ssvep_report_exists=ssvep_report_dir.exists(),
        vision_model=vision_model,
        vision_model_exists=vision_model.exists(),
        vision_profile=vision_profile,
        vision_profile_exists=vision_profile.exists(),
        hybrid_entry=hybrid_entry,
        hybrid_entry_exists=hybrid_entry.exists(),
        hybrid_config=hybrid_config,
        hybrid_config_exists=hybrid_config.exists(),
        profile_root=PROFILE_DATASET_DIR,
        materials=build_material_statuses(),
    )
