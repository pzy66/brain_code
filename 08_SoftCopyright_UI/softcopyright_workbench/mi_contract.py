from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from brain_workspace.paths import BRAIN_CODE_ROOT, MI_PROJECT_DIR, PROFILE_DATASET_DIR


MI_PROFILE_PATH = PROFILE_DATASET_DIR / "MI" / "current_mi_profile.json"
MI_STATUS_PATH = PROFILE_DATASET_DIR / "MI" / "mi_status.json"
MI_PROFILE_SCHEMA_PATH = BRAIN_CODE_ROOT / "08_SoftCopyright_UI" / "schemas" / "mi_profile.schema.json"

TRAIN_ENTRY_CANDIDATES = (
    MI_PROJECT_DIR / "run_02_training.py",
    MI_PROJECT_DIR / "code" / "training" / "train_custom_dataset.py",
)
REALTIME_ENTRY_CANDIDATES = (
    MI_PROJECT_DIR / "run_03_realtime_infer.py",
    MI_PROJECT_DIR / "code" / "realtime" / "mi_realtime_infer_only.py",
)
COLLECTION_ENTRY_CANDIDATES = (
    MI_PROJECT_DIR / "run_01_collection_only.py",
    MI_PROJECT_DIR / "code" / "collection" / "mi_data_collector.py",
)
MODEL_OUTPUT_CANDIDATES = (
    MI_PROJECT_DIR / "models",
    PROFILE_DATASET_DIR / "MI",
)
SMOKE_TEST_CANDIDATES = (
    MI_PROJECT_DIR / "tests" / "test_mi_realtime_runtime.py",
    MI_PROJECT_DIR / "tests" / "test_training_aux_usage.py",
)


@dataclass(frozen=True)
class MiContractStatus:
    state: str
    detail: str
    train_entry: Path | None
    realtime_entry: Path | None
    collection_entry: Path | None
    smoke_test: Path | None
    profile_path: Path
    profile_exists: bool
    status_path: Path
    status_exists: bool
    schema_path: Path
    schema_exists: bool


def first_existing_path(candidates: tuple[Path, ...]) -> Path | None:
    for path in candidates:
        try:
            if path.exists():
                return path
        except OSError:
            continue
    return None


def assess_mi_contract() -> MiContractStatus:
    train_entry = first_existing_path(TRAIN_ENTRY_CANDIDATES)
    realtime_entry = first_existing_path(REALTIME_ENTRY_CANDIDATES)
    collection_entry = first_existing_path(COLLECTION_ENTRY_CANDIDATES)
    smoke_test = first_existing_path(SMOKE_TEST_CANDIDATES)
    profile_exists = MI_PROFILE_PATH.exists()
    status_exists = MI_STATUS_PATH.exists()
    schema_exists = MI_PROFILE_SCHEMA_PATH.exists()

    if status_exists:
        state = "training"
        detail = "检测到 MI 状态文件；UI 将其视为训练/发布流程正在由 MI 模块管理。"
    elif profile_exists and train_entry and realtime_entry and schema_exists:
        state = "published"
        detail = "MI 训练、实时推理、profile 和 schema 已可被 UI 识别。"
    elif train_entry and realtime_entry and schema_exists and smoke_test:
        state = "ready"
        detail = "已有 MI 训练/实时入口、profile schema 和 smoke test，等待发布 current_mi_profile.json。"
    elif train_entry or realtime_entry or collection_entry:
        state = "legacy_detected"
        detail = "检测到旧版 MI 入口；新分类器入库时需补齐统一 profile/schema/smoke 契约。"
    else:
        state = "missing"
        detail = "未检测到 MI 训练或实时推理入口。"

    return MiContractStatus(
        state=state,
        detail=detail,
        train_entry=train_entry,
        realtime_entry=realtime_entry,
        collection_entry=collection_entry,
        smoke_test=smoke_test,
        profile_path=MI_PROFILE_PATH,
        profile_exists=profile_exists,
        status_path=MI_STATUS_PATH,
        status_exists=status_exists,
        schema_path=MI_PROFILE_SCHEMA_PATH,
        schema_exists=schema_exists,
    )
