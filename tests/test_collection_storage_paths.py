from __future__ import annotations

from pathlib import Path

import pytest

from brain_workspace.bootstrap import configure_qt_offscreen
from brain_workspace.paths import (
    BRAIN_CODE_ROOT,
    MI_DATASET_DIR,
    MI_PROJECT_DIR,
    SSVEP_DATASET_DIR,
    SSVEP_PROJECT_DIR,
    path_is_relative_to,
)

configure_qt_offscreen()

from apps.data_collection_ui import DEFAULT_DATASET_DIR, resolve_dataset_dir  # noqa: E402
from mi_data_collector import DEFAULT_OUTPUT_ROOT, build_initial_config_from_args, resolve_output_root  # noqa: E402
from unified_collection.paths import DEFAULT_MI_OUTPUT_ROOT, DEFAULT_SSVEP_DATASET_DIR, resolve_ssvep_dataset_dir  # noqa: E402


def test_collection_default_storage_roots_are_inside_brain_code() -> None:
    assert DEFAULT_OUTPUT_ROOT == MI_DATASET_DIR
    assert DEFAULT_MI_OUTPUT_ROOT == MI_DATASET_DIR
    assert DEFAULT_DATASET_DIR == SSVEP_DATASET_DIR
    assert DEFAULT_SSVEP_DATASET_DIR == SSVEP_DATASET_DIR

    for path in (DEFAULT_OUTPUT_ROOT, DEFAULT_MI_OUTPUT_ROOT, DEFAULT_DATASET_DIR, DEFAULT_SSVEP_DATASET_DIR):
        assert path_is_relative_to(Path(path), BRAIN_CODE_ROOT)


def test_mi_relative_output_root_is_project_relative() -> None:
    resolved = resolve_output_root("datasets/custom_mi")

    assert resolved == (MI_PROJECT_DIR / "datasets" / "custom_mi").resolve()
    assert path_is_relative_to(resolved, BRAIN_CODE_ROOT)


def test_mi_cli_output_root_is_normalized_before_window_config() -> None:
    class Args:
        synthetic = False
        serial_port = ""
        output_root = "datasets/custom_mi"
        subject_id = ""
        session_id = ""

    config = build_initial_config_from_args(Args())

    assert Path(config["output_root"]) == MI_DATASET_DIR.resolve()


def test_ssvep_relative_dataset_dir_is_project_relative() -> None:
    resolved = resolve_dataset_dir("artifacts/datasets")

    assert resolved == (SSVEP_PROJECT_DIR / "artifacts" / "datasets").resolve()
    assert path_is_relative_to(resolved, BRAIN_CODE_ROOT)


def test_unified_ssvep_relative_dataset_dir_is_project_relative() -> None:
    resolved = resolve_ssvep_dataset_dir("artifacts/datasets")

    assert resolved == SSVEP_DATASET_DIR.resolve()
    assert path_is_relative_to(resolved, BRAIN_CODE_ROOT)


def test_collection_ui_rejects_external_storage_roots() -> None:
    external = BRAIN_CODE_ROOT.parent / "_outside_brain_code_collection_path"

    with pytest.raises(ValueError, match="inside brain_code"):
        resolve_output_root(external)

    with pytest.raises(ValueError, match="inside brain_code"):
        resolve_dataset_dir(external)

    with pytest.raises(ValueError, match="inside brain_code"):
        resolve_ssvep_dataset_dir(external)
