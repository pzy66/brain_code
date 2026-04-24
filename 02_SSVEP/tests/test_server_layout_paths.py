from pathlib import PurePosixPath

import pytest

from ssvep_core.server_layout import (
    DEFAULT_SERVER_BRAIN_ROOT,
    build_server_layout,
    normalize_server_root,
)


def test_server_layout_defaults_match_brain_root() -> None:
    plan = build_server_layout()
    assert plan.root == DEFAULT_SERVER_BRAIN_ROOT
    assert plan.ssvep_root == PurePosixPath("/data1/zkx/brain/ssvep")
    assert plan.mi_root == PurePosixPath("/data1/zkx/brain/mi")
    assert plan.ssvep_dirs == (
        PurePosixPath("/data1/zkx/brain/ssvep/code"),
        PurePosixPath("/data1/zkx/brain/ssvep/data"),
        PurePosixPath("/data1/zkx/brain/ssvep/reports"),
        PurePosixPath("/data1/zkx/brain/ssvep/profiles"),
        PurePosixPath("/data1/zkx/brain/ssvep/logs"),
    )
    assert plan.mi_dirs == ()


def test_server_layout_can_optionally_expand_mi_subdirs() -> None:
    plan = build_server_layout(include_mi_subdirs=True)
    assert plan.mi_dirs == (
        PurePosixPath("/data1/zkx/brain/mi/code"),
        PurePosixPath("/data1/zkx/brain/mi/data"),
        PurePosixPath("/data1/zkx/brain/mi/reports"),
        PurePosixPath("/data1/zkx/brain/mi/profiles"),
        PurePosixPath("/data1/zkx/brain/mi/logs"),
    )


def test_server_layout_rejects_paths_outside_data1_zkx() -> None:
    with pytest.raises(ValueError):
        normalize_server_root("/home/zhangkexin/brain")
