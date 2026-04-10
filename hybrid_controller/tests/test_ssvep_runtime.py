from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from hybrid_controller.config import AppConfig
from hybrid_controller.ssvep.backend import AsyncFbccaBackend
from hybrid_controller.ssvep.runtime import (
    SSVEPRuntime,
    build_timestamped_profile_path,
    infer_profile_timestamp,
)
from hybrid_controller.ssvep.profiles import ProfileStore


def test_build_timestamped_profile_path_uses_fixed_name_format(tmp_path: Path) -> None:
    path = build_timestamped_profile_path(tmp_path, timestamp=datetime(2026, 4, 9, 15, 16, 17))
    assert path == tmp_path / "ssvep_fbcca_profile_20260409_151617.json"


def test_infer_profile_timestamp_reads_filename_first(tmp_path: Path) -> None:
    path = tmp_path / "ssvep_fbcca_profile_20260409_180501.json"
    path.write_text("{}", encoding="utf-8")
    assert infer_profile_timestamp(path) == "20260409_180501"


def test_infer_profile_timestamp_falls_back_to_payload_saved_at(tmp_path: Path) -> None:
    path = tmp_path / "custom_profile.json"
    path.write_text(json.dumps({"saved_at": "20260409_200000"}), encoding="utf-8")
    assert infer_profile_timestamp(path) == "20260409_200000"


def test_runtime_bootstraps_current_profile_state_from_current_alias(tmp_path: Path) -> None:
    profile_dir = tmp_path / "dataset" / "ssvep_profiles"
    current_path = profile_dir / "current_fbcca_profile.json"
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_text(json.dumps({"saved_at": "20260409_210000"}), encoding="utf-8")
    config = AppConfig(
        ssvep_profile_dir=profile_dir,
        ssvep_current_profile_path=current_path,
        ssvep_default_profile_path=profile_dir / "default_fbcca_profile.json",
    )
    runtime = SSVEPRuntime(
        config,
        command_callback=lambda command: None,
        status_callback=lambda message: None,
    )
    assert runtime.current_profile_path == current_path.resolve()
    assert runtime.current_profile_source == "current"
    assert runtime.last_pretrain_time == "20260409_210000"


def test_profile_store_lists_current_then_history_desc_then_fallback(tmp_path: Path) -> None:
    profile_dir = tmp_path / "dataset" / "ssvep_profiles"
    current_path = profile_dir / "current_fbcca_profile.json"
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_text(json.dumps({"saved_at": "20260409_210000"}), encoding="utf-8")
    (profile_dir / "ssvep_fbcca_profile_20260409_190000.json").write_text("{}", encoding="utf-8")
    (profile_dir / "ssvep_fbcca_profile_20260409_220000.json").write_text("{}", encoding="utf-8")
    (profile_dir / "fallback_fbcca_profile.json").write_text("{}", encoding="utf-8")
    store = ProfileStore(profile_dir, current_path)

    summaries = store.list_profiles()

    assert [summary.kind for summary in summaries[:4]] == ["current", "history", "history", "fallback"]
    assert summaries[1].name == "ssvep_fbcca_profile_20260409_220000.json"
    assert summaries[2].name == "ssvep_fbcca_profile_20260409_190000.json"
    assert store.latest_profile() is not None
    assert store.latest_profile().name == "ssvep_fbcca_profile_20260409_220000.json"


def test_runtime_prefers_latest_profile_when_current_alias_missing(tmp_path: Path) -> None:
    profile_dir = tmp_path / "dataset" / "ssvep_profiles"
    current_path = profile_dir / "current_fbcca_profile.json"
    profile_dir.mkdir(parents=True, exist_ok=True)
    latest_path = profile_dir / "ssvep_fbcca_profile_20260409_221500.json"
    latest_path.write_text(json.dumps({"saved_at": "20260409_221500"}), encoding="utf-8")
    config = AppConfig(
        ssvep_profile_dir=profile_dir,
        ssvep_current_profile_path=current_path,
        ssvep_default_profile_path=profile_dir / "default_fbcca_profile.json",
        ssvep_auto_use_latest_profile=True,
    )
    runtime = SSVEPRuntime(
        config,
        command_callback=lambda command: None,
        status_callback=lambda message: None,
    )

    active_path, source = runtime._resolve_active_profile_for_online()

    assert source == "latest"
    assert active_path == latest_path.resolve()
    assert not current_path.exists()


def test_runtime_falls_back_when_no_profile_exists(tmp_path: Path) -> None:
    profile_dir = tmp_path / "dataset" / "ssvep_profiles"
    current_path = profile_dir / "current_fbcca_profile.json"
    config = AppConfig(
        ssvep_profile_dir=profile_dir,
        ssvep_current_profile_path=current_path,
        ssvep_default_profile_path=profile_dir / "default_fbcca_profile.json",
        ssvep_auto_use_latest_profile=True,
        ssvep_allow_fallback_profile=True,
    )
    runtime = SSVEPRuntime(
        config,
        command_callback=lambda command: None,
        status_callback=lambda message: None,
    )

    active_path, source = runtime._resolve_active_profile_for_online()

    assert source == "fallback"
    assert active_path.name == "fallback_fbcca_profile.json"
    assert active_path.exists()


def test_manual_profile_selection_is_session_only(tmp_path: Path) -> None:
    profile_dir = tmp_path / "dataset" / "ssvep_profiles"
    current_path = profile_dir / "current_fbcca_profile.json"
    profile_dir.mkdir(parents=True, exist_ok=True)
    latest_path = profile_dir / "ssvep_fbcca_profile_20260409_231000.json"
    manual_path = profile_dir / "ssvep_fbcca_profile_20260409_220000.json"
    latest_path.write_text(json.dumps({"saved_at": "20260409_231000"}), encoding="utf-8")
    manual_path.write_text(json.dumps({"saved_at": "20260409_220000"}), encoding="utf-8")
    config = AppConfig(
        ssvep_profile_dir=profile_dir,
        ssvep_current_profile_path=current_path,
        ssvep_default_profile_path=profile_dir / "default_fbcca_profile.json",
        ssvep_auto_use_latest_profile=True,
    )
    runtime = SSVEPRuntime(
        config,
        command_callback=lambda command: None,
        status_callback=lambda message: None,
    )

    runtime.load_profile_from_path(manual_path)
    active_path, source = runtime._resolve_active_profile_for_online()
    assert source == "session"
    assert active_path == manual_path.resolve()

    runtime.clear_session_profile()
    active_path_2, source_2 = runtime._resolve_active_profile_for_online()
    assert source_2 == "latest"
    assert active_path_2 == latest_path.resolve()


def test_runtime_uses_default_profile_when_available_and_no_current_or_latest(tmp_path: Path) -> None:
    profile_dir = tmp_path / "dataset" / "ssvep_profiles"
    current_path = profile_dir / "current_fbcca_profile.json"
    default_path = profile_dir / "default_fbcca_profile.json"
    profile_dir.mkdir(parents=True, exist_ok=True)
    backend = AsyncFbccaBackend()
    backend.save_profile(backend.default_profile((8.0, 10.0, 12.0, 15.0)), default_path)
    config = AppConfig(
        ssvep_profile_dir=profile_dir,
        ssvep_current_profile_path=current_path,
        ssvep_default_profile_path=default_path,
        ssvep_auto_use_latest_profile=True,
        ssvep_prefer_default_profile=True,
        ssvep_allow_fallback_profile=False,
    )
    runtime = SSVEPRuntime(
        config,
        command_callback=lambda command: None,
        status_callback=lambda message: None,
    )

    active_path, source = runtime._resolve_active_profile_for_online()

    assert source == "default"
    assert active_path == default_path.resolve()
