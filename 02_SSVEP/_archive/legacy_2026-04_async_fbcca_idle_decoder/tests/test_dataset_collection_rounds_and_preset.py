from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_dataset_collection_ui import (
    CollectionConfig,
    CollectionWorker,
    DEFAULT_STABLE_ACTIVE_SEC,
    DEFAULT_STABLE_IDLE_REPEATS,
    DEFAULT_STABLE_LONG_IDLE_SEC,
    DEFAULT_STABLE_PREPARE_SEC,
    DEFAULT_STABLE_REST_SEC,
    DEFAULT_STABLE_SWITCH_TRIALS,
    DEFAULT_STABLE_TARGET_REPEATS,
    MIN_ACTIVE_SEC_FOR_TRAINING,
    ENHANCED_45M_PRESET,
    STABLE_12M_PRESET,
    _build_round_session_id,
    _resolve_cli_protocol,
    _validate_collection_protocol,
    estimate_round_seconds,
    trial_count_for_protocol,
)
from ssvep_core.dataset import (
    CollectionProtocol,
    build_collection_trials,
    build_protocol_signature,
    save_collection_dataset_bundle,
)


def test_stable_12m_round_estimate_matches_plan() -> None:
    trials = trial_count_for_protocol(
        target_repeats=DEFAULT_STABLE_TARGET_REPEATS,
        idle_repeats=DEFAULT_STABLE_IDLE_REPEATS,
        switch_trials=DEFAULT_STABLE_SWITCH_TRIALS,
    )
    assert trials == 74
    round_sec = estimate_round_seconds(
        prepare_sec=DEFAULT_STABLE_PREPARE_SEC,
        active_sec=DEFAULT_STABLE_ACTIVE_SEC,
        rest_sec=DEFAULT_STABLE_REST_SEC,
        target_repeats=DEFAULT_STABLE_TARGET_REPEATS,
        idle_repeats=DEFAULT_STABLE_IDLE_REPEATS,
        switch_trials=DEFAULT_STABLE_SWITCH_TRIALS,
        long_idle_sec=DEFAULT_STABLE_LONG_IDLE_SEC,
    )
    assert abs(float(round_sec) - 740.0) < 1e-9


def test_resolve_cli_protocol_uses_preset_values() -> None:
    name, prepare, active, rest, long_idle, target, idle, switch = _resolve_cli_protocol(
        preset_name=ENHANCED_45M_PRESET.key,
        prepare_sec=9.0,
        active_sec=9.0,
        rest_sec=9.0,
        long_idle_sec=91.0,
        target_repeats=1,
        idle_repeats=1,
        switch_trials=1,
    )
    assert name == ENHANCED_45M_PRESET.key
    assert abs(float(prepare) - float(ENHANCED_45M_PRESET.prepare_sec)) < 1e-9
    assert abs(float(active) - float(ENHANCED_45M_PRESET.active_sec)) < 1e-9
    assert abs(float(rest) - float(ENHANCED_45M_PRESET.rest_sec)) < 1e-9
    assert abs(float(long_idle) - float(ENHANCED_45M_PRESET.long_idle_sec)) < 1e-9
    assert int(target) == int(ENHANCED_45M_PRESET.target_repeats)
    assert int(idle) == int(ENHANCED_45M_PRESET.idle_repeats)
    assert int(switch) == int(ENHANCED_45M_PRESET.switch_trials)


def test_resolve_cli_protocol_custom_preserves_manual_values() -> None:
    name, prepare, active, rest, long_idle, target, idle, switch = _resolve_cli_protocol(
        preset_name="custom",
        prepare_sec=1.5,
        active_sec=3.5,
        rest_sec=2.5,
        long_idle_sec=60.0,
        target_repeats=7,
        idle_repeats=13,
        switch_trials=9,
    )
    assert name == "custom"
    assert abs(float(prepare) - 1.5) < 1e-9
    assert abs(float(active) - 3.5) < 1e-9
    assert abs(float(rest) - 2.5) < 1e-9
    assert abs(float(long_idle) - 60.0) < 1e-9
    assert int(target) == 7
    assert int(idle) == 13
    assert int(switch) == 9


def test_round_session_id_replaces_existing_round_suffix() -> None:
    session_id = _build_round_session_id("subject_demo_r01", 2)
    assert session_id == "subject_demo_r02"


def test_collection_worker_tone_event_payload() -> None:
    config = CollectionConfig(
        serial_port="auto",
        board_id=0,
        freqs=(8.0, 10.0, 12.0, 15.0),
        subject_id="subject",
        session_id="subject_session_r01",
        session_index=1,
        dataset_dir=PROJECT_DIR / "profiles",
        protocol_name=STABLE_12M_PRESET.key,
        rounds_planned=3,
        round_index=2,
    )
    worker = CollectionWorker(config)
    events: list[dict[str, int | str]] = []
    worker.trial_tone_event.connect(lambda payload: events.append(dict(payload)))  # type: ignore[arg-type]
    worker._emit_tone(event="active_start", trial_index=5, total_trials=74, retry_index=1)
    assert len(events) == 1
    event = events[0]
    assert str(event["event"]) == "active_start"
    assert int(event["round_index"]) == 2
    assert int(event["trial_index"]) == 5
    assert int(event["total_trials"]) == 74
    assert int(event["retry_index"]) == 1


def test_validate_collection_protocol_enforces_active_minimum() -> None:
    with pytest.raises(ValueError, match="active_sec must be >="):
        _validate_collection_protocol(active_sec=float(MIN_ACTIVE_SEC_FOR_TRAINING) - 0.1)
    with pytest.raises(ValueError, match="long_idle_sec must be 0 or >="):
        _validate_collection_protocol(
            active_sec=float(MIN_ACTIVE_SEC_FOR_TRAINING),
            long_idle_sec=float(MIN_ACTIVE_SEC_FOR_TRAINING) - 0.1,
        )
    _validate_collection_protocol(
        active_sec=float(MIN_ACTIVE_SEC_FOR_TRAINING),
        long_idle_sec=float(MIN_ACTIVE_SEC_FOR_TRAINING),
    )


def test_long_idle_round_estimate_adds_trial_and_duration() -> None:
    trials = trial_count_for_protocol(
        target_repeats=DEFAULT_STABLE_TARGET_REPEATS,
        idle_repeats=DEFAULT_STABLE_IDLE_REPEATS,
        switch_trials=DEFAULT_STABLE_SWITCH_TRIALS,
        long_idle_sec=60.0,
    )
    assert trials == 75
    round_sec = estimate_round_seconds(
        prepare_sec=DEFAULT_STABLE_PREPARE_SEC,
        active_sec=DEFAULT_STABLE_ACTIVE_SEC,
        rest_sec=DEFAULT_STABLE_REST_SEC,
        target_repeats=DEFAULT_STABLE_TARGET_REPEATS,
        idle_repeats=DEFAULT_STABLE_IDLE_REPEATS,
        switch_trials=DEFAULT_STABLE_SWITCH_TRIALS,
        long_idle_sec=60.0,
    )
    assert abs(float(round_sec) - 805.0) < 1e-9


def test_build_collection_trials_appends_long_idle_when_enabled() -> None:
    protocol = CollectionProtocol(
        name="custom",
        prepare_sec=1.0,
        active_sec=5.0,
        rest_sec=4.0,
        target_repeats=2,
        idle_repeats=2,
        switch_trials=1,
        long_idle_sec=60.0,
    )
    trials = build_collection_trials((8.0, 10.0, 12.0, 15.0), protocol=protocol, seed=123, session_index=1)
    assert str(trials[-1].label) == "long_idle"
    assert trials[-1].expected_freq is None


def test_protocol_signature_changes_when_long_idle_changes() -> None:
    base = build_protocol_signature(
        sampling_rate=250,
        protocol_config={
            "prepare_sec": 1.0,
            "active_sec": 5.0,
            "rest_sec": 4.0,
            "target_repeats": 10,
            "idle_repeats": 20,
            "switch_trials": 14,
            "long_idle_sec": 0.0,
        },
        freqs=(8.0, 10.0, 12.0, 15.0),
        board_eeg_channels=(1, 2, 3, 4, 5, 6, 7, 8),
    )
    with_long_idle = build_protocol_signature(
        sampling_rate=250,
        protocol_config={
            "prepare_sec": 1.0,
            "active_sec": 5.0,
            "rest_sec": 4.0,
            "target_repeats": 10,
            "idle_repeats": 20,
            "switch_trials": 14,
            "long_idle_sec": 60.0,
        },
        freqs=(8.0, 10.0, 12.0, 15.0),
        board_eeg_channels=(1, 2, 3, 4, 5, 6, 7, 8),
    )
    assert base != with_long_idle


def test_save_collection_dataset_bundle_marks_long_idle_stage() -> None:
    protocol = CollectionProtocol(
        name="custom",
        prepare_sec=1.0,
        active_sec=5.0,
        rest_sec=4.0,
        target_repeats=2,
        idle_repeats=2,
        switch_trials=1,
        long_idle_sec=60.0,
    )
    trials = build_collection_trials((8.0, 10.0, 12.0, 15.0), protocol=protocol, seed=123, session_index=1)
    trial_segments = []
    quality_rows = []
    for order_index, trial in enumerate(trials):
        samples = 15000 if str(trial.label) == "long_idle" else 1250
        trial_segments.append((trial, np.zeros((samples, 8), dtype=np.float64)))
        quality_rows.append(
            {
                "order_index": order_index,
                "target_samples": samples,
                "used_samples": samples,
                "retry_count": 0,
            }
        )
    dataset_root = Path(tempfile.mkdtemp(prefix="ssvep_collection_test_", dir=str(PROJECT_DIR)))
    try:
        result = save_collection_dataset_bundle(
            dataset_root=dataset_root,
            session_id="subject001_r01",
            subject_id="subject001",
            serial_port="COM4",
            board_id=0,
            sampling_rate=250,
            freqs=(8.0, 10.0, 12.0, 15.0),
            board_eeg_channels=(1, 2, 3, 4, 5, 6, 7, 8),
            protocol_config={
                "protocol_name": "custom",
                "prepare_sec": 1.0,
                "active_sec": 5.0,
                "rest_sec": 4.0,
                "long_idle_sec": 60.0,
                "target_repeats": 2,
                "idle_repeats": 2,
                "switch_trials": 1,
            },
            trial_segments=trial_segments,
            quality_rows=quality_rows,
        )
        manifest = json.loads(Path(result["dataset_manifest"]).read_text(encoding="utf-8"))
        long_idle_rows = [row for row in manifest["trials"] if row["label"] == "long_idle"]
        assert len(long_idle_rows) == 1
        assert str(long_idle_rows[0]["stage"]) == "long_idle"
        assert int(long_idle_rows[0]["target_samples"]) == 15000
    finally:
        shutil.rmtree(dataset_root, ignore_errors=True)
