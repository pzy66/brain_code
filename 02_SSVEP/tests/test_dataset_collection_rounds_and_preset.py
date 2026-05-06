from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import apps.data_collection_ui as collection_ui
from apps.data_collection_ui import (
    CollectionConfig,
    CollectionWorker,
    DatasetCollectionWindow,
    SpeechPromptPlayer,
    ACTIVE_START_CUE_SEC,
    ACTIVE_STIMULUS_ARM_SEC,
    PHASE_CAL_ACTIVE,
    PHASE_CAL_REST,
    DEFAULT_STABLE_ACTIVE_SEC,
    DEFAULT_STABLE_IDLE_REPEATS,
    DEFAULT_STABLE_LONG_IDLE_SEC,
    DEFAULT_STABLE_PREPARE_SEC,
    DEFAULT_STABLE_REST_SEC,
    DEFAULT_STABLE_SWITCH_TRIALS,
    DEFAULT_STABLE_TARGET_REPEATS,
    MIN_ACTIVE_SEC_FOR_TRAINING,
    MIN_PREPARE_SEC_FOR_VOICE,
    MIN_REST_SEC_BETWEEN_TRIALS,
    ENHANCED_45M_PRESET,
    STIM_REFRESH_RATE_HZ,
    STABLE_12M_PRESET,
    TONE_EVENT_ACTIVE_END,
    TONE_EVENT_ACTIVE_START,
    TONE_EVENT_PREPARE_START,
    STIMULUS_BACKEND_HEADLESS_NO_VISUAL,
    STIMULUS_BACKEND_PYQT_FULLSCREEN,
    _build_round_session_id,
    build_collection_output_session_id,
    estimate_active_stimulus_arm_sec,
    _resolve_cli_protocol,
    _validate_collection_protocol,
    estimate_round_seconds,
    estimate_stimulus_sample_window_frame_offset,
    prompt_text_for_freq,
    resolve_collection_stim_refresh_rate_hz,
    resolve_collection_stimulus_mode,
    stimulus_backend_metadata,
    stimulus_sample_window_alignment_metadata,
    tone_sequence_duration_sec,
    tone_sequence_for_event,
    trial_count_for_protocol,
    validate_stimulus_frequency_set,
    build_parser,
)
from ssvep_core.stimulus_profiles import (
    DEFAULT_STIMULUS_PROFILE_ID,
    STIMULUS_PROFILE_COMFORT_FBCCA_V1,
    get_stimulus_profile,
    stimulus_profile_metadata,
)
from apps.async_fbcca_validation_ui import (
    FourArrowStimWidget,
    PHASE_VALIDATION,
    STIMULUS_MODE_ELAPSED_TIME_SINE,
    STIMULUS_MODE_FRAME_LOCKED_SINE,
    stimulus_frame_qc_report,
    stimulus_luminance,
    stimulus_luminance_elapsed,
    stimulus_luminance_frame_locked,
    validate_stimulus_mode,
)
from ssvep_core.dataset import (
    CollectionProtocol,
    build_collection_trials,
    build_protocol_signature,
    save_collection_dataset_bundle,
)


def _get_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


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
    expected = 74.0 * (
        float(DEFAULT_STABLE_PREPARE_SEC)
        + float(ACTIVE_START_CUE_SEC)
        + float(ACTIVE_STIMULUS_ARM_SEC)
        + float(DEFAULT_STABLE_ACTIVE_SEC)
        + float(DEFAULT_STABLE_REST_SEC)
    )
    assert abs(float(round_sec) - expected) < 1e-9


def test_round_estimate_uses_selected_stim_refresh_rate() -> None:
    round_sec = estimate_round_seconds(
        prepare_sec=1.0,
        active_sec=5.0,
        rest_sec=4.0,
        target_repeats=1,
        idle_repeats=0,
        switch_trials=0,
        refresh_rate_hz=60.0,
    )
    expected = 4.0 * (1.0 + ACTIVE_START_CUE_SEC + (1.0 / 60.0) + 5.0 + 4.0)
    assert abs(float(round_sec) - expected) < 1e-9


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


def test_round_session_id_sanitizes_path_like_base() -> None:
    session_id = _build_round_session_id(r"..\outside/session:demo_r01", 2)
    assert session_id == "outside_session_demo_r02"


def test_aborted_collection_uses_unique_output_session_id() -> None:
    assert (
        build_collection_output_session_id("subject_demo_r01", collection_aborted=False, stamp="20260424_120000")
        == "subject_demo_r01"
    )
    assert (
        build_collection_output_session_id("subject_demo_r01", collection_aborted=True, stamp="20260424 120000")
        == "subject_demo_r01_aborted_20260424_120000"
    )
    assert (
        build_collection_output_session_id(r"..\outside/session:demo", collection_aborted=False)
        == "outside_session_demo"
    )


def test_collection_output_session_id_avoids_existing_saved_session(tmp_path: Path) -> None:
    existing = tmp_path / "subject_demo_r01"
    existing.mkdir(parents=True)
    (existing / "session_manifest.json").write_text("{}", encoding="utf-8")
    assert (
        build_collection_output_session_id(
            "subject_demo_r01",
            collection_aborted=False,
            dataset_dir=tmp_path,
            stamp="20260424_120000",
        )
        == "subject_demo_r01_rerun_20260424_120000"
    )


def test_collection_worker_tone_event_payload() -> None:
    config = CollectionConfig(
        serial_port="auto",
        board_id=0,
        freqs=(8.0, 10.0, 12.0, 15.0),
        subject_id="subject",
        session_id="subject_session_r01",
        session_index=1,
        dataset_dir=PROJECT_DIR / "artifacts" / "datasets",
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


def test_collection_worker_voice_prompt_payload_contains_request_id() -> None:
    config = CollectionConfig(
        serial_port="auto",
        board_id=0,
        freqs=(8.0, 10.0, 12.0, 15.0),
        subject_id="subject",
        session_id="subject_session_r01",
        session_index=1,
        dataset_dir=PROJECT_DIR / "artifacts" / "datasets",
        protocol_name=STABLE_12M_PRESET.key,
        sync_voice_prompt=True,
    )
    worker = CollectionWorker(config)
    events: list[dict[str, int | str | bool]] = []
    worker.voice_prompt_event.connect(lambda payload: events.append(dict(payload)))  # type: ignore[arg-type]
    request_id = worker._emit_voice_prompt(
        text="看上方",
        trial_index=5,
        total_trials=74,
        retry_index=1,
    )
    assert request_id > 0
    assert len(events) == 1
    event = events[0]
    assert str(event["text"]) == "看上方"
    assert bool(event["stop"]) is False
    assert int(event["request_id"]) == request_id


def test_collection_worker_voice_prompt_wait_returns_after_matching_ack() -> None:
    config = CollectionConfig(
        serial_port="auto",
        board_id=0,
        freqs=(8.0, 10.0, 12.0, 15.0),
        subject_id="subject",
        session_id="subject_session_r01",
        session_index=1,
        dataset_dir=PROJECT_DIR / "artifacts" / "datasets",
        protocol_name=STABLE_12M_PRESET.key,
        sync_voice_prompt=True,
    )
    worker = CollectionWorker(config)
    request_id = worker._emit_voice_prompt(
        text="看上方",
        trial_index=1,
        total_trials=1,
        retry_index=0,
    )
    worker.notify_voice_prompt_finished(request_id + 1)
    assert worker._voice_prompt_finished_event.is_set() is False
    worker.notify_voice_prompt_finished(request_id)
    start = time.perf_counter()
    elapsed = worker._wait_for_voice_prompt_finished(request_id)
    assert elapsed <= time.perf_counter() - start + 1e-9
    assert time.perf_counter() - start < 0.25


def test_collection_worker_voice_prompt_timeout_forces_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(collection_ui, "VOICE_PROMPT_FINISH_TIMEOUT_SEC", 0.01)
    config = CollectionConfig(
        serial_port="auto",
        board_id=0,
        freqs=(8.0, 10.0, 12.0, 15.0),
        subject_id="subject",
        session_id="subject_session_r01",
        session_index=1,
        dataset_dir=PROJECT_DIR / "artifacts" / "datasets",
        protocol_name=STABLE_12M_PRESET.key,
        sync_voice_prompt=True,
    )
    worker = CollectionWorker(config)
    events: list[dict[str, int | str | bool]] = []
    worker.voice_prompt_event.connect(lambda payload: events.append(dict(payload)))  # type: ignore[arg-type]
    request_id = worker._emit_voice_prompt(
        text="看上方",
        trial_index=1,
        total_trials=1,
        retry_index=0,
    )
    elapsed = worker._wait_for_voice_prompt_finished(
        request_id,
        trial_index=1,
        total_trials=1,
        retry_index=0,
    )
    assert elapsed >= 0.0
    assert len(events) == 2
    assert bool(events[0]["stop"]) is False
    assert bool(events[1]["stop"]) is True
    assert worker._voice_prompt_finished_event.is_set() is True


def test_collection_worker_voice_prompt_runs_inside_prepare_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_order: list[str] = []
    monkeypatch.setattr(collection_ui, "play_collection_tone_event", lambda payload: prompt_order.append("tone"))
    config = CollectionConfig(
        serial_port="auto",
        board_id=0,
        freqs=(8.0, 10.0, 12.0, 15.0),
        subject_id="subject",
        session_id="subject_session_r01",
        session_index=1,
        dataset_dir=PROJECT_DIR / "artifacts" / "datasets",
        protocol_name=STABLE_12M_PRESET.key,
        prepare_sec=1.0,
        sync_voice_prompt=True,
    )
    worker = CollectionWorker(config)
    wait_timeouts: list[float | None] = []
    sleeps: list[float] = []

    def fake_wait(request_id: int, **kwargs: object) -> float:
        prompt_order.append("voice_wait")
        wait_timeouts.append(kwargs.get("timeout_sec"))  # type: ignore[arg-type]
        return 0.7

    worker._wait_for_voice_prompt_finished = fake_wait  # type: ignore[method-assign]
    worker._sleep_interruptible = lambda seconds: sleeps.append(float(seconds)) or False  # type: ignore[method-assign]

    interrupted = worker._run_prepare_window(
        request_id=1,
        trial_index=1,
        total_trials=1,
        retry_index=0,
    )

    assert interrupted is False
    assert wait_timeouts == [1.0]
    assert prompt_order == ["voice_wait", "tone"]
    assert len(sleeps) == 1
    assert abs(float(sleeps[0]) - 0.3) < 0.05


def test_collection_worker_can_ack_active_stimulus_phase() -> None:
    config = CollectionConfig(
        serial_port="auto",
        board_id=0,
        freqs=(8.0, 10.0, 12.0, 15.0),
        subject_id="subject",
        session_id="subject_session_r01",
        session_index=1,
        dataset_dir=PROJECT_DIR / "artifacts" / "datasets",
        protocol_name=STABLE_12M_PRESET.key,
        sync_stimulus_phase=True,
    )
    worker = CollectionWorker(config)
    worker.notify_stimulus_phase_applied(
        {
            "mode": "calibration_active",
            "frame_index": 3,
            "presented_t_sec": 0.05,
            "cue_freq": 8.0,
        }
    )
    assert worker._stimulus_phase_applied_event.is_set()
    ready, payload = worker._wait_for_stimulus_phase_applied()
    assert ready is True
    assert int(payload["frame_index"]) == 3
    assert abs(float(payload["presented_t_sec"]) - 0.05) < 1e-9
    assert str(payload.get("ack_wall_time", "")).strip()


def test_collection_worker_sleep_interrupts_after_stop_request() -> None:
    config = CollectionConfig(
        serial_port="auto",
        board_id=0,
        freqs=(8.0, 10.0, 12.0, 15.0),
        subject_id="subject",
        session_id="subject_session_r01",
        session_index=1,
        dataset_dir=PROJECT_DIR / "artifacts" / "datasets",
        protocol_name=STABLE_12M_PRESET.key,
    )
    worker = CollectionWorker(config)
    worker.request_stop()
    start = time.perf_counter()
    assert worker._sleep_interruptible(5.0) is True
    assert time.perf_counter() - start < 0.25


def test_collection_worker_saves_partial_dataset_on_runtime_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBoard:
        def start_stream(self, *args, **kwargs) -> None:
            return None

        def stop_stream(self) -> None:
            return None

        def release_session(self) -> None:
            return None

        def get_board_data(self) -> np.ndarray:
            return np.zeros((4, 0), dtype=np.float64)

    class FakeBoardShim:
        @staticmethod
        def get_sampling_rate(board_id: int) -> int:
            return 250

        @staticmethod
        def get_eeg_channels(board_id: int) -> list[int]:
            return [0, 1, 2, 3]

    read_calls = {"count": 0}

    def fake_read_recent_eeg_segment(*args, **kwargs):
        read_calls["count"] += 1
        if int(read_calls["count"]) == 1:
            return np.ones((1000, 4), dtype=np.float64), 1000, 1000
        raise RuntimeError("simulated read failure")

    monkeypatch.setattr(collection_ui, "BoardShim", FakeBoardShim)
    monkeypatch.setattr(
        collection_ui,
        "prepare_board_session",
        lambda board_id, serial_port: (FakeBoard(), "COM7", ["COM7", "COM8"]),
    )
    monkeypatch.setattr(collection_ui, "ensure_stream_ready", lambda board, fs: 512)
    monkeypatch.setattr(collection_ui, "read_recent_eeg_segment", fake_read_recent_eeg_segment)
    monkeypatch.setattr(collection_ui, "play_collection_tone_event", lambda payload: None)
    monkeypatch.setattr(collection_ui, "play_collection_tone_event_sync", lambda payload: None)
    monkeypatch.setattr(collection_ui.CollectionWorker, "_sleep_interruptible", lambda self, seconds: False)

    config = CollectionConfig(
        serial_port="auto",
        board_id=0,
        freqs=(8.0, 10.0, 12.0, 15.0),
        subject_id="subject",
        session_id="subject_session_r01",
        session_index=1,
        dataset_dir=tmp_path,
        protocol_name="custom",
        prepare_sec=0.0,
        active_sec=4.0,
        rest_sec=0.0,
        target_repeats=1,
        idle_repeats=0,
        switch_trials=0,
        long_idle_sec=0.0,
        rounds_planned=2,
        round_index=1,
    )
    worker = CollectionWorker(config)
    done_payloads: list[dict[str, object]] = []
    error_texts: list[str] = []
    worker.done.connect(lambda payload: done_payloads.append(dict(payload)))  # type: ignore[arg-type]
    worker.error.connect(lambda text: error_texts.append(str(text)))  # type: ignore[arg-type]

    worker.run()

    assert len(done_payloads) == 1
    payload = done_payloads[0]
    assert bool(payload.get("collection_aborted")) is True
    assert int(payload.get("collected_trials", 0)) == 1
    assert int(payload.get("total_trials", 0)) == 4
    manifest_path = Path(str(payload.get("dataset_manifest", "")))
    npz_path = Path(str(payload.get("dataset_npz", "")))
    assert manifest_path.exists()
    assert npz_path.exists()
    assert any("已保存部分数据" in text for text in error_texts)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_config = dict(manifest.get("protocol_config", {}))
    quality_summary = dict(manifest.get("quality_summary", {}))
    trial_rows = list(manifest.get("trials", []))

    assert protocol_config.get("collection_aborted") is True
    assert protocol_config.get("aborted_reason") == "runtime_failure"
    assert protocol_config.get("failure_reason") == "simulated read failure"
    assert bool(quality_summary.get("collection_aborted")) is True
    assert int(quality_summary.get("saved_trial_count", 0)) == 1
    assert len(trial_rows) == 1
    assert str(trial_rows[0].get("active_window_started_at", "")).strip()
    assert str(trial_rows[0].get("segment_captured_at", "")).strip()


def test_collection_worker_saves_raw_board_when_failure_happens_before_first_valid_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBoard:
        def __init__(self) -> None:
            self._chunks = [
                np.ones((4, 4), dtype=np.float64),
                np.full((4, 8), 2.0, dtype=np.float64),
                np.full((4, 16), 3.0, dtype=np.float64),
            ]

        def start_stream(self, *args, **kwargs) -> None:
            return None

        def stop_stream(self) -> None:
            return None

        def release_session(self) -> None:
            return None

        def get_board_data(self) -> np.ndarray:
            if self._chunks:
                return self._chunks.pop(0)
            return np.zeros((4, 0), dtype=np.float64)

    class FakeBoardShim:
        @staticmethod
        def get_sampling_rate(board_id: int) -> int:
            return 250

        @staticmethod
        def get_eeg_channels(board_id: int) -> list[int]:
            return [0, 1, 2, 3]

    monkeypatch.setattr(collection_ui, "BoardShim", FakeBoardShim)
    monkeypatch.setattr(
        collection_ui,
        "prepare_board_session",
        lambda board_id, serial_port: (FakeBoard(), "COM7", ["COM7"]),
    )
    monkeypatch.setattr(collection_ui, "ensure_stream_ready", lambda board, fs: 512)
    monkeypatch.setattr(
        collection_ui,
        "read_recent_eeg_segment",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("simulated first trial read failure")),
    )
    monkeypatch.setattr(collection_ui, "play_collection_tone_event", lambda payload: None)
    monkeypatch.setattr(collection_ui, "play_collection_tone_event_sync", lambda payload: None)
    monkeypatch.setattr(collection_ui.CollectionWorker, "_sleep_interruptible", lambda self, seconds: False)

    config = CollectionConfig(
        serial_port="auto",
        board_id=0,
        freqs=(8.0, 10.0, 12.0, 15.0),
        subject_id="subject",
        session_id="subject_session_r01",
        session_index=1,
        dataset_dir=tmp_path,
        protocol_name="custom",
        prepare_sec=0.0,
        active_sec=4.0,
        rest_sec=0.0,
        target_repeats=1,
        idle_repeats=0,
        switch_trials=0,
        long_idle_sec=0.0,
    )
    worker = CollectionWorker(config)
    done_payloads: list[dict[str, object]] = []
    error_texts: list[str] = []
    worker.done.connect(lambda payload: done_payloads.append(dict(payload)))  # type: ignore[arg-type]
    worker.error.connect(lambda text: error_texts.append(str(text)))  # type: ignore[arg-type]

    worker.run()

    assert len(done_payloads) == 1
    payload = done_payloads[0]
    assert bool(payload.get("collection_aborted")) is True
    assert int(payload.get("collected_trials", -1)) == 0
    manifest_path = Path(str(payload.get("dataset_manifest", "")))
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_config = dict(manifest.get("protocol_config", {}))
    quality_summary = dict(manifest.get("quality_summary", {}))
    continuous_meta = dict(manifest.get("continuous_board", {}))
    files = dict(manifest.get("files", {}))

    assert manifest.get("trials") == []
    assert protocol_config.get("collection_aborted") is True
    assert protocol_config.get("aborted_reason") == "runtime_failure"
    assert protocol_config.get("failure_reason") == "simulated first trial read failure"
    assert int(quality_summary.get("saved_trial_count", -1)) == 0
    assert continuous_meta.get("saved") is True
    assert Path(str(files.get("continuous_board_npz", ""))).exists()
    assert any("原始板卡数据" in text for text in error_texts)


def test_collection_worker_saves_raw_board_when_stopped_during_warmup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBoard:
        def __init__(self) -> None:
            self._chunk = np.arange(4 * 12, dtype=np.float64).reshape(4, 12)

        def start_stream(self, *args, **kwargs) -> None:
            return None

        def stop_stream(self) -> None:
            return None

        def release_session(self) -> None:
            return None

        def get_board_data(self) -> np.ndarray:
            chunk = self._chunk
            self._chunk = np.zeros((4, 0), dtype=np.float64)
            return chunk

    class FakeBoardShim:
        @staticmethod
        def get_sampling_rate(board_id: int) -> int:
            return 250

        @staticmethod
        def get_eeg_channels(board_id: int) -> list[int]:
            return [0, 1, 2, 3]

    monkeypatch.setattr(collection_ui, "BoardShim", FakeBoardShim)
    monkeypatch.setattr(
        collection_ui,
        "prepare_board_session",
        lambda board_id, serial_port: (FakeBoard(), "COM7", ["COM7"]),
    )
    monkeypatch.setattr(collection_ui, "ensure_stream_ready", lambda board, fs: 12)
    monkeypatch.setattr(collection_ui.CollectionWorker, "_sleep_interruptible", lambda self, seconds: True)

    config = CollectionConfig(
        serial_port="auto",
        board_id=0,
        freqs=(8.0, 10.0, 12.0, 15.0),
        subject_id="subject",
        session_id="subject_session_r01",
        session_index=1,
        dataset_dir=tmp_path,
        protocol_name="custom",
        prepare_sec=1.0,
        active_sec=4.0,
        rest_sec=2.0,
        target_repeats=1,
        idle_repeats=0,
        switch_trials=0,
        long_idle_sec=0.0,
    )
    worker = CollectionWorker(config)
    done_payloads: list[dict[str, object]] = []
    worker.done.connect(lambda payload: done_payloads.append(dict(payload)))  # type: ignore[arg-type]

    worker.run()

    assert len(done_payloads) == 1
    payload = done_payloads[0]
    assert bool(payload.get("collection_aborted")) is True
    assert int(payload.get("collected_trials", -1)) == 0
    manifest_path = Path(str(payload.get("dataset_manifest", "")))
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_config = dict(manifest.get("protocol_config", {}))
    quality_summary = dict(manifest.get("quality_summary", {}))
    continuous_meta = dict(manifest.get("continuous_board", {}))
    files = dict(manifest.get("files", {}))

    assert manifest.get("trials") == []
    assert protocol_config.get("collection_aborted") is True
    assert protocol_config.get("aborted_reason") == "runtime_failure"
    assert protocol_config.get("failure_reason") == "user_stop_during_warmup"
    assert int(quality_summary.get("saved_trial_count", -1)) == 0
    assert continuous_meta.get("saved") is True
    assert continuous_meta.get("shape") == [4, 12]
    assert Path(str(files.get("continuous_board_npz", ""))).exists()


def test_collection_worker_simulation_mode_skips_board_and_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collection_ui,
        "prepare_board_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("simulation should not connect board")),
    )
    monkeypatch.setattr(
        collection_ui,
        "save_collection_dataset_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("simulation should not save dataset")),
    )
    monkeypatch.setattr(collection_ui, "play_collection_tone_event", lambda payload: None)
    monkeypatch.setattr(collection_ui, "play_collection_tone_event_sync", lambda payload: None)
    monkeypatch.setattr(collection_ui.CollectionWorker, "_sleep_interruptible", lambda self, seconds: False)

    config = CollectionConfig(
        serial_port="auto",
        board_id=0,
        freqs=(8.0, 10.0, 12.0, 15.0),
        subject_id="subject",
        session_id="subject_session_r01",
        session_index=1,
        dataset_dir=tmp_path,
        protocol_name="custom",
        prepare_sec=0.0,
        active_sec=2.0,
        rest_sec=0.0,
        target_repeats=1,
        idle_repeats=0,
        switch_trials=0,
        long_idle_sec=0.0,
        rounds_planned=1,
        round_index=1,
        stimulus_backend=STIMULUS_BACKEND_PYQT_FULLSCREEN,
        simulation_only=True,
    )
    worker = CollectionWorker(config)
    done_payloads: list[dict[str, object]] = []
    worker.done.connect(lambda payload: done_payloads.append(dict(payload)))  # type: ignore[arg-type]

    worker.run()

    assert len(done_payloads) == 1
    payload = done_payloads[0]
    assert bool(payload.get("simulation_only")) is True
    assert bool(payload.get("collection_aborted")) is False
    assert int(payload.get("executed_trials", 0)) == 4
    assert int(payload.get("total_trials", 0)) == 4
    assert int(payload.get("collected_trials", 0)) == 0
    assert str(payload.get("dataset_manifest", "")) == ""
    assert str(payload.get("dataset_npz", "")) == ""


def test_collection_main_headless_simulation_only_runs_without_board_or_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        collection_ui,
        "prepare_board_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("simulation should not connect board")),
    )
    monkeypatch.setattr(
        collection_ui,
        "save_collection_dataset_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("simulation should not save dataset")),
    )
    monkeypatch.setattr(collection_ui, "play_collection_tone_event", lambda payload: None)
    monkeypatch.setattr(collection_ui, "play_collection_tone_event_sync", lambda payload: None)
    monkeypatch.setattr(collection_ui.CollectionWorker, "_sleep_interruptible", lambda self, seconds: False)

    exit_code = collection_ui.main(
        [
            "--headless",
            "--simulation-only",
            "--preset",
            "custom",
            "--prepare-sec",
            "1",
            "--active-sec",
            "1.5",
            "--rest-sec",
            "2",
            "--target-repeats",
            "1",
            "--idle-repeats",
            "0",
            "--switch-trials",
            "0",
            "--dataset-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Simulation-only + headless will not render visual stimulus" in captured.out
    assert "Simulation-only run finished: no dataset saved." in captured.out
    assert "Dataset manifest:" not in captured.out
    assert list(tmp_path.iterdir()) == []


def test_collection_tone_sequences_cover_prepare_start_and_end() -> None:
    prepare_sequence = tone_sequence_for_event(TONE_EVENT_PREPARE_START)
    start_sequence = tone_sequence_for_event(TONE_EVENT_ACTIVE_START)
    end_sequence = tone_sequence_for_event(TONE_EVENT_ACTIVE_END)

    assert len(prepare_sequence) == 3
    assert len(start_sequence) == 1
    assert len(end_sequence) == 3
    assert int(start_sequence[0][0]) > int(end_sequence[0][0])
    assert abs(tone_sequence_duration_sec(TONE_EVENT_ACTIVE_START) - 0.12) < 1e-9
    assert tone_sequence_for_event("unknown") == ()


def test_collection_voice_prompt_maps_center_and_directions() -> None:
    freqs = (8.0, 10.0, 12.0, 15.0)

    assert prompt_text_for_freq(freqs, None) == "看中间"
    assert prompt_text_for_freq(freqs, 8.0) == "看上方"
    assert prompt_text_for_freq(freqs, 10.0) == "看左方"
    assert prompt_text_for_freq(freqs, 12.0) == "看下方"
    assert prompt_text_for_freq(freqs, 15.0) == "看右方"
    assert prompt_text_for_freq(freqs, 99.0) == "看目标"


def test_stimulus_luminance_period_matches_target_frequency() -> None:
    for freq in (8.0, 10.0, 12.0, 15.0):
        period_sec = 1.0 / float(freq)
        assert abs(stimulus_luminance_elapsed(freq, 0.0, mean=0.5, amp=0.5, phi=0.0) - 0.5) < 1e-12
        assert abs(stimulus_luminance_elapsed(freq, period_sec / 4.0, mean=0.5, amp=0.5, phi=0.0) - 1.0) < 1e-12
        assert abs(stimulus_luminance_elapsed(freq, period_sec / 2.0, mean=0.5, amp=0.5, phi=0.0) - 0.5) < 1e-12
        assert abs(stimulus_luminance_elapsed(freq, 3.0 * period_sec / 4.0, mean=0.5, amp=0.5, phi=0.0) - 0.0) < 1e-12
        assert abs(stimulus_luminance_elapsed(freq, period_sec, mean=0.5, amp=0.5, phi=0.0) - 0.5) < 1e-12
        assert abs(stimulus_luminance(freq, period_sec, mean=0.5, amp=0.5, phi=0.0) - 0.5) < 1e-12


def test_frame_locked_stimulus_luminance_matches_sampled_sine() -> None:
    refresh_rate_hz = 60.0
    freq = 15.0
    assert abs(stimulus_luminance_frame_locked(freq, 0, refresh_rate_hz, mean=0.5, amp=0.5, phi=0.0) - 0.5) < 1e-12
    assert abs(stimulus_luminance_frame_locked(freq, 1, refresh_rate_hz, mean=0.5, amp=0.5, phi=0.0) - 1.0) < 1e-12
    assert abs(stimulus_luminance_frame_locked(freq, 2, refresh_rate_hz, mean=0.5, amp=0.5, phi=0.0) - 0.5) < 1e-12
    assert abs(stimulus_luminance_frame_locked(freq, 3, refresh_rate_hz, mean=0.5, amp=0.5, phi=0.0) - 0.0) < 1e-12


def test_frame_locked_qc_report_peaks_at_target_frequencies() -> None:
    report = stimulus_frame_qc_report(
        freqs=(8.0, 10.0, 12.0, 15.0),
        refresh_rate_hz=60.0,
        active_sec=5.0,
        stimulus_mode=STIMULUS_MODE_FRAME_LOCKED_SINE,
        mean=0.5,
        amp=0.5,
        phi=0.0,
    )
    assert int(report["frame_count"]) == 300
    peaks = {float(row["target_hz"]): float(row["peak_hz"]) for row in report["rows"]}
    assert peaks == {8.0: 8.0, 10.0: 10.0, 12.0: 12.0, 15.0: 15.0}


def test_validate_stimulus_mode_rejects_unknown_values() -> None:
    assert validate_stimulus_mode(STIMULUS_MODE_ELAPSED_TIME_SINE) == STIMULUS_MODE_ELAPSED_TIME_SINE
    assert validate_stimulus_mode(STIMULUS_MODE_FRAME_LOCKED_SINE) == STIMULUS_MODE_FRAME_LOCKED_SINE
    with pytest.raises(ValueError, match="stimulus_mode"):
        validate_stimulus_mode("unknown")


def test_collection_cli_defaults_to_auto_comfort_stimulus_mode() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert str(args.stimulus_profile_id) == DEFAULT_STIMULUS_PROFILE_ID
    assert str(args.stimulus_mode) == "auto"
    args = parser.parse_args(["--stimulus-mode", STIMULUS_MODE_FRAME_LOCKED_SINE])
    assert str(args.stimulus_mode) == STIMULUS_MODE_FRAME_LOCKED_SINE


def test_comfort_fbcca_profile_uses_lower_contrast_and_ramp() -> None:
    profile = get_stimulus_profile(STIMULUS_PROFILE_COMFORT_FBCCA_V1)
    assert profile.freqs == (8.0, 10.0, 12.0, 15.0)
    assert abs(float(profile.mean) - 0.40) < 1e-12
    assert abs(float(profile.amp) - 0.20) < 1e-12
    assert abs(float(profile.ramp_sec) - 0.30) < 1e-12
    assert abs(float(profile.luminance_min) - 0.20) < 1e-12
    assert abs(float(profile.luminance_max) - 0.60) < 1e-12
    assert abs(float(profile.michelson_contrast) - 0.50) < 1e-12

    report = stimulus_frame_qc_report(
        freqs=profile.freqs,
        refresh_rate_hz=240.0,
        active_sec=3.0,
        stimulus_mode=STIMULUS_MODE_FRAME_LOCKED_SINE,
        mean=float(profile.mean),
        amp=float(profile.amp),
        phi=float(profile.phi),
        ramp_sec=float(profile.ramp_sec),
    )
    assert report["clipping"] is False
    assert float(report["luminance_min"]) > 0.0
    assert float(report["luminance_max"]) < 1.0
    peaks = {float(row["target_hz"]): float(row["peak_hz"]) for row in report["rows"]}
    assert peaks == {8.0: 8.0, 10.0: 10.0, 12.0: 12.0, 15.0: 15.0}


def test_collection_auto_stimulus_mode_prefers_frame_locked_only_at_stable_240hz() -> None:
    mode, reason = resolve_collection_stimulus_mode(
        stimulus_profile_id=STIMULUS_PROFILE_COMFORT_FBCCA_V1,
        refresh_rate_hz=240.0,
        requested_mode="auto",
    )
    assert mode == STIMULUS_MODE_FRAME_LOCKED_SINE
    assert reason == "stable_240hz_frame_locked"

    mode, reason = resolve_collection_stimulus_mode(
        stimulus_profile_id=STIMULUS_PROFILE_COMFORT_FBCCA_V1,
        refresh_rate_hz=144.0,
        requested_mode="auto",
    )
    assert mode == STIMULUS_MODE_ELAPSED_TIME_SINE
    assert reason == "fallback_refresh_not_confirmed_240hz"

    mode, reason = resolve_collection_stimulus_mode(
        stimulus_profile_id=STIMULUS_PROFILE_COMFORT_FBCCA_V1,
        refresh_rate_hz=144.0,
        requested_mode=STIMULUS_MODE_FRAME_LOCKED_SINE,
    )
    assert mode == STIMULUS_MODE_FRAME_LOCKED_SINE
    assert reason == "manual"


def test_collection_cli_defaults_to_legacy_manual_refresh_rate() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert abs(float(args.stim_refresh_rate_hz) - float(STIM_REFRESH_RATE_HZ)) < 1e-9
    args = parser.parse_args(["--stim-refresh-rate-hz", "0"])
    assert float(args.stim_refresh_rate_hz) == 0.0


def test_stimulus_backend_metadata_marks_headless_as_not_rendered() -> None:
    headless = stimulus_backend_metadata(STIMULUS_BACKEND_HEADLESS_NO_VISUAL)
    assert headless["stimulus_backend"] == STIMULUS_BACKEND_HEADLESS_NO_VISUAL
    assert headless["stimulus_rendered_by_this_process"] is False
    assert headless["stimulus_mode_applies_to_rendered_stimulus"] is False

    pyqt = stimulus_backend_metadata(STIMULUS_BACKEND_PYQT_FULLSCREEN)
    assert pyqt["stimulus_backend"] == STIMULUS_BACKEND_PYQT_FULLSCREEN
    assert pyqt["stimulus_rendered_by_this_process"] is True
    assert pyqt["stimulus_mode_applies_to_rendered_stimulus"] is True


def test_collection_config_defaults_to_headless_no_visual_backend() -> None:
    config = CollectionConfig(
        serial_port="auto",
        board_id=0,
        freqs=(8.0, 10.0, 12.0, 15.0),
        subject_id="subject",
        session_id="subject_session_r01",
        session_index=1,
        dataset_dir=PROJECT_DIR / "artifacts" / "datasets",
    )
    assert config.stimulus_backend == STIMULUS_BACKEND_HEADLESS_NO_VISUAL


def test_collection_worker_active_phase_uses_current_trial_active_sec() -> None:
    config = CollectionConfig(
        serial_port="auto",
        board_id=0,
        freqs=(8.0, 10.0, 12.0, 15.0),
        subject_id="subject",
        session_id="subject_session_r01",
        session_index=1,
        dataset_dir=PROJECT_DIR / "artifacts" / "datasets",
        active_sec=3.0,
        long_idle_sec=60.0,
    )
    worker = CollectionWorker(config)
    phases: list[dict[str, object]] = []
    worker.phase_changed.connect(lambda payload: phases.append(dict(payload)))  # type: ignore[arg-type]

    worker._current_trial_active_sec = 60.0
    worker._emit_phase(PHASE_CAL_ACTIVE, "active", "", flicker=True, cue_freq=None)
    worker._emit_phase(PHASE_CAL_REST, "rest", "", flicker=False, cue_freq=None)

    assert abs(float(phases[0]["active_sec"]) - 60.0) < 1e-9
    assert abs(float(phases[1]["active_sec"]) - 3.0) < 1e-9


def test_resolve_collection_stim_refresh_rate_hz_uses_screen_value() -> None:
    class _Screen:
        def refreshRate(self) -> float:
            return 144.0

    assert abs(resolve_collection_stim_refresh_rate_hz(_Screen()) - 144.0) < 1e-9


def test_resolve_collection_stim_refresh_rate_hz_falls_back_to_default_on_invalid_screen() -> None:
    class _ZeroScreen:
        def refreshRate(self) -> float:
            return 0.0

    class _ErrorScreen:
        def refreshRate(self) -> float:
            raise RuntimeError("screen unavailable")

    assert abs(resolve_collection_stim_refresh_rate_hz(_ZeroScreen()) - float(STIM_REFRESH_RATE_HZ)) < 1e-9
    assert abs(resolve_collection_stim_refresh_rate_hz(_ErrorScreen()) - float(STIM_REFRESH_RATE_HZ)) < 1e-9


def test_dataset_collection_window_refresh_rate_override_prefers_manual_value() -> None:
    _ = _get_qapp()
    window = DatasetCollectionWindow(serial_port="auto", board_id=0, freqs=(8.0, 10.0, 12.0, 15.0))
    try:
        window.stim_refresh_rate_spin.setValue(240.0)
        assert abs(window._resolve_stim_refresh_rate_hz() - 240.0) < 1e-9
    finally:
        window.close()


def test_dataset_collection_window_timing_controls_enforce_audio_and_rest_minimums() -> None:
    _ = _get_qapp()
    window = DatasetCollectionWindow(serial_port="auto", board_id=0, freqs=(8.0, 10.0, 12.0, 15.0))
    try:
        assert abs(float(window.prepare_spin.minimum()) - float(MIN_PREPARE_SEC_FOR_VOICE)) < 1e-9
        assert abs(float(window.rest_spin.minimum()) - float(MIN_REST_SEC_BETWEEN_TRIALS)) < 1e-9
        assert float(ENHANCED_45M_PRESET.rest_sec) >= float(MIN_REST_SEC_BETWEEN_TRIALS)
    finally:
        window.close()


def test_dataset_collection_window_refresh_rate_zero_uses_screen_value() -> None:
    _ = _get_qapp()
    window = DatasetCollectionWindow(serial_port="auto", board_id=0, freqs=(8.0, 10.0, 12.0, 15.0))

    class _Screen:
        def refreshRate(self) -> float:
            return 144.0

    try:
        window._stim_target_screen = lambda: _Screen()  # type: ignore[method-assign]
        window.stim_refresh_rate_spin.setValue(0.0)
        assert abs(window._resolve_stim_refresh_rate_hz() - 144.0) < 1e-9
    finally:
        window.close()


def test_speech_prompt_player_fallback_finishes_only_matching_request() -> None:
    _ = _get_qapp()
    player = SpeechPromptPlayer()
    finished: list[int] = []
    player.playback_finished.connect(lambda request_id: finished.append(int(request_id)))  # type: ignore[arg-type]

    player._awaiting_completion = True
    player._pending_request_id = 11
    player._finish_if_pending(10)
    assert finished == []

    player._finish_if_pending(11)
    assert finished == [11]
    assert player._awaiting_completion is False
    assert player._pending_request_id == 0


def test_four_arrow_stim_widget_emits_active_phase_frame_presented_once() -> None:
    _ = _get_qapp()
    widget = FourArrowStimWidget(
        freqs=(8.0, 10.0, 12.0, 15.0),
        refresh_rate_hz=60.0,
        mean=0.5,
        amp=0.5,
        phi=0.0,
        stimulus_mode=STIMULUS_MODE_FRAME_LOCKED_SINE,
    )
    try:
        payloads: list[dict[str, object]] = []
        widget.active_phase_frame_presented.connect(  # type: ignore[arg-type]
            lambda payload: payloads.append(dict(payload))
        )
        widget.apply_phase(
            {
                "mode": PHASE_CAL_ACTIVE,
                "title": "采集中",
                "detail": "",
                "flicker": True,
                "cue_freq": 8.0,
            }
        )
        widget._maybe_emit_active_phase_frame_presented(t_sec=0.125, frame_index=0)
        widget._maybe_emit_active_phase_frame_presented(t_sec=0.140, frame_index=1)

        assert len(payloads) == 1
        assert str(payloads[0]["mode"]) == PHASE_CAL_ACTIVE
        assert bool(payloads[0]["flicker"]) is True
        assert int(payloads[0]["frame_index"]) == 0
    finally:
        widget.stop_clock()
        widget.close()


def test_four_arrow_stim_widget_does_not_rearm_active_ack_during_same_active_phase() -> None:
    _ = _get_qapp()
    widget = FourArrowStimWidget(
        freqs=(8.0, 10.0, 12.0, 15.0),
        refresh_rate_hz=60.0,
        mean=0.5,
        amp=0.5,
        phi=0.0,
        stimulus_mode=STIMULUS_MODE_FRAME_LOCKED_SINE,
    )
    try:
        payloads: list[dict[str, object]] = []
        widget.active_phase_frame_presented.connect(  # type: ignore[arg-type]
            lambda payload: payloads.append(dict(payload))
        )
        active_payload = {
            "mode": PHASE_CAL_ACTIVE,
            "title": "active",
            "detail": "",
            "flicker": True,
            "cue_freq": 8.0,
        }

        widget.apply_phase(active_payload)
        widget._maybe_emit_active_phase_frame_presented(t_sec=0.0, frame_index=0)
        widget.apply_phase({**active_payload, "title": "still active"})
        widget._maybe_emit_active_phase_frame_presented(t_sec=0.1, frame_index=6)

        assert len(payloads) == 1
    finally:
        widget.stop_clock()
        widget.close()


def test_four_arrow_stim_widget_keeps_pending_ack_on_same_phase_reapply() -> None:
    _ = _get_qapp()
    widget = FourArrowStimWidget(
        freqs=(8.0, 10.0, 12.0, 15.0),
        refresh_rate_hz=60.0,
        mean=0.5,
        amp=0.5,
        phi=0.0,
        stimulus_mode=STIMULUS_MODE_ELAPSED_TIME_SINE,
    )
    try:
        payloads: list[dict[str, object]] = []
        widget.active_phase_frame_presented.connect(  # type: ignore[arg-type]
            lambda payload: payloads.append(dict(payload))
        )
        active_payload = {
            "mode": PHASE_CAL_ACTIVE,
            "title": "active",
            "detail": "",
            "flicker": True,
            "cue_freq": 8.0,
        }

        widget.apply_phase(active_payload)
        widget.apply_phase({**active_payload, "title": "active updated"})
        widget._maybe_emit_active_phase_frame_presented(t_sec=0.02, frame_index=0)

        assert len(payloads) == 1
        assert payloads[0]["mode"] == PHASE_CAL_ACTIVE
    finally:
        widget.stop_clock()
        widget.close()


def test_four_arrow_stim_widget_rearms_ack_when_ack_phase_changes() -> None:
    _ = _get_qapp()
    widget = FourArrowStimWidget(
        freqs=(8.0, 10.0, 12.0, 15.0),
        refresh_rate_hz=60.0,
        mean=0.5,
        amp=0.5,
        phi=0.0,
        stimulus_mode=STIMULUS_MODE_ELAPSED_TIME_SINE,
    )
    try:
        payloads: list[dict[str, object]] = []
        widget.active_phase_frame_presented.connect(  # type: ignore[arg-type]
            lambda payload: payloads.append(dict(payload))
        )

        widget.apply_phase(
            {
                "mode": PHASE_VALIDATION,
                "title": "validation",
                "detail": "",
                "flicker": True,
                "cue_freq": None,
            }
        )
        widget._maybe_emit_active_phase_frame_presented(t_sec=0.0, frame_index=0)
        widget.apply_phase(
            {
                "mode": PHASE_CAL_ACTIVE,
                "title": "active",
                "detail": "",
                "flicker": True,
                "cue_freq": 8.0,
            }
        )
        widget._maybe_emit_active_phase_frame_presented(t_sec=0.1, frame_index=6)

        assert [payload["mode"] for payload in payloads] == [PHASE_VALIDATION, PHASE_CAL_ACTIVE]
    finally:
        widget.stop_clock()
        widget.close()


def test_four_arrow_stim_widget_emits_validation_frame_presented_once() -> None:
    _ = _get_qapp()
    widget = FourArrowStimWidget(
        freqs=(8.0, 10.0, 12.0, 15.0),
        refresh_rate_hz=60.0,
        mean=0.5,
        amp=0.5,
        phi=0.0,
        stimulus_mode=STIMULUS_MODE_ELAPSED_TIME_SINE,
    )
    try:
        payloads: list[dict[str, object]] = []
        widget.active_phase_frame_presented.connect(  # type: ignore[arg-type]
            lambda payload: payloads.append(dict(payload))
        )
        validation_payload = {
            "mode": PHASE_VALIDATION,
            "title": "validation",
            "detail": "",
            "flicker": True,
            "cue_freq": None,
        }

        widget.apply_phase(validation_payload)
        widget._maybe_emit_active_phase_frame_presented(t_sec=0.0, frame_index=0)
        widget.apply_phase({**validation_payload, "title": "still validation"})
        widget._maybe_emit_active_phase_frame_presented(t_sec=0.1, frame_index=6)

        assert len(payloads) == 1
        assert payloads[0]["mode"] == PHASE_VALIDATION
    finally:
        widget.stop_clock()
        widget.close()


def test_dataset_collection_window_embedded_preview_disables_flicker_during_fullscreen() -> None:
    _ = _get_qapp()
    window = DatasetCollectionWindow(serial_port="auto", board_id=0, freqs=(8.0, 10.0, 12.0, 15.0))
    try:
        phase = {"mode": PHASE_CAL_ACTIVE, "title": "采集中", "detail": "", "flicker": True, "cue_freq": 8.0}
        assert bool(window._phase_for_embedded_preview(phase).get("flicker")) is True
        window.fullscreen_window = object()  # type: ignore[assignment]
        preview_phase = window._phase_for_embedded_preview(phase)
        assert bool(preview_phase.get("flicker")) is False
        assert float(preview_phase.get("cue_freq", 0.0)) == 8.0
    finally:
        window.fullscreen_window = None
        window.close()


def test_dataset_collection_window_writes_resolved_port_back_after_connect() -> None:
    _ = _get_qapp()
    window = DatasetCollectionWindow(serial_port="auto", board_id=0, freqs=(8.0, 10.0, 12.0, 15.0))
    try:
        window.serial_edit.setText("auto")
        window._on_connected(
            {
                "requested_serial_port": "auto",
                "resolved_serial_port": "COM7",
                "attempted_ports": ["COM7", "COM8"],
                "sampling_rate": 250,
                "ready_samples": 512,
            }
        )
        assert window.serial_edit.text() == "COM7"
        log_text = window.log_text.toPlainText()
        assert "COM7" in log_text
        assert "下次采集" in log_text
    finally:
        window.close()


def test_dataset_collection_window_connect_skips_device_in_simulation_mode() -> None:
    _ = _get_qapp()
    window = DatasetCollectionWindow(
        serial_port="auto",
        board_id=0,
        freqs=(8.0, 10.0, 12.0, 15.0),
        simulation_only_default=True,
    )
    try:
        window._connect_device()
        assert window.connect_thread is None
        assert window.connect_worker is None
        assert "流程测试模式" in window.phase_label.text()
        assert "不会连接设备" in window.log_text.toPlainText()
    finally:
        window.close()


def test_dataset_collection_window_simulation_done_does_not_advance_rounds() -> None:
    _ = _get_qapp()
    window = DatasetCollectionWindow(
        serial_port="auto",
        board_id=0,
        freqs=(8.0, 10.0, 12.0, 15.0),
        simulation_only_default=True,
    )
    try:
        window.rounds_completed = 1
        window._refresh_round_status()
        window._on_done(
            {
                "simulation_only": True,
                "collection_aborted": False,
                "executed_trials": 4,
                "total_trials": 4,
                "round_index": 2,
                "rounds_planned": 3,
            }
        )
        assert window.rounds_completed == 1
        assert "流程测试完成" in window.phase_label.text()
        assert "未连接设备，未保存数据" in window.log_text.toPlainText()
    finally:
        window.close()


def test_dataset_collection_window_close_during_collection_requests_stop_without_closing() -> None:
    _ = _get_qapp()
    window = DatasetCollectionWindow(serial_port="auto", board_id=0, freqs=(8.0, 10.0, 12.0, 15.0))

    class FakeWorker:
        def __init__(self) -> None:
            self.request_stop_count = 0

        def request_stop(self) -> None:
            self.request_stop_count += 1

    class FakeCloseEvent:
        def __init__(self) -> None:
            self.accepted = False
            self.ignored = False

        def accept(self) -> None:
            self.accepted = True

        def ignore(self) -> None:
            self.ignored = True

    fake_worker = FakeWorker()
    close_event = FakeCloseEvent()
    try:
        window.worker = fake_worker  # type: ignore[assignment]
        window.worker_thread = object()  # type: ignore[assignment]
        window.closeEvent(close_event)
        assert fake_worker.request_stop_count == 1
        assert close_event.ignored is True
        assert close_event.accepted is False
        assert window.phase_label.text() == "正在停止..."
        assert "等待完成后再关闭窗口" in window.log_text.toPlainText()
    finally:
        window.worker = None
        window.worker_thread = None
        window.close()


def test_stimulus_sample_window_alignment_metadata_accounts_for_prearm_and_start_cue() -> None:
    refresh_rate_hz = 60.0
    freqs = (8.0, 10.0, 12.0, 15.0)
    metadata = stimulus_sample_window_alignment_metadata(
        freqs,
        refresh_rate_hz=refresh_rate_hz,
        backend=STIMULUS_BACKEND_PYQT_FULLSCREEN,
    )

    assert math.isclose(estimate_active_stimulus_arm_sec(refresh_rate_hz), 1.0 / 60.0)
    assert estimate_stimulus_sample_window_frame_offset(refresh_rate_hz) == 8
    assert metadata["stimulus_sequence_t0_reference"] == "active_phase_ui_apply"
    assert metadata["stimulus_sample_window_alignment"] == "prearmed_before_eeg_buffer_clear"
    assert metadata["stimulus_sample_window_frame_offset_estimate"] == 8
    assert math.isclose(
        float(metadata["stimulus_sample_window_offset_sec_estimate"]),
        (1.0 / 60.0) + float(ACTIVE_START_CUE_SEC),
    )
    assert math.isclose(
        float(metadata["stimulus_sample_window_display_frame_offset_sec_estimate"]),
        8.0 / 60.0,
    )

    cycles = dict(metadata["stimulus_sample_window_phase_cycles_by_freq"])
    radians = dict(metadata["stimulus_sample_window_phase_rad_by_freq"])
    for freq in freqs:
        key = f"{freq:g}Hz"
        expected_cycles = (freq * 8.0 / refresh_rate_hz) % 1.0
        assert math.isclose(float(cycles[key]), expected_cycles, abs_tol=1e-12)
        assert math.isclose(float(radians[key]), expected_cycles * 2.0 * math.pi, abs_tol=1e-12)


def test_headless_sample_window_alignment_metadata_is_not_applicable() -> None:
    metadata = stimulus_sample_window_alignment_metadata(
        (8.0, 10.0, 12.0, 15.0),
        refresh_rate_hz=60.0,
        backend=STIMULUS_BACKEND_HEADLESS_NO_VISUAL,
    )
    assert metadata["stimulus_sequence_t0_reference"] == "not_rendered_by_this_process"
    assert metadata["stim_phi_reference"] == "not_applicable"
    assert metadata["stimulus_sample_window_frame_offset_estimate"] is None
    assert metadata["stimulus_sample_window_display_frame_offset_sec_estimate"] is None
    assert metadata["stimulus_sample_window_phase_rad_by_freq"] == {}


def test_stimulus_frequency_validation_uses_display_nyquist() -> None:
    validate_stimulus_frequency_set((8.0, 10.0, 12.0, 15.0), refresh_rate_hz=60.0)
    with pytest.raises(ValueError, match="below half the display refresh rate"):
        validate_stimulus_frequency_set((8.0, 10.0, 12.0, 30.0), refresh_rate_hz=60.0)


def test_validate_collection_protocol_enforces_timing_minimums() -> None:
    with pytest.raises(ValueError, match="prepare_sec must be >="):
        _validate_collection_protocol(
            prepare_sec=float(MIN_PREPARE_SEC_FOR_VOICE) - 0.1,
            active_sec=float(MIN_ACTIVE_SEC_FOR_TRAINING),
            rest_sec=float(MIN_REST_SEC_BETWEEN_TRIALS),
        )
    with pytest.raises(ValueError, match="active_sec must be >="):
        _validate_collection_protocol(
            prepare_sec=float(MIN_PREPARE_SEC_FOR_VOICE),
            active_sec=float(MIN_ACTIVE_SEC_FOR_TRAINING) - 0.1,
            rest_sec=float(MIN_REST_SEC_BETWEEN_TRIALS),
        )
    with pytest.raises(ValueError, match="rest_sec must be >="):
        _validate_collection_protocol(
            prepare_sec=float(MIN_PREPARE_SEC_FOR_VOICE),
            active_sec=float(MIN_ACTIVE_SEC_FOR_TRAINING),
            rest_sec=float(MIN_REST_SEC_BETWEEN_TRIALS) - 0.1,
        )
    with pytest.raises(ValueError, match="long_idle_sec must be 0 or >="):
        _validate_collection_protocol(
            prepare_sec=float(MIN_PREPARE_SEC_FOR_VOICE),
            active_sec=float(MIN_ACTIVE_SEC_FOR_TRAINING),
            rest_sec=float(MIN_REST_SEC_BETWEEN_TRIALS),
            long_idle_sec=float(MIN_ACTIVE_SEC_FOR_TRAINING) - 0.1,
        )
    _validate_collection_protocol(
        prepare_sec=float(MIN_PREPARE_SEC_FOR_VOICE),
        active_sec=float(MIN_ACTIVE_SEC_FOR_TRAINING),
        rest_sec=float(MIN_REST_SEC_BETWEEN_TRIALS),
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
    expected = (
        74.0
        * (
            float(DEFAULT_STABLE_PREPARE_SEC)
            + float(ACTIVE_START_CUE_SEC)
            + float(ACTIVE_STIMULUS_ARM_SEC)
            + float(DEFAULT_STABLE_ACTIVE_SEC)
            + float(DEFAULT_STABLE_REST_SEC)
        )
        + float(DEFAULT_STABLE_PREPARE_SEC)
        + float(ACTIVE_START_CUE_SEC)
        + float(ACTIVE_STIMULUS_ARM_SEC)
        + 60.0
        + float(DEFAULT_STABLE_REST_SEC)
    )
    assert abs(float(round_sec) - expected) < 1e-9


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


def test_protocol_signature_changes_when_stimulus_mode_changes() -> None:
    base_config = {
        "prepare_sec": 1.0,
        "active_sec": 5.0,
        "rest_sec": 4.0,
        "target_repeats": 10,
        "idle_repeats": 20,
        "switch_trials": 14,
        "long_idle_sec": 0.0,
        "stim_refresh_rate_hz": 60.0,
        "stimulus_backend": STIMULUS_BACKEND_PYQT_FULLSCREEN,
    }
    elapsed = build_protocol_signature(
        sampling_rate=250,
        protocol_config={**base_config, "stimulus_mode": STIMULUS_MODE_ELAPSED_TIME_SINE},
        freqs=(8.0, 10.0, 12.0, 15.0),
        board_eeg_channels=(1, 2, 3, 4, 5, 6, 7, 8),
    )
    frame_locked = build_protocol_signature(
        sampling_rate=250,
        protocol_config={**base_config, "stimulus_mode": STIMULUS_MODE_FRAME_LOCKED_SINE},
        freqs=(8.0, 10.0, 12.0, 15.0),
        board_eeg_channels=(1, 2, 3, 4, 5, 6, 7, 8),
    )
    assert elapsed != frame_locked


def test_protocol_signature_includes_comfort_stimulus_profile_fields() -> None:
    base_config = {
        "prepare_sec": 1.0,
        "active_sec": 5.0,
        "rest_sec": 4.0,
        "target_repeats": 10,
        "idle_repeats": 20,
        "switch_trials": 14,
        "long_idle_sec": 0.0,
        "stimulus_backend": STIMULUS_BACKEND_PYQT_FULLSCREEN,
        **stimulus_profile_metadata(
            STIMULUS_PROFILE_COMFORT_FBCCA_V1,
            stimulus_mode=STIMULUS_MODE_FRAME_LOCKED_SINE,
            refresh_rate_hz=240.0,
            mode_selection_reason="stable_240hz_frame_locked",
            comfort_rating=3,
            screen_brightness_note="50_percent",
            frame_interval_stats={"p95_ms": 4.3, "max_ms": 5.0},
        ),
    }
    base = build_protocol_signature(
        sampling_rate=250,
        protocol_config=base_config,
        freqs=(8.0, 10.0, 12.0, 15.0),
        board_eeg_channels=(1, 2, 3, 4, 5, 6, 7, 8),
    )
    changed_amp = build_protocol_signature(
        sampling_rate=250,
        protocol_config={**base_config, "stim_amp": 0.25, "stim_luminance_max": 0.65},
        freqs=(8.0, 10.0, 12.0, 15.0),
        board_eeg_channels=(1, 2, 3, 4, 5, 6, 7, 8),
    )
    changed_ramp = build_protocol_signature(
        sampling_rate=250,
        protocol_config={**base_config, "ramp_sec": 0.0},
        freqs=(8.0, 10.0, 12.0, 15.0),
        board_eeg_channels=(1, 2, 3, 4, 5, 6, 7, 8),
    )
    changed_profile = build_protocol_signature(
        sampling_rate=250,
        protocol_config={**base_config, "stimulus_profile_id": "legacy_full_contrast"},
        freqs=(8.0, 10.0, 12.0, 15.0),
        board_eeg_channels=(1, 2, 3, 4, 5, 6, 7, 8),
    )
    changed_frame_stats = build_protocol_signature(
        sampling_rate=250,
        protocol_config={**base_config, "frame_interval_stats": {"p95_ms": 99.0}},
        freqs=(8.0, 10.0, 12.0, 15.0),
        board_eeg_channels=(1, 2, 3, 4, 5, 6, 7, 8),
    )
    changed_comfort = build_protocol_signature(
        sampling_rate=250,
        protocol_config={**base_config, "comfort_rating": 4},
        freqs=(8.0, 10.0, 12.0, 15.0),
        board_eeg_channels=(1, 2, 3, 4, 5, 6, 7, 8),
    )
    changed_brightness_note = build_protocol_signature(
        sampling_rate=250,
        protocol_config={**base_config, "screen_brightness_note": "80_percent"},
        freqs=(8.0, 10.0, 12.0, 15.0),
        board_eeg_channels=(1, 2, 3, 4, 5, 6, 7, 8),
    )

    assert base != changed_amp
    assert base != changed_ramp
    assert base != changed_profile
    assert base != changed_frame_stats
    assert base != changed_comfort
    assert base != changed_brightness_note


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
        active_sec = 60.0 if str(trial.label) == "long_idle" else 5.0
        trial_segments.append((trial, np.zeros((samples, 8), dtype=np.float64)))
        quality_rows.append(
            {
                "order_index": order_index,
                "target_samples": samples,
                "used_samples": samples,
                "active_sec": active_sec,
                "retry_count": 0,
            }
        )
    temp_root = PROJECT_DIR / ".tmp_test_artifacts"
    temp_root.mkdir(parents=True, exist_ok=True)
    dataset_root = temp_root / f"ssvep_collection_test_{uuid4().hex}"
    dataset_root.mkdir(parents=True, exist_ok=False)
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
        assert abs(float(long_idle_rows[0]["active_sec"]) - 60.0) < 1e-9
    finally:
        shutil.rmtree(dataset_root, ignore_errors=True)
