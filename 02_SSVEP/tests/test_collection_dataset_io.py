from __future__ import annotations

import shutil
import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core.async_fbcca_idle_standalone import TrialSpec
from apps.async_fbcca_validation_ui import STIMULUS_MODE_FRAME_LOCKED_SINE
from apps.data_collection_ui import STIMULUS_BACKEND_PYQT_FULLSCREEN
import ssvep_core.dataset as dataset_module
from ssvep_core.dataset import load_collection_dataset, save_collection_dataset_bundle


def _mock_segment(samples: int, channels: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(rng.standard_normal((samples, channels)), dtype=np.float64)


def test_collection_dataset_bundle_roundtrip() -> None:
    artifacts = PROJECT_DIR / ".tmp_test_artifacts" / "datasets_test_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    case_dir = artifacts / "collection_roundtrip_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    try:
        segments = [
            (TrialSpec(label="8Hz", expected_freq=8.0, trial_id=0, block_index=0), _mock_segment(1000, 4, 1)),
            (TrialSpec(label="idle", expected_freq=None, trial_id=1, block_index=0), _mock_segment(980, 4, 2)),
            (TrialSpec(label="switch_to_10Hz", expected_freq=10.0, trial_id=2, block_index=1), _mock_segment(1000, 4, 3)),
        ]
        payload = save_collection_dataset_bundle(
            dataset_root=case_dir,
            session_id="session_test_001",
            subject_id="subject_test",
            serial_port="COM4",
            board_id=0,
            sampling_rate=250,
            freqs=(8.0, 10.0, 12.0, 15.0),
            board_eeg_channels=(0, 1, 2, 3),
            protocol_config={
                "protocol_name": "enhanced_45m",
                "active_sec": 4.0,
                "collection_aborted": False,
                "requested_session_id": "session_test_001",
                "saved_session_id": "session_test_001",
                "active_start_buffer_clear_timing": "before_start_cue",
                "active_saved_window": "last_active_sec_after_start_cue",
                "active_end_cue_timing": "after_segment_capture",
                "stimulus_mode": STIMULUS_MODE_FRAME_LOCKED_SINE,
                "stim_refresh_rate_hz": 60.0,
                "stim_mean": 0.5,
                "stim_amp": 0.5,
                "stim_phi": 0.0,
                "stim_frame_formula": "luminance(frame)=mean+amp*sin(2*pi*freq*frame/refresh_rate_hz+phi)",
                "stimulus_backend": STIMULUS_BACKEND_PYQT_FULLSCREEN,
                "stimulus_rendered_by_this_process": True,
                "stimulus_sample_window_frame_offset_estimate": 8,
                "stimulus_sample_window_display_frame_offset_sec_estimate": 8.0 / 60.0,
                "stimulus_sample_window_phase_cycles_by_freq": {
                    "8Hz": 0.06666666666666665,
                    "10Hz": 0.33333333333333326,
                    "12Hz": 0.6,
                    "15Hz": 0.0,
                },
            },
            trial_segments=segments,
            quality_rows=[
                {
                    "order_index": 0,
                    "target_samples": 1000,
                    "active_sec": 4.0,
                    "sample_ratio": 1.0,
                    "retry_count": 0,
                    "active_start_tone_started_at": "2026-04-24T10:00:00+08:00",
                    "active_window_started_at": "2026-04-24T10:00:00.120+08:00",
                    "active_window_ended_at": "2026-04-24T10:00:04.120+08:00",
                    "segment_captured_at": "2026-04-24T10:00:04.130+08:00",
                    "active_end_tone_started_at": "2026-04-24T10:00:04.131+08:00",
                    "stimulus_phase_apply_requested_at": "2026-04-24T10:00:00.010+08:00",
                    "stimulus_first_frame_presented_at": "2026-04-24T10:00:00.026+08:00",
                    "stimulus_first_frame_presented_t_sec": 0.016,
                    "stimulus_first_frame_frame_index": 0,
                    "stimulus_first_frame_cue_freq": 8.0,
                    "stimulus_first_frame_mode": "calibration_active",
                    "stimulus_first_frame_ack_latency_sec": 0.016,
                    "stimulus_first_frame_ack_timed_out": False,
                    "stimulus_frame_interval_stats": {
                        "count": 240,
                        "mean_ms": 4.1667,
                        "p95_ms": 4.4,
                        "max_ms": 5.2,
                        "refresh_rate_hz_estimate": 240.0,
                    },
                    "stimulus_profile_id": "comfort_fbcca_v1",
                    "stim_mean": 0.4,
                    "stim_amp": 0.2,
                    "ramp_sec": 0.3,
                    "board_buffer_cleared_at": "2026-04-24T10:00:00.027+08:00",
                    "board_buffer_clear_samples": 32,
                },
                {"order_index": 1, "target_samples": 1000, "active_sec": 4.0, "sample_ratio": 0.98, "retry_count": 2},
                {"order_index": 2, "target_samples": 1000, "active_sec": 4.0, "sample_ratio": 1.0, "retry_count": 1},
            ],
            continuous_board_data=np.arange(6 * 24, dtype=np.float64).reshape(6, 24),
            continuous_board_info={
                "marker_channel": 4,
                "timestamp_channel": 5,
                "source": "unit_test",
            },
        )
        loaded = load_collection_dataset(Path(payload["dataset_manifest"]))
        assert loaded.session_id == "session_test_001"
        assert loaded.subject_id == "subject_test"
        assert loaded.sampling_rate == 250
        assert loaded.freqs == (8.0, 10.0, 12.0, 15.0)
        assert loaded.board_eeg_channels == (0, 1, 2, 3)
        assert len(loaded.trial_segments) == len(segments)
        trial_rows = list(loaded.manifest.get("trials", []))
        assert len(trial_rows) == len(segments)
        assert isinstance(loaded.manifest.get("protocol_signature", ""), str)
        assert str(loaded.manifest.get("protocol_signature", "")).startswith("sha1:")
        assert str(loaded.manifest.get("protocol_config", {}).get("protocol_signature", "")).startswith("sha1:")
        protocol_config = dict(loaded.manifest.get("protocol_config", {}))
        assert protocol_config.get("stimulus_mode") == STIMULUS_MODE_FRAME_LOCKED_SINE
        assert protocol_config.get("collection_aborted") is False
        assert protocol_config.get("requested_session_id") == "session_test_001"
        assert protocol_config.get("saved_session_id") == "session_test_001"
        assert protocol_config.get("active_start_buffer_clear_timing") == "before_start_cue"
        assert protocol_config.get("active_saved_window") == "last_active_sec_after_start_cue"
        assert protocol_config.get("active_end_cue_timing") == "after_segment_capture"
        assert protocol_config.get("stimulus_backend") == STIMULUS_BACKEND_PYQT_FULLSCREEN
        assert bool(protocol_config.get("stimulus_rendered_by_this_process")) is True
        assert int(protocol_config.get("stimulus_sample_window_frame_offset_estimate", 0)) == 8
        assert (
            abs(
                float(protocol_config.get("stimulus_sample_window_display_frame_offset_sec_estimate", 0.0))
                - (8.0 / 60.0)
            )
            < 1e-9
        )
        assert abs(float(protocol_config.get("stim_refresh_rate_hz", 0.0)) - 60.0) < 1e-9
        assert str(protocol_config.get("stim_frame_formula", "")).startswith("luminance(frame)")
        frame_stats = dict(protocol_config.get("frame_interval_stats", {}))
        assert int(frame_stats.get("trial_count", 0)) == len(segments)
        assert int(frame_stats.get("nonempty_trial_count", 0)) == 1
        assert int(frame_stats.get("sample_count_total", 0)) == 240
        assert abs(float(frame_stats.get("p95_ms_max", 0.0)) - 4.4) < 1e-9
        assert abs(float(frame_stats.get("max_ms_max", 0.0)) - 5.2) < 1e-9
        generated_at = str(loaded.manifest.get("generated_at", ""))
        assert datetime.fromisoformat(generated_at).tzinfo is not None
        quality_summary = dict(loaded.manifest.get("quality_summary", {}))
        assert int(quality_summary.get("valid_trial_count", 0)) == len(segments)
        assert bool(quality_summary.get("collection_aborted", True)) is False
        assert int(quality_summary.get("planned_trial_count", 0)) == len(segments)
        assert int(quality_summary.get("saved_trial_count", 0)) == len(segments)
        assert int(quality_summary.get("short_segment_excluded", -1)) == 0
        assert int(quality_summary.get("stimulus_first_frame_ack_timeout_count", -1)) == 0
        files = dict(loaded.manifest.get("files", {}))
        continuous_path = Path(str(files.get("continuous_board_npz", "")))
        assert continuous_path.exists()
        continuous_meta = dict(loaded.manifest.get("continuous_board", {}))
        assert continuous_meta.get("shape") == [6, 24]
        assert continuous_meta.get("marker_channel") == 4
        for row in trial_rows:
            assert int(row.get("target_samples", 0)) == 1000
            assert abs(float(row.get("active_sec", 0.0)) - 4.0) < 1e-9
            assert float(row.get("sample_ratio", 0.0)) > 0.0
            assert float(row.get("shortfall_ratio", 0.0)) >= 0.0
            assert int(row.get("retry_count", 0)) >= 0
        assert int(trial_rows[1].get("retry_count", 0)) == 2
        assert str(trial_rows[0].get("active_start_tone_started_at", "")) == "2026-04-24T10:00:00+08:00"
        assert str(trial_rows[0].get("stimulus_first_frame_presented_at", "")).strip()
        assert int(trial_rows[0].get("stimulus_first_frame_frame_index", -1)) == 0
        assert int(dict(trial_rows[0].get("stimulus_frame_interval_stats", {})).get("count", 0)) == 240
        assert str(trial_rows[0].get("stimulus_profile_id", "")) == "comfort_fbcca_v1"
        assert abs(float(trial_rows[0].get("stim_mean", 0.0)) - 0.4) < 1e-9
        assert abs(float(trial_rows[0].get("stim_amp", 0.0)) - 0.2) < 1e-9
        assert abs(float(trial_rows[0].get("ramp_sec", 0.0)) - 0.3) < 1e-9
        assert int(trial_rows[0].get("board_buffer_clear_samples", 0)) == 32
        assert datetime.fromisoformat(str(trial_rows[0].get("active_window_started_at", ""))).tzinfo is not None
        assert datetime.fromisoformat(str(trial_rows[0].get("segment_captured_at", ""))).tzinfo is not None
        for (trial_a, seg_a), (trial_b, seg_b) in zip(segments, loaded.trial_segments):
            assert trial_a.label == trial_b.label
            assert trial_a.expected_freq == trial_b.expected_freq
            assert seg_a.shape == seg_b.shape
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_collection_dataset_bundle_infers_heterogeneous_trial_metadata_without_quality_rows() -> None:
    artifacts = PROJECT_DIR / ".tmp_test_artifacts" / "datasets_test_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    case_dir = artifacts / "collection_heterogeneous_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    try:
        segments = [
            (TrialSpec(label="8Hz", expected_freq=8.0, trial_id=0, block_index=0), _mock_segment(375, 4, 21)),
            (TrialSpec(label="idle", expected_freq=None, trial_id=1, block_index=0), _mock_segment(512, 4, 22)),
            (TrialSpec(label="long_idle", expected_freq=None, trial_id=2, block_index=1), _mock_segment(1500, 4, 23)),
        ]
        payload = save_collection_dataset_bundle(
            dataset_root=case_dir,
            session_id="session_test_heterogeneous",
            subject_id="subject_test",
            serial_port="COM4",
            board_id=0,
            sampling_rate=250,
            freqs=(8.0, 10.0, 12.0, 15.0),
            board_eeg_channels=(0, 1, 2, 3),
            protocol_config={"protocol_name": "custom", "active_sec": 1.5},
            trial_segments=segments,
        )
        loaded = load_collection_dataset(Path(payload["dataset_manifest"]))
        trial_rows = list(loaded.manifest.get("trials", []))
        assert [seg.shape for _trial, seg in loaded.trial_segments] == [(375, 4), (512, 4), (1500, 4)]
        assert [int(row.get("target_samples", 0)) for row in trial_rows] == [375, 512, 1500]
        assert [float(row.get("sample_ratio", 0.0)) for row in trial_rows] == [1.0, 1.0, 1.0]
        assert abs(float(trial_rows[0].get("active_sec", 0.0)) - 1.5) < 1e-9
        assert abs(float(trial_rows[1].get("active_sec", 0.0)) - (512.0 / 250.0)) < 1e-9
        assert abs(float(trial_rows[2].get("active_sec", 0.0)) - 6.0) < 1e-9
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_collection_dataset_bundle_sanitizes_session_directory() -> None:
    artifacts = PROJECT_DIR / ".tmp_test_artifacts" / "datasets_test_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    case_dir = artifacts / "collection_session_sanitize_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload = save_collection_dataset_bundle(
            dataset_root=case_dir,
            session_id=r"..\outside/session:bad",
            subject_id=r"subject/name:bad",
            serial_port="COM4",
            board_id=0,
            sampling_rate=250,
            freqs=(8.0, 10.0, 12.0, 15.0),
            board_eeg_channels=(0, 1, 2, 3),
            protocol_config={"protocol_name": "custom", "active_sec": 1.5},
            trial_segments=[
                (TrialSpec(label="8Hz", expected_freq=8.0, trial_id=0, block_index=0), _mock_segment(375, 4, 31)),
            ],
        )

        manifest_path = Path(payload["dataset_manifest"]).resolve()
        manifest_path.relative_to(case_dir.resolve())
        assert manifest_path.parent.name == "outside_session_bad"

        loaded = load_collection_dataset(manifest_path)
        assert loaded.session_id == "outside_session_bad"
        assert loaded.subject_id == "subject_name_bad"
        protocol_config = dict(loaded.manifest.get("protocol_config", {}))
        assert protocol_config["requested_session_id"] == r"..\outside/session:bad"
        assert protocol_config["saved_session_id"] == "outside_session_bad"
        assert protocol_config["requested_subject_id"] == r"subject/name:bad"
        assert protocol_config["saved_subject_id"] == "subject_name_bad"
        assert not (case_dir.parent / "outside" / "session:bad").exists()
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_collection_dataset_bundle_falls_back_when_continuous_npz_save_fails(monkeypatch) -> None:
    artifacts = PROJECT_DIR / ".tmp_test_artifacts" / "datasets_test_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    case_dir = artifacts / "collection_continuous_fallback_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    original_save_npz = dataset_module._atomic_save_npz

    def flaky_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
        if Path(path).name == "continuous_board.npz":
            raise OSError("simulated continuous npz failure")
        original_save_npz(path, arrays)

    monkeypatch.setattr(dataset_module, "_atomic_save_npz", flaky_save_npz)
    try:
        payload = save_collection_dataset_bundle(
            dataset_root=case_dir,
            session_id="fallback_session",
            subject_id="subject001",
            serial_port="COM4",
            board_id=0,
            sampling_rate=250,
            freqs=(8.0, 10.0, 12.0, 15.0),
            board_eeg_channels=(0, 1, 2, 3),
            protocol_config={"protocol_name": "custom", "active_sec": 1.5},
            trial_segments=[
                (TrialSpec(label="8Hz", expected_freq=8.0, trial_id=0, block_index=0), _mock_segment(375, 4, 41)),
            ],
            continuous_board_data=_mock_segment(750, 6, 42).T,
        )

        manifest_path = Path(payload["dataset_manifest"])
        assert manifest_path.exists()
        assert Path(payload["dataset_npz"]).exists()
        assert Path(payload["dataset_continuous_board_npy"]).exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["continuous_board"]["saved"] is True
        assert manifest["continuous_board"]["format"] == "npy"
        assert "simulated continuous npz failure" in manifest["continuous_board"]["compressed_npz_save_error"]
        loaded = load_collection_dataset(manifest_path)
        assert loaded.session_id == "fallback_session"
        assert len(loaded.trial_segments) == 1
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_collection_dataset_loader_falls_back_to_manifest_sibling_npz() -> None:
    artifacts = PROJECT_DIR / ".tmp_test_artifacts" / "datasets_test_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    case_dir = artifacts / "collection_relocated_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    try:
        segments = [
            (TrialSpec(label="8Hz", expected_freq=8.0, trial_id=0, block_index=0), _mock_segment(1000, 4, 11)),
        ]
        payload = save_collection_dataset_bundle(
            dataset_root=case_dir,
            session_id="session_test_relocated",
            subject_id="subject_test",
            serial_port="COM4",
            board_id=0,
            sampling_rate=250,
            freqs=(8.0, 10.0, 12.0, 15.0),
            board_eeg_channels=(0, 1, 2, 3),
            protocol_config={"protocol_name": "stable_12m", "active_sec": 4.0},
            trial_segments=segments,
        )
        manifest_path = Path(payload["dataset_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["raw_trials_npz"] = r"C:\old_acquisition_pc\session\raw_trials.npz"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        loaded = load_collection_dataset(manifest_path)

        assert loaded.npz_path == (manifest_path.parent / "raw_trials.npz").resolve()
        assert loaded.session_id == "session_test_relocated"
        assert len(loaded.trial_segments) == 1
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)
