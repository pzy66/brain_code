from __future__ import annotations

import shutil
import sys
import json
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from async_fbcca_idle_standalone import TrialSpec
from ssvep_core.dataset import load_collection_dataset, save_collection_dataset_bundle


def _mock_segment(samples: int, channels: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(rng.standard_normal((samples, channels)), dtype=np.float64)


def test_collection_dataset_bundle_roundtrip() -> None:
    artifacts = PROJECT_DIR / "profiles" / "datasets_test_artifacts"
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
            protocol_config={"protocol_name": "enhanced_45m", "active_sec": 4.0},
            trial_segments=segments,
            quality_rows=[
                {"order_index": 0, "target_samples": 1000, "retry_count": 0},
                {"order_index": 1, "target_samples": 1000, "retry_count": 2},
                {"order_index": 2, "target_samples": 1000, "retry_count": 1},
            ],
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
        quality_summary = dict(loaded.manifest.get("quality_summary", {}))
        assert int(quality_summary.get("valid_trial_count", 0)) == len(segments)
        assert int(quality_summary.get("short_segment_excluded", -1)) == 0
        for row in trial_rows:
            assert int(row.get("target_samples", 0)) == 1000
            assert float(row.get("shortfall_ratio", 0.0)) >= 0.0
            assert int(row.get("retry_count", 0)) >= 0
        assert int(trial_rows[1].get("retry_count", 0)) == 2
        for (trial_a, seg_a), (trial_b, seg_b) in zip(segments, loaded.trial_segments):
            assert trial_a.label == trial_b.label
            assert trial_a.expected_freq == trial_b.expected_freq
            assert seg_a.shape == seg_b.shape
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_collection_dataset_loader_falls_back_to_manifest_sibling_npz() -> None:
    artifacts = PROJECT_DIR / "profiles" / "datasets_test_artifacts"
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
