from __future__ import annotations

import shutil
import sys
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
        )
        loaded = load_collection_dataset(Path(payload["dataset_manifest"]))
        assert loaded.session_id == "session_test_001"
        assert loaded.subject_id == "subject_test"
        assert loaded.sampling_rate == 250
        assert loaded.freqs == (8.0, 10.0, 12.0, 15.0)
        assert loaded.board_eeg_channels == (0, 1, 2, 3)
        assert len(loaded.trial_segments) == len(segments)
        for (trial_a, seg_a), (trial_b, seg_b) in zip(segments, loaded.trial_segments):
            assert trial_a.label == trial_b.label
            assert trial_a.expected_freq == trial_b.expected_freq
            assert seg_a.shape == seg_b.shape
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)
