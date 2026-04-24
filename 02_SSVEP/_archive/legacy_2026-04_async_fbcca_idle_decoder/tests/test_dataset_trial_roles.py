from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from async_fbcca_idle_standalone import TrialSpec
from ssvep_core.dataset import infer_trial_role, save_collection_dataset_bundle, summarize_collection_manifest


def test_infer_trial_role_mapping() -> None:
    assert infer_trial_role(label="8Hz", expected_freq=8.0) == "control"
    assert infer_trial_role(label="idle_center", expected_freq=None) == "clean_idle"
    assert infer_trial_role(label="switch_to_10", expected_freq=None) == "hard_idle"
    assert infer_trial_role(label="long_idle_block", expected_freq=None) == "hard_idle"


def test_saved_manifest_contains_trial_role_counts() -> None:
    tmp_path = PROJECT_DIR / ".tmp_role_test" / f"case_{uuid.uuid4().hex[:8]}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    fs = 250
    samples = int(4.0 * fs)
    data_control = np.zeros((samples, 8), dtype=np.float64)
    data_idle = np.zeros((samples, 8), dtype=np.float64)
    data_hard = np.zeros((samples, 8), dtype=np.float64)
    segments = [
        (TrialSpec(label="8Hz", expected_freq=8.0, trial_id=1, block_index=0), data_control),
        (TrialSpec(label="idle", expected_freq=None, trial_id=2, block_index=0), data_idle),
        (TrialSpec(label="switch_to_10", expected_freq=None, trial_id=3, block_index=0), data_hard),
    ]
    try:
        payload = save_collection_dataset_bundle(
            dataset_root=tmp_path,
            session_id="s1",
            subject_id="sub01",
            serial_port="COM3",
            board_id=0,
            sampling_rate=fs,
            freqs=(8.0, 10.0, 12.0, 15.0),
            board_eeg_channels=tuple(range(8)),
            protocol_config={"prepare_sec": 1.0, "active_sec": 4.0, "rest_sec": 1.0, "step_sec": 0.25},
            trial_segments=segments,
        )
        manifest_path = Path(payload["dataset_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert all("trial_role" in row for row in manifest.get("trials", []))
        summary = summarize_collection_manifest(manifest_path)
        role_counts = dict(summary.get("trial_role_counts", {}))
        assert int(role_counts.get("control", 0)) == 1
        assert int(role_counts.get("clean_idle", 0)) == 1
        assert int(role_counts.get("hard_idle", 0)) == 1
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
