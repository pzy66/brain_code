from __future__ import annotations

import json
import shutil
import sys
from uuid import uuid4
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from async_fbcca_idle_standalone import TrialSpec, save_benchmark_dataset_bundle


def test_save_benchmark_dataset_bundle_writes_manifest_and_npz() -> None:
    scratch_root = PROJECT_DIR / "profiles" / "datasets_test_artifacts"
    scratch_root.mkdir(parents=True, exist_ok=True)
    temp_dir = scratch_root / f"dataset_bundle_{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    calibration_segments = [
        (
            TrialSpec(label="8Hz", expected_freq=8.0, trial_id=1, block_index=0),
            np.ones((100, 4), dtype=np.float64),
        ),
        (
            TrialSpec(label="idle", expected_freq=None, trial_id=2, block_index=0),
            np.zeros((100, 4), dtype=np.float64),
        ),
    ]
    evaluation_segments = [
        (
            TrialSpec(label="switch_to_10Hz", expected_freq=10.0, trial_id=10, block_index=1),
            np.full((120, 4), 2.0, dtype=np.float64),
        )
    ]

    try:
        output = save_benchmark_dataset_bundle(
            dataset_root=temp_dir,
            session_id="benchmark_session_test",
            serial_port="COM4",
            board_id=0,
            sampling_rate=250,
            freqs=(8.0, 10.0, 12.0, 15.0),
            board_eeg_channels=(0, 1, 2, 3),
            protocol_config={"active_sec": 4.0},
            calibration_segments=calibration_segments,
            evaluation_segments=evaluation_segments,
            split_trial_ids={"train": [10], "gate": [], "holdout": []},
            benchmark_summary={"chosen_model": "fbcca"},
        )

        manifest_path = Path(output["dataset_manifest"])
        npz_path = Path(output["dataset_npz"])
        assert manifest_path.exists()
        assert npz_path.exists()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        npz = np.load(npz_path)
        keys = set(npz.files)
        assert keys

        for trial in manifest["trials"]:
            key = str(trial["npz_key"])
            assert key in keys
            arr = np.asarray(npz[key])
            assert arr.shape[0] == int(trial["used_samples"])
            assert arr.shape[1] == int(trial["channels"])
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
