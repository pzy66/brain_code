from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from apps.training_evaluation_ui import _parse_task, build_parser
from ssvep_core.async_fbcca_idle_standalone import TrialSpec, load_profile
from ssvep_core.dataset import LoadedDataset
from ssvep_core.fbcca_threshold_pretrain import (
    DEFAULT_FBCCA_THRESHOLD_TASK,
    FBCCAThresholdPretrainConfig,
    run_fbcca_threshold_pretrain,
)


class _FakeFBCCADecoder:
    requires_fit = False
    compute_backend_used = "cpu"

    def __init__(self) -> None:
        self.freqs = (8.0, 10.0, 12.0, 15.0)
        self.win_sec = 1.0
        self.step_sec = 0.25
        self.win_samples = 1
        self.model_params = {
            "Nh": 3,
            "_decoder_model_name": "fbcca_fixed_all8",
            "subband_weight_mode": "chen_fixed",
            "compute_backend": "cpu",
        }

    def iter_window_features(self, segment, *, expected_freq, label, trial_id, block_index):
        _ = segment
        for window_index in range(2):
            is_control = expected_freq is not None
            top1 = 0.90 if is_control else 0.10
            top2 = 0.12 if is_control else 0.09
            pred_freq = float(expected_freq if expected_freq is not None else 8.0)
            yield {
                "pred_freq": pred_freq,
                "top1_score": top1,
                "top2_score": top2,
                "ratio": top1 / max(top2, 1e-6),
                "margin": top1 - top2,
                "normalized_top1": 0.80 if is_control else 0.30,
                "score_entropy": 0.20 if is_control else 1.20,
                "correct": bool(is_control),
                "label": str(label),
                "expected_freq": None if expected_freq is None else float(expected_freq),
                "trial_id": int(trial_id),
                "block_index": int(block_index),
                "window_index": int(window_index),
            }


def _fake_dataset(tmp_root: Path) -> LoadedDataset:
    trials = [
        TrialSpec(label=f"{freq:g}Hz", expected_freq=float(freq), trial_id=index, block_index=index)
        for index, freq in enumerate((8.0, 10.0, 12.0, 15.0, 8.0, 10.0, 12.0, 15.0))
    ]
    trials.extend(
        [
            TrialSpec(label="idle", expected_freq=None, trial_id=100, block_index=100),
            TrialSpec(label="transition_idle", expected_freq=None, trial_id=101, block_index=101),
        ]
    )
    segment = np.zeros((250, 8), dtype=np.float64)
    return LoadedDataset(
        manifest_path=tmp_root / "session_manifest.json",
        npz_path=tmp_root / "raw_trials.npz",
        session_id="s1",
        subject_id="subject-test",
        sampling_rate=250,
        freqs=(8.0, 10.0, 12.0, 15.0),
        board_eeg_channels=(1, 2, 3, 4, 5, 6, 7, 8),
        protocol_config={"active_sec": 1.0, "step_sec": 0.25},
        trial_segments=[(trial, segment.copy()) for trial in trials],
        manifest={"trials": []},
    )


def test_threshold_pretrain_saves_default_fbcca_profile(monkeypatch) -> None:
    import ssvep_core.fbcca_threshold_pretrain as module

    tmp_root = PROJECT_DIR / ".tmp_test_artifacts" / f"fbcca_threshold_{uuid.uuid4().hex}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    try:
        dataset = _fake_dataset(tmp_root)
        monkeypatch.setattr(module, "load_collection_dataset", lambda _path: dataset)
        monkeypatch.setattr(module, "create_decoder", lambda *args, **kwargs: _FakeFBCCADecoder())

        config = FBCCAThresholdPretrainConfig(
            dataset_manifest_session1=tmp_root / "session_manifest.json",
            output_profile_path=tmp_root / "profile.json",
            report_path=tmp_root / "report.json",
            report_root_dir=tmp_root,
            organize_report_dir=False,
            win_sec=1.0,
            publish_realtime=False,
        )
        payload = run_fbcca_threshold_pretrain(config, log_fn=lambda _msg: None)

        profile_path = Path(payload["profile_path"])
        profile_v2_path = Path(payload["profile_v2_path"])
        assert profile_path.exists()
        assert profile_v2_path.exists()
        assert Path(payload["report_path"]).exists()
        profile = load_profile(profile_path, require_exists=True)
        assert profile.model_name == "fbcca"
        assert profile.model_params is not None
        assert profile.model_params["_decoder_model_name"] == "fbcca_fixed_all8"
        assert profile.channel_weights is None
        assert profile.subband_weight_mode == "chen_fixed"
        assert profile.recommended_for_realtime is True
        assert payload["profile_saved"] is True
        assert payload["task"] == DEFAULT_FBCCA_THRESHOLD_TASK
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_threshold_pretrain_task_is_available_from_ui_parser() -> None:
    args = build_parser().parse_args(["--task", "fbcca-threshold-pretrain"])
    assert args.task == "fbcca-threshold-pretrain"
    assert _parse_task("threshold_pretrain") == "fbcca-threshold-pretrain"
