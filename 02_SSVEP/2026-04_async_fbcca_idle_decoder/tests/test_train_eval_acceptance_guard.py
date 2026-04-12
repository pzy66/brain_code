from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from async_fbcca_idle_standalone import DEFAULT_PROFILE_PATH, TrialSpec
from ssvep_core.dataset import LoadedDataset, build_protocol_signature
from ssvep_core.train_eval import OfflineTrainEvalConfig, run_offline_train_eval
from ssvep_realtime_online_ui import DEFAULT_REALTIME_PROFILE_PATH


def test_realtime_default_profile_path_matches_training_default() -> None:
    assert Path(DEFAULT_REALTIME_PROFILE_PATH).resolve() == Path(DEFAULT_PROFILE_PATH).resolve()


def test_train_eval_does_not_save_profile_when_no_model_meets_acceptance(
    monkeypatch,
) -> None:
    import ssvep_core.train_eval as module

    tmp_root = PROJECT_DIR / ".tmp_test_artifacts" / f"acceptance_guard_{uuid.uuid4().hex}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    try:
        trial = TrialSpec(label="idle", expected_freq=None, trial_id=1, block_index=0)
        segment = np.zeros((1200, 8), dtype=np.float64)
        protocol_config = {"prepare_sec": 1.0, "active_sec": 4.0, "rest_sec": 1.0, "step_sec": 0.25}
        protocol_signature = build_protocol_signature(
            sampling_rate=250,
            protocol_config=protocol_config,
            freqs=(8.0, 10.0, 12.0, 15.0),
            board_eeg_channels=(0, 1, 2, 3, 4, 5, 6, 7),
        )
        protocol_config["protocol_signature"] = protocol_signature
        dataset = LoadedDataset(
            manifest_path=tmp_root / "session_manifest.json",
            npz_path=tmp_root / "raw_trials.npz",
            session_id="s1",
            subject_id="subj",
            sampling_rate=250,
            freqs=(8.0, 10.0, 12.0, 15.0),
            board_eeg_channels=(0, 1, 2, 3, 4, 5, 6, 7),
            protocol_config=protocol_config,
            trial_segments=[(trial, segment)],
            manifest={"protocol_signature": protocol_signature},
        )

        monkeypatch.setattr(module, "load_collection_dataset", lambda _path: dataset)
        monkeypatch.setattr(
            module,
            "_split_session_for_train_eval",
            lambda _dataset, seed: ([(trial, segment)], [(trial, segment)], [(trial, segment)]),
        )
        monkeypatch.setattr(module, "_subset_trial_segments_by_positions", lambda segments, _positions: list(segments))
        monkeypatch.setattr(module, "summarize_benchmark_robustness", lambda _rows, **_kwargs: {})

        save_calls: list[Path] = []
        monkeypatch.setattr(module, "save_profile", lambda _profile, out_path: save_calls.append(Path(out_path)))

        class _FakeBenchmarkRunner:
            def __init__(self, **kwargs):
                self.channel_modes = tuple(kwargs.get("channel_modes", ("all8",)))
                self.eval_seeds = [int(kwargs.get("seed", 0))]
                self.model_names = tuple(kwargs.get("model_names", ("fbcca",)))
                self.win_candidates = tuple(kwargs.get("win_candidates", (3.0,)))
                self.step_sec = float(kwargs.get("step_sec", 0.25))
                self.gate_policy = str(kwargs.get("gate_policy", "balanced"))
                self.channel_weight_mode = kwargs.get("channel_weight_mode")
                self.dynamic_stop_enabled = bool(kwargs.get("dynamic_stop_enabled", True))
                self.dynamic_stop_alpha = float(kwargs.get("dynamic_stop_alpha", 0.7))

            def _benchmark_single_model(self, **kwargs):
                model_name = str(kwargs.get("model_name", "fbcca"))
                result = {
                    "model_name": model_name,
                    "implementation_level": "paper-faithful",
                    "method_note": "test",
                    "metrics": {
                        "idle_fp_per_min": 0.0,
                        "control_recall": 0.2,
                        "switch_detect_rate": 0.0,
                        "switch_latency_s": 6.0,
                        "release_latency_s": 6.0,
                        "itr_bpm": 1.0,
                        "inference_ms": 1.0,
                    },
                    "fixed_window_metrics": {},
                    "dynamic_delta": {},
                }
                return object(), result

        monkeypatch.setattr(module, "BenchmarkRunner", _FakeBenchmarkRunner)

        config = OfflineTrainEvalConfig(
            dataset_manifest_session1=tmp_root / "s1.json",
            dataset_manifest_session2=None,
            output_profile_path=tmp_root / "default_profile.json",
            report_path=tmp_root / "offline_train_eval.json",
            model_names=("fbcca",),
            channel_modes=("all8",),
            multi_seed_count=1,
            seed_step=1,
            win_candidates=(2.0,),
            gate_policy="balanced",
            channel_weight_mode="fbcca_diag",
            dynamic_stop_enabled=True,
            dynamic_stop_alpha=0.7,
            seed=20260410,
        )

        payload = run_offline_train_eval(config, log_fn=lambda _msg: None)

        assert payload["profile_saved"] is False
        assert payload["chosen_meets_acceptance"] is False
        assert payload["chosen_profile_path"] is None
        assert payload["default_profile_saved"] is False
        assert payload["best_candidate_profile_saved"] is True
        assert payload["best_candidate_profile_path"] is not None
        expected_paths = [Path(payload["best_candidate_profile_path"])]
        if payload.get("best_fbcca_weighted_profile_saved"):
            expected_paths.append(Path(payload["best_fbcca_weighted_profile_path"]))
        assert save_calls == expected_paths
        assert config.output_profile_path not in save_calls
        assert "ranking_boards" in payload
        assert isinstance(payload["ranking_boards"].get("end_to_end", []), list)
        assert isinstance(payload["ranking_boards"].get("classifier_only", []), list)
        assert "stats_baseline_model" in payload
        assert config.report_path.exists()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
