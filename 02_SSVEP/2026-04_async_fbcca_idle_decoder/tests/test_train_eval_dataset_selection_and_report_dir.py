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
from ssvep_core.dataset import LoadedDataset, build_protocol_signature, discover_collection_manifests
from ssvep_core.train_eval import OfflineTrainEvalConfig, run_offline_train_eval


def test_discover_collection_manifests_summarizes_trials() -> None:
    tmp_root = PROJECT_DIR / ".tmp_test_artifacts" / f"discover_{uuid.uuid4().hex}"
    session_dir = tmp_root / "s01"
    session_dir.mkdir(parents=True, exist_ok=True)
    try:
        np.savez_compressed(session_dir / "raw_trials.npz", trial_collection_1=np.zeros((100, 8), dtype=np.float32))
        manifest = {
            "data_schema_version": "2.0",
            "session_id": "s01",
            "subject_id": "subjA",
            "generated_at": "2026-04-11T12:00:00",
            "sampling_rate": 250,
            "freqs": [8.0, 10.0, 12.0, 15.0],
            "board_eeg_channels": [0, 1, 2, 3, 4, 5, 6, 7],
            "protocol_config": {"preset_name": "stable_12m", "round_index": 1, "rounds_planned": 3},
            "protocol_signature": "sha1:testsig",
            "quality_summary": {"valid_trial_count": 1, "short_segment_excluded": 0, "retry_total": 0},
            "trials": [
                {
                    "stage": "collection",
                    "label": "8Hz",
                    "expected_freq": 8.0,
                    "trial_id": 1,
                    "block_index": 0,
                    "order_index": 0,
                    "used_samples": 100,
                    "target_samples": 100,
                    "shortfall_ratio": 0.0,
                    "retry_count": 0,
                    "channels": 8,
                    "npz_key": "trial_collection_1",
                }
            ],
            "files": {"raw_trials_npz": str(session_dir / "raw_trials.npz")},
        }
        (session_dir / "session_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        rows = discover_collection_manifests(tmp_root)
        assert len(rows) == 1
        row = rows[0]
        assert row["session_id"] == "s01"
        assert row["subject_id"] == "subjA"
        assert int(row["trial_count"]) == 1
        assert abs(float(row["shortfall_ratio_mean"]) - 0.0) < 1e-9
        assert str(row.get("protocol_signature", "")) == "sha1:testsig"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_train_eval_organize_report_dir_with_selected_manifests(monkeypatch) -> None:
    import ssvep_core.train_eval as module

    tmp_root = PROJECT_DIR / ".tmp_test_artifacts" / f"report_dir_{uuid.uuid4().hex}"
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
        monkeypatch.setattr(module, "save_profile", lambda _profile, _path: None)

        class _FakeBenchmarkRunner:
            def __init__(self, **kwargs):
                self.channel_modes = tuple(kwargs.get("channel_modes", ("all8",)))
                self.eval_seeds = [int(kwargs.get("seed", 0))]
                self.model_names = tuple(kwargs.get("model_names", ("fbcca",)))
                self.win_candidates = tuple(kwargs.get("win_candidates", (3.0,)))
                self.step_sec = float(kwargs.get("step_sec", 0.25))
                self.gate_policy = str(kwargs.get("gate_policy", "balanced"))
                self.channel_weight_mode = kwargs.get("channel_weight_mode")

            def _benchmark_single_model(self, **kwargs):
                model_name = str(kwargs.get("model_name", "fbcca"))
                return object(), {
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

        monkeypatch.setattr(module, "BenchmarkRunner", _FakeBenchmarkRunner)

        config = OfflineTrainEvalConfig(
            dataset_manifest_session1=tmp_root / "unused_s1.json",
            dataset_manifest_session2=None,
            dataset_manifests=(tmp_root / "a.json", tmp_root / "b.json"),
            output_profile_path=tmp_root / "default_profile.json",
            report_path=tmp_root / "legacy_report.json",
            report_root_dir=tmp_root / "reports",
            organize_report_dir=True,
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
        report_path = Path(payload["report_path"])
        report_dir = Path(payload["report_dir"])
        assert report_path.exists()
        assert report_dir.exists()
        assert Path(payload["selection_snapshot_path"]).exists()
        assert Path(payload["run_config_path"]).exists()
        assert Path(payload["run_log_path"]).exists()
        assert int(payload["selected_dataset_count_session1"]) == 2
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
