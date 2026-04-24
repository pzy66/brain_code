from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from async_fbcca_idle_standalone import TrialSpec
from ssvep_core.dataset import LoadedDataset, build_protocol_signature
from ssvep_core.train_eval import (
    OfflineTrainEvalConfig,
    _apply_trial_quality_filter,
    _load_session1_dataset,
    run_offline_train_eval,
)


def _build_dataset(
    *,
    session_id: str,
    subject_id: str = "subjA",
    active_sec: float = 5.0,
) -> LoadedDataset:
    trials = [
        TrialSpec(label="8Hz", expected_freq=8.0, trial_id=1, block_index=0),
        TrialSpec(label="10Hz", expected_freq=10.0, trial_id=2, block_index=0),
        TrialSpec(label="idle", expected_freq=None, trial_id=3, block_index=0),
    ]
    segments = [
        np.zeros((1200, 8), dtype=np.float64),
        np.zeros((1200, 8), dtype=np.float64),
        np.zeros((1200, 8), dtype=np.float64),
    ]
    manifest_trials = [
        {
            "order_index": 0,
            "stage": "collection",
            "label": "8Hz",
            "expected_freq": 8.0,
            "used_samples": 1200,
            "target_samples": 1200,
            "retry_count": 0,
            "npz_key": "k0",
        },
        {
            "order_index": 1,
            "stage": "collection",
            "label": "10Hz",
            "expected_freq": 10.0,
            "used_samples": 960,
            "target_samples": 1200,
            "retry_count": 0,
            "npz_key": "k1",
        },
        {
            "order_index": 2,
            "stage": "collection",
            "label": "idle",
            "expected_freq": None,
            "used_samples": 1200,
            "target_samples": 1200,
            "retry_count": 4,
            "npz_key": "k2",
        },
    ]
    protocol_config = {"prepare_sec": 1.0, "active_sec": active_sec, "rest_sec": 4.0, "step_sec": 0.25}
    protocol_signature = build_protocol_signature(
        sampling_rate=250,
        protocol_config=protocol_config,
        freqs=(8.0, 10.0, 12.0, 15.0),
        board_eeg_channels=(0, 1, 2, 3, 4, 5, 6, 7),
    )
    protocol_config["protocol_signature"] = protocol_signature
    return LoadedDataset(
        manifest_path=PROJECT_DIR / f"{session_id}_manifest.json",
        npz_path=PROJECT_DIR / f"{session_id}.npz",
        session_id=session_id,
        subject_id=subject_id,
        sampling_rate=250,
        freqs=(8.0, 10.0, 12.0, 15.0),
        board_eeg_channels=(0, 1, 2, 3, 4, 5, 6, 7),
        protocol_config=protocol_config,
        trial_segments=list(zip(trials, segments)),
        manifest={"trials": manifest_trials, "protocol_signature": protocol_signature},
    )


def test_apply_trial_quality_filter_drops_shortfall_and_retry() -> None:
    dataset = _build_dataset(session_id="s1")
    filtered, summary = _apply_trial_quality_filter(
        dataset,
        min_sample_ratio=0.9,
        max_retry_count=2,
    )
    assert len(filtered.trial_segments) == 1
    assert int(summary["total_trials"]) == 3
    assert int(summary["kept_trials"]) == 1
    assert int(summary["dropped_shortfall"]) == 1
    assert int(summary["dropped_retry"]) == 1


def test_load_session1_dataset_strict_protocol_consistency(monkeypatch) -> None:
    import ssvep_core.train_eval as module

    ds_a = _build_dataset(session_id="sA", active_sec=5.0)
    ds_b = _build_dataset(session_id="sB", active_sec=4.0)

    def _fake_loader(path: Path) -> LoadedDataset:
        return ds_a if str(path).endswith("a.json") else ds_b

    monkeypatch.setattr(module, "load_collection_dataset", _fake_loader)

    config = OfflineTrainEvalConfig(
        dataset_manifest_session1=PROJECT_DIR / "a.json",
        dataset_manifest_session2=None,
        output_profile_path=PROJECT_DIR / "profiles" / "x.json",
        report_path=PROJECT_DIR / "profiles" / "y.json",
        dataset_manifests=(PROJECT_DIR / "a.json", PROJECT_DIR / "b.json"),
        strict_protocol_consistency=True,
        strict_subject_consistency=True,
        data_policy="legacy-compatible",
        quality_min_sample_ratio=0.0,
        quality_max_retry_count=10,
    )
    with pytest.raises(RuntimeError, match="inconsistent protocol_config"):
        _load_session1_dataset(config)


def test_run_offline_train_eval_rejects_session2_subject_mismatch(monkeypatch) -> None:
    import ssvep_core.train_eval as module

    ds_s1 = _build_dataset(session_id="s1", subject_id="subjectA", active_sec=5.0)
    ds_s2 = _build_dataset(session_id="s2", subject_id="subjectB", active_sec=5.0)

    monkeypatch.setattr(
        module,
        "_load_session1_dataset",
        lambda _config: (
            ds_s1,
            (PROJECT_DIR / "s1.json",),
            {"collection"},
            [{"session_id": "s1", "total_trials": 3, "kept_trials": 3, "dropped_trials": 0}],
            {"strict_subject_consistency": True},
        ),
    )
    monkeypatch.setattr(module, "load_collection_dataset", lambda _path: ds_s2)

    config = OfflineTrainEvalConfig(
        dataset_manifest_session1=PROJECT_DIR / "s1.json",
        dataset_manifest_session2=PROJECT_DIR / "s2.json",
        output_profile_path=PROJECT_DIR / "profiles" / "x.json",
        report_path=PROJECT_DIR / "profiles" / "y.json",
        model_names=("fbcca",),
        channel_modes=("all8",),
        multi_seed_count=1,
        seed_step=1,
        win_candidates=(2.0,),
        strict_subject_consistency=True,
        strict_protocol_consistency=True,
        data_policy="legacy-compatible",
        quality_min_sample_ratio=0.9,
        quality_max_retry_count=3,
    )
    with pytest.raises(RuntimeError, match="session2 subject_id differs from session1"):
        run_offline_train_eval(config, log_fn=lambda _msg: None)


def test_load_session1_dataset_new_only_rejects_missing_protocol_signature(monkeypatch) -> None:
    import ssvep_core.train_eval as module

    ds = _build_dataset(session_id="s_missing_sig", subject_id="subjectA", active_sec=5.0)
    ds.protocol_config.pop("protocol_signature", None)
    ds.manifest.pop("protocol_signature", None)

    monkeypatch.setattr(module, "load_collection_dataset", lambda _path: ds)

    config = OfflineTrainEvalConfig(
        dataset_manifest_session1=PROJECT_DIR / "s_missing_sig.json",
        dataset_manifest_session2=None,
        output_profile_path=PROJECT_DIR / "profiles" / "x.json",
        report_path=PROJECT_DIR / "profiles" / "y.json",
        data_policy="new-only",
    )
    with pytest.raises(RuntimeError, match="requires protocol_signature"):
        _load_session1_dataset(config)
