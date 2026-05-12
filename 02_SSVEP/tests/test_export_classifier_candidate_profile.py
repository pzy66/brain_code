from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tools.export_classifier_candidate_profile import export_classifier_candidate_profile
from ssvep_core.async_fbcca_idle_standalone import load_decoder_from_profile, load_profile


def test_export_classifier_candidate_profile_roundtrip(tmp_path: Path, monkeypatch) -> None:
    freqs = (8.0, 10.5, 12.0, 15.0)
    labels = ("idle", "8", "10.5", "12", "15")
    feature_count = int(len(freqs) + 6)
    weights = np.zeros((feature_count + 1, len(labels)), dtype=np.float64)
    weights[0, 0] = 0.2
    weights[1 + 0, 1] = 8.0
    state = {
        "freqs": list(freqs),
        "labels": list(labels),
        "feature_mean": [0.0] * feature_count,
        "feature_std": [1.0] * feature_count,
        "weights": weights.tolist(),
        "l2": 0.3,
        "command_confidence_th": 0.1,
        "smoothing_windows": 1,
        "gate_policy": "confidence_threshold",
        "fit_summary": {"min_enter_windows": 1, "max_gap_windows": 0},
    }
    artifact = {
        "artifact_schema_version": "external_fbcca_classifier_candidate_v1",
        "runtime_loadable": True,
        "model_name": "fbcca_score_ridge_5class",
        "feature_contract": {
            "score_bank_mode": "command_only",
            "feature_names": [f"feature_{index}" for index in range(feature_count)],
        },
        "windowing": {"win_sec": 2.0, "step_sec": 0.25, "min_enter_windows": 1},
        "training_provenance": {
            "seed": 123,
            "score_source_name": "fbcca",
            "decoder_name": "fbcca_fixed_all8",
            "decoder_model_params": {"Nh": 5, "subband_weight_mode": "chen_fixed"},
            "required_channel_names": ["Oz", "O1", "O2", "PO3", "POz", "PO7", "PO8", "PO4"],
            "command_freqs": list(freqs),
        },
        "state": state,
        "runtime_profile_model_params": {
            "state": state,
            "score_source_name": "fbcca",
            "score_bank_mode": "command_only",
            "feature_names": [f"feature_{index}" for index in range(feature_count)],
            "decoder_name": "fbcca_fixed_all8",
            "decoder_model_params": {"Nh": 5, "subband_weight_mode": "chen_fixed"},
            "max_gap_windows": 0,
        },
        "summary_metrics": {"control_recall": 1.0, "idle_fp_per_min": 0.0},
    }
    candidate_path = tmp_path / "candidate.json"
    output_path = tmp_path / "profile.json"
    candidate_path.write_text(json.dumps(artifact), encoding="utf-8")

    profile = export_classifier_candidate_profile(candidate_path, output_path)
    loaded = load_profile(output_path, require_exists=True)

    assert profile.model_name == "fbcca_score_ridge_5class"
    assert loaded.model_name == "fbcca_score_ridge_5class"
    assert loaded.freqs == freqs
    assert loaded.model_params is not None
    assert loaded.model_params["state"]["l2"] == 0.3
    assert loaded.model_params["score_bank_mode"] == "command_only"
    assert loaded.eeg_channels == tuple(range(8))
    assert loaded.benchmark_metrics == {"control_recall": 1.0, "idle_fp_per_min": 0.0}

    def fake_score_window(_self, _window: np.ndarray) -> np.ndarray:
        return np.asarray([5.0, 0.2, 0.2, 0.2], dtype=np.float64)

    from ssvep_core import async_fbcca_idle_standalone as async_module

    monkeypatch.setattr(async_module.FBCCADecoder, "score_window", fake_score_window)
    decoder = load_decoder_from_profile(loaded, sampling_rate=250, compute_backend="cpu", gpu_warmup=False)
    result = decoder.analyze_window(np.zeros((decoder.win_samples, 8), dtype=np.float64))

    assert result["classifier_pred_label"] == "8"
    assert result["classifier_command_confidence"] > 0.1
