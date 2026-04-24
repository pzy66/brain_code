from __future__ import annotations

import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import async_fbcca_idle_standalone as async_module
from async_fbcca_idle_standalone import (
    AsyncDecisionGate,
    FBCCAEngine,
    ThresholdProfile,
    atomic_write_text,
    load_decoder_from_profile,
    load_profile,
    save_profile,
)
from ssvep_core.compute_backend import NumpyBackend


def _synthetic_window(samples: int, channels: int, freqs: tuple[float, ...], fs: int) -> np.ndarray:
    t = np.arange(int(samples), dtype=np.float64) / float(fs)
    window = np.zeros((int(samples), int(channels)), dtype=np.float64)
    for channel_index in range(int(channels)):
        freq = float(freqs[channel_index % len(freqs)])
        window[:, channel_index] = (
            np.sin(2.0 * np.pi * freq * t + channel_index * 0.11)
            + 0.1 * np.cos(2.0 * np.pi * (freq * 2.0) * t)
        )
    return window


def test_numpy_backend_microbenchmark_contains_expected_fields() -> None:
    backend = NumpyBackend(precision="float32")
    summary = backend.microbenchmark_transfer(sample_shape=(64, 8), repeats=2)
    assert summary["backend_name"] == "cpu"
    assert summary["sample_shape"] == [64, 8]
    for key in ("host_to_device_ms", "device_to_host_ms", "synchronize_ms", "warmup_overhead_ms"):
        assert key in summary
        assert float(summary[key]) >= 0.0


def test_fbcca_engine_backend_benchmark_contains_kernel_and_transfer_sections() -> None:
    engine = FBCCAEngine(
        sampling_rate=250,
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=1.5,
        step_sec=0.25,
        compute_backend="cpu",
        gpu_warmup=False,
    )
    sample_window = _synthetic_window(engine.win_samples, 8, engine.freqs, 250)
    summary = engine.benchmark_backend_path(sample_window=sample_window, repeats=1)
    assert summary["requested_backend"] == "cpu"
    assert summary["used_backend"] == "cpu"
    assert "transfer_benchmark" in summary
    assert "kernel_benchmark" in summary
    assert float(summary["kernel_benchmark"]["total_ms"]) >= 0.0


def _manual_tmp_dir(name: str) -> Path:
    root = PROJECT_DIR / ".tmp_test_artifacts" / f"{name}_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_atomic_write_text_keeps_original_file_on_replace_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_root = _manual_tmp_dir("atomic_write")
    try:
        target = tmp_root / "artifact.txt"
        target.write_text("old", encoding="utf-8")

        def _fail_replace(_src: str, _dst: str) -> None:
            raise OSError("replace failed")

        monkeypatch.setattr(async_module.os, "replace", _fail_replace)
        with pytest.raises(OSError, match="replace failed"):
            atomic_write_text(target, "new", encoding="utf-8")

        assert target.read_text(encoding="utf-8") == "old"
        assert list(tmp_root.glob("*.tmp")) == []
    finally:
        import shutil

        shutil.rmtree(tmp_root, ignore_errors=True)


def test_saved_profile_roundtrip_matches_realtime_decoder_outputs() -> None:
    tmp_root = _manual_tmp_dir("profile_roundtrip")
    try:
        profile = ThresholdProfile(
            freqs=(8.0, 10.0, 12.0, 15.0),
            win_sec=1.5,
            step_sec=0.25,
            enter_score_th=0.1,
            enter_ratio_th=1.01,
            enter_margin_th=0.01,
            exit_score_th=0.08,
            exit_ratio_th=1.0,
            min_enter_windows=1,
            min_exit_windows=1,
            model_name="fbcca",
            model_params={"Nh": 3},
            eeg_channels=(0, 1, 2, 3, 4, 5, 6, 7),
            channel_weight_mode="fbcca_diag",
            channel_weights=(1.2, 1.0, 0.9, 1.1, 1.0, 0.95, 1.05, 0.8),
            subband_weight_mode="chen_ab_subject",
            subband_weights=(0.3, 0.25, 0.2, 0.15, 0.1),
            runtime_backend_preference="cpu",
            runtime_precision_preference="float32",
        )
        profile_path = tmp_root / "weighted_profile.json"
        save_profile(profile, profile_path)
        loaded = load_profile(profile_path, require_exists=True)

        decoder_a = load_decoder_from_profile(profile, sampling_rate=250, compute_backend="cpu", gpu_precision="float32")
        decoder_b = load_decoder_from_profile(loaded, sampling_rate=250, compute_backend="cpu", gpu_precision="float32")
        window = _synthetic_window(decoder_a.win_samples, 8, profile.freqs, 250)

        analysis_a = decoder_a.analyze_window(window)
        analysis_b = decoder_b.analyze_window(window)
        gate_a = AsyncDecisionGate.from_profile(profile)
        gate_b = AsyncDecisionGate.from_profile(loaded)
        decision_a = gate_a.update(analysis_a)
        decision_b = gate_b.update(analysis_b)

        assert analysis_a["pred_freq"] == analysis_b["pred_freq"]
        assert abs(float(analysis_a["top1_score"]) - float(analysis_b["top1_score"])) <= 1e-9
        assert abs(float(analysis_a["ratio"]) - float(analysis_b["ratio"])) <= 1e-9
        assert decision_a["selected_freq"] == decision_b["selected_freq"]
    finally:
        import shutil

        shutil.rmtree(tmp_root, ignore_errors=True)
