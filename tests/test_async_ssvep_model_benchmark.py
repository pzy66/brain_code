from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "02_SSVEP"
    / "2026-04_async_fbcca_idle_decoder"
    / "async_fbcca_idle_standalone.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("async_fbcca_idle_standalone", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_segment(
    *,
    fs: int,
    sec: float,
    channels: int,
    freq: float | None,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    samples = int(round(float(fs) * float(sec)))
    t = np.arange(samples, dtype=float) / float(fs)
    data = np.zeros((samples, channels), dtype=float)
    for channel in range(channels):
        noise = 0.20 * rng.standard_normal(samples)
        if freq is None:
            signal = noise
        else:
            base = np.sin(2.0 * np.pi * float(freq) * t + 0.07 * channel)
            harmonic = 0.4 * np.cos(2.0 * np.pi * 2.0 * float(freq) * t + 0.13 * channel)
            signal = (1.0 + 0.08 * channel) * base + harmonic + noise
        data[:, channel] = signal
    return data


def build_trial_segments(module, *, repeats: int = 3):
    freqs = (8.0, 10.0, 12.0, 15.0)
    fs = 250
    sec = 4.0
    channels = 4
    segments = []
    trial_id = 0
    for block in range(repeats):
        for freq in freqs:
            trial = module.TrialSpec(
                label=f"{freq:g}Hz",
                expected_freq=float(freq),
                trial_id=trial_id,
                block_index=block,
            )
            trial_id += 1
            segments.append(
                (
                    trial,
                    make_segment(fs=fs, sec=sec, channels=channels, freq=float(freq), seed=trial_id + 100),
                )
            )
        idle_trial = module.TrialSpec(
            label="idle",
            expected_freq=None,
            trial_id=trial_id,
            block_index=block,
        )
        trial_id += 1
        segments.append((idle_trial, make_segment(fs=fs, sec=sec, channels=channels, freq=None, seed=trial_id + 200)))
    return segments


def test_model_registry_decoders_emit_four_scores() -> None:
    module = load_module()
    segments = build_trial_segments(module, repeats=3)
    train = [item for item in segments if item[0].expected_freq is not None][:8]
    window = train[0][1][-750:, :]
    model_names = ("cca", "itcca", "ecca", "msetcca", "fbcca", "trca", "trca_r", "sscor", "tdca", "oacca")

    for model_name in model_names:
        decoder = module.create_decoder(
            model_name,
            sampling_rate=250,
            freqs=(8.0, 10.0, 12.0, 15.0),
            win_sec=3.0,
            step_sec=0.25,
            model_params={"Nh": 3},
        )
        if decoder.requires_fit:
            decoder.fit(train)
        result = decoder.analyze_window(window)
        assert len(result["scores"]) == 4
        assert result["pred_freq"] in (8.0, 10.0, 12.0, 15.0)
        assert float(result["top1_score"]) >= float(result["top2_score"])


def test_split_trial_segments_for_benchmark_produces_non_empty_gate_and_holdout() -> None:
    module = load_module()
    segments = build_trial_segments(module, repeats=5)
    train, gate, holdout = module.split_trial_segments_for_benchmark(segments, seed=20260408)

    assert len(train) > 0
    assert len(gate) > 0
    assert len(holdout) > 0
    assert len(train) + len(gate) + len(holdout) == len(segments)


def test_evaluate_decoder_on_trials_returns_expected_metrics() -> None:
    module = load_module()
    segments = build_trial_segments(module, repeats=4)
    train, gate_segments, holdout = module.split_trial_segments_for_benchmark(segments, seed=42)
    decoder = module.create_decoder(
        "fbcca",
        sampling_rate=250,
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=3.0,
        step_sec=0.25,
        model_params={"Nh": 3},
    )
    gate_rows = module.build_feature_rows_with_decoder(decoder, gate_segments)
    profile = module.fit_threshold_profile(
        gate_rows,
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=3.0,
        step_sec=0.25,
        min_enter_windows=2,
        min_exit_windows=2,
    )
    metrics = module.evaluate_decoder_on_trials(decoder, profile, holdout)

    assert "idle_fp_per_min" in metrics
    assert "control_recall" in metrics
    assert "switch_latency_s" in metrics
    assert "itr_bpm" in metrics
    assert "inference_ms" in metrics


def test_profile_roundtrip_keeps_model_fields() -> None:
    module = load_module()
    profile = module.ThresholdProfile(
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=3.0,
        step_sec=0.25,
        enter_score_th=0.02,
        enter_ratio_th=1.15,
        enter_margin_th=0.003,
        exit_score_th=0.017,
        exit_ratio_th=1.09,
        min_enter_windows=2,
        min_exit_windows=2,
        model_name="trca_r",
        model_params={"Nh": 3, "state": {"dummy": True}},
        calibration_split_seed=123,
        benchmark_metrics={"idle_fp_per_min": 0.2, "control_recall": 0.9},
    )
    tmp_path = MODULE_PATH.parents[2] / ".tmp_async_ssvep_profile_roundtrip.json"
    try:
        module.save_profile(profile, tmp_path)
        loaded = module.load_profile(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    assert loaded.model_name == "trca_r"
    assert loaded.model_params is not None
    assert loaded.model_params.get("Nh") == 3
    assert loaded.calibration_split_seed == 123
    assert loaded.benchmark_metrics is not None
