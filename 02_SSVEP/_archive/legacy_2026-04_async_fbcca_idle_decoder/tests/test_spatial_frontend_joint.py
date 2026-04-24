from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from async_fbcca_idle_standalone import (  # noqa: E402
    TrialSpec,
    create_decoder,
    normalize_channel_weights,
    optimize_fbcca_diag_channel_weights,
    parse_spatial_rank_candidates,
)


def _synth_trial(
    expected_freq: float | None,
    *,
    fs: int,
    samples: int,
    channels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    t = np.arange(samples, dtype=float) / float(fs)
    data = 0.10 * rng.standard_normal((samples, channels))
    if expected_freq is not None:
        for ch in range(channels):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            data[:, ch] += 0.65 * np.sin(2.0 * np.pi * float(expected_freq) * t + phase)
            data[:, ch] += 0.25 * np.sin(2.0 * np.pi * (2.0 * float(expected_freq)) * t + 0.5 * phase)
    return np.asarray(data, dtype=np.float64)


def test_parse_spatial_rank_candidates_normalizes_values() -> None:
    assert parse_spatial_rank_candidates("1,2,3") == (1, 2, 3)
    assert parse_spatial_rank_candidates("3,2,2,1") == (1, 2, 3)
    assert parse_spatial_rank_candidates("0,-1,2") == (1, 2)


def test_fbcca_frontend_applies_channel_and_spatial_projection() -> None:
    params = {
        "Nh": 3,
        "channel_weight_mode": "fbcca_diag",
        "channel_weights": [2.0, 1.0, 1.0, 1.0],
        "spatial_filter_mode": "trca_shared",
        "spatial_filter_rank": 2,
        "spatial_filter_state": {
            "source_model": "trca",
            "projection": [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        },
    }
    decoder = create_decoder(
        "fbcca",
        sampling_rate=250,
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=1.0,
        step_sec=0.25,
        model_params=params,
    )
    window = np.ones((250, 4), dtype=float)
    transformed = decoder._apply_frontend(window)  # type: ignore[attr-defined]
    expected_weights = normalize_channel_weights(np.asarray([2.0, 1.0, 1.0, 1.0], dtype=float), channels=4)
    assert transformed.shape == (250, 2)
    assert np.allclose(transformed[:, 0], float(expected_weights[0]))
    assert np.allclose(transformed[:, 1], float(expected_weights[1]))


def test_joint_weight_optimizer_outputs_spatial_state() -> None:
    rng = np.random.default_rng(42)
    fs = 250
    samples = int(round(3.0 * fs))
    freqs = (8.0, 10.0, 12.0, 15.0)
    channels = 8
    train_segments: list[tuple[TrialSpec, np.ndarray]] = []
    gate_segments: list[tuple[TrialSpec, np.ndarray]] = []
    trial_id = 0
    for freq in freqs:
        for _ in range(2):
            train_segments.append(
                (
                    TrialSpec(label=f"train_{freq:g}", expected_freq=float(freq), trial_id=trial_id, block_index=0),
                    _synth_trial(float(freq), fs=fs, samples=samples, channels=channels, rng=rng),
                )
            )
            trial_id += 1
        for _ in range(2):
            gate_segments.append(
                (
                    TrialSpec(label=f"gate_{freq:g}", expected_freq=float(freq), trial_id=trial_id, block_index=1),
                    _synth_trial(float(freq), fs=fs, samples=samples, channels=channels, rng=rng),
                )
            )
            trial_id += 1
    for _ in range(4):
        gate_segments.append(
            (
                TrialSpec(label="idle", expected_freq=None, trial_id=trial_id, block_index=2),
                _synth_trial(None, fs=fs, samples=samples, channels=channels, rng=rng),
            )
        )
        trial_id += 1

    _weights, metadata = optimize_fbcca_diag_channel_weights(
        train_segments=train_segments,
        gate_segments=gate_segments,
        sampling_rate=fs,
        freqs=freqs,
        win_sec=1.5,
        step_sec=0.25,
        min_enter_windows=1,
        min_exit_windows=2,
        gate_policy="balanced",
        dynamic_stop_enabled=True,
        dynamic_stop_alpha=0.7,
        spatial_filter_mode="trca_shared",
        spatial_rank_candidates=(1, 2),
        joint_weight_iters=1,
        spatial_source_model="trca",
    )
    assert metadata.get("spatial_filter_mode") == "trca_shared"
    assert int(metadata.get("selected_spatial_rank", 0)) in {1, 2}
    model_params = dict(metadata.get("optimized_model_params", {}))
    assert model_params.get("spatial_filter_mode") == "trca_shared"
    assert "spatial_filter_state" in model_params
