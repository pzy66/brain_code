from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from async_fbcca_idle_standalone import build_calibration_trials
from ssvep_single_model_ui import fit_single_model_profile_from_segments


def _synth_trial(
    expected_freq: float | None,
    *,
    fs: int,
    samples: int,
    channels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    t = np.arange(samples, dtype=float) / float(fs)
    data = 0.12 * rng.standard_normal((samples, channels))
    if expected_freq is not None:
        for ch in range(channels):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            data[:, ch] += 0.6 * np.sin(2.0 * np.pi * float(expected_freq) * t + phase)
            data[:, ch] += 0.2 * np.sin(2.0 * np.pi * (2.0 * float(expected_freq)) * t + 0.5 * phase)
    return np.asarray(data, dtype=np.float64)


def test_single_model_pretrain_reports_split_metadata() -> None:
    rng = np.random.default_rng(123)
    fs = 250
    active_sec = 4.0
    samples = int(round(active_sec * fs))
    freqs = (8.0, 10.0, 12.0, 15.0)
    board_channels = tuple(range(8))

    trials = build_calibration_trials(
        freqs,
        target_repeats=5,
        idle_repeats=10,
        shuffle=True,
        seed=20260409,
    )
    segments = [
        (
            trial,
            _synth_trial(
                trial.expected_freq,
                fs=fs,
                samples=samples,
                channels=len(board_channels),
                rng=rng,
            ),
        )
        for trial in trials
    ]

    profile, metadata = fit_single_model_profile_from_segments(
        model_name="fbcca",
        trial_segments=segments,
        available_board_channels=board_channels,
        sampling_rate=fs,
        freqs=freqs,
        active_sec=active_sec,
        seed=20260409,
    )

    split = dict(metadata.get("calibration_split", {}))
    assert int(split.get("fit_segments", 0)) >= 1
    assert int(split.get("gate_segments", 0)) >= 1
    assert "holdout_segments" in split
    assert metadata.get("quality_summary_mode") in {"holdout", "gate_fallback"}
    assert isinstance(metadata.get("quality_summary_gate"), dict)
    assert isinstance(metadata.get("quality_summary"), dict)
    assert metadata.get("channel_selection_scope") == "outer_fit_only"
    outer_split = dict(metadata.get("outer_split", {}))
    assert int(outer_split.get("fit_segments", 0)) >= 1
    assert int(outer_split.get("gate_segments", 0)) >= 1
    assert "holdout_segments" in outer_split
    assert profile.metadata is not None
    assert "calibration_split" in profile.metadata
    assert str(profile.gate_policy) == "balanced"
    assert isinstance(profile.dynamic_stop, dict)
    assert bool(profile.dynamic_stop.get("enabled")) is True
    assert str(profile.channel_weight_mode) == "fbcca_diag"
    assert profile.channel_weights is not None
    assert profile.eeg_channels is not None
    assert len(tuple(profile.channel_weights)) == len(tuple(profile.eeg_channels))
