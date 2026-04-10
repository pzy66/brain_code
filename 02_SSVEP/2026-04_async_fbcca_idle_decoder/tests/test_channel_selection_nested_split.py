from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from async_fbcca_idle_standalone import TrialSpec, select_auto_eeg_channels_for_model


def _synth_trial(expected_freq: float | None, *, fs: int, samples: int, channels: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(samples, dtype=float) / float(fs)
    data = 0.12 * rng.standard_normal((samples, channels))
    if expected_freq is not None:
        for ch in range(channels):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            data[:, ch] += 0.6 * np.sin(2.0 * np.pi * float(expected_freq) * t + phase)
    return np.asarray(data, dtype=np.float64)


def test_channel_selection_reports_nested_split_fields() -> None:
    rng = np.random.default_rng(42)
    random.seed(42)
    fs = 250
    samples = 1000
    board_channels = tuple(range(8))
    freqs = (8.0, 10.0, 12.0, 15.0)

    segments: list[tuple[TrialSpec, np.ndarray]] = []
    trial_id = 0
    for _ in range(5):
        for freq in freqs:
            segments.append(
                (
                    TrialSpec(label=f"{freq:g}Hz", expected_freq=freq, trial_id=trial_id, block_index=0),
                    _synth_trial(freq, fs=fs, samples=samples, channels=len(board_channels), rng=rng),
                )
            )
            trial_id += 1
        segments.append(
            (
                TrialSpec(label="idle", expected_freq=None, trial_id=trial_id, block_index=0),
                _synth_trial(None, fs=fs, samples=samples, channels=len(board_channels), rng=rng),
            )
        )
        trial_id += 1

    selected, channel_scores = select_auto_eeg_channels_for_model(
        segments,
        model_name="fbcca",
        available_board_channels=board_channels,
        sampling_rate=fs,
        freqs=freqs,
        win_sec=3.0,
        step_sec=0.25,
        model_params={"Nh": 3},
        seed=42,
        validation_fraction=0.2,
        target_count=4,
    )

    assert len(selected) >= 1
    assert channel_scores
    for item in channel_scores:
        split_counts = dict(item["split_counts"])
        assert "fit_segments" in split_counts
        assert "threshold_segments" in split_counts
        assert "score_segments" in split_counts
        assert int(split_counts["fit_segments"]) >= 1
        assert int(split_counts["threshold_segments"]) >= 1
        assert int(split_counts["score_segments"]) >= 1
