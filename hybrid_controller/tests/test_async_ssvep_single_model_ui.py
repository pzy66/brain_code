from __future__ import annotations

import numpy as np

from hybrid_controller.ssvep import single_model as module


def make_segment(*, fs: int, sec: float, channels: int, freq: float | None, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    samples = int(round(float(fs) * float(sec)))
    t = np.arange(samples, dtype=float) / float(fs)
    data = np.zeros((samples, channels), dtype=float)
    for channel in range(channels):
        noise = 0.20 * rng.standard_normal(samples)
        if freq is None:
            signal = noise
        else:
            base = np.sin(2.0 * np.pi * float(freq) * t + 0.11 * channel)
            harmonic = 0.35 * np.cos(2.0 * np.pi * 2.0 * float(freq) * t + 0.07 * channel)
            signal = (1.0 + 0.05 * channel) * base + harmonic + noise
        data[:, channel] = signal
    return data


def build_trial_segments(module):
    freqs = (8.0, 10.0, 12.0, 15.0)
    fs = 250
    sec = 4.0
    channels = 4
    segments = []
    trial_id = 0
    for block in range(3):
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
                    make_segment(fs=fs, sec=sec, channels=channels, freq=float(freq), seed=trial_id + 13),
                )
            )
        idle = module.TrialSpec(label="idle", expected_freq=None, trial_id=trial_id, block_index=block)
        trial_id += 1
        segments.append((idle, make_segment(fs=fs, sec=sec, channels=channels, freq=None, seed=trial_id + 29)))
    return segments


def test_fit_single_model_profile_keeps_selected_model_and_state() -> None:
    segments = build_trial_segments(module)
    profile, metadata = module.fit_single_model_profile_from_segments(
        model_name="trca",
        trial_segments=segments,
        available_board_channels=(0, 1, 2, 3),
        sampling_rate=250,
        freqs=(8.0, 10.0, 12.0, 15.0),
        active_sec=4.0,
        win_sec=3.0,
        step_sec=0.25,
    )
    assert profile.model_name == "trca"
    assert profile.model_params is not None
    assert "state" in profile.model_params
    assert profile.eeg_channels is not None
    assert metadata["model_name"] == "trca"
    assert "quality_summary" in metadata


def test_default_single_profile_path_name() -> None:
    path = module.default_single_model_profile_path()
    assert str(path).endswith("single_model_profile.json")
