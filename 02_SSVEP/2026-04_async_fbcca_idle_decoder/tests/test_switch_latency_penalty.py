from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from async_fbcca_idle_standalone import BaseSSVEPDecoder, ThresholdProfile, TrialSpec, evaluate_decoder_on_trials


class ConstantDecoder(BaseSSVEPDecoder):
    def __init__(self) -> None:
        super().__init__(
            model_name="constant",
            sampling_rate=20,
            freqs=(8.0, 10.0, 12.0, 15.0),
            win_sec=1.0,
            step_sec=0.5,
            model_params={},
        )

    def score_window(self, X_window: np.ndarray) -> np.ndarray:
        _ = X_window
        return np.asarray([1.0, 0.1, 0.05, 0.02], dtype=float)


def test_switch_latency_uses_penalty_when_switch_not_detected() -> None:
    decoder = ConstantDecoder()
    profile = ThresholdProfile(
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=1.0,
        step_sec=0.5,
        enter_score_th=0.1,
        enter_ratio_th=1.05,
        enter_margin_th=0.01,
        exit_score_th=0.01,
        exit_ratio_th=1.0,
        min_enter_windows=1,
        min_exit_windows=1,
        model_name="constant",
    )
    segment = np.zeros((40, 1), dtype=np.float64)
    trials = [
        (TrialSpec(label="8Hz", expected_freq=8.0, trial_id=1, block_index=0), segment),
        (TrialSpec(label="switch_to_10Hz", expected_freq=10.0, trial_id=2, block_index=0), segment),
    ]

    metrics = evaluate_decoder_on_trials(decoder, profile, trials)

    assert int(metrics["switch_trials"]) == 1
    assert int(metrics["switch_detected_trials"]) == 0
    assert int(metrics["switch_penalty_trials"]) == 1
    assert abs(float(metrics["switch_latency_s"]) - 3.0) < 1e-9
