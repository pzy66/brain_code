from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from async_fbcca_idle_standalone import merge_idle_transition_segments


def test_merge_idle_transition_segments_concatenates_when_enough_samples() -> None:
    rest = np.zeros((250, 4), dtype=np.float64)
    prepare = np.ones((250, 4), dtype=np.float64)
    merged = merge_idle_transition_segments(rest, prepare, minimum_samples=400)
    assert merged is not None
    assert merged.shape == (500, 4)
    assert float(merged[0, 0]) == 0.0
    assert float(merged[-1, 0]) == 1.0


def test_merge_idle_transition_segments_returns_none_when_too_short() -> None:
    rest = np.zeros((80, 4), dtype=np.float64)
    merged = merge_idle_transition_segments(rest, None, minimum_samples=120)
    assert merged is None


def test_merge_idle_transition_segments_returns_none_on_channel_mismatch() -> None:
    rest = np.zeros((250, 4), dtype=np.float64)
    prepare = np.ones((250, 3), dtype=np.float64)
    merged = merge_idle_transition_segments(rest, prepare, minimum_samples=100)
    assert merged is None
