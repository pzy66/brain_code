from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from async_fbcca_idle_standalone import create_decoder, normalize_model_name


def test_legacy_fbcca_alias_and_decoder_creation() -> None:
    assert normalize_model_name("legacy_fbcca_202603") == "legacy_fbcca_202603"
    assert normalize_model_name("legacy-fbcca-202603") == "legacy_fbcca_202603"
    decoder = create_decoder(
        "legacy_fbcca_202603",
        sampling_rate=250,
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=2.0,
        step_sec=0.25,
        model_params={"Nh": 3},
    )
    assert decoder.model_name == "legacy_fbcca_202603"
    window = np.zeros((decoder.win_samples, 8), dtype=np.float64)
    scores = decoder.score_window(window)
    assert scores.shape == (4,)
    assert np.all(np.isfinite(scores))
