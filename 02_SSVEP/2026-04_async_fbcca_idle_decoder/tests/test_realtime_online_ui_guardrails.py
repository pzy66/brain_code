from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PyQt5.QtWidgets import QApplication

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from async_fbcca_idle_standalone import ThresholdProfile, load_decoder_from_profile
from ssvep_realtime_online_ui import (
    RealtimeOnlineWindow,
    _validate_loaded_profile,
    resolve_realtime_model_choice,
)


def _get_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_resolve_realtime_model_choice_uses_profile_model() -> None:
    model, mismatch = resolve_realtime_model_choice("trca", "fbcca")
    assert model == "fbcca"
    assert mismatch is True


def test_realtime_window_constructor_smoke() -> None:
    _ = _get_qapp()
    window = RealtimeOnlineWindow(serial_port="auto", board_id=0, freqs=(8.0, 10.0, 12.0, 15.0))
    try:
        assert window.model_combo.currentText() == "fbcca"
        assert "未加载" in window.profile_meta_label.text()
    finally:
        window.close()


def test_validate_loaded_profile_raises_on_channel_weight_mismatch() -> None:
    profile = ThresholdProfile(
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=1.5,
        step_sec=0.25,
        enter_score_th=0.1,
        enter_ratio_th=1.0,
        enter_margin_th=0.01,
        exit_score_th=0.08,
        exit_ratio_th=1.0,
        min_enter_windows=1,
        min_exit_windows=1,
        model_name="fbcca",
        model_params={"Nh": 3},
        eeg_channels=(0, 1, 2, 3, 4, 5, 6, 7),
        channel_weight_mode="fbcca_diag",
        channel_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        subband_weight_mode="chen_ab_subject",
        subband_weights=(0.3, 0.25, 0.2, 0.15, 0.1),
    )
    decoder = load_decoder_from_profile(profile, sampling_rate=250, compute_backend="cpu", gpu_precision="float32")
    with pytest.raises(RuntimeError, match="channel_weights mismatch"):
        _validate_loaded_profile(profile, decoder, eeg_channels=(0, 1, 2, 3))
