from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core.async_fbcca_idle_standalone import ThresholdProfile, load_decoder_from_profile
import apps.realtime_online_ui as realtime_ui
from apps.realtime_online_ui import (
    DEFAULT_REALTIME_PROFILE_PATH,
    RealtimeConfig,
    RealtimeOnlineWindow,
    RealtimeWorker,
    _profile_model_name,
    _read_probe_window,
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


def test_realtime_stimulus_area_stays_large_after_runtime_updates() -> None:
    app = _get_qapp()
    window = RealtimeOnlineWindow(serial_port="auto", board_id=0, freqs=(8.0, 10.0, 12.0, 15.0))
    window.resize(1600, 900)
    window.show()
    app.processEvents()
    try:
        window._on_profile_info(
            {
                "loaded_profile_path": str(Path("C:/") / ("very_long_profile_path_segment_" * 8) / "fbcca_profile.json"),
                "loaded_profile_model": "fbcca",
                "channel_weight_count": 8,
                "subband_weight_count": 5,
                "backend_requested": "auto",
                "backend_used": "cuda",
                "selection_summary": {
                    "selection_mode": "auto-benchmark",
                    "reason": "cuda-faster " * 16,
                },
                "shadow_summary": {
                    "shadow_mode": "enabled",
                    "gate_mode": "profile_v2",
                    "profile_v2_loaded": True,
                },
            }
        )
        window._on_phase_changed(
            {
                "mode": realtime_ui.PHASE_VALIDATION,
                "title": "实时识别中（fbcca）",
                "detail": "注视目标方块会输出结果；看中心点时不输出。",
                "flicker": True,
                "cue_freq": None,
            }
        )
        app.processEvents()

        assert window.stim.width() >= realtime_ui.REALTIME_STIM_MIN_WIDTH
        assert window.stim.height() >= realtime_ui.REALTIME_STIM_MIN_HEIGHT
    finally:
        window.close()


def test_realtime_focus_mode_hides_controls_and_keeps_stimulus_fullscreen() -> None:
    app = _get_qapp()
    window = RealtimeOnlineWindow(serial_port="auto", board_id=0, freqs=(8.0, 10.0, 12.0, 15.0))
    window.resize(1600, 900)
    window.show()
    app.processEvents()
    try:
        window._set_stimulus_focus_mode(True)
        app.processEvents()

        assert window.isFullScreen()
        assert not window._control_panel.isVisible()
        assert window.stim.width() >= realtime_ui.REALTIME_STIM_MIN_WIDTH

        window._set_stimulus_focus_mode(False)
        app.processEvents()
        assert window._control_panel.isVisible()
    finally:
        window.close()


def test_realtime_result_updates_stimulus_blue_selection() -> None:
    _ = _get_qapp()
    window = RealtimeOnlineWindow(serial_port="auto", board_id=0, freqs=(8.0, 10.0, 12.0, 15.0))
    try:
        window._on_result(
            {
                "state": "selected",
                "pred_freq": 10.0,
                "selected_freq": 10.0,
                "top1_score": 0.42,
                "ratio": 1.6,
                "decision_latency_ms": 3.0,
            }
        )

        assert window.stim.selected_freq == 10.0
        assert window.stim.decoder_state == "selected"
        assert window.stim._border_pen(10.0).color().getRgb() == realtime_ui.REALTIME_SELECTED_BORDER_COLOR.getRgb()
    finally:
        window.close()


def test_realtime_window_forwards_first_frame_ack_to_worker() -> None:
    _ = _get_qapp()
    window = RealtimeOnlineWindow(serial_port="auto", board_id=0, freqs=(8.0, 10.0, 12.0, 15.0))

    class _Worker:
        def __init__(self) -> None:
            self.payloads: list[dict[str, object]] = []

        def notify_stimulus_phase_presented(self, payload: dict[str, object]) -> None:
            self.payloads.append(dict(payload))

    worker = _Worker()
    try:
        window.worker = worker  # type: ignore[assignment]
        payload = {"mode": "validation", "flicker": True, "frame_index": 1}
        window._on_active_phase_frame_presented(payload)
        assert worker.payloads == [payload]
    finally:
        window.worker = None
        window.close()


def test_realtime_worker_wait_requires_first_frame_ack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(realtime_ui, "REALTIME_STIMULUS_PHASE_APPLY_TIMEOUT_SEC", 0.01)
    worker = RealtimeWorker(
        RealtimeConfig(
            serial_port="auto",
            board_id=0,
            freqs=(8.0, 10.0, 12.0, 15.0),
            profile_path=DEFAULT_REALTIME_PROFILE_PATH,
            model_name="fbcca",
            compute_backend="cpu",
            gpu_device=0,
            gpu_precision="float32",
            gpu_warmup=False,
            gpu_cache_policy="windows",
        )
    )
    logs: list[str] = []
    worker.log.connect(lambda text: logs.append(str(text)))  # type: ignore[arg-type]

    assert worker._wait_for_stimulus_phase_presented() is False
    assert any("first-frame acknowledgement timed out" in text for text in logs)

    worker.notify_stimulus_phase_presented({"mode": "validation", "frame_index": 0, "presented_t_sec": 0.01})
    assert worker._wait_for_stimulus_phase_presented() is True


def test_default_realtime_profile_prefers_fbcca_when_available() -> None:
    assert DEFAULT_REALTIME_PROFILE_PATH.exists()
    assert "fbcca" in _profile_model_name(DEFAULT_REALTIME_PROFILE_PATH)


def test_read_probe_window_waits_for_full_profile_window(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, float] = {}

    def fake_ensure_stream_ready(board, sampling_rate, *, minimum_sec, timeout_sec):
        calls["sampling_rate"] = float(sampling_rate)
        calls["minimum_sec"] = float(minimum_sec)
        calls["timeout_sec"] = float(timeout_sec)
        return 500

    class _Board:
        def __init__(self) -> None:
            self.requested_samples = 0

        def get_current_board_data(self, count: int) -> np.ndarray:
            self.requested_samples = int(count)
            return np.zeros((8, int(count)), dtype=float)

    board = _Board()
    monkeypatch.setattr(realtime_ui, "ensure_stream_ready", fake_ensure_stream_ready)

    ready, probe_samples, sample_matrix = _read_probe_window(
        board,
        sampling_rate=250,
        profile_win_sec=2.0,
    )

    assert ready == 500
    assert probe_samples == 500
    assert board.requested_samples == 500
    assert sample_matrix.shape == (8, 500)
    assert calls["sampling_rate"] == 250.0
    assert calls["minimum_sec"] == 2.0
    assert calls["timeout_sec"] >= 4.0


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
