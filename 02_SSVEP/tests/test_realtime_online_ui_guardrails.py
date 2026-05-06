from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core.async_fbcca_idle_standalone import ThresholdProfile, load_decoder_from_profile, load_profile
import apps.realtime_online_ui as realtime_ui
from apps.realtime_online_ui import (
    DEFAULT_REALTIME_PROFILE_PATH,
    RealtimeConfig,
    RealtimeOnlineWindow,
    RealtimePretrainConfig,
    RealtimePretrainWorker,
    RealtimeWorker,
    build_no_train_fbcca_profile,
    build_pretrain_profile_path,
    pretrain_estimated_seconds,
    pretrain_trial_count,
    realtime_pretrain_protocol_config,
    save_no_train_fbcca_profile,
    _profile_model_name,
    _read_probe_window,
    _validate_loaded_profile,
    resolve_realtime_model_choice,
)
from ssvep_core.stimulus_profiles import STIMULUS_PROFILE_COMFORT_FBCCA_V1


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
        assert "60" in window.btn_pretrain_then_start.text()
        assert "5" in window.btn_full_pretrain_then_start.text()
        assert "FBCCA" in window.btn_no_train_start.text()
    finally:
        window.close()


def test_fbcca_demo_window_locks_model_and_freqs() -> None:
    _ = _get_qapp()
    window = RealtimeOnlineWindow(serial_port="auto", board_id=0, freqs=(9.0, 11.0), demo_mode=True)
    try:
        assert window.windowTitle() == "SSVEP FBCCA Demo"
        assert window.model_combo.currentText() == "fbcca"
        assert not window.model_combo.isEnabled()
        assert window.freqs_edit.isReadOnly()

        window.freqs_edit.setText("9,11")
        cfg = window._read_config()

        assert cfg.model_name == "fbcca"
        assert cfg.freqs == realtime_ui.DEMO_EXPECTED_FREQS
        assert realtime_ui.parse_freqs(window.freqs_edit.text()) == realtime_ui.DEMO_EXPECTED_FREQS
    finally:
        window.close()


def test_fbcca_demo_profile_validation_rejects_wrong_model_or_freqs(tmp_path: Path) -> None:
    good = tmp_path / "profile.json"
    good.write_text(
        json.dumps({"model_name": "fbcca", "freqs": [8.0, 10.0, 12.0, 15.0], "model_params": {"fbcca_variant": "fbcca_cw_sw_all8"}}),
        encoding="utf-8",
    )
    wrong_model = tmp_path / "wrong_model.json"
    wrong_model.write_text(json.dumps({"model_name": "trca", "freqs": [8.0, 10.0, 12.0, 15.0]}), encoding="utf-8")
    wrong_freqs = tmp_path / "wrong_freqs.json"
    wrong_freqs.write_text(json.dumps({"model_name": "fbcca", "freqs": [8.0, 9.0, 12.0, 15.0]}), encoding="utf-8")

    payload = realtime_ui.validate_fbcca_demo_profile_path(good)
    assert payload["model_name"] == "fbcca"

    with pytest.raises(ValueError, match="model_name='fbcca'"):
        realtime_ui.validate_fbcca_demo_profile_path(wrong_model)
    with pytest.raises(ValueError, match="freqs must"):
        realtime_ui.validate_fbcca_demo_profile_path(wrong_freqs)


def test_fbcca_demo_rejects_invalid_profile_picker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _ = _get_qapp()
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"model_name": "trca", "freqs": [8.0, 10.0, 12.0, 15.0]}), encoding="utf-8")
    window = RealtimeOnlineWindow(serial_port="auto", board_id=0, freqs=(8.0, 10.0, 12.0, 15.0), demo_mode=True)
    original_profile = window.profile_edit.text()
    try:
        monkeypatch.setattr(
            realtime_ui.QFileDialog,
            "getOpenFileName",
            lambda *_args, **_kwargs: (str(invalid), "JSON (*.json)"),
        )

        window._pick_profile()

        assert window.profile_edit.text() == original_profile
        assert "Demo profile rejected" in window.log_text.toPlainText()
    finally:
        window.close()


def test_publish_fbcca_profile_to_realtime_and_reject_hybrid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "trained_fbcca_profile.json"
    source.write_text(
        json.dumps({"model_name": "fbcca", "freqs": [8.0, 10.0, 12.0, 15.0], "model_params": {"fbcca_variant": "fbcca_cw_sw_all8"}}),
        encoding="utf-8",
    )
    source_v2 = tmp_path / "trained_fbcca_profile_v2.json"
    source_v2.write_text(json.dumps({"schema": "profile_v2"}), encoding="utf-8")
    realtime_profile = tmp_path / "deployed" / "fbcca_profile.json"
    realtime_profile_v2 = tmp_path / "deployed" / "fbcca_profile_v2.json"
    hybrid_dir = tmp_path / "hybrid_profiles"
    hybrid_current = hybrid_dir / "current_fbcca_profile.json"
    monkeypatch.setattr(realtime_ui, "SSVEP_REALTIME_PROFILE_PATH", realtime_profile)
    monkeypatch.setattr(realtime_ui, "SSVEP_REALTIME_PROFILE_V2_PATH", realtime_profile_v2)
    monkeypatch.setattr(realtime_ui, "HYBRID_PROFILE_DIR", hybrid_dir)
    monkeypatch.setattr(realtime_ui, "HYBRID_CURRENT_PROFILE_PATH", hybrid_current)

    realtime_result = realtime_ui.publish_profile_to_ssvep_realtime(source)
    assert Path(realtime_result["profile_path"]) == realtime_profile
    assert Path(realtime_result["profile_v2_path"]) == realtime_profile_v2
    assert realtime_result["copied_v2"] is True
    assert json.loads(realtime_profile.read_text(encoding="utf-8"))["model_name"] == "fbcca"
    assert json.loads(realtime_profile_v2.read_text(encoding="utf-8"))["schema"] == "profile_v2"
    with pytest.raises(RuntimeError, match="hybrid_controller is disabled"):
        realtime_ui.publish_profile_to_hybrid_controller(source, timestamp=0)
    assert not hybrid_current.exists()


def test_realtime_pretrain_plan_is_about_one_minute(tmp_path: Path) -> None:
    cfg = RealtimePretrainConfig(
        serial_port="auto",
        board_id=0,
        freqs=(8.0, 10.0, 12.0, 15.0),
        base_profile_path=tmp_path / "base.json",
        fallback_profile_path=tmp_path / "fallback.json",
        output_profile_path=tmp_path / "profile.json",
        history_profile_path=tmp_path / "history.json",
        compute_backend="cpu",
        gpu_device=0,
        gpu_precision="float32",
        gpu_warmup=False,
        gpu_cache_policy="windows",
    )

    assert pretrain_trial_count(cfg) == 12
    assert pretrain_estimated_seconds(cfg) == 51.0
    history_name = build_pretrain_profile_path(timestamp=0).name
    assert history_name.startswith("ssvep_fast_fbcca_session_profile_")
    assert history_name.endswith(".json")


def test_realtime_pretrain_protocol_records_comfort_stimulus_provenance(tmp_path: Path) -> None:
    cfg = RealtimePretrainConfig(
        serial_port="auto",
        board_id=0,
        freqs=(8.0, 10.0, 12.0, 15.0),
        base_profile_path=tmp_path / "base.json",
        fallback_profile_path=tmp_path / "fallback.json",
        output_profile_path=tmp_path / "profile.json",
        history_profile_path=tmp_path / "history.json",
        compute_backend="cpu",
        gpu_device=0,
        gpu_precision="float32",
        gpu_warmup=False,
        gpu_cache_policy="windows",
        stimulus_profile_id=STIMULUS_PROFILE_COMFORT_FBCCA_V1,
        stim_refresh_rate_hz=144.0,
    )
    payload = realtime_pretrain_protocol_config(cfg, saved_trial_count=pretrain_trial_count(cfg))

    assert payload["protocol_name"] == "fast-control-pretrain-v1"
    assert payload["stimulus_profile_id"] == STIMULUS_PROFILE_COMFORT_FBCCA_V1
    assert payload["stimulus_mode"] == "elapsed_time_sine"
    assert payload["stimulus_mode_selection_reason"] == "fallback_refresh_not_confirmed_240hz"
    assert abs(float(payload["stim_mean"]) - 0.40) < 1e-12
    assert abs(float(payload["stim_amp"]) - 0.20) < 1e-12
    assert abs(float(payload["stim_luminance_min"]) - 0.20) < 1e-12
    assert abs(float(payload["stim_luminance_max"]) - 0.60) < 1e-12
    assert abs(float(payload["ramp_sec"]) - 0.30) < 1e-12
    assert abs(float(payload["stim_refresh_rate_hz"]) - 144.0) < 1e-12

    frame_locked_cfg = RealtimePretrainConfig(
        serial_port=cfg.serial_port,
        board_id=cfg.board_id,
        freqs=cfg.freqs,
        base_profile_path=cfg.base_profile_path,
        fallback_profile_path=cfg.fallback_profile_path,
        output_profile_path=cfg.output_profile_path,
        history_profile_path=cfg.history_profile_path,
        compute_backend=cfg.compute_backend,
        gpu_device=cfg.gpu_device,
        gpu_precision=cfg.gpu_precision,
        gpu_warmup=cfg.gpu_warmup,
        gpu_cache_policy=cfg.gpu_cache_policy,
        stimulus_profile_id=cfg.stimulus_profile_id,
        stim_refresh_rate_hz=240.0,
    )
    frame_locked_payload = realtime_pretrain_protocol_config(
        frame_locked_cfg,
        saved_trial_count=pretrain_trial_count(frame_locked_cfg),
    )
    assert frame_locked_payload["stimulus_mode"] == "frame_locked_sine"
    assert frame_locked_payload["stimulus_mode_selection_reason"] == "stable_240hz_frame_locked"


def test_no_train_fbcca_profile_is_direct_runtime_safe(tmp_path: Path) -> None:
    profile_path = tmp_path / "fbcca_no_train_profile.json"
    saved_profile, saved_v2 = save_no_train_fbcca_profile(profile_path, freqs=(8.0, 10.0, 12.0, 15.0))
    profile = load_profile(saved_profile, require_exists=True)
    payload_v2 = json.loads(saved_v2.read_text(encoding="utf-8"))

    assert saved_profile == profile_path.resolve()
    assert profile.model_name == "fbcca"
    assert not realtime_ui.profile_is_default_fallback(profile)
    assert dict(profile.metadata or {})["source"] == "no_train_fbcca_direct"
    assert "fast_personalization" not in dict(profile.model_params or {})
    assert payload_v2["gate"]["type"] == "global_threshold"

    built = build_no_train_fbcca_profile((8.0, 10.0, 12.0, 15.0))
    assert built.model_name == "fbcca"
    assert not realtime_ui.profile_is_default_fallback(built)


def test_no_train_button_generates_profile_and_starts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _ = _get_qapp()
    no_train_path = tmp_path / "fbcca_no_train_profile.json"
    monkeypatch.setattr(realtime_ui, "SSVEP_NO_TRAIN_FBCCA_PROFILE_PATH", no_train_path)
    window = RealtimeOnlineWindow(serial_port="auto", board_id=0, freqs=(8.0, 10.0, 12.0, 15.0), demo_mode=True)
    calls: list[str] = []
    try:
        window._start_realtime = lambda: calls.append("start")  # type: ignore[method-assign]
        window._start_no_train_fbcca_realtime()

        assert calls == ["start"]
        assert Path(window.profile_edit.text()) == no_train_path.resolve()
        payload = json.loads(no_train_path.read_text(encoding="utf-8"))
        assert payload["model_name"] == "fbcca"
        assert payload["metadata"]["source"] == "no_train_fbcca_direct"
        assert "fast_personalization" not in dict(payload.get("model_params") or {})
        assert no_train_path.with_name("fbcca_no_train_profile_v2.json").exists()
    finally:
        window.close()


def test_pretrain_profile_ready_defers_autostart_until_cleanup(tmp_path: Path) -> None:
    _ = _get_qapp()
    window = RealtimeOnlineWindow(serial_port="auto", board_id=0, freqs=(8.0, 10.0, 12.0, 15.0))
    calls: list[str] = []
    profile_path = tmp_path / "profile.json"
    history_path = tmp_path / "history.json"
    try:
        window._start_realtime = lambda: calls.append("start")  # type: ignore[method-assign]
        window._start_realtime_after_pretrain = True
        window._on_pretrain_profile_ready(
            {
                "profile_path": str(profile_path),
                "history_profile_path": str(history_path),
                "summary_text": "quality ok",
                "model_name": "fbcca",
                "selected_eeg_channels": [0, 1, 2, 3],
            }
        )

        assert calls == []
        assert Path(window.profile_edit.text()) == profile_path
        assert window._pretrain_profile_ready_for_auto_start is True

        window._on_pretrain_finished()
        assert calls == ["start"]
    finally:
        window.close()


def test_pretrain_profile_ready_surfaces_dataset_save_failure(tmp_path: Path) -> None:
    _ = _get_qapp()
    window = RealtimeOnlineWindow(serial_port="auto", board_id=0, freqs=(8.0, 10.0, 12.0, 15.0))
    profile_path = tmp_path / "profile.json"
    history_path = tmp_path / "history.json"
    try:
        window._on_pretrain_profile_ready(
            {
                "profile_path": str(profile_path),
                "history_profile_path": str(history_path),
                "summary_text": "quality ok",
                "model_name": "fbcca",
                "selected_eeg_channels": [0, 1, 2, 3],
                "dataset_save_valid": False,
                "dataset_save_error": "disk full",
            }
        )

        text = window.profile_meta_label.text()
        assert "dataset_save_valid=0" in text
        assert "dataset=n/a" in text
    finally:
        window.close()


def test_pretrain_worker_wait_requires_first_frame_ack(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(realtime_ui, "REALTIME_STIMULUS_PHASE_APPLY_TIMEOUT_SEC", 0.01)
    worker = RealtimePretrainWorker(
        RealtimePretrainConfig(
            serial_port="auto",
            board_id=0,
            freqs=(8.0, 10.0, 12.0, 15.0),
            base_profile_path=tmp_path / "base.json",
            fallback_profile_path=tmp_path / "fallback.json",
            output_profile_path=tmp_path / "profile.json",
            history_profile_path=tmp_path / "history.json",
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

    worker.notify_stimulus_phase_presented(
        {"mode": realtime_ui.PHASE_CAL_ACTIVE, "frame_index": 0, "presented_t_sec": 0.01}
    )
    assert worker._wait_for_stimulus_phase_presented() is True


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


def test_realtime_result_label_refreshes_gate_metrics_for_repeated_state() -> None:
    _ = _get_qapp()
    window = RealtimeOnlineWindow(serial_port="auto", board_id=0, freqs=(8.0, 10.0, 12.0, 15.0), demo_mode=True)
    try:
        payload = {
            "state": "tracking",
            "pred_freq": 8.0,
            "selected_freq": None,
            "margin": 0.1,
            "ratio": 1.2,
            "stable_windows": 1,
            "control_log_lr": 0.2,
            "acc_log_lr": 0.3,
        }
        window._on_result(payload)
        payload.update({"margin": 0.3, "ratio": 1.7, "stable_windows": 2, "control_log_lr": 0.4})
        window._on_result(payload)

        text = window.result_label.text()
        assert "margin=0.300" in text
        assert "ratio=1.700" in text
        assert "stable_windows=2" in text
        assert "control_lr=0.400" in text
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


def test_realtime_window_forwards_calibration_first_frame_ack_to_pretrain_worker() -> None:
    _ = _get_qapp()
    window = RealtimeOnlineWindow(serial_port="auto", board_id=0, freqs=(8.0, 10.0, 12.0, 15.0))

    class _Worker:
        def __init__(self) -> None:
            self.payloads: list[dict[str, object]] = []

        def notify_stimulus_phase_presented(self, payload: dict[str, object]) -> None:
            self.payloads.append(dict(payload))

    worker = _Worker()
    try:
        window.pretrain_worker = worker  # type: ignore[assignment]
        payload = {"mode": realtime_ui.PHASE_CAL_ACTIVE, "flicker": True, "frame_index": 1}
        window._on_active_phase_frame_presented(payload)
        assert worker.payloads == [payload]
    finally:
        window.pretrain_worker = None
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
