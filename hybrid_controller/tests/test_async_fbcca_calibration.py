from __future__ import annotations

import uuid

from hybrid_controller.ssvep import async_fbcca_idle as module


def test_fit_threshold_profile_prefers_idle_suppression_then_control_recall() -> None:
    rows = []
    for _ in range(20):
        rows.append(
            {
                "label": "8Hz",
                "expected_freq": 8.0,
                "pred_freq": 8.0,
                "top1_score": 0.030,
                "top2_score": 0.018,
                "margin": 0.012,
                "ratio": 1.67,
                "normalized_top1": 0.625,
                "score_entropy": 0.42,
                "correct": True,
            }
        )
    for _ in range(20):
        rows.append(
            {
                "label": "idle",
                "expected_freq": None,
                "pred_freq": 8.0,
                "top1_score": 0.014,
                "top2_score": 0.0125,
                "margin": 0.0015,
                "ratio": 1.12,
                "normalized_top1": 0.53,
                "score_entropy": 0.91,
                "correct": False,
            }
        )

    profile = module.fit_threshold_profile(
        rows,
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=3.0,
        step_sec=0.25,
        min_enter_windows=2,
        min_exit_windows=2,
    )

    assert profile.enter_score_th >= 0.014
    assert profile.enter_ratio_th >= 1.12
    assert profile.enter_margin_th >= 0.0015
    assert profile.exit_score_th == profile.enter_score_th * 0.85
    assert profile.exit_ratio_th == profile.enter_ratio_th * 0.95
    assert profile.enter_log_lr_th is not None
    assert profile.exit_log_lr_th is not None
    assert profile.control_feature_means is not None
    assert profile.idle_feature_means is not None


def test_profile_roundtrip_preserves_gate_behavior() -> None:
    profile = module.ThresholdProfile(
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=3.0,
        step_sec=0.25,
        enter_score_th=0.02,
        enter_ratio_th=1.15,
        enter_margin_th=0.003,
        exit_score_th=0.017,
        exit_ratio_th=1.0925,
        min_enter_windows=2,
        min_exit_windows=2,
    )
    profile_path = module.DEFAULT_PROFILE_DIR / f".tmp_async_fbcca_profile_{uuid.uuid4().hex}.json"
    try:
        module.save_profile(profile, profile_path)
        loaded = module.load_profile(profile_path)
    finally:
        profile_path.unlink(missing_ok=True)

    gate = module.AsyncDecisionGate.from_profile(loaded)
    first = gate.update(
        {
            "pred_freq": 8.0,
            "top1_score": 0.030,
            "top2_score": 0.020,
            "margin": 0.010,
            "ratio": 1.50,
        }
    )
    second = gate.update(
        {
            "pred_freq": 8.0,
            "top1_score": 0.031,
            "top2_score": 0.020,
            "margin": 0.011,
            "ratio": 1.55,
        }
    )

    assert loaded == profile
    assert first["state"] == "candidate"
    assert second["state"] == "selected"
    assert second["selected_freq"] == 8.0


def test_fit_threshold_profile_falls_back_when_no_correct_control_rows() -> None:
    rows = []
    for _ in range(24):
        rows.append(
            {
                "label": "10Hz",
                "expected_freq": 10.0,
                "pred_freq": 8.0,
                "top1_score": 0.028,
                "top2_score": 0.019,
                "margin": 0.009,
                "ratio": 1.47,
                "normalized_top1": 0.59,
                "score_entropy": 0.49,
                "correct": False,
            }
        )
    for _ in range(24):
        rows.append(
            {
                "label": "idle",
                "expected_freq": None,
                "pred_freq": 8.0,
                "top1_score": 0.013,
                "top2_score": 0.012,
                "margin": 0.001,
                "ratio": 1.08,
                "normalized_top1": 0.52,
                "score_entropy": 0.93,
                "correct": False,
            }
        )

    profile = module.fit_threshold_profile(
        rows,
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=3.0,
        step_sec=0.25,
        min_enter_windows=2,
        min_exit_windows=2,
    )

    assert profile.enter_score_th >= 0.013
    assert profile.enter_ratio_th >= 1.0
    assert profile.enter_log_lr_th is not None


def test_load_profile_can_require_existing_file() -> None:
    missing = module.DEFAULT_PROFILE_DIR / f".tmp_missing_{uuid.uuid4().hex}.json"
    try:
        try:
            module.load_profile(missing, require_exists=True)
            assert False, "expected FileNotFoundError"
        except FileNotFoundError:
            pass
    finally:
        missing.unlink(missing_ok=True)


def test_save_profile_sanitizes_nan_metadata() -> None:
    profile = module.ThresholdProfile(
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=3.0,
        step_sec=0.25,
        enter_score_th=0.02,
        enter_ratio_th=1.15,
        enter_margin_th=0.003,
        exit_score_th=0.017,
        exit_ratio_th=1.0925,
        min_enter_windows=2,
        min_exit_windows=2,
        metadata={"validation_summary": {"mean_detection_latency_sec": float("nan")}},
    )
    profile_path = module.DEFAULT_PROFILE_DIR / f".tmp_async_fbcca_nan_{uuid.uuid4().hex}.json"
    try:
        module.save_profile(profile, profile_path)
        payload = profile_path.read_text(encoding="utf-8")
    finally:
        profile_path.unlink(missing_ok=True)

    assert "NaN" not in payload
    assert "null" in payload


def test_validate_calibration_plan_requires_minimum_repeats() -> None:
    try:
        module.validate_calibration_plan(
            target_repeats=1,
            idle_repeats=2,
            active_sec=4.0,
            preferred_win_sec=3.0,
            step_sec=0.25,
        )
        assert False, "expected ValueError for target_repeats"
    except ValueError as exc:
        assert "target_repeats" in str(exc)

    try:
        module.validate_calibration_plan(
            target_repeats=2,
            idle_repeats=1,
            active_sec=4.0,
            preferred_win_sec=3.0,
            step_sec=0.25,
        )
        assert False, "expected ValueError for idle_repeats"
    except ValueError as exc:
        assert "idle_repeats" in str(exc)


def test_validate_calibration_plan_requires_enough_windows() -> None:
    try:
        module.validate_calibration_plan(
            target_repeats=2,
            idle_repeats=2,
            active_sec=2.0,
            preferred_win_sec=2.0,
            step_sec=0.25,
        )
        assert False, "expected ValueError for insufficient windows"
    except ValueError as exc:
        assert "too short" in str(exc)
