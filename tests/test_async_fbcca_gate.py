from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "02_SSVEP"
    / "2026-04_async_fbcca_idle_decoder"
    / "async_fbcca_idle_standalone.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("async_fbcca_idle_standalone", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_features(
    *,
    pred_freq: float,
    top1: float,
    top2: float,
    margin: float,
    ratio: float,
    normalized_top1: float = 0.6,
    score_entropy: float = 0.4,
    control_log_lr: float | None = None,
) -> dict:
    return {
        "pred_freq": pred_freq,
        "top1_score": top1,
        "top2_score": top2,
        "margin": margin,
        "ratio": ratio,
        "normalized_top1": normalized_top1,
        "score_entropy": score_entropy,
        "control_log_lr": control_log_lr,
    }


def test_gate_enters_selected_after_two_consistent_windows() -> None:
    module = load_module()
    gate = module.AsyncDecisionGate(
        enter_score_th=0.02,
        enter_ratio_th=1.15,
        enter_margin_th=0.003,
        exit_score_th=0.017,
        exit_ratio_th=1.09,
        min_enter_windows=2,
        min_exit_windows=2,
    )

    first = gate.update(make_features(pred_freq=8.0, top1=0.030, top2=0.020, margin=0.010, ratio=1.50))
    second = gate.update(make_features(pred_freq=8.0, top1=0.031, top2=0.020, margin=0.011, ratio=1.55))

    assert first["state"] == "candidate"
    assert first["selected_freq"] is None
    assert second["state"] == "selected"
    assert second["selected_freq"] == 8.0


def test_gate_requires_two_failures_to_release() -> None:
    module = load_module()
    gate = module.AsyncDecisionGate(
        enter_score_th=0.02,
        enter_ratio_th=1.15,
        enter_margin_th=0.003,
        exit_score_th=0.017,
        exit_ratio_th=1.09,
        min_enter_windows=2,
        min_exit_windows=2,
    )

    gate.update(make_features(pred_freq=10.0, top1=0.030, top2=0.018, margin=0.012, ratio=1.67))
    gate.update(make_features(pred_freq=10.0, top1=0.032, top2=0.018, margin=0.014, ratio=1.78))

    first_fail = gate.update(make_features(pred_freq=10.0, top1=0.010, top2=0.009, margin=0.001, ratio=1.05))
    second_fail = gate.update(make_features(pred_freq=10.0, top1=0.011, top2=0.010, margin=0.001, ratio=1.05))

    assert first_fail["state"] == "selected"
    assert first_fail["selected_freq"] == 10.0
    assert second_fail["state"] == "idle"
    assert second_fail["selected_freq"] is None


def test_idle_like_windows_never_select_target() -> None:
    module = load_module()
    gate = module.AsyncDecisionGate(
        enter_score_th=0.02,
        enter_ratio_th=1.15,
        enter_margin_th=0.003,
        exit_score_th=0.017,
        exit_ratio_th=1.09,
        min_enter_windows=2,
        min_exit_windows=2,
    )

    for _ in range(4):
        decision = gate.update(make_features(pred_freq=12.0, top1=0.011, top2=0.010, margin=0.001, ratio=1.04))
        assert decision["state"] == "idle"
        assert decision["selected_freq"] is None


def test_single_bad_window_does_not_drop_selected_state() -> None:
    module = load_module()
    gate = module.AsyncDecisionGate(
        enter_score_th=0.02,
        enter_ratio_th=1.15,
        enter_margin_th=0.003,
        exit_score_th=0.017,
        exit_ratio_th=1.09,
        min_enter_windows=2,
        min_exit_windows=2,
    )

    gate.update(make_features(pred_freq=15.0, top1=0.026, top2=0.017, margin=0.009, ratio=1.53))
    gate.update(make_features(pred_freq=15.0, top1=0.027, top2=0.017, margin=0.010, ratio=1.59))

    jitter = gate.update(make_features(pred_freq=15.0, top1=0.010, top2=0.0095, margin=0.0005, ratio=1.05))
    recovered = gate.update(make_features(pred_freq=15.0, top1=0.028, top2=0.017, margin=0.011, ratio=1.65))

    assert jitter["state"] == "selected"
    assert jitter["selected_freq"] == 15.0
    assert recovered["state"] == "selected"
    assert recovered["selected_freq"] == 15.0


def test_log_lr_gate_can_block_legacy_pass_when_idle_model_disagrees() -> None:
    module = load_module()
    gate = module.AsyncDecisionGate(
        enter_score_th=0.02,
        enter_ratio_th=1.15,
        enter_margin_th=0.003,
        exit_score_th=0.017,
        exit_ratio_th=1.09,
        min_enter_windows=2,
        min_exit_windows=2,
        enter_log_lr_th=1.5,
        exit_log_lr_th=0.2,
    )

    blocked = gate.update(
        make_features(
            pred_freq=8.0,
            top1=0.03,
            top2=0.02,
            margin=0.01,
            ratio=1.5,
            control_log_lr=0.1,
        )
    )
    admitted = gate.update(
        make_features(
            pred_freq=8.0,
            top1=0.03,
            top2=0.02,
            margin=0.01,
            ratio=1.5,
            control_log_lr=2.0,
        )
    )
    selected = gate.update(
        make_features(
            pred_freq=8.0,
            top1=0.031,
            top2=0.02,
            margin=0.011,
            ratio=1.55,
            control_log_lr=2.4,
        )
    )

    assert blocked["state"] == "idle"
    assert admitted["state"] == "candidate"
    assert selected["state"] == "selected"
    assert selected["selected_freq"] == 8.0
