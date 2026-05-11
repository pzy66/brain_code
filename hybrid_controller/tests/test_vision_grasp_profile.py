from __future__ import annotations

import json
import types

import pytest

from hybrid_controller.app import HybridControllerApplication
from hybrid_controller.config import AppConfig
from hybrid_controller.vision.grasp_profile import apply_vision_grasp_profile
from hybrid_controller.vision.grasp_profile import load_vision_grasp_profile


def _write_profile(path, **overrides) -> None:
    payload = {
        "profile_id": "unit-grasp-profile",
        "real_pick_enabled": True,
        "vision_pick_confirm_z_mm": 175.0,
        "vision_eye_in_hand_pick_radius_bias_mm": 50.0,
        "pick_cyl_radius_bias_mm": 0.0,
        "sucker_rotation_angle_quality_threshold": 0.3,
        "vision_servo_low_action_tolerance_px": 7.0,
        "vision_pick_z_tolerance_mm": 5.0,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_and_apply_vision_grasp_profile(tmp_path) -> None:
    profile_path = tmp_path / "profile.json"
    _write_profile(profile_path)
    config = AppConfig(
        vision_grasp_profile_path=profile_path,
        vision_pick_confirm_z_mm=130.0,
        vision_eye_in_hand_pick_radius_bias_mm=40.0,
    ).resolved()

    result = load_vision_grasp_profile(config)
    applied = apply_vision_grasp_profile(config, result).resolved()

    assert result.ready is True
    assert result.profile_id == "unit-grasp-profile"
    assert applied.vision_pick_confirm_z_mm == pytest.approx(175.0)
    assert applied.vision_eye_in_hand_pick_radius_bias_mm == pytest.approx(50.0)
    assert applied.sucker_rotation_angle_quality_threshold == pytest.approx(0.3)
    assert applied.vision_servo_low_action_tolerance_px == pytest.approx(7.0)
    assert applied.vision_pick_z_tolerance_mm == pytest.approx(5.0)


def test_missing_required_profile_blocks_real_pick(tmp_path) -> None:
    app = HybridControllerApplication.__new__(HybridControllerApplication)
    app.config = AppConfig(vision_grasp_profile_path=tmp_path / "missing.json").resolved()
    app._vision_grasp_profile_result = load_vision_grasp_profile(app.config)
    app.runtime_info = {}
    app.statuses = []
    app.events = []
    app._active_pick_trace = None
    app._handle_runtime_status = types.MethodType(
        lambda self, component, message: self.statuses.append((component, message)),
        app,
    )
    app.dispatch_event = types.MethodType(lambda self, event: self.events.append(event), app)
    app._finish_pick_trace = types.MethodType(lambda self, response=None, **_: setattr(self, "_trace_response", response), app)

    assert app._vision_grasp_profile_allows_real_pick() is False
    assert app._reject_pick_without_grasp_profile("PICK_CYL 0.00 160.00") is True
    assert "vision_grasp_profile_missing" in app.statuses[-1][1]
    assert getattr(app.events[-1], "type") == "robot_error"
    assert "ERR Vision grasp profile blocks PICK" in app._trace_response


def test_optional_profile_can_allow_real_pick_without_profile(tmp_path) -> None:
    app = HybridControllerApplication.__new__(HybridControllerApplication)
    app.config = AppConfig(
        vision_grasp_profile_path=tmp_path / "missing.json",
        vision_grasp_profile_real_pick_required=False,
    ).resolved()
    app._vision_grasp_profile_result = load_vision_grasp_profile(app.config)

    assert app._vision_grasp_profile_allows_real_pick() is True


def test_optional_profile_requirement_can_allow_real_pick_without_profile(tmp_path) -> None:
    app = HybridControllerApplication.__new__(HybridControllerApplication)
    app.config = AppConfig(
        vision_grasp_profile_path=tmp_path / "missing.json",
        vision_grasp_profile_required=False,
    ).resolved()
    app._vision_grasp_profile_result = load_vision_grasp_profile(app.config)

    assert app._vision_grasp_profile_allows_real_pick() is True


def test_profile_real_pick_disabled_blocks_pick(tmp_path) -> None:
    profile_path = tmp_path / "profile.json"
    _write_profile(profile_path, real_pick_enabled=False)
    app = HybridControllerApplication.__new__(HybridControllerApplication)
    app.config = AppConfig(vision_grasp_profile_path=profile_path).resolved()
    result = load_vision_grasp_profile(app.config)
    app.config = apply_vision_grasp_profile(app.config, result).resolved()
    app._vision_grasp_profile_result = result

    assert app._vision_grasp_profile_allows_real_pick() is False
