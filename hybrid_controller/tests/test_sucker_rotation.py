from __future__ import annotations

import pytest

from hybrid_controller.robot.runtime.sucker_rotation import (
    map_logical_to_servo_angle_deg,
    normalize_logical_sucker_angle_deg,
)


def test_normalize_logical_sucker_angle_wraps_to_symmetric_range() -> None:
    assert normalize_logical_sucker_angle_deg(50.0) == -40.0
    assert normalize_logical_sucker_angle_deg(-50.0) == 40.0
    assert normalize_logical_sucker_angle_deg(45.0) == 45.0


def test_sucker_rotation_mapping_applies_offset_invert_and_clamp() -> None:
    assert map_logical_to_servo_angle_deg(30.0) == 120.0
    assert map_logical_to_servo_angle_deg(30.0, offset_deg=5.0, invert=True) == 65.0
    assert map_logical_to_servo_angle_deg(44.0, offset_deg=30.0, min_deg=45.0, max_deg=135.0) == 135.0


def test_sucker_rotation_rejects_non_finite_angle() -> None:
    with pytest.raises(ValueError):
        normalize_logical_sucker_angle_deg(float("nan"))
