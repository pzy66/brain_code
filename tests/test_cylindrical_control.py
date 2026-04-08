import math

from hybrid_controller.cylindrical import (
    build_auto_z_profile,
    cartesian_to_cylindrical,
    cylindrical_to_cartesian,
    interpolate_auto_z,
)
from hybrid_controller.config import AppConfig
from hybrid_controller.controller.task_controller import TaskController


def test_cylindrical_round_trip_matches_jetmax_front_axis() -> None:
    x_mm, y_mm, z_mm = cylindrical_to_cartesian(0.0, 150.0, 160.0)
    assert math.isclose(x_mm, 0.0, abs_tol=1e-6)
    assert math.isclose(y_mm, -150.0, abs_tol=1e-6)

    theta_deg, radius_mm, height_mm = cartesian_to_cylindrical(x_mm, y_mm, z_mm)
    assert math.isclose(theta_deg, 0.0, abs_tol=1e-6)
    assert math.isclose(radius_mm, 150.0, abs_tol=1e-6)
    assert math.isclose(height_mm, 160.0, abs_tol=1e-6)


def test_auto_z_profile_is_generated_and_interpolated() -> None:
    profile = build_auto_z_profile(
        radius_limits=(60.0, 120.0),
        z_limits=(100.0, 160.0),
        radius_step_mm=10.0,
        z_step_mm=10.0,
        validator=lambda theta, radius, z: {
            "ok": z >= 110.0,
            "margin": z - 110.0,
            "neutral_distance": abs(z - 140.0),
        },
        preferred_z_mm=140.0,
        profile_theta_deg=0.0,
    )
    assert profile
    interpolated = interpolate_auto_z(profile, 95.0)
    assert 110.0 <= interpolated <= 160.0


def test_auto_z_profile_prefers_preferred_height_over_highest_valid_height() -> None:
    profile = build_auto_z_profile(
        radius_limits=(80.0, 120.0),
        z_limits=(100.0, 180.0),
        radius_step_mm=20.0,
        z_step_mm=10.0,
        validator=lambda theta, radius, z: {
            "ok": True,
            "margin": 100.0,
            "neutral_distance": abs(z - 130.0),
        },
        preferred_z_mm=130.0,
        profile_theta_deg=0.0,
    )
    assert [point[1] for point in profile] == [130.0, 130.0, 130.0]


def test_auto_z_profile_prefers_horizontal_posture_over_preferred_height() -> None:
    def validator(theta: float, radius: float, z: float) -> dict[str, float | bool]:
        posture_distance = abs(z - 170.0)
        return {
            "ok": True,
            "margin": 100.0,
            "neutral_distance": abs(z - 150.0),
            "posture_distance": posture_distance,
        }

    profile = build_auto_z_profile(
        radius_limits=(120.0, 120.0),
        z_limits=(140.0, 180.0),
        radius_step_mm=10.0,
        z_step_mm=10.0,
        validator=validator,
        preferred_z_mm=150.0,
        profile_theta_deg=0.0,
    )
    assert profile == ((120.0, 170.0),)


def test_auto_z_profile_does_not_drop_when_radius_retracts() -> None:
    profile = build_auto_z_profile(
        radius_limits=(100.0, 140.0),
        z_limits=(120.0, 180.0),
        radius_step_mm=20.0,
        z_step_mm=10.0,
        validator=lambda theta, radius, z: {
            "ok": (z >= 140.0) if radius <= 120.0 else (z >= 130.0),
            "margin": 100.0,
            "neutral_distance": abs(z - 140.0),
        },
        preferred_z_mm=140.0,
        profile_theta_deg=0.0,
    )
    z_values = [point[1] for point in profile]
    assert z_values[0] >= z_values[1] >= z_values[2]


def test_auto_z_profile_can_hold_middle_plateau_and_change_at_ends() -> None:
    def target_z(radius: float) -> float:
        if 120.0 <= radius <= 160.0:
            return 150.0
        if radius < 120.0:
            return 130.0
        return 140.0

    profile = build_auto_z_profile(
        radius_limits=(100.0, 180.0),
        z_limits=(120.0, 160.0),
        radius_step_mm=20.0,
        z_step_mm=10.0,
        validator=lambda theta, radius, z: {
            "ok": True,
            "margin": 100.0,
            "neutral_distance": abs(z - 145.0),
            "posture_distance": 0.0,
        },
        preferred_z_mm=145.0,
        profile_theta_deg=0.0,
        target_z_provider=target_z,
        posture_tolerance=0.0,
    )
    assert profile == (
        (100.0, 130.0),
        (120.0, 150.0),
        (140.0, 150.0),
        (160.0, 150.0),
        (180.0, 140.0),
    )


def test_controller_initializes_robot_cyl_and_auto_z() -> None:
    controller = TaskController(AppConfig())
    assert controller.context.robot_cyl[1] > 0.0
    assert controller.context.robot_auto_z is not None
