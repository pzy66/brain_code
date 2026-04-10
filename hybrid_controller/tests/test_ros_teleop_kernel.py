import math
import time

from hybrid_controller.adapters.rosbridge_client import RosbridgeClient
from hybrid_controller.cylindrical import CylindricalPose
from hybrid_controller.robot.runtime.teleop_kernel import CylindricalTeleopKernel


def _allow_all(theta_deg: float, radius_mm: float, z_mm: float) -> dict[str, object]:
    return {"ok": True, "message": "", "theta_deg": theta_deg, "radius_mm": radius_mm, "z_mm": z_mm}


def test_teleop_kernel_generates_steps_when_enabled() -> None:
    kernel = CylindricalTeleopKernel(
        theta_limits_deg=(-120.0, 120.0),
        radius_limits_mm=(80.0, 260.0),
        auto_z_profile=((80.0, 170.0), (260.0, 180.0)),
        validator=_allow_all,
        tick_hz=20.0,
        deadman_timeout_sec=0.2,
        theta_accel_deg_s2=400.0,
        radius_accel_mm_s2=400.0,
    )
    now = time.monotonic()
    kernel.update_command(theta_rate_deg_s=40.0, radius_rate_mm_s=60.0, enabled=True, timestamp=now)
    step = kernel.step(CylindricalPose(theta_deg=0.0, radius_mm=120.0, z_mm=175.0), now=now + 0.05)
    assert step is not None
    assert step.pose.theta_deg > 0.0
    assert step.pose.radius_mm > 120.0


def test_teleop_kernel_stops_after_deadman_timeout() -> None:
    kernel = CylindricalTeleopKernel(
        theta_limits_deg=(-120.0, 120.0),
        radius_limits_mm=(80.0, 260.0),
        auto_z_profile=((80.0, 170.0), (260.0, 180.0)),
        validator=_allow_all,
        tick_hz=20.0,
        deadman_timeout_sec=0.2,
    )
    now = time.monotonic()
    kernel.update_command(theta_rate_deg_s=30.0, radius_rate_mm_s=0.0, enabled=True, timestamp=now)
    first = kernel.step(CylindricalPose(theta_deg=0.0, radius_mm=120.0, z_mm=175.0), now=now + 0.05)
    assert first is not None
    stale = kernel.step(first.pose, now=now + 0.30)
    assert stale is None


def test_teleop_kernel_respects_validator_failures() -> None:
    kernel = CylindricalTeleopKernel(
        theta_limits_deg=(-120.0, 120.0),
        radius_limits_mm=(80.0, 260.0),
        auto_z_profile=((80.0, 170.0), (260.0, 180.0)),
        validator=lambda theta, radius, z: {"ok": radius <= 125.0, "message": "too_far"},
        tick_hz=20.0,
        deadman_timeout_sec=0.2,
        theta_accel_deg_s2=400.0,
        radius_accel_mm_s2=400.0,
    )
    now = time.monotonic()
    kernel.update_command(theta_rate_deg_s=0.0, radius_rate_mm_s=200.0, enabled=True, timestamp=now)
    step = kernel.step(CylindricalPose(theta_deg=0.0, radius_mm=125.5, z_mm=175.0), now=now + 0.05)
    assert step is None


def test_rosbridge_snapshot_mapping_prefers_cylindrical_state() -> None:
    snapshot = RosbridgeClient._message_to_snapshot(
        {
            "state": "IDLE",
            "busy": False,
            "busy_action": "",
            "carrying": False,
            "theta_deg": 15.0,
            "radius_mm": 180.0,
            "z_mm": 205.0,
            "x_mm": 46.59,
            "y_mm": -173.87,
            "auto_z_current": 205.0,
            "theta_min_deg": -120.0,
            "theta_max_deg": 120.0,
            "radius_min_mm": 50.0,
            "radius_max_mm": 280.0,
            "auto_radius_min_mm": 80.0,
            "auto_radius_max_mm": 260.0,
            "control_kernel": "cylindrical_ros_teleop",
            "last_error_code": "",
            "last_error_message": "",
            "calibration_ready": True,
            "ik_valid": True,
            "validation_error": "",
        }
    )
    assert snapshot["robot_cyl"]["theta_deg"] == 15.0
    assert snapshot["robot_cyl"]["radius_mm"] == 180.0
    assert snapshot["control_kernel"] == "cylindrical_ros_teleop"
    assert math.isclose(snapshot["robot_xy"][0], 46.59, abs_tol=1e-6)
