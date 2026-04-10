from __future__ import annotations

from hybrid_controller.adapters.teleop_fallback import RosTeleopFallbackController


def test_next_plan_requires_stall_then_generates_target() -> None:
    fallback = RosTeleopFallbackController(stall_sec=0.5, step_sec=0.25, interval_sec=0.2)
    pose = (10.0, 120.0)
    theta_limits = (-120.0, 120.0)
    radius_limits = (80.0, 260.0)

    assert (
        fallback.next_plan(
            current_pose=pose,
            theta_rate_deg_s=60.0,
            radius_rate_mm_s=100.0,
            now_monotonic=0.0,
            theta_limits_deg=theta_limits,
            radius_limits_mm=radius_limits,
        )
        is None
    )
    assert (
        fallback.next_plan(
            current_pose=pose,
            theta_rate_deg_s=60.0,
            radius_rate_mm_s=100.0,
            now_monotonic=0.3,
            theta_limits_deg=theta_limits,
            radius_limits_mm=radius_limits,
        )
        is None
    )
    plan = fallback.next_plan(
        current_pose=pose,
        theta_rate_deg_s=60.0,
        radius_rate_mm_s=100.0,
        now_monotonic=0.6,
        theta_limits_deg=theta_limits,
        radius_limits_mm=radius_limits,
    )
    assert plan is not None
    assert round(plan.target_theta_deg, 2) == 25.0
    assert round(plan.target_radius_mm, 2) == 145.0


def test_pending_blocks_repeated_plans_until_result_handled() -> None:
    fallback = RosTeleopFallbackController(stall_sec=0.1, step_sec=0.2, interval_sec=0.05)
    pose = (0.0, 120.0)
    theta_limits = (-120.0, 120.0)
    radius_limits = (80.0, 260.0)

    fallback.next_plan(
        current_pose=pose,
        theta_rate_deg_s=0.0,
        radius_rate_mm_s=80.0,
        now_monotonic=0.0,
        theta_limits_deg=theta_limits,
        radius_limits_mm=radius_limits,
    )
    first = fallback.next_plan(
        current_pose=pose,
        theta_rate_deg_s=0.0,
        radius_rate_mm_s=80.0,
        now_monotonic=0.2,
        theta_limits_deg=theta_limits,
        radius_limits_mm=radius_limits,
    )
    assert first is not None
    blocked = fallback.next_plan(
        current_pose=pose,
        theta_rate_deg_s=0.0,
        radius_rate_mm_s=80.0,
        now_monotonic=0.4,
        theta_limits_deg=theta_limits,
        radius_limits_mm=radius_limits,
    )
    assert blocked is None
    assert fallback.pending is True

    should_log_error = fallback.handle_service_result(ok=True, message="", now_monotonic=0.41)
    assert should_log_error is False
    assert fallback.pending is False


def test_non_busy_failure_requests_logging() -> None:
    fallback = RosTeleopFallbackController()
    fallback.pending = True

    assert fallback.handle_service_result(ok=False, message="timeout", now_monotonic=1.0) is True
    assert fallback.pending is False
    fallback.pending = True
    assert fallback.handle_service_result(ok=False, message="BUSY", now_monotonic=1.1) is False
    assert fallback.pending is False
