from __future__ import annotations

from hybrid_controller.adapters.teleop_ros_channel import RosTeleopPublishPlanner


def test_ros_teleop_planner_publishes_first_nonzero_command() -> None:
    planner = RosTeleopPublishPlanner(keepalive_interval_sec=0.2)
    command = planner.next_command(theta_rate_deg_s=60.0, radius_rate_mm_s=120.0, now_monotonic=1.0)
    assert command is not None
    assert command.enabled is True
    assert command.theta_rate_deg_s == 60.0
    assert command.radius_rate_mm_s == 120.0


def test_ros_teleop_planner_sends_keepalive_without_rate_change() -> None:
    planner = RosTeleopPublishPlanner(keepalive_interval_sec=0.2)
    first = planner.next_command(theta_rate_deg_s=20.0, radius_rate_mm_s=0.0, now_monotonic=0.0)
    assert first is not None and first.enabled is True

    early = planner.next_command(theta_rate_deg_s=20.0, radius_rate_mm_s=0.0, now_monotonic=0.05)
    assert early is None

    keepalive = planner.next_command(theta_rate_deg_s=20.0, radius_rate_mm_s=0.0, now_monotonic=0.25)
    assert keepalive is not None
    assert keepalive.enabled is True


def test_ros_teleop_planner_emits_stop_once_when_rates_go_zero() -> None:
    planner = RosTeleopPublishPlanner(keepalive_interval_sec=0.2)
    started = planner.next_command(theta_rate_deg_s=10.0, radius_rate_mm_s=5.0, now_monotonic=0.0)
    assert started is not None and started.enabled is True

    stop_once = planner.next_command(theta_rate_deg_s=0.0, radius_rate_mm_s=0.0, now_monotonic=0.1)
    assert stop_once is not None
    assert stop_once.enabled is False

    stop_again = planner.next_command(theta_rate_deg_s=0.0, radius_rate_mm_s=0.0, now_monotonic=0.2)
    assert stop_again is None


def test_ros_teleop_planner_republishes_after_publish_failure() -> None:
    planner = RosTeleopPublishPlanner(keepalive_interval_sec=1.0)
    first = planner.next_command(theta_rate_deg_s=30.0, radius_rate_mm_s=30.0, now_monotonic=0.0)
    assert first is not None and first.enabled is True

    planner.on_publish_failed()
    retry = planner.next_command(theta_rate_deg_s=30.0, radius_rate_mm_s=30.0, now_monotonic=0.01)
    assert retry is not None
    assert retry.enabled is True
