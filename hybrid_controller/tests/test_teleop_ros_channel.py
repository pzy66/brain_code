from __future__ import annotations

from hybrid_controller.adapters.teleop_ros_channel import RosTeleopPublishPlanner
from hybrid_controller.adapters.teleop_ros_channel import new_teleop_cmd_seq_base
from hybrid_controller.adapters.teleop_ros_channel import next_teleop_cmd_seq


def test_ros_teleop_planner_publishes_first_nonzero_command() -> None:
    planner = RosTeleopPublishPlanner(keepalive_interval_sec=0.2)
    command = planner.next_command(theta_rate_deg_s=60.0, radius_rate_mm_s=120.0, now_monotonic=1.0)
    assert command is not None
    assert command.enabled is True
    assert command.theta_rate_deg_s == 60.0
    assert command.radius_rate_mm_s == 120.0
    assert command.z_rate_mm_s == 0.0
    assert command.use_auto_z is True


def test_ros_teleop_planner_sends_keepalive_without_rate_change() -> None:
    planner = RosTeleopPublishPlanner(keepalive_interval_sec=0.2)
    first = planner.next_command(theta_rate_deg_s=20.0, radius_rate_mm_s=0.0, now_monotonic=0.0)
    assert first is not None and first.enabled is True

    early = planner.next_command(theta_rate_deg_s=20.0, radius_rate_mm_s=0.0, now_monotonic=0.05)
    assert early is None

    keepalive = planner.next_command(theta_rate_deg_s=20.0, radius_rate_mm_s=0.0, now_monotonic=0.25)
    assert keepalive is not None
    assert keepalive.enabled is True


def test_ros_teleop_planner_detects_z_rate_changes() -> None:
    planner = RosTeleopPublishPlanner(keepalive_interval_sec=0.2)
    first = planner.next_command(theta_rate_deg_s=0.0, radius_rate_mm_s=0.0, z_rate_mm_s=-12.0, now_monotonic=0.0)
    assert first is not None and first.enabled is True
    assert first.z_rate_mm_s == -12.0

    early = planner.next_command(theta_rate_deg_s=0.0, radius_rate_mm_s=0.0, z_rate_mm_s=-12.0, now_monotonic=0.05)
    assert early is None

    changed = planner.next_command(theta_rate_deg_s=0.0, radius_rate_mm_s=0.0, z_rate_mm_s=-8.0, now_monotonic=0.06)
    assert changed is not None
    assert changed.z_rate_mm_s == -8.0


def test_ros_teleop_planner_detects_auto_z_mode_changes() -> None:
    planner = RosTeleopPublishPlanner(keepalive_interval_sec=0.2)
    first = planner.next_command(
        theta_rate_deg_s=4.0,
        radius_rate_mm_s=0.0,
        use_auto_z=True,
        now_monotonic=0.0,
    )
    assert first is not None and first.use_auto_z is True

    changed = planner.next_command(
        theta_rate_deg_s=4.0,
        radius_rate_mm_s=0.0,
        use_auto_z=False,
        now_monotonic=0.05,
    )
    assert changed is not None
    assert changed.use_auto_z is False


def test_ros_teleop_planner_emits_stop_once_when_rates_go_zero() -> None:
    planner = RosTeleopPublishPlanner(keepalive_interval_sec=0.2)
    started = planner.next_command(theta_rate_deg_s=10.0, radius_rate_mm_s=5.0, now_monotonic=0.0)
    assert started is not None and started.enabled is True

    stop_once = planner.next_command(theta_rate_deg_s=0.0, radius_rate_mm_s=0.0, z_rate_mm_s=0.0, now_monotonic=0.1)
    assert stop_once is not None
    assert stop_once.enabled is False
    assert stop_once.z_rate_mm_s == 0.0
    assert stop_once.use_auto_z is True

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


def test_ros_teleop_cmd_seq_base_changes_across_wall_time() -> None:
    first = new_teleop_cmd_seq_base(now_wall=1000.0)
    later = new_teleop_cmd_seq_base(now_wall=1001.0)

    assert first > 0
    assert later > first


def test_ros_teleop_cmd_seq_wraps_inside_uint32_range() -> None:
    assert next_teleop_cmd_seq((2**32) - 2) == 1
    assert next_teleop_cmd_seq(41) == 42
