from hybrid_controller.vision.pose_buffer import RobotPoseBuffer


def test_robot_pose_buffer_returns_nearest_pose_by_capture_timestamp() -> None:
    buffer = RobotPoseBuffer(capacity=3)
    buffer.add_snapshot(
        {"robot_xy": [0.0, -120.0], "robot_cyl": {"theta_deg": 0.0, "radius_mm": 120.0, "z_mm": 190.0}},
        received_wall_ts=100.0,
        received_perf_ts=10.00,
    )
    buffer.add_snapshot(
        {"robot_xy": [30.0, -120.0], "robot_cyl": {"theta_deg": 14.0, "radius_mm": 124.0, "z_mm": 190.0}},
        received_wall_ts=100.1,
        received_perf_ts=10.20,
    )

    match = buffer.nearest(10.16)

    assert match is not None
    assert match.sample.robot_xy == (30.0, -120.0)
    assert 39.0 <= match.age_ms <= 41.0


def test_robot_pose_buffer_ignores_missing_capture_timestamp() -> None:
    buffer = RobotPoseBuffer(capacity=3)
    buffer.add_snapshot({"robot_xy": [0.0, -120.0]}, received_wall_ts=100.0, received_perf_ts=10.00)

    assert buffer.nearest(None) is None
