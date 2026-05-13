from __future__ import annotations

import types

from hybrid_controller.app import HybridControllerApplication
from hybrid_controller.config import AppConfig
from hybrid_controller.controller.state_machine import TaskState


def _make_app_stub(
    *,
    config: AppConfig | None = None,
    snapshot: dict[str, object] | None,
    snapshot_age_ms: float,
) -> HybridControllerApplication:
    app = HybridControllerApplication.__new__(HybridControllerApplication)
    app.config = config or AppConfig()
    app.runtime_info = {}

    def _fetch_snapshot(self) -> dict[str, object] | None:
        if snapshot is None:
            return None
        return dict(snapshot)

    def _fetch_age_ms(self) -> float:
        return float(snapshot_age_ms)

    app._fetch_remote_robot_snapshot = types.MethodType(_fetch_snapshot, app)
    app._compute_remote_snapshot_age_ms = types.MethodType(_fetch_age_ms, app)
    return app


def _make_manual_servo_stub(packet: dict[str, object]) -> HybridControllerApplication:
    app = HybridControllerApplication.__new__(HybridControllerApplication)
    app.config = AppConfig(
        vision_servo_max_attempts=3,
        pick_tool_offset_source="target_pixel",
        vision_eye_in_hand_pick_radius_bias_mm=0.0,
    )
    app.runtime_info = {}
    app._latest_vision_packet = packet
    app._vision_servo_pick = None
    app.sent_commands = []
    app.statuses = []
    app._fetch_remote_robot_snapshot = lambda: {
        "robot_cyl": {"theta_deg": 0.0, "radius_mm": 150.0, "z_mm": AppConfig.vision_pick_confirm_z_mm},
        "robot_xy": [0.0, -150.0],
        "robot_z": AppConfig.vision_pick_confirm_z_mm,
    }
    app._compute_remote_snapshot_age_ms = lambda: 0.0

    def _send(self, command: str, **_kwargs) -> None:
        self.sent_commands.append(str(command))

    def _status(self, source: str, message: str) -> None:
        self.statuses.append((str(source), str(message)))

    app._send_robot_text_command = types.MethodType(_send, app)
    app._handle_runtime_status = types.MethodType(_status, app)
    app._rt_get = lambda key, default=None: app.runtime_info.get(key, default)  # type: ignore[method-assign]
    app._rt_set = lambda key, value: app.runtime_info.__setitem__(key, value)  # type: ignore[method-assign]
    app._finish_pick_trace = types.MethodType(lambda self, response=None, **_: setattr(self, "_trace_response", response), app)
    app.dispatch_event = types.MethodType(lambda self, event: self.statuses.append(("event", getattr(event, "value", ""))), app)
    return app


class _DummyContext:
    def __init__(self, selected_target_id: int | None = None) -> None:
        self.selected_target_id = selected_target_id


class _DummyController:
    def __init__(self, selected_target_id: int | None = None, state: object = None) -> None:
        self.context = _DummyContext(selected_target_id)
        self.state = state


class _ContinuousRosClient:
    def __init__(self, *, connected: bool = True) -> None:
        self.connected = bool(connected)
        self.teleop_calls: list[dict[str, object]] = []
        self.stop_calls: list[dict[str, object]] = []

    def is_connected(self) -> bool:
        return self.connected

    def publish_teleop(self, **payload) -> None:
        self.teleop_calls.append(dict(payload))

    def stop_teleop(self, **payload) -> None:
        self.stop_calls.append(dict(payload))


def _packet(
    *,
    mapping_mode: str,
    calibration_ready: bool = True,
    camera_to_world_raw: tuple[float, float, float] = (12.0, -8.0, 0.0),
    world_xyz: tuple[float, float, float] | None = None,
) -> dict[str, object]:
    slot: dict[str, object] = {
        "slot_id": 1,
        "freq_hz": 8.0,
        "valid": True,
        "camera_to_world_raw": [float(camera_to_world_raw[0]), float(camera_to_world_raw[1]), float(camera_to_world_raw[2])],
    }
    if world_xyz is not None:
        slot["world_xyz"] = [float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])]
    return {
        "mapping_mode": mapping_mode,
        "calibration_ready": bool(calibration_ready),
        "slots": [slot],
    }


def _snapshot(robot_xy: tuple[float, float]) -> dict[str, object]:
    return {
        "robot_xy": [float(robot_xy[0]), float(robot_xy[1])],
        "limits_cyl": {
            "theta_deg": [-120.0, 120.0],
            "radius_mm": [50.0, 280.0],
        },
    }


def test_delta_servo_resolution_tracks_robot_pose() -> None:
    packet = _packet(mapping_mode="delta_servo", camera_to_world_raw=(12.0, -8.0, 0.0))

    app_a = _make_app_stub(snapshot=_snapshot((0.0, -120.0)), snapshot_age_ms=20.0)
    resolved_a = app_a._resolve_vision_packet(packet)
    slot_a = resolved_a["slots"][0]

    app_b = _make_app_stub(snapshot=_snapshot((40.0, -120.0)), snapshot_age_ms=20.0)
    resolved_b = app_b._resolve_vision_packet(packet)
    slot_b = resolved_b["slots"][0]

    assert slot_a["actionable"] is True
    assert slot_b["actionable"] is True
    assert slot_a["command_mode"] == "world"
    assert slot_b["command_mode"] == "world"
    assert slot_a["command_point"] == [12.0, -128.0]
    assert slot_b["command_point"] == [52.0, -128.0]
    assert app_a.runtime_info["vision_invalid_reason"] == "--"
    assert app_b.runtime_info["vision_invalid_reason"] == "--"


def test_target_pixel_offset_source_does_not_rewrite_pick_bias() -> None:
    app = _make_app_stub(
        config=AppConfig(pick_tool_offset_source="target_pixel", pick_cyl_radius_bias_mm=46.0),
        snapshot=_snapshot((0.0, -120.0)),
        snapshot_age_ms=20.0,
    )
    app._pick_cyl_radius_bias_mm = 46.0
    app._pick_cyl_tangent_bias_mm = 0.0
    app._pick_cyl_theta_bias_deg = 0.0

    assert app._rewrite_outgoing_robot_command("PICK_WORLD 10.00 -120.00") == "PICK_WORLD 10.00 -120.00"


def test_command_bias_offset_source_keeps_legacy_pick_bias() -> None:
    app = _make_app_stub(
        config=AppConfig(pick_tool_offset_source="command_bias", pick_cyl_radius_bias_mm=46.0),
        snapshot=_snapshot((0.0, -120.0)),
        snapshot_age_ms=20.0,
    )
    app._pick_cyl_radius_bias_mm = 10.0
    app._pick_cyl_tangent_bias_mm = 0.0
    app._pick_cyl_theta_bias_deg = 0.0

    rewritten = app._rewrite_outgoing_robot_command("PICK_CYL 0.00 120.00")

    assert rewritten == "PICK_CYL 0.00 130.00"


def test_delta_servo_resolution_rejects_stale_robot_snapshot() -> None:
    config = AppConfig(vision_snapshot_max_age_ms=200.0)
    app = _make_app_stub(config=config, snapshot=_snapshot((0.0, -120.0)), snapshot_age_ms=500.0)

    resolved = app._resolve_vision_packet(_packet(mapping_mode="delta_servo"))
    slot = resolved["slots"][0]

    assert slot["actionable"] is False
    assert slot["command_point"] is None
    assert slot["invalid_reason"] == "robot_snapshot_stale"
    assert app.runtime_info["vision_invalid_reason"] == "robot_snapshot_stale"


def test_delta_servo_resolution_rejects_too_dark_frame_even_without_valid_slots() -> None:
    config = AppConfig(vision_snapshot_max_age_ms=200.0)
    app = _make_app_stub(config=config, snapshot=_snapshot((0.0, -120.0)), snapshot_age_ms=10.0)

    resolved = app._resolve_vision_packet(
        {
            "mapping_mode": "delta_servo",
            "calibration_ready": True,
            "frame_block_reason": "frame_too_dark",
            "frame_quality": {"too_dark": True, "gray_mean": 16.0, "gray_p95": 16.0},
            "slots": [],
        }
    )

    assert resolved["frame_block_reason"] == "frame_too_dark"
    assert app.runtime_info["vision_invalid_reason"] == "frame_too_dark"


def test_delta_servo_resolution_rejects_valid_slot_on_too_dark_frame() -> None:
    config = AppConfig(vision_snapshot_max_age_ms=200.0)
    app = _make_app_stub(config=config, snapshot=_snapshot((0.0, -120.0)), snapshot_age_ms=10.0)

    resolved = app._resolve_vision_packet(
        {
            "mapping_mode": "delta_servo",
            "calibration_ready": True,
            "frame_block_reason": "frame_too_dark",
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "camera_to_world_raw": [0.0, 0.0, 0.0],
                    "grasp_quality": 1.0,
                }
            ],
        }
    )
    slot = resolved["slots"][0]

    assert slot["actionable"] is False
    assert slot["command_point"] is None
    assert slot["invalid_reason"] == "frame_too_dark"
    assert app.runtime_info["vision_invalid_reason"] == "frame_too_dark"


def test_absolute_base_resolution_does_not_depend_on_robot_pose() -> None:
    world_xyz = (66.0, -170.0, 0.0)
    packet = _packet(mapping_mode="absolute_base", world_xyz=world_xyz)

    app = _make_app_stub(snapshot=None, snapshot_age_ms=float("inf"))
    resolved = app._resolve_vision_packet(packet)
    slot = resolved["slots"][0]

    assert slot["actionable"] is True
    assert slot["command_mode"] == "world"
    assert slot["command_point"] == [66.0, -170.0]
    assert slot["world_xyz"] == [66.0, -170.0, 0.0]
    assert app.runtime_info["vision_invalid_reason"] == "--"


def test_resolution_rejects_target_outside_cylindrical_limits() -> None:
    app = _make_app_stub(snapshot=_snapshot((0.0, -120.0)), snapshot_age_ms=10.0)
    packet = _packet(mapping_mode="delta_servo", camera_to_world_raw=(0.0, 320.0, 0.0))

    resolved = app._resolve_vision_packet(packet)
    slot = resolved["slots"][0]

    assert slot["actionable"] is False
    assert slot["command_point"] is None
    assert slot["invalid_reason"] == "target_out_of_workspace_cyl"
    assert app.runtime_info["vision_invalid_reason"] == "target_out_of_workspace_cyl"


def test_manual_pick_slot_sends_servo_move_before_pick() -> None:
    app = _make_manual_servo_stub(
        {
            "frame_id": 10,
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "actionable": False,
                    "invalid_reason": "vision_servo_required",
                    "servo_command_mode": "cyl",
                    "servo_command_point": [30.0, 150.0],
                }
            ],
        }
    )

    sent = app._send_vision_servo_pick_move(1, app._latest_vision_packet["slots"][0])

    assert sent is True
    assert app.sent_commands == ["MOVE_CYL 30.00 150.00 130.00"]
    assert app._vision_servo_pick["attempts"] == 1
    assert app._vision_servo_pick["stage"] == "fine_center"


def test_manual_pick_slot_servo_descends_one_step_from_search_height() -> None:
    app = _make_manual_servo_stub(
        {
            "frame_id": 10,
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "actionable": True,
                    "command_mode": "world",
                    "command_point": [42.0, -130.0],
                }
            ],
        }
    )
    app._fetch_remote_robot_snapshot = lambda: {
        "robot_cyl": {"theta_deg": 12.0, "radius_mm": 175.0, "z_mm": 190.0},
        "robot_xy": [-36.4, -171.2],
        "robot_z": 190.0,
    }

    sent = app._send_vision_servo_pick_move(1, app._latest_vision_packet["slots"][0])

    assert sent is True
    assert app.sent_commands == ["MOVE_CYL 12.00 175.00 180.00"]
    assert app._vision_servo_pick["stage"] == "low_confirm"


def test_manual_pick_slot_waits_when_grasp_is_near_center_but_unstable() -> None:
    app = _make_manual_servo_stub(
        {
            "frame_id": 10,
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "actionable": False,
                    "invalid_reason": "grasp_unstable",
                    "servo_required": False,
                    "grasp_stable_frames": 1,
                    "center_distance_px": 7.0,
                }
            ],
        }
    )

    sent = app._send_vision_servo_pick_move(1, app._latest_vision_packet["slots"][0])

    assert sent is True
    assert app.sent_commands == []
    assert app._vision_servo_pick["command"] == "WAIT_STABLE"
    assert app._vision_servo_pick["waiting_for_ack"] is False


def test_pending_servo_pick_sends_pick_after_fresh_actionable_frame() -> None:
    app = _make_manual_servo_stub({"frame_id": 10, "slots": []})
    app._vision_servo_pick = {"slot_id": 1, "attempts": 1, "waiting_for_ack": False, "min_frame_id": 11}

    app._pump_pending_vision_servo_pick(
        {
            "frame_id": 11,
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "actionable": True,
                    "command_mode": "world",
                    "command_point": [42.0, -130.0],
                }
            ],
        }
    )

    assert app.sent_commands == ["PICK_WORLD 42.00 -130.00"]
    assert app._vision_servo_pick is None


def test_pending_servo_pick_default_command_bias_sends_single_40mm_forward_pick() -> None:
    app = _make_manual_servo_stub({"frame_id": 10, "slots": []})
    app.config = AppConfig(
        vision_servo_max_attempts=3,
        pick_tool_offset_source="command_bias",
        vision_eye_in_hand_pick_radius_bias_mm=40.0,
        pick_cyl_radius_bias_mm=0.0,
    )
    app._vision_servo_controller = None
    app._pick_cyl_radius_bias_mm = 0.0
    app._fetch_remote_robot_snapshot = lambda: {
        "robot_cyl": {"theta_deg": 7.0, "radius_mm": 160.0, "z_mm": app.config.vision_pick_confirm_z_mm},
        "robot_xy": [-19.5, -158.8],
        "robot_z": app.config.vision_pick_confirm_z_mm,
    }
    app._vision_servo_pick = {"slot_id": 1, "attempts": 1, "waiting_for_ack": False, "min_frame_id": 11}

    app._pump_pending_vision_servo_pick(
        {
            "frame_id": 11,
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "actionable": True,
                    "command_mode": "world",
                    "command_point": [42.0, -130.0],
                }
            ],
        }
    )

    assert app.sent_commands == ["PICK_CYL 7.00 200.00"]
    assert app._vision_servo_pick is None


def test_pending_servo_pick_bypasses_legacy_app_bias_after_forward_offset() -> None:
    app = _make_manual_servo_stub({"frame_id": 10, "slots": []})
    app.config = AppConfig(
        vision_servo_max_attempts=3,
        pick_tool_offset_source="command_bias",
        vision_eye_in_hand_pick_radius_bias_mm=40.0,
        pick_cyl_radius_bias_mm=25.0,
    )
    app._vision_servo_controller = None
    app._pick_cyl_radius_bias_mm = 25.0
    app._pick_cyl_tangent_bias_mm = 0.0
    app._pick_cyl_theta_bias_deg = 0.0
    app._fetch_remote_robot_snapshot = lambda: {
        "robot_cyl": {"theta_deg": 7.0, "radius_mm": 160.0, "z_mm": app.config.vision_pick_confirm_z_mm},
        "robot_xy": [-19.5, -158.8],
        "robot_z": app.config.vision_pick_confirm_z_mm,
    }
    app._vision_servo_pick = {"slot_id": 1, "attempts": 1, "waiting_for_ack": False, "min_frame_id": 11}

    app._pump_pending_vision_servo_pick(
        {
            "frame_id": 11,
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "actionable": True,
                    "command_mode": "world",
                    "command_point": [42.0, -130.0],
                }
            ],
        }
    )

    assert app.sent_commands == ["PICK_CYL 7.00 200.00"]
    assert app.sent_commands != ["PICK_CYL 7.00 225.00"]


def test_pending_servo_pick_lowers_to_confirm_z_before_pick_from_search_height() -> None:
    app = _make_manual_servo_stub({"frame_id": 10, "slots": []})
    app._fetch_remote_robot_snapshot = lambda: {
        "robot_cyl": {"theta_deg": 12.0, "radius_mm": 175.0, "z_mm": 190.0},
        "robot_xy": [-36.4, -171.2],
        "robot_z": 190.0,
    }
    app._vision_servo_pick = {
        "slot_id": 1,
        "attempts": 1,
        "waiting_for_ack": False,
        "min_frame_id": 11,
        "stage": "search",
    }

    app._pump_pending_vision_servo_pick(
        {
            "frame_id": 11,
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "actionable": True,
                    "command_mode": "world",
                    "command_point": [42.0, -130.0],
                }
            ],
        }
    )

    assert app.sent_commands == ["MOVE_CYL 12.00 175.00 180.00"]
    assert app._vision_servo_pick is not None
    assert app._vision_servo_pick["stage"] == "low_confirm"
    assert app._vision_servo_pick["waiting_for_ack"] is True


def test_pending_servo_pick_keeps_waiting_for_unstable_fresh_frames() -> None:
    app = _make_manual_servo_stub({"frame_id": 10, "slots": []})
    app._vision_servo_pick = {
        "slot_id": 1,
        "attempts": 1,
        "waiting_for_ack": False,
        "min_frame_id": 11,
        "stability_wait_frames": 0,
    }

    app._pump_pending_vision_servo_pick(
        {
            "frame_id": 11,
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "actionable": False,
                    "invalid_reason": "grasp_unstable",
                }
            ],
        }
    )

    assert app.sent_commands == []
    assert app._vision_servo_pick is not None
    assert app._vision_servo_pick["stability_wait_frames"] == 1


def test_pending_discrete_servo_pick_cancels_on_too_dark_frame() -> None:
    app = _make_manual_servo_stub({"frame_id": 10, "slots": []})
    app._vision_servo_pick = {"slot_id": 1, "attempts": 1, "waiting_for_ack": False, "min_frame_id": 11}

    app._pump_pending_vision_servo_pick(
        {
            "frame_id": 11,
            "frame_block_reason": "frame_too_dark",
            "frame_quality": {"too_dark": True, "gray_mean": 16.0, "gray_p95": 16.0},
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "actionable": True,
                    "command_mode": "world",
                    "command_point": [42.0, -130.0],
                }
            ],
        }
    )

    assert app.sent_commands == []
    assert app._vision_servo_pick is None
    assert app.runtime_info["vision_servo_status"] == "cancelled slot=1 reason=frame_too_dark"


def test_continuous_servo_does_not_start_without_explicit_pick_intent() -> None:
    app = _make_manual_servo_stub({"frame_id": 1, "slots": []})
    app.config = AppConfig(
        robot_transport="ros",
        vision_continuous_servo_enabled=True,
        vision_continuous_servo_command_timeout_ms=250.0,
    )
    app.ros_client = _ContinuousRosClient()
    app._continuous_vision_servo_controller = None
    app._continuous_vision_servo_pick = None
    app._teleop_cmd_seq = 0
    app.runtime_info["state_age_ms"] = 0.0
    app.controller = types.SimpleNamespace(state=None)
    app._current_robot_cyl_pose = lambda: (0.0, 150.0, 190.0)
    app._uses_ros_transport = types.MethodType(lambda self: True, app)

    app._pump_continuous_vision_servo(
        {
            "frame_id": 2,
            "queue_age_ms": 10.0,
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "actionable": False,
                    "invalid_reason": "vision_servo_required",
                    "center_distance_px": 80.0,
                    "servo_command_mode": "cyl",
                    "servo_command_point": [10.0, 180.0],
                }
            ],
        }
    )

    assert app.ros_client.teleop_calls == []
    assert app.ros_client.stop_calls == []
    assert app._continuous_vision_servo_pick is None
    assert app.runtime_info["vision_servo_status"] == "continuous_idle awaiting_pick"


def test_continuous_servo_publishes_for_existing_pending_slot_only() -> None:
    app = _make_manual_servo_stub({"frame_id": 1, "slots": []})
    app.config = AppConfig(
        robot_transport="ros",
        vision_continuous_servo_enabled=True,
        vision_continuous_servo_command_timeout_ms=250.0,
    )
    app.ros_client = _ContinuousRosClient()
    app._continuous_vision_servo_controller = None
    app._continuous_vision_servo_pick = {"slot_id": 2, "stable_frames": 0, "lost_frames": 0, "source": "manual_pick"}
    app._teleop_cmd_seq = 0
    app.runtime_info["state_age_ms"] = 0.0
    app.controller = types.SimpleNamespace(state=None)
    app._current_robot_cyl_pose = lambda: (0.0, 150.0, 190.0)
    app._uses_ros_transport = types.MethodType(lambda self: True, app)

    app._pump_continuous_vision_servo(
        {
            "frame_id": 2,
            "queue_age_ms": 10.0,
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "actionable": False,
                    "invalid_reason": "vision_servo_required",
                    "center_distance_px": 5.0,
                    "servo_command_mode": "cyl",
                    "servo_command_point": [0.0, 150.0],
                },
                {
                    "slot_id": 2,
                    "valid": True,
                    "actionable": False,
                    "invalid_reason": "vision_servo_required",
                    "center_distance_px": 80.0,
                    "servo_command_mode": "cyl",
                    "servo_command_point": [10.0, 180.0],
                },
            ],
        }
    )

    assert app.ros_client.teleop_calls
    assert app.ros_client.teleop_calls[-1]["enabled"] is True
    assert app._continuous_vision_servo_pick["slot_id"] == 2


def test_manual_pick_slot_prefers_continuous_servo_when_enabled() -> None:
    app = _make_manual_servo_stub(
        {
            "frame_id": 3,
            "queue_age_ms": 10.0,
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "actionable": True,
                    "center_distance_px": 80.0,
                    "servo_command_mode": "cyl",
                    "servo_command_point": [10.0, 180.0],
                    "command_mode": "world",
                    "command_point": [42.0, -130.0],
                }
            ],
        }
    )
    app.config = AppConfig(
        robot_transport="ros",
        vision_continuous_servo_enabled=True,
        vision_continuous_servo_command_timeout_ms=250.0,
    )
    app.ros_client = _ContinuousRosClient()
    app._continuous_vision_servo_controller = None
    app._continuous_vision_servo_pick = None
    app._teleop_cmd_seq = 0
    app.runtime_info["state_age_ms"] = 0.0
    app.controller = types.SimpleNamespace(state=None, context=types.SimpleNamespace(latest_vision_targets=[]))
    app.slot_catalog = None
    app._current_robot_cyl_pose = lambda: (0.0, 150.0, 190.0)
    app._uses_ros_transport = types.MethodType(lambda self: True, app)

    app._on_manual_pick_slot_requested(1)

    assert app.ros_client.teleop_calls
    assert app.sent_commands == []
    assert app._continuous_vision_servo_pick["slot_id"] == 1
    assert app._continuous_vision_servo_pick["source"] == "manual_pick"


def test_manual_pick_slot_blocks_when_continuous_enabled_and_slot_missing() -> None:
    app = _make_manual_servo_stub(
        {
            "frame_id": 4,
            "queue_age_ms": 10.0,
            "slots": [],
        }
    )
    app.config = AppConfig(
        robot_transport="ros",
        vision_continuous_servo_enabled=True,
        vision_continuous_servo_command_timeout_ms=250.0,
    )
    app.ros_client = _ContinuousRosClient()
    app._continuous_vision_servo_pick = None
    app.controller = types.SimpleNamespace(
        state=None,
        context=types.SimpleNamespace(
            latest_vision_targets=[
                types.SimpleNamespace(
                    id=1,
                    slot_id=1,
                    actionable=True,
                    command_mode="world",
                    command_point=[42.0, -130.0],
                )
            ]
        ),
    )
    app._uses_ros_transport = types.MethodType(lambda self: True, app)

    app._on_manual_pick_slot_requested(1)

    assert app.sent_commands == []
    assert app.ros_client.teleop_calls == []
    assert app._continuous_vision_servo_pick is None
    assert "continuous_start_blocked:slot_unavailable:1" in app.runtime_info["vision_servo_status"]
    assert [status for status in app.statuses if status[0] == "event"] == []


def test_task_pick_command_starts_continuous_servo_instead_of_direct_pick() -> None:
    app = _make_manual_servo_stub(
        {
            "frame_id": 3,
            "queue_age_ms": 10.0,
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "actionable": False,
                    "invalid_reason": "vision_servo_required",
                    "center_distance_px": 80.0,
                    "servo_command_mode": "cyl",
                    "servo_command_point": [10.0, 180.0],
                    "command": "PICK_WORLD 42.00 -130.00",
                    "command_mode": "world",
                    "command_point": [42.0, -130.0],
                }
            ],
        }
    )
    app.config = AppConfig(
        robot_transport="ros",
        vision_continuous_servo_enabled=True,
        vision_continuous_servo_command_timeout_ms=250.0,
    )
    app.ros_client = _ContinuousRosClient()
    app._continuous_vision_servo_controller = None
    app._continuous_vision_servo_pick = None
    app._teleop_cmd_seq = 0
    app.runtime_info["state_age_ms"] = 0.0
    app.controller = _DummyController(selected_target_id=1, state=TaskState.S2_PICKING)
    app._current_robot_cyl_pose = lambda: (0.0, 150.0, 190.0)
    app._uses_ros_transport = types.MethodType(lambda self: True, app)

    app._send_robot_command(types.SimpleNamespace(payload={"command": "PICK_WORLD 42.00 -130.00"}))

    assert app.sent_commands == []
    assert app.ros_client.teleop_calls
    assert app._continuous_vision_servo_pick["slot_id"] == 1
    assert app._continuous_vision_servo_pick["source"] == "task_confirm"


def test_task_pick_command_blocked_without_selected_slot_does_not_fall_back_to_pick() -> None:
    app = _make_manual_servo_stub(
        {
            "frame_id": 3,
            "queue_age_ms": 10.0,
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "actionable": True,
                    "center_distance_px": 8.0,
                    "command": "PICK_WORLD 42.00 -130.00",
                    "command_mode": "world",
                    "command_point": [42.0, -130.0],
                }
            ],
        }
    )
    app.config = AppConfig(
        robot_transport="ros",
        vision_continuous_servo_enabled=True,
        vision_continuous_servo_command_timeout_ms=250.0,
    )
    app.ros_client = _ContinuousRosClient()
    app._continuous_vision_servo_pick = None
    app.controller = _DummyController(selected_target_id=None, state=TaskState.S2_PICKING)
    app._uses_ros_transport = types.MethodType(lambda self: True, app)

    app._send_robot_command(types.SimpleNamespace(payload={"command": "PICK_WORLD 42.00 -130.00"}))

    assert app.sent_commands == []
    assert app.ros_client.teleop_calls == []
    assert app._continuous_vision_servo_pick is None
    assert "continuous_start_blocked:selected_slot_unavailable" in app.runtime_info["vision_servo_status"]


def test_task_pick_command_blocked_on_stale_state_does_not_fall_back_to_pick() -> None:
    app = _make_manual_servo_stub(
        {
            "frame_id": 3,
            "queue_age_ms": 10.0,
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "actionable": True,
                    "center_distance_px": 8.0,
                    "command": "PICK_WORLD 42.00 -130.00",
                    "command_mode": "world",
                    "command_point": [42.0, -130.0],
                }
            ],
        }
    )
    app.config = AppConfig(
        robot_transport="ros",
        vision_continuous_servo_enabled=True,
        vision_continuous_servo_command_timeout_ms=250.0,
        robot_state_stale_threshold_ms=100.0,
    )
    app.ros_client = _ContinuousRosClient()
    app._continuous_vision_servo_pick = None
    app.runtime_info["state_age_ms"] = 250.0
    app.controller = _DummyController(selected_target_id=1, state=TaskState.S2_PICKING)
    app._uses_ros_transport = types.MethodType(lambda self: True, app)

    app._send_robot_command(types.SimpleNamespace(payload={"command": "PICK_WORLD 42.00 -130.00"}))

    assert app.sent_commands == []
    assert app.ros_client.teleop_calls == []
    assert app._continuous_vision_servo_pick is None
    assert "continuous_start_blocked:robot_state_not_fresh" in app.runtime_info["vision_servo_status"]


def test_continuous_servo_stops_on_stale_frame_in_app_integration() -> None:
    app = _make_manual_servo_stub({"frame_id": 1, "slots": []})
    app.config = AppConfig(
        robot_transport="ros",
        vision_continuous_servo_enabled=True,
        vision_continuous_servo_command_timeout_ms=100.0,
    )
    app.ros_client = _ContinuousRosClient()
    app._continuous_vision_servo_controller = None
    app._continuous_vision_servo_pick = {"slot_id": 1, "stable_frames": 0}
    app._teleop_cmd_seq = 0
    app.runtime_info["state_age_ms"] = 0.0
    app.controller = types.SimpleNamespace(state=None)
    app._current_robot_cyl_pose = lambda: (0.0, 150.0, 190.0)
    app._uses_ros_transport = types.MethodType(lambda self: True, app)

    app._pump_continuous_vision_servo(
        {
            "frame_id": 3,
            "queue_age_ms": 250.0,
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "actionable": False,
                    "invalid_reason": "vision_servo_required",
                    "center_distance_px": 8.0,
                    "servo_command_mode": "cyl",
                    "servo_command_point": [0.0, 150.0],
                }
            ],
        }
    )

    assert app.ros_client.teleop_calls == []
    assert app.ros_client.stop_calls
    assert "fresh frame 1" in app.runtime_info["vision_servo_status"]
    assert app._continuous_vision_servo_pick is not None
    assert app._continuous_vision_servo_pick["stale_frames"] == 1

    app._pump_continuous_vision_servo(
        {
            "frame_id": 4,
            "queue_age_ms": 250.0,
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "actionable": False,
                    "invalid_reason": "vision_servo_required",
                    "center_distance_px": 8.0,
                    "servo_command_mode": "cyl",
                    "servo_command_point": [0.0, 150.0],
                }
            ],
        }
    )

    assert app._continuous_vision_servo_pick is not None
    assert app._continuous_vision_servo_pick["stale_frames"] == 2

    app._pump_continuous_vision_servo(
        {
            "frame_id": 5,
            "queue_age_ms": 250.0,
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "actionable": False,
                    "invalid_reason": "vision_servo_required",
                    "center_distance_px": 8.0,
                    "servo_command_mode": "cyl",
                    "servo_command_point": [0.0, 150.0],
                }
            ],
        }
    )

    assert "frame_stale" in app.runtime_info["vision_servo_status"]
    assert app._continuous_vision_servo_pick is None


def test_continuous_servo_stops_on_too_dark_frame_in_app_integration() -> None:
    app = _make_manual_servo_stub({"frame_id": 1, "slots": []})
    app.config = AppConfig(
        robot_transport="ros",
        vision_continuous_servo_enabled=True,
        vision_continuous_servo_command_timeout_ms=250.0,
    )
    app.ros_client = _ContinuousRosClient()
    app._continuous_vision_servo_controller = None
    app._continuous_vision_servo_pick = {"slot_id": 1, "stable_frames": 1}
    app._teleop_cmd_seq = 0
    app.runtime_info["state_age_ms"] = 0.0
    app.controller = types.SimpleNamespace(state=None)
    app._current_robot_cyl_pose = lambda: (0.0, 150.0, 190.0)
    app._uses_ros_transport = types.MethodType(lambda self: True, app)

    app._pump_continuous_vision_servo(
        {
            "frame_id": 3,
            "queue_age_ms": 10.0,
            "frame_block_reason": "frame_too_dark",
            "frame_quality": {"too_dark": True, "gray_mean": 16.0, "gray_p95": 16.0},
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "actionable": False,
                    "invalid_reason": "vision_servo_required",
                    "center_distance_px": 8.0,
                    "servo_command_mode": "cyl",
                    "servo_command_point": [0.0, 150.0],
                }
            ],
        }
    )

    assert app.ros_client.teleop_calls == []
    assert app.ros_client.stop_calls
    assert "frame_too_dark" in app.runtime_info["vision_servo_status"]
    assert app._continuous_vision_servo_pick is None


def test_manual_pick_slot_blocks_on_too_dark_frame_without_legacy_pick_fallback() -> None:
    app = _make_manual_servo_stub(
        {
            "frame_id": 3,
            "queue_age_ms": 10.0,
            "frame_block_reason": "frame_too_dark",
            "frame_quality": {"too_dark": True, "gray_mean": 16.0, "gray_p95": 16.0},
            "slots": [],
        }
    )
    app.config = AppConfig(
        robot_transport="ros",
        vision_continuous_servo_enabled=True,
        vision_continuous_servo_command_timeout_ms=250.0,
    )
    app.ros_client = _ContinuousRosClient()
    app._continuous_vision_servo_controller = None
    app._continuous_vision_servo_pick = None
    app._teleop_cmd_seq = 0
    app.runtime_info["state_age_ms"] = 0.0
    app.controller = types.SimpleNamespace(state=None, context=types.SimpleNamespace(latest_vision_targets=[]))
    app.slot_catalog = None
    app._current_robot_cyl_pose = lambda: (0.0, 150.0, 190.0)
    app._uses_ros_transport = types.MethodType(lambda self: True, app)

    app._on_manual_pick_slot_requested(1)

    assert app.sent_commands == []
    assert app.ros_client.teleop_calls == []
    assert app._continuous_vision_servo_pick is None
    assert "continuous_start_blocked:frame_too_dark" in app.runtime_info["vision_servo_status"]


def test_app_continuous_servo_keeps_streaming_near_center_when_correction_exists() -> None:
    app = _make_manual_servo_stub({"frame_id": 1, "slots": []})
    app.config = AppConfig(
        robot_transport="ros",
        vision_continuous_servo_enabled=True,
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_settle_stop_band_px=8.0,
        vision_continuous_servo_command_timeout_ms=250.0,
    )
    app.ros_client = _ContinuousRosClient()
    app._continuous_vision_servo_controller = None
    app._continuous_vision_servo_pick = {"slot_id": 1, "stable_frames": 0, "lost_frames": 0}
    app._teleop_cmd_seq = 0
    app.runtime_info["state_age_ms"] = 0.0
    app.controller = types.SimpleNamespace(state=None)
    app._current_robot_cyl_pose = lambda: (0.0, 150.0, 123.0)
    app._uses_ros_transport = types.MethodType(lambda self: True, app)

    app._pump_continuous_vision_servo(
        {
            "frame_id": 3,
            "queue_age_ms": 10.0,
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "actionable": False,
                    "invalid_reason": "vision_servo_required",
                    "center_distance_px": 7.2,
                    "servo_command_mode": "cyl",
                    "servo_command_point": [1.0, 154.0],
                }
            ],
        }
    )

    assert app.ros_client.teleop_calls
    assert app.ros_client.stop_calls == []
    assert app._continuous_vision_servo_pick is not None
    assert app._continuous_vision_servo_pick["stable_frames"] == 1
    assert app.runtime_info["vision_servo_status"] == "servo slot=1"


def test_app_continuous_servo_keeps_teleop_continuous_near_center() -> None:
    app = _make_manual_servo_stub({"frame_id": 1, "slots": []})
    app.config = AppConfig(
        robot_transport="ros",
        vision_continuous_servo_enabled=True,
        vision_pick_confirm_z_mm=120.0,
        vision_continuous_servo_settle_stop_band_px=1.0,
        vision_continuous_servo_fine_pulse_center_px=8.0,
        vision_continuous_servo_command_timeout_ms=250.0,
    )
    app.ros_client = _ContinuousRosClient()
    app._continuous_vision_servo_controller = None
    app._continuous_vision_servo_pick = {"slot_id": 1, "stable_frames": 0, "lost_frames": 0}
    app._teleop_cmd_seq = 0
    app.runtime_info["state_age_ms"] = 0.0
    app.controller = types.SimpleNamespace(state=None)
    app._current_robot_cyl_pose = lambda: (0.0, 150.0, 140.0)
    app._uses_ros_transport = types.MethodType(lambda self: True, app)

    app._pump_continuous_vision_servo(
        {
            "frame_id": 3,
            "queue_age_ms": 10.0,
            "slots": [
                {
                    "slot_id": 1,
                    "valid": True,
                    "actionable": False,
                    "invalid_reason": "vision_servo_required",
                    "center_distance_px": 7.2,
                    "servo_command_mode": "cyl",
                    "servo_command_point": [1.0, 154.0],
                }
            ],
        }
    )

    assert app.ros_client.teleop_calls
    assert app.ros_client.stop_calls == []


def test_app_continuous_servo_uses_low_height_discrete_refine_at_confirm_height() -> None:
    app = _make_manual_servo_stub({"frame_id": 10, "slots": []})
    app.config = AppConfig(
        robot_transport="ros",
        vision_continuous_servo_enabled=True,
        vision_pick_confirm_z_mm=120.0,
        vision_pick_z_tolerance_mm=4.0,
        vision_continuous_servo_pick_ready_center_px=2.0,
        vision_continuous_servo_low_height_discrete_refine_enabled=True,
        vision_continuous_servo_low_height_refine_max_theta_step_deg=0.25,
        vision_continuous_servo_low_height_refine_max_radius_step_mm=1.5,
        vision_continuous_servo_command_timeout_ms=250.0,
    )
    app.ros_client = _ContinuousRosClient()
    app._continuous_vision_servo_controller = None
    app._continuous_vision_servo_pick = {"slot_id": 1, "stable_frames": 1, "lost_frames": 0, "source": "manual_pick"}
    app._teleop_cmd_seq = 0
    app.runtime_info["state_age_ms"] = 0.0
    app.controller = types.SimpleNamespace(state=None)
    app._current_robot_cyl_pose = lambda: (7.80, 174.00, 120.0)
    app._uses_ros_transport = types.MethodType(lambda self: True, app)

    packet = {
        "frame_id": 10,
        "queue_age_ms": 10.0,
        "slots": [
            {
                "slot_id": 1,
                "valid": True,
                "actionable": False,
                "invalid_reason": "vision_servo_required",
                "center_distance_px": 6.0,
                "servo_command_mode": "cyl",
                "servo_command_point": [8.20, 176.30],
            }
        ],
    }
    app._latest_vision_packet = packet

    app._pump_continuous_vision_servo(packet)

    assert app.ros_client.teleop_calls == []
    assert app.ros_client.stop_calls
    assert app.sent_commands == ["MOVE_CYL 8.05 175.50 120.00"]
    assert app._continuous_vision_servo_pick["low_height_refine_attempts"] == 1
    assert app._continuous_vision_servo_pick["waiting_for_refine_ack"] is True
    assert app._continuous_vision_servo_pick["min_frame_id"] == 11
    assert "continuous_low_refine attempt=1" in app.runtime_info["vision_servo_status"]


def test_app_continuous_refine_ack_waits_for_fresh_frame_before_next_motion() -> None:
    app = _make_manual_servo_stub({"frame_id": 10, "slots": []})
    app.config = AppConfig(
        robot_transport="ros",
        vision_continuous_servo_enabled=True,
        vision_pick_confirm_z_mm=120.0,
        vision_pick_z_tolerance_mm=4.0,
        vision_continuous_servo_pick_ready_center_px=2.0,
        vision_continuous_servo_command_timeout_ms=250.0,
    )
    app.ros_client = _ContinuousRosClient()
    app._continuous_vision_servo_pick = {
        "slot_id": 1,
        "stable_frames": 1,
        "lost_frames": 0,
        "source": "manual_pick",
        "low_height_refine_attempts": 1,
        "waiting_for_refine_ack": True,
        "min_frame_id": 11,
    }
    app._teleop_cmd_seq = 0
    app.runtime_info["state_age_ms"] = 0.0
    app.controller = types.SimpleNamespace(state=None)
    app._current_robot_cyl_pose = lambda: (8.05, 175.50, 120.0)
    app._uses_ros_transport = types.MethodType(lambda self: True, app)
    app._latest_vision_packet = {"frame_id": 10, "slots": []}

    app._mark_continuous_refine_move_acknowledged()

    assert app._continuous_vision_servo_pick["waiting_for_refine_ack"] is False
    assert app._continuous_vision_servo_pick["min_frame_id"] == 11

    stale_packet = {
        "frame_id": 10,
        "queue_age_ms": 10.0,
        "slots": [
            {
                "slot_id": 1,
                "valid": True,
                "actionable": False,
                "invalid_reason": "vision_servo_required",
                "center_distance_px": 6.0,
                "servo_command_mode": "cyl",
                "servo_command_point": [8.20, 176.30],
            }
        ],
    }
    app._latest_vision_packet = stale_packet
    app._pump_continuous_vision_servo(stale_packet)

    assert app.sent_commands == []
    assert app.ros_client.teleop_calls == []
    assert "continuous_wait_fresh_frame" in app.runtime_info["vision_servo_status"]


def test_app_continuous_default_low_height_refine_attempts_do_not_block_teleop() -> None:
    app = _make_manual_servo_stub({"frame_id": 20, "slots": []})
    app.config = AppConfig(
        robot_transport="ros",
        vision_continuous_servo_enabled=True,
        vision_pick_confirm_z_mm=120.0,
        vision_pick_z_tolerance_mm=4.0,
        vision_continuous_servo_pick_ready_center_px=2.0,
        vision_continuous_servo_low_height_refine_attempts=1,
        vision_continuous_servo_low_height_discrete_refine_enabled=False,
        vision_continuous_servo_command_timeout_ms=250.0,
    )
    app.ros_client = _ContinuousRosClient()
    app._continuous_vision_servo_controller = None
    app._continuous_vision_servo_pick = {
        "slot_id": 1,
        "stable_frames": 1,
        "lost_frames": 0,
        "source": "manual_pick",
        "low_height_refine_attempts": 1,
    }
    app._teleop_cmd_seq = 0
    app.runtime_info["state_age_ms"] = 0.0
    app.controller = types.SimpleNamespace(state=None)
    app._current_robot_cyl_pose = lambda: (7.80, 174.00, 120.0)
    app._uses_ros_transport = types.MethodType(lambda self: True, app)
    packet = {
        "frame_id": 20,
        "queue_age_ms": 10.0,
        "slots": [
            {
                "slot_id": 1,
                "valid": True,
                "actionable": False,
                "invalid_reason": "vision_servo_required",
                "center_distance_px": 6.0,
                "servo_command_mode": "cyl",
                "servo_command_point": [8.20, 176.30],
            }
        ],
    }
    app._latest_vision_packet = packet

    app._pump_continuous_vision_servo(packet)

    assert app.sent_commands == []
    assert app.ros_client.teleop_calls
    assert app.ros_client.stop_calls == []
    assert app._continuous_vision_servo_pick is not None
    assert app.runtime_info["vision_servo_status"] == "servo slot=1"


def test_discrete_low_confirm_large_error_is_blocked_until_local_calibration() -> None:
    from hybrid_controller.vision.servo_controller import VisionServoController

    controller = VisionServoController(
        AppConfig(
            vision_pick_confirm_z_mm=120.0,
            vision_pick_z_tolerance_mm=4.0,
            vision_low_confirm_untrusted_error_px=12.0,
        )
    )

    decision = controller.decide(
        slot_id=1,
        slot_payload={
            "slot_id": 1,
            "valid": True,
            "actionable": False,
            "invalid_reason": "vision_servo_required",
            "center_distance_px": 15.8,
            "servo_command_mode": "cyl",
            "servo_command_point": [8.54, 174.76],
        },
        packet={"frame_id": 10},
        pending={"slot_id": 1, "state": "LOW_CONFIRM", "attempts": 1},
        current_cyl_pose=(7.98, 172.90, 120.0),
        eye_in_hand_enabled=True,
    )

    assert decision.action == "CANCEL"
    assert decision.reason == "low_confirm_alignment_untrusted"
    assert decision.trace["center_distance_px"] == 15.8


def test_continuous_servo_clears_pending_after_lost_target_threshold() -> None:
    app = _make_manual_servo_stub({"frame_id": 1, "slots": []})
    app.config = AppConfig(
        robot_transport="ros",
        vision_continuous_servo_enabled=True,
        vision_continuous_servo_lost_frames=2,
        vision_continuous_servo_command_timeout_ms=250.0,
    )
    app.ros_client = _ContinuousRosClient()
    app._continuous_vision_servo_controller = None
    app._continuous_vision_servo_pick = {"slot_id": 1, "stable_frames": 0, "lost_frames": 1}
    app._teleop_cmd_seq = 0
    app.runtime_info["state_age_ms"] = 0.0
    app.controller = types.SimpleNamespace(state=None)
    app._current_robot_cyl_pose = lambda: (0.0, 150.0, 190.0)
    app._uses_ros_transport = types.MethodType(lambda self: True, app)

    app._pump_continuous_vision_servo({"frame_id": 3, "queue_age_ms": 10.0, "slots": []})

    assert app.ros_client.teleop_calls == []
    assert app.ros_client.stop_calls
    assert "lost_target" in app.runtime_info["vision_servo_status"]
    assert app._continuous_vision_servo_pick is None


def test_app_continuous_servo_stop_at_confirm_blocks_pick_ready_command() -> None:
    app = _make_manual_servo_stub({"frame_id": 1, "slots": []})
    app.config = AppConfig(
        robot_transport="ros",
        vision_continuous_servo_enabled=True,
        vision_continuous_servo_stop_at_confirm=True,
    )
    app.ros_client = _ContinuousRosClient()
    app._continuous_vision_servo_pick = {"slot_id": 1, "stable_frames": 2, "source": "manual_pick"}
    app._teleop_cmd_seq = 0
    app.runtime_info["release_mode_effective"] = "sucker_frozen"
    decision = types.SimpleNamespace(
        action="PICK_READY",
        command="PICK_CYL 7.00 190.00",
        status="pick_ready slot=1",
        pending=types.SimpleNamespace(slot_id=1),
        pending_dict={"slot_id": 1, "stable_frames": 2},
        trace={"center_distance_px": 1.2, "confirm_z_mm": 120.0},
    )

    handled = app._apply_continuous_vision_servo_decision(decision)

    assert handled is True
    assert app.ros_client.stop_calls
    assert app.sent_commands == []
    assert app._continuous_vision_servo_pick is None
    assert app.runtime_info["vision_servo_confirm_command_blocked"] == "PICK_CYL 7.00 190.00"
    assert "continuous_stop_at_confirm" in app.runtime_info["vision_servo_status"]
    assert "no_pick" in app.runtime_info["vision_servo_status"]
    assert "sucker_frozen" in app.runtime_info["vision_servo_status"]


def test_robot_failure_events_clear_continuous_servo_pending() -> None:
    app = _make_manual_servo_stub({"frame_id": 1, "slots": []})
    app.config = AppConfig(robot_mode="fake")
    app._continuous_vision_servo_pick = {"slot_id": 1, "stable_frames": 0, "lost_frames": 0}
    app._vision_servo_pick = {"slot_id": 1}
    app._stop_teleop_motion = types.MethodType(lambda self, **_: None, app)
    app._set_ssvep_stim_enabled = types.MethodType(lambda self, *_args, **_kwargs: False, app)
    app._log_runtime = types.MethodType(lambda self, *_args, **_kwargs: None, app)
    app._update_control_scene_from_event = types.MethodType(lambda self, *_args, **_kwargs: None, app)
    app._update_runtime_health = types.MethodType(lambda self: None, app)
    app._update_ssvep_mode = types.MethodType(lambda self: None, app)
    app._refresh_view = types.MethodType(lambda self: None, app)
    app._resolve_pending_command_from_event = types.MethodType(lambda self, _event: None, app)
    app._apply_effect = types.MethodType(lambda self, _effect: None, app)
    app.logger = types.SimpleNamespace(log_event=lambda *_args, **_kwargs: None, log_effect=lambda *_args, **_kwargs: None)
    app.controller = types.SimpleNamespace(
        state=TaskState.S2_PICKING,
        context=types.SimpleNamespace(selected_target_id=1),
        handle_event=lambda _event: [],
    )

    from hybrid_controller.controller.events import Event

    HybridControllerApplication.dispatch_event(app, Event(source="robot", type="robot_error", value="unit_error"))

    assert app._continuous_vision_servo_pick is None
    assert app._vision_servo_pick is None


def test_pick_cyl_radius_out_of_limits_is_rejected() -> None:
    app = _make_manual_servo_stub({"frame_id": 1, "slots": []})
    app.config = AppConfig(robot_mode="fake", robot_auto_radius_limits_mm=(80.0, 260.0))
    app._vision_grasp_profile_result = types.SimpleNamespace(ready=True, profile=types.SimpleNamespace(real_pick_enabled=True), error="")

    assert app._reject_unsafe_pick_command("PICK_CYL 0.00 300.00") is True
    assert app.sent_commands == []
    assert "pick_radius_out_of_limits" in app.statuses[-2][1]
