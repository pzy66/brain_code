from __future__ import annotations

import types

from hybrid_controller.app import HybridControllerApplication
from hybrid_controller.config import AppConfig


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
    app.config = AppConfig(vision_servo_max_attempts=3)
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

    def _send(self, command: str) -> None:
        self.sent_commands.append(str(command))

    def _status(self, source: str, message: str) -> None:
        self.statuses.append((str(source), str(message)))

    app._send_robot_text_command = types.MethodType(_send, app)
    app._handle_runtime_status = types.MethodType(_status, app)
    return app


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


def test_delta_servo_resolution_rejects_stale_robot_snapshot() -> None:
    config = AppConfig(vision_snapshot_max_age_ms=200.0)
    app = _make_app_stub(config=config, snapshot=_snapshot((0.0, -120.0)), snapshot_age_ms=500.0)

    resolved = app._resolve_vision_packet(_packet(mapping_mode="delta_servo"))
    slot = resolved["slots"][0]

    assert slot["actionable"] is False
    assert slot["command_point"] is None
    assert slot["invalid_reason"] == "robot_snapshot_stale"
    assert app.runtime_info["vision_invalid_reason"] == "robot_snapshot_stale"


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
    assert app.sent_commands == ["MOVE_CYL 30.00 150.00 190.00"]
    assert app._vision_servo_pick["attempts"] == 1
    assert app._vision_servo_pick["stage"] == "search"


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

    assert app.sent_commands == ["MOVE_CYL 12.00 175.00 130.00"]
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
