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
