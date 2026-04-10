from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from hybrid_controller.config import AppConfig
from hybrid_controller.cylindrical import cartesian_to_cylindrical


@dataclass(frozen=True, slots=True)
class VisionResolutionResult:
    packet: dict[str, object]
    mapping_mode: str
    first_invalid_reason: str
    first_resolved_base_xy: list[float] | None
    first_resolved_cyl: list[float] | None


def _to_float_pair(value: object, fallback: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            pass
    return (float(fallback[0]), float(fallback[1]))


def _resolve_mapping_mode(raw_mode: object, fallback_mode: str) -> str:
    mode = str(raw_mode if raw_mode is not None else fallback_mode).strip().lower()
    if mode not in {"delta_servo", "absolute_base"}:
        return "delta_servo"
    return mode


def _resolve_robot_xy(snapshot: Mapping[str, object] | None) -> tuple[float, float] | None:
    if not isinstance(snapshot, Mapping):
        return None
    robot_xy_raw = snapshot.get("robot_xy")
    if not isinstance(robot_xy_raw, (tuple, list)) or len(robot_xy_raw) < 2:
        return None
    try:
        return (float(robot_xy_raw[0]), float(robot_xy_raw[1]))
    except (TypeError, ValueError):
        return None


def resolve_vision_packet(
    packet: Mapping[str, object],
    *,
    config: AppConfig,
    snapshot: Mapping[str, object] | None,
    snapshot_age_ms: float,
) -> VisionResolutionResult:
    resolved: dict[str, object] = dict(packet)
    mapping_mode = _resolve_mapping_mode(packet.get("mapping_mode"), config.vision_mapping_mode)
    resolved["mapping_mode"] = mapping_mode

    robot_xy = _resolve_robot_xy(snapshot)
    cyl_limits = snapshot.get("limits_cyl") if isinstance(snapshot, Mapping) else None
    theta_limits = _to_float_pair(
        cyl_limits.get("theta_deg") if isinstance(cyl_limits, Mapping) else None,
        config.robot_theta_limits_deg,
    )
    radius_limits = _to_float_pair(
        cyl_limits.get("radius_mm") if isinstance(cyl_limits, Mapping) else None,
        config.robot_radius_limits_mm,
    )

    slots_raw = packet.get("slots", [])
    slots_resolved: list[dict[str, object]] = []
    first_invalid_reason = "--"
    first_resolved_base_xy: list[float] | None = None
    first_resolved_cyl: list[float] | None = None
    calibration_ready = bool(packet.get("calibration_ready", False))
    snapshot_max_age_ms = float(config.vision_snapshot_max_age_ms)
    requires_calibration = bool(config.vision_action_requires_calibration)
    pick_z = float(config.robot_pick_z)

    for slot in slots_raw if isinstance(slots_raw, list) else []:
        if not isinstance(slot, Mapping):
            continue
        slot_resolved: dict[str, object] = dict(slot)
        slot_resolved["mapping_mode"] = mapping_mode
        slot_resolved["command_mode"] = "world"
        slot_resolved["command_point"] = None
        slot_resolved["actionable"] = False
        slot_resolved["invalid_reason"] = ""
        slot_resolved["resolved_base_xy"] = None
        slot_resolved["resolved_cyl"] = None
        slot_resolved["world_xyz"] = None
        slot_resolved["cylindrical_center"] = None

        if not bool(slot.get("valid", False)):
            slots_resolved.append(slot_resolved)
            continue

        if requires_calibration and not calibration_ready:
            slot_resolved["invalid_reason"] = "calibration_unavailable"
            slots_resolved.append(slot_resolved)
            if first_invalid_reason == "--":
                first_invalid_reason = str(slot_resolved["invalid_reason"])
            continue

        raw_delta = slot.get("camera_to_world_raw")
        if not isinstance(raw_delta, (tuple, list)) or len(raw_delta) < 2:
            slot_resolved["invalid_reason"] = "camera_to_world_unavailable"
            slots_resolved.append(slot_resolved)
            if first_invalid_reason == "--":
                first_invalid_reason = str(slot_resolved["invalid_reason"])
            continue
        try:
            delta_x = float(raw_delta[0])
            delta_y = float(raw_delta[1])
            delta_z = float(raw_delta[2]) if len(raw_delta) > 2 else 0.0
        except (TypeError, ValueError):
            slot_resolved["invalid_reason"] = "camera_to_world_invalid"
            slots_resolved.append(slot_resolved)
            if first_invalid_reason == "--":
                first_invalid_reason = str(slot_resolved["invalid_reason"])
            continue

        if mapping_mode == "absolute_base":
            world_xyz = slot.get("world_xyz")
            if not isinstance(world_xyz, (tuple, list)) or len(world_xyz) < 2:
                base_x = delta_x
                base_y = delta_y
            else:
                try:
                    base_x = float(world_xyz[0])
                    base_y = float(world_xyz[1])
                except (TypeError, ValueError):
                    base_x = delta_x
                    base_y = delta_y
            base_z = float(world_xyz[2]) if isinstance(world_xyz, (tuple, list)) and len(world_xyz) >= 3 else delta_z
        else:
            if robot_xy is None:
                slot_resolved["invalid_reason"] = "robot_snapshot_unavailable"
                slots_resolved.append(slot_resolved)
                if first_invalid_reason == "--":
                    first_invalid_reason = str(slot_resolved["invalid_reason"])
                continue
            if not math.isfinite(snapshot_age_ms) or snapshot_age_ms > snapshot_max_age_ms:
                slot_resolved["invalid_reason"] = "robot_snapshot_stale"
                slots_resolved.append(slot_resolved)
                if first_invalid_reason == "--":
                    first_invalid_reason = str(slot_resolved["invalid_reason"])
                continue
            base_x = float(robot_xy[0]) + delta_x
            base_y = float(robot_xy[1]) + delta_y
            base_z = delta_z

        theta_deg, radius_mm, _ = cartesian_to_cylindrical(base_x, base_y, pick_z)
        if (
            theta_deg < theta_limits[0]
            or theta_deg > theta_limits[1]
            or radius_mm < radius_limits[0]
            or radius_mm > radius_limits[1]
        ):
            slot_resolved["invalid_reason"] = "target_out_of_workspace_cyl"
            slots_resolved.append(slot_resolved)
            if first_invalid_reason == "--":
                first_invalid_reason = str(slot_resolved["invalid_reason"])
            continue

        slot_resolved["world_xyz"] = [float(base_x), float(base_y), float(base_z)]
        slot_resolved["cylindrical_center"] = [float(theta_deg), float(radius_mm), float(pick_z)]
        slot_resolved["resolved_base_xy"] = [float(base_x), float(base_y)]
        slot_resolved["resolved_cyl"] = [float(theta_deg), float(radius_mm), float(pick_z)]
        slot_resolved["command_mode"] = "world"
        slot_resolved["command_point"] = [float(base_x), float(base_y)]
        slot_resolved["actionable"] = True

        if first_resolved_base_xy is None:
            first_resolved_base_xy = [float(base_x), float(base_y)]
            first_resolved_cyl = [float(theta_deg), float(radius_mm), float(pick_z)]
        slots_resolved.append(slot_resolved)

    resolved["slots"] = slots_resolved
    return VisionResolutionResult(
        packet=resolved,
        mapping_mode=mapping_mode,
        first_invalid_reason=first_invalid_reason,
        first_resolved_base_xy=first_resolved_base_xy,
        first_resolved_cyl=first_resolved_cyl,
    )
