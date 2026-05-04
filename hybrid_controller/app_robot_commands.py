from __future__ import annotations

import math

from hybrid_controller.cylindrical import cartesian_to_cylindrical, cylindrical_to_cartesian


def extract_command_opcode(command: str) -> str:
    parts = str(command or "").strip().split()
    if not parts:
        return ""
    return str(parts[0]).upper()


def ros_command_requires_ros_route(opcode: str) -> bool:
    return str(opcode or "").upper() in {
        "MOVE_CYL",
        "MOVE_CYL_AUTO",
        "PICK_WORLD",
        "PICK_CYL",
        "SET_SUCKER_ROTATION",
        "PLACE",
        "ABORT",
        "RESET",
        "SUCKER_OFF",
    }


def rewrite_pick_command_with_bias(
    command: str,
    *,
    theta_bias_deg: float,
    radius_bias_mm: float,
    tangent_bias_mm: float = 0.0,
    pick_z_mm: float,
) -> str:
    text = str(command or "").strip()
    parts = text.split()
    if len(parts) not in {3, 4}:
        return text
    opcode = extract_command_opcode(text)
    if opcode not in {"PICK_CYL", "PICK_WORLD"}:
        return text

    try:
        raw_a = float(parts[1])
        raw_b = float(parts[2])
        angle_suffix = "" if len(parts) < 4 else " {0:.2f}".format(float(parts[3]))
    except (TypeError, ValueError):
        return text

    if (
        abs(float(theta_bias_deg)) < 1e-6
        and abs(float(radius_bias_mm)) < 1e-6
        and abs(float(tangent_bias_mm)) < 1e-6
    ):
        return text

    if opcode == "PICK_CYL":
        adjusted_theta_deg, adjusted_radius_mm, _, _ = apply_pick_tool_bias(
            theta_deg=float(raw_a),
            radius_mm=float(raw_b),
            theta_bias_deg=float(theta_bias_deg),
            radius_bias_mm=float(radius_bias_mm),
            tangent_bias_mm=float(tangent_bias_mm),
            pick_z_mm=float(pick_z_mm),
        )
        return "PICK_CYL {0:.2f} {1:.2f}{2}".format(
            float(adjusted_theta_deg),
            float(adjusted_radius_mm),
            angle_suffix,
        )

    theta_deg, radius_mm, _ = cartesian_to_cylindrical(float(raw_a), float(raw_b), float(pick_z_mm))
    _, _, adjusted_x_mm, adjusted_y_mm = apply_pick_tool_bias(
        theta_deg=float(theta_deg),
        radius_mm=float(radius_mm),
        theta_bias_deg=float(theta_bias_deg),
        radius_bias_mm=float(radius_bias_mm),
        tangent_bias_mm=float(tangent_bias_mm),
        pick_z_mm=float(pick_z_mm),
    )
    return "PICK_WORLD {0:.2f} {1:.2f}{2}".format(float(adjusted_x_mm), float(adjusted_y_mm), angle_suffix)


def apply_pick_tool_bias(
    *,
    theta_deg: float,
    radius_mm: float,
    theta_bias_deg: float,
    radius_bias_mm: float,
    tangent_bias_mm: float,
    pick_z_mm: float,
) -> tuple[float, float, float, float]:
    adjusted_theta_deg = float(theta_deg) + float(theta_bias_deg)
    adjusted_radius_mm = float(radius_mm) + float(radius_bias_mm)
    adjusted_x_mm, adjusted_y_mm, _ = cylindrical_to_cartesian(
        adjusted_theta_deg,
        adjusted_radius_mm,
        float(pick_z_mm),
    )
    if abs(float(tangent_bias_mm)) >= 1e-6:
        theta_rad = math.radians(adjusted_theta_deg)
        adjusted_x_mm += -math.cos(theta_rad) * float(tangent_bias_mm)
        adjusted_y_mm += math.sin(theta_rad) * float(tangent_bias_mm)
        adjusted_theta_deg, adjusted_radius_mm, _ = cartesian_to_cylindrical(
            adjusted_x_mm,
            adjusted_y_mm,
            float(pick_z_mm),
        )
    return (
        float(adjusted_theta_deg),
        float(adjusted_radius_mm),
        float(adjusted_x_mm),
        float(adjusted_y_mm),
    )


def _coerce_optional_angle_deg(value: object, *, min_quality: float = 0.0, quality: object = None) -> float | None:
    if value is None:
        return None
    try:
        if quality is not None and float(quality) < float(min_quality):
            return None
        angle = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(angle):
        return None
    return angle


def build_pick_command_from_mode_and_point(
    mode: str,
    point: object,
    *,
    sucker_angle_deg: object = None,
    sucker_angle_quality: object = None,
    sucker_angle_quality_threshold: float = 0.0,
) -> str | None:
    if not isinstance(point, (tuple, list)) or len(point) < 2:
        return None
    try:
        x_value = float(point[0])
        y_value = float(point[1])
    except (TypeError, ValueError):
        return None
    angle = _coerce_optional_angle_deg(
        sucker_angle_deg,
        min_quality=float(sucker_angle_quality_threshold),
        quality=sucker_angle_quality,
    )
    suffix = "" if angle is None else f" {angle:.2f}"
    mode_text = str(mode or "").strip().lower()
    if mode_text == "cyl":
        return f"PICK_CYL {x_value:.2f} {y_value:.2f}{suffix}"
    if mode_text == "world":
        return f"PICK_WORLD {x_value:.2f} {y_value:.2f}{suffix}"
    if mode_text in {"pixel", "px"}:
        return f"PICK {x_value:.2f} {y_value:.2f}"
    return None


def build_pick_command_from_slot_payload(slot: dict[str, object]) -> str | None:
    if not bool(slot.get("actionable", True)):
        return None
    threshold_raw = slot.get("grasp_angle_quality_threshold")
    try:
        angle_quality_threshold = 0.20 if threshold_raw is None else float(threshold_raw)
    except (TypeError, ValueError):
        angle_quality_threshold = 0.20
    return build_pick_command_from_mode_and_point(
        str(slot.get("command_mode", "world")),
        slot.get("command_point"),
        sucker_angle_deg=slot.get("grasp_angle_deg"),
        sucker_angle_quality=slot.get("grasp_angle_quality"),
        sucker_angle_quality_threshold=angle_quality_threshold,
    )


def build_pick_command_from_target(target: object) -> str | None:
    if not bool(getattr(target, "actionable", True)):
        return None
    return build_pick_command_from_mode_and_point(
        str(getattr(target, "command_mode", "world")),
        getattr(target, "command_point", None),
        sucker_angle_deg=getattr(target, "grasp_angle_deg", None),
        sucker_angle_quality=getattr(target, "grasp_angle_quality", None),
        sucker_angle_quality_threshold=0.20,
    )


def build_catalog_pick_command(slot: object) -> str | None:
    cylindrical_trz = getattr(slot, "cylindrical_trz", None)
    if isinstance(cylindrical_trz, (tuple, list)) and len(cylindrical_trz) >= 2:
        return f"PICK_CYL {float(cylindrical_trz[0]):.2f} {float(cylindrical_trz[1]):.2f}"
    world_xy = getattr(slot, "world_xy", None)
    if isinstance(world_xy, (tuple, list)) and len(world_xy) >= 2:
        return f"PICK_WORLD {float(world_xy[0]):.2f} {float(world_xy[1]):.2f}"
    return None
