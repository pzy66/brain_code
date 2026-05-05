from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_controller.config import AppConfig
from hybrid_controller.cylindrical import cartesian_to_cylindrical
from hybrid_controller.vision.servo_controller import VisionServoController
from hybrid_controller.vision.target_resolver import resolve_vision_packet


def _load_json(path: Path | None) -> object:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_bundle_debug(bundle_dir: Path | None) -> dict[str, object]:
    if bundle_dir is None:
        return {}
    debug_path = bundle_dir / "debug.json"
    if not debug_path.exists():
        raise FileNotFoundError(f"Missing debug.json in {bundle_dir}")
    payload = _load_json(debug_path)
    if not isinstance(payload, dict):
        raise ValueError("debug.json must contain a JSON object.")
    return payload


def _as_config(debug: Mapping[str, object]) -> AppConfig:
    runtime = debug.get("runtime")
    runtime_map = runtime if isinstance(runtime, Mapping) else {}
    return AppConfig(
        pick_tool_offset_source=str(runtime_map.get("pick_tool_offset_source", AppConfig.pick_tool_offset_source)),
        pick_cyl_radius_bias_mm=float(runtime_map.get("pick_cyl_radius_bias_mm", AppConfig.pick_cyl_radius_bias_mm)),
        pick_cyl_tangent_bias_mm=float(runtime_map.get("pick_cyl_tangent_bias_mm", AppConfig.pick_cyl_tangent_bias_mm)),
        pick_cyl_theta_bias_deg=float(runtime_map.get("pick_cyl_theta_bias_deg", AppConfig.pick_cyl_theta_bias_deg)),
    ).resolved()


def _first_slot_id(packet: Mapping[str, object]) -> int:
    slots = packet.get("slots")
    if isinstance(slots, list):
        for slot in slots:
            if not isinstance(slot, Mapping):
                continue
            try:
                return int(slot.get("slot_id", slot.get("slot", 1)))
            except (TypeError, ValueError):
                continue
    return 1


def _slot_payload(packet: Mapping[str, object], slot_id: int) -> dict[str, object] | None:
    slots = packet.get("slots")
    if not isinstance(slots, list):
        return None
    for slot in slots:
        if not isinstance(slot, Mapping):
            continue
        try:
            current_slot_id = int(slot.get("slot_id", slot.get("slot", -1)))
        except (TypeError, ValueError):
            continue
        if current_slot_id == int(slot_id):
            return dict(slot)
    return None


def _snapshot_from_debug(debug: Mapping[str, object], explicit_snapshot: object | None) -> dict[str, object] | None:
    if isinstance(explicit_snapshot, dict):
        return dict(explicit_snapshot)
    trace = debug.get("trace")
    trace_map = trace if isinstance(trace, Mapping) else {}
    robot_xy = trace_map.get("robot_xy")
    snapshot: dict[str, object] = {}
    if isinstance(robot_xy, (list, tuple)) and len(robot_xy) >= 2:
        try:
            x_mm = float(robot_xy[0])
            y_mm = float(robot_xy[1])
            snapshot["robot_xy"] = [x_mm, y_mm]
        except (TypeError, ValueError):
            pass
    robot_pose = trace_map.get("robot_pose")
    if isinstance(robot_pose, Mapping):
        snapshot["robot_cyl"] = dict(robot_pose)
        if "z_mm" in robot_pose:
            snapshot["robot_z"] = robot_pose.get("z_mm")
    elif isinstance(snapshot.get("robot_xy"), list):
        try:
            x_mm, y_mm = snapshot["robot_xy"]  # type: ignore[misc]
            theta_deg, radius_mm, z_mm = cartesian_to_cylindrical(float(x_mm), float(y_mm), float(snapshot.get("robot_z", 0.0)))
            snapshot["robot_cyl"] = {"theta_deg": theta_deg, "radius_mm": radius_mm, "z_mm": z_mm}
        except (TypeError, ValueError):
            pass
    return snapshot or None


def _current_cyl_pose(snapshot: Mapping[str, object] | None) -> tuple[float, float, float] | None:
    if not isinstance(snapshot, Mapping):
        return None
    cyl = snapshot.get("robot_cyl")
    if isinstance(cyl, Mapping):
        try:
            return (float(cyl.get("theta_deg")), float(cyl.get("radius_mm")), float(cyl.get("z_mm", snapshot.get("robot_z", 0.0))))
        except (TypeError, ValueError):
            return None
    xy = snapshot.get("robot_xy")
    if isinstance(xy, (list, tuple)) and len(xy) >= 2:
        try:
            z_mm = float(snapshot.get("robot_z", 0.0))
            return cartesian_to_cylindrical(float(xy[0]), float(xy[1]), z_mm)
        except (TypeError, ValueError):
            return None
    return None


def _is_at_confirm_z(config: AppConfig, pose: tuple[float, float, float] | None) -> bool:
    if pose is None:
        return False
    tolerance = max(0.5, float(config.vision_pick_z_tolerance_mm))
    return abs(float(pose[2]) - float(config.vision_pick_confirm_z_mm)) <= tolerance


def replay(
    *,
    debug: dict[str, object],
    packet_override: object | None = None,
    snapshot_override: object | None = None,
    slot_id: int | None = None,
    snapshot_age_ms: float = 0.0,
    frame_pose_age_ms: float | None = None,
) -> dict[str, object]:
    packet = packet_override if isinstance(packet_override, dict) else debug.get("packet")
    if not isinstance(packet, dict):
        raise ValueError("Replay requires a packet JSON object.")
    config = _as_config(debug)
    snapshot = _snapshot_from_debug(debug, snapshot_override)
    resolved = resolve_vision_packet(
        packet,
        config=config,
        snapshot=snapshot,
        snapshot_age_ms=float(snapshot_age_ms),
        frame_pose_age_ms=frame_pose_age_ms,
    )
    resolved_packet = resolved.packet
    selected_slot_id = int(slot_id if slot_id is not None else _first_slot_id(resolved_packet))
    slot = _slot_payload(resolved_packet, selected_slot_id)
    trace = debug.get("trace")
    trace_map = trace if isinstance(trace, Mapping) else {}
    pending = trace_map.get("vision_servo_pick")
    pose = _current_cyl_pose(snapshot)
    decision = VisionServoController(config).decide(
        slot_id=selected_slot_id,
        slot_payload=slot,
        packet=resolved_packet,
        pending=pending if isinstance(pending, Mapping) else None,
        current_cyl_pose=pose,
        at_confirm_z=_is_at_confirm_z(config, pose),
        eye_in_hand_enabled=bool(config.vision_eye_in_hand_pick_flow_enabled),
    )
    return {
        "slot_id": selected_slot_id,
        "resolution": {
            "mapping_mode": resolved.mapping_mode,
            "first_invalid_reason": resolved.first_invalid_reason,
            "first_resolved_base_xy": resolved.first_resolved_base_xy,
            "first_resolved_cyl": resolved.first_resolved_cyl,
        },
        "slot": slot,
        "decision": {
            "action": decision.action,
            "state": decision.state,
            "status": decision.status,
            "reason": decision.reason,
            "command": decision.command,
            "pending": decision.pending_dict,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a saved vision debug bundle through resolver and servo state machine.")
    parser.add_argument("bundle", type=Path, nargs="?", default=None, help="Debug bundle directory containing debug.json.")
    parser.add_argument("--packet", type=Path, default=None, help="Optional packet JSON override.")
    parser.add_argument("--snapshot", type=Path, default=None, help="Optional robot snapshot JSON override.")
    parser.add_argument("--slot-id", type=int, default=None)
    parser.add_argument("--snapshot-age-ms", type=float, default=0.0)
    parser.add_argument("--frame-pose-age-ms", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    debug = _load_bundle_debug(args.bundle)
    result = replay(
        debug=debug,
        packet_override=_load_json(args.packet),
        snapshot_override=_load_json(args.snapshot),
        slot_id=args.slot_id,
        snapshot_age_ms=max(0.0, float(args.snapshot_age_ms)),
        frame_pose_age_ms=None if args.frame_pose_age_ms is None or not math.isfinite(float(args.frame_pose_age_ms)) else float(args.frame_pose_age_ms),
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
