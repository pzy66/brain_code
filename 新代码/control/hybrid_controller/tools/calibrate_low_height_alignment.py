from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_controller.config import AppConfig
from hybrid_controller.config import SERVO_MEASUREMENT_POINTS
from hybrid_controller.cylindrical import cylindrical_to_cartesian
from hybrid_controller.tools.debug_vision_grasp_flow import (
    RosBridgeClient,
    _current_cyl_pose,
    _frame_pose_age_for_static_snapshot,
    _load_model,
    _low_height_local_synthetic_slot,
    _patch_low_height_local_center,
    _point_for_measurement,
    _process_frame_batch,
    _resolve_device,
    _resolve_packet,
    _select_latest_frames,
    _select_slot,
    _slot_alignment_provenance,
    _state_message_to_snapshot,
    _PersistentCaptureReader,
    _upsert_packet_slot,
)
from hybrid_controller.vision.calibration_profile import VisionCalibrationProfile
from hybrid_controller.vision.grasp_profile import apply_vision_grasp_profile
from hybrid_controller.vision.grasp_profile import load_vision_grasp_profile
from hybrid_controller.vision.low_height_alignment import fit_low_height_response_model
from hybrid_controller.vision.low_height_alignment import merge_confirm_stage_model


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _sample_alignment_target(samples: list[object]) -> tuple[float, float] | None:
    targets: list[tuple[float, float]] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        target = sample.get("alignment_target_pixel")
        if not isinstance(target, Sequence) or isinstance(target, (str, bytes)) or len(target) < 2:
            continue
        try:
            point = (float(target[0]), float(target[1]))
        except (TypeError, ValueError):
            continue
        if math.isfinite(point[0]) and math.isfinite(point[1]):
            targets.append(point)
    if not targets:
        return None
    reference = targets[0]
    for point in targets[1:]:
        if math.hypot(point[0] - reference[0], point[1] - reference[1]) > 0.5:
            raise ValueError("alignment_target_pixel changed across low-height calibration samples")
    return reference


def _parse_float_list(value: str) -> list[float]:
    result: list[float] = []
    for item in str(value or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        number = float(item)
        if not math.isfinite(number):
            raise ValueError("offset lists must contain finite numbers")
        result.append(number)
    if not result:
        raise ValueError("offset list must not be empty")
    return result


def _state_is_safe(snapshot: Mapping[str, object]) -> bool:
    state = str(snapshot.get("state", "")).strip().upper()
    if state != "IDLE":
        return False
    if bool(snapshot.get("busy", False)):
        return False
    if bool(snapshot.get("carrying", False)):
        return False
    if str(snapshot.get("last_error_code", "")).strip():
        return False
    return True


def _wait_for_idle(client: RosBridgeClient, *, timeout_sec: float) -> dict[str, object]:
    deadline = time.perf_counter() + max(0.5, float(timeout_sec))
    last_snapshot: dict[str, object] | None = None
    while time.perf_counter() <= deadline:
        message = client.fetch_state(timeout_sec=min(1.0, max(0.2, timeout_sec)))
        snapshot = _state_message_to_snapshot(message)
        last_snapshot = snapshot
        if _state_is_safe(snapshot):
            return snapshot
        time.sleep(0.15)
    raise TimeoutError(f"Timed out waiting for safe IDLE state. last={last_snapshot}")


def _settle_before_low_height_measurement(
    client: RosBridgeClient,
    *,
    timeout_sec: float,
    settle_sec: float,
) -> dict[str, object]:
    _wait_for_idle(client, timeout_sec=float(timeout_sec))
    time.sleep(max(0.0, float(settle_sec)))
    return _wait_for_idle(client, timeout_sec=float(timeout_sec))


def _move_cyl(client: RosBridgeClient, *, theta: float, radius: float, z: float, timeout_sec: float) -> dict[str, object]:
    return client.call_service(
        "/hybrid_controller/move_cyl",
        "hybrid_controller_ros/MoveCyl",
        {"theta_deg": float(theta), "radius_mm": float(radius), "z_mm": float(z)},
        timeout_sec=max(1.0, float(timeout_sec)),
    )


def _freeze_sucker(client: RosBridgeClient, *, timeout_sec: float) -> dict[str, object]:
    return client.call_service(
        "/hybrid_controller/sucker_freeze",
        "std_srvs/SetBool",
        {"data": True},
        timeout_sec=max(1.0, float(timeout_sec)),
    )


def _measure_slot(
    *,
    reader: _PersistentCaptureReader,
    cv2_module: object,
    model: object | None,
    config: AppConfig,
    calibration_profile: VisionCalibrationProfile,
    client: RosBridgeClient,
    device: str | None,
    half: bool,
    slot_id: int,
    frame_id: int,
    frames: int,
    drain_frames: int,
    timeout_sec: float,
    ros_timeout_sec: float,
    settle_sec: float = 0.0,
    debug_slots,
):
    snapshot = _settle_before_low_height_measurement(
        client,
        timeout_sec=float(ros_timeout_sec),
        settle_sec=float(settle_sec),
    )
    snapshot_age_ms = 0.0
    _, captured = reader.read(frame_count=int(frames), drain_frames=int(drain_frames), timeout_sec=float(timeout_sec))
    capture_stats = reader.transport_stats()
    # Low-height stop-settle measurements must represent the current stopped
    # pose only. Reusing multiple buffered frames or slot history can smear the
    # center across previous poses and make the search chase stale geometry.
    process_frames = _select_latest_frames(captured, 1)
    packet, last_frame, next_frame_id, next_debug_slots = _process_frame_batch(
        frames=process_frames,
        model=model,
        config=config,
        calibration_profile=calibration_profile,
        snapshot_for_stage=snapshot,
        frame_id_start=frame_id,
        slots=None,
        device=device,
        half=half,
    )
    resolved = _resolve_packet(
        packet=packet,
        config=config,
        snapshot=snapshot,
        snapshot_age_ms=snapshot_age_ms,
        frame_pose_age_ms=_frame_pose_age_for_static_snapshot(snapshot, packet),
    )
    slot = _select_slot(resolved, int(slot_id))
    pose_for_patch = _current_cyl_pose(snapshot)
    measurement_point = str(config.vision_servo_low_height_measurement_point or config.vision_servo_measurement_point)
    slot = _patch_low_height_local_center(
        packet=resolved,
        frame_bgr=last_frame,
        selected_slot=slot,
        pending=None,
        current_z_mm=None if pose_for_patch is None else float(pose_for_patch[2]),
        confirm_z_mm=float(config.vision_pick_confirm_z_mm),
        measurement_point=measurement_point,
    )
    if slot is None:
        slot = _low_height_local_synthetic_slot(
            packet=resolved,
            frame_bgr=last_frame,
            pending=None,
            current_z_mm=None if pose_for_patch is None else float(pose_for_patch[2]),
            confirm_z_mm=float(config.vision_pick_confirm_z_mm),
            measurement_point=measurement_point,
            slot_id=int(slot_id),
        )
    _upsert_packet_slot(resolved, slot)
    if slot is None or not bool(slot.get("valid", False)):
        raise RuntimeError(f"Slot {slot_id} not detected in low-height calibration frame.")
    alignment_provenance = _slot_alignment_provenance(slot, resolved)
    point = _point_for_measurement(slot, alignment_provenance["measurement_point"])
    if not isinstance(point, (tuple, list)) or len(point) < 2:
        raise RuntimeError("Detected slot is missing geometry/pixel center.")
    pose = _current_cyl_pose(snapshot)
    if pose is None:
        raise RuntimeError("Robot pose unavailable after calibration measurement.")
    x_mm, y_mm, _ = cylindrical_to_cartesian(float(pose[0]), float(pose[1]), float(pose[2]))
    sample = {
        "slot_id": int(slot_id),
        "frame_id": int(next_frame_id),
        "pose_cyl": [float(pose[0]), float(pose[1]), float(pose[2])],
        "pose_xy": [float(x_mm), float(y_mm)],
        "pixel": [float(point[0]), float(point[1])],
        "measurement_point": alignment_provenance["measurement_point"],
        "alignment_target_pixel": alignment_provenance["alignment_target_pixel"],
        "point_distances_px": alignment_provenance["point_distances_px"],
        "pixel_center": slot.get("pixel_center"),
        "pixel_center_f": slot.get("pixel_center_f"),
        "geometry_center": slot.get("geometry_center"),
        "geometry_center_f": slot.get("geometry_center_f"),
        "color_block_center": slot.get("color_block_center"),
        "color_block_center_f": slot.get("color_block_center_f"),
        "top_face_center": slot.get("top_face_center"),
        "top_face_center_f": slot.get("top_face_center_f"),
        "grasp_pixel": slot.get("grasp_pixel"),
        "grasp_pixel_f": slot.get("grasp_pixel_f"),
        "center_distance_px": slot.get("center_distance_px"),
        "confidence": slot.get("confidence"),
        "area_px": slot.get("area_px"),
        "bbox": slot.get("bbox"),
        "snapshot": snapshot,
        "packet_frame_id": packet.get("frame_id"),
        "settle_sec": float(settle_sec),
        "drain_frames": int(drain_frames),
        "captured_frames": int(len(captured)),
        "processed_frames": int(len(process_frames)),
        "slot_history_reused": False,
        "camera_transport": capture_stats,
    }
    return sample, resolved, last_frame, next_frame_id, next_debug_slots


def build_parser() -> argparse.ArgumentParser:
    defaults = AppConfig().resolved()
    parser = argparse.ArgumentParser(
        description=(
            "Safely measure a local low-height visual response model for JetMax center alignment. "
            "This tool only reads the official MJPEG stream and ROS control services; it never "
            "starts, restarts, repairs, or scans the robot camera sender."
        )
    )
    parser.add_argument("--host", default=defaults.robot_host)
    parser.add_argument("--ros-port", type=int, default=defaults.rosbridge_port)
    parser.add_argument("--slot-id", type=int, required=True)
    parser.add_argument("--z-mm", type=float, default=120.0)
    parser.add_argument("--theta-offsets-deg", default="-0.45,0,0.45")
    parser.add_argument("--radius-offsets-mm", default="-1.5,0,1.5")
    parser.add_argument("--max-theta-offset-deg", type=float, default=0.8)
    parser.add_argument("--max-radius-offset-mm", type=float, default=2.5)
    parser.add_argument("--settle-sec", type=float, default=0.45)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--drain-frames", type=int, default=8)
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--ros-timeout-sec", type=float, default=3.0)
    parser.add_argument("--command-timeout-sec", type=float, default=12.0)
    parser.add_argument(
        "--low-height-measurement-point",
        choices=tuple(sorted(SERVO_MEASUREMENT_POINTS)),
        default=None,
        help="Temporary override for the stopped low-height visual point used by this calibration.",
    )
    parser.add_argument("--detector", choices=("auto", "yolo", "fallback"), default="auto")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--device", default=str(defaults.vision_device))
    parser.add_argument("--half", action="store_true", default=bool(defaults.vision_half))
    parser.add_argument("--profile", type=Path, default=defaults.vision_calibration_profile_path)
    parser.add_argument("--vision-grasp-profile", type=Path, default=defaults.vision_grasp_profile_path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--write-profile", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Capture current pose only; do not move or write profile.")
    parser.add_argument(
        "--fresh-reopen-before-measure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reopen only the PC-side MJPEG consumer before each settled measurement to discard buffered old frames.",
    )
    parser.add_argument("--max-target-delta-mm", type=float, default=12.0)
    parser.add_argument("--max-fit-rms-px", type=float, default=1.5)
    parser.add_argument("--max-fit-max-error-px", type=float, default=3.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    theta_offsets = _parse_float_list(str(args.theta_offsets_deg))
    radius_offsets = _parse_float_list(str(args.radius_offsets_mm))
    if any(abs(value) > float(args.max_theta_offset_deg) for value in theta_offsets):
        print("[guard] theta offset exceeds --max-theta-offset-deg", file=sys.stderr)
        return 2
    if any(abs(value) > float(args.max_radius_offset_mm) for value in radius_offsets):
        print("[guard] radius offset exceeds --max-radius-offset-mm", file=sys.stderr)
        return 2

    config = AppConfig(
        robot_host=str(args.host),
        vision_calibration_profile_path=Path(args.profile),
        vision_grasp_profile_path=Path(args.vision_grasp_profile),
        vision_pick_confirm_z_mm=float(args.z_mm),
    ).resolved()
    grasp_profile = load_vision_grasp_profile(config)
    if grasp_profile.ready:
        config = apply_vision_grasp_profile(config, grasp_profile).resolved()
        config = replace(config, vision_pick_confirm_z_mm=float(args.z_mm)).resolved()
    if args.low_height_measurement_point is not None:
        config = replace(
            config,
            vision_servo_low_height_measurement_point=str(args.low_height_measurement_point),
        ).resolved()
    calibration_profile = VisionCalibrationProfile.load(Path(args.profile))
    output_dir = Path(args.output_dir) if args.output_dir is not None else config.vision_debug_bundle_dir / f"low_height_alignment_{_timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    import cv2

    device, auto_half = _resolve_device(str(args.device))
    half = bool(args.half or auto_half)
    model = _load_model(args, config)
    stream_url = config.resolve_vision_stream_url()
    report: dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "camera_contract": (
            "PC reads the single official JetMax MJPEG URL only; this tool does not start, restart, "
            "scan, repair, or mutate usb_cam.service/web_video_server/uvcvideo/devices."
        ),
        "measurement_contract": (
            "Each low-height sample is taken after robot IDLE, a settle delay, and explicit drain frames "
            "from the persistent official MJPEG stream."
        ),
        "stream_url": stream_url,
        "slot_id": int(args.slot_id),
        "z_mm": float(args.z_mm),
        "measurement_point": str(
            config.vision_servo_low_height_measurement_point or config.vision_servo_measurement_point
        ),
        "alignment_target_pixel": None,
        "alignment_target_source": "pending_first_sample",
        "final_center_tolerance_px": 2.0,
        "settle_sec": float(args.settle_sec),
        "drain_frames": int(args.drain_frames),
        "fresh_reopen_before_measure": bool(args.fresh_reopen_before_measure),
        "theta_offsets_deg": theta_offsets,
        "radius_offsets_mm": radius_offsets,
        "dry_run": bool(args.dry_run),
        "write_profile": bool(args.write_profile),
        "samples": [],
    }
    client = RosBridgeClient(host=str(args.host), port=int(args.ros_port), timeout_sec=float(args.ros_timeout_sec))
    reader: _PersistentCaptureReader | None = None
    frame_id = 0
    debug_slots = None
    center_theta: float | None = None
    center_radius: float | None = None
    return_pose_needed = False
    try:
        client.connect()
        _freeze_sucker(client, timeout_sec=float(args.ros_timeout_sec))
        initial = _wait_for_idle(client, timeout_sec=float(args.command_timeout_sec))
        start_pose = _current_cyl_pose(initial)
        if start_pose is None:
            raise RuntimeError("Initial robot cylindrical pose unavailable.")
        center_theta, center_radius, _ = start_pose
        return_pose_needed = not bool(args.dry_run)
        report["start_pose_cyl"] = [float(center_theta), float(center_radius), float(start_pose[2])]
        reader = _PersistentCaptureReader(cv2_module=cv2, stream_urls=(stream_url,), config=config, capture_backend="http")
        offsets = [(0.0, 0.0)]
        if not bool(args.dry_run):
            offsets.extend(
                (float(theta_offset), float(radius_offset))
                for theta_offset in theta_offsets
                for radius_offset in radius_offsets
                if abs(float(theta_offset)) > 1e-9 or abs(float(radius_offset)) > 1e-9
            )
        for index, (theta_offset, radius_offset) in enumerate(offsets, start=1):
            target_theta = float(center_theta) + float(theta_offset)
            target_radius = float(center_radius) + float(radius_offset)
            target_z = float(args.z_mm)
            if not bool(args.dry_run):
                response = _move_cyl(
                    client,
                    theta=target_theta,
                    radius=target_radius,
                    z=target_z,
                    timeout_sec=float(args.command_timeout_sec),
                )
                report.setdefault("moves", []).append(
                    {
                        "index": index,
                        "target_cyl": [target_theta, target_radius, target_z],
                        "response": response,
                    }
                )
                _wait_for_idle(client, timeout_sec=float(args.command_timeout_sec))
            if bool(args.fresh_reopen_before_measure) and reader is not None:
                reader.reopen()
            sample, packet, frame, frame_id, debug_slots = _measure_slot(
                reader=reader,
                cv2_module=cv2,
                model=model,
                config=config,
                calibration_profile=calibration_profile,
                client=client,
                device=device,
                half=half,
                slot_id=int(args.slot_id),
                frame_id=frame_id,
                frames=int(args.frames),
                drain_frames=int(args.drain_frames),
                timeout_sec=float(args.timeout_sec),
                ros_timeout_sec=float(args.ros_timeout_sec),
                settle_sec=float(args.settle_sec),
                debug_slots=None,
            )
            step_dir = output_dir / f"sample_{index:02d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(step_dir / "raw.jpg"), frame)
            _write_json(step_dir / "packet.json", packet)
            sample["raw_image"] = str(step_dir / "raw.jpg")
            sample["packet"] = str(step_dir / "packet.json")
            report["samples"].append(sample)
            if report.get("alignment_target_pixel") is None:
                target = sample.get("alignment_target_pixel")
                if isinstance(target, (list, tuple)) and len(target) >= 2:
                    try:
                        report["alignment_target_pixel"] = [float(target[0]), float(target[1])]
                        report["alignment_target_source"] = "sample_provenance"
                    except (TypeError, ValueError):
                        pass
            print(
                "[sample {index}] pose=({theta:.2f},{radius:.2f},{z:.2f}) pixel=({x:.1f},{y:.1f})".format(
                    index=index,
                    theta=float(sample["pose_cyl"][0]),
                    radius=float(sample["pose_cyl"][1]),
                    z=float(sample["pose_cyl"][2]),
                    x=float(sample["pixel"][0]),
                    y=float(sample["pixel"][1]),
                )
            )
        if not bool(args.dry_run):
            _move_cyl(
                client,
                theta=float(center_theta),
                radius=float(center_radius),
                z=float(args.z_mm),
                timeout_sec=float(args.command_timeout_sec),
            )
            _wait_for_idle(client, timeout_sec=float(args.command_timeout_sec))
            return_pose_needed = False
        if len(report["samples"]) >= 4:
            target_pixel = _sample_alignment_target(report["samples"])
            if target_pixel is None:
                target_pixel = (320.0, 240.0)
                report["alignment_target_pixel"] = [320.0, 240.0]
                report["alignment_target_source"] = "fallback_frame_center"
            model_result = fit_low_height_response_model(
                report["samples"],
                target_pixel=target_pixel,
                z_mm=float(args.z_mm),
                min_samples=4,
            )
            stage_payload = model_result.to_stage_model_payload(
                profile_id=f"{calibration_profile.profile_id}-confirm-local-{_timestamp()}",
                image_size=calibration_profile.image_size or (640, 480),
                center_tolerance_px=2.0,
                servo_gain=0.45,
                max_attempts=4,
            )
            report["fitted_model"] = stage_payload
            start_x, start_y, _ = cylindrical_to_cartesian(float(center_theta), float(center_radius), float(args.z_mm))
            target_delta_mm = math.hypot(
                float(model_result.target_robot_xy_mm[0]) - float(start_x),
                float(model_result.target_robot_xy_mm[1]) - float(start_y),
            )
            quality_guard = {
                "target_delta_mm": float(target_delta_mm),
                "max_target_delta_mm": float(args.max_target_delta_mm),
                "rms_pixel_error_px": float(model_result.rms_pixel_error_px),
                "max_fit_rms_px": float(args.max_fit_rms_px),
                "max_pixel_error_px": float(model_result.max_pixel_error_px),
                "max_fit_max_error_px": float(args.max_fit_max_error_px),
                "condition_number": float(model_result.condition_number),
            }
            report["fit_quality_guard"] = quality_guard
            fit_allowed = (
                target_delta_mm <= float(args.max_target_delta_mm)
                and float(model_result.rms_pixel_error_px) <= float(args.max_fit_rms_px)
                and float(model_result.max_pixel_error_px) <= float(args.max_fit_max_error_px)
            )
            if not fit_allowed:
                report["fit_rejected"] = "low_height_alignment_fit_outside_safety_guard"
            if bool(args.write_profile) and fit_allowed:
                profile_payload = _read_json(Path(args.profile))
                merged = merge_confirm_stage_model(profile_payload, stage_payload)
                backup_path = Path(args.profile).with_suffix(Path(args.profile).suffix + f".before_low_height_{_timestamp()}.bak")
                _write_json(backup_path, profile_payload)
                _write_json(Path(args.profile), merged)
                report["profile_backup"] = str(backup_path)
                report["profile_written"] = str(Path(args.profile))
            elif bool(args.write_profile):
                report["profile_write_skipped"] = "fit_quality_guard_failed"
        else:
            report["fit_skipped"] = "not_enough_samples"
        _write_json(output_dir / "low_height_alignment_report.json", report)
        print(f"[output] report: {output_dir / 'low_height_alignment_report.json'}")
        return 0
    except Exception as error:
        report["error"] = str(error)
        _write_json(output_dir / "low_height_alignment_report.json", report)
        print(f"[error] {error}", file=sys.stderr)
        print(f"[output] partial report: {output_dir / 'low_height_alignment_report.json'}", file=sys.stderr)
        return 1
    finally:
        if return_pose_needed and center_theta is not None and center_radius is not None:
            try:
                _move_cyl(
                    client,
                    theta=float(center_theta),
                    radius=float(center_radius),
                    z=float(args.z_mm),
                    timeout_sec=float(args.command_timeout_sec),
                )
                _wait_for_idle(client, timeout_sec=float(args.command_timeout_sec))
                report["returned_to_start_pose"] = [float(center_theta), float(center_radius), float(args.z_mm)]
                _write_json(output_dir / "low_height_alignment_report.json", report)
            except Exception as return_error:
                report["return_to_start_error"] = str(return_error)
                _write_json(output_dir / "low_height_alignment_report.json", report)
        if reader is not None:
            reader.close()
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
