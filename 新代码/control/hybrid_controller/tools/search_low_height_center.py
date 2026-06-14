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
from hybrid_controller.tools.calibrate_low_height_alignment import (
    _freeze_sucker,
    _measure_slot,
    _move_cyl,
    _parse_float_list,
    _wait_for_idle,
)
from hybrid_controller.tools.debug_vision_grasp_flow import (
    RosBridgeClient,
    _PersistentCaptureReader,
    _current_cyl_pose,
    _load_model,
    _resolve_device,
    _save_overlay,
)
from hybrid_controller.vision.calibration_profile import VisionCalibrationProfile
from hybrid_controller.vision.grasp_profile import apply_vision_grasp_profile
from hybrid_controller.vision.grasp_profile import load_vision_grasp_profile


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def _median(values: Sequence[float]) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return float("inf")
    mid = len(finite) // 2
    if len(finite) % 2:
        return finite[mid]
    return 0.5 * (finite[mid - 1] + finite[mid])


def _point_spread_px(points: Sequence[Sequence[float]]) -> float:
    parsed: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, Sequence) or len(point) < 2:
            continue
        try:
            x_value = float(point[0])
            y_value = float(point[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x_value) and math.isfinite(y_value):
            parsed.append((x_value, y_value))
    if len(parsed) < 2:
        return 0.0
    spread = 0.0
    for left_index, left in enumerate(parsed):
        for right in parsed[left_index + 1 :]:
            spread = max(spread, math.hypot(left[0] - right[0], left[1] - right[1]))
    return float(spread)


def _candidate_offsets(theta_step_deg: float, radius_step_mm: float, *, include_diagonal: bool) -> list[tuple[float, float]]:
    theta_step = abs(float(theta_step_deg))
    radius_step = abs(float(radius_step_mm))
    offsets = [
        (0.0, 0.0),
        (-theta_step, 0.0),
        (theta_step, 0.0),
        (0.0, -radius_step),
        (0.0, radius_step),
    ]
    if include_diagonal:
        offsets.extend(
            [
                (-theta_step, -radius_step),
                (-theta_step, radius_step),
                (theta_step, -radius_step),
                (theta_step, radius_step),
            ]
        )
    return offsets


def _filter_candidate_offsets(
    offsets: Sequence[tuple[float, float]],
    *,
    theta_direction: str,
    radius_direction: str,
) -> list[tuple[float, float]]:
    theta_mode = str(theta_direction or "both").strip().lower()
    radius_mode = str(radius_direction or "both").strip().lower()
    filtered: list[tuple[float, float]] = []
    for theta_offset, radius_offset in offsets:
        if theta_offset < 0 and theta_mode not in {"both", "negative"}:
            continue
        if theta_offset > 0 and theta_mode not in {"both", "positive"}:
            continue
        if radius_offset < 0 and radius_mode not in {"both", "negative"}:
            continue
        if radius_offset > 0 and radius_mode not in {"both", "positive"}:
            continue
        filtered.append((float(theta_offset), float(radius_offset)))
    return filtered


def _within_cyl_limits(snapshot: Mapping[str, object], *, theta: float, radius: float, z: float) -> bool:
    limits = snapshot.get("limits_cyl")
    if not isinstance(limits, Mapping):
        return True

    def _inside(name: str, value: float) -> bool:
        raw = limits.get(name)
        if not isinstance(raw, Sequence) or len(raw) < 2:
            return True
        try:
            lo = float(raw[0])
            hi = float(raw[1])
        except (TypeError, ValueError):
            return True
        return lo <= float(value) <= hi

    return _inside("theta_deg", theta) and _inside("radius_mm", radius) and _inside("z_mm", z)


def _cartesian_distance_between_cyl(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    lx, ly, lz = cylindrical_to_cartesian(float(left[0]), float(left[1]), float(left[2]))
    rx, ry, rz = cylindrical_to_cartesian(float(right[0]), float(right[1]), float(right[2]))
    return math.sqrt((lx - rx) ** 2 + (ly - ry) ** 2 + (lz - rz) ** 2)


def _pose_close(
    actual: tuple[float, float, float] | None,
    target: tuple[float, float, float],
    *,
    theta_tol_deg: float,
    radius_tol_mm: float,
    z_tol_mm: float,
) -> bool:
    if actual is None:
        return False
    return (
        abs(float(actual[0]) - float(target[0])) <= float(theta_tol_deg)
        and abs(float(actual[1]) - float(target[1])) <= float(radius_tol_mm)
        and abs(float(actual[2]) - float(target[2])) <= float(z_tol_mm)
    )


def _move_and_wait_for_pose(
    client: RosBridgeClient,
    *,
    target: tuple[float, float, float],
    command_timeout_sec: float,
    settle_sec: float,
    theta_tol_deg: float = 0.08,
    radius_tol_mm: float = 0.35,
    z_tol_mm: float = 0.35,
) -> dict[str, object]:
    _move_cyl(
        client,
        theta=float(target[0]),
        radius=float(target[1]),
        z=float(target[2]),
        timeout_sec=float(command_timeout_sec),
    )
    deadline = time.perf_counter() + max(0.5, float(command_timeout_sec))
    last_snapshot: dict[str, object] | None = None
    while time.perf_counter() <= deadline:
        snapshot = _wait_for_idle(client, timeout_sec=min(1.5, max(0.4, float(command_timeout_sec))))
        last_snapshot = snapshot
        if _pose_close(
            _current_cyl_pose(snapshot),
            target,
            theta_tol_deg=float(theta_tol_deg),
            radius_tol_mm=float(radius_tol_mm),
            z_tol_mm=float(z_tol_mm),
        ):
            time.sleep(max(0.0, float(settle_sec)))
            return snapshot
        time.sleep(0.15)
    raise TimeoutError(f"Timed out waiting for target pose {list(target)}. last={last_snapshot}")


def _measurement_summary(samples: Sequence[Mapping[str, object]], *, max_repeat_spread_px: float) -> dict[str, object]:
    distances: list[float] = []
    points: list[Sequence[float]] = []
    alignment_targets: list[Sequence[float]] = []
    measurement_points: list[str] = []
    point_distance_values: dict[str, list[float]] = {}
    bottom_edges: list[float] = []
    areas: list[float] = []
    transport_stats: list[object] = []
    captured_frames = 0
    processed_frames = 0
    for sample in samples:
        try:
            distance = float(sample.get("center_distance_px"))
        except (TypeError, ValueError):
            distance = float("inf")
        if math.isfinite(distance):
            distances.append(distance)
        point = sample.get("pixel")
        if isinstance(point, Sequence):
            points.append(point)
        target = sample.get("alignment_target_pixel")
        if isinstance(target, Sequence) and len(target) >= 2:
            alignment_targets.append(target)
        measurement_point = str(sample.get("measurement_point", "")).strip()
        if measurement_point:
            measurement_points.append(measurement_point)
        point_distances = sample.get("point_distances_px")
        if isinstance(point_distances, Mapping):
            for key, value in point_distances.items():
                try:
                    distance_value = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(distance_value):
                    point_distance_values.setdefault(str(key), []).append(distance_value)
        bbox = sample.get("bbox")
        if isinstance(bbox, Sequence) and len(bbox) >= 4:
            try:
                bottom = float(bbox[3])
            except (TypeError, ValueError):
                bottom = float("nan")
            if math.isfinite(bottom):
                bottom_edges.append(bottom)
        try:
            area = float(sample.get("area_px"))
        except (TypeError, ValueError):
            area = float("nan")
        if math.isfinite(area):
            areas.append(area)
        transport = sample.get("camera_transport")
        if isinstance(transport, Mapping):
            transport_stats.append(dict(transport))
        try:
            captured_frames += int(sample.get("captured_frames", 0) or 0)
            processed_frames += int(sample.get("processed_frames", 0) or 0)
        except (TypeError, ValueError):
            pass
    spread_px = _point_spread_px(points)
    median_dist = _median(distances)
    bottom_span_px = float(max(bottom_edges) - min(bottom_edges)) if len(bottom_edges) >= 2 else 0.0
    area_span_ratio = 0.0
    if len(areas) >= 2:
        median_area = max(1.0, _median(areas))
        area_span_ratio = float(max(areas) - min(areas)) / median_area
    transport_last = transport_stats[-1] if transport_stats else None
    stale_candidate_shift = False
    if isinstance(transport_last, Mapping):
        try:
            buffer_bytes = int(transport_last.get("buffer_bytes", 0) or 0)
        except (TypeError, ValueError):
            buffer_bytes = 0
        try:
            frames_rejected = int(transport_last.get("frames_rejected", 0) or 0)
        except (TypeError, ValueError):
            frames_rejected = 0
        stale_candidate_shift = bool(buffer_bytes > 250_000 or frames_rejected > 0)
    center_stable = bool(math.isfinite(median_dist) and spread_px <= float(max_repeat_spread_px))
    shape_stable = bool(bottom_span_px <= 3.0 and area_span_ratio <= 0.08)
    point_distance_medians = {
        key: float(_median(values))
        for key, values in sorted(point_distance_values.items())
        if values
    }
    return {
        "median_center_distance_px": float(median_dist),
        "min_center_distance_px": float(min(distances)) if distances else float("inf"),
        "max_center_distance_px": float(max(distances)) if distances else float("inf"),
        "repeat_spread_px": float(spread_px),
        "bottom_edge_span_px": float(bottom_span_px),
        "area_span_ratio": float(area_span_ratio),
        "center_stable": bool(center_stable),
        "shape_stable": bool(shape_stable),
        "stale_candidate_shift": bool(stale_candidate_shift),
        "stable": bool(center_stable and not stale_candidate_shift),
        "shape_warning": "" if shape_stable else "shape_changed_between_repeats",
        "alignment_target_pixel": (
            [float(alignment_targets[-1][0]), float(alignment_targets[-1][1])] if alignment_targets else None
        ),
        "measurement_point": measurement_points[-1] if measurement_points else "",
        "median_point_distances_px": point_distance_medians,
        "sample_count": int(len(samples)),
        "captured_frames": int(captured_frames),
        "processed_frames": int(processed_frames),
        "camera_transport_last": transport_last,
        "samples": list(samples),
    }


def _apply_report_alignment_from_summary(report: dict[str, object], summary: Mapping[str, object]) -> None:
    target = summary.get("alignment_target_pixel")
    if isinstance(target, Sequence) and not isinstance(target, (str, bytes)) and len(target) >= 2:
        try:
            report["alignment_target_pixel"] = [float(target[0]), float(target[1])]
            report["alignment_target_source"] = "sample_provenance"
        except (TypeError, ValueError):
            pass


def build_parser() -> argparse.ArgumentParser:
    defaults = AppConfig().resolved()
    parser = argparse.ArgumentParser(
        description=(
            "Safely search a low-height camera centering pose with stop-settle-measure steps. "
            "This tool only reads the official JetMax MJPEG URL and sends small MOVE_CYL commands; "
            "it never starts, restarts, repairs, scans, picks, or extends forward."
        )
    )
    parser.add_argument("--host", default=defaults.robot_host)
    parser.add_argument("--ros-port", type=int, default=defaults.rosbridge_port)
    parser.add_argument("--slot-id", type=int, required=True)
    parser.add_argument("--z-mm", type=float, default=140.0)
    parser.add_argument("--theta-step-deg", type=float, default=0.6)
    parser.add_argument("--radius-step-mm", type=float, default=2.0)
    parser.add_argument("--min-theta-step-deg", type=float, default=0.15)
    parser.add_argument("--min-radius-step-mm", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--shrink", type=float, default=0.5)
    parser.add_argument("--min-improvement-px", type=float, default=1.0)
    parser.add_argument("--center-tolerance-px", type=float, default=2.0)
    parser.add_argument(
        "--coarse-action-tolerance-px",
        type=float,
        default=8.0,
        help="Report-only transition tolerance for movement planning; final success remains --center-tolerance-px.",
    )
    parser.add_argument("--max-total-move-mm", type=float, default=18.0)
    parser.add_argument("--max-repeat-spread-px", type=float, default=4.0)
    parser.add_argument("--measurement-repeats", type=int, default=2)
    parser.add_argument(
        "--low-height-measurement-point",
        choices=tuple(sorted(SERVO_MEASUREMENT_POINTS)),
        default=None,
        help="Temporary override for the stopped low-height visual point used by this search.",
    )
    parser.add_argument(
        "--max-measure-attempts",
        type=int,
        default=None,
        help="Maximum read/recognition attempts used to collect --measurement-repeats valid low-height samples.",
    )
    parser.add_argument(
        "--fresh-reopen-before-measure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reopen only the PC-side MJPEG consumer before each settled measurement to discard buffered old frames.",
    )
    parser.add_argument(
        "--final-recheck-repeats",
        type=int,
        default=3,
        help="Repeat count for the final left-pose recheck; set 0 to disable.",
    )
    parser.add_argument(
        "--max-final-regression-px",
        type=float,
        default=2.0,
        help="Reject the stored best if final recheck is worse by more than this many pixels.",
    )
    parser.add_argument("--include-diagonal", action="store_true")
    parser.add_argument(
        "--theta-direction",
        choices=("both", "negative", "positive", "none"),
        default="both",
        help="Restrict low-height search theta offsets to avoid backlash during diagnosis.",
    )
    parser.add_argument(
        "--radius-direction",
        choices=("both", "negative", "positive", "none"),
        default="both",
        help="Restrict low-height search radius offsets to avoid backlash during diagnosis.",
    )
    parser.add_argument("--settle-sec", type=float, default=1.5)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--drain-frames", type=int, default=10)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--ros-timeout-sec", type=float, default=4.0)
    parser.add_argument("--command-timeout-sec", type=float, default=14.0)
    parser.add_argument("--detector", choices=("auto", "yolo", "fallback"), default="fallback")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--device", default=str(defaults.vision_device))
    parser.add_argument("--half", action="store_true", default=bool(defaults.vision_half))
    parser.add_argument("--profile", type=Path, default=defaults.vision_calibration_profile_path)
    parser.add_argument("--vision-grasp-profile", type=Path, default=defaults.vision_grasp_profile_path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Measure current pose only; do not move.")
    parser.add_argument("--return-to-start", action="store_true", help="Return to start pose instead of leaving at best pose.")
    return parser


def _measure_repeated(
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
    repeats: int,
    frame_id: int,
    frames: int,
    drain_frames: int,
    settle_sec: float,
    timeout_sec: float,
    ros_timeout_sec: float,
    max_repeat_spread_px: float,
    output_dir: Path,
    label: str,
    debug_slots,
    fresh_reopen_before_measure: bool = False,
    max_measure_attempts: int | None = None,
) -> tuple[dict[str, object], int, object]:
    samples: list[Mapping[str, object]] = []
    errors: list[str] = []
    packet_for_overlay: Mapping[str, object] | None = None
    frame_for_overlay = None
    next_frame_id = int(frame_id)
    next_debug_slots = debug_slots
    if bool(fresh_reopen_before_measure):
        reader.reopen()
    target_repeats = max(1, int(repeats))
    attempt_limit = max(target_repeats, int(max_measure_attempts or (target_repeats + 2)))
    repeat_index = 0
    for attempt_index in range(1, attempt_limit + 1):
        try:
            sample, packet, frame, next_frame_id, next_debug_slots = _measure_slot(
                reader=reader,
                cv2_module=cv2_module,
                model=model,
                config=config,
                calibration_profile=calibration_profile,
                client=client,
                device=device,
                half=half,
                slot_id=int(slot_id),
                frame_id=next_frame_id,
                frames=int(frames),
                drain_frames=int(drain_frames),
                timeout_sec=float(timeout_sec),
                ros_timeout_sec=float(ros_timeout_sec),
                settle_sec=float(settle_sec),
                debug_slots=next_debug_slots,
            )
        except Exception as error:
            errors.append(str(error))
            if bool(fresh_reopen_before_measure):
                reader.reopen()
            time.sleep(0.05)
            continue
        repeat_index += 1
        sample = dict(sample)
        sample["repeat_index"] = int(repeat_index)
        sample["attempt_index"] = int(attempt_index)
        samples.append(sample)
        packet_for_overlay = packet
        frame_for_overlay = frame
        step_dir = output_dir / label / f"repeat_{repeat_index:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        cv2_module.imwrite(str(step_dir / "raw.jpg"), frame)
        _write_json(step_dir / "packet.json", packet)
        _write_json(step_dir / "sample.json", sample)
        try:
            _save_overlay(
                cv2_module=cv2_module,
                frame=frame,
                packet=packet,
                selected_slot_id=int(slot_id),
                output_path=step_dir / "overlay.jpg",
            )
        except Exception:
            pass
        time.sleep(0.05)
        if repeat_index >= target_repeats:
            break
    if not samples:
        raise RuntimeError("Low-height measurement failed after retries: " + "; ".join(errors[-3:]))
    summary = _measurement_summary(samples, max_repeat_spread_px=float(max_repeat_spread_px))
    summary["label"] = label
    summary["measure_attempts"] = int(attempt_limit)
    summary["measurement_errors"] = list(errors)
    if packet_for_overlay is not None:
        summary["packet_frame_id"] = packet_for_overlay.get("frame_id")
    return summary, next_frame_id, next_debug_slots


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shrink = max(0.1, min(0.9, float(args.shrink)))
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else AppConfig().resolved().vision_debug_bundle_dir / f"low_height_center_search_{_timestamp()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AppConfig(
        robot_host=str(args.host),
        vision_calibration_profile_path=Path(args.profile),
        vision_grasp_profile_path=Path(args.vision_grasp_profile),
        vision_pick_confirm_z_mm=float(args.z_mm),
    ).resolved()
    grasp_profile = load_vision_grasp_profile(config)
    if grasp_profile.ready:
        config = apply_vision_grasp_profile(config, grasp_profile).resolved()
        config = replace(
            config,
            vision_pick_confirm_z_mm=float(args.z_mm),
        ).resolved()
    if args.low_height_measurement_point is not None:
        config = replace(
            config,
            vision_servo_low_height_measurement_point=str(args.low_height_measurement_point),
        ).resolved()
    calibration_profile = VisionCalibrationProfile.load(Path(args.profile))
    stream_url = config.resolve_vision_stream_url()
    report: dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "camera_contract": (
            "PC reads the single official JetMax MJPEG URL only; this tool does not start, restart, "
            "scan, repair, or mutate usb_cam.service/web_video_server/uvcvideo/devices."
        ),
        "measurement_contract": (
            "Low-height search uses stop-settle-measure: wait for IDLE, settle, drain transition frames, "
            "then measure repeated subpixel geometry points from the persistent official MJPEG stream."
        ),
        "stream_url": stream_url,
        "slot_id": int(args.slot_id),
        "z_mm": float(args.z_mm),
        "measurement_point": str(
            config.vision_servo_low_height_measurement_point or config.vision_servo_measurement_point
        ),
        "alignment_target_pixel": None,
        "alignment_target_source": "pending_first_sample",
        "coarse_action_tolerance_px": float(args.coarse_action_tolerance_px),
        "final_center_tolerance_px": float(args.center_tolerance_px),
        "settle_sec": float(args.settle_sec),
        "drain_frames": int(args.drain_frames),
        "fresh_reopen_before_measure": bool(args.fresh_reopen_before_measure),
        "final_recheck_repeats": int(args.final_recheck_repeats),
        "max_final_regression_px": float(args.max_final_regression_px),
        "theta_direction": str(args.theta_direction),
        "radius_direction": str(args.radius_direction),
        "dry_run": bool(args.dry_run),
        "steps": [],
    }

    import cv2

    device, auto_half = _resolve_device(str(args.device))
    half = bool(args.half or auto_half)
    model = _load_model(args, config)
    client = RosBridgeClient(host=str(args.host), port=int(args.ros_port), timeout_sec=float(args.ros_timeout_sec))
    reader: _PersistentCaptureReader | None = None
    frame_id = 0
    debug_slots = None
    best_pose: tuple[float, float, float] | None = None
    start_pose: tuple[float, float, float] | None = None
    target_leave_pose: tuple[float, float, float] | None = None
    exit_code = 0
    try:
        client.connect()
        _freeze_sucker(client, timeout_sec=float(args.ros_timeout_sec))
        initial = _wait_for_idle(client, timeout_sec=float(args.command_timeout_sec))
        current_pose = _current_cyl_pose(initial)
        if current_pose is None:
            raise RuntimeError("Initial robot cylindrical pose unavailable.")
        start_pose = (float(current_pose[0]), float(current_pose[1]), float(args.z_mm))
        report["start_pose_cyl"] = list(start_pose)
        if abs(float(current_pose[2]) - float(args.z_mm)) > 0.25 and not bool(args.dry_run):
            moved_snapshot = _move_and_wait_for_pose(
                client,
                target=(float(current_pose[0]), float(current_pose[1]), float(args.z_mm)),
                command_timeout_sec=float(args.command_timeout_sec),
                settle_sec=float(args.settle_sec),
            )
            initial = moved_snapshot
            moved_pose = _current_cyl_pose(moved_snapshot)
            if moved_pose is not None:
                start_pose = (float(moved_pose[0]), float(moved_pose[1]), float(moved_pose[2]))
                report["start_pose_cyl"] = list(start_pose)
        start_pose_for_move_limit = (float(start_pose[0]), float(start_pose[1]), float(args.z_mm))

        reader = _PersistentCaptureReader(cv2_module=cv2, stream_urls=(stream_url,), config=config, capture_backend="http")
        baseline, frame_id, debug_slots = _measure_repeated(
            reader=reader,
            cv2_module=cv2,
            model=model,
            config=config,
            calibration_profile=calibration_profile,
            client=client,
            device=device,
            half=half,
            slot_id=int(args.slot_id),
            repeats=int(args.measurement_repeats),
            frame_id=frame_id,
            frames=int(args.frames),
            drain_frames=int(args.drain_frames),
            settle_sec=float(args.settle_sec),
            timeout_sec=float(args.timeout_sec),
            ros_timeout_sec=float(args.ros_timeout_sec),
            max_repeat_spread_px=float(args.max_repeat_spread_px),
            output_dir=output_dir,
            label="baseline",
            debug_slots=debug_slots,
            fresh_reopen_before_measure=bool(args.fresh_reopen_before_measure),
            max_measure_attempts=args.max_measure_attempts,
        )
        baseline["pose_cyl"] = list(start_pose)
        report["baseline"] = baseline
        _apply_report_alignment_from_summary(report, baseline)
        print(
            "[baseline] pose=({:.2f},{:.2f},{:.2f}) dist={:.1f}px spread={:.1f}px stable={}".format(
                start_pose[0],
                start_pose[1],
                start_pose[2],
                float(baseline["median_center_distance_px"]),
                float(baseline["repeat_spread_px"]),
                bool(baseline["stable"]),
            )
        )
        best_pose = start_pose
        best_summary = baseline
        target_leave_pose = start_pose
        if bool(args.dry_run):
            report["best_pose_cyl"] = list(best_pose)
            report["best_measurement"] = best_summary
            _write_json(output_dir / "low_height_center_search_report.json", report)
            print(f"[output] report: {output_dir / 'low_height_center_search_report.json'}")
            return 0

        theta_step = abs(float(args.theta_step_deg))
        radius_step = abs(float(args.radius_step_mm))
        start_snapshot = initial
        for iteration in range(1, max(0, int(args.iterations)) + 1):
            if theta_step < float(args.min_theta_step_deg) and radius_step < float(args.min_radius_step_mm):
                report["stop_reason"] = "step_below_minimum"
                break
            iteration_record: dict[str, object] = {
                "iteration": int(iteration),
                "theta_step_deg": float(theta_step),
                "radius_step_mm": float(radius_step),
                "candidates": [],
            }
            improved = False
            candidate_summaries: list[dict[str, object]] = []
            offsets = _filter_candidate_offsets(
                _candidate_offsets(theta_step, radius_step, include_diagonal=bool(args.include_diagonal)),
                theta_direction=str(args.theta_direction),
                radius_direction=str(args.radius_direction),
            )
            iteration_record["candidate_offsets"] = [[float(t), float(r)] for t, r in offsets]
            for offset_index, (theta_offset, radius_offset) in enumerate(offsets, start=1):
                candidate_pose = (
                    float(best_pose[0]) + float(theta_offset),
                    float(best_pose[1]) + float(radius_offset),
                    float(args.z_mm),
                )
                label = f"iter_{iteration:02d}_candidate_{offset_index:02d}"
                candidate_record: dict[str, object] = {
                    "label": label,
                    "offset": [float(theta_offset), float(radius_offset)],
                    "pose_cyl": list(candidate_pose),
                    "skipped": False,
                }
                if _cartesian_distance_between_cyl(start_pose_for_move_limit, candidate_pose) > float(args.max_total_move_mm):
                    candidate_record["skipped"] = True
                    candidate_record["reason"] = "outside_max_total_move"
                    iteration_record["candidates"].append(candidate_record)
                    continue
                if not _within_cyl_limits(start_snapshot, theta=candidate_pose[0], radius=candidate_pose[1], z=candidate_pose[2]):
                    candidate_record["skipped"] = True
                    candidate_record["reason"] = "outside_cyl_limits"
                    iteration_record["candidates"].append(candidate_record)
                    continue
                _move_and_wait_for_pose(
                    client,
                    target=candidate_pose,
                    command_timeout_sec=float(args.command_timeout_sec),
                    settle_sec=float(args.settle_sec),
                )
                try:
                    summary, frame_id, debug_slots = _measure_repeated(
                        reader=reader,
                        cv2_module=cv2,
                        model=model,
                        config=config,
                        calibration_profile=calibration_profile,
                        client=client,
                        device=device,
                        half=half,
                        slot_id=int(args.slot_id),
                        repeats=int(args.measurement_repeats),
                        frame_id=frame_id,
                        frames=int(args.frames),
                        drain_frames=int(args.drain_frames),
                        settle_sec=float(args.settle_sec),
                        timeout_sec=float(args.timeout_sec),
                        ros_timeout_sec=float(args.ros_timeout_sec),
                        max_repeat_spread_px=float(args.max_repeat_spread_px),
                        output_dir=output_dir,
                        label=label,
                        debug_slots=debug_slots,
                        fresh_reopen_before_measure=bool(args.fresh_reopen_before_measure),
                        max_measure_attempts=args.max_measure_attempts,
                    )
                except Exception as error:
                    candidate_record["skipped"] = True
                    candidate_record["reason"] = "measurement_failed"
                    candidate_record["measurement_error"] = str(error)
                    iteration_record["candidates"].append(candidate_record)
                    print(
                        "[candidate] iter={} pose=({:.2f},{:.2f},{:.2f}) measurement_failed={}".format(
                            iteration,
                            candidate_pose[0],
                            candidate_pose[1],
                            candidate_pose[2],
                            error,
                        )
                    )
                    continue
                summary["pose_cyl"] = list(candidate_pose)
                candidate_record["measurement"] = summary
                iteration_record["candidates"].append(candidate_record)
                candidate_summaries.append(summary)
                print(
                    "[candidate] iter={} pose=({:.2f},{:.2f},{:.2f}) dist={:.1f}px spread={:.1f}px stable={}".format(
                        iteration,
                        candidate_pose[0],
                        candidate_pose[1],
                        candidate_pose[2],
                        float(summary["median_center_distance_px"]),
                        float(summary["repeat_spread_px"]),
                        bool(summary["stable"]),
                    )
                )
            stable_candidates = [
                item
                for item in candidate_summaries
                if bool(item.get("stable", False)) and math.isfinite(float(item.get("median_center_distance_px", float("inf"))))
            ]
            if stable_candidates:
                candidate_best = min(stable_candidates, key=lambda item: float(item["median_center_distance_px"]))
                improvement = float(best_summary["median_center_distance_px"]) - float(candidate_best["median_center_distance_px"])
                iteration_record["best_candidate"] = {
                    "pose_cyl": candidate_best.get("pose_cyl"),
                    "median_center_distance_px": candidate_best.get("median_center_distance_px"),
                    "improvement_px": float(improvement),
                }
                if improvement >= float(args.min_improvement_px):
                    best_pose = tuple(float(value) for value in candidate_best["pose_cyl"])  # type: ignore[index]
                    best_summary = candidate_best
                    target_leave_pose = best_pose
                    improved = True
                    print(
                        "[accept] iter={} best=({:.2f},{:.2f},{:.2f}) dist={:.1f}px improvement={:.1f}px".format(
                            iteration,
                            best_pose[0],
                            best_pose[1],
                            best_pose[2],
                            float(best_summary["median_center_distance_px"]),
                            improvement,
                        )
                    )
                    if float(best_summary["median_center_distance_px"]) <= float(args.center_tolerance_px):
                        iteration_record["stop_reason"] = "center_tolerance_reached"
                        report["steps"].append(iteration_record)
                        report["stop_reason"] = "center_tolerance_reached"
                        break
            if not improved:
                theta_step *= shrink
                radius_step *= shrink
                iteration_record["step_shrunk"] = True
            report["steps"].append(iteration_record)
        else:
            report["stop_reason"] = "iteration_limit"

        final_pose = start_pose if bool(args.return_to_start) else target_leave_pose
        if final_pose is not None:
            _move_and_wait_for_pose(
                client,
                target=(float(final_pose[0]), float(final_pose[1]), float(final_pose[2])),
                command_timeout_sec=float(args.command_timeout_sec),
                settle_sec=float(args.settle_sec),
            )
            report["left_pose_cyl"] = list(final_pose)
            if int(args.final_recheck_repeats) > 0:
                final_recheck, frame_id, debug_slots = _measure_repeated(
                    reader=reader,
                    cv2_module=cv2,
                    model=model,
                    config=config,
                    calibration_profile=calibration_profile,
                    client=client,
                    device=device,
                    half=half,
                    slot_id=int(args.slot_id),
                    repeats=int(args.final_recheck_repeats),
                    frame_id=frame_id,
                    frames=int(args.frames),
                    drain_frames=int(args.drain_frames),
                    settle_sec=float(args.settle_sec),
                    timeout_sec=float(args.timeout_sec),
                    ros_timeout_sec=float(args.ros_timeout_sec),
                    max_repeat_spread_px=float(args.max_repeat_spread_px),
                    output_dir=output_dir,
                    label="final_recheck",
                    debug_slots=debug_slots,
                    fresh_reopen_before_measure=bool(args.fresh_reopen_before_measure),
                    max_measure_attempts=args.max_measure_attempts,
                )
                final_recheck["pose_cyl"] = list(final_pose)
                report["final_recheck"] = final_recheck
                stored_best_px = float(best_summary["median_center_distance_px"])
                final_px = float(final_recheck["median_center_distance_px"])
                final_regression_px = final_px - stored_best_px
                report["final_recheck_guard"] = {
                    "stored_best_center_distance_px": stored_best_px,
                    "final_center_distance_px": final_px,
                    "final_regression_px": float(final_regression_px),
                    "max_final_regression_px": float(args.max_final_regression_px),
                    "final_recheck_stable": bool(final_recheck.get("stable", False)),
                }
                if (
                    not bool(final_recheck.get("stable", False))
                    or final_regression_px > float(args.max_final_regression_px)
                ):
                    report["best_recheck_failed"] = True
                    report["previous_stop_reason_before_recheck"] = str(report.get("stop_reason") or "")
                    report["stop_reason"] = "best_recheck_failed"
                    report["rejected_best_pose_cyl"] = list(final_pose)
                    report["rejected_best_measurement"] = final_recheck
                    if start_pose is not None and not bool(args.return_to_start):
                        _move_and_wait_for_pose(
                            client,
                            target=(float(start_pose[0]), float(start_pose[1]), float(start_pose[2])),
                            command_timeout_sec=float(args.command_timeout_sec),
                            settle_sec=float(args.settle_sec),
                        )
                        report["reverted_to_start_after_recheck_failed"] = True
                        report["left_pose_cyl"] = list(start_pose)
                        best_summary = baseline
                        best_pose = start_pose
                        target_leave_pose = start_pose
                    else:
                        best_summary = final_recheck
                        best_pose = final_pose
                        target_leave_pose = final_pose
                    print(
                        "[recheck] stored best rejected: final dist={:.1f}px stored={:.1f}px stable={}".format(
                            final_px,
                            stored_best_px,
                            bool(final_recheck.get("stable", False)),
                        )
                    )
                else:
                    print(
                        "[recheck] final pose confirmed: dist={:.1f}px regression={:.1f}px".format(
                            final_px,
                            final_regression_px,
                        )
                    )
        report["best_pose_cyl"] = list(best_pose) if best_pose is not None else None
        report["best_measurement"] = best_summary
        print(
            "[best] pose=({:.2f},{:.2f},{:.2f}) dist={:.1f}px".format(
                float(best_pose[0]),
                float(best_pose[1]),
                float(best_pose[2]),
                float(best_summary["median_center_distance_px"]),
            )
        )
    except Exception as error:
        report["error"] = str(error)
        print(f"[error] {error}", file=sys.stderr)
        exit_code = 1
    finally:
        if reader is not None:
            reader.close()
        try:
            client.close()
        except Exception:
            pass
        _write_json(output_dir / "low_height_center_search_report.json", report)
        print(f"[output] report: {output_dir / 'low_height_center_search_report.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
