from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_controller.app_robot_commands import extract_command_opcode
from hybrid_controller.app_robot_commands import rewrite_pick_command_with_bias
from hybrid_controller.config import AppConfig
from hybrid_controller.vision.calibration_profile import VisionCalibrationProfile
from hybrid_controller.vision.processing import (
    SlotState,
    annotate_slots_with_cylindrical,
    build_vision_packet,
    extract_candidates,
    update_slots,
)
from hybrid_controller.vision.runtime import _HttpMjpegCapture, _is_web_video_mjpeg_stream, _normalize_web_video_url
from hybrid_controller.vision.servo_controller import VisionServoController
from hybrid_controller.vision.target_resolver import resolve_vision_packet


class _EmptyDetectionResult:
    boxes = None
    masks = None


class RosBridgeClient:
    def __init__(self, *, host: str, port: int, timeout_sec: float) -> None:
        self.host = str(host)
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)
        self.ros = None

    def connect(self) -> None:
        import roslibpy

        ros = roslibpy.Ros(host=self.host, port=self.port)
        ready = threading.Event()
        error_holder: dict[str, str] = {}

        def _on_ready(*_args: object) -> None:
            ready.set()

        def _on_error(error: object) -> None:
            error_holder["error"] = str(error)
            ready.set()

        ros.on_ready(_on_ready, run_in_thread=False)
        ros.on("error", _on_error)
        ros.run()
        if not ready.wait(timeout=max(0.1, self.timeout_sec)):
            ros.close()
            raise TimeoutError("Timed out waiting for rosbridge connection.")
        if error_holder.get("error"):
            ros.close()
            raise RuntimeError(error_holder["error"])
        if not ros.is_connected:
            ros.close()
            raise RuntimeError("rosbridge not connected.")
        self.ros = ros

    def close(self) -> None:
        ros = self.ros
        self.ros = None
        if ros is not None:
            try:
                ros.close()
            except Exception:
                pass

    def fetch_state(self, *, timeout_sec: float | None = None) -> dict[str, object]:
        import roslibpy

        ros = self._require_ros()
        topic = roslibpy.Topic(ros, "/hybrid_controller/state", "hybrid_controller_ros/RobotState")
        done = threading.Event()
        holder: dict[str, object] = {"message": None}

        def _callback(message: dict[str, object]) -> None:
            holder["message"] = dict(message)
            done.set()

        topic.subscribe(_callback)
        try:
            if not done.wait(timeout=max(0.1, float(timeout_sec if timeout_sec is not None else self.timeout_sec))):
                raise TimeoutError("Timed out waiting for /hybrid_controller/state.")
        finally:
            try:
                topic.unsubscribe()
            except Exception:
                pass
        message = holder["message"]
        if not isinstance(message, dict):
            raise RuntimeError("Invalid /hybrid_controller/state payload.")
        return message

    def call_service(
        self,
        name: str,
        service_type: str,
        request: dict[str, object],
        *,
        timeout_sec: float,
    ) -> dict[str, object]:
        import roslibpy

        service = roslibpy.Service(self._require_ros(), name, service_type)
        done = threading.Event()
        holder: dict[str, object] = {"response": None, "error": None}

        def _ok(response: dict[str, object]) -> None:
            holder["response"] = dict(response)
            done.set()

        def _err(error: object) -> None:
            holder["error"] = str(error)
            done.set()

        service.call(roslibpy.ServiceRequest(request), callback=_ok, errback=_err)
        if not done.wait(timeout=max(0.1, float(timeout_sec))):
            raise TimeoutError(f"Timed out waiting for service '{name}'.")
        if holder["error"] is not None:
            raise RuntimeError(str(holder["error"]))
        response = holder["response"]
        if not isinstance(response, dict):
            raise RuntimeError(f"Invalid service response for '{name}': {response!r}")
        return response

    def _require_ros(self):
        if self.ros is None:
            raise RuntimeError("ROS client is not connected.")
        return self.ros


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_weights_path(config: AppConfig) -> Path:
    candidate = Path(config.vision_weights_path)
    if candidate.exists():
        return candidate.resolve()
    search_roots = (
        Path.cwd(),
        Path(__file__).resolve().parents[2],
        Path(__file__).resolve().parents[3],
    )
    for root in search_roots:
        alternate = (root / candidate).resolve()
        if alternate.exists():
            return alternate
    return candidate.resolve()


def _resolve_device(request: str) -> tuple[str | None, bool]:
    normalized = str(request or "auto").strip().lower()
    if normalized in {"", "auto"}:
        try:
            import torch

            if torch.cuda.is_available():
                return "0", True
        except Exception:
            pass
        return "cpu", False
    if normalized == "cpu":
        return "cpu", False
    return str(request).strip(), False


def _load_model(args: argparse.Namespace, config: AppConfig) -> object | None:
    detector = str(args.detector).strip().lower()
    if detector == "fallback":
        return None
    weights_path = Path(args.weights or _resolve_weights_path(config))
    if not weights_path.exists():
        if detector == "auto":
            print(f"[vision] YOLO weights unavailable, using color fallback: {weights_path}")
            return None
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    try:
        from ultralytics import YOLO

        print(f"[vision] Loading YOLO weights: {weights_path}")
        return YOLO(str(weights_path))
    except Exception:
        if detector == "auto":
            print("[vision] YOLO load failed, using color fallback.", file=sys.stderr)
            return None
        raise


def _override_profile_center_tolerance(
    profile: VisionCalibrationProfile,
    center_tolerance_px: float,
) -> VisionCalibrationProfile:
    tolerance = max(0.0, float(center_tolerance_px))
    stage_models = None
    if profile.stage_models:
        stage_models = {
            str(name): _override_profile_center_tolerance(stage_profile, tolerance)
            for name, stage_profile in profile.stage_models.items()
        }
    return replace(profile, center_tolerance_px=tolerance, stage_models=stage_models)


def _predict_frame(model: object | None, frame: object, *, config: AppConfig, device: str | None, half: bool) -> list[object]:
    if model is None:
        return [_EmptyDetectionResult()]
    if hasattr(model, "predict"):
        kwargs: dict[str, object] = {
            "source": frame,
            "imgsz": int(config.vision_model_imgsz),
            "conf": float(config.vision_confidence_threshold),
            "iou": float(config.vision_iou_threshold),
            "max_det": int(config.vision_max_det),
            "verbose": False,
        }
        if device:
            kwargs["device"] = device
        if half:
            kwargs["half"] = True
        return list(model.predict(**kwargs))
    return list(model(frame, verbose=False))


def _open_capture(cv2_module: object, stream_url: str, config: AppConfig):
    source = int(stream_url) if str(stream_url).isdigit() else _normalize_web_video_url(str(stream_url))
    if _is_web_video_mjpeg_stream(source):
        timeout_sec = max(0.2, float(config.vision_open_timeout_ms) / 1000.0)
        try:
            return _HttpMjpegCapture(str(source), cv2_module=cv2_module, timeout_sec=timeout_sec)
        except Exception:
            pass
    backend = getattr(cv2_module, "CAP_ANY", 0)
    if isinstance(source, str) and hasattr(cv2_module, "CAP_FFMPEG"):
        backend = getattr(cv2_module, "CAP_FFMPEG")
    try:
        capture = cv2_module.VideoCapture(source, backend)
    except TypeError:
        capture = cv2_module.VideoCapture(source)
    if hasattr(cv2_module, "CAP_PROP_BUFFERSIZE"):
        capture.set(cv2_module.CAP_PROP_BUFFERSIZE, 1)
    if hasattr(cv2_module, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        capture.set(cv2_module.CAP_PROP_OPEN_TIMEOUT_MSEC, float(config.vision_open_timeout_ms))
    if hasattr(cv2_module, "CAP_PROP_READ_TIMEOUT_MSEC"):
        capture.set(cv2_module.CAP_PROP_READ_TIMEOUT_MSEC, float(config.vision_read_timeout_ms))
    if not capture.isOpened():
        try:
            capture.release()
        except Exception:
            pass
        raise RuntimeError(f"Cannot open camera stream: {stream_url}")
    return capture


def _open_capture_with_backend(cv2_module: object, stream_url: str, config: AppConfig, *, capture_backend: str):
    backend = str(capture_backend or "auto").strip().lower()
    source = int(stream_url) if str(stream_url).isdigit() else _normalize_web_video_url(str(stream_url))
    if backend == "http" and _is_web_video_mjpeg_stream(source):
        timeout_sec = max(0.2, float(config.vision_open_timeout_ms) / 1000.0)
        return _HttpMjpegCapture(str(source), cv2_module=cv2_module, timeout_sec=timeout_sec)
    return _open_capture(cv2_module, stream_url, config)


def _read_frames_from_capture(
    *,
    capture: object,
    frame_count: int,
    drain_frames: int,
    timeout_sec: float,
) -> list[tuple[object, float]]:
    frames: list[tuple[object, float]] = []
    deadline = time.perf_counter() + max(0.5, float(timeout_sec))
    drain_remaining = max(0, int(drain_frames))
    while time.perf_counter() <= deadline and len(frames) < max(1, int(frame_count)):
        try:
            ok, frame = capture.read()  # type: ignore[attr-defined]
        except Exception:
            ok, frame = False, None
        now = time.perf_counter()
        if not ok or frame is None:
            time.sleep(0.03)
            continue
        if drain_remaining > 0:
            drain_remaining -= 1
            continue
        frames.append((frame, now))
    if not frames:
        raise RuntimeError("Timed out waiting for camera frames.")
    return frames


def _capture_frames(
    *,
    cv2_module: object,
    stream_url: str,
    config: AppConfig,
    frame_count: int,
    drain_frames: int,
    timeout_sec: float,
    capture_backend: str = "auto",
) -> list[tuple[object, float]]:
    capture = _open_capture_with_backend(cv2_module, stream_url, config, capture_backend=capture_backend)
    try:
        return _read_frames_from_capture(
            capture=capture,
            frame_count=frame_count,
            drain_frames=drain_frames,
            timeout_sec=timeout_sec,
        )
    finally:
        try:
            capture.release()
        except Exception:
            pass


def _capture_frames_from_candidates(
    *,
    cv2_module: object,
    stream_urls: tuple[str, ...],
    config: AppConfig,
    frame_count: int,
    drain_frames: int,
    timeout_sec: float,
    capture_backend: str = "auto",
) -> tuple[str, list[tuple[object, float]]]:
    errors: list[str] = []
    for candidate in stream_urls:
        stream_url = str(candidate).strip()
        if not stream_url:
            continue
        try:
            frames = _capture_frames(
                cv2_module=cv2_module,
                stream_url=stream_url,
                config=config,
                frame_count=frame_count,
                drain_frames=drain_frames,
                timeout_sec=timeout_sec,
                capture_backend=capture_backend,
            )
            return stream_url, frames
        except Exception as error:
            errors.append(f"{stream_url}: {error}")
    detail = "; ".join(errors) if errors else "no stream URLs configured"
    raise RuntimeError(f"Could not read camera frames from any stream candidate: {detail}")


class _PersistentCaptureReader:
    def __init__(
        self,
        *,
        cv2_module: object,
        stream_urls: tuple[str, ...],
        config: AppConfig,
        capture_backend: str = "auto",
    ) -> None:
        self._cv2_module = cv2_module
        self._stream_urls = tuple(str(item).strip() for item in stream_urls if str(item).strip())
        self._config = config
        self._capture_backend = str(capture_backend or "auto")
        self._capture: object | None = None
        self._stream_url: str | None = None

    @property
    def stream_url(self) -> str | None:
        return self._stream_url

    def read(
        self,
        *,
        frame_count: int,
        drain_frames: int,
        timeout_sec: float,
    ) -> tuple[str, list[tuple[object, float]]]:
        self._ensure_open()
        if self._capture is None or self._stream_url is None:
            raise RuntimeError("Persistent camera stream is not open.")
        try:
            frames = _read_frames_from_capture(
                capture=self._capture,
                frame_count=frame_count,
                drain_frames=drain_frames,
                timeout_sec=timeout_sec,
            )
        except Exception:
            self.close()
            raise
        return self._stream_url, frames

    def close(self) -> None:
        capture = self._capture
        self._capture = None
        self._stream_url = None
        if capture is None:
            return
        try:
            capture.release()  # type: ignore[attr-defined]
        except Exception:
            pass

    def _ensure_open(self) -> None:
        if self._capture is not None:
            return
        errors: list[str] = []
        for candidate in self._stream_urls:
            try:
                capture = _open_capture_with_backend(
                    self._cv2_module,
                    candidate,
                    self._config,
                    capture_backend=self._capture_backend,
                )
            except Exception as error:
                errors.append(f"{candidate}: {error}")
                continue
            self._capture = capture
            self._stream_url = candidate
            return
        detail = "; ".join(errors) if errors else "no stream URLs configured"
        raise RuntimeError(f"Could not open persistent camera stream: {detail}")


def _frame_batch_fps(frames: list[tuple[object, float]]) -> float:
    if len(frames) < 2:
        return 0.0
    elapsed = max(1e-6, float(frames[-1][1]) - float(frames[0][1]))
    return float(len(frames) - 1) / elapsed


def _select_latest_frames(
    frames: list[tuple[object, float]],
    latest_count: int | None,
) -> list[tuple[object, float]]:
    if not frames:
        return []
    if latest_count is None or int(latest_count) <= 0:
        return frames
    count = max(1, int(latest_count))
    if count >= len(frames):
        return frames
    return frames[-count:]


def _coerce_frame_pixel(value: object, frame_w: int, frame_h: int) -> tuple[float, float] | None:
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        return None
    if 0.0 <= x < float(frame_w) and 0.0 <= y < float(frame_h):
        return (x, y)
    return None


def _resolve_roi_center(config: AppConfig, frame_w: int, frame_h: int) -> tuple[int, int]:
    x = int(round(float(config.roi_center[0])))
    y = int(round(float(config.roi_center[1])))
    if 0 <= x < frame_w and 0 <= y < frame_h:
        return (x, y)
    return (frame_w // 2, frame_h // 2)


def _resolve_roi_radius(config: AppConfig, frame_w: int, frame_h: int) -> int:
    radius = int(round(float(config.roi_radius)))
    if radius > 0:
        return radius
    return max(40, int(round(min(frame_w, frame_h) * 0.28)))


def _resolve_alignment_target_pixel(
    *,
    config: AppConfig,
    calibration_profile: VisionCalibrationProfile | None,
    frame_w: int,
    frame_h: int,
    roi_center: tuple[int, int],
    calibration_stage: str | None = None,
    calibration_z_mm: float | None = None,
) -> tuple[float, float] | None:
    configured = _coerce_frame_pixel(config.vision_pick_target_pixel, frame_w, frame_h)
    if configured is not None:
        return configured
    if str(config.pick_tool_offset_source).strip().lower() == "command_bias":
        return (float(roi_center[0]), float(roi_center[1]))
    if calibration_profile is not None:
        try:
            active_profile = calibration_profile.model_for_stage(
                calibration_stage,
                z_mm=calibration_z_mm,
                allow_fallback=True,
            )
        except Exception:
            active_profile = calibration_profile
        profile_target = _coerce_frame_pixel(active_profile.target_pixel, frame_w, frame_h)
        if profile_target is not None:
            return profile_target
    if str(config.pick_tool_offset_source).strip().lower() == "target_pixel":
        return None
    return (float(roi_center[0]), float(roi_center[1]))


def _current_calibration_stage(config: AppConfig, snapshot: Mapping[str, object] | None) -> tuple[str, float]:
    robot_z = None
    if isinstance(snapshot, Mapping):
        try:
            robot_z = float(snapshot.get("robot_z"))
        except (TypeError, ValueError):
            cyl = snapshot.get("robot_cyl")
            if isinstance(cyl, Mapping):
                try:
                    robot_z = float(cyl.get("z_mm"))
                except (TypeError, ValueError):
                    robot_z = None
    search_z = float(getattr(config, "vision_pick_search_z_mm", config.robot_carry_z))
    confirm_z = float(getattr(config, "vision_pick_confirm_z_mm", config.robot_approach_z))
    pick_z = float(getattr(config, "robot_pick_z", confirm_z))
    tolerance = max(0.5, float(getattr(config, "vision_pick_z_tolerance_mm", 4.0)))
    if robot_z is None:
        return ("search", search_z)
    if abs(float(robot_z) - pick_z) <= tolerance:
        return ("pick", pick_z)
    if abs(float(robot_z) - confirm_z) <= tolerance:
        return ("confirm", confirm_z)
    if float(robot_z) < search_z - tolerance:
        return ("confirm", float(robot_z))
    return ("search", search_z)


def _state_message_to_snapshot(message: Mapping[str, object]) -> dict[str, object]:
    def _float(name: str, default: float = 0.0) -> float:
        try:
            return float(message.get(name, default))
        except (TypeError, ValueError):
            return float(default)

    snapshot = {
        "state": str(message.get("state", "")),
        "state_seq": int(message.get("state_seq", 0) or 0),
        "robot_ts": _float("robot_ts", 0.0),
        "busy": bool(message.get("busy", False)),
        "carrying": bool(message.get("carrying", False)),
        "robot_xy": [_float("x_mm"), _float("y_mm")],
        "robot_z": _float("z_mm"),
        "robot_cyl": {
            "theta_deg": _float("theta_deg"),
            "radius_mm": _float("radius_mm"),
            "z_mm": _float("z_mm"),
        },
        "limits_cyl": {
            "theta_deg": [_float("theta_min_deg", -120.0), _float("theta_max_deg", 120.0)],
            "radius_mm": [_float("radius_min_mm", 50.0), _float("radius_max_mm", 280.0)],
        },
        "limits_cyl_auto": {
            "radius_mm": [_float("auto_radius_min_mm", 80.0), _float("auto_radius_max_mm", 260.0)],
        },
        "calibration_ready": bool(message.get("calibration_ready", False)),
        "last_ack": str(message.get("last_ack", "")),
        "last_error_code": str(message.get("last_error_code", "")),
        "last_error": str(message.get("last_error_message", "")),
        "pick_tuning": {
            "pick_approach_z_mm": _float("pick_approach_z_mm", 130.0),
            "pick_descend_z_mm": _float("pick_descend_z_mm", 85.0),
            "pick_pre_suction_sec": _float("pick_pre_suction_sec", 0.0),
            "pick_bottom_hold_sec": _float("pick_bottom_hold_sec", 0.0),
            "pick_lift_sec": _float("pick_lift_sec", 0.0),
            "z_carry_floor_mm": _float("z_carry_floor_mm", 160.0),
        },
        "sucker_rotation_supported": bool(message.get("sucker_rotation_supported", False)),
    }
    return snapshot


def _current_cyl_pose(snapshot: Mapping[str, object] | None) -> tuple[float, float, float] | None:
    if not isinstance(snapshot, Mapping):
        return None
    cyl = snapshot.get("robot_cyl")
    if not isinstance(cyl, Mapping):
        return None
    try:
        return (
            float(cyl.get("theta_deg")),
            float(cyl.get("radius_mm")),
            float(cyl.get("z_mm", snapshot.get("robot_z", 0.0))),
        )
    except (TypeError, ValueError):
        return None


def _is_at_confirm_z(config: AppConfig, pose: tuple[float, float, float] | None) -> bool:
    if pose is None:
        return False
    tolerance = max(0.5, float(config.vision_pick_z_tolerance_mm))
    return abs(float(pose[2]) - float(config.vision_pick_confirm_z_mm)) <= tolerance


def _process_frame_batch(
    *,
    frames: list[tuple[object, float]],
    model: object | None,
    config: AppConfig,
    calibration_profile: VisionCalibrationProfile | None,
    snapshot_for_stage: Mapping[str, object] | None,
    frame_id_start: int,
    device: str | None,
    half: bool,
) -> tuple[dict[str, object], object, int]:
    slots = [SlotState(slot=index + 1, freq_hz=config.ssvep_freqs[index]) for index in range(config.vision_max_targets)]
    packet: dict[str, object] | None = None
    last_frame = frames[-1][0]
    frame_id = int(frame_id_start)
    batch_fps = _frame_batch_fps(frames)
    batch_capture_duration_ms = 0.0
    latest_frame_preprocess_age_ms = 0.0
    if frames:
        batch_capture_duration_ms = max(0.0, (float(frames[-1][1]) - float(frames[0][1])) * 1000.0)
        latest_frame_preprocess_age_ms = max(0.0, (time.perf_counter() - float(frames[-1][1])) * 1000.0)
    for frame, capture_ts in frames:
        frame_id += 1
        frame_h, frame_w = frame.shape[:2]
        roi_center = _resolve_roi_center(config, frame_w, frame_h)
        roi_radius = _resolve_roi_radius(config, frame_w, frame_h)
        calibration_stage, calibration_z_mm = _current_calibration_stage(config, snapshot_for_stage)
        alignment_target_pixel = _resolve_alignment_target_pixel(
            config=config,
            calibration_profile=calibration_profile,
            frame_w=frame_w,
            frame_h=frame_h,
            roi_center=roi_center,
            calibration_stage=calibration_stage,
            calibration_z_mm=calibration_z_mm,
        )
        action_center_tolerance_px = float(config.vision_servo_action_tolerance_px)
        if str(calibration_stage or "").strip().lower() == "search":
            action_center_tolerance_px = max(
                action_center_tolerance_px,
                float(getattr(config, "vision_servo_search_action_tolerance_px", action_center_tolerance_px)),
            )
        elif str(calibration_stage or "").strip().lower() in {"confirm", "pick"}:
            action_center_tolerance_px = max(
                action_center_tolerance_px,
                float(getattr(config, "vision_servo_low_action_tolerance_px", action_center_tolerance_px)),
            )
        infer_start = time.perf_counter()
        results = _predict_frame(model, frame, config=config, device=device, half=half)
        infer_ms = (time.perf_counter() - infer_start) * 1000.0
        result0 = results[0] if results else _EmptyDetectionResult()
        candidates, detected_count = extract_candidates(
            result0,
            frame_shape=(frame_h, frame_w),
            roi_center=roi_center,
            roi_radius=roi_radius,
            max_det=int(config.vision_max_targets),
            confidence_threshold=float(config.vision_confidence_threshold),
            frame_bgr=frame,
            fallback_to_frame=bool(config.vision_frame_fallback_enabled),
        )
        update_slots(
            slots,
            candidates,
            match_distance=120.0,
            lost_ttl=6,
            grasp_history_len=int(config.vision_grasp_history_frames),
            center_stability_tolerance_px=float(config.vision_center_stability_tolerance_px),
            grasp_stability_tolerance_px=float(config.vision_grasp_stability_tolerance_px),
            grasp_history_reset_px=float(config.vision_grasp_history_reset_px),
            grasp_angle_stability_tolerance_deg=float(config.vision_grasp_angle_stability_tolerance_deg),
        )
        annotate_slots_with_cylindrical(
            slots,
            calibration=None,
            calibration_profile=calibration_profile,
            frame_size=(frame_w, frame_h),
            roi_center=roi_center,
            world_scale_xy=float(config.vision_world_scale_xy),
            world_offset_xy_mm=(
                float(config.vision_world_offset_xy_mm[0]),
                float(config.vision_world_offset_xy_mm[1]),
            ),
            mapping_mode=str(config.vision_mapping_mode),
            calibration_profile_required=bool(config.vision_calibration_profile_required),
            action_error_threshold_mm=float(config.vision_action_max_error_mm),
            center_tolerance_px=float(config.vision_servo_center_tolerance_px),
            action_center_tolerance_px=action_center_tolerance_px,
            alignment_target_pixel=alignment_target_pixel,
            alignment_target_required=str(config.pick_tool_offset_source).strip().lower() == "target_pixel",
            calibration_stage=calibration_stage,
            calibration_z_mm=calibration_z_mm,
            grasp_quality_threshold=float(config.vision_grasp_quality_threshold),
            required_stable_frames=int(config.vision_grasp_stable_frames),
            grasp_angle_stability_tolerance_deg=float(config.vision_grasp_angle_stability_tolerance_deg),
        )
        calibration_ready = calibration_profile is not None and (
            calibration_profile.has_pixel_to_delta_model or calibration_profile.has_stage_models
        )
        packet = build_vision_packet(
            frame_id=frame_id,
            frame_size=(frame_w, frame_h),
            roi_center=roi_center,
            roi_radius=roi_radius,
            slots=slots,
            capture_fps=batch_fps,
            infer_ms=infer_ms,
            queue_age_ms=max(0.0, (time.perf_counter() - capture_ts) * 1000.0),
            detected_count=detected_count,
            calibration_ready=calibration_ready,
            capture_ts=float(capture_ts),
            stream_age_ms=max(0.0, (time.perf_counter() - capture_ts) * 1000.0),
            mapping_mode=str(config.vision_mapping_mode),
            calibration_profile_id="" if calibration_profile is None else calibration_profile.profile_id,
            calibration_profile_required=bool(config.vision_calibration_profile_required),
            alignment_target_pixel=alignment_target_pixel,
            calibration_stage=calibration_stage,
            calibration_z_mm=calibration_z_mm,
        )
        packet["capture_batch_frames"] = int(len(frames))
        packet["capture_batch_duration_ms"] = float(batch_capture_duration_ms)
        packet["latest_frame_preprocess_age_ms"] = float(latest_frame_preprocess_age_ms)
        last_frame = frame
    if packet is None:
        raise RuntimeError("No packet was generated from captured frames.")
    return packet, last_frame, frame_id


def _packet_frame_pose_age_ms(packet: Mapping[str, object]) -> float | None:
    for key in ("latest_frame_preprocess_age_ms", "stream_age_ms", "queue_age_ms"):
        value = packet.get(key)
        if value is None:
            continue
        try:
            age_ms = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(age_ms):
            return max(0.0, age_ms)
    return None


def _resolve_packet(
    *,
    packet: dict[str, object],
    config: AppConfig,
    snapshot: Mapping[str, object] | None,
    snapshot_age_ms: float,
    frame_pose_age_ms: float | None = None,
) -> dict[str, object]:
    if not isinstance(snapshot, Mapping):
        return dict(packet)
    result = resolve_vision_packet(
        packet,
        config=config,
        snapshot=snapshot,
        snapshot_age_ms=float(snapshot_age_ms),
        frame_pose_age_ms=frame_pose_age_ms,
    )
    return dict(result.packet)


def _slot_sort_key(slot: Mapping[str, object]) -> tuple[int, float, float]:
    if not bool(slot.get("valid", False)):
        return (9, float("inf"), 0.0)
    if bool(slot.get("actionable", False)):
        priority = 0
    elif str(slot.get("invalid_reason", "")) == "vision_servo_required":
        priority = 1
    else:
        priority = 2
    try:
        center_distance = float(slot.get("center_distance_px"))
    except (TypeError, ValueError):
        center_distance = float("inf")
    try:
        confidence = float(slot.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return (priority, center_distance, -confidence)


def _select_slot(packet: Mapping[str, object], slot_id: int | None) -> dict[str, object] | None:
    slots = packet.get("slots")
    if not isinstance(slots, list):
        return None
    valid_slots = [dict(slot) for slot in slots if isinstance(slot, Mapping)]
    if slot_id is not None:
        for slot in valid_slots:
            try:
                if int(slot.get("slot_id", slot.get("slot", -1))) == int(slot_id):
                    return slot
            except (TypeError, ValueError):
                continue
        return None
    valid_slots.sort(key=_slot_sort_key)
    if not valid_slots or not bool(valid_slots[0].get("valid", False)):
        return None
    return valid_slots[0]


def _decision_for_packet(
    *,
    packet: Mapping[str, object],
    config: AppConfig,
    snapshot: Mapping[str, object] | None,
    selected_slot: Mapping[str, object] | None,
) -> dict[str, object]:
    if selected_slot is None:
        return {
            "action": "CANCEL",
            "state": "FAILED",
            "status": "no_valid_slot",
            "reason": "no_valid_slot",
            "command": None,
            "pending": None,
        }
    slot_id = int(selected_slot.get("slot_id", selected_slot.get("slot", 0)))
    pose = _current_cyl_pose(snapshot)
    decision = VisionServoController(config).decide(
        slot_id=slot_id,
        slot_payload=selected_slot,
        packet=packet,
        pending=None,
        current_cyl_pose=pose,
        at_confirm_z=_is_at_confirm_z(config, pose),
        eye_in_hand_enabled=bool(config.vision_eye_in_hand_pick_flow_enabled),
    )
    return {
        "action": decision.action,
        "state": decision.state,
        "status": decision.status,
        "message": decision.message,
        "reason": decision.reason,
        "command": _rewrite_final_pick_command_for_debug(config=config, command=decision.command),
        "raw_command": decision.command,
        "pending": decision.pending_dict,
        "trace": dict(decision.trace),
    }


def _rewrite_final_pick_command_for_debug(*, config: AppConfig, command: str | None) -> str | None:
    if command is None:
        return None
    if str(config.pick_tool_offset_source).strip().lower() != "command_bias":
        return str(command)
    return rewrite_pick_command_with_bias(
        str(command),
        theta_bias_deg=float(getattr(config, "pick_cyl_theta_bias_deg", 0.0)),
        radius_bias_mm=float(getattr(config, "pick_cyl_radius_bias_mm", 0.0)),
        tangent_bias_mm=float(getattr(config, "pick_cyl_tangent_bias_mm", 0.0)),
        pick_z_mm=float(config.robot_pick_z),
    )


def _draw_text(cv2_module: object, image: object, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2_module.putText(image, text, origin, cv2_module.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
    cv2_module.putText(image, text, origin, cv2_module.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)


def _save_overlay(
    *,
    cv2_module: object,
    frame: object,
    packet: Mapping[str, object],
    selected_slot_id: int | None,
    output_path: Path,
) -> None:
    overlay = frame.copy()
    roi_center = packet.get("roi_center") or [frame.shape[1] // 2, frame.shape[0] // 2]
    roi_radius = int(packet.get("roi_radius", min(frame.shape[:2]) // 3) or 0)
    center = (int(round(float(roi_center[0]))), int(round(float(roi_center[1]))))
    if roi_radius > 0:
        cv2_module.circle(overlay, center, roi_radius, (80, 120, 255), 1)
    target = packet.get("alignment_target_pixel")
    if isinstance(target, (list, tuple)) and len(target) >= 2:
        point = (int(round(float(target[0]))), int(round(float(target[1]))))
        cv2_module.drawMarker(overlay, point, (255, 80, 220), cv2_module.MARKER_TILTED_CROSS, 32, 2)
        cv2_module.circle(overlay, point, 9, (255, 80, 220), 2)
        _draw_text(cv2_module, overlay, f"target_px={point[0]},{point[1]}", (10, 24), (255, 255, 255))
    slots = packet.get("slots")
    if isinstance(slots, list):
        for slot in slots:
            if not isinstance(slot, Mapping) or not bool(slot.get("valid", False)):
                continue
            slot_id = int(slot.get("slot_id", slot.get("slot", 0)) or 0)
            reason = str(slot.get("invalid_reason", ""))
            actionable = bool(slot.get("actionable", False))
            if selected_slot_id is not None and slot_id == int(selected_slot_id):
                color = (80, 255, 255)
            elif actionable:
                color = (80, 230, 80)
            elif reason == "vision_servo_required":
                color = (0, 210, 255)
            else:
                color = (80, 80, 255)
            bbox = slot.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
                cv2_module.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                label_y = max(16, y1 - 8)
            else:
                label_y = 48 + 18 * slot_id
                x1 = 10
            pixel_center = slot.get("pixel_center")
            if isinstance(pixel_center, (list, tuple)) and len(pixel_center) >= 2:
                center_point = (
                    int(round(float(pixel_center[0]))),
                    int(round(float(pixel_center[1]))),
                )
                cv2_module.drawMarker(overlay, center_point, (255, 255, 0), cv2_module.MARKER_CROSS, 22, 2)
                cv2_module.circle(overlay, center_point, 5, (255, 255, 0), 1)
            grasp = slot.get("grasp_pixel")
            if isinstance(grasp, (list, tuple)) and len(grasp) >= 2:
                point = (int(round(float(grasp[0]))), int(round(float(grasp[1]))))
                cv2_module.drawMarker(overlay, point, color, cv2_module.MARKER_CROSS, 24, 2)
                cv2_module.circle(overlay, point, 8, color, 1)
            dist = slot.get("center_distance_px")
            dist_text = "--" if dist is None else f"{float(dist):.1f}px"
            label = f"[{slot_id}] {'PICK' if actionable else reason or 'valid'} dist={dist_text}"
            _draw_text(cv2_module, overlay, label, (int(x1), int(label_y)), color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2_module.imwrite(str(output_path), overlay)


def _execute_command(
    *,
    client: RosBridgeClient,
    command: str | None,
    allow_pick: bool,
    timeout_sec: float,
) -> dict[str, object]:
    if not command:
        return {"executed": False, "reason": "command_unavailable"}
    opcode = extract_command_opcode(command)
    parts = str(command).split()
    if opcode in {"PICK_CYL", "PICK_WORLD"} and not bool(allow_pick):
        return {
            "executed": False,
            "reason": "pick_blocked_requires_allow_pick",
            "command": command,
        }
    if opcode == "MOVE_CYL" and len(parts) >= 4:
        response = client.call_service(
            "/hybrid_controller/move_cyl",
            "hybrid_controller_ros/MoveCyl",
            {
                "theta_deg": float(parts[1]),
                "radius_mm": float(parts[2]),
                "z_mm": float(parts[3]),
            },
            timeout_sec=timeout_sec,
        )
    elif opcode == "MOVE_CYL_AUTO" and len(parts) >= 3:
        response = client.call_service(
            "/hybrid_controller/move_cyl_auto",
            "hybrid_controller_ros/MoveCylAuto",
            {
                "theta_deg": float(parts[1]),
                "radius_mm": float(parts[2]),
            },
            timeout_sec=timeout_sec,
        )
    elif opcode == "PICK_CYL" and len(parts) >= 3:
        angle = float(parts[3]) if len(parts) >= 4 else None
        response = client.call_service(
            "/hybrid_controller/pick_cyl",
            "hybrid_controller_ros/PickCyl",
            {
                "theta_deg": float(parts[1]),
                "radius_mm": float(parts[2]),
                "use_sucker_rotation": angle is not None,
                "sucker_rotation_deg": 0.0 if angle is None else float(angle),
            },
            timeout_sec=max(25.0, timeout_sec),
        )
    elif opcode == "PICK_WORLD" and len(parts) >= 3:
        angle = float(parts[3]) if len(parts) >= 4 else None
        response = client.call_service(
            "/hybrid_controller/pick_world",
            "hybrid_controller_ros/PickWorld",
            {
                "x_mm": float(parts[1]),
                "y_mm": float(parts[2]),
                "use_sucker_rotation": angle is not None,
                "sucker_rotation_deg": 0.0 if angle is None else float(angle),
            },
            timeout_sec=max(25.0, timeout_sec),
        )
    else:
        return {"executed": False, "reason": f"unsupported_command:{opcode}", "command": command}
    return {"executed": True, "command": command, "response": response}


def _wait_for_idle(
    *,
    client: RosBridgeClient,
    timeout_sec: float,
    poll_sec: float = 0.4,
) -> dict[str, object] | None:
    deadline = time.perf_counter() + max(0.1, float(timeout_sec))
    last_snapshot: dict[str, object] | None = None
    while time.perf_counter() <= deadline:
        try:
            state = client.fetch_state(timeout_sec=max(0.3, min(2.0, float(poll_sec) + 0.3)))
            last_snapshot = _state_message_to_snapshot(state)
        except Exception:
            time.sleep(max(0.05, float(poll_sec)))
            continue
        if not bool(last_snapshot.get("busy", False)):
            return last_snapshot
        time.sleep(max(0.05, float(poll_sec)))
    return last_snapshot


def _slot_summary(packet: Mapping[str, object]) -> list[dict[str, object]]:
    slots = packet.get("slots")
    if not isinstance(slots, list):
        return []
    summary: list[dict[str, object]] = []
    for slot in slots:
        if not isinstance(slot, Mapping) or not bool(slot.get("valid", False)):
            continue
        summary.append(
            {
                "slot_id": int(slot.get("slot_id", slot.get("slot", 0)) or 0),
                "bbox": slot.get("bbox"),
                "pixel_center": slot.get("pixel_center"),
                "grasp_pixel": slot.get("grasp_pixel"),
                "confidence": slot.get("confidence"),
                "grasp_quality": slot.get("grasp_quality"),
                "center_stable_frames": slot.get("center_stable_frames"),
                "center_stability_px": slot.get("center_stability_px"),
                "grasp_stable_frames": slot.get("grasp_stable_frames"),
                "grasp_stability_px": slot.get("grasp_stability_px"),
                "center_distance_px": slot.get("center_distance_px"),
                "center_tolerance_px": slot.get("center_tolerance_px"),
                "action_tolerance_px": slot.get("action_tolerance_px"),
                "estimated_xy_error_mm": slot.get("estimated_xy_error_mm"),
                "servo_required": bool(slot.get("servo_required", False)),
                "actionable": bool(slot.get("actionable", False)),
                "invalid_reason": str(slot.get("invalid_reason", "")),
                "camera_to_world_raw": slot.get("camera_to_world_raw"),
                "servo_command_point": slot.get("servo_command_point"),
                "command_point": slot.get("command_point"),
                "resolved_cyl": slot.get("resolved_cyl"),
            }
        )
    return summary


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    defaults = AppConfig().resolved()
    parser = argparse.ArgumentParser(
        description="Debug camera recognition -> target resolution -> robot grasp command flow."
    )
    parser.add_argument("--host", default=defaults.robot_host)
    parser.add_argument("--ros-port", type=int, default=defaults.rosbridge_port)
    parser.add_argument("--stream-url", default="")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--profile", type=Path, default=defaults.vision_calibration_profile_path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--slot-id", type=int, default=None)
    parser.add_argument("--frames", type=int, default=max(3, int(defaults.vision_grasp_stable_frames)))
    parser.add_argument("--drain-frames", type=int, default=30)
    parser.add_argument(
        "--process-latest-frames",
        type=int,
        default=0,
        help="Process only the newest N captured frames; 0 keeps the full captured batch.",
    )
    parser.add_argument(
        "--capture-backend",
        choices=("auto", "http"),
        default="http",
        help="Camera capture backend. Use http for Hiwonder web_video_server MJPEG.",
    )
    parser.add_argument(
        "--persistent-camera",
        action="store_true",
        help="Keep one camera stream open for the whole debug run instead of opening it once per step.",
    )
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--ros-timeout-sec", type=float, default=2.0)
    parser.add_argument("--command-timeout-sec", type=float, default=12.0)
    parser.add_argument("--settle-sec", type=float, default=1.0)
    parser.add_argument("--detector", choices=("auto", "yolo", "fallback"), default="auto")
    parser.add_argument("--device", default=str(defaults.vision_device))
    parser.add_argument("--half", action="store_true", default=bool(defaults.vision_half))
    parser.add_argument(
        "--center-tolerance-px",
        type=float,
        default=None,
        help="Override the camera-centering tolerance for this debug run only.",
    )
    parser.add_argument("--no-ros", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Allow MOVE commands to be sent to the robot.")
    parser.add_argument(
        "--allow-execute-loop",
        action="store_true",
        help=(
            "Allow --execute to run more than one MOVE/PICK decision step in one process. "
            "Use with --persistent-camera for live visual-servo descent debugging."
        ),
    )
    parser.add_argument("--allow-pick", action="store_true", help="Allow PICK commands that descend and turn suction on.")
    parser.add_argument(
        "--pick-radius-bias-mm",
        type=float,
        default=defaults.vision_eye_in_hand_pick_radius_bias_mm,
        help=(
            "Apply this final cylindrical radius offset inside the vision-servo PICK decision when "
            "--pick-tool-offset-source command_bias is used. Keep pick_cyl_radius_bias_mm at 0 to avoid double bias."
        ),
    )
    parser.add_argument(
        "--pick-tool-offset-source",
        choices=("target_pixel", "command_bias"),
        default=defaults.pick_tool_offset_source,
        help="target_pixel uses profile servo.target_pixel; command_bias enables legacy explicit pick radius/tangent/theta offsets.",
    )
    parser.add_argument(
        "--confirm-z-mm",
        type=float,
        default=None,
        help="Override the visual confirmation height for this debug run only.",
    )
    parser.add_argument("--max-steps", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.execute) and int(args.max_steps) > 1 and not bool(args.allow_execute_loop):
        print(
            "[guard] Refusing --execute with --max-steps > 1 unless --allow-execute-loop is set. "
            "Run one MOVE step at a time, or use the GUI VisionRuntime persistent camera path.",
            file=sys.stderr,
        )
        return 2
    config_kwargs: dict[str, object] = {
        "robot_host": str(args.host),
        "vision_calibration_profile_path": Path(args.profile),
        "vision_eye_in_hand_pick_radius_bias_mm": float(args.pick_radius_bias_mm),
        "pick_tool_offset_source": str(args.pick_tool_offset_source),
    }
    if args.confirm_z_mm is not None:
        config_kwargs["vision_pick_confirm_z_mm"] = float(args.confirm_z_mm)
    if args.center_tolerance_px is not None:
        config_kwargs["vision_servo_center_tolerance_px"] = float(args.center_tolerance_px)
        config_kwargs["vision_servo_action_tolerance_px"] = float(args.center_tolerance_px)
    config = AppConfig(**config_kwargs).resolved()
    explicit_stream_url = str(args.stream_url).strip()
    # Default capture follows the locked JetMax camera contract: one official
    # MJPEG URL from AppConfig, no endpoint scan, no robot-side camera mutation.
    # --stream-url is reserved for manual diagnosis and should not be used in
    # normal grasp tuning.
    stream_candidates = (explicit_stream_url,) if explicit_stream_url else config.resolve_vision_stream_candidates()
    stream_url = stream_candidates[0]
    output_dir = Path(args.output_dir) if args.output_dir is not None else config.vision_debug_bundle_dir / f"vision_grasp_{_timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    import cv2

    calibration_profile = None
    if Path(args.profile).exists():
        calibration_profile = VisionCalibrationProfile.load(Path(args.profile))
        if args.center_tolerance_px is not None:
            calibration_profile = _override_profile_center_tolerance(
                calibration_profile,
                float(args.center_tolerance_px),
            )
    else:
        print(f"[vision] Calibration profile missing: {args.profile}", file=sys.stderr)

    device, auto_half = _resolve_device(str(args.device))
    half = bool(args.half or auto_half)
    model = _load_model(args, config)

    client: RosBridgeClient | None = None
    initial_snapshot: dict[str, object] | None = None
    ros_status: dict[str, object] = {"connected": False}
    if not bool(args.no_ros):
        client = RosBridgeClient(host=str(args.host), port=int(args.ros_port), timeout_sec=float(args.ros_timeout_sec))
        try:
            client.connect()
            initial_state = client.fetch_state(timeout_sec=float(args.ros_timeout_sec))
            initial_snapshot = _state_message_to_snapshot(initial_state)
            ros_status = {"connected": True, "snapshot": initial_snapshot}
            print(
                "[robot] state={state} busy={busy} theta={theta:.2f} radius={radius:.2f} z={z:.2f}".format(
                    state=initial_snapshot.get("state", ""),
                    busy=initial_snapshot.get("busy", False),
                    theta=float(initial_snapshot["robot_cyl"]["theta_deg"]),  # type: ignore[index]
                    radius=float(initial_snapshot["robot_cyl"]["radius_mm"]),  # type: ignore[index]
                    z=float(initial_snapshot["robot_cyl"]["z_mm"]),  # type: ignore[index]
                )
            )
        except Exception as error:
            ros_status = {"connected": False, "error": str(error)}
            print(f"[robot] ROS unavailable for this dry run: {error}", file=sys.stderr)
            if bool(args.execute):
                print("[robot] --execute requested but ROS is unavailable; no robot command will be sent.", file=sys.stderr)

    report: dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stream_url": stream_url,
        "weights": str(args.weights or _resolve_weights_path(config)),
        "profile": str(args.profile),
        "detector": str(args.detector),
        "execute": bool(args.execute),
        "allow_pick": bool(args.allow_pick),
        "pick_radius_bias_mm": float(args.pick_radius_bias_mm),
        "confirm_z_mm": None if args.confirm_z_mm is None else float(args.confirm_z_mm),
        "center_tolerance_px": None if args.center_tolerance_px is None else float(args.center_tolerance_px),
        "frames_requested": int(args.frames),
        "drain_frames": int(args.drain_frames),
        "process_latest_frames": int(args.process_latest_frames),
        "capture_backend": str(args.capture_backend),
        "persistent_camera": bool(args.persistent_camera),
        "ros": ros_status,
        "steps": [],
    }
    frame_id = 0
    exit_code = 0
    persistent_reader: _PersistentCaptureReader | None = None
    if bool(args.persistent_camera):
        persistent_reader = _PersistentCaptureReader(
            cv2_module=cv2,
            stream_urls=stream_candidates,
            config=config,
            capture_backend=str(args.capture_backend),
        )
    try:
        for step_index in range(max(1, int(args.max_steps))):
            print(f"[step {step_index + 1}] capturing {int(args.frames)} frame(s) from camera...")
            if persistent_reader is not None:
                stream_url, frames = persistent_reader.read(
                    frame_count=int(args.frames),
                    drain_frames=int(args.drain_frames),
                    timeout_sec=float(args.timeout_sec),
                )
            else:
                stream_url, frames = _capture_frames_from_candidates(
                    cv2_module=cv2,
                    stream_urls=stream_candidates,
                    config=config,
                    frame_count=int(args.frames),
                    drain_frames=int(args.drain_frames),
                    timeout_sec=float(args.timeout_sec),
                    capture_backend=str(args.capture_backend),
                )
            process_frames = _select_latest_frames(frames, int(args.process_latest_frames))
            report["stream_url"] = stream_url
            packet, last_frame, frame_id = _process_frame_batch(
                frames=process_frames,
                model=model,
                config=config,
                calibration_profile=calibration_profile,
                snapshot_for_stage=initial_snapshot,
                frame_id_start=frame_id,
                device=device,
                half=half,
            )
            packet["camera_frames_captured"] = int(len(frames))
            packet["camera_frames_processed"] = int(len(process_frames))

            resolve_snapshot = initial_snapshot
            snapshot_age_ms = 0.0
            if client is not None and ros_status.get("connected"):
                try:
                    state_ts = time.perf_counter()
                    state_message = client.fetch_state(timeout_sec=float(args.ros_timeout_sec))
                    resolve_snapshot = _state_message_to_snapshot(state_message)
                    snapshot_age_ms = max(0.0, (time.perf_counter() - state_ts) * 1000.0)
                except Exception as error:
                    print(f"[robot] Could not refresh state for resolution: {error}", file=sys.stderr)
            resolved_packet = _resolve_packet(
                packet=packet,
                config=config,
                snapshot=resolve_snapshot,
                snapshot_age_ms=snapshot_age_ms,
                frame_pose_age_ms=_packet_frame_pose_age_ms(packet),
            )
            resolved_packet["camera_frames_captured"] = int(len(frames))
            resolved_packet["camera_frames_processed"] = int(len(process_frames))
            selected_slot = _select_slot(resolved_packet, args.slot_id)
            selected_slot_id = None if selected_slot is None else int(selected_slot.get("slot_id", selected_slot.get("slot", 0)))
            decision = _decision_for_packet(
                packet=resolved_packet,
                config=config,
                snapshot=resolve_snapshot,
                selected_slot=selected_slot,
            )

            step_dir = output_dir / f"step_{step_index + 1:02d}"
            raw_path = step_dir / "raw.jpg"
            overlay_path = step_dir / "overlay.jpg"
            packet_path = step_dir / "packet.json"
            step_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(raw_path), last_frame)
            _save_overlay(
                cv2_module=cv2,
                frame=last_frame,
                packet=resolved_packet,
                selected_slot_id=selected_slot_id,
                output_path=overlay_path,
            )
            _write_json(packet_path, resolved_packet)

            step_report: dict[str, object] = {
                "step": step_index + 1,
                "raw_image": str(raw_path),
                "overlay_image": str(overlay_path),
                "packet": str(packet_path),
                "camera_frames_captured": int(len(frames)),
                "camera_frames_processed": int(len(process_frames)),
                "slots": _slot_summary(resolved_packet),
                "selected_slot_id": selected_slot_id,
                "selected_slot": selected_slot,
                "decision": decision,
                "snapshot": resolve_snapshot,
            }
            print(f"[step {step_index + 1}] valid_slots={len(step_report['slots'])} selected={selected_slot_id}")
            if selected_slot is not None:
                print(
                    "[step {step}] center dist={dist} tol={tol} action_tol={action_tol} "
                    "frames={processed}/{captured} age={age_ms}ms".format(
                        step=step_index + 1,
                        dist=(
                            "--"
                            if selected_slot.get("center_distance_px") is None
                            else f"{float(selected_slot.get('center_distance_px')):.1f}px"
                        ),
                        tol=(
                            "--"
                            if selected_slot.get("center_tolerance_px") is None
                            else f"{float(selected_slot.get('center_tolerance_px')):.1f}px"
                        ),
                        action_tol=(
                            "--"
                            if selected_slot.get("action_tolerance_px") is None
                            else f"{float(selected_slot.get('action_tolerance_px')):.1f}px"
                        ),
                        processed=int(len(process_frames)),
                        captured=int(len(frames)),
                        age_ms=(
                            "--"
                            if resolved_packet.get("queue_age_ms") is None
                            else f"{float(resolved_packet.get('queue_age_ms')):.0f}"
                        ),
                    )
                )
            print(
                "[step {step}] decision action={action} reason={reason} command={command}".format(
                    step=step_index + 1,
                    action=decision.get("action"),
                    reason=decision.get("reason"),
                    command=decision.get("command"),
                )
            )

            if bool(args.execute) and client is not None and ros_status.get("connected"):
                if bool(resolve_snapshot and resolve_snapshot.get("busy", False)):
                    step_report["execution"] = {"executed": False, "reason": "robot_busy"}
                    print("[robot] Robot is busy; command not sent.", file=sys.stderr)
                else:
                    execution = _execute_command(
                        client=client,
                        command=None if decision.get("command") is None else str(decision.get("command")),
                        allow_pick=bool(args.allow_pick),
                        timeout_sec=float(args.command_timeout_sec),
                    )
                    step_report["execution"] = execution
                    print(f"[robot] execution={execution}")
                    if bool(execution.get("executed", False)):
                        settled = _wait_for_idle(
                            client=client,
                            timeout_sec=max(float(args.command_timeout_sec), float(args.settle_sec)),
                        )
                        step_report["post_execution_snapshot"] = settled
                        time.sleep(max(0.0, float(args.settle_sec)))
                        initial_snapshot = settled if settled is not None else resolve_snapshot
                    if str(decision.get("action")) == "PICK" or not bool(execution.get("executed", False)):
                        report["steps"].append(step_report)
                        if str(decision.get("action")) == "WAIT_STABLE" and str(execution.get("reason")) == "command_unavailable":
                            time.sleep(max(0.0, float(args.settle_sec)))
                            initial_snapshot = resolve_snapshot
                            continue
                        break
            else:
                step_report["execution"] = {"executed": False, "reason": "dry_run"}
                report["steps"].append(step_report)
                break

            report["steps"].append(step_report)
        report_path = output_dir / "debug_vision_grasp_flow.json"
        _write_json(report_path, report)
        print(f"[output] overlay/report saved under: {output_dir}")
        print(f"[output] report: {report_path}")
    except Exception as error:
        report["error"] = str(error)
        report_path = output_dir / "debug_vision_grasp_flow.json"
        _write_json(report_path, report)
        print(f"[error] {error}", file=sys.stderr)
        print(f"[output] partial report: {report_path}", file=sys.stderr)
        exit_code = 1
    finally:
        if persistent_reader is not None:
            persistent_reader.close()
        if client is not None:
            client.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
