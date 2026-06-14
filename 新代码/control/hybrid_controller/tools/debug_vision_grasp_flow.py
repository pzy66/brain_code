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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_controller.app_robot_commands import extract_command_opcode
from hybrid_controller.app_robot_commands import rewrite_pick_command_with_bias
from hybrid_controller.adapters.teleop_ros_channel import RosTeleopPublishPlanner
from hybrid_controller.adapters.teleop_ros_channel import new_teleop_cmd_seq_base
from hybrid_controller.adapters.teleop_ros_channel import next_teleop_cmd_seq
from hybrid_controller.config import AppConfig
from hybrid_controller.config import SERVO_MEASUREMENT_POINTS
from hybrid_controller.vision.calibration_profile import VisionCalibrationProfile
from hybrid_controller.vision.continuous_servo_controller import ContinuousVisionServoController
from hybrid_controller.vision.grasp_profile import apply_vision_grasp_profile
from hybrid_controller.vision.grasp_profile import load_vision_grasp_profile
from hybrid_controller.vision.processing import (
    SlotState,
    annotate_slots_with_cylindrical,
    build_vision_packet,
    extract_candidates,
    frame_brightness_quality,
    sanitize_frame_edge_bands,
    update_slots,
)
from hybrid_controller.vision.runtime import (
    _HttpMjpegCapture,
    _frame_has_horizontal_tearing,
    _frame_is_temporal_splice,
    _is_web_video_mjpeg_stream,
    _normalize_web_video_url,
)
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
        self._state_topic = None
        self._teleop_topic = None
        self._state_lock = threading.Lock()
        self._state_ready = threading.Event()
        self._latest_state: dict[str, object] | None = None
        self._latest_state_local_ts: float = 0.0

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
        self._subscribe_state()

    def close(self) -> None:
        if self._state_topic is not None:
            try:
                self._state_topic.unsubscribe()
            except Exception:
                pass
            self._state_topic = None
        if self._teleop_topic is not None:
            try:
                self._teleop_topic.unadvertise()
            except Exception:
                pass
            self._teleop_topic = None
        ros = self.ros
        self.ros = None
        with self._state_lock:
            self._latest_state = None
            self._latest_state_local_ts = 0.0
            self._state_ready.clear()
        if ros is not None:
            try:
                ros.close()
            except Exception:
                pass

    def fetch_state(self, *, timeout_sec: float | None = None) -> dict[str, object]:
        timeout = max(0.1, float(timeout_sec if timeout_sec is not None else self.timeout_sec))
        if not self._state_ready.wait(timeout=timeout):
            raise TimeoutError("Timed out waiting for /hybrid_controller/state.")
        with self._state_lock:
            message = None if self._latest_state is None else dict(self._latest_state)
            local_ts = float(self._latest_state_local_ts)
        if not isinstance(message, dict):
            raise RuntimeError("Invalid /hybrid_controller/state payload.")
        message["_local_receive_ts"] = local_ts
        return message

    def advertise_teleop(self) -> None:
        import roslibpy

        if self._teleop_topic is not None:
            return
        self._teleop_topic = roslibpy.Topic(
            self._require_ros(),
            "/hybrid_controller/teleop_cyl_cmd",
            "hybrid_controller_ros/CylindricalTeleop",
            queue_size=1,
        )
        self._teleop_topic.advertise()

    def publish_teleop(
        self,
        *,
        theta_rate_deg_s: float,
        radius_rate_mm_s: float,
        z_rate_mm_s: float = 0.0,
        use_auto_z: bool = False,
        enabled: bool,
        cmd_seq: int,
        client_ts: float,
    ) -> None:
        import roslibpy

        self.advertise_teleop()
        if self._teleop_topic is None:
            raise RuntimeError("ROS teleop topic is not ready.")
        self._teleop_topic.publish(
            roslibpy.Message(
                {
                    "theta_rate_deg_s": float(theta_rate_deg_s),
                    "radius_rate_mm_s": float(radius_rate_mm_s),
                    "z_rate_mm_s": float(z_rate_mm_s),
                    "use_auto_z": bool(use_auto_z),
                    "enabled": bool(enabled),
                    "cmd_seq": int(max(0, int(cmd_seq))),
                    "client_ts": float(client_ts),
                }
            )
        )

    def stop_teleop(self, *, use_auto_z: bool = False, cmd_seq: int = 0) -> None:
        if self._teleop_topic is None:
            return
        self.publish_teleop(
            theta_rate_deg_s=0.0,
            radius_rate_mm_s=0.0,
            z_rate_mm_s=0.0,
            use_auto_z=bool(use_auto_z),
            enabled=False,
            cmd_seq=int(max(0, int(cmd_seq))),
            client_ts=time.time(),
        )

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

    def _subscribe_state(self) -> None:
        import roslibpy

        if self._state_topic is not None:
            return
        topic = roslibpy.Topic(
            self._require_ros(),
            "/hybrid_controller/state",
            "hybrid_controller_ros/RobotState",
            queue_length=1,
        )

        def _callback(message: dict[str, object]) -> None:
            with self._state_lock:
                self._latest_state = dict(message)
                self._latest_state_local_ts = time.perf_counter()
                self._state_ready.set()

        topic.subscribe(_callback)
        self._state_topic = topic


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
        read_timeout_sec = max(0.1, float(config.vision_read_timeout_ms) / 1000.0)
        try:
            return _HttpMjpegCapture(
                str(source),
                cv2_module=cv2_module,
                timeout_sec=timeout_sec,
                read_timeout_sec=read_timeout_sec,
            )
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
        read_timeout_sec = max(0.1, float(config.vision_read_timeout_ms) / 1000.0)
        return _HttpMjpegCapture(
            str(source),
            cv2_module=cv2_module,
            timeout_sec=timeout_sec,
            read_timeout_sec=read_timeout_sec,
        )
    return _open_capture(cv2_module, stream_url, config)


def _read_frames_from_capture(
    *,
    capture: object,
    frame_count: int,
    drain_frames: int,
    timeout_sec: float,
    latest_window_sec: float = 0.0,
) -> list[tuple[object, float]]:
    frames: list[tuple[object, float]] = []
    deadline = time.perf_counter() + max(0.5, float(timeout_sec))
    drain_remaining = max(0, int(drain_frames))
    target_count = max(1, int(frame_count))
    latest_until: float | None = None
    latest_window = max(0.0, float(latest_window_sec))
    while time.perf_counter() <= deadline:
        if latest_until is not None and time.perf_counter() >= latest_until:
            break
        try:
            ok, frame = capture.read()  # type: ignore[attr-defined]
        except Exception:
            ok, frame = False, None
        now = time.perf_counter()
        if not ok or frame is None:
            time.sleep(0.03)
            continue
        if _frame_has_horizontal_tearing(frame):
            continue
        if drain_remaining > 0:
            drain_remaining -= 1
            continue
        frames.append((frame, now))
        if len(frames) > target_count:
            del frames[:-target_count]
        if latest_window > 0.0 and latest_until is None:
            latest_until = min(deadline, now + latest_window)
        if latest_window <= 0.0 and len(frames) >= target_count:
            break
    if not frames:
        stats_func = getattr(capture, "stats", None)
        if callable(stats_func):
            try:
                stats = stats_func()
            except Exception:
                stats = None
            if isinstance(stats, Mapping):
                reason = str(stats.get("last_reject_reason", "")).strip()
                read_error = str(stats.get("last_read_error", "")).strip()
                rejected = stats.get("frames_rejected")
                accepted = stats.get("frames_accepted")
                details = []
                if reason:
                    details.append(f"last_reject={reason}")
                if read_error:
                    details.append(f"last_read_error={read_error}")
                if rejected is not None:
                    details.append(f"frames_rejected={rejected}")
                if accepted is not None:
                    details.append(f"frames_accepted={accepted}")
                if details:
                    raise RuntimeError("Timed out waiting for camera frames (" + ", ".join(details) + ").")
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
            latest_window_sec=0.0,
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
        self._last_transport_stats: dict[str, object] = {}

    @property
    def stream_url(self) -> str | None:
        return self._stream_url

    def transport_stats(self) -> dict[str, object]:
        capture = self._capture
        if capture is None:
            if self._last_transport_stats:
                stats = dict(self._last_transport_stats)
                stats.setdefault("stream_url", self._stream_url)
                stats.setdefault("capture_backend", self._capture_backend)
                stats["open"] = False
                return stats
            return {
                "stream_url": self._stream_url,
                "capture_backend": self._capture_backend,
                "open": False,
            }
        stats_func = getattr(capture, "stats", None)
        if callable(stats_func):
            try:
                stats = dict(stats_func())
            except Exception as error:
                stats = {"stats_error": str(error)}
        else:
            stats = {"reader": type(capture).__name__}
        stats.setdefault("stream_url", self._stream_url)
        stats.setdefault("capture_backend", self._capture_backend)
        stats["open"] = True
        return stats

    def read(
        self,
        *,
        frame_count: int,
        drain_frames: int,
        timeout_sec: float,
        latest_window_sec: float = 0.0,
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
                latest_window_sec=latest_window_sec,
            )
        except Exception:
            self.close()
            raise
        return self._stream_url, frames

    def reopen(self) -> None:
        """Reset only this PC-side stream consumer so the next read starts fresh."""
        self.close()
        self._ensure_open()

    def close(self) -> None:
        capture = self._capture
        self._last_transport_stats = self.transport_stats() if capture is not None else dict(self._last_transport_stats)
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


def _ibvs_jacobian_from_stage_profile(
    profile: VisionCalibrationProfile | None,
    *,
    theta_deg: float,
    radius_mm: float,
) -> tuple[float, float, float, float] | None:
    if profile is None:
        return None
    j_xy = _stage_profile_pixel_to_xy_jacobian(profile)
    if j_xy is None:
        return None
    theta_rad = math.radians(float(theta_deg))
    radius = float(radius_mm)
    dx_dtheta = -radius * math.sin(theta_rad) * math.pi / 180.0
    dy_dtheta = radius * math.cos(theta_rad) * math.pi / 180.0
    dx_dr = math.cos(theta_rad)
    dy_dr = math.sin(theta_rad)
    du_dx, du_dy, dv_dx, dv_dy = j_xy
    return (
        du_dx * dx_dtheta + du_dy * dy_dtheta,
        du_dx * dx_dr + du_dy * dy_dr,
        dv_dx * dx_dtheta + dv_dy * dy_dtheta,
        dv_dx * dx_dr + dv_dy * dy_dr,
    )


def _ibvs_jacobian_from_profile_for_snapshot(
    *,
    config: AppConfig,
    calibration_profile: VisionCalibrationProfile | None,
    snapshot: Mapping[str, object] | None,
) -> tuple[tuple[float, float, float, float], str] | None:
    pose = _current_cyl_pose(snapshot)
    if pose is None or calibration_profile is None:
        return None
    stage_name, stage_z = _current_calibration_stage(config, snapshot)
    try:
        active_stage_profile = calibration_profile.model_for_stage(
            stage_name,
            z_mm=stage_z,
            allow_fallback=True,
        )
    except Exception:
        active_stage_profile = calibration_profile
    stage_profile_z = active_stage_profile.z_mm
    if stage_profile_z is None:
        return None
    stage_band_mm = max(
        0.0,
        float(getattr(config, "vision_continuous_servo_ibvs_profile_stage_band_mm", 15.0)),
    )
    if stage_profile_z is not None and abs(float(pose[2]) - float(stage_profile_z)) > stage_band_mm:
        return None
    profile_jacobian = _ibvs_jacobian_from_stage_profile(
        active_stage_profile,
        theta_deg=float(pose[0]),
        radius_mm=float(pose[1]),
    )
    if profile_jacobian is None:
        return None
    source = f"profile_{stage_name}" if stage_profile_z is not None else "profile_global"
    return (profile_jacobian, source)


def _stage_profile_pixel_to_xy_jacobian(profile: VisionCalibrationProfile) -> tuple[float, float, float, float] | None:
    summary = profile.samples_summary if isinstance(profile.samples_summary, Mapping) else {}
    raw = summary.get("pixel_to_robot_jacobian") if isinstance(summary, Mapping) else None
    if isinstance(raw, (tuple, list)) and len(raw) >= 2:
        first = raw[0]
        second = raw[1]
        if isinstance(first, (tuple, list)) and isinstance(second, (tuple, list)) and len(first) >= 2 and len(second) >= 2:
            try:
                du_dx = float(first[0])
                dv_dx = float(first[1])
                du_dy = float(second[0])
                dv_dy = float(second[1])
            except (TypeError, ValueError):
                return None
            if all(math.isfinite(value) for value in (du_dx, du_dy, dv_dx, dv_dy)):
                return (du_dx, du_dy, dv_dx, dv_dy)
    matrix = profile.pixel_to_delta_matrix
    if matrix is None:
        return None
    try:
        inverse = np.linalg.pinv(np.asarray(matrix, dtype=np.float64)[:, :2])
    except Exception:
        return None
    if inverse.shape != (2, 2) or not np.all(np.isfinite(inverse)):
        return None
    # pixel_to_delta stores delta_xy = -J_xy^-1 * pixel_error, so J_xy = -inv(matrix).
    j_xy = -inverse
    return (float(j_xy[0, 0]), float(j_xy[0, 1]), float(j_xy[1, 0]), float(j_xy[1, 1]))


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
        "_local_receive_ts": _float("_local_receive_ts", 0.0),
        "busy": bool(message.get("busy", False)),
        "busy_action": str(message.get("busy_action", "")),
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
        "release_mode_effective": str(message.get("release_mode_effective", "")),
        "sucker_frozen": str(message.get("release_mode_effective", "")).strip().lower() == "sucker_frozen",
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


def _continuous_snapshot_blocks_teleop(snapshot: Mapping[str, object]) -> bool:
    state = str(snapshot.get("state", "")).strip().upper()
    busy_action = str(snapshot.get("busy_action", "")).strip().lower()
    if bool(snapshot.get("carrying", False)):
        return True
    if state.startswith("PICK") or state.startswith("PLACE") or state in {"ERROR", "RECOVERING"}:
        return True
    if state == "MOVING_XY" and busy_action != "teleop":
        return True
    if not bool(snapshot.get("busy", False)):
        return False
    return busy_action != "teleop"


def _current_cyl_pose(snapshot: Mapping[str, object] | None) -> tuple[float, float, float] | None:
    if not isinstance(snapshot, Mapping):
        return None
    cyl = snapshot.get("robot_cyl")
    if not isinstance(cyl, Mapping):
        return None
    try:
        pose = (
            float(cyl.get("theta_deg")),
            float(cyl.get("radius_mm")),
            float(cyl.get("z_mm", snapshot.get("robot_z", 0.0))),
        )
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in pose):
        return None
    return pose


def _cyl_horizontal_delta_mm(
    first_pose: tuple[float, float, float] | None,
    second_pose: tuple[float, float, float] | None,
) -> float | None:
    if first_pose is None or second_pose is None:
        return None
    try:
        first_theta, first_radius = float(first_pose[0]), float(first_pose[1])
        second_theta, second_radius = float(second_pose[0]), float(second_pose[1])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (first_theta, first_radius, second_theta, second_radius)):
        return None
    mean_radius = max(0.0, (first_radius + second_radius) * 0.5)
    arc_mm = math.radians(second_theta - first_theta) * mean_radius
    radial_mm = second_radius - first_radius
    return math.hypot(arc_mm, radial_mm)


def _motion_response_guard_update(
    *,
    config: AppConfig,
    anchor_pose: tuple[float, float, float] | None,
    anchor_px: tuple[float, float] | None,
    static_frames: int,
    current_pose: tuple[float, float, float] | None,
    current_px: tuple[float, float] | None,
) -> tuple[tuple[float, float, float] | None, tuple[float, float] | None, int, dict[str, object] | None]:
    if not bool(getattr(config, "vision_continuous_servo_camera_motion_guard_enabled", True)):
        return (current_pose, current_px, 0, None)
    if current_pose is None or current_px is None:
        return (None, None, 0, None)
    if anchor_pose is None or anchor_px is None:
        return (current_pose, current_px, 0, None)
    robot_delta = _cyl_horizontal_delta_mm(anchor_pose, current_pose)
    if robot_delta is None:
        return (current_pose, current_px, 0, None)
    pixel_delta = math.hypot(float(current_px[0]) - float(anchor_px[0]), float(current_px[1]) - float(anchor_px[1]))
    min_robot_mm = max(
        0.1,
        float(getattr(config, "vision_continuous_servo_camera_motion_guard_min_robot_mm", 8.0)),
    )
    max_pixel_px = max(
        0.1,
        float(getattr(config, "vision_continuous_servo_camera_motion_guard_max_pixel_px", 2.5)),
    )
    required_frames = max(
        1,
        int(getattr(config, "vision_continuous_servo_camera_motion_guard_static_frames", 5)),
    )
    if pixel_delta > max_pixel_px:
        return (current_pose, current_px, 0, None)
    if robot_delta < min_robot_mm:
        return (anchor_pose, anchor_px, 0, None)
    next_static_frames = max(0, int(static_frames)) + 1
    trace = {
        "reason": "camera_motion_response_missing",
        "robot_horizontal_delta_mm": float(robot_delta),
        "pixel_delta_px": float(pixel_delta),
        "anchor_pose_cyl": [float(anchor_pose[0]), float(anchor_pose[1]), float(anchor_pose[2])],
        "current_pose_cyl": [float(current_pose[0]), float(current_pose[1]), float(current_pose[2])],
        "anchor_px": [float(anchor_px[0]), float(anchor_px[1])],
        "current_px": [float(current_px[0]), float(current_px[1])],
        "static_frames": int(next_static_frames),
        "required_static_frames": int(required_frames),
        "min_robot_horizontal_delta_mm": float(min_robot_mm),
        "max_pixel_delta_px": float(max_pixel_px),
    }
    if next_static_frames >= required_frames:
        return (anchor_pose, anchor_px, next_static_frames, trace)
    return (anchor_pose, anchor_px, next_static_frames, None)


def _snapshot_local_age_ms(snapshot: Mapping[str, object] | None, *, now: float | None = None) -> float:
    if not isinstance(snapshot, Mapping):
        return float("inf")
    try:
        local_receive_ts = float(snapshot.get("_local_receive_ts", 0.0) or 0.0)
    except (TypeError, ValueError):
        local_receive_ts = 0.0
    if local_receive_ts <= 0.0:
        return float("inf")
    now_value = time.perf_counter() if now is None else float(now)
    return max(0.0, (now_value - local_receive_ts) * 1000.0)


def _fetch_fresh_state_snapshot(
    client: RosBridgeClient,
    *,
    timeout_sec: float,
    max_age_ms: float,
    poll_sec: float = 0.03,
) -> tuple[dict[str, object], float]:
    deadline = time.perf_counter() + max(0.1, float(timeout_sec))
    last_snapshot: dict[str, object] | None = None
    last_age_ms = float("inf")
    while time.perf_counter() <= deadline:
        message = client.fetch_state(timeout_sec=max(0.1, min(0.4, float(timeout_sec))))
        snapshot = _state_message_to_snapshot(message)
        age_ms = _snapshot_local_age_ms(snapshot)
        last_snapshot = snapshot
        last_age_ms = age_ms
        if math.isfinite(age_ms) and age_ms <= float(max_age_ms):
            return snapshot, float(age_ms)
        time.sleep(max(0.0, float(poll_sec)))
    if last_snapshot is not None:
        return last_snapshot, float(last_age_ms)
    raise TimeoutError("Timed out waiting for /hybrid_controller/state.")


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
    slots: list[SlotState] | None = None,
    device: str | None,
    half: bool,
) -> tuple[dict[str, object], object, int, list[SlotState]]:
    if slots is None:
        slots = [SlotState(slot=index + 1, freq_hz=config.ssvep_freqs[index]) for index in range(config.vision_max_targets)]
    top_mask_rows = int(getattr(config, "vision_frame_top_mask_rows", 0) or 0)
    bottom_mask_rows = int(getattr(config, "vision_frame_bottom_mask_rows", 0) or 0)
    prepared_frames: list[tuple[object, float]] = []
    effective_top_mask_rows = 0
    effective_bottom_mask_rows = 0
    for frame, capture_ts in frames:
        prepared_frame, masked_top_rows, masked_bottom_rows = sanitize_frame_edge_bands(
            frame,
            top_rows=top_mask_rows,
            bottom_rows=bottom_mask_rows,
        )
        effective_top_mask_rows = max(effective_top_mask_rows, int(masked_top_rows))
        effective_bottom_mask_rows = max(effective_bottom_mask_rows, int(masked_bottom_rows))
        prepared_frames.append((prepared_frame, capture_ts))
    frames = prepared_frames
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
        frame_quality = frame_brightness_quality(
            frame,
            min_mean=float(getattr(config, "vision_frame_min_brightness_mean", 30.0)),
            min_p95=float(getattr(config, "vision_frame_min_brightness_p95", 45.0)),
        )
        frame_quality["top_mask_rows"] = int(effective_top_mask_rows)
        frame_quality["bottom_mask_rows"] = int(effective_bottom_mask_rows)
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
            action_center_tolerance_px = float(
                getattr(config, "vision_servo_low_action_tolerance_px", action_center_tolerance_px)
            )
        low_height_shape_fallback = (
            bool(getattr(config, "vision_low_height_shape_fallback_enabled", True))
            and str(calibration_stage or "").strip().lower() in {"confirm", "pick"}
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
            prefer_frame_fallback=low_height_shape_fallback,
            fallback_min_area_ratio=float(getattr(config, "vision_low_height_shape_fallback_min_area_ratio", 1.20)),
            fallback_reject_edge_touch=bool(
                low_height_shape_fallback
                and getattr(config, "vision_low_height_reject_edge_fallback_candidates", True)
            ),
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
            servo_measurement_point=str(getattr(config, "vision_servo_measurement_point", "center")),
            low_height_servo_measurement_point=str(
                getattr(config, "vision_servo_low_height_measurement_point", "")
            ),
            low_height_confirm_z_mm=float(getattr(config, "vision_pick_confirm_z_mm", config.robot_approach_z)),
            low_height_guard_band_mm=float(getattr(config, "vision_continuous_servo_low_height_guard_band_mm", 30.0)),
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
            frame_quality=frame_quality,
        )
        packet["capture_batch_frames"] = int(len(frames))
        packet["capture_batch_duration_ms"] = float(batch_capture_duration_ms)
        packet["latest_frame_preprocess_age_ms"] = float(latest_frame_preprocess_age_ms)
        last_frame = frame
    if packet is None:
        raise RuntimeError("No packet was generated from captured frames.")
    return packet, last_frame, frame_id, slots


def _packet_frame_pose_age_ms(packet: Mapping[str, object]) -> float | None:
    ages: list[float] = []
    for key in ("stream_age_ms", "queue_age_ms"):
        value = packet.get(key)
        if value is None:
            continue
        try:
            age_ms = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(age_ms):
            ages.append(max(0.0, age_ms))
    if not ages:
        return None
    return max(ages)


def _snapshot_frame_pose_age_ms(
    snapshot: Mapping[str, object] | None,
    packet: Mapping[str, object],
) -> float | None:
    """Approximate how far the robot state sample is from the image capture time."""
    if not isinstance(snapshot, Mapping):
        return None
    try:
        state_local_ts = float(snapshot.get("_local_receive_ts", 0.0) or 0.0)
        capture_ts = float(packet.get("capture_ts", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if state_local_ts <= 0.0 or capture_ts <= 0.0:
        return None
    return abs(state_local_ts - capture_ts) * 1000.0


def _frame_pose_age_for_static_snapshot(
    snapshot: Mapping[str, object] | None,
    packet: Mapping[str, object],
) -> float | None:
    """Only compute pose/image age when the snapshot carries a local timestamp."""
    if not isinstance(snapshot, Mapping):
        return None
    try:
        capture_ts = float(packet.get("capture_ts", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if capture_ts <= 0.0:
        return None
    try:
        local_receive_ts = float(snapshot.get("_local_receive_ts", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if local_receive_ts <= 0.0:
        return None
    state = str(snapshot.get("state", "")).strip().upper()
    busy = bool(snapshot.get("busy", False))
    busy_action = str(snapshot.get("busy_action", "")).strip().lower()
    if state == "IDLE" and not busy and not busy_action:
        return None
    return _snapshot_frame_pose_age_ms(snapshot, packet)


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


def _point_pair(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        return None
    try:
        x_value = float(value[0])
        y_value = float(value[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x_value) and math.isfinite(y_value)):
        return None
    return (x_value, y_value)


def _slot_tracking_point(slot: Mapping[str, object]) -> tuple[float, float] | None:
    measurement_point = str(slot.get("measurement_point", "") or "")
    point = _point_pair(_point_for_measurement(slot, measurement_point))
    if point is not None:
        return point
    for key in (
        "geometry_center_f",
        "color_block_center_f",
        "top_face_center_f",
        "grasp_pixel_f",
        "pixel_center_f",
        "geometry_center",
        "color_block_center",
        "top_face_center",
        "grasp_pixel",
        "pixel_center",
    ):
        point = _point_pair(slot.get(key))
        if point is not None:
            return point
    return None


def _select_slot_by_previous_center(
    packet: Mapping[str, object],
    previous_center_px: object,
    *,
    max_distance_px: float = 95.0,
) -> tuple[dict[str, object] | None, float | None]:
    slots = packet.get("slots")
    previous = _point_pair(previous_center_px)
    if previous is None or not isinstance(slots, list):
        return None, None
    best_slot: dict[str, object] | None = None
    best_distance = float("inf")
    for slot_raw in slots:
        if not isinstance(slot_raw, Mapping):
            continue
        slot = dict(slot_raw)
        if not bool(slot.get("valid", False)) and str(slot.get("invalid_reason", "")) != "vision_servo_required":
            continue
        point = _slot_tracking_point(slot)
        if point is None:
            continue
        distance = math.hypot(point[0] - previous[0], point[1] - previous[1])
        if distance < best_distance:
            best_slot = slot
            best_distance = distance
    if best_slot is None or best_distance > float(max_distance_px):
        return None, None
    return best_slot, float(best_distance)


def _remap_pending_slot(
    pending: Mapping[str, object] | None,
    slot_id: int | None,
    *,
    relock_distance_px: float | None = None,
) -> dict[str, object] | None:
    if not isinstance(pending, Mapping):
        return None
    result = dict(pending)
    if slot_id is not None:
        result["slot_id"] = int(slot_id)
    if relock_distance_px is not None:
        result["target_relock_distance_px"] = float(relock_distance_px)
    return result


def _adaptive_local_block_center(
    frame_bgr: object,
    *,
    seed_px: tuple[float, float],
    crop_radius_px: int = 105,
    crop_center_px: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    candidate = _adaptive_local_block_candidate(
        frame_bgr,
        seed_px=seed_px,
        crop_radius_px=crop_radius_px,
        crop_center_px=crop_center_px,
    )
    center = None if candidate is None else _point_pair(candidate.get("center_px"))
    return None if center is None else center


def _candidate_point_for_measurement(
    candidate: Mapping[str, object],
    measurement_point: object,
) -> tuple[float, float] | None:
    mode = str(measurement_point or "").strip().lower()
    if mode in {"color_block", "color_block_subpixel"}:
        return _point_pair(candidate.get("color_center_px") or candidate.get("center_px"))
    if mode in {"top_face", "top_face_subpixel", "grasp", "grasp_subpixel"}:
        return _point_pair(
            candidate.get("top_face_center_px")
            or candidate.get("geometry_center_px")
            or candidate.get("bbox_center_px")
            or candidate.get("center_px")
        )
    return _point_pair(candidate.get("geometry_center_px") or candidate.get("bbox_center_px") or candidate.get("center_px"))


def _set_slot_point(
    slot: dict[str, object],
    float_key: str,
    int_key: str,
    point: tuple[float, float] | None,
) -> None:
    if point is None:
        return
    slot[float_key] = [float(point[0]), float(point[1])]
    slot[int_key] = [int(round(float(point[0]))), int(round(float(point[1])))]


def _adaptive_local_block_candidate(
    frame_bgr: object,
    *,
    seed_px: tuple[float, float],
    crop_radius_px: int = 105,
    crop_center_px: tuple[float, float] | None = None,
) -> dict[str, object] | None:
    import cv2

    if frame_bgr is None or not hasattr(frame_bgr, "shape"):
        return None
    frame_h, frame_w = frame_bgr.shape[:2]
    sx = max(0, min(frame_w - 1, int(round(float(seed_px[0])))))
    sy = max(0, min(frame_h - 1, int(round(float(seed_px[1])))))
    crop_center = _point_pair(crop_center_px) or (float(sx), float(sy))
    cx_crop = max(0, min(frame_w - 1, int(round(float(crop_center[0])))))
    cy_crop = max(0, min(frame_h - 1, int(round(float(crop_center[1])))))
    radius = max(32, int(crop_radius_px))
    x1 = max(0, cx_crop - radius)
    x2 = min(frame_w, cx_crop + radius + 1)
    y1 = max(0, cy_crop - radius)
    y2 = min(frame_h, cy_crop + radius + 1)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    if sx < x1 or sx >= x2 or sy < y1 or sy >= y2:
        return None
    roi = frame_bgr[y1:y2, x1:x2]
    seed = roi[sy - y1, sx - x1].astype(np.float32)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    value = hsv[:, :, 2]
    diff = np.linalg.norm(roi.astype(np.float32) - seed.reshape(1, 1, 3), axis=2)
    seed_mask = np.where(diff <= 48.0, 255, 0).astype(np.uint8)
    seed_mask = np.where(sat >= 35, seed_mask, 0).astype(np.uint8)
    color_mask = np.where((sat >= 55) & (value >= 35), 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    def _best_component(mask: object, *, prefer_seed_label: bool) -> tuple[float, int, object, object, object] | None:
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return None
        seed_label = int(labels[sy - y1, sx - x1])
        candidates: list[tuple[float, int]] = []
        roi_area = float((x2 - x1) * (y2 - y1))
        for label in range(1, count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area < 350.0:
                continue
            left = float(stats[label, cv2.CC_STAT_LEFT])
            top = float(stats[label, cv2.CC_STAT_TOP])
            width = float(stats[label, cv2.CC_STAT_WIDTH])
            height = float(stats[label, cv2.CC_STAT_HEIGHT])
            if width <= 3.0 or height <= 3.0:
                continue
            area_ratio = area / max(1.0, roi_area)
            if area_ratio > 0.70:
                continue
            fill_ratio = area / max(1.0, width * height)
            cx, cy = float(centroids[label][0]), float(centroids[label][1])
            dist = math.hypot(cx - (sx - x1), cy - (sy - y1))
            score = dist - min(area, 20000.0) / 20000.0 * 8.0
            if fill_ratio < 0.30:
                score += 15.0
            if prefer_seed_label and label == seed_label:
                score -= 20.0
            candidates.append((score, label))
        if not candidates:
            return None
        best_score, best_label = min(candidates, key=lambda item: item[0])
        return (float(best_score), int(best_label), labels, stats, centroids)

    component = _best_component(seed_mask, prefer_seed_label=True)
    if component is None:
        component = _best_component(color_mask, prefer_seed_label=False)
    if component is None:
        return None
    _, best_label, _labels, stats, centroids = component
    cx, cy = centroids[best_label]
    left = float(stats[best_label, cv2.CC_STAT_LEFT])
    top = float(stats[best_label, cv2.CC_STAT_TOP])
    width = float(stats[best_label, cv2.CC_STAT_WIDTH])
    height = float(stats[best_label, cv2.CC_STAT_HEIGHT])
    area = float(stats[best_label, cv2.CC_STAT_AREA])
    color_center = (float(cx) + float(x1), float(cy) + float(y1))
    bbox_center = (
        float(x1) + left + width / 2.0,
        float(y1) + top + height / 2.0,
    )
    component_mask = np.where(_labels == best_label, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_rect_center: tuple[float, float] | None = None
    min_rect_size: list[float] | None = None
    min_rect_angle: float | None = None
    if contours:
        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        min_rect_center = (float(rect[0][0]) + float(x1), float(rect[0][1]) + float(y1))
        min_rect_size = [float(rect[1][0]), float(rect[1][1])]
        min_rect_angle = float(rect[2])
    geometry_center = bbox_center
    return {
        "center_px": [float(color_center[0]), float(color_center[1])],
        "color_center_px": [float(color_center[0]), float(color_center[1])],
        "geometry_center_px": [float(geometry_center[0]), float(geometry_center[1])],
        "bbox_center_px": [float(bbox_center[0]), float(bbox_center[1])],
        "min_area_rect_center_px": (
            None if min_rect_center is None else [float(min_rect_center[0]), float(min_rect_center[1])]
        ),
        "top_face_center_px": [float(geometry_center[0]), float(geometry_center[1])],
        "grasp_center_px": [float(geometry_center[0]), float(geometry_center[1])],
        "area_px": float(area),
        "bbox": [
            float(x1) + left,
            float(y1) + top,
            float(x1) + left + width,
            float(y1) + top + height,
        ],
        "min_area_rect_size": min_rect_size,
        "min_area_rect_angle_deg": min_rect_angle,
        "seed_px": [float(sx), float(sy)],
        "crop_bbox": [int(x1), int(y1), int(x2), int(y2)],
    }


def _low_height_local_synthetic_slot(
    *,
    packet: dict[str, object],
    frame_bgr: object,
    pending: Mapping[str, object] | None,
    current_z_mm: float | None,
    confirm_z_mm: float,
    measurement_point: str,
    slot_id: int | None = None,
) -> dict[str, object] | None:
    if current_z_mm is None:
        return None
    try:
        if float(current_z_mm) > float(confirm_z_mm) + 30.0:
            return None
    except (TypeError, ValueError):
        return None
    target = _point_pair(packet.get("alignment_target_pixel")) or (320.0, 240.0)
    pending_center = _point_pair(pending.get("last_center_px")) if isinstance(pending, Mapping) else None
    def _neighbor_seeds(center: tuple[float, float]) -> list[tuple[float, float]]:
        offsets = (
            (0.0, 0.0),
            (24.0, 0.0),
            (-24.0, 0.0),
            (0.0, 24.0),
            (0.0, -24.0),
            (34.0, 0.0),
            (-34.0, 0.0),
            (24.0, 18.0),
            (24.0, -18.0),
            (-24.0, 18.0),
            (-24.0, -18.0),
        )
        return [(float(center[0]) + dx, float(center[1]) + dy) for dx, dy in offsets]

    seeds = _neighbor_seeds(pending_center) if pending_center is not None else []
    seeds.extend(_neighbor_seeds(target))
    deduped_seeds: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for seed in seeds:
        key = (int(round(seed[0])), int(round(seed[1])))
        if key in seen:
            continue
        seen.add(key)
        deduped_seeds.append(seed)
    seeds = deduped_seeds
    candidate: dict[str, object] | None = None
    best_score = float("inf")
    for seed in seeds:
        current = _adaptive_local_block_candidate(
            frame_bgr,
            seed_px=seed,
            crop_center_px=target,
            crop_radius_px=135,
        )
        if current is None:
            continue
        center = _point_pair(current.get("center_px"))
        if center is None:
            continue
        if pending_center is not None and math.hypot(center[0] - pending_center[0], center[1] - pending_center[1]) > 28.0:
            continue
        try:
            area = float(current.get("area_px"))
        except (TypeError, ValueError):
            area = 0.0
        distance = math.hypot(center[0] - target[0], center[1] - target[1])
        score = distance - min(max(area, 0.0), 30_000.0) / 30_000.0 * 1.5
        if score < best_score:
            best_score = float(score)
            candidate = current
    if candidate is None:
        return None
    center = _candidate_point_for_measurement(candidate, measurement_point)
    if center is None:
        return None
    if pending_center is not None and math.hypot(center[0] - pending_center[0], center[1] - pending_center[1]) > 28.0:
        return None
    effective_slot_id = slot_id
    if effective_slot_id is None and isinstance(pending, Mapping):
        try:
            effective_slot_id = int(pending.get("slot_id"))
        except (TypeError, ValueError):
            effective_slot_id = None
    if effective_slot_id is None:
        effective_slot_id = 1
    distance = float(math.hypot(center[0] - target[0], center[1] - target[1]))
    geometry_center = _point_pair(candidate.get("geometry_center_px")) or center
    color_center = _point_pair(candidate.get("color_center_px") or candidate.get("center_px")) or center
    top_face_center = _point_pair(candidate.get("top_face_center_px")) or geometry_center
    grasp_center = _point_pair(candidate.get("grasp_center_px")) or top_face_center
    pixel_center = geometry_center
    slot = {
        "slot_id": int(effective_slot_id),
        "slot": int(effective_slot_id),
        "valid": True,
        "observed": True,
        "actionable": False,
        "invalid_reason": "vision_servo_required",
        "servo_required": True,
        "measurement_point": str(measurement_point or "grasp_subpixel"),
        "alignment_target_pixel": [float(target[0]), float(target[1])],
        "center_distance_px": distance,
        "confidence": 0.35,
        "area_px": candidate.get("area_px"),
        "bbox": candidate.get("bbox"),
        "low_height_local_center_override": True,
        "low_height_local_synthetic_slot": True,
        "low_height_local_center_seed_px": candidate.get("seed_px"),
        "low_height_local_center_px": [float(center[0]), float(center[1])],
        "low_height_local_color_center_px": candidate.get("color_center_px") or candidate.get("center_px"),
        "low_height_local_geometry_center_px": candidate.get("geometry_center_px"),
        "low_height_local_bbox_center_px": candidate.get("bbox_center_px"),
        "low_height_local_min_area_rect_center_px": candidate.get("min_area_rect_center_px"),
        "low_height_local_crop_bbox": candidate.get("crop_bbox"),
        "low_height_local_min_area_rect_size": candidate.get("min_area_rect_size"),
        "low_height_local_min_area_rect_angle_deg": candidate.get("min_area_rect_angle_deg"),
    }
    _set_slot_point(slot, "pixel_center_f", "pixel_center", pixel_center)
    _set_slot_point(slot, "geometry_center_f", "geometry_center", geometry_center)
    _set_slot_point(slot, "grasp_pixel_f", "grasp_pixel", grasp_center)
    _set_slot_point(slot, "color_block_center_f", "color_block_center", color_center)
    _set_slot_point(slot, "top_face_center_f", "top_face_center", top_face_center)
    return slot


def _upsert_packet_slot(packet: dict[str, object], slot: Mapping[str, object] | None) -> None:
    if not isinstance(slot, Mapping):
        return
    slots = packet.get("slots")
    if not isinstance(slots, list):
        slots = []
        packet["slots"] = slots
    try:
        slot_id = int(slot.get("slot_id", slot.get("slot", 0)) or 0)
    except (TypeError, ValueError):
        slot_id = 0
    replacement = dict(slot)
    for index, existing in enumerate(slots):
        if not isinstance(existing, Mapping):
            continue
        try:
            existing_id = int(existing.get("slot_id", existing.get("slot", 0)) or 0)
        except (TypeError, ValueError):
            existing_id = -1
        if existing_id == slot_id:
            slots[index] = replacement
            return
    slots.append(replacement)


def _patch_low_height_local_center(
    *,
    packet: dict[str, object],
    frame_bgr: object,
    selected_slot: dict[str, object] | None,
    pending: Mapping[str, object] | None,
    current_z_mm: float | None,
    confirm_z_mm: float,
    measurement_point: str,
) -> dict[str, object] | None:
    if selected_slot is None or current_z_mm is None:
        return selected_slot
    try:
        if float(current_z_mm) > float(confirm_z_mm) + 30.0:
            return selected_slot
    except (TypeError, ValueError):
        return selected_slot
    target = _point_pair(selected_slot.get("alignment_target_pixel") or packet.get("alignment_target_pixel"))
    if target is None:
        target = (320.0, 240.0)
    seed = _point_pair(selected_slot.get("grasp_pixel_f") or selected_slot.get("geometry_center_f"))
    pending_center = None
    if isinstance(pending, Mapping):
        pending_center = _point_pair(pending.get("last_center_px"))
        seed = pending_center or seed
    frame_h, frame_w = frame_bgr.shape[:2] if hasattr(frame_bgr, "shape") else (480, 640)
    bbox = selected_slot.get("bbox")
    bbox_touches_edge = False
    bbox_area_ratio = 0.0
    if isinstance(bbox, (tuple, list)) and len(bbox) >= 4:
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
            bbox_touches_edge = x1 <= 2.0 or y1 <= 2.0 or x2 >= float(frame_w) - 2.0 or y2 >= float(frame_h) - 2.0
            bbox_area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, float(frame_w * frame_h))
        except (TypeError, ValueError):
            bbox_touches_edge = False
            bbox_area_ratio = 0.0
    try:
        selected_error = float(selected_slot.get("center_distance_px"))
    except (TypeError, ValueError):
        selected_error = float("inf")
    if pending_center is None and (bbox_touches_edge or bbox_area_ratio >= 0.28 or selected_error >= 45.0):
        seed = target
    if seed is None:
        seed = target
    local_candidate = _adaptive_local_block_candidate(
        frame_bgr,
        seed_px=seed,
        crop_center_px=target,
        crop_radius_px=135,
    )
    if local_candidate is None:
        return None if not bool(selected_slot.get("valid", False)) else selected_slot
    local_center = _candidate_point_for_measurement(local_candidate, measurement_point)
    if local_center is None:
        return None if not bool(selected_slot.get("valid", False)) else selected_slot
    if pending_center is not None and math.hypot(local_center[0] - pending_center[0], local_center[1] - pending_center[1]) > 28.0:
        return None if not bool(selected_slot.get("valid", False)) else selected_slot
    patched = dict(selected_slot)
    point_keys = {
        "grasp_subpixel": ("grasp_pixel_f", "grasp_pixel"),
        "grasp": ("grasp_pixel_f", "grasp_pixel"),
        "geometry_subpixel": ("geometry_center_f", "geometry_center"),
        "geometry": ("geometry_center_f", "geometry_center"),
        "color_block_subpixel": ("color_block_center_f", "color_block_center"),
        "color_block": ("color_block_center_f", "color_block_center"),
        "top_face_subpixel": ("top_face_center_f", "top_face_center"),
        "top_face": ("top_face_center_f", "top_face_center"),
    }
    float_key, int_key = point_keys.get(str(measurement_point), ("geometry_center_f", "geometry_center"))
    _set_slot_point(patched, "geometry_center_f", "geometry_center", _point_pair(local_candidate.get("geometry_center_px")))
    _set_slot_point(
        patched,
        "color_block_center_f",
        "color_block_center",
        _point_pair(local_candidate.get("color_center_px") or local_candidate.get("center_px")),
    )
    _set_slot_point(patched, "top_face_center_f", "top_face_center", _point_pair(local_candidate.get("top_face_center_px")))
    _set_slot_point(patched, "grasp_pixel_f", "grasp_pixel", _point_pair(local_candidate.get("grasp_center_px")))
    _set_slot_point(patched, float_key, int_key, local_center)
    patched["measurement_point"] = str(measurement_point)
    patched["center_distance_px"] = float(math.hypot(local_center[0] - target[0], local_center[1] - target[1]))
    patched["valid"] = True
    patched["observed"] = True
    patched["servo_required"] = True
    patched["actionable"] = False
    patched["invalid_reason"] = "vision_servo_required"
    patched["low_height_local_center_override"] = True
    patched["low_height_local_center_seed_px"] = [float(seed[0]), float(seed[1])]
    patched["low_height_local_center_px"] = [float(local_center[0]), float(local_center[1])]
    patched["low_height_local_color_center_px"] = local_candidate.get("color_center_px") or local_candidate.get("center_px")
    patched["low_height_local_geometry_center_px"] = local_candidate.get("geometry_center_px")
    patched["low_height_local_bbox_center_px"] = local_candidate.get("bbox_center_px")
    patched["low_height_local_min_area_rect_center_px"] = local_candidate.get("min_area_rect_center_px")
    patched["low_height_local_crop_bbox"] = local_candidate.get("crop_bbox")
    patched["low_height_local_min_area_rect_size"] = local_candidate.get("min_area_rect_size")
    patched["low_height_local_min_area_rect_angle_deg"] = local_candidate.get("min_area_rect_angle_deg")
    return patched


def _pixel_distance_to_target(point: object, target: object) -> float | None:
    if not isinstance(point, (tuple, list)) or len(point) < 2:
        return None
    if not isinstance(target, (tuple, list)) or len(target) < 2:
        return None
    try:
        px = float(point[0])
        py = float(point[1])
        tx = float(target[0])
        ty = float(target[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(tx) and math.isfinite(ty)):
        return None
    return float(math.hypot(px - tx, py - ty))


def _point_for_measurement(payload: Mapping[str, object], measurement_point: object) -> object:
    mode = str(measurement_point or "").strip().lower()
    if mode == "color_block_subpixel":
        return payload.get("color_block_center_f")
    if mode == "color_block":
        return payload.get("color_block_center")
    if mode == "top_face_subpixel":
        return payload.get("top_face_center_f") or payload.get("top_face_center")
    if mode == "top_face":
        return payload.get("top_face_center") or payload.get("top_face_center_f")
    if mode == "grasp_subpixel":
        return payload.get("grasp_pixel_f") or payload.get("grasp_pixel")
    if mode == "grasp":
        return payload.get("grasp_pixel") or payload.get("grasp_pixel_f")
    if mode == "geometry_subpixel":
        return payload.get("geometry_center_f") or payload.get("geometry_center")
    if mode == "geometry":
        return payload.get("geometry_center") or payload.get("geometry_center_f")
    if mode == "center_subpixel":
        return payload.get("pixel_center_f") or payload.get("pixel_center")
    return payload.get("pixel_center") or payload.get("pixel_center_f")


def _slot_alignment_provenance(
    slot: Mapping[str, object] | None,
    packet: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not isinstance(slot, Mapping):
        return {
            "measurement_point": "",
            "alignment_target_pixel": None,
            "point_distances_px": {},
        }
    target = slot.get("alignment_target_pixel")
    if target is None and isinstance(packet, Mapping):
        target = packet.get("alignment_target_pixel")
    distances: dict[str, object] = {}
    for key in (
        "pixel_center_f",
        "pixel_center",
        "geometry_center_f",
        "geometry_center",
        "color_block_center_f",
        "color_block_center",
        "top_face_center_f",
        "top_face_center",
        "grasp_pixel_f",
        "grasp_pixel",
    ):
        distance = _pixel_distance_to_target(slot.get(key), target)
        if distance is not None:
            distances[key] = distance
    return {
        "measurement_point": str(slot.get("measurement_point", "")),
        "alignment_target_pixel": target,
        "point_distances_px": distances,
    }


def _continuous_slot_id_for_selection(args_slot_id: int | None, pending: Mapping[str, object] | None) -> int | None:
    if args_slot_id is not None:
        return int(args_slot_id)
    if not isinstance(pending, Mapping):
        return None
    try:
        return int(pending.get("slot_id"))
    except (TypeError, ValueError):
        return None


def _decision_for_packet(
    *,
    packet: Mapping[str, object],
    config: AppConfig,
    snapshot: Mapping[str, object] | None,
    selected_slot: Mapping[str, object] | None,
    pending: Mapping[str, object] | None = None,
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
        pending=pending,
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


def _continuous_decision_for_packet(
    *,
    packet: Mapping[str, object],
    config: AppConfig,
    snapshot: Mapping[str, object] | None,
    selected_slot: Mapping[str, object] | None,
    slot_id: int | None = None,
    pending: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if selected_slot is None:
        pending_slot_id = None
        if isinstance(pending, Mapping):
            try:
                pending_slot_id = int(pending.get("slot_id"))
            except (TypeError, ValueError):
                pending_slot_id = None
        effective_slot_id = int(slot_id if slot_id is not None else pending_slot_id) if (slot_id is not None or pending_slot_id is not None) else None
        if effective_slot_id is not None:
            decision = ContinuousVisionServoController(config).decide(
                slot_id=effective_slot_id,
                slot_payload=None,
                packet=packet,
                pending=pending,
                current_cyl_pose=_current_cyl_pose(snapshot),
                frame_pose_age_ms=_packet_frame_pose_age_ms(packet),
            )
            return {
                "action": decision.action,
                "state": decision.action,
                "status": decision.status,
                "reason": decision.reason,
                "command": None,
                "raw_command": None,
                "pending": decision.pending_dict,
                "theta_rate_deg_s": float(decision.theta_rate_deg_s),
                "radius_rate_mm_s": float(decision.radius_rate_mm_s),
                "z_rate_mm_s": float(decision.z_rate_mm_s),
                "trace": dict(decision.trace),
            }
        return {
            "action": "STOP",
            "state": "FAILED",
            "status": "no_valid_slot",
            "reason": "no_valid_slot",
            "command": None,
            "pending": pending,
            "trace": {},
        }
    slot_id = int(selected_slot.get("slot_id", selected_slot.get("slot", 0)))
    decision = ContinuousVisionServoController(config).decide(
        slot_id=slot_id,
        slot_payload=selected_slot,
        packet=packet,
        pending=pending,
        current_cyl_pose=_current_cyl_pose(snapshot),
        frame_pose_age_ms=_packet_frame_pose_age_ms(packet),
    )
    command = _rewrite_final_pick_command_for_debug(config=config, command=decision.command)
    return {
        "action": decision.action,
        "state": decision.action,
        "status": decision.status,
        "reason": decision.reason,
        "command": command,
        "raw_command": decision.command,
        "pending": decision.pending_dict,
        "theta_rate_deg_s": float(decision.theta_rate_deg_s),
        "radius_rate_mm_s": float(decision.radius_rate_mm_s),
        "z_rate_mm_s": float(decision.z_rate_mm_s),
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
            mask_center = slot.get("mask_center")
            if isinstance(mask_center, (list, tuple)) and len(mask_center) >= 2:
                mask_point = (
                    int(round(float(mask_center[0]))),
                    int(round(float(mask_center[1]))),
                )
                cv2_module.drawMarker(overlay, mask_point, (160, 160, 160), cv2_module.MARKER_CROSS, 18, 1)
            geometry_center = slot.get("geometry_center")
            if isinstance(geometry_center, (list, tuple)) and len(geometry_center) >= 2:
                geometry_point = (
                    int(round(float(geometry_center[0]))),
                    int(round(float(geometry_center[1]))),
                )
                cv2_module.drawMarker(
                    overlay, geometry_point, (255, 160, 0), cv2_module.MARKER_DIAMOND, 24, 2
                )
                cv2_module.circle(overlay, geometry_point, 6, (255, 160, 0), 1)
            top_face = slot.get("top_face_center")
            if isinstance(top_face, (list, tuple)) and len(top_face) >= 2:
                top_point = (
                    int(round(float(top_face[0]))),
                    int(round(float(top_face[1]))),
                )
                cv2_module.drawMarker(overlay, top_point, (0, 255, 180), cv2_module.MARKER_TILTED_CROSS, 26, 2)
                cv2_module.circle(overlay, top_point, 7, (0, 255, 180), 1)
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


def _servo_command_point_from_slot(slot: Mapping[str, object] | None) -> tuple[float, float] | None:
    if not isinstance(slot, Mapping):
        return None
    point = slot.get("servo_command_point")
    if not isinstance(point, (tuple, list)) or len(point) < 2:
        return None
    try:
        theta = float(point[0])
        radius = float(point[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(theta) and math.isfinite(radius)):
        return None
    return (theta, radius)


def _clamp_refine_target(
    *,
    current_pose: tuple[float, float, float],
    target_theta: float,
    target_radius: float,
    max_theta_step_deg: float,
    max_radius_step_mm: float,
    step_gain: float = 1.0,
) -> tuple[float, float, float]:
    theta_limit = abs(float(max_theta_step_deg))
    radius_limit = abs(float(max_radius_step_mm))
    gain = max(0.01, min(1.0, float(step_gain)))
    theta_delta = (float(target_theta) - float(current_pose[0])) * gain
    radius_delta = (float(target_radius) - float(current_pose[1])) * gain
    theta_delta = max(-theta_limit, min(theta_limit, theta_delta))
    radius_delta = max(-radius_limit, min(radius_limit, radius_delta))
    return (
        float(current_pose[0]) + float(theta_delta),
        float(current_pose[1]) + float(radius_delta),
        float(current_pose[2]),
    )


def _continuous_stopped_motion_target(
    *,
    current_pose: tuple[float, float, float] | None,
    selected_slot: Mapping[str, object] | None,
    center_distance_px: float | None,
    center_allow_px: float | None,
    z_rate_mm_s: float,
    confirm_z_mm: float | None,
    z_tolerance_mm: float,
    z_step_mm: float,
    refine_z_band_above_confirm_mm: float,
    max_theta_step_deg: float,
    max_radius_step_mm: float,
) -> tuple[str, tuple[float, float, float], dict[str, object]] | None:
    if current_pose is None or confirm_z_mm is None:
        return None
    try:
        current_theta, current_radius, current_z = (
            float(current_pose[0]),
            float(current_pose[1]),
            float(current_pose[2]),
        )
        confirm_z = float(confirm_z_mm)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (current_theta, current_radius, current_z, confirm_z)):
        return None

    z_tolerance = max(0.0, float(z_tolerance_mm))
    center_distance = float("inf") if center_distance_px is None else float(center_distance_px)
    center_allow = 0.0 if center_allow_px is None else max(0.0, float(center_allow_px))

    if center_distance > center_allow and current_z <= confirm_z + max(0.0, float(refine_z_band_above_confirm_mm)):
        refine_point = _servo_command_point_from_slot(selected_slot)
        if refine_point is None:
            return None
        target = _clamp_refine_target(
            current_pose=current_pose,
            target_theta=float(refine_point[0]),
            target_radius=float(refine_point[1]),
            max_theta_step_deg=max_theta_step_deg,
            max_radius_step_mm=max_radius_step_mm,
        )
        if (
            abs(float(target[0]) - current_theta) < 1e-6
            and abs(float(target[1]) - current_radius) < 1e-6
        ):
            return None
        return (
            "stopped_horizontal_refine",
            target,
            {
                "center_distance_px": float(center_distance),
                "center_allow_px": float(center_allow),
                "source_point": [float(refine_point[0]), float(refine_point[1])],
                "max_theta_step_deg": float(abs(float(max_theta_step_deg))),
                "max_radius_step_mm": float(abs(float(max_radius_step_mm))),
            },
        )

    if float(z_rate_mm_s) < -1e-6 and current_z > confirm_z + z_tolerance:
        z_delta = min(max(0.1, float(z_step_mm)), max(0.0, current_z - confirm_z))
        if z_delta <= 0.0:
            return None
        return (
            "stopped_descent_step",
            (current_theta, current_radius, current_z - z_delta),
            {
                "center_distance_px": None if not math.isfinite(center_distance) else float(center_distance),
                "center_allow_px": float(center_allow),
                "z_step_mm": float(z_delta),
                "z_rate_before_stopped_move_mm_s": float(z_rate_mm_s),
            },
        )

    return None


def _wait_for_idle(
    *,
    client: RosBridgeClient,
    timeout_sec: float,
    poll_sec: float = 0.4,
    min_state_seq: int | None = None,
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
        state = str(last_snapshot.get("state", "")).strip().upper()
        if min_state_seq is not None:
            try:
                state_seq = int(last_snapshot.get("state_seq", 0) or 0)
            except (TypeError, ValueError):
                state_seq = 0
            if state_seq <= int(min_state_seq):
                time.sleep(max(0.05, float(poll_sec)))
                continue
        if (
            state in {"IDLE", "CARRY_READY"}
            and not bool(last_snapshot.get("busy", False))
            and not str(last_snapshot.get("last_error_code", "")).strip()
        ):
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
                "geometry_center": slot.get("geometry_center"),
                "geometry_center_f": slot.get("geometry_center_f"),
                "color_block_center": slot.get("color_block_center"),
                "color_block_center_f": slot.get("color_block_center_f"),
                "top_face_center": slot.get("top_face_center"),
                "top_face_center_f": slot.get("top_face_center_f"),
                "grasp_pixel_f": slot.get("grasp_pixel_f"),
                "alignment_target_pixel": slot.get("alignment_target_pixel") or packet.get("alignment_target_pixel"),
                "measurement_point": slot.get("measurement_point"),
                "point_distances_px": _slot_alignment_provenance(slot, packet).get("point_distances_px"),
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


def _continuous_confirm_recheck(
    *,
    args: argparse.Namespace,
    config: AppConfig,
    cv2_module: object,
    model: object | None,
    calibration_profile: VisionCalibrationProfile | None,
    reader: _PersistentCaptureReader,
    client: RosBridgeClient,
    output_dir: Path,
    slot_id: int | None,
    frame_id: int,
    debug_slots: list[SlotState] | None,
    device: str | None,
    half: bool,
) -> tuple[dict[str, object], int, list[SlotState] | None]:
    settle_sec = max(0.0, float(getattr(args, "continuous_confirm_recheck_settle_sec", 1.5)))
    repeats = max(1, int(getattr(args, "continuous_confirm_recheck_repeats", 3)))
    frames = max(1, int(args.frames))
    drain_frames = max(0, int(args.drain_frames))
    snapshot = _wait_for_idle(
        client=client,
        timeout_sec=max(float(args.command_timeout_sec), float(args.ros_timeout_sec)),
    )
    time.sleep(settle_sec)
    try:
        snapshot = _state_message_to_snapshot(client.fetch_state(timeout_sec=float(args.ros_timeout_sec)))
    except Exception:
        pass
    reader.reopen()
    samples: list[dict[str, object]] = []
    next_frame_id = int(frame_id)
    next_debug_slots = debug_slots
    packet_for_overlay: dict[str, object] | None = None
    frame_for_overlay = None
    for repeat_index in range(1, repeats + 1):
        if snapshot is None:
            snapshot_for_stage = None
            snapshot_age_ms = float("inf")
        else:
            snapshot_for_stage = snapshot
            snapshot_age_ms = 0.0
        _, captured = reader.read(
            frame_count=frames,
            drain_frames=drain_frames,
            timeout_sec=float(args.timeout_sec),
            latest_window_sec=max(0.0, float(getattr(args, "continuous_latest_window_ms", 0.0))) / 1000.0,
        )
        process_frames = _select_latest_frames(captured, int(args.process_latest_frames))
        packet, last_frame, next_frame_id, next_debug_slots = _process_frame_batch(
            frames=process_frames,
            model=model,
            config=config,
            calibration_profile=calibration_profile,
            snapshot_for_stage=snapshot_for_stage,
            frame_id_start=next_frame_id,
            slots=next_debug_slots,
            device=device,
            half=half,
        )
        packet["camera_frames_captured"] = int(len(captured))
        packet["camera_frames_processed"] = int(len(process_frames))
        resolved_packet = _resolve_packet(
            packet=packet,
            config=config,
            snapshot=snapshot_for_stage,
            snapshot_age_ms=snapshot_age_ms,
            frame_pose_age_ms=_frame_pose_age_for_static_snapshot(snapshot_for_stage, packet),
        )
        resolved_packet["camera_frames_captured"] = int(len(captured))
        resolved_packet["camera_frames_processed"] = int(len(process_frames))
        selected_slot = _select_slot(resolved_packet, slot_id)
        pose_for_patch = _current_cyl_pose(snapshot_for_stage)
        selected_slot = _patch_low_height_local_center(
            packet=resolved_packet,
            frame_bgr=last_frame,
            selected_slot=selected_slot,
            pending=None,
            current_z_mm=None if pose_for_patch is None else float(pose_for_patch[2]),
            confirm_z_mm=float(config.vision_pick_confirm_z_mm),
            measurement_point=str(config.vision_servo_low_height_measurement_point or config.vision_servo_measurement_point),
        )
        if selected_slot is None:
            selected_slot = _low_height_local_synthetic_slot(
                packet=resolved_packet,
                frame_bgr=last_frame,
                pending=None,
                current_z_mm=None if pose_for_patch is None else float(pose_for_patch[2]),
                confirm_z_mm=float(config.vision_pick_confirm_z_mm),
                measurement_point=str(config.vision_servo_low_height_measurement_point or config.vision_servo_measurement_point),
                slot_id=slot_id,
            )
        _upsert_packet_slot(resolved_packet, selected_slot)
        alignment_provenance = _slot_alignment_provenance(selected_slot, resolved_packet)
        sample: dict[str, object] = {
            "repeat_index": int(repeat_index),
            "selected_slot": selected_slot,
            "center_distance_px": None if selected_slot is None else selected_slot.get("center_distance_px"),
            "pixel_center_f": None if selected_slot is None else selected_slot.get("pixel_center_f"),
            "geometry_center_f": None if selected_slot is None else selected_slot.get("geometry_center_f"),
            "color_block_center_f": None if selected_slot is None else selected_slot.get("color_block_center_f"),
            "top_face_center_f": None if selected_slot is None else selected_slot.get("top_face_center_f"),
            "grasp_pixel_f": None if selected_slot is None else selected_slot.get("grasp_pixel_f"),
            "measurement_point": alignment_provenance["measurement_point"],
            "alignment_target_pixel": alignment_provenance["alignment_target_pixel"],
            "point_distances_px": alignment_provenance["point_distances_px"],
            "bbox": None if selected_slot is None else selected_slot.get("bbox"),
            "area_px": None if selected_slot is None else selected_slot.get("area_px"),
            "actionable": False if selected_slot is None else bool(selected_slot.get("actionable", False)),
            "invalid_reason": "" if selected_slot is None else str(selected_slot.get("invalid_reason", "")),
            "frame_age_ms": _packet_frame_pose_age_ms(packet),
            "camera_transport": reader.transport_stats(),
        }
        samples.append(sample)
        packet_for_overlay = resolved_packet
        frame_for_overlay = last_frame
    distances: list[float] = []
    points: list[tuple[float, float]] = []
    for sample in samples:
        try:
            distance = float(sample.get("center_distance_px"))
        except (TypeError, ValueError):
            distance = float("nan")
        if math.isfinite(distance):
            distances.append(distance)
        point = _point_for_measurement(sample, sample.get("measurement_point"))
        if isinstance(point, (tuple, list)) and len(point) >= 2:
            try:
                x_value = float(point[0])
                y_value = float(point[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x_value) and math.isfinite(y_value):
                points.append((x_value, y_value))
    median_distance = float("inf")
    if distances:
        distances_sorted = sorted(distances)
        mid = len(distances_sorted) // 2
        median_distance = (
            distances_sorted[mid]
            if len(distances_sorted) % 2
            else 0.5 * (distances_sorted[mid - 1] + distances_sorted[mid])
        )
    repeat_spread_px = 0.0
    for left_index, left in enumerate(points):
        for right in points[left_index + 1 :]:
            repeat_spread_px = max(repeat_spread_px, math.hypot(left[0] - right[0], left[1] - right[1]))
    tolerance_px = max(0.1, float(getattr(config, "vision_continuous_servo_pick_ready_center_px", 2.0)))
    max_repeat_spread_px = max(0.1, float(getattr(args, "continuous_confirm_recheck_max_spread_px", 3.0)))
    passed = bool(math.isfinite(median_distance) and median_distance <= tolerance_px and repeat_spread_px <= max_repeat_spread_px)
    sample_measurement_points = [
        str(sample.get("measurement_point", "")).strip()
        for sample in samples
        if str(sample.get("measurement_point", "")).strip()
    ]
    recheck_measurement_point = (
        sample_measurement_points[-1]
        if sample_measurement_points
        else str(getattr(config, "vision_servo_low_height_measurement_point", "") or getattr(config, "vision_servo_measurement_point", ""))
    )
    recheck: dict[str, object] = {
        "settle_sec": settle_sec,
        "repeats": repeats,
        "median_center_distance_px": median_distance,
        "repeat_spread_px": repeat_spread_px,
        "tolerance_px": tolerance_px,
        "max_repeat_spread_px": max_repeat_spread_px,
        "passed": passed,
        "measurement_point": recheck_measurement_point,
        "sample_measurement_points": sample_measurement_points,
        "alignment_target_pixel": None if packet_for_overlay is None else packet_for_overlay.get("alignment_target_pixel"),
        "pose_cyl": None if snapshot is None else (None if _current_cyl_pose(snapshot) is None else list(_current_cyl_pose(snapshot))),  # type: ignore[arg-type]
        "samples": samples,
    }
    if packet_for_overlay is not None and frame_for_overlay is not None:
        step_dir = output_dir / "continuous_confirm_recheck"
        step_dir.mkdir(parents=True, exist_ok=True)
        raw_path = step_dir / "raw.jpg"
        overlay_path = step_dir / "overlay.jpg"
        packet_path = step_dir / "packet.json"
        cv2_module.imwrite(str(raw_path), frame_for_overlay)
        _save_overlay(
            cv2_module=cv2_module,
            frame=frame_for_overlay,
            packet=packet_for_overlay,
            selected_slot_id=slot_id,
            output_path=overlay_path,
        )
        _write_json(packet_path, packet_for_overlay)
        recheck.update(
            {
                "raw_image": str(raw_path),
                "overlay_image": str(overlay_path),
                "packet": str(packet_path),
            }
        )
    return recheck, next_frame_id, next_debug_slots


def _continuous_stop_reason_is_recoverable(
    reason: str,
    *,
    trace: Mapping[str, object] | None,
    pending: Mapping[str, object] | None,
    config: AppConfig,
    recoveries_used: int = 0,
) -> bool:
    if reason in {"hold", "lost_target_wait", "frame_stale_wait", "settle_near_center", "frame_too_dark"}:
        return True
    recoverable_low_height_reasons = {"low_height_error_rebounded", "low_height_best_error_rebounded"}
    if reason not in recoverable_low_height_reasons:
        return False
    if not isinstance(pending, Mapping) or not isinstance(trace, Mapping):
        return False
    try:
        current_z = float(trace.get("current_z_mm"))
        confirm_z = float(trace.get("confirm_z_mm"))
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(current_z) and math.isfinite(confirm_z)):
        return False
    recover_band_mm = max(
        float(getattr(config, "vision_pick_z_tolerance_mm", 4.0)),
        float(getattr(config, "vision_continuous_servo_low_height_rebound_recover_band_mm", 10.0)),
    )
    max_recoveries = max(
        0,
        int(getattr(config, "vision_continuous_servo_low_height_rebound_recover_attempts", 3)),
    )
    return current_z <= confirm_z + recover_band_mm and int(recoveries_used) < max_recoveries


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _camera_transport_has_motion_artifacts(stats: Mapping[str, object] | None) -> bool:
    if not isinstance(stats, Mapping):
        return False
    reason = str(stats.get("last_reject_reason", "")).strip().lower()
    if any(token in reason for token in ("horizontal_tearing", "temporal_splice", "invalid_jpeg", "decode_failed")):
        return True
    try:
        consecutive = int(stats.get("consecutive_rejected_frames", 0) or 0)
        rejected = int(stats.get("frames_rejected", 0) or 0)
    except (TypeError, ValueError):
        consecutive = 0
        rejected = 0
    if consecutive <= 0 and rejected <= 0:
        return False
    read_error = str(stats.get("last_read_error", "")).strip().lower()
    return read_error in {"timeout", "timed out"} or "timeout" in read_error


def _continuous_low_height_refine_gate(
    recheck: Mapping[str, object],
    *,
    pick_ready_center_px: float,
    max_repeat_spread_px: float,
    best_median_center_distance_px: float | None,
    min_improvement_px: float,
) -> tuple[bool, str, float | None]:
    median = _finite_float(recheck.get("median_center_distance_px"))
    spread = _finite_float(recheck.get("repeat_spread_px"))
    if bool(recheck.get("passed", False)):
        return False, "confirm_recheck_passed_no_pick", median
    if median is None:
        return False, "confirm_recheck_no_measurement", None
    if spread is not None and spread > max(0.1, float(max_repeat_spread_px)):
        return False, "confirm_recheck_unstable_no_refine", median
    if median <= max(0.1, float(pick_ready_center_px)):
        return False, "confirm_recheck_passed_no_pick", median
    if best_median_center_distance_px is not None:
        best = _finite_float(best_median_center_distance_px)
        if best is not None and median >= best - max(0.0, float(min_improvement_px)):
            return False, "low_height_refine_no_improvement", median
    return True, "refine_after_confirm_recheck", median


def _continuous_low_height_refine_requested(
    *,
    enabled: bool,
    stop_at_confirm: bool,
    current_z_mm: float | None,
    confirm_z_mm: float | None,
    center_distance_px: float | None,
    pick_ready_center_px: float,
    guard_band_mm: float,
) -> bool:
    if not (bool(enabled) and bool(stop_at_confirm)):
        return False
    if current_z_mm is None or confirm_z_mm is None or center_distance_px is None:
        return False
    try:
        current_z = float(current_z_mm)
        confirm_z = float(confirm_z_mm)
        center_distance = float(center_distance_px)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (current_z, confirm_z, center_distance)):
        return False
    if current_z < confirm_z:
        return False
    if current_z > confirm_z + max(0.0, float(guard_band_mm)):
        return False
    return center_distance > max(0.1, float(pick_ready_center_px))


def _pose_from_confirm_recheck(recheck: Mapping[str, object]) -> tuple[float, float, float] | None:
    raw_pose = recheck.get("pose_cyl")
    if not isinstance(raw_pose, (tuple, list)) or len(raw_pose) < 3:
        return None
    try:
        theta = float(raw_pose[0])
        radius = float(raw_pose[1])
        z_value = float(raw_pose[2])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (theta, radius, z_value)):
        return None
    return (theta, radius, z_value)


def _run_continuous_servo_flow(
    *,
    args: argparse.Namespace,
    config: AppConfig,
    cv2_module: object,
    model: object | None,
    calibration_profile: VisionCalibrationProfile | None,
    stream_candidates: tuple[str, ...],
    output_dir: Path,
    client: RosBridgeClient | None,
    initial_snapshot: dict[str, object] | None,
    ros_status: dict[str, object],
    report: dict[str, object],
    device: str | None,
    half: bool,
) -> int:
    report["servo_mode"] = "continuous"
    rate_hz = max(1.0, float(args.continuous_teleop_rate_hz))
    interval_sec = 1.0 / rate_hz
    save_every = max(0, int(args.continuous_save_every))
    metrics: dict[str, object] = {
        "teleop_rate_hz": rate_hz,
        "max_duration_sec": max(0.1, float(args.continuous_max_duration_sec)),
        "z_disabled": bool(args.continuous_disable_z),
        "stopped_step_mode": bool(args.continuous_stopped_step_mode),
        "auto_stopped_on_camera_artifacts": bool(args.continuous_auto_stopped_on_camera_artifacts),
        "motion_safe_stopped_step_mode": False,
        "drain_frames_first_loop": max(0, int(args.drain_frames)),
        "drain_frames_each_loop": max(0, int(args.continuous_drain_frames_each_loop)),
        "latest_window_ms": max(0.0, float(args.continuous_latest_window_ms)),
        "commands_sent": 0,
        "stop_count": 0,
        "stopped_move_count": 0,
        "min_center_distance_px": None,
        "final_center_distance_px": None,
        "max_center_distance_px": 0.0,
        "mean_frame_age_ms": None,
        "frame_age_samples": 0,
        "camera_read_timeouts": 0,
        "camera_artifact_recoveries": 0,
        "camera_transport": {},
        "low_height_rebound_recoveries": 0,
        "low_height_refine_gain": max(0.01, min(1.0, float(args.continuous_low_height_refine_gain))),
        "low_height_refine_best_median_center_distance_px": None,
        "low_height_refine_best_pose_cyl": None,
        "low_height_refine_rollback_recheck": None,
        "max_state_age_ms": 0.0,
        "final_pick_command": None,
        "final_pose_cyl": None,
        "final_z_mm": None,
        "final_stop_reason": "",
        "local_model_required": False,
        "confirm_reached": False,
        "confirm_center_distance_px": None,
        "confirm_pose_cyl": None,
        "locked_slot_id": None,
        "release_mode_effective": None,
    }
    report["continuous"] = metrics
    if client is None or not bool(ros_status.get("connected", False)):
        report["error"] = "continuous_servo_requires_ros"
        _write_json(output_dir / "debug_vision_grasp_flow.json", report)
        print("[guard] --servo-mode continuous requires an active ROS bridge connection.", file=sys.stderr)
        return 2

    try:
        client.advertise_teleop()
    except Exception as error:
        report["error"] = f"teleop_advertise_failed: {error}"
        _write_json(output_dir / "debug_vision_grasp_flow.json", report)
        print(f"[robot] Could not advertise teleop topic: {error}", file=sys.stderr)
        return 1

    reader = _PersistentCaptureReader(
        cv2_module=cv2_module,
        stream_urls=stream_candidates,
        config=config,
        capture_backend=str(args.capture_backend),
    )
    planner = RosTeleopPublishPlanner(keepalive_interval_sec=max(0.05, min(0.12, interval_sec)))
    frame_id = 0
    debug_slots: list[SlotState] | None = None
    locked_slot_id: int | None = int(args.slot_id) if args.slot_id is not None else None
    servo_pending: dict[str, object] | None = None
    frame_ages: list[float] = []
    cmd_seq = new_teleop_cmd_seq_base()
    teleop_published = False
    camera_read_timeouts = 0
    low_height_rebound_recoveries = 0
    low_refine_best_median_distance: float | None = None
    low_refine_best_pose: tuple[float, float, float] | None = None
    motion_response_anchor_pose: tuple[float, float, float] | None = None
    motion_response_anchor_px: tuple[float, float] | None = None
    motion_response_static_frames = 0
    motion_safe_stopped_step_mode = False
    exit_code = 0
    loop_index = 0
    started_at = time.perf_counter()
    deadline = started_at + max(0.1, float(args.continuous_max_duration_sec))
    low_refine_attempts = 0

    try:
        if bool(initial_snapshot and initial_snapshot.get("busy", False)):
            report["error"] = "robot_busy_before_continuous_servo"
            print("[robot] Robot is busy before continuous servo; no teleop command sent.", file=sys.stderr)
            return 2

        while time.perf_counter() <= deadline:
            loop_started = time.perf_counter()
            loop_index += 1
            try:
                snapshot, snapshot_age_ms = _fetch_fresh_state_snapshot(
                    client,
                    timeout_sec=float(args.ros_timeout_sec),
                    max_age_ms=float(config.vision_snapshot_max_age_ms),
                )
            except Exception as error:
                snapshot = None
                snapshot_age_ms = float("inf")
                print(f"[robot] Could not refresh state during continuous servo: {error}", file=sys.stderr)

            if isinstance(snapshot, Mapping):
                release_mode = str(snapshot.get("release_mode_effective", ""))
                metrics["release_mode_effective"] = release_mode
                if math.isfinite(snapshot_age_ms):
                    metrics["max_state_age_ms"] = max(float(metrics["max_state_age_ms"]), float(snapshot_age_ms))
                if str(snapshot.get("last_error_code", "")).strip():
                    metrics["final_stop_reason"] = "robot_error"
                    report["error"] = str(snapshot.get("last_error", snapshot.get("last_error_code", "")))
                    exit_code = 1
                    break
                if _continuous_snapshot_blocks_teleop(snapshot):
                    cmd_seq = next_teleop_cmd_seq(cmd_seq)
                    client.stop_teleop(use_auto_z=False, cmd_seq=cmd_seq)
                    teleop_published = True
                    metrics["commands_sent"] = int(metrics["commands_sent"]) + 1
                    metrics["stop_count"] = int(metrics["stop_count"]) + 1
                    metrics["final_stop_reason"] = "robot_busy"
                    report["error"] = "robot_busy"
                    exit_code = 1
                    break
            else:
                release_mode = ""
            max_state_age_ms = max(1.0, float(getattr(config, "vision_continuous_servo_command_timeout_ms", 250.0)))
            if not isinstance(snapshot, Mapping) or snapshot_age_ms > max_state_age_ms:
                cmd_seq = next_teleop_cmd_seq(cmd_seq)
                client.stop_teleop(use_auto_z=False, cmd_seq=cmd_seq)
                teleop_published = True
                metrics["commands_sent"] = int(metrics["commands_sent"]) + 1
                metrics["stop_count"] = int(metrics["stop_count"]) + 1
                metrics["final_stop_reason"] = "robot_state_stale"
                report["error"] = "robot_state_stale"
                exit_code = 1
                break

            loop_drain_frames = (
                max(0, int(args.drain_frames))
                if loop_index == 1
                else max(0, int(args.continuous_drain_frames_each_loop))
            )
            try:
                stream_url, frames = reader.read(
                    frame_count=int(args.frames),
                    drain_frames=loop_drain_frames,
                    timeout_sec=float(args.timeout_sec),
                    latest_window_sec=max(0.0, float(args.continuous_latest_window_ms)) / 1000.0,
                )
                camera_read_timeouts = 0
            except Exception as error:
                camera_read_timeouts += 1
                metrics["camera_read_timeouts"] = int(metrics["camera_read_timeouts"]) + 1
                camera_transport = reader.transport_stats()
                artifact_recovery = bool(args.continuous_auto_stopped_on_camera_artifacts) and _camera_transport_has_motion_artifacts(
                    camera_transport
                )
                if artifact_recovery:
                    metrics["camera_artifact_recoveries"] = int(metrics["camera_artifact_recoveries"]) + 1
                    motion_safe_stopped_step_mode = True
                    metrics["motion_safe_stopped_step_mode"] = True
                cmd_seq = next_teleop_cmd_seq(cmd_seq)
                client.stop_teleop(use_auto_z=False, cmd_seq=cmd_seq)
                teleop_published = True
                metrics["commands_sent"] = int(metrics["commands_sent"]) + 1
                metrics["stop_count"] = int(metrics["stop_count"]) + 1
                step_report = {
                    "step": loop_index,
                    "elapsed_sec": max(0.0, time.perf_counter() - started_at),
                    "decision": {
                        "action": "STOP",
                        "reason": "camera_read_timeout_wait",
                        "pending": servo_pending,
                        "trace": {
                            "camera_read_timeouts": int(camera_read_timeouts),
                            "drain_frames": int(loop_drain_frames),
                            "error": str(error),
                            "auto_stopped_step_mode_enabled": bool(artifact_recovery),
                        },
                    },
                    "snapshot": snapshot,
                    "pre_frame_snapshot": snapshot,
                    "camera_transport": camera_transport,
                }
                report["steps"].append(step_report)
                if camera_read_timeouts < max(1, int(getattr(config, "vision_continuous_servo_stale_frames", 3))):
                    try:
                        reader.reopen()
                    except Exception as reopen_error:
                        step_report["camera_reopen_error"] = str(reopen_error)
                    print(
                        "[continuous {step}] camera read timeout; stopped teleop and reopened PC reader "
                        "({count}){suffix}".format(
                            step=loop_index,
                            count=camera_read_timeouts,
                            suffix="; motion-safe stopped-step mode enabled" if artifact_recovery else "",
                        )
                    )
                    time.sleep(max(0.0, interval_sec - (time.perf_counter() - loop_started)))
                    continue
                metrics["final_stop_reason"] = "camera_read_timeout"
                report["error"] = str(error)
                exit_code = 1
                break
            process_frames = _select_latest_frames(frames, int(args.process_latest_frames))
            report["stream_url"] = stream_url
            frame_snapshot = snapshot
            frame_snapshot_age_ms = snapshot_age_ms
            try:
                state_for_frame, state_for_frame_age_ms = _fetch_fresh_state_snapshot(
                    client,
                    timeout_sec=float(args.ros_timeout_sec),
                    max_age_ms=float(config.vision_snapshot_max_age_ms),
                )
                if isinstance(state_for_frame, Mapping):
                    frame_snapshot = state_for_frame
                    frame_snapshot_age_ms = state_for_frame_age_ms
                    release_mode = str(state_for_frame.get("release_mode_effective", release_mode))
                    metrics["release_mode_effective"] = release_mode
                    if math.isfinite(state_for_frame_age_ms):
                        metrics["max_state_age_ms"] = max(
                            float(metrics["max_state_age_ms"]),
                            float(state_for_frame_age_ms),
                        )
            except Exception as error:
                print(f"[robot] Could not refresh frame-time state during continuous servo: {error}", file=sys.stderr)
            packet, last_frame, frame_id, debug_slots = _process_frame_batch(
                frames=process_frames,
                model=model,
                config=config,
                calibration_profile=calibration_profile,
                snapshot_for_stage=frame_snapshot,
                frame_id_start=frame_id,
                slots=debug_slots,
                device=device,
                half=half,
            )
            packet["camera_frames_captured"] = int(len(frames))
            packet["camera_frames_processed"] = int(len(process_frames))
            image_age_ms = _packet_frame_pose_age_ms(packet)
            if image_age_ms is not None:
                frame_ages.append(float(image_age_ms))
            resolve_snapshot = frame_snapshot
            resolve_snapshot_age_ms = frame_snapshot_age_ms
            frame_pose_age_ms = _snapshot_frame_pose_age_ms(frame_snapshot, packet)
            decision_snapshot = frame_snapshot
            decision_snapshot_age_ms = frame_snapshot_age_ms
            try:
                state_for_decision, state_for_decision_age_ms = _fetch_fresh_state_snapshot(
                    client,
                    timeout_sec=float(args.ros_timeout_sec),
                    max_age_ms=float(config.vision_snapshot_max_age_ms),
                )
                if isinstance(state_for_decision, Mapping):
                    decision_snapshot = state_for_decision
                    decision_snapshot_age_ms = state_for_decision_age_ms
                    release_mode = str(state_for_decision.get("release_mode_effective", release_mode))
                    metrics["release_mode_effective"] = release_mode
                    if math.isfinite(state_for_decision_age_ms):
                        metrics["max_state_age_ms"] = max(
                            float(metrics["max_state_age_ms"]),
                            float(state_for_decision_age_ms),
                        )
            except Exception as error:
                print(f"[robot] Could not refresh decision-time state during continuous servo: {error}", file=sys.stderr)
            resolved_packet = _resolve_packet(
                packet=packet,
                config=config,
                snapshot=resolve_snapshot,
                snapshot_age_ms=resolve_snapshot_age_ms,
                frame_pose_age_ms=frame_pose_age_ms,
            )
            resolved_packet["camera_frames_captured"] = int(len(frames))
            resolved_packet["camera_frames_processed"] = int(len(process_frames))
            camera_transport = reader.transport_stats()
            resolved_packet["camera_transport"] = dict(camera_transport)
            if frame_pose_age_ms is not None:
                resolved_packet["frame_pose_age_ms"] = float(frame_pose_age_ms)
            if image_age_ms is not None:
                resolved_packet["image_age_ms"] = float(image_age_ms)
            relock_distance_px: float | None = None
            selected_slot = None
            if isinstance(servo_pending, Mapping):
                selected_slot, relock_distance_px = _select_slot_by_previous_center(
                    resolved_packet,
                    servo_pending.get("last_center_px"),
                )
            selection_slot_id = _continuous_slot_id_for_selection(locked_slot_id, servo_pending)
            if selected_slot is None:
                selected_slot = _select_slot(resolved_packet, selection_slot_id)
            selected_slot_id = None if selected_slot is None else int(selected_slot.get("slot_id", selected_slot.get("slot", 0)))
            if locked_slot_id is None and selected_slot_id is not None:
                locked_slot_id = int(selected_slot_id)
                metrics["locked_slot_id"] = int(locked_slot_id)
            if selected_slot_id is not None and isinstance(servo_pending, Mapping):
                servo_pending = _remap_pending_slot(
                    servo_pending,
                    selected_slot_id,
                    relock_distance_px=relock_distance_px,
                )
                locked_slot_id = int(selected_slot_id)
                metrics["locked_slot_id"] = int(locked_slot_id)
            current_pose_for_patch = _current_cyl_pose(decision_snapshot)
            selected_slot = _patch_low_height_local_center(
                packet=resolved_packet,
                frame_bgr=last_frame,
                selected_slot=selected_slot,
                pending=servo_pending,
                current_z_mm=None if current_pose_for_patch is None else float(current_pose_for_patch[2]),
                confirm_z_mm=float(config.vision_pick_confirm_z_mm),
                measurement_point=str(config.vision_servo_low_height_measurement_point or config.vision_servo_measurement_point),
            )
            if selected_slot is None:
                selected_slot = _low_height_local_synthetic_slot(
                    packet=resolved_packet,
                    frame_bgr=last_frame,
                    pending=servo_pending,
                    current_z_mm=None if current_pose_for_patch is None else float(current_pose_for_patch[2]),
                    confirm_z_mm=float(config.vision_pick_confirm_z_mm),
                    measurement_point=str(config.vision_servo_low_height_measurement_point or config.vision_servo_measurement_point),
                    slot_id=selection_slot_id,
                )
            _upsert_packet_slot(resolved_packet, selected_slot)
            selected_slot_id = None if selected_slot is None else int(selected_slot.get("slot_id", selected_slot.get("slot", 0)))
            if selected_slot_id is not None and locked_slot_id is None:
                locked_slot_id = int(selected_slot_id)
                metrics["locked_slot_id"] = int(locked_slot_id)
            if selected_slot_id is not None and isinstance(servo_pending, Mapping):
                servo_pending = _remap_pending_slot(
                    servo_pending,
                    selected_slot_id,
                    relock_distance_px=relock_distance_px,
                )
                locked_slot_id = int(selected_slot_id)
                metrics["locked_slot_id"] = int(locked_slot_id)
            decision_config = config
            if str(config.vision_continuous_servo_horizontal_mode).strip().lower() == "ibvs_dls":
                profile_jacobian = _ibvs_jacobian_from_profile_for_snapshot(
                    config=config,
                    calibration_profile=calibration_profile,
                    snapshot=decision_snapshot,
                )
                if profile_jacobian is not None:
                    jacobian_values, jacobian_source = profile_jacobian
                    decision_config = replace(
                        config,
                        vision_continuous_servo_ibvs_profile_jacobian=jacobian_values,
                        vision_continuous_servo_ibvs_jacobian_source=jacobian_source,
                    ).resolved()
            decision = _continuous_decision_for_packet(
                packet=resolved_packet,
                config=decision_config,
                snapshot=decision_snapshot,
                selected_slot=selected_slot,
                slot_id=locked_slot_id,
                pending=servo_pending,
            )
            if bool(args.continuous_disable_z) and str(decision.get("action", "")) == "SERVO":
                original_z_rate = float(decision.get("z_rate_mm_s", 0.0))
                if abs(original_z_rate) > 1e-9:
                    decision = dict(decision)
                    trace = dict(decision.get("trace", {}) if isinstance(decision.get("trace"), Mapping) else {})
                    trace["z_rate_mm_s_before_debug_disable"] = original_z_rate
                    trace["z_disabled_by_debug_flag"] = True
                    decision["trace"] = trace
                    decision["z_rate_mm_s"] = 0.0
            if str(decision.get("action", "")) == "SERVO":
                guard_px = _slot_tracking_point(selected_slot) if selected_slot is not None else None
                guard_pose = _current_cyl_pose(decision_snapshot)
                motion_response_anchor_pose, motion_response_anchor_px, motion_response_static_frames, guard_trace = (
                    _motion_response_guard_update(
                        config=config,
                        anchor_pose=motion_response_anchor_pose,
                        anchor_px=motion_response_anchor_px,
                        static_frames=motion_response_static_frames,
                        current_pose=guard_pose,
                        current_px=guard_px,
                    )
                )
                if guard_trace is not None:
                    guarded_trace = dict(
                        decision.get("trace", {}) if isinstance(decision.get("trace"), Mapping) else {}
                    )
                    guarded_trace.update(guard_trace)
                    decision = dict(decision)
                    decision.update(
                        {
                            "action": "STOP",
                            "state": "STOP",
                            "status": "stopping because camera image did not respond to robot motion",
                            "reason": "camera_motion_response_missing",
                            "theta_rate_deg_s": 0.0,
                            "radius_rate_mm_s": 0.0,
                            "z_rate_mm_s": 0.0,
                            "trace": guarded_trace,
                        }
                    )
                    metrics["camera_motion_guard"] = dict(guard_trace)
            else:
                motion_response_anchor_pose = None
                motion_response_anchor_px = None
                motion_response_static_frames = 0
            servo_pending = decision.get("pending") if isinstance(decision.get("pending"), dict) else None
            if selected_slot is not None and selected_slot.get("center_distance_px") is not None:
                try:
                    current_center_distance = float(selected_slot.get("center_distance_px"))
                    metrics["max_center_distance_px"] = max(
                        float(metrics["max_center_distance_px"]),
                        current_center_distance,
                    )
                    metrics["final_center_distance_px"] = current_center_distance
                    if metrics["min_center_distance_px"] is None:
                        metrics["min_center_distance_px"] = current_center_distance
                    else:
                        metrics["min_center_distance_px"] = min(
                            float(metrics["min_center_distance_px"]),
                            current_center_distance,
                        )
                except (TypeError, ValueError):
                    pass
            current_pose_for_metrics = _current_cyl_pose(decision_snapshot)
            if current_pose_for_metrics is not None:
                metrics["final_pose_cyl"] = [
                    float(current_pose_for_metrics[0]),
                    float(current_pose_for_metrics[1]),
                    float(current_pose_for_metrics[2]),
                ]
                metrics["final_z_mm"] = float(current_pose_for_metrics[2])
            metrics["camera_transport"] = dict(camera_transport)
            if frame_ages:
                metrics["mean_frame_age_ms"] = sum(frame_ages) / float(len(frame_ages))
                metrics["frame_age_samples"] = int(len(frame_ages))

            step_report: dict[str, object] = {
                "step": loop_index,
                "elapsed_sec": max(0.0, time.perf_counter() - started_at),
                "drain_frames": int(loop_drain_frames),
                "camera_frames_captured": int(len(frames)),
                "camera_frames_processed": int(len(process_frames)),
                "slots": _slot_summary(resolved_packet),
                "selected_slot_id": selected_slot_id,
                "selected_slot": selected_slot,
                "decision": decision,
                "snapshot": decision_snapshot,
                "frame_snapshot": frame_snapshot,
                "pre_frame_snapshot": snapshot,
                "frame_pose_age_ms": frame_pose_age_ms,
                "image_age_ms": image_age_ms,
                "camera_transport": dict(camera_transport),
            }
            should_save = save_every > 0 and (
                loop_index == 1 or loop_index % save_every == 0 or str(decision.get("action")) != "SERVO"
            )
            if should_save:
                step_dir = output_dir / f"continuous_{loop_index:03d}"
                raw_path = step_dir / "raw.jpg"
                overlay_path = step_dir / "overlay.jpg"
                packet_path = step_dir / "packet.json"
                step_dir.mkdir(parents=True, exist_ok=True)
                cv2_module.imwrite(str(raw_path), last_frame)
                _save_overlay(
                    cv2_module=cv2_module,
                    frame=last_frame,
                    packet=resolved_packet,
                    selected_slot_id=selected_slot_id,
                    output_path=overlay_path,
                )
                _write_json(packet_path, resolved_packet)
                step_report.update(
                    {
                        "raw_image": str(raw_path),
                        "overlay_image": str(overlay_path),
                        "packet": str(packet_path),
                    }
                )
            report["steps"].append(step_report)

            action = str(decision.get("action", ""))
            reason = str(decision.get("reason", ""))
            if reason == "low_height_local_model_required":
                metrics["final_stop_reason"] = reason
                metrics["local_model_required"] = True
                print(
                    "[continuous {step}] low-height residual is static; stop and run local search/calibration "
                    "before continuing.".format(step=loop_index),
                    file=sys.stderr,
                )
            if action == "SERVO":
                trace = decision.get("trace", {}) if isinstance(decision.get("trace"), Mapping) else {}
                current_z = None
                confirm_z = None
                center_distance_px = None
                try:
                    current_z = float(trace.get("current_z_mm"))
                    confirm_z = float(trace.get("confirm_z_mm"))
                    center_distance_px = float(trace.get("center_distance_px"))
                except (TypeError, ValueError):
                    current_z = confirm_z = center_distance_px = None
                pick_ready_center_px = max(0.1, float(config.vision_continuous_servo_pick_ready_center_px))
                low_height_refine_requested = _continuous_low_height_refine_requested(
                    enabled=bool(args.continuous_low_height_discrete_refine),
                    stop_at_confirm=bool(args.continuous_stop_at_confirm),
                    current_z_mm=current_z,
                    confirm_z_mm=confirm_z,
                    center_distance_px=center_distance_px,
                    pick_ready_center_px=pick_ready_center_px,
                    guard_band_mm=float(config.vision_continuous_servo_low_height_guard_band_mm),
                )
                at_confirm_z_for_stop = (
                    bool(args.continuous_stop_at_confirm)
                    and current_z is not None
                    and confirm_z is not None
                    and abs(current_z - confirm_z) <= float(config.vision_pick_z_tolerance_mm)
                )
                if at_confirm_z_for_stop or low_height_refine_requested:
                    if center_distance_px is not None and center_distance_px <= pick_ready_center_px:
                        cmd_seq = next_teleop_cmd_seq(cmd_seq)
                        client.stop_teleop(use_auto_z=False, cmd_seq=cmd_seq)
                        teleop_published = True
                        metrics["commands_sent"] = int(metrics["commands_sent"]) + 1
                        metrics["stop_count"] = int(metrics["stop_count"]) + 1
                        metrics["final_stop_reason"] = "confirm_height_centered_no_pick"
                        metrics["confirm_reached"] = True
                        metrics["confirm_center_distance_px"] = center_distance_px
                        pose = _current_cyl_pose(decision_snapshot)
                        metrics["confirm_pose_cyl"] = (
                            None if pose is None else [float(pose[0]), float(pose[1]), float(pose[2])]
                        )
                        step_report["confirm_stop"] = {
                            "reason": "confirm_height_centered_no_pick",
                            "center_distance_px": center_distance_px,
                            "pose_cyl": metrics["confirm_pose_cyl"],
                        }
                        recheck, frame_id, debug_slots = _continuous_confirm_recheck(
                            args=args,
                            config=config,
                            cv2_module=cv2_module,
                            model=model,
                            calibration_profile=calibration_profile,
                            reader=reader,
                            client=client,
                            output_dir=output_dir,
                            slot_id=locked_slot_id,
                            frame_id=frame_id,
                            debug_slots=debug_slots,
                            device=device,
                            half=half,
                        )
                        step_report["confirm_recheck"] = recheck
                        metrics["confirm_recheck"] = recheck
                        metrics["confirm_center_distance_px"] = recheck.get(
                            "median_center_distance_px",
                            metrics["confirm_center_distance_px"],
                        )
                        if not bool(recheck.get("passed", False)):
                            metrics["final_stop_reason"] = "confirm_recheck_failed_no_pick"
                            step_report["confirm_stop"]["reason"] = "confirm_recheck_failed_no_pick"
                            exit_code = max(int(exit_code), 2)
                        print(
                            "[continuous {step}] confirm height centered; stopped before +r/PICK. "
                            "center={center} recheck={recheck}".format(
                                step=loop_index,
                                center="--" if center_distance_px is None else f"{center_distance_px:.2f}px",
                                recheck=(
                                    "--"
                                    if recheck.get("median_center_distance_px") is None
                                    else f"{float(recheck.get('median_center_distance_px')):.2f}px"
                                ),
                            )
                        )
                        break
                    step_report["confirm_continue"] = {
                        "reason": (
                            "low_height_guard_not_centered"
                            if low_height_refine_requested and not at_confirm_z_for_stop
                            else "confirm_height_not_centered"
                        ),
                        "center_distance_px": center_distance_px,
                        "pick_ready_center_px": pick_ready_center_px,
                        "current_z_mm": current_z,
                        "confirm_z_mm": confirm_z,
                    }
                    if bool(args.continuous_low_height_discrete_refine):
                        if bool(args.continuous_low_height_refine_recheck):
                            cmd_seq = next_teleop_cmd_seq(cmd_seq)
                            client.stop_teleop(use_auto_z=False, cmd_seq=cmd_seq)
                            teleop_published = True
                            metrics["commands_sent"] = int(metrics["commands_sent"]) + 1
                            metrics["stop_count"] = int(metrics["stop_count"]) + 1
                            recheck, frame_id, debug_slots = _continuous_confirm_recheck(
                                args=args,
                                config=config,
                                cv2_module=cv2_module,
                                model=model,
                                calibration_profile=calibration_profile,
                                reader=reader,
                                client=client,
                                output_dir=output_dir,
                                slot_id=locked_slot_id,
                                frame_id=frame_id,
                                debug_slots=debug_slots,
                                device=device,
                                half=half,
                            )
                            step_report["confirm_recheck"] = recheck
                            metrics["confirm_recheck"] = recheck
                            allowed, gate_reason, median_distance = _continuous_low_height_refine_gate(
                                recheck,
                                pick_ready_center_px=pick_ready_center_px,
                                max_repeat_spread_px=float(args.continuous_confirm_recheck_max_spread_px),
                                best_median_center_distance_px=low_refine_best_median_distance,
                                min_improvement_px=float(args.continuous_low_height_refine_min_improvement_px),
                            )
                            step_report["confirm_continue"]["reason"] = gate_reason
                            step_report["confirm_continue"]["median_center_distance_px"] = median_distance
                            if median_distance is not None:
                                recheck_pose = _pose_from_confirm_recheck(recheck)
                                if (
                                    recheck_pose is not None
                                    and (
                                    low_refine_best_median_distance is None
                                    or median_distance < low_refine_best_median_distance
                                    )
                                ):
                                    low_refine_best_median_distance = float(median_distance)
                                    low_refine_best_pose = recheck_pose
                                if low_refine_best_median_distance is not None:
                                    metrics["low_height_refine_best_median_center_distance_px"] = (
                                        float(low_refine_best_median_distance)
                                    )
                                metrics["low_height_refine_best_pose_cyl"] = (
                                    None
                                    if low_refine_best_pose is None
                                    else [
                                        float(low_refine_best_pose[0]),
                                        float(low_refine_best_pose[1]),
                                        float(low_refine_best_pose[2]),
                                    ]
                                )
                            if gate_reason == "confirm_recheck_passed_no_pick":
                                metrics["final_stop_reason"] = "confirm_height_centered_no_pick"
                                metrics["confirm_reached"] = True
                                metrics["confirm_center_distance_px"] = median_distance
                                pose = _pose_from_confirm_recheck(recheck)
                                if pose is None:
                                    pose = _current_cyl_pose(decision_snapshot)
                                metrics["confirm_pose_cyl"] = (
                                    None if pose is None else [float(pose[0]), float(pose[1]), float(pose[2])]
                                )
                                step_report["confirm_stop"] = {
                                    "reason": "confirm_height_centered_no_pick",
                                    "center_distance_px": median_distance,
                                    "pose_cyl": metrics["confirm_pose_cyl"],
                                    "confirm_recheck": recheck,
                                }
                                break
                            if not allowed:
                                if low_refine_best_pose is not None:
                                    cmd_seq = next_teleop_cmd_seq(cmd_seq)
                                    client.stop_teleop(use_auto_z=False, cmd_seq=cmd_seq)
                                    teleop_published = True
                                    metrics["commands_sent"] = int(metrics["commands_sent"]) + 1
                                    metrics["stop_count"] = int(metrics["stop_count"]) + 1
                                    rollback_response = client.call_service(
                                        "/hybrid_controller/move_cyl",
                                        "hybrid_controller_ros/MoveCyl",
                                        {
                                            "theta_deg": float(low_refine_best_pose[0]),
                                            "radius_mm": float(low_refine_best_pose[1]),
                                            "z_mm": float(low_refine_best_pose[2]),
                                        },
                                        timeout_sec=max(1.0, float(args.command_timeout_sec)),
                                    )
                                    settled = _wait_for_idle(
                                        client=client,
                                        timeout_sec=max(float(args.command_timeout_sec), float(args.ros_timeout_sec)),
                                    )
                                    time.sleep(max(0.0, float(args.continuous_confirm_recheck_settle_sec)))
                                    planner.reset()
                                    servo_pending = None
                                    step_report["low_height_refine_rollback"] = {
                                        "reason": gate_reason,
                                        "target_cyl": [
                                            float(low_refine_best_pose[0]),
                                            float(low_refine_best_pose[1]),
                                            float(low_refine_best_pose[2]),
                                        ],
                                        "best_median_center_distance_px": float(low_refine_best_median_distance),
                                        "response": rollback_response,
                                        "settled_snapshot": settled,
                                    }
                                    rollback_recheck, frame_id, debug_slots = _continuous_confirm_recheck(
                                        args=args,
                                        config=config,
                                        cv2_module=cv2_module,
                                        model=model,
                                        calibration_profile=calibration_profile,
                                        reader=reader,
                                        client=client,
                                        output_dir=output_dir,
                                        slot_id=locked_slot_id,
                                        frame_id=frame_id,
                                        debug_slots=debug_slots,
                                        device=device,
                                        half=half,
                                    )
                                    step_report["low_height_refine_rollback"]["recheck"] = rollback_recheck
                                    metrics["low_height_refine_rollback_recheck"] = rollback_recheck
                                    metrics["confirm_center_distance_px"] = rollback_recheck.get(
                                        "median_center_distance_px",
                                        metrics.get("confirm_center_distance_px"),
                                    )
                                    rollback_pose = _pose_from_confirm_recheck(rollback_recheck)
                                    if rollback_pose is not None:
                                        metrics["confirm_pose_cyl"] = [
                                            float(rollback_pose[0]),
                                            float(rollback_pose[1]),
                                            float(rollback_pose[2]),
                                        ]
                                    if bool(rollback_recheck.get("passed", False)):
                                        metrics["final_stop_reason"] = "confirm_height_centered_no_pick"
                                        metrics["confirm_reached"] = True
                                    else:
                                        metrics["final_stop_reason"] = gate_reason
                                        exit_code = max(int(exit_code), 2)
                                    metrics["low_height_refine_rollback_pose_cyl"] = [
                                        float(low_refine_best_pose[0]),
                                        float(low_refine_best_pose[1]),
                                        float(low_refine_best_pose[2]),
                                    ]
                                if not str(metrics.get("final_stop_reason", "")).strip():
                                    metrics["final_stop_reason"] = gate_reason
                                if gate_reason != "confirm_recheck_passed_no_pick":
                                    exit_code = max(int(exit_code), 2)
                                break
                        refine_point = _servo_command_point_from_slot(selected_slot)
                        if (
                            bool(args.continuous_low_height_refine_recheck)
                            and isinstance(step_report.get("confirm_recheck"), Mapping)
                        ):
                            samples = step_report["confirm_recheck"].get("samples", [])  # type: ignore[index]
                            if isinstance(samples, list):
                                for sample in reversed(samples):
                                    if not isinstance(sample, Mapping):
                                        continue
                                    sample_slot = sample.get("selected_slot")
                                    sample_point = _servo_command_point_from_slot(sample_slot)
                                    if sample_point is not None:
                                        refine_point = sample_point
                                        break
                        pose = _current_cyl_pose(decision_snapshot)
                        if (
                            bool(args.continuous_low_height_refine_recheck)
                            and isinstance(step_report.get("confirm_recheck"), Mapping)
                        ):
                            recheck_pose = _pose_from_confirm_recheck(step_report["confirm_recheck"])  # type: ignore[arg-type]
                            if recheck_pose is not None:
                                pose = recheck_pose
                        if refine_point is not None and pose is not None:
                            low_refine_attempts += 1
                            if low_refine_attempts > int(args.continuous_low_height_refine_attempts):
                                cmd_seq = next_teleop_cmd_seq(cmd_seq)
                                client.stop_teleop(use_auto_z=False, cmd_seq=cmd_seq)
                                teleop_published = True
                                metrics["commands_sent"] = int(metrics["commands_sent"]) + 1
                                metrics["stop_count"] = int(metrics["stop_count"]) + 1
                                metrics["final_stop_reason"] = "low_height_refine_attempt_limit"
                                step_report["confirm_continue"]["reason"] = "low_height_refine_attempt_limit"
                                break
                            refine_target = _clamp_refine_target(
                                current_pose=pose,
                                target_theta=float(refine_point[0]),
                                target_radius=float(refine_point[1]),
                                max_theta_step_deg=float(args.continuous_low_height_refine_max_theta_step_deg),
                                max_radius_step_mm=float(args.continuous_low_height_refine_max_radius_step_mm),
                                step_gain=float(args.continuous_low_height_refine_gain),
                            )
                            cmd_seq = next_teleop_cmd_seq(cmd_seq)
                            client.stop_teleop(use_auto_z=False, cmd_seq=cmd_seq)
                            teleop_published = True
                            metrics["commands_sent"] = int(metrics["commands_sent"]) + 1
                            metrics["stop_count"] = int(metrics["stop_count"]) + 1
                            move_response = client.call_service(
                                "/hybrid_controller/move_cyl",
                                "hybrid_controller_ros/MoveCyl",
                                {
                                    "theta_deg": float(refine_target[0]),
                                    "radius_mm": float(refine_target[1]),
                                    "z_mm": float(refine_target[2]),
                                },
                                timeout_sec=max(1.0, float(args.command_timeout_sec)),
                            )
                            settled = _wait_for_idle(
                                client=client,
                                timeout_sec=max(float(args.command_timeout_sec), float(args.ros_timeout_sec)),
                            )
                            time.sleep(max(0.0, float(args.continuous_confirm_recheck_settle_sec)))
                            planner.reset()
                            step_report["low_height_refine_move"] = {
                                "attempt": int(low_refine_attempts),
                                "source": "servo_command_point",
                                "source_point": [float(refine_point[0]), float(refine_point[1])],
                                "step_gain": max(
                                    0.01, min(1.0, float(args.continuous_low_height_refine_gain))
                                ),
                                "target_cyl": [float(refine_target[0]), float(refine_target[1]), float(refine_target[2])],
                                "response": move_response,
                                "settled_snapshot": settled,
                            }
                            servo_pending = None
                            step_report["low_height_refine_move"]["pending_reset"] = "reset_after_stopped_refine_move"
                            print(
                                "[continuous {step}] low refine MOVE_CYL theta={theta:.3f} r={radius:.2f} "
                                "center={center:.2f}px".format(
                                    step=loop_index,
                                    theta=float(refine_target[0]),
                                    radius=float(refine_target[1]),
                                    center=float(center_distance_px),
                                )
                            )
                            sleep_sec = max(0.0, interval_sec - (time.perf_counter() - loop_started))
                            if sleep_sec > 0.0:
                                time.sleep(sleep_sec)
                            continue
                effective_stopped_step_mode = bool(args.continuous_stopped_step_mode) or bool(
                    motion_safe_stopped_step_mode
                )
                if effective_stopped_step_mode:
                    pose = _current_cyl_pose(decision_snapshot)
                    stopped_target = _continuous_stopped_motion_target(
                        current_pose=pose,
                        selected_slot=selected_slot,
                        center_distance_px=center_distance_px,
                        center_allow_px=trace.get("center_allow_descent_px"),
                        z_rate_mm_s=float(decision.get("z_rate_mm_s", 0.0)),
                        confirm_z_mm=confirm_z,
                        z_tolerance_mm=float(config.vision_pick_z_tolerance_mm),
                        z_step_mm=float(args.continuous_stopped_z_step_mm),
                        refine_z_band_above_confirm_mm=float(args.continuous_stopped_refine_z_band_mm),
                        max_theta_step_deg=float(args.continuous_stopped_max_theta_step_deg),
                        max_radius_step_mm=float(args.continuous_stopped_max_radius_step_mm),
                    )
                    if stopped_target is not None:
                        stopped_reason, stopped_pose, stopped_meta = stopped_target
                        cmd_seq = next_teleop_cmd_seq(cmd_seq)
                        client.stop_teleop(use_auto_z=False, cmd_seq=cmd_seq)
                        teleop_published = True
                        metrics["commands_sent"] = int(metrics["commands_sent"]) + 1
                        metrics["stop_count"] = int(metrics["stop_count"]) + 1
                        move_response = client.call_service(
                            "/hybrid_controller/move_cyl",
                            "hybrid_controller_ros/MoveCyl",
                            {
                                "theta_deg": float(stopped_pose[0]),
                                "radius_mm": float(stopped_pose[1]),
                                "z_mm": float(stopped_pose[2]),
                            },
                            timeout_sec=max(1.0, float(args.command_timeout_sec)),
                        )
                        settled = _wait_for_idle(
                            client=client,
                            timeout_sec=max(float(args.command_timeout_sec), float(args.ros_timeout_sec)),
                        )
                        time.sleep(max(0.0, float(args.continuous_confirm_recheck_settle_sec)))
                        planner.reset()
                        servo_pending = None
                        low_refine_attempts = 0
                        metrics["stopped_move_count"] = int(metrics["stopped_move_count"]) + 1
                        step_report["stopped_step_move"] = {
                            "reason": stopped_reason,
                            "mode": (
                                "auto_camera_artifact"
                                if motion_safe_stopped_step_mode and not bool(args.continuous_stopped_step_mode)
                                else "explicit"
                            ),
                            "target_cyl": [
                                float(stopped_pose[0]),
                                float(stopped_pose[1]),
                                float(stopped_pose[2]),
                            ],
                            "meta": stopped_meta,
                            "response": move_response,
                            "settled_snapshot": settled,
                        }
                        print(
                            "[continuous {step}] stopped MOVE_CYL reason={reason} theta={theta:.3f} "
                            "r={radius:.2f} z={z:.2f}".format(
                                step=loop_index,
                                reason=stopped_reason,
                                theta=float(stopped_pose[0]),
                                radius=float(stopped_pose[1]),
                                z=float(stopped_pose[2]),
                            )
                        )
                        sleep_sec = max(0.0, interval_sec - (time.perf_counter() - loop_started))
                        if sleep_sec > 0.0:
                            time.sleep(sleep_sec)
                        continue
                command = planner.next_command(
                    theta_rate_deg_s=float(decision.get("theta_rate_deg_s", 0.0)),
                    radius_rate_mm_s=float(decision.get("radius_rate_mm_s", 0.0)),
                    z_rate_mm_s=float(decision.get("z_rate_mm_s", 0.0)),
                    use_auto_z=False,
                    now_monotonic=time.monotonic(),
                )
                if command is not None:
                    cmd_seq = next_teleop_cmd_seq(cmd_seq)
                    fine_pulse = bool(trace.get("fine_pulse", False))
                    client.publish_teleop(
                        theta_rate_deg_s=float(command.theta_rate_deg_s),
                        radius_rate_mm_s=float(command.radius_rate_mm_s),
                        z_rate_mm_s=float(command.z_rate_mm_s),
                        use_auto_z=False,
                        enabled=bool(command.enabled),
                        cmd_seq=cmd_seq,
                        client_ts=time.time(),
                    )
                    teleop_published = True
                    metrics["commands_sent"] = int(metrics["commands_sent"]) + 1
                    if fine_pulse:
                        if not (
                            bool(args.continuous_stop_at_confirm)
                            and current_z is not None
                            and confirm_z is not None
                            and abs(current_z - confirm_z) <= float(config.vision_pick_z_tolerance_mm)
                        ):
                            time.sleep(max(0.0, min(0.08, interval_sec * 0.5)))
                        cmd_seq = next_teleop_cmd_seq(cmd_seq)
                        client.stop_teleop(use_auto_z=False, cmd_seq=cmd_seq)
                        metrics["commands_sent"] = int(metrics["commands_sent"]) + 1
                        metrics["stop_count"] = int(metrics["stop_count"]) + 1
                        planner.reset()
                        if (
                            bool(args.continuous_stop_at_confirm)
                            and current_z is not None
                            and confirm_z is not None
                            and abs(current_z - confirm_z) <= float(config.vision_pick_z_tolerance_mm)
                        ):
                            time.sleep(max(0.0, float(args.continuous_confirm_recheck_settle_sec)))
                print(
                    "[continuous {step}] slot={slot} rates theta={theta:.2f} r={radius:.2f} z={z:.2f} reason={reason}".format(
                        step=loop_index,
                        slot=selected_slot_id,
                        theta=float(decision.get("theta_rate_deg_s", 0.0)),
                        radius=float(decision.get("radius_rate_mm_s", 0.0)),
                        z=float(decision.get("z_rate_mm_s", 0.0)),
                        reason=reason,
                    )
                )
            elif action == "PICK_READY":
                cmd_seq = next_teleop_cmd_seq(cmd_seq)
                client.stop_teleop(use_auto_z=False, cmd_seq=cmd_seq)
                teleop_published = True
                metrics["commands_sent"] = int(metrics["commands_sent"]) + 1
                metrics["stop_count"] = int(metrics["stop_count"]) + 1
                planner.reset()
                pick_command = None if decision.get("command") is None else str(decision.get("command"))
                metrics["final_pick_command"] = pick_command
                if bool(args.continuous_stop_at_confirm):
                    trace = decision.get("trace", {}) if isinstance(decision.get("trace"), Mapping) else {}
                    metrics["final_stop_reason"] = "pick_ready_blocked_by_confirm_stop"
                    metrics["confirm_reached"] = True
                    metrics["confirm_center_distance_px"] = trace.get("center_distance_px")
                    pose = _current_cyl_pose(decision_snapshot)
                    metrics["confirm_pose_cyl"] = None if pose is None else [float(pose[0]), float(pose[1]), float(pose[2])]
                    recheck, frame_id, debug_slots = _continuous_confirm_recheck(
                        args=args,
                        config=config,
                        cv2_module=cv2_module,
                        model=model,
                        calibration_profile=calibration_profile,
                        reader=reader,
                        client=client,
                        output_dir=output_dir,
                        slot_id=locked_slot_id,
                        frame_id=frame_id,
                        debug_slots=debug_slots,
                        device=device,
                        half=half,
                    )
                    metrics["confirm_recheck"] = recheck
                    metrics["confirm_center_distance_px"] = recheck.get(
                        "median_center_distance_px",
                        metrics["confirm_center_distance_px"],
                    )
                    if not bool(recheck.get("passed", False)):
                        metrics["final_stop_reason"] = "confirm_recheck_failed_no_pick"
                        exit_code = max(int(exit_code), 2)
                    step_report["execution"] = {
                        "executed": False,
                        "reason": "continuous_stop_at_confirm_blocks_pick",
                        "command": pick_command,
                        "confirm_recheck": recheck,
                    }
                    print(
                        "[continuous {step}] pick_ready blocked by --continuous-stop-at-confirm "
                        "command={command} recheck={recheck}".format(
                            step=loop_index,
                            command=pick_command,
                            recheck=(
                                "--"
                                if recheck.get("median_center_distance_px") is None
                                else f"{float(recheck.get('median_center_distance_px')):.2f}px"
                            ),
                        )
                    )
                    break
                allow_real_pick = bool(args.allow_real_pick)
                sucker_frozen = str(release_mode).strip().lower() == "sucker_frozen"
                if not bool(args.allow_pick):
                    execution = {
                        "executed": False,
                        "reason": "pick_blocked_requires_allow_pick",
                        "command": pick_command,
                    }
                    exit_code = 2
                elif not sucker_frozen and not allow_real_pick:
                    execution = {
                        "executed": False,
                        "reason": "real_pick_blocked_requires_sucker_freeze_or_allow_real_pick",
                        "command": pick_command,
                        "release_mode_effective": release_mode,
                    }
                    exit_code = 2
                else:
                    execution = _execute_command(
                        client=client,
                        command=pick_command,
                        allow_pick=True,
                        timeout_sec=float(args.command_timeout_sec),
                    )
                    if bool(execution.get("executed", False)):
                        try:
                            min_state_seq = int(decision_snapshot.get("state_seq", 0) or 0)
                        except (TypeError, ValueError):
                            min_state_seq = 0
                        settled = _wait_for_idle(
                            client=client,
                            timeout_sec=max(float(args.command_timeout_sec), float(args.settle_sec)),
                            min_state_seq=min_state_seq,
                        )
                        step_report["post_execution_snapshot"] = settled
                        initial_snapshot = settled if settled is not None else snapshot
                        time.sleep(max(0.0, float(args.settle_sec)))
                step_report["execution"] = execution
                print(f"[continuous {loop_index}] pick_ready command={pick_command} execution={execution}")
                break
            else:
                cmd_seq = next_teleop_cmd_seq(cmd_seq)
                client.stop_teleop(use_auto_z=False, cmd_seq=cmd_seq)
                teleop_published = True
                metrics["commands_sent"] = int(metrics["commands_sent"]) + 1
                metrics["stop_count"] = int(metrics["stop_count"]) + 1
                planner.reset()
                metrics["final_stop_reason"] = reason
                trace = decision.get("trace", {}) if isinstance(decision.get("trace"), Mapping) else {}
                if _continuous_stop_reason_is_recoverable(
                    reason,
                    trace=trace,
                    pending=servo_pending,
                    config=config,
                    recoveries_used=low_height_rebound_recoveries,
                ):
                    if reason in {"low_height_error_rebounded", "low_height_best_error_rebounded"}:
                        low_height_rebound_recoveries += 1
                        metrics["low_height_rebound_recoveries"] = int(low_height_rebound_recoveries)
                        step_report["recovery"] = {
                            "reason": reason,
                            "attempt": int(low_height_rebound_recoveries),
                            "max_attempts": int(
                                getattr(
                                    config,
                                    "vision_continuous_servo_low_height_rebound_recover_attempts",
                                    3,
                                )
                            ),
                            "settle_sec": float(args.continuous_confirm_recheck_settle_sec),
                            "pending_preserved": servo_pending,
                        }
                        try:
                            reader.reopen()
                        except Exception as reopen_error:
                            step_report["recovery"]["camera_reopen_error"] = str(reopen_error)  # type: ignore[index]
                        time.sleep(max(0.0, float(args.continuous_confirm_recheck_settle_sec)))
                    else:
                        metrics["low_height_rebound_recoveries"] = int(low_height_rebound_recoveries)
                    time.sleep(max(0.0, interval_sec - (time.perf_counter() - loop_started)))
                    continue
                exit_code = 1
                break

            sleep_sec = max(0.0, interval_sec - (time.perf_counter() - loop_started))
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
        else:
            metrics["final_stop_reason"] = "timeout"
            exit_code = 1
    except Exception as error:
        report["error"] = str(error)
        metrics["final_stop_reason"] = "exception"
        print(f"[error] {error}", file=sys.stderr)
        exit_code = 1
    finally:
        try:
            if teleop_published:
                cmd_seq = next_teleop_cmd_seq(cmd_seq)
                client.stop_teleop(use_auto_z=False, cmd_seq=cmd_seq)
                metrics["commands_sent"] = int(metrics["commands_sent"]) + 1
                metrics["stop_count"] = int(metrics["stop_count"]) + 1
        except Exception:
            pass
        reader.close()
        report_path = output_dir / "debug_vision_grasp_flow.json"
        _write_json(report_path, report)
        print(f"[output] overlay/report saved under: {output_dir}")
        print(f"[output] report: {report_path}")
    return exit_code


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
    parser.add_argument("--vision-grasp-profile", type=Path, default=defaults.vision_grasp_profile_path)
    parser.add_argument(
        "--vision-grasp-profile-optional",
        action="store_true",
        help="Allow debug PICK decisions without a tracked vision-grasp profile.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--slot-id", type=int, default=None)
    parser.add_argument("--frames", type=int, default=max(3, int(defaults.vision_grasp_stable_frames)))
    parser.add_argument("--drain-frames", type=int, default=30)
    parser.add_argument(
        "--process-latest-frames",
        type=int,
        default=1,
        help=(
            "Process only the newest N captured frames; 0 keeps the full captured batch. "
            "Continuous visual servo should normally stay at 1 so inference does not age the control frame."
        ),
    )
    parser.add_argument(
        "--continuous-drain-frames-each-loop",
        type=int,
        default=0,
        help=(
            "Continuous-mode PC-side MJPEG buffer drain after the first loop. This only consumes frames from the "
            "already-open reader before processing the newest frame; it does not restart, scan, or modify the robot "
            "camera sender."
        ),
    )
    parser.add_argument(
        "--continuous-latest-window-ms",
        type=float,
        default=250.0,
        help=(
            "Continuous-mode latest-frame window. After the first accepted frame in a control tick, keep reading the "
            "already-open MJPEG stream for this many milliseconds and process only the newest accepted frame. "
            "This is PC-side buffering only; it does not touch the robot camera sender."
        ),
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
    parser.add_argument(
        "--monitor-only",
        action="store_true",
        help=(
            "Continuously read the official MJPEG stream and report recognition/alignment metrics only. "
            "This mode never sends robot motion or suction commands."
        ),
    )
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
    parser.add_argument(
        "--confirm-z-tolerance-mm",
        type=float,
        default=None,
        help=(
            "Override the z tolerance used for continuous stop-at-confirm debug runs. "
            "Use a small value when validating the actual z=120 low-height scene."
        ),
    )
    parser.add_argument("--servo-max-attempts", type=int, default=None)
    parser.add_argument("--move-gain", type=float, default=None)
    parser.add_argument("--fine-move-gain", type=float, default=None)
    parser.add_argument("--fine-threshold-px", type=float, default=None)
    parser.add_argument("--descent-step-mm", type=float, default=None)
    parser.add_argument("--coarse-descent-step-mm", type=float, default=None)
    parser.add_argument("--fine-descent-step-mm", type=float, default=None)
    parser.add_argument("--descent-fine-band-mm", type=float, default=None)
    parser.add_argument(
        "--servo-mode",
        choices=("discrete", "continuous"),
        default="discrete",
        help="discrete keeps the legacy MOVE/PICK loop; continuous publishes velocity commands on /hybrid_controller/teleop_cyl_cmd.",
    )
    parser.add_argument(
        "--continuous-teleop-rate-hz",
        type=float,
        default=10.0,
        help="Velocity-command publish loop rate for --servo-mode continuous.",
    )
    parser.add_argument(
        "--continuous-max-duration-sec",
        type=float,
        default=30.0,
        help="Safety timeout for --servo-mode continuous.",
    )
    parser.add_argument(
        "--continuous-save-every",
        type=int,
        default=5,
        help="Save every Nth continuous debug frame; 0 disables intermediate image writes.",
    )
    parser.add_argument("--continuous-theta-rate-limit", type=float, default=None)
    parser.add_argument("--continuous-radius-rate-limit", type=float, default=None)
    parser.add_argument("--continuous-z-rate-limit", type=float, default=None)
    parser.add_argument("--continuous-center-allow-descent-px", type=float, default=None)
    parser.add_argument("--continuous-center-stop-descent-px", type=float, default=None)
    parser.add_argument("--continuous-soft-descent-rate-scale", type=float, default=None)
    parser.add_argument("--continuous-soft-descent-min-z-above-confirm-mm", type=float, default=None)
    parser.add_argument("--continuous-low-height-guard-band-mm", type=float, default=None)
    parser.add_argument("--continuous-low-height-z-rate-scale", type=float, default=None)
    parser.add_argument("--continuous-low-height-coarse-rate-scale", type=float, default=None)
    parser.add_argument("--continuous-low-height-fine-rate-scale", type=float, default=None)
    parser.add_argument("--continuous-low-height-pause-descent-band-mm", type=float, default=None)
    parser.add_argument("--continuous-descent-low-error-z-above-confirm-mm", type=float, default=None)
    parser.add_argument("--continuous-low-height-max-theta-drift-deg", type=float, default=None)
    parser.add_argument("--continuous-low-height-max-radius-drift-mm", type=float, default=None)
    parser.add_argument("--continuous-low-height-best-error-rebound-px", type=float, default=None)
    parser.add_argument("--continuous-low-height-rebound-recover-band-mm", type=float, default=None)
    parser.add_argument("--continuous-low-height-rebound-recover-attempts", type=int, default=None)
    parser.add_argument("--continuous-fine-pulse-center-px", type=float, default=None)
    parser.add_argument("--continuous-command-timeout-ms", type=float, default=None)
    parser.add_argument("--continuous-stale-frames", type=int, default=None)
    parser.add_argument("--continuous-theta-gain", type=float, default=None)
    parser.add_argument("--continuous-radius-gain", type=float, default=None)
    parser.add_argument(
        "--continuous-horizontal-mode",
        choices=("servo_command_point", "pixel_jacobian", "pixel_axis", "ibvs_dls"),
        default=None,
        help=(
            "Horizontal control source for continuous servo. ibvs_dls uses damped image-Jacobian control; "
            "pixel_jacobian/pixel_axis are simpler debug modes."
        ),
    )
    parser.add_argument("--continuous-ibvs-gain", type=float, default=None)
    parser.add_argument("--continuous-ibvs-damping", type=float, default=None)
    parser.add_argument(
        "--continuous-ibvs-du-dtheta",
        type=float,
        default=None,
        help="IBVS DLS Jacobian entry du/dtheta in px/deg for this debug run.",
    )
    parser.add_argument(
        "--continuous-ibvs-du-dradius",
        type=float,
        default=None,
        help="IBVS DLS Jacobian entry du/dradius in px/mm for this debug run.",
    )
    parser.add_argument(
        "--continuous-ibvs-dv-dtheta",
        type=float,
        default=None,
        help="IBVS DLS Jacobian entry dv/dtheta in px/deg for this debug run.",
    )
    parser.add_argument(
        "--continuous-ibvs-dv-dradius",
        type=float,
        default=None,
        help="IBVS DLS Jacobian entry dv/dradius in px/mm for this debug run.",
    )
    parser.add_argument("--continuous-pixel-jacobian-gain", type=float, default=None)
    parser.add_argument(
        "--servo-measurement-point",
        choices=tuple(sorted(SERVO_MEASUREMENT_POINTS)),
        default=None,
        help="Debug override for the visual point used by delta-servo mapping.",
    )
    parser.add_argument(
        "--low-height-measurement-point",
        choices=tuple(sorted(SERVO_MEASUREMENT_POINTS)),
        default=None,
        help=(
            "Debug override for the visual point used only in the low-height confirm/pick stage. "
            "Leave unset to use the profile/default low-height measurement point."
        ),
    )
    parser.add_argument(
        "--continuous-disable-z",
        action="store_true",
        help=(
            "Debug-only safety switch for continuous mode: publish theta/radius servo rates but force z_rate_mm_s=0. "
            "Use this to verify horizontal centering direction before descent."
        ),
    )
    parser.add_argument(
        "--continuous-stop-at-confirm",
        action="store_true",
        help=(
            "Continuous-mode safety/debug switch: stop at confirm_z after centering/descent, "
            "then do not execute +radius extension or PICK. Use this for z=120 low-height alignment checks."
        ),
    )
    parser.add_argument(
        "--low-height-centering-check",
        action="store_true",
        help=(
            "Shortcut for the no-suction z=120 validation flow: continuous servo, persistent official MJPEG, "
            "--execute, --allow-pick, --continuous-stop-at-confirm, latest-frame processing, and no real PICK."
        ),
    )
    parser.add_argument(
        "--continuous-confirm-recheck-repeats",
        type=int,
        default=3,
        help="When --continuous-stop-at-confirm stops motion, reopen the PC-side stream and remeasure this many times.",
    )
    parser.add_argument(
        "--continuous-confirm-recheck-settle-sec",
        type=float,
        default=1.5,
        help="Settle delay before the fresh stopped-frame recheck for --continuous-stop-at-confirm.",
    )
    parser.add_argument(
        "--continuous-confirm-recheck-max-spread-px",
        type=float,
        default=3.0,
        help="Maximum repeated-point spread accepted by the stopped-frame recheck.",
    )
    parser.add_argument(
        "--continuous-low-height-discrete-refine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic fallback: at confirm_z, use stop-settle-measure MOVE_CYL refinements instead of continuous "
            "teleop rates. Disabled by default so high-to-low centering remains smooth."
        ),
    )
    parser.add_argument("--continuous-low-height-refine-attempts", type=int, default=4)
    parser.add_argument("--continuous-low-height-refine-max-theta-step-deg", type=float, default=0.25)
    parser.add_argument("--continuous-low-height-refine-max-radius-step-mm", type=float, default=4.0)
    parser.add_argument(
        "--continuous-low-height-refine-gain",
        type=float,
        default=0.45,
        help=(
            "Damping factor for stopped low-height MOVE_CYL refinements. "
            "Use <1.0 to avoid overshoot from low-height backlash/local mapping error."
        ),
    )
    parser.add_argument(
        "--continuous-low-height-refine-recheck",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Before each low-height discrete refinement, stop and require a stable multi-frame recheck.",
    )
    parser.add_argument(
        "--continuous-low-height-refine-min-improvement-px",
        type=float,
        default=0.25,
        help="Stop low-height discrete refinement when the stopped-frame median no longer improves by this many pixels.",
    )
    parser.add_argument(
        "--continuous-stopped-step-mode",
        action="store_true",
        help=(
            "Fallback diagnostic mode: convert descent/refine servo decisions into stop-then-MOVE_CYL "
            "small steps. Leave off when evaluating smooth continuous descent."
        ),
    )
    parser.add_argument(
        "--continuous-auto-stopped-on-camera-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After motion-correlated MJPEG artifacts/timeouts, stop teleop, reopen only the PC reader, "
            "and finish with stop-then-MOVE_CYL small steps. Does not change robot camera FPS or quality."
        ),
    )
    parser.add_argument("--continuous-stopped-z-step-mm", type=float, default=4.0)
    parser.add_argument("--continuous-stopped-refine-z-band-mm", type=float, default=45.0)
    parser.add_argument("--continuous-stopped-max-theta-step-deg", type=float, default=0.20)
    parser.add_argument("--continuous-stopped-max-radius-step-mm", type=float, default=1.0)
    parser.add_argument(
        "--allow-real-pick",
        action="store_true",
        help="Permit final PICK when the robot is not in sucker_frozen dry-run mode.",
    )
    parser.add_argument("--max-steps", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.low_height_centering_check):
        args.servo_mode = "continuous"
        args.persistent_camera = True
        args.execute = True
        args.allow_pick = True
        args.allow_execute_loop = True
        args.continuous_stop_at_confirm = True
        args.process_latest_frames = 1
        args.frames = 1
        args.drain_frames = 0
        args.continuous_drain_frames_each_loop = max(0, int(args.continuous_drain_frames_each_loop))
        args.timeout_sec = max(float(args.timeout_sec), 5.0)
        args.capture_backend = "http"
        args.confirm_z_mm = 120.0 if args.confirm_z_mm is None else float(args.confirm_z_mm)
        if args.confirm_z_tolerance_mm is None:
            args.confirm_z_tolerance_mm = 1.0
        args.servo_measurement_point = args.servo_measurement_point or "geometry_subpixel"
        args.allow_real_pick = False
    if str(args.servo_mode) == "continuous" and not bool(args.persistent_camera):
        print(
            "[guard] --servo-mode continuous requires --persistent-camera so the official MJPEG stream stays open.",
            file=sys.stderr,
        )
        return 2
    if str(args.servo_mode) == "continuous" and not bool(args.execute):
        print(
            "[guard] --servo-mode continuous publishes robot teleop commands and requires --execute.",
            file=sys.stderr,
        )
        return 2
    if bool(args.monitor_only):
        args.execute = False
        args.allow_pick = False
        args.allow_real_pick = False
        args.persistent_camera = True
        args.capture_backend = "http"
        args.process_latest_frames = 1
        args.frames = max(1, int(args.frames))
        args.drain_frames = max(0, int(args.drain_frames))
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
        "vision_grasp_profile_path": Path(args.vision_grasp_profile),
        "vision_grasp_profile_required": not bool(args.vision_grasp_profile_optional),
        "vision_eye_in_hand_pick_radius_bias_mm": float(args.pick_radius_bias_mm),
        "pick_tool_offset_source": str(args.pick_tool_offset_source),
    }
    if args.confirm_z_mm is not None:
        config_kwargs["vision_pick_confirm_z_mm"] = float(args.confirm_z_mm)
    if args.confirm_z_tolerance_mm is not None:
        config_kwargs["vision_pick_z_tolerance_mm"] = float(args.confirm_z_tolerance_mm)
    if args.center_tolerance_px is not None:
        config_kwargs["vision_servo_center_tolerance_px"] = float(args.center_tolerance_px)
        config_kwargs["vision_servo_action_tolerance_px"] = float(args.center_tolerance_px)
    if args.servo_max_attempts is not None:
        config_kwargs["vision_servo_max_attempts"] = int(args.servo_max_attempts)
    if args.move_gain is not None:
        config_kwargs["vision_servo_move_gain"] = float(args.move_gain)
    if args.fine_move_gain is not None:
        config_kwargs["vision_servo_fine_move_gain"] = float(args.fine_move_gain)
    if args.fine_threshold_px is not None:
        config_kwargs["vision_servo_fine_threshold_px"] = float(args.fine_threshold_px)
    if args.servo_measurement_point is not None:
        config_kwargs["vision_servo_measurement_point"] = str(args.servo_measurement_point)
    if args.low_height_measurement_point is not None:
        config_kwargs["vision_servo_low_height_measurement_point"] = str(args.low_height_measurement_point)
    if args.descent_step_mm is not None:
        config_kwargs["vision_pick_descent_step_mm"] = float(args.descent_step_mm)
        config_kwargs["vision_pick_descent_coarse_step_mm"] = float(args.descent_step_mm)
        config_kwargs["vision_pick_descent_fine_step_mm"] = float(args.descent_step_mm)
    if args.coarse_descent_step_mm is not None:
        config_kwargs["vision_pick_descent_coarse_step_mm"] = float(args.coarse_descent_step_mm)
    if args.fine_descent_step_mm is not None:
        config_kwargs["vision_pick_descent_fine_step_mm"] = float(args.fine_descent_step_mm)
    if args.descent_fine_band_mm is not None:
        config_kwargs["vision_pick_descent_fine_band_mm"] = float(args.descent_fine_band_mm)
    if args.continuous_theta_rate_limit is not None:
        config_kwargs["vision_continuous_servo_theta_rate_limit_deg_s"] = float(args.continuous_theta_rate_limit)
    if args.continuous_radius_rate_limit is not None:
        config_kwargs["vision_continuous_servo_radius_rate_limit_mm_s"] = float(args.continuous_radius_rate_limit)
    if args.continuous_z_rate_limit is not None:
        config_kwargs["vision_continuous_servo_z_rate_limit_mm_s"] = float(args.continuous_z_rate_limit)
    if args.continuous_center_allow_descent_px is not None:
        config_kwargs["vision_continuous_servo_center_allow_descent_px"] = float(args.continuous_center_allow_descent_px)
    if args.continuous_center_stop_descent_px is not None:
        config_kwargs["vision_continuous_servo_center_stop_descent_px"] = float(args.continuous_center_stop_descent_px)
    if args.continuous_soft_descent_rate_scale is not None:
        config_kwargs["vision_continuous_servo_soft_descent_rate_scale"] = float(args.continuous_soft_descent_rate_scale)
    if args.continuous_soft_descent_min_z_above_confirm_mm is not None:
        config_kwargs["vision_continuous_servo_soft_descent_min_z_above_confirm_mm"] = float(
            args.continuous_soft_descent_min_z_above_confirm_mm
        )
    if args.continuous_low_height_guard_band_mm is not None:
        config_kwargs["vision_continuous_servo_low_height_guard_band_mm"] = float(
            args.continuous_low_height_guard_band_mm
        )
    if args.continuous_low_height_z_rate_scale is not None:
        config_kwargs["vision_continuous_servo_low_height_z_rate_scale"] = float(
            args.continuous_low_height_z_rate_scale
        )
    if args.continuous_low_height_coarse_rate_scale is not None:
        config_kwargs["vision_continuous_servo_low_height_coarse_rate_scale"] = float(
            args.continuous_low_height_coarse_rate_scale
        )
    if args.continuous_low_height_fine_rate_scale is not None:
        config_kwargs["vision_continuous_servo_low_height_fine_rate_scale"] = float(
            args.continuous_low_height_fine_rate_scale
        )
    if args.continuous_low_height_pause_descent_band_mm is not None:
        config_kwargs["vision_continuous_servo_low_height_pause_descent_band_mm"] = float(
            args.continuous_low_height_pause_descent_band_mm
        )
    if args.continuous_descent_low_error_z_above_confirm_mm is not None:
        config_kwargs["vision_continuous_servo_descent_low_error_z_above_confirm_mm"] = float(
            args.continuous_descent_low_error_z_above_confirm_mm
        )
    if args.continuous_low_height_max_theta_drift_deg is not None:
        config_kwargs["vision_continuous_servo_low_height_max_theta_drift_deg"] = float(
            args.continuous_low_height_max_theta_drift_deg
        )
    if args.continuous_low_height_max_radius_drift_mm is not None:
        config_kwargs["vision_continuous_servo_low_height_max_radius_drift_mm"] = float(
            args.continuous_low_height_max_radius_drift_mm
        )
    if args.continuous_low_height_best_error_rebound_px is not None:
        config_kwargs["vision_continuous_servo_low_height_best_error_rebound_px"] = float(
            args.continuous_low_height_best_error_rebound_px
        )
    if args.continuous_low_height_rebound_recover_band_mm is not None:
        config_kwargs["vision_continuous_servo_low_height_rebound_recover_band_mm"] = float(
            args.continuous_low_height_rebound_recover_band_mm
        )
    if args.continuous_low_height_rebound_recover_attempts is not None:
        config_kwargs["vision_continuous_servo_low_height_rebound_recover_attempts"] = int(
            args.continuous_low_height_rebound_recover_attempts
        )
    if args.continuous_fine_pulse_center_px is not None:
        config_kwargs["vision_continuous_servo_fine_pulse_center_px"] = float(args.continuous_fine_pulse_center_px)
    if args.continuous_command_timeout_ms is not None:
        config_kwargs["vision_continuous_servo_command_timeout_ms"] = float(args.continuous_command_timeout_ms)
    if args.continuous_stale_frames is not None:
        config_kwargs["vision_continuous_servo_stale_frames"] = int(args.continuous_stale_frames)
    if args.continuous_theta_gain is not None:
        config_kwargs["vision_continuous_servo_theta_gain_deg_s_per_deg"] = float(args.continuous_theta_gain)
    if args.continuous_radius_gain is not None:
        config_kwargs["vision_continuous_servo_radius_gain_mm_s_per_mm"] = float(args.continuous_radius_gain)
    if args.continuous_horizontal_mode is not None:
        config_kwargs["vision_continuous_servo_horizontal_mode"] = str(args.continuous_horizontal_mode)
    if args.continuous_ibvs_gain is not None:
        config_kwargs["vision_continuous_servo_ibvs_gain"] = float(args.continuous_ibvs_gain)
    if args.continuous_ibvs_damping is not None:
        config_kwargs["vision_continuous_servo_ibvs_damping_px_per_unit"] = float(args.continuous_ibvs_damping)
    if args.continuous_ibvs_du_dtheta is not None:
        config_kwargs["vision_continuous_servo_ibvs_du_dtheta_px_per_deg"] = float(args.continuous_ibvs_du_dtheta)
    if args.continuous_ibvs_du_dradius is not None:
        config_kwargs["vision_continuous_servo_ibvs_du_dradius_px_per_mm"] = float(args.continuous_ibvs_du_dradius)
    if args.continuous_ibvs_dv_dtheta is not None:
        config_kwargs["vision_continuous_servo_ibvs_dv_dtheta_px_per_deg"] = float(args.continuous_ibvs_dv_dtheta)
    if args.continuous_ibvs_dv_dradius is not None:
        config_kwargs["vision_continuous_servo_ibvs_dv_dradius_px_per_mm"] = float(args.continuous_ibvs_dv_dradius)
    if args.continuous_pixel_jacobian_gain is not None:
        config_kwargs["vision_continuous_servo_pixel_jacobian_gain"] = float(args.continuous_pixel_jacobian_gain)
    config = AppConfig(**config_kwargs).resolved()
    grasp_profile = load_vision_grasp_profile(config)
    if grasp_profile.ready:
        config = apply_vision_grasp_profile(config, grasp_profile).resolved()
        if args.confirm_z_mm is not None:
            config = replace(config, vision_pick_confirm_z_mm=float(args.confirm_z_mm)).resolved()
        if args.confirm_z_tolerance_mm is not None:
            config = replace(config, vision_pick_z_tolerance_mm=float(args.confirm_z_tolerance_mm)).resolved()
        if args.servo_measurement_point is not None:
            config = replace(config, vision_servo_measurement_point=str(args.servo_measurement_point)).resolved()
        if args.low_height_measurement_point is not None:
            config = replace(
                config,
                vision_servo_low_height_measurement_point=str(args.low_height_measurement_point),
            ).resolved()
    elif bool(config.vision_grasp_profile_required):
        print(f"[guard] Vision grasp profile unavailable: {grasp_profile.error}", file=sys.stderr)
        return 2
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
        "servo_mode": str(args.servo_mode),
        "stream_url": stream_url,
        "weights": str(args.weights or _resolve_weights_path(config)),
        "profile": str(args.profile),
        "detector": str(args.detector),
        "execute": bool(args.execute),
        "monitor_only": bool(args.monitor_only),
        "allow_pick": bool(args.allow_pick),
        "low_height_centering_check": bool(args.low_height_centering_check),
        "pick_radius_bias_mm": float(args.pick_radius_bias_mm),
        "confirm_z_mm": float(config.vision_pick_confirm_z_mm),
        "confirm_z_tolerance_mm": float(config.vision_pick_z_tolerance_mm),
        "center_tolerance_px": None if args.center_tolerance_px is None else float(args.center_tolerance_px),
        "frames_requested": int(args.frames),
        "drain_frames": int(args.drain_frames),
        "continuous_drain_frames_each_loop": int(args.continuous_drain_frames_each_loop),
        "continuous_latest_window_ms": float(args.continuous_latest_window_ms),
        "process_latest_frames": int(args.process_latest_frames),
        "capture_backend": str(args.capture_backend),
        "persistent_camera": bool(args.persistent_camera),
        "continuous_teleop_rate_hz": float(args.continuous_teleop_rate_hz),
        "continuous_max_duration_sec": float(args.continuous_max_duration_sec),
        "continuous_horizontal_mode": str(config.vision_continuous_servo_horizontal_mode),
        "servo_measurement_point": str(config.vision_servo_measurement_point),
        "low_height_measurement_point": str(config.vision_servo_low_height_measurement_point),
        "continuous_ibvs_jacobian": {
            "source": str(config.vision_continuous_servo_ibvs_jacobian_source),
            "du_dtheta_px_per_deg": float(config.vision_continuous_servo_ibvs_du_dtheta_px_per_deg),
            "du_dradius_px_per_mm": float(config.vision_continuous_servo_ibvs_du_dradius_px_per_mm),
            "dv_dtheta_px_per_deg": float(config.vision_continuous_servo_ibvs_dv_dtheta_px_per_deg),
            "dv_dradius_px_per_mm": float(config.vision_continuous_servo_ibvs_dv_dradius_px_per_mm),
            "gain": float(config.vision_continuous_servo_ibvs_gain),
            "damping_px_per_unit": float(config.vision_continuous_servo_ibvs_damping_px_per_unit),
        },
        "camera_contract": (
            "PC reads the single official Hiwonder MJPEG stream from "
            "usb_cam.service -> /usb_cam/image_rect_color -> web_video_server:8080; "
            "this tool must not start, restart, scan, or mutate the robot camera sender."
        ),
        "ros": ros_status,
        "steps": [],
    }
    initial_profile_jacobian = _ibvs_jacobian_from_profile_for_snapshot(
        config=config,
        calibration_profile=calibration_profile,
        snapshot=initial_snapshot,
    )
    if initial_profile_jacobian is not None:
        jacobian_values, jacobian_source = initial_profile_jacobian
        report["continuous_ibvs_jacobian"]["active_profile_source"] = jacobian_source
        report["continuous_ibvs_jacobian"]["active_profile_values"] = [float(value) for value in jacobian_values]
    if str(args.servo_mode) == "continuous":
        exit_code = _run_continuous_servo_flow(
            args=args,
            config=config,
            cv2_module=cv2,
            model=model,
            calibration_profile=calibration_profile,
            stream_candidates=stream_candidates,
            output_dir=output_dir,
            client=client,
            initial_snapshot=initial_snapshot,
            ros_status=ros_status,
            report=report,
            device=device,
            half=half,
        )
        if client is not None:
            client.close()
        return int(exit_code)
    frame_id = 0
    exit_code = 0
    debug_slots: list[SlotState] | None = None
    locked_slot_id: int | None = int(args.slot_id) if args.slot_id is not None else None
    servo_pending: dict[str, object] | None = None
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
                    latest_window_sec=max(0.0, float(args.continuous_latest_window_ms)) / 1000.0,
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
            packet, last_frame, frame_id, debug_slots = _process_frame_batch(
                frames=process_frames,
                model=model,
                config=config,
                calibration_profile=calibration_profile,
                snapshot_for_stage=initial_snapshot,
                frame_id_start=frame_id,
                slots=debug_slots,
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
                frame_pose_age_ms=_frame_pose_age_for_static_snapshot(resolve_snapshot, packet),
            )
            resolved_packet["camera_frames_captured"] = int(len(frames))
            resolved_packet["camera_frames_processed"] = int(len(process_frames))
            camera_transport = (
                persistent_reader.transport_stats()
                if persistent_reader is not None
                else {"capture_backend": str(args.capture_backend), "persistent": False}
            )
            resolved_packet["camera_transport"] = dict(camera_transport)
            selected_slot = _select_slot(resolved_packet, locked_slot_id)
            pose_for_patch = _current_cyl_pose(resolve_snapshot)
            selected_slot = _patch_low_height_local_center(
                packet=resolved_packet,
                frame_bgr=last_frame,
                selected_slot=selected_slot,
                pending=servo_pending,
                current_z_mm=None if pose_for_patch is None else float(pose_for_patch[2]),
                confirm_z_mm=float(config.vision_pick_confirm_z_mm),
                measurement_point=str(config.vision_servo_low_height_measurement_point or config.vision_servo_measurement_point),
            )
            if selected_slot is not None and not bool(selected_slot.get("valid", False)):
                selected_slot = None
            if selected_slot is None:
                selected_slot = _low_height_local_synthetic_slot(
                    packet=resolved_packet,
                    frame_bgr=last_frame,
                    pending=servo_pending,
                    current_z_mm=None if pose_for_patch is None else float(pose_for_patch[2]),
                    confirm_z_mm=float(config.vision_pick_confirm_z_mm),
                    measurement_point=str(config.vision_servo_low_height_measurement_point or config.vision_servo_measurement_point),
                    slot_id=locked_slot_id,
                )
            _upsert_packet_slot(resolved_packet, selected_slot)
            selected_slot_id = None if selected_slot is None else int(selected_slot.get("slot_id", selected_slot.get("slot", 0)))
            if locked_slot_id is None and selected_slot_id is not None:
                locked_slot_id = int(selected_slot_id)
            decision = _decision_for_packet(
                packet=resolved_packet,
                config=config,
                snapshot=resolve_snapshot,
                selected_slot=selected_slot,
                pending=servo_pending,
            )
            servo_pending = decision.get("pending") if isinstance(decision.get("pending"), dict) else None
            if servo_pending is None and selected_slot is not None:
                tracking_point = _slot_tracking_point(selected_slot)
                try:
                    selected_distance = float(selected_slot.get("center_distance_px"))
                except (TypeError, ValueError):
                    selected_distance = float("nan")
                if tracking_point is not None:
                    servo_pending = {
                        "slot_id": int(selected_slot_id or selected_slot.get("slot_id", selected_slot.get("slot", 1)) or 1),
                        "stable_frames": 0,
                        "lost_frames": 0,
                        "last_center_px": [float(tracking_point[0]), float(tracking_point[1])],
                        "last_center_distance_px": (
                            None if not math.isfinite(selected_distance) else float(selected_distance)
                        ),
                        "source": "monitor_tracking",
                    }

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
                "camera_transport": dict(camera_transport),
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
                        if servo_pending is not None:
                            servo_pending["waiting_for_ack"] = False
                            servo_pending["min_frame_id"] = int(frame_id) + 1
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
                if bool(args.monitor_only):
                    sleep_sec = max(0.0, (1.0 / max(0.1, float(args.continuous_teleop_rate_hz))))
                    time.sleep(sleep_sec)
                    initial_snapshot = resolve_snapshot
                    continue
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
