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
from hybrid_controller.adapters.teleop_ros_channel import RosTeleopPublishPlanner
from hybrid_controller.adapters.teleop_ros_channel import new_teleop_cmd_seq_base
from hybrid_controller.adapters.teleop_ros_channel import next_teleop_cmd_seq
from hybrid_controller.config import AppConfig
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
        if _frame_has_horizontal_tearing(frame):
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

    def transport_stats(self) -> dict[str, object]:
        capture = self._capture
        if capture is None:
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

    def reopen(self) -> None:
        """Reset only this PC-side stream consumer so the next read starts fresh."""
        self.close()
        self._ensure_open()

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
    for key in ("pixel_center_f", "pixel_center", "geometry_center_f", "geometry_center", "grasp_pixel_f", "grasp_pixel"):
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
) -> tuple[float, float, float]:
    theta_limit = abs(float(max_theta_step_deg))
    radius_limit = abs(float(max_radius_step_mm))
    theta_delta = max(-theta_limit, min(theta_limit, float(target_theta) - float(current_pose[0])))
    radius_delta = max(-radius_limit, min(radius_limit, float(target_radius) - float(current_pose[1])))
    return (
        float(current_pose[0]) + float(theta_delta),
        float(current_pose[1]) + float(radius_delta),
        float(current_pose[2]),
    )


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
        state = str(last_snapshot.get("state", "")).strip().upper()
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
            drain_frames=drain_frames if repeat_index == 1 else 0,
            timeout_sec=float(args.timeout_sec),
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
            frame_pose_age_ms=_packet_frame_pose_age_ms(packet),
        )
        resolved_packet["camera_frames_captured"] = int(len(captured))
        resolved_packet["camera_frames_processed"] = int(len(process_frames))
        selected_slot = _select_slot(resolved_packet, slot_id)
        alignment_provenance = _slot_alignment_provenance(selected_slot, resolved_packet)
        sample: dict[str, object] = {
            "repeat_index": int(repeat_index),
            "selected_slot": selected_slot,
            "center_distance_px": None if selected_slot is None else selected_slot.get("center_distance_px"),
            "pixel_center_f": None if selected_slot is None else selected_slot.get("pixel_center_f"),
            "geometry_center_f": None if selected_slot is None else selected_slot.get("geometry_center_f"),
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
        point = sample.get("geometry_center_f") or sample.get("pixel_center_f")
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
    recheck: dict[str, object] = {
        "settle_sec": settle_sec,
        "repeats": repeats,
        "median_center_distance_px": median_distance,
        "repeat_spread_px": repeat_spread_px,
        "tolerance_px": tolerance_px,
        "max_repeat_spread_px": max_repeat_spread_px,
        "passed": passed,
        "measurement_point": str(getattr(config, "vision_servo_measurement_point", "")),
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
        "commands_sent": 0,
        "stop_count": 0,
        "max_center_distance_px": 0.0,
        "mean_frame_age_ms": None,
        "frame_age_samples": 0,
        "max_state_age_ms": 0.0,
        "final_pick_command": None,
        "final_stop_reason": "",
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

            stream_url, frames = reader.read(
                frame_count=int(args.frames),
                drain_frames=int(args.drain_frames) if loop_index == 1 else 0,
                timeout_sec=float(args.timeout_sec),
            )
            process_frames = _select_latest_frames(frames, int(args.process_latest_frames))
            report["stream_url"] = stream_url
            packet, last_frame, frame_id, debug_slots = _process_frame_batch(
                frames=process_frames,
                model=model,
                config=config,
                calibration_profile=calibration_profile,
                snapshot_for_stage=snapshot,
                frame_id_start=frame_id,
                slots=debug_slots,
                device=device,
                half=half,
            )
            packet["camera_frames_captured"] = int(len(frames))
            packet["camera_frames_processed"] = int(len(process_frames))
            frame_age_ms = _packet_frame_pose_age_ms(packet)
            if frame_age_ms is not None:
                frame_ages.append(float(frame_age_ms))
            resolved_packet = _resolve_packet(
                packet=packet,
                config=config,
                snapshot=snapshot,
                snapshot_age_ms=snapshot_age_ms,
                frame_pose_age_ms=frame_age_ms,
            )
            resolved_packet["camera_frames_captured"] = int(len(frames))
            resolved_packet["camera_frames_processed"] = int(len(process_frames))
            selection_slot_id = _continuous_slot_id_for_selection(locked_slot_id, servo_pending)
            selected_slot = _select_slot(resolved_packet, selection_slot_id)
            selected_slot_id = None if selected_slot is None else int(selected_slot.get("slot_id", selected_slot.get("slot", 0)))
            if locked_slot_id is None and selected_slot_id is not None:
                locked_slot_id = int(selected_slot_id)
                metrics["locked_slot_id"] = int(locked_slot_id)
            decision = _continuous_decision_for_packet(
                packet=resolved_packet,
                config=config,
                snapshot=snapshot,
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
            servo_pending = decision.get("pending") if isinstance(decision.get("pending"), dict) else None
            if selected_slot is not None and selected_slot.get("center_distance_px") is not None:
                try:
                    metrics["max_center_distance_px"] = max(
                        float(metrics["max_center_distance_px"]),
                        float(selected_slot.get("center_distance_px")),
                    )
                except (TypeError, ValueError):
                    pass
            if frame_ages:
                metrics["mean_frame_age_ms"] = sum(frame_ages) / float(len(frame_ages))
                metrics["frame_age_samples"] = int(len(frame_ages))

            step_report: dict[str, object] = {
                "step": loop_index,
                "elapsed_sec": max(0.0, time.perf_counter() - started_at),
                "camera_frames_captured": int(len(frames)),
                "camera_frames_processed": int(len(process_frames)),
                "slots": _slot_summary(resolved_packet),
                "selected_slot_id": selected_slot_id,
                "selected_slot": selected_slot,
                "decision": decision,
                "snapshot": snapshot,
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
                if (
                    bool(args.continuous_stop_at_confirm)
                    and current_z is not None
                    and confirm_z is not None
                    and abs(current_z - confirm_z) <= float(config.vision_pick_z_tolerance_mm)
                ):
                    pick_ready_center_px = max(0.1, float(config.vision_continuous_servo_pick_ready_center_px))
                    if center_distance_px is not None and center_distance_px <= pick_ready_center_px:
                        cmd_seq = next_teleop_cmd_seq(cmd_seq)
                        client.stop_teleop(use_auto_z=False, cmd_seq=cmd_seq)
                        teleop_published = True
                        metrics["commands_sent"] = int(metrics["commands_sent"]) + 1
                        metrics["stop_count"] = int(metrics["stop_count"]) + 1
                        metrics["final_stop_reason"] = "confirm_height_centered_no_pick"
                        metrics["confirm_reached"] = True
                        metrics["confirm_center_distance_px"] = center_distance_px
                        pose = _current_cyl_pose(snapshot)
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
                        "reason": "confirm_height_not_centered",
                        "center_distance_px": center_distance_px,
                        "pick_ready_center_px": pick_ready_center_px,
                    }
                    if bool(args.continuous_low_height_discrete_refine):
                        refine_point = _servo_command_point_from_slot(selected_slot)
                        pose = _current_cyl_pose(snapshot)
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
                                "target_cyl": [float(refine_target[0]), float(refine_target[1]), float(refine_target[2])],
                                "response": move_response,
                                "settled_snapshot": settled,
                            }
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
                    pose = _current_cyl_pose(snapshot)
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
                        settled = _wait_for_idle(
                            client=client,
                            timeout_sec=max(float(args.command_timeout_sec), float(args.settle_sec)),
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
                if reason in {"hold", "lost_target_wait", "settle_near_center"}:
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
        default=max(1, int(defaults.vision_grasp_stable_frames)),
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
    parser.add_argument("--continuous-fine-pulse-center-px", type=float, default=None)
    parser.add_argument("--continuous-command-timeout-ms", type=float, default=None)
    parser.add_argument("--continuous-theta-gain", type=float, default=None)
    parser.add_argument("--continuous-radius-gain", type=float, default=None)
    parser.add_argument(
        "--servo-measurement-point",
        choices=("center", "geometry", "grasp", "center_subpixel", "geometry_subpixel", "grasp_subpixel"),
        default=None,
        help="Debug override for the visual point used by delta-servo mapping.",
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
        default=True,
        help=(
            "At confirm_z, use stop-settle-measure MOVE_CYL refinements instead of tiny teleop rates "
            "until the strict center gate is reached."
        ),
    )
    parser.add_argument("--continuous-low-height-refine-attempts", type=int, default=4)
    parser.add_argument("--continuous-low-height-refine-max-theta-step-deg", type=float, default=0.25)
    parser.add_argument("--continuous-low-height-refine-max-radius-step-mm", type=float, default=1.5)
    parser.add_argument(
        "--allow-real-pick",
        action="store_true",
        help="Permit final PICK when the robot is not in sucker_frozen dry-run mode.",
    )
    parser.add_argument("--max-steps", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
    if args.continuous_fine_pulse_center_px is not None:
        config_kwargs["vision_continuous_servo_fine_pulse_center_px"] = float(args.continuous_fine_pulse_center_px)
    if args.continuous_command_timeout_ms is not None:
        config_kwargs["vision_continuous_servo_command_timeout_ms"] = float(args.continuous_command_timeout_ms)
    if args.continuous_theta_gain is not None:
        config_kwargs["vision_continuous_servo_theta_gain_deg_s_per_deg"] = float(args.continuous_theta_gain)
    if args.continuous_radius_gain is not None:
        config_kwargs["vision_continuous_servo_radius_gain_mm_s_per_mm"] = float(args.continuous_radius_gain)
    config = AppConfig(**config_kwargs).resolved()
    grasp_profile = load_vision_grasp_profile(config)
    if grasp_profile.ready:
        config = apply_vision_grasp_profile(config, grasp_profile).resolved()
        if args.confirm_z_mm is not None:
            config = replace(config, vision_pick_confirm_z_mm=float(args.confirm_z_mm)).resolved()
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
        "allow_pick": bool(args.allow_pick),
        "pick_radius_bias_mm": float(args.pick_radius_bias_mm),
        "confirm_z_mm": None if args.confirm_z_mm is None else float(args.confirm_z_mm),
        "center_tolerance_px": None if args.center_tolerance_px is None else float(args.center_tolerance_px),
        "frames_requested": int(args.frames),
        "drain_frames": int(args.drain_frames),
        "process_latest_frames": int(args.process_latest_frames),
        "capture_backend": str(args.capture_backend),
        "persistent_camera": bool(args.persistent_camera),
        "continuous_teleop_rate_hz": float(args.continuous_teleop_rate_hz),
        "continuous_max_duration_sec": float(args.continuous_max_duration_sec),
        "camera_contract": (
            "PC reads the single official Hiwonder MJPEG stream from "
            "usb_cam.service -> /usb_cam/image_rect_color -> web_video_server:8080; "
            "this tool must not start, restart, scan, or mutate the robot camera sender."
        ),
        "ros": ros_status,
        "steps": [],
    }
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
                frame_pose_age_ms=_packet_frame_pose_age_ms(packet),
            )
            resolved_packet["camera_frames_captured"] = int(len(frames))
            resolved_packet["camera_frames_processed"] = int(len(process_frames))
            selected_slot = _select_slot(resolved_packet, locked_slot_id)
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
