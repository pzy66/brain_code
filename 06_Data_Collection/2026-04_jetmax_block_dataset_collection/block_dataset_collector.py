#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import queue
import re
import socket
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:
    import roslibpy
except ImportError:  # pragma: no cover - optional runtime dependency
    roslibpy = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = "http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "vision" / "captures"
DEFAULT_WINDOW_SIZE = (1440, 900)
DEFAULT_ROBOT_HOST = "192.168.149.1"
DEFAULT_ROBOT_TRANSPORT = "auto"
DEFAULT_ROSBRIDGE_PORT = 9091
DEFAULT_ROBOT_PORT = 8888
DEFAULT_HOME_THETA_DEG = 0.0
DEFAULT_HOME_RADIUS_MM = 120.0
DEFAULT_HOME_Z_MM = 160.0
DEFAULT_HOME_Z_LIMITS_MM = (80.0, 212.8)
SPLIT_OPTIONS = [
    ("训练集 train", "train"),
    ("验证集 val", "val"),
    ("测试集 test", "test"),
    ("原始 raw", "raw"),
]


def ui_camera_status(status: str) -> str:
    return {
        "camera initializing": "摄像头初始化中",
        "camera reconnecting": "摄像头重连中",
        "camera connected": "摄像头已连接",
        "camera stream lost": "摄像头画面丢失",
    }.get(status, status)


def log_stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_source(raw: str) -> Union[int, str]:
    text = str(raw).strip()
    if not text:
        return 0
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def sanitize_token(raw: str, fallback: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_\-]+", "_", raw.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def frame_to_qimage(frame_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    return QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()


def compute_sharpness(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_frame_delta(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_a, gray_b)
    return float(diff.mean())


@dataclass
class AppConfig:
    source: Union[int, str]
    output_root: Path
    session_prefix: str
    image_ext: str
    jpeg_quality: int
    auto_close_sec: float
    fullscreen: bool
    robot_host: str
    robot_transport: str
    rosbridge_port: int
    robot_port: int
    robot_timeout_sec: float
    home_theta_deg: float
    home_radius_mm: float
    home_z_mm: float


@dataclass
class SaveJob:
    image: np.ndarray
    image_path: Path
    manifest_path: Path
    metadata: Dict[str, Any]


def cylindrical_from_cartesian_xy(x_mm: float, y_mm: float) -> Tuple[float, float]:
    radius_mm = math.hypot(float(x_mm), float(y_mm))
    theta_deg = math.degrees(math.atan2(-float(x_mm), -float(y_mm)))
    if abs(theta_deg) < 1e-9:
        theta_deg = 0.0
    return theta_deg, radius_mm


def compact_robot_status(status: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(status, dict):
        return None
    robot_cyl = status.get("robot_cyl")
    return {
        "state": status.get("state"),
        "busy": status.get("busy"),
        "busy_action": status.get("busy_action"),
        "carrying": status.get("carrying"),
        "robot_cyl": dict(robot_cyl) if isinstance(robot_cyl, dict) else None,
        "robot_xy": status.get("robot_xy"),
        "robot_z": status.get("robot_z"),
        "home_pose": status.get("home_pose"),
        "last_ack": status.get("last_ack"),
        "last_error": status.get("last_error"),
    }


class RobotHeightClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._last_transport = ""

    def fetch_status(self) -> Dict[str, Any]:
        status = self._run_with_transport(lambda transport: self._fetch_status_for_transport(transport))
        if self._last_transport:
            status = dict(status)
            status["transport"] = self._last_transport
        return status

    def move_home_height(self, z_mm: float) -> Dict[str, Any]:
        result = self._run_with_transport(lambda transport: self._move_home_height_for_transport(transport, z_mm))
        if self._last_transport:
            result = dict(result)
            result["transport"] = self._last_transport
        return result

    def move_cyl(self, theta_deg: float, radius_mm: float, z_mm: float) -> Dict[str, Any]:
        result = self._run_with_transport(
            lambda transport: self._move_cyl_for_transport(transport, theta_deg, radius_mm, z_mm)
        )
        if self._last_transport:
            result = dict(result)
            result["transport"] = self._last_transport
        return result

    def _run_with_transport(self, action) -> Dict[str, Any]:
        requested = str(self.config.robot_transport or "auto").strip().lower()
        transports = ["ros", "tcp"] if requested == "auto" else [requested]
        errors = []
        for transport in transports:
            try:
                result = action(transport)
                self._last_transport = transport
                return result
            except Exception as exc:
                errors.append(f"{transport}: {exc}")
        raise RuntimeError("机械臂控制连接失败；" + "；".join(errors))

    def _fetch_status_for_transport(self, transport: str) -> Dict[str, Any]:
        if transport == "ros":
            return self._fetch_ros_status()
        if transport == "tcp":
            return self._fetch_tcp_status()
        raise RuntimeError(f"Unsupported robot transport: {transport}")

    def _move_home_height_for_transport(self, transport: str, z_mm: float) -> Dict[str, Any]:
        if transport == "ros":
            return self._move_home_height_ros(z_mm)
        if transport == "tcp":
            return self._move_home_height_tcp(z_mm)
        raise RuntimeError(f"Unsupported robot transport: {transport}")

    def _move_cyl_for_transport(self, transport: str, theta_deg: float, radius_mm: float, z_mm: float) -> Dict[str, Any]:
        if transport == "ros":
            return self._move_cyl_ros(theta_deg, radius_mm, z_mm)
        if transport == "tcp":
            return self._move_cyl_tcp(theta_deg, radius_mm, z_mm)
        raise RuntimeError(f"Unsupported robot transport: {transport}")

    def _fetch_tcp_status(self) -> Dict[str, Any]:
        line = self._send_tcp_command("STATUS", wait_for={"ACK STATUS"})
        prefix = "ACK STATUS "
        if line == "ACK STATUS":
            return {}
        if not line.startswith(prefix):
            raise RuntimeError(f"Unexpected robot STATUS response: {line}")
        payload = json.loads(line[len(prefix) :].strip())
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid robot STATUS payload: {payload!r}")
        return payload

    def _move_home_height_tcp(self, z_mm: float) -> Dict[str, Any]:
        before = self._fetch_tcp_status()
        theta_deg, radius_mm = self._home_cyl_from_status(before)
        return self._move_cyl_tcp(theta_deg, radius_mm, z_mm, before=before, action="move_home_height")

    def _move_cyl_tcp(
        self,
        theta_deg: float,
        radius_mm: float,
        z_mm: float,
        *,
        before: Optional[Dict[str, Any]] = None,
        action: str = "move_cyl",
    ) -> Dict[str, Any]:
        before = self._fetch_tcp_status() if before is None else before
        target_theta_deg = float(theta_deg)
        target_radius_mm = float(radius_mm)
        target_z_mm = float(z_mm)
        ack = self._send_tcp_command(
            f"MOVE_CYL {target_theta_deg:.2f} {target_radius_mm:.2f} {target_z_mm:.2f}",
            wait_for={"ACK MOVE"},
        )
        after = self._fetch_tcp_status()
        return {
            "action": action,
            "ack": ack,
            "target": {
                "theta_deg": round(target_theta_deg, 3),
                "radius_mm": round(target_radius_mm, 3),
                "z_mm": round(target_z_mm, 3),
            },
            "before": compact_robot_status(before),
            "after": compact_robot_status(after),
        }

    def _fetch_ros_status(self) -> Dict[str, Any]:
        ros = self._connect_ros()
        try:
            return self._wait_ros_status(ros)
        finally:
            try:
                ros.close()
            except Exception:
                pass

    def _move_home_height_ros(self, z_mm: float) -> Dict[str, Any]:
        ros = self._connect_ros()
        try:
            before = self._wait_ros_status(ros)
            theta_deg, radius_mm = self._home_cyl_from_status(before)
            return self._move_cyl_with_ros(ros, theta_deg, radius_mm, z_mm, before=before, action="move_home_height")
        finally:
            try:
                ros.close()
            except Exception:
                pass

    def _move_cyl_ros(self, theta_deg: float, radius_mm: float, z_mm: float) -> Dict[str, Any]:
        ros = self._connect_ros()
        try:
            before = self._wait_ros_status(ros)
            return self._move_cyl_with_ros(ros, theta_deg, radius_mm, z_mm, before=before, action="move_cyl")
        finally:
            try:
                ros.close()
            except Exception:
                pass

    def _move_cyl_with_ros(
        self,
        ros,
        theta_deg: float,
        radius_mm: float,
        z_mm: float,
        *,
        before: Dict[str, Any],
        action: str,
    ) -> Dict[str, Any]:
        target_theta_deg = float(theta_deg)
        target_radius_mm = float(radius_mm)
        target_z_mm = float(z_mm)
        response = self._call_ros_service(
            ros,
            "/hybrid_controller/move_cyl",
            "hybrid_controller_ros/MoveCyl",
            {"theta_deg": target_theta_deg, "radius_mm": target_radius_mm, "z_mm": target_z_mm},
            timeout_sec=max(float(self.config.robot_timeout_sec), 3.0),
        )
        if not bool(response.get("ok", response.get("success", False))):
            raise RuntimeError(str(response.get("message", "move_cyl rejected")))
        after = self._wait_ros_status(
            ros,
            target_theta_deg=target_theta_deg,
            target_radius_mm=target_radius_mm,
            target_z_mm=target_z_mm,
        )
        return {
            "action": action,
            "ack": str(response.get("message", "ACCEPTED move_cyl")),
            "target": {
                "theta_deg": round(target_theta_deg, 3),
                "radius_mm": round(target_radius_mm, 3),
                "z_mm": round(target_z_mm, 3),
            },
            "before": compact_robot_status(before),
            "after": compact_robot_status(after),
        }

    def _connect_ros(self):
        if roslibpy is None:
            raise RuntimeError("roslibpy is not installed.")
        ros = roslibpy.Ros(host=str(self.config.robot_host), port=int(self.config.rosbridge_port))
        ready = threading.Event()
        error_holder: Dict[str, str] = {}

        def on_ready(*_args: object) -> None:
            ready.set()

        def on_error(error: object) -> None:
            error_holder["error"] = str(error)
            ready.set()

        ros.on_ready(on_ready, run_in_thread=False)
        ros.on("error", on_error)
        ros.run()
        if not ready.wait(timeout=max(0.2, float(self.config.robot_timeout_sec))):
            try:
                ros.close()
            except Exception:
                pass
            raise TimeoutError(f"Timed out connecting to rosbridge {self.config.robot_host}:{self.config.rosbridge_port}.")
        if error_holder.get("error"):
            try:
                ros.close()
            except Exception:
                pass
            raise RuntimeError(error_holder["error"])
        if not ros.is_connected:
            try:
                ros.close()
            except Exception:
                pass
            raise RuntimeError("rosbridge is not connected.")
        return ros

    def _wait_ros_status(
        self,
        ros,
        target_theta_deg: Optional[float] = None,
        target_radius_mm: Optional[float] = None,
        target_z_mm: Optional[float] = None,
    ) -> Dict[str, Any]:
        done = threading.Event()
        holder: Dict[str, Any] = {"message": None}
        topic = roslibpy.Topic(ros, "/hybrid_controller/state", "hybrid_controller_ros/RobotState")

        def on_message(message: Dict[str, Any]) -> None:
            snapshot = self._ros_state_to_status(message)
            if target_z_mm is not None:
                busy = bool(snapshot.get("busy", False))
                robot_cyl = snapshot.get("robot_cyl") if isinstance(snapshot.get("robot_cyl"), dict) else {}
                theta_value = float(robot_cyl.get("theta_deg", 0.0) or 0.0)
                radius_value = float(robot_cyl.get("radius_mm", 0.0) or 0.0)
                z_value = float(robot_cyl.get("z_mm", snapshot.get("robot_z", 0.0)) or 0.0)
                theta_ok = target_theta_deg is None or abs(theta_value - float(target_theta_deg)) <= 2.0
                radius_ok = target_radius_mm is None or abs(radius_value - float(target_radius_mm)) <= 4.0
                z_ok = abs(z_value - float(target_z_mm)) <= 2.5
                if busy or not (theta_ok and radius_ok and z_ok):
                    holder["message"] = snapshot
                    return
            holder["message"] = snapshot
            done.set()

        topic.subscribe(on_message)
        try:
            if not done.wait(timeout=max(0.2, float(self.config.robot_timeout_sec))):
                last = holder.get("message")
                if isinstance(last, dict) and target_z_mm is not None:
                    return last
                raise TimeoutError("Timed out waiting for /hybrid_controller/state.")
        finally:
            try:
                topic.unsubscribe()
            except Exception:
                pass
        message = holder.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Invalid /hybrid_controller/state payload.")
        return message

    def _call_ros_service(
        self,
        ros,
        name: str,
        service_type: str,
        request: Dict[str, Any],
        *,
        timeout_sec: float,
    ) -> Dict[str, Any]:
        service = roslibpy.Service(ros, name, service_type)
        done = threading.Event()
        holder: Dict[str, Any] = {"response": None, "error": None}

        def on_success(response: Dict[str, Any]) -> None:
            holder["response"] = dict(response)
            done.set()

        def on_error(error: object) -> None:
            holder["error"] = str(error)
            done.set()

        service.call(roslibpy.ServiceRequest(request), callback=on_success, errback=on_error)
        if not done.wait(timeout=float(timeout_sec)):
            raise TimeoutError(f"Timed out waiting for service '{name}'.")
        if holder["error"] is not None:
            raise RuntimeError(str(holder["error"]))
        response = holder["response"]
        if not isinstance(response, dict):
            raise RuntimeError(f"Invalid service response for '{name}': {response!r}")
        return response

    def _ros_state_to_status(self, message: Dict[str, Any]) -> Dict[str, Any]:
        theta_deg = float(message.get("theta_deg", 0.0) or 0.0)
        radius_mm = float(message.get("radius_mm", 0.0) or 0.0)
        z_mm = float(message.get("z_mm", 0.0) or 0.0)
        x_mm = float(message.get("x_mm", 0.0) or 0.0)
        y_mm = float(message.get("y_mm", 0.0) or 0.0)
        return {
            "state": str(message.get("state", "")),
            "state_seq": int(message.get("state_seq", 0) or 0),
            "busy": bool(message.get("busy", False)),
            "busy_action": str(message.get("busy_action", "")),
            "carrying": bool(message.get("carrying", False)),
            "robot_xy": [x_mm, y_mm],
            "robot_z": z_mm,
            "robot_cyl": {"theta_deg": theta_deg, "radius_mm": radius_mm, "z_mm": z_mm},
            "home_pose": None,
            "last_ack": str(message.get("last_ack", "")),
            "last_error": str(message.get("last_error_message", "")),
        }

    def _home_cyl_from_status(self, status: Dict[str, Any]) -> Tuple[float, float]:
        home_pose = status.get("home_pose")
        if isinstance(home_pose, (list, tuple)) and len(home_pose) >= 2:
            try:
                return cylindrical_from_cartesian_xy(float(home_pose[0]), float(home_pose[1]))
            except (TypeError, ValueError):
                pass
        return float(self.config.home_theta_deg), float(self.config.home_radius_mm)

    def _send_tcp_command(self, command: str, wait_for: set[str]) -> str:
        timeout_sec = max(0.2, float(self.config.robot_timeout_sec))
        deadline = time.monotonic() + timeout_sec
        buffer = ""
        with socket.create_connection((self.config.robot_host, int(self.config.robot_port)), timeout=timeout_sec) as sock:
            sock.settimeout(0.2)
            sock.sendall((command.strip() + "\n").encode("utf-8"))
            while time.monotonic() < deadline:
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="ignore")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line in wait_for or any(line.startswith(prefix + " ") for prefix in wait_for):
                        return line
                    if line == "BUSY":
                        raise RuntimeError("Robot is busy; wait for the current motion to finish.")
                    if line.startswith("ERR"):
                        raise RuntimeError(line[3:].strip() or "Robot command failed.")
            raise TimeoutError(f"Robot command timed out: {command}")


class CameraLoader:
    def __init__(self, source: Union[int, str]) -> None:
        self.source = source
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._fps = 0.0
        self._status = "camera initializing"
        self._last_capture_ts = 0.0
        self._frame_counter = 0
        self._fps_window_start = time.perf_counter()

    def start(self) -> "CameraLoader":
        if self._thread is not None:
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="dataset-camera-loader", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def status_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            age_ms = (time.perf_counter() - self._last_capture_ts) * 1000.0 if self._last_capture_ts > 0 else -1.0
            return {
                "status": self._status,
                "capture_fps": float(self._fps),
                "last_frame_age_ms": float(age_ms),
            }

    def peek_latest(self) -> Optional[Tuple[int, np.ndarray, float, float, str]]:
        with self._lock:
            if self._frame is None:
                return None
            return self._seq, self._frame, self._last_capture_ts, self._fps, self._status

    def _open_capture(self) -> cv2.VideoCapture:
        backend = cv2.CAP_ANY
        if isinstance(self.source, str) and hasattr(cv2, "CAP_FFMPEG"):
            backend = cv2.CAP_FFMPEG
        try:
            capture = cv2.VideoCapture(self.source, backend)
        except TypeError:
            capture = cv2.VideoCapture(self.source)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
        return capture

    def _set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def _run(self) -> None:
        while not self._stop_event.is_set():
            capture = self._open_capture()
            if capture is None or not capture.isOpened():
                self._set_status("camera reconnecting")
                if capture is not None:
                    capture.release()
                if self._stop_event.wait(1.0):
                    return
                continue

            with self._lock:
                self._status = "camera connected"

            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    self._set_status("camera stream lost")
                    break

                now = time.perf_counter()
                self._frame_counter += 1
                elapsed = now - self._fps_window_start
                if elapsed >= 1.0:
                    current_fps = self._frame_counter / elapsed
                    self._fps = current_fps if self._fps <= 0 else (self._fps * 0.8 + current_fps * 0.2)
                    self._frame_counter = 0
                    self._fps_window_start = now

                with self._lock:
                    self._seq += 1
                    self._frame = frame
                    self._last_capture_ts = now
                    self._status = "camera connected"

            capture.release()
            if self._stop_event.wait(0.5):
                return


class ImageWriter(QThread):
    saved = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, image_ext: str, jpeg_quality: int) -> None:
        super().__init__()
        self.image_ext = image_ext.lower()
        self.jpeg_quality = int(jpeg_quality)
        self._queue: "queue.Queue[Optional[SaveJob]]" = queue.Queue(maxsize=256)
        self._stop_requested = False

    def enqueue(self, job: SaveJob) -> bool:
        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            return False

    def request_stop(self) -> None:
        self._stop_requested = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def run(self) -> None:
        while True:
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._stop_requested:
                    return
                continue

            if job is None:
                return

            try:
                job.image_path.parent.mkdir(parents=True, exist_ok=True)
                job.manifest_path.parent.mkdir(parents=True, exist_ok=True)
                params = []
                suffix = job.image_path.suffix.lower()
                if suffix in (".jpg", ".jpeg"):
                    params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                ok = cv2.imwrite(str(job.image_path), job.image, params)
                if not ok:
                    raise RuntimeError(f"cv2.imwrite failed: {job.image_path}")
                with job.manifest_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(job.metadata, ensure_ascii=False) + "\n")
                self.saved.emit(job.metadata)
            except Exception as exc:
                self.failed.emit(str(exc))


class RobotControlWorker(QThread):
    succeeded = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        config: AppConfig,
        action: str,
        target_z_mm: Optional[float] = None,
        target_pose: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.action = str(action)
        self.target_z_mm = target_z_mm
        self.target_pose = target_pose

    def run(self) -> None:
        try:
            client = RobotHeightClient(self.config)
            if self.action == "status":
                self.succeeded.emit(
                    {
                        "action": "status",
                        "status": compact_robot_status(client.fetch_status()),
                    }
                )
                return
            if self.action == "move_home_height":
                if self.target_z_mm is None:
                    raise RuntimeError("Missing target home height.")
                self.succeeded.emit(client.move_home_height(float(self.target_z_mm)))
                return
            if self.action == "move_cyl":
                if self.target_pose is None:
                    raise RuntimeError("Missing target robot pose.")
                theta_deg, radius_mm, z_mm = self.target_pose
                self.succeeded.emit(client.move_cyl(float(theta_deg), float(radius_mm), float(z_mm)))
                return
            raise RuntimeError(f"Unsupported robot action: {self.action}")
        except Exception as exc:
            self.failed.emit(str(exc))


class CollectorWindow(QWidget):
    def __init__(self, config: AppConfig, camera: CameraLoader, writer: ImageWriter) -> None:
        super().__init__()
        self.config = config
        self.camera = camera
        self.writer = writer
        self._frame_seq = 0
        self._current_frame: Optional[np.ndarray] = None
        self._current_capture_ts = 0.0
        self._current_frame_size = (0, 0)
        self._saved_count = 0
        self._queued_count = 0
        self._last_saved_path = "-"
        self._last_saved_frame: Optional[np.ndarray] = None
        self._current_sharpness = 0.0
        self._current_delta = 0.0
        self._auto_capture_enabled = False
        self._session_dir: Optional[Path] = None
        self._manifest_path: Optional[Path] = None
        self._session_started_at = ""
        self._last_auto_capture_ts = 0.0
        self._burst_remaining = 0
        self._burst_interval_sec = 0.0
        self._last_burst_capture_ts = 0.0
        self._robot_worker: Optional[RobotControlWorker] = None
        self._last_robot_status: Optional[Dict[str, Any]] = None
        self._last_robot_height_command: Optional[Dict[str, Any]] = None
        self._robot_status_text = "未连接"

        self._build_ui()
        self._create_session()
        self._bind_signals()

        self._frame_timer = QTimer(self)
        self._frame_timer.setTimerType(Qt.PreciseTimer)
        self._frame_timer.timeout.connect(self._poll_frame)
        self._frame_timer.start(15)

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status_labels)
        self._status_timer.start(200)

        if self.config.auto_close_sec > 0:
            QTimer.singleShot(int(round(self.config.auto_close_sec * 1000.0)), self.close)

    def _build_ui(self) -> None:
        self.setWindowTitle("木块数据采集器")
        self.resize(*DEFAULT_WINDOW_SIZE)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFont(QFont("Microsoft YaHei UI", 10))

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)
        self.video_label = QLabel("等待摄像头画面...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(960, 720)
        self.video_label.setStyleSheet("background: #000; color: #fff; border: 1px solid #333;")
        left_layout.addWidget(self.video_label, stretch=1)

        self.status_label = QLabel()
        self.status_label.setFont(QFont("Microsoft YaHei UI", 10))
        self.status_label.setStyleSheet("background: #111; color: #eee; padding: 8px; border: 1px solid #333;")
        left_layout.addWidget(self.status_label)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setStretch(3, 1)

        session_group = QGroupBox("采集会话")
        session_form = QFormLayout(session_group)
        self.session_prefix_edit = QLineEdit(self.config.session_prefix)
        self.session_name_label = QLabel("-")
        self.scene_tag_edit = QLineEdit("default")
        self.scene_tag_edit.setPlaceholderText("场景标签，例如 single_block / stacked / negative")
        self.split_combo = QComboBox()
        for label, value in SPLIT_OPTIONS:
            self.split_combo.addItem(label, value)
        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("可选备注")
        session_form.addRow("会话前缀", self.session_prefix_edit)
        session_form.addRow("当前会话", self.session_name_label)
        session_form.addRow("场景标签", self.scene_tag_edit)
        session_form.addRow("数据划分", self.split_combo)
        session_form.addRow("备注", self.note_edit)
        right_layout.addWidget(session_group)

        capture_group = QGroupBox("采集控制")
        capture_layout = QGridLayout(capture_group)
        self.single_button = QPushButton("保存当前帧 [空格]")
        self.new_session_button = QPushButton("新建会话")
        self.burst_count_spin = QSpinBox()
        self.burst_count_spin.setRange(2, 50)
        self.burst_count_spin.setValue(5)
        self.burst_interval_spin = QDoubleSpinBox()
        self.burst_interval_spin.setRange(0.05, 10.0)
        self.burst_interval_spin.setDecimals(2)
        self.burst_interval_spin.setValue(0.35)
        self.burst_button = QPushButton("连拍 [B]")
        self.auto_interval_spin = QDoubleSpinBox()
        self.auto_interval_spin.setRange(0.2, 30.0)
        self.auto_interval_spin.setDecimals(2)
        self.auto_interval_spin.setValue(1.5)
        self.auto_toggle = QCheckBox("自动采集 [A]")
        self.negative_check = QCheckBox("负样本")
        capture_layout.addWidget(self.single_button, 0, 0, 1, 2)
        capture_layout.addWidget(self.new_session_button, 0, 2, 1, 2)
        capture_layout.addWidget(QLabel("连拍张数"), 1, 0)
        capture_layout.addWidget(self.burst_count_spin, 1, 1)
        capture_layout.addWidget(QLabel("连拍间隔（秒）"), 1, 2)
        capture_layout.addWidget(self.burst_interval_spin, 1, 3)
        capture_layout.addWidget(self.burst_button, 2, 0, 1, 2)
        capture_layout.addWidget(QLabel("自动采集间隔（秒）"), 2, 2)
        capture_layout.addWidget(self.auto_interval_spin, 2, 3)
        capture_layout.addWidget(self.auto_toggle, 3, 0, 1, 2)
        capture_layout.addWidget(self.negative_check, 3, 2, 1, 2)
        right_layout.addWidget(capture_group)

        robot_group = QGroupBox("机械臂位置")
        robot_layout = QGridLayout(robot_group)
        self.robot_host_label = QLabel(self._format_robot_endpoint())
        self.robot_status_label = QLabel(self._robot_status_text)
        self.robot_theta_spin = QDoubleSpinBox()
        self.robot_theta_spin.setRange(-120.0, 120.0)
        self.robot_theta_spin.setDecimals(1)
        self.robot_theta_spin.setSingleStep(5.0)
        self.robot_theta_spin.setValue(float(self.config.home_theta_deg))
        self.robot_theta_spin.setSuffix(" deg")
        self.robot_radius_spin = QDoubleSpinBox()
        self.robot_radius_spin.setRange(50.0, 280.0)
        self.robot_radius_spin.setDecimals(1)
        self.robot_radius_spin.setSingleStep(5.0)
        self.robot_radius_spin.setValue(float(self.config.home_radius_mm))
        self.robot_radius_spin.setSuffix(" mm")
        self.home_height_spin = QDoubleSpinBox()
        self.home_height_spin.setRange(DEFAULT_HOME_Z_LIMITS_MM[0], DEFAULT_HOME_Z_LIMITS_MM[1])
        self.home_height_spin.setDecimals(1)
        self.home_height_spin.setSingleStep(5.0)
        self.home_height_spin.setValue(float(self.config.home_z_mm))
        self.home_height_spin.setSuffix(" mm")
        self.robot_step_spin = QDoubleSpinBox()
        self.robot_step_spin.setRange(1.0, 50.0)
        self.robot_step_spin.setDecimals(1)
        self.robot_step_spin.setSingleStep(1.0)
        self.robot_step_spin.setValue(10.0)
        self.robot_step_spin.setSuffix(" mm")
        self.robot_status_button = QPushButton("读取位置")
        self.robot_move_button = QPushButton("移动到该位置")
        self.home_height_move_button = QPushButton("Home 水平位置")
        self.theta_left_button = QPushButton("theta -")
        self.theta_right_button = QPushButton("theta +")
        self.radius_in_button = QPushButton("r -")
        self.radius_out_button = QPushButton("r +")
        self.z_down_button = QPushButton("z -")
        self.z_up_button = QPushButton("z +")
        robot_layout.addWidget(QLabel("机器人"), 0, 0)
        robot_layout.addWidget(self.robot_host_label, 0, 1, 1, 4)
        robot_layout.addWidget(QLabel("theta"), 1, 0)
        robot_layout.addWidget(self.robot_theta_spin, 1, 1)
        robot_layout.addWidget(QLabel("r"), 1, 2)
        robot_layout.addWidget(self.robot_radius_spin, 1, 3)
        robot_layout.addWidget(QLabel("z"), 2, 0)
        robot_layout.addWidget(self.home_height_spin, 2, 1)
        robot_layout.addWidget(QLabel("步长"), 2, 2)
        robot_layout.addWidget(self.robot_step_spin, 2, 3)
        robot_layout.addWidget(self.robot_status_button, 3, 0, 1, 2)
        robot_layout.addWidget(self.robot_move_button, 3, 2, 1, 2)
        robot_layout.addWidget(self.home_height_move_button, 3, 4)
        robot_layout.addWidget(self.theta_left_button, 4, 0)
        robot_layout.addWidget(self.theta_right_button, 4, 1)
        robot_layout.addWidget(self.radius_in_button, 4, 2)
        robot_layout.addWidget(self.radius_out_button, 4, 3)
        robot_layout.addWidget(self.z_down_button, 5, 0)
        robot_layout.addWidget(self.z_up_button, 5, 1)
        robot_layout.addWidget(QLabel("状态"), 6, 0)
        robot_layout.addWidget(self.robot_status_label, 6, 1, 1, 4)
        right_layout.addWidget(robot_group)

        output_group = QGroupBox("输出信息")
        output_form = QFormLayout(output_group)
        self.output_root_label = QLabel(str(self.config.output_root))
        self.output_root_label.setWordWrap(True)
        self.saved_count_label = QLabel("0")
        self.queue_count_label = QLabel("0")
        self.last_saved_label = QLabel("-")
        self.last_saved_label.setWordWrap(True)
        output_form.addRow("保存根目录", self.output_root_label)
        output_form.addRow("已保存图片", self.saved_count_label)
        output_form.addRow("待写入队列", self.queue_count_label)
        output_form.addRow("最近保存", self.last_saved_label)
        right_layout.addWidget(output_group)

        help_group = QGroupBox("快捷键")
        help_layout = QVBoxLayout(help_group)
        help_label = QLabel(
            "空格：保存当前帧\n"
            "B：开始连拍\n"
            "A：开关自动采集\n"
            "N：切换负样本标记\n"
            "S：新建会话\n"
            "H：移动机械臂到当前位置\n"
            "Esc：退出程序"
        )
        help_label.setFont(QFont("Microsoft YaHei UI", 10))
        help_layout.addWidget(help_label)
        right_layout.addWidget(help_group)
        right_layout.addStretch(1)

        main_layout.addLayout(left_layout, stretch=4)
        main_layout.addLayout(right_layout, stretch=2)

    def _bind_signals(self) -> None:
        self.single_button.clicked.connect(lambda: self._capture_now(mode="manual"))
        self.burst_button.clicked.connect(self._start_burst_capture)
        self.auto_toggle.toggled.connect(self._toggle_auto_capture)
        self.new_session_button.clicked.connect(self._create_session)
        self.robot_status_button.clicked.connect(self._refresh_robot_status)
        self.robot_move_button.clicked.connect(self._move_robot_current_pose)
        self.home_height_move_button.clicked.connect(self._move_robot_home_height)
        self.theta_left_button.clicked.connect(lambda: self._step_robot_pose(dtheta=-1.0))
        self.theta_right_button.clicked.connect(lambda: self._step_robot_pose(dtheta=1.0))
        self.radius_in_button.clicked.connect(lambda: self._step_robot_pose(dr=-1.0))
        self.radius_out_button.clicked.connect(lambda: self._step_robot_pose(dr=1.0))
        self.z_down_button.clicked.connect(lambda: self._step_robot_pose(dz=-1.0))
        self.z_up_button.clicked.connect(lambda: self._step_robot_pose(dz=1.0))
        self.writer.saved.connect(self._on_save_success)
        self.writer.failed.connect(self._on_save_failed)

    def _format_robot_endpoint(self) -> str:
        transport = str(self.config.robot_transport or "auto").strip().lower()
        if transport == "ros":
            return f"ROS {self.config.robot_host}:{self.config.rosbridge_port}"
        if transport == "tcp":
            return f"TCP {self.config.robot_host}:{self.config.robot_port}"
        return f"auto ROS {self.config.robot_host}:{self.config.rosbridge_port} / TCP {self.config.robot_host}:{self.config.robot_port}"

    def _create_session(self) -> None:
        prefix = sanitize_token(self.session_prefix_edit.text(), "block_collect")
        session_name = f"{prefix}_{now_stamp()}"
        session_dir = self.config.output_root / session_name
        manifest_path = session_dir / "manifest.jsonl"
        session_dir.mkdir(parents=True, exist_ok=True)
        session_meta = {
            "session_name": session_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": self.config.source if isinstance(self.config.source, int) else str(self.config.source),
            "output_root": str(self.config.output_root),
            "image_ext": self.config.image_ext,
            "jpeg_quality": self.config.jpeg_quality,
            "robot_host": self.config.robot_host,
            "robot_transport": self.config.robot_transport,
            "rosbridge_port": self.config.rosbridge_port,
            "robot_port": self.config.robot_port,
            "robot_pose_cyl": self._current_robot_pose(),
            "home_height_z_mm": round(float(self.home_height_spin.value()), 3),
            "last_robot_status": compact_robot_status(self._last_robot_status),
            "last_robot_height_command": self._last_robot_height_command,
        }
        with (session_dir / "session_meta.json").open("w", encoding="utf-8") as handle:
            json.dump(session_meta, handle, ensure_ascii=False, indent=2)
        self._session_dir = session_dir
        self._manifest_path = manifest_path
        self._session_started_at = session_meta["created_at"]
        self.session_name_label.setText(session_name)
        log_stderr(f"数据采集会话已创建: {session_name}")

    def _poll_frame(self) -> None:
        latest = self.camera.peek_latest()
        if latest is None:
            return

        frame_seq, frame, capture_ts, _, _ = latest
        if frame_seq == self._frame_seq:
            self._run_scheduled_capture_tasks()
            return

        self._frame_seq = frame_seq
        self._current_capture_ts = capture_ts
        self._current_frame = frame.copy()
        height, width = self._current_frame.shape[:2]
        self._current_frame_size = (width, height)

        qimage = frame_to_qimage(self._current_frame)
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.video_label.setPixmap(scaled)

        self._current_sharpness = compute_sharpness(self._current_frame)
        if self._last_saved_frame is not None:
            self._current_delta = compute_frame_delta(self._current_frame, self._last_saved_frame)
        else:
            self._current_delta = 0.0

        self._run_scheduled_capture_tasks()

    def _run_scheduled_capture_tasks(self) -> None:
        now = time.perf_counter()
        if self._auto_capture_enabled:
            interval = float(self.auto_interval_spin.value())
            if interval > 0 and now - self._last_auto_capture_ts >= interval:
                if self._capture_now(mode="auto"):
                    self._last_auto_capture_ts = now

        if self._burst_remaining > 0:
            interval = max(0.01, self._burst_interval_sec)
            if now - self._last_burst_capture_ts >= interval:
                if self._capture_now(mode="burst"):
                    self._burst_remaining -= 1
                    self._last_burst_capture_ts = now
                else:
                    self._burst_remaining = 0

    def _build_image_path(self, mode: str) -> Tuple[Path, Dict[str, Any]]:
        assert self._session_dir is not None
        assert self._manifest_path is not None
        assert self._current_frame is not None

        timestamp = datetime.now()
        split = str(self.split_combo.currentData() or "raw").strip() or "raw"
        scene_tag = sanitize_token(self.scene_tag_edit.text(), "default")
        mode_tag = sanitize_token(mode, "manual")
        note = self.note_edit.text().strip()
        negative = bool(self.negative_check.isChecked())
        frame_id = self._frame_seq
        stem = f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]}_{scene_tag}_{mode_tag}_f{frame_id:06d}"
        image_dir = self._session_dir / "images" / split
        image_path = image_dir / f"{stem}.{self.config.image_ext}"

        metadata = {
            "timestamp": timestamp.isoformat(timespec="milliseconds"),
            "session_name": self._session_dir.name,
            "session_started_at": self._session_started_at,
            "frame_id": int(frame_id),
            "mode": mode,
            "split": split,
            "scene_tag": scene_tag,
            "negative_sample": negative,
            "note": note,
            "image_path": str(image_path.relative_to(self._session_dir)),
            "source": self.config.source if isinstance(self.config.source, int) else str(self.config.source),
            "capture_age_ms": round(max(0.0, (time.perf_counter() - self._current_capture_ts) * 1000.0), 3),
            "frame_size": [int(self._current_frame_size[0]), int(self._current_frame_size[1])],
            "sharpness": round(self._current_sharpness, 3),
            "delta_from_last_saved": round(self._current_delta, 3),
            "robot_pose_cyl": self._current_robot_pose(),
            "home_height_z_mm": round(float(self.home_height_spin.value()), 3),
            "last_robot_status": compact_robot_status(self._last_robot_status),
            "last_robot_height_command": self._last_robot_height_command,
        }
        return image_path, metadata

    def _capture_now(self, mode: str) -> bool:
        if self._current_frame is None or self._manifest_path is None:
            return False

        image_path, metadata = self._build_image_path(mode)
        job = SaveJob(
            image=self._current_frame.copy(),
            image_path=image_path,
            manifest_path=self._manifest_path,
            metadata=metadata,
        )
        if not self.writer.enqueue(job):
            QMessageBox.warning(self, "写盘队列繁忙", "保存队列已满，请稍等后重试。")
            return False

        self._queued_count += 1
        self.queue_count_label.setText(str(self._queued_count))
        return True

    def _start_burst_capture(self) -> None:
        self._burst_remaining = int(self.burst_count_spin.value())
        self._burst_interval_sec = float(self.burst_interval_spin.value())
        self._last_burst_capture_ts = 0.0
        log_stderr(f"连拍任务已启动: count={self._burst_remaining}, interval={self._burst_interval_sec:.2f}s")

    def _toggle_auto_capture(self, enabled: bool) -> None:
        self._auto_capture_enabled = bool(enabled)
        self._last_auto_capture_ts = 0.0
        state = "开启" if enabled else "关闭"
        log_stderr(f"自动采集已{state}")

    def _refresh_robot_status(self) -> None:
        self._start_robot_worker("status")

    def _move_robot_home_height(self) -> None:
        self.robot_theta_spin.setValue(float(self.config.home_theta_deg))
        self.robot_radius_spin.setValue(float(self.config.home_radius_mm))
        self._move_robot_current_pose(action="move_home_height")

    def _move_robot_current_pose(self, action: str = "move_cyl") -> None:
        self._start_robot_worker(action, target_pose=self._target_robot_pose())

    def _step_robot_pose(self, dtheta: float = 0.0, dr: float = 0.0, dz: float = 0.0) -> None:
        step = float(self.robot_step_spin.value())
        theta = float(self.robot_theta_spin.value()) + float(dtheta) * step
        radius = float(self.robot_radius_spin.value()) + float(dr) * step
        z_mm = float(self.home_height_spin.value()) + float(dz) * step
        self.robot_theta_spin.setValue(theta)
        self.robot_radius_spin.setValue(radius)
        self.home_height_spin.setValue(z_mm)
        self._move_robot_current_pose()

    def _target_robot_pose(self) -> Tuple[float, float, float]:
        return (
            float(self.robot_theta_spin.value()),
            float(self.robot_radius_spin.value()),
            float(self.home_height_spin.value()),
        )

    def _current_robot_pose(self) -> Dict[str, float]:
        theta, radius, z_mm = self._target_robot_pose()
        return {
            "theta_deg": round(theta, 3),
            "radius_mm": round(radius, 3),
            "z_mm": round(z_mm, 3),
        }

    def _start_robot_worker(
        self,
        action: str,
        target_z_mm: Optional[float] = None,
        target_pose: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        if self._robot_worker is not None and self._robot_worker.isRunning():
            QMessageBox.information(self, "机械臂忙碌", "上一条机械臂命令还在执行，请稍等。")
            return
        self._set_robot_controls_enabled(False)
        if action in {"move_home_height", "move_cyl"}:
            pose = target_pose or self._target_robot_pose()
            self._set_robot_status(f"移动中：theta={pose[0]:.1f}, r={pose[1]:.1f}, z={pose[2]:.1f}")
        else:
            self._set_robot_status("读取状态中")
        worker = RobotControlWorker(self.config, action, target_z_mm, target_pose)
        worker.succeeded.connect(self._on_robot_worker_success)
        worker.failed.connect(self._on_robot_worker_failed)
        worker.finished.connect(lambda: self._set_robot_controls_enabled(True))
        worker.finished.connect(lambda: setattr(self, "_robot_worker", None))
        self._robot_worker = worker
        worker.start()

    def _set_robot_controls_enabled(self, enabled: bool) -> None:
        self.robot_theta_spin.setEnabled(enabled)
        self.robot_radius_spin.setEnabled(enabled)
        self.home_height_spin.setEnabled(enabled)
        self.robot_step_spin.setEnabled(enabled)
        self.robot_status_button.setEnabled(enabled)
        self.robot_move_button.setEnabled(enabled)
        self.home_height_move_button.setEnabled(enabled)
        self.theta_left_button.setEnabled(enabled)
        self.theta_right_button.setEnabled(enabled)
        self.radius_in_button.setEnabled(enabled)
        self.radius_out_button.setEnabled(enabled)
        self.z_down_button.setEnabled(enabled)
        self.z_up_button.setEnabled(enabled)

    def _set_robot_status(self, text: str) -> None:
        self._robot_status_text = str(text)
        self.robot_status_label.setText(self._robot_status_text)

    def _on_robot_worker_success(self, result: Dict[str, Any]) -> None:
        action = str(result.get("action", ""))
        if action in {"move_home_height", "move_cyl"}:
            self._last_robot_height_command = result
            status = result.get("after")
            self._last_robot_status = dict(status) if isinstance(status, dict) else None
            target = result.get("target") if isinstance(result.get("target"), dict) else {}
            theta_deg = float(target.get("theta_deg", self.robot_theta_spin.value()))
            radius_mm = float(target.get("radius_mm", self.robot_radius_spin.value()))
            z_mm = float(target.get("z_mm", self.home_height_spin.value()))
            self.robot_theta_spin.setValue(theta_deg)
            self.robot_radius_spin.setValue(radius_mm)
            self.home_height_spin.setValue(z_mm)
            self._set_robot_status(f"已到 theta={theta_deg:.1f}, r={radius_mm:.1f}, z={z_mm:.1f}")
            log_stderr(f"机械臂已移动: theta={theta_deg:.1f}, r={radius_mm:.1f}, z={z_mm:.1f}")
            return
        status = result.get("status")
        self._last_robot_status = dict(status) if isinstance(status, dict) else None
        self._sync_robot_pose_from_status(self._last_robot_status)
        self._set_robot_status(self._format_robot_status(self._last_robot_status))

    def _sync_robot_pose_from_status(self, status: Optional[Dict[str, Any]]) -> None:
        if not isinstance(status, dict):
            return
        robot_cyl = status.get("robot_cyl")
        if not isinstance(robot_cyl, dict):
            return
        try:
            self.robot_theta_spin.setValue(float(robot_cyl["theta_deg"]))
            self.robot_radius_spin.setValue(float(robot_cyl["radius_mm"]))
            self.home_height_spin.setValue(float(robot_cyl["z_mm"]))
        except (KeyError, TypeError, ValueError):
            return

    def _on_robot_worker_failed(self, message: str) -> None:
        self._set_robot_status(f"失败：{message}")
        QMessageBox.warning(self, "机械臂位置调整失败", message)
        log_stderr(f"机械臂位置调整失败: {message}")

    def _format_robot_status(self, status: Optional[Dict[str, Any]]) -> str:
        if not isinstance(status, dict):
            return "状态不可用"
        robot_cyl = status.get("robot_cyl")
        if isinstance(robot_cyl, dict):
            z_mm = robot_cyl.get("z_mm", status.get("robot_z", None))
            radius_mm = robot_cyl.get("radius_mm", None)
            state = status.get("state", "unknown")
            busy = "busy" if status.get("busy") else "idle"
            if z_mm is not None and radius_mm is not None:
                return f"{state}/{busy}, r={float(radius_mm):.1f} mm, z={float(z_mm):.1f} mm"
        state = status.get("state", "unknown")
        busy = "busy" if status.get("busy") else "idle"
        return f"{state}/{busy}"

    def _on_save_success(self, metadata: Dict[str, Any]) -> None:
        self._saved_count += 1
        self._queued_count = max(0, self._queued_count - 1)
        self._last_saved_path = str((self._session_dir / metadata["image_path"]).resolve()) if self._session_dir else metadata["image_path"]
        self.saved_count_label.setText(str(self._saved_count))
        self.queue_count_label.setText(str(self._queued_count))
        self.last_saved_label.setText(self._last_saved_path)
        if self._current_frame is not None:
            self._last_saved_frame = self._current_frame.copy()

    def _on_save_failed(self, message: str) -> None:
        self._queued_count = max(0, self._queued_count - 1)
        self.queue_count_label.setText(str(self._queued_count))
        QMessageBox.critical(self, "保存失败", message)
        log_stderr(f"保存失败: {message}")

    def _update_status_labels(self) -> None:
        snapshot = self.camera.status_snapshot()
        lines = [
            f"摄像头状态：{ui_camera_status(str(snapshot['status']))}",
            f"采集帧率：{snapshot['capture_fps']:.1f} FPS",
            f"当前帧延迟：{snapshot['last_frame_age_ms']:.1f} ms",
            f"画面尺寸：{self._current_frame_size[0]} x {self._current_frame_size[1]}",
            f"清晰度：{self._current_sharpness:.1f}",
            f"与上次保存差异：{self._current_delta:.1f}",
            f"自动采集：{'开启' if self._auto_capture_enabled else '关闭'}",
            f"剩余连拍：{self._burst_remaining}",
            f"负样本：{'是' if self.negative_check.isChecked() else '否'}",
            f"Home Z：{float(self.home_height_spin.value()):.1f} mm",
            f"机械臂：{self._robot_status_text}",
        ]
        self.status_label.setText("\n".join(lines))

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space:
            self._capture_now(mode="manual")
            return
        if event.key() == Qt.Key_B:
            self._start_burst_capture()
            return
        if event.key() == Qt.Key_A:
            self.auto_toggle.setChecked(not self.auto_toggle.isChecked())
            return
        if event.key() == Qt.Key_N:
            self.negative_check.setChecked(not self.negative_check.isChecked())
            return
        if event.key() == Qt.Key_S:
            self._create_session()
            return
        if event.key() == Qt.Key_H:
            self._move_robot_current_pose()
            return
        if event.key() == Qt.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self._auto_capture_enabled = False
        self._burst_remaining = 0
        if self._robot_worker is not None and self._robot_worker.isRunning():
            self._robot_worker.wait(200)
        super().closeEvent(event)


def load_config(argv: Optional[list[str]] = None) -> AppConfig:
    parser = argparse.ArgumentParser(description="采集 JetMax 木块训练图像")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="摄像头编号或视频流 URL")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="采集数据的保存根目录")
    parser.add_argument("--session-prefix", type=str, default="block_collect", help="自动生成会话名时使用的前缀")
    parser.add_argument("--image-ext", type=str, choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--exit-after-sec", type=float, default=0.0, help="运行 N 秒后自动退出，0 表示不启用")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--robot-host", type=str, default=DEFAULT_ROBOT_HOST, help="JetMax runtime 主机地址")
    parser.add_argument(
        "--robot-transport",
        choices=["auto", "ros", "tcp"],
        default=DEFAULT_ROBOT_TRANSPORT,
        help="机械臂位置控制链路；auto 优先 ROS，失败后回退 TCP",
    )
    parser.add_argument("--rosbridge-port", type=int, default=DEFAULT_ROSBRIDGE_PORT, help="JetMax ROS bridge 端口")
    parser.add_argument("--robot-port", type=int, default=DEFAULT_ROBOT_PORT, help="JetMax TCP legacy runtime 端口")
    parser.add_argument("--robot-timeout-sec", type=float, default=12.0, help="机械臂命令等待超时时间")
    parser.add_argument("--home-theta-deg", type=float, default=DEFAULT_HOME_THETA_DEG, help="无 runtime home_pose 时使用的 Home theta")
    parser.add_argument("--home-radius-mm", type=float, default=DEFAULT_HOME_RADIUS_MM, help="无 runtime home_pose 时使用的 Home 半径")
    parser.add_argument("--home-z-mm", type=float, default=DEFAULT_HOME_Z_MM, help="界面默认 Home 高度")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root).expanduser().resolve()
    home_z_mm = max(DEFAULT_HOME_Z_LIMITS_MM[0], min(DEFAULT_HOME_Z_LIMITS_MM[1], float(args.home_z_mm)))
    return AppConfig(
        source=parse_source(args.source),
        output_root=output_root,
        session_prefix=sanitize_token(args.session_prefix, "block_collect"),
        image_ext=args.image_ext.lower(),
        jpeg_quality=max(50, min(100, int(args.jpeg_quality))),
        auto_close_sec=max(0.0, float(args.exit_after_sec)),
        fullscreen=bool(args.fullscreen),
        robot_host=str(args.robot_host),
        robot_transport=str(args.robot_transport),
        rosbridge_port=int(args.rosbridge_port),
        robot_port=int(args.robot_port),
        robot_timeout_sec=max(0.2, float(args.robot_timeout_sec)),
        home_theta_deg=float(args.home_theta_deg),
        home_radius_mm=float(args.home_radius_mm),
        home_z_mm=home_z_mm,
    )


def main(argv: Optional[list[str]] = None) -> int:
    config = load_config(argv)
    config.output_root.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    camera = CameraLoader(config.source).start()
    writer = ImageWriter(config.image_ext, config.jpeg_quality)
    writer.start()

    window = CollectorWindow(config, camera, writer)
    if config.fullscreen:
        window.showFullScreen()
    else:
        window.show()

    exit_code = 0
    try:
        exit_code = app.exec_()
    finally:
        writer.request_stop()
        writer.wait(2000)
        camera.stop()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
