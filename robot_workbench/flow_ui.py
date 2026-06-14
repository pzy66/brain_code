from __future__ import annotations

import contextlib
import io
import math
import random
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.request import Request, urlopen

import numpy as np
from PyQt5.QtCore import QEvent, QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QKeyEvent, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from hybrid_controller.adapters.vision_adapter import VisionTarget
from hybrid_controller.app_robot_commands import build_pick_command_from_target
from hybrid_controller.adapters.robot_client import RobotClient
from hybrid_controller.adapters.rosbridge_client import RosServiceResult, RosbridgeClient
from hybrid_controller.adapters.teleop_ros_channel import new_teleop_cmd_seq_base, next_teleop_cmd_seq
from hybrid_controller.config import AppConfig, build_hiwonder_camera_stream_url
from hybrid_controller.cylindrical import cylindrical_to_cartesian


CAMERA_STREAM_TIMEOUT_SEC = 2.0
EEG_DISPLAY_CHANNEL_COUNT = 8
EEG_DEFAULT_CHANNEL_NAMES = ("Ch 1", "Ch 2", "Ch 3", "Ch 4", "Ch 5", "Ch 6", "Ch 7", "Ch 8")
EEG_AUTO_SERIAL_NAMES = {"", "auto", "detect", "auto-detect", "autodetect", "*"}
DEFAULT_OPERATOR_TARGET_COUNT = 4
MAX_OPERATOR_TARGET_COUNT = 4
TARGET_COLOR_PALETTE = (
    "#f6c667",
    "#38bdf8",
    "#a855f7",
    "#f43f5e",
)
_DEFAULT_APP_CONFIG = AppConfig()
DEFAULT_VISION_WEIGHTS_PATH = str(_DEFAULT_APP_CONFIG.vision_weights_path)
DEFAULT_VISION_CALIBRATION_PROFILE_PATH = str(_DEFAULT_APP_CONFIG.vision_calibration_profile_path)


def _bounded_target_count(value: object, default: int = DEFAULT_OPERATOR_TARGET_COUNT) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        count = int(default)
    return max(1, min(MAX_OPERATOR_TARGET_COUNT, count))


def _target_ids(count: object = DEFAULT_OPERATOR_TARGET_COUNT) -> tuple[str, ...]:
    return tuple(str(index) for index in range(1, _bounded_target_count(count) + 1))


def _target_color(target_id: object) -> str:
    try:
        index = int(str(target_id)) - 1
    except (TypeError, ValueError):
        index = 0
    return TARGET_COLOR_PALETTE[index % len(TARGET_COLOR_PALETTE)]


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return result


def _coerce_xy(value: object) -> tuple[float, float] | None:
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


def _coerce_xyz(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, (tuple, list)) or len(value) < 3:
        return None
    try:
        result = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in result):
        return None
    return result


def _coerce_box(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) < 4:
        return (0.0, 0.0, 0.0, 0.0)
    try:
        x1, y1, x2, y2 = (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    except (TypeError, ValueError):
        return (0.0, 0.0, 0.0, 0.0)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _vision_tracking_point(target: VisionTarget) -> tuple[float, float]:
    return target.grasp_pixel or target.display_center or target.center_px


def _vision_alignment_point(target: VisionTarget, packet: dict[str, object] | None = None) -> tuple[float, float]:
    packet_target = _coerce_xy((packet or {}).get("alignment_target_pixel"))
    return target.alignment_target_pixel or packet_target or (320.0, 240.0)


def _vision_center_distance_px(target: VisionTarget, packet: dict[str, object] | None = None) -> float:
    center = _vision_tracking_point(target)
    target_point = _vision_alignment_point(target, packet)
    return math.hypot(float(center[0]) - float(target_point[0]), float(center[1]) - float(target_point[1]))


def _vision_targets_from_packet(packet: dict[str, object]) -> list[VisionTarget]:
    targets: list[VisionTarget] = []
    slots = packet.get("slots", [])
    if not isinstance(slots, list):
        return targets
    packet_alignment_target = _coerce_xy(packet.get("alignment_target_pixel"))
    for index, slot_raw in enumerate(slots, start=1):
        if not isinstance(slot_raw, dict) or not bool(slot_raw.get("valid", False)):
            continue
        try:
            slot_id = int(slot_raw.get("slot_id", slot_raw.get("slot", index)))
        except (TypeError, ValueError):
            slot_id = index
        if slot_id < 1 or slot_id > MAX_OPERATOR_TARGET_COUNT:
            continue
        center = _coerce_xy(slot_raw.get("pixel_center_f")) or _coerce_xy(slot_raw.get("pixel_center")) or (0.0, 0.0)
        grasp_pixel = _coerce_xy(slot_raw.get("grasp_pixel_f")) or _coerce_xy(slot_raw.get("grasp_pixel")) or center
        command_point = _coerce_xy(slot_raw.get("command_point"))
        cylindrical_center = _coerce_xyz(slot_raw.get("cylindrical_center"))
        world_xyz = _coerce_xyz(slot_raw.get("world_xyz"))
        servo_command_point = _coerce_xy(slot_raw.get("servo_command_point"))
        alignment_target = _coerce_xy(slot_raw.get("alignment_target_pixel")) or packet_alignment_target
        targets.append(
            VisionTarget(
                id=slot_id,
                slot_id=slot_id,
                bbox=_coerce_box(slot_raw.get("bbox")),
                center_px=center,
                raw_center=center,
                display_center=grasp_pixel,
                confidence=_coerce_float(slot_raw.get("confidence"), 0.0),
                command_mode=str(slot_raw.get("command_mode", "world")),
                command_point=command_point,
                freq_hz=None if slot_raw.get("freq_hz") is None else _coerce_float(slot_raw.get("freq_hz"), 0.0),
                cylindrical_center=cylindrical_center,
                world_xyz=world_xyz,
                mapping_mode=str(slot_raw.get("mapping_mode", packet.get("mapping_mode", "absolute_base"))),
                actionable=bool(slot_raw.get("actionable", command_point is not None)),
                invalid_reason=str(slot_raw.get("invalid_reason", "")),
                grasp_pixel=grasp_pixel,
                undistorted_pixel=_coerce_xy(slot_raw.get("undistorted_pixel")),
                alignment_target_pixel=alignment_target,
                estimated_xy_error_mm=(
                    None
                    if slot_raw.get("estimated_xy_error_mm") is None
                    else _coerce_float(slot_raw.get("estimated_xy_error_mm"), float("inf"))
                ),
                servo_required=bool(slot_raw.get("servo_required", False)),
                servo_command_mode=str(slot_raw.get("servo_command_mode", "cyl")),
                servo_command_point=servo_command_point,
                calibration_profile_id=str(slot_raw.get("calibration_profile_id", "")),
                grasp_quality=_coerce_float(slot_raw.get("grasp_quality"), 0.0),
                grasp_angle_deg=(
                    None if slot_raw.get("grasp_angle_deg") is None else _coerce_float(slot_raw.get("grasp_angle_deg"), 0.0)
                ),
                grasp_angle_quality=_coerce_float(slot_raw.get("grasp_angle_quality"), 0.0),
            )
        )
    return targets


def _qimage_from_bgr_frame(frame: object) -> QImage | None:
    try:
        arr = np.asarray(frame)
    except Exception:
        return None
    if arr.size == 0 or arr.ndim < 2:
        return None
    if arr.ndim == 2:
        gray = np.ascontiguousarray(arr.astype(np.uint8, copy=False))
        h, w = gray.shape[:2]
        return QImage(gray.data, w, h, int(gray.strides[0]), QImage.Format_Grayscale8).copy()
    rgb = np.ascontiguousarray(arr[:, :, :3][:, :, ::-1].astype(np.uint8, copy=False))
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, int(rgb.strides[0]), QImage.Format_RGB888).copy()


def _serial_port_is_auto(serial_port: str | None) -> bool:
    return str(serial_port or "").strip().lower() in EEG_AUTO_SERIAL_NAMES


def _serial_sort_key(device: str) -> tuple[int, int | str]:
    value = str(device).strip().upper()
    if value.startswith("COM"):
        try:
            return (0, int(value[3:]))
        except ValueError:
            pass
    return (1, value)


def _serial_port_info_is_builtin_console(info: object) -> bool:
    device = str(getattr(info, "device", "") or "").strip().upper()
    description = str(getattr(info, "description", "") or "").strip().lower()
    hwid = str(getattr(info, "hwid", "") or "").strip().lower()
    if device not in {"COM1", "COM2"}:
        return False
    if "pnp0501" in hwid:
        return True
    return ("communications port" in description or "通信端口" in description) and "usb" not in hwid


def _detect_serial_port_candidates() -> list[str]:
    try:
        from serial.tools import list_ports
    except Exception:
        return []
    ports: list[str] = []
    try:
        for info in list_ports.comports():
            device = str(getattr(info, "device", "") or "").strip()
            if device and not _serial_port_info_is_builtin_console(info):
                ports.append(device)
    except Exception:
        return []
    return sorted(dict.fromkeys(ports), key=_serial_sort_key)


def _coerce_eeg_board_id(value: str | int) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


APP_STYLE = """
QMainWindow { background-color: #eef2f7; }
QWidget {
    color: #111827;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 16px;
}
QFrame#Header {
    background: #ffffff;
    border-bottom: 1px solid #cbd5e1;
}
QLabel#HeaderTitle {
    color: #0f766e;
    font-size: 18px;
    font-weight: 900;
}
QLabel#HeaderStatus {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    color: #475569;
    font-size: 16px;
    font-weight: 800;
    padding: 7px 14px;
}
QLabel#HeaderStatus[state="ready"] {
    background: #dcfce7;
    border-color: #86efac;
    color: #166534;
}
QLabel#HeaderStatus[state="active"] {
    background: #e0f2fe;
    border-color: #7dd3fc;
    color: #075985;
}
QFrame#SideRail {
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 6px;
}
QPushButton#NavButton {
    background: transparent;
    border: none;
    border-bottom: 3px solid transparent;
    color: #475569;
    font-size: 16px;
    font-weight: 800;
    padding: 0 16px;
}
QPushButton#NavButton[active="true"] {
    background: #f8fafc;
    border-bottom-color: #0f766e;
    color: #0f766e;
}
QPushButton#NavButton:disabled { color: #94a3b8; }
QFrame#Card, QGroupBox, QFrame#DeviceCard {
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 6px;
}
QFrame#DeviceCard {
    padding: 2px;
}
QFrame#BottomBar {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
}
QFrame#StageCard, QFrame#InfoTile, QFrame#ChecklistCard, QFrame#ConnectionSummary {
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 6px;
}
QFrame#InfoTile {
    background: #f8fafc;
}
QLabel#MetricTitle {
    color: #64748b;
    font-size: 14px;
    font-weight: 800;
}
QLabel#MetricValue {
    color: #0f172a;
    font-size: 21px;
    font-weight: 900;
}
QFrame#CameraShell {
    background: #020617;
    border: 1px solid #0f172a;
    border-radius: 6px;
}
QFrame#ViewportToolbar {
    background: #ffffff;
    border: 1px solid #dbeafe;
    border-radius: 4px;
}
QLabel#DeviceTitle {
    color: #0f172a;
    font-size: 19px;
    font-weight: 900;
}
QLabel#SectionLabel {
    color: #475569;
    font-weight: 800;
}
QLabel#StatusChip {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 9px;
    color: #475569;
    font-size: 14px;
    font-weight: 900;
    padding: 6px 12px;
}
QLabel#StatusChip[state="ready"] {
    background: #dcfce7;
    border-color: #86efac;
    color: #166534;
}
QLabel#StatusChip[state="active"] {
    background: #e0f2fe;
    border-color: #7dd3fc;
    color: #075985;
}
QLabel#StatusChip[state="warn"] {
    background: #fef3c7;
    border-color: #fbbf24;
    color: #92400e;
}
QLabel#StatusChip[state="danger"] {
    background: #fee2e2;
    border-color: #fca5a5;
    color: #991b1b;
}
QGroupBox {
    margin-top: 10px;
    padding-top: 10px;
    font-size: 17px;
    font-weight: 800;
    color: #0f766e;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
}
QLineEdit, QComboBox {
    background: #ffffff;
    border: 1px solid #94a3b8;
    border-radius: 4px;
    color: #0f172a;
    font-size: 16px;
    min-height: 34px;
    padding: 8px 10px;
}
QLineEdit:focus, QComboBox:focus {
    border-color: #0f766e;
}
QPushButton#ActionButton {
    background: #f8fafc;
    border: 1px solid #94a3b8;
    border-radius: 4px;
    color: #334155;
    font-size: 16px;
    font-weight: 800;
    min-height: 42px;
    padding: 10px 16px;
}
QPushButton#ActionButton:hover:enabled { background: #e2e8f0; }
QPushButton#AccentButton {
    background: #0f766e;
    border: 1px solid #115e59;
    border-radius: 4px;
    color: #ffffff;
    font-size: 16px;
    font-weight: 800;
    min-height: 42px;
    padding: 10px 16px;
}
QPushButton#AccentButton:hover:enabled { background: #115e59; }
QPushButton#DangerButton {
    background: #fee2e2;
    border: 1px solid #dc2626;
    border-radius: 4px;
    color: #991b1b;
    font-size: 16px;
    font-weight: 800;
    min-height: 42px;
    padding: 10px 16px;
}
QPushButton[blockState="pending"] {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    color: #334155;
    font-size: 16px;
    min-height: 40px;
    padding: 10px;
}
QPushButton[blockState="active"] {
    background: #e0f2fe;
    border: 2px solid #0284c7;
    border-radius: 4px;
    color: #0369a1;
    font-size: 16px;
    font-weight: 900;
    min-height: 40px;
    padding: 9px;
}
QProgressBar {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    color: #0f172a;
    font-weight: 800;
    text-align: center;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #14b8a6,stop:1 #0284c7);
    border-radius: 3px;
}
QPlainTextEdit#LogView {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 4px;
    color: #dbeafe;
    font-family: Consolas, "Microsoft YaHei", monospace;
    font-size: 12px;
}
"""


@dataclass(slots=True)
class WorkbenchConfig:
    robot_mode: str = "real"
    robot_transport: str = "ros"
    robot_host: str = "192.168.149.1"
    robot_port: int = 8888
    rosbridge_port: int = 9091
    connect_on_start: bool = False
    eeg_serial_port: str = "auto"
    eeg_board_id: int = 0
    eeg_signal_auto_start: bool = False
    eeg_signal_window_seconds: float = 2.0
    eeg_signal_poll_interval_sec: float = 0.03
    theta_rate_deg_s: float = 80.0
    radius_rate_mm_s: float = 160.0
    fake_motion_step_sec: float = 0.08
    move_stage_ms: int = 10_000
    camera_stream_url: str = ""
    camera_auto_start: bool = True
    target_count: int = DEFAULT_OPERATOR_TARGET_COUNT
    vision_enabled: bool = False
    vision_auto_start: bool = False
    vision_weights_path: str = DEFAULT_VISION_WEIGHTS_PATH
    vision_model_imgsz: int = 768
    vision_confidence_threshold: float = 0.25
    vision_max_targets: int = DEFAULT_OPERATOR_TARGET_COUNT
    vision_max_det: int = DEFAULT_OPERATOR_TARGET_COUNT
    vision_mapping_mode: str = "delta_servo"
    vision_calibration_profile_path: str = DEFAULT_VISION_CALIBRATION_PROFILE_PATH
    vision_calibration_profile_required: bool = True
    vision_center_tolerance_px: float = 28.0
    vision_center_stable_frames: int = 3
    vision_center_max_attempts: int = 40
    vision_servo_theta_gain_deg_per_px: float = 0.022
    vision_servo_radius_gain_mm_per_px: float = 0.085
    vision_servo_max_theta_step_deg: float = 3.0
    vision_servo_max_radius_step_mm: float = 8.0
    vision_pick_forward_offset_mm: float = 40.0
    robot_runtime_auto_start: bool = True
    robot_runtime_ssh_user: str = "hiwonder"
    robot_runtime_ssh_password: str = "hiwonder"
    robot_runtime_remote_root: str = "/home/hiwonder/brain_code"
    rosbridge_connect_timeout_sec: float = 4.0
    ros_state_timeout_sec: float = 4.0
    demo_connected: bool = False
    smoke_test_ms: int = 0


TARGET_POSES: dict[str, tuple[float, float]] = {
    "1": (-36.0, 132.0),
    "2": (-12.0, 150.0),
    "3": (18.0, 166.0),
    "4": (42.0, 182.0),
}


class RobotCommandBackend(QWidget):
    status_changed = pyqtSignal(str)
    connection_changed = pyqtSignal(bool)
    connect_progress_changed = pyqtSignal(int, str, str)
    pose_changed = pyqtSignal(float, float, float)
    command_finished = pyqtSignal(str, bool, str)

    def __init__(self, config: WorkbenchConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self._ros: RosbridgeClient | None = None
        self._tcp: RobotClient | None = None
        self._connected = False
        self._fake_theta = 0.0
        self._fake_radius = 150.0
        self._fake_z = 160.0
        self._last_fake_motion_ts = time.monotonic()
        self._teleop_seq = new_teleop_cmd_seq_base()
        self._connect_lock = threading.Lock()
        self._connect_in_progress = False
        self._last_connect_percent = 0
        self._last_connect_phase = ""
        self._last_connect_detail = ""
        self._last_state_snapshot: dict[str, object] | None = None
        self._closed = False

    @property
    def connected(self) -> bool:
        if self.config.robot_mode == "fake":
            return self._connected
        if self.config.robot_transport == "ros" and self._ros is not None:
            return self._ros.is_connected()
        if self._tcp is not None:
            return self._tcp.is_connected()
        return False

    def _emit_status(self, message: object, allow_closed: bool = False) -> None:
        if self._closed and not allow_closed:
            return
        try:
            self.status_changed.emit(str(message))
        except RuntimeError:
            pass

    def _emit_connection(self, connected: bool, allow_closed: bool = False) -> None:
        if self._closed and not allow_closed:
            return
        try:
            self.connection_changed.emit(bool(connected))
        except RuntimeError:
            pass

    def _emit_pose(self, theta: float, radius: float, z_mm: float, allow_closed: bool = False) -> None:
        if self._closed and not allow_closed:
            return
        try:
            self.pose_changed.emit(float(theta), float(radius), float(z_mm))
        except RuntimeError:
            pass

    def _emit_command_finished(self, action: object, ok: object, message: object, allow_closed: bool = False) -> None:
        if self._closed and not allow_closed:
            return
        try:
            self.command_finished.emit(str(action), bool(ok), str(message))
        except RuntimeError:
            pass

    def connect_robot(self) -> None:
        self._closed = False
        if self.config.robot_mode == "fake":
            self._connected = True
            self._last_state_snapshot = self._fake_state_snapshot()
            self._emit_connect_progress(100, "连接完成", "模拟机械臂已连接，可以进入控制界面。")
            self._emit_connection(True)
            self._emit_pose(self._fake_theta, self._fake_radius, self._fake_z)
            self._emit_status("fake backend connected; simulated pose active.")
            return
        if self.connected:
            self._emit_connect_progress(100, "连接完成", "机械臂控制链路已连接。")
            self._emit_connection(True)
            self._emit_status("robot backend already connected.")
            return
        with self._connect_lock:
            if self._connect_in_progress:
                self._emit_connect_progress(
                    max(1, self._last_connect_percent),
                    self._last_connect_phase or "连接中",
                    self._last_connect_detail or "机械臂连接任务已经在运行，请等待当前进度完成。",
                )
                return
            self._connect_in_progress = True
        port = self.config.rosbridge_port if self.config.robot_transport == "ros" else self.config.robot_port
        self._emit_connect_progress(2, "准备连接", f"正在准备连接 {self.config.robot_host}:{port}。")
        worker = threading.Thread(target=self._connect_worker, name="robot-workbench-connect", daemon=True)
        worker.start()

    def _connect_worker(self) -> None:
        try:
            if self.config.robot_transport == "ros":
                self._emit_connect_progress(
                    8,
                    "检查机械臂网络",
                    f"正在检查 rosbridge {self.config.robot_host}:{int(self.config.rosbridge_port)}。",
                )
                if not self._connect_rosbridge_once(attempt_label="初次连接", progress_start=10, progress_span=30):
                    if bool(self.config.robot_runtime_auto_start):
                        self._emit_connect_progress(
                            44,
                            "自动启动远端程序",
                            "rosbridge 或状态话题暂未就绪，开始通过 SSH 拉起机械臂端程序。",
                        )
                        self._emit_status("robot runtime is unavailable; starting JetMax runtime over SSH.")
                        if self._start_remote_ros_runtime():
                            self._emit_connect_progress(
                                74,
                                "重新连接机械臂",
                                "远端程序启动完成，正在重新连接 rosbridge 和状态话题。",
                            )
                            self._emit_status("robot runtime start finished; reconnecting ROS bridge.")
                            if not self._connect_rosbridge_once(
                                attempt_label="启动后重连",
                                progress_start=76,
                                progress_span=20,
                            ):
                                raise RuntimeError("远端程序已启动，但 ROS bridge 或状态话题仍不可用。")
                        else:
                            raise RuntimeError("通过 SSH 启动 JetMax 机械臂程序失败。")
                    else:
                        raise RuntimeError("ROS bridge 或机械臂状态话题不可用。")
            else:
                self._emit_connect_progress(
                    18,
                    "连接 TCP 控制端口",
                    f"正在连接 {self.config.robot_host}:{int(self.config.robot_port)}。",
                )
                self._tcp = RobotClient(self.config.robot_host, int(self.config.robot_port))
                self._tcp.connect()
                self._emit_connect_progress(86, "TCP 已连接", "机械臂 TCP 控制端口已连接。")
            self._connected = True
            self._emit_connection(True)
            self._emit_connect_progress(100, "连接完成", "机械臂已连接，可以进入控制界面。")
            self._emit_status("robot backend connected.")
        except Exception as error:
            self._connected = False
            self._emit_connection(False)
            self._emit_connect_progress(0, "连接失败", f"机械臂连接失败：{error}")
            self._emit_status(f"connect failed: {error}")
        finally:
            with self._connect_lock:
                self._connect_in_progress = False

    def _connect_rosbridge_once(
        self,
        *,
        attempt_label: str = "连接",
        progress_start: int = 10,
        progress_span: int = 30,
    ) -> bool:
        if self._ros is not None:
            self._close_rosbridge_silently()
        progress_start = max(0, min(99, int(progress_start)))
        progress_span = max(1, int(progress_span))
        progress_end = max(progress_start + 1, min(99, progress_start + progress_span))
        self._emit_connect_progress(
            progress_start,
            "连接 rosbridge",
            f"{attempt_label}：正在建立 rosbridge 会话 {self.config.robot_host}:{int(self.config.rosbridge_port)}。",
        )
        self._ros = RosbridgeClient(
            self.config.robot_host,
            int(self.config.rosbridge_port),
            state_callback=self._on_ros_state,
            status_callback=self._on_rosbridge_status,
        )
        try:
            self._ros.connect()
        except Exception as error:
            self._emit_connect_progress(
                min(progress_end, progress_start + max(1, progress_span // 3)),
                "等待 rosbridge",
                f"{attempt_label}：rosbridge 暂未响应，稍后会继续处理。{error}",
            )
            self._close_rosbridge_silently()
            return False
        connect_timeout = max(0.5, float(self.config.rosbridge_connect_timeout_sec))
        connect_start = time.monotonic()
        connect_deadline = connect_start + connect_timeout
        last_emit = 0.0
        while time.monotonic() < connect_deadline:
            if self._ros.is_connected():
                break
            now = time.monotonic()
            if now - last_emit >= 0.35:
                fraction = min(1.0, max(0.0, (now - connect_start) / connect_timeout))
                percent = progress_start + int((progress_end - progress_start) * 0.45 * fraction)
                self._emit_connect_progress(
                    percent,
                    "等待 rosbridge",
                    f"{attempt_label}：已连上机械臂 WiFi，正在等待 rosbridge 响应。",
                )
                last_emit = now
            time.sleep(0.05)
        if not self._ros.is_connected():
            self._emit_connect_progress(
                progress_start + int((progress_end - progress_start) * 0.45),
                "等待 rosbridge",
                "rosbridge 暂未响应；如果机械臂刚启动，软件会继续尝试自动拉起远端程序。",
            )
            self._close_rosbridge_silently()
            return False
        self._emit_connect_progress(
            progress_start + int((progress_end - progress_start) * 0.52),
            "等待状态话题",
            "rosbridge 已连接，正在等待 /hybrid_controller/state 返回实时姿态。",
        )
        state_timeout = max(0.2, float(self.config.ros_state_timeout_sec))
        state_start = time.monotonic()
        state_deadline = state_start + state_timeout
        last_emit = 0.0
        while time.monotonic() < state_deadline:
            snapshot = self._ros.latest_state_snapshot()
            if snapshot is not None:
                self._on_ros_state(snapshot)
                self._emit_connect_progress(
                    progress_end,
                    "状态已确认",
                    "已收到机械臂实时状态，控制链路可用。",
                )
                return True
            now = time.monotonic()
            if now - last_emit >= 0.35:
                fraction = min(1.0, max(0.0, (now - state_start) / state_timeout))
                percent = progress_start + int((progress_end - progress_start) * (0.55 + 0.40 * fraction))
                self._emit_connect_progress(
                    percent,
                    "等待状态话题",
                    f"{attempt_label}：rosbridge 已连接，正在等待机械臂运行节点发布状态。",
                )
                last_emit = now
            time.sleep(0.05)
        self._emit_connect_progress(
            progress_end,
            "状态话题未就绪",
            "rosbridge 已连接，但机械臂运行节点还没有发布状态；准备继续自动处理。",
        )
        self._close_rosbridge_silently()
        return False

    def _start_remote_ros_runtime(self) -> bool:
        try:
            from hybrid_controller.robot.tools import jetmax_start_ros_runtime
        except Exception as error:
            self._emit_status(f"robot runtime start unavailable: {error}")
            return False
        args = [
            "--host",
            str(self.config.robot_host),
            "--user",
            str(self.config.robot_runtime_ssh_user),
            "--password",
            str(self.config.robot_runtime_ssh_password),
            "--remote-root",
            str(self.config.robot_runtime_remote_root),
            "--rosbridge-port",
            str(int(self.config.rosbridge_port)),
            "--skip-camera-check",
            "--skip-tcp-check",
        ]
        stdout = io.StringIO()
        stderr = io.StringIO()
        done = threading.Event()

        def heartbeat() -> None:
            started = time.monotonic()
            while not done.wait(0.8):
                elapsed = time.monotonic() - started
                percent = min(72, 48 + int(elapsed * 2.0))
                self._emit_connect_progress(
                    percent,
                    "自动启动远端程序",
                    f"正在通过 SSH 拉起机械臂端程序，已等待 {elapsed:.0f} 秒。",
                )

        heartbeat_thread = threading.Thread(target=heartbeat, name="robot-runtime-start-progress", daemon=True)
        heartbeat_thread.start()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = int(jetmax_start_ros_runtime.main(args))
        except Exception as error:
            done.set()
            heartbeat_thread.join(timeout=0.3)
            details = "\n".join(part.strip() for part in (stdout.getvalue(), stderr.getvalue()) if part.strip())
            if details:
                for line in details.splitlines()[-8:]:
                    self._emit_status(f"robot-start: {line}")
            self._emit_status(f"robot runtime start failed: {error}")
            self._emit_connect_progress(0, "远端启动失败", f"通过 SSH 拉起机械臂端程序失败：{error}")
            return False
        finally:
            done.set()
            heartbeat_thread.join(timeout=0.3)
        details = "\n".join(part.strip() for part in (stdout.getvalue(), stderr.getvalue()) if part.strip())
        if details:
            for line in details.splitlines()[-8:]:
                self._emit_status(f"robot-start: {line}")
        if exit_code == 0:
            self._emit_connect_progress(72, "远端启动完成", "机械臂端程序已启动，准备重新确认 rosbridge 状态。")
        else:
            self._emit_connect_progress(0, "远端启动失败", f"机械臂端启动脚本返回退出码 {exit_code}。")
        return exit_code == 0

    def close_backend(self) -> None:
        self._closed = True
        try:
            if self._ros is not None:
                self._ros.close()
            if self._tcp is not None:
                self._tcp.close()
        finally:
            self._connected = False
            self._emit_connection(False, allow_closed=True)
            self._emit_connect_progress(0, "已断开", "机械臂连接已断开。")

    def _emit_connect_progress(self, percent: int, phase: str, detail: str, allow_closed: bool = False) -> None:
        value = max(0, min(100, int(percent)))
        self._last_connect_percent = value
        self._last_connect_phase = str(phase)
        self._last_connect_detail = str(detail)
        if self._closed and not allow_closed:
            return
        try:
            self.connect_progress_changed.emit(value, self._last_connect_phase, self._last_connect_detail)
        except RuntimeError:
            pass

    def _close_rosbridge_silently(self) -> None:
        try:
            if self._ros is not None:
                self._ros.close()
        except Exception:
            pass
        finally:
            self._ros = None

    def _on_rosbridge_status(self, message: str) -> None:
        text = str(message)
        if self._connect_in_progress:
            lower = text.lower()
            if "error" in lower:
                self._emit_connect_progress(
                    max(1, self._last_connect_percent),
                    self._last_connect_phase or "等待 rosbridge",
                    "rosbridge 暂时未响应，仍在等待或准备自动拉起远端程序。",
                )
                return
            if "closed" in lower and self._last_connect_percent < 100:
                return
        self._emit_status(text)

    def _on_ros_state(self, snapshot: dict[str, object]) -> None:
        self._last_state_snapshot = dict(snapshot)
        robot_cyl = snapshot.get("robot_cyl")
        if isinstance(robot_cyl, dict):
            try:
                theta = float(robot_cyl.get("theta_deg", 0.0))
                radius = float(robot_cyl.get("radius_mm", 0.0))
                z_mm = float(robot_cyl.get("z_mm", 0.0))
                self._emit_pose(theta, radius, z_mm)
                return
            except (TypeError, ValueError):
                pass
        try:
            theta = float(snapshot.get("theta_deg", self._fake_theta))
            radius = float(snapshot.get("radius_mm", self._fake_radius))
            z_mm = float(snapshot.get("z_mm", self._fake_z))
            self._emit_pose(theta, radius, z_mm)
        except (TypeError, ValueError):
            return

    def _fake_state_snapshot(self) -> dict[str, object]:
        x_mm, y_mm, _ = cylindrical_to_cartesian(self._fake_theta, self._fake_radius, self._fake_z)
        return {
            "robot_ts": time.time(),
            "robot_xy": [float(x_mm), float(y_mm)],
            "robot_z": float(self._fake_z),
            "robot_cyl": {
                "theta_deg": float(self._fake_theta),
                "radius_mm": float(self._fake_radius),
                "z_mm": float(self._fake_z),
            },
            "limits_cyl": {
                "theta_deg": [-120.0, 120.0],
                "radius_mm": [50.0, 280.0],
            },
        }

    def latest_state_snapshot(self) -> dict[str, object] | None:
        if self.config.robot_transport == "ros" and self._ros is not None:
            snapshot = self._ros.latest_state_snapshot()
            if snapshot is not None:
                self._last_state_snapshot = dict(snapshot)
                return dict(snapshot)
        if self.config.robot_mode == "fake":
            self._last_state_snapshot = self._fake_state_snapshot()
        if self._last_state_snapshot is None:
            return None
        return dict(self._last_state_snapshot)

    def send_teleop(self, theta_rate_deg_s: float, radius_rate_mm_s: float) -> None:
        theta_rate = float(theta_rate_deg_s)
        radius_rate = float(radius_rate_mm_s)
        if self.config.robot_mode == "fake":
            now = time.monotonic()
            dt = min(0.2, max(0.0, now - self._last_fake_motion_ts))
            if dt <= 0.0:
                dt = float(self.config.fake_motion_step_sec)
            self._last_fake_motion_ts = now
            self._fake_theta = max(-120.0, min(120.0, self._fake_theta + theta_rate * dt))
            self._fake_radius = max(80.0, min(260.0, self._fake_radius + radius_rate * dt))
            self._last_state_snapshot = self._fake_state_snapshot()
            self._emit_pose(self._fake_theta, self._fake_radius, self._fake_z)
            return
        if self.config.robot_transport == "ros" and self._ros is not None and self._ros.is_connected():
            self._teleop_seq = next_teleop_cmd_seq(self._teleop_seq)
            self._ros.publish_teleop(
                theta_rate_deg_s=theta_rate,
                radius_rate_mm_s=radius_rate,
                z_rate_mm_s=0.0,
                use_auto_z=True,
                enabled=bool(theta_rate or radius_rate),
                cmd_seq=int(self._teleop_seq),
                client_ts=time.time(),
            )

    def stop_teleop(self) -> None:
        if self.config.robot_mode == "fake":
            self._last_fake_motion_ts = time.monotonic()
            return
        if self.config.robot_transport == "ros" and self._ros is not None:
            self._teleop_seq = next_teleop_cmd_seq(self._teleop_seq)
            self._ros.stop_teleop(cmd_seq=int(self._teleop_seq))

    def _current_cyl_pose_and_limits(
        self,
    ) -> tuple[float, float, float, tuple[float, float], tuple[float, float]] | None:
        snapshot = self.latest_state_snapshot()
        if not isinstance(snapshot, dict):
            return None
        cyl = snapshot.get("robot_cyl")
        if not isinstance(cyl, dict):
            return None
        try:
            theta = float(cyl.get("theta_deg", 0.0))
            radius = float(cyl.get("radius_mm", 0.0))
            z_mm = float(cyl.get("z_mm", snapshot.get("robot_z", 160.0)))
        except (TypeError, ValueError):
            return None
        limits = snapshot.get("limits_cyl")
        theta_limits = (-120.0, 120.0)
        radius_limits = (50.0, 280.0)
        if isinstance(limits, dict):
            theta_pair = _coerce_xy(limits.get("theta_deg"))
            radius_pair = _coerce_xy(limits.get("radius_mm"))
            if theta_pair is not None:
                theta_limits = (float(theta_pair[0]), float(theta_pair[1]))
            if radius_pair is not None:
                radius_limits = (float(radius_pair[0]), float(radius_pair[1]))
        return theta, radius, z_mm, theta_limits, radius_limits

    def _camera_center_servo_point(self, target: VisionTarget | None) -> tuple[float, float] | None:
        if target is None:
            return None
        pose = self._current_cyl_pose_and_limits()
        if pose is None:
            return None
        theta, radius, _z_mm, theta_limits, radius_limits = pose
        center = _vision_tracking_point(target)
        target_point = _vision_alignment_point(target)
        err_x = float(center[0]) - float(target_point[0])
        err_y = float(center[1]) - float(target_point[1])
        dtheta = _clamp_float(
            -err_x * float(self.config.vision_servo_theta_gain_deg_per_px),
            -float(self.config.vision_servo_max_theta_step_deg),
            float(self.config.vision_servo_max_theta_step_deg),
        )
        dradius = _clamp_float(
            -err_y * float(self.config.vision_servo_radius_gain_mm_per_px),
            -float(self.config.vision_servo_max_radius_step_mm),
            float(self.config.vision_servo_max_radius_step_mm),
        )
        next_theta = _clamp_float(theta + dtheta, theta_limits[0], theta_limits[1])
        next_radius = _clamp_float(radius + dradius, radius_limits[0], radius_limits[1])
        return float(next_theta), float(next_radius)

    def pick_target(self, target_id: str) -> None:
        theta, radius = TARGET_POSES.get(str(target_id), TARGET_POSES["2"])
        if self.config.robot_mode == "fake":
            self._emit_status(f"fake pick target {target_id}: theta={theta:.1f}, radius={radius:.1f}")
            QTimer.singleShot(250, lambda: self._emit_command_finished("pick", True, "fake pick complete"))
            return
        callback = self._callback_for("pick")
        if self.config.robot_transport == "ros" and self._ros is not None and self._ros.is_connected():
            try:
                self._ros.send_pick_cyl(theta, radius, callback=callback)
            except Exception as error:
                self._emit_command_finished("pick", False, str(error))
            return
        if self._tcp is not None and self._tcp.is_connected():
            try:
                self._tcp.send_pick_cyl(theta, radius)
                self._emit_command_finished("pick", True, "tcp pick command sent")
            except Exception as error:
                self._emit_command_finished("pick", False, str(error))
            return
        self._emit_command_finished("pick", False, "robot is not connected")

    def pick_vision_target(self, target_id: str, target: VisionTarget | None) -> None:
        if target is None:
            self.command_finished.emit("pick", False, f"目标 {target_id} 还没有视觉中心坐标")
            return
        command = build_pick_command_from_target(target)
        if not command:
            reason = str(target.invalid_reason or "视觉目标未达到可抓取状态")
            self._emit_command_finished("pick", False, reason)
            return
        if self.config.robot_mode == "fake":
            self._emit_status(f"fake vision pick target {target_id}: {command}")
            QTimer.singleShot(250, lambda: self._emit_command_finished("pick", True, "fake vision pick complete"))
            return
        self._send_robot_pick_command(command)

    def pick_camera_center_target(self, target_id: str, target: VisionTarget | None) -> None:
        pose = self._current_cyl_pose_and_limits()
        if pose is None:
            self._emit_command_finished("pick", False, "robot pose is unavailable for camera-center pick")
            return
        theta, radius, _z_mm, _theta_limits, radius_limits = pose
        pick_radius = _clamp_float(
            radius + float(self.config.vision_pick_forward_offset_mm),
            radius_limits[0],
            radius_limits[1],
        )
        angle = None
        if target is not None and float(target.grasp_angle_quality or 0.0) >= 0.20 and target.grasp_angle_deg is not None:
            angle = float(target.grasp_angle_deg)
        if self.config.robot_mode == "fake":
            self._emit_status(
                f"fake camera-center pick target {target_id}: theta={theta:.1f}, radius={pick_radius:.1f}"
            )
            QTimer.singleShot(250, lambda: self._emit_command_finished("pick", True, "fake camera-center pick complete"))
            return
        callback = self._callback_for("pick")
        if self.config.robot_transport == "ros" and self._ros is not None and self._ros.is_connected():
            try:
                self._ros.send_pick_cyl(float(theta), float(pick_radius), sucker_rotation_deg=angle, callback=callback)
            except Exception as error:
                self._emit_command_finished("pick", False, str(error))
            return
        if self._tcp is not None and self._tcp.is_connected():
            try:
                self._tcp.send_pick_cyl(float(theta), float(pick_radius), sucker_rotation_deg=angle)
                self._emit_command_finished("pick", True, "tcp camera-center pick command sent")
            except Exception as error:
                self._emit_command_finished("pick", False, str(error))
            return
        self._emit_command_finished("pick", False, "robot is not connected")

    def align_to_vision_target(self, target_id: str, target: VisionTarget | None) -> bool:
        if target is None:
            return False
        if target.servo_command_point is not None:
            theta, radius = target.servo_command_point
        else:
            servo_point = self._camera_center_servo_point(target)
            if servo_point is None:
                return False
            theta, radius = servo_point
        if self.config.robot_mode == "fake":
            self._fake_theta = float(theta)
            self._fake_radius = float(radius)
            self._last_state_snapshot = self._fake_state_snapshot()
            self._emit_pose(self._fake_theta, self._fake_radius, self._fake_z)
            self._emit_command_finished("vision-align", True, f"fake aligned target {target_id}")
            return True
        callback = self._callback_for("vision-align")
        if self.config.robot_transport == "ros" and self._ros is not None and self._ros.is_connected():
            try:
                self._ros.send_move_cyl_auto(float(theta), float(radius), callback=callback)
            except Exception as error:
                self._emit_command_finished("vision-align", False, str(error))
            return True
        if self._tcp is not None and self._tcp.is_connected():
            try:
                self._tcp.send_move_cyl_auto(float(theta), float(radius))
                self._emit_command_finished("vision-align", True, "tcp vision align command sent")
            except Exception as error:
                self._emit_command_finished("vision-align", False, str(error))
            return True
        self._emit_command_finished("vision-align", False, "robot is not connected")
        return True

    def _send_robot_pick_command(self, command: str) -> None:
        parts = str(command).strip().split()
        if len(parts) not in {3, 4}:
            self._emit_command_finished("pick", False, f"invalid pick command: {command}")
            return
        op = parts[0].upper()
        angle = None if len(parts) < 4 else _coerce_float(parts[3], 0.0)
        callback = self._callback_for("pick")
        try:
            a_value = float(parts[1])
            b_value = float(parts[2])
        except (TypeError, ValueError) as error:
            self._emit_command_finished("pick", False, f"invalid pick coordinate: {error}")
            return
        if self.config.robot_transport == "ros" and self._ros is not None and self._ros.is_connected():
            try:
                if op == "PICK_WORLD":
                    self._ros.send_pick_world(a_value, b_value, sucker_rotation_deg=angle, callback=callback)
                    return
                if op == "PICK_CYL":
                    self._ros.send_pick_cyl(a_value, b_value, sucker_rotation_deg=angle, callback=callback)
                    return
                raise RuntimeError(f"unsupported pick command: {op}")
            except Exception as error:
                self._emit_command_finished("pick", False, str(error))
                return
        if self._tcp is not None and self._tcp.is_connected():
            try:
                if op == "PICK_CYL":
                    self._tcp.send_pick_cyl(a_value, b_value, sucker_rotation_deg=angle)
                elif op == "PICK_WORLD":
                    self._tcp.send_command(command)
                else:
                    raise RuntimeError(f"unsupported pick command: {op}")
                self._emit_command_finished("pick", True, "tcp vision pick command sent")
            except Exception as error:
                self._emit_command_finished("pick", False, str(error))
            return
        self._emit_command_finished("pick", False, "robot is not connected")

    def place(self) -> None:
        if self.config.robot_mode == "fake":
            self._emit_status("fake place command")
            QTimer.singleShot(250, lambda: self._emit_command_finished("place", True, "fake place complete"))
            return
        callback = self._callback_for("place")
        if self.config.robot_transport == "ros" and self._ros is not None and self._ros.is_connected():
            try:
                self._ros.send_place(callback=callback)
            except Exception as error:
                self._emit_command_finished("place", False, str(error))
            return
        if self._tcp is not None and self._tcp.is_connected():
            try:
                self._tcp.send_place()
                self._emit_command_finished("place", True, "tcp place command sent")
            except Exception as error:
                self._emit_command_finished("place", False, str(error))
            return
        self._emit_command_finished("place", False, "robot is not connected")

    def abort(self) -> None:
        self.stop_teleop()
        if self.config.robot_mode == "fake":
            self._emit_command_finished("abort", True, "fake abort complete")
            return
        try:
            if self.config.robot_transport == "ros" and self._ros is not None:
                self._ros.send_abort(callback=self._callback_for("abort"))
            elif self._tcp is not None:
                self._tcp.send_command("ABORT")
        except Exception as error:
            self._emit_status(f"abort failed: {error}")

    def reset(self) -> None:
        self.stop_teleop()
        if self.config.robot_mode == "fake":
            self._fake_theta, self._fake_radius, self._fake_z = 0.0, 150.0, 160.0
            self._emit_pose(self._fake_theta, self._fake_radius, self._fake_z)
            self._emit_command_finished("reset", True, "fake reset complete")
            return
        try:
            if self.config.robot_transport == "ros" and self._ros is not None:
                self._ros.send_reset(callback=self._callback_for("reset"))
            elif self._tcp is not None:
                self._tcp.send_command("RESET")
        except Exception as error:
            self._emit_status(f"reset failed: {error}")

    def sucker_off(self) -> None:
        self.stop_teleop()
        if self.config.robot_mode == "fake":
            self._emit_command_finished("sucker_off", True, "fake sucker off complete")
            return
        try:
            if self.config.robot_transport == "ros" and self._ros is not None:
                self._ros.send_sucker_off(callback=self._callback_for("sucker_off"))
            elif self._tcp is not None:
                self._tcp.send_command("SUCKER_OFF")
        except Exception as error:
            self._emit_status(f"sucker off failed: {error}")

    def _callback_for(self, action: str) -> Callable[[RosServiceResult], None]:
        def _callback(result: RosServiceResult) -> None:
            self._emit_command_finished(str(action), bool(result.ok), str(result.message or ""))

        return _callback


class PipelineProgressWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(70)
        self.current_stage = 1
        self.stage_percent = 0.0
        self._display_scale = 1.0

    def set_display_scale(self, scale: float) -> None:
        value = max(1.0, min(1.4, float(scale)))
        if abs(value - self._display_scale) < 0.01:
            return
        self._display_scale = value
        self.setFixedHeight(int(round(70 * value)))
        self.update()

    def set_stage_progress(self, stage: int, percent: float) -> None:
        self.current_stage = max(1, min(4, int(stage)))
        self.stage_percent = max(0.0, min(100.0, float(percent)))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#f8fbff"))
        scale = self._display_scale
        margin, spacing = 18.0 * scale, 12.0 * scale
        labels = ["SSVEP 采集", "MI 采集", "模型训练", "机械臂控制"]
        total_w = max(1.0, self.width() - margin * 2 - spacing * (len(labels) - 1))
        block_w = total_w / float(len(labels))
        for idx, label in enumerate(labels, start=1):
            x = margin + (idx - 1) * (block_w + spacing)
            bar = QRectF(x, 38.0 * scale, block_w, 16.0 * scale)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#e2e8f0"))
            painter.drawRoundedRect(bar, 4, 4)
            fill_ratio = 0.0
            if self.current_stage > idx:
                fill_ratio = 1.0
            elif self.current_stage == idx:
                fill_ratio = self.stage_percent / 100.0
            if fill_ratio > 0.0:
                painter.setBrush(QColor("#0284c7"))
                painter.drawRoundedRect(QRectF(bar.left(), bar.top(), max(4.0, bar.width() * fill_ratio), bar.height()), 4, 4)
            painter.setFont(QFont("Microsoft YaHei", max(12, int(round(12 * scale))), QFont.Bold))
            painter.setPen(QColor("#0369a1") if self.current_stage == idx else QColor("#64748b"))
            suffix = "100%" if self.current_stage > idx else (f"{int(self.stage_percent)}%" if self.current_stage == idx else "等待")
            painter.drawText(QRectF(x, 6.0 * scale, block_w, 28.0 * scale), Qt.AlignCenter, f"{label} ({suffix})")


class BrainFlowEegStreamThread(QThread):
    """Lightweight BrainFlow reader for live preview only."""

    stream_ready = pyqtSignal(object)
    samples_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        *,
        serial_port: str,
        board_id: int,
        poll_interval_sec: float = 0.05,
        channel_count: int = EEG_DISPLAY_CHANNEL_COUNT,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.serial_port = str(serial_port or "auto").strip() or "auto"
        self.board_id = int(board_id)
        self.poll_interval_sec = max(0.02, float(poll_interval_sec))
        self.channel_count = max(1, int(channel_count))
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    @staticmethod
    def _is_transient_buffer_error(error: Exception) -> bool:
        text = f"{type(error).__name__}: {error}".upper()
        transient_tokens = (
            "BOARD_NOT_CREATED",
            "BOARD_NOT_READY",
            "EMPTY_BUFFER",
            "INVALID_BUFFER_SIZE",
            "STREAM_THREAD_IS_NOT_RUNNING",
            "OBTAIN BUFFER SIZE",
        )
        return any(token in text for token in transient_tokens)

    def _wait_for_stream_buffer(self, board: Any, sampling_rate: float, *, timeout_sec: float = 4.0) -> int:
        target = max(4, min(32, int(float(sampling_rate or 250.0) * 0.12)))
        deadline = time.monotonic() + max(0.8, float(timeout_sec))
        last_error: Exception | None = None
        last_status = 0.0
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            try:
                available = int(board.get_board_data_count())
            except Exception as error:
                last_error = error
                available = 0
                if not self._is_transient_buffer_error(error):
                    raise
            if available >= target:
                return available
            now = time.monotonic()
            if now - last_status >= 0.6:
                self.status_changed.emit(f"等待 BrainFlow 数据缓冲区就绪: {available}/{target} samples")
                last_status = now
            time.sleep(0.05)
        if self._stop_event.is_set():
            return 0
        detail = f" 最后错误: {last_error}" if last_error is not None else ""
        raise RuntimeError(
            "BrainFlow 数据流未就绪。请确认脑电设备已开机、串口未被其他程序占用，并等待 2-3 秒后重试。"
            f"{detail}"
        )

    def _serial_candidates(self) -> list[str]:
        if not _serial_port_is_auto(self.serial_port):
            return [self.serial_port]
        if int(self.board_id) == -1:
            return [""]
        candidates = _detect_serial_port_candidates()
        if candidates:
            return candidates
        return []

    def _prepare_board(self, BoardShim: Any, BrainFlowInputParams: Any) -> tuple[Any, str, list[str]]:
        candidates = self._serial_candidates()
        if not candidates:
            raise RuntimeError("未检测到可用串口。请在连接界面填写 COMx，或确认脑电帽接收器已插入。")

        attempted: list[str] = []
        last_error: Exception | None = None
        for candidate in candidates:
            params = BrainFlowInputParams()
            if candidate:
                params.serial_port = candidate
            board = BoardShim(int(self.board_id), params)
            try:
                self.status_changed.emit(
                    f"正在连接 BrainFlow: board_id={self.board_id}, serial={candidate or 'synthetic'}"
                )
                board.prepare_session()
                attempted.append(candidate or "synthetic")
                return board, candidate, attempted
            except Exception as error:
                attempted.append(candidate or "synthetic")
                last_error = error
                with contextlib.suppress(Exception):
                    board.release_session()
                if not _serial_port_is_auto(self.serial_port):
                    raise

        attempted_text = ", ".join(attempted)
        raise RuntimeError(f"BrainFlow 自动串口连接失败，已尝试: {attempted_text}. 最后错误: {last_error}") from last_error

    def run(self) -> None:
        board = None
        streaming = False
        try:
            import brainflow_compat  # noqa: F401
            from brainflow.board_shim import BoardShim, BrainFlowInputParams

            board, resolved_serial, attempted = self._prepare_board(BoardShim, BrainFlowInputParams)
            eeg_rows = [int(row) for row in BoardShim.get_eeg_channels(int(self.board_id))]
            selected_rows = eeg_rows[: self.channel_count]
            if not selected_rows:
                raise RuntimeError(f"board_id={self.board_id} 未返回 EEG 通道。")
            sampling_rate = float(BoardShim.get_sampling_rate(int(self.board_id)))
            board.start_stream(450000)
            streaming = True
            ready_samples = self._wait_for_stream_buffer(board, sampling_rate)
            self.status_changed.emit(f"BrainFlow 缓冲区已就绪: {ready_samples} samples")
            with contextlib.suppress(Exception):
                board.get_board_data(min(ready_samples, max(1, int(sampling_rate * 0.5))))
            channel_names = [f"Ch {index + 1}" for index in range(len(selected_rows))]
            self.stream_ready.emit(
                {
                    "sampling_rate": sampling_rate,
                    "channel_names": channel_names,
                    "selected_rows": selected_rows,
                    "board_id": int(self.board_id),
                    "serial_port": resolved_serial or "synthetic",
                    "attempted_serial_ports": attempted,
                }
            )
            self.status_changed.emit(
                f"实时 EEG 已连接: {len(selected_rows)} 通道, {sampling_rate:g} Hz, serial={resolved_serial or 'synthetic'}"
            )

            consecutive_read_errors = 0
            max_read_samples = max(8, int(sampling_rate * max(self.poll_interval_sec * 3.0, 0.12)))
            while not self._stop_event.wait(self.poll_interval_sec):
                try:
                    available = int(board.get_board_data_count())
                    if available <= 0:
                        continue
                    chunk = board.get_board_data(min(available, max_read_samples))
                    consecutive_read_errors = 0
                except Exception as error:
                    if self._is_transient_buffer_error(error) and consecutive_read_errors < 5:
                        consecutive_read_errors += 1
                        self.status_changed.emit(f"等待 BrainFlow 数据恢复: {consecutive_read_errors}/5")
                        time.sleep(0.08)
                        continue
                    self.error_occurred.emit(
                        "读取 BrainFlow 数据失败。请确认脑电设备仍在线、串口没有被占用，然后重新连接。"
                        f" 原始错误: {error}"
                    )
                    break
                if chunk is None or not np.size(chunk):
                    continue
                try:
                    preview_chunk = np.asarray(chunk[selected_rows, :], dtype=np.float32)
                except Exception as error:
                    self.error_occurred.emit(f"解析 EEG 通道失败: {error}")
                    continue
                if preview_chunk.ndim == 2 and preview_chunk.shape[1] > 0:
                    self.samples_ready.emit(preview_chunk)
        except Exception as error:
            self.error_occurred.emit(f"实时 EEG 启动失败: {error}")
        finally:
            if board is not None:
                if streaming:
                    with contextlib.suppress(Exception):
                        board.stop_stream()
                with contextlib.suppress(Exception):
                    board.release_session()
            self.status_changed.emit("实时 EEG 已停止。")


class SignalPreviewWidget(QWidget):
    def __init__(self, parent: QWidget | None = None, *, window_seconds: float = 2.0) -> None:
        super().__init__(parent)
        self.mode = "EEG"
        self.window_seconds = max(1.0, float(window_seconds))
        self.sampling_rate = 0.0
        self.channel_names = list(EEG_DEFAULT_CHANNEL_NAMES)
        self.max_points = 300
        self._sample_data = np.zeros((EEG_DISPLAY_CHANNEL_COUNT, self.max_points), dtype=np.float32)
        self._write_index = 0
        self._sample_count = 0
        self._plot_cache: list[np.ndarray] = [np.empty(0, dtype=np.float32) for _ in range(EEG_DISPLAY_CHANNEL_COUNT)]
        self._stats_cache: list[dict[str, float | str | None]] = [self._empty_channel_stats() for _ in range(EEG_DISPLAY_CHANNEL_COUNT)]
        self._cache_dirty = True
        self._cached_plot_columns = 0
        self._last_cache_perf = 0.0
        self._last_live_repaint_perf = 0.0
        self.live_stream_active = False
        self.last_sample_perf = 0.0
        self.status_text = "未连接真实 EEG 数据流，当前为占位波形。"
        self._dirty = True
        self.setMinimumHeight(420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._seed_placeholder()
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    @property
    def buffers(self) -> list[deque[float]]:
        return [deque(self._channel_window(channel_index).tolist(), maxlen=self.max_points) for channel_index in range(EEG_DISPLAY_CHANNEL_COUNT)]

    @staticmethod
    def _empty_channel_stats() -> dict[str, float | str | None]:
        return {
            "center": 0.0,
            "low": -25.0,
            "high": 25.0,
            "p2p": 0.0,
            "scale": 25.0,
            "impedance": None,
        }

    def _reset_sample_store(self, max_points: int) -> None:
        self.max_points = max(64, int(max_points))
        self._sample_data = np.zeros((EEG_DISPLAY_CHANNEL_COUNT, self.max_points), dtype=np.float32)
        self._write_index = 0
        self._sample_count = 0
        self._plot_cache = [np.empty(0, dtype=np.float32) for _ in range(EEG_DISPLAY_CHANNEL_COUNT)]
        self._stats_cache = [self._empty_channel_stats() for _ in range(EEG_DISPLAY_CHANNEL_COUNT)]
        self._cache_dirty = True
        self._cached_plot_columns = 0

    def _append_samples(self, chunk: np.ndarray) -> None:
        if chunk.ndim != 2 or chunk.shape[1] <= 0:
            return
        rows = min(EEG_DISPLAY_CHANNEL_COUNT, int(chunk.shape[0]))
        if rows <= 0:
            return
        data = np.asarray(chunk[:rows], dtype=np.float32)
        sample_count = int(data.shape[1])
        if sample_count >= self.max_points:
            self._sample_data[:rows, :] = data[:, -self.max_points :]
            self._write_index = 0
            self._sample_count = self.max_points
            self._cache_dirty = True
            return

        first = min(sample_count, self.max_points - self._write_index)
        self._sample_data[:rows, self._write_index : self._write_index + first] = data[:, :first]
        remaining = sample_count - first
        if remaining > 0:
            self._sample_data[:rows, :remaining] = data[:, first:]
        self._write_index = (self._write_index + sample_count) % self.max_points
        self._sample_count = min(self.max_points, self._sample_count + sample_count)
        self._cache_dirty = True

    def _channel_window(self, channel_index: int) -> np.ndarray:
        if self._sample_count <= 0:
            return np.empty(0, dtype=np.float32)
        channel = max(0, min(EEG_DISPLAY_CHANNEL_COUNT - 1, int(channel_index)))
        count = min(self._sample_count, self.max_points)
        start = (self._write_index - count) % self.max_points
        if start + count <= self.max_points:
            return self._sample_data[channel, start : start + count].copy()
        return np.concatenate((self._sample_data[channel, start:], self._sample_data[channel, : (start + count) % self.max_points])).astype(
            np.float32,
            copy=False,
        )

    def set_mode(self, mode: str) -> None:
        normalized = str(mode or "EEG").strip().upper()
        self.mode = "IMP" if normalized.startswith("IMP") else "EEG"
        self._cache_dirty = True
        self._dirty = True
        self.update()

    def configure_stream(self, *, sampling_rate: float, channel_names: Sequence[str]) -> None:
        fs = max(1.0, float(sampling_rate))
        names = [str(name) for name in channel_names if str(name).strip()]
        while len(names) < EEG_DISPLAY_CHANNEL_COUNT:
            names.append(f"Ch {len(names) + 1}")
        self.sampling_rate = fs
        self.channel_names = names[:EEG_DISPLAY_CHANNEL_COUNT]
        self._reset_sample_store(max(64, int(round(float(self.window_seconds) * fs))))
        self.live_stream_active = True
        self.last_sample_perf = 0.0
        self.status_text = f"BrainFlow 原始 EEG | {fs:g} Hz | {len(channel_names)} 通道 | {self.window_seconds:g}s窗口"
        self._dirty = True
        self.update()

    def append_chunk(self, payload: object) -> None:
        try:
            chunk = np.asarray(payload, dtype=np.float32)
        except Exception:
            return
        if chunk.ndim != 2 or chunk.shape[1] <= 0:
            return
        if self.max_points <= 0:
            self.configure_stream(sampling_rate=max(1.0, self.sampling_rate or 250.0), channel_names=EEG_DEFAULT_CHANNEL_NAMES)
        self._append_samples(chunk)
        self.live_stream_active = True
        self.last_sample_perf = time.perf_counter()
        self._dirty = True

    def clear_stream(self, message: str | None = None) -> None:
        self.sampling_rate = 0.0
        self.channel_names = list(EEG_DEFAULT_CHANNEL_NAMES)
        self._reset_sample_store(300)
        self.live_stream_active = False
        self.last_sample_perf = 0.0
        self.status_text = str(message or "未连接真实 EEG 数据流，当前为占位波形。")
        self._seed_placeholder()
        self._dirty = True
        self.update()

    def set_status_text(self, message: str) -> None:
        text = str(message or "").strip()
        if text and text != self.status_text:
            self.status_text = text
            self._dirty = True
            self.update()

    def _seed_placeholder(self) -> None:
        sample_axis = np.arange(180, dtype=np.float32)[None, :]
        channel_axis = np.arange(EEG_DISPLAY_CHANNEL_COUNT, dtype=np.float32)[:, None]
        noise = np.random.default_rng(42).uniform(-1.8, 1.8, size=(EEG_DISPLAY_CHANNEL_COUNT, 180)).astype(np.float32)
        values = np.sin(sample_axis * 0.06 + channel_axis * 0.42).astype(np.float32) * 12.0 + noise
        self._append_samples(values)

    def _tick(self) -> None:
        now_perf = time.perf_counter()
        if not self.live_stream_active:
            now = time.monotonic()
            values = np.zeros((EEG_DISPLAY_CHANNEL_COUNT, 1), dtype=np.float32)
            for channel_index in range(EEG_DISPLAY_CHANNEL_COUNT):
                values[channel_index, 0] = math.sin(now * 4.0 + channel_index * 0.5) * 12.0 + random.uniform(-1.8, 1.8)
            self._append_samples(values)
            self._dirty = True
        live_age_refresh_due = self.live_stream_active and now_perf - self._last_live_repaint_perf >= 0.30
        if self._dirty or live_age_refresh_due:
            self._dirty = False
            self._last_live_repaint_perf = now_perf
            self.update()

    @staticmethod
    def _robust_display_bounds(values: np.ndarray) -> tuple[float, float, float, float]:
        if values.size <= 1:
            return 0.0, -25.0, 25.0, 25.0
        center = float(np.median(values))
        if values.size >= 20:
            low = float(np.percentile(values, 5))
            high = float(np.percentile(values, 95))
        else:
            low = float(np.min(values))
            high = float(np.max(values))
        scale = max(abs(high - center), abs(center - low), 10.0)
        if scale < 25.0:
            scale = 25.0
        return center, center - scale * 1.2, center + scale * 1.2, scale

    @staticmethod
    def _format_impedance_estimate(values: np.ndarray) -> str:
        if values.size < 20:
            return "Imp --"
        std_uvolts = float(np.std(values, ddof=0))
        z_ohm = max(0.0, (np.sqrt(2.0) * std_uvolts * 1e-6) / 6e-9 - 2200.0)
        if z_ohm >= 1_000_000.0:
            return f"Imp~{z_ohm / 1_000_000.0:.2f}M"
        if z_ohm >= 1_000.0:
            return f"Imp~{z_ohm / 1_000.0:.1f}k"
        return f"Imp~{z_ohm:.0f}"

    @staticmethod
    def _downsample_for_plot(values: np.ndarray, max_columns: int) -> np.ndarray:
        if values.size <= 2 or max_columns <= 2 or values.size <= max_columns:
            return np.asarray(values, dtype=np.float32)
        indices = np.linspace(0, int(values.size) - 1, int(max_columns), dtype=np.int32)
        return np.asarray(values[indices], dtype=np.float32)

    def _refresh_plot_cache(self, plot_columns: int) -> None:
        now = time.perf_counter()
        columns = max(16, int(plot_columns))
        if (
            not self._cache_dirty
            and self._cached_plot_columns == columns
            and now - self._last_cache_perf < 0.15
        ):
            return
        self._cached_plot_columns = columns
        self._last_cache_perf = now
        self._cache_dirty = False
        max_plot_points = min(160, max(48, int(columns * 0.55)))

        for channel_index in range(EEG_DISPLAY_CHANNEL_COUNT):
            values = self._channel_window(channel_index)
            self._plot_cache[channel_index] = self._downsample_for_plot(values, max_plot_points)
            if values.size < 2:
                self._stats_cache[channel_index] = self._empty_channel_stats()
                continue
            values64 = values.astype(np.float64, copy=False)
            center, low, high, scale = self._robust_display_bounds(values64)
            p2p = float(np.ptp(values))
            stats = {
                "center": float(center),
                "low": float(low),
                "high": float(high),
                "p2p": p2p,
                "scale": float(scale),
                "impedance": None,
            }
            if self.mode == "IMP":
                stats["impedance"] = self._format_impedance_estimate(values64)
            self._stats_cache[channel_index] = stats

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        outer = QRectF(self.rect()).adjusted(8, 8, -8, -8)
        header_h = 28.0
        header = QRectF(outer.left(), outer.top(), outer.width(), header_h)
        painter.setPen(QColor("#0369a1" if self.live_stream_active else "#b45309"))
        painter.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        status = self.status_text
        if self.live_stream_active and self.last_sample_perf > 0:
            status = f"{status} | 更新 {max(0.0, time.perf_counter() - self.last_sample_perf):.2f}s"
        status = painter.fontMetrics().elidedText(status, Qt.ElideRight, max(80, int(header.width())))
        painter.drawText(header, Qt.AlignLeft | Qt.AlignVCenter, status)

        content = QRectF(outer.left(), header.bottom() + 4, outer.width(), max(20.0, outer.height() - header_h - 4))
        row_h = max(1.0, content.height() / EEG_DISPLAY_CHANNEL_COUNT)
        label_w = 48.0
        info_w = 90.0
        plot_columns = max(16, int(content.width() - label_w - info_w))
        self._refresh_plot_cache(plot_columns)
        for channel_index in range(EEG_DISPLAY_CHANNEL_COUNT):
            row_top = content.top() + channel_index * row_h
            row_rect = QRectF(content.left(), row_top, content.width(), row_h)
            mid_y = row_rect.center().y()
            painter.setPen(QPen(QColor("#e2e8f0"), 1))
            painter.drawLine(int(row_rect.left()), int(mid_y), int(row_rect.right()), int(mid_y))
            stats = self._stats_cache[channel_index]
            painter.setPen(QColor("#0369a1"))
            painter.setFont(QFont("Consolas", 9, QFont.Bold))
            label = self.channel_names[channel_index] if channel_index < len(self.channel_names) else f"Ch {channel_index + 1}"
            painter.drawText(QRectF(row_rect.left(), row_rect.top(), label_w, row_rect.height()), Qt.AlignLeft | Qt.AlignVCenter, label)

            plot_rect = QRectF(
                row_rect.left() + label_w,
                row_rect.top() + 3.0,
                max(10.0, row_rect.width() - label_w - info_w),
                max(8.0, row_rect.height() - 6.0),
            )
            values = self._plot_cache[channel_index]
            if values.size < 2:
                painter.setPen(QColor("#64748b"))
                painter.drawText(plot_rect, Qt.AlignCenter, "等待数据")
                continue

            center = float(stats.get("center") or 0.0)
            low = float(stats.get("low") or -25.0)
            high = float(stats.get("high") or 25.0)
            span = max(1.0, high - low)
            step_x = plot_rect.width() / max(1, int(values.size) - 1)
            path = QPainterPath()
            for idx, value in enumerate(values):
                x = plot_rect.left() + idx * step_x
                y_ratio = (float(value) - center) / (span / 2.0)
                y = plot_rect.center().y() - y_ratio * (plot_rect.height() / 2.0)
                y = max(plot_rect.top(), min(plot_rect.bottom(), y))
                if idx == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            painter.setPen(QPen(QColor("#0284c7"), 1.35))
            painter.drawPath(path)
            painter.setPen(QColor("#475569"))
            painter.setFont(QFont("Consolas", 8))
            if self.mode == "IMP":
                info = str(stats.get("impedance") or "Imp --")
            else:
                info = f"±{float(stats.get('scale') or 0.0):.0f}uV"
            painter.drawText(
                QRectF(plot_rect.right() + 6.0, row_rect.top(), info_w - 6.0, row_rect.height()),
                Qt.AlignLeft | Qt.AlignVCenter,
                info,
            )


class ConnectionGateWidget(QWidget):
    ready_changed = pyqtSignal(bool)
    proceed_requested = pyqtSignal()
    robot_control_requested = pyqtSignal()
    eeg_stream_requested = pyqtSignal(str, int)
    eeg_stream_stop_requested = pyqtSignal()

    def __init__(self, backend: RobotCommandBackend, config: WorkbenchConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.backend = backend
        self.config = config
        self.robot_connected = False
        self.eeg_connected = False
        self.demo_connected = False
        self._last_ready = False
        self._robot_connecting = False
        self.robot_control_unlocked = False
        self._detail_key_labels: list[QLabel] = []
        self._detail_value_labels: list[QLabel] = []
        self._secondary_labels: list[QLabel] = []
        self._build_ui()
        self.backend.connection_changed.connect(self._on_robot_connection_changed)
        self.backend.connect_progress_changed.connect(self._on_robot_connect_progress)
        self.backend.status_changed.connect(self._on_robot_status)

    def _build_ui(self) -> None:
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(12)

        header = QLabel("设备连接")
        self.lbl_connection_title = header
        header.setStyleSheet("color: #0f172a; font-size: 32px; font-weight: 900;")
        sub = QLabel("先完成机械臂和脑电帽的就绪检查。系统将按完整脑控流程进入采集、训练和机械臂控制。")
        self.lbl_connection_subtitle = sub
        sub.setWordWrap(True)
        sub.setStyleSheet("color: #475569; font-size: 17px;")
        layout.addWidget(header)
        layout.addWidget(sub)

        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        self.lbl_robot_chip = QLabel("机械臂 未连接")
        self.lbl_robot_chip.setObjectName("StatusChip")
        self.lbl_robot_chip.setProperty("state", "danger")
        self.lbl_eeg_chip = QLabel("脑电帽 未连接")
        self.lbl_eeg_chip.setObjectName("StatusChip")
        self.lbl_eeg_chip.setProperty("state", "danger")
        status_row.addWidget(self.lbl_robot_chip)
        status_row.addWidget(self.lbl_eeg_chip)
        status_row.addStretch(1)
        layout.addLayout(status_row)

        def add_detail_rows(target_layout: QVBoxLayout, rows: Sequence[tuple[str, str]]) -> None:
            grid = QGridLayout()
            grid.setHorizontalSpacing(16)
            grid.setVerticalSpacing(10)
            for row_index, (label_text, value_text) in enumerate(rows):
                key = QLabel(label_text)
                key.setStyleSheet("color: #64748b; font-size: 16px; font-weight: 800;")
                self._detail_key_labels.append(key)
                value = QLabel(value_text)
                value.setWordWrap(True)
                value.setStyleSheet("color: #0f172a; font-size: 16px;")
                self._detail_value_labels.append(value)
                grid.addWidget(key, row_index, 0, Qt.AlignTop)
                grid.addWidget(value, row_index, 1)
            grid.setColumnStretch(0, 0)
            grid.setColumnStretch(1, 1)
            target_layout.addLayout(grid)

        cards = QHBoxLayout()
        cards.setSpacing(16)

        robot_card = QFrame()
        self.robot_card = robot_card
        robot_card.setObjectName("DeviceCard")
        robot_card.setMinimumHeight(360)
        robot_card.setMaximumHeight(16777215)
        robot_card.setMinimumWidth(360)
        robot_card.setMaximumWidth(16777215)
        robot_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        robot_layout = QVBoxLayout(robot_card)
        robot_layout.setContentsMargins(18, 16, 18, 16)
        robot_layout.setSpacing(12)
        self.lbl_robot_state = QLabel("未连接")
        self.lbl_robot_state.setObjectName("StatusChip")
        self.lbl_robot_state.setProperty("state", "danger")
        robot_head = QHBoxLayout()
        robot_title = QLabel("机械臂控制链路")
        robot_title.setObjectName("DeviceTitle")
        robot_head.addWidget(robot_title, 1)
        robot_head.addWidget(self.lbl_robot_state)
        self.lbl_robot_detail = QLabel(
            f"{self.config.robot_transport.upper()}  {self.config.robot_host}:"
            f"{self.config.rosbridge_port if self.config.robot_transport == 'ros' else self.config.robot_port}"
        )
        self.lbl_robot_detail.setWordWrap(True)
        self.lbl_robot_detail.setStyleSheet("color: #475569; font-size: 16px;")
        self._secondary_labels.append(self.lbl_robot_detail)
        self.lbl_robot_phase = QLabel("等待连接")
        self.lbl_robot_phase.setStyleSheet("color: #0369a1; font-size: 17px; font-weight: 900;")
        self.robot_progress = QProgressBar()
        self.robot_progress.setRange(0, 100)
        self.robot_progress.setValue(0)
        self.robot_progress.setTextVisible(False)
        self.robot_progress.setFixedHeight(14)
        self.btn_robot_connect = QPushButton("连接机械臂")
        self.btn_robot_connect.setObjectName("AccentButton")
        self.btn_robot_disconnect = QPushButton("断开机械臂")
        self.btn_robot_disconnect.setObjectName("ActionButton")
        self.btn_robot_disconnect.setEnabled(False)
        robot_buttons = QHBoxLayout()
        robot_buttons.setSpacing(10)
        robot_buttons.addWidget(self.btn_robot_connect, 1)
        robot_buttons.addWidget(self.btn_robot_disconnect, 1)
        robot_layout.addLayout(robot_head)
        robot_layout.addWidget(self.lbl_robot_detail)
        robot_layout.addWidget(self.lbl_robot_phase)
        robot_layout.addWidget(self.robot_progress)
        robot_hint = QLabel("连接时会检查 rosbridge 与状态话题；若运行时缺失，会尝试拉起机械臂端程序。")
        robot_hint.setWordWrap(True)
        robot_hint.setStyleSheet("color: #64748b; font-size: 16px;")
        self._secondary_labels.append(robot_hint)
        robot_layout.addWidget(robot_hint)
        add_detail_rows(
            robot_layout,
            (
                ("网络", f"机械臂 WiFi 后默认连接 {self.config.robot_host}"),
                ("控制", f"rosbridge {int(self.config.rosbridge_port)} / teleop 指令通道"),
                ("状态", "收到 /hybrid_controller/state 后才允许进入后续控制"),
            ),
        )
        robot_layout.addStretch(1)
        robot_layout.addLayout(robot_buttons)
        self.btn_robot_connect.clicked.connect(self.backend.connect_robot)
        self.btn_robot_disconnect.clicked.connect(self.backend.close_backend)
        cards.addWidget(robot_card)

        eeg_card = QFrame()
        self.eeg_card = eeg_card
        eeg_card.setObjectName("DeviceCard")
        eeg_card.setMinimumHeight(360)
        eeg_card.setMaximumHeight(16777215)
        eeg_card.setMinimumWidth(360)
        eeg_card.setMaximumWidth(16777215)
        eeg_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        eeg_layout = QVBoxLayout(eeg_card)
        eeg_layout.setContentsMargins(18, 16, 18, 16)
        eeg_layout.setSpacing(12)
        self.lbl_eeg_state = QLabel("未连接")
        self.lbl_eeg_state.setObjectName("StatusChip")
        self.lbl_eeg_state.setProperty("state", "danger")
        eeg_head = QHBoxLayout()
        eeg_title = QLabel("脑电帽就绪门禁")
        eeg_title.setObjectName("DeviceTitle")
        eeg_head.addWidget(eeg_title, 1)
        eeg_head.addWidget(self.lbl_eeg_state)
        self.eeg_serial_edit = QLineEdit(str(self.config.eeg_serial_port or "auto"))
        self.eeg_serial_edit.setPlaceholderText("auto 或 COMx")
        self.eeg_board_edit = QLineEdit(str(int(self.config.eeg_board_id)))
        self.eeg_board_edit.setPlaceholderText("BrainFlow board id")
        self.lbl_eeg_detail = QLabel("这里作为脑电帽就绪门禁，连接后显示实时 8 通道 EEG 波形，并进入完整 MI/SSVEP 流程。")
        self.lbl_eeg_detail.setWordWrap(True)
        self.lbl_eeg_detail.setStyleSheet("color: #475569; font-size: 16px;")
        self._secondary_labels.append(self.lbl_eeg_detail)
        self.btn_eeg_connect = QPushButton("连接脑电帽")
        self.btn_eeg_connect.setObjectName("AccentButton")
        self.btn_eeg_disconnect = QPushButton("断开脑电帽")
        self.btn_eeg_disconnect.setObjectName("ActionButton")
        eeg_buttons = QHBoxLayout()
        eeg_buttons.setSpacing(10)
        eeg_buttons.addWidget(self.btn_eeg_connect, 1)
        eeg_buttons.addWidget(self.btn_eeg_disconnect, 1)
        eeg_layout.addLayout(eeg_head)
        eeg_layout.addWidget(QLabel("串口"))
        eeg_layout.addWidget(self.eeg_serial_edit)
        eeg_layout.addSpacing(8)
        eeg_layout.addWidget(QLabel("板卡 ID"))
        eeg_layout.addWidget(self.eeg_board_edit)
        eeg_layout.addSpacing(4)
        eeg_layout.addWidget(self.lbl_eeg_detail)
        add_detail_rows(
            eeg_layout,
            (
                ("显示", "连接后实时绘制 8 通道 EEG 波形"),
                ("识别", "脑控流程入口已就绪，保留完整 MI/SSVEP 控制链路"),
            ),
        )
        eeg_layout.addStretch(1)
        eeg_layout.addLayout(eeg_buttons)
        self.btn_eeg_connect.clicked.connect(self.connect_eeg_cap)
        self.btn_eeg_disconnect.clicked.connect(self.disconnect_eeg_cap)
        cards.addWidget(eeg_card)
        layout.addLayout(cards, 1)

        summary = QFrame()
        summary.setObjectName("ConnectionSummary")
        summary.setMaximumWidth(16777215)
        summary.setMinimumHeight(126)
        summary.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        summary_layout = QGridLayout(summary)
        summary_layout.setContentsMargins(14, 12, 14, 12)
        summary_layout.setHorizontalSpacing(12)
        summary_layout.setVerticalSpacing(8)
        summary_items = (
            ("1 连接门禁", "先确认机械臂与脑电帽，避免进入控制页后再排查设备。"),
            ("2 实时信号", "脑电帽连接后显示真实 8 通道波形；流程按 MI/SSVEP 阶段推进。"),
            ("3 机械臂控制", "进入机械臂页后以大摄像头画面为主，MI 控制移动，SSVEP 确认目标。"),
        )
        for column, (item_title, item_body) in enumerate(summary_items):
            tile = QFrame()
            tile.setObjectName("InfoTile")
            tile_layout = QVBoxLayout(tile)
            tile_layout.setContentsMargins(12, 10, 12, 10)
            title_label = QLabel(item_title)
            title_label.setObjectName("MetricValue")
            body_label = QLabel(item_body)
            body_label.setWordWrap(True)
            body_label.setStyleSheet("color: #475569; font-size: 16px;")
            self._secondary_labels.append(body_label)
            tile_layout.addWidget(title_label)
            tile_layout.addWidget(body_label, 1)
            summary_layout.addWidget(tile, 0, column)
        summary_row = QHBoxLayout()
        summary_row.addWidget(summary)
        layout.addLayout(summary_row)

        bottom = QFrame()
        bottom.setObjectName("BottomBar")
        bottom.setMaximumHeight(90)
        bottom.setMaximumWidth(16777215)
        bottom.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(16, 14, 16, 14)
        self.lbl_gate_status = QLabel("等待设备连接")
        self.lbl_gate_status.setStyleSheet("color: #475569; font-size: 16px; font-weight: 800;")
        self._secondary_labels.append(self.lbl_gate_status)
        self.btn_next = QPushButton("进入 SSVEP")
        self.btn_next.setObjectName("AccentButton")
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self.proceed_requested.emit)
        self.btn_robot_control = QPushButton("训练完成后进入机械臂")
        self.btn_robot_control.setObjectName("ActionButton")
        self.btn_robot_control.setEnabled(False)
        self.btn_robot_control.clicked.connect(self.robot_control_requested.emit)
        bottom_layout.addWidget(self.lbl_gate_status, 1)
        bottom_layout.addWidget(self.btn_robot_control)
        bottom_layout.addWidget(self.btn_next)
        bottom_row = QHBoxLayout()
        bottom_row.addWidget(bottom)
        layout.addLayout(bottom_row)
        self.apply_visual_scale(1.0)

    def apply_visual_scale(self, scale: float) -> None:
        value = max(1.0, min(1.35, float(scale)))
        self._visual_scale = value
        title_px = int(round(32 * value))
        subtitle_px = int(round(17 * value))
        body_px = int(round(16 * value))
        phase_px = int(round(17 * value))
        progress_h = int(round(14 * value))
        if value >= 1.25:
            card_min_h = 455
        elif value >= 1.1:
            card_min_h = 420
        else:
            card_min_h = 360
        self.lbl_connection_title.setStyleSheet(f"color: #0f172a; font-size: {title_px}px; font-weight: 900;")
        self.lbl_connection_subtitle.setStyleSheet(f"color: #475569; font-size: {subtitle_px}px;")
        self.lbl_robot_phase.setStyleSheet(f"color: #0369a1; font-size: {phase_px}px; font-weight: 900;")
        self.robot_progress.setFixedHeight(progress_h)
        self.robot_card.setMinimumHeight(card_min_h)
        self.eeg_card.setMinimumHeight(card_min_h)
        for label in self._detail_key_labels:
            label.setStyleSheet(f"color: #64748b; font-size: {body_px}px; font-weight: 800;")
        for label in self._detail_value_labels:
            label.setStyleSheet(f"color: #0f172a; font-size: {body_px}px;")
        for label in self._secondary_labels:
            label.setStyleSheet(f"color: #475569; font-size: {body_px}px;")
        self._refresh_ready_state()

    def _gate_status_style(self, color: str, weight: int) -> str:
        font_px = int(round(16 * getattr(self, "_visual_scale", 1.0)))
        return f"color: {color}; font-size: {font_px}px; font-weight: {int(weight)};"

    def _set_status_chip(self, label: QLabel, text: str, state: str) -> None:
        label.setText(text)
        label.setProperty("state", state)
        label.style().unpolish(label)
        label.style().polish(label)

    def connect_eeg_cap(self) -> None:
        serial_port = self.eeg_serial_edit.text().strip() or "auto"
        board_id_text = self.eeg_board_edit.text().strip() or "0"
        board_id = _coerce_eeg_board_id(board_id_text)
        if board_id is None:
            self._set_status_chip(self.lbl_eeg_state, "板卡 ID 无效", "danger")
            self.lbl_eeg_detail.setText(f"BrainFlow board_id 必须是整数，当前输入: {board_id_text!r}")
            return
        self.config.eeg_serial_port = serial_port
        self.config.eeg_board_id = int(board_id)
        self.eeg_connected = True
        self._set_status_chip(self.lbl_eeg_state, "已就绪", "ready")
        self._set_status_chip(self.lbl_eeg_chip, "脑电帽 已就绪", "ready")
        self.lbl_eeg_detail.setText(
            f"脑电帽就绪: serial={serial_port}, board_id={board_id}。正在准备实时 8 通道 BrainFlow 波形。"
        )
        self.eeg_stream_requested.emit(serial_port, int(board_id))
        self._refresh_ready_state()

    def disconnect_eeg_cap(self) -> None:
        self.eeg_connected = False
        self.eeg_stream_stop_requested.emit()
        self._set_status_chip(self.lbl_eeg_state, "未连接", "danger")
        self._set_status_chip(self.lbl_eeg_chip, "脑电帽 未连接", "danger")
        self.lbl_eeg_detail.setText("脑电信号预览已停止，重新连接后可恢复实时 8 通道波形。")
        self._refresh_ready_state()

    def _on_robot_connection_changed(self, connected: bool) -> None:
        self.robot_connected = bool(connected)
        self._set_status_chip(self.lbl_robot_state, "已连接" if connected else "未连接", "ready" if connected else "danger")
        self._set_status_chip(
            self.lbl_robot_chip,
            "机械臂 已连接" if connected else "机械臂 未连接",
            "ready" if connected else "danger",
        )
        if connected:
            self._robot_connecting = False
            self.robot_progress.setValue(100)
            self.lbl_robot_phase.setText("连接完成 100%")
        elif not self._robot_connecting:
            self.robot_progress.setValue(0)
            self.lbl_robot_phase.setText("等待连接")
        self.btn_robot_connect.setEnabled((not connected) and (not self._robot_connecting))
        self.btn_robot_disconnect.setEnabled(bool(connected) or self._robot_connecting)
        self._refresh_ready_state()

    def _on_robot_connect_progress(self, percent: int, phase: str, detail: str) -> None:
        value = max(0, min(100, int(percent)))
        phase_text = str(phase or "连接中")
        detail_text = str(detail or "")
        self._robot_connecting = 0 < value < 100
        self.robot_progress.setValue(value)
        self.lbl_robot_phase.setText(phase_text if value <= 0 else f"{phase_text} {value}%")
        if detail_text:
            self.lbl_robot_detail.setText(detail_text)
        if value == 0 and ("失败" in phase_text or "失败" in detail_text):
            self._set_status_chip(self.lbl_robot_state, "连接失败", "danger")
            self._set_status_chip(self.lbl_robot_chip, "机械臂 连接失败", "danger")
        self.btn_robot_connect.setEnabled((not self.robot_connected) and (not self._robot_connecting))
        self.btn_robot_disconnect.setEnabled(self.robot_connected or self._robot_connecting)

    def _on_robot_status(self, message: str) -> None:
        text = str(message)
        if text:
            lower = text.lower()
            if self._robot_connecting and any(token in lower for token in ("error", "failed", "timeout")):
                return
            self.lbl_robot_detail.setText(text)

    def enable_demo_connected(self) -> None:
        self.demo_connected = True
        self.robot_connected = True
        self.eeg_connected = True
        self._robot_connecting = False
        self.robot_progress.setValue(100)
        self.lbl_robot_phase.setText("演示连接 100%")
        self.lbl_robot_detail.setText("演示模式：未连接真实机械臂，仅用于查看后续训练和控制界面。")
        self.lbl_eeg_detail.setText("演示模式：未连接真实脑电帽，训练流程由界面完整展示。")
        self._set_status_chip(self.lbl_robot_state, "演示已连接", "ready")
        self._set_status_chip(self.lbl_robot_chip, "机械臂 演示已连接", "ready")
        self._set_status_chip(self.lbl_eeg_state, "演示已连接", "ready")
        self._set_status_chip(self.lbl_eeg_chip, "脑电帽 演示已连接", "ready")
        self.btn_robot_connect.setEnabled(False)
        self.btn_robot_disconnect.setEnabled(False)
        self.btn_eeg_connect.setEnabled(False)
        self.btn_eeg_disconnect.setEnabled(False)
        self._refresh_ready_state()

    def _refresh_ready_state(self) -> None:
        ready = self.robot_connected and self.eeg_connected
        self.btn_next.setEnabled(ready)
        self.btn_robot_control.setEnabled(ready and self.robot_control_unlocked)
        self.btn_robot_control.setText("进入机械臂控制" if self.robot_control_unlocked else "训练完成后进入机械臂")
        if ready and self.demo_connected:
            self.lbl_gate_status.setText("演示模式：已假装机械臂和脑电帽连接完成，可直接查看后续 UI。")
            self.lbl_gate_status.setStyleSheet(self._gate_status_style("#047857", 900))
        elif ready:
            self.lbl_gate_status.setText("机械臂和脑电帽均已连接，可以进入下一步。")
            self.lbl_gate_status.setStyleSheet(self._gate_status_style("#047857", 900))
        elif self.robot_connected:
            self.lbl_gate_status.setText("机械臂已连接，等待脑电帽连接。")
            self.lbl_gate_status.setStyleSheet(self._gate_status_style("#0369a1", 800))
        elif self.eeg_connected:
            self.lbl_gate_status.setText("脑电帽已连接，等待机械臂连接。")
            self.lbl_gate_status.setStyleSheet(self._gate_status_style("#0369a1", 800))
        else:
            self.lbl_gate_status.setText("等待设备连接")
            self.lbl_gate_status.setStyleSheet(self._gate_status_style("#0369a1", 800))
        if ready != self._last_ready:
            self._last_ready = ready
            self.ready_changed.emit(ready)

    def set_robot_control_unlocked(self, unlocked: bool) -> None:
        self.robot_control_unlocked = bool(unlocked)
        self._refresh_ready_state()


class MjpegCameraThread(QThread):
    frame_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    def __init__(self, url: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.url = str(url)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        last_emit_ts = 0.0
        while not self._stop_event.is_set():
            try:
                request = Request(self.url, headers={"User-Agent": "brain-robot-workbench/camera"})
                self.status_changed.emit(f"opening camera stream: {self.url}")
                with urlopen(request, timeout=CAMERA_STREAM_TIMEOUT_SEC) as response:
                    self.status_changed.emit("camera stream connected")
                    buffer = bytearray()
                    while not self._stop_event.is_set():
                        try:
                            chunk = response.read(4096)
                        except (TimeoutError, socket.timeout):
                            self.status_changed.emit("camera read timeout, reconnecting")
                            break
                        if not chunk:
                            self.status_changed.emit("camera stream ended, reconnecting")
                            break
                        buffer.extend(chunk)
                        while True:
                            start = buffer.find(b"\xff\xd8")
                            if start < 0:
                                if len(buffer) > 1_000_000:
                                    del buffer[:-128]
                                break
                            end = buffer.find(b"\xff\xd9", start + 2)
                            if end < 0:
                                if start > 0:
                                    del buffer[:start]
                                if len(buffer) > 2_000_000:
                                    del buffer[:-512_000]
                                break
                            jpg = bytes(buffer[start : end + 2])
                            del buffer[: end + 2]
                            now = time.monotonic()
                            if now - last_emit_ts < 1.0 / 18.0:
                                continue
                            image = QImage.fromData(jpg, "JPG")
                            if image.isNull():
                                continue
                            last_emit_ts = now
                            self.frame_ready.emit(image.copy())
            except Exception as error:
                if not self._stop_event.is_set():
                    self.status_changed.emit(f"camera stream error: {error}")
            if not self._stop_event.wait(1.0):
                continue


class RobotCameraWidget(QWidget):
    status_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.active_id = ""
        self.phase = 0
        self.stream_url = ""
        self.camera_status = "camera idle"
        self.latest_frame: QImage | None = None
        self.target_ids = _target_ids()
        self.stream_thread: MjpegCameraThread | None = None
        self.timer = QTimer(self)
        self.timer.setInterval(80)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def start_stream(self, url: str) -> None:
        stream_url = str(url).strip()
        if not stream_url:
            self._set_camera_status("camera stream url is empty")
            return
        if self.stream_thread is not None and self.stream_thread.isRunning() and self.stream_url == stream_url:
            return
        self.stop_stream()
        self.stream_url = stream_url
        self.latest_frame = None
        self.stream_thread = MjpegCameraThread(stream_url, self)
        self.stream_thread.frame_ready.connect(self._on_frame_ready)
        self.stream_thread.status_changed.connect(self._set_camera_status)
        self.stream_thread.finished.connect(self.stream_thread.deleteLater)
        self.stream_thread.start()
        self._set_camera_status("camera stream starting")

    def stop_stream(self) -> None:
        thread = self.stream_thread
        self.stream_thread = None
        if thread is not None:
            thread.stop()
            thread.wait(int(CAMERA_STREAM_TIMEOUT_SEC * 1000) + 300)
        self._set_camera_status("camera idle")

    def _set_camera_status(self, status: str) -> None:
        self.camera_status = str(status)
        self.status_changed.emit(self.camera_status)
        self.update()

    def _on_frame_ready(self, image: object) -> None:
        self.set_frame(image)

    def set_frame(self, image: object) -> None:
        if isinstance(image, QImage) and not image.isNull():
            self.latest_frame = image.copy()
            self.update()

    def _tick(self) -> None:
        self.phase += 1
        if self.active_id or self.latest_frame is None:
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#0f172a"))
        w, h = max(1, self.width()), max(1, self.height())
        center = QPointF(w / 2.0, h / 2.0)
        if self.latest_frame is not None and not self.latest_frame.isNull():
            pixmap = QPixmap.fromImage(self.latest_frame)
            scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = int((w - scaled.width()) / 2)
            y = int((h - scaled.height()) / 2)
            painter.drawPixmap(x, y, scaled)
            painter.setPen(QPen(QColor(255, 255, 255, 180), 1))
            painter.drawLine(20, int(center.y()), w - 20, int(center.y()))
            painter.drawLine(int(center.x()), 20, int(center.x()), h - 20)
            painter.setBrush(QColor("#f43f5e"))
            painter.setPen(QPen(QColor("#38bdf8"), 2))
            painter.drawEllipse(center, 11, 11)
            if self.active_id:
                painter.setBrush(QColor(2, 132, 199, 210))
                painter.setPen(QPen(QColor("#ffffff"), 2))
                tag_rect = QRectF(18, 18, 180, 44)
                painter.drawRoundedRect(tag_rect, 6, 6)
                painter.setPen(QColor("#ffffff"))
                painter.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
                painter.drawText(tag_rect, Qt.AlignCenter, f"目标 {self.active_id} 已锁定")
            painter.setPen(QColor("#dbeafe"))
            painter.setFont(QFont("Consolas", 9))
            painter.drawText(QRectF(12, h - 30, w - 24, 20), Qt.AlignLeft | Qt.AlignVCenter, self.camera_status)
            return
        painter.setPen(QPen(QColor("#334155"), 1))
        painter.drawLine(20, int(center.y()), w - 20, int(center.y()))
        painter.drawLine(int(center.x()), 20, int(center.x()), h - 20)
        ids = list(getattr(self, "target_ids", _target_ids()))
        for index, label in enumerate(ids):
            row, col = divmod(index, 2)
            x_ratio = 0.34 + col * 0.32
            y_ratio = 0.36 + row * 0.22
            color = _target_color(label)
            x, y = w * x_ratio, h * y_ratio
            active = label == self.active_id
            rect = QRectF(x - 32, y - 20, 64, 40)
            painter.setBrush(QColor(color))
            painter.setPen(QPen(QColor("#ffffff"), 2 if active else 1))
            painter.drawRoundedRect(rect, 6, 6)
            if active:
                pulse = 7 + (self.phase % 5) * 4
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor("#7dd3fc"), 2))
                painter.drawEllipse(QRectF(rect.left() - pulse, rect.top() - pulse, rect.width() + pulse * 2, rect.height() + pulse * 2))
                painter.setPen(QPen(QColor("#7dd3fc"), 1, Qt.DashLine))
                painter.drawLine(int(center.x()), int(center.y()), int(x), int(y))
            painter.setPen(QColor("#020617"))
            painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
            painter.drawText(rect, Qt.AlignCenter, label)
        painter.setPen(QPen(QColor("#38bdf8"), 2))
        painter.drawEllipse(center, 12, 12)


class RobotCameraDisplayWidget(RobotCameraWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.phase_title = "待机"
        self.phase_detail = "等待启动机械臂任务"
        self.countdown_text = ""
        self.pose_text = "theta=0.0 deg, radius=150.0 mm, z=160.0 mm"
        self.grip_text = "吸嘴: STANDBY / OPEN"
        self.keyboard_text = "MI: 连续运动意图    SSVEP: 目标选择 / 抓取确认 / 放置确认    安全停止"
        self.last_frame_ts = 0.0
        self.vision_targets: list[VisionTarget] = []
        self.vision_packet: dict[str, object] = {}
        self.vision_status_text = "视觉识别: 未启动"
        self.ssv_flicker_enabled = False
        self.ssv_selected_id = ""
        self.ssv_start_time = time.perf_counter()
        self._operation_state_key: tuple[str, str, str, str, str, str] | None = None
        self._vision_payload_key: tuple[object, ...] | None = None
        self._frame_pixmap: QPixmap | None = None
        self._scaled_frame_cache: tuple[int, int, int, int, QPixmap] | None = None
        self.ssv_frequencies = {
            target_id: frequency
            for target_id, frequency in zip(_target_ids(), (9.0, 11.0, 13.0, 15.0))
        }

    def set_operation_state(
        self,
        *,
        phase_title: str,
        phase_detail: str,
        countdown_text: str = "",
        active_id: str | None = None,
        pose_text: str | None = None,
        grip_text: str | None = None,
    ) -> None:
        self.phase_title = str(phase_title)
        self.phase_detail = str(phase_detail)
        self.countdown_text = str(countdown_text)
        if active_id is not None:
            self.active_id = str(active_id)
        if pose_text is not None:
            self.pose_text = str(pose_text)
        if grip_text is not None:
            self.grip_text = str(grip_text)
        next_key = (
            self.phase_title,
            self.phase_detail,
            self.countdown_text,
            self.active_id,
            self.pose_text,
            self.grip_text,
        )
        if next_key == self._operation_state_key:
            return
        self._operation_state_key = next_key
        self.update()

    def _on_frame_ready(self, image: object) -> None:
        self.set_frame(image)

    def set_frame(self, image: object) -> None:
        if isinstance(image, QImage) and not image.isNull():
            self.latest_frame = image.copy()
            self._frame_pixmap = QPixmap.fromImage(self.latest_frame)
            self._scaled_frame_cache = None
            self.last_frame_ts = time.monotonic()
            self.update()

    def set_vision_payload(
        self,
        *,
        targets: Sequence[VisionTarget],
        packet: dict[str, object] | None,
        status_text: str,
    ) -> None:
        next_targets = list(targets)
        next_status = str(status_text or "视觉识别: 未启动")
        next_packet = dict(packet or {})
        alignment_point = _coerce_xy(next_packet.get("alignment_target_pixel"))
        payload_key = (
            tuple(
                (
                    str(target.slot_id if target.slot_id is not None else target.id),
                    tuple(round(float(value), 2) for value in target.bbox),
                    tuple(round(float(value), 2) for value in (target.grasp_pixel or target.display_center or target.center_px)),
                    tuple(round(float(value), 2) for value in target.alignment_target_pixel)
                    if target.alignment_target_pixel is not None
                    else (),
                    bool(target.actionable),
                    bool(target.servo_required),
                    str(target.invalid_reason or ""),
                )
                for target in next_targets
            ),
            next_status,
            tuple(round(float(value), 2) for value in alignment_point) if alignment_point is not None else (),
        )
        if payload_key == self._vision_payload_key:
            return
        self._vision_payload_key = payload_key
        self.vision_targets = next_targets
        self.vision_packet = next_packet
        self.vision_status_text = next_status
        self.update()

    def set_ssvep_flicker(self, enabled: bool, *, selected_id: str = "") -> None:
        enabled = bool(enabled)
        next_selected = str(selected_id or "")
        if enabled == self.ssv_flicker_enabled and next_selected == self.ssv_selected_id:
            return
        if enabled and not self.ssv_flicker_enabled:
            self.ssv_start_time = time.perf_counter()
        self.ssv_flicker_enabled = enabled
        self.ssv_selected_id = next_selected
        if next_selected:
            self.active_id = next_selected
        elif enabled:
            self.active_id = ""
        self._operation_state_key = None
        self.update()

    def _ssvep_flash_on(self, target_id: str) -> bool:
        if not self.ssv_flicker_enabled:
            return False
        frequency = float(self.ssv_frequencies.get(str(target_id), 9.0))
        elapsed = max(0.0, time.perf_counter() - self.ssv_start_time)
        return math.sin(2.0 * math.pi * frequency * elapsed) >= 0.0

    def _tick(self) -> None:
        self.phase += 1
        if self.active_id or self.ssv_flicker_enabled or self.latest_frame is None or time.monotonic() - self.last_frame_ts > 2.0:
            self.update()

    def _camera_badge(self) -> tuple[str, QColor]:
        status = self.camera_status.lower()
        if self.latest_frame is not None and time.monotonic() - self.last_frame_ts <= 2.0:
            return "LIVE", QColor("#16a34a")
        if "connected" in status or "starting" in status or "opening" in status:
            return "连接中", QColor("#0284c7")
        if "error" in status or "timeout" in status or "failed" in status or "ended" in status:
            return "断流", QColor("#dc2626")
        if "disabled" in status:
            return "模拟", QColor("#64748b")
        return "待机", QColor("#64748b")

    def _draw_badge(self, painter: QPainter, rect: QRectF, text: str, color: QColor, text_color: QColor | None = None) -> None:
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, 5, 5)
        painter.setPen(text_color or QColor("#ffffff"))
        painter.setFont(QFont("Microsoft YaHei", max(10, min(18, int(rect.height() * 0.36))), QFont.Bold))
        painter.drawText(rect, Qt.AlignCenter, text)

    def _draw_crosshair(self, painter: QPainter, center: QPointF, w: int, h: int) -> None:
        painter.setPen(QPen(QColor(219, 234, 254, 145), 1))
        painter.drawLine(28, int(center.y()), w - 28, int(center.y()))
        painter.drawLine(int(center.x()), 28, int(center.x()), h - 28)
        painter.setBrush(QColor(244, 63, 94, 220))
        painter.setPen(QPen(QColor("#7dd3fc"), 2))
        painter.drawEllipse(center, 10, 10)

    def _draw_target_stack(self, painter: QPainter, w: int) -> None:
        ids = list(getattr(self, "target_ids", _target_ids()))
        scale = max(1.0, min(1.6, w / 1450.0))
        stack_w = int(round(40 * scale))
        stack_h = int(round(32 * scale))
        gap = int(round(6 * scale))
        cols = min(2, max(1, len(ids)))
        x = w - cols * stack_w - (cols - 1) * gap - int(round(22 * scale))
        y0 = int(round(96 * scale))
        for index, target_id in enumerate(ids):
            row, col = divmod(index, cols)
            y = y0 + row * (stack_h + gap)
            item_x = x + col * (stack_w + gap)
            active = target_id == self.active_id
            flicker = self.ssv_flicker_enabled and self._ssvep_flash_on(target_id)
            rect = QRectF(item_x, y, stack_w, stack_h)
            if self.ssv_flicker_enabled:
                painter.setBrush(QColor(_target_color(target_id)) if flicker else QColor(2, 6, 23, 230))
                painter.setPen(QPen(QColor("#ffffff" if flicker or active else "#38bdf8"), 2))
            else:
                painter.setBrush(QColor(_target_color(target_id)) if active else QColor(15, 23, 42, 185))
                painter.setPen(QPen(QColor("#ffffff" if active else "#94a3b8"), 2 if active else 1))
            painter.drawRoundedRect(rect, 6, 6)
            painter.setPen(QColor("#020617" if active or flicker else "#e2e8f0"))
            painter.setFont(QFont("Segoe UI", int(round(12 * scale)), QFont.Bold))
            painter.drawText(rect, Qt.AlignCenter, target_id)

    @staticmethod
    def _map_frame_point(point: tuple[float, float], frame_rect: QRectF, frame_w: int, frame_h: int) -> QPointF:
        x = frame_rect.left() + float(point[0]) / max(1.0, float(frame_w)) * frame_rect.width()
        y = frame_rect.top() + float(point[1]) / max(1.0, float(frame_h)) * frame_rect.height()
        return QPointF(x, y)

    def _target_status_text(self, target: VisionTarget) -> str:
        if target.actionable:
            return "可抓取"
        if target.servo_required:
            center = target.grasp_pixel or target.display_center or target.center_px
            target_px = target.alignment_target_pixel
            if target_px is not None:
                err = math.hypot(float(center[0]) - float(target_px[0]), float(center[1]) - float(target_px[1]))
                return f"需对准 {err:.0f}px"
            return "需对准"
        reason = str(target.invalid_reason or "未就绪")
        return reason[:18]

    def _draw_vision_targets(self, painter: QPainter, frame_rect: QRectF, frame_w: int, frame_h: int) -> None:
        if not self.vision_targets:
            return
        packet_target = _coerce_xy(self.vision_packet.get("alignment_target_pixel"))
        for target in self.vision_targets:
            target_id = str(target.slot_id if target.slot_id is not None else target.id)
            active = target_id == self.active_id
            if self.ssv_flicker_enabled:
                flicker = self._ssvep_flash_on(target_id)
                color = QColor(_target_color(target_id)) if flicker else QColor("#020617")
                pen = QPen(QColor("#ffffff" if flicker else "#38bdf8"), 5 if flicker else 3)
            else:
                flicker = False
                color = QColor("#22c55e" if target.actionable else ("#f59e0b" if target.servo_required else "#94a3b8"))
                if active:
                    color = QColor("#38bdf8" if target.actionable else "#f59e0b")
                pen = QPen(color, 4 if active else 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            x1, y1, x2, y2 = target.bbox
            if (x2 - x1) > 1 and (y2 - y1) > 1:
                p1 = self._map_frame_point((x1, y1), frame_rect, frame_w, frame_h)
                p2 = self._map_frame_point((x2, y2), frame_rect, frame_w, frame_h)
                box_rect = QRectF(p1, p2).normalized()
                if self.ssv_flicker_enabled:
                    fill = QColor(color)
                    fill.setAlpha(115 if flicker else 45)
                    painter.setBrush(fill)
                painter.drawRoundedRect(box_rect, 5, 5)
                painter.setBrush(Qt.NoBrush)
            center_px = target.grasp_pixel or target.display_center or target.center_px
            center = self._map_frame_point(center_px, frame_rect, frame_w, frame_h)
            painter.setBrush(color)
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawEllipse(center, 10 if self.ssv_flicker_enabled or active else 5, 10 if self.ssv_flicker_enabled or active else 5)
            alignment_px = target.alignment_target_pixel or packet_target or (frame_w / 2.0, frame_h / 2.0)
            alignment = self._map_frame_point(alignment_px, frame_rect, frame_w, frame_h)
            if active:
                painter.setPen(QPen(QColor("#facc15"), 2, Qt.DashLine))
                painter.drawLine(center, alignment)
                painter.setBrush(QColor("#facc15"))
                painter.setPen(QPen(QColor("#111827"), 1))
                painter.drawEllipse(alignment, 5, 5)
            if self.ssv_flicker_enabled:
                label = f"{target_id} {self.ssv_frequencies.get(target_id, 9.0):.0f}Hz"
            else:
                label = f"{target_id} {self._target_status_text(target)}"
            label_rect = QRectF(center.x() + 10, max(frame_rect.top() + 8, center.y() - 24), 190, 24)
            painter.setBrush(QColor(15, 23, 42, 210))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(label_rect, 4, 4)
            painter.setPen(QColor("#ffffff"))
            painter.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
            painter.drawText(label_rect.adjusted(7, 0, -6, 0), Qt.AlignLeft | Qt.AlignVCenter, label)

    def _draw_overlay(self, painter: QPainter, w: int, h: int) -> None:
        scale = max(1.0, min(1.45, min(w / 1500.0, h / 850.0)))
        margin = int(round(18 * scale))
        top_h = int(round(64 * scale))
        top = QRectF(margin, int(round(16 * scale)), w - margin * 2, top_h)
        painter.setBrush(QColor(15, 23, 42, 205))
        painter.setPen(QPen(QColor(125, 211, 252, 130), 1))
        painter.drawRoundedRect(top, 8, 8)
        badge_text, badge_color = self._camera_badge()
        badge_w = int(round(72 * scale))
        badge_h = int(round(34 * scale))
        badge_x = int(round(34 * scale))
        badge_y = int(round(30 * scale))
        self._draw_badge(painter, QRectF(badge_x, badge_y, badge_w, badge_h), badge_text, badge_color)
        painter.setPen(QColor("#f8fafc"))
        painter.setFont(QFont("Microsoft YaHei", int(round(15 * scale)), QFont.Bold))
        text_x = int(round(120 * scale))
        painter.drawText(QRectF(text_x, int(round(23 * scale)), max(200, w * 0.42), int(round(28 * scale))), Qt.AlignLeft | Qt.AlignVCenter, self.phase_title)
        painter.setPen(QColor("#bfdbfe"))
        painter.setFont(QFont("Microsoft YaHei", max(10, int(round(10 * scale)))))
        painter.drawText(QRectF(text_x, int(round(50 * scale)), max(240, w * 0.56), int(round(22 * scale))), Qt.AlignLeft | Qt.AlignVCenter, self.phase_detail)
        if self.countdown_text:
            self._draw_badge(
                painter,
                QRectF(w - int(round(148 * scale)), badge_y, int(round(108 * scale)), badge_h),
                self.countdown_text,
                QColor("#f59e0b"),
                QColor("#111827"),
            )

        self._draw_target_stack(painter, w)
        if self.active_id:
            painter.setBrush(QColor(2, 132, 199, 220))
            painter.setPen(QPen(QColor("#ffffff"), 2))
            rect = QRectF(int(round(22 * scale)), int(round(94 * scale)), int(round(150 * scale)), int(round(40 * scale)))
            painter.drawRoundedRect(rect, 7, 7)
            painter.setPen(QColor("#ffffff"))
            painter.setFont(QFont("Microsoft YaHei", int(round(12 * scale)), QFont.Bold))
            painter.drawText(rect, Qt.AlignCenter, f"目标 {self.active_id} 已选")
        elif self.ssv_flicker_enabled:
            painter.setBrush(QColor(14, 165, 233, 225))
            painter.setPen(QPen(QColor("#ffffff"), 2))
            rect = QRectF(int(round(22 * scale)), int(round(94 * scale)), int(round(210 * scale)), int(round(42 * scale)))
            painter.drawRoundedRect(rect, 7, 7)
            painter.setPen(QColor("#ffffff"))
            painter.setFont(QFont("Microsoft YaHei", int(round(12 * scale)), QFont.Bold))
            painter.drawText(rect, Qt.AlignCenter, "SSVEP 闪烁选择中")

        bottom_h = int(round(88 * scale))
        bottom = QRectF(margin, h - int(round(20 * scale)) - bottom_h, w - margin * 2, bottom_h)
        painter.setBrush(QColor(15, 23, 42, 215))
        painter.setPen(QPen(QColor(125, 211, 252, 120), 1))
        painter.drawRoundedRect(bottom, 8, 8)
        line_h = max(18, int(round(22 * scale)))
        line_x = bottom.left() + int(round(16 * scale))
        line_w = bottom.width() - int(round(32 * scale))
        painter.setPen(QColor("#e0f2fe"))
        painter.setFont(QFont("Microsoft YaHei", max(10, int(round(10 * scale))), QFont.Bold))
        painter.drawText(QRectF(line_x, bottom.top() + int(round(8 * scale)), line_w, line_h), Qt.AlignLeft | Qt.AlignVCenter, self.pose_text)
        painter.setPen(QColor("#bfdbfe"))
        painter.setFont(QFont("Microsoft YaHei", max(10, int(round(10 * scale)))))
        painter.drawText(QRectF(line_x, bottom.top() + int(round(32 * scale)), line_w, line_h), Qt.AlignLeft | Qt.AlignVCenter, self.grip_text)
        painter.setPen(QColor("#fef3c7" if "需" in self.vision_status_text or "缺少" in self.vision_status_text else "#bbf7d0"))
        painter.setFont(QFont("Microsoft YaHei", max(10, int(round(10 * scale))), QFont.Bold))
        painter.drawText(QRectF(line_x, bottom.top() + int(round(54 * scale)), line_w, line_h), Qt.AlignLeft | Qt.AlignVCenter, self.vision_status_text[:110])
        painter.setPen(QColor("#93c5fd"))
        painter.setFont(QFont("Consolas", max(9, int(round(9 * scale)))))
        painter.drawText(QRectF(line_x, bottom.top() + int(round(74 * scale)), line_w, max(16, int(round(18 * scale)))), Qt.AlignLeft | Qt.AlignVCenter, self.keyboard_text)

    def _draw_placeholder_scene(self, painter: QPainter, w: int, h: int) -> None:
        center = QPointF(w / 2.0, h / 2.0)
        painter.fillRect(self.rect(), QColor("#0f172a"))
        painter.setBrush(QColor(15, 23, 42))
        painter.setPen(QPen(QColor("#1e3a5f"), 1))
        painter.drawRoundedRect(QRectF(20, 94, w - 40, h - 190), 10, 10)
        painter.setPen(QColor("#334155"))
        for i in range(1, 4):
            y = 94 + (h - 190) * i / 4
            painter.drawLine(38, int(y), w - 38, int(y))
        for i in range(1, 4):
            x = 20 + (w - 40) * i / 4
            painter.drawLine(int(x), 112, int(x), h - 112)
        ids = list(getattr(self, "target_ids", _target_ids()))
        block_w = max(72.0, min(170.0, w * 0.09))
        block_h = max(46.0, min(105.0, h * 0.095))
        for index, label in enumerate(ids):
            row, col = divmod(index, 2)
            x_ratio = 0.34 + col * 0.32
            y_ratio = 0.36 + row * 0.22
            color = _target_color(label)
            x, y = w * x_ratio, h * y_ratio
            active = label == self.active_id
            flicker = self.ssv_flicker_enabled and self._ssvep_flash_on(label)
            rect = QRectF(x - block_w / 2.0, y - block_h / 2.0, block_w, block_h)
            if self.ssv_flicker_enabled:
                painter.setBrush(QColor(color) if flicker else QColor("#020617"))
                painter.setPen(QPen(QColor("#ffffff" if flicker else "#38bdf8"), 4 if flicker else 2))
            else:
                painter.setBrush(QColor(color))
                painter.setPen(QPen(QColor("#ffffff"), 3 if active else 1))
            painter.drawRoundedRect(rect, 7, 7)
            if active or flicker:
                pulse = 8 + (self.phase % 5) * 4
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor("#7dd3fc" if active else "#fef08a"), 2))
                painter.drawEllipse(QRectF(rect.left() - pulse, rect.top() - pulse, rect.width() + pulse * 2, rect.height() + pulse * 2))
                if active:
                    painter.setPen(QPen(QColor("#7dd3fc"), 1, Qt.DashLine))
                    painter.drawLine(int(center.x()), int(center.y()), int(x), int(y))
            painter.setPen(QColor("#020617" if flicker or active else "#e0f2fe"))
            painter.setFont(QFont("Segoe UI", max(13, min(26, int(block_h * 0.34))), QFont.Bold))
            painter.drawText(rect, Qt.AlignCenter, label)
        painter.setPen(QColor("#94a3b8"))
        painter.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        painter.drawText(QRectF(28, h - 128, w - 56, 24), Qt.AlignCenter, "等待实时摄像头画面")
        painter.setFont(QFont("Consolas", 9))
        painter.drawText(QRectF(28, h - 104, w - 56, 22), Qt.AlignCenter, self.camera_status)
        self._draw_crosshair(painter, center, w, h)

    def _scaled_frame(self, width: int, height: int) -> QPixmap | None:
        pixmap = self._frame_pixmap
        if pixmap is None or pixmap.isNull():
            return None
        cache_key = int(pixmap.cacheKey())
        cached = self._scaled_frame_cache
        if cached is not None and cached[0] == width and cached[1] == height and cached[2] == cache_key:
            return cached[4]
        scaled = pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaled_frame_cache = (width, height, cache_key, int(time.monotonic() * 1000), scaled)
        return scaled

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#0f172a"))
        w, h = max(1, self.width()), max(1, self.height())
        center = QPointF(w / 2.0, h / 2.0)
        if self.latest_frame is not None and not self.latest_frame.isNull():
            pixmap = self._frame_pixmap or QPixmap.fromImage(self.latest_frame)
            scaled = self._scaled_frame(w, h) or pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = int((w - scaled.width()) / 2)
            y = int((h - scaled.height()) / 2)
            painter.drawPixmap(x, y, scaled)
            if x > 0:
                painter.fillRect(QRectF(0, 0, x, h), QColor(2, 6, 23, 190))
                painter.fillRect(QRectF(x + scaled.width(), 0, w - x - scaled.width(), h), QColor(2, 6, 23, 190))
            if y > 0:
                painter.fillRect(QRectF(0, 0, w, y), QColor(2, 6, 23, 190))
                painter.fillRect(QRectF(0, y + scaled.height(), w, h - y - scaled.height()), QColor(2, 6, 23, 190))
            self._draw_vision_targets(painter, QRectF(x, y, scaled.width(), scaled.height()), pixmap.width(), pixmap.height())
            self._draw_crosshair(painter, center, w, h)
            self._draw_overlay(painter, w, h)
            return
        self._draw_placeholder_scene(painter, w, h)
        self._draw_overlay(painter, w, h)


class SsvepStimulusWidget(QWidget):
    FREQUENCIES = (9.0, 11.0, 13.0, 15.0)
    TARGET_NAMES = ("目标 1 (上)", "目标 2 (左)", "目标 3 (下)", "目标 4 (右)")
    PREPARE_SEC = 1.0
    ACTIVE_SEC = 5.0
    REST_SEC = 4.0
    TARGET_REPEATS = 10
    TRIAL_SEC = PREPARE_SEC + ACTIVE_SEC + REST_SEC
    TARGET_SEQUENCE = (0, 1, 2, 3) * TARGET_REPEATS
    TOTAL_TRIALS = len(TARGET_SEQUENCE)
    TOTAL_SEC = TRIAL_SEC * TOTAL_TRIALS

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.running = False
        self.flicker_enabled = False
        self.phase = 0
        self.progress = 0.0
        self.active_target = 0
        self.current_trial = 1
        self.phase_elapsed_sec = 0.0
        self.frequencies = self.FREQUENCIES
        self.target_names = self.TARGET_NAMES
        self.start_time = 0.0
        self.render_timer = QTimer(self)
        self.render_timer.setTimerType(Qt.PreciseTimer)
        self.render_timer.setInterval(16)
        self.render_timer.timeout.connect(self._on_render_tick)
        self.setMinimumHeight(620)
        self.setMaximumHeight(16777215)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_running(self, running: bool) -> None:
        self.running = bool(running)
        self._sync_protocol_from_progress()
        self.flicker_enabled = self.running and self._phase_text() == "闪烁采集"
        self.start_time = time.perf_counter()
        if self.running and not self.render_timer.isActive():
            self.render_timer.start()
        elif not self.running and self.render_timer.isActive():
            self.render_timer.stop()
        self.update()

    def set_progress(self, progress: int | float) -> None:
        previous_flicker = self.flicker_enabled
        previous_target = self.active_target
        self.progress = max(0.0, min(100.0, float(progress)))
        self._sync_protocol_from_progress()
        self.flicker_enabled = self.running and self._phase_text() == "闪烁采集"
        if self.flicker_enabled and (not previous_flicker or previous_target != self.active_target):
            self.start_time = time.perf_counter()
            self.phase = 0
        self.update()

    def advance(self) -> None:
        self._on_render_tick()

    def _on_render_tick(self) -> None:
        if self.running:
            self.phase = (self.phase + 1) % 100_000
        self.update()

    @staticmethod
    def _target_flash_on(frequency: float, elapsed_sec: float) -> bool:
        freq = max(0.1, float(frequency))
        half_period = 0.5 / freq
        return int(math.floor(max(0.0, float(elapsed_sec)) / half_period)) % 2 == 0

    def _sync_protocol_from_progress(self) -> None:
        elapsed = max(0.0, min(float(self.TOTAL_SEC), self.progress / 100.0 * float(self.TOTAL_SEC)))
        trial_index = int(min(self.TOTAL_TRIALS - 1, max(0, math.floor(elapsed / float(self.TRIAL_SEC)))))
        self.current_trial = trial_index + 1
        self.active_target = int(self.TARGET_SEQUENCE[trial_index])
        self.phase_elapsed_sec = float(elapsed - trial_index * float(self.TRIAL_SEC))

    def _phase_text(self) -> str:
        if not self.running:
            return "待采集"
        elapsed = self.phase_elapsed_sec + 1e-9
        if elapsed < self.PREPARE_SEC:
            return "准备注视"
        if elapsed < self.PREPARE_SEC + self.ACTIVE_SEC:
            return "闪烁采集"
        return "休息恢复"

    def current_task_title(self) -> str:
        if not self.running:
            return "等待开始"
        return f"目标 {self.active_target + 1} · Trial {self.current_trial}/{self.TOTAL_TRIALS}"

    def capture_label(self) -> int:
        if self.running and self._phase_text() == "闪烁采集":
            return int(self.active_target + 1)
        return 0

    def _target_rects(self, width: int, height: int) -> tuple[QRectF, QRectF, QRectF, QRectF]:
        grid = QRectF(22, 92, max(1, width - 44), max(1, height - 154))
        pad = 18.0
        box_size = max(190.0, min(330.0, grid.width() * 0.22, (grid.height() - 64.0) / 2.0))
        cx = grid.center().x()
        cy = grid.center().y()
        vertical_limit = max(0.0, grid.height() / 2.0 - box_size / 2.0 - pad * 0.25)
        horizontal_limit = max(0.0, grid.width() / 2.0 - box_size / 2.0 - pad)
        vertical_offset = min(max(150.0, box_size * 0.55), vertical_limit)
        horizontal_offset = min(max(160.0, box_size * 1.35), horizontal_limit)
        return (
            QRectF(cx - box_size / 2.0, cy - vertical_offset - box_size / 2.0, box_size, box_size),
            QRectF(cx - horizontal_offset - box_size / 2.0, cy - box_size / 2.0, box_size, box_size),
            QRectF(cx - box_size / 2.0, cy + vertical_offset - box_size / 2.0, box_size, box_size),
            QRectF(cx + horizontal_offset - box_size / 2.0, cy - box_size / 2.0, box_size, box_size),
        )

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#07111f"))
        w, h = max(1, self.width()), max(1, self.height())

        painter.setPen(QColor("#e0f2fe"))
        painter.setFont(QFont("Microsoft YaHei", 22, QFont.Bold))
        painter.drawText(QRectF(28, 18, w - 56, 36), Qt.AlignLeft | Qt.AlignVCenter, "SSVEP 训练数据采集")
        painter.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        painter.setPen(QColor("#bae6fd"))
        status = (
            f"当前阶段：{self._phase_text()}   Trial：{self.current_trial}/{self.TOTAL_TRIALS}   "
            f"注视目标：{self.active_target + 1}   频率：{self.frequencies[self.active_target]:.1f} Hz   "
            f"进度：{self.progress:.1f}%"
        )
        painter.drawText(QRectF(28, 52, w - 56, 28), Qt.AlignLeft | Qt.AlignVCenter, status)

        rects = self._target_rects(w, h)
        center = QPointF((rects[1].right() + rects[3].left()) / 2.0, (rects[0].bottom() + rects[2].top()) / 2.0)
        painter.setPen(QPen(QColor("#38bdf8"), 3))
        painter.drawLine(QPointF(center.x() - 26, center.y()), QPointF(center.x() + 26, center.y()))
        painter.drawLine(QPointF(center.x(), center.y() - 26), QPointF(center.x(), center.y() + 26))
        painter.setBrush(QColor("#0ea5e9"))
        painter.drawEllipse(center, 6, 6)
        elapsed = 0.0 if not self.running else max(0.0, time.perf_counter() - self.start_time)
        for idx, frequency in enumerate(self.frequencies):
            rect = rects[idx]
            active = idx == self.active_target
            flash_on = self.flicker_enabled and self._target_flash_on(frequency, elapsed)
            fill = QColor("#ffffff" if flash_on else ("#000000" if self.flicker_enabled else "#151B24"))
            if not active and self.running:
                fill.setAlpha(210 if self.flicker_enabled else 185)
            painter.setBrush(fill)
            painter.setPen(QPen(QColor("#0284c7" if active else "#252e3c"), 5 if active else 2))
            painter.drawRoundedRect(rect, 8, 8)

            text_color = QColor("#000000" if flash_on else ("#445161" if self.flicker_enabled else "#8b97a5"))
            painter.setPen(text_color)
            painter.setFont(QFont("Microsoft YaHei", max(22, min(34, int(rect.width() * 0.13))), QFont.Bold))
            painter.drawText(rect.adjusted(0, 12, 0, -rect.height() * 0.45), Qt.AlignCenter, self.target_names[idx])
            painter.setFont(QFont("Segoe UI", max(20, min(30, int(rect.width() * 0.11))), QFont.Bold))
            painter.drawText(rect.adjusted(0, rect.height() * 0.32, 0, -24), Qt.AlignCenter, f"{frequency:.1f} Hz")

        painter.setPen(QColor("#93c5fd"))
        painter.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        painter.drawText(
            QRectF(28, h - 36, w - 56, 24),
            Qt.AlignLeft | Qt.AlignVCenter,
            "采集节奏：准备 1 秒  →  闪烁采集 5 秒  →  休息 4 秒；每个目标 10 组，共 40 个目标试次。",
        )


class MiCueWidget(QWidget):
    DIRECTIONS = ("LEFT", "RIGHT", "FEET", "TONGUE")
    TITLE_MAP = {
        "LEFT": "左手握拳",
        "RIGHT": "右手握拳",
        "FEET": "双脚运动",
        "TONGUE": "舌头伸缩",
    }
    DESC_MAP = {
        "LEFT": "持续想象左手握拳与放松动作，不要真实移动手臂。",
        "RIGHT": "持续想象右手握拳与放松动作，不要真实移动手臂。",
        "FEET": "持续想象双脚交替踩踏动作，不要真实移动腿部。",
        "TONGUE": "持续想象舌头前伸/上抬动作，不要真实张口吐舌。",
    }
    READY_SEC = 2.0
    CUE_SEC = 4.0
    REST_SEC = 2.0
    TRIALS_PER_CLASS = 10
    TRIAL_SEC = READY_SEC + CUE_SEC + REST_SEC
    TRIAL_DIRECTIONS = DIRECTIONS * TRIALS_PER_CLASS
    TOTAL_TRIALS = len(TRIAL_DIRECTIONS)
    TOTAL_SEC = TRIAL_SEC * TOTAL_TRIALS
    READY_PROGRESS = 100.0 * READY_SEC / TOTAL_SEC
    REST_PROGRESS = 100.0 * (READY_SEC + CUE_SEC) / TOTAL_SEC

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.running = False
        self.phase = 0
        self.progress = 0.0
        self.active_cue = 0
        self.current_trial = 1
        self.phase_elapsed_sec = 0.0
        self.state = "IDLE"
        self.direction = "LEFT"
        self.cue_asset_paths = self._cue_asset_paths()
        self.cue_pixmaps = {name: self._load_cue_pixmap(path) for name, path in self.cue_asset_paths.items()}
        self.setMinimumHeight(520)
        self.setMaximumHeight(16777215)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.status_label = QLabel(self)
        self.title_label = QLabel(self)
        self.subtitle_label = QLabel(self)
        for label in (self.status_label, self.title_label, self.subtitle_label):
            label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            label.setStyleSheet("background: transparent;")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self._update_prompt_labels()

    @staticmethod
    def _cue_asset_paths() -> dict[str, Path]:
        root = Path(__file__).resolve().parents[1]
        cue_dir = root / "01_MI" / "mi_classifier_latest" / "code" / "collection" / "assets" / "cues"
        return {
            "LEFT": cue_dir / "left_hand.png",
            "RIGHT": cue_dir / "right_hand.jpg",
            "FEET": cue_dir / "feet.png",
            "TONGUE": cue_dir / "tongue.png",
        }

    @staticmethod
    def _load_cue_pixmap(path: Path) -> QPixmap:
        pixmap = QPixmap(str(path))
        if not pixmap.isNull():
            return pixmap
        try:
            from PIL import Image

            with Image.open(path) as image:
                rgba = image.convert("RGBA")
                width, height = rgba.size
                payload = rgba.tobytes("raw", "RGBA")
            qimage = QImage(payload, width, height, QImage.Format_RGBA8888).copy()
            return QPixmap.fromImage(qimage)
        except Exception:
            return QPixmap()

    def set_running(self, running: bool) -> None:
        self.running = bool(running)
        self._sync_phase_from_progress()
        self._update_prompt_labels()
        self.update()

    def set_progress(self, progress: int | float) -> None:
        self.progress = max(0.0, min(100.0, float(progress)))
        self._sync_phase_from_progress()
        self._update_prompt_labels()
        self.update()

    def advance(self) -> None:
        self.phase = (self.phase + 1) % 120
        self.update()

    def _sync_phase_from_progress(self) -> None:
        elapsed = max(0.0, min(float(self.TOTAL_SEC), self.progress / 100.0 * float(self.TOTAL_SEC)))
        trial_index = int(min(self.TOTAL_TRIALS - 1, max(0, math.floor(elapsed / float(self.TRIAL_SEC)))))
        self.current_trial = trial_index + 1
        self.direction = str(self.TRIAL_DIRECTIONS[trial_index])
        self.active_cue = int(self.DIRECTIONS.index(self.direction))
        self.phase_elapsed_sec = float(elapsed - trial_index * float(self.TRIAL_SEC))
        if not self.running:
            self.state = "IDLE"
            return
        elapsed_in_trial = self.phase_elapsed_sec + 1e-9
        if elapsed_in_trial < self.READY_SEC:
            self.state = "READY"
        elif elapsed_in_trial < self.READY_SEC + self.CUE_SEC:
            self.state = "CUE"
        else:
            self.state = "REST"

    def _phase_text(self) -> str:
        if self.state == "READY":
            return "准备阶段"
        if self.state == "CUE":
            return "运动想象"
        if self.state == "REST":
            return "休息恢复"
        return "待采集"

    def current_task_title(self) -> str:
        if self.state == "CUE":
            return f"想象：{self.TITLE_MAP[self.direction]}"
        if self.state == "READY":
            return f"准备：{self.TITLE_MAP[self.direction]}"
        if self.state == "REST":
            return "休息恢复"
        return "等待开始"

    def capture_label(self) -> int:
        if self.running and self.state == "CUE":
            return int(self.active_cue + 1)
        return 0

    def _prompt_texts(self) -> tuple[str, str, str, str]:
        trial_text = f"Trial {self.current_trial} / {self.TOTAL_TRIALS}"
        if self.state == "READY":
            return (
                f"MI 运动想象采集 · 准备阶段 · {trial_text}",
                "准备阶段",
                f"即将进入：{self.TITLE_MAP[self.direction]}。请注视中央十字，保持全身放松。",
                "#569AFF",
            )
        if self.state == "CUE":
            return (
                f"MI 运动想象采集 · 运动想象 · {trial_text} · 标签 {self.active_cue + 1}",
                f"想象任务：{self.TITLE_MAP[self.direction]}",
                self.DESC_MAP[self.direction],
                "#10B981",
            )
        if self.state == "REST":
            return (
                f"MI 运动想象采集 · 休息恢复 · {trial_text}",
                "休息恢复",
                "放空当前想象，尽量减少眨眼与吞咽次数",
                "#94A3B8",
            )
        return (
            "MI 运动想象采集 · 等待开始",
            "等待开始",
            "点击开始后进入 2 秒准备、4 秒运动想象、2 秒休息的采集范式",
            "#94A3B8",
        )

    @staticmethod
    def _set_label_font(label: QLabel, point_size: int, weight: int) -> None:
        font = QFont("Microsoft YaHei UI")
        font.setPointSize(int(point_size))
        font.setWeight(weight)
        label.setFont(font)

    def _update_prompt_labels(self) -> None:
        status, title, subtitle, title_color = self._prompt_texts()
        self.status_label.setText(status)
        self.title_label.setText(title)
        self.subtitle_label.setText(subtitle)
        self.status_label.setStyleSheet("background: transparent; color: #D2DAE5;")
        self.title_label.setStyleSheet(f"background: transparent; color: {title_color};")
        self.subtitle_label.setStyleSheet("background: transparent; color: #A4B1CD;")
        self._layout_prompt_labels()

    def _layout_prompt_labels(self) -> None:
        w, h = max(1, self.width()), max(1, self.height())
        margin_x = max(24, min(42, int(w * 0.028)))
        status_h = max(34, min(48, int(h * 0.075)))
        title_h = max(38, min(56, int(h * 0.085)))
        subtitle_h = max(26, min(38, int(h * 0.055)))
        footer_gap = max(18, min(28, int(h * 0.035)))
        title_top = h - footer_gap - title_h - subtitle_h - 8

        self.status_label.setGeometry(margin_x, 14, w - margin_x * 2, status_h)
        self.title_label.setGeometry(margin_x, title_top, w - margin_x * 2, title_h)
        self.subtitle_label.setGeometry(margin_x, title_top + title_h + 2, w - margin_x * 2, subtitle_h)
        self._set_label_font(self.status_label, max(13, min(18, int(h * 0.026))), QFont.Bold)
        self._set_label_font(self.title_label, max(22, min(34, int(h * 0.046))), QFont.Bold)
        self._set_label_font(self.subtitle_label, max(13, min(18, int(h * 0.026))), QFont.Normal)

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._layout_prompt_labels()
        super().resizeEvent(event)

    def _draw_cue_image(self, painter: QPainter, rect: QRectF, pixmap: QPixmap) -> None:
        panel = rect.adjusted(26, 8, -26, -8)
        painter.setPen(QPen(QColor("#dbe4ef"), 2))
        painter.setBrush(QColor("#ffffff"))
        painter.drawRoundedRect(panel, 10, 10)
        inner = panel.adjusted(28, 24, -28, -24)
        scaled = pixmap.scaled(inner.size().toSize(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = int(inner.left() + (inner.width() - scaled.width()) / 2)
        y = int(inner.top() + (inner.height() - scaled.height()) / 2)
        painter.drawPixmap(x, y, scaled)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.fillRect(self.rect(), QColor("#090D14"))
        w, h = max(1, self.width()), max(1, self.height())
        header_reserved = max(76, min(108, int(h * 0.15)))
        footer_reserved = max(112, min(148, int(h * 0.21)))
        drawing_rect = QRectF(48, header_reserved, w - 96, max(40, h - header_reserved - footer_reserved))
        cx, cy = drawing_rect.center().x(), drawing_rect.center().y()
        accent = QColor("#67E8B9") if self.state == "CUE" else QColor("#569AFF")

        if self.state == "READY":
            painter.setPen(QPen(accent, max(5, min(9, int(w * 0.004))), Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPointF(cx - 42, cy - 26), QPointF(cx + 42, cy - 26))
            painter.drawLine(QPointF(cx, cy - 68), QPointF(cx, cy + 16))
        elif self.state == "CUE":
            pixmap = self.cue_pixmaps.get(self.direction)
            if pixmap is not None and not pixmap.isNull():
                self._draw_cue_image(painter, drawing_rect, pixmap)
            else:
                painter.setPen(QPen(QColor("#334155"), 2))
                painter.setBrush(QColor("#111827"))
                painter.drawRoundedRect(drawing_rect.adjusted(26, 8, -26, -8), 10, 10)
        elif self.state == "REST":
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#475569"))
            painter.drawEllipse(QPointF(cx, cy - 20), 20, 20)
        else:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#1E293B"))
            painter.drawEllipse(QPointF(cx, cy), 18, 18)


class KeyboardTrainingStageWidget(QWidget):
    stage_completed = pyqtSignal()
    pipeline_updated = pyqtSignal(int, float)

    def __init__(
        self,
        stage_id: int,
        title: str,
        body: str,
        shortcut: str,
        parent: QWidget | None = None,
        *,
        stage_kind: str = "generic",
    ) -> None:
        super().__init__(parent)
        self.stage_id = int(stage_id)
        self.stage_kind = str(stage_kind or "generic").lower()
        self.shortcut = str(shortcut).upper()
        self.progress = 0
        self.capture_running = False
        self.capture_chunks: list[np.ndarray] = []
        self.capture_label_chunks: list[np.ndarray] = []
        self.captured_sample_count = 0
        self.capture_sampling_rate = 0.0
        self.capture_channel_names = list(EEG_DEFAULT_CHANNEL_NAMES)
        self.last_capture_path: Path | None = None
        self.stage_visual: QWidget | None = None
        self.ssvep_stimulus: SsvepStimulusWidget | None = None
        self.mi_cue: MiCueWidget | None = None
        self._build_ui(title, body)
        self.timer = QTimer(self)
        self.timer.setInterval(100 if self.stage_kind in {"ssvep", "mi"} else 90)
        self.timer.timeout.connect(self._tick)

    def _stage_steps(self) -> tuple[str, ...]:
        if self.stage_kind == "ssvep":
            return (
                "确认 EEG 8 通道预览区域正在显示真实信号",
                "按 1s 准备、5s 闪烁采集、4s 休息推进",
                "每个目标采集 10 组，共 40 个目标试次，只在闪烁采集期写入标签",
            )
        if self.stage_kind == "mi":
            return (
                "按 2s 准备提示、4s 运动想象、2s 休息恢复推进",
                "左手、右手、双脚、舌头每类 10 组，共 40 个主任务试次",
                "只在运动想象阶段写入 MI 训练标签，完成后进入机械臂控制",
            )
        return ("查看实时信号", "等待脑控确认结果", "确认进入下一步")

    def _run_button_text(self) -> str:
        if self.stage_kind == "ssvep":
            return "开始 SSVEP 数据采集"
        if self.stage_kind == "mi":
            return "开始 MI 数据采集"
        return "开始脑控流程演示"

    def _initial_hint(self) -> str:
        if self.stage_kind == "ssvep":
            return "点击开始后进入 SSVEP 标准采集：1s 准备、5s 闪烁、4s 休息；每目标 10 组，完成时保存训练数据。"
        if self.stage_kind == "mi":
            return "点击开始后进入 MI 标准采集：2s 准备、4s 想象、2s 休息；每类 10 组，准备和休息不入训练标签。"
        return "点击开始可跑一遍脑控流程；完成后进入下一阶段。"

    def _create_stage_visual(self) -> QWidget | None:
        if self.stage_kind == "ssvep":
            self.ssvep_stimulus = SsvepStimulusWidget(self)
            return self.ssvep_stimulus
        if self.stage_kind == "mi":
            self.mi_cue = MiCueWidget(self)
            return self.mi_cue
        return None

    def _set_stage_state(self, text: str, state: str) -> None:
        self.lbl_stage_state.setText(text)
        self.lbl_stage_state.setProperty("state", state)
        self.lbl_stage_state.style().unpolish(self.lbl_stage_state)
        self.lbl_stage_state.style().polish(self.lbl_stage_state)

    def _set_visual_running(self, running: bool) -> None:
        visual = self.stage_visual
        if visual is None:
            return
        setter = getattr(visual, "set_running", None)
        if callable(setter):
            setter(bool(running))

    def _set_visual_progress(self) -> None:
        visual = self.stage_visual
        if visual is None:
            return
        setter = getattr(visual, "set_progress", None)
        if callable(setter):
            setter(self.progress)

    def _advance_visual(self) -> None:
        visual = self.stage_visual
        if visual is None:
            return
        advance = getattr(visual, "advance", None)
        if callable(advance):
            advance()

    def _refresh_step_states(self) -> None:
        if not hasattr(self, "step_labels"):
            return
        for index, label in enumerate(self.step_labels):
            threshold = int((index + 1) * 100 / max(1, len(self.step_labels)))
            done = self.progress >= threshold
            if label.property("done") == done:
                continue
            label.setProperty("done", done)
            if done:
                label.setStyleSheet("color: #047857; font-weight: 900; padding: 7px 0;")
            else:
                label.setStyleSheet("color: #334155; font-weight: 700; padding: 7px 0;")

    def _stage_display_title(self) -> str:
        return "SSVEP 训练数据采集" if self.stage_kind == "ssvep" else "MI 训练数据采集"

    def _stage_description(self) -> str:
        if self.stage_kind == "ssvep":
            return "连接 EEG 后在此采集 SSVEP 训练样本；目标闪烁同步生成标签，完成后保存数据文件。"
        if self.stage_kind == "mi":
            return "连接 EEG 后在此采集运动想象训练样本；运动想象提示同步生成标签，完成后保存数据文件。"
        return ""

    def _stage_target_label(self) -> str:
        if self.stage_kind == "ssvep" and self.ssvep_stimulus is not None:
            return self.ssvep_stimulus.current_task_title()
        if self.stage_kind == "mi" and self.mi_cue is not None:
            return self.mi_cue.current_task_title()
        return "-"

    def _capture_class_label(self) -> int:
        if self.stage_kind == "ssvep" and self.ssvep_stimulus is not None:
            return int(self.ssvep_stimulus.capture_label())
        if self.stage_kind == "mi" and self.mi_cue is not None:
            return int(self.mi_cue.capture_label())
        return 0

    def _capture_phase_text(self) -> str:
        if not self.capture_running:
            return "待采集"
        if self.stage_kind == "ssvep" and self.ssvep_stimulus is not None:
            return self.ssvep_stimulus._phase_text()
        if self.stage_kind == "mi" and self.mi_cue is not None:
            return self.mi_cue._phase_text()
        return "采集中"

    def configure_capture_stream(self, sampling_rate: float, channel_names: Sequence[str]) -> None:
        self.capture_sampling_rate = float(sampling_rate or 0.0)
        self.capture_channel_names = [str(item) for item in channel_names] or list(EEG_DEFAULT_CHANNEL_NAMES)
        self._refresh_capture_metrics()

    def append_eeg_chunk(self, chunk: object) -> None:
        if not self.capture_running:
            return
        try:
            samples = np.asarray(chunk, dtype=np.float32)
        except Exception:
            return
        if samples.ndim != 2 or samples.shape[0] <= 0 or samples.shape[1] <= 0:
            return
        label_value = self._capture_class_label()
        if self.stage_kind in {"ssvep", "mi"} and label_value <= 0:
            return
        self.capture_chunks.append(samples.copy())
        self.capture_label_chunks.append(np.full(samples.shape[1], label_value, dtype=np.int16))
        self.captured_sample_count += int(samples.shape[1])
        self._refresh_capture_metrics()

    def _save_capture_data(self) -> Path | None:
        if not self.capture_chunks:
            return None
        data = np.concatenate(self.capture_chunks, axis=1)
        labels = np.concatenate(self.capture_label_chunks, axis=0) if self.capture_label_chunks else np.zeros(data.shape[1], dtype=np.int16)
        output_dir = Path("artifacts") / "training_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.stage_kind}_{time.strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez_compressed(
            path,
            samples=data,
            labels=labels,
            sampling_rate=np.asarray([self.capture_sampling_rate], dtype=np.float32),
            channel_names=np.asarray(self.capture_channel_names, dtype=str),
            stage_kind=np.asarray([self.stage_kind], dtype=str),
        )
        self.last_capture_path = path
        return path

    def _refresh_capture_metrics(self) -> None:
        if hasattr(self, "lbl_capture_phase"):
            self.lbl_capture_phase.setText(self._capture_phase_text())
        if hasattr(self, "lbl_capture_target"):
            self.lbl_capture_target.setText(self._stage_target_label())
        if hasattr(self, "lbl_capture_samples"):
            if self.capture_sampling_rate > 0:
                self.lbl_capture_samples.setText(f"{self.captured_sample_count / self.capture_sampling_rate:.1f} s")
            else:
                self.lbl_capture_samples.setText("0.0 s")
        if hasattr(self, "lbl_capture_rate"):
            interval = f"{1.0 / self.capture_sampling_rate:.3f} s/样本" if self.capture_sampling_rate else "等待 EEG"
            self.lbl_capture_rate.setText(interval)

    def _build_metric_tile(self, title: str, value: str) -> tuple[QFrame, QLabel]:
        tile = QFrame()
        tile.setObjectName("InfoTile")
        tile_layout = QVBoxLayout(tile)
        tile_layout.setContentsMargins(14, 10, 14, 10)
        title_label = QLabel(title)
        title_label.setObjectName("MetricTitle")
        value_label = QLabel(value)
        value_label.setObjectName("MetricValue")
        tile_layout.addWidget(title_label)
        tile_layout.addWidget(value_label)
        return tile, value_label

    def _build_acquisition_ui(self, title: str, body: str) -> None:
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.step_labels = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        header = QFrame()
        header.setObjectName("StageCard")
        header_layout = QGridLayout(header)
        header_layout.setContentsMargins(18, 14, 18, 14)
        header_layout.setHorizontalSpacing(14)
        title_label = QLabel(self._stage_display_title())
        title_label.setStyleSheet("color: #0f172a; font-size: 24px; font-weight: 900;")
        description = QLabel(self._stage_description())
        description.setWordWrap(True)
        description.setStyleSheet("color: #475569; font-size: 15px;")
        self.lbl_stage_state = QLabel("等待采集")
        self.lbl_stage_state.setObjectName("StatusChip")
        self.lbl_stage_state.setProperty("state", "pending")
        key_tile, _ = self._build_metric_tile("流程确认", "待确认")
        key_tile.setMinimumWidth(140)
        header_layout.addWidget(title_label, 0, 0)
        header_layout.addWidget(self.lbl_stage_state, 0, 1)
        header_layout.addWidget(description, 1, 0, 1, 2)
        header_layout.addWidget(key_tile, 0, 2, 2, 1)
        header_layout.setColumnStretch(0, 1)
        layout.addWidget(header, 0)

        workspace = QFrame()
        workspace.setObjectName("StageCard")
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(16, 14, 16, 14)
        workspace_layout.setSpacing(10)
        self.stage_visual = self._create_stage_visual()
        if self.stage_visual is not None:
            self.stage_visual.setMinimumHeight(520 if self.stage_kind == "ssvep" else 500)
            self.stage_visual.setMaximumHeight(16777215)
            workspace_layout.addWidget(self.stage_visual, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setMinimumHeight(28)
        workspace_layout.addWidget(self.progress_bar)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(10)
        phase_tile, self.lbl_capture_phase = self._build_metric_tile("采集阶段", "待采集")
        target_tile, self.lbl_capture_target = self._build_metric_tile("当前标签", "-")
        samples_tile, self.lbl_capture_samples = self._build_metric_tile("采集时长", "0.0 s")
        rate_tile, self.lbl_capture_rate = self._build_metric_tile("采样间隔", "等待 EEG")
        metrics.addWidget(phase_tile, 0, 0)
        metrics.addWidget(target_tile, 0, 1)
        metrics.addWidget(samples_tile, 0, 2)
        metrics.addWidget(rate_tile, 0, 3)
        workspace_layout.addLayout(metrics)
        layout.addWidget(workspace, 1)

        controls = QFrame()
        controls.setObjectName("StageCard")
        controls_layout = QGridLayout(controls)
        controls_layout.setContentsMargins(16, 12, 16, 12)
        controls_layout.setHorizontalSpacing(12)
        self.btn_run = QPushButton(self._run_button_text())
        self.btn_run.setObjectName("AccentButton")
        self.btn_complete = QPushButton("保存并进入下一步")
        self.btn_complete.setObjectName("ActionButton")
        self.lbl_operator_hint = QLabel(self._initial_hint())
        self.lbl_operator_hint.setWordWrap(True)
        self.lbl_operator_hint.setStyleSheet("color: #475569; font-size: 15px;")
        controls_layout.addWidget(self.btn_run, 0, 0)
        controls_layout.addWidget(self.btn_complete, 0, 1)
        controls_layout.addWidget(self.lbl_operator_hint, 0, 2)
        controls_layout.setColumnStretch(0, 1)
        controls_layout.setColumnStretch(1, 1)
        controls_layout.setColumnStretch(2, 2)
        layout.addWidget(controls, 0)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.hide()
        self.btn_run.clicked.connect(self.start_demo)
        self.btn_complete.clicked.connect(self.complete_stage)
        self._refresh_capture_metrics()

    def _build_ui(self, title: str, body: str) -> None:
        if self.stage_kind in {"ssvep", "mi"}:
            self._build_acquisition_ui(title, body)
            return
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        header = QFrame()
        header.setObjectName("StageCard")
        header.setMinimumHeight(112)
        header_layout = QGridLayout(header)
        header_layout.setContentsMargins(16, 14, 16, 14)
        header_layout.setHorizontalSpacing(14)
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #0f172a; font-size: 22px; font-weight: 900;")
        description = QLabel(body)
        description.setWordWrap(True)
        description.setStyleSheet("color: #475569; font-size: 14px;")
        self.lbl_stage_state = QLabel("等待开始")
        self.lbl_stage_state.setObjectName("StatusChip")
        self.lbl_stage_state.setProperty("state", "pending")
        key_tile = QFrame()
        key_tile.setObjectName("InfoTile")
        key_layout = QVBoxLayout(key_tile)
        key_layout.setContentsMargins(12, 8, 12, 8)
        key_title = QLabel("流程确认")
        key_title.setObjectName("MetricTitle")
        key_value = QLabel("待确认")
        key_value.setObjectName("MetricValue")
        key_layout.addWidget(key_title)
        key_layout.addWidget(key_value)
        header_layout.addWidget(title_label, 0, 0)
        header_layout.addWidget(self.lbl_stage_state, 0, 1)
        header_layout.addWidget(description, 1, 0)
        header_layout.addWidget(key_tile, 0, 2, 2, 1)
        header_layout.setColumnStretch(0, 1)
        layout.addWidget(header)

        controls = QFrame()
        controls.setObjectName("StageCard")
        controls_layout = QGridLayout(controls)
        controls_layout.setContentsMargins(16, 14, 16, 14)
        controls_layout.setHorizontalSpacing(12)
        controls_layout.setVerticalSpacing(10)
        row = 0
        self.stage_visual = self._create_stage_visual()
        if self.stage_visual is not None:
            controls_layout.addWidget(self.stage_visual, row, 0, 1, 2)
            controls_layout.setRowStretch(row, 3)
            row += 1
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setMinimumHeight(22)
        controls_layout.addWidget(self.progress_bar, row, 0, 1, 2)
        row += 1
        self.btn_run = QPushButton("开始脑控采集流程")
        self.btn_run.setObjectName("AccentButton")
        self.btn_complete = QPushButton("完成本阶段并进入下一步")
        self.btn_complete.setObjectName("ActionButton")
        self.btn_run.setText(self._run_button_text())
        self.btn_complete.setText("完成本阶段并进入下一步")
        controls_layout.addWidget(self.btn_run, row, 0)
        controls_layout.addWidget(self.btn_complete, row, 1)
        row += 1

        checklist = QFrame()
        checklist.setObjectName("ChecklistCard")
        checklist_layout = QVBoxLayout(checklist)
        checklist_layout.setContentsMargins(12, 10, 12, 10)
        checklist_title = QLabel("流程检查")
        checklist_title.setObjectName("MetricValue")
        checklist_layout.addWidget(checklist_title)
        if self.stage_id == 1:
            steps = ("实时 8 通道信号可见", "视觉刺激流程位置保留", "SSVEP 目标确认结果")
        else:
            steps = ("静息/提示/想象流程位置保留", "MI 运动意图输出", "完成后进入机械臂控制")
        steps = self._stage_steps()
        self.step_labels: list[QLabel] = []
        for index, step in enumerate(steps, start=1):
            label = QLabel(f"{index}. {step}")
            label.setWordWrap(True)
            label.setStyleSheet("color: #334155; font-weight: 700; padding: 7px 0;")
            checklist_layout.addWidget(label)
            self.step_labels.append(label)
        checklist_layout.addStretch(1)

        action_panel = QFrame()
        action_panel.setObjectName("ChecklistCard")
        action_layout = QVBoxLayout(action_panel)
        action_layout.setContentsMargins(12, 10, 12, 10)
        action_title = QLabel("操作状态")
        action_title.setObjectName("MetricValue")
        self.lbl_operator_hint = QLabel("点击开始可跑一遍脑控流程；完成后进入下一阶段。")
        self.lbl_operator_hint.setWordWrap(True)
        self.lbl_operator_hint.setText(self._initial_hint())
        self.lbl_operator_hint.setStyleSheet("color: #475569; font-size: 15px; line-height: 1.35;")
        action_layout.addWidget(action_title)
        action_layout.addWidget(self.lbl_operator_hint)
        action_layout.addStretch(1)
        controls_layout.addWidget(checklist, row, 0)
        controls_layout.addWidget(action_panel, row, 1)
        controls_layout.setRowStretch(row, 1)
        controls_layout.setColumnStretch(0, 1)
        controls_layout.setColumnStretch(1, 1)
        layout.addWidget(controls, 1)

        log_shell = QFrame()
        log_shell.setObjectName("StageCard")
        log_layout = QVBoxLayout(log_shell)
        log_layout.setContentsMargins(12, 10, 12, 10)
        log_title = QLabel("运行记录")
        log_title.setObjectName("MetricValue")
        self.log = QPlainTextEdit()
        self.log.setObjectName("LogView")
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(240)
        self.log.setMaximumHeight(104)
        log_layout.addWidget(log_title)
        log_layout.addWidget(self.log)
        if self.stage_kind in {"ssvep", "mi"}:
            log_shell.hide()
        layout.addWidget(log_shell)
        self.btn_run.clicked.connect(self.start_demo)
        self.btn_complete.clicked.connect(self.complete_stage)
        self._refresh_step_states()

    def start_demo(self) -> None:
        self.progress = 0
        self.progress_bar.setValue(0)
        self.capture_running = self.stage_kind in {"ssvep", "mi"}
        self.capture_chunks.clear()
        self.capture_label_chunks.clear()
        self.captured_sample_count = 0
        self.last_capture_path = None
        self.btn_run.setEnabled(False)
        self._set_stage_state("采集中", "active")
        self._set_visual_running(True)
        self._set_visual_progress()
        self._refresh_step_states()
        self._refresh_capture_metrics()
        self.lbl_stage_state.setText("采集中")
        if self.stage_kind == "ssvep":
            self.lbl_operator_hint.setText("SSVEP 数据采集中；仅 5 秒闪烁采集期写入目标标签，准备和休息不会写入训练样本。")
            self.log.appendPlainText("[capture] SSVEP collection started.")
        elif self.stage_kind == "mi":
            self.lbl_operator_hint.setText("MI 数据采集中；仅 4 秒运动想象期写入方向标签，准备和休息不会写入训练样本。")
            self.log.appendPlainText("[capture] MI collection started.")
        else:
            self.lbl_operator_hint.setText("正在按脑控流程推进。")
            self.log.appendPlainText("[bci] flow preview started.")
        self._set_stage_state("采集中", "active")
        self.timer.start()

    def _tick(self) -> None:
        if self.stage_kind == "ssvep":
            step = 100.0 / max(1.0, SsvepStimulusWidget.TOTAL_SEC * 10.0)
        elif self.stage_kind == "mi":
            step = 100.0 / max(1.0, MiCueWidget.TOTAL_SEC * 10.0)
        else:
            step = 4.0
        self.progress = min(100.0, float(self.progress) + step)
        progress_value = int(round(self.progress))
        if self.progress_bar.value() != progress_value:
            self.progress_bar.setValue(progress_value)
        self._set_visual_progress()
        self._advance_visual()
        self._refresh_step_states()
        self.pipeline_updated.emit(self.stage_id, float(self.progress))
        if self.progress >= 100:
            self.timer.stop()
            self.btn_run.setEnabled(True)
            self.capture_running = False
            self._set_visual_running(False)
            self._set_stage_state("可以保存", "ready")
            self.lbl_stage_state.setText("可以保存")
            self._refresh_capture_metrics()
            self.lbl_operator_hint.setText("本轮采集已停止。确认样本数后，点击保存并进入下一步。")
            self.log.appendPlainText("[capture] collection timer finished.")

    def complete_stage(self) -> None:
        self.timer.stop()
        self.capture_running = False
        self.progress = 100
        self.progress_bar.setValue(100)
        self._set_visual_progress()
        self._set_visual_running(False)
        self._refresh_step_states()
        self.pipeline_updated.emit(self.stage_id, 100.0)
        saved_path = self._save_capture_data() if self.stage_kind in {"ssvep", "mi"} else None
        self._set_stage_state("已保存" if saved_path is not None else "已完成", "ready")
        self.lbl_stage_state.setText("已保存" if saved_path is not None else "已完成")
        self._refresh_capture_metrics()
        if saved_path is not None:
            self.lbl_operator_hint.setText(f"训练数据已保存：{saved_path}")
            self.log.appendPlainText(f"[capture] saved {saved_path}")
        elif self.stage_kind in {"ssvep", "mi"}:
            self.lbl_operator_hint.setText("未收到 EEG 样本，本阶段未生成训练数据文件；请先启动右侧实时信号后再采集。")
            self.log.appendPlainText("[capture] no EEG samples received; no file saved.")
        else:
            self.lbl_operator_hint.setText("本阶段已确认完成。")
            self.log.appendPlainText("[operator] 阶段确认完成。")
        self.stage_completed.emit()

    def handle_shortcut(self, event: QKeyEvent) -> bool:
        if event.text().upper() == self.shortcut:
            self.complete_stage()
            return True
        return False


class PretrainingProgressWidget(QWidget):
    training_finished = pyqtSignal()
    pipeline_updated = pyqtSignal(int, float)

    TRAINING_INTERVAL_MS = 120
    TRAINING_STEP = 2.4

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.progress = 0.0
        self.training_running = False
        self.training_complete = False
        self.ssvep_data_path: Path | None = None
        self.mi_data_path: Path | None = None
        self._finished_emitted = False
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.setInterval(self.TRAINING_INTERVAL_MS)
        self.timer.timeout.connect(self._tick)

    def _build_ui(self) -> None:
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        header = QFrame()
        header.setObjectName("StageCard")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(22, 18, 22, 18)
        title = QLabel("预训练模型生成")
        title.setStyleSheet("color: #0f172a; font-size: 28px; font-weight: 900;")
        subtitle = QLabel("SSVEP 和 MI 预训练数据采集完成后，先在这里完成模型训练；训练完成前不能进入机械臂控制。")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #475569; font-size: 16px;")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addWidget(header, 0)

        panel = QFrame()
        panel.setObjectName("StageCard")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(28, 64, 28, 26)
        panel_layout.setSpacing(18)

        self.lbl_status = QLabel("等待训练启动")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #0369a1; font-size: 34px; font-weight: 900;")
        self.lbl_detail = QLabel("完成 MI 数据保存后将自动开始训练。")
        self.lbl_detail.setAlignment(Qt.AlignCenter)
        self.lbl_detail.setWordWrap(True)
        self.lbl_detail.setStyleSheet("color: #334155; font-size: 18px;")
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        self.training_progress.setMinimumHeight(34)
        self.training_progress.setFormat("%p%")

        panel_layout.addWidget(self.lbl_status)
        panel_layout.addWidget(self.lbl_detail)
        panel_layout.addWidget(self.training_progress)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(12)
        metrics.setVerticalSpacing(12)
        ssvep_tile, self.lbl_ssvep_data = self._build_metric_tile("SSVEP 数据", "等待保存")
        mi_tile, self.lbl_mi_data = self._build_metric_tile("MI 数据", "等待保存")
        model_tile, self.lbl_model_state = self._build_metric_tile("模型状态", "未训练")
        gate_tile, self.lbl_gate_state = self._build_metric_tile("机械臂入口", "训练完成后解锁")
        metrics.addWidget(ssvep_tile, 0, 0)
        metrics.addWidget(mi_tile, 0, 1)
        metrics.addWidget(model_tile, 1, 0)
        metrics.addWidget(gate_tile, 1, 1)
        panel_layout.addLayout(metrics)

        self.btn_enter_robot = QPushButton("训练完成后进入机械臂控制")
        self.btn_enter_robot.setObjectName("ActionButton")
        self.btn_enter_robot.setEnabled(False)
        panel_layout.addWidget(self.btn_enter_robot)
        panel_layout.addWidget(self._build_training_timeline(), 1)
        layout.addWidget(panel, 1)

    def _build_metric_tile(self, title: str, value: str) -> tuple[QFrame, QLabel]:
        tile = QFrame()
        tile.setObjectName("InfoTile")
        tile_layout = QVBoxLayout(tile)
        tile_layout.setContentsMargins(16, 12, 16, 12)
        title_label = QLabel(title)
        title_label.setObjectName("MetricTitle")
        value_label = QLabel(value)
        value_label.setObjectName("MetricValue")
        value_label.setWordWrap(True)
        tile_layout.addWidget(title_label)
        tile_layout.addWidget(value_label)
        return tile, value_label

    def _build_training_timeline(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("InfoTile")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(8)
        title = QLabel("训练步骤")
        title.setObjectName("MetricTitle")
        layout.addWidget(title)
        self.training_step_labels: list[QLabel] = []
        for text in ("整理训练数据", "提取有效特征", "训练分类模型", "写入控制参数"):
            label = QLabel(text)
            label.setMinimumHeight(42)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            label.setStyleSheet("color: #64748b; font-size: 18px; font-weight: 800; padding: 8px 10px;")
            layout.addWidget(label, 1)
            self.training_step_labels.append(label)
        return frame

    def start_training(self, *, ssvep_data_path: Path | None, mi_data_path: Path | None) -> None:
        self.ssvep_data_path = ssvep_data_path
        self.mi_data_path = mi_data_path
        self.progress = 0.0
        self.training_running = True
        self.training_complete = False
        self._finished_emitted = False
        self.training_progress.setValue(0)
        self._set_enter_button_kind("ActionButton")
        self.btn_enter_robot.setEnabled(False)
        self.btn_enter_robot.setText("训练完成后进入机械臂控制")
        self.lbl_ssvep_data.setText(self._data_label(ssvep_data_path))
        self.lbl_mi_data.setText(self._data_label(mi_data_path))
        self.lbl_model_state.setText("训练中")
        self.lbl_gate_state.setText("训练完成后解锁")
        self._refresh_status()
        self.pipeline_updated.emit(3, 0.0)
        self.timer.start()

    def finish_training_for_test(self) -> None:
        self.progress = 100.0
        self._finish_training()

    def _data_label(self, path: Path | None) -> str:
        if path is None:
            return "未生成文件，按流程继续演示训练"
        return Path(path).name

    def _refresh_status(self) -> None:
        if self.progress < 22:
            status = "正在整理训练数据"
            detail = "校验 SSVEP 与 MI 标签，准备进入特征提取。"
        elif self.progress < 52:
            status = "正在提取特征"
            detail = "从有效采集片段中提取训练特征，准备生成分类模型。"
        elif self.progress < 82:
            status = "正在训练分类模型"
            detail = "生成 SSVEP 与 MI 的预训练控制模型，机械臂入口保持锁定。"
        elif self.progress < 100:
            status = "正在写入控制参数"
            detail = "保存模型参数并刷新机械臂控制入口。"
        else:
            status = "训练完成"
            detail = "预训练模型已生成，可以进入机械臂控制界面。"
        self.lbl_status.setText(status)
        self.lbl_detail.setText(detail)
        self._refresh_training_steps()

    def _refresh_training_steps(self) -> None:
        if not hasattr(self, "training_step_labels"):
            return
        boundaries = (22.0, 52.0, 82.0, 100.0)
        for index, label in enumerate(self.training_step_labels):
            boundary = boundaries[index]
            if self.progress >= boundary:
                prefix = "完成"
                color = "#047857"
                bg = "#ecfdf5"
            elif index == 0 or self.progress >= boundaries[index - 1]:
                prefix = "进行中"
                color = "#0369a1"
                bg = "#e0f2fe"
            else:
                prefix = "等待"
                color = "#64748b"
                bg = "#f8fafc"
            base_text = ("整理训练数据", "提取有效特征", "训练分类模型", "写入控制参数")[index]
            state_key = f"{prefix}:{color}:{bg}"
            if label.property("stateKey") == state_key:
                continue
            label.setProperty("stateKey", state_key)
            label.setText(f"{index + 1}. {base_text}    {prefix}")
            label.setStyleSheet(
                f"color: {color}; background: {bg}; border: 1px solid #cbd5e1; "
                "border-radius: 8px; font-size: 18px; font-weight: 900; padding: 8px 12px;"
            )

    def _tick(self) -> None:
        if not self.training_running:
            return
        self.progress = min(100.0, self.progress + self.TRAINING_STEP)
        progress_value = int(round(self.progress))
        if self.training_progress.value() != progress_value:
            self.training_progress.setValue(progress_value)
        self._refresh_status()
        self.pipeline_updated.emit(3, float(self.progress))
        if self.progress >= 100.0:
            self._finish_training()

    def _finish_training(self) -> None:
        self.timer.stop()
        self.training_running = False
        self.training_complete = True
        self.progress = 100.0
        self.training_progress.setValue(100)
        self._refresh_status()
        self.lbl_model_state.setText("训练完成")
        self.lbl_gate_state.setText("已解锁")
        self._set_enter_button_kind("AccentButton")
        self.btn_enter_robot.setEnabled(True)
        self.btn_enter_robot.setText("进入机械臂控制")
        self.pipeline_updated.emit(3, 100.0)
        if not self._finished_emitted:
            self._finished_emitted = True
            self.training_finished.emit()

    def _set_enter_button_kind(self, object_name: str) -> None:
        if self.btn_enter_robot.objectName() == object_name:
            return
        self.btn_enter_robot.setObjectName(object_name)
        self.btn_enter_robot.style().unpolish(self.btn_enter_robot)
        self.btn_enter_robot.style().polish(self.btn_enter_robot)


class RobotFlowStageWidget(QWidget):
    pipeline_updated = pyqtSignal(int, float)
    MI_RECOGNITION_MS = 20_000
    FLOW_STEPS = (
        ("IDLE", "待机"),
        ("MI_MOVE_1", "1 MI移动"),
        ("DECIDE_1", "1 SSVEP确认"),
        ("SSVEP_TARGET_SELECT", "2 目标选择"),
        ("GRASP_CONFIRM", "2 抓取确认"),
        ("PICKING", "抓取执行"),
        ("MI_MOVE_2", "3 带载移动"),
        ("DECIDE_2", "3 放置确认"),
        ("PLACING", "放置执行"),
        ("TASK_DONE", "完成"),
    )

    def __init__(self, backend: RobotCommandBackend, config: WorkbenchConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.backend = backend
        self.config = config
        target_count = _bounded_target_count(self.config.target_count)
        self.config.target_count = target_count
        self.config.vision_max_targets = max(target_count, _bounded_target_count(self.config.vision_max_targets))
        self.config.vision_max_det = max(int(self.config.vision_max_targets), _bounded_target_count(self.config.vision_max_det))
        self.target_ids = _target_ids(target_count)
        self.control_phase = "IDLE"
        self.phase_remaining_ms = 0
        self.selected_target = "2" if "2" in self.target_ids else self.target_ids[0]
        self.manual_drive_enabled = False
        self.pressed_move_keys: set[int] = set()
        self.camera_stream_url = str(config.camera_stream_url or build_hiwonder_camera_stream_url(config.robot_host))
        self.vision_runtime: object | None = None
        self.vision_targets_by_id: dict[str, VisionTarget] = {}
        self.latest_vision_packet: dict[str, object] | None = None
        self.latest_vision_status = "视觉识别: 未启动"
        self._pending_grab_after_align_target: str | None = None
        self._pending_grab_after_align_mode = ""
        self._align_for_grab_in_progress = False
        self._awaiting_vision_refresh_after_align = False
        self._camera_center_stable_frames = 0
        self._camera_center_attempts = 0
        self._robot_density_key = ""
        self._flow_step_style_state: dict[str, str] = {}
        self._flow_label_font_px = 13
        self._flow_label_min_height = 30
        self._last_camera_status_chip: tuple[str, str] = ("", "")
        self._last_vision_status_chip: tuple[str, str] = ("", "")
        self.vision_app_config = self._build_vision_app_config()
        self._build_ui()
        self.flow_timer = QTimer(self)
        self.flow_timer.setInterval(80)
        self.flow_timer.timeout.connect(self._flow_tick)
        self.flash_timer = QTimer(self)
        self.flash_timer.setInterval(90)
        self.flash_timer.timeout.connect(self._flash_tick)
        self.flash_state = False
        self.backend.status_changed.connect(self._log)
        self.backend.connection_changed.connect(self._on_connection_changed)
        self.backend.pose_changed.connect(self._on_pose_changed)
        self.backend.command_finished.connect(self._on_command_finished)
        if self.config.connect_on_start:
            QTimer.singleShot(0, self.backend.connect_robot)

    def _build_vision_app_config(self) -> AppConfig:
        return AppConfig(
            robot_mode=str(self.config.robot_mode),
            robot_transport=str(self.config.robot_transport),
            robot_host=str(self.config.robot_host),
            robot_port=int(self.config.robot_port),
            rosbridge_port=int(self.config.rosbridge_port),
            control_sim_enabled=False,
            vision_mode="robot_camera_detection",
            vision_auto_start=True,
            vision_stream_url=str(self.config.camera_stream_url or ""),
            vision_weights_path=Path(self.config.vision_weights_path),
            vision_model_imgsz=int(self.config.vision_model_imgsz),
            vision_confidence_threshold=float(self.config.vision_confidence_threshold),
            vision_max_targets=_bounded_target_count(self.config.vision_max_targets),
            vision_max_det=_bounded_target_count(self.config.vision_max_det),
            vision_mapping_mode=str(self.config.vision_mapping_mode),
            vision_calibration_profile_path=Path(self.config.vision_calibration_profile_path),
            vision_calibration_profile_required=bool(self.config.vision_calibration_profile_required),
            vision_action_requires_calibration=True,
        ).resolved()

    def _responsive_side_width(self) -> int:
        width = max(1, self.width())
        if width >= 2200:
            return 420
        if width >= 1800:
            return 360
        return 284

    def _robot_density(self) -> tuple[str, float]:
        width = max(1, self.width())
        if width >= 2200:
            return "xl", 1.24
        if width >= 1800:
            return "large", 1.14
        return "base", 1.0

    def _responsive_toolbar_height(self) -> int:
        density, _ = self._robot_density()
        if density == "xl":
            return 76
        if density == "large":
            return 68
        return 54

    def _build_ui(self) -> None:
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.left_panel = QFrame()
        left_panel = self.left_panel
        left_panel.setObjectName("SideRail")
        left_panel.setFixedWidth(self._responsive_side_width())
        left_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(9, 8, 9, 8)
        left.setSpacing(7)
        left.setAlignment(Qt.AlignTop)
        rail_title = QLabel("机械臂控制")
        rail_title.setObjectName("DeviceTitle")
        rail_title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        left.addWidget(rail_title)
        robot_group = QGroupBox("机械臂连接")
        robot_layout = QVBoxLayout(robot_group)
        self.lbl_backend = QLabel(
            f"模式: {self.config.robot_mode} / {self.config.robot_transport}    "
            f"主机: {self.config.robot_host}:{self.config.rosbridge_port if self.config.robot_transport == 'ros' else self.config.robot_port}"
        )
        self.lbl_backend.setWordWrap(True)
        self.lbl_conn = QLabel("状态: 未连接")
        self.lbl_conn.setStyleSheet("font-weight: 700; color: #991b1b;")
        buttons = QHBoxLayout()
        self.btn_connect = QPushButton("连接机器人")
        self.btn_connect.setObjectName("AccentButton")
        self.btn_disconnect = QPushButton("断开")
        self.btn_disconnect.setObjectName("ActionButton")
        buttons.addWidget(self.btn_connect)
        buttons.addWidget(self.btn_disconnect)
        robot_layout.addWidget(self.lbl_backend)
        robot_layout.addWidget(self.lbl_conn)
        robot_layout.addLayout(buttons)
        self.btn_connect.clicked.connect(self.backend.connect_robot)
        self.btn_disconnect.clicked.connect(self.backend.close_backend)
        left.addWidget(robot_group)

        flow_group = QGroupBox("混合脑控流程总控")
        flow_layout = QVBoxLayout(flow_group)
        self.lbl_phase_headline = QLabel("Idle")
        self.lbl_phase_headline.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lbl_phase_headline.setMinimumHeight(34)
        self.lbl_run_status = QLabel("系统就绪：训练完成后点击“开始完整控制流程”。")
        self.lbl_run_status.setWordWrap(True)
        self.lbl_run_status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lbl_run_status.setMinimumHeight(44)
        self.lbl_run_status.setStyleSheet("color: #0369a1; font-size: 14px; font-weight: 800;")
        self.pbar_countdown = QProgressBar()
        self.pbar_countdown.setRange(0, self.MI_RECOGNITION_MS)
        self.pbar_countdown.setValue(0)
        self.btn_master_start = QPushButton("开始完整控制流程")
        self.btn_master_start.setObjectName("AccentButton")
        self.btn_manual_drive = QPushButton("MI 移动控制")
        self.btn_manual_drive.setObjectName("ActionButton")
        self.btn_primary = QPushButton("确认 / 执行")
        self.btn_primary.setObjectName("ActionButton")
        self.btn_secondary = QPushButton("继续 / 取消")
        self.btn_secondary.setObjectName("ActionButton")
        self.btn_reset = QPushButton("复位")
        self.btn_reset.setObjectName("ActionButton")
        self.btn_stop = QPushButton("安全停止")
        self.btn_stop.setObjectName("DangerButton")
        flow_layout.addWidget(self.lbl_phase_headline)
        flow_layout.addWidget(self.lbl_run_status)
        flow_layout.addWidget(self.pbar_countdown)
        flow_layout.addWidget(self.btn_master_start)
        action_grid = QGridLayout()
        action_grid.setHorizontalSpacing(6)
        action_grid.setVerticalSpacing(6)
        action_grid.addWidget(self.btn_manual_drive, 0, 0)
        action_grid.addWidget(self.btn_primary, 0, 1)
        action_grid.addWidget(self.btn_secondary, 1, 0)
        action_grid.addWidget(self.btn_stop, 1, 1)
        action_grid.addWidget(self.btn_reset, 2, 0, 1, 2)
        flow_layout.addLayout(action_grid)
        self.flow_step_labels: dict[str, QLabel] = {}
        flow_line = QGridLayout()
        flow_line.setHorizontalSpacing(4)
        flow_line.setVerticalSpacing(4)
        for index, (phase, title) in enumerate(self.FLOW_STEPS):
            label = QLabel(title)
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumHeight(30)
            label.setStyleSheet("font-size: 13px; font-weight: 800; padding: 4px 5px; border-radius: 4px;")
            flow_line.addWidget(label, index // 2, index % 2)
            self.flow_step_labels[phase] = label
        flow_layout.addLayout(flow_line)
        self.btn_master_start.clicked.connect(self._start_integrated_flow)
        self.btn_manual_drive.clicked.connect(self._toggle_manual_drive)
        self.btn_primary.clicked.connect(self._primary_decision)
        self.btn_secondary.clicked.connect(self._secondary_decision)
        self.btn_reset.clicked.connect(self._reset_robot_error)
        self.btn_stop.clicked.connect(self._stop_current_task_safely)
        left.addWidget(flow_group)

        target_group = QGroupBox("目标木块选择")
        target_layout = QGridLayout(target_group)
        target_layout.setSpacing(6)
        self.target_buttons: dict[str, QPushButton] = {}
        for index, target_id in enumerate(self.target_ids):
            button = QPushButton(f"目标 {target_id}")
            button.setMinimumHeight(36)
            button.setProperty("blockState", "active" if target_id == self.selected_target else "pending")
            button.clicked.connect(lambda checked=False, value=target_id: self._select_target(value))
            target_layout.addWidget(button, index // 2, index % 2)
            self.target_buttons[target_id] = button
        left.addWidget(target_group)

        pose_group = QGroupBox("末端位姿")
        pose_layout = QVBoxLayout(pose_group)
        self.lbl_pose = QLabel("theta=0.0 deg, radius=150.0 mm, z=160.0 mm")
        self.lbl_grip = QLabel("吸嘴状态: STANDBY / OPEN")
        pose_layout.addWidget(self.lbl_pose)
        pose_layout.addWidget(self.lbl_grip)
        left.addWidget(pose_group)

        self.log = QPlainTextEdit()
        self.log.setObjectName("LogView")
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(120)
        self.log.setMaximumHeight(88)
        left.addWidget(self.log)
        layout.addWidget(left_panel, 0)

        self.camera_card = QFrame()
        camera_card = self.camera_card
        camera_card.setObjectName("CameraShell")
        camera_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        camera_layout = QVBoxLayout(camera_card)
        camera_layout.setContentsMargins(6, 6, 6, 6)
        camera_layout.setSpacing(6)
        camera_toolbar = QFrame()
        self.camera_toolbar = camera_toolbar
        camera_toolbar.setObjectName("ViewportToolbar")
        camera_toolbar.setFixedHeight(self._responsive_toolbar_height())
        camera_header = QHBoxLayout(camera_toolbar)
        camera_header.setContentsMargins(10, 4, 8, 4)
        camera_header.setSpacing(8)
        camera_title = QLabel("实时机械臂摄像头")
        camera_title.setObjectName("DeviceTitle")
        self.lbl_camera_status = QLabel("camera idle")
        self.lbl_camera_status.setObjectName("StatusChip")
        self.lbl_camera_status.setProperty("state", "pending")
        self.lbl_vision_status = QLabel("视觉: 未启动")
        self.lbl_vision_status.setObjectName("StatusChip")
        self.lbl_vision_status.setProperty("state", "pending")
        self.btn_camera_start = QPushButton("启动")
        self.btn_camera_start.setObjectName("ActionButton")
        self.btn_camera_start.setMinimumWidth(72)
        self.btn_camera_stop = QPushButton("停止")
        self.btn_camera_stop.setObjectName("ActionButton")
        self.btn_camera_stop.setMinimumWidth(72)
        self.btn_vision_start = QPushButton("启动识别")
        self.btn_vision_start.setObjectName("AccentButton")
        self.btn_vision_start.setMinimumWidth(92)
        self.btn_vision_stop = QPushButton("停止识别")
        self.btn_vision_stop.setObjectName("ActionButton")
        self.btn_vision_stop.setMinimumWidth(92)
        camera_header.addWidget(camera_title)
        camera_header.addWidget(self.lbl_camera_status, 1)
        camera_header.addWidget(self.lbl_vision_status)
        camera_header.addWidget(self.btn_camera_start)
        camera_header.addWidget(self.btn_camera_stop)
        camera_header.addWidget(self.btn_vision_start)
        camera_header.addWidget(self.btn_vision_stop)
        camera_layout.addWidget(camera_toolbar)
        self.video_container = QWidget()
        self.video_container.setMinimumSize(760, 460)
        self.video_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_container.setStyleSheet("background: #0f172a;")
        self.video_container.installEventFilter(self)
        self.cam = RobotCameraDisplayWidget(self.video_container)
        self.cam.target_ids = self.target_ids
        self.cam.status_changed.connect(self._on_camera_status)
        self.btn_camera_start.clicked.connect(self.start_camera)
        self.btn_camera_stop.clicked.connect(self.stop_camera)
        self.btn_vision_start.clicked.connect(self.start_vision)
        self.btn_vision_stop.clicked.connect(self.stop_vision)
        self.popup_dialog = QFrame(self.video_container)
        self.popup_dialog.setStyleSheet(
            "QFrame { background: rgba(15, 23, 42, 230); border: 2px solid #0284c7; border-radius: 10px; }"
            "QLabel { color: #ffffff; font-weight: 800; }"
        )
        popup_layout = QVBoxLayout(self.popup_dialog)
        self.lbl_popup_title = QLabel("BCI 决策窗口")
        self.lbl_popup_title.setAlignment(Qt.AlignCenter)
        self.lbl_popup_desc = QLabel("等待脑控确认信号")
        self.lbl_popup_desc.setAlignment(Qt.AlignCenter)
        popup_layout.addWidget(self.lbl_popup_title)
        popup_layout.addWidget(self.lbl_popup_desc)
        self.popup_dialog.hide()
        self.flash_box_confirm = QLabel("", self.video_container)
        self.flash_box_continue = QLabel("", self.video_container)
        for label in (self.flash_box_confirm, self.flash_box_continue):
            label.setAlignment(Qt.AlignCenter)
            label.setWordWrap(True)
            label.hide()
        camera_layout.addWidget(self.video_container, 1)
        layout.addWidget(camera_card, 1)
        self._log("完整控制流程：MI 先识别 20 秒，随后相机画面中的木块进入 SSVEP 闪烁；识别到目标后进入抓取确认。")
        self._log(f"camera stream url: {self.camera_stream_url}")
        self._set_vision_status_chip(self.latest_vision_status, "pending")
        self._apply_robot_density(force=True)
        self._refresh_display_state()

    def _apply_robot_density(self, *, force: bool = False) -> None:
        density, scale = self._robot_density()
        if not force and density == self._robot_density_key:
            return
        self._robot_density_key = density
        phase_px = int(round(21 * scale))
        status_px = int(round(14 * scale))
        button_h = int(round(40 * scale))
        target_h = int(round(36 * scale))
        self._flow_label_font_px = int(round(13 * scale))
        self._flow_label_min_height = int(round(30 * scale))
        self.lbl_phase_headline.setStyleSheet(f"color: #0f172a; font-size: {phase_px}px; font-weight: 900;")
        self.lbl_run_status.setStyleSheet(f"color: #0369a1; font-size: {status_px}px; font-weight: 800;")
        self.lbl_run_status.setMinimumHeight(int(round(44 * scale)))
        for button in (
            self.btn_master_start,
            self.btn_manual_drive,
            self.btn_primary,
            self.btn_secondary,
            self.btn_reset,
            self.btn_stop,
            self.btn_camera_start,
            self.btn_camera_stop,
            self.btn_vision_start,
            self.btn_vision_stop,
        ):
            button.setMinimumHeight(button_h)
        for button in self.target_buttons.values():
            button.setMinimumHeight(target_h)
        self.log.setMaximumHeight(int(round(88 * min(1.18, scale))))
        if hasattr(self, "camera_toolbar"):
            self.camera_toolbar.setFixedHeight(self._responsive_toolbar_height())
        self._flow_step_style_state.clear()
        self._refresh_flow_steps()

    def eventFilter(self, source: object, event: QEvent) -> bool:  # noqa: N802
        if source is getattr(self, "video_container", None) and event.type() in {QEvent.Resize, QEvent.Show}:
            self._resize_camera_overlays()
        return super().eventFilter(source, event)

    def _set_camera_status_chip(self, text: str, state: str) -> None:
        current = (str(text), str(state))
        if current == self._last_camera_status_chip:
            return
        self._last_camera_status_chip = current
        self.lbl_camera_status.setText(text)
        self.lbl_camera_status.setProperty("state", state)
        self.lbl_camera_status.style().unpolish(self.lbl_camera_status)
        self.lbl_camera_status.style().polish(self.lbl_camera_status)

    def _set_vision_status_chip(self, text: str, state: str) -> None:
        current = (str(text), str(state))
        if current == self._last_vision_status_chip:
            return
        self._last_vision_status_chip = current
        self.lbl_vision_status.setText(text)
        self.lbl_vision_status.setProperty("state", state)
        self.lbl_vision_status.style().unpolish(self.lbl_vision_status)
        self.lbl_vision_status.style().polish(self.lbl_vision_status)

    def start_camera(self) -> None:
        if bool(self.config.vision_enabled) and bool(self.config.vision_auto_start):
            if self.start_vision():
                return
        if self.config.robot_mode == "fake" and not self.config.camera_stream_url:
            self._on_camera_status("camera disabled in fake mode")
            return
        self.cam.start_stream(self.camera_stream_url)

    def stop_camera(self) -> None:
        self.stop_vision()
        self.cam.stop_stream()

    def start_vision(self) -> bool:
        self.config.vision_enabled = True
        if self.vision_runtime is not None:
            return True
        self.cam.stop_stream()
        self.vision_app_config = self._build_vision_app_config()
        try:
            from hybrid_controller.vision.runtime import VisionRuntime
        except Exception as error:
            self._on_vision_status(f"视觉识别缺少依赖: {error}")
            return False
        runtime = VisionRuntime(
            self.vision_app_config,
            calibration_params=None,
            targets_callback=self._on_vision_targets,
            packet_callback=self._on_vision_packet,
            frame_callback=self._on_vision_frame,
            status_callback=self._on_vision_status,
        )
        self.vision_runtime = runtime
        runtime.start()
        if getattr(runtime, "worker", None) is None:
            self.vision_runtime = None
            return False
        self._on_camera_status("camera stream handled by vision runtime")
        self._set_vision_status_chip("视觉: 启动中", "active")
        return True

    def stop_vision(self) -> None:
        runtime = self.vision_runtime
        self.vision_runtime = None
        if runtime is not None:
            try:
                runtime.stop()
            except Exception as error:
                self._log(f"vision stop failed: {error}")
        self.vision_targets_by_id.clear()
        self.latest_vision_packet = None
        self.latest_vision_status = "视觉识别: 未启动"
        if hasattr(self, "lbl_vision_status"):
            self._set_vision_status_chip("视觉: 未启动", "pending")
        if hasattr(self, "cam"):
            self.cam.set_vision_payload(targets=[], packet=None, status_text=self.latest_vision_status)

    def _on_vision_status(self, status: str) -> None:
        text = str(status)
        self.latest_vision_status = f"视觉识别: {text}"
        lower = text.lower()
        if "missing" in lower or "缺少" in text or "error" in lower or "failed" in lower:
            state = "danger"
        elif "started" in lower or "connected" in lower or "目标" in text:
            state = "ready"
        else:
            state = "active"
        self._set_vision_status_chip(("视觉: " + text)[:64], state)
        self.cam.set_vision_payload(
            targets=list(self.vision_targets_by_id.values()),
            packet=self.latest_vision_packet,
            status_text=self.latest_vision_status,
        )
        self._log(f"vision: {text}")

    def _on_vision_frame(self, frame: object) -> None:
        image = _qimage_from_bgr_frame(frame)
        if image is not None:
            self.cam.set_frame(image)

    def _on_vision_targets(self, targets: object) -> None:
        del targets

    def _snapshot_age_ms(self, snapshot: dict[str, object] | None) -> float:
        if not isinstance(snapshot, dict):
            return float("inf")
        try:
            robot_ts = float(snapshot.get("robot_ts", 0.0) or 0.0)
        except (TypeError, ValueError):
            return float("inf")
        if robot_ts <= 0.0:
            return float("inf")
        return max(0.0, (time.time() - robot_ts) * 1000.0)

    def _resolve_vision_packet(self, packet: dict[str, object]) -> dict[str, object]:
        try:
            from hybrid_controller.vision.target_resolver import resolve_vision_packet
        except Exception as error:
            self._on_vision_status(f"视觉解析不可用: {error}")
            return dict(packet)
        snapshot = self.backend.latest_state_snapshot()
        try:
            resolution = resolve_vision_packet(
                packet,
                config=self.vision_app_config,
                snapshot=snapshot,
                snapshot_age_ms=self._snapshot_age_ms(snapshot),
                frame_pose_age_ms=None,
            )
            return dict(resolution.packet)
        except Exception as error:
            self._on_vision_status(f"视觉坐标解析失败: {error}")
            return dict(packet)

    def _on_vision_packet(self, packet: object) -> None:
        if not isinstance(packet, dict):
            return
        resolved_packet = self._resolve_vision_packet(dict(packet))
        targets = _vision_targets_from_packet(resolved_packet)
        self.latest_vision_packet = resolved_packet
        self._update_vision_targets(targets)

    def _update_vision_targets(self, targets: Sequence[VisionTarget]) -> None:
        self.vision_targets_by_id = {
            str(target.slot_id if target.slot_id is not None else target.id): target for target in targets
        }
        actionable = sum(1 for target in targets if target.actionable)
        servo_required = sum(1 for target in targets if target.servo_required)
        invalid_reason = "--"
        if self.latest_vision_packet is not None:
            for slot in self.latest_vision_packet.get("slots", []):
                if isinstance(slot, dict) and str(slot.get("invalid_reason", "")).strip():
                    invalid_reason = str(slot.get("invalid_reason"))
                    break
        if targets:
            status = f"视觉识别: {len(targets)} 个目标，{actionable} 个可抓取"
            if servo_required:
                status += f"，{servo_required} 个需先对准"
            if invalid_reason != "--" and actionable == 0:
                status += f"，{invalid_reason}"
            chip_text = f"视觉: {len(targets)} 目标 / {actionable} 可抓"
            state = "ready" if actionable else "active"
        else:
            status = "视觉识别: 未检测到小木块"
            chip_text = "视觉: 无目标"
            state = "pending"
        self.latest_vision_status = status
        self._set_vision_status_chip(chip_text, state)
        self.cam.set_vision_payload(targets=targets, packet=self.latest_vision_packet, status_text=status)
        if self._pending_grab_after_align_target and self._awaiting_vision_refresh_after_align:
            self._awaiting_vision_refresh_after_align = False
        if self._maybe_auto_grab_after_align():
            return
        self._refresh_display_state()

    def _on_camera_status(self, status: str) -> None:
        text = str(status)
        if "connected" in text:
            state = "ready"
        elif "error" in text or "timeout" in text or "failed" in text:
            state = "danger"
        elif "opening" in text or "starting" in text:
            state = "active"
        else:
            state = "pending"
        self._set_camera_status_chip(text, state)
        self._log(f"camera: {text}")
        self._refresh_display_state()

    def _display_phase_text(self) -> tuple[str, str]:
        phase_map = {
            "IDLE": ("Idle", "等待启动完整任务"),
            "MANUAL_DRIVE": ("MI 移动控制", "连续运动意图控制机械臂，释放意图后停止"),
            "MI_RECOGNITION": ("MI 识别中", "20 秒运动想象识别窗口"),
            "MI_MOVE_1": ("Stage 1 Motion Adjustment", "MI 阶段移动调整，连续运动意图控制机械臂"),
            "DECIDE_1": ("Stage 1 Decision", "SSVEP 确认：进入目标选择或继续移动"),
            "SSVEP_TARGET_SELECT": ("Stage 2 Target Selection", "四目标 SSVEP 闪烁；识别目标后锁定木块"),
            "GRASP_CONFIRM": ("Stage 2 Grasp Confirmation", "目标已锁定，等待抓取确认信号"),
            "PICKING": ("抓取执行中", "等待机械臂返回抓取结果"),
            "TARGET_LOCKED": ("目标已锁定", "SSVEP 信号已收到，闪烁停止，准备抓取"),
            "MI_MOVE_2": ("Stage 3 Carry Adjustment", "带载移动，连续运动意图控制机械臂"),
            "DECIDE_2": ("Stage 3 Placement Decision", "SSVEP 确认：放置或继续带载移动"),
            "PLACING": ("放置执行中", "等待机械臂返回放置结果"),
            "TASK_DONE": ("Finished", "移动、选择、抓取、带载移动、放置闭环完成"),
        }
        return phase_map.get(self.control_phase, (self.control_phase, ""))

    def _refresh_display_state(self) -> None:
        phase_title, phase_detail = self._display_phase_text()
        if self.lbl_phase_headline.text() != phase_title:
            self.lbl_phase_headline.setText(phase_title)
        countdown = ""
        if self.control_phase in {"MI_RECOGNITION", "MI_MOVE_1", "MI_MOVE_2"} and self.phase_remaining_ms > 0:
            countdown = f"{self.phase_remaining_ms / 1000.0:.1f}s"
        if self.control_phase == "SSVEP_TARGET_SELECT" and self.cam.ssv_flicker_enabled:
            active_target = ""
        else:
            active_target = "" if self.control_phase in {"IDLE", "MANUAL_DRIVE", "TASK_DONE"} else self.selected_target
        self.cam.set_operation_state(
            phase_title=phase_title,
            phase_detail=phase_detail,
            countdown_text=countdown,
            active_id=active_target,
            pose_text=self.lbl_pose.text(),
            grip_text=self.lbl_grip.text(),
        )
        self._refresh_action_controls()
        self._refresh_flow_steps()

    def _refresh_action_controls(self) -> None:
        if not hasattr(self, "btn_primary"):
            return
        if self.control_phase == "DECIDE_1":
            primary, secondary = "进入目标选择", "继续移动"
        elif self.control_phase == "SSVEP_TARGET_SELECT":
            primary, secondary = f"锁定目标 {self.selected_target}", "继续移动"
        elif self.control_phase == "GRASP_CONFIRM":
            primary, secondary = f"确认抓取目标 {self.selected_target}", "重新选择目标"
        elif self.control_phase == "DECIDE_2":
            primary, secondary = "确认放置", "继续搬运"
        elif self.control_phase == "PICKING":
            primary, secondary = "抓取执行中", "等待结果"
        elif self.control_phase == "PLACING":
            primary, secondary = "放置执行中", "等待结果"
        elif self.control_phase == "TASK_DONE":
            primary, secondary = "再次开始", "完成"
        else:
            primary, secondary = "确认 / 执行", "继续 / 取消"
        if self.btn_primary.text() != primary:
            self.btn_primary.setText(primary)
        if self.btn_secondary.text() != secondary:
            self.btn_secondary.setText(secondary)

    def _refresh_flow_steps(self) -> None:
        if not hasattr(self, "flow_step_labels"):
            return
        active_phase = self.control_phase
        aliases = {"MI_RECOGNITION": "MI_MOVE_1", "TARGET_LOCKED": "GRASP_CONFIRM"}
        active_key = aliases.get(active_phase, active_phase)
        font_px = max(13, int(self._flow_label_font_px))
        min_h = max(30, int(self._flow_label_min_height))
        active_style = (
            "color: #ffffff; background: #0f766e; border: 1px solid #14b8a6; "
            f"font-size: {font_px}px; font-weight: 900; padding: 5px 6px; border-radius: 4px;"
        )
        done_style = (
            "color: #ffffff; background: #047857; border: 1px solid #10b981; "
            f"font-size: {font_px}px; font-weight: 900; padding: 5px 6px; border-radius: 4px;"
        )
        idle_style = (
            "color: #334155; background: #f8fafc; border: 1px solid #cbd5e1; "
            f"font-size: {font_px}px; font-weight: 800; padding: 5px 6px; border-radius: 4px;"
        )
        for phase, label in self.flow_step_labels.items():
            if phase == active_key:
                state_key = f"active:{font_px}"
                style = active_style
            elif phase == "TASK_DONE" and self.control_phase == "TASK_DONE":
                state_key = f"done:{font_px}"
                style = done_style
            else:
                state_key = f"idle:{font_px}"
                style = idle_style
            if self._flow_step_style_state.get(phase) == state_key:
                continue
            self._flow_step_style_state[phase] = state_key
            label.setMinimumHeight(min_h)
            label.setStyleSheet(style)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "left_panel"):
            target_width = self._responsive_side_width()
            if self.left_panel.width() != target_width:
                self.left_panel.setFixedWidth(target_width)
        self._apply_robot_density()
        self._resize_camera_overlays()

    def _resize_camera_overlays(self) -> None:
        if not hasattr(self, "video_container"):
            return
        w, h = self.video_container.width(), self.video_container.height()
        if w <= 0 or h <= 0:
            return
        self.cam.setGeometry(0, 0, w, h)
        popup_w = min(560, max(360, int(w * 0.30)))
        popup_h = min(184, max(124, int(h * 0.16)))
        self.popup_dialog.setGeometry(max(10, int((w - popup_w) / 2)), max(10, int((h - popup_h) / 2)), popup_w, popup_h)
        popup_title_px = max(18, min(30, int(popup_h * 0.20)))
        popup_body_px = max(14, min(22, int(popup_h * 0.15)))
        self.lbl_popup_title.setStyleSheet(f"font-size: {popup_title_px}px; font-weight: 900;")
        self.lbl_popup_desc.setStyleSheet(f"font-size: {popup_body_px}px; font-weight: 800;")
        box_w = min(320, max(170, int(w * 0.16)))
        box_h = min(300, max(170, int(h * 0.34)))
        self.flash_box_confirm.setGeometry(16, int((h - box_h) / 2), box_w, box_h)
        self.flash_box_continue.setGeometry(w - box_w - 16, int((h - box_h) / 2), box_w, box_h)

    def _start_integrated_flow(self) -> None:
        if self.manual_drive_enabled:
            self._exit_manual_drive()
        if not self.backend.connected:
            self.lbl_run_status.setText("请先连接机械臂，再开始完整控制流程。")
            self._log("完整控制流程未启动：机械臂未连接。")
            self._refresh_display_state()
            return
        self.btn_master_start.setEnabled(False)
        self.btn_master_start.setText("流程进行中")
        self._enter_mi_move_stage(1)
        self.flow_timer.start()

    def _toggle_manual_drive(self) -> None:
        if self.manual_drive_enabled:
            self._exit_manual_drive()
            return
        self._enter_manual_drive()

    def _robot_error_snapshot(self) -> dict[str, object] | None:
        snapshot = self.backend.latest_state_snapshot()
        if not isinstance(snapshot, dict):
            return None
        if str(snapshot.get("state") or "").upper() == "ERROR":
            return snapshot
        return None

    def _reset_robot_error(self) -> None:
        self.backend.stop_teleop()
        self.backend.reset()
        self.lbl_run_status.setText("正在复位机械臂控制状态；复位完成后可继续进入 MI 移动控制。")
        self._log("reset requested.")
        self._refresh_display_state()

    def _enter_manual_drive(self) -> None:
        if not self.backend.connected:
            self.lbl_run_status.setText("请先连接机械臂，再进入 MI 移动控制。")
            self._log("MI 移动控制未启动：机械臂未连接。")
            self._refresh_display_state()
            return
        error_snapshot = self._robot_error_snapshot()
        if error_snapshot is not None:
            code = str(error_snapshot.get("last_error_code") or "").strip()
            message = str(error_snapshot.get("last_error") or "").strip()
            self.lbl_run_status.setText(f"机械臂处于 ERROR 状态，已先发送复位。{code} {message}".strip())
            self._log(f"manual drive blocked by ERROR; reset first: {code} {message}".strip())
            self._reset_robot_error()
            return
        self.flow_timer.stop()
        self.phase_remaining_ms = 0
        self.pressed_move_keys.clear()
        self._hide_decision_widgets()
        self.cam.set_ssvep_flicker(False)
        self.manual_drive_enabled = True
        self.control_phase = "MANUAL_DRIVE"
        self.btn_master_start.setEnabled(False)
        self.btn_manual_drive.setText("停止移动")
        self.pbar_countdown.setRange(0, 1)
        self.pbar_countdown.setValue(0)
        self.pbar_countdown.setFormat("MI 移动控制")
        self.lbl_run_status.setText("MI 移动控制：连续运动意图控制机械臂，释放意图后停止；安全停止可立即中断。")
        self.pipeline_updated.emit(4, 10.0)
        self._log("MI 移动控制已开启。")
        self._refresh_display_state()
        self.flow_timer.start()

    def _exit_manual_drive(self) -> None:
        if self.manual_drive_enabled or self.control_phase == "MANUAL_DRIVE":
            self.backend.stop_teleop()
            self.flow_timer.stop()
        self.manual_drive_enabled = False
        self.pressed_move_keys.clear()
        if self.control_phase == "MANUAL_DRIVE":
            self.control_phase = "IDLE"
            self.phase_remaining_ms = 0
        self.btn_manual_drive.setText("MI 移动控制")
        self.btn_master_start.setEnabled(True)
        self.pbar_countdown.setRange(0, self.MI_RECOGNITION_MS)
        self.pbar_countdown.setValue(0)
        self.pbar_countdown.setFormat("")
        self.lbl_run_status.setText("MI 移动控制已停止。可以重新启动任务，或再次进入 MI 移动控制。")
        self._log("MI 移动控制已停止。")
        self._refresh_display_state()

    def _enter_mi_recognition_stage(self) -> None:
        self.control_phase = "MI_RECOGNITION"
        self.phase_remaining_ms = self.MI_RECOGNITION_MS
        self.pressed_move_keys.clear()
        self.backend.stop_teleop()
        self.cam.set_ssvep_flicker(False)
        self._hide_decision_widgets()
        self.pbar_countdown.setRange(0, self.MI_RECOGNITION_MS)
        self.pbar_countdown.setValue(self.MI_RECOGNITION_MS)
        self.pbar_countdown.setFormat("MI 识别剩余 20.0s")
        self.lbl_run_status.setText("MI 阶段：运动想象识别窗口 20 秒，等待运动意图输出。")
        self.pipeline_updated.emit(4, 20.0)
        self._log("完整流程启动：进入 MI 20 秒识别阶段。")
        self._refresh_display_state()

    def _enter_mi_move_stage(self, stage: int) -> None:
        self.control_phase = f"MI_MOVE_{stage}"
        self.phase_remaining_ms = self.MI_RECOGNITION_MS
        self.pbar_countdown.setRange(0, self.MI_RECOGNITION_MS)
        self.pbar_countdown.setValue(self.MI_RECOGNITION_MS)
        self.pbar_countdown.setFormat("MI阶段剩余 20.0s")
        self._hide_decision_widgets()
        self.cam.set_ssvep_flicker(False)
        text = (
            "Stage 1 Motion Adjustment：MI 移动调整 20 秒，连续运动意图控制机械臂。"
            if stage == 1
            else "Stage 3 Carry Adjustment：带载移动 20 秒，连续运动意图控制机械臂。"
        )
        self.lbl_run_status.setText(text)
        self.pipeline_updated.emit(4, 20.0 if stage == 1 else 65.0)
        self._log(text)
        self._refresh_display_state()

    def _flow_tick(self) -> None:
        if self.control_phase == "MANUAL_DRIVE":
            self._pump_keyboard_teleop()
            self._refresh_display_state()
            return
        if self.control_phase in {"MI_MOVE_1", "MI_MOVE_2"}:
            self.phase_remaining_ms = max(0, self.phase_remaining_ms - self.flow_timer.interval())
            self.pbar_countdown.setValue(self.phase_remaining_ms)
            self.pbar_countdown.setFormat(f"MI阶段剩余 {self.phase_remaining_ms / 1000.0:.1f}s")
            self._pump_keyboard_teleop()
            self._refresh_display_state()
            if self.phase_remaining_ms <= 0:
                if self.control_phase == "MI_MOVE_1":
                    self._enter_decision_stage_1()
                else:
                    self._enter_decision_stage_2()

    def _enter_decision_stage_1(self) -> None:
        self.backend.stop_teleop()
        self.control_phase = "DECIDE_1"
        self.lbl_run_status.setText("Stage 1 Decision：SSVEP 确认是否进入目标选择；确认进入目标选择，继续则返回移动。")
        self._show_decision_widgets(
            title="Stage 1 Decision",
            desc="确认: 进入目标选择    继续: 返回移动",
            confirm="进入目标选择\n确认",
            cont="继续移动\n继续",
        )
        self.pipeline_updated.emit(4, 38.0)
        self._refresh_display_state()

    def _enter_ssvep_grab_stage(self) -> None:
        self._enter_ssvep_target_selection()

    def _enter_ssvep_target_selection(self) -> None:
        self.control_phase = "SSVEP_TARGET_SELECT"
        self.phase_remaining_ms = 0
        self._clear_pending_grab_after_align()
        self.flow_timer.stop()
        self.backend.stop_teleop()
        self._hide_decision_widgets()
        self.pbar_countdown.setRange(0, 0)
        self.pbar_countdown.setFormat("等待 SSVEP 识别信号")
        if self.config.vision_enabled:
            self.lbl_run_status.setText(
                "Stage 2 Target Selection：画面中的小木块正在闪烁。识别到目标后停止闪烁，并进入抓取确认。"
            )
        else:
            self.lbl_run_status.setText("Stage 2 Target Selection：占位木块正在闪烁。识别到目标后进入抓取确认。")
        self.cam.set_ssvep_flicker(True)
        self.pipeline_updated.emit(4, 50.0)
        self._log("进入 Stage 2 Target Selection：SSVEP 四目标闪烁。")
        self._refresh_display_state()

    def _lock_ssvep_target_and_pick(self, target_id: str) -> None:
        self._lock_ssvep_target_and_confirm(target_id)

    def _lock_ssvep_target_and_confirm(self, target_id: str) -> None:
        normalized_target = str(target_id)
        if normalized_target not in self.target_ids:
            self._log(f"ignored target selection outside configured range: {normalized_target}")
            return
        self.selected_target = normalized_target
        for key, button in self.target_buttons.items():
            button.setProperty("blockState", "active" if key == self.selected_target else "pending")
            button.style().unpolish(button)
            button.style().polish(button)
        self.cam.set_ssvep_flicker(False, selected_id=self.selected_target)
        self.control_phase = "GRASP_CONFIRM"
        self._clear_pending_grab_after_align()
        self.lbl_run_status.setText(f"目标 {self.selected_target} 已锁定，闪烁已停止。等待抓取确认信号；也可重新选择目标。")
        self._show_decision_widgets(
            title="Stage 2 Grasp Confirmation",
            desc=f"目标 {self.selected_target} 已锁定    确认: 抓取    重新选择: 返回选择",
            confirm=f"抓取目标 {self.selected_target}\n确认",
            cont="重新选择\n返回",
        )
        self.pbar_countdown.setRange(0, 1)
        self.pbar_countdown.setValue(1)
        self.pbar_countdown.setFormat(f"目标 {self.selected_target} 已锁定")
        self.pipeline_updated.emit(4, 58.0)
        self._log(f"SSVEP 四分类结果：锁定目标 {self.selected_target}，进入抓取确认。")
        self._refresh_display_state()

    def _selected_vision_target(self) -> VisionTarget | None:
        return self.vision_targets_by_id.get(str(self.selected_target))

    def _clear_pending_grab_after_align(self) -> None:
        self._pending_grab_after_align_target = None
        self._pending_grab_after_align_mode = ""
        self._align_for_grab_in_progress = False
        self._awaiting_vision_refresh_after_align = False
        self._camera_center_stable_frames = 0
        self._camera_center_attempts = 0

    def _target_can_use_camera_center_grab(self, target: VisionTarget | None) -> bool:
        if target is None:
            return False
        if target.actionable:
            return False
        center = _vision_tracking_point(target)
        if center is None:
            return False
        reason = str(target.invalid_reason or "").strip().lower()
        return reason in {
            "",
            "calibration_profile_unavailable",
            "calibration_unavailable",
            "vision_servo_required",
        }

    def _request_grab_alignment(self, target: VisionTarget, *, mode: str) -> bool:
        self._pending_grab_after_align_target = str(self.selected_target)
        self._pending_grab_after_align_mode = str(mode)
        self._align_for_grab_in_progress = True
        self._awaiting_vision_refresh_after_align = False
        self._camera_center_attempts += 1
        if self._camera_center_attempts > max(1, int(self.config.vision_center_max_attempts)):
            self._clear_pending_grab_after_align()
            self.lbl_run_status.setText(f"目标 {self.selected_target} 对中次数过多，已停止自动抓取，请重新选择目标。")
            self._refresh_display_state()
            return False
        if self.backend.align_to_vision_target(self.selected_target, target):
            self.lbl_run_status.setText(
                f"目标 {self.selected_target} 正在进行视觉对中；稳定到画面中心后将自动抓取。"
            )
            self._log(f"vision align requested for target {self.selected_target} mode={mode}")
            self._refresh_display_state()
            return True
        self._clear_pending_grab_after_align()
        self.lbl_run_status.setText(f"目标 {self.selected_target} 无法生成视觉对中命令，请检查机械臂状态和画面目标。")
        self._refresh_display_state()
        return False

    def _maybe_auto_grab_after_align(self) -> bool:
        target_id = self._pending_grab_after_align_target
        if not target_id:
            return False
        if self.control_phase not in {"GRASP_CONFIRM", "TARGET_LOCKED"}:
            self._clear_pending_grab_after_align()
            return False
        if str(self.selected_target) != str(target_id):
            self._clear_pending_grab_after_align()
            return False
        if self._align_for_grab_in_progress:
            return False
        if self._awaiting_vision_refresh_after_align:
            return False
        target = self.vision_targets_by_id.get(str(target_id))
        if target is None:
            self.lbl_run_status.setText(f"目标 {target_id} 已完成中心对准，等待视觉重新锁定后自动抓取。")
            return False
        if self._pending_grab_after_align_mode == "camera_center":
            distance_px = _vision_center_distance_px(target, self.latest_vision_packet)
            if distance_px <= float(self.config.vision_center_tolerance_px):
                self._camera_center_stable_frames += 1
                required = max(1, int(self.config.vision_center_stable_frames))
                self.lbl_run_status.setText(
                    f"目标 {target_id} 已接近画面中心 {distance_px:.1f}px，稳定 {self._camera_center_stable_frames}/{required}。"
                )
                if self._camera_center_stable_frames >= required:
                    self._clear_pending_grab_after_align()
                    self.control_phase = "PICKING"
                    self._hide_decision_widgets()
                    self.lbl_run_status.setText(f"目标 {target_id} 已稳定对中，正在按当前中心执行抓取。")
                    self.lbl_grip.setText("吸嘴状态: BUSY / CAMERA CENTER PICK")
                    self.lbl_grip.setStyleSheet("color: #b45309; font-weight: 800;")
                    self.backend.pick_camera_center_target(target_id, target)
                    self.pipeline_updated.emit(4, 58.0)
                    self._refresh_display_state()
                    return True
                return False
            self._camera_center_stable_frames = 0
            if self._request_grab_alignment(target, mode="camera_center"):
                self.lbl_run_status.setText(
                    f"目标 {target_id} 距离画面中心 {distance_px:.1f}px，正在继续视觉对中；对中后自动抓取。"
                )
            return False
        if target.actionable:
            self._clear_pending_grab_after_align()
            self.lbl_run_status.setText(f"目标 {target_id} 已到达抓取中心，正在自动执行抓取。")
            self._execute_physical_grab()
            return True
        if target.servo_required:
            self.lbl_run_status.setText(f"目标 {target_id} 中心对准已执行，等待视觉刷新为可抓取状态后自动抓取。")
            return False
        reason = str(target.invalid_reason or "视觉目标未达到可抓取状态")
        self._clear_pending_grab_after_align()
        self.lbl_run_status.setText(f"目标 {target_id} 中心对准后仍不可抓取：{reason}")
        return False

    def _execute_physical_grab(self) -> None:
        self.cam.set_ssvep_flicker(False, selected_id=self.selected_target)
        if self.config.vision_enabled:
            target = self._selected_vision_target()
            if target is None:
                self.lbl_run_status.setText(
                    f"目标 {self.selected_target} 已选中并停止闪烁，但视觉识别还没有锁定该木块；请确认识别已启动并让木块进入画面。"
                )
                self._log(f"vision pick blocked: target {self.selected_target} unavailable")
                self._refresh_display_state()
                return
            if target.servo_required and not target.actionable:
                self.lbl_run_status.setText(
                    f"已收到目标 {self.selected_target} 的抓取确认，正在先对准画面中心；对准完成后将自动抓取。"
                )
                self._refresh_display_state()
                self._request_grab_alignment(target, mode="vision_profile")
                return
            if self._target_can_use_camera_center_grab(target):
                self.lbl_run_status.setText(
                    f"已收到目标 {self.selected_target} 的抓取确认，正在按画面中心进行视觉对中；稳定后将自动抓取。"
                )
                self._refresh_display_state()
                self._request_grab_alignment(target, mode="camera_center")
                return
            if not target.actionable:
                reason = str(target.invalid_reason or "视觉目标未达到可抓取状态")
                self.lbl_run_status.setText(f"目标 {self.selected_target} 暂不可抓取：{reason}")
                self._log(f"vision pick blocked: target {self.selected_target}: {reason}")
                self._refresh_display_state()
                return
            self._clear_pending_grab_after_align()
            self.control_phase = "PICKING"
            self._hide_decision_widgets()
            self.lbl_run_status.setText(f"正在按视觉中心精确抓取目标 {self.selected_target}")
            self.lbl_grip.setText("吸嘴状态: BUSY / VISION PICK")
            self.lbl_grip.setStyleSheet("color: #b45309; font-weight: 800;")
            self.backend.pick_vision_target(self.selected_target, target)
            self.pipeline_updated.emit(4, 58.0)
            self._refresh_display_state()
            return
        self._clear_pending_grab_after_align()
        self.control_phase = "PICKING"
        self._hide_decision_widgets()
        self.lbl_run_status.setText(f"正在下发机械臂抓取命令：目标 {self.selected_target}")
        self.lbl_grip.setText("吸嘴状态: BUSY / PICK")
        self.lbl_grip.setStyleSheet("color: #b45309; font-weight: 800;")
        self.backend.pick_target(self.selected_target)
        self.pipeline_updated.emit(4, 58.0)
        self._refresh_display_state()

    def _enter_decision_stage_2(self) -> None:
        self.backend.stop_teleop()
        self.control_phase = "DECIDE_2"
        self.lbl_run_status.setText("Stage 3 Placement Decision：SSVEP 确认是否放置；确认放置，继续则保持带载移动。")
        self._show_decision_widgets(
            title="Stage 3 Placement Decision",
            desc="确认: 放下    继续: 带载移动",
            confirm="确认放下\n确认",
            cont="继续搬运\n继续",
        )
        self.pipeline_updated.emit(4, 82.0)
        self._refresh_display_state()

    def _execute_physical_release(self) -> None:
        self.control_phase = "PLACING"
        self._hide_decision_widgets()
        self.lbl_run_status.setText("正在下发机械臂放置命令。")
        self.lbl_grip.setText("吸嘴状态: BUSY / PLACE")
        self.lbl_grip.setStyleSheet("color: #b45309; font-weight: 800;")
        self.backend.place()
        self.pipeline_updated.emit(4, 92.0)
        self._refresh_display_state()

    def _finish_task(self) -> None:
        self.control_phase = "TASK_DONE"
        self.flow_timer.stop()
        self.backend.stop_teleop()
        self.manual_drive_enabled = False
        self._hide_decision_widgets()
        self.cam.set_ssvep_flicker(False)
        self.cam.active_id = ""
        self.cam.update()
        self.lbl_run_status.setText("完整控制流程完成：移动、确认、目标选择、抓取、带载移动和放置已经走完。")
        self.lbl_grip.setText("吸嘴状态: STANDBY / OPEN")
        self.lbl_grip.setStyleSheet("")
        self.btn_master_start.setEnabled(True)
        self.btn_master_start.setText("再次开始完整控制流程")
        self.btn_manual_drive.setText("MI 移动控制")
        self.pipeline_updated.emit(4, 100.0)
        self._log("完整控制流程完成。")
        self._refresh_display_state()

    def _primary_decision(self) -> None:
        if self.control_phase == "IDLE":
            self._start_integrated_flow()
        elif self.control_phase == "MI_RECOGNITION":
            self._enter_ssvep_grab_stage()
        elif self.control_phase == "MI_MOVE_1":
            self._enter_decision_stage_1()
        elif self.control_phase == "DECIDE_1":
            self._enter_ssvep_target_selection()
        elif self.control_phase == "SSVEP_TARGET_SELECT":
            self._lock_ssvep_target_and_confirm(self.selected_target)
        elif self.control_phase == "GRASP_CONFIRM":
            self._execute_physical_grab()
        elif self.control_phase == "TARGET_LOCKED":
            self._execute_physical_grab()
        elif self.control_phase == "TASK_DONE":
            self._start_integrated_flow()
        elif self.control_phase == "MI_MOVE_2":
            self._enter_decision_stage_2()
        elif self.control_phase == "DECIDE_2":
            self._execute_physical_release()

    def _secondary_decision(self) -> None:
        if self.control_phase in {"MI_RECOGNITION", "DECIDE_1"}:
            self._enter_mi_move_stage(1)
        elif self.control_phase == "SSVEP_TARGET_SELECT":
            self._enter_mi_move_stage(1)
        elif self.control_phase in {"GRASP_CONFIRM", "TARGET_LOCKED"}:
            self._enter_ssvep_target_selection()
        elif self.control_phase == "DECIDE_2":
            self._enter_mi_move_stage(2)

    def _select_target(self, target_id: str) -> None:
        normalized_target = str(target_id)
        if normalized_target not in self.target_ids:
            self._log(f"ignored target selection outside configured range: {normalized_target}")
            return
        if self.control_phase == "SSVEP_TARGET_SELECT" and self.cam.ssv_flicker_enabled:
            self._lock_ssvep_target_and_confirm(normalized_target)
            return
        self.selected_target = normalized_target
        self.cam.active_id = self.selected_target
        self.cam.update()
        for key, button in self.target_buttons.items():
            button.setProperty("blockState", "active" if key == self.selected_target else "pending")
            button.style().unpolish(button)
            button.style().polish(button)
        if self.control_phase == "SSVEP_TARGET_SELECT":
            self.cam.set_ssvep_flicker(True)
        elif self.control_phase == "GRASP_CONFIRM":
            self.lbl_run_status.setText(f"目标 {self.selected_target} 已锁定，闪烁已停止。等待抓取确认信号；也可重新选择目标。")
            self.lbl_popup_desc.setText(f"目标 {self.selected_target} 已锁定    确认: 抓取    重新选择: 返回选择")
            self.flash_box_confirm.setText(f"抓取目标 {self.selected_target}\n确认")
            self.pbar_countdown.setFormat(f"目标 {self.selected_target} 已锁定")
        self._log(f"目标木块已选择: {self.selected_target}")
        self._refresh_display_state()

    def handle_key_press(self, event: QKeyEvent) -> bool:
        key = int(event.key())
        if key in {Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right}:
            if self.control_phase == "IDLE":
                self._enter_manual_drive()
            if self.control_phase not in {"MANUAL_DRIVE", "MI_MOVE_1", "MI_MOVE_2"}:
                return True
            self.pressed_move_keys.add(key)
            self._pump_keyboard_teleop()
            return True
        if Qt.Key_1 <= key <= Qt.Key_4:
            target_id = str(key - Qt.Key_0)
            if self.control_phase == "SSVEP_TARGET_SELECT":
                self._lock_ssvep_target_and_confirm(target_id)
            else:
                self._select_target(target_id)
            return True
        if key in {Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space}:
            self._primary_decision()
            return True
        if key in {Qt.Key_C, Qt.Key_Backspace}:
            self._secondary_decision()
            return True
        if key == Qt.Key_G:
            if self.control_phase in {"GRASP_CONFIRM", "TARGET_LOCKED"}:
                self._execute_physical_grab()
            return True
        if key == Qt.Key_P:
            if self.control_phase == "DECIDE_2":
                self._execute_physical_release()
            return True
        if key == Qt.Key_Escape:
            self._stop_current_task_safely()
            return True
        return False

    def handle_key_release(self, event: QKeyEvent) -> bool:
        key = int(event.key())
        if key in self.pressed_move_keys:
            self.pressed_move_keys.discard(key)
            if not self.pressed_move_keys:
                self.backend.stop_teleop()
            return True
        return False

    def _pump_keyboard_teleop(self) -> None:
        if self.control_phase not in {"MANUAL_DRIVE", "MI_MOVE_1", "MI_MOVE_2"}:
            return
        theta_rate = 0.0
        radius_rate = 0.0
        keys = self.pressed_move_keys
        if Qt.Key_A in keys or Qt.Key_Left in keys:
            theta_rate -= float(self.config.theta_rate_deg_s)
        if Qt.Key_D in keys or Qt.Key_Right in keys:
            theta_rate += float(self.config.theta_rate_deg_s)
        if Qt.Key_W in keys or Qt.Key_Up in keys:
            radius_rate += float(self.config.radius_rate_mm_s)
        if Qt.Key_S in keys or Qt.Key_Down in keys:
            radius_rate -= float(self.config.radius_rate_mm_s)
        if theta_rate or radius_rate:
            self.backend.send_teleop(theta_rate, radius_rate)

    def _show_decision_widgets(self, *, title: str, desc: str, confirm: str, cont: str) -> None:
        self.lbl_popup_title.setText(title)
        self.lbl_popup_desc.setText(desc)
        self.flash_box_confirm.setText(confirm)
        self.flash_box_continue.setText(cont)
        self.popup_dialog.show()
        self.flash_box_confirm.show()
        self.flash_box_continue.show()
        self.flash_timer.start()

    def _hide_decision_widgets(self) -> None:
        self.popup_dialog.hide()
        self.flash_box_confirm.hide()
        self.flash_box_continue.hide()
        self.flash_timer.stop()

    def _flash_tick(self) -> None:
        self.flash_state = not self.flash_state
        left_bg = "#ffffff" if self.flash_state else "#020617"
        left_fg = "#020617" if self.flash_state else "#38bdf8"
        right_bg = "#020617" if self.flash_state else "#ffffff"
        right_fg = "#f43f5e" if self.flash_state else "#020617"
        font_px = max(20, min(36, int(min(self.flash_box_confirm.width(), self.flash_box_confirm.height()) * 0.16)))
        self.flash_box_confirm.setStyleSheet(
            f"background: {left_bg}; color: {left_fg}; font-size: {font_px}px; font-weight: 900; border: 3px solid #0284c7; border-radius: 8px;"
        )
        self.flash_box_continue.setStyleSheet(
            f"background: {right_bg}; color: {right_fg}; font-size: {font_px}px; font-weight: 900; border: 3px solid #f43f5e; border-radius: 8px;"
        )

    def _stop_current_task_safely(self) -> None:
        self.flow_timer.stop()
        self.flash_timer.stop()
        self.backend.abort()
        self.backend.stop_teleop()
        self.manual_drive_enabled = False
        self.control_phase = "IDLE"
        self.phase_remaining_ms = 0
        self.pressed_move_keys.clear()
        self._clear_pending_grab_after_align()
        self._hide_decision_widgets()
        self.cam.set_ssvep_flicker(False)
        self.cam.active_id = ""
        self.cam.update()
        self.btn_manual_drive.setText("MI 移动控制")
        self.btn_master_start.setEnabled(True)
        self.btn_master_start.setText("开始完整控制流程")
        self.pbar_countdown.setRange(0, self.MI_RECOGNITION_MS)
        self.pbar_countdown.setValue(0)
        self.lbl_run_status.setText("已发送安全停止，正在自动复位控制状态。")
        self.pipeline_updated.emit(4, 0.0)
        self._log("安全停止；准备自动复位。")
        QTimer.singleShot(350, self._reset_robot_error)
        self._refresh_display_state()

    def _on_connection_changed(self, connected: bool) -> None:
        if not connected and self.manual_drive_enabled:
            self._exit_manual_drive()
        self.lbl_conn.setText("状态: 已连接" if connected else "状态: 未连接")
        self.lbl_conn.setStyleSheet("font-weight: 700; color: #047857;" if connected else "font-weight: 700; color: #991b1b;")
        self._refresh_display_state()

    def _on_pose_changed(self, theta: float, radius: float, z_mm: float) -> None:
        self.lbl_pose.setText(f"theta={theta:.1f} deg, radius={radius:.1f} mm, z={z_mm:.1f} mm")
        self._refresh_display_state()

    def _on_command_finished(self, action: str, ok: bool, message: str) -> None:
        status = "OK" if ok else "FAILED"
        self._log(f"{action}: {status} {message}".strip())
        if action == "pick" and ok:
            if self.control_phase != "PICKING":
                self._log(f"ignored stale pick completion while phase={self.control_phase}")
                return
            self.lbl_grip.setText("吸嘴状态: HOLDING / LOADED")
            self.lbl_grip.setStyleSheet("color: #047857; font-weight: 900;")
            self._enter_mi_move_stage(2)
            self.flow_timer.start()
        elif action == "vision-align" and ok:
            if self.control_phase not in {"GRASP_CONFIRM", "TARGET_LOCKED"}:
                self._log(f"ignored stale vision-align completion while phase={self.control_phase}")
                return
            self._align_for_grab_in_progress = False
            self._awaiting_vision_refresh_after_align = True
            if self._pending_grab_after_align_mode == "camera_center":
                self.lbl_run_status.setText("视觉对中命令已完成，正在等待新画面确认目标稳定在中心。")
            else:
                self.lbl_run_status.setText("中心对准完成，正在等待视觉刷新确认目标可抓取；可抓取后将自动执行抓取。")
            self._refresh_display_state()
        elif action == "place" and ok:
            if self.control_phase != "PLACING":
                self._log(f"ignored stale place completion while phase={self.control_phase}")
                return
            self._finish_task()
        elif action == "reset" and ok:
            if self.control_phase == "MANUAL_DRIVE":
                self.lbl_run_status.setText("复位完成：MI 移动控制可用。")
            else:
                self.lbl_run_status.setText("复位完成：机械臂控制状态已恢复，可以进入 MI 移动控制。")
            self._refresh_display_state()
        elif not ok:
            if action == "vision-align":
                self._clear_pending_grab_after_align()
            self.lbl_run_status.setText(f"命令失败: {action} {message}")
            self._refresh_display_state()

    def _log(self, message: str) -> None:
        self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {message}")


class BrainRobotWorkbenchWindow(QMainWindow):
    def __init__(self, config: WorkbenchConfig) -> None:
        super().__init__()
        self.config = config
        if self.config.demo_connected:
            self.config.robot_mode = "fake"
            self.config.connect_on_start = False
            self.config.eeg_signal_auto_start = False
            self.config.camera_auto_start = False
            self.config.vision_enabled = False
            self.config.vision_auto_start = False
            self.config.robot_runtime_auto_start = False
        self.backend = RobotCommandBackend(config, self)
        self.eeg_thread: BrainFlowEegStreamThread | None = None
        self._last_eeg_error = ""
        self._style_density_key = ""
        self._last_preview_status: tuple[str, str] = ("", "")
        self.setWindowTitle("脑机接口机械臂一体化控制台")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 720)
        self.setStyleSheet(APP_STYLE)
        self._build_ui()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self._apply_responsive_style(force=True)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QFrame()
        self.header_frame = header
        header.setObjectName("Header")
        header.setFixedHeight(62)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 0, 14, 0)
        logo = QLabel("BCI ROBOT CONSOLE")
        logo.setObjectName("HeaderTitle")
        header_layout.addWidget(logo)
        header_layout.addSpacing(8)
        self.nav_buttons: list[QPushButton] = []
        for idx, title in enumerate(("连接", "SSVEP", "MI", "机械臂")):
            button = QPushButton(title)
            button.setObjectName("NavButton")
            button.setProperty("active", "false")
            target_page = 4 if idx == 3 else idx
            button.clicked.connect(lambda checked=False, page=target_page: self._switch_page(page))
            header_layout.addWidget(button)
            self.nav_buttons.append(button)
        self.nav_buttons[1].setEnabled(False)
        self.nav_buttons[2].setEnabled(False)
        self.nav_buttons[3].setEnabled(False)
        header_layout.addStretch()
        self.header_status = QLabel("设备待连接")
        self.header_status.setObjectName("HeaderStatus")
        self.header_status.setProperty("state", "pending")
        header_layout.addWidget(self.header_status)
        outer.addWidget(header)

        body = QHBoxLayout()
        body.setContentsMargins(8, 8, 8, 8)
        body.setSpacing(8)
        self.stack = QStackedWidget()
        self.ssvep_page = KeyboardTrainingStageWidget(
            1,
            "阶段一：SSVEP 采集与训练界面",
            "保留你的 SSVEP 训练流程界面，完成数据采集后进入预训练阶段。",
            "S",
            stage_kind="ssvep",
        )
        self.mi_page = KeyboardTrainingStageWidget(
            2,
            "阶段二：MI 运动想象采集界面",
            "保留静息、提示、MI 运动想象采集流程；完成后进入模型预训练。",
            "M",
            stage_kind="mi",
        )
        self.robot_page = RobotFlowStageWidget(self.backend, self.config)
        self.training_page = PretrainingProgressWidget()
        self.connection_page = ConnectionGateWidget(self.backend, self.config)
        self.stack.addWidget(self.connection_page)
        self.stack.addWidget(self.ssvep_page)
        self.stack.addWidget(self.mi_page)
        self.stack.addWidget(self.training_page)
        self.stack.addWidget(self.robot_page)
        body.addWidget(self.stack, 5)

        self.side_panel = QFrame()
        side = self.side_panel
        side.setObjectName("Card")
        side.setFixedWidth(self._responsive_preview_width())
        side.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(12, 12, 12, 12)
        self.lbl_preview_title = QLabel("实时 8 通道 EEG")
        self.lbl_preview_title.setStyleSheet("color: #0369a1; font-weight: 800;")
        self.lbl_preview_status = QLabel("未连接真实 EEG 数据流")
        self.lbl_preview_status.setWordWrap(True)
        self.lbl_preview_status.setStyleSheet("color: #64748b; font-size: 12px;")
        self.preview = SignalPreviewWidget(window_seconds=float(self.config.eeg_signal_window_seconds))
        toggle = QHBoxLayout()
        self.btn_eeg = QPushButton("时域波形")
        self.btn_eeg.setObjectName("AccentButton")
        self.btn_imp = QPushButton("引脚阻抗")
        self.btn_imp.setObjectName("ActionButton")
        self.btn_eeg.clicked.connect(lambda: self.preview.set_mode("EEG"))
        self.btn_imp.clicked.connect(lambda: self.preview.set_mode("IMP"))
        toggle.addWidget(self.btn_eeg)
        toggle.addWidget(self.btn_imp)
        stream_controls = QHBoxLayout()
        self.btn_start_eeg_stream = QPushButton("启动信号")
        self.btn_start_eeg_stream.setObjectName("AccentButton")
        self.btn_stop_eeg_stream = QPushButton("停止信号")
        self.btn_stop_eeg_stream.setObjectName("ActionButton")
        self.btn_stop_eeg_stream.setEnabled(False)
        self.btn_start_eeg_stream.clicked.connect(lambda: self._start_eeg_signal_stream_from_ui(force=True))
        self.btn_stop_eeg_stream.clicked.connect(lambda: self._stop_eeg_signal_stream(clear_preview=True))
        stream_controls.addWidget(self.btn_start_eeg_stream)
        stream_controls.addWidget(self.btn_stop_eeg_stream)
        side_layout.addWidget(self.lbl_preview_title)
        side_layout.addWidget(self.lbl_preview_status)
        side_layout.addWidget(self.preview, 1)
        side_layout.addLayout(toggle)
        side_layout.addLayout(stream_controls)
        body.addWidget(side, 2)
        outer.addLayout(body, 1)

        self.monitor = PipelineProgressWidget()
        outer.addWidget(self.monitor)

        self.ssvep_page.pipeline_updated.connect(self.monitor.set_stage_progress)
        self.mi_page.pipeline_updated.connect(self.monitor.set_stage_progress)
        self.training_page.pipeline_updated.connect(self.monitor.set_stage_progress)
        self.robot_page.pipeline_updated.connect(self.monitor.set_stage_progress)
        self.connection_page.ready_changed.connect(self._on_connection_gate_ready)
        self.connection_page.proceed_requested.connect(self._enter_ssvep_stage)
        self.connection_page.robot_control_requested.connect(self._enter_robot_stage_direct)
        self.connection_page.eeg_stream_requested.connect(self._start_eeg_signal_stream)
        self.connection_page.eeg_stream_stop_requested.connect(lambda: self._stop_eeg_signal_stream(clear_preview=True))
        self.ssvep_page.stage_completed.connect(self._unlock_mi)
        self.mi_page.stage_completed.connect(self._start_pretraining)
        self.training_page.training_finished.connect(self._unlock_robot)
        self.training_page.btn_enter_robot.clicked.connect(lambda: self._switch_page(4))
        self._switch_page(0)
        if self.config.demo_connected:
            QTimer.singleShot(0, self._activate_demo_connected_mode)
        elif self.config.eeg_signal_auto_start:
            QTimer.singleShot(0, self.connection_page.connect_eeg_cap)

    def _activate_demo_connected_mode(self) -> None:
        self.connection_page.enable_demo_connected()
        self.nav_buttons[1].setEnabled(True)
        self.nav_buttons[2].setEnabled(False)
        self.nav_buttons[3].setEnabled(False)
        self.lbl_preview_status.setText("演示模式：未连接真实 EEG，右侧仅用于查看训练界面布局。")
        self._switch_page(1)

    def _responsive_preview_width(self) -> int:
        width = max(1, self.width())
        if width >= 2200:
            return 430
        if width >= 1800:
            return 380
        return 340

    def _responsive_density(self) -> tuple[str, float]:
        width = max(1, int(self.width()))
        if width >= 1900:
            return "xl", 1.32
        if width >= 1600:
            return "large", 1.18
        return "base", 1.0

    def _responsive_style_override(self, density: str) -> str:
        if density == "xl":
            return """
QWidget { font-size: 21px; }
QLabel#HeaderTitle { font-size: 22px; }
QLabel#HeaderStatus { font-size: 19px; padding: 10px 18px; }
QPushButton#NavButton { font-size: 19px; padding: 0 22px; }
QLabel#MetricTitle { font-size: 18px; }
QLabel#MetricValue { font-size: 27px; }
QLabel#DeviceTitle { font-size: 27px; }
QLabel#StatusChip { border-radius: 13px; font-size: 18px; padding: 9px 18px; }
QLineEdit, QComboBox { font-size: 21px; min-height: 52px; padding: 11px 16px; }
QPushButton#ActionButton, QPushButton#AccentButton, QPushButton#DangerButton {
    font-size: 21px;
    min-height: 64px;
    padding: 16px 24px;
}
QPushButton[blockState="pending"], QPushButton[blockState="active"] {
    font-size: 21px;
    min-height: 58px;
}
QProgressBar { font-size: 19px; min-height: 18px; }
QPlainTextEdit#LogView { font-size: 17px; }
"""
        if density == "large":
            return """
QWidget { font-size: 19px; }
QLabel#HeaderTitle { font-size: 20px; }
QLabel#HeaderStatus { font-size: 18px; padding: 8px 16px; }
QPushButton#NavButton { font-size: 18px; padding: 0 20px; }
QLabel#MetricTitle { font-size: 16px; }
QLabel#MetricValue { font-size: 24px; }
QLabel#DeviceTitle { font-size: 24px; }
QLabel#StatusChip { border-radius: 12px; font-size: 16px; padding: 8px 16px; }
QLineEdit, QComboBox { font-size: 19px; min-height: 46px; padding: 10px 14px; }
QPushButton#ActionButton, QPushButton#AccentButton, QPushButton#DangerButton {
    font-size: 19px;
    min-height: 56px;
    padding: 14px 22px;
}
QPushButton[blockState="pending"], QPushButton[blockState="active"] {
    font-size: 19px;
    min-height: 50px;
}
QProgressBar { font-size: 18px; min-height: 16px; }
QPlainTextEdit#LogView { font-size: 16px; }
"""
        return ""

    def _apply_responsive_style(self, *, force: bool = False) -> None:
        density, scale = self._responsive_density()
        if force or density != self._style_density_key:
            self._style_density_key = density
            self.setStyleSheet(APP_STYLE + self._responsive_style_override(density))
            if hasattr(self, "header_frame"):
                self.header_frame.setFixedHeight(80 if density == "xl" else 70 if density == "large" else 62)
            if hasattr(self, "monitor"):
                self.monitor.set_display_scale(scale)
            if hasattr(self, "connection_page"):
                self.connection_page.apply_visual_scale(scale)

    def _set_preview_status(self, text: str, *, state: str = "pending") -> None:
        message = str(text or "").strip()
        if not message:
            return
        current = (message, str(state))
        if current == self._last_preview_status:
            return
        self._last_preview_status = current
        self.lbl_preview_status.setText(message)
        if state == "ready":
            color = "#047857"
        elif state == "error":
            color = "#b91c1c"
        else:
            color = "#64748b"
        _, scale = self._responsive_density()
        font_px = int(round(12 * max(1.0, min(1.35, scale))))
        self.lbl_preview_status.setStyleSheet(f"color: {color}; font-size: {font_px}px;")
        self.preview.set_status_text(message)
        if hasattr(self, "connection_page"):
            self.connection_page.lbl_eeg_detail.setText(message)

    def _current_eeg_inputs(self) -> tuple[str, int] | None:
        serial_port = self.connection_page.eeg_serial_edit.text().strip() or "auto"
        board_id = _coerce_eeg_board_id(self.connection_page.eeg_board_edit.text().strip() or "0")
        if board_id is None:
            self._set_preview_status("BrainFlow board_id 必须是整数。", state="error")
            return None
        self.config.eeg_serial_port = serial_port
        self.config.eeg_board_id = int(board_id)
        return serial_port, int(board_id)

    def _start_eeg_signal_stream_from_ui(self, *, force: bool = False) -> None:
        inputs = self._current_eeg_inputs()
        if inputs is None:
            return
        serial_port, board_id = inputs
        self._start_eeg_signal_stream(serial_port, board_id, force=force)

    def _start_eeg_signal_stream(self, serial_port: str, board_id: int, force: bool = False) -> None:
        serial = str(serial_port or "auto").strip() or "auto"
        board = int(board_id)
        self.config.eeg_serial_port = serial
        self.config.eeg_board_id = board
        if not force and not bool(self.config.eeg_signal_auto_start):
            self._set_preview_status("实时 EEG 自动启动已关闭，可点击“启动信号”手动连接。")
            return
        if not force and _serial_port_is_auto(serial) and board != -1 and not _detect_serial_port_candidates():
            message = "脑电帽门禁已就绪，但未检测到串口；填写 COMx 后点击“启动信号”可显示真实 8 通道 EEG。"
            self.preview.clear_stream(message)
            self._set_preview_status(message)
            return

        self._stop_eeg_signal_stream(clear_preview=False)
        self._last_eeg_error = ""
        self.preview.clear_stream("正在连接真实 BrainFlow EEG 数据流...")
        self._set_preview_status(f"正在启动实时 EEG: serial={serial}, board_id={board}")
        self.eeg_thread = BrainFlowEegStreamThread(
            serial_port=serial,
            board_id=board,
            poll_interval_sec=float(self.config.eeg_signal_poll_interval_sec),
            channel_count=EEG_DISPLAY_CHANNEL_COUNT,
            parent=self,
        )
        self.eeg_thread.stream_ready.connect(self._on_eeg_stream_ready)
        self.eeg_thread.samples_ready.connect(self.preview.append_chunk)
        self.eeg_thread.samples_ready.connect(self._on_training_samples_ready)
        self.eeg_thread.status_changed.connect(self._on_eeg_stream_status)
        self.eeg_thread.error_occurred.connect(self._on_eeg_stream_error)
        self.eeg_thread.finished.connect(self._on_eeg_signal_thread_finished)
        self.btn_start_eeg_stream.setEnabled(False)
        self.btn_stop_eeg_stream.setEnabled(True)
        self.eeg_thread.start()

    def _on_eeg_stream_ready(self, payload: object) -> None:
        info = dict(payload or {}) if isinstance(payload, dict) else {}
        sampling_rate = float(info.get("sampling_rate", 0.0) or 0.0)
        channel_names = [str(item) for item in info.get("channel_names", [])]
        selected_rows = [int(item) for item in info.get("selected_rows", [])]
        self.preview.configure_stream(sampling_rate=sampling_rate or 250.0, channel_names=channel_names or EEG_DEFAULT_CHANNEL_NAMES)
        self.ssvep_page.configure_capture_stream(sampling_rate=sampling_rate or 250.0, channel_names=channel_names or EEG_DEFAULT_CHANNEL_NAMES)
        self.mi_page.configure_capture_stream(sampling_rate=sampling_rate or 250.0, channel_names=channel_names or EEG_DEFAULT_CHANNEL_NAMES)
        message = (
            f"真实 EEG 已接入: {len(selected_rows) or len(channel_names)} 通道, "
            f"{sampling_rate:g} Hz, board_id={info.get('board_id', self.config.eeg_board_id)}, "
            f"serial={info.get('serial_port', self.config.eeg_serial_port)}"
        )
        self._set_preview_status(message, state="ready")

    def _on_training_samples_ready(self, chunk: object) -> None:
        current = self.stack.currentIndex()
        if current == 1:
            self.ssvep_page.append_eeg_chunk(chunk)
        elif current == 2:
            self.mi_page.append_eeg_chunk(chunk)

    def _on_eeg_stream_status(self, message: str) -> None:
        text = str(message or "").strip()
        if not text:
            return
        if self._last_eeg_error and "已停止" in text:
            return
        self._set_preview_status(text, state="ready" if "已连接" in text else "pending")

    def _on_eeg_stream_error(self, message: str) -> None:
        text = str(message or "").strip()
        if not text:
            return
        self._last_eeg_error = text
        self.preview.clear_stream(text)
        self._set_preview_status(text, state="error")

    def _on_eeg_signal_thread_finished(self) -> None:
        thread = self.sender()
        if thread is self.eeg_thread:
            self.eeg_thread = None
        self.btn_start_eeg_stream.setEnabled(True)
        self.btn_stop_eeg_stream.setEnabled(False)

    def _stop_eeg_signal_stream(self, *, clear_preview: bool = True) -> None:
        thread = self.eeg_thread
        if thread is not None:
            thread.stop()
            thread.wait(1500)
            if thread.isRunning():
                self._set_preview_status("实时 EEG 正在停止，等待采集线程退出。")
            else:
                self.eeg_thread = None
        self.btn_start_eeg_stream.setEnabled(True)
        self.btn_stop_eeg_stream.setEnabled(False)
        if clear_preview:
            self.preview.clear_stream("实时 EEG 已停止，当前为占位波形。")
            self._set_preview_status("实时 EEG 已停止，当前为占位波形。")

    def _switch_page(self, index: int) -> None:
        if index < 0 or index >= self.stack.count():
            return
        if index == 3:
            pass
        elif index == 4:
            if not self.nav_buttons[3].isEnabled():
                return
        elif not self.nav_buttons[index].isEnabled():
            return
        self.stack.setCurrentIndex(index)
        if index == 4:
            self.side_panel.hide()
            self.monitor.hide()
            self.preview.timer.stop()
            if self.config.camera_auto_start:
                self.robot_page.start_camera()
        else:
            self.monitor.setVisible(False)
            self.robot_page.stop_camera()
            self.side_panel.setVisible(index in {1, 2})
            if index in {1, 2} and not self.preview.timer.isActive():
                self.preview.timer.start()
            elif index in {0, 3}:
                self.preview.timer.stop()
        if index == 4:
            self.robot_page.setFocus(Qt.OtherFocusReason)
        active_nav_index = 3 if index == 4 else index if index in {0, 1, 2} else -1
        for idx, button in enumerate(self.nav_buttons):
            button.setProperty("active", "true" if idx == active_nav_index else "false")
            button.style().unpolish(button)
            button.style().polish(button)
        self._refresh_header_status(index)

    def eventFilter(self, source: object, event: QEvent) -> bool:  # noqa: N802
        if (
            hasattr(self, "stack")
            and self.stack.currentIndex() == 4
            and isinstance(source, QWidget)
            and (source is self or self.isAncestorOf(source))
            and event.type() in {QEvent.KeyPress, QEvent.KeyRelease}
        ):
            if event.type() == QEvent.KeyPress and self.robot_page.handle_key_press(event):
                event.accept()
                return True
            if event.type() == QEvent.KeyRelease and self.robot_page.handle_key_release(event):
                event.accept()
                return True
        return super().eventFilter(source, event)

    def _set_header_status(self, text: str, state: str) -> None:
        self.header_status.setText(text)
        self.header_status.setProperty("state", state)
        self.header_status.style().unpolish(self.header_status)
        self.header_status.style().polish(self.header_status)

    def _refresh_header_status(self, index: int) -> None:
        if index == 0:
            if self.connection_page.robot_connected and self.connection_page.eeg_connected:
                self._set_header_status("设备已就绪", "ready")
            else:
                self._set_header_status("等待设备连接", "pending")
        elif index == 1:
            self._set_header_status("SSVEP 数据采集", "active")
        elif index == 2:
            self._set_header_status("MI 数据采集", "active")
        elif index == 3:
            self._set_header_status("模型训练中", "active")
        elif index == 4:
            self._set_header_status("机械臂实时控制", "active")

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_responsive_style()
        if hasattr(self, "side_panel"):
            target_width = self._responsive_preview_width()
            if self.side_panel.width() != target_width:
                self.side_panel.setFixedWidth(target_width)

    def _on_connection_gate_ready(self, ready: bool) -> None:
        self._refresh_header_status(self.stack.currentIndex())
        if ready:
            self.nav_buttons[1].setEnabled(True)
            return
        self.training_page.timer.stop()
        self.training_page.training_running = False
        self.training_page.training_complete = False
        self.connection_page.set_robot_control_unlocked(False)
        for button in self.nav_buttons[1:]:
            button.setEnabled(False)
        if self.stack.currentIndex() != 0:
            self._switch_page(0)

    def _enter_ssvep_stage(self) -> None:
        if self.connection_page.robot_connected and self.connection_page.eeg_connected:
            self.nav_buttons[1].setEnabled(True)
            self._switch_page(1)

    def _enter_robot_stage_direct(self) -> None:
        if (
            self.connection_page.robot_connected
            and self.connection_page.eeg_connected
            and self.training_page.training_complete
        ):
            self.nav_buttons[1].setEnabled(True)
            self.nav_buttons[2].setEnabled(True)
            self.nav_buttons[3].setEnabled(True)
            self._switch_page(4)

    def _unlock_mi(self) -> None:
        self.nav_buttons[1].setEnabled(True)
        self.nav_buttons[2].setEnabled(True)
        self._switch_page(2)

    def _start_pretraining(self) -> None:
        self.nav_buttons[3].setEnabled(False)
        self.connection_page.set_robot_control_unlocked(False)
        self.training_page.start_training(
            ssvep_data_path=self.ssvep_page.last_capture_path,
            mi_data_path=self.mi_page.last_capture_path,
        )
        self._switch_page(3)

    def _unlock_robot(self) -> None:
        self.connection_page.set_robot_control_unlocked(True)
        self.nav_buttons[3].setEnabled(True)
        self._switch_page(4)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        current = self.stack.currentIndex()
        handled = False
        if current == 1:
            handled = self.ssvep_page.handle_shortcut(event)
        elif current == 2:
            handled = self.mi_page.handle_shortcut(event)
        elif current == 4:
            handled = self.robot_page.handle_key_press(event)
        if handled:
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if self.stack.currentIndex() == 4 and self.robot_page.handle_key_release(event):
            event.accept()
            return
        super().keyReleaseEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self._stop_eeg_signal_stream(clear_preview=False)
        self.training_page.timer.stop()
        self.robot_page.stop_camera()
        self.robot_page._stop_current_task_safely()
        self.backend.close_backend()
        event.accept()


def run_workbench(config: WorkbenchConfig) -> int:
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication([])
    window = BrainRobotWorkbenchWindow(config)
    window.show()
    if int(config.smoke_test_ms) > 0:
        QTimer.singleShot(int(config.smoke_test_ms), window.close)
    result = int(app.exec_()) if owns_app else 0
    return result
