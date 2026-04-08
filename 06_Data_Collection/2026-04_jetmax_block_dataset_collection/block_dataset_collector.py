#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import queue
import re
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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = "http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "camara" / "captures"
DEFAULT_WINDOW_SIZE = (1440, 900)
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


@dataclass
class SaveJob:
    image: np.ndarray
    image_path: Path
    manifest_path: Path
    metadata: Dict[str, Any]


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
        self.writer.saved.connect(self._on_save_success)
        self.writer.failed.connect(self._on_save_failed)

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
        if event.key() == Qt.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self._auto_capture_enabled = False
        self._burst_remaining = 0
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
    args = parser.parse_args(argv)

    output_root = Path(args.output_root).expanduser().resolve()
    return AppConfig(
        source=parse_source(args.source),
        output_root=output_root,
        session_prefix=sanitize_token(args.session_prefix, "block_collect"),
        image_ext=args.image_ext.lower(),
        jpeg_quality=max(50, min(100, int(args.jpeg_quality))),
        auto_close_sec=max(0.0, float(args.exit_after_sec)),
        fullscreen=bool(args.fullscreen),
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
