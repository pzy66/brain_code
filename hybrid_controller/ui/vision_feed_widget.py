from __future__ import annotations

import math
import threading
import time
from typing import Any, Optional

import numpy as np
from PyQt5.QtCore import QPointF, QRect, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget


def fit_rect(src_width: int, src_height: int, dst_width: int, dst_height: int) -> QRect:
    if src_width <= 0 or src_height <= 0 or dst_width <= 0 or dst_height <= 0:
        return QRect()
    scale = min(dst_width / float(src_width), dst_height / float(src_height))
    draw_width = max(1, int(round(src_width * scale)))
    draw_height = max(1, int(round(src_height * scale)))
    left = (dst_width - draw_width) // 2
    top = (dst_height - draw_height) // 2
    return QRect(left, top, draw_width, draw_height)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class StimClockThread(QThread):
    tick = pyqtSignal(float, int, float)

    def __init__(self, refresh_rate_hz: float) -> None:
        super().__init__()
        self.refresh_rate_hz = max(1.0, float(refresh_rate_hz))
        self.period_ns = int(round(1_000_000_000 / self.refresh_rate_hz))
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        self._stop_event.clear()
        frame_idx = 0
        start_ns = time.perf_counter_ns()
        next_tick_ns = start_ns
        while not self._stop_event.is_set():
            now_ns = time.perf_counter_ns()
            remain_ns = next_tick_ns - now_ns
            if remain_ns > 1_000_000:
                sleep_s = max((remain_ns - 200_000) / 1_000_000_000.0, 0.0)
                if self._stop_event.wait(sleep_s):
                    break
                continue

            while not self._stop_event.is_set():
                now_ns = time.perf_counter_ns()
                remain_ns = next_tick_ns - now_ns
                if remain_ns <= 0:
                    break
                if remain_ns > 200_000:
                    time.sleep(0)
                    continue
                if remain_ns > 50_000:
                    time.sleep(0.0001)
                    continue

            if self._stop_event.is_set():
                break

            emit_ns = time.perf_counter_ns()
            jitter_ms = (emit_ns - next_tick_ns) / 1_000_000.0
            rel_t = (next_tick_ns - start_ns) / 1_000_000_000.0
            self.tick.emit(rel_t, frame_idx, float(jitter_ms))

            frame_idx += 1
            next_tick_ns = start_ns + frame_idx * self.period_ns
            if emit_ns - next_tick_ns > self.period_ns:
                frame_idx = int((emit_ns - start_ns) // self.period_ns) + 1
                next_tick_ns = start_ns + frame_idx * self.period_ns


class VisionFeedWidget(QWidget):
    def __init__(self, *, refresh_rate_hz: float = 120.0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame_image: QImage | None = None
        self._frame_pixmap: QPixmap | None = None
        self._frame_object_id: int | None = None
        self._frame_size = (0, 0)
        self._image_rect = QRect()
        self._packet: dict[str, Any] | None = None
        self._flash_enabled = False
        self._current_t = 0.0
        self._tick_jitter_ms = 0.0
        self._last_passive_redraw_t = 0.0
        self._passive_redraw_interval_sec = 1.0 / 30.0
        self._status_text = "Waiting for vision runtime..."
        self._hud_font = QFont("Consolas", 10)
        self._label_font = QFont("Consolas", 12, QFont.Bold)
        self._clock = StimClockThread(self._resolve_refresh_rate(refresh_rate_hz))
        self._clock.tick.connect(self._on_tick)
        self._clock.start()
        self.setMinimumSize(760, 520)

    @staticmethod
    def _resolve_refresh_rate(requested_refresh_hz: float) -> float:
        requested = max(1.0, float(requested_refresh_hz))
        app = QApplication.instance()
        if app is None:
            return requested
        screen = app.primaryScreen()
        if screen is None:
            return requested
        screen_refresh = float(screen.refreshRate())
        if screen_refresh <= 1.0:
            return requested
        return min(requested, screen_refresh)

    def shutdown(self) -> None:
        if self._clock.isRunning():
            self._clock.request_stop()
            self._clock.wait(1000)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.shutdown()
        super().closeEvent(event)

    def set_payload(
        self,
        *,
        frame_bgr: np.ndarray | None,
        packet: dict[str, Any] | None,
        flash_enabled: bool = False,
        status_text: str | None = None,
    ) -> None:
        if frame_bgr is not None:
            frame_object_id = id(frame_bgr)
            if frame_object_id != self._frame_object_id:
                self._frame_image = self._frame_to_qimage(frame_bgr)
                self._frame_pixmap = QPixmap.fromImage(self._frame_image)
                self._frame_size = (int(frame_bgr.shape[1]), int(frame_bgr.shape[0]))
                self._frame_object_id = frame_object_id
        self._packet = packet
        self._flash_enabled = bool(flash_enabled)
        if status_text is not None:
            self._status_text = str(status_text)
        self.update()

    def _on_tick(self, current_t: float, _frame_idx: int, jitter_ms: float) -> None:
        self._current_t = float(current_t)
        self._tick_jitter_ms = float(jitter_ms)
        if self._flash_enabled:
            self.update()
            return
        if (self._current_t - self._last_passive_redraw_t) >= self._passive_redraw_interval_sec:
            self._last_passive_redraw_t = self._current_t
            self.update()

    @staticmethod
    def _frame_to_qimage(frame_bgr: np.ndarray) -> QImage:
        rgb = frame_bgr[:, :, ::-1].copy()
        height, width = rgb.shape[:2]
        bytes_per_line = width * 3
        return QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()

    def _ensure_image_rect(self) -> None:
        if self._frame_image is None:
            self._image_rect = QRect()
            return
        self._image_rect = fit_rect(
            self._frame_image.width(),
            self._frame_image.height(),
            max(1, self.width()),
            max(1, self.height()),
        )

    def _map_point(self, point: tuple[int, int]) -> QPointF:
        if self._image_rect.isNull() or self._frame_size[0] <= 0 or self._frame_size[1] <= 0:
            return QPointF(0.0, 0.0)
        scale_x = self._image_rect.width() / float(self._frame_size[0])
        scale_y = self._image_rect.height() / float(self._frame_size[1])
        return QPointF(
            self._image_rect.left() + point[0] * scale_x,
            self._image_rect.top() + point[1] * scale_y,
        )

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(16, 18, 24))
        self._ensure_image_rect()
        if self._frame_pixmap is not None and not self._image_rect.isNull():
            painter.drawPixmap(self._image_rect, self._frame_pixmap)
        else:
            painter.setPen(QColor(220, 220, 220))
            painter.setFont(QFont("Arial", 16, QFont.Bold))
            painter.drawText(self.rect(), Qt.AlignCenter, "Waiting for camera frames...")
        if self._packet is not None and not self._image_rect.isNull():
            self._draw_slots(painter)
            self._draw_roi(painter)
            self._draw_hud(painter)
        else:
            self._draw_status_only(painter)

    def _draw_slots(self, painter: QPainter) -> None:
        assert self._packet is not None
        if self._flash_enabled:
            for slot in self._packet.get("slots", []):
                if not slot.get("valid"):
                    continue
                polygon = slot.get("polygon") or []
                if len(polygon) < 3:
                    continue
                mapped_polygon = QPolygonF(
                    [self._map_point((int(point[0]), int(point[1]))) for point in polygon]
                )
                luminance = clamp01(
                    0.5 + 0.5 * math.sin(2.0 * math.pi * float(slot.get("freq_hz", 0.0)) * self._current_t)
                )
                gray_value = int(round(255.0 * luminance))
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(gray_value, gray_value, gray_value, 180))
                painter.drawPolygon(mapped_polygon)

        for slot in self._packet.get("slots", []):
            if not slot.get("valid"):
                continue
            polygon = slot.get("polygon") or []
            center = slot.get("pixel_center")
            border_color = QColor(0, 255, 180) if slot.get("observed", False) else QColor(255, 200, 0)
            bbox = slot.get("bbox")
            if len(polygon) >= 3:
                mapped_polygon = QPolygonF(
                    [self._map_point((int(point[0]), int(point[1]))) for point in polygon]
                )
                painter.setPen(QPen(border_color, 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawPolygon(mapped_polygon)
            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                p1 = self._map_point((int(bbox[0]), int(bbox[1])))
                p2 = self._map_point((int(bbox[2]), int(bbox[3])))
                left = int(round(min(p1.x(), p2.x())))
                top = int(round(min(p1.y(), p2.y())))
                width = max(1, int(round(abs(p2.x() - p1.x()))))
                height = max(1, int(round(abs(p2.y() - p1.y()))))
                painter.setPen(QPen(border_color, 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(left, top, width, height)
            if center is None:
                continue

            mapped_center = self._map_point((int(center[0]), int(center[1])))
            center_x = int(round(mapped_center.x()))
            center_y = int(round(mapped_center.y()))
            painter.drawLine(center_x - 8, center_y, center_x + 8, center_y)
            painter.drawLine(center_x, center_y - 8, center_x, center_y + 8)
            painter.setFont(self._label_font)
            label = f"[{slot['slot_id']}] {float(slot['freq_hz']):g}Hz"
            painter.drawText(center_x + 10, center_y - 10, label)

    def _draw_roi(self, painter: QPainter) -> None:
        assert self._packet is not None
        roi_center = self._packet.get("roi_center")
        roi_radius = int(self._packet.get("roi_radius", 0))
        if roi_center is None or roi_radius <= 0:
            return
        mapped_center = self._map_point((int(roi_center[0]), int(roi_center[1])))
        scale = self._image_rect.width() / float(self._frame_size[0]) if self._frame_size[0] > 0 else 1.0
        mapped_radius = max(1, int(round(roi_radius * scale)))
        painter.setPen(QPen(QColor(255, 128, 0), 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(mapped_center, mapped_radius, mapped_radius)

    def _draw_hud(self, painter: QPainter) -> None:
        assert self._packet is not None
        metrics = [
            self._status_text,
            f"capture_fps={float(self._packet.get('capture_fps', 0.0)):.1f}",
            f"infer_ms={float(self._packet.get('infer_ms', 0.0)):.1f}",
            f"queue_age_ms={float(self._packet.get('queue_age_ms', 0.0)):.1f}",
            f"infer_interval_ms={float(self._packet.get('infer_interval_ms', 0.0)):.1f}",
            f"frame_drop_ratio={float(self._packet.get('frame_drop_ratio', 0.0)):.2f}",
            f"detected_count={int(self._packet.get('detected_count', 0))}",
            f"slots={sum(1 for slot in self._packet.get('slots', []) if slot.get('valid'))}",
            f"ssvep_flash={'on' if self._flash_enabled else 'off'}",
            f"tick_jitter_ms={self._tick_jitter_ms:.3f}",
            f"calibration_ready={bool(self._packet.get('calibration_ready', False))}",
        ]
        painter.setFont(self._hud_font)
        line_height = 18
        box_width = 360
        box_height = line_height * len(metrics) + 16
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.drawRoundedRect(16, 16, box_width, box_height, 8, 8)
        painter.setPen(QColor(230, 230, 230))
        for index, text in enumerate(metrics):
            painter.drawText(28, 16 + 24 + index * line_height, text)

    def _draw_status_only(self, painter: QPainter) -> None:
        painter.setPen(QColor(230, 230, 230))
        painter.setFont(QFont("Consolas", 12))
        painter.drawText(self.rect(), Qt.AlignBottom | Qt.AlignHCenter, self._status_text)
