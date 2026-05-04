"""全中文运动想象采集界面。"""

from __future__ import annotations

import argparse
import base64
from collections import deque
from dataclasses import replace
import secrets
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRAIN_CODE_ROOT = PROJECT_ROOT.parents[1]
if str(BRAIN_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRAIN_CODE_ROOT))
from brain_workspace.paths import DATASETS_ROOT, MI_DATASET_DIR, resolve_data_path

DEFAULT_OUTPUT_ROOT = MI_DATASET_DIR
SHARED_ROOT = PROJECT_ROOT / "code" / "shared"
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

import src  # noqa: F401

from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes
from PyQt5.QtCore import QObject, QPointF, QRect, QRectF, Qt, QEventLoop, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QBrush, QColor, QFont, QFontMetrics, QImage, QLinearGradient, QPainter, QPainterPath, QPen
try:
    from PyQt5.QtTextToSpeech import QTextToSpeech
except Exception:  # pragma: no cover - optional Qt module
    QTextToSpeech = None
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QSpinBox,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    import winsound
except ImportError:  # pragma: no cover - non-Windows fallback
    winsound = None

from src.mi_collection import (
    CLASS_LOOKUP,
    CONTINUOUS_PROMPT_EVENT_NAMES,
    CONTINUOUS_PROMPT_LABELS,
    DEFAULT_ARTIFACT_TYPES,
    PAUSE_REJECTED_TRIAL_NOTE,
    SessionSettings,
    TrialRecord,
    build_balanced_trial_sequence,
    make_event,
    normalize_artifact_types,
    parse_channel_names,
    parse_channel_positions,
    save_mi_emergency_raw_capture,
    save_mi_session,
)
from src.serial_ports import describe_serial_port, detect_serial_ports, validate_serial_port_selection


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def resolve_output_root(value: str | Path | None = None) -> Path:
    return resolve_data_path(
        value,
        base=DATASETS_ROOT,
        default=DEFAULT_OUTPUT_ROOT,
        purpose="MI output_root",
    )


DEFAULT_CHANNEL_NAMES = ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"]
PROTOCOL_MODE_FULL = "full"
PROTOCOL_MODE_MI_ONLY = "mi_only"
IMAGERY_START_TONE_HZ = 1200
IMAGERY_START_TONE_MS = 180
IMAGERY_START_POST_TONE_GUARD_MS = 80
IMAGERY_END_TONE_HZ = 800
IMAGERY_END_TONE_MS = 180
CUE_WARNING_TONE_HZ = 980
CUE_WARNING_TONE_MS = 90
PHASE_START_TONE_HZ = 1200
PHASE_START_TONE_MS = 180
PHASE_END_TONE_HZ = 720
PHASE_END_TONE_MS = 180
MI_TRIAL_PHASES = {"baseline", "cue", "imagery", "iti"}
MARKER_FLUSH_SAMPLE_COUNT = 2.0
MARKER_FLUSH_MIN_SEC = 0.006
MARKER_FLUSH_MAX_SEC = 0.030
MARKER_FLUSH_FALLBACK_SEC = 0.012
MISSING_SAVE_RESULT_GRACE_MS = 1500
QUALITY_FLAT_CHANNEL_STD_UV = 0.5
QUALITY_FLAT_CHANNEL_PTP_UV = 1.0
QUALITY_WARNING_CHANNEL_LIMIT = 4
CUE_VOICE_PROMPTS = {
    "left_hand": "左手",
    "right_hand": "右手",
    "feet": "双脚",
    "tongue": "舌头",
}
CUE_VOICE_PROMPT_TIMEOUT_SEC = 2.0
PHASE_VOICE_PROMPT_TIMEOUT_SEC = 3.0
VOICE_PROMPT_FINISH_GUARD_MS = 250
PHASE_START_POST_TONE_GUARD_MS = 80
STANDARD_MI_CUE_SEC = 2.0
MIN_MI_CUE_SEC_FOR_STANDARD_VISUAL = 2.0
MIN_MI_CUE_SEC_FOR_NATURAL_AUDIO = 2.0
MIN_MI_ITI_SEC_BETWEEN_TRIALS = 2.0
PHASE_LABELS = {
    "idle": "等待开始",
    "baseline": "准备阶段",
    "cue": "任务提示",
    "imagery_ready": "开始音",
    "imagery": "想象执行",
    "iti": "休息恢复",
    "paused": "已暂停",
    "calibration_open": "睁眼静息",
    "calibration_closed": "闭眼静息",
    "calibration_eye_move": "眼动校准",
    "calibration_blink": "眨眼伪迹",
    "calibration_swallow": "吞咽伪迹",
    "calibration_jaw": "咬牙伪迹",
    "calibration_head": "头动伪迹",
    "practice": "想象训练",
    "run_rest": "轮次间休息",
    "idle_block": "无控制",
    "continuous": "连续仿真",
}
PHASE_ACCENT_COLORS = {
    "idle": "#64748B",
    "baseline": "#0F766E",
    "cue": "#2563EB",
    "imagery_ready": "#7C3AED",
    "imagery": "#8B5CF6",
    "iti": "#D97706",
    "paused": "#475569",
    "calibration_open": "#0F766E",
    "calibration_closed": "#0369A1",
    "calibration_eye_move": "#7C3AED",
    "calibration_blink": "#B45309",
    "calibration_swallow": "#B91C1C",
    "calibration_jaw": "#BE123C",
    "calibration_head": "#C2410C",
    "practice": "#4F46E5",
    "run_rest": "#B45309",
    "idle_block": "#0D9488",
    "continuous": "#0284C7",
}
PHASE_BACKGROUND_COLORS = {
    "idle": ("#E2E8F0", "#CBD5E1"),
    "baseline": ("#DCFCE7", "#BBF7D0"),
    "cue": ("#DBEAFE", "#BFDBFE"),
    "imagery_ready": ("#EDE9FE", "#DDD6FE"),
    "imagery": ("#F3E8FF", "#E9D5FF"),
    "iti": ("#FEF3C7", "#FDE68A"),
    "paused": ("#E5E7EB", "#CBD5E1"),
    "calibration_open": ("#DCFCE7", "#BBF7D0"),
    "calibration_closed": ("#DBEAFE", "#BFDBFE"),
    "calibration_eye_move": ("#F3E8FF", "#E9D5FF"),
    "calibration_blink": ("#FEF3C7", "#FDE68A"),
    "calibration_swallow": ("#FEE2E2", "#FECACA"),
    "calibration_jaw": ("#FCE7F3", "#FBCFE8"),
    "calibration_head": ("#FFEDD5", "#FED7AA"),
    "practice": ("#EDE9FE", "#DDD6FE"),
    "run_rest": ("#FEF3C7", "#FDE68A"),
    "idle_block": ("#CCFBF1", "#99F6E4"),
    "continuous": ("#E0F2FE", "#BAE6FD"),
}
CUE_ASSET_DIR = PROJECT_ROOT / "code" / "collection" / "assets" / "cues"
CUE_IMAGE_FILES = {
    "left_hand": "left_hand.png",
    "right_hand": "right_hand.jpg",
    "feet": "feet.png",
    "tongue": "tongue.png",
}
CLASS_UI_NAMES = {
    "left_hand": "左手",
    "right_hand": "右手",
    "feet": "双脚",
    "tongue": "舌头",
}
CLASS_UI_IMAGERY_HINTS = {
    "left_hand": "持续想象左手握拳与放松的动作，不要真实移动手臂。",
    "right_hand": "持续想象右手握拳与放松的动作，不要真实移动手臂。",
    "feet": "持续想象双脚交替踩踏或抬脚动作，不要真实移动腿部。",
    "tongue": "持续想象舌头前伸/上抬动作，不要真实张口或吐舌。",
}

DEFAULT_CONFIG = {
    "serial_port": "",
    "board_id": BoardIds.CYTON_BOARD.value,
    "subject_id": "001",
    "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "output_root": str(DEFAULT_OUTPUT_ROOT),
    "channel_names": ",".join(DEFAULT_CHANNEL_NAMES),
    "channel_positions": "0,1,2,3,4,5,6,7",
    "trials_per_class": 10,
    "baseline_sec": 2.0,
    "cue_sec": STANDARD_MI_CUE_SEC,
    "imagery_sec": 4.0,
    "iti_sec": 2.0,
    "run_count": 3,
    "max_consecutive_same_class": 2,
    "run_rest_sec": 60.0,
    "long_run_rest_every": 2,
    "long_run_rest_sec": 120.0,
    "quality_check_sec": 0.0,
    "practice_sec": 0.0,
    "calibration_open_sec": 60.0,
    "calibration_closed_sec": 60.0,
    "calibration_eye_sec": 30.0,
    "calibration_blink_sec": 20.0,
    "calibration_swallow_sec": 20.0,
    "calibration_jaw_sec": 20.0,
    "calibration_head_sec": 20.0,
    "idle_block_count": 2,
    "idle_block_sec": 60.0,
    "idle_prepare_block_count": 0,
    "idle_prepare_sec": 0.0,
    "continuous_block_count": 2,
    "continuous_block_sec": 240.0,
    "continuous_command_min_sec": 4.0,
    "continuous_command_max_sec": 5.0,
    "continuous_gap_min_sec": 2.0,
    "continuous_gap_max_sec": 3.0,
    "include_eyes_closed_rest_in_gate_neg": False,
    "artifact_types": ",".join(DEFAULT_ARTIFACT_TYPES),
    "reference_mode": "",
    "participant_state": "normal",
    "caffeine_intake": "unknown",
    "recent_exercise": "unknown",
    "sleep_note": "",
    "random_seed": 0,
    "use_separate_participant_screen": True,
    "training_cue_voice_enabled": False,
    "notes": "",
}

PARTICIPANT_STATE_OPTIONS = [
    ("normal", "正常"),
    ("tired", "疲劳"),
    ("excited", "兴奋"),
]
CAFFEINE_OPTIONS = [
    ("unknown", "未知"),
    ("none", "无"),
    ("tea", "茶"),
    ("coffee", "咖啡"),
]
RECENT_EXERCISE_OPTIONS = [
    ("unknown", "未知"),
    ("no", "无"),
    ("yes", "有"),
]


def supported_collection_board_names() -> list[str]:
    """Return board presets that satisfy the fixed 8-channel MI collection layout."""
    return [
        "CYTON_BOARD",
        "CYTON_DAISY_BOARD",
        "SYNTHETIC_BOARD",
    ]


def supported_collection_board_ids() -> set[int]:
    """Return valid board ids for the current collection workflow."""
    supported_ids: set[int] = set()
    for name in supported_collection_board_names():
        board = getattr(BoardIds, name, None)
        if board is not None:
            supported_ids.add(int(board.value))
    return supported_ids


def available_board_options() -> list[tuple[str, int]]:
    """Return a short list of common BrainFlow board presets."""
    options = []
    label_map = {
        "CYTON_BOARD": "Cyton（8 通道）",
        "CYTON_DAISY_BOARD": "Cyton Daisy（仅前 8 通道）",
        "SYNTHETIC_BOARD": "模拟板卡（演示模式）",
    }
    candidate_names = supported_collection_board_names()
    for name in candidate_names:
        board = getattr(BoardIds, name, None)
        if board is None:
            continue
        label = label_map.get(name, name.replace("_BOARD", "").replace("_", " ").title())
        options.append((label, int(board.value)))
    return options


class CueIllustrationWidget(QWidget):
    """Large central cue card used to guide the participant."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.phase = "idle"
        self.class_name: str | None = None
        self.title = "等待开始"
        self.subtitle = "连接设备后开始采集"
        self.cue_images = self._load_cue_images()
        self.setMinimumHeight(430)

    def set_state(self, *, phase: str, class_name: str | None, title: str, subtitle: str) -> None:
        self.phase = phase
        self.class_name = class_name
        self.title = title
        self.subtitle = subtitle
        self.update()

    def _resolve_accent(self) -> QColor:
        if self.class_name in CLASS_LOOKUP and self.phase in {"cue", "imagery"}:
            return QColor(str(CLASS_LOOKUP[self.class_name]["color"]))
        return QColor(PHASE_ACCENT_COLORS.get(self.phase, "#64748B"))

    @staticmethod
    def _wrapped_text_height(text: str, font: QFont, width: float) -> float:
        if not str(text).strip():
            return 0.0
        metrics = QFontMetrics(font)
        rect = metrics.boundingRect(
            QRect(0, 0, max(1, int(width)), 2000),
            int(Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap),
            text,
        )
        return float(rect.height())

    @staticmethod
    def _fit_rect(container: QRectF, image_width: float, image_height: float) -> QRectF:
        if image_width <= 0 or image_height <= 0:
            return QRectF(container)
        container_ratio = container.width() / max(container.height(), 1.0)
        image_ratio = image_width / max(image_height, 1.0)
        if container_ratio > image_ratio:
            height = container.height()
            width = height * image_ratio
            x = container.center().x() - width / 2.0
            return QRectF(x, container.top(), width, height)
        width = container.width()
        height = width / image_ratio
        y = container.center().y() - height / 2.0
        return QRectF(container.left(), y, width, height)

    def _load_cue_images(self) -> dict[str, QImage]:
        loaded: dict[str, QImage] = {}
        for class_name, filename in CUE_IMAGE_FILES.items():
            path = CUE_ASSET_DIR / filename
            if not path.exists():
                continue
            image = QImage(str(path))
            if image.isNull():
                continue
            loaded[class_name] = image
        return loaded

    def _draw_cue_image(self, painter: QPainter, rect: QRectF, class_name: str, accent: QColor) -> bool:
        image = self.cue_images.get(class_name)
        if image is None:
            return False

        frame = rect.adjusted(8, 6, -8, -6)
        painter.setPen(QPen(QColor("#D6DEE8"), 2))
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawRoundedRect(frame, 24, 24)

        inner = frame.adjusted(18, 18, -18, -18)
        target = self._fit_rect(inner, float(image.width()), float(image.height()))
        painter.drawImage(target, image)

        if self.phase in {"cue", "imagery"}:
            painter.setPen(QPen(accent, 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(frame.adjusted(2, 2, -2, -2), 22, 22)
        return True

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        bg_top, bg_bottom = PHASE_BACKGROUND_COLORS.get(self.phase, ("#E2E8F0", "#CBD5E1"))
        canvas = QLinearGradient(0.0, 0.0, 0.0, float(self.height()))
        canvas.setColorAt(0.0, QColor(bg_top))
        canvas.setColorAt(1.0, QColor(bg_bottom))
        painter.fillRect(self.rect(), QBrush(canvas))

        card = self.rect().adjusted(20, 20, -20, -20)
        accent = self._resolve_accent()
        background = QColor("#FFFFFF")
        if self.phase in {"idle", "paused"}:
            background = QColor("#F8FAFC")

        painter.setPen(QPen(QColor("#C5CFDB"), 2))
        painter.setBrush(background)
        painter.drawRoundedRect(card, 28, 28)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(accent))
        painter.drawRoundedRect(QRectF(card.left() + 30, card.top() + 24, card.width() - 60, 12), 6, 6)

        text_left = card.left() + 36
        text_width = card.width() - 72
        title_top = card.top() + 52
        title_font = QFont("Microsoft YaHei", 24, QFont.Bold)
        subtitle_font = QFont("Microsoft YaHei", 13)
        title_height = max(44.0, self._wrapped_text_height(self.title, title_font, text_width) + 8.0)
        title_rect = QRectF(text_left, title_top, text_width, title_height)
        painter.setPen(QColor("#0F172A"))
        painter.setFont(title_font)
        painter.drawText(title_rect, Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, self.title)

        subtitle_top = title_rect.bottom() + 10.0
        subtitle_height = self._wrapped_text_height(self.subtitle, subtitle_font, text_width)
        subtitle_rect = QRectF(text_left, subtitle_top, text_width, max(0.0, subtitle_height + 4.0))
        painter.setPen(QColor("#475569"))
        painter.setFont(subtitle_font)
        if subtitle_height > 0:
            painter.drawText(subtitle_rect, Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, self.subtitle)

        drawing_top = max(card.top() + 188.0, subtitle_rect.bottom() + 24.0 if subtitle_height > 0 else title_rect.bottom() + 24.0)
        drawing_bottom = card.bottom() - 36.0
        drawing_rect = QRectF(card.left() + 60, drawing_top, card.width() - 120, max(120.0, drawing_bottom - drawing_top))
        if self.phase in {"idle", "baseline", "imagery_ready", "imagery", "iti", "paused"} or self.class_name is None:
            self._draw_fixation(painter, drawing_rect, accent)
            return

        if self._draw_cue_image(painter, drawing_rect, self.class_name, accent):
            return

        if self.class_name == "left_hand":
            self._draw_hand(painter, drawing_rect, accent, mirrored=False)
        elif self.class_name == "right_hand":
            self._draw_hand(painter, drawing_rect, accent, mirrored=True)
        elif self.class_name == "feet":
            self._draw_feet(painter, drawing_rect, accent)
        elif self.class_name == "tongue":
            self._draw_tongue(painter, drawing_rect, accent)

    @staticmethod
    def _draw_fixation(painter: QPainter, rect: QRectF, color: QColor) -> None:
        center = rect.center()
        painter.setPen(QPen(color, 8, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(center.x() - 44, center.y()), QPointF(center.x() + 44, center.y()))
        painter.drawLine(QPointF(center.x(), center.y() - 44), QPointF(center.x(), center.y() + 44))

    @staticmethod
    def _draw_hand(painter: QPainter, rect: QRectF, color: QColor, *, mirrored: bool) -> None:
        width = rect.width()
        height = rect.height()
        palm = QRectF(rect.left() + width * 0.33, rect.top() + height * 0.34, width * 0.22, height * 0.28)
        if mirrored:
            palm.moveLeft(rect.right() - palm.width() - (palm.left() - rect.left()))

        painter.setPen(QPen(color, 6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(palm, 24, 24)

        finger_spacing = palm.width() / 5.0
        for finger in range(5):
            x = palm.left() + finger_spacing * (finger + 0.5)
            painter.drawLine(QPointF(x, palm.top() + 2), QPointF(x, palm.top() - height * 0.16))

        if mirrored:
            thumb = QPainterPath(QPointF(palm.right() - 2, palm.top() + palm.height() * 0.34))
            thumb.lineTo(palm.right() + width * 0.10, palm.top() + palm.height() * 0.48)
            thumb.lineTo(palm.right() + width * 0.04, palm.top() + palm.height() * 0.70)
        else:
            thumb = QPainterPath(QPointF(palm.left() + 2, palm.top() + palm.height() * 0.34))
            thumb.lineTo(palm.left() - width * 0.10, palm.top() + palm.height() * 0.48)
            thumb.lineTo(palm.left() - width * 0.04, palm.top() + palm.height() * 0.70)
        painter.drawPath(thumb)

    @staticmethod
    def _draw_feet(painter: QPainter, rect: QRectF, color: QColor) -> None:
        painter.setPen(QPen(color, 6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)
        left = QRectF(rect.left() + rect.width() * 0.23, rect.top() + rect.height() * 0.24, rect.width() * 0.20, rect.height() * 0.44)
        right = QRectF(rect.left() + rect.width() * 0.57, rect.top() + rect.height() * 0.24, rect.width() * 0.20, rect.height() * 0.44)
        for foot in (left, right):
            painter.drawRoundedRect(foot, 26, 26)
            toe_radius = foot.width() * 0.08
            toe_y = foot.top() - toe_radius * 0.4
            for toe_index in range(5):
                toe_x = foot.left() + foot.width() * (0.16 + toe_index * 0.17)
                painter.drawEllipse(QPointF(toe_x, toe_y), toe_radius, toe_radius)

    @staticmethod
    def _draw_tongue(painter: QPainter, rect: QRectF, color: QColor) -> None:
        painter.setPen(QPen(color, 6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)
        face = QPainterPath()
        face.moveTo(rect.left() + rect.width() * 0.28, rect.top() + rect.height() * 0.24)
        face.cubicTo(
            rect.left() + rect.width() * 0.58,
            rect.top() + rect.height() * 0.08,
            rect.left() + rect.width() * 0.78,
            rect.top() + rect.height() * 0.30,
            rect.left() + rect.width() * 0.70,
            rect.top() + rect.height() * 0.54,
        )
        face.cubicTo(
            rect.left() + rect.width() * 0.64,
            rect.top() + rect.height() * 0.80,
            rect.left() + rect.width() * 0.38,
            rect.top() + rect.height() * 0.84,
            rect.left() + rect.width() * 0.28,
            rect.top() + rect.height() * 0.58,
        )
        painter.drawPath(face)
        painter.drawLine(
            QPointF(rect.left() + rect.width() * 0.50, rect.top() + rect.height() * 0.44),
            QPointF(rect.left() + rect.width() * 0.66, rect.top() + rect.height() * 0.47),
        )
        tongue = QPainterPath(QPointF(rect.left() + rect.width() * 0.66, rect.top() + rect.height() * 0.47))
        tongue.cubicTo(
            rect.left() + rect.width() * 0.80,
            rect.top() + rect.height() * 0.50,
            rect.left() + rect.width() * 0.80,
            rect.top() + rect.height() * 0.64,
            rect.left() + rect.width() * 0.68,
            rect.top() + rect.height() * 0.64,
        )
        painter.drawPath(tongue)


class ParticipantDisplayWindow(QWidget):
    """Full-screen participant-facing prompt window."""

    pause_requested = pyqtSignal()
    mark_bad_requested = pyqtSignal()
    advance_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__(None, Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("受试者提示屏")
        self.setStyleSheet("background: #04111C;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 36, 40, 36)
        layout.setSpacing(18)

        self.stage_label = QLabel("等待开始")
        self.stage_label.setAlignment(Qt.AlignCenter)
        self.stage_label.setStyleSheet(
            "color: white; font-size: 28px; font-weight: bold; "
            "background: #475569; border-radius: 18px; padding: 12px 20px;"
        )
        layout.addWidget(self.stage_label)

        self.cue_widget = CueIllustrationWidget()
        self.cue_widget.setMinimumHeight(360)
        layout.addWidget(self.cue_widget, stretch=1)

        self.countdown_label = QLabel("--")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet(
            "color: #F8FAFC; font-size: 56px; font-weight: bold; "
            "background: rgba(15, 23, 42, 170); border: 2px solid #334155; border-radius: 22px; padding: 10px 20px;"
        )
        layout.addWidget(self.countdown_label)

        self.hint_label = QLabel("空格 暂停/继续    B 标记坏试次/命令失败    N 提前结束训练    Esc 停止并保存")
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.hint_label.setStyleSheet("color: #94A3B8; font-size: 15px;")
        layout.addWidget(self.hint_label)

    @staticmethod
    def _resolve_phase_accent(phase: str, class_name: str | None) -> str:
        if class_name in CLASS_LOOKUP and phase in {"cue", "imagery"}:
            return str(CLASS_LOOKUP[class_name]["color"])
        return PHASE_ACCENT_COLORS.get(phase, "#64748B")

    def set_prompt(
        self,
        *,
        phase: str,
        class_name: str | None,
        stage_text: str,
        title: str,
        subtitle: str,
        countdown_text: str,
    ) -> None:
        accent = self._resolve_phase_accent(phase, class_name)
        self.stage_label.setText(stage_text)
        self.stage_label.setStyleSheet(
            f"color: white; font-size: 28px; font-weight: bold; "
            f"background: {accent}; border-radius: 18px; padding: 12px 20px;"
        )
        self.cue_widget.set_state(
            phase=phase,
            class_name=class_name,
            title=title,
            subtitle=subtitle,
        )
        self.countdown_label.setText(countdown_text)
        self.countdown_label.setStyleSheet(
            f"color: #F8FAFC; font-size: 56px; font-weight: bold; "
            f"background: rgba(15, 23, 42, 170); border: 2px solid {accent}; border-radius: 22px; padding: 10px 20px;"
        )

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Space:
            self.pause_requested.emit()
            event.accept()
            return
        if event.key() == Qt.Key_B:
            self.mark_bad_requested.emit()
            event.accept()
            return
        if event.key() == Qt.Key_N:
            self.advance_requested.emit()
            event.accept()
            return
        if event.key() == Qt.Key_Escape:
            self.stop_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_requested.emit()
        event.ignore()


def build_fbcca_style_display_signal(
    y_raw: np.ndarray,
    sampling_rate: float,
    *,
    baseline_hz: float = 3.0,
    baseline_order: int = 1,
) -> np.ndarray:
    """Display-only preprocessing used for operator preview.

    This never mutates the source buffer and is not part of the saved-data path.
    """

    y = np.asarray(y_raw, dtype=np.float64)
    if y.size <= 10 or float(sampling_rate) <= 1.0:
        return y.copy()

    y_plot = y.copy()
    baseline = y_plot.copy()

    try:
        DataFilter.perform_lowpass(
            baseline,
            float(sampling_rate),
            float(baseline_hz),
            int(baseline_order),
            FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
            0,
        )
        y_plot = y_plot - baseline
    except Exception:
        pass

    try:
        DataFilter.remove_environmental_noise(y_plot, float(sampling_rate), NoiseTypes.FIFTY.value)
    except Exception:
        pass

    y_plot = y_plot - float(np.mean(y_plot))
    return y_plot


class RealtimeEEGPreviewWidget(QWidget):
    """Operator-facing EEG/impedance preview for connection quality checks."""

    MODE_EEG = "EEG"
    MODE_IMPEDANCE = "IMP"
    LEAD_OFF_CURRENT_AMPS = 6e-9
    SERIES_RESISTOR_OHMS = 2200.0
    DISPLAY_BASELINE_HZ = 3.0
    DISPLAY_BASELINE_ORDER = 1

    CHANNEL_COLORS = [
        "#38BDF8",
        "#22C55E",
        "#F59E0B",
        "#A855F7",
        "#EF4444",
        "#14B8A6",
        "#EAB308",
        "#F97316",
    ]

    def __init__(self, parent: QWidget | None = None, *, window_seconds: float = 5.0) -> None:
        super().__init__(parent)
        self.window_seconds = max(1.0, float(window_seconds))
        self.sampling_rate = 0.0
        self.channel_names: list[str] = []
        self.max_points = 0
        self.buffers: list[deque[float]] = []
        self.last_chunk_perf = 0.0
        self.last_repaint_perf = 0.0
        self.mode = self.MODE_EEG
        self.impedance_channel = 1
        self.last_impedance_ohms: list[float | None] = []
        self.placeholder_text = f"连接设备后显示最近 {self.window_seconds:.1f} 秒波形。"
        self._dirty = False

        self.setMinimumHeight(520)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #08111F; border: 1px solid #1E293B; border-radius: 14px;")

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(50)
        self.refresh_timer.timeout.connect(self._flush_repaint)
        self.refresh_timer.start()

    def configure_stream(self, *, sampling_rate: float, channel_names: list[str]) -> None:
        self.sampling_rate = max(1.0, float(sampling_rate))
        self.channel_names = [str(name) for name in channel_names]
        self.max_points = max(32, int(round(self.window_seconds * self.sampling_rate)))
        self.buffers = [deque(maxlen=self.max_points) for _ in self.channel_names]
        self.last_impedance_ohms = [None] * len(self.channel_names)
        self.mode = self.MODE_EEG
        self.impedance_channel = 1
        self.last_chunk_perf = 0.0
        self.last_repaint_perf = 0.0
        self.placeholder_text = "正在等待实时数据..."
        self._dirty = True
        self.update()

    def clear_stream(self, message: str | None = None) -> None:
        self.buffers = []
        self.channel_names = []
        self.max_points = 0
        self.sampling_rate = 0.0
        self.last_chunk_perf = 0.0
        self.last_repaint_perf = 0.0
        self.mode = self.MODE_EEG
        self.impedance_channel = 1
        self.last_impedance_ohms = []
        if message:
            self.placeholder_text = str(message)
        else:
            self.placeholder_text = f"连接设备后显示最近 {self.window_seconds:.1f} 秒波形。"
        self._dirty = True
        self.update()

    def set_quality_mode(self, mode: str, *, impedance_channel: int | None = None) -> None:
        normalized = self.MODE_IMPEDANCE if str(mode).strip().upper().startswith("IMP") else self.MODE_EEG
        self.mode = normalized
        if impedance_channel is not None:
            self.set_impedance_channel(int(impedance_channel))
        else:
            self.set_impedance_channel(int(self.impedance_channel))
        if self.mode == self.MODE_IMPEDANCE:
            self._update_impedance_values()
        self._dirty = True
        self.update()

    def set_impedance_channel(self, channel: int) -> int:
        channel_count = len(self.channel_names)
        if channel_count <= 0:
            self.impedance_channel = 1
            return self.impedance_channel
        self.impedance_channel = int(np.clip(int(channel), 1, channel_count))
        self._dirty = True
        return self.impedance_channel

    def append_chunk(self, eeg_chunk: np.ndarray) -> None:
        if not self.buffers:
            return

        chunk = np.asarray(eeg_chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[0] != len(self.buffers) or chunk.shape[1] == 0:
            return

        for channel_index, buffer in enumerate(self.buffers):
            buffer.extend(np.asarray(chunk[channel_index], dtype=np.float32).tolist())

        if self.mode == self.MODE_IMPEDANCE:
            self._update_impedance_values()

        self.last_chunk_perf = time.perf_counter()
        self._dirty = True

    def _flush_repaint(self) -> None:
        now = time.perf_counter()
        if not self._dirty and not (self.channel_names and now - self.last_repaint_perf >= 0.25):
            return
        self._dirty = False
        self.last_repaint_perf = now
        self.update()

    def _estimate_impedance_ohms_from_raw_window(self, y_uvolts: np.ndarray) -> float | None:
        if y_uvolts is None or y_uvolts.size < 20:
            return None
        std_uvolts = float(np.std(y_uvolts, ddof=0))
        z_ohm = (np.sqrt(2.0) * std_uvolts * 1e-6) / float(self.LEAD_OFF_CURRENT_AMPS)
        z_ohm -= float(self.SERIES_RESISTOR_OHMS)
        if not np.isfinite(z_ohm):
            return None
        return max(z_ohm, 0.0)

    @staticmethod
    def _format_impedance_value(z_ohm: float | None) -> str:
        if z_ohm is None or not np.isfinite(float(z_ohm)):
            return "Imp --"
        value = float(max(0.0, z_ohm))
        if value >= 1_000_000.0:
            return f"Imp {value / 1_000_000.0:.2f}M"
        if value >= 1_000.0:
            return f"Imp {value / 1_000.0:.1f}k"
        return f"Imp {value:.0f}"

    def _update_impedance_values(self) -> None:
        if not self.buffers or not self.last_impedance_ohms:
            return
        channel_count = min(len(self.buffers), len(self.last_impedance_ohms))
        for channel_index in range(channel_count):
            y_raw = np.asarray(self.buffers[channel_index], dtype=np.float64)
            self.last_impedance_ohms[channel_index] = self._estimate_impedance_ohms_from_raw_window(y_raw)

    def _build_plot_signal(self, y_raw: np.ndarray) -> np.ndarray:
        y = np.asarray(y_raw, dtype=np.float64)
        if y.size <= 10 or self.mode != self.MODE_EEG or self.sampling_rate <= 1.0:
            return y.copy()

        return build_fbcca_style_display_signal(
            y,
            self.sampling_rate,
            baseline_hz=float(self.DISPLAY_BASELINE_HZ),
            baseline_order=int(self.DISPLAY_BASELINE_ORDER),
        )

    def _draw_placeholder(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QColor("#CBD5E1"))
        placeholder_font = QFont("Microsoft YaHei", max(11, min(15, int(self.height() * 0.022))))
        painter.setFont(placeholder_font)
        painter.drawText(rect, Qt.AlignCenter | Qt.TextWordWrap, self.placeholder_text)

    def _layout_metrics(self, *, channel_count_hint: int | None = None) -> dict[str, object]:
        channel_count = max(1, int(channel_count_hint or len(self.channel_names) or 1))
        outer_margin = 10.0 if self.width() < 760 else 12.0
        inner_margin_x = 12.0 if self.width() < 760 else 14.0
        outer_rect = QRectF(self.rect()).adjusted(outer_margin, outer_margin, -outer_margin, -outer_margin)
        has_stream = bool(self.channel_names) and self.sampling_rate > 0.0
        header_narrow = outer_rect.width() < 590.0
        header_height = 24.0 if has_stream else 32.0
        header_rect = QRectF(
            outer_rect.left() + inner_margin_x,
            outer_rect.top() + 8.0,
            max(120.0, outer_rect.width() - inner_margin_x * 2.0),
            header_height,
        )
        content_top = header_rect.bottom() + (10.0 if has_stream else 8.0)
        content_rect = QRectF(
            outer_rect.left() + inner_margin_x,
            content_top,
            max(120.0, outer_rect.width() - inner_margin_x * 2.0),
            max(24.0, outer_rect.bottom() - content_top - 10.0),
        )

        if content_rect.height() >= 860.0:
            row_gap = 12.0 if channel_count <= 4 else 10.0 if channel_count <= 8 else 8.0
        elif content_rect.height() >= 680.0:
            row_gap = 10.0 if channel_count <= 4 else 8.0 if channel_count <= 8 else 6.0
        elif content_rect.height() >= 520.0:
            row_gap = 8.0 if channel_count <= 4 else 6.0 if channel_count <= 8 else 5.0
        else:
            row_gap = 6.0 if channel_count <= 4 else 5.0 if channel_count <= 6 else 4.0

        available_row_height = (content_rect.height() - row_gap * max(0, channel_count - 1)) / float(channel_count)
        if available_row_height < 24.0 and channel_count > 1:
            row_gap = 3.0 if content_rect.height() >= channel_count * 24.0 else 2.0
            available_row_height = (content_rect.height() - row_gap * (channel_count - 1)) / float(channel_count)

        if channel_count <= 2:
            target_row_height = 168.0
        elif channel_count <= 4:
            target_row_height = 132.0
        elif channel_count <= 8:
            target_row_height = 104.0
        elif channel_count <= 12:
            target_row_height = 82.0
        elif channel_count <= 16:
            target_row_height = 68.0
        else:
            target_row_height = 54.0
        row_height = max(22.0, min(target_row_height, available_row_height))

        label_width = 76.0 if content_rect.width() >= 920.0 else 70.0 if content_rect.width() >= 760.0 else 60.0 if content_rect.width() >= 640.0 else 54.0
        if channel_count >= 8:
            label_width = max(54.0, label_width - 2.0)

        right_info_width = 116.0 if self.mode == self.MODE_IMPEDANCE else 102.0
        if content_rect.width() >= 920.0:
            right_info_width += 12.0
        elif content_rect.width() >= 780.0:
            right_info_width += 6.0
        elif content_rect.width() < 640.0:
            right_info_width -= 8.0

        return {
            "outer_rect": outer_rect,
            "header_rect": header_rect,
            "content_rect": content_rect,
            "header_narrow": header_narrow,
            "row_gap": row_gap,
            "row_height": row_height,
            "label_width": label_width,
            "right_info_width": max(76.0, right_info_width),
            "plot_padding_y": 3.0 if row_height < 34.0 else 5.0 if row_height < 84.0 else 7.0,
        }

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        if self.mode == self.MODE_IMPEDANCE:
            self._update_impedance_values()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#08111F"))

        metrics = self._layout_metrics()
        outer_rect = metrics["outer_rect"]
        header_rect = metrics["header_rect"]
        content_rect = metrics["content_rect"]
        header_narrow = bool(metrics["header_narrow"])
        row_gap = float(metrics["row_gap"])
        row_height = float(metrics["row_height"])
        label_width = float(metrics["label_width"])
        right_info_width = float(metrics["right_info_width"])
        plot_padding_y = float(metrics["plot_padding_y"])
        painter.setPen(QPen(QColor("#22314A"), 1.0))
        painter.setBrush(QColor("#0B1727"))
        painter.drawRoundedRect(outer_rect, 14, 14)

        meta_rect = QRectF(header_rect.left(), header_rect.top(), header_rect.width(), header_rect.height())

        if self.channel_names and self.sampling_rate > 0:
            freshness_sec = 0.0 if self.last_chunk_perf <= 0 else max(0.0, time.perf_counter() - self.last_chunk_perf)
            freshness_font = QFont("Consolas", max(9, min(12, int(self.height() * 0.017))))
            freshness_rect = QRectF(meta_rect.right() - 126.0, meta_rect.top(), 126.0, meta_rect.height())
            painter.setPen(QColor("#F59E0B" if freshness_sec > 1.0 else "#94A3B8"))
            painter.setFont(freshness_font)
            painter.drawText(freshness_rect, Qt.AlignRight | Qt.AlignVCenter, f"更新 {freshness_sec:.2f}s")
        else:
            painter.setPen(QColor("#E2E8F0"))
            painter.setFont(QFont("Microsoft YaHei", max(10, min(14, int(self.height() * 0.02))), QFont.Bold))
            painter.drawText(header_rect, Qt.AlignLeft | Qt.AlignVCenter, "EEG / 阻抗预览")

        if not self.channel_names or not self.buffers:
            self._draw_placeholder(painter, content_rect)
            return

        channel_count = len(self.channel_names)

        for channel_index, channel_name in enumerate(self.channel_names):
            row_top = content_rect.top() + channel_index * (row_height + row_gap)
            row_rect = QRectF(content_rect.left(), row_top, content_rect.width(), row_height)
            plot_rect = QRectF(
                row_rect.left() + label_width,
                row_rect.top() + plot_padding_y,
                max(10.0, row_rect.width() - label_width - right_info_width),
                max(16.0, row_rect.height() - plot_padding_y * 2.0),
            )

            impedance_mode = self.mode == self.MODE_IMPEDANCE
            row_fill = QColor("#1F2937" if impedance_mode else "#0E1B2D")
            painter.setPen(Qt.NoPen)
            painter.setBrush(row_fill)
            painter.drawRoundedRect(row_rect, 10, 10)

            painter.setPen(QColor("#E2E8F0" if impedance_mode else "#94A3B8"))
            channel_font_size = max(9, min(12, int(row_height * 0.34)))
            painter.setFont(QFont("Consolas", channel_font_size, QFont.Bold))
            painter.drawText(
                QRectF(row_rect.left() + 8, row_rect.top(), label_width - 10, row_rect.height()),
                Qt.AlignLeft | Qt.AlignVCenter,
                channel_name,
            )

            grid_pen = QPen(QColor("#1E293B"), 1.0)
            painter.setPen(grid_pen)
            for grid_index in range(5):
                x = plot_rect.left() + plot_rect.width() * grid_index / 4.0
                painter.drawLine(QPointF(x, plot_rect.top()), QPointF(x, plot_rect.bottom()))
            painter.drawLine(
                QPointF(plot_rect.left(), plot_rect.center().y()),
                QPointF(plot_rect.right(), plot_rect.center().y()),
            )

            y_raw = np.asarray(self.buffers[channel_index], dtype=np.float64)
            if y_raw.size < 2:
                painter.setPen(QColor("#64748B"))
                painter.setFont(QFont("Microsoft YaHei", max(9, min(11, int(row_height * 0.30)))))
                painter.drawText(plot_rect, Qt.AlignCenter, "等待数据...")
                continue

            y = self._build_plot_signal(y_raw)

            if y.size >= 20:
                low = float(np.percentile(y, 5))
                high = float(np.percentile(y, 95))
            else:
                low = float(np.min(y))
                high = float(np.max(y))

            if high - low < 20.0:
                center = (high + low) / 2.0
                low = center - 10.0
                high = center + 10.0

            pad = max(5.0, 0.2 * (high - low))
            y_min = low - pad
            y_max = high + pad
            span = max(1.0, y_max - y_min)

            if y_min <= 0.0 <= y_max:
                zero_y = plot_rect.bottom() - ((0.0 - y_min) / span) * plot_rect.height()
                painter.setPen(QPen(QColor("#334155"), 1.0, Qt.DashLine))
                painter.drawLine(QPointF(plot_rect.left(), zero_y), QPointF(plot_rect.right(), zero_y))

            waveform = QPainterPath()
            x_scale = plot_rect.width() / max(1, y.size - 1)
            for point_index, sample in enumerate(y):
                x = plot_rect.left() + point_index * x_scale
                y_pos = plot_rect.bottom() - ((float(sample) - y_min) / span) * plot_rect.height()
                if point_index == 0:
                    waveform.moveTo(x, y_pos)
                else:
                    waveform.lineTo(x, y_pos)

            color = QColor(self.CHANNEL_COLORS[channel_index % len(self.CHANNEL_COLORS)])
            painter.setPen(
                QPen(
                    color,
                    1.5 if impedance_mode else 1.2,
                    Qt.SolidLine,
                    Qt.RoundCap,
                    Qt.RoundJoin,
                )
            )
            painter.drawPath(waveform)

            info_text = ""
            info_color = QColor("#64748B")
            if impedance_mode:
                z_ohm = None
                if channel_index < len(self.last_impedance_ohms):
                    z_ohm = self.last_impedance_ohms[channel_index]
                if z_ohm is None:
                    info_text = "Imp --"
                    info_color = QColor("#94A3B8")
                else:
                    info_text = self._format_impedance_value(z_ohm)
                    info_color = QColor("#FDE68A")
            else:
                info_text = f"P2P {np.ptp(y_raw):.0f}uV"

            painter.setPen(info_color)
            info_font_size = max(8, min(11, int(row_height * 0.31)))
            painter.setFont(QFont("Consolas", info_font_size))
            painter.drawText(
                QRectF(plot_rect.right() + 6, row_rect.top(), right_info_width - 6, row_rect.height()),
                Qt.AlignLeft | Qt.AlignVCenter,
                info_text,
            )


class BoardCaptureWorker(QObject):
    """Background BrainFlow worker used only for recording, not analysis."""

    MODE_EEG = "EEG"
    MODE_IMPEDANCE = "IMP"

    connection_ready = pyqtSignal(object)
    preview_data_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    session_data_ready = pyqtSignal(object)
    quality_mode_switch_finished = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(
        self,
        *,
        board_id: int,
        serial_port: str,
        channel_positions: list[int],
        channel_names: list[str],
        poll_interval_sec: float = 0.05,
    ) -> None:
        super().__init__()
        self.board_id = int(board_id)
        self.serial_port = serial_port
        self.channel_positions = list(channel_positions)
        self.channel_names = list(channel_names)
        self.poll_interval_sec = float(poll_interval_sec)
        self.stop_event = threading.Event()
        self.board_lock = threading.Lock()
        self.board: BoardShim | None = None
        self.selected_rows: list[int] = []
        self.marker_row: int | None = None
        self.timestamp_row: int | None = None
        self.package_num_row: int | None = None
        self.board_descr: dict[str, object] = {}
        self.sampling_rate: float | None = None
        self.current_quality_mode = self.MODE_EEG
        self._hw_impedance_channel: int | None = None
        self.sample_count_lock = threading.Lock()
        self.total_samples_captured = 0
        self.quality_mode_request_lock = threading.Lock()
        self._pending_quality_mode_request: dict[str, object] | None = None

    @staticmethod
    def build_impedance_command(channel: int, test_p: bool = True, test_n: bool = False) -> str:
        p = 1 if test_p else 0
        n = 1 if test_n else 0
        return f"z{int(channel)}{p}{n}Z"

    def supports_impedance_mode(self) -> bool:
        cyton_ids: set[int] = set()
        for board_name in ("CYTON_BOARD", "CYTON_DAISY_BOARD"):
            board_enum = getattr(BoardIds, board_name, None)
            if board_enum is not None:
                cyton_ids.add(int(board_enum.value))
        return int(self.board_id) in cyton_ids

    def _selected_channel_count(self) -> int:
        if self.selected_rows:
            return max(1, int(len(self.selected_rows)))
        if self.channel_names:
            return max(1, int(len(self.channel_names)))
        try:
            return max(1, int(len(BoardShim.get_eeg_channels(self.board_id))))
        except Exception:
            return 8

    def _marker_flush_delay_sec(self) -> float:
        if self.sampling_rate is None or float(self.sampling_rate) <= 0:
            return MARKER_FLUSH_FALLBACK_SEC
        sample_based_delay = MARKER_FLUSH_SAMPLE_COUNT / float(self.sampling_rate)
        return float(np.clip(sample_based_delay, MARKER_FLUSH_MIN_SEC, MARKER_FLUSH_MAX_SEC))

    @staticmethod
    def _is_transient_decode_error(error: Exception) -> bool:
        """Detect transient serial decode noise (for example OpenBCI packet byte 0xA0)."""
        text = str(error).strip().lower()
        return (
            ("utf-8" in text and "decode" in text and "invalid start byte" in text)
            or ("byte 0xa0" in text)
        )

    @staticmethod
    def _safe_stop_stream(board: BoardShim) -> None:
        """Best-effort stop to avoid stream-state exceptions during mode switches."""
        try:
            board.stop_stream()
        except Exception:
            pass

    @staticmethod
    def _concat_capture_chunks(chunks: list[np.ndarray]) -> np.ndarray:
        valid = [np.asarray(chunk, dtype=np.float64) for chunk in chunks if np.asarray(chunk).ndim == 2]
        valid = [chunk for chunk in valid if int(chunk.shape[1]) > 0]
        if not valid:
            raise ValueError("no BrainFlow samples were captured")
        row_count = int(valid[0].shape[0])
        if any(int(chunk.shape[0]) != row_count for chunk in valid):
            raise ValueError("captured BrainFlow chunks have inconsistent row counts")
        total_samples = int(sum(int(chunk.shape[1]) for chunk in valid))
        merged = np.empty((row_count, total_samples), dtype=np.float64)
        cursor = 0
        for chunk in valid:
            width = int(chunk.shape[1])
            merged[:, cursor : cursor + width] = chunk
            cursor += width
        return merged

    def _start_stream_with_retry(
        self,
        board: BoardShim,
        *,
        buffer_size: int = 450000,
        retries: int = 3,
        retry_delay_sec: float = 0.08,
    ) -> None:
        """Start stream with short retries for transient state races."""
        last_error: Exception | None = None
        for attempt in range(max(1, int(retries))):
            try:
                board.start_stream(int(buffer_size))
                return
            except Exception as error:
                last_error = error
                self._safe_stop_stream(board)
                time.sleep(float(retry_delay_sec) * float(attempt + 1))
        if last_error is not None:
            raise last_error

    def _config_board_with_retry(
        self,
        board: BoardShim,
        command: str,
        *,
        retries: int = 5,
        retry_delay_sec: float = 0.08,
    ) -> None:
        """Send one board command with retries for transient UTF-8 decode noise."""
        last_error: Exception | None = None
        normalized_command = str(command)
        for attempt in range(max(1, int(retries))):
            try:
                board.config_board(normalized_command)
                return
            except Exception as error:
                last_error = error
                if not self._is_transient_decode_error(error):
                    raise
                self._safe_stop_stream(board)
                try:
                    board.get_board_data()
                except Exception:
                    pass
                time.sleep(float(retry_delay_sec) * float(attempt + 1))
        if last_error is not None:
            raise RuntimeError(
                f"Failed command {normalized_command!r} after {int(retries)} retries: {last_error}"
            ) from last_error

    def _fast_switch_impedance_channel(self, board: BoardShim, target_channel: int) -> bool:
        previous_channel = self._hw_impedance_channel
        if previous_channel is None:
            return False
        if int(previous_channel) == int(target_channel):
            return True

        try:
            self._config_board_with_retry(
                board,
                self.build_impedance_command(int(previous_channel), False, False),
                retries=1,
            )
            self._config_board_with_retry(
                board,
                self.build_impedance_command(int(target_channel), True, False),
                retries=1,
            )
        except Exception:
            return False

        self._hw_impedance_channel = int(target_channel)
        return True

    def switch_quality_mode_sync(
        self,
        *,
        target_mode: str,
        target_channel: int = 1,
        reset_default: bool = False,
    ) -> tuple[bool, str]:
        mode = str(target_mode).strip().upper()
        if mode not in {self.MODE_EEG, self.MODE_IMPEDANCE}:
            return False, f"不支持的质量检查模式：{target_mode!r}"
        if mode == self.MODE_IMPEDANCE and not self.supports_impedance_mode():
            return False, "当前阻抗模式仅支持 Cyton / Cyton Daisy 板卡。"

        with self.board_lock:
            if self.board is None:
                return False, "设备尚未连接。"

            board = self.board
            channel_count = self._selected_channel_count()

            try:
                if mode == self.current_quality_mode and not bool(reset_default):
                    return True, ""

                self._safe_stop_stream(board)
                time.sleep(0.05)

                if self.supports_impedance_mode():
                    for ch in range(1, channel_count + 1):
                        self._config_board_with_retry(
                            board,
                            self.build_impedance_command(ch, False, False),
                        )
                    if bool(reset_default) or mode == self.MODE_EEG:
                        self._config_board_with_retry(board, "d")
                    if mode == self.MODE_IMPEDANCE:
                        for ch in range(1, channel_count + 1):
                            self._config_board_with_retry(
                                board,
                                self.build_impedance_command(ch, True, False),
                            )
                    self._hw_impedance_channel = None

                self._start_stream_with_retry(board, buffer_size=450000)
            except Exception as error:
                try:
                    self._start_stream_with_retry(board, buffer_size=450000, retries=1)
                except Exception:
                    pass
                return False, f"切换质量检查模式失败：{error}"

            self.current_quality_mode = mode

        if mode == self.MODE_IMPEDANCE:
            self.status_changed.emit("质量检查模式 -> 8通道阻抗")
        elif bool(reset_default):
            self.status_changed.emit("质量检查模式 -> EEG（恢复默认设置）")
        else:
            self.status_changed.emit("质量检查模式 -> EEG")
        return True, ""

    @pyqtSlot(str, int, bool)
    def request_quality_mode_switch(self, target_mode: str, target_channel: int, reset_default: bool) -> None:
        with self.quality_mode_request_lock:
            self._pending_quality_mode_request = {
                "target_mode": str(target_mode),
                "target_channel": int(target_channel),
                "reset_default": bool(reset_default),
            }

    def _build_connection_error_message(self, error: Exception) -> str:
        base_message = f"设备采集线程出错：{error}"
        serial_port = str(self.serial_port or "").strip()
        error_text = str(error)
        if not serial_port:
            return base_message
        if "UNABLE_TO_OPEN_PORT_ERROR" not in error_text and "unable to prepare streaming session" not in error_text.lower():
            return base_message

        diagnostics = describe_serial_port(serial_port)
        detected_ports = [str(item) for item in diagnostics.get("detected_ports", []) if str(item).strip()]
        windows_status = str(diagnostics.get("windows_status", "")).strip()
        problem_code = str(diagnostics.get("problem_code", "")).strip()
        problem_status = str(diagnostics.get("problem_status", "")).strip()
        detail_parts = [f"无法打开串口 {serial_port}。"]
        if windows_status:
            detail_parts.append(f"Windows Ports 状态：{windows_status}。")
        if problem_code:
            detail_parts.append(f"问题代码：{problem_code}。")
        if problem_status:
            detail_parts.append(f"问题详情：{problem_status}。")
        detail_parts.append("当前检测到的可用串口：" + (", ".join(detected_ports) if detected_ports else "无") + "。")
        detail_parts.append("请确认设备已通电、数据线支持数据传输、串口未被其他程序占用，然后点击“刷新串口”后重试。")
        return base_message + "\n" + "".join(detail_parts)

    @pyqtSlot()
    def run(self) -> None:
        final_payload = None
        data_chunks: list[np.ndarray] = []
        try:
            self.stop_event.clear()
            try:
                BoardShim.release_all_sessions()
            except Exception:
                pass

            params = BrainFlowInputParams()
            if self.serial_port:
                params.serial_port = self.serial_port

            with self.board_lock:
                self.board = BoardShim(self.board_id, params)
                self.board.prepare_session()
                self._start_stream_with_retry(self.board, buffer_size=450000)
                self.current_quality_mode = self.MODE_EEG
                self._hw_impedance_channel = None

            eeg_rows = BoardShim.get_eeg_channels(self.board_id)
            if len(self.channel_positions) > len(eeg_rows):
                raise ValueError(
                    f"当前板卡只提供 {len(eeg_rows)} 个 EEG 通道，但你选择了 {len(self.channel_positions)} 个位置。"
                )

            self.selected_rows = [int(eeg_rows[index]) for index in self.channel_positions]
            self.marker_row = int(BoardShim.get_marker_channel(self.board_id))
            self.timestamp_row = int(BoardShim.get_timestamp_channel(self.board_id))
            try:
                self.package_num_row = int(BoardShim.get_package_num_channel(self.board_id))
            except Exception:
                self.package_num_row = None
            try:
                board_descr = BoardShim.get_board_descr(self.board_id)
                self.board_descr = dict(board_descr) if isinstance(board_descr, dict) else {}
            except Exception:
                self.board_descr = {}
            self.sampling_rate = float(BoardShim.get_sampling_rate(self.board_id))

            self.connection_ready.emit(
                {
                    "sampling_rate": self.sampling_rate,
                    "selected_rows": self.selected_rows,
                    "marker_row": self.marker_row,
                    "timestamp_row": self.timestamp_row,
                    "package_num_row": self.package_num_row,
                    "board_descr": dict(self.board_descr),
                    "channel_names": self.channel_names,
                }
            )
            self.status_changed.emit(
                f"设备已连接 | 采样率 {self.sampling_rate:g} Hz | 通道 {', '.join(self.channel_names)}"
            )

            while not self.stop_event.wait(self.poll_interval_sec):
                if self.board is None:
                    break
                pending_quality_mode_request = None
                with self.quality_mode_request_lock:
                    if self._pending_quality_mode_request is not None:
                        pending_quality_mode_request = dict(self._pending_quality_mode_request)
                        self._pending_quality_mode_request = None
                if pending_quality_mode_request is not None:
                    ok, message = self.switch_quality_mode_sync(
                        target_mode=str(pending_quality_mode_request["target_mode"]),
                        target_channel=int(pending_quality_mode_request["target_channel"]),
                        reset_default=bool(pending_quality_mode_request["reset_default"]),
                    )
                    self.quality_mode_switch_finished.emit(
                        {
                            "ok": bool(ok),
                            "message": str(message),
                            "target_mode": str(pending_quality_mode_request["target_mode"]),
                            "target_channel": int(pending_quality_mode_request["target_channel"]),
                            "reset_default": bool(pending_quality_mode_request["reset_default"]),
                        }
                    )
                    continue
                try:
                    with self.board_lock:
                        chunk = None if self.board is None else self.board.get_board_data()
                except Exception:
                    chunk = None
                if chunk is not None and np.size(chunk):
                    chunk_array = np.asarray(chunk, dtype=np.float64)
                    data_chunks.append(chunk_array)
                    with self.sample_count_lock:
                        self.total_samples_captured += int(chunk_array.shape[1])
                    if self.selected_rows:
                        try:
                            preview_chunk = np.asarray(chunk_array[self.selected_rows, :], dtype=np.float32)
                        except Exception:
                            preview_chunk = None
                        if preview_chunk is not None and preview_chunk.size:
                            self.preview_data_ready.emit(preview_chunk)
        except Exception as error:
            self.error_occurred.emit(self._build_connection_error_message(error))
        finally:
            if self.board is not None:
                try:
                    with self.board_lock:
                        tail_chunk = None if self.board is None else self.board.get_board_data()
                except Exception:
                    tail_chunk = None
                if tail_chunk is not None and np.size(tail_chunk):
                    tail_array = np.asarray(tail_chunk, dtype=np.float64)
                    data_chunks.append(tail_array)
                    with self.sample_count_lock:
                        self.total_samples_captured += int(tail_array.shape[1])

                if data_chunks:
                    try:
                        full_data = self._concat_capture_chunks(data_chunks)
                    except Exception as error:
                        self.error_occurred.emit(f"合并采集数据失败：{error}")
                        full_data = None
                else:
                    full_data = None

                if full_data is not None and np.size(full_data):
                    final_payload = {
                        "brainflow_data": np.asarray(full_data, dtype=np.float64),
                        "sampling_rate": None if self.sampling_rate is None else float(self.sampling_rate),
                        "selected_rows": list(self.selected_rows),
                        "marker_row": self.marker_row,
                        "timestamp_row": self.timestamp_row,
                        "package_num_row": self.package_num_row,
                        "board_descr": dict(self.board_descr),
                        "total_samples_captured": self.captured_sample_count_sync(),
                    }

                with self.board_lock:
                    try:
                        if self.board is not None:
                            self.board.stop_stream()
                    except Exception:
                        pass
                    try:
                        if self.board is not None:
                            self.board.release_session()
                    except Exception:
                        pass
                    self.board = None

            if final_payload is not None:
                self.session_data_ready.emit(final_payload)
            self.status_changed.emit("设备已断开。")
            self.finished.emit()

    @pyqtSlot(float)
    def insert_marker(self, marker_code: float) -> None:
        if self.board is None:
            return
        try:
            self.board.insert_marker(float(marker_code))
            time.sleep(self._marker_flush_delay_sec())
        except Exception as error:
            self.error_occurred.emit(f"写入标记失败：{error}")

    def insert_marker_sync(self, marker_code: float, *, wait_for_flush: bool = True) -> tuple[bool, str]:
        with self.board_lock:
            if self.board is None:
                return False, "设备未连接，无法写入标记。"
            try:
                self.board.insert_marker(float(marker_code))
                if wait_for_flush:
                    time.sleep(self._marker_flush_delay_sec())
            except Exception as error:
                return False, f"写入标记失败：{error}"
        return True, ""

    def captured_sample_count_sync(self) -> int:
        with self.sample_count_lock:
            return int(self.total_samples_captured)

    @pyqtSlot()
    def request_stop(self) -> None:
        self.stop_event.set()


class MiSpeechPromptPlayer(QObject):
    playback_finished = pyqtSignal(int)
    _fallback_finished = pyqtSignal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._engine = None
        self._pending_request_id = 0
        self._awaiting_completion = False
        self._fallback_lock = threading.Lock()
        self._fallback_process: subprocess.Popen | None = None
        self._fallback_generation = 0
        self._fallback_finished.connect(self._finish_if_pending)
        if QTextToSpeech is not None:
            try:
                self._engine = QTextToSpeech(self)
                self._engine.setRate(0.0)
                self._engine.setVolume(1.0)
                self._engine.stateChanged.connect(self._on_state_changed)
            except Exception:
                self._engine = None

    def say(self, text: str, *, request_id: int, timeout_sec: float = PHASE_VOICE_PROMPT_TIMEOUT_SEC) -> None:
        rid = int(request_id)
        message = str(text or "").strip()
        if rid <= 0:
            return
        self.stop()
        if not message:
            QTimer.singleShot(0, lambda rid=rid: self.playback_finished.emit(rid))
            return
        if self._should_prefer_windows_fallback() and self._start_windows_fallback(
            message,
            request_id=rid,
            timeout_sec=timeout_sec,
        ):
            return
        if self._try_qt_say(message, request_id=rid, timeout_sec=timeout_sec):
            return
        if self._start_windows_fallback(message, request_id=rid, timeout_sec=timeout_sec):
            return
        QTimer.singleShot(0, lambda rid=rid: self.playback_finished.emit(rid))

    def _should_prefer_windows_fallback(self) -> bool:
        if not sys.platform.startswith("win"):
            return False
        if self._engine is None:
            return True
        return self._engine.__class__.__name__ == "QTextToSpeech"

    def stop(self) -> None:
        self._awaiting_completion = False
        self._pending_request_id = 0
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass
        with self._fallback_lock:
            self._fallback_generation += 1
            process = self._fallback_process
            self._fallback_process = None
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass

    def _try_qt_say(self, text: str, *, request_id: int, timeout_sec: float) -> bool:
        if self._engine is None:
            return False
        try:
            self._pending_request_id = int(request_id)
            self._awaiting_completion = True
            self._engine.say(str(text))
            timeout_ms = int(max(0.1, float(timeout_sec)) * 1000)
            QTimer.singleShot(timeout_ms, lambda rid=int(request_id): self._timeout_if_pending(rid))
            return True
        except Exception:
            self._awaiting_completion = False
            self._pending_request_id = 0
            return False

    def _start_windows_fallback(self, text: str, *, request_id: int, timeout_sec: float) -> bool:
        if not sys.platform.startswith("win"):
            return False
        self._pending_request_id = int(request_id)
        self._awaiting_completion = True
        with self._fallback_lock:
            generation = int(self._fallback_generation)
        threading.Thread(
            target=self._run_windows_fallback,
            args=(str(text), int(request_id), generation, float(timeout_sec)),
            daemon=True,
        ).start()
        return True

    def _run_windows_fallback(self, text: str, request_id: int, generation: int, timeout_sec: float) -> None:
        quoted_text = str(text).replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech\n"
            "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
            "try {\n"
            "    $synth.Volume = 100\n"
            "    $synth.Rate = 0\n"
            f"    $synth.Speak('{quoted_text}')\n"
            "} finally {\n"
            "    $synth.Dispose()\n"
            "}\n"
        )
        encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        command = [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-EncodedCommand",
            encoded,
        ]
        process: subprocess.Popen | None = None
        should_finish = True
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            with self._fallback_lock:
                if int(self._fallback_generation) != int(generation):
                    should_finish = False
                else:
                    self._fallback_process = process
            if not should_finish:
                try:
                    process.terminate()
                except Exception:
                    pass
                return
            process.wait(timeout=max(0.1, float(timeout_sec)))
        except Exception:
            if process is not None:
                try:
                    process.terminate()
                except Exception:
                    pass
        finally:
            with self._fallback_lock:
                if self._fallback_process is process:
                    self._fallback_process = None
            if should_finish:
                self._fallback_finished.emit(int(request_id))

    def _finish_if_pending(self, request_id: int) -> None:
        if not self._awaiting_completion:
            return
        if int(request_id) != int(self._pending_request_id):
            return
        self._awaiting_completion = False
        self._pending_request_id = 0
        self.playback_finished.emit(int(request_id))

    def _timeout_if_pending(self, request_id: int) -> None:
        if not self._awaiting_completion:
            return
        if int(request_id) != int(self._pending_request_id):
            return
        self._awaiting_completion = False
        self._pending_request_id = 0
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass
        self.playback_finished.emit(int(request_id))

    def _on_state_changed(self, state) -> None:
        if not self._awaiting_completion:
            return
        ready_state = getattr(QTextToSpeech, "Ready", None) if QTextToSpeech is not None else None
        if ready_state is None:
            return
        try:
            is_ready = int(state) == int(ready_state)
        except Exception:
            is_ready = state == ready_state
        if is_ready:
            self._finish_if_pending(int(self._pending_request_id))


class SessionSaveWorker(QObject):
    """Background save worker so disk export does not block the UI thread."""

    save_completed = pyqtSignal(dict)
    save_failed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        *,
        payload: dict[str, object],
        settings: SessionSettings,
        event_log: list[dict[str, object]],
        trial_records: list[TrialRecord],
        emergency_only: bool = False,
        emergency_reason: str = "",
    ) -> None:
        super().__init__()
        self.payload = dict(payload)
        self.settings = settings
        self.event_log = list(event_log)
        self.trial_records = list(trial_records)
        self.emergency_only = bool(emergency_only)
        self.emergency_reason = str(emergency_reason or "")

    def _save_emergency_raw(self, *, original_error: str) -> dict[str, object]:
        recovery = save_mi_emergency_raw_capture(
            brainflow_data=np.asarray(self.payload["brainflow_data"]),
            sampling_rate=float(self.payload["sampling_rate"]),
            eeg_rows=list(self.payload["selected_rows"]),
            marker_row=None if self.payload.get("marker_row") is None else int(self.payload["marker_row"]),
            timestamp_row=(
                None if self.payload["timestamp_row"] is None else int(self.payload["timestamp_row"])
            ),
            package_num_row=(
                None if self.payload.get("package_num_row") is None else int(self.payload["package_num_row"])
            ),
            board_descr=dict(self.payload.get("board_descr") or {}),
            settings=self.settings,
            event_log=self.event_log,
            trial_records=self.trial_records,
            original_error=str(original_error),
        )
        return {
            **dict(recovery),
            "emergency_raw_only": True,
            "full_save_error": str(original_error),
            "trial_count": int(len(self.trial_records)),
            "accepted_trial_count": int(sum(1 for trial in self.trial_records if bool(trial.accepted))),
            "rejected_trial_count": int(sum(1 for trial in self.trial_records if not bool(trial.accepted))),
            "session_dir": str(recovery.get("session_dir", "")),
            "board_data_path": str(recovery.get("emergency_board_data_path", "")),
            "fif_path": "",
            "segments_csv_path": "",
            "mi_epochs_path": "",
            "gate_epochs_path": "",
            "artifact_epochs_path": "",
            "continuous_path": "",
            "manifest_csv_path": "",
        }

    @pyqtSlot()
    def run(self) -> None:
        try:
            if self.emergency_only:
                result = self._save_emergency_raw(original_error=self.emergency_reason or "emergency_raw_only")
            else:
                result = save_mi_session(
                    brainflow_data=np.asarray(self.payload["brainflow_data"]),
                    sampling_rate=float(self.payload["sampling_rate"]),
                    eeg_rows=list(self.payload["selected_rows"]),
                    marker_row=int(self.payload["marker_row"]),
                    timestamp_row=(
                        None if self.payload["timestamp_row"] is None else int(self.payload["timestamp_row"])
                    ),
                    package_num_row=(
                        None if self.payload.get("package_num_row") is None else int(self.payload["package_num_row"])
                    ),
                    board_descr=dict(self.payload.get("board_descr") or {}),
                    settings=self.settings,
                    event_log=self.event_log,
                    trial_records=self.trial_records,
                )
        except Exception as error:
            try:
                recovery = self._save_emergency_raw(original_error=str(error))
            except Exception as recovery_error:
                self.save_failed.emit(f"保存会话失败：{error}\n\n紧急原始数据保存也失败：{recovery_error}")
            else:
                self.save_completed.emit(recovery)
        else:
            self.save_completed.emit(result)
        finally:
            self.finished.emit()


class MIDataCollectorWindow(QMainWindow):
    """主采集窗口，负责流程控制与状态展示。"""

    marker_requested = pyqtSignal(float)
    worker_stop_requested = pyqtSignal()
    preview_mode_switch_requested = pyqtSignal(str, int, bool)

    def __init__(self, initial_config: dict | None = None) -> None:
        super().__init__()
        self.config = dict(DEFAULT_CONFIG)
        if initial_config:
            self.config.update(initial_config)

        self.worker_thread: QThread | None = None
        self.worker: BoardCaptureWorker | None = None
        self.save_thread: QThread | None = None
        self.save_worker: SessionSaveWorker | None = None
        self.device_info: dict | None = None
        self.capture_on_stop = False
        self.session_running = False
        self.session_paused = False
        self.phase_prompt_pending = False
        self.waiting_for_save = False
        self.emergency_save_only = False
        self.emergency_save_reason = ""
        self.current_phase = "idle"
        self.phase_deadline = 0.0
        self.phase_started_perf = 0.0
        self.pause_started_perf = 0.0
        self.remaining_phase_sec = 0.0
        self.session_start_perf = 0.0
        self.sequence: list[str] = []
        self.sequence_by_run: list[list[str]] = []
        self.trials_per_run = 0
        self.current_run_index = 0
        self.current_run_trial_index = 0
        self.pending_run_rest_before_index: int | None = None
        self.event_log: list[dict[str, object]] = []
        self.trial_records: list[TrialRecord] = []
        self.current_trial_index = -1
        self.current_trial: TrialRecord | None = None
        self.current_settings: SessionSettings | None = None
        self.completed_trials = 0
        self.operator_hidden_for_session = False
        self.use_separate_participant_screen = bool(self.config.get("use_separate_participant_screen", True))
        self.calibration_plan: list[dict[str, object]] = []
        self.calibration_step_index = -1
        self.idle_block_index = 0
        self.continuous_block_index = 0
        self.continuous_prompt_plan: list[dict[str, object]] = []
        self.continuous_prompt_index = -1
        self.current_continuous_prompt: dict[str, object] | None = None
        self.continuous_schedule_after_runs: list[int] = []
        self.next_continuous_schedule_index = 0
        self.marker_failure_active = False
        self.marker_failure_message = ""
        self.runtime_failure_active = False
        self.runtime_failure_message = ""
        self.config_panel_widget: QWidget | None = None
        self.config_panel_layout: QVBoxLayout | None = None
        self.config_bottom_panel: QWidget | None = None
        self.config_section_label: QLabel | None = None
        self.config_section_combo: QComboBox | None = None
        self.config_stack: QStackedWidget | None = None
        self.config_groups: list[QWidget] = []
        self.config_forms: list[QFormLayout] = []
        self.config_grid_columns = 0
        self.control_group: QGroupBox | None = None
        self.control_layout: QGridLayout | None = None
        self.control_layout_columns = 0
        self.main_splitter: QSplitter | None = None
        self.session_tabs: QTabWidget | None = None
        self.operator_preview_panel: QWidget | None = None
        self.preview_group: QGroupBox | None = None
        self.hero_title_label: QLabel | None = None
        self.hero_subtitle_label: QLabel | None = None
        self.root_layout: QVBoxLayout | None = None
        self.body_layout: QHBoxLayout | None = None
        self.preview_widget: RealtimeEEGPreviewWidget | None = None
        self.preview_status_label: QLabel | None = None
        self.preview_mode_label: QLabel | None = None
        self.preview_layout: QVBoxLayout | None = None
        self.preview_control_layout: QGridLayout | None = None
        self.preview_to_eeg_button: QPushButton | None = None
        self.preview_to_imp_button: QPushButton | None = None
        self.preview_prev_ch_button: QPushButton | None = None
        self.preview_next_ch_button: QPushButton | None = None
        self.preview_reset_button: QPushButton | None = None
        self.preview_mode = RealtimeEEGPreviewWidget.MODE_EEG
        self.preview_impedance_channel = 1
        self.preview_mode_switch_pending = False
        self.preview_mode_switch_target: dict[str, object] | None = None
        self.pending_session_start_context: tuple[SessionSettings, list[list[str]]] | None = None
        self.audio_prompts_enabled = True
        self.voice_prompt_request_id = 0
        self.speech_prompt_player = MiSpeechPromptPlayer(self)
        self.log_group: QGroupBox | None = None
        self.log_text: QTextEdit | None = None
        self.protocol_text_label: QLabel | None = None
        self.operator_notice_label: QLabel | None = None
        self.config_tip_label: QLabel | None = None
        self.session_panel_layout: QVBoxLayout | None = None
        self._typography_signature: tuple[tuple[str, int], ...] | None = None
        self._control_button_signature: tuple[object, ...] | None = None
        self._responsive_chrome_signature: tuple[int, int, str] | None = None
        self._preview_focus_layout_active = False

        self.phase_timer = QTimer(self)
        self.phase_timer.setTimerType(Qt.PreciseTimer)
        self.phase_timer.setInterval(20)
        self.phase_timer.timeout.connect(self._on_phase_tick_guarded)

        self._init_ui()
        self.participant_window = ParticipantDisplayWindow()
        self.participant_window.pause_requested.connect(self.toggle_pause)
        self.participant_window.mark_bad_requested.connect(self.mark_bad_trial)
        self.participant_window.advance_requested.connect(self.request_phase_advance)
        self.participant_window.stop_requested.connect(self.stop_and_save)
        self.apply_default_values()
        self.refresh_board_input_state()
        self._apply_phase_theme("idle", None)

    def _current_ui_mode(self) -> str:
        if self.session_running or self.waiting_for_save:
            return "session"
        if self.device_info is not None:
            return "preview"
        return "startup"

    def _set_splitter_sizes_if_needed(self, first: int, second: int) -> None:
        if self.main_splitter is None:
            return
        target_sizes = [max(0, int(first)), max(0, int(second))]
        current_sizes = self.main_splitter.sizes()
        if len(current_sizes) == 2 and all(abs(int(current) - int(target)) <= 12 for current, target in zip(current_sizes, target_sizes)):
            return
        self.main_splitter.setSizes(target_sizes)

    def _desired_config_columns(self) -> int:
        if self.config_panel_widget is None:
            return 3
        width = self.config_panel_widget.width()
        height = self.config_panel_widget.height()
        if width >= 1500:
            columns = 4
        elif width >= 980:
            columns = 3
        else:
            columns = 2
        # When vertical space is tight, prefer more columns to avoid clipping without scrollbars.
        if height < 760:
            columns = max(columns, 3 if width >= 980 else 2)
        return columns

    def _refresh_config_group_layout(self, force: bool = False) -> None:
        if self.session_tabs is not None:
            self.session_tabs.setDocumentMode(self.width() < 1500 or self.height() < 880)
        if self.control_layout is None:
            return
        if self.config_panel_widget is None:
            return

        preview_only_mode = self._current_ui_mode() == "preview"
        if self.config_panel_layout is not None:
            self.config_panel_layout.setStretch(0, 0)
            self.config_panel_layout.setStretch(1, 0 if preview_only_mode else 1)
            self.config_panel_layout.setStretch(2, 1 if preview_only_mode else 0)
        if self.config_bottom_panel is not None:
            self.config_bottom_panel.setSizePolicy(
                QSizePolicy.Preferred,
                QSizePolicy.Expanding if preview_only_mode else QSizePolicy.Maximum,
            )
        if self.control_group is not None:
            self.control_group.setSizePolicy(
                QSizePolicy.Preferred,
                QSizePolicy.Expanding if preview_only_mode else QSizePolicy.Maximum,
            )
        config_width = self.config_panel_widget.width()
        wrap_all_rows = config_width < 360
        if self.config_section_label is not None:
            if preview_only_mode or config_width < 380:
                self.config_section_label.setVisible(False)
            else:
                section_label_text = "分区" if config_width < 460 else "配置分区"
                self.config_section_label.setVisible(True)
                self.config_section_label.setText(section_label_text)
                self.config_section_label.setMinimumWidth(
                    self.config_section_label.fontMetrics().horizontalAdvance(section_label_text) + 8
                )
        if self.config_section_combo is not None:
            self.config_section_combo.setMinimumWidth(0)
            self.config_section_combo.setVisible(not preview_only_mode)
        if self.config_stack is not None:
            self.config_stack.setVisible(not preview_only_mode)
        for form in self.config_forms:
            form.setRowWrapPolicy(QFormLayout.WrapAllRows if wrap_all_rows else QFormLayout.WrapLongRows)
            form.setHorizontalSpacing(5 if wrap_all_rows else 6)
            form.setVerticalSpacing(4 if wrap_all_rows else 6)
            for row in range(form.rowCount()):
                label_item = form.itemAt(row, QFormLayout.LabelRole)
                label = label_item.widget() if label_item is not None else None
                if isinstance(label, QLabel):
                    label.setWordWrap(True)
                    label.setAlignment(Qt.AlignLeft | (Qt.AlignTop if wrap_all_rows else Qt.AlignVCenter))
                    label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        active_buttons = self._visible_control_buttons_in_display_order()
        if preview_only_mode:
            desired_columns = self._preview_control_layout_columns(config_width, len(active_buttons))
        elif config_width < 300:
            desired_columns = 1
        elif config_width < 420:
            desired_columns = 2
        else:
            desired_columns = 3
        desired_columns = max(1, min(desired_columns, len(active_buttons) if active_buttons else 1))
        if not force and desired_columns == self.control_layout_columns and not preview_only_mode:
            return
        self.control_layout_columns = desired_columns
        while self.control_layout.count():
            item = self.control_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
        if preview_only_mode:
            row_count = max(1, int(np.ceil(len(active_buttons) / float(max(1, desired_columns)))))
            spacing = 12 if desired_columns == 2 else 10
            available_height = max(240, int(self.config_panel_widget.height()) - 48)
            raw_row_height = int((available_height - spacing * max(0, row_count - 1) - 28) / float(row_count))
            row_height = int(np.clip(raw_row_height, 64, 104))
            button_heights = {1: row_height, 2: row_height, 3: row_height}
            button_max_heights = {1: 16777215, 2: 16777215, 3: 16777215}
            row_min_heights = {1: row_height, 2: row_height, 3: row_height}
        else:
            compact_controls = self.height() < 780 or self.session_running or self._preview_focus_active()
            if compact_controls:
                spacing = 6 if desired_columns <= 2 else 5
                button_heights = {1: 34, 2: 32, 3: 30}
                button_max_heights = {1: 40, 2: 38, 3: 36}
                row_min_heights = {1: 38, 2: 34, 3: 32}
            else:
                spacing = 8 if desired_columns == 2 else 10
                button_heights = {1: 38, 2: 36, 3: 34}
                button_max_heights = {1: 46, 2: 46, 3: 40}
                row_min_heights = {1: 44, 2: 44, 3: 38}
            raw_row_height = button_heights[desired_columns]
        self._refresh_control_button_chrome(
            preview_only_mode=preview_only_mode,
            desired_columns=desired_columns,
            spacing=spacing,
            estimated_row_height=max(raw_row_height, button_heights[desired_columns]),
        )
        self.control_layout.setHorizontalSpacing(spacing)
        self.control_layout.setVerticalSpacing(spacing)
        for column in range(3):
            self.control_layout.setColumnStretch(column, 0)
        for row in range(4):
            self.control_layout.setRowStretch(row, 0)
            self.control_layout.setRowMinimumHeight(row, 0)
        for column in range(desired_columns):
            self.control_layout.setColumnStretch(column, 1)
        if preview_only_mode:
            for row in range(max(1, int(np.ceil(len(active_buttons) / float(max(1, desired_columns)))))):
                self.control_layout.setRowStretch(row, 1)

        for index, button in enumerate(active_buttons):
            row = index // desired_columns
            column = index % desired_columns
            button.show()
            button.setMinimumHeight(button_heights[desired_columns])
            button.setMaximumHeight(button_max_heights[desired_columns])
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding if preview_only_mode else QSizePolicy.Fixed)
            button.setMinimumWidth(0)
            self.control_layout.addWidget(button, row, column)
            self.control_layout.setRowMinimumHeight(row, row_min_heights[desired_columns])
        if preview_only_mode:
            configured_buttons = set(active_buttons)
            for button in self._control_buttons_in_display_order():
                if button in configured_buttons:
                    continue
                button.setMinimumHeight(button_heights[desired_columns])
                button.setMaximumHeight(button_max_heights[desired_columns])
                button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                button.setMinimumWidth(0)
                button.resize(max(button.width(), button.sizeHint().width()), button_heights[desired_columns])

    @staticmethod
    def _preview_control_layout_columns(config_width: int, visible_button_count: int) -> int:
        return 1

    def _control_buttons_in_display_order(self) -> list[QPushButton]:
        buttons = [
            self.connect_button,
            self.start_button,
            self.start_mi_only_button,
            self.show_plan_button,
            self.show_mi_only_plan_button,
            self.pause_button,
            self.bad_trial_button,
            self.stop_button,
            self.disconnect_button,
        ]
        return [button for button in buttons if button is not None]

    def _visible_control_buttons_in_display_order(self) -> list[QPushButton]:
        return [button for button in self._control_buttons_in_display_order() if not button.isHidden()]

    def _apply_control_button_visibility(self) -> None:
        connected = self.device_info is not None
        if not connected:
            visible_names = {"btnConnect"}
        elif not self.session_running and not self.waiting_for_save:
            visible_names = {"btnStart", "btnStartMiOnly", "btnShowPlan", "btnShowMiOnlyPlan", "btnDisconnect"}
        else:
            visible_names = {"btnPause", "btnWarn", "btnStop"}

        changed = False
        for button in self._control_buttons_in_display_order():
            should_show = button.objectName() in visible_names
            if button.isHidden() == should_show:
                button.setHidden(not should_show)
                changed = True
        if changed:
            self._refresh_config_group_layout(force=True)

    @staticmethod
    def _configure_form_layout(form: QFormLayout) -> None:
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignTop)
        form.setSpacing(6)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

    @staticmethod
    def _generate_session_seed() -> int:
        """Generate a per-session random seed and keep it in a visible 6-digit range."""
        return int(secrets.randbelow(900000) + 100000)

    def _init_ui(self) -> None:
        self.setWindowTitle("运动想象采集台（仅采集，不判别）")
        self.resize(1700, 960)
        self.setMinimumSize(980, 640)
        self.setStyleSheet(
            """
            QMainWindow { background: #EDF2F7; }
            QStatusBar {
                background: #0F172A;
                color: #E2E8F0;
                border-top: 1px solid #334155;
            }
            QLabel { color: #102A43; }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #D5DFEB;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 8px;
                background: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: #1E293B;
                background: #FFFFFF;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
                border: 1px solid #CBD5E1;
                border-radius: 8px;
                padding: 6px 9px;
                background: #FFFFFF;
                selection-background-color: #2563EB;
                selection-color: #FFFFFF;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {
                border: 2px solid #2563EB;
            }
            QPushButton {
                border: 1px solid transparent;
                border-radius: 8px;
                padding: 8px 14px;
                font-weight: bold;
                background: #1D4ED8;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background: #1E40AF;
            }
            QPushButton:pressed {
                background: #1E3A8A;
            }
            QPushButton:disabled {
                background: #D1D5DB;
                color: #64748B;
                border-color: #D1D5DB;
            }
            QPushButton#btnConnect { background: #0F766E; }
            QPushButton#btnConnect:hover { background: #0B5E59; }
            QPushButton#btnStart { background: #2563EB; }
            QPushButton#btnStart:hover { background: #1D4ED8; }
            QPushButton#btnPause { background: #7C3AED; }
            QPushButton#btnPause:hover { background: #6D28D9; }
            QPushButton#btnWarn { background: #EA580C; }
            QPushButton#btnWarn:hover { background: #C2410C; }
            QPushButton#btnStop { background: #DC2626; }
            QPushButton#btnStop:hover { background: #B91C1C; }
            QPushButton#btnDisconnect { background: #334155; }
            QPushButton#btnDisconnect:hover { background: #1E293B; }
            QTextEdit#logPanel {
                background: #0B1727;
                color: #D1E3FF;
                border: 1px solid #1E3A5F;
                border-radius: 8px;
                font-family: Consolas, "Microsoft YaHei";
            }
            QProgressBar {
                border: 1px solid #CBD5E1;
                border-radius: 8px;
                text-align: center;
                background: #F8FAFC;
                color: #1E293B;
            }
            QProgressBar::chunk {
                background: #0F766E;
                border-radius: 7px;
            }
            QTabWidget::pane {
                border: 1px solid #D5DFEB;
                border-radius: 10px;
                background: #FFFFFF;
                top: -1px;
            }
            QTabBar::tab {
                min-width: 64px;
                padding: 8px 14px;
                margin-right: 4px;
                border: 1px solid #D5DFEB;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                background: #F8FAFC;
                color: #475569;
            }
            QTabBar::tab:selected {
                background: #FFFFFF;
                color: #0F172A;
            }
            """
        )

        central = QWidget()
        self.setCentralWidget(central)
        self.root_layout = QVBoxLayout(central)
        self.root_layout.setContentsMargins(18, 18, 18, 18)
        self.root_layout.setSpacing(14)

        self.hero_title_label = QLabel("运动想象数据采集（纯采集，不判别）")
        self.hero_title_label.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        self.hero_title_label.setStyleSheet("color: #0F172A; letter-spacing: 1px;")
        self.root_layout.addWidget(self.hero_title_label)

        self.hero_subtitle_label = QLabel("当前版本按标准试次流程工作：每个试次都包含准备、提示、想象和休息，只负责事件标记与数据保存。")
        self.hero_subtitle_label.setWordWrap(True)
        self.hero_subtitle_label.setStyleSheet(
            "color: #334155; background: #FFFFFF; border: 1px solid #D5DFEB; border-radius: 8px; padding: 8px 12px;"
        )
        self.root_layout.addWidget(self.hero_subtitle_label)

        self.body_layout = QHBoxLayout()
        self.body_layout.setSpacing(14)
        self.root_layout.addLayout(self.body_layout, stretch=1)
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.config_panel_widget = self._build_config_panel()
        session_panel = self._build_session_panel()
        self.main_splitter.addWidget(self.config_panel_widget)
        self.main_splitter.addWidget(session_panel)
        self.main_splitter.setStretchFactor(0, 11)
        self.main_splitter.setStretchFactor(1, 9)
        self.main_splitter.setSizes([920, 820])
        self.main_splitter.splitterMoved.connect(lambda _pos, _idx: self._refresh_config_group_layout())
        self.body_layout.addWidget(self.main_splitter, stretch=1)

        self.operator_preview_panel = self._build_focus_panel()
        self.operator_preview_panel.setParent(self)
        self.operator_preview_panel.hide()
        self._refresh_config_group_layout(force=True)

        self.log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(self.log_group)
        self.log_text = QTextEdit()
        self.log_text.setObjectName("logPanel")
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(92)
        log_layout.addWidget(self.log_text)
        self.root_layout.addWidget(self.log_group, stretch=0)

        self._apply_responsive_typography()
        self.statusBar().showMessage("准备就绪")
        self._update_responsive_chrome()
        self._refresh_preview_text_heights()
        self._refresh_status_badges()
        self._set_operator_notice("待机：连接设备后先完成 EEG/阻抗质量检查。", "idle")

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_responsive_typography()
        self._refresh_config_group_layout()
        self._update_responsive_chrome(force=True)
        self._refresh_preview_text_heights()

    def _update_responsive_chrome(self, *, force: bool = False) -> None:
        width = self.width()
        height = self.height()
        ui_mode = self._current_ui_mode()
        compact = height < 880 or width < 1500
        ultra_compact = height < 720 or width < 1100
        stacked = width < 900
        preview_focus = ui_mode == "preview"
        signature = (int(width), int(height), ui_mode)
        if not force and signature == self._responsive_chrome_signature:
            return
        self._responsive_chrome_signature = signature

        if preview_focus:
            outer_margin = 8 if ultra_compact else 10 if compact else 12
            section_spacing = 6 if ultra_compact else 8 if compact else 10
        elif ui_mode == "startup":
            outer_margin = 12 if ultra_compact else 16 if compact else 20
            section_spacing = 10 if ultra_compact else 12 if compact else 16
        else:
            outer_margin = 10 if ultra_compact else 14 if compact else 18
            section_spacing = 8 if ultra_compact else 10 if compact else 14

        if self.root_layout is not None:
            self.root_layout.setContentsMargins(outer_margin, outer_margin, outer_margin, outer_margin)
            self.root_layout.setSpacing(section_spacing)
        if self.body_layout is not None:
            self.body_layout.setSpacing(section_spacing)
        if self.hero_subtitle_label is not None:
            self.hero_subtitle_label.setVisible(ui_mode == "startup" and not ultra_compact)
        if self.hero_title_label is not None:
            self.hero_title_label.setVisible(ui_mode == "startup")
        if self.config_tip_label is not None:
            self.config_tip_label.setVisible(ui_mode == "startup" and not compact)
        if self.preview_group is not None:
            self.preview_group.setVisible(ui_mode != "startup")
        if self.session_tabs is not None:
            self.session_tabs.setVisible(ui_mode != "preview")
        if self.log_group is not None:
            self.log_group.setVisible(not ultra_compact and ui_mode != "preview")
        if self.log_text is not None:
            if self.session_running:
                self.log_text.setMinimumHeight(56 if ultra_compact else 60 if compact else 76)
            else:
                self.log_text.setMinimumHeight(64 if ultra_compact else 72 if compact else 92)

        if self.main_splitter is not None:
            desired_orientation = Qt.Vertical if stacked else Qt.Horizontal
            if self.main_splitter.orientation() != desired_orientation:
                self.main_splitter.setOrientation(desired_orientation)
                if desired_orientation == Qt.Vertical:
                    self._set_splitter_sizes_if_needed(int(height * 0.40), int(height * 0.60))
                else:
                    self._set_splitter_sizes_if_needed(920, 820)
                self._refresh_config_group_layout(force=True)
            if stacked:
                self.main_splitter.setStretchFactor(0, 4)
                self.main_splitter.setStretchFactor(1, 6)
            elif preview_focus:
                self.main_splitter.setStretchFactor(0, 4)
                self.main_splitter.setStretchFactor(1, 16)
            elif ui_mode == "startup":
                self.main_splitter.setStretchFactor(0, 13)
                self.main_splitter.setStretchFactor(1, 9)
            elif compact:
                self.main_splitter.setStretchFactor(0, 5)
                self.main_splitter.setStretchFactor(1, 5)
            else:
                self.main_splitter.setStretchFactor(0, 11)
                self.main_splitter.setStretchFactor(1, 9)

        if self.preview_widget is not None:
            if preview_focus:
                self.preview_widget.setMinimumHeight(480 if ultra_compact else 520 if compact else 560)
            elif ui_mode == "startup":
                self.preview_widget.setMinimumHeight(320 if ultra_compact else 380 if compact else 460)
            else:
                self.preview_widget.setMinimumHeight(280 if ultra_compact else 340 if compact else 460)

        if self.session_panel_layout is not None:
            if preview_focus:
                self.session_panel_layout.setStretch(0, 1)
                self.session_panel_layout.setStretch(1, 0)
            elif ui_mode == "startup":
                self.session_panel_layout.setStretch(0, 0)
                self.session_panel_layout.setStretch(1, 1)
            elif ultra_compact:
                self.session_panel_layout.setStretch(0, 4)
                self.session_panel_layout.setStretch(1, 3)
            elif compact:
                self.session_panel_layout.setStretch(0, 6)
                self.session_panel_layout.setStretch(1, 3)
            else:
                self.session_panel_layout.setStretch(0, 6)
                self.session_panel_layout.setStretch(1, 2)

        if self.session_tabs is not None:
            self.session_tabs.setDocumentMode(compact)
            self.session_tabs.tabBar().setExpanding(True)
        if self.preview_group is not None:
            if ui_mode == "preview":
                self.preview_group.setTitle("质量检查（EEG / 阻抗）")
            elif ui_mode == "startup":
                self.preview_group.setTitle("EEG / 阻抗预览")
            else:
                self.preview_group.setTitle("采集中监看（EEG / 阻抗）")
        if self.protocol_text_label is not None:
            protocol_text = self._protocol_copy(compact=compact, ultra_compact=ultra_compact)
            if self.protocol_text_label.text() != protocol_text:
                self.protocol_text_label.setText(protocol_text)
        self._apply_preview_focus_layout()

    def _preview_stream_available(self) -> bool:
        if self.device_info is not None:
            channel_names = self.device_info.get("channel_names", [])
            if isinstance(channel_names, list) and channel_names:
                return True
            selected_rows = self.device_info.get("selected_rows", [])
            if isinstance(selected_rows, list) and selected_rows:
                return True
        return (
            self.preview_widget is not None
            and bool(self.preview_widget.channel_names)
            and float(self.preview_widget.sampling_rate) > 0.0
        )

    def _preview_focus_active(self) -> bool:
        return self.device_info is not None and not self.session_running and not self.waiting_for_save

    def _apply_preview_focus_layout(self, *, force: bool = False) -> None:
        ui_mode = self._current_ui_mode()
        preview_focus = ui_mode == "preview"
        del force
        self._preview_focus_layout_active = preview_focus

        if self.session_tabs is not None:
            self.session_tabs.setVisible(ui_mode != "preview")
        if self.preview_group is not None:
            self.preview_group.setVisible(ui_mode != "startup")
        if self.config_panel_widget is not None:
            if preview_focus and self.main_splitter is not None and self.main_splitter.orientation() == Qt.Horizontal:
                if self.width() >= 1750:
                    max_width = 460
                elif self.width() >= 1600:
                    max_width = 430
                elif self.width() >= 1400:
                    max_width = 390
                elif self.width() >= 1150:
                    max_width = 340
                else:
                    max_width = 300
                self.config_panel_widget.setMaximumWidth(max_width)
            else:
                self.config_panel_widget.setMaximumWidth(16777215)

        if self.main_splitter is not None and self.main_splitter.orientation() == Qt.Horizontal:
            total = sum(self.main_splitter.sizes())
            if total <= 0:
                total = max(900, self.width() - 80)
            if preview_focus:
                if total <= 0:
                    total = max(900, self.width() - 80)
                if self.width() >= 1750:
                    config_width = max(360, min(460, int(total * 0.24)))
                elif self.width() >= 1600:
                    config_width = max(340, min(430, int(total * 0.25)))
                elif self.width() >= 1400:
                    config_width = max(310, min(390, int(total * 0.26)))
                elif self.width() >= 1150:
                    config_width = max(280, min(340, int(total * 0.27)))
                else:
                    config_width = max(240, min(300, int(total * 0.28)))
                min_preview_width = 680 if self.width() < 1150 else 760 if self.width() < 1400 else 820
                config_width = min(config_width, max(240, int(total - min_preview_width)))
                preview_width = max(600, int(total - config_width))
                self._set_splitter_sizes_if_needed(config_width, preview_width)
            elif ui_mode == "startup":
                if self.width() >= 1750:
                    config_width = max(980, min(1120, int(total * 0.58)))
                elif self.width() >= 1600:
                    config_width = max(900, min(1020, int(total * 0.57)))
                elif self.width() >= 1400:
                    config_width = max(760, min(900, int(total * 0.56)))
                elif self.width() >= 1150:
                    config_width = max(620, min(740, int(total * 0.57)))
                else:
                    config_width = max(540, min(620, int(total * 0.58)))
                overview_width = max(320, int(total - config_width))
                self._set_splitter_sizes_if_needed(config_width, overview_width)
            else:
                self._set_splitter_sizes_if_needed(920, 820)

        self._refresh_config_group_layout(force=True)
        self._apply_preview_controls_layout()
        self._refresh_preview_text_heights()

    def _apply_preview_controls_layout(self) -> None:
        if self.preview_layout is None or self.preview_control_layout is None:
            return
        if self.preview_status_label is None or self.preview_mode_label is None:
            return

        preview_focus = self._preview_focus_active()
        preview_width = self.preview_group.width() if self.preview_group is not None else self.width()
        wide_focus_toolbar = preview_focus and preview_width >= 620
        show_full_focus_labels = preview_focus and preview_width >= 980

        self.preview_status_label.setVisible(not preview_focus)
        self.preview_status_label.setStyleSheet(
            "color: #334155; background: #F8FAFC; border-radius: 8px; padding: 8px 10px;"
        )
        self.preview_status_label.setMinimumHeight(0 if preview_focus else 46)

        self.preview_layout.setContentsMargins(4 if preview_focus else 9, 4 if preview_focus else 9, 4 if preview_focus else 9, 4 if preview_focus else 9)
        self.preview_layout.setSpacing(4 if preview_focus else 9)
        self.preview_control_layout.setHorizontalSpacing(5 if preview_focus else 8)
        self.preview_control_layout.setVerticalSpacing(3 if preview_focus else 8)
        self.preview_control_layout.setContentsMargins(0, 0, 0, 0)

        self.preview_mode_label.setWordWrap(not wide_focus_toolbar)
        self.preview_mode_label.setStyleSheet(
            "color: #0F172A; font-weight: 600;"
            if not preview_focus
            else "color: #0F172A; font-weight: 600; background: #E0F2FE; border: 1px solid #BAE6FD; border-radius: 8px; padding: 6px 10px;"
        )

        buttons = (
            self.preview_to_eeg_button,
            self.preview_to_imp_button,
        )
        short_texts = {
            self.preview_to_eeg_button: "EEG",
            self.preview_to_imp_button: "阻抗",
        }
        full_texts = {
            self.preview_to_eeg_button: "EEG模式",
            self.preview_to_imp_button: "阻抗模式",
        }
        for button in buttons:
            if button is None:
                continue
            self.preview_control_layout.removeWidget(button)
            button_text = full_texts[button] if (not preview_focus or show_full_focus_labels) else short_texts[button]
            button.setText(button_text)
            button.setToolTip(full_texts[button])
            button.setMinimumHeight(28 if preview_focus else 34)
            button.setMinimumWidth(button.fontMetrics().horizontalAdvance(button_text) + (24 if preview_focus else 28))
            if preview_focus:
                button.setMaximumHeight(32)
                button.setStyleSheet("padding: 3px 8px;")
            else:
                button.setMaximumHeight(16777215)
                button.setStyleSheet("")
        self.preview_control_layout.removeWidget(self.preview_mode_label)

        while self.preview_control_layout.count():
            item = self.preview_control_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()

        controls_in_order = [
            self.preview_mode_label,
            self.preview_to_eeg_button,
            self.preview_to_imp_button,
        ]
        for widget in controls_in_order:
            if widget is not None:
                widget.show()
        for widget in (self.preview_prev_ch_button, self.preview_next_ch_button, self.preview_reset_button):
            if widget is not None:
                widget.hide()

        if wide_focus_toolbar:
            self.preview_control_layout.addWidget(self.preview_mode_label, 0, 0, 1, 2)
            self.preview_control_layout.addWidget(self.preview_to_eeg_button, 0, 2)
            self.preview_control_layout.addWidget(self.preview_to_imp_button, 0, 3)
            for index in range(4):
                self.preview_control_layout.setColumnStretch(index, 0 if index >= 2 else 1)
        elif preview_focus:
            self.preview_control_layout.addWidget(self.preview_mode_label, 0, 0, 1, 2)
            self.preview_control_layout.addWidget(self.preview_to_eeg_button, 1, 0)
            self.preview_control_layout.addWidget(self.preview_to_imp_button, 1, 1)
            for index in range(2):
                self.preview_control_layout.setColumnStretch(index, 1)
        else:
            self.preview_control_layout.addWidget(self.preview_mode_label, 0, 0, 1, 2)
            self.preview_control_layout.addWidget(self.preview_to_eeg_button, 1, 0)
            self.preview_control_layout.addWidget(self.preview_to_imp_button, 1, 1)
            self.preview_control_layout.setColumnStretch(0, 1)
            self.preview_control_layout.setColumnStretch(1, 1)
        self._refresh_preview_toggle_button_styles()

    def _protocol_copy(self, *, compact: bool, ultra_compact: bool) -> str:
        if ultra_compact:
            return (
                "操作员：先完成连接质量检查，再开始正式采集。\n"
                "入口：完整流程，或仅 MI 主任务（跳过静息/伪迹/无控制/连续仿真）。\n"
                "单试次默认：注视 2s → 提示 2s → 想象 4s → 放松 2.5s。"
            )
        if compact:
            return (
                "操作员：连接后先检查波形和接触质量，稳定后开始正式采集。\n"
                "入口：完整流程，或仅 MI 主任务（跳过静息/伪迹/无控制/连续仿真）。\n"
                "单试次默认：注视 2s → 提示 2s → 想象 4s → 放松 2.5s。"
            )
        return (
            "操作员流程：连接设备后先在当前界面观察原始波形并调整参数，确认稳定后再开始正式采集。\n"
            "入口一：完整流程，包含静息/伪迹校准 → 多轮次主任务 → 无控制 → 连续仿真；想象训练保留为可选阶段，默认关闭。\n"
            "入口二：仅 MI 主任务，直接进入四分类 MI trial，跳过静息、伪迹、无控制和连续仿真块。\n"
            "主任务单试次固定顺序：注视 2s → 提示(2s) → 想象(4s) → 放松(2.5s)。\n"
            "点击开始后，如勾选受试者全屏，当前屏幕会直接切到受试者提示界面。"
        )

    def _refresh_preview_text_heights(self) -> None:
        label_specs = (
            (self.preview_status_label, 24, 46),
            (self.preview_mode_label, 12, 22),
            (self.protocol_text_label, 24, 120),
            (self.config_tip_label, 18, 60),
        )
        for label, padding, minimum in label_specs:
            if label is None:
                continue
            available_width = max(
                220,
                label.contentsRect().width()
                if label.width() > 0 and label.contentsRect().width() > 0
                else label.sizeHint().width(),
            )
            text_rect = label.fontMetrics().boundingRect(
                QRect(0, 0, available_width, 2000),
                int(label.alignment()) | Qt.TextWordWrap,
                label.text(),
            )
            label.setMinimumHeight(max(int(minimum), int(text_rect.height() + padding)))
            label.updateGeometry()

    def _build_typography_profile(self) -> dict[str, int]:
        width = max(1, self.width())
        height = max(1, self.height())
        preview_focus = self._preview_focus_active()
        if width >= 1600 and height >= 900:
            profile = {
                "form": 14,
                "section": 14,
                "small": 13,
                "tab": 14,
                "button": 14,
                "button_height": 38,
                "group": 14,
                "hero": 26,
                "phase": 27,
                "countdown": 21,
                "instruction": 17,
                "banner": 16,
            }
        elif width >= 1400 and height >= 820:
            profile = {
                "form": 13,
                "section": 14,
                "small": 12,
                "tab": 13,
                "button": 13,
                "button_height": 36,
                "group": 13,
                "hero": 24,
                "phase": 25,
                "countdown": 20,
                "instruction": 16,
                "banner": 15,
            }
        else:
            profile = {
                "form": 12,
                "section": 13,
                "small": 12,
                "tab": 12,
                "button": 12,
                "button_height": 34,
                "group": 12,
                "hero": 22,
                "phase": 24,
                "countdown": 19,
                "instruction": 15,
                "banner": 14,
            }

        if preview_focus:
            if width >= 1400:
                reductions = {
                    "form": (12, 1),
                    "section": (13, 1),
                    "small": (12, 1),
                    "tab": (12, 1),
                    "button": (12, 1),
                    "group": (12, 1),
                    "hero": (23, 1),
                    "phase": (25, 1),
                    "countdown": (19, 1),
                    "instruction": (15, 1),
                    "banner": (14, 1),
                }
                for key, (floor, delta) in reductions.items():
                    profile[key] = max(floor, int(profile[key]) - delta)
                profile["button_height"] = max(34, int(profile["button_height"]) - 2)
            elif width >= 1100:
                reductions = {
                    "section": (12, 1),
                    "tab": (11, 1),
                    "button": (11, 1),
                    "group": (11, 1),
                }
                for key, (floor, delta) in reductions.items():
                    profile[key] = max(floor, int(profile[key]) - delta)
                profile["button_height"] = max(32, int(profile["button_height"]) - 1)

        return profile

    def _refresh_control_button_chrome(
        self,
        *,
        preview_only_mode: bool,
        desired_columns: int,
        spacing: int,
        estimated_row_height: int,
    ) -> None:
        active_buttons = self._visible_control_buttons_in_display_order()
        chrome_buttons = self._control_buttons_in_display_order() if preview_only_mode else active_buttons
        if not chrome_buttons:
            return

        def _button_text(button: QPushButton) -> str:
            name = button.objectName()
            primary_default_map = {
                "btnConnect": "连接设备",
                "btnStart": "开始完整流程",
                "btnStartMiOnly": "直接进入 MI 主任务",
                "btnShowPlan": "显示完整流程计划",
                "btnShowMiOnlyPlan": "显示 MI 主任务计划",
                "btnDisconnect": "断开设备",
            }
            primary_preview_map = {
                "btnConnect": "连接设备",
                "btnStart": "开始完整流程",
                "btnStartMiOnly": "直接进入 MI 主任务",
                "btnShowPlan": "显示完整流程计划",
                "btnShowMiOnlyPlan": "显示 MI 主任务计划",
                "btnDisconnect": "断开设备",
            }
            if name in primary_default_map:
                mapping = primary_preview_map if preview_only_mode else primary_default_map
                return mapping.get(name, button.text().replace("\n", ""))
            warn_text = "标记命令失败" if self.current_phase == "continuous" else "标记坏试次"
            default_map = {
                "btnConnect": "连接设备",
                "btnStart": "开始完整流程",
                "btnStartMiOnly": "直接进入MI主任务",
                "btnShowPlan": "显示完整流程计划",
                "btnShowMiOnlyPlan": "显示MI主任务计划",
                "btnPause": "继续" if self.session_paused else "暂停",
                "btnWarn": warn_text,
                "btnStop": "停止并保存",
                "btnDisconnect": "断开设备",
            }
            preview_map = {
                "btnConnect": "连接\n设备",
                "btnStart": "完整\n流程",
                "btnStartMiOnly": "仅MI\n主任务",
                "btnShowPlan": "显示完整\n流程计划",
                "btnShowMiOnlyPlan": "显示MI\n任务计划",
                "btnPause": "继续" if self.session_paused else "暂停",
                "btnWarn": "标记\n命令失败" if self.current_phase == "continuous" else "标记\n坏试次",
                "btnStop": "停止\n并保存",
                "btnDisconnect": "断开\n设备",
            }
            mapping = preview_map if preview_only_mode else default_map
            return mapping.get(name, button.text().replace("\n", ""))

        for button in chrome_buttons:
            next_text = _button_text(button)
            if button.text() != next_text:
                button.setText(next_text)
        self._refresh_control_button_tooltips()

        if preview_only_mode and self.control_group is not None:
            control_rect = self.control_group.contentsRect()
            column_width = max(
                120,
                int((control_rect.width() - spacing * max(0, desired_columns - 1)) / float(max(1, desired_columns))),
            )
            vertical_padding = int(np.clip(estimated_row_height * 0.08, 8, 18))
            horizontal_padding = 16 if desired_columns >= 2 else 20
            radius = int(np.clip(estimated_row_height * 0.08, 8, 14))
            font_size = int(np.clip(estimated_row_height * (0.12 if desired_columns >= 2 else 0.10), 16, 28))
            while font_size > 12:
                metrics = QFontMetrics(QFont("Microsoft YaHei", font_size, QFont.DemiBold))
                widest_label = max(
                    max(metrics.horizontalAdvance(line) for line in button.text().splitlines() or [""])
                    for button in chrome_buttons
                )
                if widest_label + horizontal_padding * 2 <= column_width - 12:
                    break
                font_size -= 1
            signature: tuple[object, ...] = (
                "preview",
                desired_columns,
                column_width,
                estimated_row_height,
                font_size,
                vertical_padding,
                horizontal_padding,
                radius,
            )
            if signature == self._control_button_signature:
                return
            self._control_button_signature = signature
            button_font = QFont("Microsoft YaHei", font_size, QFont.DemiBold)
            button_style = (
                "QPushButton { "
                f"padding: {vertical_padding}px {horizontal_padding}px; "
                f"border-radius: {radius}px; "
                "}"
                "QPushButton:disabled { "
                "background: #DCE5F0; "
                "color: #7B8794; "
                "border: 1px solid #D0D8E3; "
                "}"
            )
            for button in chrome_buttons:
                button.setFont(button_font)
                button.setStyleSheet(button_style)
            return

        profile = self._build_typography_profile()
        signature = ("default", desired_columns, int(profile["button"]), int(profile["button_height"]))
        if signature == self._control_button_signature:
            return
        self._control_button_signature = signature
        button_font = QFont("Microsoft YaHei", int(profile["button"]), QFont.DemiBold)
        for button in chrome_buttons:
            button.setFont(button_font)
            button.setStyleSheet("")

    def _apply_responsive_typography(self) -> None:
        profile = self._build_typography_profile()
        signature = tuple(sorted(profile.items()))
        if signature == self._typography_signature:
            return
        self._typography_signature = signature

        self.setFont(QFont("Microsoft YaHei", int(profile["form"])))

        group_font = QFont("Microsoft YaHei", int(profile["group"]), QFont.DemiBold)
        for group in self.findChildren(QGroupBox):
            group.setFont(group_font)

        form_font = QFont("Microsoft YaHei", int(profile["form"]))
        form_control_height = max(30, int(QFontMetrics(form_font).height() + 12))
        for widget_type in (QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox):
            for widget in self.findChildren(widget_type):
                widget.setFont(form_font)
                if widget_type is not QCheckBox:
                    widget.setMinimumHeight(form_control_height)

        for text_edit in self.findChildren(QTextEdit):
            if text_edit is self.log_text:
                text_edit.setFont(QFont("Consolas", max(12, int(profile["form"]))))
            else:
                text_edit.setFont(form_font)

        button_font = QFont("Microsoft YaHei", int(profile["button"]), QFont.DemiBold)
        for button in self.findChildren(QPushButton):
            button.setFont(button_font)
            button.setMinimumHeight(max(button.minimumHeight(), int(profile["button_height"])))

        tab_font = QFont("Microsoft YaHei", int(profile["tab"]), QFont.DemiBold)
        if self.config_section_combo is not None:
            self.config_section_combo.setFont(tab_font)
        if self.session_tabs is not None:
            self.session_tabs.tabBar().setFont(tab_font)

        if self.hero_title_label is not None:
            self.hero_title_label.setFont(QFont("Microsoft YaHei", int(profile["hero"]), QFont.Bold))
        if self.hero_subtitle_label is not None:
            self.hero_subtitle_label.setFont(QFont("Microsoft YaHei", int(profile["section"])))
        if self.protocol_text_label is not None:
            self.protocol_text_label.setFont(QFont("Microsoft YaHei", int(profile["section"])))
        if self.config_tip_label is not None:
            self.config_tip_label.setFont(QFont("Microsoft YaHei", int(profile["small"])))

        detail_font = QFont("Microsoft YaHei", int(profile["section"]))
        for label in (
            self.device_label,
            self.summary_label,
            self.current_label,
            self.sequence_summary_label,
            self.accepted_label,
            self.rejected_label,
            self.preview_status_label,
            self.sequence_label,
        ):
            if label is not None:
                label.setFont(detail_font)

        if self.preview_mode_label is not None:
            self.preview_mode_label.setFont(QFont("Microsoft YaHei", int(profile["section"]), QFont.DemiBold))
        if self.operator_notice_label is not None:
            self.operator_notice_label.setFont(QFont("Microsoft YaHei", int(profile["section"]), QFont.DemiBold))
        if self.sequence_hint_label is not None:
            self.sequence_hint_label.setFont(QFont("Microsoft YaHei", int(profile["small"])))
        if self.progress_text is not None:
            self.progress_text.setFont(QFont("Microsoft YaHei", int(profile["small"])))
        if self.next_task_label is not None:
            self.next_task_label.setFont(QFont("Microsoft YaHei", int(profile["small"])))
        if self.trial_banner_label is not None:
            self.trial_banner_label.setFont(QFont("Microsoft YaHei", int(profile["banner"]), QFont.DemiBold))
        if self.instruction_label is not None:
            self.instruction_label.setFont(QFont("Microsoft YaHei", int(profile["instruction"])))
        if self.phase_label is not None:
            self.phase_label.setFont(QFont("Microsoft YaHei", int(profile["phase"]), QFont.Bold))
        if self.countdown_label is not None:
            self.countdown_label.setFont(QFont("Microsoft YaHei", int(profile["countdown"]), QFont.Bold))

    def _build_config_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.config_panel_layout = layout
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.subject_edit = QLineEdit()
        self.subject_edit.setPlaceholderText("例如 S01 或 张三")
        self.session_edit = QLineEdit()
        self.output_edit = QLineEdit()
        browse_button = QPushButton("选择目录")
        browse_button.clicked.connect(self.browse_output_directory)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, stretch=1)
        output_row.addWidget(browse_button)
        output_widget = QWidget()
        output_widget.setLayout(output_row)

        self.board_combo = QComboBox()
        for label, board_id in available_board_options():
            self.board_combo.addItem(label, board_id)
            index = self.board_combo.count() - 1
            self.board_combo.setItemData(index, f"{label} ({board_id})", Qt.ToolTipRole)
        self.board_combo.currentIndexChanged.connect(self.refresh_board_input_state)

        self.serial_combo = QComboBox()
        self.serial_combo.setEditable(True)
        self.refresh_ports_button = QPushButton("刷新串口")
        self.refresh_ports_button.clicked.connect(self.refresh_serial_ports)
        serial_row = QHBoxLayout()
        serial_row.setContentsMargins(0, 0, 0, 0)
        serial_row.setSpacing(8)
        serial_row.addWidget(self.serial_combo, stretch=1)
        serial_row.addWidget(self.refresh_ports_button, stretch=0)
        serial_widget = QWidget()
        serial_widget.setLayout(serial_row)
        self.channel_names_edit = QLineEdit()
        self.channel_positions_edit = QLineEdit()
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(1, 999)
        self.baseline_spin = QDoubleSpinBox()
        self.baseline_spin.setRange(0.5, 60.0)
        self.baseline_spin.setDecimals(1)
        self.baseline_spin.setSingleStep(0.5)
        self.cue_spin = QDoubleSpinBox()
        self.cue_spin.setRange(MIN_MI_CUE_SEC_FOR_STANDARD_VISUAL, 20.0)
        self.cue_spin.setDecimals(1)
        self.cue_spin.setSingleStep(0.5)
        self.cue_voice_check = QCheckBox("训练用 cue 语音")
        self.cue_voice_check.setToolTip(
            "默认关闭：正式采集只使用视觉 cue 和短提示音。打开后每个 cue 会短语音播报类别，并自动要求提示阶段至少 2 秒。"
        )
        self.cue_voice_check.toggled.connect(self._sync_cue_voice_timing_constraints)
        self.imagery_spin = QDoubleSpinBox()
        self.imagery_spin.setRange(0.5, 20.0)
        self.imagery_spin.setDecimals(1)
        self.imagery_spin.setSingleStep(0.5)
        self.iti_spin = QDoubleSpinBox()
        self.iti_spin.setRange(MIN_MI_ITI_SEC_BETWEEN_TRIALS, 30.0)
        self.iti_spin.setDecimals(1)
        self.iti_spin.setSingleStep(0.5)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(100000, 999999)
        self.seed_spin.setReadOnly(True)
        self.seed_spin.setToolTip("每次开始采集时自动生成并保存到会话元数据")
        self.separate_screen_check = QCheckBox("开始采集后弹出受试者全屏提示窗")
        self.notes_edit = QTextEdit()
        self.notes_edit.setMinimumHeight(64)
        self.run_count_spin = QSpinBox()
        self.run_count_spin.setRange(1, 12)
        self.run_count_spin.setToolTip("MI 主任务 run 数；可以设为 1。")
        self.max_consecutive_spin = QSpinBox()
        self.max_consecutive_spin.setRange(1, 6)
        self.run_rest_spin = QDoubleSpinBox()
        self.run_rest_spin.setRange(0.0, 600.0)
        self.run_rest_spin.setDecimals(1)
        self.long_run_every_spin = QSpinBox()
        self.long_run_every_spin.setRange(0, 10)
        self.long_run_rest_spin = QDoubleSpinBox()
        self.long_run_rest_spin.setRange(0.0, 900.0)
        self.long_run_rest_spin.setDecimals(1)

        self.practice_spin = QDoubleSpinBox()
        self.practice_spin.setRange(0.0, 1200.0)
        self.practice_spin.setDecimals(1)
        self.calib_open_spin = QDoubleSpinBox()
        self.calib_open_spin.setRange(0.0, 600.0)
        self.calib_open_spin.setDecimals(1)
        self.calib_closed_spin = QDoubleSpinBox()
        self.calib_closed_spin.setRange(0.0, 600.0)
        self.calib_closed_spin.setDecimals(1)
        self.calib_eye_spin = QDoubleSpinBox()
        self.calib_eye_spin.setRange(0.0, 600.0)
        self.calib_eye_spin.setDecimals(1)
        self.calib_blink_spin = QDoubleSpinBox()
        self.calib_blink_spin.setRange(0.0, 300.0)
        self.calib_blink_spin.setDecimals(1)
        self.calib_swallow_spin = QDoubleSpinBox()
        self.calib_swallow_spin.setRange(0.0, 300.0)
        self.calib_swallow_spin.setDecimals(1)
        self.calib_jaw_spin = QDoubleSpinBox()
        self.calib_jaw_spin.setRange(0.0, 300.0)
        self.calib_jaw_spin.setDecimals(1)
        self.calib_head_spin = QDoubleSpinBox()
        self.calib_head_spin.setRange(0.0, 300.0)
        self.calib_head_spin.setDecimals(1)

        self.idle_count_spin = QSpinBox()
        self.idle_count_spin.setRange(0, 10)
        self.idle_sec_spin = QDoubleSpinBox()
        self.idle_sec_spin.setRange(0.0, 600.0)
        self.idle_sec_spin.setDecimals(1)

        self.continuous_count_spin = QSpinBox()
        self.continuous_count_spin.setRange(0, 10)
        self.continuous_count_spin.setToolTip("连续模式段只能插在 MI run 结束边界；设为 0 可关闭连续模式。")
        self.continuous_sec_spin = QDoubleSpinBox()
        self.continuous_sec_spin.setRange(0.0, 1200.0)
        self.continuous_sec_spin.setDecimals(1)
        self.cont_cmd_min_spin = QDoubleSpinBox()
        self.cont_cmd_min_spin.setRange(0.5, 30.0)
        self.cont_cmd_min_spin.setDecimals(1)
        self.cont_cmd_max_spin = QDoubleSpinBox()
        self.cont_cmd_max_spin.setRange(0.5, 30.0)
        self.cont_cmd_max_spin.setDecimals(1)
        self.cont_gap_min_spin = QDoubleSpinBox()
        self.cont_gap_min_spin.setRange(0.0, 30.0)
        self.cont_gap_min_spin.setDecimals(1)
        self.cont_gap_max_spin = QDoubleSpinBox()
        self.cont_gap_max_spin.setRange(0.0, 30.0)
        self.cont_gap_max_spin.setDecimals(1)
        self.run_count_spin.valueChanged.connect(self._sync_continuous_count_limit)
        self._sync_continuous_count_limit()

        self.artifact_types_edit = QLineEdit()
        self.reference_edit = QLineEdit()
        self.sleep_edit = QLineEdit()
        self.eyes_closed_for_gate_check = QCheckBox("闭眼静息加入门控负类")
        self.participant_state_combo = QComboBox()
        for key, label in PARTICIPANT_STATE_OPTIONS:
            self.participant_state_combo.addItem(label, key)
        self.caffeine_combo = QComboBox()
        for key, label in CAFFEINE_OPTIONS:
            self.caffeine_combo.addItem(label, key)
        self.exercise_combo = QComboBox()
        for key, label in RECENT_EXERCISE_OPTIONS:
            self.exercise_combo.addItem(label, key)

        session_group = QGroupBox("会话信息")
        session_form = QFormLayout(session_group)
        self._configure_form_layout(session_form)
        session_form.addRow("被试编号/名称", self.subject_edit)
        session_form.addRow("会话编号", self.session_edit)
        session_form.addRow("输出目录", output_widget)

        participant_group = QGroupBox("主观状态")
        participant_form = QFormLayout(participant_group)
        self._configure_form_layout(participant_form)
        participant_form.addRow("状态（主观）", self.participant_state_combo)
        participant_form.addRow("咖啡/茶", self.caffeine_combo)
        participant_form.addRow("刚运动", self.exercise_combo)
        participant_form.addRow("睡眠备注", self.sleep_edit)
        participant_form.addRow("参考电极设置", self.reference_edit)

        notes_group = QGroupBox("备注")
        notes_form = QFormLayout(notes_group)
        self._configure_form_layout(notes_form)
        notes_form.addRow("实验备注", self.notes_edit)

        board_group = QGroupBox("设备参数")
        board_form = QFormLayout(board_group)
        self._configure_form_layout(board_form)
        board_form.addRow("板卡类型", self.board_combo)
        board_form.addRow("串口", serial_widget)
        board_form.addRow("通道名称", self.channel_names_edit)
        board_form.addRow("通道位置", self.channel_positions_edit)

        timing_group = QGroupBox("MI 试次流程")
        timing_form = QFormLayout(timing_group)
        self._configure_form_layout(timing_form)
        timing_form.addRow("每类试次数", self.trials_spin)
        timing_form.addRow("准备阶段（秒）", self.baseline_spin)
        timing_form.addRow("提示阶段（秒）", self.cue_spin)
        timing_form.addRow("想象阶段（秒）", self.imagery_spin)
        timing_form.addRow("休息阶段（秒）", self.iti_spin)
        timing_form.addRow("", self.cue_voice_check)

        run_group = QGroupBox("轮次与导出")
        run_form = QFormLayout(run_group)
        self._configure_form_layout(run_form)
        run_form.addRow("轮次数量", self.run_count_spin)
        run_form.addRow("同类最多连续", self.max_consecutive_spin)
        run_form.addRow("轮次休息（秒）", self.run_rest_spin)
        run_form.addRow("每几轮加长休息", self.long_run_every_spin)
        run_form.addRow("加长休息（秒）", self.long_run_rest_spin)

        export_group = QGroupBox("导出")
        export_form = QFormLayout(export_group)
        self._configure_form_layout(export_form)
        export_form.addRow("随机种子（自动）", self.seed_spin)
        export_form.addRow("", self.separate_screen_check)

        calibration_group = QGroupBox("静息与训练准备")
        calibration_form = QFormLayout(calibration_group)
        self._configure_form_layout(calibration_form)
        calibration_form.addRow("睁眼静息（秒）", self.calib_open_spin)
        calibration_form.addRow("闭眼静息（秒）", self.calib_closed_spin)
        calibration_form.addRow("", self.eyes_closed_for_gate_check)
        calibration_form.addRow("想象训练（秒，0=关闭）", self.practice_spin)

        artifact_group = QGroupBox("伪迹校准")
        artifact_form = QFormLayout(artifact_group)
        self._configure_form_layout(artifact_form)
        artifact_form.addRow("眼动（秒）", self.calib_eye_spin)
        artifact_form.addRow("眨眼（秒）", self.calib_blink_spin)
        artifact_form.addRow("吞咽（秒）", self.calib_swallow_spin)
        artifact_form.addRow("咬牙（秒）", self.calib_jaw_spin)
        artifact_form.addRow("头动（秒）", self.calib_head_spin)
        artifact_form.addRow("伪迹类型", self.artifact_types_edit)

        post_group = QGroupBox("无控制")
        post_form = QFormLayout(post_group)
        self._configure_form_layout(post_form)
        post_form.addRow("无控制段数", self.idle_count_spin)
        post_form.addRow("无控制时长（秒）", self.idle_sec_spin)

        continuous_group = QGroupBox("连续模式")
        continuous_form = QFormLayout(continuous_group)
        self._configure_form_layout(continuous_form)
        continuous_form.addRow("连续模式段数", self.continuous_count_spin)
        continuous_form.addRow("连续模式时长（秒）", self.continuous_sec_spin)
        continuous_form.addRow("命令最短时长（秒）", self.cont_cmd_min_spin)
        continuous_form.addRow("命令最长时长（秒）", self.cont_cmd_max_spin)
        continuous_form.addRow("命令间隔最短（秒）", self.cont_gap_min_spin)
        continuous_form.addRow("命令间隔最长（秒）", self.cont_gap_max_spin)

        control_group = QGroupBox("操作")
        control_layout = QGridLayout(control_group)
        self.control_group = control_group
        self.control_layout = control_layout
        control_layout.setHorizontalSpacing(10)
        control_layout.setVerticalSpacing(10)
        self.connect_button = QPushButton("连接设备")
        self.connect_button.setObjectName("btnConnect")
        self.start_button = QPushButton("开始完整流程")
        self.start_button.setObjectName("btnStart")
        self.start_mi_only_button = QPushButton("直接进入 MI 主任务")
        self.start_mi_only_button.setObjectName("btnStartMiOnly")
        self.show_plan_button = QPushButton("显示完整流程计划")
        self.show_plan_button.setObjectName("btnShowPlan")
        self.show_mi_only_plan_button = QPushButton("显示 MI 主任务计划")
        self.show_mi_only_plan_button.setObjectName("btnShowMiOnlyPlan")
        self.pause_button = QPushButton("暂停")
        self.pause_button.setObjectName("btnPause")
        self.bad_trial_button = QPushButton("标记坏试次")
        self.bad_trial_button.setObjectName("btnWarn")
        self.stop_button = QPushButton("停止并保存")
        self.stop_button.setObjectName("btnStop")
        self.disconnect_button = QPushButton("断开设备")
        self.disconnect_button.setObjectName("btnDisconnect")
        self._refresh_control_button_tooltips()
        self.connect_button.clicked.connect(self.connect_device)
        self.start_button.clicked.connect(self.start_session)
        self.start_mi_only_button.clicked.connect(self.start_mi_only_session)
        self.show_plan_button.clicked.connect(self.show_full_session_plan)
        self.show_mi_only_plan_button.clicked.connect(self.show_mi_only_session_plan)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.bad_trial_button.clicked.connect(self.mark_bad_trial)
        self.stop_button.clicked.connect(self.stop_and_save)
        self.disconnect_button.clicked.connect(self.disconnect_device)
        control_layout.addWidget(self.connect_button, 0, 0)
        control_layout.addWidget(self.start_button, 0, 1)
        control_layout.addWidget(self.start_mi_only_button, 0, 2)
        control_layout.addWidget(self.show_plan_button, 1, 0)
        control_layout.addWidget(self.show_mi_only_plan_button, 1, 1)
        control_layout.addWidget(self.disconnect_button, 1, 2)
        control_layout.addWidget(self.pause_button, 2, 0)
        control_layout.addWidget(self.bad_trial_button, 2, 1)
        control_layout.addWidget(self.stop_button, 2, 2)
        control_layout.setColumnStretch(0, 1)
        control_layout.setColumnStretch(1, 1)
        control_layout.setColumnStretch(2, 1)
        self.config_groups = [
            session_group,
            participant_group,
            notes_group,
            board_group,
            timing_group,
            run_group,
            export_group,
            calibration_group,
            artifact_group,
            post_group,
            continuous_group,
        ]
        self.config_forms = [
            session_form,
            participant_form,
            notes_form,
            board_form,
            timing_form,
            run_form,
            export_form,
            calibration_form,
            artifact_form,
            post_form,
            continuous_form,
        ]
        section_row = QHBoxLayout()
        section_row.setContentsMargins(0, 0, 0, 0)
        section_row.setSpacing(8)
        section_label = QLabel("配置分区")
        self.config_section_label = section_label
        section_label.setStyleSheet("color: #475569;")
        self.config_section_combo = QComboBox()
        self.config_section_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.config_stack = QStackedWidget()
        config_sections = [
            ("会话", session_group),
            ("状态", participant_group),
            ("备注", notes_group),
            ("设备", board_group),
            ("MI", timing_group),
            ("轮次", run_group),
            ("导出", export_group),
            ("静息", calibration_group),
            ("伪迹", artifact_group),
            ("无控制", post_group),
            ("连续", continuous_group),
        ]
        for title, group in config_sections:
            self.config_section_combo.addItem(title)
            self.config_stack.addWidget(group)
        self.config_section_combo.currentIndexChanged.connect(self.config_stack.setCurrentIndex)
        default_section_index = self.config_section_combo.findText("设备")
        if default_section_index < 0:
            default_section_index = 0
        self.config_section_combo.setCurrentIndex(default_section_index)
        section_row.addWidget(section_label, stretch=0)
        section_row.addWidget(self.config_section_combo, stretch=1)
        layout.addLayout(section_row)
        self.config_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.config_stack, stretch=1)

        self.config_tip_label = QLabel(
            "连接设备后先在右侧原始波形面板完成实验员测试；也可以先点击“显示完整流程计划”或“显示 MI 主任务计划”核对阶段、事件和预计总时长。"
            "若勾选“弹出受试者全屏提示窗”，点击“开始完整流程”或“直接进入 MI 主任务”后会直接切到受试者全屏。"
            "运行中快捷键：空格 暂停/继续，B 标记坏试次（连续模式下标记命令失败），"
            "N 提前结束训练阶段，Esc 停止并保存。"
        )
        self.config_tip_label.setWordWrap(True)
        self.config_tip_label.setStyleSheet("color: #475569; padding: 4px;")
        bottom_panel = QWidget()
        self.config_bottom_panel = bottom_panel
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)
        bottom_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        control_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        bottom_layout.addWidget(control_group, stretch=0)
        self.config_tip_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        bottom_layout.addWidget(self.config_tip_label, stretch=0)
        layout.addWidget(bottom_panel, stretch=0)
        self._refresh_config_group_layout(force=True)
        return panel

    def _build_focus_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        cue_group = QGroupBox("任务提示预览")
        cue_layout = QVBoxLayout(cue_group)

        header_row = QHBoxLayout()
        self.phase_label = QLabel("等待开始")
        self.phase_label.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        self.phase_label.setAlignment(Qt.AlignCenter)
        self.phase_label.setMinimumHeight(56)
        self.phase_label.setStyleSheet(
            "color: #FFFFFF; background: #64748B; border-radius: 14px; padding: 8px 14px;"
        )
        header_row.addWidget(self.phase_label, stretch=1)

        self.countdown_label = QLabel("剩余时间：--")
        self.countdown_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.countdown_label.setStyleSheet(
            "color: #334155; font-weight: bold; "
            "background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 12px; padding: 10px 14px;"
        )
        header_row.addWidget(self.countdown_label, stretch=0)
        cue_layout.addLayout(header_row)

        self.trial_banner_label = QLabel("当前试次：未开始")
        self.trial_banner_label.setStyleSheet(
            "color: #1E293B; background: #EEF2FF; border-radius: 8px; padding: 8px 10px;"
        )
        cue_layout.addWidget(self.trial_banner_label)

        self.operator_notice_label = QLabel("待机：连接设备后先完成 EEG/阻抗质量检查。")
        self.operator_notice_label.setWordWrap(True)
        self.operator_notice_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        cue_layout.addWidget(self.operator_notice_label)

        self.cue_widget = CueIllustrationWidget()
        cue_layout.addWidget(self.cue_widget, stretch=1)

        self.instruction_label = QLabel("连接设备后可选择“开始完整流程”或“直接进入 MI 主任务”。")
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setStyleSheet(
            "color: #0F172A; background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 12px; padding: 12px;"
        )
        cue_layout.addWidget(self.instruction_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        cue_layout.addWidget(self.progress_bar)

        footer_row = QHBoxLayout()
        self.progress_text = QLabel("总进度：0 / 0")
        self.progress_text.setStyleSheet("color: #475569;")
        footer_row.addWidget(self.progress_text, stretch=1)

        self.next_task_label = QLabel("下一任务：--")
        self.next_task_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.next_task_label.setStyleSheet("color: #475569;")
        footer_row.addWidget(self.next_task_label, stretch=0)
        cue_layout.addLayout(footer_row)

        layout.addWidget(cue_group, stretch=1)
        return panel

    def _build_session_panel(self) -> QWidget:
        panel = QWidget()
        self.session_panel_layout = QVBoxLayout(panel)
        self.session_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.session_panel_layout.setSpacing(14)

        protocol_group = QGroupBox("实验逻辑")
        protocol_layout = QVBoxLayout(protocol_group)
        self.protocol_text_label = QLabel(self._protocol_copy(compact=False, ultra_compact=False))
        self.protocol_text_label.setWordWrap(True)
        self.protocol_text_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.protocol_text_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.protocol_text_label.setStyleSheet("color: #334155;")
        protocol_layout.addWidget(self.protocol_text_label)
        protocol_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        status_group = QGroupBox("当前状态")
        status_layout = QGridLayout(status_group)
        status_layout.setHorizontalSpacing(8)
        status_layout.setVerticalSpacing(8)
        self.device_label = QLabel("设备：未连接")
        self.summary_label = QLabel("状态：等待连接设备")
        self.current_label = QLabel("当前任务：无")
        self.sequence_summary_label = QLabel("计划：尚未生成试次序列")
        self.accepted_label = QLabel("有效试次：0")
        self.rejected_label = QLabel("坏试次：0")
        for label in (
            self.device_label,
            self.summary_label,
            self.current_label,
            self.sequence_summary_label,
            self.accepted_label,
            self.rejected_label,
        ):
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            label.setMinimumHeight(34)
        status_layout.addWidget(self.device_label, 0, 0, 1, 2)
        status_layout.addWidget(self.summary_label, 1, 0, 1, 2)
        status_layout.addWidget(self.current_label, 2, 0, 1, 2)
        status_layout.addWidget(self.sequence_summary_label, 3, 0, 1, 2)
        status_layout.addWidget(self.accepted_label, 4, 0)
        status_layout.addWidget(self.rejected_label, 4, 1)
        status_layout.setColumnStretch(0, 1)
        status_layout.setColumnStretch(1, 1)
        status_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        preview_group = QGroupBox("质量检查（EEG / 阻抗）")
        self.preview_group = preview_group
        preview_layout = QVBoxLayout(preview_group)
        self.preview_layout = preview_layout
        self.preview_status_label = QLabel(
            "连接后先在这里做质量检查。EEG 预览采用 FBCCA 风格的仅显示预处理，采集保存的数据始终保持原始数据。"
        )
        self.preview_status_label.setWordWrap(True)
        self.preview_status_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.preview_status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.preview_status_label.setStyleSheet(
            "color: #334155; background: #F8FAFC; border-radius: 8px; padding: 8px 10px;"
        )
        preview_layout.addWidget(self.preview_status_label)

        preview_control_layout = QGridLayout()
        self.preview_control_layout = preview_control_layout
        preview_control_layout.setHorizontalSpacing(8)
        preview_control_layout.setVerticalSpacing(8)
        self.preview_mode_label = QLabel("质量检查模式：等待设备连接")
        self.preview_mode_label.setWordWrap(True)
        self.preview_mode_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.preview_mode_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.preview_mode_label.setStyleSheet("color: #0F172A; font-weight: 600;")
        preview_control_layout.addWidget(self.preview_mode_label, 0, 0, 1, 3)

        self.preview_to_eeg_button = QPushButton("EEG模式")
        self.preview_to_imp_button = QPushButton("阻抗模式")
        self.preview_prev_ch_button = QPushButton("上一通道")
        self.preview_next_ch_button = QPushButton("下一通道")
        self.preview_reset_button = QPushButton("恢复默认")
        for button in (
            self.preview_to_eeg_button,
            self.preview_to_imp_button,
            self.preview_prev_ch_button,
            self.preview_next_ch_button,
            self.preview_reset_button,
        ):
            button.setMinimumHeight(34)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setEnabled(False)

        self.preview_to_eeg_button.clicked.connect(self.on_preview_to_eeg_clicked)
        self.preview_to_imp_button.clicked.connect(self.on_preview_to_imp_clicked)
        self.preview_prev_ch_button.clicked.connect(self.on_preview_prev_channel_clicked)
        self.preview_next_ch_button.clicked.connect(self.on_preview_next_channel_clicked)
        self.preview_reset_button.clicked.connect(self.on_preview_reset_clicked)
        self.preview_prev_ch_button.hide()
        self.preview_next_ch_button.hide()
        self.preview_reset_button.hide()

        preview_control_layout.addWidget(self.preview_to_eeg_button, 1, 0)
        preview_control_layout.addWidget(self.preview_to_imp_button, 1, 1)
        preview_control_layout.setColumnStretch(0, 1)
        preview_control_layout.setColumnStretch(1, 1)
        preview_layout.addLayout(preview_control_layout)

        self.preview_widget = RealtimeEEGPreviewWidget(window_seconds=5.0)
        self.preview_widget.setMinimumHeight(460)
        preview_layout.addWidget(self.preview_widget, stretch=1)
        self.session_panel_layout.addWidget(preview_group, stretch=5)

        order_group = QGroupBox("试次安排")
        order_layout = QVBoxLayout(order_group)
        self.sequence_hint_label = QLabel("灰色：未开始  蓝边：当前试次  实色：已完成  删除线：坏试次")
        self.sequence_hint_label.setWordWrap(True)
        self.sequence_hint_label.setStyleSheet("color: #64748B; background: #F8FAFC; border-radius: 8px; padding: 6px 8px;")
        order_layout.addWidget(self.sequence_hint_label)
        self.sequence_label = QLabel("当前还未生成试次顺序。")
        self.sequence_label.setWordWrap(True)
        self.sequence_label.setTextFormat(Qt.RichText)
        self.sequence_label.setStyleSheet(
            "color: #334155; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px;"
        )
        order_layout.addWidget(self.sequence_label)
        self.session_tabs = QTabWidget()
        self.session_tabs.setUsesScrollButtons(False)
        self.session_tabs.addTab(status_group, "状态")
        self.session_tabs.addTab(order_group, "试次")
        self.session_tabs.addTab(protocol_group, "说明")
        self.session_tabs.currentChanged.connect(lambda _index: self._refresh_preview_text_heights())
        self.session_panel_layout.addWidget(self.session_tabs, stretch=2)
        return panel

    def _sync_continuous_count_limit(self, _value: int = 0) -> None:
        run_count = max(1, int(self.run_count_spin.value()))
        limit = min(run_count, 10)
        self.continuous_count_spin.setMaximum(limit)
        if int(self.continuous_count_spin.value()) > limit:
            self.continuous_count_spin.setValue(limit)

    def apply_default_values(self) -> None:
        self.subject_edit.setText(self.config["subject_id"])
        self.session_edit.setText(self.config["session_id"])
        self.output_edit.setText(self.config["output_root"])
        self.refresh_serial_ports()
        self.serial_combo.setCurrentText(self.config["serial_port"])
        self.channel_names_edit.setText(self.config["channel_names"])
        self.channel_positions_edit.setText(self.config["channel_positions"])
        self.trials_spin.setValue(int(self.config["trials_per_class"]))
        self.baseline_spin.setValue(float(self.config["baseline_sec"]))
        self.cue_voice_check.setChecked(bool(self.config.get("training_cue_voice_enabled", False)))
        self.cue_spin.setValue(float(self.config["cue_sec"]))
        self._sync_cue_voice_timing_constraints()
        self.imagery_spin.setValue(float(self.config["imagery_sec"]))
        self.iti_spin.setValue(float(self.config["iti_sec"]))
        self.run_count_spin.setValue(int(self.config["run_count"]))
        self.max_consecutive_spin.setValue(int(self.config["max_consecutive_same_class"]))
        self.run_rest_spin.setValue(float(self.config["run_rest_sec"]))
        self.long_run_every_spin.setValue(int(self.config["long_run_rest_every"]))
        self.long_run_rest_spin.setValue(float(self.config["long_run_rest_sec"]))
        self.practice_spin.setValue(float(self.config["practice_sec"]))
        self.calib_open_spin.setValue(float(self.config["calibration_open_sec"]))
        self.calib_closed_spin.setValue(float(self.config["calibration_closed_sec"]))
        self.calib_eye_spin.setValue(float(self.config["calibration_eye_sec"]))
        self.calib_blink_spin.setValue(float(self.config["calibration_blink_sec"]))
        self.calib_swallow_spin.setValue(float(self.config["calibration_swallow_sec"]))
        self.calib_jaw_spin.setValue(float(self.config["calibration_jaw_sec"]))
        self.calib_head_spin.setValue(float(self.config["calibration_head_sec"]))
        self.idle_count_spin.setValue(int(self.config["idle_block_count"]))
        self.idle_sec_spin.setValue(float(self.config["idle_block_sec"]))
        self.continuous_count_spin.setValue(int(self.config["continuous_block_count"]))
        self.continuous_sec_spin.setValue(float(self.config["continuous_block_sec"]))
        self.cont_cmd_min_spin.setValue(float(self.config["continuous_command_min_sec"]))
        self.cont_cmd_max_spin.setValue(float(self.config["continuous_command_max_sec"]))
        self.cont_gap_min_spin.setValue(float(self.config["continuous_gap_min_sec"]))
        self.cont_gap_max_spin.setValue(float(self.config["continuous_gap_max_sec"]))
        artifact_types_value = self.config["artifact_types"]
        if isinstance(artifact_types_value, (list, tuple)):
            artifact_types_text = ",".join(str(item).strip() for item in artifact_types_value if str(item).strip())
        else:
            artifact_types_text = str(artifact_types_value)
        self.artifact_types_edit.setText(artifact_types_text)
        self.eyes_closed_for_gate_check.setChecked(bool(self.config.get("include_eyes_closed_rest_in_gate_neg", False)))
        self.reference_edit.setText(str(self.config["reference_mode"]))
        self.sleep_edit.setText(str(self.config["sleep_note"]))
        for idx in range(self.participant_state_combo.count()):
            if str(self.participant_state_combo.itemData(idx)) == str(self.config["participant_state"]):
                self.participant_state_combo.setCurrentIndex(idx)
                break
        for idx in range(self.caffeine_combo.count()):
            if str(self.caffeine_combo.itemData(idx)) == str(self.config["caffeine_intake"]):
                self.caffeine_combo.setCurrentIndex(idx)
                break
        for idx in range(self.exercise_combo.count()):
            if str(self.exercise_combo.itemData(idx)) == str(self.config["recent_exercise"]):
                self.exercise_combo.setCurrentIndex(idx)
                break
        configured_seed = int(self.config.get("random_seed", 0))
        if configured_seed < self.seed_spin.minimum() or configured_seed > self.seed_spin.maximum():
            configured_seed = self._generate_session_seed()
        self.seed_spin.setValue(configured_seed)
        self.separate_screen_check.setChecked(bool(self.config.get("use_separate_participant_screen", True)))
        self.use_separate_participant_screen = bool(self.separate_screen_check.isChecked())
        self.notes_edit.setPlainText(self.config["notes"])
        for index in range(self.board_combo.count()):
            if int(self.board_combo.itemData(index)) == int(self.config["board_id"]):
                self.board_combo.setCurrentIndex(index)
                break
        self.update_button_states()

    def refresh_serial_ports(self) -> None:
        selected = self.serial_combo.currentText().strip()
        ports = detect_serial_ports()

        self.serial_combo.blockSignals(True)
        self.serial_combo.clear()
        for port in ports:
            self.serial_combo.addItem(port)

        if selected and self.serial_combo.findText(selected) < 0:
            self.serial_combo.addItem(selected)

        if selected:
            self.serial_combo.setCurrentText(selected)
        elif ports:
            self.serial_combo.setCurrentIndex(0)
        else:
            self.serial_combo.setCurrentText("")
        self.serial_combo.blockSignals(False)

    def refresh_board_input_state(self) -> None:
        synthetic = getattr(BoardIds, "SYNTHETIC_BOARD", None)
        is_synthetic = synthetic is not None and int(self.current_board_id()) == int(synthetic.value)
        can_edit = self.board_combo.isEnabled()
        self.serial_combo.setEnabled(can_edit and not is_synthetic)
        self.refresh_ports_button.setEnabled(can_edit and not is_synthetic)
        line_edit = self.serial_combo.lineEdit()
        if line_edit is not None:
            line_edit.setPlaceholderText("演示模式不需要串口" if is_synthetic else "例如 COM3")

    def current_board_id(self) -> int:
        return int(self.board_combo.currentData())

    def board_display_name(self) -> str:
        return str(self.board_combo.currentText())

    def browse_output_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_edit.text().strip() or str(DEFAULT_OUTPUT_ROOT))
        if selected:
            self.output_edit.setText(selected)

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.statusBar().showMessage(message, 4000)

    def show_error(self, message: str) -> None:
        if self.operator_hidden_for_session:
            self.restore_operator_window()
        self.log(message)
        QMessageBox.critical(self, "错误", message)

    def _save_in_progress(self) -> bool:
        return bool(self.waiting_for_save or self.save_thread is not None)

    def _clear_session_result_tooltips(self) -> None:
        for label in (self.current_label, self.next_task_label):
            if label is not None:
                label.setToolTip("")

    def _quality_warnings_from_report(self, quality_report: object) -> list[str]:
        if not isinstance(quality_report, dict):
            return []

        channels = quality_report.get("channels")
        if not isinstance(channels, list):
            return []

        flat_channels: list[str] = []
        for channel in channels:
            if not isinstance(channel, dict):
                continue
            name = str(channel.get("channel_name", "")).strip() or "未知通道"
            try:
                std_uv = float(channel.get("std_uV", np.nan))
                ptp_uv = float(channel.get("peak_to_peak_uV", np.nan))
                non_finite_count = int(channel.get("non_finite_sample_count", 0) or 0)
            except Exception:
                continue
            if non_finite_count > 0:
                flat_channels.append(name)
            elif not np.isfinite(std_uv) or not np.isfinite(ptp_uv):
                flat_channels.append(name)
            elif std_uv <= QUALITY_FLAT_CHANNEL_STD_UV or ptp_uv <= QUALITY_FLAT_CHANNEL_PTP_UV:
                flat_channels.append(name)

        if not flat_channels:
            return []

        shown = flat_channels[:QUALITY_WARNING_CHANNEL_LIMIT]
        suffix = "" if len(flat_channels) <= len(shown) else f" 等 {len(flat_channels)} 个通道"
        return [
            f"检测到近似平线通道：{', '.join(shown)}{suffix}。"
            "保存文件已经写入，但正式采集前请检查电极接触、参考电极和地线。"
        ]

    def _data_flow_warnings_from_report(self, data_flow_report: object) -> list[str]:
        if not isinstance(data_flow_report, dict):
            return []

        warnings: list[str] = []
        if data_flow_report.get("marker_event_count_match") is False:
            marker_count = data_flow_report.get("marker_count")
            event_count = data_flow_report.get("event_count")
            warnings.append(f"Marker 数量与事件日志不一致：marker={marker_count}, event={event_count}。")

        if data_flow_report.get("timestamp_available") and data_flow_report.get("timestamp_monotonic") is False:
            warnings.append("BrainFlow timestamp 非单调，建议检查串口连接和采集线程是否被阻塞。")

        try:
            timestamp_non_finite = int(data_flow_report.get("timestamp_non_finite_count") or 0)
        except Exception:
            timestamp_non_finite = 0
        if timestamp_non_finite > 0:
            warnings.append(f"BrainFlow timestamp 存在 {timestamp_non_finite} 个非有限值。")

        try:
            package_jumps = int(data_flow_report.get("package_jump_count") or 0)
        except Exception:
            package_jumps = 0
        if package_jumps > 0:
            warnings.append(f"检测到 {package_jumps} 处包号跳变，可能存在串口丢包或读取中断。")

        try:
            package_non_finite = int(data_flow_report.get("package_non_finite_count") or 0)
        except Exception:
            package_non_finite = 0
        if package_non_finite > 0:
            warnings.append(f"BrainFlow package number 存在 {package_non_finite} 个非有限值。")

        return warnings
    def _format_serial_port_validation_error(self, serial_port: str, validation: dict[str, object]) -> str:
        selected_port = str(validation.get("requested_port") or serial_port).strip() or str(serial_port).strip()
        detected_ports = [str(item) for item in validation.get("detected_ports", []) if str(item).strip()]
        windows_status = str(validation.get("windows_status", "")).strip()
        problem_code = str(validation.get("problem_code", "")).strip()
        problem_status = str(validation.get("problem_status", "")).strip()

        if str(validation.get("reason", "")) == "windows_unavailable":
            parts = [f"当前串口 {selected_port} 被系统标记为不可用。"]
            if windows_status:
                parts.append(f"Windows 状态：{windows_status}。")
            if problem_code:
                parts.append(f"问题代码：{problem_code}。")
            if problem_status:
                parts.append(f"问题详情：{problem_status}。")
            if detected_ports:
                parts.append(f"当前可用串口：{', '.join(detected_ports)}。")
            parts.append("请检查设备连接、驱动和串口占用后重试。")
            return "".join(parts)

        if detected_ports:
            return (
                f"当前串口 {selected_port} 不在已检测到的可用设备列表中。"
                f"已检测到：{', '.join(detected_ports)}。请点击“刷新串口”并确认设备实际连接到正确端口。"
            )
        return "当前没有检测到可用串口。请确认设备已通电、数据线已插稳，并点击“刷新串口”后重试。"

    def _build_serial_port_override_note(self, validation: dict[str, object]) -> str:
        selected_port = str(validation.get("requested_port", "")).strip()
        return (
            f"当前未枚举到可用串口，将按人工指定的 {selected_port} 尝试连接。"
            "若连接失败，请检查设备供电、数据线、驱动和串口占用后再重试。"
        )

    def collect_settings(self, *, log_serial_warning: bool = False) -> SessionSettings:
        try:
            channel_names = parse_channel_names(self.channel_names_edit.text(), expected_count=None)
        except Exception as error:
            raise ValueError(f"通道名称配置有误：{error}") from error
        if not channel_names:
            raise ValueError("通道名称不能为空。")

        try:
            channel_positions = parse_channel_positions(self.channel_positions_edit.text(), expected_count=len(channel_names))
        except Exception as error:
            raise ValueError(f"通道位置配置有误：{error}") from error
        if list(channel_names) != list(DEFAULT_CHANNEL_NAMES):
            raise ValueError(f"通道名称必须固定为：{', '.join(DEFAULT_CHANNEL_NAMES)}")
        if list(channel_positions) != list(range(len(DEFAULT_CHANNEL_NAMES))):
            raise ValueError("通道位置必须固定为 0,1,2,3,4,5,6,7。")

        serial_port = self.serial_combo.currentText().strip()
        board_id = self.current_board_id()
        if int(board_id) not in supported_collection_board_ids():
            raise ValueError("当前采集流程只支持 Cyton 8 通道、Cyton Daisy（前 8 通道）或模拟板卡。")
        synthetic = getattr(BoardIds, "SYNTHETIC_BOARD", None)
        is_synthetic = synthetic is not None and int(board_id) == int(synthetic.value)
        if not is_synthetic and not serial_port:
            raise ValueError("当前板卡需要串口，请先选择有效串口。")
        if not is_synthetic:
            serial_validation = validate_serial_port_selection(serial_port)
            if not bool(serial_validation.get("ok")):
                raise ValueError(self._format_serial_port_validation_error(serial_port, serial_validation))
            if log_serial_warning and str(serial_validation.get("reason", "")) == "manual_override":
                self.log(self._build_serial_port_override_note(serial_validation))

        output_root_path = resolve_output_root(self.output_edit.text().strip())
        try:
            output_root_path.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            raise ValueError(f"输出目录不可用：{error}") from error
        output_root = str(output_root_path)

        continuous_command_min_sec = float(self.cont_cmd_min_spin.value())
        continuous_command_max_sec = float(self.cont_cmd_max_spin.value())
        continuous_gap_min_sec = float(self.cont_gap_min_spin.value())
        continuous_gap_max_sec = float(self.cont_gap_max_spin.value())
        run_count = int(self.run_count_spin.value())
        continuous_block_count = int(self.continuous_count_spin.value())
        continuous_block_sec = float(self.continuous_sec_spin.value())
        cue_sec = float(self.cue_spin.value())
        if self._training_cue_voice_enabled():
            cue_sec = max(cue_sec, MIN_MI_CUE_SEC_FOR_NATURAL_AUDIO)

        raw_artifact_types = self.artifact_types_edit.text().strip()
        artifact_types = normalize_artifact_types(raw_artifact_types)
        if not artifact_types:
            artifact_types = list(DEFAULT_ARTIFACT_TYPES)

        subject_token = self.subject_edit.text().strip()
        if not subject_token:
            raise ValueError("被试编号/名称不能为空。")

        return SessionSettings(
            subject_id=subject_token,
            session_id=self.session_edit.text().strip() or datetime.now().strftime("%Y%m%d_%H%M%S"),
            output_root=output_root,
            board_id=board_id,
            serial_port=serial_port,
            channel_names=channel_names,
            channel_positions=channel_positions,
            trials_per_class=int(self.trials_spin.value()),
            baseline_sec=float(self.baseline_spin.value()),
            cue_sec=cue_sec,
            imagery_sec=float(self.imagery_spin.value()),
            iti_sec=float(self.iti_spin.value()),
            protocol_mode=PROTOCOL_MODE_FULL,
            run_count=run_count,
            max_consecutive_same_class=int(self.max_consecutive_spin.value()),
            run_rest_sec=float(self.run_rest_spin.value()),
            long_run_rest_every=int(self.long_run_every_spin.value()),
            long_run_rest_sec=float(self.long_run_rest_spin.value()),
            quality_check_sec=0.0,
            practice_sec=float(self.practice_spin.value()),
            calibration_open_sec=float(self.calib_open_spin.value()),
            calibration_closed_sec=float(self.calib_closed_spin.value()),
            calibration_eye_sec=float(self.calib_eye_spin.value()),
            calibration_blink_sec=float(self.calib_blink_spin.value()),
            calibration_swallow_sec=float(self.calib_swallow_spin.value()),
            calibration_jaw_sec=float(self.calib_jaw_spin.value()),
            calibration_head_sec=float(self.calib_head_spin.value()),
            idle_block_count=int(self.idle_count_spin.value()),
            idle_block_sec=float(self.idle_sec_spin.value()),
            idle_prepare_block_count=0,
            idle_prepare_sec=0.0,
            continuous_block_count=continuous_block_count,
            continuous_block_sec=continuous_block_sec,
            continuous_command_min_sec=continuous_command_min_sec,
            continuous_command_max_sec=continuous_command_max_sec,
            continuous_gap_min_sec=continuous_gap_min_sec,
            continuous_gap_max_sec=continuous_gap_max_sec,
            include_eyes_closed_rest_in_gate_neg=bool(self.eyes_closed_for_gate_check.isChecked()),
            artifact_types=list(artifact_types),
            reference_mode=self.reference_edit.text().strip(),
            participant_state=str(self.participant_state_combo.currentData()),
            caffeine_intake=str(self.caffeine_combo.currentData()),
            recent_exercise=str(self.exercise_combo.currentData()),
            sleep_note=self.sleep_edit.text().strip(),
            random_seed=int(self.seed_spin.value()),
            operator="",
            notes=self.notes_edit.toPlainText().strip(),
            board_name=self.board_display_name(),
        )

    @staticmethod
    def _validate_protocol_settings(settings: SessionSettings) -> None:
        continuous_block_count = int(settings.continuous_block_count)
        continuous_block_sec = float(settings.continuous_block_sec)
        if continuous_block_count <= 0 or continuous_block_sec <= 0:
            return

        run_count = int(settings.run_count)
        command_min = float(settings.continuous_command_min_sec)
        command_max = float(settings.continuous_command_max_sec)
        gap_min = float(settings.continuous_gap_min_sec)
        gap_max = float(settings.continuous_gap_max_sec)

        if command_max < command_min:
            raise ValueError("连续模式命令最长时长不能小于最短时长。")
        if gap_max < gap_min:
            raise ValueError("连续模式间隔最长时长不能小于最短时长。")
        if continuous_block_count > run_count:
            raise ValueError(
                f"连续模式段数不能大于 MI run 数（当前连续模式 {continuous_block_count} 段，"
                f"MI run {run_count} 轮）。如果只采一轮 MI，请把连续模式段数设为 0 或 1；"
                "点击“直接进入 MI 主任务”会自动关闭连续模式。"
            )
        if continuous_block_sec < command_min:
            raise ValueError("连续模式时长不能小于命令最短时长，否则无法生成任何连续命令。")

    @staticmethod
    def _protocol_mode_label(protocol_mode: str) -> str:
        return "仅 MI 主任务" if str(protocol_mode) == PROTOCOL_MODE_MI_ONLY else "完整流程"

    def _apply_protocol_mode_overrides(self, settings: SessionSettings) -> SessionSettings:
        if str(settings.protocol_mode) != PROTOCOL_MODE_MI_ONLY:
            return settings
        return replace(
            settings,
            practice_sec=0.0,
            calibration_open_sec=0.0,
            calibration_closed_sec=0.0,
            calibration_eye_sec=0.0,
            calibration_blink_sec=0.0,
            calibration_swallow_sec=0.0,
            calibration_jaw_sec=0.0,
            calibration_head_sec=0.0,
            run_rest_sec=0.0,
            long_run_rest_every=0,
            long_run_rest_sec=0.0,
            idle_block_count=0,
            idle_block_sec=0.0,
            idle_prepare_block_count=0,
            idle_prepare_sec=0.0,
            continuous_block_count=0,
            continuous_block_sec=0.0,
            include_eyes_closed_rest_in_gate_neg=False,
        )

    @staticmethod
    def _format_duration_text(duration_sec: float) -> str:
        seconds = max(0.0, float(duration_sec))
        rounded_seconds = round(seconds)
        if abs(seconds - rounded_seconds) > 1e-6:
            return f"{seconds:.1f} 秒"
        total_seconds = int(rounded_seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        parts: list[str] = []
        if hours > 0:
            parts.append(f"{hours} 小时")
        if minutes > 0:
            parts.append(f"{minutes} 分")
        if secs > 0 or not parts:
            parts.append(f"{secs} 秒")
        return " ".join(parts)

    @staticmethod
    def _planned_run_rest_duration(settings: SessionSettings, completed_run_index: int) -> float:
        every = int(settings.long_run_rest_every)
        if every > 0 and completed_run_index % every == 0:
            return max(0.0, float(settings.long_run_rest_sec))
        return max(0.0, float(settings.run_rest_sec))

    def _planned_imagery_start_prompt_overhead_sec(self) -> float:
        if not bool(getattr(self, "audio_prompts_enabled", True)):
            return 0.0
        return max(0.0, float(IMAGERY_START_TONE_MS + IMAGERY_START_POST_TONE_GUARD_MS) / 1000.0)

    def _build_session_plan(self, settings: SessionSettings) -> dict[str, object]:
        run_count = max(0, int(settings.run_count))
        class_count = len(CLASS_LOOKUP)
        trials_per_run = max(0, int(settings.trials_per_class)) * class_count
        total_trials = trials_per_run * run_count
        trial_phase_duration_sec = (
            max(0.0, float(settings.baseline_sec))
            + max(0.0, float(settings.cue_sec))
            + max(0.0, float(settings.imagery_sec))
            + max(0.0, float(settings.iti_sec))
        )
        trial_start_prompt_overhead_sec = self._planned_imagery_start_prompt_overhead_sec()
        trial_duration_sec = trial_phase_duration_sec + trial_start_prompt_overhead_sec
        mi_duration_sec = run_count * trials_per_run * trial_duration_sec
        calibration_plan = self._build_calibration_plan(settings)
        continuous_schedule = self._build_continuous_schedule(settings)
        stages: list[dict[str, object]] = []
        total_duration_sec = 0.0

        def add_stage(title: str, duration_sec: float) -> None:
            nonlocal total_duration_sec
            duration_value = max(0.0, float(duration_sec))
            total_duration_sec += duration_value
            if duration_value <= 0:
                return
            stages.append({"title": title, "duration_sec": duration_value})

        if calibration_plan:
            for step in calibration_plan:
                add_stage(PHASE_LABELS.get(str(step["phase"]), str(step["phase"])), float(step["duration"]))

        if float(settings.practice_sec) > 0:
            add_stage("想象训练", float(settings.practice_sec))

        continuous_index = 0
        for run_index in range(1, run_count + 1):
            add_stage(
                f"MI run {run_index}（{trials_per_run} 个试次）",
                trials_per_run * trial_duration_sec,
            )
            if continuous_index < len(continuous_schedule) and int(continuous_schedule[continuous_index]) == run_index:
                continuous_index += 1
                add_stage(f"连续模式 {continuous_index}", float(settings.continuous_block_sec))
            if run_index < run_count:
                rest_duration = self._planned_run_rest_duration(settings, run_index)
                if rest_duration > 0:
                    add_stage(f"轮次间休息（run {run_index} 后）", rest_duration)

        if int(settings.idle_block_count) > 0 and float(settings.idle_block_sec) > 0:
            for block_index in range(1, int(settings.idle_block_count) + 1):
                add_stage(f"无控制 {block_index}", float(settings.idle_block_sec))

        return {
            "title": f"{self._protocol_mode_label(settings.protocol_mode)}计划",
            "protocol_mode": str(settings.protocol_mode),
            "run_count": run_count,
            "class_count": class_count,
            "trials_per_class": int(settings.trials_per_class),
            "trials_per_run": trials_per_run,
            "total_trials": total_trials,
            "trial_phase_duration_sec": trial_phase_duration_sec,
            "trial_start_prompt_overhead_sec": trial_start_prompt_overhead_sec,
            "trial_duration_sec": trial_duration_sec,
            "mi_duration_sec": mi_duration_sec,
            "total_duration_sec": total_duration_sec,
            "continuous_schedule": list(continuous_schedule),
            "stages": stages,
            "baseline_sec": float(settings.baseline_sec),
            "cue_sec": float(settings.cue_sec),
            "imagery_sec": float(settings.imagery_sec),
            "iti_sec": float(settings.iti_sec),
        }

    def _format_session_plan(self, plan: dict[str, object]) -> str:
        stages = [dict(item) for item in plan.get("stages", [])]

        lines = [
            str(plan.get("title", "采集计划")),
            "",
            f"总时间：{self._format_duration_text(float(plan.get('total_duration_sec', 0.0)))}",
        ]
        prompt_overhead_sec = float(plan.get("trial_start_prompt_overhead_sec", 0.0))
        if prompt_overhead_sec > 0:
            lines.append(
                f"每试次开始音过渡：{self._format_duration_text(prompt_overhead_sec)}"
                "（用于对齐 UI、提示音和 imagery_start marker，不计入正式 imagery segment）"
            )
        lines.extend(["", "阶段列表："])

        for index, stage in enumerate(stages, start=1):
            duration_sec = float(stage.get("duration_sec", 0.0))
            lines.append(f"{index}. {stage.get('title', '--')}：{self._format_duration_text(duration_sec)}")

        return "\n".join(lines)

    def _show_session_plan_dialog(self, *, title: str, plan_text: str) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(920, 680)
        layout = QVBoxLayout(dialog)
        intro_label = QLabel("以下内容按当前界面配置动态生成，用于开始前快速核对流程顺序、各段时间和总时间。")
        intro_label.setWordWrap(True)
        plan_text_edit = QTextEdit(dialog)
        plan_text_edit.setReadOnly(True)
        plan_text_edit.setPlainText(plan_text)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(dialog.accept)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(close_button)
        layout.addWidget(intro_label, stretch=0)
        layout.addWidget(plan_text_edit, stretch=1)
        layout.addLayout(button_row)
        dialog.exec_()

    def _show_session_plan_for_protocol_mode(self, protocol_mode: str) -> None:
        if self.device_info is None or self.worker is None:
            self.show_error("请先连接设备，再查看计划。")
            return
        if self.session_running or self.waiting_for_save:
            self.show_error("当前会话正在运行或保存中，暂时不能查看新的开始计划。")
            return
        if self.preview_mode_switch_pending:
            self.show_error("质量检查模式仍在切换中，请稍候再试。")
            return

        try:
            settings = self.collect_settings()
            if str(protocol_mode or PROTOCOL_MODE_FULL) != str(settings.protocol_mode):
                settings = replace(settings, protocol_mode=str(protocol_mode or PROTOCOL_MODE_FULL))
            settings = self._apply_protocol_mode_overrides(settings)
            self._validate_protocol_settings(settings)
            plan = self._build_session_plan(settings)
        except Exception as error:
            self.show_error(str(error))
            return

        self._show_session_plan_dialog(title=f"{self._protocol_mode_label(settings.protocol_mode)}计划", plan_text=self._format_session_plan(plan))

    def on_worker_status(self, message: str) -> None:
        self.log(message)

    def on_worker_error(self, message: str) -> None:
        if self.session_running and not self.waiting_for_save:
            self._abort_session_due_to_runtime_failure(str(message), source="worker_error")
            return
        self.show_error(message)

    def set_config_enabled(self, enabled: bool) -> None:
        widgets = [
            self.subject_edit,
            self.session_edit,
            self.output_edit,
            self.board_combo,
            self.serial_combo,
            self.refresh_ports_button,
            self.channel_names_edit,
            self.channel_positions_edit,
            self.trials_spin,
            self.baseline_spin,
            self.cue_spin,
            self.imagery_spin,
            self.iti_spin,
            self.run_count_spin,
            self.max_consecutive_spin,
            self.run_rest_spin,
            self.long_run_every_spin,
            self.long_run_rest_spin,
            self.practice_spin,
            self.calib_open_spin,
            self.calib_closed_spin,
            self.calib_eye_spin,
            self.calib_blink_spin,
            self.calib_swallow_spin,
            self.calib_jaw_spin,
            self.calib_head_spin,
            self.idle_count_spin,
            self.idle_sec_spin,
            self.continuous_count_spin,
            self.continuous_sec_spin,
            self.cont_cmd_min_spin,
            self.cont_cmd_max_spin,
            self.cont_gap_min_spin,
            self.cont_gap_max_spin,
            self.artifact_types_edit,
            self.eyes_closed_for_gate_check,
            self.reference_edit,
            self.participant_state_combo,
            self.caffeine_combo,
            self.exercise_combo,
            self.sleep_edit,
            self.seed_spin,
            self.separate_screen_check,
            self.cue_voice_check,
            self.notes_edit,
        ]
        for widget in widgets:
            widget.setEnabled(enabled)
        self.refresh_board_input_state()

    def _training_cue_voice_enabled(self) -> bool:
        return bool(getattr(self, "cue_voice_check", None) is not None and self.cue_voice_check.isChecked())

    def _sync_cue_voice_timing_constraints(self, _checked: bool | None = None) -> None:
        if getattr(self, "cue_spin", None) is None:
            return
        if self._training_cue_voice_enabled():
            self.cue_spin.setMinimum(MIN_MI_CUE_SEC_FOR_NATURAL_AUDIO)
            if float(self.cue_spin.value()) < MIN_MI_CUE_SEC_FOR_NATURAL_AUDIO:
                self.cue_spin.setValue(MIN_MI_CUE_SEC_FOR_NATURAL_AUDIO)
            self.cue_spin.setToolTip("训练用 cue 语音已开启：提示阶段至少 2.0 秒，避免语音压到开始音。")
        else:
            self.cue_spin.setMinimum(MIN_MI_CUE_SEC_FOR_STANDARD_VISUAL)
            self.cue_spin.setToolTip("标准校准默认 2.0 秒视觉 cue，让受试者确认任务；正式 imagery 从开始音和 imagery_start marker 后开始。")

    def _class_ui_name(self, class_name: str | None) -> str:
        if not class_name:
            return ""
        if class_name in CLASS_UI_NAMES:
            return CLASS_UI_NAMES[class_name]
        info = CLASS_LOOKUP.get(class_name, {})
        return str(info.get("short_name") or info.get("display_name") or class_name)

    def _class_prep_hint(self, class_name: str | None) -> str:
        name = self._class_ui_name(class_name)
        return f"任务类别：{name}。听到开始音后开始想象，当前保持身体不动。"

    def _class_imagery_hint(self, class_name: str | None) -> str:
        if not class_name:
            return "请保持想象状态，不要做真实动作。"
        return CLASS_UI_IMAGERY_HINTS.get(class_name, f"持续想象{self._class_ui_name(class_name)}运动，不要做真实动作。")

    def _phase_texts(self, phase: str, class_name: str | None) -> tuple[str, str]:
        if phase == "idle":
            return "等待开始", "连接设备后先观察原始波形并确认稳定，再选择“开始完整流程”或“直接进入 MI 主任务”。"
        if phase == "calibration_open":
            return "睁眼静息", "注视中央十字，不做任何运动想象，保持自然眨眼。"
        if phase == "calibration_closed":
            return "闭眼静息", "闭眼放松，不做运动想象。"
        if phase == "calibration_eye_move":
            return "眼动校准", "按提示做左右/上下眼动，不转头。"
        if phase == "calibration_blink":
            return "眨眼伪迹", "按节奏眨眼，包含慢眨与快眨。"
        if phase == "calibration_swallow":
            return "吞咽伪迹", "自然吞咽，不做运动想象。"
        if phase == "calibration_jaw":
            return "咬牙伪迹", "轻微咬牙，避免大幅头动。"
        if phase == "calibration_head":
            return "头动伪迹", "轻微左右/上下摆头，动作保持可控。"
        if phase == "practice":
            return "想象训练", "强调本体感觉想象：想象肌肉发力和关节感觉，而不是视觉动画。"
        if phase == "baseline":
            return "准备阶段", "请注视中央十字，保持放松稳定；任务类别会在下一阶段出现。"
        if phase == "imagery_ready":
            return "开始音", "保持注视，听到短提示音完整结束后立即开始想象。"
        if phase == "iti":
            return "休息恢复", "放空当前想象内容，等待下一轮任务提示。"
        if phase == "run_rest":
            return "轮次间休息", "请放松并准备下一轮，可口头反馈哪个类别更难想象。"
        if phase == "idle_block":
            return "无控制", "保持注意但不要做运动想象，自然眨眼即可。"
        if phase == "continuous":
            if self.current_continuous_prompt is not None:
                label = str(self.current_continuous_prompt.get("class_label", ""))
                if label == "no_control":
                    return "连续仿真", "当前命令：无控制（保持注意，不执行运动想象）。"
                if self.current_settings is not None:
                    cmd_min = float(self.current_settings.continuous_command_min_sec)
                    cmd_max = float(self.current_settings.continuous_command_max_sec)
                else:
                    cmd_min = float(self.config.get("continuous_command_min_sec", DEFAULT_CONFIG["continuous_command_min_sec"]))
                    cmd_max = float(self.config.get("continuous_command_max_sec", DEFAULT_CONFIG["continuous_command_max_sec"]))
                if abs(cmd_max - cmd_min) < 1e-6:
                    duration_text = f"{cmd_min:g} 秒"
                else:
                    duration_text = f"{cmd_min:g}-{cmd_max:g} 秒"
                return "连续仿真", f"当前命令：{self._class_ui_name(label)}（持续 {duration_text}）。"
            return "连续仿真", "等待下一条随机命令。"
        if phase == "paused":
            return "已暂停", "恢复后将从当前阶段继续。"
        if class_name in CLASS_LOOKUP:
            name = self._class_ui_name(class_name)
            if phase == "cue":
                return f"任务提示：{name}", self._class_prep_hint(class_name)
            if phase == "imagery":
                return f"想象：{name}", self._class_imagery_hint(class_name)
        return PHASE_LABELS.get(phase, phase), ""

    def _participant_prompt(self) -> tuple[str, str, str, str | None]:
        if self.session_paused:
            return ("已暂停", "已暂停", "请保持稳定，等待继续。", None)
        if self.current_phase == "calibration_open":
            return ("静息", "睁眼静息", "注视中央十字，不做运动想象。", None)
        if self.current_phase == "calibration_closed":
            return ("静息", "闭眼静息", "闭眼并保持放松，不做运动想象。", None)
        if self.current_phase == "calibration_eye_move":
            return ("伪迹校准", "眼动", "按提示只动眼睛，不要转头。", None)
        if self.current_phase == "calibration_blink":
            return ("伪迹校准", "眨眼", "按节奏眨眼。", None)
        if self.current_phase == "calibration_swallow":
            return ("伪迹校准", "吞咽", "自然吞咽，不做想象。", None)
        if self.current_phase == "calibration_jaw":
            return ("伪迹校准", "咬牙", "轻微咬牙。", None)
        if self.current_phase == "calibration_head":
            return ("伪迹校准", "头动", "轻微摆头。", None)
        if self.current_phase == "practice":
            return ("训练", "想象训练", "想象真实发力和关节感觉，不要做真实动作。", None)
        if self.current_phase == "baseline":
            return ("准备", "准备阶段", "请注视中央标记，保持放松与稳定。", None)
        if self.current_phase == "imagery_ready":
            return ("开始音", "听提示音", "滴声结束后立即开始想象。", None)
        if self.current_phase == "iti":
            return ("休息", "休息恢复", "请放空当前想象，等待下一个试次。", None)
        if self.current_phase == "run_rest":
            return ("休息", "轮次间休息", "请放松并保持注意，稍后继续。", None)
        if self.current_phase == "idle_block":
            return ("无控制", "无控制", "保持注意，不要执行任何运动想象。", None)
        if self.current_phase == "continuous":
            if self.current_continuous_prompt is not None:
                label = str(self.current_continuous_prompt.get("class_label", ""))
                if label == "no_control":
                    return ("连续模式", "无控制", "保持注意，不执行运动想象。", None)
                return ("连续模式", f"命令：{self._class_ui_name(label)}", "持续执行该想象直到下一条命令。", label)
            return ("连续模式", "等待命令", "请等待随机命令。", None)
        if self.current_phase == "paused":
            return ("已暂停", "已暂停", "请保持稳定，等待继续。", None)
        if self.current_phase == "idle":
            return ("等待开始", "等待开始", "请等待实验开始。", None)
        if self.current_trial is not None:
            class_name = self.current_trial.class_name
            name = self._class_ui_name(class_name)
            if self.current_phase == "cue":
                return ("任务提示", name, self._class_prep_hint(class_name), class_name)
            if self.current_phase == "imagery":
                return ("想象执行", f"想象：{name}", self._class_imagery_hint(class_name), class_name)
        return ("等待开始", "等待开始", "请等待实验开始。", None)

    def _participant_countdown_text(self) -> str:
        if self.current_phase == "idle":
            return "--"
        if self.current_phase == "imagery_ready":
            return "--"
        if self.phase_prompt_pending:
            return "听提示"
        if self.session_paused:
            return f"{self.remaining_phase_sec:.1f}s"
        return f"{max(0.0, self.remaining_phase_sec):.1f}s"

    def _sync_participant_display(self) -> None:
        if not self.use_separate_participant_screen or self.participant_window is None:
            return
        stage_text, title, subtitle, class_name = self._participant_prompt()
        self.participant_window.set_prompt(
            phase=self.current_phase if not self.session_paused else "paused",
            class_name=class_name,
            stage_text=stage_text,
            title=title,
            subtitle=subtitle,
            countdown_text=self._participant_countdown_text(),
        )

    def show_participant_display(self) -> None:
        if not self.use_separate_participant_screen:
            return
        self._sync_participant_display()
        self.participant_window.showFullScreen()
        self.participant_window.raise_()
        self.participant_window.activateWindow()
        self.operator_hidden_for_session = True
        self.hide()
        self._process_ui_events()

    def _process_ui_events(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        try:
            app.processEvents(QEventLoop.ExcludeUserInputEvents | QEventLoop.ExcludeSocketNotifiers, 50)
        except Exception:
            pass

    def _current_marker_flush_delay_sec(self) -> float:
        delay_func = getattr(self.worker, "_marker_flush_delay_sec", None)
        if callable(delay_func):
            try:
                return max(0.0, float(delay_func()))
            except Exception:
                pass
        return MARKER_FLUSH_FALLBACK_SEC

    def _wait_remaining_marker_flush_since(self, marker_insert_started_perf: float) -> None:
        delay_sec = self._current_marker_flush_delay_sec()
        elapsed_sec = max(0.0, time.perf_counter() - float(marker_insert_started_perf))
        remaining_sec = max(0.0, delay_sec - elapsed_sec)
        if remaining_sec > 0:
            time.sleep(remaining_sec)

    def _show_phase_visuals_before_prompt(
        self,
        *,
        phase: str,
        duration_sec: float,
        class_name: str | None,
    ) -> None:
        self.phase_timer.stop()
        self.current_phase = str(phase)
        self.session_paused = False
        self.phase_prompt_pending = True
        self.pause_started_perf = 0.0
        self.remaining_phase_sec = max(0.0, float(duration_sec))
        now = time.perf_counter()
        self.phase_started_perf = now
        self.phase_deadline = now + self.remaining_phase_sec
        self._refresh_phase_display()
        self.update_countdown_text()
        self.update_button_states()
        self._process_ui_events()

    def restore_operator_window(self) -> None:
        if not self.use_separate_participant_screen:
            return
        self.operator_hidden_for_session = False
        self.participant_window.hide()
        if not self.isVisible():
            self.showNormal()
        self.raise_()
        self.activateWindow()

    def _resolve_phase_accent(self, phase: str, class_name: str | None) -> str:
        if class_name in CLASS_LOOKUP and phase in {"cue", "imagery"}:
            return str(CLASS_LOOKUP[class_name]["color"])
        return PHASE_ACCENT_COLORS.get(phase, "#64748B")

    @staticmethod
    def _badge_style(tone: str) -> str:
        styles = {
            "idle": ("#F8FAFC", "#CBD5E1", "#334155"),
            "ok": ("#ECFDF5", "#A7F3D0", "#065F46"),
            "active": ("#EFF6FF", "#93C5FD", "#1D4ED8"),
            "saving": ("#EEF2FF", "#C7D2FE", "#3730A3"),
            "warning": ("#FFF7ED", "#FDBA74", "#9A3412"),
            "error": ("#FEF2F2", "#FCA5A5", "#991B1B"),
            "paused": ("#F1F5F9", "#94A3B8", "#334155"),
        }
        background, border, color = styles.get(str(tone), styles["idle"])
        return (
            f"color: {color}; background: {background}; border: 1px solid {border}; "
            "border-radius: 8px; padding: 7px 10px;"
        )

    def _set_operator_notice(self, text: str, tone: str = "idle") -> None:
        if self.operator_notice_label is None:
            return
        self.operator_notice_label.setText(str(text))
        self.operator_notice_label.setStyleSheet(self._badge_style(tone))

    def _operator_notice_for_phase(self, phase: str, class_name: str | None) -> tuple[str, str]:
        if self.session_paused:
            return ("暂停中：当前试次或命令已按无效数据处理。", "paused")
        if self.current_trial is not None and phase in MI_TRIAL_PHASES and not bool(
            getattr(self.current_trial, "accepted", True)
        ):
            return ("当前试次已标记为坏试次，仍会走完流程但不会进入训练集。", "warning")
        if phase == "continuous" and self._current_continuous_prompt_is_failed():
            return ("当前命令已标记失败，仍会走完命令但不会参与连续评估。", "warning")
        if phase == "baseline":
            return ("准备：保持静止；如已发现干扰，可直接标记坏试次。", "idle")
        if phase == "cue":
            name = self._class_ui_name(class_name)
            return (f"提示：{name}，听到开始音后再执行想象。", "active")
        if phase == "imagery_ready":
            return ("开始音：提示音完整结束后才写入 imagery_start marker 并启动想象计时。", "active")
        if phase == "imagery":
            name = self._class_ui_name(class_name)
            return (f"执行：持续想象{name}，不要真实移动。", "active")
        if phase == "iti":
            return ("休息：放空上一轮想象，等待下一试次。", "idle")
        if phase.startswith("calibration_"):
            return ("校准：按提示完成静息或伪迹动作。", "active")
        if phase == "practice":
            return ("训练：确认受试者理解想象方式后再进入正式试次。", "active")
        if phase == "continuous":
            return ("连续模式：按随机命令执行；失败命令会被排除。", "active")
        if phase in {"run_rest", "idle_block"}:
            return ("休息/无控制：保持稳定，不执行运动想象。", "idle")
        if self._save_in_progress():
            return ("保存：请等待后台写盘完成，不要断开程序。", "saving")
        return ("待机：连接设备后先完成 EEG/阻抗质量检查。", "idle")

    def _infer_status_tone(self, label: QLabel | None) -> str:
        if label is None:
            return "idle"
        text = label.text()
        if any(token in text for token in ("失败", "中止", "未保存", "丢弃")):
            return "error"
        if any(token in text for token in ("暂停", "坏试次", "质量异常", "等待保存")):
            return "warning"
        if any(token in text for token in ("保存中", "正在保存", "正在断开", "正在准备", "正在连接")):
            return "saving"
        if any(token in text for token in ("采集中", "正在进行", "正在执行", "已开始", "连续模式")):
            return "active"
        if any(token in text for token in ("已连接", "已准备", "保存完成", "有效试次")):
            return "ok"
        return "idle"

    def _refresh_status_badges(self) -> None:
        for label in (self.device_label, self.summary_label, self.current_label, self.sequence_summary_label):
            if label is not None:
                label.setStyleSheet(self._badge_style(self._infer_status_tone(label)))
        if self.accepted_label is not None:
            self.accepted_label.setStyleSheet(self._badge_style("ok"))
        if self.rejected_label is not None:
            tone = "warning" if any(char.isdigit() and char != "0" for char in self.rejected_label.text()) else "idle"
            self.rejected_label.setStyleSheet(self._badge_style(tone))

    def _refresh_control_button_tooltips(self) -> None:
        if self.current_phase == "continuous":
            if self.current_continuous_prompt is None:
                bad_trial_tooltip = "当前没有正在执行的连续命令。"
            elif self._current_continuous_prompt_is_failed():
                bad_trial_tooltip = "当前连续命令已标记失败。"
            else:
                bad_trial_tooltip = "标记当前连续命令失败。"
        elif self.current_trial is None or self.current_phase not in MI_TRIAL_PHASES:
            bad_trial_tooltip = "仅在主试次阶段可标记坏试次。"
        elif self._current_mi_trial_can_be_marked_bad():
            bad_trial_tooltip = "标记当前试次为坏试次。"
        else:
            bad_trial_tooltip = "当前试次已标记为坏试次。"

        tooltips = {
            self.connect_button: "连接当前配置的 BrainFlow 设备并开始 EEG 质量预览。",
            self.start_button: "按完整流程采集：校准、训练、MI 主任务、无控制和连续模式。",
            self.start_mi_only_button: "跳过校准和后续模块，只采集 MI 主任务试次。",
            self.show_plan_button: "按当前参数预览完整流程的阶段顺序和预计时长。",
            self.show_mi_only_plan_button: "按当前参数预览仅 MI 主任务的阶段顺序和预计时长。",
            self.pause_button: "暂停计时并写入 pause marker；当前试次或命令会按无效数据处理。",
            self.bad_trial_button: bad_trial_tooltip,
            self.stop_button: "结束当前会话，写入结束 marker，并保存已采集数据。",
            self.disconnect_button: "停止预览并断开设备；采集中请先停止并保存。",
        }
        for button, tooltip in tooltips.items():
            if button is not None:
                button.setToolTip(tooltip)

    def _apply_phase_theme(self, phase: str, class_name: str | None) -> None:
        accent = self._resolve_phase_accent(phase, class_name)
        self.phase_label.setStyleSheet(
            f"color: #FFFFFF; background: {accent}; border-radius: 10px; padding: 8px 14px;"
        )
        self.countdown_label.setStyleSheet(
            f"color: #1E293B; font-weight: bold; "
            f"background: #F8FAFC; border: 1px solid {accent}; border-radius: 8px; padding: 10px 14px;"
        )
        self.instruction_label.setStyleSheet(
            f"color: #0F172A; background: #FFFFFF; "
            f"border: 1px solid {accent}; border-radius: 8px; padding: 12px;"
        )
        self.progress_bar.setStyleSheet(
            f"QProgressBar {{border: 1px solid #CBD5E1; border-radius: 8px; text-align: center; "
            f"background: #F8FAFC; color: #1E293B;}} "
            f"QProgressBar::chunk {{background: {accent}; border-radius: 7px;}}"
        )

    def _refresh_phase_display(self) -> None:
        class_name = None if self.current_trial is None else self.current_trial.class_name
        title, subtitle = self._phase_texts(self.current_phase, class_name)
        self.phase_label.setText(PHASE_LABELS.get(self.current_phase, self.current_phase))
        self.instruction_label.setText(subtitle or "请保持稳定并等待下一步提示。")
        self._apply_phase_theme(self.current_phase if not self.session_paused else "paused", class_name)
        notice_text, notice_tone = self._operator_notice_for_phase(self.current_phase, class_name)
        self._set_operator_notice(notice_text, notice_tone)
        self.cue_widget.set_state(
            phase=self.current_phase if not self.session_paused else "paused",
            class_name=class_name,
            title=title,
            subtitle=subtitle,
        )
        self._update_preview_status()
        self._update_trial_meta_labels()
        self._refresh_status_badges()
        self._sync_participant_display()

    def _preview_channel_count(self) -> int:
        if self.device_info is not None:
            rows = self.device_info.get("selected_rows", [])
            if isinstance(rows, list) and rows:
                return max(1, int(len(rows)))
        if self.preview_widget is not None and self.preview_widget.channel_names:
            return max(1, int(len(self.preview_widget.channel_names)))
        return max(1, len(DEFAULT_CHANNEL_NAMES))

    def _refresh_preview_toggle_button_styles(self) -> None:
        preview_focus = self._preview_focus_active()
        padding_v = 3 if preview_focus else 6
        padding_h = 9 if preview_focus else 12
        radius = 8

        target_mode = None
        if self.preview_mode_switch_pending and self.preview_mode_switch_target is not None:
            target_mode = str(self.preview_mode_switch_target.get("mode", RealtimeEEGPreviewWidget.MODE_EEG))
        current_mode = str(target_mode or self.preview_mode)

        schemes = {
            RealtimeEEGPreviewWidget.MODE_EEG: {
                "active_bg": "#2563EB",
                "active_border": "#1D4ED8",
                "active_fg": "#FFFFFF",
                "idle_bg": "#EFF6FF",
                "idle_border": "#93C5FD",
                "idle_fg": "#1D4ED8",
                "hover_bg": "#DBEAFE",
            },
            RealtimeEEGPreviewWidget.MODE_IMPEDANCE: {
                "active_bg": "#D97706",
                "active_border": "#B45309",
                "active_fg": "#FFFFFF",
                "idle_bg": "#FFF7ED",
                "idle_border": "#FBBF24",
                "idle_fg": "#B45309",
                "hover_bg": "#FFEDD5",
            },
        }

        for button, mode in (
            (self.preview_to_eeg_button, RealtimeEEGPreviewWidget.MODE_EEG),
            (self.preview_to_imp_button, RealtimeEEGPreviewWidget.MODE_IMPEDANCE),
        ):
            if button is None:
                continue
            scheme = schemes[mode]
            is_active = self.device_info is not None and current_mode == mode
            if not button.isEnabled():
                if is_active:
                    background = scheme["idle_bg"]
                    border = scheme["idle_border"]
                    foreground = scheme["idle_fg"]
                else:
                    background = "#E2E8F0"
                    border = "#CBD5E1"
                    foreground = "#94A3B8"
                hover = background
            elif is_active:
                background = scheme["active_bg"]
                border = scheme["active_border"]
                foreground = scheme["active_fg"]
                hover = scheme["active_border"]
            else:
                background = scheme["idle_bg"]
                border = scheme["idle_border"]
                foreground = scheme["idle_fg"]
                hover = scheme["hover_bg"]
            button.setStyleSheet(
                "QPushButton { "
                f"background: {background}; color: {foreground}; border: 1px solid {border}; "
                f"border-radius: {radius}px; padding: {padding_v}px {padding_h}px; "
                "font-weight: 700; } "
                f"QPushButton:hover {{ background: {hover}; }} "
                f"QPushButton:pressed {{ background: {border}; }} "
                "QPushButton:disabled { opacity: 1; }"
            )

    def _apply_preview_mode_label_display(self) -> None:
        if self.preview_mode_label is None:
            return

        preview_focus = self._preview_focus_active()
        focus_compact = preview_focus and (
            (self.preview_group.width() if self.preview_group is not None else self.width()) < 760
        )
        if self.preview_mode_switch_pending and self.preview_mode_switch_target is not None:
            target_mode = str(self.preview_mode_switch_target.get("mode", RealtimeEEGPreviewWidget.MODE_EEG))
            label_text = "切换中：8通道阻抗" if target_mode == RealtimeEEGPreviewWidget.MODE_IMPEDANCE else "切换中：EEG预览"
            tooltip_text = "质量检查：正在切换到 8 通道阻抗模式" if target_mode == RealtimeEEGPreviewWidget.MODE_IMPEDANCE else "质量检查：正在切换到 EEG 预览"
            tone = "switch"
        elif self.device_info is None:
            label_text = "等待连接"
            tooltip_text = "质量检查：等待设备连接"
            tone = "idle"
        elif self.preview_mode == RealtimeEEGPreviewWidget.MODE_IMPEDANCE:
            label_text = "8通道阻抗"
            tooltip_text = "质量检查模式：8 通道阻抗，显示原始波形并同步估计每条通道的接触阻抗。"
            tone = "impedance"
        else:
            label_text = "EEG 8通道波形"
            tooltip_text = "质量检查模式：EEG 预览仅对显示做轻量预处理，保存仍保持原始采集数据。"
            tone = "eeg"

        display_text = tooltip_text if not preview_focus else label_text
        tone_styles = {
            "idle": "color: #334155; background: #E2E8F0; border: 1px solid #CBD5E1;",
            "switch": "color: #312E81; background: #E0E7FF; border: 1px solid #C7D2FE;",
            "eeg": "color: #1D4ED8; background: #DBEAFE; border: 1px solid #93C5FD;",
            "impedance": "color: #92400E; background: #FEF3C7; border: 1px solid #FCD34D;",
        }
        self.preview_mode_label.setText(display_text)
        self.preview_mode_label.setToolTip(tooltip_text)
        self.preview_mode_label.setStyleSheet(
            f"{tone_styles[tone]} font-weight: 700; border-radius: 8px; padding: 6px 10px;"
        )
        self._refresh_preview_toggle_button_styles()

    def _compose_preview_status_text(self) -> str:
        if self.preview_mode_switch_pending and self.preview_mode_switch_target is not None:
            target_mode = str(self.preview_mode_switch_target.get("mode", RealtimeEEGPreviewWidget.MODE_EEG))
            if target_mode == RealtimeEEGPreviewWidget.MODE_IMPEDANCE:
                return "正在切换到 8 通道阻抗质量检查模式。切换完成后会同时显示 8 条通道的原始波形和阻抗估计。"
            return "正在切换回 EEG 预览模式并恢复默认板卡设置。完成后可直接开始正式采集。"

        if self.preview_mode == RealtimeEEGPreviewWidget.MODE_IMPEDANCE:
            mode_hint = "当前模式：8 通道阻抗模式，显示原始波形并同步估计每条通道的接触阻抗。"
        else:
            mode_hint = "当前模式：EEG 模式，仅对显示做轻量预处理；实际保存的数据始终保持原始采集值。"

        if self.device_info is None:
            return (
                "连接设备后先在预览面板完成人工质量检查。建议至少观察 30 秒以上，确认无掉线、坏道、饱和、异常漂移和接触不良。"
                f"{mode_hint}"
            )
        if self.session_running:
            return f"正式采集中：当前面板仅用于辅助监看。{mode_hint}"
        return (
            "设备已连接：请先在当前预览视图完成人工质量检查，确认 8 通道稳定、"
            f"无掉线/饱和/漂移后，再开始正式采集。{mode_hint}"
        )

    def _set_preview_mode_label(self) -> None:
        if self.preview_mode_label is None:
            return
        self._apply_preview_mode_label_display()
        self._refresh_preview_text_heights()

    def _apply_preview_mode_locally(self, mode: str, *, channel: int | None = None) -> None:
        normalized = (
            RealtimeEEGPreviewWidget.MODE_IMPEDANCE
            if str(mode).strip().upper().startswith("IMP")
            else RealtimeEEGPreviewWidget.MODE_EEG
        )
        channel_count = self._preview_channel_count()
        target_channel = self.preview_impedance_channel if channel is None else int(channel)
        target_channel = int(np.clip(target_channel, 1, channel_count))
        self.preview_mode = normalized
        self.preview_impedance_channel = target_channel
        if self.preview_widget is not None:
            self.preview_widget.set_quality_mode(normalized)
        self._set_preview_mode_label()
        self._update_preview_status()

    def _log_preview_mode_switch(self, mode: str, *, channel: int, reset_default: bool) -> None:
        if mode == RealtimeEEGPreviewWidget.MODE_IMPEDANCE:
            self.log("质量检查模式 -> 8通道阻抗")
        elif bool(reset_default):
            self.log("质量检查模式 -> EEG（恢复默认设置）")
        else:
            self.log("质量检查模式 -> EEG")

    def _switch_preview_mode(
        self,
        mode: str,
        *,
        channel: int | None = None,
        reset_default: bool = False,
        show_error_dialog: bool = True,
        write_log: bool = True,
    ) -> bool:
        normalized = (
            RealtimeEEGPreviewWidget.MODE_IMPEDANCE
            if str(mode).strip().upper().startswith("IMP")
            else RealtimeEEGPreviewWidget.MODE_EEG
        )
        channel_count = self._preview_channel_count()
        target_channel = self.preview_impedance_channel if channel is None else int(channel)
        target_channel = int(np.clip(target_channel, 1, channel_count))

        if self.worker is None or self.device_info is None:
            self._apply_preview_mode_locally(normalized, channel=target_channel)
            return True

        if normalized == self.preview_mode and target_channel == self.preview_impedance_channel and not bool(reset_default):
            self._apply_preview_mode_locally(normalized, channel=target_channel)
            return True

        if normalized == RealtimeEEGPreviewWidget.MODE_IMPEDANCE and not self.worker.supports_impedance_mode():
            message = "当前阻抗模式仅支持 Cyton / Cyton Daisy 板卡。"
            if show_error_dialog:
                self.show_error(message)
            else:
                self.log(message)
            return False

        if self.preview_mode_switch_pending:
            return False

        self.preview_mode_switch_pending = True
        self.preview_mode_switch_target = {
            "mode": normalized,
            "channel": int(target_channel),
            "reset_default": bool(reset_default),
            "show_error_dialog": bool(show_error_dialog),
            "write_log": bool(write_log),
        }
        self._set_preview_mode_label()
        self._update_preview_status()
        self.update_button_states()
        self.preview_mode_switch_requested.emit(normalized, int(target_channel), bool(reset_default))
        return True

    def on_preview_to_eeg_clicked(self) -> None:
        self._switch_preview_mode(RealtimeEEGPreviewWidget.MODE_EEG, channel=self.preview_impedance_channel)

    def on_preview_to_imp_clicked(self) -> None:
        self._switch_preview_mode(RealtimeEEGPreviewWidget.MODE_IMPEDANCE, channel=self.preview_impedance_channel)

    def on_preview_prev_channel_clicked(self) -> None:
        channel_count = self._preview_channel_count()
        target_channel = self.preview_impedance_channel - 1
        if target_channel < 1:
            target_channel = channel_count
        if self.preview_mode == RealtimeEEGPreviewWidget.MODE_IMPEDANCE:
            self._switch_preview_mode(
                RealtimeEEGPreviewWidget.MODE_IMPEDANCE,
                channel=target_channel,
            )
        else:
            self._apply_preview_mode_locally(self.preview_mode, channel=target_channel)

    def on_preview_next_channel_clicked(self) -> None:
        channel_count = self._preview_channel_count()
        target_channel = self.preview_impedance_channel + 1
        if target_channel > channel_count:
            target_channel = 1
        if self.preview_mode == RealtimeEEGPreviewWidget.MODE_IMPEDANCE:
            self._switch_preview_mode(
                RealtimeEEGPreviewWidget.MODE_IMPEDANCE,
                channel=target_channel,
            )
        else:
            self._apply_preview_mode_locally(self.preview_mode, channel=target_channel)

    def on_preview_reset_clicked(self) -> None:
        self._switch_preview_mode(
            RealtimeEEGPreviewWidget.MODE_EEG,
            channel=self.preview_impedance_channel,
            reset_default=True,
        )

    @pyqtSlot(object)
    def on_preview_mode_switch_finished(self, payload: object) -> None:
        payload_dict = dict(payload) if isinstance(payload, dict) else {}
        request = dict(self.preview_mode_switch_target or {})
        self.preview_mode_switch_pending = False
        self.preview_mode_switch_target = None

        ok = bool(payload_dict.get("ok"))
        target_mode = str(payload_dict.get("target_mode", request.get("mode", self.preview_mode)))
        target_channel = int(payload_dict.get("target_channel", request.get("channel", self.preview_impedance_channel)))
        reset_default = bool(payload_dict.get("reset_default", request.get("reset_default", False)))
        message = str(payload_dict.get("message", "")).strip()
        show_error_dialog = bool(request.get("show_error_dialog", True))
        write_log = bool(request.get("write_log", True))

        if ok:
            self._apply_preview_mode_locally(target_mode, channel=target_channel)
            if write_log and (self.worker is None or self.device_info is None):
                self._log_preview_mode_switch(target_mode, channel=target_channel, reset_default=reset_default)
            pending_session_start = self.pending_session_start_context
            self.pending_session_start_context = None
            if pending_session_start is not None:
                settings, sequence_by_run = pending_session_start
                self._begin_session_with_settings(settings, sequence_by_run)
                return
        else:
            self.pending_session_start_context = None
            if not message:
                message = "切换质量检查模式失败。"
            if show_error_dialog:
                self.show_error(message)
            else:
                self.log(message)

        self._set_preview_mode_label()
        self._update_preview_status()
        self.update_button_states()

    def _update_preview_status(self) -> None:
        if self.preview_status_label is None:
            return
        self.preview_status_label.setText(self._compose_preview_status_text())
        self._apply_preview_mode_label_display()
        self._refresh_preview_text_heights()

    def _start_formal_protocol(self) -> None:
        if self.use_separate_participant_screen:
            self.show_participant_display()
        else:
            self.participant_window.hide()
        if self.calibration_plan:
            first_step = self.calibration_plan[0]
            self._show_phase_visuals_before_prompt(
                phase=str(first_step["phase"]),
                duration_sec=float(first_step["duration"]),
                class_name=None,
            )
            self.record_event("calibration_start")
            if self.marker_failure_active:
                return
            self._start_next_calibration_step()
            return
        self._start_post_calibration_sequence()

    def _update_trial_meta_labels(self) -> None:
        total = len(self.sequence)
        if total <= 0:
            self.trial_banner_label.setText("当前试次：未开始")
            self.next_task_label.setText("下一任务：--")
            self.sequence_summary_label.setText("计划：尚未生成试次序列")
            self._refresh_status_badges()
            return

        if self.current_trial is not None:
            current_text = f"当前试次：第 {self.current_trial.trial_id} / {total} 个"
            if self.current_phase != "baseline":
                current_text = f"{current_text} - {self.current_trial.display_name}"
        else:
            current_text = f"当前试次：未开始，共 {total} 个"
        self.trial_banner_label.setText(current_text)

        if self.current_phase == "baseline" and self.current_trial is not None:
            next_text = "下一任务：即将显示任务提示"
        elif self.current_trial_index + 1 < total:
            upcoming = self.sequence[self.current_trial_index + 1]
            next_text = f"下一任务：{CLASS_LOOKUP[upcoming]['display_name']}"
        else:
            next_text = "下一任务：当前已是最后一个试次"
        self.next_task_label.setText(next_text)

        rejected = sum(1 for trial in self.trial_records if not trial.accepted)
        self.sequence_summary_label.setText(
            f"计划：总试次 {total} | 已完成 {self.completed_trials} | 坏试次 {rejected}"
        )
        self._refresh_status_badges()

    def _mark_incomplete_trial_if_needed(self) -> None:
        if self.current_trial is None or self.current_phase not in {"baseline", "cue", "imagery"}:
            return
        self.current_trial.accepted = False
        if not self.current_trial.note:
            self.current_trial.note = "当前会话手动提前结束"

    def _event_already_recorded(
        self,
        event_name: str,
        *,
        trial_id: int | None,
        class_name: str | None,
        run_index: int | None = None,
        run_trial_index: int | None = None,
        block_index: int | None = None,
        prompt_index: int | None = None,
    ) -> bool:
        for event in reversed(self.event_log):
            if str(event.get("event_name", "")) != str(event_name):
                continue
            if trial_id is not None and int(event.get("trial_id", -1)) != int(trial_id):
                continue
            if class_name is not None and str(event.get("class_name", "")) != str(class_name):
                continue
            if run_index is not None and int(event.get("run_index", -1)) != int(run_index):
                continue
            if run_trial_index is not None and int(event.get("run_trial_index", -1)) != int(run_trial_index):
                continue
            if block_index is not None and int(event.get("block_index", -1)) != int(block_index):
                continue
            if prompt_index is not None and int(event.get("prompt_index", -1)) != int(prompt_index):
                continue
            return True
        return False

    def _record_event_once(
        self,
        event_name: str,
        *,
        trial_id: int | None = None,
        class_name: str | None = None,
        run_index: int | None = None,
        run_trial_index: int | None = None,
        block_index: int | None = None,
        prompt_index: int | None = None,
        command_duration_sec: float | None = None,
        execution_success: int | bool | None = None,
        wait_for_marker_flush: bool = True,
    ) -> None:
        if self._event_already_recorded(
            event_name,
            trial_id=trial_id,
            class_name=class_name,
            run_index=run_index,
            run_trial_index=run_trial_index,
            block_index=block_index,
            prompt_index=prompt_index,
        ):
            return
        self.record_event(
            event_name,
            trial_id=trial_id,
            class_name=class_name,
            run_index=run_index,
            run_trial_index=run_trial_index,
            block_index=block_index,
            prompt_index=prompt_index,
            command_duration_sec=command_duration_sec,
            execution_success=execution_success,
            wait_for_marker_flush=wait_for_marker_flush,
        )

    def connect_device(self) -> None:
        if self.worker_thread is not None:
            self.show_error("设备线程已经在运行，请不要重复连接。")
            return
        if self._save_in_progress():
            self.show_error("当前正在保存数据，请等待保存完成。")
            return

        try:
            settings = self.collect_settings(log_serial_warning=True)
        except Exception as error:
            self.show_error(str(error))
            return

        self.device_label.setText("设备：正在连接…")
        self.summary_label.setText("状态：正在准备 BrainFlow 采集线程")
        self.current_label.setText("当前任务：无")
        self._set_operator_notice("连接：正在初始化设备和实时预览。", "saving")
        self.log(f"开始连接设备：{settings.board_name}")

        self.worker_thread = QThread(self)
        self.worker = BoardCaptureWorker(
            board_id=settings.board_id,
            serial_port=settings.serial_port,
            channel_positions=settings.channel_positions,
            channel_names=settings.channel_names,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.connection_ready.connect(self.on_connection_ready)
        self.worker.preview_data_ready.connect(self.on_preview_data_ready)
        self.worker.status_changed.connect(self.on_worker_status)
        self.worker.error_occurred.connect(self.on_worker_error)
        self.worker.session_data_ready.connect(self.on_session_data_ready)
        self.worker.quality_mode_switch_finished.connect(self.on_preview_mode_switch_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.on_worker_thread_finished)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_stop_requested.connect(self.worker.request_stop, Qt.DirectConnection)
        self.preview_mode_switch_requested.connect(self.worker.request_quality_mode_switch, Qt.DirectConnection)

        self.update_button_states()
        self.set_config_enabled(False)
        self.worker_thread.start()

    def on_connection_ready(self, info: dict) -> None:
        self.device_info = dict(info)
        self.preview_mode_switch_pending = False
        self.preview_mode_switch_target = None
        self.pending_session_start_context = None
        self._clear_session_result_tooltips()
        sampling_rate = float(info.get("sampling_rate", 0.0))
        channel_names = [str(item) for item in info.get("channel_names", [])]
        self.device_label.setText(
            f"设备：已连接 | 采样率 {sampling_rate:g} Hz | 通道 {', '.join(channel_names)}"
        )
        self.summary_label.setText("状态：设备已准备好，请先检查原始波形稳定性")
        self.current_label.setText("当前任务：实验员测试 / 等待开始")
        self._set_operator_notice("质量检查：确认 8 通道稳定后再开始正式采集。", "ok")
        self.preview_mode = RealtimeEEGPreviewWidget.MODE_EEG
        self.preview_impedance_channel = 1
        if self.preview_widget is not None:
            self.preview_widget.configure_stream(sampling_rate=sampling_rate, channel_names=channel_names)
            self.preview_widget.set_quality_mode(self.preview_mode, impedance_channel=self.preview_impedance_channel)
        self._set_preview_mode_label()
        self._update_preview_status()
        self._update_trial_meta_labels()
        self.log("设备连接完成。")
        self.update_button_states()

    def on_preview_data_ready(self, preview_chunk: object) -> None:
        if self.preview_widget is None:
            return
        try:
            preview_array = np.asarray(preview_chunk, dtype=np.float32)
        except Exception:
            return
        if preview_array.ndim != 2 or preview_array.size == 0:
            return
        self.preview_widget.append_chunk(preview_array)

    def _current_mi_trial_can_be_marked_bad(self) -> bool:
        return (
            self.current_trial is not None
            and self.current_phase in MI_TRIAL_PHASES
            and bool(getattr(self.current_trial, "accepted", True))
        )

    def _current_continuous_prompt_is_failed(self) -> bool:
        if self.current_phase != "continuous" or self.current_continuous_prompt is None:
            return False
        try:
            return int(self.current_continuous_prompt.get("execution_success", 1)) == 0
        except Exception:
            return False

    def _current_continuous_prompt_can_be_marked_failed(self) -> bool:
        return (
            self.current_phase == "continuous"
            and self.current_continuous_prompt is not None
            and not self._current_continuous_prompt_is_failed()
        )

    def update_button_states(self) -> None:
        connected = self.device_info is not None
        save_busy = self._save_in_progress()
        busy_connecting = self.worker_thread is not None and self.device_info is None and not save_busy
        preview_switch_busy = self.preview_mode_switch_pending

        self.connect_button.setEnabled(not connected and not busy_connecting and not save_busy and not preview_switch_busy)
        self.start_button.setEnabled(connected and not self.session_running and not save_busy and not preview_switch_busy)
        self.start_mi_only_button.setEnabled(connected and not self.session_running and not save_busy and not preview_switch_busy)
        self.show_plan_button.setEnabled(connected and not self.session_running and not save_busy and not preview_switch_busy)
        self.show_mi_only_plan_button.setEnabled(
            connected and not self.session_running and not save_busy and not preview_switch_busy
        )
        self.pause_button.setEnabled(self.session_running and not save_busy)
        self.pause_button.setText("继续" if self.session_paused else "暂停")
        if self.current_phase == "continuous":
            self.bad_trial_button.setText("标记命令失败")
        else:
            self.bad_trial_button.setText("标记坏试次")
        self.bad_trial_button.setEnabled(
            self.session_running
            and not self.session_paused
            and (self._current_mi_trial_can_be_marked_bad() or self._current_continuous_prompt_can_be_marked_failed())
        )
        self.stop_button.setEnabled(self.session_running and self.worker_thread is not None and not save_busy)
        self.disconnect_button.setEnabled(
            connected and not self.session_running and not save_busy and not preview_switch_busy
        )
        self._apply_control_button_visibility()

        preview_controls_enabled = connected and not self.session_running and not save_busy and not preview_switch_busy
        supports_impedance_fn = getattr(self.worker, "supports_impedance_mode", None)
        supports_impedance = bool(supports_impedance_fn()) if callable(supports_impedance_fn) else False
        if self.preview_to_eeg_button is not None:
            self.preview_to_eeg_button.setEnabled(preview_controls_enabled)
        if self.preview_to_imp_button is not None:
            self.preview_to_imp_button.setEnabled(preview_controls_enabled and supports_impedance)
        if self.preview_prev_ch_button is not None:
            self.preview_prev_ch_button.setEnabled(False)
            self.preview_prev_ch_button.hide()
        if self.preview_next_ch_button is not None:
            self.preview_next_ch_button.setEnabled(False)
            self.preview_next_ch_button.hide()
        if self.preview_reset_button is not None:
            self.preview_reset_button.setEnabled(False)
            self.preview_reset_button.hide()
        self._refresh_control_button_tooltips()
        self._refresh_status_badges()
        self._set_preview_mode_label()

        editable = self.worker_thread is None and not save_busy
        self.set_config_enabled(editable)
        self._update_responsive_chrome()

    def _build_calibration_plan(self, settings: SessionSettings) -> list[dict[str, object]]:
        plan: list[dict[str, object]] = []

        def add_step(phase: str, start_event: str, end_event: str, duration: float) -> None:
            if duration <= 0:
                return
            plan.append(
                {
                    "phase": phase,
                    "start_event": start_event,
                    "end_event": end_event,
                    "duration": float(duration),
                }
            )

        add_step("calibration_open", "eyes_open_rest_start", "eyes_open_rest_end", settings.calibration_open_sec)
        add_step("calibration_closed", "eyes_closed_rest_start", "eyes_closed_rest_end", settings.calibration_closed_sec)
        if "eye_movement" in settings.artifact_types:
            add_step("calibration_eye_move", "eye_movement_block_start", "eye_movement_block_end", settings.calibration_eye_sec)
        if "blink" in settings.artifact_types:
            add_step("calibration_blink", "blink_block_start", "blink_block_end", settings.calibration_blink_sec)
        if "swallow" in settings.artifact_types:
            add_step("calibration_swallow", "swallow_block_start", "swallow_block_end", settings.calibration_swallow_sec)
        if "jaw" in settings.artifact_types:
            add_step("calibration_jaw", "jaw_block_start", "jaw_block_end", settings.calibration_jaw_sec)
        if "head_motion" in settings.artifact_types:
            add_step("calibration_head", "head_motion_block_start", "head_motion_block_end", settings.calibration_head_sec)
        return plan

    def _build_continuous_schedule(self, settings: SessionSettings) -> list[int]:
        block_count = max(0, int(settings.continuous_block_count))
        run_count = max(0, int(settings.run_count))
        if block_count <= 0 or run_count <= 0 or float(settings.continuous_block_sec) <= 0:
            return []
        if block_count == 1:
            return [run_count]

        schedule: list[int] = []
        for block_index in range(block_count):
            boundary = int(np.ceil(float((block_index + 1) * run_count) / float(block_count)))
            boundary = min(max(1, boundary), run_count)
            if not schedule or schedule[-1] != boundary:
                schedule.append(boundary)
        schedule.sort()
        return schedule

    def _phase_audio_label(self, phase: str, class_name: str | None = None) -> str:
        if phase == "baseline":
            if self.current_trial is not None:
                return f"第 {self.current_trial.trial_id} 试次准备阶段"
            return "准备阶段"
        if phase == "cue":
            name = self._class_ui_name(class_name)
            return f"{name}任务提示" if name else "任务提示"
        if phase == "imagery":
            name = self._class_ui_name(class_name)
            return f"{name}运动想象" if name else "运动想象"
        if phase == "iti":
            return "休息恢复"
        if phase == "run_rest":
            return "轮次间休息"
        if phase == "idle_block":
            return "无控制"
        if phase == "continuous":
            return "连续仿真"
        if phase == "calibration":
            return "校准流程"
        return PHASE_LABELS.get(phase, str(phase))

    def _phase_start_voice_text(self, phase: str, class_name: str | None = None) -> str:
        if phase == "cue":
            prompt = CUE_VOICE_PROMPTS.get(str(class_name or ""))
            if prompt:
                return prompt
            name = self._class_ui_name(class_name)
            return name or "任务"
        if phase == "run_rest":
            return "休息"
        if phase in MI_TRIAL_PHASES:
            return ""
        label = self._phase_audio_label(phase, class_name)
        return label if label else ""

    def _should_play_phase_start_tone(self, phase: str) -> bool:
        if phase == "imagery":
            return True
        if phase in MI_TRIAL_PHASES:
            return False
        return True

    def _should_play_phase_end_tone(self, phase: str) -> bool:
        if phase == "imagery":
            return True
        if phase in MI_TRIAL_PHASES:
            return False
        return True

    def _play_prompt_tone_blocking(self, *, frequency: int, duration_ms: int) -> None:
        if not self.audio_prompts_enabled:
            return
        if winsound is not None:
            try:
                winsound.Beep(int(frequency), int(duration_ms))
                return
            except Exception:
                pass
        app = QApplication.instance()
        if app is not None:
            app.beep()
        time.sleep(max(0.0, float(duration_ms) / 1000.0))

    def _play_imagery_start_tone_blocking(self) -> None:
        if not self.audio_prompts_enabled:
            return
        self._play_prompt_tone_blocking(frequency=IMAGERY_START_TONE_HZ, duration_ms=IMAGERY_START_TONE_MS)
        time.sleep(max(0.0, float(IMAGERY_START_POST_TONE_GUARD_MS) / 1000.0))

    def _play_prompt_tone_async(self, *, frequency: int, duration_ms: int) -> None:
        if not self.audio_prompts_enabled:
            return

        def _play() -> None:
            if winsound is not None:
                try:
                    winsound.Beep(int(frequency), int(duration_ms))
                    return
                except Exception:
                    pass
            app = QApplication.instance()
            if app is not None:
                QTimer.singleShot(0, app.beep)

        threading.Thread(target=_play, daemon=True).start()

    def _next_voice_prompt_request_id(self) -> int:
        self.voice_prompt_request_id += 1
        return int(self.voice_prompt_request_id)

    def _play_voice_prompt_blocking(self, text: str, *, timeout_sec: float) -> None:
        voice_text = str(text or "").strip()
        if not self.audio_prompts_enabled or not voice_text:
            return
        request_id = self._next_voice_prompt_request_id()
        finished = False
        loop = QEventLoop(self)

        def _on_finished(finished_id: int) -> None:
            nonlocal finished
            if int(finished_id) != int(request_id):
                return
            finished = True
            loop.quit()

        self.speech_prompt_player.playback_finished.connect(_on_finished)
        try:
            self.speech_prompt_player.say(
                voice_text,
                request_id=request_id,
                timeout_sec=max(0.1, float(timeout_sec)),
            )
            QTimer.singleShot(int(max(0.1, float(timeout_sec)) * 1000) + 50, loop.quit)
            if not finished:
                loop.exec_(QEventLoop.ExcludeUserInputEvents | QEventLoop.ExcludeSocketNotifiers)
        finally:
            try:
                self.speech_prompt_player.playback_finished.disconnect(_on_finished)
            except Exception:
                pass
        if not finished:
            self.speech_prompt_player.stop()
        time.sleep(max(0.0, float(VOICE_PROMPT_FINISH_GUARD_MS) / 1000.0))

    def _play_voice_prompt_async(self, text: str, *, timeout_sec: float) -> None:
        voice_text = str(text or "").strip()
        if not self.audio_prompts_enabled or not voice_text:
            return
        self.speech_prompt_player.say(
            voice_text,
            request_id=self._next_voice_prompt_request_id(),
            timeout_sec=max(0.1, float(timeout_sec)),
        )

    def _play_phase_start_prompt_blocking(self, phase: str, class_name: str | None = None) -> None:
        if phase == "imagery":
            self._stop_voice_prompt()
        if not self.audio_prompts_enabled:
            return
        self._process_ui_events()
        voice_text = self._phase_start_voice_text(phase, class_name)
        if voice_text:
            timeout_sec = CUE_VOICE_PROMPT_TIMEOUT_SEC if str(phase) == "cue" else PHASE_VOICE_PROMPT_TIMEOUT_SEC
            self._play_voice_prompt_blocking(voice_text, timeout_sec=timeout_sec)
        if self._should_play_phase_start_tone(phase):
            if phase == "imagery":
                self._play_imagery_start_tone_blocking()
            else:
                self._play_prompt_tone_blocking(frequency=PHASE_START_TONE_HZ, duration_ms=PHASE_START_TONE_MS)
                time.sleep(max(0.0, float(PHASE_START_POST_TONE_GUARD_MS) / 1000.0))

    def _play_cue_prompt_during_phase(self, class_name: str | None) -> None:
        if not self.audio_prompts_enabled:
            return
        self._play_prompt_tone_async(frequency=CUE_WARNING_TONE_HZ, duration_ms=CUE_WARNING_TONE_MS)
        if not self._training_cue_voice_enabled():
            return
        voice_text = self._phase_start_voice_text("cue", class_name)
        if not voice_text:
            return
        timeout_sec = CUE_VOICE_PROMPT_TIMEOUT_SEC
        if self.current_settings is not None:
            timeout_sec = min(timeout_sec, max(0.5, float(self.current_settings.cue_sec)))
        self._play_voice_prompt_async(voice_text, timeout_sec=timeout_sec)

    def _play_phase_end_prompt_blocking(self, phase: str, class_name: str | None = None) -> None:
        del class_name
        if not self.audio_prompts_enabled:
            return
        if self._should_play_phase_end_tone(phase):
            self._play_prompt_tone_blocking(frequency=PHASE_END_TONE_HZ, duration_ms=PHASE_END_TONE_MS)

    def _play_imagery_tone(self, kind: str) -> None:
        if not self.audio_prompts_enabled:
            return
        if kind == "start":
            frequency = IMAGERY_START_TONE_HZ
            duration_ms = IMAGERY_START_TONE_MS
        else:
            frequency = IMAGERY_END_TONE_HZ
            duration_ms = IMAGERY_END_TONE_MS

        def _play() -> None:
            if winsound is not None:
                try:
                    winsound.Beep(frequency, duration_ms)
                    return
                except Exception:
                    pass
            app = QApplication.instance()
            if app is not None:
                QTimer.singleShot(0, app.beep)

        threading.Thread(target=_play, daemon=True).start()

    def _stop_voice_prompt(self) -> int:
        self.speech_prompt_player.stop()
        return int(self.voice_prompt_request_id)

    def _maybe_start_scheduled_continuous(self) -> bool:
        if self.current_settings is None:
            return False
        if self.next_continuous_schedule_index >= len(self.continuous_schedule_after_runs):
            return False
        target_run = int(self.continuous_schedule_after_runs[self.next_continuous_schedule_index])
        if int(self.current_run_index) != target_run:
            return False
        self.next_continuous_schedule_index += 1
        self.start_continuous_block()
        return True

    def _start_post_calibration_sequence(self) -> None:
        if self.current_settings is None:
            return
        if self.current_settings.practice_sec > 0:
            self._show_phase_visuals_before_prompt(
                phase="practice",
                duration_sec=float(self.current_settings.practice_sec),
                class_name=None,
            )
            self._play_phase_start_prompt_blocking("practice")
            self.record_event("practice_start")
            if self.marker_failure_active:
                return
            self.enter_phase(
                phase="practice",
                duration_sec=self.current_settings.practice_sec,
                class_name=None,
                title="",
                subtitle="",
            )
            self.current_label.setText("当前任务：想象训练")
            return
        self._start_next_mi_run()

    def _start_next_calibration_step(self) -> None:
        if self.current_settings is None:
            return
        self.calibration_step_index += 1
        if self.calibration_step_index >= len(self.calibration_plan):
            self.record_event("calibration_end")
            if self.marker_failure_active:
                return
            self._play_phase_end_prompt_blocking("calibration")
            self._start_post_calibration_sequence()
            return

        step = self.calibration_plan[self.calibration_step_index]
        phase = str(step["phase"])
        duration_sec = float(step["duration"])
        self._show_phase_visuals_before_prompt(phase=phase, duration_sec=duration_sec, class_name=None)
        self._play_phase_start_prompt_blocking(phase)
        self.record_event(str(step["start_event"]))
        if self.marker_failure_active:
            return
        self.enter_phase(
            phase=phase,
            duration_sec=duration_sec,
            class_name=None,
            title="",
            subtitle="",
        )
        self.current_label.setText(f"当前任务：{PHASE_LABELS.get(str(step['phase']), str(step['phase']))}")

    def _run_rest_duration_for_completed_run(self, completed_run_index: int) -> float:
        if self.current_settings is None:
            return 0.0
        every = int(self.current_settings.long_run_rest_every)
        if every > 0 and completed_run_index % every == 0:
            return max(0.0, float(self.current_settings.long_run_rest_sec))
        return max(0.0, float(self.current_settings.run_rest_sec))

    def _start_run_rest_or_next_mi_run(self) -> None:
        if self.current_settings is None:
            self.finish_session_and_request_save(manual_stop=False)
            return
        if self.current_run_index < int(self.current_settings.run_count):
            rest_duration = self._run_rest_duration_for_completed_run(self.current_run_index)
            if rest_duration > 0:
                self._show_phase_visuals_before_prompt(
                    phase="run_rest",
                    duration_sec=rest_duration,
                    class_name=None,
                )
                self._play_phase_start_prompt_blocking("run_rest")
                self.record_event("run_rest_start", run_index=self.current_run_index)
                if self.marker_failure_active:
                    return
                self.enter_phase(
                    phase="run_rest",
                    duration_sec=rest_duration,
                    class_name=None,
                    title="",
                    subtitle="",
                )
                self.current_label.setText(f"当前任务：轮次 {self.current_run_index} 休息")
                return
            self._start_next_mi_run()
            return
        self.start_post_collection_blocks_or_finish()

    def _start_next_mi_run(self) -> None:
        if self.current_settings is None:
            return
        if self.current_run_index >= int(self.current_settings.run_count):
            self.start_post_collection_blocks_or_finish()
            return

        self.current_run_index += 1
        self.current_run_trial_index = 0
        self.record_event("mi_run_start", run_index=self.current_run_index)
        if self.marker_failure_active:
            return
        self.summary_label.setText(
            f"状态：主任务采集中，轮次 {self.current_run_index}/{int(self.current_settings.run_count)}"
        )
        self.current_label.setText(f"当前任务：轮次 {self.current_run_index} 开始")
        self.start_next_trial()

    def start_post_collection_blocks_or_finish(self) -> None:
        if self.current_settings is None:
            self.finish_session_and_request_save(manual_stop=False)
            return
        if int(self.current_settings.idle_block_count) > 0 and float(self.current_settings.idle_block_sec) > 0:
            self.idle_block_index = 0
            self.start_idle_block()
            return
        self.finish_session_and_request_save(manual_stop=False)

    def start_idle_block(self) -> None:
        if self.current_settings is None:
            return
        self.idle_block_index += 1
        self._show_phase_visuals_before_prompt(
            phase="idle_block",
            duration_sec=float(self.current_settings.idle_block_sec),
            class_name=None,
        )
        self._play_phase_start_prompt_blocking("idle_block")
        self.record_event("idle_block_start", block_index=self.idle_block_index)
        if self.marker_failure_active:
            return
        self.enter_phase(
            phase="idle_block",
            duration_sec=float(self.current_settings.idle_block_sec),
            class_name=None,
            title="",
            subtitle="",
        )
        self.current_label.setText(f"当前任务：无控制第 {self.idle_block_index} 段")

    def _finish_idle_block(self) -> None:
        if self.current_settings is None:
            return
        self.record_event("idle_block_end", block_index=self.idle_block_index)
        if self.marker_failure_active:
            return
        self._play_phase_end_prompt_blocking("idle_block")
        if self.idle_block_index < int(self.current_settings.idle_block_count):
            self.start_idle_block()
            return
        self.finish_session_and_request_save(manual_stop=False)

    def _build_continuous_prompt_plan(self, *, block_duration_sec: float) -> list[dict[str, object]]:
        if self.current_settings is None or block_duration_sec <= 0:
            return []
        rng = np.random.default_rng(int(self.current_settings.random_seed) + int(self.continuous_block_index))
        labels = list(CONTINUOUS_PROMPT_LABELS)
        command_min = float(self.current_settings.continuous_command_min_sec)
        command_max = float(self.current_settings.continuous_command_max_sec)
        gap_min = float(self.current_settings.continuous_gap_min_sec)
        gap_max = float(self.current_settings.continuous_gap_max_sec)
        current = 0.0
        prompts: list[dict[str, object]] = []
        prompt_index = 0
        while current < block_duration_sec:
            remaining_time = float(block_duration_sec - current)
            if remaining_time < command_min:
                break
            duration_high = min(command_max, remaining_time)
            duration = float(rng.uniform(command_min, duration_high))
            label_token = rng.choice(np.asarray(labels, dtype=object))
            label = str(label_token.item() if hasattr(label_token, "item") else label_token)
            prompt_index += 1
            prompts.append(
                {
                    "prompt_index": prompt_index,
                    "start_offset_sec": current,
                    "duration_sec": duration,
                    "end_offset_sec": current + duration,
                    "class_label": label,
                }
            )
            gap = float(rng.uniform(gap_min, gap_max))
            current = current + duration + gap
        return prompts

    def _close_current_continuous_prompt(self) -> None:
        if self.current_continuous_prompt is None:
            return
        prompt = dict(self.current_continuous_prompt)
        execution_success = int(bool(prompt.get("execution_success", 1)))
        self.record_event(
            "continuous_command_end",
            class_name=str(prompt["class_label"]),
            block_index=self.continuous_block_index,
            prompt_index=int(prompt["prompt_index"]),
            command_duration_sec=float(prompt["duration_sec"]),
            execution_success=execution_success,
        )
        if self.marker_failure_active:
            return
        self.current_continuous_prompt = None
        self._refresh_phase_display()

    def _update_continuous_prompt(self) -> None:
        if self.current_phase != "continuous":
            return
        elapsed = max(0.0, time.perf_counter() - self.phase_started_perf)
        if self.current_continuous_prompt is not None:
            if elapsed >= float(self.current_continuous_prompt["end_offset_sec"]):
                self._close_current_continuous_prompt()

        next_index = self.continuous_prompt_index + 1
        if next_index >= len(self.continuous_prompt_plan):
            return
        next_prompt = self.continuous_prompt_plan[next_index]
        if elapsed < float(next_prompt["start_offset_sec"]):
            return
        self.continuous_prompt_index = next_index
        self.current_continuous_prompt = dict(next_prompt)
        self.current_continuous_prompt["execution_success"] = 1
        label = str(next_prompt["class_label"])
        event_name = CONTINUOUS_PROMPT_EVENT_NAMES[label]
        self.record_event(
            event_name,
            class_name=label,
            block_index=self.continuous_block_index,
            prompt_index=int(next_prompt["prompt_index"]),
            command_duration_sec=float(next_prompt["duration_sec"]),
            execution_success=None,
        )
        if self.marker_failure_active:
            return
        self._refresh_phase_display()

    def start_continuous_block(self) -> None:
        if self.current_settings is None:
            return
        self.continuous_block_index += 1
        duration_sec = float(self.current_settings.continuous_block_sec)
        self.continuous_prompt_plan = self._build_continuous_prompt_plan(block_duration_sec=duration_sec)
        if not self.continuous_prompt_plan:
            self.continuous_block_index = max(0, self.continuous_block_index - 1)
            self.show_error("连续模式配置无法生成任何命令计划，请调整连续模式时长或命令时长后重新开始采集。")
            self.current_phase = "idle"
            self.current_continuous_prompt = None
            self.finish_session_and_request_save(manual_stop=True)
            return
        self.continuous_prompt_index = -1
        self.current_continuous_prompt = None
        self._show_phase_visuals_before_prompt(phase="continuous", duration_sec=duration_sec, class_name=None)
        self._play_phase_start_prompt_blocking("continuous")
        self.record_event("continuous_block_start", block_index=self.continuous_block_index)
        if self.marker_failure_active:
            return
        self.enter_phase(
            phase="continuous",
            duration_sec=duration_sec,
            class_name=None,
            title="",
            subtitle="",
        )
        self.current_label.setText(f"当前任务：连续模式第 {self.continuous_block_index} 段")
        self._update_continuous_prompt()

    def _finish_continuous_block(self) -> None:
        self._close_current_continuous_prompt()
        if self.marker_failure_active:
            return
        self.record_event("continuous_block_end", block_index=self.continuous_block_index)
        if self.marker_failure_active:
            return
        self._play_phase_end_prompt_blocking("continuous")
        if self._maybe_start_scheduled_continuous():
            return
        if self.current_settings is not None:
            if self.current_run_index < int(self.current_settings.run_count):
                self._start_run_rest_or_next_mi_run()
                return
            self.start_post_collection_blocks_or_finish()
            return
        self.finish_session_and_request_save(manual_stop=False)

    def _begin_session_with_settings(
        self,
        settings: SessionSettings,
        sequence_by_run: list[list[str]],
    ) -> None:
        self._clear_session_result_tooltips()
        self.current_settings = settings
        self.use_separate_participant_screen = bool(self.separate_screen_check.isChecked())
        self.sequence_by_run = sequence_by_run
        self.sequence = [item for run_items in sequence_by_run for item in run_items]
        self.trials_per_run = int(settings.trials_per_class * len(CLASS_LOOKUP))
        self.current_run_index = 0
        self.current_run_trial_index = 0
        self.event_log = []
        self.trial_records = []
        self.current_trial_index = -1
        self.current_trial = None
        self.completed_trials = 0
        self.calibration_plan = self._build_calibration_plan(settings)
        self.calibration_step_index = -1
        self.idle_block_index = 0
        self.continuous_block_index = 0
        self.continuous_schedule_after_runs = self._build_continuous_schedule(settings)
        self.next_continuous_schedule_index = 0
        self.continuous_prompt_plan = []
        self.continuous_prompt_index = -1
        self.current_continuous_prompt = None
        self.marker_failure_active = False
        self.marker_failure_message = ""
        self.runtime_failure_active = False
        self.runtime_failure_message = ""
        self.session_running = True
        self.session_paused = False
        self.phase_prompt_pending = False
        self.waiting_for_save = False
        self.capture_on_stop = True
        self.session_start_perf = time.perf_counter()

        self._update_sequence_label()
        self.refresh_trial_counters()
        if self.marker_failure_active:
            return
        self._update_progress_label()
        self._update_trial_meta_labels()

        self.summary_label.setText(f"状态：采集中，共 {len(self.sequence)} 个主任务试次")
        self.current_label.setText("当前任务：正在进入正式流程")
        self.log(f"启动模式：{self._protocol_mode_label(settings.protocol_mode)}")
        self.log(
            f"开始采集：被试 {settings.subject_id} | 会话 {settings.session_id} | "
            f"轮次 {settings.run_count} | 每类 {settings.trials_per_class} 试次"
        )
        self.log(f"本次会话随机种子（自动生成）：{settings.random_seed}")
        if self.continuous_schedule_after_runs:
            schedule_text = "、".join(str(item) for item in self.continuous_schedule_after_runs)
            self.log(f"连续模式插入点：在完成 MI run {schedule_text} 后插入。")

        self.record_event("session_start")
        if self.marker_failure_active:
            return
        self._start_formal_protocol()
        self.update_button_states()

    def start_session(self) -> None:
        self._start_session_with_protocol_mode(PROTOCOL_MODE_FULL)

    def start_mi_only_session(self) -> None:
        self._start_session_with_protocol_mode(PROTOCOL_MODE_MI_ONLY)

    def show_full_session_plan(self) -> None:
        self._show_session_plan_for_protocol_mode(PROTOCOL_MODE_FULL)

    def show_mi_only_session_plan(self) -> None:
        self._show_session_plan_for_protocol_mode(PROTOCOL_MODE_MI_ONLY)

    def _start_session_with_protocol_mode(self, protocol_mode: str) -> None:
        if self.device_info is None or self.worker is None:
            self.show_error("请先连接设备。")
            return
        if self.session_running or self._save_in_progress():
            self.show_error("当前已有采集任务在运行或正在保存。")
            return
        if self.preview_mode_switch_pending:
            self.show_error("质量检查模式仍在切换中，请稍候再试。")
            return

        # Always use a fresh seed per session and persist it in settings/metadata.
        self.seed_spin.setValue(self._generate_session_seed())

        try:
            settings = self.collect_settings()
            if str(protocol_mode or PROTOCOL_MODE_FULL) != str(settings.protocol_mode):
                settings = replace(settings, protocol_mode=str(protocol_mode or PROTOCOL_MODE_FULL))
            settings = self._apply_protocol_mode_overrides(settings)
            self._validate_protocol_settings(settings)
            sequence_by_run = [
                build_balanced_trial_sequence(
                    settings.trials_per_class,
                    settings.random_seed + run_offset,
                    settings.max_consecutive_same_class,
                )
                for run_offset in range(int(settings.run_count))
            ]
        except Exception as error:
            self.show_error(str(error))
            return

        if len(settings.channel_names) != len(self.device_info.get("selected_rows", [])):
            self.show_error("通道名称数量与连接时的通道数量不一致。请断开设备后重新连接。")
            return

        if self.worker.supports_impedance_mode() and self.preview_mode != RealtimeEEGPreviewWidget.MODE_EEG:
            self.pending_session_start_context = (settings, sequence_by_run)
            switched = self._switch_preview_mode(
                RealtimeEEGPreviewWidget.MODE_EEG,
                channel=self.preview_impedance_channel,
                reset_default=True,
                show_error_dialog=True,
                write_log=False,
            )
            if not switched:
                self.pending_session_start_context = None
            return

        self._begin_session_with_settings(settings, sequence_by_run)

    def enter_phase(
        self,
        *,
        phase: str,
        duration_sec: float,
        class_name: str | None,
        title: str,
        subtitle: str,
        started_perf: float | None = None,
    ) -> None:
        del title, subtitle
        self.current_phase = phase
        self.session_paused = False
        self.phase_prompt_pending = False
        self.pause_started_perf = 0.0
        now = time.perf_counter()
        start_perf = now if started_perf is None else min(float(started_perf), now)
        duration_value = max(0.0, float(duration_sec))
        self.phase_started_perf = start_perf
        self.phase_deadline = start_perf + duration_value
        self.remaining_phase_sec = max(0.0, self.phase_deadline - now)
        self._refresh_phase_display()
        self.update_countdown_text()
        if self.remaining_phase_sec > 0:
            self.phase_timer.start()
        else:
            self.phase_timer.stop()
            QTimer.singleShot(0, self._on_phase_tick_guarded)
        self.update_button_states()

    def update_countdown_text(self) -> None:
        if self.current_phase == "idle":
            self.countdown_label.setText("剩余时间：--")
            self._sync_participant_display()
            return

        if self.phase_prompt_pending:
            self.countdown_label.setText("提示中：倒计时将在提示音结束后开始")
            self._sync_participant_display()
            return

        if self.session_paused:
            self.countdown_label.setText(f"已暂停，当前阶段剩余 {self.remaining_phase_sec:.1f} 秒")
            self._sync_participant_display()
            return

        remaining = max(0.0, self.phase_deadline - time.perf_counter())
        self.remaining_phase_sec = remaining
        self.countdown_label.setText(f"当前阶段剩余 {remaining:.1f} 秒")
        self._sync_participant_display()

    def _on_phase_tick_guarded(self) -> None:
        try:
            self.on_phase_tick()
        except Exception as error:
            self._abort_session_due_to_runtime_failure(f"采集流程异常：{error}", source="phase_tick_exception")

    def on_phase_tick(self) -> None:
        if not self.session_running or self.session_paused:
            return

        if self.current_phase == "continuous":
            self._update_continuous_prompt()
            if self.marker_failure_active:
                return

        remaining = self.phase_deadline - time.perf_counter()
        if remaining > 0:
            self.remaining_phase_sec = remaining
            self.update_countdown_text()
            return

        self.phase_timer.stop()

        if self.current_phase.startswith("calibration_"):
            if 0 <= self.calibration_step_index < len(self.calibration_plan):
                step = self.calibration_plan[self.calibration_step_index]
                self.record_event(str(step["end_event"]))
                if self.marker_failure_active:
                    return
                self._play_phase_end_prompt_blocking(str(step["phase"]))
            self._start_next_calibration_step()
            return

        if self.current_phase == "practice":
            self.record_event("practice_end")
            if self.marker_failure_active:
                return
            self._play_phase_end_prompt_blocking("practice")
            self._start_next_mi_run()
            return

        if self.current_phase == "run_rest":
            self.record_event("run_rest_end", run_index=self.current_run_index)
            if self.marker_failure_active:
                return
            self._play_phase_end_prompt_blocking("run_rest")
            self._start_next_mi_run()
            return

        if self.current_phase == "idle_block":
            self._finish_idle_block()
            return

        if self.current_phase == "continuous":
            self._finish_continuous_block()
            return

        if self.current_phase == "baseline":
            if self.current_trial is not None:
                self.record_event(
                    "baseline_end",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
                if self.marker_failure_active:
                    return
                self._play_phase_end_prompt_blocking("baseline", self.current_trial.class_name)
                self._show_phase_visuals_before_prompt(
                    phase="cue",
                    duration_sec=self.current_settings.cue_sec if self.current_settings is not None else 0.0,
                    class_name=self.current_trial.class_name,
                )
                self.record_event(
                    "cue_start",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
                self.record_event(
                    f"cue_{self.current_trial.class_name}",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
                if self.marker_failure_active:
                    return
                self.current_label.setText(
                    f"当前任务：第 {self.current_trial.trial_id} 个试次 - {self.current_trial.display_name}（提示）"
                )
                self.enter_phase(
                    phase="cue",
                    duration_sec=self.current_settings.cue_sec if self.current_settings is not None else 0.0,
                    class_name=self.current_trial.class_name,
                    title="",
                    subtitle="",
                )
                self._play_cue_prompt_during_phase(self.current_trial.class_name)
            return

        if self.current_phase == "cue":
            if self.current_trial is not None:
                self.record_event(
                    "cue_end",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
                if self.marker_failure_active:
                    return
                self._play_phase_end_prompt_blocking("cue", self.current_trial.class_name)
            self.start_imagery_phase()
            return

        if self.current_phase == "imagery":
            if self.current_trial is not None:
                self.record_event(
                    "imagery_end",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
                if self.marker_failure_active:
                    return
                self._play_phase_end_prompt_blocking("imagery", self.current_trial.class_name)
                self._show_phase_visuals_before_prompt(
                    phase="iti",
                    duration_sec=self.current_settings.iti_sec if self.current_settings is not None else 0.0,
                    class_name=None,
                )
                self._play_phase_start_prompt_blocking("iti", self.current_trial.class_name)
                self.record_event(
                    "iti_start",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
            if self.marker_failure_active:
                return
            self.enter_phase(
                phase="iti",
                duration_sec=self.current_settings.iti_sec if self.current_settings is not None else 0.0,
                class_name=None,
                title="",
                subtitle="",
            )
            self.current_label.setText("当前任务：休息恢复")
            return

        if self.current_phase == "iti":
            if self.current_trial is not None:
                self.record_event(
                    "trial_end",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
                if self.marker_failure_active:
                    return
                self._play_phase_end_prompt_blocking("iti", self.current_trial.class_name)
                self.completed_trials = max(self.completed_trials, self.current_trial_index + 1)
                self._update_progress_label()
                self._update_sequence_label()
                self._update_trial_meta_labels()
            if self.current_settings is not None and self.current_run_trial_index < self.trials_per_run:
                self.start_next_trial()
            else:
                if self.current_settings is not None:
                    self.record_event("mi_run_end", run_index=self.current_run_index)
                    if self.marker_failure_active:
                        return
                    if self._maybe_start_scheduled_continuous():
                        return
                    if self.current_run_index < int(self.current_settings.run_count):
                        self._start_run_rest_or_next_mi_run()
                        return
                self.start_post_collection_blocks_or_finish()

    def start_next_trial(self) -> None:
        if self.current_settings is None:
            return
        if self.current_trial_index + 1 >= len(self.sequence):
            self.start_post_collection_blocks_or_finish()
            return

        self.current_trial_index += 1
        self.current_run_trial_index += 1
        class_name = self.sequence[self.current_trial_index]
        info = CLASS_LOOKUP[class_name]
        trial = TrialRecord(
            trial_id=self.current_trial_index + 1,
            class_name=class_name,
            display_name=str(info["display_name"]),
            run_index=self.current_run_index,
            run_trial_index=self.current_run_trial_index,
        )
        self.current_trial = trial
        self.trial_records.append(trial)

        self._show_phase_visuals_before_prompt(
            phase="baseline",
            duration_sec=float(self.current_settings.baseline_sec),
            class_name=None,
        )
        self._play_phase_start_prompt_blocking("baseline", class_name=class_name)
        self.record_event(
            "trial_start",
            trial_id=trial.trial_id,
            class_name=class_name,
            run_index=trial.run_index,
            run_trial_index=trial.run_trial_index,
        )
        self.record_event(
            "fixation_start",
            trial_id=trial.trial_id,
            class_name=class_name,
            run_index=trial.run_index,
            run_trial_index=trial.run_trial_index,
        )
        self.record_event(
            "baseline_start",
            trial_id=trial.trial_id,
            class_name=class_name,
            run_index=trial.run_index,
            run_trial_index=trial.run_trial_index,
        )
        self.current_label.setText(f"当前任务：第 {trial.trial_id} 个试次 - 准备阶段")
        self.summary_label.setText(f"状态：正在进行第 {trial.trial_id} / {len(self.sequence)} 个试次")
        self._update_progress_label()
        self._update_sequence_label()
        self._update_trial_meta_labels()
        if self.marker_failure_active:
            return
        self.enter_phase(
            phase="baseline",
            duration_sec=self.current_settings.baseline_sec,
            class_name=None,
            title="",
            subtitle="",
        )

    def start_imagery_phase(self) -> None:
        if self.current_settings is None or self.current_trial is None:
            return

        class_name = self.current_trial.class_name
        self._show_phase_visuals_before_prompt(
            phase="imagery_ready",
            duration_sec=0.0,
            class_name=None,
        )
        self._play_phase_start_prompt_blocking("imagery", class_name=class_name)
        imagery_started_perf = time.perf_counter()
        self.record_event(
            "imagery_start",
            trial_id=self.current_trial.trial_id,
            class_name=class_name,
            run_index=self.current_trial.run_index,
            run_trial_index=self.current_trial.run_trial_index,
            wait_for_marker_flush=False,
        )
        if self.marker_failure_active:
            return
        self.enter_phase(
            phase="imagery",
            duration_sec=self.current_settings.imagery_sec,
            class_name=class_name,
            title="",
            subtitle="",
            started_perf=imagery_started_perf,
        )
        self.current_label.setText(
            f"当前任务：第 {self.current_trial.trial_id} 个试次 - {CLASS_LOOKUP[class_name]['display_name']}（想象中）"
        )
        self.summary_label.setText(f"状态：正在执行第 {self.current_trial.trial_id} / {len(self.sequence)} 个试次")
        self._process_ui_events()
        self._wait_remaining_marker_flush_since(imagery_started_perf)
        self.record_event(
            f"imagery_{class_name}",
            trial_id=self.current_trial.trial_id,
            class_name=class_name,
            run_index=self.current_trial.run_index,
            run_trial_index=self.current_trial.run_trial_index,
        )
        if self.marker_failure_active:
            return

    @staticmethod
    def _append_note_once(note: str, addition: str) -> str:
        base = str(note).strip()
        extra = str(addition).strip()
        if not extra:
            return base
        if not base:
            return extra
        if extra in base:
            return base
        return f"{base}; {extra}"

    def _auto_reject_current_trial_for_pause(self) -> bool:
        if self.current_trial is None or self.current_phase not in MI_TRIAL_PHASES:
            return False
        if not self.current_trial.accepted:
            return False
        self.current_trial.accepted = False
        self.current_trial.note = self._append_note_once(self.current_trial.note, PAUSE_REJECTED_TRIAL_NOTE)
        self.refresh_trial_counters()
        self._update_sequence_label()
        self.log(f"第 {self.current_trial.trial_id} 个试次因暂停自动标记为坏试次。")
        return True

    def _auto_fail_current_prompt_for_pause(self) -> bool:
        if self.current_phase != "continuous" or self.current_continuous_prompt is None:
            return False
        if int(self.current_continuous_prompt.get("execution_success", 1)) == 0:
            return False
        self.current_continuous_prompt["execution_success"] = 0
        label = str(self.current_continuous_prompt.get("class_label", ""))
        prompt_index = int(self.current_continuous_prompt.get("prompt_index", 0))
        self.log(f"连续命令第 {prompt_index} 条（{self._class_ui_name(label) or label}）因暂停自动标记失败。")
        return True

    def toggle_pause(self) -> None:
        if not self.session_running or self.waiting_for_save:
            return

        if not self.session_paused:
            pause_perf = time.perf_counter()
            self.remaining_phase_sec = max(0.0, self.phase_deadline - pause_perf)
            self.pause_started_perf = pause_perf
            self.session_paused = True
            self.phase_timer.stop()
            self._stop_voice_prompt()
            trial_invalidated = self._auto_reject_current_trial_for_pause()
            prompt_failed = self._auto_fail_current_prompt_for_pause()
            if self.current_trial is not None:
                self.record_event(
                    "pause",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
            else:
                self.record_event("pause")
            if self.marker_failure_active:
                return
            self.phase_label.setText(PHASE_LABELS["paused"])
            self.cue_widget.set_state(
                phase="paused",
                class_name=None,
                title="已暂停",
                subtitle="恢复后将从当前阶段继续。",
            )
            self.countdown_label.setText(f"已暂停，当前阶段剩余 {self.remaining_phase_sec:.1f} 秒")
            if trial_invalidated:
                pause_notice = "暂停中：当前试次已自动标记为坏试次，不会进入训练集。"
            elif prompt_failed:
                pause_notice = "暂停中：当前连续命令已自动标记失败，不会参与连续评估。"
            else:
                pause_notice = "暂停中：保持稳定，恢复后继续当前阶段。"
            self.summary_label.setText(f"状态：{pause_notice}")
            self._set_operator_notice(pause_notice, "paused")
            self._apply_phase_theme("paused", None)
            self._sync_participant_display()
            self.log("采集已暂停。")
        else:
            resume_perf = time.perf_counter()
            self.session_paused = False
            paused_duration = max(0.0, resume_perf - self.pause_started_perf) if self.pause_started_perf > 0 else 0.0
            if self.phase_started_perf > 0:
                self.phase_started_perf += paused_duration
            self.pause_started_perf = 0.0
            self.phase_deadline = resume_perf + self.remaining_phase_sec
            if self.current_trial is not None:
                self.record_event(
                    "resume",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
            else:
                self.record_event("resume")
            if self.marker_failure_active:
                return
            self._refresh_phase_display()
            self.update_countdown_text()
            self.phase_timer.start()
            self.summary_label.setText("状态：采集继续进行")
            class_name = None if self.current_trial is None else self.current_trial.class_name
            self._set_operator_notice(*self._operator_notice_for_phase(self.current_phase, class_name))
            self.log("采集已继续。")

        self._refresh_status_badges()
        self.update_button_states()

    def request_phase_advance(self) -> None:
        if not self.session_running or self.waiting_for_save or self.session_paused:
            return
        if self.current_phase != "practice":
            return
        self.log("操作员确认，提前结束想象训练阶段。")
        self.phase_deadline = time.perf_counter()
        self.remaining_phase_sec = 0.0
        self.phase_timer.stop()
        self.update_countdown_text()
        self._on_phase_tick_guarded()

    def mark_bad_trial(self) -> None:
        if self.waiting_for_save or self.session_paused:
            return
        if self.current_phase == "continuous":
            self.mark_continuous_prompt_failed()
            return
        if self.current_trial is None:
            self.show_error("当前没有可标记的试次。")
            return
        if self.current_phase not in MI_TRIAL_PHASES:
            self.show_error("当前不在主试次阶段，无法标记坏试次。")
            return
        if not self.current_trial.accepted:
            self.log(f"第 {self.current_trial.trial_id} 个试次已经被标记为坏试次。")
            self.update_button_states()
            return

        self.current_trial.accepted = False
        self.current_trial.note = "操作员标记为坏试次"
        self.record_event(
            "bad_trial_marked",
            trial_id=self.current_trial.trial_id,
            class_name=self.current_trial.class_name,
            run_index=self.current_trial.run_index,
            run_trial_index=self.current_trial.run_trial_index,
        )
        if self.marker_failure_active:
            return
        self.refresh_trial_counters()
        self._update_sequence_label()
        self.log(f"已将第 {self.current_trial.trial_id} 个试次标记为坏试次。")
        self._update_trial_meta_labels()
        self._set_operator_notice("当前试次已标记为坏试次，仍会走完流程但不会进入训练集。", "warning")
        self.update_button_states()

    def mark_continuous_prompt_failed(self) -> None:
        if self.waiting_for_save or self.session_paused:
            return
        if self.current_phase != "continuous":
            self.show_error("当前不在连续模式，无法标记命令失败。")
            return
        if self.current_continuous_prompt is None:
            self.show_error("当前没有进行中的连续命令可标记。")
            return
        if int(self.current_continuous_prompt.get("execution_success", 1)) == 0:
            self.log("当前连续命令已经标记为失败。")
            self.update_button_states()
            return
        self.current_continuous_prompt["execution_success"] = 0
        label = str(self.current_continuous_prompt.get("class_label", ""))
        prompt_index = int(self.current_continuous_prompt.get("prompt_index", 0))
        self.log(f"已标记连续命令失败：第 {prompt_index} 条（{self._class_ui_name(label) or label}）。")
        self._refresh_phase_display()
        self.update_button_states()

    def refresh_trial_counters(self) -> None:
        accepted = sum(1 for trial in self.trial_records if trial.accepted)
        rejected = sum(1 for trial in self.trial_records if not trial.accepted)
        self.accepted_label.setText(f"有效试次：{accepted}")
        self.rejected_label.setText(f"坏试次：{rejected}")
        self._update_trial_meta_labels()
        self._refresh_status_badges()

    def _update_progress_label(self) -> None:
        total = len(self.sequence)
        if total <= 0:
            self.progress_bar.setValue(0)
            self.progress_text.setText("总进度：0 / 0")
            return

        started = 0
        if self.session_running and self.current_trial_index >= 0:
            started = self.current_trial_index + 1
        elif self.completed_trials:
            started = self.completed_trials

        percent = int(round(100.0 * self.completed_trials / total))
        self.progress_bar.setValue(max(0, min(100, percent)))
        self.progress_text.setText(
            f"总进度：已完成 {self.completed_trials} / {total}，当前已开始 {started} / {total}"
        )

    def _update_sequence_label(self) -> None:
        if not self.sequence:
            self.sequence_label.setText("当前还未生成试次顺序。")
            return

        parts = []
        for index, class_name in enumerate(self.sequence):
            info = CLASS_LOOKUP[class_name]
            color = str(info["color"])
            border = f"1px solid {color}"
            background = "#FFFFFF"
            text_color = color
            extra_style = ""

            if index < self.completed_trials:
                background = color
                text_color = "#FFFFFF"
            elif index == self.current_trial_index and self.session_running:
                background = "#EFF6FF"
                border = f"2px solid {color}"

            if index < len(self.trial_records) and not self.trial_records[index].accepted:
                extra_style = "text-decoration: line-through;"

            parts.append(
                f"<span style='display:inline-block; margin:2px 4px 2px 0; padding:6px 10px; "
                f"border-radius:999px; border:{border}; background:{background}; color:{text_color}; {extra_style}'>"
                f"{index + 1}. {info['short_name']}</span>"
            )

        self.sequence_label.setText("".join(parts))

    def _request_worker_stop_after_flush(self) -> None:
        if self.worker_thread is None or not self.waiting_for_save:
            return
        self.worker_stop_requested.emit()

    def finish_session_and_request_save(self, *, manual_stop: bool) -> None:
        if self.waiting_for_save or self.worker_thread is None:
            return

        self.phase_timer.stop()
        self._stop_voice_prompt()
        self.session_paused = False
        self.pause_started_perf = 0.0

        if self.session_running:
            if self.current_phase.startswith("calibration_"):
                if 0 <= self.calibration_step_index < len(self.calibration_plan):
                    step = self.calibration_plan[self.calibration_step_index]
                    self._record_event_once(str(step["end_event"]))
                self._record_event_once("calibration_end")
            elif self.current_phase == "practice":
                self._record_event_once("practice_end")
            elif self.current_phase == "run_rest":
                self._record_event_once("run_rest_end", run_index=self.current_run_index)
            elif self.current_phase == "idle_block":
                self._record_event_once("idle_block_end", block_index=self.idle_block_index)
            elif self.current_phase == "continuous":
                self._close_current_continuous_prompt()
                self._record_event_once("continuous_block_end", block_index=self.continuous_block_index)

            if self.current_phase == "baseline" and self.current_trial is not None:
                self._mark_incomplete_trial_if_needed()
                self.record_event(
                    "baseline_end",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
                self.record_event(
                    "trial_end",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
            elif self.current_phase == "cue" and self.current_trial is not None:
                self._mark_incomplete_trial_if_needed()
                self._record_event_once(
                    "cue_end",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
                self.record_event(
                    "trial_end",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
            elif self.current_phase == "imagery" and self.current_trial is not None:
                self._mark_incomplete_trial_if_needed()
                self.record_event(
                    "imagery_end",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
                self.record_event(
                    "trial_end",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                    run_index=self.current_trial.run_index,
                    run_trial_index=self.current_trial.run_trial_index,
                )
            elif self.current_phase == "iti" and self.current_trial is not None:
                if not self._event_already_recorded(
                    "trial_end",
                    trial_id=self.current_trial.trial_id,
                    class_name=self.current_trial.class_name,
                ):
                    self.record_event(
                        "trial_end",
                        trial_id=self.current_trial.trial_id,
                        class_name=self.current_trial.class_name,
                        run_index=self.current_trial.run_index,
                        run_trial_index=self.current_trial.run_trial_index,
                    )
                self.completed_trials = max(self.completed_trials, self.current_trial_index + 1)

            run_end_recorded = any(
                str(event.get("event_name", "")) == "mi_run_end"
                and int(event.get("run_index") or -1) == int(self.current_run_index)
                for event in self.event_log
            )
            if self.current_run_index > 0 and not run_end_recorded:
                self.record_event("mi_run_end", run_index=self.current_run_index)
            self._record_event_once("session_end")
            if self.marker_failure_active:
                return

        self.refresh_trial_counters()
        self._update_progress_label()
        self._update_sequence_label()
        self._update_trial_meta_labels()
        self.session_running = False
        self.waiting_for_save = True
        self.current_phase = "idle"
        self.capture_on_stop = True
        self.phase_label.setText("保存中")
        self.countdown_label.setText("正在等待设备线程返回完整数据…")
        self._apply_phase_theme("idle", None)
        self.cue_widget.set_state(
            phase="idle",
            class_name=None,
            title="正在保存",
            subtitle="请等待当前会话数据写入磁盘。",
        )
        self.participant_window.set_prompt(
            phase="idle",
            class_name=None,
            stage_text="保存中",
            title="实验结束",
            subtitle="正在保存本次采集数据，请保持静止。",
            countdown_text="...",
        )
        self.summary_label.setText("状态：采集结束，正在保存数据")
        self.current_label.setText("当前任务：保存中")
        if manual_stop:
            self.log("已手动停止采集，开始保存当前会话。")
        else:
            self.log("所有试次完成，开始保存当前会话。")

        self.update_button_states()
        # Give BrainFlow a tiny buffer window so final markers (trial_end/session_end) are
        # flushed without blocking the UI thread.
        QTimer.singleShot(80, self._request_worker_stop_after_flush)

    def stop_and_save(self) -> None:
        if self.waiting_for_save:
            self.log("当前正在保存，请稍候。")
            return
        if not self.session_running:
            self.show_error("当前没有正在进行的会话。")
            return
        self.finish_session_and_request_save(manual_stop=True)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if not self.session_running:
            super().keyPressEvent(event)
            return
        if event.key() == Qt.Key_Space:
            self.toggle_pause()
            event.accept()
            return
        if event.key() == Qt.Key_B:
            self.mark_bad_trial()
            event.accept()
            return
        if event.key() == Qt.Key_N:
            self.request_phase_advance()
            event.accept()
            return
        if event.key() == Qt.Key_Escape:
            self.stop_and_save()
            event.accept()
            return
        super().keyPressEvent(event)

    def disconnect_device(self) -> None:
        if self._save_in_progress():
            self.show_error("当前正在保存数据，请等待保存完成。")
            return
        if self.preview_mode_switch_pending:
            self.show_error("质量检查模式仍在切换中，请稍候再试。")
            return
        if self.session_running:
            self.show_error("当前正在采集，请先点击“停止并保存”。")
            return
        if self.worker_thread is None:
            self.log("当前没有已连接的设备。")
            return

        self.capture_on_stop = False
        self.summary_label.setText("状态：正在断开设备")
        self.device_label.setText("设备：正在断开…")
        self.current_label.setText("当前任务：无")
        self._set_operator_notice("断开：正在停止设备预览。", "saving")
        self.log("正在断开设备。")
        self.update_button_states()
        self.worker_stop_requested.emit()

    def record_event(
        self,
        event_name: str,
        *,
        trial_id: int | None = None,
        class_name: str | None = None,
        run_index: int | None = None,
        run_trial_index: int | None = None,
        block_index: int | None = None,
        prompt_index: int | None = None,
        command_duration_sec: float | None = None,
        execution_success: int | bool | None = None,
        wait_for_marker_flush: bool = True,
    ) -> None:
        if self.marker_failure_active:
            return
        elapsed_sec = 0.0
        if self.session_start_perf > 0:
            elapsed_sec = max(0.0, time.perf_counter() - self.session_start_perf)
        event = make_event(
            event_name,
            trial_id=trial_id,
            class_name=class_name,
            run_index=run_index,
            run_trial_index=run_trial_index,
            block_index=block_index,
            prompt_index=prompt_index,
            command_duration_sec=command_duration_sec,
            execution_success=execution_success,
            elapsed_sec=elapsed_sec,
        )
        if self.worker is None:
            self._abort_session_due_to_marker_failure("采集线程未就绪，无法写入标记。")
            return
        try:
            ok, error_message = self.worker.insert_marker_sync(
                float(event["marker_code"]),
                wait_for_flush=bool(wait_for_marker_flush),
            )
        except TypeError:
            ok, error_message = self.worker.insert_marker_sync(float(event["marker_code"]))
        if not ok:
            self._abort_session_due_to_marker_failure(error_message or "写入标记失败。")
            return
        self.event_log.append(event)

    def _abort_session_due_to_marker_failure(self, message: str) -> None:
        if self.marker_failure_active:
            return
        can_emergency_save = self.worker is not None and self.worker_thread is not None and self.current_settings is not None
        self.marker_failure_active = True
        self.marker_failure_message = str(message)
        self.capture_on_stop = bool(can_emergency_save)
        self.waiting_for_save = bool(can_emergency_save)
        self.emergency_save_only = bool(can_emergency_save)
        self.emergency_save_reason = f"marker_write_failed: {message}"
        self.phase_timer.stop()
        self._stop_voice_prompt()
        self.session_running = False
        self.session_paused = False
        self.phase_prompt_pending = False
        self.pause_started_perf = 0.0
        self.current_phase = "idle"
        self._clear_session_result_tooltips()
        self.phase_label.setText("标记失败")
        self.countdown_label.setText(
            "正在停止设备并保存紧急原始数据…" if can_emergency_save else "本次会话未保存，请检查设备连接和标记写入后重试。"
        )
        self.summary_label.setText(
            "状态：标记写入失败，正在保存紧急原始数据" if can_emergency_save else "状态：标记写入失败，本次会话已中止"
        )
        self.current_label.setText("当前任务：保存紧急原始数据" if can_emergency_save else "当前任务：请检查设备后重新采集")
        self._set_operator_notice(
            "错误：marker 写入失败，正在保存紧急原始数据。" if can_emergency_save else "错误：marker 写入失败，本次会话未保存。",
            "saving" if can_emergency_save else "error",
        )
        self._apply_phase_theme("idle", None)
        self.cue_widget.set_state(
            phase="idle",
            class_name=None,
            title="标记失败",
            subtitle=(
                "正在保存紧急原始数据，请等待完成。"
                if can_emergency_save
                else "本次会话未保存，请检查设备连接和标记写入后重新开始。"
            ),
        )
        self.participant_window.set_prompt(
            phase="idle",
            class_name=None,
            stage_text="标记失败",
            title="实验已中止",
            subtitle=(
                "检测到标记写入失败，正在保存原始数据，请保持静止。"
                if can_emergency_save
                else "检测到标记写入失败，本次会话不会保存，请联系操作员重新开始。"
            ),
            countdown_text="-",
        )
        self.update_button_states()
        if self.worker_thread is not None:
            self.worker_stop_requested.emit()
        if can_emergency_save:
            self.show_error(f"{message}\n\n已停止正式流程，正在保存紧急原始数据。该数据不一定可直接训练，但不会丢失原始矩阵。")
        else:
            self.show_error(f"{message}\n\n本次会话不会保存，请检查设备连接后重新采集。")

    def _abort_session_due_to_runtime_failure(self, message: str, *, source: str) -> None:
        if self.runtime_failure_active or self.waiting_for_save:
            return
        can_emergency_save = self.worker is not None and self.worker_thread is not None and self.current_settings is not None
        self.runtime_failure_active = True
        self.runtime_failure_message = str(message)
        self.capture_on_stop = bool(can_emergency_save)
        self.waiting_for_save = bool(can_emergency_save)
        self.emergency_save_only = bool(can_emergency_save)
        self.emergency_save_reason = f"{source}: {message}"
        self.phase_timer.stop()
        self._stop_voice_prompt()
        self.session_running = False
        self.session_paused = False
        self.phase_prompt_pending = False
        self.pause_started_perf = 0.0
        self.current_phase = "idle"
        self._clear_session_result_tooltips()
        self.phase_label.setText("运行异常")
        self.countdown_label.setText(
            "正在停止设备并保存紧急原始数据…" if can_emergency_save else "当前没有可保存的采样数据，请检查设备后重新开始。"
        )
        self.summary_label.setText(
            "状态：采集流程异常，正在保存紧急原始数据" if can_emergency_save else "状态：采集流程异常，本次会话已中止"
        )
        self.current_label.setText("当前任务：保存紧急原始数据" if can_emergency_save else "当前任务：请检查错误后重新采集")
        self._set_operator_notice(
            "错误：采集流程异常，正在保存紧急原始数据。" if can_emergency_save else "错误：采集流程异常，本次会话未保存。",
            "saving" if can_emergency_save else "error",
        )
        self._apply_phase_theme("idle", None)
        self.cue_widget.set_state(
            phase="idle",
            class_name=None,
            title="采集异常",
            subtitle=(
                "正在保存紧急原始数据，请等待完成。"
                if can_emergency_save
                else "当前没有可保存的采样数据，请检查设备后重新开始。"
            ),
        )
        self.participant_window.set_prompt(
            phase="idle",
            class_name=None,
            stage_text="采集异常",
            title="实验已中止",
            subtitle=(
                "检测到采集流程异常，正在保存原始数据，请保持静止。"
                if can_emergency_save
                else "检测到采集流程异常，请联系操作员重新开始。"
            ),
            countdown_text="-",
        )
        self.update_button_states()
        if self.worker_thread is not None:
            self.worker_stop_requested.emit()
        if can_emergency_save:
            self.show_error(f"{message}\n\n已停止正式流程，正在保存紧急原始数据。该数据不一定可直接训练，但会尽量保住当前原始矩阵。")
        else:
            self.show_error(f"{message}\n\n当前没有可保存的采样数据，请检查设备连接后重新采集。")

    def _set_save_failed_state(self, message: str) -> None:
        self.waiting_for_save = False
        self.capture_on_stop = False
        self.emergency_save_only = False
        self.emergency_save_reason = ""
        self.marker_failure_active = False
        self.marker_failure_message = ""
        self.runtime_failure_active = False
        self.runtime_failure_message = ""
        self._clear_session_result_tooltips()
        self.phase_label.setText("保存失败")
        self.countdown_label.setText("本次会话未保存，请检查原因后重新采集。")
        self.summary_label.setText("状态：保存失败，本次会话未保存")
        self.current_label.setText("当前任务：保存失败，请重新采集")
        self._set_operator_notice("错误：保存失败，请排查后重新采集。", "error")
        self._apply_phase_theme("idle", None)
        self.cue_widget.set_state(
            phase="idle",
            class_name=None,
            title="保存失败",
            subtitle="当前会话未成功写入磁盘，请排查后重新开始。",
        )
        self.trial_banner_label.setText("当前试次：本次会话未保存")
        self.next_task_label.setText("下一任务：检查原因后重新开始")
        if self.use_separate_participant_screen:
            self.restore_operator_window()
        else:
            self.participant_window.hide()
        self.show_error(message)
        self.update_button_states()

    def _handle_missing_save_result(self) -> None:
        if not self.waiting_for_save or self.save_thread is not None:
            return
        self._set_save_failed_state("保存失败：设备线程已结束，但没有返回完整会话数据。本次会话未保存，请重新采集。")

    def on_session_data_ready(self, payload: dict) -> None:
        has_regular_events = bool(self.event_log)
        if not self.capture_on_stop or self.current_settings is None or (not has_regular_events and not self.emergency_save_only):
            self.log("设备停止，但没有需要保存的有效会话。")
            return

        if self.save_thread is not None:
            self._set_save_failed_state("保存失败：检测到重复的保存任务，请重新采集。")
            return

        if self.emergency_save_only:
            self.countdown_label.setText("正在后台写入紧急原始数据，请等待保存完成…")
            self._set_operator_notice("保存：紧急原始数据写盘中，请等待完成。", "saving")
            self.log("设备线程已返回数据，开始保存紧急原始矩阵。")
        else:
            self.countdown_label.setText("正在后台写入采集数据，请等待保存完成…")
            self._set_operator_notice("保存：后台写盘中，请等待完成。", "saving")
            self.log("设备线程已返回完整数据，开始后台写盘。")

        self.save_thread = QThread(self)
        self.save_worker = SessionSaveWorker(
            payload=dict(payload),
            settings=self.current_settings,
            event_log=list(self.event_log),
            trial_records=list(self.trial_records),
            emergency_only=bool(self.emergency_save_only),
            emergency_reason=str(self.emergency_save_reason),
        )
        self.save_worker.moveToThread(self.save_thread)
        self.save_thread.started.connect(self.save_worker.run)
        self.save_worker.save_completed.connect(self.on_session_save_completed)
        self.save_worker.save_failed.connect(self._set_save_failed_state)
        self.save_worker.finished.connect(self.save_thread.quit)
        self.save_worker.finished.connect(self.save_worker.deleteLater)
        self.save_thread.finished.connect(self.on_save_thread_finished)
        self.save_thread.finished.connect(self.save_thread.deleteLater)
        self.save_thread.start()

    def on_session_save_completed(self, result: dict) -> None:
        self.waiting_for_save = False
        self.capture_on_stop = False
        emergency_raw_only = bool(result.get("emergency_raw_only", False))
        self.emergency_save_only = False
        self.emergency_save_reason = ""
        self.marker_failure_active = False
        self.marker_failure_message = ""
        self.runtime_failure_active = False
        self.runtime_failure_message = ""
        trial_count = int(result.get("trial_count", 0))
        if emergency_raw_only:
            self.summary_label.setText(f"状态：已保存紧急原始数据，记录到 {trial_count} 个试次元数据")
        else:
            self.summary_label.setText(f"状态：保存完成，共保存 {trial_count} 个试次")
        saved_session_dir = str(result["session_dir"])
        self.current_label.setText("当前任务：紧急原始数据已保存" if emergency_raw_only else "当前任务：数据已保存")
        self.current_label.setToolTip(saved_session_dir)
        self.phase_label.setText("原始数据已保存" if emergency_raw_only else "保存完成")
        self.countdown_label.setText("当前会话已完成，可重新连接并开始新的采集。")
        self._set_operator_notice(
            "完成：紧急原始数据已保存；请检查 marker/设备后再继续。" if emergency_raw_only else "完成：数据已保存，可以开始下一次采集。",
            "warning" if emergency_raw_only else "ok",
        )
        self._apply_phase_theme("idle", None)
        self.cue_widget.set_state(
            phase="idle",
            class_name=None,
            title="原始数据已保存" if emergency_raw_only else "保存完成",
            subtitle="本次只保存了紧急原始矩阵，请检查设备和 marker。" if emergency_raw_only else "可以开始下一次采集，或关闭程序。",
        )
        self.refresh_trial_counters()
        self.trial_banner_label.setText(f"当前试次：本次会话已完成，共 {trial_count} 个试次")
        self.sequence_summary_label.setText(
            f"计划：本次已保存 {trial_count} 个试次 | "
            f"有效 {int(result.get('accepted_trial_count', 0))} | 坏试次 {int(result.get('rejected_trial_count', 0))}"
        )
        self.next_task_label.setText("下一任务：可以开始新的会话")
        self.next_task_label.setToolTip(saved_session_dir)
        self.log(("紧急原始数据已保存：" if emergency_raw_only else "会话已保存：") + saved_session_dir)
        if emergency_raw_only and result.get("full_save_error"):
            self.log(f"紧急保存原因：{result['full_save_error']}")
        quality_warnings = self._quality_warnings_from_report(result.get("quality_report"))
        if quality_warnings:
            warning_text = "；".join(quality_warnings)
            short_warning = warning_text.split("。", 1)[0] + "。"
            self.summary_label.setText("状态：保存完成，但检测到质量异常")
            self.next_task_label.setText(short_warning)
            self._set_operator_notice(short_warning, "warning")
            self.log(f"质量警告：{warning_text}")
            QMessageBox.warning(self, "保存完成但质量异常", warning_text)
        data_flow_warnings = self._data_flow_warnings_from_report(result.get("data_flow_report"))
        if data_flow_warnings:
            warning_text = "；".join(data_flow_warnings)
            short_warning = warning_text.split("。", 1)[0] + "。"
            if not quality_warnings:
                self.summary_label.setText("状态：保存完成，但检测到数据流异常")
                self.next_task_label.setText(short_warning)
                self._set_operator_notice(short_warning, "warning")
            self.log(f"数据流警告：{warning_text}")
        if "save_index" in result and "run_stem" in result:
            self.log(f"采集编号：save-{int(result['save_index']):03d} | 标识：{result['run_stem']}")
        if result.get("fif_path"):
            self.log(f"原始数据：{result['fif_path']}")
        if result.get("board_data_path"):
            self.log(f"Board matrix: {result['board_data_path']}")
        if result.get("emergency_meta_path"):
            self.log(f"Emergency metadata: {result['emergency_meta_path']}")
        if result.get("segments_csv_path"):
            self.log(f"Segments CSV: {result['segments_csv_path']}")
        if result.get("mi_epochs_path"):
            self.log(f"主分类训练集：{result['mi_epochs_path']}")
        if result.get("gate_epochs_path"):
            self.log(f"门控训练集：{result['gate_epochs_path']}")
        if result.get("artifact_epochs_path"):
            self.log(f"坏窗训练集：{result['artifact_epochs_path']}")
        if result.get("continuous_path"):
            self.log(f"连续验证集：{result['continuous_path']}")
        if result.get("manifest_csv_path"):
            self.log(f"数据索引：{result['manifest_csv_path']}")
        self.session_edit.setText(datetime.now().strftime("%Y%m%d_%H%M%S"))
        if self.use_separate_participant_screen:
            self.restore_operator_window()
        else:
            self.participant_window.hide()
        self.update_button_states()

    def on_save_thread_finished(self) -> None:
        self.save_worker = None
        self.save_thread = None
        self.update_button_states()

    def on_worker_thread_finished(self) -> None:
        marker_failure_active = self.marker_failure_active
        runtime_failure_active = self.runtime_failure_active
        saved_completed = self.current_label.text().startswith("当前任务：数据已保存") or self.current_label.text().startswith(
            "当前任务：紧急原始数据已保存"
        )
        save_failed = self.current_label.text().startswith("当前任务：保存失败")
        pending_save_result = self.waiting_for_save
        self._stop_voice_prompt()
        self.worker = None
        self.worker_thread = None
        self.device_info = None
        self.session_running = False
        self.session_paused = False
        self.phase_prompt_pending = False
        self.current_phase = "idle"
        self.phase_started_perf = 0.0
        self.pause_started_perf = 0.0
        self.phase_timer.stop()
        self.sequence = []
        self.sequence_by_run = []
        self.trials_per_run = 0
        self.current_run_index = 0
        self.current_run_trial_index = 0
        self.calibration_plan = []
        self.calibration_step_index = -1
        self.idle_block_index = 0
        self.continuous_block_index = 0
        self.continuous_schedule_after_runs = []
        self.next_continuous_schedule_index = 0
        self.continuous_prompt_plan = []
        self.continuous_prompt_index = -1
        self.current_continuous_prompt = None
        self.preview_mode = RealtimeEEGPreviewWidget.MODE_EEG
        self.preview_impedance_channel = 1
        self.preview_mode_switch_pending = False
        self.preview_mode_switch_target = None
        self.pending_session_start_context = None
        self.device_label.setText("设备：未连接")
        if self.preview_widget is not None:
            self.preview_widget.clear_stream("设备断开后停止实时波形显示。")
            self.preview_widget.set_quality_mode(self.preview_mode, impedance_channel=self.preview_impedance_channel)
        self._apply_phase_theme("idle", None)
        if marker_failure_active and pending_save_result:
            self.summary_label.setText("状态：marker 写入失败，正在保存紧急原始数据")
            self.current_label.setText("当前任务：等待紧急保存结果")
            self.trial_banner_label.setText("当前试次：marker 写入失败，正在保存原始数据")
            self.next_task_label.setText("下一任务：等待紧急保存完成")
            self._set_operator_notice("保存：marker 写入失败，正在等待紧急原始数据保存结果。", "saving")
        elif marker_failure_active:
            self.summary_label.setText("状态：标记写入失败，本次会话已丢弃")
            self.current_label.setText("当前任务：检查设备后重新开始")
            self.trial_banner_label.setText("当前试次：本次会话因标记写入失败未保存")
            self.next_task_label.setText("下一任务：排查设备连接或 marker 写入")
            self._set_operator_notice("错误：marker 写入失败，本次会话已丢弃。", "error")
        elif runtime_failure_active and pending_save_result:
            self.summary_label.setText("状态：采集异常，正在保存紧急原始数据")
            self.current_label.setText("当前任务：等待紧急保存结果")
            self.trial_banner_label.setText("当前试次：采集异常，正在保存原始数据")
            self.next_task_label.setText("下一任务：等待紧急保存完成")
            self._set_operator_notice("保存：采集异常，正在等待紧急原始数据保存结果。", "saving")
        elif runtime_failure_active:
            self.summary_label.setText("状态：采集异常，但没有返回可保存数据")
            self.current_label.setText("当前任务：检查错误后重新开始")
            self.trial_banner_label.setText("当前试次：本次会话因采集异常未保存")
            self.next_task_label.setText("下一任务：排查设备连接或流程错误")
            self._set_operator_notice("错误：采集异常，未收到可保存数据。", "error")
        elif pending_save_result:
            self.current_label.setText("当前任务：等待保存数据")
            self.trial_banner_label.setText("当前试次：设备线程已结束，正在等待保存结果")
            self.next_task_label.setText("下一任务：等待保存完成")
            self._set_operator_notice("保存：设备线程已结束，正在等待保存结果。", "saving")
        elif not saved_completed and not save_failed:
            self.current_label.setText("当前任务：无")
            self.trial_banner_label.setText("当前试次：未开始")
            self.next_task_label.setText("下一任务：--")
            self._set_operator_notice("待机：连接设备后先完成 EEG/阻抗质量检查。", "idle")
        if not self.waiting_for_save and not saved_completed and not save_failed and not marker_failure_active:
            self.summary_label.setText("状态：设备已断开")
        if not pending_save_result:
            self.marker_failure_active = False
            self.marker_failure_message = ""
            self.runtime_failure_active = False
            self.runtime_failure_message = ""
        self._set_preview_mode_label()
        self._update_preview_status()
        self.update_button_states()
        if pending_save_result and self.save_thread is None:
            QTimer.singleShot(MISSING_SAVE_RESULT_GRACE_MS, self._handle_missing_save_result)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._save_in_progress():
            QMessageBox.information(self, "请稍候", "当前正在保存数据，请等待保存完成后再关闭窗口。")
            event.ignore()
            return

        if self.session_running:
            QMessageBox.information(self, "请先停止并保存", "当前会话正在运行，请先点击“停止并保存”。")
            event.ignore()
            return

        if self.worker_thread is None:
            self._stop_voice_prompt()
            self.participant_window.hide()
            event.accept()
            return

        reply = QMessageBox.question(
            self,
            "退出确认",
            "设备仍处于连接状态。关闭窗口前将先断开设备，是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            event.ignore()
            return

        self.capture_on_stop = False
        self._stop_voice_prompt()
        self.participant_window.hide()
        self.worker_stop_requested.emit()
        if self.worker_thread is not None:
            self.worker_thread.wait(5000)
        event.accept()


def build_initial_config_from_args(args: argparse.Namespace) -> dict:
    """Translate CLI options into initial window values."""
    config = {}
    if args.synthetic:
        synthetic = getattr(BoardIds, "SYNTHETIC_BOARD", None)
        if synthetic is not None:
            config["board_id"] = int(synthetic.value)
            config["serial_port"] = ""
    if args.serial_port:
        config["serial_port"] = args.serial_port
    if args.output_root:
        config["output_root"] = str(resolve_output_root(args.output_root))
    if args.subject_id:
        config["subject_id"] = args.subject_id
    if args.session_id:
        config["session_id"] = args.session_id
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="运动想象脑电数据采集器（仅采集）")
    parser.add_argument("--serial-port", type=str, default="", help="覆盖默认串口，例如 COM3")
    parser.add_argument("--output-root", type=str, default="", help="覆盖默认输出目录")
    parser.add_argument("--subject-id", type=str, default="", help="预填被试编号")
    parser.add_argument("--session-id", type=str, default="", help="预填会话编号")
    parser.add_argument("--synthetic", action="store_true", help="使用 BrainFlow 模拟板卡演示界面")
    args = parser.parse_args()

    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName("运动想象数据采集器")
    window = MIDataCollectorWindow(initial_config=build_initial_config_from_args(args))

    previous_excepthook = sys.excepthook

    def emergency_excepthook(exc_type, exc_value, exc_traceback) -> None:
        if exc_type is not KeyboardInterrupt:
            try:
                if window.session_running and not window._save_in_progress():
                    window._abort_session_due_to_runtime_failure(
                        f"未处理异常：{exc_value}",
                        source="unhandled_exception",
                    )
            except Exception:
                pass
        previous_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = emergency_excepthook
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
