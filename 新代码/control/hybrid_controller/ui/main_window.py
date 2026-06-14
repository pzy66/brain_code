from __future__ import annotations

import math
import time
from pathlib import Path

try:
    from PyQt5.QtCore import QEvent, QPointF, Qt, pyqtSignal
    from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPalette
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as error:  # pragma: no cover - UI import guard
    raise RuntimeError("PyQt5 is required to use hybrid_controller.ui.main_window") from error

from hybrid_controller.snapshot import AppSnapshot
from hybrid_controller.ui.vision_feed_widget import VisionFeedWidget

AUTO_PROFILE_VALUE = "__AUTO_PROFILE__"
_UNCHANGED = object()
DEFAULT_SSVEP_PRETRAIN_PRESET = "fast"
SSVEP_PRETRAIN_PRESETS: dict[str, dict[str, object]] = {
    "fast": {
        "label": "Fast",
        "prepare_sec": 0.5,
        "active_sec": 3.5,
        "rest_sec": 0.5,
        "target_repeats": 2,
        "idle_repeats": 4,
        "win_sec": 2.5,
        "step_sec": 0.5,
    },
    "standard": {
        "label": "Standard",
        "prepare_sec": 0.8,
        "active_sec": 4.0,
        "rest_sec": 0.8,
        "target_repeats": 3,
        "idle_repeats": 6,
        "win_sec": 3.0,
        "step_sec": 0.5,
    },
    "stable": {
        "label": "Stable",
        "prepare_sec": 1.0,
        "active_sec": 4.0,
        "rest_sec": 1.0,
        "target_repeats": 5,
        "idle_repeats": 10,
        "win_sec": 3.0,
        "step_sec": 0.5,
    },
}


def _ssvep_pretrain_estimate_seconds(preset: dict[str, object]) -> float:
    target_repeats = int(preset["target_repeats"])
    idle_repeats = int(preset["idle_repeats"])
    trial_count = 4 * target_repeats + idle_repeats
    trial_sec = float(preset["prepare_sec"]) + float(preset["active_sec"]) + float(preset["rest_sec"])
    return float(trial_count) * trial_sec


class ControlSceneWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._snapshot: dict[str, object] | None = None
        self.setMinimumSize(240, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def update_scene(self, snapshot: dict[str, object] | None) -> None:
        self._snapshot = snapshot
        self.update()

    @staticmethod
    def _world_from_cyl(theta_deg: float, radius_mm: float) -> tuple[float, float]:
        theta_rad = math.radians(float(theta_deg))
        x_mm = float(radius_mm) * math.sin(theta_rad)
        y_mm = -float(radius_mm) * math.cos(theta_rad)
        return (x_mm, y_mm)

    def _iter_world_points(self) -> list[tuple[float, float]]:
        if not self._snapshot:
            return []
        points: list[tuple[float, float]] = [(0.0, 0.0)]
        home_pose = self._snapshot.get("home_pose")
        if isinstance(home_pose, (list, tuple)) and len(home_pose) >= 2:
            points.append((float(home_pose[0]), float(home_pose[1])))
        robot_xy = self._snapshot.get("robot_xy")
        if isinstance(robot_xy, (list, tuple)) and len(robot_xy) >= 2:
            points.append((float(robot_xy[0]), float(robot_xy[1])))

        limits_cyl = self._snapshot.get("limits_cyl")
        limits_cyl_auto = self._snapshot.get("limits_cyl_auto")
        for limits in (limits_cyl, limits_cyl_auto):
            if not isinstance(limits, dict):
                continue
            theta_limits = tuple(limits.get("theta_deg", (-120.0, 120.0)))
            radius_limits = tuple(limits.get("radius_mm", (50.0, 230.0)))
            for theta_deg in (float(theta_limits[0]), float(theta_limits[1])):
                for radius_mm in (float(radius_limits[0]), float(radius_limits[1])):
                    points.append(self._world_from_cyl(theta_deg, radius_mm))
        return points

    def _build_map_point(self, margin: int, width: int, height: int):
        points = self._iter_world_points()
        if not points:
            def fallback(world_xy: tuple[float, float]) -> tuple[float, float]:
                return (margin + width / 2.0, margin + height / 2.0)
            return fallback

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        span_x = max(1.0, max_x - min_x)
        span_y = max(1.0, max_y - min_y)
        scale = min(width / span_x, height / span_y)
        draw_w = span_x * scale
        draw_h = span_y * scale
        offset_x = margin + (width - draw_w) / 2.0
        offset_y = margin + (height - draw_h) / 2.0

        def map_point(world_xy: tuple[float, float]) -> tuple[float, float]:
            x_mm = float(world_xy[0])
            y_mm = float(world_xy[1])
            # Mirror X for operator-facing view so left/right matches user perspective.
            px = offset_x + (max_x - x_mm) * scale
            # Render in operator-facing front view (vertical flipped against robot-world Y).
            py = offset_y + (y_mm - min_y) * scale
            return (px, py)

        return map_point

    def _build_annular_sector_path(
        self,
        map_point,
        theta_limits: tuple[float, float],
        radius_limits: tuple[float, float],
        *,
        steps: int = 72,
    ) -> QPainterPath:
        theta_min = float(theta_limits[0])
        theta_max = float(theta_limits[1])
        radius_min = max(0.0, float(radius_limits[0]))
        radius_max = max(radius_min, float(radius_limits[1]))
        step_count = max(8, int(steps))

        outer_points: list[QPointF] = []
        inner_points: list[QPointF] = []
        for index in range(step_count + 1):
            ratio = index / step_count
            theta_deg = theta_min + (theta_max - theta_min) * ratio
            outer_xy = self._world_from_cyl(theta_deg, radius_max)
            inner_xy = self._world_from_cyl(theta_deg, radius_min)
            outer_px = map_point(outer_xy)
            inner_px = map_point(inner_xy)
            outer_points.append(QPointF(float(outer_px[0]), float(outer_px[1])))
            inner_points.append(QPointF(float(inner_px[0]), float(inner_px[1])))

        path = QPainterPath()
        if not outer_points:
            return path
        path.moveTo(outer_points[0])
        for point in outer_points[1:]:
            path.lineTo(point)
        for point in reversed(inner_points):
            path.lineTo(point)
        path.closeSubpath()
        return path

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.fillRect(self.rect(), QColor(20, 24, 30))
        if not self._snapshot:
            qp.setPen(QColor(220, 220, 220))
            qp.drawText(self.rect(), Qt.AlignCenter, "Pose map unavailable")
            return

        margin = 18
        width = max(10, self.width() - margin * 2)
        height = max(10, self.height() - margin * 2)
        map_point = self._build_map_point(margin, width, height)
        limits_cyl = self._snapshot.get("limits_cyl") or {}
        limits_cyl_auto = self._snapshot.get("limits_cyl_auto") or {}
        theta_limits = tuple(limits_cyl.get("theta_deg", (-120.0, 120.0)))
        radius_limits = tuple(limits_cyl.get("radius_mm", (50.0, 280.0)))
        auto_theta_limits = tuple(limits_cyl_auto.get("theta_deg", theta_limits))
        auto_radius_limits = tuple(limits_cyl_auto.get("radius_mm", radius_limits))

        full_path = self._build_annular_sector_path(map_point, theta_limits, radius_limits)
        qp.setPen(QPen(QColor(85, 150, 225, 220), 2))
        qp.setBrush(QBrush(QColor(65, 120, 190, 35)))
        qp.drawPath(full_path)

        auto_path = self._build_annular_sector_path(map_point, auto_theta_limits, auto_radius_limits)
        qp.setPen(QPen(QColor(100, 220, 205, 220), 2))
        qp.setBrush(QBrush(QColor(80, 220, 180, 55)))
        qp.drawPath(auto_path)

        origin_x, origin_y = map_point((0.0, 0.0))
        qp.setPen(QPen(QColor(180, 200, 220), 2))
        qp.setBrush(QBrush(QColor(180, 200, 220, 140)))
        qp.drawEllipse(int(origin_x) - 4, int(origin_y) - 4, 8, 8)

        home_pose = self._snapshot.get("home_pose") or (0.0, -120.0, 160.0)
        hx, hy = map_point((float(home_pose[0]), float(home_pose[1])))
        qp.setPen(QPen(QColor(80, 220, 180), 2))
        qp.setBrush(QBrush(QColor(80, 220, 180, 90)))
        qp.drawRect(int(hx) - 6, int(hy) - 6, 12, 12)
        qp.drawText(int(hx) + 6, int(hy) - 6, "HOME")

        robot_xy = self._snapshot.get("robot_xy") or (0.0, 0.0)
        robot_x, robot_y = map_point((float(robot_xy[0]), float(robot_xy[1])))
        qp.setPen(QPen(QColor(255, 90, 90), 2))
        qp.setBrush(QBrush(QColor(255, 90, 90)))
        qp.drawEllipse(int(robot_x) - 9, int(robot_y) - 9, 18, 18)

        qp.setPen(QColor(230, 230, 230))
        qp.setFont(QFont("Consolas", 9))
        cyl = self._snapshot.get("robot_cyl") or {}
        qp.drawText(
            12,
            self.height() - 12,
            "theta={:.1f} r={:.1f} z={:.1f}".format(
                float(cyl.get("theta_deg", 0.0)),
                float(cyl.get("radius_mm", 0.0)),
                float(cyl.get("z_mm", 0.0)),
            ),
        )


class MainWindow(QMainWindow):
    key_pressed = pyqtSignal(str)
    key_released = pyqtSignal(str)
    robot_start_requested = pyqtSignal()
    robot_connect_requested = pyqtSignal()
    abort_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    sucker_off_requested = pyqtSignal()
    ssvep_connect_requested = pyqtSignal()
    ssvep_config_apply_requested = pyqtSignal(str, int)
    ssvep_pretrain_requested = pyqtSignal()
    ssvep_load_profile_requested = pyqtSignal()
    ssvep_open_profile_dir_requested = pyqtSignal()
    ssvep_stim_toggled = pyqtSignal(bool)
    ssvep_start_requested = pyqtSignal()
    ssvep_stop_requested = pyqtSignal()
    manual_pick_slot_requested = pyqtSignal(int)
    manual_place_requested = pyqtSignal()
    pick_radius_bias_delta_requested = pyqtSignal(float)
    pick_bias_reset_requested = pyqtSignal()
    pick_tangent_bias_delta_requested = pyqtSignal(float)
    pick_tangent_bias_reset_requested = pyqtSignal()
    pick_theta_bias_delta_requested = pyqtSignal(float)
    pick_theta_bias_reset_requested = pyqtSignal()
    pick_tuning_delta_requested = pyqtSignal(str, float)
    pick_release_mode_toggle_requested = pyqtSignal()
    pick_tuning_apply_requested = pyqtSignal()
    pick_tuning_reset_requested = pyqtSignal()
    pick_tuning_save_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hybrid Controller v1")
        self.resize(1360, 860)
        self.setFocusPolicy(Qt.StrongFocus)
        self._apply_ui_theme()

        root = QWidget(self)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        self.top_status_label = QLabel("State: idle | Sources: --")
        self.top_status_label.setObjectName("topStatus")
        self.top_status_label.setWordWrap(True)
        self.top_status_label.setMinimumWidth(0)
        self.top_status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        main_layout.addWidget(self.top_status_label)

        self._workflow_steps: list[tuple[str, str, str]] = [
            ("Step 1 · 接近目标区", "W/A/S/D 或 ⬅/➡/⬆/⬇", "先移动机械臂到可见目标区域。"),
            ("Step 2 · 进入选择", "N 或 Enter", "进入目标槽位选择状态。"),
            ("Step 3 · 选择槽位", "1 / 2 / 3 / 4", "用数字键直接选中目标槽位。"),
            ("Step 4 · 抓取确认", "Enter 或 C", "确认抓取，执行吸附动作。"),
            ("Step 5 · 搬运到放置位", "W/A/S/D 或 ⬅/➡/⬆/⬇", "保持抓取后移动到放置目标。"),
            ("Step 6 · 放置确认", "Enter 或 C", "对准放置位后确认释放。"),
        ]
        self._workflow_steps_total = len(self._workflow_steps)

        workflow_card = QFrame()
        workflow_card.setObjectName("flowCard")
        workflow_layout = QVBoxLayout(workflow_card)
        workflow_layout.setSpacing(8)
        workflow_layout.setContentsMargins(10, 10, 10, 10)

        flow_header = QHBoxLayout()
        self.flow_title_label = QLabel("全流程控制总览")
        self.flow_title_label.setObjectName("flowTitle")
        flow_header.addWidget(self.flow_title_label, stretch=1)

        self.workflow_progress_label = QLabel("进度 0/6")
        self.workflow_progress_label.setObjectName("workflowProgressLabel")
        flow_header.addWidget(self.workflow_progress_label)
        workflow_layout.addLayout(flow_header)

        self.flow_subtitle_label = QLabel("按流程推进控制：每一阶段状态会自动联动。")
        self.flow_subtitle_label.setObjectName("flowSubtitle")
        self.flow_subtitle_label.setWordWrap(True)
        workflow_layout.addWidget(self.flow_subtitle_label)

        self.workflow_progress = QProgressBar()
        self.workflow_progress.setObjectName("workflowProgressBar")
        self.workflow_progress.setRange(0, self._workflow_steps_total)
        self.workflow_progress.setTextVisible(False)
        workflow_layout.addWidget(self.workflow_progress)

        self.workflow_status_label = QLabel("系统待机")
        self.workflow_status_label.setObjectName("workflowStatusLabel")
        self.workflow_status_label.setWordWrap(True)
        workflow_layout.addWidget(self.workflow_status_label)

        self.workflow_steps_layout = QVBoxLayout()
        self.workflow_steps_layout.setSpacing(6)
        self._workflow_step_rows: list[tuple[QFrame, QLabel, QLabel, QLabel]] = []
        for step_idx, (step_text, shortcut_text, description) in enumerate(self._workflow_steps, start=1):
            step_row = QFrame()
            step_row.setObjectName("workflowStep")
            row_layout = QVBoxLayout(step_row)
            row_layout.setContentsMargins(8, 6, 8, 6)
            row_layout.setSpacing(2)

            title_row = QHBoxLayout()
            title_row.setSpacing(8)
            index_label = QLabel(str(step_idx))
            index_label.setObjectName("workflowStepIndex")
            title_label = QLabel(step_text)
            title_label.setObjectName("workflowStepTitle")
            title_label.setWordWrap(True)
            title_row.addWidget(index_label)
            title_row.addWidget(title_label, stretch=1)
            status_tag = QLabel("待开始")
            status_tag.setObjectName("workflowStepState")
            status_tag.setAlignment(Qt.AlignRight)
            title_row.addWidget(status_tag)

            detail_label = QLabel(f"{shortcut_text} | {description}")
            detail_label.setObjectName("workflowStepDetail")
            detail_label.setWordWrap(True)
            row_layout.addLayout(title_row)
            row_layout.addWidget(detail_label)
            self._workflow_step_rows.append((step_row, detail_label, status_tag, title_label))
            self.workflow_steps_layout.addWidget(step_row)

        workflow_layout.addLayout(self.workflow_steps_layout)

        self.quick_guide_label = QLabel(
            "当前控制策略：键盘替代 MI 控制；1~4 替代 SSVEP 目标选择。"
        )
        self.quick_guide_label.setWordWrap(True)
        workflow_layout.addWidget(self.quick_guide_label)
        self.quick_guide_label.setObjectName("quickGuideLabel")

        control_keys_card = QFrame()
        control_keys_card.setObjectName("shortcutCard")
        control_keys_layout = QVBoxLayout(control_keys_card)
        control_keys_layout.setSpacing(6)
        control_keys_layout.setContentsMargins(8, 8, 8, 8)
        control_keys_title = QLabel("手动控制映射（演示）")
        control_keys_title.setObjectName("shortcutTitle")
        control_keys_layout.addWidget(control_keys_title)
        self._append_shortcut_row(
            control_keys_layout,
            [("W", "move"), ("A", "move"), ("S", "move"), ("D", "move"), ("↑", "move"), ("↓", "move"), ("←", "move"), ("→", "move")],
            "移动：按住保持微调前后左右",
            chip_width=28,
        )
        self._append_shortcut_row(
            control_keys_layout,
            [("N", "logic"), ("R", "logic")],
            "N=开始选择  R=复位",
            chip_width=30,
        )
        self._append_shortcut_row(
            control_keys_layout,
            [("1", "target"), ("2", "target"), ("3", "target"), ("4", "target")],
            "1~4=目标槽位选择（代替 SSVEP）",
            chip_width=26,
        )
        self._append_shortcut_row(
            control_keys_layout,
            [("Enter", "action"), ("C", "action"), ("Esc", "danger"), ("X", "danger")],
            "Enter/C=确认  Esc/X=取消",
            chip_width=46,
        )
        workflow_layout.addWidget(control_keys_card)
        main_layout.addWidget(workflow_card)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)
        main_layout.addLayout(content_layout, stretch=1)

        self.vision_widget = VisionFeedWidget(refresh_rate_hz=240.0)
        self._vision_frame_cache = None
        self._vision_packet_cache: dict[str, object] | None = None
        self._vision_flash_cache = False
        self._vision_status_cache = "Waiting for vision runtime..."
        self._vision_last_frame_obj_id: int | None = None
        self._vision_last_packet_frame_id: int | None = None
        self._vision_last_flash: bool | None = None
        self._vision_last_status: str | None = None
        content_layout.addWidget(self.vision_widget, stretch=5)

        # Floating robot pose card anchored to top-right of the camera panel.
        self.pose_overlay = QFrame(self.vision_widget)
        self.pose_overlay.setObjectName("poseOverlay")
        pose_overlay_layout = QVBoxLayout(self.pose_overlay)
        pose_overlay_layout.setContentsMargins(8, 8, 8, 8)
        pose_overlay_layout.setSpacing(6)
        self.pose_title_label = QLabel("Robot Pose")
        self.pose_title_label.setObjectName("poseTitle")
        pose_overlay_layout.addWidget(self.pose_title_label)
        self.scene_widget = ControlSceneWidget(self.pose_overlay)
        pose_overlay_layout.addWidget(self.scene_widget, stretch=1)
        self.pose_overlay.show()
        self.pose_overlay.raise_()

        right_panel = QFrame()
        right_panel.setObjectName("sidePanel")
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_panel.setMinimumWidth(320)
        right_panel.setMaximumWidth(440)
        right_shell_layout = QVBoxLayout(right_panel)
        right_shell_layout.setContentsMargins(0, 0, 0, 0)
        right_shell_layout.setSpacing(0)
        right_scroll = QScrollArea(right_panel)
        right_scroll.setObjectName("sideScroll")
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_content = QWidget()
        right_content.setObjectName("sideContent")
        right_content.setMinimumWidth(0)
        right_content.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        right_layout = QVBoxLayout(right_content)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)

        controls_row = QGridLayout()
        controls_row.setHorizontalSpacing(6)
        controls_row.setVerticalSpacing(6)
        self.robot_start_button = QPushButton("启动机器人")
        self.robot_start_button.setProperty("controlType", "primary")
        self.robot_connect_button = QPushButton("连接机器人")
        self.robot_connect_button.setProperty("controlType", "primary")
        self.abort_button = QPushButton("中止")
        self.abort_button.setProperty("controlType", "danger")
        self.reset_button = QPushButton("复位")
        self.reset_button.setProperty("controlType", "warning")
        self.sucker_off_button = QPushButton("吸嘴关断")
        self.sucker_off_button.setProperty("controlType", "neutral")
        controls_row.addWidget(self.robot_start_button, 0, 0)
        controls_row.addWidget(self.robot_connect_button, 0, 1)
        controls_row.addWidget(self.abort_button, 1, 0)
        controls_row.addWidget(self.reset_button, 1, 1)
        controls_row.addWidget(self.sucker_off_button, 2, 0, 1, 2)
        controls_row.setColumnStretch(0, 1)
        controls_row.setColumnStretch(1, 1)
        right_layout.addLayout(controls_row)

        self.robot_start_button.clicked.connect(self.robot_start_requested.emit)
        self.robot_connect_button.clicked.connect(self.robot_connect_requested.emit)
        self.abort_button.clicked.connect(self.abort_requested.emit)
        self.reset_button.clicked.connect(self.reset_requested.emit)
        self.sucker_off_button.clicked.connect(self.sucker_off_requested.emit)

        pick_title = QLabel("抓取放置调试")
        pick_title.setObjectName("panelSectionTitle")
        right_layout.addWidget(pick_title)

        pick_row = QGridLayout()
        pick_row.setHorizontalSpacing(6)
        pick_row.setVerticalSpacing(6)
        self.pick_slot1_button = QPushButton("选槽 1")
        self.pick_slot2_button = QPushButton("选槽 2")
        self.pick_slot3_button = QPushButton("选槽 3")
        self.pick_slot4_button = QPushButton("选槽 4")
        pick_row.addWidget(self.pick_slot1_button, 0, 0)
        pick_row.addWidget(self.pick_slot2_button, 0, 1)
        pick_row.addWidget(self.pick_slot3_button, 1, 0)
        pick_row.addWidget(self.pick_slot4_button, 1, 1)
        pick_row.setColumnStretch(0, 1)
        pick_row.setColumnStretch(1, 1)
        right_layout.addLayout(pick_row)

        pick_row2 = QHBoxLayout()
        self.place_now_button = QPushButton("立即放置")
        pick_row2.addWidget(self.place_now_button)
        right_layout.addLayout(pick_row2)

        pick_bias_row = QHBoxLayout()
        self.pick_r_minus_1_button = QPushButton("R-")
        self.pick_r_plus_1_button = QPushButton("R+")
        self.pick_r_reset_button = QPushButton("R重置")
        pick_bias_row.addWidget(self.pick_r_minus_1_button)
        pick_bias_row.addWidget(self.pick_r_plus_1_button)
        pick_bias_row.addWidget(self.pick_r_reset_button)
        right_layout.addLayout(pick_bias_row)

        pick_tangent_bias_row = QHBoxLayout()
        self.pick_tangent_minus_1_button = QPushButton("切向-")
        self.pick_tangent_plus_1_button = QPushButton("切向+")
        self.pick_tangent_reset_button = QPushButton("切向归零")
        pick_tangent_bias_row.addWidget(self.pick_tangent_minus_1_button)
        pick_tangent_bias_row.addWidget(self.pick_tangent_plus_1_button)
        pick_tangent_bias_row.addWidget(self.pick_tangent_reset_button)
        right_layout.addLayout(pick_tangent_bias_row)

        pick_theta_bias_row = QHBoxLayout()
        self.pick_theta_minus_1_button = QPushButton("角度-")
        self.pick_theta_plus_1_button = QPushButton("角度+")
        self.pick_theta_reset_button = QPushButton("角度归零")
        pick_theta_bias_row.addWidget(self.pick_theta_minus_1_button)
        pick_theta_bias_row.addWidget(self.pick_theta_plus_1_button)
        pick_theta_bias_row.addWidget(self.pick_theta_reset_button)
        right_layout.addLayout(pick_theta_bias_row)

        self.pick_r_bias_label = QLabel("半径偏差: +0.0 mm")
        self.pick_r_bias_label.setObjectName("rightInfoLabel")
        right_layout.addWidget(self.pick_r_bias_label)
        self.pick_tangent_bias_label = QLabel("切向偏差: +0.0 mm")
        self.pick_tangent_bias_label.setObjectName("rightInfoLabel")
        right_layout.addWidget(self.pick_tangent_bias_label)
        self.pick_theta_bias_label = QLabel("角度偏差: +0.0 deg")
        self.pick_theta_bias_label.setObjectName("rightInfoLabel")
        right_layout.addWidget(self.pick_theta_bias_label)

        pick_tuning_title = QLabel("参数微调")
        pick_tuning_title.setObjectName("panelSectionTitle")
        right_layout.addWidget(pick_tuning_title)

        self.pick_tuning_label = QLabel("approach=130.0 descend=85.0 pre=0.25 hold=0.15 lift=0.80\nplace_z=85.0 release=release rel=0.25 post=0.10 floor=160.0")
        self.pick_tuning_label.setWordWrap(True)
        self.pick_tuning_label.setObjectName("rightInfoLabel")
        right_layout.addWidget(self.pick_tuning_label)

        pick_tuning_buttons = QGridLayout()
        pick_tuning_buttons.setHorizontalSpacing(6)
        pick_tuning_buttons.setVerticalSpacing(6)
        self.pick_tune_approach_minus_button = QPushButton("A-1")
        self.pick_tune_approach_plus_button = QPushButton("A+1")
        self.pick_tune_descend_minus_button = QPushButton("D-1")
        self.pick_tune_descend_plus_button = QPushButton("D+1")
        self.pick_tune_place_minus_button = QPushButton("P-1")
        self.pick_tune_place_plus_button = QPushButton("P+1")
        self.pick_tune_pre_minus_button = QPushButton("pre-0.05")
        self.pick_tune_pre_plus_button = QPushButton("pre+0.05")
        self.pick_tune_hold_minus_button = QPushButton("hold-0.05")
        self.pick_tune_hold_plus_button = QPushButton("hold+0.05")
        self.pick_tune_lift_minus_button = QPushButton("lift-0.05")
        self.pick_tune_lift_plus_button = QPushButton("lift+0.05")
        self.pick_tune_release_minus_button = QPushButton("rel-0.05")
        self.pick_tune_release_plus_button = QPushButton("rel+0.05")
        self.pick_tune_post_minus_button = QPushButton("post-0.05")
        self.pick_tune_post_plus_button = QPushButton("post+0.05")
        self.pick_tune_floor_minus_button = QPushButton("floor-1")
        self.pick_tune_floor_plus_button = QPushButton("floor+1")
        self.pick_tune_mode_button = QPushButton("mode: release")

        pick_tuning_buttons.addWidget(self.pick_tune_approach_minus_button, 0, 0)
        pick_tuning_buttons.addWidget(self.pick_tune_approach_plus_button, 0, 1)
        pick_tuning_buttons.addWidget(self.pick_tune_descend_minus_button, 1, 0)
        pick_tuning_buttons.addWidget(self.pick_tune_descend_plus_button, 1, 1)
        pick_tuning_buttons.addWidget(self.pick_tune_place_minus_button, 2, 0)
        pick_tuning_buttons.addWidget(self.pick_tune_place_plus_button, 2, 1)
        pick_tuning_buttons.addWidget(self.pick_tune_pre_minus_button, 3, 0)
        pick_tuning_buttons.addWidget(self.pick_tune_pre_plus_button, 3, 1)
        pick_tuning_buttons.addWidget(self.pick_tune_hold_minus_button, 4, 0)
        pick_tuning_buttons.addWidget(self.pick_tune_hold_plus_button, 4, 1)
        pick_tuning_buttons.addWidget(self.pick_tune_lift_minus_button, 5, 0)
        pick_tuning_buttons.addWidget(self.pick_tune_lift_plus_button, 5, 1)
        pick_tuning_buttons.addWidget(self.pick_tune_release_minus_button, 6, 0)
        pick_tuning_buttons.addWidget(self.pick_tune_release_plus_button, 6, 1)
        pick_tuning_buttons.addWidget(self.pick_tune_post_minus_button, 7, 0)
        pick_tuning_buttons.addWidget(self.pick_tune_post_plus_button, 7, 1)
        pick_tuning_buttons.addWidget(self.pick_tune_floor_minus_button, 8, 0)
        pick_tuning_buttons.addWidget(self.pick_tune_floor_plus_button, 8, 1)
        pick_tuning_buttons.addWidget(self.pick_tune_mode_button, 9, 0, 1, 2)
        pick_tuning_buttons.setColumnStretch(0, 1)
        pick_tuning_buttons.setColumnStretch(1, 1)
        right_layout.addLayout(pick_tuning_buttons)

        pick_tuning_action_row = QHBoxLayout()
        self.pick_tune_apply_button = QPushButton("应用参数")
        self.pick_tune_reset_button = QPushButton("重置参数")
        self.pick_tune_save_button = QPushButton("保存参数")
        pick_tuning_action_row.addWidget(self.pick_tune_apply_button)
        pick_tuning_action_row.addWidget(self.pick_tune_reset_button)
        pick_tuning_action_row.addWidget(self.pick_tune_save_button)
        right_layout.addLayout(pick_tuning_action_row)

        self.pick_slot1_button.clicked.connect(lambda: self.manual_pick_slot_requested.emit(1))
        self.pick_slot2_button.clicked.connect(lambda: self.manual_pick_slot_requested.emit(2))
        self.pick_slot3_button.clicked.connect(lambda: self.manual_pick_slot_requested.emit(3))
        self.pick_slot4_button.clicked.connect(lambda: self.manual_pick_slot_requested.emit(4))
        self.place_now_button.clicked.connect(self.manual_place_requested.emit)
        self.pick_r_minus_1_button.clicked.connect(lambda: self.pick_radius_bias_delta_requested.emit(-1.0))
        self.pick_r_plus_1_button.clicked.connect(lambda: self.pick_radius_bias_delta_requested.emit(1.0))
        self.pick_r_reset_button.clicked.connect(self.pick_bias_reset_requested.emit)
        self.pick_tangent_minus_1_button.clicked.connect(lambda: self.pick_tangent_bias_delta_requested.emit(-1.0))
        self.pick_tangent_plus_1_button.clicked.connect(lambda: self.pick_tangent_bias_delta_requested.emit(1.0))
        self.pick_tangent_reset_button.clicked.connect(self.pick_tangent_bias_reset_requested.emit)
        self.pick_theta_minus_1_button.clicked.connect(lambda: self.pick_theta_bias_delta_requested.emit(-1.0))
        self.pick_theta_plus_1_button.clicked.connect(lambda: self.pick_theta_bias_delta_requested.emit(1.0))
        self.pick_theta_reset_button.clicked.connect(self.pick_theta_bias_reset_requested.emit)
        self.pick_tune_approach_minus_button.clicked.connect(lambda: self.pick_tuning_delta_requested.emit("pick_approach_z_mm", -1.0))
        self.pick_tune_approach_plus_button.clicked.connect(lambda: self.pick_tuning_delta_requested.emit("pick_approach_z_mm", 1.0))
        self.pick_tune_descend_minus_button.clicked.connect(lambda: self.pick_tuning_delta_requested.emit("pick_descend_z_mm", -1.0))
        self.pick_tune_descend_plus_button.clicked.connect(lambda: self.pick_tuning_delta_requested.emit("pick_descend_z_mm", 1.0))
        self.pick_tune_place_minus_button.clicked.connect(lambda: self.pick_tuning_delta_requested.emit("place_descend_z_mm", -1.0))
        self.pick_tune_place_plus_button.clicked.connect(lambda: self.pick_tuning_delta_requested.emit("place_descend_z_mm", 1.0))
        self.pick_tune_pre_minus_button.clicked.connect(lambda: self.pick_tuning_delta_requested.emit("pick_pre_suction_sec", -0.05))
        self.pick_tune_pre_plus_button.clicked.connect(lambda: self.pick_tuning_delta_requested.emit("pick_pre_suction_sec", 0.05))
        self.pick_tune_hold_minus_button.clicked.connect(lambda: self.pick_tuning_delta_requested.emit("pick_bottom_hold_sec", -0.05))
        self.pick_tune_hold_plus_button.clicked.connect(lambda: self.pick_tuning_delta_requested.emit("pick_bottom_hold_sec", 0.05))
        self.pick_tune_lift_minus_button.clicked.connect(lambda: self.pick_tuning_delta_requested.emit("pick_lift_sec", -0.05))
        self.pick_tune_lift_plus_button.clicked.connect(lambda: self.pick_tuning_delta_requested.emit("pick_lift_sec", 0.05))
        self.pick_tune_release_minus_button.clicked.connect(lambda: self.pick_tuning_delta_requested.emit("place_release_sec", -0.05))
        self.pick_tune_release_plus_button.clicked.connect(lambda: self.pick_tuning_delta_requested.emit("place_release_sec", 0.05))
        self.pick_tune_post_minus_button.clicked.connect(lambda: self.pick_tuning_delta_requested.emit("place_post_release_hold_sec", -0.05))
        self.pick_tune_post_plus_button.clicked.connect(lambda: self.pick_tuning_delta_requested.emit("place_post_release_hold_sec", 0.05))
        self.pick_tune_floor_minus_button.clicked.connect(lambda: self.pick_tuning_delta_requested.emit("z_carry_floor_mm", -1.0))
        self.pick_tune_floor_plus_button.clicked.connect(lambda: self.pick_tuning_delta_requested.emit("z_carry_floor_mm", 1.0))
        self.pick_tune_mode_button.clicked.connect(self.pick_release_mode_toggle_requested.emit)
        self.pick_tune_apply_button.clicked.connect(self.pick_tuning_apply_requested.emit)
        self.pick_tune_reset_button.clicked.connect(self.pick_tuning_reset_requested.emit)
        self.pick_tune_save_button.clicked.connect(self.pick_tuning_save_requested.emit)

        ssvep_title = QLabel("SSVEP")
        ssvep_title.setObjectName("panelSectionTitle")
        right_layout.addWidget(ssvep_title)

        ssvep_config_row = QHBoxLayout()
        self.ssvep_serial_edit = QLineEdit("auto")
        self.ssvep_serial_edit.setPlaceholderText("auto or COM3")
        self.ssvep_board_edit = QLineEdit("0")
        self.ssvep_board_edit.setPlaceholderText("Board ID")
        self.ssvep_apply_config_button = QPushButton("Apply SSVEP Config")
        ssvep_config_row.addWidget(QLabel("Serial"))
        ssvep_config_row.addWidget(self.ssvep_serial_edit, stretch=2)
        ssvep_config_row.addWidget(QLabel("Board"))
        ssvep_config_row.addWidget(self.ssvep_board_edit, stretch=1)
        ssvep_config_row.addWidget(self.ssvep_apply_config_button, stretch=2)
        right_layout.addLayout(ssvep_config_row)

        ssvep_pretrain_config_row = QHBoxLayout()
        self.ssvep_pretrain_preset_combo = QComboBox()
        for preset_key, preset in SSVEP_PRETRAIN_PRESETS.items():
            self.ssvep_pretrain_preset_combo.addItem(str(preset["label"]), preset_key)
        self.ssvep_pretrain_preset_combo.setCurrentIndex(
            max(0, self.ssvep_pretrain_preset_combo.findData(DEFAULT_SSVEP_PRETRAIN_PRESET))
        )
        ssvep_pretrain_config_row.addWidget(QLabel("Pretrain"))
        ssvep_pretrain_config_row.addWidget(self.ssvep_pretrain_preset_combo, stretch=1)
        right_layout.addLayout(ssvep_pretrain_config_row)

        self.ssvep_pretrain_hint_label = QLabel("")
        self.ssvep_pretrain_hint_label.setWordWrap(True)
        right_layout.addWidget(self.ssvep_pretrain_hint_label)
        self._update_ssvep_pretrain_hint()

        ssvep_row_1 = QHBoxLayout()
        self.ssvep_connect_button = QPushButton("连接SSVEP")
        self.ssvep_pretrain_button = QPushButton("开始预训练")
        ssvep_row_1.addWidget(self.ssvep_connect_button)
        ssvep_row_1.addWidget(self.ssvep_pretrain_button)
        right_layout.addLayout(ssvep_row_1)

        ssvep_row_2 = QHBoxLayout()
        self.ssvep_profile_combo = QComboBox()
        self.ssvep_profile_combo.addItem("自动（推荐）", AUTO_PROFILE_VALUE)
        self.ssvep_profile_combo.setMinimumContentsLength(18)
        self.ssvep_profile_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.ssvep_load_profile_button = QPushButton("加载Profile")
        ssvep_row_2.addWidget(self.ssvep_profile_combo, stretch=1)
        ssvep_row_2.addWidget(self.ssvep_load_profile_button)
        right_layout.addLayout(ssvep_row_2)

        ssvep_row_3 = QHBoxLayout()
        self.ssvep_open_profile_dir_button = QPushButton("打开Profile目录")
        ssvep_row_3.addWidget(self.ssvep_open_profile_dir_button)
        right_layout.addLayout(ssvep_row_3)

        self.ssvep_profile_hint_label = QLabel("当前无Profile，可直接预训练，或直接使用fallback自动模式。")
        self.ssvep_profile_hint_label.setObjectName("helpLabel")
        self.ssvep_profile_hint_label.setWordWrap(True)
        right_layout.addWidget(self.ssvep_profile_hint_label)

        ssvep_row_4 = QHBoxLayout()
        self.ssvep_stim_toggle_button = QPushButton("开启SSVEP刺激")
        self.ssvep_stim_toggle_button.setCheckable(True)
        self.ssvep_recognition_toggle_button = QPushButton("开启SSVEP识别")
        self.ssvep_recognition_toggle_button.setCheckable(True)
        ssvep_row_4.addWidget(self.ssvep_stim_toggle_button)
        ssvep_row_4.addWidget(self.ssvep_recognition_toggle_button)
        right_layout.addLayout(ssvep_row_4)

        for button in (
            self.robot_start_button,
            self.robot_connect_button,
            self.abort_button,
            self.reset_button,
            self.sucker_off_button,
            self.pick_slot1_button,
            self.pick_slot2_button,
            self.pick_slot3_button,
            self.pick_slot4_button,
            self.place_now_button,
            self.pick_r_minus_1_button,
            self.pick_r_plus_1_button,
            self.pick_r_reset_button,
            self.pick_tangent_minus_1_button,
            self.pick_tangent_plus_1_button,
            self.pick_tangent_reset_button,
            self.pick_theta_minus_1_button,
            self.pick_theta_plus_1_button,
            self.pick_theta_reset_button,
            self.pick_tune_approach_minus_button,
            self.pick_tune_approach_plus_button,
            self.pick_tune_descend_minus_button,
            self.pick_tune_descend_plus_button,
            self.pick_tune_place_minus_button,
            self.pick_tune_place_plus_button,
            self.pick_tune_pre_minus_button,
            self.pick_tune_pre_plus_button,
            self.pick_tune_hold_minus_button,
            self.pick_tune_hold_plus_button,
            self.pick_tune_lift_minus_button,
            self.pick_tune_lift_plus_button,
            self.pick_tune_release_minus_button,
            self.pick_tune_release_plus_button,
            self.pick_tune_post_minus_button,
            self.pick_tune_post_plus_button,
            self.pick_tune_floor_minus_button,
            self.pick_tune_floor_plus_button,
            self.pick_tune_mode_button,
            self.pick_tune_apply_button,
            self.pick_tune_reset_button,
            self.pick_tune_save_button,
            self.ssvep_apply_config_button,
            self.ssvep_connect_button,
            self.ssvep_pretrain_button,
            self.ssvep_load_profile_button,
            self.ssvep_open_profile_dir_button,
            self.ssvep_stim_toggle_button,
            self.ssvep_recognition_toggle_button,
        ):
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.ssvep_profile_combo.setMinimumWidth(0)
        self.ssvep_profile_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.ssvep_apply_config_button.clicked.connect(self._emit_ssvep_config_apply)
        self.ssvep_pretrain_preset_combo.currentIndexChanged.connect(self._update_ssvep_pretrain_hint)
        self.ssvep_connect_button.clicked.connect(self.ssvep_connect_requested.emit)
        self.ssvep_pretrain_button.clicked.connect(self.ssvep_pretrain_requested.emit)
        self.ssvep_load_profile_button.clicked.connect(self.ssvep_load_profile_requested.emit)
        self.ssvep_open_profile_dir_button.clicked.connect(self.ssvep_open_profile_dir_requested.emit)
        self.ssvep_stim_toggle_button.toggled.connect(self.ssvep_stim_toggled.emit)
        self.ssvep_recognition_toggle_button.toggled.connect(self._on_ssvep_recognition_toggled)

        self.robot_label = QLabel("Robot: disconnected")
        self.robot_label.setObjectName("rightStatusLabel")
        self.robot_label.setWordWrap(True)
        self.preflight_label = QLabel("Preflight: --")
        self.preflight_label.setObjectName("rightStatusLabel")
        self.preflight_label.setWordWrap(True)
        self.cyl_label = QLabel("Robot Cyl: --")
        self.cyl_label.setObjectName("rightStatusLabel")
        self.cyl_label.setWordWrap(True)
        self.selection_label = QLabel("Selection: none")
        self.selection_label.setObjectName("rightStatusLabel")
        self.selection_label.setWordWrap(True)
        self.targets_label = QLabel("Slots: []")
        self.targets_label.setObjectName("rightStatusLabel")
        self.targets_label.setWordWrap(True)
        self.raw_input_label = QLabel("Input: mi=-- ssvep=--")
        self.raw_input_label.setObjectName("rightStatusLabel")
        self.raw_input_label.setWordWrap(True)
        self.status_label = QLabel("Status: ready")
        self.status_label.setObjectName("rightStatusLabel")
        self.status_label.setWordWrap(True)
        self.vision_servo_status_label = QLabel("连续对中状态: --")
        self.vision_servo_status_label.setObjectName("rightStatusLabel")
        self.vision_servo_status_label.setWordWrap(True)
        self.vision_servo_command_label = QLabel("连续对中命令: --")
        self.vision_servo_command_label.setObjectName("rightStatusLabel")
        self.vision_servo_command_label.setWordWrap(True)
        self.vision_servo_debug_label = QLabel("连续对中Trace: --")
        self.vision_servo_debug_label.setObjectName("rightStatusLabel")
        self.vision_servo_debug_label.setWordWrap(True)
        self.ssvep_profile_label = QLabel("SSVEP Profile: --")
        self.ssvep_profile_label.setObjectName("rightStatusLabel")
        self.ssvep_profile_label.setWordWrap(True)
        self.ssvep_runtime_label = QLabel("SSVEP Runtime: --")
        self.ssvep_runtime_label.setObjectName("rightStatusLabel")
        self.ssvep_runtime_label.setWordWrap(True)
        self.ssvep_result_label = QLabel("SSVEP Raw: --")
        self.ssvep_result_label.setObjectName("rightStatusLabel")
        self.ssvep_result_label.setWordWrap(True)
        for label in (
            self.robot_label,
            self.preflight_label,
            self.cyl_label,
            self.selection_label,
            self.targets_label,
            self.raw_input_label,
            self.status_label,
            self.vision_servo_status_label,
            self.vision_servo_command_label,
            self.vision_servo_debug_label,
            self.ssvep_profile_label,
            self.ssvep_runtime_label,
            self.ssvep_result_label,
        ):
            label.setMinimumWidth(0)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            right_layout.addWidget(label)

        right_layout.addStretch(1)
        right_scroll.setWidget(right_content)
        right_shell_layout.addWidget(right_scroll)
        content_layout.addWidget(right_panel, stretch=0)

        self.bottom_status_label = QLabel("Vision: --")
        self.bottom_status_label.setObjectName("bottomStatusLabel")
        self.bottom_status_label.setWordWrap(True)
        self.bottom_status_label.setMinimumWidth(0)
        self.bottom_status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        main_layout.addWidget(self.bottom_status_label)

        self.log_view = QTextEdit()
        self.log_view.setObjectName("logView")
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(110)
        main_layout.addWidget(self.log_view)

        self.setCentralWidget(root)
        self._position_pose_overlay()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    @staticmethod
    def _ui_stylesheet() -> str:
        return (
            "QWidget {"
            "  color: #DCE8F4;"
            "  font-family: 'Microsoft YaHei', 'Segoe UI', 'PingFang SC', 'Consolas', sans-serif;"
            "  background-color: transparent;"
            "}"
            "QMainWindow {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0B111C, stop:1 #080C12);"
            "}"
            "QFrame#flowCard, QFrame#sidePanel, QFrame#poseOverlay, QFrame#shortcutCard {"
            "  border: 1px solid #2D3B4E;"
            "  border-radius: 10px;"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1A2230, stop:1 #161D29);"
            "}"
            "QFrame#flowCard {"
            "  border: 1px solid #314A68;"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1C2840, stop:1 #151E2F);"
            "}"
            "QFrame#shortcutCard {"
            "  border: 1px solid #406487;"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #202A3E, stop:1 #161D2D);"
            "}"
            "QFrame#sidePanel {"
            "  border-top-left-radius: 12px;"
            "  border-top-right-radius: 12px;"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #171F2D, stop:1 #111927);"
            "  border: 1px solid #2A3B53;"
            "}"
            "QWidget#sideContent {"
            "  background: transparent;"
            "}"
            "QScrollArea#sideScroll {"
            "  background: transparent;"
            "  border: none;"
            "}"
            "QLabel#topStatus {"
            "  background: rgba(10, 16, 26, 0.84);"
            "  border: 1px solid #2C3A4C;"
            "  border-radius: 10px;"
            "  padding: 8px 10px;"
            "  font-size: 10pt;"
            "  font-weight: 600;"
            "}"
            "QLabel#flowTitle,"
            "QLabel#flowSubtitle,"
            "QLabel#quickGuideLabel,"
            "QLabel#workflowStatusLabel,"
            "QLabel#workflowProgressLabel,"
            "QLabel#poseTitle,"
            "QLabel#panelSectionTitle,"
            "QLabel#bottomStatusLabel,"
            "QLabel#leftPanelTitle,"
            "QLabel#rightInfoLabel,"
            "QLabel#rightStatusLabel,"
            "QLabel#workflowStepTitle,"
            "QLabel#workflowStepDetail {"
            "  color: #E6F0FC;"
            "}"
            "QLabel#flowSubtitle,"
            "QLabel#quickGuideLabel,"
            "QLabel#workflowStepDetail,"
            "QLabel#rightStatusLabel,"
            "QLabel#bottomStatusLabel,"
            "QLabel#shortcutTitle,"
            "QLabel#keyHintLabel {"
            "  color: #B8C4D6;"
            "}"
            "QLabel#helpLabel {"
            "  color: #C9D4DF;"
            "  font-size: 9pt;"
            "}"
            "QLabel#flowTitle,"
            "QLabel#panelSectionTitle,"
            "QLabel#poseTitle,"
            "QLabel#quickGuideLabel {"
            "  font-weight: 700;"
            "}"
            "QLabel#flowTitle {"
            "  font-size: 13pt;"
            "}"
            "QLabel#panelSectionTitle {"
            "  font-size: 11pt;"
            "  margin-top: 6px;"
            "  margin-bottom: 2px;"
            "}"
            "QLabel#poseTitle {"
            "  font-size: 11pt;"
            "}"
            "QLabel#panelSectionTitle {"
            "  padding: 4px 0px 4px 0px;"
            "}"
            "QLabel#shortcutTitle {"
            "  font-size: 10pt;"
            "  font-weight: 700;"
            "  color: #E8F3FF;"
            "}"
            "QLabel#keyHintLabel {"
            "  color: #BDD0E6;"
            "  font-size: 9pt;"
            "}"
            "QLabel#workflowProgressLabel {"
            "  font-size: 10pt;"
            "  font-weight: 700;"
            "  color: #B6D7FF;"
            "  border: 1px solid #2D3C53;"
            "  border-radius: 8px;"
            "  padding: 2px 8px;"
            "  background: rgba(13, 24, 39, 0.55);"
            "}"
            "QLabel#workflowStatusLabel {"
            "  font-size: 10pt;"
            "}"
            "QLabel#workflowStatusLabel[statusTone='active'] {"
            "  color: #D7F7C6;"
            "}"
            "QLabel#workflowStatusLabel[statusTone='idle'] {"
            "  color: #C8D4DE;"
            "}"
            "QLabel#workflowStatusLabel[statusTone='error'] {"
            "  color: #FFD1D6;"
            "}"
            "QProgressBar#workflowProgressBar {"
            "  border: 1px solid #2E3A4A;"
            "  border-radius: 6px;"
            "  background: #121822;"
            "  text-align: center;"
            "  min-height: 14px;"
            "}"
            "QProgressBar#workflowProgressBar::chunk {"
            "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5CC6FF, stop:1 #8EE5C0);"
            "  border-radius: 5px;"
            "}"
            "QFrame#workflowStep {"
            "  border-radius: 8px;"
            "}"
            "QFrame#shortcutRow {"
            "  border: 1px solid #2D415A;"
            "  border-radius: 8px;"
            "  background: rgba(18, 31, 46, 0.55);"
            "}"
            "QLabel#workflowStepIndex {"
            "  font-size: 11pt;"
            "  font-weight: 700;"
            "  color: #1A2230;"
            "  background: #8BC6FF;"
            "  border-radius: 9px;"
            "  min-width: 18px;"
            "  min-height: 18px;"
            "  padding: 1px 0px 1px 0px;"
            "  text-align: center;"
            "}"
            "QLabel#workflowStepTitle {"
            "  font-size: 10pt;"
            "  font-weight: 700;"
            "}"
            "QLabel#workflowStepDetail {"
            "  font-size: 9pt;"
            "}"
            "QLabel#workflowStepState {"
            "  font-size: 9pt;"
            "  font-weight: 700;"
            "}"
            "QLabel#keyChip {"
            "  font-family: 'Consolas', 'Microsoft YaHei', 'Segoe UI', sans-serif;"
            "  font-size: 9pt;"
            "  font-weight: 700;"
            "  border: 1px solid #46688B;"
            "  border-radius: 8px;"
            "  padding: 4px 6px;"
            "  color: #EAF4FF;"
            "  background: #25344A;"
            "}"
            "QLabel#keyChip[tone='move'] {"
            "  background: #1F436D;"
            "  border-color: #4F7CAA;"
            "  color: #D8ECFF;"
            "}"
            "QLabel#keyChip[tone='target'] {"
            "  background: #2E5E75;"
            "  border-color: #5AA0B9;"
            "  color: #DDF5FF;"
            "}"
            "QLabel#keyChip[tone='action'] {"
            "  background: #2D6F6A;"
            "  border-color: #63B0A8;"
            "  color: #E1FFF4;"
            "}"
            "QLabel#keyChip[tone='logic'] {"
            "  background: #6F4A6B;"
            "  border-color: #B27FB0;"
            "  color: #F8E6FF;"
            "}"
            "QLabel#keyChip[tone='danger'] {"
            "  background: #6B2A3C;"
            "  border-color: #B54F64;"
            "  color: #FFE3EB;"
            "}"
            "QLabel#workflowStepTitle[stepState='done'] {"
            "  color: #D8F1E1;"
            "}"
            "QLabel#workflowStepDetail[stepState='done'] {"
            "  color: #B8DCC6;"
            "}"
            "QLabel#workflowStepState[stepState='done'] {"
            "  color: #B8E5C8;"
            "}"
            "QLabel#workflowStepTitle[stepState='active'] {"
            "  color: #F8FFF9;"
            "}"
            "QLabel#workflowStepDetail[stepState='active'] {"
            "  color: #D8ECF4;"
            "}"
            "QLabel#workflowStepState[stepState='active'] {"
            "  color: #74C7FF;"
            "}"
            "QLabel#workflowStepTitle[stepState='pending'] {"
            "  color: #A8B8CB;"
            "}"
            "QLabel#workflowStepDetail[stepState='pending'] {"
            "  color: #8B97A3;"
            "}"
            "QLabel#workflowStepState[stepState='pending'] {"
            "  color: #8B97A3;"
            "}"
            "QLabel#workflowStepTitle[stepState='error'] {"
            "  color: #FFD1D1;"
            "}"
            "QLabel#workflowStepDetail[stepState='error'] {"
            "  color: #FFCACA;"
            "}"
            "QLabel#workflowStepState[stepState='error'] {"
            "  color: #FFD1D1;"
            "}"
            "QFrame#workflowStep {"
            "  border: 1px solid #344A64;"
            "  background: #151E2A;"
            "}"
            "QFrame#workflowStep[stepState='pending'] {"
            "  background: #151A24;"
            "  border-color: #262F3A;"
            "}"
            "QFrame#workflowStep[stepState='active'] {"
            "  background: #1F3B54;"
            "  border: 1px solid #74C7FF;"
            "}"
            "QFrame#workflowStep[stepState='done'] {"
            "  background: #1F3B54;"
            "  border: 1px solid #3E86A8;"
            "}"
            "QFrame#workflowStep[stepState='error'] {"
            "  background: #3B1F2E;"
            "  border: 1px solid #B64B5E;"
            "}"
            "QLabel#rightInfoLabel,"
            "QLabel#rightStatusLabel,"
            "QLabel#bottomStatusLabel {"
            "  font-size: 10pt;"
            "}"
            "QPushButton {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2F3F57, stop:1 #25324A);"
            "  border: 1px solid #445A77;"
            "  border-radius: 8px;"
            "  padding: 7px 10px;"
            "  color: #EDF4FF;"
            "  font-weight: 600;"
            "}"
            "QPushButton:focus {"
            "  outline: 0;"
            "  border: 1px solid #84D0FF;"
            "}"
            "QPushButton[controlType='primary'] {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2C4E80, stop:1 #25405E);"
            "  border: 1px solid #4C6A92;"
            "  color: #F0F8FF;"
            "}"
            "QPushButton[controlType='primary']:hover {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3569AA, stop:1 #2B4B74);"
            "  border-color: #89D1FF;"
            "}"
            "QPushButton[controlType='warning'] {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7A5A27, stop:1 #4A3A15);"
            "  border: 1px solid #B98D34;"
            "}"
            "QPushButton[controlType='warning']:hover {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #8D6B2F, stop:1 #574217);"
            "}"
            "QPushButton[controlType='danger'] {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6A2A35, stop:1 #4D1F2A);"
            "  border: 1px solid #B14C5A;"
            "  color: #FFE5EA;"
            "}"
            "QPushButton[controlType='danger']:hover {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #8B3745, stop:1 #6A2A35);"
            "  border-color: #F18CA0;"
            "}"
            "QPushButton[controlType='neutral'] {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2F3F57, stop:1 #25324A);"
            "  border-color: #445A77;"
            "}"
            "QPushButton:hover {"
            "  border: 1px solid #73B8FF;"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #395675, stop:1 #30435D);"
            "}"
            "QPushButton:pressed {"
            "  background: #1D2A3F;"
            "  border-color: #5A91D9;"
            "}"
            "QPushButton:disabled {"
            "  background: #1A2330;"
            "  border-color: #2B3440;"
            "  color: #768395;"
            "}"
            "QLineEdit, QTextEdit, QComboBox, QScrollArea, QPlainTextEdit {"
            "  background: #121B27;"
            "  border: 1px solid #2F3F53;"
            "  border-radius: 8px;"
            "  color: #DCE8F4;"
            "  padding: 4px 6px;"
            "}"
            "QTextEdit#logView {"
            "  background: #11151B;"
            "  color: #D8DEE9;"
            "  font: 9pt 'Consolas';"
            "}"
            "QComboBox::down-arrow {"
            "  image: none;"
            "  border: 1px solid #8EA9C6;"
            "  width: 0px;"
            "  height: 0px;"
            "}"
            "QProgressBar {"
            "  border: 1px solid #2F3F54;"
            "  border-radius: 7px;"
            "  background: #101724;"
            "  text-align: center;"
            "  min-height: 12px;"
            "}"
            "QProgressBar::chunk {"
            "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6FD2FF, stop:1 #A3FFC2);"
            "  border-radius: 6px;"
            "}"
            "QScrollBar:vertical {"
            "  background: #121B27;"
            "  width: 10px;"
            "  margin: 0;"
            "}"
            "QScrollBar::handle:vertical {"
            "  background: #456182;"
            "  min-height: 24px;"
            "  border-radius: 5px;"
            "}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical, QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {"
            "  background: none;"
            "  border: none;"
            "  height: 0px;"
            "}"
            "QFrame#poseOverlay {"
            "  background: rgba(23, 31, 44, 0.88);"
            "}"
            "QLabel {"
            "  border: none;"
            "}"
        )

    @staticmethod
    def _apply_fusion_dark_palette() -> None:
        app = QApplication.instance()
        if app is None:
            return
        app.setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(14, 20, 31))
        palette.setColor(QPalette.WindowText, QColor(224, 234, 245))
        palette.setColor(QPalette.Base, QColor(18, 24, 34))
        palette.setColor(QPalette.AlternateBase, QColor(22, 29, 42))
        palette.setColor(QPalette.ToolTipBase, QColor(26, 34, 48))
        palette.setColor(QPalette.ToolTipText, QColor(232, 242, 255))
        palette.setColor(QPalette.Text, QColor(225, 234, 245))
        palette.setColor(QPalette.Button, QColor(27, 38, 54))
        palette.setColor(QPalette.ButtonText, QColor(235, 244, 255))
        palette.setColor(QPalette.BrightText, QColor(255, 94, 94))
        palette.setColor(QPalette.Highlight, QColor(97, 179, 255))
        palette.setColor(QPalette.HighlightedText, QColor(18, 24, 36))
        app.setPalette(palette)

    def _apply_ui_theme(self) -> None:
        self._apply_fusion_dark_palette()
        self.setStyleSheet(self._ui_stylesheet())

    def shutdown(self) -> None:
        app = QApplication.instance()
        if app is not None:
            try:
                app.removeEventFilter(self)
            except Exception:
                pass
        self.vision_widget.shutdown()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._position_pose_overlay()

    def _position_pose_overlay(self) -> None:
        if not hasattr(self, "pose_overlay") or not hasattr(self, "vision_widget"):
            return
        parent = self.vision_widget
        margin = 12
        max_width = max(180, parent.width() - margin * 2)
        max_height = max(140, parent.height() - margin * 2)
        width = min(320, max_width)
        height = min(290, max_height)
        x_pos = max(margin, parent.width() - width - margin)
        y_pos = margin
        self.pose_overlay.setGeometry(int(x_pos), int(y_pos), int(width), int(height))
        self.pose_overlay.raise_()

    def _make_key_chip(self, text: str, tone: str) -> QLabel:
        chip = QLabel(text)
        chip.setObjectName("keyChip")
        chip.setAlignment(Qt.AlignCenter)
        chip.setProperty("tone", tone)
        chip.setMinimumHeight(22)
        chip.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        return chip

    def _append_shortcut_row(
        self,
        container: QVBoxLayout,
        keys: list[tuple[str, str]],
        description: str,
        *,
        chip_width: int = 34,
    ) -> None:
        row = QFrame()
        row.setObjectName("shortcutRow")
        row_layout = QHBoxLayout(row)
        row_layout.setSpacing(6)
        row_layout.setContentsMargins(0, 0, 0, 0)
        for text, tone in keys:
            chip = self._make_key_chip(text, tone)
            chip.setMinimumWidth(chip_width)
            row_layout.addWidget(chip)
        row_layout.addSpacing(4)
        hint_label = QLabel(description)
        hint_label.setObjectName("keyHintLabel")
        hint_label.setWordWrap(True)
        row_layout.addWidget(hint_label, stretch=1)
        container.addWidget(row)

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showMaximized()
            return
        self.showFullScreen()

    def eventFilter(self, watched, event):  # noqa: N802
        event_type = event.type()
        if event_type not in (QEvent.KeyPress, QEvent.KeyRelease):
            return super().eventFilter(watched, event)
        if not self.isActiveWindow():
            return super().eventFilter(watched, event)
        if event.isAutoRepeat():
            return False
        if event_type == QEvent.KeyPress and event.key() in (Qt.Key_F11,):
            self._toggle_fullscreen()
            event.accept()
            return True
        token = self._key_to_token(event.key())
        if token is None:
            return super().eventFilter(watched, event)
        if event_type == QEvent.KeyPress:
            self.key_pressed.emit(token)
        else:
            self.key_released.emit(token)
        event.accept()
        return True

    def update_snapshot(self, snapshot: AppSnapshot) -> None:
        self.update_panels(snapshot)

    def update_panels(self, snapshot: AppSnapshot) -> None:
        state = snapshot.task_state
        robot = snapshot.robot
        vision = snapshot.vision
        ssvep = snapshot.ssvep
        vision_health_compact = self._compact_text(vision.health, max_len=120)
        robot_status_compact = self._compact_text(snapshot.last_robot_status, max_len=72)
        robot_error_compact = self._compact_text(snapshot.last_error, max_len=80)
        profile_path_compact = self._compact_path(ssvep.profile_path)
        latest_profile_compact = self._compact_path(ssvep.latest_profile_path)
        keyboard_mode = (
            str(snapshot.input_profile).strip().lower() == "operator_keyboard"
            and str(snapshot.move_source).strip().lower() == "sim"
            and str(snapshot.decision_source).strip().lower() == "sim"
        )
        input_label = "Keyboard" if keyboard_mode else str(snapshot.input_profile)
        mi_label = "disabled" if snapshot.move_source != "mi" else "enabled"
        ssvep_label = "disabled" if snapshot.decision_source != "ssvep" else "enabled"

        self._set_label_text(
            self.top_status_label,
            "State={} | Input={} | MI={} | SSVEP={} | robot={} vision={} | timer={}".format(
                state,
                input_label,
                mi_label,
                ssvep_label,
                snapshot.robot_mode,
                snapshot.vision_mode,
                self._format_timer(snapshot.motion_deadline_ts),
            )
        )
        self._update_workflow_panel(
            state=str(state).strip(),
            robot_connected=bool(robot.connected),
            keyboard_mode=keyboard_mode,
            frozen_targets_count=len(snapshot.frozen_targets),
            carrying=bool(snapshot.carrying),
        )
        self._set_label_text(
            self.robot_label,
            "Robot: connected={} health={} ack={} err={}".format(
                robot.connected,
                robot.health,
                robot.last_ack,
                robot.last_error,
            )
        )
        self._set_label_text(
            self.preflight_label,
            "Preflight: ok={} calibration_ready={} msg={}".format(
                robot.preflight_ok,
                robot.calibration_ready,
                robot.preflight_message,
            )
        )
        robot_cyl = robot.robot_cyl
        if robot_cyl is None and robot.scene_snapshot:
            robot_cyl = robot.scene_snapshot.get("robot_cyl")
        self._set_label_text(
            self.cyl_label,
            "Robot Cyl: {} | auto_z={} | kernel={}".format(
                robot_cyl,
                robot.auto_z_current,
                robot.control_kernel,
            )
        )
        self._set_label_text(
            self.selection_label,
            "Selection: id={} raw_center={}".format(
                snapshot.selected_target_id,
                snapshot.selected_target_raw_center,
            )
        )
        if vision.packet is not None:
            slot_summaries = []
            mapping_mode = str(vision.packet.get("mapping_mode", "absolute_base"))
            for slot in vision.packet.get("slots", []):
                if not isinstance(slot, dict) or not slot.get("valid"):
                    continue
                summary = self._format_slot_summary(slot)
                if summary:
                    slot_summaries.append(summary)
            self._set_label_text(
                self.targets_label,
                "Slots({}): {}".format(mapping_mode, ", ".join(slot_summaries) if slot_summaries else "[]"),
            )
        else:
            self._set_label_text(self.targets_label, f"Slots: {[target['id'] for target in snapshot.frozen_targets]}")
        if keyboard_mode:
            self._set_label_text(
                self.raw_input_label,
                "Input: keyboard active | 键盘控制已生效 | N=开始/选择 | R=复位 | WASD/方向键=移动 | 1-4=目标选择 | Enter/C=确认 | Esc/X=取消",
            )
        else:
            self._set_label_text(self.raw_input_label, "Input: ssvep={}".format(snapshot.last_ssvep_raw))
        self._set_label_text(
            self.status_label,
            "Status: robot={} error={} carrying={} vision={}".format(
                robot_status_compact,
                robot_error_compact,
                snapshot.carrying,
                vision_health_compact,
            )
        )
        self.update_vision_servo_debug()
        self._set_label_text(
            self.ssvep_profile_label,
            "SSVEP Profile: model={} source={} debug={} count={}\n{}\nlatest={}\nlast_pretrain={}".format(
                ssvep.model_name,
                ssvep.profile_source,
                ssvep.debug_keyboard,
                ssvep.profile_count,
                profile_path_compact,
                latest_profile_compact,
                ssvep.last_pretrain_time,
            )
        )
        self._set_label_text(
            self.ssvep_runtime_label,
            "SSVEP Runtime: running={} busy={} connected={} mode={} status={} err={}".format(
                ssvep.running,
                ssvep.busy,
                ssvep.connected,
                ssvep.mode,
                self._compact_text(ssvep.runtime_status, max_len=72),
                self._compact_text(ssvep.last_error, max_len=72),
            )
        )
        self._set_label_text(
            self.ssvep_result_label,
            "SSVEP Raw: state={} selected={} margin={} ratio={} stable={}".format(
                ssvep.last_state,
                ssvep.last_selected_freq,
                ssvep.last_margin,
                ssvep.last_ratio,
                ssvep.last_stable_windows,
            )
        )
        self._set_label_text(
            self.bottom_status_label,
            "Vision: {} | SSVEP mode={} | target_freq_map={}".format(
                vision_health_compact,
                ssvep.mode,
                self._compact_text(list(snapshot.target_frequency_map), max_len=72),
            )
        )

        if keyboard_mode:
            self._set_label_text(
                self.ssvep_profile_hint_label,
                "当前处于键盘手动模式：SSVEP控制已停用；如需独立调试SSVEP可运行 02_SSVEP。",
            )
        else:
            self._set_label_text(self.ssvep_profile_hint_label, str(ssvep.status_hint))
        self._update_profile_combo(
            ssvep.available_profiles,
            selected_path=ssvep.profile_path,
            auto_selected=ssvep.profile_source in {"latest", "fallback", "default", "current", "uninitialized"},
        )
        self._set_button_text(self.ssvep_connect_button, "重新连接SSVEP" if ssvep.connected else "连接SSVEP")
        ssvep_runtime_idle = not (
            ssvep.busy
            or ssvep.running
            or ssvep.connect_active
            or ssvep.pretrain_active
            or ssvep.online_active
        )
        self._set_button_enabled(self.ssvep_apply_config_button, (not keyboard_mode) and ssvep_runtime_idle)
        self.ssvep_pretrain_preset_combo.setEnabled((not keyboard_mode) and ssvep_runtime_idle)
        self.ssvep_serial_edit.setEnabled(not keyboard_mode)
        self.ssvep_board_edit.setEnabled(not keyboard_mode)
        self.ssvep_profile_combo.setEnabled(not keyboard_mode)
        self._set_button_enabled(self.ssvep_connect_button, (not keyboard_mode) and not ssvep.busy)
        self._set_button_enabled(self.ssvep_pretrain_button, (not keyboard_mode) and ssvep.connected and not ssvep.busy)
        self._set_button_enabled(self.ssvep_load_profile_button, (not keyboard_mode) and not ssvep.busy)
        stim_enabled = bool(ssvep.stim_enabled)
        self._set_button_checked(self.ssvep_stim_toggle_button, stim_enabled)
        self._set_button_text(self.ssvep_stim_toggle_button, "关闭SSVEP刺激" if stim_enabled else "开启SSVEP刺激")
        self._set_button_enabled(self.ssvep_stim_toggle_button, not keyboard_mode)

        recognition_enabled = bool(ssvep.running)
        self._set_button_checked(self.ssvep_recognition_toggle_button, recognition_enabled)
        self._set_button_text(
            self.ssvep_recognition_toggle_button,
            "关闭SSVEP识别" if recognition_enabled else "开启SSVEP识别",
        )
        self._set_button_enabled(
            self.ssvep_recognition_toggle_button,
            (not keyboard_mode) and (recognition_enabled or (ssvep.connected and not ssvep.busy)),
        )
        self._set_button_enabled(self.ssvep_open_profile_dir_button, not keyboard_mode)
        self._set_button_text(
            self.robot_start_button,
            "启动中..." if robot.start_active else ("待命中" if robot.connected else "启动机器人"),
        )
        self._set_button_enabled(self.robot_start_button, not robot.start_active)
        self._set_button_text(self.robot_connect_button, "断开连接" if robot.connected else "连接机器人")
        self._set_button_enabled(self.robot_connect_button, not robot.start_active)
        manual_enabled = bool(robot.connected)
        self._set_button_enabled(self.sucker_off_button, manual_enabled)
        self._set_button_enabled(self.pick_slot1_button, manual_enabled)
        self._set_button_enabled(self.pick_slot2_button, manual_enabled)
        self._set_button_enabled(self.pick_slot3_button, manual_enabled)
        self._set_button_enabled(self.pick_slot4_button, manual_enabled)
        self._set_button_enabled(self.place_now_button, manual_enabled)
        self._set_button_enabled(self.pick_r_minus_1_button, manual_enabled)
        self._set_button_enabled(self.pick_r_plus_1_button, manual_enabled)
        self._set_button_enabled(self.pick_r_reset_button, manual_enabled)
        self._set_button_enabled(self.pick_tangent_minus_1_button, manual_enabled)
        self._set_button_enabled(self.pick_tangent_plus_1_button, manual_enabled)
        self._set_button_enabled(self.pick_tangent_reset_button, manual_enabled)
        self._set_button_enabled(self.pick_theta_minus_1_button, manual_enabled)
        self._set_button_enabled(self.pick_theta_plus_1_button, manual_enabled)
        self._set_button_enabled(self.pick_theta_reset_button, manual_enabled)
        self._set_button_enabled(self.pick_tune_approach_minus_button, manual_enabled)
        self._set_button_enabled(self.pick_tune_approach_plus_button, manual_enabled)
        self._set_button_enabled(self.pick_tune_descend_minus_button, manual_enabled)
        self._set_button_enabled(self.pick_tune_descend_plus_button, manual_enabled)
        self._set_button_enabled(self.pick_tune_place_minus_button, manual_enabled)
        self._set_button_enabled(self.pick_tune_place_plus_button, manual_enabled)
        self._set_button_enabled(self.pick_tune_pre_minus_button, manual_enabled)
        self._set_button_enabled(self.pick_tune_pre_plus_button, manual_enabled)
        self._set_button_enabled(self.pick_tune_hold_minus_button, manual_enabled)
        self._set_button_enabled(self.pick_tune_hold_plus_button, manual_enabled)
        self._set_button_enabled(self.pick_tune_lift_minus_button, manual_enabled)
        self._set_button_enabled(self.pick_tune_lift_plus_button, manual_enabled)
        self._set_button_enabled(self.pick_tune_release_minus_button, manual_enabled)
        self._set_button_enabled(self.pick_tune_release_plus_button, manual_enabled)
        self._set_button_enabled(self.pick_tune_post_minus_button, manual_enabled)
        self._set_button_enabled(self.pick_tune_post_plus_button, manual_enabled)
        self._set_button_enabled(self.pick_tune_floor_minus_button, manual_enabled)
        self._set_button_enabled(self.pick_tune_floor_plus_button, manual_enabled)
        self._set_button_enabled(self.pick_tune_mode_button, manual_enabled)
        self._set_button_enabled(self.pick_tune_apply_button, manual_enabled)
        self._set_button_enabled(self.pick_tune_reset_button, True)
        self._set_button_enabled(self.pick_tune_save_button, True)

        self.scene_widget.update_scene(robot.scene_snapshot)

    def _update_workflow_panel(
        self,
        *,
        state: str,
        robot_connected: bool,
        keyboard_mode: bool,
        frozen_targets_count: int,
        carrying: bool,
    ) -> None:
        if not robot_connected:
            stage = 0
            next_hint = "请先连接机械臂，再开启流程。"
            command_hint = "按键策略已就绪：W/A/S/D 或方向键移动，1-4 选择。"
        elif state == "idle":
            stage = 0
            next_hint = "机械臂已连接，等待你的下一步。"
            command_hint = "待机中：R 复位，N 进入选位，Esc/X 取消。"
        elif state == "s1_mi_move":
            stage = 1
            next_hint = "请移动到可见目标区域。"
            command_hint = "W/S/A/D 或方向键：前后左右微调；继续对准可抓取区域。"
        elif state == "s1_decision":
            stage = 2
            next_hint = "准备进入选择，等待确认。"
            command_hint = "按 N 或 Enter 进入目标选择模式。"
        elif state == "s2_target_select":
            stage = 3
            if frozen_targets_count <= 0:
                next_hint = "等待检测到候选槽位后继续。"
                command_hint = "先移动到目标区边界，目标出现后按 1~4 选定目标。"
            else:
                next_hint = "直接使用数字键选定目标槽位。"
                command_hint = "1 / 2 / 3 / 4 直接选定不同槽位。"
        elif state == "s2_grab_confirm":
            stage = 4
            next_hint = "确认目标后执行抓取。"
            command_hint = "Enter 或 C 开始抓取；Esc/X 取消当前选中。"
        elif state == "s2_picking":
            stage = 4
            next_hint = "抓取中，保持静止等待动作结束。"
            command_hint = "请勿重复下发指令，等待执行结果。"
        elif state == "s3_mi_carry":
            stage = 5
            next_hint = "抓取成功，移动到放置目标点。"
            command_hint = "继续用 W/S/A/D 或方向键平移到目标槽位上方。"
        elif state == "s3_decision":
            stage = 6
            next_hint = "确认放置位置。"
            command_hint = "Enter/C 对齐后确认放置；Esc/X 取消流程。"
        elif state == "s3_placing":
            stage = 6
            next_hint = "放置中，完成后回到待机。"
            command_hint = "等待放置动作执行完成后再进行下一轮。"
        elif state == "finished":
            stage = self._workflow_steps_total
            next_hint = "本轮流程完成，等待下一次操作。"
            command_hint = "按 N 进入下一次选择，或 R 重置到初始位。"
        elif state == "error":
            stage = -1
            next_hint = "检测到异常，请查看日志与状态。"
            command_hint = "必要时按 Esc/X 停止动作后重置。"
        else:
            stage = 0
            next_hint = f"未知状态: {state}"
            command_hint = "保持键盘控制：W/A/S/D or方向键移动，1~4选择。"

        if stage <= 0:
            self.workflow_progress.setValue(0)
        else:
            self.workflow_progress.setValue(min(self._workflow_steps_total, stage))
        self._set_label_text(self.workflow_progress_label, f"进度 {min(stage, self._workflow_steps_total)}/{self._workflow_steps_total}")

        mode_text = "Keyboard" if keyboard_mode else "Hybrid"
        carrying_text = "抓取中" if carrying else "空载"
        self._set_label_text(
            self.workflow_status_label,
            f"模式={mode_text} | 持载={carrying_text} | 下一步建议：{next_hint}"
        )
        self._set_style_property(self.workflow_status_label, "statusTone", "error" if state == "error" else "active" if stage > 0 else "idle")
        for index, (row_widget, detail_label, state_label, title_label) in enumerate(self._workflow_step_rows, start=1):
            if stage == -1:
                row_label_state = "error"
                state_text = "异常"
            elif stage == 0:
                row_label_state = "pending"
                state_text = "待开始"
            elif index < stage:
                row_label_state = "done"
                state_text = "已完成"
            elif index == stage:
                row_label_state = "active"
                state_text = "进行中"
            else:
                row_label_state = "pending"
                state_text = "待开始"

            self._set_style_property(row_widget, "stepState", row_label_state)
            row_widget.setVisible(True)
            title_text = self._workflow_steps[index - 1][0] if index - 1 < len(self._workflow_steps) else ""
            self._set_label_text(title_label, title_text)
            self._set_style_property(title_label, "stepState", row_label_state)
            self._set_label_text(detail_label, f"{self._workflow_steps[index - 1][1]} | {self._workflow_steps[index - 1][2]}")
            self._set_style_property(detail_label, "stepState", row_label_state)
            self._set_label_text(state_label, state_text)
            self._set_style_property(state_label, "stepState", row_label_state)

        if stage == self._workflow_steps_total and not keyboard_mode:
            self.flow_subtitle_label.setText("本轮完成，可直接开始下一轮。")
        else:
            self.flow_subtitle_label.setText("按阶段跟随流程推进，当前状态与步骤会联动。")

        self._set_label_text(self.quick_guide_label, command_hint)

    def update_vision_payload(
        self,
        *,
        frame_bgr=_UNCHANGED,
        packet=_UNCHANGED,
        flash_enabled: bool | None = None,
        status_text: str | None = None,
        force: bool = False,
    ) -> None:
        if frame_bgr is not _UNCHANGED:
            self._vision_frame_cache = frame_bgr
        if packet is not _UNCHANGED:
            self._vision_packet_cache = packet
        if flash_enabled is not None:
            self._vision_flash_cache = bool(flash_enabled)
        if status_text is not None:
            self._vision_status_cache = str(status_text)

        frame_obj_id = None if self._vision_frame_cache is None else id(self._vision_frame_cache)
        packet_frame_id = None
        if isinstance(self._vision_packet_cache, dict):
            raw_frame_id = self._vision_packet_cache.get("frame_id")
            if raw_frame_id is not None:
                try:
                    packet_frame_id = int(raw_frame_id)
                except (TypeError, ValueError):
                    packet_frame_id = None

        changed = force
        changed = changed or (frame_obj_id != self._vision_last_frame_obj_id)
        changed = changed or (packet_frame_id != self._vision_last_packet_frame_id)
        changed = changed or (self._vision_flash_cache != self._vision_last_flash)
        changed = changed or (self._vision_status_cache != self._vision_last_status)
        if not changed:
            return

        self._vision_last_frame_obj_id = frame_obj_id
        self._vision_last_packet_frame_id = packet_frame_id
        self._vision_last_flash = self._vision_flash_cache
        self._vision_last_status = self._vision_status_cache
        self.vision_widget.set_payload(
            frame_bgr=self._vision_frame_cache,
            packet=self._vision_packet_cache,
            flash_enabled=self._vision_flash_cache,
            status_text=self._vision_status_cache,
        )

    def append_log(self, message: str) -> None:
        self.log_view.append(message)

    def update_vision_servo_debug(self, payload: dict[str, object] | None = None) -> None:
        data = payload or {}
        self._set_label_text(
            self.vision_servo_status_label,
            "连续对中状态: action={action} context={context} slot={slot_id} source={source} status={status} decision={decision_status}".format(
                action=data.get("action", "--"),
                context=data.get("context", "--"),
                slot_id=data.get("slot_id", "--"),
                source=data.get("source", "--"),
                status=data.get("status", "--"),
                decision_status=data.get("decision_status", "--"),
            )
        )
        self._set_label_text(
            self.vision_servo_command_label,
            "连续对中命令: {command} | rate={rates} | center={center_distance}px | z={current_z}/{confirm_z}".format(
                command=data.get("command", "--"),
                rates=data.get("rates", "--"),
                center_distance=data.get("center_distance_px", "--"),
                current_z=data.get("current_z_mm", "--"),
                confirm_z=data.get("confirm_z_mm", "--"),
            )
        )
        self._set_label_text(
            self.vision_servo_debug_label,
            "连续对中Trace: center_px={center_px} | {trace}".format(
                center_px=data.get("center_px", "--"),
                trace=data.get("trace", "--"),
            )
        )

    def update_pick_bias_display(
        self,
        radius_bias_mm: float,
        theta_bias_deg: float,
        tangent_bias_mm: float = 0.0,
    ) -> None:
        self._set_label_text(self.pick_r_bias_label, "半径偏差: {0:+.1f} mm".format(float(radius_bias_mm)))
        self._set_label_text(
            self.pick_tangent_bias_label,
            "切向偏差: {0:+.1f} mm".format(float(tangent_bias_mm)),
        )
        self._set_label_text(self.pick_theta_bias_label, "角度偏差: {0:+.1f} deg".format(float(theta_bias_deg)))

    def set_ssvep_runtime_config(self, *, serial_port: str, board_id: int) -> None:
        self.ssvep_serial_edit.setText(str(serial_port or "auto"))
        self.ssvep_board_edit.setText(str(int(board_id)))

    def ssvep_pretrain_config(self) -> dict[str, object]:
        preset_key = str(self.ssvep_pretrain_preset_combo.currentData() or DEFAULT_SSVEP_PRETRAIN_PRESET)
        preset = dict(SSVEP_PRETRAIN_PRESETS.get(preset_key, SSVEP_PRETRAIN_PRESETS[DEFAULT_SSVEP_PRETRAIN_PRESET]))
        preset["preset"] = preset_key
        preset["estimated_sec"] = _ssvep_pretrain_estimate_seconds(preset)
        return preset

    def ssvep_runtime_config(self) -> dict[str, object]:
        serial_port = str(self.ssvep_serial_edit.text()).strip() or "auto"
        try:
            board_id = int(str(self.ssvep_board_edit.text()).strip() or "0")
        except ValueError:
            board_id = 0
            self.ssvep_board_edit.setText("0")
        pretrain = self.ssvep_pretrain_config()
        return {
            "serial_port": serial_port,
            "board_id": board_id,
            "prepare_sec": float(pretrain["prepare_sec"]),
            "active_sec": float(pretrain["active_sec"]),
            "rest_sec": float(pretrain["rest_sec"]),
            "target_repeats": int(pretrain["target_repeats"]),
            "idle_repeats": int(pretrain["idle_repeats"]),
            "win_sec": float(pretrain["win_sec"]),
            "step_sec": float(pretrain["step_sec"]),
        }

    def _update_ssvep_pretrain_hint(self) -> None:
        preset = self.ssvep_pretrain_config()
        trials = 4 * int(preset["target_repeats"]) + int(preset["idle_repeats"])
        estimated_sec = float(preset["estimated_sec"])
        self.ssvep_pretrain_hint_label.setText(
            "Pretrain: {trials} trials, ~{seconds:.0f}s collection, win={win:g}s".format(
                trials=trials,
                seconds=estimated_sec,
                win=float(preset["win_sec"]),
            )
        )

    def _emit_ssvep_config_apply(self) -> None:
        config = self.ssvep_runtime_config()
        self.ssvep_config_apply_requested.emit(str(config["serial_port"]), int(config["board_id"]))

    def update_pick_tuning_display(self, tuning: dict[str, object] | None) -> None:
        values = dict(tuning or {})
        approach = float(values.get("pick_approach_z_mm", 0.0))
        descend = float(values.get("pick_descend_z_mm", 0.0))
        pre = float(values.get("pick_pre_suction_sec", 0.0))
        hold = float(values.get("pick_bottom_hold_sec", 0.0))
        lift = float(values.get("pick_lift_sec", 0.0))
        place_z = float(values.get("place_descend_z_mm", 0.0))
        release_mode = str(values.get("place_release_mode", "release"))
        release_sec = float(values.get("place_release_sec", 0.0))
        post = float(values.get("place_post_release_hold_sec", 0.0))
        floor = float(values.get("z_carry_floor_mm", 0.0))
        self._set_label_text(
            self.pick_tuning_label,
            "approach={0:.1f} descend={1:.1f} pre={2:.2f} hold={3:.2f} lift={4:.2f}\n"
            "place_z={5:.1f} mode={6} rel={7:.2f} post={8:.2f} floor={9:.1f}".format(
                approach,
                descend,
                pre,
                hold,
                lift,
                place_z,
                release_mode,
                release_sec,
                post,
                floor,
            ),
        )
        self._set_button_text(self.pick_tune_mode_button, f"模式: {release_mode}")

    def selected_ssvep_profile_path(self) -> str | None:
        selected = self.ssvep_profile_combo.currentData()
        if not selected or str(selected) == AUTO_PROFILE_VALUE:
            return None
        return str(selected)

    def is_ssvep_profile_auto_selected(self) -> bool:
        selected = self.ssvep_profile_combo.currentData()
        return not selected or str(selected) == AUTO_PROFILE_VALUE

    def _on_ssvep_recognition_toggled(self, enabled: bool) -> None:
        if enabled:
            self.ssvep_start_requested.emit()
            return
        self.ssvep_stop_requested.emit()

    def _update_profile_combo(
        self,
        profiles: tuple[tuple[str, str], ...],
        *,
        selected_path: str,
        auto_selected: bool = False,
    ) -> None:
        previous_path = self.selected_ssvep_profile_path() or ""
        target_path = selected_path or previous_path
        items = [("自动（推荐）", AUTO_PROFILE_VALUE)]
        items.extend(list(profiles))
        if not profiles:
            items.append(("暂无 Profile", ""))
        self.ssvep_profile_combo.blockSignals(True)
        self.ssvep_profile_combo.clear()
        selected_index = 0
        for index, (label, path) in enumerate(items):
            self.ssvep_profile_combo.addItem(label, path)
        if auto_selected and str(path) == AUTO_PROFILE_VALUE:
            selected_index = index
        elif path and path == target_path:
            selected_index = index
        self.ssvep_profile_combo.setCurrentIndex(selected_index)
        self.ssvep_profile_combo.blockSignals(False)

    @staticmethod
    def _set_style_property(widget: QWidget, prop: str, value: object | None) -> None:
        current = widget.property(prop)
        next_value = None if value is None else value
        if current == next_value:
            return
        widget.setProperty(prop, next_value)
        style = widget.style()
        if style is not None:
            style.unpolish(widget)
            style.polish(widget)
        widget.update()

    @staticmethod
    def _compact_text(value: object, *, max_len: int = 96) -> str:
        text = str(value)
        limit = max(8, int(max_len))
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    @staticmethod
    def _compact_path(value: object, *, max_len: int = 80) -> str:
        text = str(value or "").strip()
        if not text:
            return "--"
        try:
            name = Path(text).name
        except Exception:
            name = text
        if name:
            return MainWindow._compact_text(name, max_len=max_len)
        return MainWindow._compact_text(text, max_len=max_len)

    @staticmethod
    def _set_label_text(widget: QLabel, text: str) -> None:
        next_text = str(text)
        if widget.text() == next_text:
            return
        widget.setText(next_text)

    @staticmethod
    def _set_button_text(widget: QPushButton, text: str) -> None:
        next_text = str(text)
        if widget.text() == next_text:
            return
        widget.setText(next_text)

    @staticmethod
    def _set_button_enabled(widget: QPushButton, enabled: bool) -> None:
        next_enabled = bool(enabled)
        if widget.isEnabled() == next_enabled:
            return
        widget.setEnabled(next_enabled)

    @staticmethod
    def _set_button_checked(widget: QPushButton, checked: bool) -> None:
        next_checked = bool(checked)
        if widget.isChecked() == next_checked:
            return
        widget.blockSignals(True)
        widget.setChecked(next_checked)
        widget.blockSignals(False)

    @staticmethod
    def _format_timer(deadline: object) -> str:
        if not deadline:
            return "--"
        remaining = max(0.0, float(deadline) - time.time())
        return f"{remaining:.1f}s"

    @staticmethod
    def _format_slot_summary(slot: dict[str, object]) -> str | None:
        try:
            slot_id = int(slot.get("slot_id", 0))
            freq_hz = float(slot.get("freq_hz", 0.0))
        except (TypeError, ValueError):
            return None

        actionable = bool(slot.get("actionable", False))
        invalid_reason = str(slot.get("invalid_reason", "")).strip()
        status_suffix = " OK" if actionable else (" X:" + invalid_reason if invalid_reason else " X")
        if bool(slot.get("servo_required", False)) and not actionable:
            status_suffix = " SERVO"
        err = slot.get("estimated_xy_error_mm")
        if err is not None:
            try:
                status_suffix += " e={:.1f}mm".format(float(err))
            except (TypeError, ValueError):
                pass

        cyl = slot.get("cylindrical_center")
        if isinstance(cyl, (tuple, list)) and len(cyl) >= 2:
            try:
                theta = float(cyl[0])
                radius = float(cyl[1])
            except (TypeError, ValueError):
                return "[{}] {}Hz{}".format(slot_id, freq_hz, status_suffix)
            return "[{}] {}Hz theta={:.1f} r={:.1f}{}".format(slot_id, freq_hz, theta, radius, status_suffix)

        return "[{}] {}Hz{}".format(slot_id, freq_hz, status_suffix)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.isAutoRepeat():
            return
        token = self._key_to_token(event.key())
        if token is not None:
            self.key_pressed.emit(token)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.isAutoRepeat():
            return
        token = self._key_to_token(event.key())
        if token is not None:
            self.key_released.emit(token)
            event.accept()
            return
        super().keyReleaseEvent(event)

    @staticmethod
    def _key_to_token(key: int) -> str | None:
        key_map = {
            Qt.Key_N: "n",
            Qt.Key_R: "r",
            Qt.Key_A: "a",
            Qt.Key_D: "d",
            Qt.Key_W: "w",
            Qt.Key_S: "s",
            Qt.Key_Left: "left",
            Qt.Key_Right: "right",
            Qt.Key_Up: "up",
            Qt.Key_Down: "down",
            Qt.Key_Return: "enter",
            Qt.Key_Enter: "enter",
            Qt.Key_C: "c",
            Qt.Key_Escape: "esc",
            Qt.Key_X: "x",
            Qt.Key_1: "1",
            Qt.Key_2: "2",
            Qt.Key_3: "3",
            Qt.Key_4: "4",
        }
        return key_map.get(key)
