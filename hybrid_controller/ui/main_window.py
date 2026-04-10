from __future__ import annotations

import math
import time
from pathlib import Path

try:
    from PyQt5.QtCore import QEvent, QPointF, Qt, pyqtSignal
    from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
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
    ssvep_connect_requested = pyqtSignal()
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

        root = QWidget(self)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        self.top_status_label = QLabel("State: idle | Sources: --")
        self.top_status_label.setObjectName("topStatus")
        self.top_status_label.setStyleSheet("font: 12pt 'Consolas'; color: #E6E6E6;")
        self.top_status_label.setWordWrap(True)
        self.top_status_label.setMinimumWidth(0)
        self.top_status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        main_layout.addWidget(self.top_status_label)

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
        self.pose_overlay.setStyleSheet("QFrame#poseOverlay { background: rgba(23, 27, 34, 230); border: 1px solid #2E3540; border-radius: 6px; }")
        pose_overlay_layout = QVBoxLayout(self.pose_overlay)
        pose_overlay_layout.setContentsMargins(8, 8, 8, 8)
        pose_overlay_layout.setSpacing(6)
        self.pose_title_label = QLabel("Robot Pose")
        self.pose_title_label.setStyleSheet("font: bold 11pt 'Arial'; color: #F0F4F8; border: none;")
        pose_overlay_layout.addWidget(self.pose_title_label)
        self.scene_widget = ControlSceneWidget(self.pose_overlay)
        pose_overlay_layout.addWidget(self.scene_widget, stretch=1)
        self.pose_overlay.show()
        self.pose_overlay.raise_()

        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_panel.setMinimumWidth(320)
        right_panel.setMaximumWidth(440)
        right_panel.setStyleSheet("QFrame { background: #171B22; border: 1px solid #2E3540; }")
        right_shell_layout = QVBoxLayout(right_panel)
        right_shell_layout.setContentsMargins(0, 0, 0, 0)
        right_shell_layout.setSpacing(0)
        right_scroll = QScrollArea(right_panel)
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_content = QWidget()
        right_content.setMinimumWidth(0)
        right_content.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        right_layout = QVBoxLayout(right_content)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)

        controls_row = QGridLayout()
        controls_row.setHorizontalSpacing(6)
        controls_row.setVerticalSpacing(6)
        self.robot_start_button = QPushButton("启动机械臂")
        self.robot_connect_button = QPushButton("连接机器人")
        self.abort_button = QPushButton("Abort")
        self.reset_button = QPushButton("Reset")
        controls_row.addWidget(self.robot_start_button, 0, 0)
        controls_row.addWidget(self.robot_connect_button, 0, 1)
        controls_row.addWidget(self.abort_button, 1, 0)
        controls_row.addWidget(self.reset_button, 1, 1)
        controls_row.setColumnStretch(0, 1)
        controls_row.setColumnStretch(1, 1)
        right_layout.addLayout(controls_row)

        self.robot_start_button.clicked.connect(self.robot_start_requested.emit)
        self.robot_connect_button.clicked.connect(self.robot_connect_requested.emit)
        self.abort_button.clicked.connect(self.abort_requested.emit)
        self.reset_button.clicked.connect(self.reset_requested.emit)

        pick_title = QLabel("Pick/Place Debug")
        pick_title.setStyleSheet("font: bold 11pt 'Arial'; color: #F0F4F8;")
        right_layout.addWidget(pick_title)

        pick_row = QGridLayout()
        pick_row.setHorizontalSpacing(6)
        pick_row.setVerticalSpacing(6)
        self.pick_slot1_button = QPushButton("Pick 1")
        self.pick_slot2_button = QPushButton("Pick 2")
        self.pick_slot3_button = QPushButton("Pick 3")
        self.pick_slot4_button = QPushButton("Pick 4")
        pick_row.addWidget(self.pick_slot1_button, 0, 0)
        pick_row.addWidget(self.pick_slot2_button, 0, 1)
        pick_row.addWidget(self.pick_slot3_button, 1, 0)
        pick_row.addWidget(self.pick_slot4_button, 1, 1)
        pick_row.setColumnStretch(0, 1)
        pick_row.setColumnStretch(1, 1)
        right_layout.addLayout(pick_row)

        pick_row2 = QHBoxLayout()
        self.place_now_button = QPushButton("Place")
        pick_row2.addWidget(self.place_now_button)
        right_layout.addLayout(pick_row2)

        pick_bias_row = QHBoxLayout()
        self.pick_r_minus_1_button = QPushButton("r-1")
        self.pick_r_plus_1_button = QPushButton("r+1")
        self.pick_r_reset_button = QPushButton("r reset")
        pick_bias_row.addWidget(self.pick_r_minus_1_button)
        pick_bias_row.addWidget(self.pick_r_plus_1_button)
        pick_bias_row.addWidget(self.pick_r_reset_button)
        right_layout.addLayout(pick_bias_row)

        pick_theta_bias_row = QHBoxLayout()
        self.pick_theta_minus_1_button = QPushButton("th-1")
        self.pick_theta_plus_1_button = QPushButton("th+1")
        self.pick_theta_reset_button = QPushButton("th reset")
        pick_theta_bias_row.addWidget(self.pick_theta_minus_1_button)
        pick_theta_bias_row.addWidget(self.pick_theta_plus_1_button)
        pick_theta_bias_row.addWidget(self.pick_theta_reset_button)
        right_layout.addLayout(pick_theta_bias_row)

        self.pick_r_bias_label = QLabel("Pick r bias: +0.0 mm")
        self.pick_r_bias_label.setStyleSheet("font: 10pt 'Consolas'; color: #D8DEE9; border: none;")
        right_layout.addWidget(self.pick_r_bias_label)
        self.pick_theta_bias_label = QLabel("Pick theta bias: +0.0 deg")
        self.pick_theta_bias_label.setStyleSheet("font: 10pt 'Consolas'; color: #D8DEE9; border: none;")
        right_layout.addWidget(self.pick_theta_bias_label)

        pick_tuning_title = QLabel("Pick Tuning")
        pick_tuning_title.setStyleSheet("font: bold 10pt 'Arial'; color: #F0F4F8;")
        right_layout.addWidget(pick_tuning_title)

        self.pick_tuning_label = QLabel("approach=130.0 descend=85.0 pre=0.25 hold=0.15 lift=0.80\nplace_z=85.0 release=release rel=0.25 post=0.10 floor=160.0")
        self.pick_tuning_label.setWordWrap(True)
        self.pick_tuning_label.setStyleSheet("font: 9pt 'Consolas'; color: #D8DEE9; border: none;")
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
        self.pick_tune_apply_button = QPushButton("应用到机器人")
        self.pick_tune_reset_button = QPushButton("恢复默认")
        self.pick_tune_save_button = QPushButton("保存配置")
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
        ssvep_title.setStyleSheet("font: bold 11pt 'Arial'; color: #F0F4F8;")
        right_layout.addWidget(ssvep_title)

        ssvep_row_1 = QHBoxLayout()
        self.ssvep_connect_button = QPushButton("连接设备")
        self.ssvep_pretrain_button = QPushButton("开始预训练")
        ssvep_row_1.addWidget(self.ssvep_connect_button)
        ssvep_row_1.addWidget(self.ssvep_pretrain_button)
        right_layout.addLayout(ssvep_row_1)

        ssvep_row_2 = QHBoxLayout()
        self.ssvep_profile_combo = QComboBox()
        self.ssvep_profile_combo.addItem("自动（最新训练）", AUTO_PROFILE_VALUE)
        self.ssvep_profile_combo.setMinimumContentsLength(18)
        self.ssvep_profile_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.ssvep_load_profile_button = QPushButton("加载选中")
        ssvep_row_2.addWidget(self.ssvep_profile_combo, stretch=1)
        ssvep_row_2.addWidget(self.ssvep_load_profile_button)
        right_layout.addLayout(ssvep_row_2)

        ssvep_row_3 = QHBoxLayout()
        self.ssvep_open_profile_dir_button = QPushButton("打开 Profile 目录")
        ssvep_row_3.addWidget(self.ssvep_open_profile_dir_button)
        right_layout.addLayout(ssvep_row_3)

        self.ssvep_profile_hint_label = QLabel("当前没有已训练 Profile，可先预训练，或直接用默认 fallback 启动。")
        self.ssvep_profile_hint_label.setWordWrap(True)
        self.ssvep_profile_hint_label.setStyleSheet(
            "font: 9pt 'Microsoft YaHei'; color: #C9D4DF; border: none;"
        )
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
            self.pick_slot1_button,
            self.pick_slot2_button,
            self.pick_slot3_button,
            self.pick_slot4_button,
            self.place_now_button,
            self.pick_r_minus_1_button,
            self.pick_r_plus_1_button,
            self.pick_r_reset_button,
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

        self.ssvep_connect_button.clicked.connect(self.ssvep_connect_requested.emit)
        self.ssvep_pretrain_button.clicked.connect(self.ssvep_pretrain_requested.emit)
        self.ssvep_load_profile_button.clicked.connect(self.ssvep_load_profile_requested.emit)
        self.ssvep_open_profile_dir_button.clicked.connect(self.ssvep_open_profile_dir_requested.emit)
        self.ssvep_stim_toggle_button.toggled.connect(self.ssvep_stim_toggled.emit)
        self.ssvep_recognition_toggle_button.toggled.connect(self._on_ssvep_recognition_toggled)

        self.robot_label = QLabel("Robot: disconnected")
        self.robot_label.setWordWrap(True)
        self.preflight_label = QLabel("Preflight: --")
        self.preflight_label.setWordWrap(True)
        self.cyl_label = QLabel("Robot Cyl: --")
        self.cyl_label.setWordWrap(True)
        self.selection_label = QLabel("Selection: none")
        self.selection_label.setWordWrap(True)
        self.targets_label = QLabel("Slots: []")
        self.targets_label.setWordWrap(True)
        self.raw_input_label = QLabel("Input: mi=-- ssvep=--")
        self.raw_input_label.setWordWrap(True)
        self.status_label = QLabel("Status: ready")
        self.status_label.setWordWrap(True)
        self.ssvep_profile_label = QLabel("SSVEP Profile: --")
        self.ssvep_profile_label.setWordWrap(True)
        self.ssvep_runtime_label = QLabel("SSVEP Runtime: --")
        self.ssvep_runtime_label.setWordWrap(True)
        self.ssvep_result_label = QLabel("SSVEP Raw: --")
        self.ssvep_result_label.setWordWrap(True)
        for label in (
            self.robot_label,
            self.preflight_label,
            self.cyl_label,
            self.selection_label,
            self.targets_label,
            self.raw_input_label,
            self.status_label,
            self.ssvep_profile_label,
            self.ssvep_runtime_label,
            self.ssvep_result_label,
        ):
            label.setStyleSheet("font: 10pt 'Consolas'; color: #D8DEE9; border: none;")
            label.setMinimumWidth(0)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            right_layout.addWidget(label)

        right_layout.addStretch(1)
        right_scroll.setWidget(right_content)
        right_shell_layout.addWidget(right_scroll)
        content_layout.addWidget(right_panel, stretch=0)

        self.bottom_status_label = QLabel("Vision: --")
        self.bottom_status_label.setStyleSheet("font: 10pt 'Consolas'; color: #E6E6E6;")
        self.bottom_status_label.setWordWrap(True)
        self.bottom_status_label.setMinimumWidth(0)
        self.bottom_status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        main_layout.addWidget(self.bottom_status_label)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(110)
        self.log_view.setStyleSheet("background: #11151B; color: #D8DEE9; font: 9pt 'Consolas';")
        main_layout.addWidget(self.log_view)

        self.setCentralWidget(root)
        self._position_pose_overlay()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

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

        self._set_label_text(
            self.top_status_label,
            "State={} | move={} decision={} robot={} vision={} | timer={}".format(
                state,
                snapshot.move_source,
                snapshot.decision_source,
                snapshot.robot_mode,
                snapshot.vision_mode,
                self._format_timer(snapshot.motion_deadline_ts),
            )
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

        self._set_label_text(self.ssvep_profile_hint_label, str(ssvep.status_hint))
        self._update_profile_combo(
            ssvep.available_profiles,
            selected_path=ssvep.profile_path,
            auto_selected=ssvep.profile_source in {"latest", "fallback", "default", "current", "uninitialized"},
        )
        self._set_button_text(self.ssvep_connect_button, "重新连接设备" if ssvep.connected else "连接设备")
        self._set_button_enabled(self.ssvep_connect_button, not ssvep.busy)
        self._set_button_enabled(self.ssvep_pretrain_button, ssvep.connected and not ssvep.busy)
        self._set_button_enabled(self.ssvep_load_profile_button, not ssvep.busy)
        stim_enabled = bool(ssvep.stim_enabled)
        self._set_button_checked(self.ssvep_stim_toggle_button, stim_enabled)
        self._set_button_text(self.ssvep_stim_toggle_button, "关闭SSVEP刺激" if stim_enabled else "开启SSVEP刺激")
        self._set_button_enabled(self.ssvep_stim_toggle_button, True)

        recognition_enabled = bool(ssvep.running)
        self._set_button_checked(self.ssvep_recognition_toggle_button, recognition_enabled)
        self._set_button_text(
            self.ssvep_recognition_toggle_button,
            "关闭SSVEP识别" if recognition_enabled else "开启SSVEP识别",
        )
        self._set_button_enabled(
            self.ssvep_recognition_toggle_button,
            recognition_enabled or (ssvep.connected and not ssvep.busy),
        )
        self._set_button_enabled(self.ssvep_open_profile_dir_button, True)
        self._set_button_text(
            self.robot_start_button,
            "启动中..." if robot.start_active else ("重启机械臂" if robot.connected else "启动机械臂"),
        )
        self._set_button_enabled(self.robot_start_button, not robot.start_active)
        self._set_button_text(self.robot_connect_button, "重连机器人" if robot.connected else "连接机器人")
        self._set_button_enabled(self.robot_connect_button, not robot.start_active)
        manual_enabled = bool(robot.connected)
        self._set_button_enabled(self.pick_slot1_button, manual_enabled)
        self._set_button_enabled(self.pick_slot2_button, manual_enabled)
        self._set_button_enabled(self.pick_slot3_button, manual_enabled)
        self._set_button_enabled(self.pick_slot4_button, manual_enabled)
        self._set_button_enabled(self.place_now_button, manual_enabled)
        self._set_button_enabled(self.pick_r_minus_1_button, manual_enabled)
        self._set_button_enabled(self.pick_r_plus_1_button, manual_enabled)
        self._set_button_enabled(self.pick_r_reset_button, manual_enabled)
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

    def update_pick_bias_display(self, radius_bias_mm: float, theta_bias_deg: float) -> None:
        self._set_label_text(self.pick_r_bias_label, "Pick r bias: {0:+.1f} mm".format(float(radius_bias_mm)))
        self._set_label_text(self.pick_theta_bias_label, "Pick theta bias: {0:+.1f} deg".format(float(theta_bias_deg)))

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
        self._set_button_text(self.pick_tune_mode_button, f"mode: {release_mode}")

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
        items = [("自动（最新训练）", AUTO_PROFILE_VALUE)]
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
    def _compact_text(value: object, *, max_len: int = 96) -> str:
        text = str(value)
        limit = max(8, int(max_len))
        if len(text) <= limit:
            return text
        return text[: limit - 1] + "…"

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
