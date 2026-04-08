from __future__ import annotations

import time

try:
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen
    from PyQt5.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget
except ImportError as error:  # pragma: no cover - UI import guard
    raise RuntimeError("PyQt5 is required to use hybrid_controller.ui.main_window") from error


class ControlSceneWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._snapshot: dict[str, object] | None = None
        self.setMinimumHeight(300)

    def update_scene(self, snapshot: dict[str, object] | None) -> None:
        self._snapshot = snapshot
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.fillRect(self.rect(), QColor(20, 24, 30))
        if not self._snapshot:
            qp.setPen(QColor(220, 220, 220))
            qp.drawText(self.rect(), Qt.AlignCenter, "Control scene unavailable")
            return

        limits_x = tuple(self._snapshot["limits_x"])
        limits_y = tuple(self._snapshot["limits_y"])
        margin = 24
        width = max(10, self.width() - margin * 2)
        height = max(10, self.height() - margin * 2)

        def map_point(world_xy: tuple[float, float]) -> tuple[float, float]:
            x, y = world_xy
            px = margin + (float(x) - float(limits_x[0])) / (float(limits_x[1]) - float(limits_x[0])) * width
            py = margin + (float(limits_y[1]) - float(y)) / (float(limits_y[1]) - float(limits_y[0])) * height
            return px, py

        qp.setPen(QPen(QColor(90, 160, 220), 2, Qt.DashLine))
        qp.setBrush(Qt.NoBrush)
        qp.drawRect(margin, margin, width, height)

        home_x, home_y, _ = self._snapshot["home_pose"]
        hx, hy = map_point((home_x, home_y))
        qp.setPen(QPen(QColor(80, 220, 180), 2))
        qp.setBrush(QBrush(QColor(80, 220, 180, 90)))
        qp.drawRect(int(hx) - 6, int(hy) - 6, 12, 12)
        qp.drawText(int(hx) + 8, int(hy) - 8, "HOME")

        qp.setFont(QFont("Arial", 10, QFont.Bold))
        for slot in self._snapshot.get("pick_slots", []):
            sx, sy = map_point(tuple(slot["world_xy"]))
            if slot["occupied"]:
                fill = QColor(245, 170, 60)
            else:
                fill = QColor(90, 90, 90)
            outline = QColor(255, 220, 80) if slot.get("selected") else QColor(240, 240, 240)
            qp.setPen(QPen(outline, 2))
            qp.setBrush(QBrush(fill))
            qp.drawEllipse(int(sx) - 12, int(sy) - 12, 24, 24)
            qp.drawText(int(sx) - 8, int(sy) - 18, str(slot["slot_id"]))

        for slot in self._snapshot.get("place_slots", []):
            sx, sy = map_point(tuple(slot["world_xy"]))
            fill = QColor(70, 180, 120) if slot["occupied"] else QColor(50, 90, 70)
            qp.setPen(QPen(QColor(190, 255, 210), 2))
            qp.setBrush(QBrush(fill))
            qp.drawRect(int(sx) - 14, int(sy) - 10, 28, 20)
            qp.drawText(int(sx) - 10, int(sy) - 16, str(slot["slot_id"]))

        robot_x, robot_y = map_point(tuple(self._snapshot["robot_xy"]))
        qp.setPen(QPen(QColor(255, 90, 90), 2))
        qp.setBrush(QBrush(QColor(255, 90, 90)))
        qp.drawEllipse(int(robot_x) - 10, int(robot_y) - 10, 20, 20)
        if self._snapshot.get("carrying_target_id") is not None:
            qp.setPen(QPen(QColor(255, 235, 120), 1))
            qp.setBrush(QBrush(QColor(255, 235, 120)))
            qp.drawEllipse(int(robot_x) - 4, int(robot_y) - 24, 8, 8)

        qp.setPen(QColor(230, 230, 230))
        qp.setFont(QFont("Arial", 9))
        qp.drawText(
            margin,
            self.height() - 10,
            f"phase={self._snapshot.get('action_phase')} busy={self._snapshot.get('busy_action')} "
            f"ack={self._snapshot.get('last_ack')} err={self._snapshot.get('last_error')}",
        )


class MainWindow(QMainWindow):
    key_pressed = pyqtSignal(str)
    key_released = pyqtSignal(str)
    abort_requested = pyqtSignal()
    reset_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hybrid Controller v1")
        self.resize(980, 760)

        root = QWidget(self)
        layout = QVBoxLayout(root)

        self.state_label = QLabel("State: idle")
        self.sources_label = QLabel("Sources: move=sim decision=sim robot=fake vision=fake")
        self.simulation_label = QLabel("Simulation: profile=formal scenario=basic enabled=True")
        self.robot_label = QLabel("Robot: disconnected")
        self.timer_label = QLabel("Timer: --")
        self.cyl_label = QLabel("Robot Cyl: --")
        self.targets_label = QLabel("Targets: []")
        self.freq_map_label = QLabel("Freq map: []")
        self.selection_label = QLabel("Selection: none")
        self.raw_input_label = QLabel("Raw: mi=-- ssvep=--")
        self.status_label = QLabel("Status: ready")
        self.preflight_label = QLabel("Preflight: --")
        self.scene_widget = ControlSceneWidget()
        self.scene_view = QTextEdit()
        self.scene_view.setReadOnly(True)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.abort_button = QPushButton("Abort")
        self.reset_button = QPushButton("Reset")
        controls_row = QHBoxLayout()
        controls_row.addWidget(self.abort_button)
        controls_row.addWidget(self.reset_button)

        self.abort_button.clicked.connect(self.abort_requested.emit)
        self.reset_button.clicked.connect(self.reset_requested.emit)

        for widget in (
            self.state_label,
            self.sources_label,
            self.simulation_label,
            self.robot_label,
            self.timer_label,
            self.cyl_label,
            self.targets_label,
            self.freq_map_label,
            self.selection_label,
            self.raw_input_label,
            self.status_label,
            self.preflight_label,
            self.scene_widget,
            self.scene_view,
            self.log_view,
        ):
            if widget is self.scene_widget:
                layout.addLayout(controls_row)
            layout.addWidget(widget)

        self.setCentralWidget(root)

    def update_snapshot(
        self,
        snapshot: dict[str, object],
        runtime_info: dict[str, object],
        *,
        simulation_snapshot: dict[str, object] | None = None,
    ) -> None:
        state = snapshot["state"]
        context = snapshot["context"]
        self.state_label.setText(f"State: {state}")
        self.sources_label.setText(
            "Sources: "
            f"move={runtime_info['move_source']} decision={runtime_info['decision_source']} "
            f"robot={runtime_info['robot_mode']} vision={runtime_info['vision_mode']}"
        )
        self.simulation_label.setText(
            "Simulation: "
            f"enabled={runtime_info['simulation_enabled']} profile={runtime_info['timing_profile']} "
            f"scenario={runtime_info['scenario_name']}"
        )
        self.robot_label.setText(
            "Robot: "
            f"connected={runtime_info['robot_connected']} health={runtime_info['robot_health']} "
            f"last_ack={runtime_info['last_robot_ack']} last_err={runtime_info['last_robot_error']}"
        )

        deadline = context["motion_deadline_ts"]
        if deadline:
            remaining = max(0.0, float(deadline) - time.time())
            timer_text = f"{remaining:.1f}s remaining"
        else:
            timer_text = "--"
        self.timer_label.setText(f"Timer: {timer_text}")
        robot_cyl = runtime_info.get("robot_cyl")
        if robot_cyl is None and simulation_snapshot:
            robot_cyl = simulation_snapshot.get("robot_cyl")
        auto_z = runtime_info.get("auto_z_current")
        control_kernel = runtime_info.get("control_kernel")
        if control_kernel is None and simulation_snapshot:
            control_kernel = simulation_snapshot.get("control_kernel")
        self.cyl_label.setText(
            "Robot Cyl: "
            f"{robot_cyl} auto_z={auto_z} kernel={control_kernel} "
            f"limits={runtime_info.get('limits_cyl')}"
        )

        frozen_targets = context["frozen_targets"]
        self.targets_label.setText(f"Targets: {[target['id'] for target in frozen_targets]}")
        self.freq_map_label.setText(f"Freq map: {runtime_info['target_frequency_map']}")
        self.selection_label.setText(
            "Selection: "
            f"id={context['selected_target_id']} raw_center={context['selected_target_raw_center']}"
        )
        self.raw_input_label.setText(
            f"Raw: mi={runtime_info['last_mi_raw']} ssvep={runtime_info['last_ssvep_raw']}"
        )
        self.status_label.setText(
            "Status: "
            f"robot={context['last_robot_status']} error={context['last_error']} carrying={context['carrying']} "
            f"vision={runtime_info['vision_health']}"
        )
        self.preflight_label.setText(
            "Preflight: "
            f"ok={runtime_info.get('preflight_ok')} "
            f"calibration_ready={runtime_info.get('calibration_ready')} "
            f"message={runtime_info.get('preflight_message', '--')}"
        )
        self.scene_widget.update_scene(simulation_snapshot)
        self.scene_view.setPlainText(self._format_scene(simulation_snapshot))

    def append_log(self, message: str) -> None:
        self.log_view.append(message)

    @staticmethod
    def _format_scene(snapshot: dict[str, object] | None) -> str:
        if not snapshot:
            return "Simulation scene: unavailable"
        lines = [
            f"Scenario: {snapshot['scenario_name']}",
            f"Revision: {snapshot['revision']}",
            f"Robot XY: {snapshot['robot_xy']} z={snapshot.get('robot_z')}",
            f"Robot Cyl: {snapshot.get('robot_cyl')}",
            f"Limits: x={snapshot['limits_x']} y={snapshot['limits_y']}",
            f"Limits Cyl: {snapshot.get('limits_cyl')}",
            f"Home: {snapshot['home_pose']}",
            f"Busy: {snapshot['busy_action']}",
            f"Phase: {snapshot.get('action_phase')}",
            f"Calibration ready: {snapshot.get('calibration_ready')}",
            f"Auto Z: enabled={snapshot.get('auto_z_enabled')} current={snapshot.get('auto_z_current')}",
            f"Control kernel: {snapshot.get('control_kernel')}",
            f"IK: valid={snapshot.get('ik_valid')} error={snapshot.get('validation_error')}",
            f"Carrying target: {snapshot['carrying_target_id']}",
            f"Last ack: {snapshot.get('last_ack')}",
            f"Last world error: {snapshot['last_error']}",
            "Pick slots:",
        ]
        for target in snapshot.get("pick_slots", []):
            lines.append(
                f"  #{target['slot_id']} world={target['world_xy']} "
                f"occupied={target['occupied']} selected={target.get('selected', False)}"
            )
        lines.append("Place slots:")
        for target in snapshot.get("place_slots", []):
            lines.append(
                f"  #{target['slot_id']} world={target['world_xy']} occupied={target['occupied']}"
            )
        return "\n".join(lines)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.isAutoRepeat():
            event.accept()
            return
        token = self._event_to_token(event)
        if token:
            self.key_pressed.emit(token)
        event.accept()

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.isAutoRepeat():
            event.accept()
            return
        token = self._event_to_token(event)
        if token:
            self.key_released.emit(token)
        event.accept()

    @staticmethod
    def _event_to_token(event) -> str:
        key = event.key()
        token = ""
        if key in (Qt.Key_Return, Qt.Key_Enter):
            token = "enter"
        elif key == Qt.Key_Escape:
            token = "escape"
        elif key == Qt.Key_Left:
            token = "left"
        elif key == Qt.Key_Right:
            token = "right"
        elif key == Qt.Key_Up:
            token = "up"
        elif key == Qt.Key_Down:
            token = "down"
        if not token:
            token = event.text().lower()
        return token
