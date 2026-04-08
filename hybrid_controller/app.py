from __future__ import annotations

import argparse
import sys
import threading
from dataclasses import replace
from typing import Callable

from hybrid_controller.adapters.control_sim_slots import ControlSimSlotCatalog
from hybrid_controller.adapters.mi_adapter import MIAdapter
from hybrid_controller.adapters.robot_client import RobotClient, fetch_robot_status
from hybrid_controller.adapters.sim_input import SimInputAdapter
from hybrid_controller.adapters.ssvep_adapter import SSVEPAdapter
from hybrid_controller.cylindrical import cartesian_to_cylindrical
from hybrid_controller.config import AppConfig
from hybrid_controller.controller.events import Effect, Event
from hybrid_controller.controller.state_machine import TaskState
from hybrid_controller.controller.task_controller import TaskController
from hybrid_controller.debug.event_logger import EventLogger
from hybrid_controller.debug.fake_robot_server import FakeRobotServer
from hybrid_controller.debug.simulation_world import SimulationWorld
from hybrid_controller.integrations.mi_runtime import MIRuntime
from hybrid_controller.integrations.ssvep_runtime import SSVEPRuntime
from hybrid_controller.integrations.vision_runtime import VisionRuntime

try:
    from PyQt5.QtCore import QObject, QTimer, pyqtSignal
    from PyQt5.QtWidgets import QApplication
except ImportError as error:  # pragma: no cover - UI import guard
    raise RuntimeError("PyQt5 is required to run hybrid_controller.app") from error

from hybrid_controller.ui.main_window import MainWindow


class _RuntimeSignalBridge(QObject):
    event_received = pyqtSignal(object)
    runtime_status_received = pyqtSignal(str, str)
    remote_snapshot_received = pyqtSignal(object)


class HybridControllerApplication:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.controller = TaskController(config)
        self.logger = EventLogger(config.event_log_path)
        self.sim_input = SimInputAdapter(move_source=config.move_source, decision_source=config.decision_source)
        self.ssvep_adapter = SSVEPAdapter(config.ssvep_freqs)
        self.mi_adapter = MIAdapter(config)
        self.main_window = MainWindow()
        self.slot_catalog = ControlSimSlotCatalog(config) if config.control_sim_enabled else None
        self.simulation_world: SimulationWorld | None = None
        if config.control_sim_enabled and config.robot_mode == "fake":
            self.simulation_world = SimulationWorld(config, scenario_name=config.scenario_name)
        self.fake_robot_server: FakeRobotServer | None = None
        self.vision_runtime: VisionRuntime | None = None
        self.ssvep_runtime: SSVEPRuntime | None = None
        self.mi_runtime: MIRuntime | None = None
        self.timers: dict[str, QTimer] = {}
        self._last_logged_world_revision: int | None = None
        self._latest_world_snapshot: dict[str, object] | None = None
        self._scene_revision = 0
        self._scene_pick_slots: dict[int, dict[str, object]] = {}
        self._scene_place_slots: dict[int, dict[str, object]] = {}
        self._scene_carrying_target_id: int | None = None
        self._scene_last_error: str | None = None
        self._pressed_move_tokens: set[str] = set()
        self._remote_snapshot_lock = threading.Lock()
        self._remote_snapshot_cache: dict[str, object] | None = None
        self._remote_snapshot_inflight = False
        self._reset_control_scene_state()
        self._bridge = _RuntimeSignalBridge()
        self._bridge.event_received.connect(self.dispatch_event)
        self._bridge.runtime_status_received.connect(self._handle_runtime_status)
        self._bridge.remote_snapshot_received.connect(self._on_remote_snapshot_received)

        self.runtime_info: dict[str, object] = {
            "simulation_enabled": config.simulation_enabled,
            "timing_profile": config.timing_profile,
            "scenario_name": config.scenario_name,
            "move_source": config.move_source,
            "decision_source": config.decision_source,
            "robot_mode": config.robot_mode,
            "vision_mode": config.vision_mode,
            "robot_connected": False,
            "robot_health": "unknown",
            "vision_health": "unknown",
            "last_robot_ack": "--",
            "last_robot_error": "--",
            "last_ssvep_raw": "--",
            "last_mi_raw": "--",
            "target_frequency_map": [],
            "preflight_ok": True,
            "preflight_message": "not_required",
            "calibration_ready": None,
            "robot_cyl": None,
            "limits_cyl": None,
            "auto_z_current": None,
            "control_kernel": "cylindrical_kernel",
        }

        self.robot_client = RobotClient(
            config.robot_host,
            config.robot_port,
            event_callback=self._queue_event,
            timeout_sec=config.robot_timeout_sec,
            reconnect_delay_sec=config.robot_reconnect_delay_sec,
        )

        self.main_window.key_pressed.connect(self._on_key_pressed)
        self.main_window.key_released.connect(self._on_key_released)
        self.main_window.abort_requested.connect(self._on_abort_requested)
        self.main_window.reset_requested.connect(self._on_reset_requested)
        self._start_ui_refresh_timer()
        self._start_teleop_timer()
        self._setup_robot_mode()
        self._setup_vision_mode()
        self._setup_brain_sources()
        self._update_ssvep_mode()
        self._capture_world_snapshot(reason="startup", force=True)
        self._refresh_view()

    def shutdown(self) -> None:
        for timer in list(self.timers.values()):
            timer.stop()
            timer.deleteLater()
        self.timers.clear()

        if self.vision_runtime is not None:
            self.vision_runtime.stop()
            self.vision_runtime = None
        if self.ssvep_runtime is not None:
            self.ssvep_runtime.stop()
            self.ssvep_runtime = None
        if self.mi_runtime is not None:
            self.mi_runtime.stop()
            self.mi_runtime = None

        self.robot_client.close()
        if self.fake_robot_server is not None:
            self.fake_robot_server.stop()
            self.fake_robot_server = None

    def dispatch_event(self, event: Event) -> None:
        if event.type == "start_task" and not self._can_start_task():
            self.logger.log_event(event, self.controller.state.value)
            self.main_window.append_log(
                f"Task start blocked by preflight: {self.runtime_info.get('preflight_message', 'unknown')}"
            )
            self._update_runtime_health()
            self._refresh_view()
            return
        selected_before = self.controller.context.selected_target_id
        self.logger.log_event(event, self.controller.state.value)
        controller_event = event
        if event.source == "robot" and event.type == "robot_ack":
            self.runtime_info["last_robot_ack"] = str(event.value)
            ack = str(event.value or "").strip().upper()
            if ack == "ABORT":
                self._stop_teleop_motion(send_command=False, reason="robot_abort_ack")
                controller_event = None
                effects = self._force_controller_error("Abort requested by operator.")
                self._update_control_scene_from_event(
                    Event(source="robot", type="robot_error", value="Abort requested by operator."),
                    selected_before=selected_before,
                )
                for effect in effects:
                    self.logger.log_effect(effect, self.controller.state.value)
                    self._apply_effect(effect)
                self._update_runtime_health()
                self._update_ssvep_mode()
                self._refresh_view()
                return
            if ack == "RESET":
                self._stop_teleop_motion(send_command=False, reason="robot_reset_ack")
                controller_event = Event(source="system", type="reset_task", timestamp=event.timestamp)
        if event.source == "robot" and event.type == "robot_error":
            self._stop_teleop_motion(send_command=False, reason="robot_error")
            self.runtime_info["last_robot_error"] = str(event.value)
        effects = [] if controller_event is None else self.controller.handle_event(controller_event)
        self._update_control_scene_from_event(event, selected_before=selected_before)
        for effect in effects:
            self.logger.log_effect(effect, self.controller.state.value)
            self._apply_effect(effect)
        if self.config.control_sim_enabled and controller_event is not None and controller_event.type == "reset_task":
            self._reset_control_scene_state()
            self._publish_control_sim_targets()
        self._update_runtime_health()
        self._update_ssvep_mode()
        self._refresh_view()

    def dispatch_ssvep_command(self, command: object) -> None:
        self.runtime_info["last_ssvep_raw"] = str(command)
        self.logger.log_raw_input("ssvep", command)
        event = self.ssvep_adapter.process_command(command)
        if event is None:
            self.main_window.append_log(f"SSVEP ignored: {command}")
            return
        self.dispatch_event(event)

    def dispatch_mi_result(self, result: dict[str, object], *, timestamp_ms: int | None = None) -> None:
        self.runtime_info["last_mi_raw"] = self._summarize_mi_result(result)
        self.logger.log_raw_input("mi", result)
        event = self.mi_adapter.process_result(result, timestamp_ms=timestamp_ms)
        if event is not None:
            self.dispatch_event(event)

    def _can_start_task(self) -> bool:
        if not self._requires_preflight():
            return True
        return bool(self.runtime_info.get("preflight_ok", False))

    def _force_controller_error(self, message: str) -> list[Effect]:
        effects: list[Effect] = []
        self.controller.context.pending_robot_xy = None
        self.controller.context.last_error = str(message)
        self.controller.context.robot_busy = True
        self.controller._cancel_active_timer(effects)
        self.controller._set_state(TaskState.ERROR, effects)
        return effects

    def _requires_preflight(self) -> bool:
        return self.config.robot_mode in {"fake-remote", "real"}

    def _preflight_ignores_calibration(self) -> bool:
        return self.config.vision_mode in {"fixed_world_slots", "fixed_cyl_slots"}

    def _evaluate_preflight_from_snapshot(self, snapshot: dict[str, object] | None) -> None:
        if not self._requires_preflight():
            self.runtime_info["preflight_ok"] = True
            self.runtime_info["preflight_message"] = "not_required"
            return
        if snapshot is None:
            self.runtime_info["preflight_ok"] = False
            self.runtime_info["preflight_message"] = "status_unavailable"
            return
        state = str(snapshot.get("state") or "")
        busy = bool(snapshot.get("busy"))
        carrying = bool(snapshot.get("carrying", snapshot.get("carrying_target_id") is not None))
        calibration_ready = snapshot.get("calibration_ready")
        self.runtime_info["calibration_ready"] = calibration_ready
        if state not in {"IDLE", "CARRY_READY"}:
            self.runtime_info["preflight_ok"] = False
            self.runtime_info["preflight_message"] = f"state={state}"
            return
        if busy:
            self.runtime_info["preflight_ok"] = False
            self.runtime_info["preflight_message"] = "robot_busy"
            return
        if carrying:
            self.runtime_info["preflight_ok"] = False
            self.runtime_info["preflight_message"] = "robot_carrying_target"
            return
        if not self._preflight_ignores_calibration() and calibration_ready is False:
            self.runtime_info["preflight_ok"] = False
            self.runtime_info["preflight_message"] = "calibration_not_ready"
            return
        self.runtime_info["preflight_ok"] = True
        self.runtime_info["preflight_message"] = "ready"

    def _setup_robot_mode(self) -> None:
        if self.config.robot_mode == "fake":
            if self.simulation_world is None:
                raise RuntimeError("Simulation world is required for fake robot mode.")
            self.fake_robot_server = FakeRobotServer(self.config.robot_host, self.config.robot_port, self.simulation_world)
            self.fake_robot_server.start()
            self._log_runtime(
                "robot",
                f"Fake robot server started on {self.config.robot_host}:{self.config.robot_port} "
                f"(scenario={self.config.scenario_name}, profile={self.config.timing_profile})",
            )
        elif self.config.robot_mode == "fake-remote":
            self._log_runtime(
                "robot",
                f"Using remote fake robot on {self.config.robot_host}:{self.config.robot_port} "
                f"(scenario={self.config.scenario_name}, profile={self.config.timing_profile})",
            )

        try:
            self.robot_client.connect()
            ok = self.robot_client.healthcheck(timeout_sec=self.config.robot_ping_timeout_sec)
            self.runtime_info["robot_connected"] = self.robot_client.is_connected()
            self.runtime_info["robot_health"] = "ok" if ok else "ping_failed"
            if ok:
                self._log_runtime("robot", "Robot healthcheck passed.")
            else:
                self._log_runtime("robot", "Robot healthcheck failed.")
            initial_snapshot = self._fetch_remote_robot_snapshot_direct()
            if initial_snapshot is not None:
                with self._remote_snapshot_lock:
                    self._remote_snapshot_cache = dict(initial_snapshot)
            self._capture_world_snapshot(reason="connect", force=True)
            self._evaluate_preflight_from_snapshot(self._latest_world_snapshot)
        except Exception as error:
            self.runtime_info["robot_connected"] = False
            self.runtime_info["robot_health"] = "connect_failed"
            self.runtime_info["last_robot_error"] = str(error)
            self.runtime_info["preflight_ok"] = False
            self.runtime_info["preflight_message"] = "connect_failed"
            self._log_runtime("robot", f"Robot connect failed: {error}")

    def _setup_vision_mode(self) -> None:
        if self.config.control_sim_enabled and self.config.vision_mode in {"slots", "fake", "fixed_world_slots", "fixed_cyl_slots"}:
            self._publish_control_sim_targets()
            self.runtime_info["vision_health"] = f"{self.config.vision_mode}:{self.config.slot_profile}"
            return

            self.vision_runtime = VisionRuntime(
                self.config,
            targets_callback=lambda targets: self._queue_event(
                Event(source="vision", type="vision_update", value=targets)
            ),
            status_callback=lambda message: self._queue_runtime_status("vision", message),
        )
        self.vision_runtime.start()
        self.runtime_info["vision_health"] = "starting"

    def _setup_brain_sources(self) -> None:
        if self.config.move_source == "mi":
            self.mi_adapter.start()
            self.mi_runtime = MIRuntime(
                self.config,
                result_callback=self.dispatch_mi_result,
                status_callback=lambda message: self._queue_runtime_status("mi", message),
            )
            self.mi_runtime.start()
        else:
            self.mi_adapter.stop()

        if self.config.decision_source == "ssvep":
            self.ssvep_runtime = SSVEPRuntime(
                self.config,
                command_callback=self.dispatch_ssvep_command,
                status_callback=lambda message: self._queue_runtime_status("ssvep", message),
            )
            self.ssvep_runtime.start()

    def _apply_effect(self, effect: Effect) -> None:
        handlers: dict[str, Callable[[Effect], None]] = {
            "start_timer": self._start_timer,
            "cancel_timer": self._cancel_timer,
            "robot_command": self._send_robot_command,
            "state_changed": self._handle_state_changed,
            "log": self._handle_log_effect,
        }
        handler = handlers.get(effect.type)
        if handler is not None:
            handler(effect)

    def _start_timer(self, effect: Effect) -> None:
        timer_id = effect.payload["timer_id"]
        duration_ms = int(round(float(effect.payload["duration_sec"]) * 1000.0))
        self._cancel_timer(Effect("cancel_timer", {"timer_id": timer_id}))
        timer = QTimer(self.main_window)
        timer.setSingleShot(True)
        timer.timeout.connect(
            lambda tid=timer_id: self.dispatch_event(Event(source="system", type="timer_expired", value=tid))
        )
        timer.start(duration_ms)
        self.timers[timer_id] = timer

    def _cancel_timer(self, effect: Effect) -> None:
        timer_id = effect.payload["timer_id"]
        timer = self.timers.pop(timer_id, None)
        if timer is not None:
            timer.stop()
            timer.deleteLater()

    def _send_robot_command(self, effect: Effect) -> None:
        command = str(effect.payload["command"])
        try:
            self.robot_client.send_command(command)
            self.main_window.append_log(f"Robot <= {command}")
        except Exception as error:
            self._handle_runtime_status("robot", f"Robot send failed: {error}")
            self.dispatch_event(Event(source="robot", type="robot_error", value=f"Robot send failed: {error}"))

    def _handle_state_changed(self, effect: Effect) -> None:
        self.main_window.append_log(f"State: {effect.payload['from']} -> {effect.payload['to']}")
        old_state = str(effect.payload.get("from", ""))
        new_state = str(effect.payload.get("to", ""))
        if old_state in {TaskState.S1_MI_MOVE.value, TaskState.S3_MI_CARRY.value} and new_state not in {
            TaskState.S1_MI_MOVE.value,
            TaskState.S3_MI_CARRY.value,
        }:
            self._stop_teleop_motion(send_command=True, reason=f"leave_motion_state:{new_state}")
        if self.config.control_sim_enabled and effect.payload.get("to") == TaskState.S2_TARGET_SELECT.value:
            self._publish_control_sim_targets()

    def _handle_log_effect(self, effect: Effect) -> None:
        self.main_window.append_log(str(effect.payload.get("message", "")))

    def _publish_control_sim_targets(self) -> None:
        targets = self._current_control_selection_targets()
        self.logger.log_vision_snapshot(targets, self.controller.state.value, scenario=self.config.scenario_name)
        self.dispatch_event(Event(source="vision", type="vision_update", value=targets))

    def _update_ssvep_mode(self) -> None:
        if self.controller.state == TaskState.S2_TARGET_SELECT:
            mode = "target_selection"
        elif self.controller.state in {TaskState.S1_DECISION, TaskState.S2_GRAB_CONFIRM, TaskState.S3_DECISION}:
            mode = "binary"
        else:
            mode = "idle"
        self.ssvep_adapter.set_mode(mode)
        if self.ssvep_runtime is not None:
            self.ssvep_runtime.set_mode(mode)
        self.runtime_info["target_frequency_map"] = self._build_target_frequency_map(mode)

    def _build_target_frequency_map(self, mode: str) -> list[tuple[str, object]]:
        if mode == "binary":
            return [
                (f"{self.config.ssvep_freqs[0]:g}Hz", "confirm"),
                (f"{self.config.ssvep_freqs[3]:g}Hz", "cancel"),
            ]
        if mode == "target_selection":
            mapping: list[tuple[str, object]] = []
            for index, target in enumerate(self.controller.context.frozen_targets):
                if index >= len(self.config.ssvep_freqs):
                    break
                mapping.append((f"{self.config.ssvep_freqs[index]:g}Hz", target.id))
            return mapping
        return []

    def _update_runtime_health(self) -> None:
        self.runtime_info["robot_connected"] = self.robot_client.is_connected()
        if self.config.robot_mode in {"fake", "fake-remote", "real"}:
            if self.runtime_info.get("robot_health") not in {"connect_failed", "send_failed"}:
                self.runtime_info["robot_health"] = "ok" if self.robot_client.is_connected() else "disconnected"
        if self.vision_runtime is not None:
            self.runtime_info["vision_health"] = self.vision_runtime.healthcheck()["running"]
        elif self.config.control_sim_enabled and self.config.vision_mode in {"slots", "fake", "fixed_world_slots", "fixed_cyl_slots"}:
            self.runtime_info["vision_health"] = f"{self.config.vision_mode}:{self.config.slot_profile}"

    def _refresh_view(self) -> None:
        self._request_remote_snapshot()
        self._capture_world_snapshot(reason="refresh")
        self.main_window.update_snapshot(
            self.controller.snapshot(),
            dict(self.runtime_info),
            simulation_snapshot=self._latest_world_snapshot,
        )

    def _on_key_pressed(self, token: str) -> None:
        self.logger.log_raw_input("sim_key", token)
        if self._is_move_token(token):
            self._handle_move_key_pressed(token)
            return
        for event in self.sim_input.handle_key_token(token):
            self.dispatch_event(event)

    def _on_key_released(self, token: str) -> None:
        self.logger.log_raw_input("sim_key_release", token)
        if self._is_move_token(token):
            self._handle_move_key_released(token)

    def _on_abort_requested(self) -> None:
        self._stop_teleop_motion(send_command=False, reason="abort_button")
        try:
            self.robot_client.send_command("ABORT")
            self.main_window.append_log("Robot <= ABORT")
        except Exception as error:
            self._handle_runtime_status("robot", f"Robot ABORT failed: {error}")
            self.dispatch_event(Event(source="robot", type="robot_error", value=f"Robot ABORT failed: {error}"))

    def _on_reset_requested(self) -> None:
        self._stop_teleop_motion(send_command=False, reason="reset_button")
        try:
            self.robot_client.send_command("RESET")
            self.main_window.append_log("Robot <= RESET")
        except Exception as error:
            self._handle_runtime_status("robot", f"Robot RESET failed: {error}")
            self.dispatch_event(Event(source="robot", type="robot_error", value=f"Robot RESET failed: {error}"))

    def _start_ui_refresh_timer(self) -> None:
        timer = QTimer(self.main_window)
        timer.timeout.connect(self._refresh_view)
        timer.start(self.config.ui_refresh_interval_ms)
        self.timers["ui-refresh"] = timer

    def _start_teleop_timer(self) -> None:
        timer = QTimer(self.main_window)
        timer.timeout.connect(self._pump_teleop_command)
        timer.start(int(self.config.teleop_repeat_interval_ms))
        self.timers["teleop-step"] = timer

    def _handle_runtime_status(self, component: str, message: str) -> None:
        self._log_runtime(component, message)
        if component == "vision":
            self.runtime_info["vision_health"] = message
        elif component == "robot":
            self.runtime_info["robot_health"] = "send_failed" if "send failed" in message.lower() else message
        self._refresh_view()

    def _log_runtime(self, component: str, message: str) -> None:
        self.logger.log_runtime_status(component, message)
        self.main_window.append_log(f"[{component}] {message}")

    def _is_move_token(self, token: str) -> bool:
        return str(token).strip().lower() in {"a", "d", "w", "s", "left", "right", "up", "down"}

    def _is_teleop_enabled(self) -> bool:
        return self.config.move_source == "sim"

    def _in_motion_state(self) -> bool:
        return self.controller.state in {TaskState.S1_MI_MOVE, TaskState.S3_MI_CARRY}

    def _handle_move_key_pressed(self, token: str) -> None:
        if not self._is_teleop_enabled():
            return
        if not self._in_motion_state():
            return
        self._pressed_move_tokens.add(str(token).strip().lower())
        self._pump_teleop_command()

    def _handle_move_key_released(self, token: str) -> None:
        normalized = str(token).strip().lower()
        self._pressed_move_tokens.discard(normalized)
        if not self._pressed_move_tokens:
            self._stop_teleop_motion(send_command=False, reason="key_release")

    def _compute_teleop_delta(self) -> tuple[float, float]:
        dtheta = 0.0
        dr = 0.0
        pressed = self._pressed_move_tokens
        if "a" in pressed or "left" in pressed:
            dtheta -= float(self.config.teleop_theta_step_deg)
        if "d" in pressed or "right" in pressed:
            dtheta += float(self.config.teleop_theta_step_deg)
        if "w" in pressed or "up" in pressed:
            dr += float(self.config.teleop_radius_step_mm)
        if "s" in pressed or "down" in pressed:
            dr -= float(self.config.teleop_radius_step_mm)
        if dtheta == 0.0 and dr == 0.0:
            return (0.0, 0.0)
        return (dtheta, dr)

    def _pump_teleop_command(self) -> None:
        if not self._is_teleop_enabled():
            return
        if not self._in_motion_state():
            self._stop_teleop_motion(send_command=False, reason="not_in_motion_state")
            return
        if self.controller.context.pending_robot_xy is not None:
            return
        if self.controller.context.robot_busy:
            return
        dtheta, dr = self._compute_teleop_delta()
        if dtheta == 0.0 and dr == 0.0:
            return
        self.dispatch_event(Event(source="sim", type="move", value={"dtheta": dtheta, "dr": dr}))

    def _stop_teleop_motion(self, *, send_command: bool, reason: str) -> None:
        self._pressed_move_tokens.clear()

    def _capture_world_snapshot(self, *, reason: str, force: bool = False) -> None:
        snapshot = None
        preflight_snapshot = None
        if self.simulation_world is not None:
            snapshot = self.simulation_world.snapshot()
            preflight_snapshot = snapshot
        else:
            local_snapshot = self._build_control_scene_snapshot()
            remote_snapshot = self._fetch_remote_robot_snapshot()
            preflight_snapshot = remote_snapshot
            if remote_snapshot is not None:
                snapshot = self._merge_scene_snapshots(local_snapshot, remote_snapshot)
            else:
                snapshot = local_snapshot
        if snapshot is None:
            self._latest_world_snapshot = None
            return
        self._sync_controller_from_snapshot(snapshot)
        self._latest_world_snapshot = snapshot
        self._evaluate_preflight_from_snapshot(preflight_snapshot)
        revision = int(snapshot.get("revision", 0))
        if force or revision != self._last_logged_world_revision:
            self.logger.log_world_snapshot(snapshot, reason=reason)
            self._last_logged_world_revision = revision

    def _fetch_remote_robot_snapshot(self) -> dict[str, object] | None:
        if self.config.robot_mode not in {"fake-remote", "real"}:
            return None
        with self._remote_snapshot_lock:
            if self._remote_snapshot_cache is None:
                return None
            return dict(self._remote_snapshot_cache)

    def _fetch_remote_robot_snapshot_direct(self) -> dict[str, object] | None:
        if self.config.robot_mode not in {"fake-remote", "real"}:
            return None
        try:
            return fetch_robot_status(
                self.config.robot_host,
                self.config.robot_port,
                timeout_sec=max(self.config.robot_timeout_sec, 0.2),
            )
        except Exception:
            return None

    def _merge_scene_snapshots(
        self,
        local_snapshot: dict[str, object] | None,
        remote_snapshot: dict[str, object],
    ) -> dict[str, object]:
        if local_snapshot is None:
            return dict(remote_snapshot)
        merged = dict(local_snapshot)
        merged["revision"] = int(remote_snapshot.get("revision", merged.get("revision", 0)))
        for key in (
            "robot_xy",
            "robot_z",
            "robot_cyl",
            "home_pose",
            "limits_x",
            "limits_y",
            "limits_cyl",
            "approach_z",
            "pick_z",
            "carry_z",
            "auto_z_enabled",
            "auto_z_current",
            "control_kernel",
            "ik_valid",
            "validation_error",
            "busy",
            "busy_action",
            "state",
            "action_phase",
            "last_error",
            "last_error_code",
            "calibration_ready",
            "carrying",
        ):
            if key in remote_snapshot:
                merged[key] = remote_snapshot[key]
        if "position_xyz" in remote_snapshot and "robot_xy" not in remote_snapshot:
            position = tuple(remote_snapshot["position_xyz"])
            merged["robot_xy"] = position[:2]
            merged["robot_z"] = position[2]
        if "carrying" in remote_snapshot:
            merged["carrying_target_id"] = merged.get("carrying_target_id") if remote_snapshot["carrying"] else None
        for key in ("pick_slots", "place_slots"):
            if key in remote_snapshot and remote_snapshot[key]:
                merged[key] = remote_snapshot[key]
        return merged

    def _sync_controller_from_snapshot(self, snapshot: dict[str, object]) -> None:
        robot_xy = snapshot.get("robot_xy")
        if isinstance(robot_xy, (list, tuple)) and len(robot_xy) == 2:
            self.controller.context.robot_xy = (float(robot_xy[0]), float(robot_xy[1]))
        robot_cyl = snapshot.get("robot_cyl")
        if isinstance(robot_cyl, dict):
            self.controller.context.robot_cyl = (
                float(robot_cyl.get("theta_deg", self.controller.context.robot_cyl[0])),
                float(robot_cyl.get("radius_mm", self.controller.context.robot_cyl[1])),
                float(robot_cyl.get("z_mm", self.controller.context.robot_cyl[2])),
            )
        elif isinstance(robot_xy, (list, tuple)) and len(robot_xy) == 2:
            theta_deg, radius_mm, z_mm = cartesian_to_cylindrical(
                float(robot_xy[0]),
                float(robot_xy[1]),
                float(snapshot.get("robot_z", self.config.robot_carry_z)),
            )
            self.controller.context.robot_cyl = (theta_deg, radius_mm, z_mm)
        if "busy" in snapshot:
            self.controller.context.robot_busy = bool(snapshot["busy"])
        if "state" in snapshot:
            self.controller.context.robot_execution_state = str(snapshot["state"])
        if "carrying" in snapshot:
            self.controller.context.carrying = bool(snapshot["carrying"])
        if "auto_z_current" in snapshot and snapshot["auto_z_current"] is not None:
            self.controller.context.robot_auto_z = float(snapshot["auto_z_current"])
        self.runtime_info["calibration_ready"] = snapshot.get("calibration_ready")
        self.runtime_info["robot_cyl"] = snapshot.get("robot_cyl")
        self.runtime_info["limits_cyl"] = snapshot.get("limits_cyl")
        self.runtime_info["auto_z_current"] = snapshot.get("auto_z_current")
        self.runtime_info["control_kernel"] = snapshot.get("control_kernel")
        if snapshot.get("state") == "ERROR":
            self.controller.context.last_error = str(snapshot.get("last_error") or self.controller.context.last_error)

    def _queue_event(self, event: Event) -> None:
        self._bridge.event_received.emit(event)

    def _queue_runtime_status(self, component: str, message: str) -> None:
        self._bridge.runtime_status_received.emit(str(component), str(message))

    def _request_remote_snapshot(self) -> None:
        if self.config.robot_mode not in {"fake-remote", "real"}:
            return
        with self._remote_snapshot_lock:
            if self._remote_snapshot_inflight:
                return
            self._remote_snapshot_inflight = True

        def worker() -> None:
            snapshot: dict[str, object] | None = None
            try:
                snapshot = self._fetch_remote_robot_snapshot_direct()
            except Exception:
                snapshot = None
            self._bridge.remote_snapshot_received.emit(snapshot)

        threading.Thread(target=worker, name="remote-status-poll", daemon=True).start()

    def _on_remote_snapshot_received(self, snapshot: object) -> None:
        payload = snapshot if isinstance(snapshot, dict) else None
        with self._remote_snapshot_lock:
            self._remote_snapshot_cache = None if payload is None else dict(payload)
            self._remote_snapshot_inflight = False

    @staticmethod
    def _summarize_mi_result(result: dict[str, object]) -> str:
        prediction = result.get("stable_prediction_display_name") or result.get("prediction_display_name") or "--"
        confidence = result.get("stable_confidence")
        if confidence is None:
            confidence = result.get("confidence")
        if confidence is None:
            return str(prediction)
        return f"{prediction} ({float(confidence):.3f})"

    def _reset_control_scene_state(self) -> None:
        if self.slot_catalog is None:
            self._scene_pick_slots = {}
            self._scene_place_slots = {}
            self._scene_carrying_target_id = None
            self._scene_last_error = None
            return

        visible_pick_slot_ids = {slot.slot_id for slot in self.slot_catalog.list_pick_slots(source=self._pick_slot_source())}
        if self.config.scenario_name == "empty_roi":
            visible_pick_slot_ids = set()
        elif self.config.scenario_name == "sparse_targets":
            visible_pick_slot_ids = {1}

        self._scene_pick_slots = {}
        for slot in self.slot_catalog.list_pick_slots(source=self._pick_slot_source()):
            occupied = slot.slot_id in visible_pick_slot_ids
            self._scene_pick_slots[slot.slot_id] = {
                "slot": slot,
                "occupied": occupied,
                "selected": False,
                "object_id": slot.slot_id if occupied else None,
            }

        self._scene_place_slots = {
            slot.slot_id: {"slot": slot, "object_id": None} for slot in self.slot_catalog.list_place_slots()
        }
        self._scene_carrying_target_id = None
        self._scene_last_error = None
        self._scene_revision += 1

    def _current_control_selection_targets(self) -> list[object]:
        if self.slot_catalog is None:
            return []
        targets: list[object] = []
        for index, slot in enumerate(self.slot_catalog.list_pick_slots(source=self._pick_slot_source())):
            slot_state = self._scene_pick_slots.get(slot.slot_id)
            if not slot_state or not slot_state["occupied"]:
                continue
            targets.append(
                slot.to_target(
                    confidence=max(0.7, 1.0 - index * 0.05),
                    command_mode=self._target_command_mode(),
                )
            )
        return targets

    def _pick_slot_source(self) -> str:
        if self.config.vision_mode in {"fixed_world_slots", "fixed_cyl_slots"}:
            return "hardware"
        return "sim"

    def _target_command_mode(self) -> str:
        if self.config.vision_mode == "fixed_world_slots":
            return "world"
        if self.config.vision_mode == "fixed_cyl_slots":
            return "cyl"
        return "pixel"

    def _update_control_scene_from_event(self, event: Event, *, selected_before: int | None) -> None:
        if self.slot_catalog is None:
            return
        if event.type == "robot_error":
            self._scene_last_error = str(event.value)
            self._scene_revision += 1
            return
        if event.type == "robot_busy":
            self._scene_last_error = "BUSY"
            self._scene_revision += 1
            return
        if event.type == "target_selected":
            selected_id = self.controller.context.selected_target_id
            for slot_state in self._scene_pick_slots.values():
                slot_state["selected"] = slot_state["slot"].slot_id == selected_id
            self._scene_revision += 1
            return
        if event.type == "decision_cancel":
            for slot_state in self._scene_pick_slots.values():
                slot_state["selected"] = False
            self._scene_revision += 1
            return
        if event.type == "robot_ack":
            ack = str(event.value or "").strip().upper()
            if ack == "PICK_DONE" and selected_before is not None and selected_before in self._scene_pick_slots:
                self._scene_pick_slots[selected_before]["occupied"] = False
                self._scene_pick_slots[selected_before]["selected"] = False
                self._scene_pick_slots[selected_before]["object_id"] = None
                self._scene_carrying_target_id = selected_before
                self._scene_last_error = None
                self._scene_revision += 1
                self._publish_control_sim_targets()
                return
            if ack == "PLACE_DONE":
                if self._scene_carrying_target_id is not None:
                    place_slot = self.slot_catalog.nearest_place_slot(self.controller.context.robot_xy)
                    if place_slot is not None:
                        self._scene_place_slots[place_slot.slot_id]["object_id"] = self._scene_carrying_target_id
                self._scene_carrying_target_id = None
                self._scene_last_error = None
                self._scene_revision += 1
                self._publish_control_sim_targets()
                return

    def _build_control_scene_snapshot(self) -> dict[str, object] | None:
        if self.slot_catalog is None:
            return None
        return {
            "revision": self._scene_revision,
            "scenario_name": self.config.scenario_name,
            "robot_xy": self.controller.context.robot_xy,
            "robot_z": self.config.robot_pick_z if self.controller.state in {TaskState.S2_PICKING, TaskState.S3_PLACING} else self.config.robot_carry_z,
            "home_pose": (
                self.config.robot_start_xy[0],
                self.config.robot_start_xy[1],
                self.config.robot_carry_z,
            ),
            "limits_x": self.config.robot_limits_x,
            "limits_y": self.config.robot_limits_y,
            "robot_cyl": {
                "theta_deg": self.controller.context.robot_cyl[0],
                "radius_mm": self.controller.context.robot_cyl[1],
                "z_mm": self.controller.context.robot_cyl[2],
            },
            "limits_cyl": {
                "theta_deg": self.config.robot_theta_limits_deg,
                "radius_mm": self.config.robot_radius_limits_mm,
                "z_mm": self.config.robot_height_limits_mm,
            },
            "travel_z": self.config.robot_travel_z,
            "approach_z": self.config.robot_approach_z,
            "pick_z": self.config.robot_pick_z,
            "carry_z": self.config.robot_carry_z,
            "auto_z_enabled": True,
            "auto_z_current": self.controller.context.robot_auto_z,
            "control_kernel": self.runtime_info.get("control_kernel") or "cylindrical_kernel",
            "busy_action": self._scene_busy_action(),
            "busy": self._scene_busy_action() is not None,
            "state": self.controller.state.value,
            "action_phase": self.controller.state.value,
            "carrying": self._scene_carrying_target_id is not None,
            "calibration_ready": False if self.config.vision_mode == "fixed_world_slots" else True,
            "carrying_target_id": self._scene_carrying_target_id,
            "last_ack": self.runtime_info["last_robot_ack"],
            "last_error": self._scene_last_error or self.runtime_info["last_robot_error"],
            "pick_slots": [
                {
                    **slot_state["slot"].to_dict(),
                    "occupied": slot_state["occupied"],
                    "selected": slot_state["selected"],
                    "object_id": slot_state["object_id"],
                }
                for slot_state in self._scene_pick_slots.values()
            ],
            "place_slots": [
                {
                    **slot_state["slot"].to_dict(),
                    "object_id": slot_state["object_id"],
                    "occupied": slot_state["object_id"] is not None,
                }
                for slot_state in self._scene_place_slots.values()
            ],
        }

    def _scene_busy_action(self) -> str | None:
        if self.controller.state == TaskState.S2_PICKING:
            return "pick"
        if self.controller.state == TaskState.S3_PLACING:
            return "place"
        if self.controller.state in {TaskState.S1_MI_MOVE, TaskState.S3_MI_CARRY}:
            return "move"
        return None


def build_config_from_args(args: argparse.Namespace) -> AppConfig:
    config = AppConfig(
        control_sim_enabled=not args.no_simulation,
        sim_process_mode="dual",
        slot_profile=args.slot_profile,
        simulation_enabled=not args.no_simulation,
        timing_profile=args.timing_profile,
        scenario_name=args.scenario_name,
        robot_mode=args.robot_mode,
        vision_mode=args.vision_mode,
        move_source=args.move_source,
        decision_source=args.decision_source,
        robot_host=args.robot_host,
        robot_port=args.robot_port,
        vision_stream_url=args.vision_stream_url,
    )
    stage_motion_sec = getattr(args, "stage_motion_sec", None)
    continue_motion_sec = getattr(args, "continue_motion_sec", None)
    if stage_motion_sec is not None:
        config = replace(config, stage_motion_sec=float(stage_motion_sec))
    if continue_motion_sec is not None:
        config = replace(config, continue_motion_sec=float(continue_motion_sec))
    return config.resolved()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hybrid Controller v1")
    parser.add_argument("--robot-mode", choices=("fake", "fake-remote", "real"), default="fake")
    parser.add_argument(
        "--vision-mode",
        choices=("slots", "fake", "real", "fixed_world_slots", "fixed_cyl_slots"),
        default="slots",
    )
    parser.add_argument("--move-source", choices=("sim", "mi"), default="sim")
    parser.add_argument("--decision-source", choices=("sim", "ssvep"), default="sim")
    parser.add_argument("--timing-profile", choices=("formal", "fast"), default="formal")
    parser.add_argument("--scenario-name", default="basic")
    parser.add_argument("--slot-profile", default="default")
    parser.add_argument("--no-simulation", action="store_true", help="Disable simulation world wiring.")
    parser.add_argument("--robot-host", default=AppConfig.robot_host)
    parser.add_argument("--robot-port", type=int, default=AppConfig.robot_port)
    parser.add_argument("--vision-stream-url", default="")
    parser.add_argument("--stage-motion-sec", type=float, default=None)
    parser.add_argument("--continue-motion-sec", type=float, default=None)
    parser.add_argument("--smoke-test-ms", type=int, default=0, help="Auto quit after N ms for smoke tests.")
    args = parser.parse_args(argv)

    config = build_config_from_args(args)
    qt_app = QApplication(sys.argv if argv is None else argv)
    runtime = HybridControllerApplication(config)
    runtime.main_window.show()
    if args.smoke_test_ms > 0:
        QTimer.singleShot(args.smoke_test_ms, qt_app.quit)
    exit_code = 0
    try:
        exit_code = qt_app.exec_()
    finally:
        runtime.shutdown()
    return int(exit_code)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
