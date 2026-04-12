from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import socket
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

from hybrid_controller.adapters.control_sim_slots import ControlSimSlotCatalog
from hybrid_controller.adapters.input_orchestrator import InputOrchestrator
from hybrid_controller.adapters.input_provider import InputHealth
from hybrid_controller.adapters.mi_input import MiInputProvider
from hybrid_controller.adapters.remote_snapshot_poller import RemoteSnapshotPoller
from hybrid_controller.adapters.rosbridge_client import RosServiceResult, RosbridgeClient
from hybrid_controller.adapters.robot_client import RobotClient, fetch_robot_status
from hybrid_controller.adapters.sim_input import SimInputAdapter
from hybrid_controller.adapters.ssvep_adapter import SSVEPAdapter
from hybrid_controller.adapters.teleop_fallback import RosTeleopFallbackController
from hybrid_controller.adapters.teleop_ros_channel import RosTeleopPublishPlanner
from hybrid_controller.app_projection import build_ui_snapshot, sync_coordinator_states_from_runtime_info
from hybrid_controller.app_robot_commands import (
    build_catalog_pick_command,
    build_pick_command_from_mode_and_point,
    build_pick_command_from_slot_payload,
    build_pick_command_from_target,
    extract_command_opcode,
    rewrite_pick_command_with_bias,
    ros_command_requires_ros_route,
)
from hybrid_controller.cylindrical import cartesian_to_cylindrical, cylindrical_to_cartesian
from hybrid_controller.coordinators import RobotCoordinator, SSVEPCoordinator, UiCoordinator, VisionCoordinator
from hybrid_controller.config import AppConfig
from hybrid_controller.controller.events import Effect, Event
from hybrid_controller.controller.state_machine import TaskState
from hybrid_controller.controller.task_controller import TaskController
from hybrid_controller.observability.event_logger import EventLogger
from hybrid_controller.runtime_state import (
    RobotSnapshotEnvelope,
    RuntimeAction,
    RuntimeInfoCompat,
    RuntimeStore,
)
from hybrid_controller.ssvep.runtime import SSVEPRuntime
from hybrid_controller.vision.processing import packet_to_targets
from hybrid_controller.vision.runtime import VisionRuntime
from hybrid_controller.vision.target_resolver import resolve_vision_packet

try:
    from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
    from PyQt5.QtWidgets import QApplication, QFileDialog
except ImportError as error:  # pragma: no cover - UI import guard
    raise RuntimeError("PyQt5 is required to run hybrid_controller.app") from error

from hybrid_controller.ui.main_window import MainWindow

_UNSET = object()


class _RuntimeSignalBridge(QObject):
    event_received = pyqtSignal(object)
    runtime_status_received = pyqtSignal(str, str)
    robot_start_finished = pyqtSignal(bool, str)
    ssvep_state_received = pyqtSignal(object)
    remote_snapshot_received = pyqtSignal(object)
    vision_packet_received = pyqtSignal(object)
    vision_frame_received = pyqtSignal(object)


class HybridControllerApplication:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.controller = TaskController(config)
        self.logger = EventLogger(config.event_log_path)
        self.sim_input = SimInputAdapter(
            move_source=config.move_source,
            decision_source=config.decision_source,
            ssvep_keyboard_debug_enabled=config.ssvep_keyboard_debug_enabled,
        )
        self.ssvep_adapter = SSVEPAdapter(config.ssvep_freqs)
        self.mi_provider = MiInputProvider(
            backend=str(config.mi_backend),
            poll_interval_ms=int(config.mi_poll_interval_ms),
        )
        self.input_orchestrator = InputOrchestrator(
            sim_input=self.sim_input,
            ssvep_adapter=self.ssvep_adapter,
            move_source=str(config.move_source),
            decision_source=str(config.decision_source),
            ssvep_keyboard_debug_enabled=bool(config.ssvep_keyboard_debug_enabled),
            mi_provider=self.mi_provider,
            mi_enabled=bool(config.mi_enabled),
            mi_command_cooldown_ms=int(config.mi_command_cooldown_ms),
        )
        self.main_window = MainWindow()
        self.slot_catalog = ControlSimSlotCatalog(config) if config.control_sim_enabled else None
        self.vision_runtime: VisionRuntime | None = None
        self.ssvep_runtime: SSVEPRuntime | None = None
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
        self._remote_snapshot_envelope: RobotSnapshotEnvelope | None = None
        self._remote_snapshot_poller: RemoteSnapshotPoller | None = None
        self._latest_vision_packet: dict[str, object] | None = None
        self._latest_vision_frame = None
        self._vision_frame_lock = threading.Lock()
        self._vision_frame_staging = None
        self._vision_frame_signal_pending = False
        self._last_teleop_warn_ts = 0.0
        self._last_mi_health_signature: tuple[object, ...] | None = None
        self._last_mi_status_emit_ts = 0.0
        self._pick_cyl_radius_bias_mm = float(self.config.pick_cyl_radius_bias_mm)
        self._pick_cyl_theta_bias_deg = float(self.config.pick_cyl_theta_bias_deg)
        self._pick_tuning_defaults = self._build_pick_tuning_defaults()
        self._pick_tuning_state = self._load_pick_tuning_profile()
        self._pick_tuning_local_dirty = False
        self._active_pick_trace: dict[str, object] | None = None
        self._teleop_ros_planner = RosTeleopPublishPlanner(
            keepalive_interval_sec=max(float(self.config.teleop_ros_keepalive_interval_ms) / 1000.0, 0.02)
        )
        self._teleop_fallback = RosTeleopFallbackController(
            stall_sec=0.45,
            step_sec=0.35,
            interval_sec=0.35,
        )
        self._teleop_cmd_seq = 0
        self._ros_last_connected = False
        self._ros_reconnect_attempt = 0
        self._ros_reconnect_next_ts = 0.0
        self._ros_connected_since_ts = 0.0
        self._last_auto_robot_start_ts = 0.0
        self._last_ros_runtime_unavailable_log_ts = 0.0
        self._ros_stale_detection_count = 0
        self._last_ros_disconnect_ts = 0.0
        self._auto_start_blocked = False
        self._auto_start_block_reason = ""
        self._next_pending_command_seq = 1
        self._pending_command: dict[str, object] | None = None
        self.ros_client: RosbridgeClient | None = None
        self._shutdown_started = False
        self._reset_control_scene_state()
        self._bridge = _RuntimeSignalBridge()
        self._bridge.event_received.connect(self.dispatch_event)
        self._bridge.runtime_status_received.connect(self._handle_runtime_status)
        self._bridge.robot_start_finished.connect(self._on_robot_start_finished)
        self._bridge.ssvep_state_received.connect(self._on_ssvep_state_received)
        self._bridge.remote_snapshot_received.connect(self._on_remote_snapshot_received)
        self._bridge.vision_packet_received.connect(self._on_vision_packet_received)
        self._bridge.vision_frame_received.connect(self._on_vision_frame_received)
        self._refresh_dirty = False
        self._refresh_coalesce_timer = QTimer(self.main_window)
        self._refresh_coalesce_timer.setSingleShot(True)
        self._refresh_coalesce_timer.timeout.connect(self._refresh_panels)
        self.robot_coordinator = RobotCoordinator(config)
        self.vision_coordinator = VisionCoordinator(config)
        self.ssvep_coordinator = SSVEPCoordinator(config)
        self.ui_coordinator = UiCoordinator()
        self.robot_coordinator.configure_sender(self._send_robot_text_command)

        runtime_seed: dict[str, object] = {
            "simulation_enabled": config.simulation_enabled,
            "timing_profile": config.timing_profile,
            "scenario_name": config.scenario_name,
            "move_source": config.move_source,
            "decision_source": config.decision_source,
            "mi_backend": config.mi_backend,
            "mi_enabled": bool(config.mi_enabled),
            "mi_status": "inactive",
            "mi_connected": False,
            "mi_running": False,
            "mi_last_error": "--",
            "robot_mode": config.robot_mode,
            "robot_transport": config.robot_transport,
            "vision_mode": config.vision_mode,
            "robot_connected": False,
            "robot_start_active": False,
            "robot_health": "unknown",
            "vision_health": "unknown",
            "vision_mapping_mode": config.vision_mapping_mode,
            "vision_invalid_reason": "--",
            "vision_snapshot_age_ms": float("inf"),
            "state_age_ms": float("inf"),
            "vision_last_resolved_base_xy": None,
            "vision_last_resolved_cyl": None,
            "last_robot_ack": "--",
            "last_robot_error": "--",
            "last_ssvep_raw": "--",
            "target_frequency_map": [],
            "preflight_ok": True,
            "preflight_message": "not_required",
            "calibration_ready": None,
            "robot_cyl": None,
            "limits_cyl": None,
            "auto_z_current": None,
            "control_kernel": "cylindrical_kernel",
            "ssvep_runtime_status": "stopped",
            "ssvep_running": False,
            "ssvep_stim_enabled": False,
            "ssvep_busy": False,
            "ssvep_connected": False,
            "ssvep_connect_active": False,
            "ssvep_pretrain_active": False,
            "ssvep_online_active": False,
            "ssvep_profile_path": str(config.ssvep_current_profile_path),
            "ssvep_profile_source": "fallback",
            "ssvep_last_pretrain_time": "--",
            "ssvep_latest_profile_path": "--",
            "ssvep_profile_count": 0,
            "ssvep_available_profiles": (),
            "ssvep_allow_fallback_profile": config.ssvep_allow_fallback_profile,
            "ssvep_status_hint": "当前没有已训练 Profile，可先预训练，或直接用默认 fallback 启动。",
            "ssvep_mode": "idle",
            "ssvep_last_state": "--",
            "ssvep_last_selected_freq": "--",
            "ssvep_last_margin": "--",
            "ssvep_last_ratio": "--",
            "ssvep_last_stable_windows": "--",
            "ssvep_last_error": "--",
            "ssvep_model_name": config.ssvep_model_name,
            "ssvep_debug_keyboard": config.ssvep_keyboard_debug_enabled,
            "pick_tuning": dict(self._pick_tuning_state),
            "post_pick_settle_z": self._pick_tuning_state.get("z_carry_floor_mm"),
            "pending_command": "--",
            "pending_command_seq": 0,
            "pending_command_deadline_ts": 0.0,
            "release_mode_effective": "--",
        }
        self.runtime_store = RuntimeStore.from_config(config)
        self.runtime_store.dispatch(RuntimeAction.update(runtime_seed))
        self.runtime_info: RuntimeInfoCompat = RuntimeInfoCompat(self.runtime_store)

        self.robot_client = RobotClient(
            config.robot_host,
            config.robot_port,
            event_callback=self._queue_event,
            timeout_sec=config.robot_timeout_sec,
            reconnect_delay_sec=config.robot_reconnect_delay_sec,
        )

        self.main_window.key_pressed.connect(self._on_key_pressed)
        self.main_window.key_released.connect(self._on_key_released)
        self.main_window.robot_start_requested.connect(self._on_robot_start_requested)
        self.main_window.robot_connect_requested.connect(self._on_robot_connect_requested)
        self.main_window.abort_requested.connect(self._on_abort_requested)
        self.main_window.reset_requested.connect(self._on_reset_requested)
        self.main_window.ssvep_connect_requested.connect(self._on_ssvep_connect_requested)
        self.main_window.ssvep_pretrain_requested.connect(self._on_ssvep_pretrain_requested)
        self.main_window.ssvep_load_profile_requested.connect(self._on_ssvep_load_profile_requested)
        self.main_window.ssvep_open_profile_dir_requested.connect(self._on_ssvep_open_profile_dir_requested)
        self.main_window.ssvep_stim_toggled.connect(self._on_ssvep_stim_toggled)
        self.main_window.ssvep_start_requested.connect(self._on_ssvep_start_requested)
        self.main_window.ssvep_stop_requested.connect(self._on_ssvep_stop_requested)
        self.main_window.manual_pick_slot_requested.connect(self._on_manual_pick_slot_requested)
        self.main_window.manual_place_requested.connect(self._on_manual_place_requested)
        self.main_window.pick_radius_bias_delta_requested.connect(self._on_pick_radius_bias_delta_requested)
        self.main_window.pick_bias_reset_requested.connect(self._on_pick_bias_reset_requested)
        self.main_window.pick_theta_bias_delta_requested.connect(self._on_pick_theta_bias_delta_requested)
        self.main_window.pick_theta_bias_reset_requested.connect(self._on_pick_theta_bias_reset_requested)
        self.main_window.pick_tuning_delta_requested.connect(self._on_pick_tuning_delta_requested)
        self.main_window.pick_release_mode_toggle_requested.connect(self._on_pick_release_mode_toggle_requested)
        self.main_window.pick_tuning_apply_requested.connect(self._on_pick_tuning_apply_requested)
        self.main_window.pick_tuning_reset_requested.connect(self._on_pick_tuning_reset_requested)
        self.main_window.pick_tuning_save_requested.connect(self._on_pick_tuning_save_requested)
        self.main_window.update_pick_bias_display(self._pick_cyl_radius_bias_mm, self._pick_cyl_theta_bias_deg)
        self.main_window.update_pick_tuning_display(self._pick_tuning_state)
        self._report_runtime_environment()
        self._start_ui_refresh_timer()
        self._start_teleop_timer()
        self._setup_robot_mode()
        self._start_remote_snapshot_poller()
        self._request_remote_snapshot()
        self._setup_vision_mode()
        self._setup_brain_sources()
        self._update_ssvep_mode()
        self._capture_world_snapshot(reason="startup", force=True)
        self._refresh_view()

    def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True

        try:
            self.main_window.shutdown()
            self.main_window.close()
            QApplication.processEvents()
        except Exception:
            pass

        for timer in list(self.timers.values()):
            timer.stop()
            timer.deleteLater()
        self.timers.clear()

        try:
            self.input_orchestrator.shutdown()
        except Exception:
            pass

        if self.vision_runtime is not None:
            self.vision_runtime.stop()
            self.vision_runtime = None
        if self.ssvep_runtime is not None:
            self.ssvep_runtime.stop()
            self.ssvep_runtime = None

        if self._remote_snapshot_poller is not None:
            self._remote_snapshot_poller.stop()
            self._remote_snapshot_poller = None

        if self.ros_client is not None:
            self.ros_client.close()
            self.ros_client = None
        self.robot_client.close()
        self.logger.shutdown()

    def dispatch_event(self, event: Event) -> None:
        if event.type == "start_task" and not self._can_start_task():
            self.logger.log_event(event, self.controller.state.value)
            self.main_window.append_log(
                f"Task start blocked by preflight: {self._rt_get('preflight_message', 'unknown')}"
            )
            self._update_runtime_health()
            self._refresh_view()
            return
        selected_before = self.controller.context.selected_target_id
        self.logger.log_event(event, self.controller.state.value)
        controller_event = event
        self._resolve_pending_command_from_event(event)
        if event.source == "robot" and event.type == "robot_ack":
            self._rt_set("last_robot_ack", str(event.value))
            ack = str(event.value or "").strip().upper()
            if ack == "PICK_DONE":
                self._finish_pick_trace(response=f"ACK {ack}")
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
            self._rt_set("last_robot_error", str(event.value))
            self._finish_pick_trace(response=f"ERR {event.value}")
        if event.source == "robot" and event.type == "robot_disconnected":
            self._last_ros_disconnect_ts = time.monotonic()
            self._stop_teleop_motion(send_command=False, reason="robot_disconnected")
            self._rt_update(
                {
                    "robot_connected": False,
                    "robot_health": "reconnecting",
                }
            )
            controller_event = None
        if event.source == "robot" and event.type == "robot_busy":
            self._finish_pick_trace(response="BUSY")
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
        self._rt_set("last_ssvep_raw", str(command))
        self.logger.log_raw_input("ssvep", command)
        events = self.input_orchestrator.handle_ssvep_command(command)
        if not events:
            self.main_window.append_log(f"SSVEP ignored: {command}")
            return
        for event in events:
            self.dispatch_event(event)

    def _can_start_task(self) -> bool:
        if not self._requires_preflight():
            return True
        return bool(self._rt_get("preflight_ok", False))

    def _force_controller_error(self, message: str) -> list[Effect]:
        effects: list[Effect] = []
        self.controller.context.pending_robot_xy = None
        self.controller.context.last_error = str(message)
        self.controller.context.robot_busy = True
        self.controller._cancel_active_timer(effects)
        self.controller._set_state(TaskState.ERROR, effects)
        return effects

    def _requires_preflight(self) -> bool:
        return self.config.robot_mode == "real"

    def _rt_get(self, key: str, default: object = None) -> object:
        store = getattr(self, "runtime_store", None)
        if isinstance(store, RuntimeStore):
            return store.get_value(str(key), default)
        runtime_info = getattr(self, "runtime_info", None)
        if isinstance(runtime_info, dict):
            return runtime_info.get(str(key), default)
        getter = getattr(runtime_info, "get", None)
        if callable(getter):
            try:
                return getter(str(key), default)
            except Exception:
                return default
        return default

    def _rt_set(self, key: str, value: object) -> None:
        store = getattr(self, "runtime_store", None)
        if isinstance(store, RuntimeStore):
            store.dispatch(RuntimeAction.set(str(key), value))
            return
        runtime_info = getattr(self, "runtime_info", None)
        if runtime_info is None:
            return
        try:
            runtime_info[str(key)] = value  # type: ignore[index]
        except Exception:
            return

    def _rt_update(self, payload: dict[str, object]) -> None:
        if not payload:
            return
        store = getattr(self, "runtime_store", None)
        if isinstance(store, RuntimeStore):
            store.dispatch(RuntimeAction.update(payload))
            return
        runtime_info = getattr(self, "runtime_info", None)
        if runtime_info is None:
            return
        for key, value in payload.items():
            try:
                runtime_info[str(key)] = value  # type: ignore[index]
            except Exception:
                continue

    def _set_preflight(self, *, ok: bool, message: str, calibration_ready: object = _UNSET) -> None:
        payload: dict[str, object] = {
            "preflight_ok": bool(ok),
            "preflight_message": str(message),
        }
        if calibration_ready is not _UNSET:
            payload["calibration_ready"] = calibration_ready
        self._rt_update(payload)

    def _uses_ros_transport(self) -> bool:
        return self.config.robot_mode == "real" and self.config.robot_transport == "ros"

    def _queue_remote_snapshot(self, snapshot: dict[str, object]) -> None:
        if self._shutdown_started:
            return
        envelope = RobotSnapshotEnvelope(
            payload=dict(snapshot),
            ts=time.time(),
            transport="ros_push",
            ok=True,
            error="",
        )
        try:
            self._bridge.remote_snapshot_received.emit(envelope)
        except RuntimeError:
            return

    def _preflight_ignores_calibration(self) -> bool:
        return self.config.vision_mode in {"fixed_world_slots", "fixed_cyl_slots"}

    def _evaluate_preflight_from_snapshot(self, snapshot: dict[str, object] | None) -> None:
        if not self._requires_preflight():
            self._set_preflight(ok=True, message="not_required")
            return
        if snapshot is None:
            self._set_preflight(ok=False, message="status_unavailable")
            return
        state = str(snapshot.get("state") or "")
        busy = bool(snapshot.get("busy"))
        carrying = bool(snapshot.get("carrying", snapshot.get("carrying_target_id") is not None))
        calibration_ready = snapshot.get("calibration_ready")
        self._rt_set("calibration_ready", calibration_ready)
        if state not in {"IDLE", "CARRY_READY"}:
            self._set_preflight(ok=False, message=f"state={state}")
            return
        if busy:
            self._set_preflight(ok=False, message="robot_busy")
            return
        if carrying:
            self._set_preflight(ok=False, message="robot_carrying_target")
            return
        if not self._preflight_ignores_calibration() and calibration_ready is False:
            self._set_preflight(ok=False, message="calibration_not_ready")
            return
        self._set_preflight(ok=True, message="ready")

    def _setup_robot_mode(self) -> None:
        if self._uses_ros_transport():
            if not self._probe_tcp_port(
                host=self.config.robot_host,
                port=int(self.config.rosbridge_port),
                timeout_sec=float(self.config.ros_runtime_probe_timeout_sec),
            ):
                self._ros_last_connected = False
                self._ros_connected_since_ts = 0.0
                self._ros_stale_detection_count = 0
                self._rt_update(
                    {
                        "robot_connected": False,
                        "robot_health": "rosbridge_port_closed",
                        "preflight_ok": False,
                        "preflight_message": "rosbridge_port_closed",
                    }
                )
                self._log_runtime(
                    "robot",
                    "ROS bridge port is unreachable: ws://{0}:{1}. Please ensure robot runtime is running.".format(
                        str(self.config.robot_host),
                        int(self.config.rosbridge_port),
                    ),
                )
                self._maybe_auto_start_robot_runtime("rosbridge_port_closed")
                return
            try:
                self.ros_client = RosbridgeClient(
                    self.config.robot_host,
                    self.config.rosbridge_port,
                    state_callback=self._queue_remote_snapshot,
                    event_callback=self._queue_event,
                    status_callback=lambda message: self._queue_runtime_status("robot", message),
                )
                self.ros_client.connect()
                deadline = time.time() + max(self.config.rosbridge_timeout_sec, 0.5)
                while time.time() < deadline:
                    if self.ros_client.is_connected():
                        break
                    time.sleep(0.05)
                self._rt_update(
                    {
                        "robot_connected": self.ros_client.is_connected(),
                        "robot_health": "ok" if self.ros_client.is_connected() else "rosbridge_connecting",
                    }
                )
                if self.ros_client.is_connected():
                    self._ros_last_connected = True
                    self._ros_reconnect_attempt = 0
                    self._ros_reconnect_next_ts = 0.0
                    self._ros_connected_since_ts = time.monotonic()
                    self._ros_stale_detection_count = 0
                    self._log_runtime(
                        "robot",
                        f"ROS bridge connected on ws://{self.config.robot_host}:{self.config.rosbridge_port}",
                    )
                    self._sync_pick_tuning_from_robot()
                else:
                    self._ros_last_connected = False
                    self._ros_connected_since_ts = 0.0
                    self._log_runtime(
                        "robot",
                        f"ROS bridge connecting on ws://{self.config.robot_host}:{self.config.rosbridge_port}",
                    )
                self._capture_world_snapshot(reason="connect", force=True)
                self._evaluate_preflight_from_snapshot(self._latest_world_snapshot)
                return
            except Exception as error:
                self._ros_last_connected = False
                self._ros_connected_since_ts = 0.0
                self._ros_stale_detection_count = 0
                self._rt_update(
                    {
                        "robot_connected": False,
                        "robot_health": "connect_failed",
                        "last_robot_error": str(error),
                        "preflight_ok": False,
                        "preflight_message": "connect_failed",
                    }
                )
                self._log_runtime("robot", f"ROS bridge connect failed: {error}")
                self._maybe_auto_start_robot_runtime("rosbridge_connect_failed")
                return

        self._ros_last_connected = False
        self._ros_reconnect_attempt = 0
        self._ros_reconnect_next_ts = 0.0
        self._ros_connected_since_ts = 0.0
        try:
            self.robot_client.connect()
            ok = self.robot_client.healthcheck(timeout_sec=self.config.robot_ping_timeout_sec)
            self._rt_update(
                {
                    "robot_connected": self.robot_client.is_connected(),
                    "robot_health": "ok" if ok else "ping_failed",
                }
            )
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
            self._rt_update(
                {
                    "robot_connected": False,
                    "robot_health": "connect_failed",
                    "last_robot_error": str(error),
                    "preflight_ok": False,
                    "preflight_message": "connect_failed",
                }
            )
            self._log_runtime("robot", f"Robot connect failed: {error}")

    @staticmethod
    def _probe_tcp_port(*, host: str, port: int, timeout_sec: float) -> bool:
        try:
            with socket.create_connection((str(host), int(port)), timeout=max(0.1, float(timeout_sec))):
                return True
        except Exception:
            return False

    def _maybe_auto_start_robot_runtime(self, reason: str) -> bool:
        if not self._uses_ros_transport():
            return False
        if bool(self._auto_start_blocked):
            now = time.monotonic()
            if (now - float(self._last_ros_runtime_unavailable_log_ts)) >= 5.0:
                self._last_ros_runtime_unavailable_log_ts = float(now)
                self._queue_runtime_status(
                    "robot",
                    "Auto-start is temporarily blocked: {0}".format(
                        str(self._auto_start_block_reason or "manual intervention required")
                    ),
                )
            return False
        if not bool(getattr(self.config, "robot_auto_start_on_ros_unavailable", False)):
            return False
        if bool(self._rt_get("robot_start_active", False)):
            return False
        now = time.monotonic()
        cooldown_sec = max(1.0, float(getattr(self.config, "robot_auto_start_cooldown_sec", 20.0)))
        if (now - float(self._last_auto_robot_start_ts)) < cooldown_sec:
            return False
        self._last_auto_robot_start_ts = float(now)
        self._queue_runtime_status("robot", f"Auto-start robot runtime due to: {str(reason)}")
        self._on_robot_start_requested()
        return True

    def _maybe_recover_ros_runtime_from_stale_state(self, *, state_age_ms: float) -> None:
        if not self._uses_ros_transport():
            self._ros_stale_detection_count = 0
            return
        if not bool(getattr(self.config, "robot_auto_restart_on_state_stale", False)):
            self._ros_stale_detection_count = 0
            return
        now = time.monotonic()
        disconnect_cooldown_sec = 25.0
        if (now - float(self._last_ros_disconnect_ts)) < disconnect_cooldown_sec:
            self._ros_stale_detection_count = 0
            return
        if self.ros_client is None or not self.ros_client.is_connected():
            self._ros_stale_detection_count = 0
            return
        grace_sec = max(1.0, float(getattr(self.config, "ros_runtime_state_grace_sec", 3.0)))
        connected_since = float(self._ros_connected_since_ts)
        if connected_since <= 0.0:
            self._ros_connected_since_ts = time.monotonic()
            self._ros_stale_detection_count = 0
            return
        if (time.monotonic() - connected_since) < grace_sec:
            self._ros_stale_detection_count = 0
            return
        stale_threshold_ms = float(getattr(self.config, "robot_state_stale_threshold_ms", 700.0))
        no_fresh_state = (not math.isfinite(float(state_age_ms))) or float(state_age_ms) > stale_threshold_ms
        if not no_fresh_state:
            self._ros_stale_detection_count = 0
            return
        self._ros_stale_detection_count += 1
        required_stale_ms = max(stale_threshold_ms * 6.0, 6000.0)
        if float(state_age_ms) < required_stale_ms:
            return
        if int(self._ros_stale_detection_count) < 6:
            return
        if (now - float(self._last_ros_runtime_unavailable_log_ts)) >= 3.0:
            self._last_ros_runtime_unavailable_log_ts = float(now)
            self._queue_runtime_status(
                "robot",
                "ROS state stream unavailable for {0:.0f} ms. Checking robot runtime...".format(float(state_age_ms)),
            )
        self._ros_stale_detection_count = 0
        self._maybe_auto_start_robot_runtime("ros_state_stream_stale")

    @staticmethod
    def _module_available(module_name: str) -> bool:
        return importlib.util.find_spec(module_name) is not None

    @staticmethod
    def _expected_brain_vision_python() -> Path | None:
        override = os.environ.get("BRAIN_PYTHON_EXE", "").strip()
        if override:
            path = Path(override).expanduser()
            return path if path.exists() else None
        home = Path.home()
        candidates = (
            home / "miniconda3" / "envs" / "brain-vision" / "python.exe",
            home / "anaconda3" / "envs" / "brain-vision" / "python.exe",
            home / "mambaforge" / "envs" / "brain-vision" / "python.exe",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _build_pick_tuning_defaults(self) -> dict[str, object]:
        return {
            "pick_approach_z_mm": float(self.config.robot_approach_z),
            "pick_descend_z_mm": float(self.config.robot_pick_z),
            "pick_pre_suction_sec": 0.25,
            "pick_bottom_hold_sec": 0.15,
            "pick_lift_sec": 0.8,
            "place_descend_z_mm": float(self.config.robot_pick_z),
            "place_release_mode": "release",
            "place_release_sec": 0.25,
            "place_post_release_hold_sec": 0.10,
            "z_carry_floor_mm": float(self.config.robot_carry_z),
        }

    def _sanitize_pick_tuning(self, payload: dict[str, object] | None) -> dict[str, object]:
        data = dict(self._pick_tuning_defaults)
        if isinstance(payload, dict):
            data.update(payload)
        z_min = float(self.config.robot_height_limits_mm[0])
        z_max = float(self.config.robot_height_limits_mm[1])

        def _to_float(name: str, lower: float, upper: float) -> float:
            raw = data.get(name, self._pick_tuning_defaults[name])
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = float(self._pick_tuning_defaults[name])
            return max(float(lower), min(float(upper), value))

        def _to_sec(name: str) -> float:
            return _to_float(name, 0.0, 3.0)

        release_mode = str(data.get("place_release_mode", "release") or "release").strip().lower()
        if release_mode not in {"release", "off"}:
            release_mode = "release"

        return {
            "pick_approach_z_mm": _to_float("pick_approach_z_mm", z_min, z_max),
            "pick_descend_z_mm": _to_float("pick_descend_z_mm", z_min, z_max),
            "pick_pre_suction_sec": _to_sec("pick_pre_suction_sec"),
            "pick_bottom_hold_sec": _to_sec("pick_bottom_hold_sec"),
            "pick_lift_sec": _to_sec("pick_lift_sec"),
            "place_descend_z_mm": _to_float("place_descend_z_mm", z_min, z_max),
            "place_release_mode": release_mode,
            "place_release_sec": _to_sec("place_release_sec"),
            "place_post_release_hold_sec": _to_sec("place_post_release_hold_sec"),
            "z_carry_floor_mm": _to_float("z_carry_floor_mm", z_min, z_max),
        }

    def _load_pick_tuning_profile(self) -> dict[str, object]:
        path = Path(self.config.pick_tuning_profile_path)
        if not path.exists():
            return dict(self._pick_tuning_defaults)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            self._log_runtime("robot", f"Pick tuning profile load failed: {error}")
            return dict(self._pick_tuning_defaults)
        return self._sanitize_pick_tuning(payload if isinstance(payload, dict) else None)

    def _save_pick_tuning_profile(self) -> Path:
        path = Path(self.config.pick_tuning_profile_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._pick_tuning_state, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _sync_pick_tuning_from_robot(self) -> None:
        if not self._uses_ros_transport() or self.ros_client is None:
            return
        if not self.ros_client.is_connected():
            return

        def callback(result: RosServiceResult) -> None:
            if not result.ok:
                message = str(result.message or "")
                if "does not exist" in message.lower():
                    self._queue_runtime_status("robot", "Pick tuning service unavailable on robot runtime (optional).")
                else:
                    self._queue_runtime_status("robot", f"Pick tuning sync failed: {message}")
                return
            if bool(self._pick_tuning_local_dirty):
                return
            raw = dict(result.raw or {})
            tuning = {
                "pick_approach_z_mm": raw.get("pick_approach_z_mm", self._pick_tuning_state.get("pick_approach_z_mm")),
                "pick_descend_z_mm": raw.get("pick_descend_z_mm", self._pick_tuning_state.get("pick_descend_z_mm")),
                "pick_pre_suction_sec": raw.get("pick_pre_suction_sec", self._pick_tuning_state.get("pick_pre_suction_sec")),
                "pick_bottom_hold_sec": raw.get("pick_bottom_hold_sec", self._pick_tuning_state.get("pick_bottom_hold_sec")),
                "pick_lift_sec": raw.get("pick_lift_sec", self._pick_tuning_state.get("pick_lift_sec")),
                "place_descend_z_mm": raw.get("place_descend_z_mm", self._pick_tuning_state.get("place_descend_z_mm")),
                "place_release_mode": raw.get("place_release_mode", self._pick_tuning_state.get("place_release_mode")),
                "place_release_sec": raw.get("place_release_sec", self._pick_tuning_state.get("place_release_sec")),
                "place_post_release_hold_sec": raw.get(
                    "place_post_release_hold_sec", self._pick_tuning_state.get("place_post_release_hold_sec")
                ),
                "z_carry_floor_mm": raw.get("z_carry_floor_mm", self._pick_tuning_state.get("z_carry_floor_mm")),
            }
            self._pick_tuning_state = self._sanitize_pick_tuning(tuning)
            self._rt_set("pick_tuning", dict(self._pick_tuning_state))
            self.main_window.update_pick_tuning_display(self._pick_tuning_state)

        try:
            self.ros_client.get_pick_tuning(callback=callback)
        except Exception as error:
            message = str(error)
            if "does not exist" in message.lower():
                self._handle_runtime_status("robot", "Pick tuning service unavailable on robot runtime (optional).")
            else:
                self._handle_runtime_status("robot", f"Pick tuning sync failed: {message}")

    def _report_runtime_environment(self) -> None:
        current_python = Path(sys.executable).resolve()
        self._log_runtime("system", f"Desktop interpreter: {current_python}")
        missing_modules: list[str] = []
        if self._uses_ros_transport() and not self._module_available("roslibpy"):
            missing_modules.append("roslibpy")
        if self.config.robot_mode == "real" and not self._module_available("paramiko"):
            missing_modules.append("paramiko")
        if self.config.vision_mode in {"real", "robot_camera_detection"}:
            if not self._module_available("cv2"):
                missing_modules.append("cv2")
            if not self._module_available("torch"):
                missing_modules.append("torch")
        if not missing_modules:
            return

        expected_python = self._expected_brain_vision_python()
        if expected_python is None:
            expected_hint = (
                "Expected interpreter: brain-vision (not found automatically). "
                "Set BRAIN_PYTHON_EXE to an absolute python.exe path if needed."
            )
        else:
            expected_hint = f"Expected interpreter: {expected_python}"
        missing_text = ", ".join(sorted(set(missing_modules)))
        message = (
            f"Environment check failed in current interpreter ({current_python}). "
            f"Missing modules: {missing_text}. {expected_hint}"
        )
        self._log_runtime("system", message)
        if "roslibpy" in missing_modules:
            self._rt_update(
                {
                    "robot_health": "env_missing_deps",
                    "preflight_ok": False,
                    "preflight_message": "missing_roslibpy",
                    "last_robot_error": "missing_roslibpy_in_current_interpreter",
                }
            )
        if "cv2" in missing_modules or "torch" in missing_modules:
            self._rt_set("vision_health", "env_missing_deps")

    def _setup_vision_mode(self) -> None:
        if self.config.control_sim_enabled and self.config.vision_mode in {"slots", "fixed_world_slots", "fixed_cyl_slots"}:
            self._publish_control_sim_targets()
            self._rt_set("vision_health", f"{self.config.vision_mode}:{self.config.slot_profile}")
            return
        if self.config.vision_mode not in {"real", "robot_camera_detection"}:
            self._rt_set("vision_health", f"disabled:{self.config.vision_mode}")
            return

        calibration_params = self._fetch_vision_calibration_params()
        self.vision_runtime = VisionRuntime(
            self.config,
            calibration_params=calibration_params,
            targets_callback=lambda targets: None,
            packet_callback=self._queue_vision_packet,
            frame_callback=self._queue_vision_frame,
            status_callback=lambda message: self._queue_runtime_status("vision", message),
        )
        self._log_runtime(
            "vision",
            "Vision world mapping: mode={0}, scale_xy={1:.3f}, offset_xy=({2:.1f},{3:.1f}) mm".format(
                str(self.config.vision_mapping_mode),
                float(self.config.vision_world_scale_xy),
                float(self.config.vision_world_offset_xy_mm[0]),
                float(self.config.vision_world_offset_xy_mm[1]),
            ),
        )
        self.vision_runtime.start()
        if calibration_params is None:
            self._rt_set("vision_health", "starting_without_calibration")
        else:
            self._rt_set("vision_health", "starting")

    def _setup_brain_sources(self) -> None:
        self.ssvep_runtime = SSVEPRuntime(
            self.config,
            command_callback=self.dispatch_ssvep_command,
            status_callback=lambda message: self._queue_runtime_status("ssvep", message),
            state_callback=self._queue_ssvep_state,
        )
        self.ssvep_coordinator.bind_runtime(self.ssvep_runtime)
        self._rt_update(
            {
                "ssvep_profile_path": str(self.ssvep_runtime.current_profile_path or self.config.ssvep_current_profile_path),
                "ssvep_profile_source": self.ssvep_runtime.current_profile_source,
                "ssvep_last_pretrain_time": self.ssvep_runtime.last_pretrain_time or "--",
            }
        )
        initial_profiles = self.ssvep_runtime.list_profiles(limit=self.config.ssvep_recent_profile_limit)
        latest_profile = self.ssvep_runtime.latest_profile()
        self._rt_update(
            {
                "ssvep_available_profiles": tuple((profile.display_name, str(profile.path)) for profile in initial_profiles),
                "ssvep_profile_count": len(initial_profiles),
                "ssvep_latest_profile_path": str(latest_profile.path) if latest_profile is not None else "--",
                "ssvep_status_hint": self.ssvep_runtime.status_hint(initial_profiles),
                "ssvep_runtime_status": "idle (click Connect Device)",
            }
        )
        self.input_orchestrator.set_sources(
            move_source=str(self.config.move_source),
            decision_source=str(self.config.decision_source),
            ssvep_keyboard_debug_enabled=bool(self.config.ssvep_keyboard_debug_enabled),
            mi_enabled=bool(self.config.mi_enabled),
        )
        initial_mi_health = self.input_orchestrator.initialize()
        self._update_mi_runtime_state(initial_mi_health, emit_runtime_status=True)

    def _update_mi_runtime_state(self, health: InputHealth | None, *, emit_runtime_status: bool) -> None:
        if self.config.move_source != "mi":
            self._rt_update(
                {
                    "mi_status": "inactive",
                    "mi_connected": False,
                    "mi_running": False,
                    "mi_last_error": "--",
                }
            )
            return
        if health is None:
            self._rt_update(
                {
                    "mi_status": "unavailable",
                    "mi_connected": False,
                    "mi_running": False,
                    "mi_last_error": "unavailable",
                }
            )
            return

        self._rt_update(
            {
                "mi_status": str(health.status),
                "mi_connected": bool(health.connected),
                "mi_running": bool(health.running),
                "mi_last_error": str(health.last_error),
            }
        )
        if not emit_runtime_status:
            return
        signature = (
            health.status,
            bool(health.connected),
            bool(health.running),
            bool(health.ready),
            str(health.last_error),
        )
        if signature == self._last_mi_health_signature:
            return
        self._last_mi_health_signature = signature
        self._queue_runtime_status(
            "mi",
            "MI source status={0} connected={1} running={2} ready={3} error={4}".format(
                health.status,
                bool(health.connected),
                bool(health.running),
                bool(health.ready),
                str(health.last_error),
            ),
        )

    def _should_emit_mi_status(self) -> bool:
        now = time.monotonic()
        if (now - float(self._last_mi_status_emit_ts)) < 1.0:
            return False
        self._last_mi_status_emit_ts = now
        return True

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
        self._send_robot_text_command(command)

    def _send_robot_text_command(self, command: str) -> None:
        outgoing_command = self._rewrite_outgoing_robot_command(str(command))
        opcode = self._extract_command_opcode(outgoing_command)
        if opcode in {"PICK", "PICK_WORLD", "PICK_CYL"}:
            self._begin_pick_trace(command=outgoing_command)
        if self._uses_ros_transport():
            if self._send_robot_command_via_ros(outgoing_command):
                self.main_window.append_log(f"Robot <= {outgoing_command}")
                return
            if self._ros_command_requires_ros_route(opcode):
                self._reject_ros_command(outgoing_command, reason=f"ROS transport does not support command '{opcode}'.")
                return
        try:
            self.robot_client.send_command(outgoing_command)
            self.main_window.append_log(f"Robot <= {outgoing_command}")
        except Exception as error:
            self._handle_runtime_status("robot", f"Robot send failed: {error}")
            self.dispatch_event(Event(source="robot", type="robot_error", value=f"Robot send failed: {error}"))
            if opcode in {"PICK", "PICK_WORLD", "PICK_CYL"}:
                self._finish_pick_trace(response=f"ERR Robot send failed: {error}")

    @staticmethod
    def _extract_command_opcode(command: str) -> str:
        return extract_command_opcode(command)

    @staticmethod
    def _ros_command_requires_ros_route(opcode: str) -> bool:
        return ros_command_requires_ros_route(opcode)

    def _reject_ros_command(self, command: str, *, reason: str) -> None:
        message = f"ROS transport command rejected: {reason} command={command}"
        self._handle_runtime_status("robot", message)
        self.dispatch_event(Event(source="robot", type="robot_error", value=message))
        opcode = self._extract_command_opcode(command)
        if opcode in {"PICK", "PICK_WORLD", "PICK_CYL"}:
            self._finish_pick_trace(response=f"ERR {message}")

    def _begin_pick_trace(self, *, command: str) -> None:
        snapshot = self._fetch_remote_robot_snapshot()
        self._active_pick_trace = {
            "slot_id": self.controller.context.selected_target_id,
            "mapping_mode": self._rt_get("vision_mapping_mode"),
            "command": str(command),
            "resolved_base_xy": self._rt_get("vision_last_resolved_base_xy"),
            "resolved_cyl": self._rt_get("vision_last_resolved_cyl"),
            "snapshot_age_ms": self._rt_get("vision_snapshot_age_ms"),
            "robot_pose": None if not isinstance(snapshot, dict) else snapshot.get("robot_cyl") or snapshot.get("robot_xy"),
            "robot_xy": None if not isinstance(snapshot, dict) else snapshot.get("robot_xy"),
            "pick_tuning": dict(self._pick_tuning_state),
            "post_pick_settle_z": self._rt_get("post_pick_settle_z"),
            "release_mode_effective": self._rt_get("release_mode_effective"),
            "transport": "ros" if self._uses_ros_transport() else "tcp",
        }

    def _finish_pick_trace(self, *, response: str) -> None:
        trace = self._active_pick_trace
        if not isinstance(trace, dict):
            return
        trace_payload = dict(trace)
        trace_payload["response"] = str(response)
        self.logger.write("pick_trace", **trace_payload)
        self._active_pick_trace = None

    @staticmethod
    def _pending_expected_ack_for_opcode(opcode: str) -> str | None:
        op = str(opcode or "").strip().upper()
        if op in {"MOVE_CYL", "MOVE_CYL_AUTO"}:
            return "MOVE"
        if op in {"PICK_CYL", "PICK_WORLD"}:
            return "PICK_DONE"
        if op == "PLACE":
            return "PLACE_DONE"
        if op == "ABORT":
            return "ABORT"
        if op == "RESET":
            return "RESET"
        return None

    def _register_pending_command(self, *, opcode: str, command: str) -> None:
        expected_ack = self._pending_expected_ack_for_opcode(opcode)
        if expected_ack is None:
            return
        seq = int(self._next_pending_command_seq)
        self._next_pending_command_seq = seq + 1
        now = float(time.time())
        timeout_sec = max(1.0, float(self.config.robot_command_timeout_sec))
        self._pending_command = {
            "seq": seq,
            "opcode": str(opcode),
            "command": str(command),
            "expected_ack": str(expected_ack),
            "sent_ts": now,
            "deadline_ts": now + timeout_sec,
        }
        self._rt_update(
            {
                "pending_command": str(command),
                "pending_command_seq": int(seq),
                "pending_command_deadline_ts": float(now + timeout_sec),
            }
        )

    def _clear_pending_command(self) -> None:
        self._pending_command = None
        self._rt_update(
            {
                "pending_command": "--",
                "pending_command_seq": 0,
                "pending_command_deadline_ts": 0.0,
            }
        )

    def _resolve_pending_command_from_event(self, event: Event) -> None:
        pending = self._pending_command
        if not isinstance(pending, dict):
            return
        if event.source != "robot":
            return
        event_type = str(event.type or "")
        if event_type in {"robot_error", "robot_busy", "robot_disconnected"}:
            self._clear_pending_command()
            return
        if event_type != "robot_ack":
            return
        expected_ack = str(pending.get("expected_ack", "")).strip().upper()
        current_ack = str(event.value or "").strip().upper()
        if expected_ack and current_ack == expected_ack:
            self._clear_pending_command()

    def _check_pending_command_timeout(self) -> None:
        pending = self._pending_command
        if not isinstance(pending, dict):
            return
        deadline_ts = float(pending.get("deadline_ts", 0.0) or 0.0)
        if deadline_ts <= 0.0 or time.time() <= deadline_ts:
            return
        opcode = str(pending.get("opcode", "UNKNOWN"))
        command = str(pending.get("command", ""))
        self._handle_runtime_status(
            "robot",
            "Pending command timeout: opcode={0} command={1}".format(opcode, command),
        )
        timeout_message = "ERR command_timeout: opcode={0} command={1}".format(opcode, command)
        self._queue_event(Event(source="robot", type="robot_error", value=timeout_message))
        if opcode in {"PICK", "PICK_CYL", "PICK_WORLD"}:
            self._finish_pick_trace(response=timeout_message)
        self._clear_pending_command()

    def _rewrite_outgoing_robot_command(self, command: str) -> str:
        return rewrite_pick_command_with_bias(
            str(command or ""),
            theta_bias_deg=float(self._pick_cyl_theta_bias_deg),
            radius_bias_mm=float(self._pick_cyl_radius_bias_mm),
            pick_z_mm=float(self.config.robot_pick_z),
        )

    def _send_robot_command_via_ros(self, command: str) -> bool:
        if self.ros_client is None:
            return False
        parts = str(command).strip().split()
        if not parts:
            return False
        op = parts[0].upper()

        def callback(result: RosServiceResult, *, issued_command: str = command, opcode: str = op) -> None:
            if result.ok:
                self._register_pending_command(opcode=opcode, command=issued_command)
                response_message = str(result.message or "").strip()
                if response_message:
                    self._queue_runtime_status("robot", "ROS accepted: {0} -> {1}".format(opcode, response_message))
                return
            message = str(result.message or issued_command)
            if message.strip().upper() == "BUSY":
                self._queue_event(Event(source="robot", type="robot_busy", value=issued_command))
            else:
                self._queue_event(Event(source="robot", type="robot_error", value=message))
            if opcode in {"PICK", "PICK_CYL", "PICK_WORLD"}:
                self._finish_pick_trace(response=message)

        try:
            if op == "MOVE_CYL" and len(parts) == 4:
                self.ros_client.send_move_cyl(float(parts[1]), float(parts[2]), float(parts[3]), callback=callback)
                return True
            if op == "MOVE_CYL_AUTO" and len(parts) == 3:
                self.ros_client.send_move_cyl_auto(float(parts[1]), float(parts[2]), callback=callback)
                return True
            if op == "PICK_WORLD" and len(parts) == 3:
                self.ros_client.send_pick_world(float(parts[1]), float(parts[2]), callback=callback)
                return True
            if op == "PICK_CYL" and len(parts) == 3:
                self.ros_client.send_pick_cyl(float(parts[1]), float(parts[2]), callback=callback)
                return True
            if op == "PLACE":
                self.ros_client.send_place(callback=callback)
                return True
            if op == "ABORT":
                self.ros_client.send_abort(callback=callback)
                return True
            if op == "RESET":
                self.ros_client.send_reset(callback=callback)
                return True
        except Exception as error:
            self._queue_event(Event(source="robot", type="robot_error", value=f"ROS command failed: {error}"))
            if op in {"PICK", "PICK_CYL", "PICK_WORLD"}:
                self._finish_pick_trace(response=f"ERR ROS command failed: {error}")
            return True
        return False

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
        self._rt_update(
            {
                "ssvep_mode": mode,
                "target_frequency_map": self._build_target_frequency_map(mode),
            }
        )

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
        if self._uses_ros_transport():
            connected = self.ros_client.is_connected() if self.ros_client is not None else False
            self._rt_set("robot_connected", connected)
            state_age_ms = float(self._rt_get("state_age_ms", self._compute_remote_snapshot_age_ms()))
            stale_threshold_ms = float(getattr(self.config, "robot_state_stale_threshold_ms", 700.0))
            sticky_errors = {"connect_failed", "send_failed", "start_failed"}
            current_health = str(self._rt_get("robot_health", "unknown"))
            if current_health not in sticky_errors:
                if not connected:
                    self._rt_set("robot_health", "disconnected")
                    self._ros_connected_since_ts = 0.0
                elif state_age_ms > stale_threshold_ms:
                    self._rt_set("robot_health", "state_stale")
                else:
                    self._rt_set("robot_health", "ok")
            self._maybe_recover_ros_runtime_from_stale_state(state_age_ms=state_age_ms)
        else:
            connected = self.robot_client.is_connected()
            self._rt_set("robot_connected", connected)
            if self._rt_get("robot_health") not in {"connect_failed", "send_failed"}:
                self._rt_set("robot_health", "ok" if connected else "disconnected")
        if self.vision_runtime is not None:
            if self._rt_get("vision_health") in {None, "unknown"}:
                self._rt_set("vision_health", "running")
        elif self.config.control_sim_enabled and self.config.vision_mode in {"slots", "fixed_world_slots", "fixed_cyl_slots"}:
            self._rt_set("vision_health", f"{self.config.vision_mode}:{self.config.slot_profile}")

    def _refresh_view(self) -> None:
        self._refresh_dirty = True
        if not self._refresh_coalesce_timer.isActive():
            self._refresh_coalesce_timer.start(0)

    def _refresh_panels(self) -> None:
        if not self._refresh_dirty:
            return
        self._refresh_dirty = False
        refresh_start = time.perf_counter()
        self._capture_world_snapshot(reason="refresh")
        remote_age_ms = self._compute_remote_snapshot_age_ms()
        self._rt_set("remote_snapshot_age_ms", remote_age_ms)
        self._rt_set("state_age_ms", remote_age_ms)
        self._update_runtime_health()
        self._sync_coordinator_states_from_runtime_info()
        snapshot = build_ui_snapshot(
            ui_coordinator=self.ui_coordinator,
            controller_snapshot=self.controller.snapshot(),
            config=self.config,
            runtime_info=self.runtime_info,
            robot_coordinator=self.robot_coordinator,
            vision_coordinator=self.vision_coordinator,
            ssvep_coordinator=self.ssvep_coordinator,
        )
        self.main_window.update_panels(snapshot)
        self.main_window.update_pick_tuning_display(self._pick_tuning_state)
        elapsed_ms = (time.perf_counter() - refresh_start) * 1000.0
        previous_ema = float(self._rt_get("ui_refresh_ms_ema", 0.0))
        if previous_ema <= 0.0:
            next_ema = float(elapsed_ms)
        else:
            next_ema = previous_ema * 0.85 + float(elapsed_ms) * 0.15
        self._rt_set("ui_refresh_ms_ema", float(next_ema))

    def _sync_coordinator_states_from_runtime_info(self) -> None:
        sync_coordinator_states_from_runtime_info(
            config=self.config,
            runtime_info=self.runtime_info,
            latest_world_snapshot=self._latest_world_snapshot,
            latest_vision_packet=self._latest_vision_packet,
            fetch_remote_snapshot=self._fetch_remote_robot_snapshot,
            robot_coordinator=self.robot_coordinator,
            vision_coordinator=self.vision_coordinator,
            ssvep_coordinator=self.ssvep_coordinator,
        )

    def _on_key_pressed(self, token: str) -> None:
        self.logger.log_raw_input("sim_key", token)
        if self._is_move_token(token):
            self._handle_move_key_pressed(token)
            return
        for event in self.input_orchestrator.handle_key_token(token):
            self.dispatch_event(event)

    def _on_key_released(self, token: str) -> None:
        self.logger.log_raw_input("sim_key_release", token)
        if self._is_move_token(token):
            self._handle_move_key_released(token)

    def _on_robot_start_requested(self) -> None:
        if bool(self._rt_get("robot_start_active", False)):
            return
        if self.config.robot_mode != "real":
            self._handle_runtime_status("robot", "Robot start is only available in real mode.")
            return
        self._auto_start_blocked = False
        self._auto_start_block_reason = ""
        self._rt_update(
            {
                "robot_start_active": True,
                "robot_health": "starting_remote_runtime",
                "last_robot_error": "--",
            }
        )
        self._handle_runtime_status("robot", f"Starting robot runtime on {self.config.robot_host} ...")
        threading.Thread(target=self._start_robot_runtime_worker, name="robot-start-worker", daemon=True).start()

    def _start_robot_runtime_worker(self) -> None:
        script_path = Path(__file__).resolve().parent / "robot" / "tools" / "jetmax_start_ros_runtime.py"
        if not script_path.exists():
            self._bridge.robot_start_finished.emit(False, f"Start script not found: {script_path}")
            return
        self._bridge.runtime_status_received.emit("robot", f"Robot start interpreter: {sys.executable}")
        command = [sys.executable, str(script_path), "--host", str(self.config.robot_host)]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except Exception as error:
            message = f"Failed to run start script: {error}"
            if isinstance(error, PermissionError) or "WinError 10013" in str(error):
                message = (
                    "Failed to run start script: Windows blocked the SSH socket (WinError 10013). "
                    "Allow Python outbound connections or run as Administrator, then retry."
                )
            self._bridge.robot_start_finished.emit(False, message)
            return

        details = "\n".join(
            part.strip()
            for part in (result.stdout or "", result.stderr or "")
            if str(part).strip()
        )
        if result.returncode == 0:
            self._bridge.robot_start_finished.emit(True, details)
            return
        message = details or f"Start script exited with code {result.returncode}"
        self._bridge.robot_start_finished.emit(False, message)

    def _on_robot_start_finished(self, success: bool, details: str) -> None:
        self._rt_set("robot_start_active", False)
        lines = [line.strip() for line in str(details or "").splitlines() if line.strip()]
        for line in lines[-12:]:
            self.main_window.append_log(f"[robot-start] {line}")
        if success:
            self._auto_start_blocked = False
            self._auto_start_block_reason = ""
            self._rt_update(
                {
                    "robot_health": "runtime_started_remote",
                    "last_robot_error": "--",
                }
            )
            self._handle_runtime_status("robot", "Robot runtime started. Reconnecting...")
            self._on_robot_connect_requested()
            return
        error_line = lines[-1] if lines else "unknown_error"
        details_text = str(details or "")
        if "WinError 10013" in details_text or "Windows blocked the SSH socket" in details_text:
            self._auto_start_blocked = True
            self._auto_start_block_reason = "Windows blocked SSH socket (WinError 10013)"
        self._rt_update(
            {
                "robot_health": "start_failed",
                "last_robot_error": error_line,
            }
        )
        self._handle_runtime_status("robot", f"Robot start failed: {error_line}")
        self._refresh_view()

    def _on_robot_connect_requested(self) -> None:
        self._stop_teleop_motion(send_command=False, reason="robot_reconnect_button")
        if self.ros_client is not None:
            try:
                self.ros_client.close()
            except Exception:
                pass
            self.ros_client = None
        try:
            self.robot_client.close()
        except Exception:
            pass
        self._rt_update(
            {
                "robot_connected": False,
                "robot_health": "reconnecting",
                "last_robot_error": "--",
                "preflight_ok": False,
                "preflight_message": "reconnecting",
            }
        )
        self._ros_last_connected = False
        self._ros_reconnect_attempt = 0
        self._ros_reconnect_next_ts = 0.0
        self._ros_connected_since_ts = 0.0
        self._refresh_view()
        self._setup_robot_mode()
        self._request_remote_snapshot()
        self._refresh_view()

    def _on_abort_requested(self) -> None:
        self._stop_teleop_motion(send_command=False, reason="abort_button")
        self._send_robot_text_command("ABORT")

    def _on_reset_requested(self) -> None:
        self._stop_teleop_motion(send_command=False, reason="reset_button")
        self._send_robot_text_command("RESET")

    def _on_manual_pick_slot_requested(self, slot_id: int) -> None:
        command = self._build_manual_pick_command(int(slot_id))
        if command is None:
            details = ""
            packet = self._latest_vision_packet if isinstance(self._latest_vision_packet, dict) else None
            if isinstance(packet, dict):
                for slot in packet.get("slots", []):
                    if not isinstance(slot, dict):
                        continue
                    if int(slot.get("slot_id", slot.get("slot", -1))) != int(slot_id):
                        continue
                    reason = str(slot.get("invalid_reason") or "").strip()
                    if reason:
                        details = f" reason={reason}"
                    break
            self._handle_runtime_status("robot", f"Manual pick slot {int(slot_id)} unavailable.{details}")
            return
        self._send_robot_text_command(command)

    def _on_manual_place_requested(self) -> None:
        self._send_robot_text_command("PLACE")

    def _on_pick_radius_bias_delta_requested(self, delta_mm: float) -> None:
        self._pick_cyl_radius_bias_mm = float(self._pick_cyl_radius_bias_mm) + float(delta_mm)
        self.main_window.update_pick_bias_display(self._pick_cyl_radius_bias_mm, self._pick_cyl_theta_bias_deg)
        self._handle_runtime_status(
            "robot",
            "Pick bias -> r={0:+.1f}mm theta={1:+.1f}deg".format(
                self._pick_cyl_radius_bias_mm,
                self._pick_cyl_theta_bias_deg,
            ),
        )

    def _on_pick_bias_reset_requested(self) -> None:
        self._pick_cyl_radius_bias_mm = float(self.config.pick_cyl_radius_bias_mm)
        self.main_window.update_pick_bias_display(self._pick_cyl_radius_bias_mm, self._pick_cyl_theta_bias_deg)
        self._handle_runtime_status(
            "robot",
            "Pick bias reset -> r={0:+.1f}mm theta={1:+.1f}deg".format(
                self._pick_cyl_radius_bias_mm,
                self._pick_cyl_theta_bias_deg,
            ),
        )

    def _on_pick_theta_bias_delta_requested(self, delta_deg: float) -> None:
        self._pick_cyl_theta_bias_deg = float(self._pick_cyl_theta_bias_deg) + float(delta_deg)
        self.main_window.update_pick_bias_display(self._pick_cyl_radius_bias_mm, self._pick_cyl_theta_bias_deg)
        self._handle_runtime_status(
            "robot",
            "Pick bias -> r={0:+.1f}mm theta={1:+.1f}deg".format(
                self._pick_cyl_radius_bias_mm,
                self._pick_cyl_theta_bias_deg,
            ),
        )

    def _on_pick_theta_bias_reset_requested(self) -> None:
        self._pick_cyl_theta_bias_deg = float(self.config.pick_cyl_theta_bias_deg)
        self.main_window.update_pick_bias_display(self._pick_cyl_radius_bias_mm, self._pick_cyl_theta_bias_deg)
        self._handle_runtime_status(
            "robot",
            "Pick bias reset -> r={0:+.1f}mm theta={1:+.1f}deg".format(
                self._pick_cyl_radius_bias_mm,
                self._pick_cyl_theta_bias_deg,
            ),
        )

    def _on_pick_tuning_delta_requested(self, field: str, delta: float) -> None:
        name = str(field or "").strip()
        if not name or name not in self._pick_tuning_state:
            return
        current = self._pick_tuning_state.get(name)
        if name == "place_release_mode":
            return
        try:
            next_value = float(current) + float(delta)
        except (TypeError, ValueError):
            return
        self._pick_tuning_state[name] = next_value
        self._pick_tuning_state = self._sanitize_pick_tuning(self._pick_tuning_state)
        self._pick_tuning_local_dirty = True
        self._rt_set("pick_tuning", dict(self._pick_tuning_state))
        self.main_window.update_pick_tuning_display(self._pick_tuning_state)

    def _on_pick_release_mode_toggle_requested(self) -> None:
        current_mode = str(self._pick_tuning_state.get("place_release_mode", "release")).strip().lower()
        self._pick_tuning_state["place_release_mode"] = "off" if current_mode == "release" else "release"
        self._pick_tuning_state = self._sanitize_pick_tuning(self._pick_tuning_state)
        self._pick_tuning_local_dirty = True
        self._rt_set("pick_tuning", dict(self._pick_tuning_state))
        self.main_window.update_pick_tuning_display(self._pick_tuning_state)

    def _on_pick_tuning_apply_requested(self) -> None:
        if not self._uses_ros_transport() or self.ros_client is None:
            self._handle_runtime_status("robot", "Pick tuning apply requires ROS transport.")
            return
        payload = dict(self._sanitize_pick_tuning(self._pick_tuning_state))
        self._pick_tuning_state = dict(payload)
        self._rt_set("pick_tuning", dict(payload))

        def callback(result: RosServiceResult) -> None:
            if result.ok:
                self._pick_tuning_local_dirty = False
                self._queue_runtime_status("robot", "Pick tuning applied.")
                self._request_remote_snapshot()
            else:
                self._queue_runtime_status("robot", f"Pick tuning apply failed: {result.message}")

        try:
            self.ros_client.set_pick_tuning(payload, callback=callback)
        except Exception as error:
            self._handle_runtime_status("robot", f"Pick tuning apply failed: {error}")

    def _on_pick_tuning_reset_requested(self) -> None:
        self._pick_tuning_state = dict(self._pick_tuning_defaults)
        self._pick_tuning_local_dirty = True
        self._rt_set("pick_tuning", dict(self._pick_tuning_state))
        self.main_window.update_pick_tuning_display(self._pick_tuning_state)
        self._handle_runtime_status("robot", "Pick tuning reset to defaults (click Apply to send).")

    def _on_pick_tuning_save_requested(self) -> None:
        self._pick_tuning_state = self._sanitize_pick_tuning(self._pick_tuning_state)
        self._rt_set("pick_tuning", dict(self._pick_tuning_state))
        try:
            saved_path = self._save_pick_tuning_profile()
        except Exception as error:
            self._handle_runtime_status("robot", f"Pick tuning save failed: {error}")
            return
        self._handle_runtime_status("robot", f"Pick tuning saved: {saved_path}")

    def _build_manual_pick_command(self, slot_id: int) -> str | None:
        packet = self._latest_vision_packet
        if isinstance(packet, dict):
            for slot in packet.get("slots", []):
                if not isinstance(slot, dict):
                    continue
                if int(slot.get("slot_id", slot.get("slot", -1))) != int(slot_id):
                    continue
                if not bool(slot.get("valid", False)):
                    continue
                if not bool(slot.get("actionable", False)):
                    continue
                command = build_pick_command_from_slot_payload(slot)
                if command is not None:
                    return command
        for target in self.controller.context.latest_vision_targets:
            target_slot_id = getattr(target, "slot_id", None)
            target_id = getattr(target, "id", None)
            if int(target_slot_id or target_id or -1) != int(slot_id):
                continue
            command = build_pick_command_from_target(target)
            if command is not None:
                return command
        if self.slot_catalog is not None and self.config.vision_mode != "robot_camera_detection":
            for slot in self.slot_catalog.list_pick_slots(source=self._pick_slot_source()):
                if int(getattr(slot, "slot_id", -1)) != int(slot_id):
                    continue
                command = build_catalog_pick_command(slot)
                if command is not None:
                    return command
        return None

    def _build_pick_command_from_slot_payload(self, slot: dict[str, object]) -> str | None:
        return build_pick_command_from_slot_payload(slot)

    def _build_pick_command_from_target(self, target: object) -> str | None:
        return build_pick_command_from_target(target)

    def _build_pick_command_from_mode_and_point(self, mode: str, point: object) -> str | None:
        return build_pick_command_from_mode_and_point(mode, point)

    def _resolve_vision_packet(self, packet: dict[str, object]) -> dict[str, object]:
        snapshot = self._fetch_remote_robot_snapshot()
        snapshot_age_ms = float(self._compute_remote_snapshot_age_ms())
        self._rt_set("vision_snapshot_age_ms", snapshot_age_ms)
        resolution = resolve_vision_packet(
            packet,
            config=self.config,
            snapshot=snapshot,
            snapshot_age_ms=snapshot_age_ms,
        )
        self._rt_update(
            {
                "vision_mapping_mode": resolution.mapping_mode,
                "vision_invalid_reason": resolution.first_invalid_reason,
                "vision_last_resolved_base_xy": resolution.first_resolved_base_xy,
                "vision_last_resolved_cyl": resolution.first_resolved_cyl,
            }
        )
        return resolution.packet

    def _start_ui_refresh_timer(self) -> None:
        timer = QTimer(self.main_window)
        timer.timeout.connect(self._refresh_view)
        timer.start(int(self.config.ui_panel_refresh_interval_ms))
        self.timers["ui-panels"] = timer

    def _start_remote_snapshot_poller(self) -> None:
        if self.config.robot_mode != "real":
            return
        if self._uses_ros_transport():
            if self._remote_snapshot_poller is not None:
                self._remote_snapshot_poller.stop()
                self._remote_snapshot_poller = None
            return
        if self._remote_snapshot_poller is not None:
            self._remote_snapshot_poller.stop()
        self._remote_snapshot_poller = RemoteSnapshotPoller(
            interval_ms=int(self.config.remote_snapshot_poll_interval_ms),
            fetch_snapshot=self._poll_remote_snapshot_once,
            on_snapshot=self._emit_polled_remote_snapshot,
        )
        self._remote_snapshot_poller.start()

    def _start_teleop_timer(self) -> None:
        timer = QTimer(self.main_window)
        timer.setTimerType(Qt.PreciseTimer)
        timer.timeout.connect(self._on_realtime_tick)
        interval_ms = int(self.config.teleop_repeat_interval_ms)
        if self.config.move_source == "mi":
            interval_ms = max(10, int(self.config.mi_poll_interval_ms))
        timer.start(interval_ms)
        self.timers["teleop-step"] = timer

    def _on_realtime_tick(self) -> None:
        self._pump_ros_reconnect()
        self._pump_input_sources()
        self._pump_teleop_command()
        self._check_pending_command_timeout()

    def _handle_runtime_status(self, component: str, message: str) -> None:
        self._log_runtime(component, message)
        if component == "vision":
            self._rt_set("vision_health", message)
        elif component == "robot":
            lowered = message.lower()
            if "send failed" in lowered:
                self._rt_set("robot_health", "send_failed")
            elif "connect failed" in lowered or "connect_failed" in lowered:
                self._rt_set("robot_health", "connect_failed")
            elif "start failed" in lowered or "start_failed" in lowered:
                self._rt_set("robot_health", "start_failed")
            elif "reconnecting" in lowered:
                self._rt_set("robot_health", "reconnecting")
        elif component == "ssvep":
            self._rt_set("ssvep_runtime_status", message)
            if "connected" in message.lower():
                self._rt_set("ssvep_connected", True)
        elif component == "mi":
            lowered = message.lower()
            if "error" in lowered or "failed" in lowered:
                self._rt_set("mi_last_error", message)
        self._refresh_view()

    def _log_runtime(self, component: str, message: str) -> None:
        self.logger.log_runtime_status(component, message)
        self.main_window.append_log(f"[{component}] {message}")

    def _is_move_token(self, token: str) -> bool:
        return str(token).strip().lower() in {"a", "d", "w", "s", "left", "right", "up", "down"}

    def _is_teleop_enabled(self) -> bool:
        return self.config.move_source == "sim"

    def _teleop_state_allows_motion(self) -> bool:
        if self._uses_ros_transport():
            if self.controller.state in {TaskState.S2_PICKING, TaskState.S3_PLACING, TaskState.ERROR}:
                return False
            return True
        return self.controller.state in {TaskState.S1_MI_MOVE, TaskState.S3_MI_CARRY}

    def _should_log_teleop_warning(self) -> bool:
        now = time.monotonic()
        if (now - float(self._last_teleop_warn_ts)) < 1.0:
            return False
        self._last_teleop_warn_ts = now
        return True

    def _handle_move_key_pressed(self, token: str) -> None:
        if not self._is_teleop_enabled():
            if self.config.move_source == "mi" and self._should_emit_mi_status():
                self._handle_runtime_status(
                    "mi",
                    "WASD ignored because move_source=mi (waiting for MI input provider events).",
                )
            return
        if not self._teleop_state_allows_motion():
            self._handle_runtime_status(
                "robot",
                f"WASD ignored in state={self.controller.state.value}. "
                "Current gate blocks only pick/place/error while ROS teleop is active.",
            )
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

    def _compute_teleop_rates(self) -> tuple[float, float]:
        theta_rate = 0.0
        radius_rate = 0.0
        pressed = self._pressed_move_tokens
        if "a" in pressed or "left" in pressed:
            theta_rate -= float(self.config.teleop_theta_rate_deg_s)
        if "d" in pressed or "right" in pressed:
            theta_rate += float(self.config.teleop_theta_rate_deg_s)
        if "w" in pressed or "up" in pressed:
            radius_rate += float(self.config.teleop_radius_rate_mm_s)
        if "s" in pressed or "down" in pressed:
            radius_rate -= float(self.config.teleop_radius_rate_mm_s)
        if theta_rate == 0.0 and radius_rate == 0.0:
            return (0.0, 0.0)
        return (theta_rate, radius_rate)

    def _current_robot_cyl_for_teleop(self) -> tuple[float, float] | None:
        snapshot = self._fetch_remote_robot_snapshot()
        if isinstance(snapshot, dict):
            robot_cyl = snapshot.get("robot_cyl")
            if isinstance(robot_cyl, dict):
                try:
                    return (float(robot_cyl.get("theta_deg", 0.0)), float(robot_cyl.get("radius_mm", 0.0)))
                except (TypeError, ValueError):
                    return None
        local_robot_cyl = self._rt_get("robot_cyl")
        if isinstance(local_robot_cyl, dict):
            try:
                return (float(local_robot_cyl.get("theta_deg", 0.0)), float(local_robot_cyl.get("radius_mm", 0.0)))
            except (TypeError, ValueError):
                return None
        return None

    def _maybe_send_ros_service_teleop_fallback(self, *, theta_rate: float, radius_rate: float) -> None:
        if self.ros_client is None or not self.ros_client.is_connected():
            return
        current_pose = self._current_robot_cyl_for_teleop()
        if current_pose is None:
            return
        plan = self._teleop_fallback.next_plan(
            current_pose=current_pose,
            theta_rate_deg_s=float(theta_rate),
            radius_rate_mm_s=float(radius_rate),
            now_monotonic=time.monotonic(),
            theta_limits_deg=self.config.robot_theta_limits_deg,
            radius_limits_mm=self.config.robot_auto_radius_limits_mm,
        )
        if plan is None:
            return

        if self._should_log_teleop_warning():
            self._queue_runtime_status(
                "robot",
                f"ROS teleop fallback step -> theta={plan.target_theta_deg:.1f} r={plan.target_radius_mm:.1f}",
            )

        def _callback(result: RosServiceResult) -> None:
            should_log_error = self._teleop_fallback.handle_service_result(
                ok=bool(result.ok),
                message=str(result.message or ""),
                now_monotonic=time.monotonic(),
            )
            if not should_log_error:
                return
            if self._should_log_teleop_warning():
                self._queue_runtime_status(
                    "robot",
                    f"ROS teleop fallback move failed: {str(result.message or '').strip() or 'unknown_error'}",
                )

        try:
            self.ros_client.send_move_cyl_auto(
                float(plan.target_theta_deg),
                float(plan.target_radius_mm),
                callback=_callback,
            )
        except Exception as error:
            self._teleop_fallback.handle_service_call_exception()
            if self._should_log_teleop_warning():
                self._handle_runtime_status("robot", f"ROS teleop fallback call failed: {error}")

    def _pump_ros_reconnect(self) -> None:
        if not self._uses_ros_transport():
            self._ros_last_connected = False
            self._ros_reconnect_attempt = 0
            self._ros_reconnect_next_ts = 0.0
            self._ros_connected_since_ts = 0.0
            return
        if self.ros_client is None:
            return
        connected = bool(self.ros_client.is_connected())
        if connected:
            if not self._ros_last_connected:
                self._ros_reconnect_attempt = 0
                self._ros_reconnect_next_ts = 0.0
                self._ros_stale_detection_count = 0
                try:
                    self.ros_client.stop_teleop()
                except Exception:
                    pass
                self._queue_runtime_status("robot", "ROS bridge reconnected.")
            if self._ros_connected_since_ts <= 0.0:
                self._ros_connected_since_ts = time.monotonic()
            self._ros_last_connected = True
            return
        self._ros_last_connected = False
        self._ros_connected_since_ts = 0.0
        self._ros_stale_detection_count = 0
        now = time.monotonic()
        if now < float(self._ros_reconnect_next_ts):
            return
        try:
            self.ros_client.connect()
        except Exception as error:
            self._ros_reconnect_attempt += 1
            backoff = min(
                float(self.config.ros_reconnect_max_delay_sec),
                float(self.config.ros_reconnect_base_delay_sec) * (2 ** max(0, self._ros_reconnect_attempt - 1)),
            )
            jitter_scale = float(self.config.ros_reconnect_jitter_ratio)
            jitter = random.uniform(-1.0, 1.0) * backoff * max(0.0, jitter_scale)
            delay = max(0.2, backoff + jitter)
            self._ros_reconnect_next_ts = now + delay
            if self._should_log_teleop_warning():
                self._queue_runtime_status(
                    "robot",
                    "ROS reconnect failed (attempt {0}), retry in {1:.2f}s: {2}".format(
                        int(self._ros_reconnect_attempt),
                        float(delay),
                        str(error),
                    ),
                )
            return
        self._ros_reconnect_next_ts = now + max(0.2, float(self.config.ros_reconnect_base_delay_sec))

    def _pump_input_sources(self) -> None:
        result = self.input_orchestrator.poll()
        self._update_mi_runtime_state(result.mi_health, emit_runtime_status=True)
        for event in result.events:
            self.dispatch_event(event)

    def _pump_teleop_command(self) -> None:
        if not self._is_teleop_enabled():
            return
        if not self._teleop_state_allows_motion():
            self._stop_teleop_motion(send_command=False, reason="not_in_motion_state")
            return
        if self._uses_ros_transport():
            theta_rate, radius_rate = self._compute_teleop_rates()
            if self.ros_client is None or not self.ros_client.is_connected():
                if self._should_log_teleop_warning():
                    self._handle_runtime_status("robot", "ROS teleop skipped: rosbridge is not connected.")
                self._teleop_ros_planner.reset()
                return
            command = self._teleop_ros_planner.next_command(
                theta_rate_deg_s=theta_rate,
                radius_rate_mm_s=radius_rate,
                now_monotonic=time.monotonic(),
            )
            if command is None:
                return
            self._teleop_cmd_seq += 1
            try:
                self.ros_client.publish_teleop(
                    theta_rate_deg_s=float(command.theta_rate_deg_s),
                    radius_rate_mm_s=float(command.radius_rate_mm_s),
                    enabled=bool(command.enabled),
                    cmd_seq=int(self._teleop_cmd_seq),
                    client_ts=float(time.time()),
                )
            except Exception as error:
                self._teleop_ros_planner.on_publish_failed()
                self._handle_runtime_status("robot", f"ROS teleop publish failed: {error}")
                return
            if (
                bool(command.enabled)
                and bool(getattr(self.config, "teleop_ros_service_fallback_enabled", False))
            ):
                self._maybe_send_ros_service_teleop_fallback(theta_rate=theta_rate, radius_rate=radius_rate)
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
        self._teleop_ros_planner.reset()
        self._teleop_fallback.reset()
        if self._uses_ros_transport() and self.ros_client is not None:
            try:
                self.ros_client.stop_teleop()
            except Exception as error:
                self._handle_runtime_status("robot", f"ROS teleop stop failed: {error}")

    def _capture_world_snapshot(self, *, reason: str, force: bool = False) -> None:
        snapshot = None
        preflight_snapshot = None
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
        if self.config.robot_mode != "real":
            return None
        with self._remote_snapshot_lock:
            if self._remote_snapshot_cache is None:
                return None
            return dict(self._remote_snapshot_cache)

    def _compute_remote_snapshot_age_ms(self) -> float:
        if self.config.robot_mode != "real":
            return 0.0
        with self._remote_snapshot_lock:
            envelope = self._remote_snapshot_envelope
        if envelope is None:
            return float("inf")
        return self._compute_envelope_age_ms(envelope)

    @staticmethod
    def _compute_envelope_age_ms(envelope: RobotSnapshotEnvelope) -> float:
        now_ts = float(time.time())
        recv_age_ms = max(0.0, (now_ts - float(envelope.ts)) * 1000.0)
        payload = envelope.payload if isinstance(envelope.payload, dict) else None
        if payload is None:
            return recv_age_ms
        try:
            robot_ts = float(payload.get("robot_ts", 0.0) or 0.0)
        except (TypeError, ValueError):
            return recv_age_ms
        if not math.isfinite(robot_ts) or robot_ts <= 0.0:
            return recv_age_ms
        robot_age_ms = max(0.0, (now_ts - robot_ts) * 1000.0)
        # Guard against unsynchronized clocks between desktop and robot.
        if robot_age_ms > 60000.0:
            return recv_age_ms
        return max(recv_age_ms, robot_age_ms)

    def _fetch_remote_robot_snapshot_direct(self) -> dict[str, object] | None:
        if self.config.robot_mode != "real":
            return None
        if self._uses_ros_transport():
            return self._fetch_remote_robot_snapshot()
        try:
            return fetch_robot_status(
                self.config.robot_host,
                self.config.robot_port,
                timeout_sec=max(self.config.robot_timeout_sec, 0.2),
            )
        except Exception:
            return None

    def _fetch_vision_calibration_params(self) -> dict[str, object] | None:
        if not self._uses_ros_transport() or self.ros_client is None:
            return None
        try:
            params = self.ros_client.get_param("/camera_cal/block_params", timeout_sec=max(self.config.rosbridge_timeout_sec, 1.0))
        except Exception as error:
            self._log_runtime("vision", f"Calibration fetch failed: {error}")
            return None
        if not isinstance(params, dict):
            self._log_runtime("vision", "Calibration fetch returned non-dict payload.")
            return None
        return params

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
            "state_seq",
            "robot_ts",
            "action_phase",
            "last_ack",
            "last_error",
            "last_error_code",
            "calibration_ready",
            "carrying",
            "pick_tuning",
            "post_pick_settle_z",
            "release_mode_effective",
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
        self._rt_update(
            {
                "calibration_ready": snapshot.get("calibration_ready"),
                "robot_cyl": snapshot.get("robot_cyl"),
                "limits_cyl": snapshot.get("limits_cyl"),
                "auto_z_current": snapshot.get("auto_z_current"),
                "control_kernel": snapshot.get("control_kernel"),
                "state_seq": snapshot.get("state_seq"),
                "robot_ts": snapshot.get("robot_ts"),
            }
        )
        last_ack = str(snapshot.get("last_ack", "")).strip()
        if last_ack:
            self._rt_set("last_robot_ack", last_ack)
        if isinstance(snapshot.get("pick_tuning"), dict) and not bool(self._pick_tuning_local_dirty):
            self._pick_tuning_state = self._sanitize_pick_tuning(snapshot.get("pick_tuning"))
            self._rt_set("pick_tuning", dict(self._pick_tuning_state))
        self._rt_update(
            {
                "post_pick_settle_z": snapshot.get("post_pick_settle_z"),
                "release_mode_effective": snapshot.get("release_mode_effective"),
            }
        )
        if snapshot.get("state") == "ERROR":
            self.controller.context.last_error = str(snapshot.get("last_error") or self.controller.context.last_error)

    def _queue_event(self, event: Event) -> None:
        if self._shutdown_started:
            return
        try:
            self._bridge.event_received.emit(event)
        except RuntimeError:
            return

    def _queue_runtime_status(self, component: str, message: str) -> None:
        if self._shutdown_started:
            return
        try:
            self._bridge.runtime_status_received.emit(str(component), str(message))
        except RuntimeError:
            return

    def _queue_ssvep_state(self, payload: dict[str, object]) -> None:
        if self._shutdown_started:
            return
        try:
            self._bridge.ssvep_state_received.emit(dict(payload))
        except RuntimeError:
            return

    def _queue_vision_packet(self, packet: dict[str, object]) -> None:
        if self._shutdown_started:
            return
        try:
            self._bridge.vision_packet_received.emit(packet)
        except RuntimeError:
            return

    def _queue_vision_frame(self, frame) -> None:
        if self._shutdown_started:
            return
        should_emit = False
        with self._vision_frame_lock:
            self._vision_frame_staging = frame
            if not self._vision_frame_signal_pending:
                self._vision_frame_signal_pending = True
                should_emit = True
        if not should_emit:
            return
        try:
            self._bridge.vision_frame_received.emit(None)
        except RuntimeError:
            with self._vision_frame_lock:
                self._vision_frame_signal_pending = False
            return

    def _on_vision_packet_received(self, packet: object) -> None:
        if not isinstance(packet, dict):
            return
        resolved_packet = self._resolve_vision_packet(packet)
        self._latest_vision_packet = resolved_packet
        targets = packet_to_targets(resolved_packet)
        self.dispatch_event(Event(source="vision", type="vision_update", value=targets))
        valid_slots = sum(1 for slot in resolved_packet.get("slots", []) if slot.get("valid"))
        actionable_slots = sum(1 for slot in resolved_packet.get("slots", []) if slot.get("actionable"))
        queue_age_ms = float(resolved_packet.get("queue_age_ms", 0.0))
        infer_interval_ms = float(resolved_packet.get("infer_interval_ms", self.config.vision_infer_interval_ms))
        frame_drop_ratio = float(resolved_packet.get("frame_drop_ratio", 0.0))
        self._rt_update(
            {
                "queue_age_ms": queue_age_ms,
                "infer_interval_ms": infer_interval_ms,
                "frame_drop_ratio": frame_drop_ratio,
            }
        )
        remote_age_ms = float(self._rt_get("remote_snapshot_age_ms", 0.0))
        ui_refresh_ms = float(self._rt_get("ui_refresh_ms_ema", 0.0))
        mapping_mode = str(resolved_packet.get("mapping_mode", self.config.vision_mapping_mode))
        invalid_reason = str(self._rt_get("vision_invalid_reason", "--"))
        resolved_base_xy = self._rt_get("vision_last_resolved_base_xy")
        resolved_cyl = self._rt_get("vision_last_resolved_cyl")
        vision_health = (
            f"camera_fps={float(packet.get('capture_fps', 0.0)):.1f} "
            f"infer_ms={float(packet.get('infer_ms', 0.0)):.1f} "
            f"queue_age_ms={queue_age_ms:.1f} "
            f"infer_interval_ms={infer_interval_ms:.0f} "
            f"drop_ratio={frame_drop_ratio:.2f} "
            f"ui_refresh_ms={ui_refresh_ms:.1f} "
            f"snapshot_age_ms={remote_age_ms:.1f} "
            f"slots={valid_slots} actionable={actionable_slots} "
            f"mapping={mapping_mode} invalid={invalid_reason} "
            f"resolved_xy={resolved_base_xy} resolved_cyl={resolved_cyl}"
        )
        self._rt_set("vision_health", vision_health)
        self.main_window.update_vision_payload(
            packet=resolved_packet,
            flash_enabled=bool(self._rt_get("ssvep_stim_enabled", False)),
            status_text=str(self._rt_get("vision_health", "unknown")),
        )

    def _on_vision_frame_received(self, frame: object) -> None:
        del frame
        with self._vision_frame_lock:
            self._latest_vision_frame = self._vision_frame_staging
            self._vision_frame_staging = None
            self._vision_frame_signal_pending = False
        self.main_window.update_vision_payload(
            frame_bgr=self._latest_vision_frame,
            flash_enabled=bool(self._rt_get("ssvep_stim_enabled", False)),
            status_text=str(self._rt_get("vision_health", "unknown")),
        )

    def _on_ssvep_state_received(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        payload_type = str(payload.get("type", "")).strip().lower()
        if payload_type == "device_connected":
            self._rt_set("ssvep_connected", True)
        elif payload_type == "runtime_error":
            self._rt_update(
                {
                    "ssvep_last_error": str(payload.get("message", "--")),
                    "ssvep_runtime_status": "error",
                    "ssvep_running": False,
                }
            )
        elif payload_type == "profile_ready":
            profile_update: dict[str, object] = {
                "ssvep_profile_source": str(payload.get("profile_source", "trained"))
            }
            current_path = payload.get("current_profile_path") or payload.get("profile_path")
            if current_path:
                profile_update["ssvep_profile_path"] = str(current_path)
            summary_text = str(payload.get("summary_text", "")).strip()
            profile_update["ssvep_runtime_status"] = (
                f"pretrain_ready {summary_text}".strip()
                if summary_text
                else "pretrain_ready"
            )
            self._rt_update(profile_update)
        elif payload_type == "profile_loaded":
            profile_update = {
                "ssvep_profile_source": str(payload.get("profile_source", "loaded")),
            }
            profile_path = payload.get("profile_path")
            if profile_path:
                profile_update["ssvep_profile_path"] = str(profile_path)
            self._rt_update(profile_update)
        elif payload_type == "runtime_state":
            state_update: dict[str, object] = {
                "ssvep_running": bool(payload.get("running", False)),
                "ssvep_busy": bool(payload.get("busy", False)),
                "ssvep_connected": bool(payload.get("connected", False)),
                "ssvep_connect_active": bool(payload.get("connect_active", False)),
                "ssvep_pretrain_active": bool(payload.get("pretrain_active", False)),
                "ssvep_online_active": bool(payload.get("online_active", False)),
                "ssvep_mode": str(payload.get("mode", self._rt_get("ssvep_mode", "idle"))),
                "ssvep_profile_source": str(
                    payload.get("profile_source", self._rt_get("ssvep_profile_source", "fallback"))
                ),
            }
            profile_path = payload.get("profile_path")
            if profile_path:
                state_update["ssvep_profile_path"] = str(profile_path)
            state_update["ssvep_last_pretrain_time"] = payload.get("last_pretrain_time") or "--"
            state_update["ssvep_runtime_status"] = "running" if payload.get("running") else "stopped"
            state_update["ssvep_latest_profile_path"] = str(payload.get("latest_profile_path", "--"))
            state_update["ssvep_profile_count"] = int(payload.get("profile_count", 0))
            state_update["ssvep_available_profiles"] = tuple(
                (str(item.get("display_name", item.get("name", ""))), str(item.get("path", "")))
                for item in payload.get("available_profiles", [])
                if isinstance(item, dict)
            )
            state_update["ssvep_allow_fallback_profile"] = bool(
                payload.get("allow_fallback_profile", self.config.ssvep_allow_fallback_profile)
            )
            state_update["ssvep_status_hint"] = str(payload.get("status_hint", "--"))
            state_update["ssvep_last_error"] = payload.get("last_error") or "--"
            self._rt_update(state_update)
            last_result = payload.get("last_result")
            if isinstance(last_result, dict):
                self._apply_ssvep_online_result(last_result)
        elif payload_type == "online_result":
            result_payload = payload.get("payload")
            if isinstance(result_payload, dict):
                self._apply_ssvep_online_result(result_payload)
        elif payload_type == "pretrain_phase":
            phase_payload = payload.get("payload")
            if isinstance(phase_payload, dict):
                self._rt_set("ssvep_runtime_status", str(phase_payload.get("title", "pretrain")))
        elif payload_type == "online_phase":
            phase_payload = payload.get("payload")
            if isinstance(phase_payload, dict):
                self._rt_set("ssvep_runtime_status", str(phase_payload.get("title", "online")))
        self._refresh_view()

    def _apply_ssvep_online_result(self, payload: dict[str, object]) -> None:
        state = str(payload.get("state", "--"))
        selected_freq = payload.get("selected_freq")
        selected_text = (
            "--" if selected_freq is None else f"{float(selected_freq):g}Hz"
        )
        margin_text = f"{float(payload.get('margin', 0.0)):.4f}"
        ratio_text = f"{float(payload.get('ratio', 0.0)):.4f}"
        stable_text = str(int(payload.get("stable_windows", 0)))
        pred_freq = payload.get("pred_freq")
        last_raw = (
            "state={state} pred={pred} selected={selected} margin={margin} ratio={ratio} stable={stable}".format(
                state=state,
                pred="--" if pred_freq is None else f"{float(pred_freq):g}Hz",
                selected=selected_text,
                margin=margin_text,
                ratio=ratio_text,
                stable=stable_text,
            )
        )
        self._rt_update(
            {
                "ssvep_last_state": state,
                "ssvep_last_selected_freq": selected_text,
                "ssvep_last_margin": margin_text,
                "ssvep_last_ratio": ratio_text,
                "ssvep_last_stable_windows": stable_text,
                "last_ssvep_raw": last_raw,
            }
        )

    def _on_ssvep_connect_requested(self) -> None:
        if self.ssvep_runtime is None:
            return
        self.ssvep_coordinator.connect_device()

    def _on_ssvep_pretrain_requested(self) -> None:
        if self.ssvep_runtime is None:
            return
        self.ssvep_coordinator.start_pretrain()

    def _on_ssvep_load_profile_requested(self) -> None:
        if self.ssvep_runtime is None:
            return
        if self.main_window.is_ssvep_profile_auto_selected():
            self.ssvep_coordinator.clear_session_profile()
            return
        path = self.main_window.selected_ssvep_profile_path()
        if not path:
            start_dir = str(self.config.ssvep_profile_dir.resolve())
            path, _ = QFileDialog.getOpenFileName(
                self.main_window,
                "Load SSVEP Profile",
                start_dir,
                "JSON Files (*.json);;All Files (*)",
            )
        if not path:
            return
        try:
            self.ssvep_coordinator.load_profile(path)
        except Exception as error:
            self._handle_runtime_status("ssvep", f"Load profile failed: {error}")

    def _on_ssvep_open_profile_dir_requested(self) -> None:
        profile_dir = self.config.ssvep_profile_dir.resolve()
        profile_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(profile_dir))  # type: ignore[attr-defined]
        except Exception as error:
            self._handle_runtime_status("ssvep", f"Open profile dir failed: {error}")

    def _on_ssvep_start_requested(self) -> None:
        if self.ssvep_runtime is None:
            return
        self.ssvep_coordinator.start_online()

    def _on_ssvep_stop_requested(self) -> None:
        if self.ssvep_runtime is None:
            return
        self._rt_update(
            {
                "ssvep_running": False,
                "ssvep_mode": "idle",
                "ssvep_runtime_status": "stopping",
            }
        )
        self._refresh_view()
        self.ssvep_coordinator.stop_online()

    def _on_ssvep_stim_toggled(self, enabled: bool) -> None:
        self._rt_set("ssvep_stim_enabled", bool(enabled))
        self.main_window.update_vision_payload(
            flash_enabled=bool(enabled),
            status_text=str(self._rt_get("vision_health", "unknown")),
        )
        self._refresh_view()

    def _request_remote_snapshot(self) -> None:
        poller = self._remote_snapshot_poller
        if poller is None:
            return
        poller.request_now()

    def _poll_remote_snapshot_once(self) -> RobotSnapshotEnvelope:
        snapshot: dict[str, object] | None = None
        error_text = ""
        transport = "ros" if self._uses_ros_transport() else "tcp"
        try:
            if self._uses_ros_transport():
                if self.ros_client is not None:
                    snapshot = self.ros_client.latest_state_snapshot()
            else:
                snapshot = self._fetch_remote_robot_snapshot_direct()
        except Exception as error:
            snapshot = None
            error_text = str(error)
        if snapshot is None and not error_text:
            error_text = "snapshot_unavailable"
        return RobotSnapshotEnvelope(
            payload=None if snapshot is None else dict(snapshot),
            ts=time.time(),
            transport=transport,
            ok=snapshot is not None,
            error=error_text,
        )

    def _emit_polled_remote_snapshot(self, envelope: RobotSnapshotEnvelope) -> None:
        if self._shutdown_started:
            return
        try:
            self._bridge.remote_snapshot_received.emit(envelope)
        except RuntimeError:
            return

    def _on_remote_snapshot_received(self, snapshot: object) -> None:
        if isinstance(snapshot, RobotSnapshotEnvelope):
            envelope = snapshot
        elif isinstance(snapshot, dict):
            envelope = RobotSnapshotEnvelope(
                payload=dict(snapshot),
                ts=time.time(),
                transport="legacy",
                ok=True,
                error="",
            )
        else:
            envelope = RobotSnapshotEnvelope(
                payload=None,
                ts=time.time(),
                transport="unknown",
                ok=False,
                error="invalid_envelope",
            )
        payload = envelope.payload if isinstance(envelope.payload, dict) else None
        with self._remote_snapshot_lock:
            self._remote_snapshot_envelope = envelope
            if payload is not None:
                self._remote_snapshot_cache = dict(payload)
        state_age_ms = self._compute_envelope_age_ms(envelope)
        self._rt_set("state_age_ms", float(state_age_ms))
        self._rt_set("remote_snapshot_age_ms", float(state_age_ms))
        if isinstance(payload, dict):
            last_ack = str(payload.get("last_ack", "")).strip()
            if last_ack:
                self._rt_set("last_robot_ack", last_ack)
        connected = self.ros_client.is_connected() if self._uses_ros_transport() and self.ros_client is not None else self.robot_client.is_connected()
        self._rt_set("robot_connected", connected)
        if envelope.ok and payload is not None:
            self._ros_stale_detection_count = 0
            self._rt_set("robot_health", "ok")
            return
        if envelope.error:
            self._rt_set("robot_health", f"snapshot_error:{envelope.error}")

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
            "limits_cyl_auto": {
                "theta_deg": self.config.robot_theta_limits_deg,
                "radius_mm": self.config.robot_auto_radius_limits_mm,
                "z_mm": self.config.robot_height_limits_mm,
            },
            "travel_z": self.config.robot_travel_z,
            "approach_z": self.config.robot_approach_z,
            "pick_z": self.config.robot_pick_z,
            "carry_z": self.config.robot_carry_z,
            "auto_z_enabled": True,
            "auto_z_current": self.controller.context.robot_auto_z,
            "control_kernel": self._rt_get("control_kernel") or "cylindrical_kernel",
            "busy_action": self._scene_busy_action(),
            "busy": self._scene_busy_action() is not None,
            "state": self.controller.state.value,
            "action_phase": self.controller.state.value,
            "carrying": self._scene_carrying_target_id is not None,
            "calibration_ready": False if self.config.vision_mode == "fixed_world_slots" else True,
            "carrying_target_id": self._scene_carrying_target_id,
            "last_ack": self._rt_get("last_robot_ack"),
            "last_error": self._scene_last_error or self._rt_get("last_robot_error"),
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
        control_sim_enabled=True,
        sim_process_mode="external_lab",
        slot_profile=args.slot_profile,
        simulation_enabled=False,
        timing_profile=args.timing_profile,
        scenario_name=args.scenario_name,
        robot_mode=args.robot_mode,
        vision_mode=args.vision_mode,
        move_source=args.move_source,
        decision_source=args.decision_source,
        mi_backend=str(getattr(args, "mi_backend", AppConfig.mi_backend)),
        mi_enabled=bool(getattr(args, "mi_enabled", AppConfig.mi_enabled)),
        mi_poll_interval_ms=int(getattr(args, "mi_poll_interval_ms", AppConfig.mi_poll_interval_ms)),
        mi_command_cooldown_ms=int(
            getattr(args, "mi_command_cooldown_ms", AppConfig.mi_command_cooldown_ms)
        ),
        robot_host=args.robot_host,
        robot_port=args.robot_port,
        robot_transport=getattr(args, "robot_transport", AppConfig.robot_transport),
        rosbridge_port=getattr(args, "rosbridge_port", AppConfig.rosbridge_port),
        ros_reconnect_base_delay_sec=float(
            getattr(args, "ros_reconnect_base_delay_sec", AppConfig.ros_reconnect_base_delay_sec)
        ),
        ros_reconnect_max_delay_sec=float(
            getattr(args, "ros_reconnect_max_delay_sec", AppConfig.ros_reconnect_max_delay_sec)
        ),
        ros_reconnect_jitter_ratio=float(
            getattr(args, "ros_reconnect_jitter_ratio", AppConfig.ros_reconnect_jitter_ratio)
        ),
        ros_runtime_probe_timeout_sec=float(
            getattr(args, "ros_runtime_probe_timeout_sec", AppConfig.ros_runtime_probe_timeout_sec)
        ),
        ros_runtime_state_grace_sec=float(
            getattr(args, "ros_runtime_state_grace_sec", AppConfig.ros_runtime_state_grace_sec)
        ),
        robot_auto_start_on_ros_unavailable=bool(
            getattr(args, "robot_auto_start_on_ros_unavailable", AppConfig.robot_auto_start_on_ros_unavailable)
        ),
        robot_auto_restart_on_state_stale=bool(
            getattr(args, "robot_auto_restart_on_state_stale", AppConfig.robot_auto_restart_on_state_stale)
        ),
        robot_auto_start_cooldown_sec=float(
            getattr(args, "robot_auto_start_cooldown_sec", AppConfig.robot_auto_start_cooldown_sec)
        ),
        robot_command_timeout_sec=float(
            getattr(args, "robot_command_timeout_sec", AppConfig.robot_command_timeout_sec)
        ),
        robot_state_stale_threshold_ms=float(
            getattr(args, "robot_state_stale_threshold_ms", AppConfig.robot_state_stale_threshold_ms)
        ),
        vision_stream_url=args.vision_stream_url,
        vision_world_scale_xy=float(getattr(args, "vision_world_scale_xy", AppConfig.vision_world_scale_xy)),
        vision_world_offset_xy_mm=(
            float(getattr(args, "vision_world_offset_x_mm", AppConfig.vision_world_offset_xy_mm[0])),
            float(getattr(args, "vision_world_offset_y_mm", AppConfig.vision_world_offset_xy_mm[1])),
        ),
        vision_mapping_mode=str(getattr(args, "vision_mapping_mode", AppConfig.vision_mapping_mode)),
        vision_target_frame=str(getattr(args, "vision_target_frame", AppConfig.vision_target_frame)),
        vision_snapshot_max_age_ms=float(
            getattr(args, "vision_snapshot_max_age_ms", AppConfig.vision_snapshot_max_age_ms)
        ),
        vision_action_requires_calibration=bool(
            getattr(args, "vision_action_requires_calibration", AppConfig.vision_action_requires_calibration)
        ),
        pick_cyl_radius_bias_mm=float(
            getattr(args, "pick_cyl_radius_bias_mm", AppConfig.pick_cyl_radius_bias_mm)
        ),
        pick_cyl_theta_bias_deg=float(
            getattr(args, "pick_cyl_theta_bias_deg", AppConfig.pick_cyl_theta_bias_deg)
        ),
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
    parser.add_argument("--robot-mode", choices=("real",), default="real")
    parser.add_argument("--robot-transport", choices=("tcp", "ros"), default=AppConfig.robot_transport)
    parser.add_argument(
        "--vision-mode",
        choices=("slots", "real", "robot_camera_detection", "fixed_world_slots", "fixed_cyl_slots"),
        default="robot_camera_detection",
    )
    parser.add_argument("--move-source", choices=("sim", "mi"), default="sim")
    parser.add_argument("--decision-source", choices=("sim", "ssvep"), default="sim")
    parser.add_argument("--mi-backend", choices=("brainflow",), default=AppConfig.mi_backend)
    parser.add_argument("--mi-enabled", action="store_true", default=AppConfig.mi_enabled)
    parser.add_argument("--mi-disabled", action="store_false", dest="mi_enabled")
    parser.add_argument("--mi-poll-interval-ms", type=int, default=AppConfig.mi_poll_interval_ms)
    parser.add_argument("--mi-command-cooldown-ms", type=int, default=AppConfig.mi_command_cooldown_ms)
    parser.add_argument("--timing-profile", choices=("formal", "fast"), default="formal")
    parser.add_argument("--scenario-name", default="basic")
    parser.add_argument("--slot-profile", default="default")
    parser.add_argument("--robot-host", default=AppConfig.robot_host)
    parser.add_argument("--robot-port", type=int, default=AppConfig.robot_port)
    parser.add_argument("--rosbridge-port", type=int, default=AppConfig.rosbridge_port)
    parser.add_argument(
        "--ros-reconnect-base-delay-sec",
        type=float,
        default=AppConfig.ros_reconnect_base_delay_sec,
    )
    parser.add_argument(
        "--ros-reconnect-max-delay-sec",
        type=float,
        default=AppConfig.ros_reconnect_max_delay_sec,
    )
    parser.add_argument(
        "--ros-reconnect-jitter-ratio",
        type=float,
        default=AppConfig.ros_reconnect_jitter_ratio,
    )
    parser.add_argument(
        "--ros-runtime-probe-timeout-sec",
        type=float,
        default=AppConfig.ros_runtime_probe_timeout_sec,
    )
    parser.add_argument(
        "--ros-runtime-state-grace-sec",
        type=float,
        default=AppConfig.ros_runtime_state_grace_sec,
    )
    parser.add_argument(
        "--robot-auto-start-on-ros-unavailable",
        action="store_true",
        default=AppConfig.robot_auto_start_on_ros_unavailable,
    )
    parser.add_argument(
        "--robot-auto-start-disabled",
        action="store_false",
        dest="robot_auto_start_on_ros_unavailable",
    )
    parser.add_argument(
        "--robot-auto-restart-on-state-stale",
        action="store_true",
        default=AppConfig.robot_auto_restart_on_state_stale,
    )
    parser.add_argument(
        "--robot-auto-restart-on-state-stale-disabled",
        action="store_false",
        dest="robot_auto_restart_on_state_stale",
    )
    parser.add_argument(
        "--robot-auto-start-cooldown-sec",
        type=float,
        default=AppConfig.robot_auto_start_cooldown_sec,
    )
    parser.add_argument(
        "--robot-command-timeout-sec",
        type=float,
        default=AppConfig.robot_command_timeout_sec,
    )
    parser.add_argument(
        "--robot-state-stale-threshold-ms",
        type=float,
        default=AppConfig.robot_state_stale_threshold_ms,
    )
    parser.add_argument("--vision-stream-url", default="")
    parser.add_argument("--vision-world-scale-xy", type=float, default=AppConfig.vision_world_scale_xy)
    parser.add_argument("--vision-world-offset-x-mm", type=float, default=AppConfig.vision_world_offset_xy_mm[0])
    parser.add_argument("--vision-world-offset-y-mm", type=float, default=AppConfig.vision_world_offset_xy_mm[1])
    parser.add_argument(
        "--vision-mapping-mode",
        choices=("delta_servo", "absolute_base"),
        default=AppConfig.vision_mapping_mode,
    )
    parser.add_argument("--vision-target-frame", default=AppConfig.vision_target_frame)
    parser.add_argument(
        "--vision-snapshot-max-age-ms",
        type=float,
        default=AppConfig.vision_snapshot_max_age_ms,
    )
    parser.add_argument(
        "--vision-action-requires-calibration",
        action="store_true",
        default=AppConfig.vision_action_requires_calibration,
    )
    parser.add_argument(
        "--vision-action-allows-no-calibration",
        action="store_false",
        dest="vision_action_requires_calibration",
    )
    parser.add_argument("--pick-cyl-radius-bias-mm", type=float, default=AppConfig.pick_cyl_radius_bias_mm)
    parser.add_argument("--pick-cyl-theta-bias-deg", type=float, default=AppConfig.pick_cyl_theta_bias_deg)
    parser.add_argument("--stage-motion-sec", type=float, default=None)
    parser.add_argument("--continue-motion-sec", type=float, default=None)
    parser.add_argument("--smoke-test-ms", type=int, default=0, help="Auto quit after N ms for smoke tests.")
    args = parser.parse_args(argv)

    config = build_config_from_args(args)
    qt_app = QApplication(sys.argv if argv is None else argv)
    runtime = HybridControllerApplication(config)
    runtime.main_window.showMaximized()
    if args.smoke_test_ms > 0:
        QTimer.singleShot(args.smoke_test_ms, lambda: (runtime.shutdown(), qt_app.quit()))
    exit_code = 0
    try:
        exit_code = qt_app.exec_()
    finally:
        runtime.shutdown()
    return int(exit_code)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
