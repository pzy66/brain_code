from __future__ import annotations

from dataclasses import replace
from typing import Any

from hybrid_controller.config import AppConfig
from hybrid_controller.snapshot import (
    AppSnapshot,
    RobotPanelState,
    RobotRuntimeState,
    SsvEpPanelState,
    SsvEpRuntimeState,
    VisionPanelState,
    VisionRuntimeState,
)


class RobotCoordinator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._state = RobotRuntimeState(
            health="unknown",
            control_kernel="cylindrical_kernel",
        )
        self._sender = None

    def get_state(self) -> RobotRuntimeState:
        return replace(self._state)

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if not hasattr(self._state, key):
                raise AttributeError(f"Unknown robot state field: {key}")
            setattr(self._state, key, value)

    def apply_remote_snapshot(self, snapshot: dict[str, object] | None) -> None:
        self._state.remote_snapshot = None if snapshot is None else dict(snapshot)
        if snapshot is None:
            return
        self._state.calibration_ready = snapshot.get("calibration_ready")  # type: ignore[assignment]
        robot_cyl = snapshot.get("robot_cyl")
        if isinstance(robot_cyl, dict):
            self._state.robot_cyl = dict(robot_cyl)
        limits_cyl = snapshot.get("limits_cyl")
        if isinstance(limits_cyl, dict):
            self._state.limits_cyl = dict(limits_cyl)
        limits_cyl_auto = snapshot.get("limits_cyl_auto")
        if isinstance(limits_cyl_auto, dict):
            self._state.limits_cyl_auto = dict(limits_cyl_auto)
        auto_z_current = snapshot.get("auto_z_current")
        self._state.auto_z_current = None if auto_z_current is None else float(auto_z_current)
        control_kernel = snapshot.get("control_kernel")
        if control_kernel is not None:
            self._state.control_kernel = str(control_kernel)
        last_error = snapshot.get("last_error")
        if last_error:
            self._state.last_error = str(last_error)

    def set_scene_snapshot(self, snapshot: dict[str, object] | None) -> None:
        self._state.scene_snapshot = None if snapshot is None else dict(snapshot)

    def configure_sender(self, sender) -> None:
        self._sender = sender

    def send_text(self, command: str) -> None:
        if self._sender is None:
            raise RuntimeError("RobotCoordinator sender is not configured.")
        self._sender(str(command))

    def send_move_cyl(self, theta_deg: float, radius_mm: float, z_mm: float) -> None:
        self.send_text(f"MOVE_CYL {float(theta_deg):.2f} {float(radius_mm):.2f} {float(z_mm):.2f}")

    def send_move_cyl_auto(self, theta_deg: float, radius_mm: float) -> None:
        self.send_text(f"MOVE_CYL_AUTO {float(theta_deg):.2f} {float(radius_mm):.2f}")

    def send_pick_cyl(self, theta_deg: float, radius_mm: float) -> None:
        self.send_text(f"PICK_CYL {float(theta_deg):.2f} {float(radius_mm):.2f}")

    def send_place(self) -> None:
        self.send_text("PLACE")

    def send_abort(self) -> None:
        self.send_text("ABORT")

    def send_reset(self) -> None:
        self.send_text("RESET")


class VisionCoordinator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._state = VisionRuntimeState(mode=config.vision_mode, health="unknown")

    def get_state(self) -> VisionRuntimeState:
        return replace(self._state)

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if not hasattr(self._state, key):
                raise AttributeError(f"Unknown vision state field: {key}")
            setattr(self._state, key, value)


class SSVEPCoordinator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._state = SsvEpRuntimeState(
            profile_path=str(config.ssvep_current_profile_path),
            model_name=config.ssvep_model_name,
            debug_keyboard=config.ssvep_keyboard_debug_enabled,
            allow_fallback_profile=config.ssvep_allow_fallback_profile,
        )
        self._runtime = None

    def get_state(self) -> SsvEpRuntimeState:
        return replace(self._state)

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if not hasattr(self._state, key):
                raise AttributeError(f"Unknown SSVEP state field: {key}")
            setattr(self._state, key, value)

    def bind_runtime(self, runtime) -> None:
        self._runtime = runtime

    def connect_device(self) -> None:
        if self._runtime is None:
            raise RuntimeError("SSVEPCoordinator runtime is not bound.")
        self._runtime.connect_device()

    def start_pretrain(self) -> None:
        if self._runtime is None:
            raise RuntimeError("SSVEPCoordinator runtime is not bound.")
        self._runtime.start_pretrain()

    def load_profile(self, path: str) -> None:
        if self._runtime is None:
            raise RuntimeError("SSVEPCoordinator runtime is not bound.")
        self._runtime.load_profile_from_path(path)

    def clear_session_profile(self) -> None:
        if self._runtime is None:
            raise RuntimeError("SSVEPCoordinator runtime is not bound.")
        self._runtime.clear_session_profile()

    def start_online(self) -> None:
        if self._runtime is None:
            raise RuntimeError("SSVEPCoordinator runtime is not bound.")
        self._runtime.start_online()

    def stop_online(self) -> None:
        if self._runtime is None:
            raise RuntimeError("SSVEPCoordinator runtime is not bound.")
        self._runtime.stop_online()

    def stop(self) -> None:
        if self._runtime is None:
            raise RuntimeError("SSVEPCoordinator runtime is not bound.")
        self._runtime.stop()


class UiCoordinator:
    def build_snapshot(
        self,
        *,
        controller_snapshot: dict[str, object],
        move_source: str,
        decision_source: str,
        robot_mode: str,
        vision_mode: str,
        target_frequency_map: list[tuple[str, object]],
        last_ssvep_raw: str,
        robot_state: RobotRuntimeState,
        vision_state: VisionRuntimeState,
        ssvep_state: SsvEpRuntimeState,
    ) -> AppSnapshot:
        task_state = str(controller_snapshot.get("state", "idle"))
        task_context = dict(controller_snapshot.get("context", {}))
        robot_panel = RobotPanelState(
            connected=robot_state.connected,
            start_active=robot_state.start_active,
            health=robot_state.health,
            last_ack=robot_state.last_ack,
            last_error=robot_state.last_error,
            preflight_ok=robot_state.preflight_ok,
            preflight_message=robot_state.preflight_message,
            calibration_ready=robot_state.calibration_ready,
            robot_cyl=None if robot_state.robot_cyl is None else dict(robot_state.robot_cyl),
            auto_z_current=robot_state.auto_z_current,
            control_kernel=robot_state.control_kernel,
            scene_snapshot=None if robot_state.scene_snapshot is None else dict(robot_state.scene_snapshot),
        )
        vision_panel = VisionPanelState(
            health=vision_state.health,
            packet=vision_state.packet,
            frame=vision_state.frame,
            flash_enabled=bool(vision_state.flash_enabled),
        )
        ssvep_panel = SsvEpPanelState(
            running=ssvep_state.running,
            stim_enabled=ssvep_state.stim_enabled,
            busy=ssvep_state.busy,
            connected=ssvep_state.connected,
            connect_active=ssvep_state.connect_active,
            pretrain_active=ssvep_state.pretrain_active,
            online_active=ssvep_state.online_active,
            mode=ssvep_state.mode,
            runtime_status=ssvep_state.runtime_status,
            profile_path=ssvep_state.profile_path,
            profile_source=ssvep_state.profile_source,
            last_pretrain_time=ssvep_state.last_pretrain_time,
            latest_profile_path=ssvep_state.latest_profile_path,
            profile_count=ssvep_state.profile_count,
            available_profiles=tuple((label, path) for label, path in ssvep_state.available_profiles),
            allow_fallback_profile=ssvep_state.allow_fallback_profile,
            status_hint=ssvep_state.status_hint,
            last_error=ssvep_state.last_error,
            model_name=ssvep_state.model_name,
            debug_keyboard=ssvep_state.debug_keyboard,
            last_state=ssvep_state.last_state,
            last_selected_freq=ssvep_state.last_selected_freq,
            last_margin=ssvep_state.last_margin,
            last_ratio=ssvep_state.last_ratio,
            last_stable_windows=ssvep_state.last_stable_windows,
        )
        return AppSnapshot(
            task_state=task_state,
            task_context=task_context,
            move_source=move_source,
            decision_source=decision_source,
            robot_mode=robot_mode,
            vision_mode=vision_mode,
            motion_deadline_ts=task_context.get("motion_deadline_ts"),
            target_frequency_map=tuple(target_frequency_map),
            last_ssvep_raw=last_ssvep_raw,
            robot=robot_panel,
            vision=vision_panel,
            ssvep=ssvep_panel,
        )
