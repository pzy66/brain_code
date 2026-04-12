from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from hybrid_controller.config import AppConfig
from hybrid_controller.coordinators import RobotCoordinator, SSVEPCoordinator, UiCoordinator, VisionCoordinator
from hybrid_controller.snapshot import AppSnapshot


def sync_coordinator_states_from_runtime_info(
    *,
    config: AppConfig,
    runtime_info: Mapping[str, Any],
    latest_world_snapshot: dict[str, object] | None,
    latest_vision_packet: dict[str, object] | None,
    fetch_remote_snapshot: Callable[[], dict[str, object] | None],
    robot_coordinator: RobotCoordinator,
    vision_coordinator: VisionCoordinator,
    ssvep_coordinator: SSVEPCoordinator,
) -> None:
    robot_coordinator.update(
        connected=bool(runtime_info.get("robot_connected", False)),
        start_active=bool(runtime_info.get("robot_start_active", False)),
        health=str(runtime_info.get("robot_health", "unknown")),
        last_ack=str(runtime_info.get("last_robot_ack", "--")),
        last_error=str(runtime_info.get("last_robot_error", "--")),
        preflight_ok=bool(runtime_info.get("preflight_ok", False)),
        preflight_message=str(runtime_info.get("preflight_message", "not_required")),
        calibration_ready=runtime_info.get("calibration_ready"),
        robot_cyl=runtime_info.get("robot_cyl"),
        limits_cyl=runtime_info.get("limits_cyl"),
        limits_cyl_auto=(latest_world_snapshot or {}).get("limits_cyl_auto"),
        auto_z_current=runtime_info.get("auto_z_current"),
        control_kernel=str(runtime_info.get("control_kernel", "cylindrical_kernel")),
    )
    robot_coordinator.set_scene_snapshot(latest_world_snapshot)
    robot_coordinator.apply_remote_snapshot(fetch_remote_snapshot())

    vision_coordinator.update(
        health=str(runtime_info.get("vision_health", "unknown")),
        packet=latest_vision_packet,
        frame=None,
        flash_enabled=bool(runtime_info.get("ssvep_stim_enabled", False)),
    )

    ssvep_coordinator.update(
        running=bool(runtime_info.get("ssvep_running", False)),
        stim_enabled=bool(runtime_info.get("ssvep_stim_enabled", False)),
        busy=bool(runtime_info.get("ssvep_busy", False)),
        connected=bool(runtime_info.get("ssvep_connected", False)),
        connect_active=bool(runtime_info.get("ssvep_connect_active", False)),
        pretrain_active=bool(runtime_info.get("ssvep_pretrain_active", False)),
        online_active=bool(runtime_info.get("ssvep_online_active", False)),
        mode=str(runtime_info.get("ssvep_mode", "idle")),
        runtime_status=str(runtime_info.get("ssvep_runtime_status", "stopped")),
        profile_path=str(runtime_info.get("ssvep_profile_path", "--")),
        profile_source=str(runtime_info.get("ssvep_profile_source", "fallback")),
        last_pretrain_time=str(runtime_info.get("ssvep_last_pretrain_time", "--")),
        latest_profile_path=str(runtime_info.get("ssvep_latest_profile_path", "--")),
        profile_count=int(runtime_info.get("ssvep_profile_count", 0)),
        available_profiles=tuple(runtime_info.get("ssvep_available_profiles", ())),
        allow_fallback_profile=bool(runtime_info.get("ssvep_allow_fallback_profile", config.ssvep_allow_fallback_profile)),
        status_hint=str(runtime_info.get("ssvep_status_hint", "--")),
        last_error=str(runtime_info.get("ssvep_last_error", "--")),
        model_name=str(runtime_info.get("ssvep_model_name", config.ssvep_model_name)),
        debug_keyboard=bool(runtime_info.get("ssvep_debug_keyboard", config.ssvep_keyboard_debug_enabled)),
        last_state=str(runtime_info.get("ssvep_last_state", "--")),
        last_selected_freq=str(runtime_info.get("ssvep_last_selected_freq", "--")),
        last_margin=str(runtime_info.get("ssvep_last_margin", "--")),
        last_ratio=str(runtime_info.get("ssvep_last_ratio", "--")),
        last_stable_windows=str(runtime_info.get("ssvep_last_stable_windows", "--")),
    )


def build_ui_snapshot(
    *,
    ui_coordinator: UiCoordinator,
    controller_snapshot: dict[str, object],
    config: AppConfig,
    runtime_info: Mapping[str, Any],
    robot_coordinator: RobotCoordinator,
    vision_coordinator: VisionCoordinator,
    ssvep_coordinator: SSVEPCoordinator,
) -> AppSnapshot:
    return ui_coordinator.build_snapshot(
        controller_snapshot=controller_snapshot,
        move_source=str(config.move_source),
        decision_source=str(config.decision_source),
        robot_mode=str(config.robot_mode),
        vision_mode=str(config.vision_mode),
        target_frequency_map=list(runtime_info.get("target_frequency_map", [])),
        last_ssvep_raw=str(runtime_info.get("last_ssvep_raw", "--")),
        robot_state=robot_coordinator.get_state(),
        vision_state=vision_coordinator.get_state(),
        ssvep_state=ssvep_coordinator.get_state(),
    )
