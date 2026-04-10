from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RobotRuntimeState:
    connected: bool = False
    start_active: bool = False
    health: str = "unknown"
    last_ack: str = "--"
    last_error: str = "--"
    preflight_ok: bool = True
    preflight_message: str = "not_required"
    calibration_ready: bool | None = None
    robot_cyl: dict[str, object] | None = None
    limits_cyl: dict[str, object] | None = None
    limits_cyl_auto: dict[str, object] | None = None
    auto_z_current: float | None = None
    control_kernel: str = "cylindrical_kernel"
    scene_snapshot: dict[str, object] | None = None
    remote_snapshot: dict[str, object] | None = None


@dataclass(slots=True)
class VisionRuntimeState:
    mode: str = "slots"
    health: str = "unknown"
    packet: dict[str, object] | None = None
    frame: object | None = None
    flash_enabled: bool = False


@dataclass(slots=True)
class SsvEpRuntimeState:
    running: bool = False
    stim_enabled: bool = False
    busy: bool = False
    connected: bool = False
    connect_active: bool = False
    pretrain_active: bool = False
    online_active: bool = False
    mode: str = "idle"
    runtime_status: str = "stopped"
    profile_path: str = "--"
    profile_source: str = "fallback"
    last_pretrain_time: str = "--"
    latest_profile_path: str = "--"
    profile_count: int = 0
    available_profiles: tuple[tuple[str, str], ...] = ()
    allow_fallback_profile: bool = True
    status_hint: str = "--"
    last_error: str = "--"
    model_name: str = "fbcca"
    debug_keyboard: bool = True
    last_state: str = "--"
    last_selected_freq: str = "--"
    last_margin: str = "--"
    last_ratio: str = "--"
    last_stable_windows: str = "--"


@dataclass(frozen=True, slots=True)
class RobotPanelState:
    connected: bool
    start_active: bool
    health: str
    last_ack: str
    last_error: str
    preflight_ok: bool
    preflight_message: str
    calibration_ready: bool | None
    robot_cyl: dict[str, object] | None
    auto_z_current: float | None
    control_kernel: str
    scene_snapshot: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class VisionPanelState:
    health: str
    packet: dict[str, object] | None
    frame: object | None
    flash_enabled: bool


@dataclass(frozen=True, slots=True)
class SsvEpPanelState:
    running: bool
    stim_enabled: bool
    busy: bool
    connected: bool
    connect_active: bool
    pretrain_active: bool
    online_active: bool
    mode: str
    runtime_status: str
    profile_path: str
    profile_source: str
    last_pretrain_time: str
    latest_profile_path: str
    profile_count: int
    available_profiles: tuple[tuple[str, str], ...]
    allow_fallback_profile: bool
    status_hint: str
    last_error: str
    model_name: str
    debug_keyboard: bool
    last_state: str
    last_selected_freq: str
    last_margin: str
    last_ratio: str
    last_stable_windows: str


@dataclass(frozen=True, slots=True)
class AppSnapshot:
    task_state: str
    task_context: dict[str, object]
    move_source: str
    decision_source: str
    robot_mode: str
    vision_mode: str
    motion_deadline_ts: float | None
    target_frequency_map: tuple[tuple[str, object], ...] = field(default_factory=tuple)
    last_ssvep_raw: str = "--"
    robot: RobotPanelState = field(
        default_factory=lambda: RobotPanelState(
            False,
            False,
            "unknown",
            "--",
            "--",
            True,
            "not_required",
            None,
            None,
            None,
            "cylindrical_kernel",
            None,
        )
    )
    vision: VisionPanelState = field(default_factory=lambda: VisionPanelState("unknown", None, None, False))
    ssvep: SsvEpPanelState = field(
        default_factory=lambda: SsvEpPanelState(
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            "idle",
            "stopped",
            "--",
            "fallback",
            "--",
            "--",
            0,
            (),
            True,
            "--",
            "--",
            "fbcca",
            True,
            "--",
            "--",
            "--",
            "--",
            "--",
        )
    )

    @property
    def selected_target_id(self) -> object:
        return self.task_context.get("selected_target_id")

    @property
    def selected_target_raw_center(self) -> object:
        return self.task_context.get("selected_target_raw_center")

    @property
    def frozen_targets(self) -> list[dict[str, object]]:
        return list(self.task_context.get("frozen_targets", []))

    @property
    def carrying(self) -> bool:
        return bool(self.task_context.get("carrying", False))

    @property
    def last_robot_status(self) -> object:
        return self.task_context.get("last_robot_status")

    @property
    def last_error(self) -> object:
        return self.task_context.get("last_error")
