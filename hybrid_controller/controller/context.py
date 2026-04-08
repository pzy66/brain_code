from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class TaskContext:
    robot_xy: tuple[float, float] = (0.0, 0.0)
    robot_cyl: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pending_robot_xy: Optional[tuple[float, float]] = None
    pending_robot_cyl: Optional[tuple[float, float, float]] = None
    robot_busy: bool = False
    robot_execution_state: Optional[str] = None
    robot_auto_z: Optional[float] = None
    active_timer_id: Optional[str] = None
    motion_deadline_ts: Optional[float] = None
    frozen_targets: list[Any] = field(default_factory=list)
    selected_target_id: Optional[int] = None
    selected_target_raw_center: Optional[tuple[float, float]] = None
    selected_target_command_mode: Optional[str] = None
    selected_target_command_point: Optional[tuple[float, float]] = None
    carrying: bool = False
    last_robot_status: Optional[str] = None
    last_error: Optional[str] = None
    latest_vision_targets: list[Any] = field(default_factory=list)

    def clear_selection(self) -> None:
        self.selected_target_id = None
        self.selected_target_raw_center = None
        self.selected_target_command_mode = None
        self.selected_target_command_point = None
