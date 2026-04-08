from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from hybrid_controller.cylindrical import cartesian_to_cylindrical, cylindrical_to_cartesian


@dataclass(frozen=True, slots=True)
class ControlSimSlotSpec:
    slot_id: int
    name: str
    world_xy: tuple[float, float] | None
    pixel_xy: tuple[float, float]
    cylindrical_trz: tuple[float, float, float] | None = None


def _default_pick_slots() -> tuple[ControlSimSlotSpec, ...]:
    return (
        ControlSimSlotSpec(slot_id=1, name="Pick-1", world_xy=(-100.0, -165.0), pixel_xy=(640.0, 360.0)),
        ControlSimSlotSpec(slot_id=2, name="Pick-2", world_xy=(-30.0, -150.0), pixel_xy=(710.0, 360.0)),
        ControlSimSlotSpec(slot_id=3, name="Pick-3", world_xy=(45.0, -138.0), pixel_xy=(760.0, 420.0)),
        ControlSimSlotSpec(slot_id=4, name="Pick-4", world_xy=(110.0, -120.0), pixel_xy=(820.0, 500.0)),
    )


def _default_place_slots() -> tuple[ControlSimSlotSpec, ...]:
    return (
        ControlSimSlotSpec(slot_id=101, name="Place-A", world_xy=(-100.0, -80.0), pixel_xy=(520.0, 220.0)),
        ControlSimSlotSpec(slot_id=102, name="Place-B", world_xy=(100.0, -80.0), pixel_xy=(780.0, 220.0)),
    )


def _default_hardware_pick_slots() -> tuple[ControlSimSlotSpec, ...]:
    return (
        ControlSimSlotSpec(slot_id=1, name="HW-1", world_xy=(-100.0, -165.0), pixel_xy=(640.0, 360.0)),
        ControlSimSlotSpec(slot_id=2, name="HW-2", world_xy=(-30.0, -150.0), pixel_xy=(710.0, 360.0)),
        ControlSimSlotSpec(slot_id=3, name="HW-3", world_xy=(45.0, -138.0), pixel_xy=(760.0, 420.0)),
        ControlSimSlotSpec(slot_id=4, name="HW-4", world_xy=(110.0, -120.0), pixel_xy=(820.0, 500.0)),
    )


@dataclass(frozen=True)
class AppConfig:
    control_sim_enabled: bool = True
    sim_process_mode: str = "dual"
    slot_profile: str = "default"
    simulation_enabled: bool = True
    timing_profile: str = "formal"
    scenario_name: str = "basic"
    robot_mode: str = "fake"
    vision_mode: str = "slots"
    move_source: str = "sim"
    decision_source: str = "sim"
    stage_motion_sec: float = 10.0
    continue_motion_sec: float = 10.0
    vision_max_targets: int = 4
    mi_step_mm: float = 6.0
    sim_move_step_mm: float = 6.0
    mi_emit_interval_ms: int = 300
    mi_min_confidence: float = 0.60
    mi_stable_windows: int = 3
    robot_host: str = "127.0.0.1"
    robot_port: int = 8888
    robot_timeout_sec: float = 0.5
    robot_ping_timeout_sec: float = 1.0
    robot_reconnect_delay_sec: float = 1.0
    robot_start_xy: tuple[float, float] = (0.0, -120.0)
    robot_limits_x: tuple[float, float] = (-140.0, 140.0)
    robot_limits_y: tuple[float, float] = (-200.0, -40.0)
    robot_travel_z: float = 130.0
    robot_approach_z: float = 130.0
    robot_pick_z: float = 85.0
    robot_carry_z: float = 160.0
    robot_move_speed_mm_s: float = 150.0
    robot_target_margin_mm: float = 15.0
    robot_theta_limits_deg: tuple[float, float] = (-120.0, 120.0)
    robot_radius_limits_mm: tuple[float, float] = (50.0, 230.0)
    robot_height_limits_mm: tuple[float, float] = (80.0, 212.8)
    cylindrical_xy_workspace_enabled: bool = False
    robot_auto_z_profile_radius_step_mm: float = 5.0
    robot_auto_z_profile_height_step_mm: float = 5.0
    robot_auto_z_preferred_mm: float = 160.0
    robot_auto_z_plateau_min_radius_mm: float = 145.0
    robot_auto_z_plateau_max_radius_mm: float = 185.0
    robot_auto_z_plateau_z_mm: float = 205.0
    robot_auto_z_retract_drop_per_radius_mm: float = 0.8
    robot_auto_z_extend_drop_per_radius_mm: float = 0.4
    robot_auto_z_posture_tolerance_deg: float = 8.0
    robot_auto_z_down_per_radius_mm: float = 0.5
    robot_auto_z_up_per_radius_mm: float = 1.0
    robot_auto_z_min_delta_mm: float = 3.0
    robot_motion_min_duration_sec: float = 0.25
    robot_motion_settle_sec: float = 0.08
    robot_teleop_min_duration_sec: float = 0.12
    robot_teleop_settle_sec: float = 0.02
    motion_coordinate_mode: str = "cylindrical"
    teleop_move_step_mm: float = 8.0
    teleop_theta_step_deg: float = 4.0
    teleop_radius_step_mm: float = 8.0
    teleop_repeat_interval_ms: int = 50
    roi_center: tuple[float, float] = (640.0, 360.0)
    roi_radius: float = 260.0
    motion_bounds_x: tuple[float, float] = (-140.0, 140.0)
    motion_bounds_y: tuple[float, float] = (-200.0, -40.0)
    fake_robot_ack_delay_sec: float = 1.0
    sim_pick_delay_sec: float = 1.0
    sim_place_delay_sec: float = 1.0
    fake_vision_interval_ms: int = 1000
    sim_vision_interval_ms: int = 1000
    control_sim_slot_tolerance_px: float = 30.0
    control_sim_place_snap_distance_mm: float = 35.0
    sim_pick_slots: tuple[ControlSimSlotSpec, ...] = field(default_factory=_default_pick_slots)
    sim_place_slots: tuple[ControlSimSlotSpec, ...] = field(default_factory=_default_place_slots)
    hardware_pick_slots: tuple[ControlSimSlotSpec, ...] = field(default_factory=_default_hardware_pick_slots)
    vision_stream_url: str = ""
    vision_weights_path: Path = Path("dataset/camara/best.pt")
    vision_infer_interval_ms: int = 200
    vision_confidence_threshold: float = 0.25
    ssvep_reference_path: Path = Path(
        "brain_code/02_SSVEP/2026-03_realtime_ui_and_online_decoder/SSVEP/demo.py"
    )
    ssvep_serial_port: str = "COM3"
    ssvep_board_id: int = 0
    ssvep_sampling_rate: int = 250
    ssvep_refresh_rate_hz: float = 240.0
    ssvep_freqs: tuple[float, float, float, float] = (8.0, 10.0, 12.0, 15.0)
    ssvep_win_sec: float = 3.0
    ssvep_step_sec: float = 0.5
    ssvep_score_threshold: float = 0.02
    ssvep_ratio_threshold: float = 1.10
    ssvep_history_len: int = 5
    mi_reference_path: Path = Path(
        "brain_code/01_MI/mi_classifier_latest/code/realtime/mi_realtime_infer_only.py"
    )
    mi_model_path: Path = Path(
        "brain_code/01_MI/mi_classifier_latest/code/realtime/models/custom_mi_realtime.joblib"
    )
    mi_serial_port: str = ""
    mi_board_id: int = 0
    mi_realtime_mode: str = "continuous"
    ui_refresh_interval_ms: int = 200
    event_log_path: Path = Path("logs/hybrid_controller.jsonl")

    def resolved(self) -> "AppConfig":
        profile = str(self.timing_profile or "formal").strip().lower()
        config = self
        if profile == "fast":
            config = replace(
                config,
                stage_motion_sec=2.0,
                continue_motion_sec=2.0,
                sim_pick_delay_sec=0.2,
                sim_place_delay_sec=0.2,
                sim_vision_interval_ms=120,
            )
        return replace(
            config,
            motion_bounds_x=config.robot_limits_x,
            motion_bounds_y=config.robot_limits_y,
            fake_robot_ack_delay_sec=float(config.sim_pick_delay_sec),
            fake_vision_interval_ms=int(config.sim_vision_interval_ms),
        )

    def resolve_vision_stream_url(self) -> str:
        if self.vision_stream_url:
            return self.vision_stream_url
        return f"http://{self.robot_host}:8080/stream?topic=/usb_cam/image_rect_color"

    @property
    def robot_start_cyl(self) -> tuple[float, float, float]:
        return cartesian_to_cylindrical(
            self.robot_start_xy[0],
            self.robot_start_xy[1],
            self.robot_carry_z,
        )

    def cylindrical_target_to_world_xy(self, cylindrical_trz: tuple[float, float, float]) -> tuple[float, float]:
        x_mm, y_mm, _ = cylindrical_to_cartesian(
            cylindrical_trz[0],
            cylindrical_trz[1],
            cylindrical_trz[2],
        )
        return (x_mm, y_mm)
