from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from hybrid_controller.cylindrical import cartesian_to_cylindrical, cylindrical_to_cartesian


PACKAGE_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PACKAGE_ROOT / "dataset"
MODELS_ROOT = PACKAGE_ROOT / "models"
LOGS_ROOT = PACKAGE_ROOT / "logs"


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
    robot_mode: str = "real"
    vision_mode: str = "robot_camera_detection"
    move_source: str = "sim"
    decision_source: str = "sim"
    stage_motion_sec: float = 10.0
    continue_motion_sec: float = 10.0
    vision_max_targets: int = 4
    sim_move_step_mm: float = 6.0
    robot_host: str = "192.168.149.1"
    robot_port: int = 8888
    robot_transport: str = "tcp"
    rosbridge_port: int = 9091
    rosbridge_timeout_sec: float = 3.0
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
    robot_radius_limits_mm: tuple[float, float] = (50.0, 280.0)
    robot_auto_radius_limits_mm: tuple[float, float] = (80.0, 260.0)
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
    teleop_theta_rate_deg_s: float = 80.0
    teleop_radius_rate_mm_s: float = 160.0
    teleop_deadman_timeout_sec: float = 0.2
    teleop_ros_keepalive_interval_ms: int = 120
    teleop_ros_service_fallback_enabled: bool = False
    teleop_kernel_tick_hz: float = 20.0
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
    vision_weights_path: Path = MODELS_ROOT / "vision" / "best.pt"
    vision_infer_interval_ms: int = 80
    vision_model_imgsz: int = 512
    vision_confidence_threshold: float = 0.25
    vision_iou_threshold: float = 0.50
    vision_max_det: int = 6
    vision_device: str = "auto"
    vision_half: bool = False
    vision_warmup_runs: int = 1
    vision_adaptive_infer_enabled: bool = True
    vision_infer_interval_min_ms: int = 45
    vision_infer_interval_max_ms: int = 220
    vision_infer_target_queue_age_ms: float = 90.0
    vision_infer_hysteresis_ms: float = 12.0
    vision_infer_adjust_alpha: float = 0.35
    vision_infer_max_step_up_ms: float = 45.0
    vision_infer_max_step_down_ms: float = 30.0
    vision_reconnect_interval_ms: int = 1200
    vision_read_fail_threshold: int = 10
    vision_stream_drain_grabs: int = 12
    vision_open_timeout_ms: int = 1200
    vision_read_timeout_ms: int = 1200
    vision_probe_reads: int = 3
    vision_probe_sleep_ms: int = 60
    vision_endpoint_probe_timeout_ms: int = 250
    vision_world_scale_xy: float = 1.0
    vision_world_offset_xy_mm: tuple[float, float] = (0.0, -120.0)
    vision_mapping_mode: str = "delta_servo"
    vision_target_frame: str = "robot_base"
    vision_snapshot_max_age_ms: float = 200.0
    vision_action_requires_calibration: bool = True
    pick_cyl_radius_bias_mm: float = 0.0
    pick_cyl_theta_bias_deg: float = 0.0
    ssvep_backend: str = "async_fbcca_idle"
    ssvep_serial_port: str = "auto"
    ssvep_board_id: int = 0
    ssvep_sampling_rate: int = 250
    ssvep_refresh_rate_hz: float = 240.0
    ssvep_freqs: tuple[float, float, float, float] = (8.0, 10.0, 12.0, 15.0)
    ssvep_win_sec: float = 3.0
    ssvep_step_sec: float = 0.5
    ssvep_score_threshold: float = 0.02
    ssvep_ratio_threshold: float = 1.10
    ssvep_history_len: int = 5
    ssvep_profile_dir: Path = DATASET_ROOT / "ssvep_profiles"
    ssvep_current_profile_path: Path = DATASET_ROOT / "ssvep_profiles" / "current_fbcca_profile.json"
    ssvep_default_profile_path: Path = DATASET_ROOT / "ssvep_profiles" / "default_fbcca_profile.json"
    ssvep_allow_fallback_profile: bool = True
    ssvep_auto_use_latest_profile: bool = True
    ssvep_prefer_default_profile: bool = True
    ssvep_recent_profile_limit: int = 12
    ssvep_keyboard_debug_enabled: bool = True
    ssvep_model_name: str = "fbcca"
    ssvep_pretrain_prepare_sec: float = 1.0
    ssvep_pretrain_active_sec: float = 4.0
    ssvep_pretrain_rest_sec: float = 1.0
    ssvep_pretrain_target_repeats: int = 5
    ssvep_pretrain_idle_repeats: int = 10
    ui_panel_refresh_interval_ms: int = 120
    ui_refresh_interval_ms: int = 50
    remote_snapshot_poll_interval_ms: int = 100
    event_log_path: Path = LOGS_ROOT / "hybrid_controller.jsonl"

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
        host = str(self.robot_host).strip()
        return (
            f"http://{host}:8080/stream?"
            "topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80"
        )

    def resolve_vision_stream_candidates(self) -> tuple[str, ...]:
        if self.vision_stream_url:
            return (str(self.vision_stream_url),)
        host = str(self.robot_host).strip()
        return (
            (
                f"http://{host}:8080/stream?"
                "topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80"
            ),
            f"http://{host}:8080/stream?topic=/usb_cam/image_rect_color",
            f"http://{host}:8080/stream?topic=/usb_cam/image_raw",
            f"http://{host}:8080/stream?topic=/camera/rgb/image_raw",
            f"http://{host}:8080/stream?topic=/camera/image_raw",
            f"http://{host}:8080/?action=stream",
            f"http://{host}:8080/stream.mjpg",
            f"http://{host}:8080/video_feed",
        )

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
