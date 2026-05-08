from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from brain_workspace.paths import HYBRID_SSVEP_PROFILE_DIR, PROFILE_DATASET_DIR, VISION_DATASET_DIR
from hybrid_controller.cylindrical import cartesian_to_cylindrical, cylindrical_to_cartesian


PACKAGE_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROFILE_DATASET_DIR / "hybrid_controller"
MODELS_ROOT = VISION_DATASET_DIR / "models"
LOGS_ROOT = PACKAGE_ROOT / "logs"
LEGACY_DATASET_ROOT = PACKAGE_ROOT / "dataset"
# Locked JetMax/Hiwonder camera contract.
#
# Robot side owns the USB camera and publishes the official chain:
#   usb_cam.service -> usb_cam_node -> /usb_cam/image_rect_color -> web_video_server:8080
# PC side must only consume that single MJPEG stream. Do not add fallback URLs, do not
# probe /dev/video*, and do not start/restart camera sender processes from desktop code.
HIWONDER_CAMERA_TOPIC = "/usb_cam/image_rect_color"
HIWONDER_CAMERA_WIDTH = 640
HIWONDER_CAMERA_HEIGHT = 480
HIWONDER_CAMERA_QUALITY = 80
HIWONDER_CAMERA_STREAM_TYPE = "mjpeg"


def build_hiwonder_camera_stream_url(host: str) -> str:
    """Return the only default PC-side camera URL for the JetMax official MJPEG stream."""
    normalized_host = str(host).strip()
    return (
        f"http://{normalized_host}:8080/stream?"
        f"topic={HIWONDER_CAMERA_TOPIC}"
        f"&type={HIWONDER_CAMERA_STREAM_TYPE}"
        f"&width={HIWONDER_CAMERA_WIDTH}"
        f"&height={HIWONDER_CAMERA_HEIGHT}"
        f"&quality={HIWONDER_CAMERA_QUALITY}"
    )


def _prefer_existing_file(canonical_path: Path, legacy_path: Path) -> Path:
    if canonical_path.exists():
        return canonical_path
    if legacy_path.exists():
        return legacy_path
    return canonical_path


def _directory_has_expected_files(path: Path, *, expected_names: tuple[str, ...]) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    try:
        if expected_names:
            return any((path / name).exists() for name in expected_names)
        return any(child.is_file() for child in path.iterdir())
    except OSError:
        return False


def _prefer_dataset_dir(
    canonical_dir: Path,
    legacy_dir: Path,
    *,
    expected_names: tuple[str, ...] = (),
) -> Path:
    if _directory_has_expected_files(canonical_dir, expected_names=expected_names):
        return canonical_dir
    if _directory_has_expected_files(legacy_dir, expected_names=expected_names):
        return legacy_dir
    return canonical_dir


def _default_vision_calibration_profile_path() -> Path:
    canonical = VISION_DATASET_DIR / "calibration" / "current_profile.json"
    legacy = LEGACY_DATASET_ROOT / "vision_calibration" / "current_profile.json"
    return _prefer_existing_file(canonical, legacy)


def _default_pick_tuning_profile_path() -> Path:
    canonical = DATASET_ROOT / "robot_pick_tuning" / "current_pick_tuning.json"
    legacy = LEGACY_DATASET_ROOT / "robot_pick_tuning" / "current_pick_tuning.json"
    return _prefer_existing_file(canonical, legacy)


def _default_ssvep_profile_dir() -> Path:
    return _prefer_dataset_dir(
        HYBRID_SSVEP_PROFILE_DIR,
        LEGACY_DATASET_ROOT / "ssvep_profiles",
        expected_names=("current_fbcca_profile.json", "default_fbcca_profile.json"),
    )


DEFAULT_SSVEP_PROFILE_DIR = _default_ssvep_profile_dir()


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
    input_profile: str = "operator_keyboard"
    robot_mode: str = "real"
    vision_mode: str = "robot_camera_detection"
    move_source: str = "sim"
    decision_source: str = "sim"
    mi_backend: str = "brainflow"
    mi_enabled: bool = False
    mi_poll_interval_ms: int = 50
    mi_command_cooldown_ms: int = 120
    stage_motion_sec: float = 10.0
    continue_motion_sec: float = 10.0
    vision_max_targets: int = 4
    sim_move_step_mm: float = 6.0
    robot_host: str = "192.168.149.1"
    robot_port: int = 8888
    robot_transport: str = "tcp"
    rosbridge_port: int = 9091
    rosbridge_timeout_sec: float = 3.0
    ros_reconnect_base_delay_sec: float = 0.6
    ros_reconnect_max_delay_sec: float = 8.0
    ros_reconnect_jitter_ratio: float = 0.2
    ros_runtime_probe_timeout_sec: float = 0.6
    ros_runtime_state_grace_sec: float = 3.0
    robot_auto_start_on_ros_unavailable: bool = True
    robot_auto_restart_on_state_stale: bool = True
    robot_auto_start_cooldown_sec: float = 20.0
    robot_bootstrap_retry_enabled: bool = True
    robot_bootstrap_probe_interval_sec: float = 3.0
    robot_bootstrap_probe_timeout_sec: float = 0.35
    robot_timeout_sec: float = 0.5
    robot_ping_timeout_sec: float = 1.0
    robot_reconnect_delay_sec: float = 1.0
    robot_command_timeout_sec: float = 20.0
    robot_state_stale_threshold_ms: float = 700.0
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
    vision_pick_target_pixel: tuple[float, float] | None = None
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
    vision_auto_start: bool = False
    vision_weights_path: Path = MODELS_ROOT / "best.pt"
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
    vision_frame_pose_max_age_ms: float = 250.0
    vision_action_requires_calibration: bool = True
    vision_calibration_profile_path: Path = _default_vision_calibration_profile_path()
    vision_calibration_profile_required: bool = True
    vision_action_max_error_mm: float = 6.0
    vision_grasp_quality_threshold: float = 0.25
    vision_grasp_history_frames: int = 5
    vision_grasp_stable_frames: int = 3
    vision_grasp_stability_tolerance_px: float = 6.0
    vision_center_stability_tolerance_px: float = 6.0
    vision_grasp_angle_stability_tolerance_deg: float = 15.0
    vision_grasp_history_reset_px: float = 22.0
    vision_grasp_stability_wait_frames: int = 10
    vision_frame_fallback_enabled: bool = True
    vision_servo_center_tolerance_px: float = 6.0
    vision_servo_action_tolerance_px: float = 6.0
    vision_servo_low_action_tolerance_px: float = 8.0
    vision_servo_search_action_tolerance_px: float = 16.0
    vision_servo_move_gain: float = 0.8
    vision_servo_fine_move_gain: float = 0.4
    vision_servo_fine_threshold_px: float = 24.0
    vision_servo_max_attempts: int = 5
    vision_eye_in_hand_pick_flow_enabled: bool = True
    vision_eye_in_hand_pick_radius_bias_mm: float = 40.0
    vision_pick_search_z_mm: float = 190.0
    vision_pick_confirm_z_mm: float = 130.0
    vision_pick_descent_step_mm: float = 5.0
    vision_pick_z_tolerance_mm: float = 4.0
    vision_debug_bundle_enabled: bool = True
    vision_debug_bundle_dir: Path = LOGS_ROOT / "vision_debug"
    pick_tool_offset_source: str = "command_bias"
    vision_residual_model: str = "grid"
    vision_calibration_grid_size: int = 7
    pick_cyl_radius_bias_mm: float = 0.0
    pick_cyl_tangent_bias_mm: float = 0.0
    pick_cyl_theta_bias_deg: float = 0.0
    sucker_rotation_enabled: bool = True
    sucker_rotation_offset_deg: float = 0.0
    sucker_rotation_invert: bool = False
    sucker_rotation_min_deg: float = 45.0
    sucker_rotation_max_deg: float = 135.0
    sucker_rotation_duration_sec: float = 0.10
    sucker_rotation_angle_quality_threshold: float = 0.20
    pick_tuning_profile_path: Path = _default_pick_tuning_profile_path()
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
    ssvep_profile_dir: Path = DEFAULT_SSVEP_PROFILE_DIR
    ssvep_current_profile_path: Path = DEFAULT_SSVEP_PROFILE_DIR / "current_fbcca_profile.json"
    ssvep_default_profile_path: Path = DEFAULT_SSVEP_PROFILE_DIR / "default_fbcca_profile.json"
    ssvep_allow_fallback_profile: bool = True
    ssvep_auto_use_latest_profile: bool = True
    ssvep_prefer_default_profile: bool = True
    ssvep_recent_profile_limit: int = 12
    ssvep_runtime_enabled: bool = False
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
        offset_source = str(config.pick_tool_offset_source or "target_pixel").strip().lower()
        if offset_source not in {"target_pixel", "command_bias"}:
            offset_source = "target_pixel"
        residual_model = str(config.vision_residual_model or "grid").strip().lower()
        if residual_model not in {"grid", "idw", "none"}:
            residual_model = "grid"
        return replace(
            config,
            motion_bounds_x=config.robot_limits_x,
            motion_bounds_y=config.robot_limits_y,
            fake_robot_ack_delay_sec=float(config.sim_pick_delay_sec),
            fake_vision_interval_ms=int(config.sim_vision_interval_ms),
            pick_tool_offset_source=offset_source,
            vision_residual_model=residual_model,
            vision_calibration_grid_size=max(2, int(config.vision_calibration_grid_size)),
            vision_frame_pose_max_age_ms=max(1.0, float(config.vision_frame_pose_max_age_ms)),
        )

    def resolve_vision_stream_url(self) -> str:
        if self.vision_stream_url:
            return self.vision_stream_url
        return build_hiwonder_camera_stream_url(str(self.robot_host))

    def resolve_vision_stream_candidates(self) -> tuple[str, ...]:
        if self.vision_stream_url:
            # Explicit override is for manual diagnosis only. Normal operation must use
            # build_hiwonder_camera_stream_url(robot_host), which targets the official chain.
            return (str(self.vision_stream_url),)
        # Keep this tuple length at one. Multiple candidates caused unsafe endpoint
        # probing and can disturb the JetMax Wi-Fi/camera path during startup.
        return (
            build_hiwonder_camera_stream_url(str(self.robot_host)),
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
