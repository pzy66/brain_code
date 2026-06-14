from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from brain_workspace.paths import HYBRID_SSVEP_PROFILE_DIR, PROFILE_DATASET_DIR, VISION_DATASET_DIR
from hybrid_controller.cylindrical import cartesian_to_cylindrical, cylindrical_to_cartesian


DATASET_ROOT = PROFILE_DATASET_DIR / "hybrid_controller"
MODELS_ROOT = VISION_DATASET_DIR / "models"
PACKAGE_ROOT = Path(__file__).resolve().parent
LOGS_ROOT = PACKAGE_ROOT / "logs"
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
JETMAX_SSH_PORT = 22
JETMAX_WEB_VIDEO_PORT = 8080
JETMAX_ROSBRIDGE_PORT = 9091
JETMAX_ROS_MASTER_PORT = 11311
JETMAX_LEGACY_TCP_RUNTIME_PORT = 8888
SERVO_MEASUREMENT_POINTS = frozenset(
    {
        "center",
        "geometry",
        "grasp",
        "top_face",
        "color_block",
        "center_subpixel",
        "geometry_subpixel",
        "grasp_subpixel",
        "top_face_subpixel",
        "color_block_subpixel",
    }
)


def build_hiwonder_camera_stream_url(host: str) -> str:
    """Return the only default PC-side camera URL for the JetMax official MJPEG stream."""
    normalized_host = str(host).strip()
    return (
        f"http://{normalized_host}:{JETMAX_WEB_VIDEO_PORT}/stream?"
        f"topic={HIWONDER_CAMERA_TOPIC}"
        f"&type={HIWONDER_CAMERA_STREAM_TYPE}"
        f"&width={HIWONDER_CAMERA_WIDTH}"
        f"&height={HIWONDER_CAMERA_HEIGHT}"
        f"&quality={HIWONDER_CAMERA_QUALITY}"
    )


def normalize_servo_measurement_point(value: object, *, default: str = "geometry_subpixel") -> str:
    measurement_point = str(value or "").strip().lower()
    fallback = str(default or "geometry_subpixel").strip().lower()
    if fallback not in SERVO_MEASUREMENT_POINTS:
        fallback = "geometry_subpixel"
    if measurement_point not in SERVO_MEASUREMENT_POINTS:
        return fallback
    return measurement_point


def _default_vision_calibration_profile_path() -> Path:
    return VISION_DATASET_DIR / "calibration" / "current_profile.json"


def _default_pick_tuning_profile_path() -> Path:
    return DATASET_ROOT / "robot_pick_tuning" / "current_pick_tuning.json"


def _default_vision_grasp_profile_path() -> Path:
    return DATASET_ROOT / "vision_grasp" / "current_grasp_profile.json"


def _default_ssvep_profile_dir() -> Path:
    return HYBRID_SSVEP_PROFILE_DIR


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
    robot_port: int = JETMAX_LEGACY_TCP_RUNTIME_PORT
    robot_transport: str = "tcp"
    robot_connect_on_start: bool = False
    rosbridge_port: int = JETMAX_ROSBRIDGE_PORT
    rosbridge_timeout_sec: float = 3.0
    ros_reconnect_base_delay_sec: float = 3.0
    ros_reconnect_max_delay_sec: float = 30.0
    ros_reconnect_jitter_ratio: float = 0.25
    ros_probe_before_connect: bool = False
    ros_runtime_probe_timeout_sec: float = 0.6
    ros_runtime_state_grace_sec: float = 3.0
    robot_auto_start_on_ros_unavailable: bool = False
    robot_auto_restart_on_state_stale: bool = False
    robot_auto_start_cooldown_sec: float = 20.0
    robot_auto_start_max_attempts: int = 1
    robot_bootstrap_retry_enabled: bool = False
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
    idle_runtime_tick_interval_ms: int = 250
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
    vision_max_det: int = 4
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
    vision_read_timeout_ms: int = 3000
    vision_probe_reads: int = 3
    vision_probe_sleep_ms: int = 60
    vision_endpoint_probe_timeout_ms: int = 250
    vision_frame_top_mask_rows: int = 28
    vision_frame_bottom_mask_rows: int = 32
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
    vision_frame_min_brightness_mean: float = 30.0
    vision_frame_min_brightness_p95: float = 45.0
    vision_low_height_shape_fallback_enabled: bool = True
    vision_low_height_shape_fallback_min_area_ratio: float = 1.20
    vision_low_height_reject_edge_fallback_candidates: bool = True
    vision_servo_center_tolerance_px: float = 20.0
    vision_servo_action_tolerance_px: float = 20.0
    vision_servo_low_action_tolerance_px: float = 8.0
    vision_servo_search_action_tolerance_px: float = 16.0
    vision_servo_move_gain: float = 0.45
    vision_servo_fine_move_gain: float = 0.35
    # Only enter fine correction near the strict final threshold; 6-8 px low-height
    # error still needs enough gain to be visible in the camera feedback.
    vision_servo_fine_threshold_px: float = 3.0
    vision_servo_max_attempts: int = 12
    vision_servo_measurement_point: str = "geometry_subpixel"
    vision_servo_low_height_measurement_point: str = ""
    vision_low_confirm_untrusted_error_px: float = 12.0
    vision_eye_in_hand_pick_flow_enabled: bool = True
    vision_eye_in_hand_pick_radius_bias_mm: float = 40.0
    vision_pick_search_z_mm: float = 190.0
    vision_pick_confirm_z_mm: float = 130.0
    vision_pick_descent_step_mm: float = 5.0
    vision_pick_descent_coarse_step_mm: float = 10.0
    vision_pick_descent_fine_step_mm: float = 5.0
    vision_pick_descent_fine_band_mm: float = 25.0
    vision_pick_z_tolerance_mm: float = 4.0
    vision_continuous_servo_enabled: bool = True
    vision_continuous_servo_theta_rate_limit_deg_s: float = 10.0
    vision_continuous_servo_radius_rate_limit_mm_s: float = 18.0
    vision_continuous_servo_z_rate_limit_mm_s: float = 18.0
    # Hiwonder's object-tracking demo keeps motion smooth by updating PID output
    # every image cycle and issuing short-duration corrections, not by stopping
    # between visual updates. These gains are intentionally modest because the
    # JetMax-side 20 Hz teleop kernel applies the final acceleration ramp.
    vision_continuous_servo_theta_gain_deg_s_per_deg: float = 1.0
    vision_continuous_servo_radius_gain_mm_s_per_mm: float = 0.55
    vision_continuous_servo_z_slow_band_mm: float = 20.0
    vision_continuous_servo_z_pulse_mm: float = 0.0
    vision_continuous_servo_center_allow_descent_px: float = 8.0
    vision_continuous_servo_center_stop_descent_px: float = 48.0
    vision_continuous_servo_descent_high_error_px: float = 80.0
    vision_continuous_servo_descent_high_error_z_above_confirm_mm: float = 70.0
    vision_continuous_servo_descent_low_error_z_above_confirm_mm: float = 4.0
    vision_continuous_servo_soft_descent_enabled: bool = True
    vision_continuous_servo_soft_descent_rate_scale: float = 0.35
    vision_continuous_servo_soft_descent_min_z_above_confirm_mm: float = 18.0
    # Allow slow descent through small residual image error. Final confirm/pick
    # still uses vision_continuous_servo_pick_ready_center_px.
    vision_continuous_servo_low_height_descent_allow_px: float = 18.0
    vision_continuous_servo_pick_ready_center_px: float = 2.0
    vision_continuous_servo_fine_pulse_center_px: float = 0.0
    vision_continuous_servo_settle_stop_band_px: float = 8.0
    # Slow low-height servo before the strict 2 px final gate so backlash does not
    # turn the last few pixels into radius-direction overshoot.
    vision_continuous_servo_low_height_fine_band_px: float = 3.0
    vision_continuous_servo_low_height_fine_rate_scale: float = 0.35
    vision_continuous_servo_low_height_coarse_rate_scale: float = 0.55
    vision_continuous_servo_low_height_min_theta_rate_deg_s: float = 0.02
    vision_continuous_servo_low_height_min_radius_rate_mm_s: float = 0.04
    vision_continuous_servo_low_height_z_rate_scale: float = 0.35
    vision_continuous_servo_low_height_error_growth_stop_px: float = 1000000.0
    vision_continuous_servo_low_height_guard_band_mm: float = 30.0
    vision_continuous_servo_low_height_pause_descent_band_mm: float = 4.0
    vision_continuous_servo_low_height_unstable_servo_px: float = 60.0
    vision_continuous_servo_low_height_descent_rebound_pause_px: float = 4.0
    vision_continuous_servo_low_height_best_error_descent_pause_px: float = 4.0
    vision_continuous_servo_low_height_best_confirm_descent_allow_px: float = 40.0
    vision_continuous_servo_low_height_max_theta_drift_deg: float = 8.0
    vision_continuous_servo_low_height_max_radius_drift_mm: float = 8.0
    vision_continuous_servo_low_height_best_error_rebound_px: float = 8.0
    vision_continuous_servo_low_height_static_stop_enabled: bool = True
    vision_continuous_servo_low_height_static_error_min_px: float = 8.0
    vision_continuous_servo_low_height_static_error_max_px: float = 30.0
    vision_continuous_servo_low_height_static_frames: int = 12
    vision_continuous_servo_low_height_static_improvement_px: float = 1.0
    vision_continuous_servo_low_height_static_band_mm: float = 6.0
    vision_continuous_servo_low_height_static_pose_band_mm: float = 4.0
    vision_continuous_servo_camera_motion_guard_enabled: bool = True
    vision_continuous_servo_camera_motion_guard_min_robot_mm: float = 8.0
    vision_continuous_servo_camera_motion_guard_max_pixel_px: float = 2.5
    vision_continuous_servo_camera_motion_guard_static_frames: int = 5
    vision_continuous_servo_low_height_rebound_recover_band_mm: float = 10.0
    vision_continuous_servo_low_height_rebound_recover_attempts: int = 3
    vision_continuous_servo_low_height_discrete_refine_enabled: bool = False
    vision_continuous_servo_low_height_refine_attempts: int = 4
    vision_continuous_servo_low_height_refine_max_theta_step_deg: float = 0.25
    vision_continuous_servo_low_height_refine_max_radius_step_mm: float = 1.5
    vision_continuous_servo_stop_at_confirm: bool = False
    vision_continuous_servo_max_center_jump_px: float = 45.0
    vision_continuous_servo_max_error_growth_px: float = 35.0
    vision_continuous_servo_stable_frames: int = 2
    vision_continuous_servo_lost_frames: int = 3
    vision_continuous_servo_stale_frames: int = 3
    vision_continuous_servo_command_timeout_ms: float = 250.0
    vision_continuous_servo_min_confidence: float = 0.55
    vision_continuous_servo_hard_min_confidence: float = 0.20
    vision_continuous_servo_low_confidence_large_area_ratio: float = 12.0
    vision_continuous_servo_min_area_px: int = 1500
    vision_continuous_servo_horizontal_mode: str = "ibvs_dls"
    vision_continuous_servo_ibvs_gain: float = 0.45
    vision_continuous_servo_ibvs_damping_px_per_unit: float = 2.0
    vision_continuous_servo_ibvs_du_dtheta_px_per_deg: float = -14.0
    vision_continuous_servo_ibvs_du_dradius_px_per_mm: float = 0.0
    vision_continuous_servo_ibvs_dv_dtheta_px_per_deg: float = 0.0
    vision_continuous_servo_ibvs_dv_dradius_px_per_mm: float = 3.5
    vision_continuous_servo_ibvs_jacobian_source: str = "config"
    vision_continuous_servo_ibvs_profile_jacobian: Any = None
    vision_continuous_servo_ibvs_fitted_jacobian: Any = None
    vision_continuous_servo_ibvs_profile_stage_band_mm: float = 15.0
    vision_continuous_servo_pixel_jacobian_gain: float = 0.35
    vision_continuous_servo_pixel_jacobian_dtheta_dx: float = 0.11998811664170087
    vision_continuous_servo_pixel_jacobian_dtheta_dy: float = 0.08965988766661396
    vision_continuous_servo_pixel_jacobian_dr_dx: float = 0.05790918576393915
    vision_continuous_servo_pixel_jacobian_dr_dy: float = 0.24030828155353375
    vision_continuous_servo_pixel_axis_theta_deg_s_per_px: float = 0.08
    vision_continuous_servo_pixel_axis_radius_mm_s_per_px: float = -0.06
    vision_continuous_servo_pixel_axis_fine_band_px: float = 24.0
    vision_continuous_servo_pixel_axis_fine_rate_scale: float = 0.35
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
    vision_grasp_profile_path: Path = _default_vision_grasp_profile_path()
    vision_grasp_profile_required: bool = True
    vision_grasp_profile_real_pick_required: bool = True
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
        servo_measurement = normalize_servo_measurement_point(config.vision_servo_measurement_point)
        low_height_servo_measurement = str(config.vision_servo_low_height_measurement_point or "").strip().lower()
        if low_height_servo_measurement:
            low_height_servo_measurement = normalize_servo_measurement_point(
                low_height_servo_measurement,
                default=servo_measurement,
            )
        continuous_horizontal_mode = str(
            config.vision_continuous_servo_horizontal_mode or "servo_command_point"
        ).strip().lower()
        if continuous_horizontal_mode not in {"servo_command_point", "pixel_jacobian", "pixel_axis", "ibvs_dls"}:
            continuous_horizontal_mode = "servo_command_point"
        return replace(
            config,
            motion_bounds_x=config.robot_limits_x,
            motion_bounds_y=config.robot_limits_y,
            fake_robot_ack_delay_sec=float(config.sim_pick_delay_sec),
            fake_vision_interval_ms=int(config.sim_vision_interval_ms),
            vision_max_targets=max(1, min(4, int(config.vision_max_targets))),
            vision_max_det=max(1, min(4, int(config.vision_max_det))),
            pick_tool_offset_source=offset_source,
            vision_residual_model=residual_model,
            vision_servo_measurement_point=servo_measurement,
            vision_servo_low_height_measurement_point=low_height_servo_measurement,
            vision_continuous_servo_horizontal_mode=continuous_horizontal_mode,
            vision_calibration_grid_size=max(2, int(config.vision_calibration_grid_size)),
            vision_frame_pose_max_age_ms=max(1.0, float(config.vision_frame_pose_max_age_ms)),
            vision_frame_top_mask_rows=max(0, min(120, int(config.vision_frame_top_mask_rows))),
            vision_frame_bottom_mask_rows=max(0, min(120, int(config.vision_frame_bottom_mask_rows))),
            vision_frame_min_brightness_mean=max(0.0, float(config.vision_frame_min_brightness_mean)),
            vision_frame_min_brightness_p95=max(0.0, float(config.vision_frame_min_brightness_p95)),
            vision_pick_descent_step_mm=max(0.1, float(config.vision_pick_descent_step_mm)),
            vision_pick_descent_coarse_step_mm=max(0.1, float(config.vision_pick_descent_coarse_step_mm)),
            vision_pick_descent_fine_step_mm=max(0.1, float(config.vision_pick_descent_fine_step_mm)),
            vision_pick_descent_fine_band_mm=max(0.0, float(config.vision_pick_descent_fine_band_mm)),
            vision_continuous_servo_theta_rate_limit_deg_s=max(
                0.1, float(config.vision_continuous_servo_theta_rate_limit_deg_s)
            ),
            vision_continuous_servo_radius_rate_limit_mm_s=max(
                0.1, float(config.vision_continuous_servo_radius_rate_limit_mm_s)
            ),
            vision_continuous_servo_z_rate_limit_mm_s=max(0.1, float(config.vision_continuous_servo_z_rate_limit_mm_s)),
            vision_continuous_servo_z_pulse_mm=max(0.0, float(config.vision_continuous_servo_z_pulse_mm)),
            vision_continuous_servo_center_allow_descent_px=max(
                0.1, float(config.vision_continuous_servo_center_allow_descent_px)
            ),
            vision_continuous_servo_center_stop_descent_px=max(
                max(0.1, float(config.vision_continuous_servo_center_allow_descent_px)),
                float(config.vision_continuous_servo_center_stop_descent_px),
            ),
            vision_continuous_servo_descent_high_error_px=max(
                max(0.1, float(config.vision_continuous_servo_center_allow_descent_px)),
                float(config.vision_continuous_servo_descent_high_error_px),
            ),
            vision_continuous_servo_descent_high_error_z_above_confirm_mm=max(
                0.1, float(config.vision_continuous_servo_descent_high_error_z_above_confirm_mm)
            ),
            vision_continuous_servo_descent_low_error_z_above_confirm_mm=max(
                0.0, float(config.vision_continuous_servo_descent_low_error_z_above_confirm_mm)
            ),
            vision_continuous_servo_soft_descent_rate_scale=max(
                0.0, min(1.0, float(config.vision_continuous_servo_soft_descent_rate_scale))
            ),
            vision_continuous_servo_soft_descent_min_z_above_confirm_mm=max(
                0.0, float(config.vision_continuous_servo_soft_descent_min_z_above_confirm_mm)
            ),
            vision_continuous_servo_low_height_descent_allow_px=max(
                max(0.1, float(config.vision_continuous_servo_center_allow_descent_px)),
                float(config.vision_continuous_servo_low_height_descent_allow_px),
            ),
            vision_continuous_servo_low_height_best_confirm_descent_allow_px=max(
                max(0.1, float(config.vision_continuous_servo_low_height_descent_allow_px)),
                float(config.vision_continuous_servo_low_height_best_confirm_descent_allow_px),
            ),
            vision_continuous_servo_pick_ready_center_px=max(
                0.1, float(config.vision_continuous_servo_pick_ready_center_px)
            ),
            vision_continuous_servo_fine_pulse_center_px=max(
                0.0, float(config.vision_continuous_servo_fine_pulse_center_px)
            ),
            vision_continuous_servo_settle_stop_band_px=max(
                0.1, float(config.vision_continuous_servo_settle_stop_band_px)
            ),
            vision_continuous_servo_low_height_fine_band_px=max(
                0.1, float(config.vision_continuous_servo_low_height_fine_band_px)
            ),
            vision_continuous_servo_low_height_fine_rate_scale=max(
                0.05, min(1.0, float(config.vision_continuous_servo_low_height_fine_rate_scale))
            ),
            vision_continuous_servo_low_height_coarse_rate_scale=max(
                max(0.05, min(1.0, float(config.vision_continuous_servo_low_height_fine_rate_scale))),
                min(1.0, float(config.vision_continuous_servo_low_height_coarse_rate_scale)),
            ),
            vision_continuous_servo_low_height_z_rate_scale=max(
                0.05, min(1.0, float(config.vision_continuous_servo_low_height_z_rate_scale))
            ),
            vision_continuous_servo_low_height_error_growth_stop_px=max(
                0.1, float(config.vision_continuous_servo_low_height_error_growth_stop_px)
            ),
            vision_continuous_servo_low_height_guard_band_mm=max(
                0.0, float(config.vision_continuous_servo_low_height_guard_band_mm)
            ),
            vision_continuous_servo_low_height_pause_descent_band_mm=max(
                0.0, float(config.vision_continuous_servo_low_height_pause_descent_band_mm)
            ),
            vision_continuous_servo_low_height_unstable_servo_px=max(
                1.0, float(config.vision_continuous_servo_low_height_unstable_servo_px)
            ),
            vision_continuous_servo_low_height_descent_rebound_pause_px=max(
                0.1, float(config.vision_continuous_servo_low_height_descent_rebound_pause_px)
            ),
            vision_continuous_servo_low_height_max_theta_drift_deg=max(
                0.1, float(config.vision_continuous_servo_low_height_max_theta_drift_deg)
            ),
            vision_continuous_servo_low_height_max_radius_drift_mm=max(
                0.1, float(config.vision_continuous_servo_low_height_max_radius_drift_mm)
            ),
            vision_continuous_servo_low_height_best_error_rebound_px=max(
                0.1, float(config.vision_continuous_servo_low_height_best_error_rebound_px)
            ),
            vision_continuous_servo_low_height_static_error_min_px=max(
                0.0, float(config.vision_continuous_servo_low_height_static_error_min_px)
            ),
            vision_continuous_servo_low_height_static_error_max_px=max(
                max(0.0, float(config.vision_continuous_servo_low_height_static_error_min_px)),
                float(config.vision_continuous_servo_low_height_static_error_max_px),
            ),
            vision_continuous_servo_low_height_static_frames=max(
                1, int(config.vision_continuous_servo_low_height_static_frames)
            ),
            vision_continuous_servo_low_height_static_improvement_px=max(
                0.05, float(config.vision_continuous_servo_low_height_static_improvement_px)
            ),
            vision_continuous_servo_low_height_static_band_mm=max(
                0.0, float(config.vision_continuous_servo_low_height_static_band_mm)
            ),
            vision_continuous_servo_low_height_static_pose_band_mm=max(
                0.0, float(config.vision_continuous_servo_low_height_static_pose_band_mm)
            ),
            vision_continuous_servo_camera_motion_guard_min_robot_mm=max(
                0.1, float(config.vision_continuous_servo_camera_motion_guard_min_robot_mm)
            ),
            vision_continuous_servo_camera_motion_guard_max_pixel_px=max(
                0.1, float(config.vision_continuous_servo_camera_motion_guard_max_pixel_px)
            ),
            vision_continuous_servo_camera_motion_guard_static_frames=max(
                1, int(config.vision_continuous_servo_camera_motion_guard_static_frames)
            ),
            vision_continuous_servo_low_height_rebound_recover_band_mm=max(
                0.0, float(config.vision_continuous_servo_low_height_rebound_recover_band_mm)
            ),
            vision_continuous_servo_low_height_rebound_recover_attempts=max(
                0, int(config.vision_continuous_servo_low_height_rebound_recover_attempts)
            ),
            vision_continuous_servo_low_height_refine_attempts=max(
                1, int(config.vision_continuous_servo_low_height_refine_attempts)
            ),
            vision_continuous_servo_low_height_refine_max_theta_step_deg=max(
                0.01, float(config.vision_continuous_servo_low_height_refine_max_theta_step_deg)
            ),
            vision_continuous_servo_low_height_refine_max_radius_step_mm=max(
                0.1, float(config.vision_continuous_servo_low_height_refine_max_radius_step_mm)
            ),
            vision_continuous_servo_max_center_jump_px=max(
                1.0, float(config.vision_continuous_servo_max_center_jump_px)
            ),
            vision_continuous_servo_max_error_growth_px=max(
                1.0, float(config.vision_continuous_servo_max_error_growth_px)
            ),
            vision_continuous_servo_stable_frames=max(1, int(config.vision_continuous_servo_stable_frames)),
            vision_continuous_servo_lost_frames=max(1, int(config.vision_continuous_servo_lost_frames)),
            vision_continuous_servo_stale_frames=max(1, int(config.vision_continuous_servo_stale_frames)),
            vision_continuous_servo_command_timeout_ms=max(
                1.0, float(config.vision_continuous_servo_command_timeout_ms)
            ),
            vision_continuous_servo_min_confidence=max(
                0.0, min(1.0, float(config.vision_continuous_servo_min_confidence))
            ),
            vision_continuous_servo_hard_min_confidence=max(
                0.0, min(1.0, float(config.vision_continuous_servo_hard_min_confidence))
            ),
            vision_continuous_servo_low_confidence_large_area_ratio=max(
                1.0, float(config.vision_continuous_servo_low_confidence_large_area_ratio)
            ),
            vision_continuous_servo_min_area_px=max(1, int(config.vision_continuous_servo_min_area_px)),
            vision_continuous_servo_ibvs_gain=max(
                0.01, min(1.0, float(config.vision_continuous_servo_ibvs_gain))
            ),
            vision_continuous_servo_ibvs_damping_px_per_unit=max(
                0.0, float(config.vision_continuous_servo_ibvs_damping_px_per_unit)
            ),
            vision_continuous_servo_ibvs_profile_stage_band_mm=max(
                0.0, float(config.vision_continuous_servo_ibvs_profile_stage_band_mm)
            ),
            vision_continuous_servo_pixel_jacobian_gain=max(
                0.01, min(1.0, float(config.vision_continuous_servo_pixel_jacobian_gain))
            ),
            vision_continuous_servo_pixel_axis_fine_band_px=max(
                0.1, float(config.vision_continuous_servo_pixel_axis_fine_band_px)
            ),
            vision_continuous_servo_pixel_axis_fine_rate_scale=max(
                0.05, min(1.0, float(config.vision_continuous_servo_pixel_axis_fine_rate_scale))
            ),
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
