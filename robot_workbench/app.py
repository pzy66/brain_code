from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from hybrid_controller.run_real import _normalize_legacy_rosbridge_port

from .flow_ui import (
    DEFAULT_VISION_CALIBRATION_PROFILE_PATH,
    DEFAULT_VISION_WEIGHTS_PATH,
    WorkbenchConfig,
    run_workbench,
)


DEFAULT_ROBOT_HOST = os.environ.get("BRAIN_ROBOT_HOST", "192.168.149.1")

DEFAULT_WORKBENCH_ARGS = [
    "--robot-mode",
    "real",
    "--robot-transport",
    "ros",
    "--robot-host",
    DEFAULT_ROBOT_HOST,
    "--robot-port",
    "8888",
    "--rosbridge-port",
    "9091",
    "--no-robot-connect-on-start",
    "--robot-runtime-auto-start",
    "--camera-auto-start",
    "--enable-vision",
    "--vision-auto-start",
    "--vision-model-imgsz",
    "768",
    "--eeg-signal-auto-start",
    "--eeg-signal-window-sec",
    "2.0",
]

LEGACY_HYBRID_ARGS = [
    "--input-profile",
    "operator_keyboard",
    "--robot-mode",
    "real",
    "--robot-transport",
    "ros",
    "--robot-host",
    DEFAULT_ROBOT_HOST,
    "--robot-port",
    "8888",
    "--rosbridge-port",
    "9091",
    "--no-robot-connect-on-start",
    "--robot-auto-start-disabled",
    "--robot-auto-start-max-attempts",
    "1",
    "--robot-auto-start-cooldown-sec",
    "60",
    "--idle-runtime-tick-interval-ms",
    "250",
    "--ros-reconnect-base-delay-sec",
    "3.0",
    "--ros-reconnect-max-delay-sec",
    "30.0",
    "--ros-reconnect-jitter-ratio",
    "0.25",
    "--vision-mode",
    "fixed_world_slots",
    "--no-vision-auto-start",
    "--no-vision-continuous-servo",
    "--move-source",
    "sim",
    "--decision-source",
    "sim",
    "--timing-profile",
    "formal",
    "--scenario-name",
    "basic",
    "--stage-motion-sec",
    "300",
    "--continue-motion-sec",
    "300",
]

VISION_ASSIST_ARGS = [
    "--vision-mode",
    "robot_camera_detection",
    "--vision-auto-start",
    "--vision-continuous-servo-enabled",
]


def build_legacy_forwarded_args(argv: Sequence[str]) -> list[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--enable-vision",
        action="store_true",
        help="Enable the legacy camera detection profile. Requires optional vision dependencies.",
    )
    args, forwarded = parser.parse_known_args(list(argv))
    forwarded = _normalize_legacy_rosbridge_port(list(forwarded))
    defaults = list(LEGACY_HYBRID_ARGS)
    if args.enable_vision:
        defaults.extend(VISION_ASSIST_ARGS)
    return [*defaults, *forwarded]


def build_config_from_args(argv: Sequence[str]) -> WorkbenchConfig:
    parser = argparse.ArgumentParser(description="Keyboard-operated BCI robot workbench")
    parser.add_argument("--robot-mode", choices=("real", "fake"), default="real")
    parser.add_argument("--robot-transport", choices=("ros", "tcp"), default="ros")
    parser.add_argument("--robot-host", default=DEFAULT_ROBOT_HOST)
    parser.add_argument("--robot-port", type=int, default=8888)
    parser.add_argument("--rosbridge-port", type=int, default=9091)
    parser.add_argument("--robot-connect-on-start", action="store_true", default=False)
    parser.add_argument("--no-robot-connect-on-start", action="store_false", dest="robot_connect_on_start")
    parser.add_argument("--robot-runtime-auto-start", action="store_true", default=True)
    parser.add_argument("--no-robot-runtime-auto-start", action="store_false", dest="robot_runtime_auto_start")
    parser.add_argument("--robot-runtime-ssh-user", default="hiwonder")
    parser.add_argument("--robot-runtime-ssh-password", default="hiwonder")
    parser.add_argument("--robot-runtime-remote-root", default="/home/hiwonder/brain_code")
    parser.add_argument("--rosbridge-connect-timeout-sec", type=float, default=4.0)
    parser.add_argument("--ros-state-timeout-sec", type=float, default=4.0)
    parser.add_argument("--eeg-serial-port", default="auto")
    parser.add_argument("--eeg-board-id", type=int, default=0)
    parser.add_argument("--eeg-signal-auto-start", action="store_true", default=True)
    parser.add_argument("--no-eeg-signal-auto-start", action="store_false", dest="eeg_signal_auto_start")
    parser.add_argument("--eeg-signal-window-sec", type=float, default=2.0)
    parser.add_argument("--eeg-signal-poll-interval-sec", type=float, default=0.05)
    parser.add_argument("--move-stage-ms", type=int, default=10_000)
    parser.add_argument("--teleop-theta-rate-deg-s", type=float, default=80.0)
    parser.add_argument("--teleop-radius-rate-mm-s", type=float, default=160.0)
    parser.add_argument("--camera-stream-url", default="")
    parser.add_argument("--camera-auto-start", action="store_true", default=True)
    parser.add_argument("--no-camera-auto-start", action="store_false", dest="camera_auto_start")
    parser.add_argument("--smoke-test-ms", type=int, default=0)
    parser.add_argument(
        "--demo-connected",
        action="store_true",
        default=False,
        help="Start in UI demo mode with robot and EEG gate marked as connected.",
    )
    parser.add_argument("--target-count", type=int, default=4)
    parser.add_argument(
        "--enable-vision",
        action="store_true",
        default=True,
        help="Enable wood-block vision detection in the integrated robot camera view.",
    )
    parser.add_argument("--disable-vision", action="store_false", dest="enable_vision", help=argparse.SUPPRESS)
    parser.add_argument("--vision-auto-start", action="store_true", default=True)
    parser.add_argument("--no-vision-auto-start", action="store_false", dest="vision_auto_start")
    parser.add_argument("--vision-weights-path", default=DEFAULT_VISION_WEIGHTS_PATH)
    parser.add_argument("--vision-model-imgsz", type=int, default=768)
    parser.add_argument("--vision-confidence-threshold", type=float, default=0.25)
    parser.add_argument("--vision-max-targets", type=int, default=4)
    parser.add_argument("--vision-max-det", type=int, default=4)
    parser.add_argument("--vision-mapping-mode", choices=("delta_servo", "absolute_base"), default="delta_servo")
    parser.add_argument("--vision-calibration-profile-path", default=DEFAULT_VISION_CALIBRATION_PROFILE_PATH)
    parser.add_argument("--vision-calibration-profile-required", action="store_true", default=True)
    parser.add_argument(
        "--no-vision-calibration-profile-required",
        action="store_false",
        dest="vision_calibration_profile_required",
    )
    parser.add_argument("--legacy-hybrid-ui", action="store_true", help=argparse.SUPPRESS)
    args, _unknown = parser.parse_known_args(_normalize_legacy_rosbridge_port(list(argv)))
    demo_connected = bool(args.demo_connected)
    return WorkbenchConfig(
        robot_mode="fake" if demo_connected else str(args.robot_mode),
        robot_transport=str(args.robot_transport),
        robot_host=str(args.robot_host),
        robot_port=int(args.robot_port),
        rosbridge_port=int(args.rosbridge_port),
        connect_on_start=False if demo_connected else bool(args.robot_connect_on_start),
        eeg_serial_port=str(args.eeg_serial_port or "auto"),
        eeg_board_id=int(args.eeg_board_id),
        eeg_signal_auto_start=False if demo_connected else bool(args.eeg_signal_auto_start),
        eeg_signal_window_seconds=max(1.0, float(args.eeg_signal_window_sec)),
        eeg_signal_poll_interval_sec=max(0.02, float(args.eeg_signal_poll_interval_sec)),
        theta_rate_deg_s=float(args.teleop_theta_rate_deg_s),
        radius_rate_mm_s=float(args.teleop_radius_rate_mm_s),
        move_stage_ms=max(500, int(args.move_stage_ms)),
        camera_stream_url=str(args.camera_stream_url or ""),
        camera_auto_start=False if demo_connected else bool(args.camera_auto_start),
        target_count=max(1, min(4, int(args.target_count))),
        vision_enabled=False if demo_connected else bool(args.enable_vision),
        vision_auto_start=False if demo_connected else bool(args.vision_auto_start),
        vision_weights_path=str(args.vision_weights_path or DEFAULT_VISION_WEIGHTS_PATH),
        vision_model_imgsz=max(128, int(args.vision_model_imgsz)),
        vision_confidence_threshold=max(0.01, min(0.99, float(args.vision_confidence_threshold))),
        vision_max_targets=max(1, min(4, int(args.vision_max_targets))),
        vision_max_det=max(1, min(4, int(args.vision_max_det))),
        vision_mapping_mode=str(args.vision_mapping_mode or "delta_servo"),
        vision_calibration_profile_path=str(
            args.vision_calibration_profile_path or DEFAULT_VISION_CALIBRATION_PROFILE_PATH
        ),
        vision_calibration_profile_required=bool(args.vision_calibration_profile_required),
        robot_runtime_auto_start=False if demo_connected else bool(args.robot_runtime_auto_start),
        robot_runtime_ssh_user=str(args.robot_runtime_ssh_user or "hiwonder"),
        robot_runtime_ssh_password=str(args.robot_runtime_ssh_password or "hiwonder"),
        robot_runtime_remote_root=str(args.robot_runtime_remote_root or "/home/hiwonder/brain_code"),
        rosbridge_connect_timeout_sec=max(0.5, float(args.rosbridge_connect_timeout_sec)),
        ros_state_timeout_sec=max(0.2, float(args.ros_state_timeout_sec)),
        demo_connected=demo_connected,
        smoke_test_ms=max(0, int(args.smoke_test_ms)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = sys.argv[1:] if argv is None else list(argv)
    if "--legacy-hybrid-ui" in raw_args:
        from hybrid_controller.app import main as legacy_main

        legacy_args = [arg for arg in raw_args if arg != "--legacy-hybrid-ui"]
        return int(legacy_main(build_legacy_forwarded_args(legacy_args)))
    return int(run_workbench(build_config_from_args(raw_args)))


if __name__ == "__main__":
    raise SystemExit(main())
