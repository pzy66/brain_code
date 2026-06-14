from __future__ import annotations

from robot_workbench.app import DEFAULT_WORKBENCH_ARGS, build_config_from_args, build_legacy_forwarded_args


def _value_after(args: list[str], option: str) -> str:
    return args[args.index(option) + 1]


def test_robot_workbench_defaults_to_new_keyboard_flow_profile() -> None:
    args = list(DEFAULT_WORKBENCH_ARGS)

    assert _value_after(args, "--robot-mode") == "real"
    assert _value_after(args, "--robot-transport") == "ros"
    assert _value_after(args, "--rosbridge-port") == "9091"
    assert "--no-robot-connect-on-start" in args
    assert "--robot-runtime-auto-start" in args
    assert "--camera-auto-start" in args
    assert "--enable-vision" in args
    assert "--vision-auto-start" in args
    assert _value_after(args, "--vision-model-imgsz") == "768"
    assert "--eeg-signal-auto-start" in args
    assert _value_after(args, "--eeg-signal-window-sec") == "2.0"


def test_robot_workbench_builds_flow_config_from_overrides() -> None:
    config = build_config_from_args(
        [
            "--robot-mode",
            "fake",
            "--robot-host",
            "127.0.0.1",
            "--move-stage-ms",
            "1200",
            "--smoke-test-ms",
            "100",
            "--eeg-serial-port",
            "COM7",
            "--eeg-board-id",
            "2",
            "--no-eeg-signal-auto-start",
            "--eeg-signal-window-sec",
            "3.5",
            "--eeg-signal-poll-interval-sec",
            "0.08",
            "--camera-stream-url",
            "http://127.0.0.1:8080/stream?topic=/usb_cam/image_rect_color&type=mjpeg",
            "--no-camera-auto-start",
            "--no-robot-runtime-auto-start",
            "--robot-runtime-ssh-user",
            "robot",
            "--robot-runtime-ssh-password",
            "secret",
            "--robot-runtime-remote-root",
            "/tmp/brain_code",
            "--rosbridge-connect-timeout-sec",
            "1.5",
            "--ros-state-timeout-sec",
            "0.7",
        ]
    )

    assert config.robot_mode == "fake"
    assert config.robot_host == "127.0.0.1"
    assert config.move_stage_ms == 1200
    assert config.eeg_serial_port == "COM7"
    assert config.eeg_board_id == 2
    assert config.eeg_signal_auto_start is False
    assert config.eeg_signal_window_seconds == 3.5
    assert config.eeg_signal_poll_interval_sec == 0.08
    assert config.camera_stream_url.startswith("http://127.0.0.1:8080/stream")
    assert config.camera_auto_start is False
    assert config.vision_enabled is True
    assert config.vision_auto_start is True
    assert config.target_count == 4
    assert config.vision_max_targets == 4
    assert config.vision_max_det == 4
    assert config.vision_model_imgsz == 768
    assert config.robot_runtime_auto_start is False
    assert config.robot_runtime_ssh_user == "robot"
    assert config.robot_runtime_ssh_password == "secret"
    assert config.robot_runtime_remote_root == "/tmp/brain_code"
    assert config.rosbridge_connect_timeout_sec == 1.5
    assert config.ros_state_timeout_sec == 0.7
    assert config.smoke_test_ms == 100


def test_robot_workbench_accepts_enable_vision_as_compatibility_flag() -> None:
    config = build_config_from_args(["--enable-vision", "--robot-mode", "fake"])

    assert config.robot_mode == "fake"
    assert config.vision_enabled is True


def test_robot_workbench_can_disable_integrated_vision() -> None:
    config = build_config_from_args(["--disable-vision", "--no-vision-auto-start", "--robot-mode", "fake"])

    assert config.robot_mode == "fake"
    assert config.vision_enabled is False
    assert config.vision_auto_start is False


def test_robot_workbench_demo_connected_uses_safe_preview_defaults() -> None:
    config = build_config_from_args(
        [
            "--demo-connected",
            "--robot-mode",
            "real",
            "--robot-connect-on-start",
            "--eeg-signal-auto-start",
            "--camera-auto-start",
            "--vision-auto-start",
        ]
    )

    assert config.demo_connected is True
    assert config.robot_mode == "fake"
    assert config.connect_on_start is False
    assert config.eeg_signal_auto_start is False
    assert config.camera_auto_start is False
    assert config.vision_enabled is False
    assert config.vision_auto_start is False
    assert config.robot_runtime_auto_start is False


def test_robot_workbench_clamps_visual_target_count_to_operator_digits() -> None:
    config = build_config_from_args(["--target-count", "12", "--vision-max-targets", "12", "--vision-max-det", "20"])

    assert config.target_count == 4
    assert config.vision_max_targets == 4
    assert config.vision_max_det == 4


def test_legacy_hybrid_forwarding_still_supports_old_ui() -> None:
    args = build_legacy_forwarded_args(["--enable-vision", "--smoke-test-ms", "100"])

    assert "--enable-vision" not in args
    assert "operator_keyboard" in args
    assert "robot_camera_detection" in args
    assert "--vision-auto-start" in args
    assert args[-2:] == ["--smoke-test-ms", "100"]
