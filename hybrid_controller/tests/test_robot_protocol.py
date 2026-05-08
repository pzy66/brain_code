import math
from argparse import Namespace
from pathlib import Path

from hybrid_controller.adapters.control_sim_slots import ControlSimSlotCatalog
from hybrid_controller.adapters.robot_client import parse_robot_line
from hybrid_controller.app import build_config_from_args
from hybrid_controller.config import AppConfig
from hybrid_controller.robot.tools import jetmax_start_ros_runtime
from hybrid_controller.controller.events import Event
from hybrid_controller.controller.state_machine import TaskState
from hybrid_controller.controller.task_controller import TaskController


def test_parse_robot_protocol_lines() -> None:
    assert parse_robot_line("ACK PICK_DONE").type == "robot_ack"
    assert parse_robot_line("ACK PICK_DONE").value == "PICK_DONE"
    assert parse_robot_line("BUSY").type == "robot_busy"
    assert parse_robot_line("ERR bad").value == "bad"


def test_pick_done_only_advances_in_picking() -> None:
    controller = TaskController(AppConfig())
    controller.handle_event(Event(source="robot", type="robot_ack", value="PICK_DONE", timestamp=1.0))
    assert controller.state == TaskState.IDLE

    controller.handle_event(Event(source="system", type="start_task", timestamp=2.0))
    controller.state = TaskState.S2_PICKING
    effects = controller.handle_event(Event(source="robot", type="robot_ack", value="PICK_DONE", timestamp=3.0))
    assert controller.state == TaskState.S3_MI_CARRY
    assert any(effect.type == "start_timer" for effect in effects)


def test_place_done_only_advances_in_placing() -> None:
    controller = TaskController(AppConfig())
    controller.handle_event(Event(source="robot", type="robot_ack", value="PLACE_DONE", timestamp=1.0))
    assert controller.state == TaskState.IDLE

    controller.state = TaskState.S3_PLACING
    controller.context.carrying = True
    controller.handle_event(Event(source="robot", type="robot_ack", value="PLACE_DONE", timestamp=2.0))
    assert controller.state == TaskState.FINISHED
    assert controller.context.carrying is False


def test_robot_failure_enters_error_during_pick_and_place() -> None:
    controller = TaskController(AppConfig())
    controller.state = TaskState.S2_PICKING
    controller.handle_event(Event(source="robot", type="robot_busy", timestamp=1.0))
    assert controller.state == TaskState.ERROR

    controller.state = TaskState.S3_PLACING
    controller.handle_event(Event(source="robot", type="robot_error", value="jammed", timestamp=2.0))
    assert controller.state == TaskState.ERROR


def test_move_is_committed_only_after_ack() -> None:
    controller = TaskController(AppConfig())
    controller.handle_event(Event(source="system", type="start_task", timestamp=1.0))

    effects = controller.handle_event(Event(source="sim", type="move", value="right", timestamp=2.0))
    assert any(effect.type == "robot_command" for effect in effects)
    assert controller.context.robot_xy == AppConfig().robot_start_xy
    assert controller.context.pending_robot_cyl is not None
    assert math.isclose(controller.context.pending_robot_cyl[0], -4.0, abs_tol=1e-6)
    assert math.isclose(controller.context.pending_robot_cyl[1], 120.0, abs_tol=1e-6)

    controller.handle_event(Event(source="robot", type="robot_busy", timestamp=3.0))
    assert controller.context.robot_xy == AppConfig().robot_start_xy
    assert controller.context.pending_robot_xy is None
    assert controller.context.pending_robot_cyl is None

    controller.handle_event(Event(source="sim", type="move", value="right", timestamp=4.0))
    controller.handle_event(Event(source="robot", type="robot_ack", value="MOVE", timestamp=5.0))
    assert math.isclose(controller.context.robot_cyl[0], -4.0, abs_tol=1e-6)
    assert math.isclose(controller.context.robot_cyl[1], 120.0, abs_tol=1e-6)
    assert controller.context.pending_robot_xy is None


def test_confirm_pick_and_place_are_blocked_while_robot_busy() -> None:
    controller = TaskController(AppConfig())
    controller.state = TaskState.S2_GRAB_CONFIRM
    controller.context.selected_target_raw_center = (640.0, 360.0)
    controller.context.robot_busy = True
    effects = controller.handle_event(Event(source="sim", type="decision_confirm", timestamp=1.0))
    assert controller.state == TaskState.S2_GRAB_CONFIRM
    assert not any(effect.type == "robot_command" for effect in effects)


def test_fixed_world_slot_selection_emits_pick_world_command() -> None:
    config = AppConfig(vision_mode="fixed_world_slots")
    controller = TaskController(config)
    catalog = ControlSimSlotCatalog(config)
    target = catalog.build_selection_targets(source="hardware", command_mode="world")[0]
    controller.state = TaskState.S2_TARGET_SELECT
    controller.context.frozen_targets = [target]

    controller.handle_event(Event(source="sim", type="target_selected", value=0, timestamp=1.0))
    effects = controller.handle_event(Event(source="sim", type="decision_confirm", timestamp=2.0))

    commands = [effect.payload["command"] for effect in effects if effect.type == "robot_command"]
    assert commands == [f"PICK_WORLD {target.command_point[0]:.2f} {target.command_point[1]:.2f}"]

    controller.state = TaskState.S3_DECISION
    controller.context.robot_busy = True
    effects = controller.handle_event(Event(source="sim", type="decision_confirm", timestamp=2.0))
    assert controller.state == TaskState.S3_DECISION
    assert not any(effect.type == "robot_command" for effect in effects)


def test_fixed_cyl_slot_selection_emits_pick_cyl_command() -> None:
    config = AppConfig(vision_mode="fixed_cyl_slots")
    controller = TaskController(config)
    catalog = ControlSimSlotCatalog(config)
    target = catalog.build_selection_targets(source="hardware", command_mode="cyl")[0]
    controller.state = TaskState.S2_TARGET_SELECT
    controller.context.frozen_targets = [target]

    controller.handle_event(Event(source="sim", type="target_selected", value=0, timestamp=1.0))
    effects = controller.handle_event(Event(source="sim", type="decision_confirm", timestamp=2.0))

    commands = [effect.payload["command"] for effect in effects if effect.type == "robot_command"]
    assert commands == [f"PICK_CYL {target.command_point[0]:.2f} {target.command_point[1]:.2f}"]


def test_jetmax_camera_repair_defaults_keep_official_sender_settings() -> None:
    args = jetmax_start_ros_runtime.build_parser().parse_args([])

    assert args.skip_camera_check is True
    assert args.camera_stream_type == "mjpeg"
    assert args.camera_framerate == 20
    assert args.camera_io_method == "mmap"
    assert args.camera_quality == 80
    assert args.repair_camera_driver is False
    assert args.remove_camera_driver_override is False
    assert args.camera_driver_quirks == 128
    assert args.camera_driver_nodrop == 1
    assert args.camera_driver_timeout == 5000
    assert args.camera_driver_conf_path == "/etc/modprobe.d/hiwonder-uvcvideo.conf"
    assert args.manage_web_video is False
    assert args.repair_camera_sender is False
    assert args.allow_camera_sender_mutation is False


def test_jetmax_start_camera_stream_check_requires_explicit_opt_in() -> None:
    default_args = jetmax_start_ros_runtime.build_parser().parse_args([])
    check_args = jetmax_start_ros_runtime.build_parser().parse_args(["--check-camera-stream"])

    assert default_args.skip_camera_check is True
    assert check_args.skip_camera_check is False


def test_jetmax_start_camera_sender_mutation_is_guarded() -> None:
    parser = jetmax_start_ros_runtime.build_parser()
    repair_args = parser.parse_args(["--repair-camera-sender"])
    allowed_args = parser.parse_args(["--repair-camera-sender", "--allow-camera-sender-mutation"])
    manage_args = parser.parse_args(["--manage-web-video", "--allow-camera-sender-mutation"])

    assert jetmax_start_ros_runtime._camera_sender_mutation_requested(repair_args) is True
    assert jetmax_start_ros_runtime._camera_sender_mutation_requested(manage_args) is False
    assert repair_args.allow_camera_sender_mutation is False
    assert allowed_args.allow_camera_sender_mutation is True


def test_jetmax_camera_repair_script_clears_stale_rosparams_without_default_driver_reload() -> None:
    source = Path(jetmax_start_ros_runtime.__file__).read_text(encoding="utf-8")

    assert "rosparam delete /usb_cam" in source
    assert "remove_uvcvideo_override_file" in source
    assert "pkill -f web_video_server" not in source
    assert "uvcvideo quirks={quirks} nodrop={nodrop} timeout={timeout}" in source
    assert "timeout 6 rostopic hz /usb_cam/image_raw" not in source


def test_jetmax_runtime_start_does_not_repair_camera_sender_by_default() -> None:
    source = Path(jetmax_start_ros_runtime.__file__).read_text(encoding="utf-8")

    assert "if bool(args.repair_camera_sender) and not bool(args.skip_camera_repair):" in source
    assert "verify_official_camera_sender" in source
    assert "verify_web_video_mjpeg_stream" in source
    assert "--check-camera-stream" in source
    assert "checks.append(PortCheck(name=\"web_video_server\"" in source
    assert "persist_and_reload_uvcvideo(ssh, args=args, sudo=sudo)" in source
    assert "elif bool(args.repair_camera_driver) and not bool(args.skip_camera_driver_repair):" in source
    assert "repair_uvcvideo_driver(ssh, args=args, sudo=sudo)" in source
    assert "HYBRID_MANAGE_WEB_VIDEO" not in source
    assert "HYBRID_FORCE_RESTART_WEB_VIDEO" not in source
    assert "WEB_VIDEO_PORT=" not in source
    assert jetmax_start_ros_runtime.HIWONDER_CAMERA_TOPIC == "/usb_cam/image_rect_color"
    assert jetmax_start_ros_runtime.DEFAULT_CAMERA_STREAM_PATH == (
        "/stream?topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80"
    )


def test_jetmax_runtime_default_start_does_not_check_camera_stream() -> None:
    args = jetmax_start_ros_runtime.build_parser().parse_args([])

    checks = [jetmax_start_ros_runtime.PortCheck(name="rosbridge", port=int(args.rosbridge_port), required=True)]
    if not bool(args.skip_camera_check):
        checks.append(jetmax_start_ros_runtime.PortCheck(name="web_video_server", port=int(args.web_video_port), required=True))

    assert [check.name for check in checks] == ["rosbridge"]


def test_build_config_from_args_applies_modes() -> None:
    args = Namespace(
        timing_profile="fast",
        scenario_name="sparse_targets",
        slot_profile="default",
        robot_mode="real",
        vision_mode="fixed_world_slots",
        move_source="sim",
        decision_source="ssvep",
        robot_host="192.168.1.9",
        robot_port=9999,
        vision_stream_url="camera://demo",
        vision_auto_start=False,
        smoke_test_ms=0,
    )
    config = build_config_from_args(args)
    assert config.robot_mode == "real"
    assert config.vision_mode == "fixed_world_slots"
    assert config.move_source == "sim"
    assert config.decision_source == "ssvep"
    assert config.robot_host == "192.168.1.9"
    assert config.robot_port == 9999
    assert config.vision_stream_url == "camera://demo"
    assert config.vision_auto_start is False
    assert config.timing_profile == "fast"
    assert config.scenario_name == "sparse_targets"
    assert config.control_sim_enabled is True
    assert config.stage_motion_sec == 2.0
    assert config.sim_pick_delay_sec == 0.2


def test_build_config_from_args_supports_mi_placeholder_flags() -> None:
    args = Namespace(
        timing_profile="formal",
        scenario_name="basic",
        slot_profile="default",
        robot_mode="real",
        vision_mode="robot_camera_detection",
        move_source="mi",
        decision_source="ssvep",
        mi_backend="brainflow",
        mi_enabled=True,
        mi_poll_interval_ms=60,
        mi_command_cooldown_ms=140,
        robot_host="192.168.1.9",
        robot_port=9999,
        vision_stream_url="",
        vision_auto_start=False,
        smoke_test_ms=0,
    )
    config = build_config_from_args(args)
    assert config.move_source == "mi"
    assert config.mi_backend == "brainflow"
    assert config.mi_enabled is True
    assert config.mi_poll_interval_ms == 60
    assert config.mi_command_cooldown_ms == 140
