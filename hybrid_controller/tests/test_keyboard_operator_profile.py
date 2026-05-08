from __future__ import annotations

from argparse import Namespace
from unittest import mock

from hybrid_controller.app import HybridControllerApplication, build_config_from_args
from hybrid_controller.config import AppConfig
from hybrid_controller.run_real import DEFAULT_ARGS


def _base_args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "input_profile": AppConfig.input_profile,
        "robot_mode": "real",
        "robot_transport": AppConfig.robot_transport,
        "vision_mode": "robot_camera_detection",
        "move_source": "sim",
        "decision_source": "sim",
        "ssvep_runtime_enabled": False,
        "mi_backend": AppConfig.mi_backend,
        "mi_enabled": False,
        "mi_poll_interval_ms": AppConfig.mi_poll_interval_ms,
        "mi_command_cooldown_ms": AppConfig.mi_command_cooldown_ms,
        "timing_profile": "formal",
        "scenario_name": "basic",
        "slot_profile": "default",
        "robot_host": AppConfig.robot_host,
        "robot_port": AppConfig.robot_port,
        "rosbridge_port": AppConfig.rosbridge_port,
        "vision_stream_url": "",
        "vision_auto_start": AppConfig.vision_auto_start,
        "stage_motion_sec": None,
        "continue_motion_sec": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_run_real_defaults_to_keyboard_operator_profile() -> None:
    args = list(DEFAULT_ARGS)

    assert args[args.index("--move-source") + 1] == "sim"
    assert args[args.index("--decision-source") + 1] == "sim"
    assert args[args.index("--input-profile") + 1] == "operator_keyboard"
    assert args[args.index("--robot-mode") + 1] == "real"
    assert args[args.index("--robot-transport") + 1] == "ros"
    assert args[args.index("--vision-mode") + 1] == "robot_camera_detection"
    assert "--vision-auto-start" not in args
    assert "--enable-ssvep-runtime" not in args
    assert "--mi-enabled" not in args


def test_build_config_keyboard_operator_defaults_disable_bci_runtime_sources() -> None:
    config = build_config_from_args(_base_args())

    assert config.input_profile == "operator_keyboard"
    assert config.move_source == "sim"
    assert config.decision_source == "sim"
    assert config.mi_enabled is False
    assert config.ssvep_runtime_enabled is False
    assert config.vision_auto_start is False


def test_build_config_allows_explicit_vision_auto_start() -> None:
    config = build_config_from_args(_base_args(vision_auto_start=True))

    assert config.vision_auto_start is True


def test_build_config_allows_explicit_bci_source_override() -> None:
    config = build_config_from_args(_base_args(mi_enabled=True, move_source="mi", decision_source="sim"))

    assert config.input_profile == "operator_keyboard"
    assert config.move_source == "mi"
    assert config.mi_enabled is True


def test_build_config_ssvep_decision_explicitly_enables_runtime() -> None:
    config = build_config_from_args(
        _base_args(
            input_profile="bci_experimental",
            decision_source="ssvep",
            ssvep_runtime_enabled=False,
        )
    )

    assert config.input_profile == "bci_experimental"
    assert config.decision_source == "ssvep"
    assert config.ssvep_runtime_enabled is True


def test_default_hybrid_app_does_not_construct_bci_providers() -> None:
    config = AppConfig(
        input_profile="operator_keyboard",
        move_source="sim",
        decision_source="sim",
        mi_enabled=False,
        ssvep_runtime_enabled=False,
    )
    app = HybridControllerApplication.__new__(HybridControllerApplication)
    app.config = config

    assert app._build_mi_provider_if_enabled() is None
    assert app._ssvep_runtime_enabled() is False

    with mock.patch("hybrid_controller.adapters.mi_input.MiInputProvider") as mi_provider:
        assert app._build_mi_provider_if_enabled() is None
        mi_provider.assert_not_called()
