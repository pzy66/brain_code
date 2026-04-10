from __future__ import annotations

import pytest

from hybrid_controller.robot.runtime.runtime_core import CommandParseError, parse_command_text


def test_parse_command_text_success() -> None:
    command, args = parse_command_text("MOVE_CYL_AUTO 10 200")
    assert command == "MOVE_CYL_AUTO"
    assert args == ["10", "200"]


def test_parse_command_text_rejects_bad_arity() -> None:
    with pytest.raises(CommandParseError) as exc:
        parse_command_text("PICK_CYL 10")
    assert "requires theta r" in str(exc.value)


def test_parse_command_text_rejects_unknown_command() -> None:
    with pytest.raises(CommandParseError) as exc:
        parse_command_text("FOO 1 2")
    assert "Unsupported command" in str(exc.value)

