"""Shared TCP command parsing utilities for robot runtimes."""
from typing import List, Tuple


class CommandParseError(ValueError):
    pass


COMMAND_ARG_COUNTS = {
    "PING": 0,
    "STATUS": 0,
    "RESET": 0,
    "ABORT": 0,
    "PLACE": 0,
    "MOVE": 2,
    "MOVE_CYL": 3,
    "MOVE_CYL_AUTO": 2,
    "PICK": 2,
    "PICK_WORLD": 2,
    "PICK_CYL": 2,
}

COMMAND_ARG_MESSAGES = {
    "STATUS": "STATUS takes no arguments",
    "RESET": "RESET takes no arguments",
    "ABORT": "ABORT takes no arguments",
    "PLACE": "PLACE takes no arguments",
    "MOVE": "MOVE requires x y",
    "MOVE_CYL": "MOVE_CYL requires theta r z",
    "MOVE_CYL_AUTO": "MOVE_CYL_AUTO requires theta r",
    "PICK": "PICK requires pixel_x pixel_y",
    "PICK_WORLD": "PICK_WORLD requires x y",
    "PICK_CYL": "PICK_CYL requires theta r",
}


def parse_command_text(line: str) -> Tuple[str, List[str]]:
    text = str(line).strip()
    if not text:
        raise CommandParseError("Empty command")
    parts = text.split()
    command = parts[0].upper()
    expected = COMMAND_ARG_COUNTS.get(command)
    if expected is None:
        raise CommandParseError("Unsupported command: {}".format(text))
    args = list(parts[1:])
    if len(args) != int(expected):
        message = COMMAND_ARG_MESSAGES.get(command, "Invalid command arguments")
        raise CommandParseError(message)
    return command, args
