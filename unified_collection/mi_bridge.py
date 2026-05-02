"""Compatibility imports for the MI collection subsystem."""

from __future__ import annotations

from brain_workspace.paths import ensure_runtime_import_paths

ensure_runtime_import_paths()

from mi_data_collector import (  # noqa: E402
    BoardCaptureWorker,
    BoardIds,
    DEFAULT_CHANNEL_NAMES,
    MIDataCollectorWindow,
    RealtimeEEGPreviewWidget,
    available_board_options,
)
from src.mi_collection import parse_channel_names, parse_channel_positions  # noqa: E402
from src.serial_ports import detect_serial_ports  # noqa: E402

__all__ = [
    "BoardCaptureWorker",
    "BoardIds",
    "DEFAULT_CHANNEL_NAMES",
    "MIDataCollectorWindow",
    "RealtimeEEGPreviewWidget",
    "available_board_options",
    "parse_channel_names",
    "parse_channel_positions",
    "detect_serial_ports",
]
