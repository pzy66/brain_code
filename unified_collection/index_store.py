"""CSV index helpers for unified MI/SSVEP collection sessions."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

UNIFIED_INDEX_FIELDS = [
    "task_type",
    "subject_id",
    "session_id",
    "board_id",
    "serial_port",
    "sampling_rate",
    "started_at",
    "ended_at",
    "status",
    "native_manifest_path",
    "continuous_path",
]


def wallclock_iso_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def append_unified_collection_index(index_path: Path, row: dict[str, Any]) -> Path:
    path = Path(index_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNIFIED_INDEX_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in UNIFIED_INDEX_FIELDS})
    return path
