"""Hybrid Controller package."""

import os
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_yolo_config_dir = _repo_root / ".ultralytics"
_yolo_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_yolo_config_dir))

__all__ = []
