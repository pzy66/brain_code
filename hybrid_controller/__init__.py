"""Hybrid Controller package."""

import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_yolo_config_dir = _repo_root / ".ultralytics"
_yolo_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_yolo_config_dir))

if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from brainflow_compat import ensure_brainflow_compat


ensure_brainflow_compat()

__all__ = []
