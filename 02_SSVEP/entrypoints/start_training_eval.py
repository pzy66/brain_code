from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR.parent))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from apps.training_evaluation_ui import main as app_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    return int(app_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
