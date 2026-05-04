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
    defaults = [
        "--task",
        "fbcca-threshold-pretrain",
        "--remote-mode",
        "0",
        "--enable-local-fallback",
        "1",
    ]
    extra = list(sys.argv[1:] if argv is None else argv)
    return int(app_main([*defaults, *extra]))


if __name__ == "__main__":
    raise SystemExit(main())
