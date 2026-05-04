from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR.parent))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from apps.launcher_ui import main as launcher_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    return int(launcher_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
