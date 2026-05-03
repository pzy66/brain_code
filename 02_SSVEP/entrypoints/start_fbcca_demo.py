from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from apps.realtime_online_ui import main as app_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    extra = list(sys.argv[1:] if argv is None else argv)
    demo_defaults = [
        "--demo-mode",
        "1",
        "--freqs",
        "8,10,12,15",
        "--model",
        "fbcca",
    ]
    return int(app_main([*extra, *demo_defaults]))


if __name__ == "__main__":
    raise SystemExit(main())
