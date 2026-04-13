from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "2026-04_async_fbcca_idle_decoder" / "ssvep_server_train_client.py"


def main() -> int:
    cmd = [sys.executable, str(TARGET)]
    print("[launcher]", " ".join(cmd), flush=True)
    return int(subprocess.call(cmd))


if __name__ == "__main__":
    raise SystemExit(main())
