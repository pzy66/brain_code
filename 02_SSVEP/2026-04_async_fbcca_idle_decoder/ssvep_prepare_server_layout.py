from __future__ import annotations

import argparse
import json
import sys
from pathlib import PurePosixPath
from typing import Sequence

from ssvep_core.server_layout import (
    DEFAULT_SERVER_BRAIN_ROOT,
    build_server_layout,
    ensure_server_layout,
    render_layout_lines,
    server_layout_as_dict,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the Linux server directory layout under /data1/zkx/brain for SSVEP training."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_SERVER_BRAIN_ROOT),
        help="target POSIX root, must stay under /data1/zkx",
    )
    parser.add_argument(
        "--include-mi-subdirs",
        type=int,
        default=0,
        help="when set to 1, also create code/data/reports/profiles/logs under /mi",
    )
    parser.add_argument(
        "--apply",
        type=int,
        default=0,
        help="when set to 1, actually create directories on the current Linux host",
    )
    parser.add_argument(
        "--json",
        type=int,
        default=0,
        help="when set to 1, emit the planned layout as JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_server_layout(
        PurePosixPath(str(args.root).strip()),
        include_mi_subdirs=bool(int(args.include_mi_subdirs)),
    )
    if bool(int(args.json)):
        print(json.dumps(server_layout_as_dict(plan), ensure_ascii=False, indent=2))
    else:
        for line in render_layout_lines(plan):
            print(line)
    if not bool(int(args.apply)):
        print("dry-run only; no directories were created")
        return 0
    created = ensure_server_layout(plan)
    print(f"created_or_verified={len(created)}")
    for path in created:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
