from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core.async_fbcca_idle_standalone import json_dumps, json_safe
from ssvep_core.fbcca_base_profile_opt import (
    DEFAULT_BASE_DATASET_ROOT,
    DEFAULT_FBCCA_BASE_PROFILE_PATH,
    DEFAULT_FBCCA_BASE_REPORT_PATH,
    FBCCABaseProfileOptConfig,
    run_fbcca_base_profile_opt,
)
from ssvep_core.fbcca_local_opt import DEFAULT_FBCCA_LOCAL_SEARCH_PRESET, FBCCA_LOCAL_SEARCH_PRESETS


def _parse_manifest_csv(value: str) -> tuple[Path, ...]:
    return tuple(Path(item.strip()) for item in str(value or "").split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize the offline FBCCA base profile for the 8/10/12/15Hz demo.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_BASE_DATASET_ROOT)
    parser.add_argument("--dataset-manifests", type=str, default="")
    parser.add_argument("--output-profile", type=Path, default=DEFAULT_FBCCA_BASE_PROFILE_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_FBCCA_BASE_REPORT_PATH)
    parser.add_argument("--search-preset", choices=FBCCA_LOCAL_SEARCH_PRESETS, default=DEFAULT_FBCCA_LOCAL_SEARCH_PRESET)
    parser.add_argument("--compute-backend", type=str, default="auto")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_fbcca_base_profile_opt(
        FBCCABaseProfileOptConfig(
            dataset_manifests=_parse_manifest_csv(str(args.dataset_manifests)),
            dataset_root=Path(args.dataset_root),
            output_profile_path=Path(args.output_profile),
            report_path=Path(args.report),
            search_preset=str(args.search_preset),
            compute_backend=str(args.compute_backend),
            organize_report_dir=False,
        ),
        log_fn=lambda message: print(message, flush=True),
    )
    print(json_dumps(json_safe(payload)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
