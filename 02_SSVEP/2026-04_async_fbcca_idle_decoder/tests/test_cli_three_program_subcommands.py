from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from async_fbcca_idle_standalone import build_parser


def test_parser_supports_realtime_collect_train_eval() -> None:
    parser = build_parser()
    args_realtime = parser.parse_args(["realtime", "--serial-port", "COM4"])
    assert args_realtime.command == "realtime"
    assert args_realtime.serial_port == "COM4"

    args_collect = parser.parse_args(["collect", "--subject-id", "s01"])
    assert args_collect.command == "collect"
    assert args_collect.subject_id == "s01"

    args_train_eval = parser.parse_args(
        ["train-eval", "--dataset-manifest", str(PROJECT_DIR / "profiles" / "dummy.json")]
    )
    assert args_train_eval.command == "train-eval"
    assert str(args_train_eval.dataset_manifest).endswith("dummy.json")

