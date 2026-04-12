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
        [
            "train-eval",
            "--dataset-manifest",
            str(PROJECT_DIR / "profiles" / "dummy.json"),
            "--spatial-filter-mode",
            "trca_shared",
            "--spatial-rank-candidates",
            "1,2,3",
            "--joint-weight-iters",
            "2",
            "--spatial-source-model",
            "trca",
            "--async-decision-time-mode",
            "first-correct",
            "--data-policy",
            "new-only",
        ]
    )
    assert args_train_eval.command == "train-eval"
    assert str(args_train_eval.dataset_manifest).endswith("dummy.json")
    assert str(args_train_eval.spatial_filter_mode) == "trca_shared"
    assert str(args_train_eval.spatial_rank_candidates) == "1,2,3"
    assert str(args_train_eval.async_decision_time_mode) == "first-correct"
    assert str(args_train_eval.data_policy) == "new-only"
