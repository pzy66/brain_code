from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tools.training_evaluation_cli import _parse_task, build_parser


def test_profile_eval_alias_maps_from_pretrained_eval() -> None:
    assert _parse_task("pretrained_eval") == "profile-eval"


def test_profile_eval_parser_accepts_pretrained_profile_argument() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--task",
            "profile-eval",
            "--dataset-manifest",
            str(PROJECT_DIR / "artifacts" / "datasets" / "dummy_session" / "session_manifest.json"),
            "--pretrained-profile",
            str(PROJECT_DIR / "artifacts" / "deployed_profiles" / "default_profile.json"),
            "--profile-eval-mode",
            "fbcca-only",
            "--freeze-profile-weights",
            "1",
        ]
    )

    assert str(args.task) == "profile-eval"
    assert str(args.pretrained_profile).endswith("default_profile.json")
    assert str(args.profile_eval_mode) == "fbcca-only"
    assert int(args.freeze_profile_weights) == 1
