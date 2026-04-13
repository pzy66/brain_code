from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_training_evaluation_ui import build_parser  # noqa: E402


def test_train_eval_ui_parser_defaults_remote_first() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert int(args.remote_mode) == 1
    assert int(args.enable_local_fallback) == 0
    assert str(args.compute_backend) == "cuda"


def test_train_eval_ui_parser_supports_server_fields() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--server-host",
            "10.0.0.8",
            "--server-port",
            "22022",
            "--server-username",
            "zkx_user",
            "--server-password",
            "secret",
            "--remote-mode",
            "1",
            "--enable-local-fallback",
            "0",
        ]
    )
    assert str(args.server_host) == "10.0.0.8"
    assert int(args.server_port) == 22022
    assert str(args.server_username) == "zkx_user"
    assert str(args.server_password) == "secret"
    assert int(args.remote_mode) == 1
    assert int(args.enable_local_fallback) == 0
