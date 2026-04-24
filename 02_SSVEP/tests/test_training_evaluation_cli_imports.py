from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def test_training_eval_cli_import_does_not_load_external_replay_dependencies() -> None:
    sys.modules.pop("tools.training_evaluation_cli", None)
    sys.modules.pop("ssvep_core.fbcca_external_replay_opt", None)
    sys.modules.pop("ssvep_core.external_replay_dataset", None)

    importlib.import_module("tools.training_evaluation_cli")

    assert "ssvep_core.fbcca_external_replay_opt" not in sys.modules
    assert "ssvep_core.external_replay_dataset" not in sys.modules
