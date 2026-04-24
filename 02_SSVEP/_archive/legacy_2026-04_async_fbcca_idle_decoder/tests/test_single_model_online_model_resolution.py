from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_single_model_ui import resolve_online_model_choice


def test_resolve_online_model_choice_same_model() -> None:
    model, mismatch = resolve_online_model_choice("fbcca", "fbcca")
    assert model == "fbcca"
    assert mismatch is False


def test_resolve_online_model_choice_uses_profile_model_on_mismatch() -> None:
    model, mismatch = resolve_online_model_choice("trca", "fbcca")
    assert model == "fbcca"
    assert mismatch is True
