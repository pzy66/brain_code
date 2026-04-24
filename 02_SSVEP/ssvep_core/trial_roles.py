from __future__ import annotations

from typing import Any, Mapping, Optional


TRIAL_ROLE_CONTROL = "control"
TRIAL_ROLE_CLEAN_IDLE = "clean_idle"
TRIAL_ROLE_HARD_IDLE = "hard_idle"

_VALID_TRIAL_ROLES = {
    TRIAL_ROLE_CONTROL,
    TRIAL_ROLE_CLEAN_IDLE,
    TRIAL_ROLE_HARD_IDLE,
}


def infer_trial_role(*, label: str, expected_freq: Optional[float]) -> str:
    label_lower = str(label or "").strip().lower()
    if expected_freq is not None and not label_lower.startswith("switch_to_"):
        return TRIAL_ROLE_CONTROL
    if (
        "hard_idle" in label_lower
        or "hard idle" in label_lower
        or "switch" in label_lower
        or "transition" in label_lower
        or "scan" in label_lower
        or "long_idle" in label_lower
        or "long idle" in label_lower
    ):
        return TRIAL_ROLE_HARD_IDLE
    return TRIAL_ROLE_CLEAN_IDLE


def resolve_trial_role(row: Mapping[str, Any]) -> str:
    role = str(row.get("trial_role", "")).strip().lower()
    if role in _VALID_TRIAL_ROLES:
        return str(role)
    expected = row.get("expected_freq")
    expected_freq = None if expected is None else float(expected)
    return infer_trial_role(
        label=str(row.get("label", "")),
        expected_freq=expected_freq,
    )
