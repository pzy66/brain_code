"""Repository-local Python startup hooks."""

from __future__ import annotations

try:
    from brainflow_compat import ensure_brainflow_compat
except Exception:
    ensure_brainflow_compat = None

if ensure_brainflow_compat is not None:
    ensure_brainflow_compat()
