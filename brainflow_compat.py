"""Compatibility helpers for BrainFlow on modern setuptools environments.

BrainFlow 5.21 still falls back to ``pkg_resources.resource_filename(...)``
when ``importlib.resources.files(__name__)`` is called with a module name like
``brainflow.board_shim``. Newer Python environments may no longer ship the
legacy ``pkg_resources`` module, so importing BrainFlow can fail before any
board connection starts.

This module installs a tiny ``pkg_resources`` shim only when the real module is
not available. The shim implements the subset BrainFlow needs:
``resource_filename(package_or_name, resource_name)``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any


def _normalize_resource_parts(resource_name: Any) -> tuple[str, ...]:
    raw = str(resource_name or "").replace("\\", "/")
    return tuple(part for part in raw.split("/") if part and part != ".")


def _resolve_resource_base(package_or_name: Any) -> Path:
    module = None
    if isinstance(package_or_name, types.ModuleType):
        module = package_or_name
    else:
        module_name = str(package_or_name or "").strip()
        if not module_name:
            raise ModuleNotFoundError("pkg_resources shim requires a module name")
        module = sys.modules.get(module_name)
        if module is None:
            module = importlib.import_module(module_name)

    module_path = getattr(module, "__path__", None)
    if module_path:
        first = next(iter(module_path), None)
        if first:
            return Path(first).resolve()

    module_file = getattr(module, "__file__", None)
    if module_file:
        return Path(module_file).resolve().parent

    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None)
    if origin:
        return Path(origin).resolve().parent

    spec = importlib.util.find_spec(getattr(module, "__name__", ""))
    if spec is not None:
        if spec.submodule_search_locations:
            first = next(iter(spec.submodule_search_locations), None)
            if first:
                return Path(first).resolve()
        if spec.origin:
            return Path(spec.origin).resolve().parent

    raise ModuleNotFoundError(f"Cannot resolve resource base for {package_or_name!r}")


def resource_filename(package_or_name: Any, resource_name: Any) -> str:
    base_dir = _resolve_resource_base(package_or_name)
    parts = _normalize_resource_parts(resource_name)
    return str(base_dir.joinpath(*parts).resolve())


def ensure_pkg_resources_shim() -> types.ModuleType:
    existing = sys.modules.get("pkg_resources")
    if existing is not None:
        if not hasattr(existing, "resource_filename"):
            setattr(existing, "resource_filename", resource_filename)
        return existing

    real_spec = importlib.util.find_spec("pkg_resources")
    if real_spec is not None:
        return importlib.import_module("pkg_resources")

    shim = types.ModuleType("pkg_resources")
    shim.__all__ = ["resource_filename"]
    shim.__brain_code_shim__ = True
    shim.resource_filename = resource_filename
    sys.modules["pkg_resources"] = shim
    return shim


def ensure_brainflow_compat() -> types.ModuleType:
    return ensure_pkg_resources_shim()


ensure_brainflow_compat()
