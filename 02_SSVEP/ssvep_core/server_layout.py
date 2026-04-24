from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


DEFAULT_SERVER_BRAIN_ROOT = PurePosixPath("/data1/zkx/brain")
DEFAULT_SERVER_SSVEP_SUBDIRS = ("code", "data", "reports", "profiles", "logs")


@dataclass(frozen=True)
class ServerLayoutPlan:
    root: PurePosixPath
    ssvep_root: PurePosixPath
    mi_root: PurePosixPath
    ssvep_dirs: tuple[PurePosixPath, ...]
    mi_dirs: tuple[PurePosixPath, ...] = ()

    def all_dirs(self) -> tuple[PurePosixPath, ...]:
        ordered = [self.root, self.ssvep_root, self.mi_root]
        ordered.extend(self.ssvep_dirs)
        ordered.extend(self.mi_dirs)
        unique: list[PurePosixPath] = []
        seen: set[str] = set()
        for item in ordered:
            key = str(item)
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return tuple(unique)


def normalize_server_root(raw: str | Path | PurePosixPath | None = None) -> PurePosixPath:
    value = str(raw or DEFAULT_SERVER_BRAIN_ROOT).strip()
    path = PurePosixPath(value)
    if not path.is_absolute():
        raise ValueError(f"server root must be an absolute POSIX path: {value}")
    if len(path.parts) < 3 or tuple(path.parts[:3]) != ("/", "data1", "zkx"):
        raise ValueError(f"server root must stay under /data1/zkx: {value}")
    return path


def build_server_layout(
    root: str | Path | PurePosixPath | None = None,
    *,
    include_mi_subdirs: bool = False,
) -> ServerLayoutPlan:
    root_path = normalize_server_root(root)
    ssvep_root = root_path / "ssvep"
    mi_root = root_path / "mi"
    ssvep_dirs = tuple(ssvep_root / name for name in DEFAULT_SERVER_SSVEP_SUBDIRS)
    mi_dirs = tuple((mi_root / name) for name in DEFAULT_SERVER_SSVEP_SUBDIRS) if include_mi_subdirs else ()
    return ServerLayoutPlan(
        root=root_path,
        ssvep_root=ssvep_root,
        mi_root=mi_root,
        ssvep_dirs=ssvep_dirs,
        mi_dirs=mi_dirs,
    )


def ensure_server_layout(plan: ServerLayoutPlan) -> tuple[Path, ...]:
    if os.name != "posix":
        raise RuntimeError("apply mode is only supported on POSIX systems such as the target Linux server")
    created: list[Path] = []
    for item in plan.all_dirs():
        target = Path(str(item))
        target.mkdir(parents=True, exist_ok=True)
        created.append(target)
    return tuple(created)


def render_layout_lines(plan: ServerLayoutPlan) -> tuple[str, ...]:
    lines = [f"root: {plan.root}"]
    lines.append("ssvep:")
    lines.extend(f"  - {path}" for path in plan.ssvep_dirs)
    lines.append("mi:")
    if plan.mi_dirs:
        lines.extend(f"  - {path}" for path in plan.mi_dirs)
    else:
        lines.append(f"  - {plan.mi_root}")
    return tuple(lines)


def server_layout_as_dict(plan: ServerLayoutPlan) -> dict[str, object]:
    return {
        "root": str(plan.root),
        "ssvep_root": str(plan.ssvep_root),
        "mi_root": str(plan.mi_root),
        "ssvep_dirs": [str(path) for path in plan.ssvep_dirs],
        "mi_dirs": [str(path) for path in plan.mi_dirs],
    }
