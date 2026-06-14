from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROFILE_FILENAME_PATTERN = re.compile(r"ssvep_fbcca_profile_(\d{8}_\d{6})\.json$", flags=re.IGNORECASE)
FALLBACK_PROFILE_NAME = "fallback_fbcca_profile.json"
CURRENT_PROFILE_NAME = "current_fbcca_profile.json"


def build_timestamped_profile_path(profile_dir: Path, *, timestamp: datetime | None = None) -> Path:
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return Path(profile_dir) / f"ssvep_fbcca_profile_{stamp}.json"


def infer_profile_timestamp(path: Path) -> str | None:
    match = PROFILE_FILENAME_PATTERN.match(Path(path).name)
    if match is not None:
        return str(match.group(1))
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    saved_at = payload.get("saved_at")
    if isinstance(saved_at, str) and saved_at.strip():
        return saved_at.strip()
    return None


@dataclass(frozen=True, slots=True)
class ProfileSummary:
    name: str
    path: Path
    timestamp: str | None
    kind: str

    @property
    def display_name(self) -> str:
        label = self.name
        suffix_parts: list[str] = []
        if self.kind == "current":
            suffix_parts.append("当前")
        elif self.kind == "fallback":
            suffix_parts.append("默认")
        elif self.kind == "history":
            suffix_parts.append("历史")
        else:
            suffix_parts.append(self.kind)
        if self.timestamp:
            suffix_parts.append(self.timestamp)
        return f"{label} [{' / '.join(suffix_parts)}]"


class ProfileStore:
    def __init__(self, profile_dir: Path, current_profile_path: Path) -> None:
        self.profile_dir = Path(profile_dir).resolve()
        self.current_profile_path = Path(current_profile_path).resolve()
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.current_profile_path.parent.mkdir(parents=True, exist_ok=True)

    def copy_to_current(self, source_path: Path | str) -> None:
        source = Path(source_path).resolve()
        shutil.copy2(source, self.current_profile_path)

    def list_profiles(
        self,
        *,
        include_current_alias: bool = True,
        include_fallback: bool = True,
        limit: int | None = None,
    ) -> list[ProfileSummary]:
        summaries: list[ProfileSummary] = []
        for path in self.profile_dir.glob("*.json"):
            resolved = path.resolve()
            if not include_current_alias and resolved == self.current_profile_path:
                continue
            if not include_fallback and resolved.name.lower() == FALLBACK_PROFILE_NAME.lower():
                continue
            summaries.append(
                ProfileSummary(
                    name=resolved.name,
                    path=resolved,
                    timestamp=infer_profile_timestamp(resolved),
                    kind=self._classify_profile(resolved),
                )
            )

        def sort_key(summary: ProfileSummary) -> tuple[int, int, str]:
            kind_priority = {
                "current": 0,
                "history": 1,
                "fallback": 2,
                "other": 3,
            }.get(summary.kind, 4)
            timestamp_value = self._timestamp_sort_value(summary.timestamp)
            # History profiles should be sorted by timestamp descending (newest first).
            return (kind_priority, -timestamp_value, summary.name.lower())

        summaries.sort(key=sort_key)
        if limit is not None:
            return summaries[: max(int(limit), 0)]
        return summaries

    def latest_profile(self) -> ProfileSummary | None:
        for summary in self.list_profiles(include_current_alias=False, include_fallback=False):
            if summary.kind == "history":
                return summary
        return None

    def _classify_profile(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved == self.current_profile_path:
            return "current"
        if resolved.name.lower() == FALLBACK_PROFILE_NAME.lower():
            return "fallback"
        if PROFILE_FILENAME_PATTERN.match(resolved.name):
            return "history"
        return "other"

    @staticmethod
    def _timestamp_sort_value(timestamp: str | None) -> int:
        if not timestamp:
            return 0
        digits = "".join(char for char in timestamp if char.isdigit())
        if not digits:
            return 0
        return int(digits)
