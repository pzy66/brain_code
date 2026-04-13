from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ArtifactKey:
    subject: str
    session: str
    model: str
    version: str = "v1"
    date: str | None = None

    def resolved_date(self) -> str:
        if self.date:
            return str(self.date)
        return datetime.now().strftime("%Y%m%d")


class ArtifactStore:
    """
    Artifact layout:
      <root>/<subject>/<session>/<date>/<model>/<version>/
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve_dir(self, key: ArtifactKey) -> Path:
        directory = (
            self.root
            / str(key.subject)
            / str(key.session)
            / str(key.resolved_date())
            / str(key.model)
            / str(key.version)
        )
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def state_path(self, key: ArtifactKey) -> Path:
        return self.resolve_dir(key) / "model_state.json"

    def profile_v2_path(self, key: ArtifactKey) -> Path:
        return self.resolve_dir(key) / "profile_v2.json"

    def report_path(self, key: ArtifactKey) -> Path:
        return self.resolve_dir(key) / "benchmark_report.json"

