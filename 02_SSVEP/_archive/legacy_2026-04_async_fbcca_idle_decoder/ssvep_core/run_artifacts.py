from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


def sanitize_artifact_token(value: str, *, fallback: str) -> str:
    token = "".join(ch if (str(ch).isalnum() or ch in {"-", "_"}) else "_" for ch in str(value).strip())
    token = token.strip("_")
    return token or fallback


def make_run_tag(*, task: str, now: Optional[datetime] = None, prefix: str = "run") -> str:
    stamp_time = now or datetime.now()
    task_token = sanitize_artifact_token(str(task).replace("_", "-"), fallback="task")
    base = f"{prefix}_{stamp_time.strftime('%Y%m%d_%H%M%S')}_{task_token}"
    return sanitize_artifact_token(base, fallback=f"{prefix}_{stamp_time.strftime('%Y%m%d_%H%M%S')}")


@dataclass(frozen=True)
class SSVEPRunArtifacts:
    task: str
    run_tag: str
    root_dir: Path
    run_dir: Path
    report_json: Path
    report_md: Path
    output_profile: Path
    profile_v2: Path
    selection_snapshot: Path
    run_config: Path
    run_log: Path
    progress_snapshot: Path
    figures_dir: Path

    def to_payload(self) -> dict[str, str]:
        return {
            "task": str(self.task),
            "run_tag": str(self.run_tag),
            "root_dir": str(self.root_dir),
            "run_dir": str(self.run_dir),
            "report_json": str(self.report_json),
            "report_md": str(self.report_md),
            "output_profile": str(self.output_profile),
            "profile_v2": str(self.profile_v2),
            "selection_snapshot": str(self.selection_snapshot),
            "run_config": str(self.run_config),
            "run_log": str(self.run_log),
            "progress_snapshot": str(self.progress_snapshot),
            "figures_dir": str(self.figures_dir),
        }


def resolve_ssvep_run_artifacts(
    *,
    task: str,
    report_path: Path,
    output_profile_path: Path,
    organize_report_dir: bool,
    report_root_dir: Optional[Path] = None,
    run_tag: Optional[str] = None,
    now: Optional[datetime] = None,
) -> SSVEPRunArtifacts:
    stamp_time = now or datetime.now()
    task_value = str(task or "ssvep-run").strip() or "ssvep-run"
    requested_report = Path(report_path).expanduser().resolve()
    requested_profile = Path(output_profile_path).expanduser().resolve()
    report_name = requested_report.name if requested_report.suffix.lower() == ".json" else "report.json"
    profile_name = requested_profile.name if requested_profile.suffix.lower() == ".json" else "profile.json"
    profile_stem = Path(profile_name).stem or "profile"
    if bool(organize_report_dir):
        root_dir = (
            Path(report_root_dir).expanduser().resolve()
            if report_root_dir is not None
            else requested_report.parent
        )
        task_dir = sanitize_artifact_token(task_value.replace("_", "-"), fallback="ssvep-run")
        effective_run_tag = sanitize_artifact_token(
            run_tag or make_run_tag(task=task_value, now=stamp_time),
            fallback=make_run_tag(task=task_value, now=stamp_time),
        )
        run_dir = root_dir / stamp_time.strftime("%Y%m%d") / task_dir / effective_run_tag
        report_json = run_dir / report_name
        output_profile = run_dir / profile_name
    else:
        root_dir = requested_report.parent
        effective_run_tag = sanitize_artifact_token(
            run_tag or requested_report.parent.name,
            fallback=make_run_tag(task=task_value, now=stamp_time),
        )
        run_dir = requested_report.parent
        report_json = requested_report
        output_profile = requested_profile
    return SSVEPRunArtifacts(
        task=task_value,
        run_tag=effective_run_tag,
        root_dir=root_dir,
        run_dir=run_dir,
        report_json=report_json,
        report_md=report_json.with_suffix(".md"),
        output_profile=output_profile,
        profile_v2=output_profile.with_name(f"{profile_stem}_v2.json"),
        selection_snapshot=run_dir / "selection_snapshot.json",
        run_config=run_dir / "run_config.json",
        run_log=run_dir / "run.log",
        progress_snapshot=run_dir / "progress_snapshot.json",
        figures_dir=run_dir / "figures",
    )
