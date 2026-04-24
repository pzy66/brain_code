from __future__ import annotations

import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from apps.training_evaluation_ui import build_parser
from ssvep_core.run_artifacts import resolve_ssvep_run_artifacts


def test_training_eval_parser_accepts_current_artifact_and_remote_args() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dataset-root",
            r"C:\tmp\artifacts\datasets",
            "--output-profile",
            r"C:\tmp\artifacts\deployed_profiles\default_profile.json",
            "--report-path",
            r"C:\tmp\artifacts\runs\local\report.json",
            "--report-root-dir",
            r"C:\tmp\artifacts\runs\local",
            "--include-manifests",
            r"C:\tmp\artifacts\datasets\s1\session_manifest.json,C:\tmp\artifacts\datasets\s2\session_manifest.json",
            "--task",
            "fbcca-weighted-compare",
            "--remote-mode",
            "1",
            "--enable-local-fallback",
            "0",
            "--server-host",
            "10.0.0.8",
            "--server-port",
            "22022",
            "--server-username",
            "zkx_user",
            "--server-password",
            "secret",
            "--headless",
        ]
    )

    assert str(args.dataset_root) == r"C:\tmp\artifacts\datasets"
    assert str(args.output_profile) == r"C:\tmp\artifacts\deployed_profiles\default_profile.json"
    assert str(args.report_path) == r"C:\tmp\artifacts\runs\local\report.json"
    assert str(args.report_root_dir) == r"C:\tmp\artifacts\runs\local"
    assert str(args.include_manifests).endswith(
        r"C:\tmp\artifacts\datasets\s1\session_manifest.json,C:\tmp\artifacts\datasets\s2\session_manifest.json"
    )
    assert str(args.task) == "fbcca-weighted-compare"
    assert int(args.remote_mode) == 1
    assert int(args.enable_local_fallback) == 0
    assert str(args.server_host) == "10.0.0.8"
    assert int(args.server_port) == 22022
    assert str(args.server_username) == "zkx_user"
    assert str(args.server_password) == "secret"
    assert bool(args.headless) is True


def test_run_artifact_resolver_organizes_training_eval_run_dir() -> None:
    tmp_root = PROJECT_DIR / ".tmp_test_artifacts" / f"artifact_layout_{uuid.uuid4().hex}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    try:
        payload = resolve_ssvep_run_artifacts(
            task="model-compare",
            report_path=tmp_root / "report.json",
            output_profile_path=tmp_root / "default_profile.json",
            organize_report_dir=True,
            report_root_dir=tmp_root / "artifacts" / "runs" / "local",
            run_tag="run_unit_check",
            now=datetime(2026, 4, 15, 14, 30, 0),
        )
        expected_run_dir = tmp_root / "artifacts" / "runs" / "local" / "model-compare" / "20260415" / "run_unit_check"
        assert payload.run_dir == expected_run_dir
        assert payload.report_json == expected_run_dir / "report.json"
        assert payload.report_md == expected_run_dir / "report.md"
        assert payload.output_profile == expected_run_dir / "default_profile.json"
        assert payload.profile_v2 == expected_run_dir / "default_profile_v2.json"
        assert payload.selection_snapshot == expected_run_dir / "selection_snapshot.json"
        assert payload.run_config == expected_run_dir / "run_config.json"
        assert payload.run_log == expected_run_dir / "run.log"
        assert payload.progress_snapshot == expected_run_dir / "progress_snapshot.json"
        assert payload.figures_dir == expected_run_dir / "figures"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_run_artifact_resolver_keeps_flat_paths_when_organization_is_disabled() -> None:
    tmp_root = PROJECT_DIR / ".tmp_test_artifacts" / f"artifact_flat_{uuid.uuid4().hex}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    try:
        report_path = tmp_root / "report.json"
        profile_path = tmp_root / "profile.json"
        payload = resolve_ssvep_run_artifacts(
            task="tdca-local-opt",
            report_path=report_path,
            output_profile_path=profile_path,
            organize_report_dir=False,
            run_tag="ignored_flat",
            now=datetime(2026, 4, 15, 14, 35, 0),
        )
        assert payload.run_dir == tmp_root
        assert payload.report_json == report_path
        assert payload.output_profile == profile_path
        assert payload.report_md == tmp_root / "report.md"
        assert payload.profile_v2 == tmp_root / "profile_v2.json"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
