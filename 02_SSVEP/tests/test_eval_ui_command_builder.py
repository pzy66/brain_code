from __future__ import annotations

import json
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from apps import training_evaluation_ui as train_eval_ui
from apps.training_evaluation_ui import TrainingEvaluationWindow, build_parser
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


def test_publish_to_hybrid_controller_uses_publish_time_for_history(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "old_good_profile.json"
    source.write_text(
        json.dumps(
            {
                "model_name": "fbcca",
                "freqs": [8.0, 10.0, 12.0, 15.0],
                "saved_at": "2026-04-13T22:42:32",
            }
        ),
        encoding="utf-8",
    )
    hybrid_dir = tmp_path / "hybrid_profiles"
    current_path = hybrid_dir / "current_fbcca_profile.json"
    monkeypatch.setattr(train_eval_ui, "HYBRID_PROFILE_DIR", hybrid_dir)
    monkeypatch.setattr(train_eval_ui, "HYBRID_CURRENT_PROFILE_PATH", current_path)
    monkeypatch.setattr(train_eval_ui, "_now_stamp", lambda: "20260430_120000")

    class DummyWindow:
        def __init__(self) -> None:
            self._last_profile_path = source
            self.logs: list[str] = []

        def _log(self, text: str) -> None:
            self.logs.append(str(text))

    dummy = DummyWindow()
    TrainingEvaluationWindow._publish_profile_to_hybrid_controller(dummy)

    assert current_path.exists()
    assert (hybrid_dir / "ssvep_fbcca_profile_20260430_120000.json").exists()
    assert not (hybrid_dir / "ssvep_fbcca_profile_20260413_224232.json").exists()
    assert any("已发布到集成控制器" in item for item in dummy.logs)


def test_publish_to_ssvep_realtime_writes_dedicated_fbcca_profile(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "profile.json"
    source.write_text(
        json.dumps(
            {
                "model_name": "fbcca",
                "freqs": [8.0, 10.0, 12.0, 15.0],
                "saved_at": "2026-04-30T12:00:00",
            }
        ),
        encoding="utf-8",
    )
    source_v2 = tmp_path / "profile_v2.json"
    source_v2.write_text(json.dumps({"schema": "profile_v2"}), encoding="utf-8")
    realtime_profile = tmp_path / "deployed" / "fbcca_profile.json"
    realtime_profile_v2 = tmp_path / "deployed" / "fbcca_profile_v2.json"
    monkeypatch.setattr(train_eval_ui, "SSVEP_REALTIME_PROFILE_PATH", realtime_profile)
    monkeypatch.setattr(train_eval_ui, "SSVEP_REALTIME_PROFILE_V2_PATH", realtime_profile_v2)

    class DummyWindow:
        def __init__(self) -> None:
            self._last_profile_path = source
            self.logs: list[str] = []

        def _log(self, text: str) -> None:
            self.logs.append(str(text))

    dummy = DummyWindow()
    TrainingEvaluationWindow._publish_profile_to_ssvep_realtime(dummy)

    assert realtime_profile.exists()
    assert realtime_profile_v2.exists()
    assert json.loads(realtime_profile.read_text(encoding="utf-8"))["model_name"] == "fbcca"
    assert any("已发布到 SSVEP 实时识别" in item for item in dummy.logs)


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
