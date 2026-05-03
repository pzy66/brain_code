from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core.dataset import LoadedDataset
from ssvep_core.fbcca_base_profile_opt import (
    FBCCABaseProfileOptConfig,
    discover_fbcca_base_dataset_manifests,
    run_fbcca_base_profile_opt,
    validate_fbcca_base_dataset_manifests,
)


def _dataset(path: Path, *, freqs: tuple[float, float, float, float]) -> LoadedDataset:
    return LoadedDataset(
        manifest_path=path,
        npz_path=path.with_name("raw_trials.npz"),
        session_id=path.parent.name,
        subject_id="subject-test",
        sampling_rate=250,
        freqs=freqs,
        board_eeg_channels=(0, 1, 2, 3, 4, 5, 6, 7),
        protocol_config={},
        trial_segments=[],
        manifest={},
    )


def test_base_profile_manifest_discovery_and_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import ssvep_core.fbcca_base_profile_opt as module

    manifest_a = tmp_path / "a" / "session_manifest.json"
    manifest_b = tmp_path / "nested" / "b" / "session_manifest.json"
    manifest_a.parent.mkdir(parents=True)
    manifest_b.parent.mkdir(parents=True)
    manifest_a.write_text("{}", encoding="utf-8")
    manifest_b.write_text("{}", encoding="utf-8")

    discovered = discover_fbcca_base_dataset_manifests(tmp_path)
    assert discovered == (manifest_a.resolve(), manifest_b.resolve())

    monkeypatch.setattr(
        module,
        "load_collection_dataset",
        lambda path: _dataset(Path(path), freqs=(8.0, 10.0, 12.0, 15.0)),
    )
    captured: dict[str, object] = {}

    def fake_run(config, **_kwargs):
        captured["config"] = config
        Path(config.output_profile_path).parent.mkdir(parents=True, exist_ok=True)
        Path(config.output_profile_path).write_text("{}", encoding="utf-8")
        Path(config.output_profile_path).with_name(f"{Path(config.output_profile_path).stem}_v2.json").write_text(
            "{}",
            encoding="utf-8",
        )
        return {
            "task": "fbcca-local-opt",
            "chosen_model": "fbcca",
            "profile_saved": True,
            "profile_v2_saved": True,
            "chosen_profile_path": str(config.output_profile_path),
            "profile_v2_path": str(Path(config.output_profile_path).with_name(f"{Path(config.output_profile_path).stem}_v2.json")),
        }

    monkeypatch.setattr(module, "run_fbcca_local_opt", fake_run)
    output_profile = tmp_path / "deployed" / "fbcca_base_profile.json"
    payload = run_fbcca_base_profile_opt(
        FBCCABaseProfileOptConfig(
            dataset_root=tmp_path,
            output_profile_path=output_profile,
            report_path=tmp_path / "report.json",
            search_preset="smoke20",
            compute_backend="cpu",
        )
    )

    assert payload["task"] == "fbcca-base-profile-opt"
    assert payload["chosen_model"] == "fbcca"
    assert output_profile.exists()
    assert output_profile.with_name("fbcca_base_profile_v2.json").exists()
    cfg = captured["config"]
    assert tuple(Path(path) for path in cfg.dataset_manifests) == (manifest_a.resolve(), manifest_b.resolve())
    assert Path(cfg.output_profile_path) == output_profile
    assert cfg.search_preset == "smoke20"


def test_base_profile_rejects_frequency_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import ssvep_core.fbcca_base_profile_opt as module

    manifest = tmp_path / "session_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "load_collection_dataset",
        lambda path: _dataset(Path(path), freqs=(8.0, 9.0, 12.0, 15.0)),
    )

    with pytest.raises(ValueError, match="expected=\\(8.0, 10.0, 12.0, 15.0\\)"):
        validate_fbcca_base_dataset_manifests((manifest,))
