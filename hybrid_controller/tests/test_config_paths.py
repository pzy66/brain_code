from __future__ import annotations

from hybrid_controller.config import AppConfig


def test_default_config_prefers_existing_legacy_hybrid_dataset_files() -> None:
    config = AppConfig()

    assert config.vision_calibration_profile_path.name == "current_profile.json"
    assert "hybrid_controller\\dataset\\vision_calibration" in str(config.vision_calibration_profile_path)
    assert config.vision_calibration_profile_path.exists()

    assert config.pick_tuning_profile_path.name == "current_pick_tuning.json"
    assert "hybrid_controller\\dataset\\robot_pick_tuning" in str(config.pick_tuning_profile_path)
    assert config.pick_tuning_profile_path.exists()

    assert config.ssvep_profile_dir.name == "ssvep_profiles"
    assert "hybrid_controller\\dataset\\ssvep_profiles" in str(config.ssvep_profile_dir)
    assert config.ssvep_profile_dir.exists()

    assert config.ssvep_current_profile_path == config.ssvep_profile_dir / "current_fbcca_profile.json"
    assert config.ssvep_current_profile_path.exists()
    assert config.ssvep_default_profile_path == config.ssvep_profile_dir / "default_fbcca_profile.json"
    assert config.ssvep_default_profile_path.exists()
