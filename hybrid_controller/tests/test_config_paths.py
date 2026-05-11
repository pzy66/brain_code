from __future__ import annotations

from hybrid_controller.config import AppConfig


def test_default_config_uses_canonical_dataset_files() -> None:
    config = AppConfig()

    assert config.vision_calibration_profile_path.name == "current_profile.json"
    assert "datasets\\vision\\calibration" in str(config.vision_calibration_profile_path)
    assert config.vision_calibration_profile_path.exists()

    assert config.pick_tuning_profile_path.name == "current_pick_tuning.json"
    assert "datasets\\profiles\\hybrid_controller\\robot_pick_tuning" in str(config.pick_tuning_profile_path)
    assert config.pick_tuning_profile_path.exists()

    assert config.vision_grasp_profile_path.name == "current_grasp_profile.json"
    assert "datasets\\profiles\\hybrid_controller\\vision_grasp" in str(config.vision_grasp_profile_path)
    assert config.vision_grasp_profile_path.exists()

    assert config.ssvep_profile_dir.name == "ssvep_profiles"
    assert "datasets\\profiles\\hybrid_controller\\ssvep_profiles" in str(config.ssvep_profile_dir)
    assert config.ssvep_profile_dir.exists()

    assert config.ssvep_current_profile_path == config.ssvep_profile_dir / "current_fbcca_profile.json"
    assert config.ssvep_current_profile_path.exists()
    assert config.ssvep_default_profile_path == config.ssvep_profile_dir / "default_fbcca_profile.json"
    assert config.ssvep_default_profile_path.exists()
