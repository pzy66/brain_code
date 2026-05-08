from __future__ import annotations

import json

from hybrid_controller.tools.calibrate_suction_target_pixel import _write_profile_target


def test_write_stage_target_pixel_preserves_top_level_servo_target(tmp_path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "profile_id": "unit-profile",
                "image_size": [640, 480],
                "servo": {"target_pixel": [320.0, 240.0]},
                "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, 0], [0, 1, 0]]},
            }
        ),
        encoding="utf-8",
    )

    _write_profile_target(profile_path, (320.0, 230.0), stage="confirm", z_mm=120.0)

    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    assert payload["servo"]["target_pixel"] == [320.0, 240.0]
    assert payload["stage_models"]["confirm"]["z_mm"] == 120.0
    assert payload["stage_models"]["confirm"]["servo"]["target_pixel"] == [320.0, 230.0]
