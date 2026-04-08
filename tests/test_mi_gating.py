from hybrid_controller.adapters.mi_adapter import MIAdapter
from hybrid_controller.config import AppConfig


def test_mi_requires_stable_windows_and_confidence() -> None:
    adapter = MIAdapter(AppConfig())
    adapter.start()
    result = adapter.process_result({"stable_prediction_display_name": "Left Hand", "stable_confidence": 0.5}, timestamp_ms=100)
    assert result is None

    result = adapter.process_result({"stable_prediction_display_name": "Left Hand", "stable_confidence": 0.8}, timestamp_ms=200)
    assert result is None
    result = adapter.process_result({"stable_prediction_display_name": "Left Hand", "stable_confidence": 0.8}, timestamp_ms=300)
    assert result is None
    result = adapter.process_result({"stable_prediction_display_name": "Left Hand", "stable_confidence": 0.8}, timestamp_ms=600)
    assert result is not None
    assert result.value == "left"


def test_mi_min_interval_blocks_fast_repeats() -> None:
    adapter = MIAdapter(AppConfig(mi_emit_interval_ms=300, mi_stable_windows=1))
    adapter.start()
    first = adapter.process_result({"stable_prediction": "right_hand", "stable_confidence": 0.9}, timestamp_ms=100)
    second = adapter.process_result({"stable_prediction": "right_hand", "stable_confidence": 0.9}, timestamp_ms=200)
    third = adapter.process_result({"stable_prediction": "right_hand", "stable_confidence": 0.9}, timestamp_ms=450)
    assert first is not None
    assert second is None
    assert third is not None
    assert third.value == "right"


def test_mi_class_mapping_supports_all_four_classes() -> None:
    adapter = MIAdapter(AppConfig(mi_stable_windows=1))
    adapter.start()
    assert adapter.process_result({"stable_prediction": "left_hand", "stable_confidence": 0.9}, timestamp_ms=100).value == "left"
    assert adapter.process_result({"stable_prediction": "right_hand", "stable_confidence": 0.9}, timestamp_ms=500).value == "right"
    assert adapter.process_result({"stable_prediction": "feet", "stable_confidence": 0.9}, timestamp_ms=900).value == "backward"
    assert adapter.process_result({"stable_prediction": "tongue", "stable_confidence": 0.9}, timestamp_ms=1300).value == "forward"
