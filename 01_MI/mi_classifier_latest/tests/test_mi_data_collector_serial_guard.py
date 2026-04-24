import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_ROOT = PROJECT_ROOT / "code" / "collection"
SHARED_ROOT = PROJECT_ROOT / "code" / "shared"
if str(COLLECTION_ROOT) not in sys.path:
    sys.path.insert(0, str(COLLECTION_ROOT))
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from mi_data_collector import BoardIds, MIDataCollectorWindow, available_board_options


class MIDataCollectorSerialGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @patch("mi_data_collector.detect_serial_ports", return_value=[])
    def test_refresh_serial_ports_does_not_fallback_to_com3_when_nothing_is_detected(self, _mock_detect) -> None:
        window = MIDataCollectorWindow()
        try:
            window.serial_combo.setCurrentText("")
            window.refresh_serial_ports()
            self.assertEqual(window.serial_combo.currentText().strip(), "")
        finally:
            window.close()

    @patch(
        "mi_data_collector.validate_serial_port_selection",
        return_value={
            "ok": True,
            "reason": "manual_override",
            "requested_port": "COM4",
            "detected_ports": [],
            "windows_status": "",
            "problem_code": "",
            "problem_status": "",
        },
    )
    def test_collect_settings_allows_manual_port_override_when_inventory_is_empty(self, _mock_validate) -> None:
        window = MIDataCollectorWindow()
        try:
            window.serial_combo.setCurrentText("COM4")
            settings = window.collect_settings()
            self.assertEqual(settings.serial_port, "COM4")
        finally:
            window.close()

    @patch(
        "mi_data_collector.validate_serial_port_selection",
        return_value={
            "ok": False,
            "reason": "windows_unavailable",
            "requested_port": "COM7",
            "detected_ports": ["COM4"],
            "windows_status": "Problem",
            "problem_code": "10",
            "problem_status": "CM_PROB_FAILED_START",
        },
    )
    def test_collect_settings_rejects_windows_unavailable_serial_port(self, _mock_validate) -> None:
        window = MIDataCollectorWindow()
        try:
            window.serial_combo.setCurrentText("COM7")
            with self.assertRaisesRegex(ValueError, "被系统标记为不可用"):
                window.collect_settings()
        finally:
            window.close()

    def test_unsupported_ganglion_board_is_not_exposed(self) -> None:
        ganglion = getattr(BoardIds, "GANGLION_BOARD", None)
        if ganglion is None:
            self.skipTest("BrainFlow does not expose GANGLION_BOARD in this environment.")

        exposed_ids = {int(board_id) for _label, board_id in available_board_options()}
        self.assertNotIn(int(ganglion.value), exposed_ids)

    @patch("mi_data_collector.detect_serial_ports", return_value=["COM4"])
    def test_collect_settings_rejects_unsupported_board_even_if_injected(self, _mock_detect) -> None:
        ganglion = getattr(BoardIds, "GANGLION_BOARD", None)
        if ganglion is None:
            self.skipTest("BrainFlow does not expose GANGLION_BOARD in this environment.")

        window = MIDataCollectorWindow()
        try:
            window.board_combo.addItem("Ganglion（测试注入）", int(ganglion.value))
            window.board_combo.setCurrentIndex(window.board_combo.count() - 1)
            window.serial_combo.setCurrentText("COM4")
            with self.assertRaisesRegex(ValueError, "当前采集流程只支持"):
                window.collect_settings()
        finally:
            window.close()
