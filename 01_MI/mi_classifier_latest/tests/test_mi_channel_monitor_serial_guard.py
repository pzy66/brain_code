import os
import sys
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
REALTIME_ROOT = PROJECT_ROOT / "code" / "realtime"
SHARED_ROOT = PROJECT_ROOT / "code" / "shared"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REALTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(REALTIME_ROOT))
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

import brainflow_compat  # noqa: F401

from mi_realtime_channel_monitor import MonitorWindow


class MonitorWindowSerialGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_build_config_allows_manual_override_when_inventory_is_empty(self) -> None:
        with mock.patch("mi_realtime_channel_monitor.detect_serial_ports", return_value=[]):
            window = MonitorWindow()
        try:
            window.serial_combo.setCurrentText("COM7")
            with mock.patch(
                "mi_realtime_channel_monitor.validate_serial_port_selection",
                return_value={
                    "ok": True,
                    "reason": "manual_override",
                    "requested_port": "COM7",
                    "detected_ports": [],
                    "windows_status": "",
                    "problem_code": "",
                    "problem_status": "",
                },
            ), mock.patch.object(window, "log") as mock_log:
                config = window.build_config()

            self.assertEqual(config["serial_port"], "COM7")
            mock_log.assert_called_once()
        finally:
            window.close()

    def test_build_config_rejects_windows_unavailable_serial_port(self) -> None:
        with mock.patch("mi_realtime_channel_monitor.detect_serial_ports", return_value=[]):
            window = MonitorWindow()
        try:
            window.serial_combo.setCurrentText("COM7")
            with mock.patch(
                "mi_realtime_channel_monitor.validate_serial_port_selection",
                return_value={
                    "ok": False,
                    "reason": "windows_unavailable",
                    "requested_port": "COM7",
                    "detected_ports": ["COM4"],
                    "windows_status": "Problem",
                    "problem_code": "10",
                    "problem_status": "CM_PROB_FAILED_START",
                },
            ):
                with self.assertRaisesRegex(ValueError, "marked unavailable by Windows"):
                    window.build_config()
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
