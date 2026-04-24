import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = PROJECT_ROOT / "code" / "shared"
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from src.serial_ports import _parse_windows_pnp_ports, detect_serial_ports, validate_serial_port_selection


class SerialPortDetectionTests(TestCase):
    def test_parse_windows_pnp_ports_extracts_port_status_pairs_from_text(self) -> None:
        raw_output = """
Microsoft PnP Utility

Instance ID:                FTDIBUS\\VID_0403+PID_6015+D30DVCDYA\\0000
Device Description:         USB Serial Port (COM4)
Class Name:                 Ports
Status:                     Disconnected

Instance ID:                ROOT\\PORTS\\0000
Device Description:         Broken Port (COM3)
Class Name:                 Ports
Status:                     Problem
"""
        self.assertEqual(
            _parse_windows_pnp_ports(raw_output),
            [("COM4", "Disconnected"), ("COM3", "Problem")],
        )

    def test_parse_windows_pnp_ports_extracts_port_status_pairs_from_csv(self) -> None:
        raw_output = """InstanceId,DeviceDescription,ClassName,ClassGuid,ManufacturerName,Status,ProblemCode,ProblemStatus,DriverName,ExtensionDriverNames
"USB\\VID_1234&PID_0001\\A","USB 串行设备 (COM4)","Ports","{4d36e978-e325-11ce-bfc1-08002be10318}","Vendor","Started","","","serial.inf",""
"USB\\VID_1234&PID_0002\\B","损坏端口 (COM7)","Ports","{4d36e978-e325-11ce-bfc1-08002be10318}","Vendor","Problem","10","CM_PROB_FAILED_START","serial.inf",""
"""
        self.assertEqual(
            _parse_windows_pnp_ports(raw_output),
            [("COM4", "Started"), ("COM7", "Problem")],
        )

    @patch("src.serial_ports.os.name", "nt")
    @patch(
        "src.serial_ports._detect_windows_pnp_ports",
        return_value=[
            {"port": "COM4", "status": "Disconnected", "problem_code": "", "problem_status": "", "description": ""},
            {"port": "COM3", "status": "Problem", "problem_code": "10", "problem_status": "", "description": ""},
        ],
    )
    @patch("src.serial_ports._detect_pyserial_ports", return_value=["COM3", "COM4"])
    def test_detect_serial_ports_filters_unavailable_ports_from_windows_inventory(
        self,
        _mock_pyserial,
        _mock_windows_ports,
    ) -> None:
        self.assertEqual(detect_serial_ports(), [])

    @patch("src.serial_ports.os.name", "nt")
    @patch(
        "src.serial_ports._detect_windows_pnp_ports",
        return_value=[
            {"port": "COM4", "status": "Started", "problem_code": "", "problem_status": "", "description": ""}
        ],
    )
    @patch("src.serial_ports._detect_pyserial_ports", return_value=["COM4"])
    def test_detect_serial_ports_keeps_healthy_windows_ports(
        self,
        _mock_pyserial,
        _mock_windows_ports,
    ) -> None:
        self.assertEqual(detect_serial_ports(), ["COM4"])

    @patch("src.serial_ports.os.name", "nt")
    @patch("src.serial_ports._detect_windows_pnp_ports", return_value=[])
    @patch("src.serial_ports._detect_pyserial_ports", return_value=[])
    def test_detect_serial_ports_does_not_synthesize_placeholder_com_ports(
        self,
        _mock_pyserial,
        _mock_windows_ports,
    ) -> None:
        self.assertEqual(detect_serial_ports(), [])

    @patch("src.serial_ports.os.name", "posix")
    @patch("src.serial_ports._detect_pyserial_ports", return_value=["/dev/ttyUSB0"])
    def test_detect_serial_ports_preserves_case_sensitive_device_paths_on_posix(
        self,
        _mock_pyserial,
    ) -> None:
        self.assertEqual(detect_serial_ports(), ["/dev/ttyUSB0"])

    @patch("src.serial_ports.os.name", "nt")
    @patch("src.serial_ports._detect_windows_pnp_ports", return_value=[])
    @patch("src.serial_ports._detect_pyserial_ports", return_value=[])
    def test_validate_serial_port_selection_allows_manual_override_when_inventory_is_empty(
        self,
        _mock_pyserial,
        _mock_windows_ports,
    ) -> None:
        validation = validate_serial_port_selection("COM7")
        self.assertTrue(validation["ok"])
        self.assertEqual(validation["reason"], "manual_override")
        self.assertEqual(validation["requested_port"], "COM7")

    @patch("src.serial_ports.os.name", "nt")
    @patch("src.serial_ports._detect_windows_pnp_ports", return_value=[])
    @patch("src.serial_ports._detect_pyserial_ports", return_value=["COM4"])
    def test_validate_serial_port_selection_rejects_unknown_port_when_other_ports_are_detected(
        self,
        _mock_pyserial,
        _mock_windows_ports,
    ) -> None:
        validation = validate_serial_port_selection("COM7")
        self.assertFalse(validation["ok"])
        self.assertEqual(validation["reason"], "not_detected")
        self.assertEqual(validation["detected_ports"], ["COM4"])

    @patch("src.serial_ports.os.name", "nt")
    @patch(
        "src.serial_ports._detect_windows_pnp_ports",
        return_value=[
            {
                "port": "COM7",
                "status": "Problem",
                "problem_code": "10",
                "problem_status": "CM_PROB_FAILED_START",
                "description": "Broken Port (COM7)",
            }
        ],
    )
    @patch("src.serial_ports._detect_pyserial_ports", return_value=[])
    def test_validate_serial_port_selection_rejects_explicitly_unavailable_windows_port(
        self,
        _mock_pyserial,
        _mock_windows_ports,
    ) -> None:
        validation = validate_serial_port_selection("COM7")
        self.assertFalse(validation["ok"])
        self.assertEqual(validation["reason"], "windows_unavailable")
        self.assertEqual(validation["problem_code"], "10")
