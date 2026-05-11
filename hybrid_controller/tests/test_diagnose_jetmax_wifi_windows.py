from __future__ import annotations

from hybrid_controller.tools.diagnose_jetmax_wifi_windows import analyze_snapshot


def test_analyze_snapshot_flags_default_route_dns_and_driver_recovery() -> None:
    findings = analyze_snapshot(
        {
            "wlan_interfaces_text": "SSID                   : HW-2DC157A4",
            "routes": [
                {
                    "DestinationPrefix": "0.0.0.0/0",
                    "NextHop": "192.168.149.1",
                    "RouteMetric": 0,
                    "InterfaceMetric": 50,
                },
                {
                    "DestinationPrefix": "192.168.149.0/24",
                    "NextHop": "0.0.0.0",
                },
            ],
            "dns": [
                {
                    "AddressFamily": "IPv4",
                    "ServerAddresses": ["192.168.149.1"],
                }
            ],
            "events": [
                {"Id": 4003, "Message": "WLAN AutoConfig detected limited connectivity."},
                {"Id": 8003, "Message": "Reason: 网络被驱动程序断开连接。"},
            ],
            "advanced": [
                {
                    "RegistryKeyword": "RoamAggressiveness",
                    "DisplayName": "Roaming Aggressiveness",
                    "DisplayValue": "Medium",
                    "RegistryValue": [2],
                },
                {
                    "RegistryKeyword": "*PacketCoalescing",
                    "DisplayName": "Packet Coalescing",
                    "DisplayValue": "Enabled",
                    "RegistryValue": [1],
                },
            ],
        },
        ssid_prefix="HW-",
        robot_host="192.168.149.1",
        robot_subnet="192.168.149.0/24",
    )

    codes = {finding.code for finding in findings}
    assert "wlan_default_route_present" in codes
    assert "wlan_dns_points_to_robot" in codes
    assert "wlan_limited_connectivity_recovery" in codes
    assert "wlan_driver_disconnect" in codes
    assert "roaming_aggressiveness_not_lowest" in codes
    assert "packet_coalescing_enabled" in codes


def test_analyze_snapshot_reports_ok_when_robot_lan_is_isolated() -> None:
    findings = analyze_snapshot(
        {
            "wlan_interfaces_text": "SSID                   : HW-2DC157A4",
            "routes": [
                {
                    "DestinationPrefix": "192.168.149.0/24",
                    "NextHop": "0.0.0.0",
                }
            ],
            "dns": [{"AddressFamily": "IPv4", "ServerAddresses": []}],
            "events": [],
            "advanced": [],
        },
        ssid_prefix="HW-",
        robot_host="192.168.149.1",
        robot_subnet="192.168.149.0/24",
    )

    assert [finding.code for finding in findings] == ["no_obvious_pc_side_wifi_risk"]
