from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


DEFAULT_SSID_PREFIX = "HW-"
DEFAULT_WLAN_ALIAS = "WLAN"
DEFAULT_ROBOT_HOST = "192.168.149.1"
DEFAULT_ROBOT_SUBNET = "192.168.149.0/24"


@dataclass(frozen=True, slots=True)
class Finding:
    code: str
    severity: str
    message: str
    evidence: str = ""


def _run_text(command: list[str], *, timeout_sec: float = 10.0) -> str:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=float(timeout_sec),
        check=False,
    )
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(details or f"command failed with exit code {completed.returncode}: {command!r}")
    return completed.stdout


def _run_powershell_json(script: str, *, timeout_sec: float = 10.0) -> Any:
    payload = _run_text(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        timeout_sec=timeout_sec,
    ).strip()
    if not payload:
        return None
    return json.loads(payload)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _string(value: Any) -> str:
    return "" if value is None else str(value)


def _registry_values(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [_string(item).strip("{} ") for item in values if _string(item).strip("{} ")]


def collect_windows_snapshot(
    *,
    interface_alias: str,
    event_hours: float,
    max_events: int,
) -> dict[str, Any]:
    alias_json = json.dumps(str(interface_alias))
    since_hours = max(0.1, float(event_hours))
    max_events = max(20, int(max_events))
    return {
        "wlan_interfaces_text": _run_text(["netsh", "wlan", "show", "interfaces"], timeout_sec=10.0),
        "net_ip_config": _run_powershell_json(
            "$alias = {0}; "
            "Get-NetIPConfiguration -InterfaceAlias $alias -ErrorAction SilentlyContinue | "
            "Select-Object InterfaceAlias,InterfaceIndex,InterfaceDescription,"
            "@{{n='IPv4Address';e={{($_.IPv4Address | ForEach-Object {{$_.IPv4Address}})}}}},"
            "@{{n='IPv4DefaultGateway';e={{($_.IPv4DefaultGateway | ForEach-Object {{$_.NextHop}})}}}},"
            "@{{n='DNSServer';e={{$_.DNSServer.ServerAddresses}}}} | ConvertTo-Json -Depth 5".format(alias_json),
            timeout_sec=10.0,
        ),
        "routes": _run_powershell_json(
            "$alias = {0}; "
            "Get-NetRoute -InterfaceAlias $alias -AddressFamily IPv4 -ErrorAction SilentlyContinue | "
            "Select-Object DestinationPrefix,NextHop,RouteMetric,InterfaceMetric,State,PolicyStore | "
            "ConvertTo-Json -Depth 4".format(alias_json),
            timeout_sec=10.0,
        ),
        "dns": _run_powershell_json(
            "$alias = {0}; "
            "Get-DnsClientServerAddress -InterfaceAlias $alias -ErrorAction SilentlyContinue | "
            "Select-Object InterfaceAlias,AddressFamily,ServerAddresses | ConvertTo-Json -Depth 4".format(alias_json),
            timeout_sec=10.0,
        ),
        "ip_interface": _run_powershell_json(
            "$alias = {0}; "
            "Get-NetIPInterface -InterfaceAlias $alias -AddressFamily IPv4 -ErrorAction SilentlyContinue | "
            "Select-Object InterfaceAlias,InterfaceIndex,Dhcp,AutomaticMetric,InterfaceMetric,ConnectionState | "
            "ConvertTo-Json -Depth 4".format(alias_json),
            timeout_sec=10.0,
        ),
        "advanced": _run_powershell_json(
            "$alias = {0}; "
            "Get-NetAdapterAdvancedProperty -Name $alias -ErrorAction SilentlyContinue | "
            "Select-Object DisplayName,DisplayValue,RegistryKeyword,RegistryValue | ConvertTo-Json -Depth 4".format(
                alias_json
            ),
            timeout_sec=10.0,
        ),
        "events": _run_powershell_json(
            "$start = (Get-Date).AddHours(-{0}); "
            "$ids = 4003,8003,8001,11004,11005,11010; "
            "Get-WinEvent -LogName 'Microsoft-Windows-WLAN-AutoConfig/Operational' -MaxEvents {1} | "
            "Where-Object {{ $_.TimeCreated -ge $start -and $ids -contains $_.Id }} | "
            "Select-Object TimeCreated,Id,LevelDisplayName,Message | ConvertTo-Json -Depth 4".format(
                since_hours,
                max_events,
            ),
            timeout_sec=15.0,
        ),
        "system_events": _run_powershell_json(
            "$start = (Get-Date).AddHours(-{0}); "
            "Get-WinEvent -FilterHashtable @{{LogName='System'; StartTime=$start}} -MaxEvents {1} | "
            "Where-Object {{ "
            "$_.ProviderName -match 'Microsoft-Windows-WLAN-AutoConfig|Netwtw|Ndis|Tcpip|Dhcp' -or "
            "$_.Message -match 'Intel\\(R\\) Wi-Fi|AX211|limited connectivity|HW-|网络|驱动' "
            "}} | Select-Object TimeCreated,Id,ProviderName,LevelDisplayName,Message | ConvertTo-Json -Depth 4".format(
                since_hours,
                max_events,
            ),
            timeout_sec=15.0,
        ),
    }


def analyze_snapshot(
    snapshot: dict[str, Any],
    *,
    ssid_prefix: str,
    robot_host: str,
    robot_subnet: str,
) -> list[Finding]:
    findings: list[Finding] = []
    wlan_text = _string(snapshot.get("wlan_interfaces_text"))
    if ssid_prefix and ssid_prefix not in wlan_text:
        findings.append(
            Finding(
                code="ssid_not_detected",
                severity="warn",
                message=f"Current WLAN interface text does not show a JetMax SSID prefix '{ssid_prefix}'.",
            )
        )

    routes = _as_list(snapshot.get("routes"))
    for route in routes:
        if not isinstance(route, dict):
            continue
        if _string(route.get("DestinationPrefix")) == "0.0.0.0/0":
            next_hop = _string(route.get("NextHop"))
            findings.append(
                Finding(
                    code="wlan_default_route_present",
                    severity="warn",
                    message=(
                        "The JetMax WLAN has an IPv4 default route. For stable debugging, this Wi-Fi should normally "
                        f"carry only {robot_subnet}; Internet routing should stay on Ethernet or another adapter."
                    ),
                    evidence=f"0.0.0.0/0 via {next_hop}",
                )
            )

    dns_items = _as_list(snapshot.get("dns"))
    for item in dns_items:
        address_family = _string(item.get("AddressFamily"))
        if not isinstance(item, dict) or address_family not in {"IPv4", "2"}:
            continue
        servers = [_string(value) for value in _as_list(item.get("ServerAddresses"))]
        if robot_host in servers:
            findings.append(
                Finding(
                    code="wlan_dns_points_to_robot",
                    severity="warn",
                    message=(
                        "The JetMax WLAN DNS server points to the robot. This can make Windows/NCSI evaluate the "
                        "robot AP as a local-only or limited-connectivity network."
                    ),
                    evidence=", ".join(servers),
                )
            )

    events = [*_as_list(snapshot.get("events")), *_as_list(snapshot.get("system_events"))]
    limited = [event for event in events if isinstance(event, dict) and int(event.get("Id", 0)) == 4003]
    driver_disconnects = [
        event
        for event in events
        if isinstance(event, dict)
        and int(event.get("Id", 0)) == 8003
        and ("driver" in _string(event.get("Message")).lower() or "驱动" in _string(event.get("Message")))
    ]
    if limited:
        findings.append(
            Finding(
                code="wlan_limited_connectivity_recovery",
                severity="error",
                message="Windows WLAN AutoConfig recently attempted limited-connectivity recovery.",
                evidence=f"{len(limited)} event(s) with ID 4003",
            )
        )
    if driver_disconnects:
        findings.append(
            Finding(
                code="wlan_driver_disconnect",
                severity="error",
                message="Windows logged a driver-initiated WLAN disconnect for the recent window.",
                evidence=f"{len(driver_disconnects)} event(s) with ID 8003",
            )
        )

    advanced_by_keyword: dict[str, dict[str, Any]] = {}
    for item in _as_list(snapshot.get("advanced")):
        if isinstance(item, dict):
            advanced_by_keyword[_string(item.get("RegistryKeyword"))] = item
    roam = advanced_by_keyword.get("RoamAggressiveness")
    if roam is not None and "1" not in _registry_values(roam.get("RegistryValue")):
        findings.append(
            Finding(
                code="roaming_aggressiveness_not_lowest",
                severity="info",
                message="Intel WLAN roaming aggressiveness is not at the lowest setting.",
                evidence=f"{roam.get('DisplayName')}: {roam.get('DisplayValue')}",
            )
        )
    mimo = advanced_by_keyword.get("MIMOPowerSaveMode")
    if mimo is not None and "3" not in _registry_values(mimo.get("RegistryValue")):
        findings.append(
            Finding(
                code="mimo_power_save_not_no_smps",
                severity="info",
                message="Intel WLAN MIMO power save is not forced to No SMPS.",
                evidence=f"{mimo.get('DisplayName')}: {mimo.get('DisplayValue')}",
            )
        )
    coalescing = advanced_by_keyword.get("*PacketCoalescing")
    if coalescing is not None and "1" in _registry_values(coalescing.get("RegistryValue")):
        findings.append(
            Finding(
                code="packet_coalescing_enabled",
                severity="info",
                message="Intel WLAN packet coalescing is enabled; disable only if disconnects persist after route/DNS cleanup.",
                evidence=f"{coalescing.get('DisplayName')}: {coalescing.get('DisplayValue')}",
            )
        )

    if not findings:
        findings.append(
            Finding(
                code="no_obvious_pc_side_wifi_risk",
                severity="ok",
                message="No obvious JetMax Wi-Fi route, DNS, event, or adapter risk was found in this snapshot.",
            )
        )
    return findings


def remediation_notes(*, interface_alias: str, robot_host: str, robot_subnet: str) -> list[str]:
    return [
        "Do not ping, SSH, port-scan, or pull camera video just to diagnose Wi-Fi drops; start with WLAN events.",
        (
            f"Preferred network shape: {interface_alias} keeps only the robot LAN route ({robot_subnet}); "
            "Ethernet or another adapter keeps Internet default route and public DNS."
        ),
        (
            "If Windows keeps recovering the JetMax AP as limited connectivity, use an elevated Windows network "
            "settings change to remove the JetMax default gateway/DNS from the WLAN profile, or set the adapter IPv4 "
            f"manually to a free {robot_subnet} address with no gateway and no DNS."
        ),
        (
            "If driver disconnects remain after route/DNS cleanup, try Intel adapter settings in this order: "
            "Roaming Aggressiveness=Lowest, MIMO Power Save=No SMPS, then Packet Coalescing=Disabled."
        ),
        f"Rollback principle: restore DHCP/default gateway/DNS on {interface_alias} if robot reachability gets worse.",
    ]


def build_report(
    snapshot: dict[str, Any],
    findings: list[Finding],
    *,
    interface_alias: str,
    ssid_prefix: str,
    robot_host: str,
    robot_subnet: str,
) -> dict[str, Any]:
    return {
        "tool": "diagnose_jetmax_wifi_windows",
        "platform": platform.platform(),
        "scope": "PC-side read-only diagnostics; no robot packets are sent",
        "interface_alias": interface_alias,
        "ssid_prefix": ssid_prefix,
        "robot_host": robot_host,
        "robot_subnet": robot_subnet,
        "findings": [
            {
                "code": finding.code,
                "severity": finding.severity,
                "message": finding.message,
                "evidence": finding.evidence,
            }
            for finding in findings
        ],
        "remediation_notes": remediation_notes(
            interface_alias=interface_alias,
            robot_host=robot_host,
            robot_subnet=robot_subnet,
        ),
        "raw": snapshot,
    }


def print_human(report: dict[str, Any]) -> None:
    print("JetMax Wi-Fi PC-side diagnostic")
    print(f"Scope: {report['scope']}")
    print(f"Interface: {report['interface_alias']}")
    print("")
    print("Findings:")
    for finding in report["findings"]:
        evidence = f" ({finding['evidence']})" if finding.get("evidence") else ""
        print(f"- [{finding['severity']}] {finding['code']}: {finding['message']}{evidence}")
    print("")
    print("Remediation notes:")
    for note in report["remediation_notes"]:
        print(f"- {note}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only Windows diagnostics for JetMax Wi-Fi disconnects.",
    )
    parser.add_argument("--interface-alias", default=DEFAULT_WLAN_ALIAS)
    parser.add_argument("--ssid-prefix", default=DEFAULT_SSID_PREFIX)
    parser.add_argument("--robot-host", default=DEFAULT_ROBOT_HOST)
    parser.add_argument("--robot-subnet", default=DEFAULT_ROBOT_SUBNET)
    parser.add_argument("--event-hours", type=float, default=12.0)
    parser.add_argument("--max-events", type=int, default=240)
    parser.add_argument("--json", action="store_true", help="Print the full JSON report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if platform.system().lower() != "windows":
        print("This diagnostic is Windows-only.", file=sys.stderr)
        return 2
    try:
        snapshot = collect_windows_snapshot(
            interface_alias=str(args.interface_alias),
            event_hours=float(args.event_hours),
            max_events=int(args.max_events),
        )
        findings = analyze_snapshot(
            snapshot,
            ssid_prefix=str(args.ssid_prefix),
            robot_host=str(args.robot_host),
            robot_subnet=str(args.robot_subnet),
        )
        report = build_report(
            snapshot,
            findings,
            interface_alias=str(args.interface_alias),
            ssid_prefix=str(args.ssid_prefix),
            robot_host=str(args.robot_host),
            robot_subnet=str(args.robot_subnet),
        )
    except Exception as error:
        print(f"diagnostic failed: {error}", file=sys.stderr)
        return 1
    if bool(args.json):
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
