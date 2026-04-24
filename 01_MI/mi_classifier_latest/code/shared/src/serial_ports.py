"""Serial-port discovery helpers shared across collection and realtime tools."""

from __future__ import annotations

import csv
import io
import os
import re
import subprocess


_COM_PORT_PATTERN = re.compile(r"\bCOM\d+\b", flags=re.IGNORECASE)
_WINDOWS_UNAVAILABLE_TOKENS = {
    "disconnected",
    "error",
    "not present",
    "notpresent",
    "problem",
    "unknown",
}


def _clean_port_name(port: str) -> str:
    return str(port).strip()


def _display_port_name(port: str) -> str:
    cleaned = _clean_port_name(port)
    if _COM_PORT_PATTERN.fullmatch(cleaned):
        return cleaned.upper()
    return cleaned


def _normalize_port_name(port: str) -> str:
    cleaned = _clean_port_name(port)
    if os.name == "nt" or _COM_PORT_PATTERN.fullmatch(cleaned):
        return cleaned.upper()
    return cleaned


def _normalize_status_token(value: str) -> str:
    return re.sub(r"[\s_-]+", " ", str(value).strip().lower())


def _dedupe_keep_order(items: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        display_name = _display_port_name(item)
        normalized = _normalize_port_name(display_name)
        if not display_name or normalized in seen:
            continue
        ordered.append(display_name)
        seen.add(normalized)
    return ordered


def _detect_pyserial_ports() -> list[str]:
    try:
        from serial.tools import list_ports

        devices = [_display_port_name(port.device) for port in list_ports.comports() if str(port.device).strip()]
        return _dedupe_keep_order(sorted(devices))
    except Exception:
        return []


def _match_port_name(*candidates: str) -> str:
    for candidate in candidates:
        port_match = _COM_PORT_PATTERN.search(str(candidate))
        if port_match is not None:
            return _normalize_port_name(port_match.group(0))
    return ""


def _parse_windows_pnp_ports_csv(raw_output: str) -> list[dict[str, str]]:
    text = str(raw_output).strip()
    if not text:
        return []

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return []

    ports: list[dict[str, str]] = []
    for row in reader:
        normalized_row = {
            str(key).strip().lstrip("\ufeff"): str(value or "").strip()
            for key, value in row.items()
            if key is not None
        }
        port_name = _match_port_name(
            normalized_row.get("DeviceDescription", ""),
            normalized_row.get("InstanceId", ""),
            " ".join(normalized_row.values()),
        )
        if not port_name:
            continue
        ports.append(
            {
                "port": port_name,
                "status": normalized_row.get("Status", ""),
                "problem_code": normalized_row.get("ProblemCode", ""),
                "problem_status": normalized_row.get("ProblemStatus", ""),
                "description": normalized_row.get("DeviceDescription", ""),
            }
        )
    return ports


def _parse_windows_pnp_ports_text(raw_output: str) -> list[dict[str, str]]:
    ports: list[dict[str, str]] = []
    for block in re.split(r"(?:\r?\n){2,}", str(raw_output)):
        port_name = _match_port_name(block)
        if not port_name:
            continue
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[_normalize_status_token(key)] = value.strip()
        ports.append(
            {
                "port": port_name,
                "status": fields.get("status", ""),
                "problem_code": fields.get("problem code", ""),
                "problem_status": fields.get("problem status", ""),
                "description": fields.get("device description", ""),
            }
        )
    return ports


def _parse_windows_pnp_port_records(raw_output: str) -> list[dict[str, str]]:
    text = str(raw_output).strip()
    if not text:
        return []
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if first_line.lower().startswith("instanceid,"):
        return _parse_windows_pnp_ports_csv(text)
    return _parse_windows_pnp_ports_text(text)


def _parse_windows_pnp_ports(raw_output: str) -> list[tuple[str, str]]:
    return [
        (str(record.get("port", "")).strip().upper(), str(record.get("status", "")).strip())
        for record in _parse_windows_pnp_port_records(raw_output)
    ]


def _run_windows_pnp_ports_command(command: list[str]) -> list[dict[str, str]]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    return _parse_windows_pnp_port_records(completed.stdout.decode("utf-8", errors="replace"))


def _detect_windows_pnp_ports() -> list[dict[str, str]]:
    if os.name != "nt":
        return []
    ports = _run_windows_pnp_ports_command(["pnputil", "/enum-devices", "/class", "Ports", "/format", "csv"])
    if ports:
        return ports
    return _run_windows_pnp_ports_command(["pnputil", "/enum-devices", "/class", "Ports"])


def _is_windows_port_unavailable(record: dict[str, str]) -> bool:
    status = _normalize_status_token(record.get("status", ""))
    problem_code = str(record.get("problem_code", "")).strip()
    problem_status = _normalize_status_token(record.get("problem_status", ""))
    if problem_code:
        return True
    return any(token in status or token in problem_status for token in _WINDOWS_UNAVAILABLE_TOKENS)


def _find_windows_port_record(windows_ports: list[dict[str, str]], port: str) -> dict[str, str] | None:
    normalized = _normalize_port_name(port)
    for record in windows_ports:
        if _normalize_port_name(record.get("port", "")) == normalized:
            return dict(record)
    return None


def get_serial_port_inventory() -> dict[str, object]:
    """Return a normalized snapshot of available serial-port inventories."""
    pyserial_ports = _detect_pyserial_ports()
    windows_ports = _detect_windows_pnp_ports() if os.name == "nt" else []

    merged: list[str] = []
    unavailable_ports = {
        _normalize_port_name(record.get("port", ""))
        for record in windows_ports
        if _is_windows_port_unavailable(record)
    }

    for port in pyserial_ports:
        normalized = _normalize_port_name(port)
        if normalized and normalized not in unavailable_ports:
            merged.append(normalized)

    for record in windows_ports:
        normalized = _normalize_port_name(record.get("port", ""))
        if normalized and normalized not in unavailable_ports:
            merged.append(normalized)

    return {
        "detected_ports": _dedupe_keep_order(merged),
        "pyserial_ports": list(pyserial_ports),
        "windows_ports": [dict(record) for record in windows_ports],
    }


def _build_serial_port_validation(
    port: str,
    *,
    inventory: dict[str, object],
    allow_manual_override: bool,
) -> dict[str, object]:
    normalized = _normalize_port_name(port)
    detected_ports = [
        _normalize_port_name(item)
        for item in inventory.get("detected_ports", [])
        if _normalize_port_name(item)
    ]
    windows_ports = [
        dict(record)
        for record in inventory.get("windows_ports", [])
        if isinstance(record, dict)
    ]
    windows_record = _find_windows_port_record(windows_ports, normalized)

    result = {
        "ok": False,
        "reason": "missing_port",
        "requested_port": normalized,
        "detected_ports": detected_ports,
        "windows_status": str((windows_record or {}).get("status", "")).strip(),
        "problem_code": str((windows_record or {}).get("problem_code", "")).strip(),
        "problem_status": str((windows_record or {}).get("problem_status", "")).strip(),
        "windows_description": str((windows_record or {}).get("description", "")).strip(),
    }

    if not normalized:
        return result
    if normalized in detected_ports:
        result["ok"] = True
        result["reason"] = "detected"
        return result
    if windows_record is not None and _is_windows_port_unavailable(windows_record):
        result["reason"] = "windows_unavailable"
        return result
    if allow_manual_override and not detected_ports:
        result["ok"] = True
        result["reason"] = "manual_override"
        return result
    if windows_record is not None:
        result["ok"] = True
        result["reason"] = "windows_inventory"
        return result
    result["reason"] = "not_detected"
    return result


def detect_serial_ports() -> list[str]:
    """Detect high-confidence serial ports without synthesizing placeholder COM names."""
    inventory = get_serial_port_inventory()
    return [
        _display_port_name(item)
        for item in inventory.get("detected_ports", [])
        if _normalize_port_name(item)
    ]


def validate_serial_port_selection(port: str, *, allow_manual_override: bool = True) -> dict[str, object]:
    """Validate a selected serial port against the current machine inventory."""
    inventory = get_serial_port_inventory()
    return _build_serial_port_validation(
        port,
        inventory=inventory,
        allow_manual_override=allow_manual_override,
    )


def describe_serial_port(port: str) -> dict[str, object]:
    """Return a lightweight diagnostic snapshot for one serial port."""
    inventory = get_serial_port_inventory()
    return _build_serial_port_validation(
        port,
        inventory=inventory,
        allow_manual_override=True,
    )
