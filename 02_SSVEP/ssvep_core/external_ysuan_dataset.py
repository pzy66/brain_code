from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET
from typing import Any, Optional, Sequence

import numpy as np
import scipy.io as sio
from scipy import signal

from .async_fbcca_idle_standalone import TrialSpec, json_dumps
from .dataset import save_collection_dataset_bundle


YSUAN_FIGSHARE_RECORD = "https://doi.org/10.6084/m9.figshare.24906300"
YSUAN_ARTICLE_DOI = "https://doi.org/10.1080/27706710.2024.2418650"
YSUAN_REQUIRED_CHANNELS = ("Oz", "O1", "O2", "PO3", "POz", "PO7", "PO8", "PO4")
YSUAN_FREQS = (8.0, 10.5, 12.0, 15.0)
YSUAN_TARGET_FREQUENCIES = (8.0, 9.0, 10.5, 11.0, 12.0, 13.0, 14.0, 15.0)
YSUAN_RAW_SAMPLING_RATE = 5000
YSUAN_SAMPLING_RATE = 250
YSUAN_CS_FOCUS_SEC = 4.0
YSUAN_CS_BREAK_SEC = 0.5
YSUAN_NS1_SEC = 4.0
YSUAN_NS2_SEC = 4.0
YSUAN_NS3_SEC = 2.0
YSUAN_CS_REPETITIONS = 12
YSUAN_NS1_TRIALS = 24
YSUAN_NS2_TRIALS = 24
YSUAN_NS3_TRIALS = 48
YSUAN_DEFAULT_NS_CALIBRATION_TRIALS_PER_SUBTYPE = 4


@dataclass(frozen=True)
class YSUANLoadedSubject:
    subject: str
    root_path: Path
    data_cs: np.ndarray
    data_ns1: np.ndarray
    data_ns2: np.ndarray
    data_ns3: np.ndarray
    channel_names: tuple[str, ...]
    selected_channel_names: tuple[str, ...]
    selected_channel_indices: tuple[int, ...]
    target_frequencies: tuple[float, ...]
    raw_sampling_rate: int
    sampling_rate: int


def _normalize_channel_name(value: str) -> str:
    text = str(value).strip().lower()
    return text.replace("z", "z")


def selected_channel_indices(
    channel_names: Sequence[str],
    *,
    required_channels: Sequence[str] = YSUAN_REQUIRED_CHANNELS,
) -> tuple[int, ...]:
    normalized_to_index = {
        _normalize_channel_name(name): int(index)
        for index, name in enumerate(tuple(str(item) for item in channel_names))
    }
    indices: list[int] = []
    missing: list[str] = []
    for name in tuple(required_channels):
        key = _normalize_channel_name(str(name))
        if key not in normalized_to_index:
            missing.append(str(name))
            continue
        indices.append(int(normalized_to_index[key]))
    if missing:
        raise ValueError(f"YSU-an subject is missing required channels: {missing}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"YSU-an required channels resolved to duplicate indices: {indices}")
    return tuple(indices)


def _default_channel_names_63() -> tuple[str, ...]:
    return tuple(f"Ch{index + 1}" for index in range(63))


def _looks_like_channel_name(value: Any) -> bool:
    text = str(value).strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered in {"channel", "channels", "label", "labels", "name", "names", "x", "y", "z"}:
        return False
    if len(text) > 12:
        return False
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return False
    return bool(re.fullmatch(r"[A-Za-z]{1,4}\d{0,2}", text))


def _rows_from_xlsx(path: Path) -> list[list[str]]:
    try:
        import openpyxl  # type: ignore

        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        sheet = workbook.active
        rows: list[list[str]] = []
        for row in sheet.iter_rows(values_only=True):
            rows.append(["" if item is None else str(item).strip() for item in row])
        workbook.close()
        return rows
    except Exception:
        pass

    rows: list[list[str]] = []
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.iter():
                if item.tag.endswith("}t") or item.tag == "t":
                    shared_strings.append(item.text or "")
        sheet_name = next(
            (name for name in archive.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")),
            "",
        )
        if not sheet_name:
            return rows
        root = ET.fromstring(archive.read(sheet_name))
        for row in root.iter():
            if not (row.tag.endswith("}row") or row.tag == "row"):
                continue
            values: list[str] = []
            for cell in list(row):
                if not (cell.tag.endswith("}c") or cell.tag == "c"):
                    continue
                cell_type = cell.attrib.get("t", "")
                raw_value = ""
                for child in list(cell):
                    if child.tag.endswith("}v") or child.tag == "v":
                        raw_value = child.text or ""
                        break
                if cell_type == "s":
                    try:
                        raw_value = shared_strings[int(raw_value)]
                    except Exception:
                        raw_value = ""
                values.append(str(raw_value).strip())
            if values:
                rows.append(values)
    return rows


def _rows_from_csv_or_text(path: Path) -> list[list[str]]:
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    if path.suffix.lower() == ".csv":
        return [[str(item).strip() for item in row] for row in csv.reader(text.splitlines())]
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(re.split(r"[\s,;]+", stripped))
    return rows


def parse_ysuan_channel_loc(path: Path) -> tuple[str, ...]:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() in {".xlsx", ".xlsm"}:
        rows = _rows_from_xlsx(resolved)
    else:
        rows = _rows_from_csv_or_text(resolved)
    if not rows:
        raise ValueError(f"YSU-an channel loc did not contain readable rows: {resolved}")

    max_cols = max(len(row) for row in rows)
    best_column: list[str] = []
    best_required_count = -1
    required = {_normalize_channel_name(name) for name in YSUAN_REQUIRED_CHANNELS}
    for column in range(max_cols):
        values = [
            str(row[column]).strip()
            for row in rows
            if column < len(row) and _looks_like_channel_name(row[column])
        ]
        normalized = {_normalize_channel_name(item) for item in values}
        required_count = len(required & normalized)
        if len(values) >= 63 and required_count > best_required_count:
            best_column = values
            best_required_count = required_count
    if best_column:
        names = tuple(str(item).strip() for item in best_column[:63])
    else:
        flattened = [
            str(item).strip()
            for row in rows
            for item in row
            if _looks_like_channel_name(item)
        ]
        if len(flattened) < 63:
            raise ValueError(f"YSU-an channel loc must provide at least 63 channel labels; got {len(flattened)}")
        names = tuple(flattened[:63])
    if len(names) != 63 or any(not name for name in names):
        raise ValueError(f"YSU-an channel loc must contain 63 nonempty channel names; got {len(names)}")
    return names


def _loadmat_any(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        return dict(sio.loadmat(resolved, squeeze_me=False, struct_as_record=False))
    except NotImplementedError:
        try:
            import h5py  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise ValueError(f"YSU-an v7.3 MATLAB file requires h5py: {resolved}") from exc
        payload: dict[str, Any] = {}
        with h5py.File(resolved, "r") as handle:
            for key in handle.keys():
                payload[key] = np.asarray(handle[key])
        return payload


def _discover_payload(subject_dir: Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(subject_dir).expanduser().resolve()
    if resolved.is_file():
        payload = _loadmat_any(resolved)
        return resolved, payload
    if not resolved.is_dir():
        raise ValueError(f"YSU-an subject path is not a file or directory: {resolved}")
    required = {"data_CS", "data_NS1", "data_NS2", "data_NS3"}
    merged: dict[str, Any] = {}
    first_path: Optional[Path] = None
    for path in sorted(resolved.rglob("*.mat")):
        payload = _loadmat_any(path)
        keys = set(payload)
        if required <= keys:
            return path, payload
        for key in required:
            if key in payload and key not in merged:
                merged[key] = payload[key]
                if first_path is None:
                    first_path = path
    if required <= set(merged):
        return first_path or resolved, merged
    missing = sorted(required - set(merged))
    raise ValueError(f"YSU-an subject is missing variables {missing}: {resolved}")


def _normalize_cs_shape(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 4:
        raise ValueError(f"YSU-an data_CS must be 4-D; got {array.shape}")
    shape = tuple(int(item) for item in array.shape)
    axes = list(range(4))
    try:
        freq_axis = next(axis for axis in axes if shape[axis] == len(YSUAN_TARGET_FREQUENCIES))
        channel_axis = next(axis for axis in axes if axis != freq_axis and shape[axis] == 63)
        rep_axis = next(axis for axis in axes if axis not in {freq_axis, channel_axis} and shape[axis] == YSUAN_CS_REPETITIONS)
        time_axis = next(axis for axis in axes if axis not in {freq_axis, channel_axis, rep_axis})
    except StopIteration as exc:
        raise ValueError(
            "YSU-an data_CS must contain axes for 8 frequencies, 63 channels, and 12 repetitions; "
            f"got {array.shape}"
        ) from exc
    return np.ascontiguousarray(np.transpose(array, (freq_axis, channel_axis, time_axis, rep_axis)), dtype=np.float64)


def _normalize_ns_shape(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"YSU-an {name} must be 3-D; got {array.shape}")
    shape = tuple(int(item) for item in array.shape)
    axes = list(range(3))
    try:
        channel_axis = next(axis for axis in axes if shape[axis] == 63)
        trial_axis = next(axis for axis in axes if axis != channel_axis and shape[axis] in {YSUAN_NS1_TRIALS, YSUAN_NS2_TRIALS, YSUAN_NS3_TRIALS})
        time_axis = next(axis for axis in axes if axis not in {channel_axis, trial_axis})
    except StopIteration as exc:
        raise ValueError(f"YSU-an {name} must contain axes for 63 channels and trial repetitions; got {array.shape}") from exc
    return np.ascontiguousarray(np.transpose(array, (channel_axis, time_axis, trial_axis)), dtype=np.float64)


def _find_channel_loc_path(subject_path: Path, channel_loc_path: Path | None) -> Optional[Path]:
    if channel_loc_path is not None:
        return Path(channel_loc_path).expanduser().resolve()
    root = Path(subject_path).expanduser().resolve()
    search_root = root if root.is_dir() else root.parent
    candidates = sorted(
        [
            *search_root.glob("*Channel*Loc*.xlsx"),
            *search_root.glob("*channel*loc*.xlsx"),
            *search_root.parent.glob("*Channel*Loc*.xlsx"),
            *search_root.parent.glob("*channel*loc*.xlsx"),
        ]
    )
    return candidates[0].resolve() if candidates else None


def _subject_label_from_path(path: Path) -> str:
    resolved = Path(path)
    candidates = [resolved.stem, resolved.name, resolved.parent.name]
    for item in candidates:
        match = re.search(r"S\d{2}", str(item).upper())
        if match:
            return match.group(0)
    return resolved.stem.upper() if resolved.is_file() else resolved.name.upper()


def load_ysuan_subject(
    subject_path: Path,
    channel_loc_path: Path | None = None,
    *,
    raw_sampling_rate: int = YSUAN_RAW_SAMPLING_RATE,
    sampling_rate: int = YSUAN_SAMPLING_RATE,
    allow_default_channel_names: bool = False,
) -> YSUANLoadedSubject:
    mat_path, payload = _discover_payload(subject_path)
    loc_path = _find_channel_loc_path(Path(subject_path), channel_loc_path)
    if loc_path is None:
        if not allow_default_channel_names:
            raise ValueError("YSU-an requires Channel Loc.xlsx for strict posterior channel selection")
        channel_names = _default_channel_names_63()
    else:
        channel_names = parse_ysuan_channel_loc(loc_path)
    indices = selected_channel_indices(channel_names)
    selected_names = tuple(channel_names[index] for index in indices)
    return YSUANLoadedSubject(
        subject=_subject_label_from_path(Path(subject_path)),
        root_path=Path(subject_path).expanduser().resolve(),
        data_cs=_normalize_cs_shape(payload["data_CS"]),
        data_ns1=_normalize_ns_shape(payload["data_NS1"], name="data_NS1"),
        data_ns2=_normalize_ns_shape(payload["data_NS2"], name="data_NS2"),
        data_ns3=_normalize_ns_shape(payload["data_NS3"], name="data_NS3"),
        channel_names=tuple(str(name) for name in channel_names),
        selected_channel_names=selected_names,
        selected_channel_indices=indices,
        target_frequencies=tuple(float(freq) for freq in YSUAN_TARGET_FREQUENCIES),
        raw_sampling_rate=int(raw_sampling_rate),
        sampling_rate=int(sampling_rate),
    )


def resolve_ysuan_command_frequencies(
    freqs: Sequence[float] = YSUAN_FREQS,
) -> tuple[tuple[float, ...], dict[float, int]]:
    available = {round(float(freq), 10): int(index) for index, freq in enumerate(YSUAN_TARGET_FREQUENCIES)}
    resolved: list[float] = []
    target_map: dict[float, int] = {}
    for value in tuple(freqs or ()):
        freq = round(float(value), 10)
        if freq not in available:
            raise ValueError(f"YSU-an CS frequency is unavailable: {float(value):g}Hz")
        if any(abs(freq - existing) < 1e-9 for existing in resolved):
            raise ValueError(f"duplicate YSU-an command frequency: {float(value):g}Hz")
        resolved.append(float(freq))
        target_map[float(freq)] = int(available[float(freq)])
    if len(resolved) != 4:
        raise ValueError(f"YSU-an command frequency set must contain exactly 4 frequencies; got {len(resolved)}")
    return tuple(resolved), target_map


def _preprocess_segment(
    matrix_channels_by_time: np.ndarray,
    *,
    raw_sampling_rate: int,
    sampling_rate: int,
) -> np.ndarray:
    matrix = np.asarray(matrix_channels_by_time, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"YSU-an segment must be channels x time; got {matrix.shape}")
    matrix = matrix - np.mean(matrix, axis=1, keepdims=True)
    if int(raw_sampling_rate) > 0 and matrix.shape[1] >= 32:
        nyquist = float(raw_sampling_rate) / 2.0
        if 50.0 < nyquist - 1e-6:
            b, a = signal.iirnotch(50.0, 30.0, float(raw_sampling_rate))
            matrix = signal.filtfilt(b, a, matrix, axis=1)
    if int(raw_sampling_rate) != int(sampling_rate):
        gcd = int(np.gcd(int(raw_sampling_rate), int(sampling_rate)))
        up = int(sampling_rate) // gcd
        down = int(raw_sampling_rate) // gcd
        matrix = signal.resample_poly(matrix, up=up, down=down, axis=1)
    matrix = matrix - np.mean(matrix, axis=1, keepdims=True)
    return np.ascontiguousarray(matrix.T, dtype=np.float64)


def _slice_raw_samples(
    values: np.ndarray,
    *,
    raw_sampling_rate: int,
    duration_sec: float,
    offset_sec: float = 0.0,
) -> np.ndarray:
    start = int(round(float(offset_sec) * int(raw_sampling_rate)))
    stop = start + int(round(float(duration_sec) * int(raw_sampling_rate)))
    if values.shape[1] < stop:
        raise ValueError(
            f"YSU-an raw segment is shorter than requested {duration_sec:g}s at {raw_sampling_rate}Hz: {values.shape}"
        )
    return values[:, start:stop]


def _cs_segment(subject: YSUANLoadedSubject, *, freq_index: int, repetition_index: int) -> np.ndarray:
    raw = subject.data_cs[
        int(freq_index),
        np.asarray(subject.selected_channel_indices, dtype=int),
        :,
        int(repetition_index),
    ]
    return _preprocess_segment(
        _slice_raw_samples(
            raw,
            raw_sampling_rate=int(subject.raw_sampling_rate),
            duration_sec=YSUAN_CS_FOCUS_SEC,
            offset_sec=0.0,
        ),
        raw_sampling_rate=int(subject.raw_sampling_rate),
        sampling_rate=int(subject.sampling_rate),
    )


def _ns_segment(subject: YSUANLoadedSubject, *, ns_mode: str, trial_index: int) -> np.ndarray:
    mode = str(ns_mode).strip().lower()
    if mode == "ns1":
        source = subject.data_ns1
        duration_sec = YSUAN_NS1_SEC
    elif mode == "ns2":
        source = subject.data_ns2
        duration_sec = YSUAN_NS2_SEC
    elif mode == "ns3":
        source = subject.data_ns3
        duration_sec = YSUAN_NS3_SEC
    else:
        raise ValueError(f"unsupported YSU-an NS mode: {ns_mode}")
    raw = source[np.asarray(subject.selected_channel_indices, dtype=int), :, int(trial_index)]
    return _preprocess_segment(
        _slice_raw_samples(
            raw,
            raw_sampling_rate=int(subject.raw_sampling_rate),
            duration_sec=float(duration_sec),
            offset_sec=0.0,
        ),
        raw_sampling_rate=int(subject.raw_sampling_rate),
        sampling_rate=int(subject.sampling_rate),
    )


def build_ysuan_cs_segments(
    subject: YSUANLoadedSubject,
    *,
    freqs: Sequence[float] = YSUAN_TARGET_FREQUENCIES,
) -> list[tuple[TrialSpec, np.ndarray]]:
    selected_freqs = tuple(float(freq) for freq in freqs)
    available = {round(float(freq), 10): int(index) for index, freq in enumerate(YSUAN_TARGET_FREQUENCIES)}
    segments: list[tuple[TrialSpec, np.ndarray]] = []
    trial_id = 0
    for repetition_index in range(int(subject.data_cs.shape[3])):
        for freq in selected_freqs:
            key = round(float(freq), 10)
            if key not in available:
                raise ValueError(f"YSU-an CS frequency is unavailable: {float(freq):g}Hz")
            segments.append(
                (
                    TrialSpec(
                        label=f"{float(freq):g}Hz",
                        expected_freq=float(freq),
                        trial_id=int(trial_id),
                        block_index=int(repetition_index),
                    ),
                    _cs_segment(subject, freq_index=int(available[key]), repetition_index=int(repetition_index)),
                )
            )
            trial_id += 1
    return segments


def build_ysuan_ns_segments(
    subject: YSUANLoadedSubject,
    *,
    ns_modes: Sequence[str] = ("ns1", "ns2", "ns3"),
    trial_id_start: int = 100000,
) -> list[tuple[TrialSpec, np.ndarray]]:
    segments: list[tuple[TrialSpec, np.ndarray]] = []
    trial_id = int(trial_id_start)
    for ns_mode in tuple(str(item).strip().lower() for item in ns_modes):
        source = {"ns1": subject.data_ns1, "ns2": subject.data_ns2, "ns3": subject.data_ns3}.get(ns_mode)
        if source is None:
            raise ValueError(f"unsupported YSU-an NS mode: {ns_mode}")
        for trial_index in range(int(source.shape[2])):
            segments.append(
                (
                    TrialSpec(
                        label=f"ysu_an_{ns_mode}_trial{int(trial_index) + 1:02d}",
                        expected_freq=None,
                        trial_id=int(trial_id),
                        block_index=int(trial_index),
                    ),
                    _ns_segment(subject, ns_mode=ns_mode, trial_index=int(trial_index)),
                )
            )
            trial_id += 1
    return segments


def build_ysuan_segments(
    subject: YSUANLoadedSubject,
    *,
    freqs: Sequence[float] = YSUAN_FREQS,
    include_ns_idle: bool = True,
) -> list[tuple[TrialSpec, np.ndarray]]:
    command_freqs, _target_map = resolve_ysuan_command_frequencies(freqs)
    segments = build_ysuan_cs_segments(subject, freqs=command_freqs)
    if bool(include_ns_idle):
        segments.extend(build_ysuan_ns_segments(subject))
    return segments


def ysuan_protocol_config(
    subject: YSUANLoadedSubject,
    *,
    freqs: Sequence[float] = YSUAN_FREQS,
    saved_trial_count: int,
    include_ns_idle: bool,
) -> dict[str, Any]:
    command_freqs, target_index_by_freq = resolve_ysuan_command_frequencies(freqs)
    ns_counts = {
        "ns1": int(subject.data_ns1.shape[2]) if include_ns_idle else 0,
        "ns2": int(subject.data_ns2.shape[2]) if include_ns_idle else 0,
        "ns3": int(subject.data_ns3.shape[2]) if include_ns_idle else 0,
    }
    return {
        "protocol_name": "external-ysu-an-fourfreq-async-v1",
        "source_dataset": "YSU-an asynchronous SSVEP-BCI EEG dataset",
        "source_record": YSUAN_FIGSHARE_RECORD,
        "source_article": YSUAN_ARTICLE_DOI,
        "subject_path": str(subject.root_path),
        "sampling_rate_original_hz": int(subject.raw_sampling_rate),
        "sampling_rate": int(subject.sampling_rate),
        "cs_focus_sec": float(YSUAN_CS_FOCUS_SEC),
        "cs_discarded_break_sec": float(YSUAN_CS_BREAK_SEC),
        "ns1_sec": float(YSUAN_NS1_SEC),
        "ns2_sec": float(YSUAN_NS2_SEC),
        "ns3_sec": float(YSUAN_NS3_SEC),
        "target_repeats": int(subject.data_cs.shape[3]),
        "planned_total_trials": int(saved_trial_count),
        "saved_trial_count": int(saved_trial_count),
        "control_trial_count": int(len(command_freqs) * int(subject.data_cs.shape[3])),
        "clean_idle_trial_count": int(sum(ns_counts.values())),
        "hard_idle_trial_count": 0,
        "no_control_subtype_trial_count": ns_counts,
        "include_explicit_ns_idle": bool(include_ns_idle),
        "freqs": [float(freq) for freq in command_freqs],
        "all_target_frequencies": [float(freq) for freq in YSUAN_TARGET_FREQUENCIES],
        "target_index_by_freq": {f"{float(freq):g}": int(index) for freq, index in target_index_by_freq.items()},
        "all_channel_count": int(len(subject.channel_names)),
        "all_channel_names": [str(name) for name in subject.channel_names],
        "selected_channel_policy": "strict_required_8_channels_only",
        "selected_channel_names": [str(name) for name in subject.selected_channel_names],
        "selected_channel_indices_zero_based": [int(index) for index in subject.selected_channel_indices],
        "selected_channel_indices_one_based": [int(index) + 1 for index in subject.selected_channel_indices],
        "excluded_channel_count": int(len(subject.channel_names) - len(subject.selected_channel_names)),
        "dataset_contains_required_channels": True,
        "only_required_channels_saved": True,
        "preprocessing": {
            "baseline_removal": "per segment channel-wise mean subtraction",
            "notch_hz": 50.0,
            "downsample_to_hz": int(subject.sampling_rate),
        },
        "idle_proxy_note": "YSU-an uses explicit NS1/NS2/NS3 no-control trials rather than non-command target proxy.",
    }


def convert_ysuan_subject_to_collection(
    *,
    subject_path: Path,
    dataset_root: Path,
    channel_loc_path: Path | None = None,
    session_id: Optional[str] = None,
    subject_id: Optional[str] = None,
    freqs: Sequence[float] = YSUAN_FREQS,
    include_ns_idle: bool = True,
) -> dict[str, Any]:
    subject = load_ysuan_subject(subject_path, channel_loc_path=channel_loc_path)
    command_freqs, target_index_by_freq = resolve_ysuan_command_frequencies(freqs)
    segments = build_ysuan_segments(subject, freqs=command_freqs, include_ns_idle=bool(include_ns_idle))
    if not segments:
        raise RuntimeError(f"no YSU-an segments built for {subject_path}")
    if any(segment.shape[1] != len(YSUAN_REQUIRED_CHANNELS) for _trial, segment in segments):
        raise RuntimeError("YSU-an conversion produced a segment that is not exactly 8 channels")
    protocol = ysuan_protocol_config(
        subject,
        freqs=command_freqs,
        saved_trial_count=len(segments),
        include_ns_idle=bool(include_ns_idle),
    )
    resolved_subject = subject_id or f"ysu_an_{subject.subject.lower()}"
    freq_token = "_".join(f"{float(freq):g}".replace(".", "p") for freq in command_freqs)
    resolved_session = session_id or f"{resolved_subject}_{freq_token}_strict8_async"
    payload = save_collection_dataset_bundle(
        dataset_root=Path(dataset_root),
        session_id=resolved_session,
        subject_id=resolved_subject,
        serial_port="external_ysu_an",
        board_id=-1,
        sampling_rate=int(subject.sampling_rate),
        freqs=command_freqs,
        board_eeg_channels=tuple(range(len(YSUAN_REQUIRED_CHANNELS))),
        protocol_config=protocol,
        trial_segments=segments,
    )
    manifest_path = Path(str(payload["dataset_manifest"])).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["external_dataset_validation"] = {
        "dataset": "YSU-an",
        "required_channel_names": [str(name) for name in YSUAN_REQUIRED_CHANNELS],
        "selected_channel_names": [str(name) for name in subject.selected_channel_names],
        "selected_channel_indices_zero_based": [int(index) for index in subject.selected_channel_indices],
        "source_shapes": {
            "data_CS": [int(item) for item in subject.data_cs.shape],
            "data_NS1": [int(item) for item in subject.data_ns1.shape],
            "data_NS2": [int(item) for item in subject.data_ns2.shape],
            "data_NS3": [int(item) for item in subject.data_ns3.shape],
        },
        "used_shape_per_cs_trial": [int(YSUAN_CS_FOCUS_SEC * subject.sampling_rate), 8],
        "used_shape_per_ns3_trial": [int(YSUAN_NS3_SEC * subject.sampling_rate), 8],
        "only_required_channels_saved": True,
        "excluded_channel_count": int(len(subject.channel_names) - len(subject.selected_channel_names)),
        "include_explicit_ns_idle": bool(include_ns_idle),
        "freqs": [float(freq) for freq in command_freqs],
        "all_target_frequencies": [float(freq) for freq in YSUAN_TARGET_FREQUENCIES],
        "target_index_by_freq": {f"{float(freq):g}": int(index) for freq, index in target_index_by_freq.items()},
    }
    manifest_path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
    return {
        **payload,
        "subject": subject.subject,
        "selected_channel_names": list(subject.selected_channel_names),
        "selected_channel_indices_zero_based": list(subject.selected_channel_indices),
        "segment_count": int(len(segments)),
    }
