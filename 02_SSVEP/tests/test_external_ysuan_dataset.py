from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import scipy.io as sio

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core import external_ysuan_dataset as ysuan


def _write_channel_loc(path: Path) -> None:
    names = [f"Ch{index + 1}" for index in range(63)]
    names[52] = "PO7"
    names[54] = "PO3"
    names[55] = "POz"
    names[56] = "PO4"
    names[58] = "PO8"
    names[60] = "O1"
    names[61] = "Oz"
    names[62] = "O2"
    rows = ["index,name"]
    rows.extend(f"{index + 1},{name}" for index, name in enumerate(names))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_subject_mat(path: Path, *, raw_fs: int = 250) -> None:
    cs_samples = int(round((ysuan.YSUAN_CS_FOCUS_SEC + ysuan.YSUAN_CS_BREAK_SEC) * raw_fs))
    ns12_samples = int(round(ysuan.YSUAN_NS1_SEC * raw_fs))
    ns3_samples = int(round(ysuan.YSUAN_NS3_SEC * raw_fs))
    data_cs = np.zeros((8, 63, cs_samples, 12), dtype=np.float64)
    data_ns1 = np.zeros((63, ns12_samples, 24), dtype=np.float64)
    data_ns2 = np.zeros((63, ns12_samples, 24), dtype=np.float64)
    data_ns3 = np.zeros((63, ns3_samples, 48), dtype=np.float64)
    sio.savemat(
        path,
        {
            "data_CS": data_cs,
            "data_NS1": data_ns1,
            "data_NS2": data_ns2,
            "data_NS3": data_ns3,
        },
    )


def test_load_ysuan_subject_and_build_segments_from_synthetic_mat(tmp_path: Path) -> None:
    subject_dir = tmp_path / "S01"
    subject_dir.mkdir()
    mat_path = subject_dir / "S01.mat"
    loc_path = tmp_path / "Channel Loc.csv"
    _write_subject_mat(mat_path)
    _write_channel_loc(loc_path)

    subject = ysuan.load_ysuan_subject(
        subject_dir,
        channel_loc_path=loc_path,
        raw_sampling_rate=250,
        sampling_rate=250,
    )
    segments = ysuan.build_ysuan_segments(subject, freqs=(8.0, 10.5, 12.0, 15.0), include_ns_idle=True)

    control = [(trial, segment) for trial, segment in segments if trial.expected_freq is not None]
    idle = [(trial, segment) for trial, segment in segments if trial.expected_freq is None]
    ns3 = [(trial, segment) for trial, segment in idle if "ns3" in trial.label]
    assert subject.selected_channel_indices == (61, 60, 62, 54, 55, 52, 58, 56)
    assert len(control) == 4 * 12
    assert len(idle) == 24 + 24 + 48
    assert all(segment.shape == (1000, 8) for _trial, segment in control)
    assert all(segment.shape == (500, 8) for _trial, segment in ns3)
    assert {trial.expected_freq for trial, _segment in control} == {8.0, 10.5, 12.0, 15.0}
    assert {trial.block_index for trial, _segment in control} == set(range(12))


def test_build_ysuan_cs_segments_can_expose_all_eight_targets(tmp_path: Path) -> None:
    subject_dir = tmp_path / "S02"
    subject_dir.mkdir()
    _write_subject_mat(subject_dir / "session.mat")
    loc_path = tmp_path / "Channel Loc.csv"
    _write_channel_loc(loc_path)
    subject = ysuan.load_ysuan_subject(
        subject_dir,
        channel_loc_path=loc_path,
        raw_sampling_rate=250,
        sampling_rate=250,
    )

    segments = ysuan.build_ysuan_cs_segments(subject, freqs=ysuan.YSUAN_TARGET_FREQUENCIES)

    assert len(segments) == 8 * 12
    assert {trial.expected_freq for trial, _segment in segments} == set(ysuan.YSUAN_TARGET_FREQUENCIES)
