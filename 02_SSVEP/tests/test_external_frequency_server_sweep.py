from __future__ import annotations

from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tools import run_external_frequency_server_sweep as sweep


def test_parse_frequency_sets_requires_exactly_four_values() -> None:
    parsed = sweep._parse_frequency_sets("9.8,12,14.8,15.8;11,12,14.6,15.8")
    assert parsed == [
        (9.8, 12.0, 14.8, 15.8),
        (11.0, 12.0, 14.6, 15.8),
    ]


def test_parse_frequency_sets_deduplicates_by_rounded_key() -> None:
    parsed = sweep._parse_frequency_sets("9.8,12,14.8,15.8;9.8000001,12,14.8,15.8")
    assert parsed == [(9.8, 12.0, 14.8, 15.8)]


def test_formal_candidate_freqs_uses_explicit_sets_only_when_selection_skipped() -> None:
    explicit = [(9.8, 12.0, 14.8, 15.8), (11.0, 12.0, 14.6, 15.8)]
    actual = sweep._formal_candidate_freqs(
        {},
        max_top=3,
        explicit_sets=explicit,
        skip_selection=True,
    )
    assert actual == explicit


def test_formal_candidate_freqs_merges_explicit_and_selection_sets() -> None:
    selection = {
        "best": {"freqs": [11.0, 12.0, 14.6, 15.8]},
        "top_combinations": [
            {"freqs": [9.8, 12.0, 14.8, 15.8]},
            {"freqs": [11.0, 12.0, 14.8, 15.8]},
        ],
    }
    actual = sweep._formal_candidate_freqs(
        selection,
        max_top=2,
        explicit_sets=[(9.8, 12.0, 14.8, 15.8)],
        skip_selection=False,
    )
    assert (9.8, 12.0, 14.8, 15.8) in actual
    assert (11.0, 12.0, 14.6, 15.8) in actual
    assert (11.0, 12.0, 14.8, 15.8) in actual


def test_csv_candidate_parsers_use_defaults_and_values() -> None:
    assert sweep._csv_float_tuple("", default=(1.5, 2.0)) == (1.5, 2.0)
    assert sweep._csv_float_tuple("2.0,2.5,3.0", default=(1.5,)) == (2.0, 2.5, 3.0)
    assert sweep._csv_int_tuple("", default=(1, 2)) == (1, 2)
    assert sweep._csv_int_tuple("1,3", default=(2,)) == (1, 3)


def test_dataset_parser_defaults_and_deduplicates() -> None:
    assert sweep._csv_dataset_tuple("") == ("wang2016", "beta")
    assert sweep._csv_dataset_tuple("beta") == ("beta",)
    assert sweep._csv_dataset_tuple("wang2016,beta,wang2016") == ("wang2016", "beta")
