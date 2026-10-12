"""Choosing which scans to load: the text field's parser and the loader."""

from pathlib import Path

import pytest

from hdsfg_engine import Settings, retrieve
from hdsfg_engine.settings import format_scan_list, parse_scan_list

DATA = Path(__file__).parent.parent.parent / "20261005_50kHz_pinhole" / "data"


@pytest.mark.parametrize("text, numbers", [
    ("", []), ("all", []), ("3", [3]), ("1-4", [1, 2, 3, 4]), ("4-1", [1, 2, 3, 4]),
    ("1-3, 7, 9-10", [1, 2, 3, 7, 9, 10]), ("1–3;5", [1, 2, 3, 5]), ("2, 2, 1-2", [1, 2]),
])
def test_parse(text, numbers):
    assert parse_scan_list(text) == numbers


@pytest.mark.parametrize("text", ["a", "1-", "1-2-3", "-5", "1.5"])
def test_parse_rejects(text):
    with pytest.raises(ValueError):
        parse_scan_list(text)


def test_format_round_trip():
    assert format_scan_list([1, 2, 3, 5, 7, 8]) == "1-3, 5, 7-8"
    assert format_scan_list([]) == ""
    assert parse_scan_list(format_scan_list([1, 2, 3, 5, 7, 8])) == [1, 2, 3, 5, 7, 8]


needs_data = pytest.mark.skipif(not DATA.is_dir(), reason="needs the lab's 2026-10-05 data")


@needs_data
def test_loads_only_chosen_scans():
    result = retrieve(Settings(data_dir=str(DATA), reference_name="zqz2", sample_name="h2o2",
                               reference_file_numbers=[1, 2, 3], sample_file_numbers=[5, 7, 99]))
    assert sorted(result.reference.numbers.tolist()) == [1, 2, 3]
    assert sorted(result.sample.numbers.tolist()) == [5, 7]          # 99 does not exist


@needs_data
def test_no_chosen_scan_exists():
    with pytest.raises(FileNotFoundError, match="None of the chosen scan numbers"):
        retrieve(Settings(data_dir=str(DATA), reference_name="zqz2", sample_name="h2o2",
                          sample_file_numbers=[500]))
