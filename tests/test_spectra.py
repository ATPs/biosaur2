import pytest

from biosaur2.spectra import (
    extract_scan_number,
    group_spectra_by_faims,
    retention_time_seconds,
)


class UnitFloat(float):
    def __new__(cls, value, unit):
        instance = super().__new__(cls, value)
        instance.unit_info = unit
        return instance


def test_retention_time_converts_units_once():
    assert retention_time_seconds(UnitFloat(2.5, "minute")) == 150.0
    assert retention_time_seconds(UnitFloat(2.5, "second")) == 2.5
    assert retention_time_seconds(2.5, "minutes") == 150.0


def test_retention_time_rejects_unknown_units():
    with pytest.raises(ValueError, match="Unsupported"):
        retention_time_seconds(UnitFloat(2.5, "hour"))


def test_scan_number_is_parsed_or_null_without_index_fallback():
    assert extract_scan_number({"id": "controller=1 scan=42"}) == 42
    assert extract_scan_number({"id": "controller=1", "index": 41}) is None


def test_faims_grouping_distinguishes_null_and_zero_and_is_sorted():
    spectra = [
        {"name": "zero", "FAIMS compensation voltage": 0},
        {"name": "minus45", "FAIMS compensation voltage": -45},
        {"name": "missing"},
        {"name": "minus60", "FAIMS compensation voltage": -60},
    ]
    groups = group_spectra_by_faims(spectra)
    assert [value for value, _ in groups] == [None, -60.0, -45.0, 0.0]
    assert [items[0]["name"] for _, items in groups] == [
        "missing",
        "minus60",
        "minus45",
        "zero",
    ]
