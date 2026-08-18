"""Tests for normalized name-table generation."""

import csv
from pathlib import Path

import pytest

from search_names import clean_names


def test_clean_names_writes_expected_schema(tmp_path: Path) -> None:
    output = tmp_path / "clean_names.csv"

    records = clean_names("examples/clean_names/sample_input.csv", output)

    assert output.exists()
    assert records
    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == records
    assert rows[0]["uniqid"] == "1"
    assert rows[0]["FirstName"]


def test_clean_names_rejects_missing_name_column(tmp_path: Path) -> None:
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    input_file.write_text("identifier\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing name column"):
        clean_names(input_file, output_file, name_column="name")
