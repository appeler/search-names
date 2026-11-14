"""Tests for normalized name-table generation."""

import csv
from pathlib import Path
from unittest.mock import patch

import pandas as pd
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


def test_clean_names_reproduces_the_example_output(tmp_path: Path) -> None:
    records = clean_names(
        "examples/clean_names/sample_input.csv", tmp_path / "clean_names.csv"
    )

    with Path("examples/clean_names/sample_output.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        assert records == list(csv.DictReader(stream))


@pytest.mark.parametrize(
    ("name", "first", "middle", "last", "suffix"),
    [
        # Given names that are also titles, credentials, or abbreviations.
        ("HUTCHINSON, ASA", "ASA", "", "HUTCHINSON", ""),
        ("Royal Samuel Copeland", "ROYAL", "SAMUEL", "Copeland", ""),
        ("King Swope", "KING", "", "Swope", ""),
        ("PRESTON, PRINCE HULON JR.", "PRINCE", "HULON", "PRESTON", "JR."),
        ("RODRIGUEZ, CIRO D.", "CIRO", "D.", "RODRIGUEZ", ""),
        ("SANFORD, JOHN W.A.", "JOHN", "W.A.", "SANFORD", ""),
        ("BROWN, EDMUND G JR", "EDMUND", "G.", "BROWN", "JR."),
    ],
)
def test_clean_names_reads_name_words_as_names(
    tmp_path: Path, name: str, first: str, middle: str, last: str, suffix: str
) -> None:
    input_file = tmp_path / "input.csv"
    with input_file.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows([["Name"], [name]])

    (record,) = clean_names(input_file, tmp_path / "output.csv")

    assert record["FirstName"] == first
    assert record["MiddleInitial/Name"] == middle
    assert record["LastName"] == last
    assert record["Suffix"] == suffix
    assert record["Title"] == ""


def _write_names(path: Path, names: list[str]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows([["Name"], *([name] for name in names)])
    return path


def test_clean_names_can_ask_parsernaam_for_the_name_order(tmp_path: Path) -> None:
    input_file = _write_names(tmp_path / "input.csv", ["Dr. Nichols Richard Jr."])
    prediction = {"name": "Nichols Richard", "type": "last_first", "prob": 0.99}

    by_rules = clean_names(input_file, tmp_path / "rules.csv")
    with patch(
        "search_names.enhanced_name_parser.parse_with_parsernaam_model",
        return_value=pd.DataFrame({"parsed_name": [prediction]}),
    ):
        by_model = clean_names(
            input_file, tmp_path / "model.csv", parser_type="parsernaam"
        )

    assert (by_rules[0]["FirstName"], by_rules[0]["LastName"]) == ("NICHOLS", "Richard")
    assert (by_model[0]["FirstName"], by_model[0]["LastName"]) == ("RICHARD", "Nichols")
    assert (by_model[0]["Title"], by_model[0]["Suffix"]) == ("DR.", "JR.")


def test_clean_names_rejects_an_unknown_parser(tmp_path: Path) -> None:
    input_file = _write_names(tmp_path / "input.csv", ["Jane Doe"])

    with pytest.raises(ValueError, match="parser_type"):
        clean_names(input_file, tmp_path / "output.csv", parser_type="humanname")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # Two generation words with no comma between them.
        ("John Smith Jr III", ("JOHN", "", "Smith", "III", "", "JR.")),
        ("John Smith III, MD", ("JOHN", "", "Smith", "III", "", "MD")),
        # The suffix written between the family and the given name.
        ("SMITH, JR., JOHN", ("JOHN", "", "SMITH", "", "", "JR.")),
        # A courtesy title or a generation word where the family name belongs.
        ("JOHN SMITH MR", ("JOHN", "", "SMITH", "", "MR", "")),
        ("JR., WILLIAM PASCRELL", ("WILLIAM", "", "PASCRELL", "", "", "JR.")),
        ("John Smith PhD", ("JOHN", "", "Smith", "", "", "PHD.")),
    ],
)
def test_clean_names_places_suffixes_and_titles(
    tmp_path: Path, name: str, expected: tuple[str, ...]
) -> None:
    input_file = _write_names(tmp_path / "input.csv", [name])

    (record,) = clean_names(input_file, tmp_path / "output.csv")

    assert (
        record["FirstName"],
        record["MiddleInitial/Name"],
        record["LastName"],
        record["RomanNumeral"],
        record["Title"],
        record["Suffix"],
    ) == expected
