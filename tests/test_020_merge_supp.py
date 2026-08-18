"""Tests for deterministic prefix and nickname augmentation."""

import csv
from pathlib import Path

from search_names import augment_names


def test_augment_names_loads_prefixes_and_nicknames(tmp_path: Path) -> None:
    output = tmp_path / "augmented.csv"

    count = augment_names(
        "examples/merge_supp_data/sample_in.csv",
        output_file=output,
        prefix_file="examples/merge_supp_data/prefixes.csv",
        nickname_file="examples/merge_supp_data/nick_names.txt",
    )

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert count == len(rows)
    assert any("president" in row["prefixes"].casefold() for row in rows)
    robert = next(row for row in rows if row["FirstName"] == "ROBERT")
    assert robert["nick_names"] == "bob;bobby;rob"


def test_augment_names_replaces_colliding_output_columns(tmp_path: Path) -> None:
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    input_file.write_text(
        "seat,FirstName,prefixes,nick_names\nA,William,stale,stale\n",
        encoding="utf-8",
    )

    augment_names(input_file, output_file=output_file)

    with output_file.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["prefixes"] == ""
    assert row["nick_names"] == ""
