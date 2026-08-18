"""Tests for search-pattern preprocessing."""

import csv
from pathlib import Path

from search_names import preprocess
from search_names.pipeline.step3_preprocess import load_drop_patterns


def test_preprocess_excludes_requested_patterns(tmp_path: Path) -> None:
    output = tmp_path / "preprocessed.csv"

    count = preprocess(
        "examples/preprocess/augmented_clean_names.csv",
        output_file=output,
        drop_patterns=["Barak Obama", "Michael Jackson"],
    )

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert count == len(rows)
    assert "barak obama" not in {row["search_name"] for row in rows}
    assert "michael jackson" not in {row["search_name"] for row in rows}


def test_load_drop_patterns_skips_blanks_and_comments(tmp_path: Path) -> None:
    patterns = tmp_path / "patterns.txt"
    patterns.write_text("# note\n\nJane Doe\nJOHN SMITH\n", encoding="utf-8")

    assert load_drop_patterns(patterns) == {"jane doe", "john smith"}
