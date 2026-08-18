"""Integration tests for corpus search."""

import csv
from pathlib import Path

from search_names import search_names
from search_names.pipeline.step4_search import load_names_file


def test_fixture_search_emits_one_row_per_input(tmp_path: Path) -> None:
    output = tmp_path / "search_results.csv"
    names = load_names_file("examples/preprocess/deduped_augmented_clean_names.csv")

    stats = search_names(
        "examples/search/text_corpus.csv",
        names=names,
        output_file=output,
        processes=1,
    )

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert output.exists()
    assert stats["total_rows"] > 0
    assert len(rows) == stats["total_rows"]
    assert sum(int(row["count"]) for row in rows) == stats["total_matches"]
