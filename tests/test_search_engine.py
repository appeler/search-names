"""Tests for the unified exact/fuzzy corpus search."""

import csv
import gzip
from pathlib import Path

import pytest

from search_names.engines import SearchEngine
from search_names.pipeline.step4_search import load_names_file, search_names


def _write_corpus(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "text"])
        writer.writeheader()
        writer.writerows(
            [
                {"id": "1", "text": "John Smith spoke."},
                {"id": "2", "text": "Nothing matched."},
                {"id": "3", "text": "Jane Doe met Jon Smyth."},
            ]
        )


def test_exact_search_preserves_match_text_and_offsets() -> None:
    engine = SearchEngine([("person-1", "john smith")])

    matches = engine.search("Before John Smith after")

    assert matches == [
        {
            "uniqid": "person-1",
            "n": 1,
            "match": "John Smith",
            "start": 7,
            "end": 17,
        }
    ]


def test_exact_search_matches_names_ending_in_punctuation() -> None:
    engine = SearchEngine([("person-1", "John Smith Jr.")])

    matches = engine.search("Before John Smith Jr. after")

    assert matches[0]["match"] == "John Smith Jr."


def test_fuzzy_rule_finds_typo_at_minimum_length_boundary() -> None:
    engine = SearchEngine([("person-1", "john smith")], [(10, 1)])

    matches = engine.search("Jon Smith spoke.")

    assert matches[0]["uniqid"] == "person-1"
    assert matches[0]["match"] == "Jon Smith"


@pytest.mark.parametrize(
    ("names", "rules", "message"),
    [
        ([("1", "")], [], "search names cannot be blank"),
        ([("1", "Jane Doe")], [(0, 1)], "minimum length"),
        ([("1", "Jane Doe")], [(4, -1)], "edit distance"),
    ],
)
def test_engine_rejects_invalid_inputs(names, rules, message) -> None:
    with pytest.raises(ValueError, match=message):
        SearchEngine(names, rules)


def test_search_writes_every_row_in_input_order(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    output = tmp_path / "results.csv"
    _write_corpus(corpus)

    stats = search_names(
        corpus,
        [("john", "John Smith"), ("jane", "Jane Doe")],
        output,
        input_columns=("id", "text"),
        max_results=2,
        processes=1,
        chunk_size=1,
    )

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert stats == {"total_rows": 3, "total_matches": 2}
    assert [row["id"] for row in rows] == ["1", "2", "3"]
    assert [row["count"] for row in rows] == ["1", "0", "1"]
    assert rows[0]["name1.uniqid"] == "john"
    assert rows[2]["name1.uniqid"] == "jane"


def test_parallel_and_single_process_outputs_match(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    single_output = tmp_path / "single.csv"
    parallel_output = tmp_path / "parallel.csv"
    _write_corpus(corpus)
    kwargs = {
        "input_columns": ("id", "text"),
        "max_results": 2,
        "chunk_size": 1,
    }
    names = [("john", "John Smith"), ("jane", "Jane Doe")]

    search_names(corpus, names, single_output, processes=1, **kwargs)
    search_names(corpus, names, parallel_output, processes=2, **kwargs)

    assert parallel_output.read_bytes() == single_output.read_bytes()


def test_load_names_file_supports_gzip_and_validates_schema(tmp_path: Path) -> None:
    name_file = tmp_path / "names.csv.gz"
    with gzip.open(name_file, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["uniqid", "search_name"])
        writer.writeheader()
        writer.writerow({"uniqid": "7", "search_name": "Jane Doe"})

    assert load_names_file(name_file) == [("7", "Jane Doe")]

    invalid_file = tmp_path / "invalid.csv"
    invalid_file.write_text("name\nJane Doe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_names_file(invalid_file)


def test_search_validates_before_touching_output(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    output = tmp_path / "results.csv"
    corpus.write_text("text\nJane Doe\n", encoding="utf-8")
    output.write_text("keep me\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing columns"):
        search_names(corpus, [("1", "Jane Doe")], output, processes=1)

    assert output.read_text(encoding="utf-8") == "keep me\n"


def test_search_rejects_overwriting_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    original = "uniqid,text\n1,Jane Doe\n"
    corpus.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="cannot also be the corpus"):
        search_names(corpus, [("1", "Jane Doe")], corpus, processes=1)

    assert corpus.read_text(encoding="utf-8") == original


def test_load_names_file_rejects_blank_values(tmp_path: Path) -> None:
    name_file = tmp_path / "names.csv"
    name_file.write_text("uniqid,search_name\n1,\n", encoding="utf-8")

    with pytest.raises(ValueError, match="search names cannot be blank"):
        load_names_file(name_file)


def test_search_supports_document_sized_csv_fields(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    output = tmp_path / "results.csv"
    long_text = f"{'x' * 140_000} Jane Doe"
    with corpus.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["uniqid", "text"])
        writer.writeheader()
        writer.writerow({"uniqid": "1", "text": long_text})

    stats = search_names(corpus, [("7", "Jane Doe")], output, processes=1)

    assert stats == {"total_rows": 1, "total_matches": 1}


def test_search_failure_does_not_replace_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus.csv"
    output = tmp_path / "results.csv"
    corpus.write_text("uniqid,text\n1,Jane Doe\n", encoding="utf-8")
    output.write_text("keep me\n", encoding="utf-8")

    def fail_search(*_args, **_kwargs):
        raise RuntimeError("worker failed")

    monkeypatch.setattr(SearchEngine, "search", fail_search)
    with pytest.raises(RuntimeError, match="worker failed"):
        search_names(corpus, [("7", "Jane Doe")], output, processes=1)

    assert output.read_text(encoding="utf-8") == "keep me\n"
