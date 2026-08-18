"""Tests for safe split and merge file operations."""

from pathlib import Path

import pytest

from search_names import merge_results, split_text_corpus


def test_merge_preflights_every_schema_before_writing(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    output = tmp_path / "output.csv"
    first.write_text("id,value\n1,a\n", encoding="utf-8")
    second.write_text("id,other\n2,b\n", encoding="utf-8")
    output.write_text("keep me\n", encoding="utf-8")

    with pytest.raises(ValueError, match="schema mismatch"):
        merge_results([first, second], output)

    assert output.read_text(encoding="utf-8") == "keep me\n"


def test_merge_rejects_duplicate_columns(tmp_path: Path) -> None:
    input_file = tmp_path / "input.csv"
    input_file.write_text("id,id\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not unique"):
        merge_results([input_file], tmp_path / "output.csv")


def test_merge_supports_document_sized_csv_fields(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    output = tmp_path / "output.csv"
    long_text = "x" * 140_000
    first.write_text(f"id,text\n1,{long_text}\n", encoding="utf-8")
    second.write_text("id,text\n2,short\n", encoding="utf-8")

    assert merge_results([first, second], output) == 2
    assert long_text in output.read_text(encoding="utf-8")


def test_split_cannot_overwrite_its_input(tmp_path: Path) -> None:
    input_file = tmp_path / "input.csv"
    original = "text\none\n"
    input_file.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="cannot overwrite"):
        split_text_corpus(input_file, str(input_file))

    assert input_file.read_text(encoding="utf-8") == original


def test_split_rejects_headerless_input(tmp_path: Path) -> None:
    input_file = tmp_path / "input.csv"
    input_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="must have a header"):
        split_text_corpus(input_file, str(tmp_path / "chunk_{chunk_id}.csv"))


def test_split_supports_document_sized_csv_fields(tmp_path: Path) -> None:
    input_file = tmp_path / "input.csv"
    output_pattern = str(tmp_path / "chunk_{chunk_id}.csv")
    long_text = "x" * 140_000
    input_file.write_text(f"text\n{long_text}\n", encoding="utf-8")

    assert split_text_corpus(input_file, output_pattern) == 1
    assert long_text in (tmp_path / "chunk_1.csv").read_text(encoding="utf-8")
