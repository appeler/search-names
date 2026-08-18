"""End-to-end tests for the supported CLI commands."""

import csv
from pathlib import Path

from typer.testing import CliRunner

from search_names import __version__
from search_names.cli import app

runner = CliRunner()


def test_help_version_and_invalid_invocations() -> None:
    help_result = runner.invoke(app, ["--help"])
    version_result = runner.invoke(app, ["--version"])

    assert help_result.exit_code == 0
    assert "search-names" in help_result.stdout
    assert version_result.exit_code == 0
    assert __version__ in version_result.stdout
    assert runner.invoke(app, ["invalid-command"]).exit_code != 0
    assert runner.invoke(app, ["clean"]).exit_code != 0


def test_clean_command_writes_normalized_names(tmp_path: Path) -> None:
    input_file = tmp_path / "names.csv"
    output_file = tmp_path / "cleaned.csv"
    input_file.write_text("Name,seat\nJohn Doe,A\nJane Smith,B\n", encoding="utf-8")

    result = runner.invoke(
        app, ["clean", str(input_file), "--output", str(output_file)]
    )

    assert result.exit_code == 0, result.output
    assert output_file.exists()
    assert "Processed 2 names" in result.stdout


def test_merge_and_preprocess_commands_use_lookup_files(tmp_path: Path) -> None:
    augmented_file = tmp_path / "augmented.csv"
    preprocessed_file = tmp_path / "preprocessed.csv"
    merge_result = runner.invoke(
        app,
        [
            "merge-supp",
            "examples/merge_supp_data/sample_in.csv",
            "--output",
            str(augmented_file),
            "--prefix-file",
            "examples/merge_supp_data/prefixes.csv",
            "--nickname-file",
            "examples/merge_supp_data/nick_names.txt",
        ],
    )

    preprocess_result = runner.invoke(
        app,
        [
            "preprocess",
            str(augmented_file),
            "--output",
            str(preprocessed_file),
            "--patterns",
            "FirstName LastName",
        ],
    )

    assert merge_result.exit_code == 0, merge_result.output
    assert preprocess_result.exit_code == 0, preprocess_result.output
    assert preprocessed_file.exists()


def test_search_command_writes_stable_results(tmp_path: Path) -> None:
    corpus_file = tmp_path / "corpus.csv"
    names_file = tmp_path / "names.csv"
    output_file = tmp_path / "results.csv"
    corpus_file.write_text(
        "uniqid,text\n1,Jane Doe spoke.\n2,Nobody spoke.\n", encoding="utf-8"
    )
    names_file.write_text("uniqid,search_name\n7,Jane Doe\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "search",
            str(corpus_file),
            "--names",
            str(names_file),
            "--output",
            str(output_file),
            "--processes",
            "1",
            "--fuzzy-rule",
            "8:1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Searched 2 rows; found 1 matches" in result.stdout
    with output_file.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["count"] for row in rows] == ["1", "0"]


def test_search_command_rejects_invalid_fuzzy_rule(tmp_path: Path) -> None:
    corpus_file = tmp_path / "corpus.csv"
    corpus_file.write_text("text\nJane Doe\n", encoding="utf-8")

    result = runner.invoke(app, ["search", str(corpus_file), "--fuzzy-rule", "invalid"])

    assert result.exit_code != 0
    assert "invalid fuzzy rule" in result.output
    assert "MINIMUM_LENGTH:MAX_EDIT_DISTANCE" in result.output


def test_split_and_merge_results_commands(tmp_path: Path) -> None:
    corpus_file = tmp_path / "corpus.csv"
    chunk_pattern = str(tmp_path / "chunk_{chunk_id}.csv")
    corpus_file.write_text("text\none\ntwo\nthree\n", encoding="utf-8")

    split_result = runner.invoke(
        app,
        [
            "split",
            str(corpus_file),
            "--output",
            chunk_pattern,
            "--size",
            "2",
        ],
    )
    merged_file = tmp_path / "merged.csv"
    merge_result = runner.invoke(
        app,
        [
            "merge-results",
            str(tmp_path / "chunk_1.csv"),
            str(tmp_path / "chunk_2.csv"),
            "--output",
            str(merged_file),
        ],
    )

    assert split_result.exit_code == 0, split_result.output
    assert merge_result.exit_code == 0, merge_result.output
    with merged_file.open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 3


def test_pipeline_command_runs_all_four_stages(tmp_path: Path) -> None:
    names_file = tmp_path / "raw_names.csv"
    corpus_file = tmp_path / "corpus.csv"
    output_dir = tmp_path / "pipeline"
    names_file.write_text("Name,seat\nJohn Doe,federal:president\n", encoding="utf-8")
    corpus_file.write_text("uniqid,text\n1,John Doe spoke.\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "pipeline",
            str(names_file),
            str(corpus_file),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "04_search_results.csv").exists()
