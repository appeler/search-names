"""Search a CSV corpus with one consistent parallel implementation."""

import csv
import gzip
import multiprocessing as mp
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

from .._csv import allow_large_fields
from ..engines import RESULT_FIELDS, FuzzyRule, SearchEngine

DEFAULT_OUTPUT_FILE = "search_results.csv"
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_INPUT_COLUMNS = ("uniqid", "text")
DEFAULT_CHUNK_SIZE = 1_000


def _open_input(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


def load_names_file(
    name_file: str | Path,
    id_column: str = "uniqid",
    name_column: str = "search_name",
) -> list[tuple[str, str]]:
    """Load identifier/name pairs from CSV or compressed CSV."""
    path = Path(name_file)
    with _open_input(path) as input_stream:
        reader = csv.DictReader(input_stream)
        missing = {id_column, name_column}.difference(reader.fieldnames or [])
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"name file is missing columns: {missing_columns}")
        names = [(row[id_column], row[name_column]) for row in reader]
    if any(not identifier.strip() for identifier, _ in names):
        raise ValueError("name identifiers cannot be blank")
    if any(not name.strip() for _, name in names):
        raise ValueError("search names cannot be blank")
    return names


def _output_header(
    input_columns: tuple[str, ...],
    max_results: int,
) -> list[str]:
    header = list(input_columns)
    for result_number in range(1, max_results + 1):
        header.extend(f"name{result_number}.{field}" for field in RESULT_FIELDS)
    header.append("count")
    return header


def _format_result_row(
    row: dict[str, str],
    matches: list[dict[str, Any]],
    input_columns: tuple[str, ...],
    max_results: int,
) -> list[Any]:
    output: list[Any] = [row.get(column, "") for column in input_columns]
    for result_number in range(max_results):
        match = matches[result_number] if result_number < len(matches) else {}
        output.extend(match.get(field, "") for field in RESULT_FIELDS)
    output.append(len(matches))
    return output


def _iter_chunks(
    corpus_file: Path,
    chunk_size: int,
) -> Iterator[list[dict[str, str]]]:
    with _open_input(corpus_file) as input_stream:
        reader = csv.DictReader(input_stream)
        chunk: list[dict[str, str]] = []
        for row in reader:
            chunk.append(row)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


_WORKER_ENGINE: SearchEngine | None = None


def _initialize_worker(
    names: list[tuple[str, str]], fuzzy_rules: list[FuzzyRule]
) -> None:
    global _WORKER_ENGINE
    _WORKER_ENGINE = SearchEngine(names, fuzzy_rules)


def _search_chunk(
    payload: tuple[list[dict[str, str]], str, tuple[str, ...], int],
) -> list[list[Any]]:
    if _WORKER_ENGINE is None:
        raise RuntimeError("search worker was not initialized")
    chunk, text_column, input_columns, max_results = payload
    return [
        _format_result_row(
            row,
            _WORKER_ENGINE.search(row.get(text_column, ""), max_results),
            input_columns,
            max_results,
        )
        for row in chunk
    ]


def search_names(
    corpus_file: str | Path,
    names: list[tuple[str, str]],
    output_file: str | Path = DEFAULT_OUTPUT_FILE,
    *,
    text_column: str = DEFAULT_TEXT_COLUMN,
    input_columns: tuple[str, ...] = DEFAULT_INPUT_COLUMNS,
    max_results: int = 20,
    fuzzy_rules: list[FuzzyRule] | None = None,
    processes: int = 4,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, int]:
    """Search every corpus row and write a stable, wide result table."""
    if processes < 1:
        raise ValueError("processes must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if max_results < 1:
        raise ValueError("max_results must be positive")
    if not input_columns or any(not column for column in input_columns):
        raise ValueError("input_columns must contain nonempty column names")

    corpus_path = Path(corpus_file)
    output_path = Path(output_file)
    if output_path.resolve() == corpus_path.resolve():
        raise ValueError("output file cannot also be the corpus file")

    output_header = _output_header(input_columns, max_results)
    if len(output_header) != len(set(output_header)):
        raise ValueError("output column names must be unique")

    allow_large_fields()
    with _open_input(corpus_path) as input_stream:
        reader = csv.DictReader(input_stream)
        corpus_columns = list(reader.fieldnames or [])
    if not corpus_columns:
        raise ValueError("corpus file must have a header")
    if len(corpus_columns) != len(set(corpus_columns)):
        raise ValueError("corpus column names must be unique")
    missing = {text_column, *input_columns}.difference(corpus_columns)
    if missing:
        raise ValueError(
            f"corpus file is missing columns: {', '.join(sorted(missing))}"
        )

    rules = fuzzy_rules or []
    chunks = _iter_chunks(corpus_path, chunk_size)
    payloads = ((chunk, text_column, input_columns, max_results) for chunk in chunks)

    total_rows = 0
    total_matches = 0
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_stream:
            temporary_path = Path(output_stream.name)
            writer = csv.writer(output_stream)
            writer.writerow(output_header)

            if processes == 1:
                _initialize_worker(names, rules)
                result_chunks = map(_search_chunk, payloads)
                pool = None
            else:
                pool = mp.get_context("spawn").Pool(
                    processes,
                    initializer=_initialize_worker,
                    initargs=(names, rules),
                )
                result_chunks = pool.imap(_search_chunk, payloads)

            try:
                for result_rows in result_chunks:
                    writer.writerows(result_rows)
                    total_rows += len(result_rows)
                    total_matches += sum(int(row[-1]) for row in result_rows)
            except BaseException:
                if pool is not None:
                    pool.terminate()
                    pool.join()
                raise
            else:
                if pool is not None:
                    pool.close()
                    pool.join()

        temporary_path.replace(output_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return {"total_rows": total_rows, "total_matches": total_matches}
