"""Merge search-result CSV files with strict schema validation."""

import csv
from pathlib import Path

from ._csv import allow_large_fields

DEFAULT_OUTPUT_FILE = "merged_search_results.csv"


def merge_results(
    input_files: list[str | Path],
    output_file: str | Path = DEFAULT_OUTPUT_FILE,
) -> int:
    """Concatenate compatible CSV files and return the number of data rows."""
    if not input_files:
        raise ValueError("at least one input file is required")

    allow_large_fields()

    input_paths = [Path(path) for path in input_files]
    output_path = Path(output_file)
    if output_path.resolve() in {path.resolve() for path in input_paths}:
        raise ValueError("output file cannot also be an input file")

    expected_columns: list[str] | None = None
    for input_path in input_paths:
        with input_path.open(encoding="utf-8", newline="") as input_stream:
            columns = list(csv.DictReader(input_stream).fieldnames or [])
        if not columns:
            raise ValueError(f"CSV file has no header: {input_path}")
        if len(columns) != len(set(columns)):
            raise ValueError(f"CSV column names are not unique in {input_path}")
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise ValueError(f"CSV schema mismatch in {input_path}")

    if expected_columns is None:
        raise RuntimeError("CSV schema was not initialized")

    row_count = 0
    with output_path.open("w", encoding="utf-8", newline="") as output_stream:
        writer = csv.DictWriter(output_stream, fieldnames=expected_columns)
        writer.writeheader()
        for input_path in input_paths:
            with input_path.open(encoding="utf-8", newline="") as input_stream:
                reader = csv.DictReader(input_stream)
                for row in reader:
                    writer.writerow(row)
                    row_count += 1
    return row_count
