"""Split a CSV corpus into deterministic, schema-preserving chunks."""

import csv
from pathlib import Path
from typing import TextIO

from ._csv import allow_large_fields

DEFAULT_OUTPUT_PATTERN = "{basename}_{chunk_id:04d}.csv"


def split_text_corpus(
    input_file: str | Path,
    output_pattern: str = DEFAULT_OUTPUT_PATTERN,
    chunk_size: int = 1_000,
) -> int:
    """Split a CSV file and return the number of chunks written."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    input_path = Path(input_file)
    allow_large_fields()
    chunk_id = 0
    output_stream: TextIO | None = None
    writer: csv.DictWriter[str] | None = None

    try:
        with input_path.open(encoding="utf-8", newline="") as input_stream:
            reader = csv.DictReader(input_stream)
            source_columns = list(reader.fieldnames or [])
            if not source_columns:
                raise ValueError("input file must have a header")
            if len(source_columns) != len(set(source_columns)):
                raise ValueError("input column names must be unique")
            add_identifier = "uniqid" not in source_columns
            output_columns = (
                ["uniqid", *source_columns] if add_identifier else source_columns
            )

            for row_number, row in enumerate(reader):
                if row_number % chunk_size == 0:
                    if output_stream is not None:
                        output_stream.close()
                    chunk_id += 1
                    output_path = Path(
                        output_pattern.format(
                            basename=input_path.stem,
                            chunk_id=chunk_id,
                        )
                    )
                    if output_path.resolve() == input_path.resolve():
                        raise ValueError(
                            "a chunk output cannot overwrite the input file"
                        )
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_stream = output_path.open("w", encoding="utf-8", newline="")
                    writer = csv.DictWriter(output_stream, fieldnames=output_columns)
                    writer.writeheader()

                if add_identifier:
                    row["uniqid"] = str(row_number)
                if writer is None:
                    raise RuntimeError("CSV writer was not initialized")
                writer.writerow(row)
    finally:
        if output_stream is not None:
            output_stream.close()
    return chunk_id
