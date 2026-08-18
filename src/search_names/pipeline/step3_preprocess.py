"""Create and disambiguate search patterns from normalized name records."""

import csv
import itertools
from pathlib import Path

from Levenshtein import distance

DEFAULT_OUTPUT = "deduped_augmented_clean_names.csv"
DEFAULT_PATTERNS = ("FirstName LastName", "NickName LastName", "Prefix LastName")


def load_drop_patterns(filename: str | Path | None) -> set[str]:
    """Load case-insensitive patterns to exclude."""
    if filename is None:
        return set()
    with Path(filename).open(encoding="utf-8") as input_stream:
        return {
            line.strip().casefold()
            for line in input_stream
            if line.strip() and not line.lstrip().startswith("#")
        }


def _pattern_values(row: dict[str, str], field: str) -> list[str]:
    if field == "Prefix":
        return row.get("prefixes", "").split(";")
    if field == "NickName":
        return row.get("nick_names", "").split(";")
    if field not in row:
        raise ValueError(f"input file is missing pattern column: {field}")
    return [row[field]]


def _maximum_edit_distance(name: str, length_thresholds: tuple[int, ...]) -> int:
    return sum(len(name) > threshold for threshold in length_thresholds)


def preprocess_names(
    input_file: str | Path,
    patterns: tuple[str, ...] | list[str] = DEFAULT_PATTERNS,
    output_file: str | Path = DEFAULT_OUTPUT,
    edit_length_thresholds: tuple[int, ...] | list[int] = (),
    drop_patterns: set[str] | list[str] | None = None,
) -> int:
    """Build long-form patterns and remove ambiguous near-duplicates."""
    excluded = {pattern.casefold() for pattern in (drop_patterns or [])}
    thresholds = tuple(edit_length_thresholds)

    with Path(input_file).open(encoding="utf-8", newline="") as input_stream:
        reader = csv.DictReader(input_stream)
        source_columns = [
            column for column in (reader.fieldnames or []) if column != "search_name"
        ]
        generated: list[dict[str, str]] = []
        for source_row in reader:
            for pattern in patterns:
                value_groups = [
                    _pattern_values(source_row, field) for field in pattern.split()
                ]
                for combination in itertools.product(*value_groups):
                    search_name = " ".join(
                        value.strip().casefold()
                        for value in combination
                        if value.strip()
                    )
                    if len(search_name.split()) < 2 or search_name in excluded:
                        continue
                    row = {column: source_row[column] for column in source_columns}
                    row["search_name"] = search_name
                    generated.append(row)

    ambiguous_indexes: set[int] = set()
    for left_index, left_row in enumerate(generated):
        if left_index in ambiguous_indexes:
            continue
        left_name = left_row["search_name"]
        maximum_distance = _maximum_edit_distance(left_name, thresholds)
        for right_index in range(left_index + 1, len(generated)):
            right_row = generated[right_index]
            if distance(left_name, right_row["search_name"]) > maximum_distance:
                continue
            if left_row.get("uniqid") != right_row.get("uniqid"):
                ambiguous_indexes.update({left_index, right_index})
            else:
                ambiguous_indexes.add(right_index)

    output_rows = [
        row for index, row in enumerate(generated) if index not in ambiguous_indexes
    ]
    with Path(output_file).open("w", encoding="utf-8", newline="") as output_stream:
        writer = csv.DictWriter(
            output_stream, fieldnames=[*source_columns, "search_name"]
        )
        writer.writeheader()
        writer.writerows(output_rows)
    return len(output_rows)
