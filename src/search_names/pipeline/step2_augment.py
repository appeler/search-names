"""Add prefix and nickname variants to normalized name records."""

import csv
from pathlib import Path

DEFAULT_OUTPUT = "augmented_clean_names.csv"
DEFAULT_NAME_LOOKUP = "FirstName"
DEFAULT_PREFIX_LOOKUP = "seat"
OUTPUT_COLUMNS = ("prefixes", "nick_names")


def load_prefixes(filename: str | Path | None, lookup_column: str) -> dict[str, str]:
    """Load prefix values keyed by the configured lookup column."""
    if filename is None:
        return {}
    with Path(filename).open(encoding="utf-8", newline="") as input_stream:
        reader = csv.DictReader(input_stream)
        missing = {lookup_column, "prefixes"}.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"prefix file is missing columns: {', '.join(sorted(missing))}"
            )
        return {row[lookup_column]: row["prefixes"] for row in reader}


def load_nick_names(filename: str | Path | None) -> dict[str, str]:
    """Load ``name[,name]-nickname[,nickname]`` mappings."""
    if filename is None:
        return {}
    nicknames: dict[str, str] = {}
    with Path(filename).open(encoding="utf-8") as input_stream:
        for line_number, raw_line in enumerate(input_stream, start=1):
            line = raw_line.strip().casefold()
            if not line or line.startswith("#"):
                continue
            if "-" not in line:
                raise ValueError(f"invalid nickname mapping on line {line_number}")
            names_text, nicknames_text = line.split("-", maxsplit=1)
            names = [value.strip() for value in names_text.split(",") if value.strip()]
            values = [
                value.strip() for value in nicknames_text.split(",") if value.strip()
            ]
            if not names or not values:
                raise ValueError(f"invalid nickname mapping on line {line_number}")
            joined_values = ";".join(values)
            for name in names:
                if name in nicknames:
                    raise ValueError(f"duplicate nickname mapping for {name!r}")
                nicknames[name] = joined_values
    return nicknames


def augment_names(
    input_file: str | Path,
    prefix_column: str = DEFAULT_PREFIX_LOOKUP,
    name_column: str = DEFAULT_NAME_LOOKUP,
    output_file: str | Path = DEFAULT_OUTPUT,
    prefix_file: str | Path | None = None,
    nickname_file: str | Path | None = None,
) -> int:
    """Write a copy of the input with deterministic prefix/nickname columns."""
    prefixes = load_prefixes(prefix_file, prefix_column)
    nicknames = load_nick_names(nickname_file)

    with Path(input_file).open(encoding="utf-8", newline="") as input_stream:
        reader = csv.DictReader(input_stream)
        missing = {prefix_column, name_column}.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"input file is missing columns: {', '.join(sorted(missing))}"
            )
        source_columns = [
            column
            for column in (reader.fieldnames or [])
            if column not in OUTPUT_COLUMNS
        ]
        rows = []
        for source_row in reader:
            row = {column: source_row[column] for column in source_columns}
            row["prefixes"] = prefixes.get(source_row[prefix_column], "")
            row["nick_names"] = nicknames.get(source_row[name_column].casefold(), "")
            rows.append(row)

    with Path(output_file).open("w", encoding="utf-8", newline="") as output_stream:
        writer = csv.DictWriter(
            output_stream, fieldnames=[*source_columns, *OUTPUT_COLUMNS]
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
