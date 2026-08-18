"""Normalize a CSV column of personal names into explicit components."""

import csv
import re
from pathlib import Path

from nameparser import HumanName

DEFAULT_OUTPUT = "clean_names.csv"
OUTPUT_COLUMNS = (
    "uniqid",
    "FirstName",
    "MiddleInitial/Name",
    "LastName",
    "RomanNumeral",
    "Title",
    "Suffix",
)
_ROMAN_NUMERALS = frozenset(
    {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"}
)
_STANDARD_SUFFIX = re.compile(r"(JR|SR|PHD)[^.]", flags=re.IGNORECASE)


def _parse_name(raw_name: str) -> dict[str, str]:
    cleaned = re.sub(r"\s*\(.*?\)\s*", " ", raw_name)
    cleaned = re.sub(r"\s*['\"].*?['\"]\s*", " ", cleaned)
    parsed = HumanName(cleaned)

    if not parsed.last:
        suffix_parts = parsed.suffix.split(",")
        if len(suffix_parts) >= 2:
            parsed = HumanName(f"{parsed.first}, {suffix_parts[1]} {suffix_parts[0]}")

    first_name = parsed.first.casefold()
    middle_name = parsed.middle.casefold()
    title = parsed.title
    roman_numeral = ""
    suffix_parts = []
    for suffix_part in parsed.suffix.split(","):
        normalized_suffix = suffix_part.strip()
        if normalized_suffix.upper() in _ROMAN_NUMERALS:
            roman_numeral = normalized_suffix
        elif normalized_suffix:
            suffix_parts.append(normalized_suffix)

    middle_parts = middle_name.split()
    if middle_parts:
        trailing_middle = middle_parts[-1].rstrip(".")
        if len(middle_parts) > 1 and trailing_middle.upper() in _ROMAN_NUMERALS:
            roman_numeral = trailing_middle
            middle_parts.pop()
        elif trailing_middle in {"mr", "ms"}:
            title = trailing_middle
            middle_parts.pop()

    if title.upper() in {"POPE", "BARON", "MAHDI"}:
        first_name = f"{title.casefold()} {first_name}".strip()
        title = ""

    middle_name = " ".join(
        f"{part}." if len(part) == 1 else part for part in middle_parts
    )
    suffix = _STANDARD_SUFFIX.sub(r"\1.", f"{', '.join(suffix_parts)} ").strip()

    return {
        "FirstName": first_name.upper(),
        "MiddleInitial/Name": middle_name.upper(),
        "LastName": parsed.last,
        "RomanNumeral": roman_numeral.upper(),
        "Title": title.upper(),
        "Suffix": suffix.upper(),
    }


def clean_names(
    input_file: str | Path,
    output_file: str | Path = DEFAULT_OUTPUT,
    name_column: str = "Name",
    keep_duplicates: bool = False,
) -> list[dict[str, str]]:
    """Parse names, preserve source columns, and write normalized CSV records."""
    input_path = Path(input_file)
    output_path = Path(output_file)
    records: list[dict[str, str]] = []
    seen_names: set[tuple[str, str, str, str]] = set()

    with input_path.open(encoding="utf-8", newline="") as input_stream:
        reader = csv.DictReader(input_stream)
        if name_column not in (reader.fieldnames or []):
            raise ValueError(f"input file is missing name column: {name_column}")
        source_columns = [
            column
            for column in (reader.fieldnames or [])
            if column not in OUTPUT_COLUMNS
        ]

        for source_row in reader:
            for raw_name in re.split(r"[&/]", source_row[name_column]):
                components = _parse_name(raw_name)
                identity = (
                    components["FirstName"],
                    components["MiddleInitial/Name"],
                    components["LastName"],
                    components["RomanNumeral"],
                )
                if not keep_duplicates and identity in seen_names:
                    continue
                seen_names.add(identity)
                record = {column: source_row[column] for column in source_columns}
                record.update(components)
                record["uniqid"] = str(len(records) + 1)
                records.append(record)

    with output_path.open("w", encoding="utf-8", newline="") as output_stream:
        writer = csv.DictWriter(
            output_stream, fieldnames=[*source_columns, *OUTPUT_COLUMNS]
        )
        writer.writeheader()
        writer.writerows(records)
    return records
