"""Normalize a CSV column of personal names into explicit components."""

import csv
import re
from pathlib import Path

from ..enhanced_name_parser import NameParser, ParsedName, ParserType

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
_STANDARD_SUFFIX = re.compile(r"\b(JR|SR|PHD)\b(?!\.)", flags=re.IGNORECASE)
_GENERATION_WORDS = frozenset({"JR", "SR"}) | _ROMAN_NUMERALS
_COURTESY_TITLES = frozenset({"MR", "MS", "MRS"})


def _strip_asides(raw_name: str) -> str:
    """Drop bracketed and quoted asides, which hold nicknames and notes."""
    cleaned = re.sub(r"\s*\(.*?\)\s*", " ", raw_name)
    return re.sub(r"\s*['\"].*?['\"]\s*", " ", cleaned)


def _bare(word: str) -> str:
    return word.rstrip(".").upper()


def _components(parsed: ParsedName, parser: NameParser) -> dict[str, str]:
    comma_parts = [part.strip() for part in (parsed.suffix or "").split(",")]
    if len(comma_parts) >= 2 and not parsed.last_name:
        parsed = parser.parse_with_nameparser(
            f"{parsed.first_name or ''}, {comma_parts[1]} {comma_parts[0]}"
        )
    elif len(comma_parts) >= 2 and not parsed.first_name:
        # "SMITH, JR., JOHN": the given name was read as a second suffix.
        parsed = parser.parse_with_nameparser(
            f"{parsed.last_name}, {' '.join(comma_parts[1:])}, {comma_parts[0]}"
        )

    first_name = (parsed.first_name or "").casefold()
    last_name = parsed.last_name or ""
    title = parsed.title or ""
    middle_words = (parsed.middle_name or "").split()
    suffix_text = parsed.suffix or ""

    # A name that ends in "MR" or opens with "JR.," leaves that word where the
    # family name belongs; the family name is then the last middle word.
    if middle_words and _bare(last_name) in _COURTESY_TITLES:
        title, last_name = last_name, middle_words.pop()
    elif (
        middle_words
        and _bare(last_name) in _GENERATION_WORDS
        # "Vi" and a lone "X" are names; a generation word is dotted, or
        # more than one capital.
        and (last_name.endswith(".") or (len(last_name) > 1 and last_name.isupper()))
    ):
        suffix_text = ", ".join(part for part in (last_name, suffix_text) if part)
        last_name = middle_words.pop()

    roman_numeral = ""
    suffix_parts = []
    for suffix_part in suffix_text.split(","):
        kept_words = []
        for word in suffix_part.split():
            if word.upper() in _ROMAN_NUMERALS:
                roman_numeral = word
            else:
                kept_words.append(word)
        if kept_words:
            suffix_parts.append(" ".join(kept_words))

    middle_parts = [word.casefold() for word in middle_words]
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
    suffix = _STANDARD_SUFFIX.sub(r"\1.", ", ".join(suffix_parts))

    return {
        "FirstName": first_name.upper(),
        "MiddleInitial/Name": middle_name.upper(),
        "LastName": last_name,
        "RomanNumeral": roman_numeral.upper(),
        "Title": title.upper(),
        "Suffix": suffix.upper(),
    }


def clean_names(
    input_file: str | Path,
    output_file: str | Path = DEFAULT_OUTPUT,
    name_column: str = "Name",
    keep_duplicates: bool = False,
    parser_type: ParserType = "nameparser",
) -> list[dict[str, str]]:
    """Parse names, preserve source columns, and write normalized CSV records.

    Args:
        input_file: CSV file with a column of names.
        output_file: Where to write the normalized records.
        name_column: Column that holds the names.
        keep_duplicates: Keep names whose components repeat an earlier row.
        parser_type: Which parser splits a name into first and last name.
            "nameparser" reads given name first unless a comma says otherwise.
            "parsernaam" asks a model which order each name is written in.
            "auto" asks the model only for names with a common Indian name word.

    Returns:
        The normalized records, in output order.

    Raises:
        ValueError: If the name column is missing or parser_type is unknown.
    """
    parser = NameParser(parser_type=parser_type)
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

        entries = [
            (source_row, raw_name)
            for source_row in reader
            for raw_name in re.split(r"[&/]", source_row[name_column])
        ]

    parsed_names = parser.parse([_strip_asides(raw_name) for _, raw_name in entries])
    for (source_row, _), parsed in zip(entries, parsed_names, strict=True):
        components = _components(parsed, parser)
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
