#!/usr/bin/env python

import argparse
import csv
import re

from nameparser import HumanName

from ..logging_config import get_logger
from ..streaming_utils import process_large_csv, should_use_streaming

logger = get_logger("pipeline.step1_clean")

DEFAULT_OUTPUT = "clean_names.csv"
re_std_suffix = re.compile(r"(JR|SR|PHD)[^\.]", flags=re.I)


def parse_command_line(argv):
    """Parse command line options"""
    parser = argparse.ArgumentParser(description="Clean name")

    parser.add_argument("input", help="Input file name")

    parser.add_argument(
        "-o",
        "--out",
        action="store",
        type=str,
        dest="outfile",
        default=DEFAULT_OUTPUT,
        help=f"Output file in CSV (default: {DEFAULT_OUTPUT!s})",
    )
    parser.add_argument(
        "-c",
        "--column",
        action="store",
        type=str,
        dest="column",
        default="Name",
        help="Column name file in CSV contains Name list \
                        (default: Name)",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        dest="all",
        default=False,
        help="Export all names (not take duplicate names out)\
                        (default: False)",
    )
    return parser.parse_args(argv)


def clean_names(infile, outfile=DEFAULT_OUTPUT, col="Name", all=False):
    """ Read names and pre-process
        Returns unique names in format "FirstName LastName AnyRomanNumeral"\
        or "FirstName LastName"
    """
    logger.info("Processing and exporting, please wait...")

    ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    # `of` is used later behind `if outfile:`, which is the same condition that
    # binds it -- true today, and unprovable, so bind it unconditionally rather
    # than rely on two guards staying in step.
    of = None
    writer = None
    if outfile:
        try:
            of = open(outfile, "w")
        except OSError:
            outfile = None

    with open(infile) as f:
        reader = csv.DictReader(f)
        if outfile:
            writer = csv.DictWriter(
                of,
                fieldnames=reader.fieldnames
                + [
                    "uniqid",
                    "FirstName",
                    "MiddleInitial/Name",
                    "LastName",
                    "RomanNumeral",
                    "Title",
                    "Suffix",
                ],
            )
            writer.writeheader()
        rowid = 0
        allnames = []
        allnameswithid = []
        for r in reader:
            rname = r[col]
            for name in re.split("[&/]", rname):
                name, n = re.subn(r"\s*\(.*\)\s*", " ", name)
                if n > 0:
                    # print "Remove Parenthesis...", name
                    pass
                name, n = re.subn(r'\s*[\'"].*[\"\']\s*', " ", name)
                if n > 0:
                    # print "Remove Quote...", name
                    pass
                name = HumanName(name)
                if name.last == "":
                    a = name.suffix.split(",")
                    if len(a) >= 2:
                        name = HumanName(name.first + ", " + a[1] + " " + a[0])
                first = name.first.lower()
                mid = name.middle.lower()
                roman = ""
                title = name.title

                last = ""
                suffix_list = []
                for s in name.suffix.split(","):
                    if s.strip() in ROMAN:
                        roman = s
                        last = name.last.lower() + " " + roman.strip().lower()
                    else:
                        suffix_list.append(s)
                if last == "":
                    last = name.last.lower()
                suffix = ", ".join(suffix_list)

                if last == "":
                    logger.debug(f"Empty last name for: {repr(name)}")

                # Fixed ROMAN and Title in Middle
                if mid != "":
                    m_list = mid.split()
                    m = m_list[-1].strip()
                    m = m.strip(".")
                    if len(m_list) > 1 and m.upper() in ROMAN:
                        roman = m
                        mid = " ".join(m_list[:-1])
                        # print rname, "==>", roman, "==>", mid
                    if m in ["mr", "ms"]:
                        title = m
                        mid = " ".join(m_list[:-1])
                        # print rname, "==>", title, "==>", mid

                # Adhoc fixed for Title
                if title in ["POPE", "BARON", "MAHDI"]:
                    first = title + " " + first
                    # print rname, "==>", title, "==>", first
                    title = ""

                # Standardize Jr/Sr suffix
                suffix = re_std_suffix.sub(r"\1.", suffix + " ").strip()

                # Standardize Middle Initial
                std_mid = []
                for m in mid.split():
                    if len(m) == 1:
                        m = m + "."
                    std_mid.append(m)
                mid = " ".join(std_mid)

                if all or (first, mid, last) not in allnames:
                    rowid += 1
                    r["uniqid"] = rowid
                    allnameswithid.append((r["uniqid"], first, mid, last, r["seat"].strip()))
                    allnames.append((first, mid, last))
                    s = {
                        "FirstName": first.upper(),
                        "MiddleInitial/Name": mid.upper(),
                        "LastName": name.last,
                        "RomanNumeral": roman.upper(),
                        "Title": title.upper(),
                        "Suffix": suffix.upper(),
                    }
                    t = dict(r, **s)
                    if outfile:
                        # Bound under this same condition further up; asserted
                        # so the two guards cannot drift apart silently.
                        assert writer is not None
                        writer.writerow(t)
        if outfile:
            of.close()
        logger.info("Done.")
        return allnameswithid
    return None


def clean_names_streaming(
    infile: str,
    outfile: str = DEFAULT_OUTPUT,
    col: str = "Name",
    all: bool = False,
    chunk_size: int = 1000,
):
    """
    Memory-efficient version of clean_names for large files.

    Args:
        infile: Input CSV file path
        outfile: Output CSV file path
        col: Column name containing names
        all: Export all names (don't remove duplicates)
        chunk_size: Number of rows to process at once
    """

    def process_chunk(rows):
        """Process a chunk of rows."""
        processed_rows = []
        name_set = set() if not all else None

        for i, row in enumerate(rows):
            try:
                orig_name = row[col]
                if not orig_name or not orig_name.strip():
                    continue

                # Clean and parse name
                name = orig_name.strip()
                name = re.sub(r"\s+", " ", name)  # normalize whitespace

                parsed = HumanName(name)

                # Build output row
                new_row = row.copy()
                new_row.update(
                    {
                        "uniqid": f"{hash(orig_name)}_{i}",  # Simple unique ID
                        "FirstName": parsed.first.title(),
                        "MiddleName": parsed.middle.title(),
                        "LastName": parsed.last.title(),
                        "Prefix": parsed.title.title(),
                        "Suffix": parsed.suffix.title(),
                    }
                )

                # Check for duplicates if needed
                if not all:
                    name_key = f"{new_row['FirstName']} {new_row['LastName']}".lower()
                    if name_key in name_set:
                        continue
                    name_set.add(name_key)

                processed_rows.append(new_row)

            except Exception as e:
                logger.warning(f"Error processing name '{row.get(col, '')}': {e}")
                continue

        return processed_rows

    # Use streaming for large files
    if should_use_streaming(infile):
        logger.info(f"Using streaming for large file: {infile}")
        stats = process_large_csv(infile, outfile, process_chunk, chunk_size)
        logger.info(f"Processed {stats['total_rows']} rows in {stats['chunks_processed']} chunks")
        return stats["total_rows"]
    else:
        # Use original function for smaller files
        return clean_names(infile, outfile, col, all)
