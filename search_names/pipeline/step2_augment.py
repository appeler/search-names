#!/usr/bin/env python

import argparse
import csv

from ..logging_config import get_logger

logger = get_logger("pipeline.step2_augment")

DEFAULT_OUTPUT = "augmented_clean_names.csv"
DEFAULT_NAME_LOOKUP = "FirstName"
DEFAULT_PREFIX_LOOKUP = "seat"
PREFIX_FILE = "prefixes.csv"
NICK_NAMES_FILE = "nick_names.txt"


def parse_command_line(argv):
    """Parse command line options"""
    parser = argparse.ArgumentParser(description="Merge supplement data")

    parser.add_argument("input", help="Input file name")

    parser.add_argument(
        "-o",
        "--out",
        type=str,
        dest="outfile",
        default=DEFAULT_OUTPUT,
        help=f"Output file in CSV (default: {DEFAULT_OUTPUT!s})",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        dest="name",
        default=DEFAULT_NAME_LOOKUP,
        help=f"Name of column use for nick name look up\
                        (default: {DEFAULT_NAME_LOOKUP!s})",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        dest="prefix",
        default=DEFAULT_PREFIX_LOOKUP,
        help=f"Name of column use for prefix look up\
                        (default: {DEFAULT_PREFIX_LOOKUP!s})",
    )
    parser.add_argument(
        "--prefix-file",
        type=str,
        dest="prefix_file",
        default=PREFIX_FILE,
        help=f"CSV File contains list of prefixes\
                        (default: {PREFIX_FILE!s})",
    )
    parser.add_argument(
        "--nick-name-file",
        type=str,
        dest="nickname_file",
        default=NICK_NAMES_FILE,
        help=f"Text File contains list of nick names\
                        (default: {NICK_NAMES_FILE!s})",
    )
    return parser.parse_args(argv)


def load_prefixes(filename, col):
    prefixes = {}
    try:
        with open(filename) as f:
            reader = csv.DictReader(f)
            for r in reader:
                prefixes[r[col]] = r["prefixes"]
    except Exception:
        logger.warning(f"Prefix file {filename} not found")

    return prefixes


def load_nick_names(filename):
    nick_names = {}
    try:
        with open(filename) as f:
            for line in f:
                line = line.strip().lower()
                if len(line):  # null string
                    continue
                a = line.split("-")
                if len(a) >= 2:
                    names = a[0].split(",")
                    names = [s.strip() for s in names]
                    nicks = a[1].split(",")
                    nicks = [s.strip() for s in nicks]
                    nicks = ";".join(nicks)
                    for n in names:
                        if n in nick_names:
                            logger.warning(f"Duplicate nick name '{nicks}' for '{n}'")
                        else:
                            nick_names[n] = nicks
                else:
                    logger.warning(f"Invalid nick name line '{line}'")
    except Exception:
        logger.warning(f"Nick name file {filename} not found")

    return nick_names


def augment_names(
    infile=None,
    prefixarg=DEFAULT_PREFIX_LOOKUP,
    name=DEFAULT_NAME_LOOKUP,
    outfile=DEFAULT_OUTPUT,
    prefix_file=PREFIX_FILE,
    nickname_file=NICK_NAMES_FILE,
):
    """Merge supplement data to names file"""

    prefixes = load_prefixes(prefix_file, prefixarg)
    nick_names = load_nick_names(nickname_file)

    # Bound before the try, not inside it: the finally below closes them, so an
    # exception raised before these assignments made the cleanup handler itself
    # fail with NameError -- masking whatever actually went wrong.
    f = None
    o = None
    try:
        f = open(infile)
        reader = csv.DictReader(f)
        o = open(outfile, "w")
        writer = csv.DictWriter(o, fieldnames=reader.fieldnames + ["prefixes", "nick_names"])
        writer.writeheader()

        for i, r in enumerate(reader):
            logger.debug(f"#{i}: {r[name].lower()}")
            # Prefix
            k = r[prefixarg]
            if k in prefixes:
                prefix = prefixes[k]
            else:
                prefix = ""
            r["prefixes"] = prefix
            # Nick name
            k = r[name].lower()
            if k in nick_names:
                nick = nick_names[k]
            else:
                nick = ""
            r["nick_names"] = nick
            writer.writerow(r)
    except Exception as e:
        # The log call used to sit *after* the bare `raise`, so it never ran and
        # this path re-raised silently.
        logger.error(f"Error: {e}")
        raise
    finally:
        if o:
            o.close()
        if f:
            f.close()

    logger.info("Done.")
