#!/usr/bin/env python

import argparse
import logging
import sys
from csv import DictReader, DictWriter

LOG_FILE = "merge_results.log"
DEFAULT_OUTPUT_FILE = "merged_search_results.csv"


def setup_logger():
    """Set up logging"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        filename=LOG_FILE,
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def merge_results(infile=None, outfile=DEFAULT_OUTPUT_FILE):
    out = open(outfile, "w")
    count = 0
    writer = None
    try:
        for i in infile:
            logging.info(f"Merging...: '{i}'")
            with open(i) as f:
                reader = DictReader(f)
                # `writer is None` rather than `n == 0`: same behaviour, and
                # it does not depend on enumerate starting at zero or on the
                # first file being readable.
                if writer is None:
                    writer = DictWriter(out, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for r in reader:
                    writer.writerow(r)
                    count += 1
    except Exception as e:
        logging.error(e)
    finally:
        out.close()

    logging.info(f"Done! (merge: {count} rows, from {len(infile)} files")


def main(argv=sys.argv[1:]):
    """Parse command line options"""
    parser = argparse.ArgumentParser(description="Merge search results from multiple chunks")

    parser.add_argument("inputs", nargs="*", help="CSV input file(s) name")

    parser.add_argument(
        "-o",
        "--out",
        type=str,
        dest="outfile",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output file in CSV (default: {DEFAULT_OUTPUT_FILE:s})",
    )

    args = parser.parse_args(argv)

    setup_logger()

    logging.info(str(args))

    merge_results(args.inputs, args.outfile)

    return 0


if __name__ == "__main__":
    sys.exit(main())
