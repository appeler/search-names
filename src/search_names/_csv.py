"""Shared CSV limits for document-sized text fields."""

import csv
import sys


def allow_large_fields() -> None:
    """Raise the process-wide CSV field limit to the platform maximum."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10
