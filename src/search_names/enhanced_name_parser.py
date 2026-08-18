"""Enhanced name parser with support for both HumanName and parsernaam."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal, overload

import pandas as pd

# Import nameparser (always available)
from nameparser import HumanName
from parsernaam import parse_names as parse_with_parsernaam_model

from .logging_config import get_logger

logger = get_logger("enhanced_name_parser")

ParserType = Literal["auto", "humanname", "parsernaam"]

_INDIAN_NAME_TOKENS = frozenset(
    {
        "kumar",
        "kumari",
        "devi",
        "singh",
        "das",
        "rao",
        "reddy",
        "sharma",
        "gupta",
        "patel",
        "shah",
        "mehta",
        "varma",
        "krishna",
        "ram",
        "sai",
        "venkat",
        "raj",
        "mohan",
        "swamy",
        "naidu",
        "choudhury",
        "mukherjee",
        "chatterjee",
    }
)


@dataclass
class ParsedName:
    """Unified parsed name representation."""

    original: str
    first_name: str | None = None
    middle_name: str | None = None
    last_name: str | None = None
    title: str | None = None
    suffix: str | None = None
    nickname: str | None = None
    model_score: float | None = None
    parser_used: str = "humanname"

    def full_name(self) -> str:
        """Get full name from components."""
        parts = []
        if self.title:
            parts.append(self.title)
        if self.first_name:
            parts.append(self.first_name)
        if self.middle_name:
            parts.append(self.middle_name)
        if self.last_name:
            parts.append(self.last_name)
        if self.suffix:
            parts.append(self.suffix)
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original": self.original,
            "first_name": self.first_name,
            "middle_name": self.middle_name,
            "last_name": self.last_name,
            "title": self.title,
            "suffix": self.suffix,
            "nickname": self.nickname,
            "model_score": self.model_score,
            "parser_used": self.parser_used,
        }


class NameParser:
    """Enhanced name parser with multiple backend support."""

    def __init__(
        self,
        parser_type: ParserType = "auto",
        batch_size: int = 100,
        ml_threshold: float = 0.8,
    ):
        """Initialize name parser.

        Args:
            parser_type: "humanname", "parsernaam", or "auto"
            batch_size: Batch size for parsernaam processing
            ml_threshold: Confidence threshold for ML predictions

        Raises:
            ValueError: If any parser setting is invalid.
        """
        if parser_type not in {"auto", "humanname", "parsernaam"}:
            raise ValueError("parser_type must be 'auto', 'humanname', or 'parsernaam'")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not 0 <= ml_threshold <= 1:
            raise ValueError("ml_threshold must be between 0 and 1")

        self.parser_type = parser_type
        self.batch_size = batch_size
        self.ml_threshold = ml_threshold

        logger.info("Initialized NameParser with type: %s", self.parser_type)

    def parse_with_humanname(self, name: str) -> ParsedName:
        """Parse name using HumanName."""
        if not name.strip():
            return ParsedName(original=name, parser_used="humanname")

        try:
            parsed = HumanName(name)

            return ParsedName(
                original=name,
                first_name=parsed.first or None,
                middle_name=parsed.middle or None,
                last_name=parsed.last or None,
                title=parsed.title or None,
                suffix=parsed.suffix or None,
                nickname=parsed.nickname or None,
                parser_used="humanname",
            )
        except Exception as error:
            logger.error("Error parsing name %r with HumanName: %s", name, error)
            return ParsedName(original=name, parser_used="humanname")

    def parse_with_parsernaam(self, names: list[str]) -> list[ParsedName]:
        """Parse names using parsernaam (batch processing)."""
        try:
            parsed_names: list[ParsedName] = []
            for offset in range(0, len(names), self.batch_size):
                batch = names[offset : offset + self.batch_size]
                results = parse_with_parsernaam_model(pd.DataFrame({"name": batch}))
                if "parsed_name" not in results.columns or len(results) != len(batch):
                    raise ValueError("parsernaam returned an unexpected result schema")

                for original, prediction in zip(
                    batch, results["parsed_name"], strict=True
                ):
                    parsed = self._parsed_name_from_prediction(original, prediction)
                    if (
                        parsed.model_score is not None
                        and parsed.model_score < self.ml_threshold
                    ):
                        parsed = self.parse_with_humanname(original)
                    parsed_names.append(parsed)

            return parsed_names

        except Exception as error:
            logger.error("Error parsing names with parsernaam: %s", error)
            # Fall back to humanname
            return [self.parse_with_humanname(name) for name in names]

    @staticmethod
    def _parsed_name_from_prediction(original: str, prediction: object) -> ParsedName:
        """Convert parsernaam's documented prediction record."""
        if not isinstance(prediction, Mapping):
            raise TypeError("parsernaam prediction must be a mapping")

        label = prediction.get("type")
        probability = prediction.get("prob")
        if label not in {"first", "last", "first_last", "last_first", "unknown"}:
            raise ValueError(f"unsupported parsernaam label: {label!r}")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError("parsernaam probability must be between 0 and 1")

        tokens = original.split()
        first_name = None
        middle_name = None
        last_name = None

        if label == "first":
            first_name = original.strip() or None
        elif label == "last":
            last_name = original.strip() or None
        elif label in {"first_last", "last_first"} and len(tokens) >= 2:
            leading, *middle, trailing = tokens
            middle_name = " ".join(middle) or None
            if label == "first_last":
                first_name, last_name = leading, trailing
            else:
                last_name, first_name = leading, trailing

        return ParsedName(
            original=original,
            first_name=first_name,
            middle_name=middle_name,
            last_name=last_name,
            model_score=float(probability),
            parser_used="parsernaam",
        )

    def is_indian_name(self, name: str) -> bool:
        """Check if name appears to be Indian."""
        tokens = {token.casefold() for token in re.findall(r"[^\W_]+", name)}
        return not tokens.isdisjoint(_INDIAN_NAME_TOKENS)

    @overload
    def parse(self, name: str) -> ParsedName: ...

    @overload
    def parse(self, name: list[str]) -> list[ParsedName]: ...

    def parse(self, name: str | list[str]) -> ParsedName | list[ParsedName]:
        """Parse one or more names.

        Args:
            name: Single name string or list of names

        Returns:
            ParsedName or list of ParsedName objects
        """
        # Handle single name
        if isinstance(name, str):
            if self.parser_type == "humanname":
                return self.parse_with_humanname(name)
            if self.parser_type == "parsernaam":
                results = self.parse_with_parsernaam([name])
                return results[0] if results else ParsedName(original=name)
            if self.is_indian_name(name):
                results = self.parse_with_parsernaam([name])
                return results[0] if results else self.parse_with_humanname(name)
            return self.parse_with_humanname(name)

        # Handle list of names
        names_list = name

        if self.parser_type == "humanname":
            return [self.parse_with_humanname(n) for n in names_list]
        if self.parser_type == "parsernaam":
            return self.parse_with_parsernaam(names_list)

        indian_indices = [
            index
            for index, candidate in enumerate(names_list)
            if self.is_indian_name(candidate)
        ]
        indian_names = [names_list[index] for index in indian_indices]
        parsed_indian = self.parse_with_parsernaam(indian_names)
        parsed_by_index = dict(zip(indian_indices, parsed_indian, strict=True))
        return [
            parsed_by_index[index]
            if index in parsed_by_index
            else self.parse_with_humanname(candidate)
            for index, candidate in enumerate(names_list)
        ]

    def parse_dataframe(
        self, df: pd.DataFrame, name_column: str = "name", add_components: bool = True
    ) -> pd.DataFrame:
        """Parse names in a DataFrame.

        Args:
            df: Input DataFrame
            name_column: Column containing names
            add_components: Whether to add parsed components as new columns

        Returns:
            DataFrame with parsed names

        Raises:
            TypeError: If the name column contains non-string values.
            ValueError: If the requested name column does not exist.
        """
        if name_column not in df.columns:
            raise ValueError(f"Column '{name_column}' not found in DataFrame")

        result = df.copy()
        values = result[name_column].tolist()
        if not all(isinstance(value, str) for value in values):
            raise TypeError(f"Column '{name_column}' must contain only strings")
        names: list[str] = values
        parsed = self.parse(names)

        if add_components:
            # Add parsed components as new columns
            result["parsed_first_name"] = [p.first_name for p in parsed]
            result["parsed_middle_name"] = [p.middle_name for p in parsed]
            result["parsed_last_name"] = [p.last_name for p in parsed]
            result["parsed_title"] = [p.title for p in parsed]
            result["parsed_suffix"] = [p.suffix for p in parsed]
            result["parser_model_score"] = [p.model_score for p in parsed]
            result["parser_used"] = [p.parser_used for p in parsed]
        else:
            result["parsed_name"] = parsed

        return result


def parse_names(
    names: str | list[str] | pd.DataFrame,
    parser_type: ParserType = "auto",
    name_column: str = "name",
) -> ParsedName | list[ParsedName] | pd.DataFrame:
    """Convenience function to parse names.

    Args:
        names: Single name, list of names, or DataFrame
        parser_type: "humanname", "parsernaam", or "auto"
        name_column: Column name if input is DataFrame

    Returns:
        Parsed results in same format as input
    """
    parser = NameParser(parser_type=parser_type)

    if isinstance(names, pd.DataFrame):
        return parser.parse_dataframe(names, name_column=name_column)
    return parser.parse(names)


def compare_parsers(name: str) -> dict[str, ParsedName]:
    """Compare results from different parsers.

    Args:
        name: Name to parse

    Returns:
        Dictionary with results from each parser
    """
    results = {}

    # Parse with HumanName
    hn_parser = NameParser(parser_type="humanname")
    results["humanname"] = hn_parser.parse(name)

    # Parse with parsernaam
    pn_parser = NameParser(parser_type="parsernaam")
    results["parsernaam"] = pn_parser.parse(name)

    # Parse with auto
    auto_parser = NameParser(parser_type="auto")
    results["auto"] = auto_parser.parse(name)

    return results
