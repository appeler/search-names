"""Correct, deterministic exact and fuzzy name matching."""

from collections import defaultdict
from typing import Any

import regex

MAX_RESULTS = 20
RESULT_FIELDS = ("uniqid", "n", "match", "start", "end")
FuzzyRule = tuple[int, int]


class SearchEngine:
    """Compile name patterns once and search many documents."""

    def __init__(
        self,
        names: list[tuple[str, str]],
        fuzzy_rules: list[FuzzyRule] | None = None,
    ) -> None:
        """Validate rules and compile grouped name patterns.

        Args:
            names: ``(identifier, name)`` pairs.
            fuzzy_rules: ``(minimum name length, maximum edit distance)`` pairs.

        Raises:
            ValueError: If a name or fuzzy rule is invalid.
        """
        self.fuzzy_rules = self._validate_fuzzy_rules(fuzzy_rules or [])
        grouped_names: dict[str, list[str]] = defaultdict(list)
        for identifier, name in names:
            normalized_name = name.strip()
            if not normalized_name:
                raise ValueError("search names cannot be blank")
            grouped_names[str(identifier)].append(normalized_name)

        self.patterns = [
            (identifier, self._compile(names_for_identifier))
            for identifier, names_for_identifier in grouped_names.items()
        ]

    @staticmethod
    def _validate_fuzzy_rules(rules: list[FuzzyRule]) -> list[FuzzyRule]:
        validated: list[FuzzyRule] = []
        for minimum_length, edit_distance in rules:
            if minimum_length < 1:
                raise ValueError("fuzzy-rule minimum lengths must be positive")
            if edit_distance < 0:
                raise ValueError("fuzzy-rule edit distances cannot be negative")
            validated.append((minimum_length, edit_distance))
        return sorted(validated)

    def _edit_distance(self, name: str) -> int:
        distance = 0
        for minimum_length, candidate_distance in self.fuzzy_rules:
            if len(name) >= minimum_length:
                distance = candidate_distance
        return distance

    def _compile(self, names: list[str]) -> regex.Pattern[str]:
        alternatives = []
        for name in names:
            escaped_name = regex.escape(name)
            edit_distance = self._edit_distance(name)
            if edit_distance:
                alternatives.append(f"(?:{escaped_name}){{e<={edit_distance}}}")
            else:
                alternatives.append(escaped_name)
        return regex.compile(
            rf"(?<!\w)(?:{'|'.join(alternatives)})(?!\w)", flags=regex.IGNORECASE
        )

    def search(self, text: str, max_results: int = MAX_RESULTS) -> list[dict[str, Any]]:
        """Return up to ``max_results`` matches in stable pattern order."""
        if max_results < 1:
            raise ValueError("max_results must be positive")

        matches: list[dict[str, Any]] = []
        for identifier, pattern in self.patterns:
            for match in pattern.finditer(text):
                matches.append(
                    {
                        "uniqid": identifier,
                        "n": 1,
                        "match": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
                if len(matches) == max_results:
                    return matches
        return matches
