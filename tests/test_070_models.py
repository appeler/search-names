"""Tests for the NLP result models."""

import pytest
from pydantic import ValidationError

from search_names.models import EntityLinkingResult, EntityMention


def test_entity_mention_validates_span() -> None:
    mention = EntityMention(text="John Doe", label="PERSON", start=10, end=18)

    assert mention.text == "John Doe"


@pytest.mark.parametrize(("start", "end"), [(18, 10), (10, 10), (-1, 10)])
def test_entity_mention_rejects_invalid_span(start: int, end: int) -> None:
    with pytest.raises(ValidationError):
        EntityMention(text="John Doe", label="PERSON", start=start, end=end)


def test_entity_linking_result_validates_score() -> None:
    mention = EntityMention(text="John Doe", label="PERSON", start=0, end=8)

    result = EntityLinkingResult(
        mention=mention,
        linked_entity_id="person-1",
        linked_entity_name="John Doe",
        match_method="semantic",
        score=0.9,
    )

    assert result.linked_entity_id == "person-1"
    with pytest.raises(ValidationError):
        EntityLinkingResult(mention=mention, score=1.1)
