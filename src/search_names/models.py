"""Validated result types used by the NLP components."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class EntityMention(BaseModel):
    """A named-entity span returned by spaCy."""

    text: str = Field(min_length=1)
    label: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_positions(self) -> "EntityMention":
        """Require a nonempty, forward character span."""
        if self.end <= self.start:
            raise ValueError("end position must be greater than start position")
        return self


class EntityLinkingResult(BaseModel):
    """The canonical entity selected for one mention, if any."""

    mention: EntityMention
    linked_entity_id: str | None = None
    linked_entity_name: str | None = None
    match_method: Literal["exact", "normalized", "semantic"] | None = None
    score: float | None = Field(None, ge=-1.0, le=1.0)
    alternative_entities: list[dict[str, Any]] = Field(default_factory=list)
