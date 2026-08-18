"""Optional spaCy recognition and Sentence Transformer name linking."""

import re
from typing import Any

from .logging_config import get_logger
from .models import EntityLinkingResult, EntityMention

logger = get_logger("nlp_engine")

DEFAULT_SIMILARITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class NLPEngineError(Exception):
    """Exception raised for NLP engine related errors."""


class SpacyNER:
    """spaCy-based Named Entity Recognition for person detection."""

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        disable_components: list[str] | None = None,
    ):
        """Initialize spaCy NER.

        Args:
            model_name: spaCy model to load
            disable_components: Pipeline components to disable for speed
        """
        self.model_name = model_name
        self.nlp = None

        # Default components to disable for speed (keep only NER)
        if disable_components is None:
            disable_components = [
                "tok2vec",
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
            ]

        self.disable_components = disable_components
        self._load_model()

    def _load_model(self):
        """Load spaCy model with error handling."""
        try:
            # Lazy import of spacy
            import spacy

            self.nlp = spacy.load(self.model_name, disable=self.disable_components)
            logger.info("Loaded spaCy model: %s", self.model_name)
        except OSError as e:
            error_msg = (
                f"Could not load spaCy model '{self.model_name}'. "
                f"Install with: python -m spacy download {self.model_name}"
            )
            logger.error(error_msg)
            raise NLPEngineError(error_msg) from e

    def extract_entities(
        self, text: str, entity_types: set[str] | None = None
    ) -> list[EntityMention]:
        """Extract named entities from text.

        Args:
            text: Input text
            entity_types: Set of entity types to extract (default: {"PERSON"})

        Returns:
            List of entity mentions

        Raises:
            NLPEngineError: If the spaCy model is unavailable or processing fails.
        """
        if entity_types is None:
            entity_types = {"PERSON"}

        if not self.nlp:
            raise NLPEngineError("spaCy model not loaded")

        try:
            doc = self.nlp(text)
            entities = []

            for ent in doc.ents:
                if ent.label_ in entity_types:
                    entity = EntityMention(
                        text=ent.text,
                        label=ent.label_,
                        start=ent.start_char,
                        end=ent.end_char,
                    )
                    entities.append(entity)

            return entities

        except Exception as error:
            raise NLPEngineError("Could not extract entities") from error

    def extract_person_entities(
        self, text: str, min_length: int = 2
    ) -> list[EntityMention]:
        """Extract person entities specifically.

        Args:
            text: Input text
            min_length: Minimum length of person names to extract

        Returns:
            List of person entity mentions

        Raises:
            ValueError: If ``min_length`` is less than one.
        """
        if min_length < 1:
            raise ValueError("min_length must be positive")
        entities = self.extract_entities(text, {"PERSON"})

        # Filter by minimum length and clean up
        filtered_entities = []
        for entity in entities:
            if len(entity.text.strip()) >= min_length:
                # Clean up the text
                cleaned_text = entity.text.strip()
                if cleaned_text:
                    entity.text = cleaned_text
                    filtered_entities.append(entity)

        return filtered_entities

    def is_person_context(
        self, text: str, start: int, end: int, context_window: int = 50
    ) -> bool:
        """Check if a mention appears in a person context.

        Args:
            text: Full text
            start: Start position of mention
            end: End position of mention
            context_window: Characters before/after to check

        Returns:
            True if mention appears in person context
        """
        # Extract context around the mention
        context_start = max(0, start - context_window)
        context_end = min(len(text), end + context_window)
        context = text[context_start:context_end].lower()

        # Person indicators
        person_indicators = {
            # Titles
            "mr.",
            "mrs.",
            "ms.",
            "dr.",
            "prof.",
            "professor",
            "senator",
            "congressman",
            "congresswoman",
            "president",
            "vice president",
            "governor",
            "mayor",
            "judge",
            # Actions/verbs
            "said",
            "told",
            "stated",
            "announced",
            "declared",
            "argued",
            "claimed",
            "according to",
            "spokesperson",
            "representative",
            # Relationships
            "son of",
            "daughter of",
            "wife of",
            "husband of",
            "father",
            "mother",
            "brother",
            "sister",
            "colleague",
            "friend",
            "partner",
        }

        return any(indicator in context for indicator in person_indicators)


class SemanticSimilarity:
    """Semantic similarity matching using sentence transformers."""

    def __init__(
        self,
        model_name: str = DEFAULT_SIMILARITY_MODEL,
        revision: str | None = None,
        local_files_only: bool = False,
    ):
        """Initialize semantic similarity model.

        Args:
            model_name: Sentence transformer model name
            revision: Immutable Hugging Face model revision. The bundled default is pinned.
            local_files_only: If True, only use locally cached models
        """
        self.model_name = model_name
        self.revision = revision
        if revision is None and model_name == DEFAULT_SIMILARITY_MODEL:
            self.revision = DEFAULT_SIMILARITY_REVISION
        self.local_files_only = local_files_only
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load sentence transformer model."""
        try:
            # Lazy import of sentence_transformers
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                self.model_name,
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
            logger.info("Loaded sentence transformer model: %s", self.model_name)
        except Exception as e:
            error_msg = f"Could not load sentence transformer model '{self.model_name}'"
            logger.error(error_msg)
            raise NLPEngineError(error_msg) from e

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score between -1 and 1.

        Raises:
            NLPEngineError: If the model is unavailable or encoding fails.
        """
        if not self.model:
            raise NLPEngineError("Sentence transformer model not loaded")

        try:
            # Lazy import of numpy
            import numpy as np

            embeddings = self.model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as error:
            raise NLPEngineError("Could not compute semantic similarity") from error

    def find_similar_names(
        self, target_name: str, candidate_names: list[str], threshold: float = 0.8
    ) -> list[tuple[str, float]]:
        """Find names similar to target name.

        Args:
            target_name: Name to match against
            candidate_names: List of candidate names
            threshold: Minimum similarity threshold

        Returns:
            List of (name, similarity_score) tuples above threshold

        Raises:
            NLPEngineError: If the sentence-transformer model is unavailable.
            ValueError: If ``threshold`` is outside the valid cosine range.
        """
        if not self.model:
            raise NLPEngineError("Sentence transformer model not loaded")
        if not -1 <= threshold <= 1:
            raise ValueError("threshold must be between -1 and 1")

        similar_names = []

        for candidate in candidate_names:
            similarity = self.compute_similarity(target_name, candidate)
            if similarity >= threshold:
                similar_names.append((candidate, similarity))

        # Sort by similarity score (descending)
        similar_names.sort(key=lambda x: x[1], reverse=True)
        return similar_names


class EntityLinker:
    """Simple entity linking using string matching and semantic similarity."""

    def __init__(
        self,
        knowledge_base: dict[str, dict[str, Any]],
        semantic_model: SemanticSimilarity | None = None,
    ):
        """Initialize entity linker.

        Args:
            knowledge_base: Dict mapping canonical names to entity info
            semantic_model: Optional semantic similarity model
        """
        self.knowledge_base = knowledge_base
        self.semantic_model = semantic_model

        # Create lookup indices
        self._create_indices()

    def _create_indices(self):
        """Create lookup indices for fast matching."""
        self.exact_match_index: dict[str, str] = {}
        self.normalized_index: dict[str, str] = {}

        for canonical_name, entity_info in self.knowledge_base.items():
            if not isinstance(canonical_name, str) or not canonical_name.strip():
                raise ValueError("knowledge-base names must be nonempty strings")
            if not isinstance(entity_info, dict):
                raise TypeError("knowledge-base entries must be dictionaries")
            self._register_name(canonical_name, canonical_name)

            aliases = entity_info.get("aliases", [])
            if not isinstance(aliases, list) or not all(
                isinstance(alias, str) and alias.strip() for alias in aliases
            ):
                raise TypeError("knowledge-base aliases must be nonempty strings")
            for alias in aliases:
                self._register_name(alias, canonical_name)

    def _register_name(self, name: str, canonical_name: str) -> None:
        """Register an exact and normalized name without ambiguous collisions."""
        normalized = self._normalize_name(name)
        for index, key in (
            (self.exact_match_index, name),
            (self.normalized_index, normalized),
        ):
            existing = index.get(key)
            if existing is not None and existing != canonical_name:
                raise ValueError(f"ambiguous knowledge-base name: {name!r}")
            index[key] = canonical_name

    def _normalize_name(self, name: str) -> str:
        """Normalize name for matching."""
        normalized = re.sub(r"[^\w\s]", "", name.casefold())
        return " ".join(normalized.split())

    def _entity_id(self, canonical_name: str) -> str:
        """Return the declared entity identifier or the canonical name."""
        return str(self.knowledge_base[canonical_name].get("id", canonical_name))

    def link_entity(
        self, mention: EntityMention, similarity_threshold: float = 0.8
    ) -> EntityLinkingResult:
        """Link entity mention to knowledge base.

        Args:
            mention: Entity mention to link
            similarity_threshold: Threshold for semantic similarity matching

        Returns:
            Entity linking result

        Raises:
            ValueError: If ``similarity_threshold`` is outside the cosine range.
        """
        if not -1 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between -1 and 1")
        mention_text = mention.text

        # Try exact match first
        if mention_text in self.exact_match_index:
            canonical_name = self.exact_match_index[mention_text]
            return EntityLinkingResult(
                mention=mention,
                linked_entity_id=self._entity_id(canonical_name),
                linked_entity_name=canonical_name,
                match_method="exact",
                score=None,
                alternative_entities=[],
            )

        # Try normalized match
        normalized_mention = self._normalize_name(mention_text)
        if normalized_mention in self.normalized_index:
            canonical_name = self.normalized_index[normalized_mention]
            return EntityLinkingResult(
                mention=mention,
                linked_entity_id=self._entity_id(canonical_name),
                linked_entity_name=canonical_name,
                match_method="normalized",
                score=None,
                alternative_entities=[],
            )

        # Try semantic similarity if available
        if self.semantic_model:
            candidates = list(self.knowledge_base.keys())
            similar_names = self.semantic_model.find_similar_names(
                mention_text, candidates, similarity_threshold
            )

            if similar_names:
                best_match, score = similar_names[0]
                alternatives = [
                    {"entity_name": name, "score": alternative_score}
                    for name, alternative_score in similar_names[1:6]
                ]

                return EntityLinkingResult(
                    mention=mention,
                    linked_entity_id=self._entity_id(best_match),
                    linked_entity_name=best_match,
                    match_method="semantic",
                    score=score,
                    alternative_entities=alternatives,
                )

        # No match found
        return EntityLinkingResult(
            mention=mention,
            linked_entity_id=None,
            linked_entity_name=None,
            match_method=None,
            score=None,
            alternative_entities=[],
        )


class NLPEngine:
    """Main NLP engine combining all components."""

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        similarity_model: str = DEFAULT_SIMILARITY_MODEL,
        similarity_revision: str | None = None,
        knowledge_base: dict[str, dict[str, Any]] | None = None,
        enable_ner: bool = True,
        enable_similarity: bool = False,
        enable_linking: bool = False,
    ):
        """Initialize NLP engine.

        Args:
            spacy_model: spaCy model name
            similarity_model: Sentence transformer model name
            similarity_revision: Immutable Hugging Face model revision
            knowledge_base: Entity knowledge base for linking
            enable_ner: Enable named entity recognition
            enable_similarity: Enable semantic similarity
            enable_linking: Enable entity linking

        Raises:
            ValueError: If entity linking is enabled without a knowledge base.
        """
        self.spacy_ner = None
        self.semantic_similarity = None
        self.entity_linker = None

        if enable_ner:
            self.spacy_ner = SpacyNER(spacy_model)

        if enable_similarity:
            self.semantic_similarity = SemanticSimilarity(
                similarity_model, revision=similarity_revision
            )

        if enable_linking:
            if not knowledge_base:
                raise ValueError(
                    "knowledge_base is required when entity linking is enabled"
                )
            self.entity_linker = EntityLinker(knowledge_base, self.semantic_similarity)

    def process_text(
        self,
        text: str,
        extract_entities: bool = True,
        link_entities: bool = False,
        min_name_length: int = 2,
        similarity_threshold: float = 0.8,
    ) -> dict[str, Any]:
        """Process text with all available NLP components.

        Args:
            text: Input text
            extract_entities: Whether to extract entities
            link_entities: Whether to link entities
            min_name_length: Minimum name length for entity extraction
            similarity_threshold: Threshold for entity linking

        Returns:
            Dictionary with processing results

        Raises:
            NLPEngineError: If a requested component was not enabled.
            ValueError: If linking is requested without entity extraction.
        """
        if extract_entities and self.spacy_ner is None:
            raise NLPEngineError("NER was not enabled")
        if link_entities and not extract_entities:
            raise ValueError("entity linking requires entity extraction")
        if link_entities and self.entity_linker is None:
            raise NLPEngineError("entity linking was not enabled")

        results = {
            "text": text,
            "entities": [],
            "linked_entities": [],
            "person_entities": [],
        }

        # Extract entities
        if extract_entities and self.spacy_ner:
            entities = self.spacy_ner.extract_person_entities(text, min_name_length)
            results["entities"] = entities

            # Filter to just person entities
            person_entities = [e for e in entities if e.label == "PERSON"]
            results["person_entities"] = person_entities

            # Link entities if requested
            if link_entities and self.entity_linker:
                linked_entities = []
                for entity in person_entities:
                    linking_result = self.entity_linker.link_entity(
                        entity, similarity_threshold
                    )
                    linked_entities.append(linking_result)

                results["linked_entities"] = linked_entities

        return results

    def enhance_name_search(
        self,
        search_names: list[str],
        text_corpus: str,
        context_window: int = 100,
    ) -> list[dict[str, Any]]:
        """Enhance name search with NLP context analysis.

        Args:
            search_names: Names to search for
            text_corpus: Text to search in
            context_window: Context window around matches

        Returns:
            Enhanced search results with context analysis

        Raises:
            ValueError: If a name is blank or ``context_window`` is negative.
        """
        if context_window < 0:
            raise ValueError("context_window cannot be negative")
        if any(not search_name.strip() for search_name in search_names):
            raise ValueError("search names cannot be blank")
        results = []

        for search_name in search_names:
            matches = []

            # Simple string search (this could be enhanced with fuzzy matching)
            start_pos = 0
            while True:
                pos = text_corpus.lower().find(search_name.lower(), start_pos)
                if pos == -1:
                    break

                end_pos = pos + len(search_name)

                # Extract context
                context_start = max(0, pos - context_window)
                context_end = min(len(text_corpus), end_pos + context_window)
                context = text_corpus[context_start:context_end]

                # Check if it's a person context using spaCy
                is_person_context = False
                if self.spacy_ner:
                    is_person_context = self.spacy_ner.is_person_context(
                        text_corpus, pos, end_pos, context_window
                    )

                match = {
                    "text": search_name,
                    "start": pos,
                    "end": end_pos,
                    "context": context,
                    "is_person_context": is_person_context,
                }

                matches.append(match)
                start_pos = end_pos

            result = {
                "search_name": search_name,
                "matches": matches,
                "match_count": len(matches),
                "person_matches": len([m for m in matches if m["is_person_context"]]),
            }

            results.append(result)

        return results
