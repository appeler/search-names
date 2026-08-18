"""
Tests for nlp_engine module
"""

import unittest
from unittest.mock import MagicMock, patch

from search_names.models import EntityMention
from search_names.nlp_engine import (
    DEFAULT_SIMILARITY_REVISION,
    EntityLinker,
    NLPEngine,
    NLPEngineError,
    SemanticSimilarity,
    SpacyNER,
)


class TestNLPEngineError(unittest.TestCase):
    """Test NLPEngineError exception."""

    def test_nlp_engine_error(self):
        """Test NLPEngineError exception creation."""
        error = NLPEngineError("Test error message")
        self.assertEqual(str(error), "Test error message")


class TestSpacyNER(unittest.TestCase):
    """Test SpacyNER class with actual spacy."""

    def test_spacy_ner_real_model(self):
        """Test SpacyNER with real model."""
        ner = SpacyNER("en_core_web_sm")
        self.assertEqual(ner.model_name, "en_core_web_sm")
        self.assertIsNotNone(ner.nlp)

        # Test actual entity extraction
        entities = ner.extract_entities(
            "John Doe is meeting Jane Smith in New York.", {"PERSON", "GPE"}
        )

        # Check we got some entities
        self.assertGreater(len(entities), 0)

        # Verify entity structure
        for entity in entities:
            self.assertIsInstance(entity, EntityMention)
            self.assertIn(entity.label, {"PERSON", "GPE"})

    def test_extract_person_entities_real(self):
        """Test person entity extraction with real model."""
        ner = SpacyNER()
        text = "Barack Obama met with Angela Merkel yesterday."
        entities = ner.extract_person_entities(text, min_length=2)

        # Should find at least one person
        self.assertGreater(len(entities), 0)

        # All should be PERSON entities
        for entity in entities:
            self.assertEqual(entity.label, "PERSON")
            self.assertGreater(len(entity.text.strip()), 1)

    def test_is_person_context_real(self):
        """Test person context detection with real examples."""
        ner = SpacyNER()

        # Test with clear person context
        text1 = "Mr. John Smith said that the project was on track."
        self.assertTrue(ner.is_person_context(text1, 4, 14, context_window=50))

        # Test without person context
        text2 = "The company Apple Inc. reported earnings."
        self.assertFalse(ner.is_person_context(text2, 12, 22, context_window=50))

    def test_spacy_ner_invalid_model(self):
        """Test SpacyNER with invalid model name."""
        with self.assertRaises(NLPEngineError) as context:
            SpacyNER("non_existent_model")
        self.assertIn("Could not load spaCy model", str(context.exception))

    def test_extract_entities_error_handling(self):
        """Test error handling during entity extraction."""
        ner = SpacyNER()
        ner.nlp = None  # Simulate model not loaded

        with self.assertRaises(NLPEngineError) as context:
            ner.extract_entities("Test text")

        self.assertIn("spaCy model not loaded", str(context.exception))

    def test_extract_entities_runtime_error_is_not_a_false_empty_result(self):
        """Surface inference failures instead of reporting no entities."""
        ner = SpacyNER()
        ner.nlp = MagicMock(side_effect=RuntimeError("inference failed"))

        with self.assertRaises(NLPEngineError):
            ner.extract_entities("John Doe")


class TestSemanticSimilarity(unittest.TestCase):
    """Test SemanticSimilarity class."""

    def test_semantic_similarity_with_real_model(self):
        """Test with actual sentence transformer model."""
        # Using a small model for testing
        similarity = SemanticSimilarity("sentence-transformers/all-MiniLM-L6-v2")

        self.assertIsNotNone(similarity.model)
        self.assertEqual(similarity.revision, DEFAULT_SIMILARITY_REVISION)

        # Test actual similarity computation
        score1 = similarity.compute_similarity("dog", "puppy")
        score2 = similarity.compute_similarity("dog", "airplane")

        # Dog and puppy should be more similar than dog and airplane
        self.assertGreater(score1, score2)
        self.assertGreater(score1, 0.5)  # Should be reasonably similar
        self.assertLess(score2, 0.5)  # Should be dissimilar

    def test_semantic_similarity_with_local_only(self):
        """Test with local_files_only flag."""
        # After downloading in CI, this should work with cached model
        similarity = SemanticSimilarity(
            "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True
        )
        self.assertIsNotNone(similarity.model)

    def test_find_similar_names_real(self):
        """Test finding similar names with real model."""
        similarity = SemanticSimilarity()

        candidates = ["John Smith", "Jon Smith", "Jane Doe", "Apple Inc"]
        similar = similarity.find_similar_names("John Smith", candidates, threshold=0.7)

        # Should find John Smith as exact match
        self.assertGreater(len(similar), 0)
        self.assertEqual(similar[0][0], "John Smith")
        self.assertAlmostEqual(similar[0][1], 1.0, places=2)

        # Jon Smith should be similar
        names = [name for name, _ in similar]
        self.assertIn("Jon Smith", names)

    def test_compute_similarity_error_handling(self):
        """Test error handling in compute_similarity."""
        similarity = SemanticSimilarity()
        similarity.model = None  # Simulate model not loaded

        with self.assertRaises(NLPEngineError):
            similarity.compute_similarity("text1", "text2")

    def test_semantic_similarity_invalid_model(self):
        """Test initialization with invalid model name."""
        with self.assertRaises(NLPEngineError) as context:
            SemanticSimilarity("non-existent-model", local_files_only=True)
        self.assertIn(
            "Could not load sentence transformer model", str(context.exception)
        )


class TestEntityLinker(unittest.TestCase):
    """Test EntityLinker class."""

    def setUp(self):
        """Set up test data."""
        self.knowledge_base = {
            "John Smith": {
                "id": "person_1",
                "aliases": ["J. Smith", "Johnny Smith"],
                "occupation": "Engineer",
            },
            "Jane Doe": {
                "id": "person_2",
                "aliases": ["J. Doe"],
                "occupation": "Doctor",
            },
        }

    def test_entity_linker_init(self):
        """Test EntityLinker initialization."""
        linker = EntityLinker(self.knowledge_base)

        self.assertEqual(linker.knowledge_base, self.knowledge_base)

        # Check indices were created
        self.assertIn("John Smith", linker.exact_match_index)
        self.assertIn("J. Smith", linker.exact_match_index)
        self.assertIn("john smith", linker.normalized_index)

    def test_normalize_name(self):
        """Test name normalization."""
        linker = EntityLinker(self.knowledge_base)

        normalized = linker._normalize_name("John O'Connor-Smith Jr.")
        self.assertEqual(normalized, "john oconnorsmith jr")

    def test_link_entity_exact_match(self):
        """Test entity linking with exact match."""
        linker = EntityLinker(self.knowledge_base)

        mention = EntityMention(text="John Smith", label="PERSON", start=0, end=10)

        result = linker.link_entity(mention)

        self.assertEqual(result.linked_entity_id, "person_1")
        self.assertEqual(result.linked_entity_name, "John Smith")
        self.assertEqual(result.match_method, "exact")
        self.assertIsNone(result.score)

    def test_link_entity_alias_match(self):
        """Test entity linking with alias match."""
        linker = EntityLinker(self.knowledge_base)

        mention = EntityMention(text="J. Smith", label="PERSON", start=0, end=8)

        result = linker.link_entity(mention)

        self.assertEqual(result.linked_entity_id, "person_1")
        self.assertEqual(result.match_method, "exact")
        self.assertIsNone(result.score)

    def test_link_entity_normalized_match(self):
        """Test entity linking with normalized match."""
        linker = EntityLinker(self.knowledge_base)

        mention = EntityMention(text="JOHN SMITH", label="PERSON", start=0, end=10)

        result = linker.link_entity(mention)

        self.assertEqual(result.linked_entity_id, "person_1")
        self.assertEqual(result.match_method, "normalized")
        self.assertIsNone(result.score)

    def test_link_entity_semantic_match(self):
        """Test entity linking with semantic similarity."""
        semantic_model = SemanticSimilarity()
        linker = EntityLinker(self.knowledge_base, semantic_model)

        mention = EntityMention(
            text="Jon Smith",  # Similar but not exact
            label="PERSON",
            start=0,
            end=9,
        )

        result = linker.link_entity(mention, similarity_threshold=0.7)

        # Should find John Smith as closest match
        self.assertEqual(result.linked_entity_id, "person_1")
        self.assertEqual(result.match_method, "semantic")
        self.assertGreater(result.score, 0.7)

    def test_link_entity_no_match(self):
        """Test entity linking when no match found."""
        linker = EntityLinker(self.knowledge_base)

        mention = EntityMention(text="Unknown Person", label="PERSON", start=0, end=14)

        result = linker.link_entity(mention)

        self.assertIsNone(result.linked_entity_id)
        self.assertIsNone(result.linked_entity_name)
        self.assertIsNone(result.match_method)
        self.assertIsNone(result.score)

    def test_entity_linker_rejects_ambiguous_aliases(self):
        """Reject aliases that normalize to different canonical entities."""
        knowledge_base = {
            "John Smith": {"id": "1", "aliases": ["J. Smith"]},
            "Jane Smith": {"id": "2", "aliases": ["J Smith"]},
        }

        with self.assertRaisesRegex(ValueError, "ambiguous knowledge-base name"):
            EntityLinker(knowledge_base)

    def test_link_entity_validates_similarity_threshold(self):
        """Validate the threshold even when semantic linking is disabled."""
        linker = EntityLinker(self.knowledge_base)
        mention = EntityMention(text="Unknown", label="PERSON", start=0, end=7)

        with self.assertRaisesRegex(ValueError, "between -1 and 1"):
            linker.link_entity(mention, similarity_threshold=2)


class TestNLPEngine(unittest.TestCase):
    """Test main NLPEngine class."""

    def setUp(self):
        """Set up test data."""
        self.sample_kb = {"John Smith": {"id": "1"}, "Jane Doe": {"id": "2"}}

    def test_nlp_engine_with_real_components(self):
        """Test NLP engine with real components."""
        engine = NLPEngine(
            enable_ner=True,
            enable_similarity=True,
            enable_linking=False,
        )

        self.assertIsNotNone(engine.spacy_ner)
        self.assertIsNotNone(engine.semantic_similarity)

        # Test processing
        results = engine.process_text(
            "John Doe is meeting Jane Smith.", extract_entities=True
        )

        self.assertIn("entities", results)
        self.assertIn("person_entities", results)
        self.assertGreater(len(results["entities"]), 0)

    def test_nlp_engine_init_with_errors(self):
        """Surface failure when a requested component cannot initialize."""
        with self.assertRaises(NLPEngineError):
            NLPEngine(spacy_model="invalid_model", enable_ner=True)

    def test_nlp_engine_linking_requires_knowledge_base(self):
        """Reject linking that has no knowledge base."""
        with self.assertRaises(ValueError):
            NLPEngine(enable_ner=False, enable_linking=True)

    def test_process_text_no_ner(self):
        """Surface requests for an NLP component that was not enabled."""
        engine = NLPEngine(enable_ner=False)

        with self.assertRaisesRegex(NLPEngineError, "NER was not enabled"):
            engine.process_text("John Doe is here", extract_entities=True)

    def test_process_text_rejects_unavailable_linking(self):
        """Surface linking requests when the linker was not enabled."""
        engine = NLPEngine(enable_ner=True, enable_linking=False)

        with self.assertRaisesRegex(NLPEngineError, "linking was not enabled"):
            engine.process_text("John Doe is here", link_entities=True)

    def test_enhance_name_search_real(self):
        """Test enhanced name search with real NER."""
        engine = NLPEngine(enable_ner=True, enable_similarity=False)

        text = "John Doe said that Jane Smith was present at the meeting."
        search_names = ["John Doe", "Jane Smith", "Bob Wilson"]

        results = engine.enhance_name_search(search_names, text)

        self.assertEqual(len(results), 3)

        # Check John Doe result
        john_result = next((r for r in results if r["search_name"] == "John Doe"), None)
        self.assertIsNotNone(john_result)
        self.assertEqual(john_result["match_count"], 1)

        # Check Bob Wilson has no matches
        bob_result = next(
            (r for r in results if r["search_name"] == "Bob Wilson"), None
        )
        self.assertIsNotNone(bob_result)
        self.assertEqual(bob_result["match_count"], 0)

    def test_enhance_name_search_rejects_blank_names(self):
        """Avoid the non-advancing empty-string search case."""
        engine = NLPEngine(enable_ner=False)

        with self.assertRaisesRegex(ValueError, "cannot be blank"):
            engine.enhance_name_search([""], "text")


class TestIntegration(unittest.TestCase):
    """Integration tests for NLP components."""

    def test_full_pipeline_real(self):
        """Test full NLP pipeline with real models."""
        knowledge_base = {
            "Barack Obama": {"id": "person_1", "aliases": ["Obama", "President Obama"]},
            "Angela Merkel": {"id": "person_2", "aliases": ["Merkel"]},
        }

        engine = NLPEngine(
            knowledge_base=knowledge_base,
            enable_ner=True,
            enable_similarity=True,
            enable_linking=True,
        )

        text = "Barack Obama met with Angela Merkel to discuss climate change."

        results = engine.process_text(
            text,
            extract_entities=True,
            link_entities=True,
        )

        # Verify results structure
        self.assertIn("entities", results)
        self.assertIn("person_entities", results)
        self.assertIn("linked_entities", results)

        # Should find at least one person entity
        self.assertGreater(len(results["person_entities"]), 0)

        # If linking worked, should have linked entities
        if results["linked_entities"]:
            # Check that at least one entity was successfully linked
            linked = [e for e in results["linked_entities"] if e.linked_entity_id]
            self.assertGreater(len(linked), 0)


# Tests that use mocks for edge cases and error conditions
class TestMockedErrorCases(unittest.TestCase):
    """Test error cases using mocks."""

    @patch("spacy.load")
    def test_spacy_load_error(self, mock_load):
        """Test handling of spacy load errors."""
        mock_load.side_effect = OSError("Model not found")

        with self.assertRaises(NLPEngineError) as context:
            SpacyNER("invalid_model")

        self.assertIn("Could not load spaCy model", str(context.exception))

    @patch("sentence_transformers.SentenceTransformer")
    def test_sentence_transformer_load_error(self, mock_transformer):
        """Test handling of sentence transformer load errors."""
        mock_transformer.side_effect = Exception("Download failed")

        with self.assertRaises(NLPEngineError) as context:
            SemanticSimilarity("invalid_model")

        self.assertIn(
            "Could not load sentence transformer model", str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
