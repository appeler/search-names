"""
Tests for enhanced_name_parser module
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from search_names.enhanced_name_parser import (
    NameParser,
    ParsedName,
    compare_parsers,
    parse_names,
)


class TestParsedName(unittest.TestCase):
    """Test ParsedName dataclass."""

    def test_parsed_name_creation(self):
        """Test creating a ParsedName object."""
        parsed = ParsedName(
            original="John Doe",
            first_name="John",
            last_name="Doe",
            model_score=0.95,
            parser_used="humanname",
        )

        self.assertEqual(parsed.original, "John Doe")
        self.assertEqual(parsed.first_name, "John")
        self.assertEqual(parsed.last_name, "Doe")
        self.assertEqual(parsed.model_score, 0.95)

    def test_full_name_method(self):
        """Test full_name method."""
        parsed = ParsedName(
            original="Dr. John M. Doe Jr.",
            title="Dr.",
            first_name="John",
            middle_name="M.",
            last_name="Doe",
            suffix="Jr.",
        )

        self.assertEqual(parsed.full_name(), "Dr. John M. Doe Jr.")

    def test_full_name_partial(self):
        """Test full_name with missing components."""
        parsed = ParsedName(original="John Doe", first_name="John", last_name="Doe")

        self.assertEqual(parsed.full_name(), "John Doe")

    def test_to_dict_method(self):
        """Test to_dict method."""
        parsed = ParsedName(
            original="John Doe", first_name="John", last_name="Doe", model_score=0.9
        )

        result = parsed.to_dict()

        self.assertIsInstance(result, dict)
        self.assertEqual(result["original"], "John Doe")
        self.assertEqual(result["first_name"], "John")
        self.assertEqual(result["last_name"], "Doe")
        self.assertEqual(result["model_score"], 0.9)
        self.assertEqual(result["parser_used"], "humanname")


class TestNameParser(unittest.TestCase):
    """Test NameParser class."""

    def test_name_parser_init_default(self):
        """Test NameParser initialization with defaults."""
        parser = NameParser()

        self.assertEqual(parser.parser_type, "auto")
        self.assertEqual(parser.batch_size, 100)
        self.assertEqual(parser.ml_threshold, 0.8)

    def test_name_parser_parsernaam_available(self):
        """Test parsernaam parser is always available."""
        parser = NameParser(parser_type="parsernaam")

        # parsernaam should be available as main dependency
        self.assertEqual(parser.parser_type, "parsernaam")

    def test_name_parser_rejects_invalid_configuration(self):
        """Reject invalid parser settings at construction time."""
        with self.assertRaises(ValueError):
            NameParser(parser_type="unknown")
        with self.assertRaises(ValueError):
            NameParser(batch_size=0)
        with self.assertRaises(ValueError):
            NameParser(ml_threshold=1.1)

    @patch("search_names.enhanced_name_parser.HumanName")
    def test_parse_with_humanname(self, mock_humanname):
        """Test parsing with HumanName."""
        mock_parsed = MagicMock()
        mock_parsed.first = "John"
        mock_parsed.middle = "M"
        mock_parsed.last = "Doe"
        mock_parsed.title = "Dr"
        mock_parsed.suffix = "Jr"
        mock_parsed.nickname = None
        mock_humanname.return_value = mock_parsed

        parser = NameParser(parser_type="humanname")
        result = parser.parse_with_humanname("Dr John M Doe Jr")

        self.assertIsInstance(result, ParsedName)
        self.assertEqual(result.first_name, "John")
        self.assertEqual(result.middle_name, "M")
        self.assertEqual(result.last_name, "Doe")
        self.assertEqual(result.title, "Dr")
        self.assertEqual(result.suffix, "Jr")
        self.assertEqual(result.parser_used, "humanname")

    @patch("search_names.enhanced_name_parser.HumanName")
    def test_parse_with_humanname_error(self, mock_humanname):
        """Test error handling in humanname parsing."""
        mock_humanname.side_effect = Exception("Parse error")

        parser = NameParser()
        result = parser.parse_with_humanname("Invalid Name")

        self.assertIsNone(result.model_score)
        self.assertEqual(result.parser_used, "humanname")

    def test_parse_with_parsernaam(self):
        """Test parsing with parsernaam using real module."""
        parser = NameParser(parser_type="parsernaam")
        results = parser.parse_with_parsernaam(["John Doe"])

        # Should return parsed results
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], ParsedName)
        self.assertEqual(results[0].parser_used, "parsernaam")
        self.assertEqual(results[0].original, "John Doe")
        self.assertEqual(results[0].first_name, "John")
        self.assertEqual(results[0].last_name, "Doe")
        self.assertGreater(results[0].model_score, 0.8)

    def test_parse_with_parsernaam_batch(self):
        """Test parsernaam with batch processing."""
        parser = NameParser()
        results = parser.parse_with_parsernaam(["John Doe", "Rajesh Kumar"])

        # Should parse both names
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, ParsedName)
            self.assertEqual(result.parser_used, "parsernaam")

    def test_is_indian_name(self):
        """Test Indian name detection."""
        parser = NameParser()

        # Test Indian names
        self.assertTrue(parser.is_indian_name("Rajesh Kumar"))
        self.assertTrue(parser.is_indian_name("Priya Sharma"))
        self.assertTrue(parser.is_indian_name("Venkat Reddy"))
        self.assertTrue(parser.is_indian_name("Anita Patel"))

        # Test non-Indian names
        self.assertFalse(parser.is_indian_name("John Smith"))
        self.assertFalse(parser.is_indian_name("Jane Doe"))
        self.assertFalse(parser.is_indian_name("Robert Johnson"))
        self.assertFalse(parser.is_indian_name("Abraham Lincoln"))

    @patch("search_names.enhanced_name_parser.parse_with_parsernaam_model")
    def test_parsernaam_last_first_schema(self, mock_parse):
        """Map parsernaam's documented last_first record to components."""
        mock_parse.return_value = pd.DataFrame(
            {
                "name": ["Nakamura Hiro"],
                "parsed_name": [
                    {"name": "Nakamura Hiro", "type": "last_first", "prob": 0.97}
                ],
            }
        )

        result = NameParser(parser_type="parsernaam").parse("Nakamura Hiro")

        self.assertEqual(result.first_name, "Hiro")
        self.assertEqual(result.last_name, "Nakamura")
        self.assertEqual(result.model_score, 0.97)

    @patch("search_names.enhanced_name_parser.parse_with_parsernaam_model")
    def test_parsernaam_threshold_falls_back(self, mock_parse):
        """Use deterministic parsing when the model score is below threshold."""
        mock_parse.return_value = pd.DataFrame(
            {
                "name": ["John Doe"],
                "parsed_name": [
                    {"name": "John Doe", "type": "last_first", "prob": 0.51}
                ],
            }
        )

        result = NameParser(parser_type="parsernaam", ml_threshold=0.8).parse(
            "John Doe"
        )

        self.assertEqual(result.parser_used, "humanname")
        self.assertEqual(result.first_name, "John")
        self.assertEqual(result.last_name, "Doe")

    def test_parsernaam_rejects_invalid_probability(self):
        """Reject malformed model probabilities before constructing results."""
        parser = NameParser(parser_type="parsernaam")

        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            parser._parsed_name_from_prediction(
                "John Doe", {"type": "first_last", "prob": float("nan")}
            )

    def test_parse_single_name(self):
        """Test parsing a single name."""
        parser = NameParser(parser_type="humanname")
        result = parser.parse("John Doe")

        self.assertIsInstance(result, ParsedName)
        self.assertEqual(result.original, "John Doe")

    def test_parse_list_of_names(self):
        """Test parsing a list of names."""
        parser = NameParser(parser_type="humanname")
        names = ["John Doe", "Jane Smith"]
        results = parser.parse(names)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], ParsedName)
        self.assertIsInstance(results[1], ParsedName)

    def test_parse_auto_mode(self):
        """Test auto mode with mixed names using real parsernaam."""
        parser = NameParser(parser_type="auto")

        # Test with mixed names
        names = ["John Smith", "Rajesh Kumar", "Jane Doe"]
        results = parser.parse(names)

        self.assertEqual(len(results), 3)
        self.assertIsInstance(results[0], ParsedName)
        self.assertIsInstance(results[1], ParsedName)
        self.assertIsInstance(results[2], ParsedName)

        # John Smith should use humanname
        self.assertEqual(results[0].parser_used, "humanname")
        # Rajesh Kumar should use parsernaam (Indian name detection)
        self.assertEqual(results[1].parser_used, "parsernaam")
        # Jane Doe should use humanname
        self.assertEqual(results[2].parser_used, "humanname")

    def test_parse_dataframe(self):
        """Test parsing names in a DataFrame."""
        df = pd.DataFrame({"name": ["John Doe", "Jane Smith"], "id": [1, 2]})

        parser = NameParser(parser_type="humanname")
        result_df = parser.parse_dataframe(df, name_column="name")

        # Check new columns were added
        self.assertIn("parsed_first_name", result_df.columns)
        self.assertIn("parsed_last_name", result_df.columns)
        self.assertIn("parser_model_score", result_df.columns)
        self.assertIn("parser_used", result_df.columns)

        # Check data integrity
        self.assertEqual(len(result_df), 2)

    def test_parse_dataframe_preserves_input(self):
        """Return a new frame while preserving index and replacing collisions."""
        df = pd.DataFrame(
            {"name": ["John Doe"], "parsed_first_name": ["stale"]},
            index=pd.Index([42], name="row_id"),
        )

        result = NameParser(parser_type="humanname").parse_dataframe(df)

        self.assertEqual(result.index.tolist(), [42])
        self.assertEqual(result.loc[42, "parsed_first_name"], "John")
        self.assertEqual(df.loc[42, "parsed_first_name"], "stale")

    def test_parse_dataframe_invalid_column(self):
        """Test error handling for invalid column."""
        df = pd.DataFrame({"id": [1, 2]})

        parser = NameParser()
        with self.assertRaises(ValueError) as context:
            parser.parse_dataframe(df, name_column="invalid")

        self.assertIn("Column 'invalid' not found", str(context.exception))

    def test_parse_dataframe_no_components(self):
        """Test DataFrame parsing without adding components."""
        df = pd.DataFrame({"name": ["John Doe"], "id": [1]})

        parser = NameParser()
        result_df = parser.parse_dataframe(df, add_components=False)

        # Should add parsed_name column instead
        self.assertIn("parsed_name", result_df.columns)
        self.assertNotIn("parsed_first_name", result_df.columns)

        # Check that parsed_name contains ParsedName objects
        self.assertIsInstance(result_df["parsed_name"].iloc[0], ParsedName)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def test_parse_names_single(self):
        """Test parse_names with single name."""
        result = parse_names("John Doe")

        self.assertIsInstance(result, ParsedName)
        self.assertEqual(result.original, "John Doe")

    def test_parse_names_list(self):
        """Test parse_names with list."""
        results = parse_names(["John Doe", "Jane Smith"])

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)

    def test_parse_names_dataframe(self):
        """Test parse_names with DataFrame."""
        df = pd.DataFrame({"name": ["John Doe"]})
        result_df = parse_names(df)

        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertIn("parsed_first_name", result_df.columns)

    def test_compare_parsers(self):
        """Test compare_parsers function with real parsers."""
        results = compare_parsers("John Doe")

        # All parsers should be available
        self.assertIn("humanname", results)
        self.assertIn("parsernaam", results)
        self.assertIn("auto", results)

        # Each should return a ParsedName object
        for result in results.values():
            self.assertIsInstance(result, ParsedName)
            self.assertEqual(result.original, "John Doe")

    def test_compare_parsers_indian_name(self):
        """Test compare_parsers with an Indian name."""
        results = compare_parsers("Rajesh Kumar")

        # All parsers should be available
        self.assertIn("humanname", results)
        self.assertIn("parsernaam", results)
        self.assertIn("auto", results)

        # Each should return a ParsedName object
        for result in results.values():
            self.assertIsInstance(result, ParsedName)
            self.assertEqual(result.original, "Rajesh Kumar")


class TestIntegration(unittest.TestCase):
    """Integration tests for name parsing."""

    def test_full_parsing_pipeline(self):
        """Test full parsing pipeline with various names."""
        names = [
            "John Doe",
            "Dr. Jane Smith Jr.",
            "Robert James Johnson III",
            "Mary-Jane O'Connor",
        ]

        parser = NameParser(parser_type="humanname")
        results = parser.parse(names)

        self.assertEqual(len(results), len(names))

        # Check each result
        for result, original in zip(results, names, strict=False):
            self.assertEqual(result.original, original)
            self.assertIsNotNone(result.first_name or result.last_name)
            self.assertEqual(result.parser_used, "humanname")

    def test_dataframe_integration(self):
        """Test DataFrame integration."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["John Doe", "Jane Smith", "Bob Johnson"],
                "age": [30, 25, 35],
            }
        )

        # Parse names
        result_df = parse_names(df, parser_type="humanname")

        # Verify structure
        self.assertEqual(len(result_df), 3)
        self.assertIn("parsed_first_name", result_df.columns)
        self.assertIn("parsed_last_name", result_df.columns)

        # Verify parsing
        self.assertIsNotNone(result_df["parsed_first_name"].iloc[0])
        self.assertIsNotNone(result_df["parsed_last_name"].iloc[0])

    def test_error_recovery(self):
        """Test error recovery in parsing."""
        # Include some problematic names
        names = [
            "John Doe",
            "",  # Empty name
            "   ",  # Whitespace only
            "A",  # Single letter
            "Jane Smith",
        ]

        parser = NameParser()
        results = parser.parse(names)

        # Should handle all names without crashing
        self.assertEqual(len(results), len(names))

        # Check that valid names were parsed
        self.assertIsNotNone(results[0].first_name)
        self.assertIsNotNone(results[4].first_name)


if __name__ == "__main__":
    unittest.main()
