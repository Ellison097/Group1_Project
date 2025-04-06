import re
import unittest
from unittest.mock import MagicMock, patch  # Import mocking tools

import pandas as pd
import requests
from thefuzz import fuzz  # Required for is_similar logic

# Assuming your source files are in a 'src' directory relative to the project root
# Adjust the import paths if your structure is different
from src.api_integration import get_paper_metadata, is_fsrdc_related, reconstruct_abstract
from src.graph_analysis import ResearchGraphBuilder, safe_eval


# --- Replicated Logic for Testing ---
# Replicating _clean_text logic from DataProcessor for easier testing
def clean_text_standalone(text: str) -> str:
    """Clean text data (standalone version for testing)"""
    if pd.isna(text):
        return ""
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters
    text = re.sub(r"[^\w\s]", " ", text)
    # Remove extra spaces
    text = " ".join(text.split())
    return text


# Replicating is_similar logic from process_api_data for easier testing
def is_similar_standalone(title1, title2, threshold=80):
    """Compare two titles for similarity (standalone version for testing)"""
    if pd.isna(title1) or pd.isna(title2):
        return False
    # Use fuzz.ratio for similarity comparison
    return fuzz.ratio(str(title1).lower(), str(title2).lower()) >= threshold


# Replicating count_keywords logic from process_api_data
def count_keywords_standalone(keywords_str):
    """Calculate keyword count (standalone version for testing)"""
    if pd.isna(keywords_str):
        return 0
    # Split by comma and space, filter empty strings
    keywords = [kw for kw in str(keywords_str).split(", ") if kw]
    return len(keywords)


# Replicating _extract_year logic from DataProcessor
def extract_year_standalone(date_str: str) -> str:
    """Extract year from date string (standalone version for testing)"""
    if pd.isna(date_str):
        return ""
    # Ensure it's a string for consistent processing
    date_str = str(date_str)
    if not date_str:
        return ""

    try:
        # Try direct numeric conversion first (handles "2020", 2020.0, etc.)
        numeric_year = pd.to_numeric(date_str, errors="coerce")
        if pd.notna(numeric_year) and 1000 <= numeric_year <= 9999:
            return str(int(numeric_year))

        # If not directly numeric, try parsing as datetime
        return str(pd.to_datetime(date_str).year)
    except:
        # If standard parsing fails, try regex to find a 4-digit year
        match = re.search(r"\b(19[89]\d|20\d\d)\b", date_str)  # Look for 1980-2099
        if match:
            return match.group(1)
        # Fallback if nothing works
        return ""


# --- End Replicated Logic ---


class TestUtilityFunctions(unittest.TestCase):
    def test_reconstruct_abstract(self):
        """Test abstract reconstruction from inverted index."""
        inverted_index_1 = {"This": [0], "is": [1], "a": [2], "test": [3]}
        expected_1 = "This is a test"
        self.assertEqual(reconstruct_abstract(inverted_index_1), expected_1)

        inverted_index_2 = {"complex": [1], "example": [2], "A": [0], "more": [3]}
        expected_2 = "A complex example more"
        self.assertEqual(reconstruct_abstract(inverted_index_2), expected_2)

        inverted_index_3 = {}
        expected_3 = "No abstract available."
        self.assertEqual(reconstruct_abstract(inverted_index_3), expected_3)

        inverted_index_4 = None
        expected_4 = "No abstract available."
        self.assertEqual(reconstruct_abstract(inverted_index_4), expected_4)

    def test_safe_eval(self):
        """Test safe evaluation of string lists."""
        self.assertEqual(safe_eval("['apple', 'banana']"), ["apple", "banana"])
        self.assertEqual(safe_eval('["item1", "item2"]'), ["item1", "item2"])
        self.assertEqual(safe_eval("['mixed', 123]"), ["mixed", 123])  # Note: literal_eval handles this
        self.assertEqual(safe_eval("single item"), ["single item"])  # Falls back to splitting
        self.assertEqual(safe_eval("item1, item2, item3"), ["item1", "item2", "item3"])  # Falls back to splitting
        self.assertEqual(safe_eval(""), [])
        self.assertEqual(safe_eval(None), [])
        self.assertEqual(safe_eval(pd.NA), [])
        self.assertEqual(safe_eval("[malformed"), ["[malformed"])  # Falls back to splitting
        self.assertEqual(safe_eval(["already", "list"]), ["already", "list"])  # Handles existing lists
        self.assertEqual(safe_eval(123), [])  # Handles non-string/list types

    def test_clean_text_standalone(self):
        """Test standalone text cleaning function."""
        self.assertEqual(clean_text_standalone("  Test String! "), "test string")
        self.assertEqual(clean_text_standalone("Another Example."), "another example")
        self.assertEqual(clean_text_standalone("Case TEST"), "case test")
        self.assertEqual(clean_text_standalone("With\nNewline"), "with newline")
        self.assertEqual(clean_text_standalone("Multiple   Spaces"), "multiple spaces")
        self.assertEqual(clean_text_standalone(None), "")
        self.assertEqual(clean_text_standalone(""), "")

    def test_is_similar_standalone(self):
        """Test standalone title similarity function."""
        self.assertTrue(is_similar_standalone("Research Output Analysis", "Research Output Analysis", threshold=80))
        self.assertTrue(is_similar_standalone("Research Output Analysis", "research output analysis", threshold=80))
        self.assertTrue(
            is_similar_standalone("FSRDC Research", "FSRDC Research!", threshold=80)
        )  # Ignores punctuation due to lowercasing
        self.assertFalse(is_similar_standalone("Completely Different", "Title Here", threshold=80))
        self.assertTrue(
            is_similar_standalone("Analyzing FSRDC Data", "Analysis of FSRDC Data", threshold=75)
        )  # Adjust threshold for test
        self.assertFalse(is_similar_standalone("Analyzing FSRDC Data", "Analysis of FSRDC Data", threshold=90))
        self.assertFalse(is_similar_standalone(None, "Title", threshold=80))
        self.assertFalse(is_similar_standalone("Title", None, threshold=80))

    def test_normalize_institution_name(self):
        """Test institution name normalization."""
        # Need an instance of ResearchGraphBuilder. We provide a dummy path.
        # This assumes __init__ doesn't immediately fail if the file isn't readable,
        # and that _normalize_institution_name doesn't depend on self.data.
        try:
            # Provide a dummy path that likely doesn't exist or is empty
            builder = ResearchGraphBuilder(data_path="dummy_path_for_testing.csv")
        except Exception:
            # If instantiation fails completely, skip this test
            self.skipTest("Could not instantiate ResearchGraphBuilder for testing _normalize_institution_name")
            return

        self.assertEqual(
            builder._normalize_institution_name("University of Pennsylvania"), "university of pennsylvania"
        )
        self.assertEqual(
            builder._normalize_institution_name("Univ. of Penn"), "university of penn"
        )  # Basic replacement
        self.assertEqual(
            builder._normalize_institution_name("  National Bureau, Econ Res. "), "national bureau econ research"
        )
        self.assertEqual(builder._normalize_institution_name("Some Institute Inc."), "some institute")
        self.assertEqual(
            builder._normalize_institution_name("Tech Corp"), "tech corporation"
        )  # Corp suffix removal needs adjustment in original code
        self.assertEqual(builder._normalize_institution_name("Med School"), "medical school")
        self.assertEqual(builder._normalize_institution_name(None), "")
        self.assertEqual(builder._normalize_institution_name(""), "")

    def test_is_fsrdc_related(self):
        """Test the FSRDC keyword matching logic."""
        keywords = ["fsrdc", "census bureau", "restricted data"]
        paper_data_match_title = {"title": "Analysis using FSRDC data"}
        paper_data_match_abstract = {"abstract": "We used data from the Census Bureau."}
        paper_data_match_inst = {"author_institutions": [["University", "Census Bureau RDC"]]}
        paper_data_match_affil = {"raw_affiliation_strings": [["Affiliation with restricted data access"]]}
        paper_data_match_keyword = {"keywords": "economics, restricted data, policy"}
        paper_data_no_match = {"title": "Other research", "abstract": "Public data analysis"}
        paper_data_empty = {}
        paper_data_none = None

        self.assertTrue(is_fsrdc_related(paper_data_match_title, keywords))
        self.assertTrue(is_fsrdc_related(paper_data_match_abstract, keywords))
        self.assertTrue(is_fsrdc_related(paper_data_match_inst, keywords))
        self.assertTrue(is_fsrdc_related(paper_data_match_affil, keywords))
        self.assertTrue(is_fsrdc_related(paper_data_match_keyword, keywords))
        self.assertFalse(is_fsrdc_related(paper_data_no_match, keywords))
        self.assertFalse(is_fsrdc_related(paper_data_empty, keywords))
        self.assertFalse(is_fsrdc_related(paper_data_none, keywords))

    def test_count_keywords_standalone(self):
        """Test the standalone keyword counting logic."""
        self.assertEqual(count_keywords_standalone("kw1, kw2, kw3"), 3)
        self.assertEqual(count_keywords_standalone("kw1"), 1)
        self.assertEqual(count_keywords_standalone(""), 0)
        self.assertEqual(count_keywords_standalone(None), 0)
        self.assertEqual(count_keywords_standalone("kw1, "), 1)  # Handles trailing comma/space

    def test_extract_year_standalone(self):
        """Test the standalone year extraction logic."""
        self.assertEqual(extract_year_standalone("2023-01-15"), "2023")
        self.assertEqual(extract_year_standalone("2021"), "2021")
        self.assertEqual(extract_year_standalone(2020.0), "2020")
        self.assertEqual(extract_year_standalone(1999), "1999")
        self.assertEqual(extract_year_standalone("Published Jun 2019"), "2019")
        self.assertEqual(extract_year_standalone("Invalid Date"), "")
        self.assertEqual(extract_year_standalone(None), "")
        self.assertEqual(extract_year_standalone(""), "")
        self.assertEqual(extract_year_standalone(99), "")  # Not a 4-digit year

    @patch("src.api_integration.requests.get")  # Mock 'requests.get' within the api_integration module
    def test_get_paper_metadata_success(self, mock_get):
        """Test get_paper_metadata with mocked successful API response."""
        # Configure the mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None  # Simulate successful HTTP status
        mock_response.json.return_value = {  # Simulate the JSON data returned by OpenAlex
            "results": [
                {
                    "title": "Mock Paper Title",
                    "publication_year": 2023,
                    "doi": "10.1234/mock.doi",
                    "cited_by_count": 10,
                    "datasets": ["dataset1"],
                    "concepts": [{"display_name": "Keyword1"}, {"display_name": "Keyword2"}],
                    "authorships": [
                        {
                            "author": {"display_name": "Author One"},
                            "raw_author_name": "One, Author",
                            "institutions": [{"display_name": "Inst A"}],
                            "raw_affiliation_strings": ["Inst A Raw"],
                            "affiliations": [{"raw_affiliation_string": "Inst A Detailed"}],
                        }
                    ],
                    "abstract_inverted_index": {"Mock": [0], "Abstract": [1]},
                    "cited_by_api_url": "http://example.com/cited_by",  # Need this for the second API call
                }
            ]
        }
        # Second mock response for the cited_by_api_url call
        mock_cited_by_response = MagicMock()
        mock_cited_by_response.raise_for_status.return_value = None
        mock_cited_by_response.json.return_value = {"results": []}  # Simulate no citing papers found

        # Set the side_effect to return different mocks for consecutive calls
        mock_get.side_effect = [mock_response, mock_cited_by_response]

        # Call the function under test
        result = get_paper_metadata("Any Title Query", sleep_time=0)  # Use sleep_time=0 for tests

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result["title"], "Mock Paper Title")
        self.assertEqual(result["year"], 2023)
        self.assertEqual(result["doi"], "10.1234/mock.doi")
        self.assertEqual(result["keywords"], "Keyword1, Keyword2")
        self.assertEqual(result["abstract"], "Mock Abstract")
        self.assertEqual(result["author_names"], ["Author One"])
        self.assertEqual(result["unique_institutions"], ["Inst A"])
        self.assertEqual(result["citing_works"], [])  # Check the result of the second API call

        # Check that requests.get was called twice (once for search, once for cited_by)
        self.assertEqual(mock_get.call_count, 2)
        # Check the URL of the first call
        mock_get.assert_any_call(
            "https://api.openalex.org/works?search=Any%20Title%20Query", headers=unittest.mock.ANY
        )
        # Check the URL of the second call
        mock_get.assert_any_call("http://example.com/cited_by", headers=unittest.mock.ANY)

    @patch("src.api_integration.requests.get")
    def test_get_paper_metadata_no_results(self, mock_get):
        """Test get_paper_metadata when API returns no results."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"results": []}  # Simulate empty results
        mock_get.return_value = mock_response

        result = get_paper_metadata("NonExistent Title", sleep_time=0)

        self.assertIsNone(result)
        mock_get.assert_called_once()  # Only one call needed if no results

    @patch("src.api_integration.requests.get")
    def test_get_paper_metadata_api_error(self, mock_get):
        """Test get_paper_metadata when API call fails."""
        mock_response = MagicMock()
        # Simulate an HTTP error (e.g., 404 Not Found)
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error")
        mock_get.return_value = mock_response

        result = get_paper_metadata("Error Title", sleep_time=0)

        self.assertIsNone(result)
        mock_get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
