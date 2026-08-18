"""Four-stage workflow for name processing.

This module provides a systematic 4-step pipeline for processing names:

Step 1: Clean - Parse and standardize raw names
Step 2: Augment - Add supplementary data (nicknames, prefixes)
Step 3: Preprocess - Create search patterns and deduplicate
Step 4: Search - Execute deterministic exact or fuzzy search

Each step builds on the previous one, creating a complete name processing workflow.
"""

from .step1_clean import clean_names
from .step2_augment import augment_names
from .step3_preprocess import preprocess_names
from .step4_search import search_names

__all__ = [
    "augment_names",
    "clean_names",
    "preprocess_names",
    "search_names",
]
