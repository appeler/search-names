"""Search and resolve personal names in text corpora."""

from importlib.metadata import PackageNotFoundError, version

from . import engines, models, nlp_engine
from .logging_config import get_logger, setup_logging
from .merge_results import merge_results
from .pipeline import clean_names, search_names
from .pipeline import preprocess_names as preprocess
from .pipeline.step2_augment import augment_names
from .split_text_corpus import split_text_corpus

try:
    __version__ = version("search-names")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "augment_names",
    "clean_names",
    "engines",
    "get_logger",
    "merge_results",
    "models",
    "nlp_engine",
    "preprocess",
    "search_names",
    "setup_logging",
    "split_text_corpus",
]
