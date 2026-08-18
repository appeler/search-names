# search-names

[![CI](https://github.com/appeler/search-names/actions/workflows/ci.yml/badge.svg)](https://github.com/appeler/search-names/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/search-names.svg)](https://pypi.org/project/search-names/)
[![Python](https://img.shields.io/pypi/pyversions/search-names.svg)](https://pypi.org/project/search-names/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://appeler.github.io/search-names/)

`search-names` prepares lists of personal names and searches them in CSV text
corpora. Its four-stage workflow cleans names, adds aliases and titles, creates
search patterns, and runs deterministic exact or fuzzy matching.

Name matching is approximate record linkage, not identity verification. Common
names, aliases, OCR errors, and incomplete source data can create false matches
or missed matches. Review matches in context; do not use them alone for
consequential decisions or to infer protected or sensitive traits.

## Installation

```bash
pip install search-names
```

The installation includes the name parsers, spaCy interface, Sentence
Transformers interface, and CLI.

## Search a corpus

The search input is a CSV or `.csv.gz` file. The names are `(identifier, name)`
pairs. Output contains one row per input row in the same order, fixed match
slots, and a `count` column.

```python
from search_names import search_names

stats = search_names(
    "articles.csv",
    [("person-1", "Jane Doe"), ("person-2", "John Smith")],
    "matches.csv",
    text_column="article_text",
    input_columns=("article_id", "article_text"),
    max_results=10,
    processes=4,
)

print(stats)
```

Fuzzy rules are `(minimum name length, maximum edit distance)` pairs. Later
rules take precedence for longer names:

```python
stats = search_names(
    "ocr_articles.csv.gz",
    [("person-1", "Jane Doe"), ("person-2", "John Smith")],
    "matches.csv",
    fuzzy_rules=[(8, 1), (15, 2)],
    processes=4,
)
```

To load a pattern table produced by the preprocessing stage:

```python
from search_names.pipeline.step4_search import load_names_file

names = load_names_file(
    "preprocessed_names.csv",
    id_column="uniqid",
    name_column="search_name",
)
```

CSV and compressed CSV are supported import/export boundaries. The package does
not ship schema-less CSV runtime assets or learned model weights in its wheel.

## Prepare names

```python
from search_names import augment_names, clean_names, preprocess

clean_names(
    input_file="raw_names.csv",
    output_file="clean_names.csv",
    name_column="Name",
    keep_duplicates=False,
)

augment_names(
    input_file="clean_names.csv",
    prefix_column="seat",
    name_column="FirstName",
    output_file="augmented_names.csv",
    prefix_file="prefixes.csv",
    nickname_file="nick_names.txt",
)

preprocess(
    input_file="augmented_names.csv",
    patterns=["FirstName LastName", "NickName LastName", "Prefix LastName"],
    output_file="preprocessed_names.csv",
    edit_length_thresholds=[10, 15],
    drop_patterns=["ambiguous pattern"],
)
```

The preprocessing stage’s edit-length thresholds are retained in its output.
The search API uses the explicit `fuzzy_rules` pairs shown above.

## Parse names

`NameParser` combines `nameparser` with `parsernaam`. The latter returns a model
label (`first`, `last`, `first_last`, or `last_first`) and probability;
`search-names` converts that record to explicit components. Probabilities below
`ml_threshold` fall back to deterministic `nameparser` parsing. Deterministic
parses have `model_score=None`; the package does not invent a confidence value.

```python
from search_names.enhanced_name_parser import NameParser

parser = NameParser(parser_type="parsernaam", ml_threshold=0.8)
parsed = parser.parse("Nakamura Hiro")

print(parsed.first_name, parsed.last_name, parsed.model_score)
```

DataFrame parsing returns a copy, preserves the input index, and replaces any
colliding parsed-output columns deliberately:

```python
result = parser.parse_dataframe(frame, name_column="full_name")
```

## NLP components

The NLP module exposes spaCy NER, semantic similarity, and entity linking:

```python
from search_names.nlp_engine import NLPEngine

engine = NLPEngine(
    knowledge_base={"Jane Doe": {"aliases": ["J. Doe"]}},
    enable_ner=True,
    enable_similarity=True,
    enable_linking=True,
)
result = engine.process_text("Jane Doe spoke today.", link_entities=True)
```

Entity-linking results identify the lookup method (`exact`, `normalized`, or
`semantic`). Only semantic matches have a numeric score, which is cosine
similarity rather than a calibrated probability.

The default Sentence Transformer is downloaded from an immutable 40-character
Hugging Face revision. The Hugging Face client automatically honors its normal
authentication settings, including `HF_TOKEN`; public downloads do not require
a token. The spaCy English model must be installed separately for NER:

```bash
python -m spacy download en_core_web_sm
```

## Command line

```bash
search-names --help
search-names clean raw_names.csv --output clean_names.csv
search-names preprocess augmented_names.csv --output preprocessed_names.csv
search-names search articles.csv --names preprocessed_names.csv \
  --output matches.csv --text-column text --processes 4
search-names search ocr_articles.csv --names preprocessed_names.csv \
  --fuzzy-rule 8:1 --fuzzy-rule 15:2
```

## Development

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv build
```

The [API documentation](https://appeler.github.io/search-names/) is generated
from this README and the package docstrings, so usage and reference material do
not drift into separate hand-maintained copies.

## License

MIT. See the repository's
[LICENSE](https://github.com/appeler/search-names/blob/master/LICENSE).
