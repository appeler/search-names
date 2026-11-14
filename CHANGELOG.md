# Changelog

## [Unreleased]

## [0.7.0] - 2026-11-13

### Added

- `clean_names` takes `parser_type` (`--parser` on the command line) to choose
  what splits first and last name: `nameparser` (the default), `parsernaam`,
  or `auto`.
- `NameParser`, `ParsedName`, and `parse_names` are importable from
  `search_names`. Importing the package does not load `parsernaam` or torch;
  they load on the first model parse.

### Changed

- Requires Python 3.12 or later.
- Requires `nameparser` 3.0.1 or later and uses its `parse()` API.
  `nameparser` 3.0 removed the `HumanName` class the parser was built on.
- Renamed the rule-based parser from `humanname` to `nameparser`: pass
  `parser_type="nameparser"`, call `NameParser.parse_with_nameparser`, and
  expect `parser_used == "nameparser"` and a `nameparser` key from
  `compare_parsers`.
- Some names parse differently under `nameparser` 3; its migration guide lists
  every change under "Changed in 3.0". The ones most likely to matter here:
  - Given names that are also titles or credentials are read as given names.
    `HUTCHINSON, ASA` gives first name `ASA`, where it was read as a suffix,
    and `Royal Samuel Copeland` gives first name `ROYAL`, where it was a title.
  - Credentials after a name are suffixes. `Jane Doe MSW` gives last name
    `Doe`, where it gave `MSW`.
  - `Delos`, `Sta.`, `Sto.`, and `Graf` join the surname. `Jasper Delos Ward`
    and `WARD, JASPER DELOS` give last name `Delos Ward`, which is wrong when
    the word is a middle name.

### Fixed

- `parser_type="parsernaam"` no longer treats titles, suffixes, and commas as
  name words. `Dr. Juan Q. Xavier de la Vega III` gave first name `III` and
  last name `Dr.`; it now gives `Juan`, `de la Vega`, title `Dr.`, and suffix
  `III`. `parsernaam` is given only the name words and decides their order.
- A fresh install failed on import once `nameparser` 3.0 was released.

## [0.6.0] - 2026-08-18

### Changed

- Consolidated exact, fuzzy, serial, and parallel corpus search behind one
  deterministic API.
- Corrected `parsernaam` result handling and pinned the default Hugging Face
  model revision.
- Removed unused configuration, job-model, alternate-engine, and demo APIs.
- Adopted the shared `py-canon` CI, documentation, and release workflows.

Release notes are generated from merged pull requests and published with each
[GitHub release](https://github.com/appeler/search-names/releases).
