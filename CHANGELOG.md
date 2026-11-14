# Changelog

## [Unreleased]

## [0.7.0] - 2026-11-13

### Changed

- Requires Python 3.12 or later.
- Requires `nameparser` 3.0.1 or later and uses its `parse()` API.
  `nameparser` 3.0 removed the `HumanName` class the parser was built on.
- Renamed the rule-based parser from `humanname` to `nameparser`: pass
  `parser_type="nameparser"`, call `NameParser.parse_with_nameparser`, and
  expect `parser_used == "nameparser"` and a `nameparser` key from
  `compare_parsers`.
- A few names parse differently under `nameparser` 3. Given names that are
  also titles or credentials are now read as given names: `HUTCHINSON, ASA`
  gives first name `ASA`, where it was read as a suffix, and
  `Royal Samuel Copeland` gives first name `ROYAL`, where it was read as a
  title. Some words are now surname particles: `Jasper Delos Ward` gives last
  name `Delos Ward`.

### Fixed

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
