# Changelog

## [Unreleased]

## [0.7.0] - 2026-11-13

### Changed

- Requires Python 3.12 or later.

### Fixed

- Capped `nameparser` below 3.0. That release removed the `HumanName` class
  the name parser is built on, so a fresh install of 0.6.0 failed on import.

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
