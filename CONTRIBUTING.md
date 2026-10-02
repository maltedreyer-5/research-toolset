# Contributing

Thank you for considering a contribution. Bug reports, fixes, new output
language catalogs and documentation improvements are all welcome.

## Before you start

- For anything larger than a small fix, please open an issue first, so that we
  can agree on the approach before you invest time.
- Security problems: please follow [SECURITY.md](SECURITY.md) instead of
  opening a public issue.

## Development setup

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
ruff check .
python -m pytest -q
```

The test suite needs no network, no LLM and no SearXNG: external services are
replaced by test doubles. Both commands also run in CI for every pull request.

## Guidelines

- **Language.** Code, comments, docstrings, commit messages and documentation
  are English. Text that ends up in reports belongs in the output catalogs
  (`src/locales/en.toml`, plus `de.toml` if you can); see
  [output languages](docs/output-languages.md). Interface text is English in
  the code and wrapped in `tr()`, with the German translation in
  `src/ui/locales/de.toml` if you can. `tests/test_ui_i18n.py` warns about
  texts without a translation (they show in English; the maintainers add
  them) and fails for entries the code no longer uses. See
  [interface languages](docs/output-languages.md#interface-languages).
- **Prompts** are English. A prompt whose output reaches the report must end
  with `prompt_language_line(...)` so that the model writes in the selected
  output language.
- **Decisions that depend on meaning** go to a classifier with a typed result,
  a confidence and a conservative fallback (`src/pipeline/classifiers/`), not
  to a keyword heuristic.
- **Tests.** New behaviour needs a test. Several test doubles answer prompts
  by keyword and fall back to a neutral default when nothing matches — a test
  can then pass without exercising the path it is meant to test. Check that
  yours does (for example by asserting on the calls a double received), or
  let your double fail on unmatched prompts.
- **No real personal data** in tests, examples or documentation. Use invented
  names and `example.org` / `example.edu`.
- **Dependencies.** Keep new runtime dependencies to a minimum and state their
  licence in the pull request; they must be compatible with the MIT licence.

## Adding an output language

Copy `src/locales/en.toml`, translate it, and open a pull request with the new
catalog. Placeholders (`{n}`, `{name}`, …) must stay unchanged; the start-up
validation reports mismatches.

## Pull requests

- Keep a pull request focused on one change.
- Describe what changes for users, and add an entry under *Unreleased* in
  [CHANGELOG.md](CHANGELOG.md) if it is user-visible.
- Make sure `ruff check .` and `python -m pytest -q` pass.

By contributing you agree that your contribution is licensed under the
[MIT licence](LICENSE) of this project.
