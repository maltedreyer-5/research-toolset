# Output languages

The *output language* — the language of reports and exports — is chosen
per run in the interface. The language of the interface itself is set
separately (see [Interface languages](#interface-languages) below); a
page's language is preselected as the output language when it is enabled.

## What follows the output language

- Everything the tool itself writes into a report or export: headings, the
  report header and footer, notes and warning banners, the bibliography check
  report, Word export appendices, the search protocol of the analysis modes,
  date formats. These texts come from catalogs.
- Everything the model writes for the reader: the report, summaries,
  analysis sections, diagnosis messages. Prompts are English; each prompt whose
  output reaches the report ends with an instruction naming the output
  language (the catalog's `llm_name`).

Search queries are not affected: the planner chooses search languages by
topic, and literature queries stay English because the scholarly APIs work
best with English terms.

## Enabling languages

```bash
OUTPUT_LANGUAGES=en,de          # languages offered in the interface
DEFAULT_OUTPUT_LANGUAGE=de      # preselected
```

English is always available. With only one language enabled, the selector is
hidden.

## Catalogs

A catalog is a TOML file named after the language code:

```toml
[meta]
name = "Deutsch"          # shown in the interface
llm_name = "German"       # used in the instruction to the model

[strings]
"footer.title" = "Hinweise zur Erstellung"
"search.n_hits" = "**{n} Treffer**"
# ...
```

- `src/locales/en.toml` is the reference: every key the code uses exists there.
- `src/locales/de.toml` ships with the tool.
- Keys missing from a catalog fall back to English, so a partial catalog is
  usable.
- Placeholders such as `{n}` must match the English entry exactly; a date
  format key (`footer.datetime_format`, `word.date_format`) takes a
  `strftime` pattern.

At start-up every enabled catalog is validated: an unknown language, a
missing catalog or a placeholder mismatch stops the start with a message that
names the key.

## Adding a language

1. Copy `src/locales/en.toml` to `<code>.toml` (e.g. `nl.toml`) in a directory
   of your choice.
2. Set `name` and `llm_name` and translate the strings you need.
3. Point `OUTPUT_CATALOG_DIR` at the directory and add the code to
   `OUTPUT_LANGUAGES`.

No code change is needed. A catalog in `OUTPUT_CATALOG_DIR` takes precedence
over a built-in one with the same code, so you can also adapt the German
wording this way.

## What stays German on purpose

Some German text in the code is data, not interface: German stop words and
titles used to detect languages and names, German headings recognised in
bibliographies, and German spellings the parsers still accept when a model
answers in German.

## Interface languages

The interface text is English in the code (`tr("Start research")`) and
translated from catalogs in `src/ui/locales/` — the English text is the
key, as with gettext:

```toml
[meta]
name = "Deutsch"          # shown in the language switch

[strings]
"🔍 Start research" = "🔍 Recherche starten"
"📥 Source {n}: {title}" = "📥 Quelle {n}: {title}"
```

```bash
UI_LANGUAGES=en,de        # languages offered (default: all catalogs)
DEFAULT_UI_LANGUAGE=de    # language of the start page (default: en)
```

- Every language is its own page: the default language at the root, the
  others under `/<code>` (e.g. `/en`). The switch in the header links them;
  the browser remembers the last choice and the start page forwards to it.
- A missing entry shows the English text. Placeholders must match the
  English key; a mismatch stops the start with a message naming the key.
- Switching the language loads the other page: a chat stored in the browser
  (with `BROWSER_STORAGE_SECRET` set) comes along, a running research stays
  on the page where it was started.
- `tests/test_ui_i18n.py` warns when a text in the code has no German entry
  (it then shows in English) and fails when the catalog holds entries the
  code no longer uses.

The chat assistant's system prompt exists in both languages
(`src/prompts/research.py`); the clickable action phrases are recognised
in either language on every page.

