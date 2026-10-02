"""Pre-fill the mandatory fields of an analysis mode from the chat history.

Motivation: the analysis modes require structured inputs (decision
question, options, criteria, ...). Whoever discussed the matter in the
chat beforehand has already formulated all of this and would only have
to rearrange it into form fields. That is what this module does.

Deliberate decisions:

1. **A suggestion, not automation.** The values land in the fields and
   are NOT submitted. The person reads them, corrects them and starts
   the run themselves. The existing preflight validator remains the
   authority that decides "complete".

2. **Better empty than invented.** A hallucinated "budget: 50,000 €" in
   a decision context would be worse than an empty field, because it
   would slip into the report unnoticed. The prompt insists on this, and
   `sanitize_suggestions` discards what does not fit the frame.

3. **Gradio-free.** Only logic lives here: build the prompt, clean up
   the LLM answer. The module is thus testable without the UI; wiring
   the buttons stays in `gradio_app.py`.

4. **Harvest model.** This is mechanical rearranging, not analysis —
   it belongs on the fast model, not the reasoning model.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from src.ui.i18n import tr

logger = logging.getLogger(__name__)

# Maximum number of characters of chat history that go into the prompt.
# The history is the only source — generous, but capped so that a
# long chat does not blow up the call.
MAX_CONVERSATION_CHARS = 24_000

# From how many characters of conversation a suggestion makes sense at
# all. Below that, the model would inevitably guess.
MIN_CONVERSATION_CHARS = 80


AUTOFILL_PROMPT = """\
You help to pre-fill a form from a conversation that has already taken \
place.

Below is a conversation between a person and a \
research assistant. The fields of a form are to be \
filled from it.

STRICT RULES:
- Use ONLY information that actually appears in the conversation. \
Invent nothing — no figures, budgets, deadlines, names \
or conditions that are not there.
- If a field cannot be supported from the conversation, return an \
empty string "" for it. An empty field is explicitly correct \
and better than a guess.
- Be brief and write in the language of the conversation.
- For fields of type "choice" you MUST return exactly one of the allowed \
values (unchanged spelling) or "".
- For fields of type "textarea" with a line format: one item per \
line, no bullet characters.

FORM FIELDS:
{fields}

CONVERSATION:
{conversation}

Answer ONLY with a JSON object that uses exactly the field keys above. \
No explanations, no code blocks.
Example: {{"field_a": "value from the conversation", "field_b": ""}}
"""


def format_requirements(requirements: list) -> str:
    """Describe the form fields for the prompt.

    Uses the `Requirement` attributes from analysis_pipeline (field,
    label, kind, choices, placeholder, required).
    """
    lines: list[str] = []
    for req in requirements:
        field = getattr(req, "field", "")
        if not field:
            continue
        label = getattr(req, "label", field)
        kind = getattr(req, "kind", None) or getattr(req, "type", "text")
        required = getattr(req, "required", True)

        line = f'- "{field}" — {label} (Typ: {kind}'
        line += ", mandatory)" if required else ", optional)"

        choices = getattr(req, "choices", None) or getattr(req, "options", None)
        if choices:
            line += "\n  Allowed values (exactly one of them): " + " | ".join(
                str(c[1] if isinstance(c, (tuple, list)) else c) for c in choices
            )

        hint = getattr(req, "description", "") or getattr(req, "placeholder", "")
        if hint:
            line += f"\n  Hint: {str(hint).strip()}"

        max_len = getattr(req, "max_length", None)
        if max_len:
            line += f"\n  At most {max_len} characters."
        lines.append(line)
    return "\n".join(lines)


def format_conversation(chat_history: list, extra_text: str = "") -> str:
    """Build the conversation text for the prompt.

    Takes only roles and text content; the most recent end of the
    history matters most, so on overflow it is shortened at the front.
    """
    parts: list[str] = []
    for msg in chat_history or []:
        if not isinstance(msg, dict):
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        role = msg.get("role", "user")
        who = "Person" if role == "user" else "Assistant"
        parts.append(f"{who}: {content}")

    extra = (extra_text or "").strip()
    if extra:
        parts.append(f"Person: {extra}")

    text = "\n\n".join(parts)
    if len(text) > MAX_CONVERSATION_CHARS:
        # Shorten at the front — the end of the conversation is more relevant
        text = "[... earlier history shortened ...]\n\n" + text[
            -MAX_CONVERSATION_CHARS:
        ]
    return text


def build_autofill_prompt(requirements: list, conversation: str) -> str:
    """Assemble the prompt from the field description and the history."""
    return AUTOFILL_PROMPT.format(
        fields=format_requirements(requirements),
        conversation=conversation,
    )


def sanitize_suggestions(raw: Any, requirements: list) -> dict[str, str]:
    """Turn the LLM answer into safe field values.

    Everything that does not belong in the form is discarded:
      - unknown keys (the model likes to invent some)
      - choice values that are not in the allowed list
      - non-strings (lists/dicts are not guessed; for line fields they
        are joined into lines)
      - excess length (hard cap)

    The result only contains fields with a non-empty suggestion — empty
    fields are not touched at all, so that an autofill never overwrites
    something the person has already typed.
    """
    if not isinstance(raw, dict):
        return {}

    by_field = {
        getattr(r, "field", ""): r for r in requirements
        if getattr(r, "field", "")
    }
    out: dict[str, str] = {}

    for field, req in by_field.items():
        value = raw.get(field)
        if value is None:
            continue

        # Turn lists (occur with line fields) into lines
        if isinstance(value, (list, tuple)):
            value = "\n".join(str(v).strip() for v in value if str(v).strip())
        elif not isinstance(value, str):
            value = str(value)

        value = value.strip()
        if not value:
            continue

        choices = getattr(req, "choices", None) or getattr(req, "options", None)
        if choices:
            match = _match_choice(value, choices)
            if match is None:
                logger.debug(
                    "Autofill: value %r for field %r not among the choices — discarded",
                    value, field,
                )
                continue
            value = match

        max_len = getattr(req, "max_length", None)
        if max_len and len(value) > max_len:
            value = value[:max_len].rstrip()

        out[field] = value

    return out


def _match_choice(value: str, choices) -> Optional[str]:
    """Map a value (or its label, also as shown in the interface) to an
    allowed stored value."""
    pairs = [(c[0], c[1]) if isinstance(c, (tuple, list)) else (c, c)
             for c in choices]
    for label, val in pairs:
        if value in (str(val), str(label), tr(str(label))):
            return str(val)
    low = value.casefold().strip()
    for label, val in pairs:
        if low in (str(val).casefold().strip(), str(label).casefold().strip(),
                   tr(str(label)).casefold().strip()):
            return str(val)
    return None


def empty_required_fields(requirements: list, current_values: list) -> list[str]:
    """Which mandatory fields are (still) empty?

    `current_values` is in the order of `requirements` — the way Gradio
    delivers the component values. Missing positions count as empty.

    Beware of None: `str(None)` would be "None" and thus wrongly
    "filled in" — which is why the `or ""` must be inside `str()`.
    """
    out: list[str] = []
    for i, req in enumerate(requirements):
        if not getattr(req, "required", True):
            continue
        value = current_values[i] if i < len(current_values) else ""
        if not str(value or "").strip():
            out.append(getattr(req, "field", ""))
    return [f for f in out if f]


def plan_form_autofill(
    requirements: list, current_values: list, suggestions: dict,
) -> dict[int, str]:
    """Determine which form index gets which value.

    Fills only empty mandatory fields. What the person entered
    themselves stays untouched — a suggestion must never overwrite a
    deliberate input.

    Returns:
        {index_in_requirements: value}. Empty if there is nothing to do.
    """
    empty = set(empty_required_fields(requirements, current_values))
    if not empty:
        return {}
    out: dict[int, str] = {}
    for i, req in enumerate(requirements):
        field = getattr(req, "field", "")
        if field in empty and suggestions.get(field):
            out[i] = suggestions[field]
    return out


async def suggest_preflight_values(
    checker,
    chat_history: list,
    dual_llm,
    extra_text: str = "",
) -> tuple[dict[str, str], str]:
    """Suggest field values from the chat history.

    Returns:
        (values, message). `values` is a dict {field: value} and only
        contains fields that are supported by the conversation.
        `message` is a short text for the UI — empty if all went well.
    """
    requirements = checker.get_requirements()
    conversation = format_conversation(chat_history, extra_text)

    if len(conversation) < MIN_CONVERSATION_CHARS:
        return {}, tr(
            "Not enough conversation for a suggestion — "
            "describe your request in the chat first."
        )

    prompt = build_autofill_prompt(requirements, conversation)

    try:
        raw = await dual_llm.harvest_complete(
            [{"role": "user", "content": prompt}], max_tokens=2048,
        )
    except Exception as e:
        logger.warning("Autofill call failed: %s", type(e).__name__)
        return {}, tr("Suggestion failed ({error}).", error=type(e).__name__)

    parsed = _parse_json(raw)
    if parsed is None:
        logger.warning("Autofill: answer was not JSON (%d characters)",
                       len(raw or ""))
        return {}, tr("The suggestion could not be evaluated.")

    values = sanitize_suggestions(parsed, requirements)
    if not values:
        return {}, (
            tr("No field could be derived reliably from the conversation.")
        )

    logger.info("Autofill: %d of %d fields suggested",
                len(values), len(requirements))
    return values, ""


def _parse_json(text: str) -> Optional[dict]:
    """Extract a JSON object from the answer (also from a code block)."""
    if not text:
        return None
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1] if "```" in clean[3:] else clean[3:]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip()
    try:
        result = json.loads(clean)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = clean.find("{"), clean.rfind("}")
    if start >= 0 and end > start:
        try:
            result = json.loads(clean[start:end + 1])
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            return None
    return None
