"""Clickable action phrases in chat answers.

The chat system prompt tells the model to write these phrases; a small
script in the UI turns them into clickable links. Both sides are built
from this table, so the prompt text and the recognition pattern cannot
drift apart. Each action has one primary phrase (used in the prompt)
and aliases the model may produce instead, e.g. when it answers in the
user's language.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


def _js_escape(text: str) -> str:
    """Escape only JavaScript regex metacharacters (valid in any flag mode)."""
    return re.sub(r"([.*+?^${}()|\[\]\\/])", r"\\\1", text)


@dataclass(frozen=True)
class ChatAction:
    action: str          # data-action value handled by the UI script
    emoji: str
    phrase: str          # primary phrase, used in the system prompt
    aliases: tuple[str, ...]
    title: str           # tooltip

    @property
    def label(self) -> str:
        return f"{self.emoji} {self.phrase}"

    def js_regex(self) -> str:
        """JavaScript regex source matching emoji + any known phrase."""
        alts = "|".join(_js_escape(p) for p in (self.phrase, *self.aliases))
        return f"({_js_escape(self.emoji)}\\s*(?:{alts}))"


# Both interface languages are recognised on every page: a chat restored
# from the browser may come from the other language's page.
CHAT_ACTIONS: tuple[ChatAction, ...] = (
    ChatAction("focus", "💬", "Discuss request",
               ("Discuss further", "Auftrag besprechen", "Weiter diskutieren", "Besprechen"),
               "Focus the input field"),
    ChatAction("adopt", "📋", "Adopt suggestion",
               ("Use as request", "Vorschlag übernehmen", "Als Auftrag nutzen", "Übernehmen"),
               "Copy the cleaned-up suggestion into the input field"),
    ChatAction("research", "🔍", "Start research",
               ("Research now", "Recherche starten", "Jetzt recherchieren"),
               "Start the research"),
    ChatAction("litcheck", "📚", "Check references",
               ("Check bibliography", "Check literature", "Literatur prüfen", "Literaturprüfung"),
               "Check a reference list"),
)

ACTIONS_BY_NAME = {a.action: a for a in CHAT_ACTIONS}


def js_replace_rules() -> str:
    """JavaScript statements that wrap every action phrase in a link span.

    Tooltips are in the current interface language (src.ui.i18n).
    """
    from src.ui.i18n import tr
    out = []
    for a in CHAT_ACTIONS:
        out.append(
            "changed = changed.replace(new RegExp(%s, 'g'), "
            "'<span class=\"chat-action\" data-action=\"%s\" "
            "style=\"cursor:pointer;color:var(--primary-500);"
            "text-decoration:underline;font-weight:600\" title=%s>$1</span>');"
            % (json.dumps(a.js_regex(), ensure_ascii=False), a.action,
               json.dumps(tr(a.title), ensure_ascii=False).replace("'", "\\'"))
        )
    return "\n                ".join(out)
