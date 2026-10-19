# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The look of the course, from the style guide (course/stijl in the design notes).

Colour means a role, never decoration. Lessons never write a colour code: they use the
eight roles (`c-param`, `c-loss`, ...) or CSS variables (`var(--ea-loss)`), which follow
the light and dark theme on their own. Everything that is not a role is ink and grey.
"""
from __future__ import annotations

import html
import json
import re

from everything_ai import paths

ROLES = ("data", "param", "pred", "residual", "loss", "grad", "hyper", "hidden")


def tokens() -> dict:
    return json.loads((paths.PACKAGE / "tokens.json").read_text(encoding="utf-8"))


# 16 px line icons, stroke = currentColor (style guide, chapter 9)
_ICONS = {
    "empty": '<circle cx="8" cy="8" r="6.2"/>',
    "wrong": '<circle cx="8" cy="8" r="6.2"/><path d="M8 1.8 A6.2 6.2 0 0 1 14.2 8 L8 8 Z" fill="currentColor" stroke="none"/>',
    "almost": '<circle cx="8" cy="8" r="6.2"/><path d="M8 1.8 A6.2 6.2 0 1 1 1.8 8 L8 8 Z" fill="currentColor" stroke="none"/>',
    "correct": '<circle cx="8" cy="8" r="7" fill="currentColor" stroke="none"/><path class="ea-tick" d="M4.8 8.3 7 10.4 11.3 5.8" stroke="var(--ea-page)" stroke-width="1.8"/>',
    "error": '<path d="M5.5 4.5 2 8l3.5 3.5M10.5 4.5 14 8l-3.5 3.5"/>',
    "slow": '<circle cx="8" cy="8.5" r="5.7"/><path d="M8 5.5v3.2l2 1.3M6.3 1.5h3.4"/>',
    "broken": '<path d="M9.8 2.2a3.3 3.3 0 0 0-3.9 4.4L2.2 10.3a1.3 1.3 0 0 0 1.8 1.8l3.7-3.7a3.3 3.3 0 0 0 4.4-3.9l-2 2-1.8-.5-.5-1.8z"/>',
    "note": '<circle cx="8" cy="8" r="6.2"/><path d="M8 7.3v4M8 4.9v.2"/>',
    "tick": '<path d="M3.5 8.4 6.6 11.3 12.5 4.9"/>',
    "box": '<rect x="2.5" y="2.5" width="11" height="11" rx="2"/>',
    "home": '<path d="M2.2 7.4 8 2.6l5.8 4.8M3.8 6.2v7.2h3V10h2.4v3.4h3V6.2"/>',
}

# check state -> CSS modifier from the style guide (theme.css)
_CHECK_CLASS = {"empty": "empty", "wrong": "notyet", "almost": "almost", "correct": "ok",
                "error": "code", "slow": "slow", "broken": "broken"}


def icon(name: str) -> str:
    return ('<svg class="ea-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{_ICONS[name]}</svg>')


def check_box(state: str, word: str, body_html: str) -> str:
    """A check message: an icon and the state word first, never a status colour."""
    return (f'<div class="ea-check ea-check--{_CHECK_CLASS[state]}" role="status">{icon(state)}'
            f'<span class="ea-state">{html.escape(word)}.</span> {body_html}</div>')


def note(body_html: str, title: str | None = None) -> str:
    """A 'watch out' or other aside: an inset panel with an i-icon, no colour."""
    head = f"<b>{html.escape(title)}</b> " if title else ""
    return f'<div class="ea-note">{icon("note")}{head}{body_html}</div>'


def task(label: str, body_html: str) -> str:
    """One short assignment, with the label TASK / OPDRACHT in a small ink block."""
    return f'<div class="ea-task"><span class="ea-label">{html.escape(label)}</span><span>{body_html}</span></div>'


def role(role_name: str, text: str) -> str:
    """A word in the colour of its role, always with weight too (colour is never the only signal)."""
    assert role_name in ROLES, role_name
    return f'<span class="c-{role_name}">{text}</span>'


def formula(tex: str) -> str:
    """A display formula, centred in the text column. A symbol written `[[param:w]]` gets the text tone of its role.
    KaTeX cannot read the CSS colour variables, so the formula is drawn once for the light theme and once for the
    dark one, and the course's CSS shows only the one that fits the theme."""
    themes = tokens()["themes"]
    parts = []
    for theme in ("light", "dark"):
        roles = themes[theme]["role"]

        def colour(m, roles=roles):
            assert m.group(1) in ROLES, m.group(1)
            # The inner braces make a real group, so KaTeX keeps the spacing around a minus inside it.
            return r"\textcolor{%s}{{%s}}" % (roles[m.group(1)]["text"], m.group(2))
        coloured = re.sub(r"\[\[(\w+):(.+?)\]\]", colour, tex)
        parts.append(f'<div class="ea-formula ea-only-{theme}"><marimo-tex class="arithmatex">||[{html.escape(coloured, quote=False)}||]</marimo-tex></div>')
    return "".join(parts)
