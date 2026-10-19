# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson lm-base-vs-chat: as_text, which writes a conversation out as one text that ends with the
marker of the assistant, and next_turn, which adds an answer and a new question to a conversation. And the course's
small base model of the plays, which counts which character follows the last 7 characters in Tiny Shakespeare, with
the picture of a conversation written out as one text."""
import html
import random
import re
from collections import Counter
from functools import cache

from everything_ai import paths
from everything_ai.check import FunctionBox, Mistake

DATA = "data/tiny-shakespeare.txt"   # about 1.1 MB of plays by Shakespeare: see data/README.md
LOOK = 7                             # the model looks at the last 7 characters of the text so far
MARKER = {"system": "SYSTEM:", "user": "USER:", "assistant": "ASSISTANT:"}
GUESS_TEXT = "USER:\nWhat is a token?\n\nASSISTANT:\n"   # the text of the guess
EXAMPLE = [{"role": "system", "content": "Answer in one line."}, {"role": "user", "content": "Who is Romeo?"}]   # the worked example


@cache
def text() -> str:
    return (paths.ROOT / DATA).read_text(encoding="utf-8")


@cache
def _all() -> Counter:
    return Counter(text())


@cache
def _counts(context: str) -> Counter:
    """How often each character follows `context` in the plays."""
    t, out, start = text(), Counter(), 0
    while True:
        i = t.find(context, start)
        if i < 0 or i + len(context) >= len(t):
            return out
        out[t[i + len(context)]] += 1
        start = i + 1


def next_counts(s: str) -> Counter:
    """What follows the last 7 characters of s in the plays; when those never occur, the last 6, and so on."""
    for k in range(min(LOOK, len(s)), -1, -1):
        c = _counts(s[len(s) - k:]) if k else _all()
        if c:
            return c
    return _all()


@cache
def write_turn(s: str, seed: int = 1, limit: int = 300) -> str:
    """The course's base model continues s, one character at a time, up to the empty line that ends a speech in the
    plays (or `limit` characters). The same seed gives the same text."""
    rng, out = random.Random(seed), ""
    while len(out) < limit and not out.endswith("\n\n"):
        c = next_counts(s + out)
        options = sorted(c)
        out = out + rng.choices(options, weights=[c[ch] for ch in options])[0]
    return out.strip()


def quoted(s: str) -> str:
    """A text the model wrote, as a quote in Markdown: every line its own line, with nothing read as Markdown."""
    return "\n".join("> " + html.escape(line, quote=False).replace("*", "&#42;").replace("_", "&#95;") + "  " for line in s.split("\n"))


def as_text_course(messages) -> str:
    out = ""
    for m in messages:
        out = out + MARKER[m["role"]] + "\n" + m["content"] + "\n\n"
    return out + MARKER["assistant"] + "\n"


# --- the picture: a conversation written out as one text, and the turn the model writes ---------------------------------

def turn_picture(messages, reply: str, labels: dict) -> str:
    """The text a model gets: each message as its marker and its content, the assistant's marker last, and then the
    model's own turn in the colour of what the model gives. On the right, which role each block has."""
    W, LH, X2 = 470, 21, 270
    rows = []   # (text, kind, side label)
    for m in messages:
        rows.append((MARKER[m["role"]], "marker", labels[m["role"]]))
        for line in m["content"].split("\n"):
            rows.append((line, "content", ""))
        rows.append(("", "content", ""))
    rows.append((MARKER["assistant"], "marker", labels["assistant"]))
    for line in reply.split("\n"):
        rows.append((line, "reply", ""))
    rows.append(("", "end", labels["end"]))
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    y0 = 44
    p.append(f'<rect x="0" y="{y0 - 17}" width="{X2 - 14}" height="{len(rows) * LH + 10}" rx="6" style="fill:var(--ea-inset)"/>')
    for i, (line, kind, side) in enumerate(rows):
        y = y0 + i * LH
        style = {"marker": "font-weight:700;fill:var(--ea-ink)", "content": "fill:var(--ea-ink)",
                 "reply": "fill:var(--ea-pred-text)", "end": "fill:var(--ea-pred-text)"}[kind]
        shown = line if kind != "end" else "↵"
        p.append(f'<text x="10" y="{y}" style="font-size:14px;font-family:var(--ea-font-code);{style}">'
                 f'{html.escape(shown)}</text>')
        if side:
            p.append(f'<text x="{X2}" y="{y}" style="font-size:13px;fill:var(--ea-ink-2)">{html.escape(side)}</text>')
    H = y0 + len(rows) * LH
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


# The lesson is done when these are correct. "base-to-chat" and "role-of" are the choice questions in the lesson.
REQUIRED = ["base-to-chat", "role-of", "as-text", "next-turn"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


_HELLO = [_msg("user", "Hello")]
_SHOP = [_msg("system", "Answer in Dutch."), _msg("user", "When do you open?")]
_LONG = [_msg("system", "Be brief."), _msg("user", "Hi"), _msg("assistant", "Hello! How can I help?"),
         _msg("user", "Name a colour.")]
_LINES = [_msg("user", "Two lines:\nthe first and the second")]

# --- exercise 2: as_text ------------------------------------------------------------------------------------------------

_T_HEAD = ('MARKER = {"system": "SYSTEM:", "user": "USER:", "assistant": "ASSISTANT:"}\n\n\n'
           "def as_text(messages):\n    text = \"\"\n    for m in messages:\n")
_T_END = '    return text + MARKER["assistant"] + "\\n"\n'
_T_START = _T_HEAD + "        ...\n" + _T_END


def _t(inner: str = '        text = text + MARKER[m["role"]] + "\\n" + m["content"] + "\\n\\n"\n') -> str:
    return _T_HEAD + inner + _T_END


# --- exercise 3: next_turn ----------------------------------------------------------------------------------------------

_N_HEAD = "def next_turn(messages, reply, question):\n    new = []\n    for m in messages:\n        new.append(m)\n"
_N_END = "    return new\n"
_N_START = _N_HEAD + "    ...\n" + _N_END


def _n(inner: str = '    new.append({"role": "assistant", "content": reply})\n'
                    '    new.append({"role": "user", "content": question})\n') -> str:
    return _N_HEAD + inner + _N_END


def _has_text_message(got, text) -> bool:
    return isinstance(got, list) and any(isinstance(x, str) for x in got)


EXERCISES = {
    "as-text": FunctionBox(
        name="as-text",
        function="as_text",
        start=_T_START,
        solution=_t(),
        visible=[(_HELLO,)],
        hidden=[(_SHOP,), (_LONG,), (_LINES,), ([],)],
        first=[_DOTS],
        mistakes=[
            Mistake("role-key", error=lambda e: isinstance(e, KeyError) and e.args[:1] == ("role",),
                    en="`MARKER` has no key `\"role\"`: its keys are the roles themselves, `\"system\"`, `\"user\"` and `\"assistant\"`. The role of the message `m` is `m[\"role\"]`: look that up in `MARKER`.",
                    nl="`MARKER` heeft geen sleutel `\"role\"`: de sleutels zijn de rollen zelf, `\"system\"`, `\"user\"` en `\"assistant\"`. De rol van het bericht `m` is `m[\"role\"]`: zoek die op in `MARKER`."),
            Mistake("whole-message", error=lambda e: isinstance(e, TypeError) and "unhashable" in str(e),
                    en="You look something up in a dictionary with a key that cannot be a key, such as a whole message. A message is itself a dictionary; `MARKER` needs only its role, `m[\"role\"]`.",
                    nl="Je zoekt iets op in een dictionary met een sleutel die geen sleutel kan zijn, zoals een heel bericht. Een bericht is zelf een dictionary; `MARKER` heeft alleen de rol ervan nodig, `m[\"role\"]`."),
            Mistake("role-not-marker", wrong=_t('        text = text + m["role"] + ":\\n" + m["content"] + "\\n\\n"\n'),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you write the role as it is in the message. The marker of a role is in `MARKER`, such as `MARKER[\"user\"]`, which is `\"USER:\"`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je schrijft de rol zoals die in het bericht staat. De markering van een rol staat in `MARKER`, zoals `MARKER[\"user\"]`, en dat is `\"USER:\"`."),
            Mistake("no-empty-line", wrong=_t('        text = text + MARKER[m["role"]] + "\\n" + m["content"] + "\\n"\n'),
                    en="`{call}` gives `{got}`, and it should give `{want}`: after each message comes an empty line, as after a speech in the plays. That takes two `\\n`: one to end the line, one for the empty line.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: na elk bericht komt een lege regel, zoals na elke beurt in de toneelstukken. Daar zijn twee `\\n` voor nodig: één om de regel af te sluiten, één voor de lege regel."),
            Mistake("same-line", wrong=_t('        text = text + MARKER[m["role"]] + m["content"] + "\\n\\n"\n'),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the marker and the content stick together. The marker stands on a line of its own: put a `\\n` after it.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de markering en de inhoud plakken aan elkaar. De markering staat op een eigen regel: zet er een `\\n` achter."),
            Mistake("no-marker", wrong=_t('        text = text + m["content"] + "\\n\\n"\n'),
                    en="`{call}` gives `{got}`, and it should give `{want}`: there is no marker before the messages, so the model cannot tell who said what. Start each message with the marker of its role, on a line of its own.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: er staat geen markering voor de berichten, dus het model kan niet zien wie wat zei. Begin elk bericht met de markering van zijn rol, op een eigen regel."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with one line that adds the message `m` to `text`: its marker, the content, and the line breaks.",
                      "nl": "Vervang de `...` door één regel die het bericht `m` aan `text` toevoegt: de markering, de inhoud en de regeleinden."},
            "arguments": {"en": "The check calls `{call}`, with one list of messages, and the `def` line of your function has another number of names between its parentheses. Keep `(messages)`.",
                          "nl": "De check roept `{call}` aan, met één lijst berichten, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(messages)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare the two piece by piece: does each message have its marker on a line of its own, then its content, then an empty line?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk de twee stukje voor stukje: heeft elk bericht zijn markering op een eigen regel, dan de inhoud, dan een lege regel?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every role, and for a conversation of any length?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke rol, en voor een gesprek van elke lengte?"},
            "correct": {"en": "Your function is right in all {n} cases: every message with its marker, and the assistant's marker last, so that what comes next is the answer.",
                        "nl": "Je functie klopt in alle {n} gevallen: elk bericht met zijn markering, en de markering van `assistant` als laatste, zodat wat volgt het antwoord is."},
        },
    ),
    "next-turn": FunctionBox(
        name="next-turn",
        function="next_turn",
        start=_N_START,
        solution=_n(),
        visible=[(_HELLO, "Good day to you.", "Who are you?")],
        hidden=[(_SHOP, "We open at nine.", "And on Sunday?"), (_LONG, "Blue.", "Another one?"), ([], "Hi!", "Hello?")],
        first=[_DOTS],
        mistakes=[
            Mistake("plain-text", matches=_has_text_message,
                    en="`{call}` gives `{got}`, and it should give `{want}`: there is a plain text in your list. Every message is a dictionary with a `\"role\"` and a `\"content\"`, as in `messages`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: er staat een losse tekst in je lijst. Elk bericht is een dictionary met een `\"role\"` en een `\"content\"`, zoals in `messages`."),
            Mistake("swapped", wrong=_n('    new.append({"role": "user", "content": reply})\n'
                                        '    new.append({"role": "assistant", "content": question})\n'),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the roles are the wrong way round. The reply was written by the model, and the question comes from the person.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de rollen zijn omgedraaid. Het antwoord schreef het model, en de vraag komt van de persoon."),
            Mistake("reply-as-user", wrong=_n('    new.append({"role": "user", "content": reply})\n'
                                              '    new.append({"role": "user", "content": question})\n'),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the reply is a `user` message in your list. The model wrote it, so its role is `assistant`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: het antwoord is in je lijst een `user`-bericht. Het model schreef het, dus de rol is `assistant`."),
            Mistake("no-reply", wrong=_n('    new.append({"role": "user", "content": question})\n'),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the reply is missing. The model keeps nothing between two turns: without its own answer in the list, it no longer knows what it said.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: het antwoord ontbreekt. Het model onthoudt niets tussen twee beurten: zonder zijn eigen antwoord in de lijst weet het niet meer wat het zei."),
            Mistake("no-question", wrong=_n('    new.append({"role": "assistant", "content": reply})\n'),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the new question is missing. After the reply, add the question as a message too.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de nieuwe vraag ontbreekt. Voeg na het antwoord ook de vraag als bericht toe."),
            Mistake("nothing-added", wrong=_n('    messages.append({"role": "assistant", "content": reply})\n'
                                              '    messages.append({"role": "user", "content": question})\n'),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function gives back only the conversation so far. Add the reply and the question to `new`, the list it gives back.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je functie geeft alleen het gesprek tot nu toe terug. Voeg het antwoord en de vraag toe aan `new`, de lijst die hij teruggeeft."),
            Mistake("append-two", error=lambda e: isinstance(e, TypeError) and "append()" in str(e),
                    en="`append` adds one item at a time. Call it twice: once with the reply as a message, once with the question.",
                    nl="`append` voegt één ding tegelijk toe. Roep het twee keer aan: één keer met het antwoord als bericht, één keer met de vraag."),
            Mistake("append-gives-none", error=lambda e: isinstance(e, AttributeError) and "NoneType" in str(e) and "append" in str(e),
                    en="`new` is `None` here. `append` changes the list itself and gives nothing back, so `new = new.append(...)` leaves `new` with nothing. Write `new.append(...)` on its own, without `new =`.",
                    nl="`new` is hier `None`. `append` verandert de lijst zelf en geeft niets terug, dus na `new = new.append(...)` is `new` niets meer. Schrijf `new.append(...)` los, zonder `new =`."),
            Mistake("order", wrong=_n('    new.append({"role": "user", "content": question})\n'
                                      '    new.append({"role": "assistant", "content": reply})\n'),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the question comes before the reply in your list. The reply was given first; the new question comes after it.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: in je lijst staat de vraag voor het antwoord. Het antwoord kwam eerst; de nieuwe vraag komt erna."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with two lines: add the reply and then the question to `new`, each as a message.",
                      "nl": "Vervang de `...` door twee regels: voeg het antwoord en daarna de vraag aan `new` toe, elk als bericht."},
            "arguments": {"en": "The check calls `{call}`, with a list of messages, a reply and a question, and the `def` line of your function has another number of names between its parentheses. Keep `(messages, reply, question)`.",
                          "nl": "De check roept `{call}` aan, met een lijst berichten, een antwoord en een vraag, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(messages, reply, question)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Look at the last two messages: which role does each have, and which text?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Kijk naar de laatste twee berichten: welke rol heeft elk, en welke tekst?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any conversation, reply and question?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elk gesprek, elk antwoord en elke vraag?"},
            "correct": {"en": "Your function is right in all {n} cases: the conversation so far, the model's reply as an `assistant` message, and the new question as a `user` message.",
                        "nl": "Je functie klopt in alle {n} gevallen: het gesprek tot nu toe, het antwoord van het model als `assistant`-bericht, en de nieuwe vraag als `user`-bericht."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("as-text", _T_START, "empty", None),
    ("as-text", _t(), "correct", None),
    ("as-text", _t('        text = text + MARKER[m["role"]] + "\\n"\n        text = text + m["content"] + "\\n\\n"\n'), "correct", None),
    ("as-text", _t('        text = text + m["role"] + ":\\n" + m["content"] + "\\n\\n"\n'), "wrong", "role-not-marker"),
    ("as-text", _t('        text = text + MARKER[m["role"]] + "\\n" + m["content"] + "\\n"\n'), "wrong", "no-empty-line"),
    ("as-text", _t('        text = text + MARKER[m["role"]] + m["content"] + "\\n\\n"\n'), "wrong", "same-line"),
    ("as-text", _t('        text = text + m["content"] + "\\n\\n"\n'), "wrong", "no-marker"),
    ("as-text", _t('        text = text + MARKER["role"] + "\\n" + m["content"] + "\\n\\n"\n'), "error", "role-key"),
    ("as-text", _t('        text = text + MARKER[m] + "\\n" + m["content"] + "\\n\\n"\n'), "error", "whole-message"),
    ("as-text", _t('        text = text + "USER:\\n" + m["content"] + "\\n\\n"\n'), "almost", None),
    ("next-turn", _N_START, "empty", None),
    ("next-turn", _n(), "correct", None),
    ("next-turn", _n('    new = new + [{"role": "assistant", "content": reply}, {"role": "user", "content": question}]\n'), "correct", None),
    ("next-turn", _n('    new.append(reply)\n    new.append(question)\n'), "wrong", "plain-text"),
    ("next-turn", _n('    new.append({"role": "user", "content": reply})\n    new.append({"role": "assistant", "content": question})\n'), "wrong", "swapped"),
    ("next-turn", _n('    new.append({"role": "user", "content": reply})\n    new.append({"role": "user", "content": question})\n'), "wrong", "reply-as-user"),
    ("next-turn", _n('    new.append({"role": "user", "content": question})\n'), "wrong", "no-reply"),
    ("next-turn", _n('    new.append({"role": "assistant", "content": reply})\n'), "wrong", "no-question"),
    ("next-turn", _n('    new.append({"role": "user", "content": question})\n    new.append({"role": "assistant", "content": reply})\n'), "wrong", "order"),
    ("next-turn", _n('    messages.append({"role": "assistant", "content": reply})\n    messages.append({"role": "user", "content": question})\n'), "wrong", "nothing-added"),
    ("next-turn", _n('    new.append({"role": "assistant", "content": reply}, {"role": "user", "content": question})\n'), "error", "append-two"),
    ("next-turn", _n('    new = new.append({"role": "assistant", "content": reply})\n    new.append({"role": "user", "content": question})\n'), "error", "append-gives-none"),
    ("next-turn", "def next_turn(messages, reply, question):\n    messages.append({\"role\": \"assistant\", \"content\": reply})\n    messages.append({\"role\": \"user\", \"content\": question})\n    return messages\n", "correct", None),
]
