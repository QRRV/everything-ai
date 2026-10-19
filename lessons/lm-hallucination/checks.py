# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson lm-hallucination: found_share, the share of a model's lines that a source really has. And the
course's small base model of the plays, which counts which character follows the last 7 characters in Tiny
Shakespeare: the first lines it writes for Romeo, how sure it is of them, and the picture that sets them beside the
real lines of Romeo."""
import html
import math
import random
import re
from collections import Counter
from functools import cache
from statistics import mean

from everything_ai import paths
from everything_ai.check import FunctionBox, Mistake, _as_code

DATA = "data/tiny-shakespeare.txt"   # about 1.1 MB of plays by Shakespeare: see data/README.md
LOOK = 7                             # the model looks at the last 7 characters of the text so far
START = "ROMEO:\n"                   # where a speech of Romeo starts in the plays
N = 40                               # how many lines the guess and the picture use


@cache
def text() -> str:
    return (paths.ROOT / DATA).read_text(encoding="utf-8")


@cache
def play_lines() -> frozenset:
    """Every line of the plays, without spaces at the ends: the source the lesson checks against."""
    return frozenset(line.strip() for line in text().split("\n"))


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


def next_probs(s: str) -> dict:
    """The probability of each next character after s: what follows its last 7 characters in the plays; when those
    never occur, its last 6, and so on."""
    for k in range(min(LOOK, len(s)), -1, -1):
        c = _counts(s[len(s) - k:]) if k else _all()
        if c:
            n = sum(c.values())
            return {ch: c[ch] / n for ch in c}
    return {}


@cache
def write_rest(s: str, seed: int = 1) -> str:
    """The course's model continues s up to the end of the line: the same seed gives the same text."""
    rng, out = random.Random(seed), ""
    while not out.endswith("\n") and len(out) < 300:
        p = next_probs(s + out)
        options = sorted(p)
        out = out + rng.choices(options, weights=[p[ch] for ch in options])[0]
    return out.rstrip("\n")


def romeo_line(seed: int) -> str:
    """The first line the model writes after ROMEO:, with this seed."""
    return write_rest(START, seed).strip()


@cache
def line_prob(line: str) -> float:
    """How sure the model is of a line of Romeo: the average probability it gives each character of the line, and
    the end of the line, after ROMEO:."""
    s = line + "\n"
    return mean(next_probs(START + s[:i]).get(ch, 0.0) for i, ch in enumerate(s))


@cache
def made_up() -> tuple:
    """The first N lines the model writes after ROMEO:, seed 1, 2, 3 and on; a seed that starts with an empty line
    gives no line of Romeo, so it is passed over."""
    out, seed = [], 0
    while len(out) < N:
        seed = seed + 1
        line = romeo_line(seed)
        if line:
            out.append(line)
    return tuple(out)


@cache
def real() -> tuple:
    """The first lines of the first N speeches of Romeo in the plays."""
    return tuple(m.group(1).strip() for m in re.finditer(r"\nROMEO:\n([^\n]+)\n", text()))[:N]


def found() -> list:
    """The model's lines that the plays really have."""
    return [line for line in made_up() if line in play_lines()]


def how_sure(line: str) -> float:
    """line_prob, rounded to 2 decimals: what the lesson prints."""
    return round(line_prob(line), 2)


def mean_prob(lines) -> float:
    return mean(line_prob(line) for line in lines)


# --- the picture: how sure the model is of its own lines, and of real lines -----------------------------------------------

def sure_strips(labels: dict) -> str:
    """Two rows of dots on one axis, the average probability per character: the model's own lines, and the real lines
    of Romeo; each with its mean as a short line. The dots are in the colour of what a model gives."""
    W, L, R, TOP, ROW = 470, 12, 18, 34, 64
    lo, hi = 0.4, 0.9
    x = lambda p: L + (p - lo) / (hi - lo) * (W - L - R)
    rows = [(labels["made"], made_up()), (labels["real"], real())]
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    for r, (name, lines) in enumerate(rows):
        y = TOP + r * ROW
        p.append(f'<text x="{L}" y="{y + 12}" style="font-size:13px;font-weight:600;fill:var(--ea-ink)">{html.escape(name)}</text>')
        cy = y + 34
        for line in lines:
            p.append(f'<circle cx="{x(line_prob(line)):.1f}" cy="{cy}" r="4.5" style="fill:var(--ea-pred-wash);'
                     f'stroke:var(--ea-pred);stroke-width:1.5"/>')
        m = mean_prob(lines)
        p.append(f'<line x1="{x(m):.1f}" x2="{x(m):.1f}" y1="{cy - 12}" y2="{cy + 12}" style="stroke:var(--ea-ink);stroke-width:2"/>')
        p.append(f'<text x="{x(m) + 5:.1f}" y="{cy - 6}" style="font-size:12px;fill:var(--ea-ink)">'
                 f'{html.escape(labels["mean"].format(m=labels["number"](m)))}</text>')
    ya = TOP + 2 * ROW + 6
    p.append(f'<line x1="{L}" x2="{W - R}" y1="{ya}" y2="{ya}" style="stroke:var(--ea-ink-3);stroke-width:1"/>')
    for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        p.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{ya}" y2="{ya + 4}" style="stroke:var(--ea-ink-3)"/>')
        p.append(f'<text x="{x(t):.1f}" y="{ya + 17}" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:middle">'
                 f'{html.escape(labels["tick"](t))}</text>')
    p.append(f'<text x="{(W + L - R) / 2:.1f}" y="{ya + 35}" style="font-size:13px;fill:var(--ea-ink-2);text-anchor:middle">'
             f'{html.escape(labels["axis"])}</text>')
    H = ya + 42
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


# The lesson is done when these are correct. "cutoff" and "sure-not-true" are choice questions in the lesson.
REQUIRED = ["cutoff", "found-share", "sure-not-true"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_F_HEAD = "def found_share(lines, source):\n    found = 0\n    for line in lines:\n"
_F_END = "    return found / len(lines)\n"
_F_START = _F_HEAD + "        ...\n" + _F_END


def _f(inner: str = "        if line in source:\n            found = found + 1\n") -> str:
    return _F_HEAD + inner + _F_END


def _shown(v) -> str:
    """What found_share gave, for the messages: a share rounded to 4 decimals, so 2 / 3 shows as 0.6667 and not with
    float noise; a whole number as 0 or 1; anything else as Python shows it."""
    if isinstance(v, float) and math.isfinite(v):
        v = round(v, 4) + 0.0
        return str(int(v)) if v.is_integer() else repr(v)
    return _as_code(v) if isinstance(v, (str, list)) else repr(v)


_MODEL = ["Good night.", "Be merry.", "Whither?", "Away, away!", "Away!"]
_SOURCE = ["Whither?", "Good night.", "Away!"]

EXERCISES = {
    "found-share": FunctionBox(
        name="found-share",
        function="found_share",
        start=_F_START,
        solution=_f(),
        visible=[(_MODEL, _SOURCE)],
        show=_shown,
        hidden=[(["a", "b"], ["b", "c", "a"]), (["one", "two", "three", "four", "five"], ["two"]),
                (["x", "y", "z", "w"], []), (["Away!", "Away!"], {"Away!"}),
                (["We are", "we are", "We are ", "We were"], ["We are"])],
        first=[_DOTS],
        mistakes=[
            Mistake("not-in", wrong=_f("        if line not in source:\n            found = found + 1\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you count the lines that are not in the source. The share asked for is that of the lines that are.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt de regels die niet in de bron staan. Gevraagd is het deel van de regels die er wel in staan."),
            Mistake("set-to-one", wrong=_f("        if line in source:\n            found = 1\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `found` becomes 1 for every line that is found, and stays 1. Make it go up by one each time: `found = found + 1`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `found` wordt 1 bij elke regel die gevonden wordt, en blijft 1. Laat hem elke keer één omhoog gaan: `found = found + 1`."),
            Mistake("count-all", wrong=_f("        found = found + 1\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you count every line, found or not. Count a line only when it is in `source`, with an `if`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt elke regel, gevonden of niet. Tel een regel alleen als hij in `source` staat, met een `if`."),
            Mistake("none-found", wrong=_f("        if line == source:\n            found = found + 1\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: not one line is counted as found. Does your `if` check whether `line` is one of the lines of `source`, with `in`?",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: geen enkele regel telt als gevonden. Kijkt je `if` of `line` een van de regels van `source` is, met `in`?"),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the `...` with an `if` that adds one to `found` when the line is in `source`.",
                      "nl": "Vervang de `...` door een `if` die één bij `found` optelt als de regel in `source` staat."},
            "arguments": {"en": "The check calls `{call}`, with the model's lines and a source, and the `def` line of your function has another number of names between its parentheses. Keep `(lines, source)`.",
                          "nl": "De check roept `{call}` aan, met de regels van het model en een bron, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(lines, source)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Count by hand: which of the lines are in the source, and how many lines are there?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Tel met de hand: welke van de regels staan in de bron, en hoeveel regels zijn er?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any lines and any source, also an empty one?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor alle regels en elke bron, ook een lege?"},
            "correct": {"en": "Your function is right in all {n} cases: it counts the lines the source really has, and divides by the number of lines.",
                        "nl": "Je functie klopt in alle {n} gevallen: hij telt de regels die echt in de bron staan, en deelt door het aantal regels."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("found-share", _F_START, "empty", None),
    ("found-share", _f(), "correct", None),
    ("found-share", _f("        if line in source:\n            found = 1 + found\n"), "correct", None),
    ("found-share", _f("        if line not in source:\n            found = found + 1\n"), "wrong", "not-in"),
    ("found-share", _f("        if line in source:\n            found = 1\n"), "wrong", "set-to-one"),
    ("found-share", _f("        found = found + 1\n"), "wrong", "count-all"),
    ("found-share", _f("        if line == source:\n            found = found + 1\n"), "wrong", "none-found"),
    ("found-share", _f("        if line in source[0]:\n            found = found + 1\n"), "wrong", None),
    ("found-share", _f("        if line in source:\n            found = found + 1\n        else:\n            found = found - 1\n"), "wrong", None),
    ("found-share", _F_HEAD + "        if line in source:\n            found = found + 1\n    return found // len(lines)\n", "wrong", None),
    ("found-share", _F_HEAD + "        if line in source:\n            found = found + 1\n    return found / len(source)\n", "wrong", None),
]
