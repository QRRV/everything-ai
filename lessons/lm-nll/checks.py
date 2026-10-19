# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson lm-nll: nll, the average negative log-likelihood of a text under a table of bigram
probabilities. And the picture of the surprise of every pair of a line from the plays."""
import html
import math
import re
from functools import cache

import numpy as np

from everything_ai import paths
from everything_ai.check import FunctionBox, Mistake

DATA = "data/tiny-shakespeare.txt"   # about 1.1 MB of plays by Shakespeare: see data/README.md


@cache
def text() -> str:
    return (paths.ROOT / DATA).read_text(encoding="utf-8")


@cache
def chars() -> tuple[str, ...]:
    """The 65 characters of the plays, in sorted order: the order of the rows and the columns."""
    return tuple(sorted(set(text())))


@cache
def course_probs() -> np.ndarray:
    """The course's own matrix of probabilities: row i is the distribution of the next character after chars()[i]."""
    t, cs = text(), chars()
    counts = {}
    for i in range(len(t) - 1):
        counts[t[i:i + 2]] = counts.get(t[i:i + 2], 0) + 1
    n = np.array([[counts.get(a + b, 0) for b in cs] for a in cs], dtype=float)
    return n / n.sum(axis=1, keepdims=True)


def prob(a: str, b: str) -> float:
    cs = chars()
    return float(course_probs()[cs.index(a), cs.index(b)])


def surprises(s: str) -> list[float]:
    """The surprise, minus the log of the probability, of every pair of s under the model of the plays."""
    with np.errstate(divide="ignore"):
        return [float(-np.log(prob(s[i], s[i + 1]))) for i in range(len(s) - 1)]


def text_nll(s: str) -> float:
    """The average negative log-likelihood of s under the model of the plays: infinite when a pair never came."""
    return float(np.mean(surprises(s)))


@cache
def plays_nll() -> float:
    """The average NLL of the model of the plays on the plays themselves, over all 1,115,393 pairs."""
    t, cs = text(), chars()
    codes = np.searchsorted(np.array(cs), np.array(list(t)))   # chars() is sorted, so this is each character's row
    p = course_probs()[codes[:-1], codes[1:]]
    return float(-np.log(p).mean())


def grouped(n: int, lang: str) -> str:
    """A whole number with its thousands marked: 1,115,393 in English, 1.115.393 in Dutch."""
    s = f"{n:,}"
    return s.replace(",", ".") if lang == "nl" else s


def uniform_nll() -> float:
    return math.log(len(chars()))


def pair_count() -> int:
    return len(text()) - 1


# --- the picture: the surprise of every pair of a line ------------------------------------------------------------------

def surprise_bars(s: str, labels: dict) -> str:
    """A bar per pair of s, as high as its surprise under the model of the plays, with the average as a dashed line.
    Surprise is what the loss adds up, so the bars are the loss colour."""
    vals = surprises(s)
    n = len(vals)
    W, LEFT, RIGHT, TOP, PH = 540, 30, 70, 34, 150   # RIGHT: room for the label of the average, after the bars
    END = W - RIGHT
    bw = (END - LEFT) / n
    top = max(4.0, math.ceil(max(vals)))
    def y(v):
        return TOP + PH - PH * v / top
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    for k in range(0, int(top) + 1):
        p.append(f'<line x1="{LEFT}" x2="{END}" y1="{y(k):.1f}" y2="{y(k):.1f}" style="stroke:var(--ea-line);stroke-width:1"/>'
                 f'<text x="{LEFT - 6}" y="{y(k) + 4:.1f}" style="font-size:11px;fill:var(--ea-ink-2);text-anchor:end">{k}</text>')
    for i, v in enumerate(vals):
        x = LEFT + i * bw + 2
        p.append(f'<rect x="{x:.1f}" y="{y(v):.1f}" width="{bw - 4:.1f}" height="{TOP + PH - y(v):.1f}" style="fill:var(--ea-loss)"/>')
        pair = s[i:i + 2].replace(" ", "␣")
        p.append(f'<text x="{x + (bw - 4) / 2:.1f}" y="{TOP + PH + 15}" style="font-size:11px;font-family:var(--ea-font-code);'
                 f'fill:var(--ea-ink-2);text-anchor:middle">{html.escape(pair)}</text>')
    mean = sum(vals) / n
    word, _, number = labels["mean"].rpartition(" ")   # "average 2.16": the word above the line, the number below it
    p.append(f'<line x1="{LEFT}" x2="{END}" y1="{y(mean):.1f}" y2="{y(mean):.1f}" style="stroke:var(--ea-loss-text);stroke-width:1.5;stroke-dasharray:5 4"/>'
             f'<text x="{END + 6}" y="{y(mean) - 4:.1f}" style="font-size:12px;fill:var(--ea-loss-text)">{html.escape(word)}'
             f'<tspan x="{END + 6}" dy="15">{html.escape(number)}</tspan></text>')
    H = TOP + PH + 24
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


# The lesson is done when these are correct. "by-hand" and "most-surprising" are the choice questions in the lesson.
REQUIRED = ["by-hand", "nll", "most-surprising"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


class _Matrix(np.ndarray):
    """A table that a check message writes the way the lesson's code does: np.array([[...]]). The learner's code gets
    a plain copy, so that what it computes is an ordinary array."""

    def __repr__(self):
        a = np.asarray(self)
        if a.ndim != 2:
            return repr(a)
        return f"np.array({[[int(x) if x == int(x) else float(x) for x in row] for row in a.tolist()]!r})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _shown(v) -> str:
    """A number for the messages, rounded to 3 decimals; infinity as "inf", as NumPy writes it."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        if isinstance(v, np.ndarray):   # a whole row instead of one number: its numbers rounded, without float noise
            return repr([_shown(x) for x in v.ravel().tolist()]).replace("'", "")
        return repr(v)
    if math.isinf(x) or math.isnan(x):
        return repr(x)
    return repr(round(x, 3) + 0.0)


_P3 = np.array([[0.1, 0.6, 0.3], [0.4, 0, 0.6], [0.8, 0.2, 0]]).view(_Matrix)
_C3 = ["a", "b", "c"]
_P4 = np.array([[0, 0.5, 0.3, 0.2], [0.6, 0, 0.1, 0.3], [0.2, 0.7, 0, 0.1], [0.5, 0.1, 0.4, 0]]).view(_Matrix)
_C4 = [" ", "e", "h", "t"]

_HEAD = ("import numpy as np\n\n\ndef nll(P, chars, text):\n    row_of = {}\n    for k in range(len(chars)):\n"
         "        row_of[chars[k]] = k\n    total = 0\n    for i in range(len(text) - 1):\n")
_END = "    return total / (len(text) - 1)\n"
_START = _HEAD + "        ...\n" + _END


def _n(inner: str = "        p = P[row_of[text[i]], row_of[text[i + 1]]]\n        total = total - np.log(p)\n") -> str:
    return _HEAD + inner + _END


EXERCISES = {
    "nll": FunctionBox(
        name="nll",
        function="nll",
        start=_START,
        solution=_n(),
        visible=[(_P3, _C3, "abca")],
        hidden=[(_P3, _C3, "acab"), (_P3, _C3, "ca"), (_P4, _C4, "the the"), (_P4, _C4, "het"), (_P3, _C3, "abb")],
        show=_shown,
        first=[_DOTS],
        mistakes=[
            Mistake("plus", wrong=_n("        p = P[row_of[text[i]], row_of[text[i + 1]]]\n        total = total + np.log(p)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your NLL is negative. The log of a probability below 1 is negative; the surprise is minus the log: subtract `np.log(p)` from the total.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je NLL is negatief. De log van een kans onder 1 is negatief; de verrassing is min de log: trek `np.log(p)` van het totaal af."),
            Mistake("swapped", wrong=_n("        p = P[row_of[text[i + 1]], row_of[text[i]]]\n        total = total - np.log(p)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you look up the pair the wrong way round. The row is the character before, `text[i]`, and the column the one after it, `text[i + 1]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je zoekt het paar andersom op. De rij is het teken ervoor, `text[i]`, en de kolom het teken erna, `text[i + 1]`."),
            Mistake("math-log", error=lambda e: isinstance(e, ValueError) and "math domain" in str(e),
                    en="With `{call}`, your function stops with an error: `math.log` only takes numbers above 0. `np.log(0)` gives `-inf` instead, so that a pair the model never saw gets an infinite surprise. Use `np.log`.",
                    nl="Met `{call}` stopt je functie met een fout: `math.log` neemt alleen getallen boven 0. `np.log(0)` geeft dan `-inf`, zodat een paar dat het model nooit zag een oneindige verrassing krijgt. Gebruik `np.log`."),
            Mistake("probs", wrong=_n("        p = P[row_of[text[i]], row_of[text[i + 1]]]\n        total = total + p\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you add up the probabilities themselves. Add up their surprises instead: minus the log of each.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt de kansen zelf op. Tel in plaats daarvan hun verrassingen op: min de log van elk."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the `...` with steps 2 and 3 of the recipe: the probability of the pair at `i`, and its surprise added to the total.",
                      "nl": "Vervang de `...` door stap 2 en 3 van het recept: de kans van het paar bij `i`, en de verrassing ervan opgeteld bij het totaal."},
            "arguments": {"en": "The check calls `{call}`, with a table, the characters and a text, and the `def` line of your function has another number of names between its parentheses. Keep `(P, chars, text)`.",
                          "nl": "De check roept `{call}` aan, met een tabel, de tekens en een tekst, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(P, chars, text)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your two lines with steps 2 and 3 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je twee regels met stap 2 en 3 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every table and every text?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke tabel en elke tekst?"},
            "correct": {"en": "Your function is right in all {n} cases: the average surprise per character, lower when the model predicted the text better.",
                        "nl": "Je functie klopt in alle {n} gevallen: de gemiddelde verrassing per teken, lager als het model de tekst beter voorspelde."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("nll", _START, "empty", None),
    ("nll", _n(), "correct", None),
    ("nll", _n("        total = total - np.log(P[row_of[text[i]], row_of[text[i + 1]]])\n"), "correct", None),
    ("nll", _n("        p = P[row_of[text[i]], row_of[text[i + 1]]]\n        total = total + np.log(p)\n"), "wrong", "plus"),
    ("nll", _n("        p = P[row_of[text[i + 1]], row_of[text[i]]]\n        total = total - np.log(p)\n"), "wrong", "swapped"),
    ("nll", _n("        p = P[row_of[text[i]], row_of[text[i + 1]]]\n        total = total + p\n"), "wrong", "probs"),
    ("nll", "import math\n" + _n("        p = P[row_of[text[i]], row_of[text[i + 1]]]\n        total = total - math.log(p)\n"), "error", "math-log"),
    ("nll", _n("        p = P[row_of[text[i]]]\n        total = total - np.log(p)\n"), "wrong", None),
]
