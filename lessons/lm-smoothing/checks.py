# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson lm-smoothing: smooth, add-k smoothing of a count matrix. And the numbers and the picture: the
NLL of the plays with and without smoothing, on the part that was counted and on a part that was not."""
import html
import math
import re
from functools import cache

import numpy as np

from everything_ai import paths
from everything_ai.check import FunctionBox, Mistake

DATA = "data/tiny-shakespeare.txt"   # about 1.1 MB of plays by Shakespeare: see data/README.md
SHARE = 0.9                          # the first 90% of the plays is counted; the last 10% is the text to measure on


@cache
def text() -> str:
    return (paths.ROOT / DATA).read_text(encoding="utf-8")


@cache
def chars() -> tuple[str, ...]:
    return tuple(sorted(set(text())))


@cache
def _codes() -> np.ndarray:
    return np.searchsorted(np.array(chars()), np.array(list(text())))


def _counts(codes: np.ndarray) -> np.ndarray:
    n = np.zeros((len(chars()), len(chars())))
    np.add.at(n, (codes[:-1], codes[1:]), 1)
    return n


@cache
def _parts() -> tuple[np.ndarray, np.ndarray]:
    codes = _codes()
    cut = int(len(codes) * SHARE)
    return codes[:cut], codes[cut:]


def _smoothed(n: np.ndarray, k: float) -> np.ndarray:
    m = n + k
    return m / m.sum(axis=1, keepdims=True)


def _nll(p: np.ndarray, codes: np.ndarray) -> float:
    with np.errstate(divide="ignore"):
        return float(-np.log(p[codes[:-1], codes[1:]]).mean())


@cache
def nll_split(k: float) -> tuple[float, float]:
    """With the first 90% counted and add-k smoothing: the NLL on the counted part, and on the last 10%."""
    first, last = _parts()
    p = _smoothed(_counts(first), k)
    return _nll(p, first), _nll(p, last)


def unseen_in_last() -> tuple[int, int]:
    """How many pairs of the last 10% never came in the first 90%, and how many pairs the last 10% has."""
    first, last = _parts()
    n = _counts(first)
    return int((n[last[:-1], last[1:]] == 0).sum()), len(last) - 1


def zero_cells() -> int:
    return int((_counts(_codes()) == 0).sum())


def row_count(ch: str) -> int:
    return int(_counts(_codes())[chars().index(ch)].sum())


def grouped(n: int, lang: str) -> str:
    s = f"{n:,}"
    return s.replace(",", ".") if lang == "nl" else s


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: infinity as a word, the rest with the decimal mark of the language."""
    if math.isinf(x):
        return {"en": "infinite", "nl": "oneindig"}[lang]
    s = f"{x:.{digits}f}"
    return s.replace(".", ",") if lang == "nl" else s


# --- the picture: the row of "Z" before and after add-1 smoothing ------------------------------------------------------

def row_bars(ch: str, labels: dict, k: float = 1) -> str:
    """Two panels with a bar for each of the 65 characters after ch, in the order of the table: the probabilities
    without smoothing, where many are 0, and with add-k smoothing, where every character gets a sliver."""
    n = _counts(_codes())[chars().index(ch)]
    panels = [(labels["before"], n / n.sum()), (labels["after"], (n + k) / (n + k).sum())]
    W, LEFT, PH, TOP = 540, 34, 84, 30
    bw = (W - LEFT) / len(n)
    top = max(float(panels[0][1].max()), 0.01)
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    for i, (name, probs) in enumerate(panels):
        y0 = TOP + i * (PH + 34)
        zeros = int((probs == 0).sum())
        p.append(f'<text x="0" y="{y0 + 12}" style="font-size:13px;font-weight:600;fill:var(--ea-ink)">{html.escape(name.format(n=zeros))}</text>')
        base = y0 + 18 + PH - 6
        p.append(f'<line x1="{LEFT}" x2="{W}" y1="{base}" y2="{base}" style="stroke:var(--ea-line)"/>')
        for j, v in enumerate(probs):
            h = (PH - 10) * float(v) / top
            if v > 0:
                p.append(f'<rect x="{LEFT + j * bw + 0.5:.1f}" y="{base - max(h, 1.5):.1f}" width="{bw - 1:.1f}" height="{max(h, 1.5):.1f}" style="fill:var(--ea-pred)"/>')
    H = TOP + 2 * (PH + 34) + 12
    p.append(f'<text x="{LEFT}" y="{H - 6}" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(labels["axis"])}</text>')
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


# The lesson is done when these are correct. "unseen-prob" and "pick-k" are the choice questions in the lesson.
REQUIRED = ["unseen-prob", "smooth", "pick-k"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


class _Matrix(np.ndarray):
    """A count matrix that a check message writes the way the lesson's code does: np.array([[...]]). The learner's
    code gets a plain copy, so that what it computes is an ordinary array."""

    def __repr__(self):
        a = np.asarray(self)
        if a.ndim != 2:
            return repr(a)
        return f"np.array({[[int(x) if x == int(x) else float(x) for x in row] for row in a.tolist()]!r})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _rows(v) -> str:
    """A small matrix for the messages, as a list of rows, rounded to 3 decimals; nan and inf as they are."""
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return repr(v)
    if a.ndim != 2:
        return repr(v)
    a = np.round(a, 3) + 0.0
    return repr([[int(x) if math.isfinite(x) and x == int(x) else float(x) for x in row] for row in a.tolist()])


_N1 = np.array([[3, 1, 0], [0, 0, 2]]).view(_Matrix)
_N2 = np.array([[5, 0, 0, 3], [1, 1, 0, 0], [0, 0, 0, 4]]).view(_Matrix)
_N3 = np.array([[0, 0], [2, 6]]).view(_Matrix)

_HEAD = "def smooth(N, k):\n"
_START = _HEAD + "    ...\n"


def _s(body: str = "    M = N + k\n    return M / M.sum(axis=1, keepdims=True)\n") -> str:
    return _HEAD + body


EXERCISES = {
    "smooth": FunctionBox(
        name="smooth",
        function="smooth",
        start=_START,
        solution=_s(),
        visible=[(_N1, 1)],
        hidden=[(_N1, 0.5), (_N2, 1), (_N2, 2), (_N3, 1)],
        show=_rows,
        first=[_DOTS],
        mistakes=[
            Mistake("old-sums", wrong=_s("    return (N + k) / N.sum(axis=1, keepdims=True)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you divide by the sums of the old counts, so the rows no longer add up to 1. Divide by the sums of the new counts, with `k` added.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je deelt door de sommen van de oude tellingen, dus de rijen tellen niet meer op tot 1. Deel door de sommen van de nieuwe tellingen, met `k` erbij."),
            Mistake("after", wrong=_s("    return N / N.sum(axis=1, keepdims=True) + k\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you add `k` to the probabilities, after dividing. Add it to the counts, before you divide.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt `k` bij de kansen op, na het delen. Tel het bij de tellingen op, voordat je deelt."),
            Mistake("once", wrong=_s("    return (N + k) / (N.sum(axis=1, keepdims=True) + k)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you add `k` once to each row's sum, but it was added to every count of the row. Add up the new counts themselves: then `k` is in the sum once for every count of the row.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt `k` één keer bij de som van elke rij op, maar het kwam bij elke telling van de rij. Tel de nieuwe tellingen zelf op: dan zit `k` er één keer per telling van de rij in."),
            Mistake("no-keepdims", error=lambda e: isinstance(e, ValueError) and "broadcast" in str(e),
                    en="With `{call}`, NumPy cannot stretch the shapes to fit each other. Are your row sums still a column? `M.sum(axis=1)` alone gives them as one flat row; with `keepdims=True` they stay a column, so that broadcasting divides each row by its own sum.",
                    nl="Met `{call}` kan NumPy de shapes niet zo uitrekken dat ze passen. Zijn je rijsommen nog een kolom? `M.sum(axis=1)` alleen geeft ze als één platte rij; met `keepdims=True` blijven ze een kolom, zodat broadcasting elke rij door haar eigen som deelt."),
            Mistake("in-place", error=lambda e: isinstance(e, TypeError) and "Cannot cast ufunc" in str(e),
                    en="With `{call}`, your code puts numbers with a decimal point back into `N` itself, and this `N` holds whole numbers only. Make a new matrix instead, as in step 1 of the recipe.",
                    nl="Met `{call}` zet je code getallen met een komma terug in `N` zelf, en deze `N` bevat alleen gehele getallen. Maak in plaats daarvan een nieuwe matrix, zoals in stap 1 van het recept."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the `...` with steps 1 and 2 of the recipe: add `k` to every count, and divide every row by its new sum.",
                      "nl": "Vervang de `...` door stap 1 en 2 van het recept: tel `k` bij elke telling op, en deel elke rij door haar nieuwe som."},
            "arguments": {"en": "The check calls `{call}`, with a count matrix and a number `k`, and the `def` line of your function has another number of names between its parentheses. Keep `(N, k)`.",
                          "nl": "De check roept `{call}` aan, met een telmatrix en een getal `k`, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(N, k)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does every row of your result add up to 1, and does a count of 0 get more than 0?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Telt elke rij van je uitkomst op tot 1, en krijgt een telling van 0 meer dan 0?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every `k`, also one that is not 1?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke `k`, ook een die niet 1 is?"},
            "correct": {"en": "Your function is right in all {n} cases: every pair gets at least a little probability, and every row still adds up to 1.",
                        "nl": "Je functie klopt in alle {n} gevallen: elk paar krijgt minstens een beetje kans, en elke rij telt nog steeds op tot 1."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("smooth", _START, "empty", None),
    ("smooth", _s(), "correct", None),
    ("smooth", _s("    return (N + k) / (N + k).sum(axis=1, keepdims=True)\n"), "correct", None),
    ("smooth", _s("    return (N + k) / N.sum(axis=1, keepdims=True)\n"), "wrong", "old-sums"),
    ("smooth", _s("    return N / N.sum(axis=1, keepdims=True) + k\n"), "wrong", "after"),
    ("smooth", _s("    return (N + k) / (N.sum(axis=1, keepdims=True) + k)\n"), "wrong", "once"),
    ("smooth", _s("    M = N + k\n    return M / M.sum(axis=1)\n"), "error", "no-keepdims"),
    ("smooth", _s("    N += k\n    return N / N.sum(axis=1, keepdims=True)\n"), "error", "in-place"),
]
