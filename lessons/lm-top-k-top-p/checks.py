# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson lm-top-k-top-p: top_k, which keeps only the k most likely characters of a row, and top_p,
which keeps the smallest set of most likely characters that holds probability p. And the pictures: which characters
after a space each keeps, and what the model of the plays then writes."""
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


def keep_top_k(p, k: int) -> np.ndarray:
    """The course's top-k: the k most likely kept, the rest 0, divided by their sum."""
    p = np.asarray(p, dtype=float)
    cut = sorted(p)[-k]
    keep = np.where(p >= cut, p, 0)
    return keep / keep.sum()


def keep_top_p(p, top: float) -> np.ndarray:
    """The course's top-p: the most likely kept until together they hold at least top, the rest 0."""
    p = np.asarray(p, dtype=float)
    total, cut = 0, 0
    for q in sorted(p, reverse=True):
        total = total + q
        if total >= top:
            cut = q
            break
    keep = np.where(p >= cut, p, 0)
    return keep / keep.sum()


def row(ch: str) -> np.ndarray:
    return course_probs()[chars().index(ch)]


def possible_after(ch: str) -> int:
    return int((row(ch) > 0).sum())


def kept_after(ch: str, top: float) -> int:
    return int((keep_top_p(row(ch), top) > 0).sum())


def shown(ch: str) -> str:
    return {" ": "␣", "\n": "↵"}.get(ch, ch)


def text_block(s: str) -> str:
    """A text the model wrote, in a box that wraps its lines, so that a long line stays readable on a phone."""
    return (f'<pre style="white-space:pre-wrap;word-break:break-word;margin:0.25rem 0 0.75rem;padding:0.6rem 0.8rem;'
            f'border-radius:6px;background:var(--ea-inset);color:var(--ea-ink);font-family:var(--ea-font-code);'
            f'font-size:0.875rem;line-height:1.45">{html.escape(s)}</pre>')


def sample_text(keep, seed: int = 1, n: int = 160, start: str = "T") -> str:
    """What the model of the plays writes when every row first goes through `keep`: sampling, with seed `seed`."""
    P, cs = course_probs(), chars()
    row_of = {c: k for k, c in enumerate(cs)}
    rng = np.random.default_rng(seed)
    out = start
    while len(out) < n:
        out = out + cs[int(rng.choice(len(cs), p=keep(P[row_of[out[-1]]])))]
    return out


def tail_pairs(text: str, top: float = 0.9) -> int:
    """How many pairs of `text` top-p would have cut off: the next character is outside the kept set of the row of the
    character before it."""
    P, cs = course_probs(), chars()
    row_of = {c: k for k, c in enumerate(cs)}
    return sum(1 for i in range(len(text) - 1)
               if keep_top_p(P[row_of[text[i]]], top)[row_of[text[i + 1]]] == 0)


def tail_share(ch: str, top: float = 0.9) -> float:
    """The probability that top-p drops after ch: the share of the characters it cuts off."""
    r = row(ch)
    return float(r[keep_top_p(r, top) == 0].sum())


# --- the picture: the row after a space, sorted, with what top-k and top-p keep ----------------------------------------

def kept_bars(ch: str, labels: dict, k: int = 5, top: float = 0.9, show: int = 30) -> str:
    """Two panels of the `show` most likely characters after ch, sorted from large to small: the bars that top-k keeps
    and the bars that top-p keeps in the colour of the model's probabilities, the bars they drop light grey."""
    r = row(ch)
    order = [j for j in np.argsort(-r, kind="stable") if r[j] > 0]
    rest = len(order) - show
    order = order[:show]
    cs = chars()
    panels = [(labels["k"], keep_top_k(r, k)), (labels["p"], keep_top_p(r, top))]
    W, BW, GAP, PH, TOP = 480, 13, 3, 92, 30
    scale = (PH - 26) / float(r[order[0]])
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    for n_panel, (name, kept) in enumerate(panels):
        y0 = TOP + n_panel * (PH + 34)
        n_kept = int((kept > 0).sum())
        p.append(f'<text x="0" y="{y0 + 12}" style="font-size:13px;font-weight:600;fill:var(--ea-ink)">'
                 f'{html.escape(name.format(n=n_kept))}</text>')
        base = y0 + 18 + PH - 20
        for i, j in enumerate(order):
            x = i * (BW + GAP)
            h = max(float(r[j]) * scale, 1.5)
            colour = "fill:var(--ea-pred)" if kept[j] > 0 else "fill:var(--ea-ink-3);fill-opacity:0.35"
            p.append(f'<rect x="{x}" y="{base - h:.1f}" width="{BW}" height="{h:.1f}" style="{colour}"/>')
            p.append(f'<text x="{x + BW / 2:.1f}" y="{base + 13}" style="font-size:11px;fill:var(--ea-ink-2);text-anchor:middle">'
                     f'{html.escape(shown(cs[j]))}</text>')
    if rest > 0:
        p.append(f'<text x="0" y="{TOP + 2 * (PH + 34) - 2}" style="font-size:12px;fill:var(--ea-ink-2)">'
                 f'{html.escape(labels["rest"].format(n=rest))}</text>')
    H = TOP + 2 * (PH + 34) + 4
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


# The lesson is done when these are correct. "keeps" is the choice question in the lesson.
REQUIRED = ["keeps", "top-k", "top-p"]

# A cut above every probability keeps nothing: keep is all zeros, and keep / keep.sum() is 0 / 0, nan everywhere.
_NOTHING = Mistake("nothing-kept", matches=lambda got, text: _all_nan(got),
                   en="`{call}` gives `{got}`: `nan`, not a number. No probability in `p` is at least your `cut`, so `keep` has only zeros, and `keep / keep.sum()` divides 0 by 0. The cut should be one of the probabilities in `p`.",
                   nl="`{call}` geeft `{got}`: `nan`, not a number: geen getal. Geen enkele kans in `p` is minstens je `cut`, dus `keep` bevat alleen nullen, en `keep / keep.sum()` deelt 0 door 0. De grens hoort een van de kansen in `p` te zijn.")

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


class _Vector(np.ndarray):
    """Probabilities that a check message writes the way the lesson's code does: np.array([...]). The learner's code
    gets a plain copy, so that what it computes is an ordinary array."""

    def __repr__(self):
        a = np.asarray(self)
        if a.ndim != 1:
            return repr(a)
        return f"np.array({[int(x) if x == int(x) else float(x) for x in a.tolist()]!r})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _vec(values) -> np.ndarray:
    return np.array(values, dtype=float).view(_Vector)


def _rounded(v) -> str:
    """Probabilities for the messages, rounded to 3 decimals, so that no float noise shows; nan and inf as they are:
    int() of them fails, and a cut that keeps nothing gives 0 / 0."""
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return repr(v)
    if a.ndim != 1:
        return repr(v)
    return repr([int(x) if math.isfinite(x) and x == int(x) else round(float(x), 3) for x in (a + 0.0).tolist()])


def _all_nan(got) -> bool:
    try:
        a = np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return False
    return a.size > 0 and bool(np.isnan(a).all())


def _close(got, want) -> bool:
    try:
        return np.allclose(np.asarray(got, dtype=float), np.asarray(want, dtype=float), rtol=1e-6, atol=1e-9)
    except (TypeError, ValueError):
        return False


_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_P1 = _vec([0.1, 0.5, 0.15, 0.25])
_P2 = _vec([0.3, 0.1, 0.4, 0.15, 0.05])
_P3 = _vec([0.05, 0.6, 0.35])
_P4 = _vec([0.5, 0.25, 0.2, 0.05])
_P5 = _vec([0.1, 0.5, 0.15, 0.2, 0.05])

# --- exercise 2: top_k --------------------------------------------------------------------------------------------------

_K_HEAD = "import numpy as np\n\n\ndef top_k(p, k):\n"
_K_END = "    keep = np.where(p >= cut, p, 0)\n    return keep / keep.sum()\n"
_K_START = _K_HEAD + "    cut = ...\n" + _K_END


def _k(cut: str = "sorted(p)[-k]") -> str:
    return _K_HEAD + f"    cut = {cut}\n" + _K_END


# --- exercise 3: top_p --------------------------------------------------------------------------------------------------

_P_HEAD = "import numpy as np\n\n\ndef top_p(p, top):\n    total = 0\n    for q in sorted(p, reverse=True):\n        total = total + q\n"
_P_END = "    keep = np.where(p >= cut, p, 0)\n    return keep / keep.sum()\n"
_P_START = _P_HEAD + "        ...\n" + _P_END


def _p(inner: str = "        if total >= top:\n            cut = q\n            break\n") -> str:
    return _P_HEAD + inner + _P_END


EXERCISES = {
    "top-k": FunctionBox(
        name="top-k",
        function="top_k",
        start=_K_START,
        solution=_k(),
        visible=[(_P5, 2)],
        hidden=[(_P1, 2), (_P1, 1), (_P2, 3), (_P3, 3), (_P2, 4)],
        show=_rounded,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            _NOTHING,
            Mistake("small-end", wrong=_k("sorted(p)[k]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `sorted(p)` goes from small to large, so `sorted(p)[k]` counts from the small end. The k-th largest is at place `-k`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `sorted(p)` loopt van klein naar groot, dus `sorted(p)[k]` telt vanaf de kleine kant. De k-de grootste staat op plaats `-k`."),
            Mistake("one-more", wrong=_k("sorted(p)[-k - 1]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you keep one character too many. With `sorted(p)[-1]` the cut is the largest, so `sorted(p)[-k]` is the k-th largest.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je houdt één teken te veel. Met `sorted(p)[-1]` is de grens de grootste, dus `sorted(p)[-k]` is de k-de grootste."),
            Mistake("max", wrong=_k("max(p)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: with the largest as the cut, only one character is left, whatever `k` is. Use `k`: the k-th largest is `sorted(p)[-k]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: met de grootste als grens blijft er maar één teken over, wat `k` ook is. Gebruik `k`: de k-de grootste is `sorted(p)[-k]`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the cut: the k-th largest probability in `p`.",
                      "nl": "Vervang de `...` door de grens: de k-de grootste kans in `p`."},
            "arguments": {"en": "The check calls `{call}`, with probabilities and a number `k`, and the `def` line of your function has another number of names between its parentheses. Keep `(p, k)`.",
                          "nl": "De check roept `{call}` aan, met kansen en een getal `k`, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(p, k)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do exactly `k` characters keep a probability above 0?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Houden precies `k` tekens een kans boven 0?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every `k`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke `k`?"},
            "correct": {"en": "Your function is right in all {n} cases: the k most likely keep their share, the others get 0.",
                        "nl": "Je functie klopt in alle {n} gevallen: de k waarschijnlijkste houden hun deel, de andere krijgen 0."},
        },
    ),
    "top-p": FunctionBox(
        name="top-p",
        function="top_p",
        start=_P_START,
        solution=_p(),
        visible=[(_P2, 0.8)],
        hidden=[(_P1, 0.3), (_P2, 0.5), (_P3, 0.9), (_P4, 0.75), (_P2, 0.99)],
        show=_rounded,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            _NOTHING,
            Mistake("no-break", wrong=_p("        if total >= top:\n            cut = q\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every probability is kept. Does the loop stop with `break` as soon as the total reaches `top`? Without it, the loop goes on and `cut` ends as the smallest.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elke kans blijft. Stopt de loop met `break` zodra het totaal `top` haalt? Zonder `break` gaat de loop door en wordt `cut` de kleinste."),
            Mistake("above", wrong=_p("        if total > top:\n            cut = q\n            break\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: when the total is exactly `top`, you go on to the next character. The set may hold exactly `top`: use `>=`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: als het totaal precies `top` is, ga je door naar het volgende teken. De set mag precies `top` bevatten: gebruik `>=`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with an `if`: once the total is at least `top`, keep `q` as the cut and stop the loop.",
                      "nl": "Vervang de `...` door een `if`: zodra het totaal minstens `top` is, houd je `q` als grens en stop je de loop."},
            "arguments": {"en": "The check calls `{call}`, with probabilities and a number `top`, and the `def` line of your function has another number of names between its parentheses. Keep `(p, top)`.",
                          "nl": "De check roept `{call}` aan, met kansen en een getal `top`, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(p, top)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your lines with step 3 of the recipe: which character makes the total reach `top`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je regels met stap 3 van het recept: bij welk teken haalt het totaal `top`?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every `top`, also one that the most likely character reaches on its own?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke `top`, ook een die het waarschijnlijkste teken alleen al haalt?"},
            "correct": {"en": "Your function is right in all {n} cases: it keeps the most likely characters until together they hold at least `top`, and no more.",
                        "nl": "Je functie klopt in alle {n} gevallen: hij houdt de waarschijnlijkste tekens tot ze samen minstens `top` bevatten, en niet meer."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("top-k", _K_START, "empty", None),
    ("top-k", _k("sorted(p, reverse=True)[k - 1]"), "correct", None),
    ("top-k", _k("sorted(p)[k]"), "wrong", "small-end"),
    ("top-k", _k("sorted(p)[-k - 1]"), "wrong", "one-more"),
    ("top-k", _k("max(p)"), "wrong", "max"),
    ("top-k", _k("k"), "wrong", "nothing-kept"),
    ("top-p", _P_START, "empty", None),
    ("top-p", _p("        if total >= top:\n            cut = q\n            break\n"), "correct", None),
    ("top-p", _p("        if total >= top:\n            cut = q\n"), "wrong", "no-break"),
    ("top-p", _p("        if total > top:\n            cut = q\n            break\n"), "wrong", "above"),
    ("top-p", _p("        if total >= top:\n            cut = total\n            break\n"), "wrong", "nothing-kept"),
]
