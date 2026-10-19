# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson info-cross-entropy-kl: cross_entropy, the surprise in bits when you expect q and the truth is
p, and kl, the extra surprise above the entropy of p. And the data: the species of the Palmer penguins, against three
equal shares and against the shares of one island; with the picture of a fair coin that you expect to land heads with
probability q."""
import math
import re
from functools import cache

import numpy as np

from everything_ai import paths
from everything_ai.check import FunctionBox, Mistake

PENGUINS = "data/penguins.csv"   # 344 penguins of the Palmer Archipelago: see data/README.md
SPECIES = ("Adelie", "Gentoo", "Chinstrap")


@cache
def _table():
    import pandas as pd
    return pd.read_csv(paths.ROOT / PENGUINS)


def species_counts(island: str | None = None) -> list:
    """How many penguins of each species, on all islands or on one."""
    d = _table()
    if island:
        d = d[d["island"] == island]
    counts = d["species"].value_counts()
    return [int(counts.get(s, 0)) for s in SPECIES]


def bits(p, q) -> float:
    """The cross-entropy H(p, q) in bits; infinite when q gives 0 to an outcome that p has."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    keep = p > 0
    if np.any(q[keep] == 0):
        return math.inf
    return float(-(p[keep] * np.log2(q[keep])).sum())


def coin(p_heads: float, q_heads: float) -> float:
    return bits([p_heads, 1 - p_heads], [q_heads, 1 - q_heads])


def number(x: float, digits: int, lang: str) -> str:
    s = f"{x:.{digits}f}"
    return s.replace(".", ",") if lang == "nl" else s


# --- the picture: a fair coin, coded as if it lands heads with probability q ----------------------------------------------

def coin_picture(labels: dict) -> str:
    """The cross-entropy of a fair coin against the q you expect, with the entropy of the fair coin, 1 bit, as the
    floor, and at q = 0.9 the gap between them: the KL divergence."""
    from everything_ai import chart
    qs = [0.02 + 0.96 * i / 200 for i in range(201)]
    panel = chart.Panel((0, 1), (0, 3.2), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    panel.line(qs, [coin(0.5, q) for q in qs], "loss", 2.6)
    panel.line([0, 1], [1, 1], "ink-3", 1.2, dashed=True)
    panel.text(0.05, 0.86, labels["floor"], "ink-2", "start")
    top = coin(0.5, 0.9)
    panel.arrow(0.9, 1.0, 0.9, top, "loss", 1.8)
    panel.text(0.93, 0.78, labels["gap"].format(kl=labels["number"](top - 1, 2)), "loss", "end")   # below the floor, free of the curve
    panel.points([0.5], [1.0], "loss", 5)
    return chart.figure([panel], alt=labels["alt"], width=520, height=300)


# The lesson is done when these are correct. "when-equal" is the choice question in the lesson.
REQUIRED = ["when-equal", "cross-entropy", "kl"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _num(v) -> str:
    if isinstance(v, (list, tuple, np.ndarray)) and np.ndim(v) > 0:   # a forgotten sum: one number per outcome
        return "[" + ", ".join(_num(x) for x in np.ravel(np.asarray(v, dtype=object))) + "]"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return repr(v)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    text = f"{round(x, 4) + 0.0:.4f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _close(got, want) -> bool:
    try:
        g, w = float(got), float(want)
    except (TypeError, ValueError):
        return False
    if math.isinf(w):
        return g == w
    return math.isclose(g, w, rel_tol=1e-7, abs_tol=1e-9)


class _Vector(np.ndarray):
    """Probabilities that a check message writes the way the lesson's code does: np.array([...]). The learner's code
    gets a plain copy."""

    def __repr__(self):
        return f"np.array({[int(v) if float(v) == int(v) else float(v) for v in np.asarray(self).tolist()]!r})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _v(values) -> np.ndarray:
    return np.array(values, dtype=float).view(_Vector)


_P1, _Q1 = _v([0.5, 0.25, 0.125, 0.125]), _v([0.25, 0.125, 0.5, 0.125])   # not symmetric: swapping p and q gives another number
_CASES_HIDDEN = [(_v([0.5, 0.5]), _v([0.9, 0.1])), (_v([0.9, 0.1]), _v([0.5, 0.5])), (_v([0.2, 0.3, 0.5]), _v([0.2, 0.3, 0.5])),
                 (_v([1.0, 0.0]), _v([0.25, 0.75])), (_v([0.5, 0.0, 0.5]), _v([0.25, 0.5, 0.25])),
                 (_v([0.5, 0.0, 0.5]), _v([0.5, 0.0, 0.5]))]

_NAN = Mistake("zero-probability", matches=lambda got, text: isinstance(got, (float, np.floating)) and math.isnan(got),
               en="`{call}` gives `nan`, short for not a number. An outcome with probability 0 in `p` gives 0 times a log, and the log of 0 is minus infinity: 0 times minus infinity is not a number. Such an outcome never happens, so leave it out.",
               nl="`{call}` geeft `nan`, kort voor not a number: geen getal. Een uitkomst met kans 0 in `p` geeft 0 keer een log, en de log van 0 is min oneindig: 0 keer min oneindig is geen getal. Zo'n uitkomst gebeurt nooit, dus laat haar weg.")

_NO_SUM = {
    "en": "`{call}` gives `{got}`: one number for each outcome. The {what} is one number: add them up, with `np.sum`.",
    "nl": "`{call}` geeft `{got}`: één getal voor elke uitkomst. De {what_nl} is één getal: tel ze op, met `np.sum`.",
}


def _no_sum(what: str, what_nl: str) -> Mistake:
    return Mistake("no-sum", matches=lambda got, text: isinstance(got, (list, tuple, np.ndarray)) and np.size(got) > 1,
                   en=_NO_SUM["en"].replace("{what}", what), nl=_NO_SUM["nl"].replace("{what_nl}", what_nl))


# --- exercise 2: the cross-entropy ---------------------------------------------------------------------------------------

_X_HEAD = "import numpy as np\n\n\ndef cross_entropy(p, q):\n"
_X_START = _X_HEAD + "    # the surprise in bits when you expect q and the truth is p\n    ...\n"


def _x(body: str = "    keep = p > 0\n    return -np.sum(p[keep] * np.log2(q[keep]))\n") -> str:
    return _X_HEAD + body


# --- exercise 3: the KL divergence ---------------------------------------------------------------------------------------

_K_HEAD = "import numpy as np\n\n\ndef kl(p, q):\n"
_K_START = _K_HEAD + "    # the extra surprise in bits, above the entropy of p, when you expect q and the truth is p\n    ...\n"


def _k(body: str = "    keep = p > 0\n    return np.sum(p[keep] * np.log2(p[keep] / q[keep]))\n") -> str:
    return _K_HEAD + body


EXERCISES = {
    "cross-entropy": FunctionBox(
        name="cross-entropy",
        function="cross_entropy",
        start=_X_START,
        solution=_x(),
        visible=[(_P1, _Q1)],
        hidden=_CASES_HIDDEN,
        show=_num,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            _NAN,
            _no_sum("cross-entropy", "cross-entropy"),
            Mistake("swapped", wrong=_x("    keep = q > 0\n    return -np.sum(q[keep] * np.log2(p[keep]))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `p` and `q` have swapped places. The truth `p` says how often each outcome comes, so it does the weighing; what you expect, `q`, decides the surprise, so it goes in the log.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `p` en `q` zijn van plaats gewisseld. De waarheid `p` zegt hoe vaak elke uitkomst komt, dus die weegt; wat je verwacht, `q`, bepaalt de verrassing, dus die staat in de log."),
            Mistake("entropy", wrong=_x("    keep = p > 0\n    return -np.sum(p[keep] * np.log2(p[keep]))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the entropy of `p`, as if you expected the truth. The surprise comes from what you expect: the log of `q`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is de entropie van `p`, alsof je de waarheid verwachtte. De verrassing komt van wat je verwacht: de log van `q`."),
            Mistake("nats", wrong=_x("    keep = p > 0\n    return -np.sum(p[keep] * np.log(q[keep]))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is in nats, with the natural log. In bits the log has base 2: `np.log2`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is in nats, met de natuurlijke log. In bits heeft de log grondtal 2: `np.log2`."),
            Mistake("sign", matches=lambda got, text: isinstance(got, (float, np.floating)) and got < 0,
                    en="`{call}` gives `{got}`, and it should give `{want}`: below 0, and a cross-entropy never is. Each surprise, minus the log of a probability, is 0 or more: where does the minus go?",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: onder 0, en een cross-entropy is dat nooit. Elke verrassing, min de log van een kans, is 0 of meer: waar staat de min?"),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: the surprise under `q`, weighted by `p`, added up.",
                      "nl": "Vervang de `...` door de body van de functie: de verrassing onder `q`, gewogen met `p`, opgeteld."},
            "arguments": {"en": "The check calls `{call}`, with two arrays of probabilities, and the `def` line of your function has another number of names between its parentheses. Keep `(p, q)`.",
                          "nl": "De check roept `{call}` aan, met twee arrays kansen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(p, q)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. For each outcome: what is its surprise under `q`, and how often does it come under `p`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Bij elke uitkomst: wat is de verrassing onder `q`, en hoe vaak komt ze onder `p`?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work when `p` has a probability of 0?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie als `p` een kans van 0 heeft?"},
            "correct": {"en": "Your cross-entropy is right in all {n} cases: never below the entropy of `p`, and equal to it when `q` is `p`.",
                        "nl": "Je cross-entropy klopt in alle {n} gevallen: nooit onder de entropie van `p`, en gelijk eraan als `q` gelijk is aan `p`."},
        },
    ),
    "kl": FunctionBox(
        name="kl",
        function="kl",
        start=_K_START,
        solution=_k(),
        visible=[(_P1, _Q1)],
        hidden=_CASES_HIDDEN,
        show=_num,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            _NAN,
            _no_sum("KL divergence", "KL-divergentie"),
            Mistake("reversed", wrong=_k("    keep = q > 0\n    return np.sum(q[keep] * np.log2(q[keep] / p[keep]))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the KL divergence the other way round, as if `q` were the truth. KL is not symmetric: weigh by the truth `p`, and put `p / q` in the log.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is de KL-divergentie de andere kant op, alsof `q` de waarheid is. KL is niet symmetrisch: weeg met de waarheid `p`, en zet `p / q` in de log."),
            Mistake("flipped-log", wrong=_k("    keep = p > 0\n    return np.sum(p[keep] * np.log2(q[keep] / p[keep]))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the sign is wrong, so the result is 0 or below. KL is never negative: in the log comes `p / q`, the truth over what you expect.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: het teken klopt niet, dus de uitkomst is 0 of lager. KL is nooit negatief: in de log staat `p / q`, de waarheid gedeeld door wat je verwacht."),
            Mistake("cross-entropy", wrong=_k("    keep = p > 0\n    return -np.sum(p[keep] * np.log2(q[keep]))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the whole cross-entropy. KL is only the extra part: take the entropy of `p` off.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is de hele cross-entropy. KL is alleen het extra deel: trek de entropie van `p` eraf."),
            Mistake("nats", wrong=_k("    keep = p > 0\n    return np.sum(p[keep] * np.log(p[keep] / q[keep]))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is in nats, with the natural log. In bits the log has base 2: `np.log2`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is in nats, met de natuurlijke log. In bits heeft de log grondtal 2: `np.log2`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: the cross-entropy minus the entropy of `p`, or the same in one sum.",
                      "nl": "Vervang de `...` door de body van de functie: de cross-entropy min de entropie van `p`, of hetzelfde in één som."},
            "arguments": {"en": "The check calls `{call}`, with two arrays of probabilities, and the `def` line of your function has another number of names between its parentheses. Keep `(p, q)`.",
                          "nl": "De check roept `{call}` aan, met twee arrays kansen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(p, q)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Work out the cross-entropy and the entropy of `p` by hand: how much bigger is the first?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken de cross-entropy en de entropie van `p` met de hand uit: hoeveel groter is de eerste?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work when `p` has a probability of 0, and when `q` is `p`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie als `p` een kans van 0 heeft, en als `q` gelijk is aan `p`?"},
            "correct": {"en": "Your KL divergence is right in all {n} cases: 0 when `q` is `p`, and not the same in both directions.",
                        "nl": "Je KL-divergentie klopt in alle {n} gevallen: 0 als `q` gelijk is aan `p`, en niet hetzelfde in beide richtingen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("cross-entropy", _X_START, "empty", None),
    ("cross-entropy", _x(), "correct", None),
    ("cross-entropy", _x("    total = 0\n    for i in range(len(p)):\n        if p[i] > 0:\n            total = total - p[i] * np.log2(q[i])\n    return total\n"), "correct", None),
    ("cross-entropy", _x("    return -np.sum(p * np.log2(q))\n"), "wrong", "zero-probability"),
    ("cross-entropy", _x("    keep = q > 0\n    return -np.sum(q[keep] * np.log2(p[keep]))\n"), "wrong", "swapped"),
    ("cross-entropy", _x("    keep = p > 0\n    return -np.sum(p[keep] * np.log2(p[keep]))\n"), "wrong", "entropy"),
    ("cross-entropy", _x("    keep = p > 0\n    return -np.sum(p[keep] * np.log(q[keep]))\n"), "wrong", "nats"),
    ("cross-entropy", _x("    return -np.sum(q * np.log2(p))\n"), "wrong", None),   # swapped, without leaving out zeros: told on the example
    ("cross-entropy", _x("    keep = p > 0\n    return -(p[keep] * np.log2(q[keep]))\n"), "wrong", "no-sum"),
    ("cross-entropy", _x("    keep = p > 0\n    return np.sum(p[keep] * np.log2(q[keep]))\n"), "wrong", "sign"),
    ("kl", _K_START, "empty", None),
    ("kl", _k(), "correct", None),
    ("kl", _k("    keep = p > 0\n    p, q = p[keep], q[keep]\n    return -np.sum(p * np.log2(q)) + np.sum(p * np.log2(p))\n"), "correct", None),
    ("kl", _k("    return np.sum(p * np.log2(p / q))\n"), "wrong", "zero-probability"),
    ("kl", _k("    keep = q > 0\n    return np.sum(q[keep] * np.log2(q[keep] / p[keep]))\n"), "wrong", "reversed"),
    ("kl", _k("    keep = p > 0\n    return np.sum(p[keep] * np.log2(q[keep] / p[keep]))\n"), "wrong", "flipped-log"),
    ("kl", _k("    keep = p > 0\n    return -np.sum(p[keep] * np.log2(q[keep]))\n"), "wrong", "cross-entropy"),
    ("kl", _k("    keep = p > 0\n    return np.sum(p[keep] * np.log(p[keep] / q[keep]))\n"), "wrong", "nats"),
    ("kl", _k("    keep = p > 0\n    return p[keep] * np.log2(p[keep] / q[keep])\n"), "wrong", "no-sum"),
]
