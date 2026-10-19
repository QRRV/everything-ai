# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson stats-mle: avg_nll, the average negative log-likelihood of a probability p for data of zeros
and ones, and best_p, the p from a grid that makes the data most likely. And the picture: the average negative
log-likelihood of p for a coin that landed heads 7 times in 10."""
import math
import re

import numpy as np

from everything_ai.check import FunctionBox, Mistake


def avg_nll(p: float, data) -> float:
    x = np.asarray(data, dtype=float)
    return float(-np.mean(x * np.log(p) + (1 - x) * np.log(1 - p)))


def number(x: float, digits: int, lang: str) -> str:
    s = f"{x:.{digits}f}"
    return s.replace(".", ",") if lang == "nl" else s


# --- the picture: the average NLL of p for 7 heads in 10 tosses ------------------------------------------------------------

def coin_picture(labels: dict) -> str:
    from everything_ai import chart
    data = [1] * 7 + [0] * 3
    ps = [0.02 + 0.96 * i / 200 for i in range(201)]
    panel = chart.Panel((0, 1), (0, 2.6), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    panel.line(ps, [avg_nll(p, data) for p in ps], "loss", 2.6)
    best = avg_nll(0.7, data)
    panel.points([0.7], [best], "loss", 5)
    panel.text(0.7, best - 0.22, labels["best"].format(v=labels["number"](best, 3)), "loss")
    panel.points([0.5], [avg_nll(0.5, data)], "ink-3", 4)
    panel.text(0.48, avg_nll(0.5, data) - 0.25, labels["fair"].format(v=labels["number"](avg_nll(0.5, data), 3)), "ink-2", "end")   # under the curve
    return chart.figure([panel], alt=labels["alt"], width=520, height=300)


# The lesson is done when these are correct. "why-log" is the choice question in the lesson.
REQUIRED = ["why-log", "avg-nll", "best-p"]

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
        return math.isclose(float(got), float(want), rel_tol=1e-7, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


class _Vector(np.ndarray):
    """Numbers that a check message writes the way the lesson's code does: np.array([...]). The learner's code gets a
    plain copy."""

    def __repr__(self):
        return f"np.array({[int(v) if float(v) == int(v) else round(float(v), 4) for v in np.asarray(self).tolist()]!r})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _v(values) -> np.ndarray:
    return np.array(values, dtype=float).view(_Vector)


# --- exercise 2: the average negative log-likelihood -----------------------------------------------------------------------

_N_HEAD = "import numpy as np\n\n\ndef avg_nll(p, data):\n"
_N_START = _N_HEAD + "    # the average negative log-likelihood of p, for an array of zeros and ones\n    ...\n"


def _n(body: str = "    return -np.mean(data * np.log(p) + (1 - data) * np.log(1 - p))\n") -> str:
    return _N_HEAD + body


# --- exercise 3: the p that makes the data most likely --------------------------------------------------------------------

_B_HEAD = "import numpy as np\n\n\ndef best_p(data, grid):\n"
_B_START = _B_HEAD + "    # the p from grid with the lowest average negative log-likelihood for data\n    ...\n"
_B_NLL = "        nll = -np.mean(data * np.log(p) + (1 - data) * np.log(1 - p))\n"


def _b(compare: str = "nll < best_nll", keep: str = "best = p", first: str = "    best = grid[0]\n    best_nll = np.inf\n") -> str:
    return (_B_HEAD + first + "    for p in grid:\n" + _B_NLL + f"        if {compare}:\n            {keep}\n            best_nll = nll\n"
            "    return best\n")


_GRID = _v([0.1, 0.3, 0.5, 0.7, 0.9])
_FINE = _v([round(0.05 * i, 2) for i in range(1, 20)])

EXERCISES = {
    "avg-nll": FunctionBox(
        name="avg-nll",
        function="avg_nll",
        start=_N_START,
        solution=_n(),
        visible=[(0.75, _v([1, 0, 1, 1]))],
        hidden=[(0.5, _v([1, 0, 1, 1])), (0.2, _v([0, 0, 0, 1, 0])), (0.9, _v([1, 1, 1])), (0.4, _v([0, 1]))],
        show=_num,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("ones-only", wrong=_n("    return -np.mean(data * np.log(p))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the zeros cost nothing in your version. A zero happens with probability `1 - p`, so it costs `-np.log(1 - p)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de nullen kosten bij jou niets. Een nul heeft kans `1 - p`, dus die kost `-np.log(1 - p)`."),
            Mistake("no-minus", wrong=_n("    return np.mean(data * np.log(p) + (1 - data) * np.log(1 - p))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the average log-likelihood, 0 or below. The negative log-likelihood has a minus in front, so that lower is better.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is de gemiddelde log-likelihood, 0 of lager. De negative log-likelihood heeft een min ervoor, zodat lager beter is."),
            Mistake("sum", wrong=_n("    return -np.sum(data * np.log(p) + (1 - data) * np.log(1 - p))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the total, not the average. Divide by the number of examples, or use the mean.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is het totaal, niet het gemiddelde. Deel door het aantal voorbeelden, of neem het gemiddelde."),
            Mistake("log2", wrong=_n("    return -np.mean(data * np.log2(p) + (1 - data) * np.log2(1 - p))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is in bits, with `np.log2`. The negative log-likelihood here uses the natural log, `np.log`, and counts in nats.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is in bits, met `np.log2`. De negative log-likelihood gebruikt hier de natuurlijke log, `np.log`, en telt in nats."),
            Mistake("probability", wrong=_n("    return np.mean(data * p + (1 - data) * (1 - p))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the average probability of the data, not its negative log-likelihood. Take minus the log of each probability before you average.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is de gemiddelde kans van de data, niet de negative log-likelihood. Neem min de log van elke kans voordat je middelt."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: for each example the probability `p` gives it, minus its log, averaged.",
                      "nl": "Vervang de `...` door de body van de functie: voor elk voorbeeld de kans die `p` eraan geeft, min de log daarvan, gemiddeld."},
            "arguments": {"en": "The check calls `{call}`, with a probability and an array, and the `def` line of your function has another number of names between its parentheses. Keep `(p, data)`.",
                          "nl": "De check roept `{call}` aan, met een kans en een array, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(p, data)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. What probability does `p` give a one, and what does it give a zero?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Welke kans geeft `p` aan een één, en welke aan een nul?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every `p` between 0 and 1, and for data of any length?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke `p` tussen 0 en 1, en voor data van elke lengte?"},
            "correct": {"en": "Your function is right in all {n} cases: the average surprise of the data under `p`.",
                        "nl": "Je functie klopt in alle {n} gevallen: de gemiddelde verrassing van de data onder `p`."},
        },
    ),
    "best-p": FunctionBox(
        name="best-p",
        function="best_p",
        start=_B_START,
        solution=_b(),
        visible=[(_v([0, 1, 0, 0, 1, 0, 0, 0, 1, 0]), _GRID)],
        hidden=[(_v([0, 0, 0, 1]), _FINE), (_v([1, 0]), _GRID), (_v([1, 1, 1, 1, 0]), _FINE), (_v([0, 0, 1, 0, 0]), _GRID),
                (_v([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), _v([0.02, 0.2, 0.5, 0.8]))],
        show=_num,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("highest", wrong=_b(compare="nll > best_nll", first="    best = grid[0]\n    best_nll = -np.inf\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you keep the p with the highest negative log-likelihood, the one that makes the data least likely. Check the minus sign and which way your comparison goes.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je houdt de p met de hoogste negative log-likelihood, die de data het minst waarschijnlijk maakt. Kijk het minteken na, en welke kant je vergelijking op werkt."),
            Mistake("last", wrong=_b(compare="True"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the last p of the grid, whatever the data. Does every zero cost `-np.log(1 - p)` too, and do you keep a p only when its average is lower than the best so far?",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is de laatste p van het grid, wat de data ook is. Kost elke nul ook `-np.log(1 - p)`, en houd je een p alleen als zijn gemiddelde lager is dan de beste tot nu toe?"),
            Mistake("index", wrong=_B_HEAD + "    nlls = []\n    for p in grid:\n" + _B_NLL.replace("        nll = ", "        nlls.append(").rstrip("\n") + ")\n    return np.argmin(nlls)\n",
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the place of the best p in the grid, not the p itself. Look it up in `grid`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is de plek van de beste p in het grid, niet de p zelf. Zoek hem op in `grid`."),
            Mistake("share", wrong=_B_HEAD + "    return np.mean(data)\n",
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the share of ones itself, and it need not be in the grid. Give back the p of the grid with the lowest average negative log-likelihood.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is het aandeel enen zelf, en dat hoeft niet in het grid te staan. Geef de p uit het grid met de laagste gemiddelde negative log-likelihood terug."),
            Mistake("nearest", wrong=_B_HEAD + "    return grid[np.argmin(np.abs(grid - np.mean(data)))]\n",
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the p of the grid nearest to the share of ones. But the average negative log-likelihood is not the same on both sides: a p close to 0 or 1 costs a lot. Compare the averages themselves.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is de p uit het grid die het dichtst bij het aandeel enen ligt. Maar de gemiddelde negative log-likelihood is niet aan beide kanten gelijk: een p dicht bij 0 of 1 kost veel. Vergelijk de gemiddelden zelf."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: work out the average negative log-likelihood of every `p` in `grid`, and give back the `p` with the lowest.",
                      "nl": "Vervang de `...` door de body van de functie: reken de gemiddelde negative log-likelihood van elke `p` in `grid` uit, en geef de `p` met de laagste terug."},
            "arguments": {"en": "The check calls `{call}`, with the data and a grid of values of p, and the `def` line of your function has another number of names between its parentheses. Keep `(data, grid)`.",
                          "nl": "De check roept `{call}` aan, met de data en een grid van waarden van p, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(data, grid)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Which p of the grid gives this data the lowest average negative log-likelihood, and does your function give back that p itself?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Welke p uit het grid geeft deze data de laagste gemiddelde negative log-likelihood, en geeft je functie precies die p terug?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any data and any grid?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor alle data en elk grid?"},
            "correct": {"en": "Your function is right in all {n} cases: each time the p of the grid that makes the data most likely, the maximum likelihood estimate on that grid.",
                        "nl": "Je functie klopt in alle {n} gevallen: elke keer de p uit het grid die de data het waarschijnlijkst maakt, de maximum-likelihoodschatting op dat grid."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("avg-nll", _N_START, "empty", None),
    ("avg-nll", _n(), "correct", None),
    ("avg-nll", _n("    probs = np.where(data == 1, p, 1 - p)\n    return -np.mean(np.log(probs))\n"), "correct", None),
    ("avg-nll", _n("    return -np.mean(data * np.log(p))\n"), "wrong", "ones-only"),
    ("avg-nll", _n("    return np.mean(data * np.log(p) + (1 - data) * np.log(1 - p))\n"), "wrong", "no-minus"),
    ("avg-nll", _n("    return -np.sum(data * np.log(p) + (1 - data) * np.log(1 - p))\n"), "wrong", "sum"),
    ("avg-nll", _n("    return np.mean(data * p + (1 - data) * (1 - p))\n"), "wrong", "probability"),
    ("avg-nll", _n("    return -np.mean(data * np.log2(p) + (1 - data) * np.log2(1 - p))\n"), "wrong", "log2"),
    ("avg-nll", _n("    return -np.mean(data * np.log(1 - p) + (1 - data) * np.log(p))\n"), "wrong", None),
    ("best-p", _B_START, "empty", None),
    ("best-p", _b(), "correct", None),
    ("best-p", _B_HEAD + "    nlls = []\n    for p in grid:\n        nlls.append(-np.mean(data * np.log(p) + (1 - data) * np.log(1 - p)))\n    return grid[np.argmin(nlls)]\n", "correct", None),
    ("best-p", _b(compare="nll > best_nll", first="    best = grid[0]\n    best_nll = -np.inf\n"), "wrong", "highest"),
    ("best-p", _b(compare="True"), "wrong", "last"),
    ("best-p", _B_HEAD + "    nlls = []\n    for p in grid:\n        nlls.append(-np.mean(data * np.log(p) + (1 - data) * np.log(1 - p)))\n    return np.argmin(nlls)\n", "wrong", "index"),
    ("best-p", _B_HEAD + "    return np.mean(data)\n", "wrong", "share"),
    ("best-p", _B_HEAD + "    return grid[np.argmin(np.abs(grid - np.mean(data)))]\n", "wrong", "nearest"),
    ("best-p", _n() + "\n\n" + _B_HEAD + "    nlls = []\n    for p in grid:\n        nlls.append(avg_nll(p, data))\n    return grid[np.argmin(nlls)]\n", "correct", None),
]
