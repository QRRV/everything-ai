# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-bias-variance: the learning curve of a curve of degree 6, trained on more and more of a
training set. And the pictures: twenty training sets of temperatures made by the rule of ml-overfitting, each with a
straight line and a curve of degree 9 fitted on it, for a growing number of temperatures per set; and the learning
curves of the worked example."""
import functools
import re

from everything_ai.check import Code, CodeBox, Mistake

NOISE = 0.75                      # the noise is uniform between -1.5 and 1.5 °C: its variance, 3² / 12, is the MSE no model gets below
SETS, POOL = 20, 400              # twenty training sets, each the first n of its own 400 temperatures
SET_SEED = 17
SIZES = [12, 25, 50, 100, 200, 400]   # the stops of the slider: 12 as in ml-overfitting, then about twice as many each time
DEGREES = (1, 9)                  # a straight line, and the curve of degree 9 that overfits in ml-overfitting
THREE = 15.0                      # three in the afternoon, where the rule is highest: 20 °C
EXAMPLE_SEED, EXAMPLE_N, EXAMPLE_TRAIN = 75, 300, 200   # the worked example: 200 temperatures to train on, 100 to validate
CURVE_SIZES = [12, 25, 50, 100, 200]


def rule(h):
    """The temperature the rule of ml-overfitting gives at hour h: highest at three in the afternoon."""
    return 20 - 0.05 * (h - 15) ** 2


@functools.lru_cache(maxsize=1)
def _pools():
    """Twenty sets of 400 temperatures, each read at random hours of a day like the one of ml-overfitting. Training
    set k with n temperatures is the first n of pool k, so a set grows when the slider moves on."""
    import numpy as np
    rng = np.random.default_rng(SET_SEED)
    pools = []
    for _ in range(SETS):
        hours = 24 * rng.random(POOL)
        pools.append((hours, rule(hours) + 3 * (rng.random(POOL) - 0.5)))
    return pools


@functools.lru_cache(maxsize=None)
def fits(degree: int, n: int) -> tuple:
    """The weights of the curve of this degree on each of the twenty training sets of n temperatures."""
    import numpy as np
    return tuple(tuple(np.polyfit(h[:n], t[:n], degree)) for h, t in _pools())


def at_three(degree: int, n: int) -> list[float]:
    """What each of the twenty fits says at three in the afternoon, where the rule gives 20 °C."""
    import numpy as np
    return [float(np.polyval(w, THREE)) for w in fits(degree, n)]


def spread(degree: int, n: int) -> tuple[float, float]:
    """The lowest and the highest of the twenty temperatures at three in the afternoon, rounded to tenths."""
    p = at_three(degree, n)
    return round(min(p), 1), round(max(p), 1)


_GRID = [24 * i / 96 for i in range(97)]   # every quarter of an hour


def _fit_panel(degree: int, n: int, title: str, labels: dict[str, str]):
    import numpy as np
    from everything_ai import chart
    p = chart.Panel((0, 24), (6, 24), title=title, xlabel=labels["hour"], ylabel=labels["temp"], xticks=[0, 6, 12, 18, 24])
    hours = [0, 24] if degree == 1 else _GRID   # a straight line needs only its two ends
    for w in fits(degree, n):
        # far outside the chart a curve of degree 9 runs to millions: those parts are cut off, so drawn at its edge
        p.line(hours, [min(max(float(v), -20.0), 50.0) for v in np.polyval(w, hours)], "pred", 1.1)
    p.line(_GRID, [rule(h) for h in _GRID], "data", 2.4, dashed=True)
    p.points([THREE], [rule(THREE)], "data", 4.5)
    return p


def picture(n: int, labels: dict[str, str]) -> str:
    """Two charts side by side: the twenty straight lines and the twenty curves of degree 9, each fitted on its own
    training set of n temperatures, with the rule dashed and its value at three in the afternoon as a dot. Drawn as SVG
    text, once for every stop of the slider (L.explore)."""
    from everything_ai import chart
    return chart.figure([_fit_panel(1, n, labels["line"], labels), _fit_panel(9, n, labels["nine"], labels)],
                        alt=labels["alt"], width=330, height=270)


def far_off(degree: int, n: int) -> int:
    """How many of the twenty fits leave the chart, from 6 to 24 °C, somewhere between midnight and midnight."""
    import numpy as np
    return sum(1 for w in fits(degree, n) if not all(6 <= float(v) <= 24 for v in np.polyval(w, _GRID)))


@functools.lru_cache(maxsize=1)
def _example():
    """The 300 temperatures of the worked example: the first 200 to train on, the last 100 to validate."""
    import numpy as np
    rng = np.random.default_rng(EXAMPLE_SEED)
    hours = 24 * rng.random(size=EXAMPLE_N)
    temps = 20 - 0.05 * (hours - 15) ** 2 + 3 * (rng.random(size=EXAMPLE_N) - 0.5)
    return hours[:EXAMPLE_TRAIN], temps[:EXAMPLE_TRAIN], hours[EXAMPLE_TRAIN:], temps[EXAMPLE_TRAIN:]


@functools.lru_cache(maxsize=None)
def learning_curve(degree: int) -> tuple:
    """The worked example's learning curve: for every n in CURVE_SIZES, the MSE of the curve of this degree trained on
    the first n training temperatures, on those n and on the 100 validation temperatures."""
    import numpy as np
    x, y, xv, yv = _example()
    out = []
    for n in CURVE_SIZES:
        w = np.polyfit(x[:n], y[:n], degree)
        out.append((float(np.mean((y[:n] - np.polyval(w, x[:n])) ** 2)), float(np.mean((yv - np.polyval(w, xv)) ** 2))))
    return tuple(out)


def curves_picture(labels: dict[str, str]) -> str:
    """The learning curves of the worked example, degree 1 and degree 9 side by side, both axes on a log scale: the
    training MSE solid with filled dots, the validation MSE dashed with open dots, each named at its line in the chart of
    degree 9, and the noise as a grey dashed line."""
    from everything_ai import chart
    panels = []
    for degree, title in ((1, labels["line"]), (9, labels["nine"])):
        train = [a for a, _ in learning_curve(degree)]
        val = [b for _, b in learning_curve(degree)]
        p = chart.Panel((10, 240), (0.1, 100), title=title, xlabel=labels["n"], ylabel=labels["mse"], xticks=CURVE_SIZES,
                        xlog=True, ylog=True)
        p.line([10, 240], [NOISE, NOISE], "ink-3", 1.2, dashed=True)
        if degree == 1:   # once, where no curve comes near it; on the right the curves of degree 9 run along it
            p.text(11, NOISE * 1.6, labels["noise"], "ink-2", anchor="start")
        p.line(CURVE_SIZES, train, "loss", 2.2)
        p.line(CURVE_SIZES, val, "loss", 2.2, dashed=True)
        p.points(CURVE_SIZES, train, "loss", 4)
        p.points(CURVE_SIZES, val, "loss", 4, hollow=True)
        if degree == 9:   # here the two lines start far apart: each name next to its first dot, clear of the other line
            p.text(15, val[0], labels["val"], "loss", anchor="start")
            p.text(11, train[0] / 1.9, labels["train"], "loss", anchor="start")
        panels.append(p)
    return chart.figure(panels, alt=labels["alt"], width=330, height=270)


# The lesson is done when these are correct. "bias-or-variance" and "more-data" are the choice questions in the lesson.
REQUIRED = ["bias-or-variance", "learning-curve", "more-data"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# 150 temperatures by the same rule: the first 100 to train on, the last 50 to validate, as in the worked example.
_HEAD = ("import numpy as np\n\nrng = np.random.default_rng(76)\nhours = 24 * rng.random(size=150)\n"
         "temps = 20 - 0.05 * (hours - 15) ** 2 + 3 * (rng.random(size=150) - 0.5)\n"
         "x_train, y_train = hours[:100], temps[:100]\nx_val, y_val = hours[100:], temps[100:]\n"
         "degree = 6\ntrain_mses = []\nval_mses = []\nfor n in [10, 20, 50, 100]:\n")
_VAL = "    val_mses.append(np.mean((y_val - np.polyval(w, x_val)) ** 2))\n"
_LAST = "train_mses, val_mses"
_START = _HEAD + "    w = ...\n    train_mses.append(...)\n" + _VAL + _LAST + "\n"
_FIT = "    w = np.polyfit(x_train[:n], y_train[:n], degree)\n"
_TRAIN = "    train_mses.append(np.mean((y_train[:n] - np.polyval(w, x_train[:n])) ** 2))\n"
_SOLUTION = _HEAD + _FIT + _TRAIN + _VAL + _LAST


def _lc(fit: str = _FIT, train: str = _TRAIN, val: str = _VAL, last: str = _LAST) -> str:
    return _HEAD + fit + train + val + last


@functools.lru_cache(maxsize=None)
def box_curve(degree: int = 6) -> tuple:
    """What the solution in the box gives with this degree: the training MSEs and the validation MSEs."""
    import numpy as np
    rng = np.random.default_rng(76)
    hours = 24 * rng.random(size=150)
    temps = 20 - 0.05 * (hours - 15) ** 2 + 3 * (rng.random(size=150) - 0.5)
    train, val = [], []
    for n in [10, 20, 50, 100]:
        w = np.polyfit(hours[:n], temps[:n], degree)
        train.append(float(np.mean((temps[:n] - np.polyval(w, hours[:n])) ** 2)))
        val.append(float(np.mean((temps[100:] - np.polyval(w, hours[100:])) ** 2)))
    return train, val


def _swapped(got, text) -> bool:
    """The two right lists, but the other way round: first the validation MSEs."""
    from everything_ai.check import same_value
    train, val = box_curve()
    return same_value(got, (val, train))


def _stops(e: BaseException) -> bool:
    return "same length" in str(e)


def _broadcast(e: BaseException) -> bool:
    return "could not be broadcast" in str(e)


_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
    "empty": {"en": "Replace the two `...`: the weights of a curve of degree `degree` fitted on the first `n` training temperatures, and its MSE on those `n`.",
              "nl": "Vervang de twee `...`: de weights van een curve van graad `degree`, geleerd van de eerste `n` trainingstemperaturen, en zijn MSE op die `n`."},
    "keep": {"en": "Your code may be right, but the check needs the line `{line}` as it was: put it back. The check tries degree 1 itself; try other degrees in the scratchpad.",
             "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. Graad 1 probeert de check zelf; andere graden probeer je in het kladblok."},
    "no_value": {"en": "The last line of the box shows nothing. Put `train_mses, val_mses` on the last line, so the check can see them.",
                 "nl": "De laatste regel van het vak laat niets zien. Zet `train_mses, val_mses` op de laatste regel, zodat de check ze kan zien."},
    "wrong": {"en": "Your last line gives `{got}`. Do you fit with `np.polyfit` on the first `n` training temperatures, `x_train[:n]` and `y_train[:n]`, and take the mean of the squared residuals on those same `n`?",
              "nl": "Je laatste regel geeft `{got}`. Laat je `np.polyfit` leren van de eerste `n` trainingstemperaturen, `x_train[:n]` en `y_train[:n]`, en neem je het gemiddelde van de gekwadrateerde residuen op diezelfde `n`?"},
    "other": {"en": "Right for degree 6. But with {where}, your last line gives `{got}`, and it should give about `{want}`. Do you use `degree` itself in `np.polyfit`?",
              "nl": "Goed voor graad 6. Maar met {where} geeft je laatste regel `{got}`, en dat moet ongeveer `{want}` zijn. Gebruik je `degree` zelf in `np.polyfit`?"},
    "other_error": {"en": "Right for degree 6. But with {where}, your code stops with an error. Do you use `degree` itself in `np.polyfit`?",
                    "nl": "Goed voor graad 6. Maar met {where} stopt je code met een fout. Gebruik je `degree` zelf in `np.polyfit`?"},
    "correct": {"en": "With degree 6, the training MSE climbs from about 0.08 to 0.76 and the validation MSE falls from about 1.99 to 0.78: at 100 temperatures they are close together, near the noise. The extra temperatures took the variance away. The check also tried degree 1: there both MSEs stay between about 5.1 and 7.4, far above the noise, whatever `n` is.",
                "nl": "Met graad 6 klimt de training-MSE van ongeveer 0,08 naar 0,76 en daalt de validation-MSE van ongeveer 1,99 naar 0,78: bij 100 temperaturen liggen ze dicht bij elkaar, vlak bij de ruis. De extra temperaturen namen de variantie weg. De check probeerde ook graad 1: daar blijven beide MSE's tussen ongeveer 5,1 en 7,4, ver boven de ruis, wat `n` ook is."},
}

EXERCISES = {
    "learning-curve": CodeBox(
        name="learning-curve",
        start=_START,
        solution=_SOLUTION,
        digits=2,
        keep=["degree = 6"],
        # The check also runs the box with degree 1: numbers typed in by hand do not change with it.
        others=[{"degree": Code("1")}],
        first=[_DOTS],
        mistakes=[
            Mistake("lengths", error=_stops,
                    en="Python stops: `np.polyfit` gets hours and temperatures of different lengths. Take the first `n` of both: `x_train[:n]` and `y_train[:n]`.",
                    nl="Python stopt: `np.polyfit` krijgt uren en temperaturen van verschillende lengte. Neem van allebei de eerste `n`: `x_train[:n]` en `y_train[:n]`."),
            # before "broadcast": np.polyval(x, w) gives 7 values, so the subtraction stops in the same way
            Mistake("polyval-order", syntax=r"np\.polyval\(\s*x_train",
                    en="Python stops: `np.polyval` takes the weights first and then the hours, as in the line below it: `np.polyval(w, x_train[:n])`.",
                    nl="Python stopt: `np.polyval` krijgt eerst de weights en dan de uren, zoals in de regel eronder: `np.polyval(w, x_train[:n])`."),
            Mistake("broadcast", error=_broadcast,
                    en="Python stops: you subtract two arrays of different lengths. In the training MSE, take the first `n` of both: `y_train[:n]` and `x_train[:n]`.",
                    nl="Python stopt: je trekt twee arrays van verschillende lengte van elkaar af. Neem in de training-MSE van allebei de eerste `n`: `y_train[:n]` en `x_train[:n]`."),
            # before the others: the right two lists, but the last line shows them the other way round
            Mistake("swapped", matches=_swapped,
                    en="Your last line gives the two right lists, but the other way round: first the validation MSEs. Make the last line `train_mses, val_mses` again: that is how the check reads them.",
                    nl="Je laatste regel geeft de twee goede lijsten, maar andersom: eerst de validation-MSE's. Maak van de laatste regel weer `train_mses, val_mses`: zo leest de check ze."),
            Mistake("fit-all", wrong=_lc(fit="    w = np.polyfit(x_train, y_train, degree)\n"),
                    en="Your last line gives `{got}`: the validation MSE is the same for every `n`, because every curve learned from all 100 training temperatures. Fit on the first `n` only: `x_train[:n]` and `y_train[:n]`.",
                    nl="Je laatste regel geeft `{got}`: de validation-MSE is voor elke `n` hetzelfde, omdat elke curve van alle 100 trainingstemperaturen leerde. Laat hem alleen van de eerste `n` leren: `x_train[:n]` en `y_train[:n]`."),
            Mistake("train-all", wrong=_lc(train="    train_mses.append(np.mean((y_train - np.polyval(w, x_train)) ** 2))\n"),
                    en="Your last line gives `{got}`: your training MSE is measured on all 100 training temperatures, also the ones the curve did not learn from. Measure it on the first `n`, the ones it trained on: `y_train[:n]` and `x_train[:n]`.",
                    nl="Je laatste regel geeft `{got}`: je training-MSE is gemeten op alle 100 trainingstemperaturen, ook op die waarvan de curve niet leerde. Meet hem op de eerste `n`, die waarop hij trainde: `y_train[:n]` en `x_train[:n]`."),
            Mistake("train-on-val", wrong=_lc(train="    train_mses.append(np.mean((y_val - np.polyval(w, x_val)) ** 2))\n"),
                    en="Your last line gives `{got}`: the two lists are the same, because your training MSE is measured on the validation set. Measure it on the first `n` training temperatures: `y_train[:n]` and `x_train[:n]`.",
                    nl="Je laatste regel geeft `{got}`: de twee lijsten zijn gelijk, omdat je training-MSE op de validation set gemeten is. Meet hem op de eerste `n` trainingstemperaturen: `y_train[:n]` en `x_train[:n]`."),
            Mistake("no-square", wrong=_lc(train="    train_mses.append(np.mean(y_train[:n] - np.polyval(w, x_train[:n])))\n"),
                    en="Your last line gives `{got}`: your training MSE is the mean residual, without squaring, so positive and negative residuals cancel. Square each residual first: `** 2`.",
                    nl="Je laatste regel geeft `{got}`: je training-MSE is het gemiddelde residu, zonder kwadraat, dus positieve en negatieve residuen vallen tegen elkaar weg. Kwadrateer eerst elk residu: `** 2`."),
        ],
        texts=_TEXTS,
    ),
}


# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("learning-curve", _START, "empty", None),
    ("learning-curve", _SOLUTION, "correct", None),
    ("learning-curve", _lc(fit="    w = np.polyfit(x_train[:n], y_train[:n], deg=degree)\n",
                           train="    pred = np.polyval(w, x_train[:n])\n    train_mses.append(((pred - y_train[:n]) ** 2).mean())\n"), "correct", None),
    ("learning-curve", _lc(train="    train_mses.append(np.sum((y_train[:n] - np.polyval(w, x_train[:n])) ** 2) / n)\n"), "correct", None),
    ("learning-curve", _lc(last="val_mses, train_mses"), "wrong", "swapped"),
    ("learning-curve", _lc(fit="    w = np.polyfit(x_train, y_train, degree)\n"), "wrong", "fit-all"),
    ("learning-curve", _lc(train="    train_mses.append(np.mean((y_train - np.polyval(w, x_train)) ** 2))\n"), "wrong", "train-all"),
    ("learning-curve", _lc(train="    train_mses.append(np.mean((y_val - np.polyval(w, x_val)) ** 2))\n"), "wrong", "train-on-val"),
    ("learning-curve", _lc(train="    train_mses.append(np.mean(y_train[:n] - np.polyval(w, x_train[:n])))\n"), "wrong", "no-square"),
    ("learning-curve", _lc(fit="    w = np.polyfit(x_train[:n], y_train, degree)\n"), "error", "lengths"),
    ("learning-curve", _lc(train="    train_mses.append(np.mean((y_train[:n] - np.polyval(w, x_train)) ** 2))\n"), "error", "broadcast"),
    ("learning-curve", _lc(train="    train_mses.append(np.mean((y_train[:n] - np.polyval(x_train[:n], w)) ** 2))\n"), "error", "polyval-order"),
    ("learning-curve", _lc(fit="    w = np.polyfit(x_train[:n], y_train[:n], 6)\n"), "almost", "other-value"),
    ("learning-curve", _SOLUTION.replace("degree = 6", "degree = 2"), "wrong", "changed-line"),
    ("learning-curve", _lc(fit="    w = ...\n"), "wrong", "dots-left"),
]
