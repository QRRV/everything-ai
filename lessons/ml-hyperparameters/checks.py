# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-hyperparameters: sorting the numbers of a model into parameters, which training learns,
and hyperparameters, which you choose; and training a line with three learning rates and comparing the results."""
import math
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort

# The lesson is done when these are correct.
REQUIRED = ["who-sets", "three-rates"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_R_HEAD = ("import numpy as np\n\n\ndef fit(x, y, lr, steps):\n    w = 0.0\n    b = 0.0\n    for step in range(steps):\n"
           "        r = y - (w * x + b)\n        w = w - lr * (-2 * (r * x).mean())\n        b = b - lr * (-2 * r.mean())\n"
           "    return w, b\n\n\nsizes = np.array([0.6, 1.4, 0.9, 1.8, 1.1, 0.7, 1.5, 1.2, 0.5, 1.9])\n"
           "prices = np.array([150, 270, 200, 330, 230, 160, 290, 240, 130, 350])\nsteps = 200\nmses = []\n"
           "for lr in [0.01, 0.1, 0.3]:\n")
_R_START = _R_HEAD + "    ...\nmses\n"
_R_BODY = "    w, b = fit(sizes, prices, lr, steps)\n    mses.append(np.mean((prices - (w * sizes + b)) ** 2))\n"
_R = _R_HEAD + _R_BODY + "mses"


def _r(body: str) -> str:
    return _R_HEAD + body + "mses"


def _empty_list(got, text) -> bool:
    return isinstance(got, list) and len(got) == 0


def _one_value(got, text) -> bool:
    """One MSE instead of three: the append runs after the loop, or `mses` is made again in each round."""
    return isinstance(got, list) and len(got) == 1


def _too_many(got, text) -> bool:
    """More than three values: the append runs in a loop of its own inside the loop over the learning rates."""
    return isinstance(got, list) and len(got) > 3


def _same_three(got, text) -> bool:
    """Three times the same number: every round trained with one fixed learning rate, whichever one it was."""
    try:
        values = [float(v) for v in got]
    except (TypeError, ValueError):
        return False
    return len(values) == 3 and all(math.isclose(v, values[0], rel_tol=1e-9) for v in values)


EXERCISES = {
    "who-sets": Sort(
        name="who-sets",
        groups={
            "parameter": {"en": "Parameter: training learns it", "nl": "Parameter: het trainen leert hem"},
            "hyper": {"en": "Hyperparameter: you choose it", "nl": "Hyperparameter: jij kiest hem"},
        },
        items={
            "weight": {"en": "**The weight for the area** in a line that predicts house prices.",
                       "nl": "**De weight voor de oppervlakte** in een lijn die huizenprijzen voorspelt."},
            "rate": {"en": "**The learning rate** $\\eta$ of gradient descent.",
                     "nl": "**De learning rate** $\\eta$ van gradient descent."},
            "bias": {"en": "**The bias term** $b$ of that line.",
                     "nl": "**De bias term** $b$ van die lijn."},
            "steps": {"en": "**The number of steps** gradient descent takes.",
                      "nl": "**Het aantal stappen** dat gradient descent zet."},
            "degree": {"en": "**The degree** of a polynomial you use as a curve.",
                       "nl": "**De graad** van een polynoom die je als curve gebruikt."},
            "poly-weights": {"en": "**A weight of a polynomial** of degree 3, fitted to temperatures.",
                             "nl": "**Een weight van een polynoom** van graad 3, aangepast aan temperaturen."},
        },
        solution={"weight": "parameter", "rate": "hyper", "bias": "parameter", "steps": "hyper", "degree": "hyper",
                  "poly-weights": "parameter"},
        nudge={
            "weight": {"en": "Do you type it in before training, or does gradient descent change it at every step?",
                       "nl": "Typ je hem in vóór het trainen, of verandert gradient descent hem bij elke stap?"},
            "rate": {"en": "Does gradient descent change it, or does it stay what you set it to?",
                     "nl": "Verandert gradient descent hem, of blijft hij wat jij erin zette?"},
            "bias": {"en": "Where does its value come from at the end of training?",
                     "nl": "Waar komt zijn waarde aan het eind van het trainen vandaan?"},
            "steps": {"en": "Who decides how long the loop runs?",
                      "nl": "Wie bepaalt hoe lang de loop doorgaat?"},
            "degree": {"en": "Does `np.polyfit` work out the degree, or do you pass it in?",
                       "nl": "Rekent `np.polyfit` de graad uit, of geef jij hem mee?"},
            "poly-weights": {"en": "Does `np.polyfit` give it back, or do you pass it in?",
                             "nl": "Geeft `np.polyfit` hem terug, of geef jij hem mee?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the numbers again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de getallen opnieuw uit."},
            "empty": {"en": "Put each number in a group: learned by training, or chosen by you.",
                      "nl": "Zet elk getal in een groep: geleerd door het trainen, of door jou gekozen."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs in the other group.",
                      "nl": "{item} hoort in de andere groep."},
            "correct": {"en": "All {n} are right. Each time the question was the same: does training change it, or do you set it before training starts?",
                        "nl": "Alle {n} kloppen. Elke keer was de vraag dezelfde: verandert het trainen het, of zet jij het vóór het trainen begint?"},
        },
    ),
    "three-rates": CodeBox(
        name="three-rates",
        start=_R_START,
        solution=_R,
        keep=["for lr in [0.01, 0.1, 0.3]:"],
        # The check also runs the box with another number of steps: numbers typed in by hand do not change with it.
        others=[{"steps": Code("50")}],
        first=[_DOTS],
        mistakes=[
            Mistake("no-append", matches=_empty_list,
                    en="Your last line gives an empty list: nothing was added to `mses`. Add the MSE of each run with `mses.append(...)`.",
                    nl="Je laatste regel geeft een lege lijst: er is niets aan `mses` toegevoegd. Voeg de MSE van elke run toe met `mses.append(...)`."),
            Mistake("one-value", matches=_one_value,
                    en="Your last line gives one MSE, and there are three learning rates. Is the line with `mses.append` inside the loop, indented like the line with `fit`, so that it adds to `mses` in every round?",
                    nl="Je laatste regel geeft één MSE, en er zijn drie learning rates. Staat de regel met `mses.append` in de loop, even ver ingesprongen als de regel met `fit`, zodat hij in elke ronde iets aan `mses` toevoegt?"),
            Mistake("too-many", matches=_too_many, detail=lambda got, text: {"n": len(got)},
                    en="Your last line gives {n} values, and there are three learning rates. Do you add one MSE in each round of the loop, not in a loop of its own?",
                    nl="Je laatste regel geeft {n} waarden, en er zijn drie learning rates. Voeg je in elke ronde van de loop één MSE toe, niet in een eigen loop?"),
            Mistake("weights", wrong=_r("    w, b = fit(sizes, prices, lr, steps)\n    mses.append(w)\n"),
                    en="Your last line gives the weight `w` of each run: a parameter that training learned. Add the MSE of each run instead: the mean of the squared residuals.",
                    nl="Je laatste regel geeft de weight `w` van elke run: een parameter die het trainen leerde. Voeg in plaats daarvan de MSE van elke run toe: het gemiddelde van de gekwadrateerde residuen."),
            Mistake("one-rate", wrong=_r("    w, b = fit(sizes, prices, 0.1, steps)\n    mses.append(np.mean((prices - (w * sizes + b)) ** 2))\n"), matches=_same_three,
                    en="Your last line gives the same MSE three times: every run used the same learning rate. Pass `lr`, the learning rate of this round of the loop, to `fit`.",
                    nl="Je laatste regel geeft drie keer dezelfde MSE: elke run gebruikte dezelfde learning rate. Geef `lr`, de learning rate van deze ronde van de loop, mee aan `fit`."),
            Mistake("swapped", error=lambda e: isinstance(e, TypeError) and "float" in str(e) and "integer" in str(e),
                    en="Python stops in `fit`: it gets its values in the order of its `def` line, `(x, y, lr, steps)`. Is a learning rate in the place of `steps`? `range` needs a whole number of steps.",
                    nl="Python stopt in `fit`: die krijgt zijn waarden in de volgorde van zijn `def`-regel, `(x, y, lr, steps)`. Staat er een learning rate op de plek van `steps`? `range` heeft een geheel aantal stappen nodig."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with two lines: train with `fit` for this learning rate, and add the MSE of the trained line to `mses`.",
                      "nl": "Vervang de `...` door twee regels: train met `fit` voor deze learning rate, en voeg de MSE van de getrainde lijn toe aan `mses`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
            "no_value": {"en": "The last line of the box shows nothing. Put `mses` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `mses` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you train with `fit(sizes, prices, lr, steps)` in each round, and add the mean of the squared residuals of that line to `mses`?",
                      "nl": "Je laatste regel geeft `{got}`. Train je in elke ronde met `fit(sizes, prices, lr, steps)`, en voeg je het gemiddelde van de gekwadrateerde residuen van die lijn toe aan `mses`?"},
            "other": {"en": "Right for 200 steps. But with {where}, your last line gives `{got}`, and it should give about `[580, 49.6, 11.0]`. Do you pass `steps` itself to `fit`?",
                      "nl": "Goed voor 200 stappen. Maar met {where} geeft je laatste regel `{got}`, en dat moet ongeveer `[580, 49.6, 11.0]` zijn. Geef je `steps` zelf mee aan `fit`?"},
            "other_error": {"en": "Right for 200 steps. But with {where}, your code stops with an error. Do you pass `steps` itself to `fit`?",
                            "nl": "Goed voor 200 stappen. Maar met {where} stopt je code met een fout. Geef je `steps` zelf mee aan `fit`?"},
            "correct": {"en": "After 200 steps the training MSE is about 125 with η = 0.01, about 10.1 with 0.1 and about 9.9 with 0.3: with the smallest learning rate, training is still far from the lowest point. The check also tried 50 steps.",
                        "nl": "Na 200 stappen is de training-MSE ongeveer 125 met η = 0,01, ongeveer 10,1 met 0,1 en ongeveer 9,9 met 0,3: met de kleinste learning rate is het trainen nog ver van het laagste punt. De check probeerde ook 50 stappen."},
        },
    ),
}

_SOLVED = {"weight": "parameter", "rate": "hyper", "bias": "parameter", "steps": "hyper", "degree": "hyper", "poly-weights": "parameter"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("who-sets", {}, "empty", None),
    ("who-sets", _SOLVED, "correct", None),
    ("who-sets", {"weight": "parameter", "rate": "hyper"}, "wrong", "unfinished"),
    ("who-sets", {**_SOLVED, "rate": "parameter"}, "wrong", "wrong:rate"),
    ("who-sets", {**_SOLVED, "poly-weights": "hyper"}, "wrong", "wrong:poly-weights"),
    ("who-sets", {**_SOLVED, "degree": "parameter"}, "wrong", "wrong:degree"),
    ("three-rates", _R_START, "empty", None),
    ("three-rates", _R, "correct", None),
    ("three-rates", _r("    w, b = fit(sizes, prices, lr, steps)\n    predictions = w * sizes + b\n    mses.append(((prices - predictions) ** 2).mean())\n"), "correct", None),
    ("three-rates", _r("    w, b = fit(x=sizes, y=prices, lr=lr, steps=steps)\n    mses.append(np.mean((w * sizes + b - prices) ** 2))\n"), "correct", None),
    ("three-rates", _r("    w, b = fit(sizes, prices, lr, steps)\n"), "wrong", "no-append"),
    ("three-rates", _r("    w, b = fit(sizes, prices, lr, steps)\n    mses.append(w)\n"), "wrong", "weights"),
    ("three-rates", _r("    w, b = fit(sizes, prices, 0.1, steps)\n    mses.append(np.mean((prices - (w * sizes + b)) ** 2))\n"), "wrong", "one-rate"),
    ("three-rates", _r("    w, b = fit(sizes, prices, 0.01, steps)\n    mses.append(np.mean((prices - (w * sizes + b)) ** 2))\n"), "wrong", "one-rate"),
    ("three-rates", _r("    w, b = fit(sizes, prices, lr, steps)\nmses.append(np.mean((prices - (w * sizes + b)) ** 2))\n"), "wrong", "one-value"),
    ("three-rates", _r("    w, b = fit(sizes, prices, lr, steps)\n    for s in sizes:\n        mses.append(np.mean((prices - (w * sizes + b)) ** 2))\n"), "wrong", "too-many"),
    ("three-rates", _r("    w, b = fit(sizes, prices, steps, lr)\n    mses.append(np.mean((prices - (w * sizes + b)) ** 2))\n"), "error", "swapped"),
    ("three-rates", _r("    w, b = fit(sizes, prices, lr, 200)\n    mses.append(np.mean((prices - (w * sizes + b)) ** 2))\n"), "almost", "other-value"),
]
