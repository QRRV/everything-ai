# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson prob-expectation: the expected value of a distribution, as a function, and the variance of
a bet with NumPy. And the picture: two bets with the same expected value and a different spread."""
import re

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake

# Two bets of the picture: what you win, and how likely. Both give 5 on average.
SAFE = {4: 0.5, 6: 0.5}
WILD = {0: 0.5, 10: 0.5}


def picture(labels: dict[str, str]) -> str:
    """A bar per outcome of each bet, as high as its probability, and a dashed line at the expected value."""
    from everything_ai import chart
    panels = []
    for dist, title in ((SAFE, labels["safe"]), (WILD, labels["wild"])):
        p = chart.Panel((-1, 11), (0, 1), title=title, xlabel=labels["x"], ylabel=labels["y"], xticks=[0, 2, 4, 6, 8, 10])
        for value, prob in dist.items():
            p.rect(value - 0.4, 0, value + 0.4, prob, "data")
        mean = sum(v * q for v, q in dist.items())
        p.line([mean, mean], [0, 1], "ink-3", 1.4, dashed=True)
        panels.append(p)
    return chart.figure(panels, alt=labels["alt"], width=330, height=250)


# The lesson is done when these are correct. "game" is the choice question in the lesson.
REQUIRED = ["game", "expected", "variance"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_E_START = "def expected(dist):\n    ...\n"
_E = "def expected(dist):\n    total = 0\n    for value in dist:\n        total = total + value * dist[value]\n    return total"


def _e(new: str) -> str:
    return "def expected(dist):\n" + new


_V_HEAD = "import numpy as np\n\nvalues = np.array([0, 5, 20])\nprobs = np.array([0.5, 0.3, 0.2])\nmean = (values * probs).sum()\n"
_V_START = _V_HEAD + "variance = ...\nvariance\n"
_V = _V_HEAD + "variance = ((values - mean) ** 2 * probs).sum()\nvariance"


def _v(new: str) -> str:
    return _V.replace("variance = ((values - mean) ** 2 * probs).sum()", new)


EXERCISES = {
    "expected": FunctionBox(
        name="expected",
        function="expected",
        start=_E_START,
        solution=_E,
        visible=[({1: 0.5, 2: 0.25, 4: 0.25},)],
        hidden=[({0: 0.9, 10: 0.1},), ({1: 1 / 6, 2: 1 / 6, 3: 1 / 6, 4: 1 / 6, 5: 1 / 6, 6: 1 / 6},), ({5: 1.0},), ({-2: 0.5, 2: 0.5},),
                ({0: 0.7, 1: 0.2, 3: 0.1},)],
        first=[_DOTS],
        mistakes=[
            Mistake("plain-mean", wrong=_e("    total = 0\n    for value in dist:\n        total = total + value\n    return total / len(dist)"),
                    en="`{call}` gives `{got}`: the plain mean of the values, as if each were equally likely. Weight each value by its probability: `value * dist[value]`.",
                    nl="`{call}` geeft `{got}`: het gewone gemiddelde van de waarden, alsof elke waarde even waarschijnlijk is. Weeg elke waarde met zijn kans: `value * dist[value]`."),
            Mistake("probs-only", wrong=_e("    total = 0\n    for value in dist:\n        total = total + dist[value]\n    return total"),
                    en="`{call}` gives `{got}`: the probabilities added up, and they always make 1. Multiply each probability by its value first.",
                    nl="`{call}` geeft `{got}`: de kansen opgeteld, en die zijn samen altijd 1. Vermenigvuldig eerst elke kans met zijn waarde."),
            Mistake("values-only", wrong=_e("    total = 0\n    for value in dist:\n        total = total + value\n    return total"),
                    en="`{call}` gives `{got}`: the values added up, without their probabilities. Multiply each value by its probability before you add.",
                    nl="`{call}` geeft `{got}`: de waarden opgeteld, zonder hun kansen. Vermenigvuldig elke waarde met zijn kans voordat je optelt."),
            Mistake("return-in-loop", wrong=_e("    total = 0\n    for value in dist:\n        total = total + value * dist[value]\n        return total"),
                    en="`{call}` gives `{got}`: only the first value times its probability. Is your `return` inside the loop? Move it to the left, under `for`, so that the loop goes through every value first.",
                    nl="`{call}` geeft `{got}`: alleen de eerste waarde keer zijn kans. Staat je `return` in de loop? Zet hem naar links, onder `for`, zodat de loop eerst elke waarde langsgaat."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the expected value: each value times its probability, added up, given back with `return`.",
                      "nl": "Vervang de `...` door de verwachtingswaarde: elke waarde keer zijn kans, opgeteld, teruggegeven met `return`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with one dictionary, and the `def` line of your function has a different number of names between its parentheses. Keep `(dist)`.",
                          "nl": "De check roept `{call}` aan, met één dictionary, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(dist)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you multiply each value by its probability, and add all those products up?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vermenigvuldig je elke waarde met zijn kans, en tel je al die producten op?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Do you work it out from `dist` itself, for any values?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken je het uit met `dist` zelf, voor elke waarde?"},
            "correct": {"en": "Your function is right in all {n} cases. 1 × 0.5 + 2 × 0.25 + 4 × 0.25 = 2: the average over many draws, which is not the plain mean of 1, 2 and 4. For a fair die it gives 3.5.",
                        "nl": "Je functie klopt in alle {n} gevallen. 1 × 0,5 + 2 × 0,25 + 4 × 0,25 = 2: het gemiddelde over heel veel trekkingen, en dat is niet het gewone gemiddelde van 1, 2 en 4. Bij een eerlijke dobbelsteen geeft hij 3,5."},
        },
    ),
    "variance": CodeBox(
        name="variance",
        start=_V_START,
        solution=_V,
        keep=["values = np.array([0, 5, 20])", "mean = (values * probs).sum()"],
        # The check also runs the box with other probabilities: a number typed in by hand does not change with them.
        others=[{"probs": Code("np.array([0.6, 0.2, 0.2])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("unweighted", wrong=_v("variance = ((values - mean) ** 2).mean()"),
                    en="Your last line gives `{got}`: the plain mean of the squared distances, as if each value were equally likely. Weight each one by its probability: `* probs`, and then `.sum()`.",
                    nl="Je laatste regel geeft `{got}`: het gewone gemiddelde van de gekwadrateerde afstanden, alsof elke waarde even waarschijnlijk is. Weeg elke afstand met zijn kans: `* probs`, en dan `.sum()`."),
            Mistake("no-square", wrong=_v("variance = ((values - mean) * probs).sum()"),
                    en="Your last line gives about 0: the differences above and below the expected value cancel out. Square each difference first: `** 2`.",
                    nl="Je laatste regel geeft ongeveer 0: de verschillen boven en onder de verwachtingswaarde vallen tegen elkaar weg. Kwadrateer eerst elk verschil: `** 2`."),
            Mistake("std", wrong=_v("variance = ((values - mean) ** 2 * probs).sum() ** 0.5"),
                    en="Your last line gives `{got}`: the square root of the variance, the standard deviation. The task asks for the variance itself: leave out `** 0.5`.",
                    nl="Je laatste regel geeft `{got}`: de wortel uit de variantie, de standaardafwijking. De opdracht vraagt de variantie zelf: laat `** 0.5` weg."),
            Mistake("no-center", wrong=_v("variance = (values ** 2 * probs).sum()"),
                    en="Your last line gives `{got}`: the squares of the values themselves. The variance squares the distance to the expected value: `values - mean` first.",
                    nl="Je laatste regel geeft `{got}`: de kwadraten van de waarden zelf. De variantie kwadrateert de afstand tot de verwachtingswaarde: eerst `values - mean`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the variance: the squared distance of each value to `mean`, times its probability, added up.",
                      "nl": "Vervang de `...` door de variantie: de gekwadrateerde afstand van elke waarde tot `mean`, keer zijn kans, opgeteld."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other probabilities by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere kansen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `variance` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `variance` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you take each value's distance to `mean`, square it, multiply by its probability, and add up?",
                      "nl": "Je laatste regel geeft `{got}`. Neem je van elke waarde de afstand tot `mean`, kwadrateer je die, vermenigvuldig je met zijn kans, en tel je op?"},
            "other": {"en": "Right for these probabilities. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out from `values`, `probs` and `mean` themselves?",
                      "nl": "Goed voor deze kansen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `values`, `probs` en `mean` zelf?"},
            "other_error": {"en": "Right for these probabilities. But with {where}, your code stops with an error. Do you work it out from `values`, `probs` and `mean` themselves?",
                            "nl": "Goed voor deze kansen. Maar met {where} stopt je code met een fout. Reken je het uit met `values`, `probs` en `mean` zelf?"},
            "correct": {"en": "The expected value is 5.5, and the variance 57.25: a standard deviation of about 7.6. The prize swings far around its average. The check also tried other probabilities.",
                        "nl": "De verwachtingswaarde is 5,5, en de variantie 57,25: een standaardafwijking van ongeveer 7,6. De prijs schommelt ver rond zijn gemiddelde. De check probeerde ook andere kansen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("expected", _E_START, "empty", None),
    ("expected", _E, "correct", None),
    ("expected", _e("    return sum([v * p for v, p in dist.items()])"), "correct", None),
    ("expected", _e("    total = 0.0\n    for value in dist:\n        total += dist[value] * value\n    return total"), "correct", None),
    ("expected", _e("    total = 0\n    for value in dist:\n        total = total + value\n    return total / len(dist)"), "wrong", "plain-mean"),
    ("expected", _e("    total = 0\n    for value in dist:\n        total = total + dist[value]\n    return total"), "wrong", "probs-only"),
    ("expected", _e("    total = 0\n    for value in dist:\n        total = total + value\n    return total"), "wrong", "values-only"),
    ("expected", _e("    total = 0\n    for value in dist:\n        total = total + value * dist[value]\n        return total"), "wrong", "return-in-loop"),
    ("expected", _e("    return 2.0"), "almost", None),
    ("variance", _V_START, "empty", None),
    ("variance", _V, "correct", None),
    ("variance", _v("variance = (probs * (values - mean) ** 2).sum()"), "correct", None),
    ("variance", _v("variance = np.dot(probs, (values - mean) ** 2)"), "correct", None),
    ("variance", _v("variance = (values ** 2 * probs).sum() - mean ** 2"), "correct", None),
    ("variance", _v("variance = ((values - mean) ** 2).mean()"), "wrong", "unweighted"),
    ("variance", _v("variance = ((values - mean) * probs).sum()"), "wrong", "no-square"),
    ("variance", _v("variance = ((values - mean) ** 2 * probs).sum() ** 0.5"), "wrong", "std"),
    ("variance", _v("variance = (values ** 2 * probs).sum()"), "wrong", "no-center"),
    ("variance", _v("variance = 57.25"), "almost", "other-value"),
]
