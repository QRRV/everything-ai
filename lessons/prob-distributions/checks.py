# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson prob-distributions: normal_density, the bell curve as a formula, and share_within, the share
of values within k standard deviations of the mean. And the penguins of the Palmer Archipelago: their sex as a
Bernoulli distribution, their species as a categorical one, and the bill length of the Adélie penguins as a normal
one, with the picture of its histogram and bell curve."""
import math
import re
from functools import cache

import numpy as np

from everything_ai import paths
from everything_ai.check import FunctionBox, Mistake

DATA = "data/penguins.csv"   # 344 penguins of the Palmer Archipelago: see data/README.md


@cache
def penguins():
    import pandas as pd
    return pd.read_csv(paths.ROOT / DATA)


def bills() -> np.ndarray:
    """The bill lengths of the Adélie penguins, in mm, without the missing ones."""
    d = penguins()
    return d[d["species"] == "Adelie"]["bill_length_mm"].dropna().to_numpy()


def within(k: float) -> float:
    x = bills()
    return float(np.mean(np.abs(x - x.mean()) < k * x.std()))


def number(x: float, digits: int, lang: str) -> str:
    s = f"{x:.{digits}f}"
    return s.replace(".", ",") if lang == "nl" else s


# --- the picture: the histogram of the bill lengths, with the bell curve of their mean and standard deviation ------------

def picture(labels: dict) -> str:
    from everything_ai import chart
    x = bills()
    m, s = float(x.mean()), float(x.std())
    edges = np.arange(32, 47, 1.0)
    counts, _ = np.histogram(x, bins=edges)
    p = chart.Panel((31.5, 46.5), (0, 27), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    for a, c in zip(edges[:-1], counts):
        if c:
            p.rect(a + 0.05, 0, a + 0.95, float(c), "data")
    xs = [31.5 + 15 * i / 200 for i in range(201)]
    p.line(xs, [len(x) * 1.0 / (s * math.sqrt(2 * math.pi)) * math.exp(-(v - m) ** 2 / (2 * s * s)) for v in xs], "pred", 2.6)
    for v in (m - s, m + s):
        p.line([v, v], [0, 25], "ink-3", 1.2, dashed=True)
    p.text(m, 25.6, labels["band"], "ink")
    return chart.figure([p], alt=labels["alt"], width=560, height=300)


# The lesson is done when these are correct. "which" is the choice question in the lesson.
REQUIRED = ["which", "density", "share-within"]

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
        return math.isclose(float(got), float(want), rel_tol=1e-6, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


class _Vector(np.ndarray):
    """Numbers that a check message writes the way the lesson's code does: np.array([...]). The learner's code gets
    a plain copy."""

    def __repr__(self):
        return f"np.array({[int(v) if float(v) == int(v) else float(v) for v in np.asarray(self).tolist()]!r})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _v(values) -> np.ndarray:
    return np.array(values, dtype=float).view(_Vector)


# --- exercise 2: the density of the normal distribution ------------------------------------------------------------------

_D_HEAD = "import math\n\n\ndef normal_density(x, mu, sigma):\n"
_D_START = _D_HEAD + "    return ...\n"


def _d(body: str = "1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-(x - mu) ** 2 / (2 * sigma ** 2))") -> str:
    return _D_HEAD + f"    return {body}\n"


# --- exercise 3: the share within k standard deviations ----------------------------------------------------------------

_S_HEAD = ("import numpy as np\n\n\ndef share_within(x, k):\n    m = np.mean(x)\n    s = np.std(x)\n"
           "    low = m - k * s\n    high = m + k * s\n")
_S_END = "    return len(inside) / len(x)\n"
_S_START = ("import numpy as np\n\n\ndef share_within(x, k):\n"
            "    # the share of the values of x within k standard deviations of their mean\n    ...\n")


def _s(first: str = "x[x > low]", second: str = "inside[inside < high]") -> str:
    return _S_HEAD + f"    inside = {first}\n    inside = {second}\n" + _S_END


EXERCISES = {
    "density": FunctionBox(
        name="density",
        function="normal_density",
        start=_D_START,
        solution=_d(),
        visible=[(4, 1, 3)],
        hidden=[(0, 0, 1), (38.8, 38.8, 2.7), (5, 3, 0.5), (-1, 2, 3), (0, 0, 0.2)],
        show=_num,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("variance-front", wrong=_d("1 / (sigma ** 2 * math.sqrt(2 * math.pi)) * math.exp(-(x - mu) ** 2 / (2 * sigma ** 2))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the number in front has σ, the standard deviation, not σ²; the square belongs only in the exponent.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: voor de e-macht staat σ, de standaardafwijking, niet σ². Het kwadraat hoort alleen in de exponent."),
            Mistake("sigma-not-squared", wrong=_d("1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-(x - mu) ** 2 / (2 * sigma))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: in the exponent, divide by 2σ², with σ squared.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: in de exponent deel je door 2σ², met σ in het kwadraat."),
            Mistake("no-two", wrong=_d("1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-(x - mu) ** 2 / sigma ** 2)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: in the exponent, divide by 2σ², not by σ²: the 2 is missing.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: in de exponent deel je door 2σ², niet door σ²: de 2 ontbreekt."),
            Mistake("plus", wrong=_d("1 / (sigma * math.sqrt(2 * math.pi)) * math.exp((x - mu) ** 2 / (2 * sigma ** 2))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the exponent needs a minus sign. Without it, the curve rises away from the mean instead of falling.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de exponent heeft een minteken nodig. Zonder minteken stijgt de kromme naarmate je verder van het gemiddelde komt, in plaats van te dalen."),
            Mistake("no-square", wrong=_d("1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-(x - mu) / (2 * sigma ** 2))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: in the exponent, `x - mu` must be squared. Without the square, values below the mean get a density that is too high, and the curve is not symmetric.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: in de exponent moet `x - mu` in het kwadraat. Zonder kwadraat krijgen waarden onder het gemiddelde een te hoge dichtheid, en is de kromme niet symmetrisch."),
            Mistake("no-front", wrong=_d("math.exp(-(x - mu) ** 2 / (2 * sigma ** 2))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the number in front is missing, 1 divided by the product of σ and the square root of 2π. It makes the area under the curve 1.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: het getal ervoor ontbreekt, 1 gedeeld door het product van σ en de wortel uit 2π. Dat maakt de oppervlakte onder de kromme 1."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the formula of the normal density: the number in front, times e to the power of the exponent.",
                      "nl": "Vervang de `...` door de formule van de normale dichtheid: het getal ervoor, keer e tot de macht de exponent."},
            "arguments": {"en": "The check calls `{call}`, with `x`, `mu` and `sigma`, and the `def` line of your function has another number of names between its parentheses. Keep `(x, mu, sigma)`.",
                          "nl": "De check roept `{call}` aan, met `x`, `mu` en `sigma`, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x, mu, sigma)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your line piece by piece with the formula: the number in front, and the exponent.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je regel stukje voor stukje met de formule: het getal ervoor, en de exponent."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every mean and every standard deviation?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elk gemiddelde en elke standaardafwijking?"},
            "correct": {"en": "Your density is right in all {n} cases: highest at the mean, and lower the further away, the faster the smaller σ is.",
                        "nl": "Je dichtheid klopt in alle {n} gevallen: het hoogst bij het gemiddelde, en lager hoe verder weg, sneller naarmate σ kleiner is."},
        },
    ),
    "share-within": FunctionBox(
        name="share-within",
        function="share_within",
        start=_S_START,
        solution=_s(),
        visible=[(_v([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 1)],
        hidden=[(_v([2, 4, 4, 4, 5, 5, 7, 9]), 1), (_v([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 2), (_v([10, 20, 30, 40]), 0.5),
                (_v([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 1.5)],
        show=_num,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("x-again", wrong=_s(second="x[x < high]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the share of all values below the upper edge, also the ones below the lower edge. Keep only the values between the two edges: filter what is left after the first mask.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is het deel van alle waarden onder de bovenrand, ook die onder de onderrand. Houd alleen de waarden tussen de twee randen: filter wat er na de eerste mask over is."),
            Mistake("wrong-way", wrong=_s(second="inside[inside > high]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the share of values above the upper edge, outside the band. Keep the ones below it.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is het deel van de waarden boven de bovenrand, buiten de band. Houd de waarden eronder."),
            Mistake("edge-in", wrong=_s(first="x[x >= low]", second="inside[inside <= high]"), state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: the mean is 5 and the standard deviation 2, so the band runs from 3 to 7, and the 7 lies exactly on its upper edge. Count only the values strictly between the edges, with `>` and `<`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: het gemiddelde is 5 en de standaardafwijking 2, dus de band loopt van 3 tot 7, en de 7 ligt precies op de bovenrand. Tel alleen de waarden echt tussen de randen, met `>` en `<`."),
            Mistake("count", wrong=_S_HEAD + "    inside = x[x > low]\n    inside = inside[inside < high]\n    return len(inside)\n",
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is how many values lie in the band. The share is that number divided by the number of values, `len(x)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is hoeveel waarden er in de band liggen. Het deel is dat aantal gedeeld door het aantal waarden, `len(x)`."),
            Mistake("variance", wrong=_S_HEAD.replace("np.std(x)", "np.var(x)") + "    inside = x[x > low]\n    inside = inside[inside < high]\n" + _S_END,
                    en="`{call}` gives `{got}`, and it should give `{want}`: `np.var` gives the variance, σ², so your band is far too wide. The band is k standard deviations wide on each side: use `np.std`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `np.var` geeft de variantie, σ², dus je band is veel te breed. De band is aan elke kant k standaardafwijkingen breed: gebruik `np.std`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: the mean, the standard deviation, the two edges of the band, and the share of the values between them.",
                      "nl": "Vervang de `...` door de body van de functie: het gemiddelde, de standaardafwijking, de twee randen van de band, en het deel van de waarden daartussen."},
            "arguments": {"en": "The check calls `{call}`, with an array and a number `k`, and the `def` line of your function has another number of names between its parentheses. Keep `(x, k)`.",
                          "nl": "De check roept `{call}` aan, met een array en een getal `k`, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x, k)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Work out the two edges of the band by hand, and count which values lie between them.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken de twee randen van de band met de hand uit, en tel welke waarden ertussen liggen."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every array and every `k`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke array en elke `k`?"},
            "correct": {"en": "Your function is right in all {n} cases: the share of values within k standard deviations of the mean.",
                        "nl": "Je functie klopt in alle {n} gevallen: het deel van de waarden binnen k standaardafwijkingen van het gemiddelde."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("density", _D_START, "empty", None),
    ("density", _d(), "correct", None),
    ("density", _d("math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))"), "correct", None),
    ("density", _d("1 / (sigma ** 2 * math.sqrt(2 * math.pi)) * math.exp(-(x - mu) ** 2 / (2 * sigma ** 2))"), "wrong", "variance-front"),
    ("density", _d("1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-(x - mu) ** 2 / (2 * sigma))"), "wrong", "sigma-not-squared"),
    ("density", _d("1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-(x - mu) ** 2 / sigma ** 2)"), "wrong", "no-two"),
    ("density", _d("1 / (sigma * math.sqrt(2 * math.pi)) * math.exp((x - mu) ** 2 / (2 * sigma ** 2))"), "wrong", "plus"),
    ("density", _d("math.exp(-(x - mu) ** 2 / (2 * sigma ** 2))"), "wrong", "no-front"),
    ("density", _d("1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-(x - mu) / (2 * sigma ** 2))"), "wrong", "no-square"),
    ("share-within", _S_START, "empty", None),
    ("share-within", _s(), "correct", None),
    ("share-within", _s(first="x[x < high]", second="inside[inside > low]"), "correct", None),
    ("share-within", _s(second="x[x < high]"), "wrong", "x-again"),
    ("share-within", _s(second="inside[inside > high]"), "wrong", "wrong-way"),
    ("share-within", _s(first="x[x >= low]", second="inside[inside <= high]"), "almost", "edge-in"),
    ("share-within", _S_HEAD + "    inside = x[x > low]\n    inside = inside[inside < high]\n    return len(inside)\n", "wrong", "count"),
    ("share-within", _S_HEAD.replace("np.std(x)", "np.var(x)") + "    inside = x[x > low]\n    inside = inside[inside < high]\n" + _S_END, "wrong", "variance"),
    ("share-within", _S_HEAD.replace("np.std(x)", "np.std(x, ddof=1)") + "    inside = x[x > low]\n    inside = inside[inside < high]\n" + _S_END, "almost", None),
    ("share-within", "import numpy as np\n\n\ndef share_within(x, k):\n    return np.mean(np.abs(x - np.mean(x)) < k * np.std(x))\n", "correct", None),
]
