# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson stats-bootstrap: bootstrap_means, the means of many resamples of an array, and interval, the
middle part of those means as a confidence interval. And the data: the bill lengths of the 68 Chinstrap penguins of
the Palmer Archipelago, and a model that got 87 of 100 test examples right, with the picture of its bootstrap."""
import math
import re
from functools import cache

import numpy as np

from everything_ai import paths
from everything_ai.check import FunctionBox, Mistake, run_code

DATA = "data/penguins.csv"   # 344 penguins of the Palmer Archipelago: see data/README.md


@cache
def _penguins():
    import pandas as pd
    return pd.read_csv(paths.ROOT / DATA)


def chinstrap_bills() -> np.ndarray:
    """The bill lengths of the Chinstrap penguins, in mm: 68 of them, none missing."""
    d = _penguins()
    return d[d["species"] == "Chinstrap"]["bill_length_mm"].dropna().to_numpy()


def test_results() -> np.ndarray:
    """A model's results on 100 test examples: 1 for right, 0 for wrong; 87 right."""
    return np.array([1] * 87 + [0] * 13)


def boot(x, n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(x, size=len(x)).mean() for _ in range(n_boot)])


def number(x: float, digits: int, lang: str) -> str:
    s = f"{x:.{digits}f}"
    return s.replace(".", ",") if lang == "nl" else s


# --- the picture: the bootstrap of the 100 test results ---------------------------------------------------------------------

def boot_picture(labels: dict) -> str:
    """How often each accuracy came out in 2000 resamples of the 100 test results, with the middle 95% marked."""
    from everything_ai import chart
    means = boot(test_results())
    low, high = np.quantile(means, [0.025, 0.975])
    values, counts = np.unique(np.round(means, 2), return_counts=True)
    top = float(counts.max())
    p = chart.Panel((0.745, 0.985), (0, top * 1.3), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    for v, c in zip(values, counts):
        p.rect(float(v) - 0.0042, 0, float(v) + 0.0042, float(c), "data")
    for v in (low, high):
        p.line([float(v), float(v)], [0, top * 1.12], "ink-3", 1.2, dashed=True)
    p.text(float(low + high) / 2, top * 1.17, labels["band"].format(low=labels["number"](low, 2), high=labels["number"](high, 2)), "ink")
    return chart.figure([p], alt=labels["alt"], width=560, height=300)


# The lesson is done when these are correct. "more-data" is the choice question in the lesson.
REQUIRED = ["more-data", "boot-means", "interval"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


class _Vector(np.ndarray):
    """An array that a check message writes the way the lesson's code does: np.array([...]). The learner's code gets a
    plain copy."""

    def __repr__(self):
        return f"np.array({[int(v) if float(v) == int(v) else float(v) for v in np.asarray(self).tolist()]!r})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _v(values) -> np.ndarray:
    return np.array(values, dtype=float).view(_Vector)


class _Results(np.ndarray):
    """The 100 test results, written in a check message as np.array([1] * 87 + [0] * 13) instead of as 100 numbers."""

    def __repr__(self):
        a = np.asarray(self)
        return f"np.array([1] * {int(a.sum())} + [0] * {int(len(a) - a.sum())})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _plain(x: float, digits: int, lang: str) -> str:
    text = f"{round(float(x), digits) + 0.0:.{digits}f}".rstrip("0").rstrip(".")
    text = "0" if text == "-0" else text
    return text.replace(".", ",") if lang == "nl" else text


def _two(x: float, lang: str) -> str:
    """Two significant digits: 5.04 is 5, 1.083 is 1.1 and 0.0707 is 0.071; random means differ in the third."""
    x = float(x)
    digits = 1 if abs(x) < 1e-12 else max(0, 1 - math.floor(math.log10(abs(x))))
    return _plain(x, digits, lang)


def _many(v):
    """An array of means, or what came back instead, in words: how many there are, their middle and their spread."""
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return repr(v)
    if a.ndim == 0:
        return {"en": f"one number, {_plain(a, 4, 'en')}", "nl": f"één getal, {_plain(a, 4, 'nl')}"}
    if a.ndim != 1:
        return {"en": f"an array of shape {a.shape}", "nl": f"een array met shape {a.shape}"}
    if len(a) == 0 or not np.all(np.isfinite(a)):
        return repr(a.tolist()[:5])
    return {lang: (f"{len(a)} means around {_two(a.mean(), lang)}, with spread {_two(a.std(), lang)}" if lang == "en" else
                   f"{len(a)} gemiddelden rond {_two(a.mean(), lang)}, met spreiding {_two(a.std(), lang)}") for lang in ("en", "nl")}


def _like(got, want) -> bool:
    """Two sets of bootstrap means are alike when they are as many, lie around the same middle and spread as much: random
    numbers, so not the same ones, unless the learner used the same seed and the same calls."""
    try:
        g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    except (TypeError, ValueError):
        return False
    if g.shape != w.shape or g.ndim != 1 or not np.all(np.isfinite(g)):
        return False
    s = w.std()
    if s == 0:
        return g.std() < 1e-12 and abs(g.mean() - w.mean()) < 1e-9
    return abs(g.mean() - w.mean()) < 6 * s / math.sqrt(len(w)) and 0.85 < g.std() / s < 1.15


# --- exercise 2: the bootstrap means ---------------------------------------------------------------------------------------

_B_HEAD = "import numpy as np\n\n\ndef bootstrap_means(x, n_boot, seed):\n"
_B_START = _B_HEAD + "    # the means of n_boot resamples of x, each drawn with replacement, as an array\n    ...\n"


def _b(draw: str = "rng.choice(x, size=len(x))", what: str = "np.mean") -> str:
    return (_B_HEAD + "    rng = np.random.default_rng(seed)\n    means = []\n    for _ in range(n_boot):\n"
            f"        means.append({what}({draw}))\n    return np.array(means)\n")


# --- exercise 3: the interval ----------------------------------------------------------------------------------------------

_I_HEAD = "import numpy as np\n\n\ndef interval(means, level):\n"
_I_START = _I_HEAD + "    # the middle part of the means that holds the share level of them, as two numbers: low and high\n    ...\n"


def _i(cut: str = "[(1 - level) / 2, (1 + level) / 2]") -> str:
    return _I_HEAD + f"    return np.quantile(means, {cut})\n"


def _num(v):
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return repr(v)
    if a.ndim == 0:
        return _plain(a, 4, "en")
    return "[" + ", ".join(_plain(x, 4, "en") for x in a.ravel()) + "]"


def _pair(got, want) -> bool:
    try:
        g = np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return False
    return g.shape == (2,) and np.allclose(g, want, rtol=1e-9, atol=1e-9)


class _Range(np.ndarray):
    """np.arange(start, stop), written that way in a check message instead of as all its numbers."""

    def __repr__(self):
        a = np.asarray(self)
        return f"np.arange({int(a[0])}, {int(a[-1]) + 1})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _r(start: int, stop: int) -> np.ndarray:
    return np.arange(start, stop).view(_Range)


_SAMPLE = _v([2, 4, 6, 8])


def _seeded(text: str):
    """What bootstrap_means gives twice with seed 3 and once with seed 4, for [2, 4, 6, 8]; None when it cannot run."""
    fn = run_code(text).names.get("bootstrap_means")
    try:
        return [np.asarray(fn(np.array([2.0, 4.0, 6.0, 8.0]), 200, s), dtype=float) for s in (3, 3, 4)]
    except Exception:
        return None


def _no_seed(text: str) -> bool:
    r = _seeded(text)
    return r is not None and not np.array_equal(r[0], r[1])


def _fixed_seed(text: str) -> bool:
    r = _seeded(text)
    return r is not None and np.array_equal(r[0], r[1]) and np.array_equal(r[0], r[2])

EXERCISES = {
    "boot-means": FunctionBox(
        name="boot-means",
        function="bootstrap_means",
        start=_B_START,
        solution=_b(),
        visible=[(_SAMPLE, 1000, 0)],
        hidden=[(_v([2, 10]), 2000, 4), (test_results().view(_Results), 2000, 1), (_v(range(1, 11)), 2000, 5), (_v([3, 3, 3, 9]), 1500, 2)],
        show=_many,
        compare=_like,
        first=[_DOTS],
        form=[
            Mistake("no-seed", state="almost", matches=lambda got, text: _no_seed(text),
                    en="Your means spread as a bootstrap should, but called twice with the same seed, your function gives other means: its random number generator is not made with `seed`. Make it with `rng = np.random.default_rng(seed)`, so that the same seed gives the same means.",
                    nl="Je gemiddelden spreiden zoals een bootstrap hoort, maar als je je functie twee keer met dezelfde seed aanroept, geeft hij andere gemiddelden: zijn random number generator is niet met `seed` gemaakt. Maak hem met `rng = np.random.default_rng(seed)`, zodat dezelfde seed dezelfde gemiddelden geeft."),
            Mistake("fixed-seed", state="almost", matches=lambda got, text: _fixed_seed(text),
                    en="Your means spread as a bootstrap should, but another seed gives the very same means: your random number generator is not made with `seed`. Make it with `rng = np.random.default_rng(seed)`.",
                    nl="Je gemiddelden spreiden zoals een bootstrap hoort, maar een andere seed geeft precies dezelfde gemiddelden: je random number generator is niet met `seed` gemaakt. Maak hem met `rng = np.random.default_rng(seed)`."),
        ],
        mistakes=[
            Mistake("no-replacement", wrong=_b("rng.permutation(x)"),
                    en="`{call}` gives {got}, and it should give {want}: every mean is the same, so every resample holds each value exactly once. Draw with replacement, so that a value can come twice or not at all: `rng.choice(x, size=len(x))`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk gemiddelde is hetzelfde, dus elke resample bevat elke waarde precies één keer. Trek met teruglegging, zodat een waarde twee keer of helemaal niet kan komen: `rng.choice(x, size=len(x))`."),
            Mistake("same-resample", wrong=_B_HEAD + "    means = []\n    for _ in range(n_boot):\n        rng = np.random.default_rng(seed)\n        means.append(rng.choice(x, size=len(x)).mean())\n    return np.array(means)\n",
                    en="`{call}` gives {got}, and it should give {want}: every mean is the same, so every resample is the same one. Draw a new resample in each round, from one random number generator made before the loop: a generator made again with the same seed starts over, and draws the same values.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk gemiddelde is hetzelfde, dus elke resample is dezelfde. Trek in elke ronde een nieuwe resample, uit één random number generator die je vóór de loop maakt: een generator die je opnieuw maakt met dezelfde seed begint opnieuw, en trekt dezelfde waarden."),
            Mistake("size", wrong=_b("rng.choice(x, size=n_boot)"),
                    en="`{call}` gives {got}, and it should give {want}: the spread is too small, because each resample holds `n_boot` values. A resample is as large as the data itself: `size=len(x)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de spreiding is te klein, omdat elke resample `n_boot` waarden bevat. Een resample is even groot als de data zelf: `size=len(x)`."),
            Mistake("one-mean", matches=lambda got, text: np.ndim(got) == 0 and not isinstance(got, (str, type(None))),
                    en="`{call}` gives {got}, and it should give {want}: return all the means, one per resample, as an array, not just one.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: geef alle gemiddelden terug, één per resample, als array, niet maar één."),
            Mistake("std", wrong=_b(what="np.std"),
                    en="`{call}` gives {got}, and it should give {want}: you take the standard deviation of each resample. Take its mean: the bootstrap shows how much the mean moves.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je neemt de standaardafwijking van elke resample. Neem het gemiddelde: de bootstrap laat zien hoeveel het gemiddelde beweegt."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: a random number generator with the seed, then `n_boot` times a resample of `x` and its mean.",
                      "nl": "Vervang de `...` door de body van de functie: een random number generator met de seed, dan `n_boot` keer een resample van `x` en het gemiddelde ervan."},
            "arguments": {"en": "The check calls `{call}`, with the data, the number of resamples and a seed, and the `def` line of your function has another number of names between its parentheses. Keep `(x, n_boot, seed)`.",
                          "nl": "De check roept `{call}` aan, met de data, het aantal resamples en een seed, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x, n_boot, seed)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Are there `n_boot` means, each the mean of `len(x)` values drawn from `x` with replacement?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Zijn het `n_boot` gemiddelden, elk het gemiddelde van `len(x)` waarden die met teruglegging uit `x` getrokken zijn?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for every array and every number of resamples?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke array en elk aantal resamples?"},
            "correct": {"en": "Your function is right in all {n} cases: as many means as asked, around the mean of the data, and spread as much as a bootstrap should.",
                        "nl": "Je functie klopt in alle {n} gevallen: zoveel gemiddelden als gevraagd, rond het gemiddelde van de data, en zo gespreid als een bootstrap hoort."},
        },
    ),
    "interval": FunctionBox(
        name="interval",
        function="interval",
        start=_I_START,
        solution=_i(),
        visible=[(_r(1, 101), 0.9)],
        hidden=[(_r(1, 101), 0.95), (_r(0, 1001), 0.99), (_v([5, 1, 4, 2, 3]), 0.5), (_r(1, 101), 0.8)],
        show=_num,
        compare=_pair,
        first=[_DOTS],
        mistakes=[
            Mistake("whole-cut", wrong=_i("[1 - level, level]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you cut `1 - level` off each side, so twice as much as there is to cut, and the middle part holds less than `level`. Cut half of it off each side: `(1 - level) / 2`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je haalt aan elke kant `1 - level` eraf, dus twee keer zoveel als er af moet, en het middelste deel bevat minder dan `level`. Haal aan elke kant de helft eraf: `(1 - level) / 2`."),
            Mistake("level-as-cut", wrong=_i("[level, 1 - level]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the low end is the quantile at `(1 - level) / 2`, close to 0, and the high end at `(1 + level) / 2`, close to 1.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de onderkant is het kwantiel bij `(1 - level) / 2`, dicht bij 0, en de bovenkant het kwantiel bij `(1 + level) / 2`, dicht bij 1."),
            Mistake("swapped-ends", wrong=_i("[(1 + level) / 2, (1 - level) / 2]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the right two ends, but high first. Give the low end first, then the high end.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de goede twee kanten, maar de hoge eerst. Geef eerst de onderkant, dan de bovenkant."),
            Mistake("one-end", matches=lambda got, text: np.ndim(got) == 0 and not isinstance(got, (str, type(None))),
                    en="`{call}` gives `{got}`, and it should give `{want}`: an interval has two ends. Give both, low and high: `np.quantile` takes a list of two fractions.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: een interval heeft twee kanten. Geef ze allebei, laag en hoog: `np.quantile` neemt een lijst van twee fracties."),
            Mistake("two-spreads", wrong=_I_HEAD + "    return np.array([means.mean() - 2 * means.std(), means.mean() + 2 * means.std()])\n",
                    en="`{call}` gives `{got}`, and it should give `{want}`: two standard deviations on each side of the mean give about the 95% interval only when the means form a hill like the one in the picture, and they leave out `level`. The bootstrap reads the interval off the means themselves, with `np.quantile`, whatever their shape and whatever the level.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: twee standaardafwijkingen aan elke kant van het gemiddelde geven alleen ongeveer het 95%-interval als de gemiddelden een heuvel vormen zoals in de figuur, en ze laten `level` weg. De bootstrap leest het interval af van de gemiddelden zelf, met `np.quantile`, welke vorm ze ook hebben en welk niveau je ook vraagt."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: the two quantiles of `means` that leave the middle share `level` between them.",
                      "nl": "Vervang de `...` door de body van de functie: de twee kwantielen van `means` die het middelste deel `level` tussen zich laten."},
            "arguments": {"en": "The check calls `{call}`, with the means and the level, and the `def` line of your function has another number of names between its parentheses. Keep `(means, level)`.",
                          "nl": "De check roept `{call}` aan, met de gemiddelden en het niveau, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(means, level)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Which share is left over outside the interval, and how much of it goes to each side?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Welk deel blijft er buiten het interval over, en hoeveel daarvan gaat naar elke kant?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every level?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elk niveau?"},
            "correct": {"en": "Your function is right in all {n} cases: the middle share `level` of the means, cut off evenly at both sides.",
                        "nl": "Je functie klopt in alle {n} gevallen: het middelste deel `level` van de gemiddelden, aan beide kanten even ver afgesneden."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("boot-means", _B_START, "empty", None),
    ("boot-means", _b(), "correct", None),
    ("boot-means", _B_HEAD + "    rng = np.random.default_rng(seed)\n    return rng.choice(x, size=(n_boot, len(x))).mean(axis=1)\n", "correct", None),
    ("boot-means", _B_HEAD + "    rng = np.random.default_rng(seed)\n    means = []\n    for _ in range(n_boot):\n        i = rng.integers(0, len(x), size=len(x))\n        means.append(x[i].mean())\n    return means\n", "correct", None),
    ("boot-means", _b("rng.permutation(x)"), "wrong", "no-replacement"),
    ("boot-means", _B_HEAD + "    rng = np.random.default_rng(seed)\n    means = []\n    for _ in range(n_boot):\n        means.append(rng.choice(x, size=len(x), replace=False).mean())\n    return means\n", "wrong", "no-replacement"),
    ("boot-means", _b("rng.choice(x, size=n_boot)"), "wrong", "size"),
    ("boot-means", _B_HEAD + "    rng = np.random.default_rng(seed)\n    return rng.choice(x, size=len(x)).mean()\n", "wrong", "one-mean"),
    ("boot-means", _b(what="np.std"), "wrong", "std"),
    ("boot-means", _B_HEAD + "    means = []\n    for _ in range(n_boot):\n        rng = np.random.default_rng(seed)\n        means.append(rng.choice(x, size=len(x)).mean())\n    return np.array(means)\n", "wrong", "same-resample"),
    ("boot-means", _B_HEAD + "    rng = np.random.default_rng(seed)\n    r = rng.choice(x, size=len(x))\n    return np.array([r.mean()] * n_boot)\n", "wrong", "same-resample"),
    ("boot-means", _B_HEAD + "    rng = np.random.default_rng()\n    return rng.choice(x, size=(n_boot, len(x))).mean(axis=1)\n", "almost", "no-seed"),
    ("boot-means", _B_HEAD + "    return np.random.choice(x, size=(n_boot, len(x))).mean(axis=1)\n", "almost", "no-seed"),
    ("boot-means", _B_HEAD + "    rng = np.random.default_rng(0)\n    return rng.choice(x, size=(n_boot, len(x))).mean(axis=1)\n", "almost", "fixed-seed"),
    ("interval", _I_START, "empty", None),
    ("interval", _i(), "correct", None),
    ("interval", _I_HEAD + "    cut = (1 - level) / 2\n    low = np.quantile(means, cut)\n    high = np.quantile(means, 1 - cut)\n    return low, high\n", "correct", None),
    ("interval", _i("[1 - level, level]"), "wrong", "whole-cut"),
    ("interval", _i("[level, 1 - level]"), "wrong", "level-as-cut"),
    ("interval", _I_HEAD + "    return np.quantile(means, level)\n", "wrong", "one-end"),
    ("interval", _I_HEAD + "    return np.array([means.mean() - 2 * means.std(), means.mean() + 2 * means.std()])\n", "wrong", "two-spreads"),
    ("interval", _i("[0.05, 0.95]"), "almost", None),
    ("interval", _i("[(1 + level) / 2, (1 - level) / 2]"), "wrong", "swapped-ends"),
    ("boot-means", _b("rng.choice(x, size=len(x) + 1)"), "almost", None),
]
