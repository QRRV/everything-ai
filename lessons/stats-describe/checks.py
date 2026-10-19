# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson stats-describe: the 0.75 quantile of the travel times of 15 students, and a function that
works out the standard deviation step by step. And the histogram in the picture."""
import re

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake

# How many minutes 15 students took to get to school one morning; one of them missed the bus.
TIMES = [25, 18, 32, 20, 95, 27, 12, 40, 22, 25, 35, 15, 30, 20, 45]


def picture(labels: dict[str, str]) -> str:
    """The histogram of the travel times, with the median and the mean as two lines."""
    import matplotlib.pyplot as plt
    import numpy as np
    from everything_ai import plots
    times = np.array(TIMES)
    fig = plt.figure(figsize=(5.6, 3.0))
    plt.hist(times, bins=range(0, 101, 10), edgecolor=plots.PAGE)
    median, mean = np.median(times), times.mean()
    plt.axvline(median, color=plots.INK, linestyle="-")
    plt.axvline(mean, color=plots.INK, linestyle="--")
    # Each line has its name next to it, above the bars: the median to the left of its line, the mean to the right.
    plt.text(median - 1.5, 6.6, labels["median"], ha="right", va="bottom", color=plots.INK)
    plt.text(mean + 1.5, 6.6, labels["mean"], ha="left", va="bottom", color=plots.INK)
    plt.ylim(0, 7.6)
    plt.xlabel(labels["x"])
    plt.ylabel(labels["y"])
    plt.title(labels["title"])
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


# The lesson is done when these are correct. "typical-house" is the choice question in the lesson.
REQUIRED = ["typical-house", "slow-trips", "spread"]

_TIMES_LINE = f"times = np.array({TIMES})"
_SLOW_START = "import numpy as np\n\n" + _TIMES_LINE + "\nslow = ...\nslow\n"
_SLOW = "import numpy as np\n\n" + _TIMES_LINE + "\nslow = np.quantile(times, 0.75)\nslow"

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _slow(new: str) -> str:
    return _SLOW.replace("slow = np.quantile(times, 0.75)", new)


_SPREAD_START = "import numpy as np\n\n\ndef spread(values):\n    x = np.array(values)\n    ...\n"
_SPREAD = ("import numpy as np\n\n\ndef spread(values):\n    x = np.array(values)\n"
           "    differences = x - x.mean()\n    return (differences ** 2).mean() ** 0.5")


def _spread(new: str) -> str:
    return _SPREAD.replace("    differences = x - x.mean()\n    return (differences ** 2).mean() ** 0.5", new)


EXERCISES = {
    "slow-trips": CodeBox(
        name="slow-trips",
        start=_SLOW_START,
        solution=_SLOW,
        keep=["import numpy as np", _TIMES_LINE],
        # The check also runs the box with another group of students: a number typed in by hand does not change with it.
        others=[{"times": Code("np.array([8, 10, 11, 14, 15, 15, 16, 18, 22, 26, 60])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("decimal-comma", syntax=r"\b0,\d",
                    en="Python writes a decimal number with a point: `0.75`. With a comma, `0,75`, it reads two numbers, 0 and 75.",
                    nl="Python schrijft een kommagetal met een punt: `0.75`. Met een komma, `0,75`, leest hij twee getallen, 0 en 75."),
            Mistake("percent", error=lambda e: isinstance(e, ValueError) and "must be in the range [0, 1]" in str(e),
                    en="`np.quantile` takes the array first, and then a fraction from 0 to 1, not a percentage: 3 of every 4 is `0.75`.",
                    nl="`np.quantile` krijgt eerst de array, en dan een fractie van 0 tot 1, geen percentage: 3 van elke 4 is `0.75`."),
            Mistake("array-method", error=lambda e: isinstance(e, AttributeError) and re.search(r"has no attribute '(quantile|median|percentile)'", str(e)) is not None,
                    en="An array has `.mean()`, `.std()`, `.min()` and `.max()`, but no method for the median or a quantile: those are NumPy functions that get the array, such as `np.median(times)`.",
                    nl="Een array heeft `.mean()`, `.std()`, `.min()` en `.max()`, maar geen methode voor de mediaan of een kwantiel: dat zijn functies van NumPy die de array krijgen, zoals `np.median(times)`."),
            Mistake("percentile", wrong=_slow("slow = np.percentile(times, 0.75)"),
                    en="Your last line gives `{got}`. `np.percentile` counts in hundredths, so `0.75` there is 0.75%, almost the fastest trip. Use `np.quantile` with a fraction, or `np.percentile` with a percentage.",
                    nl="Je laatste regel geeft `{got}`. `np.percentile` telt in honderdsten, dus `0.75` is daar 0,75%, bijna de snelste rit. Gebruik `np.quantile` met een fractie, of `np.percentile` met een percentage."),
            Mistake("example-fraction", wrong=_slow("slow = np.quantile(times, 0.9)"),
                    en="Your last line gives `{got}`: the 0.9 quantile from the worked example, within which 9 of every 10 trips are done. The task asks for 3 of every 4.",
                    nl="Je laatste regel geeft `{got}`: het 0,9-kwantiel uit het uitgewerkte voorbeeld, waarbinnen 9 van elke 10 ritten klaar zijn. De opdracht vraagt om 3 van elke 4."),
            Mistake("fast-side", wrong=_slow("slow = np.quantile(times, 0.25)"),
                    en="Your last line gives `{got}`: the 0.25 quantile, below which only the fastest quarter of the trips lies. The task asks for the time within which 3 of every 4 are done.",
                    nl="Je laatste regel geeft `{got}`: het 0,25-kwantiel, waaronder alleen het snelste kwart van de ritten ligt. De opdracht vraagt om de tijd waarbinnen 3 van elke 4 klaar zijn."),
            Mistake("median", wrong=_slow("slow = np.median(times)"),
                    en="Your last line gives `{got}`: the median, within which half of the trips are done. The task asks for 3 of every 4.",
                    nl="Je laatste regel geeft `{got}`: de mediaan, waarbinnen de helft van de ritten klaar is. De opdracht vraagt om 3 van elke 4."),
            Mistake("slowest", wrong=_slow("slow = times.max()"),
                    en="Your last line gives `{got}`: the slowest trip of all, the one of the student who missed the bus. The task asks for the time within which 3 of every 4 trips are done.",
                    nl="Je laatste regel geeft `{got}`: de langzaamste rit van allemaal, die van de leerling die de bus miste. De opdracht vraagt om de tijd waarbinnen 3 van elke 4 ritten klaar zijn."),
            Mistake("mean", wrong=_slow("slow = times.mean()"),
                    en="Your last line gives `{got}`: the mean. The task asks for the time within which 3 of every 4 trips are done: a quantile.",
                    nl="Je laatste regel geeft `{got}`: het gemiddelde. De opdracht vraagt om de tijd waarbinnen 3 van elke 4 ritten klaar zijn: een kwantiel."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the time within which 3 of every 4 trips are done: a quantile of `times`.",
                      "nl": "Vervang de `...` door de tijd waarbinnen 3 van elke 4 ritten klaar zijn: een kwantiel van `times`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other times by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere tijden."},
            "no_value": {"en": "The last line of the box shows nothing. Put `slow` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `slow` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be the time within which 3 of every 4 trips are done. Which fraction is 3 of every 4, and which function takes it?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet de tijd zijn waarbinnen 3 van elke 4 ritten klaar zijn. Welke fractie is 3 van elke 4, en welke functie krijgt die?"},
            "other": {"en": "Right for these students. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out from `times` itself?",
                      "nl": "Goed voor deze leerlingen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `times` zelf?"},
            "other_error": {"en": "Right for these students. But with {where}, your code stops with an error. Do you work it out from `times` itself?",
                            "nl": "Goed voor deze leerlingen. Maar met {where} stopt je code met een fout. Reken je het uit met `times` zelf?"},
            "correct": {"en": "3 of every 4 trips are done within 33.5 minutes. The check also tried another group of students, and your code gave their 0.75 quantile too.",
                        "nl": "3 van elke 4 ritten zijn binnen 33,5 minuut klaar. De check probeerde ook een andere groep leerlingen, en je code gaf ook daar het 0,75-kwantiel."},
        },
    ),
    "spread": FunctionBox(
        name="spread",
        function="spread",
        start=_SPREAD_START,
        solution=_SPREAD,
        visible=[([2, 4, 4, 4, 5, 5, 7, 9],)],
        hidden=[([5, 5, 5],), ([1, 3],), ([10, 20, 30, 40],), ([-3, 3],)],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("half-power", wrong=_spread("    differences = x - x.mean()\n    return (differences ** 2).mean() ** 1 / 2"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. Python works out `** 1` first, and then divides by 2, so `** 1 / 2` is not a square root. Write `** 0.5`, or `** (1 / 2)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Python rekent eerst `** 1` uit, en deelt dan door 2, dus `** 1 / 2` is geen wortel. Schrijf `** 0.5`, of `** (1 / 2)`."),
            Mistake("no-root", wrong=_spread("    differences = x - x.mean()\n    return (differences ** 2).mean()"),
                    en="`{call}` gives `{got}`: the mean of the squared differences, in squared units. Take its square root, `** 0.5`, to get back to the unit of the values.",
                    nl="`{call}` geeft `{got}`: het gemiddelde van de gekwadrateerde verschillen, in gekwadrateerde eenheden. Neem de wortel, `** 0.5`, om terug te gaan naar de eenheid van de waarden."),
            Mistake("no-square", wrong=_spread("    differences = x - x.mean()\n    return differences.mean()"),
                    en="`{call}` gives `{got}`: the differences above the mean and below it cancel each other out, so their mean is always 0. Square each difference first: `** 2`.",
                    nl="`{call}` geeft `{got}`: de verschillen boven en onder het gemiddelde heffen elkaar op, dus hun gemiddelde is altijd 0. Kwadrateer eerst elk verschil: `** 2`."),
            Mistake("absolute", wrong=_spread("    differences = x - x.mean()\n    return abs(differences).mean()"),
                    en="`{call}` gives `{got}`: the mean of the differences without their sign. That is a measure of spread too, but not the standard deviation, which is `{want}` here: square each difference, take the mean, and then the square root.",
                    nl="`{call}` geeft `{got}`: het gemiddelde van de verschillen zonder hun teken. Dat is ook een maat voor de spreiding, maar niet de standaardafwijking, die hier `{want}` is: kwadrateer elk verschil, neem het gemiddelde, en dan de wortel."),
            Mistake("no-mean-first", wrong=_spread("    return (x ** 2).mean() ** 0.5"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you square the values themselves, not their differences from the mean. First work out `x - x.mean()`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je kwadrateert de waarden zelf, niet hun verschillen met het gemiddelde. Reken eerst `x - x.mean()` uit."),
            Mistake("one-less", wrong=_spread("    differences = x - x.mean()\n    return ((differences ** 2).sum() / (len(x) - 1)) ** 0.5"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you divide by one less than the number of values, as some other tools do. Here, take the plain mean of the squared differences, as `np.std` does.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je deelt door één minder dan het aantal waarden, zoals sommige andere tools doen. Neem hier het gewone gemiddelde van de gekwadrateerde verschillen, zoals `np.std` doet."),
        ],
        # The exercise is about the steps: np.std gives the right number without them.
        form=[
            Mistake("ready-made", state="almost", matches=lambda got, text: re.search(r"^[^#\n]*\bstd\(", text, re.MULTILINE) is not None,
                    en="Right: that is the standard deviation. But the exercise asks you to work it out yourself, step by step, to see what it is made of: the differences from the mean, squared, their mean, and the square root.",
                    nl="Goed: dat is de standaardafwijking. Maar de oefening vraagt je hem zelf uit te rekenen, stap voor stap, om te zien waar hij uit bestaat: de verschillen met het gemiddelde, gekwadrateerd, hun gemiddelde, en de wortel."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the steps of the standard deviation, and give the result back with `return`.",
                      "nl": "Vervang de `...` door de stappen van de standaardafwijking, en geef de uitkomst terug met `return`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with one list, and the `def` line of your function has a different number of names between its parentheses. Keep `(values)`.",
                          "nl": "De check roept `{call}` aan, met één lijst, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(values)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you take the difference of each value from the mean, square those, take their mean, and then the square root?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Neem je het verschil van elke waarde met het gemiddelde, kwadrateer je die, neem je hun gemiddelde, en dan de wortel?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does every step work for any list, of any length?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt elke stap voor elke lijst, van elke lengte?"},
            "correct": {"en": "Your function is right in all {n} cases. Three equal values have a standard deviation of 0: there is no spread at all.",
                        "nl": "Je functie klopt in alle {n} gevallen. Drie gelijke waarden hebben een standaardafwijking van 0: er is helemaal geen spreiding."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_S = "import numpy as np\n\n" + _TIMES_LINE + "\n"
_F = "import numpy as np\n\n\ndef spread(values):\n    x = np.array(values)\n"
CASES = [
    ("slow-trips", _SLOW_START, "empty", None),
    ("slow-trips", _SLOW, "correct", None),
    ("slow-trips", _S + "slow = np.percentile(times, 75)\nslow", "correct", None),
    ("slow-trips", _S + "print(times.max())\nslow = np.quantile(times, 0.75)\nslow", "correct", None),
    ("slow-trips", _S + "slow = np.quantile(times, 75)\nslow", "error", "percent"),
    ("slow-trips", _S + "slow = np.quantile(times, 0,75)\nslow", "error", "decimal-comma"),
    ("slow-trips", _S + "slow = times.quantile(0.75)\nslow", "error", "array-method"),
    ("slow-trips", _S + "slow = np.percentile(times, 0.75)\nslow", "wrong", "percentile"),
    ("slow-trips", _S + "slow = np.quantile(times, 0.9)\nslow", "wrong", "example-fraction"),
    ("slow-trips", _S + "slow = np.quantile(times, 0.25)\nslow", "wrong", "fast-side"),
    ("slow-trips", _S + "slow = np.median(times)\nslow", "wrong", "median"),
    ("slow-trips", _S + "slow = times.max()\nslow", "wrong", "slowest"),
    ("slow-trips", _S + "slow = times.mean()\nslow", "wrong", "mean"),
    ("slow-trips", _S + "slow = 33.5\nslow", "almost", "other-value"),
    ("slow-trips", _S + "slow = np.quantile(times, 0.75)\nprint(slow)", "wrong", "print-last"),
    ("slow-trips", _S + "slow = np.quantile(times, ...)\nslow", "wrong", "dots-left"),
    ("spread", _SPREAD_START, "empty", None),
    ("spread", _SPREAD, "correct", None),
    ("spread", _F + "    m = x.mean()\n    total = 0\n    for v in x:\n        total = total + (v - m) ** 2\n    return (total / len(x)) ** 0.5", "correct", None),
    ("spread", _F + "    d = x - x.mean()\n    return ((d * d).sum() / len(x)) ** 0.5", "correct", None),
    ("spread", _F + "    return x.std()", "almost", "ready-made"),
    ("spread", _F + "    return np.std(x)", "almost", "ready-made"),
    ("spread", _F + "    d = x - x.mean()\n    return (d ** 2).mean()", "wrong", "no-root"),
    ("spread", _F + "    d = x - x.mean()\n    return d.mean()", "wrong", "no-square"),
    ("spread", _F + "    d = x - x.mean()\n    return abs(d).mean()", "wrong", "absolute"),
    ("spread", _F + "    return (x ** 2).mean() ** 0.5", "wrong", "no-mean-first"),
    ("spread", _F + "    d = x - x.mean()\n    return ((d ** 2).sum() / (len(x) - 1)) ** 0.5", "wrong", "one-less"),
    ("spread", _F + "    d = x - x.mean()\n    return (d ** 2).mean() ** 1 / 2", "wrong", "half-power"),
    ("spread", _F + "    return 2.0", "almost", None),
]
