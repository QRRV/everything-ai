# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson np-axis: the mean temperature of each day, along each row, and a model's prediction for
each photo, with argmax along each row. And the numbers in the picture."""
import re

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake

# The picture: a matrix, and its sums along each axis.
PICTURE = [[1, 2, 3], [4, 5, 6]]


def picture() -> tuple[list[list[str]], list[str], list[str]]:
    """The matrix, its sums per column (axis 0) and its sums per row (axis 1), as text."""
    down = [sum(col) for col in zip(*PICTURE)]
    across = [sum(row) for row in PICTURE]
    return [[str(x) for x in row] for row in PICTURE], [str(x) for x in down], [str(x) for x in across]


# The lesson is done when these are correct. "week-totals" is the choice question in the lesson.
REQUIRED = ["week-totals", "day-means", "predictions"]

_TEMPS_LINE = "temps = np.array([[10, 16, 13], [8, 14, 11], [12, 18, 15], [9, 15, 12]])"
_MEANS_START = "import numpy as np\n\n" + _TEMPS_LINE + "\nday_means = ...\nday_means\n"
_MEANS = "import numpy as np\n\n" + _TEMPS_LINE + "\nday_means = temps.mean(axis=1)\nday_means"


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_AXIS_TWO = Mistake("axis-two", error=lambda e: type(e).__name__ == "AxisError",
                    en="A matrix has two axes, 0 and 1: counting starts at 0, so there is no axis 2 or higher. Which of the two goes along each row? See step 3 of the recipe.",
                    nl="Een matrix heeft twee axes, 0 en 1: tellen begint bij 0, dus er is geen axis 2 of hoger. Welke van de twee gaat langs elke rij? Kijk bij stap 3 van het recept.")
_STILL_LIST = Mistake("still-list", error=lambda e: isinstance(e, AttributeError) and "'list'" in str(e),
                      en="With `{call}`, `scores` is still a list, and a list has no `.argmax`. Keep the line `scores = np.array(scores)` above it.",
                      nl="Met `{call}` is `scores` nog een lijst, en een lijst heeft geen `.argmax`. Houd de regel `scores = np.array(scores)` erboven.")

_PRED_START = "import numpy as np\n\n\ndef predictions(scores):\n    scores = np.array(scores)\n    ...\n"
_PRED = "import numpy as np\n\n\ndef predictions(scores):\n    scores = np.array(scores)\n    return scores.argmax(axis=1)"

EXERCISES = {
    "day-means": CodeBox(
        name="day-means",
        start=_MEANS_START,
        solution=_MEANS,
        keep=["import numpy as np", _TEMPS_LINE],
        # The check also runs the box with other temperatures: numbers typed in by hand do not change with them.
        others=[{"temps": Code("np.array([[1, 2, 6], [4, 4, 4]])")}],
        first=[_DOTS],
        mistakes=[
            _AXIS_TWO,
            Mistake("per-time", wrong=_MEANS.replace("axis=1", "axis=0"),
                    en="You get three numbers: one per time of day, from going down the rows with `axis=0`. The task asks for one mean per day, so per row: along each row.",
                    nl="Je krijgt drie getallen: één per dagdeel, door met `axis=0` langs de rijen omlaag te gaan. De opdracht vraagt één gemiddelde per dag, dus per rij: langs elke rij."),
            Mistake("no-axis", wrong=_MEANS.replace("temps.mean(axis=1)", "temps.mean()"),
                    en="That is one number: the mean of all twelve temperatures. The task asks for one mean per day: tell `.mean` which axis to work along.",
                    nl="Dat is één getal: het gemiddelde van alle twaalf temperaturen. De opdracht vraagt één gemiddelde per dag: geef `.mean` mee langs welke axis hij moet werken."),
            Mistake("sum-not-mean", wrong=_MEANS.replace("temps.mean(axis=1)", "temps.sum(axis=1)"),
                    en="These are the totals of each day: that is what `.sum` gives. The mean is the total divided by how many numbers there are: use `.mean`.",
                    nl="Dit zijn de totalen van elke dag: dat geeft `.sum`. Het gemiddelde is het totaal gedeeld door het aantal getallen: gebruik `.mean`."),
            Mistake("divide-by-days", wrong=_MEANS.replace("temps.mean(axis=1)", "temps.sum(axis=1) / len(temps)"),
                    en="You divide each day's total by 4, the number of days: `len(temps)` counts the rows. A day has 3 temperatures; `.mean` divides by the right number for you.",
                    nl="Je deelt het totaal van elke dag door 4, het aantal dagen: `len(temps)` telt de rijen. Een dag heeft 3 temperaturen; `.mean` deelt voor jou door het goede aantal."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the mean of each day: one mean per row of `temps`.",
                      "nl": "Vervang de `...` door het gemiddelde van elke dag: één gemiddelde per rij van `temps`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
            "no_value": {"en": "The last line of the box shows nothing. Put `day_means` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `day_means` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives:\n\n```\n{got}\n```\n\nand it should be four means, one per day: the first day, 10, 16 and 13, has a mean of 13. Does it take the mean, and does it go along each row?",
                      "nl": "Je laatste regel geeft:\n\n```\n{got}\n```\n\nen dat moeten vier gemiddelden zijn, één per dag: de eerste dag, 10, 16 en 13, heeft een gemiddelde van 13. Neemt hij het gemiddelde, en gaat hij langs elke rij?"},
            "other": {"en": "Right for these temperatures. But with {where}, your last line gives:\n\n```\n{got}\n```\n\nand it should give `{want}`. Do you calculate the means from `temps` itself, or did you type numbers in?",
                      "nl": "Goed voor deze temperaturen. Maar met {where} geeft je laatste regel:\n\n```\n{got}\n```\n\nen dat moet `{want}` zijn. Bereken je de gemiddelden uit `temps` zelf, of typte je getallen in?"},
            "correct": {"en": "`axis=1` goes along each row, so you get one mean per day: `array([13., 11., 15., 12.])`.",
                        "nl": "`axis=1` gaat langs elke rij, dus krijg je één gemiddelde per dag: `array([13., 11., 15., 12.])`."},
        },
    ),
    "predictions": FunctionBox(
        name="predictions",
        function="predictions",
        start=_PRED_START,
        solution=_PRED,
        visible=[([[0.2, 0.7, 0.1], [0.6, 0.3, 0.1], [0.1, 0.2, 0.7], [0.3, 0.3, 0.4]],)],
        hidden=[([[0.1, 0.9]],), ([[5, 1, 2], [0, 0, 3]],), ([[0.5, 0.5, 0.0]],), ([[1], [2]],)],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            _AXIS_TWO, _STILL_LIST,
            Mistake("per-animal", wrong=_PRED.replace("axis=1", "axis=0"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. With `axis=0`, you go down the rows: for each animal, which photo gave it the highest score. The task asks for one answer per photo, so along each row.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Met `axis=0` ga je langs de rijen omlaag: per dier, welke foto het de hoogste score gaf. De opdracht vraagt één antwoord per foto, dus langs elke rij."),
            Mistake("no-axis", wrong=_PRED.replace("scores.argmax(axis=1)", "scores.argmax()"),
                    en="`{call}` gives `{got}`: one number, the place of the highest score in the whole array, counted row after row. The task asks for one answer per photo: tell `.argmax` which axis to work along.",
                    nl="`{call}` geeft `{got}`: één getal, de plek van de hoogste score in de hele array, rij na rij geteld. De opdracht vraagt één antwoord per foto: geef `.argmax` mee langs welke axis hij moet werken."),
            Mistake("max-not-argmax", wrong=_PRED.replace("argmax", "max"),
                    en="`{call}` gives `{got}`: the highest scores themselves. The task asks where the highest score is, its index: 0 for cat, 1 for dog, 2 for bird. That is `.argmax`.",
                    nl="`{call}` geeft `{got}`: de hoogste scores zelf. De opdracht vraagt waar de hoogste score staat, zijn index: 0 voor kat, 1 voor hond, 2 voor vogel. Dat is `.argmax`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the index of the highest score in each row, given back with `return`.",
                      "nl": "Vervang de `...` door de index van de hoogste score in elke rij, teruggegeven met `return`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with one list of scores, and the `def` line of your function has a different number of names between its parentheses. Keep `(scores)`.",
                          "nl": "De check roept `{call}` aan, met één lijst scores, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(scores)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does your function give, for each row, the index of its highest score?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Geeft je functie voor elke rij de index van zijn hoogste score?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for every number of photos and animals, and when two scores are equally high?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elk aantal foto's en dieren, en als twee scores even hoog zijn?"},
            "correct": {"en": "Your function is right in all {n} cases. When two scores are equally high, `.argmax` gives the first of them.",
                        "nl": "Je functie klopt in alle {n} gevallen. Zijn twee scores even hoog, dan geeft `.argmax` de eerste."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_T = "import numpy as np\n\n" + _TEMPS_LINE + "\n"
_F = "import numpy as np\n\n\ndef predictions(scores):\n    scores = np.array(scores)\n"
CASES = [
    ("day-means", _MEANS_START, "empty", None),
    ("day-means", _MEANS, "correct", None),
    ("day-means", _T + "day_means = temps.sum(axis=1) / 3\nday_means", "correct", None),
    ("day-means", _T + "day_means = np.mean(temps, axis=1)\nday_means", "correct", None),
    ("day-means", _T + "day_means = temps.mean(axis=0)\nday_means", "wrong", "per-time"),
    ("day-means", _T + "day_means = temps.mean()\nday_means", "wrong", "no-axis"),
    ("day-means", _T + "day_means = temps.sum(axis=1)\nday_means", "wrong", "sum-not-mean"),
    ("day-means", _T + "day_means = temps.mean(axis=2)\nday_means", "error", "axis-two"),
    ("day-means", _T + "day_means = np.array([13, 11, 15, 12])\nday_means", "almost", "other-value"),
    ("day-means", _T + "day_means = temps.sum(axis=1) / len(temps)\nday_means", "wrong", "divide-by-days"),
    ("day-means", _T + "day_means = []\nfor day in temps:\n    day_means.append(day.mean())\nday_means", "correct", None),
    ("day-means", _T + "means = temps.mean(axis=1)\nday_means = means\nday_means", "correct", None),
    ("day-means", _T + "day_means = temps.mean(axis=3)\nday_means", "error", "axis-two"),
    ("day-means", _T + "day_means = temps.max(axis=1)\nday_means", "wrong", None),
    ("predictions", _PRED_START, "empty", None),
    ("predictions", _PRED, "correct", None),
    ("predictions", _F + "    return np.argmax(scores, axis=1)", "correct", None),
    ("predictions", _PRED.replace("axis=1", "axis=0"), "wrong", "per-animal"),
    ("predictions", _PRED.replace("scores.argmax(axis=1)", "scores.argmax()"), "wrong", "no-axis"),
    ("predictions", _PRED.replace("argmax", "max"), "wrong", "max-not-argmax"),
    ("predictions", _PRED.replace("axis=1", "axis=2"), "error", "axis-two"),
    ("predictions", "import numpy as np\n\n\ndef predictions(scores):\n    return scores.argmax(axis=1)", "error", "still-list"),
    ("predictions", _F + "    out = []\n    for row in scores:\n        best = 0\n        for i in range(len(row)):\n            if row[i] > row[best]:\n                best = i\n        out.append(best)\n    return out", "correct", None),
    ("predictions", _F + "    out = []\n    for row in scores:\n        best = 0\n        for i in range(len(row)):\n            if row[i] >= row[best]:\n                best = i\n        out.append(best)\n    return out", "almost", None),
]
