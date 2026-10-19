# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-residuals: the residuals of taxi rides, worked out in a box, and a function that finds the
house whose price the model predicts too low by the most. And the picture of the residuals."""
import re

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake

# The houses that were sold, from the lesson ml-vocabulary, and the model of ml-model-is-a-function.
AREA = [62, 85, 110, 48, 95, 130, 73, 58]
PRICE = [245, 310, 395, 198, 320, 465, 280, 232]
W, B = 3, 50


def picture(labels: dict[str, str]) -> str:
    """The houses as points, the line of the model, and each residual as a vertical stroke from the line to the point."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    fig = plt.figure(figsize=(5.6, 4.4))
    for x, y in zip(AREA, PRICE):
        plt.plot([x, x], [W * x + B, y], color=plots.RESIDUAL, linestyle="-", linewidth=2.2, zorder=4)
    plt.scatter(AREA, PRICE, color=plots.DATA, s=16, zorder=3)
    xs = [40, 140]
    plt.plot(xs, [W * x + B for x in xs], color=plots.PRED, linestyle="-")
    big = max(range(len(AREA)), key=lambda i: PRICE[i] - (W * AREA[i] + B))
    plt.text(AREA[big] - 2, (PRICE[big] + W * AREA[big] + B) / 2, labels["biggest"], va="center", ha="right", color=plots.RESIDUAL_TEXT)
    plt.xlabel(labels["x"])
    plt.ylabel(labels["y"])
    plt.title(labels["title"])
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


# The lesson is done when these are correct. "one-residual" is the choice question in the lesson.
REQUIRED = ["one-residual", "residuals", "too-low"]

# Five rides of the taxi of ml-model-is-a-function: distance in km, and the fare that was paid, in euros.
_HEAD = ("import numpy as np\n\nkm = np.array([3, 12, 7, 20, 5])\n"
         "fare = np.array([12, 33, 23, 56, 15])\nw = 2.5\nb = 4\n")
_RES_START = _HEAD + "residuals = ...\nresiduals\n"
_RES = _HEAD + "residuals = fare - (w * km + b)\nresiduals"

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _res(new: str) -> str:
    return _RES.replace("residuals = fare - (w * km + b)", new)


_LOW_START = ("import numpy as np\n\n\ndef too_low(area, price, w, b):\n    area = np.array(area)\n"
              "    price = np.array(price)\n    ...\n")
_LOW = ("import numpy as np\n\n\ndef too_low(area, price, w, b):\n    area = np.array(area)\n"
        "    price = np.array(price)\n    residuals = price - (w * area + b)\n    return residuals.argmax()")


def _low(new: str) -> str:
    return _LOW.replace("    residuals = price - (w * area + b)\n    return residuals.argmax()", new)


EXERCISES = {
    "residuals": CodeBox(
        name="residuals",
        start=_RES_START,
        solution=_RES,
        keep=["import numpy as np", "km = np.array([3, 12, 7, 20, 5])", "fare = np.array([12, 33, 23, 56, 15])", "w = 2.5", "b = 4"],
        # The check also runs the box with other parameters: numbers typed in by hand do not change with them.
        others=[{"w": 2, "b": 5}],
        first=[_DOTS],
        mistakes=[
            Mistake("flipped", wrong=_res("residuals = (w * km + b) - fare"),
                    en="Your last line gives `{got}`: every sign is flipped. That is the prediction minus the label; a residual is the label minus the prediction, `y - y_pred`.",
                    nl="Je laatste regel geeft `{got}`: elk teken staat andersom. Dat is de voorspelling min het label; een residu is het label min de voorspelling, `y - y_pred`."),
            Mistake("brackets", wrong=_res("residuals = fare - w * km + b"),
                    en="Your last line gives `{got}`. Without brackets, Python subtracts only `w * km`, and then adds `b`. Put the whole prediction between brackets: `fare - (w * km + b)`.",
                    nl="Je laatste regel geeft `{got}`. Zonder haakjes trekt Python alleen `w * km` af, en telt het daarna `b` op. Zet de hele voorspelling tussen haakjes: `fare - (w * km + b)`."),
            Mistake("no-bias", wrong=_res("residuals = fare - w * km"),
                    en="Your last line gives `{got}`: the bias term `b` is missing from the prediction. The prediction is `w * km + b`.",
                    nl="Je laatste regel geeft `{got}`: de bias term `b` ontbreekt in de voorspelling. De voorspelling is `w * km + b`."),
            Mistake("no-sign", wrong=_res("residuals = abs(fare - (w * km + b))"),
                    en="Your last line gives `{got}`: every residual without its sign. A residual keeps its sign: positive when the model predicted too low, negative when too high.",
                    nl="Je laatste regel geeft `{got}`: elk residu zonder zijn teken. Een residu houdt zijn teken: positief als het model te laag voorspelde, negatief als het te hoog voorspelde."),
            Mistake("predictions", wrong=_res("residuals = w * km + b"),
                    en="Your last line gives `{got}`: the predictions themselves. A residual is how far each prediction is off: the label minus the prediction.",
                    nl="Je laatste regel geeft `{got}`: de voorspellingen zelf. Een residu is hoe ver elke voorspelling ernaast zit: het label min de voorspelling."),
            Mistake("one-number", matches=lambda got, text: type(got).__name__ in ("int", "float", "int64", "float64"),
                    en="Your last line gives one number, `{got}`. A residual belongs to one example: the task asks for one residual per ride, an array of five.",
                    nl="Je laatste regel geeft één getal, `{got}`. Een residu hoort bij één voorbeeld: de opdracht vraagt om één residu per rit, een array van vijf."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the residuals of the five rides: the fare minus the prediction.",
                      "nl": "Vervang de `...` door de residuen van de vijf ritten: de ritprijs min de voorspelling."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other parameters by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere parameters."},
            "no_value": {"en": "The last line of the box shows nothing. Put `residuals` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `residuals` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be, for each ride, its fare minus its prediction. Do you work out the predictions with `w` and `b`, and subtract them whole?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet voor elke rit de ritprijs min de voorspelling zijn. Reken je de voorspellingen uit met `w` en `b`, en trek je ze er in hun geheel van af?"},
            "other": {"en": "Right for these parameters. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work out the predictions with `w` and `b` themselves?",
                      "nl": "Goed voor deze parameters. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je de voorspellingen uit met `w` en `b` zelf?"},
            "other_error": {"en": "Right for these parameters. But with {where}, your code stops with an error. Do you work out the predictions with `w` and `b` themselves?",
                            "nl": "Goed voor deze parameters. Maar met {where} stopt je code met een fout. Reken je de voorspellingen uit met `w` en `b` zelf?"},
            "correct": {"en": "Five residuals, in euros: 0.5, −1, 1.5, 2 and −1.5. Three rides cost more than the model predicted, and two less. The check also tried other parameters, and your code followed them.",
                        "nl": "Vijf residuen, in euro: 0,5, −1, 1,5, 2 en −1,5. Drie ritten kostten meer dan het model voorspelde, en twee minder. De check probeerde ook andere parameters, en je code volgde ze."},
        },
    ),
    "too-low": FunctionBox(
        name="too-low",
        function="too_low",
        start=_LOW_START,
        solution=_LOW,
        visible=[(AREA, PRICE, W, B)],
        hidden=[([10, 20, 30], [40, 0, 95], 3, 0), ([50, 100], [200, 400], 3, 50), (AREA, PRICE, 3.5, 20), ([1, 2, 3], [4, 9, 8], 2, 3)],
        first=[_DOTS],
        mistakes=[
            Mistake("too-high", wrong=_low("    residuals = price - (w * area + b)\n    return residuals.argmin()"),
                    en="`{call}` gives `{got}`: the house the model predicts too high by the most, with the most negative residual. The answer should be `{want}`. A residual is the label minus the prediction, and the task asks for too low: the largest residual.",
                    nl="`{call}` geeft `{got}`: het huis dat het model het verst te hoog voorspelt, met het meest negatieve residu. Het antwoord moet `{want}` zijn. Een residu is het label min de voorspelling, en de opdracht vraagt om te laag: het grootste residu."),
            Mistake("value", wrong=_low("    residuals = price - (w * area + b)\n    return residuals.max()"),
                    en="`{call}` gives `{got}`: the largest residual itself. The task asks which house: its place in the array, from 0.",
                    nl="`{call}` geeft `{got}`: het grootste residu zelf. De opdracht vraagt welk huis: zijn plek in de array, vanaf 0."),
            Mistake("either-way", wrong=_low("    residuals = price - (w * area + b)\n    return abs(residuals).argmax()"),
                    en="`{call}` gives `{got}`: the house that is furthest off, too low or too high. The answer should be `{want}`. The task asks only for too low: the largest residual, with its sign.",
                    nl="`{call}` geeft `{got}`: het huis dat het verst ernaast zit, te laag of te hoog. Het antwoord moet `{want}` zijn. De opdracht vraagt alleen om te laag: het grootste residu, met zijn teken."),
            Mistake("highest-price", wrong=_low("    return price.argmax()"),
                    en="`{call}` gives `{got}`: the most expensive house. The task asks for the house whose price the model predicts too low by the most: work out the residuals first.",
                    nl="`{call}` geeft `{got}`: het duurste huis. De opdracht vraagt om het huis waarvan het model de prijs het verst te laag voorspelt: reken eerst de residuen uit."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the place of the house the model predicts too low by the most, given back with `return`.",
                      "nl": "Vervang de `...` door de plek van het huis dat het model het verst te laag voorspelt, teruggegeven met `return`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Keep `(area, price, w, b)`.",
                          "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(area, price, w, b)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you work out the residuals, the price minus the prediction, and give back the place of the largest one?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken je de residuen uit, de prijs min de voorspelling, en geef je de plek van het grootste terug?"},
            "almost": {"en": "Right for the houses. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for other houses and other parameters too?",
                       "nl": "Goed voor de huizen. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij ook voor andere huizen en andere parameters?"},
            "correct": {"en": "Your function is right in all {n} cases. With the model of the lesson, house 5, of 130 square meters, sold for 25 thousand euros more than predicted.",
                        "nl": "Je functie klopt in alle {n} gevallen. Met het model van de les is huis 5, van 130 vierkante meter, verkocht voor 25 duizend euro meer dan voorspeld."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_F = "import numpy as np\n\n\ndef too_low(area, price, w, b):\n    area = np.array(area)\n    price = np.array(price)\n"
CASES = [
    ("residuals", _RES_START, "empty", None),
    ("residuals", _RES, "correct", None),
    ("residuals", _HEAD + "y_pred = w * km + b\nresiduals = fare - y_pred\nresiduals", "correct", None),
    ("residuals", _HEAD + "residuals = [f - (w * k + b) for k, f in zip(km, fare)]\nresiduals", "correct", None),
    ("residuals", _res("residuals = (w * km + b) - fare"), "wrong", "flipped"),
    ("residuals", _res("residuals = fare - w * km + b"), "wrong", "brackets"),
    ("residuals", _res("residuals = fare - w * km"), "wrong", "no-bias"),
    ("residuals", _res("residuals = abs(fare - (w * km + b))"), "wrong", "no-sign"),
    ("residuals", _res("residuals = np.abs(fare - (w * km + b))"), "wrong", "no-sign"),
    ("residuals", _res("residuals = w * km + b"), "wrong", "predictions"),
    ("residuals", _res("residuals = (fare - (w * km + b)).sum()"), "wrong", "one-number"),
    ("residuals", _res("residuals = np.array([0.5, -1, 1.5, 2, -1.5])"), "almost", "other-value"),
    ("residuals", _res("residuals = fare - (2.5 * km + 4)"), "almost", "other-value"),
    ("residuals", _HEAD + "residuals = fare - (w * km + b)\nprint(residuals)", "wrong", "print-last"),
    ("too-low", _LOW_START, "empty", None),
    ("too-low", _LOW, "correct", None),
    ("too-low", _F + "    y_pred = w * area + b\n    return np.argmax(price - y_pred)", "correct", None),
    ("too-low", _low("    residuals = price - (w * area + b)\n    return residuals.argmin()"), "wrong", "too-high"),
    ("too-low", _low("    residuals = (w * area + b) - price\n    return residuals.argmax()"), "wrong", "too-high"),
    ("too-low", _low("    residuals = price - (w * area + b)\n    return residuals.max()"), "wrong", "value"),
    ("too-low", _low("    residuals = price - (w * area + b)\n    return abs(residuals).argmax()"), "wrong", "either-way"),
    ("too-low", _low("    return price.argmax()"), "wrong", "highest-price"),
    ("too-low", _F + "    return 5", "almost", None),
]
