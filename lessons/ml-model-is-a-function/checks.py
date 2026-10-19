# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-model-is-a-function: the two parameters of a taxi fare, set in a box, and a model with
two features, each with its own weight. And the picture of the houses with a line."""
import re

from everything_ai.check import CodeBox, FunctionBox, Mistake

# The houses that were sold, from the lesson ml-vocabulary: area in m², price in thousands of euros.
AREA = [62, 85, 110, 48, 95, 130, 73, 58]
PRICE = [245, 310, 395, 198, 320, 465, 280, 232]
W, B = 3, 50


def picture(labels: dict[str, str]) -> str:
    """The sold houses as points, and the line of the model w * x + b with W and B."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    fig = plt.figure(figsize=(5.6, 3.2))
    plt.scatter(AREA, PRICE, color=plots.DATA, zorder=3)
    xs = [40, 140]
    plt.plot(xs, [W * x + B for x in xs], color=plots.PRED, linestyle="-")
    plt.text(141, W * 140 + B, labels["line"], va="center", ha="left", color=plots.PRED_TEXT)
    plt.xlim(35, 165)
    plt.xlabel(labels["x"])
    plt.ylabel(labels["y"])
    plt.title(labels["title"])
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


# The lesson is done when these are correct. "weight-meaning" is the choice question in the lesson.
REQUIRED = ["weight-meaning", "taxi", "two-features"]

_TAXI_START = "def fare(km, w, b):\n    return w * km + b\n\n\nw = ...\nb = ...\nkm = 12\nfare(km, w, b)\n"
_TAXI = "def fare(km, w, b):\n    return w * km + b\n\n\nw = 2.5\nb = 4\nkm = 12\nfare(km, w, b)"

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _taxi(w, b) -> str:
    return _TAXI.replace("w = 2.5\nb = 4", f"w = {w}\nb = {b}")


_TWO_START = "def predict(area, rooms, w_area, w_rooms, b):\n    ...\n"
_TWO = "def predict(area, rooms, w_area, w_rooms, b):\n    return w_area * area + w_rooms * rooms + b"


def _two(new: str) -> str:
    return _TWO.replace("    return w_area * area + w_rooms * rooms + b", new)


EXERCISES = {
    "taxi": CodeBox(
        name="taxi",
        start=_TAXI_START,
        solution=_TAXI,
        keep=["def fare(km, w, b):", "    return w * km + b", "km = 12", "fare(km, w, b)"],
        # The check also works out a ride of another length: the parameters stay, the feature changes.
        others=[{"km": 3}],
        first=[_DOTS, Mistake("decimal-comma", matches=lambda got, text: re.search(r"^[ \t]*[wb][ \t]*=[ \t]*-?\d+,[ \t]*\d", text, re.MULTILINE) is not None,
                              en="Python writes a decimal number with a point, not a comma: `2.5`, not `2,5`.",
                              nl="Python schrijft een kommagetal met een punt, niet met een komma: `2.5`, niet `2,5`.")],
        mistakes=[
            Mistake("swapped", wrong=_taxi(4, 2.5),
                    en="Your last line gives `{got}`: the start price is now multiplied by the kilometers. The weight `w` goes with each kilometer, and the bias term `b` is paid once, at the start.",
                    nl="Je laatste regel geeft `{got}`: nu wordt de startprijs met de kilometers vermenigvuldigd. De weight `w` hoort bij elke kilometer, en de bias term `b` betaal je één keer, bij het begin."),
            Mistake("no-start", wrong=_taxi(2.5, 0),
                    en="Your last line gives `{got}`: only the kilometers. Every ride also costs 4 euros to start: that is the bias term `b`.",
                    nl="Je laatste regel geeft `{got}`: alleen de kilometers. Elke rit kost ook 4 euro om te beginnen: dat is de bias term `b`."),
            Mistake("cents", wrong=_taxi(250, 400),
                    en="Your last line gives `{got}`: those are cents, not euros. The task gives the prices in euros: `2.5` and `4`.",
                    nl="Je laatste regel geeft `{got}`: dat zijn centen, geen euro's. De opdracht geeft de prijzen in euro's: `2.5` en `4`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...` with the parameters of the taxi: `w` and `b`.",
                      "nl": "Vervang de twee `...` door de parameters van de taxi: `w` en `b`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. Only `w` and `b` change.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. Alleen `w` en `b` veranderen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `fare(km, w, b)` back on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `fare(km, w, b)` terug op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and a ride of 12 kilometers should cost 4 euros plus 12 times 2.5 euros. Which number goes with every kilometer, and which one is paid once?",
                      "nl": "Je laatste regel geeft `{got}`, en een rit van 12 kilometer moet 4 euro kosten plus 12 keer 2,5 euro. Welk getal hoort bij elke kilometer, en welk betaal je één keer?"},
            "other": {"en": "Right for 12 kilometers. But with {where}, your last line gives `{got}`, and it should give `{want}`. Are `w` and `b` the prices of the taxi, the same for every ride?",
                      "nl": "Goed voor 12 kilometer. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Zijn `w` en `b` de prijzen van de taxi, voor elke rit dezelfde?"},
            "other_error": {"en": "Right for 12 kilometers. But with {where}, your code stops with an error. Are `w` and `b` plain numbers?",
                            "nl": "Goed voor 12 kilometer. Maar met {where} stopt je code met een fout. Zijn `w` en `b` gewone getallen?"},
            "correct": {"en": "A ride of 12 kilometers costs 34 euros: 4 to start, and 12 times 2.5. The check also tried a ride of 3 kilometers, 11.5 euros, with the same two parameters.",
                        "nl": "Een rit van 12 kilometer kost 34 euro: 4 om te beginnen, en 12 keer 2,5. De check probeerde ook een rit van 3 kilometer, 11,5 euro, met dezelfde twee parameters."},
        },
    ),
    "two-features": FunctionBox(
        name="two-features",
        function="predict",
        start=_TWO_START,
        solution=_TWO,
        visible=[(100, 4, 2, 25, 45)],
        hidden=[(73, 3, 2.5, 10, 30), (60, 2, 2.5, 10, 30), (100, 4, 3, 0, 50), (0, 0, 2.5, 10, 30), (85, 3, 2, 15, -20)],
        first=[_DOTS],
        mistakes=[
            Mistake("no-bias", wrong=_two("    return w_area * area + w_rooms * rooms"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the bias term `b` is missing. It is added once, whatever the features are.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de bias term `b` ontbreekt. Die tel je één keer op, wat de features ook zijn."),
            Mistake("swapped-weights", wrong=_two("    return w_area * rooms + w_rooms * area + b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: each weight is multiplied by the other feature. `w_area` goes with `area`, and `w_rooms` with `rooms`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elke weight wordt met de andere feature vermenigvuldigd. `w_area` hoort bij `area`, en `w_rooms` bij `rooms`."),
            Mistake("one-weight", wrong=_two("    return w_area * (area + rooms) + b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: one weight for both features. Each feature has its own weight: `w_area * area` and `w_rooms * rooms`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: één weight voor beide features. Elke feature heeft zijn eigen weight: `w_area * area` en `w_rooms * rooms`."),
            Mistake("added-weights", wrong=_two("    return w_area + area + w_rooms + rooms + b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function adds the weights to the features. A weight multiplies its feature: `w_area * area`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je functie telt de weights bij de features op. Een weight vermenigvuldig je met zijn feature: `w_area * area`."),
            Mistake("area-only", wrong=_two("    return w_area * area + b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the rooms are missing. Each feature adds its own part: also `w_rooms * rooms`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de kamers ontbreken. Elke feature telt zijn eigen deel mee: ook `w_rooms * rooms`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the prediction of the model, given back with `return`.",
                      "nl": "Vervang de `...` door de voorspelling van het model, teruggegeven met `return`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with five values, and the `def` line of your function has a different number of names between its parentheses. Keep `(area, rooms, w_area, w_rooms, b)`.",
                          "nl": "De check roept `{call}` aan, met vijf waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(area, rooms, w_area, w_rooms, b)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does each feature get multiplied by its own weight, with the bias term added once?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Wordt elke feature met zijn eigen weight vermenigvuldigd, en tel je de bias term één keer op?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for every house and every choice of the parameters?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elk huis en elke keuze van de parameters?"},
            "correct": {"en": "Your model is right in all {n} cases. With two features it has three parameters: a weight for each feature, and one bias term.",
                        "nl": "Je model klopt in alle {n} gevallen. Met twee features heeft het drie parameters: een weight voor elke feature, en één bias term."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_T = "def fare(km, w, b):\n    return w * km + b\n\n\n"
_F = "def predict(area, rooms, w_area, w_rooms, b):\n"
CASES = [
    ("taxi", _TAXI_START, "empty", None),
    ("taxi", _TAXI, "correct", None),
    ("taxi", _T + "w = 2.50\nb = 4.0\nkm = 12\nfare(km, w, b)", "correct", None),
    ("taxi", _taxi(4, 2.5), "wrong", "swapped"),
    ("taxi", _taxi(2.5, 0), "wrong", "no-start"),
    ("taxi", _taxi(250, 400), "wrong", "cents"),
    ("taxi", _taxi(2, 10), "almost", "other-value"),
    ("taxi", _taxi(3, 4), "wrong", None),
    ("taxi", _T + "w = 2.5\nb = ...\nkm = 12\nfare(km, w, b)", "wrong", "dots-left"),
    ("taxi", _T + "w = 2,5\nb = 4\nkm = 12\nfare(km, w, b)", "wrong", "decimal-comma"),
    ("taxi", _T + "w = 2,5\nb = 4,0\nkm = 12\nfare(km, w, b)", "wrong", "decimal-comma"),
    ("two-features", _TWO_START, "empty", None),
    ("two-features", _TWO, "correct", None),
    ("two-features", _F + "    y_pred = w_area * area + w_rooms * rooms + b\n    return y_pred", "correct", None),
    ("two-features", _F + "    return b + rooms * w_rooms + area * w_area", "correct", None),
    ("two-features", _two("    return w_area * area + w_rooms * rooms"), "wrong", "no-bias"),
    ("two-features", _two("    return w_area * rooms + w_rooms * area + b"), "wrong", "swapped-weights"),
    ("two-features", _two("    return w_area * (area + rooms) + b"), "wrong", "one-weight"),
    ("two-features", _two("    return w_area + area + w_rooms + rooms + b"), "wrong", "added-weights"),
    ("two-features", _two("    return w_area * area + b"), "wrong", "area-only"),
    ("two-features", _F + "    print(w_area * area + w_rooms * rooms + b)", "wrong", "prints"),
    ("two-features", _F + "    return 345", "almost", None),
]
