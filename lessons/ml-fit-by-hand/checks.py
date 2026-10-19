# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-fit-by-hand: the bias term with the lowest loss, found with a slider, and the same search
as a function. And the picture: the houses with the line, next to the loss for every value of one parameter."""
import functools
import re

from everything_ai.check import FunctionBox, Mistake, Value

# The houses that were sold, from the lesson ml-vocabulary: area in m², price in thousands of euros.
AREA = [62, 85, 110, 48, 95, 130, 73, 58]
PRICE = [245, 310, 395, 198, 320, 465, 280, 232]

# The slider of each parameter in the picture: (lowest, highest), and the other parameter, which stays fixed.
RANGES = {"w": (2.0, 4.2), "b": (0, 120)}
FIXED = {"w": 50, "b": 3}   # moving w keeps b = 50; moving b keeps w = 3


def loss(w: float, b: float) -> float:
    """The mean squared error of the model w · area + b on the houses."""
    return sum((p - (w * a + b)) ** 2 for a, p in zip(AREA, PRICE)) / len(AREA)


@functools.cache
def _curve(param: str) -> tuple[list[float], list[float]]:
    """The loss for 201 values of the parameter that moves, worked out once."""
    low, high = RANGES[param]
    xs = [low + (high - low) * i / 200 for i in range(201)]
    return xs, [loss(x, FIXED["w"]) if param == "w" else loss(FIXED["b"], x) for x in xs]


def picture(param: str, value: float, labels: dict[str, str]) -> str:
    """Two charts side by side. Left: the houses, the line of the model and its residuals. Right: the loss for every
    value of the parameter that moves, with a point where it is now. Drawn as SVG text, for every stop of the slider
    once (L.explore), each within a millisecond."""
    from everything_ai import chart
    w, b = (value, FIXED["w"]) if param == "w" else (FIXED["b"], value)
    left = chart.Panel((40, 140), (100, 650), title=labels["left"], xlabel=labels["x"], ylabel=labels["y"])
    for x, y in zip(AREA, PRICE):
        left.line([x, x], [w * x + b, y], "residual", 1.6)
    left.line([40, 140], [w * 40 + b, w * 140 + b], "pred")
    left.points(AREA, PRICE, "data", 3.5)
    xs, ys = _curve(param)
    right = chart.Panel(RANGES[param], (0, max(ys) * 1.05), title=labels["right"], xlabel=param, ylabel=labels["loss"])
    right.line(xs, ys, "loss")
    right.points([value], [loss(w, b)], "param", 5.5)
    return chart.figure([left, right], alt=labels["alt"], width=330, height=260, drag=1)   # drag along the loss; two fit a play block side by side


# The lesson is done when these are correct. "direction" is the choice question in the lesson.
REQUIRED = ["best-b", "direction", "search"]

_BEST_B = 58   # with w = 3, as the slider goes in steps of 1


def _b_loss(b) -> float:
    return round(loss(3, b), 2)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SEARCH_START = ("import numpy as np\n\n\ndef best_b(bs, area, price, w):\n    area = np.array(area)\n"
                 "    price = np.array(price)\n    ...\n")
_SEARCH = ("import numpy as np\n\n\ndef best_b(bs, area, price, w):\n    area = np.array(area)\n"
           "    price = np.array(price)\n    losses = []\n    for b in bs:\n        residuals = price - (w * area + b)\n"
           "        losses.append((residuals ** 2).mean())\n    losses = np.array(losses)\n    return bs[losses.argmin()]")


def _search(new: str) -> str:
    return _SEARCH.replace("    losses = np.array(losses)\n    return bs[losses.argmin()]", new)


EXERCISES = {
    "best-b": Value(
        name="best-b",
        start=20,
        solution=_BEST_B,
        outcome=_b_loss,
        compare=lambda got, want: got <= want + 1e-9,
        prose_numbers=True,
        mistakes=[
            Mistake("close", state="almost", matches=lambda got, b: abs(b - _BEST_B) <= 3,
                    en="With b = {value}, the loss is {got}. Is there a b right next to it with an even lower loss?",
                    nl="Met b = {value} is de loss {got}. Is er een b vlak ernaast met een nog lagere loss?"),
            Mistake("too-low", matches=lambda got, b: b < _BEST_B,
                    en="With b = {value}, the loss is {got}. Most houses lie above the line. Which way does b move the line?",
                    nl="Met b = {value} is de loss {got}. De meeste huizen liggen boven de lijn. Welke kant op schuift b de lijn?"),
            Mistake("too-high", matches=lambda got, b: b > _BEST_B,
                    en="With b = {value}, the loss is {got}. Most houses lie below the line. Which way does b move the line?",
                    nl="Met b = {value} is de loss {got}. De meeste huizen liggen onder de lijn. Welke kant op schuift b de lijn?"),
        ],
        texts={
            "missing": {"en": "I cannot see the slider. Run the cell with the slider again.",
                        "nl": "Ik zie de slider niet. Voer de cel met de slider opnieuw uit."},
            "empty": {"en": "The slider is still at b = 20, where the loss is 1537.75. Move it, and watch the loss.",
                      "nl": "De slider staat nog op b = 20, waar de loss 1537,75 is. Verschuif hem, en kijk naar de loss."},
            "wrong": {"en": "With b = {value}, the loss is {got}. There is a b with a lower loss.",
                      "nl": "Met b = {value} is de loss {got}. Er is een b met een lagere loss."},
            "correct": {"en": "With b = {value}, the loss is {got}, and no other b on the slider gives a lower one. The model of the earlier lessons, with b = 50, had a loss of 172.75.",
                        "nl": "Met b = {value} is de loss {got}, en geen andere b op de slider geeft een lagere. Het model uit de vorige lessen, met b = 50, had een loss van 172,75."},
        },
    ),
    "search": FunctionBox(
        name="search",
        function="best_b",
        start=_SEARCH_START,
        solution=_SEARCH,
        visible=[([40, 50, 60, 70], AREA, PRICE, 3)],
        hidden=[([55, 56, 57, 58, 59, 60], AREA, PRICE, 3), ([0, 10, 20], [1, 2, 3], [12, 14, 16], 2), ([5, -5, 0], [1, 2], [2, 4], 2),
                ([2.5, 3.5, 4.5], [10, 20, 30], [33, 64, 93], 3), ([-2, 0, 2], [1, 2], [2, 4], 2), ([0, 1, 2, 3], [1, 2, 3], [1, 2, 9], 1)],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("list-argmin", error=lambda e: isinstance(e, AttributeError) and "'list' object has no attribute 'arg" in str(e),
                    en="With `{call}`, your function stops: a list has no `.argmin()`. That is a method of an array. Make an array of your list first, with `np.array(...)`.",
                    nl="Met `{call}` stopt je functie: een lijst heeft geen `.argmin()`. Dat is een methode van een array. Maak eerst een array van je lijst, met `np.array(...)`."),
            Mistake("fixed-w", state="almost", wrong=_SEARCH.replace("(w * area + b)", "(3 * area + b)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function works with w = 3, whatever `w` it gets. Use the name `w`, then it works for every weight.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je functie rekent met w = 3, welke `w` hij ook krijgt. Gebruik de naam `w`, dan werkt hij voor elke weight."),
            Mistake("loss", wrong=_search("    return min(losses)"),
                    en="`{call}` gives `{got}`: the lowest loss itself. The task asks which b gives it.",
                    nl="`{call}` geeft `{got}`: de laagste loss zelf. De opdracht vraagt welke b die geeft."),
            Mistake("place", wrong=_search("    losses = np.array(losses)\n    return losses.argmin()"),
                    en="`{call}` gives `{got}`: the place of the lowest loss in the list. The task asks for the b at that place: `bs[...]`.",
                    nl="`{call}` geeft `{got}`: de plek van de laagste loss in de lijst. De opdracht vraagt om de b op die plek: `bs[...]`."),
            Mistake("highest", wrong=_search("    losses = np.array(losses)\n    return bs[losses.argmax()]"),
                    en="`{call}` gives `{got}`: the b with the highest loss. The best b has the lowest.",
                    nl="`{call}` geeft `{got}`: de b met de hoogste loss. De beste b heeft de laagste."),
            Mistake("no-square", wrong=_SEARCH.replace("losses.append((residuals ** 2).mean())", "losses.append(residuals.mean())"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the loss is the mean of the squared residuals. Without the square, plus and minus cancel each other out.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de loss is het gemiddelde van de gekwadrateerde residuen. Zonder kwadraat vallen plus en min tegen elkaar weg."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the search: the loss for every b in `bs`, and then the b with the lowest loss, given back with `return`.",
                      "nl": "Vervang de `...` door de zoektocht: de loss voor elke b in `bs`, en dan de b met de laagste loss, teruggegeven met `return`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Keep `(bs, area, price, w)`.",
                          "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(bs, area, price, w)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you work out the loss for every b, and give back the b with the lowest?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken je de loss uit voor elke b, en geef je de b met de laagste terug?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for any list of b's, and for other houses and weights?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elke lijst met b's, en voor andere huizen en weights?"},
            "correct": {"en": "Your function is right in all {n} cases. It does what you did with the slider: try every b, and keep the one with the lowest loss.",
                        "nl": "Je functie klopt in alle {n} gevallen. Hij doet wat jij met de slider deed: elke b proberen, en die met de laagste loss houden."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner set or typed, state, known mistake).
_F = "import numpy as np\n\n\ndef best_b(bs, area, price, w):\n    area = np.array(area)\n    price = np.array(price)\n"
CASES = [
    ("best-b", 20, "empty", None),
    ("best-b", 58, "correct", None),
    ("best-b", 57, "almost", "close"),
    ("best-b", 61, "almost", "close"),
    ("best-b", 50, "wrong", "too-low"),
    ("best-b", 0, "wrong", "too-low"),
    ("best-b", 90, "wrong", "too-high"),
    ("search", _SEARCH_START, "empty", None),
    ("search", _SEARCH, "correct", None),
    ("search", _F + "    best = bs[0]\n    lowest = ((price - (w * area + best)) ** 2).mean()\n    for b in bs:\n"
               "        loss = ((price - (w * area + b)) ** 2).mean()\n        if loss < lowest:\n            best = b\n"
               "            lowest = loss\n    return best", "correct", None),
    ("search", _F + "    losses = [((price - (w * area + b)) ** 2).mean() for b in bs]\n    return bs[np.argmin(losses)]", "correct", None),
    ("search", _search("    return min(losses)"), "wrong", "loss"),
    ("search", _search("    losses = np.array(losses)\n    return losses.argmin()"), "wrong", "place"),
    ("search", _search("    losses = np.array(losses)\n    return bs[losses.argmax()]"), "wrong", "highest"),
    ("search", _SEARCH.replace("losses.append((residuals ** 2).mean())", "losses.append(residuals.mean())"), "wrong", "no-square"),
    ("search", _F + "    return 60", "almost", None),
    ("search", _SEARCH.replace("(w * area + b)", "(3 * area + b)"), "almost", "fixed-w"),
    ("search", _search("    return bs[losses.argmin()]"), "error", "list-argmin"),
    ("search", _SEARCH.replace("(residuals ** 2).mean()", "np.abs(residuals).mean()"), "almost", None),
]
