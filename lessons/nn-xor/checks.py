# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-xor: sorting tasks on two inputs into those one straight line can split and those it
cannot, one update step of the perceptron, and the argument why no neuron fires for XOR. And the picture: the four
corners of XOR with the line of a neuron whose weight and bias term move in a field (L.plane), and its mistakes."""
import math
import re

from everything_ai.check import FunctionBox, Mistake, Sort

CORNERS = [(0, 0), (0, 1), (1, 0), (1, 1)]
XOR = [0, 1, 1, 0]


def fires(w2: float, b: float, w1: float = 1.0) -> list[int]:
    # z rounded: the field moves in tenths, and a corner exactly on the line (z = 0) must not fire by float noise
    return [1 if round(w1 * x1 + w2 * x2 + b, 9) > 0 else 0 for x1, x2 in CORNERS]


def wrong_corners(w2: float, b: float) -> list[tuple[int, int]]:
    return [c for c, f, y in zip(CORNERS, fires(w2, b), XOR) if f != y]


def mistakes(w2: float, b: float) -> int:
    return len(wrong_corners(w2, b))


def picture(w2: float, b: float, labels: dict[str, str]) -> str:
    """The four corners of XOR, filled where the label is 1 and open where it is 0; the line where the neuron with the
    weights 1 and w2 and this bias term has z = 0; and a purple ring around each corner it gets wrong."""
    from everything_ai import chart
    lo, hi = -0.3, 1.3
    p = chart.Panel((lo, hi), (lo, hi), title=labels["title"], xlabel="x₁", ylabel="x₂", xticks=[0, 1])
    wrong = wrong_corners(w2, b)   # first, so the dot of its corner stays visible inside each ring
    p.points([x for x, _ in wrong], [y for _, y in wrong], "loss", 13, hollow=True)
    if abs(w2) > 1e-9:   # x₂ = −(x₁ + b) / w₂
        p.line([lo, hi], [-(lo + b) / w2, -(hi + b) / w2], "pred", 2.4)
    else:                # x₁ = −b: a vertical line
        p.line([-b, -b], [lo, hi], "pred", 2.4)
    ones = [c for c, y in zip(CORNERS, XOR) if y]
    zeros = [c for c, y in zip(CORNERS, XOR) if not y]
    p.points([x for x, _ in ones], [y for _, y in ones], "data", 7)
    p.points([x for x, _ in zeros], [y for _, y in zeros], "data", 7, hollow=True)
    return chart.figure([p], alt=labels["alt"], width=340, height=320)


# The lesson is done when these are correct. "follows" is the choice question in the lesson.
REQUIRED = ["one-line", "perceptron-step", "follows"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _close_pair(got, want) -> bool:
    """(the weights, the bias term), the same as the course's within a hair."""
    import numpy as np
    try:
        gw, gb = got
        ww, wb = want
        gw, ww = np.asarray(gw, dtype=float), np.asarray(ww, dtype=float)
        return gw.shape == ww.shape and np.allclose(gw, ww, rtol=1e-9, atol=1e-12) and math.isclose(float(gb), float(wb), rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _plain(v) -> str:
    """A value as the lessons write it: lists and numbers, never array(...) or np.float64(...)."""
    import numpy as np
    if isinstance(v, tuple):
        return "(" + ", ".join(_plain(x) for x in v) + ")"
    try:
        return str((np.round(np.asarray(v, dtype=float), 4) + 0.0).tolist())
    except (TypeError, ValueError):
        return repr(v)


_P_HEAD = ("import numpy as np\n\n\ndef perceptron_step(x, y, w, b, lr):\n    x = np.array(x)\n    w = np.array(w)\n"
           "    z = np.dot(w, x) + b\n    if z > 0:\n        pred = 1\n    else:\n        pred = 0\n")
_P_START = _P_HEAD + "    ...\n    return w, b\n"
_P_BODY = "    w = w + lr * (y - pred) * x\n    b = b + lr * (y - pred)\n    return w, b"
_P = _P_HEAD + _P_BODY


def _p(body: str) -> str:
    return _P_HEAD + body


EXERCISES = {
    "one-line": Sort(
        name="one-line",
        groups={
            "line": {"en": "One straight line can split it", "nl": "Eén rechte lijn kan het splitsen"},
            "no-line": {"en": "No straight line can split it", "nl": "Geen rechte lijn kan het splitsen"},
        },
        items={
            "and": {"en": "**AND.** 1 only when both inputs are 1.", "nl": "**AND.** 1 alleen als beide inputs 1 zijn."},
            "or": {"en": "**OR.** 1 when at least one input is 1.", "nl": "**OR.** 1 als minstens één input 1 is."},
            "xor": {"en": "**XOR.** 1 when exactly one input is 1.", "nl": "**XOR.** 1 als precies één input 1 is."},
            "nand": {"en": "**NAND.** 1 unless both inputs are 1.", "nl": "**NAND.** 1, behalve als beide inputs 1 zijn."},
            "first": {"en": "**The first input.** 1 when the first input is 1, whatever the second.",
                      "nl": "**De eerste input.** 1 als de eerste input 1 is, wat de tweede ook is."},
            "same": {"en": "**Same.** 1 when both inputs are the same: both 0 or both 1.",
                     "nl": "**Gelijk.** 1 als beide inputs gelijk zijn: allebei 0 of allebei 1."},
        },
        solution={"and": "line", "or": "line", "xor": "no-line", "nand": "line", "first": "line", "same": "no-line"},
        nudge={
            "and": {"en": "Only (1, 1) is a 1. Can a line cut off that one corner?", "nl": "Alleen (1, 1) is een 1. Kan een lijn die ene hoek afsnijden?"},
            "or": {"en": "Only (0, 0) is a 0. Can a line cut off that one corner?", "nl": "Alleen (0, 0) is een 0. Kan een lijn die ene hoek afsnijden?"},
            "xor": {"en": "The 1s sit on one diagonal, the 0s on the other. Try it in the field above.",
                    "nl": "De enen liggen op de ene diagonaal, de nullen op de andere. Probeer het in het veld hierboven."},
            "nand": {"en": "Only (1, 1) is a 0. Can a line cut off that one corner?", "nl": "Alleen (1, 1) is een 0. Kan een lijn die ene hoek afsnijden?"},
            "first": {"en": "The 1s are the right side of the square. Is there a line between left and right?",
                      "nl": "De enen zijn de rechterkant van het vierkant. Past er een lijn tussen links en rechts?"},
            "same": {"en": "The 1s sit on one diagonal, the 0s on the other: is that different from XOR?",
                     "nl": "De enen liggen op de ene diagonaal, de nullen op de andere: is dat anders dan bij XOR?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the tasks again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de taken opnieuw uit."},
            "empty": {"en": "Put each task in a group: one straight line can split its 1s from its 0s, or no straight line can.",
                      "nl": "Zet elke taak in een groep: één rechte lijn kan de enen van de nullen scheiden, of geen rechte lijn kan dat."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs in the other group.", "nl": "{item} hoort in de andere groep."},
            "correct": {"en": "All {n} are right. When the 1s are one corner, three corners or one side, a line can split them; when they sit on a diagonal, as with XOR and its opposite, no line can.",
                        "nl": "Alle {n} kloppen. Zijn de enen één hoek, drie hoeken of één kant, dan kan een lijn ze scheiden; liggen ze op een diagonaal, zoals bij XOR en zijn tegenhanger, dan kan geen lijn dat."},
        },
    ),
    "perceptron-step": FunctionBox(
        name="perceptron-step",
        function="perceptron_step",
        start=_P_START,
        solution=_P,
        visible=[([1, 1], 1, [0.0, 0.0], 0.0, 0.5)],
        hidden=[([1, 0], 0, [1, 0.5], 0, 0.5), ([0, 1], 0, [1, 0.5], -1, 0.5), ([1, 1], 0, [1.0, 1.0], -0.5, 0.25),
                ([0.5, 2.0, 1.0], 1, [0.2, -0.3, 0.1], -0.5, 0.1), ([2, 1], 1, [0.5, 0.5], 0.0, 1.0)],
        compare=_close_pair,
        show=_plain,
        first=[_DOTS],
        mistakes=[
            Mistake("one-value", matches=lambda got, text: not isinstance(got, (tuple, list)),
                    en="`{call}` gives `{got}`: one thing, and the check expects two, the new weights and the new bias term. Give both back: `return w, b`.",
                    nl="`{call}` geeft `{got}`: één ding, en de check verwacht er twee, de nieuwe weights en de nieuwe bias term. Geef ze allebei terug: `return w, b`."),
            Mistake("sign", wrong=_p("    w = w + lr * (pred - y) * x\n    b = b + lr * (pred - y)\n    return w, b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your step goes the wrong way, away from the label. Use `y - pred`: when the neuron should have fired and did not, that is 1, and the weights grow.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je stap gaat de verkeerde kant op, van het label af. Gebruik `y - pred`: als het neuron had moeten vuren en dat niet deed, is dat 1, en groeien de weights."),
            Mistake("no-lr", wrong=_p("    w = w + (y - pred) * x\n    b = b + (y - pred)\n    return w, b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: without the learning rate, every step is the whole input. Multiply by `lr`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: zonder learning rate is elke stap de hele input. Vermenigvuldig met `lr`."),
            Mistake("only-w", wrong=_p("    w = w + lr * (y - pred) * x\n    return w, b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `b` stays the same. Update the bias term too: `b = b + lr * (y - pred)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `b` blijft hetzelfde. Werk ook de bias term bij: `b = b + lr * (y - pred)`."),
            Mistake("no-x", wrong=_p("    w = w + lr * (y - pred)\n    b = b + lr * (y - pred)\n    return w, b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every weight moves by the same amount. Each weight moves by `lr * (y - pred)` times its own input: multiply by `x`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elke weight schuift even ver op. Elke weight schuift met `lr * (y - pred)` keer zijn eigen input: vermenigvuldig met `x`."),
            Mistake("use-z", wrong=_p("    w = w + lr * (y - z) * x\n    b = b + lr * (y - z)\n    return w, b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you move by `y - z`, how far `z` lies from the label. The perceptron looks only at its prediction: it moves by `y - pred`, which is 0 when the prediction is right.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je schuift met `y - z`, hoe ver `z` van het label ligt. Het perceptron kijkt alleen naar zijn voorspelling: het schuift met `y - pred`, en dat is 0 als de voorspelling klopt."),
            Mistake("always", wrong=_p("    w = w + lr * y * x\n    b = b + lr * y\n    return w, b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you move by the label `y` alone. The step is `lr * (y - pred)`: 0 when the prediction is right, and below 0 when the neuron fired but should not have.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je schuift alleen met het label `y`. De stap is `lr * (y - pred)`: 0 als de voorspelling klopt, en onder 0 als het neuron vuurde maar dat niet had moeten doen."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with step 2 of the recipe: move `w` and `b` by `lr` times `y - pred`, the weights also times `x`.",
                      "nl": "Vervang de `...` door stap 2 van het recept: verschuif `w` en `b` met `lr` keer `y - pred`, de weights ook keer `x`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with five values, and the `def` line of your function has a different number of names between its parentheses. Keep `(x, y, w, b, lr)`.",
                          "nl": "De check roept `{call}` aan, met vijf waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x, y, w, b, lr)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you add `lr * (y - pred) * x` to `w`, and `lr * (y - pred)` to `b`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Tel je `lr * (y - pred) * x` op bij `w`, en `lr * (y - pred)` bij `b`?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Is your step 0 when the prediction is right, and does each weight move by its own input times `lr * (y - pred)`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Is je stap 0 als de voorspelling klopt, en schuift elke weight met zijn eigen input keer `lr * (y - pred)`?"},
            "correct": {"en": "Your function is right in all {n} cases. In the example the neuron should have fired and did not, so the weights and the bias term each grow by 0.5.",
                        "nl": "Je functie klopt in alle {n} gevallen. In het voorbeeld had het neuron moeten vuren en deed het dat niet, dus groeien de weights en de bias term elk met 0,5."},
        },
    ),
}

_SOLVED = {"and": "line", "or": "line", "xor": "no-line", "nand": "line", "first": "line", "same": "no-line"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("one-line", {}, "empty", None),
    ("one-line", _SOLVED, "correct", None),
    ("one-line", {"and": "line", "or": "line"}, "wrong", "unfinished"),
    ("one-line", {**_SOLVED, "xor": "line"}, "wrong", "wrong:xor"),
    ("one-line", {**_SOLVED, "same": "line"}, "wrong", "wrong:same"),
    ("one-line", {**_SOLVED, "nand": "no-line"}, "wrong", "wrong:nand"),
    ("perceptron-step", _P_START, "empty", None),
    ("perceptron-step", _P, "correct", None),
    ("perceptron-step", _p("    w = w + lr * (y - pred) * x\n    b += lr * (y - pred)\n    return w, b"), "correct", None),
    ("perceptron-step", _p("    for i in range(len(w)):\n        w[i] = w[i] + lr * (y - pred) * x[i]\n    b = b + lr * (y - pred)\n    return w, b"), "correct", None),
    ("perceptron-step", _p("    w += lr * (y - pred) * x\n    b += lr * (y - pred)\n    return list(w), b"), "correct", None),
    ("perceptron-step", _p("    if pred != y:\n        w = w + lr * (y - pred) * x\n        b = b + lr * (y - pred)\n    return w, b"), "correct", None),
    ("perceptron-step", _p("    w = w + lr * (y - pred) * x\n    b = b + lr * (y - pred)\n    return w"), "wrong", "one-value"),
    ("perceptron-step", _p("    w = w + lr * (pred - y) * x\n    b = b + lr * (pred - y)\n    return w, b"), "wrong", "sign"),
    ("perceptron-step", _p("    w = w + (y - pred) * x\n    b = b + (y - pred)\n    return w, b"), "wrong", "no-lr"),
    ("perceptron-step", _p("    w = w + lr * (y - pred) * x\n    return w, b"), "wrong", "only-w"),
    ("perceptron-step", _p("    w = w + lr * y * x\n    b = b + lr * y\n    return w, b"), "wrong", "always"),
    ("perceptron-step", _p("    w = w + lr * (y - pred)\n    b = b + lr * (y - pred)\n    return w, b"), "wrong", "no-x"),
    ("perceptron-step", _p("    w = w + lr * (y - z) * x\n    b = b + lr * (y - z)\n    return w, b"), "wrong", "use-z"),
]
