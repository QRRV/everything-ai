# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-hidden-layer: a forward pass through a hidden layer with ReLU and a linear output
neuron, and choosing the output weights so that the network computes XOR. And the picture: where a network of three
step neurons fires and which corners it gets wrong, as the bias term of its second hidden neuron moves on a slider."""
import re

from everything_ai.check import FunctionBox, CodeBox, Mistake

CORNERS = [(0, 0), (0, 1), (1, 0), (1, 1)]
XOR = [0, 1, 1, 0]


def _step(z: float) -> int:
    return 1 if z > 0 else 0


def network(x1: float, x2: float, b2: float) -> int:
    """The XOR network of the lesson: h₁ = step(x₁ + x₂ − 0.5), h₂ = step(x₁ + x₂ + b₂), ŷ = step(h₁ − h₂ − 0.5)."""
    h1, h2 = _step(x1 + x2 - 0.5), _step(x1 + x2 + b2)
    return _step(h1 - h2 - 0.5)


def picture(b2: float, labels: dict[str, str]) -> str:
    """The plane of the two inputs: small blue dots where the network fires, the two lines of its hidden neurons
    dashed, the four corners of XOR, filled for 1 and open for 0, and a purple ring around each corner the network
    gets wrong, as in nn-xor; beside it, how many corners it gets wrong for every b₂ on the slider. Drawn as SVG text,
    once for every stop of the slider (L.explore); dragging across the second chart sets b₂."""
    from everything_ai import chart
    lo, hi = -0.25, 1.25
    p = chart.Panel((lo, hi), (lo, hi), title=labels["title"], xlabel="x₁", ylabel="x₂", xticks=[0, 1])
    grid = [lo + 0.05 * k for k in range(31)]
    on = [(a, c) for a in grid for c in grid if network(a, c, b2)]
    p.points([a for a, _ in on], [c for _, c in on], "pred", 2.2)
    p.line([lo, hi], [0.5 - lo, 0.5 - hi], "ink-3", 1.6, dashed=True)     # h₁: x₁ + x₂ = 0.5
    p.line([lo, hi], [-b2 - lo, -b2 - hi], "ink-3", 1.6, dashed=True)     # h₂: x₁ + x₂ = −b₂
    ringed = wrong_corners(b2)   # over the blue dots, under the corners, so each corner stays visible inside its ring
    p.points([a for a, _ in ringed], [c for _, c in ringed], "loss", 13, hollow=True)
    ones = [c for c, y in zip(CORNERS, XOR) if y]
    zeros = [c for c, y in zip(CORNERS, XOR) if not y]
    p.points([a for a, _ in ones], [c for _, c in ones], "data", 7)
    p.points([a for a, _ in zeros], [c for _, c in zeros], "data", 7, hollow=True)
    # beside it, for every b₂ on the slider: how many corners the network gets wrong
    b2s = [round(-2.5 + 0.1 * i, 1) for i in range(26)]
    k = chart.Panel((-2.6, 0.1), (-0.3, 4.3), title=labels["wrong_title"], xlabel="b₂", ylabel=labels["wrong"], yticks=[0, 1, 2, 3, 4])
    k.line([b2, b2], [-0.3, 4.3], "ink-3", 1.2)
    k.points(b2s, [wrong(v) for v in b2s], "loss", 3)   # dots, not a line: the count jumps between two stops
    k.points([b2], [wrong(b2)], "loss", 6)
    return chart.figure([p, k], alt=labels["alt"], width=330, height=320, drag=1)   # two fit side by side in a play block


def outputs(b2: float) -> list[int]:
    return [network(a, c, b2) for a, c in CORNERS]


def wrong_corners(b2: float) -> list[tuple[int, int]]:
    """The corners the network gets wrong."""
    return [c for c, o, y in zip(CORNERS, outputs(b2), XOR) if o != y]


def wrong(b2: float) -> int:
    """How many of the four corners the network gets wrong."""
    return len(wrong_corners(b2))


# The lesson is done when these are correct. "collapse" is the choice question in the lesson.
REQUIRED = ["collapse", "forward", "output-weights"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_F_HEAD = "import numpy as np\n\n\ndef forward(x, W, b, v, c):\n    h = []\n    for i in range(len(W)):\n"
_F_START = _F_HEAD + "        ...\n    return np.dot(v, h) + c\n"
_F_BODY = "        z = np.dot(W[i], x) + b[i]\n        h.append(max(0, z))\n    return np.dot(v, h) + c"
_F = _F_HEAD + _F_BODY


def _f(body: str) -> str:
    return _F_HEAD + body + "\n    return np.dot(v, h) + c"



_O_HEAD = "import numpy as np\n\n\ndef relu(z):\n    return max(0, z)\n\n\n"
_O_LOOP = ("outputs = []\nfor x in [[0, 0], [0, 1], [1, 0], [1, 1]]:\n    h = [relu(x[0] + x[1]), relu(x[0] + x[1] - 1)]\n"
           "    outputs.append(np.dot(v, h) + c)\noutputs")
_O_START = _O_HEAD + "v = ...\nc = ...\n\n" + _O_LOOP + "\n"
_O = _O_HEAD + "v = [1, -2]\nc = 0\n\n" + _O_LOOP


def _o(v: str, c: str) -> str:
    return _O_HEAD + f"v = {v}\nc = {c}\n\n" + _O_LOOP


def _first_not_zero(got) -> bool:
    """The output for (0, 0) is not 0: there it is just `c`."""
    try:
        return len(got) == 4 and abs(float(got[0])) > 1e-9
    except (TypeError, ValueError):
        return False


def _three_right(got) -> bool:
    """Right for (0, 0), (0, 1) and (1, 0), but not 0 for (1, 1). Not a list of four numbers, such as the lists
    of arrays a single number `v` gives: no."""
    try:
        return (isinstance(got, list) and len(got) == 4 and abs(float(got[1]) - 1) < 1e-9
                and abs(float(got[2]) - 1) < 1e-9 and abs(float(got[3])) > 1e-9)
    except (TypeError, ValueError):
        return False


EXERCISES = {
    "forward": FunctionBox(
        name="forward",
        function="forward",
        start=_F_START,
        solution=_F,
        visible=[([1, 2], [[1, -1], [0.5, 0.5]], [0, -1], [2, 1], 0.5)],
        hidden=[([2, 1], [[1, -1], [0.5, 0.5]], [0, -1], [2, 1], 0.5), ([0, 0], [[1, 1], [1, 1]], [0.5, -1], [1, -1], 0), ([2.0, -1.0], [[0.5, 1.0], [-1.0, 2.0], [1.0, 1.0]], [0.5, 1.0, -0.5], [1.0, -1.0, 2.0], 0.3),
                ([1.0, 2.0, 3.0], [[1.0, 0.0, -1.0]], [0.5], [3.0], -1.0), ([0.5, 0.5], [[2.0, -2.0], [-1.0, -1.0]], [0.0, 0.5], [1.0, 1.0], 0.0)],
        first=[_DOTS],
        mistakes=[
            Mistake("no-activation", wrong=_f("        z = np.dot(W[i], x) + b[i]\n        h.append(z)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your hidden neurons pass on `z` itself, without an activation function. Put each `z` through ReLU: `max(0, z)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je hidden neurons geven `z` zelf door, zonder activation function. Haal elke `z` door ReLU: `max(0, z)`."),
            Mistake("step", wrong=_f("        z = np.dot(W[i], x) + b[i]\n        if z > 0:\n            h.append(1)\n        else:\n            h.append(0)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your hidden neurons use a step, so they give 1 instead of `z` when `z` is above 0. ReLU passes `z` on as it is: `max(0, z)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je hidden neurons gebruiken een stap, dus ze geven 1 in plaats van `z` als `z` boven 0 ligt. ReLU geeft `z` door zoals hij is: `max(0, z)`."),
            Mistake("min", wrong=_f("        z = np.dot(W[i], x) + b[i]\n        h.append(min(0, z))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `min(0, z)` keeps the negative part and drops the positive one. ReLU is the other way round: `max(0, z)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `min(0, z)` houdt het negatieve deel en laat het positieve vallen. ReLU is andersom: `max(0, z)`."),
            Mistake("no-bias", wrong=_f("        z = np.dot(W[i], x)\n        h.append(max(0, z))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you leave out the bias terms of the hidden neurons. Hidden neuron `i` adds `b[i]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je laat de bias terms van de hidden neurons weg. Hidden neuron `i` telt `b[i]` erbij op."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with hidden neuron `i`: its `z` from row `i` of `W` and `b[i]`, and its output through ReLU added to `h`.",
                      "nl": "Vervang de `...` door hidden neuron `i`: zijn `z` uit rij `i` van `W` en `b[i]`, en zijn output door ReLU toegevoegd aan `h`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with five values, and the `def` line of your function has a different number of names between its parentheses. Keep `(x, W, b, v, c)`.",
                          "nl": "De check roept `{call}` aan, met vijf waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x, W, b, v, c)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does hidden neuron `i` work out `np.dot(W[i], x) + b[i]`, and add `max(0, z)` to `h`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Rekent hidden neuron `i` `np.dot(W[i], x) + b[i]` uit, en voegt het `max(0, z)` toe aan `h`?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Do you work it out from `x`, `W`, `b`, `v` and `c` themselves, for any number of inputs and hidden neurons?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken je het uit met `x`, `W`, `b`, `v` en `c` zelf, voor elk aantal inputs en hidden neurons?"},
            "correct": {"en": "Your function is right in all {n} cases. In the example the first hidden neuron has `z` = −1 and gives 0; the second gives 0.5, and the output is 2 · 0 + 1 · 0.5 + 0.5 = 1.",
                        "nl": "Je functie klopt in alle {n} gevallen. In het voorbeeld heeft het eerste hidden neuron `z` = −1 en geeft het 0; het tweede geeft 0,5, en de output is 2 · 0 + 1 · 0,5 + 0,5 = 1."},
        },
    ),
    "output-weights": CodeBox(
        name="output-weights",
        start=_O_START,
        solution=_O,
        keep=["return max(0, z)", "for x in [[0, 0], [0, 1], [1, 0], [1, 1]]:", "h = [relu(x[0] + x[1]), relu(x[0] + x[1] - 1)]", "outputs.append(np.dot(v, h) + c)"],
        first=[
            _DOTS,
            Mistake("typed", matches=lambda got, text: re.search(r"^[^#\n]*outputs\s*=\s*\[\s*\d", text, re.MULTILINE) is not None,
                    en="You type the outputs in yourself. Let the network work them out: choose only `v` and `c`, and leave the loop as it is.",
                    nl="Je typt de uitkomsten zelf in. Laat het netwerk ze uitrekenen: kies alleen `v` en `c`, en laat de loop zoals hij is."),
        ],
        mistakes=[
            Mistake("origin", matches=lambda got, text: _first_not_zero(got),
                    en="Your last line gives `{got}`: for (0, 0) both hidden neurons give 0, so the output is just `c`. XOR gives 0 there: `c` should be 0.",
                    nl="Je laatste regel geeft `{got}`: bij (0, 0) geven beide hidden neurons 0, dus de output is gewoon `c`. XOR geeft daar 0: `c` moet 0 zijn."),
            Mistake("one-one", matches=lambda got, text: _three_right(got),
                    en="Your last line gives `{got}`: right for (0, 0), (0, 1) and (1, 0), but not 0 for (1, 1). There the hidden neurons give 2 and 1: the second weight has to take away the 2 the first one adds.",
                    nl="Je laatste regel geeft `{got}`: goed bij (0, 0), (0, 1) en (1, 0), maar geen 0 bij (1, 1). Daar geven de hidden neurons 2 en 1: de tweede weight moet de 2 weghalen die de eerste erbij doet."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...` with the output weights `v`, two numbers, and the output bias term `c`, so that the outputs are 0, 1, 1 and 0.",
                      "nl": "Vervang de twee `...` door de output-weights `v`, twee getallen, en de bias term `c` van de output, zodat de outputs 0, 1, 1 en 0 zijn."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. Choose only `v` and `c`.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. Kies alleen `v` en `c`."},
            "no_value": {"en": "The last line of the box shows nothing. Put `outputs` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `outputs` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be `[0, 1, 1, 0]`. Work out `h` for each corner: (0, 0) gives [0, 0], (0, 1) and (1, 0) give [1, 0], and (1, 1) gives [2, 1]. Which `v` and `c` turn those into 0, 1, 1 and 0?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet `[0, 1, 1, 0]` zijn. Reken `h` uit bij elke hoek: (0, 0) geeft [0, 0], (0, 1) en (1, 0) geven [1, 0], en (1, 1) geeft [2, 1]. Welke `v` en `c` maken daar 0, 1, 1 en 0 van?"},
            "correct": {"en": "XOR: the first hidden neuron counts the ones, the second only starts above one 1, and the output neuron takes it away twice. One neuron on its own cannot do this.",
                        "nl": "XOR: het eerste hidden neuron telt de enen, het tweede begint pas boven één 1, en het output-neuron haalt dat twee keer weg. Eén neuron alleen kan dit niet."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("forward", _F_START, "empty", None),
    ("forward", _F, "correct", None),
    ("forward", _f("        h.append(max(0, np.dot(W[i], x) + b[i]))"), "correct", None),
    ("forward", _f("        z = np.dot(W[i], x) + b[i]\n        if z > 0:\n            h.append(z)\n        else:\n            h.append(0)"), "correct", None),
    ("forward", _f("        z = np.dot(W[i], x) + b[i]\n        h.append(z)"), "wrong", "no-activation"),
    ("forward", _f("        z = np.dot(W[i], x) + b[i]\n        if z > 0:\n            h.append(1)\n        else:\n            h.append(0)"), "wrong", "step"),
    ("forward", _f("        z = np.dot(W[i], x) + b[i]\n        h.append(min(0, z))"), "wrong", "min"),
    ("forward", _f("        z = np.dot(W[i], x)\n        h.append(max(0, z))"), "wrong", "no-bias"),
    ("output-weights", _O_START, "empty", None),
    ("output-weights", _O, "correct", None),
    ("output-weights", _o("[1, -2]", "0.0"), "correct", None),
    ("output-weights", _o("[1, -2]", "1"), "wrong", "origin"),
    ("output-weights", _o("[1, -1]", "0"), "wrong", "one-one"),
    ("output-weights", _o("[1, 0]", "0"), "wrong", "one-one"),
    ("output-weights", _o("[2, -2]", "0"), "wrong", None),
    ("output-weights", _o("1", "0"), "wrong", None),
    ("output-weights", _o("np.array([1.0, -2.0])", "0.0"), "correct", None),
    ("output-weights", _O.replace("v = [1, -2]\nc = 0\n", "v = [1, -2]\nc = 0\noutputs = [0, 1, 1, 0]\n"), "wrong", "typed"),
]
