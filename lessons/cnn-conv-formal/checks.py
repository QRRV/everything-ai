# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cnn-conv-formal: out_size, the size of the output of a convolution from the size of the input,
the kernel size, the stride and the padding; and conv2d, a convolution with stride and padding in NumPy."""
import re

import numpy as np

from everything_ai import draw
from everything_ai.check import FunctionBox, Mistake


def conv_picture(labels: dict):
    """A 5 by 5 image with a frame of 1 zero, a 3 by 3 filter and stride 2: an output of 3 by 3."""
    return draw.conv_grids(5, 3, 2, 1, labels=labels)


# The lesson is done when these are correct. "receptive" is the choice question in the lesson.
REQUIRED = ["receptive", "out-size", "conv2d"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _grid(v):
    """An array or nested list for the messages: whole numbers without .0, as a list of lists."""
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return repr(v)
    if a.ndim == 0:
        return repr(int(a)) if a == round(float(a)) else repr(round(float(a), 4))
    return repr(a.astype(int).tolist() if np.all(a == np.round(a)) else np.round(a, 4).tolist())


def _same_grid(got, want) -> bool:
    try:
        g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    except (TypeError, ValueError):
        return False
    return g.shape == w.shape and bool(np.allclose(g, w))


def _same_int(got, want) -> bool:
    return isinstance(got, (int, np.integer)) and not isinstance(got, bool) and got == want


# --- exercise 2: out_size -------------------------------------------------------------------------------------------------

_O_START = "def out_size(n, k, s, p):\n    return ...\n"


def _o(expr: str = "(n + 2 * p - k) // s + 1") -> str:
    return f"def out_size(n, k, s, p):\n    return {expr}\n"


_O = _o()

# --- exercise 3: conv2d ---------------------------------------------------------------------------------------------------

_C_HEAD = ("import numpy as np\n\n\ndef conv2d(img, K, s, p):\n    img = np.array(img)\n    K = np.array(K)\n"
           "    n = len(img)\n    k = len(K)\n")
_PAD = "    x = np.zeros((n + 2 * p, n + 2 * p))\n    x[p:p + n, p:p + n] = img\n"
_SIZE = "    m = (n + 2 * p - k) // s + 1\n"
_LOOP = "    out = np.zeros((m, m))\n    for r in range(m):\n        for c in range(m):\n"
_STEP = "            out[r, c] = (x[r * s:r * s + k, c * s:c * s + k] * K).sum()\n"
_C_TAIL = "    return out\n"
_C_START = (_C_HEAD + "    # 1. the padded image: a frame of p zeros around img\n    x = ...\n    x[p:p + n, p:p + n] = img\n"
            "    # 2. the output size\n    m = ...\n" + _LOOP + "            # 3. the filter at the place that starts at row r * s and column c * s\n"
            "            out[r, c] = ...\n" + _C_TAIL)


def _c(pad: str = _PAD, size: str = _SIZE, step: str = _STEP) -> str:
    return _C_HEAD + pad + size + _LOOP + step + _C_TAIL


_C = _c()

_IMG = [[1, 2, 0, 1], [0, 1, 3, 1], [2, 1, 0, 0], [1, 0, 1, 2]]
_IMG5 = [[1, 0, 2, 1, 0], [0, 1, 1, 0, 2], [3, 0, 1, 2, 1], [1, 2, 0, 0, 1], [0, 1, 2, 1, 0]]

EXERCISES = {
    "out-size": FunctionBox(
        name="out-size",
        function="out_size",
        start=_O_START,
        solution=_O,
        visible=[(28, 5, 2, 0)],
        hidden=[(28, 3, 1, 1), (32, 3, 2, 1), (7, 7, 1, 0), (10, 4, 3, 2), (227, 11, 4, 0)],
        compare=_same_int,
        python_values=True,   # a whole float shows as `12.0`, not as `12`
        first=[_DOTS],
        mistakes=[
            Mistake("float", matches=lambda got, text: isinstance(got, float),
                    en="`{call}` gives `{got}`, and it should give `{want}`: a number of places is a whole number, an int, not a float. `/` always gives a float, even when it divides evenly; `//` rounds down and gives an int, and a place where the filter does not fit any more does not count.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: een aantal plekken is een heel getal, een int, geen float. `/` geeft altijd een float, ook als de deling precies uitkomt; `//` rondt af naar beneden en geeft een int, en een plek waar het filter niet meer past, telt niet."),
            Mistake("no-plus-one", wrong=_o("(n + 2 * p - k) // s"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the first place is missing. `(n + 2 * p - k) // s` counts the steps after the first place; add 1 for the first place itself.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de eerste plek ontbreekt. `(n + 2 * p - k) // s` telt de stappen na de eerste plek; tel er 1 bij op voor de eerste plek zelf."),
            Mistake("one-side", wrong=_o("(n + p - k) // s + 1"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the padding goes on both sides, so the padded image is `n + 2 * p` wide.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de padding komt aan beide kanten, dus de afbeelding met padding is `n + 2 * p` breed."),
            Mistake("stride-last", wrong=_o("(n + 2 * p - k + 1) // s"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you divide the number of places of stride 1 by `s`. Count the steps of `s` after the first place, `(n + 2 * p - k) // s`, and add the first place.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je deelt het aantal plekken bij stride 1 door `s`. Tel de stappen van `s` na de eerste plek, `(n + 2 * p - k) // s`, en tel de eerste plek erbij."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the formula for the size of the output.",
                      "nl": "Vervang de `...` door de formule voor de grootte van de output."},
            "arguments": {"en": "The check calls `{call}`, with four numbers, and the `def` line of your function has another number of names between its parentheses. Keep `(n, k, s, p)`.",
                          "nl": "De check roept `{call}` aan, met vier getallen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(n, k, s, p)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your line with the formula in The idea.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je regel met de formule in Het idee."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your formula use all four numbers?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je formule alle vier de getallen?"},
            "correct": {"en": "Your formula gives every output size, also for the first layer of AlexNet: 227 pixels, an 11 by 11 filter and stride 4 give 55 by 55.",
                        "nl": "Je formule geeft elke outputgrootte, ook voor de eerste layer van AlexNet: 227 pixels, een filter van 11 bij 11 en stride 4 geven 55 bij 55."},
        },
    ),
    "conv2d": FunctionBox(
        name="conv2d",
        function="conv2d",
        start=_C_START,
        solution=_C,
        visible=[(_IMG, [[1, 0], [0, -1]], 2, 0)],
        hidden=[(_IMG, [[1, 1, 1], [1, 1, 1], [1, 1, 1]], 1, 1), (_IMG5, [[1, 0, -1], [1, 0, -1], [1, 0, -1]], 2, 1),
                (_IMG5, [[2]], 1, 0), (_IMG, [[1, 2], [3, 4]], 1, 0), (_IMG5, [[0, 1], [1, 0]], 3, 2)],
        compare=_same_grid,
        show=_grid,
        first=[_DOTS],
        mistakes=[
            Mistake("no-stride", wrong=_c(step="            out[r, c] = (x[r:r + k, c:c + k] * K).sum()\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your filter moves one place at a time, whatever the stride. Output place `r` starts at row `r * s` of the padded image.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je filter schuift één plek tegelijk op, wat de stride ook is. Outputplek `r` begint bij rij `r * s` van het opgevulde plaatje."),
            Mistake("size-float", error=lambda e: isinstance(e, TypeError) and "integer" in str(e),
                    en="With `{call}`, your output size is a fraction, and `np.zeros` and `range` need a whole number. Use `//`, as in exercise 2.",
                    nl="Met `{call}` is je outputgrootte een breuk, en `np.zeros` en `range` hebben een heel getal nodig. Gebruik `//`, zoals in oefening 2."),
            Mistake("no-pad", syntax=r"^[ \t]*out\[.*=.*\bimg\[",
                    en="Your filter slides over `img`, not over the padded image `x`: with padding, the pieces at the edge come out too small, and Python stops. Take every piece from `x`.",
                    nl="Je filter schuift over `img`, niet over de afbeelding met padding `x`: met padding komen de stukken aan de rand te klein uit, en Python stopt. Neem elk stuk uit `x`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...`: the padded image of zeros, the output size, and the sum of the piece times the filter at every place.",
                      "nl": "Vervang de drie `...`: de afbeelding met een rand van nullen, de outputgrootte, en de som van het stuk keer het filter op elke plek."},
            "arguments": {"en": "The check calls `{call}`, with an image, a filter and two numbers, and the `def` line of your function has another number of names between its parentheses. Keep `(img, K, s, p)`.",
                          "nl": "De check roept `{call}` aan, met een plaatje, een filter en twee getallen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(img, K, s, p)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your three lines with steps 1 to 3 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je drie regels met stap 1 tot en met 3 van het recept."},
            "almost": {"en": "Right for the example, with stride 2 and no padding. But `{call}` gives `{got}`, and it should give `{want}`. Do your lines use `s` and `p` themselves, not fixed numbers, and is the frame made of zeros?",
                       "nl": "Goed voor het voorbeeld, met stride 2 en zonder padding. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruiken je regels `s` en `p` zelf, geen vaste getallen, en bestaat de rand uit nullen?"},
            "correct": {"en": "Your convolution works for every kernel size, stride and padding: with padding 1, a 3 by 3 filter keeps a 4 by 4 image 4 by 4, and stride 2 halves it.",
                        "nl": "Je convolutie werkt voor elke kernel size, stride en padding: met padding 1 houdt een filter van 3 bij 3 een afbeelding van 4 bij 4 op 4 bij 4, en stride 2 halveert het."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("out-size", _O_START, "empty", None),
    ("out-size", _o("int((n + 2 * p - k) / s) + 1"), "correct", None),
    ("out-size", _o("(n + 2 * p - k) / s + 1"), "wrong", "float"),
    ("out-size", _o("(n + 2 * p - k) // s + 1.0"), "wrong", "float"),
    ("out-size", _o("(n + 2 * p - k) // s"), "wrong", "no-plus-one"),
    ("out-size", _o("(n + p - k) // s + 1"), "wrong", "one-side"),
    ("out-size", _o("(n + 2 * p - k + 1) // s"), "wrong", "stride-last"),
    ("out-size", _o("(n - k) // s + 1"), "almost", None),
    ("conv2d", _C_START, "empty", None),
    ("conv2d", _c(size="    m = len(range(0, n + 2 * p - k + 1, s))\n"), "correct", None),
    ("conv2d", _c(step="            out[r, c] = (x[r:r + k, c:c + k] * K).sum()\n"), "wrong", "no-stride"),
    ("conv2d", _c(step="            out[r, c] = (img[r * s:r * s + k, c * s:c * s + k] * K).sum()\n"), "error", "no-pad"),
    ("conv2d", _c(size="    m = (n + 2 * p - k) / s + 1\n"), "error", "size-float"),
    ("conv2d", _c(size="    m = (n + 2 * p - k) // 2 + 1\n"), "almost", None),
    # padding with a loop that reads img[r, c] is right; with another error in it, the message is not no-pad
    ("conv2d", _c(pad="    x = np.zeros((n + 2 * p, n + 2 * p))\n    for r in range(n):\n        for c in range(n):\n            x[r + p, c + p] = img[r, c]\n"), "correct", None),
    ("conv2d", _c(pad="    x = np.zeros((n + 2 * p, n + 2 * p))\n    for r in range(n):\n        for c in range(n):\n            x[r + p, c + p] = img[r, c]\n",
                  size="    m = (n + 2 * p - k) / s + 1\n"), "error", "size-float"),
]
