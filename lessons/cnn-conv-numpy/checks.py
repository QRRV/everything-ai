# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cnn-conv-numpy: patches, im2col, every patch of an input as one row of a matrix; and
conv_matmul, the whole convolution as one matrix multiplication of the filters with those rows."""
import re

import numpy as np

from everything_ai import draw
from everything_ai.check import FunctionBox, Mistake

PICTURE_IN = [[2, 0, 1], [1, 3, 0], [0, 1, 2]]


def patches_picture(labels: dict):
    """One channel of 3 by 3 and its patch matrix for a filter of 2 by 2: the patch at row 0, column 1 and the row
    it becomes are marked."""
    a = np.array(PICTURE_IN)
    rows = []
    for r in range(2):
        for c in range(2):
            rows.append([str(int(v)) for v in a[r:r + 2, c:c + 2].reshape(4)])
    as_text = [[str(v) for v in row] for row in PICTURE_IN]
    patch = {(0, 1), (0, 2), (1, 1), (1, 2)}
    row_1 = {(1, j) for j in range(4)}
    return draw.array_grids([(labels["input"], as_text, ["(3, 3)", labels["input_note"]], patch),
                             (labels["output"], rows, ["(4, 4)", labels["output_note"]], row_1)],
                            indices=False, title=labels["title"])


# The lesson is done when these are correct. "shape" is the choice question in the lesson.
REQUIRED = ["shape", "patches", "conv-matmul"]

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
    """An array for the messages: whole numbers without .0, as nested lists."""
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return repr(v)
    return repr(a.astype(int).tolist() if np.all(a == np.round(a)) else np.round(a, 4).tolist())


def _same_grid(got, want) -> bool:
    try:
        g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    except (TypeError, ValueError):
        return False
    return g.shape == w.shape and bool(np.allclose(g, w))


def _reshape_error(e) -> bool:
    return isinstance(e, ValueError) and "cannot reshape array" in str(e)


# --- exercise 2: patches --------------------------------------------------------------------------------------------------

_P_HEAD = ("import numpy as np\n\n\ndef patches(x, k):\n    x = np.array(x)\n    c_in = len(x)\n    m = len(x[0]) - k + 1\n"
           "    rows = []\n    for r in range(m):\n        for c in range(m):\n")
_P_STEP = "            rows.append(x[:, r:r + k, c:c + k].reshape(c_in * k * k))\n"
_P_START = (_P_HEAD + "            # the patch at r, c of every channel, as one flat row of c_in * k * k numbers\n"
            "            rows.append(...)\n    return np.array(rows)\n")


def _p(step: str = _P_STEP) -> str:
    return _P_HEAD + step + "    return np.array(rows)\n"


_P = _p()

# --- exercise 3: conv_matmul ----------------------------------------------------------------------------------------------

_M_HEAD = "import numpy as np\n\n\ndef conv_matmul(P, W, m):\n    P = np.array(P)\n    W = np.array(W)\n    c_out = len(W)\n"
_M_WM = "    Wm = W.reshape(c_out, len(P[0]))\n"
_M_OUT = "    return (Wm @ P.T).reshape(c_out, m, m)\n"
_M_START = (_M_HEAD + "    # every filter as one row, as long as a row of P\n    Wm = ...\n"
            "    # one row per filter and one column per place; then an m by m output channel per filter\n    return ...\n")


def _m(wm: str = _M_WM, out: str = _M_OUT) -> str:
    return _M_HEAD + wm + out


_M = _m()


def _patches(x, k):
    """The course's patches, for the hidden cases of conv_matmul."""
    x = np.array(x)
    m = len(x[0]) - k + 1
    return [x[:, r:r + k, c:c + k].reshape(-1).tolist() for r in range(m) for c in range(m)]


_X2 = [[[1, 2, 0, 1], [0, 1, 2, 2], [3, 0, 1, 0], [1, 1, 0, 2]], [[0, 1, 1, 0], [2, 0, 0, 1], [1, 1, 2, 0], [0, 2, 1, 1]]]
_W2 = [[[[1, 0], [0, 1]], [[0, 1], [1, 0]]], [[[1, 1], [0, 0]], [[0, 0], [-1, 1]]], [[[0, 0], [0, 1]], [[2, 0], [0, 0]]]]

EXERCISES = {
    "patches": FunctionBox(
        name="patches",
        function="patches",
        start=_P_START,
        solution=_P,
        visible=[([[[1, 2, 0], [0, 1, 1], [2, 0, 1]], [[0, 1, 0], [1, 0, 1], [0, 1, 0]]], 2)],
        hidden=[([[[1, 2, 0, 1], [3, 1, 2, 0], [0, 2, 1, 3], [1, 0, 3, 2]]], 3), (_X2, 2),
                ([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[0, 1], [1, 0]]], 1), ([[[4, 1, 0], [2, 2, 3], [1, 0, 5]]], 3)],
        compare=_same_grid,
        show=_grid,
        first=[_DOTS],
        mistakes=[
            Mistake("one-channel", wrong=_p("            rows.append(x[0, r:r + k, c:c + k].reshape(k * k))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your rows only hold the first channel. A patch has a piece of every channel: `x[:, r:r + k, c:c + k]`, flat with `.reshape(c_in * k * k)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je rijen bevatten alleen het eerste channel. Een patch heeft een deel van elk channel: `x[:, r:r + k, c:c + k]`, plat met `.reshape(c_in * k * k)`."),
            Mistake("swapped", wrong=_p("            rows.append(x[:, c:c + k, r:r + k].reshape(c_in * k * k))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your rows go down the columns of the image. Row 1 must be the patch one place to the right, at `r` 0 and `c` 1: in `x[:, r:r + k, c:c + k]`, `r` picks the rows of the image and `c` the columns.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je rijen gaan langs de kolommen van de afbeelding omlaag. Rij 1 moet de patch één plek naar rechts zijn, bij `r` 0 en `c` 1: in `x[:, r:r + k, c:c + k]` kiest `r` de rijen van de afbeelding en `c` de kolommen."),
            Mistake("not-flat", wrong=_p("            rows.append(x[:, r:r + k, c:c + k])\n"),
                    en="`{call}` gives an array of shape (m · m, c_in, k, k): your rows are not flat, every row is still a patch of shape (c_in, k, k). `.reshape(c_in * k * k)` makes it one row of numbers.",
                    nl="`{call}` geeft een array met shape (m · m, c_in, k, k): je rijen zijn niet plat, elke rij is nog een patch met shape (c_in, k, k). `.reshape(c_in * k * k)` maakt er één rij getallen van."),
            Mistake("reshape-size", error=_reshape_error,
                    en="What you cut out does not have as many numbers as `.reshape` asks for. A patch is `x[:, r:r + k, c:c + k]`, of shape (c_in, k, k): `c_in * k * k` numbers, and `.reshape` must ask for exactly that many.",
                    nl="Wat je uitknipt heeft niet zoveel getallen als `.reshape` vraagt. Een patch is `x[:, r:r + k, c:c + k]`, met shape (c_in, k, k): `c_in * k * k` getallen, en `.reshape` moet om precies zoveel vragen."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: the patch at `r`, `c` of every channel, as one flat row.",
                      "nl": "Vervang de `...`: de patch op `r`, `c` van elk channel, als één platte rij."},
            "arguments": {"en": "The check calls `{call}`, with an input and a kernel size, and the `def` line of your function has another number of names between its parentheses. Keep `(x, k)`.",
                          "nl": "De check roept `{call}` aan, met een input en een kernel size, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x, k)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your line with step 2 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je regel met stap 2 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any number of channels and any kernel size?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elk aantal channels en elke kernel size?"},
            "correct": {"en": "Your function is im2col: every place of the filter becomes one row. A filter as large as the image gives one row, with the whole image in it; a filter of 1 by 1 gives a row per pixel, with its channels.",
                        "nl": "Je functie is im2col: elke plek van het filter wordt één rij. Een filter zo groot als de afbeelding geeft één rij, met de hele afbeelding erin; een filter van 1 bij 1 geeft een rij per pixel, met zijn channels."},
        },
    ),
    "conv-matmul": FunctionBox(
        name="conv-matmul",
        function="conv_matmul",
        start=_M_START,
        solution=_M,
        visible=[([[1, 0, 2, 1, 0, 1, 1, 0], [0, 2, 1, 0, 1, 1, 0, 0], [2, 1, 0, 3, 1, 0, 2, 1], [1, 0, 3, 1, 0, 0, 1, 0]],
                  [[[[1, 0], [0, 1]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[1, 1], [1, 1]]]], 2)],
        hidden=[(_patches(_X2, 2), _W2, 3),
                (_patches([[[1, 2, 0, 1], [3, 1, 2, 0], [0, 2, 1, 3], [1, 0, 3, 2]]], 3), [[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]], 2),
                (_patches([[[4, 1, 0], [2, 2, 3], [1, 0, 5]], [[1, 1, 1], [0, 0, 0], [2, 2, 2]]], 3),
                 [[[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]], 1)],
        compare=_same_grid,
        show=_grid,
        first=[_DOTS],
        mistakes=[
            Mistake("rows-per-place", wrong=_m(out="    return (P @ Wm.T).reshape(c_out, m, m)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the right numbers, in the wrong places. `P @ Wm.T` has a row per place and a column per filter, and `.reshape` fills row after row, so it mixes the filters. Take `Wm @ P.T`, with a row per filter.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de goede getallen, op de verkeerde plekken. `P @ Wm.T` heeft een rij per plek en een kolom per filter, en `.reshape` vult rij na rij, dus het mengt de filters. Neem `Wm @ P.T`, met een rij per filter."),
            Mistake("no-reshape", wrong=_m(out="    return Wm @ P.T\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the right numbers, with one row per filter. Give every filter its output channel of m by m: `.reshape(c_out, m, m)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de goede getallen, met één rij per filter. Geef elk filter zijn output channel van m bij m: `.reshape(c_out, m, m)`."),
            Mistake("sizes", error=lambda e: isinstance(e, ValueError) and ("matmul" in str(e) or "not aligned" in str(e)),
                    en="The sizes of your matrix multiplication do not fit. `Wm` should have shape (c_out, c_in · k · k), a row per filter, and `P` has shape (m · m, c_in · k · k), a row per place. With `P.T`, of shape (c_in · k · k, m · m), the inner sizes are the same.",
                    nl="De maten van je matrixvermenigvuldiging passen niet. `Wm` moet shape (c_out, c_in · k · k) hebben, een rij per filter, en `P` heeft shape (m · m, c_in · k · k), een rij per plek. Met `P.T`, met shape (c_in · k · k, m · m), zijn de binnenste maten gelijk."),
            Mistake("star", error=lambda e: isinstance(e, ValueError) and "broadcast" in str(e),
                    en="Python cannot pair the numbers of your two arrays one by one: their shapes differ. `*` multiplies number by number; a matrix multiplication, a dot product of every row with every column, is `@`.",
                    nl="Python kan de getallen van je twee arrays niet één voor één aan elkaar koppelen: hun shapes verschillen. `*` vermenigvuldigt getal voor getal; een matrixvermenigvuldiging, een inproduct van elke rij met elke kolom, is `@`."),
            Mistake("no-c-in", error=lambda e: isinstance(e, NameError) and getattr(e, "name", None) in ("c_in", "k"),
                    en="`conv_matmul` only gets `P`, `W` and `m`, so `c_in` and `k` do not exist inside it. A row of `P` holds the `c_in * k * k` numbers of a patch, so a filter row is as long as a row of `P`: `len(P[0])` numbers.",
                    nl="`conv_matmul` krijgt alleen `P`, `W` en `m`, dus `c_in` en `k` bestaan er niet in. Een rij van `P` bevat de `c_in * k * k` getallen van een patch, dus een filterrij is even lang als een rij van `P`: `len(P[0])` getallen."),
            Mistake("reshape-size", error=_reshape_error,
                    en="`.reshape` gets other sizes than the array has. A filter has as many numbers as a row of `P`: `W.reshape(c_out, len(P[0]))`. And `Wm @ P.T` has `c_out * m * m` numbers, for `.reshape(c_out, m, m)`.",
                    nl="`.reshape` krijgt andere maten dan de array heeft. Een filter heeft evenveel getallen als een rij van `P`: `W.reshape(c_out, len(P[0]))`. En `Wm @ P.T` heeft `c_out * m * m` getallen, voor `.reshape(c_out, m, m)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the filters as rows, and the product with the rows of `P`, reshaped into output channels.",
                      "nl": "Vervang de twee `...`: de filters als rijen, en het product met de rijen van `P`, omgevormd tot output channels."},
            "arguments": {"en": "The check calls `{call}`, with a patch matrix, filters and the number of places per side, and the `def` line of your function has another number of names between its parentheses. Keep `(P, W, m)`.",
                          "nl": "De check roept `{call}` aan, met een matrix van patches, filters en het aantal plekken per kant, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(P, W, m)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your lines with steps 3 and 4 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je regels met stap 3 en 4 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any number of filters and channels?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elk aantal filters en channels?"},
            "correct": {"en": "Your function computes a whole convolutional layer with one `@`: every number of the output is the dot product of one filter and one patch, all at once.",
                        "nl": "Je functie rekent een hele convolutional layer uit met één `@`: elk getal van de output is het inproduct van één filter en één patch, allemaal tegelijk."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("patches", _P_START, "empty", None),
    ("patches", _p("            rows.append(x[:, r:r + k, c:c + k].flatten())\n"), "correct", None),
    ("patches", _p("            rows.append(x[0, r:r + k, c:c + k].reshape(k * k))\n"), "wrong", "one-channel"),
    ("patches", _p("            rows.append(x[:, c:c + k, r:r + k].reshape(c_in * k * k))\n"), "wrong", "swapped"),
    ("patches", _p("            rows.append(x[:, r:r + k, c:c + k])\n"), "wrong", "not-flat"),
    ("patches", _p("            rows.append(x[:, r:r + k, c:c + k].reshape(k * k))\n"), "error", "reshape-size"),
    ("patches", _p("            rows.append(x[r:r + k, c:c + k].reshape(c_in * k * k))\n"), "error", "reshape-size"),
    ("patches", _p("            rows.append(x[:, r:r + 2, c:c + 2].reshape(c_in * 4))\n"), "almost", None),
    ("conv-matmul", _M_START, "empty", None),
    ("conv-matmul", _m(wm="    Wm = W.reshape(c_out, -1)\n", out="    return (P @ Wm.T).T.reshape(c_out, m, m)\n"), "correct", None),
    ("conv-matmul", _m(out="    return (P @ Wm.T).reshape(c_out, m, m)\n"), "wrong", "rows-per-place"),
    ("conv-matmul", _m(out="    return Wm @ P.T\n"), "wrong", "no-reshape"),
    ("conv-matmul", _m(out="    return (Wm @ P).reshape(c_out, m, m)\n"), "error", "sizes"),
    ("conv-matmul", _m(out="    return np.dot(Wm, P).reshape(c_out, m, m)\n"), "error", "sizes"),
    ("conv-matmul", _m(wm="    Wm = W.reshape(c_out, c_in * k * k)\n"), "error", "no-c-in"),
    ("conv-matmul", _m(out="    return (Wm * P.T).reshape(c_out, m, m)\n"), "error", "star"),
    ("conv-matmul", _m(wm="    c_in = len(W[0])\n    k = len(W[0][0])\n    Wm = W.reshape(c_out, c_in * k * k)\n"), "correct", None),
    ("conv-matmul", _m(wm="    Wm = W.reshape(c_out, 4)\n"), "error", "reshape-size"),
    ("conv-matmul", _m(out="    return (Wm @ P.T).reshape(c_out, 2, 2)\n"), "error", "reshape-size"),
]
