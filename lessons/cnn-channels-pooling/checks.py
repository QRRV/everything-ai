# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cnn-channels-pooling: conv_channels, a convolution of an input with several channels by
several filters, each giving one output channel; and max_pool, max pooling of every channel."""
import re

import numpy as np

from everything_ai import draw
from everything_ai.check import FunctionBox, Mistake

POOL_IN = [[1, 3, 2, 0], [4, 2, 1, 5], [0, 1, 3, 2], [2, 6, 1, 1]]


def pool_picture(labels: dict):
    """A 4 by 4 channel and its max pooling of 2 by 2: the largest number of each block is marked."""
    a = np.array(POOL_IN)
    out = [[int(a[2 * r:2 * r + 2, 2 * c:2 * c + 2].max()) for c in range(2)] for r in range(2)]
    marked = set()
    for r in range(2):
        for c in range(2):
            block = a[2 * r:2 * r + 2, 2 * c:2 * c + 2]
            i, j = np.unravel_index(int(block.argmax()), block.shape)
            marked.add((2 * r + int(i), 2 * c + int(j)))
    as_text = lambda rows: [[str(v) for v in row] for row in rows]
    return draw.array_grids([(labels["input"], as_text(POOL_IN), [labels["input_caption"]], marked),
                             (labels["output"], as_text(out), [labels["output_caption"]])],
                            indices=False, title=labels["title"], block=2)


# The lesson is done when these are correct. "weights" is the choice question in the lesson.
REQUIRED = ["weights", "conv-channels", "max-pool"]

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


# --- exercise 2: conv_channels --------------------------------------------------------------------------------------------

_C_HEAD = ("import numpy as np\n\n\ndef conv_channels(x, W):\n    x = np.array(x)\n    W = np.array(W)\n"
           "    c_out = len(W)\n    k = len(W[0][0])\n    m = len(x[0]) - k + 1\n    out = np.zeros((c_out, m, m))\n"
           "    for o in range(c_out):\n        for r in range(m):\n            for c in range(m):\n")
_C_STEP = "                out[o, r, c] = (x[:, r:r + k, c:c + k] * W[o]).sum()\n"
_C_START = _C_HEAD + "                # filter o on the piece at r, c of every input channel, all added up\n                out[o, r, c] = ...\n    return out\n"


def _c(step: str = _C_STEP) -> str:
    return _C_HEAD + step + "    return out\n"


_C = _c()

# --- exercise 3: max_pool -------------------------------------------------------------------------------------------------

_P_HEAD = ("import numpy as np\n\n\ndef max_pool(x, size):\n    x = np.array(x)\n    channels = len(x)\n"
           "    m = len(x[0]) // size\n    out = np.zeros((channels, m, m))\n    for ch in range(channels):\n"
           "        for r in range(m):\n            for c in range(m):\n")
_P_STEP = "                out[ch, r, c] = x[ch, r * size:r * size + size, c * size:c * size + size].max()\n"
_P_START = _P_HEAD + "                # the largest number of the block of size by size at r, c\n                out[ch, r, c] = ...\n    return out\n"


def _p(step: str = _P_STEP) -> str:
    return _P_HEAD + step + "    return out\n"


_P = _p()

_X2 = [[[1, 0, 2, 1], [0, 1, 1, 0], [2, 1, 0, 1], [1, 0, 1, 2]],
       [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]]
_X3 = [[[1, 2, 0], [0, 1, 3], [2, 1, 1]], [[0, 0, 1], [1, 2, 0], [0, 1, 0]], [[3, 1, 0], [0, 0, 1], [1, 1, 2]]]
_W_ONE = [[[[1, 0], [0, -1]]]]

EXERCISES = {
    "conv-channels": FunctionBox(
        name="conv-channels",
        function="conv_channels",
        start=_C_START,
        solution=_C,
        visible=[([[[1, 2, 0], [0, 1, 3], [2, 1, 1]], [[0, 1, 1], [1, 0, 2], [3, 1, 0]]],
                  [[[[1, 0], [0, 1]], [[0, 1], [1, 0]]], [[[1, 1], [1, 1]], [[-1, 0], [0, 0]]]])],
        hidden=[(_X2, [[[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 1], [0, 1, 0], [1, 0, 0]]]]),
                ([[[1, 2, 0], [0, 1, 3], [2, 1, 1]]], _W_ONE),
                (_X3, [[[[1]], [[1]], [[1]]], [[[1]], [[0]], [[-1]]], [[[0]], [[2]], [[0]]]])],
        compare=_same_grid,
        show=_grid,
        first=[_DOTS],
        mistakes=[
            Mistake("first-channel", wrong=_c("                out[o, r, c] = (x[0, r:r + k, c:c + k] * W[o][0]).sum()\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your filters only look at the first input channel. A filter has a slice for every input channel: `x[:, r:r + k, c:c + k] * W[o]` takes all of them, and `.sum()` adds them up.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je filters kijken alleen naar het eerste input channel. Een filter heeft een plak voor elk input channel: `x[:, r:r + k, c:c + k] * W[o]` neemt ze allemaal, en `.sum()` telt ze op."),
            Mistake("same-filter", wrong=_c("                out[o, r, c] = (x[:, r:r + k, c:c + k] * W[0]).sum()\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every output channel uses the first filter. Output channel `o` comes from filter `o`: `W[o]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elk output channel gebruikt het eerste filter. Output channel `o` komt van filter `o`: `W[o]`."),
            Mistake("all-filters", wrong=_c("                out[o, r, c] = (x[:, r:r + k, c:c + k] * W).sum()\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every output channel is the same, the sum over all filters together. Output channel `o` uses filter `o` alone: `W[o]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elk output channel is hetzelfde, de som over alle filters samen. Output channel `o` gebruikt alleen filter `o`: `W[o]`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: the piece at `r`, `c` of every input channel, times filter `o`, all added up.",
                      "nl": "Vervang de `...`: het stuk op `r`, `c` van elk input channel, keer filter `o`, alles opgeteld."},
            "arguments": {"en": "The check calls `{call}`, with an input and filters, and the `def` line of your function has another number of names between its parentheses. Keep `(x, W)`.",
                          "nl": "De check roept `{call}` aan, met een input en filters, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x, W)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your line with step 2 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je regel met stap 2 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any number of channels and any filter size?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elk aantal channels en elke filtergrootte?"},
            "correct": {"en": "Your layer makes one output channel per filter, each from all input channels together. Three filters of 1 by 1 on three channels mix the channels at every pixel, without looking at neighbors.",
                        "nl": "Je layer maakt één output channel per filter, elk uit alle input channels samen. Drie filters van 1 bij 1 op drie channels mengen de channels op elke pixel, zonder naar de buren te kijken."},
        },
    ),
    "max-pool": FunctionBox(
        name="max-pool",
        function="max_pool",
        start=_P_START,
        solution=_P,
        visible=[([[[2, 0, 1, 3], [1, 4, 0, 2], [3, 1, 5, 0], [0, 2, 1, 1]]], 2)],
        hidden=[(_X2, 2), ([[[1, 0], [0, 1]], [[5, 2], [3, 4]]], 2), (_X3, 3), ([[[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1], [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0], [2, 2, 2, 2, 2, 2], [9, 0, 0, 0, 0, 0]]], 3),
                ([[[-1, -2], [-3, -4]]], 2), ([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], 2)],
        compare=_same_grid,
        show=_grid,
        first=[_DOTS],
        mistakes=[
            Mistake("no-stride", wrong=_p("                out[ch, r, c] = x[ch, r:r + size, c:c + size].max()\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your blocks overlap, because they start one place apart. Pooling uses blocks next to each other: block `r` starts at row `r * size`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je blokken overlappen, omdat ze één plek uit elkaar beginnen. Pooling gebruikt blokken naast elkaar: blok `r` begint bij rij `r * size`."),
            Mistake("mixes-channels", wrong=_p("                out[ch, r, c] = x[:, r * size:r * size + size, c * size:c * size + size].max()\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your pooling takes the largest number over all channels at once, so the channels get mixed. Pooling works on every channel on its own: take the block from channel `ch` only, `x[ch, ...]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je pooling neemt het grootste getal over alle channels tegelijk, dus de channels raken gemengd. Pooling werkt op elk channel apart: neem het blok alleen uit channel `ch`, `x[ch, ...]`."),
            Mistake("mean", wrong=_p("                out[ch, r, c] = x[ch, r * size:r * size + size, c * size:c * size + size].mean()\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is average pooling, the mean of each block. Max pooling keeps the largest number: `.max()`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is average pooling, het gemiddelde van elk blok. Max pooling houdt het grootste getal: `.max()`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: the largest number of the block of `size` by `size` that belongs to `r`, `c`.",
                      "nl": "Vervang de `...`: het grootste getal van het blok van `size` bij `size` dat bij `r`, `c` hoort."},
            "arguments": {"en": "The check calls `{call}`, with an input and a block size, and the `def` line of your function has another number of names between its parentheses. Keep `(x, size)`.",
                          "nl": "De check roept `{call}` aan, met een input en een blokgrootte, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x, size)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your line with step 3 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je regel met stap 3 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every block size and every channel?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke blokgrootte en elk channel?"},
            "correct": {"en": "Your pooling keeps the largest number of every block, per channel: an image of 6 by 6 with blocks of 3 becomes 2 by 2, and a row or column that does not fill a whole block is left out, as `//` says.",
                        "nl": "Je pooling houdt het grootste getal van elk blok, per channel: een afbeelding van 6 bij 6 met blokken van 3 wordt 2 bij 2, en een rij of kolom die geen heel blok vult, valt weg, zoals `//` zegt."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("conv-channels", _C_START, "empty", None),
    ("conv-channels", _c("                out[o, r, c] = np.sum(x[:, r:r + k, c:c + k] * W[o])\n"), "correct", None),
    ("conv-channels", _c("                out[o, r, c] = (x[0, r:r + k, c:c + k] * W[o][0]).sum()\n"), "wrong", "first-channel"),
    ("conv-channels", _c("                out[o, r, c] = (x[:, r:r + k, c:c + k] * W[0]).sum()\n"), "wrong", "same-filter"),
    ("conv-channels", _c("                out[o, r, c] = np.sum(x[:, r:r + k, c:c + k] * W)\n"), "wrong", "all-filters"),
    ("max-pool", _P_START, "empty", None),
    ("max-pool", _p("                out[ch, r, c] = np.max(x[ch][r * size:(r + 1) * size, c * size:(c + 1) * size])\n"), "correct", None),
    ("max-pool", _p("                out[ch, r, c] = x[ch, r:r + size, c:c + size].max()\n"), "wrong", "no-stride"),
    ("max-pool", _p("                out[ch, r, c] = x[ch, r * size:r * size + size, c * size:c * size + size].mean()\n"), "wrong", "mean"),
    ("max-pool", _p("                out[ch, r, c] = x[0, r * size:r * size + size, c * size:c * size + size].max()\n"), "almost", None),
    ("max-pool", _p("                out[ch, r, c] = x[:, r * size:r * size + size, c * size:c * size + size].max()\n"), "wrong", "mixes-channels"),
]
