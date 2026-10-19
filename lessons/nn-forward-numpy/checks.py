# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-forward-numpy: the shapes through a network with two hidden layers, and a forward pass
for all examples at once, with `X @ W.T + b` and ReLU as `np.maximum`. And the picture: one layer of the worked
example as grids, with the row of one example, the row of one neuron and the number they give."""
import re

from everything_ai.check import FunctionBox, CodeBox, Mistake

# The worked example: 4 examples with 2 inputs, a hidden layer of 3 neurons.
X = [[1, 2], [3, 1], [0, 2], [2, 2]]
W1 = [[1, 1], [0, 1], [1, -1]]
B1 = [-2, 0, 0]


def picture(notes: list[str]) -> list[tuple]:
    """The grids for draw.array_grids: X, W1 and X @ W1.T + b1, with the second example, the third neuron and the
    number in row 2, column 3 drawn thick. `notes` are the second caption lines, in the lesson's language."""
    Z = [[sum(x * w for x, w in zip(row, neuron)) + b for neuron, b in zip(W1, B1)] for row in X]
    cells = lambda rows: [[str(v).replace("-", "−") for v in row] for row in rows]
    return [("X", cells(X), ["shape (4, 2)", notes[0]], {(1, 0), (1, 1)}),
            ("W1", cells(W1), ["shape (3, 2)", notes[1]], {(2, 0), (2, 1)}),
            ("X @ W1.T + b1", cells(Z), ["shape (4, 3)", notes[2]], {(1, 2)})]


# The lesson is done when these are correct. "test-sizes" is the choice question in the lesson.
REQUIRED = ["shapes", "forward", "test-sizes"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# --- exercise 1: the shapes ---------------------------------------------------------------------------------------

_NAMES = ["shape_W1", "shape_H1", "shape_W2", "shape_H2", "shape_W3", "shape_y_pred"]
_RIGHT = ["(16, 20)", "(32, 16)", "(8, 16)", "(32, 8)", "(3, 8)", "(32, 3)"]
_S_END = "[" + ", ".join(_NAMES) + "]"
_S_START = "shape_X = (32, 20)\n" + "".join(f"{n} = ...\n" for n in _NAMES) + _S_END + "\n"


def _s(*shapes: str) -> str:
    return "shape_X = (32, 20)\n" + "".join(f"{n} = {s}\n" for n, s in zip(_NAMES, shapes)) + _S_END


_S = _s(*_RIGHT)


def _at(got, k: int):
    """Shape number k of the learner's list, as a tuple of ints, or None."""
    try:
        return tuple(int(v) for v in got[k])
    except (TypeError, ValueError, IndexError):
        return None


def _same_shapes(got, want) -> bool:
    """The same shapes, written as tuples or as lists: the numbers count, not the brackets."""
    try:
        return len(got) == len(want) and all(tuple(g) == tuple(w) for g, w in zip(got, want))
    except TypeError:
        return False


def _w_flipped(got, text) -> bool:
    """A weight matrix written as (inputs, neurons)."""
    return any(_at(got, k) == want for k, want in [(0, (20, 16)), (2, (16, 8)), (4, (8, 3))])


def _h_flipped(got, text) -> bool:
    """An output written as (neurons, examples)."""
    return any(_at(got, k) == want for k, want in [(1, (16, 32)), (3, (8, 32)), (5, (3, 32))])


def _h_no_examples(got, text) -> bool:
    """An output without the examples: one number per neuron."""
    return any(_at(got, k) == want for k, want in [(1, (16,)), (3, (8,)), (5, (3,))])


# --- exercise 2: the forward pass ---------------------------------------------------------------------------------

_F_HEAD = ("import numpy as np\n\n\ndef forward(X, W1, b1, W2, b2):\n    X = np.array(X)\n    W1 = np.array(W1)\n"
           "    b1 = np.array(b1)\n    W2 = np.array(W2)\n    b2 = np.array(b2)\n")
_F_START = _F_HEAD + "    H = ...\n    return ...\n"
_HIDDEN = "    H = np.maximum(0, X @ W1.T + b1)\n"
_OUT = "    return H @ W2.T + b2"
_F = _F_HEAD + _HIDDEN + _OUT


def _f(hidden: str = _HIDDEN, out: str = _OUT) -> str:
    return _F_HEAD + hidden + out


def _close(got, want) -> bool:
    """The same shape and the same numbers as the course's outputs."""
    import numpy as np
    try:
        g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return False
    return g.shape == w.shape and bool(np.allclose(g, w, rtol=1e-6, atol=1e-9))


def _plain(v) -> str:
    """A value as the lessons write it: lists and numbers to 4 decimals, never array(...) or np.float64(...)."""
    import numpy as np
    try:
        return str((np.round(np.asarray(v, dtype=float), 4) + 0.0).tolist())
    except (TypeError, ValueError):
        return repr(v)


def _other_shape(got, text) -> bool:
    """The right numbers for the example, but not one row per example and one column: a flat array, or one row."""
    import numpy as np
    try:
        g = np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return False
    return g.shape != (2, 1) and g.size == 2 and bool(np.allclose(g.ravel(), [-1.5, 1.5]))


_VISIBLE = ([[1, 2], [2, 0]], [[1, -1], [0, 1], [2, 1]], [0, -1, -3], [[2, 1, -3]], [0.5])

EXERCISES = {
    "shapes": CodeBox(
        name="shapes",
        start=_S_START,
        solution=_S,
        keep=[_S_END],
        compare=_same_shapes,
        first=[_DOTS],
        mistakes=[
            Mistake("w-flipped", matches=_w_flipped,
                    en="Your last line gives `{got}`. A weight matrix has one row per neuron and one column per input: the first hidden layer has 16 neurons and 20 inputs, so `W1` has shape (16, 20). In `X @ W1.T` it is flipped, and then it is (20, 16).",
                    nl="Je laatste regel geeft `{got}`. Een weight-matrix heeft één rij per neuron en één kolom per input: de eerste hidden layer heeft 16 neuronen en 20 inputs, dus `W1` heeft shape (16, 20). In `X @ W1.T` wordt hij omgeklapt, en dan is hij (20, 16)."),
            Mistake("h-flipped", matches=_h_flipped,
                    en="Your last line gives `{got}`. The output of a layer has one row per example and one column per neuron, as `X` has one row per example: `H1` has shape (32, 16).",
                    nl="Je laatste regel geeft `{got}`. De output van een layer heeft één rij per voorbeeld en één kolom per neuron, zoals `X` één rij per voorbeeld heeft: `H1` heeft shape (32, 16)."),
            Mistake("no-examples", matches=_h_no_examples,
                    en="Your last line gives `{got}`. That is the output for one example. The network works out all 32 examples at once, so each output has one row per example: `H1` has shape (32, 16).",
                    nl="Je laatste regel geeft `{got}`. Dat is de output voor één voorbeeld. Het netwerk rekent alle 32 voorbeelden tegelijk uit, dus elke output heeft één rij per voorbeeld: `H1` heeft shape (32, 16)."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace each `...` with the shape of that array, as a tuple such as `(32, 20)`.",
                      "nl": "Vervang elke `...` door de shape van die array, als tuple zoals `(32, 20)`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
            "no_value": {"en": "The last line of the box shows nothing. Put the list of shapes on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet de lijst met shapes op de laatste regel, zodat de check hem kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Go layer by layer: a `W` is (neurons, inputs), and the output of its layer is (examples, neurons). The inputs of a layer are the neurons of the layer before it.",
                      "nl": "Je laatste regel geeft `{got}`. Ga layer voor layer: een `W` is (neuronen, inputs), en de output van zijn layer is (voorbeelden, neuronen). De inputs van een layer zijn de neuronen van de layer ervoor."},
            "correct": {"en": "(16, 20), (32, 16), (8, 16), (32, 8), (3, 8) and (32, 3). The inputs of each layer are the neurons of the layer before it, and the 32 examples stay the rows all the way.",
                        "nl": "(16, 20), (32, 16), (8, 16), (32, 8), (3, 8) en (32, 3). De inputs van elke layer zijn de neuronen van de layer ervoor, en de 32 voorbeelden blijven de hele weg de rijen."},
        },
    ),
    "forward": FunctionBox(
        name="forward",
        function="forward",
        start=_F_START,
        solution=_F,
        visible=[_VISIBLE],
        hidden=[([[0.5, -1.0, 2.0]], [[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]], [0.0, 1.0], [[1.0, -1.0], [2.0, 0.5], [-1.0, 1.0]], [0.0, 0.5, -0.5]),
                ([[1, 0], [0, 1], [1, 1]], [[2, -1], [1, 2]], [0, 0], [[1, -1]], [0]),
                ([[1.0, 2.0], [0.0, -1.0], [3.0, 1.0], [-2.0, 0.5]], [[0.5, -1.0], [1.0, 1.0], [-1.0, 0.0], [0.2, 0.3]], [0.1, -0.5, 1.0, 0.0],
                 [[1.0, -2.0, 0.5, 1.0]], [-0.2]),
                ([[2.0, 1.0, 0.0, -1.0]], [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]], [-1.0, 0.0, -2.0], [[3.0, -1.0, 1.0], [0.0, 2.0, -2.0]], [1.0, -1.0])],
        compare=_close,
        show=_plain,
        first=[_DOTS],
        mistakes=[
            Mistake("python-max", error=lambda e: isinstance(e, ValueError) and "truth value" in str(e),
                    en="With `{call}`, Python stops: your code asks whether a whole array is larger than 0, with `max(0, ...)` or an `if`, and Python cannot answer that with one yes or no. `np.maximum(0, Z)` takes the larger of 0 and each number of `Z`.",
                    nl="Met `{call}` stopt Python: je code vraagt of een hele array groter is dan 0, met `max(0, ...)` of een `if`, en daar kan Python niet met één ja of nee op antwoorden. `np.maximum(0, Z)` neemt het grootste van 0 en elk getal van `Z`."),
            Mistake("no-transpose", syntax=r"@\s*W[12]\b(?!\s*\.T)|np\.dot\([^,()]*,\s*W[12]\s*\)",
                    en="NumPy stops on the line `{line}`: a weight matrix has one row per neuron, shape (neurons, inputs), so the inner numbers of the multiplication do not fit. Flip it first, with `.T`: `X @ W1.T` and `H @ W2.T`.",
                    nl="NumPy stopt op de regel `{line}`: een weight-matrix heeft één rij per neuron, shape (neuronen, inputs), dus de binnenste getallen van de vermenigvuldiging passen niet. Klap hem eerst om, met `.T`: `X @ W1.T` en `H @ W2.T`."),
            Mistake("w-first", syntax=r"\bW[12]\s*@|np\.dot\(\s*W[12]\b",
                    en="NumPy stops with an error. On the line `{line}` the weight matrix comes first, as in `W @ x` for one example. With the examples as the rows of `X`, `X` comes first and the weight matrix is flipped: `X @ W1.T` and `H @ W2.T`, so that each row stays one example.",
                    nl="NumPy stopt met een fout. Op de regel `{line}` staat de weight-matrix voorop, zoals in `W @ x` voor één voorbeeld. Met de voorbeelden als rijen van `X` komt `X` voorop en klap je de weight-matrix om: `X @ W1.T` en `H @ W2.T`, zodat elke rij één voorbeeld blijft."),
            Mistake("star", syntax=r"\b(X|H)\s*\*\s*W[12]\b",
                    en="NumPy stops on the line `{line}`: `*` multiplies number by number and adds nothing up. A layer needs matrix multiplication: `@`.",
                    nl="NumPy stopt op de regel `{line}`: `*` vermenigvuldigt getal voor getal en telt niets op. Een layer heeft matrixvermenigvuldiging nodig: `@`."),
            Mistake("no-relu", wrong=_f(hidden="    H = X @ W1.T + b1\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your hidden layer passes on its `z` without an activation function. Put it through ReLU: `np.maximum(0, ...)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je hidden layer geeft zijn `z` door zonder activation function. Haal hem door ReLU: `np.maximum(0, ...)`."),
            Mistake("min", wrong=_f(hidden="    H = np.minimum(0, X @ W1.T + b1)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `np.minimum` keeps the negative part and drops the positive one. ReLU is the other way round: `np.maximum(0, ...)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `np.minimum` houdt het negatieve deel en laat het positieve vallen. ReLU is andersom: `np.maximum(0, ...)`."),
            Mistake("relu-output", wrong=_f(out="    return np.maximum(0, H @ W2.T + b2)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you also put the output layer through ReLU, so a negative output becomes 0. Here the output layer has no activation function.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je haalt ook de output layer door ReLU, dus een negatieve output wordt 0. Hier heeft de output layer geen activation function."),
            Mistake("bias-after", wrong=_f(hidden="    H = np.maximum(0, X @ W1.T) + b1\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you add `b1` after ReLU. The bias terms belong to `z`, before the activation function: `np.maximum(0, X @ W1.T + b1)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt `b1` op na ReLU. De bias terms horen bij `z`, vóór de activation function: `np.maximum(0, X @ W1.T + b1)`."),
            Mistake("no-b1", wrong=_f(hidden="    H = np.maximum(0, X @ W1.T)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you leave out the bias terms of the hidden layer. Add `b1` before ReLU.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je laat de bias terms van de hidden layer weg. Tel `b1` erbij op vóór ReLU."),
            Mistake("no-b2", wrong=_f(out="    return H @ W2.T"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you leave out the bias terms of the output layer. Add `b2`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je laat de bias terms van de output layer weg. Tel `b2` erbij op."),
            Mistake("hidden-only", wrong=_f(out="    return H"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the output of the hidden layer. Send `H` through the output layer too: `H @ W2.T + b2`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is de output van de hidden layer. Stuur `H` ook door de output layer: `H @ W2.T + b2`."),
            Mistake("other-shape", matches=_other_shape,
                    en="`{call}` gives `{got}`, and it should give `{want}`: the right numbers, but in another shape. The output of the network has one row per example and one column per output neuron, so here shape (2, 1), as `H @ W2.T + b2` gives.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de goede getallen, maar in een andere shape. De output van het netwerk heeft één rij per voorbeeld en één kolom per output-neuron, hier dus shape (2, 1), zoals `H @ W2.T + b2` geeft."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: the hidden layer with ReLU in `H`, and the output layer after `return`.",
                      "nl": "Vervang de twee `...`: de hidden layer met ReLU in `H`, en de output layer na `return`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with five values, and the `def` line of your function has a different number of names between its parentheses. Keep `(X, W1, b1, W2, b2)`.",
                          "nl": "De check roept `{call}` aan, met vijf waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X, W1, b1, W2, b2)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does the hidden layer work out `np.maximum(0, X @ W1.T + b1)`, and the output layer `H @ W2.T + b2`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Rekent de hidden layer `np.maximum(0, X @ W1.T + b1)` uit, en de output layer `H @ W2.T + b2`?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Do you work it out from `X`, `W1`, `b1`, `W2` and `b2` themselves, for any number of examples, inputs and neurons?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken je het uit met `X`, `W1`, `b1`, `W2` en `b2` zelf, voor elk aantal voorbeelden, inputs en neuronen?"},
            "correct": {"en": "Your function is right in all {n} cases. In the example the hidden layer gives (0, 1, 1) and (2, 0, 1), and the outputs are −1.5 and 1.5: one row per example, one column for the output neuron.",
                        "nl": "Je functie klopt in alle {n} gevallen. In het voorbeeld geeft de hidden layer (0, 1, 1) en (2, 0, 1), en de outputs zijn −1,5 en 1,5: één rij per voorbeeld, één kolom voor het output-neuron."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("shapes", _S_START, "empty", None),
    ("shapes", _S, "correct", None),
    ("shapes", _s("(16, 20)", "(32, 16)", "(8, 16)", "(32, 8)", "(3, 8)", "(32, 3)").replace("(16, 20)\n", "(16,20)\n"), "correct", None),
    ("shapes", _s("(20, 16)", "(32, 16)", "(16, 8)", "(32, 8)", "(8, 3)", "(32, 3)"), "wrong", "w-flipped"),
    ("shapes", _s("(16, 20)", "(16, 32)", "(8, 16)", "(8, 32)", "(3, 8)", "(3, 32)"), "wrong", "h-flipped"),
    ("shapes", _s("(16, 20)", "(16,)", "(8, 16)", "(8,)", "(3, 8)", "(3,)"), "wrong", "no-examples"),
    ("shapes", _s("(16, 20)", "(32, 16)", "(8, 20)", "(32, 8)", "(3, 20)", "(32, 3)"), "wrong", None),
    ("shapes", _s("[16, 20]", "[32, 16]", "[8, 16]", "[32, 8]", "[3, 8]", "[32, 3]"), "correct", None),
    ("forward", _F_START, "empty", None),
    ("forward", _F, "correct", None),
    ("forward", _f(hidden="    Z = X @ W1.T + b1\n    H = np.maximum(Z, 0)\n"), "correct", None),
    ("forward", _f(hidden="    H = np.maximum(0, np.dot(X, W1.T) + b1)\n", out="    y_pred = H @ W2.T + b2\n    return y_pred"), "correct", None),
    ("forward", _f(hidden="    Z = X @ W1.T + b1\n    H = Z * (Z > 0)\n"), "correct", None),
    ("forward", _f(hidden="    H = max(0, X @ W1.T + b1)\n"), "error", "python-max"),
    ("forward", _f(hidden="    Z = X @ W1.T + b1\n    H = Z if Z > 0 else 0\n"), "error", "python-max"),
    ("forward", _f(hidden="    H = np.maximum(0, X @ W1 + b1)\n"), "error", "no-transpose"),
    ("forward", _f(out="    return H @ W2 + b2"), "error", "no-transpose"),
    ("forward", _f(hidden="    H = np.maximum(0, np.dot(X, W1) + b1)\n"), "error", "no-transpose"),
    ("forward", _f(hidden="    H = np.maximum(0, W1 @ X + b1)\n"), "error", "w-first"),
    ("forward", _f(hidden="    H = np.maximum(0, W1 @ X.T + b1)\n"), "error", "w-first"),
    ("forward", _f(hidden="    H = np.maximum(0, (W1 @ X.T).T + b1)\n", out="    return (W2 @ H.T).T + b2"), "correct", None),
    ("forward", _f(hidden="    H = np.maximum(0, X * W1 + b1)\n"), "error", "star"),
    ("forward", _f(hidden="    H = X @ W1.T + b1\n"), "wrong", "no-relu"),
    ("forward", _f(hidden="    H = np.minimum(0, X @ W1.T + b1)\n"), "wrong", "min"),
    ("forward", _f(out="    return np.maximum(0, H @ W2.T + b2)"), "wrong", "relu-output"),
    ("forward", _f(hidden="    H = np.maximum(0, X @ W1.T) + b1\n"), "wrong", "bias-after"),
    ("forward", _f(hidden="    H = np.maximum(0, X @ W1.T)\n"), "wrong", "no-b1"),
    ("forward", _f(out="    return H @ W2.T"), "wrong", "no-b2"),
    ("forward", _f(out="    return H"), "wrong", "hidden-only"),
    ("forward", _f(out="    return H @ W2[0] + b2"), "wrong", "other-shape"),
]
