# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-vanishing-exploding: chain_gradient, the gradient that comes back through a chain of
sigmoid layers with one weight each, and relu_grad_sizes, how large the gradient is after each layer of a deep ReLU
network on the way back. And the picture: the size of the gradient through 20 ReLU layers, for three spreads of the
starting weights."""
import math
import re

import numpy as np

from everything_ai.check import FunctionBox, Mistake


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu_sizes(scale: float, layers: int = 20, d: int = 50, seed: int = 0) -> list[float]:
    """The mean absolute gradient after each layer, going back, for a random ReLU network whose weights have the
    spread `scale / sqrt(d)`; the gradient at the top is 1 for every output."""
    rng = np.random.default_rng(seed)
    Ws = [rng.normal(0, scale / math.sqrt(d), (d, d)) for _ in range(layers)]
    h = rng.normal(size=d)
    zs = []
    for W in Ws:
        z = h @ W.T
        zs.append(z)
        h = np.maximum(z, 0)
    g = np.ones(d)
    sizes = []
    for W, z in zip(reversed(Ws), reversed(zs)):
        g = (g * (z > 0)) @ W
        sizes.append(float(np.mean(np.abs(g))))
    return sizes


def sizes_picture(labels: dict) -> str:
    from everything_ai import chart
    steps = list(range(1, 21))
    panel = chart.Panel((0.5, 20.5), (1e-3, 1e4), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], ylog=True,
                        xticks=[1, 5, 10, 15, 20])
    places = {"big": (12.5, 700, "end"), "right": (13, 10, "middle"), "small": (8, 0.012, "middle")}   # beside each line, not on it
    for scale, dashed, key in ((2.0, False, "big"), (math.sqrt(2), True, "right"), (1.0, False, "small")):
        panel.line(steps, relu_sizes(scale), "grad", 2.4 if not dashed else 2.0, dashed=dashed)
        x, y, anchor = places[key]
        panel.text(x, y, labels[key], "grad", anchor)
    return chart.figure([panel], alt=labels["alt"], width=560, height=310)


# The lesson is done when these are correct. "nan-loss" is the choice question in the lesson.
REQUIRED = ["nan-loss", "chain-gradient", "relu-sizes"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _plain(x) -> str:
    """A number with 4 significant digits, written out, never in e-notation: 0.0000003479, not 3.479e-07."""
    x = float(x)
    if not math.isfinite(x):
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    if x == 0:
        return "0"
    text = np.format_float_positional(x, precision=4, unique=False, fractional=False, trim="-")
    return text.rstrip(".")


def _show(v) -> str:
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return repr(v)
    if a.ndim == 0:
        return _plain(a)
    return "[" + ", ".join(_plain(x) for x in a.ravel()) + "]"


def _close(got, want) -> bool:
    try:
        g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    except (TypeError, ValueError):
        return False
    return g.shape == w.shape and bool(np.allclose(g, w, rtol=1e-6, atol=0))


class _Matrix(np.ndarray):
    """An array that a check message writes the way the lesson's code does: np.array([...]). The learner's code gets
    a plain copy."""

    def __repr__(self):
        return f"np.array({np.asarray(self).tolist()!r})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _m(rows) -> np.ndarray:
    return np.array(rows, dtype=float).view(_Matrix)


# --- exercise 2: a chain of sigmoid layers ------------------------------------------------------------------------------

_C_HEAD = ("import numpy as np\n\n\ndef sigmoid(z):\n    return 1 / (1 + np.exp(-z))\n\n\ndef chain_gradient(ws, x):\n")
_C_START = _C_HEAD + "    # the gradient that comes back to x through one sigmoid layer for each weight in ws\n    ...\n"


def _c(factor: str = "w * sigmoid(z) * (1 - sigmoid(z))", z: str = "w * h") -> str:
    return (_C_HEAD + "    h = x\n    zs = []\n    for w in ws:\n" + f"        z = {z}\n" + "        zs.append(z)\n        h = sigmoid(z)\n"
            "    g = 1.0\n    for k in reversed(range(len(ws))):\n        w = ws[k]\n        z = zs[k]\n" + f"        g = g * {factor}\n" + "    return g\n")


# --- exercise 3: a deep ReLU network ----------------------------------------------------------------------------------------

_R_HEAD = "import numpy as np\n\n\ndef relu_grad_sizes(Ws, x):\n"
_R_START = _R_HEAD + "    # the mean absolute gradient after each layer on the way back, from the top layer to the first\n    ...\n"


def _r(step: str = "(g * (z > 0)) @ W", size: str = "np.mean(np.abs(g))", order: str = "reversed(range(len(Ws)))") -> str:
    return (_R_HEAD + "    h = x\n    zs = []\n    for W in Ws:\n        z = h @ W.T\n        zs.append(z)\n        h = np.maximum(z, 0)\n"
            "    g = np.ones(len(h))\n    sizes = []\n" + f"    for k in {order}:\n" + "        W = Ws[k]\n        z = zs[k]\n" + f"        g = {step}\n" + f"        sizes.append({size})\n"
            "    return sizes\n")


_W_A = _m([[1.0, -1.0], [0.5, 2.0]])
_W_B = _m([[2.0, 0.0], [-1.0, 1.0]])
_W_C = _m([[0.5, 1.0, -1.0], [1.0, -0.5, 0.5], [-1.0, 1.0, 2.0]])
_W_D = _m([[1.0, 0.5, -1.0], [-0.5, 1.0, 1.0]])
_W_E = _m([[1.0, -1.0], [0.5, 0.5], [-1.0, 2.0], [2.0, 1.0]])
# a learner's mistake that only shows when the layers keep different places: the z of the last layer for every layer
_R_ONE_Z = _R_HEAD + "    h = x\n    for W in Ws:\n        z = h @ W.T\n        h = np.maximum(z, 0)\n    g = np.ones(len(h))\n    sizes = []\n    for W in reversed(Ws):\n        g = (g * (z > 0)) @ W\n        sizes.append(np.mean(np.abs(g)))\n    return sizes\n"
# the size taken before the layer multiplies the gradient, so the list starts with the size of the ones
_R_SIZE_FIRST = _R_HEAD + "    h = x\n    zs = []\n    for W in Ws:\n        z = h @ W.T\n        zs.append(z)\n        h = np.maximum(z, 0)\n    g = np.ones(len(h))\n    sizes = []\n    for k in reversed(range(len(Ws))):\n        sizes.append(np.mean(np.abs(g)))\n        g = (g * (zs[k] > 0)) @ Ws[k]\n    return sizes\n"

EXERCISES = {
    "chain-gradient": FunctionBox(
        name="chain-gradient",
        function="chain_gradient",
        start=_C_START,
        solution=_c(),
        visible=[([2.0, 1.0, 1.5], 0.5)],
        hidden=[([2.0, -1.0, 0.5, 3.0], 1.0), ([0.5], -2.0), ([4.0] * 6, 0.5), ([1.0] * 10, 0.5)],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("no-weight", wrong=_c("sigmoid(z) * (1 - sigmoid(z))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: each layer multiplies by the slope of its sigmoid, but also by its weight, because `z = w * h`. The factor of a layer is `w * sigmoid(z) * (1 - sigmoid(z))`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elke layer vermenigvuldigt met de helling van zijn sigmoid, maar ook met zijn weight, omdat `z = w * h`. De factor van een layer is `w * sigmoid(z) * (1 - sigmoid(z))`."),
            Mistake("no-slope", wrong=_c("w * sigmoid(z)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you multiply by what the sigmoid gives, not by its slope. The slope of the sigmoid at `z` is `sigmoid(z) * (1 - sigmoid(z))`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je vermenigvuldigt met wat de sigmoid geeft, niet met zijn helling. De helling van de sigmoid bij `z` is `sigmoid(z) * (1 - sigmoid(z))`."),
            Mistake("same-input", wrong=_c(z="w * x"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every layer gets `x` itself. Each layer gets what the layer before it gave: `z = w * h`, with `h` the output of the layer before.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elke layer krijgt `x` zelf. Elke layer krijgt wat de layer ervoor gaf: `z = w * h`, met `h` de output van de layer ervoor."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: the forward pass, keeping each `z`, and then the way back, multiplying the gradient by each layer's factor.",
                      "nl": "Vervang de `...` door de body van de functie: de forward pass, waarbij je elke `z` bewaart, en dan de weg terug, waarbij je de gradient met de factor van elke layer vermenigvuldigt."},
            "arguments": {"en": "The check calls `{call}`, with a list of weights and a number, and the `def` line of your function has another number of names between its parentheses. Keep `(ws, x)`.",
                          "nl": "De check roept `{call}` aan, met een lijst weights en een getal, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(ws, x)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Work out the factor of the last layer by hand: its weight times the slope of the sigmoid at its `z`.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken de factor van de laatste layer met de hand uit: zijn weight keer de helling van de sigmoid bij zijn `z`."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any number of layers, with a different weight in each?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elk aantal layers, met in elke layer een andere weight?"},
            "correct": {"en": "Your function is right in all {n} cases. Ten layers with weight 1 leave less than a millionth of the gradient.",
                        "nl": "Je functie klopt in alle {n} gevallen. Tien layers met weight 1 laten minder dan een miljoenste van de gradient over."},
        },
    ),
    "relu-sizes": FunctionBox(
        name="relu-sizes",
        function="relu_grad_sizes",
        start=_R_START,
        solution=_r(),
        visible=[([_W_A, _W_B], _m([1.0, -1.0]))],
        hidden=[([_W_B, _W_A, _W_B], _m([0.5, -1.0])), ([_W_B, _W_A, _W_A], _m([1.0, 2.0])), ([_W_C, _W_C], _m([1.0, 1.0, -1.0])),
                ([_W_C], _m([-1.0, 0.5, 0.5])), ([_W_D, _W_E], _m([1.0, -0.5, 2.0]))],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("no-mask", wrong=_r("g @ W"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the gradient goes back through the ReLU unchanged. Where `z` was 0 or below, the ReLU gave 0 whatever came in, so the gradient there is 0: `g * (z > 0)` first.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de gradient gaat ongewijzigd terug door de ReLU. Waar `z` 0 of lager was, gaf de ReLU 0 wat er ook binnenkwam, dus daar is de gradient 0: eerst `g * (z > 0)`."),
            Mistake("transposed", wrong=_r("(g * (z > 0)) @ W.T"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: on the way back, a layer `h @ W.T` gives its input the gradient times `W`, not times `W.T`, as in the backward pass of a whole layer.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: op de terugweg geeft een layer `h @ W.T` zijn input de gradient keer `W`, niet keer `W.T`, zoals in de backward pass van een hele layer."),
            Mistake("not-reversed", wrong=_r(order="range(len(Ws))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you go back through the layers in the order of the forward pass. The way back starts at the last layer: `reversed(range(len(Ws)))`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je gaat terug door de layers in de volgorde van de forward pass. De weg terug begint bij de laatste layer: `reversed(range(len(Ws)))`."),
            Mistake("no-abs", wrong=_r(size="np.mean(g)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: positive and negative gradients cancel in the mean. The size needs the mean of the absolute values: `np.mean(np.abs(g))`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: positieve en negatieve gradients heffen elkaar op in het gemiddelde. Voor de grootte neem je het gemiddelde van de absolute waarden: `np.mean(np.abs(g))`."),
            Mistake("one-z", wrong=_R_ONE_Z, state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: every layer keeps the gradient where the `z` of the last layer was above 0. Keep the `z` of each layer in a list in the forward pass, and use the `z` of each layer on the way back.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elke layer houdt de gradient waar de `z` van de laatste layer boven 0 lag. Bewaar in de forward pass de `z` van elke layer in een lijst, en gebruik op de terugweg de `z` van elke layer."),
            Mistake("size-first", wrong=_R_SIZE_FIRST,
                    en="`{call}` gives `{got}`, and it should give `{want}`: you add the size before the layer multiplies the gradient, so your list starts with the size of the ones at the top. Add it after the new `g`: each number is the size after a layer.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je zet de grootte in de lijst voordat de layer de gradient vermenigvuldigt, dus je lijst begint met de grootte van de enen bovenaan. Zet hem erin na de nieuwe `g`: elk getal is de grootte na een layer."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: the forward pass, keeping each `z`, then the way back from a gradient of ones, with the size after each layer.",
                      "nl": "Vervang de `...` door de body van de functie: de forward pass, waarbij je elke `z` bewaart, dan de weg terug vanaf een gradient van enen, met de grootte na elke layer."},
            "arguments": {"en": "The check calls `{call}`, with a list of weight matrices and one example, and the `def` line of your function has another number of names between its parentheses. Keep `(Ws, x)`.",
                          "nl": "De check roept `{call}` aan, met een lijst weight-matrices en één voorbeeld, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(Ws, x)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Work out the first number by hand: the ones, kept where the last `z` was above 0, times the last `W`, and their mean size.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken het eerste getal met de hand uit: de enen, bewaard waar de laatste `z` boven 0 lag, keer de laatste `W`, en hun gemiddelde grootte."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any number of layers and any width?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elk aantal layers en elke breedte?"},
            "correct": {"en": "Your function is right in all {n} cases: the size of the gradient after each layer, on the way back.",
                        "nl": "Je functie klopt in alle {n} gevallen: de grootte van de gradient na elke layer, op de terugweg."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("chain-gradient", _C_START, "empty", None),
    ("chain-gradient", _c(), "correct", None),
    ("chain-gradient", _C_HEAD + "    factors = []\n    h = x\n    for w in ws:\n        h = sigmoid(w * h)\n        factors.append(w * h * (1 - h))\n    return np.prod(factors)\n", "correct", None),
    ("chain-gradient", _c("sigmoid(z) * (1 - sigmoid(z))"), "wrong", "no-weight"),
    ("chain-gradient", _c("w * sigmoid(z)"), "wrong", "no-slope"),
    ("chain-gradient", _c(z="w * x"), "wrong", "same-input"),
    ("relu-sizes", _R_START, "empty", None),
    ("relu-sizes", _r(), "correct", None),
    ("relu-sizes", _r("(g * (np.maximum(z, 0) > 0)) @ W"), "correct", None),
    ("relu-sizes", _R_HEAD + "    h = x\n    zs = []\n    for W in Ws:\n        z = h @ W.T\n        zs.append(z)\n        h = np.maximum(z, 0)\n    g = np.ones(len(h))\n    sizes = []\n    for W, z in zip(reversed(Ws), reversed(zs)):\n        g = (g * (z > 0)) @ W\n        sizes.append(np.mean(np.abs(g)))\n    return sizes\n", "correct", None),
    ("relu-sizes", _r("g @ W"), "wrong", "no-mask"),
    ("relu-sizes", _r("(g * (z > 0)) @ W.T"), "wrong", "transposed"),
    ("relu-sizes", _r(order="range(len(Ws))"), "wrong", "not-reversed"),
    ("relu-sizes", _r(size="np.mean(g)"), "wrong", "no-abs"),
    ("relu-sizes", _R_ONE_Z, "almost", "one-z"),
    ("relu-sizes", _R_SIZE_FIRST, "wrong", "size-first"),
]
