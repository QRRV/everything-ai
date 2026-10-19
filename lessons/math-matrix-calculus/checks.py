# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson math-matrix-calculus: grad_W, the gradient of the mean squared error of two linear models at
once, −2/n · Rᵀ X with R = Y − X Wᵀ, and layer_grads, the gradients of Y = X @ W.T for W and for X, from the gradient
G for Y. And the picture: the shapes that decide which product and which transpose give each gradient."""
import html
import math
import re

import numpy as np

from everything_ai.check import FunctionBox, Mistake


class _Matrix(np.ndarray):
    """An array that a check message writes the way the lesson's code does: np.array([...]). The learner's code gets
    a plain copy, so that what it computes is an ordinary array."""

    def __repr__(self):
        a = np.asarray(self)
        return f"np.array({_plain(a.tolist())!r})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _plain(v):
    if isinstance(v, list):
        return [_plain(x) for x in v]
    return int(v) if float(v) == int(v) else float(v)


def _m(rows) -> np.ndarray:
    return np.array(rows, dtype=float).view(_Matrix)


def _num(x: float) -> str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    text = f"{round(x, 3) + 0.0:.3f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _shown(v) -> str:
    """An array or a list of arrays for the messages, as np.array([...]) with its numbers rounded to 3 decimals."""
    if isinstance(v, (list, tuple)) and v and all(isinstance(x, np.ndarray) for x in v):
        inner = ", ".join(_shown(x) for x in v)
        return f"[{inner}]" if isinstance(v, list) else f"({inner})"
    if isinstance(v, np.ndarray):
        def rows(a):
            return "[" + ", ".join(rows(x) for x in a) + "]" if a.ndim else _num(float(a))
        return f"np.array({rows(np.asarray(v, dtype=float))})"
    try:
        return _num(float(v))
    except (TypeError, ValueError):
        return repr(v)


def _close(got, want) -> bool:
    if isinstance(want, (list, tuple)):
        return (isinstance(got, (list, tuple)) and len(got) == len(want)
                and all(_close(g, w) for g, w in zip(got, want)))
    try:
        g = np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return False
    w = np.asarray(want, dtype=float)
    return g.shape == w.shape and bool(np.allclose(g, w, rtol=1e-7, atol=1e-9))


# --- the picture: the shapes of Y = X @ W and of the two gradients ------------------------------------------------------

def shapes_picture(labels: dict) -> str:
    """Boxes drawn to the size of their shape: X (4, 3), W (2, 3), Y and G (4, 2); then dW = Gᵀ @ X and dX = G @ W,
    each next to the matrix whose shape it has. X is data, W a parameter, Y computed inside the model, G and the
    two gradients in the gradient colour."""
    U = 15   # pixels per row or column
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']

    def box(x, y, rows, cols, name, shape, role):
        w, h = cols * U, rows * U
        fill = f"var(--ea-{role}-wash)"
        stroke = f"var(--ea-{role})"
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" style="fill:{fill};stroke:{stroke};stroke-width:1.5"/>')
        p.append(f'<text x="{x + w / 2}" y="{y + h / 2 + 5}" style="font-size:14px;font-weight:600;fill:var(--ea-ink);text-anchor:middle">{html.escape(name)}</text>')
        p.append(f'<text x="{x + w / 2}" y="{y + h + 15}" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(shape)}</text>')
        return x + w

    def sign(x, y, s):
        p.append(f'<text x="{x}" y="{y}" style="font-size:16px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(s)}</text>')

    rows = [(40, labels["forward"], [("X", 4, 3, "(4, 3)", "data"), ("@", None), ("Wᵀ", 3, 2, "(3, 2)", "param"),
                                     ("=", None), ("Y", 4, 2, "(4, 2)", "hidden")]),
            (156, labels["for_w"], [("Gᵀ", 2, 4, "(2, 4)", "grad"), ("@", None), ("X", 4, 3, "(4, 3)", "data"),
                                    ("=", None), ("dW", 2, 3, "(2, 3)", "grad")]),
            (272, labels["for_x"], [("G", 4, 2, "(4, 2)", "grad"), ("@", None), ("W", 2, 3, "(2, 3)", "param"),
                                    ("=", None), ("dX", 4, 3, "(4, 3)", "grad")])]
    for y, words, parts in rows:
        p.append(f'<text x="0" y="{y}" style="font-size:13px;fill:var(--ea-ink-2)">{html.escape(words)}</text>')
        x = 0
        for part in parts:
            if part[1] is None:
                sign(x + 14, y + 40, part[0])
                x += 28
            else:
                name, r, c, shape, role = part
                x = box(x, y + 10, r, c, name, shape, role) + 4
    W, H = 330, 372
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


# The lesson is done when these are correct. "grad-x" is the choice question in the lesson.
REQUIRED = ["grad-x", "grad-two", "layer-grads"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_SHAPES = Mistake(
    "shapes", error=lambda e: isinstance(e, ValueError) and ("mismatch in its core dimension" in str(e) or "not aligned" in str(e)),
    en="The product stops with an error: the inner numbers of the two shapes differ. `(n, k) @ (k, m)` needs the same `k` in the middle. Write down the shape of each matrix: which one needs a transpose?",
    nl="Het product stopt met een fout: de binnenste getallen van de twee shapes verschillen. `(n, k) @ (k, m)` heeft dezelfde `k` in het midden nodig. Schrijf de shape van elke matrix op: welke heeft een getransponeerde nodig?")

# --- exercise 2: the gradient of the mean squared error of two models at once ---------------------------------------

_W_HEAD = ("import numpy as np\n\n\ndef grad_W(X, W, Y):\n"
           "    R = Y - X @ W.T   # the residuals: one row per example, one column per model\n    n = len(Y)\n")
_W_START = ("import numpy as np\n\n\ndef grad_W(X, W, Y):\n"
            "    # the gradient for W of the mean squared error, with the residuals R = Y - X @ W.T\n    ...\n")


def _gw(body: str = "-2 / n * R.T @ X") -> str:
    return _W_HEAD + f"    return {body}\n"


_X1 = _m([[1, 1, 0], [0, 2, 1], [3, 0, 1], [1, 1, 2]])
_W1 = _m([[1, 0, 1], [0.5, 1, 0]])
_Y1 = _m([[1, 2], [2, 2], [4, 1], [2, 2]])
_X2 = _m([[1, 0], [0, 1], [1, 1]])
_W2 = _m([[0.5, -1], [2, 0], [1, 1]])
_Y2 = _m([[1, 0, 2], [0, 1, 1], [2, 2, 0]])

# --- exercise 3: the gradients of a layer -------------------------------------------------------------------------------

_L_HEAD = "import numpy as np\n\n\ndef layer_grads(X, W, G):\n    # Y = X @ W.T, and G is the gradient of the loss for Y\n"
_L_START = _L_HEAD + "    # give back [dW, dX]\n    ...\n"


def _lg(dw: str = "G.T @ X", dx: str = "G @ W") -> str:
    return _L_HEAD + f"    dW = {dw}\n    dX = {dx}\n    return [dW, dX]\n"


_A = _m([[1, 0, 2], [0, 1, 1], [2, 1, 0], [1, 1, 1]])
_B = _m([[1, 0, 1], [-1, 2, 0]])
_G = _m([[1, 0], [0, 1], [1, 1], [0.5, -1]])
_A2 = _m([[1, 2], [0, 1]])
_B2 = _m([[1, 2], [0, -1], [1, 0]])
_G2 = _m([[1, 0, 2], [0, 1, -1]])

EXERCISES = {
    "grad-two": FunctionBox(
        name="grad-two",
        function="grad_W",
        start=_W_START,
        solution=_gw(),
        visible=[(_X1, _W1, _Y1)],
        hidden=[(_X2, _W2, _Y2), (_X1, _m([[0, 0, 0], [0, 0, 0]]), _Y1), (_X2, _m([[1, 1], [1, 1], [1, 1]]), _m([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))],
        show=_shown,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            _SHAPES,
            Mistake("transposed", wrong=_gw("-2 / n * X.T @ R"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your gradient is the transpose, one row per feature. `W` has one row per model, so its gradient has one row per model too.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je gradient is de getransponeerde, met één rij per feature. `W` heeft één rij per model, dus zijn gradient ook."),
            Mistake("sign", wrong=_gw("2 / n * R.T @ X"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the sign is the other way round. A residual is the label minus the prediction, so it goes down when the prediction goes up: the derivative of r² for the prediction is −2r, with a minus.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: het teken is omgedraaid. Een residu is het label min de voorspelling, dus het daalt als de voorspelling stijgt: de afgeleide van r² naar de voorspelling is −2r, met een min."),
            Mistake("sum-not-mean", wrong=_gw("-2 * R.T @ X"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the gradient of the sum of the squared residuals. The loss of each model is their mean, so divide by `n` too.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is de gradient van de som van de gekwadrateerde residuen. De loss van elk model is hun gemiddelde, dus deel ook door `n`."),
            Mistake("no-two", wrong=_gw("-R.T @ X / n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the 2 from the square is missing. The derivative of r² is 2r.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de 2 van het kwadraat ontbreekt. De afgeleide van r² is 2r."),
            Mistake("residuals-only", wrong=_gw("-2 / n * R"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the gradient for the predictions, one row per example. The gradient for `W` has the shape of `W`, so it still has to go through `X`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is de gradient naar de voorspellingen, met één rij per voorbeeld. De gradient naar `W` heeft de shape van `W`, dus hij moet nog door `X` heen."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: the residuals, the number of examples, and the gradient for `W`, one product with a transpose and the number in front.",
                      "nl": "Vervang de `...` door de body van de functie: de residuen, het aantal voorbeelden, en de gradient naar `W`, één product met een getransponeerde en het getal ervoor."},
            "arguments": {"en": "The check calls `{call}`, with `X`, `W` and `Y`, and the `def` line of your function has another number of names between its parentheses. Keep `(X, W, Y)`.",
                          "nl": "De check roept `{call}` aan, met `X`, `W` en `Y`, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X, W, Y)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. What shape should the gradient have, and which product of `R` and `X` gives it?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Welke shape moet de gradient hebben, en welk product van `R` en `X` geeft die?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any number of examples, features and models?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elk aantal voorbeelden, features en modellen?"},
            "correct": {"en": "Your gradient is right in all {n} cases: one number per weight of each model, from the residuals of all examples at once.",
                        "nl": "Je gradient klopt in alle {n} gevallen: één getal per weight van elk model, uit de residuen van alle voorbeelden tegelijk."},
        },
    ),
    "layer-grads": FunctionBox(
        name="layer-grads",
        function="layer_grads",
        start=_L_START,
        solution=_lg(),
        visible=[(_A, _B, _G)],
        hidden=[(_A2, _B2, _G2), (_A, _B, _m([[0, 0], [0, 0], [0, 0], [0, 0]]))],
        show=_shown,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            _SHAPES,
            Mistake("dw-transposed", wrong=_lg(dw="X.T @ G"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your `dW` is the transpose of the gradient, with the shape of `W` swapped. `dW` has the shape of `W`: one row per output, one column per feature.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je `dW` is de getransponeerde van de gradient, met de shape van `W` omgedraaid. `dW` heeft de shape van `W`: één rij per output, één kolom per feature."),
            Mistake("dx-transposed", wrong=_lg(dx="W.T @ G.T"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your `dX` is the transpose of the gradient, with the shape of `X` swapped. `dX` has the shape of `X`: one row per example.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je `dX` is de getransponeerde van de gradient, met de shape van `X` omgedraaid. `dX` heeft de shape van `X`: één rij per voorbeeld."),
            Mistake("swapped", wrong=_lg(dw="G @ W", dx="G.T @ X"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `dW` and `dX` are swapped. Each gradient has the shape of its own matrix: `dW` that of `W`, `dX` that of `X`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `dW` en `dX` zijn omgewisseld. Elke gradient heeft de shape van zijn eigen matrix: `dW` die van `W`, `dX` die van `X`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: the gradients for `W` and for `X`, each a product of `G` and the other matrix with the shape of its own matrix, given back as `[dW, dX]`.",
                      "nl": "Vervang de `...` door de body van de functie: de gradients naar `W` en naar `X`, elk een product van `G` en de andere matrix met de shape van zijn eigen matrix, teruggegeven als `[dW, dX]`."},
            "arguments": {"en": "The check calls `{call}`, with `X`, `W` and `G`, and the `def` line of your function has another number of names between its parentheses. Keep `(X, W, G)`.",
                          "nl": "De check roept `{call}` aan, met `X`, `W` en `G`, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X, W, G)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does each gradient have the shape of its own matrix, and does the product give exactly that shape?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Heeft elke gradient de shape van zijn eigen matrix, en geeft het product precies die shape?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for other shapes too?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie ook voor andere shapes?"},
            "correct": {"en": "Your gradients are right in all {n} cases: `dW` with the shape of `W`, `dX` with the shape of `X`, each from `G` and the other matrix.",
                        "nl": "Je gradients kloppen in alle {n} gevallen: `dW` met de shape van `W`, `dX` met de shape van `X`, elk uit `G` en de andere matrix."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("grad-two", _W_START, "empty", None),
    ("grad-two", _gw(), "correct", None),
    ("grad-two", _gw("-2 * (R.T @ X) / len(X)"), "correct", None),
    ("grad-two", _gw("2 * (X @ W.T - Y).T @ X / n"), "correct", None),
    ("grad-two", _gw("-2 / n * R @ X"), "error", "shapes"),
    ("grad-two", _gw("-2 / n * X.T @ R"), "wrong", "transposed"),
    ("grad-two", _gw("2 / n * R.T @ X"), "wrong", "sign"),
    ("grad-two", _gw("-2 * R.T @ X"), "wrong", "sum-not-mean"),
    ("grad-two", _gw("-R.T @ X / n"), "wrong", "no-two"),
    ("grad-two", _gw("-2 / n * R"), "wrong", "residuals-only"),
    ("layer-grads", _L_START, "empty", None),
    ("layer-grads", _lg(), "correct", None),
    ("layer-grads", _lg(dw="(X.T @ G).T", dx="(W.T @ G.T).T"), "correct", None),
    ("layer-grads", _lg(dx="G @ W.T"), "error", "shapes"),
    ("layer-grads", _lg(dw="X.T @ G"), "wrong", "dw-transposed"),
    ("layer-grads", _lg(dx="W.T @ G.T"), "wrong", "dx-transposed"),
    ("layer-grads", _lg(dw="G @ W", dx="G.T @ X"), "wrong", "swapped"),
]
