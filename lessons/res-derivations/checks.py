# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-derivations: line2_sides and line3_sides, the two sides of the two steps He and others
(2015) give one sentence each in the derivation of their initialization, worked out from samples. And the picture: the
variance of the pre-activations layer after layer, as the derivation predicts it and as it comes out."""
import functools
import math
import re

import numpy as np

from everything_ai.check import FunctionBox, Mistake, run_code

WIDTH, LAYERS, BATCH, SEEDS = 256, 10, 1000, 8


def _layer_vars(var_w: float, seed: int) -> list:
    """Var[y_l] for l = 1 to LAYERS: y_0 standard normal, then y_l = W_l max(0, y_{l-1}) with weights N(0, var_w)."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=(BATCH, WIDTH))
    out = []
    for _ in range(LAYERS):
        W = rng.normal(0, math.sqrt(var_w), size=(WIDTH, WIDTH))
        y = np.maximum(0, y) @ W.T
        out.append(float(np.var(y)))
    return out


@functools.lru_cache(maxsize=4)
def measured(var_w: float) -> tuple:
    """Var[y_l] for every layer, the geometric mean over SEEDS runs."""
    runs = np.log(np.array([_layer_vars(var_w, s) for s in range(SEEDS)]))
    return tuple(float(v) for v in np.exp(runs.mean(axis=0)))


def predicted(var_w: float) -> list:
    """Var[y_l] by line (4): each layer multiplies the variance by n Var[w] / 2, from Var[y_0] = 1."""
    return [(WIDTH * var_w / 2) ** l for l in range(1, LAYERS + 1)]


@functools.lru_cache(maxsize=1)
def guess_value() -> float:
    """Var[z] for a layer of 256 inputs from a ReLU and weights of variance 1/256, from 20,000 samples."""
    rng = np.random.default_rng(5)
    X = np.maximum(0, rng.normal(size=(20000, WIDTH)))
    W = rng.normal(0, math.sqrt(1 / WIDTH), size=(WIDTH, WIDTH))
    return float(np.var(X @ W.T))


def layers_picture(labels: dict) -> str:
    """The variance of the pre-activations after each layer, on a log scale: the prediction of line (4) as a grey line,
    and what comes out as dots in the hidden colour, for Var[w] = 2/n and Var[w] = 1/n."""
    from everything_ai import chart
    p = chart.Panel((0.5, LAYERS + 0.5), (0.0005, 3), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    ylog=True, xticks=[1, 4, 7, 10])
    ls = list(range(1, LAYERS + 1))
    for var_w, (tx, ty) in ((2 / WIDTH, (6, 0.4)), (1 / WIDTH, (1.2, 0.006))):
        p.line(ls, predicted(var_w), "ink-3", 1.6)
        p.points(ls, list(measured(var_w)), "hidden")
        p.text(tx, ty, labels["two"] if var_w == 2 / WIDTH else labels["one"], "ink", "start")
    svg = chart.figure([p], alt=labels["alt"], width=330, height=260)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# --- exercise 1: line2_sides ---------------------------------------------------------------------------------------------------

_2_HEAD = ("import numpy as np\n\n\ndef line2_sides(ws, xs):\n    ws, xs = np.array(ws), np.array(xs)\n")
_2_LEFT = "    left = np.var(ws * xs)\n"
_2_RIGHT = "    right = np.var(ws) * np.mean(xs ** 2)\n"
_2_START = (_2_HEAD + "    # the left side of line (2): the variance of the products w x\n    left = ...\n"
            "    # the right side: the variance of w times the mean of the squares of x\n    right = ...\n    return left, right\n")


def _l2(left: str = _2_LEFT, right: str = _2_RIGHT) -> str:
    return _2_HEAD + left + right + "    return left, right\n"


_2_WRONG = {
    "var-x": _l2(right="    right = np.var(ws) * np.var(xs)\n"),
    "mean-squared": _l2(right="    right = np.var(ws) * np.mean(xs) ** 2\n"),
    "left-mean": _l2(left="    left = np.mean(ws * xs)\n"),
    "mean-w-squared": _l2(right="    right = np.mean(ws ** 2) * np.mean(xs ** 2)\n"),
}

# --- exercise 2: line3_sides ---------------------------------------------------------------------------------------------------

_3_HEAD = "import numpy as np\n\n\ndef line3_sides(ys):\n    ys = np.array(ys)\n"
_3_LEFT = "    left = np.mean(np.maximum(0, ys) ** 2)\n"
_3_RIGHT = "    right = 0.5 * np.var(ys)\n"
_3_START = (_3_HEAD + "    # the left side of line (3): the mean of the squares of x = max(0, y)\n    left = ...\n"
            "    # the right side: half the variance of y\n    right = ...\n    return left, right\n")


def _l3(left: str = _3_LEFT, right: str = _3_RIGHT) -> str:
    return _3_HEAD + left + right + "    return left, right\n"


_3_WRONG = {
    "var-relu": _l3(left="    left = np.var(np.maximum(0, ys))\n"),
    "square-first": _l3(left="    left = np.mean(np.maximum(0, ys ** 2))\n"),
    "no-half": _l3(right="    right = np.var(ys)\n"),
}


def _pair(v):
    try:
        if isinstance(v, (str, bytes, dict)) or len(v) != 2:
            return None
        return [float(v[0]), float(v[1])]
    except Exception:   # a set, an array of arrays and the like: not two numbers
        return None


def _close(got, want) -> bool:
    a, b = _pair(got), _pair(want)
    # abs_tol 1e-4: two numbers that differ by more always look different with 4 decimals, as {got} and {want} show them
    return a is not None and b is not None and all(math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-4) for x, y in zip(a, b))


def _num(x: float) -> str:
    return f"{round(x, 4) + 0.0:.4f}".rstrip("0").rstrip(".")


def _rounded(v) -> str:
    """Anything else a function gave back, as Python shows it, but with every number rounded to 4 decimals, also
    inside a tuple, a list, a dict or an array: never float noise or np.float64(...)."""
    if isinstance(v, np.ndarray):
        return f"array({_rounded(v.tolist())})"
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float):
        return _num(v)
    if isinstance(v, (list, tuple)):
        inner = ", ".join(_rounded(x) for x in v)
        return f"[{inner}]" if isinstance(v, list) else f"({inner}{',' if len(v) == 1 else ''})"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {_rounded(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (set, frozenset)):
        return "{" + ", ".join(_rounded(x) for x in v) + "}"
    return repr(v)


def _show(v):
    """Two sides as a pair rounded to 4 decimals, with a decimal comma and a semicolon in Dutch; anything else as code."""
    p = _pair(v)
    if p is None:
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            return {"en": _num(float(v)), "nl": _num(float(v)).replace(".", ",")}
        return f"`{_rounded(v)}`"
    a, b = _num(p[0]), _num(p[1])
    return {"en": f"({a}, {b})", "nl": f"({a.replace('.', ',')}; {b.replace('.', ',')})"}


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_DDOF = Mistake("ddof", matches=lambda got, text: re.search(r"^[^#\n]*ddof\s*=\s*1", text, re.MULTILINE) is not None,
                en="`{call}` gives {got}, and it should give {want}: your code has `ddof=1`, which divides by one less than the number of values, as some tools do. The sides of the line divide by the number of samples, as `np.mean` does, and `np.var` without `ddof`; on many samples the two hardly differ.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn: je code heeft `ddof=1`, dat door één minder deelt dan het aantal waarden, zoals sommige tools doen. De kanten van de regel delen door het aantal samples, zoals `np.mean` doet, en `np.var` zonder `ddof`; op veel samples scheelt dat bijna niets.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

EXERCISES = {
    "line2-sides": FunctionBox(
        name="line2-sides",
        function="line2_sides",
        start=_2_START,
        solution=_l2(),
        visible=[([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], [0.0, 0.0, 1.0, 1.0, 3.0, 3.0])],
        hidden=[([0.5, -0.5, 1.0, -1.0], [1.0, 3.0, 0.0, 2.0]), ([1.0, -1.0, 2.0, -2.0, 0.0], [0.5, 0.5, 1.5, 1.5, 4.0]),
                ([0.2, 0.4, -0.6], [1.0, 2.0, 3.0]), ([1.0, 2.0, 3.0], [2.0, 0.0, 1.0])],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("var-x", wrong=_2_WRONG["var-x"],
                    en="`{call}` gives {got}, and it should give {want}: your right side has the variance of x, as the formula of Initialization does. Line (2) has the mean of the squares, E[x²]: after a ReLU, x does not have mean 0, and then the two differ.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je rechterkant heeft de variantie van x, zoals de formule van Initialisatie. Regel (2) heeft het gemiddelde van de kwadraten, E[x²]: na een ReLU heeft x geen gemiddelde 0, en dan verschillen de twee."),
            Mistake("mean-squared", wrong=_2_WRONG["mean-squared"],
                    en="`{call}` gives {got}, and it should give {want}: your right side squares the mean of x, (E[x])². Line (2) takes the mean of the squares, E[x²]: first square, then average, `np.mean(xs ** 2)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je rechterkant kwadrateert het gemiddelde van x, (E[x])². Regel (2) neemt het gemiddelde van de kwadraten, E[x²]: eerst kwadrateren, dan middelen, `np.mean(xs ** 2)`."),
            Mistake("left-mean", wrong=_2_WRONG["left-mean"],
                    en="`{call}` gives {got}, and it should give {want}: your left side is the mean of the products, and line (2) is about their variance, Var[w x]: `np.var(ws * xs)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je linkerkant is het gemiddelde van de producten, en regel (2) gaat over hun variantie, Var[w x]: `np.var(ws * xs)`."),
            Mistake("mean-w-squared", wrong=_2_WRONG["mean-w-squared"],
                    en="`{call}` gives {got}, and it should give {want}: your right side has the mean of the squares of w, E[w²], where line (2) has Var[w]. The two are only equal when the weights have mean 0, and here they do not.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je rechterkant heeft het gemiddelde van de kwadraten van w, E[w²], waar regel (2) Var[w] heeft. Die twee zijn alleen gelijk als de weights gemiddelde 0 hebben, en hier hebben ze dat niet."),
            _DDOF,
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the variance of the products for the left side, and for the right side the variance of w times the mean of the squares of x.",
                      "nl": "Vervang de twee `...`: de variantie van de producten voor de linkerkant, en voor de rechterkant de variantie van w maal het gemiddelde van de kwadraten van x."},
            "arguments": {"en": "The check calls `{call}`, with the weights and the inputs, and the `def` line of your function has another number of names between its parentheses. Keep `(ws, xs)`.",
                          "nl": "De check roept `{call}` aan, met de weights en de inputs, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(ws, xs)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Left: the variance of the products w x. Right: the variance of w, times the mean of the squares of x.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Links: de variantie van de producten w x. Rechts: de variantie van w, maal het gemiddelde van de kwadraten van x."},
            "almost": {"en": "Right for the example, but there the two sides happen to be equal. `{call}` gives {got}, and it should give {want}. Does each side work out its own formula, and does your function give the left side first?",
                       "nl": "Goed voor het voorbeeld, maar daar zijn de twee kanten toevallig gelijk. `{call}` geeft {got}, en dat moet {want} zijn. Rekent elke kant zijn eigen formule uit, en geeft je functie de linkerkant eerst?"},
            "correct": {"en": "Both sides of line (2) are right in all {n} cases. On these few numbers they need not be equal; on many samples that meet the assumptions, they are.",
                        "nl": "Beide kanten van regel (2) kloppen in alle {n} gevallen. Op deze paar getallen hoeven ze niet gelijk te zijn; op veel samples die aan de aannames voldoen wel."},
        },
    ),
    "line3-sides": FunctionBox(
        name="line3-sides",
        function="line3_sides",
        start=_3_START,
        solution=_l3(),
        visible=[([-2.0, -1.0, 0.0, 1.0, 2.0],)],
        hidden=[([1.0, 2.0, 3.0],), ([-3.0, 0.5, 0.5, 2.0],), ([-1.0, -1.0],), ([0.0, 4.0, -4.0, 1.0, -1.0, 2.0],)],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("var-relu", wrong=_3_WRONG["var-relu"],
                    en="`{call}` gives {got}, and it should give {want}: your left side is the variance of max(0, y). Line (3) is about E[x²], the mean of the squares of x = max(0, y), without taking off the mean: `np.mean(np.maximum(0, ys) ** 2)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je linkerkant is de variantie van max(0, y). Regel (3) gaat over E[x²], het gemiddelde van de kwadraten van x = max(0, y), zonder het gemiddelde eraf te halen: `np.mean(np.maximum(0, ys) ** 2)`."),
            Mistake("square-first", wrong=_3_WRONG["square-first"],
                    en="`{call}` gives {got}, and it should give {want}: your left side is the mean of the squares of y itself, E[y²]. A ReLU after the square does nothing, because a square is never below 0. First x = max(0, y), then square it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je linkerkant is het gemiddelde van de kwadraten van y zelf, E[y²]. Een ReLU na het kwadraat doet niets, want een kwadraat is nooit negatief. Eerst x = max(0, y), dan het kwadraat."),
            Mistake("no-half", wrong=_3_WRONG["no-half"],
                    en="`{call}` gives {got}, and it should give {want}: your right side is the whole variance of y. The ReLU keeps half of a symmetric y: the right side of line (3) is half of it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je rechterkant is de hele variantie van y. De ReLU houdt de helft van een symmetrische y over: de rechterkant van regel (3) is de helft ervan."),
            _DDOF,
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the mean of the squares of max(0, y) for the left side, and half the variance of y for the right side.",
                      "nl": "Vervang de twee `...`: het gemiddelde van de kwadraten van max(0, y) voor de linkerkant, en de helft van de variantie van y voor de rechterkant."},
            "arguments": {"en": "The check calls `{call}`, with the values y before the ReLU, and the `def` line of your function has another number of names between its parentheses. Keep `(ys)`.",
                          "nl": "De check roept `{call}` aan, met de waarden y vóór de ReLU, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(ys)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Left: x = max(0, y), squared, then averaged. Right: half the variance of y.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Links: x = max(0, y), gekwadrateerd, dan gemiddeld. Rechts: de helft van de variantie van y."},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does it work for lists that are not symmetric around 0?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt het ook voor lijsten die niet symmetrisch rond 0 liggen?"},
            "correct": {"en": "Both sides of line (3) are right in all {n} cases. For (1, 2, 3) they are far apart: that y is not symmetric around 0.",
                        "nl": "Beide kanten van regel (3) kloppen in alle {n} gevallen. Voor (1, 2, 3) liggen ze ver uit elkaar: die y ligt niet symmetrisch rond 0."},
        },
    ),
}

# The lesson is done when these are correct. "assumption" is the choice question.
REQUIRED = ["line2-sides", "line3-sides", "assumption"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("line2-sides", _2_START, "empty", None),
    ("line2-sides", _l2(), "correct", None),
    ("line2-sides", _l2(left="    left = np.mean((ws * xs) ** 2) - np.mean(ws * xs) ** 2\n",
                        right="    right = np.std(ws) ** 2 * np.mean(xs * xs)\n"), "correct", None),
    ("line2-sides", _l2(left="    left = round(float(np.var(ws * xs)), 4)\n", right="    right = round(float(np.var(ws) * np.mean(xs ** 2)), 4)\n"), "correct", None),
    ("line2-sides", _l2(right="    right = ...\n"), "wrong", "dots-left"),
    ("line2-sides", _l2(left="    left = np.var(ws * xs, ddof=1)\n", right="    right = np.var(ws, ddof=1) * np.mean(xs ** 2)\n"), "wrong", "ddof"),
    ("line2-sides", _2_HEAD + _2_LEFT + _2_RIGHT + "    return right, left\n", "almost", None),
    ("line2-sides", _2_HEAD + "    return {'left': np.var(ws * xs), 'right': np.var(ws) * np.mean(xs ** 2)}\n", "wrong", None),
] + [("line2-sides", code, "wrong", key) for key, code in _2_WRONG.items()] + [
    ("line3-sides", _3_START, "empty", None),
    ("line3-sides", _l3(), "correct", None),
    ("line3-sides", _l3(left="    x = np.where(ys > 0, ys, 0)\n    left = (x ** 2).mean()\n", right="    right = np.var(ys) / 2\n"), "correct", None),
    ("line3-sides", _l3(left="    left = ...\n"), "wrong", "dots-left"),
    ("line3-sides", _l3(right="    right = 0.5 * np.var(ys, ddof=1)\n"), "wrong", "ddof"),
    ("line3-sides", _l3(left="    left = np.mean(ys ** 2)\n"), "wrong", "square-first"),
] + [("line3-sides", code, "wrong", key) for key, code in _3_WRONG.items()]


def _functions(text2, text3):
    """The learner's two functions where they are right, else the course's."""
    def course(src, name):
        return run_code(src).names[name]
    f2 = run_code(text2).names.get("line2_sides") if text2 else None
    f3 = run_code(text3).names.get("line3_sides") if text3 else None
    return (f2 or course(_l2(), "line2_sides"), f3 or course(_l3(), "line3_sides"), f2 is not None, f3 is not None)


def demo(text2, text3, samples: int = 200_000) -> dict:
    """Both sides of lines (2) and (3) on many samples: once with the derivation's assumptions, once with one broken
    (weights with mean 1; y with mean 0.5)."""
    f2, f3, yours2, yours3 = _functions(text2, text3)
    rng = np.random.default_rng(0)
    ys = rng.normal(size=samples)
    xs = np.maximum(0, rng.normal(size=samples))
    ws = rng.normal(0, 1, size=samples)
    rows = {
        "2-holds": f2(ws.tolist(), xs.tolist()),
        "2-broken": f2((ws + 1).tolist(), xs.tolist()),
        "3-holds": f3(ys.tolist()),
        "3-broken": f3((ys + 0.5).tolist()),
    }
    return {"rows": {k: (float(v[0]), float(v[1])) for k, v in rows.items()}, "yours": (yours2, yours3)}
