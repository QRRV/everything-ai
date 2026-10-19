# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-gradient-check: numerical_grad, the gradient of a function estimated with central
differences, one parameter at a time, and relative_error, how far a gradient from backprop lies from that estimate.
And the picture: how far the two kinds of estimate of the slope of e to the power x at 1 lie off, for each step h."""
import math
import re

import numpy as np

from everything_ai.check import FunctionBox, Mistake


# --- the picture: the error of a forward and a central difference, for every h ------------------------------------------------

def errors(h: float) -> tuple[float, float]:
    """The relative error of the forward and the central difference of e^x at x = 1, with step h."""
    true = math.e
    forward = (math.exp(1 + h) - math.exp(1)) / h
    central = (math.exp(1 + h) - math.exp(1 - h)) / (2 * h)
    return abs(forward - true) / true, abs(central - true) / true


def h_picture(labels: dict) -> str:
    from everything_ai import chart
    hs = [10 ** (-k / 8) for k in range(8, 97)]   # from 0.1 down to 10 to the power -12
    fw = [max(errors(h)[0], 1e-13) for h in hs]
    ce = [max(errors(h)[1], 1e-13) for h in hs]
    panel = chart.Panel((1e-12, 0.1), (1e-12, 1), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], xlog=True, ylog=True)
    panel.line(hs, fw, "ink-3", 2.0, dashed=True)
    panel.line(hs, ce, "ink", 2.4)
    panel.text(0.05, 0.12, labels["forward"], "ink-2", "end")
    panel.text(0.08, 1e-9, labels["central"], "ink", "end")
    panel.text(3e-11, 0.02, labels["rounding"], "ink-2", "start")
    return chart.figure([panel], alt=labels["alt"], width=560, height=310)


# The lesson is done when these are correct. "which-bug" is the choice question in the lesson.
REQUIRED = ["which-bug", "numerical-grad", "relative-error"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


class _Vector(np.ndarray):
    """An array that a check message writes the way the lesson's code does: np.array([...]). The learner's code gets a
    plain copy."""

    def __repr__(self):
        return f"np.array({[float(v) for v in np.asarray(self).tolist()]!r})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _v(values) -> np.ndarray:
    return np.array(values, dtype=float).view(_Vector)


class _Fn:
    """A function the check passes in, written in a message as the lambda it is."""

    def __init__(self, text: str):
        self.text = text
        self.fn = eval(text, {"np": np})

    def __call__(self, w):
        return self.fn(w)

    def __repr__(self):
        return self.text

    def __deepcopy__(self, memo):
        return self


def _plain(x) -> str:
    text = f"{round(float(x), 4) + 0.0:.4f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _num(v) -> str:
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return repr(v)
    if a.ndim == 0:
        return _plain(a)
    if not np.all(np.isfinite(a)):
        return "[" + ", ".join("nan" if math.isnan(x) else ("inf" if x > 0 else "-inf") if math.isinf(x) else _plain(x) for x in a.ravel()) + "]"
    return "[" + ", ".join(_plain(x) for x in a.ravel()) + "]"


def _close(got, want, rtol: float = 1e-6) -> bool:
    try:
        g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    except (TypeError, ValueError):
        return False
    return g.shape == w.shape and bool(np.allclose(g, w, rtol=rtol, atol=1e-8))


def _one_number(got, text) -> bool:
    return np.ndim(got) == 0 and isinstance(got, (int, float, np.floating, np.integer)) and not isinstance(got, bool)


# --- exercise 2: the numerical gradient -------------------------------------------------------------------------------------

_N_HEAD = "import numpy as np\n\n\ndef numerical_grad(f, w, h):\n"
_N_START = _N_HEAD + "    # the slope of f for each number of w, by central differences, as an array\n    ...\n"


def _n(slope: str = "(f(w + step) - f(w - step)) / (2 * h)", step: str = "        step = np.zeros(len(w))\n") -> str:
    return (_N_HEAD + "    grad = np.zeros(len(w))\n    for j in range(len(w)):\n" + step
            + "        step[j] = h\n" + f"        grad[j] = {slope}\n" + "    return grad\n")


_IN_PLACE = (_N_HEAD + "    grad = np.zeros(len(w))\n    for j in range(len(w)):\n        w[j] = w[j] + h\n        up = f(w)\n"
             "        w[j] = w[j] - 2 * h\n        down = f(w)\n        grad[j] = (up - down) / (2 * h)\n    return grad\n")

_SQUARES = _Fn("lambda w: np.sum(w ** 2)")
_SUM_SQUARED = _Fn("lambda w: np.sum(w) ** 2")
_CUBES = _Fn("lambda w: np.sum(w ** 3)")
_PRODUCT = _Fn("lambda w: w[0] * w[1] * w[2]")

# --- exercise 3: the relative error -----------------------------------------------------------------------------------------

_R_HEAD = "import numpy as np\n\n\ndef relative_error(a, n):\n"
_R_START = _R_HEAD + "    # for each parameter: how far the gradient a from backprop lies from the estimate n, relative to their size\n    ...\n"


def _r(body: str = "np.abs(a - n) / np.maximum(np.abs(a), np.abs(n))") -> str:
    return _R_HEAD + f"    return {body}\n"


EXERCISES = {
    "numerical-grad": FunctionBox(
        name="numerical-grad",
        function="numerical_grad",
        start=_N_START,
        solution=_n(),
        visible=[(_SQUARES, _v([1, -2]), 0.001)],
        hidden=[(_CUBES, _v([2, 1, -1]), 0.001), (_SUM_SQUARED, _v([1, 2, 3]), 0.0001), (_PRODUCT, _v([1, 2, 3]), 0.001),
                (_CUBES, _v([0.5]), 0.01)],
        show=_num,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("forward", wrong=_n("(f(w + step) - f(w)) / h"), state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: you nudge only up, as in the derivative as a slope. That is a little off wherever the curve bends. Nudge down as well, and divide by `2 * h`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je zet alleen een stapje omhoog, zoals bij de afgeleide als helling. Dat zit er een beetje naast waar de kromme buigt. Zet ook een stapje omlaag, en deel door `2 * h`."),
            Mistake("one-h", wrong=_n("(f(w + step) - f(w - step)) / h"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: twice the slope. From `w - h` to `w + h` is a distance of `2 * h`: divide by that.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: twee keer de helling. Van `w - h` tot `w + h` is een afstand van `2 * h`: deel daardoor."),
            Mistake("step-kept", wrong=_n(step="        step = np.zeros(len(w)) if j == 0 else step\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: from the second number on, your nudge still holds the `h` of the numbers before it, so you nudge several at once. Start each number with a fresh array of zeros.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: vanaf het tweede getal bevat je stapje nog de `h` van de getallen ervoor, dus je verschuift er meer tegelijk. Begin elk getal met een nieuwe array van nullen."),
            Mistake("in-place", wrong=_IN_PLACE, state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: you change `w` itself and leave the last nudge in it, so each next number is measured at a point that has moved. Nudge a copy, such as `w + step`, and leave `w` as it was.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je verandert `w` zelf en laat het laatste stapje erin staan, dus elk volgend getal wordt gemeten op een punt dat verschoven is. Verschuif een kopie, zoals `w + step`, en laat `w` zoals hij was."),
            Mistake("one-number", matches=_one_number,
                    en="`{call}` gives `{got}`, one number, and it should give `{want}`: one slope for each number of `w`, as an array.",
                    nl="`{call}` geeft `{got}`, één getal, en dat moet `{want}` zijn: één helling voor elk getal van `w`, als array."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: for each number of `w`, nudge it up and down by `h`, and the slope between.",
                      "nl": "Vervang de `...` door de body van de functie: verschuif elk getal van `w` omhoog en omlaag met `h`, en neem de helling daartussen."},
            "arguments": {"en": "The check calls `{call}`, with a function, an array and a step, and the `def` line of your function has another number of names between its parentheses. Keep `(f, w, h)`.",
                          "nl": "De check roept `{call}` aan, met een functie, een array en een stap, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(f, w, h)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Work out the slope of the first number by hand: what is `f` a little above it and a little below it?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken de helling van het eerste getal met de hand uit: wat is `f` er iets boven en iets onder?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work when the numbers of `w` work together in `f`, and for every `h`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie ook als de getallen van `w` in `f` samenwerken, en voor elke `h`?"},
            "correct": {"en": "Your function is right in all {n} cases: one slope for each number, each nudged up and down on its own.",
                        "nl": "Je functie klopt in alle {n} gevallen: één helling voor elk getal, elk apart omhoog en omlaag verschoven."},
        },
    ),
    "relative-error": FunctionBox(
        name="relative-error",
        function="relative_error",
        start=_R_START,
        solution=_r(),
        visible=[(_v([2, -3, 0.4]), _v([2, -4, 0.5]))],
        hidden=[(_v([1000, 0.001]), _v([1001, 0.002])), (_v([-1, -2]), _v([-1.5, 2])), (_v([4, -0.02]), _v([3, -0.01])),
                (_v([0.3]), _v([0.3]))],
        show=_num,
        # a relative error is read by its order of size: a tiny number added below the line, against dividing by 0, is fine
        compare=lambda got, want: _close(got, want, rtol=1e-4),
        first=[_DOTS],
        mistakes=[
            Mistake("absolute", wrong=_r("np.abs(a - n)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the difference itself. Divide it by the larger of the two sizes, so that a gradient of 1000 and one of 0.001 are judged alike.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is het verschil zelf. Deel het door de grootste van de twee groottes, zodat een gradient van 1000 en een van 0,001 op dezelfde manier beoordeeld worden."),
            Mistake("by-a", wrong=_r("np.abs(a - n) / np.abs(a)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you divide by the size of `a` only. Divide by the larger of `|a|` and `|n|`, so that it does not matter which of the two is the gradient.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je deelt alleen door de grootte van `a`. Deel door de grootste van `|a|` en `|n|`, zodat het niet uitmaakt welke van de twee de gradient is."),
            Mistake("by-n", wrong=_r("np.abs(a - n) / np.abs(n)"), state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: you divide by the size of `n` only. Divide by the larger of `|a|` and `|n|`, so that it does not matter which of the two is the estimate.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je deelt alleen door de grootte van `n`. Deel door de grootste van `|a|` en `|n|`, zodat het niet uitmaakt welke van de twee de schatting is."),
            Mistake("sign", wrong=_r("(a - n) / np.maximum(np.abs(a), np.abs(n))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: an error below 0 is still an error. Take the size of the difference, with `np.abs`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: een fout onder 0 is nog steeds een fout. Neem de grootte van het verschil, met `np.abs`."),
            Mistake("max-no-abs", wrong=_r("np.abs(a - n) / np.maximum(a, n)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: for negative gradients, the larger of `a` and `n` is not the larger size. Take the larger of `np.abs(a)` and `np.abs(n)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: bij negatieve gradients is de grootste van `a` en `n` niet de grootste grootte. Neem de grootste van `np.abs(a)` en `np.abs(n)`."),
            Mistake("python-max", error=lambda e: isinstance(e, ValueError) and "truth value" in str(e),
                    en="With `{call}`, Python stops: your code asks which of two whole arrays is larger, with Python's `max` or an `if`, and Python cannot answer that with one yes or no. `np.maximum(x, y)` takes the larger of two arrays, place by place.",
                    nl="Met `{call}` stopt Python: je code vraagt welke van twee hele arrays groter is, met `max` van Python of een `if`, en daar kan Python niet met één ja of nee op antwoorden. `np.maximum(x, y)` neemt plek voor plek de grootste van twee arrays."),
            Mistake("one-number", matches=_one_number,
                    en="`{call}` gives `{got}`, one number, and it should give `{want}`: one relative error for each parameter, as an array, so that you see which one is off.",
                    nl="`{call}` geeft `{got}`, één getal, en dat moet `{want}` zijn: één relatieve fout per parameter, als array, zodat je ziet welke ernaast zit."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: the size of the difference, divided by the larger size of the two.",
                      "nl": "Vervang de `...` door de body van de functie: de grootte van het verschil, gedeeld door de grootste grootte van de twee."},
            "arguments": {"en": "The check calls `{call}`, with two arrays, and the `def` line of your function has another number of names between its parentheses. Keep `(a, n)`.",
                          "nl": "De check roept `{call}` aan, met twee arrays, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(a, n)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Work out the second number by hand: the difference, and the larger of the two sizes.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken het tweede getal met de hand uit: het verschil, en de grootste van de twee groottes."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for large and small gradients, and for negative ones?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor grote en kleine gradients, en voor negatieve?"},
            "correct": {"en": "Your function is right in all {n} cases: the difference relative to the size, for each parameter, whatever its scale or sign.",
                        "nl": "Je functie klopt in alle {n} gevallen: het verschil ten opzichte van de grootte, per parameter, welke schaal of welk teken het ook heeft."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("numerical-grad", _N_START, "empty", None),
    ("numerical-grad", _n(), "correct", None),
    ("numerical-grad", _N_HEAD + "    grad = []\n    for j in range(len(w)):\n        up = w.copy()\n        up[j] += h\n        down = w.copy()\n        down[j] -= h\n        grad.append((f(up) - f(down)) / (2 * h))\n    return np.array(grad)\n", "correct", None),
    ("numerical-grad", _N_HEAD + "    grad = np.zeros(len(w))\n    for j in range(len(w)):\n        old = w[j]\n        w[j] = old + h\n        up = f(w)\n        w[j] = old - h\n        down = f(w)\n        w[j] = old\n        grad[j] = (up - down) / (2 * h)\n    return grad\n", "correct", None),
    ("numerical-grad", _n("(f(w + step) - f(w)) / h"), "almost", "forward"),
    ("numerical-grad", _n("(f(w + step) - f(w - step)) / h"), "wrong", "one-h"),
    ("numerical-grad", _n(step="        step = np.zeros(len(w)) if j == 0 else step\n"), "wrong", "step-kept"),
    ("numerical-grad", _IN_PLACE, "almost", "in-place"),
    ("numerical-grad", _N_HEAD + "    step = np.zeros(len(w))\n    step[0] = h\n    return (f(w + step) - f(w - step)) / (2 * h)\n", "wrong", "one-number"),
    ("relative-error", _R_START, "empty", None),
    ("relative-error", _r(), "correct", None),
    ("relative-error", _R_HEAD + "    out = []\n    for i in range(len(a)):\n        out.append(abs(a[i] - n[i]) / max(abs(a[i]), abs(n[i])))\n    return np.array(out)\n", "correct", None),
    ("relative-error", _r("np.abs(a - n)"), "wrong", "absolute"),
    ("relative-error", _r("np.abs(a - n) / np.abs(a)"), "wrong", "by-a"),
    ("relative-error", _r("np.abs(a - n) / np.abs(n)"), "almost", "by-n"),
    ("relative-error", _r("abs(a - n) / max(abs(a), abs(n))"), "error", "python-max"),
    ("relative-error", _r("np.abs(a - n) / np.minimum(np.abs(a), np.abs(n))"), "wrong", None),
    ("relative-error", _r("np.abs(a - n) / (np.maximum(np.abs(a), np.abs(n)) + 1e-8)"), "correct", None),
    ("relative-error", _r("(a - n) / np.maximum(np.abs(a), np.abs(n))"), "wrong", "sign"),
    ("relative-error", _r("np.abs(a - n) / np.maximum(a, n)"), "wrong", "max-no-abs"),
    ("relative-error", _r("np.max(np.abs(a - n) / np.maximum(np.abs(a), np.abs(n)))"), "wrong", "one-number"),
]
