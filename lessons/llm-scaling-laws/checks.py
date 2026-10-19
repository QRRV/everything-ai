# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-scaling-laws: fit_power_law, a power law fitted to the part of the loss above a floor."""
import math
import re

from everything_ai.check import FunctionBox, Mistake

# Chinchilla's fit (Hoffmann and others 2022, approach 3): L(N, D) = E + A / N**ALPHA + B / D**BETA, in nats per token.
E, A, ALPHA = 1.69, 406.4, 0.34


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def chinchilla(n: float) -> float:
    """The loss of a model of n parameters by Chinchilla's fit, with endless data."""
    return E + A / n ** ALPHA


def floor_picture(labels: dict) -> str:
    """Chinchilla's loss against the number of parameters on a log-log plot, with endless data: it bends towards the
    floor E. Three small runs, a straight line through them and on to 100 billion parameters, and the floor. The loss is
    purple, the prediction of the line the prediction colour, the floor ink."""
    from everything_ai import chart
    p = chart.Panel((1e7, 1e11), (0.9, 4.2), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xlog=True, ylog=True, yticks=[1, 2, 3, 4])
    ns = [10 ** (7 + i / 20) for i in range(81)]
    p.line(ns, [chinchilla(n) for n in ns], "loss", 2.4)
    small = [1e7, 10 ** 7.5, 1e8]
    k = math.log10(chinchilla(1e8) / chinchilla(1e7))   # the slope of the line through the first and last small run
    p.line([1e7, 1e11], [chinchilla(1e7), chinchilla(1e7) * 10 ** (4 * k)], "pred", 1.8, dashed=True)
    p.line([1e7, 1e11], [E, E], "ink-3", 1.2, dashed=True)
    p.points(small, [chinchilla(n) for n in small], "loss", r=4.5)
    p.text(2.2e7, 1.58, labels["floor"], "ink-2", "start")
    p.text(1.2e9, 2.25, labels["curve"], "loss", "start")
    p.text(1.2e9, 1.07, labels["line"], "pred", "start")
    svg = chart.figure([p], alt=labels["alt"], width=330, height=280)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# --- exercise 1: fit_power_law -------------------------------------------------------------------------------------------------

_HEAD = ("import numpy as np\n\n\ndef fit_power_law(x, loss, floor):\n"
         "    # x: the sizes, such as parameters or tokens; loss: the loss at each size; floor: the loss no model gets below\n")
_LOGS = "    log_x = np.log10(x)\n    log_rest = np.log10(loss - floor)\n"
_FIT = "    k, b = np.polyfit(log_x, log_rest, 1)\n"
_A = "    a = 10 ** b\n"
_TAIL = "    return a, k\n"
_START = (_HEAD + "    # the logs of x and of the part of the loss above the floor\n    ...\n"
          "    # a straight line through them: its slope is k, its intercept the log of a\n    ...\n" + _TAIL)


def _f(logs: str = _LOGS, fit: str = _FIT, a: str = _A, tail: str = _TAIL) -> str:
    return _HEAD + logs + fit + a + tail


class _Named:
    """An array of a test case, written in the check's messages the way the lesson writes it."""

    def __init__(self, shown: str):
        self.shown = shown

    def __repr__(self):
        return self.shown

    def __deepcopy__(self, memo):
        import numpy as np
        return eval(self.shown, {"np": np})


def _case(x: str, loss: str, floor: float) -> tuple:
    return (_Named(f"np.array({x})"), _Named(f"np.array({loss})"), floor)


def _pair(v):
    try:
        if isinstance(v, (str, bytes)) or len(v) != 2:
            return None
        a, k = (float(t) for t in v)
        return a, k
    except (TypeError, ValueError):
        return None


def _close(got, want) -> bool:
    g, w = _pair(got), _pair(want)
    return g is not None and all(math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-9) for x, y in zip(g, w))


def _sig(v: float) -> str:
    """3 significant digits, never in e-notation: 128, 0.301, 406, 0.00123."""
    if v != v or math.isinf(v):
        return str(v)
    if abs(v) < 1e-9:
        return "0"
    digits = max(0, 2 - math.floor(math.log10(abs(v))))
    text = f"{round(v, digits):.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _show(v) -> str:
    """A pair as (a, k), and any other number or row of numbers without float noise."""
    p = _pair(v)
    if p is not None:
        return f"({_sig(p[0])}, {_sig(p[1])})"
    try:
        return _sig(float(v))
    except (TypeError, ValueError):
        pass
    try:
        if not isinstance(v, (str, bytes)):
            return "(" + ", ".join(_sig(float(t)) for t in v) + ")"
    except (TypeError, ValueError):
        pass
    return repr(v)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


EXERCISES = {
    "fit-power-law": FunctionBox(
        name="fit-power-law",
        function="fit_power_law",
        start=_START,
        solution=_f(),
        visible=[_case("[1e6, 1e7, 1e8, 1e9]", "[3.7, 2.7, 2.2, 1.95]", 1.7)],
        hidden=[_case("[1, 2, 4, 8, 16]", "[3.1, 2.62, 2.33, 2.17, 2.07]", 2.0), _case("[10, 100, 1000]", "[5.0, 2.5, 1.25]", 0),
                _case("[1e7, 1e8, 1e9, 1e10]", "[3.384, 2.464, 2.044, 1.852]", 1.69)],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("rounded", matches=lambda got, text: "round(" in text, state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: they may look the same, but you round a or k, and the check compares more digits than it shows. Give back the numbers themselves, without `round`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: ze lijken misschien gelijk, maar je rondt a of k af, en de check vergelijkt meer cijfers dan hij laat zien. Geef de getallen zelf terug, zonder `round`."),
            Mistake("no-floor", wrong=_f(logs="    log_x = np.log10(x)\n    log_rest = np.log10(loss)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your line goes through the logs of the whole loss. Only the part above the floor is a power law: take the log of `loss - floor`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je lijn gaat door de logs van de hele loss. Alleen het deel boven de bodem is een machtswet: neem de log van `loss - floor`."),
            Mistake("swapped", wrong=_f(fit="    b, k = np.polyfit(log_x, log_rest, 1)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you take the slope for the intercept. `np.polyfit` with degree 1 gives the slope first and then the intercept: `k, b = np.polyfit(...)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je neemt de helling voor het snijpunt. `np.polyfit` met graad 1 geeft eerst de helling en dan het snijpunt: `k, b = np.polyfit(...)`."),
            Mistake("log-a", wrong=_f(a="    a = b\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your k is right, but your a is the intercept itself, the log of a. Turn it back: `a = 10 ** b`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je k klopt, maar je a is het snijpunt zelf, de log van a. Reken hem terug: `a = 10 ** b`."),
            Mistake("ln-mix", wrong=_f(logs="    log_x = np.log(x)\n    log_rest = np.log(loss - floor)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your k is right, but your a is not. You take natural logs with `np.log` and then 10 to the power of the intercept. Use one base both ways: `np.log10` with `10 **`, or `np.log` with `np.exp`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je k klopt, maar je a niet. Je neemt natuurlijke logs met `np.log` en dan 10 tot de macht het snijpunt. Gebruik twee keer hetzelfde grondtal: `np.log10` met `10 **`, of `np.log` met `np.exp`."),
            Mistake("axes", wrong=_f(fit="    k, b = np.polyfit(log_rest, log_x, 1)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your line has the sizes up the side and the loss along the bottom. `np.polyfit` takes the numbers along the bottom first: `np.polyfit(log_x, log_rest, 1)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je lijn heeft de groottes langs de zijkant en de loss langs de onderkant. `np.polyfit` krijgt eerst de getallen langs de onderkant: `np.polyfit(log_x, log_rest, 1)`."),
            Mistake("two-points", wrong=_f(fit="    k = (log_rest[-1] - log_rest[0]) / (log_x[-1] - log_x[0])\n    b = log_rest[0] - k * log_x[0]\n"), state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: your line goes through the first and the last point only. Measured losses wobble a little; a fit through all points, with `np.polyfit`, evens that out.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je lijn gaat alleen door het eerste en het laatste punt. Gemeten losses wiebelen een beetje; een fit door alle punten, met `np.polyfit`, middelt dat uit."),
            Mistake("order", wrong=_f(tail="    return k, a\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the same two numbers, the other way round. Give back `a, k`: first a, then the exponent.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dezelfde twee getallen, andersom. Geef `a, k` terug: eerst a, dan de exponent."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: the logs of `x` and of the loss above the floor, and a straight line through them.",
                      "nl": "Vervang de twee `...`: de logs van `x` en van de loss boven de bodem, en een rechte lijn erdoorheen."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the sizes, the losses and the floor, and the `def` line of your function has another number of names between its parentheses. Keep `(x, loss, floor)`.",
                          "nl": "De check roept `{call}` aan, met de groottes, de losses en de bodem, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x, loss, floor)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you fit a straight line through the log of `x` and the log of the loss above the floor, and give back 10 to the power of its intercept and its slope?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Fit je een rechte lijn door de log van `x` en de log van de loss boven de bodem, en geef je 10 tot de macht zijn snijpunt en zijn helling terug?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your fit work with a floor of 0, and with losses that wobble a little?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je fit ook met een bodem van 0, en met losses die een beetje wiebelen?"},
            "correct": {"en": "Your fit finds a and k in all {n} cases. On the losses of Chinchilla's law it gives back their law: a close to their 406.4, and k close to their −0.34.",
                        "nl": "Je fit vindt a en k in alle {n} gevallen. Op de losses van de wet van Chinchilla geeft hij hun wet terug: a dicht bij hun 406,4, en k dicht bij hun −0,34."},
        },
    ),
}

# The lesson is done when these are correct. "which-runs" is the choice question, "why-floor" the open question.
REQUIRED = ["fit-power-law", "which-runs", "why-floor"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("fit-power-law", _START, "empty", None),
    ("fit-power-law", _f(), "correct", None),
    ("fit-power-law", _f(logs="    log_x = np.log(x)\n    log_rest = np.log(loss - floor)\n", a="    a = np.exp(b)\n"), "correct", None),
    ("fit-power-law", _HEAD + "    w = np.polyfit(np.log10(x), np.log10(loss - floor), 1)\n    return 10 ** w[1], w[0]\n", "correct", None),
    ("fit-power-law", _f(logs="    log_x = np.log10(x)\n    log_rest = np.log10(loss)\n"), "wrong", "no-floor"),
    ("fit-power-law", _f(fit="    b, k = np.polyfit(log_x, log_rest, 1)\n"), "wrong", "swapped"),
    ("fit-power-law", _f(a="    a = b\n"), "wrong", "log-a"),
    ("fit-power-law", _f(logs="    log_x = np.log(x)\n    log_rest = np.log(loss - floor)\n"), "wrong", "ln-mix"),
    ("fit-power-law", _f(fit="    k, b = np.polyfit(log_rest, log_x, 1)\n"), "wrong", "axes"),
    ("fit-power-law", _f(fit="    k = (log_rest[-1] - log_rest[0]) / (log_x[-1] - log_x[0])\n    b = log_rest[0] - k * log_x[0]\n"), "almost", "two-points"),
    ("fit-power-law", _f(tail="    return k, a\n"), "wrong", "order"),
    ("fit-power-law", _f(tail="    return k, b\n"), "wrong", None),
    ("fit-power-law", _f(tail="    return round(a, 1), round(k, 3)\n"), "almost", "rounded"),
    ("fit-power-law", _f(tail="    return k\n"), "wrong", None),
]
