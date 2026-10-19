# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson math-derivative-rules: df, the derivative of x³ · log(x) with the product rule, and dk, the
derivative of log(1 + eˣ) with the chain rule. And the picture: eˣ and log(x) with the slope their rules give."""
import math
import re

from everything_ai.check import FunctionBox, Mistake

def _num(v) -> str:
    """A number the way the lesson writes it, rounded to 3 decimals, never in e-notation."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) and not hasattr(v, "__float__"):
        return repr(v)
    v = float(v)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    text = f"{round(v, 3) + 0.0:.3f}".rstrip("0").rstrip(".")
    return text if text != "-0" else "0"


def _close(got, want) -> bool:
    try:
        g, w = float(got), float(want)
    except (TypeError, ValueError):
        return False
    return math.isclose(g, w, rel_tol=1e-7, abs_tol=1e-9)


# --- the picture: eˣ and log(x), each with the slope its rule gives at one point -----------------------------------------

def picture(labels: dict) -> str:
    from everything_ai import chart
    xs = [-2 + 4 * i / 200 for i in range(201)]
    a = chart.Panel((-2, 2), (-0.5, 7.5), title=labels["exp"], xlabel="x", ylabel=labels["exp_y"])
    a.line(xs, [math.exp(v) for v in xs], "data")
    a.line([0.2, 1.8], [math.e * (1 - 0.8), math.e * (1 + 0.8)], "grad", 2.6)
    a.points([1], [math.e], "data", 5.5)
    a.text(1.05, math.e + 2.4, labels["exp_note"], "grad", "end")
    ls = [0.1 + 3.9 * i / 200 for i in range(201)]
    b = chart.Panel((0, 4), (-2.5, 1.8), title=labels["log"], xlabel="x", ylabel=labels["log_y"])
    b.line(ls, [math.log(v) for v in ls], "data")
    b.line([1, 3], [math.log(2) - 0.5, math.log(2) + 0.5], "grad", 2.6)
    b.points([2], [math.log(2)], "data", 5.5)
    b.text(3.9, -1.3, labels["log_note"], "grad", "end")
    return chart.figure([a, b], alt=labels["alt"], width=300, height=270)


# The lesson is done when these are correct. "exp-3x" is the choice question in the lesson.
REQUIRED = ["exp-3x", "product", "softplus"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _measures(text: str) -> bool:
    """`df` or `dk` itself calls `f` or `k` at x plus or minus a step: it measures the slope. A check line outside the
    function or a print, as in step 4 of the recipe, does not count, and neither does a comment."""
    body, inside = [], False
    for line in text.splitlines():
        if re.match(r"def d[fk]\(", line):
            inside = True
        elif line.strip() and not line[0].isspace():
            inside = False
        if inside and not line.strip().startswith("print("):   # a print that checks is not a measurement
            body.append(line.split("#")[0])
    return re.search(r"\b[fk]\(\s*x\s*[+-]", "\n".join(body)) is not None


# A slope measured with a step is never exactly what the rules give. No numbers in the message: with a tiny step the
# measured and the exact slope round to the same three decimals, and with a very tiny step they pass the check.
_MEASURED = Mistake(
    "measured", matches=lambda got, text: _measures(text),
    en="Your function measures the slope with a step, and a measured slope is never exactly the derivative. Measuring is a good check, but the rules give the exact derivative: write it as a formula in `x`.",
    nl="Je functie meet de helling met een stap, en een gemeten helling is nooit precies de afgeleide. Meten is een goede controle, maar de regels geven de precieze afgeleide: schrijf hem als formule in `x`.")


# --- exercise 2: the product rule, for x³ · log(x) ----------------------------------------------------------------------

_D_HEAD = "import numpy as np\n\n\ndef f(x):\n    return x ** 3 * np.log(x)\n\n\ndef df(x):\n"
_D_START = _D_HEAD + "    return ...\n"


def _d(body: str = "3 * x ** 2 * np.log(x) + x ** 3 * (1 / x)") -> str:
    return _D_HEAD + f"    return {body}\n"


# --- exercise 3: the chain rule, for log(1 + eˣ) ------------------------------------------------------------------------

_G_HEAD = "import numpy as np\n\n\ndef k(x):\n    return np.log(1 + np.exp(x))\n\n\ndef dk(x):\n"
_G_START = _G_HEAD + "    return ...\n"


def _g(body: str = "np.exp(x) / (1 + np.exp(x))") -> str:
    return _G_HEAD + f"    return {body}\n"


EXERCISES = {
    "product": FunctionBox(
        name="product",
        function="df",
        start=_D_START,
        solution=_d(),
        visible=[(2,)],
        hidden=[(1,), (0.5,), (3,), (10,)],
        show=_num,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("times-slopes", wrong=_d("3 * x ** 2 * (1 / x)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you multiply the two derivatives. The product rule has two terms: the derivative of the first part times the second part, plus the first part times the derivative of the second.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je vermenigvuldigt de twee afgeleiden. De productregel heeft twee termen: de afgeleide van het eerste deel keer het tweede deel, plus het eerste deel keer de afgeleide van het tweede."),
            Mistake("first-only", wrong=_d("3 * x ** 2 * np.log(x)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you have the first term of the product rule only. Add the second: the first part, `x ** 3`, times the derivative of `np.log(x)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je hebt alleen de eerste term van de productregel. Tel de tweede erbij op: het eerste deel, `x ** 3`, keer de afgeleide van `np.log(x)`."),
            Mistake("second-only", wrong=_d("x ** 3 * (1 / x)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you have the second term of the product rule only. Add the first: the derivative of `x ** 3` times `np.log(x)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je hebt alleen de tweede term van de productregel. Tel de eerste erbij op: de afgeleide van `x ** 3` keer `np.log(x)`."),
            _MEASURED,
        ],
        form=[_MEASURED],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the derivative of `f`, by the product rule.",
                      "nl": "Vervang de `...` door de afgeleide van `f`, met de productregel."},
            "arguments": {"en": "The check calls `{call}`, with one number, and the `def` line of your function has another number of names between its parentheses. Keep `(x)`.",
                          "nl": "De check roept `{call}` aan, met één getal, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Which two parts does `f` multiply, and what is the derivative of each?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Welke twee delen vermenigvuldigt `f`, en wat is de afgeleide van elk?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every `x` above 0?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke `x` boven 0?"},
            "correct": {"en": "Your derivative is right at all {n} points: the product rule, with the power rule for one part and the rule for the log for the other.",
                        "nl": "Je afgeleide klopt in alle {n} punten: de productregel, met de machtsregel voor het ene deel en de regel voor de log voor het andere."},
        },
    ),
    "softplus": FunctionBox(
        name="softplus",
        function="dk",
        start=_G_START,
        solution=_g(),
        visible=[(1,)],
        hidden=[(-1,), (-2,), (3,), (0.5,)],
        show=_num,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("no-inner", wrong=_g("1 / (1 + np.exp(x))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you have the outer derivative, 1 divided by the inner value, but not times the derivative of the inner part. The chain rule multiplies the two: what is the derivative of `1 + np.exp(x)`?",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je hebt de buitenste afgeleide, 1 gedeeld door de binnenste waarde, maar niet keer de afgeleide van het binnenste deel. De kettingregel vermenigvuldigt die twee: wat is de afgeleide van `1 + np.exp(x)`?"),
            Mistake("no-outer", wrong=_g("np.exp(x)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you have the derivative of the inner part, `np.exp(x)`, but not the outer one. The log gives 1 divided by its input, worked out at the inner value `1 + np.exp(x)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je hebt de afgeleide van het binnenste deel, `np.exp(x)`, maar niet de buitenste. De log geeft 1 gedeeld door zijn input, uitgerekend bij de binnenste waarde `1 + np.exp(x)`."),
            Mistake("outer-at-x", wrong=_g("np.exp(x) / x"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you work out the outer derivative at `x`, not at the inner value. The log is taken of `1 + np.exp(x)`, so its derivative is 1 divided by that.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je rekent de buitenste afgeleide uit bij `x`, niet bij de binnenste waarde. De log wordt genomen van `1 + np.exp(x)`, dus zijn afgeleide is 1 gedeeld door die waarde."),
            _MEASURED,
        ],
        form=[_MEASURED],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the derivative of `k`, by the chain rule.",
                      "nl": "Vervang de `...` door de afgeleide van `k`, met de kettingregel."},
            "arguments": {"en": "The check calls `{call}`, with one number, and the `def` line of your function has another number of names between its parentheses. Keep `(x)`.",
                          "nl": "De check roept `{call}` aan, met één getal, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. What is the outer part of `k`, and what is inside it? The chain rule needs the derivative of each.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Wat is het buitenste deel van `k`, en wat zit erin? De kettingregel heeft de afgeleide van elk nodig."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every `x`, also below 0?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke `x`, ook onder 0?"},
            "correct": {"en": "Your derivative is right at all {n} points: the rule for the log at the inner value, times the derivative of the inner part.",
                        "nl": "Je afgeleide klopt in alle {n} punten: de regel voor de log bij de binnenste waarde, keer de afgeleide van het binnenste deel."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("product", _D_START, "empty", None),
    ("product", _d(), "correct", None),
    ("product", _d("3 * x ** 2 * np.log(x) + x ** 2"), "correct", None),
    ("product", _d("x ** 2 * (3 * np.log(x) + 1)"), "correct", None),
    ("product", _d("3 * x ** 2 * (1 / x)"), "wrong", "times-slopes"),
    ("product", _d("3 * x ** 2 * np.log(x)"), "wrong", "first-only"),
    ("product", _d("x ** 3 * (1 / x)"), "wrong", "second-only"),
    ("product", _d("(f(x + 0.001) - f(x)) / 0.001"), "wrong", "measured"),
    ("product", _d("(f(x + 1e-5) - f(x - 1e-5)) / 2e-5"), "wrong", "measured"),   # passes every point: caught by form
    ("product", _d("3 * x ** 2 * np.log(x) + x ** 3") + "\n\nh = 0.001\nprint(df(2), (f(2 + h) - f(2)) / h)\nx = 2\nprint((f(x + h) - f(x)) / h)\n", "wrong", None),
    ("product", _d() + "\n\nx = 2\nh = 0.001\nprint(df(x), (f(x + h) - f(x)) / h)\n", "correct", None),
    ("softplus", _G_START, "empty", None),
    ("softplus", _g(), "correct", None),
    ("softplus", _g("1 / (1 + np.exp(-x))"), "correct", None),
    ("softplus", _g("1 / (1 + np.exp(x)) * np.exp(x)"), "correct", None),
    ("softplus", _g("1 / (1 + np.exp(x))"), "wrong", "no-inner"),
    ("softplus", _g("np.exp(x)"), "wrong", "no-outer"),
    ("softplus", _g("np.exp(x) / x"), "wrong", "outer-at-x"),
    ("softplus", _g("1 / x * np.exp(x)"), "wrong", "outer-at-x"),
    ("softplus", _g("(k(x + 0.001) - k(x)) / 0.001"), "wrong", "measured"),
]
