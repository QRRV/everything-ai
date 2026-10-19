# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson math-partial-derivatives: grad_f, the gradient of x · y² + log(x) as a formula, and dz_dt, the
derivative of z = u · v along its two paths, with u = t² and v = e^(2t). And the picture: the two paths from t to z."""
import math
import re

from everything_ai import draw
from everything_ai.check import FunctionBox, Mistake


def _plain(v):
    """A NumPy array or number as plain Python: an array becomes a list, a NumPy number a float."""
    return v.tolist() if hasattr(v, "tolist") else v


def _num(v) -> str:
    """A number rounded to 3 decimals, never in e-notation; a list of numbers, or an array, as a list."""
    v = _plain(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_num(x) for x in v) + "]"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return repr(v)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    text = f"{round(v, 3) + 0.0:.3f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _close(got, want) -> bool:
    got = _plain(got)   # a NumPy array of the right numbers is right too
    if isinstance(want, (list, tuple)):
        return (isinstance(got, (list, tuple)) and len(got) == len(want)
                and all(_close(g, w) for g, w in zip(got, want)))
    try:
        return math.isclose(float(got), float(want), rel_tol=1e-7, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


# --- the picture: the two paths from t to z = u · v, at t = 2 -------------------------------------------------------------

def paths_picture(labels: dict):
    t = 2
    u, v = t ** 2, 3 * t + 1
    nodes = [("t", 0, 0.5, "t", str(t), labels["t"], None),
             ("u", 1, 0, "u = t²", str(u), labels["u"], None),
             ("v", 1, 1, "v = 3t + 1", str(v), labels["v"], None),
             ("z", 2, 0.5, "z = u · v", str(u * v), labels["z"], None)]
    edges = [("t", "u", f"du/dt = {2 * t}"), ("t", "v", "dv/dt = 3"),
             ("u", "z", f"∂z/∂u = {v}"), ("v", "z", f"∂z/∂v = {u}")]
    return draw.computational_graph(nodes, edges, title=labels["title"], alt=labels["alt"])


# The lesson is done when these are correct. "partial-y" is the choice question in the lesson.
REQUIRED = ["partial-y", "gradient-formula", "two-paths"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# A derivative worked out with a step, such as `(f(x + h, y) - f(x, y)) / h`: only inside grad_f or dz_dt, and not in a
# comment or a print, so a check with a tiny step (step 5 of the recipe) does not count.


def _measures(text: str) -> bool:
    body, inside = [], False
    for line in text.split("\n"):
        if re.match(r"def (grad_f|dz_dt)\(", line):
            inside = True
            continue
        if inside and line.strip() and not line.startswith((" ", "\t")):
            inside = False
        if inside and not line.strip().startswith(("#", "print(")):   # a print that checks is not a measurement
            body.append(line.split("#")[0])
    return re.search(r"\b[fz]\([^()\n]*[+-]", "\n".join(body)) is not None

# Also in `form`: with a very small step, a measured slope can come within the tolerance. No numbers in the message: rounded
# to 3 decimals, a measured slope often looks the same as the exact one.
_MEASURED = Mistake(
    "measured",
    matches=lambda got, text: _measures(text),
    en="You measure a slope with a step, and a measured slope only comes close. The derivative rules give it exactly: write each derivative as a formula.",
    nl="Je meet een helling met een stap, en een gemeten helling komt alleen in de buurt. De differentieerregels geven hem precies: schrijf elke afgeleide als formule.")

# --- exercise 2: the gradient of f(x, y) = x · y² + log(x) ----------------------------------------------------------------

_G_HEAD = ("import numpy as np\n\n\ndef f(x, y):\n    return x * y ** 2 + np.log(x)\n\n\n"
           "def grad_f(x, y):\n")
_G_START = _G_HEAD + "    # the gradient of f: a list, first the partial derivative for x, then for y\n    ...\n"


def _g(dx: str = "y ** 2 + 1 / x", dy: str = "2 * x * y", ret: str = "[df_dx, df_dy]") -> str:
    return _G_HEAD + f"    df_dx = {dx}\n    df_dy = {dy}\n    return {ret}\n"


# --- exercise 3: z = u · v along two paths ------------------------------------------------------------------------------

_P_HEAD = ("import numpy as np\n\n\ndef dz_dt(t):\n    u = t ** 2\n    v = np.exp(2 * t)\n"
           "    dz_du = v   # ∂z/∂u\n    dz_dv = u   # ∂z/∂v\n    du_dt = 2 * t\n")
_P_START = "import numpy as np\n\n\ndef dz_dt(t):\n    # z = u · v, with u = t² and v = e^(2t)\n    ...\n"


def _p(dv: str = "2 * np.exp(2 * t)", ret: str = "dz_du * du_dt + dz_dv * dv_dt") -> str:
    return _P_HEAD + f"    dv_dt = {dv}\n    return {ret}\n"


EXERCISES = {
    "gradient-formula": FunctionBox(
        name="gradient-formula",
        function="grad_f",
        start=_G_START,
        solution=_g(),
        visible=[(2, 3)],
        hidden=[(1, 1), (0.5, -2), (4, 0), (3, 1.5)],
        show=_num,
        compare=_close,
        first=[_DOTS],
        form=[_MEASURED],
        mistakes=[
            Mistake("swapped", wrong=_g(ret="[df_dy, df_dx]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the two partial derivatives are in the wrong order. The gradient lists them in the order of the inputs: first for `x`, then for `y`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de twee partiële afgeleiden staan in de verkeerde volgorde. De gradient zet ze in de volgorde van de inputs: eerst voor `x`, dan voor `y`."),
            Mistake("no-log", wrong=_g(dx="y ** 2"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the partial derivative for `x` misses a term. `f` is a sum of two parts with `x` in them, `x * y ** 2` and `np.log(x)`; each gives its own derivative.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de partiële afgeleide naar `x` mist een term. `f` is een som van twee delen met `x` erin, `x * y ** 2` en `np.log(x)`; elk geeft zijn eigen afgeleide."),
            Mistake("log-in-y", wrong=_g(dy="2 * x * y + 1 / x"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: for `y`, `np.log(x)` is a fixed number, so its derivative is 0. Only the parts with `y` in them count.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: voor `y` is `np.log(x)` een vast getal, dus de afgeleide ervan is 0. Alleen de delen met `y` erin tellen mee."),
            Mistake("x-not-held", wrong=_g(dy="2 * y"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: in `x * y ** 2`, `x` is a number in front when you take the derivative for `y`, and a number in front stays in front.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: in `x * y ** 2` is `x` een getal ervoor als je de afgeleide naar `y` neemt, en een getal ervoor blijft ervoor staan."),
            _MEASURED,
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: the partial derivative of `f` for `x` and for `y`, returned as a list.",
                      "nl": "Vervang de `...` door de body van de functie: de partiële afgeleide van `f` naar `x` en naar `y`, teruggegeven als lijst."},
            "arguments": {"en": "The check calls `{call}`, with two numbers, and the `def` line of your function has another number of names between its parentheses. Keep `(x, y)`.",
                          "nl": "De check roept `{call}` aan, met twee getallen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x, y)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. For each partial derivative: which variable is held fixed, and what is the derivative of each part of `f`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Bij elke partiële afgeleide: welke variabele houd je vast, en wat is de afgeleide van elk deel van `f`?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every `x` above 0 and every `y`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke `x` boven 0 en elke `y`?"},
            "correct": {"en": "Your gradient is right at all {n} points: for each input, the others held fixed, and the derivative rules for the parts.",
                        "nl": "Je gradient klopt in alle {n} punten: voor elke input de andere vastgehouden, en de differentieerregels voor de delen."},
        },
    ),
    "two-paths": FunctionBox(
        name="two-paths",
        function="dz_dt",
        start=_P_START,
        solution=_p(),
        visible=[(1,)],
        hidden=[(0,), (0.5,), (-1,), (2,)],
        show=_num,
        compare=_close,
        first=[_DOTS],
        form=[_MEASURED],
        mistakes=[
            Mistake("one-path-u", wrong=_p(ret="dz_du * du_dt"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you follow only the path through `u`. `t` also reaches `z` through `v`; add that path too.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je volgt alleen het pad via `u`. `t` bereikt `z` ook via `v`; tel dat pad erbij op."),
            Mistake("one-path-v", wrong=_p(ret="dz_dv * dv_dt"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you follow only the path through `v`. `t` also reaches `z` through `u`; add that path too.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je volgt alleen het pad via `v`. `t` bereikt `z` ook via `u`; tel dat pad erbij op."),
            Mistake("paths-times", wrong=_p(ret="dz_du * du_dt * dz_dv * dv_dt"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you multiply the two paths. Along one path you multiply the local derivatives; the paths themselves are added.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je vermenigvuldigt de twee paden. Langs één pad vermenigvuldig je de lokale afgeleiden; de paden zelf tel je op."),
            Mistake("no-inner", wrong=_p(dv="np.exp(2 * t)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your derivative of `v` misses the derivative of the inner part. `v` is `e` to the power `2 * t`, so the chain rule multiplies by the derivative of `2 * t`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je afgeleide van `v` mist de afgeleide van het binnenste deel. `v` is `e` tot de macht `2 * t`, dus de kettingregel vermenigvuldigt met de afgeleide van `2 * t`."),
            Mistake("inner-value", wrong=_p(dv="2 * t * np.exp(2 * t)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your derivative of `v` multiplies by `2 * t`, the inner part itself. The chain rule multiplies by the derivative of the inner part, and the derivative of `2 * t` is 2. At `t = 1` the two are the same, so the example did not show it.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je afgeleide van `v` vermenigvuldigt met `2 * t`, het binnenste deel zelf. De kettingregel vermenigvuldigt met de afgeleide van het binnenste deel, en de afgeleide van `2 * t` is 2. Bij `t = 1` zijn die twee gelijk, dus het voorbeeld liet het niet zien."),
            _MEASURED,
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: `u` and `v`, the local derivatives along both paths, and the two paths added.",
                      "nl": "Vervang de `...` door de body van de functie: `u` en `v`, de lokale afgeleiden langs beide paden, en de twee paden opgeteld."},
            "arguments": {"en": "The check calls `{call}`, with one number, and the `def` line of your function has another number of names between its parentheses. Keep `(t)`.",
                          "nl": "De check roept `{call}` aan, met één getal, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(t)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Along which paths does `t` reach `z`, and what do you multiply along each?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Langs welke paden bereikt `t` de `z`, en wat vermenigvuldig je langs elk?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every `t`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke `t`?"},
            "correct": {"en": "Your derivative is right at all {n} points: along each path the local derivatives multiplied, and the two paths added.",
                        "nl": "Je afgeleide klopt in alle {n} punten: langs elk pad de lokale afgeleiden vermenigvuldigd, en de twee paden opgeteld."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("gradient-formula", _G_START, "empty", None),
    ("gradient-formula", _g(), "correct", None),
    ("gradient-formula", _g(dx="1 / x + y * y", dy="2 * y * x"), "correct", None),
    ("gradient-formula", _g(ret="[df_dy, df_dx]"), "wrong", "swapped"),
    ("gradient-formula", _g(dx="y ** 2"), "wrong", "no-log"),
    ("gradient-formula", _g(dy="2 * x * y + 1 / x"), "wrong", "log-in-y"),
    ("gradient-formula", _g(dy="2 * y"), "wrong", "x-not-held"),
    ("gradient-formula", _g(ret="np.array([df_dx, df_dy])"), "correct", None),
    ("gradient-formula", _g(ret="(df_dx, df_dy)"), "correct", None),
    ("gradient-formula", _g(dx="(f(x + 0.001, y) - f(x, y)) / 0.001"), "wrong", "measured"),
    ("gradient-formula", _g(dy="(f(x, y + 0.001) - f(x, y)) / 0.001"), "wrong", "measured"),
    ("gradient-formula", _g(dx="(f(x + 1e-7, y) - f(x, y)) / 1e-7"), "wrong", "measured"),
    # a wrong formula, checked with a tiny step as in step 5 of the recipe: that is not measuring
    ("gradient-formula", _g(dx="y ** 2 + np.log(x)").replace("    return [", "    print(df_dx, (f(x + 0.001, y) - f(x, y)) / 0.001)\n    return ["), "wrong", None),
    ("two-paths", _P_START, "empty", None),
    ("two-paths", _p(), "correct", None),
    ("two-paths", _p(dv="np.exp(2 * t) * 2", ret="du_dt * dz_du + dv_dt * dz_dv"), "correct", None),
    ("two-paths", _p(ret="dz_du * du_dt"), "wrong", "one-path-u"),
    ("two-paths", _p(ret="dz_dv * dv_dt"), "wrong", "one-path-v"),
    ("two-paths", _p(ret="dz_du * du_dt * dz_dv * dv_dt"), "wrong", "paths-times"),
    ("two-paths", _p(dv="np.exp(2 * t)"), "wrong", "no-inner"),
    ("two-paths", _p(dv="2 * t * np.exp(2 * t)"), "wrong", "inner-value"),
]
