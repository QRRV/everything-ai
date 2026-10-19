# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson math-slope-formulas: the power rule as a function, and a claim about a slope formula checked by
measuring. And the picture: x² with its slope at one x, measured and from the formula."""
import re

from everything_ai.check import CodeBox, FunctionBox, Mistake, run_code

H = 0.001


def measured(x: float) -> float:
    """The slope of x² at x, measured with a tiny step h = 0.001."""
    return ((x + H) ** 2 - x ** 2) / H


def picture(x: float, labels: dict[str, str]) -> str:
    """The function x² from −3 to 3, with a point at x and a short line with the slope there. Drawn as SVG text, for every
    stop of the slider once (L.explore); dragging across the chart sets x."""
    from everything_ai import chart
    xs = [-3 + 6 * i / 200 for i in range(201)]
    p = chart.Panel((-3.2, 3.2), (-2, 13), title=labels["title"], xlabel="x", ylabel="f(x)")
    p.line(xs, [v ** 2 for v in xs], "data")
    p.line([x - 0.6, x + 0.6], [x ** 2 - 1.2 * x, x ** 2 + 1.2 * x], "grad", 2.6)
    p.points([x], [x ** 2], "data", 5.5)
    return chart.figure([p], alt=labels["alt"], width=460, height=290, drag=0)   # its text stays 11 px or more in a play block at 700 px


# The lesson is done when these are correct. "x4" is the choice question in the lesson.
REQUIRED = ["x4", "power", "claim"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_POWER_START = "def power_slope(n, x):\n    ...\n"
_POWER = "def power_slope(n, x):\n    return n * x ** (n - 1)"


def _power(new: str) -> str:
    return _POWER.replace("    return n * x ** (n - 1)", new)


_CLAIM_HEAD = "def f(x):\n    return x ** 3\n\n\nx = 2\nh = 0.001\n"
_CLAIM_START = _CLAIM_HEAD + "measured = ...\nmeasured\n"
_CLAIM = _CLAIM_HEAD + "measured = (f(x + h) - f(x)) / h\nmeasured"


def _claim(new: str) -> str:
    return _CLAIM.replace("measured = (f(x + h) - f(x)) / h", new)


def _close(got, want) -> bool:
    """A measured slope depends a little on h: a smaller h, or a step to both sides, is right too."""
    try:
        return abs(float(got) - float(want)) <= 0.01 * abs(float(want))
    except (TypeError, ValueError):
        return False


def _ignores_f(text: str) -> bool:
    """Does `measured` stay the same when f gives twice as much? Then it was not measured with f: a number typed in,
    or a formula. A measurement doubles with f, however it is written."""
    doubled = text.replace("    return x ** 3", "    return 2 * x ** 3", 1)
    one, two = run_code(text), run_code(doubled)
    if one.error is not None or two.error is not None or not one.has_value or not two.has_value:
        return False
    try:
        return abs(float(one.value) - float(two.value)) <= 1e-9 * max(1.0, abs(float(one.value)))
    except (TypeError, ValueError):
        return False


EXERCISES = {
    "power": FunctionBox(
        name="power",
        function="power_slope",
        start=_POWER_START,
        solution=_POWER,
        visible=[(3, 2)],
        hidden=[(2, 5), (1, 7), (4, 1), (5, -1), (2, 0.5)],
        python_values=True,
        first=[_DOTS, Mistake("caret", matches=lambda got, text: "^" in text,
                              en="A power in Python is `**`, not `^`: x to the power 3 is `x ** 3`.",
                              nl="Een macht is in Python `**`, niet `^`: x tot de macht 3 is `x ** 3`.")],
        mistakes=[
            Mistake("same-power", wrong=_power("    return n * x ** n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function keeps the power the same. The rule takes one off: x to the power `n - 1`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je functie houdt de macht gelijk. De regel haalt er één af: x tot de macht `n - 1`."),
            Mistake("no-factor", wrong=_power("    return x ** (n - 1)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the factor `n` in front is missing. The old power comes to the front: `n * ...`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de factor `n` vooraan ontbreekt. De oude macht komt naar voren: `n * ...`."),
            Mistake("times-x", wrong=_power("    return n * x"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: n times x is the rule for x² only. For other powers, x keeps a power: `x ** (n - 1)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: n keer x is alleen de regel voor x². Bij andere machten houdt x een macht: `x ** (n - 1)`."),
            Mistake("no-brackets", wrong=_power("    return n * x ** n - 1"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. Python works out `x ** n` first, and then subtracts 1. Put the new power between brackets: `x ** (n - 1)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Python rekent eerst `x ** n` uit, en trekt er dan 1 van af. Zet de nieuwe macht tussen haakjes: `x ** (n - 1)`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the slope of x to the power n at x, from the power rule, given back with `return`.",
                      "nl": "Vervang de `...` door de helling van x tot de macht n bij x, volgens de machtsregel, teruggegeven met `return`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(n, x)`.",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(n, x)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. The power rule: x to the power n has the slope n times x to the power n − 1.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. De machtsregel: x tot de macht n heeft de helling n keer x tot de macht n − 1."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for every power and every x?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elke macht en elke x?"},
            "correct": {"en": "Your function is right in all {n} cases. `power_slope(3, 2)` gives 12: the slope of x³ at x = 2 is 3 · 2² = 12, exactly, without measuring.",
                        "nl": "Je functie klopt in alle {n} gevallen. `power_slope(3, 2)` geeft 12: de helling van x³ bij x = 2 is 3 · 2² = 12, precies, zonder te meten."},
        },
    ),
    "claim": CodeBox(
        name="claim",
        start=_CLAIM_START,
        solution=_CLAIM,
        keep=["def f(x):", "    return x ** 3", "x = 2"],
        # The check also measures at another x: a number typed in by hand does not change with it.
        others=[{"x": 3}],
        compare=_close,
        first=[_DOTS,
               Mistake("formula", matches=lambda got, text: _ignores_f(text),
                       en="Your `measured` does not use `f`. To check a formula, you measure: change x by `h`, and look at what `f` does.",
                       nl="Je `measured` gebruikt `f` niet. Om een formule te controleren, meet je: verander x met `h`, en kijk wat `f` doet.")],
        mistakes=[
            Mistake("no-divide", wrong=_claim("measured = f(x + h) - f(x)"),
                    en="Your last line gives `{got}`: the change in f itself. A slope is that change divided by the step: divide it by `h`.",
                    nl="Je laatste regel geeft `{got}`: de verandering van f zelf. Een helling is die verandering gedeeld door de stap: deel hem door `h`."),
            Mistake("flipped", wrong=_claim("measured = (f(x) - f(x + h)) / h"),
                    en="Your last line gives `{got}`: the right size, but the wrong sign. Subtract f where you started from f a step further, not the other way round.",
                    nl="Je laatste regel geeft `{got}`: de goede grootte, maar het verkeerde teken. Trek f waar je begon af van f een stap verder, niet andersom."),
            Mistake("brackets", wrong=_claim("measured = f(x + h) - f(x) / h"),
                    en="Your last line gives `{got}`. Without brackets, Python divides only `f(x)` by `h`. Put the whole difference between brackets.",
                    nl="Je laatste regel geeft `{got}`. Zonder haakjes deelt Python alleen `f(x)` door `h`. Zet het hele verschil tussen haakjes."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the slope of `f` at `x`, measured with the tiny step `h`.",
                      "nl": "Vervang de `...` door de helling van `f` bij `x`, gemeten met het heel kleine stapje `h`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another x by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere x."},
            "no_value": {"en": "The last line of the box shows nothing. Put `measured` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `measured` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be the change in `f` when x goes up by `h`, divided by `h`. Do you divide the whole difference, and is `h` still tiny?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet de verandering van `f` zijn als x met `h` omhooggaat, gedeeld door `h`. Deel je het hele verschil, en is `h` nog heel klein?"},
            "other": {"en": "Right for x = 2. But with {where}, your last line gives `{got}`, and it should give about `{want}`. Do you work it out with `f`, `x` and `h` themselves?",
                      "nl": "Goed voor x = 2. Maar met {where} geeft je laatste regel `{got}`, en dat moet ongeveer `{want}` zijn. Reken je het uit met `f`, `x` en `h` zelf?"},
            "other_error": {"en": "Right for x = 2. But with {where}, your code stops with an error. Do you work it out with `f`, `x` and `h` themselves?",
                            "nl": "Goed voor x = 2. Maar met {where} stopt je code met een fout. Reken je het uit met `f`, `x` en `h` zelf?"},
            "correct": {"en": "You measured about 12. At x = 2 the claim 3x gives 3 · 2 = 6: far off, so the claim is wrong. The power rule gives 3x², and 3 · 2² = 12: that matches. The check also measured at x = 3: about 27, and 3 · 3² = 27.",
                        "nl": "Je mat ongeveer 12. Bij x = 2 geeft de bewering 3x: 3 · 2 = 6, ver ernaast, dus de bewering klopt niet. De machtsregel geeft 3x², en 3 · 2² = 12: dat klopt. De check mat ook bij x = 3: ongeveer 27, en 3 · 3² = 27."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_P = "def power_slope(n, x):\n"
CASES = [
    ("power", _POWER_START, "empty", None),
    ("power", _POWER, "correct", None),
    ("power", _P + "    new_power = n - 1\n    return n * x ** new_power", "correct", None),
    ("power", _P + "    return n * pow(x, n - 1)", "correct", None),
    ("power", _power("    return n * x ** n"), "wrong", "same-power"),
    ("power", _power("    return x ** (n - 1)"), "wrong", "no-factor"),
    ("power", _power("    return n * x"), "wrong", "times-x"),
    ("power", _power("    return n * x ** n - 1"), "wrong", "no-brackets"),
    ("power", _P + "    return 12", "almost", None),
    ("claim", _CLAIM_START, "empty", None),
    ("claim", _CLAIM, "correct", None),
    ("claim", _claim("measured = (f(x + 0.0001) - f(x)) / 0.0001"), "correct", None),
    ("claim", _claim("measured = (f(x + h) - f(x - h)) / (2 * h)"), "correct", None),
    ("claim", _claim("measured = 3 * x ** 2"), "wrong", "formula"),
    ("claim", _claim("measured = 3 * x"), "wrong", "formula"),
    ("claim", _claim("change = f(x + h) - f(x)\nmeasured = change / h"), "correct", None),
    ("claim", _claim("measured = ((x + h) ** 3 - x ** 3) / h"), "wrong", "formula"),
    ("claim", _claim("measured = 12.006"), "wrong", "formula"),
    ("power", _power("    return n * x ^ (n - 1)"), "wrong", "caret"),
    ("claim", _claim("measured = f(x + h) - f(x)"), "wrong", "no-divide"),
    ("claim", _claim("measured = (f(x) - f(x + h)) / h"), "wrong", "flipped"),
    ("claim", _claim("measured = f(x + h) - f(x) / h"), "wrong", "brackets"),
    ("claim", _claim("measured = (f(2 + h) - f(2)) / h"), "almost", "other-value"),
    ("claim", _CLAIM_HEAD + "measured = (f(x + h) - f(x)) / h\nprint(measured)", "wrong", "print-last"),
]
