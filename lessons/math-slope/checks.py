# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson math-slope: the slope of the loss for b, measured with a finite difference, and a function
that uses the slope for w to take one step downhill. And the picture: the loss for every w, with its slope at one w."""
import re

from everything_ai.check import CodeBox, FunctionBox, Mistake

# The houses that were sold, from the lesson ml-vocabulary: area in m², price in thousands of euros.
AREA = [62, 85, 110, 48, 95, 130, 73, 58]
PRICE = [245, 310, 395, 198, 320, 465, 280, 232]
H = 0.001


def loss(w: float, b: float = 50) -> float:
    """The mean squared error of the model w · area + b on the houses."""
    return sum((p - (w * a + b)) ** 2 for a, p in zip(AREA, PRICE)) / len(AREA)


def slope(w: float, b: float = 50) -> float:
    """The slope of the loss for w, measured as the lesson does: a finite difference with h = 0.001."""
    return (loss(w + H, b) - loss(w, b)) / H


def picture(w: float, labels: dict[str, str]) -> str:
    """The loss for every w, with b = 50, and at the chosen w a point and a short line with the slope there. Drawn as
    SVG text, for every stop of the slider once (L.explore); dragging across the chart sets w. 500 wide: in the block, at
    700 px, its text stays 11 px or more."""
    from everything_ai import chart
    xs = [2.0 + 2.2 * i / 200 for i in range(201)]
    s = slope(w)
    p = chart.Panel((2.0, 4.2), (-800, 9800), title=labels["title"], xlabel="w", ylabel=labels["loss"])
    p.line(xs, [loss(x) for x in xs], "loss")
    p.line([w - 0.25, w + 0.25], [loss(w) - 0.25 * s, loss(w) + 0.25 * s], "grad", 2.6)
    p.points([w], [loss(w)], "param", 5.5)
    return chart.figure([p], alt=labels["alt"], width=460, height=290, drag=0)   # its text stays 11 px or more in a play block at 700 px


# The lesson is done when these are correct. "which-way" is the choice question in the lesson.
REQUIRED = ["which-way", "slope-b", "next-w"]

_DATA = ("import numpy as np\n\narea = np.array([62, 85, 110, 48, 95, 130, 73, 58])\n"
         "price = np.array([245, 310, 395, 198, 320, 465, 280, 232])\n\n\n"
         "def loss(w, b):\n    residuals = price - (w * area + b)\n    return (residuals ** 2).mean()\n\n\n")
_B_HEAD = _DATA + "w = 3\nb = 50\nh = 0.001\n"
_B_START = _B_HEAD + "slope_b = ...\nslope_b\n"
_B = _B_HEAD + "slope_b = (loss(w, b + h) - loss(w, b)) / h\nslope_b"

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _slope_b(new: str) -> str:
    return _B.replace("slope_b = (loss(w, b + h) - loss(w, b)) / h", new)


def _close(got, want) -> bool:
    """A finite difference depends a little on h: a smaller h, or a step to both sides, is right too."""
    try:
        return abs(float(got) - float(want)) <= 0.01 * abs(float(want)) + 0.01
    except (TypeError, ValueError):
        return False


_NEXT_START = _DATA + "def next_w(w, b, step):\n    h = 0.001\n    ...\n"
_NEXT = (_DATA + "def next_w(w, b, step):\n    h = 0.001\n    slope = (loss(w + h, b) - loss(w, b)) / h\n"
         "    if slope > 0:\n        return w - step\n    return w + step")


def _next(new: str) -> str:
    return _NEXT.replace("    if slope > 0:\n        return w - step\n    return w + step", new)


EXERCISES = {
    "slope-b": CodeBox(
        name="slope-b",
        start=_B_START,
        solution=_B,
        keep=["area = np.array([62, 85, 110, 48, 95, 130, 73, 58])", "price = np.array([245, 310, 395, 198, 320, 465, 280, 232])",
              "def loss(w, b):", "    residuals = price - (w * area + b)", "    return (residuals ** 2).mean()", "w = 3", "b = 50"],
        # The check also measures the slope at another b: a number typed in by hand does not change with it.
        others=[{"b": 70}],
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("one-value", error=lambda e: isinstance(e, TypeError) and "required positional argument: 'b'" in str(e),
                    en="In the box, `loss` needs two values, `loss(w, b)`, and your code gives it one. The worked example had only w; here b goes in too.",
                    nl="In het vak heeft `loss` twee waarden nodig, `loss(w, b)`, en je code geeft er één. Het uitgewerkte voorbeeld had alleen w; hier gaat b er ook in."),
            Mistake("nudged-w", wrong=_slope_b("slope_b = (loss(w + h, b) - loss(w, b)) / h"),
                    en="Your last line gives `{got}`: the slope for w. This time you change b by a tiny step, and w stays as it is.",
                    nl="Je laatste regel geeft `{got}`: de helling voor w. Deze keer verander je b met een heel klein stapje, en blijft w zoals hij is."),
            Mistake("no-divide", wrong=_slope_b("slope_b = loss(w, b + h) - loss(w, b)"),
                    en="Your last line gives `{got}`: the change in the loss itself. A slope is that change divided by the step: divide it by `h`.",
                    nl="Je laatste regel geeft `{got}`: de verandering van de loss zelf. Een helling is die verandering gedeeld door de stap: deel hem door `h`."),
            Mistake("flipped", wrong=_slope_b("slope_b = (loss(w, b) - loss(w, b + h)) / h"),
                    en="Your last line gives `{got}`: the right size, but the wrong sign. Subtract the loss where you started from the loss a step further, not the other way round.",
                    nl="Je laatste regel geeft `{got}`: de goede grootte, maar het verkeerde teken. Trek de loss waar je begon af van de loss een stap verder, niet andersom."),
            Mistake("brackets", wrong=_slope_b("slope_b = loss(w, b + h) - loss(w, b) / h"),
                    en="Your last line gives `{got}`. Without brackets, Python divides only `loss(w, b)` by `h`. Put the whole difference between brackets.",
                    nl="Je laatste regel geeft `{got}`. Zonder haakjes deelt Python alleen `loss(w, b)` door `h`. Zet het hele verschil tussen haakjes."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the slope of the loss for b, measured with a tiny step `h`.",
                      "nl": "Vervang de `...` door de helling van de loss voor b, gemeten met een heel klein stapje `h`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another b by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere b."},
            "no_value": {"en": "The last line of the box shows nothing. Put `slope_b` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `slope_b` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be the change in the loss when b goes up by `h`, divided by `h`. Which parameter do you change, and do you divide the whole difference?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet de verandering van de loss zijn als b met `h` omhooggaat, gedeeld door `h`. Welke parameter verander je, en deel je het hele verschil?"},
            "other": {"en": "Right for b = 50. But with {where}, your last line gives `{got}`, and it should give about `24.5`. Do you work it out with `loss`, `b` and `h` themselves?",
                      "nl": "Goed voor b = 50. Maar met {where} geeft je laatste regel `{got}`, en dat moet ongeveer `24.5` zijn. Reken je het uit met `loss`, `b` en `h` zelf?"},
            "other_error": {"en": "Right for b = 50. But with {where}, your code stops with an error. Do you work it out with `loss`, `b` and `h` themselves?",
                            "nl": "Goed voor b = 50. Maar met {where} stopt je code met een fout. Reken je het uit met `loss`, `b` en `h` zelf?"},
            "correct": {"en": "The slope for b is about −15.5. Negative: when b gets a little bigger, the loss goes down. That fits the previous lesson, where b = 58 had a lower loss than 50. The check also tried b = 70: there the slope is about 24.5, positive, so a smaller b is better.",
                        "nl": "De helling voor b is ongeveer −15,5. Negatief: als b een beetje groter wordt, daalt de loss. Dat past bij de vorige les, waar b = 58 een lagere loss had dan 50. De check probeerde ook b = 70: daar is de helling ongeveer 24,5, positief, dus daar is een kleinere b beter."},
        },
    ),
    "next-w": FunctionBox(
        name="next-w",
        function="next_w",
        start=_NEXT_START,
        solution=_NEXT,
        visible=[(2.5, 50, 0.1)],
        hidden=[(3.5, 50, 0.1), (3.2, 50, 0.5), (3.0, 70, 0.1), (1.0, 230, 0.1), (2.0, 100, 0.5)],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("one-value", error=lambda e: isinstance(e, TypeError) and "required positional argument: 'b'" in str(e),
                    en="In the box, `loss` needs two values, `loss(w, b)`, and your code gives it one. The worked example had only w; here b goes in too.",
                    nl="In het vak heeft `loss` twee waarden nodig, `loss(w, b)`, en je code geeft er één. Het uitgewerkte voorbeeld had alleen w; hier gaat b er ook in."),
            Mistake("slope-for-b", state="almost", wrong=_NEXT.replace("loss(w + h, b) - loss(w, b)", "loss(w, b + h) - loss(w, b)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. Your function changes b by `h`: that is the slope for b. To move w, change w by `h`, and keep b as it is.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je functie verandert b met `h`: dat is de helling voor b. Om w te verschuiven, verander je w met `h`, en laat je b zoals hij is."),
            Mistake("step-times-slope", wrong=_next("    return w - step * slope"),
                    en="`{call}` gives `{got}`: w moved by `step` times the slope. Here the slope only says which way; the step itself is `step`.",
                    nl="`{call}` geeft `{got}`: w is verschoven met `step` keer de helling. Hier zegt de helling alleen welke kant op; de stap zelf is `step`."),
            Mistake("uphill", wrong=_next("    if slope > 0:\n        return w + step\n    return w - step"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your step goes uphill. A negative slope means the loss goes down when w goes up.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je stap gaat bergop. Een negatieve helling betekent dat de loss daalt als w groter wordt."),
            Mistake("whole-slope", wrong=_next("    return w - slope"),
                    en="`{call}` gives `{got}`: w moved by the whole slope. The slope only says which way; the step itself is `step`.",
                    nl="`{call}` geeft `{got}`: w is met de hele helling verschoven. De helling zegt alleen welke kant op; de stap zelf is `step`."),
            Mistake("slope", wrong=_next("    return slope"),
                    en="`{call}` gives `{got}`: the slope itself. The task asks for the next w: one step downhill.",
                    nl="`{call}` geeft `{got}`: de helling zelf. De opdracht vraagt om de volgende w: één stap bergaf."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the slope for w, and then one step downhill, given back with `return`.",
                      "nl": "Vervang de `...` door de helling voor w, en dan één stap bergaf, teruggegeven met `return`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has a different number of names between its parentheses. Keep `(w, b, step)`.",
                          "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(w, b, step)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you measure the slope for w, and step against it?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Meet je de helling voor w, en stap je ertegenin?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function measure the slope at the w and b it gets, and step against it?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Meet je functie de helling bij de w en b die hij krijgt, en stapt hij ertegenin?"},
            "correct": {"en": "Your function is right in all {n} cases. It does one step of fitting by hand, but it no longer has to try both ways: the slope says which way is down.",
                        "nl": "Je functie klopt in alle {n} gevallen. Hij doet één stap van het passen met de hand, maar hoeft niet meer beide kanten te proberen: de helling zegt welke kant bergaf is."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_N = _DATA + "def next_w(w, b, step):\n    h = 0.001\n"
CASES = [
    ("slope-b", _B_START, "empty", None),
    ("slope-b", _B, "correct", None),
    ("slope-b", _slope_b("slope_b = (loss(w, b + 0.0001) - loss(w, b)) / 0.0001"), "correct", None),
    ("slope-b", _slope_b("slope_b = (loss(w, b + h) - loss(w, b - h)) / (2 * h)"), "correct", None),
    ("slope-b", _slope_b("slope_b = (loss(w + h, b) - loss(w, b)) / h"), "wrong", "nudged-w"),
    ("slope-b", _slope_b("slope_b = loss(w, b + h) - loss(w, b)"), "wrong", "no-divide"),
    ("slope-b", _slope_b("slope_b = (loss(w, b) - loss(w, b + h)) / h"), "wrong", "flipped"),
    ("slope-b", _slope_b("slope_b = loss(w, b + h) - loss(w, b) / h"), "wrong", "brackets"),
    ("slope-b", _slope_b("slope_b = -15.5"), "almost", "other-value"),
    ("slope-b", _slope_b("slope_b = (loss(3, 50 + h) - loss(3, 50)) / h"), "almost", "other-value"),
    ("slope-b", _B_HEAD + "slope_b = (loss(w, b + h) - loss(w, b)) / h\nprint(slope_b)", "wrong", "print-last"),
    ("next-w", _NEXT_START, "empty", None),
    ("next-w", _NEXT, "correct", None),
    ("next-w", _N + "    slope = (loss(w + h, b) - loss(w, b)) / h\n    if slope < 0:\n        return w + step\n    else:\n        return w - step", "correct", None),
    ("next-w", _N + "    if loss(w + h, b) < loss(w, b):\n        return w + step\n    return w - step", "correct", None),
    ("next-w", _next("    if slope > 0:\n        return w + step\n    return w - step"), "wrong", "uphill"),
    ("next-w", _next("    return w - step"), "wrong", None),
    ("next-w", _next("    return w - slope"), "wrong", "whole-slope"),
    ("next-w", _next("    return slope"), "wrong", "slope"),
    ("next-w", _N + "    return 2.6", "almost", None),
    ("next-w", _next("    return w + step"), "almost", None),
    ("next-w", _NEXT.replace("loss(w + h, b) - loss(w, b)", "loss(w, b + h) - loss(w, b)"), "almost", "slope-for-b"),
    ("next-w", _NEXT.replace("loss(w + h, b) - loss(w, b)", "loss(w + h, 50) - loss(w, 50)"), "almost", None),
    ("next-w", _N + "    if w < 3.1:\n        return w + step\n    return w - step", "almost", None),
    ("next-w", _next("    return w - step * slope"), "wrong", "step-times-slope"),
    ("next-w", _NEXT.replace("loss(w + h, b) - loss(w, b)", "loss(w + h) - loss(w)"), "error", "one-value"),
    ("slope-b", _slope_b("slope_b = (loss(b + h) - loss(b)) / h"), "error", "one-value"),
]
