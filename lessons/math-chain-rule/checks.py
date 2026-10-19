# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson math-chain-rule: the slope of the loss of one example as a function, and the slope with
respect to the bias term. And the picture: the loss of one student over w, with its slope from the chain rule."""
import re

from everything_ai.check import CodeBox, FunctionBox, Mistake

# The student of the picture and the worked example: 4 hours of study, a grade of 7. The model predicts w · hours.
HOURS, GRADE = 4, 7


def parts(w: float) -> tuple[float, float, float, float]:
    """For the student, at this w: the residual r, the slope of the loss r² with respect to r, the slope of r with
    respect to w, and their product: the slope of the loss with respect to w."""
    r = GRADE - w * HOURS
    return r, 2 * r, -HOURS, 2 * r * -HOURS + 0.0   # + 0.0: never a minus zero


def picture(w: float, labels: dict[str, str]) -> str:
    """The loss (7 − 4w)² for w from 0 to 3.5, with a point at w and a short line with the slope from the chain rule
    there. Drawn as SVG text, for every stop of the slider once (L.explore); dragging across the chart sets w."""
    from everything_ai import chart
    ws = [3.5 * i / 140 for i in range(141)]
    loss = (GRADE - w * HOURS) ** 2
    slope = parts(w)[3]
    p = chart.Panel((0, 3.5), (-4, 52), title=labels["title"], xlabel="w", ylabel=labels["loss"])
    p.line(ws, [(GRADE - v * HOURS) ** 2 for v in ws], "loss")
    p.line([w - 0.3, w + 0.3], [loss - 0.3 * slope, loss + 0.3 * slope], "grad", 2.6)
    p.points([w], [loss], "param", 5.5)
    return chart.figure([p], alt=labels["alt"], width=460, height=290, drag=0)   # its text stays 11 px or more in a play block at 700 px


# The lesson is done when these are correct. "nested" is the choice question in the lesson.
REQUIRED = ["nested", "loss-slope", "slope-b"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SLOPE_START = "def loss_slope(w, x, y):\n    ...\n"
_SLOPE = "def loss_slope(w, x, y):\n    r = y - w * x\n    return 2 * r * -x"


def _slope(new: str) -> str:
    return "def loss_slope(w, x, y):\n    r = y - w * x\n" + new


_HEAD = "x = 4\ny = 7\nw = 1.5\nb = 0.25\nr = y - (w * x + b)\n"
_B_START = _HEAD + "slope_b = ...\nslope_b\n"
_B = _HEAD + "slope_b = 2 * r * -1\nslope_b"


def _b(new: str) -> str:
    return _B.replace("slope_b = 2 * r * -1", new)


EXERCISES = {
    "loss-slope": FunctionBox(
        name="loss-slope",
        function="loss_slope",
        start=_SLOPE_START,
        solution=_SLOPE,
        visible=[(1.5, 4, 7)],
        hidden=[(3, 85, 310), (0, 2, 5), (1, -2, 4), (0.5, 4, 1), (2, 3, 6)],
        first=[_DOTS],
        mistakes=[
            Mistake("no-inner", wrong=_slope("    return 2 * r"),
                    en="`{call}` gives `{got}`: only the slope of the outer step, 2r. Multiply it by the slope of the inner step: how fast does r change when w goes up?",
                    nl="`{call}` geeft `{got}`: alleen de helling van de buitenste stap, 2r. Vermenigvuldig die met de helling van de binnenste stap: hoe snel verandert r als w omhooggaat?"),
            Mistake("sign", wrong=_slope("    return 2 * r * x"),
                    en="`{call}` gives `{got}`: the right size, but the wrong sign. When w goes up, the prediction goes up, so r = y − w·x goes down: its slope is −x.",
                    nl="`{call}` geeft `{got}`: de goede grootte, maar het verkeerde teken. Als w omhooggaat, gaat de voorspelling omhoog, dus r = y − w·x gaat omlaag: de helling is −x."),
            Mistake("no-two", wrong=_slope("    return r * -x"),
                    en="`{call}` gives `{got}`: half of what it should be. The outer step is r², and the power rule gives it the slope 2r.",
                    nl="`{call}` geeft `{got}`: de helft van wat het moet zijn. De buitenste stap is r², en de machtsregel geeft die de helling 2r."),
            Mistake("outer-at-w", wrong=_slope("    return 2 * w * -x"),
                    en="`{call}` gives `{got}`: you worked out the outer slope, 2r, with w instead of r. Fill in the inner value first: r = y − w·x.",
                    nl="`{call}` geeft `{got}`: je rekende de buitenste helling, 2r, uit met w in plaats van r. Vul eerst de binnenste waarde in: r = y − w·x."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the slope of the loss (y − w·x)² with respect to w, worked out with the chain rule, and give it back with `return`.",
                      "nl": "Vervang de `...` door de helling van de loss (y − w·x)² naar w, uitgerekend met de kettingregel, en geef die terug met `return`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with three numbers, and the `def` line of your function has a different number of names between its parentheses. Keep `(w, x, y)`.",
                          "nl": "De check roept `{call}` aan, met drie getallen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(w, x, y)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you multiply the outer slope 2r (with r = y − w·x) by the inner slope?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vermenigvuldig je de buitenste helling 2r (waarbij r = y − w·x) met de binnenste helling?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Do you work it out from `w`, `x` and `y` themselves?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken je het uit met `w`, `x` en `y` zelf?"},
            "correct": {"en": "Your function is right in all {n} cases. For the student, r = 7 − 1.5 × 4 = 1, so the slope is 2 × 1 × (−4) = −8: the loss goes down when w goes up.",
                        "nl": "Je functie klopt in alle {n} gevallen. Voor de leerling is r = 7 − 1,5 × 4 = 1, dus de helling is 2 × 1 × (−4) = −8: de loss daalt als w omhooggaat."},
        },
    ),
    "slope-b": CodeBox(
        name="slope-b",
        start=_B_START,
        solution=_B,
        keep=["x = 4", "y = 7", "w = 1.5", "r = y - (w * x + b)"],
        # The check also runs the box with another bias term: a number typed in by hand does not change with it.
        others=[{"b": 2}],
        first=[_DOTS],
        mistakes=[
            Mistake("copied-x", wrong=_b("slope_b = 2 * r * -x"),
                    en="Your last line gives `{got}`: the slope with respect to w. For b, the inner slope is different: when b goes up by 1, how much does r = y − (w·x + b) change?",
                    nl="Je laatste regel geeft `{got}`: de helling naar w. Voor b is de binnenste helling anders: als b 1 omhooggaat, hoeveel verandert r = y − (w·x + b) dan?"),
            Mistake("no-inner", wrong=_b("slope_b = 2 * r"),
                    en="Your last line gives `{got}`: only the outer slope, 2r. What is the inner slope: how much does r change when b goes up by 1?",
                    nl="Je laatste regel geeft `{got}`: alleen de buitenste helling, 2r. Wat is de binnenste helling: hoeveel verandert r als b 1 omhooggaat?"),
            Mistake("no-two", wrong=_b("slope_b = -r"),
                    en="Your last line gives `{got}`: half of what it should be. The outer step is r², with the slope 2r.",
                    nl="Je laatste regel geeft `{got}`: de helft van wat het moet zijn. De buitenste stap is r², met de helling 2r."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the slope of the loss r² with respect to b: the outer slope times the inner slope.",
                      "nl": "Vervang de `...` door de helling van de loss r² naar b: de buitenste helling keer de binnenste helling."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another value of b by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere bias term."},
            "no_value": {"en": "The last line of the box shows nothing. Put `slope_b` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `slope_b` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you multiply the outer slope, 2r, by the inner slope: how much r changes when b goes up by 1?",
                      "nl": "Je laatste regel geeft `{got}`. Vermenigvuldig je de buitenste helling, 2r, met de binnenste helling: hoeveel r verandert als b 1 omhooggaat?"},
            "other": {"en": "Right for this b. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out with `r` itself?",
                      "nl": "Goed voor deze b. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `r` zelf?"},
            "other_error": {"en": "Right for this b. But with {where}, your code stops with an error. Do you work it out with `r` itself?",
                            "nl": "Goed voor deze b. Maar met {where} stopt je code met een fout. Reken je het uit met `r` zelf?"},
            "correct": {"en": "The residual r is 7 − (1.5 × 4 + 0.25) = 0.75, so the slope with respect to b is 2 × 0.75 × (−1) = −1.5: raising b lowers the loss. The check also tried b = 2, where the prediction is too high and the slope is +2.",
                        "nl": "Het residu r is 7 − (1,5 × 4 + 0,25) = 0,75, dus de helling naar b is 2 × 0,75 × (−1) = −1,5: een grotere b maakt de loss kleiner. De check probeerde ook b = 2, waar de voorspelling te hoog is en de helling +2."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("loss-slope", _SLOPE_START, "empty", None),
    ("loss-slope", _SLOPE, "correct", None),
    ("loss-slope", "def loss_slope(w, x, y):\n    return 2 * (w * x - y) * x", "correct", None),
    ("loss-slope", "def loss_slope(w, x, y):\n    r = y - w * x\n    dloss_dr = 2 * r\n    dr_dw = -x\n    return dloss_dr * dr_dw", "correct", None),
    ("loss-slope", _slope("    return 2 * r"), "wrong", "no-inner"),
    ("loss-slope", _slope("    return 2 * r * x"), "wrong", "sign"),
    ("loss-slope", _slope("    return r * -x"), "wrong", "no-two"),
    ("loss-slope", _slope("    return 2 * w * -x"), "wrong", "outer-at-w"),
    ("loss-slope", "def loss_slope(w, x, y):\n    return -8.0", "almost", None),
    ("slope-b", _B_START, "empty", None),
    ("slope-b", _B, "correct", None),
    ("slope-b", _b("slope_b = -2 * r"), "correct", None),
    ("slope-b", _b("slope_b = 2 * (w * x + b - y)"), "correct", None),
    ("slope-b", _b("slope_b = 2 * r * -x"), "wrong", "copied-x"),
    ("slope-b", _b("slope_b = 2 * r"), "wrong", "no-inner"),
    ("slope-b", _b("slope_b = -r"), "wrong", "no-two"),
    ("slope-b", _b("slope_b = -1.5"), "almost", "other-value"),
    ("slope-b", _HEAD + "slope_b = 2 * r * -1\nprint(slope_b)", "wrong", "print-last"),
]
