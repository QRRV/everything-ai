# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-logistic-regression: the probability a logistic regression gives one example, and how many
hours a student must study to land on the decision boundary. And the picture of the students of ml-vocabulary with
the decision boundary of the lesson's model, moved by the bias term on the slider."""
import csv
import re
from pathlib import Path

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake

STUDENTS = Path(__file__).parent.parent / "ml-vocabulary" / "students.csv"
W = (1, 0.5)   # the lesson's model: a weight for the hours of study and one for the hours of sleep


def students() -> list[tuple[float, float, bool]]:
    """The students whose result is known: (hours, sleep, passed)."""
    with open(STUDENTS, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [(float(r["hours"]), float(r["sleep"]), r["passed"] == "yes") for r in rows if r["passed"]]


def sides(b: float) -> tuple[int, int]:
    """How many students the model with the weights W and the bias term b puts on the wrong side of the decision
    boundary, and how many lie exactly on it: there the logit is 0, and the model is unsure. With b a multiple of 0.5
    every logit is exact, so `== 0` is safe."""
    wrong = on = 0
    for h, s, passed in students():
        z = W[0] * h + W[1] * s + b
        if z == 0:
            on += 1
        elif (z > 0) != passed:
            wrong += 1
    return wrong, on


def picture(b: float, labels: dict[str, str]) -> str:
    """The students by hours of study and of sleep, passed filled and failed open, with the decision boundary
    hours + 0.5 · sleep + b = 0 as a line; and for every b on the slider how many students lie on the wrong side of
    it, and how many exactly on it. Drawn as SVG text, once for every stop of the slider (L.explore); dragging
    across the second chart sets b."""
    from everything_ai import chart
    p = chart.Panel((0, 10), (4, 10), title=labels["title"], xlabel=labels["hours"], ylabel=labels["sleep"])
    sleep = [4, 10]   # the bottom and the top of the frame: for every b on the slider the line stays inside it
    p.line([(-b - W[1] * s) / W[0] for s in sleep], sleep, "pred", 2.4)
    done = students()
    p.points([h for h, s, ok in done if ok], [s for h, s, ok in done if ok], "data", 6)
    p.points([h for h, s, ok in done if not ok], [s for h, s, ok in done if not ok], "data", 6, hollow=True)
    p.text(8.6, 9.5, labels["above"], "ink-2")
    p.text(1.6, 4.5, labels["below"], "ink-2")
    # beside it, for every b on the slider: the students on the wrong side (filled) and exactly on the line (open)
    bs = [-10 + 0.5 * i for i in range(9)]
    counts = [sides(v) for v in bs]
    m = chart.Panel((-10.2, -5.8), (-0.3, 3.3), title=labels["wrong"], xlabel="b", ylabel=labels["count"], xticks=[-10, -9, -8, -7, -6],
                    yticks=[0, 1, 2, 3])
    m.line([b, b], [-0.3, 3.3], "ink-3", 1.2)
    m.line(bs, [w for w, _ in counts], "loss", 2.0)
    m.points(bs, [w for w, _ in counts], "loss", 4.5)
    on = [(v, o) for v, (_, o) in zip(bs, counts) if o]   # only where some lie on the line, so no open dot hides a 0
    m.points([v for v, _ in on], [o for _, o in on], "loss", 4.5, hollow=True)
    m.text(on[0][0] + 0.25, on[0][1], labels["on"], "loss", "start")   # named at the first open dot: the title names the filled ones
    return chart.figure([p, m], alt=labels["alt"], width=330, height=300, drag=1)   # two fit side by side in a play block


# The lesson is done when these are correct. "on-boundary" is the choice question in the lesson.
REQUIRED = ["on-boundary", "probability", "hours-needed"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_P_START = "import numpy as np\n\n\ndef probability(x, w, b):\n    ...\n"
_P = "import numpy as np\n\n\ndef probability(x, w, b):\n    z = np.dot(w, x) + b\n    return 1 / (1 + np.exp(-z))"


def _p(body: str) -> str:
    return "import numpy as np\n\n\ndef probability(x, w, b):\n" + body


_H_HEAD = "import numpy as np\n\nw = np.array([1, 0.5])\nb = -8\nsleep = 7\n"
_H_START = _H_HEAD + "hours = ...\nhours\n"
_H = _H_HEAD + "hours = (-b - w[1] * sleep) / w[0]\nhours"


def _h(new: str) -> str:
    return _H.replace("hours = (-b - w[1] * sleep) / w[0]", new)


EXERCISES = {
    "probability": FunctionBox(
        name="probability",
        function="probability",
        start=_P_START,
        solution=_P,
        visible=[([7, 6], [1, 0.5], -8)],
        hidden=[([4, 8], [1, 0.5], -8), ([2, 9], [1, 0.5], -8), ([1, 2, 3], [0.5, -1, 0.2], 0.3), ([10, 0], [0.2, 5], -1)],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("no-sigmoid", wrong=_p("    return np.dot(w, x) + b"),
                    en="`{call}` gives `{got}`: the logit, which can be any number. Put it through the sigmoid to get a probability.",
                    nl="`{call}` geeft `{got}`: de logit, die elk getal kan zijn. Haal hem door de sigmoid om een kans te krijgen."),
            Mistake("no-bias", wrong=_p("    return 1 / (1 + np.exp(-np.dot(w, x)))"),
                    en="`{call}` gives `{got}`: the bias term is missing. The logit is the dot product plus `b`.",
                    nl="`{call}` geeft `{got}`: de bias term ontbreekt. De logit is het inproduct plus `b`."),
            Mistake("bias-outside", wrong=_p("    return 1 / (1 + np.exp(-np.dot(w, x))) + b"),
                    en="`{call}` gives about −7, not between 0 and 1. The bias term belongs inside the sigmoid: add `b` to the dot product first.",
                    nl="`{call}` geeft ongeveer −7, niet tussen 0 en 1. De bias term hoort binnen de sigmoid: tel `b` eerst bij het inproduct op."),
            Mistake("minus-part", wrong=_p("    return 1 / (1 + np.exp(-np.dot(w, x) + b))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the sigmoid of the dot product minus `b`. The logit is the dot product plus `b`, and the minus sign in `np.exp(-z)` goes in front of all of it: `-(np.dot(w, x) + b)`, with brackets.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de sigmoid van het inproduct min `b`. De logit is het inproduct plus `b`, en het minteken in `np.exp(-z)` hoort voor het geheel: `-(np.dot(w, x) + b)`, met haakjes."),
            Mistake("list-times", error=lambda e: isinstance(e, TypeError) and "can't multiply sequence" in str(e),
                    en="With `{call}`, your function stops with an error: `w` and `x` are lists here, and `*` does not multiply two lists number by number. `np.dot(w, x)` takes lists too.",
                    nl="Met `{call}` stopt je functie met een fout: `w` en `x` zijn hier lijsten, en `*` vermenigvuldigt twee lijsten niet getal voor getal. `np.dot(w, x)` neemt ook lijsten aan."),
            Mistake("no-minus", wrong=_p("    z = np.dot(w, x) + b\n    return 1 / (1 + np.exp(z))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the minus sign in `np.exp(-z)` is missing, so a higher logit gets a lower probability.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: het minteken in `np.exp(-z)` ontbreekt, dus krijgt een hogere logit een lagere kans."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the logit, the dot product plus `b`, and a `return` line with its sigmoid.",
                      "nl": "Vervang de `...` door de logit, het inproduct plus `b`, en een `return`-regel met de sigmoid daarvan."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the features, the weights and the bias term, and the `def` line of your function has a different number of names between its parentheses. Keep `(x, w, b)`.",
                          "nl": "De check roept `{call}` aan, met de features, de weights en de bias term, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x, w, b)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Is it the sigmoid of the dot product of `w` and `x`, plus `b`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Is het de sigmoid van het inproduct van `w` en `x`, plus `b`?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for any weights, any bias term and any number of features?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor alle weights, elke bias term en elk aantal features?"},
            "correct": {"en": "Your function is right in all {n} cases. The student with 7 hours of study and 6 of sleep gets 0.88: a pass. A student with 4 and 8 lands exactly on the decision boundary: 0.5.",
                        "nl": "Je functie klopt in alle {n} gevallen. De leerling met 7 uur leren en 6 uur slaap krijgt 0,88: geslaagd. Een leerling met 4 en 8 komt precies op de decision boundary: 0,5."},
        },
    ),
    "hours-needed": CodeBox(
        name="hours-needed",
        start=_H_START,
        solution=_H,
        keep=["b = -8"],
        # The check also runs the box with other weights: a number typed in by hand does not change with them.
        others=[{"w": Code("np.array([2, 0.5])"), "sleep": 9}],
        first=[_DOTS],
        mistakes=[
            Mistake("sign-b", wrong=_h("hours = (b - w[1] * sleep) / w[0]"),
                    en="Your last line gives `{got}`: negative hours. Moving `b` to the other side of `= 0` changes its sign: `-b`.",
                    nl="Je laatste regel geeft `{got}`: negatieve uren. Als je `b` naar de andere kant van `= 0` haalt, draait zijn teken om: `-b`."),
            Mistake("sign-sleep", wrong=_h("hours = (-b + w[1] * sleep) / w[0]"),
                    en="Your last line gives `{got}`: more sleep would then need more study, but sleep helps here. Moving `w[1] * sleep` to the other side of `= 0` changes its sign: `- w[1] * sleep`.",
                    nl="Je laatste regel geeft `{got}`: meer slaap zou dan meer leren vragen, maar slaap helpt hier. Als je `w[1] * sleep` naar de andere kant van `= 0` haalt, draait het teken om: `- w[1] * sleep`."),
            Mistake("weights-swapped", wrong=_h("hours = (-b - w[0] * sleep) / w[1]"),
                    en="Your last line gives `{got}`: the weights the wrong way round. `w[0]` belongs to the hours, `w[1]` to the sleep.",
                    nl="Je laatste regel geeft `{got}`: de weights andersom. `w[0]` hoort bij de uren leren, `w[1]` bij de slaap."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the hours of study that put a student with `sleep` hours of sleep exactly on the decision boundary.",
                      "nl": "Vervang de `...` door de uren leren die een leerling met `sleep` uur slaap precies op de decision boundary zetten."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other weights by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere weights."},
            "no_value": {"en": "The last line of the box shows nothing. Put `hours` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `hours` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. On the decision boundary the logit is 0: `w[0] * hours + w[1] * sleep + b = 0`. Did you solve that for `hours`?",
                      "nl": "Je laatste regel geeft `{got}`. Op de decision boundary is de logit 0: `w[0] * hours + w[1] * sleep + b = 0`. Heb je dat opgelost naar `hours`?"},
            "other": {"en": "Right for these weights. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out from `w`, `b` and `sleep` themselves, and divide by the weight of the hours?",
                      "nl": "Goed voor deze weights. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `w`, `b` en `sleep` zelf, en deel je door de weight van de uren?"},
            "other_error": {"en": "Right for these weights. But with {where}, your code stops with an error. Do you work it out from `w`, `b` and `sleep` themselves?",
                            "nl": "Goed voor deze weights. Maar met {where} stopt je code met een fout. Reken je het uit met `w`, `b` en `sleep` zelf?"},
            "correct": {"en": "A student who slept 7 hours needs 4.5 hours of study for a probability of exactly 0.5; with more, the model predicts a pass. The check also tried a model that counts each hour of study twice, for a student who slept 9 hours: then 1.75 hours are enough.",
                        "nl": "Een leerling die 7 uur sliep, moet 4,5 uur leren voor een kans van precies 0,5; met meer voorspelt het model dat de leerling slaagt. De check probeerde ook een model dat elk uur leren dubbel telt, voor een leerling die 9 uur sliep: dan is 1,75 uur genoeg."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("probability", _P_START, "empty", None),
    ("probability", _P, "correct", None),
    ("probability", _p("    return 1 / (1 + np.exp(-(np.dot(w, x) + b)))"), "correct", None),
    ("probability", _p("    z = b\n    for i in range(len(x)):\n        z = z + w[i] * x[i]\n    return 1 / (1 + np.exp(-z))"), "correct", None),
    ("probability", _p("    return np.dot(w, x) + b"), "wrong", "no-sigmoid"),
    ("probability", _p("    return 1 / (1 + np.exp(-np.dot(w, x)))"), "wrong", "no-bias"),
    ("probability", _p("    return 1 / (1 + np.exp(-np.dot(w, x))) + b"), "wrong", "bias-outside"),
    ("probability", _p("    z = np.dot(w, x) + b\n    return 1 / (1 + np.exp(z))"), "wrong", "no-minus"),
    ("probability", _p("    return 0.8807970779778823"), "almost", None),
    ("probability", _p("    z = np.dot(w, x) - b\n    return 1 / (1 + np.exp(-z))"), "wrong", "minus-part"),
    ("probability", _p("    return 1 / (1 + np.exp(-(np.sum(np.array(w) * np.array(x)) + b)))"), "correct", None),
    ("probability", _p("    return 1 / (1 + np.exp(-(sum(w * x) + b)))"), "error", "list-times"),
    ("hours-needed", _H_START, "empty", None),
    ("hours-needed", _H, "correct", None),
    ("hours-needed", _h("hours = -(b + w[1] * sleep) / w[0]"), "correct", None),
    ("hours-needed", _h("hours = (b - w[1] * sleep) / w[0]"), "wrong", "sign-b"),
    ("hours-needed", _h("hours = (-b + w[1] * sleep) / w[0]"), "wrong", "sign-sleep"),
    ("hours-needed", _h("hours = (-b - w[0] * sleep) / w[1]"), "wrong", "weights-swapped"),
    ("hours-needed", _h("hours = -b - w[1] * sleep"), "almost", "other-value"),
    ("hours-needed", _h("hours = 4.5"), "almost", "other-value"),
]
