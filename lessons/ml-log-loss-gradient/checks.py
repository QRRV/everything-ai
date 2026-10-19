# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-log-loss-gradient: the gradient of the log loss of one example, and the mean gradient over
the students of ml-vocabulary. And the picture of the sigmoid with the line of its slope at the z on the slider."""
import math
import re

from everything_ai.check import FunctionBox, CodeBox, Mistake


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def picture(z: float, labels: dict[str, str]) -> str:
    """The sigmoid for z from −6 to 6, with a point at z and a short line with the slope σ(z)(1 − σ(z)) there.
    Drawn as SVG text, once for every stop of the slider (L.explore); dragging across the chart sets z."""
    from everything_ai import chart
    zs = [-6 + 12 * i / 120 for i in range(121)]
    p = sigmoid(z)
    slope = p * (1 - p)
    c = chart.Panel((-6, 6), (-0.05, 1.05), title=labels["title"], xlabel=labels["z"], ylabel=labels["p"])
    c.line(zs, [sigmoid(v) for v in zs], "pred")
    c.line([z - 1.5, z + 1.5], [p - 1.5 * slope, p + 1.5 * slope], "grad", 2.6)
    c.points([z], [p], "pred", 5.5)
    return chart.figure([c], alt=labels["alt"], width=460, height=290, drag=0)   # its text stays 11 px or more in a play block at 700 px


# The lesson is done when these are correct. "not-spam-slope" is the choice question in the lesson.
REQUIRED = ["not-spam-slope", "one-gradient", "mean-gradient"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_G_HEAD = "import numpy as np\n\n\ndef one_gradient(x, y, w, b):\n    x = np.array(x)\n    p = 1 / (1 + np.exp(-(np.dot(w, x) + b)))\n"
_G_START = _G_HEAD + "    ...\n"
_G = _G_HEAD + "    return (p - y) * x"


def _g(back: str) -> str:
    return _G_HEAD + "    return " + back


_M_HEAD = ("import numpy as np\n\nhours = np.array([2, 8, 5, 1, 6, 3, 9])\nsleep = np.array([6, 7, 8, 5, 6, 8, 8])\n"
           "passed = np.array([0, 1, 1, 0, 1, 0, 1])\nw = np.array([1, 0.5])\nb = -8\n"
           "p = 1 / (1 + np.exp(-(w[0] * hours + w[1] * sleep + b)))\n")
_M_START = _M_HEAD + "grad = ...\ngrad\n"
_M_LINE = "grad = np.array([np.mean((p - passed) * hours), np.mean((p - passed) * sleep), np.mean(p - passed)])"
_M = _M_HEAD + _M_LINE + "\ngrad"


def _m(new: str) -> str:
    return _M.replace(_M_LINE, new)


EXERCISES = {
    "one-gradient": FunctionBox(
        name="one-gradient",
        function="one_gradient",
        start=_G_START,
        solution=_G,
        visible=[([2, 1], 1, [0.5, -1], 0)],
        hidden=[([1, 3], 0, [0.2, 0.1], -0.1), ([4, 1], 1, [-0.5, 2], 1), ([1, 1, 1], 0, [1, 1, 1], 0)],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("sign", wrong=_g("(y - p) * x"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the same size, the other sign. The gradient starts from the predicted probability minus the label, `p - y`, not `y - p`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dezelfde grootte, het andere teken. De gradient begint met de voorspelde kans min het label, `p - y`, niet `y - p`."),
            Mistake("no-x", wrong=_g("p - y"),
                    en="`{call}` gives `{got}`: one number, the slope for the bias term. Each weight gets its own slope: multiply by its feature, `* x`.",
                    nl="`{call}` geeft `{got}`: één getal, de helling voor de bias term. Elke weight krijgt een eigen helling: vermenigvuldig met zijn feature, `* x`."),
            Mistake("mse-chain", wrong=_g("(p - y) * p * (1 - p) * x"),
                    en="`{call}` gives `{got}`: with the slope of the sigmoid, `p * (1 - p)`, still in it. For log loss it cancels against the slope of the log: what is left is `(p - y) * x`.",
                    nl="`{call}` geeft `{got}`: met de helling van de sigmoid, `p * (1 - p)`, er nog in. Bij log loss valt die weg tegen de helling van de log: wat overblijft is `(p - y) * x`."),
            Mistake("no-sigmoid", wrong=_g("(np.dot(w, x) + b - y) * x"),
                    en="`{call}` gives `{got}`: the logit minus the label. The gradient uses the probability `p`, after the sigmoid.",
                    nl="`{call}` geeft `{got}`: de logit min het label. De gradient gebruikt de kans `p`, na de sigmoid."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a `return` line that gives back the gradient for the weights: the probability minus the label, times the features.",
                      "nl": "Vervang de `...` door een `return`-regel die de gradient voor de weights teruggeeft: de kans min het label, keer de features."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the features, the label, the weights and the bias term, and the `def` line of your function has a different number of names between its parentheses. Keep `(x, y, w, b)`.",
                          "nl": "De check roept `{call}` aan, met de features, het label, de weights en de bias term, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x, y, w, b)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Is it the predicted probability minus the label, times the features?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Is het de voorspelde kans min het label, keer de features?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for both labels and any number of features?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor beide labels en elk aantal features?"},
            "correct": {"en": "Your function is right in all {n} cases. For the spam mail with probability 0.5, the gradient is −1 and −0.5: raise both weights a little, and the loss goes down.",
                        "nl": "Je functie klopt in alle {n} gevallen. Voor de spammail met kans 0,5 is de gradient −1 en −0,5: maak beide weights een beetje groter, en de loss daalt."},
        },
    ),
    "mean-gradient": CodeBox(
        name="mean-gradient",
        start=_M_START,
        solution=_M,
        keep=["passed = np.array([0, 1, 1, 0, 1, 0, 1])", "w = np.array([1, 0.5])", "b = -8", "p = 1 / (1 + np.exp(-(w[0] * hours + w[1] * sleep + b)))"],
        # The check also runs the box with another bias term: a number typed in by hand does not change with it.
        others=[{"b": -6}],
        first=[_DOTS],
        mistakes=[
            Mistake("sign", wrong=_m("grad = np.array([np.mean((passed - p) * hours), np.mean((passed - p) * sleep), np.mean(passed - p)])"),
                    en="Your last line gives `{got}`: the right sizes, the wrong signs. Take the probability minus the label: `p - passed`.",
                    nl="Je laatste regel geeft `{got}`: de goede groottes, de verkeerde tekens. Neem de kans min het label: `p - passed`."),
            Mistake("sum", wrong=_m("grad = np.array([np.sum((p - passed) * hours), np.sum((p - passed) * sleep), np.sum(p - passed)])"),
                    en="Your last line gives `{got}`: the gradients of the students added up, 7 times too big. The log loss is a mean, so its gradient is a mean too: `np.mean`.",
                    nl="Je laatste regel geeft `{got}`: de gradients van de leerlingen opgeteld, 7 keer te groot. De log loss is een gemiddelde, dus zijn gradient ook: `np.mean`."),
            Mistake("no-features", wrong=_m("grad = np.array([np.mean(p - passed), np.mean(p - passed), np.mean(p - passed)])"),
                    en="Your last line gives `{got}`: three times the slope for the bias term. The weight of the hours gets `p - passed` times `hours`, the weight of the sleep `p - passed` times `sleep`.",
                    nl="Je laatste regel geeft `{got}`: drie keer de helling voor de bias term. De weight van de uren krijgt `p - passed` keer `hours`, de weight van de slaap `p - passed` keer `sleep`."),
            Mistake("mse-chain", wrong=_m("grad = np.array([np.mean((p - passed) * p * (1 - p) * hours), np.mean((p - passed) * p * (1 - p) * sleep), np.mean((p - passed) * p * (1 - p))])"),
                    en="Your last line gives `{got}`: with the slope of the sigmoid, `p * (1 - p)`, still in it. For log loss it cancels: leave it out.",
                    nl="Je laatste regel geeft `{got}`: met de helling van de sigmoid, `p * (1 - p)`, er nog in. Bij log loss valt die weg: laat hem weg."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with an array of three slopes: for the weight of the hours, the weight of the sleep and the bias term.",
                      "nl": "Vervang de `...` door een array met drie hellingen: voor de weight van de uren, de weight van de slaap en de bias term."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another bias term by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere bias term."},
            "no_value": {"en": "The last line of the box shows nothing. Put `grad` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `grad` op de laatste regel, zodat de check hem kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Is each slope the mean of `p - passed` times its feature, and the last one the mean of `p - passed` itself?",
                      "nl": "Je laatste regel geeft `{got}`. Is elke helling het gemiddelde van `p - passed` keer zijn feature, en de laatste het gemiddelde van `p - passed` zelf?"},
            "other": {"en": "Right for this model. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out from `p` and `passed` themselves?",
                      "nl": "Goed voor dit model. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `p` en `passed` zelf?"},
            "other_error": {"en": "Right for this model. But with {where}, your code stops with an error. Do you work it out from `p` and `passed` themselves?",
                            "nl": "Goed voor dit model. Maar met {where} stopt je code met een fout. Reken je het uit met `p` en `passed` zelf?"},
            "correct": {"en": "The gradient is about (−0.33, −0.22, −0.04): all three negative, so raising both weights and the bias term a little lowers the log loss. The check also tried b = −6: then all three are positive.",
                        "nl": "De gradient is ongeveer (−0,33; −0,22; −0,04): alle drie negatief, dus de weights en de bias term een beetje groter maken verlaagt de log loss. De check probeerde ook b = −6: dan zijn alle drie positief."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("one-gradient", _G_START, "empty", None),
    ("one-gradient", _G, "correct", None),
    ("one-gradient", _G_HEAD + "    error = p - y\n    return error * x", "correct", None),
    ("one-gradient", _G_HEAD + "    grad = []\n    for xi in x:\n        grad.append((p - y) * xi)\n    return np.array(grad)", "correct", None),
    ("one-gradient", _g("(y - p) * x"), "wrong", "sign"),
    ("one-gradient", _g("p - y"), "wrong", "no-x"),
    ("one-gradient", _g("(p - y) * p * (1 - p) * x"), "wrong", "mse-chain"),
    ("one-gradient", _g("(np.dot(w, x) + b - y) * x"), "wrong", "no-sigmoid"),
    ("mean-gradient", _M_START, "empty", None),
    ("mean-gradient", _M, "correct", None),
    ("mean-gradient", _m("error = p - passed\ngrad = np.array([(error * hours).mean(), (error * sleep).mean(), error.mean()])"), "correct", None),
    ("mean-gradient", _m("grad = np.array([np.mean((passed - p) * hours), np.mean((passed - p) * sleep), np.mean(passed - p)])"), "wrong", "sign"),
    ("mean-gradient", _m("grad = np.array([np.sum((p - passed) * hours), np.sum((p - passed) * sleep), np.sum(p - passed)])"), "wrong", "sum"),
    ("mean-gradient", _m("grad = np.array([np.mean(p - passed), np.mean(p - passed), np.mean(p - passed)])"), "wrong", "no-features"),
    ("mean-gradient", _m("grad = np.array([np.mean((p - passed) * p * (1 - p) * hours), np.mean((p - passed) * p * (1 - p) * sleep), np.mean((p - passed) * p * (1 - p))])"), "wrong", "mse-chain"),
    ("mean-gradient", _m("grad = np.array([-0.3343465980433783, -0.21898385695076808, -0.03521909817678232])"), "almost", "other-value"),
]
