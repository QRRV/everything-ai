# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-sigmoid: the sigmoid as a function, and the way back from a probability to its logit.
And the picture of the sigmoid with a point at the logit on the slider."""
import math
import re

from everything_ai.check import CodeBox, FunctionBox, Mistake


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def picture(z: float, labels: dict[str, str]) -> str:
    """The sigmoid for z from −6 to 6, with the lines at 0, 0.5 and 1, and a point at z. Drawn as SVG text, once
    for every stop of the slider (L.explore); dragging across the chart sets z."""
    from everything_ai import chart
    zs = [-6 + 12 * i / 120 for i in range(121)]
    p = chart.Panel((-6, 6), (-0.05, 1.05), title=labels["title"], xlabel=labels["z"], ylabel=labels["p"])
    for level in (0, 0.5, 1):
        p.line([-6, 6], [level, level], "ink-3", 1.2, dashed=True)
    p.line(zs, [sigmoid(v) for v in zs], "pred")
    p.line([z, z], [0, sigmoid(z)], "ink-3", 1.2)
    p.points([z], [sigmoid(z)], "pred", 5.5)
    return chart.figure([p], alt=labels["alt"], width=460, height=290, drag=0)   # its text stays 11 px or more in a play block at 700 px


# The lesson is done when these are correct. "minus-two" is the choice question in the lesson.
REQUIRED = ["minus-two", "sigmoid", "back-to-logit"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_S_START = "import numpy as np\n\n\ndef sigmoid(z):\n    ...\n"
_S = "import numpy as np\n\n\ndef sigmoid(z):\n    return 1 / (1 + np.exp(-z))"


def _s(body: str) -> str:
    return "import numpy as np\n\n\ndef sigmoid(z):\n    return " + body


_L_HEAD = "import numpy as np\n\np = 0.9\n"
_L_START = _L_HEAD + "z = ...\nz\n"
_L = _L_HEAD + "z = np.log(p / (1 - p))\nz"


def _l(new: str) -> str:
    return _L.replace("z = np.log(p / (1 - p))", new)


EXERCISES = {
    "sigmoid": FunctionBox(
        name="sigmoid",
        function="sigmoid",
        start=_S_START,
        solution=_S,
        visible=[(0,)],
        hidden=[(2,), (-3,), (10,), (-1.5,)],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("no-minus", wrong=_s("1 / (1 + np.exp(z))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function gives σ(−z), which is 1 − σ(z), the probability of the other class. A higher logit then gets a lower probability. Is the top of the fraction 1, and is there a minus sign in `np.exp(-z)`?",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je functie geeft σ(−z), en dat is 1 − σ(z), de kans op de andere klasse. Een hogere logit krijgt dan een lagere kans. Is de bovenkant van de breuk 1, en staat er een minteken in `np.exp(-z)`?"),
            Mistake("no-brackets", wrong=_s("1 / 1 + np.exp(-z)"),
                    en="`{call}` gives `{got}`, above 1: your function gives 1 + `np.exp(-z)`, the bottom of the fraction, and not 1 divided by it. In `1 / 1 + np.exp(-z)`, Python divides first, 1 / 1, and then adds. Put parentheses around the bottom: `1 / (1 + np.exp(-z))`.",
                    nl="`{call}` geeft `{got}`, boven 1: je functie geeft 1 + `np.exp(-z)`, de onderkant van de breuk, en niet 1 gedeeld door die onderkant. In `1 / 1 + np.exp(-z)` deelt Python eerst, 1 / 1, en telt daarna op. Zet haakjes om de onderkant: `1 / (1 + np.exp(-z))`."),
            Mistake("no-one", wrong=_s("1 / np.exp(-z)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. Without the 1 in the bottom, your function gives `np.exp(z)`: `sigmoid(10)` would give about 22,026, far above 1, and that cannot be a probability. The bottom is 1 plus `np.exp(-z)`: `1 / (1 + np.exp(-z))`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Zonder de 1 in de onderkant geeft je functie `np.exp(z)`: `sigmoid(10)` zou ongeveer 22.026 geven, ver boven 1, en dat kan geen kans zijn. De onderkant is 1 plus `np.exp(-z)`: `1 / (1 + np.exp(-z))`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a `return` line that gives back the sigmoid of `z`.",
                      "nl": "Vervang de `...` door een `return`-regel die de sigmoid van `z` teruggeeft."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with one number, and the `def` line of your function has a different number of names between its parentheses. Keep `(z)`.",
                          "nl": "De check roept `{call}` aan, met één getal, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(z)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Is it 1 divided by 1 plus `np.exp(-z)`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Is het 1 gedeeld door 1 plus `np.exp(-z)`?"},
            "almost": {"en": "Right for 0. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for every logit, positive and negative?",
                       "nl": "Goed voor 0. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elke logit, positief en negatief?"},
            "correct": {"en": "Your function is right in all {n} cases. A logit of 0 gives 0.5, 10 gives 0.99995 and −3 gives 0.047: always between 0 and 1, and the higher the logit, the higher the probability.",
                        "nl": "Je functie klopt in alle {n} gevallen. Een logit van 0 geeft 0,5, 10 geeft 0,99995 en −3 geeft 0,047: altijd tussen 0 en 1, en hoe hoger de logit, hoe hoger de kans."},
        },
    ),
    "back-to-logit": CodeBox(
        name="back-to-logit",
        start=_L_START,
        solution=_L,
        keep=["p = 0.9"],
        # The check also runs the box with another probability: a number typed in by hand does not change with it.
        others=[{"p": 0.2}],
        first=[_DOTS],
        mistakes=[
            Mistake("odds", wrong=_l("z = p / (1 - p)"),
                    en="Your last line gives about 9: the odds, p / (1 − p), how many times as likely the class is as not the class. The logit is the log of that.",
                    nl="Je laatste regel geeft ongeveer 9: de odds, p / (1 − p), hoeveel keer zo waarschijnlijk de klasse is als niet de klasse. De logit is de log daarvan."),
            Mistake("log-p", wrong=_l("z = np.log(p)"),
                    en="Your last line gives `{got}`: the log of p alone. The logit is the log of p divided by 1 − p.",
                    nl="Je laatste regel geeft `{got}`: alleen de log van p. De logit is de log van p gedeeld door 1 − p."),
            Mistake("upside-down", wrong=_l("z = np.log((1 - p) / p)"),
                    en="Your last line gives `{got}`: the right size, but negative, because the fraction is upside down. With p above 0.5 the logit is positive: `p / (1 - p)`.",
                    nl="Je laatste regel geeft `{got}`: de goede grootte, maar negatief, omdat de breuk op zijn kop staat. Met p boven 0,5 is de logit positief: `p / (1 - p)`."),
            Mistake("brackets", wrong=_l("z = np.log(p / 1 - p)"),
                    en="Your last line gives `{got}`, the log of 0: Python divides first, p / 1, and then subtracts p, which leaves 0. Put parentheses around the bottom: `p / (1 - p)`.",
                    nl="Je laatste regel geeft `{got}`, de log van 0: Python deelt eerst, p / 1, en trekt daarna p af, en dan blijft er 0 over. Zet haakjes om de onderkant: `p / (1 - p)`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the logit of `p`: the log of `p` divided by `1 - p`.",
                      "nl": "Vervang de `...` door de logit van `p`: de log van `p` gedeeld door `1 - p`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another probability by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere kans."},
            "no_value": {"en": "The last line of the box shows nothing. Put `z` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `z` op de laatste regel, zodat de check hem kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Is it the log of `p` divided by `1 - p`?",
                      "nl": "Je laatste regel geeft `{got}`. Is het de log van `p` gedeeld door `1 - p`?"},
            "other": {"en": "Right for 0.9. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out from `p` itself?",
                      "nl": "Goed voor 0,9. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `p` zelf?"},
            "other_error": {"en": "Right for 0.9. But with {where}, your code stops with an error. Do you work it out from `p` itself?",
                            "nl": "Goed voor 0,9. Maar met {where} stopt je code met een fout. Reken je het uit met `p` zelf?"},
            "correct": {"en": "A probability of 0.9 needs a logit of about 2.2, and the sigmoid of 2.2 gives 0.9 back. The check also tried 0.2: a logit of about −1.39, negative, because 0.2 is below 0.5.",
                        "nl": "Bij een kans van 0,9 hoort een logit van ongeveer 2,2, en de sigmoid van 2,2 geeft 0,9 terug. De check probeerde ook 0,2: een logit van ongeveer −1,39, negatief, omdat 0,2 onder 0,5 ligt."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("sigmoid", _S_START, "empty", None),
    ("sigmoid", _S, "correct", None),
    ("sigmoid", "import math\n\n\ndef sigmoid(z):\n    return 1 / (1 + math.exp(-z))", "correct", None),
    ("sigmoid", _s("np.exp(z) / (np.exp(z) + 1)"), "correct", None),
    ("sigmoid", _s("1 / (1 + np.exp(z))"), "wrong", "no-minus"),
    ("sigmoid", _s("np.exp(-z) / (1 + np.exp(-z))"), "wrong", "no-minus"),
    ("sigmoid", _s("1 / 1 + np.exp(-z)"), "wrong", "no-brackets"),
    ("sigmoid", _s("1 / np.exp(-z)"), "wrong", "no-one"),
    ("sigmoid", _s("np.exp(z)"), "wrong", "no-one"),
    ("sigmoid", _s("1 + np.exp(-z)"), "wrong", "no-brackets"),
    ("sigmoid", _s("1 - 1 / (1 + np.exp(-z))"), "wrong", "no-minus"),
    ("sigmoid", _s("(1 + np.tanh(z / 2)) / 2"), "correct", None),
    ("sigmoid", _s("0.5"), "almost", None),
    ("back-to-logit", _L_START, "empty", None),
    ("back-to-logit", _L, "correct", None),
    ("back-to-logit", _l("z = np.log(p) - np.log(1 - p)"), "correct", None),
    ("back-to-logit", _l("z = -np.log(1 / p - 1)"), "correct", None),
    ("back-to-logit", _l("z = p / (1 - p)"), "wrong", "odds"),
    ("back-to-logit", _l("z = np.log(p)"), "wrong", "log-p"),
    ("back-to-logit", _l("z = np.log((1 - p) / p)"), "wrong", "upside-down"),
    ("back-to-logit", _l("z = np.log(p / 1 - p)"), "wrong", "brackets"),
    ("back-to-logit", _l("z = np.log(0.9 / 0.1)"), "almost", "other-value"),
    ("back-to-logit", _L.replace("p = 0.9", "p = 0.8"), "wrong", "changed-line"),
]
