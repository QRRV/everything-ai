# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-log-loss: the log loss of one example, and the mean log loss of five mails with the
formula that handles both classes at once. And the picture of the log loss and the squared residual of a spam mail."""
import math
import re

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake


def picture(p: float, labels: dict[str, str]) -> str:
    """For a mail that is spam (y = 1): the log loss −log(ŷ) and the squared residual (1 − ŷ)² for ŷ from 0.007 to 1,
    with a point on each at the ŷ of the slider. Drawn as SVG text, once for every stop of the slider
    (L.explore); dragging across the chart sets ŷ."""
    from everything_ai import chart
    ps = [0.007 + 0.993 * i / 150 for i in range(151)]
    c = chart.Panel((0, 1), (0, 5), title=labels["title"], xlabel=labels["p"], ylabel=labels["loss"])
    c.line(ps, [-math.log(v) for v in ps], "loss", 2.4)
    c.line(ps, [(1 - v) ** 2 for v in ps], "loss", 2.0, dashed=True)
    c.points([p, p], [-math.log(p), (1 - p) ** 2], "loss", 5.5)
    c.text(0.2, 3.3, labels["log"], "ink-2")
    # under the dashed line, from just right of the y-axis: clear of the axis and of both lines in both languages
    c.text(0.02, 0.18, labels["squared"], "ink-2", anchor="start")
    # 460 px wide, so its text stays 11 px or more in a play block at 700 px; 330 high, so the label fits under the dashed line
    return chart.figure([c], alt=labels["alt"], width=460, height=330, drag=0)


# The lesson is done when these are correct. "not-spam" is the choice question in the lesson.
REQUIRED = ["not-spam", "one-loss", "mean-loss"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_O_START = "import numpy as np\n\n\ndef one_loss(y, p):\n    ...\n"
_O = ("import numpy as np\n\n\ndef one_loss(y, p):\n    if y == 1:\n        return -np.log(p)\n    else:\n"
      "        return -np.log(1 - p)")


def _o(spam: str, other: str) -> str:
    return f"import numpy as np\n\n\ndef one_loss(y, p):\n    if y == 1:\n        return {spam}\n    else:\n        return {other}"


_M_HEAD = "import numpy as np\n\ny = np.array([1, 1, 0, 0, 1])\np = np.array([0.8, 0.4, 0.1, 0.3, 0.95])\n"
_M_START = _M_HEAD + "loss = ...\nloss\n"
_M = _M_HEAD + "loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))\nloss"


def _m(new: str) -> str:
    return _M.replace("loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))", new)


EXERCISES = {
    "one-loss": FunctionBox(
        name="one-loss",
        function="one_loss",
        start=_O_START,
        solution=_O,
        visible=[(1, 0.9)],
        hidden=[(0, 0.9), (1, 0.5), (0, 0.2), (1, 0.01)],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("no-minus", wrong=_o("np.log(p)", "np.log(1 - p)"),
                    en="`{call}` gives `{got}`: negative. The log of a probability below 1 is negative; the loss is minus that, so it is 0 or more.",
                    nl="`{call}` geeft `{got}`: negatief. De log van een kans onder 1 is negatief; de loss is min dat, dus 0 of meer."),
            Mistake("ignores-y", wrong=_o("-np.log(p)", "-np.log(p)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. With `y = 0` the mail is not spam: the model gave that `1 - p`. Take the log of the probability of what really happened.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Met `y = 0` is de mail geen spam: daar gaf het model `1 - p` aan. Neem de log van de kans op wat er echt gebeurde."),
            Mistake("swapped", wrong=_o("-np.log(1 - p)", "-np.log(p)"),
                    en="`{call}` gives `{got}`: a high probability for spam, on a mail that is spam, gets a large loss. With `y = 1`, the model gave `p` to what really happened, not `1 - p`.",
                    nl="`{call}` geeft `{got}`: een hoge kans op spam, bij een mail die spam is, krijgt een grote loss. Met `y = 1` gaf het model `p` aan wat er echt gebeurde, niet `1 - p`."),
            Mistake("squared", wrong=_o("(1 - p) ** 2", "p ** 2"),
                    en="`{call}` gives about 0.01: the squared residual of the mean squared error. Log loss takes minus the log of the probability of what really happened.",
                    nl="`{call}` geeft ongeveer 0,01: het gekwadrateerde residu van de mean squared error. Log loss neemt min de log van de kans op wat er echt gebeurde."),
            Mistake("base-10", wrong=_o("-np.log10(p)", "-np.log10(1 - p)"),
                    en="`{call}` gives `{got}`: the log with base 10. In this course, log means base $e$: `np.log`.",
                    nl="`{call}` geeft `{got}`: de log met grondtal 10. In deze course betekent log grondtal $e$: `np.log`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with an `if` on `y`: for spam, minus the log of `p`; otherwise, minus the log of `1 - p`.",
                      "nl": "Vervang de `...` door een `if` op `y`: bij spam min de log van `p`; anders min de log van `1 - p`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with a label and a probability, and the `def` line of your function has a different number of names between its parentheses. Keep `(y, p)`.",
                          "nl": "De check roept `{call}` aan, met een label en een kans, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(y, p)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Is it minus the log of the probability the model gave to what really happened?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Is het min de log van de kans die het model gaf aan wat er echt gebeurde?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for both labels, 1 and 0?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor beide labels, 1 en 0?"},
            "correct": {"en": "Your function is right in all {n} cases. A spam mail with 0.9 costs 0.11; the same 0.9 on a mail that is not spam costs 2.30. And 0.01 for a spam mail costs 4.61: confidently wrong is expensive.",
                        "nl": "Je functie klopt in alle {n} gevallen. Een spammail met 0,9 kost 0,11; dezelfde 0,9 bij een mail die geen spam is, kost 2,30. En 0,01 voor een spammail kost 4,61: zeker en fout is duur."},
        },
    ),
    "mean-loss": CodeBox(
        name="mean-loss",
        start=_M_START,
        solution=_M,
        keep=["y = np.array([1, 1, 0, 0, 1])"],
        # The check also runs the box with other probabilities: a number typed in by hand does not change with them.
        others=[{"p": Code("np.array([0.6, 0.7, 0.2, 0.1, 0.9])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("sum", wrong=_m("loss = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))"),
                    en="Your last line gives `{got}`: the losses added up. The log loss is their mean, so that it does not grow with the number of examples: `np.mean`.",
                    nl="Je laatste regel geeft `{got}`: de losses opgeteld. De log loss is hun gemiddelde, zodat hij niet groeit met het aantal voorbeelden: `np.mean`."),
            Mistake("no-minus", wrong=_m("loss = np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))"),
                    en="Your last line gives `{got}`: negative, because the logs of probabilities are negative. Put a minus in front.",
                    nl="Je laatste regel geeft `{got}`: negatief, omdat de logs van kansen negatief zijn. Zet er een min voor."),
            Mistake("spam-only", wrong=_m("loss = -np.mean(y * np.log(p))"),
                    en="Your last line gives `{got}`: only the spam mails count, because `y` is 0 for the others. Add `(1 - y) * np.log(1 - p)` for the mails that are not spam.",
                    nl="Je laatste regel geeft `{got}`: alleen de spammails tellen, omdat `y` 0 is voor de andere. Tel er `(1 - y) * np.log(1 - p)` bij op voor de mails die geen spam zijn."),
            Mistake("ignores-y", wrong=_m("loss = -np.mean(np.log(p))"),
                    en="Your last line gives `{got}`: minus the log of `p` for every mail, also for the mails that are not spam. For those, the probability of what really happened is `1 - p`.",
                    nl="Je laatste regel geeft `{got}`: min de log van `p` voor elke mail, ook voor de mails die geen spam zijn. Daar is de kans op wat er echt gebeurde `1 - p`."),
            Mistake("mse", wrong=_m("loss = np.mean((y - p) ** 2)"),
                    en="Your last line gives `{got}`: the mean squared error. The log loss takes minus the log of the probability of what really happened.",
                    nl="Je laatste regel geeft `{got}`: de mean squared error. De log loss neemt min de log van de kans op wat er echt gebeurde."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the mean log loss of the five mails, with the formula from *The idea*.",
                      "nl": "Vervang de `...` door de gemiddelde log loss van de vijf mails, met de formule uit *Het idee*."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other probabilities by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere kansen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `loss` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `loss` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you take, for each mail, the log of `p` or of `1 - p`, depending on `y`, then the mean, with a minus in front?",
                      "nl": "Je laatste regel geeft `{got}`. Neem je voor elke mail de log van `p` of van `1 - p`, afhankelijk van `y`, dan het gemiddelde, met een min ervoor?"},
            "other": {"en": "Right for these probabilities. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out from `y` and `p` themselves?",
                      "nl": "Goed voor deze kansen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `y` en `p` zelf?"},
            "other_error": {"en": "Right for these probabilities. But with {where}, your code stops with an error. Do you work it out from `y` and `p` themselves?",
                            "nl": "Goed voor deze kansen. Maar met {where} stopt je code met een fout. Reken je het uit met `y` en `p` zelf?"},
            "correct": {"en": "The mean log loss is about 0.33. Most of it comes from the second mail: spam, with only 0.4, a surprise of 0.92. The check also tried other probabilities.",
                        "nl": "De gemiddelde log loss is ongeveer 0,33. Het meeste komt van de tweede mail: spam, met maar 0,4, een verrassing van 0,92. De check probeerde ook andere kansen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("one-loss", _O_START, "empty", None),
    ("one-loss", _O, "correct", None),
    ("one-loss", "import numpy as np\n\n\ndef one_loss(y, p):\n    return -(y * np.log(p) + (1 - y) * np.log(1 - p))", "correct", None),
    ("one-loss", "import math\n\n\ndef one_loss(y, p):\n    if y == 0:\n        p = 1 - p\n    return -math.log(p)", "correct", None),
    ("one-loss", _o("np.log(p)", "np.log(1 - p)"), "wrong", "no-minus"),
    ("one-loss", _o("-np.log(p)", "-np.log(p)"), "wrong", "ignores-y"),
    ("one-loss", _o("-np.log(1 - p)", "-np.log(p)"), "wrong", "swapped"),
    ("one-loss", _o("(1 - p) ** 2", "p ** 2"), "wrong", "squared"),
    ("one-loss", _o("-np.log10(p)", "-np.log10(1 - p)"), "wrong", "base-10"),
    ("one-loss", "import numpy as np\n\n\ndef one_loss(y, p):\n    return 0.10536051565782628", "almost", None),
    ("mean-loss", _M_START, "empty", None),
    ("mean-loss", _M, "correct", None),
    ("mean-loss", _m("loss = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()"), "correct", None),
    ("mean-loss", _m("loss = np.mean(-np.log(y * p + (1 - y) * (1 - p)))"), "correct", None),
    ("mean-loss", _m("loss = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))"), "wrong", "sum"),
    ("mean-loss", _m("loss = np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))"), "wrong", "no-minus"),
    ("mean-loss", _m("loss = -np.mean(y * np.log(p))"), "wrong", "spam-only"),
    ("mean-loss", _m("loss = -np.mean(np.log(p))"), "wrong", "ignores-y"),
    ("mean-loss", _m("loss = np.mean((y - p) ** 2)"), "wrong", "mse"),
    ("mean-loss", _m("loss = 0.3305526074344948"), "almost", "other-value"),
]
