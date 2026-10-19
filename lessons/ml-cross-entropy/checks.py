# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-cross-entropy: the cross-entropy of one example from its probabilities and its label, and
the mean cross-entropy of many examples with one-hot labels. And the picture of three probabilities and the cross-entropy
as the logit of the right class moves on the slider."""
import math
import re

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake

OTHERS = (2.0, 0.0)   # the logits of class 0 (cat) and class 2 (bird) in the picture; the slider sets class 1 (dog)
RIGHT = 1             # the right class in the picture: the photo is of a dog


def probabilities(z_right: float) -> list[float]:
    """Softmax of (2, z_right, 0), with the largest logit taken off first."""
    zs = [OTHERS[0], z_right, OTHERS[1]]
    top = max(zs)
    es = [math.exp(z - top) for z in zs]
    return [e / sum(es) for e in es]


def loss(z_right: float) -> float:
    """The cross-entropy of the picture: minus the log of the probability of the right class."""
    return -math.log(probabilities(z_right)[RIGHT])


def picture(z_right: float, labels: dict[str, str], shown: list[str]) -> str:
    """One chart, three bars as high as the probabilities, the right class's drawn heavier; the other, the
    cross-entropy −log(p) against the probability p of the right class, with a point at the slider's. `shown` are the probabilities
    written out, in the lesson's language. Drawn as SVG text, once for every stop of the slider (L.explore)."""
    from everything_ai import chart
    ps = probabilities(z_right)
    bars = chart.Panel((-0.6, 2.6), (0, 1.12), title=labels["bars"], xlabel=labels["class"], ylabel=labels["p"],
                       xticks=[0, 1, 2])
    for i, p in enumerate(ps):
        bars.rect(i - 0.3, 0, i + 0.3, p, "pred")
        if i == RIGHT:   # the right class: a second outline, so it stands out without another colour
            bars.line([i - 0.3, i - 0.3, i + 0.3, i + 0.3], [0, p, p, 0], "pred", 3.2)
        bars.text(i, p + 0.05, shown[i], "ink" if i == RIGHT else "ink-2")
    xs = [0.005 + 0.995 * i / 150 for i in range(151)]
    curve = chart.Panel((0, 1), (0, 5.5), title=labels["curve"], xlabel=labels["p_right"], ylabel=labels["loss"])
    curve.line(xs, [-math.log(x) for x in xs], "loss", 2.4)
    curve.points([ps[RIGHT]], [-math.log(ps[RIGHT])], "loss", 5.5)
    return chart.figure([bars, curve], alt=labels["alt"], width=330, height=300)   # two fit side by side in a play block

# The lesson is done when these are correct. "no-idea" is the choice question in the lesson.
REQUIRED = ["no-idea", "cross-entropy", "batch-loss"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_E_HEAD = "import numpy as np\n\n\ndef cross_entropy(p, label):\n"
_E_START = _E_HEAD + "    ...\n"
_E = _E_HEAD + "    return -np.log(p[label])"


def _e(body: str) -> str:
    return _E_HEAD + "    return " + body


def _rounded(v):
    """A number to six decimals, so that an answer rounded to the four of the lesson never shows the same as the
    right one; anything else the way Python shows it."""
    try:
        text = f"{float(v):.6f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return repr(v)
    return "0" if text == "-0" else text


_B_HEAD = ("import numpy as np\n\nprobs = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4], [0.6, 0.1, 0.3]])\n"
           "labels = np.array([0, 1, 1, 2])\nonehot = np.eye(3)[labels]\n")
_B_START = _B_HEAD + "loss = ...\nloss\n"
_B = _B_HEAD + "loss = -(onehot * np.log(probs)).sum(axis=1).mean()\nloss"


def _b(new: str) -> str:
    return _B.replace("loss = -(onehot * np.log(probs)).sum(axis=1).mean()", new)


EXERCISES = {
    "cross-entropy": FunctionBox(
        name="cross-entropy",
        function="cross_entropy",
        start=_E_START,
        solution=_E,
        visible=[([0.7, 0.2, 0.1], 1)],
        hidden=[([0.1, 0.9], 0), ([1.0], 0), ([0.25, 0.25, 0.25, 0.25], 3), ([0.05, 0.15, 0.3, 0.5], 2)],
        show=_rounded,
        first=[_DOTS],
        mistakes=[
            Mistake("no-minus", wrong=_e("np.log(p[label])"),
                    en="`{call}` gives `{got}`: negative. The log of a probability below 1 is negative; the loss is minus that.",
                    nl="`{call}` geeft `{got}`: negatief. De log van een kans onder 1 is negatief; de loss is min dat."),
            Mistake("best-class", wrong=_e("-np.log(max(p))"),
                    en="`{call}` gives `{got}`: the surprise of the class with the highest probability. Cross-entropy looks at the right class, the label: `p[label]`.",
                    nl="`{call}` geeft `{got}`: de verrassing van de klasse met de hoogste kans. Cross-entropy kijkt naar de goede klasse, het label: `p[label]`."),
            Mistake("all-classes", wrong=_e("-np.log(np.array(p)).sum()"),
                    en="`{call}` gives `{got}`: the surprises of all classes added up. Only the right class counts: minus the log of `p[label]`.",
                    nl="`{call}` geeft `{got}`: de verrassingen van alle klassen opgeteld. Alleen de goede klasse telt: min de log van `p[label]`."),
            Mistake("one-minus", wrong=_e("-np.log(1 - p[label])"),
                    en="`{call}` gives `{got}`: minus the log of `1 - p[label]`, as for the label 0 in log loss. Here `p[label]` is already the probability of what really happened.",
                    nl="`{call}` geeft `{got}`: min de log van `1 - p[label]`, zoals bij het label 0 in log loss. Hier is `p[label]` al de kans op wat er echt gebeurde."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a `return` line: minus the log of the probability of the right class.",
                      "nl": "Vervang de `...` door een `return`-regel: min de log van de kans op de goede klasse."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the probabilities and the label, and the `def` line of your function has a different number of names between its parentheses. Keep `(p, label)`.",
                          "nl": "De check roept `{call}` aan, met de kansen en het label, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(p, label)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Is it minus the log of `p[label]`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Is het min de log van `p[label]`?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for any probabilities and any label?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor alle kansen en elk label?"},
            "correct": {"en": "Your function is right in all {n} cases. A model that is sure and right, `[1.0]`, costs 0. And with two classes it is log loss: `[0.1, 0.9]` with the label 0 costs 2.30, the log loss of a mail that is not spam but got 0.9 for spam.",
                        "nl": "Je functie klopt in alle {n} gevallen. Een model dat zeker is en gelijk heeft, `[1.0]`, kost 0. En met twee klassen is het log loss: `[0.1, 0.9]` met het label 0 kost 2,30, de log loss van een mail die geen spam is maar 0,9 voor spam kreeg."},
        },
    ),
    "batch-loss": CodeBox(
        name="batch-loss",
        start=_B_START,
        solution=_B,
        keep=["onehot = np.eye(3)[labels]"],
        # The check also runs the box with other labels: a number typed in by hand does not change with them.
        others=[{"labels": Code("np.array([2, 1, 0, 0])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("no-mean", wrong=_b("loss = -(onehot * np.log(probs)).sum()"),
                    en="Your last line gives `{got}`: the surprises of all four photos added up. The loss is their mean, so it does not grow with the number of examples: `.sum(axis=1).mean()`.",
                    nl="Je laatste regel geeft `{got}`: de verrassingen van alle vier de foto's opgeteld. De loss is hun gemiddelde, zodat hij niet groeit met het aantal voorbeelden: `.sum(axis=1).mean()`."),
            Mistake("per-photo", wrong=_b("loss = -(onehot * np.log(probs)).sum(axis=1)"),
                    en="Your last line gives the surprise of each photo, one number each. The loss is one number: their mean, `.mean()`.",
                    nl="Je laatste regel geeft de verrassing van elke foto, één getal per foto. De loss is één getal: hun gemiddelde, `.mean()`."),
            Mistake("columns", wrong=_b("loss = -(onehot * np.log(probs)).sum(axis=0).mean()"),
                    en="Your last line gives `{got}`: the surprises added up and divided by 3, the number of classes, as with `.sum(axis=0).mean()`. Add up along each row, `axis=1`, one number per photo, and take the mean over the photos.",
                    nl="Je laatste regel geeft `{got}`: de verrassingen opgeteld en gedeeld door 3, het aantal klassen, zoals met `.sum(axis=0).mean()`. Tel langs elke rij op, `axis=1`, één getal per foto, en neem het gemiddelde over de foto's."),
            Mistake("mean-all", wrong=_b("loss = -(onehot * np.log(probs)).mean()"),
                    en="Your last line gives `{got}`: the mean of all 12 numbers, also the 0s of the other classes, so it is 3 times too small. First add up each row, `.sum(axis=1)`, then take the mean over the photos.",
                    nl="Je laatste regel geeft `{got}`: het gemiddelde van alle 12 getallen, ook de nullen van de andere klassen, dus 3 keer te klein. Tel eerst elke rij op, `.sum(axis=1)`, en neem dan het gemiddelde over de foto's."),
            Mistake("no-minus", wrong=_b("loss = (onehot * np.log(probs)).sum(axis=1).mean()"),
                    en="Your last line gives `{got}`: negative, because the logs of probabilities are negative. Put a minus in front.",
                    nl="Je laatste regel geeft `{got}`: negatief, omdat de logs van kansen negatief zijn. Zet er een min voor."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the mean cross-entropy of the four photos, with `onehot` and `np.log(probs)`.",
                      "nl": "Vervang de `...` door de gemiddelde cross-entropy van de vier foto's, met `onehot` en `np.log(probs)`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other labels by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere labels."},
            "no_value": {"en": "The last line of the box shows nothing. Put `loss` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `loss` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you multiply `onehot` by `np.log(probs)`, add up each row, take the mean, and put a minus in front?",
                      "nl": "Je laatste regel geeft `{got}`. Vermenigvuldig je `onehot` met `np.log(probs)`, tel je elke rij op, neem je het gemiddelde, en zet je er een min voor?"},
            "other": {"en": "Right for these labels. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out from `onehot` and `probs` themselves?",
                      "nl": "Goed voor deze labels. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `onehot` en `probs` zelf?"},
            "other_error": {"en": "Right for these labels. But with {where}, your code stops with an error. Do you work it out from `onehot` and `probs` themselves?",
                            "nl": "Goed voor deze labels. Maar met {where} stopt je code met een fout. Reken je het uit met `onehot` en `probs` zelf?"},
            "correct": {"en": "The mean cross-entropy is about 0.83. The second photo costs least, 0.22, with 0.8 for its right class; the last two cost most, 1.20 each, with only 0.3. The check also tried other labels.",
                        "nl": "De gemiddelde cross-entropy is ongeveer 0,83. De tweede foto kost het minst, 0,22, met 0,8 voor zijn goede klasse; de laatste twee kosten het meest, elk 1,20, met maar 0,3. De check probeerde ook andere labels."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("cross-entropy", _E_START, "empty", None),
    ("cross-entropy", _E, "correct", None),
    ("cross-entropy", "import math\n\n\ndef cross_entropy(p, label):\n    return -math.log(p[label])", "correct", None),
    ("cross-entropy", _E_HEAD + "    onehot = np.eye(len(p))[label]\n    return -(onehot * np.log(p)).sum()", "correct", None),
    ("cross-entropy", _e("np.log(p[label])"), "wrong", "no-minus"),
    ("cross-entropy", _e("-np.log(max(p))"), "wrong", "best-class"),
    ("cross-entropy", _e("-np.log(np.array(p)).sum()"), "wrong", "all-classes"),
    ("cross-entropy", _e("-np.log(1 - p[label])"), "wrong", "one-minus"),
    ("cross-entropy", _e("-np.log(p[label + 1])"), "wrong", None),
    ("cross-entropy", _e("round(-np.log(p[label]), 4)"), "wrong", None),
    ("batch-loss", _B_START, "empty", None),
    ("batch-loss", _B, "correct", None),
    ("batch-loss", _b("loss = np.mean(-np.sum(onehot * np.log(probs), axis=1))"), "correct", None),
    ("batch-loss", _b("loss = -(onehot * np.log(probs)).sum() / len(labels)"), "correct", None),
    ("batch-loss", _b("loss = -(onehot * np.log(probs)).sum()"), "wrong", "no-mean"),
    ("batch-loss", _b("loss = -(onehot * np.log(probs)).sum(axis=1)"), "wrong", "per-photo"),
    ("batch-loss", _b("loss = -(onehot * np.log(probs)).sum(axis=0).mean()"), "wrong", "columns"),
    ("batch-loss", _b("loss = -(onehot * np.log(probs)).sum() / 3"), "wrong", "columns"),
    ("batch-loss", _b("loss = -(onehot * np.log(probs)).mean()"), "wrong", "mean-all"),
    ("batch-loss", _b("loss = (onehot * np.log(probs)).sum(axis=1).mean()"), "wrong", "no-minus"),
    ("batch-loss", _b("loss = 0.8310590851315067"), "almost", "other-value"),
]
