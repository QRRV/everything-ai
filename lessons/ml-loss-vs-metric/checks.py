# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-loss-vs-metric: the accuracy of a model for many classes, the share of its predictions that
are right. And the picture of the log loss and the accuracy of a pass-or-fail model as its bias term moves on the
slider: the loss is smooth, the accuracy a staircase."""
import math
import re

from everything_ai.check import Code, CodeBox, Mistake

HOURS = [1, 2, 2.5, 3, 4, 4.5, 5, 6, 7, 8]   # ten students: hours of study, and whether they passed
PASSED = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]
W = 1.0                                    # the weight for the hours stays fixed; the slider moves b
BS = [round(-8 + 0.1 * k, 1) for k in range(81)]


def _probabilities(b: float) -> list[float]:
    return [1 / (1 + math.exp(-(W * h + b))) for h in HOURS]


def log_loss(b: float) -> float:
    return -sum(y * math.log(p) + (1 - y) * math.log(1 - p) for y, p in zip(PASSED, _probabilities(b))) / len(HOURS)


def right(b: float) -> int:
    """How many of the ten students the model puts on the right side: a probability of 0.5 or more means passed."""
    return sum((p >= 0.5) == (y == 1) for y, p in zip(PASSED, _probabilities(b)))


def picture(b: float, labels: dict[str, str]) -> str:
    """The log loss for every b, smooth, and the accuracy for every b, a staircase, each with a point at the slider's
    b. Drawn as SVG text, once for every stop of the slider (L.explore); dragging across the log loss sets b."""
    from everything_ai import chart
    loss = chart.Panel((-8, 0), (0.5, 1.75), title=labels["loss_title"], xlabel="b", ylabel=labels["loss"])
    loss.line(BS, [log_loss(v) for v in BS], "loss", 2.4)
    loss.points([b], [log_loss(b)], "param", 5.5)   # the b you set, as in Fit a line by hand
    acc = chart.Panel((-8, 0), (0, 1.05), title=labels["acc_title"], xlabel="b", ylabel=labels["acc"])
    xs, ys = [], []
    for v in BS:   # a staircase: flat between the values of b where a student crosses 0.5, and straight up or down there
        a = right(v) / len(HOURS)
        if ys and a != ys[-1]:
            xs.append(v)
            ys.append(ys[-1])
        xs.append(v)
        ys.append(a)
    acc.line(xs, ys, "ink", 2.4)
    acc.points([b], [right(b) / len(HOURS)], "ink", 5.5)
    return chart.figure([loss, acc], alt=labels["alt"], width=330, height=280, drag=0)   # two fit side by side in a play block


# The lesson is done when these are correct. "no-slope" and "which-model" are the choice questions in the lesson.
REQUIRED = ["no-slope", "accuracy", "which-model"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_A_HEAD = ("import numpy as np\n\nprobs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.2, 0.5, 0.3], [0.45, 0.35, 0.2], "
           "[0.3, 0.3, 0.4], [0.1, 0.8, 0.1]])\nlabels = np.array([0, 2, 1, 1, 0, 1])\n")
_A_START = _A_HEAD + "accuracy = ...\naccuracy\n"
_A_LINE = "accuracy = np.mean(probs.argmax(axis=1) == labels)"
_A = _A_HEAD + _A_LINE + "\naccuracy"


def _a(new: str) -> str:
    return _A.replace(_A_LINE, new)


EXERCISES = {
    "accuracy": CodeBox(
        name="accuracy",
        start=_A_START,
        solution=_A,
        # The check also runs the box with other labels: a number typed in by hand does not change with them.
        others=[{"labels": Code("np.array([0, 1, 1, 0, 2, 1])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("count", wrong=_a("accuracy = np.sum(probs.argmax(axis=1) == labels)"),
                    en="Your last line gives `{got}`: how many photos the model gets right. Accuracy is the share of them: the mean, `np.mean`, not the sum.",
                    nl="Je laatste regel geeft `{got}`: hoeveel foto's het model goed heeft. Accuracy is het deel ervan: het gemiddelde, `np.mean`, niet de som."),
            # Both stop Python with shapes that do not fit; "columns" is told apart by the argmax over axis 0 the learner typed.
            Mistake("columns", syntax=r"argmax\(\s*(?:probs\s*,\s*)?(?:axis\s*=\s*)?0\s*\)",
                    en="Python stops: `probs.argmax(axis=0)` gives one number per class, a column, while `labels` has one per photo. The predicted class of each photo is the largest along its row: `axis=1`.",
                    nl="Python stopt: `probs.argmax(axis=0)` geeft één getal per klasse, een kolom, terwijl `labels` er één per foto heeft. De voorspelde klasse van elke foto is de grootste langs zijn rij: `axis=1`."),
            Mistake("shapes", error=lambda e: isinstance(e, ValueError) and "broadcast" in str(e),
                    en="Python stops: `labels` has one number per photo, and what you compare it with has a different shape. `probs` has a whole row of probabilities per photo; compare `labels` with one class per photo, the most likely one.",
                    nl="Python stopt: `labels` heeft één getal per foto, en wat je ermee vergelijkt heeft een andere shape. `probs` heeft een hele rij kansen per foto; vergelijk `labels` met één klasse per foto, de waarschijnlijkste."),
            Mistake("least", wrong=_a("accuracy = np.mean(probs.argmin(axis=1) == labels)"),
                    en="Your last line gives `{got}`: `argmin` picks the class the model thinks least likely. The prediction is the most likely class: `argmax`.",
                    nl="Je laatste regel geeft `{got}`: `argmin` kiest de klasse die het model het minst waarschijnlijk vindt. De voorspelling is de waarschijnlijkste klasse: `argmax`."),
            Mistake("sure", wrong=_a("accuracy = np.mean(probs.max(axis=1))"),
                    en="Your last line gives `{got}`: the mean of the largest probabilities, how sure the model is. Accuracy is how often its most likely class is the label: compare `probs.argmax(axis=1)` with `labels`.",
                    nl="Je laatste regel geeft `{got}`: het gemiddelde van de grootste kansen, hoe zeker het model is. Accuracy is hoe vaak zijn waarschijnlijkste klasse het label is: vergelijk `probs.argmax(axis=1)` met `labels`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the share of photos whose most likely class is their label.",
                      "nl": "Vervang de `...` door het deel van de foto's waarvan de waarschijnlijkste klasse hun label is."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "no_value": {"en": "The last line of the box shows nothing. Put `accuracy` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `accuracy` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you take the most likely class of each photo with `argmax` along each row, compare it with `labels`, and take the mean?",
                      "nl": "Je laatste regel geeft `{got}`. Neem je de waarschijnlijkste klasse van elke foto met `argmax` langs elke rij, vergelijk je die met `labels`, en neem je het gemiddelde?"},
            "other": {"en": "Right for these labels. But with {where}, your last line gives `{got}`, and it should give about `0.83`. Do you work it out from `probs` and `labels` themselves?",
                      "nl": "Goed voor deze labels. Maar met {where} geeft je laatste regel `{got}`, en dat moet ongeveer `0.83` zijn. Reken je het uit met `probs` en `labels` zelf?"},
            "other_error": {"en": "Right for these labels. But with {where}, your code stops with an error. Do you work it out from `probs` and `labels` themselves?",
                            "nl": "Goed voor deze labels. Maar met {where} stopt je code met een fout. Reken je het uit met `probs` en `labels` zelf?"},
            "correct": {"en": "About 0.67: four of the six photos are right. The fourth photo counts as wrong: its right class got 0.35, but class 0 got 0.45, and accuracy only looks at the most likely class. The check also tried other labels.",
                        "nl": "Ongeveer 0,67: vier van de zes foto's zijn goed. De vierde foto telt als fout: zijn goede klasse kreeg 0,35, maar klasse 0 kreeg 0,45, en accuracy kijkt alleen naar de waarschijnlijkste klasse. De check probeerde ook andere labels."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("accuracy", _A_START, "empty", None),
    ("accuracy", _A, "correct", None),
    ("accuracy", _a("accuracy = (probs.argmax(axis=1) == labels).mean()"), "correct", None),
    ("accuracy", _a("accuracy = np.sum(np.argmax(probs, axis=1) == labels) / len(labels)"), "correct", None),
    ("accuracy", _a("accuracy = np.sum(probs.argmax(axis=1) == labels)"), "wrong", "count"),
    ("accuracy", _a("accuracy = np.mean(probs.argmax(axis=0) == labels)"), "error", "columns"),
    ("accuracy", _a("accuracy = np.mean(np.argmax(probs, axis=0) == labels)"), "error", "columns"),
    ("accuracy", _a("accuracy = np.mean(probs == labels)"), "error", "shapes"),
    ("accuracy", _a("accuracy = np.mean(probs.argmin(axis=1) == labels)"), "wrong", "least"),
    ("accuracy", _a("accuracy = np.mean(probs.max(axis=1))"), "wrong", "sure"),
    ("accuracy", _a("accuracy = 4 / 6"), "almost", "other-value"),
]
