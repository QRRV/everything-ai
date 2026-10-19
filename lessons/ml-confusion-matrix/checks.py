# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-confusion-matrix: precision and recall of a spam filter from its predictions, and the
confusion matrix of a model for three classes. And the picture of forty mails: the confusion matrix, precision and
recall as the decision threshold moves on the slider."""
import re

from everything_ai.check import Code, CodeBox, Mistake

# Forty mails: the probability of spam a filter gave each, and whether it is spam. Ten are spam.
SPAM = [0.95, 0.9, 0.85, 0.8, 0.7, 0.65, 0.55, 0.45, 0.35, 0.2]
HAM = [0.75, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.2, 0.15, 0.15, 0.12, 0.1, 0.1, 0.08, 0.08, 0.06, 0.05, 0.05, 0.05,
       0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01]
THRESHOLDS = [round(0.05 * k, 2) for k in range(1, 20)]   # the slider: 0.05 to 0.95


def counts(t: float) -> dict[str, int]:
    """The four cells of the confusion matrix at threshold t: a probability of t or more means the filter says spam."""
    tp = sum(p >= t for p in SPAM)
    fp = sum(p >= t for p in HAM)
    return {"tp": tp, "fn": len(SPAM) - tp, "fp": fp, "tn": len(HAM) - fp}


def scores(t: float) -> dict[str, float | None]:
    c = counts(t)
    said = c["tp"] + c["fp"]
    return {"precision": c["tp"] / said if said else None, "recall": c["tp"] / len(SPAM),
            "accuracy": (c["tp"] + c["tn"]) / (len(SPAM) + len(HAM))}


def picture(t: float, labels: dict[str, str]) -> str:
    """Precision (solid) and recall (dashed) for every threshold, with the slider's threshold marked. Drawn as SVG
    text, once for every stop of the slider (L.explore); dragging across the chart sets the threshold."""
    from everything_ai import chart
    c = chart.Panel((0, 1), (0, 1.05), title=labels["title"], xlabel=labels["threshold"], ylabel=labels["share"])
    ts = [v for v in THRESHOLDS if scores(v)["precision"] is not None]
    c.line(ts, [scores(v)["precision"] for v in ts], "ink", 2.4)
    c.line(THRESHOLDS, [scores(v)["recall"] for v in THRESHOLDS], "ink", 2.4, dashed=True)
    s = scores(t)
    if s["precision"] is not None:
        c.points([t], [s["precision"]], "ink", 5.5)
    c.points([t], [s["recall"]], "ink", 8, hollow=True)
    c.text(0.27, 0.4, labels["precision"], "ink-2")
    c.text(0.9, 0.45, labels["recall"], "ink-2")
    return chart.figure([c], alt=labels["alt"], width=460, height=280, drag=0)   # its text stays 11 px or more in a play block at 700 px


# The lesson is done when these are correct. "never-spam" is the choice question in the lesson.
REQUIRED = ["never-spam", "precision-recall", "matrix"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_P_HEAD = ("import numpy as np\n\npred = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0])\n"
           "y = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0])\ntp = np.sum(pred * y)\n")
_P_START = _P_HEAD + "fp = ...\nfn = ...\nprecision = ...\nrecall = ...\nprecision, recall\n"
_P_BODY = "fp = np.sum(pred * (1 - y))\nfn = np.sum((1 - pred) * y)\nprecision = tp / (tp + fp)\nrecall = tp / (tp + fn)\n"
_P = _P_HEAD + _P_BODY + "precision, recall"


def _p(body: str) -> str:
    return _P_HEAD + body + "precision, recall"


def _near(got, want, tol: float) -> bool:
    """Two numbers close to the two the check wants, such as precision and recall rounded or typed in."""
    try:
        return len(got) == len(want) and all(abs(float(g) - w) <= tol for g, w in zip(got, want))
    except (TypeError, ValueError):
        return False


_M_HEAD = ("import numpy as np\n\nlabels = np.array([0, 1, 2, 1, 0, 2, 2, 1, 0, 2])\npreds = np.array([0, 1, 1, 1, 0, 2, 0, 2, 0, 2])\n"
           "cm = np.zeros((3, 3))\nfor i in range(len(labels)):\n")
_M_START = _M_HEAD + "    ...\ncm\n"
_M_BODY = "    cm[labels[i], preds[i]] = cm[labels[i], preds[i]] + 1\n"
_M = _M_HEAD + _M_BODY + "cm"


def _m(body: str) -> str:
    return _M_HEAD + body + "cm"


EXERCISES = {
    "precision-recall": CodeBox(
        name="precision-recall",
        start=_P_START,
        solution=_P,
        keep=["tp = np.sum(pred * y)"],
        # The check also runs the box with other predictions: numbers typed in by hand do not change with them.
        others=[{"pred": Code("np.array([1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("swapped", wrong=_p("fp = np.sum(pred * (1 - y))\nfn = np.sum((1 - pred) * y)\nprecision = tp / (tp + fn)\nrecall = tp / (tp + fp)\n"),
                    en="Your last line gives `{got}`: precision and recall the other way round. Precision is `tp` divided by all mails the filter called spam, `tp + fp`, where a false positive has `pred` 1 and `y` 0; recall is `tp` divided by all mails that really are spam, `tp + fn`.",
                    nl="Je laatste regel geeft `{got}`: precision en recall andersom. Precision is `tp` gedeeld door alle mails die het filter spam noemde, `tp + fp`, waarbij een false positive `pred` 1 en `y` 0 heeft; recall is `tp` gedeeld door alle mails die echt spam zijn, `tp + fn`."),
            Mistake("over-all", wrong=_p("fp = np.sum(pred * (1 - y))\nfn = np.sum((1 - pred) * y)\nprecision = tp / len(y)\nrecall = tp / len(y)\n"),
                    en="Your last line gives `{got}`: both divide by all twelve mails. Precision only looks at the mails the filter called spam, `tp + fp`, and recall only at the mails that are spam, `tp + fn`.",
                    nl="Je laatste regel geeft `{got}`: allebei delen ze door alle twaalf mails. Precision kijkt alleen naar de mails die het filter spam noemde, `tp + fp`, en recall alleen naar de mails die spam zijn, `tp + fn`."),
            # 4 / 5 and 4 / 6 rounded, such as 0.8 and 0.67 typed in, or round(..., 2)
            Mistake("rounded", state="almost", matches=lambda got, text: _near(got, (4 / 5, 4 / 6), 0.01),
                    en="Your last line gives `{got}`: close, but rounded. Let Python work out both shares from `tp`, `fp` and `fn`, without rounding and without typing in a number.",
                    nl="Je laatste regel geeft `{got}`: dichtbij, maar afgerond. Laat Python beide delen uitrekenen uit `tp`, `fp` en `fn`, zonder afronden en zonder zelf een getal in te typen."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the four `...`: the false positives, the false negatives, and then precision and recall.",
                      "nl": "Vervang de vier `...`: de false positives, de false negatives, en dan precision en recall."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other predictions by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere voorspellingen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `precision, recall` on the last line, so the check can see them.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `precision, recall` op de laatste regel, zodat de check ze kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Is precision `tp` divided by all mails called spam, `tp + fp`, and recall `tp` divided by all mails that are spam, `tp + fn`?",
                      "nl": "Je laatste regel geeft `{got}`. Is precision `tp` gedeeld door alle mails die spam genoemd werden, `tp + fp`, en recall `tp` gedeeld door alle mails die spam zijn, `tp + fn`?"},
            "other": {"en": "Right for these predictions. But with {where}, your last line gives `{got}`, and it should give about `(0.625, 0.83)`. Do you work it out with `pred` and `y` themselves?",
                      "nl": "Goed voor deze voorspellingen. Maar met {where} geeft je laatste regel `{got}`, en dat moet ongeveer `(0.625, 0.83)` zijn. Reken je het uit met `pred` en `y` zelf?"},
            "other_error": {"en": "Right for these predictions. But with {where}, your code stops with an error. Do you work it out with `pred` and `y` themselves?",
                            "nl": "Goed voor deze voorspellingen. Maar met {where} stopt je code met een fout. Reken je het uit met `pred` en `y` zelf?"},
            "correct": {"en": "Precision 0.8 and recall about 0.67: four of the five mails called spam are spam, and the filter finds four of the six spam mails. The check also tried other predictions.",
                        "nl": "Precision 0,8 en recall ongeveer 0,67: vier van de vijf mails die spam genoemd worden, zijn spam, en het filter vindt vier van de zes spammails. De check probeerde ook andere voorspellingen."},
        },
    ),
    "matrix": CodeBox(
        name="matrix",
        start=_M_START,
        solution=_M,
        keep=["for i in range(len(labels)):"],
        # The check also runs the box with other predictions: a matrix typed in by hand does not change with them.
        others=[{"preds": Code("np.array([0, 1, 2, 1, 1, 2, 2, 0, 0, 1])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("transposed", wrong=_m("    cm[preds[i], labels[i]] = cm[preds[i], labels[i]] + 1\n"),
                    en="Your matrix has the rows and columns the other way round: a row for each predicted class. The rows are the true classes, `labels[i]`, and the columns the predicted ones, `preds[i]`.",
                    nl="Je matrix heeft de rijen en kolommen andersom: een rij voor elke voorspelde klasse. De rijen zijn de echte klassen, `labels[i]`, en de kolommen de voorspelde, `preds[i]`."),
            Mistake("set-one", wrong=_m("    cm[labels[i], preds[i]] = 1\n"),
                    en="Your matrix only has 0s and 1s: it marks which cells occur, not how often. Add 1 to the cell each time: `cm[labels[i], preds[i]] + 1`.",
                    nl="Je matrix heeft alleen nullen en enen: hij geeft aan welke vakjes voorkomen, niet hoe vaak. Tel er elke keer 1 bij op: `cm[labels[i], preds[i]] + 1`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with one line: add 1 to the cell in row `labels[i]` and column `preds[i]`.",
                      "nl": "Vervang de `...` door één regel: tel 1 op bij het vakje in rij `labels[i]` en kolom `preds[i]`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other predictions by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere voorspellingen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `cm` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `cm` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does each photo add 1 to the cell in row `labels[i]`, its true class, and column `preds[i]`, its predicted class?",
                      "nl": "Je laatste regel geeft `{got}`. Telt elke foto 1 op bij het vakje in rij `labels[i]`, zijn echte klasse, en kolom `preds[i]`, zijn voorspelde klasse?"},
            "other": {"en": "Right for these predictions. But with {where}, your last line gives `{got}`, another matrix than it should. Do you count with `labels` and `preds` themselves?",
                      "nl": "Goed voor deze voorspellingen. Maar met {where} geeft je laatste regel `{got}`, een andere matrix dan zou moeten. Tel je met `labels` en `preds` zelf?"},
            "other_error": {"en": "Right for these predictions. But with {where}, your code stops with an error. Do you count with `labels` and `preds` themselves?",
                            "nl": "Goed voor deze voorspellingen. Maar met {where} stopt je code met een fout. Tel je met `labels` en `preds` zelf?"},
            "correct": {"en": "The diagonal holds the 7 right predictions: 3 cats, 2 dogs and 2 birds. Off the diagonal: one dog taken for a bird, one bird for a cat and one for a dog. The check also tried other predictions.",
                        "nl": "Op de diagonaal staan de 7 goede voorspellingen: 3 katten, 2 honden en 2 vogels. Ernaast: één hond die voor een vogel werd aangezien, één vogel voor een kat en één voor een hond. De check probeerde ook andere voorspellingen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("precision-recall", _P_START, "empty", None),
    ("precision-recall", _P, "correct", None),
    ("precision-recall", _p("fp = np.sum(pred) - tp\nfn = np.sum(y) - tp\nprecision = tp / np.sum(pred)\nrecall = tp / np.sum(y)\n"), "correct", None),
    ("precision-recall", _p("fp = np.sum((pred == 1) * (y == 0))\nfn = np.sum((pred == 0) * (y == 1))\nprecision = tp / (tp + fp)\nrecall = tp / (tp + fn)\n"), "correct", None),
    ("precision-recall", _p("fp = np.sum(pred * (1 - y))\nfn = np.sum((1 - pred) * y)\nprecision = tp / (tp + fn)\nrecall = tp / (tp + fp)\n"), "wrong", "swapped"),
    ("precision-recall", _p("fp = np.sum((1 - pred) * y)\nfn = np.sum(pred * (1 - y))\nprecision = tp / (tp + fp)\nrecall = tp / (tp + fn)\n"), "wrong", "swapped"),
    ("precision-recall", _p("fp = np.sum(pred * (1 - y))\nfn = np.sum((1 - pred) * y)\nprecision = tp / len(y)\nrecall = tp / len(y)\n"), "wrong", "over-all"),
    ("precision-recall", _p("fp = 0\nfn = 0\nfor i in range(len(y)):\n    fp = fp + pred[i] * (1 - y[i])\n    fn = fn + (1 - pred[i]) * y[i]\nprecision = tp / (tp + fp)\nrecall = tp / (tp + fn)\n"), "correct", None),
    ("precision-recall", _p("fp = 1\nfn = 2\nprecision = tp / (tp + fp)\nrecall = tp / (tp + fn)\n"), "almost", "other-value"),
    ("precision-recall", _p("fp = np.sum(pred * (1 - y))\nfn = np.sum((1 - pred) * y)\nprecision = 0.8\nrecall = 0.67\n"), "almost", "rounded"),
    ("precision-recall", _p("fp = np.sum(pred * (1 - y))\nfn = np.sum((1 - pred) * y)\nprecision = round(tp / (tp + fp), 2)\nrecall = round(tp / (tp + fn), 2)\n"), "almost", "rounded"),
    ("matrix", _M_START, "empty", None),
    ("matrix", _M, "correct", None),
    ("matrix", _m("    cm[labels[i], preds[i]] += 1\n"), "correct", None),
    ("matrix", _m("    cm[preds[i], labels[i]] = cm[preds[i], labels[i]] + 1\n"), "wrong", "transposed"),
    ("matrix", _m("    cm[labels[i], preds[i]] = 1\n"), "wrong", "set-one"),
]
