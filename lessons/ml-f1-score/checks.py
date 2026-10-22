# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-f1-score: the F1 score of a detector of 5s from its predictions. And the picture: precision,
recall, their plain mean and their F1 score as bars, for every point of the plane of precision and recall."""
import re

from everything_ai.check import Code, CodeBox, Mistake

# The worked example: four detectors of 5s on the last 1,000 of the 6,000 training digits of QMNIST (BSD-3-Clause,
# data/README.md), 80 of them a 5, as (true positives, false positives, false negatives, true negatives). The two in
# the middle are a logistic regression trained on the first 5,000 digits, with the decision threshold 0.05 and 0.5;
# tests/test_ml_f1_score.py trains it again and finds these counts.
DETECTORS = {"never": (0, 0, 80, 920), "low": (77, 120, 3, 800), "half": (52, 3, 28, 917), "always": (80, 920, 0, 0)}
STOPS = [round(0.05 * k, 2) for k in range(21)]   # precision and recall on the plane: 0 to 1 in steps of 0.05


def f1(p: float, r: float) -> float:
    """The harmonic mean of precision p and recall r; 0 when both are 0."""
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def counts_f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn)


def bars(p: float, r: float, labels: dict[str, str]) -> str:
    """Four bars from 0 to 1: precision, recall, their plain mean and their F1 score, each with its words above it.
    Drawn as SVG text, once for every point of the plane (L.plane). The words carry the numbers, in the lesson's
    language; the axis gets its decimal comma from L.plane."""
    from everything_ai import chart
    c = chart.Panel((0, 1), (0, 4), title=labels["title"], xticks=[0, 0.5, 1], yticks=[])
    rows = [(labels["precision"], p, "ink-2"), (labels["recall"], r, "ink-2"),
            (labels["mean"], (p + r) / 2, "ink-3"), (labels["f1"], f1(p, r), "ink")]
    for i, (words, value, role) in enumerate(rows):
        top = 4 - i
        c.text(0.01, top - 0.28, words, role, anchor="start")
        if value > 0:
            c.rect(0, top - 0.88, value, top - 0.52, role)
    return chart.figure([c], alt=labels["alt"], width=330, height=250)


# The lesson is done when these are correct. "lopsided" and "rare-pick" are the choice questions in the lesson.
REQUIRED = ["lopsided", "f1", "rare-pick"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# Twenty digits: 4 of them a 5 (y), and the detector calls 8 of them a 5 (pred): 3 true positives, 5 false positives,
# 1 false negative and 11 true negatives. F1 is 0.5; every known mistake gives another number.
_F_HEAD = ("import numpy as np\n\npred = np.array([0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1])\n"
           "y = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])\ntp = np.sum(pred * y)\n")
_F_START = _F_HEAD + "fp = ...\nfn = ...\nf1 = ...\nf1\n"
_F_COUNTS = "fp = np.sum(pred * (1 - y))\nfn = np.sum((1 - pred) * y)\n"
_F = _F_HEAD + _F_COUNTS + "f1 = 2 * tp / (2 * tp + fp + fn)\nf1"
_TN = "np.sum((1 - pred) * (1 - y))"


def _f(body: str, counts: str = _F_COUNTS) -> str:
    return _F_HEAD + counts + body + "f1"


EXERCISES = {
    "f1": CodeBox(
        name="f1",
        start=_F_START,
        solution=_F,
        digits=4,
        keep=["tp = np.sum(pred * y)"],
        # The check also runs the box with other digits: numbers typed in by hand do not change with them. Here tp, fp
        # and fn all differ from the 3, 5 and 1 above (2, 2 and 3, F1 4 / 9), and y holds 5 fives instead of 4, so that
        # every count shows, also one worked out from the 4 fives, such as fn = 4 - tp.
        others=[{"pred": Code("np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])"),
                 "y": Code("np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("mean", wrong=_f("f1 = (tp / (tp + fp) + tp / (tp + fn)) / 2\n"),
                    en="Your last line gives `{got}`: the plain mean of precision and recall. F1 is their harmonic mean, which stays close to the smaller of the two: `2 * tp / (2 * tp + fp + fn)`, step 3 of the recipe.",
                    nl="Je laatste regel geeft `{got}`: het gewone gemiddelde van precision en recall. F1 is hun harmonisch gemiddelde, dat dicht bij de kleinste van de twee blijft: `2 * tp / (2 * tp + fp + fn)`, stap 3 van het recept."),
            Mistake("no-two", wrong=_f("f1 = tp / (2 * tp + fp + fn)\n"),
                    en="Your last line gives `{got}`: half of the F1 score. Is the 2 above the line missing? `2 * tp / (2 * tp + fp + fn)`, or `2 * precision * recall / (precision + recall)`.",
                    nl="Je laatste regel geeft `{got}`: de helft van de F1 score. Mist de 2 boven de streep? `2 * tp / (2 * tp + fp + fn)`, of `2 * precision * recall / (precision + recall)`."),
            Mistake("product", wrong=_f("f1 = (tp / (tp + fp)) * (tp / (tp + fn))\n"),
                    en="Your last line gives `{got}`: precision times recall. F1 is twice that product, divided by the sum of the two; straight from the counts it is `2 * tp / (2 * tp + fp + fn)`.",
                    nl="Je laatste regel geeft `{got}`: precision maal recall. F1 is twee keer dat product, gedeeld door de som van de twee; rechtstreeks uit de aantallen is het `2 * tp / (2 * tp + fp + fn)`."),
            Mistake("with-tn", wrong=_f(f"f1 = 2 * tp / (2 * tp + fp + fn + {_TN})\n"),
                    en="Your last line gives `{got}`: the true negatives are below the line too. F1 leaves them out on purpose, so that the many other digits cannot make the detector look better: `2 * tp / (2 * tp + fp + fn)`.",
                    nl="Je laatste regel geeft `{got}`: de true negatives staan ook onder de streep. F1 laat ze met opzet weg, zodat de vele andere cijfers de detector niet beter kunnen laten lijken: `2 * tp / (2 * tp + fp + fn)`."),
            Mistake("accuracy", wrong=_f(f"f1 = (tp + {_TN}) / len(y)\n"),
                    en="Your last line gives `{got}`: the accuracy, the share of right predictions. F1 uses only the true positives, the false positives and the false negatives: step 3 of the recipe.",
                    nl="Je laatste regel geeft `{got}`: de accuracy, het deel goede voorspellingen. F1 gebruikt alleen de true positives, de false positives en de false negatives: stap 3 van het recept."),
            # np.sum(pred) for fp, or np.sum(y) for fn: both give 0.4 here
            Mistake("counted-all", wrong=_f("f1 = 2 * tp / (2 * tp + fp + fn)\n", "fp = np.sum(pred)\nfn = np.sum((1 - pred) * y)\n"),
                    en="Your last line gives `{got}`. Do `fp` and `fn` count only the mistakes? `np.sum(pred)` counts every digit called a 5 and `np.sum(y)` every 5, the 3 true positives included. A false positive has `pred` 1 and `y` 0, a false negative `pred` 0 and `y` 1.",
                    nl="Je laatste regel geeft `{got}`. Tellen `fp` en `fn` alleen de vergissingen? `np.sum(pred)` telt elk cijfer dat een 5 genoemd wordt en `np.sum(y)` elke 5, de 3 true positives meegeteld. Een false positive heeft `pred` 1 en `y` 0, een false negative `pred` 0 en `y` 1."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the three `...`: the false positives, the false negatives, and then the F1 score.",
                      "nl": "Vervang de drie `...`: de false positives, de false negatives, en dan de F1 score."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other digits by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere cijfers."},
            "no_value": {"en": "The last line of the box shows nothing. Put `f1` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `f1` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do `fp` and `fn` count the false positives and the false negatives, and is `f1` equal to `2 * tp / (2 * tp + fp + fn)`?",
                      "nl": "Je laatste regel geeft `{got}`. Tellen `fp` en `fn` de false positives en de false negatives, en is `f1` gelijk aan `2 * tp / (2 * tp + fp + fn)`?"},
            "other": {"en": "Right for these digits. But with {where}, your last line gives `{got}`, and it should give about `{want}`. Do you work it out with `pred` and `y` themselves, without rounding?",
                      "nl": "Goed voor deze cijfers. Maar met {where} geeft je laatste regel `{got}`, en dat moet ongeveer `{want}` zijn. Reken je het uit met `pred` en `y` zelf, zonder af te ronden?"},
            "other_error": {"en": "Right for these digits. But with {where}, your code stops with an error. Do you work it out with `pred` and `y` themselves?",
                            "nl": "Goed voor deze cijfers. Maar met {where} stopt je code met een fout. Reken je het uit met `pred` en `y` zelf?"},
            "correct": {"en": "F1 is 0.5: 3 true positives against 5 false positives and 1 false negative. The accuracy, 0.7, and the plain mean of precision 0.375 and recall 0.75, about 0.56, both look better than that. The check also tried other digits.",
                        "nl": "F1 is 0,5: 3 true positives tegen 5 false positives en 1 false negative. De accuracy, 0,7, en het gewone gemiddelde van precision 0,375 en recall 0,75, ongeveer 0,56, zien er allebei beter uit. De check probeerde ook andere cijfers."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("f1", _F_START, "empty", None),
    ("f1", _F, "correct", None),
    ("f1", _f("precision = tp / (tp + fp)\nrecall = tp / (tp + fn)\nf1 = 2 * precision * recall / (precision + recall)\n"), "correct", None),
    ("f1", _f("f1 = 2 * tp / (2 * tp + fp + fn)\n", "fp = np.sum(pred) - tp\nfn = np.sum(y) - tp\n"), "correct", None),
    ("f1", _f("f1 = 2 * tp / (2 * tp + fp + fn)\n", "fp = np.sum((pred == 1) * (y == 0))\nfn = np.sum((pred == 0) * (y == 1))\n"), "correct", None),
    # F1 does not change when fp and fn trade places, so mixing them up still gives the right number
    ("f1", _f("f1 = 2 * tp / (2 * tp + fp + fn)\n", "fp = np.sum((1 - pred) * y)\nfn = np.sum(pred * (1 - y))\n"), "correct", None),
    ("f1", _f("f1 = tp / (tp + (fp + fn) / 2)\n"), "correct", None),
    ("f1", _f("f1 = (tp / (tp + fp) + tp / (tp + fn)) / 2\n"), "wrong", "mean"),
    ("f1", _f("f1 = tp / (2 * tp + fp + fn)\n"), "wrong", "no-two"),
    ("f1", _f("precision = tp / (tp + fp)\nrecall = tp / (tp + fn)\nf1 = precision * recall / (precision + recall)\n"), "wrong", "no-two"),
    ("f1", _f("f1 = (tp / (tp + fp)) * (tp / (tp + fn))\n"), "wrong", "product"),
    ("f1", _f(f"tn = {_TN}\nf1 = 2 * tp / (2 * tp + fp + fn + tn)\n"), "wrong", "with-tn"),
    ("f1", _f("f1 = np.mean(pred == y)\n"), "wrong", "accuracy"),
    ("f1", _f("f1 = 2 * tp / (2 * tp + fp + fn)\n", "fp = np.sum(pred)\nfn = np.sum((1 - pred) * y)\n"), "wrong", "counted-all"),
    ("f1", _f("f1 = 2 * tp / (2 * tp + fp + fn)\n", "fp = np.sum(pred * (1 - y))\nfn = np.sum(y)\n"), "wrong", "counted-all"),
    ("f1", _f("f1 = 2 * tp / (2 * tp + fp + fn)\n", "fp = 5\nfn = 1\n"), "almost", "other-value"),
    ("f1", _f("f1 = 2 * tp / (2 * tp + fp + fn)\n", "fp = np.sum(pred * (1 - y))\nfn = 1\n"), "almost", "other-value"),
    ("f1", _f("f1 = 2 * 3 / (2 * 3 + fp + fn)\n"), "almost", "other-value"),
    ("f1", _f("f1 = 2 * tp / (2 * tp + fp + fn)\n", "fp = np.sum(pred * (1 - y))\nfn = 4 - tp\n"), "almost", "other-value"),
    ("f1", _f("f1 = 0.5\n"), "almost", "other-value"),
    ("f1", _f("f1 = round(2 * tp / (2 * tp + fp + fn), 2)\n"), "almost", "other-value"),
    ("f1", _f("f1 = ...\n"), "wrong", "dots-left"),
]
