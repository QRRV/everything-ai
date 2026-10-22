# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-roc-auc: the true and the false positive rate of a filter at one threshold, and its AUC
by counting pairs. And the picture of the forty mails of ml-confusion-matrix: the mails by their probability with the
threshold of the slider, and the ROC curve with the dot of that threshold and the area under it."""
import re

from everything_ai.check import Code, CodeBox, Mistake

# The forty mails of ml-confusion-matrix: the probability of spam a filter gave each. Ten are spam.
SPAM = [0.95, 0.9, 0.85, 0.8, 0.7, 0.65, 0.55, 0.45, 0.35, 0.2]
HAM = [0.75, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.2, 0.15, 0.15, 0.12, 0.1, 0.1, 0.08, 0.08, 0.06, 0.05, 0.05, 0.05,
       0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01]
THRESHOLDS = [round(0.05 * k, 2) for k in range(21)]   # the slider: 0 to 1; at 1 nothing is called spam, at 0 everything
EXAMPLE = [1.0, 0.8, 0.6, 0.45, 0.2, 0.0]                # the thresholds of the worked example


def rates(t: float) -> dict[str, float]:
    """The counts and the two rates at threshold t: a probability of t or more means the filter says spam."""
    tp, fp = sum(p >= t for p in SPAM), sum(p >= t for p in HAM)
    return {"tp": tp, "fp": fp, "tpr": tp / len(SPAM), "fpr": fp / len(HAM)}


def curve() -> list[tuple[float, float]]:
    """The ROC curve: (false positive rate, true positive rate) at every threshold that changes something, from above
    the highest probability, (0, 0), down to the lowest, (1, 1). A spam mail and another mail with the same probability
    cross together: the curve goes diagonally there."""
    out = [(0.0, 0.0)]
    for t in sorted(set(SPAM + HAM), reverse=True):
        r = rates(t)
        out.append((r["fpr"], r["tpr"]))
    return out


def columns() -> list[float]:
    """The height of the column of area for each mail that is not spam, from the highest probability to the lowest:
    the share of spam mails with a higher probability, and half the share with the same. Each column is 1/30 wide."""
    return [(sum(s > h for s in SPAM) + 0.5 * sum(s == h for s in SPAM)) / len(SPAM) for h in sorted(HAM, reverse=True)]


def auc() -> float:
    """The area under the curve: the share of the 10 × 30 pairs in which the spam mail has the higher probability,
    a tie counting half."""
    return sum(columns()) / len(HAM)


def _bin(p: float) -> int:
    """The column of width 0.05 that a probability falls in: [0.05k, 0.05(k + 1)), counted in whole hundredths."""
    return round(p * 100) // 5


def picture(t: float, labels: dict[str, str]) -> str:
    """Left: every mail as a dot above its probability, spam filled in a row near the top, the others open and stacked
    in columns of 0.05, with the threshold as a line and a key above it, so that the line never runs through a word;
    dragging across this chart sets the threshold. Right: the ROC curve, the diagonal of guessing, the area under the
    curve (also under the slanted step of a tie), and the dot of the threshold. Drawn as SVG text, once for every stop
    of the slider (L.explore)."""
    from everything_ai import chart
    left = chart.Panel((0, 1), (0, 14.2), title=labels["mails"], xlabel=labels["probability"], yticks=[])
    left.points([0.05 * _bin(s) + 0.025 for s in SPAM], [12] * len(SPAM), "data", 4.5)
    stack: dict[int, int] = {}
    xs, ys = [], []
    for h in HAM:
        k = _bin(h)
        xs.append(0.05 * k + 0.025)
        ys.append(0.7 + 1.1 * stack.get(k, 0))
        stack[k] = stack.get(k, 0) + 1
    left.points(xs, ys, "data", 4.5, hollow=True)
    left.line([t, t], [0, 12.7], "hyper", 2.2)
    left.points([0.03], [13.6], "data", 4.5).text(0.065, 13.6, labels["spam"], "ink-2", anchor="start")
    left.points([0.33], [13.6], "data", 4.5, hollow=True).text(0.365, 13.6, labels["other"], "ink-2", anchor="start")
    right = chart.Panel((0, 1), (0, 1.02), title=labels["roc"], xlabel=labels["fpr"], ylabel=labels["tpr"])
    pts = curve()
    right.area([a for a, _ in pts], [b for _, b in pts], "ink-3")
    right.line([0, 1], [0, 1], "ink-3", 1.2, dashed=True)
    right.text(0.62, 0.48, labels["guess"], "ink-2", anchor="start")
    right.line([a for a, _ in pts], [b for _, b in pts], "ink", 2.4)
    r = rates(t)
    right.points([r["fpr"]], [r["tpr"]], "ink", 6)
    return chart.figure([left, right], alt=labels["alt"], width=330, height=280, drag=0)


# The lesson is done when these are correct. "order" is the choice question in the lesson.
REQUIRED = ["rates", "auc", "order"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# Twelve mails, split by their label as in the worked example: five spam mails in `pos`, seven others in `neg`. At the
# threshold 0.5 the filter calls four spam: 3 true positives and 1 false positive, so the true positive rate is 3 / 5
# and the false positive rate 1 / 7. Every known mistake gives other rates.
_POS = "pos = np.array([0.9, 0.8, 0.6, 0.4, 0.15])\n"
_NEG = "neg = np.array([0.75, 0.45, 0.4, 0.3, 0.2, 0.1, 0.05])\n"
_R_HEAD = "import numpy as np\n\n" + _POS + _NEG + "t = 0.5\n"
_R_START = _R_HEAD + "tpr = ...\nfpr = ...\ntpr, fpr\n"
_R_TPR = "tpr = np.sum(pos >= t) / len(pos)\n"
_R_FPR = "fpr = np.sum(neg >= t) / len(neg)\n"
_R = _R_HEAD + _R_TPR + _R_FPR + "tpr, fpr"


def _r(body: str, last: str = "tpr, fpr") -> str:
    return _R_HEAD + body + last


def _last_line(text: str) -> str:
    """The last line of code, without spaces and round brackets, such as "tpr,fpr"."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip() and not line.strip().startswith("#")]
    return lines[-1].replace(" ", "").strip("()") if lines else ""


def _right_in_a_list(got, text) -> bool:
    """The two right rates, but in a list: [tpr, fpr]."""
    try:
        return isinstance(got, list) and len(got) == 2 and abs(float(got[0]) - 0.6) < 1e-9 and abs(float(got[1]) - 1 / 7) < 1e-9
    except (TypeError, ValueError):
        return False


# The same twelve mails; one spam mail and one other mail both have 0.4. Of the 35 pairs, the spam mail is higher in
# 26, and one is a tie: the AUC is 26.5 / 35.
_A_HEAD = "import numpy as np\n\n" + _POS + _NEG + "above = 0\nfor n in neg:\n"
_A_START = _A_HEAD + "    above = ...\nauc = ...\nauc\n"
_A_LOOP = "    above = above + np.sum(pos > n) + 0.5 * np.sum(pos == n)\n"
_A_DIVIDE = "auc = above / (len(pos) * len(neg))\n"
_A = _A_HEAD + _A_LOOP + _A_DIVIDE + "auc"


def _a(loop: str = _A_LOOP, divide: str = _A_DIVIDE) -> str:
    return _A_HEAD + loop + divide + "auc"


def _near(value: float, tol: float):
    """A number close to `value` but not equal to it, such as the AUC rounded or typed in."""
    def matches(got, text) -> bool:
        try:
            return 1e-9 < abs(float(got) - value) <= tol
        except (TypeError, ValueError):
            return False
    return matches


_BOX_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

EXERCISES = {
    "rates": CodeBox(
        name="rates",
        start=_R_START,
        solution=_R,
        digits=4,
        keep=["t = 0.5"],
        # The check also runs the box with another threshold: numbers typed in by hand do not change with it.
        others=[{"t": 0.3}],
        first=[_DOTS],
        mistakes=[
            # before "swapped": the right rates, but the last line shows them the other way round
            Mistake("fpr-first", matches=lambda got, text: _last_line(text) == "fpr,tpr",
                    en="Your last line gives `{got}`: first `fpr`, then `tpr`. The check reads them the other way round: make the last line `tpr, fpr` again.",
                    nl="Je laatste regel geeft `{got}`: eerst `fpr`, dan `tpr`. De check leest ze andersom: maak van de laatste regel weer `tpr, fpr`."),
            Mistake("list", state="almost", matches=_right_in_a_list,
                    en="Your last line gives the right numbers, but in a list: `{got}`. Make it `tpr, fpr` again, without the square brackets: that is how the check reads them.",
                    nl="Je laatste regel geeft de goede getallen, maar in een lijst: `{got}`. Maak er weer `tpr, fpr` van, zonder de vierkante haken: zo leest de check ze."),
            Mistake("swapped", wrong=_r("tpr = np.sum(neg >= t) / len(neg)\nfpr = np.sum(pos >= t) / len(pos)\n"),
                    en="Your last line gives `{got}`: the two rates the other way round. The true positive rate is about the spam mails, `pos`; the false positive rate about the other mails, `neg`.",
                    nl="Je laatste regel geeft `{got}`: de twee rates andersom. De true positive rate gaat over de spammails, `pos`; de false positive rate over de andere mails, `neg`."),
            Mistake("precision", wrong=_r("tpr = np.sum(pos >= t) / (np.sum(pos >= t) + np.sum(neg >= t))\n" + _R_FPR),
                    en="Your last line gives `{got}`. Your `tpr` divides by all mails at or above `t`, the ones called spam: that is precision. The true positive rate is recall: divide by the spam mails, `len(pos)`.",
                    nl="Je laatste regel geeft `{got}`. Je `tpr` deelt door alle mails op of boven `t`, de mails die spam genoemd worden: dat is precision. De true positive rate is recall: deel door de spammails, `len(pos)`."),
            Mistake("alarms", wrong=_r(_R_TPR + "fpr = np.sum(neg >= t) / (np.sum(pos >= t) + np.sum(neg >= t))\n"),
                    en="Your last line gives `{got}`. Your `fpr` divides by all mails at or above `t`, the ones called spam: that is the share of false alarms among the alarms. The false positive rate divides by the mails that are not spam, `len(neg)`.",
                    nl="Je laatste regel geeft `{got}`. Je `fpr` deelt door alle mails op of boven `t`, de mails die spam genoemd worden: dat is het deel vals alarm onder de alarmen. De false positive rate deelt door de mails die geen spam zijn, `len(neg)`."),
            Mistake("over-spam", wrong=_r(_R_TPR + "fpr = np.sum(neg >= t) / len(pos)\n"),
                    en="Your last line gives `{got}`. Your `fpr` divides by the spam mails. The false positive rate is a share of the mails that are not spam: divide by `len(neg)`.",
                    nl="Je laatste regel geeft `{got}`. Je `fpr` deelt door de spammails. De false positive rate is een deel van de mails die geen spam zijn: deel door `len(neg)`."),
            Mistake("over-all", wrong=_r("tpr = np.sum(pos >= t) / (len(pos) + len(neg))\nfpr = np.sum(neg >= t) / (len(pos) + len(neg))\n"),
                    en="Your last line gives `{got}`: both divide by all twelve mails. Each rate is a share of one class: the true positive rate of the spam mails, `len(pos)`, the false positive rate of the others, `len(neg)`.",
                    nl="Je laatste regel geeft `{got}`: allebei delen ze door alle twaalf mails. Elke rate is een deel van één klasse: de true positive rate van de spammails, `len(pos)`, de false positive rate van de andere, `len(neg)`."),
            Mistake("left-alone", wrong=_r(_R_TPR + "fpr = np.sum(neg < t) / len(neg)\n"),
                    en="Your last line gives `{got}`. Your `fpr` counts the other mails below `t`: the share rightly left alone, which is 1 minus the false positive rate. Count the ones at or above `t`: `neg >= t`.",
                    nl="Je laatste regel geeft `{got}`. Je `fpr` telt de andere mails onder `t`: het deel dat terecht met rust gelaten wordt, 1 min de false positive rate. Tel de mails op of boven `t`: `neg >= t`."),
            # fpr = 1 - tpr, or the share of the spam below t: both give 0.4
            Mistake("one-minus", wrong=_r(_R_TPR + "fpr = 1 - tpr\n"),
                    en="Your last line gives `{got}`. Your `fpr` is 1 minus the true positive rate: the share of the spam mails that the filter misses. The false positive rate is about the other mails: the share of `neg` at or above `t`.",
                    nl="Je laatste regel geeft `{got}`. Je `fpr` is 1 min de true positive rate: het deel van de spammails dat het filter mist. De false positive rate gaat over de andere mails: het deel van `neg` op of boven `t`."),
        ],
        texts=_BOX_TEXTS | {
            "empty": {"en": "Replace the two `...`: the true positive rate in `tpr`, and the false positive rate in `fpr`.",
                      "nl": "Vervang de twee `...`: de true positive rate in `tpr`, en de false positive rate in `fpr`."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another threshold by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere threshold."},
            "no_value": {"en": "The last line of the box shows nothing. Put `tpr, fpr` on the last line, so the check can see them.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `tpr, fpr` op de laatste regel, zodat de check ze kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Is `tpr` the spam mails at or above `t` divided by all spam mails, and `fpr` the other mails at or above `t` divided by all other mails?",
                      "nl": "Je laatste regel geeft `{got}`. Is `tpr` de spammails op of boven `t` gedeeld door alle spammails, en `fpr` de andere mails op of boven `t` gedeeld door alle andere mails?"},
            "other": {"en": "Right for this threshold. But with {where}, your last line gives `{got}`, and it should give about `(0.8, 0.5714)`. Do you count the mails at or above `t`, with `>=`, in `pos` and `neg` themselves?",
                      "nl": "Goed voor deze threshold. Maar met {where} geeft je laatste regel `{got}`, en dat moet ongeveer `(0.8, 0.5714)` zijn. Tel je de mails op of boven `t`, met `>=`, in `pos` en `neg` zelf?"},
            "other_error": {"en": "Right for this threshold. But with {where}, your code stops with an error. Do you work it out with `pos`, `neg` and `t` themselves?",
                            "nl": "Goed voor deze threshold. Maar met {where} stopt je code met een fout. Reken je het uit met `pos`, `neg` en `t` zelf?"},
            "correct": {"en": "A true positive rate of 0.6 and a false positive rate of about 0.14: at 0.5 the filter finds 3 of the 5 spam mails, and calls 1 of the 7 others spam. The check also tried another threshold.",
                        "nl": "Een true positive rate van 0,6 en een false positive rate van ongeveer 0,14: bij 0,5 vindt het filter 3 van de 5 spammails, en noemt het 1 van de 7 andere mails spam. De check probeerde ook een andere threshold."},
        },
    ),
    "auc": CodeBox(
        name="auc",
        start=_A_START,
        solution=_A,
        digits=4,
        keep=["for n in neg:"],
        # The check also runs the box with six spam mails: an AUC, or a number of pairs, typed in does not change.
        others=[{"pos": Code("np.array([0.95, 0.85, 0.7, 0.5, 0.25, 0.08])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("no-ties", wrong=_a("    above = above + np.sum(pos > n)\n"),
                    en="Your last line gives `{got}`: a little too low. The spam mail and the other mail with 0.4 are a tie, and a tie counts half: add `0.5 * np.sum(pos == n)` as well.",
                    nl="Je laatste regel geeft `{got}`: iets te laag. De spammail en de andere mail met 0,4 staan gelijk, en een gelijke stand telt half: tel er ook `0.5 * np.sum(pos == n)` bij op."),
            Mistake("ties-whole", wrong=_a("    above = above + np.sum(pos >= n)\n"),
                    en="Your last line gives `{got}`: a little too high. With `>=`, the tie of the two mails with 0.4 counts as a whole pair in the right order. A tie counts half: `np.sum(pos > n) + 0.5 * np.sum(pos == n)`.",
                    nl="Je laatste regel geeft `{got}`: iets te hoog. Met `>=` telt de gelijke stand van de twee mails met 0,4 als een heel paar in de goede volgorde. Een gelijke stand telt half: `np.sum(pos > n) + 0.5 * np.sum(pos == n)`."),
            Mistake("reversed", wrong=_a("    above = above + np.sum(pos < n) + 0.5 * np.sum(pos == n)\n"),
                    en="Your last line gives `{got}`: the share of pairs in the wrong order, 1 minus the AUC. Count the spam mails with a higher probability than `n`: `pos > n`.",
                    nl="Je laatste regel geeft `{got}`: het deel van de paren in de verkeerde volgorde, 1 min de AUC. Tel de spammails met een hogere kans dan `n`: `pos > n`."),
            Mistake("no-accumulate", wrong=_a("    above = np.sum(pos > n) + 0.5 * np.sum(pos == n)\n"),
                    en="Your last line gives `{got}`, much too low: each turn of the loop replaces `above`, so it only holds the count for the last other mail. Add to what is already in it, as `_higher` does in the worked example: start the line with `above = above +`.",
                    nl="Je laatste regel geeft `{got}`, veel te laag: elke ronde van de lus vervangt `above`, dus daarin staat alleen de telling voor de laatste andere mail. Tel op bij wat er al in staat, zoals `_higher` in het uitgewerkte voorbeeld: begin de regel met `above = above +`."),
            Mistake("over-sum", wrong=_a(divide="auc = above / (len(pos) + len(neg))\n"),
                    en="Your last line gives `{got}`, more than 1: you divide by the number of mails, 12. Divide by the number of pairs: each of the 5 spam mails with each of the 7 others, `len(pos) * len(neg)`.",
                    nl="Je laatste regel geeft `{got}`, meer dan 1: je deelt door het aantal mails, 12. Deel door het aantal paren: elk van de 5 spammails met elk van de 7 andere, `len(pos) * len(neg)`."),
            Mistake("over-neg", wrong=_a(divide="auc = above / len(neg)\n"),
                    en="Your last line gives `{got}`, more than 1: you divide by the 7 other mails only. Each of them makes a pair with each of the 5 spam mails: divide by `len(pos) * len(neg)`.",
                    nl="Je laatste regel geeft `{got}`, meer dan 1: je deelt alleen door de 7 andere mails. Elk daarvan vormt een paar met elk van de 5 spammails: deel door `len(pos) * len(neg)`."),
            Mistake("over-pos", wrong=_a(divide="auc = above / len(pos)\n"),
                    en="Your last line gives `{got}`, more than 1: you divide by the 5 spam mails only. Each of them makes a pair with each of the 7 others: divide by `len(pos) * len(neg)`.",
                    nl="Je laatste regel geeft `{got}`, meer dan 1: je deelt alleen door de 5 spammails. Elk daarvan vormt een paar met elk van de 7 andere: deel door `len(pos) * len(neg)`."),
            # 26.5 / 35 rounded, such as 0.76 typed in, or round(..., 2)
            Mistake("rounded", state="almost", matches=_near(26.5 / 35, 0.006),
                    en="Your last line gives `{got}`: close, but rounded. Let Python work out the AUC from `above`, without rounding and without typing in a number.",
                    nl="Je laatste regel geeft `{got}`: dichtbij, maar afgerond. Laat Python de AUC uitrekenen uit `above`, zonder afronden en zonder zelf een getal in te typen."),
        ],
        texts=_BOX_TEXTS | {
            "empty": {"en": "Replace the two `...`: in the loop, add to `above` the spam mails above `n` and half the ties; then divide by the number of pairs.",
                      "nl": "Vervang de twee `...`: tel in de lus bij `above` de spammails boven `n` op en de helft van die met dezelfde kans; deel daarna door het aantal paren."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other probabilities by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere kansen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `auc` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `auc` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does `above` grow by the spam mails above `n` and half the ties, and do you divide by the number of pairs, `len(pos) * len(neg)`?",
                      "nl": "Je laatste regel geeft `{got}`. Groeit `above` met de spammails boven `n` en de helft van die met dezelfde kans, en deel je door het aantal paren, `len(pos) * len(neg)`?"},
            "other": {"en": "Right for these probabilities. But with {where}, your last line gives `{got}`, and it should give about `0.7143`. Do you count with `pos` and `n` themselves, and divide by `len(pos) * len(neg)`?",
                      "nl": "Goed voor deze kansen. Maar met {where} geeft je laatste regel `{got}`, en dat moet ongeveer `0.7143` zijn. Tel je met `pos` en `n` zelf, en deel je door `len(pos) * len(neg)`?"},
            "other_error": {"en": "Right for these probabilities. But with {where}, your code stops with an error. Do you count with `pos` and `n` themselves?",
                            "nl": "Goed voor deze kansen. Maar met {where} stopt je code met een fout. Tel je met `pos` en `n` zelf?"},
            "correct": {"en": "An AUC of about 0.757: in 26 of the 35 pairs the spam mail has the higher probability, and one pair is a tie. The check also tried other probabilities.",
                        "nl": "Een AUC van ongeveer 0,757: in 26 van de 35 paren heeft de spammail de hogere kans, en één paar staat gelijk. De check probeerde ook andere kansen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("rates", _R_START, "empty", None),
    ("rates", _R, "correct", None),
    ("rates", _r("tpr = np.sum(pos >= t) / 5\nfpr = np.sum(neg >= t) / 7\n"), "correct", None),
    ("rates", _r("tpr = np.mean(pos >= t)\nfpr = np.mean(neg >= t)\n"), "correct", None),
    ("rates", _r("tp = np.sum(pos >= t)\nfn = np.sum(pos < t)\nfp = np.sum(neg >= t)\ntn = np.sum(neg < t)\n"
                 "tpr = tp / (tp + fn)\nfpr = fp / (fp + tn)\n"), "correct", None),
    ("rates", _r(_R_TPR + _R_FPR, "fpr, tpr"), "wrong", "fpr-first"),
    ("rates", _r(_R_TPR + _R_FPR, "[tpr, fpr]"), "almost", "list"),
    ("rates", _r("tpr = np.sum(neg >= t) / len(neg)\nfpr = np.sum(pos >= t) / len(pos)\n"), "wrong", "swapped"),
    ("rates", _r("tpr = np.sum(pos >= t) / (np.sum(pos >= t) + np.sum(neg >= t))\n" + _R_FPR), "wrong", "precision"),
    ("rates", _r(_R_TPR + "fpr = np.sum(neg >= t) / (np.sum(pos >= t) + np.sum(neg >= t))\n"), "wrong", "alarms"),
    ("rates", _r(_R_TPR + "fpr = np.sum(neg >= t) / len(pos)\n"), "wrong", "over-spam"),
    ("rates", _r("tpr = np.sum(pos >= t) / (len(pos) + len(neg))\nfpr = np.sum(neg >= t) / (len(pos) + len(neg))\n"), "wrong", "over-all"),
    ("rates", _r(_R_TPR + "fpr = np.sum(neg < t) / len(neg)\n"), "wrong", "left-alone"),
    ("rates", _r(_R_TPR + "fpr = 1 - tpr\n"), "wrong", "one-minus"),
    ("rates", _r(_R_TPR + "fpr = np.sum(pos < t) / len(pos)\n"), "wrong", "one-minus"),
    ("rates", _r("tpr = 0.6\nfpr = 1 / 7\n"), "almost", "other-value"),
    ("rates", _r("tpr = np.sum(pos > t) / len(pos)\nfpr = np.sum(neg > t) / len(neg)\n"), "almost", "other-value"),
    ("rates", _r(_R_TPR + _R_FPR, "print(tpr, fpr)"), "wrong", "print-last"),
    ("rates", _R.replace("t = 0.5\n", "").replace(">= t", ">= 0.5"), "wrong", "changed-line"),
    ("rates", _r(_R_TPR + "fpr = ...\n"), "wrong", "dots-left"),
    ("auc", _A_START, "empty", None),
    ("auc", _A, "correct", None),
    ("auc", _a("    above = above + np.sum(pos > n) + np.sum(pos == n) / 2\n", "auc = above / 35\n"), "almost", "other-value"),
    ("auc", _a("    above = above + np.sum(pos > n) + np.sum(pos == n) / 2\n", "auc = above / (pos.size * neg.size)\n"), "correct", None),
    ("auc", _a("    for q in pos:\n        if q > n:\n            above = above + 1\n        if q == n:\n            above = above + 0.5\n"), "correct", None),
    ("auc", _a("    above = above + np.sum(pos > n)\n"), "wrong", "no-ties"),
    ("auc", _a("    above = above + np.sum(pos >= n)\n"), "wrong", "ties-whole"),
    ("auc", _a("    above = above + np.sum(pos < n) + 0.5 * np.sum(pos == n)\n"), "wrong", "reversed"),
    ("auc", _a("    above = np.sum(pos > n) + 0.5 * np.sum(pos == n)\n"), "wrong", "no-accumulate"),
    ("auc", _a(divide="auc = above / (len(pos) + len(neg))\n"), "wrong", "over-sum"),
    ("auc", _a(divide="auc = above / len(neg)\n"), "wrong", "over-neg"),
    ("auc", _a(divide="auc = above / len(pos)\n"), "wrong", "over-pos"),
    ("auc", _a(divide="auc = round(above / (len(pos) * len(neg)), 2)\n"), "almost", "rounded"),
    ("auc", _a(divide="auc = 0.757\n"), "almost", "rounded"),
    ("auc", _a(divide="auc = 26.5 / 35\n"), "almost", "other-value"),
    ("auc", _A_HEAD + _A_LOOP + _A_DIVIDE + "print(auc)", "wrong", "print-last"),
    ("auc", _A.replace("for n in neg:", "for m in neg:").replace("> n)", "> m)").replace("== n)", "== m)"), "wrong", "changed-line"),
    ("auc", _a("    above = ...\n"), "wrong", "dots-left"),
]
