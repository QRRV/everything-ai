# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-fair-thresholds: searching, on the table UCI Adult, the pair of thresholds with equal
recall for women and men that gets the most people right, and deciding about new people with a threshold per group;
with the pictures of what each fairness criterion gives on the test set, of the learner's search and of the learner's
decisions."""
import math
import numbers
import re

from everything_ai.check import CodeBox, Mistake
from everything_ai import yours

# What the demo works out (tests/test_rai_fair_thresholds.py checks it). Per criterion: the thresholds for women and
# men, chosen on the people to train on with a gap below 0.01, and on the test set how often the model is right and,
# for women and men, the yes rate, the recall and the false positive rate.
RESULTS = {
    "one": {"thresholds": (0.5, 0.5), "right": 0.8344,
            "yes": (0.0705, 0.2479), "recall": (0.4356, 0.5697), "fpr": (0.0259, 0.1101)},
    "yes": {"thresholds": (0.29, 0.67), "right": 0.8170,
            "yes": (0.1369, 0.1453), "recall": (0.6441, 0.3725), "fpr": (0.0749, 0.0480)},
    "recall": {"thresholds": (0.38, 0.52), "right": 0.8324,
               "yes": (0.1024, 0.2365), "recall": (0.5542, 0.5507), "fpr": (0.0472, 0.1019)},
    "odds": {"thresholds": (0.71, 0.77), "right": 0.8047,
             "yes": (0.0310, 0.0791), "recall": (0.2237, 0.2196), "fpr": (0.0075, 0.0189)},
}
EVERYONE_NO = 0.7638   # how often a no for everyone is right on the test set: the share that earns 50,000 or less

# Exercise 1: per threshold, the recall and how many it gets right among the people to train on, per sex.
RECALL_WOMEN = {0.3: 0.675, 0.35: 0.596, 0.4: 0.534, 0.45: 0.476, 0.5: 0.433}
RECALL_MEN = {0.4: 0.684, 0.45: 0.628, 0.5: 0.576, 0.55: 0.526, 0.6: 0.466}
RIGHT_WOMEN = {0.3: 9739, 0.35: 9785, 0.4: 9819, 0.45: 9839, 0.5: 9843}
RIGHT_MEN = {0.4: 17200, 0.45: 17290, 0.5: 17337, 0.55: 17321, 0.6: 17192}
GAP = 0.02   # the largest recall gap the search of Exercise 1 allows, not included

# Exercise 2: four new people, and the thresholds for equal recall of the demo.
PEOPLE = [("women", 0.45), ("men", 0.45), ("women", 0.33), ("men", 0.6)]
THRESHOLD = {"women": 0.38, "men": 0.52}


def search(recall_women: dict, recall_men: dict, right_women: dict, right_men: dict, gap: float = GAP):
    """The pair (threshold for women, threshold for men) whose recalls differ by less than `gap` and that gets the
    most people right; None when no pair does."""
    best, most = None, 0
    for t_women in recall_women:
        for t_men in recall_men:
            if abs(recall_women[t_women] - recall_men[t_men]) < gap:
                right = right_women[t_women] + right_men[t_men]
                if right > most:
                    best, most = (t_women, t_men), right
    return best


def decide(people: list, threshold: dict) -> list:
    """For each (sex, probability), True for a yes: the probability is above the threshold of that sex."""
    return [score > threshold[sex] for sex, score in people]


def _num(x: float, decimal: str, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", decimal)


def _count(n: int, lang: str) -> str:
    """A whole number with a sign between thousands: 27,140 in English, 27.140 in Dutch."""
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"


def picture_criteria(labels: dict) -> str:
    """Three panels with a row per criterion, in the order of RESULTS: the yes rate, the recall and the false positive
    rate of women (open bars) and men (filled bars) on the test set, with how often the model is right at the right of
    each row. `labels` gives "title", "rows" (four names), "xlabels" (three), "right" (the word before the accuracy),
    "key", "decimal" and "alt"."""
    from everything_ai import chart
    keys = list(RESULTS)
    top = 1.6 * len(keys) + 0.1
    panels = []
    for k, (measure, role, xmax) in enumerate((("yes", "data", 1), ("recall", "data", 1), ("fpr", "loss", 0.2))):
        ticks = [0, xmax / 2, xmax]
        p = chart.Panel(x=(0, xmax), y=(0, top), xlabel=labels["xlabels"][k], yticks=[], xticks=ticks,
                        xtick_labels=[f"{t:g}" for t in ticks])
        for i, (key, name) in enumerate(zip(keys, labels["rows"])):
            women, men = RESULTS[key][measure]
            y = top - 1.6 * (i + 1)
            if k == 0:
                p.text(0.002, y + 1.2, name, "ink", anchor="start")
            if k == 2:
                p.text(xmax, y + 1.2, labels["right"] + " " + _num(RESULTS[key]["right"], labels["decimal"], 3),
                       "ink-2", anchor="end")
            p.rect(0, y + 0.55, women, y + 0.85, role=role)
            p.rect(0, y + 0.15, men, y + 0.45, role=role, solid=True)
            p.text(women + 0.015 * xmax, y + 0.7, _num(women, labels["decimal"]), "ink-2", anchor="start")
            p.text(men + 0.015 * xmax, y + 0.3, _num(men, labels["decimal"]), "ink-2", anchor="start")
        panels.append(p)
    svg = chart.figure(panels, alt=labels["alt"], width=250, height=330)
    if labels["decimal"] == ",":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(labels["title"], svg + f'<div style="{_UI}">{labels["key"]}</div>')


def _number(x) -> float | None:
    """A real number that is not True or False, as a float; else None."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _pair(v) -> tuple | None:
    """The learner's pair of thresholds, when it is one: a tuple or list of two numbers."""
    if not isinstance(v, (tuple, list)) or len(v) != 2:
        return None
    a, b = _number(v[0]), _number(v[1])
    return None if a is None or b is None else (a, b)


def _same_pair(got, want) -> bool:
    a, b = _pair(got), _pair(want)
    if b is None:
        return got is None and want is None
    return a is not None and abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9


def _is_pair(pair):
    def match(got, text):
        return _same_pair(got, pair)
    return match


def _one_number(got, text) -> bool:
    return _number(got) is not None


def _answers(v) -> list | None:
    """The learner's list of yes and no, when it is one: a list of True and False (also NumPy's)."""
    if not isinstance(v, list):
        return None
    out = []
    for x in v:
        if type(x).__name__ not in ("bool", "bool_"):
            return None
        out.append(bool(x))
    return out


def _same_answers(got, want) -> bool:
    a = _answers(got)
    return a is not None and a == list(want)


def _is_answers(answers):
    def match(got, text):
        return _answers(got) == answers
    return match


def _numbers_in_list(got, text) -> bool:
    return isinstance(got, list) and len(got) > 0 and all(_number(x) is not None for x in got)


def _one_answer(got, text) -> bool:
    return type(got).__name__ in ("bool", "bool_")


def _fewer_answers(got, text) -> bool:
    a = _answers(got)
    return a is not None and len(a) < len(PEOPLE)


_SEARCH_WORDS = {
    "en": {"title": "Your search: the pairs with a recall gap below 0.02", "xlabel": "threshold for men",
           "ylabel": "threshold for women", "yours": "your pair",
           "key": "Each square is a pair of thresholds. Framed: the gap between the two recalls is below 0.02, with the number of people the pair gets right. Thick line: your best.",
           "alt": "A grid of 5 by 5 pairs of thresholds, women down and men across. Pairs with a recall gap below 0.02: {passed}. Your best: {mine}."},
    "nl": {"title": "Jouw zoektocht: de paren met een verschil in recall onder 0,02", "xlabel": "threshold voor mannen",
           "ylabel": "threshold voor vrouwen", "yours": "jouw paar",
           "key": "Elk vakje is een paar thresholds. Omlijnd: het verschil tussen de twee recalls is kleiner dan 0,02, met het aantal mensen dat het paar goed heeft. Dikke lijn: jouw beste.",
           "alt": "Een rooster van 5 bij 5 paren thresholds, vrouwen omlaag en mannen opzij. Paren met een verschil in recall onder 0,02: {passed}. Jouw beste: {mine}."},
}


def _tables(names: dict) -> tuple | None:
    """The four dicts of the box as the learner has them, when each has 1 to 8 numbers per threshold, and the two
    of a sex have the same thresholds; else None."""
    out = []
    for name in ("recall_women", "recall_men", "right_women", "right_men"):
        d = names.get(name)
        if not isinstance(d, dict) or not 1 <= len(d) <= 8:
            return None
        if any(_number(k) is None or _number(x) is None for k, x in d.items()):
            return None
        out.append(d)
    if set(out[0]) != set(out[2]) or set(out[1]) != set(out[3]):
        return None
    return tuple(out)


def _search_picture(seen) -> str:
    """A grid of the pairs of thresholds, women down and men across: the pairs with a recall gap below GAP framed,
    with how many they get right, and the learner's `best` in a thick line."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    names = mine.names if isinstance(mine.names, dict) else {}
    tables = _tables(names)
    best = _pair(names.get("best"))
    if tables is None or (names.get("best") is not None and best is None):
        return ""
    recall_women, recall_men, right_women, right_men = tables
    w = _SEARCH_WORDS[seen.lang]
    sign = "," if seen.lang == "nl" else "."
    women, men = sorted(recall_women), sorted(recall_men)
    rows, cols = len(women), len(men)
    panel = chart.Panel(x=(-0.8, cols), y=(0, rows), xlabel=w["xlabel"], ylabel=w["ylabel"],
                        xticks=[c + 0.5 for c in range(cols)], xtick_labels=[f"{t:g}" for t in men],
                        yticks=[])
    passed = []
    for r, t_women in enumerate(women):
        y = rows - r - 1
        panel.text(-0.4, y + 0.5, f"{t_women:g}".replace(".", sign), "ink-3", anchor="middle")
        for c, t_men in enumerate(men):
            gap = abs(float(recall_women[t_women]) - float(recall_men[t_men]))
            if gap < GAP:
                panel.rect(c + 0.08, y + 0.1, c + 0.92, y + 0.9, role="data")
                right = float(right_women[t_women]) + float(right_men[t_men])
                words = _count(int(right), seen.lang) if right == int(right) else _num(right, sign)
                panel.text(c + 0.5, y + 0.5, words, "ink", anchor="middle")
                passed.append(f"{_num(t_women, sign)} {_num(t_men, sign)}: {words}")
            else:
                panel.text(c + 0.5, y + 0.5, "·", "ink-3", anchor="middle")
            if best is not None and abs(best[0] - t_women) < 1e-9 and abs(best[1] - t_men) < 1e-9:
                panel.line([c + 0.04, c + 0.96, c + 0.96, c + 0.04, c + 0.04],
                           [y + 0.06, y + 0.06, y + 0.94, y + 0.94, y + 0.06], role="ink", width=3)
    mine_words = f"{_num(best[0], sign)} {_num(best[1], sign)}" if best is not None else "–"
    svg = chart.figure([panel], alt=w["alt"].format(passed="; ".join(passed) or "–", mine=mine_words),
                       width=520, height=330)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{w["key"]}</div>')


_DECIDE_WORDS = {
    "en": {"title": "Your decisions: a threshold per sex", "xlabel": "the model's probability of a yes",
           "women": "women", "men": "men", "threshold": "threshold",
           "key": "Each dot is a person, at their probability; filled: your yes, open: your no. The dashed line is the threshold of their sex.",
           "alt": "Two rows, women and men, from 0 to 1, each with its threshold as a dashed line. Your answers: {shown}."},
    "nl": {"title": "Jouw beslissingen: een threshold per geslacht", "xlabel": "de kans op een ja volgens het model",
           "women": "vrouwen", "men": "mannen", "threshold": "threshold",
           "key": "Elke stip is een persoon, bij de eigen kans; gevuld: jouw ja, open: jouw nee. De gestippelde lijn is de threshold van het geslacht.",
           "alt": "Twee rijen, vrouwen en mannen, van 0 tot 1, elk met de threshold als gestippelde lijn. Jouw antwoorden: {shown}."},
}


def _decide_picture(seen) -> str:
    """Per sex a row from 0 to 1 with its threshold, and each person as a dot at their probability: filled for the
    learner's yes, open for a no."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    names = mine.names if isinstance(mine.names, dict) else {}
    people, threshold, yes = names.get("people"), names.get("threshold"), _answers(names.get("yes"))
    if not isinstance(people, list) or not isinstance(threshold, dict) or yes is None or len(yes) != len(people):
        return ""
    groups = [g for g in threshold if _number(threshold[g]) is not None and 0 <= threshold[g] <= 1]
    if not 1 <= len(groups) <= 4 or len(people) > 12:
        return ""
    for person in people:
        if not isinstance(person, tuple) or len(person) != 2 or person[0] not in groups:
            return ""
        if _number(person[1]) is None or not 0 <= person[1] <= 1:
            return ""
    w = _DECIDE_WORDS[seen.lang]
    sign = "," if seen.lang == "nl" else "."
    top = 1.6 * len(groups) + 0.1
    panel = chart.Panel(x=(0, 1), y=(0, top), xlabel=w["xlabel"], yticks=[], xticks=[0, 0.25, 0.5, 0.75, 1],
                        xtick_labels=["0", "0.25", "0.5", "0.75", "1"])
    shown = []
    for i, g in enumerate(groups):
        y = top - 1.6 * (i + 1)
        t = float(threshold[g])
        panel.text(0.002, y + 1.2, w.get(g, g), "ink", anchor="start", halo=True)   # Panel.text escapes it
        panel.line([0, 1], [y + 0.5, y + 0.5], role="ink-3", width=1)
        panel.line([t, t], [y + 0.1, y + 0.95], role="ink-3", width=1.2, dashed=True)
        panel.text(t + 0.01, y + 0.95, w["threshold"] + " " + _num(t, sign), "ink-3", anchor="start", halo=True)
        for (sex, score), answer in zip(people, yes):
            if sex == g:
                panel.points([float(score)], [y + 0.5], role="data", r=6, hollow=not answer)
                shown.append(f"{w.get(g, g)} {_num(float(score), sign)} {'ja' if seen.lang == 'nl' else 'yes'}"
                             if answer else
                             f"{w.get(g, g)} {_num(float(score), sign)} {'nee' if seen.lang == 'nl' else 'no'}")
    svg = chart.figure([panel], alt=w["alt"].format(shown="; ".join(shown)), width=520,
                       height=max(170, 60 * len(groups) + 90))
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{w["key"]}</div>')


# The lesson is done when these are correct.
REQUIRED = ["search", "decide", "unequal"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

_TABLE_LINES = [f"recall_women = {RECALL_WOMEN}", f"recall_men = {RECALL_MEN}",
                f"right_women = {RIGHT_WOMEN}", f"right_men = {RIGHT_MEN}"]
_SEARCH_HEAD = ("# UCI Adult, the people to train on: per threshold the recall, and how many the threshold gets right\n"
                + _TABLE_LINES[0] + "\n" + _TABLE_LINES[1] + "\n" + _TABLE_LINES[2] + "\n" + _TABLE_LINES[3] + "\n"
                + "\nbest = None\nmost = 0\nfor t_women in recall_women:\n    for t_men in recall_men:\n")
_SEARCH_START = _SEARCH_HEAD + "        ...\nbest\n"
_SEARCH_BODY = ("        gap = abs(recall_women[t_women] - recall_men[t_men])\n"
                "        if gap < 0.02:\n"
                "            right = right_women[t_women] + right_men[t_men]\n"
                "            if right > most:\n"
                "                most = right\n"
                "                best = (t_women, t_men)\n")
_SEARCH_SOLUTION = _SEARCH_HEAD + _SEARCH_BODY + "best\n"


def _sh(body: str, last: str = "best\n") -> str:
    return _SEARCH_HEAD + body + last


def _shswap(old: str, new: str) -> str:
    assert _SEARCH_BODY.count(old) == 1, old
    return _sh(_SEARCH_BODY.replace(old, new))


_PEOPLE_LINE = 'people = [("women", 0.45), ("men", 0.45), ("women", 0.33), ("men", 0.6)]'
_THRESHOLD_LINE = 'threshold = {"women": 0.38, "men": 0.52}'
_DECIDE_HEAD = ("# four new people: their sex and the model's probability of a yes\n" + _PEOPLE_LINE + "\n"
                "# the thresholds for equal recall of the demo\n" + _THRESHOLD_LINE + "\n"
                "\nyes = []\nfor sex, score in people:\n")
_DECIDE_START = _DECIDE_HEAD + "    ...\nyes\n"
_DECIDE_SOLUTION = _DECIDE_HEAD + "    yes.append(score > threshold[sex])\nyes\n"


def _d(body: str, last: str = "yes\n") -> str:
    return _DECIDE_HEAD + body + last


# The check also runs the search on a made-up table, where the pair with the smallest gap is not the best.
_OTHER_TABLES = {"recall_women": {0.3: 0.7, 0.35: 0.65, 0.4: 0.6, 0.45: 0.52, 0.5: 0.45},
                 "recall_men": {0.4: 0.75, 0.45: 0.695, 0.5: 0.612, 0.55: 0.545, 0.6: 0.46},
                 "right_women": {0.3: 800, 0.35: 820, 0.4: 830, 0.45: 835, 0.5: 838},
                 "right_men": {0.4: 1600, 0.45: 1640, 0.5: 1660, 0.55: 1650, 0.6: 1620}}

EXERCISES = {
    "search": CodeBox(
        name="search",
        picture=_search_picture,
        start=_SEARCH_START,
        solution=_SEARCH_SOLUTION,
        keep=_TABLE_LINES,
        compare=_same_pair,
        digits=3,
        others=[_OTHER_TABLES],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_gap`. In the box the names have none: `recall_women`, `recall_men`, `right_women`, `right_men`, `best`, `most`, `t_women`, `t_men`, `gap`, `right`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_gap`. In het vak hebben de namen dat niet: `recall_women`, `recall_men`, `right_women`, `right_men`, `best`, `most`, `t_women`, `t_men`, `gap`, `right`."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks a dict for a threshold it does not have. The women's dicts have the women's thresholds as keys, the men's dicts the men's: `recall_women[t_women]`, `right_women[t_women]`, `recall_men[t_men]`, `right_men[t_men]`.",
                    nl="Je code vraagt een dict om een threshold die het niet heeft. De dicts van de vrouwen hebben de thresholds van de vrouwen als sleutels, die van de mannen die van de mannen: `recall_women[t_women]`, `right_women[t_women]`, `recall_men[t_men]`, `right_men[t_men]`."),
            Mistake("nothing", matches=lambda got, text: got is None,
                    en="Your `best` is still `None`: no pair got through both `if`s. Does the first one keep the pairs with a small gap, `if gap < 0.02:`, and the second the pair with more right than so far, `if right > most:`?",
                    nl="Je `best` is nog steeds `None`: geen enkel paar kwam door beide `if`s. Houdt de eerste de paren met een klein verschil, `if gap < 0.02:`, en de tweede het paar met meer goed dan tot nu toe, `if right > most:`?"),
            Mistake("one-number", matches=_one_number,
                    en="Your `best` is one number, not a pair of thresholds. Keep the pair itself: `best = (t_women, t_men)`.",
                    nl="Je `best` is één getal, geen paar thresholds. Bewaar het paar zelf: `best = (t_women, t_men)`."),
            Mistake("no-gap", matches=_is_pair((0.5, 0.5)),
                    en="Your pair, 0.5 and 0.5, gets the most people right of all 25 pairs, but its recalls are 0.433 and 0.576. Count only the pairs whose gap, without its sign, is below 0.02: `gap = abs(...)` and `if gap < 0.02:`.",
                    nl="Je paar, 0,5 en 0,5, heeft de meeste mensen goed van alle 25 paren, maar de recalls zijn 0,433 en 0,576. Tel alleen de paren waarvan het verschil, zonder teken, kleiner is dan 0,02: `gap = abs(...)` en `if gap < 0.02:`."),
            Mistake("swapped", matches=_is_pair((0.55, 0.4)),
                    en="You found the right pair, but the wrong way round: the threshold for women comes first, `(t_women, t_men)`.",
                    nl="Je vond het goede paar, maar andersom: de threshold voor vrouwen komt eerst, `(t_women, t_men)`."),
            Mistake("last-pair", matches=_is_pair((0.45, 0.6)),
                    en="Your pair, 0.45 and 0.6, is the last pair with a gap below 0.02, not the one that gets the most people right. Keep a pair only `if right > most:`, and then also remember `most = right`.",
                    nl="Je paar, 0,45 en 0,6, is het laatste paar met een verschil onder 0,02, niet het paar dat de meeste mensen goed heeft. Bewaar een paar alleen `if right > most:`, en onthoud dan ook `most = right`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the search of the worked example: for each pair the gap between the two recalls, and for a gap below 0.02 how many the pair gets right; keep in `best` the pair with the most.",
                      "nl": "Vervang de `...` door de zoektocht van het uitgewerkte voorbeeld: per paar het verschil tussen de twee recalls, en bij een verschil onder 0,02 hoeveel het paar goed heeft; bewaar in `best` het paar met de meeste."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the lines of `recall_women`, `recall_men`, `right_women` and `right_men` exactly as they were: put them back. The check tries other numbers by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `recall_women`, `recall_men`, `right_women` en `right_men` nodig zoals ze waren: zet ze terug. De check probeert zelf andere getallen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `best` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `best` op de laatste regel, zodat de check het ziet."},
            "wrong": {"en": "Your last line gives `{got}`. Is `gap` the gap between the two recalls without its sign, do you count only the pairs with a gap below 0.02, and do you keep the pair that gets the most people right?",
                      "nl": "Je laatste regel geeft `{got}`. Is `gap` het verschil tussen de twee recalls zonder teken, tel je alleen de paren met een verschil onder 0,02, en bewaar je het paar dat de meeste mensen goed heeft?"},
            "other": {"en": "Right for these numbers. But with {where}, your last line gives `{got}`, and it should give `{want}`. Of the pairs with a recall gap below 0.02, do you keep the one with the most right, using the four dicts themselves?",
                      "nl": "Goed voor deze getallen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Bewaar je, van de paren met een verschil in recall onder 0,02, het paar met de meeste goed, en gebruik je de vier dicts zelf?"},
            "other_error": {"en": "Right for these numbers. But with {where}, your code stops with an error. Do you use the four dicts themselves, with `t_women` and `t_men` as keys?",
                            "nl": "Goed voor deze getallen. Maar met {where} stopt je code met een fout. Gebruik je de vier dicts zelf, met `t_women` en `t_men` als sleutels?"},
            "correct": {"en": "A threshold of 0.4 for women and 0.55 for men: recalls of 0.534 and 0.526, and 27,140 of the 32,561 people to train on right. One threshold of 0.5 for both gets 27,180 right: equal recall costs 40 right answers here.",
                        "nl": "Een threshold van 0,4 voor vrouwen en 0,55 voor mannen: een recall van 0,534 en 0,526, en 27.140 van de 32.561 mensen om op te trainen goed. Eén threshold van 0,5 voor beide heeft er 27.180 goed: gelijke recall kost hier 40 goede antwoorden."},
        },
    ),
    "decide": CodeBox(
        name="decide",
        picture=_decide_picture,
        start=_DECIDE_START,
        solution=_DECIDE_SOLUTION,
        keep=[_PEOPLE_LINE, _THRESHOLD_LINE],
        compare=_same_answers,
        # The check also runs the box with other people, and with other thresholds.
        others=[{"people": [("men", 0.5), ("women", 0.4), ("men", 0.53), ("women", 0.2)]},
                {"threshold": {"women": 0.3, "men": 0.62}}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_score`. In the box the names have none: `people`, `threshold`, `yes`, `sex`, `score`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_score`. In het vak hebben de namen dat niet: `people`, `threshold`, `yes`, `sex`, `score`."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks `threshold` for a key it does not have. Its keys are the sexes, `\"women\"` and `\"men\"`: look up the person's own with `threshold[sex]`, without quotes around `sex`.",
                    nl="Je code vraagt `threshold` om een sleutel die het niet heeft. De sleutels zijn de geslachten, `\"women\"` en `\"men\"`: zoek die van de persoon op met `threshold[sex]`, zonder aanhalingstekens om `sex`."),
            Mistake("one-answer", matches=_one_answer,
                    en="Your `yes` is one answer, not a list. Add each answer to the list: `yes.append(...)`.",
                    nl="Je `yes` is één antwoord, geen lijst. Voeg elk antwoord aan de lijst toe: `yes.append(...)`."),
            Mistake("numbers", matches=_numbers_in_list,
                    en="Your list holds numbers, not yes or no. Add the comparison itself: `score > threshold[sex]` is `True` for a yes.",
                    nl="Je lijst bevat getallen, geen ja of nee. Voeg de vergelijking zelf toe: `score > threshold[sex]` is `True` bij een ja."),
            Mistake("fewer", matches=_fewer_answers,
                    en="Your list has fewer answers than there are people. Does `yes.append(...)` stand inside the loop, indented by four spaces?",
                    nl="Je lijst heeft minder antwoorden dan er mensen zijn. Staat `yes.append(...)` in de loop, vier spaties ingesprongen?"),
            Mistake("one-threshold", matches=_is_answers([False, False, False, True]),
                    en="Your list gives the woman with 0.45 a no, so the threshold used for the woman was 0.45 or more, such as 0.5 or the men's 0.52. Each person gets the threshold of their own sex: `threshold[sex]`.",
                    nl="Je lijst geeft de vrouw met 0,45 een nee, dus de threshold voor de vrouw was 0,45 of meer, zoals 0,5 of die van de mannen, 0,52. Elke persoon krijgt de threshold van het eigen geslacht: `threshold[sex]`."),
            Mistake("women-threshold", matches=_is_answers([True, True, False, True]),
                    en="Your list gives the man with 0.45 a yes, so the threshold used for the man was below 0.45, such as the women's 0.38. Each person gets the threshold of their own sex: `threshold[sex]`.",
                    nl="Je lijst geeft de man met 0,45 een ja, dus de threshold voor de man lag onder 0,45, zoals die van de vrouwen, 0,38. Elke persoon krijgt de threshold van het eigen geslacht: `threshold[sex]`."),
            Mistake("flipped", matches=_is_answers([False, True, True, False]),
                    en="Your answers are the other way round. A yes means the probability is above the threshold: `score > threshold[sex]`.",
                    nl="Je antwoorden zijn precies andersom. Een ja betekent dat de kans boven de threshold ligt: `score > threshold[sex]`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with one line that adds to `yes` whether this person's `score` is above the threshold of their `sex`.",
                      "nl": "Vervang de `...` door één regel die aan `yes` toevoegt of de `score` van deze persoon boven de threshold van het eigen geslacht, `sex`, ligt."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the lines of `people` and `threshold` exactly as they were: put them back. The check tries other people and thresholds by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `people` en `threshold` nodig zoals ze waren: zet ze terug. De check probeert zelf andere mensen en thresholds."},
            "no_value": {"en": "The last line of the box shows nothing. Put `yes` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `yes` op de laatste regel, zodat de check het ziet."},
            "wrong": {"en": "Your last line gives `{got}`. Does each person get `score > threshold[sex]`, the threshold of their own sex?",
                      "nl": "Je laatste regel geeft `{got}`. Krijgt elke persoon `score > threshold[sex]`, de threshold van het eigen geslacht?"},
            "other": {"en": "Right for these people. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `score`, `sex` and `threshold` themselves, and not their numbers?",
                      "nl": "Goed voor deze mensen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `score`, `sex` en `threshold` zelf, en niet de getallen?"},
            "other_error": {"en": "Right for these people. But with {where}, your code stops with an error. Do you use `people` and `threshold` themselves?",
                            "nl": "Goed voor deze mensen. Maar met {where} stopt je code met een fout. Gebruik je `people` en `threshold` zelf?"},
            "correct": {"en": "The woman and the man with a probability of 0.45 get different answers: the woman gets a yes, because the women's threshold is 0.38, and the man gets a no, because the men's is 0.52. The model treated them the same; the thresholds did not.",
                        "nl": "De vrouw en de man met een kans van 0,45 krijgen verschillende antwoorden: de vrouw een ja, omdat de threshold van de vrouwen 0,38 is, en de man een nee, omdat die van de mannen 0,52 is. Het model behandelde ze hetzelfde; de thresholds niet."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("search", _SEARCH_START, "empty", None),
    ("search", _SEARCH_SOLUTION, "correct", None),
    ("search", _sh("        if abs(recall_women[t_women] - recall_men[t_men]) < 0.02:\n"
                   "            if right_women[t_women] + right_men[t_men] > most:\n"
                   "                most = right_women[t_women] + right_men[t_men]\n"
                   "                best = [t_women, t_men]\n"), "correct", None),
    ("search", _shswap("gap < 0.02", "gap <= 0.02"), "correct", None),
    ("search", _shswap("gap = abs(", "gap = _abs("), "error", "underscore"),
    ("search", _shswap("recall_men[t_men])", "recall_men[t_women])"), "error", "key"),
    ("search", _shswap("if gap < 0.02:", "if gap < 0:"), "wrong", "nothing"),
    ("search", _shswap("best = (t_women, t_men)", "best = right"), "wrong", "one-number"),
    ("search", _shswap("gap = abs(recall_women[t_women] - recall_men[t_men])",
                       "gap = recall_women[t_women] - recall_men[t_men]"), "wrong", "no-gap"),
    ("search", _shswap("best = (t_women, t_men)", "best = (t_men, t_women)"), "wrong", "swapped"),
    ("search", _shswap("                most = right\n", ""), "wrong", "last-pair"),
    ("search", _shswap("if right > most:", "if right < most:"), "wrong", "nothing"),
    ("search", _shswap("if gap < 0.02:", "if gap < 0.06:"), "wrong", None),
    ("search", _sh("        gap = abs(recall_women[t_women] - recall_men[t_men])\n"
                   "        if gap < most:\n"
                   "            most = gap\n"
                   "            best = (t_women, t_men)\n", "best\n").replace("most = 0\n", "most = 1\n"),
     "almost", "other-value"),
    ("search", _SEARCH_SOLUTION.replace("0.4: 17200", "0.4: 27200"), "wrong", "changed-line"),
    ("search", _sh("        ...\n        best = (0.4, 0.55)\n"), "wrong", "dots-left"),
    ("decide", _DECIDE_START, "empty", None),
    ("decide", _DECIDE_SOLUTION, "correct", None),
    ("decide", _d("    if score > threshold[sex]:\n        yes.append(True)\n    else:\n        yes.append(False)\n"),
     "correct", None),
    ("decide", _d("    yes.append(_score > threshold[sex])\n"), "error", "underscore"),
    ("decide", _d("    yes.append(score > threshold[\"sex\"])\n"), "error", "key"),
    ("decide", _d("    yes = score > threshold[sex]\n"), "wrong", "one-answer"),
    ("decide", _d("    yes.append(threshold[sex])\n"), "wrong", "numbers"),
    ("decide", _d("    answer = score > threshold[sex]\nyes.append(answer)\n"), "wrong", "fewer"),
    ("decide", _d("    yes.append(score > 0.5)\n"), "wrong", "one-threshold"),
    ("decide", _d("    yes.append(score > threshold[\"women\"])\n"), "wrong", "women-threshold"),
    ("decide", _d("    yes.append(score < threshold[sex])\n"), "wrong", "flipped"),
    ("decide", _d("    yes.append(score > 0.3)\n"), "wrong", None),
    ("decide", _d("    yes.append(score > 0.4 + 0.12 * (sex == \"men\"))\n"), "almost", "other-value"),
    ("decide", _DECIDE_SOLUTION.replace('("men", 0.6)', '("men", 0.4)'), "wrong", "changed-line"),
    ("decide", _d("    ...\n    yes.append(True)\n"), "wrong", "dots-left"),
]
