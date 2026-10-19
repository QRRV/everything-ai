# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-figure-and-table: finding the best row of a results table among the rows measured under
the same conditions. And the pictures: two scores with the axis from 91 and from 0, and a loss on a log scale."""
import re

from everything_ai import chart
from everything_ai.check import CodeBox, Mistake

SCORES = {"A": 91.2, "B": 91.8}   # the made-up accuracies of the worked example, in %


def bars_picture(labels: dict[str, str]) -> str:
    """The same two scores twice: on an axis that starts at 91, and on one that starts at 0."""
    panels = []
    for title, lo, hi in ((labels["cut"], 91.0, 92.0), (labels["full"], 0.0, 100.0)):
        p = chart.Panel((0, 3), (lo, hi), title=title, ylabel=labels["y"], xticks=[])
        for k, (name, v) in enumerate(SCORES.items()):
            p.rect(0.55 + k, lo, 1.45 + k, v, role="ink")
            p.text(1 + k, v + (hi - lo) * 0.05, f"{name}: {v}".replace(".", labels["point"]), role="ink")
        panels.append(p)
    return chart.figure(panels, alt=labels["alt"], width=280, height=260)


LOSS_START, LOSS_END, STEPS = 2.0, 0.02, 100


def loss_picture(labels: dict[str, str]) -> str:
    """A loss that falls from 2 to 0.02 in 100 steps, on a log scale: a straight line."""
    xs = list(range(STEPS + 1))
    ys = [LOSS_START * (LOSS_END / LOSS_START) ** (x / STEPS) for x in xs]
    p = chart.Panel((0, STEPS), (0.01, 3), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], ylog=True)
    p.line(xs, ys, role="loss")
    return chart.figure([p], alt=labels["alt"], width=460, height=280)


# The lesson is done when these are correct. "times-smaller" is the choice question, "abstract" the open question.
REQUIRED = ["times-smaller", "fair-best", "abstract"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own lines.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen regels.")

# --- exercise 2: fair-best ----------------------------------------------------------------------------------------------

_ROWS = ('rows = [("Ours", "full", 84.1), ("Model X", "full", 82.7), ("Model Y", "easy half", 89.5), '
         '("Model Z", "full", 83.9)]\n')
_F_HEAD = _ROWS + "best_name = None\nbest_score = 0\nfor name, test_set, score in rows:\n"
_F_START = _F_HEAD + "    ...\nbest_name\n"


def _f(body: str = "    if test_set == \"full\":\n        if score > best_score:\n            best_name = name\n            best_score = score\n") -> str:
    return _F_HEAD + body + "best_name\n"


_F = _f()
# B, the best row but on another test set, scores below 84.1: a code that skips rows by their score instead of their test set
# (`if score < 89:`) keeps B here, and so does a code that skips only "Model Y" or "easy half".
_OTHER = [("A", "full", 70.0), ("B", "small", 79.0), ("C", "full", 75.5), ("D", "full", 75.0)]

EXERCISES = {
    "fair-best": CodeBox(
        name="fair-best",
        start=_F_START,
        solution=_F,
        others=[{"rows": _OTHER}],
        first=[_DOTS],
        mistakes=[
            Mistake("any-test-set", wrong=_f("    if score > best_score:\n        best_name = name\n        best_score = score\n"),
                    en="Your last line gives `{got}`: the highest score of all rows, also of the row on the easy half of the test set. That row was measured under other conditions. Only compare rows on the full test set.",
                    nl="Je laatste regel geeft `{got}`: de hoogste score van alle rijen, ook van de rij op de makkelijke helft van de test set. Die rij is onder andere omstandigheden gemeten. Vergelijk alleen rijen op de volledige test set."),
            Mistake("last-full", wrong=_f("    if test_set == \"full\":\n        best_name = name\n        best_score = score\n"),
                    en="Your last line gives `{got}`: the last row on the full test set, not the best. Keep a row only when its score is higher than `best_score`, and then keep that score in `best_score` too.",
                    nl="Je laatste regel geeft `{got}`: de laatste rij op de volledige test set, niet de beste. Bewaar een rij alleen als zijn score hoger is dan `best_score`, en bewaar die score dan ook in `best_score`."),
            Mistake("lowest", wrong=_F_HEAD.replace("best_score = 0", "best_score = 100") + "    if test_set == \"full\":\n        if score < best_score:\n            best_name = name\n            best_score = score\nbest_name\n",
                    en="Your last line gives `{got}`: the row with the lowest score on the full test set. For a loss lower is better, but for accuracy higher is better: keep a row when its score is higher than `best_score`.",
                    nl="Je laatste regel geeft `{got}`: de rij met de laagste score op de volledige test set. Bij een loss is lager beter, maar bij accuracy is hoger beter: bewaar een rij als zijn score hoger is dan `best_score`."),
            Mistake("none-kept", wrong=_f("    if test_set == \"Full\":\n        if score > best_score:\n            best_name = name\n            best_score = score\n"),
                    en="Your last line gives `None`: no row was ever kept in `best_name`. Check the lines that should keep one: is `\"full\"` written exactly as in the table, does `best_score` start at 0, and is a row kept when its score is higher than `best_score`?",
                    nl="Je laatste regel geeft `None`: er is nooit een rij in `best_name` bewaard. Kijk naar de regels die er een moeten bewaren: staat `\"full\"` er precies zoals in de tabel, begint `best_score` op 0, en wordt een rij bewaard als zijn score hoger is dan `best_score`?"),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the `...` with the lines that keep the best row, but only among the rows on the full test set.",
                      "nl": "Vervang de `...` door de regels die de beste rij bewaren, maar alleen onder de rijen op de volledige test set."},
            "no_value": {"en": "The last line of the box shows nothing. Put `best_name` back on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `best_name` terug op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Which row has the highest score among the rows whose test set is \"full\"?",
                      "nl": "Je laatste regel geeft `{got}`. Welke rij heeft de hoogste score onder de rijen waarvan de test set \"full\" is?"},
            "other": {"en": "Right for this table. But with other rows, such as {where}, your code gives `{got}`. Does it keep only the rows whose test set is `\"full\"`, whatever the names and scores in the table?",
                      "nl": "Goed voor deze tabel. Maar met andere rijen, zoals {where}, geeft je code `{got}`. Bewaart hij alleen rijen waarvan de test set `\"full\"` is, wat de namen en scores in de tabel ook zijn?"},
            "other_error": {"en": "Right for this table. But with other rows, such as {where}, your code stops with an error. Does it work for any list of rows?",
                            "nl": "Goed voor deze tabel. Maar met andere rijen, zoals {where}, stopt je code met een fout. Werkt hij voor elke lijst rijen?"},
            "correct": {"en": "Your code finds the best of the fair rows: \"Ours\", with 84.1 against 83.9 for Model Z. Model Y's 89.5 was measured on the easy half, so it does not count.",
                        "nl": "Je code vindt de beste van de eerlijke rijen: \"Ours\", met 84,1 tegen 83,9 voor Model Z. De 89,5 van Model Y is op de makkelijke helft gemeten, dus die telt niet mee."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("fair-best", _F_START, "empty", None),
    ("fair-best", _f("    if test_set == \"full\" and score > best_score:\n        best_name = name\n        best_score = score\n"), "correct", None),
    ("fair-best", _f("    if score > best_score:\n        best_name = name\n        best_score = score\n"), "wrong", "any-test-set"),
    ("fair-best", _f("    if test_set == \"full\":\n        best_name = name\n        best_score = score\n"), "wrong", "last-full"),
    ("fair-best", _f("    if test_set == \"full\":\n        if score > best_score:\n            best_name = name\n"), "wrong", "last-full"),
    ("fair-best", _f("    if name == \"Ours\":\n        best_name = name\n"), "almost", "other-value"),
    ("fair-best", _f("    if score > best_score:\n        if test_set == \"full\":\n            best_name = name\n            best_score = score\n"), "correct", None),
    ("fair-best", _f("    if test_set == \"full\":\n        if score >= best_score:\n            best_name = name\n            best_score = score\n"), "correct", None),
    ("fair-best", _f("    if test_set == \"full\":\n        if score < best_score:\n            best_name = name\n            best_score = score\n"), "wrong", "none-kept"),
    ("fair-best", _f("    if test_set == \"Full\":\n        if score > best_score:\n            best_name = name\n            best_score = score\n"), "wrong", "none-kept"),
    ("fair-best", _F_HEAD.replace("best_score = 0", "best_score = 100") + "    if test_set == \"full\" and score < best_score:\n        best_name = name\n        best_score = score\nbest_name\n", "wrong", "lowest"),
    ("fair-best", _f("    if score < 89:\n        if score > best_score:\n            best_name = name\n            best_score = score\n"), "almost", "other-value"),
    ("fair-best", _f("    if name != \"Model Y\":\n        if score > best_score:\n            best_name = name\n            best_score = score\n"), "almost", "other-value"),
]
