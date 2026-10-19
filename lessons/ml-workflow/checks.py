# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-workflow: sorting what a team does into the six steps of the workflow, and picking a
model by the metric agreed on beforehand. And the picture of the six steps as a loop."""
import html
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort


def picture(boxes: list[tuple[str, str]], back: str, title: str, alt: str) -> str:
    """The six steps as a loop in two columns, narrow enough for a phone: 1 to 3 down the left, 4 to 6 up the right,
    and a dashed arrow from 6 back to 3, where the round starts again, explained by a key below. Ink only: no step is a role of a model."""
    BW, BH, GX, GY = 190, 58, 70, 34
    top = 30 if title else 4
    W = 2 * BW + GX
    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
    places = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]   # (column, row) of steps 1 to 6

    def box(c, r):
        return c * (BW + GX), top + r * (BH + GY)

    for (name, sub), (c, r) in zip(boxes, places):
        x, y = box(c, r)
        parts.append(f'<rect x="{x + 1}" y="{y}" width="{BW - 2}" height="{BH}" rx="8" '
                     f'style="fill:var(--ea-ink);fill-opacity:0.04;stroke:var(--ea-ink-3);stroke-width:1.2"/>'
                     f'<text x="{x + BW / 2}" y="{y + 24}" style="font-size:15px;font-weight:650;fill:var(--ea-ink);text-anchor:middle">'
                     f'{html.escape(name)}</text>'
                     f'<text x="{x + BW / 2}" y="{y + 44}" style="font-size:13px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(sub)}</text>')
    line = 'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"'

    def head(x, y, dx, dy):
        """An arrow head at (x, y), pointing along (dx, dy)."""
        return f'<path d="M{x - 6 * dx - 5 * dy} {y - 6 * dy - 5 * dx} L{x} {y} L{x - 6 * dx + 5 * dy} {y - 6 * dy + 5 * dx}" {line}/>'

    xl, xr = BW / 2, BW + GX + BW / 2
    for r in range(2):   # 1 → 2 → 3, down the left
        a, b = top + r * (BH + GY) + BH + 6, top + (r + 1) * (BH + GY) - 6
        parts.append(f'<path d="M{xl} {a} V{b}" {line}/>' + head(xl, b, 0, 1))
    y3 = top + 2 * (BH + GY) + BH / 2   # 3 → 4, along the bottom
    parts.append(f'<path d="M{BW + 6} {y3} H{BW + GX - 6}" {line}/>' + head(BW + GX - 6, y3, 1, 0))
    for r in (2, 1):     # 4 → 5 → 6, up the right
        a, b = top + r * (BH + GY) - 6, top + (r - 1) * (BH + GY) + BH + 6
        parts.append(f'<path d="M{xr} {a} V{b}" {line}/>' + head(xr, b, 0, -1))
    # 6 back to 3, where the round starts again: left from step 6, down the gap between the columns, and left
    # into step 3, above the arrow from 3 to 4
    xg, y6, yb = BW + GX / 2, top + BH / 2, y3 - 16
    parts.append(f'<path d="M{BW + GX - 6} {y6} H{xg} V{yb} H{BW + 6}" {line} stroke-dasharray="5 4"/>' + head(BW + 6, yb, -1, 0))
    yk = top + 3 * BH + 2 * GY + 26   # the key: what the dashed arrow means
    parts.append(f'<path d="M0 {yk - 4} H34" {line} stroke-dasharray="5 4"/>' + head(34, yk - 4, 1, 0)
                 + f'<text x="44" y="{yk}" style="font-size:13px;fill:var(--ea-ink)">{html.escape(back)}</text>')
    H = yk + 6
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:min(40rem,{W + 2}px)">'
            f'<svg viewBox="0 0 {W + 2} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(alt)}</title>{"".join(parts)}</svg></figure>')


# The lesson is done when these are correct. "need-ml" is the choice question in the lesson.
REQUIRED = ["need-ml", "steps", "pick-model"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# A comparison that leaves out a precision of exactly 0.9: `> 0.9`, `0.9 <`, or `<= 0.9` to skip a filter.
_STRICT = r"(?<![<>=!])>(?!=)\s*0?\.90*(?![\d.])|(?<![\d.])0?\.90*\s*<(?!=)|<=\s*0?\.90*(?![\d.])"

_NAMES = 'names = ["A", "B", "C", "D", "E"]'
_PRECISION = "precision = [0.85, 0.93, 0.92, 0.97, 0.91]"
_RECALL = "recall = [0.90, 0.40, 0.75, 0.60, 0.30]"
_P_HEAD = (f"{_NAMES}\n{_PRECISION}\n{_RECALL}\n\nbest = None\nbest_recall = 0\nfor i in range(len(names)):\n")
_P_START = _P_HEAD + "    ...\nbest\n"
_P_BODY = ("    if precision[i] >= 0.9:\n        if recall[i] > best_recall:\n            best = names[i]\n"
           "            best_recall = recall[i]\n")
_P = _P_HEAD + _P_BODY + "best"


def _p(body: str) -> str:
    return _P_HEAD + body + "best"


EXERCISES = {
    "steps": Sort(
        name="steps",
        groups={
            "question": {"en": "1\\. Question", "nl": "1\\. Vraag"},
            "metric": {"en": "2\\. Metric", "nl": "2\\. Metric"},
            "data": {"en": "3\\. Data", "nl": "3\\. Data"},
            "baseline": {"en": "4\\. Baseline", "nl": "4\\. Baseline"},
            "model": {"en": "5\\. Model", "nl": "5\\. Model"},
            "again": {"en": "6\\. Round again", "nl": "6\\. Nog een ronde"},
        },
        items={
            "purpose": {"en": "**Say what it is for.** Spam goes to a folder of its own, so people see less of it.",
                        "nl": "**Zeggen waar het voor is.** Spam gaat naar een eigen map, zodat mensen er minder van zien."},
            "goal": {"en": "**Agree on a goal.** At most 1 in 20 mails in the spam folder may be a real one.",
                     "nl": "**Een doel afspreken.** Hooguit 1 op de 20 mails in de spammap mag een echte zijn."},
            "permission": {"en": "**Ask permission.** Find out whether the mails may be used for training.",
                           "nl": "**Toestemming vragen.** Uitzoeken of de mails gebruikt mogen worden om te trainen."},
            "split": {"en": "**Put mails aside.** A fifth of the mails become the test set, before anyone looks at them closely.",
                      "nl": "**Mails apart zetten.** Een vijfde van de mails wordt de test set, voordat iemand ze goed bekijkt."},
            "never": {"en": "**Never spam.** Work out what a filter that never moves a mail scores.",
                      "nl": "**Nooit spam.** Uitrekenen wat een filter scoort dat nooit een mail verplaatst."},
            "train": {"en": "**Train.** Fit logistic regression on the training set, and pick its threshold on the validation set.",
                      "nl": "**Trainen.** Logistische regressie passen op de training set, en de threshold kiezen op de validation set."},
            "read": {"en": "**Read the false alarms.** Read the real mails the filter moved, and look for what they have in common.",
                     "nl": "**De valse alarmen lezen.** De echte mails lezen die het filter verplaatste, en zoeken wat ze gemeen hebben."},
        },
        solution={"purpose": "question", "goal": "metric", "permission": "data", "split": "data", "never": "baseline",
                  "train": "model", "read": "again"},
        nudge={
            "purpose": {"en": "Is this about how to measure, or about what the filter should do at all?",
                        "nl": "Gaat dit over hoe je meet, of over wat het filter überhaupt moet doen?"},
            "goal": {"en": "Is this about what the filter is for, or about how good it has to be, in a number?",
                     "nl": "Gaat dit over waar het filter voor is, of over hoe goed het moet zijn, in een getal?"},
            "permission": {"en": "Is this about the model, or about the examples it learns from?",
                           "nl": "Gaat dit over het model, of over de voorbeelden waarvan het leert?"},
            "split": {"en": "Does this happen to the examples, or to a model?",
                      "nl": "Gebeurt dit met de voorbeelden, of met een model?"},
            "never": {"en": "Does this filter learn anything? What is the simplest thing to compare a model with?",
                      "nl": "Leert dit filter iets? Wat is het eenvoudigste om een model mee te vergelijken?"},
            "train": {"en": "Which step learns from the training set and chooses on the validation set?",
                      "nl": "Welke stap leert van de training set en kiest op de validation set?"},
            "read": {"en": "Does this happen before there is a model, or after, to see where it goes wrong?",
                     "nl": "Gebeurt dit voordat er een model is, of erna, om te zien waar het misgaat?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the steps again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de stappen opnieuw uit."},
            "empty": {"en": "Put each thing the team does in the step of the workflow it belongs to.",
                      "nl": "Zet alles wat het team doet bij de stap van de workflow waar het hoort."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs in another step.",
                      "nl": "{item} hoort bij een andere stap."},
            "correct": {"en": "All {n} are right. The goal comes before any training, the test set is put aside before anyone looks, and reading the false alarms starts the next round.",
                        "nl": "Alle {n} kloppen. Het doel komt vóór het trainen, de test set gaat apart voordat iemand kijkt, en het lezen van de valse alarmen begint de volgende ronde."},
        },
    ),
    "pick-model": CodeBox(
        name="pick-model",
        start=_P_START,
        solution=_P,
        keep=[_NAMES, _PRECISION, _RECALL],
        # The check also runs the box with other models: a name typed in by hand does not change with them.
        others=[{"precision": Code("[0.90, 0.95, 0.88, 0.99, 0.93]"), "recall": Code("[0.80, 0.55, 0.95, 0.20, 0.60]")}],
        first=[
            _DOTS,
            Mistake("strict", state="almost", matches=lambda got, text: re.search(_STRICT, text) is not None,
                    en="Almost: your code leaves out a filter whose precision is exactly 0.9. The goal is a precision of at least 0.9, so 0.9 itself counts: `>= 0.9`.",
                    nl="Bijna: je code laat een filter weg met een precision van precies 0,9. Het doel is een precision van minstens 0,9, dus 0,9 telt mee: `>= 0.9`."),
        ],
        mistakes=[
            Mistake("no-goal", wrong=_p("    if recall[i] > best_recall:\n        best = names[i]\n        best_recall = recall[i]\n"),
                    en="Your last line gives `{got}`: the filter with the highest recall of all. But it has too many false alarms: first keep only the filters with a precision of at least 0.9.",
                    nl="Je laatste regel geeft `{got}`: het filter met de hoogste recall van allemaal. Maar dat geeft te veel vals alarm: houd eerst alleen de filters met een precision van minstens 0,9 over."),
            Mistake("most-precise", wrong=_p("    if precision[i] >= 0.9:\n        if precision[i] > best_recall:\n            best = names[i]\n            best_recall = precision[i]\n"),
                    en="Your last line gives `{got}`: the filter with the highest precision. The goal only asks for a precision of at least 0.9; among the filters that reach it, the one with the highest recall wins.",
                    nl="Je laatste regel geeft `{got}`: het filter met de hoogste precision. Het doel vraagt alleen een precision van minstens 0,9; van de filters die dat halen, wint het filter met de hoogste recall."),
            Mistake("first-ok", wrong=_p("    if precision[i] >= 0.9:\n        if best is None:\n            best = names[i]\n"),
                    en="Your last line gives `{got}`: the first filter with a precision of at least 0.9. The others may find more spam: compare their recall too.",
                    nl="Je laatste regel geeft `{got}`: het eerste filter met een precision van minstens 0,9. De andere filters vinden misschien meer spam: vergelijk ook hun recall."),
            Mistake("position", wrong=_p("    if precision[i] >= 0.9:\n        if recall[i] > best_recall:\n            best = i\n            best_recall = recall[i]\n"),
                    en="Your last line gives `{got}`: the position of the filter in the lists, not its name. The name of filter `i` is `names[i]`.",
                    nl="Je laatste regel geeft `{got}`: de plek van het filter in de lijsten, niet zijn naam. De naam van filter `i` is `names[i]`."),
            Mistake("last-ok", wrong=_p("    if precision[i] >= 0.9:\n        best = names[i]\n"),
                    en="Your last line gives `{got}`: the last filter with a precision of at least 0.9. Keep a filter only when its recall is higher than the best so far, and then keep that recall in `best_recall` too.",
                    nl="Je laatste regel geeft `{got}`: het laatste filter met een precision van minstens 0,9. Houd een filter alleen als zijn recall hoger is dan de beste tot nu toe, en bewaar die recall dan ook in `best_recall`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the choice: keep a filter when its precision is at least 0.9 and its recall is higher than `best_recall`.",
                      "nl": "Vervang de `...` door de keuze: houd een filter als zijn precision minstens 0,9 is en zijn recall hoger dan `best_recall`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other filters by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere filters."},
            "no_value": {"en": "The last line of the box shows nothing. Put `best` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `best` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Is `best` the name of the filter with the highest recall among the filters with a precision of at least 0.9?",
                      "nl": "Je laatste regel geeft `{got}`. Is `best` de naam van het filter met de hoogste recall van de filters met een precision van minstens 0,9?"},
            "other": {"en": "Right for these filters. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you choose with `precision` and `recall` themselves?",
                      "nl": "Goed voor deze filters. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Kies je met `precision` en `recall` zelf?"},
            "other_error": {"en": "Right for these filters. But with {where}, your code stops with an error. Do you choose with `precision` and `recall` themselves?",
                            "nl": "Goed voor deze filters. Maar met {where} stopt je code met een fout. Kies je met `precision` en `recall` zelf?"},
            "correct": {"en": "It is C. Four filters reach a precision of at least 0.9, and of those C finds the most spam, with a recall of 0.75. A finds even more, but with too many false alarms. The check also tried other filters.",
                        "nl": "Het is C. Vier filters halen een precision van minstens 0,9, en daarvan vindt C de meeste spam, met een recall van 0,75. A vindt er nog meer, maar met te veel vals alarm. De check probeerde ook andere filters."},
        },
    ),
}

_SOLVED = {"purpose": "question", "goal": "metric", "permission": "data", "split": "data", "never": "baseline", "train": "model", "read": "again"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("steps", {}, "empty", None),
    ("steps", _SOLVED, "correct", None),
    ("steps", {"purpose": "question", "goal": "metric"}, "wrong", "unfinished"),
    ("steps", {**_SOLVED, "goal": "question"}, "wrong", "wrong:goal"),
    ("steps", {**_SOLVED, "split": "model"}, "wrong", "wrong:split"),
    ("steps", {**_SOLVED, "read": "baseline"}, "wrong", "wrong:read"),
    ("pick-model", _P_START, "empty", None),
    ("pick-model", _P, "correct", None),
    ("pick-model", _p("    if precision[i] >= 0.9 and recall[i] > best_recall:\n        best = names[i]\n        best_recall = recall[i]\n"), "correct", None),
    ("pick-model", _p("    if precision[i] < 0.9:\n        continue\n    if recall[i] > best_recall:\n        best = names[i]\n        best_recall = recall[i]\n"), "correct", None),
    ("pick-model", _p("    if precision[i] > 0.9:\n        if recall[i] > best_recall:\n            best = names[i]\n            best_recall = recall[i]\n"), "almost", "strict"),
    ("pick-model", _p("    if 0.9 < precision[i]:\n        if recall[i] > best_recall:\n            best = names[i]\n            best_recall = recall[i]\n"), "almost", "strict"),
    ("pick-model", _p("    if precision[i] <= 0.9:\n        continue\n    if recall[i] > best_recall:\n        best = names[i]\n        best_recall = recall[i]\n"), "almost", "strict"),
    ("pick-model", _p("    if 0.9 <= precision[i]:\n        if best_recall < recall[i]:\n            best = names[i]\n            best_recall = recall[i]\n"), "correct", None),
    ("pick-model", _p("    if precision[i] >= 0.9:\n        if recall[i] > best_recall:\n            best = i\n            best_recall = recall[i]\n"), "wrong", "position"),
    ("pick-model", _p("    if recall[i] > best_recall:\n        best = names[i]\n        best_recall = recall[i]\n"), "wrong", "no-goal"),
    ("pick-model", _p("    if precision[i] >= 0.9:\n        if precision[i] > best_recall:\n            best = names[i]\n            best_recall = precision[i]\n"), "wrong", "most-precise"),
    ("pick-model", _p("    if precision[i] >= 0.9:\n        if best is None:\n            best = names[i]\n"), "wrong", "first-ok"),
    ("pick-model", _p("    if precision[i] >= 0.9:\n        best = names[i]\n"), "wrong", "last-ok"),
    ("pick-model", _p("    if precision[i] >= 0.9:\n        if recall[i] > best_recall:\n            best = names[i]\n"), "wrong", "last-ok"),
    ("pick-model", _p("    best = \"C\"\n"), "almost", "other-value"),
    ("pick-model", _P.replace(_RECALL, "recall = [0.90, 0.40, 0.75, 0.60, 0.95]"), "wrong", "changed-line"),
]
