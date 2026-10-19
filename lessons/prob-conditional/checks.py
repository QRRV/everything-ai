# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson prob-conditional: P(A | B) from a list of cases, as a function, and both directions of a
medical test from a table of counts. And the picture: a table of 100 mails, with the column that a condition keeps."""
import html
import re

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake

# 100 mails: spam or not, and whether they contain the word "free".
MAILS = [[15, 5], [10, 70]]   # rows: spam, not spam; columns: with "free", without


def picture(labels: dict[str, str]) -> str:
    """The table of 100 mails, with totals. The column of the mails with "free" is marked: P(spam | free) counts only
    there. Ink only."""
    rows = [(labels["spam"], MAILS[0]), (labels["normal"], MAILS[1]),
            (labels["total"], [MAILS[0][0] + MAILS[1][0], MAILS[0][1] + MAILS[1][1]])]
    cols = [labels["free"], labels["plain"], labels["total"]]
    CW, RH, LEFT, TOP = 130, 34, 120, 44
    parts = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    for j, name in enumerate(cols):
        on = j == 0
        parts.append(f'<text x="{LEFT + j * CW + CW / 2}" y="{TOP - 10}" style="font-size:13px;font-weight:600;'
                     f'fill:var(--ea-{"ink" if on else "ink-2"});text-anchor:middle">{html.escape(name)}</text>')
    for i, (name, values) in enumerate(rows):
        y = TOP + i * RH
        values = list(values) + [sum(values)]
        parts.append(f'<text x="{LEFT - 12}" y="{y + RH / 2 + 5}" style="font-size:13px;fill:var(--ea-ink-2);text-anchor:end">{html.escape(name)}</text>')
        for j, v in enumerate(values):
            on = j == 0
            stroke = "stroke:var(--ea-ink);stroke-width:2" if on else "stroke:var(--ea-line);stroke-width:1"
            parts.append(f'<rect x="{LEFT + j * CW + 3}" y="{y + 3}" width="{CW - 6}" height="{RH - 6}" rx="4" style="fill:var(--ea-inset);{stroke}"/>'
                         f'<text x="{LEFT + j * CW + CW / 2}" y="{y + RH / 2 + 5}" style="font-family:var(--ea-font-code);font-size:14px;'
                         f'fill:var(--ea-{"ink" if on else "ink-3"});text-anchor:middle;font-weight:{600 if on and i < 2 else 400}">{v}</text>')
    y = TOP + 3 * RH + 26
    parts.append(f'<text x="0" y="{y}" style="font-family:var(--ea-font-code);font-size:14px;fill:var(--ea-ink)">{html.escape(labels["caption"])}</text>')
    W, H = LEFT + 3 * CW + 2, y + 10
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:min(100%,{W}px)">'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(labels["alt"])}" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(parts)}</svg></figure>')


# The lesson is done when these are correct. "studied" is the choice question in the lesson.
REQUIRED = ["studied", "p-given", "test-both-ways"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_G_START = "def p_given(cases, a, b):\n    ...\n"
_G = ("def p_given(cases, a, b):\n    with_b = 0\n    both = 0\n    for outcome, condition in cases:\n        if condition == b:\n"
      "            with_b = with_b + 1\n            if outcome == a:\n                both = both + 1\n    return both / with_b")


def _g(new: str) -> str:
    return "def p_given(cases, a, b):\n" + new


# The `if outcome == a` at the same depth as `if condition == b`: it counts a in every case, not only among b.
_EVERYWHERE = _g("    with_b = 0\n    both = 0\n    for outcome, condition in cases:\n        if condition == b:\n"
                 "            with_b = with_b + 1\n        if outcome == a:\n            both = both + 1\n    return both / with_b")

_MAILS_CASE = [("spam", "free"), ("normal", "free"), ("spam", "free"), ("spam", "free"), ("normal", "plain"), ("spam", "plain"), ("spam", "plain")]

_T_HEAD = 'counts = {("ill", "positive"): 9, ("ill", "negative"): 1, ("healthy", "positive"): 99, ("healthy", "negative"): 891}\n'
_T_START = _T_HEAD + "p_ill_given_pos = ...\np_pos_given_ill = ...\n[p_ill_given_pos, p_pos_given_ill]\n"
_ILL_POS = 'counts[("ill", "positive")] / (counts[("ill", "positive")] + counts[("healthy", "positive")])'
_POS_ILL = 'counts[("ill", "positive")] / (counts[("ill", "positive")] + counts[("ill", "negative")])'
_T = _T_HEAD + f"p_ill_given_pos = {_ILL_POS}\np_pos_given_ill = {_POS_ILL}\n[p_ill_given_pos, p_pos_given_ill]"


def _t(first: str, second: str) -> str:
    return _T_HEAD + f"p_ill_given_pos = {first}\np_pos_given_ill = {second}\n[p_ill_given_pos, p_pos_given_ill]"


_ALL = 'counts[("ill", "positive")] / 1000'

EXERCISES = {
    "p-given": FunctionBox(
        name="p-given",
        function="p_given",
        start=_G_START,
        solution=_G,
        visible=[(_MAILS_CASE, "spam", "free")],
        hidden=[([("rain", "clouds"), ("dry", "clouds"), ("dry", "sun")], "rain", "clouds"), ([("a", "x")], "a", "x"),
                ([("pass", "study"), ("fail", "study"), ("pass", "study"), ("pass", "none"), ("fail", "none")], "fail", "study"),
                ([("spam", "free"), ("normal", "plain")], "spam", "plain")],
        first=[_DOTS],
        mistakes=[
            Mistake("joint", wrong=_g("    both = 0\n    for outcome, condition in cases:\n        if outcome == a and condition == b:\n"
                                      "            both = both + 1\n    return both / len(cases)"),
                    en="`{call}` gives `{got}`: the share of all cases where both happened. P(A | B) only looks at the cases where B happened: divide by how many there are.",
                    nl="`{call}` geeft `{got}`: het deel van alle gevallen waarin allebei gebeurde. P(A | B) kijkt alleen naar de gevallen waarin B gebeurde: deel door hoeveel dat er zijn."),
            Mistake("swapped", wrong=_g("    with_a = 0\n    both = 0\n    for outcome, condition in cases:\n        if outcome == a:\n"
                                        "            with_a = with_a + 1\n            if condition == b:\n                both = both + 1\n    return both / with_a"),
                    en="`{call}` gives `{got}`: that is P(B | A), the other way round. Count among the cases where the condition is `b`, not where the outcome is `a`.",
                    nl="`{call}` geeft `{got}`: dat is P(B | A), andersom. Tel onder de gevallen waarin de voorwaarde `b` is, niet waarin de uitkomst `a` is."),
            Mistake("everywhere", wrong=_EVERYWHERE,
                    en="`{call}` gives `{got}`, more than 1, so it cannot be a probability. You count outcome `a` in every case; count it only among the cases whose condition is `b` (step 3 of the recipe): put that `if` inside the other one.",
                    nl="`{call}` geeft `{got}`, meer dan 1, dus dat kan geen kans zijn. Je telt uitkomst `a` in alle gevallen; tel hem alleen onder de gevallen met voorwaarde `b` (stap 3 van het recept): zet die `if` binnen de andere."),
            Mistake("ignore-b", wrong=_g("    with_a = 0\n    for outcome, condition in cases:\n        if outcome == a:\n            with_a = with_a + 1\n    return with_a / len(cases)"),
                    en="`{call}` gives `{got}`: P(A) over all cases, as if `b` did not matter. Keep only the cases where the condition is `b`.",
                    nl="`{call}` geeft `{got}`: P(A) over alle gevallen, alsof `b` er niet toe doet. Houd alleen de gevallen over waarin de voorwaarde `b` is."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with P(`a` | `b`): among the cases whose condition is `b`, the share whose outcome is `a`, given back with `return`.",
                      "nl": "Vervang de `...` door P(`a` | `b`): onder de gevallen met voorwaarde `b`, het deel met uitkomst `a`, teruggegeven met `return`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with a list of cases and two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(cases, a, b)`.",
                          "nl": "De check roept `{call}` aan, met een lijst gevallen en twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(cases, a, b)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you count the cases whose condition is `b`, and among those the ones whose outcome is `a`, and divide?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Tel je de gevallen met voorwaarde `b`, en daaronder die met uitkomst `a`, en deel je?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Do you work it out from `cases`, `a` and `b` themselves?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken je het uit met `cases`, `a` en `b` zelf?"},
            "correct": {"en": "Your function is right in all {n} cases. 4 mails have \"free\", and 3 of those are spam: P(spam | free) = 0.75, while P(spam) over all 7 mails is about 0.71.",
                        "nl": "Je functie klopt in alle {n} gevallen. 4 mails bevatten \"free\", en 3 daarvan zijn spam: P(spam | free) = 0,75, terwijl P(spam) over alle 7 mails ongeveer 0,71 is."},
        },
    ),
    "test-both-ways": CodeBox(
        name="test-both-ways",
        start=_T_START,
        solution=_T,
        keep=[_T_HEAD.strip()],
        # Other counts, in which both conditional probabilities change: a number typed in by hand shows up.
        others=[{"counts": Code('{("ill", "positive"): 16, ("ill", "negative"): 4, ("healthy", "positive"): 98, ("healthy", "negative"): 882}')}],
        first=[_DOTS],
        mistakes=[
            Mistake("swapped", wrong=_t(_POS_ILL, _ILL_POS),
                    en="Your last line gives `{got}`: the two the other way round. P(ill | positive) counts among the positive tests; P(positive | ill) among the people who are ill.",
                    nl="Je laatste regel geeft `{got}`: de twee andersom. P(ziek | positief) telt onder de positieve tests; P(positief | ziek) onder de mensen die ziek zijn."),
            Mistake("all", wrong=_t(_ALL, _ALL),
                    en="Your last line gives `{got}`: twice the share of all 1000 people who are ill and test positive. Each conditional probability divides by the cases of its condition only.",
                    nl="Je laatste regel geeft `{got}`: twee keer het deel van alle 1000 mensen dat ziek is en positief test. Elke voorwaardelijke kans deelt alleen door de gevallen van zijn voorwaarde."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace both `...` with a conditional probability, worked out from `counts`.",
                      "nl": "Vervang beide `...` door een voorwaardelijke kans, uitgerekend uit `counts`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other counts by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere tellingen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `[p_ill_given_pos, p_pos_given_ill]` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `[p_ill_given_pos, p_pos_given_ill]` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. For each one: which cases does its condition keep, and how many of those also have the other?",
                      "nl": "Je laatste regel geeft `{got}`. Voor elk van de twee: welke gevallen houdt de voorwaarde over, en hoeveel daarvan hebben ook het andere?"},
            "other": {"en": "Right for these counts. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out from `counts` itself?",
                      "nl": "Goed voor deze tellingen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `counts` zelf?"},
            "other_error": {"en": "Right for these counts. But with {where}, your code stops with an error. Do you work it out from `counts` itself?",
                            "nl": "Goed voor deze tellingen. Maar met {where} stopt je code met een fout. Reken je het uit met `counts` zelf?"},
            "correct": {"en": "Only 9 of the 108 positive tests, about 0.083, come from someone who is ill, because healthy people are so many more. Yet P(positive | ill) is 0.9. The two directions can be far apart. The check also tried other counts.",
                        "nl": "Van de 108 positieve tests komen er maar 9 van iemand die ziek is, ongeveer 0,083, omdat er zoveel meer gezonde mensen zijn. Toch is P(positief | ziek) 0,9. De twee richtingen kunnen ver uit elkaar liggen. De check probeerde ook andere tellingen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("p-given", _G_START, "empty", None),
    ("p-given", _G, "correct", None),
    ("p-given", _g("    kept = []\n    for outcome, condition in cases:\n        if condition == b:\n            kept.append(outcome)\n    return kept.count(a) / len(kept)"), "correct", None),
    ("p-given", _g("    with_b = 0\n    both = 0\n    for pair in cases:\n        if pair[1] == b:\n            with_b += 1\n            if pair[0] == a:\n                both += 1\n    return both / with_b"), "correct", None),
    ("p-given", _g("    both = 0\n    for outcome, condition in cases:\n        if outcome == a and condition == b:\n            both = both + 1\n    return both / len(cases)"), "wrong", "joint"),
    ("p-given", _g("    with_a = 0\n    both = 0\n    for outcome, condition in cases:\n        if outcome == a:\n            with_a = with_a + 1\n            if condition == b:\n                both = both + 1\n    return both / with_a"), "wrong", "swapped"),
    ("p-given", _g("    with_a = 0\n    for outcome, condition in cases:\n        if outcome == a:\n            with_a = with_a + 1\n    return with_a / len(cases)"), "wrong", "ignore-b"),
    ("p-given", _EVERYWHERE, "wrong", "everywhere"),
    ("p-given", _g("    return 0.75"), "almost", None),
    ("test-both-ways", _T_START, "empty", None),
    ("test-both-ways", _T, "correct", None),
    ("test-both-ways", _T_HEAD + "both = counts[(\"ill\", \"positive\")]\npositive = both + counts[(\"healthy\", \"positive\")]\nill = both + counts[(\"ill\", \"negative\")]\n"
     "p_ill_given_pos = both / positive\np_pos_given_ill = both / ill\n[p_ill_given_pos, p_pos_given_ill]", "correct", None),
    ("test-both-ways", _t(_POS_ILL, _ILL_POS), "wrong", "swapped"),
    ("test-both-ways", _t(_ALL, _ALL), "wrong", "all"),
    ("test-both-ways", _t("9 / 108", "9 / 10"), "almost", "other-value"),
    ("test-both-ways", _t(_ILL_POS, "9 / 10"), "almost", "other-value"),
    ("test-both-ways", _T.replace("): 9,", "): 10,", 1), "wrong", "changed-line"),
]
