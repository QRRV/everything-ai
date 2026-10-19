# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-reading-claims: sorting statements about a model into what was measured and what goes
beyond the measurement."""
import html

from everything_ai.check import Sort


def ladder(labels: dict) -> str:
    """How a claim grows: boxes from the measurement at the bottom to the headline at the top, each wider than the one
    below it. Ink only: nothing here is a role of a model. Boxes of HTML, not an SVG, so the text wraps and keeps its
    size in a narrow window."""
    rows = labels["rows"]
    n = len(rows)
    parts = [f'<div style="font-size:1rem;font-weight:600;color:var(--ea-ink);margin-bottom:0.5rem">{html.escape(labels["title"])}</div>']
    for k, (name, text) in enumerate(reversed(rows)):
        width = 64 + 36 * (n - 1 - k) / max(1, n - 1)   # the top row, the headline, is the widest
        parts.append(f'<div style="width:{width:.0f}%;box-sizing:border-box;margin:0 auto 0.5rem;padding:0.45rem 0.75rem;'
                     f'border:1.2px solid var(--ea-ink-3);border-radius:6px;background:var(--ea-inset);text-align:center;'
                     f'font-size:0.875rem;line-height:1.35;color:var(--ea-ink-2)">'
                     f'<span style="display:block;font-weight:600;color:var(--ea-ink)">{html.escape(name)}</span>'
                     f'{html.escape(text)}</div>')
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(labels["alt"])}" '
            f'style="margin:0.25rem 0 0.75rem;max-width:600px;font-family:var(--ea-font-ui)">{"".join(parts)}</figure>')

# The lesson is done when these are correct. "contamination" is the choice question, "check-claim" the open question.
REQUIRED = ["measured", "contamination", "check-claim"]

EXERCISES = {
    "measured": Sort(
        name="measured",
        groups={
            "measured": {"en": "Measured", "nl": "Gemeten"},
            "beyond": {"en": "Goes beyond what was measured", "nl": "Gaat verder dan wat gemeten is"},
        },
        items={
            "understands": {"en": "The model understands mathematics", "nl": "Het model begrijpt wiskunde"},
            "score": {"en": "The model got 92% of 1,000 test questions right", "nl": "Het model had 92% van 1.000 testvragen goed"},
            "teachers": {"en": "The model will soon replace maths teachers", "nl": "Het model vervangt binnenkort wiskundeleraren"},
            "higher": {"en": "It scored 4 points higher than last year's best model, on the same benchmark", "nl": "Het scoorde 4 punten hoger dan het beste model van vorig jaar, op dezelfde benchmark"},
            "lower": {"en": "On a new set of similar questions, its score was 9 points lower than on the benchmark", "nl": "Op een nieuwe set vergelijkbare vragen was zijn score 9 punten lager dan op de benchmark"},
            "all-problems": {"en": "The model gets 92% of school maths problems right", "nl": "Het model heeft 92% van de verhaaltjessommen van de basisschool goed"},
        },
        solution={"understands": "beyond", "score": "measured", "teachers": "beyond", "higher": "measured", "lower": "measured", "all-problems": "beyond"},
        nudge={
            "understands": {"en": "Which number, on which test, would show \"understands\"?",
                            "nl": "Welk getal, op welke test, zou \"begrijpt\" laten zien?"},
            "teachers": {"en": "Did anyone measure what happens in a classroom?",
                         "nl": "Heeft iemand gemeten wat er in een klaslokaal gebeurt?"},
            "all-problems": {"en": "Was it tested on all school maths problems, or on a fixed set of them?",
                             "nl": "Is het getest op alle verhaaltjessommen van de basisschool, of op een vaste set ervan?"},
            "score": {"en": "Is this a number on a test that someone ran?",
                      "nl": "Is dit een getal uit een test die iemand echt heeft uitgevoerd?"},
            "higher": {"en": "Is this a comparison of two scores on the same test?",
                       "nl": "Is dit een vergelijking van twee scores op dezelfde test?"},
            "lower": {"en": "Is this a number that someone measured, even though it is not good news?",
                      "nl": "Is dit een getal dat iemand mat, ook al is het geen goed nieuws?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the statements again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de uitspraken opnieuw uit."},
            "empty": {"en": "Put each statement in a group: measured, or goes beyond what was measured.",
                      "nl": "Zet elke uitspraak in een groep: gemeten, of gaat verder dan wat gemeten is."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The statement \u201c{item}\u201d belongs in the other group.",
                      "nl": "De uitspraak \u2018{item}\u2019 hoort in de andere groep."},
            "correct": {"en": "All {n} are right. A score on 1,000 test questions, a comparison on the same test and a lower score on new questions are measurements. Understanding mathematics and replacing teachers are claims that no test score shows, and 92% of all school maths problems says more than 92% of one set of them.",
                        "nl": "Alle {n} kloppen. Een score op 1.000 testvragen, een vergelijking op dezelfde test en een lagere score op nieuwe vragen zijn metingen. Wiskunde begrijpen en leraren vervangen zijn claims die geen testscore laat zien, en 92% van alle verhaaltjessommen zegt meer dan 92% van één set ervan."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner chose, state, known mistake).
_RIGHT = {"understands": "beyond", "score": "measured", "teachers": "beyond", "higher": "measured", "lower": "measured", "all-problems": "beyond"}
CASES = [
    ("measured", {}, "empty", None),
    ("measured", _RIGHT, "correct", None),
    ("measured", {"understands": "beyond", "score": "measured"}, "wrong", "unfinished"),
    ("measured", {**_RIGHT, "lower": "beyond"}, "wrong", "wrong:lower"),
]
