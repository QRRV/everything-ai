# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-three-pass: sorting reading tasks by the pass they belong to. And the picture of the
three passes."""
import html

from everything_ai.check import Sort


def passes(labels: dict) -> str:
    """The three passes as cards side by side, each with its time, what you read and what you know after it. Ink only:
    nothing here is a role of a model. A grid of HTML cards, so they stack in a narrow window."""
    cards = []
    for name, time, read, after in labels["passes"]:
        cards.append(f'<div style="box-sizing:border-box;padding:0.55rem 0.75rem;border:1.2px solid var(--ea-ink-3);'
                     f'border-radius:6px;background:var(--ea-inset);font-size:0.875rem;line-height:1.35;color:var(--ea-ink-2)">'
                     f'<div style="font-weight:600;color:var(--ea-ink)">{html.escape(name)}</div>'
                     f'<div style="color:var(--ea-ink-3);margin-bottom:0.25rem">{html.escape(time)}</div>'
                     f'<div><b>{html.escape(labels["read"])}</b> {html.escape(read)}</div>'
                     f'<div style="margin-top:0.25rem"><b>{html.escape(labels["after"])}</b> {html.escape(after)}</div></div>')
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(labels["alt"])}" '
            f'style="margin:0.25rem 0 0.75rem;max-width:640px;font-family:var(--ea-font-ui)">'
            f'<div style="font-size:1rem;font-weight:600;color:var(--ea-ink);margin-bottom:0.5rem">{html.escape(labels["title"])}</div>'
            f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:0.5rem">{"".join(cards)}</div>'
            f'</figure>')


# The lesson is done when these are correct. "stuck" is the choice question, "differences" the open question.
REQUIRED = ["which-pass", "stuck", "differences"]

EXERCISES = {
    "which-pass": Sort(
        name="which-pass",
        groups={
            "first": {"en": "First pass", "nl": "Eerste ronde"},
            "second": {"en": "Second pass", "nl": "Tweede ronde"},
            "third": {"en": "Third pass", "nl": "Derde ronde"},
        },
        items={
            "results": {"en": "Check, in the caption and the text, that the rows of the results table were measured in the same way",
                        "nl": "In het bijschrift en de tekst controleren of de rijen van de resultatentabel op dezelfde manier gemeten zijn"},
            "headings": {"en": "Read the section headings and the conclusion, and decide whether to read on",
                         "nl": "De kopjes van de paragrafen en de conclusie lezen, en beslissen of je verder leest"},
            "norm": {"en": "Find out exactly where the paper puts each layer normalization, and compare that with your own block",
                     "nl": "Precies uitzoeken waar de paper elke layer normalization zet, en dat vergelijken met je eigen block"},
            "kind": {"en": "Name the kind of contribution, and the earlier work it builds on",
                     "nl": "De soort bijdrage noemen, en het eerdere werk waarop hij voortbouwt"},
            "unread": {"en": "Mark the references you have not read and will need",
                       "nl": "De referenties aanstrepen die je niet gelezen hebt en nodig zult hebben"},
            "rebuild": {"en": "Write down every setting you would need to train the model yourself, and what the paper leaves out",
                        "nl": "Elke instelling opschrijven die je nodig hebt om het model zelf te trainen, en wat de paper weglaat"},
        },
        solution={"results": "second", "headings": "first", "norm": "third", "kind": "first", "unread": "second", "rebuild": "third"},
        nudge={
            "results": {"en": "Does this need the figures and tables read with care, but not every detail of the method?",
                        "nl": "Vraagt dit om de figuren en tabellen zorgvuldig te lezen, maar niet om elk detail van de methode?"},
            "headings": {"en": "How long does this take, and what do you decide at the end of it?",
                         "nl": "Hoe lang duurt dit, en wat beslis je aan het eind ervan?"},
            "norm": {"en": "Is this a look at the whole, or a detail you only need to rebuild the model?",
                     "nl": "Is dit een blik op het geheel, of een detail dat je pas nodig hebt om het model na te bouwen?"},
            "kind": {"en": "Which of the five Cs of the first look are these?",
                     "nl": "Welke van de vijf C's van de eerste blik zijn dit?"},
            "unread": {"en": "When do you meet references you do not know: while skimming, or while reading the text itself?",
                       "nl": "Wanneer kom je referenties tegen die je niet kent: bij het doorbladeren, of bij het lezen van de tekst zelf?"},
            "rebuild": {"en": "Could you do this without going through every detail?",
                        "nl": "Kun je dit zonder elk detail door te nemen?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the tasks again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de taken opnieuw uit."},
            "empty": {"en": "Put each task with the pass it belongs to.",
                      "nl": "Zet elke taak bij de ronde waar hij hoort."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The task “{item}” belongs to another pass.",
                      "nl": "De taak ‘{item}’ hoort bij een andere ronde."},
            "correct": {"en": "All {n} are right. The headings and the kind of contribution are the first look; the results table and the references to read are the second pass; where each layer normalization goes and every setting to train it yourself are the third.",
                        "nl": "Alle {n} kloppen. De kopjes en de soort bijdrage zijn de eerste blik; de resultatentabel en de referenties om te lezen zijn de tweede ronde; waar elke layer normalization staat en elke instelling om zelf te trainen zijn de derde."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner chose, state, known mistake).
_RIGHT = {"results": "second", "headings": "first", "norm": "third", "kind": "first", "unread": "second", "rebuild": "third"}
CASES = [
    ("which-pass", {}, "empty", None),
    ("which-pass", _RIGHT, "correct", None),
    ("which-pass", {"headings": "first", "kind": "first"}, "wrong", "unfinished"),
    ("which-pass", {**_RIGHT, "results": "first"}, "wrong", "wrong:results"),
    ("which-pass", {**_RIGHT, "norm": "second"}, "wrong", "wrong:norm"),
]
