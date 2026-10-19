# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-how-research-is-published: sorting papers by the kind of contribution they make."""
import html

from everything_ai.check import Sort


def kinds(labels: dict) -> str:
    """The six kinds of contribution as cards: what each adds, what to check, and a real example. Ink only: nothing
    here is a role of a model. A grid of HTML cards, not an SVG, so the text wraps and the cards stack in a narrow
    window."""
    cards = []
    for name, adds, check, example in labels["kinds"]:
        cards.append(f'<div style="box-sizing:border-box;padding:0.55rem 0.75rem;border:1.2px solid var(--ea-ink-3);'
                     f'border-radius:6px;background:var(--ea-inset);font-size:0.875rem;line-height:1.35;color:var(--ea-ink-2)">'
                     f'<div style="font-weight:600;color:var(--ea-ink);margin-bottom:0.25rem">{html.escape(name)}</div>'
                     f'<div><b>{html.escape(labels["adds"])}</b> {html.escape(adds)}</div>'
                     f'<div><b>{html.escape(labels["check"])}</b> {html.escape(check)}</div>'
                     f'<div style="margin-top:0.25rem;color:var(--ea-ink-3)">{html.escape(example)}</div></div>')
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(labels["alt"])}" '
            f'style="margin:0.25rem 0 0.75rem;max-width:640px;font-family:var(--ea-font-ui)">'
            f'<div style="font-size:1rem;font-weight:600;color:var(--ea-ink);margin-bottom:0.5rem">{html.escape(labels["title"])}</div>'
            f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:0.5rem">{"".join(cards)}</div>'
            f'</figure>')

# The lesson is done when these are correct. "venue" is the choice question, "place-paper" the open question.
REQUIRED = ["kinds", "venue", "place-paper"]

EXERCISES = {
    "kinds": Sort(
        name="kinds",
        groups={
            "method": {"en": "Method", "nl": "Methode"},
            "analysis": {"en": "Empirical analysis", "nl": "Empirische analyse"},
            "theory": {"en": "Theory", "nl": "Theorie"},
            "benchmark": {"en": "Benchmark or dataset", "nl": "Benchmark of dataset"},
            "position": {"en": "Position paper", "nl": "Position paper"},
            "reproduction": {"en": "Reproduction", "nl": "Reproductie"},
        },
        items={
            "amount": {"en": "Trains the same model on 1,000 to 100,000 examples, five seeds for each amount, and measures how its accuracy on the test set grows with the amount of data",
                       "nl": "Traint hetzelfde model op 1.000 tot 100.000 voorbeelden, met vijf seeds per hoeveelheid, en meet hoe zijn accuracy op de test set groeit met de hoeveelheid data"},
            "missing": {"en": "Proposes a new way to fill in missing values in a table, which gives more accurate models than filling in the mean, on five datasets",
                        "nl": "Stelt een nieuwe manier voor om ontbrekende waarden in een tabel in te vullen, die nauwkeurigere modellen geeft dan het gemiddelde invullen, op vijf datasets"},
            "rerun": {"en": "Runs the experiments of a 2023 paper again with the authors' code, and finds the same pattern but a gain half as big",
                      "nl": "Doet de experimenten van een paper uit 2023 opnieuw met de code van de auteurs, en vindt hetzelfde patroon maar een winst die half zo groot is"},
            "law": {"en": "Collects 2,000 questions about Dutch law, each answer checked by two lawyers, so that models can be compared on them",
                    "nl": "Verzamelt 2.000 vragen over het Nederlandse recht, met elk antwoord gecontroleerd door twee juristen, zodat modellen erop te vergelijken zijn"},
            "code": {"en": "Argues that every paper in machine learning should publish its code, and says why that matters",
                     "nl": "Betoogt dat elke paper in machine learning zijn code moet publiceren, en zegt waarom dat ertoe doet"},
            "proof": {"en": "Proves that, under three assumptions it states, a training method always ends at the lowest possible loss",
                      "nl": "Bewijst dat een trainingsmethode, onder drie aannames die de paper noemt, altijd eindigt bij de laagst mogelijke loss"},
        },
        solution={"amount": "analysis", "missing": "method", "rerun": "reproduction", "law": "benchmark", "code": "position", "proof": "theory"},
        nudge={
            "amount": {"en": "Is there anything new to use in it, or does it measure how something that already exists behaves?",
                      "nl": "Zit er iets nieuws in om te gebruiken, of meet hij hoe iets bestaands zich gedraagt?"},
            "missing": {"en": "Is there a new way of doing something in it, compared with the old way?",
                      "nl": "Zit er een nieuwe manier van doen in, vergeleken met de oude manier?"},
            "rerun": {"en": "Whose experiments are these?",
                      "nl": "Van wie zijn deze experimenten?"},
            "law": {"en": "What does this paper give others to use themselves?",
                    "nl": "Wat geeft deze paper anderen om zelf te gebruiken?"},
            "code": {"en": "Does it measure or build anything, or does it argue for something?",
                        "nl": "Meet of bouwt hij iets, of pleit hij ergens voor?"},
            "proof": {"en": "Was anything measured here, or shown with mathematics?",
                      "nl": "Is hier iets gemeten, of met wiskunde aangetoond?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the papers again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de papers opnieuw uit."},
            "empty": {"en": "Put each paper with the kind of contribution it makes.",
                      "nl": "Zet elke paper bij de soort bijdrage die hij levert."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The paper “{item}” belongs with another kind.",
                      "nl": "De paper ‘{item}’ hoort bij een andere soort."},
            "correct": {"en": "All {n} are right. Accuracy measured for many amounts of data is an empirical analysis; a new way to fill in missing values is a method; someone else's experiments done again are a reproduction; 2,000 checked questions are a benchmark; an argument for publishing code is a position paper; a proof is theory.",
                        "nl": "Alle {n} kloppen. Accuracy gemeten bij veel hoeveelheden data is een empirische analyse; een nieuwe manier om ontbrekende waarden in te vullen is een methode; de experimenten van een ander opnieuw gedaan zijn een reproductie; 2.000 gecontroleerde vragen zijn een benchmark; een pleidooi om code te publiceren is een position paper; een bewijs is theorie."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner chose, state, known mistake).
_RIGHT = {"amount": "analysis", "missing": "method", "rerun": "reproduction", "law": "benchmark", "code": "position", "proof": "theory"}
CASES = [
    ("kinds", {}, "empty", None),
    ("kinds", _RIGHT, "correct", None),
    ("kinds", {"missing": "method", "law": "benchmark"}, "wrong", "unfinished"),
    ("kinds", {**_RIGHT, "amount": "method"}, "wrong", "wrong:amount"),
    ("kinds", {**_RIGHT, "law": "analysis"}, "wrong", "wrong:law"),
]
