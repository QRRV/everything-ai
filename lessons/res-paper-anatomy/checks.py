# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-paper-anatomy: sorting pieces of a paper by the part they come from. And the picture of
the parts of a paper."""
import html

from everything_ai.check import Sort


def anatomy(labels: dict) -> str:
    """A paper as a stack of parts, each with the question it answers: the main text, then the references and the
    appendix. Ink only: nothing here is a role of a model. HTML, not an SVG, so the text wraps in a narrow window."""
    def part(name, question):
        return (f'<div style="display:flex;gap:0.6rem;align-items:baseline;padding:0.35rem 0.6rem;border-bottom:1px solid var(--ea-line)">'
                f'<span style="flex:0 0 38%;font-weight:600;color:var(--ea-ink)">{html.escape(name)}</span>'
                f'<span style="flex:1;color:var(--ea-ink-2)">{html.escape(question)}</span></div>')

    def block(title, parts):
        return (f'<div style="border:1.2px solid var(--ea-ink-3);border-radius:6px;background:var(--ea-inset);margin-bottom:0.5rem">'
                f'<div style="padding:0.3rem 0.6rem;font-size:0.8rem;color:var(--ea-ink-3)">{html.escape(title)}</div>'
                f'{"".join(part(n, q) for n, q in parts)}</div>')

    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(labels["alt"])}" '
            f'style="margin:0.25rem 0 0.75rem;max-width:560px;font-family:var(--ea-font-ui);font-size:0.875rem;line-height:1.35">'
            f'<div style="font-size:1rem;font-weight:600;color:var(--ea-ink);margin-bottom:0.5rem">{html.escape(labels["title"])}</div>'
            f'{block(labels["main"], labels["main_parts"])}{block(labels["after"], labels["after_parts"])}</figure>')


# The lesson is done when these are correct. "fair-baselines" is the choice question, "repro-list" the open question.
REQUIRED = ["where", "fair-baselines", "repro-list"]

EXERCISES = {
    "where": Sort(
        name="where",
        groups={
            "intro": {"en": "Introduction", "nl": "Introductie"},
            "related": {"en": "Related work", "nl": "Related work"},
            "method": {"en": "Method", "nl": "Methode"},
            "experiments": {"en": "Experiments", "nl": "Experimenten"},
            "limits": {"en": "Conclusion and limitations", "nl": "Conclusie en beperkingen"},
            "appendix": {"en": "Appendix", "nl": "Appendix"},
        },
        items={
            "table": {"en": "“Table 2: accuracy on the test set for each width, the mean and the spread over five seeds.”",
                      "nl": "“Tabel 2: accuracy op de test set voor elke breedte, het gemiddelde en de spreiding over vijf seeds.”"},
            "settings": {"en": "“B.2 Training details. All networks train with SGD, a learning rate of 0.1 and a batch size of 128, for 20 epochs.”",
                         "nl": "“B.2 Trainingsdetails. Alle netwerken trainen met SGD, een learning rate van 0,1 en een batch size van 128, 20 epochs lang.”"},
            "contrib": {"en": "“Our contributions are: (1) a comparison at equal numbers of parameters; (2) a rule for choosing the width.”",
                        "nl": "“Onze bijdragen zijn: (1) een vergelijking bij gelijke aantallen parameters; (2) een regel om de breedte te kiezen.”"},
            "only-digits": {"en": "“We only tested handwritten digits; whether this holds for photos, we do not know.”",
                            "nl": "“We testten alleen handgeschreven cijfers; of dit ook voor foto's geldt, weten we niet.”"},
            "pixels": {"en": "“Each image of 28 by 28 pixels becomes a vector of 784 numbers, which goes into the first hidden layer, with a ReLU after every hidden layer.”",
                       "nl": "“Elk plaatje van 28 bij 28 pixels wordt een vector van 784 getallen, die naar de eerste hidden layer gaat, met een ReLU na elke hidden layer.”"},
            "earlier": {"en": "“Lee et al. (2019) compared networks of one to four hidden layers, and Park et al. (2021) widths from 64 to 1,024. In both, the bigger networks also had more parameters; ours do not.”",
                        "nl": "“Lee et al. (2019) vergeleken netwerken met één tot vier hidden layers, en Park et al. (2021) breedtes van 64 tot 1.024. In beide hadden de grotere netwerken ook meer parameters; de onze niet.”"},
        },
        solution={"table": "experiments", "settings": "appendix", "contrib": "intro", "only-digits": "limits", "pixels": "method", "earlier": "related"},
        nudge={
            "table": {"en": "Is this how the network works, or what came out when it was measured?",
                      "nl": "Is dit hoe het netwerk werkt, of wat eruit kwam toen het gemeten werd?"},
            "settings": {"en": "Which part numbers its sections with letters, and holds the details someone needs to rebuild the work?",
                         "nl": "Welk deel nummert zijn paragrafen met letters, en bevat de details die iemand nodig heeft om het werk na te bouwen?"},
            "contrib": {"en": "Where does a paper say, early on, what it adds?",
                        "nl": "Waar zegt een paper, al vroeg, wat hij toevoegt?"},
            "only-digits": {"en": "Is this a result, or where the result stops?",
                            "nl": "Is dit een resultaat, of waar het resultaat ophoudt?"},
            "pixels": {"en": "Does this say what came out, or what exactly the model does?",
                       "nl": "Zegt dit wat eruit kwam, of wat het model precies doet?"},
            "earlier": {"en": "Does this say what the paper itself adds, or what earlier papers did and how this one differs from them?",
                        "nl": "Zegt dit wat de paper zelf toevoegt, of wat eerdere papers deden en hoe deze daarvan verschilt?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the pieces again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de stukjes opnieuw uit."},
            "empty": {"en": "Put each piece with the part of the paper it most likely comes from.",
                      "nl": "Zet elk stukje bij het deel van de paper waar het waarschijnlijk uit komt."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The piece {item} most likely comes from another part.",
                      "nl": "Het stukje {item} komt waarschijnlijk uit een ander deel."},
            "correct": {"en": "All {n} are right. Each piece answers the question of its part, and the exact settings that someone needs to rebuild the work are in the appendix.",
                        "nl": "Alle {n} kloppen. Elk stukje beantwoordt de vraag van zijn deel, en de precieze instellingen die iemand nodig heeft om het werk na te bouwen, staan in de appendix."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner chose, state, known mistake).
_RIGHT = {"table": "experiments", "settings": "appendix", "contrib": "intro", "only-digits": "limits", "pixels": "method", "earlier": "related"}
CASES = [
    ("where", {}, "empty", None),
    ("where", _RIGHT, "correct", None),
    ("where", {"table": "experiments", "pixels": "method"}, "wrong", "unfinished"),
    ("where", {**_RIGHT, "settings": "method"}, "wrong", "wrong:settings"),
    ("where", {**_RIGHT, "contrib": "related"}, "wrong", "wrong:contrib"),
    ("where", {**_RIGHT, "earlier": "intro"}, "wrong", "wrong:earlier"),
]
