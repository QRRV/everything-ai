# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-research-cycle: sorting the sentences of one study into question, hypothesis, experiment
and paper. And the picture of the research cycle."""
import html

from everything_ai.check import Sort


def cycle(labels: dict) -> str:
    """The research cycle as four boxes in a square, with arrows round it: question, hypothesis, experiment, paper,
    and back from the paper to a new question. Ink only: nothing here is a role of a model. Boxes of HTML, so the text
    wraps and keeps its size in a narrow window."""
    boxes = labels["boxes"]   # four (name, text) pairs, in the order of the cycle

    def box(k: int) -> str:
        name, text = boxes[k]
        return (f'<div style="box-sizing:border-box;padding:0.5rem 0.7rem;border:1.2px solid var(--ea-ink-3);border-radius:8px;'
                f'background:var(--ea-inset);font-size:0.875rem;line-height:1.35;color:var(--ea-ink-2);text-align:center">'
                f'<span style="display:block;font-weight:650;color:var(--ea-ink)">{k + 1}. {html.escape(name)}</span>'
                f'{html.escape(text)}</div>')

    arrow = 'style="display:flex;align-items:center;justify-content:center;color:var(--ea-ink-2);font-size:1.25rem"'
    note = 'style="font-size:0.8rem;color:var(--ea-ink-2);text-align:center"'
    grid = (f'<div style="display:grid;grid-template-columns:1fr 2rem 1fr;grid-template-rows:auto 2.2rem auto;gap:0.25rem">'
            f'{box(0)}<div {arrow}>→</div>{box(1)}'
            f'<div {arrow}>↑ <span {note}>&nbsp;{html.escape(labels["again"])}</span></div><div></div><div {arrow}>↓</div>'
            f'{box(3)}<div {arrow}>←</div>{box(2)}</div>')
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(labels["alt"])}" '
            f'style="margin:0.25rem 0 0.75rem;max-width:560px;font-family:var(--ea-font-ui)">'
            f'<div style="font-size:1rem;font-weight:600;color:var(--ea-ink);margin-bottom:0.5rem">{html.escape(labels["title"])}</div>'
            f'{grid}</figure>')


# The lesson is done when these are correct. "testable" is the choice question, "design" the open question.
REQUIRED = ["cycle-steps", "testable", "design"]

EXERCISES = {
    "cycle-steps": Sort(
        name="cycle-steps",
        groups={
            "question": {"en": "Question", "nl": "Vraag"},
            "hypothesis": {"en": "Hypothesis", "nl": "Hypothese"},
            "experiment": {"en": "Experiment", "nl": "Experiment"},
            "paper": {"en": "Paper", "nl": "Paper"},
        },
        items={
            "fewer": {"en": "The chatbot gets fewer of the 300 test questions right in Dutch than in English.",
                      "nl": "De chatbot heeft in het Nederlands minder van de 300 testvragen goed dan in het Engels."},
            "online": {"en": "The researchers write down what they did and found, and put the questions and answers online.",
                       "nl": "De onderzoekers schrijven op wat ze deden en vonden, en zetten de vragen en antwoorden online."},
            "worse": {"en": "The researchers want to find out whether the chatbot is worse at Dutch than at English.",
                      "nl": "De onderzoekers willen weten of de chatbot slechter is in het Nederlands dan in het Engels."},
            "both": {"en": "The same chatbot answers all 300 questions in both languages, and every answer is marked right or wrong.",
                     "nl": "Dezelfde chatbot beantwoordt alle 300 vragen in beide talen, en elk antwoord wordt goed of fout gerekend."},
            "other": {"en": "Is the gap as big for other languages?",
                      "nl": "Is het verschil even groot voor andere talen?"},
            "translate": {"en": "A translator translates the 300 questions into Dutch.",
                          "nl": "Een vertaler vertaalt de 300 vragen naar het Nederlands."},
        },
        # In mixed order on purpose: in the order of the cycle, the answer would run down the groups.
        solution={"fewer": "hypothesis", "online": "paper", "worse": "question", "both": "experiment",
                  "other": "question", "translate": "experiment"},
        nudge={
            "fewer": {"en": "Does this sentence give an answer that a count could prove wrong?",
                      "nl": "Geeft deze zin een antwoord dat een telling kan weerleggen?"},
            "online": {"en": "When does this happen, and what does it let others do?",
                       "nl": "Wanneer gebeurt dit, en wat kunnen anderen daardoor doen?"},
            "worse": {"en": "Does this sentence give an answer that a count could prove wrong?",
                      "nl": "Geeft deze zin een antwoord dat een telling kan weerleggen?"},
            "both": {"en": "What do the researchers get out of this, and what do they need it for?",
                     "nl": "Wat levert dit de onderzoekers op, en waarvoor hebben ze dat nodig?"},
            "other": {"en": "A result often raises something new to find out. Where in the cycle does that go?",
                      "nl": "Een uitkomst roept vaak iets nieuws op om uit te zoeken. Waar in de cyclus hoort dat?"},
            "translate": {"en": "What do the researchers need the Dutch questions for?",
                          "nl": "Waarvoor hebben de onderzoekers de Nederlandse vragen nodig?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the sentences again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de zinnen opnieuw uit."},
            "empty": {"en": "Put each sentence in a step of the cycle: question, hypothesis, experiment or paper.",
                      "nl": "Zet elke zin bij een stap van de cyclus: vraag, hypothese, experiment of paper."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The sentence “{item}” belongs to another step.",
                      "nl": "De zin ‘{item}’ hoort bij een andere stap."},
            "correct": {"en": "All {n} are right. Two questions, the first and the new one the result raised; one hypothesis that a count can prove wrong; two parts of the experiment; and the paper, so that others can check it.",
                        "nl": "Alle {n} kloppen. Twee vragen, de eerste en de nieuwe die de uitkomst opriep; één hypothese die een telling kan weerleggen; twee delen van het experiment; en de paper, zodat anderen het kunnen controleren."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner chose, state, known mistake).
_RIGHT = {"worse": "question", "fewer": "hypothesis", "translate": "experiment", "both": "experiment", "online": "paper", "other": "question"}
CASES = [
    ("cycle-steps", {}, "empty", None),
    ("cycle-steps", _RIGHT, "correct", None),
    ("cycle-steps", {"worse": "question", "fewer": "hypothesis"}, "wrong", "unfinished"),
    ("cycle-steps", {**_RIGHT, "fewer": "question"}, "wrong", "wrong:fewer"),
    ("cycle-steps", {**_RIGHT, "other": "paper"}, "wrong", "wrong:other"),
]
