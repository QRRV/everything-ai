# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-choose-paper: sorting candidate claims into a good first reproduction and not now."""
from everything_ai.check import Sort

LAPTOP = 1e11                      # FLOP/s that training a small model reaches on a laptop's CPU, about
LAPTOP_DAY = LAPTOP * 86_400
GPU = 312e12 * 0.4                 # an A100 in BF16 at 40% of its peak
GPU_WEEK = GPU * 7 * 86_400
# Four candidate experiments: parameters and training tokens of one run.
CANDIDATES = [(0.4e6, 0.25e9), (10e6, 1e9), (124e6, 10e9), (7e9, 2e12)]


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def budget_picture(labels: dict) -> str:
    """The training compute of one run of four candidate experiments, 6ND, as bars on a log scale, against a day of a
    laptop and a week of one GPU, dashed. Compute is neither learned nor a loss: ink."""
    from everything_ai import chart
    p = chart.Panel((1e14, 1e24), (-0.5, 5.2), title=labels["title"], xlabel=labels["x"], ylabel="", xlog=True, yticks=[])
    rows = [4.4 - i * 1.05 for i in range(len(CANDIDATES))]
    for x, name, y in ((LAPTOP_DAY, labels["laptop"], 0.45), (GPU_WEEK, labels["gpu"], -0.2)):
        bottom = -0.5
        for row in sorted(rows):   # the dashed line stops at each bar's name, so it never runs through the words
            p.line([x, x], [bottom, row + 0.18], "ink", 1.4, dashed=True)
            bottom = row + 0.64
        p.line([x, x], [bottom, 5.2], "ink", 1.4, dashed=True)
        p.text(x * 1.4, y, name, "ink-2", "start")
    for (n, d), name, y in zip(CANDIDATES, labels["names"], rows):
        for _ in range(3):
            p.rect(1e14, y - 0.22, 6 * n * d, y + 0.22, "ink-3")
        p.text(1.4e14, y + 0.42, name, "ink", "start")
    return chart.figure([p], alt=labels["alt"], width=330, height=310)


_SOLUTION = {"grokking": "good", "big": "not", "dropout": "good", "private": "not", "smoothing": "good", "understand": "not"}

EXERCISES = {
    "first-repro": Sort(
        name="first-repro",
        groups={
            "good": {"en": "A good first reproduction", "nl": "Een goede eerste reproductie"},
            "not": {"en": "Not now", "nl": "Nu niet"},
        },
        items={
            "grokking": {"en": "In a small transformer trained on modular addition, the validation accuracy jumps to almost 100% long after the training accuracy did.",
                         "nl": "In een kleine transformer, getraind op optellen modulo een getal, springt de validation accuracy naar bijna 100%, lang nadat de training accuracy dat deed."},
            "big": {"en": "A model of 70 billion parameters, trained on 15 trillion tokens, beats earlier models on a benchmark.",
                    "nl": "Een model van 70 miljard parameters, getraind op 15 biljoen tokens, verslaat eerdere modellen op een benchmark."},
            "dropout": {"en": "Dropout of 0.1 lowers the validation loss of a small character-level GPT on Tiny Shakespeare.",
                        "nl": "Dropout van 0,1 verlaagt de validation loss van een kleine GPT op de tekens van Tiny Shakespeare."},
            "private": {"en": "Our method gives better diagnoses on the records of one hospital, which cannot be shared.",
                        "nl": "Onze methode geeft betere diagnoses op de dossiers van één ziekenhuis, die niet gedeeld mogen worden."},
            "smoothing": {"en": "With batch normalization, a small network on the handwritten digits reaches the same accuracy at a 10 times higher learning rate.",
                          "nl": "Met batch normalization haalt een klein netwerk op de handgeschreven cijfers dezelfde accuracy bij een 10 keer hogere learning rate."},
            "understand": {"en": "Our approach is a step towards machines that truly understand language.",
                           "nl": "Onze aanpak is een stap naar machines die taal echt begrijpen."},
        },
        solution=_SOLUTION,
        nudge={
            "grokking": {"en": "How big is the model, and where does the data come from?",
                         "nl": "Hoe groot is het model, en waar komt de data vandaan?"},
            "big": {"en": "Put 6ND next to the picture: how far right does that bar go?",
                    "nl": "Leg 6ND naast het plaatje: hoe ver naar rechts gaat die staaf?"},
            "dropout": {"en": "You trained this GPT yourself: how long did that take?",
                        "nl": "Je hebt deze GPT zelf getraind: hoe lang duurde dat?"},
            "private": {"en": "Can you get the data the claim rests on?",
                        "nl": "Kun je aan de data komen waar de claim op rust?"},
            "smoothing": {"en": "Is the model small, the data public, and the claim a number you can measure?",
                          "nl": "Is het model klein, de data openbaar, en de claim een getal dat je kunt meten?"},
            "understand": {"en": "Which result would show this claim false?",
                           "nl": "Welk resultaat zou laten zien dat deze claim niet klopt?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the claims again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de claims opnieuw uit."},
            "empty": {"en": "Put each claim in a group: a good first reproduction, or not now.",
                      "nl": "Zet elke claim in een groep: een goede eerste reproductie, of nu niet."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The claim “{item}” belongs in the other group.",
                      "nl": "De claim ‘{item}’ hoort in de andere groep."},
            "correct": {"en": "All {n} are right. The small transformer, the small GPT and the small network on the digits fit a laptop, use public or generated data, and make a claim you can measure. The 70-billion model needs far more compute, the hospital's records cannot be had, and \"truly understand\" names no result that could show it false.",
                        "nl": "Alle {n} kloppen. De kleine transformer, de kleine GPT en het kleine netwerk op de cijfers passen op een laptop, gebruiken openbare of gegenereerde data en doen een claim die je kunt meten. Het model van 70 miljard vraagt veel meer compute, de dossiers van het ziekenhuis krijg je niet, en \"echt begrijpen\" noemt geen resultaat dat kan laten zien dat het niet klopt."},
        },
    ),
}

# The lesson is done when these are correct. "falsifiable" is the choice question, "my-plan" the open question.
REQUIRED = ["first-repro", "falsifiable", "my-plan"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("first-repro", {}, "empty", None),
    ("first-repro", dict(_SOLUTION), "correct", None),
    ("first-repro", {"grokking": "good", "big": "not"}, "wrong", "unfinished"),
    ("first-repro", {**_SOLUTION, "smoothing": "not"}, "wrong", "wrong:smoothing"),
]
