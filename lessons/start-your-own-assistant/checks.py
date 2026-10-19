# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson start-your-own-assistant: which requests keep the thinking with you."""
from everything_ai.check import Sort

# The lesson is done when these are correct. "trust-check" and "new-word" are choice questions in the lesson.
REQUIRED = ["ask-kinds", "trust-check", "new-word"]

_KEEPS = {"en": "After an example, a hint or a question back, is there still a step left for you? See step 2 of the recipe.",
          "nl": "Blijft er na een voorbeeld, een hint of een vraag terug nog een stap voor jou over? Kijk naar stap 2 van het recept."}
_TAKES = {"en": "Does this get you a finished answer, or your work done for you? See step 2 of the recipe.",
          "nl": "Krijg je hiermee een kant-en-klaar antwoord, of doet de assistent je werk? Kijk naar stap 2 van het recept."}

EXERCISES = {
    "ask-kinds": Sort(
        name="ask-kinds",
        groups={
            "keeps": {"en": "Keeps the thinking with you", "nl": "Jij blijft denken"},
            "takes": {"en": "Takes the thinking away", "nl": "De assistent denkt voor je"},
        },
        items={
            "example": {"en": "**Another example.** \"Explain the idea behind exercise 2 with another example, from cooking.\"",
                        "nl": "**Ander voorbeeld.** \"Leg het idee achter oefening 2 uit met een ander voorbeeld, uit de keuken.\""},
            "fill": {"en": "**Fill it in.** \"Fill in all the exercises of this lesson for me.\"",
                     "nl": "**Invullen.** \"Vul alle oefeningen van deze les voor me in.\""},
            "hint": {"en": "**One hint.** \"Give me one hint for exercise 2, not the answer.\"",
                     "nl": "**Eén hint.** \"Geef me één hint voor oefening 2, niet het antwoord.\""},
            "answer": {"en": "**Just the answer.** \"What is the answer to exercise 3? No need to explain.\"",
                       "nl": "**Alleen het antwoord.** \"Wat is het antwoord op oefening 3? Uitleg hoeft niet.\""},
            "question": {"en": "**A question back.** \"Ask me a question that helps me see my mistake.\"",
                         "nl": "**Een vraag terug.** \"Stel me een vraag waardoor ik mijn fout zelf zie.\""},
            "rewrite": {"en": "**Make it right.** \"Change my answer so that the check says Correct.\"",
                        "nl": "**Maak het goed.** \"Pas mijn antwoord aan zodat de check Goed zegt.\""},
        },
        solution={"example": "keeps", "fill": "takes", "hint": "keeps", "answer": "takes", "question": "keeps", "rewrite": "takes"},
        nudge={"example": _KEEPS, "hint": _KEEPS, "question": _KEEPS, "fill": _TAKES, "answer": _TAKES, "rewrite": _TAKES},
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the requests again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de verzoeken opnieuw uit."},
            "empty": {"en": "Pick a group for each request.",
                      "nl": "Kies voor elk verzoek een groep."},
            "unfinished": {"en": "Everything you picked so far is right. Pick a group for the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe koos, klopt. Kies ook voor de rest een groep (nog {n})."},
            "wrong": {"en": "“{item}” is in the wrong group.",
                      "nl": "‘{item}’ staat in de verkeerde groep."},
            "correct": {"en": "All {n} are right. After an example, a hint or a question back, the last step is still yours.",
                        "nl": "Alle {n} kloppen. Na een voorbeeld, een hint of een vraag terug is de laatste stap nog steeds van jou."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner chose, state, known mistake).
_RIGHT = {"example": "keeps", "fill": "takes", "hint": "keeps", "answer": "takes", "question": "keeps", "rewrite": "takes"}
CASES = [
    ("ask-kinds", {}, "empty", None),
    ("ask-kinds", _RIGHT, "correct", None),
    ("ask-kinds", {"example": "keeps", "fill": "takes"}, "wrong", "unfinished"),
    ("ask-kinds", {**_RIGHT, "hint": "takes"}, "wrong", "wrong:hint"),
    ("ask-kinds", {**_RIGHT, "rewrite": "keeps"}, "wrong", "wrong:rewrite"),
    ("ask-kinds", {**_RIGHT, "answer": "keeps", "fill": "keeps"}, "wrong", "wrong:fill"),
]
