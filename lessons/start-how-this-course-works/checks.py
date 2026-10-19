# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson start-how-this-course-works: what is right, known mistakes, messages in both languages."""
from everything_ai.check import CodeBox, Mistake

# The lesson is done when these are correct. "parts" is the choice question in the lesson.
REQUIRED = ["sum", "parts"]

EXERCISES = {
    "sum": CodeBox(
        name="sum",
        start="3 + 4",
        solution="3 + 7",
        outcome="value",
        expect=10,
        mistakes=[
            # Python cannot read these; the pattern finds out why.
            Mistake("equals-sign", syntax=r"(?<![=!<>])=(?!=)",
                    en="You do not need `=` in the box: Python works out the result itself. Remove the `=` and what comes after it.",
                    nl="Je hebt geen `=` nodig in het vak: Python rekent de uitkomst zelf uit. Haal de `=` weg, en wat erachter staat."),
            Mistake("word-for-number", syntax=r"[A-Za-z]",
                    en="Write numbers as digits, like `7`, not as words. Python only knows the words it was taught.",
                    nl="Schrijf getallen als cijfers, zoals `7`, niet als woorden. Python kent alleen de woorden die het geleerd heeft."),
            # Python can read these, but the outcome is not quite it.
            Mistake("text-not-number", matches=lambda got, text: got == "10", state="almost",
                    en="You wrote `\"10\"` between quotes, and then it is text, not a number. Remove the quotes, or write a sum that gives 10.",
                    nl="Je schreef `\"10\"` tussen aanhalingstekens, en dan is het een tekst, geen getal. Haal de aanhalingstekens weg, of schrijf een som die 10 geeft."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "The box still says `3 + 4`, which gives 7. Change one of the numbers so the result is 10.",
                      "nl": "In het vak staat nog `3 + 4`, en dat geeft 7. Verander een van de getallen, zodat de uitkomst 10 is."},
            "no_value": {"en": "Your box has no sum in it. Write a sum, like the `3 + 4` that was there.",
                         "nl": "Er staat geen som in je vak. Schrijf een som, zoals de `3 + 4` die er stond."},
            "wrong": {"en": "Your sum gives {got}, not 10. Change one of the numbers and look again.",
                      "nl": "Je som geeft {got}, niet 10. Verander een van de getallen en kijk opnieuw."},
            "correct": {"en": "Your sum gives 10. The check looked at the result, not at how you wrote it: `5 + 5` and `3 + 7` are both right.",
                        "nl": "Je som geeft 10. De check keek naar de uitkomst, niet naar hoe je het schreef: `5 + 5` en `3 + 7` zijn allebei goed."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("sum", "3 + 4", "empty", None),
    ("sum", "", "empty", None),
    ("sum", "3 + 7", "correct", None),
    ("sum", "5 + 5", "correct", None),
    ("sum", "12 - 2", "correct", None),
    ("sum", "10", "correct", None),
    ("sum", "10.0", "correct", None),
    ("sum", "3 + 4 = 10", "error", "equals-sign"),
    ("sum", "three + seven", "error", "word-for-number"),
    ("sum", '"10"', "almost", "text-not-number"),
    ("sum", "3 + 8", "wrong", None),
    ("sum", "3 +", "error", None),
    ("sum", "10 / 0", "error", None),
    ("sum", "x = 10", "wrong", "no-value"),
]
