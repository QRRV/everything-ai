# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson start-using-ai-tools, and the made-up next words the lesson shows."""
from everything_ai.check import Sort

# A made-up example of what a language model could see after "The windmill in our town was built in".
# Several years sound likely; the model writes one of them, just as fluently as a year it really knows.
NEXT_WORDS = {
    "en": [("1782", 0.24), ("1790", 0.19), ("1856", 0.14), ("1650", 0.09), ("the", 0.05)],
    "nl": [("1782", 0.24), ("1790", 0.19), ("1856", 0.14), ("1650", 0.09), ("het", 0.05)],
}

# The lesson is done when these are correct. "how-check" is a choice question in the lesson.
REQUIRED = ["what-to-check", "how-check"]

_FACT = {"en": "Would the day go wrong if this were not true? Then it is a fact the day depends on.",
         "nl": "Gaat de dag mis als dit niet klopt? Dan is het een feit waar de dag van afhangt."}
_TIP = {"en": "Is this a fact, or a suggestion that you decide on yourself?",
        "nl": "Is dit een feit, of een voorstel waar je zelf over beslist?"}

EXERCISES = {
    "what-to-check": Sort(
        name="what-to-check",
        groups={
            "fact": {"en": "A fact the day depends on", "nl": "Een feit waar de dag van afhangt"},
            "no-fact": {"en": "Not a fact", "nl": "Geen feit"},
        },
        items={
            "hours": {"en": "**Opening hours.** The museum is open from 10:00 to 17:00.",
                      "nl": "**Openingstijden.** Het museum is open van 10.00 tot 17.00 uur."},
            "lift": {"en": "**Wheelchair.** Every room can be reached by lift.",
                     "nl": "**Rolstoel.** Elke zaal is met een lift te bereiken."},
            "break": {"en": "**Tip.** Take a break halfway, so the day does not get too tiring.",
                      "nl": "**Tip.** Neem halverwege een pauze, dan wordt de dag niet te zwaar."},
            "price": {"en": "**Tickets.** A ticket costs 15 euros, and people over 65 pay 10.",
                      "nl": "**Kaartjes.** Een kaartje kost 15 euro, en 65-plussers betalen 10."},
            "first": {"en": "**Idea.** Ask your grandmother which room she wants to see first.",
                      "nl": "**Idee.** Vraag je oma welke zaal ze als eerste wil zien."},
            "train": {"en": "**Train.** The 9:12 train gets you there in 40 minutes.",
                      "nl": "**Trein.** Met de trein van 9.12 uur ben je er in 40 minuten."},
        },
        solution={"hours": "fact", "lift": "fact", "break": "no-fact", "price": "fact", "first": "no-fact", "train": "fact"},
        nudge={"hours": _FACT, "lift": _FACT, "price": _FACT, "train": _FACT, "break": _TIP, "first": _TIP},
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the six parts again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de zes delen opnieuw uit."},
            "empty": {"en": "Pick a group for each part of the answer.",
                      "nl": "Kies voor elk deel van het antwoord een groep."},
            "unfinished": {"en": "Everything you picked so far is right. Pick a group for the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe koos, klopt. Kies ook voor de rest een groep (nog {n})."},
            "wrong": {"en": "“{item}” is in the wrong group.",
                      "nl": "‘{item}’ staat in de verkeerde groep."},
            "correct": {"en": "All {n} are right: four facts the day depends on, which you check in step 3, and two suggestions you decide on yourself.",
                        "nl": "Alle {n} kloppen: vier feiten waar de dag van afhangt, die je in stap 3 controleert, en twee voorstellen waar je zelf over beslist."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner chose, state, known mistake).
_RIGHT = {"hours": "fact", "lift": "fact", "break": "no-fact", "price": "fact", "first": "no-fact", "train": "fact"}
CASES = [
    ("what-to-check", {}, "empty", None),
    ("what-to-check", _RIGHT, "correct", None),
    ("what-to-check", {"hours": "fact", "lift": "fact"}, "wrong", "unfinished"),
    ("what-to-check", {**_RIGHT, "lift": "no-fact"}, "wrong", "wrong:lift"),
    ("what-to-check", {**_RIGHT, "first": "fact"}, "wrong", "wrong:first"),
    ("what-to-check", {**_RIGHT, "train": "no-fact", "break": "fact"}, "wrong", "wrong:break"),
]
