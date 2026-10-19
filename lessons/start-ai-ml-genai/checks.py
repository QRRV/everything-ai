# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson start-ai-ml-genai: sorting systems into the four circles."""
from everything_ai.check import Sort

# The lesson is done when these are correct. "inside" is the choice question in the lesson.
REQUIRED = ["circles", "inside"]

EXERCISES = {
    "circles": Sort(
        name="circles",
        nested=True,
        groups={
            "ai": {"en": "AI", "nl": "AI"},
            "ml": {"en": "Machine learning", "nl": "Machine learning"},
            "dl": {"en": "Deep learning", "nl": "Deep learning"},
            "genai": {"en": "Generative AI", "nl": "Generatieve AI"},
        },
        items={
            "plants": {"en": "**Plant app.** Tells you which plant is in your photo. It learned that from millions of photos of plants with their names, with a neural network of many layers.",
                       "nl": "**Planten-app.** Zegt welke plant er op je foto staat. Dat leerde hij van miljoenen foto's van planten met hun naam, met een neural network van veel layers."},
            "spam": {"en": "**Spam filter.** Learned from a million emails, each marked spam or not spam, which words point to spam. It counts words; there is no neural network in it.",
                     "nl": "**Spamfilter.** Leerde van een miljoen e-mails, elk gemarkeerd als spam of geen spam, welke woorden op spam wijzen. Hij telt woorden; er zit geen neural network in."},
            "chat": {"en": "**Chat assistant.** Writes a new answer to your question, word by word. Behind it is a large neural network that learned from a huge amount of text.",
                     "nl": "**Chatassistent.** Schrijft woord voor woord een nieuw antwoord op je vraag. Erachter zit een groot neural network dat leerde van een enorme hoeveelheid tekst."},
            "chess": {"en": "**Chess program.** Looks many moves ahead and scores each position with rules that chess players wrote down.",
                      "nl": "**Schaakprogramma.** Rekent veel zetten vooruit en geeft elke stelling een score met regels die schakers hebben opgeschreven."},
            "music": {"en": "**Music maker.** Makes a new song from a short description, with a neural network that learned from many songs.",
                      "nl": "**Muziekmaker.** Maakt een nieuw nummer bij een korte beschrijving, met een neural network dat leerde van heel veel nummers."},
        },
        solution={"plants": "dl", "spam": "ml", "chat": "genai", "chess": "ai", "music": "genai"},
        nudge={
            "plants": {"en": "Does it make something new, or does it name what is already in the photo?",
                       "nl": "Maakt hij iets nieuws, of noemt hij wat er al op de foto staat?"},
            "spam": {"en": "Is there a neural network in it?",
                     "nl": "Zit er een neural network in?"},
            "chat": {"en": "Does it make something new?",
                     "nl": "Maakt hij iets nieuws?"},
            "chess": {"en": "Who made its rules: people, or the computer from examples?",
                      "nl": "Wie maakte de regels: mensen, of de computer uit voorbeelden?"},
            "music": {"en": "Does it make something new?",
                      "nl": "Maakt hij iets nieuws?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the systems again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de systemen opnieuw uit."},
            "empty": {"en": "Pick a circle for each system.",
                      "nl": "Kies voor elk systeem een cirkel."},
            "unfinished": {"en": "Everything you picked so far is right. Pick a circle for the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe koos, klopt. Kies ook voor de rest een cirkel (nog {n})."},
            "too_wide": {"en": "{item}: that circle fits, but a smaller one fits too. Pick the smallest circle that fits.",
                         "nl": "{item}: die cirkel past, maar een kleinere past ook. Kies de kleinste cirkel die past."},
            "too_narrow": {"en": "{item} does not fit in that circle.",
                           "nl": "{item} past niet in die cirkel."},
            "correct": {"en": "All {n} systems are in the smallest circle they fit in.",
                        "nl": "Alle {n} systemen staan in de kleinste cirkel waar ze in passen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner chose, state, known mistake).
_RIGHT = {"plants": "dl", "spam": "ml", "chat": "genai", "chess": "ai", "music": "genai"}
CASES = [
    ("circles", {}, "empty", None),
    ("circles", _RIGHT, "correct", None),
    ("circles", {"plants": "dl", "spam": "ml"}, "wrong", "unfinished"),
    ("circles", {**_RIGHT, "plants": "ml"}, "almost", "too_wide:plants"),
    ("circles", {**_RIGHT, "chat": "dl"}, "almost", "too_wide:chat"),
    ("circles", {**_RIGHT, "chess": "ml"}, "wrong", "too_narrow:chess"),
    ("circles", {**_RIGHT, "spam": "dl"}, "wrong", "too_narrow:spam"),
    ("circles", {"plants": "genai", "chess": "ml"}, "wrong", "too_narrow:plants"),
]
