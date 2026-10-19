# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson start-short-history, and the events and winters its timeline shows."""
from everything_ai.check import Sort

# The timeline. Years of the winters are approximate; historians do not agree on exact years.
WINTERS = [(1974, 1980), (1987, 1993)]
# A sketch of how much hope and money there was for AI, from 0 (little) to 1 (a lot), as of 2026. Not a measurement.
WAVE = [(1950, 0.12), (1956, 0.45), (1970, 0.72), (1974, 0.22), (1980, 0.28), (1985, 0.70), (1987, 0.64),
        (1991, 0.18), (1993, 0.24), (2000, 0.34), (2011, 0.44), (2014, 0.68), (2020, 0.80), (2023, 1.0), (2026, 1.0)]
EVENTS = {
    "en": [(1956, "Dartmouth", "A summer workshop under a new name, proposed in 1955: artificial intelligence."),
           (1958, "Perceptron", "One of the first neurons that learn: a small calculation with numbers it adjusts after each mistake."),
           (1973, "Lighthill report", "A British report is critical of AI, and Britain cuts most of its money for AI research."),
           (1986, "Backpropagation", "A way to let neural networks with layers learn from their mistakes becomes widely known."),
           (1997, "Deep Blue", "A chess computer beats the world champion."),
           (2012, "AlexNet", "A deep neural network recognizes things in photos far better than anything before."),
           (2022, "ChatGPT", "A chat assistant that anyone can use reaches the general public.")],
    "nl": [(1956, "Dartmouth", "Een zomerworkshop onder een nieuwe naam, voorgesteld in 1955: artificial intelligence."),
           (1958, "Perceptron", "Een van de eerste neuronen die leren: een kleine berekening met getallen die hij na elke fout bijstelt."),
           (1973, "Lighthill-rapport", "Een Brits rapport is kritisch over AI, en Groot-Brittannië schrapt het meeste geld voor AI-onderzoek."),
           (1986, "Backpropagation", "Een manier om neural networks met layers van hun fouten te laten leren, raakt breed bekend."),
           (1997, "Deep Blue", "Een schaakcomputer verslaat de wereldkampioen."),
           (2012, "AlexNet", "Een diep neural network herkent dingen op foto's veel beter dan alles ervoor."),
           (2022, "ChatGPT", "Een chatassistent die iedereen kan gebruiken, bereikt het grote publiek.")],
}

# The lesson is done when these are correct. "winter" and "why-2012" are choice questions in the lesson.
REQUIRED = ["rules-or-learning", "winter", "why-2012"]

EXERCISES = {
    "rules-or-learning": Sort(
        name="rules-or-learning",
        groups={
            "rules": {"en": "Rules written by people", "nl": "Regels van mensen"},
            "learning": {"en": "Learned from examples", "nl": "Geleerd van voorbeelden"},
        },
        items={
            "eliza": {"en": "**ELIZA (1966).** A chat program that answered with patterns its maker had written down. To \"I am unhappy\" it could answer \"Do you think coming here will help you not to be unhappy?\".",
                      "nl": "**ELIZA (1966).** Een chatprogramma dat antwoordde met patronen die de maker had opgeschreven. Op \"I am unhappy\" kon het antwoorden: \"Do you think coming here will help you not to be unhappy?\"."},
            "perceptron": {"en": "**Perceptron (1958).** Looked at examples, and adjusted its numbers a little after every mistake.",
                           "nl": "**Perceptron (1958).** Bekeek voorbeelden, en stelde na elke fout zijn getallen een beetje bij."},
            "expert": {"en": "**Expert system (1980s).** Gave advice in one narrow field, with hundreds of if-then rules that its makers wrote down together with human experts.",
                       "nl": "**Expertsysteem (jaren tachtig).** Gaf advies in één smal vakgebied, met honderden als-dan-regels die de makers samen met menselijke experts opschreven."},
            "alexnet": {"en": "**AlexNet (2012).** A neural network that got good at recognizing things in photos by seeing more than a million photos, each marked with what is on it.",
                        "nl": "**AlexNet (2012).** Een neural network dat goed werd in het herkennen van dingen op foto's door meer dan een miljoen foto's te zien, elk met wat erop staat."},
            "shrdlu": {"en": "**SHRDLU (1970).** Moved blocks in a small world on the screen when you typed what to do, following steps its maker had programmed.",
                       "nl": "**SHRDLU (1970).** Verplaatste blokken in een kleine wereld op het scherm als je typte wat het moest doen, met stappen die de maker had geprogrammeerd."},
            "chatgpt": {"en": "**ChatGPT (2022).** A chat assistant whose neural network got good at writing by seeing a huge amount of text.",
                        "nl": "**ChatGPT (2022).** Een chatassistent waarvan het neural network goed werd in schrijven door een enorme hoeveelheid tekst te zien."},
        },
        solution={"eliza": "rules", "perceptron": "learning", "expert": "rules", "alexnet": "learning",
                  "shrdlu": "rules", "chatgpt": "learning"},
        nudge={
            "eliza": {"en": "Where did its patterns come from?", "nl": "Waar kwamen de patronen vandaan?"},
            "perceptron": {"en": "Who changed its numbers: a person, or the perceptron itself after a mistake?",
                           "nl": "Wie veranderde zijn getallen: een mens, of het perceptron zelf na een fout?"},
            "expert": {"en": "Who wrote its rules?", "nl": "Wie schreef de regels?"},
            "alexnet": {"en": "Did anyone write down what a cat looks like, or did it get good by seeing photos?",
                        "nl": "Schreef iemand op hoe een kat eruitziet, of werd het goed door foto's te zien?"},
            "shrdlu": {"en": "Where did its steps come from?", "nl": "Waar kwamen de stappen vandaan?"},
            "chatgpt": {"en": "Did people write down how it should answer, or did it get good by seeing text?",
                        "nl": "Schreven mensen op hoe het moest antwoorden, of werd het goed door tekst te zien?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the systems again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de systemen opnieuw uit."},
            "empty": {"en": "Pick a group for each system.",
                      "nl": "Kies voor elk systeem een groep."},
            "unfinished": {"en": "Everything you picked so far is right. Pick a group for the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe koos, klopt. Kies ook voor de rest een groep (nog {n})."},
            "wrong": {"en": "“{item}” is in the wrong group.",
                      "nl": "‘{item}’ staat in de verkeerde groep."},
            "correct": {"en": "All {n} are right. Look at the years too: rules were the main way until the 1990s, learning took over after that.",
                        "nl": "Alle {n} kloppen. Kijk ook naar de jaartallen: tot in de jaren negentig vooral regels, daarna nam leren het over."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner chose, state, known mistake).
_RIGHT = {"eliza": "rules", "perceptron": "learning", "expert": "rules", "alexnet": "learning", "shrdlu": "rules", "chatgpt": "learning"}
CASES = [
    ("rules-or-learning", {}, "empty", None),
    ("rules-or-learning", _RIGHT, "correct", None),
    ("rules-or-learning", {"eliza": "rules", "perceptron": "learning"}, "wrong", "unfinished"),
    ("rules-or-learning", {**_RIGHT, "eliza": "learning"}, "wrong", "wrong:eliza"),
    ("rules-or-learning", {**_RIGHT, "perceptron": "rules"}, "wrong", "wrong:perceptron"),
    ("rules-or-learning", {**_RIGHT, "chatgpt": "rules", "shrdlu": "learning"}, "wrong", "wrong:shrdlu"),
]
