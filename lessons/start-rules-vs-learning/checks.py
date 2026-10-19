# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson start-rules-vs-learning, and the mornings the lesson uses.

A rule here is "coat when it is colder than T degrees". A rule's mistakes are the mornings
where it says something else than what really happened."""
from everything_ai.check import Mistake, Value

# (temperature in degrees Celsius, wore a coat). Your own twelve mornings, then your neighbour's.
MINE = [(4, True), (7, True), (9, True), (10, True), (12, True), (13, True),
        (15, False), (16, False), (18, False), (20, False), (22, False), (25, False)]
NEIGHBOUR = [(6, True), (9, True), (11, True), (13, True), (15, True), (17, True), (18, True),
          (16, False), (19, False), (21, False), (23, False), (26, False)]
RULES = range(0, 27)   # every T the slider and the computer try


def mistakes(mornings, t) -> int:
    """How many mornings the rule "coat when colder than t" gets wrong."""
    return sum((temp < t) != coat for temp, coat in mornings)


def learn(mornings) -> list[int]:
    """The computer's recipe: try every rule, count its mistakes, keep the rules with the fewest."""
    counts = {t: mistakes(mornings, t) for t in RULES}
    fewest = min(counts.values())
    return [t for t, n in counts.items() if n == fewest]


# The lesson is done when these are correct. "what-changed" is the choice question in the lesson.
REQUIRED = ["neighbour", "what-changed"]

EXERCISES = {
    "neighbour": Value(
        name="neighbour",
        start=14,
        solution=19,
        outcome=lambda t: mistakes(NEIGHBOUR, t),
        mistakes=[
            Mistake("two-mistakes", matches=lambda got, t: got == 2, state="almost",
                    en="With {value} degrees your rule makes 2 mistakes on your neighbor's mornings. Close: there is a rule with fewer.",
                    nl="Met {value} graden maakt je regel 2 fouten op de ochtenden van je buur. Dichtbij: er is een regel met minder."),
        ],
        texts={
            "missing": {"en": "I cannot see the slider. Run the cell with the slider again.",
                        "nl": "Ik zie de schuifbalk niet. Voer de cel met de schuifbalk opnieuw uit."},
            "empty": {"en": "The slider is still at your own rule, 14 degrees. On your neighbor's mornings it makes 3 mistakes. Move the slider.",
                      "nl": "De schuifbalk staat nog op je eigen regel, 14 graden. Op de ochtenden van je buur maakt die 3 fouten. Verschuif hem."},
            "wrong": {"en": "With {value} degrees your rule makes {got} mistakes on your neighbor's mornings. There is a rule with fewer. Watch the rings while you move the slider: each ring is a mistake.",
                      "nl": "Met {value} graden maakt je regel {got} fouten op de ochtenden van je buur. Er is een regel met minder. Let op de ringen terwijl je de schuifbalk verschuift: elke ring is een fout."},
            "correct": {"en": "With {value} degrees your rule makes only 1 mistake, and no rule of this kind makes fewer. One morning never fits: at 16 degrees your neighbor wore no coat.",
                        "nl": "Met {value} graden maakt je regel maar 1 fout, en geen enkele regel van deze soort maakt er minder. Eén ochtend past nooit: bij 16 graden droeg je buur geen jas."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, where the slider is, state, known mistake).
CASES = [
    ("neighbour", 14, "empty", None),
    ("neighbour", 19, "correct", None),
    ("neighbour", 16, "almost", "two-mistakes"),
    ("neighbour", 18, "almost", "two-mistakes"),
    ("neighbour", 20, "almost", "two-mistakes"),
    ("neighbour", 10, "wrong", None),
    ("neighbour", 25, "wrong", None),
    ("neighbour", 0, "wrong", None),
]
