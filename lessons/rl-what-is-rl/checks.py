# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-what-is-rl: which part of a task is the agent, the environment, an action and a reward;
discounted_return, the return of a list of rewards with a discount factor gamma."""
import re

from everything_ai.check import FunctionBox, Mistake, Sort


def discounted_return(rewards, gamma) -> float:
    """The course's own return: r_1 + gamma * r_2 + gamma^2 * r_3 + ..."""
    G, factor = 0.0, 1.0
    for r in rewards:
        G += factor * r
        factor *= gamma
    return G


# The lesson is done when these are correct. "gamma-zero" is the choice question in the lesson.
REQUIRED = ["parts", "discounted-return", "gamma-zero"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _short(v):
    """A number for the check's messages: rounded to 4 decimals, whole numbers without .0."""
    try:
        x = round(float(v), 4)
    except (TypeError, ValueError):
        return repr(v)
    return repr(int(x)) if x == int(x) else repr(x)


# --- exercise 2: discounted_return --------------------------------------------------------------------------------------

_D_HEAD = "def discounted_return(rewards, gamma):\n    G = 0\n    factor = 1\n    for r in rewards:\n"
_D_START = _D_HEAD + "        G = ...\n        factor = ...\n    return G\n"


def _d(body: str = "        G = G + factor * r\n        factor = factor * gamma\n") -> str:
    return _D_HEAD + body + "    return G\n"


_D = _d()

EXERCISES = {
    "parts": Sort(
        name="parts",
        groups={
            "agent": {"en": "Agent", "nl": "Agent"},
            "environment": {"en": "Environment", "nl": "Environment"},
            "action": {"en": "Action", "nl": "Action"},
            "reward": {"en": "Reward", "nl": "Reward"},
        },
        items={
            "rooms": {"en": "The rooms, with their dust and furniture", "nl": "De kamers, met hun stof en meubels"},
            "knight": {"en": "Move a knight", "nl": "Een paard verzetten"},
            "vacuum": {"en": "A robot vacuum cleaner", "nl": "Een robotstofzuiger"},
            "win": {"en": "+1 for a won game, −1 for a lost one", "nl": "+1 voor een gewonnen partij, −1 voor een verloren partij"},
            "turn": {"en": "Turn left", "nl": "Naar links draaien"},
            "program": {"en": "A program that plays chess", "nl": "Een programma dat schaakt"},
            "clean": {"en": "+1 for every square metre that gets clean", "nl": "+1 voor elke vierkante meter die schoon wordt"},
            "board": {"en": "The board and the opponent", "nl": "Het bord en de tegenstander"},
        },
        solution={"rooms": "environment", "knight": "action", "vacuum": "agent", "win": "reward",
                  "turn": "action", "program": "agent", "clean": "reward", "board": "environment"},
        nudge={
            "vacuum": {"en": "Who chooses what happens next in the cleaning?",
                       "nl": "Wie kiest wat er bij het schoonmaken gebeurt?"},
            "program": {"en": "Who decides which move to make?",
                        "nl": "Wie beslist welke zet er komt?"},
            "rooms": {"en": "Is this what chooses, or what the choices act on?",
                      "nl": "Is dit wat kiest, of waar de keuzes op werken?"},
            "board": {"en": "Does the program choose what the opponent does?",
                      "nl": "Kiest het programma wat de tegenstander doet?"},
            "turn": {"en": "Is this something the agent does, or something it gets?",
                     "nl": "Is dit iets wat de agent doet, of iets wat hij krijgt?"},
            "knight": {"en": "Is this something the agent does, or something it gets?",
                       "nl": "Is dit iets wat de agent doet, of iets wat hij krijgt?"},
            "clean": {"en": "Is this a choice, or a number that says how good a step was?",
                      "nl": "Is dit een keuze, of een getal dat zegt hoe goed een stap was?"},
            "win": {"en": "Is this a choice, or a number that says how good the game went?",
                    "nl": "Is dit een keuze, of een getal dat zegt hoe goed het spel ging?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the parts again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de onderdelen opnieuw uit."},
            "empty": {"en": "Put each part of the two tasks in a group: agent, environment, action or reward.",
                      "nl": "Zet elk onderdeel van de twee taken in een groep: agent, environment, action of reward."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs in another group.",
                      "nl": "{item} hoort in een andere groep."},
            "correct": {"en": "All {n} are right. In both tasks, the agent chooses actions, the environment is everything it acts on, and the rewards are the numbers that say how well it went.",
                        "nl": "Alle {n} kloppen. In beide taken kiest de agent actions, is de environment alles waarop hij werkt, en zijn de rewards de getallen die zeggen hoe goed het ging."},
        },
    ),
    "discounted-return": FunctionBox(
        name="discounted-return",
        function="discounted_return",
        start=_D_START,
        solution=_D,
        visible=[([0, 0, 0, 1], 0.9)],
        hidden=[([1, 2, 3], 0.5), ([5], 0.9), ([], 0.9), ([1, 1, 1, 1, 1], 1), ([-1, 0, 10], 0), ([2, -4, 8, 6], 0.25)],
        show=_short,
        first=[_DOTS],
        mistakes=[
            Mistake("no-discount", wrong=_d("        G = G + r\n        factor = factor * gamma\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every reward counts in full, as if `gamma` were 1. Multiply every reward by `factor`, and make `factor` itself `gamma` times as large after each reward; it is 1 only for the first one.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elke reward telt volledig mee, alsof `gamma` 1 is. Vermenigvuldig elke reward met `factor`, en maak `factor` zelf na elke reward `gamma` keer zo groot; alleen voor de eerste is hij 1."),
            Mistake("one-early", wrong=_d("        factor = factor * gamma\n        G = G + factor * r\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: even the first reward is discounted once. The first reward counts in full, with a factor of 1: add it with that factor, and only then make the factor `gamma` times as large.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: zelfs de eerste reward wordt één keer verdisconteerd. De eerste reward telt volledig, met een factor van 1: tel hem op met die factor, en maak de factor pas daarna `gamma` keer zo groot."),
            Mistake("gamma-once", wrong=_d("        G = G + gamma * r\n        factor = factor * gamma\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every reward is multiplied by `gamma` once, however late it comes. Multiply by `factor` instead: it gets `gamma` times as large at every step, so a later reward counts less.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elke reward wordt één keer met `gamma` vermenigvuldigd, hoe laat hij ook komt. Vermenigvuldig in plaats daarvan met `factor`: die wordt bij elke stap `gamma` keer zo groot, dus een latere reward telt minder."),
            Mistake("factor-adds", wrong=_d("        G = G + factor * r\n        factor = factor + gamma\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your `factor` grows by `gamma` at every step, so later rewards count more. Multiply the factor by `gamma` instead.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je `factor` groeit bij elke stap met `gamma`, dus latere rewards tellen zwaarder. Vermenigvuldig de factor in plaats daarvan met `gamma`."),
            # the start says `G = ...`: right for the example, where only the last reward is not 0
            Mistake("overwrites", wrong=_d("        G = factor * r\n        factor = factor * gamma\n"), state="almost",
                    en="Right for the example, where only the last reward is not 0. But `{call}` gives `{got}`, and it should give `{want}`: each round puts this reward times its factor in `G` instead of adding it, so only the last reward is left. Add it to what `G` already has, as step 3 of the recipe says.",
                    nl="Goed voor het voorbeeld, waarin alleen de laatste reward niet 0 is. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn: elke ronde zet deze reward keer zijn factor in `G` in plaats van hem erbij op te tellen, dus alleen de laatste reward blijft over. Tel hem op bij wat `G` al heeft, zoals stap 3 van het recept zegt."),
            Mistake("return-in-loop", wrong=_d("        G = G + factor * r\n        factor = factor * gamma\n        return G\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function gives `G` back after the first reward already. The line `return G` belongs after the loop, with as many spaces in front as `for`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je functie geeft `G` al na de eerste reward terug. De regel `return G` hoort na de loop, met evenveel spaties ervoor als `for`."),
        ],
        digits=4,
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the two `...`: what the return gets from this reward, and the factor for the next one.",
                      "nl": "Vervang de twee `...`: wat de return van deze reward krijgt, en de factor voor de volgende."},
            "arguments": {"en": "The check calls `{call}`, with a list of rewards and a discount factor, and the `def` line of your function has another number of names between its parentheses. Keep `(rewards, gamma)`.",
                          "nl": "De check roept `{call}` aan, met een lijst rewards en een discount factor, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(rewards, gamma)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your two lines with step 3 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je twee regels met stap 3 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any list of rewards, also negative ones, and any `gamma`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke lijst rewards, ook negatieve, en elke `gamma`?"},
            "correct": {"en": "Your function is right in all {n} cases, also with negative rewards, an empty list and `gamma` = 1. It adds every reward with the factor of its step.",
                        "nl": "Je functie klopt in alle {n} gevallen, ook met negatieve rewards, een lege lijst en `gamma` = 1. Hij telt elke reward op met de factor van zijn stap."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("discounted-return", _D_START, "empty", None),
    ("discounted-return", _d("        G += factor * r\n        factor *= gamma\n"), "correct", None),
    ("discounted-return", _d("        G = G + r\n        factor = factor * gamma\n"), "wrong", "no-discount"),
    ("discounted-return", _d("        factor = factor * gamma\n        G = G + factor * r\n"), "wrong", "one-early"),
    ("discounted-return", _d("        G = G + gamma * r\n        factor = factor * gamma\n"), "wrong", "gamma-once"),
    ("discounted-return", _d("        G = G + factor * r\n        factor = factor + gamma\n"), "wrong", "factor-adds"),
    ("discounted-return", _d("        G = factor * r\n        factor = factor * gamma\n"), "almost", "overwrites"),
    ("discounted-return", _d("        G = G + factor * r\n        factor = factor * gamma\n        return G\n"), "wrong", "return-in-loop"),
    ("discounted-return", _d("        G = G + factor * r\n        factor = factor * 0.9\n"), "almost", None),
    ("discounted-return", "def discounted_return(rewards, gamma):\n    G = 0\n    for r in rewards[::-1]:\n        G = r + gamma * G\n    return G\n", "correct", None),
]
