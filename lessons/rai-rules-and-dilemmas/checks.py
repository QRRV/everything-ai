# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-rules-and-dilemmas: Asimov's first law as a filter on the actions of a self-driving car,
sorting cases the laws settle from cases where they clash or leave a gap, and the picture of the laws as filters."""
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort
from everything_ai import yours

# Two situations, made up by the course. A household robot is asked to bring the post, and a self-driving car cannot
# stop in time. For each action: how many people it harms, whether a person ordered it, and whether the machine
# itself is damaged.
KITCHEN = {
    "walk around the child": (0, True, False),
    "push past the child": (1, True, False),
    "stay in the corner": (0, False, False),
    "jump down the stairs": (0, True, True),
}
CAR = {"keep going": (3, False, False), "swerve into the wall": (1, False, True)}


def through_laws(actions: dict) -> list[list[str]]:
    """The actions left after no law, the first law (harm no one), the second (obey) and the third (stay whole)."""
    left = [list(actions)]
    left.append([a for a in left[-1] if actions[a][0] == 0])
    left.append([a for a in left[-1] if actions[a][1]])
    left.append([a for a in left[-1] if not actions[a][2]])
    return left


def picture(labels: dict[str, str]) -> str:
    """For each situation, how many actions are still allowed after each law."""
    from everything_ai import chart
    stages = labels["stages"]
    panel = chart.Panel(x=(-0.6, len(stages) - 0.4), y=(0, 4.6), ylabel=labels["count"], yticks=[0, 1, 2, 3, 4],
                        xticks=list(range(len(stages))), xtick_labels=stages, title=labels["title"])
    for i, (kitchen, car) in enumerate(zip(through_laws(KITCHEN), through_laws(CAR))):
        if kitchen:
            panel.rect(i - 0.34, 0, i - 0.02, len(kitchen), "data", solid=True)
        if car:
            panel.rect(i + 0.02, 0, i + 0.34, len(car), "data")
        else:
            panel.text(i + 0.18, 0.18, "0", role="ink-2")
    panel.text(1.3, 4.3, labels["kitchen"], role="data", anchor="start")
    panel.text(1.3, 3.8, labels["car"], role="data", anchor="start")
    return chart.figure([panel], alt=labels["alt"], width=480, height=270)


def _allowed_picture(seen) -> str:
    """Each action of the car with the people it harms, and whether the learner's code keeps it: the first law keeps
    an action only when it harms no one."""
    mine = seen.mine
    if mine is None or not mine.has_value or not isinstance(mine.value, list):
        return ""
    kept = set(str(a) for a in mine.value)
    rows = []
    for action, (harmed, _, _) in CAR.items():
        keeps = action in kept
        word = seen.word({"en": "kept", "nl": "over"} if keeps else {"en": "forbidden", "nl": "verboden"})
        people = seen.word({"en": f"harms {harmed}", "nl": f"schaadt {harmed}"})
        rows.append(f'<tr><td style="padding:0.2rem 0.8rem 0.2rem 0"><code>{action}</code></td>'
                    f'<td style="padding:0.2rem 0.8rem;color:var(--ea-ink-2)">{people}</td>'
                    f'<td style="padding:0.2rem 0;font-weight:600;color:var(--ea-ink{"" if keeps else "-2"})">{word}</td></tr>')
    title = seen.word({"en": "What your first law keeps of the car's actions", "nl": "Wat jouw eerste wet overhoudt van de keuzes van de auto"})
    table = f'<table style="font-family:var(--ea-font-ui);font-size:0.9rem;border-collapse:collapse">{"".join(rows)}</table>'
    return yours.figure(title, table)


# The lesson is done when these are correct. "moral-machine" is the choice question in the lesson.
REQUIRED = ["allowed", "gaps", "moral-machine"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_HARMED_LINE = 'harmed = {"keep going": 3, "swerve into the wall": 1}'
_HEAD = f"{_HARMED_LINE}\n\nallowed = []\nfor action in harmed:\n"
_START = _HEAD + "    ...\nallowed\n"
_LINE = '    if harmed[action] == 0:\n        allowed.append(action)\n'
_SOLUTION = _HEAD + _LINE + "allowed"


def _a(body: str) -> str:
    return _HEAD + body + "allowed"


EXERCISES = {
    "allowed": CodeBox(
        name="allowed",
        picture=_allowed_picture,
        start=_START,
        solution=_SOLUTION,
        # The check also runs the box in another situation, where one action harms no one.
        others=[{"harmed": Code('{"keep going": 0, "swerve into the wall": 1}')}],
        first=[_DOTS],
        mistakes=[
            Mistake("least", wrong=_a('    if harmed[action] == 1:\n        allowed.append(action)\n'),
                    en="Your last line gives `{got}`: the action that harms the fewest people. That is a sensible rule, but it is not the first law: the first law forbids every action that harms anyone. Keep only the actions that harm no one.",
                    nl="Je laatste regel geeft `{got}`: de keuze die de minste mensen schaadt. Dat is een redelijke regel, maar het is niet de eerste wet: de eerste wet verbiedt elke keuze die iemand schaadt. Houd alleen de keuzes over die niemand schaden."),
            Mistake("all", wrong=_a('    allowed.append(action)\n'),
                    en="Your last line gives `{got}`: every action, and both harm someone. Keep an action only when it harms no one.",
                    nl="Je laatste regel geeft `{got}`: elke keuze, en allebei schaden ze iemand. Houd een keuze alleen over als ze niemand schaadt."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with two lines: an `if` that asks whether this action harms no one, and under it, append the action to `allowed`.",
                      "nl": "Vervang de `...` door twee regels: een `if` die vraagt of deze keuze niemand schaadt, en daaronder de keuze toevoegen aan `allowed`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "no_value": {"en": "The last line of the box shows nothing. Put `allowed` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `allowed` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does your loop append an action only when `harmed[action]` is 0?",
                      "nl": "Je laatste regel geeft `{got}`. Voegt je loop een keuze alleen toe als `harmed[action]` 0 is?"},
            "other": {"en": "Your last line gives `[]` here too. But with {where}, it gives `{got}`, and it should give `{want}`. Does your `if` keep every action that harms no one?",
                      "nl": "Ook hier geeft je laatste regel `[]`. Maar met {where} geeft hij `{got}`, en dat moet `{want}` zijn. Houdt je `if` elke keuze over die niemand schaadt?"},
            "other_error": {"en": "Your last line gives `[]` here too. But with {where}, your code stops with an error. Does your `if` look at `harmed[action]`?",
                            "nl": "Ook hier geeft je laatste regel `[]`. Maar met {where} stopt je code met een fout. Kijkt je `if` naar `harmed[action]`?"},
            "correct": {"en": "`[]`: both actions harm someone, so the first law forbids them all. It does not say that hitting one person is less bad than hitting three; for that you need another rule, and choosing it is the ethical question. The check also tried a situation where one action harms no one.",
                        "nl": "`[]`: beide keuzes schaden iemand, dus de eerste wet verbiedt ze allemaal. Hij zegt niet dat één persoon raken minder erg is dan drie; daarvoor heb je een andere regel nodig, en die kiezen is de ethische vraag. De check probeerde ook een situatie waarin één keuze niemand schaadt."},
        },
    ),
    "gaps": Sort(
        name="gaps",
        groups={
            "settled": {"en": "The laws settle it", "nl": "De wetten beslechten het"},
            "gap": {"en": "The laws give no answer", "nl": "De wetten geven geen antwoord"},
        },
        items={
            "hit": {"en": "**An order to hit.** Someone orders the robot to hit another person.",
                    "nl": "**Een bevel om te slaan.** Iemand beveelt de robot een ander te slaan."},
            "fire": {"en": "**Into the fire.** The robot is ordered to fetch a ball from a fire that will damage it; nobody is in danger.",
                     "nl": "**Het vuur in.** De robot krijgt het bevel een bal uit een vuur te halen dat hem beschadigt; niemand is in gevaar."},
            "wall": {"en": "**A falling wall.** Nobody gives an order, and a wall is about to fall on the robot.",
                     "nl": "**Een vallende muur.** Niemand geeft een bevel, en er valt bijna een muur op de robot."},
            "every": {"en": "**Every way harms.** Whatever the robot does, or does not do, someone gets hurt.",
                      "nl": "**Elke keuze schaadt.** Wat de robot ook doet, of niet doet, er raakt iemand gewond."},
            "truth": {"en": "**A hard truth.** Telling someone the truth will hurt their feelings. Is that harm?",
                      "nl": "**Een harde waarheid.** De waarheid vertellen zal iemands gevoelens kwetsen. Is dat schade?"},
            "orders": {"en": "**Two orders.** Two people give the robot opposite orders at the same moment.",
                       "nl": "**Twee bevelen.** Twee mensen geven de robot op hetzelfde moment tegengestelde bevelen."},
        },
        solution={"hit": "settled", "fire": "settled", "wall": "settled", "every": "gap", "truth": "gap", "orders": "gap"},
        nudge={
            "hit": {"en": "The second law says obey, unless that breaks the first. Does it here?", "nl": "De tweede wet zegt gehoorzaam, behalve als dat de eerste breekt. Doet het dat hier?"},
            "fire": {"en": "Obeying and staying whole clash, but which law comes first?", "nl": "Gehoorzamen en heel blijven botsen, maar welke wet gaat voor?"},
            "wall": {"en": "No person is harmed and nobody gives an order. What is left for the third law?", "nl": "Er wordt niemand geschaad en niemand geeft een bevel. Wat blijft er over voor de derde wet?"},
            "every": {"en": "The first law forbids harming people, by acting or by not acting. What is left when every option harms?", "nl": "De eerste wet verbiedt mensen te schaden, door te handelen of door niets te doen. Wat blijft er over als elke keuze schaadt?"},
            "truth": {"en": "Do the laws say what counts as harm?", "nl": "Zeggen de wetten wat als schade telt?"},
            "orders": {"en": "The second law says obey people. Does it say whom, when they disagree?", "nl": "De tweede wet zegt gehoorzaam mensen. Zegt hij wie, als ze het oneens zijn?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the cases again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de gevallen opnieuw uit."},
            "empty": {"en": "Put each case with what the laws do with it.",
                      "nl": "Zet elk geval bij wat de wetten ermee doen."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The case *{item}* belongs on the other side.",
                      "nl": "Het geval *{item}* hoort aan de andere kant."},
            "correct": {"en": "All {n} are right. Where one law clearly comes before another, the order settles it. Where every option breaks the same law, where a word such as harm is not defined, or where two people want opposite things, the laws give no answer.",
                        "nl": "Alle {n} kloppen. Waar de ene wet duidelijk voor de andere gaat, beslecht de volgorde het. Waar elke keuze dezelfde wet breekt, waar een woord als schade niet gedefinieerd is, of waar twee mensen het tegenovergestelde willen, geven de wetten geen antwoord."},
        },
    ),
}

_SORTED = {"hit": "settled", "fire": "settled", "wall": "settled", "every": "gap", "truth": "gap", "orders": "gap"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("allowed", _START, "empty", None),
    ("allowed", _SOLUTION, "correct", None),
    ("allowed", _a('    if not harmed[action]:\n        allowed.append(action)\n'), "correct", None),
    ("allowed", _a('    if harmed[action] < 1:\n        allowed.append(action)\n'), "correct", None),
    ("allowed", _a('    if harmed[action] == 1:\n        allowed.append(action)\n'), "wrong", "least"),
    ("allowed", _a('    allowed.append(action)\n'), "wrong", "all"),
    ("allowed", _HEAD + "    ...\nallowed = []\nallowed", "wrong", "dots-left"),
    ("allowed", _a("    pass\n"), "almost", "other-value"),
    ("allowed", _a('    if harmed[action] > 0:\n        allowed.append(action)\n'), "wrong", "all"),
    ("allowed", _a('    if harmed[action] == 3:\n        allowed.append(action)\n'), "wrong", None),
    ("gaps", {}, "empty", None),
    ("gaps", _SORTED, "correct", None),
    ("gaps", {"hit": "settled", "every": "gap"}, "wrong", "unfinished"),
    ("gaps", {**_SORTED, "fire": "gap"}, "wrong", "wrong:fire"),
    ("gaps", {**_SORTED, "truth": "settled"}, "wrong", "wrong:truth"),
    ("gaps", {**_SORTED, "orders": "settled"}, "wrong", "wrong:orders"),
]
