# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-dark-patterns: which subscriptions take more steps to leave than to join, sorting designs
into nudges, dark patterns and hypernudges, and the picture of the steps."""
import html
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort
from everything_ai import yours

# Five subscriptions, made up by the course: how many steps it takes to join, and how many to leave again.
JOIN = {"music": 2, "news": 3, "fitness": 2, "cloud": 4, "games": 1}
LEAVE = {"music": 2, "news": 9, "fitness": 7, "cloud": 3, "games": 6}


def harder(join: dict, leave: dict) -> list:
    """The subscriptions that take more steps to leave than to join, in the order of `join`."""
    return [s for s in join if leave[s] > join[s]]


def picture(labels: dict) -> str:
    """For each subscription, the steps to join (an open dot) and to leave (a filled dot) with a line between, and
    `labels["harder"]` beside the ones that are harder to leave. `labels` gives "title", "x", "harder" and "alt"."""
    from everything_ai import chart
    rows = len(JOIN)
    panel = chart.Panel(x=(-4.2, 17.0), y=(-0.6, rows - 0.4), title=labels["title"], xlabel=labels["x"], yticks=[],
                        xticks=[0, 2, 4, 6, 8, 10])
    worse = harder(JOIN, LEAVE)
    for k, name in enumerate(JOIN):
        y = rows - 1 - k
        panel.text(-4.1, y - 0.12, name, "ink-2", "start")
        panel.line([JOIN[name], LEAVE[name]], [y, y], "ink-2", width=1.4)
        panel.points([JOIN[name]], [y], "data", r=6, hollow=True)
        panel.points([LEAVE[name]], [y], "data", r=6)
        if name in worse:
            panel.text(10.2, y - 0.12, labels["harder"], "ink", "start")
    return chart.figure([panel], alt=labels["alt"], width=600, height=32 * rows + 80)


def _harder_picture(seen) -> str:
    """Each subscription, whether the learner's list holds it, and whether it should."""
    mine = seen.mine
    if mine is None or not mine.has_value or not isinstance(mine.value, list):
        return ""
    right = harder(JOIN, LEAVE)
    got = [str(x) for x in mine.value]
    rows = []
    for name in JOIN:
        listed = seen.word({"en": "in your list", "nl": "in je lijst"}) if name in got else ""
        should = name in right
        mark = "" if (name in got) == should else seen.word(
            {"en": "(the course: in it)" if should else "(the course: not in it)",
             "nl": "(de course: erin)" if should else "(de course: niet erin)"})
        rows.append(f'<tr><td style="padding:0.15rem 0.8rem 0.15rem 0"><code>{name}</code></td>'
                    f'<td style="padding:0.15rem 0.8rem;color:var(--ea-ink-2)">{JOIN[name]} &rarr; {LEAVE[name]}</td>'
                    f'<td style="padding:0.15rem 0.8rem;font-weight:600;color:var(--ea-ink)">{listed}</td>'
                    f'<td style="padding:0.15rem 0;color:var(--ea-ink-2)">{mark}</td></tr>')
    extra = [x for x in got if x not in JOIN]
    if extra:
        rows.append(f'<tr><td colspan="4" style="padding:0.15rem 0;color:var(--ea-ink-2)">'
                    + seen.word({"en": "also in your list: ", "nl": "ook in je lijst: "})
                    + html.escape(", ".join(extra)) + "</td></tr>")
    title = seen.word({"en": "Steps to join and to leave, and your list", "nl": "Stappen om lid te worden en op te zeggen, en jouw lijst"})
    table = f'<table style="font-family:var(--ea-font-ui);font-size:0.9rem;border-collapse:collapse">{"".join(rows)}</table>'
    return yours.figure(title, table)


# The lesson is done when these are correct. "manipulation" is the choice question in the lesson.
REQUIRED = ["harder", "kinds", "manipulation"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _line(name: str, values: dict) -> str:
    return f"{name} = {values!r}".replace("'", '"')


_JOIN_LINE = _line("join", JOIN)
_HEAD = ("# how many steps it takes\n" + _JOIN_LINE + "\n" + _line("leave", LEAVE)
         + "\n\nharder = []\nfor service in join:\n")
_START = _HEAD + "    ...\nharder\n"
_LINES = "    if leave[service] > join[service]:\n        harder.append(service)\n"
_SOLUTION = _HEAD + _LINES + "harder"


def _h(body: str) -> str:
    return _HEAD + body + "harder"


EXERCISES = {
    "harder": CodeBox(
        name="harder",
        picture=_harder_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_JOIN_LINE],
        # The check also runs the box with other steps to leave.
        others=[{"leave": Code('{"music": 5, "news": 3, "fitness": 2, "cloud": 4, "games": 1}')}],
        first=[_DOTS],
        mistakes=[
            Mistake("equal", wrong=_h(_LINES.replace(">", ">=")),
                    en="Your last line gives `{got}`: `music` takes 2 steps both ways, so it is not harder to leave. Use `>`, not `>=`.",
                    nl="Je laatste regel geeft `{got}`: `music` kost beide kanten op 2 stappen, dus opzeggen is niet moeilijker. Gebruik `>`, niet `>=`."),
            Mistake("flipped", wrong=_h(_LINES.replace(">", "<")),
                    en="Your last line gives `{got}`: those take fewer steps to leave than to join. Which of the two should be larger?",
                    nl="Je laatste regel geeft `{got}`: die kosten minder stappen om op te zeggen dan om lid te worden. Welke van de twee moet groter zijn?"),
            Mistake("numbers", wrong=_h(_LINES.replace("harder.append(service)", "harder.append(leave[service])")),
                    en="Your last line gives `{got}`: those are numbers of steps. Append the name of the subscription, `service`.",
                    nl="Je laatste regel geeft `{got}`: dat zijn aantallen stappen. Voeg de naam van het abonnement toe, `service`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with two lines: an `if` that compares the steps of this subscription, and under it an `append`.",
                      "nl": "Vervang de `...` door twee regels: een `if` die de stappen van dit abonnement vergelijkt, en daaronder een `append`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other steps by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere stappen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `harder` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `harder` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does your `if` append a subscription only when leaving takes more steps than joining?",
                      "nl": "Je laatste regel geeft `{got}`. Voegt je `if` een abonnement alleen toe als opzeggen meer stappen kost dan lid worden?"},
            "other": {"en": "Right for these subscriptions. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you compare `leave` and `join` themselves?",
                      "nl": "Goed voor deze abonnementen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Vergelijk je `leave` en `join` zelf?"},
            "other_error": {"en": "Right for these subscriptions. But with {where}, your code stops with an error. Do you compare `leave` and `join` themselves?",
                            "nl": "Goed voor deze abonnementen. Maar met {where} stopt je code met een fout. Vergelijk je `leave` en `join` zelf?"},
            "correct": {"en": "`{got}`: three of the five take more steps to leave than to join, `news` three times as many. Making it hard to leave is one of the best-known dark patterns. The check also tried other steps.",
                        "nl": "`{got}`: drie van de vijf kosten meer stappen om op te zeggen dan om lid te worden, `news` drie keer zo veel. Opzeggen moeilijk maken is een van de bekendste dark patterns. De check probeerde ook andere stappen."},
        },
    ),
    "kinds": Sort(
        name="kinds",
        groups={
            "nudge": {"en": "A nudge", "nl": "Een nudge"},
            "dark": {"en": "A dark pattern", "nl": "Een dark pattern"},
            "hyper": {"en": "A hypernudge", "nl": "Een hypernudge"},
        },
        items={
            "salad": {"en": "**The salad.** A school canteen puts fruit and salad first in line, and the sweets at the end.",
                      "nl": "**De salade.** Een schoolkantine zet fruit en salade vooraan, en de snoep achteraan."},
            "pension": {"en": "**The pension.** New staff save for a pension unless they say no, and one form is enough to stop.",
                        "nl": "**Het pensioen.** Nieuwe medewerkers sparen voor een pensioen tenzij ze nee zeggen, en één formulier is genoeg om te stoppen."},
            "ticked": {"en": "**The ticked box.** Travel insurance is added to a ticket, in a box that is ticked in advance.",
                       "nl": "**Het aangevinkte vakje.** Bij een ticket zit een reisverzekering, in een vakje dat al is aangevinkt."},
            "timer": {"en": "**The timer.** \"Offer ends in 10 minutes\", and when you reload the page, the clock starts again.",
                      "nl": "**De timer.** \"Aanbieding eindigt over 10 minuten\", en als je de pagina opnieuw laadt, begint de klok weer."},
            "feed": {"en": "**The feed.** An app changes what it shows you every few seconds, from what kept you scrolling before.",
                     "nl": "**De feed.** Een app verandert om de paar seconden wat hij je laat zien, op basis van waar je eerder bij bleef hangen."},
            "playlist": {"en": "**The playlist.** A music app picks the next song every few minutes, from what you skipped and played again, so nobody gets the same list.",
                         "nl": "**De playlist.** Een muziekapp kiest om de paar minuten het volgende nummer, op basis van wat je oversloeg en opnieuw afspeelde, dus niemand krijgt dezelfde lijst."},
        },
        solution={"salad": "nudge", "pension": "nudge", "ticked": "dark", "timer": "dark", "feed": "hyper", "playlist": "hyper"},
        nudge={
            "salad": {"en": "Is anything hidden or forbidden? And in whose interest is it?", "nl": "Wordt er iets verstopt of verboden? En in wiens belang is het?"},
            "pension": {"en": "Can people see the choice and undo it easily?", "nl": "Kunnen mensen de keuze zien en makkelijk terugdraaien?"},
            "ticked": {"en": "Did the person choose this, or was it chosen for them in the hope they would not notice?", "nl": "Koos de persoon dit, of koos iemand anders het, in de hoop dat het niet zou opvallen?"},
            "timer": {"en": "Is the hurry real?", "nl": "Is de haast echt?"},
            "feed": {"en": "Is it the same for everyone, or does it change with data about you?", "nl": "Is het voor iedereen hetzelfde, of verandert het met data over jou?"},
            "playlist": {"en": "Does it adapt to each person, from data about them?", "nl": "Past het zich aan elke persoon aan, met data over die persoon?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the designs again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de ontwerpen opnieuw uit."},
            "empty": {"en": "Put each design with the kind it belongs to.",
                      "nl": "Zet elk ontwerp bij de soort waar het bij hoort."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The design *{item}* belongs to another kind.",
                      "nl": "Het ontwerp *{item}* hoort bij een andere soort."},
            "correct": {"en": "All {n} are right. A nudge steers openly, and stopping is easy; a dark pattern works against the person, by hiding or rushing; a hypernudge adapts to each person, from data about them, all the time.",
                        "nl": "Alle {n} kloppen. Een nudge stuurt openlijk, en stoppen is makkelijk; een dark pattern werkt tegen de persoon in, door te verstoppen of op te jagen; een hypernudge past zich steeds aan elke persoon aan, met data over die persoon."},
        },
    ),
}

_SORTED = {"salad": "nudge", "pension": "nudge", "ticked": "dark", "timer": "dark", "feed": "hyper", "playlist": "hyper"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("harder", _START, "empty", None),
    ("harder", _SOLUTION, "correct", None),
    ("harder", _h("    if join[service] < leave[service]:\n        harder.append(service)\n"), "correct", None),
    ("harder", _h(_LINES.replace(">", ">=")), "wrong", "equal"),
    ("harder", _h(_LINES.replace(">", "<")), "wrong", "flipped"),
    ("harder", _h(_LINES.replace("harder.append(service)", "harder.append(leave[service])")), "wrong", "numbers"),
    ("harder", _h("    harder.append(service)\n"), "wrong", None),
    ("harder", _SOLUTION.replace(_JOIN_LINE, _JOIN_LINE.replace('"games": 1', '"games": 9')), "wrong", "changed-line"),
    ("harder", _HEAD + "    pass\nharder = ['news', 'fitness', 'games']\nharder", "almost", "other-value"),
    ("harder", _HEAD + "    ...\nharder = []\nharder", "wrong", "dots-left"),
    ("kinds", {}, "empty", None),
    ("kinds", _SORTED, "correct", None),
    ("kinds", {"salad": "nudge", "feed": "hyper"}, "wrong", "unfinished"),
    ("kinds", {**_SORTED, "pension": "dark"}, "wrong", "wrong:pension"),
    ("kinds", {**_SORTED, "timer": "hyper"}, "wrong", "wrong:timer"),
    ("kinds", {**_SORTED, "playlist": "dark"}, "wrong", "wrong:playlist"),
]
