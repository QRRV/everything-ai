# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-case-dutch-algorithms: how many times as often DUO picked students with a non-European
migration background for a home visit, sorting what the people affected could not see in four Dutch cases, and the
pictures of who was in the group at each step and of the learner's shares."""
import html
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort
from everything_ai import yours

# DUO, the students who had the grant for living away from home on 1 February 2014 (Algorithm Audit, "Vooringenomenheid
# voorkomen - Addendum", 2024, with data of Statistics Netherlands, which rounds counts to tens). The report gives
# shares; the counts are worked out by the course from them: 22.0% of the 248,650 students had a non-European migration
# background, 36.0% of the 42,800 that the profile labeled high risk, and 63.2% of the 2,810 picked for a home visit,
# rounded to whole students. "everyone else" is everyone without a non-European migration background; the report
# compares with students of Dutch origin, which gives 2.0 and 6.2 instead of 1.99 and 6.09.
STUDENTS = {"non-European": 54703, "everyone else": 193947}
HIGH_RISK = {"non-European": 15408, "everyone else": 27392}
VISITED = {"non-European": 1776, "everyone else": 1034}
# The share with a non-European migration background at each step, as the report gives it.
STEPS = [("students", 248650, 0.220), ("high risk", 42800, 0.360), ("visited", 2810, 0.632)]


def shares(picked: dict, students: dict) -> dict:
    """For each group, the share of its students who were picked."""
    return {g: picked[g] / students[g] for g in students}


def times(picked: dict, students: dict) -> float:
    """How many times as often a non-European student was picked as everyone else."""
    s = shares(picked, students)
    return s["non-European"] / s["everyone else"]


_UI = "font-family:var(--ea-font-ui)"


def _number(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def picture(labels: dict) -> str:
    """Who was in the group at each step of DUO's checks: for all the students, those labeled high risk and those
    picked for a home visit, a bar with the share that had a non-European migration background, and a dashed mark at
    the share among all the students. `labels` gives "title", "rows" (three texts), "axis" (the words under the axis),
    "decimal" (the decimal sign) and "alt"."""
    from everything_ai import chart
    panel = chart.Panel(x=(0, 100), y=(0, 3.2), xlabel=labels["axis"], yticks=[], xticks=[0, 20, 40, 60, 80, 100])
    roles = ["data", "pred", "data"]
    for i, ((_, _, share), words) in enumerate(zip(STEPS, labels["rows"])):
        y = 2.55 - i
        v = share * 100
        panel.rect(0, y - 0.42, v, y - 0.02, role=roles[i], solid=i == 1)
        panel.text(1, y + 0.2, words, "ink", anchor="start", halo=True)
        panel.text(v + 1.5, y - 0.2, f"{v:.1f}%".replace(".", labels["decimal"]), "ink", anchor="start")
        panel.line([STEPS[0][2] * 100] * 2, [y - 0.5, y + 0.06], role="ink", width=1.5, dashed=True)
    return yours.figure(labels["title"], chart.figure([panel], alt=labels["alt"], width=460, height=260))


def _visited_picture(seen) -> str:
    """The learner's share picked per group, per 100 students, as a bar, with a dashed line at the right share, and
    above it their `times`, rounded to one decimal."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None:
        return ""
    picked = mine.names.get("picked")
    if not isinstance(picked, dict):
        return ""
    want = shares(VISITED, STUDENTS)
    groups = list(STUDENTS)
    top = 4
    panel = chart.Panel(x=(-0.6, len(groups) - 0.4), y=(0, top),
                        ylabel=seen.word({"en": "picked per 100 students", "nl": "gekozen per 100 studenten"}),
                        yticks=[0, 1, 2, 3, 4], xticks=list(range(len(groups))), xtick_labels=groups)
    for i, g in enumerate(groups):
        v = _number(picked.get(g))
        if v is not None and v > 0:
            panel.rect(i - 0.3, 0, i + 0.3, min(v * 100, top), role="data", solid=True)
        panel.line([i - 0.38, i + 0.38], [want[g] * 100] * 2, role="ink", width=2, dashed=True)
    title = seen.word({"en": "Your share picked per group", "nl": "Jouw deel gekozen per groep"})
    words = {"en": ("your code", "right"), "nl": ("jouw code", "goed")}[seen.lang]
    ratio = _number(mine.value) if mine.has_value else None
    said = ""
    if ratio is not None:
        said = (f'<div style="{_UI};font-size:0.9rem;color:var(--ea-ink);margin-bottom:0.2rem">'
                f'<code>times = {html.escape(f"{ratio:.1f}")}</code></div>')
    bar = ('<svg width="14" height="12" aria-hidden="true"><rect x="1" y="1" width="12" height="10" style="fill:var(--ea-data);'
           'stroke:var(--ea-data);stroke-width:1.2"/></svg>')
    dash = ('<svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-ink);'
            'stroke-width:2;stroke-dasharray:5 5"/></svg>')
    key = (f'<div style="{_UI};font-size:0.82rem;color:var(--ea-ink-2);display:flex;gap:1.2rem;flex-wrap:wrap;margin-top:0.3rem">'
           f'<span>{bar} {words[0]}</span><span>{dash} {words[1]}</span></div>')
    return yours.figure(title, said + chart.figure([panel], alt=title, width=420, height=250) + key)


# The lesson is done when these are correct. "register" is the choice question in the lesson.
REQUIRED = ["visited", "unseen", "register"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _line(name: str, values) -> str:
    return f"{name} = {values!r}".replace("'", '"')


_STUDENTS_LINE = _line("students", STUDENTS)
_HEAD = ("# DUO: students with the grant for living away from home on 1 February 2014, worked out from the report\n"
         + _STUDENTS_LINE + "\n# of those, picked for a home visit\n" + _line("visited", VISITED) + "\n\n"
         "picked = {}\nfor group in students:\n")
_START = _HEAD + "    ...\ntimes = ...\ntimes\n"
_LINE = "    picked[group] = visited[group] / students[group]\n"
_TIMES = 'times = picked["non-European"] / picked["everyone else"]\n'
_SOLUTION = _HEAD + _LINE + _TIMES + "times"


def _f(line: str, ratio: str = _TIMES) -> str:
    return _HEAD + line + ratio + "times"


EXERCISES = {
    "visited": CodeBox(
        name="visited",
        picture=_visited_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_STUDENTS_LINE],
        # The check also runs the box with home visits in proportion to the size of each group.
        others=[{"visited": Code('{"non-European": 618, "everyone else": 2192}')}],
        first=[_DOTS],
        digits=3,
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_high_risk`. In the box the names have none: `students`, `visited`, `picked`, `group`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_high_risk`. In het vak hebben de namen dat niet: `students`, `visited`, `picked`, `group`."),
            Mistake("no-ratio", matches=lambda got, text: isinstance(got, dict),
                    en="Your last line gives `{got}`: the share of each group. Now divide the non-European share by that of everyone else, and put the result in `times`.",
                    nl="Je laatste regel geeft `{got}`: het deel van elke groep. Deel nu het niet-Europese deel door dat van alle anderen, en zet de uitkomst in `times`."),
            Mistake("dicts-divided", error=lambda e: isinstance(e, TypeError) and "dict" in str(e),
                    en="Your code divides whole dictionaries. Use the counts of this one group: `visited[group]` and `students[group]`.",
                    nl="Je code deelt hele dictionaries. Gebruik de aantallen van deze ene groep: `visited[group]` en `students[group]`."),
            Mistake("upside-down", wrong=_f("    picked[group] = visited[group] / students[group]\n",
                                            'times = picked["everyone else"] / picked["non-European"]\n'),
                    en="Your last line gives `{got}`, less than 1: something is upside down. A share is the students picked divided by all the students of the group, and `times` puts the non-European share on top.",
                    nl="Je laatste regel geeft `{got}`, minder dan 1: er staat iets op zijn kop. Een deel is de gekozen studenten gedeeld door alle studenten van de groep, en `times` zet het niet-Europese deel bovenaan."),
            Mistake("counts", wrong=_f("    picked[group] = visited[group]\n"),
                    en="Your last line gives `{got}`: the counts compared. The other group is far larger, so first divide each count by the students of its group.",
                    nl="Je laatste regel geeft `{got}`: de aantallen vergeleken. De andere groep is veel groter, dus deel elk aantal eerst door de studenten van die groep."),
            Mistake("difference", wrong=_f(_LINE, 'times = picked["non-European"] - picked["everyone else"]\n'),
                    en="Your last line gives `{got}`: the difference between the two shares. *How many times as often* asks you to divide one share by the other.",
                    nl="Je laatste regel geeft `{got}`: het verschil tussen de twee delen. *Hoeveel keer zo vaak* vraagt je het ene deel door het andere te delen."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the first `...` with one line that puts under `picked[group]` the share of this group's students who got a home visit, and the second with how many times as large the non-European share is.",
                      "nl": "Vervang de eerste `...` door één regel die onder `picked[group]` het deel zet van de studenten uit deze groep die een huisbezoek kregen, en de tweede door hoeveel keer zo groot het niet-Europese deel is."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other numbers by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere aantallen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `times` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `times` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does your loop put under each group its visited students divided by all its students, and does `times` divide the non-European share by the other?",
                      "nl": "Je laatste regel geeft `{got}`. Zet je loop onder elke groep de bezochte studenten gedeeld door alle studenten van die groep, en deelt `times` het niet-Europese deel door het andere?"},
            "other": {"en": "Right for these numbers. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `visited` and `students` themselves?",
                      "nl": "Goed voor deze aantallen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `visited` en `students` zelf?"},
            "other_error": {"en": "Right for these numbers. But with {where}, your code stops with an error. Do you use `visited` and `students` themselves?",
                            "nl": "Goed voor deze aantallen. Maar met {where} stopt je code met een fout. Gebruik je `visited` en `students` zelf?"},
            "correct": {"en": "`{got}`: a student with a non-European migration background was picked for a home visit about 6 times as often as everyone else, while the profile labeled them high risk about 2 times as often. From the score to the visit, the gap grew about three times as large. The check also tried other numbers.",
                        "nl": "`{got}`: een student met een niet-Europese migratieachtergrond kreeg ongeveer 6 keer zo vaak een huisbezoek als alle anderen, terwijl het profiel hen ongeveer 2 keer zo vaak een hoog risico gaf. Van de score tot het bezoek werd het verschil ongeveer drie keer zo groot. De check probeerde ook andere aantallen."},
        },
    ),
    "unseen": Sort(
        name="unseen",
        groups={
            "counted": {"en": "What counted in the score", "nl": "Wat er meetelde in de score"},
            "decided": {"en": "Whether a person really decided", "nl": "Of er echt een mens besliste"},
            "worked": {"en": "Whether the system worked", "nl": "Of het systeem werkte"},
        },
        items={
            "distance": {"en": "**The distance.** At DUO, a short distance to the parents' address raised a student's score.",
                         "nl": "**De afstand.** Bij DUO maakte een korte afstand tot het adres van de ouders de score van een student hoger."},
            "language": {"en": "**The language.** In Rotterdam, about twenty features were about how well someone spoke or wrote Dutch.",
                         "nl": "**De taal.** In Rotterdam gingen ongeveer twintig features over hoe goed iemand Nederlands sprak of schreef."},
            "sign-off": {"en": "**The sign-off.** For three Uber drivers, the staff in between did no more than a symbolic act.",
                         "nl": "**Het akkoord.** Bij drie chauffeurs van Uber deden de medewerkers ertussen niet meer dan een louter symbolische handeling."},
            "deductions": {"en": "**The deductions.** At Ola, an automated process marked rides invalid and imposed deductions or fines, with no person in between.",
                           "nl": "**De kortingen.** Bij Ola keurde een automatisch proces ritten af en legde het kortingen of boetes op, zonder een mens ertussen."},
            "evaluation": {"en": "**The evaluation.** DUO never evaluated how its algorithm worked.",
                           "nl": "**De evaluatie.** DUO heeft nooit geëvalueerd hoe het algoritme werkte."},
            "effects": {"en": "**The effects.** An evaluation in 2017 found no signs that CAS had the effects the police hoped for.",
                        "nl": "**De effecten.** Een evaluatie in 2017 vond geen aanwijzingen dat CAS de effecten had waarop de politie hoopte."},
        },
        solution={"distance": "counted", "language": "counted", "sign-off": "decided", "deductions": "decided",
                  "evaluation": "worked", "effects": "worked"},
        nudge={
            "distance": {"en": "Is this about the features that raised a score, or about what happened after it?", "nl": "Gaat dit over de features die een score hoger maakten, of over wat er daarna gebeurde?"},
            "language": {"en": "What went into the model here?", "nl": "Wat ging er hier het model in?"},
            "sign-off": {"en": "Did a person really look at the case before the driver lost the work?", "nl": "Keek er echt een mens naar de zaak voordat de chauffeur het werk kwijtraakte?"},
            "deductions": {"en": "Who decided about the deductions?", "nl": "Wie besliste er over de kortingen?"},
            "evaluation": {"en": "Without an evaluation, what does nobody know?", "nl": "Wat weet niemand zonder evaluatie?"},
            "effects": {"en": "Is this about what went into CAS, or about what came out of using it?", "nl": "Gaat dit over wat er in CAS ging, of over wat het gebruik ervan opleverde?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the facts again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de feiten opnieuw uit."},
            "empty": {"en": "Put each fact under what it is about.",
                      "nl": "Zet elk feit bij waar het over gaat."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The fact *{item}* is about something else.",
                      "nl": "Het feit *{item}* gaat over iets anders."},
            "correct": {"en": "All {n} are right. An entry in the Algorithm Register asks about the same three things: which data an algorithm uses, how people oversee it, and its risks and impact assessments.",
                        "nl": "Alle {n} kloppen. Een beschrijving in het Algoritmeregister vraagt naar dezelfde drie dingen: welke data een algoritme gebruikt, hoe mensen er toezicht op houden, en de risico's en de beoordelingen daarvan."},
        },
    ),
}

_SORTED = {"distance": "counted", "language": "counted", "sign-off": "decided", "deductions": "decided",
           "evaluation": "worked", "effects": "worked"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("visited", _START, "empty", None),
    ("visited", _SOLUTION, "correct", None),
    ("visited", _f("    picked[group] = 1 / (students[group] / visited[group])\n"), "correct", None),
    ("visited", _HEAD + _LINE + "times = picked\ntimes", "wrong", "no-ratio"),
    ("visited", _f("    picked[group] = visited / students\n"), "error", "dicts-divided"),
    ("visited", _f("    picked[group] = _high_risk[group] / _students[group]\n"), "error", "underscore"),
    ("visited", _f(_LINE, 'times = picked["everyone else"] / picked["non-European"]\n'), "wrong", "upside-down"),
    ("visited", _f("    picked[group] = students[group] / visited[group]\n"), "wrong", "upside-down"),
    ("visited", _f("    picked[group] = visited[group]\n"), "wrong", "counts"),
    ("visited", _f(_LINE, 'times = picked["non-European"] - picked["everyone else"]\n'), "wrong", "difference"),
    ("visited", _f(_LINE, "times = 3\n"), "wrong", None),
    ("visited", _SOLUTION.replace('"everyone else": 193947', '"everyone else": 193000'), "wrong", "changed-line"),
    ("visited", _f(_LINE, f"times = {times(VISITED, STUDENTS)!r}\n"), "almost", "other-value"),
    ("visited", _HEAD + _LINE + "times = ...\ntimes", "wrong", "dots-left"),
    ("unseen", {}, "empty", None),
    ("unseen", _SORTED, "correct", None),
    ("unseen", {"distance": "counted", "effects": "worked"}, "wrong", "unfinished"),
    ("unseen", {**_SORTED, "distance": "worked"}, "wrong", "wrong:distance"),
    ("unseen", {**_SORTED, "sign-off": "counted"}, "wrong", "wrong:sign-off"),
    ("unseen", {**_SORTED, "effects": "decided"}, "wrong", "wrong:effects"),
]
