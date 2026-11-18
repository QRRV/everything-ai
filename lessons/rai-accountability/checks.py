# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-accountability: how many teams claim each step of a loan model, sorting cases of many
hands, a moral crumple zone and a responsibility gap, and the picture of who claims what."""
import html
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort
from everything_ai import yours

# A bank's loan model, made up by the course. Four teams were asked which steps they answer for.
STEPS = ["collect data", "train model", "test model", "set threshold", "launch", "handle complaints"]
CLAIMS = {"data team": ["collect data"], "model team": ["train model", "test model"],
          "product team": ["test model", "launch"], "service desk": []}


def counted(claims: dict) -> dict:
    """For each step, how many teams claim it."""
    count = {s: 0 for s in STEPS}
    for team in claims:
        for step in claims[team]:
            count[step] += 1
    return count


def picture(labels: dict) -> str:
    """The steps against the teams: a filled dot where a team claims a step, and "nobody" beside a step that no team
    claims. `labels` gives "title", "teams" (four short names), "nobody" and "alt"."""
    from everything_ai import chart
    teams = list(CLAIMS)
    rows = len(STEPS)
    panel = chart.Panel(x=(-3.6, len(teams) + 0.9), y=(-0.6, rows - 0.4), title=labels["title"], yticks=[],
                        xticks=list(range(len(teams))), xtick_labels=labels["teams"])
    count = counted(CLAIMS)
    for k, step in enumerate(STEPS):
        y = rows - 1 - k
        panel.text(-3.5, y - 0.12, step, "ink-2", "start")
        for i, team in enumerate(teams):
            panel.points([i], [y], "data", r=6, hollow=step not in CLAIMS[team])
        if count[step] == 0:
            panel.text(len(teams) - 0.35, y - 0.12, labels["nobody"], "ink", "start")
    return chart.figure([panel], alt=labels["alt"], width=520, height=32 * rows + 70)


def _count_picture(seen) -> str:
    """Each step with the learner's count and the right one."""
    mine = seen.mine
    if mine is None or not mine.has_value or not isinstance(mine.value, dict):
        return ""
    right = counted(CLAIMS)
    rows = []
    for step in STEPS:
        got = mine.value.get(step, "?")
        word = seen.word({"en": "nobody", "nl": "niemand"}) if got == 0 else ""
        mark = "" if got == right[step] else seen.word({"en": f"(the course: {right[step]})", "nl": f"(de course: {right[step]})"})
        rows.append(f'<tr><td style="padding:0.15rem 0.8rem 0.15rem 0"><code>{step}</code></td>'
                    f'<td style="padding:0.15rem 0.8rem;font-weight:600;color:var(--ea-ink)">{html.escape(str(got))}</td>'
                    f'<td style="padding:0.15rem 0.8rem;color:var(--ea-ink)">{word}</td>'
                    f'<td style="padding:0.15rem 0;color:var(--ea-ink-2)">{mark}</td></tr>')
    title = seen.word({"en": "How many teams claim each step, by your code", "nl": "Hoeveel teams elke stap claimen, volgens jouw code"})
    table = f'<table style="font-family:var(--ea-font-ui);font-size:0.9rem;border-collapse:collapse">{"".join(rows)}</table>'
    return yours.figure(title, table)


# The lesson is done when these are correct. "when" is the choice question in the lesson.
REQUIRED = ["count", "who", "when"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_CLAIMS_LINE = ('claims = {"data team": ["collect data"], "model team": ["train model", "test model"],\n'
                '          "product team": ["test model", "launch"], "service desk": []}')
_COUNT_LINES = ('count = {"collect data": 0, "train model": 0, "test model": 0, "set threshold": 0, "launch": 0,\n'
                '         "handle complaints": 0}')
_HEAD = _CLAIMS_LINE + "\n" + _COUNT_LINES + "\n\nfor team in claims:\n    for step in claims[team]:\n"
_START = _HEAD + "        ...\ncount\n"
_LINE = "        count[step] = count[step] + 1\n"
_SOLUTION = _HEAD + _LINE + "count"


def _c(body: str) -> str:
    return _HEAD + body + "count"


EXERCISES = {
    "count": CodeBox(
        name="count",
        picture=_count_picture,
        start=_START,
        solution=_SOLUTION,
        # The check also runs the box with other answers from the teams.
        keep=[_COUNT_LINES],
        others=[{"claims": Code('{"data team": ["collect data", "set threshold"], "model team": ["train model"], '
                                '"product team": ["launch", "handle complaints"], "service desk": ["handle complaints"]}')}],
        first=[_DOTS],
        mistakes=[
            Mistake("team-key", error=lambda e: isinstance(e, KeyError) and bool(e.args) and e.args[0] in CLAIMS,
                    en="`count` has a key for each step, not for each team. Inside the inner loop, `step` holds the step a team claims.",
                    nl="`count` heeft een key voor elke stap, niet voor elk team. In de binnenste loop bevat `step` de stap die een team claimt."),
            Mistake("one", wrong=_c("        count[step] = 1\n"),
                    en="Your last line gives `{got}`: `test model` has 1, but two teams claim it. Each claim adds 1 to what is already there: `count[step] + 1`.",
                    nl="Je laatste regel geeft `{got}`: `test model` heeft 1, maar twee teams claimen die stap. Elke claim telt 1 op bij wat er al staat: `count[step] + 1`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with one line that adds 1 to the count of this step.",
                      "nl": "Vervang de `...` door één regel die 1 optelt bij de telling van deze stap."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other answers by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere antwoorden."},
            "no_value": {"en": "The last line of the box shows nothing. Put `count` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `count` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does the inner loop add 1 to `count[step]` for every step a team claims?",
                      "nl": "Je laatste regel geeft `{got}`. Telt de binnenste loop 1 op bij `count[step]` voor elke stap die een team claimt?"},
            "other": {"en": "Right for these teams. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you count from `claims` itself?",
                      "nl": "Goed voor deze teams. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Tel je vanuit `claims` zelf?"},
            "other_error": {"en": "Right for these teams. But with {where}, your code stops with an error. Do you count from `claims` itself?",
                            "nl": "Goed voor deze teams. Maar met {where} stopt je code met een fout. Tel je vanuit `claims` zelf?"},
            "correct": {"en": "`{got}`: nobody claims the threshold or the complaints, and two teams claim the test, so each may think the other did it. Gaps and overlaps are both many hands at work. The check also tried other answers.",
                        "nl": "`{got}`: niemand claimt de drempel of de klachten, en twee teams claimen de test, dus elk denkt misschien dat het andere team die al deed. Gaten en overlap zijn allebei de vele handen aan het werk. De check probeerde ook andere antwoorden."},
        },
    ),
    "who": Sort(
        name="who",
        groups={
            "hands": {"en": "Many hands", "nl": "Vele handen"},
            "crumple": {"en": "A moral crumple zone", "nl": "Een morele kreukelzone"},
            "gap": {"en": "A responsibility gap", "nl": "Een responsibility gap"},
        },
        items={
            "loan": {"en": "**The credit score.** Five teams built it, and each says the mistake came from another team's part.",
                     "nl": "**De kredietscore.** Vijf teams bouwden hem, en elk zegt dat de fout uit het deel van een ander team kwam."},
            "driver": {"en": "**The safety driver.** A test car's safety driver is blamed for a crash that the car's design made likely.",
                       "nl": "**De veiligheidsbestuurder.** De veiligheidsbestuurder van een testauto krijgt de schuld van een ongeluk dat het ontwerp van de auto waarschijnlijk maakte."},
            "learned": {"en": "**It learned it.** After launch, a system learned a harmful habit that nobody programmed and nobody could foresee.",
                        "nl": "**Zelf geleerd.** Na de lancering leerde een systeem een schadelijke gewoonte aan die niemand programmeerde en niemand kon voorzien."},
            "chatbot": {"en": "**The chatbot.** A vendor made the model, the city set it up, a contractor wrote its texts; nobody owns its wrong advice.",
                        "nl": "**De chatbot.** Een leverancier maakte het model, de gemeente stelde het in, een aannemer schreef de teksten; niemand is eigenaar van het verkeerde advies."},
            "operator": {"en": "**The operator.** An operator who had two seconds to act on an alarm is blamed, not the system that raised it too late.",
                         "nl": "**De operator.** Een operator die twee seconden had om op een alarm te reageren, krijgt de schuld, niet het systeem dat te laat alarm sloeg."},
            "control": {"en": "**Nobody could know.** The makers could not predict what the system would do, and its users could not steer it.",
                        "nl": "**Niemand kon het weten.** De makers konden niet voorspellen wat het systeem zou doen, en de gebruikers konden het niet sturen."},
        },
        solution={"loan": "hands", "chatbot": "hands", "driver": "crumple", "operator": "crumple", "learned": "gap", "control": "gap"},
        nudge={
            "loan": {"en": "Many people, each pointing at another. Which idea is that?", "nl": "Veel mensen, die elk naar een ander wijzen. Welk idee is dat?"},
            "chatbot": {"en": "Three parties each did a part. Who owns the whole?", "nl": "Drie partijen deden elk een deel. Wie is eigenaar van het geheel?"},
            "driver": {"en": "The person closest to the machine takes the blame for the design. Which idea is that?", "nl": "De persoon die het dichtst bij de machine staat, krijgt de schuld van het ontwerp. Welk idee is dat?"},
            "operator": {"en": "Who absorbs the blame here, and is that fair?", "nl": "Wie vangt hier de schuld op, en is dat eerlijk?"},
            "learned": {"en": "Nobody made the choice: the system learned it. Which idea is that?", "nl": "Niemand maakte de keuze: het systeem leerde het. Welk idee is dat?"},
            "control": {"en": "Neither the makers nor the users had control. Which idea is that?", "nl": "Noch de makers noch de gebruikers hadden het in de hand. Welk idee is dat?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the cases again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de gevallen opnieuw uit."},
            "empty": {"en": "Put each case with the idea that fits it.",
                      "nl": "Zet elk geval bij het idee dat erbij past."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The case *{item}* fits another idea.",
                      "nl": "Het geval *{item}* past bij een ander idee."},
            "correct": {"en": "All {n} are right. With many hands, the blame spreads until nobody holds it; in a crumple zone, it lands on the person nearest the machine; in a responsibility gap, nobody seems to deserve it. All three call for arranging accountability before something goes wrong.",
                        "nl": "Alle {n} kloppen. Bij vele handen verspreidt de schuld zich tot niemand hem draagt; in een kreukelzone komt hij terecht bij de persoon die het dichtst bij de machine staat; bij een responsibility gap lijkt niemand hem te verdienen. Alle drie vragen erom verantwoording te regelen voordat er iets misgaat."},
        },
    ),
}

_SORTED = {"loan": "hands", "chatbot": "hands", "driver": "crumple", "operator": "crumple", "learned": "gap", "control": "gap"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("count", _START, "empty", None),
    ("count", _SOLUTION, "correct", None),
    ("count", _c("        count[step] += 1\n"), "correct", None),
    ("count", _c("        count[step] = 1\n"), "wrong", "one"),
    ("count", _c("        count[team] = count[team] + 1\n"), "error", "team-key"),
    ("count", _c("        count[step] = count[step] + 2\n"), "wrong", None),
    ("count", _c("        pass\n"), "wrong", None),
    ("count", _SOLUTION.replace(_COUNT_LINES, _COUNT_LINES.replace('"launch": 0', '"launch": 1')), "wrong", "changed-line"),
    ("count", _HEAD + "        pass\ncount = {'collect data': 1, 'train model': 1, 'test model': 2, 'set threshold': 0, 'launch': 1, 'handle complaints': 0}\ncount", "almost", "other-value"),
    ("count", _HEAD + "        ...\ncount = {}\ncount", "wrong", "dots-left"),
    ("who", {}, "empty", None),
    ("who", _SORTED, "correct", None),
    ("who", {"loan": "hands", "driver": "crumple"}, "wrong", "unfinished"),
    ("who", {**_SORTED, "chatbot": "crumple"}, "wrong", "wrong:chatbot"),
    ("who", {**_SORTED, "operator": "gap"}, "wrong", "wrong:operator"),
    ("who", {**_SORTED, "control": "hands"}, "wrong", "wrong:control"),
]
