# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-case-health-credit: which made-up patients a program picks by their costs instead of by
their illnesses, sorting goals from the proxies that stand in for them, and the pictures of the health algorithm and of
the learner's picks."""
import html
import re

from everything_ai.check import CodeBox, Mistake, Sort
from everything_ai import yours

# What Obermeyer and colleagues (Science, 2019) report, at the score above which patients were picked automatically
# (the top 3 percent): chronic illnesses of Black and White patients, and the share of Black patients among those
# picked, with the cost label and in their simulation of a score that means the same health in both groups.
ILLNESSES_AT_TOP = {"Black": 4.8, "White": 3.8}
BLACK_PICKED = {"cost": 17.7, "simulated": 46.5}

# Eight patients made up by the course: chronic illnesses, and the costs of their care in the next year, in dollars.
# A chronic illness costs 2,000 dollars of care for the Black patients (B) and 3,000 for the White patients (W).
ILLNESSES = {"B1": 5, "B2": 4, "B3": 2, "B4": 1, "W1": 4, "W2": 3, "W3": 2, "W4": 1}
COSTS = {p: n * (2000 if p[0] == "B" else 3000) for p, n in ILLNESSES.items()}
BY_NEED = 4        # the worked example picks patients with at least this many chronic illnesses
BY_COST = 9000     # the exercise picks patients whose costs are at least this many dollars
OTHER_COST = 8000  # the other threshold the check tries


def picked(values: dict, threshold: float) -> list:
    return [p for p in values if values[p] >= threshold]


_UI = "font-family:var(--ea-font-ui)"


def picture(labels: dict) -> str:
    """Two panels of what the study found: the chronic illnesses of Black and White patients at the score where
    patients were picked, and the share of Black patients among those picked, with the cost label (filled) and in the
    simulation (hollow). `labels` gives "title", "groups" (two texts), "illnesses", "share", "cost", "simulated",
    "decimal" and "alt"."""
    from everything_ai import chart
    d = labels["decimal"]
    left = chart.Panel(x=(-0.6, 1.6), y=(0, 6), ylabel=labels["illnesses"], yticks=[0, 2, 4, 6],
                       xticks=[0, 1], xtick_labels=labels["groups"])
    for i, g in enumerate(ILLNESSES_AT_TOP):
        v = ILLNESSES_AT_TOP[g]
        left.rect(i - 0.3, 0, i + 0.3, v, role="data", solid=True)
        left.text(i, v + 0.4, f"{v:.1f}".replace(".", d), "ink")
    right = chart.Panel(x=(-0.6, 1.6), y=(0, 60), ylabel=labels["share"], yticks=[0, 20, 40, 60],
                        xticks=[0, 1], xtick_labels=[labels["cost"], labels["simulated"]])
    for i, k in enumerate(BLACK_PICKED):
        v = BLACK_PICKED[k]
        right.rect(i - 0.3, 0, i + 0.3, v, role="data", solid=k == "cost")
        right.text(i, v + 4, f"{v:.1f}%".replace(".", d), "ink")
    return yours.figure(labels["title"], chart.figure([left, right], alt=labels["alt"], width=480, height=250))


def _number(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _picked_picture(seen) -> str:
    """Each made-up patient as a dot at their illnesses and costs, Black patients as circles and White patients as
    triangles; the patients in the learner's `picked` are named in bold, and a line marks their `threshold`."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None:
        return ""
    got = mine.value if mine.has_value and isinstance(mine.value, list) else []
    threshold = _number(mine.names.get("threshold"))
    panel = chart.Panel(x=(0, 6), y=(0, 14000), xlabel=seen.word({"en": "chronic illnesses", "nl": "chronische ziektes"}),
                        ylabel=seen.word({"en": "costs next year (dollars)", "nl": "kosten volgend jaar (dollar)"}),
                        xticks=[0, 1, 2, 3, 4, 5, 6], yticks=[0, 4000, 8000, 12000])
    if threshold is not None and 0 < threshold < 14000:
        panel.line([0, 6], [threshold] * 2, role="ink-2", width=1.5, dashed=True)
        panel.text(0.1, threshold + 450, seen.word({"en": "picked from here", "nl": "gekozen vanaf hier"}), "ink-2",
                   anchor="start", halo=True)
    for k, start in enumerate(("B", "W")):
        names = [p for p in ILLNESSES if p[0] == start]
        panel.group([ILLNESSES[p] for p in names], [COSTS[p] for p in names], k)
    for p in ILLNESSES:
        mark = "✓ " if p in got else ""
        panel.text(ILLNESSES[p] + 0.12, COSTS[p] + 350, mark + p, "ink" if p in got else "ink-3", anchor="start")
    title = seen.word({"en": "Who your list picks", "nl": "Wie jouw lijst kiest"})
    words = {"en": ("Black patients", "White patients", "✓ in your list"),
             "nl": ("zwarte patiënten", "witte patiënten", "✓ in jouw lijst")}[seen.lang]
    key = (f'<div style="{_UI};font-size:0.82rem;color:var(--ea-ink-2);display:flex;gap:1.2rem;flex-wrap:wrap;margin-top:0.3rem">'
           f'<span><span style="color:var(--ea-group-1)">●</span> {words[0]}</span>'
           f'<span><span style="color:var(--ea-group-2)">▲</span> {words[1]}</span><span>{words[2]}</span></div>')
    return yours.figure(title, chart.figure([panel], alt=title, width=440, height=270) + key)


# The lesson is done when these are correct. "regulator" is the choice question in the lesson.
REQUIRED = ["picked", "proxies", "regulator"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _line(name: str, values) -> str:
    return f"{name} = {values!r}".replace("'", '"')


_ILL_LINE = _line("illnesses", ILLNESSES)
_COST_LINE = _line("costs", COSTS)
_HEAD = ("# chronic illnesses and costs of care next year, in dollars, made up by the course\n"
         f"{_ILL_LINE}\n{_COST_LINE}\n\nthreshold = {BY_COST}\n")
_START = _HEAD + "...\npicked\n"
_BODY = "picked = []\nfor patient in costs:\n    if costs[patient] >= threshold:\n        picked.append(patient)\n"
_SOLUTION = _HEAD + _BODY + "picked"


def _s(body: str) -> str:
    return _HEAD + body + "picked"


def _swap(old: str, new: str) -> str:
    assert _BODY.count(old) == 1
    return _s(_BODY.replace(old, new))


EXERCISES = {
    "picked": CodeBox(
        name="picked",
        picture=_picked_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_ILL_LINE, _COST_LINE],
        # The check also runs the box with another threshold.
        others=[{"threshold": OTHER_COST}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_costs`. In the box the names have none: `illnesses`, `costs`, `threshold`, `picked`, `patient`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_costs`. In het vak hebben de namen dat niet: `illnesses`, `costs`, `threshold`, `picked`, `patient`."),
            Mistake("by-need", wrong=_swap("costs[patient] >= threshold", "illnesses[patient] >= 4"),
                    en="Your last line gives `{got}`: the patients with at least 4 chronic illnesses, as in the worked example. The program picks by costs: compare `costs[patient]` with `threshold`.",
                    nl="Je laatste regel geeft `{got}`: de patiënten met minstens 4 chronische ziektes, zoals in het uitgewerkte voorbeeld. Het programma kiest op kosten: vergelijk `costs[patient]` met `threshold`."),
            Mistake("greater", wrong=_swap(">= threshold", "> threshold"),
                    en="Your last line gives `{got}`: W2 is missing. W2's costs are exactly `threshold`, and *at least* includes the threshold itself: use `>=`.",
                    nl="Je laatste regel geeft `{got}`: W2 ontbreekt. De kosten van W2 zijn precies `threshold`, en *minstens* telt de drempel zelf mee: gebruik `>=`."),
            Mistake("below", wrong=_swap("costs[patient] >= threshold", "costs[patient] < threshold"),
                    en="Your last line gives `{got}`: the patients below the threshold. The program picks the patients whose costs are at least `threshold`.",
                    nl="Je laatste regel geeft `{got}`: de patiënten onder de drempel. Het programma kiest de patiënten van wie de kosten minstens `threshold` zijn."),
            Mistake("costs-listed", wrong=_swap("picked.append(patient)", "picked.append(costs[patient])"),
                    en="Your last line gives `{got}`: the costs of the patients who are picked. Put the patient in the list, not their costs.",
                    nl="Je laatste regel geeft `{got}`: de kosten van de patiënten die gekozen worden. Zet de patiënt in de lijst, niet de kosten."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the lines of the worked example: start with an empty list `picked`, and put in it every patient whose costs are at least `threshold`.",
                      "nl": "Vervang de `...` door de regels van het uitgewerkte voorbeeld: begin met een lege lijst `picked`, en zet daarin elke patiënt van wie de kosten minstens `threshold` zijn."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the lines of `illnesses` and `costs` exactly as they were: put them back. The check tries another threshold by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `illnesses` en `costs` nodig zoals ze waren: zet ze terug. De check probeert zelf een andere drempel."},
            "no_value": {"en": "The last line of the box shows nothing. Put `picked` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `picked` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does your loop put each patient whose `costs[patient]` is at least `threshold` in `picked`?",
                      "nl": "Je laatste regel geeft `{got}`. Zet je loop elke patiënt van wie `costs[patient]` minstens `threshold` is in `picked`?"},
            "other": {"en": "Right for this threshold. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `threshold` itself?",
                      "nl": "Goed voor deze drempel. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `threshold` zelf?"},
            "other_error": {"en": "Right for this threshold. But with {where}, your code stops with an error. Do you use `threshold` itself?",
                            "nl": "Goed voor deze drempel. Maar met {where} stopt je code met een fout. Gebruik je `threshold` zelf?"},
            "correct": {"en": "`{got}`: by costs, the program picks W2, with 3 chronic illnesses, and leaves out B2, with 4, because less is spent on B2's care. The worked example picked by illnesses: B1, B2 and W1. The check also tried a threshold of `8000`: then B2 is in, and so are B1, W1 and W2.",
                        "nl": "`{got}`: op kosten kiest het programma W2, met 3 chronische ziektes, en laat het B2 weg, met 4, omdat er minder aan de zorg voor B2 wordt uitgegeven. Het uitgewerkte voorbeeld koos op ziektes: B1, B2 en W1. De check probeerde ook een drempel van `8000`: dan zit B2 erin, en ook B1, W1 en W2."},
        },
    ),
    "proxies": Sort(
        name="proxies",
        groups={
            "goal": {"en": "What you really want to know", "nl": "Wat je eigenlijk wilt weten"},
            "proxy": {"en": "What gets measured instead", "nl": "Wat er in plaats daarvan gemeten wordt"},
        },
        items={
            "need": {"en": "**Need for care.** How ill a patient is, and how much extra care would help.",
                     "nl": "**Zorgbehoefte.** Hoe ziek een patiënt is, en hoeveel extra zorg zou helpen."},
            "costs": {"en": "**Costs.** How much is spent on a patient's care in the next year.",
                      "nl": "**Kosten.** Hoeveel er in het volgende jaar aan de zorg voor een patiënt wordt uitgegeven."},
            "crime": {"en": "**A new crime.** Whether a defendant commits another crime.",
                      "nl": "**Een nieuw misdrijf.** Of een verdachte opnieuw een misdrijf pleegt."},
            "arrest": {"en": "**Arrested again.** Whether a defendant is arrested again within two years.",
                       "nl": "**Opnieuw opgepakt.** Of een verdachte binnen twee jaar opnieuw wordt opgepakt."},
            "interest": {"en": "**Interest.** Whether a reader really cares about an article.",
                         "nl": "**Interesse.** Of een lezer een nieuwsbericht echt belangrijk vindt."},
            "clicks": {"en": "**Clicks.** Whether a reader clicks on an article.",
                       "nl": "**Klikken.** Of een lezer op een nieuwsbericht klikt."},
        },
        solution={"need": "goal", "costs": "proxy", "crime": "goal", "arrest": "proxy", "interest": "goal", "clicks": "proxy"},
        nudge={
            "need": {"en": "Can you count this directly in the records of a hospital?", "nl": "Kun je dit direct tellen in de gegevens van een ziekenhuis?"},
            "costs": {"en": "Is this what the program wants to know, or what is easy to find in the bills?", "nl": "Is dit wat het programma wil weten, of wat makkelijk in de rekeningen te vinden is?"},
            "crime": {"en": "Does anyone see every crime that is committed?", "nl": "Ziet iemand elk misdrijf dat gepleegd wordt?"},
            "arrest": {"en": "Does an arrest prove a crime? And who gets arrested depends on where the police look.", "nl": "Bewijst opgepakt worden een misdrijf? En wie er wordt opgepakt, hangt af van waar de politie zoekt."},
            "interest": {"en": "Can a website see this directly?", "nl": "Kan een website dit direct zien?"},
            "clicks": {"en": "Is a click always interest, or sometimes a striking headline?", "nl": "Is een klik altijd interesse, of soms een opvallende kop?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the six things again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de zes dingen opnieuw uit."},
            "empty": {"en": "Put each thing under what it is.",
                      "nl": "Zet elk ding bij wat het is."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "*{item}* belongs on the other side.",
                      "nl": "*{item}* hoort aan de andere kant."},
            "correct": {"en": "All {n} are right. Each proxy is easier to count than its goal, and each comes apart from it for some people: less is spent on some patients, who is arrested depends on where the police look, and a click is not always interest.",
                        "nl": "Alle {n} kloppen. Elke proxy is makkelijker te tellen dan het doel, en elke proxy wijkt er voor sommige mensen van af: aan sommige patiënten wordt minder uitgegeven, wie er wordt opgepakt, hangt af van waar de politie zoekt, en een klik is niet altijd interesse."},
        },
    ),
}

_SORTED = {"need": "goal", "costs": "proxy", "crime": "goal", "arrest": "proxy", "interest": "goal", "clicks": "proxy"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("picked", _START, "empty", None),
    ("picked", _SOLUTION, "correct", None),
    ("picked", _swap("for patient in costs:", "for patient in illnesses:"), "correct", None),
    ("picked", _swap("costs[patient] >= threshold", "threshold <= costs[patient]"), "correct", None),
    ("picked", _s(_BODY.replace("in costs", "in _costs")), "error", "underscore"),
    ("picked", _swap("costs[patient] >= threshold", "illnesses[patient] >= 4"), "wrong", "by-need"),
    ("picked", _swap(">= threshold", "> threshold"), "wrong", "greater"),
    ("picked", _swap("costs[patient] >= threshold", "costs[patient] < threshold"), "wrong", "below"),
    ("picked", _swap("picked.append(patient)", "picked.append(costs[patient])"), "wrong", "costs-listed"),
    ("picked", _s("picked = ['B1']\n"), "wrong", None),
    ("picked", _SOLUTION.replace('"B2": 8000', '"B2": 9000'), "wrong", "changed-line"),
    ("picked", _swap("costs[patient] >= threshold", "costs[patient] >= 9000"), "almost", "other-value"),
    ("picked", _s("...\npicked = []\n"), "wrong", "dots-left"),
    ("proxies", {}, "empty", None),
    ("proxies", _SORTED, "correct", None),
    ("proxies", {"need": "goal", "clicks": "proxy"}, "wrong", "unfinished"),
    ("proxies", {**_SORTED, "costs": "goal"}, "wrong", "wrong:costs"),
    ("proxies", {**_SORTED, "arrest": "goal"}, "wrong", "wrong:arrest"),
    ("proxies", {**_SORTED, "interest": "proxy"}, "wrong", "wrong:interest"),
]
