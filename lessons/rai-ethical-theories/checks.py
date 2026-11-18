# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-ethical-theories: the share of the worst-off group that each policy of a hospital
program helps, sorting arguments by the ethical theory they come from, and the picture of the two policies."""
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort
from everything_ai import yours

# The hospital of the lesson, made up by the course: 300 patients live near the hospital and 100 far away, and a
# follow-up program can choose its patients in two ways. For each way: the readmissions it prevents in each group.
PEOPLE = {"near": 300, "far": 100}
PREVENTED = {"nearby first": {"near": 30, "far": 2}, "need first": {"near": 18, "far": 10}}


def totals() -> dict[str, int]:
    """What a consequentialist adds up: the readmissions each policy prevents, over everyone."""
    return {p: g["near"] + g["far"] for p, g in PREVENTED.items()}


def worst() -> dict[str, float]:
    """What someone behind Rawls's veil looks at: the share helped in the group that is worst off."""
    return {p: min(g["near"] / PEOPLE["near"], g["far"] / PEOPLE["far"]) for p, g in PREVENTED.items()}


def picture(labels: dict[str, str]) -> str:
    """Two panels: the readmissions each policy prevents in total, and the share helped in the worst-off group."""
    from everything_ai import chart
    names = [labels["nearby"], labels["need"]]
    left = chart.Panel((0, 3), (0, 40), title=labels["totals"], ylabel=labels["totals_y"], xticks=[])
    for i, total in enumerate(totals().values()):
        left.rect(i + 1 - 0.3, 0, i + 1 + 0.3, total, "data")
        left.text(i + 1, total + 2, names[i])
    right = chart.Panel((0, 3), (0, 0.08), title=labels["worst"], ylabel=labels["worst_y"], xticks=[])
    for i, share in enumerate(worst().values()):
        right.rect(i + 1 - 0.3, 0, i + 1 + 0.3, share, "data")
        right.text(i + 1, share + 0.004, names[i])
    return chart.figure([left, right], alt=labels["alt"], width=300, height=260)


def _worst_picture(seen) -> str:
    """The learner's share for each policy as a bar, the right share as a dashed line, and the share helped in each
    of the two groups as a dot: the right bar reaches the lower dot."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value or not isinstance(mine.value, dict) or not seen.want.has_value:
        return ""
    got, want = mine.value, seen.want.value
    policies = list(PREVENTED)

    def number(v) -> float:
        try:
            return min(0.13, max(0.0, float(v))) if not isinstance(v, (bool, str)) else 0.0
        except (TypeError, ValueError):
            return 0.0
    panel = chart.Panel(x=(-0.6, len(policies) + 0.1), y=(0, 0.13), ylabel=seen.word({"en": "share helped", "nl": "aandeel geholpen"}),
                        yticks=[0, 0.05, 0.1], xticks=list(range(len(policies))), xtick_labels=policies)
    for i, policy in enumerate(policies):
        if number(got.get(policy)) > 0:
            panel.rect(i - 0.3, 0, i + 0.3, number(got.get(policy)), role="data")
        panel.line([i - 0.38, i + 0.38], [number(want.get(policy))] * 2, role="ink", width=2, dashed=True)
        for g in PEOPLE:
            share = PREVENTED[policy][g] / PEOPLE[g]
            panel.points([i + 0.45], [share], role="ink", r=4, hollow=True)
            panel.text(i + 0.53, share, seen.word({"near": {"en": "near", "nl": "dichtbij"}, "far": {"en": "far", "nl": "ver weg"}}[g]), anchor="start")
    title = seen.word({"en": "Your share helped in the worst-off group, per policy", "nl": "Jouw aanaandeel geholpen in de groep die het slechtst af is, per aanpak"})
    words = {"en": ("your code", "right", "share helped in each group"), "nl": ("jouw code", "goed", "aandeel geholpen per groep")}[seen.lang]
    key = (f'<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);display:flex;gap:1.2rem;'
           f'flex-wrap:wrap;margin-top:0.3rem">'
           f'<span><svg width="14" height="12" aria-hidden="true"><rect x="1" y="1" width="12" height="10" style="fill:var(--ea-data);'
           f'fill-opacity:0.18;stroke:var(--ea-data);stroke-width:1.2"/></svg> {words[0]}</span>'
           f'<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-ink);'
           f'stroke-width:2;stroke-dasharray:5 5"/></svg> {words[1]}</span>'
           f'<span><svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="4" style="fill:none;stroke:var(--ea-ink);stroke-width:1.5"/></svg> '
           f'{words[2]}</span></div>')
    return yours.figure(title, chart.figure([panel], alt=title, width=360, height=240) + key)


# The lesson is done when these are correct. "ubuntu" is the choice question in the lesson.
REQUIRED = ["worst", "theories", "ubuntu"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_NO_KEY = Mistake("no-key", error=lambda e: isinstance(e, KeyError) and bool(e.args) and isinstance(e.args[0], str),
                  en="One of your keys is not in the dictionary. `groups` and `people` both have the keys `\"near\"` and `\"far\"`, in small letters, each between quotes.",
                  nl="Een van je keys staat niet in de dictionary. `groups` en `people` hebben allebei de keys `\"near\"` en `\"far\"`, in kleine letters, elk tussen aanhalingstekens.")

_PEOPLE_LINE = 'people = {"near": 300, "far": 100}'
_PREVENTED_LINE = 'prevented = {"nearby first": {"near": 30, "far": 2}, "need first": {"near": 18, "far": 10}}'
_HEAD = f"{_PEOPLE_LINE}\n{_PREVENTED_LINE}\n\nworst = {{}}\nfor policy, groups in prevented.items():\n"
_START = _HEAD + "    ...\nworst\n"
_LINE = '    worst[policy] = min(groups["near"] / people["near"], groups["far"] / people["far"])\n'
_SOLUTION = _HEAD + _LINE + "worst"


def _w(body: str) -> str:
    return _HEAD + body + "worst"


EXERCISES = {
    "worst": CodeBox(
        name="worst",
        picture=_worst_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_PEOPLE_LINE],
        # The check also runs the box with other numbers: there the group that is worst off changes.
        others=[{"prevented": Code('{"nearby first": {"near": 24, "far": 10}, "need first": {"near": 27, "far": 6}}')}],
        first=[_DOTS],
        digits=3,
        mistakes=[
            _NO_KEY,
            Mistake("best", wrong=_w('    worst[policy] = max(groups["near"] / people["near"], groups["far"] / people["far"])\n'),
                    en="Your last line gives `{got}`: the share helped in the group that is best off. Behind the veil you could end up in either group, so look at the group that is worst off: the smaller share.",
                    nl="Je laatste regel geeft `{got}`: het aandeel dat geholpen wordt in de groep die het best af is. Achter de sluier kun je in elke groep terechtkomen, dus kijk naar de groep die het slechtst af is: het kleinere aandeel."),
            Mistake("counts", wrong=_w('    worst[policy] = min(groups["near"], groups["far"])\n'),
                    en="Your last line gives `{got}`: the smaller number of readmissions prevented, not a share. The groups differ in size, 300 and 100, so first divide each number by the people in its group.",
                    nl="Je laatste regel geeft `{got}`: het kleinere aantal voorkomen heropnames, geen aandeel. De groepen zijn niet even groot, 300 en 100, dus deel elk aantal eerst door de mensen in de groep."),
            Mistake("swapped", wrong=_w('    worst[policy] = min(groups["near"] / people["far"], groups["far"] / people["near"])\n'),
                    en="Your last line gives `{got}`. Each group's number goes with its own size: the readmissions prevented near the hospital divided by the people near it, and the same for far away.",
                    nl="Je laatste regel geeft `{got}`. Het aantal van elke groep hoort bij haar eigen grootte: de voorkomen heropnames dichtbij gedeeld door de mensen dichtbij, en net zo voor ver weg."),
            Mistake("overall", wrong=_w('    worst[policy] = (groups["near"] + groups["far"]) / (people["near"] + people["far"])\n'),
                    en="Your last line gives `{got}`: the share helped over all 400 patients. That mixes the two groups again; work out the share for each group, and keep the smaller one.",
                    nl="Je laatste regel geeft `{got}`: het aandeel dat geholpen wordt over alle 400 patiënten. Dat gooit de twee groepen weer op één hoop; reken het aandeel per groep uit, en houd het kleinste over."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with one line: put the share helped in the worst-off group of this policy in `worst`, under the policy.",
                      "nl": "Vervang de `...` door één regel: zet het aandeel dat geholpen wordt in de groep die het slechtst af is in `worst`, onder de aanpak."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other numbers by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere getallen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `worst` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `worst` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does each round divide the number of each group by the people in that group, and keep the smaller of the two shares?",
                      "nl": "Je laatste regel geeft `{got}`. Deelt elke ronde het aantal van elke groep door de mensen in die groep, en houdt hij het kleinste van de twee aandelen over?"},
            "other": {"en": "Right for these numbers. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out from `groups` and `people` themselves?",
                      "nl": "Goed voor deze getallen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `groups` en `people` zelf?"},
            "other_error": {"en": "Right for these numbers. But with {where}, your code stops with an error. Do you work it out from `groups` and `people` themselves?",
                            "nl": "Goed voor deze getallen. Maar met {where} stopt je code met een fout. Reken je het uit met `groups` en `people` zelf?"},
            "correct": {"en": "0.02 for nearby first and 0.06 for need first. Nearby first prevents more readmissions in total, 32 against 28, but helps only 2 of the 100 patients who live far away; need first helps three times as large a share of its worst-off group. The check also tried other numbers.",
                        "nl": "0,02 voor nearby first en 0,06 voor need first. Nearby first voorkomt in totaal meer heropnames, 32 tegen 28, maar helpt maar 2 van de 100 patiënten die ver weg wonen; need first helpt een drie keer zo groot aandeel van de groep die er het slechtst aan toe is. De check probeerde ook andere getallen."},
        },
    ),
    "theories": Sort(
        name="theories",
        groups={
            "consequences": {"en": "Consequences", "nl": "Gevolgen"},
            "duties": {"en": "Duties and rights", "nl": "Plichten en rechten"},
            "virtues": {"en": "Virtues", "nl": "Deugden"},
            "care": {"en": "Care", "nl": "Zorg"},
        },
        items={
            "total": {"en": "**The total.** Nearby first prevents 32 readmissions and need first 28, so nearby first is better.",
                      "nl": "**Het totaal.** Nearby first voorkomt 32 heropnames en need first 28, dus nearby first is beter."},
            "later": {"en": "**Later on.** If patients far away stop trusting the hospital, more of them come back sicker later; count that too.",
                      "nl": "**Op de lange duur.** Als patiënten ver weg het ziekenhuis niet meer vertrouwen, komen er later meer zieker terug; tel dat ook mee."},
            "right": {"en": "**An equal right.** Every patient has the same right to the program, wherever they live.",
                      "nl": "**Een gelijk recht.** Elke patiënt heeft hetzelfde recht op het programma, waar die ook woont."},
            "means": {"en": "**Not a means.** Leaving patients out because they bring the total down uses them for the sake of a number.",
                      "nl": "**Geen middel.** Patiënten weglaten omdat ze het totaal omlaag halen, gebruikt ze voor een getal."},
            "doctor": {"en": "**A good doctor.** A doctor with good character would not let an address decide who gets care.",
                       "nl": "**Een goede dokter.** Een dokter met een goed karakter laat een adres niet bepalen wie zorg krijgt."},
            "hospital": {"en": "**What kind of hospital.** Do we want to be a hospital that is just and honest, or one that only chases a number?",
                         "nl": "**Wat voor ziekenhuis.** Willen we een ziekenhuis zijn dat rechtvaardig en eerlijk is, of een dat alleen een getal najaagt?"},
            "ask": {"en": "**Ask them.** Talk with the patients who live far away: what would they need to be able to come?",
                    "nl": "**Vraag het hun.** Praat met de patiënten die ver weg wonen: wat hebben ze nodig om te kunnen komen?"},
            "alone": {"en": "**Who depends on us.** Look first after the patients who go home to nobody, and keep the bond with their nurse.",
                      "nl": "**Wie van ons afhangt.** Zorg eerst voor de patiënten die naar een leeg huis gaan, en houd de band met hun verpleegkundige."},
        },
        solution={"total": "consequences", "later": "consequences", "right": "duties", "means": "duties",
                  "doctor": "virtues", "hospital": "virtues", "ask": "care", "alone": "care"},
        nudge={
            "total": {"en": "What does this argument count?", "nl": "Wat telt dit argument?"},
            "later": {"en": "It looks further ahead, but what does it still add up?", "nl": "Het kijkt verder vooruit, maar wat telt het nog steeds op?"},
            "right": {"en": "Does this argument depend on how many readmissions are prevented?", "nl": "Hangt dit argument af van hoeveel heropnames er worden voorkomen?"},
            "means": {"en": "Which rule about how you may treat people does this name?", "nl": "Welke regel over hoe je mensen mag behandelen noemt dit?"},
            "doctor": {"en": "Is this about a rule, an outcome, or the kind of person who decides?", "nl": "Gaat dit over een regel, een uitkomst, of over wat voor iemand er beslist?"},
            "hospital": {"en": "It asks about character, of a hospital this time. Which theory asks that?", "nl": "Het vraagt naar karakter, deze keer van een ziekenhuis. Welke theorie vraagt dat?"},
            "ask": {"en": "Is this about a total, a rule, or about listening to the people involved?", "nl": "Gaat dit over een totaal, een regel, of over luisteren naar de mensen om wie het gaat?"},
            "alone": {"en": "It starts from relationships and from who depends on others. Which theory does that?", "nl": "Het begint bij relaties en bij wie van anderen afhangt. Welke theorie doet dat?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the arguments again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de argumenten opnieuw uit."},
            "empty": {"en": "Put each argument with the theory it comes from.",
                      "nl": "Zet elk argument bij de theorie waar het uit komt."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The argument *{item}* belongs with another theory.",
                      "nl": "Het argument *{item}* hoort bij een andere theorie."},
            "correct": {"en": "All {n} are right. Two arguments add up outcomes, two hold on to a right or a rule, two ask about character and two start from the people involved.",
                        "nl": "Alle {n} kloppen. Twee argumenten tellen uitkomsten op, twee houden vast aan een recht of een regel, twee vragen naar karakter en twee beginnen bij de mensen om wie het gaat."},
        },
    ),
}

_SORTED = {"total": "consequences", "later": "consequences", "right": "duties", "means": "duties",
           "doctor": "virtues", "hospital": "virtues", "ask": "care", "alone": "care"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("worst", _START, "empty", None),
    ("worst", _SOLUTION, "correct", None),
    ("worst", _w('    near = groups["near"] / people["near"]\n    far = groups["far"] / people["far"]\n    worst[policy] = min(near, far)\n'), "correct", None),
    ("worst", _w('    worst[policy] = 1.0\n    for g in people:\n        worst[policy] = min(worst[policy], groups[g] / people[g])\n'), "correct", None),
    ("worst", _w('    worst[policy] = max(groups["near"] / people["near"], groups["far"] / people["far"])\n'), "wrong", "best"),
    ("worst", _w('    worst[policy] = min(groups["near"], groups["far"])\n'), "wrong", "counts"),
    ("worst", _w('    worst[policy] = min(groups["near"] / people["far"], groups["far"] / people["near"])\n'), "wrong", "swapped"),
    ("worst", _w('    worst[policy] = (groups["near"] + groups["far"]) / (people["near"] + people["far"])\n'), "wrong", "overall"),
    ("worst", _w('    worst[policy] = groups["far"] / people["far"]\n'), "wrong", None),
    ("worst", _HEAD + "    ...\nworst = {'nearby first': 0.02, 'need first': 0.06}\nworst", "wrong", "dots-left"),
    ("worst", _w("    worst = {'nearby first': 0.02, 'need first': 0.06}\n"), "almost", "other-value"),
    ("worst", _SOLUTION.replace(_PEOPLE_LINE, 'people = {"near": 300, "far": 200}'), "wrong", "changed-line"),
    ("worst", _w('    worst[policy] = min(groups["Near"] / people["near"], groups["far"] / people["far"])\n'), "error", "no-key"),
    ("theories", {}, "empty", None),
    ("theories", _SORTED, "correct", None),
    ("theories", {"total": "consequences", "ask": "care"}, "wrong", "unfinished"),
    ("theories", {**_SORTED, "later": "care"}, "wrong", "wrong:later"),
    ("theories", {**_SORTED, "means": "consequences"}, "wrong", "wrong:means"),
    ("theories", {**_SORTED, "hospital": "duties"}, "wrong", "wrong:hospital"),
    ("theories", {**_SORTED, "alone": "virtues"}, "wrong", "wrong:alone"),
]
