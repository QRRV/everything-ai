# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-case-uber-crash: how many seconds of the pedestrian's path the test car's system still
used when it saw the emergency, sorting the findings of the NTSB, and what happened in court; with the pictures of
the last six seconds."""
import html
import re

from everything_ai.check import CodeBox, Mistake, Sort
from everything_ai import yours

# The NTSB report (HAR-19/03): what the system took the pedestrian for, in seconds before impact. Between -3.8 and -2.7
# it switched several more times between vehicle and other; this list leaves those out, which does not change the
# result at -1.2. At -1.2 the system saw an emergency.
KINDS = {-5.6: "vehicle", -5.2: "other", -4.2: "vehicle", -2.6: "bicycle", -1.5: "other", -1.2: "bicycle"}
NOW = -1.2
# A made-up object for the worked example, seen once a second.
EXAMPLE = {-5.0: "bicycle", -4.0: "bicycle", -3.0: "other", -2.0: "other", -1.0: "other"}
EXAMPLE_NOW = -1.0

# The rest of the report used in the pictures: braking held back for 1 second from the emergency, then a sound and a
# gentle slowdown from -0.2; the operator looked down from 6 seconds before impact for 5 seconds.
HELD_BACK = (-1.2, -0.2)
LOOKING_DOWN = (-6.0, -1.0)


def kept(kinds: dict, now: float) -> tuple[float, float]:
    """The simplified rule of the report: when the kind of an object changes, the system forgets its path so far. The
    moment from which it still counts the path, and how many seconds that is at `now`."""
    previous = ""
    start = None
    for time in kinds:
        if kinds[time] != previous:
            start = time
        previous = kinds[time]
    return start, now - start


_UI = "font-family:var(--ea-font-ui)"


def picture(labels: dict) -> str:
    """The last six seconds before the crash in three rows: what the system took the pedestrian for, what the system
    did, and where the operator looked. `labels` gives "title", "rows" (three texts), "kinds" (a word for vehicle,
    other and bicycle), "switches", "held", "key", "down", "impact", "xlabel", "decimal" and "alt"."""
    from everything_ai import chart
    p = chart.Panel(x=(-6.4, 0.6), y=(0, 4.4), xlabel=labels["xlabel"], yticks=[],
                    xticks=[-6, -5, -4, -3, -2, -1, 0], xtick_labels=["-6", "-5", "-4", "-3", "-2", "-1", "0"])
    rows = labels["rows"]
    # row 1: the kinds, with their words above or below the dots, so that close moments do not overlap
    p.text(-6.3, 3.75, rows[0], "ink-2", anchor="start")
    times = list(KINDS)
    p.line([times[0], NOW], [2.95, 2.95], role="ink-3", width=1.5)
    p.line([-3.8, -2.7], [2.95, 2.95], role="ink", width=1.5, dashed=True)
    p.text(-3.25, 2.55, labels["switches"], "ink-3")
    p.points(times, [2.95] * len(times), role="ink", r=4)
    below = (-5.2, -1.5)
    for t in times:
        p.text(t, 2.55 if t in below else 3.3, labels["kinds"][KINDS[t]], "ink")
    # row 2: what the system did
    p.text(-6.3, 2.0, rows[1], "ink-2", anchor="start")
    p.rect(HELD_BACK[0], 1.3, HELD_BACK[1], 1.6, role="ink-3")
    p.rect(HELD_BACK[1], 1.3, 0, 1.6, role="ink", solid=True)
    p.text(HELD_BACK[0] - 0.1, 1.45, labels["held"], "ink", anchor="end")
    # row 3: the operator
    p.text(-6.3, 0.95, rows[2], "ink-2", anchor="start")
    p.rect(LOOKING_DOWN[0], 0.25, LOOKING_DOWN[1], 0.55, role="ink-3")
    p.text(-3.5, 0.4, labels["down"], "ink")
    p.rect(LOOKING_DOWN[1], 0.25, 0, 0.55, role="ink", solid=True)
    # the crash
    p.line([0, 0], [0, 4.0], role="ink", width=1.5)
    p.text(0.05, 4.2, labels["impact"], "ink", anchor="end")
    svg = chart.figure([p], alt=labels["alt"], width=540, height=340)
    key = (f'<div style="{_UI};font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem">'
           f'{html.escape(labels["key"])}</div>')
    return yours.figure(labels["title"], (chart.with_decimal_comma(svg) if labels["decimal"] == "," else svg) + key)


def _number(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _kept_picture(seen) -> str:
    """The moments of the report on a line of seconds, the time since the first detection as a light bar, and the
    learner's `history` as a dark bar that ends at the emergency."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    got = _number(mine.value)
    if got is None:
        return ""
    p = chart.Panel(x=(-6.4, 0.6), y=(0, 2.7), xlabel=seen.word({"en": "seconds before the crash", "nl": "seconden voor het ongeluk"}),
                    yticks=[], xticks=[-6, -5, -4, -3, -2, -1, 0], xtick_labels=["-6", "-5", "-4", "-3", "-2", "-1", "0"])
    times = list(KINDS)
    p.points(times, [2.05] * len(times), role="ink", r=3.5)
    for t in times:
        p.text(t, 1.7 if t in (-5.2, -1.5) else 2.4, KINDS[t], "ink-3")
    p.rect(times[0], 0.95, NOW, 1.3, role="ink-3")
    p.text((times[0] + NOW) / 2, 1.12, seen.word({"en": "since the first detection", "nl": "sinds de eerste detectie"}), "ink-2")
    start = max(-6.4, min(NOW, NOW - got))
    if NOW - start > 0.02:
        p.rect(start, 0.3, NOW, 0.65, role="ink", solid=True)
    else:
        p.line([NOW, NOW], [0.25, 0.7], role="ink", width=3)
    p.text(NOW + 0.1, 0.47, f"history = {got:.1f}", "ink", anchor="start")
    title = seen.word({"en": "The path your rule still counts at -1.2", "nl": "Het pad dat jouw regel bij -1,2 nog telt"})
    return yours.figure(title, chart.figure([p], alt=title, width=520, height=240))


# The lesson is done when these are correct. "court" is the choice question in the lesson.
REQUIRED = ["history", "verdict", "court"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _line(name: str, values) -> str:
    return f"{name} = {values!r}".replace("'", '"')


_KINDS_LINE = _line("kinds", KINDS)
_HEAD = ("# seconds before the crash, and what the system took the pedestrian for (NTSB; a few switches left out)\n"
         + _KINDS_LINE + "\nnow = -1.2\n\n")
_START = _HEAD + "...\nhistory\n"
_BODY = ('previous = ""\nfor time in kinds:\n    if kinds[time] != previous:\n        start = time\n'
         "    previous = kinds[time]\nhistory = now - start\n")
_SOLUTION = _HEAD + _BODY + "history"


def _s(body: str) -> str:
    return _HEAD + body + "history"


def _swap(old: str, new: str) -> str:
    assert _BODY.count(old) == 1
    return _s(_BODY.replace(old, new))


EXERCISES = {
    "history": CodeBox(
        name="history",
        picture=_kept_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_KINDS_LINE, "now = -1.2"],
        # The check also runs the box with other moments and another emergency.
        others=[{"kinds": {-6.0: "other", -4.5: "other", -3.0: "bicycle", -2.0: "bicycle", -0.5: "bicycle"}, "now": -0.5}],
        first=[_DOTS],
        digits=2,
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_kinds`. In the box the names have none: `kinds`, `now`, `previous`, `time`, `start`, `history`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_kinds`. In het vak hebben de namen dat niet: `kinds`, `now`, `previous`, `time`, `start`, `history`."),
            Mistake("same", error=lambda e: isinstance(e, NameError) and "'start'" in str(e),
                    en="`start` never gets a value: your `if` never comes true. Does the `if` compare the kind with the one before, while `previous` still holds the old kind? And does it look for a *different* kind, with `!=`?",
                    nl="`start` krijgt nooit een waarde: je `if` is nooit waar. Vergelijkt de `if` de soort met de vorige, terwijl `previous` nog de oude soort bevat? En zoekt hij een *andere* soort, met `!=`?"),
            Mistake("whole", matches=lambda got, text: abs(got - 4.4) < 1e-6 if isinstance(got, (int, float)) else False,
                    en="Your last line gives `{got}`: the seconds since the system first detected the pedestrian, at -5.6. But the rule forgets the path each time the kind changes: set `start` again at each change.",
                    nl="Je laatste regel geeft `{got}`: de seconden sinds het systeem de voetganger voor het eerst detecteerde, bij -5,6. Maar de regel vergeet het pad telkens als de soort verandert: zet `start` bij elke verandering opnieuw."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the lines of the worked example: walk through `kinds`, move `start` each time the kind changes, and work out `history`.",
                      "nl": "Vervang de `...` door de regels van het uitgewerkte voorbeeld: loop door `kinds`, verschuif `start` telkens als de soort verandert, en reken `history` uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the lines of `kinds` and `now` exactly as they were: put them back. The check tries other moments by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `kinds` en `now` nodig zoals ze waren: zet ze terug. De check probeert zelf andere momenten."},
            "no_value": {"en": "The last line of the box shows nothing. Put `history` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `history` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does `start` move to `time` only when the kind differs from `previous`, does `previous` take each kind in turn, and is `history` the seconds from `start` to `now`?",
                      "nl": "Je laatste regel geeft `{got}`. Verschuift `start` alleen naar `time` als de soort anders is dan `previous`, krijgt `previous` elke soort op zijn beurt, en is `history` het aantal seconden van `start` tot `now`?"},
            "other": {"en": "Right for these moments. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does `start` move only when the kind changes, does `previous` remember the kind each time, and is `history` the seconds from `start` to `now`?",
                      "nl": "Goed voor deze momenten. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Verschuift `start` alleen als de soort verandert, onthoudt `previous` elke keer de soort, en is `history` het aantal seconden van `start` tot `now`?"},
            "other_error": {"en": "Right for these moments. But with {where}, your code stops with an error. Do you use `kinds` and `now` themselves?",
                            "nl": "Goed voor deze momenten. Maar met {where} stopt je code met een fout. Gebruik je `kinds` en `now` zelf?"},
            "correct": {"en": "`0.0`: by the simplified rule, when the system saw the emergency, 1.2 seconds before the crash, the kind had just changed to bicycle, so it counted the path from that moment and had no seconds of it. It had detected the pedestrian 4.4 seconds earlier.",
                        "nl": "`0.0`: volgens de vereenvoudigde regel was de soort, toen het systeem het noodgeval zag, 1,2 seconde voor het ongeluk, net veranderd in `bicycle`, dus telde het het pad vanaf dat moment en had het er geen seconde van. Het had de voetganger 4,4 seconden eerder gedetecteerd."},
        },
    ),
    "verdict": Sort(
        name="verdict",
        groups={
            "cause": {"en": "The probable cause", "nl": "De waarschijnlijke oorzaak"},
            "uber": {"en": "Contributing: Uber", "nl": "Droeg bij: Uber"},
            "further": {"en": "Further factors", "nl": "Verdere factoren"},
        },
        items={
            "phone": {"en": "**The phone.** The operator did not watch the road or the system, distracted by a phone.",
                      "nl": "**De telefoon.** De operator lette niet op de weg of het systeem, afgeleid door een telefoon."},
            "risks": {"en": "**The risks.** The procedures for assessing safety risks were inadequate.",
                      "nl": "**De risico's.** De procedures om veiligheidsrisico's te beoordelen schoten tekort."},
            "watching": {"en": "**The oversight.** The oversight of the vehicle operators was ineffective.",
                         "nl": "**Het toezicht.** Het toezicht op de operators werkte niet."},
            "trust": {"en": "**The trust.** There was no adequate way to keep the operators from relying on the system too much.",
                      "nl": "**Het vertrouwen.** Er was geen goede manier om te voorkomen dat de operators te veel op het systeem leunden."},
            "crossing": {"en": "**The crossing.** The pedestrian crossed the road at a place without a crosswalk.",
                         "nl": "**De oversteek.** De voetganger stak de weg over op een plek zonder zebrapad."},
            "state": {"en": "**The state.** Arizona's transport department did not oversee the testing enough.",
                      "nl": "**De staat.** Het vervoersdepartement van Arizona hield te weinig toezicht op de tests."},
        },
        solution={"phone": "cause", "risks": "uber", "watching": "uber", "trust": "uber", "crossing": "further", "state": "further"},
        nudge={
            "phone": {"en": "Which one finding did the NTSB call the probable cause?", "nl": "Welke ene bevinding noemde de NTSB de waarschijnlijke oorzaak?"},
            "risks": {"en": "Whose procedures were these?", "nl": "Van wie waren deze procedures?"},
            "watching": {"en": "Who was meant to oversee the operators?", "nl": "Wie moest toezicht houden op de operators?"},
            "trust": {"en": "Who could have built something against too much trust?", "nl": "Wie had iets tegen te veel vertrouwen kunnen inbouwen?"},
            "crossing": {"en": "Was this Uber's doing?", "nl": "Lag dit aan Uber?"},
            "state": {"en": "Is Arizona part of Uber?", "nl": "Hoort Arizona bij Uber?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the findings again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de bevindingen opnieuw uit."},
            "empty": {"en": "Sort each finding as the NTSB did.",
                      "nl": "Deel elke bevinding in zoals de NTSB dat deed."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The finding *{item}* belongs somewhere else.",
                      "nl": "De bevinding *{item}* hoort ergens anders."},
            "correct": {"en": "All {n} are right. One cause and five more: the NTSB called Uber's three a consequence of its inadequate safety culture.",
                        "nl": "Alle {n} kloppen. Eén oorzaak en nog vijf: de drie van Uber noemde de NTSB een gevolg van een veiligheidscultuur die tekortschoot."},
        },
    ),
}

_SORTED = {"phone": "cause", "risks": "uber", "watching": "uber", "trust": "uber", "crossing": "further", "state": "further"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("history", _START, "empty", None),
    ("history", _SOLUTION, "correct", None),
    ("history", _swap("    if kinds[time] != previous:\n        start = time\n    previous = kinds[time]\n",
                      "    if previous != kinds[time]:\n        start = time\n        previous = kinds[time]\n"), "correct", None),
    ("history", _s(_BODY.replace("in kinds", "in _kinds")), "error", "underscore"),
    ("history", _swap("kinds[time] != previous", "kinds[time] == previous"), "error", "same"),
    ("history", _s("history = now - -5.6\n"), "wrong", "whole"),
    ("history", _swap("    if kinds[time] != previous:\n        start = time\n    previous = kinds[time]\n",
                      "    previous = kinds[time]\n    if kinds[time] != previous:\n        start = time\n"), "error", "same"),
    ("history", _s("history = 3\n"), "wrong", None),
    ("history", _s("history = 0.0\n"), "almost", "other-value"),
    ("history", _SOLUTION.replace('-5.6: "vehicle"', '-5.6: "other"'), "wrong", "changed-line"),
    ("history", _SOLUTION.replace("now = -1.2", "now = -1.5"), "wrong", "changed-line"),
    ("history", _swap("    previous = kinds[time]\n", ""), "almost", "other-value"),
    ("history", _swap("now - start", "start - now"), "almost", "other-value"),
    ("history", _s("...\nhistory = 0\n"), "wrong", "dots-left"),
    ("verdict", {}, "empty", None),
    ("verdict", _SORTED, "correct", None),
    ("verdict", {"phone": "cause", "state": "further"}, "wrong", "unfinished"),
    ("verdict", {**_SORTED, "phone": "uber"}, "wrong", "wrong:phone"),
    ("verdict", {**_SORTED, "trust": "cause"}, "wrong", "wrong:trust"),
    ("verdict", {**_SORTED, "state": "uber"}, "wrong", "wrong:state"),
]
