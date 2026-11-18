# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-feedback-loops: working out how many crimes the police found per day they were in a
district, sorting data into what police found where a system sent them and what people reported themselves; with
the picture of five runs of a feedback loop between two districts with the same crime, and the picture of the share
of all crimes found beside the crimes found per day."""
import re

from everything_ai.check import CodeBox, Mistake, Sort
from everything_ai import yours

# Made-up records of one year: the days with police, and the crimes they found, per district.
DAYS = {"North": 330, "South": 35}
FOUND = {"North": 99, "South": 11}
RATE = 0.3        # the demo: the chance that police find a crime on a day they are in a district, the same in both
SEEDS = [1, 2, 3, 4, 5]


def urn(seed: int, days: int = 1000, rate: float = RATE) -> tuple[list, list]:
    """The demo of the lesson, after the urn of Ensign and colleagues: each day police go to district 0 with a chance
    equal to its share of the crimes found so far, and find a crime there with chance `rate`. Gives the crimes found
    in the end, and the share of district 0 after each day."""
    import numpy as np
    rng = np.random.default_rng(seed)
    found, shares = [1, 1], []
    for _ in range(days):
        share = found[0] / (found[0] + found[1])
        sent = 0 if rng.random() < share else 1
        if rng.random() < rate:
            found[sent] += 1
        shares.append(found[0] / (found[0] + found[1]))
    return found, shares


def share_of_found(found: dict) -> dict:
    total = sum(found.values())
    return {d: found[d] / total for d in found}


def per_day(found: dict, days: dict) -> dict:
    return {d: found[d] / days[d] for d in found}


def picture(labels: dict) -> str:
    """The five runs of the demo: the share of the crimes found that are in district 0, after each day, with a dashed
    line at the share of the crime that is really there, a half. `labels` gives "title", "xlabel", "ylabel", "true",
    "decimal" (the decimal sign along the axes) and "alt" (with {finals} for the five shares at the end). Orange: what
    the loop learns."""
    from everything_ai import chart
    p = chart.Panel(x=(0, 1000), y=(0, 1), xlabel=labels["xlabel"], ylabel=labels["ylabel"],
                    xticks=[0, 250, 500, 750, 1000], xtick_labels=["0", "250", "500", "750", "1000"],
                    yticks=[0, 0.25, 0.5, 0.75, 1])
    p.line([0, 1000], [0.5, 0.5], role="ink-3", width=1, dashed=True)
    p.text(990, 0.535, labels["true"], "ink-3", anchor="end", halo=True)
    finals = []
    for seed in SEEDS:
        found, shares = urn(seed)
        p.line(list(range(1, 1001)), shares, role="param", width=1.6)
        finals.append(f"{found[0] / sum(found):.2f}".replace(".", labels["decimal"]))
    alt = labels["alt"].format(finals="; ".join(finals))
    svg = chart.figure([p], alt=alt, width=520, height=280)
    return yours.figure(labels["title"], chart.with_decimal_comma(svg) if labels["decimal"] == "," else svg)


def _rates(v) -> dict | None:
    """The learner's dict of a number per district, when it is one."""
    if type(v) is not dict or not 1 <= len(v) <= 6:
        return None
    for k, x in v.items():
        if not isinstance(k, str) or isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
    return v


def _counts(v) -> bool:
    if type(v) is not dict or not 1 <= len(v) <= 6:
        return False
    for k, x in v.items():
        if not isinstance(k, str) or isinstance(x, bool) or not isinstance(x, int) or not 0 <= x <= 100000:
            return False
    return True


_WORDS = {
    "en": {"title": "Found per day with police: {shown}", "xlabel": "share", "all": "of all crimes found",
           "day": "found per day with police", "key": "open: share of all crimes found, filled: found per day with police",
           "found": "{found} found in {days} days"},
    "nl": {"title": "Gevonden per dag met politie: {shown}", "xlabel": "deel", "all": "van alle gevonden misdrijven",
           "day": "gevonden per dag met politie", "key": "open: deel van alle gevonden misdrijven, gevuld: gevonden per dag met politie",
           "found": "{found} gevonden in {days} dagen"},
}


def _per_day_picture(seen) -> str:
    """Per district an open bar for its share of all the crimes found, and a filled bar for the learner's crimes found
    per day with police, with the counts behind them."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value or _rates(mine.value) is None:
        return ""
    names = mine.names if isinstance(mine.names, dict) else {}
    days, found = names.get("days", DAYS), names.get("found", FOUND)
    if not _counts(days) or not _counts(found) or set(days) != set(found) or sum(found.values()) == 0:
        return ""
    value = mine.value
    if set(value) != set(found) or not all(0 <= value[d] <= 1 for d in value):
        return ""
    w = _WORDS[seen.lang]
    share = share_of_found(found)
    districts = list(found)
    top = 1.35 * len(districts) + 0.1
    p = chart.Panel(x=(0, 1), y=(0, top), xlabel=w["xlabel"], yticks=[],
                    xticks=[0, 0.25, 0.5, 0.75, 1], xtick_labels=["0", "0.25", "0.5", "0.75", "1"])
    for i, d in enumerate(districts):
        y = top - 0.85 - 1.35 * i
        p.text(0.005, y + 0.62, d + ": " + w["found"].format(found=found[d], days=days[d]), "ink",
               anchor="start")
        p.rect(0, y + 0.12, share[d], y + 0.42, role="data")
        p.rect(0, y - 0.24, value[d], y + 0.06, role="data", solid=True)
    sign = "," if seen.lang == "nl" else "."

    def num(x):
        return f"{x:.2f}".replace(".", sign)

    title = w["title"].format(shown=", ".join(f"{d} {num(value[d])}" for d in districts))
    alt = title + ". " + "; ".join(f"{d}: {num(share[d])} {w['all']}, {num(value[d])} {w['day']}" for d in districts)
    svg = chart.figure([p], alt=alt, width=520, height=220)
    key = (f'<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem">'
           f'{w["key"]}</div>')
    return yours.figure(title, (chart.with_decimal_comma(svg) if seen.lang == "nl" else svg) + key)


def _close(got, want) -> bool:
    """The same districts, and per district the same number to two decimals: rounded or not."""
    a, b = _rates(got), _rates(want)
    if a is None or b is None or a.keys() != b.keys():
        return False
    return all(abs(a[k] - b[k]) < 0.006 for k in a)


# The lesson is done when these are correct. "hiring" is the choice question in the lesson.
REQUIRED = ["per-day", "sources", "hiring"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_DAYS_LINE = 'days = {"North": 330, "South": 35}'
_FOUND_LINE = 'found = {"North": 99, "South": 11}'
_HEAD = ("# made-up records of one year: days with police, and crimes they found, per district\n"
         + _DAYS_LINE + "\n" + _FOUND_LINE + "\n\n")
_START = _HEAD + "...\nper_day\n"
_BODY = ("per_day = {}\nfor district in found:\n"
         "    per_day[district] = round(found[district] / days[district], 2)\n")
_SOLUTION = _HEAD + _BODY + "per_day"


def _s(body: str) -> str:
    return _HEAD + body + "per_day"


def _swap(old: str, new: str) -> str:
    assert _BODY.count(old) == 1
    return _s(_BODY.replace(old, new))


def _near(want: dict):
    return lambda got, text: _close(got, want)


EXERCISES = {
    "per-day": CodeBox(
        name="per-day",
        picture=_per_day_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_DAYS_LINE, _FOUND_LINE],
        compare=_close,
        digits=2,
        # The check also runs the box with other days and other crimes found.
        others=[{"days": {"North": 220, "South": 145}}, {"found": {"North": 66, "South": 14}}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_found`. In the box the names have none: `days`, `found`, `district`, `per_day`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_found`. In het vak hebben de namen dat niet: `days`, `found`, `district`, `per_day`."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks for a key that is not there. `days` and `found` each have `\"North\"` and `\"South\"`, and `per_day` starts empty: give it a value with `per_day[district] = ...` before you read it.",
                    nl="Je code vraagt om een sleutel die er niet is. `days` en `found` hebben elk `\"North\"` en `\"South\"`, en `per_day` begint leeg: geef hem een waarde met `per_day[district] = ...` voordat je die leest."),
            Mistake("one-decimal", matches=_near({"North": 0.3, "South": 0.3}),
                    en="Your last line gives `{got}`: rounded to one decimal, the two districts look the same. Round to two decimals: `round(..., 2)`.",
                    nl="Je laatste regel geeft `{got}`: afgerond op één decimaal lijken de twee wijken gelijk. Rond af op twee decimalen: `round(..., 2)`."),
            Mistake("share", matches=_near({"North": 0.9, "South": 0.1}),
                    en="Your last line gives `{got}`: each district's share of all the crimes found, as in the worked example. Divide by the days the police were in that district instead: `days[district]`.",
                    nl="Je laatste regel geeft `{got}`: het deel van alle gevonden misdrijven per wijk, zoals in het uitgewerkte voorbeeld. Deel in plaats daarvan door de dagen dat de politie in die wijk was: `days[district]`."),
            Mistake("inverted", matches=_near({"North": 3.33, "South": 3.18}),
                    en="Your last line gives `{got}`: days per crime found. Turn the division around: crimes found divided by days, `found[district] / days[district]`.",
                    nl="Je laatste regel geeft `{got}`: dagen per gevonden misdrijf. Draai de deling om: gevonden misdrijven gedeeld door dagen, `found[district] / days[district]`."),
            Mistake("year", matches=_near({"North": 0.27, "South": 0.03}),
                    en="Your last line gives `{got}`: crimes found per day of the whole year. Divide by the days the police were in that district: `days[district]`.",
                    nl="Je laatste regel geeft `{got}`: gevonden misdrijven per dag van het hele jaar. Deel door de dagen dat de politie in die wijk was: `days[district]`."),
            Mistake("number", matches=lambda got, text: isinstance(got, (int, float)) and not isinstance(got, bool),
                    en="Your last line gives one number, `{got}`. `per_day` is a dict with a number per district: start with `per_day = {{}}` and fill it in a loop over the districts.",
                    nl="Je laatste regel geeft één getal, `{got}`. `per_day` is een dict met een getal per wijk: begin met `per_day = {{}}` en vul die in een loop over de wijken."),
            Mistake("list", matches=lambda got, text: isinstance(got, list),
                    en="Your last line gives a list, `{got}`. `per_day` is a dict with a number per district: `per_day[district] = ...`.",
                    nl="Je laatste regel geeft een lijst, `{got}`. `per_day` is een dict met een getal per wijk: `per_day[district] = ...`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a loop over the districts, as in the worked example, that puts in `per_day` the crimes found per day with police.",
                      "nl": "Vervang de `...` door een loop over de wijken, zoals in het uitgewerkte voorbeeld, die in `per_day` de gevonden misdrijven per dag met politie zet."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the lines of `days` and `found` exactly as they were: put them back. The check tries other numbers by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `days` en `found` nodig zoals ze waren: zet ze terug. De check probeert zelf andere getallen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `per_day` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `per_day` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you divide, per district, the crimes found there by the days the police were there?",
                      "nl": "Je laatste regel geeft `{got}`. Deel je per wijk de misdrijven die er gevonden zijn door de dagen dat de politie er was?"},
            "other": {"en": "Right for these numbers. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `days` and `found` themselves, and not their numbers?",
                      "nl": "Goed voor deze getallen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `days` en `found` zelf, en niet hun getallen?"},
            "other_error": {"en": "Right for these numbers. But with {where}, your code stops with an error. Do you use `days` and `found` themselves?",
                            "nl": "Goed voor deze getallen. Maar met {where} stopt je code met een fout. Gebruik je `days` en `found` zelf?"},
            "correct": {"en": "`{{'North': 0.3, 'South': 0.31}}`: on about 3 in 10 days with police, they found a crime, in both districts. The totals said nine to one, but police were in North on 330 of the 365 days. As far as these records go, the difference is in where the police looked, not in the crime.",
                        "nl": "`{{'North': 0.3, 'South': 0.31}}`: op ongeveer 3 van de 10 dagen met politie vond die een misdrijf, in beide wijken. De totalen zeiden negen tegen één, maar de politie was op 330 van de 365 dagen in North. Voor zover deze gegevens het laten zien, zit het verschil in waar de politie keek, niet in de misdrijven."},
        },
    ),
    "sources": Sort(
        name="sources",
        groups={
            "found": {"en": "Found because the system sent someone", "nl": "Gevonden omdat het systeem iemand stuurde"},
            "reported": {"en": "Reported by people themselves", "nl": "Door mensen zelf gemeld"},
        },
        items={
            "patrol": {"en": "**On patrol.** An officer on patrol in the square the system chose sees a fight in the street.",
                       "nl": "**Op patrouille.** Een agent op patrouille in het vak dat het systeem koos, ziet een vechtpartij op straat."},
            "stop": {"en": "**A stop.** Police stop a car in the area the system chose and find drugs in it.",
                     "nl": "**Een controle.** De politie houdt een auto aan in het gebied dat het systeem koos, en vindt er drugs in."},
            "flagged": {"en": "**A flagged car.** A camera system flags a car, and an officer follows it up.",
                        "nl": "**Een gemarkeerde auto.** Een camerasysteem markeert een auto, en een agent gaat erop af."},
            "call": {"en": "**An emergency call.** A resident calls the emergency number about a break-in next door.",
                     "nl": "**Een noodoproep.** Een bewoner belt 112 over een inbraak bij de buren."},
            "burglary": {"en": "**At the station.** The victim of a burglary goes to the police station to report it.",
                         "nl": "**Op het bureau.** Iemand bij wie is ingebroken, gaat naar het politiebureau om aangifte te doen."},
            "shop": {"en": "**Online.** A shop owner reports a theft on the website of the police.",
                     "nl": "**Online.** Een winkelier doet aangifte van een diefstal op de website van de politie."},
        },
        solution={"patrol": "found", "stop": "found", "flagged": "found", "call": "reported", "burglary": "reported",
                  "shop": "reported"},
        nudge={
            "patrol": {"en": "Why was the officer there to see it?",
                       "nl": "Waarom was de agent daar om het te zien?"},
            "stop": {"en": "Who chose where the police would stop cars?",
                     "nl": "Wie koos waar de politie auto's zou aanhouden?"},
            "flagged": {"en": "Would the officer have seen this car without the system?",
                        "nl": "Had de agent deze auto ook zonder het systeem gezien?"},
            "call": {"en": "Does this crime come in whether or not police were sent there?",
                     "nl": "Komt dit misdrijf binnen, of er nu politie naartoe gestuurd is of niet?"},
            "burglary": {"en": "Who brings this crime to the police?",
                         "nl": "Wie brengt dit misdrijf bij de politie?"},
            "shop": {"en": "Did the system play a part in this report?",
                     "nl": "Speelde het systeem een rol bij deze aangifte?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the cases again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de gevallen opnieuw uit."},
            "empty": {"en": "Sort each case by how the crime came to the police.",
                      "nl": "Deel elk geval in bij hoe het misdrijf bij de politie kwam."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The case *{item}* belongs somewhere else.",
                      "nl": "Het geval *{item}* hoort ergens anders."},
            "correct": {"en": "All {n} are right. On patrol, a stop and a flagged car were found because a system sent someone: they feed the loop. The call, the report at the station and the one online come in without the system: they dampen the loop.",
                        "nl": "Alle {n} kloppen. Op patrouille, een controle en een gemarkeerde auto zijn gevonden omdat een systeem iemand stuurde: die voeden de loop. De noodoproep, de aangifte op het bureau en die online komen binnen zonder het systeem: die dempen de loop."},
        },
    ),
}

_SORTED = {"patrol": "found", "stop": "found", "flagged": "found", "call": "reported", "burglary": "reported",
           "shop": "reported"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("per-day", _START, "empty", None),
    ("per-day", _SOLUTION, "correct", None),
    ("per-day", _swap("round(found[district] / days[district], 2)", "found[district] / days[district]"), "correct", None),
    ("per-day", _s("per_day = {}\nfor district in days:\n    rate = found[district] / days[district]\n"
                   "    per_day[district] = rate\n"), "correct", None),
    ("per-day", _s(_BODY.replace("in found", "in _found")), "error", "underscore"),
    ("per-day", _swap("days[district]", 'days["Noord"]'), "error", "key"),
    ("per-day", _swap(", 2)", ", 1)"), "wrong", "one-decimal"),
    ("per-day", _s("total = found[\"North\"] + found[\"South\"]\nper_day = {}\nfor district in found:\n"
                   "    per_day[district] = round(found[district] / total, 2)\n"), "wrong", "share"),
    ("per-day", _swap("found[district] / days[district]", "days[district] / found[district]"), "wrong", "inverted"),
    ("per-day", _swap("found[district] / days[district]", "found[district] / 365"), "wrong", "year"),
    ("per-day", _s("per_day = found[\"North\"] / days[\"North\"]\n"), "wrong", "number"),
    ("per-day", _s("per_day = []\nfor district in found:\n"
                   "    per_day.append(round(found[district] / days[district], 2))\n"), "wrong", "list"),
    ("per-day", _swap("found[district] / days[district]", "found[district] / days[district] * 100"), "wrong", None),
    ("per-day", _s("per_day = {\"North\": 0.3, \"South\": 0.31}\n"), "almost", "other-value"),
    ("per-day", _SOLUTION.replace("\"South\": 35}", "\"South\": 50}"), "wrong", "changed-line"),
    ("per-day", _s("...\nper_day = {\"North\": 0.3, \"South\": 0.31}\n"), "wrong", "dots-left"),
    ("sources", {}, "empty", None),
    ("sources", _SORTED, "correct", None),
    ("sources", {"patrol": "found", "call": "reported"}, "wrong", "unfinished"),
    ("sources", {**_SORTED, "flagged": "reported"}, "wrong", "wrong:flagged"),
    ("sources", {**_SORTED, "call": "found"}, "wrong", "wrong:call"),
    ("sources", {**_SORTED, "stop": "reported"}, "wrong", "wrong:stop"),
]
