# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-abstraction-traps: in cases made up by the course, the share of a group that a risk tool
recommends releasing and the share the judges released, and per county the share of cases where the judge decided
otherwise than the tool; with the picture of each."""
import math
import numbers
import re

from everything_ai import yours
from everything_ai.check import CodeBox, Mistake

# --- the cases --------------------------------------------------------------------------------------------------------
# Made up by the course, after the pattern Stevenson (2018) found in Kentucky: one court with a rural and an urban
# county, two groups of defendants, the tool's recommendation and the judge's decision. "release" is release without
# money; "bond" is a money bond. The rural judges follow the tool; the urban judges decide otherwise in half the cases,
# for group A and group B alike. Group A lives mostly in the rural county, group B mostly in the urban one.
COURT = [
    {"group": "A", "county": "rural", "tool": "release", "judge": "release"},
    {"group": "B", "county": "rural", "tool": "release", "judge": "release"},
    {"group": "A", "county": "rural", "tool": "release", "judge": "release"},
    {"group": "A", "county": "urban", "tool": "release", "judge": "bond"},
    {"group": "B", "county": "urban", "tool": "release", "judge": "release"},
    {"group": "B", "county": "urban", "tool": "release", "judge": "bond"},
    {"group": "A", "county": "rural", "tool": "bond", "judge": "bond"},
    {"group": "B", "county": "urban", "tool": "release", "judge": "bond"},
    {"group": "A", "county": "rural", "tool": "release", "judge": "release"},
    {"group": "B", "county": "rural", "tool": "bond", "judge": "bond"},
    {"group": "B", "county": "urban", "tool": "bond", "judge": "bond"},
    {"group": "A", "county": "urban", "tool": "bond", "judge": "bond"},
    {"group": "B", "county": "urban", "tool": "release", "judge": "bond"},
    {"group": "A", "county": "rural", "tool": "release", "judge": "release"},
    {"group": "B", "county": "urban", "tool": "release", "judge": "release"},
    {"group": "A", "county": "rural", "tool": "release", "judge": "release"},
]
COUNTIES = ["rural", "urban"]
# Another court that the check tries in exercise 3: here the rural judge decides otherwise once, the urban never.
OTHER_COURT = [
    {"group": "A", "county": "rural", "tool": "release", "judge": "bond"},
    {"group": "B", "county": "rural", "tool": "release", "judge": "release"},
    {"group": "A", "county": "urban", "tool": "bond", "judge": "bond"},
    {"group": "B", "county": "urban", "tool": "release", "judge": "release"},
]


def share(cases: list, group: str, key: str) -> float:
    """The share of the cases of `group` in which `key` ("tool" or "judge") says release."""
    rows = [c for c in cases if c["group"] == group]
    return sum(1 for c in rows if c[key] == "release") / len(rows)


def overrides(cases: list, counties: list) -> dict:
    """Per county, the share of its cases in which the judge decided otherwise than the tool."""
    out = {}
    for county in counties:
        rows = [c for c in cases if c["county"] == county]
        out[county] = sum(1 for c in rows if c["judge"] != c["tool"]) / len(rows)
    return out


# --- pictures ---------------------------------------------------------------------------------------------------------
def _number(x) -> float | None:
    """A real number that is not True or False, as a float; else None."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _show(x: float, lang: str, digits: int = 2) -> str:
    text = f"{x:.{digits}f}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"
_FIELDS = ("group", "county", "tool", "judge")


def _cases_ok(cases) -> bool:
    """The learner's cases, when they still look like the lesson's."""
    if not isinstance(cases, list) or not 1 <= len(cases) <= 40:
        return False
    return all(type(c) is dict and all(isinstance(c.get(f), str) for f in _FIELDS) for c in cases)


_SHARE_WORDS = {
    "en": {"title": "Your share for group {group}, {key}: {share}", "xlabel": "share released without money",
           "row": "{group}, {who}: {share}", "tool": "tool", "judge": "judges",
           "key": "filled: your share; open: the other three, worked out by the check",
           "alt": "Four bars: for groups A and B, the share the tool recommends releasing and the share the judges released: {shown}. Yours is {mine}."},
    "nl": {"title": "Jouw aandeel voor groep {group}, {key}: {share}", "xlabel": "deel vrijgelaten zonder geld",
           "row": "{group}, {who}: {share}", "tool": "tool", "judge": "rechters",
           "key": "gevuld: jouw aandeel; open: de andere drie, uitgerekend door de check",
           "alt": "Vier balken: voor groep A en B het deel dat de tool wil vrijlaten en het deel dat de rechters vrijlieten: {shown}. Het jouwe is {mine}."},
}


def _share_picture(seen) -> str:
    """A bar per group and per decision maker, the tool and the judges, as long as the share released: the learner's own
    share filled, the other three open. The tool's bars are the same for both groups; the judges' are not."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    value = _number(mine.value)
    names = mine.names if isinstance(mine.names, dict) else {}
    cases, group, key = names.get("cases"), names.get("group"), names.get("key")
    if value is None or not 0 <= value <= 1 or not _cases_ok(cases) or key not in ("tool", "judge"):
        return ""
    groups = [g for g in ("A", "B") if any(c["group"] == g for c in cases)]
    if group not in groups:
        return ""
    w = _SHARE_WORDS[seen.lang]
    rows = []
    for g in groups:
        for k in ("tool", "judge"):
            v = value if (g, k) == (group, key) else share(cases, g, k)
            rows.append((g, k, v))
    top = 1.1 * len(rows) + 0.4
    ticks = [0, 0.25, 0.5, 0.75, 1]
    p = chart.Panel(x=(0, 1.05), y=(0, top), xlabel=w["xlabel"], yticks=[], xticks=ticks,
                    xtick_labels=[_show(t, seen.lang) for t in ticks])
    for i, (g, k, v) in enumerate(rows):
        y = top - 0.4 - 1.1 * (i + 1)
        p.text(0.004, y + 0.68, w["row"].format(group=g, who=w[k], share=_show(v, seen.lang, 3)), "ink",
               anchor="start", halo=True)
        p.rect(0, y + 0.12, max(v, 0.003), y + 0.42, role="pred" if k == "tool" else "data",
               solid=(g, k) == (group, key))
    shown = "; ".join(f"{g} {w[k]} {_show(v, seen.lang, 3)}" for g, k, v in rows)
    svg = chart.figure([p], alt=w["alt"].format(shown=shown, mine=_show(value, seen.lang, 3)),
                       width=520, height=46 * len(rows) + 90)
    svg = chart.with_decimal_comma(svg) if seen.lang == "nl" else svg
    title = w["title"].format(group=group, key=key, share=_show(value, seen.lang, 3))
    return yours.figure(title, svg + f'<div style="{_UI}">{w["key"]}</div>')


_OVERRIDE_WORDS = {
    "en": {"title": "Your overrides: {shown}", "xlabel": "share of the county's cases where the judge decided otherwise",
           "row": "{county}: {share}", "who": "cases of group A: {a}, of group B: {b}",
           "key": "per county your share; under it, how many of the county's cases belong to each group",
           "alt": "One bar per county, as long as your share of cases where the judge decided otherwise than the tool: {shown}."},
    "nl": {"title": "Jouw afwijkingen: {shown}", "xlabel": "deel van de zaken in het district waarin de rechter anders besliste",
           "row": "{county}: {share}", "who": "zaken van groep A: {a}, van groep B: {b}",
           "key": "per district jouw deel; daaronder hoeveel zaken van het district bij elke groep horen",
           "alt": "Eén balk per district, zo lang als jouw deel van de zaken waarin de rechter anders besliste dan de tool: {shown}."},
}


def _overrides_picture(seen) -> str:
    """A bar per county as long as the learner's share of overridden cases, with under it how many of the county's
    cases belong to group A and to group B: the counties differ, the groups within a county do not."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value or type(mine.value) is not dict or not 1 <= len(mine.value) <= 6:
        return ""
    names = mine.names if isinstance(mine.names, dict) else {}
    cases = names.get("cases")
    if not _cases_ok(cases):
        return ""
    values = {}
    for county, v in mine.value.items():
        x = _number(v)
        if not isinstance(county, str) or x is None or not 0 <= x <= 1:
            return ""
        values[county] = x
    w = _OVERRIDE_WORDS[seen.lang]
    top = 1.5 * len(values) + 0.3
    ticks = [0, 0.25, 0.5, 0.75, 1]
    p = chart.Panel(x=(0, 1.05), y=(0, top), xlabel=w["xlabel"], yticks=[], xticks=ticks,
                    xtick_labels=[_show(t, seen.lang) for t in ticks])
    for i, (county, v) in enumerate(values.items()):
        y = top - 0.3 - 1.5 * (i + 1)
        a = sum(1 for c in cases if c["county"] == county and c["group"] == "A")
        b = sum(1 for c in cases if c["county"] == county and c["group"] == "B")
        p.text(0.004, y + 1.08, w["row"].format(county=county, share=_show(v, seen.lang)), "ink", anchor="start",
               halo=True)
        p.rect(0, y + 0.52, max(v, 0.003), y + 0.82, role="data", solid=True)
        p.text(0.004, y + 0.18, w["who"].format(a=a, b=b), "ink-3", anchor="start", halo=True)
    shown = ", ".join(f"{c} {_show(v, seen.lang)}" for c, v in values.items())
    svg = chart.figure([p], alt=w["alt"].format(shown=shown), width=520, height=64 * len(values) + 90)
    svg = chart.with_decimal_comma(svg) if seen.lang == "nl" else svg
    return yours.figure(w["title"].format(shown=shown), svg + f'<div style="{_UI}">{w["key"]}</div>')


# The lesson is done when these are correct. "trap" and "first" are choice questions in the lesson.
REQUIRED = ["share", "trap", "overrides", "first"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _near(want: float):
    """A mistake that matches this number."""
    return lambda got, text: _number(got) is not None and abs(_number(got) - want) < 1e-9


def _is(want: dict):
    """A mistake that matches exactly this dict, with numbers compared closely."""
    def match(got, text):
        if type(got) is not dict or list(got) != list(want):
            return False
        return all(_number(got[k]) is not None and abs(_number(got[k]) - want[k]) < 1e-9 for k in want)
    return match


def _line(d: dict) -> str:
    return f"    {d!r},".replace("'", '"')


_CASE_LINES = [_line(c) for c in COURT]
_CASES_BLOCK = "cases = [\n" + "\n".join(_CASE_LINES) + "\n]\n"
_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

# --- exercise 1: the share released ---------------------------------------------------------------------------------
_KEY_LINE = 'key = "judge"'
_GROUP_LINE = 'group = "B"'
_S_HEAD = ("# made up by the course: cases in one court, with the tool's recommendation and the judge's decision;\n"
           "# \"release\" is release without money, \"bond\" a money bond\n" + _CASES_BLOCK +
           "# whose decisions to count, \"tool\" or \"judge\", and for which group\n" + _KEY_LINE + "\n" + _GROUP_LINE
           + "\n\n")
_S_START = _S_HEAD + "...\nshare\n"
_S_BODY = ("total = 0\nreleased = 0\nfor case in cases:\n    if case[\"group\"] == group:\n        total = total + 1\n"
           "        if case[key] == \"release\":\n            released = released + 1\nshare = released / total\n")
_S_SOLUTION = _S_HEAD + _S_BODY + "share"


def _s(body: str) -> str:
    return _S_HEAD + body + "share"


def _s_swap(old: str, new: str) -> str:
    assert _S_BODY.count(old) == 1
    return _s(_S_BODY.replace(old, new))


_WANT_SHARE = share(COURT, "B", "judge")                                     # 3 of 8
_RELEASED_B = sum(1 for c in COURT if c["group"] == "B" and c["judge"] == "release")
_ALL_CASES = _RELEASED_B / len(COURT)                                         # 3 of all 16
_EVERYONE = sum(1 for c in COURT if c["judge"] == "release") / len(COURT)     # 8 of 16
_OTHER_SIDE = share(COURT, "A", "judge")                                      # 5 of 8, and B not released
_TOOL = share(COURT, "B", "tool")                                             # the same for A

# --- exercise 3: the overrides per county ---------------------------------------------------------------------------
_COUNTIES_LINE = "counties = " + repr(COUNTIES).replace("'", '"')
_O_HEAD = ("# the same cases\n" + _CASES_BLOCK + _COUNTIES_LINE + "\n\n")
_O_START = _O_HEAD + "...\noverrides\n"
_O_BODY = ("overrides = {}\nfor county in counties:\n    total = 0\n    changed = 0\n    for case in cases:\n"
           "        if case[\"county\"] == county:\n            total = total + 1\n"
           "            if case[\"judge\"] != case[\"tool\"]:\n                changed = changed + 1\n"
           "    overrides[county] = changed / total\n")
_O_SOLUTION = _O_HEAD + _O_BODY + "overrides"


def _o(body: str) -> str:
    return _O_HEAD + body + "overrides"


def _o_swap(old: str, new: str) -> str:
    assert _O_BODY.count(old) == 1
    return _o(_O_BODY.replace(old, new))


_WANT_OVERRIDES = overrides(COURT, COUNTIES)                                 # rural 0.0, urban 0.5
_COUNTS = {"rural": 0, "urban": 4}
_OF_ALL = {"rural": 0.0, "urban": 4 / len(COURT)}
_FOLLOWED = {county: 1 - v for county, v in _WANT_OVERRIDES.items()}
_BOND = {county: sum(1 for c in COURT if c["county"] == county and c["judge"] == "bond") / 8 for county in COUNTIES}
_RELEASE = {county: 1 - v for county, v in _BOND.items()}

EXERCISES = {
    "share": CodeBox(
        name="share",
        picture=_share_picture,
        start=_S_START,
        solution=_S_SOLUTION,
        keep=["cases = ["] + _CASE_LINES + [_KEY_LINE, _GROUP_LINE],
        # The check also runs the box for the tool, and for group A.
        others=[{"key": "tool"}, {"group": "A"}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_cases`. In the box the names have none: `cases`, `key`, `group`, `case`, `total`, `released`, `share`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_cases`. In het vak hebben de namen dat niet: `cases`, `key`, `group`, `case`, `total`, `released`, `share`."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks a case for a key it does not have. Each case has `\"group\"`, `\"county\"`, `\"tool\"` and `\"judge\"`. To look up the decision that `key` names, write `case[key]`, without quotes around `key`.",
                    nl="Je code vraagt een zaak om een sleutel die hij niet heeft. Elke zaak heeft `\"group\"`, `\"county\"`, `\"tool\"` en `\"judge\"`. Om de beslissing op te zoeken die `key` noemt, schrijf je `case[key]`, zonder aanhalingstekens om `key`."),
            Mistake("zero", error=lambda e: isinstance(e, ZeroDivisionError),
                    en="Your code divides by 0: `total` was still 0 at that point. Is `share = released / total` after the loop, not inside it? And does the `if` compare `case[\"group\"]` with `group`, without quotes around `group`?",
                    nl="Je code deelt door 0: `total` was op dat moment nog 0. Staat `share = released / total` na de loop, en niet erin? En vergelijkt de `if` `case[\"group\"]` met `group`, zonder aanhalingstekens om `group`?"),
            Mistake("count", matches=_near(_RELEASED_B),
                    en="Your last line gives 3: the number of cases of group B that the judges released. A share divides that by the number of cases of the group: `share = released / total`.",
                    nl="Je laatste regel geeft 3: het aantal zaken van groep B dat de rechters vrijlieten. Een aandeel deelt dat door het aantal zaken van de groep: `share = released / total`."),
            Mistake("all-cases", matches=_near(_ALL_CASES),
                    en="Your last line gives 0.1875, 3 of all 16 cases. Divide by the cases of the group only: add 1 to `total` inside the `if` on `case[\"group\"]`.",
                    nl="Je laatste regel geeft 0,1875, 3 van alle 16 zaken. Deel alleen door de zaken van de groep: tel 1 op bij `total` binnen de `if` op `case[\"group\"]`."),
            Mistake("everyone", matches=_near(_EVERYONE),
                    en="Your last line gives 0.5, which is also the share of all 16 cases that the judges released. Do you count in `total` only the cases where `case[\"group\"] == group`, in `released` only those of them where `case[key] == \"release\"`, and divide `released / total`?",
                    nl="Je laatste regel geeft 0,5, wat ook het deel van alle 16 zaken is dat de rechters vrijlieten. Tel je in `total` alleen de zaken waarin `case[\"group\"] == group`, in `released` alleen de zaken daarvan waarin `case[key] == \"release\"`, en deel je `released / total`?"),
            Mistake("other-side", matches=_near(_OTHER_SIDE),
                    en="Your last line gives 0.625. That is the share of group A that the judges released, and also the share of group B that they did not release. Count the cases of `group` where `case[key] == \"release\"`.",
                    nl="Je laatste regel geeft 0,625. Dat is het deel van groep A dat de rechters vrijlieten, en ook het deel van groep B dat ze niet vrijlieten. Tel de zaken van `group` waarin `case[key] == \"release\"`."),
            Mistake("tool", matches=_near(_TOOL),
                    en="Your last line gives 0.75, the share that the tool recommends releasing, for group A and for group B alike. `key` says whose decisions to count, here the judge's: look them up with `case[key]`.",
                    nl="Je laatste regel geeft 0,75, het deel dat de tool wil vrijlaten, voor groep A en voor groep B. `key` zegt wiens beslissingen je telt, hier die van de rechter: zoek ze op met `case[key]`."),
            Mistake("percent", matches=_near(100 * _WANT_SHARE),
                    en="You give a percentage. The check wants a share between 0 and 1: leave out the `* 100`.",
                    nl="Je geeft een percentage. De check wil een aandeel tussen 0 en 1: laat de `* 100` weg."),
            Mistake("upside", matches=_near(1 / _WANT_SHARE),
                    en="You divide the other way round, `total / released`. The share is `released / total`.",
                    nl="Je deelt andersom, `total / released`. Het aandeel is `released / total`."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...` with your code: count the cases of `group` in `total`, and those of them where `case[key]` is `\"release\"` in `released`; then `share = released / total`.",
                      "nl": "Vervang de `...` door je code: tel de zaken van `group` in `total`, en de zaken daarvan waarin `case[key]` `\"release\"` is in `released`; dan `share = released / total`."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the lines of `cases`, `key` and `group` exactly as they were: put them back. The check tries the tool and group A by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `cases`, `key` en `group` nodig zoals ze waren: zet ze terug. De check probeert zelf de tool en groep A."},
            "no_value": {"en": "The last line of the box shows nothing. Put `share` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `share` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you count in `total` the cases with `case[\"group\"] == group`, in `released` those of them with `case[key] == \"release\"`, and divide `released / total`?",
                      "nl": "Je laatste regel geeft `{got}`. Tel je in `total` de zaken met `case[\"group\"] == group`, in `released` de zaken daarvan met `case[key] == \"release\"`, en deel je `released / total`?"},
            "other": {"en": "Right for group B and the judges. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `key` and `group` themselves, and not `\"judge\"` and `\"B\"`?",
                      "nl": "Goed voor groep B en de rechters. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `key` en `group` zelf, en niet `\"judge\"` en `\"B\"`?"},
            "other_error": {"en": "Right for group B and the judges. But with {where}, your code stops with an error. Do you use `key` and `group` themselves?",
                            "nl": "Goed voor groep B en de rechters. Maar met {where} stopt je code met een fout. Gebruik je `key` en `group` zelf?"},
            "correct": {"en": "`0.375`: the judges released 3 of the 8 cases of group B. With `key = \"tool\"` you get 0.75, and for group A too: the tool recommends releasing both groups alike. The judges released 0.625 of group A. The gap of 0.25 is not in the tool's recommendations; it is in what people did with them. A fairness check on the tool alone would have found nothing: the framing trap.",
                        "nl": "`0.375`: de rechters lieten 3 van de 8 zaken van groep B vrij. Met `key = \"tool\"` krijg je 0,75, en voor groep A ook: de tool wil beide groepen even vaak vrijlaten. De rechters lieten 0,625 van groep A vrij. Het verschil van 0,25 zit niet in de aanbevelingen van de tool, maar in wat mensen ermee deden. Een fairness-check op alleen de tool had niets gevonden: de framing trap."},
        },
    ),
    "overrides": CodeBox(
        name="overrides",
        picture=_overrides_picture,
        start=_O_START,
        solution=_O_SOLUTION,
        keep=["cases = ["] + _CASE_LINES + [_COUNTIES_LINE],
        # The check also runs the box with another court.
        others=[{"cases": OTHER_COURT}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_cases`. In the box the names have none: `cases`, `counties`, `county`, `case`, `total`, `changed`, `overrides`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_cases`. In het vak hebben de namen dat niet: `cases`, `counties`, `county`, `case`, `total`, `changed`, `overrides`."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks a case or `overrides` for a key it does not have. Each case has `\"group\"`, `\"county\"`, `\"tool\"` and `\"judge\"`; put a county in `overrides` with `overrides[county] = ...`.",
                    nl="Je code vraagt een zaak of `overrides` om een sleutel die er niet in zit. Elke zaak heeft `\"group\"`, `\"county\"`, `\"tool\"` en `\"judge\"`; zet een district in `overrides` met `overrides[county] = ...`."),
            Mistake("zero", error=lambda e: isinstance(e, ZeroDivisionError),
                    en="Your code divides by 0: `total` was still 0 at that point. Is `overrides[county] = changed / total` after the loop over the cases, not inside it? And does the `if` compare `case[\"county\"]` with `county`, without quotes around `county`?",
                    nl="Je code deelt door 0: `total` was op dat moment nog 0. Staat `overrides[county] = changed / total` na de loop over de zaken, en niet erin? En vergelijkt de `if` `case[\"county\"]` met `county`, zonder aanhalingstekens om `county`?"),
            Mistake("counts", matches=_is(_COUNTS),
                    en="Your dict holds counts: 0 and 4 cases where the judge decided otherwise. The check wants shares: divide by the number of cases in the county, `changed / total`.",
                    nl="Je dict bevat aantallen: 0 en 4 zaken waarin de rechter anders besliste. De check wil aandelen: deel door het aantal zaken in het district, `changed / total`."),
            Mistake("of-all", matches=_is(_OF_ALL),
                    en="For `urban` your dict gives 0.25, 4 of all 16 cases. Divide by the cases of that county only: set `total = 0` and `changed = 0` at the top of the loop over the counties, and count only the cases with `case[\"county\"] == county`.",
                    nl="Voor `urban` geeft je dict 0,25, 4 van alle 16 zaken. Deel alleen door de zaken van dat district: zet `total = 0` en `changed = 0` bovenaan in de loop over de districten, en tel alleen de zaken met `case[\"county\"] == county`."),
            Mistake("followed", matches=_is(_FOLLOWED),
                    en="Your dict holds the shares where the judge *followed* the tool. The check wants the cases where the judge decided otherwise: `case[\"judge\"] != case[\"tool\"]`.",
                    nl="Je dict bevat de aandelen waarin de rechter de tool *volgde*. De check wil de zaken waarin de rechter anders besliste: `case[\"judge\"] != case[\"tool\"]`."),
            Mistake("judge-only", matches=lambda got, text: _is(_BOND)(got, text) or _is(_RELEASE)(got, text),
                    en="Your dict holds a share of the judges' decisions on their own, release or bond. Deciding otherwise compares two decisions: count the cases where `case[\"judge\"] != case[\"tool\"]`.",
                    nl="Je dict bevat een deel van alleen de beslissingen van de rechters, vrijlating of borgsom. Anders beslissen vergelijkt twee beslissingen: tel de zaken waarin `case[\"judge\"] != case[\"tool\"]`."),
            Mistake("percent", matches=_is({c: 100 * v for c, v in _WANT_OVERRIDES.items()}),
                    en="You give percentages. The check wants shares between 0 and 1: leave out the `* 100`.",
                    nl="Je geeft percentages. De check wil aandelen tussen 0 en 1: laat de `* 100` weg."),
            Mistake("last-only", matches=lambda got, text: type(got) is dict and list(got) == ["urban"],
                    en="Your dict only holds `urban`, the last county. Is `overrides[county] = changed / total` inside the loop over the counties, and do you make `overrides = {{}}` once, before that loop?",
                    nl="Je dict bevat alleen `urban`, het laatste district. Staat `overrides[county] = changed / total` binnen de loop over de districten, en maak je `overrides = {{}}` één keer, vóór die loop?"),
            Mistake("list", matches=lambda got, text: isinstance(got, list),
                    en="Your last line gives a list. The check wants a dict from each county to its share: `overrides[county] = changed / total`.",
                    nl="Je laatste regel geeft een lijst. De check wil een dict van elk district naar zijn aandeel: `overrides[county] = changed / total`."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...` with your code: start with `overrides = {{}}`, and for each county count its cases in `total` and those where the judge decided otherwise than the tool in `changed`; then `overrides[county] = changed / total`.",
                      "nl": "Vervang de `...` door je code: begin met `overrides = {{}}`, en tel voor elk district de zaken in `total` en de zaken waarin de rechter anders besliste dan de tool in `changed`; dan `overrides[county] = changed / total`."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the lines of `cases` and `counties` exactly as they were: put them back. The check tries another court by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `cases` en `counties` nodig zoals ze waren: zet ze terug. De check probeert zelf een andere rechtbank."},
            "no_value": {"en": "The last line of the box shows nothing. Put `overrides` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `overrides` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. For each county, do you set `total` and `changed` to 0, count its cases and those with `case[\"judge\"] != case[\"tool\"]`, and store `changed / total` under the county?",
                      "nl": "Je laatste regel geeft `{got}`. Zet je voor elk district `total` en `changed` op 0, tel je de zaken ervan en die met `case[\"judge\"] != case[\"tool\"]`, en zet je `changed / total` onder het district?"},
            "other": {"en": "Right for these cases. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you count from `cases` itself, and not write the shares down?",
                      "nl": "Goed voor deze zaken. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Tel je vanuit `cases` zelf, en schrijf je de aandelen niet over?"},
            "other_error": {"en": "Right for these cases. But with {where}, your code stops with an error. Do you count from `cases` itself?",
                            "nl": "Goed voor deze zaken. Maar met {where} stopt je code met een fout. Tel je vanuit `cases` zelf?"},
            "correct": {"en": "`{{'rural': 0.0, 'urban': 0.5}}`: the rural judges always followed the tool, the urban judges decided otherwise in half the cases, and within each county as often for group A as for group B. But 6 of the 8 cases of group A are rural, and 6 of the 8 of group B urban: that is where the gap of exercise 1 comes from. Stevenson found this pattern in Kentucky: the racial gap grew because judges in different counties responded differently to the law, while within the same county White and Black defendants saw similar increases in release.",
                        "nl": "`{{'rural': 0.0, 'urban': 0.5}}`: de rechters op het platteland volgden de tool altijd, die in de stad besloten in de helft van de zaken anders, en binnen elk district even vaak voor groep A als voor groep B. Maar 6 van de 8 zaken van groep A liggen op het platteland, en 6 van de 8 van groep B in de stad: daar komt het verschil uit oefening 1 vandaan. Stevenson vond dit patroon in Kentucky: het verschil naar ras groeide doordat rechters in verschillende districten verschillend op de wet reageerden, terwijl binnen hetzelfde district witte en zwarte verdachten een vergelijkbare toename van vrijlating zagen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("share", _S_START, "empty", None),
    ("share", _S_SOLUTION, "correct", None),
    ("share", _s("mine = []\nfor case in cases:\n    if case[\"group\"] == group:\n        mine.append(case)\n"
                 "released = 0\nfor case in mine:\n    if case[key] == \"release\":\n        released = released + 1\n"
                 "share = released / len(mine)\n"), "correct", None),
    ("share", _s(_S_BODY.replace("in cases:", "in _cases:")), "error", "underscore"),
    ("share", _s_swap("case[key] == \"release\"", "case[\"key\"] == \"release\""), "error", "key"),
    ("share", _s_swap("case[\"group\"] == group", "case[\"group\"] == \"group\""), "error", "zero"),
    ("share", _s_swap("\nshare = released / total", "\n    share = released / total"), "error", "zero"),
    ("share", _s_swap("share = released / total", "share = total / len(cases)"), "wrong", "everyone"),
    ("share", _s_swap("share = released / total", "share = released"), "wrong", "count"),
    ("share", _s_swap("share = released / total", "share = released / len(cases)"), "wrong", "all-cases"),
    ("share", _s("total = 0\nreleased = 0\nfor case in cases:\n    total = total + 1\n"
                 "    if case[key] == \"release\":\n        released = released + 1\nshare = released / total\n"),
     "wrong", "everyone"),
    ("share", _s_swap("case[key] == \"release\"", "case[key] == \"bond\""), "wrong", "other-side"),
    ("share", _s_swap("case[key] == \"release\"", "case[\"tool\"] == \"release\""), "wrong", "tool"),
    ("share", _s_swap("share = released / total", "share = released / total * 100"), "wrong", "percent"),
    ("share", _s_swap("share = released / total", "share = total / released"), "wrong", "upside"),
    ("share", _s(_S_BODY + "share = share + 0.1\n"), "wrong", None),
    ("share", _s("share = 0.375\n"), "almost", "other-value"),
    ("share", _s_swap("case[key] == \"release\"", "case[\"judge\"] == \"release\""), "almost", "other-value"),
    ("share", _S_SOLUTION.replace('group = "B"', 'group = "A"'), "wrong", "changed-line"),
    ("share", _s("...\nshare = 0.375\n"), "wrong", "dots-left"),
    ("overrides", _O_START, "empty", None),
    ("overrides", _O_SOLUTION, "correct", None),
    ("overrides", _o("overrides = {}\nfor county in counties:\n    mine = []\n    for case in cases:\n"
                     "        if case[\"county\"] == county:\n            mine.append(case)\n    changed = 0\n"
                     "    for case in mine:\n        if case[\"judge\"] != case[\"tool\"]:\n            changed = changed + 1\n"
                     "    overrides[county] = changed / len(mine)\n"), "correct", None),
    ("overrides", _o(_O_BODY.replace("in cases:", "in _cases:")), "error", "underscore"),
    ("overrides", _o_swap("case[\"judge\"] != case[\"tool\"]", "case[\"Judge\"] != case[\"tool\"]"), "error", "key"),
    ("overrides", _o_swap("case[\"county\"] == county", "case[\"county\"] == \"county\""), "error", "zero"),
    ("overrides", _o_swap("    overrides[county] = changed / total", "        overrides[county] = changed / total"),
     "error", "zero"),
    ("overrides", _o_swap("    overrides[county] = changed / total", "overrides[county] = changed / total"),
     "wrong", "last-only"),
    ("overrides", _o_swap("overrides[county] = changed / total", "overrides[county] = changed"), "wrong", "counts"),
    ("overrides", _o("overrides = {}\ntotal = 0\nchanged = 0\nfor county in counties:\n    for case in cases:\n"
                     "        if case[\"county\"] == county:\n            total = total + 1\n"
                     "            if case[\"judge\"] != case[\"tool\"]:\n                changed = changed + 1\n"
                     "    overrides[county] = changed / total\n"), "wrong", "of-all"),
    ("overrides", _o_swap("overrides[county] = changed / total", "overrides[county] = changed / len(cases)"),
     "wrong", "of-all"),
    ("overrides", _o_swap("case[\"judge\"] != case[\"tool\"]", "case[\"judge\"] == case[\"tool\"]"), "wrong", "followed"),
    ("overrides", _o_swap("case[\"judge\"] != case[\"tool\"]", "case[\"judge\"] == \"bond\""), "wrong", "judge-only"),
    ("overrides", _o_swap("case[\"judge\"] != case[\"tool\"]", "case[\"judge\"] == \"release\""), "wrong", "judge-only"),
    ("overrides", _o_swap("overrides[county] = changed / total", "overrides[county] = changed / total * 100"),
     "wrong", "percent"),
    ("overrides", _o(_O_BODY.replace("overrides = {}\nfor county in counties:\n",
                                     "for county in counties:\n    overrides = {}\n")), "wrong", "last-only"),
    ("overrides", _o(_O_BODY.replace("overrides = {}\n", "overrides = []\n").replace(
        "overrides[county] = changed / total", "overrides.append(changed / total)")), "wrong", "list"),
    ("overrides", _o(_O_BODY + "overrides[\"urban\"] = 0.4\n"), "wrong", None),
    ("overrides", _o("overrides = {\"rural\": 0.0, \"urban\": 0.5}\n"), "almost", "other-value"),
    ("overrides", _O_SOLUTION.replace('counties = ["rural", "urban"]', 'counties = ["urban"]'), "wrong", "changed-line"),
    ("overrides", _o("...\noverrides = {\"rural\": 0.0, \"urban\": 0.5}\n"), "wrong", "dots-left"),
]
