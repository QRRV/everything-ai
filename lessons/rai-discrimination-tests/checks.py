# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-discrimination-tests: how many times as often equal applications with a native Dutch name
succeeded in a correspondence study of the SCP, and the hit rate per group of an officer who uses one bar for made-up
drivers; with the picture of the demo's searches and hit rates, and the pictures of the learner's ratios and hit
rates."""
import math
import numbers
import re

from everything_ai.check import CodeBox, Mistake
from everything_ai import yours

# The demo: made-up drivers, per group a chance of carrying something and how many drivers have it, and two ways to
# set the bar of each group.
DRIVERS = {"group A": {0.01: 70, 0.2: 20, 0.6: 10}, "group B": {0.01: 70, 0.2: 20, 0.6: 10}}
WAYS = {"same": {"group A": 0.5, "group B": 0.5}, "lower": {"group A": 0.5, "group B": 0.1}}

# The worked example: Bertrand and Mullainathan (2004), 2,435 CVs per kind of name; the calls back are worked out
# from their 9.65 and 6.45 percent.
SENT = 2435
CALLED = {"White names": 235, "African-American names": 157}

# Exercise 1: the chance of success per name in the SCP's study in the region of The Hague (2014).
SUCCESS = {"native Dutch": 0.34, "Hindustani": 0.23, "Moroccan": 0.19}
FIRST = "native Dutch"

# Exercise 2: made-up counts with the chances of the example of Simoiu, Corbett-Davies and Goel (2017).
HIT_DRIVERS = {"group A": {0.01: 90, 0.75: 10}, "group B": {0.01: 80, 0.5: 20}}
BAR = 0.1


def search(drivers: dict, bars: dict) -> dict:
    """Per group: how many drivers have a chance of at least the group's bar, and how many finds that gives on
    average, as (searched, found)."""
    out = {}
    for group, kinds in drivers.items():
        searched = found = 0
        for chance, count in kinds.items():
            if chance >= bars[group]:
                searched += count
                found += chance * count
        out[group] = (searched, found)
    return out


def ratios(success: dict, first: str = FIRST) -> dict:
    """Per group: how many times as often the first group succeeds."""
    return {g: success[first] / s for g, s in success.items()}


def hit_rates(drivers: dict, bar: float) -> dict:
    """Per group: the share of the searches that finds something, with one bar for everyone."""
    return {g: found / searched for g, (searched, found) in search(drivers, {g: bar for g in drivers}).items()}


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"


def _num(x: float, decimal: str) -> str:
    return f"{x:.2f}".replace(".", decimal)


def picture_hits(labels: dict) -> str:
    """Two panels with a row per way of WAYS: the share of the drivers searched (left) and the hit rate (right), for
    group A (open bars) and group B (filled bars). `labels` gives "title", "rows" (two names), "xlabels" (two), "key",
    "decimal" and "alt"."""
    from everything_ai import chart
    keys = list(WAYS)
    top = 1.6 * len(keys) + 0.1
    panels = []
    for k, xlabel in enumerate(labels["xlabels"]):
        p = chart.Panel(x=(0, 1), y=(0, top), xlabel=xlabel, yticks=[], xticks=[0, 0.5, 1],
                        xtick_labels=["0", "0.5", "1"])
        for i, (key, name) in enumerate(zip(keys, labels["rows"])):
            done = search(DRIVERS, WAYS[key])
            values = []
            for group in DRIVERS:
                searched, found = done[group]
                values.append(searched / sum(DRIVERS[group].values()) if k == 0 else found / searched)
            y = top - 1.6 * (i + 1)
            if k == 0:
                p.text(0.002, y + 1.2, name, "ink", anchor="start")
            p.rect(0, y + 0.55, values[0], y + 0.85, role="data")
            p.rect(0, y + 0.15, values[1], y + 0.45, role="data", solid=True)
            p.text(values[0] + 0.015, y + 0.7, _num(values[0], labels["decimal"]), "ink-2", anchor="start")
            p.text(values[1] + 0.015, y + 0.3, _num(values[1], labels["decimal"]), "ink-2", anchor="start")
        panels.append(p)
    svg = chart.figure(panels, alt=labels["alt"], width=300, height=230)
    if labels["decimal"] == ",":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(labels["title"], svg + f'<div style="{_UI}">{labels["key"]}</div>')


def _number(x) -> float | None:
    """A real number that is not True or False, as a float; else None."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return None
    return float(x)


def _shares(v, low: float = 0, high: float = 1) -> dict | None:
    """The learner's number per group, when it is a dict of 1 to 6 numbers from `low` to `high`."""
    if not isinstance(v, dict) or not 1 <= len(v) <= 6:
        return None
    out = {}
    for k, x in v.items():
        x = _number(x)
        if not isinstance(k, str) or x is None or not math.isfinite(x) or not low <= x <= high:
            return None
        out[k] = x
    return out


def _close(got, want) -> bool:
    """A number per group, the same groups, each the same to two decimals."""
    a, b = _shares(got, 0, 1000), _shares(want, 0, 1000)
    return (a is not None and b is not None and set(a) == set(b)
            and all(abs(a[k] - b[k]) < 0.006 for k in a))


def _like(make, keys):
    """A mistake where every group's number is what `make()` gives, to two decimals."""
    def match(got, text):
        if not isinstance(got, dict) or set(got) != set(keys):
            return False
        want = make()
        return all(_number(got[k]) is not None and abs(got[k] - want[k]) < 0.006 for k in keys)
    return match


def _rounded_once(got, text) -> bool:
    """The ratios of the SCP, each rounded to one decimal."""
    want = ratios(SUCCESS)
    return (isinstance(got, dict) and set(got) == set(want)
            and all(_number(got[k]) is not None and abs(got[k] - round(want[k], 1)) < 1e-9 for k in want))


def _fewer(keys):
    def match(got, text):
        return isinstance(got, dict) and 1 <= len(got) < len(keys) and set(got) <= set(keys)
    return match


def _other(keys):
    def match(got, text):
        return isinstance(got, dict) and len(got) >= 1 and not set(got) <= set(keys)
    return match


def _above(limit: float):
    def match(got, text):
        return isinstance(got, dict) and any(_number(x) is not None and x > limit for x in got.values())
    return match


def _all_one(got, text) -> bool:
    return (isinstance(got, dict) and set(got) == set(HIT_DRIVERS)
            and all(_number(x) is not None and abs(x - 1) < 1e-9 for x in got.values()))


_RATIO_WORDS = {
    "en": {"title": "Your ratios: how many times as often native Dutch applicants succeed",
           "xlabel": "times as often", "same": "1: the same chance",
           "alt": "One bar per group, as long as its ratio, with a dashed line at 1: {shown}."},
    "nl": {"title": "Jouw verhoudingen: hoeveel keer zo vaak autochtoon Nederlandse sollicitanten slagen",
           "xlabel": "keer zo vaak", "same": "1: dezelfde kans",
           "alt": "Eén balk per groep, zo lang als de verhouding, met een gestippelde lijn bij 1: {shown}."},
}


def _ratio_picture(seen) -> str:
    """Per group a bar as long as the learner's ratio, with a dashed line at 1, where a group succeeds as often as
    the first."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    names = mine.names if isinstance(mine.names, dict) else {}
    value = _shares(names.get("ratio"), 0, 10)
    if value is None:
        return ""
    w = _RATIO_WORDS[seen.lang]
    sign = "," if seen.lang == "nl" else "."
    keys = list(value)
    xmax = 2 if max(value.values()) <= 1.9 else math.ceil(max(value.values()) + 0.1)
    ticks = [t / 2 for t in range(0, 2 * xmax + 1)]
    top = 1.2 * len(keys)
    panel = chart.Panel(x=(0, xmax), y=(0, top + 0.6), xlabel=w["xlabel"], yticks=[], xticks=ticks,
                        xtick_labels=[f"{t:g}" for t in ticks])
    panel.line([1, 1], [0, top + 0.6], role="ink-3", width=1, dashed=True)
    panel.text(1 + 0.02 * xmax, top + 0.3, w["same"], "ink-3", anchor="start", halo=True)
    for i, k in enumerate(keys):
        y = top - 1.2 * (i + 1)
        panel.text(0.005 * xmax, y + 0.72, k + ": " + _num(value[k], sign), "ink", anchor="start", halo=True)   # Panel.text escapes it
        panel.rect(0, y + 0.15, value[k], y + 0.45, role="data", solid=True)
    shown = "; ".join(f"{k} {_num(value[k], sign)}" for k in keys)
    svg = chart.figure([panel], alt=w["alt"].format(shown=shown), width=520, height=max(170, 44 * len(keys) + 90))
    return yours.figure(w["title"], chart.with_decimal_comma(svg) if seen.lang == "nl" else svg)


_HITS_WORDS = {
    "en": {"title": "Your hit rates with one bar for everyone", "xlabel": "hit rate: the share of searches that finds something",
           "bar": "the bar {bar}",
           "alt": "One bar per group, as long as its hit rate, with a dashed line at the bar of {bar}: {shown}."},
    "nl": {"title": "Jouw hit rates met één lat voor iedereen", "xlabel": "hit rate: het deel van de doorzoekingen dat iets vindt",
           "bar": "de lat {bar}",
           "alt": "Eén balk per groep, zo lang als de hit rate, met een gestippelde lijn bij de lat van {bar}: {shown}."},
}


def _hits_picture(seen) -> str:
    """Per group a bar as long as the learner's hit rate, with a dashed line at the bar the officer uses for
    everyone."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    names = mine.names if isinstance(mine.names, dict) else {}
    value = _shares(names.get("hit_rate"))
    bar = _number(names.get("bar"))
    if value is None or bar is None or not 0 <= bar <= 1:
        return ""
    w = _HITS_WORDS[seen.lang]
    sign = "," if seen.lang == "nl" else "."
    keys = list(value)
    top = 1.2 * len(keys)
    panel = chart.Panel(x=(0, 1), y=(0, top + 0.6), xlabel=w["xlabel"], yticks=[], xticks=[0, 0.25, 0.5, 0.75, 1],
                        xtick_labels=["0", "0.25", "0.5", "0.75", "1"])
    panel.line([bar, bar], [0, top + 0.6], role="ink-3", width=1, dashed=True)
    shown_bar = f"{bar:g}".replace(".", sign)
    panel.text(bar + 0.01, top + 0.3, w["bar"].format(bar=shown_bar), "ink-3", anchor="start", halo=True)
    for i, k in enumerate(keys):
        y = top - 1.2 * (i + 1)
        panel.text(0.005, y + 0.72, k + ": " + _num(value[k], sign), "ink", anchor="start", halo=True)   # Panel.text escapes it
        panel.rect(0, y + 0.15, value[k], y + 0.45, role="data", solid=True)
    shown = "; ".join(f"{k} {_num(value[k], sign)}" for k in keys)
    svg = chart.figure([panel], alt=w["alt"].format(bar=shown_bar, shown=shown), width=520,
                       height=max(170, 44 * len(keys) + 90))
    return yours.figure(w["title"], chart.with_decimal_comma(svg) if seen.lang == "nl" else svg)


# The lesson is done when these are correct.
REQUIRED = ["ratio", "hits", "labels"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

_SUCCESS_LINE = 'success = {"native Dutch": 0.34, "Hindustani": 0.23, "Moroccan": 0.19}'
_RATIO_HEAD = ("# the SCP, region of The Hague, 2014: the chance of success of equal applications, per name\n"
               + _SUCCESS_LINE + "\n\nratio = {}\nfor group in success:\n")
_RATIO_START = _RATIO_HEAD + "    ...\nratio\n"
_RATIO_BODY = '    ratio[group] = success["native Dutch"] / success[group]\n'
_RATIO_SOLUTION = _RATIO_HEAD + _RATIO_BODY + "ratio"


def _r(body: str, last: str = "ratio") -> str:
    return _RATIO_HEAD + body + last


_DRIVERS_LINE = 'drivers = {"group A": {0.01: 90, 0.75: 10}, "group B": {0.01: 80, 0.5: 20}}'
_BAR_LINE = f"bar = {BAR}"
_HITS_HEAD = ("# made-up drivers: per group, a chance of carrying something and how many drivers have it\n"
              + _DRIVERS_LINE + "\n# one bar for everyone\n" + _BAR_LINE + "\n\n"
              "hit_rate = {}\nfor group in drivers:\n    searched = 0\n    found = 0\n"
              "    for chance, count in drivers[group].items():\n")
_HITS_TAIL = "    hit_rate[group] = found / searched\nhit_rate"
_HITS_START = _HITS_HEAD + "        ...\n" + _HITS_TAIL + "\n"
_HITS_BODY = ("        if chance >= bar:\n"
              "            searched = searched + count\n"
              "            found = found + chance * count\n")
_HITS_SOLUTION = _HITS_HEAD + _HITS_BODY + _HITS_TAIL


def _h(body: str, tail: str = _HITS_TAIL) -> str:
    return _HITS_HEAD + body + tail


def _hswap(old: str, new: str) -> str:
    assert _HITS_BODY.count(old) == 1, old
    return _h(_HITS_BODY.replace(old, new))


def _inverted():
    return {g: s / SUCCESS[FIRST] for g, s in SUCCESS.items()}


def _difference():
    return {g: SUCCESS[FIRST] - s for g, s in SUCCESS.items()}


def _no_if():
    return {g: sum(c * n for c, n in kinds.items()) / sum(kinds.values()) for g, kinds in HIT_DRIVERS.items()}


def _below():
    out = {}
    for g, kinds in HIT_DRIVERS.items():
        low = {c: n for c, n in kinds.items() if c < BAR}
        out[g] = sum(c * n for c, n in low.items()) / sum(low.values())
    return out


def _chance_once():
    out = {}
    for g, kinds in HIT_DRIVERS.items():
        high = {c: n for c, n in kinds.items() if c >= BAR}
        out[g] = sum(high) / sum(high.values())
    return out


EXERCISES = {
    "ratio": CodeBox(
        name="ratio",
        picture=_ratio_picture,
        start=_RATIO_START,
        solution=_RATIO_SOLUTION,
        keep=[_SUCCESS_LINE],
        compare=_close,
        digits=2,
        # The check also runs the box with other chances of success.
        others=[{"success": {"native Dutch": 0.5, "Hindustani": 0.25, "Moroccan": 0.4}}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_ratio`. In the box the names have none: `success`, `ratio`, `group`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_ratio`. In het vak hebben de namen dat niet: `success`, `ratio`, `group`."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks `success` for a key it does not have. The first group is `\"native Dutch\"`, written exactly like that; `group` itself goes without quotes.",
                    nl="Je code vraagt `success` om een sleutel die het niet heeft. De eerste groep is `\"native Dutch\"`, precies zo geschreven; `group` zelf staat zonder aanhalingstekens."),
            Mistake("one-number", matches=lambda got, text: _number(got) is not None,
                    en="Your last line gives one number, not one per group: write `ratio[group] = ...`, so that `ratio` gets a number for each group.",
                    nl="Je laatste regel geeft één getal, niet een getal per groep: schrijf `ratio[group] = ...`, zodat `ratio` voor elke groep een getal krijgt."),
            Mistake("rounded", matches=_rounded_once,
                    en="Your ratios are rounded to one decimal, like the \"1.5 times\" of the SCP. The check compares two decimals: round to 2 decimals, or do not round at all.",
                    nl="Je verhoudingen zijn afgerond op één decimaal, zoals de 1,5 keer van het SCP. De check vergelijkt op twee decimalen: rond af op 2 decimalen, of rond helemaal niet af."),
            Mistake("percent", matches=_above(10),
                    en="Your ratios are far too large. A ratio is not a percentage: divide the two chances, and do not multiply by 100.",
                    nl="Je verhoudingen zijn veel te groot. Een verhouding is geen percentage: deel de twee kansen, en vermenigvuldig niet met 100."),
            Mistake("inverted", matches=_like(_inverted, SUCCESS),
                    en="Your ratios are 1 or below: you divided the chance of each group by that of native Dutch applicants. The question is how many times as often native Dutch applicants succeed: `success[\"native Dutch\"] / success[group]`.",
                    nl="Je verhoudingen zijn 1 of kleiner: je deelde de kans van elke groep door die van autochtoon Nederlandse sollicitanten. De vraag is hoeveel keer zo vaak autochtoon Nederlandse sollicitanten slagen: `success[\"native Dutch\"] / success[group]`."),
            Mistake("difference", matches=_like(_difference, SUCCESS),
                    en="You subtracted the two chances. The SCP gives how many times as often: divide them.",
                    nl="Je trok de twee kansen van elkaar af. Het SCP geeft hoeveel keer zo vaak: deel ze door elkaar."),
            Mistake("one-group", matches=_fewer(SUCCESS),
                    en="Your `ratio` has fewer groups than `success`. Does the line `ratio[group] = ...` stand inside the loop, indented by four spaces?",
                    nl="Je `ratio` heeft minder groepen dan `success`. Staat de regel `ratio[group] = ...` in de loop, vier spaties ingesprongen?"),
            Mistake("other-keys", matches=_other(SUCCESS),
                    en="Your `ratio` has other keys than `success`. Use `group` itself as the key, without quotes: `ratio[group] = ...`.",
                    nl="Je `ratio` heeft andere sleutels dan `success`. Gebruik `group` zelf als sleutel, zonder aanhalingstekens: `ratio[group] = ...`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` in the loop with one line: `ratio[group]`, the chance of native Dutch applicants divided by the chance of `group`.",
                      "nl": "Vervang de `...` in de loop door één regel: `ratio[group]`, de kans van autochtoon Nederlandse sollicitanten gedeeld door de kans van `group`."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the line of `success` exactly as it was: put it back. The check tries other numbers by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel van `success` nodig zoals hij was: zet hem terug. De check probeert zelf andere getallen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `ratio` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `ratio` op de laatste regel, zodat de check het ziet."},
            "wrong": {"en": "Your last line gives `{got}`. For each group: is it the chance of native Dutch applicants divided by the chance of that group?",
                      "nl": "Je laatste regel geeft `{got}`. Is het voor elke groep de kans van autochtoon Nederlandse sollicitanten gedeeld door de kans van die groep?"},
            "other": {"en": "Right for these numbers. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `success` itself, and not its numbers?",
                      "nl": "Goed voor deze getallen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `success` zelf, en niet de getallen?"},
            "other_error": {"en": "Right for these numbers. But with {where}, your code stops with an error. Do you use `success` and `group` themselves?",
                            "nl": "Goed voor deze getallen. Maar met {where} stopt je code met een fout. Gebruik je `success` en `group` zelf?"},
            "correct": {"en": "Native Dutch applicants succeeded 1.48 times as often as applicants with a Hindustani name, and 1.79 times as often as applicants with a Moroccan name: the \"1.5 times\" and \"1.8 times\" of the SCP. The applications differed only in the name, so the difference comes from the name.",
                        "nl": "Autochtoon Nederlandse sollicitanten slaagden 1,48 keer zo vaak als sollicitanten met een Hindoestaanse naam, en 1,79 keer zo vaak als sollicitanten met een Marokkaanse naam: de 1,5 en 1,8 keer van het SCP. De sollicitaties verschilden alleen in de naam, dus het verschil komt door de naam."},
        },
    ),
    "hits": CodeBox(
        name="hits",
        picture=_hits_picture,
        start=_HITS_START,
        solution=_HITS_SOLUTION,
        keep=[_DRIVERS_LINE, _BAR_LINE],
        compare=_close,
        digits=2,
        # The check also runs the box with a bar so low that everyone is searched.
        others=[{"bar": 0.005}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_found`. In the box the names have none: `drivers`, `bar`, `hit_rate`, `group`, `searched`, `found`, `chance`, `count`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_found`. In het vak hebben de namen dat niet: `drivers`, `bar`, `hit_rate`, `group`, `searched`, `found`, `chance`, `count`."),
            Mistake("bars", error=lambda e: isinstance(e, NameError) and "'bars'" in str(e),
                    en="Your code uses `bars[group]` from *Try it*. Here every group has the same bar, `bar`: `if chance >= bar:`.",
                    nl="Je code gebruikt `bars[group]` uit *Probeer het*. Hier heeft elke groep dezelfde lat, `bar`: `if chance >= bar:`."),
            Mistake("nobody", error=lambda e: isinstance(e, ZeroDivisionError),
                    en="Your code divides by 0: `searched` stays 0 in a group. Inside the `if`, add the drivers of that kind to it: `searched = searched + count`.",
                    nl="Je code deelt door 0: `searched` blijft in een groep 0. Tel er binnen de `if` de bestuurders van die soort bij op: `searched = searched + count`."),
            Mistake("all-found", matches=_all_one,
                    en="Your hit rates are 1: `found` counts every searched driver as a find. A driver with a chance of 0.75 is 0.75 of a find on average: `found = found + chance * count`.",
                    nl="Je hit rates zijn 1: `found` telt elke doorzochte bestuurder als een vondst. Een bestuurder met een kans van 0,75 is gemiddeld 0,75 vondst: `found = found + chance * count`."),
            Mistake("kinds", matches=_above(1),
                    en="A hit rate above 1 cannot be: `searched` counts the kinds of driver, not the drivers. Add `count`: `searched = searched + count`.",
                    nl="Een hit rate boven 1 kan niet: `searched` telt de soorten bestuurders, niet de bestuurders. Tel `count` erbij op: `searched = searched + count`."),
            Mistake("no-if", matches=_like(_no_if, HIT_DRIVERS),
                    en="Your hit rates count every driver, also the ones below the bar. The officer searches only the drivers whose chance is at least `bar`: put the two lines under `if chance >= bar:`.",
                    nl="Je hit rates tellen elke bestuurder, ook die onder de lat. De agent doorzoekt alleen de bestuurders met een kans van minstens `bar`: zet de twee regels onder `if chance >= bar:`."),
            Mistake("below", matches=_like(_below, HIT_DRIVERS),
                    en="You searched the drivers below the bar. The officer searches the drivers whose chance is at least `bar`: `if chance >= bar:`.",
                    nl="Je doorzocht de bestuurders onder de lat. De agent doorzoekt de bestuurders met een kans van minstens `bar`: `if chance >= bar:`."),
            Mistake("chance-once", matches=_like(_chance_once, HIT_DRIVERS),
                    en="Your `found` adds the chance once per kind of driver, not once per driver: multiply it by `count`.",
                    nl="Je `found` telt de kans één keer per soort bestuurder op, niet één keer per bestuurder: vermenigvuldig hem met `count`."),
            Mistake("one-group", matches=_fewer(HIT_DRIVERS),
                    en="Your `hit_rate` has fewer groups than `drivers`. Does the line `hit_rate[group] = ...` still stand inside the outer loop, indented by four spaces?",
                    nl="Je `hit_rate` heeft minder groepen dan `drivers`. Staat de regel `hit_rate[group] = ...` nog in de buitenste loop, vier spaties ingesprongen?"),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` in the inner loop with an `if` for the drivers whose chance is at least `bar`, and under it the two lines that add to `searched` and `found`, as in *Try it*.",
                      "nl": "Vervang de `...` in de binnenste loop door een `if` voor de bestuurders met een kans van minstens `bar`, en daaronder de twee regels die `searched` en `found` ophogen, zoals in *Probeer het*."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the lines of `drivers` and `bar` exactly as they were: put them back. The check tries another bar by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `drivers` en `bar` nodig zoals ze waren: zet ze terug. De check probeert zelf een andere lat."},
            "no_value": {"en": "The last line of the box shows nothing. Put `hit_rate` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `hit_rate` op de laatste regel, zodat de check het ziet."},
            "wrong": {"en": "Your last line gives `{got}`. For each kind of driver with a chance of at least `bar`: do you add `count` to `searched`, and `chance * count` to `found`?",
                      "nl": "Je laatste regel geeft `{got}`. Tel je bij elke soort bestuurder met een kans van minstens `bar` het getal `count` op bij `searched`, en `chance * count` bij `found`?"},
            "other": {"en": "Right for these numbers, where only one kind of driver per group is at or above the bar. But with {where}, every kind is searched, and your last line gives `{got}` where it should give `{want}`. Do you compare with `bar` itself, and add each kind to `searched` and `found` (`searched = searched + count`) instead of replacing them?",
                      "nl": "Goed voor deze getallen, waar per groep maar één soort bestuurder op of boven de lat zit. Maar met {where} wordt elke soort doorzocht, en geeft je laatste regel `{got}`, waar het `{want}` moet zijn. Vergelijk je met `bar` zelf, en tel je elke soort op bij `searched` en `found` (`searched = searched + count`), in plaats van ze te vervangen?"},
            "other_error": {"en": "Right for these numbers. But with {where}, your code stops with an error. Do you use `drivers` and `bar` themselves?",
                            "nl": "Goed voor deze getallen. Maar met {where} stopt je code met een fout. Gebruik je `drivers` en `bar` zelf?"},
            "correct": {"en": "Group A: 0.75, group B: 0.50. The officer used the same bar for both, and the searches of group B still find something less often. The outcome test would point to a lower bar for group B, and be wrong: that is infra-marginality. Simoiu, Corbett-Davies and Goel give these chances, with a bar of 10 percent: \"searches of whites will be successful 75% of the time whereas searches of blacks will be successful only 50% of the time.\"",
                        "nl": "Groep A: 0,75, groep B: 0,50. De agent gebruikte voor allebei dezelfde lat, en toch vinden de doorzoekingen bij groep B minder vaak iets. De outcome test zou wijzen op een lagere lat voor groep B, en zich vergissen: dat is infra-marginality. Simoiu, Corbett-Davies en Goel geven deze kansen, met een lat van 10 procent: dan vinden doorzoekingen bij witte bestuurders in 75 procent van de gevallen iets, en bij zwarte bestuurders maar in 50 procent."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("ratio", _RATIO_START, "empty", None),
    ("ratio", _RATIO_SOLUTION, "correct", None),
    ("ratio", _r('    ratio[group] = round(success["native Dutch"] / success[group], 2)\n'), "correct", None),
    ("ratio", _r('    first = success["native Dutch"]\n    ratio[group] = first / success[group]\n'), "correct", None),
    ("ratio", _r('    ratio[group] = success["native Dutch"] / _success[group]\n'), "error", "underscore"),
    ("ratio", _r('    ratio[group] = success["Dutch"] / success[group]\n'), "error", "key"),
    ("ratio", _r('    ratio[group] = success["native Dutch"] / success["group"]\n'), "error", "key"),
    ("ratio", _r('    ratio = success["native Dutch"] / success[group]\n'), "wrong", "one-number"),
    ("ratio", _r('    ratio[group] = round(success["native Dutch"] / success[group], 1)\n'), "wrong", "rounded"),
    ("ratio", _r('    ratio[group] = 100 * success["native Dutch"] / success[group]\n'), "wrong", "percent"),
    ("ratio", _r('    ratio[group] = success[group] / success["native Dutch"]\n'), "wrong", "inverted"),
    ("ratio", _r('    ratio[group] = round(success[group] / success["native Dutch"], 2)\n'), "wrong", "inverted"),
    ("ratio", _r('    ratio[group] = success["native Dutch"] - success[group]\n'), "wrong", "difference"),
    ("ratio", _r('    share = success["native Dutch"] / success[group]\nratio[group] = share\n'), "wrong", "one-group"),
    ("ratio", _r('    ratio["group"] = success["native Dutch"] / success[group]\n'), "wrong", "other-keys"),
    ("ratio", _r('    ratio[group] = success["Moroccan"] / success[group]\n'), "wrong", None),
    ("ratio", _r('    ratio[group] = 0.34 / success[group]\n'), "almost", "other-value"),
    ("ratio", _RATIO_SOLUTION.replace(_SUCCESS_LINE, 'success = {"native Dutch": 0.34, "Hindustani": 0.25, "Moroccan": 0.19}'),
     "wrong", "changed-line"),
    ("ratio", _r("    ...\n    ratio[group] = 1\n"), "wrong", "dots-left"),
    ("hits", _HITS_START, "empty", None),
    ("hits", _HITS_SOLUTION, "correct", None),
    ("hits", _hswap("chance >= bar", "chance > bar"), "correct", None),
    ("hits", _h("        if chance >= bar:\n            searched = searched + count\n"
                "            found = found + count * chance\n",
                "    hit_rate[group] = round(found / searched, 2)\nhit_rate"), "correct", None),
    ("hits", _hswap("found = found + chance * count", "found = _found + chance * count"), "error", "underscore"),
    ("hits", _hswap("            searched = searched + count\n", ""), "error", "nobody"),
    ("hits", _hswap("chance >= bar", "chance >= bars[group]"), "error", "bars"),
    ("hits", _hswap("found = found + chance * count", "found = found + count"), "wrong", "all-found"),
    ("hits", _hswap("searched = searched + count", "searched = searched + 1"), "wrong", "kinds"),
    ("hits", _h("        searched = searched + count\n        found = found + chance * count\n"), "wrong", "no-if"),
    ("hits", _hswap("chance >= bar", "chance < bar"), "wrong", "below"),
    ("hits", _hswap("found = found + chance * count", "found = found + chance"), "wrong", "chance-once"),
    ("hits", _h(_HITS_BODY, "hit_rate[group] = found / searched\nhit_rate"), "wrong", "one-group"),
    ("hits", _hswap("found = found + chance * count", "found = found + chance * chance * count"), "wrong", None),
    ("hits", _hswap("chance >= bar", "chance >= 0.1"), "almost", "other-value"),
    ("hits", _h("        if chance >= bar:\n            searched = count\n            found = chance * count\n"), "almost", "other-value"),
    ("hits", _h("        if chance >= bar:\n            searched = searched + 1\n            found = found + chance\n"), "almost", "other-value"),
    ("hits", _HITS_SOLUTION.replace(_BAR_LINE, "bar = 0.6"), "wrong", "changed-line"),
    ("hits", _h("        ...\n        searched = 1\n        found = 1\n"), "wrong", "dots-left"),
]
