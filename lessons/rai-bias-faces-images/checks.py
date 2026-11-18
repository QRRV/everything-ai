# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-bias-faces-images: the error rate of one value of one attribute, added up from the cells of
Gender Shades (Buolamwini and Gebru 2018), and the mean gap between the share of women that an image generator draws
and the share in the US labor statistics, from Table 3 of Luccioni and colleagues (2023); with the picture of each."""
import math
import numbers
import re

from everything_ai import yours
from everything_ai.check import CodeBox, Mistake

# --- Gender Shades ----------------------------------------------------------------------------------------------------
# Per system and per group: the faces the system found and labeled ("classified") and the faces it labeled wrongly
# ("errors"). From the supplementary material of Gender Shades: the number of people per Fitzpatrick type and gender
# minus the faces each system did not detect (its Table 2), and the error counts (its Table 1). The rates match
# Table 4 of the paper only with the undetected faces left out, the rule Raji and Buolamwini (2019) state.
SYSTEMS = {
    "Microsoft": [("darker", "female", 265, 55), ("darker", "male", 300, 18),
                  ("lighter", "female", 295, 5), ("lighter", "male", 385, 0)],
    "Face++": [("darker", "female", 258, 89), ("darker", "male", 293, 2),
               ("lighter", "female", 296, 29), ("lighter", "male", 384, 3)],
    "IBM": [("darker", "female", 271, 94), ("darker", "male", 317, 38),
            ("lighter", "female", 296, 21), ("lighter", "male", 384, 1)],
}
# The people per group in the Pilot Parliaments Benchmark (1,270 in all), and Table 4 of Gender Shades: the error rate
# in percent per system, for women, men, darker, lighter, and the four combinations.
PEOPLE = {("darker", "female"): 271, ("darker", "male"): 318, ("lighter", "female"): 296, ("lighter", "male"): 385}
TABLE_4 = {"Microsoft": (10.7, 2.6, 12.9, 0.7, 20.8, 6.0, 1.7, 0.0),
           "Face++": (21.3, 0.7, 16.5, 4.7, 34.5, 0.7, 9.8, 0.8),
           "IBM": (20.3, 5.6, 22.4, 3.2, 34.7, 12.0, 7.1, 0.3)}


def cells(system: str = "Microsoft") -> list:
    """The cells of one system, as the lesson's code writes them: a dict per group."""
    return [{"skin": s, "gender": g, "classified": n, "errors": e} for s, g, n, e in SYSTEMS[system]]


def error_rate(rows: list, key: str, value: str) -> float:
    """The errors of every cell whose `key` is `value`, divided by the faces classified in those cells."""
    picked = [c for c in rows if c[key] == value]
    return sum(c["errors"] for c in picked) / sum(c["classified"] for c in picked)


def mean_of_rates(rows: list, key: str, value: str) -> float:
    """The known mistake: the mean of the cells' rates, as if the cells were the same size."""
    rates = [c["errors"] / c["classified"] for c in rows if c[key] == value]
    return sum(rates) / len(rates)


# --- image generators -------------------------------------------------------------------------------------------------
# Luccioni, Akiki, Mitchell and Jernite (2023), Table 3: the 146 occupations of the US Bureau of Labor Statistics in
# five groups by their share of women, from the fifth with the fewest to the fifth with the most. "bls" is the percent
# women in the statistics; the others are the percent of images in clusters linked to women, for Stable Diffusion 1.4,
# Stable Diffusion 2 and Dall-E 2.
QUINTILES = [
    {"bls": 7.5, "sd14": 8.5, "sd2": 5.3, "dalle2": 1.4},
    {"bls": 26.5, "sd14": 15.2, "sd2": 14.9, "dalle2": 2.8},
    {"bls": 47.1, "sd14": 32.2, "sd2": 23.1, "dalle2": 6.0},
    {"bls": 68.4, "sd14": 54.3, "sd2": 46.3, "dalle2": 21.6},
    {"bls": 86.8, "sd14": 83.0, "sd2": 78.3, "dalle2": 56.3},
]
SYSTEM_NAMES = {"sd14": "Stable Diffusion 1.4", "sd2": "Stable Diffusion 2", "dalle2": "Dall·E 2"}


def gap(rows: list, system: str) -> float:
    """The mean over the groups of the system's share of women minus the share in the statistics."""
    return sum(q[system] - q["bls"] for q in rows) / len(rows)


# --- pictures ---------------------------------------------------------------------------------------------------------
def _number(x) -> float | None:
    """A real number that is not True or False, as a float; else None."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _show(x: float, lang: str, digits: int = 3) -> str:
    text = f"{x:.{digits}f}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _cells_ok(rows) -> bool:
    """The learner's cells, when they still look like the lesson's: 1 to 8 dicts with a skin, a gender and counts."""
    if not isinstance(rows, list) or not 1 <= len(rows) <= 8:
        return False
    for c in rows:
        if type(c) is not dict or not isinstance(c.get("skin"), str) or not isinstance(c.get("gender"), str):
            return False
        n, e = _number(c.get("classified")), _number(c.get("errors"))
        if n is None or e is None or not 0 < n <= 5000 or not 0 <= e <= n:
            return False
    return True


_RATE_WORDS = {
    "en": {"title": "Your error rate for {key} = {value}: {rate}", "xlabel": "error rate",
           "mine": "your rate {rate}", "key": "filled: the cells you add up; open: the other cells",
           "alt": "One bar per group, as long as its error rate: {shown}. The cells with {key} = {value} are filled. A dashed line at your rate, {rate}."},
    "nl": {"title": "Jouw foutpercentage voor {key} = {value}: {rate}", "xlabel": "deel fout",
           "mine": "jouw uitkomst {rate}", "key": "gevuld: de vakjes die je optelt; open: de andere vakjes",
           "alt": "Eén balk per groep, zo lang als het deel dat fout ging: {shown}. De vakjes met {key} = {value} zijn gevuld. Een gestippelde lijn bij jouw uitkomst, {rate}."},
}


def _rate_picture(seen) -> str:
    """A bar per cell as long as its error rate, filled for the cells the learner adds up and open for the others,
    with a dashed line at the learner's rate: it lies between the rates of the cells it mixes."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    rate = _number(mine.value)
    names = mine.names if isinstance(mine.names, dict) else {}
    rows, key, value = names.get("cells"), names.get("key"), names.get("value")
    if rate is None or not _cells_ok(rows) or key not in ("skin", "gender") or not isinstance(value, str):
        return ""
    w = _RATE_WORDS[seen.lang]
    rates = [c["errors"] / c["classified"] for c in rows]
    right = max(0.4, max(rates) * 1.15, min(rate, 1.0) * 1.15)
    top = 1.1 * len(rows) + 0.9
    ticks = chart.ticks(0, right, 4)
    p = chart.Panel(x=(0, right), y=(0, top), xlabel=w["xlabel"], yticks=[], xticks=ticks,
                    xtick_labels=[_show(t, seen.lang, 1) for t in ticks])
    for i, (c, r) in enumerate(zip(rows, rates)):
        y = top - 0.9 - 1.1 * (i + 1)
        p.text(0.003 * right, y + 0.68, f'{c["skin"]} {c["gender"]}: {_show(r, seen.lang)}', "ink", anchor="start",
               halo=True)
        p.rect(0, y + 0.12, max(r, 0.002 * right), y + 0.42, role="loss", solid=c.get(key) == value)
    if 0 <= rate <= right:
        p.line([rate, rate], [0, top - 0.75], role="ink-2", width=1.5, dashed=True)
        p.text(min(rate + 0.01 * right, 0.62 * right), top - 0.45, w["mine"].format(rate=_show(rate, seen.lang)),
               "ink-2", anchor="start", halo=True)
    shown = "; ".join(f'{c["skin"]} {c["gender"]} {_show(r, seen.lang)}' for c, r in zip(rows, rates))
    svg = chart.figure([p], alt=w["alt"].format(shown=shown, key=key, value=value, rate=_show(rate, seen.lang)),
                       width=520, height=max(200, 46 * len(rows) + 110))
    svg = chart.with_decimal_comma(svg) if seen.lang == "nl" else svg
    title = w["title"].format(key=key, value=value, rate=_show(rate, seen.lang))
    return yours.figure(title, svg + f'<div style="{_UI}">{w["key"]}</div>')


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"

_GAP_WORDS = {
    "en": {"title": "Your gap for {system}: {gap} percentage points", "xlabel": "percent women",
           "rows": ["group 1 (fewest women)", "group 2", "group 3", "group 4", "group 5 (most women)"],
           "key": "open: the labor statistics (bls); filled: the images of {system}",
           "alt": "One row per group of occupations, with an open dot at the percent women in the statistics and a filled dot at the percent in the images of {system}: {shown}."},
    "nl": {"title": "Jouw verschil voor {system}: {gap} procentpunt", "xlabel": "procent vrouwen",
           "rows": ["groep 1 (minste vrouwen)", "groep 2", "groep 3", "groep 4", "groep 5 (meeste vrouwen)"],
           "key": "open: de arbeidsstatistiek (bls); gevuld: de afbeeldingen van {system}",
           "alt": "Eén rij per groep beroepen, met een open stip bij het percentage vrouwen in de statistiek en een gevulde stip bij het percentage in de afbeeldingen van {system}: {shown}."},
}


def _quintiles_ok(rows, system) -> bool:
    if not isinstance(rows, list) or len(rows) != 5 or not isinstance(system, str):
        return False
    for q in rows:
        if type(q) is not dict:
            return False
        for k in ("bls", system):
            v = _number(q.get(k))
            if v is None or not 0 <= v <= 100:
                return False
    return True


def _gap_picture(seen) -> str:
    """Per group of occupations, an open dot at the share of women in the statistics and a filled dot at the share in
    the images of the system, with a line between them: the gaps the learner averages."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    value = _number(mine.value)
    names = mine.names if isinstance(mine.names, dict) else {}
    rows, system = names.get("quintiles"), names.get("system")
    if value is None or not _quintiles_ok(rows, system):
        return ""
    w = _GAP_WORDS[seen.lang]
    shown_name = SYSTEM_NAMES.get(system, system)
    top = 5 * 1.1 + 0.3
    p = chart.Panel(x=(0, 100), y=(0, top), xlabel=w["xlabel"], yticks=[], xticks=[0, 25, 50, 75, 100],
                    xtick_labels=["0", "25", "50", "75", "100"])
    for i, (q, name) in enumerate(zip(rows, w["rows"])):
        y = top - 0.3 - 1.1 * (i + 1)
        real, made = float(q["bls"]), float(q[system])
        p.text(1, y + 0.72, name, "ink", anchor="start", halo=True)
        p.line([real, made], [y + 0.3, y + 0.3], role="ink-3", width=1.5)
        p.points([real], [y + 0.3], role="ink", r=5, hollow=True)
        p.points([made], [y + 0.3], role="ink", r=5)
    shown = "; ".join(f'{name}: {_show(float(q["bls"]), seen.lang, 1)} {"en" if seen.lang == "nl" else "and"} '
                      f'{_show(float(q[system]), seen.lang, 1)}' for q, name in zip(rows, w["rows"]))
    svg = chart.figure([p], alt=w["alt"].format(system=shown_name, shown=shown), width=520, height=320)
    svg = chart.with_decimal_comma(svg) if seen.lang == "nl" else svg
    title = w["title"].format(system=shown_name, gap=_show(value, seen.lang, 2))
    return yours.figure(title, svg + f'<div style="{_UI}">{w["key"].format(system=shown_name)}</div>')


# The lesson is done when these are correct. "subgroups" and "law" are choice questions in the lesson.
REQUIRED = ["rate", "subgroups", "gap", "law"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _near(want: float, tol: float):
    """A mistake that matches a number within `tol` of `want`."""
    def matches(got, text):
        x = _number(got)
        return x is not None and abs(x - want) < tol
    return matches


def _close(tol: float):
    def compare(got, want):
        a, b = _number(got), _number(want)
        return a is not None and b is not None and abs(a - b) < tol
    return compare


# --- exercise 1: the error rate of one value of one attribute --------------------------------------------------------
_MS = cells("Microsoft")


def _cell_line(c: dict) -> str:
    return f"    {c!r},".replace("'", '"')


_CELL_LINES = [_cell_line(c) for c in _MS]
_KEY_LINE, _VALUE_LINE = 'key = "gender"', 'value = "female"'
_R_HEAD = ("# Microsoft in Gender Shades (2018): per group, the faces it classified and its errors\n"
           "cells = [\n" + "\n".join(_CELL_LINES) + "\n]\n"
           "# the group to measure: every cell whose key has this value\n" + _KEY_LINE + "\n" + _VALUE_LINE + "\n\n")
_R_START = _R_HEAD + "...\nrate\n"
_R_BODY = ("classified = 0\nerrors = 0\nfor cell in cells:\n    if cell[key] == value:\n"
           "        classified = classified + cell[\"classified\"]\n        errors = errors + cell[\"errors\"]\n"
           "rate = errors / classified\n")
_R_SOLUTION = _R_HEAD + _R_BODY + "rate"


def _r(body: str) -> str:
    return _R_HEAD + body + "rate"


def _r_swap(old: str, new: str) -> str:
    assert _R_BODY.count(old) == 1
    return _r(_R_BODY.replace(old, new))


_WOMEN = error_rate(_MS, "gender", "female")                                  # 60 / 560
_ALL = sum(c["errors"] for c in _MS) / sum(c["classified"] for c in _MS)        # 78 / 1245
_ONE_CELL = [c["errors"] / c["classified"] for c in _MS if c["gender"] == "female"]   # 55 / 265 and 5 / 295

# --- exercise 3: the gap of an image generator -----------------------------------------------------------------------


def _q_line(q: dict) -> str:
    return f"    {q!r},".replace("'", '"')


_Q_LINES = [_q_line(q) for q in QUINTILES]
_SYSTEM_LINE = 'system = "dalle2"'
_G_HEAD = ("# Luccioni and colleagues (2023), Table 3: occupations in five groups by their share of women,\n"
           "# with the percent women in the statistics (bls) and in the images of three systems\n"
           "quintiles = [\n" + "\n".join(_Q_LINES) + "\n]\n"
           "# the system to measure\n" + _SYSTEM_LINE + "\n\n")
_G_START = _G_HEAD + "...\ngap\n"
_G_BODY = ("total = 0\nfor quintile in quintiles:\n    total = total + quintile[system] - quintile[\"bls\"]\n"
           "gap = total / len(quintiles)\n")
_G_SOLUTION = _G_HEAD + _G_BODY + "gap"


def _g(body: str) -> str:
    return _G_HEAD + body + "gap"


def _g_swap(old: str, new: str) -> str:
    assert _G_BODY.count(old) == 1
    return _g(_G_BODY.replace(old, new))


_DIFFS = [q["dalle2"] - q["bls"] for q in QUINTILES]   # -6.1, -23.7, -41.1, -46.8, -30.5
_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

EXERCISES = {
    "rate": CodeBox(
        name="rate",
        picture=_rate_picture,
        start=_R_START,
        solution=_R_SOLUTION,
        keep=["cells = ["] + _CELL_LINES + [_KEY_LINE, _VALUE_LINE],
        compare=_close(0.0006),
        digits=3,
        # The check also runs the box for the darker cells and for the men.
        others=[{"key": "skin", "value": "darker"}, {"value": "male"}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_cells`. In the box the names have none: `cells`, `key`, `value`, `cell`, `classified`, `errors`, `rate`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_cells`. In het vak hebben de namen dat niet: `cells`, `key`, `value`, `cell`, `classified`, `errors`, `rate`."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks for a key that is not there. Each cell has `\"skin\"`, `\"gender\"`, `\"classified\"` and `\"errors\"`. To pick the group, use the name `key` itself, without quotes: `cell[key]`.",
                    nl="Je code vraagt om een sleutel die er niet is. Elk vakje heeft `\"skin\"`, `\"gender\"`, `\"classified\"` en `\"errors\"`. Gebruik voor de groep de naam `key` zelf, zonder aanhalingstekens: `cell[key]`."),
            Mistake("zero", error=lambda e: isinstance(e, ZeroDivisionError),
                    en="Your code divides by zero: no faces were added to the total. Does the `if` compare `cell[key]` with `value`, and does each cell it picks add its `\"classified\"`?",
                    nl="Je code deelt door nul: er zijn geen gezichten bij het totaal opgeteld. Vergelijkt de `if` `cell[key]` met `value`, en telt elk vakje dat hij kiest zijn `\"classified\"` op?"),
            Mistake("everyone", matches=_near(_ALL, 0.0006),
                    en="Your last line gives `{got}`: the error rate of all faces, 78 of 1,245. Add only the cells whose `key` is `value`: put an `if` in the loop.",
                    nl="Je laatste regel geeft `{got}`: het deel fout van alle gezichten, 78 van de 1.245. Tel alleen de vakjes op waarvan `key` gelijk is aan `value`: zet een `if` in de loop."),
            Mistake("mean", matches=_near(mean_of_rates(_MS, "gender", "female"), 0.0006),
                    en="Your last line gives `{got}`: the mean of the two cells' rates. But the cells differ in size, 265 and 295 faces. Add up the errors and the faces of the cells first, and divide once, at the end.",
                    nl="Je laatste regel geeft `{got}`: het gemiddelde van de twee foutpercentages. Maar de vakjes zijn niet even groot: 265 en 295 gezichten. Tel eerst de fouten en de gezichten van de vakjes op, en deel één keer, aan het eind."),
            Mistake("right-share", matches=_near(1 - _WOMEN, 0.0006),
                    en="Your last line gives `{got}`: the share it labeled *right*. The error rate is the share it labeled wrongly: `errors / classified`.",
                    nl="Je laatste regel geeft `{got}`: het deel dat het systeem *goed* labelde. Het foutpercentage is het deel dat het fout labelde: `errors / classified`."),
            Mistake("percent", matches=_near(100 * _WOMEN, 0.06),
                    en="Your last line gives `{got}`, a percentage. The check wants a share from 0 to 1, as in the worked example: leave out the `* 100`.",
                    nl="Je laatste regel geeft `{got}`, een percentage. De check wil een deel van 0 tot 1, zoals in het uitgewerkte voorbeeld: laat de `* 100` weg."),
            Mistake("upside", matches=_near(1 / _WOMEN, 0.006),
                    en="Your last line gives `{got}`, more than 1: you divided the faces by the errors. Divide the errors by the faces: `errors / classified`.",
                    nl="Je laatste regel geeft `{got}`, meer dan 1: je deelde de gezichten door de fouten. Deel de fouten door de gezichten: `errors / classified`."),
            Mistake("one-cell", matches=lambda got, text: any(_near(r, 0.0006)(got, text) for r in _ONE_CELL),
                    en="Your last line gives `{got}`: the rate of one cell, not of the group. Add each cell's counts to the totals, `classified = classified + cell[\"classified\"]`, instead of replacing them.",
                    nl="Je laatste regel geeft `{got}`: het foutpercentage van één vakje, niet van de groep. Tel de aantallen van elk vakje bij de totalen op, `classified = classified + cell[\"classified\"]`, in plaats van ze te vervangen."),
            Mistake("of-errors", matches=_near(60 / 78, 0.0006),
                    en="Your last line gives `{got}`: the women's share of all errors, 60 of 78. The error rate divides the women's errors by the women's faces, 560.",
                    nl="Je laatste regel geeft `{got}`: het aandeel van de vrouwen in alle fouten, 60 van de 78. Het foutpercentage deelt de fouten bij vrouwen door de gezichten van vrouwen, 560."),
            Mistake("all-faces", matches=_near(60 / 1245, 0.0006),
                    en="Your last line gives `{got}`: the women's errors divided by all 1,245 faces. Divide by the faces of the cells you picked: add `cell[\"classified\"]` inside the `if`.",
                    nl="Je laatste regel geeft `{got}`: de fouten bij vrouwen gedeeld door alle 1.245 gezichten. Deel door de gezichten van de vakjes die je koos: tel `cell[\"classified\"]` op binnen de `if`."),
            Mistake("other-side", matches=_near(error_rate(_MS, "gender", "male"), 0.0006),
                    en="Your last line gives `{got}`: the rate of the cells whose `key` is *not* `value`, the men. Pick the cells with `cell[key] == value`.",
                    nl="Je laatste regel geeft `{got}`: het foutpercentage van de vakjes waarvan `key` *niet* `value` is, de mannen. Kies de vakjes met `cell[key] == value`."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...` with your code: start with `classified = 0` and `errors = 0`, add up the counts of the cells whose `key` is `value`, and divide at the end.",
                      "nl": "Vervang de `...` door je code: begin met `classified = 0` en `errors = 0`, tel de aantallen op van de vakjes waarvan `key` gelijk is aan `value`, en deel aan het eind."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the lines of `cells`, `key` and `value` exactly as they were: put them back. The check tries other values by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `cells`, `key` en `value` nodig zoals ze waren: zet ze terug. De check probeert zelf andere waarden."},
            "no_value": {"en": "The last line of the box shows nothing. Put `rate` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `rate` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you add up the `\"errors\"` and the `\"classified\"` of every cell with `cell[key] == value`, and then divide the errors by the faces?",
                      "nl": "Je laatste regel geeft `{got}`. Tel je de `\"errors\"` en de `\"classified\"` op van elk vakje met `cell[key] == value`, en deel je daarna de fouten door de gezichten?"},
            "other": {"en": "Right for the women. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `key` and `value` themselves, and not the words `\"gender\"` and `\"female\"`?",
                      "nl": "Goed voor de vrouwen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `key` en `value` zelf, en niet de woorden `\"gender\"` en `\"female\"`?"},
            "other_error": {"en": "Right for the women. But with {where}, your code stops with an error. Do you use `key` and `value` themselves?",
                            "nl": "Goed voor de vrouwen. Maar met {where} stopt je code met een fout. Gebruik je `key` en `value` zelf?"},
            "correct": {"en": "`0.107`: 60 errors on 560 women. For darker-skinned people it is 0.129. Both lie far below the 0.208 of darker-skinned women: a number per attribute mixes the worst cell with a better one. That is why Gender Shades reports the combinations.",
                        "nl": "`0.107`: 60 fouten bij 560 vrouwen. Voor mensen met een donkerdere huid is het 0,129. Allebei liggen ver onder de 0,208 van vrouwen met een donkerdere huid: een getal per eigenschap mengt het slechtste vakje met een beter. Daarom geeft Gender Shades de combinaties."},
        },
    ),
    "gap": CodeBox(
        name="gap",
        picture=_gap_picture,
        start=_G_START,
        solution=_G_SOLUTION,
        keep=["quintiles = ["] + _Q_LINES + [_SYSTEM_LINE],
        compare=_close(0.006),
        digits=2,
        # The check also runs the box for the two versions of Stable Diffusion.
        others=[{"system": "sd14"}, {"system": "sd2"}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_quintiles`. In the box the names have none: `quintiles`, `system`, `quintile`, `total`, `gap`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_quintiles`. In het vak hebben de namen dat niet: `quintiles`, `system`, `quintile`, `total`, `gap`."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks a group for a key it does not have. Each group has `\"bls\"`, `\"sd14\"`, `\"sd2\"` and `\"dalle2\"`. To pick the system, use the name `system` itself, without quotes: `quintile[system]`.",
                    nl="Je code vraagt een groep om een sleutel die hij niet heeft. Elke groep heeft `\"bls\"`, `\"sd14\"`, `\"sd2\"` en `\"dalle2\"`. Gebruik voor het systeem de naam `system` zelf, zonder aanhalingstekens: `quintile[system]`."),
            Mistake("sum", matches=_near(sum(_DIFFS), 0.006),
                    en="Your last line gives `{got}`: the sum of the five differences. The gap is their mean: divide the total by the number of groups, `len(quintiles)`.",
                    nl="Je laatste regel geeft `{got}`: de som van de vijf verschillen. Het verschil dat de check wil, is hun gemiddelde: deel het totaal door het aantal groepen, `len(quintiles)`."),
            Mistake("count", matches=lambda got, text: _near(sum(_DIFFS) / 4, 0.006)(got, text) or _near(sum(_DIFFS) / 6, 0.006)(got, text),
                    en="Your last line gives `{got}`: you divided by another number than the number of groups. There are five: divide by `len(quintiles)`.",
                    nl="Je laatste regel geeft `{got}`: je deelde door een ander getal dan het aantal groepen. Dat zijn er vijf: deel door `len(quintiles)`."),
            Mistake("made-only", matches=_near(sum(q["dalle2"] for q in QUINTILES) / 5, 0.006),
                    en="Your last line gives `{got}`: the mean share of women in the images. The gap compares it with the statistics: subtract `quintile[\"bls\"]` in each group.",
                    nl="Je laatste regel geeft `{got}`: het gemiddelde deel vrouwen in de afbeeldingen. Het verschil vergelijkt dat met de statistiek: trek in elke groep `quintile[\"bls\"]` ervan af."),
            Mistake("bls-only", matches=_near(sum(q["bls"] for q in QUINTILES) / 5, 0.006),
                    en="Your last line gives `{got}`: the mean share of women in the statistics. The gap is the system's share minus that: `quintile[system] - quintile[\"bls\"]`.",
                    nl="Je laatste regel geeft `{got}`: het gemiddelde deel vrouwen in de statistiek. Het verschil is het deel van het systeem min dat: `quintile[system] - quintile[\"bls\"]`."),
            Mistake("one-group", matches=lambda got, text: any(_near(d, 0.006)(got, text) for d in (_DIFFS[0], _DIFFS[-1], _DIFFS[-1] / 5)),
                    en="Your last line gives `{got}`: your total holds the difference of one group only. Add each group's difference to the total, `total = total + ...`, instead of replacing it.",
                    nl="Je laatste regel geeft `{got}`: je totaal bevat het verschil van maar één groep. Tel het verschil van elke groep bij het totaal op, `total = total + ...`, in plaats van het te vervangen."),
            Mistake("flipped", matches=lambda got, text: _number(got) is not None and _number(got) > 0,
                    en="Your last line gives `{got}`, a positive number: that would mean Dall·E 2 draws *more* women than the statistics, but it draws fewer in every group. Take the system's share minus the statistics' share: `quintile[system] - quintile[\"bls\"]`.",
                    nl="Je laatste regel geeft `{got}`, een positief getal: dan zou Dall·E 2 *meer* vrouwen tekenen dan de statistiek, maar het tekent er in elke groep minder. Neem het deel van het systeem min dat van de statistiek: `quintile[system] - quintile[\"bls\"]`."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...` with your code: start with `total = 0`, add each group's `quintile[system] - quintile[\"bls\"]`, and divide at the end.",
                      "nl": "Vervang de `...` door je code: begin met `total = 0`, tel van elke groep `quintile[system] - quintile[\"bls\"]` op, en deel aan het eind."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the lines of `quintiles` and `system` exactly as they were: put them back. The check tries the other systems by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `quintiles` en `system` nodig zoals ze waren: zet ze terug. De check probeert zelf de andere systemen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `gap` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `gap` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you add up `quintile[system] - quintile[\"bls\"]` over the five groups, and divide the total by `len(quintiles)`?",
                      "nl": "Je laatste regel geeft `{got}`. Tel je `quintile[system] - quintile[\"bls\"]` op over de vijf groepen, en deel je het totaal door `len(quintiles)`?"},
            "other": {"en": "Right for Dall·E 2. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `system` itself, and not the word `\"dalle2\"`?",
                      "nl": "Goed voor Dall·E 2. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `system` zelf, en niet het woord `\"dalle2\"`?"},
            "other_error": {"en": "Right for Dall·E 2. But with {where}, your code stops with an error. Do you use `system` itself?",
                            "nl": "Goed voor Dall·E 2. Maar met {where} stopt je code met een fout. Gebruik je `system` zelf?"},
            "correct": {"en": "`-29.64`: in these groups of occupations, Dall·E 2 draws on average 29.64 percentage points fewer women than the statistics. Stable Diffusion 2 gives −13.68 and Stable Diffusion 1.4 −8.62. Luccioni and colleagues saw the same order in the captions of the images: of the captions that named a gender, 19.96 percent said woman for Dall·E 2 and 38.04 percent for Stable Diffusion 1.4, against 47.03 percent women in the statistics.",
                        "nl": "`-29.64`: in deze groepen beroepen tekent Dall·E 2 gemiddeld 29,64 procentpunt minder vrouwen dan de statistiek. Stable Diffusion 2 geeft −13,68 en Stable Diffusion 1.4 −8,62. Luccioni en collega's zagen dezelfde volgorde in de bijschriften van de afbeeldingen: van de bijschriften die een gender noemden, zei 19,96 procent vrouw bij Dall·E 2 en 38,04 procent bij Stable Diffusion 1.4, tegenover 47,03 procent vrouwen in de statistiek."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("rate", _R_START, "empty", None),
    ("rate", _R_SOLUTION, "correct", None),
    ("rate", _r("classified = 0\nerrors = 0\nfor cell in cells:\n    if cell[key] == value:\n"
                "        classified += cell[\"classified\"]\n        errors += cell[\"errors\"]\n"
                "rate = round(errors / classified, 3)\n"), "correct", None),
    ("rate", _r("faces = []\nwrong = []\nfor cell in cells:\n    if cell[key] == value:\n"
                "        faces.append(cell[\"classified\"])\n        wrong.append(cell[\"errors\"])\n"
                "rate = sum(wrong) / sum(faces)\n"), "correct", None),
    ("rate", _r(_R_BODY.replace("in cells", "in _cells")), "error", "underscore"),
    ("rate", _r_swap("cell[key] == value", "cell[\"key\"] == value"), "error", "key"),
    ("rate", _r_swap("cell[key] == value", "cell[key] == \"Female\""), "error", "zero"),
    ("rate", _r("classified = 0\nerrors = 0\nfor cell in cells:\n    classified = classified + cell[\"classified\"]\n"
                "    errors = errors + cell[\"errors\"]\nrate = errors / classified\n"), "wrong", "everyone"),
    ("rate", _r("total = 0\ncount = 0\nfor cell in cells:\n    if cell[key] == value:\n"
                "        total = total + cell[\"errors\"] / cell[\"classified\"]\n        count = count + 1\n"
                "rate = total / count\n"), "wrong", "mean"),
    ("rate", _r_swap("rate = errors / classified", "rate = 1 - errors / classified"), "wrong", "right-share"),
    ("rate", _r_swap("rate = errors / classified", "rate = errors / classified * 100"), "wrong", "percent"),
    ("rate", _r_swap("rate = errors / classified", "rate = classified / errors"), "wrong", "upside"),
    ("rate", _r_swap("classified = classified + cell", "classified = cell").replace("errors = errors + cell",
                                                                                    "errors = cell"),
     "wrong", "one-cell"),
    ("rate", _r("all_errors = 0\nerrors = 0\nfor cell in cells:\n    all_errors = all_errors + cell[\"errors\"]\n"
                "    if cell[key] == value:\n        errors = errors + cell[\"errors\"]\nrate = errors / all_errors\n"),
     "wrong", "of-errors"),
    ("rate", _r("classified = 0\nerrors = 0\nfor cell in cells:\n    classified = classified + cell[\"classified\"]\n"
                "    if cell[key] == value:\n        errors = errors + cell[\"errors\"]\nrate = errors / classified\n"),
     "wrong", "all-faces"),
    ("rate", _r_swap("cell[key] == value", "cell[key] != value"), "wrong", "other-side"),
    ("rate", _r_swap("rate = errors / classified", "rate = errors / classified / 2"), "wrong", None),
    ("rate", _r("rate = 60 / 560\n"), "almost", "other-value"),
    ("rate", _r_swap("cell[key] == value", "cell[\"gender\"] == \"female\""), "almost", "other-value"),
    ("rate", _R_SOLUTION.replace('value = "female"', 'value = "male"'), "wrong", "changed-line"),
    ("rate", _r("...\nrate = 60 / 560\n"), "wrong", "dots-left"),
    ("gap", _G_START, "empty", None),
    ("gap", _G_SOLUTION, "correct", None),
    ("gap", _g("differences = []\nfor quintile in quintiles:\n"
               "    differences.append(quintile[system] - quintile[\"bls\"])\n"
               "gap = sum(differences) / len(differences)\n"), "correct", None),
    ("gap", _g("total = 0\nfor quintile in quintiles:\n    total += quintile[system] - quintile[\"bls\"]\n"
               "gap = round(total / 5, 2)\n"), "correct", None),
    ("gap", _g(_G_BODY.replace("in quintiles", "in _quintiles")), "error", "underscore"),
    ("gap", _g_swap("quintile[system]", "quintile[\"system\"]"), "error", "key"),
    ("gap", _g_swap("gap = total / len(quintiles)", "gap = total"), "wrong", "sum"),
    ("gap", _g_swap("gap = total / len(quintiles)", "gap = total / 4"), "wrong", "count"),
    ("gap", _g_swap(" - quintile[\"bls\"]", ""), "wrong", "made-only"),
    ("gap", _g_swap("total + quintile[system] - quintile[\"bls\"]", "total + quintile[\"bls\"]"), "wrong", "bls-only"),
    ("gap", _g_swap("total = total + quintile", "total = quintile"), "wrong", "one-group"),
    ("gap", _g_swap("quintile[system] - quintile[\"bls\"]", "quintile[\"bls\"] - quintile[system]"), "wrong",
     "flipped"),
    ("gap", _g("total = 0\nfor quintile in quintiles:\n    total = total + abs(quintile[system] - quintile[\"bls\"])\n"
               "gap = total / len(quintiles)\n"), "wrong", "flipped"),
    ("gap", _g_swap("gap = total / len(quintiles)", "gap = total / len(quintiles) - 1"), "wrong", None),
    ("gap", _g("gap = -29.64\n"), "almost", "other-value"),
    ("gap", _g_swap("quintile[system]", "quintile[\"dalle2\"]"), "almost", "other-value"),
    ("gap", _G_SOLUTION.replace('system = "dalle2"', 'system = "sd2"'), "wrong", "changed-line"),
    ("gap", _g("...\ngap = -29.64\n"), "wrong", "dots-left"),
]
