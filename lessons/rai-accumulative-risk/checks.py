# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-accumulative-risk: one blow or many small steps. The learner writes resilience, a made-up
model of Kasirzadeh's accumulative hypothesis in which each year's disruptions take something off a society's
resilience and it wins some back, and collapse_year, the first year in which a shock would take that resilience to 0
or below. A demo changes the yearly harm."""
import contextlib
import copy
import html
import io
import re

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, same_value, with_time_limit

# --- the data --------------------------------------------------------------------------------------------------------

START = 100                    # made-up: the resilience of a healthy society, and the most it can have
HARMS = [6, 3, 8, 5, 9, 4, 7, 10, 6, 8]   # made-up: what the disruptions of each of ten years take off
RECOVERY = 2                   # made-up: what the society wins back each year
SHOCK = 60                     # made-up: what a shock, such as an attack on the power grids, takes off at once
YEARS = 30                     # the years of the demo
DEMO_HARMS = [1, 2, 3, 4, 5]   # the stops of the demo's slider: the harm of every year


def resilience(harms, recovery):
    """What the learner writes in Exercise 1: the society's resilience at the end of each year, starting from 100;
    each year takes off that year's harm and wins back the recovery, never above 100."""
    result = []
    level = 100
    for harm in harms:
        level = min(100, level - harm + recovery)
        result.append(level)
    return result


def collapse_year(levels, shock):
    """What the learner writes in Exercise 2: the first year, counted from 1, in which the shock would take the
    resilience to 0 or below, and None if that happens in no year."""
    year = 0
    for level in levels:
        year = year + 1
        if level <= shock:
            return year
    return None


def demo_levels(harm):
    """The demo: the resilience at the end of each of the 30 years, with the same harm every year."""
    return resilience([harm] * YEARS, RECOVERY)


def demo_year(harm):
    """The year in which the demo's shock would topple the society, or None within the 30 years."""
    return collapse_year(demo_levels(harm), SHOCK)


def first_topple(harms=DEMO_HARMS):
    """The smallest harm of the slider at which the shock topples the society within the 30 years."""
    return next(h for h in harms if demo_year(h) is not None)


# --- pictures --------------------------------------------------------------------------------------------------------


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:middle")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_CODE = "font-family:var(--ea-font-code);font-size:0.78rem"


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is (its cells, a style of the row), and each cell is (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{h}</th>' for h in head)
    body = "".join(f'<tr style="{style}">' + "".join(f'<td style="{_CELL}{";" + s if s else ""}">{cell}</td>'
                                                    for cell, s in cells_) + "</tr>" for cells_, style in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


def _code(text: str) -> str:
    return f'<span style="{_CODE}">{html.escape(text)}</span>'


def _number(v):
    """v as a float, or None for something that is not a number, such as True or a text."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError, OverflowError):
        return None


def _value(v, digits: int = 4, limit: int = 70) -> str:
    """A result as the check shows it: a number with at most `digits` decimals, a text in quotes, a list or a dict of
    them, else what Python shows, cut short when it is longer than `limit`."""
    if isinstance(v, (list, tuple)):
        text = "[" + ", ".join(_value(x, digits, 10 ** 6) for x in v) + "]"
    elif isinstance(v, dict):
        text = "{" + ", ".join(f"{_value(k, digits)}: {_value(x, digits)}" for k, x in v.items()) + "}"
    elif isinstance(v, str):
        text = '"' + v + '"'
    else:
        f = _number(v)
        if f is None or f != f or f in (float("inf"), float("-inf")):
            text = repr(v) if f is None else str(f)
        else:
            text = f"{f:.{digits}f}".rstrip("0").rstrip(".")
            text = "0" if text == "-0" else text
    return text if len(text) <= limit else text[:limit - 3] + "..."


def show_levels(v) -> str:
    return _value(v, 2, 200)


def show_year(v) -> str:
    return _value(v, 2, 200)


def _same(got, want) -> bool:
    """Whether the learner's list is the course's, as the check compares them."""
    return isinstance(got, list) and same_value(got, want)


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _all gives for a call that stopped with an error


def _all(fn, calls: list) -> list:
    """The learner's function on copies of each list of arguments in `calls`, all within one time limit of a check:
    a list of the results, with _STOPPED for each call that stops with an error; the Timeout passes on."""
    def one(args):
        try:
            return fn(*(copy.deepcopy(a) for a in args))
        except Timeout:
            raise
        except Exception:   # noqa: BLE001   the check above says what went wrong
            return _STOPPED

    with contextlib.redirect_stdout(io.StringIO()):
        return with_time_limit(lambda: [one(args) for args in calls])


def _rings(got, xs: list, low: float, high: float) -> tuple[list, list]:
    """The learner's numbers that are inside the panel, as rings at the x of each place in the list."""
    mine = got if isinstance(got, list) else []
    rx, ry = [], []
    for k, x in enumerate(xs):
        f = _number(mine[k]) if k < len(mine) else None
        if f is not None and low <= f <= high:
            rx.append(x)
            ry.append(f)
    return rx, ry



# --- the demo --------------------------------------------------------------------------------------------------------

_DEMO_WORDS = {
    "en": {"title": "Resilience over 30 years with a harm of {h} a year",
           "x": "year", "y": "resilience", "shock": "shock of 60", "fall": "year {y}",
           "said": "With a harm of {h} a year and a recovery of 2, the shock of 60 would topple the society in year {y}, "
                   "when its resilience is {lvl}.",
           "stays": "With a harm of {h} a year, the recovery of 2 makes up for it: the resilience stays at 100, and the "
                    "shock of 60 is absorbed every year.",
           "holds": "With a harm of {h} a year, the resilience falls to {end} in 30 years, still above 60: the shock is "
                    "absorbed every year.",
           "note": "A made-up model, not Kasirzadeh's (the paper has no model with numbers): each year the harm comes off and 2 is won "
                   "back, never above 100. The dashed line is the shock: once the resilience is at it or below, the "
                   "shock takes it to 0 or below."},
    "nl": {"title": "Veerkracht over 30 jaar bij een schade van {h} per jaar",
           "x": "jaar", "y": "veerkracht", "shock": "schok van 60", "fall": "jaar {y}",
           "said": "Met een schade van {h} per jaar en een herstel van 2 zou de schok van 60 de samenleving in jaar {y} "
                   "omverwerpen, bij een veerkracht van {lvl}.",
           "stays": "Met een schade van {h} per jaar maakt het herstel van 2 het goed: de veerkracht blijft 100, en de "
                    "samenleving vangt de schok van 60 elk jaar op.",
           "holds": "Met een schade van {h} per jaar zakt de veerkracht in 30 jaar naar {end}, nog boven 60: de "
                    "samenleving vangt de schok elk jaar op.",
           "note": "Een verzonnen model, niet dat van Kasirzadeh (het paper heeft geen model met getallen): elk jaar gaat de schade eraf "
                   "en komt er 2 bij, nooit boven 100. De gestippelde lijn is de schok: zodra de veerkracht daarop of "
                   "daaronder ligt, brengt de schok de veerkracht op 0 of lager."},
}


def demo_said(harm, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    levels = demo_levels(harm)
    year = demo_year(harm)
    if year is not None:
        return w["said"].format(h=harm, y=year, lvl=levels[year - 1])
    if levels[-1] == START:
        return w["stays"].format(h=harm)
    return w["holds"].format(h=harm, end=levels[-1])


def demo_frame(v, lang: str) -> tuple[str, str]:
    """The demo: the resilience over the 30 years at the chosen yearly harm, the shock as a dashed line, and the year
    in which the shock would topple the society marked."""
    harm = min(DEMO_HARMS, key=lambda k: abs(k - float(v)))
    w = _DEMO_WORDS[lang]
    levels = [START] + demo_levels(harm)
    year = demo_year(harm)
    panel = chart.Panel(x=(0, 30.5), y=(0, 112), xlabel=w["x"], ylabel=w["y"], xticks=[0, 5, 10, 15, 20, 25, 30],
                        yticks=[0, 20, 40, 60, 80, 100])
    panel.line([0, YEARS], [SHOCK, SHOCK], role="ink-3", width=1.6, dashed=True)
    panel.text(0.6, SHOCK - 7, w["shock"], role="ink-2", anchor="start", halo=True)
    panel.line(list(range(YEARS + 1)), levels, role="data")
    if year is not None:
        panel.points([year], [levels[year]], role="ink", r=4.5)
        # to the right of the point and above the line, which falls from there
        panel.text(year + 0.7, levels[year] + 5, w["fall"].format(y=year), role="ink", anchor="start", halo=True)
    title = w["title"].format(h=harm)
    said = demo_said(harm, lang)
    svg = chart.figure([panel], alt=title + ". " + said, width=380, height=240)
    if lang == "nl":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(title, svg + _note(w["note"]), alt=title + ". " + said), said


# --- the pictures under the checks -----------------------------------------------------------------------------------

_LEVELS_WORDS = {
    "en": {"title": "Your resilience over ten made-up years",
           "x": "year", "y": "resilience",
           "head": ["harms", "recovery", "the course's", "yours"], "stopped": "error",
           "note": "The chart is the first case of the check: ten years of made-up harms, with a recovery of 2. A dot "
                   "is the course's resilience at the end of each year and a ring is yours. The table is every case; a "
                   "dashed row is one where your function gives something other than the course's."},
    "nl": {"title": "Jouw veerkracht over tien verzonnen jaren",
           "x": "jaar", "y": "veerkracht",
           "head": ["harms", "recovery", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De grafiek is het eerste geval van de check: tien jaar verzonnen schade, met een herstel van 2. Een "
                   "stip is de veerkracht van de course aan het eind van elk jaar en een ring de jouwe. De tabel bevat "
                   "elk geval; een gestippelde rij is er een waar jouw functie iets anders geeft dan die van de course."},
}


def resilience_picture(seen) -> str:
    """The first case as the resilience at the end of each year, the course's as dots and the learner's as rings, and
    every case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _LEVELS_WORDS[seen.lang]
    got = _all(fn, LEVEL_CASES)
    rows = []
    for i, case in enumerate(LEVEL_CASES):
        want = resilience(*case)
        if got[i] is _STOPPED:
            shown_, off = html.escape(w["stopped"]), True
        else:
            shown_, off = _code(show_levels(got[i])), not _same(got[i], want)
        rows.append(([(_code(_value(case[0])), ""), (_code(_value(case[1])), "text-align:right"),
                      (_code(show_levels(want)), ""), (shown_, "font-weight:600")], _OFF if off else ""))
    want = resilience(HARMS, RECOVERY)
    xs = list(range(1, len(HARMS) + 1))
    panel = chart.Panel(x=(0.5, 10.5), y=(0, 112), xlabel=w["x"], ylabel=w["y"], xticks=[2, 4, 6, 8, 10],
                        yticks=[0, 20, 40, 60, 80, 100])
    panel.line(xs, want, role="ink-3")
    panel.points(*_rings(got[0], xs, -2, 112), role="ink", r=7, hollow=True)
    panel.points(xs, want, role="ink", r=3)
    svg = chart.figure([panel], alt=w["title"], width=380, height=240)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


_YEAR_WORDS = {
    "en": {"title": "Your year of collapse for the ten made-up years",
           "x": "year", "y": "resilience", "shock": "shock of 60",
           "head": ["levels", "shock", "the course's", "yours"], "stopped": "error",
           "note": "The chart is the first case of the check: the resilience of Exercise 1's first case, and a shock of "
                   "60, the dashed line. The dot is the course's year, the first one at or below the shock, and the "
                   "ring is yours, at the resilience of that year. The table is every case; a dashed row is one where "
                   "your function gives something other than the course's."},
    "nl": {"title": "Jouw jaar van instorten voor de tien verzonnen jaren",
           "x": "jaar", "y": "veerkracht", "shock": "schok van 60",
           "head": ["levels", "shock", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De grafiek is het eerste geval van de check: de veerkracht uit het eerste geval van oefening 1, en "
                   "een schok van 60, de gestippelde lijn. De stip is het jaar van de course, het eerste op of onder de "
                   "schok, en de ring het jouwe, bij de veerkracht van dat jaar. De tabel bevat elk geval; een "
                   "gestippelde rij is er een waar jouw functie iets anders geeft dan die van de course."},
}


def collapse_picture(seen) -> str:
    """The first case as the resilience over the ten years with the shock as a dashed line, the course's year as a
    dot and the learner's as a ring, and every case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _YEAR_WORDS[seen.lang]
    got = _all(fn, YEAR_CASES)
    rows = []
    for i, case in enumerate(YEAR_CASES):
        want = collapse_year(*case)
        if got[i] is _STOPPED:
            shown_, off = html.escape(w["stopped"]), True
        else:
            shown_, off = _code(show_year(got[i])), not same_value(got[i], want)
        rows.append(([(_code(_value(case[0])), ""), (_code(_value(case[1])), "text-align:right"),
                      (_code(show_year(want)), ""), (shown_, "font-weight:600")], _OFF if off else ""))
    levels = YEAR_CASES[0][0]
    want = collapse_year(*YEAR_CASES[0])
    xs = list(range(1, len(levels) + 1))
    panel = chart.Panel(x=(0.5, 10.5), y=(0, 112), xlabel=w["x"], ylabel=w["y"], xticks=[2, 4, 6, 8, 10],
                        yticks=[0, 20, 40, 60, 80, 100])
    panel.line([0.5, 10.5], [SHOCK, SHOCK], role="ink-3", width=1.6, dashed=True)
    panel.text(0.8, SHOCK - 7, w["shock"], role="ink-2", anchor="start", halo=True)
    panel.line(xs, levels, role="ink-3")
    panel.points(xs, levels, role="ink-3", r=2.5)
    mine = got[0]
    if isinstance(mine, int) and not isinstance(mine, bool) and 1 <= mine <= len(levels):
        panel.points([mine], [levels[mine - 1]], role="ink", r=8, hollow=True)
    panel.points([want], [levels[want - 1]], role="ink", r=4)
    svg = chart.figure([panel], alt=w["title"], width=380, height=240)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["resilience", "collapse", "locked", "update"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# exercise 1: resilience; the first case is ten made-up years with a recovery of 2, then two good years at the top,
# harms the recovery makes up for, no years, one bad year and a slow way back, and five heavy years
LEVEL_CASES = [(HARMS, RECOVERY), ([1, 1, 5], 2), ([3, 3, 3], 3), ([], 2), ([10, 0, 0], 4), ([20] * 5, 5)]
_R_START = "def resilience(harms, recovery):\n    ...\n"
_R = ("def resilience(harms, recovery):\n"
      "    result = []\n"
      "    level = 100\n"
      "    for harm in harms:\n"
      "        level = min(100, level - harm + recovery)\n"
      "        result.append(level)\n"
      "    return result\n")
_R_RESET = _R.replace("min(100, level - harm + recovery)", "min(100, 100 - harm + recovery)")
_R_CAP = _R.replace("min(100, level - harm + recovery)", "level - harm + recovery")
_R_NOREC = _R.replace("min(100, level - harm + recovery)", "min(100, level - harm)")
_R_SIGN = _R.replace("min(100, level - harm + recovery)", "min(100, level + harm - recovery)")
_R_ZERO = _R.replace("    level = 100\n", "    level = 0\n")
_R_BEFORE = _R.replace("        level = min(100, level - harm + recovery)\n        result.append(level)\n",
                       "        result.append(level)\n        level = min(100, level - harm + recovery)\n")
_R_LAST = _R.replace("    return result\n", "    return level\n")
_R_EARLY = _R.replace("    return result\n", "        return result\n")

# exercise 2: collapse_year; the first case is the resilience of exercise 1's first case and a shock of 60, then a
# year exactly at the shock, a society that absorbs it every year, no years, a collapse in the first year, and a
# bigger shock
YEAR_CASES = [(resilience(HARMS, RECOVERY), SHOCK), ([80, 70, 60, 50], 60), ([100, 100], 60), ([], 60),
              ([50, 90], 60), ([95, 90, 85], 90)]
_Y_START = "def collapse_year(levels, shock):\n    ...\n"
_Y = ("def collapse_year(levels, shock):\n"
      "    year = 0\n"
      "    for level in levels:\n"
      "        year = year + 1\n"
      "        if level <= shock:\n"
      "            return year\n"
      "    return None\n")
_Y_STRICT = _Y.replace("level <= shock", "level < shock")
_Y_AFTER = _Y.replace("        year = year + 1\n        if level <= shock:\n            return year\n",
                      "        if level <= shock:\n            return year\n        year = year + 1\n")
_Y_LEVEL = _Y.replace("            return year\n", "            return level\n")
_Y_SURVIVE = _Y.replace("level <= shock", "level >= shock")
_Y_NOUGHT = _Y.replace("    return None\n", "    return 0\n")
_Y_YEAR = _Y.replace("    return None\n", "    return year\n")


EXERCISES = {
    "resilience": FunctionBox(
        name="resilience",
        picture=resilience_picture,
        function="resilience",
        start=_R_START,
        solution=_R,
        visible=[LEVEL_CASES[0]],
        hidden=LEVEL_CASES[1:],
        first=[_DOTS],
        show=show_levels,
        mistakes=[
            Mistake("reset", wrong=_R_RESET,
                    en="`{call}` gives {got}, and it should give {want}: each year starts again from 100, so no harm builds up. Start each year from where the year before ended: from `level`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk jaar begint weer bij 100, dus er bouwt zich geen schade op. Laat elk jaar beginnen waar het jaar ervoor eindigde: bij `level`."),
            Mistake("cap", wrong=_R_CAP,
                    en="`{call}` gives {got}, and it should give {want}: in a good year, your resilience rises above 100. A society cannot become more resilient than a healthy one: keep it at 100 at most, as in the worked example.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in een goed jaar stijgt je veerkracht boven 100. Een samenleving kan niet veerkrachtiger worden dan een gezonde samenleving: houd de veerkracht op hoogstens 100, zoals in het uitgewerkte voorbeeld."),
            Mistake("recovery", wrong=_R_NOREC,
                    en="`{call}` gives {got}, and it should give {want}: your society never wins anything back. Each year, the society also wins back `recovery`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je samenleving wint nooit iets terug. Elk jaar komt er ook `recovery` bij."),
            Mistake("sign", wrong=_R_SIGN,
                    en="`{call}` gives {got}, and it should give {want}: you add the harm and take off the recovery. A harm takes resilience away, and the recovery gives some back.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt de schade op en haalt het herstel eraf. Schade neemt veerkracht weg, en het herstel geeft wat terug."),
            Mistake("zero", wrong=_R_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: your resilience starts at 0. A healthy society starts at 100.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je veerkracht begint bij 0. Een gezonde samenleving begint bij 100."),
            Mistake("before", wrong=_R_BEFORE,
                    en="`{call}` gives {got}, and it should give {want}: you append the resilience before the year's harm and recovery, so each year shows the year before. Work out the new resilience first, then append it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je voegt de veerkracht toe vóór de schade en het herstel van dat jaar, dus elk jaar laat het jaar ervoor zien. Reken eerst de nieuwe veerkracht uit, en voeg die dan toe."),
            Mistake("last", wrong=_R_LAST,
                    en="`{call}` gives {got}, and it should give {want}: you return only the resilience of the last year. Return the list `result`, with a resilience for each year.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft alleen de veerkracht van het laatste jaar terug. Geef de lijst `result` terug, met een veerkracht voor elk jaar."),
            Mistake("early", wrong=_R_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your `return` stands inside the loop, so the function stops after the first year. Put `return result` after the loop, level with `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de functie stopt na het eerste jaar. Zet `return result` na de loop, op gelijke hoogte met `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start `level` at 100, and for each `harm` in `harms`, work out the resilience at the end of that year and append it to a list `result`.",
                      "nl": "Vervang de `...`: laat `level` bij 100 beginnen, reken voor elke `harm` in `harms` de veerkracht aan het eind van dat jaar uit, en voeg die toe aan een lijst `result`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Begin `result` as an empty list and `level` at 100. In a `for` loop over `harms`, work out the new `level` from the old one, the year's `harm` and `recovery`, never above 100, as in the worked example, and append it; after the loop, return `result`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begin `result` als een lege lijst en `level` bij 100. Reken in een `for`-loop over `harms` de nieuwe `level` uit met de oude, de `harm` van dat jaar en `recovery`, nooit boven 100, zoals in het uitgewerkte voorbeeld, en voeg die toe; geef na de loop `result` terug."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `harms` and `recovery` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie wel de `harms` en `recovery` uit de aanroep?"},
            "correct": {"en": "Your `resilience` is right in all {n} cases. Under the check, your rings lie on the course's line.",
                        "nl": "Je `resilience` klopt in alle {n} gevallen. Onder de check liggen jouw ringen op de lijn van de course."},
        },
    ),
    "collapse": FunctionBox(
        name="collapse",
        picture=collapse_picture,
        function="collapse_year",
        start=_Y_START,
        solution=_Y,
        visible=[YEAR_CASES[0]],
        hidden=YEAR_CASES[1:],
        first=[_DOTS],
        show=show_year,
        mistakes=[
            Mistake("strict", wrong=_Y_STRICT,
                    en="`{call}` gives {got}, and it should give {want}: at a resilience equal to the shock, the shock takes it to exactly 0, and 0 counts as well.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: bij een veerkracht gelijk aan de schok brengt de schok die op precies 0, en 0 telt ook."),
            Mistake("after", wrong=_Y_AFTER,
                    en="`{call}` gives {got}, and it should give {want}: your years start at 0. The first year is year 1: add 1 to `year` before you look at the level.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je jaren beginnen bij 0. Het eerste jaar is jaar 1: tel 1 op bij `year` voordat je naar de veerkracht kijkt."),
            Mistake("level", wrong=_Y_LEVEL,
                    en="`{call}` gives {got}, and it should give {want}: you return the resilience of that year. Return the year itself, `year`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft de veerkracht van dat jaar terug. Geef het jaar zelf terug, `year`."),
            Mistake("survive", wrong=_Y_SURVIVE,
                    en="`{call}` gives {got}, and it should give {want}: you return the first year in which the society would absorb the shock. Return the first year in which the shock would take its resilience to 0 or below.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft het eerste jaar terug waarin de samenleving de schok zou opvangen. Geef het eerste jaar terug waarin de schok de veerkracht op 0 of lager zou brengen."),
            Mistake("nought", wrong=_Y_NOUGHT,
                    en="`{call}` gives {got}, and it should give {want}: when the shock topples the society in no year, you return 0. Return `None`: there is no such year.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: als de schok de samenleving in geen enkel jaar omverwerpt, geef je 0 terug. Geef `None` terug: zo'n jaar is er niet."),
            Mistake("year", wrong=_Y_YEAR,
                    en="`{call}` gives {got}, and it should give {want}: when the shock topples the society in no year, you return the number of years. Return `None`: there is no such year.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: als de schok de samenleving in geen enkel jaar omverwerpt, geef je het aantal jaren terug. Geef `None` terug: zo'n jaar is er niet."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: count the years from 1 in a `for` loop over `levels`, return the first year in which the shock would take the resilience to 0 or below, and return `None` after the loop.",
                      "nl": "Vervang de `...`: tel de jaren vanaf 1 in een `for`-loop over `levels`, geef het eerste jaar terug waarin de schok de veerkracht op 0 of lager zou brengen, en geef na de loop `None` terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Begin `year` at 0. In a `for` loop over `levels`, add 1 to `year`, and return `year` as soon as the shock would take that year's `level` to 0 or below; after the loop, return `None`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begin `year` bij 0. Tel in een `for`-loop over `levels` 1 op bij `year`, en geef `year` terug zodra de schok de `level` van dat jaar op 0 of lager zou brengen; geef na de loop `None` terug."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `levels` and `shock` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie wel de `levels` en `shock` uit de aanroep?"},
            "correct": {"en": "Your `collapse_year` is right in all {n} cases. Under the check, your ring lies around the course's dot.",
                        "nl": "Je `collapse_year` klopt in alle {n} gevallen. Onder de check ligt jouw ring om de stip van de course."},
        },
    ),
}

CASES = [
    ("resilience", _R_START, "empty", None),
    ("resilience", _R, "correct", None),
    ("resilience", "def resilience(harms, recovery):\n    result = []\n    level = 100\n    for harm in harms:\n        level = level - harm + recovery\n        if level > 100:\n            level = 100\n        result.append(level)\n    return result\n", "correct", None),
    ("resilience", "def resilience(harms, recovery):\n    result = []\n    for k in range(len(harms)):\n        before = result[-1] if result else 100\n        result.append(min(100, before - harms[k] + recovery))\n    return result\n", "correct", None),
    ("resilience", _R_RESET, "wrong", "reset"),
    ("resilience", _R_CAP, "wrong", "cap"),
    ("resilience", _R_NOREC, "wrong", "recovery"),
    ("resilience", _R_SIGN, "wrong", "sign"),
    ("resilience", _R_ZERO, "wrong", "zero"),
    ("resilience", _R_BEFORE, "wrong", "before"),
    ("resilience", _R_LAST, "wrong", "last"),
    ("resilience", _R_EARLY, "wrong", "early"),
    ("resilience", "def resilience(harms, recovery):\n    return " + repr(resilience(HARMS, RECOVERY)) + "\n", "almost", None),
    ("collapse", _Y_START, "empty", None),
    ("collapse", _Y, "correct", None),
    ("collapse", "def collapse_year(levels, shock):\n    for year, level in enumerate(levels, start=1):\n        if level - shock <= 0:\n            return year\n", "correct", None),
    ("collapse", "def collapse_year(levels, shock):\n    for k in range(len(levels)):\n        if levels[k] <= shock:\n            return k + 1\n    return None\n", "correct", None),
    ("collapse", _Y_STRICT, "wrong", "strict"),
    ("collapse", _Y_AFTER, "wrong", "after"),
    ("collapse", _Y_LEVEL, "wrong", "level"),
    ("collapse", _Y_SURVIVE, "wrong", "survive"),
    ("collapse", _Y_NOUGHT, "wrong", "nought"),
    ("collapse", _Y_YEAR, "wrong", "year"),
    ("collapse", "def collapse_year(levels, shock):\n    return " + repr(collapse_year(resilience(HARMS, RECOVERY), SHOCK)) + "\n", "almost", None),
]
