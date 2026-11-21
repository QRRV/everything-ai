# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-xrisk-counterpositions: other positions. The learner writes exposure, the years each of
a few made-up dangerous technologies is out before the protection against it arrives, a made-up model of differential
technology development, and progress, the progress made by the end of each year when each year adds the share of the
world's compute that is available, a made-up model of Buterin's soft pause. A demo brings every protection sooner."""
import contextlib
import copy
import html
import io
import re

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, same_value, with_time_limit

# --- the data --------------------------------------------------------------------------------------------------------

RISKS = [3, 6, 7, 9, 11]     # made-up: the year from now in which each of five dangerous technologies arrives
GUARDS = [6, 5, 12, 10, 11]  # made-up: the year in which the protection against each of them arrives
SHARES = [1, 1, 0.05, 0.05, 1, 1, 1, 1]   # made-up: the share of the world's compute in each of eight years; a pause
#                                           of 2 years at 95% less, within Buterin's "~90-99%" for "1-2 years"
DEMO_SOONER = [0, 1, 2, 3, 4, 5, 6]       # the stops of the demo's slider: how many years sooner every protection comes


def exposure(risks, guards):
    """What the learner writes in Exercise 1: for each pair, the years the dangerous technology is out before the
    protection against it arrives, and 0 when the protection is there first."""
    result = []
    for k in range(len(risks)):
        result.append(max(0, guards[k] - risks[k]))
    return result


def progress(shares):
    """What the learner writes in Exercise 2: the progress made by the end of each year, when each year adds the share
    of the world's compute that is available in it."""
    result = []
    total = 0
    for share in shares:
        total = total + share
        result.append(total)
    return result


def demo_guards(sooner):
    """The demo: the year each protection arrives when every protection comes `sooner` years sooner."""
    return [g - sooner for g in GUARDS]


def demo_total(sooner):
    """The demo: the unguarded years of the five pairs together."""
    return sum(exposure(RISKS, demo_guards(sooner)))


def first_none(values=DEMO_SOONER):
    """The smallest value of the slider at which no dangerous technology is unguarded."""
    return next(s for s in values if demo_total(s) == 0)


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


def show_list(v) -> str:
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
    "en": {"title": "Unguarded years when every protection comes {s} sooner",
           "title0": "Unguarded years when every protection comes on its own schedule",
           "year": "{n} year", "years": "{n} years",
           "x": "years from now", "y": "pair", "danger": "danger arrives", "guard": "protection arrives",
           "said": "With every protection {s} sooner, the five dangerous technologies are unguarded for {t} in all.",
           "said0": "With every protection on its own schedule, the five dangerous technologies are unguarded for {t} "
                    "in all.",
           "none": "With every protection {s} sooner, no dangerous technology is unguarded: each protection is there by "
                   "the time its danger arrives.",
           "same": "If every technology, dangers and protections alike, came {s} sooner, the gaps would stay the same: "
                   "still 9 years in all.",
           "note": "A made-up model, not Bostrom's or Sandbrink and colleagues': five pairs of a dangerous technology "
                   "and the protection against it. A dot is the year the danger arrives, a ring the year its "
                   "protection arrives, and a thick line the years in between, when the danger is unguarded."},
    "nl": {"title": "Onbeschermde jaren als elke bescherming {s} eerder komt",
           "title0": "Onbeschermde jaren als elke bescherming volgens de eigen planning komt",
           "year": "{n} jaar", "years": "{n} jaar",
           "x": "jaren vanaf nu", "y": "paar", "danger": "gevaar komt", "guard": "bescherming komt",
           "said": "Als elke bescherming {s} eerder komt, zijn de vijf gevaarlijke technologieën samen {t} onbeschermd.",
           "said0": "Als elke bescherming volgens de eigen planning komt, zijn de vijf gevaarlijke technologieën samen "
                    "{t} onbeschermd.",
           "none": "Als elke bescherming {s} eerder komt, is geen enkele gevaarlijke technologie onbeschermd: elke "
                   "bescherming is er op tijd.",
           "same": "Als elke technologie, gevaren en beschermingen allebei, {s} eerder kwam, bleven de gaten even groot: "
                   "samen nog steeds 9 jaar.",
           "note": "Een verzonnen model, niet dat van Bostrom of van Sandbrink en anderen: vijf paren van een gevaarlijke "
                   "technologie en de bescherming ertegen. Een stip is het jaar waarin het gevaar komt, een ring het "
                   "jaar waarin de bescherming komt, en een dikke lijn de jaren ertussen, als het gevaar onbeschermd "
                   "is."},
}


def _years(n, w) -> str:
    return (w["year"] if n == 1 else w["years"]).format(n=n)


def demo_said(sooner, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    total = demo_total(sooner)
    if sooner == 0:
        return w["said0"].format(t=_years(total, w))
    if total == 0:
        return w["none"].format(s=_years(sooner, w))
    return w["said"].format(s=_years(sooner, w), t=_years(total, w))


def demo_frame(v, lang: str) -> tuple[str, str]:
    """The demo: each pair as a row, with the year its danger arrives, the year its protection arrives when every
    protection comes the chosen number of years sooner, and the unguarded years between them."""
    sooner = min(DEMO_SOONER, key=lambda k: abs(k - float(v)))
    w = _DEMO_WORDS[lang]
    guards = demo_guards(sooner)
    panel = chart.Panel(x=(-1.6, 13), y=(0.4, 6.2), xlabel=w["x"], ylabel=w["y"], xticks=[0, 2, 4, 6, 8, 10, 12],
                        yticks=[1, 2, 3, 4, 5])
    for k in range(len(RISKS)):
        row = len(RISKS) - k   # pair 1 at the top
        if guards[k] > RISKS[k]:
            panel.line([RISKS[k], guards[k]], [row, row], role="loss", width=5)
        panel.points([guards[k]], [row], role="ink", r=6, hollow=True)
        panel.points([RISKS[k]], [row], role="ink", r=3.5)
    panel.points([0.2], [5.85], role="ink", r=3.5)
    panel.text(0.6, 5.85, w["danger"], role="ink-2", anchor="start", halo=True)
    panel.points([6.2], [5.85], role="ink", r=6, hollow=True)
    panel.text(6.7, 5.85, w["guard"], role="ink-2", anchor="start", halo=True)
    title = w["title0"] if sooner == 0 else w["title"].format(s=_years(sooner, w))
    said = demo_said(sooner, lang)
    svg = chart.figure([panel], alt=title + ". " + said, width=380, height=240)
    if lang == "nl":
        svg = chart.with_decimal_comma(svg)
    lines = [said] + ([w["same"].format(s=_years(sooner, w))] if sooner else [])
    body = svg + "".join(_note(t) for t in lines) + _note(w["note"])
    return yours.figure(title, body, alt=title + ". " + said), said


# --- the pictures under the checks -----------------------------------------------------------------------------------

_EXPOSURE_WORDS = {
    "en": {"title": "Your unguarded years for the five made-up pairs",
           "x": "pair", "y": "unguarded years",
           "head": ["risks", "guards", "the course's", "yours"], "stopped": "error",
           "note": "The chart is the first case of the check: five made-up pairs of a dangerous technology and the "
                   "protection against it. A dot is the course's number of unguarded years for each pair and a ring is "
                   "yours. The table is every case; a dashed row is one where your function gives something other "
                   "than the course's."},
    "nl": {"title": "Jouw onbeschermde jaren voor de vijf verzonnen paren",
           "x": "paar", "y": "onbeschermde jaren",
           "head": ["risks", "guards", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De grafiek is het eerste geval van de check: vijf verzonnen paren van een gevaarlijke technologie en "
                   "de bescherming ertegen. Een stip is het aantal onbeschermde jaren van de course voor elk paar en "
                   "een ring het jouwe. De tabel bevat elk geval; een gestippelde rij is er een waar jouw functie iets "
                   "anders geeft dan die van de course."},
}


def exposure_picture(seen) -> str:
    """The first case as the unguarded years of each pair, the course's as dots and the learner's as rings, and every
    case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _EXPOSURE_WORDS[seen.lang]
    got = _all(fn, EXPOSURE_CASES)
    rows = []
    for i, case in enumerate(EXPOSURE_CASES):
        want = exposure(*case)
        if got[i] is _STOPPED:
            shown_, off = html.escape(w["stopped"]), True
        else:
            shown_, off = _code(show_list(got[i])), not _same(got[i], want)
        rows.append(([(_code(_value(case[0])), ""), (_code(_value(case[1])), ""),
                      (_code(show_list(want)), ""), (shown_, "font-weight:600")], _OFF if off else ""))
    want = exposure(RISKS, GUARDS)
    xs = list(range(1, len(RISKS) + 1))
    panel = chart.Panel(x=(0.5, 5.5), y=(-1.4, 6.5), xlabel=w["x"], ylabel=w["y"], xticks=xs,
                        yticks=[-1, 0, 1, 2, 3, 4, 5, 6])   # room below 0 for a negative ring
    for x, y in zip(xs, want):
        panel.line([x, x], [0, y], role="ink-3", width=1.6)
    panel.points(*_rings(got[0], xs, -1.4, 6.5), role="ink", r=7, hollow=True)
    panel.points(xs, want, role="ink", r=3)
    svg = chart.figure([panel], alt=w["title"], width=380, height=240)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


_PROGRESS_WORDS = {
    "en": {"title": "Your progress over eight made-up years with a soft pause",
           "x": "year", "y": "progress", "pause": "pause", "without": "without a pause",
           "head": ["shares", "the course's", "yours"], "stopped": "error",
           "note": "The chart is the first case of the check: eight made-up years with a pause of 2 years at 95% less "
                   "compute. A dot is the course's progress at the end of each year and a ring is yours; the dashed "
                   "line is the progress without a pause. The table is every case; a dashed row is one where your "
                   "function gives something other than the course's."},
    "nl": {"title": "Jouw vooruitgang over acht verzonnen jaren met een zachte pauze",
           "x": "jaar", "y": "vooruitgang", "pause": "pauze", "without": "zonder pauze",
           "head": ["shares", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De grafiek is het eerste geval van de check: acht verzonnen jaren met een pauze van 2 jaar met 95% "
                   "minder compute. Een stip is de vooruitgang van de course aan het eind van elk jaar en een ring de "
                   "jouwe; de gestippelde lijn is de vooruitgang zonder pauze. De tabel bevat elk geval; een "
                   "gestippelde rij is er een waar jouw functie iets anders geeft dan die van de course."},
}


def progress_picture(seen) -> str:
    """The first case as the progress at the end of each year, the course's as dots and the learner's as rings, the
    progress without a pause as a dashed line, and every case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _PROGRESS_WORDS[seen.lang]
    got = _all(fn, PROGRESS_CASES)
    rows = []
    for i, case in enumerate(PROGRESS_CASES):
        want = progress(*case)
        if got[i] is _STOPPED:
            shown_, off = html.escape(w["stopped"]), True
        else:
            shown_, off = _code(show_list(got[i])), not _same(got[i], want)
        rows.append(([(_code(_value(case[0])), ""), (_code(show_list(want)), ""), (shown_, "font-weight:600")],
                     _OFF if off else ""))
    want = progress(SHARES)
    xs = list(range(1, len(SHARES) + 1))
    panel = chart.Panel(x=(0.5, 8.5), y=(0, 9), xlabel=w["x"], ylabel=w["y"], xticks=xs, yticks=[0, 2, 4, 6, 8])
    panel.line(xs, xs, role="ink-3", width=1.4, dashed=True)
    panel.text(5.4, 6.9, w["without"], role="ink-2", anchor="end", halo=True)
    panel.text(3.5, 3.1, w["pause"], role="ink-2", anchor="middle", halo=True)
    panel.line(xs, want, role="ink-3")
    panel.points(*_rings(got[0], xs, 0, 9), role="ink", r=7, hollow=True)
    panel.points(xs, want, role="ink", r=3)
    svg = chart.figure([panel], alt=w["title"], width=380, height=240)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["exposure", "progress", "normal", "dacc"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# exercise 1: exposure; the first case is the five made-up pairs, then a pair with its protection first, a lone pair
# with its protection first, no pairs, three dangers in the same year, and two pairs the other way round
EXPOSURE_CASES = [(RISKS, GUARDS), ([1, 2], [4, 2]), ([5], [3]), ([], []), ([2, 2, 2], [3, 5, 1]), ([0, 4], [4, 0])]
_E_START = "def exposure(risks, guards):\n    ...\n"
_E = ("def exposure(risks, guards):\n"
      "    result = []\n"
      "    for k in range(len(risks)):\n"
      "        result.append(max(0, guards[k] - risks[k]))\n"
      "    return result\n")
_E_SIGN = _E.replace("max(0, guards[k] - risks[k])", "max(0, risks[k] - guards[k])")
_E_NOMAX = _E.replace("max(0, guards[k] - risks[k])", "guards[k] - risks[k]")
_E_ABS = _E.replace("max(0, guards[k] - risks[k])", "abs(guards[k] - risks[k])")
_E_MIN = _E.replace("max(0, guards[k] - risks[k])", "min(0, guards[k] - risks[k])")
_E_GUARD = _E.replace("max(0, guards[k] - risks[k])", "guards[k]")
_E_FIRST = _E.replace("max(0, guards[k] - risks[k])", "max(0, guards[0] - risks[0])")
_E_EARLY = _E.replace("    return result\n", "        return result\n")

# exercise 2: progress; the first case is eight made-up years with a soft pause of 2 years at 95% less compute, then
# three normal years, two years at half, no years, one year at 1%, and four years at a tenth
PROGRESS_CASES = [(SHARES,), ([1, 1, 1],), ([0.5, 0.5],), ([],), ([1, 0.01, 1],), ([0.1, 0.1, 0.1, 0.1],)]
_P_START = "def progress(shares):\n    ...\n"
_P = ("def progress(shares):\n"
      "    result = []\n"
      "    total = 0\n"
      "    for share in shares:\n"
      "        total = total + share\n"
      "        result.append(total)\n"
      "    return result\n")
_P_SHARE = _P.replace("        result.append(total)\n", "        result.append(share)\n")
_P_ONE = _P.replace("total = total + share", "total = total + 1")
_P_START1 = _P.replace("    total = 0\n", "    total = 1\n")
_P_BEFORE = _P.replace("        total = total + share\n        result.append(total)\n",
                       "        result.append(total)\n        total = total + share\n")
_P_TIMES = _P.replace("total = total + share", "total = total * share")
_P_LAST = _P.replace("    return result\n", "    return total\n")
_P_EARLY = _P.replace("    return result\n", "        return result\n")


EXERCISES = {
    "exposure": FunctionBox(
        name="exposure",
        picture=exposure_picture,
        function="exposure",
        start=_E_START,
        solution=_E,
        visible=[EXPOSURE_CASES[0]],
        hidden=EXPOSURE_CASES[1:],
        first=[_DOTS],
        show=show_list,
        mistakes=[
            Mistake("sign", wrong=_E_SIGN,
                    en="`{call}` gives {got}, and it should give {want}: you count the years the protection is there before its danger. Count the years from the danger's arrival until its protection arrives.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt de jaren dat de bescherming er al is vóór het gevaar. Tel de jaren vanaf de komst van het gevaar tot de bescherming komt."),
            Mistake("nomax", wrong=_E_NOMAX,
                    en="`{call}` gives {got}, and it should give {want}: when a protection arrives before its danger, you give a negative number. A danger cannot be unguarded for less than 0 years: keep it at 0 at least, as in the worked example.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: als een bescherming vóór het gevaar komt, geef je een negatief getal. Een gevaar kan niet minder dan 0 jaar onbeschermd zijn: houd het op minstens 0, zoals in het uitgewerkte voorbeeld."),
            Mistake("abs", wrong=_E_ABS,
                    en="`{call}` gives {got}, and it should give {want}: you also count the years a protection is ready before its danger. Those years are not unguarded: such a pair gets 0.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt ook de jaren dat een bescherming al klaar is vóór het gevaar. Die jaren zijn niet onbeschermd: zo'n paar krijgt 0."),
            Mistake("min", wrong=_E_MIN,
                    en="`{call}` gives {got}, and it should give {want}: you keep the number at 0 at most, so a danger is never unguarded. Keep it at 0 at least, as in the worked example.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je houdt het getal op hoogstens 0, dus een gevaar is nooit onbeschermd. Houd het op minstens 0, zoals in het uitgewerkte voorbeeld."),
            Mistake("guard", wrong=_E_GUARD,
                    en="`{call}` gives {got}, and it should give {want}: you give the year each protection arrives. Give the years each danger is out before its protection arrives.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft het jaar waarin elke bescherming komt. Geef de jaren dat elk gevaar er is voordat de bescherming komt."),
            Mistake("first", wrong=_E_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: you look at the first pair every time. Use `risks[k]` and `guards[k]`, the pair of this round of the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je kijkt elke keer naar het eerste paar. Gebruik `risks[k]` en `guards[k]`, het paar van deze ronde van de loop."),
            Mistake("early", wrong=_E_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your `return` stands inside the loop, so the function stops after the first pair. Put `return result` after the loop, level with `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de functie stopt na het eerste paar. Zet `return result` na de loop, op gelijke hoogte met `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: for each pair, work out the years its danger is out before its protection arrives, never below 0, and append them to a list `result`.",
                      "nl": "Vervang de `...`: reken voor elk paar uit hoeveel jaar het gevaar er is voordat de bescherming komt, nooit minder dan 0, en voeg dat toe aan een lijst `result`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Begin `result` as an empty list. In a `for` loop with `k` over `range(len(risks))`, work out the unguarded years of pair `k` from `risks[k]` and `guards[k]`, never below 0, as in the worked example, and append them; after the loop, return `result`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begin `result` als een lege lijst. Reken in een `for`-loop met `k` over `range(len(risks))` de onbeschermde jaren van paar `k` uit met `risks[k]` en `guards[k]`, nooit minder dan 0, zoals in het uitgewerkte voorbeeld, en voeg die toe; geef na de loop `result` terug."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `risks` and `guards` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie wel de `risks` en `guards` uit de aanroep?"},
            "correct": {"en": "Your `exposure` is right in all {n} cases. Under the check, your rings lie on the course's dots.",
                        "nl": "Je `exposure` klopt in alle {n} gevallen. Onder de check liggen jouw ringen om de stippen van de course."},
        },
    ),
    "progress": FunctionBox(
        name="progress",
        picture=progress_picture,
        function="progress",
        start=_P_START,
        solution=_P,
        visible=[PROGRESS_CASES[0]],
        hidden=PROGRESS_CASES[1:],
        first=[_DOTS],
        show=show_list,
        mistakes=[
            Mistake("share", wrong=_P_SHARE,
                    en="`{call}` gives {got}, and it should give {want}: you give each year's share, not the progress so far. Append `total`, which keeps everything up to and including that year.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft het aandeel van elk jaar, niet de vooruitgang tot nu toe. Voeg `total` toe, waarin alles tot en met dat jaar zit."),
            Mistake("one", wrong=_P_ONE,
                    en="`{call}` gives {got}, and it should give {want}: every year of yours adds 1, also in a pause. A year adds the share of compute available in it: `share`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: bij jou levert elk jaar 1 op, ook in een pauze. Een jaar levert het aandeel compute op dat er dat jaar is: `share`."),
            Mistake("start", wrong=_P_START1,
                    en="`{call}` gives {got}, and it should give {want}: your progress starts at 1. Before the first year there is no progress yet: start `total` at 0.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je vooruitgang begint bij 1. Vóór het eerste jaar is er nog geen vooruitgang: laat `total` bij 0 beginnen."),
            Mistake("before", wrong=_P_BEFORE,
                    en="`{call}` gives {got}, and it should give {want}: you append the progress before the year's share, so each year shows the year before. Add the share first, then append.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je voegt de vooruitgang toe vóór het aandeel van dat jaar, dus elk jaar laat het jaar ervoor zien. Tel eerst het aandeel erbij, en voeg dan toe."),
            Mistake("times", wrong=_P_TIMES,
                    en="`{call}` gives {got}, and it should give {want}: your progress stays at 0. Each year's share comes on top of what was there: it does not scale it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je vooruitgang blijft 0. Het aandeel van elk jaar komt bovenop wat er al was: het vergroot of verkleint dat niet."),
            Mistake("last", wrong=_P_LAST,
                    en="`{call}` gives {got}, and it should give {want}: you return only the progress at the end. Return the list `result`, with the progress at the end of each year.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft alleen de vooruitgang aan het eind terug. Geef de lijst `result` terug, met de vooruitgang aan het eind van elk jaar."),
            Mistake("early", wrong=_P_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your `return` stands inside the loop, so the function stops after the first year. Put `return result` after the loop, level with `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de functie stopt na het eerste jaar. Zet `return result` na de loop, op gelijke hoogte met `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start `total` at 0, and for each `share` in `shares`, work out the progress at the end of that year and append it to a list `result`.",
                      "nl": "Vervang de `...`: laat `total` bij 0 beginnen, reken voor elke `share` in `shares` de vooruitgang aan het eind van dat jaar uit, en voeg die toe aan een lijst `result`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Begin `result` as an empty list and `total` at 0. In a `for` loop over `shares`, let each year's `share` come on top of `total`, then append `total`; after the loop, return `result`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begin `result` als een lege lijst en `total` bij 0. Laat in een `for`-loop over `shares` de `share` van elk jaar bovenop `total` komen, en voeg dan `total` toe; geef na de loop `result` terug."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `shares` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie wel de `shares` uit de aanroep?"},
            "correct": {"en": "Your `progress` is right in all {n} cases. Under the check, your rings lie on the course's line.",
                        "nl": "Je `progress` klopt in alle {n} gevallen. Onder de check liggen jouw ringen op de lijn van de course."},
        },
    ),
}

CASES = [
    ("exposure", _E_START, "empty", None),
    ("exposure", _E, "correct", None),
    ("exposure", "def exposure(risks, guards):\n    result = []\n    for risk, guard in zip(risks, guards):\n        if guard > risk:\n            result.append(guard - risk)\n        else:\n            result.append(0)\n    return result\n", "correct", None),
    ("exposure", "def exposure(risks, guards):\n    return [max(0, g - r) for r, g in zip(risks, guards)]\n", "correct", None),
    ("exposure", _E_SIGN, "wrong", "sign"),
    ("exposure", _E_NOMAX, "wrong", "nomax"),
    ("exposure", _E_ABS, "wrong", "abs"),
    ("exposure", _E_MIN, "wrong", "min"),
    ("exposure", _E_GUARD, "wrong", "guard"),
    ("exposure", _E_FIRST, "wrong", "first"),
    ("exposure", _E_EARLY, "wrong", "early"),
    ("exposure", "def exposure(risks, guards):\n    return " + repr(exposure(RISKS, GUARDS)) + "\n", "almost", None),
    ("progress", _P_START, "empty", None),
    ("progress", _P, "correct", None),
    ("progress", "def progress(shares):\n    result = []\n    for k in range(len(shares)):\n        result.append(sum(shares[:k + 1]))\n    return result\n", "correct", None),
    ("progress", "def progress(shares):\n    result = []\n    for share in shares:\n        before = result[-1] if result else 0\n        result.append(before + share)\n    return result\n", "correct", None),
    ("progress", _P_SHARE, "wrong", "share"),
    ("progress", _P_ONE, "wrong", "one"),
    ("progress", _P_START1, "wrong", "start"),
    ("progress", _P_BEFORE, "wrong", "before"),
    ("progress", _P_TIMES, "wrong", "times"),
    ("progress", _P_LAST, "wrong", "last"),
    ("progress", _P_EARLY, "wrong", "early"),
    ("progress", "def progress(shares):\n    return " + repr(progress(SHARES)) + "\n", "almost", None),
]
