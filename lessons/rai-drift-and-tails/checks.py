# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-drift-and-tails: drifting into failure, and black swans. The learner writes stretch, the
accepted limit that grows with every deviation that caused no accident (the 'elastic waistband' of the Columbia
report), and running_mean, the mean after each value. A demo draws heights (a thin tail) and incident sizes (a long
tail) with NumPy and shows how one rare incident moves the mean."""
import contextlib
import copy
import functools
import html
import io
import math
import re

import numpy as np

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the data --------------------------------------------------------------------------------------------------------

SEED = 7                                         # the seed of the demo's random generator
DRAWS = [10, 100, 1000, 10000, 100000, 1000000]  # the demo's slider: how many draws of each
SHAPE = 1.5                                      # the long tail: (1 - r) ** (-1 / SHAPE), Newman's exponent 2.5
LONG_RUN = 3                                     # the long-run mean of that recipe, (2.5 - 1) / (2.5 - 2) by Newman


@functools.lru_cache(maxsize=1)
def samples() -> tuple:
    """The demo's draws: 1,000,000 heights in cm, each 140 plus 80 times the mean of 12 random numbers (a thin tail
    around 180), then 1,000,000 incident sizes by Newman's recipe (a long tail from 1). Made once, then kept."""
    rng = np.random.default_rng(SEED)
    n = DRAWS[-1]
    heights = 140 + 80 * rng.random(size=(n, 12)).mean(axis=1)
    sizes = (1 - rng.random(size=n)) ** (-1 / SHAPE)
    return heights, sizes


def stretch(deviations, design):
    """What the learner writes in Exercise 1: the accepted limit after each flight, starting at the design limit; a
    deviation above the limit that caused no accident becomes the new limit."""
    limit = design
    accepted = []
    for deviation in deviations:
        if deviation > limit:
            limit = deviation
        accepted.append(limit)
    return accepted


def running_mean(values):
    """What the learner writes in Exercise 2: the mean of the first one, two, three ... values."""
    total = 0
    count = 0
    means = []
    for value in values:
        total = total + value
        count = count + 1
        means.append(total / count)
    return means


def lottery(win_chance: float, win: float = 15, loss: float = 10000) -> float:
    """The textbook's lottery: win `win` dollars with chance `win_chance`, else lose `loss`; the expected value."""
    return win_chance * win + (1 - win_chance) * -loss


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
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError, OverflowError):
        return None


def _value(v) -> str:
    """A result as the tables show it: a number with at most four decimals, a list of them, else what Python shows,
    cut short when it is long."""
    if isinstance(v, list):
        text = "[" + ", ".join(_value(x) for x in v) + "]"
    else:
        f = _number(v)
        if f is None or f != f or f in (float("inf"), float("-inf")):
            text = repr(v) if f is None else str(f)
        else:
            text = f"{f:.4f}".rstrip("0").rstrip(".")
            text = "0" if text == "-0" else text
    return text if len(text) <= 48 else text[:45] + "..."


def _same(got, want) -> bool:
    """Whether a list the learner gave holds the same numbers as the course's, up to rounding."""
    if not isinstance(got, list) or len(got) != len(want):
        return False
    return all(_number(g) is not None and math.isclose(g, w, rel_tol=1e-6, abs_tol=1e-9) for g, w in zip(got, want))


def _prose(x: float, lang: str, digits: int = 1) -> str:
    """A number in a picture's words, rounded to `digits` decimals: 1,000 and 3.5 in English, 1000 and 3,5 in Dutch,
    which uses a dot only from 10.000."""
    x = round(float(x), digits)
    if digits == 0 or x == int(x) and abs(x) >= 1000:
        n = int(round(x))
        if lang == "en":
            return f"{n:,}"
        return f"{n:,}".replace(",", ".") if abs(n) >= 10000 else str(n)
    text = f"{x:,.{digits}f}"
    if lang == "nl":
        whole, _, part = text.partition(".")
        whole = whole.replace(",", "")
        if abs(int(whole)) >= 10000:
            whole = f"{int(whole):,}".replace(",", ".")
        return whole + "," + part
    return text


def _percent(x: float, lang: str) -> str:
    """A share as a whole percentage, or with one decimal below 10%: 26% or 0.3%."""
    p = x * 100
    text = f"{p:.0f}%" if p >= 9.95 else f"{p:.1f}%"
    return text.replace(".", ",") if lang == "nl" else text


_DEMO_WORDS = {
    "en": {"title": "{n} draws of each: a thin tail and a long tail",
           "head": ["", "mean", "median", "largest", "largest's share of the total"],
           "rows": ["heights in cm, thin tail", "incident sizes, long tail"],
           "said": "With {n} draws, the heights have mean {hm} cm and their largest is {hx} cm; the incident sizes have "
                   "mean {sm} and median {sd}, and their largest, {sx}, is {share} of their total.",
           "x": "draws so far", "y": "mean of the incident sizes", "run": "mean so far", "long": "long-run mean, 3",
           "note": "The heights are 140 plus 80 times the mean of 12 random numbers; the incident sizes follow Newman's "
                   "recipe with exponent 2.5. The same seed gives the same draws, so a bigger number on the slider adds "
                   "draws to the ones before. The line is the mean of the incident sizes after each draw, with the "
                   "draws on a log scale."},
    "nl": {"title": "{n} trekkingen van elk: een dunne staart en een lange staart",
           "head": ["", "gemiddelde", "mediaan", "grootste", "aandeel van de grootste in het totaal"],
           "rows": ["lengtes in cm, dunne staart", "groottes van incidenten, lange staart"],
           "said": "Met {n} trekkingen hebben de lengtes een gemiddelde van {hm} cm en is de grootste {hx} cm; de "
                   "groottes van de incidenten hebben een gemiddelde van {sm} en een mediaan van {sd}, en de grootste, "
                   "{sx}, is {share} van hun totaal.",
           "x": "trekkingen tot nu toe", "y": "gemiddelde grootte van incidenten", "run": "gemiddelde tot nu toe",
           "long": "gemiddelde op lange termijn, 3",
           "note": "De lengtes zijn 140 plus 80 maal het gemiddelde van 12 willekeurige getallen; de groottes van de "
                   "incidenten volgen het recept van Newman met exponent 2,5. Dezelfde seed geeft dezelfde "
                   "trekkingen, dus een groter aantal op de schuif voegt trekkingen toe aan de vorige. De lijn is het "
                   "gemiddelde van de groottes na elke trekking, met de trekkingen op een logaritmische schaal."},
}


def _demo_numbers(n: int) -> dict:
    heights, sizes = samples()
    h, s = heights[:n], sizes[:n]
    return {"hm": float(h.mean()), "hd": float(np.median(h)), "hx": float(h.max()), "hshare": float(h.max() / h.sum()),
            "sm": float(s.mean()), "sd": float(np.median(s)), "sx": float(s.max()), "share": float(s.max() / s.sum())}


def demo_frame(v, lang: str) -> tuple[str, str]:
    """The demo: n heights and n incident sizes, their mean, median and largest, the largest's share of the total, and
    the mean of the incident sizes after each draw up to n."""
    n = min(DRAWS, key=lambda m: abs(math.log10(m) - math.log10(max(float(v), 1))))
    w = _DEMO_WORDS[lang]
    d = _demo_numbers(n)
    rows = [([(html.escape(w["rows"][0]), ""), (_code(_prose(d["hm"], lang)), "text-align:right"),
              (_code(_prose(d["hd"], lang)), "text-align:right"), (_code(_prose(d["hx"], lang)), "text-align:right"),
              (_code(_percent(d["hshare"], lang) if d["hshare"] >= 0.0005 else "0%"), "text-align:right")], ""),
            ([(html.escape(w["rows"][1]), ""), (_code(_prose(d["sm"], lang, 2)), "text-align:right"),
              (_code(_prose(d["sd"], lang, 2)), "text-align:right"), (_code(_prose(d["sx"], lang)), "text-align:right"),
              (_code(_percent(d["share"], lang)), "text-align:right")], "")]
    _, sizes = samples()
    steps = sorted({int(round(10 ** (k / 40))) for k in range(0, 241)} | {n})
    steps = [k for k in steps if k <= n]
    means = np.cumsum(sizes[:n])
    ys = [float(means[k - 1] / k) for k in steps]
    top = max(6.0, math.ceil(max(ys) + 0.5))
    panel = chart.Panel(x=(1, DRAWS[-1]), y=(0, top), xlabel=w["x"], ylabel=w["y"], xlog=True,
                        yticks=list(range(0, int(top) + 1)))
    panel.line([1, DRAWS[-1]], [LONG_RUN, LONG_RUN], role="ink-3", dashed=True)
    panel.line(steps, ys, role="loss")
    panel.points([n], [ys[-1]], role="loss")
    for y, name, role, dashed in ((top - 0.6, "run", "loss", False), (top - 1.3, "long", "ink-3", True)):
        panel.line([1.5, 4], [y, y], role=role, dashed=dashed)   # a key at the top left
        panel.text(5, y, w[name], role=role, anchor="start")
    title = w["title"].format(n=_prose(n, lang, 0))
    said = w["said"].format(n=_prose(n, lang, 0), hm=_prose(d["hm"], lang), hx=_prose(d["hx"], lang),
                            sm=_prose(d["sm"], lang, 2), sd=_prose(d["sd"], lang, 2), sx=_prose(d["sx"], lang),
                            share=_percent(d["share"], lang))
    svg = chart.figure([panel], alt=title + ". " + said, width=380, height=240)
    if lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = _table(rows, [html.escape(h) for h in w["head"]]) + svg + _note(w["note"])
    return yours.figure(title, body, alt=title + ". " + said), said


# --- the pictures under the checks -----------------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _all gives for a function that stopped with an error


def _all(fn, calls: list):
    """The learner's function on copies of each list of arguments in `calls`, all within one time limit of a check;
    _STOPPED when it stops with an error, and the Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: [fn(*(copy.deepcopy(a) for a in args)) for args in calls])
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


_STRETCH_WORDS = {
    "en": {"title": "Your stretch beside the course's",
           "head": ["deviations", "design", "the course's", "yours"], "stopped": "error",
           "x": "flight", "y": "size of the deviation", "design": "design limit", "course": "accepted, the course's",
           "yours": "accepted, yours", "flight_dev": "deviation of a flight",
           "note": "The chart is the first case of the check: the dots are the deviations of each flight, the dashed "
                   "line the design limit, the line what counts as acceptable after each flight, and the rings your "
                   "stretch. A dashed row is one where your stretch gives something other than the course's."},
    "nl": {"title": "Jouw stretch naast die van de course",
           "head": ["deviations", "design", "van de course", "de jouwe"], "stopped": "fout",
           "x": "vlucht", "y": "grootte van de afwijking", "design": "ontwerpgrens",
           "course": "geaccepteerd, van de course", "yours": "geaccepteerd, de jouwe",
           "flight_dev": "afwijking van een vlucht",
           "note": "De grafiek is het eerste geval van de check: de stippen zijn de afwijkingen van elke vlucht, de "
                   "gestippelde lijn is de ontwerpgrens, de lijn wat na elke vlucht als aanvaardbaar geldt, en de "
                   "ringen zijn jouw stretch. Een gestippelde rij is er een waar jouw stretch iets anders geeft dan "
                   "die van de course."},
}


def stretch_picture(seen) -> str:
    """The first case as a chart, the deviations of each flight with the design limit and the accepted limit, the
    course's and the learner's; and every case of the check in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _STRETCH_WORDS[seen.lang]
    got = _all(fn, STRETCH_CASES)
    rows = []
    for i, (deviations, design) in enumerate(STRETCH_CASES):
        want = stretch(deviations, design)
        if got is _STOPPED:
            shown, off = html.escape(w["stopped"]), True
        else:
            shown, off = _code(_value(got[i])), not _same(got[i], want)
        rows.append(([(_code(_value(deviations)), ""), (_code(_value(design)), "text-align:right"),
                      (_code(_value(want)), ""), (shown, "font-weight:600")], _OFF if off else ""))
    deviations, design = STRETCH_CASES[0]
    want = stretch(deviations, design)
    flights = list(range(1, len(deviations) + 1))
    panel = chart.Panel(x=(0.5, len(deviations) + 0.5), y=(0, 4), xlabel=w["x"], ylabel=w["y"], xticks=flights,
                        yticks=[0, 1, 2, 3, 4])
    panel.line([0.5, len(deviations) + 0.5], [design, design], role="ink-3", dashed=True)
    xs, ys = [], []
    for x, y in zip(flights, want):   # a step line: the limit holds from one flight to the next
        xs += [x - 0.5, x + 0.5]
        ys += [y, y]
    panel.line(xs, ys, role="ink")
    mine = None if got is _STOPPED else got[0]
    if isinstance(mine, list) and len(mine) == len(deviations) and all(_number(m) is not None for m in mine):
        panel.points(flights, [min(max(float(m), 0), 4) for m in mine], role="loss", r=6, hollow=True)
    panel.points(flights, deviations, role="ink-2", r=3.5)   # after the rings, so a ring never hides a deviation
    for y, name, role, dashed in ((3.75, "course", "ink", False), (3.35, "design", "ink-3", True)):
        panel.line([0.6, 0.95], [y, y], role=role, dashed=dashed)   # a key at the top left
        panel.text(1.05, y, w[name], role=role, anchor="start")
    panel.points([0.78], [2.95], role="loss", r=6, hollow=True)
    panel.text(1.05, 2.95, w["yours"], role="loss", anchor="start")
    panel.points([0.78], [2.55], role="ink-2", r=3.5)
    panel.text(1.05, 2.55, w["flight_dev"], role="ink-2", anchor="start")
    svg = chart.figure([panel], alt=w["title"], width=360, height=230)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


_MEAN_WORDS = {
    "en": {"title": "Your running_mean beside the course's",
           "head": ["values", "the course's", "yours"], "stopped": "error",
           "x": "incident sizes so far", "y": "mean so far", "course": "the course's", "yours": "yours",
           "note": "The chart is your running_mean on the first 10,000 incident sizes of the demo, beside the "
                   "course's, with the number of sizes on a log scale. The table is every case of the check; a dashed "
                   "row is one where your running_mean gives something other than the course's."},
    "nl": {"title": "Jouw running_mean naast die van de course",
           "head": ["values", "van de course", "de jouwe"], "stopped": "fout",
           "x": "groottes van incidenten tot nu toe", "y": "gemiddelde tot nu toe", "course": "van de course",
           "yours": "de jouwe",
           "note": "De grafiek is jouw running_mean op de eerste 10.000 groottes van incidenten uit de demo, naast die "
                   "van de course, met het aantal groottes op een logaritmische schaal. De tabel bevat elk geval van "
                   "de check; een gestippelde rij is er een waar jouw running_mean iets anders geeft dan die van de "
                   "course."},
}

PICTURE_DRAWS = 10000   # how many incident sizes the picture under Exercise 2 gives to the learner's running_mean


def mean_picture(seen) -> str:
    """The learner's running_mean on the first 10,000 incident sizes of the demo, beside the course's, on a log scale;
    and every case of the check in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _MEAN_WORDS[seen.lang]
    got = _all(fn, MEAN_CASES)
    rows = []
    for i, (values,) in enumerate(MEAN_CASES):
        want = running_mean(values)
        if got is _STOPPED:
            shown, off = html.escape(w["stopped"]), True
        else:
            shown, off = _code(_value(got[i])), not _same(got[i], want)
        rows.append(([(_code(_value(values)), ""), (_code(_value(want)), ""), (shown, "font-weight:600")],
                     _OFF if off else ""))
    _, sizes = samples()
    data = [float(x) for x in sizes[:PICTURE_DRAWS]]
    want = running_mean(data)
    mine = _all(fn, [(data,)])
    mine = None if mine is _STOPPED else mine[0]
    steps = sorted({int(round(10 ** (k / 40))) for k in range(0, 161)})
    # the ticks as powers of ten, as in the demo, so that 10000 does not run off the right edge
    panel = chart.Panel(x=(1, PICTURE_DRAWS), y=(0, 7), xlabel=w["x"], ylabel=w["y"], xlog=True,
                        xticks=[1, 10, 100, 1000, 10000], xtick_labels=["10⁰", "10¹", "10²", "10³", "10⁴"],
                        yticks=[0, 1, 2, 3, 4, 5, 6, 7])
    panel.line(steps, [want[k - 1] for k in steps], role="ink")
    if isinstance(mine, list) and len(mine) == len(data) and all(_number(mine[k - 1]) is not None for k in steps):
        panel.line(steps, [min(max(float(mine[k - 1]), 0), 7) for k in steps], role="loss", dashed=True)
    for y, name, role, dashed in ((1.4, "course", "ink", False), (0.7, "yours", "loss", True)):
        panel.line([700, 2500], [y, y], role=role, dashed=dashed)   # a key at the bottom right, below the line
        panel.text(550, y, w[name], role=role, anchor="end")
    svg = chart.figure([panel], alt=w["title"], width=360, height=220)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["stretch", "running_mean", "drift", "whose_swan"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# exercise 1: stretch; the first case stretches three times, one falls back below the limit, one starts above it
STRETCH_CASES = [([0.5, 1.4, 1.2, 2.0, 1.8, 3.1], 1), ([0.2, 0.8], 1), ([3, 2, 1], 1), ([1, 1, 1], 1),
                 ([2.5, 0.1, 2.6], 2), ([], 2), ([5], 10), ([1.5, 1.2, 1.4, 1.6], 1)]

_S_START = "def stretch(deviations, design):\n    ...\n"
_S = ("def stretch(deviations, design):\n"
      "    limit = design\n"
      "    accepted = []\n"
      "    for deviation in deviations:\n"
      "        if deviation > limit:\n"
      "            limit = deviation\n"
      "        accepted.append(limit)\n"
      "    return accepted\n")
_S_DESIGN = _S.replace("        if deviation > limit:\n            limit = deviation\n",
                       "        if deviation > design:\n            limit = deviation\n")
_S_LATE = _S.replace("        if deviation > limit:\n            limit = deviation\n        accepted.append(limit)\n",
                     "        accepted.append(limit)\n        if deviation > limit:\n            limit = deviation\n")
_S_DEVIATION = _S.replace("        accepted.append(limit)\n", "        accepted.append(deviation)\n")
_S_NEVER = _S.replace("            limit = deviation\n", "            design = deviation\n")
_S_END = _S.replace("    return accepted\n", "    return limit\n")
_S_MIN = _S.replace("        if deviation > limit:\n", "        if deviation < limit:\n")

# exercise 2: running_mean; the third case has one big value among small ones
MEAN_CASES = [([2, 4, 6],), ([10],), ([1, 1, 1, 97],), ([5, 3],), ([],), ([0.5, 1.5, 1],), ([3, 3, 3, 3, 3, 1000],),
              ([-2, 2, -2, 2],)]

_M_START = "def running_mean(values):\n    ...\n"
_M = ("def running_mean(values):\n"
      "    total = 0\n"
      "    count = 0\n"
      "    means = []\n"
      "    for value in values:\n"
      "        total = total + value\n"
      "        count = count + 1\n"
      "        means.append(total / count)\n"
      "    return means\n")
_M_ALL = _M.replace("means.append(total / count)", "means.append(total / len(values))")
_M_RESET = _M.replace("    for value in values:\n        total = total + value\n",
                      "    for value in values:\n        total = 0\n        total = total + value\n")
_M_VALUE = _M.replace("means.append(total / count)", "means.append(value)")
_M_LAST = _M.replace("    return means\n", "    return means[-1]\n")
_M_BEFORE = _M.replace("        total = total + value\n        count = count + 1\n        means.append(total / count)\n",
                       "        count = count + 1\n        means.append(total / count)\n        total = total + value\n")
_M_SUM = _M.replace("means.append(total / count)", "means.append(total)")

EXERCISES = {
    "stretch": FunctionBox(
        name="stretch",
        picture=stretch_picture,
        function="stretch",
        start=_S_START,
        solution=_S,
        visible=[STRETCH_CASES[0]],
        hidden=STRETCH_CASES[1:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("design", wrong=_S_DESIGN,
                    en="`{call}` gives {got}, and it should give {want}: you compare each deviation with `design`, so after a smaller deviation the limit falls back. What counts as acceptable is the largest deviation so far: compare with `limit`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je vergelijkt elke afwijking met `design`, dus na een kleinere afwijking zakt de grens terug. Wat als aanvaardbaar geldt, is de grootste afwijking tot nu toe: vergelijk met `limit`."),
            Mistake("late", wrong=_S_LATE,
                    en="`{call}` gives {got}, and it should give {want}: you add the limit before the flight has moved it, so each limit is one flight late. Move `limit` first, then add it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je voegt de grens toe voordat de vlucht die heeft veranderd, dus elke grens loopt één vlucht achter. Pas eerst `limit` aan en voeg die daarna toe."),
            Mistake("deviation", wrong=_S_DEVIATION,
                    en="`{call}` gives {got}, and it should give {want}: you add each deviation itself. The list holds what counts as acceptable after each flight: add `limit`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je voegt elke afwijking zelf toe. De lijst bevat wat na elke vlucht als aanvaardbaar geldt: voeg `limit` toe."),
            Mistake("never", wrong=_S_NEVER,
                    en="`{call}` gives {got}, and it should give {want}: the limit you add never moves. When a deviation is bigger than `limit`, `limit` itself becomes that deviation.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de grens die je toevoegt, gaat nooit omhoog. Als een afwijking groter is dan `limit`, wordt `limit` zelf die afwijking."),
            Mistake("end", wrong=_S_END,
                    en="`{call}` gives {got}, and it should give {want}: you return only the last limit. Return the list `accepted`, with the limit after each flight.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft alleen de laatste grens terug. Geef de lijst `accepted` terug, met de grens na elke vlucht."),
            Mistake("min", wrong=_S_MIN,
                    en="`{call}` gives {got}, and it should give {want}: you move the limit when a deviation is smaller. It stretches when a deviation is bigger than the limit and nothing went wrong.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je verandert de grens als een afwijking kleiner is. Hij rekt op als een afwijking groter is dan de grens en er niets misging."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start `limit` at `design`, go through the deviations, move `limit` up when a deviation is bigger, and add `limit` to `accepted` after each flight.",
                      "nl": "Vervang de `...`: begin `limit` bij `design`, loop door de afwijkingen, maak `limit` hoger als een afwijking groter is, en voeg na elke vlucht `limit` toe aan `accepted`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Start `limit` at `design` and `accepted` as an empty list. For each `deviation`: when it is bigger than `limit`, it becomes the new `limit`; then add `limit` to `accepted`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begin `limit` bij `design` en `accepted` als een lege lijst. Voor elke `deviation`: als die groter is dan `limit`, wordt die de nieuwe `limit`; voeg daarna `limit` toe aan `accepted`."},
            "almost": {"en": "Right for the first flights. But `{call}` gives {got}, and it should give {want}. Does your function use the `deviations` and the `design` it gets?",
                       "nl": "Goed voor de eerste vluchten. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `deviations` en de `design` die ze krijgt?"},
            "correct": {"en": "Your `stretch` is right in all {n} cases. Under the check, the chart shows how the accepted limit climbs away from the design limit.",
                        "nl": "Je `stretch` klopt in alle {n} gevallen. Onder de check laat de grafiek zien hoe de geaccepteerde grens wegklimt van de ontwerpgrens."},
        },
    ),
    "running_mean": FunctionBox(
        name="running_mean",
        picture=mean_picture,
        function="running_mean",
        start=_M_START,
        solution=_M,
        visible=[MEAN_CASES[0]],
        hidden=MEAN_CASES[1:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("all", wrong=_M_ALL,
                    en="`{call}` gives {got}, and it should give {want}: you divide by the number of all values each time. After the first value only one value counts: divide by `count`, the values so far.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je deelt elke keer door het aantal van alle waarden. Na de eerste waarde telt maar één waarde mee: deel door `count`, de waarden tot nu toe."),
            Mistake("reset", wrong=_M_RESET,
                    en="`{call}` gives {got}, and it should give {want}: `total` starts again at 0 inside the loop, so it only holds the last value. Set `total = 0` once, before the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `total` begint in de loop steeds opnieuw op 0, dus het bevat alleen de laatste waarde. Zet `total = 0` één keer, voor de loop."),
            Mistake("value", wrong=_M_VALUE,
                    en="`{call}` gives {got}, and it should give {want}: you add each value itself. Add the mean so far: `total` divided by `count`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je voegt elke waarde zelf toe. Voeg het gemiddelde tot nu toe toe: `total` gedeeld door `count`."),
            Mistake("last", wrong=_M_LAST,
                    en="`{call}` gives {got}, and it should give {want}: you return only the last mean. Return the list `means`, with the mean after each value.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft alleen het laatste gemiddelde terug. Geef de lijst `means` terug, met het gemiddelde na elke waarde."),
            Mistake("before", wrong=_M_BEFORE,
                    en="`{call}` gives {got}, and it should give {want}: you compute the mean before you add the value to `total`, so each mean misses its newest value. Add the value first.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je rekent het gemiddelde uit voordat je de waarde bij `total` optelt, dus elk gemiddelde mist de nieuwste waarde. Tel de waarde eerst op."),
            Mistake("sum", wrong=_M_SUM,
                    en="`{call}` gives {got}, and it should give {want}: you add the total so far, not the mean. Divide `total` by `count`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je voegt het totaal tot nu toe toe, niet het gemiddelde. Deel `total` door `count`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: keep a `total` and a `count`, and after each value add the mean so far to the list `means`.",
                      "nl": "Vervang de `...`: houd een `total` en een `count` bij, en voeg na elke waarde het gemiddelde tot nu toe toe aan de lijst `means`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Start `total` and `count` at 0 and `means` as an empty list. For each `value`, add it to `total`, add 1 to `count`, and add the mean so far to `means`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begin `total` en `count` bij 0 en `means` als een lege lijst. Tel elke `value` op bij `total`, tel 1 op bij `count`, en voeg het gemiddelde tot nu toe toe aan `means`."},
            "almost": {"en": "Right for the first list. But `{call}` gives {got}, and it should give {want}. Does your function use the `values` it gets?",
                       "nl": "Goed voor de eerste lijst. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `values` die ze krijgt?"},
            "correct": {"en": "Your `running_mean` is right in all {n} cases. Under the check, it runs on 10,000 incident sizes from the demo.",
                        "nl": "Je `running_mean` klopt in alle {n} gevallen. Onder de check draait je functie op 10.000 groottes van incidenten uit de demo."},
        },
    ),
}

CASES = [
    ("stretch", _S_START, "empty", None),
    ("stretch", _S, "correct", None),
    ("stretch", "def stretch(deviations, design):\n    accepted = []\n    for deviation in deviations:\n        design = max(design, deviation)\n        accepted.append(design)\n    return accepted\n", "correct", None),
    ("stretch", "def stretch(deviations, design):\n    return [max([design] + deviations[:i + 1]) for i in range(len(deviations))]\n", "correct", None),
    ("stretch", _S_DESIGN, "wrong", "design"),
    ("stretch", _S_LATE, "wrong", "late"),
    ("stretch", _S_DEVIATION, "wrong", "deviation"),
    ("stretch", _S_NEVER, "wrong", "never"),
    ("stretch", _S_END, "wrong", "end"),
    ("stretch", _S_MIN, "wrong", "min"),
    ("stretch", "def stretch(deviations, design):\n    return [1, 1.4, 1.4, 2.0, 2.0, 3.1]\n", "almost", None),
    ("running_mean", _M_START, "empty", None),
    ("running_mean", _M, "correct", None),
    ("running_mean", "def running_mean(values):\n    return [sum(values[:i + 1]) / (i + 1) for i in range(len(values))]\n", "correct", None),
    ("running_mean", "import numpy as np\n\n\ndef running_mean(values):\n    means = []\n    for i in range(len(values)):\n        means.append(float(np.mean(values[:i + 1])))\n    return means\n", "correct", None),
    ("running_mean", _M_ALL, "wrong", "all"),
    ("running_mean", _M_RESET, "wrong", "reset"),
    ("running_mean", _M_VALUE, "wrong", "value"),
    ("running_mean", _M_LAST, "wrong", "last"),
    ("running_mean", _M_BEFORE, "wrong", "before"),
    ("running_mean", _M_SUM, "wrong", "sum"),
    ("running_mean", "def running_mean(values):\n    return [2, 3, 4]\n", "almost", None),
]
