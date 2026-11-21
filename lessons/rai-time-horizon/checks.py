# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-time-horizon: how long a task can a model do? The learner writes success, the chance that
an agent completes a task of a given length by METR's logistic curve over log2 of the task length (Kwa et al., 2025;
the base 2 from METR's code), and doubling_days, the doubling time of the 50% time horizon from a straight line through
log2 of the horizons against the release date, on METR's published data (Time Horizon 1.1, CC BY 4.0). A demo shows
how o3's horizon shrinks when the success asked for goes up from 50% towards 99%."""
import contextlib
import copy
import html
import io
import math
import re

import numpy as np

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the curve of one agent ------------------------------------------------------------------------------------------

def success(horizon, minutes, beta):
    """METR's curve: the chance that an agent with this 50% time horizon (in minutes) and steepness beta completes a
    task that takes skilled people this many minutes, sigma((log2 h - log2 t) * beta)."""
    z = (np.log2(horizon) - np.log2(minutes)) * beta
    return 1 / (1 + np.exp(-z))


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def horizon_at(h50: float, beta: float, p: float) -> float:
    """The length of task at which the same curve gives the chance p: h50 * 2 ** (-logit(p) / beta)."""
    return h50 * 2 ** (-logit(p) / beta)


# o3 in METR's data (Time Horizon 1.1, page of 8 September 2026): 50% horizon 119.7 minutes, 80% horizon 30.0 minutes;
# the steepness follows from the two, since the 80% horizon lies logit(0.8) / beta doublings below the 50% one
O3_H50 = 119.7
O3_H80 = 30.0
O3_BETA = math.log(4) / math.log2(O3_H50 / O3_H80)


# --- the trend: METR's leading models from 2023 on --------------------------------------------------------------------

# METR, Time Horizon 1.1 (benchmark_results_1_1.yaml, page of 8 September 2026, CC BY 4.0): the models the file marks
# as the best when they came out, released from 2023 on, with a 50% horizon of at most 16 hours, as in METR's fit.
# Minutes rounded to one decimal; days after GPT-4's release on 14 March 2023. Anthropic's models by month.
MODELS = ["GPT-4", "GPT-4 (2023-11)", "GPT-4o", "Anthropic (2024-06)", "o1-preview", "Anthropic (2024-10)", "o1",
          "Anthropic (2025-02)", "o3", "GPT-5", "Gemini 3 Pro", "Anthropic (2025-11)", "GPT-5.2", "Anthropic (2026-02)"]
DAYS = [0, 237, 426, 464, 548, 588, 632, 713, 764, 877, 980, 986, 1003, 1059]
MINUTES = [4.0, 4.0, 7.0, 11.4, 20.3, 20.5, 38.8, 60.4, 119.7, 203.0, 224.3, 293.0, 352.2, 718.8]


def doubling_days(days, minutes):
    """The doubling time in days of the line through log2 of the horizons against the release date."""
    slope, start = np.polyfit(days, np.log2(minutes), 1)
    return 1 / slope


def two_point(days, minutes, a: int, b: int) -> float:
    """The doubling time from two models alone, a and b: the days between them over the doublings between them."""
    return (days[b] - days[a]) / math.log2(minutes[b] / minutes[a])


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:middle")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_CODE = "font-family:var(--ea-font-code);font-size:0.78rem"

# a log axis of task lengths in minutes, from half a minute to about a day and a half
_T_RANGE = (0.5, 2000)
_T_TICKS = [1, 4, 15, 60, 240, 960]
_T_LABELS = {"en": ["1 min", "4 min", "15 min", "1 h", "4 h", "16 h"], "nl": ["1 min", "4 min", "15 min", "1 u", "4 u", "16 u"]}


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


def number(v: float, lang: str, digits: int = 1) -> str:
    text = f"{v:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def _num(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def show_value(v) -> str:
    """A result as the check shows it: a number to at most six decimals, NumPy's numbers as plain ones, anything else
    as Python shows it, cut short when it is long."""
    if isinstance(v, (np.floating, np.integer)) and not isinstance(v, np.bool_):
        v = v.item()
    text = repr(round(v, 6)) if isinstance(v, float) and math.isfinite(v) else repr(v)
    return text if len(text) <= 200 else text[:197] + "..."


def same_number(got, want) -> bool:
    """Whether the learner's number is the course's: within a millionth of it, relative to its size."""
    return _num(got) and math.isfinite(float(got)) and abs(float(got) - want) <= 1e-6 * max(1.0, abs(want))


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
            with np.errstate(all="ignore"):
                return fn(*(copy.deepcopy(a) for a in args))
        except Timeout:
            raise
        except Exception:   # noqa: BLE001   the check above says what went wrong
            return _STOPPED

    with contextlib.redirect_stdout(io.StringIO()):
        return with_time_limit(lambda: [one(args) for args in calls])


def _rows(fn, cases: list, function, box, words: dict) -> tuple:
    """The rows of the table under a check: the call, the course's result and the learner's, a dashed row where they
    differ; and the learner's results."""
    got = _all(fn, cases)
    rows = []
    for i, case in enumerate(cases):
        want = function(*copy.deepcopy(case))
        if got[i] is _STOPPED:
            shown_, off = html.escape(words["stopped"]), True
        else:
            shown_, off = _code(show_value(got[i])), not same_number(got[i], want)
        rows.append(([(_code(box._call_text(case)), ""), (_code(show_value(want)), ""), (shown_, "font-weight:600")],
                     _OFF if off else ""))
    return rows, got


def _curve(h: float, beta: float) -> tuple:
    """The points of one agent's curve over the log axis of task lengths."""
    ts = [_T_RANGE[0] * (_T_RANGE[1] / _T_RANGE[0]) ** (j / 120) for j in range(121)]
    return ts, [float(success(h, t, beta)) for t in ts]


# exercise 1: success, drawn as the chance of success against the task length

_SUCCESS_WORDS = {
    "en": {"title": "Your success curve", "x": "task length for skilled people (log scale)", "y": "chance of success",
           "head": ["the call", "the course's", "yours"], "stopped": "error",
           "note": "The solid line is the course's curve for a horizon of 120 minutes and beta 0.6, the dashed line for "
                   "60 minutes and beta 1.0. The dots are the course's answers for the cases on these two curves, the "
                   "rings yours. The table shows every case of the check; a dashed row is one where your function gives "
                   "something other than the course's."},
    "nl": {"title": "Jouw succescurve", "x": "taaklengte voor vakmensen (logschaal)", "y": "kans op succes",
           "head": ["de aanroep", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De doorgetrokken lijn is de curve van de course voor een horizon van 120 minuten en beta 0,6, de "
                   "gestippelde lijn voor 60 minuten en beta 1,0. De stippen zijn de antwoorden van de course voor de "
                   "gevallen op deze twee curves, de ringen de jouwe. De tabel bevat elk geval van de check; een "
                   "gestippelde rij is er een waar jouw functie iets anders geeft dan die van de course."},
}


def success_picture(seen) -> str:
    """Under the check of exercise 1: two of the course's curves over a log axis of task lengths, the course's answers
    as dots and the learner's as rings; every case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _SUCCESS_WORDS[seen.lang]
    rows, got = _rows(fn, SUCCESS_CASES, success, EXERCISES["success"], w)
    panel = chart.Panel(x=_T_RANGE, y=(0, 1.0), xlabel=w["x"], ylabel=w["y"], xlog=True, xticks=_T_TICKS,
                        xtick_labels=_T_LABELS[seen.lang], yticks=[0, 0.25, 0.5, 0.75, 1.0])
    panel.line(*_curve(120, 0.6), role="ink-3", width=1.8)
    panel.line(*_curve(60, 1.0), role="ink-3", width=1.6, dashed=True)
    for i, (h, t, beta) in enumerate(SUCCESS_CASES):
        if (h, beta) in ((120, 0.6), (60, 1.0)):
            if _num(got[i]) and 0 <= float(got[i]) <= 1:
                panel.points([t], [float(got[i])], role="ink", r=8, hollow=True)
            panel.points([t], [float(success(h, t, beta))], role="ink", r=4)
    svg = chart.figure([panel], alt=w["title"], width=440, height=250)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# exercise 2: doubling_days, drawn as the horizons against the release date with the course's line and the learner's

_DOUBLING_WORDS = {
    "en": {"title": "Your doubling time", "x": "days after 14 March 2023", "y": "50% horizon, minutes (log scale)",
           "head": ["the call", "the course's", "yours"], "stopped": "error",
           "note": "The dots are METR's 14 models of the first case. The solid line is the course's fit; the dashed line "
                   "rises as fast as your doubling time says, through the middle of the dots, where every straight line "
                   "fitted this way passes. The table shows every case of the check; a dashed row is one where your "
                   "function gives something other than the course's."},
    "nl": {"title": "Jouw verdubbelingstijd", "x": "dagen na 14 maart 2023", "y": "50%-horizon, minuten (logschaal)",
           "head": ["de aanroep", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De stippen zijn de 14 modellen van METR uit het eerste geval. De doorgetrokken lijn is de fit van de "
                   "course; de gestippelde lijn stijgt zo snel als jouw verdubbelingstijd zegt, door het midden van de "
                   "stippen, waar elke rechte lijn die zo gefit is doorheen gaat. De tabel bevat elk geval van de check; "
                   "een gestippelde rij is er een waar jouw functie iets anders geeft dan die van de course."},
}

_Y_RANGE = (2, 1500)


def _line_through_middle(per_day: float) -> tuple:
    """A line in log2 minutes with this slope per day through the middle of METR's points, kept inside the picture."""
    mx, my = sum(DAYS) / len(DAYS), sum(math.log2(m) for m in MINUTES) / len(MINUTES)
    xs, ys = [], []
    for j in range(0, 107):
        x = 10 * j
        up = my + per_day * (x - mx)   # log2 of the minutes; only within the picture is it raised to a power
        if math.log2(_Y_RANGE[0]) <= up <= math.log2(_Y_RANGE[1]):
            xs.append(x)
            ys.append(2 ** up)
    return xs, ys


def doubling_picture(seen) -> str:
    """Under the check of exercise 2: METR's horizons against the release date on a log axis, the course's line, and
    a line that rises as fast as the learner's doubling time; every case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _DOUBLING_WORDS[seen.lang]
    rows, got = _rows(fn, DOUBLING_CASES, doubling_days, EXERCISES["doubling"], w)
    panel = chart.Panel(x=(0, 1080), y=_Y_RANGE, xlabel=w["x"], ylabel=w["y"], ylog=True, xticks=[0, 250, 500, 750, 1000],
                        yticks=[4, 15, 60, 240, 960])
    panel.points(DAYS, MINUTES, role="ink", r=4)
    xs, ys = _line_through_middle(1 / doubling_days(DAYS, MINUTES))
    panel.line(xs, ys, role="ink-3", width=1.8)
    mine = got[0]
    if mine is not _STOPPED and _num(mine) and math.isfinite(float(mine)) and float(mine) != 0:
        xs, ys = _line_through_middle(1 / float(mine))
        if len(xs) >= 2:
            panel.line(xs, ys, role="ink", width=1.6, dashed=True)
    svg = chart.figure([panel], alt=w["title"], width=440, height=260)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the demo: o3's horizon at higher levels of success ------------------------------------------------------------------

P_VALUES = list(range(50, 100))   # the demo's stops: 50% to 99%
P_START = 50

DEMO_WORDS = {
    "en": {"x": "task length for skilled people (log scale)", "y": "chance of success (%)", "fifty": "50%"},
    "nl": {"x": "taaklengte voor vakmensen (logschaal)", "y": "kans op succes (%)", "fifty": "50%"},
}


def minutes_text(t: float, lang: str) -> str:
    """A length of task as the demo writes it: minutes with one decimal."""
    word = "minuten" if lang == "nl" else "minutes"
    if round(t, 1) == 1.0:
        word = "minuut" if lang == "nl" else "minute"
    return f"{number(t, lang)} {word}"


def demo_frame(p: int, lang: str) -> tuple:
    """The demo at a success level of p percent: o3's curve, the level as a dashed line, and the length of task where
    the curve meets it; and the sentence under it."""
    w = DEMO_WORDS[lang]
    panel = chart.Panel(x=_T_RANGE, y=(0, 100), xlabel=w["x"], ylabel=w["y"], xlog=True, xticks=_T_TICKS,
                        xtick_labels=_T_LABELS[lang], yticks=[0, 25, 50, 75, 100])
    ts, ps = _curve(O3_H50, O3_BETA)
    panel.line(_T_RANGE, [p, p], role="hyper", width=1.4, dashed=True)
    panel.line(ts, [100 * v for v in ps], role="ink-2", width=2.2)
    t = horizon_at(O3_H50, O3_BETA, p / 100)
    panel.line([t, t], [0, p], role="ink-3", width=1.2, dashed=True)
    if p != 50:   # where the 50% horizon lies, for comparison, with its label below the curve
        panel.points([O3_H50], [50], role="ink-3", r=4)
        panel.text(O3_H50, 34, w["fifty"], role="ink-3", halo=True)
    panel.points([t], [p], role="ink", r=5.5)
    label = minutes_text(t, lang)
    if p <= 90:
        panel.text(t * 1.25, p + 6, label, role="ink", anchor="start", halo=True)
    else:   # high up there is no room beside the dot: the label goes low, beside the dashed line down from it
        panel.text(t * 1.2, 10, label, role="ink", anchor="start", halo=True)
    svg = chart.figure([panel], alt=demo_alt(p, lang), width=440, height=250)
    if lang == "nl":
        svg = chart.with_decimal_comma(svg)
    return svg, demo_said(p, lang)


def demo_alt(p: int, lang: str) -> str:
    t = minutes_text(horizon_at(O3_H50, O3_BETA, p / 100), lang)
    if lang == "nl":
        return (f"De kans dat o3 een taak voltooit tegen de taaklengte op een logschaal: hoog bij korte taken en dalend "
                f"bij lange; de kans van {p}% hoort bij taken van {t}.")
    return (f"The chance that o3 completes a task against the task length on a log scale: high for short tasks and "
            f"falling for long ones; the chance of {p}% belongs to tasks of {t}.")


def demo_said(p: int, lang: str) -> str:
    """What the sentence under the demo says at a success level of p percent."""
    t = horizon_at(O3_H50, O3_BETA, p / 100)
    shorter = number(O3_H50 / t, lang)
    if lang == "nl":
        if p == 50:
            return (f"Bij **50%** is de horizon van o3 **{minutes_text(t, lang)}**: de 50%-time horizon zelf. Schuif "
                    f"omhoog en kijk hoe snel die korter wordt.")
        return (f"Bij **{p}%** is de horizon van o3 **{minutes_text(t, lang)}**: {shorter} keer zo kort als de "
                f"{minutes_text(O3_H50, lang)} bij 50%.")
    if p == 50:
        return (f"At **50%**, o3's horizon is **{minutes_text(t, lang)}**: the 50% time horizon itself. Slide up and "
                f"see how fast it gets shorter.")
    return (f"At **{p}%**, o3's horizon is **{minutes_text(t, lang)}**: {shorter} times shorter than its "
            f"{minutes_text(O3_H50, lang)} at 50%.")


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["success", "doubling", "headline", "forecast"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# exercise 1: success; the first case is a horizon of 120 minutes, a task of 30 and beta 0.6
SUCCESS_CASES = [
    (120, 30, 0.6),
    (120, 120, 0.6),
    (120, 480, 0.6),
    (120, 7.5, 0.6),
    (60, 15, 1.0),
    (60, 240, 1.0),
    (203.0, 60, 0.58),
    (1, 2, 0.5),
]
_S_DEF = "import numpy as np\n\n\ndef success(horizon, minutes, beta):\n"
_S_START = _S_DEF + "    ...\n"
_S = (_S_DEF +
      "    z = (np.log2(horizon) - np.log2(minutes)) * beta\n"
      "    return 1 / (1 + np.exp(-z))\n")
_S_FLIP = _S.replace("np.log2(horizon) - np.log2(minutes)", "np.log2(minutes) - np.log2(horizon)")
_S_NOLOG = _S.replace("(np.log2(horizon) - np.log2(minutes))", "(horizon - minutes)")
_S_NATURAL = _S.replace("np.log2", "np.log")
_S_TEN = _S.replace("np.log2", "np.log10")
_S_BRACKETS = _S.replace("(np.log2(horizon) - np.log2(minutes)) * beta", "np.log2(horizon) - np.log2(minutes) * beta")
_S_ODDS = _S.replace("    return 1 / (1 + np.exp(-z))\n", "    return np.exp(z)\n")
_S_LOGIT = _S.replace("    return 1 / (1 + np.exp(-z))\n", "    return z\n")

# exercise 2: doubling_days; the first case is METR's 14 models, the second the two of the worked example
DOUBLING_CASES = [
    (DAYS, MINUTES),
    ([0, 877], [4.0, 203.0]),
    ([0, 10, 20], [1, 2, 4]),
    ([0, 365], [10, 80]),
    ([0, 30, 60, 90], [2, 3, 8, 16]),
    ([0, 100, 200], [5, 10, 30]),
]
_D_DEF = "import numpy as np\n\n\ndef doubling_days(days, minutes):\n"
_D_START = _D_DEF + "    ...\n"
_D = (_D_DEF +
      "    slope, start = np.polyfit(days, np.log2(minutes), 1)\n"
      "    return 1 / slope\n")
_D_SLOPE = _D.replace("    return 1 / slope\n", "    return slope\n")
_D_ORDER = _D.replace("slope, start = np.polyfit", "start, slope = np.polyfit")
_D_NATURAL = _D.replace("np.log2(minutes)", "np.log(minutes)")
_D_TEN = _D.replace("np.log2(minutes)", "np.log10(minutes)")
_D_NOLOG = _D.replace("np.log2(minutes)", "minutes")
_D_SWAP = _D.replace("np.polyfit(days, np.log2(minutes), 1)", "np.polyfit(np.log2(minutes), days, 1)").replace(
    "    return 1 / slope\n", "    return slope\n")
_D_ENDS = (_D_DEF +
           "    doublings = np.log2(minutes[-1] / minutes[0])\n"
           "    return (days[-1] - days[0]) / doublings\n")

_NAMED = [(DAYS, "days"), (MINUTES, "minutes")]   # the data the texts call by name


def _lit(value) -> str:
    """A value the way the lesson writes it in code: METR's lists by name."""
    for data, name in _NAMED:
        if type(value) is type(data) and value == data:
            return name
    if isinstance(value, list):
        return "[" + ", ".join(_lit(v) for v in value) + "]"
    return repr(value)


class NamedBox(FunctionBox):
    """A FunctionBox whose messages write a call the way the lesson does: `doubling_days(days, minutes)`."""

    def _call_text(self, case) -> str:
        return f"{self.function}({', '.join(_lit(a) for a in case)})"


EXERCISES = {
    "success": NamedBox(
        name="success",
        picture=success_picture,
        function="success",
        start=_S_START,
        solution=_S,
        visible=[SUCCESS_CASES[0]],
        hidden=SUCCESS_CASES[1:],
        first=[_DOTS],
        show=show_value,
        compare=same_number,
        mistakes=[
            Mistake("flip", wrong=_S_FLIP,
                    en="`{call}` gives {got}, and it should give {want}: in your curve, a longer task gets a higher chance. A task shorter than the horizon should come out above one half. Look at the order of the two logs, and at the minus sign in `np.exp(-z)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in je curve krijgt een langere taak een hogere kans. Een taak die korter is dan de horizon hoort boven de helft uit te komen. Kijk naar de volgorde van de twee logs, en naar het minteken in `np.exp(-z)`."),
            Mistake("nolog", wrong=_S_NOLOG,
                    en="`{call}` gives {got}, and it should give {want}: your curve works with the minutes themselves. METR's curve works with their log base 2, so that every doubling of the task length is one equal step.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je curve werkt met de minuten zelf. De curve van METR werkt met hun log met grondtal 2, zodat elke verdubbeling van de taaklengte één gelijke stap is."),
            Mistake("natural", wrong=_S_NATURAL,
                    en="`{call}` gives {got}, and it should give {want}: your curve counts `beta` per factor $e$ of the task length, as with `np.log`, which has base $e$. METR's `beta` counts per doubling of the task length, so the curve uses `np.log2`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je curve telt `beta` per factor $e$ van de taaklengte, zoals met `np.log`, dat grondtal $e$ heeft. De `beta` van METR telt per verdubbeling van de taaklengte, dus de curve gebruikt `np.log2`."),
            Mistake("ten", wrong=_S_TEN,
                    en="`{call}` gives {got}, and it should give {want}: `np.log10` has base 10. METR's `beta` counts per doubling of the task length, so the curve uses `np.log2`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `np.log10` heeft grondtal 10. De `beta` van METR telt per verdubbeling van de taaklengte, dus de curve gebruikt `np.log2`."),
            Mistake("brackets", wrong=_S_BRACKETS,
                    en="`{call}` gives {got}, and it should give {want}: in your code, `beta` multiplies only the log of `minutes`. It multiplies the whole difference of the two logs, so that difference needs brackets.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in je code vermenigvuldigt `beta` alleen de log van `minutes`. Die vermenigvuldigt het hele verschil van de twee logs, dus dat verschil heeft haakjes nodig."),
            Mistake("odds", wrong=_S_ODDS,
                    en="`{call}` gives {got}, and it should give {want}: that is the odds of success, how many times likelier success is than failure. The chance is the sigmoid of the logit, between 0 and 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat zijn de odds op succes, hoeveel keer zo waarschijnlijk succes is als mislukken. De kans is de sigmoid van de logit, tussen 0 en 1."),
            Mistake("logit", wrong=_S_LOGIT,
                    en="`{call}` gives {got}, and it should give {want}: that is the logit `z`. The chance is the sigmoid of it, between 0 and 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is de logit `z`. De kans is de sigmoid ervan, tussen 0 en 1."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: work out the logit from the two logs base 2 and `beta`, and give back its sigmoid.",
                      "nl": "Vervang de `...`: reken de logit uit met de twee logs met grondtal 2 en `beta`, en geef de sigmoid ervan terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The logit is the log base 2 of the horizon minus the log base 2 of the task length, times `beta`; the chance is its sigmoid. A task as long as the horizon gets exactly one half.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De logit is de log met grondtal 2 van de horizon min die van de taaklengte, keer `beta`; de kans is de sigmoid ervan. Een taak zo lang als de horizon krijgt precies de helft."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `horizon`, `minutes` and `beta` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie wel de `horizon`, `minutes` en `beta` uit de aanroep?"},
            "correct": {"en": "Your `success` is right in all {n} cases. A task of 30 minutes is two doublings below a horizon of 120, so the logit is 2 times 0.6 = 1.2, and the chance about 77%; a task of 480 minutes, two doublings above, gets about 23%.",
                        "nl": "Je `success` klopt in alle {n} gevallen. Een taak van 30 minuten ligt twee verdubbelingen onder een horizon van 120, dus is de logit 2 keer 0,6 = 1,2, en de kans ongeveer 77%; een taak van 480 minuten, twee verdubbelingen erboven, krijgt ongeveer 23%."},
        },
    ),
    "doubling": NamedBox(
        name="doubling",
        picture=doubling_picture,
        function="doubling_days",
        start=_D_START,
        solution=_D,
        visible=[DOUBLING_CASES[0]],
        hidden=DOUBLING_CASES[1:],
        first=[_DOTS],
        show=show_value,
        compare=same_number,
        mistakes=[
            Mistake("slope", wrong=_D_SLOPE,
                    en="`{call}` gives {got}, and it should give {want}: that is the slope, the doublings per day. The doubling time is the days per doubling: one divided by the slope.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is de helling, de verdubbelingen per dag. De verdubbelingstijd is het aantal dagen per verdubbeling: één gedeeld door de helling."),
            Mistake("order", wrong=_D_ORDER,
                    en="`{call}` gives {got}, and it should give {want}: `np.polyfit` gives the slope first, then where the line starts. Your code takes them the other way round.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `np.polyfit` geeft eerst de helling, dan waar de lijn begint. Je code neemt ze andersom."),
            Mistake("natural", wrong=_D_NATURAL,
                    en="`{call}` gives {got}, and it should give {want}: with `np.log`, base $e$, the slope counts factors of $e$ per day, not doublings. Use `np.log2`, so that one step up is one doubling.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: met `np.log`, grondtal $e$, telt de helling factoren $e$ per dag, geen verdubbelingen. Gebruik `np.log2`, zodat één stap omhoog één verdubbeling is."),
            Mistake("ten", wrong=_D_TEN,
                    en="`{call}` gives {got}, and it should give {want}: with `np.log10`, the slope counts factors of 10 per day, not doublings. Use `np.log2`, so that one step up is one doubling.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: met `np.log10` telt de helling factoren 10 per dag, geen verdubbelingen. Gebruik `np.log2`, zodat één stap omhoog één verdubbeling is."),
            Mistake("nolog", wrong=_D_NOLOG,
                    en="`{call}` gives {got}, and it should give {want}: your line goes through the minutes themselves. Growth by the same factor every so many days is a straight line only on a log scale, so fit the line through the log base 2 of the minutes.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je lijn gaat door de minuten zelf. Groei met dezelfde factor om de zoveel dagen is alleen op een logschaal een rechte lijn, dus fit de lijn door de log met grondtal 2 van de minuten."),
            Mistake("swap", wrong=_D_SWAP,
                    en="`{call}` gives {got}, and it should give {want}: you fitted the days against the logs, and that line is not the same when the dots do not lie on one line. METR fits the log of the horizon against the release date: `np.polyfit(days, ...)`, with the days first.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je hebt de dagen tegen de logs gefit, en die lijn is niet dezelfde als de stippen niet op één lijn liggen. METR fit de log van de horizon tegen de releasedatum: `np.polyfit(days, ...)`, met de dagen eerst."),
            Mistake("ends", wrong=_D_ENDS,
                    en="`{call}` gives {got}, and it should give {want}: you use only the first and the last model, as the worked example did with two. The fit uses every model, so that one model above or below the trend weighs less.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je gebruikt alleen het eerste en het laatste model, zoals het uitgewerkte voorbeeld met twee deed. De fit gebruikt elk model, zodat één model boven of onder de trend minder zwaar weegt."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: fit a straight line through the log base 2 of `minutes` against `days`, and give back the days per doubling.",
                      "nl": "Vervang de `...`: fit een rechte lijn door de log met grondtal 2 van `minutes` tegen `days`, en geef het aantal dagen per verdubbeling terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Fit a straight line with `np.polyfit` through the log base 2 of the minutes against the days, with degree 1: its slope is the doublings per day. The doubling time is one divided by that slope.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Fit met `np.polyfit` een rechte lijn door de log met grondtal 2 van de minuten tegen de dagen, met graad 1: de helling is het aantal verdubbelingen per dag. De verdubbelingstijd is één gedeeld door die helling."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `days` and `minutes` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie wel de `days` en `minutes` uit de aanroep?"},
            "correct": {"en": "Your `doubling_days` is right in all {n} cases. The 50% time horizon of METR's leading models from 2023 on doubled about every 129 days, in this data of September 2026: METR's own file gives 128.744 days from the unrounded numbers.",
                        "nl": "Je `doubling_days` klopt in alle {n} gevallen. De 50%-time horizon van de beste modellen van METR vanaf 2023 verdubbelde ongeveer elke 129 dagen, in deze data van september 2026: het bestand van METR zelf geeft 128,744 dagen uit de niet-afgeronde getallen."},
        },
    ),
}

CASES = [
    ("success", _S_START, "empty", None),
    ("success", _S, "correct", None),
    ("success", _S_DEF + "    z = np.log2(horizon / minutes) * beta\n    return 1 / (1 + np.exp(-z))\n", "correct", None),
    ("success", "import math\n\n\ndef success(horizon, minutes, beta):\n"
                "    z = (math.log2(horizon) - math.log2(minutes)) * beta\n    return 1 / (1 + math.exp(-z))\n",
     "correct", None),
    ("success", _S_DEF + "    odds = np.exp(beta * np.log2(horizon / minutes))\n    return odds / (1 + odds)\n", "correct", None),
    ("success", _S_DEF + "    odds = (horizon / minutes) ** beta\n    return odds / (1 + odds)\n", "wrong", "natural"),
    ("success", _S_FLIP, "wrong", "flip"),
    ("success", _S.replace("np.exp(-z)", "np.exp(z)"), "wrong", "flip"),
    ("success", _S_NOLOG, "wrong", "nolog"),
    ("success", _S_NATURAL, "wrong", "natural"),
    ("success", _S_TEN, "wrong", "ten"),
    ("success", _S_BRACKETS, "wrong", "brackets"),
    ("success", _S_ODDS, "wrong", "odds"),
    ("success", _S_LOGIT, "wrong", "logit"),
    ("success", _S_DEF + "    return 0.768525\n", "almost", None),
    ("doubling", _D_START, "empty", None),
    ("doubling", _D, "correct", None),
    ("doubling", _D_DEF + "    fit = np.polyfit(days, np.log2(minutes), 1)\n    return 1 / fit[0]\n", "correct", None),
    ("doubling", _D_DEF + "    x = np.array(days, dtype=float)\n    y = np.log2(minutes)\n"
                          "    slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)\n"
                          "    return 1 / slope\n", "correct", None),
    ("doubling", _D_SLOPE, "wrong", "slope"),
    ("doubling", _D_ORDER, "wrong", "order"),
    ("doubling", _D_NATURAL, "wrong", "natural"),
    ("doubling", _D_TEN, "wrong", "ten"),
    ("doubling", _D_NOLOG, "wrong", "nolog"),
    ("doubling", _D_SWAP, "wrong", "swap"),
    ("doubling", _D_ENDS, "wrong", "ends"),
    ("doubling", _D_DEF + "    return 128.698008\n", "almost", None),
]
