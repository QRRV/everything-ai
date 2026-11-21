# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-expert-forecasts: what experts expect. The learner writes describe, the mean, the median
and the 0.25 and 0.75 quantiles of twenty made-up answers to a question like the survey's, and mixture, the chances of
several hypotheses each times its weight and added up, with the weights of Cotra's biological anchors and made-up
chances. A demo adds answers of 100 and shows the mean and the median."""
import contextlib
import copy
import html
import io
import re

import numpy as np

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, same_value, with_time_limit

# --- the data --------------------------------------------------------------------------------------------------------

# made-up: twenty answers, in percent, to a question like the survey's about the chance of extinction or a similarly severe outcome
ANSWERS = [10, 0, 5, 50, 1, 20, 10, 2, 100, 5, 1, 10, 30, 0, 15, 5, 80, 1, 20, 10]
# Cotra (2020): the weights of the six hypotheses and a seventh possibility, in this order: short, medium and long
# horizon neural network, lifetime, genome and evolution anchor, and more computation than any of them
WEIGHTS = [0.2, 0.3, 0.15, 0.05, 0.1, 0.1, 0.1]
CHANCES = [0.5, 0.3, 0.1, 0.9, 0.4, 0.0, 0.0]   # made-up: the chance under each possibility, by one made-up year
DEMO_EXTRA = [0, 1, 2, 3, 4, 5]   # the stops of the demo's slider: how many extra answers of 100


def describe(answers):
    """What the learner writes in Exercise 1: the mean, the median, and the 0.25 and 0.75 quantiles of the answers."""
    a = np.array(answers)
    mean = a.mean()
    median = np.median(a)
    low = np.quantile(a, 0.25)
    high = np.quantile(a, 0.75)
    return [mean, median, low, high]


def mixture(weights, chances):
    """What the learner writes in Exercise 2: each chance times its weight, added up."""
    w = np.array(weights)
    c = np.array(chances)
    return np.sum(w * c)


def demo_answers(extra):
    """The demo: the twenty made-up answers with `extra` more answers of 100."""
    return ANSWERS + [100] * extra


def demo_stats(extra):
    """The demo: the mean and the median of the answers with `extra` more answers of 100."""
    a = np.array(demo_answers(extra))
    return float(a.mean()), float(np.median(a))


def first_triple(values=DEMO_EXTRA):
    """The smallest value of the slider at which the mean is more than three times the median."""
    return next(k for k in values if demo_stats(k)[0] > 3 * demo_stats(k)[1])


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
    if isinstance(v, (bool, np.bool_)):
        return None
    if isinstance(v, np.ndarray) and v.ndim == 0:
        v = v.item()
    if isinstance(v, np.generic):
        v = v.item()
    if not isinstance(v, (int, float)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError, OverflowError):
        return None


def _value(v, digits: int = 4, limit: int = 70) -> str:
    """A result as the check shows it: a number with at most `digits` decimals, a text in quotes, a list of them, else
    what Python shows, cut short when it is longer than `limit`."""
    if isinstance(v, np.ndarray) and v.ndim > 0:
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        text = "[" + ", ".join(_value(x, digits, 10 ** 6) for x in v) + "]"
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


def show_value(v) -> str:
    return _value(v, 4, 200)


def _as_list(v):
    """A tuple or an array of the learner's as a list, so that a right answer in another form counts."""
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, np.ndarray) and v.ndim == 1:
        return v.tolist()
    return v


def same_list(got, want) -> bool:
    """Whether the learner's four numbers are the course's, as a list, a tuple or an array."""
    got = _as_list(got)
    return isinstance(got, list) and same_value(got, list(want))


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


def _pct(v: float, lang: str) -> str:
    """A number of percent as the lesson writes it: at most two decimals, with a decimal comma in Dutch."""
    text = f"{v:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", ",") if lang == "nl" else text


def _stack(values: list, base: float, step: float) -> tuple[list, list]:
    """Dots for a list of answers: one column per value, the same values stacked from `base` up in steps of `step`."""
    xs, ys, seen_ = [], [], {}
    for v in values:
        n = seen_.get(v, 0)
        seen_[v] = n + 1
        xs.append(v)
        ys.append(base + n * step)
    return xs, ys


# --- the demo --------------------------------------------------------------------------------------------------------

_DEMO_WORDS = {
    "en": {"title": "The twenty answers with {k} extra answers of 100",
           "title0": "The twenty made-up answers",
           "x": "answer (percent)", "mean": "mean {m}", "median": "median {d}",
           "said": "With {k} extra answers of 100, the mean is {m} and the median {d}.",
           "said1": "With 1 extra answer of 100, the mean is {m} and the median {d}.",
           "said0": "Of the twenty made-up answers, the mean is {m} and the median {d}.",
           "note": "Made-up answers, not the survey's. Each dot is one answer; the same answers stand in a column. The "
                   "solid line is the median, the dashed line the mean."},
    "nl": {"title": "De twintig antwoorden met {k} extra antwoorden van 100",
           "title0": "De twintig verzonnen antwoorden",
           "x": "antwoord (procent)", "mean": "gemiddelde {m}", "median": "mediaan {d}",
           "said": "Met {k} extra antwoorden van 100 is het gemiddelde {m} en de mediaan {d}.",
           "said1": "Met 1 extra antwoord van 100 is het gemiddelde {m} en de mediaan {d}.",
           "said0": "Van de twintig verzonnen antwoorden is het gemiddelde {m} en de mediaan {d}.",
           "note": "Verzonnen antwoorden, niet die van de enquête. Elke stip is één antwoord; dezelfde antwoorden staan "
                   "in een kolom. De doorgetrokken lijn is de mediaan, de gestippelde lijn het gemiddelde."},
}


def demo_said(extra, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    m, d = demo_stats(extra)
    key = "said0" if extra == 0 else "said1" if extra == 1 else "said"
    return w[key].format(k=extra, m=_pct(m, lang), d=_pct(d, lang))


def demo_frame(v, lang: str) -> tuple[str, str]:
    """The demo: every answer as a dot, the same answers stacked, with the median as a solid line and the mean as a
    dashed one."""
    extra = min(DEMO_EXTRA, key=lambda k: abs(k - float(v)))
    w = _DEMO_WORDS[lang]
    m, d = demo_stats(extra)
    panel = chart.Panel(x=(-4, 104), y=(0, 3.3), xlabel=w["x"], ylabel="", xticks=[0, 20, 40, 60, 80, 100], yticks=[])
    panel.line([d, d], [0, 3.05], role="ink-2", width=2)
    panel.line([m, m], [0, 2.5], role="ink-2", width=2, dashed=True)
    panel.points(*_stack(demo_answers(extra), 0.3, 0.33), role="ink", r=4)
    panel.text(d + 1.2, 2.95, w["median"].format(d=_pct(d, lang)), role="ink-2", anchor="start", halo=True)
    panel.text(m + 1.2, 2.35, w["mean"].format(m=_pct(m, lang)), role="ink-2", anchor="start", halo=True)
    title = w["title0"] if extra == 0 else w["title"].format(k=extra)
    said = demo_said(extra, lang)
    svg = chart.figure([panel], alt=title + ". " + said, width=380, height=220)
    if lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _note(said) + _note(w["note"])
    return yours.figure(title, body, alt=title + ". " + said), said


# --- the pictures under the checks -----------------------------------------------------------------------------------

_DESCRIBE_WORDS = {
    "en": {"title": "Your four numbers for the twenty made-up answers",
           "x": "answer (percent)", "rows": ["mean", "median", "low", "high"], "answers": "answers",
           "head": ["answers", "the course's", "yours"], "stopped": "error",
           "note": "The chart is the first case of the check: the twenty made-up answers at the top, one dot each, and "
                   "below them your four numbers as rings around the course's dots. The table is every case; a "
                   "dashed row is one where your function gives something other than the course's."},
    "nl": {"title": "Jouw vier getallen voor de twintig verzonnen antwoorden",
           "x": "antwoord (procent)", "rows": ["mean", "median", "low", "high"], "answers": "antwoorden",
           "head": ["answers", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De grafiek is het eerste geval van de check: bovenaan de twintig verzonnen antwoorden, elk een stip, "
                   "en daaronder jouw vier getallen als ringen om de stippen van de course. De tabel bevat elk geval; "
                   "een gestippelde rij is er een waar jouw functie iets anders geeft dan die van de course."},
}

DESCRIBE_CASES = [(ANSWERS,), ([1, 2, 3, 4, 90],), ([5],), ([0, 100],), ([10, 10, 10, 10],), ([0, 0, 0, 1],)]


def describe_picture(seen) -> str:
    """The first case: the answers as dots at the top, and the four numbers, the course's as dots and the learner's
    as rings, each on its own row; every case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _DESCRIBE_WORDS[seen.lang]
    got = _all(fn, DESCRIBE_CASES)
    rows = []
    for i, case in enumerate(DESCRIBE_CASES):
        want = describe(*case)
        if got[i] is _STOPPED:
            shown_, off = html.escape(w["stopped"]), True
        else:
            shown_, off = _code(show_value(got[i])), not same_list(got[i], want)
        rows.append(([(_code(_value(case[0], limit=40)), ""), (_code(show_value(want)), ""), (shown_, "font-weight:600")],
                     _OFF if off else ""))
    want = describe(ANSWERS)
    panel = chart.Panel(x=(-4, 132), y=(0.3, 7.6), xlabel=w["x"], ylabel="", xticks=[0, 20, 40, 60, 80, 100], yticks=[])
    panel.points(*_stack(ANSWERS, 5.2, 0.42), role="ink-3", r=3.5)
    panel.text(131, 6.0, w["answers"], role="ink-3", anchor="end", halo=True)
    mine = _as_list(got[0]) if got[0] is not _STOPPED else None
    for k, name in enumerate(w["rows"]):
        y = 4 - k
        panel.line([-4, 104], [y, y], role="ink-3", width=0.6, dashed=True)
        panel.text(131, y - 0.12, name, role="ink-2", anchor="end", halo=True)
        panel.points([float(want[k])], [y], role="ink", r=3)
        f = _number(mine[k]) if isinstance(mine, list) and k < len(mine) else None
        if f is not None and -4 <= f <= 104:
            panel.points([f], [y], role="ink", r=7, hollow=True)
    svg = chart.figure([panel], alt=w["title"], width=380, height=250)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


_MIXTURE_WORDS = {
    "en": {"title": "Your mixture of the seven possibilities",
           "x": "the weights, side by side", "y": "chance", "mixture": "mixture",
           "head": ["weights", "chances", "the course's", "yours"], "stopped": "error",
           "note": "The chart is the first case of the check. Each bar is one possibility, as wide as its weight and as "
                   "high as its made-up chance, from left to right: short, medium and long horizon, lifetime, genome, "
                   "evolution, and more than any of them. The mixture is the average height of the bars, the dashed "
                   "line; on the right, the course's mixture is a dot and yours a ring. The table is every case; a "
                   "dashed row is one where your function gives something other than the course's."},
    "nl": {"title": "Jouw mengsel van de zeven mogelijkheden",
           "x": "gewicht per mogelijkheid, naast elkaar", "y": "kans", "mixture": "mengsel",
           "head": ["weights", "chances", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De grafiek is het eerste geval van de check. Elke balk is een mogelijkheid, zo breed als het gewicht en "
                   "zo hoog als de verzonnen kans, van links naar rechts: korte, middellange en lange horizon, "
                   "lifetime, genome, evolution, en meer dan al die hypotheses voorspellen. Het mengsel is de gemiddelde hoogte "
                   "van de balken, de gestippelde lijn; rechts is het mengsel van de course een stip en het jouwe een "
                   "ring. De tabel bevat elk geval; een gestippelde rij is er een waar jouw functie iets anders geeft "
                   "dan die van de course."},
}

MIXTURE_CASES = [(WEIGHTS, CHANCES), ([0.25, 0.75], [1.0, 0.2]), ([1.0], [0.3]), ([0.6, 0.3, 0.1], [0.0, 0.5, 1.0]),
                 ([0.1, 0.9], [0.9, 0.1]), (WEIGHTS, [1.0] * 7)]


def mixture_picture(seen) -> str:
    """The first case as a bar per possibility, as wide as its weight and as high as its chance, the course's mixture as
    a dashed line, and the course's and the learner's mixture beside it; every case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _MIXTURE_WORDS[seen.lang]
    got = _all(fn, MIXTURE_CASES)
    rows = []
    for i, case in enumerate(MIXTURE_CASES):
        want = mixture(*case)
        if got[i] is _STOPPED:
            shown_, off = html.escape(w["stopped"]), True
        else:
            shown_, off = _code(show_value(got[i])), not same_value(got[i], want)
        rows.append(([(_code(_value(case[0])), ""), (_code(_value(case[1])), ""), (_code(show_value(want)), ""),
                      (shown_, "font-weight:600")], _OFF if off else ""))
    want = float(mixture(WEIGHTS, CHANCES))
    panel = chart.Panel(x=(0, 1.25), y=(0, 1.05), xlabel=w["x"], ylabel=w["y"], xticks=[0, 0.25, 0.5, 0.75, 1],
                        yticks=[0, 0.25, 0.5, 0.75, 1])
    left = 0.0
    for wt, ch in zip(WEIGHTS, CHANCES):
        if ch > 0:
            panel.rect(left, 0, left + wt, ch, role="ink-3")
        else:
            panel.line([left, left + wt], [0.004, 0.004], role="ink-3", width=2)
        left += wt
    panel.line([0, 1.0], [want, want], role="ink-2", width=1.6, dashed=True)
    panel.points([1.12], [want], role="ink", r=3)
    panel.text(1.12, want + 0.1, w["mixture"], role="ink-2", anchor="middle", halo=True)
    f = _number(got[0]) if got[0] is not _STOPPED else None
    if f is not None and 0 <= f <= 1.05:
        panel.points([1.12], [f], role="ink", r=7, hollow=True)
    svg = chart.figure([panel], alt=w["title"], width=380, height=240)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["describe", "mixture", "headline", "pdoom"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# exercise 1: describe; the first case is the twenty made-up answers, then five answers with one far out, one answer,
# two answers far apart, four equal answers, and three zeros with a 1
_D_START = "import numpy as np\n\n\ndef describe(answers):\n    ...\n"
_D = ("import numpy as np\n\n\n"
      "def describe(answers):\n"
      "    a = np.array(answers)\n"
      "    mean = a.mean()\n"
      "    median = np.median(a)\n"
      "    low = np.quantile(a, 0.25)\n"
      "    high = np.quantile(a, 0.75)\n"
      "    return [mean, median, low, high]\n")
_D_SWAP = _D.replace("return [mean, median, low, high]", "return [median, mean, low, high]")
_D_MINMAX = _D.replace("np.quantile(a, 0.25)", "a.min()").replace("np.quantile(a, 0.75)", "a.max()")
_D_PERCENTILE = _D.replace("np.quantile(a, 0.25)", "np.percentile(a, 0.25)").replace("np.quantile(a, 0.75)",
                                                                                     "np.percentile(a, 0.75)")
_D_FLIP = _D.replace("return [mean, median, low, high]", "return [mean, median, high, low]")
_D_PERCENT = _D.replace("np.quantile(a, 0.25)", "np.quantile(a, 25)").replace("np.quantile(a, 0.75)",
                                                                              "np.quantile(a, 75)")
_D_LIST = _D.replace("mean = a.mean()", "mean = answers.mean()")
_D_MIDDLE = _D.replace("median = np.median(a)", "median = a[len(a) // 2]")

# exercise 2: mixture; the first case is Cotra's weights with the made-up chances, then two hypotheses, one, three,
# two the other way round, and every chance 1
_M_START = "import numpy as np\n\n\ndef mixture(weights, chances):\n    ...\n"
_M = ("import numpy as np\n\n\n"
      "def mixture(weights, chances):\n"
      "    w = np.array(weights)\n"
      "    c = np.array(chances)\n"
      "    return np.sum(w * c)\n")
_M_MEAN = _M.replace("np.sum(w * c)", "np.mean(w * c)")
_M_CHANCES = _M.replace("np.sum(w * c)", "np.sum(c)")
_M_AVERAGE = _M.replace("np.sum(w * c)", "np.mean(c)")
_M_PLUS = _M.replace("np.sum(w * c)", "np.sum(w + c)")
_M_WEIGHTS = _M.replace("np.sum(w * c)", "np.sum(w)")
_M_MAX = _M.replace("np.sum(w * c)", "np.max(w * c)")


EXERCISES = {
    "describe": FunctionBox(
        name="describe",
        picture=describe_picture,
        function="describe",
        start=_D_START,
        solution=_D,
        visible=[DESCRIBE_CASES[0]],
        hidden=DESCRIBE_CASES[1:],
        first=[_DOTS],
        show=show_value,
        compare=same_list,
        mistakes=[
            Mistake("percent", error=lambda e: isinstance(e, ValueError) and "Quantiles must be in the range [0, 1]" in str(e),
                    en="`np.quantile` gets a fraction from 0 to 1, not a percentage: a quarter is `0.25`, three quarters `0.75`.",
                    nl="`np.quantile` krijgt een fractie van 0 tot 1, geen percentage: een kwart is `0.25`, driekwart `0.75`."),
            Mistake("list-mean", error=lambda e: isinstance(e, AttributeError) and "'list' object has no attribute" in str(e),
                    en="A list has no `.mean()` or other methods for numbers. Make an array `a` of `answers` with `np.array` first, and work out the four numbers from `a`.",
                    nl="Een lijst heeft geen `.mean()` of andere methoden voor getallen. Maak eerst een array `a` van `answers` met `np.array`, en reken de vier getallen uit met `a`."),
            Mistake("swap", wrong=_D_SWAP,
                    en="`{call}` gives {got}, and it should give {want}: your first two numbers are the wrong way round. The list starts with the mean, then the median.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je eerste twee getallen staan andersom. De lijst begint met het gemiddelde, en dan de mediaan."),
            Mistake("minmax", wrong=_D_MINMAX,
                    en="`{call}` gives {got}, and it should give {want}: your `low` and `high` are the smallest and the largest answer. They should be the 0.25 and 0.75 quantiles, between which half of the answers lie.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `low` en `high` zijn het kleinste en het grootste antwoord. Het moeten het 0,25- en het 0,75-kwantiel zijn, waartussen de helft van de antwoorden ligt."),
            Mistake("percentile", wrong=_D_PERCENTILE,
                    en="`{call}` gives {got}, and it should give {want}: `np.percentile` counts in hundredths, so `0.25` there is 0.25%, almost the smallest answer. Use `np.quantile` with a fraction, or `np.percentile` with a percentage.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `np.percentile` telt in honderdsten, dus `0.25` is daar 0,25%, bijna het kleinste antwoord. Gebruik `np.quantile` met een fractie, of `np.percentile` met een percentage."),
            Mistake("flip", wrong=_D_FLIP,
                    en="`{call}` gives {got}, and it should give {want}: your last two numbers are the wrong way round. First `low`, the 0.25 quantile, then `high`, the 0.75 quantile.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je laatste twee getallen staan andersom. Eerst `low`, het 0,25-kwantiel, en dan `high`, het 0,75-kwantiel."),
            Mistake("middle", wrong=_D_MIDDLE,
                    en="`{call}` gives {got}, and it should give {want}: your median is the answer that happens to stand in the middle of the list. The answers are not sorted: `np.median` finds the middle answer of the sorted answers.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je mediaan is het antwoord dat toevallig midden in de lijst staat. De antwoorden zijn niet gesorteerd: `np.median` vindt het middelste antwoord van de gesorteerde antwoorden."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: make an array `a` of `answers`, work out `mean`, `median`, `low` and `high`, and return them in a list in that order.",
                      "nl": "Vervang de `...`: maak een array `a` van `answers`, reken `mean`, `median`, `low` en `high` uit, en geef ze in die volgorde terug in een lijst."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Make an array `a` of `answers`. `mean` is its mean, `median` its median, `low` its 0.25 quantile and `high` its 0.75 quantile; return `[mean, median, low, high]`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Maak een array `a` van `answers`. `mean` is het gemiddelde ervan, `median` de mediaan, `low` het 0,25-kwantiel en `high` het 0,75-kwantiel; geef `[mean, median, low, high]` terug."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `answers` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie wel de `answers` uit de aanroep?"},
            "correct": {"en": "Your `describe` is right in all {n} cases. Under the check, your rings lie around the course's dots.",
                        "nl": "Je `describe` klopt in alle {n} gevallen. Onder de check liggen jouw ringen om de stippen van de course."},
        },
    ),
    "mixture": FunctionBox(
        name="mixture",
        picture=mixture_picture,
        function="mixture",
        start=_M_START,
        solution=_M,
        visible=[MIXTURE_CASES[0]],
        hidden=MIXTURE_CASES[1:],
        first=[_DOTS],
        show=show_value,
        mistakes=[
            Mistake("mean", wrong=_M_MEAN,
                    en="`{call}` gives {got}, and it should give {want}: you take the mean of the products, so you divide by the number of hypotheses. The weights already add up to 1: add the products up.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je neemt het gemiddelde van de producten, dus je deelt door het aantal hypotheses. De getallen in `weights` tellen al op tot 1: tel de producten op."),
            Mistake("chances", wrong=_M_CHANCES,
                    en="`{call}` gives {got}, and it should give {want}: you add up the chances without their weights, so the answer can be more than 1. Multiply each chance by its weight first.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt de kansen op zonder de kansen te wegen, dus het antwoord kan groter zijn dan 1. Vermenigvuldig eerst elke kans met het gewicht."),
            Mistake("average", wrong=_M_AVERAGE,
                    en="`{call}` gives {got}, and it should give {want}: you take the plain mean of the chances, so every hypothesis counts the same. Each chance counts as much as its weight: multiply, then add up.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je neemt het gewone gemiddelde van de kansen, dus elke hypothese telt even zwaar. Elke kans telt zo zwaar als het gewicht: vermenigvuldig, en tel dan op."),
            Mistake("plus", wrong=_M_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: you add each weight to its chance. A weight scales its chance: multiply them.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt elk gewicht bij de kans op. Een gewicht schaalt de kans: vermenigvuldig ze."),
            Mistake("weights", wrong=_M_WEIGHTS,
                    en="`{call}` gives {got}, and it should give {want}: you add up the weights alone, which always gives 1 here. Multiply each weight by its chance first.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt alleen de getallen in `weights` op, en dat geeft hier altijd 1. Vermenigvuldig eerst elk gewicht met de kans."),
            Mistake("max", wrong=_M_MAX,
                    en="`{call}` gives {got}, and it should give {want}: you give the largest product, the share of one hypothesis. Every hypothesis adds its share: add the products up.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft het grootste product, het deel van één hypothese. Elke hypothese levert een deel: tel de producten op."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: make arrays `w` of `weights` and `c` of `chances`, and return each chance times its weight, added up.",
                      "nl": "Vervang de `...`: maak arrays `w` van `weights` en `c` van `chances`, en geef elke kans maal het gewicht terug, opgeteld."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Make arrays `w` of `weights` and `c` of `chances`. Multiply them, which multiplies each weight by its own chance, and return the sum of the products.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Maak arrays `w` van `weights` en `c` van `chances`. Vermenigvuldig ze, zodat elk gewicht met de eigen kans wordt vermenigvuldigd, en geef de som van de producten terug."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `weights` and `chances` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie wel de `weights` en `chances` uit de aanroep?"},
            "correct": {"en": "Your `mixture` is right in all {n} cases. Under the check, your ring lies around the course's dot.",
                        "nl": "Je `mixture` klopt in alle {n} gevallen. Onder de check ligt jouw ring om de stip van de course."},
        },
    ),
}

CASES = [
    ("describe", _D_START, "empty", None),
    ("describe", _D, "correct", None),
    ("describe", "import numpy as np\n\n\ndef describe(answers):\n    a = np.array(answers)\n    return [np.mean(a), np.median(a), np.percentile(a, 25), np.percentile(a, 75)]\n", "correct", None),
    ("describe", "import numpy as np\n\n\ndef describe(answers):\n    a = np.array(answers)\n    return a.mean(), np.median(a), np.quantile(a, 0.25), np.quantile(a, 0.75)\n", "correct", None),
    ("describe", "import numpy as np\n\n\ndef describe(answers):\n    return np.array([np.mean(answers), np.median(answers), *np.quantile(answers, [0.25, 0.75])])\n", "correct", None),
    ("describe", _D_PERCENT, "error", "percent"),
    ("describe", _D_LIST, "error", "list-mean"),
    ("describe", _D_SWAP, "wrong", "swap"),
    ("describe", _D_MINMAX, "wrong", "minmax"),
    ("describe", _D_PERCENTILE, "wrong", "percentile"),
    ("describe", _D_FLIP, "wrong", "flip"),
    ("describe", _D_MIDDLE, "wrong", "middle"),
    ("describe", "import numpy as np\n\n\ndef describe(answers):\n    return " + repr([float(x) for x in describe(ANSWERS)]) + "\n", "almost", None),
    ("mixture", _M_START, "empty", None),
    ("mixture", _M, "correct", None),
    ("mixture", "import numpy as np\n\n\ndef mixture(weights, chances):\n    total = 0\n    for k in range(len(weights)):\n        total = total + weights[k] * chances[k]\n    return total\n", "correct", None),
    ("mixture", "import numpy as np\n\n\ndef mixture(weights, chances):\n    return np.dot(weights, chances)\n", "correct", None),
    ("mixture", "import numpy as np\n\n\ndef mixture(weights, chances):\n    return sum(w * c for w, c in zip(weights, chances))\n", "correct", None),
    ("mixture", _M_MEAN, "wrong", "mean"),
    ("mixture", _M_CHANCES, "wrong", "chances"),
    ("mixture", _M_AVERAGE, "wrong", "average"),
    ("mixture", _M_PLUS, "wrong", "plus"),
    ("mixture", _M_WEIGHTS, "wrong", "weights"),
    ("mixture", _M_MAX, "wrong", "max"),
    ("mixture", "import numpy as np\n\n\ndef mixture(weights, chances):\n    return 0.29\n", "almost", None),
]
