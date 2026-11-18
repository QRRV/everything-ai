# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-pdp-ice: a random forest on three columns of UCI Adult, grown on one half of the table.
Set age to one value for every person, ask the forest and average: a partial dependence plot; keep one line per person
instead: ICE curves. The learner writes one person's line for a tree of depth 2, averages the lines of the example of
Goldstein and colleagues, where the average is flat while every line is steep, says what a flat average does not show,
and what is wrong at the left end of the curve for age; with the slider over education, the learner's lines for 40
people of the test half, and the learner's average under the steep lines."""
import functools
import html
import math
import random
import re

from everything_ai import chart, yours
from everything_ai.check import CodeBox, FunctionBox, Mistake

COLUMNS = ["age", "education_num", "hours_per_week"]
AGES = list(range(17, 91))   # every age from 17 to 90, the youngest and the oldest in the table
SOME = 1000                  # the worked example averages over the first 1000 people of the test half
EDUCATION = list(range(1, 17))   # the stops of the slider: education_num
PER_LEVEL = 12               # the slider draws the lines of the first 12 people of the test half with that education
EDUCATION_NAMES = {1: "Preschool", 2: "1st-4th", 3: "5th-6th", 4: "7th-8th", 5: "9th", 6: "10th", 7: "11th",
                   8: "12th", 9: "HS-grad", 10: "Some-college", 11: "Assoc-voc", 12: "Assoc-acdm", 13: "Bachelors",
                   14: "Masters", 15: "Prof-school", 16: "Doctorate"}


@functools.lru_cache(maxsize=1)
def table():
    """The three columns of UCI Adult as floats, who earns more than 50,000 dollars, and the positions of the training
    half and of the test half (the table shuffled with seed 0 and cut in two). Read-only."""
    import numpy as np
    import pandas as pd
    from everything_ai import paths
    people = pd.read_csv(paths.ROOT / "data" / "adult.csv.gz")
    X = people[COLUMNS].to_numpy().astype(float)
    y = (people["income"] == ">50K").to_numpy()
    order = np.random.default_rng(0).permutation(len(y))
    train, test = order[: len(y) // 2], order[len(y) // 2:]
    for a in (X, y, train, test):
        a.flags.writeable = False
    return X, y, train, test


@functools.lru_cache(maxsize=1)
def forest():
    """The forest of the worked example: 100 trees whose leaves hold at least 20 people, grown on the training half.
    About half a second."""
    from sklearn.ensemble import RandomForestClassifier
    X, y, train, test = table()
    return RandomForestClassifier(min_samples_leaf=20, random_state=0).fit(X[train], y[train])


def lines_of(rows) -> "np.ndarray":
    """The forest's ICE curves for these rows: what it says for each row with age set to each value in AGES, one row
    of the result per person."""
    import numpy as np
    rows = np.asarray(rows, dtype=float)
    every = np.repeat(rows, len(AGES), axis=0)
    every[:, 0] = np.tile(AGES, len(rows))
    return forest().predict_proba(every)[:, 1].reshape(len(rows), len(AGES))


@functools.lru_cache(maxsize=1)
def everyone() -> tuple:
    """The partial dependence of the forest on age over the first 1000 people of the test half, at every age in AGES."""
    X, y, train, test = table()
    return tuple(float(v) for v in lines_of(X[test][:SOME]).mean(axis=0))


def level_rows(e: int):
    """The first PER_LEVEL people of the test half with education_num e."""
    X, y, train, test = table()
    rows = X[test]
    return rows[rows[:, 1] == e][:PER_LEVEL]


def rise(line) -> float:
    """How much a curve over AGES rises from age 25 to age 45."""
    return float(line[AGES.index(45)] - line[AGES.index(25)])


def _show(x: float, lang: str, digits: int = 2) -> str:
    """A number with this many decimals and a real minus sign; a decimal comma in Dutch."""
    text = f"{x:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    text = text.replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _signed(x: float, lang: str) -> str:
    """A change with its sign: +0.48, −0.02."""
    text = _show(x, lang)
    return text if text.startswith("−") else "+" + text


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"

_LEVEL_WORDS = {
    "en": {"title": "the forest's ICE curves for age", "x": "age", "y": "chance of a yes",
           "alt": "Thin lines: for {n} people of the test half with education_num {e}, what the forest says when their "
                  "age is set to each age from 17 to 90. A thick line: their average. A dashed line: the average over "
                  "the 1000 people of the worked example. From age 25 to 45, their lines rise by {lo} to {hi}; the "
                  "average over the 1000 rises by {all}.",
           "said": "{n} people with education_num {e} ({name}): from age 25 to 45 their lines rise by {lo} to {hi}, "
                   "<span class=\"c-pred ea-num\">{mean}</span> on average. The average over all 1000 people, the "
                   "partial dependence, rises by {all}."},
    "nl": {"title": "de ICE curves van het forest voor leeftijd", "x": "leeftijd", "y": "kans op een ja",
           "alt": "Dunne lijnen: voor {n} mensen van de testhelft met education_num {e}, wat het forest zegt als hun "
                  "leeftijd op elke leeftijd van 17 tot 90 wordt gezet. Een dikke lijn: hun gemiddelde. Een gestippelde "
                  "lijn: het gemiddelde over de 1000 mensen van het uitgewerkte voorbeeld. Van 25 tot 45 jaar stijgen "
                  "hun lijnen met {lo} tot {hi}; het gemiddelde over de 1000 stijgt met {all}.",
           "said": "{n} mensen met education_num {e} ({name}): van 25 tot 45 jaar stijgen hun lijnen met {lo} tot {hi}, "
                   "gemiddeld <span class=\"c-pred ea-num\">{mean}</span>. Het gemiddelde over alle 1000 mensen, de "
                   "partial dependence, stijgt met {all}."},
}


def level_picture(e: int, lang: str) -> tuple[str, str]:
    """For the slider of *The idea*: the forest's ICE curves for age of the first 12 people of the test half with
    education_num e, their average, and the average over the 1000 people of the worked example; and the sentence."""
    w = _LEVEL_WORDS[lang]
    e = int(e)
    lines = lines_of(level_rows(e))
    p = chart.Panel(x=(17, 90), y=(0, 1), title=w["title"], xlabel=w["x"], ylabel=w["y"],
                    yticks=[0, 0.25, 0.5, 0.75, 1])
    for line in lines:
        p.line(AGES, line, role="ink-3", width=1.1)
    mean = lines.mean(axis=0)
    p.line(AGES, everyone(), role="pred", width=2, dashed=True)
    p.line(AGES, mean, role="pred", width=3)
    rises = [rise(line) for line in lines]
    fill = {"n": len(lines), "e": e, "name": EDUCATION_NAMES[e], "lo": _signed(min(rises), lang),
            "hi": _signed(max(rises), lang), "mean": _signed(rise(mean), lang), "all": _signed(rise(everyone()), lang)}
    svg = chart.figure([p], alt=w["alt"].format(**fill), width=460, height=280)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    return svg, w["said"].format(**fill)


# --- exercise 1: one person's line --------------------------------------------------------------------------------

# A tree of depth 2 (random_state=0) grown on the training half from the three columns: the share of people with a
# yes in each leaf, as the test grows it again, and rounded to two decimals as the box writes them.
TREE2 = {"splits": [(1, 12.5), (0, 33.5), (0, 28.5)], "leaves": [0.0557, 0.2362, 0.1112, 0.5536]}


def model(row) -> float:
    """The tree of depth 2, as the box writes it."""
    if row[1] <= 12.5:
        if row[0] <= 33.5:
            return 0.06
        return 0.24
    if row[0] <= 28.5:
        return 0.11
    return 0.55


PICTURED = 40   # the picture of exercise 1 runs the learner's ice() for the first 40 people of the test half


def _first_rows() -> list:
    X, y, train, test = table()
    out = []
    for row in X[test][:PICTURED].tolist():
        out.append([int(v) for v in row])
    return out


def _mine(seen):
    """The learner's function, or None when the code stopped or the name is not a function."""
    mine = seen.mine
    if mine is None:
        return None
    fn = mine.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _number(v):
    """A float from what a function gave back, or None when it is not one number."""
    if isinstance(v, bool):
        return None
    if hasattr(v, "item") and getattr(v, "size", 1) == 1:
        v = v.item()
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    return None


def _numbers(got, n: int):
    """A list of n floats from what the learner's code gave, or None."""
    if not isinstance(got, (list, tuple)) or len(got) != n:
        return None
    out = [_number(v) for v in got]
    return None if any(v is None for v in out) else out


def _at(ages: list, lang: str, most: int) -> str:
    """Ages in words, at most `most` of them: "age 34", "ages 29 and 34", "ages 18, 19, 20 and more"; in Dutch
    "34 jaar", "29 en 34 jaar", "18, 19, 20 jaar en verder"."""
    words = [str(a) for a in ages]
    if len(words) > most:
        listed = ", ".join(words[:most - 1])
        return f"ages {listed} and more" if lang == "en" else f"{listed} jaar en verder"
    listed = words[0] if len(words) == 1 else ", ".join(words[:-1]) + (" and " if lang == "en" else " en ") + words[-1]
    if lang == "en":
        return ("age " if len(words) == 1 else "ages ") + listed
    return listed + " jaar"


def _shape(line: list, lang: str) -> str:
    """One kind of line in words: where it changes, and from what to what."""
    steps = [AGES[k] for k in range(1, len(line)) if not math.isclose(line[k], line[k - 1], abs_tol=1e-9)]
    if not steps:
        return ("{v} at every age" if lang == "en" else "{v} bij elke leeftijd").format(v=_show(line[0], lang))
    if lang == "en":
        said = "from {a} to {b}, changing at {at}"
    elif len(steps) == 1:
        said = "van {a} naar {b}, met een sprong bij {at}"
    else:
        said = "van {a} naar {b}, met sprongen bij {at}"
    return said.format(a=_show(line[0], lang), b=_show(line[-1], lang), at=_at(steps, lang, 4))


_ICE_WORDS = {
    "en": {"title": "Your ice(), for 40 people of the test half", "x": "age", "y": "what the tree says",
           "alt": "Thin lines: what your ice() gives for 40 people of the test half, called with column 0 and the ages "
                  "from 17 to 90 as the grid. A thick line: their average. {kinds} {average}",
           "person": "person", "people": "people",
           "kinds": "{k} different lines: {parts}.",
           "moves": "Their average changes at {avg}.",
           "pd": "Their average, the partial dependence, changes at {avg}.",
           "still": "Their average is {v} at every age.",
           "key": "{kinds} {average}"},
    "nl": {"title": "Jouw ice(), voor 40 mensen van de testhelft", "x": "leeftijd", "y": "wat de boom zegt",
           "alt": "Dunne lijnen: wat je ice() geeft voor 40 mensen van de testhelft, aangeroepen met kolom 0 en de "
                  "leeftijden van 17 tot 90 als grid. Een dikke lijn: hun gemiddelde. {kinds} {average}",
           "person": "persoon", "people": "mensen",
           "kinds": "{k} verschillende lijnen: {parts}.",
           "moves": "Hun gemiddelde verandert bij {avg}.",
           "pd": "Hun gemiddelde, de partial dependence, verandert bij {avg}.",
           "still": "Hun gemiddelde is {v} bij elke leeftijd.",
           "key": "{kinds} {average}"},
}


def _ice_picture(seen) -> str:
    fn = _mine(seen)
    if fn is None:
        return ""
    rows = _first_rows()

    def run():
        out = []
        for row in rows:
            line = _numbers(fn(list(row), 0, list(AGES)), len(AGES))
            if line is None or any(not -10 <= v <= 10 for v in line):
                return None
            out.append(line)
        return out

    from everything_ai import check as checking
    try:
        lines = checking.with_time_limit(run)
    except Exception:
        return ""
    if lines is None:
        return ""
    lang = seen.lang
    w = _ICE_WORDS[lang]
    kinds = {}
    for line in lines:
        kinds.setdefault(tuple(round(v, 9) for v in line), []).append(line)
    parts = []
    for same in sorted(kinds.values(), key=len, reverse=True)[:4]:
        parts.append(f"{len(same)} {w['person'] if len(same) == 1 else w['people']}: " + _shape(same[0], lang))
    if len(kinds) > 4:
        parts.append("…")
    mean = [sum(line[k] for line in lines) / len(lines) for k in range(len(AGES))]
    avg = [AGES[k] for k in range(1, len(AGES)) if not math.isclose(mean[k], mean[k - 1], abs_tol=1e-9)]
    # the average is the partial dependence only when every line is the person's own ICE curve
    truth = []
    for row in rows:
        line = []
        for age in AGES:
            changed = list(row)
            changed[0] = age
            line.append(model(changed))
        truth.append(line)
    own = all(math.isclose(a, b, abs_tol=1e-9) for mine, right in zip(lines, truth) for a, b in zip(mine, right))
    if not avg:
        average = w["still"].format(v=_show(mean[0], lang))
    else:
        average = w["pd" if own else "moves"].format(avg=_at(avg, lang, 6))
    k = len(kinds)
    kinds_text = w["kinds"].format(k=k, parts="; ".join(parts)) if k > 1 else (
        ("Every line is the same: " if lang == "en" else "Elke lijn is gelijk: ") + _shape(lines[0], lang) + ".")
    fill = {"kinds": kinds_text, "average": average}
    top = max(1.0, max(max(line) for line in lines) * 1.1)
    low = min(0.0, min(min(line) for line in lines) * 1.1)
    p = chart.Panel(x=(17, 90), y=(low, top), title="", xlabel=w["x"], ylabel=w["y"])
    for line in lines:
        p.line(AGES, line, role="ink-3", width=1.1)
    p.line(AGES, mean, role="pred", width=3)
    svg = chart.figure([p], alt=w["alt"].format(**fill), width=520, height=260)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(w["key"].format(**fill))}</div>')


# --- exercise 2: the average of the lines, in the example of Goldstein and colleagues -----------------------------

def gold_model(row) -> float:
    """The formula of Goldstein and colleagues without its noise, as the box writes it."""
    x1, x2, x3 = row
    if x3 >= 0:
        return 0.2 * x1 - 5 * x2 + 10 * x2
    return 0.2 * x1 - 5 * x2


def gold_rows() -> list:
    """The 1000 rows the box draws with random.seed(0)."""
    rng = random.Random(0)
    rows = []
    for i in range(1000):
        rows.append([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)])
    return rows


GOLD_GRID = [-1, -0.5, 0, 0.5, 1]
DRAWN = 40   # the picture of exercise 2 draws the lines of the first 40 rows

_AVG_WORDS = {
    "en": {"title": "Your average, under the lines of 40 rows", "x": "X2", "y": "what the model says",
           "alt": "Thin lines: for each of the first 40 rows, what the model says with X2 set to each value from −1 to 1: "
                  "{up} lines rise by 10, {down} fall by 10. A thick line with dots: your average, from {a} at X2 = −1 "
                  "to {b} at X2 = 1.",
           "key": "Your average goes from {a} at X2 = −1 to {b} at X2 = 1. Of the 40 lines drawn, {up} rise by 10 and "
                  "{down} fall by 10: every line has a slope of +5 or −5.",
           "flat": " Your average hardly moves: the lines that rise and the lines that fall cancel out."},
    "nl": {"title": "Jouw gemiddelde, onder de lijnen van 40 rijen", "x": "X2", "y": "wat het model zegt",
           "alt": "Dunne lijnen: voor elk van de eerste 40 rijen, wat het model zegt met X2 op elke waarde van −1 tot 1: "
                  "{up} lijnen stijgen met 10, {down} dalen met 10. Een dikke lijn met punten: je gemiddelde, van {a} "
                  "bij X2 = −1 tot {b} bij X2 = 1.",
           "key": "Je gemiddelde gaat van {a} bij X2 = −1 naar {b} bij X2 = 1. Van de 40 getekende lijnen stijgen er "
                  "{up} met 10 en dalen er {down} met 10: elke lijn heeft een helling van +5 of −5.",
           "flat": " Je gemiddelde beweegt nauwelijks: de lijnen die stijgen en de lijnen die dalen heffen elkaar op."},
}


def _avg_picture(seen) -> str:
    mine = seen.mine
    if mine is None:
        return ""
    grid = _numbers(mine.names.get("grid"), len(GOLD_GRID))
    got = _numbers(mine.value, len(GOLD_GRID))
    if grid is None or got is None or grid != [float(v) for v in GOLD_GRID]:
        return ""
    rows = mine.names.get("rows")
    if not isinstance(rows, list) or len(rows) < DRAWN or any(_numbers(r, 3) is None for r in rows[:DRAWN]):
        return ""
    lang = seen.lang
    w = _AVG_WORDS[lang]
    drawn = [_numbers(r, 3) for r in rows[:DRAWN]]
    # the lines cancel out in the average only when the average is the partial dependence of the box's rows
    every = [_numbers(r, 3) for r in rows]
    truth = None
    if all(r is not None for r in every):
        truth = [sum(gold_model([r[0], v, r[2]]) for r in every) / len(every) for v in GOLD_GRID]
    up = sum(1 for r in drawn if r[2] >= 0)
    fill = {"a": _show(got[0], lang, 3), "b": _show(got[-1], lang, 3), "up": up, "down": DRAWN - up}
    ends = []
    for r in drawn:
        ends.extend([gold_model([r[0], -1, r[2]]), gold_model([r[0], 1, r[2]])])
    top = max(6.0, max(got) + 0.5, max(ends))
    low = min(-6.0, min(got) - 0.5, min(ends))
    p = chart.Panel(x=(-1, 1), y=(low, top), title="", xlabel=w["x"], ylabel=w["y"], xticks=[-1, -0.5, 0, 0.5, 1])
    for r in drawn:
        p.line([-1, 1], [gold_model([r[0], -1, r[2]]), gold_model([r[0], 1, r[2]])], role="ink-3", width=1.1)
    p.line(GOLD_GRID, got, role="pred", width=3)
    p.points(GOLD_GRID, got, role="pred", r=4.5)
    true = truth is not None and all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(got, truth))
    key = w["key"].format(**fill) + (w["flat"] if true and max(got) - min(got) < 0.5 else "")
    svg = chart.figure([p], alt=w["alt"].format(**fill), width=520, height=260)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(key)}</div>')


# The lesson is done when these are correct. "flat" and "absurd" are choice questions.
REQUIRED = ["ice", "average", "flat", "absurd"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

# --- exercise 1
_E_HEAD = ("def model(row):\n"
           "    # a tree of depth 2 grown on the training half: the share of yes in each leaf, rounded\n"
           "    if row[1] <= 12.5:\n"
           "        if row[0] <= 33.5:\n"
           "            return 0.06\n"
           "        return 0.24\n"
           "    if row[0] <= 28.5:\n"
           "        return 0.11\n"
           "    return 0.55\n"
           "\n"
           "\n"
           "def ice(row, j, grid):\n"
           "    # row: one person, such as [30, 10, 40] for age, education_num and hours_per_week\n"
           "    # the line: what model says for row with column j set to each value of grid in turn\n")
_E_START = _E_HEAD + "    ...\n"


def _e(copy: str = "        changed = list(row)\n", put: str = "        changed[j] = v\n",
       add: str = "        line.append(model(changed))\n", ret: str = "    return line\n") -> str:
    """The body of ice with one part changed: the copy, which column gets the value, what is added, the return."""
    return _E_HEAD + "    line = []\n    for v in grid:\n" + copy + put + add + ret


_E_SOLUTION = _e()
_E_VISIBLE = [([30, 10, 40], 0, [20, 30, 40, 50])]
_E_HIDDEN = [
    ([45, 14, 50], 0, [20, 25, 30, 35]),
    ([40, 10, 40], 1, [9, 12, 13, 16]),
    ([25, 16, 60], 2, [20, 40, 60]),
    ([33, 12, 40], 0, [33, 34]),
    ([28, 9, 40], 1, [13]),
    ([50, 13, 40], 0, [17, 90, 29]),
]
_E_OWN = _e(add="        line.append(model(row))\n")
_E_VALUES = _e(add="        line.append(v)\n")
_E_FIRST = _e(add="        line.append(model(changed))\n        return line\n", ret="")
_E_AGE = _e(put="        changed[0] = v\n")

# --- exercise 2
_G_HEAD = ("import random\n"
           "\n"
           "\n"
           "def model(row):\n"
           "    # the formula of Goldstein and colleagues, without its noise: 0.2 X1 - 5 X2, and + 10 X2 where X3 >= 0\n"
           "    x1, x2, x3 = row\n"
           "    if x3 >= 0:\n"
           "        return 0.2 * x1 - 5 * x2 + 10 * x2\n"
           "    return 0.2 * x1 - 5 * x2\n"
           "\n"
           "\n"
           "random.seed(0)\n"
           "rows = []\n"
           "for i in range(1000):\n"
           "    rows.append([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])\n"
           "grid = [-1, -0.5, 0, 0.5, 1]\n"
           "average = []\n")
_G_LOOP = "for v in grid:\n"
_G_START = _G_HEAD + _G_LOOP + "    ...\naverage\n"


def _g(start: str = "    total = 0\n", put: str = "        changed[1] = v\n",
       add: str = "    average.append(total / len(rows))\n", before: str = "") -> str:
    """The loop of exercise 2 with one part changed: where the total starts, which column gets the value, what is
    added to average, and a line before the loop."""
    return (_G_HEAD + before + _G_LOOP + start + "    for row in rows:\n        changed = list(row)\n" + put
            + "        total = total + model(changed)\n" + add + "average\n")


_G_SOLUTION = _g()
_G_NO_SET = _g(put="")
_G_COLUMN = _g(put="        changed[0] = v\n")
_G_SUM = _g(add="    average.append(total)\n")
_G_GRID = _g(add="    average.append(total / len(grid))\n")
_G_RUNNING = _g(start="", before="total = 0\n")
_G_INSIDE = _g(add="").replace("        total = total + model(changed)\n",
                               "        total = total + model(changed)\n        average.append(total / len(rows))\n")


EXERCISES = {
    "ice": FunctionBox(
        name="ice",
        picture=_ice_picture,
        function="ice",
        start=_E_START,
        solution=_E_SOLUTION,
        visible=_E_VISIBLE,
        hidden=_E_HIDDEN,
        first=[_DOTS],
        mistakes=[
            Mistake("own-row", wrong=_E_OWN,
                    en="`{call}` gives {got}, and it should give {want}. Every value in your line is what `model` says for the row as it is, so the line never moves. Give `model` the copy in which column `j` has the value from the grid: `model(changed)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Elke waarde in je lijn is wat `model` zegt voor de rij zoals hij is, dus de lijn beweegt nooit. Geef `model` de kopie waarin kolom `j` de waarde uit de grid heeft: `model(changed)`."),
            Mistake("values", wrong=_E_VALUES,
                    en="`{call}` gives {got}, and it should give {want}. Your line holds the values of `grid` themselves. For each value, add what `model` says for the copy with that value in column `j`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je lijn bevat de waarden van `grid` zelf. Voeg voor elke waarde toe wat `model` zegt voor de kopie met die waarde in kolom `j`."),
            Mistake("first-value", wrong=_E_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: only what the model says for the first value of the grid. If your `return` stands inside the `for` loop, Python leaves the function after the first value. Put `return line` after the loop, as far in as `line = []`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen wat het model zegt bij de eerste waarde van de grid. Als je `return` binnen de `for`-loop staat, verlaat Python de functie na de eerste waarde. Zet `return line` na de loop, even ver ingesprongen als `line = []`."),
            Mistake("age-only", wrong=_E_AGE,
                    en="`{call}` gives {got}, and it should give {want}. Your line changes age, column 0, whatever `j` is. Put the value in column `j`: `changed[j] = v`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je lijn verandert de leeftijd, kolom 0, wat `j` ook is. Zet de waarde in kolom `j`: `changed[j] = v`."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...` with the body of `ice`: start an empty list, and for each value of `grid`, copy the row, put the value in column `j`, and add what `model` says for the copy. Then return the list.",
                      "nl": "Vervang de `...` door de inhoud van `ice`: begin een lege lijst, en kopieer voor elke waarde van `grid` de rij, zet de waarde in kolom `j` en voeg toe wat `model` zegt voor de kopie. Geef daarna de lijst terug."},
            "cleared": _CLEARED,
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Work it out by hand: copy the row, set column `j` to the first value of the grid, and follow the questions of `model`. Then the next value. Which step is different in your function?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Reken het met de hand uit: kopieer de rij, zet kolom `j` op de eerste waarde van de grid, en volg de vragen van `model`. Dan de volgende waarde. Welke stap is anders in je functie?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for any row, any column `j` and any grid, and does it use `row`, `j` and `grid` as they are given?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke rij, elke kolom `j` en elke grid, en gebruikt hij `row`, `j` en `grid` zoals ze worden meegegeven?"},
            "correct": {"en": "Your `ice` is right in all {n} cases. The picture runs it for 40 people of the test half at every age from 17 to 90: each line has one step, at 29 or at 34, and their average has two.",
                        "nl": "Je `ice` klopt in alle {n} gevallen. De tekening voert hem uit voor 40 mensen van de testhelft bij elke leeftijd van 17 tot 90: elke lijn heeft één sprong, bij 29 of bij 34, en hun gemiddelde heeft er twee."},
        },
    ),
    "average": CodeBox(
        name="average",
        picture=_avg_picture,
        start=_G_START,
        solution=_G_SOLUTION,
        first=[_DOTS],
        mistakes=[
            Mistake("no-set", wrong=_G_NO_SET,
                    en="Your `average` is the same number five times: what the model says on average for the rows as they are. Each row keeps its own X2, so the value of the grid never reaches the model. In the copy, set column 1 to the value: `changed[1] = v`.",
                    nl="Je `average` is vijf keer hetzelfde getal: wat het model gemiddeld zegt voor de rijen zoals ze zijn. Elke rij houdt zijn eigen X2, dus de waarde van de grid komt nooit bij het model. Zet in de kopie kolom 1 op de waarde: `changed[1] = v`."),
            Mistake("column", wrong=_G_COLUMN,
                    en="Your `average` is what you get when you set X1, column 0, to each value of the grid. X2 is column 1 (Python counts from 0): `changed[1] = v`.",
                    nl="Je `average` is wat je krijgt als je X1, kolom 0, op elke waarde van de grid zet. X2 is kolom 1 (Python telt vanaf 0): `changed[1] = v`."),
            Mistake("sum", wrong=_G_SUM,
                    en="Your numbers are 1000 times too big: they are the totals over the 1000 rows. The average is the total divided by the number of rows, `len(rows)`.",
                    nl="Je getallen zijn 1000 keer te groot: het zijn de totalen over de 1000 rijen. Het gemiddelde is het totaal gedeeld door het aantal rijen, `len(rows)`."),
            Mistake("grid-length", wrong=_G_GRID,
                    en="Your numbers are 200 times too big: you divide the total by `len(grid)`, the 5 values of the grid. The average is over the rows: divide by `len(rows)`.",
                    nl="Je getallen zijn 200 keer te groot: je deelt het totaal door `len(grid)`, de 5 waarden van de grid. Het gemiddelde gaat over de rijen: deel door `len(rows)`."),
            Mistake("running", wrong=_G_RUNNING,
                    en="Your total keeps growing from one value of the grid to the next, so each number also counts the values before it. Start `total = 0` inside the loop over the grid, before the loop over the rows.",
                    nl="Je totaal blijft groeien van de ene waarde van de grid naar de volgende, dus elk getal telt ook de waarden ervoor mee. Begin `total = 0` binnen de loop over de grid, vóór de loop over de rijen."),
            Mistake("inside", wrong=_G_INSIDE,
                    en="Your `average` has 5000 numbers instead of 5: you add to it once for every row, inside the loop over the rows. Add once for each value of the grid, after the loop over the rows.",
                    nl="Je `average` heeft 5000 getallen in plaats van 5: je voegt er bij elke rij iets aan toe, binnen de loop over de rijen. Voeg één keer per waarde van de grid toe, na de loop over de rijen."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...`: for each value `v` of the grid, add up what `model` says for each row with column 1 set to `v`, and add that total divided by the number of rows to `average`.",
                      "nl": "Vervang de `...`: tel voor elke waarde `v` van de grid op wat `model` zegt voor elke rij met kolom 1 op `v`, en voeg dat totaal gedeeld door het aantal rijen toe aan `average`."},
            "cleared": _CLEARED,
            "no_value": {"en": "The last line of the box shows nothing. Put `average` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `average` op de laatste regel, zodat de check hem kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. For each value `v` of the grid: copy each row, set its column 1 to `v`, add up what `model` says, and divide by `len(rows)`. Which step is different in your code?",
                      "nl": "Je laatste regel geeft `{got}`. Voor elke waarde `v` van de grid: kopieer elke rij, zet kolom 1 op `v`, tel op wat `model` zegt en deel door `len(rows)`. Welke stap is anders in je code?"},
            "correct": {"en": "This is the partial dependence on X2: from 0.063 at −1 to −0.057 at 1, almost flat. Yet X2 is in every line of the model, with a slope of +5 or −5: 494 of the 1000 rows have X3 ≥ 0. The picture shows your average under the lines.",
                        "nl": "Dit is de partial dependence op X2: van 0,063 bij −1 tot −0,057 bij 1, bijna vlak. Toch zit X2 in elke lijn van het model, met een helling van +5 of −5: 494 van de 1000 rijen hebben X3 ≥ 0. De tekening laat je gemiddelde zien onder de lijnen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("ice", _E_START, "empty", None),
    ("ice", _E_SOLUTION, "correct", None),
    ("ice", _E_HEAD + "    line = []\n    for value in grid:\n        line.append(model(row[:j] + [value] + row[j + 1:]))\n"
            "    return line\n", "correct", None),
    ("ice", _E_OWN, "wrong", "own-row"),
    ("ice", _E_VALUES, "wrong", "values"),
    ("ice", _E_FIRST, "wrong", "first-value"),
    ("ice", _E_AGE, "wrong", "age-only"),
    ("ice", _E_HEAD + "    return [0.06, 0.06, 0.24, 0.24]\n", "almost", None),
    ("ice", _e(put="        changed[j] = grid[0]\n"), "wrong", None),
    ("ice", _e(ret="    return\n"), "wrong", "unreached"),
    ("ice", _E_HEAD + "    ...\n    return [0.06, 0.06, 0.24, 0.24]\n", "wrong", "dots-left"),
    ("average", _G_START, "empty", None),
    ("average", _G_SOLUTION, "correct", None),
    ("average", _G_HEAD + _G_LOOP + "    total = 0\n    for x1, x2, x3 in rows:\n        total += model([x1, v, x3])\n"
                "    average.append(total / 1000)\naverage\n", "correct", None),
    ("average", _G_NO_SET, "wrong", "no-set"),
    ("average", _G_COLUMN, "wrong", "column"),
    ("average", _G_SUM, "wrong", "sum"),
    ("average", _G_GRID, "wrong", "grid-length"),
    ("average", _G_RUNNING, "wrong", "running"),
    ("average", _G_INSIDE, "wrong", "inside"),
    ("average", _g(put="        changed[2] = v\n"), "wrong", None),
    ("average", _G_SOLUTION.replace("    total = 0\n", "    total = 0\n    ...\n"), "wrong", "dots-left"),
]
