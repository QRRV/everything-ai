# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-lime: the random forest of rai-pdp-ice on three columns of UCI Adult, explained for one
person of the test half with the course's own LIME, which does what the lime package does without bins: 5000 rows
drawn around the mean of each column, the forest's chance of yes for each, a closeness from each row's distance to the
person, and a straight line fitted with those closenesses as sample weights. The learner writes the closeness, writes
the slope of a weighted line, says what a very wide kernel width does, and says how a model can fool LIME. With the
slider over the kernel width, the three weights of 10 runs; under the exercises, the learner's closeness for three
widths, and the learner's slope for hours_per_week beside the forest's chance for this person."""
import functools
import html
import math
import re

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake

COLUMNS = ["age", "education_num", "hours_per_week"]
PERSON = 5                          # the person explained: position 5 of the test half
ROWS = 5000                         # how many rows LIME makes, as in the lime package
SEEDS = list(range(10))             # the runs of the slider
LIME_WIDTH = 0.75 * math.sqrt(3)    # the lime package's kernel width for three columns: 0.75 times the root of 3
WIDTHS = [0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1, 1.3, 2, 3, 5, 10]   # the stops of the slider; 1.3 stands for LIME_WIDTH


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


@functools.lru_cache(maxsize=1)
def scale() -> tuple:
    """The mean and the standard deviation of each column on the training half. Read-only."""
    X, y, train, test = table()
    mean, std = X[train].mean(axis=0), X[train].std(axis=0)
    for a in (mean, std):
        a.flags.writeable = False
    return mean, std


def person():
    """The row of the person explained: age 41, education_num 13, 40 hours a week."""
    X, y, train, test = table()
    return X[test][PERSON]


def chance(rows) -> list:
    """The forest's chance of yes for each row."""
    return forest().predict_proba(rows)[:, 1]


@functools.lru_cache(maxsize=16)
def made_up(seed: int) -> tuple:
    """LIME's rows of one run, in standard deviations from the mean of each column (z), and the forest's chance of yes
    for each of them, as rows = mean + z * std. Read-only."""
    import numpy as np
    mean, std = scale()
    z = np.random.default_rng(seed).normal(size=(ROWS, 3))
    said = chance(mean + z * std)
    for a in (z, said):
        a.flags.writeable = False
    return z, said


def distances(seed: int):
    """Each row's distance to the person, in standard deviations."""
    import numpy as np
    mean, std = scale()
    z, said = made_up(seed)
    return np.linalg.norm(z - (person() - mean) / std, axis=1)


def closeness_of(d, width: float):
    """How much a row at distance d counts: exp(-d² / (2 · width²)), as in the lime package."""
    import numpy as np
    return np.exp(-d ** 2 / (2 * width ** 2))


def width_of(v: float) -> float:
    """The kernel width a stop of the slider stands for: 1.3 is the lime package's own, 0.75 times the root of 3."""
    return LIME_WIDTH if v == 1.3 else float(v)


def lime(seed: int, width: float, model=None) -> tuple:
    """One run of the course's LIME: the three weights of the line, and what the closenesses add up to."""
    from sklearn.linear_model import LinearRegression
    z, said = made_up(seed)
    close = closeness_of(distances(seed), width)
    line = (model or LinearRegression()).fit(z, said, sample_weight=close)
    return line.coef_, float(close.sum())


@functools.lru_cache(maxsize=32)
def runs(v: float) -> tuple:
    """The weights of the 10 runs at a stop of the slider (10 by 3), and the average of what the closenesses add up
    to. Read-only."""
    import numpy as np
    out = [lime(s, width_of(v)) for s in SEEDS]
    weights = np.array([w for w, total in out])
    weights.flags.writeable = False
    return weights, float(np.mean([total for w, total in out]))


def orders(weights) -> list:
    """For each run, the columns from the biggest weight to the smallest."""
    out = []
    for w in weights:
        out.append(tuple(sorted(range(3), key=lambda j: -w[j])))
    return out


def outside(seed: int = 0) -> int:
    """How many of LIME's rows of this run lie outside what the table holds: younger than 17, education_num below 1
    or above 16, or fewer than 1 hour a week."""
    mean, std = scale()
    z, said = made_up(seed)
    rows = mean + z * std
    out = (rows[:, 0] < 17) | (rows[:, 1] < 1) | (rows[:, 1] > 16) | (rows[:, 2] < 1)
    return int(out.sum())


def _show(x: float, lang: str, digits: int = 3) -> str:
    """A number with this many decimals and a real minus sign; a decimal comma in Dutch."""
    text = f"{x:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    text = text.replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _width(v: float, lang: str) -> str:
    text = f"{v:g}"
    return text.replace(".", ",") if lang == "nl" else text


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"

# --- the slider of *The idea*: the weights of 10 runs at each kernel width -------------------------------------------

_WIDTH_WORDS = {
    "en": {"title": "LIME's weights for this person, 10 runs", "y": "weight per standard deviation",
           "alt": "For each of the three columns, 10 dots: the weight LIME gives it in each of 10 runs with kernel width "
                  "{w}, with a short line at their average. Age {a}, education_num {e}, hours_per_week {h} on average.",
           "said": "Width {w}: together the 5000 rows count on average as much as {total} rows at the person. On "
                   "average age weighs "
                   "<span class=\"c-param ea-num\">{a}</span>, education_num <span class=\"c-param ea-num\">{e}</span> "
                   "and hours_per_week <span class=\"c-param ea-num\">{h}</span>; {top} weighs most in {k} of the 10 "
                   "runs, and {same} of the 10 runs put the three in the same order.",
           "default": " (lime's default)"},
    "nl": {"title": "De weights van LIME voor deze persoon, 10 runs", "y": "weight per standaardafwijking",
           "alt": "Voor elk van de drie kolommen 10 punten: de weight die LIME hem geeft in elk van 10 runs met kernel "
                  "width {w}, met een kort streepje bij hun gemiddelde. Gemiddeld age {a}, education_num {e}, "
                  "hours_per_week {h}.",
           "said": "Width {w}: samen tellen de 5000 rijen gemiddeld even zwaar als {total} rijen op de plek van de "
                   "persoon. Gemiddeld weegt age "
                   "<span class=\"c-param ea-num\">{a}</span>, education_num <span class=\"c-param ea-num\">{e}</span> "
                   "en hours_per_week <span class=\"c-param ea-num\">{h}</span>; {top} weegt het zwaarst in {k} van de "
                   "10 runs, en {same} van de 10 runs zetten de drie in dezelfde volgorde.",
           "default": " (de standaard van lime)"},
}


@functools.lru_cache(maxsize=1)
def _span() -> tuple:
    """The lowest and highest weight over all stops of the slider, so that every stop is drawn on the same scale."""
    low, high = 0.0, 0.0
    for v in WIDTHS:
        weights, total = runs(v)
        low, high = min(low, float(weights.min())), max(high, float(weights.max()))
    return low, high


def most(values: list) -> tuple:
    """The most common value of a list, and how often it comes; on a tie, the one that comes first."""
    best, count = values[0], 0
    for v in values:
        n = values.count(v)
        if n > count:
            best, count = v, n
    return best, count


def width_picture(v: float, lang: str) -> tuple[str, str]:
    """For the slider of *The idea*: the three weights of each of the 10 runs at this kernel width, and the sentence
    with their averages, which column weighs most how often, and how often the order is the same."""
    import numpy as np
    w = _WIDTH_WORDS[lang]
    weights, total = runs(v)
    low, high = _span()
    pad = (high - low) * 0.06
    p = chart.Panel(x=(0.4, 3.6), y=(low - pad, high + pad), title=w["title"], xlabel="", ylabel=w["y"],
                    xticks=[1, 2, 3], xtick_labels=COLUMNS)
    p.line([0.4, 3.6], [0, 0], role="ink-3", width=1.2)
    offsets = np.linspace(-0.27, 0.27, len(SEEDS))
    for j in range(3):
        p.points([j + 1 + o for o in offsets], weights[:, j].tolist(), role="param", r=3.4)
        m = float(weights[:, j].mean())
        p.line([j + 1 - 0.36, j + 1 + 0.36], [m, m], role="param", width=3)
    top, k = most([int(np.argmax(r)) for r in weights])
    order, same = most(orders(weights))
    mean = weights.mean(axis=0)
    fill = {"w": _width(v, lang) + (w["default"] if v == 1.3 else ""), "total": _show(total, lang, 1),
            "a": _show(mean[0], lang), "e": _show(mean[1], lang), "h": _show(mean[2], lang), "top": COLUMNS[top],
            "k": k, "same": same}
    svg = chart.figure([p], alt=w["alt"].format(**fill), width=460, height=280)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    return svg, w["said"].format(**fill)


# --- exercise 1: closeness -------------------------------------------------------------------------------------------

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


GRID = [i / 20 for i in range(81)]   # distances from 0 to 4 for the curves of the picture
PICTURE_WIDTHS = [0.3, 1.3, 3]       # the widths the picture draws; 1.3 stands for LIME_WIDTH

_CLOSE_WORDS = {
    "en": {"title": "Your closeness(), for three kernel widths", "x": "distance to the person, in standard deviations",
           "y": "your closeness()", "label": "width {w}",
           "alt": "Three curves: what your closeness() gives for distances from 0 to 4, with kernel width 0.3, 1.3 "
                  "(solid) and 3. {key}",
           "key": "Your closeness() gives {a} at distance 1 with width 1.3, and {b} with width 0.3. For the 5000 rows "
                  "of the worked example it adds up to {s13} with width 1.3, and to {s03} with width 0.3."},
    "nl": {"title": "Jouw closeness(), voor drie kernel widths", "x": "afstand tot de persoon, in standaardafwijkingen",
           "y": "jouw closeness()", "label": "width {w}",
           "alt": "Drie krommen: wat je closeness() geeft voor afstanden van 0 tot 4, met kernel width 0,3, 1,3 "
                  "(doorgetrokken) en 3. {key}",
           "key": "Je closeness() geeft {a} op afstand 1 met width 1,3, en {b} met width 0,3. Voor de 5000 rijen uit "
                  "het uitgewerkte voorbeeld telt hij op tot {s13} met width 1,3, en tot {s03} met width 0,3."},
}


@functools.lru_cache(maxsize=1)
def _distance_list() -> tuple:
    """The distances of the 5000 rows of the worked example's first run, as a tuple of floats."""
    return tuple(float(d) for d in distances(0).tolist())


def _close_picture(seen) -> str:
    fn = _mine(seen)
    if fn is None:
        return ""
    real = list(_distance_list())

    def run():
        curves = []
        for v in PICTURE_WIDTHS:
            got = _numbers(fn(list(GRID), width_of(v)), len(GRID))
            if got is None or any(not 0 <= c <= 1 for c in got):
                return None
            curves.append(got)
        sums = []
        for v in (1.3, 0.3):
            got = _numbers(fn(list(real), width_of(v)), len(real))
            if got is None or any(not 0 <= c <= 1 for c in got):
                return None
            sums.append(sum(got))
        return curves, sums

    from everything_ai import check as checking
    try:
        got = checking.with_time_limit(run)
    except Exception:
        return ""
    if got is None:
        return ""
    curves, sums = got
    lang = seen.lang
    w = _CLOSE_WORDS[lang]
    fill = {"a": _show(curves[1][20], lang), "b": _show(curves[0][20], lang), "s13": _show(sums[0], lang, 1),
            "s03": _show(sums[1], lang, 1)}
    key = w["key"].format(**fill)
    p = chart.Panel(x=(0, 4), y=(0, 1.08), title="", xlabel=w["x"], ylabel=w["y"], xticks=[0, 1, 2, 3, 4])
    for v, ys in zip(PICTURE_WIDTHS, curves):
        p.line(GRID, ys, role="hyper", width=2.6 if v == 1.3 else 1.6, dashed=v != 1.3)
    for v, ys, at in zip(PICTURE_WIDTHS, curves, (6, 32, 62)):
        p.text(GRID[at] + 0.08, min(ys[at] + 0.07, 1.03), w["label"].format(w=_width(v, lang)), role="ink-2",
               anchor="start", halo=True)
    svg = chart.figure([p], alt=w["alt"].format(key=key), width=520, height=260)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(key)}</div>')


# --- exercise 2: the slope of a weighted line, for hours_per_week ----------------------------------------------------

HOURS = list(range(1, 100))   # the hours of the forest's line for this person in the picture


@functools.lru_cache(maxsize=1)
def hours_rows() -> tuple:
    """The rows of the picture of exercise 2: 5000 values of hours_per_week drawn around its mean (seed 0), with age
    41 and education_num 13 kept; the forest's chance for each; and each row's distance to the person in standard
    deviations of hours. As tuples of floats."""
    import numpy as np
    mean, std = scale()
    hours = mean[2] + np.random.default_rng(0).normal(size=ROWS) * std[2]
    rows = np.column_stack([np.full(ROWS, person()[0]), np.full(ROWS, person()[1]), hours])
    d = np.abs(hours - person()[2]) / std[2]
    return tuple(hours.tolist()), tuple(chance(rows).tolist()), tuple(d.tolist())


@functools.lru_cache(maxsize=1)
def hours_line() -> tuple:
    """The forest's chance of yes for this person at each value in HOURS, age and education kept."""
    import numpy as np
    rows = np.column_stack([np.full(len(HOURS), person()[0]), np.full(len(HOURS), person()[1]), HOURS])
    return tuple(chance(rows).tolist())


def slope_of(xs, ys, close) -> float:
    """The slope of the straight line through (xs, ys), each point counted by its closeness."""
    import numpy as np
    xs, ys, close = np.asarray(xs), np.asarray(ys), np.asarray(close)
    mx, my = (close * xs).sum() / close.sum(), (close * ys).sum() / close.sum()
    return float((close * (xs - mx) * (ys - my)).sum() / (close * (xs - mx) ** 2).sum())


_SLOPE_WORDS = {
    "en": {"title": "Your local_slope(), for hours_per_week near this person", "x": "hours_per_week",
           "y": "the forest's chance of yes",
           "alt": "A curve: the forest's chance of yes for this person, aged 41 with education_num 13, at each value of "
                  "hours_per_week, with a dot at 40 hours. Through the dot, three lines with your slopes: solid for "
                  "kernel width 0.3, dashed for 1.3, grey and dashed for every row counting fully. {key}",
           "legend": "solid: width 0.3; dashed: width 1.3; grey: every row counts fully",
           "key": "Your local_slope() gives {a} per hour with width 0.3, {b} with lime's width 1.3, and {c} when every "
                  "row counts fully."},
    "nl": {"title": "Jouw local_slope(), voor hours_per_week dicht bij deze persoon", "x": "hours_per_week",
           "y": "kans op ja van het forest",
           "alt": "Een kromme: de kans op ja die het forest deze persoon geeft, 41 jaar met education_num 13, bij elke "
                  "waarde van hours_per_week, met een punt bij 40 uur. Door het punt drie lijnen met jouw hellingen: "
                  "doorgetrokken voor kernel width 0,3, gestreept voor 1,3, grijs en gestreept als elke rij volledig "
                  "meetelt. {key}",
           "legend": "doorgetrokken: width 0,3; gestreept: width 1,3; grijs: elke rij telt volledig mee",
           "key": "Je local_slope() geeft {a} per uur met width 0,3, {b} met de width 1,3 van lime, en {c} als elke rij "
                  "volledig meetelt."},
}


def _slope_picture(seen) -> str:
    fn = _mine(seen)
    if fn is None:
        return ""
    hours, said, d = hours_rows()

    def run():
        out = []
        for v in (0.3, 1.3, None):
            close = [1.0] * len(d) if v is None else closeness_of_list(d, width_of(v))
            s = _number(fn(list(hours), list(said), close))
            if s is None:
                return None
            out.append(s)
        return out

    from everything_ai import check as checking
    try:
        got = checking.with_time_limit(run)
    except Exception:
        return ""
    if got is None:
        return ""
    lang = seen.lang
    w = _SLOPE_WORDS[lang]
    fill = {"a": _show(got[0], lang, 4), "b": _show(got[1], lang, 4), "c": _show(got[2], lang, 4)}
    key = w["key"].format(**fill)
    x0, y0 = float(person()[2]), float(chance([person()])[0])
    p = chart.Panel(x=(1, 99), y=(0, 1), title="", xlabel=w["x"], ylabel=w["y"], xticks=[1, 20, 40, 60, 80, 99])
    p.line(HOURS, hours_line(), role="pred", width=2)
    for s, role, width, dashed in zip(got, ("param", "param", "ink-3"), (2.4, 2, 1.6), (False, True, True)):
        p.line([10, 70], [y0 + s * (10 - x0), y0 + s * (70 - x0)], role=role, width=width, dashed=dashed)
    p.points([x0], [y0], role="pred", r=5)
    svg = chart.figure([p], alt=w["alt"].format(key=key), width=520, height=270)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    words = html.escape(w["legend"]) + "<br>" + html.escape(key)
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{words}</div>')


def closeness_of_list(d, width: float) -> list:
    """closeness_of for a list of distances, as a list of floats."""
    out = []
    for x in d:
        out.append(math.exp(-x ** 2 / (2 * width ** 2)))
    return out


# The lesson is done when these are correct. "wide" and "fooled" are choice questions.
REQUIRED = ["closeness", "slope", "wide", "fooled"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

# --- exercise 1
_C_HEAD = ("import math\n"
           "\n"
           "\n"
           "def closeness(distances, width):\n"
           "    # distances: a list of distances to the person, in standard deviations; width: the kernel width\n"
           "    # for each distance d: exp(-d² / (2 · width²)), 1 at the person and almost 0 far away\n")
_C_START = _C_HEAD + "    ...\n"


def _c(formula: str = "math.exp(-d ** 2 / (2 * width ** 2))", ret: str = "    return out\n") -> str:
    """The body of closeness with one part changed: the formula, or the return."""
    return _C_HEAD + "    out = []\n    for d in distances:\n        out.append(" + formula + ")\n" + ret


_C_SOLUTION = _c()
_C_NO_SQUARE = _c("math.exp(-d / (2 * width ** 2))")
_C_NO_TWO = _c("math.exp(-d ** 2 / width ** 2)")
_C_WIDTH_ONCE = _c("math.exp(-d ** 2 / (2 * width))")
_C_PLUS = _c("math.exp(d ** 2 / (2 * width ** 2))")
_C_FIRST = _c(ret="        return out\n")
_C_VISIBLE = [([0, 1, 2], 1)]
_C_HIDDEN = [
    ([0.5, 3], 0.3),
    ([], 1.3),
    ([1, 1, 2.5], 1.3),
    ([0, 0.2, 4], 2),
    ([10], 10),
]

# --- exercise 2
_S_HEAD = ("def local_slope(xs, ys, closeness):\n"
           "    # xs: one column of the rows; ys: the model's chance for each row; closeness: how much each row counts\n"
           "    # the slope of the straight line through the rows, each row counted by its closeness\n"
           "    total = sum(closeness)\n"
           "    mean_x = 0\n"
           "    mean_y = 0\n"
           "    for i in range(len(xs)):\n"
           "        mean_x = mean_x + closeness[i] * xs[i] / total\n"
           "        mean_y = mean_y + closeness[i] * ys[i] / total\n")
_S_START = _S_HEAD + "    ...\n"


def _s(up: str = "closeness[i] * (xs[i] - mean_x) * (ys[i] - mean_y)",
       down: str = "closeness[i] * (xs[i] - mean_x) ** 2", ret: str = "    return up / down\n") -> str:
    """The rest of local_slope with one part changed: what up and down add for each row, or the return."""
    return (_S_HEAD + "    up = 0\n    down = 0\n    for i in range(len(xs)):\n"
            f"        up = up + {up}\n        down = down + {down}\n" + ret)


_S_SOLUTION = _s()
_S_UNWEIGHTED = _s(up="(xs[i] - mean_x) * (ys[i] - mean_y)", down="(xs[i] - mean_x) ** 2")
_S_FLIPPED = _s(ret="    return down / up\n")
_S_NO_MEAN = _s(up="closeness[i] * xs[i] * ys[i]", down="closeness[i] * xs[i] ** 2")
_S_DOWN_Y = _s(down="closeness[i] * (ys[i] - mean_y) ** 2")
_S_FIRST = _s(ret="        return up / down\n")
_S_VISIBLE = [([1, 2, 3, 4], [1, 3, 2, 4], [1, 1, 1, 0])]
_S_HIDDEN = [
    ([0, 1, 2], [1, 3, 5], [1, 2, 3]),
    ([10, 20, 30, 40], [0.2, 0.3, 0.3, 0.6], [0.1, 1, 1, 0.1]),
    ([1, 2], [5, 3], [1, 1]),
    ([2, 4, 6, 8, 10], [0.5, 0.4, 0.45, 0.3, 0.1], [0.2, 0.9, 1, 0.9, 0.2]),
]

EXERCISES = {
    "closeness": FunctionBox(
        name="closeness",
        picture=_close_picture,
        function="closeness",
        start=_C_START,
        solution=_C_SOLUTION,
        visible=_C_VISIBLE,
        hidden=_C_HIDDEN,
        digits=3,
        first=[_DOTS],
        mistakes=[
            Mistake("no-square", wrong=_C_NO_SQUARE,
                    en="`{call}` gives {got}, and it should give {want}. Your closeness uses the distance itself where the formula has its square, `d ** 2`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je `closeness` gebruikt de afstand zelf waar de formule zijn kwadraat heeft, `d ** 2`."),
            Mistake("no-two", wrong=_C_NO_TWO,
                    en="`{call}` gives {got}, and it should give {want}. That is $\\exp(-d^2 / w^2)$, the paper's formula without the 2. This lesson follows the lime package, which divides by `2 * width ** 2`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Dat is $\\exp(-d^2 / w^2)$, de formule uit het paper zonder de 2. Deze les volgt het lime-package, dat deelt door `2 * width ** 2`."),
            Mistake("width-once", wrong=_C_WIDTH_ONCE,
                    en="`{call}` gives {got}, and it should give {want}. Your closeness divides by 2 times the width, not by 2 times its square: `2 * width ** 2`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je `closeness` deelt door 2 keer de width, niet door 2 keer zijn kwadraat: `2 * width ** 2`."),
            Mistake("plus", wrong=_C_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: your closeness grows with the distance, above 1. The minus sign is missing: `-d ** 2`, which Python reads as minus the square.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `closeness` groeit met de afstand, boven 1. Het minteken ontbreekt: `-d ** 2`, wat Python leest als min het kwadraat."),
            Mistake("first", wrong=_C_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your list stops after the first distance. If your `return` stands inside the `for` loop, Python leaves the function there. Put it after the loop, as far in as `out = []`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je lijst stopt na de eerste afstand. Als je `return` binnen de `for`-loop staat, verlaat Python de functie daar. Zet hem na de loop, even ver ingesprongen als `out = []`."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...` with the body of `closeness`: start an empty list, and for each distance `d` append `math.exp(-d ** 2 / (2 * width ** 2))`. Return the list.",
                      "nl": "Vervang de `...` door de inhoud van `closeness`: begin een lege lijst, en voeg voor elke afstand `d` de waarde van `math.exp(-d ** 2 / (2 * width ** 2))` toe. Geef de lijst terug."},
            "cleared": _CLEARED,
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Work out the first two by hand: at distance 0 every row counts 1, and at distance 1 with width 1 it is $e^{{-1/2}}$.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Reken de eerste twee met de hand uit: op afstand 0 telt elke rij 1, en op afstand 1 met width 1 is het $e^{{-1/2}}$."},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use `width` as it is given, and work for a list of any length, an empty one too?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `width` zoals hij wordt meegegeven, en werkt hij voor een lijst van elke lengte, ook een lege?"},
            "correct": {"en": "Your `closeness` is right in all {n} cases. With lime's width 1.3, a row 1 standard deviation from the person counts 0.744 and a row 2 away 0.306; with width 0.3, a row 1 away counts 0.004. The 5000 rows of the worked example add up to 1965.1 with width 1.3, and to 72.1 with width 0.3: then they count as much as 72 rows at the person.",
                        "nl": "Je `closeness` klopt in alle {n} gevallen. Met de width 1,3 van lime telt een rij op 1 standaardafwijking van de persoon 0,744 en een rij op 2 standaardafwijkingen 0,306; met width 0,3 telt een rij op 1 standaardafwijking 0,004. De 5000 rijen uit het uitgewerkte voorbeeld tellen op tot 1965,1 met width 1,3, en tot 72,1 met width 0,3: dan tellen ze even zwaar als 72 rijen op de plek van de persoon."},
        },
    ),
    "slope": FunctionBox(
        name="slope",
        picture=_slope_picture,
        function="local_slope",
        start=_S_START,
        solution=_S_SOLUTION,
        visible=_S_VISIBLE,
        hidden=_S_HIDDEN,
        digits=3,
        first=[_DOTS],
        mistakes=[
            Mistake("unweighted", wrong=_S_UNWEIGHTED,
                    en="`{call}` gives {got}, and it should give {want}. Your slope comes out as if every row counted fully in `up` and `down`, so far rows count as much as near ones. Multiply each term by `closeness[i]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je helling komt uit alsof elke rij volledig meetelt in `up` en `down`, dus verre rijen tellen even zwaar als rijen dichtbij. Vermenigvuldig elke term met `closeness[i]`."),
            Mistake("flipped", wrong=_S_FLIPPED,
                    en="`{call}` gives {got}, and it should give {want}: 1 divided by the slope. Divide the sum over both deviations by the sum over the squared deviations of `xs`: `up / down`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: 1 gedeeld door de helling. Deel de som over beide afwijkingen door de som over de gekwadrateerde afwijkingen van `xs`: `up / down`."),
            Mistake("no-mean", wrong=_S_NO_MEAN,
                    en="`{call}` gives {got}, and it should give {want}: the slope of a line forced through 0. Subtract the means first: `xs[i] - mean_x` and `ys[i] - mean_y`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de helling van een lijn die door 0 moet. Trek eerst de gemiddelden af: `xs[i] - mean_x` en `ys[i] - mean_y`."),
            Mistake("down-y", wrong=_S_DOWN_Y,
                    en="`{call}` gives {got}, and it should give {want}. You divide by the spread of `ys`; the slope divides by the spread of `xs`: `(xs[i] - mean_x) ** 2`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je deelt door de spreiding van `ys`; de helling deelt door de spreiding van `xs`: `(xs[i] - mean_x) ** 2`."),
            Mistake("first", wrong=_S_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your function stops after the first row. If your `return` stands inside the `for` loop, Python leaves the function there. Put it after the loop, as far in as `up = 0`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie stopt na de eerste rij. Als je `return` binnen de `for`-loop staat, verlaat Python de functie daar. Zet hem na de loop, even ver ingesprongen als `up = 0`."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...`: sum in `up` each row's closeness times its deviation in `xs` times its deviation in `ys`, and in `down` its closeness times its squared deviation in `xs`. Return `up / down`.",
                      "nl": "Vervang de `...`: tel in `up` voor elke rij zijn nabijheid keer zijn afwijking in `xs` keer zijn afwijking in `ys` op, en in `down` zijn nabijheid keer zijn gekwadrateerde afwijking in `xs`. Geef `up / down` terug."},
            "cleared": _CLEARED,
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. In the example the last row counts 0, so the line goes through the first three rows only: their means are 2 and 2. Work out `up` and `down` for them by hand.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. In het voorbeeld telt de laatste rij 0, dus de lijn gaat alleen door de eerste drie rijen: hun gemiddelden zijn 2 en 2. Reken `up` en `down` voor hen met de hand uit."},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use the closenesses as they are given, also when they are not 0 or 1?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de waarden van `closeness` zoals ze worden meegegeven, ook als ze niet 0 of 1 zijn?"},
            "correct": {"en": "Right in all {n} cases. The picture runs your `local_slope` on 5000 rows for hours_per_week, with this person's age and education kept: with width 0.3 the slope is 0.0348 per hour, because the forest's chance jumps from 0.394 at 39 hours to 0.610 at 42; with lime's width 1.3 it is 0.0138; and with every row counting fully, 0.0109. The same person, the same forest: how near \"near\" is decides the answer.",
                        "nl": "Goed in alle {n} gevallen. De tekening voert je `local_slope` uit op 5000 rijen voor hours_per_week, met de leeftijd en de opleiding van deze persoon vast: met width 0,3 is de helling 0,0348 per uur, omdat de kans van het forest springt van 0,394 bij 39 uur naar 0,610 bij 42 uur; met de width 1,3 van lime is hij 0,0138; en als elke rij volledig meetelt 0,0109. Dezelfde persoon, hetzelfde forest: hoe dichtbij \"dichtbij\" is, bepaalt het antwoord."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("closeness", _C_START, "empty", None),
    ("closeness", _C_SOLUTION, "correct", None),
    ("closeness", _C_HEAD + "    out = []\n    for i in range(len(distances)):\n"
                  "        out.append(math.e ** (-(distances[i] ** 2) / (2 * width * width)))\n    return out\n",
     "correct", None),
    ("closeness", _C_NO_SQUARE, "wrong", "no-square"),
    ("closeness", _C_NO_TWO, "wrong", "no-two"),
    ("closeness", _C_WIDTH_ONCE, "wrong", "width-once"),
    ("closeness", _C_PLUS, "wrong", "plus"),
    ("closeness", _C_FIRST, "wrong", "first"),
    ("closeness", _C_HEAD + "    return [1.0, 0.6065306597126334, 0.1353352832366127]\n", "almost", None),
    ("closeness", _c(ret="    return\n"), "wrong", "no-return"),
    ("closeness", _C_HEAD + "    ...\n    return []\n", "wrong", "dots-left"),
    ("slope", _S_START, "empty", None),
    ("slope", _S_SOLUTION, "correct", None),
    ("slope", _S_HEAD + "    up = 0\n    down = 0\n    for i in range(len(xs)):\n        dx = xs[i] - mean_x\n"
              "        up += closeness[i] * dx * (ys[i] - mean_y)\n        down += closeness[i] * dx * dx\n"
              "    return up / down\n", "correct", None),
    ("slope", _S_UNWEIGHTED, "wrong", "unweighted"),
    ("slope", _S_FLIPPED, "wrong", "flipped"),
    ("slope", _S_NO_MEAN, "wrong", "no-mean"),
    ("slope", _S_DOWN_Y, "wrong", "down-y"),
    ("slope", _S_FIRST, "wrong", "first"),
    ("slope", _S_HEAD + "    return 0.5\n", "almost", None),
    ("slope", _s(ret="    return\n"), "wrong", "no-return"),
    ("slope", _S_HEAD + "    ...\n    return 0.5\n", "wrong", "dots-left"),
]
