# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-shapley-values: the random forest of the last lessons, now on four columns of UCI Adult,
explained for one person of the test half with exact Shapley values. A coalition is the set of columns that are known;
its value is the forest's average chance of yes over 200 background rows of the training half, with the known columns
taken from the person (the interventional kind, as Molnar fills in absent features). The learner writes that value and
the Shapley formula with its weights, says what a Shapley value does not say, and says what a different background does.
With the slider over the background, the four values as a waterfall from the background's average to the person's
chance; under the exercises, the learner's 16 coalition values, and the learner's four values as the same waterfall."""
import functools
import html
import itertools
import math
import re

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake

COLUMNS = ["age", "education_num", "hours_per_week", "capital_gain"]
PERSON = 5                       # the person explained: position 5 of the test half
BACKGROUND = 200                 # how many rows of the training half fill in the columns that are not known
TOPS = [8, 9, 10, 12, 13, 16]    # the stops of the slider: the highest education_num in the background; 16 is everyone


@functools.lru_cache(maxsize=1)
def table():
    """The four columns of UCI Adult as floats, who earns more than 50,000 dollars, and the positions of the training
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
    About a second."""
    from sklearn.ensemble import RandomForestClassifier
    X, y, train, test = table()
    return RandomForestClassifier(min_samples_leaf=20, random_state=0).fit(X[train], y[train])


def person():
    """The row of the person explained: age 41, education_num 13, 40 hours a week, no capital gain."""
    X, y, train, test = table()
    return X[test][PERSON]


def chance(rows):
    """The forest's chance of yes for each row, as a NumPy array."""
    return forest().predict_proba(rows)[:, 1]


@functools.lru_cache(maxsize=None)
def background(top: int = 16):
    """200 people of the training half with education_num at most `top`, drawn with seed 0. With 16, the background
    of the worked example: everyone may be drawn. Read-only."""
    import numpy as np
    X, y, train, test = table()
    rows = X[train][X[train][:, 1] <= top]
    out = rows[np.random.default_rng(0).choice(len(rows), BACKGROUND, replace=False)]
    out.flags.writeable = False
    return out


def coalitions() -> list:
    """The 16 coalitions of four columns, as tuples of 0s and 1s, in the order of itertools.product."""
    return list(itertools.product([0, 1], repeat=len(COLUMNS)))


@functools.lru_cache(maxsize=None)
def values(top: int = 16) -> dict:
    """The value of each coalition: the forest's average chance of yes over the background, with the known columns
    taken from the person. A fresh dict of floats each time it is read is not needed: callers do not change it."""
    import numpy as np
    bg, me = background(top), person()
    out = {}
    for known in coalitions():
        rows = np.where(np.array(known) == 1, me, bg)
        out[known] = float(chance(rows).mean())
    return out


def weight(s: int, n: int) -> float:
    """How much a coalition of s columns counts in the Shapley value of a column outside it, with n columns."""
    return math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)


def shapley_of(v: dict, i: int) -> float:
    """The Shapley value of column i in the game v, by the formula: over the coalitions without i."""
    total = 0.0
    for known in v:
        if known[i] == 0:
            with_i = list(known)
            with_i[i] = 1
            total += weight(sum(known), len(known)) * (v[tuple(with_i)] - v[known])
    return total


def shapley_by_orders(v: dict) -> list:
    """The Shapley values of all columns in the game v, as the worked example works them out: what each column adds
    when it joins, averaged over every order in which the columns can join."""
    n = len(next(iter(v)))
    orders = list(itertools.permutations(range(n)))
    out = [0.0] * n
    for order in orders:
        known = [0] * n
        for i in order:
            before = v[tuple(known)]
            known[i] = 1
            out[i] += (v[tuple(known)] - before) / len(orders)
    return out


@functools.lru_cache(maxsize=None)
def shares(top: int = 16) -> tuple:
    """The four Shapley values of the person for the background with education_num at most `top`."""
    v = values(top)
    return tuple(shapley_of(v, i) for i in range(len(COLUMNS)))


def _show(x: float, lang: str, digits: int = 3, sign: bool = False) -> str:
    """A number with this many decimals and a real minus sign, with a plus sign if asked; a decimal comma in Dutch."""
    text = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-+")
    text = text.replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"

# --- the waterfall: from the background's average, one column at a time, to the person's chance ---------------------

_FALL_WORDS = {
    "en": {"x": "the forest's chance of yes", "start": "average {v}", "end": "this person {v}"},
    "nl": {"x": "kans op ja van het forest", "start": "gemiddelde {v}", "end": "deze persoon {v}"},
}


def waterfall(parts, start: float, end: float, lang: str, span: tuple | None = None) -> chart.Panel:
    """A panel with one bar per column, top to bottom, each with its name and part above it: each bar begins where
    the one above it ends, the first at `start`, and is as long as that column's part. A dashed line at `start` and
    one at `end`. With `span`, the range of the x-axis; otherwise it fits the bars and both lines."""
    w = _FALL_WORDS[lang]
    n = len(parts)
    ends, at = [], start
    for v in parts:
        ends.append((at, at + v))
        at += v
    if span is None:
        lo = min([0.0, start, end] + [min(a, b) for a, b in ends])
        hi = max([start, end] + [max(a, b) for a, b in ends])
        pad = (hi - lo) * 0.05 + 1e-9
        span = (lo - pad, hi + pad)
    p = chart.Panel(x=span, y=(-0.95, n + 0.5), title="", xlabel=w["x"], ylabel="", yticks=[])

    def anchor(x):   # a label near an edge of the chart runs inward, so that it is not cut off
        at = (x - span[0]) / (span[1] - span[0])
        return "start" if at < 0.2 else "end" if at > 0.8 else "middle"

    p.line([start, start], [-0.35, n - 0.1], role="ink-3", width=1.4, dashed=True)
    p.line([end, end], [-0.35, n - 0.45], role="pred", width=1.6, dashed=True)
    p.text(start, n + 0.2, w["start"].format(v=_show(start, lang)), role="ink-2", anchor=anchor(start), halo=True)
    p.text(end, -0.62, w["end"].format(v=_show(end, lang)), role="pred", anchor=anchor(end), halo=True)
    for j, (v, (a, b)) in enumerate(zip(parts, ends)):
        y = n - 1 - j
        p.rect(min(a, b), y - 0.2, max(a, b), y + 0.2, role="pred", solid=True)
        if j + 1 < n:
            p.line([b, b], [y - 0.2, y - 0.8], role="ink-3", width=1.2)
        p.text(span[0], y + 0.42, f"{COLUMNS[j]} {_show(v, lang, sign=True)}", role="ink", anchor="start", halo=True)
    return p


# --- the slider of *The idea*: the four values for backgrounds with less education --------------------------------------

_TOP_WORDS = {
    "en": {"title": "Shapley values for this person, by background",
           "alt": "Four bars from the average chance of yes of a background of 200 people with education_num at most "
                  "{k}, {avg}, to this person's chance, {fx}: age {a}, education_num {e}, hours_per_week {h}, "
                  "capital_gain {c}.",
           "said": "Background: 200 people of the training half with education_num at most {k}{all}. Their average "
                   "chance of yes is {avg}; this person's is {fx}. age gets "
                   "<span class=\"c-pred ea-num\">{a}</span>, education_num <span class=\"c-pred ea-num\">{e}</span>, "
                   "hours_per_week <span class=\"c-pred ea-num\">{h}</span> and capital_gain "
                   "<span class=\"c-pred ea-num\">{c}</span>: together {sum}, which is {fx} minus {avg}.",
           "all": ", so anyone may be drawn"},
    "nl": {"title": "Shapley-waarden voor deze persoon, per achtergrond",
           "alt": "Vier balken van de gemiddelde kans op ja van een achtergrond van 200 mensen met education_num "
                  "hoogstens {k}, {avg}, naar de kans van deze persoon, {fx}: age {a}, education_num {e}, "
                  "hours_per_week {h}, capital_gain {c}.",
           "said": "Achtergrond: 200 mensen uit de helft om op te trainen met education_num hoogstens {k}{all}. Hun "
                   "gemiddelde kans op ja is {avg}; die van deze persoon {fx}. age krijgt "
                   "<span class=\"c-pred ea-num\">{a}</span>, education_num <span class=\"c-pred ea-num\">{e}</span>, "
                   "hours_per_week <span class=\"c-pred ea-num\">{h}</span> en capital_gain "
                   "<span class=\"c-pred ea-num\">{c}</span>: samen {sum}, en dat is {fx} min {avg}.",
           "all": ", dus iedereen kan getrokken worden"},
}


@functools.lru_cache(maxsize=1)
def _top_span() -> tuple:
    """One range of the x-axis for every stop of the slider, so that the stops can be compared."""
    lo, hi = 0.0, 0.0
    for k in TOPS:
        v = values(k)
        at = v[(0, 0, 0, 0)]
        hi = max(hi, at)
        for s in shares(k):
            at += s
            lo, hi = min(lo, at), max(hi, at)
    return (lo - 0.02, hi + 0.04)


def background_picture(top: int, lang: str) -> tuple[str, str]:
    """For the slider of *The idea*: the waterfall of the four values for the background with education_num at most
    `top`, and the sentence with the background's average, the four values and what they add up to."""
    w = _TOP_WORDS[lang]
    v, parts = values(top), shares(top)
    start, end = v[(0, 0, 0, 0)], v[(1, 1, 1, 1)]
    p = waterfall(parts, start, end, lang, span=_top_span())
    p.title = w["title"]
    fill = {"k": top, "all": w["all"] if top == 16 else "", "avg": _show(start, lang), "fx": _show(end, lang),
            "a": _show(parts[0], lang, sign=True), "e": _show(parts[1], lang, sign=True),
            "h": _show(parts[2], lang, sign=True), "c": _show(parts[3], lang, sign=True),
            "sum": _show(sum(parts), lang)}
    svg = chart.figure([p], alt=w["alt"].format(**fill), width=460, height=290)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    return svg, w["said"].format(**fill)


# --- helpers for the pictures of the exercises --------------------------------------------------------------------------

class Model:
    """A model the check hands to the learner's function: called on a list of rows, it gives a list with its answer
    for each row. Written in the check's messages by its name, and copied as itself, since it does not change."""

    def __init__(self, fn, name: str):
        self.fn, self.name = fn, name

    def __call__(self, rows):
        return [float(c) for c in self.fn(rows)]

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return self


def _add_up(rows):
    out = []
    for r in rows:
        out.append(sum(r))
    return out


def _product(rows):
    out = []
    for r in rows:
        x = 1
        for c in r:
            x = x * c
        out.append(x)
    return out


def _biggest(rows):
    out = []
    for r in rows:
        out.append(max(r))
    return out


ADD_UP = Model(_add_up, "add_up")        # answers the sum of each row
PRODUCT = Model(_product, "product")     # answers the product of each row
BIGGEST = Model(_biggest, "biggest")     # answers the largest number in each row
FOREST = Model(lambda rows: chance(rows), "forest")


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


# --- exercise 1: the value of a coalition ---------------------------------------------------------------------------

_VALUE_WORDS = {
    "en": {"title": "Your value(), for the 16 coalitions of this person", "x": "how many columns are known",
           "y": "your value()",
           "alt": "16 dots, one for each coalition of the four columns, by how many columns it knows: what your value() "
                  "gives for this person with the 200 people of the background. Thin lines go from each of the 8 "
                  "coalitions without education_num to the same coalition with it. {key}",
           "key": "Your value() gives {v0} with nothing known and {v4} with all four known. Over the 200 people of the "
                  "background the forest's average chance of yes is {avg}, and for this person {fx}. The thin lines "
                  "go from each coalition without education_num to the same one with it: knowing education_num changes "
                  "your value by {lo} to {hi}."},
    "nl": {"title": "Jouw value(), voor de 16 coalities van deze persoon", "x": "hoeveel kolommen bekend zijn",
           "y": "jouw value()",
           "alt": "16 punten, één voor elke coalitie van de vier kolommen, naar hoeveel kolommen ze kent: wat je value() "
                  "geeft voor deze persoon met de 200 mensen van de achtergrond. Dunne lijnen gaan van elk van de 8 "
                  "coalities zonder education_num naar dezelfde coalitie met education_num. {key}",
           "key": "Je value() geeft {v0} als niets bekend is en {v4} als alle vier bekend zijn. Over de 200 mensen van "
                  "de achtergrond is de gemiddelde kans op ja van het forest {avg}, en voor deze persoon {fx}. De "
                  "dunne lijnen gaan van elke coalitie zonder education_num naar dezelfde met education_num: weten wat "
                  "education_num is, verandert je value met {lo} tot {hi}."},
}


@functools.lru_cache(maxsize=1)
def _background_lists() -> tuple:
    """The background of the worked example and the person, as plain lists of floats, for the learner's function."""
    return [r.tolist() for r in background()], person().tolist()


def _value_picture(seen) -> str:
    fn = _mine(seen)
    if fn is None:
        return ""
    import copy
    rows, me = _background_lists()
    forest()

    def run():
        got = {}
        for known in coalitions():
            g = _number(fn(FOREST, list(me), copy.deepcopy(rows), known))
            if g is None:
                return None
            got[known] = g
        return got

    from everything_ai import check as checking
    try:
        got = checking.with_time_limit(run)
    except Exception:
        return ""
    if got is None:
        return ""
    lang = seen.lang
    w = _VALUE_WORDS[lang]
    steps = []
    for known in coalitions():
        if known[1] == 0:
            with_i = list(known)
            with_i[1] = 1
            steps.append((known, tuple(with_i)))
    diffs = [got[b] - got[a] for a, b in steps]
    v = values()
    fill = {"v0": _show(got[(0, 0, 0, 0)], lang), "v4": _show(got[(1, 1, 1, 1)], lang),
            "avg": _show(v[(0, 0, 0, 0)], lang), "fx": _show(v[(1, 1, 1, 1)], lang),
            "lo": _show(min(diffs), lang), "hi": _show(max(diffs), lang)}
    key = w["key"].format(**fill)
    # each coalition at its number of known columns, spread a little so that coalitions of the same size stand apart
    place, seen_size = {}, {}
    sizes = {s: sum(1 for k in coalitions() if sum(k) == s) for s in range(5)}
    for known in coalitions():
        s = sum(known)
        j = seen_size.get(s, 0)
        seen_size[s] = j + 1
        place[known] = s + (0 if sizes[s] == 1 else -0.3 + 0.6 * j / (sizes[s] - 1))
    ys = list(got.values())
    lo, hi = min(ys + [0.0]), max(ys)
    pad = (hi - lo) * 0.08 + 1e-9
    p = chart.Panel(x=(-0.5, 4.5), y=(lo - pad, hi + pad), title="", xlabel=w["x"], ylabel=w["y"],
                    xticks=[0, 1, 2, 3, 4])
    for a, b in steps:
        p.line([place[a], place[b]], [got[a], got[b]], role="ink-3", width=1.2)
    p.points([place[k] for k in coalitions()], [got[k] for k in coalitions()], role="pred", r=4)
    svg = chart.figure([p], alt=w["alt"].format(key=key), width=520, height=270)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(key)}</div>')


# --- exercise 2: the Shapley value of a column ----------------------------------------------------------------------

_SHAPLEY_WORDS = {
    "en": {"title": "Your shapley(), for the four columns of this person",
           "alt": "Four bars from the background's average chance of yes, {avg}, one for each column, each as long as "
                  "what your shapley() gives it, with a dashed line at this person's chance, {fx}. {key}",
           "key": "Your shapley() gives age {a}, education_num {e}, hours_per_week {h} and capital_gain {c}: together "
                  "{sum}. This person's chance minus the background's average is {gap}."},
    "nl": {"title": "Jouw shapley(), voor de vier kolommen van deze persoon",
           "alt": "Vier balken vanaf de gemiddelde kans op ja van de achtergrond, {avg}, één voor elke kolom, elk zo "
                  "lang als wat je shapley() hem geeft, met een gestreepte lijn bij de kans van deze persoon, {fx}. "
                  "{key}",
           "key": "Je shapley() geeft age {a}, education_num {e}, hours_per_week {h} en capital_gain {c}: samen {sum}. "
                  "De kans van deze persoon min het gemiddelde van de achtergrond is {gap}."},
}


def _shapley_picture(seen) -> str:
    fn = _mine(seen)
    if fn is None:
        return ""
    v = values()

    def run():
        out = []
        for i in range(len(COLUMNS)):
            g = _number(fn(dict(v), i))
            if g is None:
                return None
            out.append(g)
        return out

    from everything_ai import check as checking
    try:
        got = checking.with_time_limit(run)
    except Exception:
        return ""
    if got is None:
        return ""
    lang = seen.lang
    w = _SHAPLEY_WORDS[lang]
    start, end = v[(0, 0, 0, 0)], v[(1, 1, 1, 1)]
    fill = {"a": _show(got[0], lang, sign=True), "e": _show(got[1], lang, sign=True),
            "h": _show(got[2], lang, sign=True), "c": _show(got[3], lang, sign=True), "sum": _show(sum(got), lang),
            "gap": _show(end - start, lang), "avg": _show(start, lang), "fx": _show(end, lang)}
    key = w["key"].format(**fill)
    p = waterfall(got, start, end, lang)
    svg = chart.figure([p], alt=w["alt"].format(**fill, key=key), width=520, height=290)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(key)}</div>')


# The lesson is done when these are correct. "meaning" and "background" are choice questions.
REQUIRED = ["value", "shapley", "meaning", "background"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

# --- exercise 1
_V_HEAD = ("def value(model, person, background, known):\n"
           "    # model(rows) gives a list with the model's answer for each row\n"
           "    # known[j] is 1 when column j is known, from person, and 0 when it comes from a background row\n")
_V_START = _V_HEAD + "    ...\n"


def _v(take: str = "            if known[j] == 1:\n                filled.append(person[j])\n"
                   "            else:\n                filled.append(row[j])\n",
       end: str = "    chances = model(rows)\n    return sum(chances) / len(chances)\n") -> str:
    """The body of value with one part changed: how a row is filled in, or the end."""
    return (_V_HEAD + "    rows = []\n    for row in background:\n        filled = []\n"
            "        for j in range(len(person)):\n" + take + "        rows.append(filled)\n" + end)


_V_SOLUTION = _v()
_V_SWAPPED = _v(take="            if known[j] == 1:\n                filled.append(row[j])\n"
                     "            else:\n                filled.append(person[j])\n")
_V_TOTAL = _v(end="    chances = model(rows)\n    return sum(chances)\n")
_V_FIRST = _v(end="        chances = model(rows)\n        return sum(chances) / len(chances)\n")
_V_PERSON = _V_HEAD + "    return model([person])[0]\n"
_V_AVERAGE = _V_HEAD + "    chances = model(background)\n    return sum(chances) / len(chances)\n"
_V_ROWS = [[1, 2, 3], [3, 1, 0]]
_V_VISIBLE = [(ADD_UP, [4, 0, 1], _V_ROWS, (1, 0, 1))]
_V_HIDDEN = [
    (ADD_UP, [4, 0, 1], _V_ROWS, (0, 0, 0)),
    (ADD_UP, [4, 0, 1], _V_ROWS, (1, 1, 1)),
    (PRODUCT, [2, 5], [[1, 1], [3, 0], [0, 4]], (1, 0)),
    (PRODUCT, [2, 5], [[1, 1], [3, 0], [0, 4]], (0, 1)),
    (BIGGEST, [7, 1, 1, 1], [[0, 9, 9, 9]], (1, 0, 0, 0)),
    (BIGGEST, [1, 6, 0, 2], [[5, 0, 1, 1], [0, 2, 8, 0], [1, 1, 1, 9], [4, 4, 0, 0], [2, 7, 3, 1]], (0, 1, 1, 0)),
]

# --- exercise 2
_SH_HEAD = ("import math\n"
            "\n"
            "\n"
            "def weight(s, n):\n"
            "    # how much a coalition of s columns counts in the Shapley value of a column outside it, of n columns\n"
            "    return math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)\n"
            "\n"
            "\n"
            "def shapley(v, i):\n"
            "    # v[known] is the value of a coalition: known is a tuple of 0s and 1s, one for each column\n")
_SH_START = _SH_HEAD + "    ...\n"


def _sh(add: str = "weight(sum(known), len(known)) * (v[tuple(with_i)] - v[known])",
        ret: str = "    return total\n") -> str:
    """The body of shapley with one part changed: what each coalition adds, or the return."""
    return (_SH_HEAD + "    total = 0\n    for known in v:\n        if known[i] == 0:\n"
            "            with_i = list(known)\n            with_i[i] = 1\n"
            f"            total = total + {add}\n" + ret)


_SH_SOLUTION = _sh()
_SH_UNWEIGHTED = _sh(add="v[tuple(with_i)] - v[known]")
_SH_FLIPPED = _sh(add="weight(sum(known), len(known)) * (v[known] - v[tuple(with_i)])")
_SH_MEAN = _sh(add="(v[tuple(with_i)] - v[known]) / (len(v) / 2)")
_SH_FOUR = _sh(add="weight(sum(known), 4) * (v[tuple(with_i)] - v[known])")
_SH_FIRST = _sh(ret="        return total\n")
_SH_TWO = {(0, 0): 0, (1, 0): 2, (0, 1): 4, (1, 1): 10}
_SH_VISIBLE = [(_SH_TWO, 0)]


def _game(n: int, rule) -> dict:
    """A game of n columns: the value of each coalition by `rule`, a function of the tuple of 0s and 1s."""
    out = {}
    for known in itertools.product([0, 1], repeat=n):
        out[known] = rule(known)
    return out


_SH_HIDDEN = [
    (_SH_TWO, 1),
    ({(0, 0): 0.25, (1, 0): 0.25, (0, 1): 0.5, (1, 1): 0.6}, 0),
    (_game(3, lambda k: sum(k) ** 2), 2),
    (_game(3, lambda k: 0.2 + 0.1 * k[0] + 0.3 * k[0] * k[2]), 0),
    (_game(3, lambda k: 0.2 + 0.1 * k[0] + 0.3 * k[0] * k[2]), 1),
    (_game(4, lambda k: k[0] + 2 * k[1] + 3 * k[2] + 4 * k[3] + 5 * k[0] * k[1]), 1),
    (_game(3, lambda k: k[0] * k[1] * k[2]), 0),
    (_game(4, lambda k: 0.1 + 0.2 * k[1] + 0.6 * k[0] * k[1] * k[3]), 3),
]

EXERCISES = {
    "value": FunctionBox(
        name="value",
        picture=_value_picture,
        function="value",
        start=_V_START,
        solution=_V_SOLUTION,
        visible=_V_VISIBLE,
        hidden=_V_HIDDEN,
        digits=3,
        first=[_DOTS],
        mistakes=[
            Mistake("swapped", wrong=_V_SWAPPED,
                    en="`{call}` gives {got}, the value of the coalition of the other columns, and it should give {want}. Where `known[j]` is 1, the column is known and comes from `person`; where it is 0, it comes from the background row.",
                    nl="`{call}` geeft {got}, de waarde van de coalitie van de andere kolommen, en dat moet {want} zijn. Waar `known[j]` 1 is, is de kolom bekend en komt hij uit `person`; waar hij 0 is, komt hij uit de rij van de achtergrond."),
            Mistake("total", wrong=_V_TOTAL,
                    en="`{call}` gives {got}, and it should give {want}: the sum of the model's answers, not their average. Divide by how many there are, `len(chances)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de som van de antwoorden van het model, niet hun gemiddelde. Deel door hoeveel het er zijn, `len(chances)`."),
            Mistake("first", wrong=_V_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your value stops after the first background row. If your `return` stands inside the `for` loop, Python leaves the function there. Put it after the loop, as far in as `rows = []`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `value` stopt na de eerste rij van de achtergrond. Als je `return` binnen de `for`-loop staat, verlaat Python de functie daar. Zet hem na de loop, even ver ingesprongen als `rows = []`."),
            Mistake("person-only", wrong=_V_PERSON,
                    en="`{call}` gives {got}, and it should give {want}: the model's answer for the person, whatever `known` says. Make a row for each background row, with only the known columns from the person.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het antwoord van het model voor de persoon, wat `known` ook zegt. Maak een rij voor elke rij van de achtergrond, met alleen de bekende kolommen uit de persoon."),
            Mistake("background-only", wrong=_V_AVERAGE,
                    en="`{call}` gives {got}, and it should give {want}: the model's average over the background as it is, whatever `known` says. In each background row, put the person's value in the known columns first.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het gemiddelde van het model over de achtergrond zoals die is, wat `known` ook zegt. Zet in elke rij van de achtergrond eerst de waarde van de persoon in de bekende kolommen."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...`: for each row of `background`, make a new row with `person[j]` where `known[j]` is 1 and `row[j]` where it is 0. Ask `model` about all of them, and return the average.",
                      "nl": "Vervang de `...`: maak voor elke rij van `background` een nieuwe rij met `person[j]` waar `known[j]` 1 is en `row[j]` waar hij 0 is. Vraag `model` naar allemaal, en geef het gemiddelde terug."},
            "cleared": _CLEARED,
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Work the example out by hand: with the person's first and last column, the two rows become [4, 2, 1] and [4, 1, 1], and `add_up` answers 7 and 6.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Reken het voorbeeld met de hand uit: met de eerste en de laatste kolom van de persoon worden de twee rijen [4, 2, 1] en [4, 1, 1], en `add_up` antwoordt 7 en 6."},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for every `known`, every number of columns and every background, also with one row or with five?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke `known`, elk aantal kolommen en elke achtergrond, ook met één rij of met vijf?"},
            "correct": {"en": "Your `value` is right in all {n} cases. The picture runs it with the forest for the person of the worked example: with nothing known it gives 0.225, the forest's average over the 200 people of the background, and with all four known 0.451, the person's own chance. Knowing education_num adds between 0.139 and 0.226, depending on what else is known.",
                        "nl": "Je `value` klopt in alle {n} gevallen. De tekening voert hem uit met het forest voor de persoon uit het uitgewerkte voorbeeld: als niets bekend is geeft hij 0,225, het gemiddelde van het forest over de 200 mensen van de achtergrond, en als alle vier bekend zijn 0,451, de eigen kans van de persoon. Weten wat education_num is, voegt tussen 0,139 en 0,226 toe, afhankelijk van wat er verder bekend is."},
        },
    ),
    "shapley": FunctionBox(
        name="shapley",
        picture=_shapley_picture,
        function="shapley",
        start=_SH_START,
        solution=_SH_SOLUTION,
        visible=_SH_VISIBLE,
        hidden=_SH_HIDDEN,
        digits=3,
        first=[_DOTS],
        mistakes=[
            Mistake("unweighted", wrong=_SH_UNWEIGHTED,
                    en="`{call}` gives {got}, and it should give {want}: you add up what column `i` adds to each coalition without weighing it. Multiply each difference by `weight(sum(known), len(known))`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt op wat kolom `i` aan elke coalitie toevoegt, zonder het te wegen. Vermenigvuldig elk verschil met `weight(sum(known), len(known))`."),
            Mistake("flipped", wrong=_SH_FLIPPED,
                    en="`{call}` gives {got}, and it should give {want}: the right size with the wrong sign. What column `i` adds is the value with it minus the value without it: `v[tuple(with_i)] - v[known]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de goede grootte met het verkeerde teken. Wat kolom `i` toevoegt, is de waarde mét de kolom min de waarde zonder: `v[tuple(with_i)] - v[known]`."),
            Mistake("mean", wrong=_SH_MEAN,
                    en="`{call}` gives {got}, and it should give {want}: every coalition counts the same in your sum. That only holds with two columns. Weigh each difference by `weight(sum(known), len(known))`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in je som telt elke coalitie even zwaar. Dat klopt alleen bij twee kolommen. Weeg elk verschil met `weight(sum(known), len(known))`."),
            Mistake("four", wrong=_SH_FOUR,
                    en="`{call}` gives {got}, and it should give {want}: your weights are those of four columns. The number of columns is the length of a coalition: `len(known)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je rekent elk gewicht uit voor vier kolommen. Het aantal kolommen is de lengte van een coalitie: `len(known)`."),
            Mistake("first", wrong=_SH_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your function stops after the first coalition without column `i`. If your `return` stands inside the `for` loop, Python leaves the function there. Put it after the loop, as far in as `total = 0`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie stopt na de eerste coalitie zonder kolom `i`. Als je `return` binnen de `for`-loop staat, verlaat Python de functie daar. Zet hem na de loop, even ver ingesprongen als `total = 0`."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...`: go over the coalitions in `v`; for each one with `known[i] == 0`, make the same coalition with column `i`, and add its weight times the difference of the two values to `total`. Return `total`.",
                      "nl": "Vervang de `...`: loop langs de coalities in `v`; maak voor elke coalitie met `known[i] == 0` dezelfde coalitie met kolom `i`, en tel zijn gewicht keer het verschil van de twee waarden op bij `total`. Geef `total` terug."},
            "cleared": _CLEARED,
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Work the example out by hand: column 0 adds 2 to the empty coalition and 6 to the coalition of column 1, and with two columns each counts `weight(0, 2)` = `weight(1, 2)` = 0.5.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Reken het voorbeeld met de hand uit: kolom 0 voegt 2 toe aan de lege coalitie en 6 aan de coalitie van kolom 1, en bij twee kolommen telt elk `weight(0, 2)` = `weight(1, 2)` = 0,5."},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use the values in `v` as they are given, for any column `i` and any number of columns?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de waarden in `v` zoals ze worden meegegeven, voor elke kolom `i` en elk aantal kolommen?"},
            "correct": {"en": "Right in all {n} cases. The picture runs your `shapley` on the 16 values of the worked example: age gets +0.087, education_num +0.181, hours_per_week −0.010 and capital_gain −0.032. Together 0.226, exactly the person's chance, 0.451, minus the background's average, 0.225. The 24 orders of the worked example give the same four numbers.",
                        "nl": "Goed in alle {n} gevallen. De tekening voert je `shapley` uit op de 16 waarden uit het uitgewerkte voorbeeld: age krijgt +0,087, education_num +0,181, hours_per_week −0,010 en capital_gain −0,032. Samen 0,226, precies de kans van de persoon, 0,451, min het gemiddelde van de achtergrond, 0,225. De 24 volgordes uit het uitgewerkte voorbeeld geven dezelfde vier getallen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("value", _V_START, "empty", None),
    ("value", _V_SOLUTION, "correct", None),
    ("value", _V_HEAD + "    rows = []\n    for row in background:\n        filled = list(row)\n"
              "        for j in range(len(known)):\n            if known[j]:\n                filled[j] = person[j]\n"
              "        rows.append(filled)\n    total = 0\n    for c in model(rows):\n        total = total + c\n"
              "    return total / len(background)\n", "correct", None),
    ("value", _V_SWAPPED, "wrong", "swapped"),
    ("value", _V_TOTAL, "wrong", "total"),
    ("value", _V_FIRST, "wrong", "first"),
    ("value", _V_PERSON, "wrong", "person-only"),
    ("value", _V_AVERAGE, "wrong", "background-only"),
    ("value", _V_HEAD + "    return 6.5\n", "almost", None),
    ("value", _v(end="    chances = model(rows)\n    return\n"), "wrong", "no-return"),
    ("value", _V_HEAD + "    ...\n    return 6.5\n", "wrong", "dots-left"),
    ("shapley", _SH_START, "empty", None),
    ("shapley", _SH_SOLUTION, "correct", None),
    ("shapley", _SH_HEAD + "    total = 0\n    for known, worth in v.items():\n        if known[i] == 1:\n"
                "            continue\n        with_i = known[:i] + (1,) + known[i + 1:]\n"
                "        total += weight(sum(known), len(known)) * (v[with_i] - worth)\n    return total\n",
     "correct", None),
    ("shapley", _SH_UNWEIGHTED, "wrong", "unweighted"),
    ("shapley", _SH_FLIPPED, "wrong", "flipped"),
    ("shapley", _SH_MEAN, "wrong", "mean"),
    ("shapley", _SH_FOUR, "wrong", "four"),
    ("shapley", _SH_FIRST, "wrong", "first"),
    ("shapley", _SH_HEAD + "    return 4.0\n", "almost", None),
    ("shapley", _sh(ret="    return\n"), "wrong", "unreached"),
    ("shapley", _SH_HEAD + "    ...\n    return 4.0\n", "wrong", "dots-left"),
]
