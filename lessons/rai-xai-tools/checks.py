# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-xai-tools: the forest and the person of rai-shapley-values (four columns of UCI Adult),
explained the ways the libraries do it. The slider of *The idea* draws ten backgrounds of 1, 10, 100 or 1000 rows of
the training half and gives the person's exact Shapley values for each. The worked example plays the game that shap's
TreeExplainer plays without data, by hand: go down each tree and, at a question about an unknown column, take both
branches by how many training rows went each way. The learner writes the permutation method, a Shapley value from a
list of orders, and the Shapley kernel of KernelSHAP; the pictures draw the learner's estimate as the orders grow, and
the learner's weights for 4 and for 14 columns. The two choice questions are about TreeExplainer with and without data,
and about KernelSHAP's default l1_reg."""
import functools
import html
import itertools
import math
import re

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake

COLUMNS = ["age", "education_num", "hours_per_week", "capital_gain"]
PERSON = 5                       # the person explained: position 5 of the test half, as in rai-shapley-values
BACKGROUND = 200                 # the background of rai-shapley-values: 200 rows of the training half, seed 0
SIZES = [1, 10, 100, 1000]       # the stops of the slider: how many rows each background has
DRAWS = 10                       # how many backgrounds the slider draws for each size, with seeds 0 to 9
ORDERS = 100                     # how many random orders the picture of exercise 1 hands the learner's function


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
    """The forest of rai-shapley-values: 100 trees whose leaves hold at least 20 people, grown on the training half."""
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


def coalitions() -> list:
    """The 16 coalitions of four columns, as tuples of 0s and 1s, in the order of itertools.product."""
    return list(itertools.product([0, 1], repeat=len(COLUMNS)))


def game(rows) -> dict:
    """The value of each coalition with these background rows: the forest's average chance of yes, with the known
    columns taken from the person (the interventional game of rai-shapley-values)."""
    import numpy as np
    me = person()
    out = {}
    for known in coalitions():
        out[known] = float(chance(np.where(np.array(known) == 1, me, rows)).mean())
    return out


def weight(s: int, n: int) -> float:
    """How much a coalition of s columns counts in the Shapley value of a column outside it, with n columns."""
    return math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)


def shapley_of(v: dict, i: int) -> float:
    """The Shapley value of column i in the game v, by the formula of rai-shapley-values."""
    total = 0.0
    for known in v:
        if known[i] == 0:
            with_i = list(known)
            with_i[i] = 1
            total += weight(sum(known), len(known)) * (v[tuple(with_i)] - v[known])
    return total


def shares_of(v: dict) -> tuple:
    return tuple(shapley_of(v, i) for i in range(len(next(iter(v)))))


@functools.lru_cache(maxsize=1)
def background():
    """The 200 rows of rai-shapley-values. Read-only."""
    import numpy as np
    X, y, train, test = table()
    out = X[train][np.random.default_rng(0).choice(len(train), BACKGROUND, replace=False)]
    out.flags.writeable = False
    return out


@functools.lru_cache(maxsize=1)
def values() -> dict:
    """The 16 values of rai-shapley-values: the game over its 200 background rows."""
    return game(background())


@functools.lru_cache(maxsize=1)
def shares() -> tuple:
    """The person's four exact Shapley values over the 200 rows: +0.087, +0.181, -0.010, -0.032."""
    return shares_of(values())


@functools.lru_cache(maxsize=None)
def draws(size: int) -> tuple:
    """For DRAWS backgrounds of `size` rows of the training half (seeds 0 to 9, without replacement): each one's
    average chance of yes, the value of the empty coalition, and the person's four Shapley values."""
    import numpy as np
    X, y, train, test = table()
    out = []
    for seed in range(DRAWS):
        rows = X[train][np.random.default_rng(seed).choice(len(train), size, replace=False)]
        v = game(rows)
        out.append((v[(0, 0, 0, 0)], shares_of(v)))
    return tuple(out)


@functools.lru_cache(maxsize=1)
def halves() -> tuple:
    """As shap's masker does with a background of 200 rows: keep 100 of them. Ten such draws (seeds 0 to 9): each
    one's average and the four Shapley values."""
    import numpy as np
    bg = background()
    out = []
    for seed in range(DRAWS):
        v = game(bg[np.random.default_rng(seed).choice(BACKGROUND, 100, replace=False)])
        out.append((v[(0, 0, 0, 0)], shares_of(v)))
    return tuple(out)


# --- the game TreeExplainer plays without data: down the trees, by how many training rows went each way ---------------

def down(tree, node: int, known, row) -> float:
    """The tree's expected chance of yes for `row` when only the columns in `known` are known: follow the row at a
    question about a known column, and take both branches, weighed by their training rows, at an unknown one."""
    t = tree.tree_
    if t.children_left[node] == -1:
        return float(t.value[node, 0, 1] / t.value[node, 0].sum())
    left, right = t.children_left[node], t.children_right[node]
    column = t.feature[node]
    if known[column] == 1:
        return down(tree, left if row[column] <= t.threshold[node] else right, known, row)
    w = t.weighted_n_node_samples
    return float((w[left] * down(tree, left, known, row) + w[right] * down(tree, right, known, row)) / w[node])


@functools.lru_cache(maxsize=1)
def tree_values() -> dict:
    """The value of each coalition in the game of the trees: the forest's average over its trees of `down`."""
    me = person()
    trees = forest().estimators_
    out = {}
    for known in coalitions():
        total = 0.0
        for tree in trees:
            total += down(tree, 0, known, me)
        out[known] = total / len(trees)
    return out


@functools.lru_cache(maxsize=1)
def tree_shares() -> tuple:
    return shares_of(tree_values())


# --- KernelSHAP: the weighted regression on the coalitions ---------------------------------------------------------------

def shapley_kernel(m: int, s: int) -> float:
    """Lundberg and Lee's Shapley kernel: the weight of a coalition with s of the m columns known."""
    if s == 0 or s == m:
        return math.inf
    return (m - 1) / (math.comb(m, s) * s * (m - s))


def kernel_shap(v: dict, kernel=shapley_kernel) -> list:
    """The values of the weighted linear regression of KernelSHAP on all coalitions of the game v: the empty and the
    full coalition are kept exact (their weight is infinite) by taking the base value and the last column out of the
    regression, as Lundberg and Lee describe; the other coalitions count with kernel(m, s)."""
    import numpy as np
    m = len(next(iter(v)))
    empty, full = (0,) * m, (1,) * m
    rows, said, weights = [], [], []
    for known in v:
        s = sum(known)
        if 0 < s < m:
            rows.append([known[j] - known[m - 1] for j in range(m - 1)])
            said.append(v[known] - v[empty] - known[m - 1] * (v[full] - v[empty]))
            weights.append(kernel(m, s))
    root = np.sqrt(np.array(weights, dtype=float))
    phi, *_ = np.linalg.lstsq(np.array(rows, dtype=float) * root[:, None], np.array(said) * root, rcond=None)
    out = [float(p) for p in phi]
    return out + [v[full] - v[empty] - sum(out)]


def kernel_samples(m: int) -> int:
    """How many coalitions KernelSHAP's nsamples="auto" asks for with m varying columns: 2m + 2048, but no more than
    the 2^m - 2 coalitions between the empty and the full one when m is at most 30."""
    n = 2 * m + 2 ** 11
    return min(n, 2 ** m - 2) if m <= 30 else n


@functools.lru_cache(maxsize=1)
def orders() -> tuple:
    """The ORDERS random orders of the picture of exercise 1: np.random.default_rng(0).permutation(4), 100 times."""
    import numpy as np
    rng = np.random.default_rng(0)
    out = []
    for _ in range(ORDERS):
        out.append(tuple(int(j) for j in rng.permutation(len(COLUMNS))))
    return tuple(out)


def sampled_of(v: dict, i: int, these) -> float:
    """The permutation method: what column i adds when it joins, averaged over the orders given."""
    total = 0.0
    for order in these:
        known = [0] * len(order)
        for j in order:
            before = v[tuple(known)]
            known[j] = 1
            if j == i:
                total += v[tuple(known)] - before
    return total / len(these)


def runs_of_ten() -> list:
    """education_num's estimate from 10 random orders, in 10 runs (seeds 0 to 9)."""
    import numpy as np
    v = values()
    out = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        these = []
        for _ in range(10):
            these.append(tuple(int(j) for j in rng.permutation(len(COLUMNS))))
        out.append(sampled_of(v, 1, these))
    return out


def _show(x: float, lang: str, digits: int = 3, sign: bool = False) -> str:
    """A number with this many decimals and a real minus sign, with a plus sign if asked; a decimal comma in Dutch."""
    text = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-+")
    text = text.replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _small(x: float, lang: str) -> str:
    """A number with three significant digits and never in e-notation, such as 0.0000773; infinity as ∞."""
    if math.isinf(x):
        return "∞" if x > 0 else "−∞"
    if x == 0:
        return "0"
    digits = max(3, -math.floor(math.log10(abs(x))) + 2)
    return _show(x, lang, digits=digits)


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"

# --- the slider of *The idea*: ten backgrounds of each size ----------------------------------------------------------------

_SIZE_WORDS = {
    "en": {"title": "This person's Shapley values, 10 backgrounds of {k}", "x": "Shapley value",
           "rows": "rows", "row": "row",
           "alt": "Four rows of 10 dots, one row for each column, one dot for each of 10 backgrounds of {k} {rows} drawn "
                  "from the training half: the person's Shapley value for that column with that background. {said}",
           "said": "Ten backgrounds of {k} {rows} each, drawn from the training half. education_num gets from "
                   "<span class=\"c-pred ea-num\">{e0}</span> to <span class=\"c-pred ea-num\">{e1}</span>, age from "
                   "{a0} to {a1}, hours_per_week from {h0} to {h1} and capital_gain from {c0} to {c1}. The average "
                   "chance of yes of the background, where the values start, goes from {v0} to {v1}; in each draw "
                   "the four values add up to 0.451 minus that draw's average."},
    "nl": {"title": "Shapley-waarden van deze persoon, 10 achtergronden van {k}", "x": "Shapley-waarde",
           "rows": "rijen", "row": "rij",
           "alt": "Vier rijen van 10 punten, één rij voor elke kolom, één punt voor elk van 10 achtergronden van {k} "
                  "{rows} uit de helft om op te trainen: de Shapley-waarde van de persoon voor die kolom bij die "
                  "achtergrond. {said}",
           "said": "Tien achtergronden van elk {k} {rows}, getrokken uit de helft om op te trainen. education_num "
                   "krijgt van <span class=\"c-pred ea-num\">{e0}</span> tot <span class=\"c-pred ea-num\">{e1}</span>, "
                   "age van {a0} tot {a1}, hours_per_week van {h0} tot {h1} en capital_gain van {c0} tot {c1}. De "
                   "gemiddelde kans op ja van de achtergrond, waar de waarden beginnen, gaat van {v0} tot {v1}; bij "
                   "elke trekking tellen de vier waarden op tot 0,451 min het gemiddelde van die trekking."},
}


@functools.lru_cache(maxsize=1)
def _size_span() -> tuple:
    """One range of the x-axis for every stop of the slider, so that the stops can be compared."""
    lo, hi = 0.0, 0.0
    for k in SIZES:
        for avg, parts in draws(k):
            lo, hi = min([lo] + list(parts)), max([hi] + list(parts))
    pad = (hi - lo) * 0.04
    return (lo - pad, hi + pad)


def spread(k: int) -> dict:
    """For backgrounds of k rows: the lowest and highest average, and the lowest and highest value of each column."""
    out = {"avg": (min(a for a, p in draws(k)), max(a for a, p in draws(k)))}
    for j, name in enumerate(COLUMNS):
        out[name] = (min(p[j] for a, p in draws(k)), max(p[j] for a, p in draws(k)))
    return out


def size_picture(k: int, lang: str) -> tuple[str, str]:
    """For the slider of *The idea*: a row of 10 dots for each column, the person's Shapley values for 10 backgrounds
    of k rows, and the sentence with the ranges."""
    w = _SIZE_WORDS[lang]
    span = _size_span()
    n = len(COLUMNS)
    p = chart.Panel(x=span, y=(-0.7, n - 0.1), title=w["title"].format(k=k), xlabel=w["x"], ylabel="", yticks=[])
    p.line([0, 0], [-0.4, n - 0.5], role="ink-3", width=1.2, dashed=True)
    sp = spread(k)
    for j, name in enumerate(COLUMNS):
        y = n - 1 - j
        xs = [parts[j] for avg, parts in draws(k)]
        p.points(xs, [y - 0.12] * len(xs), role="pred", r=4)
        p.text(span[0], y + 0.25, name, role="ink", anchor="start", halo=True)
    fill = {"k": k, "rows": w["row"] if k == 1 else w["rows"],
            "e0": _show(sp["education_num"][0], lang, sign=True), "e1": _show(sp["education_num"][1], lang, sign=True),
            "a0": _show(sp["age"][0], lang, sign=True), "a1": _show(sp["age"][1], lang, sign=True),
            "h0": _show(sp["hours_per_week"][0], lang, sign=True), "h1": _show(sp["hours_per_week"][1], lang, sign=True),
            "c0": _show(sp["capital_gain"][0], lang, sign=True), "c1": _show(sp["capital_gain"][1], lang, sign=True),
            "v0": _show(sp["avg"][0], lang), "v1": _show(sp["avg"][1], lang)}
    said = w["said"].format(**fill)
    alt = w["alt"].format(k=k, rows=fill["rows"], said=re.sub(r"<[^>]+>", "", said))
    svg = chart.figure([p], alt=alt, width=460, height=270)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    return svg, said


# --- helpers for the pictures of the exercises --------------------------------------------------------------------------

def _mine(seen):
    """The learner's function, or None when the code stopped or the name is not a function."""
    mine = seen.mine
    if mine is None:
        return None
    fn = mine.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _number(v, finite: bool = True):
    """A float from what a function gave back, or None when it is not one number (or, with `finite`, not a finite
    one)."""
    if isinstance(v, bool):
        return None
    if hasattr(v, "item") and getattr(v, "size", 1) == 1:
        v = v.item()
    if isinstance(v, (int, float)) and not math.isnan(v) and (math.isfinite(v) or not finite):
        return float(v)
    return None


def _run(run):
    from everything_ai import check as checking
    try:
        return checking.with_time_limit(run)
    except Exception:
        return None


# --- exercise 1: a Shapley value from sampled orders --------------------------------------------------------------------

_SAMPLED_WORDS = {
    "en": {"title": "Your sampled(), for education_num, as the orders grow", "x": "how many of the 100 orders",
           "y": "your sampled()", "exact": "exact {v}",
           "alt": "A line of what your sampled() gives for education_num of this person with the first 1 to 100 of 100 "
                  "random orders, and a dashed line at the exact Shapley value, {exact}. {key}",
           "key": "With the first order your sampled() gives education_num {a}; with the first 10 orders {b}, and with "
                  "all 100 {c}. The exact Shapley value of the last lesson is {exact}. With all 100 orders your four "
                  "estimates add up to {sum}; the person's chance minus the background's average is {gap}."},
    "nl": {"title": "Jouw sampled(), voor education_num, bij steeds meer volgordes", "x": "hoeveel van de 100 volgordes",
           "y": "jouw sampled()", "exact": "exact {v}",
           "alt": "Een lijn van wat je sampled() geeft voor education_num van deze persoon met de eerste 1 tot 100 van "
                  "100 willekeurige volgordes, en een gestreepte lijn bij de exacte Shapley-waarde, {exact}. {key}",
           "key": "Met de eerste volgorde geeft je sampled() education_num {a}; met de eerste 10 volgordes {b}, en met "
                  "alle 100 {c}. De exacte Shapley-waarde uit de vorige les is {exact}. Met alle 100 volgordes tellen je "
                  "vier schattingen op tot {sum}; de kans van de persoon min het gemiddelde van de achtergrond is {gap}."},
}


def _sampled_picture(seen) -> str:
    fn = _mine(seen)
    if fn is None:
        return ""
    v = values()
    these = orders()

    def run():
        line = []
        for n in range(1, len(these) + 1):
            g = _number(fn(dict(v), 1, list(these[:n])))
            if g is None:
                return None
            line.append(g)
        four = []
        for i in range(len(COLUMNS)):
            g = _number(fn(dict(v), i, list(these)))
            if g is None:
                return None
            four.append(g)
        return line, four

    got = _run(run)
    if got is None:
        return ""
    line, four = got
    lang = seen.lang
    w = _SAMPLED_WORDS[lang]
    exact = shares()[1]
    fill = {"a": _show(line[0], lang, sign=True), "b": _show(line[9], lang, sign=True),
            "c": _show(line[-1], lang, sign=True), "exact": _show(exact, lang, sign=True),
            "sum": _show(sum(four), lang), "gap": _show(v[(1, 1, 1, 1)] - v[(0, 0, 0, 0)], lang)}
    key = w["key"].format(**fill)
    ys = line + [exact]
    lo, hi = min(ys + [0.0]), max(ys + [0.0])
    pad = (hi - lo) * 0.08 + 1e-9
    n = len(line)
    p = chart.Panel(x=(0, n + 1), y=(lo - pad, hi + pad), title="", xlabel=w["x"], ylabel=w["y"],
                    xticks=[1, 25, 50, 75, 100])
    p.line([1, n], [exact, exact], role="ink-3", width=1.4, dashed=True)
    p.text(n, exact - pad * 1.6, w["exact"].format(v=_show(exact, lang, sign=True)), role="ink-2", anchor="end",
           halo=True)
    p.line(list(range(1, n + 1)), line, role="pred", width=2)
    svg = chart.figure([p], alt=w["alt"].format(exact=fill["exact"], key=key), width=520, height=270)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(key)}</div>')


# --- exercise 2: the Shapley kernel ---------------------------------------------------------------------------------------

_KERNEL_WORDS = {
    "en": {"title": "Your kernel_weight(), for 4 and for 14 columns", "x": "how many columns are known",
           "y": "your weight", "four": "4 columns", "many": "14 columns",
           "alt": "Two bar charts of what your kernel_weight() gives a coalition by how many columns it knows, between "
                  "none and all: for 4 columns and for 14. {key}",
           "key": "For 4 columns your weights are {w1}, {w2} and {w3}, with 1, 2 and 3 columns known. For 14 columns a "
                  "coalition with 1 known weighs {m1}, with 7 known {m7} and with 13 known {m13}{ratio}.",
           "ratio": ": the one with 1 known counts {r} times as much as the one with 7"},
    "nl": {"title": "Jouw kernel_weight(), voor 4 en voor 14 kolommen", "x": "hoeveel kolommen bekend zijn",
           "y": "jouw gewicht", "four": "4 kolommen", "many": "14 kolommen",
           "alt": "Twee staafdiagrammen van wat je kernel_weight() een coalitie geeft, naar hoeveel kolommen ze kent, "
                  "tussen geen en alle: voor 4 kolommen en voor 14. {key}",
           "key": "Voor 4 kolommen geeft je functie {w1}, {w2} en {w3}, met 1, 2 en 3 kolommen bekend. Voor 14 kolommen "
                  "weegt een coalitie met 1 bekende {m1}, met 7 bekende {m7} en met 13 bekende {m13}{ratio}.",
           "ratio": ": die met 1 bekende telt {r} keer zo zwaar als die met 7"},
}


def _kernel_picture(seen) -> str:
    fn = _mine(seen)
    if fn is None:
        return ""

    def run():
        out = {}
        for m in (4, 14):
            for s in range(1, m):
                g = _number(fn(m, s))
                if g is None:
                    return None
                out[(m, s)] = g
        return out

    got = _run(run)
    if got is None:
        return ""
    lang = seen.lang
    w = _KERNEL_WORDS[lang]
    ratio = ""
    if got[(14, 7)] > 0 and got[(14, 1)] > 0:
        r = got[(14, 1)] / got[(14, 7)]
        ratio = w["ratio"].format(r=_show(r, lang, digits=0) if r >= 10 else _show(r, lang, digits=2))
    fill = {"w1": _small(got[(4, 1)], lang), "w2": _small(got[(4, 2)], lang), "w3": _small(got[(4, 3)], lang),
            "m1": _small(got[(14, 1)], lang), "m7": _small(got[(14, 7)], lang), "m13": _small(got[(14, 13)], lang),
            "ratio": ratio}
    key = w["key"].format(**fill)
    panels = []
    for m, title in ((4, w["four"]), (14, w["many"])):
        ys = [got[(m, s)] for s in range(1, m)]
        lo, hi = min(ys + [0.0]), max(ys + [0.0])
        pad = (hi - lo) * 0.08 + 1e-9
        p = chart.Panel(x=(0.4, m - 0.4), y=(lo - pad, hi + pad), title=title, xlabel=w["x"], ylabel=w["y"],
                        xticks=list(range(1, m)) if m == 4 else [1, 4, 7, 10, 13])
        for s in range(1, m):
            p.rect(s - 0.32, min(0.0, got[(m, s)]), s + 0.32, max(0.0, got[(m, s)]), role="hyper", solid=True)
        panels.append(p)
    svg = chart.figure(panels, alt=w["alt"].format(key=key), width=280, height=250)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(key)}</div>')


# The lesson is done when these are correct. "data" and "ten" are choice questions.
REQUIRED = ["sampled", "kernel", "data", "ten"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

# --- exercise 1
_SA_HEAD = ("def sampled(v, i, orders):\n"
            "    # v[known] is the value of a coalition: known is a tuple of 0s and 1s, one for each column\n"
            "    # each order is a tuple with every column once, in the order in which they join\n")
_SA_START = _SA_HEAD + "    ...\n"


def _sa(add: str = "            if j == i:\n                total = total + v[tuple(known)] - before\n",
        take: str = "            before = v[tuple(known)]\n            known[j] = 1\n",
        reset: bool = True, ret: str = "    return total / len(orders)\n") -> str:
    """The body of sampled with one part changed."""
    fresh = "        known = [0] * len(order)\n"
    return (_SA_HEAD + "    total = 0\n" + ("" if reset else "    known = [0] * len(orders[0])\n")
            + "    for order in orders:\n" + (fresh if reset else "") + "        for j in order:\n" + take + add + ret)


_SA_SOLUTION = _sa()
_SA_EVERY = _sa(add="            total = total + v[tuple(known)] - before\n")
_SA_SUM = _sa(ret="    return total\n")
_SA_FACTORIAL = "import math\n\n\n" + _sa(ret="    return total / math.factorial(len(orders[0]))\n")
_SA_FIRST = _sa(ret="        return total / len(orders)\n")
_SA_ONCE = _sa(reset=False)
_SA_AFTER = _sa(take="            known[j] = 1\n            before = v[tuple(known)]\n")
_SA_TWO = {(0, 0): 0, (1, 0): 2, (0, 1): 4, (1, 1): 10}
_SA_VISIBLE = [(_SA_TWO, 0, [(1, 0), (1, 0), (0, 1)])]


def _game(n: int, rule) -> dict:
    """A game of n columns: the value of each coalition by `rule`, a function of the tuple of 0s and 1s."""
    out = {}
    for known in itertools.product([0, 1], repeat=n):
        out[known] = rule(known)
    return out


_SA_HIDDEN = [
    (_SA_TWO, 1, [(1, 0), (1, 0), (0, 1)]),
    (_SA_TWO, 0, [(0, 1)]),
    (_game(3, lambda k: 0.2 + 0.1 * k[0] + 0.3 * k[0] * k[2]), 2, [(0, 1, 2), (2, 1, 0), (1, 2, 0), (1, 0, 2)]),
    (_game(3, lambda k: k[0] * k[1] * k[2] + 0.5 * k[1]), 1, [(0, 2, 1), (1, 0, 2), (2, 1, 0), (0, 2, 1), (2, 0, 1)]),
    (_game(4, lambda k: k[0] + 2 * k[1] + 3 * k[2] + 4 * k[3] + 5 * k[0] * k[1]), 0,
     [(1, 2, 3, 0), (0, 1, 2, 3), (3, 1, 0, 2)]),
    (_game(4, lambda k: 0.1 + 0.2 * k[1] + 0.6 * k[0] * k[1] * k[3]), 3, [(3, 2, 1, 0), (0, 1, 3, 2)]),
]

# --- exercise 2
_KW_HEAD = ("import math\n"
            "\n"
            "\n"
            "def kernel_weight(m, s):\n"
            "    # the weight of a coalition with s of the m columns known, in the regression of KernelSHAP\n")
_KW_START = _KW_HEAD + "    ...\n"
_KW_ENDS = "    if s == 0 or s == m:\n        return math.inf\n"


def _kw(top: str = "(m - 1)", bottom: str = "(math.comb(m, s) * s * (m - s))", ends: str = _KW_ENDS) -> str:
    return _KW_HEAD + ends + f"    return {top} / {bottom}\n"


_KW_SOLUTION = _kw()
_KW_NO_COMB = _kw(bottom="(s * (m - s))")
_KW_M = _kw(top="m")
_KW_ZERO = _kw(ends="    if s == 0 or s == m:\n        return 0\n")
_KW_SHAPLEY = (_KW_HEAD + _KW_ENDS
               + "    return math.factorial(s) * math.factorial(m - s - 1) / math.factorial(m)\n")
_KW_VISIBLE = [(4, 1)]
_KW_HIDDEN = [(4, 0), (4, 4), (4, 2), (4, 3), (14, 1), (14, 7), (3, 1), (2, 1), (1, 0), (6, 2)]

EXERCISES = {
    "sampled": FunctionBox(
        name="sampled",
        picture=_sampled_picture,
        function="sampled",
        start=_SA_START,
        solution=_SA_SOLUTION,
        visible=_SA_VISIBLE,
        hidden=_SA_HIDDEN,
        digits=3,
        first=[_DOTS],
        mistakes=[
            Mistake("every", wrong=_SA_EVERY,
                    en="`{call}` gives {got}, and it should give {want}: the value of all columns minus the value of none, what all the columns add together in every order. Only what column `i` adds counts: add to `total` only when the column that joins is `i`, `if j == i:`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de waarde van alle kolommen min die van geen, wat alle kolommen samen toevoegen in elke volgorde. Alleen wat kolom `i` toevoegt telt: tel alleen bij `total` op als de kolom die aanschuift `i` is, `if j == i:`."),
            Mistake("sum", wrong=_SA_SUM,
                    en="`{call}` gives {got}, and it should give {want}: the sum over the orders, not the average. Divide `total` by the number of orders, `len(orders)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de som over de volgordes, niet het gemiddelde. Deel `total` door het aantal volgordes, `len(orders)`."),
            Mistake("factorial", wrong=_SA_FACTORIAL,
                    en="`{call}` gives {got}, and it should give {want}: the sum divided by the number of all possible orders of the columns, not by the number of orders in `orders`. Some orders may be missing and some may come twice: divide by `len(orders)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de som gedeeld door het aantal mogelijke volgordes van de kolommen, niet door het aantal volgordes in `orders`. Er kunnen volgordes ontbreken en er kunnen er dubbel in zitten: deel door `len(orders)`."),
            Mistake("first", wrong=_SA_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: only the first order counts. If your `return` stands inside the `for` loop over the orders, Python leaves the function there; and `known` must start again at all 0s for each order.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen de eerste volgorde telt. Als je `return` binnen de `for`-loop over de volgordes staat, verlaat Python de functie daar; en `known` moet voor elke volgorde opnieuw met allemaal nullen beginnen."),
            Mistake("zero", wrong=_SA_AFTER,
                    en="`{call}` gives {got}, and it should give {want}: 0 in every case, as if column `i` never changed the value. Take `before`, the value without the column, before you set `known[j]` to 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: 0 in elk geval, alsof kolom `i` de waarde nooit verandert. Neem `before`, de waarde zonder de kolom, voordat je `known[j]` op 1 zet."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...`: for each order, start `known` at all 0s and let its columns join one at a time; when column `i` joins, add what the value gains to `total`. Return `total / len(orders)`.",
                      "nl": "Vervang de `...`: begin voor elke volgorde `known` met allemaal nullen en laat de kolommen een voor een aanschuiven; als kolom `i` aanschuift, tel op bij `total` wat de waarde erbij krijgt. Geef `total / len(orders)` terug."},
            "cleared": _CLEARED,
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Work the example out by hand: in the order (1, 0) Ash, column 0, joins Bo and adds 10 − 4 = 6; in (0, 1) Ash comes first and adds 2. Two times 6 and once 2, over three orders.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Reken het voorbeeld met de hand uit: in de volgorde (1, 0) schuift Ash, kolom 0, aan bij Bo en voegt 10 − 4 = 6 toe; in (0, 1) komt Ash eerst en voegt 2 toe. Twee keer 6 en één keer 2, over drie volgordes."},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for any column `i`, any number of columns and any list of orders, also one with a single order?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke kolom `i`, elk aantal kolommen en elke lijst met volgordes, ook een met maar één volgorde?"},
            "correct": {"en": "Right in all {n} cases. The picture runs your `sampled` on the 16 values of the last lesson with the first 1 to 100 of 100 random orders: with one order education_num gets +0.222, with 10 +0.188, with 100 +0.182, against the exact +0.181. However many orders you take, the four estimates add up to exactly 0.226: each order shares out the whole gap.",
                        "nl": "Goed in alle {n} gevallen. De tekening voert je `sampled` uit op de 16 waarden uit de vorige les met de eerste 1 tot 100 van 100 willekeurige volgordes: met één volgorde krijgt education_num +0,222, met 10 +0,188, met 100 +0,182, tegen exact +0,181. Hoeveel volgordes je ook neemt, de vier schattingen tellen precies op tot 0,226: elke volgorde verdeelt het hele verschil."},
        },
    ),
    "kernel": FunctionBox(
        name="kernel",
        picture=_kernel_picture,
        function="kernel_weight",
        start=_KW_START,
        solution=_KW_SOLUTION,
        visible=_KW_VISIBLE,
        hidden=_KW_HIDDEN,
        digits=4,
        first=[_DOTS],
        mistakes=[
            Mistake("no-comb", wrong=_KW_NO_COMB,
                    en="`{call}` gives {got}, and it should give {want}: what all coalitions with `s` of the `m` columns weigh together. Each one weighs that divided by their number, `math.comb(m, s)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: wat alle coalities met `s` van de `m` kolommen samen wegen. Elk ervan weegt dat gedeeld door hun aantal, `math.comb(m, s)`."),
            Mistake("shapley", wrong=_KW_SHAPLEY,
                    en="`{call}` gives {got}, and it should give {want}: the weight of the Shapley formula of the last lesson, s!(m − s − 1)!/m!. The regression of KernelSHAP has a weight of its own: (m − 1) divided by `math.comb(m, s) * s * (m - s)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het gewicht uit de formule van Shapley van de vorige les, s!(m − s − 1)!/m!. De regressie van KernelSHAP heeft een eigen gewicht: (m − 1) gedeeld door `math.comb(m, s) * s * (m - s)`."),
            Mistake("m", wrong=_KW_M,
                    en="`{call}` gives {got}, and it should give {want}: m / (C(m, s) · s · (m − s)). The top of the fraction is `m - 1`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: m / (C(m, s) · s · (m − s)). Boven de streep staat `m - 1`."),
            Mistake("zero-ends", wrong=_KW_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: the empty and the full coalition weigh infinitely much, `math.inf`. That makes the values start at the background's average and add up to the prediction.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de lege en de volle coalitie wegen oneindig zwaar, `math.inf`. Daardoor beginnen de waarden bij het gemiddelde van de achtergrond en tellen ze op tot de voorspelling."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...`: return `math.inf` when `s` is 0 or `m`, and otherwise (m − 1) divided by `math.comb(m, s) * s * (m - s)`.",
                      "nl": "Vervang de `...`: geef `math.inf` terug als `s` 0 of `m` is, en anders (m − 1) gedeeld door `math.comb(m, s) * s * (m - s)`."},
            "cleared": _CLEARED,
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Work the example out by hand: with 4 columns there are `math.comb(4, 1)` = 4 coalitions with 1 known, so the weight is 3 / (4 · 1 · 3).",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Reken het voorbeeld met de hand uit: bij 4 kolommen zijn er `math.comb(4, 1)` = 4 coalities met 1 bekende, dus het gewicht is 3 / (4 · 1 · 3)."},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for every `m` and every `s` from 0 to `m`, including the empty and the full coalition?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke `m` en elke `s` van 0 tot `m`, ook voor de lege en de volle coalitie?"},
            "correct": {"en": "Right in all {n} cases. For 4 columns the weights are 0.25, 0.125 and 0.25; for 14 columns a coalition with 1 known counts 924 times as much as one with 7. With these weights, the regression on the 14 coalitions of four columns, between none and all, gives exactly the Shapley values of the last lesson: +0.087, +0.181, −0.010 and −0.032.",
                        "nl": "Goed in alle {n} gevallen. Voor 4 kolommen krijgen coalities met 1, 2 en 3 bekende kolommen 0,25, 0,125 en 0,25; voor 14 kolommen telt een coalitie met 1 bekende 924 keer zo zwaar als een met 7. Met dit gewicht geeft de regressie op de 14 coalities van vier kolommen, tussen geen en alle, precies de Shapley-waarden van de vorige les: +0,087, +0,181, −0,010 en −0,032."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("sampled", _SA_START, "empty", None),
    ("sampled", _SA_SOLUTION, "correct", None),
    ("sampled", _SA_HEAD + "    total = 0\n    for order in orders:\n        place = order.index(i)\n"
                "        before = [0] * len(order)\n        for j in order[:place]:\n            before[j] = 1\n"
                "        after = list(before)\n        after[i] = 1\n"
                "        total += v[tuple(after)] - v[tuple(before)]\n    return total / len(orders)\n",
     "correct", None),
    ("sampled", _SA_EVERY, "wrong", "every"),
    ("sampled", _SA_SUM, "wrong", "sum"),
    ("sampled", _SA_FACTORIAL, "wrong", "factorial"),
    ("sampled", _SA_FIRST, "wrong", "first"),
    ("sampled", _SA_ONCE, "wrong", "first"),
    ("sampled", _SA_AFTER, "wrong", "zero"),
    ("sampled", _SA_HEAD + "    return 14 / 3\n", "almost", None),
    ("sampled", _sa(ret="    return\n"), "wrong", "no-return"),
    ("sampled", _SA_HEAD + "    ...\n    return 14 / 3\n", "wrong", "dots-left"),
    ("kernel", _KW_START, "empty", None),
    ("kernel", _KW_SOLUTION, "correct", None),
    ("kernel", _KW_HEAD + "    if 0 < s < m:\n        return (m - 1) / math.comb(m, s) / s / (m - s)\n"
               "    return float(\"inf\")\n", "correct", None),
    ("kernel", _KW_NO_COMB, "wrong", "no-comb"),
    ("kernel", _KW_SHAPLEY, "wrong", "shapley"),
    ("kernel", _KW_M, "wrong", "m"),
    ("kernel", _KW_ZERO, "wrong", "zero-ends"),
    ("kernel", _KW_HEAD + "    return 0.25\n", "almost", None),
    ("kernel", _kw(ends="    if s == 0 or s == m:\n        return\n"), "wrong", "unreached"),
    ("kernel", _KW_HEAD + "    ...\n    return 0.25\n", "wrong", "dots-left"),
]
