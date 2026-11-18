# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-permutation-importance: a random forest on three columns of UCI Adult and a fourth column
of random numbers, grown on one half of the table and tested on the other. Its own importance, from the splits on the
training data, puts the random numbers first; shuffling a column of the test half and measuring how much worse the
forest gets (permutation importance) gives them nothing. The learner writes permutation importance for a small tree,
works out a tree's impurity importance from its nodes, says which half to measure on, and what leaving out one of two
copies of a column shows; with what the forests of the slider measured, the learner's importance() on the test half,
and the learner's shares."""
import functools
import html
import math
import re

from everything_ai import chart, yours
from everything_ai.check import CodeBox, FunctionBox, Mistake

COLUMNS = ["age", "education_num", "hours_per_week"]
NOISE_SEED = 2                  # the column of random numbers: whole numbers from 0 to 999, drawn with this seed
LEAVES = [1, 2, 5, 10, 20, 50]  # the stops of the slider: the fewest people a leaf of the forest may hold

# What a forest of 100 trees (random_state=0) measured on the four columns for each stop of the slider: how often it
# is right on the training half and on the test half, its own importance (feature_importances_), and the mean and the
# spread of permutation importance on the test half (n_repeats=5, random_state=0); the columns in the order age,
# education_num, hours_per_week, noise. tests/test_rai_permutation_importance.py grows the forests again. With 1, the
# default, the forest is the one of the worked example.
MEASURED = {
    1: {"train": 0.9959, "test": 0.7601, "mdi": [0.243, 0.152, 0.123, 0.482],
        "perm": [0.0288, 0.0444, 0.0132, -0.0002], "sd": [0.0011, 0.0013, 0.0013, 0.0017]},
    2: {"train": 0.9264, "test": 0.7803, "mdi": [0.275, 0.197, 0.136, 0.391],
        "perm": [0.0306, 0.0531, 0.0172, 0.0003], "sd": [0.0017, 0.0013, 0.0013, 0.0015]},
    5: {"train": 0.8598, "test": 0.79, "mdi": [0.307, 0.27, 0.149, 0.274],
        "perm": [0.0277, 0.0553, 0.0177, -0.0005], "sd": [0.0016, 0.0009, 0.0008, 0.0012]},
    10: {"train": 0.8315, "test": 0.794, "mdi": [0.327, 0.325, 0.161, 0.187],
         "perm": [0.0268, 0.0587, 0.0187, -0.0011], "sd": [0.0011, 0.0007, 0.001, 0.001]},
    20: {"train": 0.8179, "test": 0.7969, "mdi": [0.339, 0.377, 0.167, 0.117],
         "perm": [0.0279, 0.0621, 0.018, -0.0003], "sd": [0.0009, 0.0008, 0.0011, 0.0007]},
    50: {"train": 0.8073, "test": 0.7985, "mdi": [0.344, 0.431, 0.173, 0.052],
         "perm": [0.0273, 0.0633, 0.0163, -0.0009], "sd": [0.0009, 0.001, 0.0006, 0.0004]},
}

# A tree of depth 3 (random_state=0) grown on the training half from the three columns, as scikit-learn keeps it in
# tree_: for each node the column it splits on (-2 for a leaf), the node to its left and to its right (-1 for a leaf),
# its Gini impurity, rounded to four decimals, and how many people of the training half reach it. Exercise 2 starts
# from these; the test grows the tree again.
TREE = {
    "feature": [1, 0, 0, -2, -2, 2, -2, -2, 0, 0, -2, -2, 2, -2, -2],
    "left": [1, 2, 3, -1, -1, 6, -1, -1, 9, 10, -1, -1, 13, -1, -1],
    "right": [8, 5, 4, -1, -1, 7, -1, -1, 12, 11, -1, -1, 14, -1, -1],
    "impurity": [0.3618, 0.266, 0.1053, 0.0339, 0.1975, 0.3609, 0.3066, 0.4597, 0.4989, 0.1977, 0.084, 0.3, 0.4943,
                 0.4971, 0.4481],
    "people": [24421, 18358, 7966, 4698, 3268, 10392, 7487, 2905, 6063, 1052, 546, 506, 5011, 2709, 2302],
}


@functools.lru_cache(maxsize=1)
def table():
    """The three columns of UCI Adult and the column of random numbers as one array, who earns more than 50,000
    dollars, and the positions of the training half and of the test half (the table shuffled with seed 0 and cut in
    two). Read-only."""
    import numpy as np
    import pandas as pd
    from everything_ai import paths
    people = pd.read_csv(paths.ROOT / "data" / "adult.csv.gz")
    noise = np.random.default_rng(NOISE_SEED).integers(0, 1000, size=len(people))
    X = np.column_stack([people[COLUMNS].to_numpy(), noise])
    y = (people["income"] == ">50K").to_numpy()
    order = np.random.default_rng(0).permutation(len(y))
    train, test = order[: len(y) // 2], order[len(y) // 2:]
    for a in (X, y, train, test):
        a.flags.writeable = False
    return X, y, train, test


def measure(leaf: int) -> dict:
    """Grows the forest of one stop of the slider and measures it, as MEASURED holds it. Some seconds."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    X, y, train, test = table()
    forest = RandomForestClassifier(min_samples_leaf=leaf, random_state=0).fit(X[train], y[train])
    shuffled = permutation_importance(forest, X[test], y[test], n_repeats=5, random_state=0)
    return {"train": round(float(forest.score(X[train], y[train])), 4),
            "test": round(float(forest.score(X[test], y[test])), 4),
            "mdi": [round(float(v), 3) for v in forest.feature_importances_],
            "perm": [round(float(v), 4) for v in shuffled.importances_mean],
            "sd": [round(float(v), 4) for v in shuffled.importances_std]}


def _show(x: float, lang: str, digits: int = 1) -> str:
    """A number with this many decimals and a real minus sign; a decimal comma in Dutch."""
    text = f"{x:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    text = text.replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _pct(share: float, lang: str) -> str:
    """A share as a percentage with one decimal: 48.2."""
    return _show(share * 100, lang)


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"

_WORDS = {
    "en": {"labels": ["age", "education", "hours", "noise"],
           "own": "the forest's own importance", "own_y": "share of the importance (%)",
           "shuffled": "shuffled on the test half", "shuffled_y": "points of accuracy lost",
           "alt": "Two bar charts over age, education, hours and noise, the column of random numbers, for a forest whose "
                  "leaves hold at least {leaf} {people}. Left, open bars, the forest's own importance: age {m0}%, "
                  "education {m1}%, hours {m2}%, noise {m3}%. Right, filled bars, how many percentage points of "
                  "accuracy the forest loses on the test half when a column is shuffled: age {p0}, education {p1}, "
                  "hours {p2}, noise {p3}.",
           "person": "person", "persons": "people",
           "said": "Leaves of at least {leaf} {people}: the forest is right for {train}% of the training half and "
                   "{test}% of the test half. Its own importance gives the random numbers "
                   "<span class=\"c-loss ea-num\">{m3}%</span>; their permutation importance on the test half is "
                   "{p3} points."},
    "nl": {"labels": ["leeftijd", "opleiding", "uren", "ruis"],
           "own": "de eigen importance van het forest", "own_y": "deel van de importance (%)",
           "shuffled": "gehusseld op de testhelft", "shuffled_y": "procentpunten accuracy kwijt",
           "alt": "Twee staafdiagrammen over leeftijd, opleiding, uren en ruis, de kolom met willekeurige getallen, voor "
                  "een forest waarvan de bladeren minstens {leaf} {people} bevatten. Links, open staven, de eigen "
                  "importance van het forest: leeftijd {m0}%, opleiding {m1}%, uren {m2}%, ruis {m3}%. Rechts, gevulde "
                  "staven, hoeveel procentpunten accuracy het forest op de testhelft kwijtraakt als een kolom door elkaar "
                  "wordt gehusseld: leeftijd {p0}, opleiding {p1}, uren {p2}, ruis {p3}.",
           "person": "persoon", "persons": "personen",
           "said": "Bladeren van minstens {leaf} {people}: het forest zit bij {train}% van de helft om op te trainen goed en bij "
                   "{test}% van de testhelft. Zijn eigen importance geeft de willekeurige getallen "
                   "<span class=\"c-loss ea-num\">{m3}%</span>; hun permutation importance op de testhelft is "
                   "{p3} procentpunt."},
}


def leaf_picture(leaf: int, lang: str) -> tuple[str, str]:
    """For the slider of *The idea*: the forest's own importance and permutation importance on the test half, for the
    forest whose leaves hold at least `leaf` people; and the sentence under them."""
    w = _WORDS[lang]
    m = MEASURED[int(leaf)]
    xs = [0.5, 1.5, 2.5, 3.5]
    own = chart.Panel(x=(0, 4), y=(0, 50), title=w["own"], xlabel="", ylabel=w["own_y"], xticks=xs,
                      xtick_labels=w["labels"], yticks=[0, 10, 20, 30, 40, 50])
    shuffled = chart.Panel(x=(0, 4), y=(-1, 7), title=w["shuffled"], xlabel="", ylabel=w["shuffled_y"], xticks=xs,
                           xtick_labels=w["labels"], yticks=[0, 2, 4, 6])
    for i in range(4):
        own.rect(i + 0.2, 0, i + 0.8, m["mdi"][i] * 100, role="loss")
        shuffled.rect(i + 0.2, 0, i + 0.8, m["perm"][i] * 100, role="loss", solid=True)
    fill = {"leaf": leaf, "people": w["person"] if leaf == 1 else w["persons"], "train": _pct(m["train"], lang),
            "test": _pct(m["test"], lang)}
    for i in range(4):
        fill[f"m{i}"] = _pct(m["mdi"][i], lang)
        fill[f"p{i}"] = _show(m["perm"][i] * 100, lang, 2)
    svg = chart.figure([own, shuffled], alt=w["alt"].format(**fill), width=330, height=250)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    return svg, w["said"].format(**fill)


# --- exercise 1: permutation importance by hand ------------------------------------------------------------------

def model(row) -> bool:
    """The tree of depth 2 from rai-rashomon, as the box writes it: yes for education above 12.5 and age above 33.5."""
    return row[1] > 12.5 and row[0] > 33.5


_ROWS = {}


def _rows() -> dict:
    """The three columns of the test half as Python lists, who earns more, and the order of seed 0 that the picture
    shuffles with."""
    if not _ROWS:
        import numpy as np
        X, y, train, test = table()
        _ROWS["rows"] = X[test][:, :3].tolist()
        _ROWS["labels"] = y[test].tolist()
        _ROWS["order"] = np.random.default_rng(0).permutation(len(test)).tolist()
    return _ROWS


def by_hand(j: int) -> float:
    """What the solution of exercise 1 gives for column j on the test half, with the order of seed 0."""
    r = _rows()
    right = shuffled = 0
    for i, row in enumerate(r["rows"]):
        right += model(row) == r["labels"][i]
        other = list(row)
        other[j] = r["rows"][r["order"][i]][j]
        shuffled += model(other) == r["labels"][i]
    return (right - shuffled) / len(r["rows"])


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


_IMP_WORDS = {
    "en": {"title": "Your importance(), on the tree of depth 2 and the test half",
           "labels": ["age", "education", "hours"],
           "alt": "Three bars: what your importance() gives for each column on the 24,421 people of the test half, for "
                  "the tree that says yes for education above 12.5 and age above 33.5: age {a}, education {b}, hours "
                  "{c} percentage points.",
           "key": "The tree is right for {base}% of the test half. Shuffling age costs it {a} points, education {b}, "
                  "hours {c}: the tree never asks about hours."},
    "nl": {"title": "Jouw importance(), op de boom met diepte 2 en de testhelft",
           "labels": ["leeftijd", "opleiding", "uren"],
           "alt": "Drie staven: wat je importance() voor elke kolom geeft bij de 24.421 mensen van de testhelft, voor de "
                  "boom die ja zegt bij een opleiding boven 12,5 en een leeftijd boven 33,5: leeftijd {a}, opleiding "
                  "{b}, uren {c} procentpunt.",
           "key": "De boom zit bij {base}% van de testhelft goed. Leeftijd door elkaar husselen kost hem {a} "
                  "procentpunt, opleiding {b}, uren {c}: de boom vraagt nooit naar de uren."},
}


def base_right() -> float:
    """How often the tree of depth 2 is right on the test half."""
    r = _rows()
    right = 0
    for row, label in zip(r["rows"], r["labels"]):
        right += model(row) == label
    return right / len(r["rows"])


def _imp_picture(seen) -> str:
    fn = _mine(seen)
    if fn is None:
        return ""
    r = _rows()

    def run():
        out = []
        for j in range(3):
            rows = []
            for row in r["rows"]:
                rows.append(list(row))
            got = _number(fn(rows, list(r["labels"]), j, list(r["order"])))
            if got is None or not -1 <= got <= 1:
                return None
            out.append(got)
        return out

    from everything_ai import check as checking
    try:
        got = checking.with_time_limit(run)
    except Exception:
        return ""
    if got is None:
        return ""
    w = _IMP_WORDS[seen.lang]
    top = max(10.0, max(got) * 110)
    low = min(0.0, min(got) * 110)
    p = chart.Panel(x=(0, 3), y=(low, top), title="", xlabel="", ylabel=_WORDS[seen.lang]["shuffled_y"],
                    xticks=[0.5, 1.5, 2.5], xtick_labels=w["labels"])
    for i, v in enumerate(got):
        p.rect(i + 0.2, 0, i + 0.8, v * 100, role="loss", solid=True)
        p.text(i + 0.5, max(v * 100, 0) + top * 0.04, _show(v * 100, seen.lang, 2), "ink")
    fill = {"a": _show(got[0] * 100, seen.lang, 2), "b": _show(got[1] * 100, seen.lang, 2),
            "c": _show(got[2] * 100, seen.lang, 2), "base": _pct(base_right(), seen.lang)}
    svg = chart.figure([p], alt=w["alt"].format(**fill), width=440, height=240)
    svg = chart.with_decimal_comma(svg) if seen.lang == "nl" else svg
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(w["key"].format(**fill))}</div>')


# --- exercise 2: a tree's impurity importance --------------------------------------------------------------------

_SHARE_WORDS = {
    "en": {"title": "Your share: the impurity importance of the tree of depth 3",
           "labels": ["age", "education", "hours"],
           "alt": "Three open bars: the share of the tree's drop in impurity that each column gets, from your code: age "
                  "{a}%, education {b}%, hours {c}%.",
           "key": "Age {a}%, education {b}%, hours {c}%: education does the most to make the parts purer. A forest's "
                  "feature_importances_ is this share for each of its trees, averaged over the trees. In the worked "
                  "example, over 100 trees with {nodes} nodes in all, it gives the column of random numbers 48.2%."},
    "nl": {"title": "Jouw share: de impurity importance van de boom met diepte 3",
           "labels": ["leeftijd", "opleiding", "uren"],
           "alt": "Drie open staven: het deel van de daling in impurity van de boom dat elke kolom krijgt, uit je code: "
                  "leeftijd {a}%, opleiding {b}%, uren {c}%.",
           "key": "Leeftijd {a}%, opleiding {b}%, uren {c}%: opleiding maakt de delen het meest zuiver. De "
                  "feature_importances_ van een forest is dit deel voor elk van zijn bomen, gemiddeld over de bomen. In "
                  "het uitgewerkte voorbeeld, over 100 bomen met samen {nodes} knopen, geeft het de kolom met "
                  "willekeurige getallen 48,2%."},
}

NODES = 925442   # the nodes of the 100 trees of the forest of the worked example; the test counts them


def _share_picture(seen) -> str:
    mine = seen.mine
    if mine is None:
        return ""
    share = mine.value
    if not isinstance(share, list) or len(share) != 3:
        return ""
    got = [_number(v) for v in share]
    if any(v is None or not 0 <= v <= 1 for v in got) or not math.isclose(sum(got), 1, abs_tol=1e-6):
        return ""
    w = _SHARE_WORDS[seen.lang]
    p = chart.Panel(x=(0, 3), y=(0, 60), title="", xlabel="", ylabel=_WORDS[seen.lang]["own_y"],
                    xticks=[0.5, 1.5, 2.5], xtick_labels=w["labels"], yticks=[0, 20, 40, 60])
    for i, v in enumerate(got):
        p.rect(i + 0.2, 0, i + 0.8, v * 100, role="loss")
        p.text(i + 0.5, v * 100 + 2.4, _pct(v, seen.lang), "ink")
    sep = "." if seen.lang == "nl" else ","
    fill = {"a": _pct(got[0], seen.lang), "b": _pct(got[1], seen.lang), "c": _pct(got[2], seen.lang),
            "nodes": f"{NODES:,}".replace(",", sep)}
    svg = chart.figure([p], alt=w["alt"].format(**fill), width=440, height=240)
    svg = chart.with_decimal_comma(svg) if seen.lang == "nl" else svg
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(w["key"].format(**fill))}</div>')


# The lesson is done when these are correct. "train" and "copy" are choice questions.
REQUIRED = ["importance", "impurity", "train", "copy"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

# --- exercise 1
_I_HEAD = ("def model(row):\n"
           "    # the tree of depth 2 from Many good models: yes for education above 12.5 and age above 33.5\n"
           "    return row[1] > 12.5 and row[0] > 33.5\n"
           "\n"
           "\n"
           "def importance(rows, labels, j, order):\n"
           "    # rows: one list per person, such as [45, 14, 50] for age, education_num and hours_per_week\n"
           "    # labels: True for whoever earns more than 50,000 dollars; j: the column to shuffle\n"
           "    # order: the people in a shuffled order: person i gets column j of person order[i]\n")
_I_START = _I_HEAD + "    ...\n"


def _i(copy: str = "        row = list(rows[i])\n", swap: str = "        row[j] = rows[order[i]][j]\n",
       ret: str = "    return (right - shuffled) / len(rows)\n") -> str:
    """The body of importance with one part changed: how the shuffled row is made, which value goes into column j,
    and what is returned."""
    return (_I_HEAD + "    right = 0\n    shuffled = 0\n    for i in range(len(rows)):\n"
            "        if model(rows[i]) == labels[i]:\n            right = right + 1\n" + copy + swap
            + "        if model(row) == labels[i]:\n            shuffled = shuffled + 1\n" + ret)


_I_SOLUTION = _i()

# Small made-up people: the first is the example in the lesson's text.
_P6 = [[45, 14, 50], [28, 13, 40], [52, 9, 45], [36, 16, 60], [61, 10, 20], [39, 15, 45]]
_P5 = _P6[:5]
_I_VISIBLE = [(_P6, [True, False, False, True, False, True], 1, [1, 0, 3, 4, 5, 2])]
_I_HIDDEN = [
    (_P5, [True, False, False, True, True], 0, [1, 0, 3, 2, 4]),
    (_P5, [True, False, False, True, True], 2, [4, 3, 2, 1, 0]),
    ([[40, 13, 40], [30, 9, 40]], [True, False], 1, [1, 0]),
    ([[40, 13, 40], [30, 15, 40], [50, 8, 50]], [False, True, True], 0, [2, 0, 1]),
    ([[34, 13, 40], [33, 13, 40], [35, 12, 40], [70, 16, 10]], [True, False, False, True], 1, [3, 2, 1, 0]),
    ([[25, 16, 40], [60, 16, 40], [45, 10, 40], [38, 14, 45], [50, 9, 30], [29, 13, 50]],
     [False, True, False, True, False, True], 0, [5, 4, 3, 2, 1, 0]),
    ([[44, 14, 40], [44, 14, 40]], [True, True], 1, [0, 1]),
]

_I_NO_COPY = _i(copy="        row = rows[i]\n")
_I_WHOLE = _i(copy="        row = list(rows[order[i]])\n", swap="")
_I_OWN = _i(swap="        row[j] = rows[i][j]\n")
_I_FIRST = _i(ret="        return (right - shuffled) / len(rows)\n")

# --- exercise 2
_T_HEAD = ("# A tree of depth 3, grown on the training half from age, education_num and hours_per_week (columns 0, 1, 2).\n"
           "# For each node: the column it splits on (-2 for a leaf), the node to its left and to its right (-1 for a\n"
           "# leaf), its Gini impurity, and how many people of the training half reach it. Node 0 is the top.\n")


def _literal(name: str, values: list, width: int = 110) -> str:
    """`name = [v, v, ...]` over as many lines as it needs, none longer than `width`."""
    lines, line = [], f"{name} = ["
    indent = " " * len(line)
    for i, v in enumerate(values):
        piece = f"{v}" + ("]" if i == len(values) - 1 else ",")
        if len(line) + 1 + len(piece) > width and line.strip() not in (f"{name} = [", ""):
            lines.append(line.rstrip())
            line = indent + piece
        else:
            line = line + piece if line.endswith("[") else line + " " + piece
    lines.append(line)
    return "\n".join(lines) + "\n"


_T_DATA = "".join(_literal(k, TREE[k]) for k in ("feature", "left", "right", "impurity", "people"))
_T_TAIL = ("total = gain[0] + gain[1] + gain[2]\nshare = []\nfor c in range(3):\n    share.append(gain[c] / total)\n"
           "share\n")
_T_START = _T_HEAD + _T_DATA + "gain = [0, 0, 0]\nfor n in range(len(feature)):\n    ...\n" + _T_TAIL


def _t(test: str = "    if left[n] != -1:\n", drop: str = "people[n] * impurity[n] - people[a] * impurity[a] - "
       "people[b] * impurity[b]", add: str = "drop", pad: str = "    ") -> str:
    """The loop of exercise 2 with one part changed: which nodes count, what a split takes away, and what is added."""
    return (_T_HEAD + _T_DATA + "gain = [0, 0, 0]\nfor n in range(len(feature)):\n" + test
            + f"{pad}    a = left[n]\n{pad}    b = right[n]\n{pad}    drop = {drop}\n"
            f"{pad}    gain[feature[n]] = gain[feature[n]] + {add}\n" + _T_TAIL)


_T_SOLUTION = _t()
_T_LEAVES = _t(test="", pad="")


EXERCISES = {
    "importance": FunctionBox(
        name="importance",
        picture=_imp_picture,
        function="importance",
        start=_I_START,
        solution=_I_SOLUTION,
        visible=_I_VISIBLE,
        hidden=_I_HIDDEN,
        first=[_DOTS],
        mistakes=[
            Mistake("accuracy", wrong=_i(ret="    return shuffled / len(rows)\n"),
                    en="`{call}` gives {got}, and it should give {want}. Your function gives how often the model is right with the shuffled column. The importance is how much worse it gets: right before shuffling minus right after, divided by the number of people.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie geeft hoe vaak het model goed zit met de door elkaar gehusselde kolom. De importance is hoeveel slechter het wordt: goed vóór het husselen min goed erna, gedeeld door het aantal mensen."),
            Mistake("sign", wrong=_i(ret="    return (shuffled - right) / len(rows)\n"),
                    en="`{call}` gives {got}, and it should give {want}: the same size, the other sign. You subtract the wrong way round. A column the model leans on makes it worse when shuffled, which should give a number above 0: right before minus right after.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: even groot, het andere teken. Je trekt andersom af. Een kolom waar het model op leunt, maakt het slechter als je hem husselt, en dat moet een getal boven 0 geven: goed vóór min goed erna."),
            Mistake("count", wrong=_i(ret="    return right - shuffled\n"),
                    en="`{call}` gives {got}, and it should give {want}. Your function gives how many people more the model gets right, not what share. Divide by the number of people, `len(rows)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie geeft hoeveel mensen meer het model goed heeft, niet welk deel. Deel door het aantal mensen, `len(rows)`."),
            Mistake("no-copy", wrong=_I_NO_COPY,
                    en="`{call}` gives {got}, and it should give {want}. `row = rows[i]` gives no new list but a second name for the same list, so changing `row[j]` changes `rows` too, and later people get values that were already shuffled. Make a copy: `row = list(rows[i])`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. `row = rows[i]` geeft geen nieuwe lijst maar een tweede naam voor dezelfde lijst, dus `row[j]` veranderen verandert ook `rows`, en latere mensen krijgen waarden die al gehusseld waren. Maak een kopie: `row = list(rows[i])`."),
            Mistake("whole-row", wrong=_I_WHOLE,
                    en="`{call}` gives {got}, and it should give {want}. Your shuffled row is the whole row of person `order[i]`. Only column `j` should come from that person: copy the row of person `i`, and replace `row[j]` with `rows[order[i]][j]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je gehusselde rij is de hele rij van persoon `order[i]`. Alleen kolom `j` moet van die persoon komen: kopieer de rij van persoon `i`, en vervang `row[j]` door `rows[order[i]][j]`."),
            Mistake("own", wrong=_I_OWN,
                    en="`{call}` gives {got}, and it should give {want}. Your shuffled row keeps its own value in column `j`, so nothing changes. Take the value of person `order[i]`: `rows[order[i]][j]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je gehusselde rij houdt in kolom `j` zijn eigen waarde, dus er verandert niets. Neem de waarde van persoon `order[i]`: `rows[order[i]][j]`."),
            Mistake("first-person", wrong=_I_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: what you get from the first person alone. If your `return` stands inside the `for` loop over the people, Python leaves the function after the first person.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: wat je krijgt van alleen de eerste persoon. Als je `return` binnen de `for`-loop over de mensen staat, verlaat Python de functie na de eerste persoon."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...` with the body of `importance`: count how many people the model gets right, then how many it gets right when column `j` of each person comes from person `order[i]`, and return the difference as a share.",
                      "nl": "Vervang de `...` door de inhoud van `importance`: tel bij hoeveel mensen het model goed zit, dan bij hoeveel als kolom `j` van elke persoon van persoon `order[i]` komt, en geef het verschil terug als deel."},
            "cleared": _CLEARED,
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Work it out by hand: for each person, does the model get the row right? And with column `j` taken from person `order[i]`? Then right before minus right after, divided by the number of people. Which step is different in your function?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Reken het met de hand uit: zit het model voor elke persoon goed met de rij? En met kolom `j` van persoon `order[i]`? Dan goed vóór min goed erna, gedeeld door het aantal mensen. Welke stap is anders in je functie?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for any number of people, any column `j` and any order, and does it use `model`, `rows`, `labels` and `order` as they are given?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk aantal mensen, elke kolom `j` en elke volgorde, en gebruikt hij `model`, `rows`, `labels` en `order` zoals ze worden meegegeven?"},
            "correct": {"en": "Your `importance` is right in all {n} cases. The picture runs it on the 24,421 people of the test half: shuffling education costs the tree the most, and shuffling hours costs nothing, because the tree never asks about hours.",
                        "nl": "Je `importance` klopt in alle {n} gevallen. De tekening voert hem uit op de 24.421 mensen van de testhelft: opleiding door elkaar husselen kost de boom het meest, en de uren husselen kost niets, want de boom vraagt nooit naar de uren."},
        },
    ),
    "impurity": CodeBox(
        name="impurity",
        picture=_share_picture,
        start=_T_START,
        solution=_T_SOLUTION,
        first=[_DOTS],
        mistakes=[
            Mistake("leaves", wrong=_T_LEAVES,
                    en="Your loop also counts the leaves. A leaf has no split: there `left[n]` is `-1` and `feature[n]` is `-2`, which Python reads as the second place from the end of `gain`. Count only the nodes that split: `if left[n] != -1:`.",
                    nl="Je loop telt ook de bladeren mee. Een blad splitst niet: daar is `left[n]` `-1` en `feature[n]` `-2`, wat Python leest als de tweede plaats van achteren in `gain`. Tel alleen de knopen die splitsen: `if left[n] != -1:`."),
            Mistake("parent", wrong=_t(drop="people[n] * impurity[n]"),
                    en="Your `drop` is the impurity of the node itself, times its people. What a split does is the difference: the node's impurity times its people, minus that of the two nodes below it, `a` and `b`.",
                    nl="Je `drop` is de impurity van de knoop zelf, maal zijn mensen. Wat een splitsing doet, is het verschil: de impurity van de knoop maal zijn mensen, min die van de twee knopen eronder, `a` en `b`."),
            Mistake("unweighted", wrong=_t(drop="impurity[n] - impurity[a] - impurity[b]"),
                    en="Your `drop` subtracts the impurities without the people. A split that makes 24,421 people purer weighs more than one for 1,052: multiply each impurity by how many people reach that node, `people[n] * impurity[n]`.",
                    nl="Je `drop` trekt de impurities af zonder de mensen. Een splitsing die 24.421 mensen zuiverder maakt, weegt zwaarder dan een voor 1.052: vermenigvuldig elke impurity met hoeveel mensen bij die knoop komen, `people[n] * impurity[n]`."),
            Mistake("one-child", wrong=_t(drop="people[n] * impurity[n] - people[a] * impurity[a]"),
                    en="Your `drop` subtracts only the node on the left. A split makes two parts: subtract both, `people[a] * impurity[a]` and `people[b] * impurity[b]`.",
                    nl="Je `drop` trekt alleen de knoop links af. Een splitsing maakt twee delen: trek ze allebei af, `people[a] * impurity[a]` en `people[b] * impurity[b]`."),
            Mistake("splits", wrong=_t(add="1"),
                    en="Your `share` counts how often the tree splits on each column, not how much each split helps. Add the split's `drop` to `gain[feature[n]]`, so a split that makes the parts much purer counts for more.",
                    nl="Je `share` telt hoe vaak de boom op elke kolom splitst, niet hoeveel elke splitsing helpt. Tel de `drop` van de splitsing op bij `gain[feature[n]]`, dan telt een splitsing die de delen veel zuiverder maakt zwaarder."),
            Mistake("column", error=lambda e: isinstance(e, IndexError),
                    en="Your code stops with an `IndexError`: it asks for a place that a list does not have. `gain` has only three places, one per column. The column a node splits on is `feature[n]`: add to `gain[feature[n]]`.",
                    nl="Je code stopt met een `IndexError`: hij vraagt om een plaats die een lijst niet heeft. `gain` heeft maar drie plaatsen, één per kolom. De kolom waarop een knoop splitst, is `feature[n]`: tel op bij `gain[feature[n]]`."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...`: for each node that splits, work out how much purer the split makes the parts, weighed by the people, and add it to `gain` for the column the node splits on.",
                      "nl": "Vervang de `...`: reken voor elke knoop die splitst uit hoeveel zuiverder de splitsing de delen maakt, gewogen naar de mensen, en tel dat op bij `gain` voor de kolom waarop de knoop splitst."},
            "cleared": _CLEARED,
            "no_value": {"en": "The last line of the box shows nothing. Put `share` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `share` op de laatste regel, zodat de check hem kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. For each node with `left[n] != -1`, the drop is `people[n] * impurity[n]` minus the same for the nodes `left[n]` and `right[n]`; it goes to `gain[feature[n]]`. Which step is different in your code?",
                      "nl": "Je laatste regel geeft `{got}`. Voor elke knoop met `left[n] != -1` is de daling `people[n] * impurity[n]` min hetzelfde voor de knopen `left[n]` en `right[n]`; die gaat naar `gain[feature[n]]`. Welke stap is anders in je code?"},
            "correct": {"en": "This is the tree's impurity importance: age 37.2%, education 50.8%, hours 12.0%, the same as its `feature_importances_`. It needs nothing but the tree: the impurities and counts come from the training half while the tree grows. The picture shows your shares.",
                        "nl": "Dit is de impurity importance van de boom: leeftijd 37,2%, opleiding 50,8%, uren 12,0%, hetzelfde als zijn `feature_importances_`. Er is niets anders voor nodig dan de boom: de impurities en de aantallen komen uit de helft om op te trainen, terwijl de boom groeit. De tekening laat je delen zien."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("importance", _I_START, "empty", None),
    ("importance", _I_SOLUTION, "correct", None),
    ("importance", _I_HEAD + "    before = 0\n    after = 0\n    for i in range(len(rows)):\n        row = rows[i][:]\n"
                   "        row[j] = rows[order[i]][j]\n        before += model(rows[i]) == labels[i]\n"
                   "        after += model(row) == labels[i]\n    return before / len(rows) - after / len(rows)\n",
     "correct", None),
    ("importance", _i(ret="    return shuffled / len(rows)\n"), "wrong", "accuracy"),
    ("importance", _i(ret="    return (shuffled - right) / len(rows)\n"), "wrong", "sign"),
    ("importance", _i(ret="    return right - shuffled\n"), "wrong", "count"),
    ("importance", _I_NO_COPY, "wrong", "no-copy"),
    ("importance", _I_WHOLE, "wrong", "whole-row"),
    ("importance", _I_OWN, "wrong", "own"),
    ("importance", _I_FIRST, "wrong", "first-person"),
    ("importance", _I_HEAD + "    return 4 / 6\n", "almost", None),
    ("importance", _i(swap="        row[j] = rows[order[0]][j]\n"), "wrong", None),
    ("importance", _i(ret=""), "wrong", "unreached"),
    ("importance", _I_HEAD + "    ...\n    return 4 / 6\n", "wrong", "dots-left"),
    ("impurity", _T_START, "empty", None),
    ("impurity", _T_SOLUTION, "correct", None),
    ("impurity", _T_HEAD + _T_DATA + "gain = [0, 0, 0]\nfor n in range(len(feature)):\n    if feature[n] >= 0:\n"
                 "        kids = people[left[n]] * impurity[left[n]] + people[right[n]] * impurity[right[n]]\n"
                 "        gain[feature[n]] += people[n] * impurity[n] - kids\n" + _T_TAIL, "correct", None),
    ("impurity", _T_LEAVES, "wrong", "leaves"),
    ("impurity", _t(drop="people[n] * impurity[n]"), "wrong", "parent"),
    ("impurity", _t(drop="impurity[n] - impurity[a] - impurity[b]"), "wrong", "unweighted"),
    ("impurity", _t(drop="people[n] * impurity[n] - people[a] * impurity[a]"), "wrong", "one-child"),
    ("impurity", _t(add="1"), "wrong", "splits"),
    ("impurity", _t().replace("gain[feature[n]] = gain[feature[n]]", "gain[n] = gain[n]"), "error", "column"),
    ("impurity", _t(drop="people[n] * impurity[n] - people[b] * impurity[b]"), "wrong", None),
    ("impurity", _T_SOLUTION.replace("    if left[n] != -1:\n", "    if left[n] != -1:\n        ...\n"), "wrong",
     "dots-left"),
]
