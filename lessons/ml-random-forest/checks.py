# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-random-forest: a random forest on the Palmer penguins, built from scikit-learn's trees the
way the lesson builds it. A bootstrap sample of the training penguins, the vote of the trees, and why each split looks
at a few random features. And the picture: the training penguins by bill length and bill depth with the decision
boundary of the vote of the first n trees, and the validation mistakes of ten such forests for every n."""
import functools
import re

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake

DATA = "data/penguins.csv"   # the Palmer penguins (CC0): see data/README.md
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]   # in the order np.unique and scikit-learn sort them
N_TRAIN, N_VAL = 206, 68     # of the 342 penguins with all measurements, split with seed 0 as in ml-project-tabular
TWO = ["bill_length_mm", "bill_depth_mm"]
FOUR = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
SIZES = [1, 2, 3, 5, 10, 20, 50, 100]   # the stops of the slider: how many trees vote
FORESTS = 10                 # forests with their own random draws, for the spread in the picture
TREES = 100                  # the trees of each forest; a forest of n trees is the first n of them
EXAMPLE_TREES = 5            # the forest of the worked example: the first 5 trees of forest 0
TICKS = [1, 2, 5, 10, 20, 50, 100]


@functools.lru_cache(maxsize=1)
def split():
    """The training and the validation penguins as arrays of bill length and bill depth, and their species, as in the
    worked example: seed 0, then 206 and 68 penguins. Read-only, so code that is given them cannot change them. The
    species also as numbers, 0 to 2 in the order of SPECIES, for the course's own trees: scikit-learn sorts the species
    in that same order, so a tree learns the same from the numbers as from the names, only faster."""
    import numpy as np
    import pandas as pd
    from everything_ai import paths
    df = pd.read_csv(paths.ROOT / DATA)
    df = df[df["body_mass_g"].isna() == False]   # noqa: E712 (the lesson's own way, with what the learner knows)
    order = np.random.default_rng(0).permutation(len(df))
    train, val = df.iloc[order[:N_TRAIN]], df.iloc[order[N_TRAIN:N_TRAIN + N_VAL]]
    out = {"X_train": np.array(train[TWO]), "y_train": np.array(train["species"]),
           "X_val": np.array(val[TWO]), "y_val": np.array(val["species"]), "X4_train": np.array(train[FOUR])}
    out["c_train"] = np.array([SPECIES.index(v) for v in out["y_train"]])
    out["c_val"] = np.array([SPECIES.index(v) for v in out["y_val"]])
    for a in out.values():
        a.flags.writeable = False
    return out


@functools.lru_cache(maxsize=FORESTS * 2)
def forest(seed: int = 0, max_features=1, features: str = "two") -> tuple:
    """The trees of one forest, as the worked example grows them: a generator with this seed draws each tree's bootstrap
    sample, and tree i decides its random features with random_state 100 * seed + i. For each tree: the positions of
    its bootstrap sample and the tree itself, which says a species as its number in SPECIES."""
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    s = split()
    X = s["X_train"] if features == "two" else s["X4_train"]
    rng = np.random.default_rng(seed)
    out = []
    for i in range(TREES):
        rows = rng.integers(0, N_TRAIN, size=N_TRAIN)
        tree = DecisionTreeClassifier(max_features=max_features, random_state=100 * seed + i)
        out.append((rows, tree.fit(X[rows], s["c_train"][rows])))
    return tuple(out)


def vote(votes):
    """The course's version of exercise 2: for each column, one penguin, the species most rows, the trees, say; a tie
    goes to the first in alphabetical order, as np.unique sorts them."""
    import numpy as np
    said = []
    for j in range(votes.shape[1]):
        names, counts = np.unique(votes[:, j], return_counts=True)
        said.append(names[np.argmax(counts)])
    return np.array(said)


@functools.lru_cache(maxsize=FORESTS * 2)
def votes(seed: int = 0, max_features=1):
    """What each tree of the forest says for each validation penguin, as species: one row per tree."""
    import numpy as np
    codes = np.array([tree.predict(split()["X_val"]) for _, tree in forest(seed, max_features)])
    return np.array(SPECIES)[codes]


@functools.lru_cache(maxsize=FORESTS * 2)
def _counts(seed: int, max_features):
    """counts[n - 1, j, c]: how many of the first n trees say species c for validation penguin j."""
    import numpy as np
    codes = np.array([tree.predict(split()["X_val"]) for _, tree in forest(seed, max_features)])
    return np.cumsum(np.eye(len(SPECIES), dtype=int)[codes], axis=0)


def tree_mistakes(seed: int = 0, max_features=1) -> list:
    """How many validation penguins each single tree of a forest gets wrong; with max_features None, the trees of
    bagging, which may look at both measurements in every split."""
    import numpy as np
    y = split()["y_val"]
    return [int(np.sum(v != y)) for v in votes(seed, max_features)]


def mistakes(seed: int = 0, n: int = TREES, max_features=1) -> int:
    """How many validation penguins the vote of the first n trees of a forest gets wrong; a tie goes to the first
    species, as in vote."""
    import numpy as np
    return int(np.sum(np.argmax(_counts(seed, max_features)[n - 1], axis=1) != split()["c_val"]))


def spread(n: int, max_features=1) -> list:
    """The validation mistakes of the ten forests with n trees each. Growing all ten takes a few seconds, so the picture
    uses MEASURED, which the tests compare with this."""
    return [mistakes(seed, n, max_features) for seed in range(FORESTS)]


# spread(n) for every n of the slider, forests 0 to 9 in that order; tests/test_ml_random_forest.py works it out again.
MEASURED = {1: [8, 6, 2, 5, 4, 7, 4, 3, 8, 3], 2: [5, 3, 2, 5, 4, 7, 4, 2, 5, 3], 3: [6, 3, 3, 3, 3, 5, 2, 4, 4, 3],
            5: [3, 3, 3, 3, 3, 2, 2, 4, 3, 3], 10: [3, 3, 4, 3, 3, 3, 3, 4, 3, 2], 20: [3, 4, 3, 3, 3, 3, 4, 3, 3, 3],
            50: [3, 3, 3, 3, 3, 3, 4, 3, 3, 3], 100: [3] * 10}


def variance(values) -> float:
    """The variance of a list of numbers, each as likely: their mean squared distance from their mean."""
    m = sum(values) / len(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def distinct(seed: int = 0, i: int = 0) -> int:
    """How many different penguins the bootstrap sample of tree i of a forest holds."""
    import numpy as np
    return len(np.unique(forest(seed)[i][0]))


def roots(max_features, seed: int = 0) -> dict:
    """On all four measurements: how many of the 100 trees ask their first question about each measurement."""
    out = {}
    for _, tree in forest(seed, max_features, "four"):
        name = FOUR[tree.tree_.feature[0]]
        out[name] = out.get(name, 0) + 1
    return out


def sklearn_mistakes() -> int:
    """The validation mistakes of scikit-learn's RandomForestClassifier with 100 trees, as the worked example makes it."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    s = split()
    f = RandomForestClassifier(n_estimators=TREES, max_features=1, random_state=0).fit(s["X_train"], s["y_train"])
    return int(np.sum(f.predict(s["X_val"]) != s["y_val"]))


# --- the picture ------------------------------------------------------------------------------------------------------

X_RANGE, Y_RANGE = (31.0, 61.0), (12.5, 22.5)
_NX, _NY = 80, 80


@functools.lru_cache(maxsize=1)
def _grid_votes():
    """For every cell of an 80 by 80 grid, how many of the first n trees of forest 0 say each species, for every n."""
    import numpy as np
    xe, ye = np.linspace(*X_RANGE, _NX + 1), np.linspace(*Y_RANGE, _NY + 1)
    gx, gy = np.meshgrid((xe[1:] + xe[:-1]) / 2, (ye[1:] + ye[:-1]) / 2)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    said = np.array([tree.predict(grid) for _, tree in forest(0)])   # (trees, cells), each a species as its number
    counts = np.cumsum(np.eye(len(SPECIES), dtype=int)[said], axis=0)   # counts[n - 1, cell, c]
    return xe, ye, counts


@functools.lru_cache(maxsize=len(SIZES))
def regions(n: int):
    """The species the vote of the first n trees says in every cell: rows bottom to top, columns left to right."""
    import numpy as np
    xe, ye, counts = _grid_votes()
    return xe, ye, np.argmax(counts[n - 1], axis=1).reshape(_NY, _NX)   # the first of equal counts, as in vote


def boundary(xe, ye, r) -> list:
    """The edges between cells where the species changes, joined into straight runs: (x0, y0, x1, y1)."""
    nx, ny = len(xe) - 1, len(ye) - 1
    lines = []
    for i in range(nx - 1):   # a vertical piece between columns i and i + 1, row after row
        start = None
        for j in range(ny + 1):
            edge = j < ny and r[j, i] != r[j, i + 1]
            if edge and start is None:
                start = j
            if not edge and start is not None:
                lines.append((xe[i + 1], ye[start], xe[i + 1], ye[j]))
                start = None
    for j in range(ny - 1):   # a horizontal piece between rows j and j + 1, column after column
        start = None
        for i in range(nx + 1):
            edge = i < nx and r[j, i] != r[j + 1, i]
            if edge and start is None:
                start = i
            if not edge and start is not None:
                lines.append((xe[start], ye[j + 1], xe[i], ye[j + 1]))
                start = None
    return lines


def picture(n: int, labels: dict[str, str]) -> str:
    """Left: the training penguins by bill length and bill depth, a colour and shape per species (style guide 1.7; the
    text before the picture says which shape is which), and in blue the decision boundary of the vote of the first n
    trees of forest 0. Right: for every number of trees, the validation mistakes of the ten forests, a line from the
    fewest to the most and a dot for each, filled for forest 0; this n marked. Drawn as SVG text, once for every stop of
    the slider (L.explore); dragging across the right chart sets n."""
    from everything_ai import chart
    s = split()
    p = chart.Panel(X_RANGE, Y_RANGE, title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=[35, 40, 45, 50, 55, 60], yticks=[14, 16, 18, 20, 22])
    for x0, y0, x1, y1 in boundary(*regions(n)):
        p.line([x0, x1], [y0, y1], "pred", 2.0)
    for c, name in enumerate(SPECIES):
        rows = s["X_train"][s["y_train"] == name]
        p.group(rows[:, 0], rows[:, 1], c, r=3.6)
    e = chart.Panel((0.8, 130), (0, 10), title=labels["every"], xlabel=labels["trees"], ylabel=labels["mistakes"],
                    xticks=TICKS, yticks=[0, 2, 4, 6, 8, 10], xlog=True)
    e.line([n, n], [0, 10], "ink-3", 1.2)
    for size in SIZES:
        m = MEASURED[size]
        e.line([size, size], [min(m), max(m)], "loss", 1.6)
        e.points([size] * len(m), m, "loss", 3.0, hollow=True)
    e.points(SIZES, [mistakes(0, size) for size in SIZES], "loss", 3.4)
    e.points([n], [mistakes(0, n)], "loss", 6.0)
    return chart.figure([p, e], alt=labels["alt"], width=330, height=300, drag=1)


# The lesson is done when these are correct. "random-features" is the choice question in the lesson.
REQUIRED = ["bootstrap", "vote", "random-features"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


# --- exercise 1: a bootstrap sample -----------------------------------------------------------------------------------

_B_HEAD = "import numpy as np\n\nrng = np.random.default_rng(1)\nn = 206\n"
_B_START = _B_HEAD + "rows = ...\nrows\n"
_B = _B_HEAD + "rows = rng.integers(0, n, size=n)\nrows"


def _b(line: str) -> str:
    return _B_HEAD + f"rows = {line}\nrows"


def _numbers(got):
    """The positions as a list of Python numbers, or None when it is not a row of whole numbers."""
    import numpy as np
    try:
        a = np.asarray(got)
    except Exception:
        return None
    if a.ndim != 1 or a.dtype.kind not in "iu":
        return None
    return [int(v) for v in a]


def _is_bootstrap(got, want) -> bool:
    """A bootstrap sample of n positions: n whole numbers, each from 0 to n - 1, with at least one twice. Random numbers
    may come out otherwise than the course's, so the check looks at what makes a bootstrap sample, not at the numbers."""
    rows, n = _numbers(got), len(want)
    return (rows is not None and len(rows) == n and all(0 <= v < n for v in rows) and 1 < len(set(rows)) < n
            and not _never_last(rows) and not _never_first(rows))


def _drawn_like(rows, low: int, high: int) -> bool:
    """Whether rows are exactly what the box's generator, seed 1, gives with rng.integers(low, high, size=n). One
    sample cannot show that a position is never drawn, but the same draw with the same seed gives the same numbers."""
    import numpy as np
    return len(rows) > 1 and rows == np.random.default_rng(1).integers(low, high, size=len(rows)).tolist()


def _never_last(rows) -> bool:
    """rng.integers(0, n - 1, size=n), or rng.choice(n - 1, size=n), which draws the same: never position n - 1."""
    return _drawn_like(rows, 0, len(rows) - 1)


def _never_first(rows) -> bool:
    """rng.integers(1, n, size=n): never position 0."""
    return _drawn_like(rows, 1, len(rows))


def _has(prop):
    return lambda got, text: (lambda rows: rows is not None and prop(rows))(_numbers(got))


EXERCISES = {
    "bootstrap": CodeBox(
        name="bootstrap",
        start=_B_START,
        solution=_B,
        compare=_is_bootstrap,
        keep=["n = 206"],
        # The check also runs the box with another n: a number typed in by hand does not change with it.
        others=[{"n": 50}],
        first=[_DOTS],
        mistakes=[
            Mistake("one-number", matches=lambda got, text: _numbers([got]) is not None and not hasattr(got, "__len__"),
                    en="Your last line gives `{got}`: one number. `rng.integers` gives as many numbers as `size=` says; a bootstrap sample has `n` of them.",
                    nl="Je laatste regel geeft `{got}`: één getal. `rng.integers` geeft zoveel getallen als `size=` zegt; een bootstrap-steekproef heeft er `n`."),
            Mistake("floats", matches=lambda got, text: hasattr(got, "dtype") and got.dtype.kind == "f",
                    en="Your positions are numbers with a decimal point. A position in an array is a whole number: `rng.integers(0, n, size=n)` gives whole numbers from 0 to `n - 1`.",
                    nl="Je posities zijn getallen met een komma. Een positie in een array is een heel getal: `rng.integers(0, n, size=n)` geeft hele getallen van 0 tot en met `n - 1`."),
            Mistake("no-replacement", matches=_has(lambda rows: len(set(rows)) == len(rows) > 1),
                    en="Every position in your `rows` comes exactly once: that is drawing without replacement, the same penguins in another order, and a tree on them learns the same as on the whole training set. Draw with replacement, so a penguin can come twice: `rng.integers`.",
                    nl="Elke positie in je `rows` komt precies één keer voor: dat is trekken zonder teruglegging, dezelfde pinguïns in een andere volgorde, en een boom leert daarvan hetzelfde als van de hele training set. Trek met teruglegging, zodat een pinguïn twee keer kan komen: `rng.integers`."),
            Mistake("from-one", matches=_has(lambda rows: max(rows) == len(rows)),
                    en="Your `rows` holds the position `n` itself, and the positions of 206 penguins go from 0 to 205. `rng.integers(0, n, size=n)` stops just below `n`.",
                    nl="Je `rows` bevat de positie `n` zelf, en de posities van 206 pinguïns lopen van 0 tot en met 205. `rng.integers(0, n, size=n)` stopt net onder `n`."),
            Mistake("never-last", matches=_has(_never_last),
                    en="Your `rows` can never hold the position `n - 1`, 205: the last penguin is never drawn, because you draw up to, but not including, `n - 1`. For the positions 0 to `n - 1`, draw up to `n`: `rng.integers(0, n, size=n)`.",
                    nl="Je `rows` kan nooit de positie `n - 1`, 205, bevatten: de laatste pinguïn wordt nooit getrokken, want je trekt tot `n - 1`, en `n - 1` zelf doet niet mee. Voor de posities 0 tot en met `n - 1` trek je tot `n`: `rng.integers(0, n, size=n)`."),
            Mistake("never-first", matches=_has(_never_first),
                    en="Your `rows` can never hold the position 0: the first penguin is never drawn. Positions count from 0, so draw from 0: `rng.integers(0, n, size=n)`.",
                    nl="Je `rows` kan nooit de positie 0 bevatten: de eerste pinguïn wordt nooit getrokken. Posities tellen vanaf 0, dus trek vanaf 0: `rng.integers(0, n, size=n)`."),
            Mistake("size", matches=_has(lambda rows: len(rows) != 206),
                    en="Your `rows` holds {count} positions. A bootstrap sample is as large as the training set: `size=n`.",
                    nl="Je `rows` bevat {count} posities. Een bootstrap-steekproef is even groot als de training set: `size=n`.",
                    detail=lambda got, text: {"count": len(_numbers(got))}),
            Mistake("all-same", matches=_has(lambda rows: len(set(rows)) == 1),
                    en="Every position in your `rows` is the same, so the sample is one penguin, many times. Draw each position at random: `rng.integers(0, n, size=n)`.",
                    nl="Elke positie in je `rows` is dezelfde, dus de steekproef is één pinguïn, heel vaak. Trek elke positie willekeurig: `rng.integers(0, n, size=n)`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with `n` positions from 0 to `n - 1`, drawn with replacement with `rng`.",
                      "nl": "Vervang de `...` door `n` posities van 0 tot en met `n - 1`, met teruglegging getrokken met `rng`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another n by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere n."},
            "no_value": {"en": "The last line of the box shows nothing. Put `rows` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `rows` op de laatste regel, zodat de check hem kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. A bootstrap sample is `n` whole numbers from 0 to `n - 1`, drawn with replacement. Do you draw them with `rng.integers`?",
                      "nl": "Je laatste regel geeft `{got}`. Een bootstrap-steekproef is `n` hele getallen van 0 tot en met `n - 1`, met teruglegging getrokken. Trek je ze met `rng.integers`?"},
            "other": {"en": "Right for this n. But with {where}, your `rows` is not a bootstrap sample of that many penguins. Do you use `n` in your line, instead of 206?",
                      "nl": "Goed voor deze n. Maar met {where} is je `rows` geen bootstrap-steekproef van zoveel pinguïns. Gebruik je `n` in je regel, in plaats van 206?"},
            "other_error": {"en": "Right for this n. But with {where}, your code stops with an error. Do you use `n` in your line, instead of 206?",
                            "nl": "Goed voor deze n. Maar met {where} stopt je code met een fout. Gebruik je `n` in je regel, in plaats van 206?"},
            "correct": {"en": "A bootstrap sample: 206 positions from 0 to 205, some more than once, so others are missing. On average a bootstrap sample holds about 63% of the penguins. The check also tried another n.",
                        "nl": "Een bootstrap-steekproef: 206 posities van 0 tot en met 205, sommige vaker dan één keer, dus andere ontbreken. Gemiddeld bevat een bootstrap-steekproef ongeveer 63% van de pinguïns. De check probeerde ook een andere n."},
        },
    ),
}


# --- exercise 2: the vote ---------------------------------------------------------------------------------------------

class _Named:
    """A test array that the check's messages write by its name, such as `votes`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import numpy as np
        return np.array(self.value)


_A, _C, _G = "Adelie", "Chinstrap", "Gentoo"
# Five trees, one row each, and four penguins, one column each: no tie in any column, and in each column another species
# wins than in the row of the same number, so a vote per tree instead of per penguin gives something else.
_V = _Named("votes", [[_A, _G, _C, _G], [_C, _G, _C, _A], [_A, _C, _G, _A], [_A, _G, _C, _C], [_G, _G, _C, _A]])
# Three trees and six penguins: more penguins than trees.
_V2 = _Named("votes2", [[_G, _A, _A, _C, _G, _A], [_G, _A, _C, _C, _A, _A], [_C, _C, _C, _G, _G, _A]])
# Seven trees and two penguins, with a narrow majority of 4 against 3.
_V3 = _Named("votes3", [[_A, _C], [_C, _C], [_A, _G], [_C, _C], [_A, _G], [_A, _G], [_C, _C]])


def _shown(v) -> str:
    """What a function gave back, as the lesson writes it: a species as "Adelie", several as a list, a number as it is."""
    import numpy as np
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_shown(x) for x in v) + "]"
    return str(v)


_V_HEAD = "import numpy as np\n\n\ndef vote(votes):\n    said = []\n    for j in range(votes.shape[1]):\n"
_V_START = _V_HEAD + "        names, counts = np.unique(..., return_counts=True)\n        said.append(...)\n    return np.array(said)\n"
_V_SOL = _V_HEAD + "        names, counts = np.unique(votes[:, j], return_counts=True)\n        said.append(names[np.argmax(counts)])\n    return np.array(said)\n"


def _v(column: str = "votes[:, j]", winner: str = "names[np.argmax(counts)]", back: str = "np.array(said)") -> str:
    return (_V_HEAD + f"        names, counts = np.unique({column}, return_counts=True)\n        said.append({winner})\n"
            f"    return {back}\n")


EXERCISES["vote"] = FunctionBox(
    name="vote",
    function="vote",
    start=_V_START,
    solution=_V_SOL,
    visible=[(_V,)],
    show=_shown,
    hidden=[(_V2,), (_V3,)],
    first=[_DOTS],
    mistakes=[
        Mistake("per-tree", wrong=_v(column="votes[j]"),
                en="`{call}` gives `{got}`, and it should give `{want}`. `votes[j]` is row `j`: what one tree says for every penguin. The votes for penguin `j` are in column `j`: `votes[:, j]`.",
                nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. `votes[j]` is rij `j`: wat één boom voor elke pinguïn zegt. De stemmen voor pinguïn `j` staan in kolom `j`: `votes[:, j]`."),
        Mistake("whole-array", wrong=_v(column="votes"),
                en="`{call}` gives `{got}`: the same species for every penguin. `np.unique(votes)` counts the votes of the whole table; count those of penguin `j` only: `votes[:, j]`.",
                nl="`{call}` geeft `{got}`: dezelfde soort voor elke pinguïn. `np.unique(votes)` telt de stemmen van de hele tabel; tel alleen die van pinguïn `j`: `votes[:, j]`."),
        Mistake("fewest", wrong=_v(winner="names[np.argmin(counts)]"),
                en="`{call}` gives `{got}`, and it should give `{want}`. `np.argmin` picks the species with the fewest votes; the vote goes to the most: `np.argmax`.",
                nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. `np.argmin` kiest de soort met de minste stemmen; de stemming gaat naar de meeste: `np.argmax`."),
        Mistake("count", wrong=_v(winner="np.max(counts)"),
                en="`{call}` gives `{got}`: how many votes each winner got, not which species it is. Look up the species at the position of the most votes in `names`.",
                nl="`{call}` geeft `{got}`: hoeveel stemmen elke winnaar kreeg, niet welke soort het is. Zoek de soort op de positie van de meeste stemmen op in `names`."),
        Mistake("position", wrong=_v(winner="np.argmax(counts)"),
                en="`{call}` gives `{got}`: the position of each winner in `names`, counting from 0, not the species itself. Look it up: `names[...]`.",
                nl="`{call}` geeft `{got}`: de positie van elke winnaar in `names`, geteld vanaf 0, niet de soort zelf. Zoek hem op: `names[...]`."),
        Mistake("first-name", wrong=_v(winner="names[0]"),
                en="`{call}` gives `{got}`, and it should give `{want}`. `names[0]` is the first species in alphabetical order among the votes, not the one with the most. Which position in `counts` holds the most?",
                nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. `names[0]` is de eerste soort op alfabet onder de stemmen, niet die met de meeste. Op welke positie in `counts` staan de meeste?"),
        Mistake("first-tree", wrong=_v(back="votes[0]"),
                en="`{call}` gives `{got}`: what the first tree says. The forest's answer is the vote of all trees, `said`.",
                nl="`{call}` geeft `{got}`: wat de eerste boom zegt. Het antwoord van het forest is de stemming van alle bomen, `said`."),
    ],
    texts={
        "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                    "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
        "empty": {"en": "Replace the two `...`: the votes for penguin `j`, a column of `votes`, and the species with the most of them.",
                  "nl": "Vervang de twee `...`: de stemmen voor pinguïn `j`, een kolom van `votes`, en de soort met de meeste daarvan."},
        "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                    "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
        "arguments": {"en": "The check calls `{call}`, with one value, and the `def` line of your function has a different number of names between its parentheses. Keep `(votes)`.",
                      "nl": "De check roept `{call}` aan, met één waarde, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(votes)`."},
        "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you count the votes of column `j`, `votes[:, j]`, and add the species with the most to `said`?",
                  "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Tel je de stemmen van kolom `j`, `votes[:, j]`, en voeg je de soort met de meeste toe aan `said`?"},
        "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use the `votes` it gets, every column of it?",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie de `votes` die hij krijgt, elke kolom ervan?"},
        "correct": {"en": "Your function is right in all {n} cases. In the example no tree says what the vote says for all four penguins, and still the vote follows most of them.",
                    "nl": "Je functie klopt in alle {n} gevallen. In het voorbeeld zegt geen enkele boom voor alle vier de pinguïns wat de stemming zegt, en toch volgt de stemming de meeste bomen."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("bootstrap", _B_START, "empty", None),
    ("bootstrap", _B, "correct", None),
    ("bootstrap", _b("rng.integers(n, size=n)"), "correct", None),
    ("bootstrap", _b("rng.choice(n, size=n)"), "correct", None),
    ("bootstrap", _b("rng.choice(np.arange(n), size=n, replace=True)"), "correct", None),
    ("bootstrap", _b("np.array([rng.integers(0, n) for i in range(n)])"), "correct", None),
    ("bootstrap", _b("rng.permutation(n)"), "wrong", "no-replacement"),
    ("bootstrap", _b("np.arange(n)"), "wrong", "no-replacement"),
    ("bootstrap", _b("rng.choice(n, size=n, replace=False)"), "wrong", "no-replacement"),
    ("bootstrap", _b("rng.integers(0, n)"), "wrong", "one-number"),
    ("bootstrap", _b("rng.random(n) * n"), "wrong", "floats"),
    ("bootstrap", _b("rng.integers(1, n + 1, size=n)"), "wrong", "from-one"),
    ("bootstrap", _b("rng.integers(0, n - 1, size=n)"), "wrong", "never-last"),
    ("bootstrap", _b("rng.choice(n - 1, size=n)"), "wrong", "never-last"),
    ("bootstrap", _b("rng.integers(1, n, size=n)"), "wrong", "never-first"),
    ("bootstrap", _b("rng.integers(0, n, size=100)"), "wrong", "size"),
    ("bootstrap", _b("np.zeros(n, dtype=int)"), "wrong", "all-same"),
    ("bootstrap", _b("rng.integers(0, 206, size=206)"), "almost", "other-value"),
    ("bootstrap", _B_HEAD + "rows = rng.integers(0, n, size=n)\nprint(rows)\n", "wrong", "print-last"),
    ("vote", _V_START, "empty", None),
    ("vote", _V_SOL, "correct", None),
    ("vote", _v(winner="names[counts.argmax()]"), "correct", None),
    ("vote", _v(back="said"), "correct", None),
    ("vote", _V_HEAD + "        column = votes[:, j]\n        names, counts = np.unique(column, return_counts=True)\n        best = np.argmax(counts)\n"
             "        said.append(names[best])\n    return np.array(said)\n", "correct", None),
    ("vote", _v(column="votes[j]"), "wrong", "per-tree"),
    ("vote", _v(column="votes"), "wrong", "whole-array"),
    ("vote", _v(winner="names[np.argmin(counts)]"), "wrong", "fewest"),
    ("vote", _v(winner="np.max(counts)"), "wrong", "count"),
    ("vote", _v(winner="np.argmax(counts)"), "wrong", "position"),
    ("vote", _v(winner="names[0]"), "wrong", "first-name"),
    ("vote", _v(back="votes[0]"), "wrong", "first-tree"),
    ("vote", _V_SOL.replace("range(votes.shape[1])", "range(4)"), "almost", None),
    ("vote", _V_HEAD + "        names, counts = np.unique(votes[:, j], return_counts=True)\n        said.append(names[np.argmax(counts)])\n"
             "    print(np.array(said))\n", "wrong", "prints"),
]
