# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-decision-tree: a decision tree on the Palmer penguins. The mistakes of one split, the
maximum depth chosen on the validation set, and reading the tree of the worked example. And the picture: the
training penguins by bill length and body mass with the decision boundary of a tree of each depth, and its mistakes
on the training and the validation penguins for every depth."""
import functools
import re

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake

DATA = "data/penguins.csv"   # the Palmer penguins (CC0): see data/README.md
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]   # in the order np.unique and scikit-learn sort them
N_TRAIN, N_VAL = 206, 68     # of the 342 penguins with all measurements, split with seed 0 as in ml-project-tabular
FOUR = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
TWO = ["bill_length_mm", "body_mass_g"]   # the picture: the trees on these two do not depend on random_state
DEPTHS = [1, 2, 3, 4, 5, 6, 7]            # the slider; at 7 every part of the picture's tree is pure
EX_DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8]      # exercise 2
FIRST_T = 45                              # the one split the worked example counts first: bill length at most 45 mm


@functools.lru_cache(maxsize=1)
def split():
    """The training and the validation set, as in the worked example and in ml-project-tabular: seed 0, then 206 and
    68 penguins; the 68 after them are the test set, which stays out of sight."""
    import numpy as np
    import pandas as pd
    from everything_ai import paths
    df = pd.read_csv(paths.ROOT / DATA)
    df = df[df["body_mass_g"].isna() == False]   # noqa: E712 (the lesson's own way, with what the learner knows)
    order = np.random.default_rng(0).permutation(len(df))
    return df.iloc[order[:N_TRAIN]], df.iloc[order[N_TRAIN:N_TRAIN + N_VAL]]


@functools.lru_cache(maxsize=64)
def tree(features: tuple = tuple(TWO), depth: int | None = None):
    """scikit-learn's tree on these measurements of the training penguins, to this maximum depth, with random_state 0
    as in the lesson: when two questions split equally well, it decides which one it asks."""
    from sklearn.tree import DecisionTreeClassifier
    train, _ = split()
    return DecisionTreeClassifier(max_depth=depth, random_state=0).fit(train[list(features)], train["species"])


def mistakes(features: tuple = tuple(TWO), depth: int | None = None, on: str = "val") -> int:
    """How many training ("train") or validation ("val") penguins that tree puts in the wrong species."""
    import numpy as np
    rows = split()[0 if on == "train" else 1]
    return int(np.sum(tree(tuple(features), depth).predict(rows[list(features)]) != rows["species"]))


def curve(features: tuple = tuple(TWO), depths: tuple = tuple(DEPTHS)) -> tuple:
    """The training and the validation mistakes for every depth: the second chart of the picture."""
    return (tuple(mistakes(features, d, "train") for d in depths), tuple(mistakes(features, d, "val") for d in depths))


def best_depth(features: tuple = tuple(FOUR), depths: tuple = tuple(EX_DEPTHS)) -> int:
    """The depth with the fewest validation mistakes; the first of them, as np.argmin takes it."""
    m = [mistakes(features, d) for d in depths]
    return depths[m.index(min(m))]


def leaves(depth: int, features: tuple = tuple(TWO)) -> int:
    return int(tree(tuple(features), depth).get_n_leaves())


def depth_reached(features: tuple = tuple(TWO)) -> int:
    """How deep the tree grows without a limit: until every part is pure."""
    return int(tree(tuple(features), None).get_depth())


# --- by hand, as the worked example does it ---------------------------------------------------------------------------

def split_mistakes(x, y, t):
    """The course's version of exercise 1: the mistakes of the split x <= t when each part says its most common class."""
    import numpy as np
    wrong = 0
    for part in [y[x <= t], y[x > t]]:
        names, counts = np.unique(part, return_counts=True)
        wrong = wrong + len(part) - max(counts)
    return wrong


def parts(feature: str = "bill_length_mm", t: float = FIRST_T) -> list:
    """The first loop of the worked example: for each part, its species, how many of each, and its mistakes."""
    import numpy as np
    train, _ = split()
    x, y = np.array(train[feature]), np.array(train["species"])
    out = []
    for part in [y[x <= t], y[x > t]]:
        names, counts = np.unique(part, return_counts=True)
        out.append((list(names), [int(c) for c in counts], int(len(part) - max(counts))))
    return out


def best_split(rows=None, features: tuple = tuple(FOUR)) -> list:
    """The second loop of the worked example: for every measurement, the first threshold with the fewest mistakes, and those
    mistakes; the thresholds are the measured values but the largest, from small to large."""
    import numpy as np
    rows = split()[0] if rows is None else rows
    y = np.array(rows["species"])
    out = []
    for f in features:
        x = np.array(rows[f])
        best, fewest = None, len(y)
        for t in np.unique(x)[:-1]:
            m = split_mistakes(x, y, t)
            if m < fewest:
                best, fewest = float(t), int(m)
        out.append((f, best, fewest))
    return out


# --- the picture ------------------------------------------------------------------------------------------------------

X_RANGE, Y_RANGE = (31.0, 61.0), (2500.0, 6500.0)


def regions(depth: int):
    """The tree's species in every cell between its own thresholds, so the boundary lies exactly on them: the edges of
    the cells left to right and bottom to top, and the species, rows bottom to top, columns left to right."""
    import numpy as np
    t = tree(tuple(TWO), depth)
    cut = t.tree_.threshold[t.tree_.feature >= 0]
    on = t.tree_.feature[t.tree_.feature >= 0]
    xe = np.array([X_RANGE[0], *sorted({v for v, f in zip(cut, on) if f == 0 and X_RANGE[0] < v < X_RANGE[1]}), X_RANGE[1]])
    ye = np.array([Y_RANGE[0], *sorted({v for v, f in zip(cut, on) if f == 1 and Y_RANGE[0] < v < Y_RANGE[1]}), Y_RANGE[1]])
    cx, cy = (xe[1:] + xe[:-1]) / 2, (ye[1:] + ye[:-1]) / 2
    gx, gy = np.meshgrid(cx, cy)
    import pandas as pd
    grid = pd.DataFrame({TWO[0]: gx.ravel(), TWO[1]: gy.ravel()})
    said = t.predict(grid)
    return xe, ye, np.array([SPECIES.index(s) for s in said]).reshape(len(cy), len(cx))


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


def species_in_picture(depth: int) -> int:
    import numpy as np
    return len(np.unique(regions(depth)[2]))


def picture(depth: int, labels: dict[str, str]) -> str:
    """Left: the training penguins by bill length and body mass, a colour and shape per species (style guide 1.7; the
    text before the picture says which shape is which), and in blue the decision boundary of the tree of this depth.
    Right: the mistakes on the training penguins (solid) and on the validation penguins (dashed) for every depth, with
    this depth marked. Drawn as SVG text, once for every stop of the slider (L.explore); dragging across the right
    chart sets the depth."""
    from everything_ai import chart
    train, _ = split()
    p = chart.Panel(X_RANGE, Y_RANGE, title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=[35, 40, 45, 50, 55, 60], yticks=[3000, 4000, 5000, 6000])
    for x0, y0, x1, y1 in boundary(*regions(depth)):
        p.line([x0, x1], [y0, y1], "pred", 2.0)
    for c, name in enumerate(SPECIES):
        rows = train[train["species"] == name]
        p.group(rows[TWO[0]], rows[TWO[1]], c, r=3.6)
    tr, va = curve()
    e = chart.Panel((0.6, 7.4), (0, 50), title=labels["every"], xlabel=labels["depth"], ylabel=labels["mistakes"],
                    xticks=DEPTHS, yticks=[0, 10, 20, 30, 40, 50])
    k = DEPTHS.index(depth)
    e.line([depth, depth], [0, max(tr[k], va[k])], "ink-3", 1.2)
    e.line(DEPTHS, tr, "loss", 2.0)
    e.line(DEPTHS, va, "loss", 2.0, dashed=True)
    e.points(DEPTHS, tr, "loss", 3.2, hollow=True)
    e.points(DEPTHS, va, "loss", 3.2, hollow=True)
    e.points([depth, depth], [tr[k], va[k]], "loss", 6.0)
    e.text(2.15, 44, labels["train"], "loss", anchor="start")
    e.text(4.2, 15.5, labels["val"], "loss", anchor="start")
    return chart.figure([p, e], alt=labels["alt"], width=330, height=300, drag=1)


# The lesson is done when these are correct. "read-tree" is the choice question in the lesson.
REQUIRED = ["split-mistakes", "choose-depth", "read-tree"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


# --- exercise 1: split_mistakes ---------------------------------------------------------------------------------------

class _Named:
    """A test array that the check's messages write by its name, such as `x`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import numpy as np
        return np.array(self.value)


# Twelve made-up penguins: their flipper length and their species. The thresholds the check asks about lie between the
# smallest and the largest flipper, so neither part is ever empty; some are equal to a flipper, where <= and < differ,
# and some leave three species in one part, where the fewest votes and the mistakes differ.
_FX = [181, 190, 186, 195, 197, 201, 210, 215, 222, 217, 199, 205]
_FY = ["Adelie", "Chinstrap", "Adelie", "Adelie", "Chinstrap", "Chinstrap", "Gentoo", "Gentoo", "Gentoo", "Chinstrap",
       "Adelie", "Gentoo"]
# Eight more, by bill length, for a second set of penguins.
_BX = [36.5, 46.1, 39.2, 49.8, 50.5, 47.5, 38.8, 45.2]
_BY = ["Adelie", "Gentoo", "Adelie", "Chinstrap", "Gentoo", "Chinstrap", "Adelie", "Adelie"]
_X, _Y = _Named("x", _FX), _Named("y", _FY)
_X2, _Y2 = _Named("bill", _BX), _Named("species", _BY)

_S_HEAD = "import numpy as np\n\n\ndef split_mistakes(x, y, t):\n    wrong = 0\n"
_S_UNIQUE = "        names, counts = np.unique(part, return_counts=True)\n"
_S_START = _S_HEAD + "    for part in [..., ...]:\n" + _S_UNIQUE + "        wrong = ...\n    return wrong\n"
_S = _S_HEAD + "    for part in [y[x <= t], y[x > t]]:\n" + _S_UNIQUE + "        wrong = wrong + len(part) - max(counts)\n    return wrong\n"


def _s(parts: str = "[y[x <= t], y[x > t]]", wrong: str = "wrong + len(part) - max(counts)", back: str = "    return wrong\n") -> str:
    return _S_HEAD + f"    for part in {parts}:\n" + _S_UNIQUE + f"        wrong = {wrong}\n" + back


EXERCISES = {
    "split-mistakes": FunctionBox(
        name="split-mistakes",
        function="split_mistakes",
        start=_S_START,
        solution=_S,
        visible=[(_X, _Y, 200)],
        hidden=[(_X, _Y, 195), (_X, _Y, 190), (_X, _Y, 205), (_X, _Y, 215), (_X, _Y, 186), (_X2, _Y2, 42.0),
                (_X2, _Y2, 47.5)],
        first=[_DOTS],
        mistakes=[
            Mistake("right-ones", wrong=_s(wrong="wrong + max(counts)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. `max(counts)` is how many penguins of a part have its most common species: those it gets right. Its mistakes are the others: `len(part) - max(counts)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. `max(counts)` is hoeveel pinguïns van een deel de soort hebben die er het vaakst in voorkomt: die heeft het goed. De fouten zijn de andere: `len(part) - max(counts)`."),
            Mistake("fewest-species", wrong=_s(wrong="wrong + min(counts)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. `min(counts)` counts only the rarest species of a part; with three species in a part, the second one is wrong too. Its mistakes are all penguins but those of the most common species: `len(part) - max(counts)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. `min(counts)` telt alleen de zeldzaamste soort van een deel; met drie soorten in een deel is de tweede ook fout. De fouten zijn alle pinguïns behalve die van de soort die het vaakst voorkomt: `len(part) - max(counts)`."),
            Mistake("strict", wrong=_s(parts="[y[x < t], y[x >= t]]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. A penguin with exactly `t` belongs to the first part: the question is *at most* `t`, so `x <= t`, and the other part `x > t`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Een pinguïn met precies `t` hoort bij het eerste deel: de vraag is *hooguit* `t`, dus `x <= t`, en het andere deel `x > t`."),
            Mistake("both-sides", wrong=_s(parts="[y[x <= t], y[x >= t]]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. A penguin with exactly `t` is now in both parts. The second part is the penguins the first leaves out: `x > t`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Een pinguïn met precies `t` zit nu in beide delen. Het tweede deel zijn de pinguïns die het eerste overlaat: `x > t`."),
            Mistake("last-part", wrong=_s(wrong="len(part) - max(counts)"),
                    en="`{call}` gives `{got}`: the mistakes of the second part only. In the loop, `wrong` gets the mistakes of each part in turn, so the first part's are gone. Add them up: `wrong = wrong + ...`.",
                    nl="`{call}` geeft `{got}`: alleen de fouten van het tweede deel. In de loop krijgt `wrong` om de beurt de fouten van elk deel, dus die van het eerste deel zijn weg. Tel ze op: `wrong = wrong + ...`."),
            Mistake("first-part", wrong=_s(back="        return wrong\n"),
                    en="`{call}` gives `{got}`: the mistakes of the first part only. Your `return` stands inside the loop, so the function stops after the first part. Move it to the left, under `for`.",
                    nl="`{call}` geeft `{got}`: alleen de fouten van het eerste deel. Je `return` staat binnen de loop, dus de functie stopt na het eerste deel. Zet hem naar links, onder `for`."),
            Mistake("measurements", wrong=_s(parts="[x[x <= t], x[x > t]]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. `x[x <= t]` gives the measurements of the penguins in a part; for their species, take `y[x <= t]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. `x[x <= t]` geeft de metingen van de pinguïns in een deel; voor hun soorten neem je `y[x <= t]`."),
            Mistake("no-split", wrong=_s(parts="[y, y]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. Both of your parts are all the penguins: `y` as a whole. The first part is `y[x <= t]`, the second `y[x > t]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je beide delen zijn alle pinguïns: heel `y`. Het eerste deel is `y[x <= t]`, het tweede `y[x > t]`."),
            Mistake("species-count", wrong=_s(wrong="wrong + len(part) - len(names)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. `len(names)` is how many different species a part has, not how many penguins its most common species has: that is `max(counts)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. `len(names)` is hoeveel verschillende soorten een deel heeft, niet hoeveel pinguïns de soort heeft die er het vaakst in voorkomt: dat is `max(counts)`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the three `...`: the species of the two parts in the list, and in the loop the mistakes of each part added to `wrong`.",
                      "nl": "Vervang de drie `...`: de soorten van de twee delen in de lijst, en in de loop de fouten van elk deel opgeteld bij `wrong`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has a different number of names between its parentheses. Keep `(x, y, t)`.",
                          "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x, y, t)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Are the two parts `y[x <= t]` and `y[x > t]`, and do you add up `len(part) - max(counts)` for each?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Zijn de twee delen `y[x <= t]` en `y[x > t]`, en tel je voor elk `len(part) - max(counts)` op?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use the `x`, `y` and `t` it gets?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie de `x`, `y` en `t` die hij krijgt?"},
            "correct": {"en": "Your function is right in all {n} cases. At 200 mm each part keeps two Chinstraps among the other species: 4 mistakes.",
                        "nl": "Je functie klopt in alle {n} gevallen. Bij 200 mm zitten er in elk deel twee Chinstraps tussen een andere soort: 4 fouten."},
        },
    ),
}


# --- exercise 2: choose the maximum depth on the validation set -------------------------------------------------------

_FEATURES = 'features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]'
_TREE_LINE = "    tree = DecisionTreeClassifier(max_depth=depth, random_state=0)"
_D_HEAD = ("import numpy as np\nimport pandas as pd\nfrom sklearn.tree import DecisionTreeClassifier\n\n"
           'df = pd.read_csv("data/penguins.csv")\ndf = df[df["body_mass_g"].isna() == False]\n'
           "order = np.random.default_rng(0).permutation(len(df))\ntrain = df.iloc[order[:206]]\nval = df.iloc[order[206:274]]\n"
           + _FEATURES + "\n\ndepths = [1, 2, 3, 4, 5, 6, 7, 8]\nmistakes = []\nfor depth in depths:\n" + _TREE_LINE + "\n")
_D_TAIL = "    mistakes.append(wrong)\n"
_D_BODY = ('    tree.fit(train[features], train["species"])\n    pred = tree.predict(val[features])\n'
           '    wrong = np.sum(pred != val["species"])\n')
_D_START = _D_HEAD + "    ...\n" + _D_TAIL + "best = ...\nbest\n"
_D = _D_HEAD + _D_BODY + _D_TAIL + "best = depths[np.argmin(mistakes)]\nbest"


def _d(body: str = _D_BODY, choose: str = "best = depths[np.argmin(mistakes)]\n") -> str:
    return _D_HEAD + body + _D_TAIL + choose + "best"


def _position(got, text) -> bool:
    return re.search(r"best\s*=\s*(np\.argmin\(|np\.array\(mistakes\)\.argmin\(|mistakes\.index\()", text) is not None


_BEST = best_depth()
_FEWEST = min(mistakes(tuple(FOUR), d) for d in EX_DEPTHS)
_DEEP = mistakes(tuple(FOUR), None)

EXERCISES["choose-depth"] = CodeBox(
    name="choose-depth",
    start=_D_START,
    solution=_D,
    keep=["for depth in depths:", _TREE_LINE.strip()],
    # The check also runs the box with other depths: a depth typed in by hand does not change with them.
    others=[{"depths": Code("[1, 2, 3]")}],
    first=[_DOTS],
    mistakes=[
        Mistake("all-columns", error=lambda e: isinstance(e, ValueError) and "could not convert string to float" in str(e),
                en="Your code stops: the whole table has columns with words, such as the species and the island, and a tree learns from numbers. Give `fit` and `predict` only the measurements: `train[features]` and `val[features]`.",
                nl="Je code stopt: de hele tabel heeft kolommen met woorden, zoals de soort en het eiland, en een boom leert van getallen. Geef `fit` en `predict` alleen de metingen: `train[features]` en `val[features]`."),
        Mistake("predict-columns", error=lambda e: isinstance(e, ValueError) and "feature names should match" in str(e),
                en="Your code stops: `predict` gets other columns than the ones `fit` learned from. Give both only the four measurements: `train[features]` and `val[features]`.",
                nl="Je code stopt: `predict` krijgt andere kolommen dan die waarvan `fit` leerde. Geef ze allebei alleen de vier metingen: `train[features]` en `val[features]`."),
        Mistake("compare-train", error=lambda e: isinstance(e, ValueError) and ("Lengths must match" in str(e) or "identically-labeled" in str(e)),
                en="Your code stops: the two sides of `!=` have a different number of penguins. Compare the predictions for `val` with the species of `val`: `tree.predict(val[features]) != val[\"species\"]`.",
                nl="Je code stopt: de twee kanten van `!=` hebben een ander aantal pinguïns. Vergelijk de voorspellingen voor `val` met de soorten van `val`: `tree.predict(val[features]) != val[\"species\"]`."),
        Mistake("not-fitted", error=lambda e: type(e).__name__ == "NotFittedError",
                en="Your code stops: the tree has to learn before it can predict. Call `tree.fit(train[features], train[\"species\"])` before `tree.predict`.",
                nl="Je code stopt: de boom moet eerst leren voordat hij kan voorspellen. Roep `tree.fit(train[features], train[\"species\"])` aan vóór `tree.predict`."),
        Mistake("list-argmin", error=lambda e: isinstance(e, AttributeError) and "'list' object has no attribute 'argmin'" in str(e),
                en="A list has no `.argmin()`; that belongs to a NumPy array. `np.argmin(mistakes)` takes the list as it is.",
                nl="Een lijst heeft geen `.argmin()`; die hoort bij een NumPy-array. `np.argmin(mistakes)` neemt de lijst zoals hij is."),
        Mistake("highest", matches=lambda got, text: "argmax" in text,
                en="`np.argmax` gives the position of the most mistakes: the worst depth. The best one has the fewest: `np.argmin`.",
                nl="`np.argmax` geeft de positie van de meeste fouten: de slechtste diepte. De beste heeft de minste: `np.argmin`."),
        Mistake("position", matches=_position,
                en="Your last line gives `{got}`: the position of the fewest mistakes in the list, counting from 0, not the depth itself. Look up the depth at that position: `depths[...]`.",
                nl="Je laatste regel geeft `{got}`: de positie van de minste fouten in de lijst, geteld vanaf 0, niet de diepte zelf. Zoek de diepte op die positie op: `depths[...]`."),
        Mistake("other-features", matches=lambda got, text: re.search(r"\.fit\(\s*train\[", text) is not None and re.search(r"\.fit\(\s*train\[\s*features\s*\]", text) is None,
                en="Your last line gives `{got}`. Your tree learns from other columns than the four measurements in `features`, such as the two of the picture. Give it `train[features]`, and predict with `val[features]`.",
                nl="Je laatste regel geeft `{got}`. Je boom leert van andere kolommen dan de vier metingen in `features`, zoals de twee van het beeld. Geef hem `train[features]`, en voorspel met `val[features]`."),
        # These by the text, not by the outcome: other mistakes also end with a deep tree or with depth 1.
        Mistake("on-training", matches=lambda got, text: re.search(r"predict\(\s*train\b", text) is not None,
                en="Your last line gives `{got}`. You count the mistakes on the training penguins, the ones the tree learned from: a deeper tree always does better there. Count them on the validation penguins: `tree.predict(val[features])`, compared with `val[\"species\"]`.",
                nl="Je laatste regel geeft `{got}`. Je telt de fouten op de trainingspinguïns, die waarvan de boom leerde: daar doet een diepere boom het altijd beter. Tel ze op de pinguïns uit de validation set: `tree.predict(val[features])`, vergeleken met `val[\"species\"]`."),
        Mistake("fit-on-val", matches=lambda got, text: re.search(r"fit\(\s*val\b", text) is not None,
                en="Your last line gives `{got}`. Your tree learns from the validation penguins, so it is measured on penguins it has seen. Let it learn from `train`, and measure it on `val`.",
                nl="Je laatste regel geeft `{got}`. Je boom leert van de pinguïns uit de validation set, dus hij wordt gemeten op pinguïns die hij al zag. Laat hem leren van `train`, en meet hem op `val`."),
        Mistake("right-ones", matches=lambda got, text: re.search(r"predict\([^\n]*\)\s*==|==\s*\w*\.?predict\(|pred\s*==|==\s*pred\b", text) is not None,
                en="Your last line gives `{got}`, the depth that is right the least often: with `==` you count the penguins the tree gets right. Count the ones it gets wrong, with `!=`.",
                nl="Je laatste regel geeft `{got}`, de diepte die het minst vaak goed is: met `==` tel je de pinguïns die de boom goed heeft. Tel die hij fout heeft, met `!=`."),
        Mistake("lowest-count", matches=lambda got, text: got == _FEWEST and re.search(r"best\s*=\s*((np\.)?min\(\s*mistakes\s*\)|mistakes\[)", text) is not None,
                en="Your last line gives `{got}`: the fewest mistakes themselves, not the depth that makes them. At which position in `mistakes` is it, and which depth is at that position in `depths`?",
                nl="Je laatste regel geeft `{got}`: het kleinste aantal fouten zelf, niet de diepte die ze maakt. Op welke positie in `mistakes` staat het, en welke diepte staat op die positie in `depths`?"),
    ],
    texts={
        "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                    "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
        "empty": {"en": "Replace the two `...`: in the loop, let the tree learn from `train`, and put in `wrong` how many validation penguins it gets wrong; then put the depth with the fewest mistakes in `best`.",
                  "nl": "Vervang de twee `...`: laat de boom in de loop leren van `train`, en zet in `wrong` hoeveel pinguïns uit de validation set hij fout heeft; zet daarna de diepte met de minste fouten in `best`."},
        "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                    "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
        "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other depths by itself.",
                 "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere dieptes."},
        "no_value": {"en": "The last line of the box shows nothing. Put `best` on the last line, so the check can see it.",
                     "nl": "De laatste regel van het vak laat niets zien. Zet `best` op de laatste regel, zodat de check het kan zien."},
        "wrong": {"en": "Your last line gives `{got}`. Do you count, for every depth, the validation penguins whose prediction is not their species, and take the depth with the fewest?",
                  "nl": "Je laatste regel geeft `{got}`. Tel je voor elke diepte de pinguïns uit de validation set waarvan de voorspelling niet hun soort is, en neem je de diepte met de minste?"},
        "other": {"en": "Right for these depths. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you look up the best depth in `depths` itself, and on a tie take the first, as `np.argmin` does?",
                  "nl": "Goed voor deze dieptes. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Zoek je de beste diepte op in `depths` zelf, en neem je bij gelijke stand de eerste, zoals `np.argmin` doet?"},
        "other_error": {"en": "Right for these depths. But with {where}, your code stops with an error. Do you look up the best depth in `depths` itself?",
                        "nl": "Goed voor deze dieptes. Maar met {where} stopt je code met een fout. Zoek je de beste diepte op in `depths` zelf?"},
        "correct": {"en": f"Depth {_BEST}, with {_FEWEST} of the 68 validation penguins wrong, as many as logistic regression in the project; depths 2 and 3 get one penguin more wrong. Without a limit the tree gets {_DEEP} wrong. The check also tried other depths.",
                    "nl": f"Diepte {_BEST}, met {_FEWEST} van de 68 pinguïns uit de validation set fout, evenveel als logistische regressie in het project; diepte 2 en 3 hebben er één pinguïn meer fout. Zonder grens heeft de boom er {_DEEP} fout. De check probeerde ook andere dieptes."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("split-mistakes", _S_START, "empty", None),
    ("split-mistakes", _S, "correct", None),
    ("split-mistakes", _s(wrong="wrong + (len(part) - max(counts))"), "correct", None),
    ("split-mistakes", _s(wrong="wrong + len(part) - counts.max()"), "correct", None),
    ("split-mistakes", _s(wrong="wrong + len(part) - counts[np.argmax(counts)]"), "correct", None),
    ("split-mistakes", _s(wrong="wrong + np.sum(counts) - np.max(counts)"), "correct", None),
    ("split-mistakes", _s(wrong="wrong + max(counts)"), "wrong", "right-ones"),
    ("split-mistakes", _s(wrong="wrong + min(counts)"), "wrong", "fewest-species"),
    ("split-mistakes", _s(parts="[y[x < t], y[x >= t]]"), "wrong", "strict"),
    ("split-mistakes", _s(parts="[y[x <= t], y[x >= t]]"), "wrong", "both-sides"),
    ("split-mistakes", _s(wrong="len(part) - max(counts)"), "wrong", "last-part"),
    ("split-mistakes", _s(back="        return wrong\n"), "wrong", "first-part"),
    ("split-mistakes", _s(parts="[x[x <= t], x[x > t]]"), "wrong", "measurements"),
    ("split-mistakes", _s(parts="[y, y]"), "wrong", "no-split"),
    ("split-mistakes", _s(wrong="wrong + len(part) - len(names)"), "wrong", "species-count"),
    ("split-mistakes", _s(parts="[y[x <= 200], y[x > 200]]"), "almost", None),
    ("split-mistakes", _S_HEAD + "    for part in [y[x <= t], y[x > t]]:\n" + _S_UNIQUE + "        wrong = wrong + len(part) - max(counts)\n    print(wrong)\n", "wrong", "prints"),
    ("choose-depth", _D_START, "empty", None),
    ("choose-depth", _D, "correct", None),
    ("choose-depth", _d('    tree.fit(train[features], train["species"])\n    wrong = np.sum(tree.predict(val[features]) != val["species"])\n'), "correct", None),
    ("choose-depth", _d('    tree.fit(train[features], train["species"])\n    wrong = 0\n    for s, p in zip(val["species"], tree.predict(val[features])):\n        if s != p:\n            wrong = wrong + 1\n'), "correct", None),
    ("choose-depth", _d(_D_BODY.replace("np.sum(", "np.mean(")), "correct", None),
    ("choose-depth", _d(choose="best = depths[mistakes.index(min(mistakes))]\n"), "correct", None),
    ("choose-depth", _d(choose="best = depths[np.argmax(mistakes)]\n"), "wrong", "highest"),
    ("choose-depth", _d(choose="best = np.argmin(mistakes)\n"), "wrong", "position"),
    ("choose-depth", _d(choose="best = mistakes.index(min(mistakes))\n"), "wrong", "position"),
    ("choose-depth", _d(_D_BODY.replace("predict(val[features])", "predict(train[features])").replace('!= val["species"]', '!= train["species"]')), "wrong", "on-training"),
    ("choose-depth", _d(_D_BODY.replace("fit(train[features], train[", "fit(val[features], val[")), "wrong", "fit-on-val"),
    ("choose-depth", _d(_D_BODY.replace("!=", "==")), "wrong", "right-ones"),
    ("choose-depth", _d(choose="best = min(mistakes)\n"), "wrong", "lowest-count"),
    ("choose-depth", _d(choose="best = mistakes[np.argmin(mistakes)]\n"), "wrong", "lowest-count"),
    # the two measurements of the picture: best depth 2, which is not the fewest mistakes themselves
    ("choose-depth", _d(_D_BODY.replace("train[features]", 'train[["bill_length_mm", "body_mass_g"]]').replace("val[features]", 'val[["bill_length_mm", "body_mass_g"]]')), "wrong", "other-features"),
    ("choose-depth", _d(_D_BODY.replace("predict(val[features])", "predict(val)")), "error", "predict-columns"),
    ("choose-depth", _d(_D_BODY.replace('!= val["species"]', '!= train["species"]')), "error", "compare-train"),
    ("choose-depth", _d(_D_BODY.replace("fit(train[features]", "fit(train")), "error", "all-columns"),
    ("choose-depth", _d('    pred = tree.predict(val[features])\n    wrong = np.sum(pred != val["species"])\n'), "error", "not-fitted"),
    ("choose-depth", _d(choose="best = depths[mistakes.argmin()]\n"), "error", "list-argmin"),
    ("choose-depth", _d(_D_BODY.replace('    wrong = np.sum(pred != val["species"])\n', "    wrong = 1\n")), "wrong", None),
    ("choose-depth", _d(choose=f"best = {_BEST}\n"), "almost", "other-value"),
]
