# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-knn: k-nearest neighbors on the Palmer penguins. The function knn_predict, k chosen by the
mistakes on the validation set, and k = 1 on the training penguins themselves. And the picture: the training
penguins by bill length and body mass with the decision boundary of k-NN, and the validation mistakes of every k."""
import functools
import re

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake

DATA = "data/penguins.csv"   # the Palmer penguins (CC0): see data/README.md
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]   # in the order np.unique sorts them, so the vote breaks a tie the same way
SEED = 27                    # the shuffle of the worked example
N_TRAIN, N_VAL = 206, 68     # of the 342 penguins with all measurements; the last 68 are the test set
TWO = ("bill_length_mm", "body_mass_g")            # the worked example and the picture
OTHER = ("flipper_length_mm", "body_mass_g")       # exercise 2
KS = [1, 3, 5, 9, 15, 25, 41, 71, 121, 205]        # the values of k: the slider, the guess and exercise 2
TICKS = [1, 5, 15, 41, 121]                        # the values of k written along the axis of the picture


def knn_predict(X_train, y_train, x, k):
    """The course's k-NN, as the lesson writes it: the k training examples nearest to x vote."""
    import numpy as np
    dists = np.linalg.norm(X_train - x, axis=1)
    nearest = np.argsort(dists, kind="stable")[:k]
    names, counts = np.unique(y_train[nearest], return_counts=True)
    return names[np.argmax(counts)]


@functools.lru_cache(maxsize=1)
def _table():
    import numpy as np
    import pandas as pd
    from everything_ai import paths
    df = pd.read_csv(paths.ROOT / DATA)
    df = df[df["body_mass_g"].isna() == False]   # noqa: E712 (the lesson's own way, with what the learner knows)
    return df, np.random.default_rng(SEED).permutation(len(df))


@functools.lru_cache(maxsize=4)
def split(features: tuple = TWO) -> dict:
    """The training and validation penguins of the worked example with these measurements, standardized with the
    mean and standard deviation of the training set, as the worked example does; and the raw measurements, for
    the picture. The arrays are read-only, so code that is given them cannot change them for the next check."""
    import numpy as np
    df, order = _table()
    X, y = np.array(df[list(features)]), np.array(df["species"])
    train, val = order[:N_TRAIN], order[N_TRAIN:N_TRAIN + N_VAL]
    mean, std = X[train].mean(axis=0), X[train].std(axis=0)
    out = {"X_train": (X[train] - mean) / std, "y_train": y[train], "X_val": (X[val] - mean) / std, "y_val": y[val],
           "raw_train": X[train], "mean": mean, "std": std}
    for a in out.values():
        a.flags.writeable = False
    return out


@functools.lru_cache(maxsize=4)
def mistakes(features: tuple = TWO) -> tuple:
    """For every k in KS, how many of the validation penguins k-NN gets wrong."""
    s = split(features)
    return tuple(int(sum(knn_predict(s["X_train"], s["y_train"], x, k) != label for x, label in zip(s["X_val"], s["y_val"])))
                 for k in KS)


def best(features: tuple = TWO) -> int:
    """The k with the fewest validation mistakes; the first of them, as np.argmin takes it."""
    m = mistakes(features)
    return KS[m.index(min(m))]


def train_mistakes(k: int = 1) -> int:
    """How many of the training penguins k-NN gets wrong when they ask themselves."""
    s = split()
    return int(sum(knn_predict(s["X_train"], s["y_train"], x, k) != label for x, label in zip(s["X_train"], s["y_train"])))


def predicted(k: int) -> dict:
    """How many validation penguins k-NN calls each species."""
    s = split()
    said = [knn_predict(s["X_train"], s["y_train"], x, k) for x in s["X_val"]]
    return {name: said.count(name) for name in SPECIES}


# The picture: a grid over the plane of bill length and body mass, where k-NN predicts a species at every point.
X_RANGE, Y_RANGE = (31.0, 61.0), (2500.0, 6500.0)
_NX, _NY = 76, 76


@functools.lru_cache(maxsize=1)
def _votes():
    """For every point of the grid, the species of its training penguins from nearest to farthest, as numbers."""
    import numpy as np
    s = split()
    xs = np.linspace(*X_RANGE, _NX)
    ys = np.linspace(*Y_RANGE, _NY)
    gx, gy = np.meshgrid(xs, ys)
    grid = (np.stack([gx.ravel(), gy.ravel()], axis=1) - s["mean"]) / s["std"]
    dists = np.linalg.norm(grid[:, None, :] - s["X_train"][None, :, :], axis=2)
    order = np.argsort(dists, axis=1, kind="stable")
    labels = np.array([SPECIES.index(v) for v in s["y_train"]])
    # counts[i, j, c]: of the j + 1 nearest training penguins of grid point i, how many are species c
    counts = np.cumsum(np.eye(len(SPECIES), dtype=int)[labels[order]], axis=1)
    return xs, ys, counts


@functools.lru_cache(maxsize=len(KS))
def regions(k: int):
    """The species k-NN predicts at every point of the grid: rows bottom to top, columns left to right."""
    import numpy as np
    xs, ys, counts = _votes()
    return np.argmax(counts[:, k - 1, :], axis=1).reshape(_NY, _NX)   # the first of equal counts: as np.unique and np.argmax


def _boundary(k: int) -> list:
    """The edges between grid cells where the prediction changes, joined into straight runs: (x0, y0, x1, y1)."""
    xs, ys, _ = _votes()
    r = regions(k)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    lines = []
    for i in range(_NX - 1):   # a vertical piece between columns i and i + 1, row after row
        start = None
        for j in range(_NY + 1):
            edge = j < _NY and r[j, i] != r[j, i + 1]
            if edge and start is None:
                start = j
            if not edge and start is not None:
                x = (xs[i] + xs[i + 1]) / 2
                lines.append((x, ys[start] - dy / 2, x, ys[j - 1] + dy / 2))
                start = None
    for j in range(_NY - 1):   # a horizontal piece between rows j and j + 1, column after column
        start = None
        for i in range(_NX + 1):
            edge = i < _NX and r[j, i] != r[j + 1, i]
            if edge and start is None:
                start = i
            if not edge and start is not None:
                y = (ys[j] + ys[j + 1]) / 2
                lines.append((xs[start] - dx / 2, y, xs[i - 1] + dx / 2, y))
                start = None
    return lines


def species_in_picture(k: int) -> int:
    """How many species k-NN predicts somewhere in the picture."""
    import numpy as np
    return len(np.unique(regions(k)))


def picture(k: int, labels: dict[str, str]) -> str:
    """Left: the training penguins by bill length and body mass, a colour and shape per species (style guide 1.7; the
    text before the picture says which shape is which, as the boundary leaves no corner free for a key), and in blue
    the decision boundary of k-NN with this k. Right: the validation mistakes of every k, on a log scale, with
    this k marked. Drawn as SVG text, once for every stop of the slider (L.explore); dragging across the right chart
    sets k."""
    from everything_ai import chart
    s = split()
    p = chart.Panel(X_RANGE, Y_RANGE, title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=[35, 40, 45, 50, 55, 60], yticks=[3000, 4000, 5000, 6000])
    for x0, y0, x1, y1 in _boundary(k):
        p.line([x0, x1], [y0, y1], "pred", 2.0)
    for c, name in enumerate(SPECIES):
        rows = s["raw_train"][s["y_train"] == name]
        p.group(rows[:, 0], rows[:, 1], c, r=3.6)
    m = mistakes()
    # a tick at every second k, so that the numbers do not run into each other at the right; the dots show every k
    e = chart.Panel((0.8, 260), (0, 42), title=labels["every"], xlabel="k", ylabel=labels["mistakes"], xticks=TICKS,
                    yticks=[0, 10, 20, 30, 40], xlog=True)
    e.line([k, k], [0, m[KS.index(k)]], "ink-3", 1.2)
    e.line(KS, m, "loss", 2.0)
    e.points(KS, m, "loss", 3.2, hollow=True)
    e.points([k], [m[KS.index(k)]], "loss", 6.5)
    return chart.figure([p, e], alt=labels["alt"], width=330, height=300, drag=1)


# The lesson is done when these are correct. "k1-train" is the choice question in the lesson.
REQUIRED = ["knn-predict", "choose-k", "k1-train"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


# --- exercise 1: knn_predict ------------------------------------------------------------------------------------------

class _Named:
    """A test array that the check's messages write by its name, such as `X_train`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import numpy as np
        return np.array(self.value)


# Ten made-up penguins, already standardized: no two equally far from any point the check asks about, and no tie in
# a vote, so every right way of writing it gives the same answer.
_PX = [[0.3, -0.8], [0.0, -0.6], [-0.1, -1.3], [0.8, -1.2], [-0.6, -0.8], [-1.2, -0.3], [0.4, 0.6], [1.0, 0.9],
       [1.4, 1.5], [1.1, -0.2]]
_PY = ["Chinstrap", "Adelie", "Adelie", "Chinstrap", "Adelie", "Adelie", "Gentoo", "Gentoo", "Gentoo", "Chinstrap"]
_X, _Y = _Named("X_train", _PX), _Named("y_train", _PY)


def _x(a: float, b: float) -> _Named:
    return _Named(f"np.array([{a}, {b}])", [a, b])


def _shown(v) -> str:
    """What a function gave back, as the lesson writes it: a species as "Adelie", several as a list, a number as it is."""
    import numpy as np
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        return "[" + ", ".join(_shown(x) for x in v) + "]"
    return str(v)


_K_HEAD = "import numpy as np\n\n\ndef knn_predict(X_train, y_train, x, k):\n    dists = np.linalg.norm(X_train - x, axis=1)\n"
_K_NAMES = "    names, counts = np.unique(y_train[nearest], return_counts=True)\n"
_K_START = _K_HEAD + "    nearest = ...\n" + _K_NAMES + "    return ...\n"
_K = _K_HEAD + '    nearest = np.argsort(dists, kind="stable")[:k]\n' + _K_NAMES + "    return names[np.argmax(counts)]\n"


def _k(nearest: str, back: str = "names[np.argmax(counts)]") -> str:
    return _K_HEAD + f"    nearest = {nearest}\n" + _K_NAMES + f"    return {back}\n"


_SORT_VALUES = Mistake(
    "sorted-distances", error=lambda e: isinstance(e, IndexError) and "integer" in str(e),
    en="`{call}` stops: `np.sort` gives the distances themselves, sorted, and you cannot pick penguins out of `y_train` with distances. `np.argsort` gives their positions, from nearest to farthest.",
    nl="`{call}` stopt: `np.sort` geeft de afstanden zelf, gesorteerd, en met afstanden kun je geen pinguïns uit `y_train` halen. `np.argsort` geeft hun posities, van dichtbij naar ver weg.")

EXERCISES = {
    "knn-predict": FunctionBox(
        name="knn-predict",
        function="knn_predict",
        start=_K_START,
        solution=_K,
        visible=[(_X, _Y, _x(0.2, -0.9), 3)],
        show=_shown,
        hidden=[(_X, _Y, _x(0.2, -0.9), 1), (_X, _Y, _x(-0.9, -0.5), 5), (_X, _Y, _x(0.6, 0.3), 1), (_X, _Y, _x(1.0, 1.1), 3),
                (_X, _Y, _x(0.7, -0.7), 3), (_X, _Y, _x(0.0, 0.0), 9), (_X, _Y, _x(0.9, -0.5), 5)],
        first=[_DOTS],
        mistakes=[
            _SORT_VALUES,
            Mistake("one-neighbor", wrong=_k("np.argmin(dists)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. Your function always lets only the nearest penguin decide, whatever `k` is. Take the first `k` positions of `np.argsort(dists)`, so that all `k` nearest vote.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je functie laat altijd alleen de dichtstbijzijnde pinguïn beslissen, wat `k` ook is. Neem de eerste `k` posities van `np.argsort(dists)`, zodat alle `k` dichtstbijzijnde meestemmen."),
            Mistake("farthest", wrong=_k("np.argsort(dists)[-k:]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. `np.argsort` puts the nearest penguin first, so `[-k:]` takes the `k` farthest. Take the first `k`: `[:k]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. `np.argsort` zet de dichtstbijzijnde pinguïn vooraan, dus `[-k:]` neemt de `k` verste. Neem de eerste `k`: `[:k]`."),
            Mistake("all-neighbors", wrong=_k("np.argsort(dists)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. Without `[:k]`, every training penguin votes, not just the `k` nearest.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Zonder `[:k]` stemt elke pinguïn uit de training set mee, niet alleen de `k` dichtstbijzijnde."),
            Mistake("count", wrong=_k('np.argsort(dists, kind="stable")[:k]', "np.max(counts)"),
                    en="`{call}` gives `{got}`: how many votes the winner got, not which species it is. Look up the species at the position of the most votes in `names`.",
                    nl="`{call}` geeft `{got}`: hoeveel stemmen de winnaar kreeg, niet welke soort het is. Zoek de soort op de positie van de meeste stemmen op in `names`."),
            Mistake("position", wrong=_k('np.argsort(dists, kind="stable")[:k]', "np.argmax(counts)"),
                    en="`{call}` gives `{got}`: the position of the winner in `names`, counting from 0, not the species itself. Look it up: `names[...]`.",
                    nl="`{call}` geeft `{got}`: de positie van de winnaar in `names`, geteld vanaf 0, niet de soort zelf. Zoek hem op: `names[...]`."),
            Mistake("first-name", wrong=_k('np.argsort(dists, kind="stable")[:k]', "names[0]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. `names[0]` is the first species in alphabetical order among the neighbors, not the one with the most votes. Which position in `counts` holds the most?",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. `names[0]` is de eerste soort op alfabet onder de buren, niet die met de meeste stemmen. Op welke positie in `counts` staan de meeste?"),
            Mistake("fewest", wrong=_k('np.argsort(dists, kind="stable")[:k]', "names[np.argmin(counts)]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. `np.argmin` picks the species with the fewest votes; the vote goes to the most: `np.argmax`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. `np.argmin` kiest de soort met de minste stemmen; de stemming gaat naar de meeste: `np.argmax`."),
            Mistake("votes", wrong=_k('np.argsort(dists, kind="stable")[:k]', "y_train[nearest]"),
                    en="`{call}` gives `{got}`: the species of all `k` neighbors. The function gives back one species: the one most of them have.",
                    nl="`{call}` geeft `{got}`: de soorten van alle `k` buren. De functie geeft één soort terug: die de meesten van hen hebben."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: the positions of the `k` nearest training penguins in `nearest`, and after `return` the species that got the most votes.",
                      "nl": "Vervang de twee `...`: de posities van de `k` dichtstbijzijnde pinguïns uit de training set in `nearest`, en na `return` de soort met de meeste stemmen."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Keep `(X_train, y_train, x, k)`.",
                          "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X_train, y_train, x, k)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you take the first `k` positions of `np.argsort(dists)`, and give back the species with the most votes?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Neem je de eerste `k` posities van `np.argsort(dists)`, en geef je de soort met de meeste stemmen terug?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use `k`, and the point `x` it gets?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie `k`, en het punt `x` dat hij krijgt?"},
            "correct": {"en": "Your function is right in all {n} cases. With `k = 1` the nearest penguin of the example is a Chinstrap; with `k = 3` two Adelies outvote it.",
                        "nl": "Je functie klopt in alle {n} gevallen. Met `k = 1` is de dichtstbijzijnde pinguïn van het voorbeeld een Chinstrap; met `k = 3` stemmen twee Adelies hem weg."},
        },
    ),
}


# --- exercise 2: choose k on the validation set ------------------------------------------------------------------------

_C_HEAD = "import numpy as np\n\nks = [1, 3, 5, 9, 15, 25, 41, 71, 121, 205]\nmistakes = []\nfor k in ks:\n    wrong = 0\n    for i in range(len(X_val)):\n"
_C_TAIL = "    mistakes.append(wrong)\n"
_C_START = _C_HEAD + "        ...\n" + _C_TAIL + "best = ...\nbest\n"
_C_IF = "        if knn_predict(X_train, y_train, X_val[i], k) != y_val[i]:\n            wrong = wrong + 1\n"
_C = _C_HEAD + _C_IF + _C_TAIL + "best = ks[np.argmin(mistakes)]\nbest"


def _c(body: str = _C_IF, choose: str = "best = ks[np.argmin(mistakes)]\n") -> str:
    return _C_HEAD + body + _C_TAIL + choose + "best"


def _position(got, text) -> bool:
    return re.search(r"best\s*=\s*(np\.argmin\(|np\.array\(mistakes\)\.argmin\(|mistakes\.index\()", text) is not None


def _names() -> dict:
    """What the box may use without making it: the penguins of exercise 2, and the course's knn_predict."""
    s = split(OTHER)
    return {name: s[name] for name in ("X_train", "y_train", "X_val", "y_val")} | {"knn_predict": knn_predict}


_OTHER_MISTAKES = mistakes(OTHER)
_BEST_OTHER = best(OTHER)

EXERCISES["choose-k"] = CodeBox(
    name="choose-k",
    start=_C_START,
    solution=_C,
    names=_names(),
    keep=["for k in ks:"],
    # The check also runs the box with other values of k: a k typed in by hand does not change with them.
    others=[{"ks": Code("[1, 3, 5]")}],
    first=[_DOTS],
    mistakes=[
        Mistake("whole-set", error=lambda e: isinstance(e, ValueError) and "broadcast" in str(e),
                en="Your code stops: `knn_predict` gets one penguin at a time, `X_val[i]`, not all of `X_val` at once.",
                nl="Je code stopt: `knn_predict` krijgt één pinguïn tegelijk, `X_val[i]`, niet heel `X_val` in één keer."),
        Mistake("all-labels", syntax=r"!=\s*y_val\s*:|^\s*if\s+y_val\s*!=",
                en="Your code stops on the line `{line}`. `y_val` holds the species of all 68 validation penguins; compare the prediction with the species of this one penguin, `y_val[i]`.",
                nl="Je code stopt op de regel `{line}`. `y_val` bevat de soorten van alle 68 pinguïns uit de validation set; vergelijk de voorspelling met de soort van deze ene pinguïn, `y_val[i]`."),
        Mistake("list-argmin", error=lambda e: isinstance(e, AttributeError) and "'list' object has no attribute 'argmin'" in str(e),
                en="A list has no `.argmin()`; that belongs to a NumPy array. `np.argmin(mistakes)` takes the list as it is.",
                nl="Een lijst heeft geen `.argmin()`; die hoort bij een NumPy-array. `np.argmin(mistakes)` neemt de lijst zoals hij is."),
        Mistake("highest", matches=lambda got, text: "argmax" in text,
                en="`np.argmax` gives the position of the most mistakes: the worst k. The best one has the fewest: `np.argmin`.",
                nl="`np.argmax` geeft de positie van de meeste fouten: de slechtste k. De beste heeft de minste: `np.argmin`."),
        Mistake("position", matches=_position,
                en="Your last line gives `{got}`: the position of the fewest mistakes in the list, counting from 0, not k itself. Look up k at that position: `ks[...]`.",
                nl="Je laatste regel geeft `{got}`: de positie van de minste fouten in de lijst, geteld vanaf 0, niet k zelf. Zoek k op die positie op: `ks[...]`."),
        # These two by the text, not by the outcome: other mistakes also end with best = 1 or best = 205.
        Mistake("on-training", matches=lambda got, text: re.search(r"knn_predict\([^)\n]*X_train\[i\]", text) is not None,
                en="Your last line gives `{got}`. You ask k-NN about the training penguins themselves: with a small k, each one is its own nearest neighbor. Ask it about the validation penguins, `X_val[i]`, and compare with `y_val[i]`.",
                nl="Je laatste regel geeft `{got}`. Je vraagt k-NN naar de trainingspinguïns zelf: met een kleine k is elk zijn eigen dichtstbijzijnde buur. Vraag het naar de pinguïns uit de validation set, `X_val[i]`, en vergelijk met `y_val[i]`."),
        Mistake("right-ones", matches=lambda got, text: re.search(r"knn_predict\([^\n]*\)\s*==|==\s*knn_predict\(", text) is not None,
                en="Your last line gives `{got}`, the k that is right the least often: with `==` you count the penguins k-NN gets right. Count the ones it gets wrong, with `!=`.",
                nl="Je laatste regel geeft `{got}`, de k die het minst vaak goed is: met `==` tel je de pinguïns die k-NN goed heeft. Tel die het fout heeft, met `!=`."),
        Mistake("lowest-count", wrong=_c(choose="best = min(mistakes)\n"),
                en="Your last line gives `{got}`: the fewest mistakes themselves, not the k that makes them. At which position in `mistakes` is it, and which k is at that position in `ks`?",
                nl="Je laatste regel geeft `{got}`: het kleinste aantal fouten zelf, niet de k die ze maakt. Op welke positie in `mistakes` staat het, en welke k staat op die positie in `ks`?"),
    ],
    texts={
        "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                    "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
        "empty": {"en": "Replace the two `...`: in the loop, add 1 to `wrong` for every validation penguin that k-NN gets wrong; then put the k with the fewest mistakes in `best`.",
                  "nl": "Vervang de twee `...`: tel in de loop 1 op bij `wrong` voor elke pinguïn uit de validation set die k-NN fout heeft; zet daarna de k met de minste fouten in `best`."},
        "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                    "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
        "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other values of k by itself.",
                 "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere waarden van k."},
        "no_value": {"en": "The last line of the box shows nothing. Put `best` on the last line, so the check can see it.",
                     "nl": "De laatste regel van het vak laat niets zien. Zet `best` op de laatste regel, zodat de check het kan zien."},
        "wrong": {"en": "Your last line gives `{got}`. Do you count, for every k, the validation penguins whose prediction is not their species, and take the k with the fewest?",
                  "nl": "Je laatste regel geeft `{got}`. Tel je voor elke k de pinguïns uit de validation set waarvan de voorspelling niet hun soort is, en neem je de k met de minste?"},
        "other": {"en": "Right for these values of k. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you look up the best k in `ks` itself?",
                  "nl": "Goed voor deze waarden van k. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Zoek je de beste k op in `ks` zelf?"},
        "other_error": {"en": "Right for these values of k. But with {where}, your code stops with an error. Do you look up the best k in `ks` itself?",
                        "nl": "Goed voor deze waarden van k. Maar met {where} stopt je code met een fout. Zoek je de beste k op in `ks` zelf?"},
        "correct": {"en": f"k = {_BEST_OTHER}, with {min(_OTHER_MISTAKES)} of the 68 validation penguins wrong. With bill length instead of flipper length, k = {best()} made only {min(mistakes())}: flipper length tells Adelies and Chinstraps apart less well. The check also tried other values of k.",
                    "nl": f"k = {_BEST_OTHER}, met {min(_OTHER_MISTAKES)} van de 68 pinguïns uit de validation set fout. Met de snavellengte in plaats van de flipperlengte maakte k = {best()} er maar {min(mistakes())}: aan de flipperlengte zie je Adelies en Chinstraps minder goed uit elkaar. De check probeerde ook andere waarden van k."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("knn-predict", _K_START, "empty", None),
    ("knn-predict", _K, "correct", None),
    ("knn-predict", _k("np.argsort(dists)[:k]"), "correct", None),
    ("knn-predict", _K_HEAD + "    nearest = np.argsort(dists)[0:k]\n    votes = y_train[nearest]\n    names, counts = np.unique(votes, return_counts=True)\n"
                    "    best = np.argmax(counts)\n    return names[best]\n", "correct", None),
    ("knn-predict", _k("np.argsort(dists)[:k]", "names[counts.argmax()]"), "correct", None),
    ("knn-predict", _k("np.sort(dists)[:k]"), "error", "sorted-distances"),
    ("knn-predict", _k("np.argmin(dists)"), "wrong", "one-neighbor"),
    ("knn-predict", _k("np.argsort(dists)[:k]", "y_train[nearest[0]]"), "wrong", "one-neighbor"),
    ("knn-predict", _k("np.argsort(dists)[-k:]"), "wrong", "farthest"),
    ("knn-predict", _k("np.argsort(dists)"), "wrong", "all-neighbors"),
    ("knn-predict", _k("np.argsort(dists)[:k]", "np.max(counts)"), "wrong", "count"),
    ("knn-predict", _k("np.argsort(dists)[:k]", "np.argmax(counts)"), "wrong", "position"),
    ("knn-predict", _k("np.argsort(dists)[:k]", "names[0]"), "wrong", "first-name"),
    ("knn-predict", _k("np.argsort(dists)[:k]", "names[np.argmin(counts)]"), "wrong", "fewest"),
    ("knn-predict", _k("np.argsort(dists)[:k]", "y_train[nearest]"), "wrong", "votes"),
    ("knn-predict", _k("np.argsort(dists)[:3]"), "almost", None),
    ("knn-predict", _K_HEAD + "    nearest = np.argsort(dists)[:k]\n" + _K_NAMES + "    print(names[np.argmax(counts)])\n", "wrong", "prints"),
    ("choose-k", _C_START, "empty", None),
    ("choose-k", _C, "correct", None),
    ("choose-k", _c("        if knn_predict(X_train, y_train, X_val[i], k) != y_val[i]:\n            wrong += 1\n"), "correct", None),
    ("choose-k", _c("        wrong = wrong + int(knn_predict(X_train, y_train, X_val[i], k) != y_val[i])\n", "best = ks[mistakes.index(min(mistakes))]\n"), "correct", None),
    ("choose-k", _c(choose="best = ks[np.array(mistakes).argmin()]\n"), "correct", None),
    ("choose-k", _c(choose="best = ks[np.argmax(mistakes)]\n"), "wrong", "highest"),
    ("choose-k", _c(choose="best = np.argmin(mistakes)\n"), "wrong", "position"),
    ("choose-k", _c(choose="best = mistakes.index(min(mistakes))\n"), "wrong", "position"),
    ("choose-k", _c("        if knn_predict(X_train, y_train, X_train[i], k) != y_train[i]:\n            wrong = wrong + 1\n"), "wrong", "on-training"),
    ("choose-k", _c("        if knn_predict(X_train, y_train, X_val[i], k) == y_val[i]:\n            wrong = wrong + 1\n"), "wrong", "right-ones"),
    ("choose-k", _c("        wrong = wrong + 1\n"), "wrong", None),   # best = 1 as well, but not on the training penguins
    ("choose-k", _c(choose="best = ks[np.argmin(wrong)]\n"), "wrong", None),
    ("choose-k", _c("        if knn_predict(X_train, y_train, X_val[i], k) != y_train[i]:\n            wrong = wrong + 1\n"), "wrong", None),   # best = 205, but no ==
    ("choose-k", _c(choose="best = min(mistakes)\n"), "wrong", "lowest-count"),
    ("choose-k", _c("        if knn_predict(X_train, y_train, X_val[i], k) != y_val:\n            wrong = wrong + 1\n"), "error", "all-labels"),
    ("choose-k", _c(choose="best = ks[mistakes.argmin()]\n"), "error", "list-argmin"),
    ("choose-k", _c("        if knn_predict(X_train, y_train, X_val, k) != y_val[i]:\n            wrong = wrong + 1\n"), "error", "whole-set"),
    ("choose-k", _c(choose=f"best = {_BEST_OTHER}\n"), "almost", "other-value"),
]
