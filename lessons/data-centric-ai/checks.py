# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson data-centric-ai: the 274 penguins outside the test set of ml-knn, with bill length and body
mass. The course makes 41 of their 274 species labels wrong itself, at random (NOISE_SEED), so it knows which ones
are wrong; the 68 test penguins keep their right labels. Every penguin gets the probability of its label from a k-NN
with k = 9 that stored only the other four folds of cross-validation: an out-of-fold prediction. Checking the
labels from the lowest probability up finds the wrong ones; fixing them changes the mistakes of k-NN on the test set.
The functions p_given and suspects. tests/test_data_centric_ai.py computes the numbers in the lesson again."""
import contextlib
import functools
import io
import re
import warnings

from everything_ai import chart, check as checking, yours
from everything_ai.check import FunctionBox, Mistake

DATA = "data/penguins.csv"   # the Palmer penguins (CC0): see data/README.md
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]   # in the order np.unique sorts them, so a tie is broken the same way
SEED = 27                    # the shuffle of ml-knn and ml-cross-validation: 274 penguins outside the test set, 68 in it
N_REST, FOLD, FOLDS = 274, 55, 5
TWO = ["bill_length_mm", "body_mass_g"]
K = 9                        # the k cross-validation chose in ml-cross-validation
NOISE_SEED, N_WRONG = 15, 41  # 41 of the 274 labels made wrong, 15%; seed 15 gives a typical result (see the tests)
CHECK = 30                   # how many labels the lesson checks by hand
CHECKS = [0, 10, 20, 30, 40, 50, 60, 80, 100]   # the stops of the slider
KS = [1, 3, 5, 9, 15, 25, 41, 71, 121]           # the values of k cross-validation tried in ml-cross-validation


@functools.lru_cache(maxsize=1)
def _data():
    import numpy as np
    import pandas as pd
    from everything_ai import paths
    df = pd.read_csv(paths.ROOT / DATA)
    df = df[df["body_mass_g"].isna() == False]   # noqa: E712 (the lesson's own way, with what the learner knows)
    order = np.random.default_rng(SEED).permutation(len(df))
    return np.array(df[TWO]), np.array(df["species"]), order[:N_REST], order[N_REST:]


def make_wrong(seed: int = NOISE_SEED, n: int = N_WRONG):
    """The labels as the lesson's code makes them: a copy of the right ones, with n of the 274 outside the test set
    changed to one of the two other species, drawn at random. Gives (the labels, the positions of the wrong ones)."""
    import numpy as np
    _, y, rest, _ = _data()
    rng = np.random.default_rng(seed)
    given = np.array(y)
    wrong = rng.permutation(rest)[:n]
    for i in wrong:
        others = []
        for s in SPECIES:
            if s != y[i]:
                others.append(s)
        given[i] = others[rng.integers(0, 2)]
    return given, wrong


@functools.lru_cache(maxsize=1)
def given_labels():
    return make_wrong()


def folds():
    """The validation and training positions of each of the five folds, as ml-cross-validation cuts them."""
    import numpy as np
    _, _, rest, _ = _data()
    return [(rest[FOLD * f:FOLD * (f + 1)], np.concatenate([rest[:FOLD * f], rest[FOLD * (f + 1):]])) for f in range(FOLDS)]


def _nearest(train, val, k: int):
    """For every penguin at the positions `val`, the positions of its k nearest among `train`, standardized with the
    mean and standard deviation of `train`; equally far ones in their order, as np.argsort with kind="stable"."""
    import numpy as np
    X, _, _, _ = _data()
    mean, std = X[train].mean(axis=0), X[train].std(axis=0)
    A, B = (X[train] - mean) / std, (X[val] - mean) / std
    return train[np.argsort(np.linalg.norm(B[:, None, :] - A[None, :, :], axis=2), axis=1, kind="stable")[:, :k]]


def out_of_fold(labels=None, k: int = K):
    """The probability k-NN gives each of the 274 penguins' own label, in the order of the 274, each from a k-NN that
    stored only the other four folds: the share of its k nearest there with the same label."""
    import numpy as np
    labels = given_labels()[0] if labels is None else labels
    p = []
    for val, train in folds():
        near = _nearest(train, val, k)
        p += [float(np.mean(labels[near[j]] == labels[i])) for j, i in enumerate(val)]
    return np.array(p)


def in_sample(k: int, labels=None):
    """The same, from one k-NN that stored all 274, the penguin itself too."""
    import numpy as np
    labels = given_labels()[0] if labels is None else labels
    _, _, rest, _ = _data()
    near = _nearest(rest, rest, k)
    return np.array([float(np.mean(labels[near[j]] == labels[i])) for j, i in enumerate(rest)])


@functools.lru_cache(maxsize=1)
def probabilities():
    return tuple(out_of_fold())


def is_wrong():
    """For each of the 274, in their order: is its label one the course made wrong?"""
    import numpy as np
    _, _, rest, _ = _data()
    return np.isin(rest, given_labels()[1])


def found(order, c: int) -> int:
    """How many of the wrong labels are among the first c of `order`, positions among the 274."""
    import numpy as np
    return int(np.sum(is_wrong()[np.asarray(order[:c], dtype=int)]))


@functools.lru_cache(maxsize=1)
def suspect_order() -> tuple:
    import numpy as np
    return tuple(int(j) for j in np.argsort(np.array(probabilities()), kind="stable"))


def fixed_labels(c: int):
    """The labels with the c most suspect of the 274 checked and set right."""
    import numpy as np
    _, y, rest, _ = _data()
    labels = np.array(given_labels()[0])
    picked = rest[list(suspect_order()[:c])]
    labels[picked] = y[picked]
    return labels


def test_mistakes(labels, k: int) -> int:
    """How many of the 68 test penguins k-NN gets wrong when all 274 vote with these labels, as the course's k-NN."""
    import numpy as np
    _, y, rest, test = _data()
    wrong = 0
    for j, row in enumerate(_nearest(rest, test, k)):
        names, counts = np.unique(labels[row], return_counts=True)
        wrong += int(names[np.argmax(counts)] != y[test[j]])
    return wrong


@functools.lru_cache(maxsize=None)
def mistakes_after(c: int, k: int) -> int:
    return test_mistakes(fixed_labels(c), k)


@functools.lru_cache(maxsize=None)
def mistakes_right(k: int) -> int:
    """With all labels right."""
    return test_mistakes(_data()[1], k)


@functools.lru_cache(maxsize=1)
def cv_choice() -> tuple:
    """The k cross-validation chooses on the labels with the wrong ones, as in ml-cross-validation, and the mistakes
    it then makes on the test set."""
    import numpy as np
    labels = given_labels()[0]
    totals = []
    for k in KS:
        total = 0
        for val, train in folds():
            for j, row in enumerate(_nearest(train, val, k)):
                names, counts = np.unique(labels[row], return_counts=True)
                total += int(names[np.argmax(counts)] != labels[val[j]])
        totals.append(total)
    best = KS[int(np.argmin(totals))]
    return best, test_mistakes(labels, best)


def at_random(c: int) -> float:
    """How many wrong labels c labels checked at random hold on average."""
    return c * N_WRONG / N_REST


def number(x: float, lang: str, digits: int = 1) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def said(c: int, lang: str) -> str:
    """What the slider says at c labels checked."""
    h, m1, m9 = found(suspect_order(), c), mistakes_after(c, 1), mistakes_after(c, K)
    r1, r9 = mistakes_right(1), mistakes_right(K)
    luck = m1 < r1 or m9 < r9
    if lang == "nl":
        if c == 0:
            head = f"Je controleert niets: 0 van de {N_WRONG} verkeerde labels gevonden."
        else:
            head = (f"Je controleert de {c} verdachtste labels en vindt {h} van de {N_WRONG} verkeerde; {c} willekeurige "
                    f"bevatten er gemiddeld {number(at_random(c), lang)}.")
        tail = (f" Als je verbetert wat je vond, zit k = 1 bij {m1} van de 68 testpinguïns fout en k = 9 bij {m9}; met alle "
                f"labels goed bij {r1} en {r9}.")
        if c == 0:
            tail = (f" Zonder te verbeteren zit k = 1 bij {m1} van de 68 testpinguïns fout en k = 9 bij {m9}; met alle "
                    f"labels goed bij {r1} en {r9}.")
        if luck:
            tail += " Minder dan met alle labels goed: bij 68 testpinguïns is een pinguïn of twee toeval."
        return head + tail
    if c == 0:
        head = f"You check nothing: 0 of the {N_WRONG} wrong labels found."
    else:
        head = (f"You check the {c} most suspect labels and find {h} of the {N_WRONG} wrong ones; {c} checked at random "
                f"find {number(at_random(c), lang)} on average.")
    tail = (f" With what you found fixed, k = 1 gets {m1} of the 68 test penguins wrong and k = 9 gets {m9}; with all "
            f"labels right, {r1} and {r9}.")
    if c == 0:
        tail = (f" Without fixing, k = 1 gets {m1} of the 68 test penguins wrong and k = 9 gets {m9}; with all labels "
                f"right, {r1} and {r9}.")
    if luck:
        tail += " Fewer than with all labels right: with 68 test penguins, a penguin or two is luck."
    return head + tail


_WORDS = {
    "found": {"en": "Wrong labels found", "nl": "Verkeerde labels gevonden"},
    "checked": {"en": "labels checked", "nl": "labels gecontroleerd"},
    "suspect": {"en": "most suspect first", "nl": "verdachtste eerst"},
    "random": {"en": "at random", "nl": "willekeurig"},
    "all": {"en": "41 wrong", "nl": "41 verkeerd"},
    "test": {"en": "Mistakes on the 68 test penguins", "nl": "Fouten op de 68 testpinguïns"},
}


def words(lang: str) -> dict:
    return {k: v[lang] for k, v in _WORDS.items()}


def picture(c: int, w: dict, lang: str) -> str:
    """Left: how many of the 41 wrong labels you find, checking the most suspect first or at random, for 0 to 100
    labels checked. Right: the mistakes of k-NN with k = 1 and k = 9 on the 68 test penguins after fixing what you
    found. A dot at c on every line. Drawn as SVG text, once for every stop of the slider (L.explore)."""
    order = suspect_order()
    xs = list(range(0, 101))
    left = chart.Panel((0, 100), (0, 46), title=w["found"], xlabel=w["checked"], xticks=[0, 20, 40, 60, 80, 100],
                       yticks=[0, 10, 20, 30, 40])
    left.line([0, 100], [N_WRONG, N_WRONG], role="ink-3", width=1, dashed=True)
    left.line(xs, [found(order, x) for x in xs], role="loss", width=2.4)
    left.line([0, 100], [0, at_random(100)], role="loss", width=1.4, dashed=True)
    left.points([c, c], [found(order, c), at_random(c)], role="loss", r=5.5, hollow=True)
    left.text(2, N_WRONG + 2.6, w["all"], role="ink-2", anchor="start")
    left.text(48, 31, w["suspect"], role="loss", anchor="start", halo=True)
    left.text(99, at_random(100) + 2.5, w["random"], role="loss", anchor="end", halo=True)
    right = chart.Panel((0, 100), (0, 16), title=w["test"], xlabel=w["checked"], xticks=[0, 20, 40, 60, 80, 100],
                        yticks=[0, 5, 10, 15])
    for k, width in ((1, 2.4), (K, 1.4)):
        ys = [mistakes_after(x, k) for x in CHECKS]
        right.line(CHECKS, ys, role="loss", width=width)
        right.points([c], [mistakes_after(c, k)], role="loss", r=5.5, hollow=True)
    for k in (1, K):
        right.text(99, mistakes_after(100, k) + 0.9, f"k = {k}", role="loss", anchor="end", halo=True)
    return chart.figure([left, right], alt=w["alt"], width=330, height=250)


# The lesson is done when these are correct. "self" is the choice question in the lesson.
REQUIRED = ["p-given", "suspects", "self"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


class _Named:
    """A test value that the check's messages write by its name, such as `X_train`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import numpy as np
        return np.array(self.value)


def _shown(v):
    """What a function gave back, as the lesson writes it: a number with up to six digits, positions as a list."""
    import numpy as np
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        # a list is written as Python writes it, with a point in both languages
        return "[" + ", ".join(_float(x) if isinstance(x, float) else str(_shown(x)) for x in v) + "]"
    if isinstance(v, (bool, int)):
        return str(v)
    if isinstance(v, float):
        text = _float(v)
        return {"en": text, "nl": text.replace(".", ",")}
    return repr(v)[:60]


def _float(x: float) -> str:
    """A float with up to six digits; a whole one keeps its `.0`, so that 0.0 does not look like the position 0."""
    text = f"{x:.6g}"
    return text + ".0" if text.lstrip("-").isdigit() else text


def _same_number(got, want) -> bool:
    try:
        return abs(float(got) - float(want)) < 1e-9
    except (TypeError, ValueError):
        return False


def _same_positions(got, want) -> bool:
    """The same positions in the same order; a wrong version that gives numbers that are no positions, such as sorted
    probabilities, is compared as numbers."""
    import numpy as np
    try:
        g, w = np.asarray(got), np.asarray(want)
        if g.shape != w.shape or g.dtype.kind not in "iuf" or w.dtype.kind not in "iuf":
            return False
        if w.dtype.kind in "iu":
            return g.dtype.kind in "iu" and bool(np.all(g == w))
        return g.dtype.kind == "f" and bool(np.allclose(g, w))
    except (TypeError, ValueError):
        return False


def _function(seen_):
    fn = seen_.mine.names.get(seen_.exercise.function) if seen_.mine is not None else None
    return fn if callable(fn) else None


def _quiet(fn, *args):
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return checking.with_time_limit(lambda: fn(*args)), True
    except Exception:
        return None, False


def _key(text: str) -> str:
    return ('<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.15rem">'
            + text + "</div>")


# --- exercise 1: the probability of a penguin's own label ---------------------------------------------------------------

# Ten made-up penguins, already standardized, the ones of ml-knn: no two equally far from any point the check asks about.
_PX = [[0.3, -0.8], [0.0, -0.6], [-0.1, -1.3], [0.8, -1.2], [-0.6, -0.8], [-1.2, -0.3], [0.4, 0.6], [1.0, 0.9],
       [1.4, 1.5], [1.1, -0.2]]
_PY = ["Chinstrap", "Adelie", "Adelie", "Chinstrap", "Adelie", "Adelie", "Gentoo", "Gentoo", "Gentoo", "Chinstrap"]
_X, _Y = _Named("X_train", _PX), _Named("y_train", _PY)


def _x(a: float, b: float) -> _Named:
    return _Named(f"np.array([{a}, {b}])", [a, b])


_P_HEAD = "import numpy as np\n\n\ndef p_given(X_train, y_train, x, label, k):\n    dists = np.linalg.norm(X_train - x, axis=1)\n"


def _pg(nearest: str = 'np.argsort(dists, kind="stable")[:k]', back: str = "np.mean(y_train[nearest] == label)") -> str:
    return _P_HEAD + f"    nearest = {nearest}\n    return {back}\n"


_P_START = _pg("...", "...")
_P = _pg()
_P_VISIBLE = (_X, _Y, _x(0.2, -0.9), "Chinstrap", 3)
_P_HIDDEN = [(_X, _Y, _x(0.2, -0.9), "Adelie", 3), (_X, _Y, _x(0.2, -0.9), "Gentoo", 3), (_X, _Y, _x(1.0, 1.1), "Gentoo", 3),
             (_X, _Y, _x(0.0, 0.0), "Adelie", 9), (_X, _Y, _x(0.9, -0.5), "Chinstrap", 5), (_X, _Y, _x(-0.9, -0.5), "Adelie", 1),
             (_X, _Y, _x(0.7, -0.7), "Chinstrap", 4)]

_P_WORDS = {
    "title": {"en": "Your probabilities of the 274 labels, out of fold", "nl": "Jouw kansen van de 274 labels, out-of-fold"},
    "bars": {"en": "Penguins per probability of their own label", "nl": "Pinguïns per kans op hun eigen label"},
    "x": {"en": "probability of the label, k = 9", "nl": "kans op het label, k = 9"},
    "key": {"en": "purple: a label the course made wrong; gray: a right label.", "nl": "paars: een label dat de course verkeerd maakte; grijs: een goed label."},
    "said": {"en": "With your function, {lw} of the 41 wrong labels get a probability of 1/3 or lower, and {lr} of the 233 right ones.",
             "nl": "Met jouw functie krijgen {lw} van de 41 verkeerde labels een kans van 1/3 of lager, en {lr} van de 233 goede."},
}


def _p_values(fn):
    """What the learner's p_given gives each of the 274 penguins, out of fold, with k = 9; None if it is no probability."""
    import numpy as np
    X, _, _, _ = _data()
    labels = given_labels()[0]

    def all_of_them():
        out = []
        for val, train in folds():
            mean, std = X[train].mean(axis=0), X[train].std(axis=0)
            A = (X[train] - mean) / std
            for i in val:
                out.append(float(fn(A, np.array(labels[train]), (X[i] - mean) / std, labels[i], K)))
        return out

    values, ok = _quiet(all_of_them)
    if not ok or values is None or any(not (0 <= v <= 1) for v in values):
        return None
    return values


def _pg_picture(seen_) -> str:
    lang = seen_.lang
    fn = _function(seen_)
    if fn is None:
        return ""
    values = _p_values(fn)
    if values is None:
        return ""
    w = {k: v[lang] for k, v in _P_WORDS.items()}
    wrong = is_wrong()
    bins = [round(v * K) for v in values]
    p = chart.Panel((-0.7, 9.7), (0, 115), title=w["bars"], xlabel=w["x"], xticks=[0, 3, 6, 9],
                    xtick_labels=["0", "0.33", "0.67", "1"], yticks=[0, 25, 50, 75, 100])
    for b in range(K + 1):
        nw = sum(1 for x, bad in zip(bins, wrong) if x == b and bad)
        nr = sum(1 for x, bad in zip(bins, wrong) if x == b and not bad)
        if nw:
            p.rect(b - 0.38, 0, b + 0.38, nw, role="loss", solid=True)
        if nr:
            p.rect(b - 0.38, nw, b + 0.38, nw + nr, role="ink-3", solid=True)
    lw = sum(1 for x, bad in zip(bins, wrong) if x <= 3 and bad)
    lr = sum(1 for x, bad in zip(bins, wrong) if x <= 3 and not bad)
    text = w["said"].format(lw=lw, lr=lr)
    return yours.figure(w["title"], chart.figure([p], alt=text, width=460, height=240) + _key(w["key"] + " " + text), alt=text)


_SORT_VALUES = Mistake(
    "sorted-distances", error=lambda e: isinstance(e, IndexError) and "integer" in str(e),
    en="`{call}` stops: `np.sort` gives the distances themselves, sorted, and you cannot pick penguins out of `y_train` with distances. `np.argsort` gives their positions, from nearest to farthest.",
    nl="`{call}` stopt: `np.sort` geeft de afstanden zelf, gesorteerd, en met afstanden kun je geen pinguïns uit `y_train` halen. `np.argsort` geeft hun posities, van dichtbij naar ver weg.")

EXERCISES = {
    "p-given": FunctionBox(
        name="p-given",
        picture=_pg_picture,
        function="p_given",
        start=_P_START,
        solution=_P,
        show=_shown,
        compare=_same_number,
        visible=[_P_VISIBLE],
        hidden=_P_HIDDEN,
        first=[_DOTS],
        mistakes=[
            _SORT_VALUES,
            Mistake("species", matches=lambda got, text: isinstance(got, str) or type(got).__name__ == "str_",
                    en="`{call}` gives {got}: a species, the one the vote picks. Here you want a probability: the share of the `k` nearest that have `label`.",
                    nl="`{call}` geeft {got}: een soort: de winnaar van de stemming. Hier wil je een kans: het deel van de `k` dichtstbijzijnde dat `label` heeft."),
            Mistake("farthest", wrong=_pg("np.argsort(dists)[-k:]"),
                    en="`{call}` gives {got}, and it should give {want}. `np.argsort` puts the nearest penguin first, so `[-k:]` takes the `k` farthest. Take the first `k`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. `np.argsort` zet de dichtstbijzijnde pinguïn vooraan, dus `[-k:]` neemt de `k` verste. Neem de eerste `k`."),
            Mistake("all-neighbors", wrong=_pg("np.argsort(dists)"),
                    en="`{call}` gives {got}, and it should give {want}: your share is over all training penguins, not over the `k` nearest. Keep only the first `k` positions of `np.argsort(dists)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je deel gaat over alle pinguïns uit de training set, niet over de `k` dichtstbijzijnde. Houd alleen de eerste `k` posities van `np.argsort(dists)`."),
            Mistake("one-more", wrong=_pg("np.argsort(dists)[:k + 1]"),
                    en="`{call}` gives {got}, and it should give {want}: your function takes one neighbor too many. `[:k]` already gives `k` positions.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie neemt één buur te veel. `[:k]` geeft al `k` posities."),
            Mistake("count", wrong=_pg(back="np.sum(y_train[nearest] == label)"),
                    en="`{call}` gives {got}: how many of the `k` nearest have `label`. A probability is a share: divide by `k`, or take the mean.",
                    nl="`{call}` geeft {got}: hoeveel van de `k` dichtstbijzijnde `label` hebben. Een kans is een deel: deel door `k`, of neem het gemiddelde."),
            Mistake("positions", wrong=_pg(back="np.mean(nearest == label)"),
                    en="`{call}` gives {got}: `nearest` holds positions, numbers, and no number is a species. Pick their labels with `y_train[nearest]`.",
                    nl="`{call}` geeft {got}: `nearest` bevat posities, getallen, en geen getal is een soort. Pak hun labels met `y_train[nearest]`."),
            Mistake("first-rows", wrong=_pg(back="np.mean(y_train[:k] == label)"),
                    en="`{call}` gives {got}, and it should give {want}: `y_train[:k]` are the first `k` penguins of the table, not the nearest. Pick the labels at the positions in `nearest`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `y_train[:k]` zijn de eerste `k` pinguïns van de tabel, niet de dichtstbijzijnde. Pak de labels op de posities in `nearest`."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the two `...`: the positions of the `k` nearest training penguins in `nearest`, and after `return` the share of them that have `label`.",
                      "nl": "Vervang de twee `...`: de posities van de `k` dichtstbijzijnde pinguïns uit de training set in `nearest`, en na `return` het deel daarvan dat `label` heeft."},
            "arguments": {"en": "The check calls `{call}`, with five values, and the `def` line of your function has a different number of names between its parentheses. Keep `(X_train, y_train, x, label, k)`.",
                          "nl": "De check roept `{call}` aan, met vijf waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X_train, y_train, x, label, k)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}: the share of the `k` nearest training penguins whose label is `label`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn: het deel van de `k` dichtstbijzijnde pinguïns uit de training set met als label `label`."},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use `x`, `label` and `k`, the values it gets?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `x`, `label` en `k`, de waarden die ze krijgt?"},
            "correct": {"en": "Your function is right in all {n} cases. Of the 3 nearest of the example, one is a Chinstrap: a label Chinstrap gets probability 1/3, a label Adelie 2/3.",
                        "nl": "Je functie klopt in alle {n} gevallen. Van de 3 dichtstbijzijnde van het voorbeeld is er één een Chinstrap: een label Chinstrap krijgt kans 1/3, een label Adelie 2/3."},
        },
    ),
}


# --- exercise 2: the most suspect first ---------------------------------------------------------------------------------

_S_HEAD = "import numpy as np\n\n\ndef suspects(p, n):\n"


def _su(order: str = 'np.argsort(p, kind="stable")', back: str = "order[:n]") -> str:
    return _S_HEAD + f"    order = {order}\n    return {back}\n"


_S_START = _su("...", "...")
_S = _su()


def _p(values: list) -> _Named:
    return _Named("np.array([" + ", ".join(str(v) for v in values) + "])", values)


_S_VISIBLE = (_p([0.89, 0.11, 1.0, 0.0, 0.56, 0.22]), 3)
_S_HIDDEN = [(_p([0.89, 0.11, 1.0, 0.0, 0.56, 0.22]), 1), (_p([0.5, 0.0, 0.0, 1.0]), 2), (_p([0.3, 0.9, 0.1]), 3),
             (_p([1.0, 0.78, 0.33, 0.67, 0.44, 0.0, 0.11, 0.89]), 4), (_p([0.67, 0.22, 0.22, 0.56, 0.0]), 3)]

_S_WORDS = {
    "title": {"en": "Checking the labels in your order", "nl": "De labels controleren in jouw volgorde"},
    "found": {"en": "Wrong labels found", "nl": "Verkeerde labels gevonden"},
    "checked": {"en": "labels checked", "nl": "labels gecontroleerd"},
    "yours": {"en": "your order", "nl": "jouw volgorde"},
    "right": {"en": "lowest first", "nl": "laagste eerst"},
    "random": {"en": "at random", "nl": "willekeurig"},
    "said": {"en": "Checking the first 30 of your order finds {h} of the 41 wrong labels; 30 at random find 4.5 on average.",
             "nl": "Als je de eerste 30 van jouw volgorde controleert, vind je {h} van de 41 verkeerde labels; 30 willekeurige bevatten er gemiddeld 4,5."},
    "lower": {"en": " The lowest probabilities first find {r}: the thin line.", "nl": " Met de laagste kansen eerst vind je er {r}: de dunne lijn."},
}


SHOWN = 100   # the picture asks the learner's function for the 100 most suspect


def _su_picture(seen_) -> str:
    import numpy as np
    lang = seen_.lang
    fn = _function(seen_)
    if fn is None:
        return ""
    P = probabilities()
    got, ok = _quiet(fn, np.array(P), SHOWN)
    try:
        order = [int(j) for j in np.asarray(got)] if ok else None
    except (TypeError, ValueError):
        order = None
    if order is None or len(order) != SHOWN or len(set(order)) != SHOWN or not all(0 <= j < N_REST for j in order) \
            or np.asarray(got).dtype.kind not in "iu":
        return ""
    w = {k: v[lang] for k, v in _S_WORDS.items()}
    xs = list(range(0, 101))
    p = chart.Panel((0, 100), (0, 46), title=w["found"], xlabel=w["checked"], xticks=[0, 20, 40, 60, 80, 100],
                    yticks=[0, 10, 20, 30, 40])
    best = suspect_order()
    # the same probabilities in the same order; equal ones may come in another order
    same = [P[j] for j in order] == sorted(P)[:SHOWN]
    if not same:
        p.line(xs, [found(best, x) for x in xs], role="loss", width=1)
        p.text(62, found(best, 60) + 2.5, w["right"], role="loss", anchor="start", halo=True)
    p.line(xs, [found(order, x) for x in xs], role="loss", width=2.4)
    p.line([0, 100], [0, at_random(100)], role="loss", width=1.4, dashed=True)
    p.text(99, at_random(100) + 2.5, w["random"], role="loss", anchor="end", halo=True)
    mine = found(order, 60)
    p.text(62, mine - 5 if mine > 12 else mine + 2.5, w["yours"], role="loss", anchor="start", halo=True)
    text = w["said"].format(h=found(order, CHECK))
    if not same:
        text += w["lower"].format(r=found(best, CHECK))
    return yours.figure(w["title"], chart.figure([p], alt=text, width=460, height=240) + _key(text), alt=text)


EXERCISES["suspects"] = FunctionBox(
    name="suspects",
    picture=_su_picture,
    function="suspects",
    start=_S_START,
    solution=_S,
    show=_shown,
    compare=_same_positions,
    visible=[_S_VISIBLE],
    hidden=_S_HIDDEN,
    first=[_DOTS],
    mistakes=[
        Mistake("values", wrong=_su("np.sort(p)"),
                en="`{call}` gives {got}: the probabilities themselves, sorted. You want the positions of the penguins they belong to, so that you know which labels to check: `np.argsort`.",
                nl="`{call}` geeft {got}: de kansen zelf, gesorteerd. Je wilt de posities van de pinguïns waar ze bij horen, zodat je weet welke labels je controleert: `np.argsort`."),
        Mistake("ties", state="almost", wrong=_su('np.argsort(-p, kind="stable")[::-1]'),
                en="`{call}` gives {got}, and it should give {want}: the right probabilities, but equal ones in the wrong order. Keep equal probabilities in their order, the first one first: `kind=\"stable\"`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn: de goede kansen, maar gelijke in de verkeerde volgorde. Houd gelijke kansen in hun volgorde, de eerste eerst: `kind=\"stable\"`."),
        Mistake("most-sure", wrong=_su('np.argsort(p, kind="stable")[::-1]'),
                en="`{call}` gives {got}, and it should give {want}: the highest probabilities first, the labels the model is surest of. Suspect are the lowest.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn: de hoogste kansen eerst, de labels waar het model het zekerst van is. Verdacht zijn de laagste."),
        Mistake("last-n", wrong=_su(back="order[-n:]"),
                en="`{call}` gives {got}, and it should give {want}: `[-n:]` takes the end of the order, the highest probabilities. `np.argsort` puts the lowest first.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn: `[-n:]` neemt het einde van de volgorde, de hoogste kansen. `np.argsort` zet de laagste vooraan."),
        Mistake("one-more", wrong=_su(back="order[:n + 1]"),
                en="`{call}` gives {got}, and it should give {want}: one position too many. `[:n]` already gives `n`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn: één positie te veel. `[:n]` geeft er al `n`."),
        Mistake("skip-first", wrong=_su(back="order[1:n + 1]"),
                en="`{call}` gives {got}, and it should give {want}: your list skips the first position, the most suspect label. Counting starts at 0.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn: je lijst slaat de eerste positie over, het verdachtste label. Tellen begint bij 0."),
        Mistake("everything", wrong=_su(back="order"),
                en="`{call}` gives {got}: every position, in order. You check only the first `n`: keep those.",
                nl="`{call}` geeft {got}: alle posities, op volgorde. Je controleert alleen de eerste `n`: houd die."),
        Mistake("lowest-only", wrong=_su("np.argmin(p)", "order"),
                en="`{call}` gives {got}: one position, of the lowest probability. You want `n` of them, the lowest first: `np.argsort` sorts them all.",
                nl="`{call}` geeft {got}: één positie, van de laagste kans. Je wilt er `n`, de laagste eerst: `np.argsort` zet ze allemaal op volgorde."),
    ],
    texts=_SAME | {
        "empty": {"en": "Replace the two `...`: the positions from the lowest probability to the highest in `order`, and after `return` the first `n` of them.",
                  "nl": "Vervang de twee `...`: de posities van de laagste kans tot de hoogste in `order`, en na `return` de eerste `n` daarvan."},
        "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(p, n)`.",
                      "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(p, n)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}: the positions of the `n` lowest probabilities, the lowest first.",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn: de posities van de `n` laagste kansen, de laagste eerst."},
        "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use `p` and `n`, the values it gets?",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `p` en `n`, de waarden die ze krijgt?"},
        "correct": {"en": "Your function is right in all {n} cases. Penguin 3, with probability 0, comes first, then 1 and 5: check those labels first.",
                    "nl": "Je functie klopt in alle {n} gevallen. Pinguïn 3, met kans 0, komt eerst, dan 1 en 5: controleer die labels eerst."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("p-given", _P_START, "empty", None),
    ("p-given", _pg(back="..."), "wrong", "dots-left"),
    ("p-given", _P, "correct", None),
    ("p-given", _pg("np.argsort(dists)[:k]", "np.sum(y_train[nearest] == label) / k"), "correct", None),
    ("p-given", _pg("np.argsort(dists)[0:k]", "len(y_train[nearest][y_train[nearest] == label]) / len(nearest)"), "correct", None),
    ("p-given", _P_HEAD + "    nearest = np.argsort(dists)[:k]\n    same = 0\n    for i in nearest:\n        if y_train[i] == label:\n            same = same + 1\n    return same / k\n", "correct", None),
    ("p-given", _pg("np.sort(dists)[:k]"), "error", "sorted-distances"),
    ("p-given", _P_HEAD + "    nearest = np.argsort(dists)[:k]\n    names, counts = np.unique(y_train[nearest], return_counts=True)\n    return names[np.argmax(counts)]\n", "wrong", "species"),
    ("p-given", _pg("np.argsort(dists)[-k:]"), "wrong", "farthest"),
    ("p-given", _pg("np.argsort(dists)"), "wrong", "all-neighbors"),
    ("p-given", _pg("np.argsort(dists)[:k + 1]"), "wrong", "one-more"),
    ("p-given", _pg(back="np.sum(y_train[nearest] == label)"), "wrong", "count"),
    ("p-given", _pg(back="np.mean(nearest == label)"), "wrong", "positions"),
    ("p-given", _pg(back="np.mean(y_train[:k] == label)"), "wrong", "first-rows"),
    ("suspects", _S_START, "empty", None),
    ("suspects", _su(back="..."), "wrong", "dots-left"),
    ("suspects", _S, "correct", None),
    ("suspects", _su("np.argsort(p)", "order[0:n]"), "correct", None),
    ("suspects", _S_HEAD + "    return np.argsort(p, kind=\"stable\")[:n]\n", "correct", None),
    ("suspects", _su("np.sort(p)"), "wrong", "values"),
    ("suspects", _su('np.argsort(-p, kind="stable")[::-1]'), "almost", "ties"),
    ("suspects", _su('np.argsort(p, kind="stable")[::-1]'), "wrong", "most-sure"),
    ("suspects", _su(back="order[-n:]"), "wrong", "last-n"),
    ("suspects", _su(back="order[:n + 1]"), "wrong", "one-more"),
    ("suspects", _su(back="order[1:n + 1]"), "wrong", "skip-first"),
    ("suspects", _su(back="order"), "wrong", "everything"),
    ("suspects", _su("np.argmin(p)", "order"), "wrong", "lowest-only"),
]
