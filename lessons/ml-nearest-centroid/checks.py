# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-nearest-centroid: the nearest centroid on the Palmer penguins of ml-knn. The functions
centroids (one mean per class) and nc_predict (the class of the nearest mean), and why a class in two clumps fools
it. And the pictures: the straight boundaries between the centroids beside the boundary of k-NN, and the plane in
which the learner drags a new penguin."""
import functools
import importlib.util
import re

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake


def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_centroid",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_KNN = _lesson_checks("ml-knn")   # the penguins, their split and k-NN, as the learner met them
SPECIES = _KNN.SPECIES
TWO = _KNN.TWO
X_RANGE, Y_RANGE = _KNN.X_RANGE, _KNN.Y_RANGE
N_TRAIN, N_VAL = _KNN.N_TRAIN, _KNN.N_VAL
BEST_K = _KNN.best()             # the k of k-NN with the fewest validation mistakes on these two measurements: 15
split = _KNN.split


# --- the course's nearest centroid, as the lesson writes it ---------------------------------------------------------

def centroids(X_train, y_train):
    import numpy as np
    names = np.unique(y_train)
    means = []
    for name in names:
        means.append(X_train[y_train == name].mean(axis=0))
    return names, np.array(means)


def nc_predict(names, means, x):
    import numpy as np
    dists = np.linalg.norm(means - x, axis=1)
    return names[np.argmin(dists)]


@functools.lru_cache(maxsize=4)
def fitted(features: tuple = TWO):
    """The names and the centroids of the training penguins, standardized as in ml-knn."""
    s = split(features)
    return centroids(s["X_train"], s["y_train"])


@functools.lru_cache(maxsize=4)
def nc_mistakes(features: tuple = TWO, on: str = "val") -> int:
    """How many of the validation (or training) penguins the nearest centroid gets wrong."""
    s = split(features)
    names, means = fitted(features)
    X, y = s[f"X_{on}"], s[f"y_{on}"]
    return int(sum(nc_predict(names, means, x) != label for x, label in zip(X, y)))


def knn_mistakes(k: int, features: tuple = TWO) -> int:
    return _KNN.mistakes(features)[_KNN.KS.index(k)]


def raw_centroids() -> list:
    """The centroids in millimeters and grams, for the pictures: standardizing moves and stretches every axis, so
    the mean of the standardized penguins is the standardized mean."""
    s = split()
    names, means = fitted()
    return [(str(n), float(m[0] * s["std"][0] + s["mean"][0]), float(m[1] * s["std"][1] + s["mean"][1])) for n, m in zip(names, means)]


def standardized(bill: float, mass: float):
    import numpy as np
    s = split()
    return (np.array([bill, mass]) - s["mean"]) / s["std"]


def distances(bill: float, mass: float) -> list:
    """The distance from a penguin of this bill length and body mass to each centroid, standardized."""
    import numpy as np
    names, means = fitted()
    return [float(d) for d in np.linalg.norm(means - standardized(bill, mass), axis=1)]


def grouped(n: int, lang: str) -> str:
    """A whole number with its thousands marked, as the language writes it: 4,000 or 4.000."""
    return f"{int(n):,}".replace(",", "." if lang == "nl" else ",")


def nc_says(bill: float, mass: float) -> str:
    names, means = fitted()
    return str(nc_predict(names, means, standardized(bill, mass)))


def knn_says(bill: float, mass: float, k: int = BEST_K) -> str:
    s = split()
    return str(_KNN.knn_predict(s["X_train"], s["y_train"], standardized(bill, mass), k))


# --- the pictures ---------------------------------------------------------------------------------------------------

def _boundaries() -> list:
    """The straight boundaries of the nearest centroid in the picture, in millimeters and grams: for every pair of
    species, the line of the points equally far from their two centroids, where no third centroid is nearer. Each
    is a piece of a straight line; the three meet in one point."""
    import numpy as np
    s = split()
    names, means = fitted()
    out = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            mid, d = (means[a] + means[b]) / 2, means[b] - means[a]
            along = np.array([-d[1], d[0]]) / np.linalg.norm(d)   # the bisector runs at right angles to d
            ts = np.linspace(-12, 12, 4801)
            pts = mid[None, :] + ts[:, None] * along[None, :]
            raw = pts * s["std"] + s["mean"]
            inside = ((raw[:, 0] >= X_RANGE[0]) & (raw[:, 0] <= X_RANGE[1]) & (raw[:, 1] >= Y_RANGE[0]) & (raw[:, 1] <= Y_RANGE[1]))
            dists = np.linalg.norm(pts[:, None, :] - means[None, :, :], axis=2)
            others = np.delete(dists, [a, b], axis=1)
            keep = inside & (others.min(axis=1) > dists[:, a] - 1e-9) if others.size else inside
            if keep.any():
                i, j = np.flatnonzero(keep)[[0, -1]]
                out.append((float(raw[i, 0]), float(raw[i, 1]), float(raw[j, 0]), float(raw[j, 1])))
    return out


def _penguins(p, r: float = 3.4):
    s = split()
    for c, name in enumerate(SPECIES):
        rows = s["raw_train"][s["y_train"] == name]
        p.group(rows[:, 0], rows[:, 1], c, r=r)


def _centroid_marks(p):
    for c, (name, x, y) in enumerate(raw_centroids()):
        p.points([x], [y], "ink", r=9, hollow=True)
        p.group([x], [y], SPECIES.index(name), r=6.5)


def compare_picture(labels: dict) -> str:
    """Left: the 206 training penguins with the three centroids, large and ringed, and the straight boundaries of the
    nearest centroid. Right: the same penguins with the boundary of k-NN with the best k, which bends."""
    from everything_ai import chart
    ticks = dict(xticks=[35, 40, 45, 50, 55, 60], yticks=[3000, 4000, 5000, 6000])
    left = chart.Panel(X_RANGE, Y_RANGE, title=labels["nc"], xlabel=labels["x"], ylabel=labels["y"], **ticks)
    for x0, y0, x1, y1 in _boundaries():
        left.line([x0, x1], [y0, y1], "pred", 2.2)
    _penguins(left)
    _centroid_marks(left)
    right = chart.Panel(X_RANGE, Y_RANGE, title=labels["knn"], xlabel=labels["x"], ylabel=labels["y"], **ticks)
    for x0, y0, x1, y1 in _KNN._boundary(BEST_K):
        right.line([x0, x1], [y0, y1], "pred", 2.0)
    _penguins(right)
    return chart.figure([left, right], alt=labels["alt"], width=330, height=300)


# the stops of the plane of the play block: bill length in mm, body mass in g
BILLS = [34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58]
MASSES = [2800, 3200, 3600, 4000, 4400, 4800, 5200, 5600, 6000, 6400]


def plane_picture(bill: float, mass: float, labels: dict) -> str:
    """The boundaries and the centroids, and a new penguin at (bill, mass) with a line to each centroid: the line
    to the nearest drawn in ink, the others in grey."""
    from everything_ai import chart
    p = chart.Panel(X_RANGE, Y_RANGE, title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=[35, 40, 45, 50, 55, 60], yticks=[3000, 4000, 5000, 6000])
    for x0, y0, x1, y1 in _boundaries():
        p.line([x0, x1], [y0, y1], "pred", 2.2)
    d = distances(bill, mass)
    near = int(min(range(len(d)), key=lambda i: d[i]))
    for i, (name, x, y) in enumerate(raw_centroids()):
        p.line([bill, x], [mass, y], "ink" if i == near else "ink-3", 2.0 if i == near else 1.2, dashed=i != near)
    _centroid_marks(p)
    p.points([bill], [mass], "ink", r=5.5)
    return chart.figure([p], alt=labels["alt"], width=360, height=320)


# --- how the learner's answers are written ---------------------------------------------------------------------------

_Named = _KNN._Named


def _shown(v) -> str:
    """What a function gave back, as the lesson writes it: a species as "Adelie", numbers rounded to 3 decimals."""
    import numpy as np
    if isinstance(v, np.ndarray) and v.dtype.kind == "f":
        return str(np.round(v, 3).tolist())
    if isinstance(v, (float, np.floating)):
        return str(round(float(v), 3))
    return _KNN._shown(v)


def _shown_fit(v) -> str:
    """What centroids gave back: the names, and every mean rounded to 3 decimals."""
    import numpy as np
    if isinstance(v, tuple) and len(v) == 2:
        names, means = v
        try:
            m = np.round(np.asarray(means, dtype=float), 3).tolist()
        except (TypeError, ValueError):
            m = means
        return f"({_shown(names)}, {m})"
    return _shown(v)


def _same_fit(got, want) -> bool:
    import numpy as np
    if not (isinstance(got, tuple) and len(got) == 2):
        return False
    try:
        names, means = [str(n) for n in np.asarray(got[0]).tolist()], np.asarray(got[1], dtype=float)
    except (TypeError, ValueError):
        return False
    return (names == [str(n) for n in want[0].tolist()] and means.shape == want[1].shape
            and bool(np.allclose(means, want[1], atol=1e-9)))


def _means_of(got):
    import numpy as np
    try:
        return np.asarray(got[1], dtype=float) if isinstance(got, tuple) and len(got) == 2 else None
    except (TypeError, ValueError):
        return None


def _all_same_rows(got, text) -> bool:
    m = _means_of(got)
    return m is not None and m.ndim == 2 and len(m) > 1 and bool((abs(m - m[0]) < 1e-12).all())


def _one_number_each(got, text) -> bool:
    m = _means_of(got)
    return (m is not None and m.ndim == 1 and isinstance(got, tuple) and len(m) == len(got[0])
            and re.search(r"\.mean\(\s*\)", text) is not None)


# The lesson is done when these are correct. "two-clumps" is the choice question in the lesson.
REQUIRED = ["centroids", "nc-predict", "two-clumps"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# Ten made-up penguins of ml-knn, already standardized, and two more sets: three measurements, and two species in
# an order that is not the alphabet's.
_X, _Y = _KNN._X, _KNN._Y
_X3 = _Named("X3", [[0.1, 1.0, -0.5], [0.3, 0.8, -0.1], [-1.0, 0.2, 0.4], [-0.6, 0.0, 0.9], [-0.8, -0.4, 0.5], [1.2, -1.1, 0.0]])
_Y3 = _Named("y3", ["Gentoo", "Gentoo", "Adelie", "Adelie", "Adelie", "Chinstrap"])
_X2 = _Named("X2", [[2.0, 1.0], [-1.0, 0.5], [1.5, 2.0], [-2.0, -1.5]])
_Y2 = _Named("y2", ["Gentoo", "Adelie", "Gentoo", "Adelie"])

# --- exercise 1: centroids ------------------------------------------------------------------------------------------

_C_HEAD = ("import numpy as np\n\n\ndef centroids(X_train, y_train):\n    names = np.unique(y_train)\n    means = []\n"
           "    for name in names:\n")
_C_TAIL = "    return names, np.array(means)\n"


def _c(line: str = "means.append(X_train[y_train == name].mean(axis=0))") -> str:
    return _C_HEAD + f"        {line}\n" + _C_TAIL


_C_START = _c("...")
_C = _c()

# --- exercise 2: nc_predict -----------------------------------------------------------------------------------------


def _p(dists: str = "np.linalg.norm(means - x, axis=1)", back: str = "names[np.argmin(dists)]") -> str:
    return ("import numpy as np\n\n\ndef nc_predict(names, means, x):\n"
            f"    dists = {dists}\n    return {back}\n")


_P_START = _p("...", "...")
_P = _p()


def _fit_ten():
    """The names and the centroids of the ten made-up penguins, as the course's centroids gives them, each written
    by its name in the check's messages."""
    import numpy as np
    names, means = centroids(np.array(_KNN._PX), np.array(_KNN._PY))
    return _Named("names", names.tolist()), _Named("means", means.tolist())


_N, _M = _fit_ten()
_x = _KNN._x

EXERCISES = {
    "centroids": FunctionBox(
        name="centroids",
        picture=yours.centroids(0),
        function="centroids",
        start=_C_START,
        solution=_C,
        show=_shown_fit,
        compare=_same_fit,
        visible=[(_X, _Y)],
        hidden=[(_X3, _Y3), (_X2, _Y2)],
        first=[_DOTS],
        mistakes=[
            Mistake("whole-set", matches=_all_same_rows,
                    en="`{call}` gives the same mean for every species: the mean of all the penguins. Take only the rows of this species, `X_train[y_train == name]`, before you take the mean.",
                    nl="`{call}` geeft voor elke soort hetzelfde gemiddelde: dat van alle pinguïns. Neem alleen de rijen van deze soort, `X_train[y_train == name]`, voordat je het gemiddelde neemt."),
            Mistake("no-axis", matches=_one_number_each,
                    en="`{call}` gives one number for each species: `.mean()` without an axis takes the mean of all the numbers of its rows, bill lengths and body masses together. A centroid has a mean for each measurement: `.mean(axis=0)`, down the rows.",
                    nl="`{call}` geeft één getal voor elke soort: `.mean()` zonder axis neemt het gemiddelde van alle getallen van zijn rijen, snavellengte en lichaamsgewicht door elkaar. Een centroid heeft een gemiddelde voor elke meting: `.mean(axis=0)`, langs de rijen."),
            Mistake("across", error=lambda e: isinstance(e, ValueError) and "inhomogeneous" in str(e),
                    en="With `{call}`, your function stops at `np.array(means)`: the species give rows of different lengths. Does each species give one mean for each measurement, with `.mean(axis=0)`, down the rows?",
                    nl="Met `{call}` stopt je functie bij `np.array(means)`: de soorten geven rijen van verschillende lengte. Geeft elke soort één gemiddelde voor elke meting, met `.mean(axis=0)`, langs de rijen?"),
            Mistake("sum", wrong=_c("means.append(X_train[y_train == name].sum(axis=0))"),
                    en="`{call}` gives {got}: the sums of each species, not their means. A species with more penguins gets a larger sum. Use `.mean(axis=0)`.",
                    nl="`{call}` geeft {got}: de sommen van elke soort, niet hun gemiddelden. Een soort met meer pinguïns krijgt een grotere som. Gebruik `.mean(axis=0)`."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the `...` with one line: put the mean of the training penguins of this species, one mean for each measurement, at the end of `means`.",
                      "nl": "Vervang de `...` door één regel: zet het gemiddelde van de trainingspinguïns van deze soort, één gemiddelde voor elke meting, achteraan in `means`."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(X_train, y_train)`.",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X_train, y_train)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each species get the mean of its own rows, one number for each measurement?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Krijgt elke soort het gemiddelde van zijn eigen rijen, één getal voor elke meting?"},
            "almost": {"en": "Right for the ten penguins. But `{call}` gives {got}, and it should give {want}. Does your function work for any number of measurements and any species?",
                       "nl": "Goed voor de tien pinguïns. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk aantal metingen en elke soort?"},
            "correct": {"en": "Your function is right in all {n} cases: one centroid for each species, also with three measurements and with two species. That is all the nearest centroid learns: for the ten penguins, 6 numbers.",
                        "nl": "Je functie klopt in alle {n} gevallen: één centroid voor elke soort, ook met drie metingen en met twee soorten. Meer leert de nearest centroid niet: voor de tien pinguïns 6 getallen."},
        },
    ),
    "nc-predict": FunctionBox(
        name="nc-predict",
        picture=yours.steps(call=0),
        function="nc_predict",
        start=_P_START,
        solution=_P,
        show=_shown,
        visible=[(_N, _M, _x(0.2, -0.9))],
        hidden=[(_N, _M, _x(-1.0, -0.2)), (_N, _M, _x(1.2, 1.4)), (_N, _M, _x(0.9, -0.4)), (_N, _M, _x(0.3, 0.3)), (_N, _M, _x(-1.0, 1.0))],
        first=[_DOTS],
        mistakes=[
            Mistake("farthest", wrong=_p(back="names[np.argmax(dists)]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. `np.argmax` picks the centroid that is farthest away; the nearest has the smallest distance: `np.argmin`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. `np.argmax` kiest de centroid die het verst weg ligt; de dichtstbijzijnde heeft de kleinste afstand: `np.argmin`."),
            Mistake("position", wrong=_p(back="np.argmin(dists)"),
                    en="`{call}` gives `{got}`: the position of the nearest centroid, counting from 0, not its species. Look it up in `names`.",
                    nl="`{call}` geeft `{got}`: de positie van de dichtstbijzijnde centroid, geteld vanaf 0, niet zijn soort. Zoek hem op in `names`."),
            Mistake("one-distance", matches=lambda got, text: re.search(r"np\.linalg\.norm\((?![^\n]*axis)", text) is not None,
                    en="`{call}` gives `{got}`, and it should give `{want}`. Without `axis=1`, `np.linalg.norm` gives one number for all the centroids together, so the smallest is always the first. Give each row its own distance: `axis=1`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Zonder `axis=1` geeft `np.linalg.norm` één getal voor alle centroids samen, dus de kleinste is altijd de eerste. Geef elke rij zijn eigen afstand: `axis=1`."),
            Mistake("distances", wrong=_p(back="dists"),
                    en="`{call}` gives {got}: the distance to each centroid. The function gives back one species: that of the nearest.",
                    nl="`{call}` geeft {got}: de afstand tot elke centroid. De functie geeft één soort terug: die van de dichtstbijzijnde."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the two `...`: the distance from `x` to each centroid in `dists`, and after `return` the species of the nearest.",
                      "nl": "Vervang de twee `...`: de afstand van `x` tot elke centroid in `dists`, en na `return` de soort van de dichtstbijzijnde."},
            "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has a different number of names between its parentheses. Keep `(names, means, x)`.",
                          "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(names, means, x)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you work out the distance to every centroid, and give back the species of the smallest?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Bereken je de afstand tot elke centroid, en geef je de soort van de kleinste terug?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use the point `x` it gets, and the straight-line distance to every centroid, `np.linalg.norm`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie het punt `x` dat hij krijgt, en de afstand in een rechte lijn tot elke centroid, `np.linalg.norm`?"},
            "correct": {"en": "Your function is right in all {n} cases. For the example, k-NN with k = 3 said Adelie: two Adelies were among the three nearest penguins. The centroid of the Chinstraps is nearer than that of the Adelies, so the nearest centroid says Chinstrap.",
                        "nl": "Je functie klopt in alle {n} gevallen. Voor het voorbeeld zei k-NN met k = 3 Adelie: onder de drie dichtstbijzijnde pinguïns waren twee Adelies. De centroid van de Chinstraps ligt dichterbij dan die van de Adelies, dus de nearest centroid zegt Chinstrap."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("centroids", _C_START, "empty", None),
    ("centroids", _C, "correct", None),
    ("centroids", _c("means.append(np.mean(X_train[y_train == name], axis=0))"), "correct", None),
    ("centroids", _C_HEAD + "        rows = X_train[y_train == name]\n        means.append(rows.sum(axis=0) / len(rows))\n" + _C_TAIL, "correct", None),
    ("centroids", _c("means.append(X_train.mean(axis=0))"), "wrong", "whole-set"),
    ("centroids", _c("means.append(X_train[y_train == name].mean())"), "wrong", "no-axis"),
    ("centroids", _c("means.append(X_train[y_train == name].mean(axis=1))"), "error", "across"),
    ("centroids", _c("means.append(X_train[y_train == name].sum(axis=0))"), "wrong", "sum"),
    ("centroids", _c("means.append(X_train[y_train == name][0])"), "wrong", None),
    ("centroids", _c("means.append(X_train[y_train == name].mean(axis=0)[0])"), "wrong", None),
    ("nc-predict", _P_START, "empty", None),
    ("nc-predict", _p(back="..."), "wrong", "dots-left"),
    ("nc-predict", _P, "correct", None),
    ("nc-predict", _p("np.sqrt(((means - x) ** 2).sum(axis=1))"), "correct", None),
    ("nc-predict", _p(back="names[dists.argmin()]"), "correct", None),
    ("nc-predict", _p(back="names[np.argmax(dists)]"), "wrong", "farthest"),
    ("nc-predict", _p(back="np.argmin(dists)"), "wrong", "position"),
    ("nc-predict", _p("np.linalg.norm(means - x)"), "wrong", "one-distance"),
    ("nc-predict", _p(back="dists"), "wrong", "distances"),
    ("nc-predict", _p(back="names[0]"), "wrong", None),
    ("nc-predict", _p("np.abs(means - x).sum(axis=1)"), "almost", None),
    ("nc-predict", _p("np.abs(means - x).max(axis=1)"), "almost", None),
    ("nc-predict", _p("-(means @ x)"), "almost", None),
]
