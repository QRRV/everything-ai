# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-dbscan: DBSCAN on 400 made-up points (a ring, a clump in its middle, a curved band beside
it and 20 stray points), next to k-means and single linkage on the same points. The clusters and noise for every
radius eps, the function core_points, and the clusters and noise for five values of min_samples."""
import contextlib
import functools
import io
import re
import warnings

from everything_ai import chart, check as checking, yours
from everything_ai.check import CodeBox, FunctionBox, Mistake

SEED = 1
EPS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]   # the stops of the slider
MIN_SAMPLES = 5                              # the slider's min_samples
SAMPLES = [1, 2, 5, 10, 20]                  # the values of min_samples in exercise 2, with eps 0.4
X_RANGE, Y_RANGE = (-3.3, 6.6), (-3.4, 3.6)
PARTS = ["ring", "clump", "band", "stray"]


# --- the course's own core points, as the lesson writes them ---------------------------------------------------------

def core_points(X, eps, min_samples):
    import numpy as np
    core = []
    for x in X:
        distances = np.linalg.norm(X - x, axis=1)
        count = np.sum(distances <= eps)
        core.append(count >= min_samples)
    return np.array(core)


# --- the points ------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def shapes() -> dict:
    """The 400 points, made by the course from SEED: 200 on a ring of radius 2, 80 in a clump at its centre, 100 on a
    curved band to its right and 20 strays anywhere in the box. `part` says which of the four each point is (0 to 3).
    Read-only."""
    import numpy as np
    rng = np.random.default_rng(SEED)
    a = rng.uniform(0, 2 * np.pi, 200)
    r = 2.0 + rng.normal(0, 0.08, 200)
    ring = np.column_stack([r * np.cos(a), r * np.sin(a)])
    clump = rng.normal(0, 0.25, (80, 2))
    t = rng.uniform(np.pi * 1.15, np.pi * 1.85, 100)
    rr = 1.6 + rng.normal(0, 0.08, 100)
    band = np.column_stack([4.6 + rr * np.cos(t), 1.2 + rr * np.sin(t)])
    stray = np.column_stack([rng.uniform(-3, 6.5, 20), rng.uniform(-3, 3, 20)])
    out = {"X": np.vstack([ring, clump, band, stray]), "part": np.array([0] * 200 + [1] * 80 + [2] * 100 + [3] * 20)}
    for v in out.values():
        v.flags.writeable = False
    return out


@functools.lru_cache(maxsize=None)
def fit(eps: float, min_samples: int = MIN_SAMPLES) -> dict:
    """DBSCAN on the points: the label of every point (−1 for noise), which points are core points, how many clusters
    and noise points there are, and the cluster sizes from large to small."""
    import numpy as np
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(shapes()["X"])
    labels = db.labels_.copy()
    core = np.zeros(len(labels), dtype=bool)
    core[db.core_sample_indices_] = True
    sizes = sorted((int(v) for v in np.bincount(labels[labels >= 0])), reverse=True) if (labels >= 0).any() else []
    for v in (labels, core):
        v.flags.writeable = False
    return {"labels": labels, "core": core, "clusters": len(sizes), "noise": int((labels == -1).sum()), "sizes": sizes}


@functools.lru_cache(maxsize=None)
def others() -> dict:
    """k-means with k = 3 and single linkage cut into 3 clusters on the same points: the cluster of every point."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.cluster import KMeans
    X = shapes()["X"]
    out = {"kmeans": KMeans(n_clusters=3, n_init=10, random_state=0).fit(X).labels_,
           "single": fcluster(linkage(X, "single"), 3, criterion="maxclust") - 1}
    for v in out.values():
        v.flags.writeable = False
    return out


def by_size(labels) -> dict:
    """{cluster: colour}: the three largest clusters get the colours 0, 1 and 2, every other cluster colour 3."""
    import numpy as np
    found = [int(c) for c in np.unique(labels) if c >= 0]
    order = sorted(found, key=lambda c: (-int((labels == c).sum()), c))
    return {c: min(i, 3) for i, c in enumerate(order)}


def number(x: float, lang: str, digits: int = 1) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def listed(values: list, lang: str) -> str:
    """1, 2 and 3; in Dutch 1, 2 en 3."""
    words = [str(v) for v in values]
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + (" and " if lang == "en" else " en ") + words[-1]


# --- the pictures ----------------------------------------------------------------------------------------------------

def _panel(title: str, x: str, y: str) -> chart.Panel:
    return chart.Panel(X_RANGE, Y_RANGE, title=title, xlabel=x, ylabel=y, xticks=[-2, 0, 2, 4, 6], yticks=[-3, -2, -1, 0, 1, 2, 3])


def _draw(p: chart.Panel, labels, core=None) -> None:
    """The points in the colour and shape of their cluster: core points large, the other points of a cluster small,
    noise (label −1) as small open grey dots."""
    X = shapes()["X"]
    colour = by_size(labels)
    noise = labels == -1
    if noise.any():
        p.points(X[noise, 0], X[noise, 1], "ink-3", r=2.4, hollow=True)
    for c, k in sorted(colour.items(), key=lambda kv: -kv[1]):
        rows = labels == c
        if core is None:
            p.group(X[rows, 0], X[rows, 1], k, r=2.8)
        else:
            p.group(X[rows & core, 0], X[rows & core, 1], k, r=3.0)
            if (rows & ~core).any():
                p.group(X[rows & ~core, 0], X[rows & ~core, 1], k, r=1.7)


def centre() -> int:
    """The point of the ring the dashed circle is drawn around: the ring point nearest to (−2, 0)."""
    import numpy as np
    s = shapes()
    d = np.linalg.norm(s["X"] - np.array([-2.0, 0.0]), axis=1)
    d[s["part"] != 0] = np.inf
    return int(np.argmin(d))


def picture(eps: float, labels_: dict) -> str:
    """DBSCAN with this eps and min_samples 5: every cluster in its colour and shape (the three largest in their own,
    the others in the fourth), core points large, border points small, noise as open grey dots; a dashed circle of
    radius eps around one point of the ring. Drawn as SVG text, once for every stop of the slider (L.explore)."""
    import numpy as np
    f = fit(eps)
    p = _panel(labels_["title"], labels_["x"], labels_["y"])
    _draw(p, f["labels"], f["core"])
    cx, cy = shapes()["X"][centre()]
    t = np.linspace(0, 2 * np.pi, 49)
    p.line(cx + eps * np.cos(t), cy + eps * np.sin(t), "hyper", 1.6, dashed=True)
    return chart.figure([p], alt=labels_["alt"], width=340, height=270)


def picture_others(labels_: dict) -> str:
    """k-means with k = 3 and single linkage cut into 3 clusters, side by side, each point in the colour and shape of
    its cluster."""
    o = others()
    panels = []
    for key in ("kmeans", "single"):
        p = _panel(labels_[key], labels_["x"], labels_["y"])
        _draw(p, o[key])
        panels.append(p)
    return chart.figure(panels, alt=labels_["alt"], width=330, height=260)


# The lesson is done when these are correct. "single" is the choice question in the lesson.
REQUIRED = ["core-points", "min-samples", "single"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


# --- exercise 1: core_points -----------------------------------------------------------------------------------------

class _Val:
    """A test value that the check's messages write by its name, such as `X`; the function gets an array."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import numpy as np
        return np.array(self.value, dtype=float)


def _shown(v) -> str:
    import numpy as np
    if isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, bool):
        return str(round(float(v), 3))
    if isinstance(v, np.ndarray) and v.dtype.kind in "iuf":
        return str(np.round(v.astype(float), 3).tolist())
    if isinstance(v, np.ndarray) and v.dtype.kind == "b":
        return str(v.tolist())
    return repr(v)


# Eight points with a clump of five; a grid where neighbours lie exactly eps apart; the numbers of the worked example;
# min_samples 1; three features.
_P = _Val("X", [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0, 0.0], [3.0, 1.0], [3.4, 1.2], [5.0, 0.0]])
_GRID = _Val("grid", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
_LINE = _Val("line", [[1.0], [2.0], [4.0], [8.0], [9.0], [15.0]])
_THREE = _Val("three", [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.2, 0.1, 0.0], [5.0, 5.0, 5.0]])

_C_HEAD = "import numpy as np\n\n\ndef core_points(X, eps, min_samples):\n    core = []\n    for x in X:\n        distances = np.linalg.norm(X - x, axis=1)\n"


def _c(count: str = "np.sum(distances <= eps)", add: str = "count >= min_samples") -> str:
    return _C_HEAD + f"        count = {count}\n        core.append({add})\n    return np.array(core)\n"


_C_START = _c("...", "...")
_C = _c()

_C_WORDS = {
    "title": {"en": "Your core_points of X, eps 0.6, min_samples 3", "nl": "Jouw core_points van X, eps 0,6, min_samples 3"},
    "said": {"en": "Large: the points your function calls core points. Dashed: a circle of radius eps around each of them. The course's core points are the first four, the square of points on the left; your function finds {mine}.",
             "nl": "Groot: de punten die jouw functie core points noemt. Gestreept: een cirkel met straal eps om elk ervan. De core points van de course zijn de eerste vier, het vierkant van punten links; jouw functie vindt er {mine}."},
    "right": {"en": "Large: the four core points, each with at least 3 points within the dashed circle of radius eps around it, itself included. Small: the four other points.",
              "nl": "Groot: de vier core points, elk met minstens 3 punten binnen de gestreepte cirkel met straal eps eromheen, zichzelf meegeteld. Klein: de vier andere punten."},
}


def _cp_picture(seen) -> str:
    """The visible case: the eight points, the ones the learner's function calls core points large, with a dashed
    circle of radius eps around each, the others small."""
    import numpy as np
    ex = seen.exercise
    mine = seen.mine
    if mine is None:
        return ""
    fn = mine.names.get(ex.function)
    if not callable(fn):
        return ""
    X = np.array(_P.value)
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = checking.with_time_limit(lambda: fn(X.copy(), 0.6, 3))
        got = np.asarray(got)
        if got.shape != (len(X),) or got.dtype.kind not in "bi":
            return ""
        got = got.astype(bool)
    except Exception:
        return ""
    want = core_points(X, 0.6, 3)
    lang = seen.lang
    p = chart.Panel((-0.8, 5.6), (-0.9, 1.9), xticks=[0, 1, 2, 3, 4, 5], yticks=[0, 1])
    t = np.linspace(0, 2 * np.pi, 49)
    for x in X[got]:
        p.line(x[0] + 0.6 * np.cos(t), x[1] + 0.6 * np.sin(t), "hyper", 1.0, dashed=True)
    if got.any():
        p.group(X[got, 0], X[got, 1], 0, r=5.5)
    if (~got).any():
        p.group(X[~got, 0], X[~got, 1], 0, r=2.6)
    right = bool((got == want).all())
    words = {k: v[lang] for k, v in _C_WORDS.items()}
    text = words["right"] if right else words["said"].format(mine=int(got.sum()))
    key = ('<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.15rem">'
           + text + "</div>")
    return yours.figure(words["title"], chart.figure([p], alt=text, width=340, height=200) + key, alt=text)


EXERCISES = {
    "core-points": FunctionBox(
        name="core-points",
        picture=_cp_picture,
        function="core_points",
        start=_C_START,
        solution=_C,
        show=_shown,
        visible=[(_P, 0.6, 3)],
        hidden=[(_GRID, 1.0, 3), (_LINE, 2.0, 2), (_LINE, 2.0, 3), (_P, 0.6, 1), (_THREE, 1.0, 2)],
        first=[_DOTS],
        mistakes=[
            Mistake("strict", wrong=_c(count="np.sum(distances < eps)"),
                    en="`{call}` gives {got}, and it should give {want}. In `grid` the neighbors lie exactly `eps` apart, and they count: a point counts when its distance is at most `eps`, so `<=`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. In `grid` liggen de buren precies `eps` uit elkaar, en die tellen mee: een punt telt als zijn afstand hoogstens `eps` is, dus `<=`."),
            Mistake("one-more", wrong=_c(add="count > min_samples"),
                    en="`{call}` gives {got}: your function asks for one point more than `min_samples`. A point with exactly `min_samples` points within `eps`, itself included, is a core point: count the point itself, as scikit-learn does, and compare with `>=`.",
                    nl="`{call}` geeft {got}: je functie vraagt één punt meer dan `min_samples`. Een punt met precies `min_samples` punten binnen `eps`, zichzelf meegeteld, is een core point: tel het punt zelf mee, zoals scikit-learn, en vergelijk met `>=`."),
            Mistake("count", wrong=_c(add="count"),
                    en="`{call}` gives {got}: the counts themselves. The function should say for every point whether it is a core point: `True` or `False`, with a comparison.",
                    nl="`{call}` geeft {got}: de tellingen zelf. De functie moet voor elk punt zeggen of het een core point is: `True` of `False`, met een vergelijking."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the two `...`: how many points lie within `eps` of `x`, and whether that is enough for a core point.",
                      "nl": "Vervang de twee `...`: hoeveel punten binnen `eps` van `x` liggen, en of dat genoeg is voor een core point."},
            "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has a different number of names between its parentheses. Keep `(X, eps, min_samples)`.",
                          "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X, eps, min_samples)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Do you count the distances that are at most `eps`, and compare that count with `min_samples`?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Tel je de afstanden die hoogstens `eps` zijn, en vergelijk je dat aantal met `min_samples`?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for any `eps`, any `min_samples` and any number of features?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke `eps`, elke `min_samples` en elk aantal features?"},
            "correct": {"en": "Your function is right in all {n} cases. For `line`, the numbers of the worked example, with `eps` 2: with `min_samples` 2 every number but 15 is a core point, with 3 only the 2, as scikit-learn found.",
                        "nl": "Je functie klopt in alle {n} gevallen. Voor `line`, de getallen uit het uitgewerkte voorbeeld, met `eps` 2: met `min_samples` 2 is elk getal behalve 15 een core point, met 3 alleen de 2, zoals scikit-learn vond."},
        },
    ),
}


# --- exercise 2: clusters and noise for five values of min_samples ---------------------------------------------------

def _m(noise: str = "int(np.sum(labels == -1))", clusters: str = "len(np.unique(labels[labels >= 0]))") -> str:
    return ("import numpy as np\nfrom sklearn.cluster import DBSCAN\n\nfound = {}\nfor min_samples in [1, 2, 5, 10, 20]:\n"
            "    labels = DBSCAN(eps=0.4, min_samples=min_samples).fit(X).labels_\n"
            f"    n_noise = {noise}\n    n_clusters = {clusters}\n    found[min_samples] = [n_clusters, n_noise]\nfound\n")


_M_START = _m("...", "...")
_M = _m()


def _names() -> dict:
    """What the box may use without making it: the 400 points of the picture."""
    return {"X": shapes()["X"]}


EXERCISES["min-samples"] = CodeBox(
    name="min-samples",
    picture=yours.outcome_bars(SAMPLES, title={"en": "Your clusters and noise per min_samples, eps 0.4", "nl": "Jouw clusters en ruis per min_samples, eps 0,4"},
                               xlabel="min_samples", panels=[{"en": "clusters", "nl": "clusters"}, {"en": "noise points", "nl": "noise points"}]),
    start=_M_START,
    solution=_M,
    names=_names(),
    keep=["for min_samples in [1, 2, 5, 10, 20]:"],
    first=[_DOTS],
    mistakes=[
        Mistake("noise-cluster", wrong=_m(clusters="len(np.unique(labels))"),
                en="Your numbers of clusters are one too high wherever there is noise: `np.unique(labels)` also holds −1, and noise is not a cluster. Count only the labels from 0 up.",
                nl="Je aantallen clusters zijn één te hoog overal waar ruis is: `np.unique(labels)` bevat ook −1, en ruis is geen cluster. Tel alleen de labels vanaf 0."),
        Mistake("max", wrong=_m(clusters="max(labels)"),
                en="Your numbers of clusters are one too low: the clusters are numbered from 0, so the highest label is one less than the number of clusters.",
                nl="Je aantallen clusters zijn één te laag: de clusters zijn genummerd vanaf 0, dus het hoogste label is één minder dan het aantal clusters."),
        Mistake("all", wrong=_m(noise="len(labels == -1)"),
                en="Every `n_noise` is 400: `labels == -1` has a True or False for every point, so its length is always 400. Count only the Trues, with `np.sum`.",
                nl="Elke `n_noise` is 400: `labels == -1` heeft een True of False voor elk punt, dus de lengte is altijd 400. Tel alleen de Trues, met `np.sum`."),
    ],
    texts={
        **_SAME,
        "empty": {"en": "Replace the two `...`: how many points have the label −1, and how many clusters there are.",
                  "nl": "Vervang de twee `...`: hoeveel punten het label −1 hebben, en hoeveel clusters er zijn."},
        "no_value": {"en": "The last line of the box shows nothing. Put `found` on the last line, so the check can see it.",
                     "nl": "De laatste regel van het vak laat niets zien. Zet `found` op de laatste regel, zodat de check hem kan zien."},
        "wrong": {"en": "Your numbers are not those of the course. Is `n_noise` the number of labels that are −1, and `n_clusters` the number of different labels from 0 up?",
                  "nl": "Je getallen zijn niet die van de course. Is `n_noise` het aantal labels dat −1 is, en `n_clusters` het aantal verschillende labels vanaf 0?"},
        "correct": {"en": "With `min_samples` 1 every point is a core point: 16 clusters and no noise, 13 of them a stray or two on their own. With 2, still 5 clusters; with 5 the three shapes and 15 noise points. With 10 the ring breaks into three pieces, and with 20 only the clump stays whole: 198 noise points.",
                    "nl": "Met `min_samples` 1 is elk punt een core point: 16 clusters en geen ruis, waarvan 13 met maar een of twee losse punten. Met 2 nog steeds 5 clusters; met 5 de drie vormen en 15 noise points. Met 10 breekt de ring in drie stukken, en met 20 blijft alleen de klont heel: 198 noise points."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("core-points", _C_START, "empty", None),
    ("core-points", _c(add="..."), "wrong", "dots-left"),
    ("core-points", _C, "correct", None),
    ("core-points", _c(count="len(distances[distances <= eps])"), "correct", None),
    ("core-points", _c(count="sum(distances <= eps)"), "correct", None),
    ("core-points", _c(add="bool(count >= min_samples)"), "correct", None),
    ("core-points", _c(count="np.sum(distances < eps)"), "wrong", "strict"),
    ("core-points", _c(count="np.sum(distances <= eps) - 1"), "wrong", "one-more"),
    ("core-points", _c(add="count > min_samples"), "wrong", "one-more"),
    ("core-points", _c(add="count"), "wrong", "count"),
    ("core-points", _c(count="np.sum(distances <= eps)", add="count >= 3"), "almost", None),
    ("min-samples", _M_START, "empty", None),
    ("min-samples", _m(clusters="..."), "wrong", "dots-left"),
    ("min-samples", _M, "correct", None),
    ("min-samples", _m(noise="np.sum(labels == -1)", clusters="len(set(labels)) - (1 if -1 in labels else 0)"), "correct", None),
    ("min-samples", _m(clusters="max(labels) + 1"), "correct", None),
    ("min-samples", _m(clusters="len(np.unique(labels))"), "wrong", "noise-cluster"),
    ("min-samples", _m(clusters="len(set(labels))"), "wrong", "noise-cluster"),
    ("min-samples", _m(clusters="max(labels)"), "wrong", "max"),
    ("min-samples", _m(noise="len(labels == -1)"), "wrong", "all"),
]
