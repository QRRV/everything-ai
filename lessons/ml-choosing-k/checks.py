# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-choosing-k: how many clusters k-means should make, on the Palmer penguins by bill length
and flipper length. The within-cluster sum of squares by hand, k-means from ten starts and the worse solution one of
them ends in, and k chosen by the silhouette score. And the picture: for every k, every penguin joined to its
centroid, and the within-cluster sum of squares of every k."""
import functools
import re

from everything_ai.check import Code, CodeBox, Mistake

DATA = "data/penguins.csv"   # the Palmer penguins (CC0): see data/README.md
TWO = ["bill_length_mm", "flipper_length_mm"]
KS = list(range(1, 9))       # the stops of the slider
SILHOUETTE_KS = list(range(2, 9))   # the silhouette needs at least two clusters
SEEDS = list(range(10))      # the starts of exercise 2: np.random.default_rng(seed).permutation(342)[:3]


@functools.lru_cache(maxsize=1)
def penguins() -> dict:
    """The 342 penguins with all four measurements, by bill length and flipper length, standardized with their own
    mean and standard deviation, as in ml-kmeans; and their species. Read-only."""
    import numpy as np
    import pandas as pd
    from everything_ai import paths
    df = pd.read_csv(paths.ROOT / DATA)
    df = df[df["flipper_length_mm"].isna() == False]   # noqa: E712 (the lesson's own way, with what the learner knows)
    raw = np.array(df[TWO])
    out = {"raw": raw, "X": (raw - raw.mean(axis=0)) / raw.std(axis=0), "species": np.array(df["species"])}
    for a in out.values():
        a.flags.writeable = False
    return out


@functools.lru_cache(maxsize=None)
def fit(k: int) -> dict:
    """KMeans(n_clusters=k, n_init=10, random_state=0) on the standardized penguins: centroids, clusters, the within-
    cluster sum of squares and, from k = 2, the silhouette score."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    X = penguins()["X"]
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    return {"centers": km.cluster_centers_, "labels": km.labels_, "wcss": float(km.inertia_),
            "silhouette": float(silhouette_score(X, km.labels_)) if k > 1 else None}


def by_hand(X, start):
    """k-means by hand from these starting rows until nothing changes, as exercise 2 writes it: the within-cluster
    sum of squares it ends with."""
    import numpy as np
    centers = X[start]
    labels = np.zeros(len(X)) - 1
    while True:
        new = np.argmin(np.array([np.linalg.norm(X - c, axis=1) for c in centers]), axis=0)
        if np.sum(new != labels) == 0:
            break
        labels = new
        centers = np.array([X[labels == j].mean(axis=0) for j in range(len(centers))])
    return float(np.sum((X - centers[labels]) ** 2))


# --- the picture ------------------------------------------------------------------------------------------------------

def picture(k: int, labels: dict[str, str]) -> str:
    """Left: the penguins by bill length and flipper length, standardized, each joined by a thin line to the centroid
    of its cluster, and the k centroids in orange with a dark ring, the colour of what is learned. The within-cluster
    sum of squares is the sum of the squared lengths of those lines. Right: that sum for every k, in purple, the colour
    of what a model makes small, with this k marked. Drawn as SVG text, once for every stop of the slider
    (L.explore); dragging across the right chart sets k."""
    from everything_ai import chart
    X = penguins()["X"]
    f = fit(k)
    centers = f["centers"]
    p = chart.Panel((-2.4, 3.1), (-2.3, 2.4), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=[-2, -1, 0, 1, 2, 3], yticks=[-2, -1, 0, 1, 2])
    for (x, y), j in zip(X, f["labels"]):
        p.line([x, centers[j][0]], [y, centers[j][1]], "ink-3", 0.7)
    p.points(X[:, 0], X[:, 1], "ink-2", 1.9)
    p.points(centers[:, 0], centers[:, 1], "ink", 7.0)
    p.points(centers[:, 0], centers[:, 1], "param", 5.4)
    sums = [fit(j)["wcss"] for j in KS]
    e = chart.Panel((0.5, 8.5), (0, 720), title=labels["every"], xlabel=labels["k"], ylabel=labels["wcss"],
                    xticks=KS, yticks=[0, 200, 400, 600])
    e.line([k, k], [0, sums[k - 1]], "ink-3", 1.2)
    e.line(KS, sums, "loss", 2.0)
    e.points(KS, sums, "loss", 3.2, hollow=True)
    e.points([k], [sums[k - 1]], "loss", 6.0)
    return chart.figure([p, e], alt=labels["alt"], width=330, height=300, drag=1)


# The lesson is done when these are correct. "which-k" is the choice question in the lesson.
REQUIRED = ["wcss", "starts", "best-k", "which-k"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

def _broadcast(e) -> bool:
    return isinstance(e, ValueError) and "could not be broadcast" in str(e)


_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_HEAD = ("import numpy as np\nimport pandas as pd\nfrom sklearn.cluster import KMeans\n\n"
         "df = pd.read_csv(\"data/penguins.csv\")\ndf = df[df[\"flipper_length_mm\"].isna() == False]\n"
         "raw = np.array(df[[\"bill_length_mm\", \"flipper_length_mm\"]])\n"
         "X = (raw - raw.mean(axis=0)) / raw.std(axis=0)\n\n")


# --- exercise 1: the within-cluster sum of squares by hand ----------------------------------------------------------

_HEAD4 = ("import numpy as np\nimport pandas as pd\nfrom sklearn.cluster import KMeans\n\n"
          "df = pd.read_csv(\"data/penguins.csv\")\ndf = df[df[\"flipper_length_mm\"].isna() == False]\n"
          "four = np.array(df[[\"bill_length_mm\", \"bill_depth_mm\", \"flipper_length_mm\", \"body_mass_g\"]])\n"
          "X4 = (four - four.mean(axis=0)) / four.std(axis=0)\n\n")
_KM = "km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X4)"
_W = "np.sum((X4 - km.cluster_centers_[km.labels_]) ** 2)"


def _w(wcss: str = _W) -> str:
    return _HEAD4 + f"{_KM}\nwcss = {wcss}\nwcss, km.inertia_\n"


_W_START = _w("...")
_W_SOLUTION = _w()

EXERCISES = {
    "wcss": CodeBox(
        name="wcss",
        start=_W_START,
        solution=_W_SOLUTION,
        keep=[_KM],
        others=[{"km": Code("KMeans(n_clusters=5, n_init=10, random_state=0).fit(X4)")}],
        digits=1,
        first=[_DOTS],
        mistakes=[
            Mistake("distances", wrong=_w("np.sum(np.linalg.norm(X4 - km.cluster_centers_[km.labels_], axis=1))"),
                    en="Your sum is not `km.inertia_`: you added the distances themselves. The within-cluster sum of squares adds their squares, so square every difference before you add.",
                    nl="Je som is niet `km.inertia_`: je telde de afstanden zelf op. De within-cluster sum of squares telt hun kwadraten op, dus kwadrateer elk verschil voordat je optelt."),
            Mistake("not-squared", wrong=_w("np.sum(X4 - km.cluster_centers_[km.labels_])"),
                    en="Your sum is about 0: the differences above and below a centroid cancel out, because the centroid is their mean. Square them first with `** 2`.",
                    nl="Je som is ongeveer 0: de verschillen boven en onder een centroid vallen tegen elkaar weg, omdat de centroid hun gemiddelde is. Kwadrateer ze eerst met `** 2`."),
            Mistake("mean", wrong=_w("np.mean((X4 - km.cluster_centers_[km.labels_]) ** 2)"),
                    en="You took the mean of the squares, over every penguin and all four features. The within-cluster sum of squares is their sum: `np.sum`.",
                    nl="Je nam het gemiddelde van de kwadraten, over elke pinguïn en alle vier de features. De within-cluster sum of squares is hun som: `np.sum`."),
            Mistake("all-centroids", error=_broadcast, wrong=_w("np.sum((X4 - km.cluster_centers_) ** 2)"),
                    en="Your code stops with an error: `km.cluster_centers_` has 3 rows and `X4` has 342. `km.cluster_centers_[km.labels_]` gives every penguin the centroid of its own cluster.",
                    nl="Je code stopt met een fout: `km.cluster_centers_` heeft 3 rijen en `X4` heeft er 342. `km.cluster_centers_[km.labels_]` geeft elke pinguïn de centroid van zijn eigen cluster."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: the sum, over all penguins and all four features, of the squared difference between a penguin and the centroid of its cluster.",
                      "nl": "Vervang de `...`: de som, over alle pinguïns en alle vier de features, van het gekwadrateerde verschil tussen een pinguïn en de centroid van zijn cluster."},
            "no_value": {"en": "The last line of the box shows nothing. Put `wcss, km.inertia_` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `wcss, km.inertia_` op de laatste regel, zodat de check hem kan zien."},
            "wrong": {"en": "Your sum is not `km.inertia_`. Does `km.cluster_centers_[km.labels_]` give every penguin its centroid, and do you square the differences and add them all with `np.sum`?",
                      "nl": "Je som is niet `km.inertia_`. Geeft `km.cluster_centers_[km.labels_]` elke pinguïn zijn centroid, en kwadrateer je de verschillen en tel je ze allemaal op met `np.sum`?"},
            "other": {"en": "Right for k = 3. But with {where}, your sum is not `km.inertia_`. Do you use `km` and `X4`?",
                      "nl": "Goed voor k = 3. Maar met {where} is je som niet `km.inertia_`. Gebruik je `km` en `X4`?"},
            "correct": {"en": "379.4, the same as `km.inertia_`. It is larger than the 157.8 with two measurements: every penguin now has four differences from its centroid instead of two. So compare the sums of different k on the same features only.",
                        "nl": "379,4, hetzelfde als `km.inertia_`. Dat is meer dan de 157,8 met twee metingen: elke pinguïn heeft nu vier verschillen met zijn centroid in plaats van twee. Vergelijk de sommen van verschillende k dus alleen op dezelfde features."},
        },
    ),
}


# --- exercise 2: ten starts -----------------------------------------------------------------------------------------

_BY_HAND = ("def by_hand(X, start):\n    centers = X[start]\n    labels = np.zeros(len(X)) - 1\n    while True:\n"
            "        dists = []\n        for c in centers:\n            dists.append(np.linalg.norm(X - c, axis=1))\n"
            "        new = np.argmin(np.array(dists), axis=0)\n        if np.sum(new != labels) == 0:\n            break\n"
            "        labels = new\n        means = []\n        for j in range(len(centers)):\n"
            "            means.append(X[labels == j].mean(axis=0))\n        centers = np.array(means)\n"
            "    return np.sum((X - centers[labels]) ** 2)\n\n")


def _st(start: str = "np.random.default_rng(seed).permutation(len(X))[:3]", result: str = "by_hand(X, start)") -> str:
    return (_HEAD + _BY_HAND + "results = []\nfor seed in range(10):\n" f"    start = {start}\n"
            f"    results.append({result})\nmin(results), max(results)\n")


_ST_START = _st("...", "...")
_ST = _st()

EXERCISES["starts"] = CodeBox(
    name="starts",
    start=_ST_START,
    solution=_ST,
    digits=1,
    first=[_DOTS],
    mistakes=[
        Mistake("same-start", wrong=_st("np.random.default_rng(0).permutation(len(X))[:3]"),
                en="All ten of your starts are the same: `default_rng(0)` draws the same penguins every time. Give it the `seed` of the loop.",
                nl="Al je tien starts zijn hetzelfde: `default_rng(0)` trekt elke keer dezelfde pinguïns. Geef het de `seed` van de loop."),
        Mistake("four-clusters", wrong=_st("np.random.default_rng(seed).permutation(len(X))[:4]"),
                en="Your starts have four centroids, so k-means makes four clusters. Take the first three of the shuffled rows: `[:3]`.",
                nl="Je starts hebben vier centroids, dus maakt k-means vier clusters. Neem de eerste drie van de geschudde rijen: `[:3]`."),
    ],
    texts={
        **_TEXTS,
        "empty": {"en": "Replace the two `...`: three rows picked at random with this `seed`, and the within-cluster sum of squares k-means by hand ends with from them.",
                  "nl": "Vervang de twee `...`: drie willekeurige rijen met deze `seed`, en de within-cluster sum of squares waarop k-means met de hand vanaf die rijen eindigt."},
        "no_value": {"en": "The last line of the box shows nothing. Put `min(results), max(results)` on the last line, so the check can see it.",
                     "nl": "De laatste regel van het vak laat niets zien. Zet `min(results), max(results)` op de laatste regel, zodat de check hem kan zien."},
        "wrong": {"en": "Your ten results are not those of the course. Is every `start` `np.random.default_rng(seed).permutation(len(X))[:3]`, and do you add `by_hand(X, start)` to `results`?",
                  "nl": "Je tien uitkomsten zijn niet die van de course. Is elke `start` `np.random.default_rng(seed).permutation(len(X))[:3]`, en voeg je `by_hand(X, start)` toe aan `results`?"},
        "correct": {"en": "157.8 and 206.1. Nine of the ten starts end in the solution of `KMeans`, with 157.8; the start of seed 0 ends at 206.1, where no penguin changes cluster any more either. Had you run k-means from that one start only, you would have kept 206.1.",
                    "nl": "157,8 en 206,1. Negen van de tien starts eindigen in de oplossing van `KMeans`, met 157,8; de start van seed 0 eindigt op 206,1, waar ook geen enkele pinguïn meer van cluster verandert. Had je k-means alleen vanaf die ene start gedraaid, dan had je 206,1 gehouden."},
    },
)


# --- exercise 3: k by the silhouette score --------------------------------------------------------------------------

def _b(score: str = "silhouette_score(X, km.labels_)", ks: str = "range(2, 9)") -> str:
    return (_HEAD.replace("from sklearn.cluster import KMeans\n", "from sklearn.cluster import KMeans\nfrom sklearn.metrics import silhouette_score\n")
            + f"best = 0\ntop = -1\nfor k in {ks}:\n    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)\n"
            f"    score = {score}\n    if score > top:\n        best = k\n        top = score\nbest, top\n")


_B_START = _b("...")
_B = _b()


def _one_label(e) -> bool:
    return isinstance(e, ValueError) and "Number of labels is 1" in str(e)


EXERCISES["best-k"] = CodeBox(
    name="best-k",
    start=_B_START,
    solution=_B,
    digits=3,
    first=[_DOTS],
    mistakes=[
        Mistake("by-wcss", wrong=_b("km.inertia_"),
                en="You chose by the within-cluster sum of squares, and the largest is that of the smallest k: fewer clusters, larger sum. Score every k with `silhouette_score(X, km.labels_)`.",
                nl="Je koos op de within-cluster sum of squares, en de grootste is die van de kleinste k: minder clusters, grotere som. Geef elke k een score met `silhouette_score(X, km.labels_)`."),
        Mistake("by-small-wcss", wrong=_b("-km.inertia_"),
                en="You chose the smallest within-cluster sum of squares, and that is the largest k you try: more clusters, smaller sum. Score every k with `silhouette_score(X, km.labels_)`.",
                nl="Je koos de kleinste within-cluster sum of squares, en dat is de grootste k die je probeert: meer clusters, kleinere som. Geef elke k een score met `silhouette_score(X, km.labels_)`."),
        Mistake("from-one", error=_one_label, wrong=_b(ks="range(1, 9)"),
                en="Your code stops with an error: with k = 1 there is no other cluster, so there is no silhouette. Start the loop at 2: `range(2, 9)`.",
                nl="Je code stopt met een fout: met k = 1 is er geen ander cluster, dus geen silhouette. Begin de loop bij 2: `range(2, 9)`."),
    ],
    texts={
        **_TEXTS,
        "empty": {"en": "Replace the `...`: the silhouette score of the clusters of this `km`.",
                  "nl": "Vervang de `...`: de silhouette score van de clusters van deze `km`."},
        "no_value": {"en": "The last line of the box shows nothing. Put `best, top` on the last line, so the check can see it.",
                     "nl": "De laatste regel van het vak laat niets zien. Zet `best, top` op de laatste regel, zodat de check hem kan zien."},
        "wrong": {"en": "Your best k is not that of the course. Is `score` `silhouette_score(X, km.labels_)`, for every k from 2 to 8?",
                  "nl": "Je beste k is niet die van de course. Is `score` `silhouette_score(X, km.labels_)`, voor elke k van 2 tot en met 8?"},
        "correct": {"en": "k = 2, with 0.543: just above k = 3, with 0.522, and k = 4, with 0.507. By the silhouette, two clusters are the most clearly apart: mostly the Adelies against the rest.",
                    "nl": "k = 2, met 0,543: net boven k = 3, met 0,522, en k = 4, met 0,507. Volgens de silhouette score liggen twee clusters het duidelijkst uit elkaar: vooral de Adelies tegen de rest."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("wcss", _W_START, "empty", None),
    ("wcss", _W_SOLUTION, "correct", None),
    ("wcss", _w("((X4 - km.cluster_centers_[km.labels_]) ** 2).sum()"), "correct", None),
    ("wcss", _w("np.sum(np.linalg.norm(X4 - km.cluster_centers_[km.labels_], axis=1) ** 2)"), "correct", None),
    ("wcss", _w("np.sum(np.linalg.norm(X4 - km.cluster_centers_[km.labels_], axis=1))"), "wrong", "distances"),
    ("wcss", _w("np.sum(X4 - km.cluster_centers_[km.labels_])"), "wrong", "not-squared"),
    ("wcss", _w("np.mean((X4 - km.cluster_centers_[km.labels_]) ** 2)"), "wrong", "mean"),
    ("wcss", _w("np.sum((X4 - km.cluster_centers_) ** 2)"), "error", "all-centroids"),
    ("wcss", _w("379.4"), "wrong", None),
    ("starts", _ST_START, "empty", None),
    ("starts", _ST, "correct", None),
    ("starts", _st(result="float(by_hand(X, start))"), "correct", None),
    ("starts", _st("np.random.default_rng(0).permutation(len(X))[:3]"), "wrong", "same-start"),
    ("starts", _st("np.random.default_rng(seed).permutation(len(X))[:4]"), "wrong", "four-clusters"),
    ("best-k", _B_START, "empty", None),
    ("best-k", _B, "correct", None),
    ("best-k", _b("km.inertia_"), "wrong", "by-wcss"),
    ("best-k", _b("-km.inertia_"), "wrong", "by-small-wcss"),
    ("best-k", _b(ks="range(1, 9)"), "error", "from-one"),
]
