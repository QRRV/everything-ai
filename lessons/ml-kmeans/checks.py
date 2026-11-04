# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-kmeans: k-means on the Palmer penguins without their species. One step by hand (every
penguin to its nearest centroid, every centroid to the mean of its penguins), the steps repeated until nothing
changes, scikit-learn's KMeans on all four measurements, and why four measurements match the species less well than
two. And the picture: the clusters and centroids after every step, next to the species k-means never saw."""
import functools
import re

from everything_ai.check import Code, CodeBox, Mistake

DATA = "data/penguins.csv"   # the Palmer penguins (CC0): see data/README.md
TWO = ["bill_length_mm", "flipper_length_mm"]
FOUR = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]   # in the order np.unique sorts them: the group colours of the picture
START = [216, 54, 240]   # the starting centroids: np.random.default_rng(37).choice(342, 3, replace=False), two Gentoos and an Adelie
OTHER_START = [0, 151, 274]   # another start, for the checks: one penguin of each species
GROUP = [2, 0, 1]        # the group colour of each cluster: the species most of its penguins have at the end


@functools.lru_cache(maxsize=1)
def penguins() -> dict:
    """The 342 penguins with all four measurements, in the order of the file: the two measurements of the lesson in
    millimetres and standardized with their own mean and standard deviation, the four standardized, and the species.
    Read-only."""
    import numpy as np
    import pandas as pd
    from everything_ai import paths
    df = pd.read_csv(paths.ROOT / DATA)
    df = df[df["flipper_length_mm"].isna() == False]   # noqa: E712 (the lesson's own way, with what the learner knows)
    raw, four = np.array(df[TWO]), np.array(df[FOUR])
    out = {"raw": raw, "X": (raw - raw.mean(axis=0)) / raw.std(axis=0), "mean": raw.mean(axis=0), "std": raw.std(axis=0),
           "four": four, "X4": (four - four.mean(axis=0)) / four.std(axis=0), "species": np.array(df["species"])}
    for a in out.values():
        a.flags.writeable = False
    return out


def assign(X, centers):
    """Every example to its nearest centroid, as the lesson writes it."""
    import numpy as np
    return np.argmin(np.array([np.linalg.norm(X - c, axis=1) for c in centers]), axis=0)


def move(X, labels, k):
    """Every centroid to the mean of its examples."""
    import numpy as np
    return np.array([X[labels == j].mean(axis=0) for j in range(k)])


@functools.lru_cache(maxsize=2)
def steps(start: tuple = tuple(START)) -> tuple:
    """k-means by hand from these starting penguins, as in the lesson: for every step the centroids, the cluster of
    every penguin, and how many penguins changed cluster (None at the start). The last step changes nothing."""
    import numpy as np
    X = penguins()["X"]
    centers, labels, out = X[list(start)], None, []
    while True:
        new = assign(X, centers)
        changed = None if labels is None else int(np.sum(new != labels))
        out.append((centers, new, changed))
        if changed == 0:
            return tuple(out)
        labels = new
        centers = move(X, labels, len(start))


def matched(species, labels, k: int = 3) -> int:
    """How many examples are in a cluster whose most common species is their own."""
    import numpy as np
    return int(sum(np.unique(species[labels == j], return_counts=True)[1].max() for j in range(k)))


def kmeans(X, k: int = 3, **settings):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, **({"n_init": 10, "random_state": 0} | settings)).fit(X)


STEPS = list(range(len(steps())))   # the stops of the slider: the start and every step after it


# --- the picture ------------------------------------------------------------------------------------------------------

def picture(step: int, labels: dict[str, str]) -> str:
    """Left: the penguins by bill length and flipper length, a colour and shape per cluster after this step (style
    guide 1.7: here colour and shape mean a group; each cluster gets the colour of the species most of its penguins
    have at the end), and the centroids in orange with a dark ring, the colour of what is learned, with an arrow from where they were
    one step before. Right: the same penguins with their species, which k-means never saw, and the final centroids.
    Drawn as SVG text, once for every stop of the slider (L.explore)."""
    from everything_ai import chart
    p = penguins()
    raw, mean, std = p["raw"], p["mean"], p["std"]
    all_steps = steps()
    centers, assigned, _ = all_steps[step]
    left = chart.Panel((31, 61), (168, 236), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                       xticks=[35, 40, 45, 50, 55, 60], yticks=[170, 190, 210, 230])
    for j in range(len(START)):
        rows = assigned == j
        left.group(raw[rows, 0], raw[rows, 1], GROUP[j], r=2.8)
    mm = centers * std + mean
    if step > 0:
        before = all_steps[step - 1][0] * std + mean
        for (x0, y0), (x1, y1) in zip(before, mm):
            if abs(x1 - x0) + abs(y1 - y0) > 0.3:
                left.arrow(x0, y0, x1, y1, "param", 2.0)
    left.points(mm[:, 0], mm[:, 1], "ink", 7.6)
    left.points(mm[:, 0], mm[:, 1], "param", 6.0)
    right = chart.Panel((31, 61), (168, 236), title=labels["species"], xlabel=labels["x"], ylabel=labels["y"],
                        xticks=[35, 40, 45, 50, 55, 60], yticks=[170, 190, 210, 230])
    for g, name in enumerate(SPECIES):
        rows = p["species"] == name
        right.group(raw[rows, 0], raw[rows, 1], g, r=2.8)
    final = all_steps[-1][0] * std + mean
    right.points(final[:, 0], final[:, 1], "ink", 7.6)
    right.points(final[:, 0], final[:, 1], "param", 6.0)
    return chart.figure([left, right], alt=labels["alt"], width=330, height=300)


# The lesson is done when these are correct. "why-four" is the choice question in the lesson.
REQUIRED = ["one-step", "until-stable", "four", "why-four"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_HEAD = ("import numpy as np\nimport pandas as pd\n\n"
         "df = pd.read_csv(\"data/penguins.csv\")\ndf = df[df[\"flipper_length_mm\"].isna() == False]\n"
         "raw = np.array(df[[\"bill_length_mm\", \"flipper_length_mm\"]])\n"
         "X = (raw - raw.mean(axis=0)) / raw.std(axis=0)\n"
         "centers = X[[216, 54, 240]]\n\n")
_KEEP = ["centers = X[[216, 54, 240]]"]
_OTHER = [{"centers": Code("X[[0, 151, 274]]")}, {"centers": Code("X[[255, 21, 6]]")}]

_NORM = "np.linalg.norm(X - c, axis=1)"
_ARGMIN = "np.argmin(np.array(dists), axis=0)"
_MEAN = "X[labels == j].mean(axis=0)"


# --- exercise 1: one step by hand ---------------------------------------------------------------------------------------

def _s(norm: str = _NORM, argmin: str = _ARGMIN, mean: str = _MEAN) -> str:
    return (_HEAD + "dists = []\nfor c in centers:\n" f"    dists.append({norm})\nlabels = {argmin}\n"
            "new = []\nfor j in range(3):\n" f"    new.append({mean})\ncenters = np.array(new)\n"
            "centers * raw.std(axis=0) + raw.mean(axis=0)\n")


_S_START = _s("...", "...", "...")
_S = _s()


def _error(*words):
    return lambda e: any(w in str(e) for w in words)


EXERCISES = {
    "one-step": CodeBox(
        name="one-step",
        start=_S_START,
        solution=_S,
        keep=_KEEP,
        others=_OTHER,
        digits=1,
        first=[_DOTS],
        mistakes=[
            Mistake("farthest", wrong=_s(argmin="np.argmax(np.array(dists), axis=0)"),
                    en="Every penguin went to its farthest centroid: `np.argmax` gives the largest distance. Take `np.argmin`.",
                    nl="Elke pinguïn ging naar zijn verste centroid: `np.argmax` geeft de grootste afstand. Neem `np.argmin`."),
            Mistake("mean-of-all", wrong=_s(mean="X.mean(axis=0)"),
                    en="Your three centroids are the same point, the mean of all penguins. Each centroid is the mean of its own penguins: `X[labels == j]`.",
                    nl="Je drie centroids zijn hetzelfde punt, het gemiddelde van alle pinguïns. Elke centroid is het gemiddelde van zijn eigen pinguïns: `X[labels == j]`."),
            Mistake("in-millimetres", wrong=_s(mean="raw[labels == j].mean(axis=0)"),
                    en="Your centroids are the means in millimeters, and the last line turns them into millimeters once more. Take the mean of the standardized `X`, in which the distances were measured.",
                    nl="Je centroids zijn de gemiddelden in millimeters, en de laatste regel maakt er nog een keer millimeters van. Neem het gemiddelde van de gestandaardiseerde `X`, waarin de afstanden gemeten zijn."),
            Mistake("argmin-other-axis", error=_error("boolean index did not match", "dimension is 3"),
                    en="Your code stops with an error: `labels` must hold one cluster for every penguin, 342 of them. Is every distance `np.linalg.norm(X - c, axis=1)`, one number for each penguin, and does `np.argmin` look along `axis=0`, down the three rows of `np.array(dists)`?",
                    nl="Je code stopt met een fout: `labels` moet voor elke pinguïn één cluster hebben, 342 in totaal. Is elke afstand `np.linalg.norm(X - c, axis=1)`, één getal voor elke pinguïn, en kijkt `np.argmin` langs `axis=0`, langs de drie rijen van `np.array(dists)` omlaag?"),
            Mistake("mean-without-axis", error=_error("could not be broadcast", "inhomogeneous"),
                    en="Your code stops with an error: every centroid must be a point, with a mean for each of the two features. Take the mean down the penguins of the cluster, with `.mean(axis=0)`.",
                    nl="Je code stopt met een fout: elke centroid moet een punt zijn, met een gemiddelde voor elk van de twee features. Neem het gemiddelde langs de pinguïns van het cluster omlaag, met `.mean(axis=0)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...`: the distance from every penguin to centroid `c`, the nearest centroid of every penguin, and the mean of the penguins of cluster `j`.",
                      "nl": "Vervang de drie `...`: de afstand van elke pinguïn tot centroid `c`, de dichtstbijzijnde centroid van elke pinguïn, en het gemiddelde van de pinguïns van cluster `j`."},
            "no_value": {"en": "The last line of the box shows nothing. Put `centers * raw.std(axis=0) + raw.mean(axis=0)` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `centers * raw.std(axis=0) + raw.mean(axis=0)` op de laatste regel, zodat de check hem kan zien."},
            "wrong": {"en": "Your centroids are not those of the course. Is every distance `np.linalg.norm(X - c, axis=1)`, does `np.argmin` look along `axis=0`, and is each centroid the mean of `X[labels == j]` along `axis=0`?",
                      "nl": "Je centroids zijn niet die van de course. Is elke afstand `np.linalg.norm(X - c, axis=1)`, kijkt `np.argmin` langs `axis=0`, en is elke centroid het gemiddelde van `X[labels == j]` langs `axis=0`?"},
            "other": {"en": "Right for this start. But from another start, with {where}, your code gives other centroids than the course's. Do you use `centers` and `X`, and the steps of the course?",
                      "nl": "Goed voor deze start. Maar vanaf een andere start, met {where}, geeft je code andere centroids dan die van de course. Gebruik je `centers` en `X`, en de stappen van de course?"},
            "correct": {"en": "After one step the centroids are at 47.6 mm and 217.6 mm, at 41.6 mm and 192.6 mm, and at 53.2 mm and 217.4 mm. The second, at the start an Adelie, now lies among the Adelies and Chinstraps; the other two both came down from 230 to about 217 mm in flipper length, into the middle of the Gentoos, which they still share.",
                        "nl": "Na één stap liggen de centroids op 47,6 mm en 217,6 mm, op 41,6 mm en 192,6 mm, en op 53,2 mm en 217,4 mm. De tweede, bij de start een Adelie, ligt nu tussen de Adelies en de Chinstraps; de andere twee zakten allebei in flipperlengte van 230 naar ongeveer 217 mm, midden tussen de Gentoos, die ze nog delen."},
        },
    ),
}


# --- exercise 2: until nothing changes ----------------------------------------------------------------------------------

def _u(changed: str = "np.sum(new != labels)", stop: str = "changed == 0") -> str:
    return (_HEAD + "labels = np.zeros(len(X)) - 1\nfor step in range(20):\n    dists = []\n    for c in centers:\n"
            "        dists.append(np.linalg.norm(X - c, axis=1))\n    new = np.argmin(np.array(dists), axis=0)\n"
            f"    changed = {changed}\n    print(step, changed)\n    if {stop}:\n        break\n    labels = new\n"
            "    means = []\n    for j in range(3):\n        means.append(X[labels == j].mean(axis=0))\n"
            "    centers = np.array(means)\n"
            "step, np.unique(labels, return_counts=True)[1]\n")


_U_START = _u("...", "...")
_U = _u()


def _ambiguous(e) -> bool:
    return isinstance(e, ValueError) and "truth value of an array" in str(e)


EXERCISES["until-stable"] = CodeBox(
    name="until-stable",
    start=_U_START,
    solution=_U,
    keep=_KEEP,
    others=_OTHER,
    first=[_DOTS],
    mistakes=[
        Mistake("stopped-at-start", wrong=_u(stop="changed > 0"),
                en="Your loop stopped at the start, before the centroids moved once. At step 0 every penguin gets its first cluster, so all 342 change and none stays the same. Count the penguins that changed, `np.sum(new != labels)`, and stop when that is 0.",
                nl="Je loop stopte bij de start, voordat de centroids één keer waren verschoven. Bij stap 0 krijgt elke pinguïn zijn eerste cluster, dus veranderen ze alle 342 en blijft er geen hetzelfde. Tel de pinguïns die veranderden, `np.sum(new != labels)`, en stop als dat 0 is."),
        Mistake("no-sum", error=_ambiguous,
                en="Your code stops with an error: `new != labels` gives True or False for every penguin, and `if` cannot judge 342 of them at once. Count the True ones with `np.sum`.",
                nl="Je code stopt met een fout: `new != labels` geeft True of False voor elke pinguïn, en `if` kan er geen 342 tegelijk beoordelen. Tel de True's met `np.sum`."),
    ],
    texts={
        **_TEXTS,
        "empty": {"en": "Replace the two `...`: how many penguins changed cluster in this step, and when the loop stops.",
                  "nl": "Vervang de twee `...`: hoeveel pinguïns in deze stap van cluster veranderden, en wanneer de loop stopt."},
        "no_value": {"en": "The last line of the box shows nothing. Put `step, np.unique(labels, return_counts=True)[1]` on the last line, so the check can see it.",
                     "nl": "De laatste regel van het vak laat niets zien. Zet `step, np.unique(labels, return_counts=True)[1]` op de laatste regel, zodat de check hem kan zien."},
        "wrong": {"en": "Your loop does not stop where the course's does. Does `changed` count the penguins whose cluster in `new` differs from the one in `labels`, and does the loop stop when that is 0?",
                  "nl": "Je loop stopt niet waar die van de course stopt. Telt `changed` de pinguïns waarvan het cluster in `new` anders is dan dat in `labels`, en stopt de loop als dat 0 is?"},
        "other": {"en": "Right for this start. But from another start, with {where}, your loop stops elsewhere than the course's. Do you count with `new` and `labels`?",
                  "nl": "Goed voor deze start. Maar vanaf een andere start, met {where}, stopt je loop ergens anders dan die van de course. Tel je met `new` en `labels`?"},
        "correct": {"en": "At step 7 no penguin changes cluster any more, so the centroids stay where they are: k-means is done. Along the way 50, 24, 16, 20, 11 and 4 penguins changed, and the clusters hold 127, 151 and 64 penguins.",
                    "nl": "Bij stap 7 verandert geen enkele pinguïn meer van cluster, dus blijven de centroids waar ze zijn: k-means is klaar. Onderweg veranderden er 50, 24, 16, 20, 11 en 4, en de clusters hebben 127, 151 en 64 pinguïns."},
    },
)


# --- exercise 3: all four measurements, with scikit-learn ---------------------------------------------------------------

_F_HEAD = ("import numpy as np\nimport pandas as pd\nfrom sklearn.cluster import KMeans\n\n"
           "df = pd.read_csv(\"data/penguins.csv\")\ndf = df[df[\"flipper_length_mm\"].isna() == False]\n"
           "four = np.array(df[[\"bill_length_mm\", \"bill_depth_mm\", \"flipper_length_mm\", \"body_mass_g\"]])\n"
           "X4 = (four - four.mean(axis=0)) / four.std(axis=0)\nspecies = np.array(df[\"species\"])\n\n")


def _f(km: str = "KMeans(n_clusters=3, n_init=10, random_state=0).fit(X4)", most: str = "counts.max()") -> str:
    return (_F_HEAD + f"km = {km}\nright = 0\nfor j in range(3):\n"
            "    names, counts = np.unique(species[km.labels_ == j], return_counts=True)\n"
            f"    right = right + {most}\nright\n")


_F_START = _f("...", "...")
_F = _f()

EXERCISES["four"] = CodeBox(
    name="four",
    start=_F_START,
    solution=_F,
    first=[_DOTS],
    mistakes=[
        Mistake("unscaled", wrong=_f("KMeans(n_clusters=3, n_init=10, random_state=0).fit(four)"),
                en="You get `{got}`: without standardizing, the distance is almost only body mass, in grams. Fit on the standardized `X4`.",
                nl="Je krijgt `{got}`: zonder standaardiseren is de afstand bijna alleen het lichaamsgewicht, in grammen. Fit op de gestandaardiseerde `X4`."),
        Mistake("one-start", wrong=_f("KMeans(n_clusters=3, random_state=0).fit(X4)"),
                en="You get `{got}`: without `n_init=10`, k-means here runs from one start, and from a start it can end in another solution. Give it `n_init=10`: it then runs from 10 starts and keeps the one whose penguins lie closest to their centroids. The one start here happens to match the species better, but its penguins lie further from their centroids: k-means looks for the smallest distances, not for species. The next lesson looks at this.",
                nl="Je krijgt `{got}`: zonder `n_init=10` draait k-means hier vanaf één start, en vanaf een start kan het in een andere oplossing eindigen. Geef het `n_init=10`: het draait dan vanaf 10 starts en houdt de oplossing waarin de pinguïns het dichtst bij hun centroids liggen. De ene start hier past toevallig beter bij de soorten, maar zijn pinguïns liggen verder van hun centroids: k-means zoekt de kleinste afstanden, niet de soorten. De volgende les kijkt hiernaar."),
        Mistake("all-counted", wrong=_f(most="counts.sum()"),
                en="You get `{got}`, every penguin: `counts.sum()` counts all penguins of the cluster. Count only those of its most common species: `counts.max()`.",
                nl="Je krijgt `{got}`, elke pinguïn: `counts.sum()` telt alle pinguïns van het cluster. Tel alleen die van de soort die er het vaakst in zit: `counts.max()`."),
    ],
    texts={
        **_TEXTS,
        "empty": {"en": "Replace the two `...`: k-means with 3 clusters, 10 starts and `random_state=0`, fitted on `X4`, and the number of penguins of the most common species in cluster `j`.",
                  "nl": "Vervang de twee `...`: k-means met 3 clusters, 10 starts en `random_state=0`, gefit op `X4`, en het aantal pinguïns van de soort die het vaakst in cluster `j` zit."},
        "no_value": {"en": "The last line of the box shows nothing. Put `right` on the last line, so the check can see it.",
                     "nl": "De laatste regel van het vak laat niets zien. Zet `right` op de laatste regel, zodat de check hem kan zien."},
        "wrong": {"en": "You get `{got}`. Is it `KMeans(n_clusters=3, n_init=10, random_state=0).fit(X4)`, and do you add the largest count of each cluster?",
                  "nl": "Je krijgt `{got}`. Is het `KMeans(n_clusters=3, n_init=10, random_state=0).fit(X4)`, en tel je het grootste aantal van elk cluster op?"},
        "correct": {"en": "`{got}` of the 342: fewer than the 327 with two measurements. All 123 Gentoos have a cluster of their own, but 24 Adelies are in the cluster of the Chinstraps.",
                    "nl": "`{got}` van de 342: minder dan de 327 met twee metingen. Alle 123 Gentoos hebben een eigen cluster, maar 24 Adelies zitten in het cluster van de Chinstraps."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("one-step", _S_START, "empty", None),
    ("one-step", _S, "correct", None),
    ("one-step", _HEAD + "dists = np.array([np.linalg.norm(X - c, axis=1) for c in centers])\nlabels = np.argmin(dists, axis=0)\n"
     "centers = np.array([X[labels == j].mean(axis=0) for j in range(3)])\ncenters * raw.std(axis=0) + raw.mean(axis=0)\n", "correct", None),
    ("one-step", _s(norm="np.sqrt(np.sum((X - c) ** 2, axis=1))"), "correct", None),
    ("one-step", _s(argmin="np.argmax(np.array(dists), axis=0)"), "wrong", "farthest"),
    ("one-step", _s(mean="X.mean(axis=0)"), "wrong", "mean-of-all"),
    ("one-step", _s(mean="raw[labels == j].mean(axis=0)"), "wrong", "in-millimetres"),
    ("one-step", _s(argmin="np.argmin(np.array(dists), axis=1)"), "error", "argmin-other-axis"),
    ("one-step", _s(mean="X[labels == j].mean()"), "error", "mean-without-axis"),
    ("one-step", _HEAD.replace("centers = X[[216, 54, 240]]", "centers = X[[0, 151, 274]]") + _S[len(_HEAD):], "wrong", "changed-line"),
    ("until-stable", _U_START, "empty", None),
    ("until-stable", _U, "correct", None),
    ("until-stable", _u(stop="changed < 1"), "correct", None),
    ("until-stable", _u("np.sum(new == labels)"), "wrong", "stopped-at-start"),
    ("until-stable", _u(stop="changed > 0"), "wrong", "stopped-at-start"),
    ("until-stable", _u("new != labels"), "error", "no-sum"),
    ("until-stable", _u("np.sum(new - labels)"), "almost", "other-value"),
    ("one-step", _s(norm="np.linalg.norm(X - c, axis=0)"), "error", "argmin-other-axis"),
    ("one-step", _s(mean="X[labels == j].mean(axis=1)"), "error", "mean-without-axis"),
    ("four", _F_START, "empty", None),
    ("four", _F, "correct", None),
    ("four", _f(most="max(counts)"), "correct", None),
    ("four", _f("KMeans(n_clusters=3, n_init=10, random_state=0).fit(four)"), "wrong", "unscaled"),
    ("four", _f("KMeans(n_clusters=3, random_state=0).fit(X4)"), "wrong", "one-start"),
    ("four", _f(most="counts.sum()"), "wrong", "all-counted"),
    ("four", _f(most="counts[0]"), "wrong", None),
]
