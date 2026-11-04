# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-pca: the best shadow of the Palmer penguins. The share of the spread that the first
principal component keeps, by hand in NumPy, for bill length and bill depth; how many components of the four
measurements keep 90%; and k-means on the shadows of two components. And the picture: for every direction, the
penguins by bill length and flipper length, in millimeters from their mean, with their shadows on the line, and the
mean squared distance to the line of every direction."""
import functools
import math
import re

from everything_ai.check import Code, CodeBox, Mistake

DATA = "data/penguins.csv"   # the Palmer penguins (CC0): see data/README.md
TWO = ["bill_length_mm", "flipper_length_mm"]
DEPTH = ["bill_length_mm", "bill_depth_mm"]
FOUR = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
ANGLES = list(range(0, 180, 5))   # the stops of the slider, in degrees: 0 along bill length, 90 along flipper length


@functools.lru_cache(maxsize=1)
def penguins() -> dict:
    """The 342 penguins with all four measurements: by bill length and flipper length in millimeters, minus their
    mean; the four measurements as they are and standardized; and the species. Read-only."""
    import numpy as np
    import pandas as pd
    from everything_ai import paths
    df = pd.read_csv(paths.ROOT / DATA)
    df = df[df["flipper_length_mm"].isna() == False]   # noqa: E712 (the lesson's own way, with what the learner knows)
    two = np.array(df[TWO])
    four = np.array(df[FOUR])
    out = {"M": two - two.mean(axis=0), "depth": np.array(df[DEPTH]), "four": four,
           "X4": (four - four.mean(axis=0)) / four.std(axis=0), "species": np.array(df["species"])}
    for a in out.values():
        a.flags.writeable = False
    return out


def direction(angle: float):
    """The direction of length 1 at this angle, in degrees from the axis of bill length."""
    import numpy as np
    return np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])


def shadow(angle: float) -> dict:
    """For the line through the mean in this direction: the shadow of every penguin on it, the share of the spread
    the shadows keep, and the mean squared distance from a penguin to the line."""
    import numpy as np
    M = penguins()["M"]
    w = direction(angle)
    s = M @ w
    total = float(np.mean(np.sum(M ** 2, axis=1)))
    kept = float(np.mean(s ** 2))
    lost = float(np.mean(np.sum((M - np.outer(s, w)) ** 2, axis=1)))
    return {"w": w, "s": s, "kept": kept, "lost": lost, "total": total, "share": kept / total}


def matched(species, labels, k=3) -> int:
    """How many examples are in a cluster where their own label is the most common one."""
    import numpy as np
    return int(sum(np.unique(species[labels == j], return_counts=True)[1].max() for j in range(k)))


# --- the picture ------------------------------------------------------------------------------------------------------

def picture(angle: int, labels: dict[str, str]) -> str:
    """Left: the penguins by bill length and flipper length, in millimeters from their mean, with both axes on the
    same scale, so that right angles look right. The line through the mean in this direction is in orange, the colour
    of what is learned; the shadow of every penguin on it is in blue, what the model gives for an input; and the
    distance from every penguin to its shadow is a thin red line, the colour of a residual. Right: the mean squared
    distance to the line for every direction, in purple, the colour of what is made small, with this direction marked.
    Drawn as SVG text, once for every stop of the slider (L.explore); dragging across the right chart sets the angle.
    338 high: with the margins of chart.Panel, the left plot is then as high as it is wide."""
    from everything_ai import chart
    import numpy as np
    M = penguins()["M"]
    f = shadow(angle)
    w, s = f["w"], f["s"]
    P = np.outer(s, w)
    p = chart.Panel((-32, 32), (-32, 32), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=[-30, -15, 0, 15, 30], yticks=[-30, -15, 0, 15, 30])
    for (x, y), (a, b) in zip(M, P):
        p.line([x, a], [y, b], "residual", 0.6)
    p.line([-46 * w[0], 46 * w[0]], [-46 * w[1], 46 * w[1]], "param", 2.2)
    p.points(M[:, 0], M[:, 1], "ink-2", 1.9)
    p.points(P[:, 0], P[:, 1], "pred", 1.6)
    lost = [shadow(a)["lost"] for a in ANGLES]
    e = chart.Panel((-5, 180), (0, 230), title=labels["every"], xlabel=labels["angle"], ylabel=labels["lost"],
                    xticks=[0, 45, 90, 135, 180], yticks=[0, 50, 100, 150, 200])
    e.line([angle, angle], [0, f["lost"]], "ink-3", 1.2)
    e.line(ANGLES, lost, "loss", 2.0)
    e.points([angle], [f["lost"]], "loss", 5.0)
    return chart.figure([p, e], alt=labels["alt"], width=330, height=338, drag=1)


# The lesson is done when these are correct. "why-scale" is the choice question in the lesson.
REQUIRED = ["first-share", "how-many", "two-shadows", "why-scale"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_READ = ("df = pd.read_csv(\"data/penguins.csv\")\ndf = df[df[\"flipper_length_mm\"].isna() == False]\n")


# --- exercise 1: the share of the first component, by hand ------------------------------------------------------------

_H1 = ("import numpy as np\nimport pandas as pd\nfrom sklearn.decomposition import PCA\n\n" + _READ
       + "raw = np.array(df[[\"bill_length_mm\", \"bill_depth_mm\"]])\nM = raw - raw.mean(axis=0)\n"
       "total = np.mean(np.sum(M ** 2, axis=1))\npca = PCA(n_components=2).fit(M)\n\n")
_KEEP1 = ["pca = PCA(n_components=2).fit(M)"]


def _one(w: str = "pca.components_[0]", s: str = "M @ w", share: str = "np.mean(shadow ** 2) / total") -> str:
    return _H1 + f"w = {w}\nshadow = {s}\nshare = {share}\nshare, pca.explained_variance_ratio_[0]\n"


_ONE_START = _one("pca.components_[0]", "...", "...")
_ONE = _one()

EXERCISES = {
    "first-share": CodeBox(
        name="first-share",
        start=_ONE_START,
        solution=_ONE,
        keep=_KEEP1,
        digits=3,
        first=[_DOTS],
        mistakes=[
            Mistake("not-divided", wrong=_one(share="np.mean(shadow ** 2)"),
                    en="You get the spread of the shadows itself, in square millimeters. The share is that spread divided by the spread of all, `total`.",
                    nl="Je krijgt de spreiding van de schaduwen zelf, in vierkante millimeters. Het deel is die spreiding gedeeld door de spreiding van alles, `total`."),
            Mistake("second", wrong=_one("pca.components_[1]"),
                    en="That is the share of the second principal component: `components_[1]`. The first is `components_[0]`.",
                    nl="Dat is het deel van de tweede principal component: `components_[1]`. De eerste is `components_[0]`."),
            Mistake("times", wrong=_one(s="M * w"),
                    en="`M * w` multiplies feature by feature and leaves two numbers for every penguin. The shadow is one number, the dot product: `M @ w`.",
                    nl="`M * w` vermenigvuldigt feature voor feature en laat twee getallen per pinguïn over. De schaduw is één getal, het inproduct: `M @ w`."),
            Mistake("one-sum", matches=lambda got, text: re.search(r"shadow\s*=\s*np\.sum\(", text) is not None,
                    en="`np.sum` adds up over all penguins and leaves one number. The shadow is one number for every penguin, the dot product with `w`: `M @ w`.",
                    nl="`np.sum` telt op over alle pinguïns en laat één getal over. De schaduw is één getal voor elke pinguïn, het inproduct met `w`: `M @ w`."),
            Mistake("not-squared", wrong=_one(share="np.mean(shadow) / total"),
                    en="Your share is about 0: the shadows lie around 0, so those below 0 cancel those above. Their spread is the mean of their squares: `shadow ** 2`.",
                    nl="Je deel is ongeveer 0: de schaduwen liggen rond 0, dus heffen die onder 0 die erboven op. Hun spreiding is het gemiddelde van hun kwadraten: `shadow ** 2`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the shadow of every penguin on `w`, and the share of the spread those shadows keep.",
                      "nl": "Vervang de twee `...`: de schaduw van elke pinguïn op `w`, en het deel van de spreiding dat die schaduwen bewaren."},
            "no_value": {"en": "The last line of the box shows nothing. Put `share, pca.explained_variance_ratio_[0]` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `share, pca.explained_variance_ratio_[0]` op de laatste regel, zodat de check hem kan zien."},
            "wrong": {"en": "Your share is not `pca.explained_variance_ratio_[0]`. Is `shadow` `M @ w`, and is `share` the mean of the squared shadows divided by `total`?",
                      "nl": "Je deel is niet `pca.explained_variance_ratio_[0]`. Is `shadow` `M @ w`, en is `share` het gemiddelde van de gekwadrateerde schaduwen gedeeld door `total`?"},
            "correct": {"en": "0.892, the same as `explained_variance_ratio_[0]`. The first component is about (0.995, −0.096): almost along bill length, tilted a little down in bill depth. Bill length varies more in millimeters, so its direction keeps the most.",
                        "nl": "0,892, hetzelfde als `explained_variance_ratio_[0]`. De eerste component is ongeveer (0,995, −0,096): bijna langs de snavellengte, een beetje omlaag gekanteld in snavelhoogte. De snavellengte varieert over meer millimeters, dus bewaart haar richting het meest."},
        },
    ),
}


# --- exercise 2: how many components for 90% ------------------------------------------------------------------------

_H2 = ("import numpy as np\nimport pandas as pd\nfrom sklearn.decomposition import PCA\n\n" + _READ
       + "four = np.array(df[[\"bill_length_mm\", \"bill_depth_mm\", \"flipper_length_mm\", \"body_mass_g\"]])\n"
       "X4 = (four - four.mean(axis=0)) / four.std(axis=0)\npca = PCA(n_components=4).fit(X4)\n\n")


def _many(add: str = "pca.explained_variance_ratio_[n]", stop: str = "kept >= 0.9", data: str | None = None) -> str:
    head = _H2 if data is None else _H2.replace("pca = PCA(n_components=4).fit(X4)", f"pca = PCA(n_components=4).fit({data})")
    return head + f"kept = 0\nfor n in range(4):\n    kept = kept + {add}\n    if {stop}:\n        break\nn + 1, kept\n"


_MANY_START = _many("...", "...")
_MANY = _many()

EXERCISES["how-many"] = CodeBox(
    name="how-many",
    start=_MANY_START,
    solution=_MANY,
    digits=3,
    first=[_DOTS],
    mistakes=[
        Mistake("not-ratio", wrong=_many("pca.explained_variance_[n]"),
                en="`explained_variance_` is the spread of the shadows on each component itself, 2.76 for the first: more than 0.9 on its own. The share is `explained_variance_ratio_[n]`.",
                nl="`explained_variance_` is de spreiding van de schaduwen op elke component zelf, 2,76 voor de eerste: in zijn eentje al meer dan 0,9. Het deel is `explained_variance_ratio_[n]`."),
        Mistake("stop-below", wrong=_many(stop="kept < 0.9"),
                en="Your loop stopped after the first component, with 0.688: it stops while `kept` is still below 0.9. Stop once it has reached 0.9: `kept >= 0.9`.",
                nl="Je loop stopte na de eerste component, met 0,688: hij stopt terwijl `kept` nog onder 0,9 is. Stop zodra hij 0,9 heeft gehaald: `kept >= 0.9`."),
        Mistake("first-only", wrong=_many("pca.explained_variance_ratio_[0]"),
                en="You add the share of the first component every time. Add that of component `n`: `pca.explained_variance_ratio_[n]`.",
                nl="Je telt elke keer het deel van de eerste component op. Tel dat van component `n` op: `pca.explained_variance_ratio_[n]`."),
        Mistake("unscaled", wrong=_many(data="four"),
                en="You get one component that keeps everything: without standardizing, body mass in grams makes up almost all of the spread. Fit on the standardized `X4`.",
                nl="Je krijgt één component die alles bewaart: zonder standaardiseren maakt het lichaamsgewicht in grammen bijna alle spreiding. Fit op de gestandaardiseerde `X4`."),
    ],
    texts={
        **_TEXTS,
        "empty": {"en": "Replace the two `...`: the share of component `n`, added to `kept`, and when the loop stops.",
                  "nl": "Vervang de twee `...`: het deel van component `n`, opgeteld bij `kept`, en wanneer de loop stopt."},
        "no_value": {"en": "The last line of the box shows nothing. Put `n + 1, kept` on the last line, so the check can see it.",
                     "nl": "De laatste regel van het vak laat niets zien. Zet `n + 1, kept` op de laatste regel, zodat de check hem kan zien."},
        "wrong": {"en": "Your loop does not stop where the course's does. Do you add `pca.explained_variance_ratio_[n]`, and stop once `kept` is 0.9 or more?",
                  "nl": "Je loop stopt niet waar die van de course stopt. Tel je `pca.explained_variance_ratio_[n]` op, en stop je zodra `kept` 0,9 of meer is?"},
        "correct": {"en": "3 components, with 0.973. The first keeps 0.688 and the second 0.193 more, together 0.882: just short of 90%. So two numbers per penguin keep 88% of the spread of four.",
                    "nl": "3 componenten, met 0,973. De eerste bewaart 0,688 en de tweede nog 0,193, samen 0,882: net geen 90%. Twee getallen per pinguïn bewaren dus 88% van de spreiding van vier."},
    },
)


# --- exercise 3: k-means on two shadows -------------------------------------------------------------------------------

_H3 = ("import numpy as np\nimport pandas as pd\nfrom sklearn.cluster import KMeans\nfrom sklearn.decomposition import PCA\n\n" + _READ
       + "four = np.array(df[[\"bill_length_mm\", \"bill_depth_mm\", \"flipper_length_mm\", \"body_mass_g\"]])\n"
       "X4 = (four - four.mean(axis=0)) / four.std(axis=0)\nspecies = np.array(df[\"species\"])\n"
       "pca = PCA(n_components=2).fit(X4)\n\n")


def _km(z: str = "pca.transform(X4)", on: str = "Z") -> str:
    return (_H3 + f"Z = {z}\nkm = KMeans(n_clusters=3, n_init=10, random_state=0).fit({on})\nright = 0\nfor j in range(3):\n"
            "    names, counts = np.unique(species[km.labels_ == j], return_counts=True)\n"
            "    right = right + counts.max()\nright\n")


_KM_START = _km("...", "...")
_KM = _km()

EXERCISES["two-shadows"] = CodeBox(
    name="two-shadows",
    start=_KM_START,
    solution=_KM,
    first=[_DOTS],
    mistakes=[
        Mistake("all-four", wrong=_km(on="X4"),
                en="You get `{got}`: that is k-means on all four measurements, as in the k-means lesson. Fit on the two shadows in `Z`.",
                nl="Je krijgt `{got}`: dat is k-means op alle vier de metingen, zoals in de les over k-means. Fit op de twee schaduwen in `Z`."),
        Mistake("first-two-measurements", wrong=_km(z="X4[:, :2]"),
                en="You get `{got}`: `X4[:, :2]` is bill length and bill depth, the first two measurements, not the first two components. The shadows of every penguin on the components are `pca.transform(X4)`. That this is more than with two components is the point: PCA keeps the most spread, not what tells the species apart.",
                nl="Je krijgt `{got}`: `X4[:, :2]` is snavellengte en snavelhoogte, de eerste twee metingen, niet de eerste twee componenten. De schaduwen van elke pinguïn op de componenten zijn `pca.transform(X4)`. Dat dit meer is dan met twee componenten, is precies het punt: PCA bewaart de meeste spreiding, niet wat de soorten uit elkaar houdt."),
    ],
    texts={
        **_TEXTS,
        "empty": {"en": "Replace the two `...`: the shadows of every penguin on the two components, and what k-means is fitted on.",
                  "nl": "Vervang de twee `...`: de schaduwen van elke pinguïn op de twee componenten, en waarop k-means wordt gefit."},
        "no_value": {"en": "The last line of the box shows nothing. Put `right` on the last line, so the check can see it.",
                     "nl": "De laatste regel van het vak laat niets zien. Zet `right` op de laatste regel, zodat de check hem kan zien."},
        "wrong": {"en": "You get `{got}`. Is `Z` `pca.transform(X4)`, and is k-means fitted on `Z`?",
                  "nl": "Je krijgt `{got}`. Is `Z` `pca.transform(X4)`, en wordt k-means gefit op `Z`?"},
        "correct": {"en": "`{got}` of the 342, against 313 on all four measurements: 33 Adelies are now in the cluster of the Chinstraps, against 24. PCA keeps the most spread, not what tells the species apart.",
                    "nl": "`{got}` van de 342, tegen 313 op alle vier de metingen: 33 Adelies zitten nu in het cluster van de Chinstraps, tegen 24. PCA bewaart de meeste spreiding, niet wat de soorten uit elkaar houdt."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("first-share", _ONE_START, "empty", None),
    ("first-share", _ONE, "correct", None),
    ("first-share", _one(s="np.dot(M, w)"), "correct", None),
    ("first-share", _one(share="np.mean((M @ w) ** 2) / total"), "correct", None),
    ("first-share", _one(share="np.mean(shadow ** 2)"), "wrong", "not-divided"),
    ("first-share", _one("pca.components_[1]"), "wrong", "second"),
    ("first-share", _one(s="M * w"), "wrong", "times"),
    ("first-share", _one(share="np.mean(shadow) / total"), "wrong", "not-squared"),
    ("first-share", _one(s="np.sum(M * w)"), "wrong", "one-sum"),
    ("first-share", _one(share="0.892"), "wrong", None),
    ("how-many", _MANY_START, "empty", None),
    ("how-many", _MANY, "correct", None),
    ("how-many", _many(stop="kept > 0.9"), "correct", None),
    ("how-many", _many(stop="kept < 0.9"), "wrong", "stop-below"),
    ("how-many", _many("pca.explained_variance_[n]"), "wrong", "not-ratio"),
    ("how-many", _many("pca.explained_variance_ratio_[0]"), "wrong", "first-only"),
    ("how-many", _many(data="four"), "wrong", "unscaled"),
    ("two-shadows", _KM_START, "empty", None),
    ("two-shadows", _KM, "correct", None),
    ("two-shadows", _km(z="X4 @ pca.components_.T"), "correct", None),
    ("two-shadows", _km(on="X4"), "wrong", "all-four"),
    ("two-shadows", _km(z="X4[:, :2]"), "wrong", "first-two-measurements"),
]
