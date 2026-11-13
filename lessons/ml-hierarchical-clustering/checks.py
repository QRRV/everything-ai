# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-hierarchical-clustering: agglomerative clustering on the penguins of ml-choosing-k, by bill
length and flipper length, with four linkages (single, complete, average and Ward). The tree of every linkage, its
dendrogram cut into 1 to 4 clusters next to the penguins in their clusters, the function single_link, and how large
the largest of three clusters is for every linkage."""
import contextlib
import functools
import importlib.util
import io
import re
import warnings

from everything_ai import chart, check as checking, paths, yours
from everything_ai.check import CodeBox, FunctionBox, Mistake


def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_hc",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _lesson_checks("ml-choosing-k")
DATA = _K.DATA                    # the Palmer penguins (CC0): see data/README.md
TWO = _K.TWO                      # bill length and flipper length
penguins = _K.penguins            # the 342 penguins, standardized as in ml-kmeans and ml-choosing-k
METHODS = ["single", "complete", "average", "ward"]
KS = [1, 2, 3, 4]                 # the stops of the first slider: four clusters at most, one colour and shape each
NAMES = {"single": {"en": "single linkage", "nl": "single linkage"},
         "complete": {"en": "complete linkage", "nl": "complete linkage"},
         "average": {"en": "average linkage", "nl": "average linkage"},
         "ward": {"en": "Ward", "nl": "Ward"}}


# --- the course's own single linkage, as the lesson writes it -------------------------------------------------------

def single_link(A, B):
    import numpy as np
    distances = []
    for a in A:
        for b in B:
            distances.append(np.linalg.norm(a - b))
    return min(distances)


# --- the trees ------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def tree(method: str):
    """scipy's linkage matrix of the penguins: one row per merge, with the two clusters it merges, their distance
    (the height in the dendrogram) and the size of the new cluster. Clusters 0 to 341 are the penguins, 342 and up
    the clusters the rows make, in order."""
    from scipy.cluster.hierarchy import linkage
    Z = linkage(penguins()["X"], method)
    Z.flags.writeable = False
    return Z


@functools.lru_cache(maxsize=None)
def layout(method: str) -> tuple:
    """Where every node of the dendrogram lies: x is the leaf's place in the order of the tree (scipy's), or for a
    merge the middle of its two children; h is 0 for a leaf and the merge's height otherwise."""
    from scipy.cluster.hierarchy import leaves_list
    Z = tree(method)
    n = len(Z) + 1
    x, h = [0.0] * (2 * n - 1), [0.0] * (2 * n - 1)
    for place, leaf in enumerate(leaves_list(Z)):
        x[int(leaf)] = float(place)
    for i, (a, b, height, _) in enumerate(Z):
        x[n + i] = (x[int(a)] + x[int(b)]) / 2
        h[n + i] = float(height)
    return x, h


@functools.lru_cache(maxsize=None)
def clusters(method: str, k: int) -> dict:
    """The k clusters of the tree as {node: colour}: the nodes left when the top k − 1 merges are undone. Going from
    k − 1 to k clusters splits one cluster in two; the larger part keeps its colour and the smaller gets the next
    one, so a cluster keeps its colour on the slider."""
    Z = tree(method)
    n = len(Z) + 1
    if k == 1:
        return {2 * n - 2: 0}
    before = clusters(method, k - 1)
    top = max(before)                       # the highest merge still done: the next one to undo
    a, b = int(Z[top - n][0]), int(Z[top - n][1])
    size = lambda node: 1 if node < n else int(Z[node - n][3])   # noqa: E731
    big, small = (a, b) if size(a) >= size(b) else (b, a)
    out = {node: c for node, c in before.items() if node != top}
    out[big], out[small] = before[top], k - 1
    return out


@functools.lru_cache(maxsize=None)
def labels(method: str, k: int):
    """The colour of every penguin's cluster, 0 to k − 1."""
    import numpy as np
    Z = tree(method)
    n = len(Z) + 1
    out = np.zeros(n, dtype=int)
    for node, c in clusters(method, k).items():
        stack = [node]
        while stack:
            j = stack.pop()
            if j < n:
                out[j] = c
            else:
                stack += [int(Z[j - n][0]), int(Z[j - n][1])]
    out.flags.writeable = False
    return out


def sizes(method: str, k: int) -> list:
    """How many penguins each cluster holds, from large to small."""
    import numpy as np
    return sorted((int(v) for v in np.bincount(labels(method, k), minlength=k)), reverse=True)


def cut(method: str, k: int) -> float | None:
    """A height between the last merge the k clusters keep and the first they undo; None for one cluster."""
    Z = tree(method)
    return None if k == 1 else float((Z[-k][2] + Z[-k + 1][2]) / 2)


def number(x: float, lang: str, digits: int = 1) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def listed(values: list, lang: str) -> str:
    """1, 2 and 3; in Dutch 1, 2 en 3."""
    words = [str(v) for v in values]
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + (" and " if lang == "en" else " en ") + words[-1]


# --- the picture ----------------------------------------------------------------------------------------------------

def _trace(method: str, node: int, stops: set) -> list:
    """One line through the dendrogram below a node, from the node's top back to it, every bracket drawn on the way
    (some twice): one polyline per colour instead of one per merge keeps the picture small. A node in `stops` counts
    as a leaf, at its own height."""
    Z = tree(method)
    n = len(Z) + 1
    x, h = layout(method)
    out = []
    todo = [("node", node)]
    while todo:
        kind, j = todo.pop()
        if kind == "point":
            out.append(j)
            continue
        if j < n or (j in stops and j != node):
            out.append((x[j], h[j]))
            continue
        a, b = int(Z[j - n][0]), int(Z[j - n][1])
        top = h[j]
        # in order: up the left leg, the left subtree, across, the right subtree, back to the middle
        steps = [("point", (x[j], top)), ("point", (x[a], top)), ("node", a), ("point", (x[a], top)),
                 ("point", (x[b], top)), ("node", b), ("point", (x[b], top)), ("point", (x[j], top))]
        todo += reversed(steps)
    return out


def picture(method: str, k: int, labels_: dict) -> str:
    """Left: the dendrogram of this linkage, every merge a bracket at its height; the branches of each of the k
    clusters in that cluster's colour, the merges above them in grey, and the cut as a dashed line in the colour of
    what you choose. Right: the penguins by bill length and flipper length, standardized, in the colour and shape of
    their cluster. Drawn as SVG text, once for every stop of the slider (L.explore)."""
    Z = tree(method)
    n = len(Z) + 1
    top = float(Z[-1][2])
    d = chart.Panel((-4, n + 3), (0, top * 1.06), title=labels_["tree"], xlabel=labels_["leaves"], ylabel=labels_["height"],
                    xticks=[])
    parts = clusters(method, k)
    if k > 1:
        grey = _trace(method, 2 * n - 2, set(parts))
        d.line([p[0] for p in grey], [p[1] for p in grey], "ink-3", 1.2)
    for node, c in parts.items():
        if node >= n:
            line = _trace(method, node, set())
            d.line([p[0] for p in line], [p[1] for p in line], f"group-{c + 1}", 1.0)
    line = cut(method, k)
    if line is not None:
        d.line([-4, n + 3], [line, line], "hyper", 1.6, dashed=True)
    X = penguins()["X"]
    lab = labels(method, k)
    p = chart.Panel((-2.4, 3.1), (-2.3, 2.4), title=labels_["title"], xlabel=labels_["x"], ylabel=labels_["y"],
                    xticks=[-2, -1, 0, 1, 2, 3], yticks=[-2, -1, 0, 1, 2])
    for c in range(k):
        rows = X[lab == c]
        p.group(rows[:, 0], rows[:, 1], c, r=3.4)
    return chart.figure([d, p], alt=labels_["alt"], width=330, height=290)


# The lesson is done when these are correct. "chain" is the choice question in the lesson.
REQUIRED = ["single-link", "largest", "chain"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


# --- exercise 1: single_link ----------------------------------------------------------------------------------------

class _Val:
    """A test value that the check's messages write by its name, such as `A`; the function gets an array."""

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
    return repr(v)


# Two made-up clusters of two features; the numbers of the worked example as clusters on a line; two clusters with
# three features; one example each; and two clusters where the nearest pair is not that of the means.
_A = _Val("A", [[0.0, 0.0], [1.0, 0.0]])
_B = _Val("B", [[4.0, 0.0], [3.0, 4.0], [5.0, 1.0]])
_LEFT = _Val("left", [[1.0], [2.0], [4.0]])
_RIGHT = _Val("right", [[8.0], [9.0]])
_A3 = _Val("A3", [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
_B3 = _Val("B3", [[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]])
_ONE = _Val("one", [[0.0, 0.0]])
_OTHER = _Val("other", [[3.0, 4.0]])
_LONG = _Val("long", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
_NEAR = _Val("near", [[4.5, 1.0], [6.0, 3.0]])

_SL_HEAD = ("import numpy as np\n\n\ndef single_link(A, B):\n    distances = []\n    for a in A:\n        for b in B:\n")


def _sl(append: str = "np.linalg.norm(a - b)", back: str = "min(distances)") -> str:
    return _SL_HEAD + f"            distances.append({append})\n    return {back}\n"


_SL_START = _sl("...", "...")
_SL = _sl()

_SL_WORDS = {
    "title": {"en": "Your single_link of A and B", "nl": "Jouw single_link van A en B"},
    "said": {"en": "A, circles, and B, triangles, with a gray line for every pair. Thick: the closest pair, {want} apart. Your function gives {got}.",
             "nl": "A, rondjes, en B, driehoeken, met een grijze lijn voor elk paar. Dik: het dichtstbijzijnde paar, op {want} van elkaar. Jouw functie geeft {got}."},
    "right": {"en": "A, circles, and B, triangles, with a gray line for every pair. Thick: the closest pair, {want} apart: what your function gives.",
              "nl": "A, rondjes, en B, driehoeken, met een grijze lijn voor elk paar. Dik: het dichtstbijzijnde paar, op {want} van elkaar: wat jouw functie geeft."},
}


def _sl_picture(seen) -> str:
    """The visible case: the points of A and B, every pair joined by a thin grey line and the closest pair by a thick
    one, with what the learner's function gives against that distance."""
    import numpy as np
    ex = seen.exercise
    mine = seen.mine
    if mine is None:
        return ""
    fn = mine.names.get(ex.function)
    if not callable(fn):
        return ""
    A, B = np.array(_A.value), np.array(_B.value)
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = checking.with_time_limit(lambda: fn(A.copy(), B.copy()))
        got = float(got)
    except Exception:
        return ""
    want = single_link(A, B)
    lang = seen.lang
    p = chart.Panel((-0.5, 5.5), (-0.6, 4.6), xticks=[0, 1, 2, 3, 4, 5], yticks=[0, 1, 2, 3, 4])
    best = None
    for a in A:
        for b in B:
            p.line([a[0], b[0]], [a[1], b[1]], "ink-3", 0.8)
            if best is None or np.linalg.norm(a - b) < np.linalg.norm(best[0] - best[1]):
                best = (a, b)
    p.line([best[0][0], best[1][0]], [best[0][1], best[1][1]], "ink", 2.6)
    p.group(A[:, 0], A[:, 1], 0, r=5.5)
    p.group(B[:, 0], B[:, 1], 1, r=5.5)
    right = abs(got - want) < 1e-6
    words = {k: v[lang] for k, v in _SL_WORDS.items()}
    text = (words["right"] if right else words["said"]).format(got=number(got, lang, 3), want=number(want, lang, 3))
    key = ('<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.15rem">'
           + text + "</div>")
    return yours.figure(words["title"], chart.figure([p], alt=text, width=320, height=250) + key, alt=text)


def _array_truth(e) -> bool:
    return isinstance(e, ValueError) and "truth value of an array" in str(e)


EXERCISES = {
    "single-link": FunctionBox(
        name="single-link",
        picture=_sl_picture,
        function="single_link",
        start=_SL_START,
        solution=_SL,
        show=_shown,
        visible=[(_A, _B)],
        hidden=[(_LEFT, _RIGHT), (_A3, _B3), (_ONE, _OTHER), (_LONG, _NEAR)],
        first=[_DOTS],
        mistakes=[
            Mistake("largest", wrong=_sl(back="max(distances)"),
                    en="`{call}` gives {got}: the distance of the farthest pair. That is complete linkage. Single linkage takes the closest pair: `min`.",
                    nl="`{call}` geeft {got}: de afstand van het verste paar. Dat is complete linkage. Single linkage neemt het dichtstbijzijnde paar: `min`."),
            Mistake("mean", wrong=_sl(back="np.mean(distances)"),
                    en="`{call}` gives {got}: the mean distance over all pairs. That is average linkage. Single linkage takes the closest pair: `min`.",
                    nl="`{call}` geeft {got}: de gemiddelde afstand over alle paren. Dat is average linkage. Single linkage neemt het dichtstbijzijnde paar: `min`."),
            Mistake("no-norm", error=_array_truth, wrong=_sl(append="a - b"),
                    en="With `{call}`, your function stops at `min`: `a - b` is a difference with a number for every feature, not one distance, and `min` cannot compare those. Take its length: `np.linalg.norm(a - b)`.",
                    nl="Met `{call}` stopt je functie bij `min`: `a - b` is een verschil met een getal voor elke feature, niet één afstand, en `min` kan die niet vergelijken. Neem de lengte ervan: `np.linalg.norm(a - b)`."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the two `...`: the distance between `a` and `b`, and then the smallest of all those distances.",
                      "nl": "Vervang de twee `...`: de afstand tussen `a` en `b`, en dan de kleinste van al die afstanden."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(A, B)`.",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(A, B)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Do you append the distance of every pair, `np.linalg.norm(a - b)`, and give back the smallest?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Voeg je de afstand van elk paar toe, `np.linalg.norm(a - b)`, en geef je de kleinste terug?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for clusters of any size and any number of features?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor clusters van elke grootte en elk aantal features?"},
            "correct": {"en": "Your function is right in all {n} cases. For `left` and `right`, the numbers 1, 2, 4 and 8, 9 of the worked example, it gives 4: the gap between 4 and 8, the height at which single linkage merged them there.",
                        "nl": "Je functie klopt in alle {n} gevallen. Voor `left` en `right`, de getallen 1, 2, 4 en 8, 9 uit het uitgewerkte voorbeeld, geeft hij 4: het gat tussen 4 en 8, de hoogte waarop single linkage ze daar samenvoegde."},
        },
    ),
}


# --- exercise 2: the largest of three clusters for every linkage ----------------------------------------------------

_HEAD = ("import numpy as np\nimport pandas as pd\nfrom scipy.cluster.hierarchy import linkage, fcluster\n\n"
         "df = pd.read_csv(\"data/penguins.csv\")\ndf = df[df[\"flipper_length_mm\"].isna() == False]\n"
         "raw = np.array(df[[\"bill_length_mm\", \"flipper_length_mm\"]])\n"
         "X = (raw - raw.mean(axis=0)) / raw.std(axis=0)\n\n")


def _lg(tree_: str = "linkage(X, method)", cut_: str = "fcluster(Z, 3, criterion=\"maxclust\")") -> str:
    return (_HEAD + "largest = {}\nfor method in [\"single\", \"complete\", \"average\", \"ward\"]:\n"
            f"    Z = {tree_}\n    labels = {cut_}\n"
            "    counts = np.unique(labels, return_counts=True)[1]\n    largest[method] = int(max(counts))\nlargest\n")


_LG_START = _lg("...", "...")
_LG = _lg()


def _all_same(got, text) -> bool:
    return isinstance(got, dict) and len(got) == 4 and len({v for v in got.values()}) == 1


EXERCISES["largest"] = CodeBox(
    name="largest",
    picture=yours.outcome_bars(METHODS, title={"en": "Your largest of three clusters, per linkage", "nl": "Jouw grootste van drie clusters, per linkage"},
                               xlabel="linkage", ylabel={"en": "penguins", "nl": "pinguïns"}),
    start=_LG_START,
    solution=_LG,
    first=[_DOTS],
    mistakes=[
        Mistake("two", wrong=_lg(cut_="fcluster(Z, 2, criterion=\"maxclust\")"),
                en="Your numbers are those of two clusters: single linkage, for one, puts 341 penguins together. Cut each tree into three: `fcluster(Z, 3, criterion=\"maxclust\")`.",
                nl="Je getallen zijn die van twee clusters: single linkage, bijvoorbeeld, zet 341 pinguïns bij elkaar. Snijd elke boom in drie: `fcluster(Z, 3, criterion=\"maxclust\")`."),
        Mistake("no-criterion", wrong=_lg(cut_="fcluster(Z, 3)"),
                en="Every linkage gives one cluster of all 342 penguins: without `criterion=\"maxclust\"`, the 3 in `fcluster(Z, 3)` is not a number of clusters but a threshold on another measure. Add `criterion=\"maxclust\"`.",
                nl="Elke linkage geeft één cluster van alle 342 pinguïns: zonder `criterion=\"maxclust\"` is de 3 in `fcluster(Z, 3)` geen aantal clusters maar een drempel op een andere maat. Voeg `criterion=\"maxclust\"` toe."),
        Mistake("one-method", matches=_all_same,
                en="Every linkage gives the same number: does `linkage` get `method`, the linkage of this turn of the loop, as its second value?",
                nl="Elke linkage geeft hetzelfde getal: krijgt `linkage` als tweede waarde `method`, de linkage van deze ronde van de loop?"),
    ],
    texts={
        **_SAME,
        "empty": {"en": "Replace the two `...`: the tree of the penguins with this `method`, and its cut into three clusters.",
                  "nl": "Vervang de twee `...`: de boom van de pinguïns met deze `method`, en zijn snede in drie clusters."},
        "no_value": {"en": "The last line of the box shows nothing. Put `largest` on the last line, so the check can see it.",
                     "nl": "De laatste regel van het vak laat niets zien. Zet `largest` op de laatste regel, zodat de check hem kan zien."},
        "wrong": {"en": "Your largest clusters are not those of the course. Is `Z` `linkage(X, method)`, and `labels` its cut into three clusters, `fcluster(Z, 3, criterion=\"maxclust\")`?",
                  "nl": "Je grootste clusters zijn niet die van de course. Is `Z` `linkage(X, method)`, en `labels` zijn snede in drie clusters, `fcluster(Z, 3, criterion=\"maxclust\")`?"},
        "correct": {"en": "340 for single linkage, 176 for average linkage, 151 for complete linkage and 152 for Ward. With single and average linkage, the two other clusters are tiny: 1 and 1, and 165 and 1. Complete linkage and Ward split the penguins into three real groups.",
                    "nl": "340 voor single linkage, 176 voor average linkage, 151 voor complete linkage en 152 voor Ward. Bij single en average linkage zijn de twee andere clusters piepklein: 1 en 1, en 165 en 1. Complete linkage en Ward verdelen de pinguïns in drie echte groepen."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("single-link", _SL_START, "empty", None),
    ("single-link", _sl(back="..."), "wrong", "dots-left"),
    ("single-link", _SL, "correct", None),
    ("single-link", _sl("np.sqrt(np.sum((a - b) ** 2))"), "correct", None),
    ("single-link", _sl(back="np.min(distances)"), "correct", None),
    ("single-link", _sl(back="max(distances)"), "wrong", "largest"),
    ("single-link", _sl(back="np.mean(distances)"), "wrong", "mean"),
    ("single-link", _sl(append="a - b"), "error", "no-norm"),
    ("single-link", _sl(back="distances[0]"), "wrong", None),
    ("single-link", "import numpy as np\n\n\ndef single_link(A, B):\n    return np.linalg.norm(np.mean(A, axis=0) - np.mean(B, axis=0))\n", "wrong", None),
    ("largest", _LG_START, "empty", None),
    ("largest", _lg(cut_="..."), "wrong", "dots-left"),
    ("largest", _LG, "correct", None),
    ("largest", _lg(cut_="fcluster(Z, 3, \"maxclust\")"), "correct", None),
    ("largest", _lg(cut_="fcluster(Z, 2, criterion=\"maxclust\")"), "wrong", "two"),
    ("largest", _lg(cut_="fcluster(Z, 3)"), "wrong", "no-criterion"),
    ("largest", _lg(tree_="linkage(X, \"ward\")"), "wrong", "one-method"),
    ("largest", _lg(tree_="linkage(X)"), "wrong", "one-method"),
]
