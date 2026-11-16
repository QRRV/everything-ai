# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-vector-index: an IVF index of Granite Embedding's embeddings of the course's points
(data/course-summaries-granite.npz). k-means puts the points in CLUSTERS clusters, a list per cluster holds the numbers
of its points (make_lists), and a search compares a question only with the points of the clusters whose centroids are
nearest (ivf_search). For the 28 questions of the course: how often that finds the same three points as a search
through every vector, and how many vectors it compares; the pictures, and the search to try with the learner's own
functions."""
import copy
import functools
import html
import json

import numpy as np

from everything_ai import summaries, yours
from everything_ai.check import MISSING, FunctionBox, Mistake, run_code, with_time_limit

POINTS = summaries.points("en")   # the knowledge base: every point of every summary, in English
VECTORS = summaries.vectors("points", "en")   # Granite's embedding of each point, a row per point
QUESTIONS = summaries.questions()
QVECTORS = summaries.vectors("questions", "en")
ANSWERABLE = [q for q in QUESTIONS if q["kind"] != "none"]
ABOUT = summaries.about()
DIMS = VECTORS.shape[1]
CLUSTERS = 16    # the clusters of the index: about 80 points each
MORE = 32        # the other number of clusters the course tried
K = 3
SEED = 0
WORKED = "q18"   # the worked example: its best point is in the third nearest cluster
SHOWN = (1, 2, 3)   # the numbers of clusters to look in that the picture of ivf_search shows
MILLION = 1_000_000
BIG_CLUSTERS, BIG_PROBES = 1_000, 10   # the index of a million vectors the lesson works out


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def question(qid: str) -> dict:
    return next(q for q in QUESTIONS if q["id"] == qid)


def question_vector(q: dict) -> np.ndarray:
    return QVECTORS[QUESTIONS.index(q)]


def gold(q: dict) -> int | None:
    """The number of the point that answers a question; None when no point does."""
    return POINTS.index(summaries.text_of(q["gold"])) if q["gold"] else None


def title_of(i: int, lang: str) -> str:
    """The title of the lesson a point comes from."""
    return ABOUT[i]["title"][lang]


# --- the index: k-means as in ml-kmeans, and a list of points per cluster -----------------------------------------------

@functools.cache
def _kmeans(n: int) -> tuple:
    """The centroids and the label of every point, of k-means with n clusters on the points' embeddings."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n, n_init=10, random_state=SEED).fit(VECTORS)
    return km.cluster_centers_.astype(np.float32), [int(c) for c in km.labels_]


def centroids(n: int = CLUSTERS) -> np.ndarray:
    return _kmeans(n)[0]


def labels(n: int = CLUSTERS) -> list:
    return _kmeans(n)[1]


# --- exercise 1: make_lists ----------------------------------------------------------------------------------------------

START_LISTS = '''def make_lists(labels, n):
    lists = []
    for c in range(n):
        lists.append([])
    return lists
'''
SOLUTION_LISTS = START_LISTS.replace("lists.append([])", "lists.append([i for i in range(len(labels)) if labels[i] == c])")
LABELS_NOT_POINTS = START_LISTS.replace("lists.append([])", "lists.append([labels[i] for i in range(len(labels)) if labels[i] == c])")
SIZES = START_LISTS.replace("lists.append([])", "lists.append(len([i for i in range(len(labels)) if labels[i] == c]))")
make_lists = run_code(SOLUTION_LISTS).names["make_lists"]


@functools.cache
def index(n: int = CLUSTERS) -> dict:
    """The course's index: the centroids, a list of the numbers of the points of each cluster, and the embeddings."""
    return {"centroids": centroids(n), "lists": make_lists(labels(n), n), "vectors": VECTORS}


def _plain(v):
    """What a function gave back, as plain Python: lists of whole numbers, also from NumPy."""
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def _same(got, want) -> bool:
    return _plain(got) == _plain(want)


def _show(v) -> str:
    return json.dumps(_plain(v)) if isinstance(_plain(v), list) else " ".join(repr(v).split())


def _title_of_most(points: list, lang: str) -> str:
    """The title of the lesson most of the points come from."""
    counts: dict = {}
    for i in points:
        counts[ABOUT[i]["lesson"]] = counts.get(ABOUT[i]["lesson"], 0) + 1
    best = max(sorted(counts), key=lambda lesson_id: counts[lesson_id])
    return next(a["title"][lang] for a in ABOUT if a["lesson"] == best)


def cluster_title(c: int, lang: str) -> str:
    """The title of the lesson most points of a cluster of the course's index come from."""
    return _title_of_most(index()["lists"][c], lang)


_LABEL = "font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)"

_LISTS_WORDS = {
    "en": {"title": "The {n} lists your make_lists made of the course's points", "points": "{n} points", "one": "1 point",
           "most": "most from: {title}", "total": "Together {n} points; the course has {all}.",
           "alt": "A bar per cluster of the index, as long as the number of points in its list, with the lesson most of them come from."},
    "nl": {"title": "De {n} lijsten die je make_lists van de punten van de course maakte", "points": "{n} punten", "one": "1 punt",
           "most": "de meeste uit: {title}", "total": "Samen {n} punten; de course heeft er {all}.",
           "alt": "Een balk per cluster van de index, zo lang als het aantal punten in zijn lijst, met de les waar de meeste vandaan komen."},
}


def lists_picture(seen) -> str:
    """The learner's make_lists on the labels of the course's k-means: a bar per cluster, as long as its list, with
    the lesson most of its points come from."""
    w = _LISTS_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("make_lists") if mine is not None else None
    if not callable(fn):
        return ""
    found = labels()
    try:
        got = _plain(with_time_limit(lambda: fn(list(found), CLUSTERS), 10))
    except Exception:   # the learner's code; the check says what went wrong
        return ""
    if not (isinstance(got, list) and got and len(got) <= 64 and all(isinstance(x, list) for x in got)):
        return ""
    valid = [[i for i in members if isinstance(i, int) and 0 <= i < len(POINTS)] for members in got]
    longest = max([len(m) for m in valid] + [1])
    rows = []
    for c, members in enumerate(valid):
        most = html.escape(w["most"].format(title=_title_of_most(members, seen.lang))) if members else ""
        size = w["one"] if len(members) == 1 else w["points"].format(n=number(len(members), seen.lang))
        rows.append(f'<span style="{_LABEL};text-align:right">{c}</span>'
                    f'<span style="display:block;height:12px;width:{100 * len(members) / longest:.1f}%;background:var(--ea-data);border-radius:2px"></span>'
                    f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
                    f'{html.escape(size)} <span style="color:var(--ea-ink-2)">{most}</span></span>')
    total = sum(len(m) for m in valid)
    body = (f'<div style="display:grid;grid-template-columns:auto minmax(0,7rem) minmax(0,1fr);gap:0.2rem 0.5rem;align-items:center">'
            f'{"".join(rows)}</div><div style="{_LABEL};margin-top:0.3rem">'
            f'{html.escape(w["total"].format(n=number(total, seen.lang), all=number(len(POINTS), seen.lang)))}</div>')
    return yours.figure(w["title"].format(n=len(valid)), body, alt=w["alt"])


# --- exercise 2: ivf_search ----------------------------------------------------------------------------------------------

START_SEARCH = '''import numpy as np


def ivf_search(q, index, k, probes):
    distances = np.linalg.norm(index["centroids"] - q, axis=1)
    nearest = np.argsort(distances)[:probes]
    found = []
    for i in range(len(index["vectors"])):
        found.append((index["vectors"][i] @ q, i))
    return [i for score, i in sorted(found, reverse=True)[:k]]
'''
_EVERY = '''    for i in range(len(index["vectors"])):
        found.append((index["vectors"][i] @ q, i))
'''
SOLUTION_SEARCH = START_SEARCH.replace(_EVERY, '''    for c in nearest:
        for i in index["lists"][c]:
            found.append((index["vectors"][i] @ q, i))
''')
FIRST_ONLY = START_SEARCH.replace('''    for i in range(len(index["vectors"])):
''', '''    for i in index["lists"][nearest[0]]:
''')
PLACES = START_SEARCH.replace(_EVERY, '''    candidates = []
    for c in nearest:
        for i in index["lists"][c]:
            candidates.append(i)
    for j in range(len(candidates)):
        found.append((index["vectors"][candidates[j]] @ q, j))
''')
ivf_search = run_code(SOLUTION_SEARCH).names["ivf_search"]


class _Array(np.ndarray):
    """An array that a check message writes the way the lesson's code does: np.array([...]). The learner's code gets
    a plain copy."""

    def __repr__(self):
        return f"np.array({json.dumps(np.asarray(self).tolist())})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _a(values) -> np.ndarray:
    return np.array(values, dtype=float).view(_Array)


def _ints(values) -> np.ndarray:
    return np.array(values, dtype=int).view(_Array)


class _Named(dict):
    """An index that a check message calls by its name, as the lesson shows it. The learner's code gets a plain copy."""

    def __init__(self, name: str, values: dict):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return {k: copy.deepcopy(v, memo) for k, v in self.items()}


# six made-up vectors of two numbers in three clusters, as the lesson shows it; and the same with a fourth, empty cluster
SMALL = _Named("SMALL", {"centroids": _a([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]), "lists": [[0, 1], [2, 3], [4, 5]],
                         "vectors": _a([[0.9, 0.1], [0.8, -0.2], [0.1, 0.9], [-0.2, 0.8], [-0.9, 0.1], [-0.7, -0.3]])})
EMPTY = _Named("EMPTY", {"centroids": _a([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]), "lists": [[0, 1], [2, 3], [4, 5], []],
                         "vectors": _a([[0.9, 0.1], [0.8, -0.2], [0.1, 0.9], [-0.2, 0.8], [-0.9, 0.1], [-0.7, -0.3]])})
SMALL_CODE = '''SMALL = {"centroids": np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
         "lists": [[0, 1], [2, 3], [4, 5]],
         "vectors": np.array([[0.9, 0.1], [0.8, -0.2], [0.1, 0.9], [-0.2, 0.8], [-0.9, 0.1], [-0.7, -0.3]])}'''


class _Counted(np.ndarray):
    """An array that counts how many of its rows the learner's code compares with something: a row it takes out, the
    rows of a slice or a list of row numbers, and every row for a calculation with the whole array."""

    def _count(self, n: int) -> None:
        self.tally[0] += n

    def __getitem__(self, key):
        rows = np.arange(len(self))
        try:
            self._count(np.atleast_1d(rows[key[0] if isinstance(key, tuple) else key]).size)
        except (IndexError, TypeError, ValueError):
            pass
        return np.ndarray.__getitem__(self.view(np.ndarray), key)

    def __iter__(self):
        self._count(len(self))
        return iter(self.view(np.ndarray))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        self._count(len(self))
        return getattr(ufunc, method)(*_uncounted(inputs), **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        self._count(len(self))
        return func(*_uncounted(args), **kwargs)


def _uncounted(values):
    if isinstance(values, _Counted):
        return values.view(np.ndarray)
    if isinstance(values, (list, tuple)):
        return type(values)(_uncounted(v) for v in values)
    return values


def counted_index() -> tuple:
    """A copy of the course's index whose centroids and vectors count what the learner's code compares, and a function
    that gives the count so far."""
    course = index()
    parts = {}
    for name in ("centroids", "vectors"):
        parts[name] = np.array(course[name]).view(_Counted)
        parts[name].tally = [0]
    return ({"centroids": parts["centroids"], "lists": copy.deepcopy(course["lists"]), "vectors": parts["vectors"]},
            lambda: parts["centroids"].tally[0] + parts["vectors"].tally[0])


def comparisons(q: np.ndarray, probes: int, n: int = CLUSTERS) -> int:
    """How many vectors a search of the index compares a question with: every centroid, and the points of the
    `probes` nearest clusters."""
    near = np.argsort(np.linalg.norm(centroids(n) - q, axis=1))[:probes]
    return n + sum(len(index(n)["lists"][c]) for c in near)


def flat(q: np.ndarray, k: int = K) -> list:
    """The k points a search through every vector finds."""
    return [int(i) for i in np.argsort(-(VECTORS @ q), kind="stable")[:k]]


def cluster_places(q: np.ndarray, points: list, n: int = CLUSTERS) -> list:
    """For each point, the place of its cluster among the clusters nearest the question: 1 for the nearest."""
    order = [int(c) for c in np.argsort(np.linalg.norm(centroids(n) - q, axis=1))]
    return [order.index(labels(n)[i]) + 1 for i in points]


_SEARCH_WORDS = {
    "en": {"title": "What your ivf_search finds for: {question}", "probes": "Looking in {p} of the {n} clusters: your ivf_search compared {c} vectors",
           "probe": "Looking in 1 of the {n} clusters: your ivf_search compared {c} vectors", "every": "Through every vector: {c} compared",
           "same": "also among the three nearest of all", "other": "not among the three nearest of all",
           "error": "Your ivf_search stops with an error on the course's index."},
    "nl": {"title": "Wat je ivf_search vindt voor: {question}", "probes": "In {p} van de {n} clusters kijken: je ivf_search vergeleek {c} vectoren",
           "probe": "In 1 van de {n} clusters kijken: je ivf_search vergeleek {c} vectoren", "every": "Door elke vector: {c} vergeleken",
           "same": "ook bij de drie dichtstbijzijnde van alle", "other": "niet bij de drie dichtstbijzijnde van alle",
           "error": "Je ivf_search stopt met een fout op de index van de course."},
}


def _point_rows(found: list, want: list) -> str:
    rows = []
    for i in found[:K]:
        known = isinstance(i, int) and 0 <= i < len(POINTS)
        style = ("background:var(--ea-data-wash);border-left:3px solid var(--ea-data)" if i in want else
                 "border-left:3px solid var(--ea-loss)")
        text = POINTS[i] if known else repr(i)
        rows.append(f'<div style="margin:2px 0;padding:0.1rem 0.4rem;{style};font-family:var(--ea-font-ui);font-size:12px;'
                    f'color:var(--ea-ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{html.escape(text)}</div>')
    return "".join(rows)


def _swatch(label: str, color: str) -> str:
    return (f'<span style="display:inline-flex;align-items:center;gap:0.25rem;margin-right:0.8rem;{_LABEL}">'
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:var(--ea-{color})"></span>'
            f'{html.escape(label)}</span>')


def searched(fn, qid: str, probes_list, lang: str) -> str:
    """The learner's ivf_search on the course's index for one question: per number of clusters to look in, how many
    vectors that compares and the three points it finds, each marked by whether a search through every vector finds
    it too; and those three."""
    w = _SEARCH_WORDS[lang]
    q = question(qid)
    qv = question_vector(q)
    want = flat(qv)
    blocks = []
    for p in probes_list:
        given, compared = counted_index()
        try:
            got = _plain(with_time_limit(lambda: fn(qv.copy(), given, K, p), 10))
        except Exception:   # the learner's code; the check says what went wrong
            return f'<p style="{_LABEL}">{html.escape(w["error"])}</p>'
        if not isinstance(got, list):
            return ""
        head = (w["probe"] if p == 1 else w["probes"]).format(p=p, n=CLUSTERS, c=number(compared(), lang))
        blocks.append(f'<div style="margin:0.3rem 0 0.5rem"><div style="{_LABEL}">{html.escape(head)}</div>{_point_rows(got, want)}</div>')
    head = w["every"].format(c=number(len(POINTS), lang))
    blocks.append(f'<div style="margin:0.3rem 0 0.5rem"><div style="{_LABEL}">{html.escape(head)}</div>{_point_rows(want, want)}</div>')
    title = w["title"].format(question=q["text"][lang])
    legend = _swatch(w["same"], "data") + _swatch(w["other"], "loss")
    return yours.figure(title, "".join(blocks) + f"<div>{legend}</div>", alt=title)


def search_picture(seen) -> str:
    """The learner's ivf_search on the course's index, for the question of the worked example, looking in 1, 2 and 3
    clusters."""
    mine = seen.mine
    fn = mine.names.get("ivf_search") if mine is not None else None
    if not callable(fn):
        return ""
    return searched(fn, WORKED, SHOWN, seen.lang)


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

EXERCISES = {
    "make_lists": FunctionBox(
        name="make_lists",
        picture=lists_picture,
        function="make_lists",
        start=START_LISTS,
        solution=SOLUTION_LISTS,
        show=_show,
        compare=_same,
        visible=[([0, 1, 0, 2, 1], 3)],
        hidden=[([1, 1], 3), (_ints([2, 0, 2, 1]), 3), ([0, 0, 0], 1), ([], 2)],
        mistakes=[
            Mistake("labels", wrong=LABELS_NOT_POINTS,
                    en="`{call}` gives `{got}`: the labels of the points, the numbers of their clusters. The list of a cluster should hold the numbers of its points: `i`, not `labels[i]`.",
                    nl="`{call}` geeft `{got}`: de labels van de punten, de nummers van hun clusters. De lijst van een cluster hoort de nummers van zijn punten te bevatten: `i`, niet `labels[i]`."),
            Mistake("sizes", wrong=SIZES,
                    en="`{call}` gives `{got}`: how many points each cluster has. The index needs the points themselves: for cluster `c`, the list of the numbers `i` of its points.",
                    nl="`{call}` geeft `{got}`: hoeveel punten elk cluster heeft. De index heeft de punten zelf nodig: voor cluster `c` de lijst van de nummers `i` van zijn punten."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "Every cluster still gets an empty list. Give cluster `c` the numbers `i` of the points whose label is `c`.",
                      "nl": "Elk cluster krijgt nog een lege lijst. Geef cluster `c` de nummers `i` van de punten met label `c`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: for each cluster, from 0 to `n - 1`, the list of the numbers of its points.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: voor elk cluster, van 0 tot `n - 1`, de lijst van de nummers van zijn punten."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function give an empty list to a cluster that no point is in, and does it work for labels in a NumPy array?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Geeft je functie een lege lijst aan een cluster waar geen punt in zit, en werkt hij voor labels in een NumPy-array?"},
            "no_function": {"en": "Your box does not make a function `make_lists`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `make_lists`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `make_lists` is right in all {n} cases: for each cluster, the numbers of its points.",
                        "nl": "Je `make_lists` klopt in alle {n} gevallen: voor elk cluster de nummers van zijn punten."},
        },
    ),
    "ivf_search": FunctionBox(
        name="ivf_search",
        picture=search_picture,
        function="ivf_search",
        start=START_SEARCH,
        solution=SOLUTION_SEARCH,
        show=_show,
        compare=_same,
        visible=[(_a([0.6, 0.8]), SMALL, 2, 1)],
        hidden=[(_a([0.6, 0.8]), SMALL, 2, 2), (_a([0.6, 0.8]), SMALL, 2, 3), (_a([-0.6, -0.8]), SMALL, 1, 1),
                (_a([0.6, 0.8]), SMALL, 5, 1), (_a([0.1, -0.99]), EMPTY, 2, 1), (_a([0.1, -0.99]), EMPTY, 2, 2)],
        mistakes=[
            Mistake("every", wrong=START_SEARCH,
                    en="`{call}` gives `{got}`: the `k` nearest of all vectors, because your search still compares `q` with every vector. Compare it only with the points in the lists of the clusters in `nearest`.",
                    nl="`{call}` geeft `{got}`: de `k` dichtstbijzijnde van alle vectoren, omdat je zoekfunctie `q` nog met elke vector vergelijkt. Vergelijk hem alleen met de punten in de lijsten van de clusters in `nearest`."),
            Mistake("first-only", wrong=FIRST_ONLY,
                    en="`{call}` gives `{got}`: your search looked only in the nearest cluster. Look in every cluster of `nearest`: a loop over `nearest`, and in it a loop over the list of that cluster.",
                    nl="`{call}` geeft `{got}`: je zoekfunctie keek alleen in het dichtstbijzijnde cluster. Kijk in elk cluster van `nearest`: een loop over `nearest`, en daarin een loop over de lijst van dat cluster."),
            Mistake("places", wrong=PLACES,
                    en="`{call}` gives `{got}`: places in your list of candidates, not the numbers of the points. Give each score the number of its point, the `i` from the list of its cluster.",
                    nl="`{call}` geeft `{got}`: plaatsen in je lijst met kandidaten, niet de nummers van de punten. Geef elke score het nummer van zijn punt, de `i` uit de lijst van zijn cluster."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "The box still compares the question with every vector: `nearest` is never used. Loop over the clusters in `nearest`, and give only the points in their lists, `index[\"lists\"][c]`, a score.",
                      "nl": "Het vak vergelijkt de vraag nog met elke vector: `nearest` wordt nergens gebruikt. Loop over de clusters in `nearest`, en geef alleen de punten in hun lijsten, `index[\"lists\"][c]`, een score."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the `k` points with the highest dot product with `q`, among the points of the `probes` clusters whose centroids are nearest to `q`.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de `k` punten met het hoogste inproduct met `q`, tussen de punten van de `probes` clusters waarvan de centroids het dichtst bij `q` liggen."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function look in every cluster of `nearest`, also in an empty one, and does it work with a `k` larger than the number of points it compares?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Kijkt je functie in elk cluster van `nearest`, ook in een leeg cluster, en werkt hij met een `k` groter dan het aantal punten dat hij vergelijkt?"},
            "no_function": {"en": "Your box does not make a function `ivf_search`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `ivf_search`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `ivf_search` is right in all {n} cases: it compares the question only with the points of the `probes` nearest clusters.",
                        "nl": "Je `ivf_search` klopt in alle {n} gevallen: hij vergelijkt de vraag alleen met de punten van de `probes` dichtstbijzijnde clusters."},
        },
    ),
}

# The lesson is done when these are right; "find-more" and "quiet-miss" are choice questions in the lesson.
REQUIRED = ["make_lists", "ivf_search", "find-more", "quiet-miss"]

_LOOP = '''        members = []
        for i in range(len(labels)):
            if labels[i] == c:
                members.append(i)
        lists.append(members)
'''
CASES = [
    ("make_lists", START_LISTS, "empty", None),
    ("make_lists", "", "empty", None),
    ("make_lists", SOLUTION_LISTS, "correct", None),
    ("make_lists", SOLUTION_LISTS.replace("def make_lists(", "def lists_of("), "wrong", "no-function"),
    ("make_lists", START_LISTS.replace("        lists.append([])\n", _LOOP), "correct", None),
    ("make_lists", START_LISTS.replace("lists.append([])", "lists.append(np.arange(len(labels))[np.array(labels) == c])")
                              .replace("def make_lists", "import numpy as np\n\n\ndef make_lists"), "correct", None),
    ("make_lists", LABELS_NOT_POINTS, "wrong", "labels"),
    ("make_lists", SIZES, "wrong", "sizes"),
    ("make_lists", SOLUTION_LISTS.replace("range(n)", "range(n - 1)"), "wrong", None),
    ("make_lists", SOLUTION_LISTS.replace("range(n)", "range(max(labels) + 1)"), "almost", None),
    ("ivf_search", START_SEARCH, "empty", None),
    ("ivf_search", "", "empty", None),
    ("ivf_search", SOLUTION_SEARCH, "correct", None),
    ("ivf_search", SOLUTION_SEARCH.replace("def ivf_search(", "def search("), "wrong", "no-function"),
    ("ivf_search", START_SEARCH.replace(_EVERY, '''    candidates = []
    for c in nearest:
        candidates = candidates + index["lists"][c]
    for i in candidates:
        found.append((index["vectors"][i] @ q, i))
'''), "correct", None),
    ("ivf_search", START_SEARCH.replace("    found = []\n", "    found = list()\n"), "wrong", "every"),
    ("ivf_search", FIRST_ONLY, "wrong", "first-only"),
    ("ivf_search", PLACES, "wrong", "places"),
    ("ivf_search", SOLUTION_SEARCH.replace("sorted(found, reverse=True)", "sorted(found)"), "wrong", None),
    ("ivf_search", SOLUTION_SEARCH.replace("[:k]", "[:2]"), "almost", None),
]


# --- what the course measured --------------------------------------------------------------------------------------------

def _same_three(probes: int, n: int = CLUSTERS) -> tuple:
    """For a number of clusters to look in: for how many questions the index finds the same three points as a search
    through every vector, the mean number of vectors it compares, and for how many questions a point answers it puts
    that point among its three."""
    same = compared = golds = 0
    for q in QUESTIONS:
        qv = question_vector(q)
        got = ivf_search(qv, index(n), K, probes)
        same += got == flat(qv)
        compared += comparisons(qv, probes, n)
        golds += gold(q) is not None and gold(q) in got
    return same, round(compared / len(QUESTIONS)), golds


@functools.cache
def curve() -> list:
    """Per number of clusters to look in, 1 to CLUSTERS: (probes, same three, mean vectors compared, right point among
    the three)."""
    return [(p,) + _same_three(p) for p in range(1, CLUSTERS + 1)]


@functools.cache
def findings() -> dict:
    c = curve()
    sizes = [len(members) for members in index()["lists"]]
    return {"questions": len(QUESTIONS), "answerable": len(ANSWERABLE), "points": len(POINTS), "clusters": CLUSTERS,
            "smallest": min(sizes), "largest": max(sizes),
            "one": c[0], "two": c[1], "four": c[3], "seven": c[6],
            "all_at": next(p for p, same, _, _ in c if same == len(QUESTIONS)),
            "all_compared": next(cmp for p, same, cmp, _ in c if same == len(QUESTIONS)),
            "flat_gold": sum(gold(q) in flat(question_vector(q)) for q in ANSWERABLE),
            "best_nearest": sum(cluster_places(question_vector(q), flat(question_vector(q), 1))[0] == 1 for q in QUESTIONS),
            "multiplications": len(POINTS) * DIMS, "million_multiplications": MILLION * DIMS,
            "million_gb": MILLION * DIMS * 4 / 1e9, "big_compared": BIG_CLUSTERS + BIG_PROBES * MILLION // BIG_CLUSTERS}


@functools.cache
def more_clusters() -> tuple:
    """With MORE clusters, looking in 1: same three, mean vectors compared, right point among the three."""
    return _same_three(1, MORE)


def worked() -> dict:
    """The worked example: the nearest clusters of the question and their distances, the places of the clusters of the
    three points of a search through every vector, what looking in 1 and in 3 clusters finds, and the best scores."""
    q = question(WORKED)
    qv = question_vector(q)
    order = [int(c) for c in np.argsort(np.linalg.norm(centroids() - qv, axis=1))]
    want = flat(qv)
    one, three = ivf_search(qv, index(), K, 1), ivf_search(qv, index(), K, 3)
    return {"question": q, "nearest": order[:3], "distances": [float(np.linalg.norm(centroids()[c] - qv)) for c in order[:3]],
            "sizes": [len(index()["lists"][c]) for c in order[:3]], "want": want, "places": cluster_places(qv, want),
            "one": one, "three": three, "compared_one": comparisons(qv, 1), "compared_three": comparisons(qv, 3),
            "best_one": float(VECTORS[one[0]] @ qv), "best_all": float(VECTORS[want[0]] @ qv)}


_CURVE_WORDS = {
    "en": {"title": "Looking in more of the {n} clusters: the same three points, and the vectors compared",
           "same": "questions with the same three points as a search through every vector, of {q}",
           "probes": "clusters looked in", "compared": "vectors compared, on average; through every vector: {n}",
           "alt": "A bar per number of clusters looked in, 1 to {n}: above, as high as the number of the {q} questions for which the index finds the same three points as a search through every vector; below, as long as the average number of vectors it compared."},
    "nl": {"title": "In meer van de {n} clusters kijken: dezelfde drie punten, en de vergeleken vectoren",
           "same": "vragen met dezelfde drie punten als een zoekfunctie door elke vector, van de {q}",
           "probes": "bekeken clusters", "compared": "vergeleken vectoren, gemiddeld; door elke vector: {n}",
           "alt": "Een balk per aantal bekeken clusters, 1 tot {n}: boven zo hoog als het aantal van de {q} vragen waarvoor de index dezelfde drie punten vindt als een zoekfunctie door elke vector; onder zo lang als het gemiddelde aantal vergeleken vectoren."},
}


def curve_picture(lang: str) -> str:
    """A column per number of clusters to look in: above, a bar for the questions for which the index finds the same
    three points as a search through every vector; below, a bar for the average number of vectors it compares."""
    w = _CURVE_WORDS[lang]
    n_q = len(QUESTIONS)
    most = max(cmp for _, _, cmp, _ in curve())
    small = "font-family:var(--ea-font-code);font-size:11px"
    cols = []
    for p, same, compared, _ in curve():
        cols.append(f'<div style="display:flex;flex-direction:column;align-items:center;gap:2px;min-width:0">'
                    f'<span style="{small};color:var(--ea-ink)">{same}</span>'
                    f'<div style="height:110px;width:100%;display:flex;align-items:flex-end">'
                    f'<div style="width:100%;height:{100 * same / n_q:.1f}%;background:var(--ea-data);border-radius:2px 2px 0 0"></div></div>'
                    f'<span style="{small};color:var(--ea-ink);border-top:1px solid var(--ea-axis);width:100%;text-align:center;padding:2px 0">{p}</span>'
                    f'<div style="height:70px;width:100%"><div style="width:100%;height:{100 * compared / most:.1f}%;background:var(--ea-ink-3);border-radius:0 0 2px 2px"></div></div>'
                    f'<span style="{small};color:var(--ea-ink-2);writing-mode:vertical-rl;transform:rotate(180deg);white-space:nowrap">'
                    f'{html.escape(number(compared, lang))}</span></div>')
    grid = (f'<div style="display:grid;grid-template-columns:repeat({CLUSTERS},minmax(0,1fr));gap:3px;align-items:start">'
            f'{"".join(cols)}</div>')
    legend = (f'<div style="{_LABEL};margin-top:0.4rem">{_swatch(w["same"].format(q=n_q), "data")}</div>'
              f'<div style="{_LABEL}">{html.escape(w["probes"])} · '
              f'{_swatch(w["compared"].format(n=number(len(POINTS), lang)), "ink-3")}</div>')
    return yours.figure(w["title"].format(n=CLUSTERS), grid + legend, alt=w["alt"].format(n=CLUSTERS, q=n_q))


# --- try it: your ivf_search on a question of the course -----------------------------------------------------------------

def learner_search(lists_text, search_text):
    """The learner's ivf_search, when both exercises are right; else None. The course's index is made with the course's
    make_lists, which gives the same lists as a right one of the learner."""
    for name, text in (("make_lists", lists_text), ("ivf_search", search_text)):
        if text is MISSING or text is None or EXERCISES[name].check(text).state != "correct":
            return None
    return run_code(search_text).names["ivf_search"]


def question_options(lang: str) -> dict:
    """The questions of the course, as a dropdown shows them: their text in the lesson's language, to their id."""
    return {q["text"][lang]: q["id"] for q in QUESTIONS}
