# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-network-data: the network of flights of OpenFlights as a graph without directions, its
degree distribution against that of a random network with as many nodes and edges, its hubs, and how many flights
apart its airports are with the biggest hubs closed or as many airports closed at random; the download of the routes;
and the functions degree_counts and flights_from. The routes are downloaded (everything_ai.flights), so the numbers in
COUNTS and CLOSED were measured on them by the course, and tests/test_ml_network_data.py measures them again when the
routes are there. Numbers and pictures made from the routes contain information from OpenFlights, made available
under the ODbL."""
import contextlib
import errno
import functools
import html
import io
import urllib.error
import warnings

from everything_ai import chart, check as checking, flights, hub, yours
from everything_ai.check import FunctionBox, Mistake
from everything_ai.lesson import download_card

ROUTES, AIRPORTS, EDGES = 67663, 3425, 19256   # routes in routes.dat; nodes and edges of the network without directions
HUBS = [("AMS", 248), ("FRA", 244), ("CDG", 240), ("IST", 236), ("ATL", 217)]   # the five biggest hubs and their degrees
TOP = 34   # the biggest hubs that make 1% of the airports; they are at one end of TOP_EDGES of the edges
TOP_EDGES = 5495
CONNECTED = 3397   # the airports that hang together; the other 28 have routes only among themselves, in 7 small groups

# For every degree, how many airports have it: the degree distribution of the network.
COUNTS = {1: 768, 2: 744, 3: 407, 4: 237, 5: 163, 6: 111, 7: 84, 8: 83, 9: 79, 10: 47, 11: 43, 12: 36, 13: 32, 14: 18,
          15: 23, 16: 18, 17: 19, 18: 26, 19: 26, 20: 22, 21: 19, 22: 12, 23: 16, 24: 7, 25: 8, 26: 16, 27: 3, 28: 7,
          29: 13, 30: 15, 31: 7, 32: 14, 33: 10, 34: 6, 35: 10, 36: 2, 37: 9, 38: 14, 39: 7, 40: 4, 41: 13, 42: 3,
          43: 4, 44: 4, 45: 5, 46: 2, 47: 8, 48: 5, 49: 6, 50: 6, 51: 5, 52: 4, 53: 1, 54: 3, 55: 7, 56: 3, 57: 7,
          58: 4, 59: 3, 60: 1, 61: 4, 62: 3, 63: 2, 64: 3, 65: 2, 66: 2, 67: 3, 68: 2, 69: 3, 70: 2, 72: 4, 73: 2,
          74: 3, 75: 2, 76: 1, 77: 2, 78: 3, 79: 1, 80: 2, 81: 4, 82: 1, 83: 3, 85: 4, 86: 3, 87: 1, 89: 3, 90: 2,
          91: 2, 92: 2, 93: 2, 94: 2, 95: 2, 96: 2, 97: 1, 98: 2, 101: 1, 102: 1, 103: 3, 104: 2, 105: 2, 108: 1,
          110: 1, 111: 1, 112: 1, 113: 2, 117: 1, 118: 1, 122: 2, 123: 2, 124: 1, 125: 1, 127: 2, 131: 1, 133: 2,
          134: 1, 135: 1, 137: 1, 138: 1, 139: 1, 141: 1, 144: 2, 147: 2, 148: 1, 149: 1, 150: 1, 153: 3, 154: 1,
          159: 1, 162: 2, 163: 1, 165: 1, 169: 1, 170: 1, 172: 1, 188: 2, 190: 1, 192: 1, 206: 1, 207: 1, 217: 1,
          236: 1, 240: 1, 244: 1, 248: 1}

# With h airports closed, the h biggest hubs ("hubs", the highest degree first, equal degrees by code) or the first h
# of the airports in a random order ("random", numpy's default_rng(0).permutation of the codes in alphabetical order):
# the airports still open, how many of them hang together in the biggest part, the fewest flights between two open
# airports that can reach each other on average, and in % of all pairs of open airports, those at most 1, 2, ..., 10
# flights apart.
STOPS = [0, 10, 20, 50, 100, 200]
CLOSED = {
    (0, "hubs"): (3425, 3397, 4.1, [0.3, 5.8, 32.0, 67.6, 86.5, 94.3, 97.1, 98.1, 98.3, 98.4]),
    (0, "random"): (3425, 3397, 4.1, [0.3, 5.8, 32.0, 67.6, 86.5, 94.3, 97.1, 98.1, 98.3, 98.4]),
    (10, "hubs"): (3415, 3336, 4.23, [0.3, 4.7, 26.0, 61.7, 82.3, 91.0, 94.1, 95.1, 95.4, 95.4]),
    (10, "random"): (3415, 3371, 4.1, [0.3, 5.8, 31.8, 67.2, 85.9, 93.5, 96.3, 97.1, 97.4, 97.4]),
    (20, "hubs"): (3405, 3283, 4.34, [0.3, 4.0, 22.3, 56.1, 78.5, 88.1, 91.5, 92.6, 92.9, 93.0]),
    (20, "random"): (3405, 3359, 4.1, [0.3, 5.8, 31.8, 67.1, 85.7, 93.4, 96.1, 97.0, 97.3, 97.3]),
    (50, "hubs"): (3375, 3153, 4.77, [0.2, 2.6, 13.6, 39.1, 64.8, 78.3, 84.1, 86.3, 87.0, 87.2]),
    (50, "random"): (3375, 3303, 4.12, [0.3, 5.6, 30.6, 65.1, 83.8, 91.7, 94.6, 95.5, 95.7, 95.8]),
    (100, "hubs"): (3325, 2903, 5.74, [0.2, 1.5, 6.3, 17.4, 34.4, 52.4, 65.7, 72.2, 74.8, 75.8]),
    (100, "random"): (3325, 3223, 4.15, [0.3, 5.4, 29.2, 63.2, 81.8, 89.8, 92.7, 93.6, 93.9, 94.0]),
    (200, "hubs"): (3225, 2288, 7.92, [0.1, 0.6, 2.0, 4.8, 9.2, 15.3, 22.7, 30.4, 37.2, 42.3]),
    (200, "random"): (3225, 3075, 4.1, [0.3, 5.4, 29.1, 62.5, 80.1, 87.3, 89.9, 90.6, 90.9, 90.9]),
}


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


@functools.lru_cache(maxsize=1)
def random_degrees() -> tuple:
    """The degrees of a random network with as many nodes and edges as the network of flights: every edge between two
    nodes picked at random (numpy's default_rng(0)), until there are EDGES different edges."""
    import numpy as np
    rng = np.random.default_rng(0)
    edges = set()
    while len(edges) < EDGES:
        i, j = rng.integers(AIRPORTS, size=2)
        if i != j:
            edges.add((min(i, j), max(i, j)))
    degree = np.zeros(AIRPORTS, dtype=int)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    return tuple(int(d) for d in degree)


def random_counts() -> dict:
    out: dict = {}
    for d in random_degrees():
        out[d] = out.get(d, 0) + 1
    return dict(sorted(out.items()))


@functools.lru_cache(maxsize=1)
def fit() -> tuple:
    """The straight line closest to the dots on the log-log plot from degree 2 to 100: its slope and where it is at
    degree 1, as logs with base 10."""
    import numpy as np
    ks = [k for k in COUNTS if 2 <= k <= 100]
    slope, at_one = np.polyfit(np.log10(ks), np.log10([COUNTS[k] for k in ks]), 1)
    return float(slope), float(at_one)


def degrees_picture(labels: dict) -> str:
    """The degree distribution twice: on normal axes, and on a log-log plot with the random network's as open dots and
    the straight line of `fit` from degree 2 to 100, dashed. `labels`: "normal", "loglog", "degree", "airports",
    "alt"."""
    normal = chart.Panel(x=(0, 250), y=(0, 800), title=labels["normal"], xlabel=labels["degree"], ylabel=labels["airports"])
    normal.points(list(COUNTS), list(COUNTS.values()), role="data", r=2.5)
    loglog = chart.Panel(x=(1, 300), y=(1, 1000), title=labels["loglog"], xlabel=labels["degree"], ylabel=labels["airports"],
                         xlog=True, ylog=True)
    slope, at_one = fit()
    loglog.line([2, 100], [10 ** (at_one + slope * 0.30103), 10 ** (at_one + slope * 2)], role="ink-3", width=1.6, dashed=True)
    rc = random_counts()
    loglog.points(list(rc), list(rc.values()), role="data", r=3, hollow=True)
    loglog.points(list(COUNTS), list(COUNTS.values()), role="data", r=3)
    return chart.figure([normal, loglog], alt=labels["alt"])


def closed_picture(h: int, labels: dict) -> str:
    """With h airports closed: the share of the pairs of open airports at most 1 to 10 flights apart, with the h
    biggest hubs closed (solid), with h airports closed at random (dashed) and with every airport open (grey).
    `labels`: "title", "flights", "pairs", "alt"."""
    p = chart.Panel(x=(1, 10), y=(0, 100), title=labels["title"], xlabel=labels["flights"], ylabel=labels["pairs"],
                    xticks=list(range(1, 11)))
    xs = list(range(1, 11))
    p.line(xs, CLOSED[(0, "hubs")][3], role="ink-3", width=2.2)
    p.line(xs, CLOSED[(h, "random")][3], role="data", width=2.2, dashed=True)
    p.line(xs, CLOSED[(h, "hubs")][3], role="data", width=2.4)
    return chart.figure([p], alt=labels["alt"], width=460, height=270)


def closed_said(h: int, lang: str) -> str:
    """The sentence under the chart of closed_picture."""
    n = lambda x, d=0: number(x, lang, d)   # noqa: E731
    o, c, m, w = CLOSED[(h, "hubs")]
    if h == 0:
        if lang == "nl":
            return (f"Elk vliegveld open: {n(c)} van de {n(o)} vliegvelden hangen samen. Van alle paren vliegvelden "
                    f"liggen er {n(w[3], 1)}% hoogstens 4 vluchten uit elkaar, en twee vliegvelden die elkaar kunnen "
                    f"bereiken gemiddeld {n(m, 2)} vluchten.")
        return (f"Every airport open: {n(c)} of the {n(o)} airports hang together. Of all pairs of airports, "
                f"{n(w[3], 1)}% are at most 4 flights apart, and two airports that can reach each other are on average "
                f"{n(m, 2)} flights apart.")
    _, rc, rm, rw = CLOSED[(h, "random")]
    if lang == "nl":
        return (f"De {n(h)} grootste hubs dicht (doorgetrokken): van de {n(o)} vliegvelden die open zijn, hangen er "
                f"{n(c)} samen, {n(w[3], 1)}% van de paren ligt hoogstens 4 vluchten uit elkaar, gemiddeld {n(m, 2)} "
                f"vluchten. {n(h)} vliegvelden willekeurig dicht (gestippeld): {n(rc)} hangen samen, {n(rw[3], 1)}% "
                f"binnen 4 vluchten, gemiddeld {n(rm, 2)}. Grijs: elk vliegveld open, {n(CLOSED[(0, 'hubs')][3][3], 1)}%.")
    return (f"The {n(h)} biggest hubs closed (solid): of the {n(o)} airports still open, {n(c)} hang together, "
            f"{n(w[3], 1)}% of the pairs are at most 4 flights apart, on average {n(m, 2)} flights. {n(h)} airports "
            f"closed at random (dashed): {n(rc)} hang together, {n(rw[3], 1)}% within 4 flights, on average "
            f"{n(rm, 2)}. Grey: every airport open, {n(CLOSED[(0, 'hubs')][3][3], 1)}%.")


# --- the download ----------------------------------------------------------------------------------------------------

def flights_or_stop(button, texts: dict):
    """everything_ai.flights for the lesson's cells, once routes.dat is downloaded. When it is not there yet, the cell
    stops with `texts["ask"]` and the button; a click downloads it, with the progress in the cell. `texts` has "ask",
    "busy" ({done} and {total} in MB, with one decimal), "no_net", "not_pinned", "no_room" and "failed" ({why}): why it
    did not work; and "here"."""
    import marimo as mo
    failed = None
    if not flights.ready() and button.value:
        total = number(hub.PINNED[flights.ROUTES]["size"] / 1e6, "en", 1)

        def progress(done, size):
            mo.output.replace(mo.md(texts["busy"].format(done=number(done / 1e6, "en", 1), total=total)))
        try:
            hub.download(*flights.ROUTES, progress=progress)
        except hub.NotPinned:
            failed = texts["not_pinned"]
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            failed = texts["no_net"]
        except OSError as e:
            failed = texts["no_room"] if e.errno == errno.ENOSPC else texts["failed"].format(why=html.escape(str(e)))
    if not flights.ready():
        mo.stop(True, download_card(texts["ask"], button, failed))
    mo.output.replace(mo.md(texts["here"]))
    return flights


# The lesson is done when these are correct. "fragile" is the choice question in the lesson.
REQUIRED = ["degree-counts", "flights-from", "fragile"]


class _Val:
    """A test value that the check's messages write by its name, such as `small`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value)


# The small network of the exercises: two hubs, H and K, joined by an edge; A and B fly to H and to each other.
SMALL = {"H": {"A", "B", "C", "D", "K"}, "K": {"H", "E", "F", "G"}, "A": {"H", "B"}, "B": {"H", "A"},
         "C": {"H"}, "D": {"H"}, "E": {"K"}, "F": {"K"}, "G": {"K"}}
_SMALL = _Val("small", SMALL)
_LINE = _Val("line", {"P": {"Q"}, "Q": {"P", "R"}, "R": {"Q", "S"}, "S": {"R"}})
_STAR = _Val("star", {"X": {"P", "Q", "R", "S", "T"}, "P": {"X"}, "Q": {"X"}, "R": {"X"}, "S": {"X"}, "T": {"X"}})
_TWO = _Val("two_parts", {"A": {"B"}, "B": {"A"}, "C": {"D", "E"}, "D": {"C", "E"}, "E": {"C", "D"}, "F": set()})
_RING = _Val("ring", {"A": {"B", "F"}, "B": {"A", "C"}, "C": {"B", "D"}, "D": {"C", "E"}, "E": {"D", "F"}, "F": {"E", "A"}})
_LISTS = _Val("lists", {"M": ["N", "O"], "N": ["M"], "O": ["M", "P"], "P": ["O"]})


def _shown(v):
    if isinstance(v, dict) and len(v) > 12:
        return {"en": f"a dictionary of {len(v):,} entries", "nl": f"een dictionary van {number(len(v), 'nl')} items"}
    return repr(v)


_DOTS = Mistake("dots-left", matches=lambda got, text: any("..." in line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _call(seen, *args):
    fn = seen.mine.names.get(seen.exercise.function) if seen.mine is not None else None
    if not callable(fn):
        return None
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return checking.with_time_limit(lambda: fn(*(checking._copy(a) for a in args)))
    except Exception:
        return None


def _key(text: str) -> str:
    return ('<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.15rem">'
            + text + "</div>")


def _whole_counts(got) -> dict | None:
    """The entries of a degree distribution that can be drawn: a whole number of nodes above 0 at a whole degree."""
    if not isinstance(got, dict):
        return None
    out = {}
    for k, v in got.items():
        if isinstance(k, bool) or isinstance(v, bool) or not isinstance(k, (int, float)) or not isinstance(v, (int, float)):
            return None
        if k == int(k) and v == int(v) and v > 0:
            out[int(k)] = int(v)
    return out


# --- exercise 1: how many nodes have each degree ---------------------------------------------------------------------

_D_HEAD = "def degree_counts(graph):\n    counts = {}\n    for node in graph:\n"


def _d(degree: str = "len(graph[node])", count: str = "counts[degree] + 1") -> str:
    return (_D_HEAD + f"        degree = {degree}\n        if degree not in counts:\n            counts[degree] = 0\n"
            f"        counts[degree] = {count}\n    return counts\n")


_D_START = _d("...", "...")
_D = _d()


def degree_counts(graph) -> dict:
    """The course's degree_counts: for every degree, how many nodes of graph have it."""
    counts: dict = {}
    for node in graph:
        counts[len(graph[node])] = counts.get(len(graph[node]), 0) + 1
    return counts


_DC_WORDS = {
    "title_all": {"en": "Your degree_counts for all 3,425 airports", "nl": "Jouw degree_counts voor alle 3.425 vliegvelden"},
    "title_small": {"en": "Your degree_counts for the small network", "nl": "Jouw degree_counts voor het kleine netwerk"},
    "degree": {"en": "degree", "nl": "graad"},
    "nodes": {"en": "airports", "nl": "vliegvelden"},
    "small_nodes": {"en": "nodes", "nl": "nodes"},
    "right_all": {"en": "A dot for every degree your function counts, on a log-log plot: 768 airports with degree 1, 744 with degree 2, and one each for the biggest hubs, up to the 248 of Amsterdam. From degree 2 to about 100 the dots fall along a straight line; above that, they thin out.",
                  "nl": "Een stip voor elke graad die je functie telt, op een log-log plot: 768 vliegvelden met graad 1, 744 met graad 2, en één voor elk van de grootste hubs, tot de 248 van Amsterdam. Van graad 2 tot ongeveer 100 liggen de stippen langs een rechte lijn; daarboven dunnen ze uit."},
    "said_all": {"en": "A dot for every degree your function counts, on a log-log plot: {k} degrees, which add up to {total} airports. The course counts 144 degrees, which add up to all 3,425 airports.",
                 "nl": "Een stip voor elke graad die je functie telt, op een log-log plot: {k} graden, samen {total} vliegvelden. De course telt 144 graden, samen alle 3.425 vliegvelden."},
    "right_small": {"en": "A bar for every degree your function counts in the small network: 5 nodes with degree 1, 2 with degree 2, and the hubs, K with 4 and H with 5. Download the routes above the worked example, and at your next check this picture shows your counts for all 3,425 airports.",
                    "nl": "Een staaf voor elke graad die je functie telt in het kleine netwerk: 5 nodes met graad 1, 2 met graad 2, en de hubs, K met 4 en H met 5. Download de routes boven het uitgewerkte voorbeeld, dan laat dit plaatje bij je volgende check jouw tellingen voor alle 3.425 vliegvelden zien."},
    "said_small": {"en": "A bar for every degree your function counts in the small network: {k} degrees, which add up to {total} nodes. The small network has 9 nodes, in 4 degrees: 1, 2, 4 and 5.",
                   "nl": "Een staaf voor elke graad die je functie telt in het kleine netwerk: {k} graden, samen {total} nodes. Het kleine netwerk heeft 9 nodes, met 4 graden: 1, 2, 4 en 5."},
}


def _dc_picture(seen) -> str:
    lang = seen.lang
    words_ = {k: v[lang] for k, v in _DC_WORDS.items()}
    whole = flights.ready()
    graph = flights.graph() if whole else SMALL
    got = _whole_counts(_call(seen, graph))
    if not got:
        return ""
    right = checking.same_value(got, degree_counts(graph))
    total = sum(got.values())
    if whole:
        top_x = max(300, max(got) * 1.2)
        top_y = max(1000, max(got.values()) * 1.2)
        p = chart.Panel(x=(1, top_x), y=(1, top_y), xlabel=words_["degree"], ylabel=words_["nodes"], xlog=True, ylog=True)
        p.points(list(got), list(got.values()), role="data", r=3)
        text = words_["right_all"] if right else words_["said_all"].format(k=number(len(got), lang), total=number(total, lang))
        return yours.figure(words_["title_all"], chart.figure([p], alt=text, width=460, height=260) + _key(text), alt=text)
    top_x = max(got) + 1
    top_y = max(got.values()) + 1
    p = chart.Panel(x=(0, top_x), y=(0, top_y), xlabel=words_["degree"], ylabel=words_["small_nodes"],
                    xticks=list(range(0, top_x + 1)) if top_x <= 12 else None,
                    yticks=list(range(0, top_y + 1)) if top_y <= 8 else None)   # whole numbers of nodes
    for k, v in got.items():
        p.rect(k - 0.35, 0, k + 0.35, v, role="data", solid=True)
    text = words_["right_small"] if right else words_["said_small"].format(k=number(len(got), lang), total=number(total, lang))
    return yours.figure(words_["title_small"], chart.figure([p], alt=text, width=330, height=220) + _key(text), alt=text)


def _counts_are_degrees(got, text) -> bool:
    """The degree of every node instead of how many nodes have each degree: the node names as keys."""
    return isinstance(got, dict) and len(got) > 0 and all(isinstance(k, str) for k in got)


EXERCISES = {
    "degree-counts": FunctionBox(
        name="degree-counts",
        picture=_dc_picture,
        function="degree_counts",
        start=_D_START,
        solution=_D,
        show=_shown,
        visible=[(_SMALL,)],
        hidden=[(_LINE,), (_STAR,), (_TWO,), (_LISTS,)],
        first=[_DOTS],
        mistakes=[
            Mistake("ones", wrong=_d(count="1"),
                    en="`{call}` gives {got}: every degree once, with 1. Five nodes of the small network have degree 1, but your count stays at 1. Add 1 to the count each time: `counts[degree] + 1`.",
                    nl="`{call}` geeft {got}: elke graad één keer, met 1. Vijf nodes van het kleine netwerk hebben graad 1, maar jouw telling blijft op 1 staan. Tel er elke keer 1 bij op: `counts[degree] + 1`."),
            Mistake("nodes", wrong=_d(degree="len(graph)"),
                    en="`{call}` gives {got}: `len(graph)` is the number of nodes of the whole network, the same for every node. The degree of one node is the number of its neighbors: `len(graph[node])`.",
                    nl="`{call}` geeft {got}: `len(graph)` is het aantal nodes van het hele netwerk, voor elke node hetzelfde. De graad van één node is het aantal buren: `len(graph[node])`."),
            Mistake("degree-sum", wrong=_d(count="counts[degree] + degree"),
                    en="`{call}` gives {got}: for each degree you add up the degrees, not the nodes. Each node adds 1 to the count of its degree.",
                    nl="`{call}` geeft {got}: per graad tel je de graden op, niet de nodes. Elke node telt 1 op bij de telling van zijn graad."),
            Mistake("degrees", matches=_counts_are_degrees,
                    en="`{call}` gives {got}: the degree of every node, with the names of the nodes as keys. The check wants the other way around: for every degree, as key, how many nodes have it.",
                    nl="`{call}` geeft {got}: de graad van elke node, per node, met de naam van de node vooraan. De check wil het andersom: voor elke graad hoeveel nodes die hebben."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the two `...`: the degree of `node`, and its count plus one.",
                      "nl": "Vervang de twee `...`: de graad van `node`, en zijn telling plus één."},
            "arguments": {"en": "The check calls `{call}`, with one value, and the `def` line of your function has a different number of names between its parentheses. Keep `(graph)`.",
                          "nl": "De check roept `{call}` aan, met één waarde, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(graph)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For every node: its degree is the length of its set of neighbors, and the count of that degree goes up by 1.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Voor elke node: zijn graad is de lengte van zijn verzameling buren, en de telling van die graad gaat 1 omhoog."},
            "almost": {"en": "Right for the small network. But `{call}` gives {got}, and it should give {want}. Does your function use `graph`, and nothing of the small network?",
                       "nl": "Goed voor het kleine netwerk. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `graph`, en niets van het kleine netwerk?"},
            "correct": {"en": "Your function is right in all {n} cases. For the network of flights it gives 144 degrees: 768 airports with degree 1, and at the other end one airport each with 217, 236, 240, 244 and 248.",
                        "nl": "Je functie klopt in alle {n} gevallen. Voor het netwerk van vluchten geeft hij 144 graden: 768 vliegvelden met graad 1, en aan de andere kant één vliegveld elk met 217, 236, 240, 244 en 248."},
        },
    ),
}


# --- exercise 2: the fewest flights from one airport to every other ---------------------------------------------------

_F_HEAD = ("from collections import deque\n\n\ndef flights_from(graph, start):\n    flights = {start: 0}\n"
           "    frontier = deque([start])\n    while len(frontier) > 0:\n        airport = frontier.popleft()\n"
           "        for nxt in graph[airport]:\n")


def _f(test: str = "nxt not in flights", value: str = "flights[airport] + 1") -> str:
    return (_F_HEAD + f"            if {test}:\n                flights[nxt] = {value}\n                frontier.append(nxt)\n"
            "    return flights\n")


_F_START = _f("...", "...")
_F = _f()


def flights_from(graph, start) -> dict:
    """The course's flights_from: the fewest steps from start to every node it can reach, by breadth-first search."""
    from collections import deque
    out = {start: 0}
    frontier = deque([start])
    while frontier:
        node = frontier.popleft()
        for nxt in graph[node]:
            if nxt not in out:
                out[nxt] = out[node] + 1
                frontier.append(nxt)
    return out


def _within(got) -> list | None:
    """From a result of flights_from: how many other nodes it reaches with at most 1, 2, ... steps."""
    if not isinstance(got, dict) or not got:
        return None
    steps = [v for v in got.values() if isinstance(v, (int, float)) and not isinstance(v, bool) and v == int(v) and v >= 0]
    if len(steps) != len(got) or max(steps) > 60:
        return None
    return [sum(1 for v in steps if 0 < v <= k) for k in range(1, int(max(steps)) + 1)]


_FF_WORDS = {
    "title_all": {"en": "Your flights_from Amsterdam and from Abilene", "nl": "Jouw flights_from Amsterdam en vanuit Abilene"},
    "title_small": {"en": "Your flights_from(small, \"C\")", "nl": "Jouw flights_from(small, \"C\")"},
    "flights": {"en": "at most this many flights", "nl": "hoogstens zoveel vluchten"},
    "reached": {"en": "airports reached", "nl": "bereikte vliegvelden"},
    "right_all": {"en": "Solid: from Amsterdam, 248 other airports with at most 1 flight, 1,861 with at most 2 and all 3,396 of the big part with at most 7. Dashed: from Abilene, with one neighbor, 1, then 188, then 1,361 with at most 3, and all 3,396 with at most 9. One flight to a hub, and you are almost everywhere.",
                  "nl": "Doorgetrokken: vanuit Amsterdam 248 andere vliegvelden met hoogstens 1 vlucht, 1.861 met hoogstens 2 en alle 3.396 van het grote deel met hoogstens 7. Gestippeld: vanuit Abilene, met één buur, 1, dan 188, dan 1.361 met hoogstens 3, en alle 3.396 met hoogstens 9. Eén vlucht naar een hub, en je bent bijna overal."},
    "said_all": {"en": "Solid: from Amsterdam, your function reaches {ams} other airports, with at most {ams_k} flights. Dashed: from Abilene, {abi}, with at most {abi_k}. The course's reaches all 3,396 other airports of the big part from both: from Amsterdam with at most 7 flights, from Abilene with at most 9.",
                 "nl": "Doorgetrokken: vanuit Amsterdam bereikt je functie {ams} andere vliegvelden, met hoogstens {ams_k} vluchten. Gestippeld: vanuit Abilene {abi}, met hoogstens {abi_k}. Die van de course bereikt vanuit allebei alle 3.396 andere vliegvelden van het grote deel: vanuit Amsterdam met hoogstens 7 vluchten, vanuit Abilene met hoogstens 9."},
    "right_small": {"en": "Your numbers for the small network from C, by number of flights. From C everything is at most 3 flights away: H in 1, the hub K and the other neighbors of H in 2, and the neighbors of K in 3. Download the routes above the worked example, and at your next check this picture shows the whole network of flights.",
                    "nl": "Jouw getallen voor het kleine netwerk vanuit C, per aantal vluchten. Vanuit C ligt alles hoogstens 3 vluchten weg: H in 1, de hub K en de andere buren van H in 2, en de buren van K in 3. Download de routes boven het uitgewerkte voorbeeld, dan laat dit plaatje bij je volgende check het hele netwerk van vluchten zien."},
    "said_small": {"en": "Your numbers for the small network from C, by number of flights. The course's: C 0, H 1, A, B, D and K 2, and E, F and G 3.",
                   "nl": "Jouw getallen voor het kleine netwerk vanuit C, per aantal vluchten. Die van de course: C 0, H 1, A, B, D en K 2, en E, F en G 3."},
}


def _ff_picture(seen) -> str:
    lang = seen.lang
    words_ = {k: v[lang] for k, v in _FF_WORDS.items()}
    if flights.ready():
        graph = flights.graph()
        ams, abi = _call(seen, graph, "AMS"), _call(seen, graph, "ABI")
        lines = [_within(ams), _within(abi)]
        if not all(lines):
            return ""
        longest = max(len(w) for w in lines)
        p = chart.Panel(x=(0, longest), y=(0, max(3500, max(max(w) for w in lines) * 1.1)), xlabel=words_["flights"],
                        ylabel=words_["reached"], xticks=list(range(0, longest + 1)) if longest <= 14 else None)
        for w, dashed in zip(lines, (False, True)):
            p.line(list(range(0, len(w) + 1)), [0] + w, role="data", width=2.2, dashed=dashed)
        right = checking.same_value(ams, flights_from(graph, "AMS")) and checking.same_value(abi, flights_from(graph, "ABI"))
        text = words_["right_all"] if right else words_["said_all"].format(
            ams=number(lines[0][-1], lang), ams_k=len(lines[0]), abi=number(lines[1][-1], lang), abi_k=len(lines[1]))
        return yours.figure(words_["title_all"], chart.figure([p], alt=text, width=460, height=260) + _key(text), alt=text)
    got = _call(seen, SMALL, "C")
    if _within(got) is None:
        return ""
    rings: dict = {}
    for node, k in sorted(got.items(), key=lambda kv: (kv[1], str(kv[0]))):
        rings.setdefault(int(k), []).append(html.escape(str(node)))
    rows = "".join(f'<span style="color:var(--ea-ink-2);text-align:right">{k}</span>'
                   f'<span style="color:var(--ea-ink)">{", ".join(names)}</span>' for k, names in rings.items())
    body = (f'<div style="display:grid;grid-template-columns:2.2rem minmax(0,1fr);gap:0.2rem 0.6rem;font-family:var(--ea-font-ui);'
            f'font-size:0.9rem;max-width:460px;margin:0.3rem 0">{rows}</div>')
    text = words_["right_small"] if checking.same_value(got, flights_from(SMALL, "C")) else words_["said_small"]
    return yours.figure(words_["title_small"], body + _key(text), alt=text)


EXERCISES["flights-from"] = FunctionBox(
    name="flights-from",
    picture=_ff_picture,
    function="flights_from",
    start=_F_START,
    solution=_F,
    show=_shown,
    visible=[(_SMALL, "C")],
    hidden=[(_SMALL, "H"), (_LINE, "P"), (_TWO, "C"), (_RING, "A"), (_LISTS, "P")],
    first=[_DOTS],
    mistakes=[
        Mistake("all-one", wrong=_f(value="1"),
                en="`{call}` gives {got}: every airport 1 flight from the start. That holds only for its neighbors. A neighbor of `airport` is one flight further than `airport` itself: `flights[airport] + 1`.",
                nl="`{call}` geeft {got}: elk vliegveld 1 vlucht van het begin. Dat klopt alleen voor de buren. Een buur van `airport` ligt één vlucht verder dan `airport` zelf: `flights[airport] + 1`."),
        Mistake("no-step", wrong=_f(value="flights[airport]"),
                en="`{call}` gives {got}: every airport 0 flights from the start. A neighbor of `airport` is one flight further: add 1.",
                nl="`{call}` geeft {got}: elk vliegveld 0 vluchten van het begin. Een buur van `airport` ligt één vlucht verder: tel er 1 bij op."),
        Mistake("order", wrong=_f(value="len(flights)"),
                en="`{call}` gives {got}: `len(flights)` counts the airports found so far, so it goes up by one at every airport. The number of flights to a neighbor is that of `airport` plus 1.",
                nl="`{call}` geeft {got}: `len(flights)` telt de vliegvelden die tot nu toe gevonden zijn, dus gaat het bij elk vliegveld één omhoog. Het aantal vluchten naar een buur is dat van `airport` plus 1."),
    ],
    texts=_SAME | {
        "empty": {"en": "Replace the two `...`: when a neighbor gets a number of flights, and which number.",
                  "nl": "Vervang de twee `...`: wanneer een buur een aantal vluchten krijgt, en welk aantal."},
        "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(graph, start)`.",
                      "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(graph, start)`."},
        "slow": {"en": "Your function is still not done after a few seconds. Does an airport go into the frontier again and again? Give a neighbor a number, and put it in the frontier, only when it has no number yet: `nxt not in flights`.",
                 "nl": "Je functie is na een paar seconden nog niet klaar. Komt een vliegveld steeds opnieuw in de frontier? Geef een buur alleen een aantal, en zet hem alleen in de frontier, als hij nog geen aantal heeft: `nxt not in flights`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}. A neighbor that has no number yet gets the number of `airport` plus 1; one that has a number keeps it, as it was found with fewer flights.",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Een buur die nog geen aantal heeft, krijgt het aantal van `airport` plus 1; een buur die er al een heeft, houdt dat, want die is met minder vluchten gevonden."},
        "almost": {"en": "Right for the small network from C. But `{call}` gives {got}, and it should give {want}. Does your function use `graph` and `start`, and nothing of the example?",
                   "nl": "Goed voor het kleine netwerk vanuit C. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `graph` en `start`, en niets van het voorbeeld?"},
        "correct": {"en": "Your function is right in all {n} cases. From Abilene, with one neighbor, it reaches 1,361 other airports with at most 3 flights, and all 3,396 of the big part with at most 9.",
                    "nl": "Je functie klopt in alle {n} gevallen. Vanuit Abilene, met één buur, bereikt hij 1.361 andere vliegvelden met hoogstens 3 vluchten, en alle 3.396 van het grote deel met hoogstens 9."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("degree-counts", _D_START, "empty", None),
    ("degree-counts", _d(count="..."), "wrong", "dots-left"),
    ("degree-counts", _D, "correct", None),
    ("degree-counts", _d(count="1 + counts[degree]"), "correct", None),
    ("degree-counts", _d(count="1"), "wrong", "ones"),
    ("degree-counts", _d(degree="len(graph)"), "wrong", "nodes"),
    ("degree-counts", _d(count="counts[degree] + degree"), "wrong", "degree-sum"),
    ("degree-counts", "def degree_counts(graph):\n    counts = {}\n    for node in graph:\n        counts[node] = len(graph[node])\n    return counts\n",
     "wrong", "degrees"),
    ("degree-counts", "def degree_counts(graph):\n    return {5: 1, 4: 1, 2: 2, 1: 5}\n", "almost", None),
    ("flights-from", _F_START, "empty", None),
    ("flights-from", _f(value="..."), "wrong", "dots-left"),
    ("flights-from", _F, "correct", None),
    ("flights-from", _f(test="flights.get(nxt) is None"), "correct", None),
    ("flights-from", _f(value="1"), "wrong", "all-one"),
    ("flights-from", _f(value="flights[airport]"), "wrong", "no-step"),
    ("flights-from", _f(value="len(flights)"), "wrong", "order"),
    ("flights-from", _f(value="flights[start] + 1"), "wrong", "all-one"),
    ("flights-from", _f(test="True"), "slow", None),
    ("flights-from", "def flights_from(graph, start):\n    return {\"C\": 0, \"H\": 1, \"A\": 2, \"B\": 2, \"D\": 2, \"K\": 2, \"E\": 3, \"F\": 3, \"G\": 3}\n",
     "almost", None),
]
