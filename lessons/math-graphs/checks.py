# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson math-graphs: the graphs of the lesson and the course's own adjacency list and matrix, which
draw the figures and the play block; to_list and to_matrix, the two the learner writes; and the pictures of what they
build, drawn as the graph it stores."""
import re

import numpy as np

from everything_ai import draw, yours
from everything_ai.check import FunctionBox, Mistake


class Named(list):
    """A list the check calls by the name the lesson gives it, such as EDGES, so that a check message stays short; the
    learner's code gets a plain copy."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return list(self)


# --- the graphs of the lesson -------------------------------------------------------------------------------------

# Seven people, A to G, and who knows whom: an edge has no direction.
NAMES = Named(["A", "B", "C", "D", "E", "F", "G"], "NAMES")
EDGES = Named([("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "E"),
               ("D", "E"), ("D", "F"), ("E", "F"), ("F", "G")], "EDGES")
AT = {"A": (0, 1), "B": (1, 0), "C": (1, 2), "D": (2, 0), "E": (2, 2), "F": (3, 1), "G": (4, 1)}

# Four towns, numbered 0 to 3, and the roads between them with their length in km.
ROADS = Named([(0, 1, 4), (1, 2, 3), (0, 2, 9), (2, 3, 2)], "ROADS")
ROAD_AT = {0: (0, 1), 1: (1, 0), 2: (2, 1), 3: (3, 1)}


def to_list(names, edges):
    """The course's adjacency list, as in the lesson: every name a list of its neighbors, in the order of the edges."""
    graph = {}
    for name in names:
        graph[name] = []
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    return graph


def to_matrix(names, edges):
    """The course's adjacency matrix of a graph without weights, as in the worked example."""
    number = {}
    for i in range(len(names)):
        number[names[i]] = i
    adj = np.zeros((len(names), len(names)))
    for a, b in edges:
        adj[number[a], number[b]] = 1
        adj[number[b], number[a]] = 1
    return adj


GRAPH = to_list(NAMES, EDGES)
ADJ = to_matrix(NAMES, EDGES)


def kinds(titles: dict, alt: str):
    """Three small graphs side by side: one without directions, one with, one with weights."""
    at = {"A": (0, 1), "B": (1, 0), "C": (2, 1), "D": (1, 2)}
    road = {"P": (0, 1), "Q": (1, 0), "R": (2, 1), "S": (3, 1)}
    return draw.graph_panels([
        {"title": titles["undirected"], "at": at, "unit": 58, "ink": True,
         "edges": [("A", "B"), ("B", "C"), ("A", "D"), ("C", "D"), ("B", "D")]},
        {"title": titles["directed"], "at": at, "unit": 58, "ink": True, "directed": True,
         "edges": [("A", "B"), ("C", "B"), ("B", "D"), ("D", "B"), ("D", "A")]},
        {"title": titles["weighted"], "at": road, "unit": 58, "ink": True,
         "edges": [("P", "Q", 4), ("Q", "R", 3), ("P", "R", 9), ("R", "S", 2)]},
    ], alt=alt)


def views(k: int, titles: dict, alt: str):
    """The graph of the seven people, its adjacency list and its adjacency matrix, with node number `k` marked in all
    three: its ring and edges in the drawing, its line in the list, its row in the matrix. For the play block."""
    name = NAMES[k]
    return draw.graph_panels([
        {"title": titles["graph"], "at": AT, "edges": EDGES, "mark": name, "near": GRAPH[name]},
        {"kind": "list", "title": titles["list"], "rows": [(n, GRAPH[n]) for n in NAMES], "mark": name},
        {"kind": "matrix", "title": titles["matrix"], "names": [f"{i} {n}" for i, n in enumerate(NAMES)],
         "values": [[int(v) for v in row] for row in ADJ], "mark": k},
    ], alt=alt)


# --- how the learner's graph is judged ----------------------------------------------------------------------------

def _neighbor_lists(value):
    """A dictionary of neighbor lists with every list sorted, so that any order of the neighbors counts as right; a
    set or a tuple counts as a list. Anything else stays as it is."""
    if not isinstance(value, dict):
        return value
    out = {}
    for k, v in value.items():
        if isinstance(v, (list, tuple, set)):
            out[k] = sorted(v, key=repr)
        else:
            return value
    return out


def same_graph(a, b) -> bool:
    return _neighbor_lists(a) == _neighbor_lists(b)


def _one_way(got) -> bool:
    """A dictionary of lists in which some name has a neighbor that does not have it back."""
    if not isinstance(got, dict) or not all(isinstance(v, (list, tuple, set)) for v in got.values()):
        return False
    return any(a not in got.get(b, ()) for a, nbrs in got.items() for b in nbrs)


def _not_lists(got) -> bool:
    return isinstance(got, dict) and any(not isinstance(v, (list, tuple, set)) for v in got.values())


def _square(got) -> bool:
    return isinstance(got, np.ndarray) and got.ndim == 2 and got.shape[0] == got.shape[1]


def _not_symmetric(got) -> bool:
    return _square(got) and got.dtype.kind in "biuf" and not (got == got.T).all()


def _only_ones(got) -> bool:
    return (_square(got) and got.dtype.kind in "biuf" and (got == got.T).all() and (got != 0).any()
            and set(got[got != 0].tolist()) == {1})


# The lesson is done when these are correct. "column" is the choice question in the lesson.
REQUIRED = ["to-list", "to-matrix", "column"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


# --- exercise 1: to_list --------------------------------------------------------------------------------------------

def _l(put: str = "graph[a].append(b)\ngraph[b].append(a)") -> str:
    """to_list as the box holds it, with these lines for each edge: step 2 of the recipe."""
    put = put.replace("\n", "\n        ")
    return ("def to_list(names, edges):\n"
            "    graph = {}\n"
            "    for name in names:\n"
            "        graph[name] = []\n"
            "    for a, b in edges:\n"
            f"        {put}\n"
            "    return graph\n")


_L_START = _l("...")
_L = _l()

# --- exercise 2: to_matrix ------------------------------------------------------------------------------------------


def _m(make: str = "adj = np.zeros((n, n))", put: str = "adj[i, j] = w\nadj[j, i] = w") -> str:
    """to_matrix as the box holds it, with these lines: the matrix of zeros, and what each edge puts in it."""
    put = put.replace("\n", "\n        ")
    return ("import numpy as np\n\n\n"
            "def to_matrix(n, edges):\n"
            f"    {make}\n"
            "    for i, j, w in edges:\n"
            f"        {put}\n"
            "    return adj\n")


_M_START = _m("adj = ...", "...")
_M = _m()


def _matrix_picture(seen) -> str:
    """The learner's matrix as numbers, beside the right one where it differs, and as the graph it stores."""
    return yours.arrays(0)(seen) + yours.graph(0, at=ROAD_AT)(seen)


EXERCISES = {
    "to-list": FunctionBox(
        name="to-list",
        picture=yours.graph(0, at=AT),
        function="to_list",
        start=_L_START,
        solution=_L,
        compare=same_graph,
        visible=[(NAMES, EDGES)],
        hidden=[(["P", "Q", "R"], [("P", "Q")]), (["x"], []), (["K", "L", "M", "N"], [("K", "L"), ("M", "N"), ("L", "M")])],
        first=[_DOTS],
        mistakes=[
            Mistake("overwrite", wrong=_l("graph[a] = [b]\ngraph[b] = [a]"),
                    en="`{call}` gives {got}, and it should give {want}. `graph[a] = [b]` puts a new list in the place of the old one, so each name keeps only its last neighbor. Add to the list that is there with `append`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. `graph[a] = [b]` zet een nieuwe lijst op de plek van de oude, dus elke naam houdt alleen zijn laatste buur. Voeg toe aan de lijst die er al staat, met `append`."),
            Mistake("not-a-list", matches=lambda got, text: _not_lists(got),
                    en="`{call}` gives {got}. A name now has one neighbor where its list belongs: `graph[a] = b` replaces the list by a name. Keep the list and add the neighbor to it: `graph[a].append(b)`.",
                    nl="`{call}` geeft {got}. Een naam heeft nu één buur waar zijn lijst hoort: `graph[a] = b` vervangt de lijst door een naam. Houd de lijst en voeg de buur eraan toe: `graph[a].append(b)`."),
            Mistake("one-way", matches=lambda got, text: _one_way(got),
                    en="`{call}` does not give the right graph: some names have a neighbor that does not have them back. You put each edge in one list only. An edge here has no direction, so `b` goes in the list of `a`, and `a` in the list of `b`.",
                    nl="`{call}` geeft niet de goede graaf: sommige namen hebben een buur die hen niet terug heeft. Je zet elke edge maar in één lijst. Een edge heeft hier geen richting, dus `b` komt in de lijst van `a`, en `a` in de lijst van `b`."),
            Mistake("append-two", error=lambda e: isinstance(e, TypeError) and "append" in str(e) and "argument" in str(e),
                    en="With `{call}`, your function stops with an error: `append` takes one value at a time. Write two lines: `graph[a].append(b)`, then `graph[b].append(a)`.",
                    nl="Met `{call}` stopt je functie met een fout: `append` neemt één waarde tegelijk. Schrijf twee regels: `graph[a].append(b)`, en dan `graph[b].append(a)`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with step 2 of the recipe: for an edge from `a` to `b`, put each of the two in the list of the other.",
                      "nl": "Vervang de `...` door stap 2 van het recept: zet bij een edge tussen `a` en `b` elk van de twee in de lijst van de ander."},
            "arguments": {"en": "The check calls `{call}`, with the names and the edges, and the `def` line of your function has another number of names between its parentheses. Keep `(names, edges)`.",
                          "nl": "De check roept `{call}` aan, met de namen en de edges, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(names, edges)`."},
            "wrong": {"en": "`{call}` does not give the right graph. The picture under this message draws yours beside the right one. Compare your lines with step 2 of the recipe.",
                      "nl": "`{call}` geeft niet de goede graaf. Het plaatje onder dit bericht tekent de jouwe naast de goede. Vergelijk je regels met stap 2 van het recept."},
            "almost": {"en": "Right for the seven people. But `{call}` gives {got}, and it should give {want}. Does every name get a list, also a name without edges?",
                       "nl": "Goed voor de zeven mensen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Krijgt elke naam een lijst, ook een naam zonder edges?"},
            "correct": {"en": "Your adjacency list is right in all {n} cases, also for a name without edges and for a graph without any.",
                        "nl": "Je adjacency list klopt in alle {n} gevallen, ook voor een naam zonder edges en voor een graaf zonder één edge."},
        },
    ),
    "to-matrix": FunctionBox(
        name="to-matrix",
        picture=_matrix_picture,
        function="to_matrix",
        start=_M_START,
        solution=_M,
        visible=[(4, ROADS)],
        hidden=[(5, [(0, 1, 2), (1, 2, 5), (3, 1, 7)]), (3, []), (2, [(1, 0, 6)])],
        first=[_DOTS],
        mistakes=[
            Mistake("zeros-two", error=lambda e: isinstance(e, TypeError) and "data type" in str(e),
                    en="With `{call}`, your function stops with an error: `np.zeros(n, n)` takes the second `n` as the kind of number. Give the shape as one tuple, with two pairs of parentheses: `np.zeros((n, n))`.",
                    nl="Met `{call}` stopt je functie met een fout: `np.zeros(n, n)` leest de tweede `n` als de soort getal. Geef de shape als één tuple, met twee paar haakjes: `np.zeros((n, n))`."),
            Mistake("zeros-vector", error=lambda e: isinstance(e, IndexError) and "too many indices" in str(e),
                    en="With `{call}`, your function stops with an error: `np.zeros(n)` makes a vector of `n` numbers, with one index, and `adj[i, j]` needs two. A matrix with a row and a column per node is `np.zeros((n, n))`.",
                    nl="Met `{call}` stopt je functie met een fout: `np.zeros(n)` maakt een vector van `n` getallen, met één index, en `adj[i, j]` heeft er twee nodig. Een matrix met een rij en een kolom per node is `np.zeros((n, n))`."),
            Mistake("too-small", error=lambda e: isinstance(e, IndexError) and "out of bounds" in str(e),
                    en="With `{call}`, your function stops with an error: a node number is larger than your matrix. The nodes are numbered 0 to `n - 1`, so the matrix needs `n` rows and `n` columns: `np.zeros((n, n))`.",
                    nl="Met `{call}` stopt je functie met een fout: een nodenummer is groter dan je matrix. De nodes hebben de nummers 0 tot en met `n - 1`, dus de matrix heeft `n` rijen en `n` kolommen nodig: `np.zeros((n, n))`."),
            Mistake("one-way", matches=lambda got, text: _not_symmetric(got),
                    en="`{call}` does not give the right matrix: your matrix has the weight of a road at row `i`, column `j`, but not at row `j`, column `i`. A road goes both ways, so it belongs in both places, and the matrix equals its transpose.",
                    nl="`{call}` geeft niet de goede matrix: je matrix heeft de lengte van een weg op rij `i`, kolom `j`, maar niet op rij `j`, kolom `i`. Een weg gaat twee kanten op, dus hij hoort op beide plekken, en de matrix is gelijk aan zijn getransponeerde."),
            Mistake("ones", matches=lambda got, text: _only_ones(got),
                    en="`{call}` does not give the right matrix: it holds a 1 for every road. In this graph, every edge has a weight: put the length `w` of the road in the matrix, so that it says how many km each road is.",
                    nl="`{call}` geeft niet de goede matrix: er staat een 1 voor elke weg. In deze graaf heeft elke edge een gewicht: zet de lengte `w` van de weg in de matrix, dan zegt hij hoeveel km elke weg is."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: make a matrix of zeros with a row and a column for each node, then put the length of each road in it.",
                      "nl": "Vervang de twee `...`: maak een matrix met nullen met een rij en een kolom voor elke node, en zet er dan de lengte van elke weg in."},
            "arguments": {"en": "The check calls `{call}`, with the number of nodes and the edges, and the `def` line of your function has another number of names between its parentheses. Keep `(n, edges)`.",
                          "nl": "De check roept `{call}` aan, met het aantal nodes en de edges, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(n, edges)`."},
            "wrong": {"en": "`{call}` does not give the right matrix. The picture under this message shows yours beside the right one, with the boxes that differ outlined.",
                      "nl": "`{call}` geeft niet de goede matrix. Het plaatje onder dit bericht laat de jouwe naast de goede zien, met de vakjes die verschillen omlijnd."},
            "almost": {"en": "Right for the four towns. But `{call}` does not give the right matrix. Does your function work for a node without edges, and for a graph without any?",
                       "nl": "Goed voor de vier steden. Maar `{call}` geeft niet de goede matrix. Werkt je functie ook voor een node zonder edges, en voor een graaf zonder één edge?"},
            "correct": {"en": "Your adjacency matrix is right in all {n} cases: the length of every road stands twice, at row `i`, column `j` and at row `j`, column `i`.",
                        "nl": "Je adjacency matrix klopt in alle {n} gevallen: de lengte van elke weg staat er twee keer in, op rij `i`, kolom `j` en op rij `j`, kolom `i`."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("to-list", _L_START, "empty", None),
    ("to-list", _l("graph[a].append(b)\n..."), "wrong", "dots-left"),
    ("to-list", _l("graph[a].append(b)"), "wrong", "one-way"),
    ("to-list", _l("graph[b].append(a)"), "wrong", "one-way"),
    ("to-list", _l("graph[a] = [b]\ngraph[b] = [a]"), "wrong", "overwrite"),
    ("to-list", _l("graph[a] = b\ngraph[b] = a"), "wrong", "not-a-list"),
    ("to-list", _l("graph[a].append(b, a)"), "error", "append-two"),
    ("to-list", _L, "correct", None),
    ("to-list", _l("graph[b].append(a)\ngraph[a].append(b)"), "correct", None),
    ("to-list", _l("graph[a] = graph[a] + [b]\ngraph[b] = graph[b] + [a]"), "correct", None),
    # another right one: sets of neighbors instead of lists
    ("to-list", "def to_list(names, edges):\n    graph = {}\n    for name in names:\n        graph[name] = set()\n"
                "    for a, b in edges:\n        graph[a].add(b)\n        graph[b].add(a)\n    return graph\n", "correct", None),
    # right for the seven people, whose names all have an edge, but it only makes a list for a name with an edge
    ("to-list", "def to_list(names, edges):\n    graph = {}\n    for a, b in edges:\n        if a not in graph:\n"
                "            graph[a] = []\n        if b not in graph:\n            graph[b] = []\n"
                "        graph[a].append(b)\n        graph[b].append(a)\n    return graph\n", "almost", None),
    ("to-matrix", _M_START, "empty", None),
    ("to-matrix", _m(put="..."), "wrong", "dots-left"),
    ("to-matrix", _m(put="adj[i, j] = w"), "wrong", "one-way"),
    ("to-matrix", _m(put="adj[i, j] = 1\nadj[j, i] = 1"), "wrong", "ones"),
    ("to-matrix", _m("adj = np.zeros(n, n)"), "error", "zeros-two"),
    ("to-matrix", _m("adj = np.zeros(n)"), "error", "zeros-vector"),
    ("to-matrix", _m("adj = np.zeros((n - 1, n - 1))"), "error", "too-small"),
    ("to-matrix", _M, "correct", None),
    ("to-matrix", _m(put="adj[i][j] = w\nadj[j][i] = w"), "correct", None),
    ("to-matrix", _m("adj = np.zeros((n, n), dtype=int)"), "correct", None),
    ("to-matrix", _m(put="adj[i, j] = adj[i, j] + w\nadj[j, i] = adj[j, i] + w"), "correct", None),
]
