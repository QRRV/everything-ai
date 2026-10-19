# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-embeddings-idea: most_similar, the row of a table of embeddings most like a given row by
cosine similarity. And the embeddings themselves: the course's digit reader (data/course-qmnist-cnn.pt) turns each of
the 1,000 test digits of QMNIST into 2 numbers, and the picture draws them as a map."""

import functools
import re

import numpy as np

from everything_ai import paths
from everything_ai.check import FunctionBox, Mistake, _show_rounded

DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md
SHOWN = 300                       # the picture draws the first 300 test digits, so that the map stays readable


@functools.lru_cache(maxsize=1)
def _embeddings():
    import torch
    from everything_ai.models import digit_cnn
    d = np.load(paths.ROOT / DATA)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)   # the same numbers on every computer
    try:
        model = digit_cnn()
        with torch.no_grad():
            E = model.embedding(torch.tensor(d["test_images"], dtype=torch.float32) / 255).double().numpy()
    finally:
        torch.set_num_threads(threads)
    return E, d["test_labels"].astype(int)


def embeddings():
    """The embeddings of the 1,000 test digits, one row of 2 numbers per image, and their labels."""
    E, labels = _embeddings()
    return E.copy(), labels.copy()


def map_picture(labels: dict) -> str:
    """The embedding map: each of the first SHOWN test digits drawn as its own label, at its 2 numbers."""
    from everything_ai import chart
    E, digits = _embeddings()
    panel = chart.Panel((-280, 280), (-280, 280), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    for (a, b), k in zip(E[:SHOWN], digits[:SHOWN]):
        panel.text(float(a), float(b), str(k), "hidden", "middle")
    return chart.figure([panel], alt=labels["alt"], width=330, height=330)


# The lesson is done when these are correct. "same-direction" and "most-like-4" are the choice questions in the lesson.
REQUIRED = ["same-direction", "most-similar", "most-like-4"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


class _Table(np.ndarray):
    """A table of embeddings that a check message writes the way the lesson's code does: np.array([[...], ...]).
    The learner's code gets a plain copy."""

    def __repr__(self):
        rows = ", ".join("[" + ", ".join(repr(float(v)) for v in row) + "]" for row in np.asarray(self).tolist())
        return f"np.array([{rows}])"

    def __deepcopy__(self, memo):
        return np.array(self)


def _t(rows) -> np.ndarray:
    return np.array(rows, dtype=float).view(_Table)


# --- exercise 2: the most similar row -----------------------------------------------------------------------------------------

_M_HEAD = "import numpy as np\n\n\ndef most_similar(E, i):\n"
_M_SETUP = "    best_j = -1\n    best = -2\n    for j in range(len(E)):\n"
_M_START = (_M_HEAD + "    # the number j of the row of E most like row i: the largest cosine similarity, with j not i\n" + _M_SETUP
            + "        ...\n    return best_j\n")
_COS = "np.dot(E[i], E[j]) / (np.linalg.norm(E[i]) * np.linalg.norm(E[j]))"


def _m(test: str = "j != i", cos: str = _COS, better: str = "cos > best") -> str:
    return (_M_HEAD + _M_SETUP + f"        if {test}:\n            cos = {cos}\n            if {better}:\n"
            "                best = cos\n                best_j = j\n    return best_j\n")


_DISTANCE = (_M_HEAD + "    best_j = -1\n    best = 10 ** 9\n    for j in range(len(E)):\n        if j != i:\n"
             "            d = np.linalg.norm(E[i] - E[j])\n            if d < best:\n                best = d\n                best_j = j\n    return best_j\n")

_VISIBLE = (_t([[1.0, 0.0], [10.0, 1.0], [0.5, 0.5], [-1.0, 0.0]]), 0)
_HIDDEN = [
    (_t([[1.0, 0.0], [0.9, 0.1], [5.0, 5.0], [0.0, -1.0]]), 0),        # the dot product alone picks the long row
    (_t([[1.0, 0.0], [10.0, 1.0], [0.5, 0.5], [-1.0, 0.0]]), 2),       # another row of the same table
    (_t([[0.0, 3.0], [2.0, 2.0], [-0.1, 1.0], [0.0, -3.0]]), 3),
    (_t([[3.0, 1.0, 0.0], [0.0, 1.0, 2.0], [6.0, 2.5, 0.5], [1.0, 1.0, 1.0], [-3.0, -1.0, 0.0]]), 1),
    (_t([[2.0, 1.0], [-2.0, -1.0]]), 0),                                 # only one other row
    (_t([[3.0, 4.0], [1.0, 0.0], [-4.0, 3.0]]), 1),                     # the answer is row 0: a loop from 1 misses it
]

EXERCISES = {
    "most-similar": FunctionBox(
        name="most-similar",
        function="most_similar",
        start=_M_START,
        solution=_m(),
        visible=[_VISIBLE],
        hidden=_HIDDEN,
        first=[_DOTS],
        show=lambda v: _show_rounded(v, 3),   # 0.995, not 0.9950371902099892
        mistakes=[
            Mistake("itself", wrong=_m(test="True"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is row `i` itself. Every row is most like itself, with a cosine similarity of 1. Skip `j` when it is `i`: `if j != i:`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is rij `i` zelf. Elke rij lijkt het meest op zichzelf, met een cosine similarity van 1. Sla `j` over als die `i` is: `if j != i:`."),
            Mistake("distance", wrong=_DISTANCE,
                    en="`{call}` gives `{got}`, and it should give `{want}`: you take the row that is closest by distance. Here, a row that points the same way can be far away, because it is longer. Compare the directions: the cosine similarity, the dot product divided by the two norms.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je neemt de rij die het dichtst bij ligt, gemeten met de afstand. Hier kan een rij die dezelfde kant op wijst ver weg liggen, omdat hij langer is. Vergelijk de richtingen: de cosine similarity, het inproduct gedeeld door de twee normen."),
            Mistake("dot-only", wrong=_m(cos="np.dot(E[i], E[j])"), state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: you compare the dot products alone, and then a long row wins even if it points another way. Divide by the two norms: `np.dot(E[i], E[j]) / (np.linalg.norm(E[i]) * np.linalg.norm(E[j]))`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je vergelijkt alleen de inproducten, en dan wint een lange rij ook als hij een andere kant op wijst. Deel door de twee normen: `np.dot(E[i], E[j]) / (np.linalg.norm(E[i]) * np.linalg.norm(E[j]))`."),
            Mistake("never-chosen", wrong=_m(better="cos < best"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `best_j` never changes, so no row gets chosen. When `cos` is larger than the best so far, keep it in `best` and its row number `j` in `best_j`. With `best = -2`, the first other row always gets kept, because a cosine similarity is never below −1.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `best_j` verandert nooit, dus er wordt geen rij gekozen. Is `cos` groter dan de beste tot nu toe, bewaar hem dan in `best` en zijn rijnummer `j` in `best_j`. Met `best = -2` wordt de eerste andere rij altijd bewaard, want een cosine similarity is nooit kleiner dan −1."),
            Mistake("best-value", wrong=_m().replace("return best_j", "return best"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the largest cosine similarity itself, not the number of its row. Keep the row number in `best_j = j`, and give back `best_j`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is de grootste cosine similarity zelf, niet het nummer van zijn rij. Bewaar het rijnummer met `best_j = j`, en geef `best_j` terug."),
            Mistake("from-one", wrong=_m().replace("range(len(E))", "range(1, len(E))"), state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: row 0 never gets compared, because your loop starts at 1. In the worked example it starts at 1 because image 0 is the given one; here the given row is `i`. Go over all rows, `for j in range(len(E)):`, and skip only row `i`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: rij 0 wordt nooit vergeleken, omdat je loop bij 1 begint. In het uitgewerkte voorbeeld begint hij bij 1 omdat plaatje 0 het gegeven plaatje is; hier is rij `i` de gegeven rij. Loop over alle rijen, `for j in range(len(E)):`, en sla alleen rij `i` over."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the inside of the loop: skip `j` when it is `i`, work out the cosine similarity of rows `i` and `j`, and keep `j` when its cosine similarity is the largest so far.",
                      "nl": "Vervang de `...` door wat er in de loop gebeurt: sla `j` over als die `i` is, reken de cosine similarity van rij `i` en rij `j` uit, en houd `j` bij als zijn cosine similarity de grootste tot nu toe is."},
            "arguments": {"en": "The check calls `{call}`, with a table and a row number, and the `def` line of your function has another number of names between its parentheses. Keep `(E, i)`.",
                          "nl": "De check roept `{call}` aan, met een tabel en een rijnummer, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(E, i)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. For each other row `j`, do you work out the cosine similarity with row `i`, and keep `best` and `best_j` when it is larger than `best`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken je voor elke andere rij `j` de cosine similarity met rij `i` uit, en houd je `best` en `best_j` bij als die groter is dan `best`?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your loop compare row `i` with every other row, but not with itself? And by the cosine similarity, so that the length of a row does not count?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijkt je loop rij `i` met elke andere rij, maar niet met zichzelf? En op de cosine similarity, zodat de lengte van een rij niet meetelt?"},
            "correct": {"en": "Your function finds the row most like row `i` in all {n} cases: by direction, not by length.",
                        "nl": "Je functie vindt in alle {n} gevallen de rij die het meest op rij `i` lijkt: op richting, niet op lengte."},
        },
    ),
}

CASES = [
    ("most-similar", _M_START, "empty", None),
    ("most-similar", _m(), "correct", None),
    ("most-similar", _M_HEAD + "    best_j = -1\n    best = -2\n    for j in range(len(E)):\n        if j == i:\n            continue\n"
                     "        a = E[i] / np.linalg.norm(E[i])\n        b = E[j] / np.linalg.norm(E[j])\n        if np.dot(a, b) > best:\n"
                     "            best = np.dot(a, b)\n            best_j = j\n    return best_j\n", "correct", None),
    ("most-similar", _m(test="True"), "wrong", "itself"),
    ("most-similar", _DISTANCE, "wrong", "distance"),
    ("most-similar", _m(cos="np.dot(E[i], E[j])"), "almost", "dot-only"),
    ("most-similar", _m(better="cos < best"), "wrong", "never-chosen"),
    ("most-similar", _m().replace("                best_j = j\n", ""), "wrong", "never-chosen"),   # forgot best_j = j
    ("most-similar", _m().replace("return best_j", "return best"), "wrong", "best-value"),
    ("most-similar", _m().replace("best_j = j", "best_j = best"), "wrong", "best-value"),
    ("most-similar", _m().replace("range(len(E))", "range(1, len(E))"), "almost", "from-one"),
    ("most-similar", _m(test="True").replace("range(len(E))", "range(1, len(E))"), "almost", None),   # step 3 copied, without the skip
    ("most-similar", _m(better="cos >= best"), "correct", None),
    ("most-similar", _M_HEAD + "    coss = []\n    for j in range(len(E)):\n        coss.append(" + _COS + ")\n    coss[i] = -2\n    return np.argmax(coss)\n",
     "correct", None),
    ("most-similar", "def most_similar(E):\n    return 0\n", "error", "arguments"),
]
