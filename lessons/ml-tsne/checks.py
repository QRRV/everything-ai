# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-tsne: the 1,000 test digits of QMNIST on a flat map, by PCA and by t-SNE. The maps of
t-SNE for four perplexities (made once by data/tools/make_tsne_maps.py), how often a digit's nearest neighbor on a map
is the same digit, the neighbor probabilities of one digit for every width, and the functions neighbor_probs and
perplexity."""
import contextlib
import functools
import io
import re
import warnings

from everything_ai import chart, check as checking, paths, yours
from everything_ai.check import FunctionBox, Mistake

DATA = "data/qmnist-small.npz"        # the first 1,000 test images of QMNIST: see data/README.md
MAPS = "data/qmnist-tsne-maps.npz"    # their t-SNE maps, made by data/tools/make_tsne_maps.py
PERPLEXITIES = [2, 5, 30, 100]        # the stops of the first slider
WIDTHS = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5]   # the stops of the second slider
ONE = 0                               # the digit whose neighbors the second slider shows: the first test digit
TOP = 30                              # how many of its neighbors the second picture shows


# --- the course's own functions, as the lesson writes them -----------------------------------------------------------

def neighbor_probs(d, width):
    import numpy as np
    z = -d ** 2 / (2 * width ** 2)
    e = np.exp(z)
    return e / np.sum(e)


def perplexity(p):
    import numpy as np
    h = -np.sum(p * np.log(p))
    return np.exp(h)


def same_neighbor(Z, y) -> float:
    """The share of the examples whose nearest other example, in Z, has the same label."""
    import numpy as np
    same = 0
    for i in range(len(Z)):
        j = np.argsort(np.linalg.norm(Z - Z[i], axis=1))[1]
        same = same + int(y[j] == y[i])
    return same / len(Z)


# --- the digits and their maps ---------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def digits() -> dict:
    """The 1,000 test digits: X, 784 numbers from 0 to 1 each, and y, the digits. Read-only."""
    import numpy as np
    with np.load(paths.ROOT / DATA) as d:
        out = {"X": d["test_images"].astype(float) / 255, "y": d["test_labels"].astype(int)}
    for v in out.values():
        v.flags.writeable = False
    return out


@functools.lru_cache(maxsize=1)
def pca_map() -> dict:
    """The digits on the plane of their first two principal components, and the share of the spread it keeps."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(digits()["X"])
    Z = pca.transform(digits()["X"])
    Z.flags.writeable = False
    return {"Z": Z, "kept": float(pca.explained_variance_ratio_.sum())}


@functools.lru_cache(maxsize=None)
def tsne_map(p: int, seed: int = 0):
    """The t-SNE map of the digits with this perplexity, as data/tools/make_tsne_maps.py made it; seed 1 only for
    perplexity 30."""
    import numpy as np
    with np.load(paths.ROOT / MAPS) as d:
        Z = d[f"p{p}" + (f"-seed{seed}" if seed else "")].astype(float)
    Z.flags.writeable = False
    return Z


@functools.lru_cache(maxsize=None)
def same(p: int | None) -> float:
    """same_neighbor of the t-SNE map with perplexity p; of the PCA map for None; of the 784 numbers for 0."""
    Z = pca_map()["Z"] if p is None else digits()["X"] if p == 0 else tsne_map(p)
    return same_neighbor(Z, digits()["y"])


@functools.lru_cache(maxsize=1)
def distances_of_one():
    """The distance from digit ONE to each of the 999 other digits, in the 784 numbers, with their digits."""
    import numpy as np
    X, y = digits()["X"], digits()["y"]
    d = np.linalg.norm(X - X[ONE], axis=1)
    keep = np.arange(len(X)) != ONE
    out = (d[keep], y[keep])
    for v in out:
        v.flags.writeable = False
    return out


def width_facts(width: float) -> dict:
    """For digit ONE and this width: the perplexity, the largest probability, how many neighbors get more than 1%,
    and how much of the probability goes to the same digit."""
    import numpy as np
    d, y = distances_of_one()
    p = neighbor_probs(d, width)
    return {"perplexity": float(perplexity(p)), "top": float(p.max()), "over": int(np.sum(p > 0.01)),
            "same": float(p[y == digits()["y"][ONE]].sum())}


def width_for(target: float) -> float:
    """The width that gives digit ONE this perplexity, found by halving, as t-SNE does for every point."""
    d, _ = distances_of_one()
    lo, hi = 1e-3, 100.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if perplexity(neighbor_probs(d, mid)) < target else (lo, mid)
    return (lo + hi) / 2


def number(x: float, lang: str, digits_: int = 1) -> str:
    text = f"{x:,.{digits_}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


# --- the pictures ----------------------------------------------------------------------------------------------------

def _map_panel(Z, title: str, labels_: dict):
    """A map of the 1,000 digits: every digit a small dot, and every digit's name at the middle (the median) of its
    dots. The axes have no numbers: on a map of t-SNE they mean nothing."""
    import numpy as np
    y = digits()["y"]
    lo, hi = Z.min(axis=0), Z.max(axis=0)
    pad = (hi - lo) * 0.05
    p = chart.Panel((float(lo[0] - pad[0]), float(hi[0] + pad[0])), (float(lo[1] - pad[1]), float(hi[1] + pad[1])),
                    title=title, xlabel=labels_["x"], ylabel=labels_["y"], xticks=[], yticks=[])
    p.points(Z[:, 0], Z[:, 1], "ink-3", r=2)
    at = np.array([np.median(Z[y == k], axis=0) for k in range(10)]) / (hi - lo)   # the middles, on a square of 1 by 1
    for _ in range(20):            # two names closer than 0.06 of the map move apart along the line between them
        for i in range(10):
            for j in range(i + 1, 10):
                gap = at[j] - at[i]
                n = float(np.linalg.norm(gap))
                if n < 0.06:
                    push = (gap / n if n > 0 else np.array([1.0, 0.0])) * (0.06 - n) / 2
                    at[i] -= push
                    at[j] += push
    for k, (mx, my) in enumerate(at * (hi - lo)):
        p.text(float(mx), float(my), str(k), "ink", halo=True)
    return p


def picture_map(Z, labels_: dict) -> str:
    """One map; for the slider once for every stop."""
    return chart.figure([_map_panel(Z, labels_["title"], labels_)], alt=labels_["alt"], width=340, height=300)



def picture_width(width: float, labels_: dict) -> str:
    """The TOP largest neighbor probabilities of digit ONE for this width, large to small, on an axis from 0 to 1: a
    bar per neighbor, filled when the neighbor is the same digit, open otherwise."""
    import numpy as np
    d, y = distances_of_one()
    p = neighbor_probs(d, width)
    order = np.argsort(-p)[:TOP]
    panel = chart.Panel((-0.8, TOP - 0.2), (0, 1.0), title=labels_["title"], xlabel=labels_["x"], ylabel=labels_["y"],
                        xticks=[0, 9, 19, 29], xtick_labels=["1", "10", "20", "30"], yticks=[0, 0.25, 0.5, 0.75, 1])
    mine = digits()["y"][ONE]
    for i, j in enumerate(order):   # two groups: filled and open (style guide 1.7), no colour
        panel.rect(i - 0.32, 0, i + 0.32, float(p[j]), role="data" if y[j] == mine else "ink-3", solid=bool(y[j] == mine))
    return chart.figure([panel], alt=labels_["alt"], width=340, height=250)


# The lesson is done when these are correct. "islands" is the choice question in the lesson.
REQUIRED = ["neighbor-probs", "perplexity", "islands"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


class _Val:
    """A test value that the check's messages write by its name, such as `d`; the function gets an array."""

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


# --- exercise 1: neighbor_probs --------------------------------------------------------------------------------------

_D = _Val("d", [1.0, 2.0, 3.0])
_FAR = _Val("far", [5.0, 6.0, 9.0, 12.0])
_EQUAL = _Val("equal", [2.0, 2.0, 2.0, 2.0])
_ONE_D = _Val("one", [4.0])

_N_HEAD = "import numpy as np\n\n\ndef neighbor_probs(d, width):\n"


def _n(z: str = "-d ** 2 / (2 * width ** 2)", back: str = "e / np.sum(e)") -> str:
    return _N_HEAD + f"    z = {z}\n    e = np.exp(z)\n    return {back}\n"


_N_START = _n("...", "...")
_N = _n()


def _probs_picture(seen) -> str:
    """The visible case: a bar per neighbor for the learner's probabilities, with the course's as a dashed line."""
    import numpy as np
    ex = seen.exercise
    mine = seen.mine
    if mine is None:
        return ""
    fn = mine.names.get(ex.function)
    if not callable(fn):
        return ""
    d = np.array(_D.value)
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = np.asarray(checking.with_time_limit(lambda: fn(d.copy(), 1.0)), dtype=float)
        if got.shape != d.shape or not np.isfinite(got).all():
            return ""
    except Exception:
        return ""
    want = neighbor_probs(d, 1.0)
    lang = seen.lang
    hi = float(max(got.max(), want.max(), 1.0))
    p = chart.Panel((-0.6, 2.6), (min(0.0, float(got.min())), hi * 1.08), xlabel={"en": "distance", "nl": "afstand"}[lang],
                    ylabel={"en": "probability", "nl": "kans"}[lang], xticks=[0, 1, 2], xtick_labels=["1", "2", "3"])
    for i in range(3):
        if got[i] > 0:
            p.rect(i - 0.34, 0, i + 0.34, float(got[i]), role="data")
        p.line([i - 0.42, i + 0.42], [float(want[i])] * 2, role="ink", width=2, dashed=True)
    total = float(got.sum())
    words = {"en": f"Your probabilities for d = [1, 2, 3] and width 1, as bars; the right ones dashed. Together yours make {number(total, 'en', 3)}.",
             "nl": f"Jouw kansen voor d = [1, 2, 3] en breedte 1, als staven; de goede gestreept. Samen maken de jouwe {number(total, 'nl', 3)}."}[lang]
    title = {"en": "Your neighbor_probs(d, 1.0)", "nl": "Jouw neighbor_probs(d, 1.0)"}[lang]
    key = ('<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.15rem">'
           + words + "</div>")
    return yours.figure(title, chart.figure([p], alt=words, width=330, height=220) + key, alt=words)


EXERCISES = {
    "neighbor-probs": FunctionBox(
        name="neighbor-probs",
        picture=_probs_picture,
        function="neighbor_probs",
        start=_N_START,
        solution=_N,
        show=_shown,
        visible=[(_D, 1.0)],
        hidden=[(_D, 2.0), (_FAR, 3.0), (_EQUAL, 1.5), (_ONE_D, 1.0)],
        first=[_DOTS],
        mistakes=[
            Mistake("no-minus", wrong=_n(z="d ** 2 / (2 * width ** 2)"),
                    en="`{call}` gives {got}: the farthest neighbor gets the largest probability. Put a minus in front, so a larger distance gives a smaller number.",
                    nl="`{call}` geeft {got}: de verste buur krijgt de grootste kans. Zet er een min voor, zodat een grotere afstand een kleiner getal geeft."),
            Mistake("no-square", wrong=_n(z="-d / (2 * width ** 2)"),
                    en="`{call}` gives {got}, and it should give {want}. The distance goes in squared: `d ** 2`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. De afstand gaat er gekwadrateerd in: `d ** 2`."),
            Mistake("no-sum", wrong=_n(back="e"),
                    en="`{call}` gives {got}: these do not add up to 1. Divide by their sum, as softmax does: `e / np.sum(e)`.",
                    nl="`{call}` geeft {got}: die tellen niet op tot 1. Deel door hun som, zoals softmax doet: `e / np.sum(e)`."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the two `...`: the number for every neighbor, and the probabilities made from them.",
                      "nl": "Vervang de twee `...`: het getal voor elke buur, en de kansen die je ervan maakt."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(d, width)`.",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(d, width)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is `z` minus the squared distance divided by `2 * width ** 2`, and do you divide `e` by its sum?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Is `z` min de gekwadrateerde afstand gedeeld door `2 * width ** 2`, en deel je `e` door zijn som?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use `width`, and work for any number of neighbors?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `width`, en werkt hij voor elk aantal buren?"},
            "correct": {"en": "Your function is right in all {n} cases. With width 1, the neighbor at distance 1 gets 0.806 and the one at 3 only 0.015; with width 2 they get 0.487 and 0.179: a larger width spreads the probability over more neighbors.",
                        "nl": "Je functie klopt in alle {n} gevallen. Met breedte 1 krijgt de buur op afstand 1 0,806 en die op 3 maar 0,015; met breedte 2 krijgen ze 0,487 en 0,179: een grotere breedte verdeelt de kans over meer buren."},
        },
    ),
}


# --- exercise 2: perplexity ------------------------------------------------------------------------------------------

_FOUR = _Val("four", [0.25, 0.25, 0.25, 0.25])
_P = _Val("p", [0.7, 0.2, 0.1])
_TWO = _Val("two", [0.5, 0.5])
_SURE = _Val("sure", [0.97, 0.01, 0.01, 0.01])
_THIRTY = _Val("thirty", [1 / 30] * 30)

_P_HEAD = "import numpy as np\n\n\ndef perplexity(p):\n"


def _p(h: str = "-np.sum(p * np.log(p))", back: str = "np.exp(h)") -> str:
    return _P_HEAD + f"    h = {h}\n    return {back}\n"


_P_START = _p("...", "...")
_P_SOL = _p()

EXERCISES["perplexity"] = FunctionBox(
    name="perplexity",
    picture=yours.steps(call=0),
    function="perplexity",
    start=_P_START,
    solution=_P_SOL,
    show=_shown,
    visible=[(_FOUR,)],
    hidden=[(_P,), (_TWO,), (_SURE,), (_THIRTY,)],
    first=[_DOTS],
    mistakes=[
        Mistake("no-exp", wrong=_p(back="h"),
                en="`{call}` gives {got}: that is `h` itself. The perplexity is `np.exp(h)`, which turns it back into a number of neighbors.",
                nl="`{call}` geeft {got}: dat is `h` zelf. De perplexity is `np.exp(h)`, dat het weer een aantal buren maakt."),
        Mistake("no-minus", wrong=_p(h="np.sum(p * np.log(p))"),
                en="`{call}` gives {got}: one divided by the right number. `np.log(p)` is negative for every probability below 1, so `h` needs a minus in front.",
                nl="`{call}` geeft {got}: één gedeeld door het goede getal. `np.log(p)` is negatief voor elke kans onder 1, dus `h` heeft een min ervoor nodig."),
        Mistake("no-p", wrong=_p(h="-np.sum(np.log(p))"),
                en="`{call}` gives {got}, and it should give {want}. Weigh every `np.log(p)` by its own `p`, as cross-entropy does: `-np.sum(p * np.log(p))`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Weeg elke `np.log(p)` met zijn eigen `p`, zoals cross-entropy doet: `-np.sum(p * np.log(p))`."),
    ],
    texts=_SAME | {
        "empty": {"en": "Replace the two `...`: `h`, minus the sum of `p` times its log, and the perplexity made from it.",
                  "nl": "Vervang de twee `...`: `h`, min de som van `p` keer zijn log, en de perplexity die je ervan maakt."},
        "arguments": {"en": "The check calls `{call}`, with one value, and the `def` line of your function has a different number of names between its parentheses. Keep `(p)`.",
                      "nl": "De check roept `{call}` aan, met één waarde, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(p)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is `h` minus the sum of `p * np.log(p)`, and do you give back `np.exp(h)`?",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Is `h` min de som van `p * np.log(p)`, en geef je `np.exp(h)` terug?"},
        "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for probabilities that are not all equal?",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor kansen die niet allemaal gelijk zijn?"},
        "correct": {"en": "Your function is right in all {n} cases. Four equal probabilities give 4, thirty give 30; `p`, with 0.7, 0.2 and 0.1, gives 2.2: about two neighbors that count. For the 7 of the slider, `perplexity(neighbor_probs(d, 2.0))` is 28.6: width 2.0 is about what perplexity 30 asks.",
                    "nl": "Je functie klopt in alle {n} gevallen. Vier gelijke kansen geven 4, dertig geven 30; `p`, met 0,7, 0,2 en 0,1, geeft 2,2: ongeveer twee buren die tellen. Voor de 7 van de slider is `perplexity(neighbor_probs(d, 2.0))` 28,6: breedte 2,0 is ongeveer wat perplexity 30 vraagt."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("neighbor-probs", _N_START, "empty", None),
    ("neighbor-probs", _n(back="..."), "wrong", "dots-left"),
    ("neighbor-probs", _N, "correct", None),
    ("neighbor-probs", _n(back="np.exp(z) / np.exp(z).sum()"), "correct", None),
    ("neighbor-probs", _n(z="-(d / width) ** 2 / 2"), "correct", None),
    ("neighbor-probs", _n(z="d ** 2 / (2 * width ** 2)"), "wrong", "no-minus"),
    ("neighbor-probs", _n(z="-d / (2 * width ** 2)"), "wrong", "no-square"),
    ("neighbor-probs", _n(back="e"), "wrong", "no-sum"),
    ("neighbor-probs", _n(z="-d ** 2 / 2"), "almost", None),
    ("perplexity", _P_START, "empty", None),
    ("perplexity", _p(back="..."), "wrong", "dots-left"),
    ("perplexity", _P_SOL, "correct", None),
    ("perplexity", _p(h="-np.sum(p * np.log2(p))", back="2 ** h"), "correct", None),
    ("perplexity", _p(back="h"), "wrong", "no-exp"),
    ("perplexity", _p(h="np.sum(p * np.log(p))"), "wrong", "no-minus"),
    ("perplexity", _p(h="-np.sum(np.log(p))"), "wrong", "no-p"),
    ("perplexity", _p(back="len(p)"), "almost", None),
]
