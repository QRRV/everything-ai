# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-no-free-lunch: every model assumes something about the data, and averaged over all
problems none wins. Five made-up datasets with the four models of the module (k-NN, the nearest centroid, the widest
street and a decision tree), scored with 5-fold cross-validation and drawn with their boundaries; and the tiny world
of three yes/no features, with the functions all_worlds and mean_right."""
import functools
import html
import re

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake

SEED, N = 7, 300                  # one draw of the five datasets, 300 examples each
DATASETS = ["clumps", "line", "ring", "blocks", "coins"]
NAMES = {
    "clumps": {"en": "two round clumps", "nl": "twee ronde klonten"},
    "line": {"en": "a straight line", "nl": "een rechte lijn"},
    "ring": {"en": "a ring", "nl": "een ring"},
    "blocks": {"en": "blocks", "nl": "blokken"},
    "coins": {"en": "coin flips", "nl": "muntworpen"},
}
MODELS = ["knn", "nc", "street", "tree"]
MODEL_NAMES = {
    "knn": {"en": "k-NN", "nl": "k-NN"},
    "nc": {"en": "nearest centroid", "nl": "nearest centroid"},
    "street": {"en": "widest street", "nl": "breedste straat"},
    "tree": {"en": "tree", "nl": "boom"},
}
RANGE = (-3.6, 3.6)


# --- the course's two functions, as the lesson writes them ----------------------------------------------------------

def all_worlds(n):
    worlds = [[]]
    for i in range(n):
        longer = []
        for world in worlds:
            longer.append(world + [0])
            longer.append(world + [1])
        worlds = longer
    return worlds


def mean_right(predictions, worlds):
    total = 0
    for world in worlds:
        right = 0
        for i in range(len(predictions)):
            if predictions[i] == world[i]:
                right = right + 1
        total = total + right / len(predictions)
    return total / len(worlds)


# --- the five datasets and the four models ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def datasets() -> dict:
    """The five made-up datasets, each (X, y) with two features and 300 examples of class 0 or 1, from one draw."""
    import numpy as np
    r = np.random.default_rng(SEED)
    n = N
    out = {}
    # two round clumps that overlap: the classes differ in where their clump lies
    a = r.normal([-0.6, -0.6], 1.0, (n // 2, 2))
    b = r.normal([0.6, 0.6], 1.0, (n // 2, 2))
    out["clumps"] = (np.vstack([a, b]), np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int))
    # a straight diagonal line between the classes, each a long band along it, shifted along it
    t, s, y = r.uniform(-3, 3, n), r.uniform(0.1, 1.2, n), (r.random(n) < 0.5).astype(int)
    u, v = t + np.where(y == 1, 1.2, -1.2), np.where(y == 1, s, -s)
    out["line"] = (np.c_[(u - v) / np.sqrt(2), (u + v) / np.sqrt(2)], y)
    # a round clump inside a ring
    angle = r.uniform(0, 2 * np.pi, n)
    radius = np.where(np.arange(n) < n // 2, r.uniform(0, 1.0, n), r.uniform(1.4, 2.4, n))
    out["ring"] = (np.c_[radius * np.cos(angle), radius * np.sin(angle)], (np.arange(n) >= n // 2).astype(int))
    # blocks: the class changes at a threshold on the first feature and at one on the second
    X = r.uniform(-2, 2, (n, 2))
    out["blocks"] = (X, ((X[:, 0] > 0.5) ^ (X[:, 1] > -0.5)).astype(int))
    # coin flips: the class has nothing to do with the features
    X = r.uniform(-2, 2, (n, 2))
    out["coins"] = (X, (r.random(n) < 0.5).astype(int))
    return out


def _model(key: str):
    from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    return {"knn": lambda: KNeighborsClassifier(5), "nc": lambda: NearestCentroid(),
            "street": lambda: SVC(kernel="linear", C=1), "tree": lambda: DecisionTreeClassifier(max_depth=4, random_state=0)}[key]()


@functools.lru_cache(maxsize=1)
def scores() -> dict:
    """For every dataset, the accuracy of every model with 5-fold cross-validation, rounded to 3 decimals."""
    from sklearn.model_selection import KFold, cross_val_score
    out = {}
    for d in DATASETS:
        X, y = datasets()[d]
        cv = KFold(5, shuffle=True, random_state=0)
        out[d] = {m: round(float(cross_val_score(_model(m), X, y, cv=cv).mean()), 3) for m in MODELS}
    return out


def best(d: str) -> str | None:
    """The model with the highest accuracy on dataset d; None on the coin flips, where the class has nothing to do with
    the features and every model is close to always saying the larger class (163 of 300)."""
    if d == "coins":
        return None
    s = scores()[d]
    return max(MODELS, key=lambda m: s[m])


def verdict(v: int, lang: str) -> str:
    """What the slider says at dataset v (1 to 5): its name, and the best model with its accuracy."""
    d = DATASETS[v - 1]
    top = best(d)
    if top is None:
        y = datasets()[d][1]
        share = f'<span class="ea-num">{number(max(y.mean(), 1 - y.mean()), lang)}</span>'
        return {"en": f"Dataset {v}: {NAMES[d]['en']}. Best: none; no model does better than always saying the larger class, {share}.",
                "nl": f"Dataset {v}: {NAMES[d]['nl']}. Het beste: geen; geen model doet het beter dan altijd de grootste klasse zeggen, {share}."}[lang]
    said = {"en": "Dataset {v}: {name}. Best: {model}, with {score}.", "nl": "Dataset {v}: {name}. Het beste: {model}, met {score}."}
    return said[lang].format(v=v, name=NAMES[d][lang], model=MODEL_NAMES[top][lang],
                             score=f'<span class="ea-num">{number(scores()[d][top], lang)}</span>')


@functools.lru_cache(maxsize=None)
def _fit(d: str, m: str):
    X, y = datasets()[d]
    return _model(m).fit(X, y)


_G = 90   # grid points along each axis, for the boundaries of k-NN and the tree


@functools.lru_cache(maxsize=None)
def _grid_edges(d: str, m: str) -> list:
    """Where the prediction of the model changes between two cells of a grid: the boundary, as (x0, y0, x1, y1)."""
    import numpy as np
    xs = np.linspace(*RANGE, _G)
    gx, gy = np.meshgrid(xs, xs)
    r = _fit(d, m).predict(np.c_[gx.ravel(), gy.ravel()]).reshape(_G, _G)
    step = xs[1] - xs[0]
    lines = []
    for i in range(_G - 1):
        start = None
        for j in range(_G + 1):
            edge = j < _G and r[j, i] != r[j, i + 1]
            if edge and start is None:
                start = j
            if not edge and start is not None:
                x = (xs[i] + xs[i + 1]) / 2
                lines.append((x, xs[start] - step / 2, x, xs[j - 1] + step / 2))
                start = None
    for j in range(_G - 1):
        start = None
        for i in range(_G + 1):
            edge = i < _G and r[j, i] != r[j + 1, i]
            if edge and start is None:
                start = i
            if not edge and start is not None:
                y = (xs[j] + xs[j + 1]) / 2
                lines.append((xs[start] - step / 2, y, xs[i - 1] + step / 2, y))
                start = None
    return [tuple(max(RANGE[0], min(RANGE[1], float(v))) for v in seg) for seg in lines]


def _line_edges(d: str, m: str) -> list:
    """The straight boundary of the nearest centroid or the widest street, w·x + b = 0, inside the picture."""
    import numpy as np
    f = _fit(d, m)
    if m == "nc":
        m0, m1 = f.centroids_
        w, b = m1 - m0, (m0 @ m0 - m1 @ m1) / 2
    else:
        w, b = f.coef_[0], f.intercept_[0]
    pts = []
    if abs(w[1]) > 1e-12:
        pts += [(x, -(b + w[0] * x) / w[1]) for x in RANGE]
    if abs(w[0]) > 1e-12:
        pts += [(-(b + w[1] * y) / w[0], y) for y in RANGE]
    inside = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in pts
                     if RANGE[0] - 1e-9 <= x <= RANGE[1] + 1e-9 and RANGE[0] - 1e-9 <= y <= RANGE[1] + 1e-9})
    if len(inside) < 2 or inside[0] == inside[-1]:
        return []
    (x0, y0), (x1, y1) = inside[0], inside[-1]
    return [(x0, y0, x1, y1)]


def edges(d: str, m: str) -> list:
    return _line_edges(d, m) if m in ("nc", "street") else _grid_edges(d, m)


def number(x: float, lang: str) -> str:
    text = f"{x:.3f}"
    return text.replace(".", ",") if lang == "nl" else text


def models_picture(d: str, labels: dict, lang: str) -> str:
    """The dataset four times, class 0 filled and class 1 open, each with the boundary one model draws after training
    on all 300 examples, and above it the model's accuracy with cross-validation; the best is marked."""
    import numpy as np
    X, y = datasets()[d]
    keep = (np.abs(X) <= RANGE[1]).all(axis=1)
    top = best(d)
    panels = []
    for m in MODELS:
        title = f"{MODEL_NAMES[m][lang]}: {number(scores()[d][m], lang)}" + (" ★" if m == top else "")
        p = chart.Panel(RANGE, RANGE, title=title, xlabel=labels["x"], ylabel=labels["y"], xticks=[-3, 0, 3], yticks=[-3, 0, 3])
        for k in (0, 1):
            rows = X[keep & (y == k)]
            p.points(rows[:, 0], rows[:, 1], "ink-2", r=2.6, hollow=k == 1)
        for x0, y0, x1, y1 in edges(d, m):
            p.line([x0, x1], [y0, y1], "pred", 2.2)
        panels.append(p)
    return chart.figure(panels, alt=labels["alt"], width=250, height=250)


# --- how the learner's answers are written ---------------------------------------------------------------------------

class _Val:
    """A test value that the check's messages write by its name, such as `worlds`; the function gets a plain copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value)


def _shown(v) -> str:
    """What a function gave back, as the lesson writes it: a list as it is, a number rounded to 3 decimals."""
    if isinstance(v, float):
        return str(round(v, 3))
    if isinstance(v, list) and len(v) > 8:
        return "[" + ", ".join(_shown(x) for x in v[:4]) + f", … ({len(v)})]"
    return repr(v)


def _as_worlds(got):
    """The learner's worlds as a sorted list of tuples, or None when it is not a list of lists of numbers."""
    if not isinstance(got, list):
        return None
    out = []
    for w in got:
        if not isinstance(w, list) or not all(isinstance(v, int) and not isinstance(v, bool) for v in w):
            return None
        out.append(tuple(w))
    return sorted(out)


def _same_worlds(got, want) -> bool:
    mine = _as_worlds(got)
    return mine is not None and mine == _as_worlds(want)


_WORLD_WORDS = {
    "title": {"en": "The worlds of {call}", "nl": "De werelden van {call}"},
    "all": {"en": "All {n} worlds, each once.", "nl": "Alle {n} werelden, elk één keer."},
    "same": {"en": "{got} worlds, but not every way once: every way the {k} examples can be labeled must come exactly once. A purple row is a world that comes twice, or has the wrong length.",
             "nl": "{got} werelden, maar niet elke manier één keer: elke manier waarop de {k} voorbeelden een label kunnen krijgen, moet precies één keer voorkomen. Een paarse rij is een wereld die twee keer voorkomt, of de verkeerde lengte heeft."},
    "count": {"en": "Your function gives {got} worlds, and there are {want}: every way the {k} examples can be labeled, each once. A purple row is a world that comes twice, or has the wrong length.",
              "nl": "Je functie geeft {got} werelden, en er zijn er {want}: elke manier waarop de {k} voorbeelden een label kunnen krijgen, elk één keer. Een paarse rij is een wereld die twee keer voorkomt, of de verkeerde lengte heeft."},
    "missing": {"en": "Missing: {missing}.", "nl": "Ontbreekt: {missing}."},
    "more": {"en": " and {n} more", "nl": " en nog {n}"},
}


def _worlds_picture(seen) -> str:
    """The worlds the learner's all_worlds gives for the example, a row each, with a purple row for a world that
    comes twice or has the wrong length, and the worlds that are missing."""
    out = yours._outcome(seen, 0)
    if out is None:
        return ""
    got, want, start, what = out
    rows = _as_worlds(got) if isinstance(got, list) else None
    if rows is None or len(rows) > 64:
        return ""
    lang = seen.lang
    words = {k: v[lang] for k, v in _WORLD_WORDS.items()}
    k = len(want[0]) if want else 0
    seen_rows, cells = set(), []
    for w in [tuple(x) for x in got]:
        bad = w in seen_rows or len(w) != k
        seen_rows.add(w)
        colour = "var(--ea-loss)" if bad else "var(--ea-ink)"
        cells.append(f'<div style="font-family:var(--ea-font-code);font-size:0.86rem;color:{colour}">'
                     f'{html.escape(str(list(w)))}</div>')
    missing = [list(w) for w in sorted(tuple(x) for x in want) if w not in seen_rows]
    right = len(got) == len(want) and not missing and len(seen_rows) == len(got)
    if right:
        text = words["all"].format(n=len(want))
    else:
        text = words["same" if len(got) == len(want) else "count"].format(got=len(got), want=len(want), k=k)
    if missing and not right:
        text += " " + words["missing"].format(missing=", ".join(str(m) for m in missing[:4]) + (words["more"].format(n=len(missing) - 4) if len(missing) > 4 else ""))
    body = ('<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(6.5rem,1fr));gap:0.2rem 0.8rem;'
            'max-width:30rem">' + "".join(cells) + "</div>"
            '<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.35rem">'
            + html.escape(text) + "</div>")
    title = words["title"].format(call=f"{seen.exercise.function}({start[0][1]!r})" if start else seen.exercise.function)
    return yours.figure(title, body, alt=text)


# The lesson is done when these are correct. "one-dataset" is the choice question in the lesson.
REQUIRED = ["all-worlds", "mean-right", "one-dataset"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# --- exercise 1: all_worlds -------------------------------------------------------------------------------------------

_A_HEAD = "def all_worlds(n):\n    worlds = [[]]\n    for i in range(n):\n        longer = []\n        for world in worlds:\n"
_A_TAIL = "        worlds = longer\n    return worlds\n"


def _a(body: str = "longer.append(world + [0])\nlonger.append(world + [1])") -> str:
    return _A_HEAD + "".join(f"            {line}\n" for line in body.split("\n")) + _A_TAIL


_A_START = _a("...")
_A = _a()

# --- exercise 2: mean_right -------------------------------------------------------------------------------------------


def _m(inner: str = "if predictions[i] == world[i]:\n    right = right + 1", back: str = "total / len(worlds)") -> str:
    return ("def mean_right(predictions, worlds):\n    total = 0\n    for world in worlds:\n        right = 0\n"
            "        for i in range(len(predictions)):\n"
            + "".join(f"            {line}\n" for line in inner.split("\n"))
            + f"        total = total + right / len(predictions)\n    return {back}\n")


_M_START = _m("...", "...")
_M = _m()

_W3 = _Val("worlds", all_worlds(3))
_W4 = _Val("worlds4", all_worlds(4))
_W2 = _Val("worlds2", all_worlds(2))
_FIRST = _Val("first_is_0", [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]])     # the worlds in which the first unseen example is 0
_PAIR = _Val("two_worlds", [[0, 0], [0, 1]])

EXERCISES = {
    "all-worlds": FunctionBox(
        name="all-worlds",
        picture=_worlds_picture,
        function="all_worlds",
        start=_A_START,
        solution=_A,
        show=_shown,
        compare=_same_worlds,
        visible=[(3,)],
        hidden=[(1,), (2,), (4,), (0,)],
        first=[_DOTS],
        mistakes=[
            Mistake("one-way", wrong=_a("longer.append(world + [0])"),
                    en="`{call}` gives {got}: only one world. Every world of `worlds` should give two longer ones, one ending in 0 and one in 1, so the number of worlds doubles with every example.",
                    nl="`{call}` geeft {got}: maar één wereld. Elke wereld in `worlds` hoort twee langere te geven, een die op 0 eindigt en een die op 1 eindigt, zodat het aantal werelden met elk voorbeeld verdubbelt."),
            Mistake("same-list", wrong=_a("world.append(0)\nlonger.append(world)\nlonger.append(world)"),
                    en="`{call}` gives {got}. `world.append(0)` changes the world itself, and both places in `longer` hold that same list. `world + [0]` makes a new list and leaves `world` as it was.",
                    nl="`{call}` geeft {got}. `world.append(0)` verandert de wereld zelf, en beide plaatsen in `longer` bevatten diezelfde lijst. `world + [0]` maakt een nieuwe lijst en laat `world` zoals hij was."),
            Mistake("numbers", matches=lambda got, text: isinstance(got, list) and len(got) > 0 and all(isinstance(v, int) for v in got),
                    en="`{call}` gives {got}: numbers, not worlds. Each new world is the old one with one more label at the end: `world + [0]` and `world + [1]`.",
                    nl="`{call}` geeft {got}: getallen, geen werelden. Elke nieuwe wereld is de oude met één label meer aan het eind: `world + [0]` en `world + [1]`."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the `...` with two lines: put in `longer` the world with a 0 added at the end, and the world with a 1 added.",
                      "nl": "Vervang de `...` door twee regels: zet in `longer` de wereld met een 0 erachter, en de wereld met een 1 erachter."},
            "arguments": {"en": "The check calls `{call}`, with one value, and the `def` line of your function has a different number of names between its parentheses. Keep `(n)`.",
                          "nl": "De check roept `{call}` aan, met één waarde, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(n)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}, in any order. Does every world give two new ones, one ending in 0 and one in 1?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn, in elke volgorde. Geeft elke wereld twee nieuwe, een die op 0 eindigt en een die op 1 eindigt?"},
            "almost": {"en": "Right for 3 examples. But `{call}` gives {got}, and it should give {want}. Does your function work for any number of examples, also 0?",
                       "nl": "Goed voor 3 voorbeelden. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk aantal voorbeelden, ook 0?"},
            "correct": {"en": "Your function is right in all {n} cases: 2 worlds for 1 example, 8 for 3, 16 for 4, and one empty world for 0. Every example more doubles them: for 20 unseen examples, more than a million worlds, all fitting the same training set.",
                        "nl": "Je functie klopt in alle {n} gevallen: 2 werelden voor 1 voorbeeld, 8 voor 3, 16 voor 4, en één lege wereld voor 0. Elk voorbeeld meer verdubbelt ze: voor 20 onbekende voorbeelden ruim een miljoen werelden, die allemaal bij dezelfde training set passen."},
        },
    ),
    "mean-right": FunctionBox(
        name="mean-right",
        picture=yours.steps(call=0),
        function="mean_right",
        start=_M_START,
        solution=_M,
        show=_shown,
        visible=[([0, 1, 1], _W3)],
        hidden=[([1, 1, 0], _W3), ([0, 1, 1], _FIRST), ([1, 0, 1, 0], _W4), ([0, 0], _PAIR), ([1, 1], _W2)],
        first=[_DOTS],
        mistakes=[
            Mistake("sum", wrong=_m(back="total"),
                    en="`{call}` gives {got}: the sum over all the worlds. Divide by the number of worlds for the mean: `total / len(worlds)`.",
                    nl="`{call}` geeft {got}: de som over alle werelden. Deel door het aantal werelden voor het gemiddelde: `total / len(worlds)`."),
            Mistake("wrong-count", wrong=_m("if predictions[i] != world[i]:\n    right = right + 1"),
                    en="`{call}` gives {got}, and it should give {want}. You count the examples where the prediction and the world differ: the mistakes, not the right answers.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je telt de voorbeelden waar de voorspelling en de wereld verschillen: de fouten, niet de goede antwoorden."),
            Mistake("half", matches=lambda got, text: re.search(r"return\s+0?\.5\b", text) is not None,
                    en="`{call}` gives {got}, and it should give {want}. Over all possible worlds the answer is always 0.5, but `worlds` need not be all of them: work out the mean for the worlds you get.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Over alle mogelijke werelden is het antwoord altijd 0,5, maar `worlds` hoeft ze niet allemaal te bevatten: reken het gemiddelde uit voor de werelden die je krijgt."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the two `...`: count a right answer when the prediction for example `i` equals the label in this world, and after `return` the mean over the worlds.",
                      "nl": "Vervang de twee `...`: tel een goed antwoord als de voorspelling voor voorbeeld `i` gelijk is aan het label in deze wereld, en na `return` het gemiddelde over de werelden."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(predictions, worlds)`.",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(predictions, worlds)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Do you count, in every world, the predictions that equal its labels, and take the mean of the shares over the worlds?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Tel je in elke wereld de voorspellingen die gelijk zijn aan haar labels, en neem je het gemiddelde van de aandelen over de werelden?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for any worlds, also when they are not all of them?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke set werelden, ook als het ze niet allemaal zijn?"},
            "correct": {"en": "Your function is right in all {n} cases. Over all worlds every rule gets 0.5, whatever it predicts. Only if you keep just the worlds where the first unseen example is 0 does the rule that says 0 there do better: 0.667. An assumption pays only where it holds.",
                        "nl": "Je functie klopt in alle {n} gevallen. Over alle werelden krijgt elke regel 0,5, wat hij ook voorspelt. Pas als je alleen de werelden houdt waarin het eerste onbekende voorbeeld 0 is, doet de regel die daar 0 zegt het beter: 0,667. Een aanname betaalt zich alleen uit waar ze klopt."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("all-worlds", _A_START, "empty", None),
    ("all-worlds", _A, "correct", None),
    ("all-worlds", _a("longer.append(world + [1])\nlonger.append(world + [0])"), "correct", None),
    ("all-worlds", _a("longer.append([0] + world)\nlonger.append([1] + world)"), "correct", None),
    ("all-worlds", _a("longer.append(world + [0])"), "wrong", "one-way"),
    ("all-worlds", _a("world.append(0)\nlonger.append(world)\nlonger.append(world)"), "wrong", "same-list"),
    ("all-worlds", _a("longer.append(0)\nlonger.append(1)"), "wrong", "numbers"),
    ("all-worlds", _a("longer.append(world + [0])\nlonger.append(world + [0])"), "wrong", None),
    ("mean-right", _M_START, "empty", None),
    ("mean-right", _m(back="..."), "wrong", "dots-left"),
    ("mean-right", _M, "correct", None),
    ("mean-right", _m("if world[i] == predictions[i]:\n    right = right + 1"), "correct", None),
    ("mean-right", _m(back="total"), "wrong", "sum"),
    ("mean-right", _m("if predictions[i] != world[i]:\n    right = right + 1"), "wrong", "wrong-count"),
    ("mean-right", _m(back="0.5"), "wrong", "half"),
    ("mean-right", _m(back="total / len(predictions)"), "wrong", None),
]
