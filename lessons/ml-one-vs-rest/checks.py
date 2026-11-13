# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-one-vs-rest: three species from classifiers for two. One-vs-rest (a model per species
against the other two, and the highest score wins) and one-vs-one (a model per pair of species, and the most votes
win), on the penguins of ml-knn by bill length and body mass; and an output code. The functions ovr_predict and
ovo_votes, and the pictures: the boundaries of both ways, a penguin of the learner's own, and the learner's
one-vs-rest on every validation penguin."""
import contextlib
import functools
import importlib.util
import io
import re
import warnings

from everything_ai import chart, check as checking, paths, yours
from everything_ai.check import FunctionBox, Mistake


def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_ovr",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_KNN = _lesson_checks("ml-knn")   # the penguins and their split, as the learner met them
SPECIES = _KNN.SPECIES            # Adelie, Chinstrap, Gentoo
TWO = _KNN.TWO                    # bill length and body mass
C_ALL = 1.0                       # the price of slack of every street in this lesson, the middle of the last lesson's slider
X_RANGE, Y_RANGE = (32.0, 60.0), (2600.0, 6400.0)
BILLS = [34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58]
MASSES = [2800, 3200, 3600, 4000, 4400, 4800, 5200, 5600, 6000, 6400]


# --- the course's two functions, as the lesson writes them ----------------------------------------------------------

def ovr_predict(names, W, b, x):
    import numpy as np
    scores = x @ W + b
    return names[np.argmax(scores)]


def ovo_votes(x, names, pairs):
    votes = {}
    for name in names:
        votes[name] = 0
    for first, second, w, b in pairs:
        if x @ w + b > 0:
            votes[second] = votes[second] + 1
        else:
            votes[first] = votes[first] + 1
    return votes


# --- the streets: three for one-vs-rest, three for one-vs-one, rounded to 3 decimals as the lesson prints them -------

def _street(X, y) -> tuple:
    """w and b of the soft-margin street with C = C_ALL for labels y of -1 and +1, rounded as the lesson prints them."""
    import numpy as np
    from sklearn.svm import SVC
    fit = SVC(kernel="linear", C=C_ALL, tol=1e-8).fit(X, y)
    return [round(float(v), 3) for v in fit.coef_[0]], round(float(fit.intercept_[0]), 3)


@functools.lru_cache(maxsize=1)
def ovr_models() -> tuple:
    """W, with a column per species (its model: that species +1, the other two -1), and b, a number per species."""
    import numpy as np
    s = _KNN.split(TWO)
    cols, bs = [], []
    for name in SPECIES:
        w, b = _street(s["X_train"], np.where(s["y_train"] == name, 1, -1))
        cols.append(w)
        bs.append(b)
    return [[cols[j][i] for j in range(len(SPECIES))] for i in range(2)], bs


@functools.lru_cache(maxsize=1)
def ovo_models() -> list:
    """(first, second, w, b) for every pair of species, trained on the penguins of those two only: second +1, first -1."""
    import numpy as np
    s = _KNN.split(TWO)
    out = []
    for i in range(len(SPECIES)):
        for j in range(i + 1, len(SPECIES)):
            keep = np.isin(s["y_train"], [SPECIES[i], SPECIES[j]])
            w, b = _street(s["X_train"][keep], np.where(s["y_train"][keep] == SPECIES[j], 1, -1))
            out.append((SPECIES[i], SPECIES[j], w, b))
    return out


def _std(bill, mass):
    import numpy as np
    s = _KNN.split(TWO)
    return (np.stack([np.asarray(bill, dtype=float), np.asarray(mass, dtype=float)], axis=-1) - s["mean"]) / s["std"]


def _ovr_class(Z):
    """The species one-vs-rest says for every row of Z (standardized), as numbers."""
    import numpy as np
    W, b = ovr_models()
    return np.argmax(Z @ np.array(W) + np.array(b), axis=-1)


def _ovo_count(Z):
    import numpy as np
    votes = np.zeros(Z.shape[:-1] + (len(SPECIES),), dtype=int)
    for first, second, w, b in ovo_models():
        yes = Z @ np.array(w) + b > 0
        votes[..., SPECIES.index(second)] += yes
        votes[..., SPECIES.index(first)] += ~yes
    return votes


def _ovo_class(Z):
    import numpy as np
    return np.argmax(_ovo_count(Z), axis=-1)   # the first of equal counts, as np.argmax does


def scores(bill: float, mass: float) -> list:
    """The score of each one-vs-rest model for a penguin of this bill length and body mass."""
    import numpy as np
    W, b = ovr_models()
    return [float(v) for v in _std(bill, mass) @ np.array(W) + np.array(b)]


def votes(bill: float, mass: float) -> list:
    return [int(v) for v in _ovo_count(_std(bill, mass))]


def grouped(n: int, lang: str) -> str:
    """A whole number with its thousands marked, as the language writes it: 4,000 or 4.000."""
    return f"{int(n):,}".replace(",", "." if lang == "nl" else ",")


def ovr_says(bill: float, mass: float) -> str:
    return SPECIES[int(_ovr_class(_std(bill, mass)))]


def ovo_says(bill: float, mass: float) -> str:
    return SPECIES[int(_ovo_class(_std(bill, mass)))]


@functools.lru_cache(maxsize=1)
def facts() -> dict:
    """What the two ways do on the training and validation sets, and on the plane of the picture."""
    import numpy as np
    s = _KNN.split(TWO)
    out = {"n_train": len(s["y_train"]), "n_val": len(s["y_val"])}
    names = np.array(SPECIES)
    for part in ("train", "val"):
        Z, y = s[f"X_{part}"], s[f"y_{part}"]
        r, o = names[_ovr_class(Z)], names[_ovo_class(Z)]
        out[f"ovr_{part}"], out[f"ovo_{part}"] = int((r != y).sum()), int((o != y).sum())
        out[f"differ_{part}"] = int((r != o).sum())
        W, b = ovr_models()
        sc = Z @ np.array(W) + np.array(b)
        out[f"no_yes_{part}"], out[f"several_{part}"] = int(((sc > 0).sum(1) == 0).sum()), int(((sc > 0).sum(1) > 1).sum())
        out[f"ties_{part}"] = int((_ovo_count(Z).max(1) == 1).sum())
    gx, gy = np.meshgrid(np.linspace(*X_RANGE, 281), np.linspace(*Y_RANGE, 381))
    G = _std(gx.ravel(), gy.ravel())
    W, b = ovr_models()
    sg = G @ np.array(W) + np.array(b)
    out["plane_no_yes"] = float(((sg > 0).sum(1) == 0).mean())
    out["plane_several"] = float(((sg > 0).sum(1) > 1).mean())
    out["plane_differ"] = float((_ovr_class(G) != _ovo_class(G)).mean())
    out["plane_ties"] = float((_ovo_count(G).max(1) == 1).mean())
    return out


def models(k: int) -> tuple:
    """How many models one-vs-rest and one-vs-one need for k classes."""
    return k, k * (k - 1) // 2


# --- the pictures ---------------------------------------------------------------------------------------------------

def _raw_line(w, b):
    """The line w·z + b = 0 of standardized z, as a·x + c = 0 in millimetres and grams: (a, c)."""
    import numpy as np
    s = _KNN.split(TWO)
    a = np.asarray(w, dtype=float) / s["std"]
    return a, float(b - a @ s["mean"])


def _clip(a, c):
    """The piece of the line a·x + c = 0 inside the picture, as two points, or None."""
    pts = []
    if abs(a[1]) > 1e-15:
        pts += [(x, -(c + a[0] * x) / a[1]) for x in X_RANGE]
    if abs(a[0]) > 1e-15:
        pts += [(-(c + a[1] * y) / a[0], y) for y in Y_RANGE]
    inside = sorted({(round(x, 9), round(y, 9)) for x, y in pts
                     if X_RANGE[0] - 1e-9 <= x <= X_RANGE[1] + 1e-9 and Y_RANGE[0] - 1e-9 <= y <= Y_RANGE[1] + 1e-9})
    return (inside[0], inside[-1]) if len(inside) >= 2 and inside[0] != inside[-1] else None


def _edges(lines, rule, n: int = 700) -> list:
    """The pieces of these lines (each (w, b) in standardized units) where the species that `rule` gives changes
    from one side to the other: the boundaries of the rule, as (x0, y0, x1, y1) in millimetres and grams."""
    import numpy as np
    sx, sy = X_RANGE[1] - X_RANGE[0], Y_RANGE[1] - Y_RANGE[0]
    out = []
    for w, b in lines:
        seg = _clip(*_raw_line(w, b))
        if seg is None:
            continue
        (x0, y0), (x1, y1) = seg
        du, dv = (x1 - x0) / sx, (y1 - y0) / sy
        length = (du * du + dv * dv) ** 0.5
        nu, nv = -dv / length * 0.003, du / length * 0.003   # a step of 0.3 % of the picture to each side
        t = (np.arange(n) + 0.5) / n
        px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        left = rule(_std(px + nu * sx, py + nv * sy))
        right = rule(_std(px - nu * sx, py - nv * sy))
        edge = left != right
        start = None
        for i in range(n + 1):
            if i < n and edge[i] and start is None:
                start = i
            if (i == n or not edge[i]) and start is not None:
                ta, tb = start / n, i / n
                out.append((x0 + ta * (x1 - x0), y0 + ta * (y1 - y0), x0 + tb * (x1 - x0), y0 + tb * (y1 - y0)))
                start = None
    return out


@functools.lru_cache(maxsize=1)
def _ovr_edges() -> list:
    import numpy as np
    W, b = np.array(ovr_models()[0]), np.array(ovr_models()[1])
    lines = [(W[:, i] - W[:, j], b[i] - b[j]) for i in range(len(SPECIES)) for j in range(i + 1, len(SPECIES))]
    return _edges(lines, _ovr_class)


@functools.lru_cache(maxsize=1)
def _ovo_edges() -> list:
    return _edges([(w, b) for _, _, w, b in ovo_models()], _ovo_class)


def _model_lines(which: str) -> list:
    """The lines where one model's score is 0, clipped to the picture."""
    import numpy as np
    if which == "ovr":
        W, b = np.array(ovr_models()[0]), ovr_models()[1]
        lines = [(W[:, i], b[i]) for i in range(len(SPECIES))]
    else:
        lines = [(w, b) for _, _, w, b in ovo_models()]
    return [seg for seg in (_clip(*_raw_line(w, b)) for w, b in lines) if seg]


def _penguins(panel, part: str = "train", r: float = 3.2):
    import numpy as np
    s = _KNN.split(TWO)
    raw = s["X_" + part] * s["std"] + s["mean"]
    for k, name in enumerate(SPECIES):
        rows = raw[s["y_" + part] == name]
        panel.group(rows[:, 0], rows[:, 1], k, r=r)


def _panel(title: str, labels: dict):
    return chart.Panel(X_RANGE, Y_RANGE, title=title, xlabel=labels["x"], ylabel=labels["y"],
                       xticks=[35, 40, 45, 50, 55, 60], yticks=[3000, 4000, 5000, 6000])


def ways_picture(labels: dict) -> str:
    """Two panels, the training penguins in each. Left, one-vs-rest: dashed, the line of each of its three models
    (where its score is 0), and solid, where the highest score changes species. Right, one-vs-one: dashed, the line
    of each pair's model, and solid, where the most votes change species."""
    panels = []
    for which, title, edges in (("ovr", labels["ovr"], _ovr_edges()), ("ovo", labels["ovo"], _ovo_edges())):
        p = _panel(title, labels)
        for (x0, y0), (x1, y1) in _model_lines(which):
            p.line([x0, x1], [y0, y1], "ink-3", 1.3, dashed=True)
        _penguins(p)
        for x0, y0, x1, y1 in edges:
            p.line([x0, x1], [y0, y1], "pred", 2.4)
        panels.append(p)
    return chart.figure(panels, alt=labels["alt"], width=330, height=300)


def plane_picture(bill: float, mass: float, labels: dict) -> str:
    """A penguin of the learner's own at (bill, mass), with the training penguins, the boundaries of one-vs-rest
    (solid) and those of one-vs-one (dashed)."""
    p = _panel(labels["title"], labels)
    _penguins(p, r=2.8)
    for x0, y0, x1, y1 in _ovo_edges():
        p.line([x0, x1], [y0, y1], "ink", 1.6, dashed=True)
    for x0, y0, x1, y1 in _ovr_edges():
        p.line([x0, x1], [y0, y1], "pred", 2.4)
    p.points([bill], [mass], "ink", r=7.5, hollow=True)
    p.points([bill], [mass], "ink", r=3.2)
    return chart.figure([p], alt=labels["alt"], width=360, height=320)


# --- how the learner's answers are written ---------------------------------------------------------------------------

class _Val:
    """A test value that the check's messages write by its name, such as `W` or `pairs`; the function gets a plain
    copy: an array for numbers, a list for names, and for pairs a list of (first, second, array, number)."""

    def __init__(self, name: str, value, kind: str = "array"):
        self.name, self.value, self.kind = name, value, kind

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import numpy as np
        if self.kind == "names":
            return list(self.value)
        if self.kind == "pairs":
            return [(f, s, np.array(w, dtype=float), float(b)) for f, s, w, b in self.value]
        return np.array(self.value, dtype=float)


def _x(*values) -> _Val:
    return _Val("np.array([" + ", ".join(str(float(v)) for v in values) + "])", list(values))


def _shown(v) -> str:
    """What a function gave back, as the lesson writes it: a species as "Adelie", numbers rounded to 3 decimals."""
    import numpy as np
    if isinstance(v, (np.str_, str)):
        return repr(str(v))
    if isinstance(v, np.ndarray) and v.dtype.kind in "iuf":
        return str(np.round(v.astype(float), 3).tolist())
    if isinstance(v, (float, np.floating)):
        return str(round(float(v), 3))
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return str(int(v))
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_shown(k)}: {_shown(x)}" for k, x in v.items()) + "}"
    return repr(v)


def _same_votes(got, want) -> bool:
    if not isinstance(got, dict):
        return False
    try:
        return {str(k): int(v) for k, v in got.items()} == {str(k): int(v) for k, v in want.items()} and all(
            float(v) == int(v) for v in got.values())
    except (TypeError, ValueError):
        return False


_OVR_WORDS = {
    "title": {"en": "Your one-vs-rest on the {n} validation penguins", "nl": "Jouw one-vs-rest op de {n} pinguïns uit de validation set"},
    "said": {"en": "Each penguin drawn as the species your function says, with a ring where that is not its real species: {got} rings. The course's one-vs-rest gets {want} wrong.",
             "nl": "Elke pinguïn getekend als de soort die jouw functie zegt, met een ring waar dat niet zijn echte soort is: {got} ringen. De one-vs-rest van de course heeft er {want} fout."},
}


def _ovr_picture(seen) -> str:
    """The learner's ovr_predict, with the course's W and b, on every validation penguin: each drawn as the species
    the function says, ringed where that is not its species."""
    import numpy as np
    ex = seen.exercise
    mine = seen.mine
    if mine is None:
        return ""
    fn = mine.names.get(ex.function)
    if not callable(fn):
        return ""
    s = _KNN.split(TWO)
    W, b = ovr_models()
    said = []
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            def run():
                return [fn(list(SPECIES), np.array(W), np.array(b), np.array(x)) for x in s["X_val"]]
            said = checking.with_time_limit(run)
    except Exception:
        return ""
    said = [str(v) for v in said]
    if not all(v in SPECIES for v in said):
        return ""
    raw = s["X_val"] * s["std"] + s["mean"]
    lang = seen.lang
    p = _panel("", {"x": "bill length (mm)" if lang == "en" else "snavellengte (mm)",
                    "y": "body mass (g)" if lang == "en" else "lichaamsgewicht (g)"})
    said = np.array(said)
    for k, name in enumerate(SPECIES):
        rows = raw[said == name]
        p.group(rows[:, 0], rows[:, 1], k, r=3.8)
    wrong = said != s["y_val"]
    if wrong.any():
        p.points(raw[wrong, 0], raw[wrong, 1], "loss", r=8.5, hollow=True)
    want = facts()["ovr_val"]
    words = {k: v[lang] for k, v in _OVR_WORDS.items()}
    title = words["title"].format(n=len(said))
    text = words["said"].format(got=int(wrong.sum()), want=want)
    key = ('<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.15rem">'
           + text + "</div>")
    return yours.figure(title, chart.figure([p], alt=title + ". " + text, width=360, height=300) + key, alt=text)


# The lesson is done when these are correct. "output-code" is the choice question in the lesson.
REQUIRED = ["ovr-predict", "ovo-votes", "output-code"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# The three species and the course's one-vs-rest models; a made-up set of four classes with two measurements, and
# one of three species with three measurements.
_NAMES = _Val("names", SPECIES, "names")
_W = _Val("W", ovr_models()[0])
_B = _Val("b", ovr_models()[1])
_NAMES4 = _Val("names4", ["cat", "dog", "fox", "owl"], "names")
_W4 = _Val("W4", [[1.0, -0.5, 0.2, -1.0], [0.3, 0.8, -1.2, 0.1]])
_B4 = _Val("b4", [-0.2, 0.1, 0.0, -0.4])
_W3 = _Val("W3", [[1.2, -0.4, -0.6], [-0.3, 0.9, -0.5], [0.5, -0.2, 0.7]])
_B3 = _Val("b3", [-0.5, -0.3, -0.6])
_PAIRS = _Val("pairs", ovo_models(), "pairs")
_PAIRS4 = _Val("pairs4", [("cat", "dog", [1.0, -0.5], 0.2), ("cat", "fox", [0.4, 1.0], -0.1), ("cat", "owl", [-1.0, 0.3], 0.0),
                          ("dog", "fox", [0.6, 0.6], -0.3), ("dog", "owl", [-0.2, 1.1], 0.4), ("fox", "owl", [1.3, -0.2], -0.5)],
               "pairs")
_PAIRS3 = _Val("pairs3", [("Adelie", "Chinstrap", [1.1, -0.6, 0.2], -0.3), ("Adelie", "Gentoo", [0.7, 0.9, -0.4], -0.2),
                          ("Chinstrap", "Gentoo", [-0.8, 1.2, 0.5], 0.1)], "pairs")

# --- exercise 1: ovr_predict ----------------------------------------------------------------------------------------


def _o(score: str = "x @ W + b", back: str = "names[np.argmax(scores)]") -> str:
    return ("import numpy as np\n\n\ndef ovr_predict(names, W, b, x):\n"
            f"    scores = {score}\n    return {back}\n")


_O_START = _o("...", "...")
_O = _o()

# --- exercise 2: ovo_votes ------------------------------------------------------------------------------------------

_V_HEAD = ("def ovo_votes(x, names, pairs):\n    votes = {}\n    for name in names:\n        votes[name] = 0\n"
           "    for first, second, w, b in pairs:\n")
_V_TAIL = "    return votes\n"


def _v(body: str = "if x @ w + b > 0:\n    votes[second] = votes[second] + 1\nelse:\n    votes[first] = votes[first] + 1") -> str:
    return _V_HEAD + "".join(f"        {line}\n" for line in body.split("\n")) + _V_TAIL


_V_START = _v("...")
_V = _v()

EXERCISES = {
    "ovr-predict": FunctionBox(
        name="ovr-predict",
        picture=_ovr_picture,
        function="ovr_predict",
        start=_O_START,
        solution=_O,
        show=_shown,
        visible=[(_NAMES, _W, _B, _x(-1.0, -0.4))],
        hidden=[(_NAMES, _W, _B, _x(0.361, 0.058)), (_NAMES, _W, _B, _x(0.901, 0.378)), (_NAMES, _W, _B, _x(-0.2, 1.0)),
                (_NAMES, _W, _B, _x(1.0, -0.8)),
                (_NAMES4, _W4, _B4, _x(0.5, 1.0)), (_NAMES, _W3, _B3, _x(0.2, 0.4, 1.0))],
        first=[_DOTS],
        mistakes=[
            Mistake("lowest", wrong=_o(back="names[np.argmin(scores)]"),
                    en="`{call}` gives {got}, and it should give {want}. `np.argmin` picks the model with the lowest score: the one most sure it is *not* its species. Take the highest: `np.argmax`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. `np.argmin` kiest het model met de laagste score: het model dat het zekerst weet dat het *niet* zijn soort is. Neem de hoogste: `np.argmax`."),
            Mistake("position", wrong=_o(back="np.argmax(scores)"),
                    en="`{call}` gives {got}: the position of the highest score, counting from 0, not its species. Look it up in `names`.",
                    nl="`{call}` geeft {got}: de positie van de hoogste score, geteld vanaf 0, niet de soort. Zoek hem op in `names`."),
            Mistake("first-yes", wrong=_o(back="names[np.argmax(scores > 0)]"),
                    en="`{call}` gives {got}, and it should give {want}. You take the first model that says yes, with a score above 0. Here no model says yes, or more than one does; the highest score still picks exactly one species.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je neemt het eerste model dat ja zegt, met een score boven 0. Hier zegt geen enkel model ja, of meer dan één; de hoogste score kiest toch precies één soort."),
            Mistake("no-b", wrong=_o("x @ W"),
                    en="`{call}` gives {got}, and it should give {want}. Without `+ b` every model's line goes through the middle of the data. Add each model's own `b`: `x @ W + b`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Zonder `+ b` gaat de lijn van elk model door het midden van de data. Tel de eigen `b` van elk model erbij op: `x @ W + b`."),
            Mistake("scores", wrong=_o(back="scores"),
                    en="`{call}` gives {got}: the score of each model. The function gives back one species: that of the highest score.",
                    nl="`{call}` geeft {got}: de score van elk model. De functie geeft één soort terug: die van de hoogste score."),
            Mistake("shapes", error=lambda e: isinstance(e, ValueError) and "matmul" in str(e),
                    en="With `{call}`, `@` stops: the shapes do not fit. `W` has a row for each measurement and a column for each model, so `x @ W` gives one score for each model.",
                    nl="Met `{call}` stopt `@`: de shapes passen niet. `W` heeft een rij voor elke meting en een kolom voor elk model, dus `x @ W` geeft één score voor elk model."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the two `...`: the score of every model in `scores`, and after `return` the species of the highest score.",
                      "nl": "Vervang de twee `...`: de score van elk model in `scores`, en na `return` de soort van de hoogste score."},
            "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Keep `(names, W, b, x)`.",
                          "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(names, W, b, x)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Do you work out a score for every model, and give back the species of the highest?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Bereken je een score voor elk model, en geef je de soort van de hoogste terug?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work when no model, or more than one, says yes, and for any number of classes and measurements?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie ook als geen enkel model ja zegt, of meer dan één, en voor elk aantal klassen en metingen?"},
            "correct": {"en": "Your function is right in all {n} cases, also where all three scores are below 0: no model says yes, and the highest score still picks one species. The picture runs it on every validation penguin.",
                        "nl": "Je functie klopt in alle {n} gevallen, ook waar alle drie de scores onder 0 liggen: geen model zegt ja, en de hoogste score kiest toch één soort. Het beeld voert hem uit op elke pinguïn uit de validation set."},
        },
    ),
    "ovo-votes": FunctionBox(
        name="ovo-votes",
        picture=yours.bars(call=0),
        function="ovo_votes",
        start=_V_START,
        solution=_V,
        show=_shown,
        compare=_same_votes,
        visible=[(_x(0.361, 0.058), _NAMES, _PAIRS)],
        hidden=[(_x(-1.0, -0.4), _NAMES, _PAIRS), (_x(0.9, 1.3), _NAMES, _PAIRS), (_x(0.5, -1.2), _NAMES, _PAIRS),
                (_x(0.3, -0.6), _NAMES4, _PAIRS4), (_x(0.2, 0.4, 1.0), _NAMES, _PAIRS3)],
        first=[_DOTS],
        mistakes=[
            Mistake("flipped", wrong=_v("if x @ w + b > 0:\n    votes[first] = votes[first] + 1\nelse:\n    votes[second] = votes[second] + 1"),
                    en="`{call}` gives {got}, and it should give {want}. Each pair's model learned +1 for `second` and -1 for `first`: a score above 0 is a vote for `second`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Het model van elk paar leerde +1 voor `second` en -1 voor `first`: een score boven 0 is een stem voor `second`."),
            Mistake("no-else", wrong=_v("if x @ w + b > 0:\n    votes[second] = votes[second] + 1"),
                    en="`{call}` gives {got}, and it should give {want}. Every pair's model votes, also when its score is not above 0: then its vote goes to `first`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Het model van elk paar stemt, ook als zijn score niet boven 0 ligt: dan gaat zijn stem naar `first`."),
            Mistake("no-b", wrong=_v("if x @ w > 0:\n    votes[second] = votes[second] + 1\nelse:\n    votes[first] = votes[first] + 1"),
                    en="`{call}` gives {got}, and it should give {want}. The score of a pair's model is `x @ w + b`: without `b` its line goes through the middle of the data.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. De score van het model van een paar is `x @ w + b`: zonder `b` gaat zijn lijn door het midden van de data."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the `...` with four lines: work out the score of this pair's model, and give one vote to `second` if it is above 0, otherwise to `first`.",
                      "nl": "Vervang de `...` door vier regels: bereken de score van het model van dit paar, en geef één stem aan `second` als die boven 0 ligt, anders aan `first`."},
            "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has a different number of names between its parentheses. Keep `(x, names, pairs)`.",
                          "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x, names, pairs)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does every pair's model give exactly one vote, to `second` for a score above 0 and to `first` otherwise?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Geeft het model van elk paar precies één stem, aan `second` bij een score boven 0 en anders aan `first`?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for any penguin, any number of classes and any number of measurements?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke pinguïn, elk aantal klassen en elk aantal metingen?"},
            "correct": {"en": "Your function is right in all {n} cases. For the example, one-vs-rest said Chinstrap, with no model saying yes; the three pairs give Gentoo two votes, and the penguin is a Gentoo.",
                        "nl": "Je functie klopt in alle {n} gevallen. Voor het voorbeeld zei one-vs-rest Chinstrap, zonder dat een model ja zei; de drie paren geven Gentoo twee stemmen, en de pinguïn is een Gentoo."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("ovr-predict", _O_START, "empty", None),
    ("ovr-predict", _o(back="..."), "wrong", "dots-left"),
    ("ovr-predict", _O, "correct", None),
    ("ovr-predict", _o("np.dot(x, W) + b"), "correct", None),
    ("ovr-predict", _o(back="names[scores.argmax()]"), "correct", None),
    ("ovr-predict", _o(back="names[np.argmin(scores)]"), "wrong", "lowest"),
    ("ovr-predict", _o(back="np.argmax(scores)"), "wrong", "position"),
    ("ovr-predict", _o(back="names[np.argmax(scores > 0)]"), "wrong", "first-yes"),
    ("ovr-predict", _o("x @ W"), "wrong", "no-b"),
    ("ovr-predict", _o(back="scores"), "wrong", "scores"),
    ("ovr-predict", _o("W @ x + b"), "error", "shapes"),
    ("ovr-predict", _o(back="names[0]"), "almost", None),
    ("ovo-votes", _V_START, "empty", None),
    ("ovo-votes", _V, "correct", None),
    ("ovo-votes", _v("score = x @ w + b\nif score > 0:\n    votes[second] = votes[second] + 1\nelse:\n    votes[first] = votes[first] + 1"), "correct", None),
    ("ovo-votes", _v("if x @ w + b <= 0:\n    votes[first] = votes[first] + 1\nelse:\n    votes[second] = votes[second] + 1"), "correct", None),
    ("ovo-votes", _v("if x @ w + b > 0:\n    votes[first] = votes[first] + 1\nelse:\n    votes[second] = votes[second] + 1"), "wrong", "flipped"),
    ("ovo-votes", _v("if x @ w + b > 0:\n    votes[second] = votes[second] + 1"), "wrong", "no-else"),
    ("ovo-votes", _v("if x @ w > 0:\n    votes[second] = votes[second] + 1\nelse:\n    votes[first] = votes[first] + 1"), "wrong", "no-b"),
]
