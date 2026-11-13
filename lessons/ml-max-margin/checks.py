# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-max-margin: the widest street between the Chinstraps and the Gentoos of ml-knn, by bill
length and body mass. The functions margin (the room the nearest example has, on its own side of a line) and slacks
(how far each example lies inside the street or beyond it), and why only the support vectors decide. And the
pictures: the street for every C, on the training set and on the validation set, and the learner's margin and slacks
drawn around their line."""
import functools
import importlib.util
import math
import re

from everything_ai import chart, check as checking, course, paths, yours
from everything_ai.check import FunctionBox, Mistake


def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_margin",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_KNN = _lesson_checks("ml-knn")   # the penguins and their split, as the learner met them
SPECIES = _KNN.SPECIES
TWO = _KNN.TWO                    # bill length and body mass
X_RANGE, Y_RANGE = (32.0, 60.0), (2600.0, 6400.0)
PAIR = ("Chinstrap", "Gentoo")    # y = -1 and y = +1
CS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 1000]
HARD = 1e6                        # a price so high that no penguin may lie inside the street: the hard margin
_EDGE = 1e-4                      # how close to an edge of the street a penguin counts as on it: the solver stops near 1


# --- the course's margin and slacks, as the lesson writes them -------------------------------------------------------

def margin(w, b, X, y):
    import numpy as np
    sides = y * (X @ w + b)
    return np.min(sides) / np.linalg.norm(w)


def slacks(w, b, X, y):
    s = 1 - y * (X @ w + b)
    s[s < 0] = 0
    return s


# --- the Chinstraps and the Gentoos, and their street for every C -----------------------------------------------------

@functools.lru_cache(maxsize=1)
def pair() -> dict:
    """The Chinstraps and Gentoos of the training and validation sets of ml-knn, standardized as there (with all 206
    training penguins), with y = -1 for a Chinstrap and +1 for a Gentoo; and their raw measurements."""
    import numpy as np
    s = _KNN.split(TWO)
    out = {}
    for part in ("train", "val"):
        keep = np.isin(s[f"y_{part}"], PAIR)
        out[f"X_{part}"] = s[f"X_{part}"][keep]
        out[f"y_{part}"] = np.where(s[f"y_{part}"][keep] == PAIR[1], 1, -1)
        out[f"raw_{part}"] = out[f"X_{part}"] * s["std"] + s["mean"]
    out["mean"], out["std"] = s["mean"], s["std"]
    return out


@functools.lru_cache(maxsize=None)
def street(C: float) -> tuple:
    """w and b of the soft-margin street for this C on the training set: the smallest 1/2·|w|² + C·(sum of the
    slacks), found by scikit-learn's solver for it. With C = HARD, the widest street with every penguin outside it."""
    import numpy as np
    from sklearn.svm import SVC
    p = pair()
    fit = SVC(kernel="linear", C=C, tol=1e-10).fit(p["X_train"], p["y_train"])
    return tuple(float(v) for v in fit.coef_[0]), float(fit.intercept_[0])


def facts(C: float) -> dict:
    """What the street of this C does: its width (in standardized units), the training penguins on its edges or
    inside it, those on the wrong side, the sum of their slacks, and the validation penguins on the wrong side."""
    import numpy as np
    p = pair()
    w, b = street(C)
    w = np.array(w)
    side = p["y_train"] * (p["X_train"] @ w + b)
    side_val = p["y_val"] * (p["X_val"] @ w + b)
    return {"width": float(2 / np.linalg.norm(w)), "edge": int((side <= 1 + _EDGE).sum()), "inside": int((side < 1 - _EDGE).sum()),
            "wrong": int((side < 0).sum()), "slack": float(slacks(w, b, p["X_train"], p["y_train"]).sum()),
            "val_wrong": int((side_val < 0).sum()), "n_train": len(side), "n_val": len(side_val)}


def c_text(c: float, lang: str) -> str:
    """A value of C as the lesson writes it: 0.01 and 1,000, in Dutch 0,01 and 1.000."""
    text = f"{c:,g}"
    return text.translate(str.maketrans(",.", ".,")) if lang == "nl" else text


def best_cs() -> list:
    """The values of C with the fewest validation mistakes."""
    fewest = min(facts(C)["val_wrong"] for C in CS)
    return [C for C in CS if facts(C)["val_wrong"] == fewest]


# --- pictures -----------------------------------------------------------------------------------------------------

def _segment(w, b, t, xr, yr):
    """The piece of the line w·x + b = t inside the box xr × yr, as two points, or None."""
    pts = []
    if abs(w[1]) > 1e-12:
        pts += [(x, (t - b - w[0] * x) / w[1]) for x in xr]
    if abs(w[0]) > 1e-12:
        pts += [((t - b - w[1] * y) / w[0], y) for y in yr]
    inside = sorted({(round(x, 9), round(y, 9)) for x, y in pts
                     if xr[0] - 1e-9 <= x <= xr[1] + 1e-9 and yr[0] - 1e-9 <= y <= yr[1] + 1e-9})
    return (inside[0], inside[-1]) if len(inside) >= 2 and inside[0] != inside[-1] else None


def _raw_line(w, b):
    """The line w·z + b in standardized units as w'·x + b' in millimeters and grams."""
    p = pair()
    wr = [w[0] / p["std"][0], w[1] / p["std"][1]]
    return wr, b - wr[0] * p["mean"][0] - wr[1] * p["mean"][1]


def _draw_street(panel, w, b, xr, yr, edges: bool = True):
    for t in ((-1, 1) if edges else ()):
        seg = _segment(w, b, t, xr, yr)
        if seg:
            panel.line([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], "pred", 1.4, dashed=True)
    seg = _segment(w, b, 0, xr, yr)
    if seg:
        panel.line([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], "pred", 2.4)


def _penguins(panel, raw, y, r: float = 3.6):
    for k, sign in ((SPECIES.index(PAIR[0]), -1), (SPECIES.index(PAIR[1]), 1)):
        rows = raw[y == sign]
        panel.group(rows[:, 0], rows[:, 1], k, r=r)


def street_picture(C: float, labels: dict) -> str:
    """The street of this C in millimeters and grams: left the training penguins, with a ring in ink around each one
    on an edge of the street or inside it; right the validation penguins, with a ring in the colour of mistakes
    around each one on the wrong side. Label keys: train, val, x, y, alt."""
    import numpy as np
    p = pair()
    w, b = street(C)
    wr, br = _raw_line(w, b)
    ticks = dict(xticks=[35, 40, 45, 50, 55, 60], yticks=[3000, 4000, 5000, 6000])
    panels = []
    for part, title in (("train", labels["train"]), ("val", labels["val"])):
        panel = chart.Panel(X_RANGE, Y_RANGE, title=title, xlabel=labels["x"], ylabel=labels["y"], **ticks)
        _draw_street(panel, wr, br, X_RANGE, Y_RANGE)
        raw, y = p[f"raw_{part}"], p[f"y_{part}"]
        _penguins(panel, raw, y)
        side = y * (p[f"X_{part}"] @ np.array(w) + b)
        if part == "train":
            ring = raw[side <= 1 + _EDGE]
            panel.points(ring[:, 0], ring[:, 1], "ink", r=7, hollow=True)
        wrong = raw[side < 0]
        if len(wrong):
            panel.points(wrong[:, 0], wrong[:, 1], "loss", r=8.5, hollow=True)
        panels.append(panel)
    return chart.figure(panels, alt=labels["alt"], width=330, height=290)


# --- how the learner's answers are written and drawn -----------------------------------------------------------------

class _Val:
    """A test value that the check's messages write by its name, such as `X` or `np.array([1.0, 1.0])`; the function
    gets a plain copy: an array for a list, a number for a number."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import numpy as np
        return np.array(self.value, dtype=float) if isinstance(self.value, list) else self.value


def _w(*values) -> _Val:
    return _Val("np.array([" + ", ".join(str(float(v)) for v in values) + "])", list(values))


def _shown(v) -> str:
    """A number rounded to 3 decimals, an array as a list of such numbers."""
    import numpy as np
    if isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
        return str(np.round(v.astype(float), 3).tolist())
    if isinstance(v, (float, np.floating)):
        return str(round(float(v), 3))
    if isinstance(v, np.generic):
        return str(v.item())
    return repr(v)[:60]


def _number(x: float, lang: str, digits: int = 2) -> str:
    text = f"{x:.{digits}f}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _case(seen, call: int = 0):
    """The arguments of case `call` as plain values (w, b, X, y), what the learner's function gives back for it and
    what the course's gives, or None."""
    import numpy as np
    out = yours._outcome(seen, call)
    if out is None:
        return None
    got, want, start, _ = out
    if len(start) < 4:
        return None
    try:
        w, b, X, y = (checking._copy(start[i][1]) for i in range(4))
        w, X, y = np.asarray(w, dtype=float), np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        b = float(b)
    except (TypeError, ValueError):
        return None
    if X.ndim != 2 or X.shape[1] != 2 or w.shape != (2,):
        return None
    return w, b, X, y, got, want


def _box(X, pad: float = 0.45):
    lo, hi = X.min(axis=0) - pad, X.max(axis=0) + pad
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def _key(items: list) -> str:
    return ('<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.15rem">'
            + "".join(f"<div>{t}</div>" for t in items) + "</div>")


_MARGIN_WORDS = {
    "title": {"en": "Your margin around the line of {call}", "nl": "Jouw marge rond de lijn van {call}"},
    "band": {"en": "Dashed: your margin, {got}, on both sides of the line. The ringed example is the nearest, {want} from the line: it should just touch a dashed line.",
             "nl": "Gestippeld: jouw marge, {got}, aan beide kanten van de lijn. Het omcirkelde voorbeeld ligt het dichtstbij, op {want} van de lijn: het hoort net een gestippelde lijn te raken."},
    "band-many": {"en": "Dashed: your margin, {got}, on both sides of the line. The {n} ringed examples are the nearest, each {want} from the line: they should just touch a dashed line.",
                  "nl": "Gestippeld: jouw marge, {got}, aan beide kanten van de lijn. De {n} omcirkelde voorbeelden liggen het dichtstbij, elk op {want} van de lijn: ze horen net een gestippelde lijn te raken."},
    "none": {"en": "Your margin, {got}, cannot be drawn as a distance. The ringed example is the nearest, {want} from the line.",
             "nl": "Jouw marge, {got}, kun je niet als afstand tekenen. Het omcirkelde voorbeeld ligt het dichtstbij, op {want} van de lijn."},
    "none-many": {"en": "Your margin, {got}, cannot be drawn as a distance. The {n} ringed examples are the nearest, each {want} from the line.",
                  "nl": "Jouw marge, {got}, kun je niet als afstand tekenen. De {n} omcirkelde voorbeelden liggen het dichtstbij, elk op {want} van de lijn."},
}


def _margin_picture(seen) -> str:
    """The examples of the first case and the line w·x + b = 0, with the learner's margin drawn as a band on both sides
    of it, and the nearest example ringed: the band of a right margin just touches it."""
    import numpy as np
    case = _case(seen)
    if case is None:
        return ""
    w, b, X, y, got, want = case
    try:
        got = float(got)
    except (TypeError, ValueError):
        return ""
    lang = seen.lang
    xr, yr = _box(X)
    panel = chart.Panel(xr, yr)
    norm = float(np.linalg.norm(w))
    drawn = math.isfinite(got) and 0 < got < 10
    if drawn:
        for t in (-got * norm, got * norm):
            seg = _segment(w, b, t, xr, yr)
            if seg:
                panel.line([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], "ink-2", 1.4, dashed=True)
    seg = _segment(w, b, 0, xr, yr)
    if seg:
        panel.line([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], "pred", 2.4)
    for k, sign in ((1, -1), (2, 1)):
        rows = X[y == sign]
        panel.group(rows[:, 0], rows[:, 1], k, r=5)
    side = y * (X @ w + b) / norm
    near = X[np.isclose(side, side.min())]
    panel.points(near[:, 0], near[:, 1], "ink", r=9, hollow=True)
    key = ("band" if drawn else "none") + ("-many" if len(near) > 1 else "")
    words = _MARGIN_WORDS[key][lang].format(got=_number(got, lang, 3), want=_number(float(side.min()), lang, 3), n=len(near))
    title = _MARGIN_WORDS["title"][lang].format(call=f"{seen.exercise.function}(…)")
    fig = chart.figure([panel], alt=words, width=300, height=260)
    return yours.figure(title, fig + _key([words]), alt=words)


_SLACK_WORDS = {
    "title": {"en": "Your slacks beside the examples", "nl": "Jouw slacks naast de voorbeelden"},
    "key": {"en": "The line and, dashed, the edges of its street. Beside every example its slack from your function, when it is not 0; {k} of them differ from the right ones, marked in purple.",
            "nl": "De lijn en, gestippeld, de randen van zijn straat. Naast elk voorbeeld zijn slack uit jouw functie, als die niet 0 is; {k} daarvan verschillen van de goede, paars gemarkeerd."},
    "key-right": {"en": "The line and, dashed, the edges of its street. Beside every example its slack from your function, when it is not 0: inside the street below 1, on the wrong side above 1.",
                  "nl": "De lijn en, gestippeld, de randen van zijn straat. Naast elk voorbeeld zijn slack uit jouw functie, als die niet 0 is: in de straat onder 1, aan de verkeerde kant boven 1."},
}


def _slacks_picture(seen) -> str:
    """The examples of the first case, the line w·x + b = 0 and the edges of its street, w·x + b = ±1, with beside every
    example the slack the learner's function gives it, when that is not 0; a slack that differs from the right one is
    written in the colour of mistakes."""
    import numpy as np
    case = _case(seen)
    if case is None:
        return ""
    w, b, X, y, got, want = case
    try:
        got = np.asarray(got, dtype=float).reshape(-1)
        want = np.asarray(want, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return ""
    if got.shape != want.shape or not np.all(np.isfinite(got)):
        return ""
    lang = seen.lang
    xr, yr = _box(X, pad=0.55)
    panel = chart.Panel(xr, yr)
    _draw_street(panel, w, b, xr, yr)
    for k, sign in ((1, -1), (2, 1)):
        rows = X[y == sign]
        panel.group(rows[:, 0], rows[:, 1], k, r=5)
    differ = ~np.isclose(got, want, atol=1e-6)
    for (x0, x1), g, bad in zip(X, got, differ):
        if abs(g) > 1e-9 or bad:
            panel.text(x0, x1 + 0.17 * (yr[1] - yr[0]) / 4, _number(float(g), lang), "loss" if bad else "ink-2")
    k = int(differ.sum())
    words = (_SLACK_WORDS["key"][lang].format(k=k) if k else _SLACK_WORDS["key-right"][lang])
    fig = chart.figure([panel], alt=words, width=320, height=280)
    return yours.figure(_SLACK_WORDS["title"][lang], fig + _key([words]), alt=words)


# --- the exercises ------------------------------------------------------------------------------------------------

# The lesson is done when these are correct. "not-a-support-vector" is the choice question in the lesson.
REQUIRED = ["margin", "slacks", "not-a-support-vector"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# Ten made-up penguins, already standardized: five Chinstraps (y = -1) and five Gentoos (y = +1). The line x₁ + x₂ = 0
# separates them; it is also their widest street. And three measurements for a set of six.
_PX = [[-1.2, -0.8], [-0.6, -1.4], [-1.5, 0.1], [-0.2, -0.9], [-0.9, -0.2], [0.9, 1.1], [0.3, 0.8], [1.4, 0.2], [0.6, 1.6], [1.1, 0.5]]
_PY = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]
_X, _Y = _Val("X", _PX), _Val("y", _PY)
_X3 = _Val("X3", [[0.5, 1.0, 0.2], [1.2, 0.1, 0.9], [0.8, 0.6, 1.5], [-0.7, -0.4, -0.2], [-1.1, 0.3, -0.8], [-0.2, -1.3, 0.1]])
_Y3 = _Val("y3", [1, 1, 1, -1, -1, -1])

# --- exercise 1: margin -----------------------------------------------------------------------------------------------


def _m(sides: str = "y * (X @ w + b)", back: str = "np.min(sides) / np.linalg.norm(w)") -> str:
    return ("import numpy as np\n\n\ndef margin(w, b, X, y):\n"
            f"    sides = {sides}\n    return {back}\n")


_M_START = _m("...", "...")
_M = _m()

# --- exercise 2: slacks -----------------------------------------------------------------------------------------------


def _s(first: str = "1 - y * (X @ w + b)", mask: str = "s < 0") -> str:
    return ("import numpy as np\n\n\ndef slacks(w, b, X, y):\n"
            f"    s = {first}\n    to_zero = {mask}\n    s[to_zero] = 0\n    return s\n")


_S_START = _s("...", "...")
_S = _s()

EXERCISES = {
    "margin": FunctionBox(
        name="margin",
        picture=_margin_picture,
        function="margin",
        start=_M_START,
        solution=_M,
        show=_shown,
        visible=[(_w(1, 1), 0.0, _X, _Y)],
        hidden=[(_w(3, 3), 0.0, _X, _Y), (_w(1, 0), -0.2, _X, _Y), (_w(0, 1), 0.0, _X, _Y), (_w(1, -0.5, 2), 0.3, _X3, _Y3)],
        first=[_DOTS],
        mistakes=[
            Mistake("no-norm", wrong=_m(back="np.min(sides)"),
                    en="`{call}` gives {got}, and it should give {want}. `np.min(sides)` is not yet a distance: it grows when you multiply `w` and `b` by the same number, and the line stays where it is. Divide by the length of `w`, `np.linalg.norm(w)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. `np.min(sides)` is nog geen afstand: het groeit als je `w` en `b` met hetzelfde getal vermenigvuldigt, en de lijn blijft waar hij is. Deel door de lengte van `w`, `np.linalg.norm(w)`."),
            Mistake("squared", wrong=_m(back="np.min(sides) / np.sum(w ** 2)"),
                    en="`{call}` gives {got}, and it should give {want}. You divide by the sum of the squares of `w`; the length of `w` is the square root of that: `np.linalg.norm(w)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je deelt door de som van de kwadraten van `w`; de lengte van `w` is de wortel daarvan: `np.linalg.norm(w)`."),
            Mistake("no-b", wrong=_m(sides="y * (X @ w)"),
                    en="`{call}` gives {got}, and it should give {want}. Without `+ b` you measure from a line through the middle, not from this one: `y * (X @ w + b)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Zonder `+ b` meet je vanaf een lijn door het midden, niet vanaf deze: `y * (X @ w + b)`."),
            Mistake("no-y", wrong=_m(sides="X @ w + b"),
                    en="`{call}` gives {got}, a negative number, while every example lies on its own side. `X @ w + b` is negative on the side of y = −1; multiply by `y`, so that every example on its own side counts as positive.",
                    nl="`{call}` geeft {got}, een negatief getal, terwijl elk voorbeeld aan zijn eigen kant ligt. `X @ w + b` is negatief aan de kant van y = −1; vermenigvuldig met `y`, zodat elk voorbeeld aan zijn eigen kant positief telt."),
            Mistake("abs", wrong=_m(sides="np.abs(X @ w + b)"),
                    en="`{call}` gives {got}, and it should give {want}: one example lies on the wrong side of this line, so the margin is negative. `np.abs` makes every distance positive and hides that. Multiply by `y` instead.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: één voorbeeld ligt aan de verkeerde kant van deze lijn, dus de marge is negatief. `np.abs` maakt elke afstand positief en verbergt dat. Vermenigvuldig in plaats daarvan met `y`."),
            Mistake("largest", wrong=_m(back="np.max(sides) / np.linalg.norm(w)"),
                    en="`{call}` gives {got}: the room of the example farthest from the line. The margin is the room of the nearest: `np.min`.",
                    nl="`{call}` geeft {got}: de ruimte van het voorbeeld dat het verst van de lijn ligt. De marge is de ruimte van het dichtstbijzijnde: `np.min`."),
            Mistake("mean", wrong=_m(back="np.mean(sides) / np.linalg.norm(w)"),
                    en="`{call}` gives {got}: the mean room of all the examples. The margin is the room of the nearest: `np.min`.",
                    nl="`{call}` geeft {got}: de gemiddelde ruimte van alle voorbeelden. De marge is de ruimte van het dichtstbijzijnde: `np.min`."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the two `...`: in `sides` how far each example lies on its own side, and after `return` the smallest of them as a distance.",
                      "nl": "Vervang de twee `...`: in `sides` hoe ver elk voorbeeld aan zijn eigen kant ligt, en na `return` de kleinste daarvan als afstand."},
            "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Keep `(w, b, X, y)`.",
                          "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(w, b, X, y)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Do you take, for every example, `y` times `X @ w + b`, the smallest of those, and divide by the length of `w`?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Neem je voor elk voorbeeld `y` keer `X @ w + b`, daarvan de kleinste, en deel je door de lengte van `w`?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for every line, also one with an example on the wrong side, and for three measurements?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke lijn, ook een met een voorbeeld aan de verkeerde kant, en voor drie metingen?"},
            "correct": {"en": "Your function is right in all {n} cases. With `3 * w` and `3 * b` the line stays where it is, and so does your margin, 0.778: dividing by the length of `w` makes it a real distance.",
                        "nl": "Je functie klopt in alle {n} gevallen. Met `3 * w` en `3 * b` blijft de lijn waar hij is, en je marge ook, 0,778: door te delen door de lengte van `w` is het een echte afstand."},
        },
    ),
    "slacks": FunctionBox(
        name="slacks",
        picture=_slacks_picture,
        function="slacks",
        start=_S_START,
        solution=_S,
        show=_shown,
        visible=[(_w(0.5, 0.5), 0.0, _X, _Y)],
        hidden=[(_w(0, 2), 0.0, _X, _Y), (_w(0.8, 0.4), -0.3, _X, _Y), (_w(1, 1), 0.0, _X, _Y), (_w(1, -0.5, 2), 0.3, _X3, _Y3)],
        first=[_DOTS],
        mistakes=[
            Mistake("negative", wrong="import numpy as np\n\n\ndef slacks(w, b, X, y):\n    s = 1 - y * (X @ w + b)\n    return s\n",
                    en="`{call}` gives {got}: some slacks are negative. An example outside the street, on its own side, costs nothing: its slack is 0. Set the values below 0 to 0.",
                    nl="`{call}` geeft {got}: sommige slacks zijn negatief. Een voorbeeld buiten de straat, aan zijn eigen kant, kost niets: zijn slack is 0. Zet de waarden onder 0 op 0."),
            Mistake("divided", wrong=_s("(1 - y * (X @ w + b)) / np.linalg.norm(w)"),
                    en="`{call}` gives {got}, and it should give {want}. The slack is counted in margins, not as a distance: an example on the line has slack 1, whatever the length of `w`. Leave out `/ np.linalg.norm(w)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. De slack wordt geteld in marges, niet als afstand: een voorbeeld op de lijn heeft slack 1, wat de lengte van `w` ook is. Laat `/ np.linalg.norm(w)` weg."),
            Mistake("flipped", wrong=_s("y * (X @ w + b) - 1"),
                    en="`{call}` gives {got}, and it should give {want}: your slacks are high for the examples far outside the street. The slack is `1 - y * (X @ w + b)`: how far an example lies short of its edge of the street.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je slacks zijn hoog voor de voorbeelden ver buiten de straat. De slack is `1 - y * (X @ w + b)`: hoe ver een voorbeeld van zijn rand van de straat af blijft."),
            Mistake("no-one", wrong=_s("-y * (X @ w + b)"),
                    en="`{call}` gives {got}, and it should give {want}. Without the `1 -`, only an example on the wrong side of the line gets a slack. An example inside the street costs something too: start from 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Zonder de `1 -` krijgt alleen een voorbeeld aan de verkeerde kant van de lijn een slack. Een voorbeeld in de straat kost ook iets: begin bij 1."),
            Mistake("no-y", wrong=_s("1 - (X @ w + b)"),
                    en="`{call}` gives {got}, and it should give {want}. Without `y`, the examples of y = −1 get a slack as soon as they lie on their own side. Multiply `X @ w + b` by `y`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Zonder `y` krijgen de voorbeelden met y = −1 een slack zodra ze aan hun eigen kant liggen. Vermenigvuldig `X @ w + b` met `y`."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the two `...`: in `s` how far each example lies short of its edge of the street, and in `to_zero` the mask of the values below 0.",
                      "nl": "Vervang de twee `...`: in `s` hoe ver elk voorbeeld van zijn rand van de straat af blijft, en in `to_zero` het mask van de waarden onder 0."},
            "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Keep `(w, b, X, y)`.",
                          "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(w, b, X, y)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is the slack of each example `1 - y * (X @ w + b)`, with every value below 0 set to 0?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Is de slack van elk voorbeeld `1 - y * (X @ w + b)`, met elke waarde onder 0 op 0 gezet?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for every line, also one with an example on the wrong side, and for three measurements?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke lijn, ook een met een voorbeeld aan de verkeerde kant, en voor drie metingen?"},
            "correct": {"en": "Your function is right in all {n} cases. With `np.array([0.0, 2.0])` the third Chinstrap lies on the wrong side, with a slack of 1.2; with `np.array([1.0, 1.0])` every example lies outside the street, and every slack is 0.",
                        "nl": "Je functie klopt in alle {n} gevallen. Met `np.array([0.0, 2.0])` ligt de derde Chinstrap aan de verkeerde kant, met een slack van 1,2; met `np.array([1.0, 1.0])` ligt elk voorbeeld buiten de straat, en is elke slack 0."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("margin", _M_START, "empty", None),
    ("margin", _m(back="..."), "wrong", "dots-left"),
    ("margin", _m(back="np.min(sides)"), "wrong", "no-norm"),
    ("margin", _m(back="np.min(sides) / np.sum(w ** 2)"), "wrong", "squared"),
    ("margin", _m(sides="X @ w + b"), "wrong", "no-y"),
    ("margin", _m(sides="y * (X @ w)"), "wrong", "no-b"),
    ("margin", _m(sides="np.abs(X @ w + b)"), "wrong", "abs"),
    ("margin", _m(back="np.max(sides) / np.linalg.norm(w)"), "wrong", "largest"),
    ("margin", _m(back="np.mean(sides) / np.linalg.norm(w)"), "wrong", "mean"),
    ("margin", _M, "correct", None),
    ("margin", _m(back="min(sides) / np.sqrt(np.sum(w ** 2))"), "correct", None),
    ("margin", "import numpy as np\n\n\ndef margin(w, b, X, y):\n    return np.min(y * (np.dot(X, w) + b) / np.linalg.norm(w))\n", "correct", None),
    ("slacks", _S_START, "empty", None),
    ("slacks", _s(mask="..."), "wrong", "dots-left"),
    ("slacks", _s("(1 - y * (X @ w + b)) / np.linalg.norm(w)"), "wrong", "divided"),
    ("slacks", "import numpy as np\n\n\ndef slacks(w, b, X, y):\n    s = 1 - y * (X @ w + b)\n    return s\n", "wrong", "negative"),
    ("slacks", _s("y * (X @ w + b) - 1"), "wrong", "flipped"),
    ("slacks", _s("-y * (X @ w + b)"), "wrong", "no-one"),
    ("slacks", _s("1 - (X @ w + b)"), "wrong", "no-y"),
    ("slacks", _S, "correct", None),
    ("slacks", "import numpy as np\n\n\ndef slacks(w, b, X, y):\n    s = 1 - y * (X @ w + b)\n    return s * (s > 0)\n", "correct", None),
]
