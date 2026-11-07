# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-canny: thinning the edge strength to lines of one pixel, and the low threshold of the
hysteresis threshold that finds the outlines of the fruit in the drawing best. And the pictures: the drawing with its
edge strength, its thinned edge strength, one threshold and two; and for the slider, the edges at one low threshold
next to how many pixels are wrong at every low threshold."""
import functools
import re

import numpy as np

from everything_ai.check import FunctionBox, Mistake, Value

FRUIT = "data/photos/fruit.png"          # the drawing of cv-color-spaces: 360 by 240 pixels, RGB
LABELS = "data/photos/fruit-labels.png"  # what every pixel is: 0 the tablecloth, 1 to 4 a fruit
GAUSS = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
KY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
HIGH, LOW = 200, 100                     # the two thresholds of the worked example
LOWS = list(range(0, HIGH + 1, 10))      # the stops of the slider: the low threshold, with the high one at 200
TOP = 1200                               # the top of the axis of the slider, in pixels


def read(path: str) -> np.ndarray:
    """An image file as an array, as the lesson reads it."""
    from PIL import Image
    return np.array(Image.open(path))


def _windows(img) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(np.asarray(img, dtype=float), (3, 3))


def gauss_of(img) -> np.ndarray:
    """The Gaussian filter of 3 by 3 of cv-denoise-filters, only where the whole window fits."""
    return (_windows(img) * GAUSS).sum(axis=(2, 3))


def sobel_of(img) -> tuple[np.ndarray, np.ndarray]:
    """The two Sobel results of cv-sobel-edges, gx to the right and gy downward, only where the whole window fits."""
    w = _windows(img)
    return (w * KX).sum(axis=(2, 3)), (w * KY).sum(axis=(2, 3))


def step(gx, gy) -> tuple[int, int]:
    """The step to the neighbor across the edge, along the image gradient rounded to one of four directions: (0, 1)
    to the right, (1, 0) down, (1, 1) down to the right and (1, -1) down to the left. The neighbor on the other side
    is the step back. 0.41 is the slope halfway between flat and diagonal: the tangent of 22.5 degrees."""
    if abs(gy) <= 0.41 * abs(gx):
        return 0, 1
    if abs(gx) <= 0.41 * abs(gy):
        return 1, 0
    if gx * gy > 0:
        return 1, 1
    return 1, -1


def thin_of(strength, gx, gy) -> np.ndarray:
    """Edge thinning: every pixel that is at least as strong as its two neighbors across the edge keeps its edge
    strength, every other pixel gets 0; so does the frame of one pixel around it, which has no neighbor on one side."""
    s, gx, gy = (np.asarray(a, dtype=float) for a in (strength, gx, gy))
    out = np.zeros(s.shape)
    if s.shape[0] < 3 or s.shape[1] < 3:
        return out
    ax, ay = np.abs(gx), np.abs(gy)
    flat, steep = ay <= 0.41 * ax, ax <= 0.41 * ay
    dr = np.where(flat, 0, 1)
    dc = np.where(flat, 1, np.where(steep, 0, np.where(gx * gy > 0, 1, -1)))
    r, c = np.mgrid[1:s.shape[0] - 1, 1:s.shape[1] - 1]
    dr, dc = dr[1:-1, 1:-1], dc[1:-1, 1:-1]
    mid = s[1:-1, 1:-1]
    keep = (mid >= s[r + dr, c + dc]) & (mid >= s[r - dr, c - dc])
    out[1:-1, 1:-1] = np.where(keep, mid, 0)
    return out


def hysteresis_of(thin, low, high) -> np.ndarray:
    """The hysteresis threshold: every pixel above `high` is an edge, and so is every pixel above `low` that is
    connected to one of those through pixels above `low`, also diagonally. With a to-do list that grows while a loop
    goes through it, as in the lesson."""
    thin = np.asarray(thin, dtype=float)
    edges = thin > high
    todo = [tuple(p) for p in np.argwhere(edges)]
    rows, cols = thin.shape
    for r, c in todo:
        for rr in range(max(r - 1, 0), min(r + 2, rows)):
            for cc in range(max(c - 1, 0), min(c + 2, cols)):
                if thin[rr, cc] > low and not edges[rr, cc]:
                    edges[rr, cc] = True
                    todo.append((rr, cc))
    return edges


@functools.lru_cache(maxsize=1)
def _steps() -> tuple[np.ndarray, ...]:
    gray = read(FRUIT).mean(axis=2)
    smoothed = gauss_of(gray)
    gx, gy = sobel_of(smoothed)
    strength = np.hypot(gx, gy)
    thin = thin_of(strength, gx, gy)
    for a in (gray, smoothed, gx, gy, strength, thin):
        a.flags.writeable = False
    return gray, smoothed, gx, gy, strength, thin


def gray_drawing() -> np.ndarray:
    """The drawing in grays: the mean of red, green and blue of every pixel."""
    return _steps()[0]


def gradient() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """gx, gy and the edge strength of the smoothed drawing: 4 smaller than the drawing in each direction."""
    return _steps()[2:5]


def thinned() -> np.ndarray:
    return _steps()[5]


def _near(mask) -> np.ndarray:
    """True where the window of 3 by 3 around a pixel holds a True of `mask`."""
    m = np.asarray(mask, dtype=bool)
    p = np.pad(m, 1)
    out = np.zeros_like(m)
    for i in range(3):
        for j in range(3):
            out |= p[i:i + m.shape[0], j:j + m.shape[1]]
    return out


@functools.lru_cache(maxsize=1)
def outlines() -> np.ndarray:
    """The answer key: the pixels where one thing of the drawing meets another, a fruit and the tablecloth or two
    fruits, from the labels; cut to the place of the edge strength (2 pixels off each side)."""
    lab = read(LABELS).astype(int)[2:-2, 2:-2]
    key = np.zeros(lab.shape, dtype=bool)
    down = lab[:-1] != lab[1:]
    key[:-1] |= down
    key[1:] |= down
    right = lab[:, :-1] != lab[:, 1:]
    key[:, :-1] |= right
    key[:, 1:] |= right
    key.flags.writeable = False
    return key


def counts(edges) -> tuple[int, int]:
    """How an edge map compares with the outlines: how many edge pixels have no outline pixel in their window of 3 by
    3 (too many), and how many outline pixels have no edge pixel in theirs (missed)."""
    e = np.asarray(edges, dtype=bool)
    key = outlines()
    return int((e & ~_near(key)).sum()), int((key & ~_near(e)).sum())


def wrong(edges) -> int:
    too_many, missed = counts(edges)
    return too_many + missed


@functools.lru_cache(maxsize=1)
def _curve() -> tuple[tuple[int, int], ...]:
    """For every stop of the slider: too many and missed, with the high threshold at 200; worked out once, because
    the slider draws a picture for every stop."""
    thin = thinned()
    return tuple(counts(hysteresis_of(thin, low, HIGH)) for low in LOWS)


def counts_at(low) -> tuple[int, int]:
    return _curve()[LOWS.index(int(low))]


def wrong_at(low) -> int:
    """What the slider leads to: how many pixels are wrong at low threshold `low`."""
    return sum(counts_at(low))


def best_low() -> int:
    """The low threshold with the fewest wrong pixels: the answer of exercise 3."""
    return LOWS[int(np.argmin([sum(c) for c in _curve()]))]


def single(t) -> int:
    """How many pixels are wrong with one threshold t on the thinned edge strength."""
    return wrong(thinned() > t)


def _photo_axes(ax, img, title, cmap="gray_r", vmin=0, vmax=400):
    if np.ndim(img) == 3:
        ax.imshow(img, interpolation="nearest")
    else:
        ax.imshow(np.clip(np.asarray(img, dtype=float), vmin, vmax), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title, fontsize=10)


def canny_picture(labels: dict[str, str]) -> str:
    """Six pictures: the drawing; its edge strength; the edge strength after thinning, dark where it is strong, like
    ink on paper; one threshold of 200; one threshold of 100; and the hysteresis threshold with 100 and 200."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    thin = thinned()
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.4), dpi=110, gridspec_kw={"hspace": 0.14, "wspace": 0.04})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01)
    _photo_axes(axes[0, 0], read(FRUIT), labels["drawing"])
    _photo_axes(axes[0, 1], gradient()[2], labels["strength"])
    _photo_axes(axes[1, 0], thin, labels["thin"])
    _photo_axes(axes[1, 1], thin > HIGH, labels["high"], vmax=1)
    _photo_axes(axes[2, 0], thin > LOW, labels["low"], vmax=1)
    _photo_axes(axes[2, 1], hysteresis_of(thin, LOW, HIGH), labels["both"], vmax=1)
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


def low_frame(low: int, labels: dict[str, str]) -> tuple[str, dict]:
    """For the slider: the edges with the hysteresis threshold at `low` and 200, and next to it how many pixels are
    wrong at every low threshold, with `low` marked. With what the sentence under it says."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    low = int(low)
    curve = [sum(c) for c in _curve()]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.75, 1.0], "wspace": 0.22})
    _photo_axes(left, hysteresis_of(thinned(), low, HIGH), labels["edges"].format(low=low), vmax=1)
    # the axis stops at 1200, so that the low point shows; a value above it is a triangle at the top
    right.plot(LOWS, curve, color=plots.LOSS, linewidth=1.8)
    inside = [(x, v) for x, v in zip(LOWS, curve) if v <= TOP]
    right.plot([x for x, _ in inside], [v for _, v in inside], color=plots.LOSS, marker="o", markersize=3.5, linestyle="none")
    above = [x for x, v in zip(LOWS, curve) if v > TOP]
    right.plot(above, [TOP] * len(above), color=plots.LOSS, marker="^", markersize=6.5, linestyle="none", clip_on=False)
    here = curve[LOWS.index(low)]
    right.plot([low, low], [0, min(here, TOP)], color=plots.INK_3, linestyle=":", linewidth=1.2)
    right.plot([low], [min(here, TOP)], marker="o" if here <= TOP else "^", color=plots.HYPER, markersize=8, clip_on=False)
    right.set_xlim(-8, 208)
    right.set_ylim(0, TOP)
    right.set_xticks([0, 50, 100, 150, 200])
    right.set_xlabel(labels["low"])
    right.set_title(labels["wrong"], fontsize=10, loc="left")
    html = plots.render(fig, alt=labels["alt"].format(low=low))
    plt.close(fig)
    too_many, missed = counts_at(low)
    return html, {"wrong": too_many + missed, "too_many": too_many, "missed": missed}


def _numbers(got):
    """What the learner's function gave back as an array of floats, or None if it is not numbers."""
    try:
        return np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return None


def _listed(v) -> str:
    """An image the way the lesson writes it: an array as a list of rows, on one line, without needless decimals."""
    a = _numbers(v)
    if a is None or a.ndim != 2:
        return str(v.tolist() if isinstance(v, np.ndarray) else v)
    rows = [[int(x) if float(x).is_integer() else round(float(x), 2) for x in row] for row in a]
    return str(rows)


def _same_image(got, want) -> bool:
    """Right when every pixel has the course's number, up to rounding, in the same shape."""
    a = _numbers(got)
    return a is not None and a.shape == np.shape(want) and bool(np.allclose(a, np.asarray(want, dtype=float), atol=1e-9))


def _never_filled(got, text) -> bool:
    """Only zeros back, and no line that stores something in `out`."""
    a = _numbers(got)
    return (a is not None and a.size > 0 and bool(np.all(a == 0))
            and re.search(r"^[ \t]*out\[", text, re.MULTILINE) is None)


# The lesson is done when these are correct. "neighbors" is a choice question and "two" an open question in the
# lesson.
REQUIRED = ["neighbors", "thin", "low", "two"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME = {"en": "`{call}` gives `{got}`, and it should give `{want}`.", "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn."}


def _m(key, en, nl, **kw):
    """A mistake whose message starts with what the call gives and what it should give."""
    return Mistake(key, en=_SAME["en"] + " " + en, nl=_SAME["nl"] + " " + nl, **kw)


_HEAD = ("import numpy as np\n\n\n"
         "def step(gx, gy):\n"
         "    if abs(gy) <= 0.41 * abs(gx):\n        return 0, 1\n"
         "    if abs(gx) <= 0.41 * abs(gy):\n        return 1, 0\n"
         "    if gx * gy > 0:\n        return 1, 1\n"
         "    return 1, -1\n\n\n"
         "def thin(strength, gx, gy):\n"
         "    strength = np.array(strength, dtype=float)\n"
         "    gx, gy = np.array(gx), np.array(gy)\n"
         "    out = np.zeros(strength.shape)\n"
         "    for r in range(1, len(out) - 1):\n        for c in range(1, len(out[0]) - 1):\n"
         "            dr, dc = step(gx[r, c], gy[r, c])\n")
_TAIL = "    return out\n"
_START = _HEAD + "            ...\n" + _TAIL


def _with(test: str, store: str = "strength[r, c]") -> str:
    """The course's code with another test, and another number stored where it passes."""
    return _HEAD + f"            if {test}:\n                out[r, c] = {store}\n" + _TAIL


_BOTH = "strength[r, c] >= max(strength[r + dr, c + dc], strength[r - dr, c - dc])"
_AND = "strength[r, c] >= strength[r + dr, c + dc] and strength[r, c] >= strength[r - dr, c - dc]"
_SOLUTION = _with(_BOTH)


def _rows(*row, n=3):
    """An image of n rows that are all the same."""
    return [list(row) for _ in range(n)]


# The example of exercise 2: an edge from top to bottom, strongest in the middle column. gx is the edge strength
# itself and gy is 0: the brightness grows to the right.
EDGE = _rows(0, 40, 90, 60, 0)
ZEROS = _rows(0, 0, 0, 0, 0)
_PI = [[3, 1, 4, 1, 5], [9, 2, 6, 5, 3], [5, 8, 9, 7, 9], [3, 2, 3, 8, 4], [6, 2, 6, 4, 3]]
_E = [[2, -7, 1, 8, -2], [8, 1, -8, 2, 8], [-4, 5, 9, -4, 5], [2, -3, 5, 3, -6], [0, 2, -8, 7, 4]]
_PHI = [[1, 6, -1, 8, 0], [-3, 3, 9, 8, -8], [7, 4, -9, 8, 9], [4, -8, 2, 0, 4], [5, -8, 6, 8, 3]]
_DOWN = [[0, 0, 0], [40, 40, 40], [90, 90, 90], [60, 60, 60], [0, 0, 0]]
_ACROSS = {-4: 2, -3: 6, -2: 15, -1: 50, 0: 100, 1: 60, 2: 20, 3: 5, 4: 1}   # a ridge, by the distance to it
_DIAGONAL = [[_ACROSS[c - r] for c in range(5)] for r in range(5)]           # from top left to bottom right
_ANTI = [[_ACROSS[r + c - 4] for c in range(5)] for r in range(5)]           # from top right to bottom left
_TIE = _rows(0, 50, 90, 90, 50, 0)
_TWO = _rows(0, 90, 30, 80, 0)

EXERCISES = {
    "thin": FunctionBox(
        name="thin",
        function="thin",
        start=_START,
        solution=_SOLUTION,
        visible=[(EDGE, EDGE, ZEROS)],
        hidden=[(_DOWN, [[0] * 3] * 5, _DOWN),
                (_DIAGONAL, [[-1] * 5] * 5, [[1] * 5] * 5),
                (_ANTI, [[1] * 5] * 5, [[1] * 5] * 5),
                (_TWO, _TWO, ZEROS),
                (_TIE, _TIE, _rows(0, 0, 0, 0, 0, 0)),
                ([[5, 7, 5], [7, 9, 7], [5, 7, 5]], [[0] * 3] * 3, [[3] * 3] * 3),
                (_PI, _E, _PHI)],
        show=_listed,
        compare=_same_image,
        first=[_DOTS],
        mistakes=[
            Mistake("many", error=lambda e: isinstance(e, ValueError) and "sequence" in str(e),
                    en="With `{call}`, your function stops with an error: `out[r, c]` is one place, and it gets a whole array. Store one number in it: the edge strength of the pixel itself, `strength[r, c]`.",
                    nl="Met `{call}` stopt je functie met een fout: `out[r, c]` is één plek, en die krijgt een hele array. Sla er één getal in op: de randsterkte van de pixel zelf, `strength[r, c]`."),
            Mistake("thin-name", error=lambda e: isinstance(e, TypeError) and "item assignment" in str(e),
                    en="With `{call}`, your function stops with an error: inside the function, `thin` is the function itself, not an array. Here the result is called `out`: store it in `out[r, c]`.",
                    nl="Met `{call}` stopt je functie met een fout: binnen de functie is `thin` de functie zelf, geen array. De uitkomst heet hier `out`: sla hem op in `out[r, c]`."),
            Mistake("zeros", matches=_never_filled,
                    en="`{call}` gives only zeros: `out` is still the array of zeros it started as. Where the pixel is at least as strong as its two neighbors, store its edge strength: `out[r, c] = strength[r, c]`.",
                    nl="`{call}` geeft alleen nullen: `out` is nog steeds de array met nullen van het begin. Waar de pixel minstens zo sterk is als zijn twee buren, sla je zijn randsterkte op: `out[r, c] = strength[r, c]`."),
            _m("copy", "Every pixel keeps its edge strength, so the edge is as wide as before. Keep it only where it is at least as strong as both neighbors across the edge: `strength[r + dr, c + dc]` and `strength[r - dr, c - dc]`.",
               "Elke pixel houdt zijn randsterkte, dus de rand is even breed als eerst. Houd hem alleen waar hij minstens zo sterk is als allebei de buren dwars op de rand: `strength[r + dr, c + dc]` en `strength[r - dr, c - dc]`.",
               wrong=_HEAD + "            out[r, c] = strength[r, c]\n" + _TAIL),
            _m("one-side", "You compare with one neighbor only: the one a step `(dr, dc)` further. The other neighbor across the edge is a step back: `strength[r - dr, c - dc]`. Keep the pixel only when it is at least as strong as both.",
               "Je vergelijkt met maar één buur: die een stap `(dr, dc)` verder. De andere buur dwars op de rand is een stap terug: `strength[r - dr, c - dc]`. Houd de pixel alleen als hij minstens zo sterk is als allebei.",
               wrong=_with("strength[r, c] >= strength[r + dr, c + dc]")),
            _m("one-of", "Your pixel stays when it is at least as strong as one of its two neighbors, and on a slope it always is. It has to be at least as strong as both: as the larger of the two, `max(strength[r + dr, c + dc], strength[r - dr, c - dc])`.",
               "Je pixel blijft staan als hij minstens zo sterk is als één van zijn twee buren, en op een helling is hij dat altijd. Hij moet minstens zo sterk zijn als allebei: als de grootste van de twee, `max(strength[r + dr, c + dc], strength[r - dr, c - dc])`.",
               wrong=_with(_BOTH.replace("max(", "min("))),
            _m("along", "You compare with the neighbors along the edge, and they are part of the same edge, about as strong. `dr` is the step in the row and `dc` in the column: the neighbors across the edge are `strength[r + dr, c + dc]` and `strength[r - dr, c - dc]`.",
               "Je vergelijkt met de buren langs de rand, en die horen bij dezelfde rand, ongeveer even sterk. `dr` is de stap in de rij en `dc` in de kolom: de buren dwars op de rand zijn `strength[r + dr, c + dc]` en `strength[r - dr, c - dc]`.",
               wrong=_with("strength[r, c] >= max(strength[r + dc, c + dr], strength[r - dc, c - dr])")),
            _m("left-right", "You always compare with the neighbors on the left and the right, whichever way the edge runs. Take the step that `step` gives: `strength[r + dr, c + dc]` and `strength[r - dr, c - dc]`.",
               "Je vergelijkt altijd met de buren links en rechts, welke kant de rand ook op loopt. Neem de stap die `step` geeft: `strength[r + dr, c + dc]` en `strength[r - dr, c - dc]`.",
               wrong=_with("strength[r, c] >= max(strength[r, c + 1], strength[r, c - 1])")),
            _m("gx", "You compare `gx`, one part of the image gradient; in the example it happens to be the same as the edge strength. Compare the edge strength: `strength[r, c]` with `strength[r + dr, c + dc]` and `strength[r - dr, c - dc]`.",
               "Je vergelijkt `gx`, één deel van de beeldgradient; in het voorbeeld is dat toevallig hetzelfde als de randsterkte. Vergelijk de randsterkte: `strength[r, c]` met `strength[r + dr, c + dc]` en `strength[r - dr, c - dc]`.",
               matches=lambda got, text: re.search(r"\bgx\[\s*r\s*[-+]\s*dr", text) is not None),
            _m("strict", "Here two neighboring pixels are equally strong, and both are the top of the edge: with `>`, neither of them is larger than the other, and the edge gets a hole. Use `>=`: at least as strong.",
               "Hier zijn twee buurpixels even sterk, en allebei zijn ze de top van de rand: met `>` is geen van beide groter dan de ander, en krijgt de rand een gat. Gebruik `>=`: minstens zo sterk.",
               wrong=_with(_BOTH.replace(">=", ">"))),
            _m("ones", "You store a 1 where the pixel stays. Store its edge strength, `strength[r, c]`: the hysteresis threshold compares that number with its two thresholds.",
               "Je slaat een 1 op waar de pixel blijft. Sla zijn randsterkte op, `strength[r, c]`: de hysteresedrempel vergelijkt dat getal met zijn twee drempels.",
               wrong=_with(_BOTH, "1")),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the test of edge thinning: where `strength[r, c]` is at least as strong as both neighbors across the edge, `out[r, c]` gets it.",
                      "nl": "Vervang de `...` door de voorwaarde van het uitdunnen: waar `strength[r, c]` minstens zo sterk is als allebei de buren dwars op de rand, krijgt `out[r, c]` die waarde."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the edge strength, `gx` and `gy`, and the `def` line of your function has a different number of names between its parentheses. Keep `(strength, gx, gy)`.",
                          "nl": "De check roept `{call}` aan, met de randsterkte, `gx` en `gy`, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(strength, gx, gy)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. A pixel keeps its edge strength only when it is at least as strong as `strength[r + dr, c + dc]` and as `strength[r - dr, c - dc]`; every other place stays 0.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Een pixel houdt zijn randsterkte alleen als hij minstens zo sterk is als `strength[r + dr, c + dc]` en als `strength[r - dr, c - dc]`; elke andere plek blijft 0."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does every pixel compare itself with the two neighbors that `step` points to, whichever way the edge runs?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijkt elke pixel zich met de twee buren waar `step` naar wijst, welke kant de rand ook op loopt?"},
            "correct": {"en": "Your function is right in all {n} cases. In the example only 90 stays: 40 is weaker than its neighbor 90, and 60 too. The edge is one pixel wide now, in the middle of the hill.",
                        "nl": "Je functie klopt in alle {n} gevallen. In het voorbeeld blijft alleen 90 over: 40 is zwakker dan zijn buur 90, en 60 ook. De rand is nu één pixel breed, midden op de heuvel."},
        },
    ),
    "low": Value(
        name="low",
        start=HIGH,
        solution=70,           # best_low(); the test of the lesson measures it again
        outcome=wrong_at,
        prose_numbers=True,
        mistakes=[
            Mistake("low", matches=lambda got, t: int(t) < best_low(),
                    en="With a low threshold of {value}, {got} pixels are wrong. Edges of the checks that touch an outline now count as edges too: move the slider back to the right.",
                    nl="Met een lage drempel van {value} zijn {got} pixels fout. Randen van de ruiten die een omtrek raken, tellen nu ook als rand: schuif terug naar rechts."),
            Mistake("high", matches=lambda got, t: int(t) > best_low(),
                    en="With a low threshold of {value}, {got} pixels are wrong. Faint stretches of outline are still lost: move the slider further to the left.",
                    nl="Met een lage drempel van {value} zijn {got} pixels fout. Er vallen nog zwakke stukken omtrek weg: schuif verder naar links."),
        ],
        texts={
            "missing": {"en": "I cannot see the slider. Run the cell with the slider again.",
                        "nl": "Ik zie de slider niet. Voer de cel met de slider opnieuw uit."},
            "empty": {"en": "The low threshold is still 200, as high as the high one: that is one threshold, and 1119 pixels are wrong, nearly all of them pieces of outline that are missed. Move the slider to the left.",
                      "nl": "De lage drempel staat nog op 200, even hoog als de hoge: dat is één drempel, en 1119 pixels zijn fout, bijna allemaal stukken omtrek die gemist worden. Schuif de slider naar links."},
            "wrong": {"en": "With a low threshold of {value}, {got} pixels are wrong. Look for the low threshold where that is smallest.",
                      "nl": "Met een lage drempel van {value} zijn {got} pixels fout. Zoek de lage drempel waar dat het kleinst is."},
            "correct": {"en": "A low threshold of {value} is right: {got} pixels are wrong, the fewest of all. With one threshold, in steps of 10, the best is 100, and then 307 pixels are wrong.",
                        "nl": "Een lage drempel van {value} klopt: {got} pixels zijn fout, het minst van allemaal. Met één drempel, in stappen van 10, is 100 het best, en dan zijn er 307 pixels fout."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("thin", _START, "empty", None),
    ("thin", _SOLUTION, "correct", None),
    ("thin", _with(_AND), "correct", None),
    ("thin", _HEAD + "            s = strength[r, c]\n            if s >= max(strength[r + dr, c + dc], strength[r - dr, c - dc]):\n"
     "                out[r, c] = s\n" + _TAIL, "correct", None),
    ("thin", _HEAD + "            if strength[r, c] < strength[r + dr, c + dc] or strength[r, c] < strength[r - dr, c - dc]:\n"
     "                continue\n            out[r, c] = strength[r, c]\n" + _TAIL, "correct", None),
    ("thin", _HEAD + "            out[r, c] = strength\n" + _TAIL, "error", "many"),
    ("thin", _HEAD + "            s = strength[r, c]\n            if s >= max(strength[r + dr, c + dc], strength[r - dr, c - dc]):\n"
     "                thin[r, c] = s\n" + _TAIL, "error", "thin-name"),
    ("thin", _with("gx[r, c] >= max(gx[r + dr, c + dc], gx[r - dr, c - dc])"), "wrong", "gx"),
    ("thin", _HEAD + "            strength[r, c] >= strength[r + dr, c + dc]\n" + _TAIL, "wrong", "zeros"),
    ("thin", _HEAD + "            out[r, c] = strength[r, c]\n" + _TAIL, "wrong", "copy"),
    ("thin", _with("strength[r, c] >= strength[r + dr, c + dc]"), "wrong", "one-side"),
    ("thin", _with("strength[r, c] >= strength[r + dr, c + dc] and strength[r, c] >= strength[r + dr, c + dc]"), "wrong", "one-side"),
    ("thin", _with(_AND.replace(" and ", " or ")), "wrong", "one-of"),
    ("thin", _with(_BOTH.replace("max(", "min(")), "wrong", "one-of"),
    ("thin", _with("strength[r, c] >= strength[r + dc, c + dr] and strength[r, c] >= strength[r - dc, c - dr]"), "wrong", "along"),
    ("thin", _with("strength[r, c] >= max(strength[r, c + 1], strength[r, c - 1])"), "wrong", "left-right"),
    ("thin", _with(_AND.replace(">=", ">")), "wrong", "strict"),
    ("thin", _with(_BOTH.replace(">=", ">")), "wrong", "strict"),
    ("thin", _with(_BOTH, "1"), "wrong", "ones"),
    ("thin", _with(_BOTH, "True"), "wrong", "ones"),
    ("thin", _with("strength[r, c] >= strength[r - 1:r + 2, c - 1:c + 2].max()"), "almost", None),
    ("thin", _with(_BOTH, "strength[r, c] // 2"), "wrong", None),
    ("thin", _HEAD + "            ...\n            out[r, c] = 0\n" + _TAIL, "wrong", "dots-left"),
    ("low", HIGH, "empty", None),
    ("low", 70, "correct", None),
    ("low", 60, "wrong", "low"),
    ("low", 0, "wrong", "low"),
    ("low", 80, "wrong", "high"),
    ("low", 150, "wrong", "high"),
]
