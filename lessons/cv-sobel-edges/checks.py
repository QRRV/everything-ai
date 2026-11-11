# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-sobel-edges: the edge strength of an image with two Sobel filters, and the threshold that
marks the edges of the grainy photo, smoothed first, closest to the edges of the clean photo. And the pictures: the
photo of the coffee cup next to its two Sobel results and its edge strength; and for the slider, the edges of the
smoothed grainy photo at one threshold next to how many pixels are marked wrong at every threshold."""
import functools
import re

import numpy as np

from everything_ai.check import FunctionBox, Mistake, Value
from everything_ai import yours

COFFEE = "data/photos/coffee.png"       # 600 by 400 pixels, RGB; from the sample data of scikit-image
PART = (slice(0, 300), slice(100, 500))  # the part with the cup, as in cv-denoise-filters
SEED, SIGMA = 0, 25                     # the grain of cv-denoise-filters: seed 0, standard deviation 25
GAUSS = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])     # how fast the brightness grows to the right
KY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])     # how fast it grows downward
EDGE = 150                              # the edges of the clean photo: an edge strength above 150
THRESHOLDS = list(range(0, 401, 25))    # the stops of the slider
TOP = 12                                # the top of the axis of the slider, in percent


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
    """The two Sobel results, gx to the right and gy downward, only where the whole window fits."""
    w = _windows(img)
    return (w * KX).sum(axis=(2, 3)), (w * KY).sum(axis=(2, 3))


def strength_of(img) -> np.ndarray:
    """The edge strength: the length of the image gradient (gx, gy) of every pixel."""
    gx, gy = sobel_of(img)
    return np.hypot(gx, gy)


def make_grain(clean, seed=SEED, sigma=SIGMA) -> np.ndarray:
    """The grain of the worked example: a random number with mean 0 and standard deviation 25 added to every gray."""
    rng = np.random.default_rng(seed)
    return clean + rng.normal(0, sigma, size=clean.shape)


@functools.lru_cache(maxsize=1)
def _photos() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean = read(COFFEE).max(axis=2)[PART]
    grainy = make_grain(clean)
    smoothed = gauss_of(grainy)
    for a in (clean, grainy, smoothed):
        a.flags.writeable = False
    return clean, grainy, smoothed


def clean_photo() -> np.ndarray:
    """The photo of the lesson: the part of the coffee photo with the cup, in grays."""
    return _photos()[0]


def grainy_photo() -> np.ndarray:
    return _photos()[1]


def smoothed_photo() -> np.ndarray:
    """The grainy photo after the Gaussian filter: 2 smaller in each direction."""
    return _photos()[2]


@functools.lru_cache(maxsize=1)
def clean_edges() -> np.ndarray:
    """The answer key: the pixels of the clean photo with an edge strength above 150."""
    edges = strength_of(clean_photo()) > EDGE
    edges.flags.writeable = False
    return edges


def wrong(edges) -> float:
    """The share of the pixels, in percent, where an edge map differs from the edges of the clean photo: an edge where
    the clean photo has none, or none where it has one. A map that is smaller is compared with the middle part."""
    key = clean_edges()
    b = (key.shape[0] - edges.shape[0]) // 2
    return float(100 * (edges != key[b:key.shape[0] - b, b:key.shape[1] - b]).mean())


@functools.lru_cache(maxsize=1)
def _curves() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """For every stop of the slider: how many pixels are wrong for the smoothed grainy photo, and for the grainy photo
    without smoothing; worked out once, because the slider draws a picture for every stop."""
    smooth, raw = strength_of(smoothed_photo()), strength_of(grainy_photo())
    return tuple(wrong(smooth > t) for t in THRESHOLDS), tuple(wrong(raw > t) for t in THRESHOLDS)


def best_threshold(smoothed: bool = True) -> int:
    """The threshold with the fewest wrong pixels: the answer of exercise 3 (and, without smoothing, the one its
    message names)."""
    curve = _curves()[0 if smoothed else 1]
    return THRESHOLDS[int(np.argmin(curve))]


def wrong_at(t, smoothed: bool = True) -> float:
    """What the slider leads to: the share of wrong pixels at threshold t, in percent, to one decimal."""
    return round(_curves()[0 if smoothed else 1][THRESHOLDS.index(int(t))], 1)


def _photo_axes(ax, img, title, cmap="gray", vmin=0, vmax=255):
    ax.imshow(np.clip(img, vmin, vmax), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title, fontsize=10)


def edges_picture(labels: dict[str, str]) -> str:
    """The photo of the cup, its two Sobel results with 0 as middle gray, and its edge strength: dark where it is
    strong, like ink on paper."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    clean = clean_photo()
    gx, gy = sobel_of(clean)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), dpi=110, gridspec_kw={"hspace": 0.12, "wspace": 0.04})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
    _photo_axes(axes[0, 0], clean, labels["photo"])
    _photo_axes(axes[0, 1], gx, labels["gx"], vmin=-400, vmax=400)
    _photo_axes(axes[1, 0], gy, labels["gy"], vmin=-400, vmax=400)
    _photo_axes(axes[1, 1], np.hypot(gx, gy), labels["strength"], cmap="gray_r", vmin=0, vmax=400)
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


def threshold_frame(t: int, labels: dict[str, str]) -> tuple[str, dict]:
    """For the slider: the edges of the smoothed grainy photo at threshold t, and next to it how many pixels are wrong
    at every threshold, with t marked. With what the sentence under it says."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    t = int(t)
    curve = _curves()[0]
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.8, 2.9), gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.3})
    _photo_axes(left, strength_of(smoothed_photo()) > t, labels["edges"].format(t=t), cmap="gray_r", vmin=0, vmax=1)
    # the axis stops at 12%, so that the low point shows; a value above it is a triangle at the top
    right.plot(THRESHOLDS, curve, color=plots.LOSS, linewidth=1.8, linestyle="-")   # plots.style() cycles line styles
    inside = [(x, v) for x, v in zip(THRESHOLDS, curve) if v <= TOP]
    right.plot([x for x, _ in inside], [v for _, v in inside], color=plots.LOSS, marker="o", markersize=3.5, linestyle="none")
    above = [x for x, v in zip(THRESHOLDS, curve) if v > TOP]
    right.plot(above, [TOP] * len(above), color=plots.LOSS, marker="^", markersize=6.5, linestyle="none", clip_on=False)
    here = curve[THRESHOLDS.index(t)]
    right.plot([t, t], [0, min(here, TOP)], color=plots.INK_3, linestyle=":", linewidth=1.2)
    right.plot([t], [min(here, TOP)], marker="o" if here <= TOP else "^", color=plots.HYPER, markersize=8, clip_on=False)
    right.set_xlim(-15, 415)
    right.set_ylim(0, TOP)
    right.set_xticks([0, 100, 200, 300, 400])
    right.set_xlabel(labels["threshold"])
    right.set_title(labels["wrong"], fontsize=10, loc="left")
    html = plots.render(fig, alt=labels["alt"].format(t=t))
    plt.close(fig)
    return html, {"wrong": wrong_at(t)}


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


# The lesson is done when these are correct. "direction" is a choice question and "smooth" an open question in the
# lesson.
REQUIRED = ["direction", "strength", "threshold", "smooth"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME = {"en": "`{call}` gives `{got}`, and it should give `{want}`.", "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn."}


def _m(key, en, nl, **kw):
    """A mistake whose message starts with what the call gives and what it should give."""
    return Mistake(key, en=_SAME["en"] + " " + en, nl=_SAME["nl"] + " " + nl, **kw)


_HEAD = ("import numpy as np\n\nKX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])\n"
         "KY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])\n\n\n"
         "def edge_strength(img):\n    img = np.array(img, dtype=float)\n"
         "    out = np.zeros((len(img) - 2, len(img[0]) - 2))\n"
         "    for r in range(len(out)):\n        for c in range(len(out[0])):\n"
         "            window = img[r:r + 3, c:c + 3]\n")
_TAIL = "    return out\n"
_GXGY = "            gx = (window * KX).sum()\n            gy = (window * KY).sum()\n"
_START = _HEAD + "            ...\n" + _TAIL
_SOLUTION = _HEAD + _GXGY + "            out[r, c] = np.linalg.norm([gx, gy])\n" + _TAIL


def _with(last: str) -> str:
    """The course's code with another last line in the loop."""
    return _HEAD + _GXGY + f"            out[r, c] = {last}\n" + _TAIL


# The example of exercise 2: a ramp that grows 3 grays per pixel to the right and 4 downward.
RAMP = [[0, 3, 6, 9], [4, 7, 10, 13], [8, 11, 14, 17], [12, 15, 18, 21]]

_SUM_EN = "You add `gx` and `gy`. The length of a vector is the square root of the sum of the squares, as in [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en): `np.linalg.norm([gx, gy])`."
_SUM_NL = "Je telt `gx` en `gy` op. De lengte van een vector is de wortel uit de som van de kwadraten, zoals in [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl): `np.linalg.norm([gx, gy])`."

EXERCISES = {
    "strength": FunctionBox(
        name="strength",
        picture=yours.arrays(call=0, shade=True),
        function="edge_strength",
        start=_START,
        solution=_SOLUTION,
        visible=[(RAMP,)],
        hidden=[([[0, 0, 0, 9, 9], [0, 0, 0, 9, 9], [0, 0, 0, 9, 9], [0, 0, 0, 9, 9]],),
                ([[0, 0, 0, 0], [0, 0, 0, 0], [9, 9, 9, 9], [9, 9, 9, 9]],),
                ([[0, 0, 0], [0, 0, 255], [0, 255, 255]],),
                ([[10, 10, 10], [10, 100, 10], [10, 10, 10]],),
                (np.array([[200, 200, 10, 10], [200, 200, 10, 10], [200, 200, 10, 10]], dtype=np.uint8),),
                ([[3, 1, 4, 1, 5], [9, 2, 6, 5, 3], [5, 8, 9, 7, 9], [3, 2, 3, 8, 4], [6, 2, 6, 4, 3]],)],
        show=_listed,
        compare=_same_image,
        first=[_DOTS],
        mistakes=[
            Mistake("many", error=lambda e: isinstance(e, ValueError) and "sequence" in str(e),
                    en="With `{call}`, your function stops with an error: `out[r, c]` is one place, and it gets a whole grid of numbers. `window * KX` is still nine numbers: `.sum()` adds them up to one, `gx`. Store one number in `out[r, c]`: the length of `(gx, gy)`.",
                    nl="Met `{call}` stopt je functie met een fout: `out[r, c]` is één plek, en die krijgt een heel rooster met getallen. `window * KX` is nog negen getallen: `.sum()` telt ze op tot één, `gx`. Sla één getal op in `out[r, c]`: de lengte van `(gx, gy)`."),
            Mistake("zeros", matches=_never_filled,
                    en="`{call}` gives only zeros: `out` is still the array of zeros it started as. Store the length of `(gx, gy)` in it: `out[r, c] = ...`.",
                    nl="`{call}` geeft alleen nullen: `out` is nog steeds de array met nullen van het begin. Sla de lengte van `(gx, gy)` erin op: `out[r, c] = ...`."),
            _m("only-x", "That is only `gx`, how fast the brightness grows to the right. The ramp also grows 4 grays per pixel downward: the edge strength is the length of `(gx, gy)` together.",
               "Dat is alleen `gx`, hoe snel de helderheid naar rechts groeit. Het verloop loopt ook 4 grijstinten per pixel op naar beneden: de randsterkte is de lengte van `(gx, gy)` samen.",
               wrong=_with("gx")),
            _m("only-y", "That is only `gy`, how fast the brightness grows downward. The ramp also grows 3 grays per pixel to the right: the edge strength is the length of `(gx, gy)` together.",
               "Dat is alleen `gy`, hoe snel de helderheid naar beneden groeit. Het verloop loopt ook 3 grijstinten per pixel op naar rechts: de randsterkte is de lengte van `(gx, gy)` samen.",
               wrong=_with("gy")),
            _m("sum", _SUM_EN, _SUM_NL,
               wrong=_with("abs(gx) + abs(gy)")),
            _m("plain-sum", _SUM_EN, _SUM_NL, wrong=_with("gx + gy")),
            _m("no-sum", "`window * KX` is still nine numbers, so `np.linalg.norm` takes the length of all eighteen products together. Add up the nine first with `.sum()`: `gx = (window * KX).sum()`, and the same for `gy`.",
               "`window * KX` is nog negen getallen, dus `np.linalg.norm` neemt de lengte van alle achttien producten samen. Tel eerst de negen op met `.sum()`: `gx = (window * KX).sum()`, en hetzelfde voor `gy`.",
               wrong=_HEAD + "            gx = window * KX\n            gy = window * KY\n            out[r, c] = np.linalg.norm([gx, gy])\n" + _TAIL),
            Mistake("norm-args", error=lambda e: isinstance(e, ValueError) and "dimensions to norm" in str(e),
                    en="With `{call}`, your function stops with an error: `np.linalg.norm` takes one vector. Put `gx` and `gy` together in a list: `np.linalg.norm([gx, gy])`, with square brackets.",
                    nl="Met `{call}` stopt je functie met een fout: `np.linalg.norm` neemt één vector. Zet `gx` en `gy` samen in een lijst: `np.linalg.norm([gx, gy])`, met vierkante haken."),
            _m("no-root", "That is the sum of the squares, `gx ** 2 + gy ** 2`: the square of the length. Take its square root, with `** 0.5`, or use `np.linalg.norm([gx, gy])`.",
               "Dat is de som van de kwadraten, `gx ** 2 + gy ** 2`: het kwadraat van de lengte. Neem de wortel ervan, met `** 0.5`, of gebruik `np.linalg.norm([gx, gy])`.",
               wrong=_with("gx ** 2 + gy ** 2")),
            _m("matmul", "`window @ KX` is a matrix multiplication. A filter multiplies place by place, `window * KX`, and adds up the nine products with `.sum()`.",
               "`window @ KX` is een matrixvermenigvuldiging. Een filter vermenigvuldigt plek voor plek, `window * KX`, en telt de negen producten op met `.sum()`.",
               wrong=_HEAD + "            gx = (window @ KX).sum()\n            gy = (window @ KY).sum()\n"
                     "            out[r, c] = np.linalg.norm([gx, gy])\n" + _TAIL),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with what happens to each window: `gx` with `KX`, `gy` with `KY`, and the length of `(gx, gy)` stored in `out[r, c]`.",
                      "nl": "Vervang de `...` door wat er met elk venster gebeurt: `gx` met `KX`, `gy` met `KY`, en de lengte van `(gx, gy)` opgeslagen in `out[r, c]`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with one image, and the `def` line of your function has a different number of names between its parentheses. Keep `(img)`.",
                          "nl": "De check roept `{call}` aan, met één afbeelding, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(img)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Every place `out[r, c]` gets the length of `(gx, gy)`: `gx` is `window` times `KX`, added up, and `gy` the same with `KY`.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Elke plek `out[r, c]` krijgt de lengte van `(gx, gy)`: `gx` is `window` keer `KX`, opgeteld, en `gy` hetzelfde met `KY`."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does every place get the length of its own `(gx, gy)`, whatever the image?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Krijgt elke plek de lengte van zijn eigen `(gx, gy)`, welke afbeelding het ook is?"},
            "correct": {"en": "Your function is right in all {n} cases. On the ramp, `gx` is 24 and `gy` is 32 everywhere: 8 times the slope of 3 and of 4 grays per pixel. The length of `(24, 32)` is 40, as the length of `(3, 4)` is 5.",
                        "nl": "Je functie klopt in alle {n} gevallen. Op het verloop is `gx` overal 24 en `gy` 32: 8 keer de helling van 3 en van 4 grijstinten per pixel. `(24, 32)` heeft lengte 40, net zoals `(3, 4)` lengte 5 heeft."},
        },
    ),
    "threshold": Value(
        name="threshold",
        start=0,
        solution=175,          # best_threshold(); the test of the lesson measures it again
        outcome=wrong_at,
        prose_numbers=True,
        mistakes=[
            Mistake("low", matches=lambda got, t: int(t) < best_threshold(),
                    en="With a threshold of {value}, {got}% of the pixels are wrong. Many of the edges are grain that the Gaussian filter left: move the slider further right.",
                    nl="Met een drempel van {value} is {got}% van de pixels fout. Veel van de randen zijn korrel die het Gauss-filter liet staan: schuif verder naar rechts."),
            Mistake("high", matches=lambda got, t: int(t) > best_threshold(),
                    en="With a threshold of {value}, {got}% of the pixels are wrong. A threshold this high loses more faint real edges than it removes grain: move the slider back.",
                    nl="Met een drempel van {value} is {got}% van de pixels fout. Bij zo'n hoge drempel verdwijnen er meer zwakke echte randen dan er korrel wegvalt: schuif terug."),
        ],
        texts={
            "missing": {"en": "I cannot see the slider. Run the cell with the slider again.",
                        "nl": "Ik zie de slider niet. Voer de cel met de slider opnieuw uit."},
            "empty": {"en": "The slider is still at 0, where it started: almost every pixel has an edge strength above 0, so almost everything counts as an edge, and 94.0% of the pixels are wrong. Move it to the right.",
                      "nl": "De slider staat nog op 0, waar hij begon: bijna elke pixel heeft een randsterkte boven 0, dus bijna alles telt als rand, en 94,0% van de pixels is fout. Schuif hem naar rechts."},
            "wrong": {"en": "With a threshold of {value}, {got}% of the pixels are wrong. Look for the threshold where that is smallest.",
                      "nl": "Met een drempel van {value} is {got}% van de pixels fout. Zoek de drempel waar dat het kleinst is."},
            "correct": {"en": "A threshold of {value} is right: {got}% of the pixels are wrong, the fewest of all thresholds. Without the Gaussian filter first, the best threshold is 300, and still 4.1% of the pixels are wrong.",
                        "nl": "Een drempel van {value} klopt: {got}% van de pixels is fout, het minst van alle drempels. Zonder het Gauss-filter eerst is de beste drempel 300, en is nog 4,1% van de pixels fout."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("strength", _START, "empty", None),
    ("strength", _SOLUTION, "correct", None),
    ("strength", _with("(gx ** 2 + gy ** 2) ** 0.5"), "correct", None),
    ("strength", _with("np.sqrt(gx ** 2 + gy ** 2)"), "correct", None),
    ("strength", _with("np.hypot(gx, gy)"), "correct", None),
    ("strength", _HEAD + "            out[r, c] = np.linalg.norm([(window * KX).sum(), (window * KY).sum()])\n" + _TAIL, "correct", None),
    ("strength", _with("gx"), "wrong", "only-x"),
    ("strength", _with("gy"), "wrong", "only-y"),
    ("strength", _with("abs(gx) + abs(gy)"), "wrong", "sum"),
    ("strength", _with("gx ** 2 + gy ** 2"), "wrong", "no-root"),
    ("strength", _HEAD + "            gx = (window @ KX).sum()\n            gy = (window @ KY).sum()\n"
     "            out[r, c] = np.linalg.norm([gx, gy])\n" + _TAIL, "wrong", "matmul"),
    ("strength", _HEAD + "            out[r, c] = window * KX\n" + _TAIL, "error", "many"),
    ("strength", _HEAD + _GXGY + "            np.linalg.norm([gx, gy])\n" + _TAIL, "wrong", "zeros"),
    ("strength", _with("gx + gy"), "wrong", "plain-sum"),
    ("strength", _HEAD + "            gx = window * KX\n            gy = window * KY\n            out[r, c] = np.linalg.norm([gx, gy])\n" + _TAIL, "wrong", "no-sum"),
    ("strength", _with("np.linalg.norm(gx, gy)"), "error", "norm-args"),
    ("strength", _with("gx - gy"), "wrong", None),
    ("strength", _with("np.linalg.norm([gx, gy]) // 1"), "almost", None),
    ("strength", _HEAD + "            ...\n            out[r, c] = 0\n" + _TAIL, "wrong", "dots-left"),
    ("threshold", 0, "empty", None),
    ("threshold", 175, "correct", None),
    ("threshold", 150, "wrong", "low"),
    ("threshold", 25, "wrong", "low"),
    ("threshold", 200, "wrong", "high"),
    ("threshold", 400, "wrong", "high"),
]
