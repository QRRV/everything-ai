# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-connected-components: how many shapes a small binary image has when pixels that touch at a
corner count as touching, the bounding box of the shape around one pixel, found with a breadth-first search, the
smallest size that keeps only the four apples of the drawing of fruit, and why a filter by size keeps more of the
tablecloth than an opening. And the pictures: the search from the middle of an apple, step by step, next to the four
shapes with their boxes; and for the slider, the shapes of at least a size next to how many pixels are wrong at every
size."""
import functools
import re
from collections import deque

import numpy as np

from everything_ai.check import FunctionBox, Mistake, Value
from everything_ai import yours

FRUIT = "data/photos/fruit.png"          # the drawing of cv-color-spaces: 360 by 240 pixels, RGB
LABELS = "data/photos/fruit-labels.png"  # what every pixel is: 0 the tablecloth, 1 an apple, 2 to 4 other fruit
APPLE = 1
RED = 0.45                               # the saturation threshold of the rule for red, as in cv-morphology
OPEN = 3                                 # the square of the opening of cv-morphology
CLOSE = 5                                # the square of the closing: the smallest that fills the holes of the shine
SMALLEST = [1, 2, 3, 4, 5, 10, 100, 1000, 2000, 3000]   # the stops of the slider: the smallest size a shape keeps
CROP = (slice(20, 120), slice(20, 120))  # the part of the drawing around the apple at the top left
MIDDLE = (70, 70)                        # the middle of that apple: the start of the search on the slider
STEPS = [0, 1, 2, 5, 10, 15, 20, 25, 30, 33]   # the stops of the slider of the search: 33 is the last step


def read(path: str) -> np.ndarray:
    """An image file as an array, as the lesson reads it."""
    from PIL import Image
    return np.array(Image.open(path))


@functools.lru_cache(maxsize=1)
def red_mask() -> np.ndarray:
    """The rule for red of cv-color-spaces, with 0.45 instead of 0.5, as the worked example makes it."""
    from matplotlib.colors import rgb_to_hsv
    hsv = rgb_to_hsv(read(FRUIT) / 255)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    mask = ((h < 0.05) | (h > 0.95)) & (s > RED)
    mask.flags.writeable = False
    return mask


@functools.lru_cache(maxsize=1)
def apples() -> np.ndarray:
    """Where the course drew an apple."""
    out = read(LABELS) == APPLE
    out.flags.writeable = False
    return out


def _squares(img, k, outside):
    """Every window of k by k, centered on a pixel, as k * k shifted copies of the image; outside the image, the
    value `outside`, which never changes the outcome: True for an erosion, False for a dilation."""
    img = np.asarray(img, dtype=bool)
    h = k // 2
    p = np.full((img.shape[0] + 2 * h, img.shape[1] + 2 * h), outside)
    p[h:h + img.shape[0], h:h + img.shape[1]] = img
    return [p[i:i + img.shape[0], j:j + img.shape[1]] for i in range(k) for j in range(k)]


def erode_of(img, k: int) -> np.ndarray:
    """Erosion with a square of k by k, as in cv-morphology: only the part of the window inside the image counts."""
    return np.logical_and.reduce(_squares(img, k, True))


def dilate_of(img, k: int) -> np.ndarray:
    """Dilation with a square of k by k."""
    return np.logical_or.reduce(_squares(img, k, False))


@functools.lru_cache(maxsize=1)
def opened() -> np.ndarray:
    """The red pixels after the opening of 3 by 3 of cv-morphology."""
    out = dilate_of(erode_of(red_mask(), OPEN), OPEN)
    out.flags.writeable = False
    return out


@functools.lru_cache(maxsize=1)
def cleaned() -> np.ndarray:
    """The opening of 3 by 3, closed with a square of 5 by 5: the end of cv-morphology, as the worked example makes it."""
    out = erode_of(dilate_of(opened(), CLOSE), CLOSE)
    out.flags.writeable = False
    return out


def white_neighbors(img, r, c) -> list[tuple[int, int]]:
    """The white pixels in the window of 3 by 3 around (r, c), only the part inside the image; (r, c) itself too."""
    return [(rr, cc) for rr in range(max(r - 1, 0), min(r + 2, len(img)))
            for cc in range(max(c - 1, 0), min(c + 2, len(img[0]))) if img[rr][cc]]


def shape_from(img, start, seen: set) -> list[tuple[int, int]]:
    """The shape around `start`: a breadth-first search over the white pixels that touch, also at a corner. The
    pixels in the order the search takes them from the queue, as the lesson's `shape_from` gives them."""
    seen.add(start)
    queue = deque([start])
    shape = []
    while queue:
        r, c = queue.popleft()
        shape.append((r, c))
        for nxt in white_neighbors(img, r, c):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return shape


def shapes_of(mask) -> list[list[tuple[int, int]]]:
    """Every shape of a binary image, in the order a scan row by row from the top finds them: the lesson's
    `find_shapes`."""
    mask = np.asarray(mask, dtype=bool)
    seen: set = set()
    out = []
    for r0, c0 in zip(*np.nonzero(mask)):
        start = (int(r0), int(c0))
        if start not in seen:
            out.append(shape_from(mask, start, seen))
    return out


def box(shape) -> tuple[int, int, int, int]:
    """The bounding box of a shape as the lesson writes it: (top, left, bottom, right), the smallest and the largest
    row and column."""
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return min(rows), min(cols), max(rows), max(cols)


def steps_from(mask, start) -> np.ndarray:
    """For every pixel of the shape around `start`: after how many steps the search finds it; -1 elsewhere."""
    mask = np.asarray(mask, dtype=bool)
    steps = np.full(mask.shape, -1)
    steps[start] = 0
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nxt in white_neighbors(mask, r, c):
            if steps[nxt] < 0:
                steps[nxt] = steps[r, c] + 1
                queue.append(nxt)
    return steps


@functools.lru_cache(maxsize=1)
def red_shapes() -> list[list[tuple[int, int]]]:
    """The shapes of the rule for red, before any tidying up."""
    return shapes_of(red_mask())


@functools.lru_cache(maxsize=None)
def kept(n: int) -> np.ndarray:
    """The rule for red with only the shapes of at least n pixels: the filter by size of the slider."""
    out = np.zeros(red_mask().shape, dtype=bool)
    for s in red_shapes():
        if len(s) >= n:
            out[tuple(np.array(s).T)] = True
    out.flags.writeable = False
    return out


def wrong(mask) -> int:
    """How many pixels a mask has wrong: white where the course drew no apple, or black where it drew one."""
    return int((np.asarray(mask, dtype=bool) != apples()).sum())


def cloth(mask) -> int:
    """How many white pixels of a mask lie on the tablecloth, where the course drew no apple."""
    return int((np.asarray(mask, dtype=bool) & ~apples()).sum())


@functools.lru_cache(maxsize=None)
def smallest_outcome(n: int) -> dict:
    """What the slider leads to at a smallest size of n: the shapes left and the pixels wrong."""
    return {"shapes": sum(len(s) >= n for s in red_shapes()), "wrong": wrong(kept(int(n)))}


def _photo_axes(ax, img, title):
    if np.ndim(img) == 3:
        ax.imshow(img, interpolation="nearest")
    else:
        ax.imshow(np.asarray(img, dtype=float), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title, fontsize=10)


def _draw_box(ax, b, color, offset=(0, 0)):
    """A bounding box as a line around its pixels: from the outer edge of the top-left pixel to that of the
    bottom-right one."""
    top, left, bottom, right = b
    top, bottom = top - offset[0] - 0.5, bottom - offset[0] + 0.5
    left, right = left - offset[1] - 0.5, right - offset[1] + 0.5
    ax.plot([left, right, right, left, left], [top, top, bottom, bottom, top], color=color, linestyle="-", linewidth=1.4)


@functools.lru_cache(maxsize=1)
def _search_steps() -> np.ndarray:
    """After how many steps the search from the middle of the apple at the top left finds every pixel of it."""
    out = steps_from(cleaned(), MIDDLE)
    out.flags.writeable = False
    return out


def search_outcome(k: int) -> dict:
    """After k steps of the search from the middle of the apple: the pixels it has found, and how many of them are
    still in the queue. A step is one layer: when the search has taken out every pixel found in the step before,
    the queue holds exactly the pixels of this step."""
    steps = _search_steps()
    k = int(k)
    return {"found": int(((steps >= 0) & (steps <= k)).sum()), "queue": int((steps == k).sum()),
            "apple": int((steps >= 0).sum())}


def search_frame(k: int, labels: dict[str, str]) -> str:
    """For the slider of the search: the apple at the top left after k steps. Light gray what the search has taken out
    of the queue, white what is still in the queue, dark gray what it has not found yet; black is no apple."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    steps = _search_steps()
    k = int(k)
    tones = np.where(steps < 0, 0.0, np.where(steps > k, 0.3, np.where(steps == k, 1.0, 0.68)))
    fig, ax = plt.subplots(figsize=(3.2, 3.3))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
    _photo_axes(ax, tones[CROP], labels["title"].format(k=k))
    html = plots.render(fig, alt=labels["alt"].format(k=k, **search_outcome(k)))
    plt.close(fig)
    return html


def boxes_picture(labels: dict[str, str]) -> str:
    """The drawing with the box of every shape after the tidying up, and its size above the box."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    _photo_axes(ax, read(FRUIT), "")
    for s in shapes_of(cleaned()):
        b = box(s)
        _draw_box(ax, b, plots.PRED)
        ax.text(b[1] - 0.5, b[0] - 3, str(len(s)), color=plots.PRED_TEXT, fontsize=9, ha="left", va="bottom",
                bbox={"facecolor": plots.PAGE, "edgecolor": "none", "pad": 1.5})
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


def smallest_frame(n: int, labels: dict[str, str]) -> tuple[str, dict]:
    """For the slider: the rule for red with only the shapes of at least n pixels, each in its box when there are
    few, and next to it how many pixels are wrong at every size, with n marked. With what the sentence under it
    says."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    n = int(n)
    curve = [smallest_outcome(s)["wrong"] for s in SMALLEST]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.6, 1.0], "wspace": 0.22})
    _photo_axes(left, kept(n), labels["kept_1"] if n == 1 else labels["kept"].format(n=n))
    if smallest_outcome(n)["shapes"] <= 10:
        for s in red_shapes():
            if len(s) >= n:
                _draw_box(left, box(s), plots.PRED)
    # one place per stop of the slider, the stops in order: the sizes between them are not drawn
    at = list(range(len(SMALLEST)))
    right.plot(at, curve, color=plots.LOSS, linewidth=1.8, marker="o", markersize=3.5)
    i = SMALLEST.index(n)
    right.plot([i, i], [0, curve[i]], color=plots.INK_3, linestyle=":", linewidth=1.2)
    right.plot([i], [curve[i]], marker="o", color=plots.HYPER, markersize=8)
    right.set_xlim(-0.6, len(SMALLEST) - 0.4)
    right.set_ylim(0, 3000)
    right.set_xticks(at)
    right.set_xticklabels([str(v) for v in SMALLEST], fontsize=7, rotation=60)
    right.set_xlabel(labels["size"])
    right.set_title(labels["wrong"], fontsize=10, loc="left")
    html = plots.render(fig, alt=labels["alt"].format(n=n))
    plt.close(fig)
    return html, smallest_outcome(n)


def _numbers(got):
    """What the learner's function gave back as an array of floats, or None if it is not numbers."""
    try:
        return np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return None


def _shown_box(v) -> str:
    """A box the way the lesson writes it: four whole numbers between parentheses."""
    a = _numbers(v)
    if a is None or a.ndim != 1:
        return str(v)
    return "(" + ", ".join(str(int(x)) if float(x).is_integer() else str(round(float(x), 2)) for x in a) + ")"


def _same_box(got, want) -> bool:
    """Right when the four numbers are the course's, in the same order; a tuple, a list or an array of them."""
    a = _numbers(got)
    return a is not None and a.shape == (4,) and bool(np.array_equal(a, np.asarray(want, dtype=float)))


def _four(got, text) -> bool:
    """Only the four neighbors of the maze: (r - 1, c) in the code, and no neighbor at a corner, such as (r - 1, c - 1)."""
    return (re.search(r"\(\s*r\s*-\s*1\s*,\s*c\s*\)", text) is not None
            and re.search(r"\(\s*r\s*-\s*1\s*,\s*c\s*[-+]\s*1\s*\)", text) is None)


def _negative(got, text) -> bool:
    """A box with a row or a column below 0: a neighbor at -1, which Python takes from the end."""
    a = _numbers(got)
    return a is not None and a.ndim == 1 and bool((a < 0).any())


# The lesson is done when these are correct. "count" is a choice question and "filter" an open question in the lesson.
REQUIRED = ["count", "box", "smallest", "filter"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace each `...` with your own code.",
                nl="Er staat nog een `...` in je code. Vervang elke `...` door je eigen code.")

_SAME = {"en": "`{call}` gives `{got}`, and it should give `{want}`.", "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn."}


def _m(key, en, nl, **kw):
    """A mistake whose message starts with what the call gives and what it should give."""
    return Mistake(key, en=_SAME["en"] + " " + en, nl=_SAME["nl"] + " " + nl, **kw)


_NEIGHBORS = ("from collections import deque\n\n"
              "import numpy as np\n\n\n"
              "def white_neighbors(img, r, c):\n"
              "    result = []\n"
              "    for nr in range(max(r - 1, 0), min(r + 2, len(img))):\n"
              "        for nc in range(max(c - 1, 0), min(c + 2, len(img[0]))):\n"
              "            if img[nr, nc]:\n"
              "                result.append((nr, nc))\n"
              "    return result\n\n\n")
_HEAD = (_NEIGHBORS +
         "def box_of(img, start):\n"
         "    img = np.array(img)\n"
         "    seen = set()\n"
         "    seen.add(start)\n"
         "    queue = deque([start])\n"
         "    shape = []\n"
         "    while len(queue) > 0:\n")
_ROWS = ("    rows = np.array(shape)[:, 0]\n"
         "    cols = np.array(shape)[:, 1]\n")
_STEP = ("        r, c = queue.popleft()\n"
         "        shape.append((r, c))\n"
         "        for nxt in white_neighbors(img, r, c):\n"
         "            if nxt not in seen:\n"
         "                seen.add(nxt)\n"
         "                queue.append(nxt)\n")
_RETURN = "rows.min(), cols.min(), rows.max(), cols.max()"
_START = _HEAD + "        ...\n" + _ROWS + "    return ...\n"


def _with(step: str = _STEP, back: str = _RETURN) -> str:
    """The course's code with another step of the loop, or another box given back."""
    return _HEAD + step + _ROWS + f"    return {back}\n"


_SOLUTION = _with()

# Another way to the same answer: Python's own min and max over the pixels, without the two lines with np.array.
_BY_HAND = (_HEAD + _STEP
            + "    return min(p[0] for p in shape), min(p[1] for p in shape), max(p[0] for p in shape), max(p[1] for p in shape)\n")


def _image(*rows: str) -> list[list[int]]:
    """A small binary image, written as rows of 0 and 1."""
    return [[int(x) for x in row] for row in rows]


# The example of exercise 2: two shapes; the pixel at row 3, column 2 touches the first one only at a corner.
SHAPES = _image("000000", "011000", "010011", "001001", "000000")
# The example in *The idea*: four shapes; the pixel at row 4, column 0 touches shape 3 only at a corner.
IDEA = _image("1100011", "0100110", "0000000", "0110001", "1000001")
# The image of exercise 1.
COUNT = _image("01001", "10010", "00100", "10001", "11011")
_DOT = _image("00000", "00000", "00100", "00000", "00000")
_EDGE = _image("1111", "0001", "1001", "1011")
_LINE = _image("10000", "01000", "00100", "00010")
_U = _image("1001", "1001", "1111")
_WRAP = _image("1100", "0000", "0000", "1000")
_PI = (np.array([[3, 1, 4, 1, 5, 9], [2, 6, 5, 3, 5, 8], [9, 7, 9, 3, 2, 3], [8, 4, 6, 2, 6, 4], [3, 3, 8, 3, 2, 7]]) > 5).astype(int).tolist()
_BOOLS = [[False, True, False, False, False], [False, False, True, True, False], [False, False, False, False, False],
          [True, True, False, False, True]]

EXERCISES = {
    "box": FunctionBox(
        name="box",
        picture=yours.steps(call=0),
        function="box_of",
        start=_START,
        solution=_SOLUTION,
        visible=[(SHAPES, (1, 1))],
        hidden=[(SHAPES, (2, 4)), (SHAPES, (3, 2)), (_DOT, (2, 2)), (_WRAP, (0, 0)), (_EDGE, (0, 0)), (_EDGE, (3, 0)),
                (_LINE, (0, 0)), (_U, (0, 3)), (_PI, (1, 1)), (_PI, (4, 2)), (_BOOLS, (1, 3))],
        show=_shown_box,
        compare=_same_box,
        first=[_DOTS],
        mistakes=[
            Mistake("empty-shape", error=lambda e: isinstance(e, IndexError) and "too many indices" in str(e),
                    en="With `{call}`, your function stops with an error in the line with `rows`: `shape` is still empty, so `np.array(shape)` has no columns. Add every pixel you take from the queue to `shape`, with `shape.append((r, c))`. If you do, but only for a pixel that is not in `seen` yet: the start is in `seen` from the first lines on, so it is skipped. Here you look in `seen` for each neighbor, before it goes into the queue.",
                    nl="Met `{call}` stopt je functie met een fout in de regel met `rows`: `shape` is nog leeg, dus `np.array(shape)` heeft geen kolommen. Voeg elke pixel die je uit de wachtrij haalt toe aan `shape`, met `shape.append((r, c))`. Doe je dat al, maar alleen voor een pixel die nog niet in `seen` zit? De start zit vanaf de eerste regels in `seen`, dus die sla je over. Hier kijk je voor elke buur in `seen`, voordat hij de wachtrij in gaat."),
            Mistake("border", error=lambda e: isinstance(e, IndexError) and "out of bounds" in str(e),
                    en="With `{call}`, your function stops with an error: it looks at a neighbor outside the image. Past the last row or column there is nothing, and a row or column below 0 Python takes from the end, so the search can wander off. `white_neighbors(img, r, c)` gives only the neighbors inside the image: use it for the neighbors.",
                    nl="Met `{call}` stopt je functie met een fout: hij kijkt naar een buur buiten het beeld. Na de laatste rij of kolom is er niets, en een rij of kolom onder 0 neemt Python vanaf het eind, dus de zoektocht kan gaan dwalen. `white_neighbors(img, r, c)` geeft alleen de buren binnen het beeld: gebruik die voor de buren."),
            Mistake("pop-arguments", error=lambda e: isinstance(e, TypeError) and "popleft" in str(e),
                    en="With `{call}`, your function stops with an error: `popleft()` takes the pixel at the front, and needs nothing between its parentheses.",
                    nl="Met `{call}` stopt je functie met een fout: `popleft()` pakt de pixel vooraan, en heeft niets tussen de haakjes nodig."),
            _m("negative", "A row or column below 0 is not in the image: at `r = 0`, `r - 1` is -1, and Python takes row -1 from the end, the last row. `white_neighbors(img, r, c)` gives only the neighbors inside the image: use it for the neighbors.",
               "Een rij of kolom onder 0 zit niet in het beeld: bij `r = 0` is `r - 1` gelijk aan -1, en Python neemt rij -1 vanaf het eind, de laatste rij. `white_neighbors(img, r, c)` geeft alleen de buren binnen het beeld: gebruik die voor de buren.",
               matches=_negative),
            _m("start-missing", "Your box misses the start pixel: add the pixel you take from the queue to `shape`, with `shape.append((r, c))`, and not each neighbor you put in.",
               "Je box mist de startpixel: voeg de pixel die je uit de wachtrij haalt toe aan `shape`, met `shape.append((r, c))`, en niet elke buur die je erin zet.",
               wrong=_with("        r, c = queue.popleft()\n"
                           "        for nxt in white_neighbors(img, r, c):\n"
                           "            if nxt not in seen:\n"
                           "                seen.add(nxt)\n"
                           "                queue.append(nxt)\n"
                           "                shape.append(nxt)\n")),
            _m("alone", "Your box holds only the start pixel: the search never puts a neighbor in the queue. For every white neighbor from `white_neighbors(img, r, c)` that is not in `seen` yet: add it to `seen`, and put it at the back of the queue with `queue.append`.",
               "Je box bevat alleen de startpixel: de zoektocht zet nooit een buur in de wachtrij. Voor elke witte buur uit `white_neighbors(img, r, c)` die nog niet in `seen` zit: voeg hem toe aan `seen`, en zet hem achteraan de wachtrij met `queue.append`.",
               wrong=_with("        r, c = queue.popleft()\n        shape.append((r, c))\n")),
            _m("four", "Your search finds only the neighbors above, below, on the left and on the right, as in the maze, so it misses a pixel that touches the shape only at a corner. `white_neighbors(img, r, c)` gives all eight around it.",
               "Je zoektocht vindt alleen de buren erboven, eronder, links en rechts, zoals in het doolhof, dus hij mist een pixel die de vorm alleen met een hoek raakt. `white_neighbors(img, r, c)` geeft ze alle acht.",
               matches=_four),
            _m("black", "Your search also walks over black pixels, so it fills the whole image. Put only white neighbors in the queue: `white_neighbors(img, r, c)` gives those.",
               "Je zoektocht loopt ook over zwarte pixels, dus hij vult het hele beeld. Zet alleen witte buren in de wachtrij: `white_neighbors(img, r, c)` geeft die.",
               wrong=_with("        r, c = queue.popleft()\n        shape.append((r, c))\n"
                           "        for nr in range(max(r - 1, 0), min(r + 2, len(img))):\n"
                           "            for nc in range(max(c - 1, 0), min(c + 2, len(img[0]))):\n"
                           "                if (nr, nc) not in seen:\n"
                           "                    seen.add((nr, nc))\n"
                           "                    queue.append((nr, nc))\n")),
            _m("order", "The numbers are right, but in another order: first the top and the left side, then the bottom and the right side. Give back `(top, left, bottom, right)`: `rows.min(), cols.min(), rows.max(), cols.max()`.",
               "De getallen kloppen, maar in een andere volgorde: eerst de bovenkant en de linkerkant, dan de onderkant en de rechterkant. Geef `(boven, links, onder, rechts)` terug: `rows.min(), cols.min(), rows.max(), cols.max()`.",
               wrong=_with(back="rows.min(), rows.max(), cols.min(), cols.max()")),
            _m("xy", "The columns come first in your box, the way many detection programs write a box, with x before y. This lesson writes the rows first, as NumPy does: `(top, left, bottom, right)`, so `rows.min(), cols.min(), rows.max(), cols.max()`.",
               "In je box komen de kolommen eerst, zoals veel programma's voor detectie een box schrijven, met x voor y. Deze les schrijft de rijen eerst, zoals NumPy: `(boven, links, onder, rechts)`, dus `rows.min(), cols.min(), rows.max(), cols.max()`.",
               wrong=_with(back="cols.min(), rows.min(), cols.max(), rows.max()")),
            _m("plus-one", "The bottom and the right side are one too far: the box ends at the last row and the last column of the shape themselves, `rows.max()` and `cols.max()`, without `+ 1`.",
               "De onderkant en de rechterkant liggen er één te ver: de box eindigt bij de laatste rij en de laatste kolom van de vorm zelf, `rows.max()` en `cols.max()`, zonder `+ 1`.",
               wrong=_with(back="rows.min(), cols.min(), rows.max() + 1, cols.max() + 1")),
            _m("size", "The last two numbers are the height and the width of the box. This lesson gives the box as its four sides: the bottom is the largest row, `rows.max()`, and the right side the largest column, `cols.max()`.",
               "De laatste twee getallen zijn de hoogte en de breedte van de box. Deze les geeft de box als zijn vier zijden: de onderkant is de grootste rij, `rows.max()`, en de rechterkant de grootste kolom, `cols.max()`.",
               wrong=_with(back="rows.min(), cols.min(), rows.max() - rows.min() + 1, cols.max() - cols.min() + 1")),
            _m("whole", "That is the box around every white pixel of the image, also those of other shapes. The box of the shape holds only the pixels the search finds from `start`: the pixels in `shape`.",
               "Dat is de box om elke witte pixel van het beeld, ook die van andere vormen. De box van de vorm bevat alleen de pixels die de zoektocht vanaf `start` vindt: de pixels in `shape`.",
               wrong=_HEAD + "        queue.popleft()\n"
               + "    rows = np.nonzero(img)[0]\n    cols = np.nonzero(img)[1]\n    return " + _RETURN + "\n"),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the first `...` with one pass of the loop: take the oldest pixel from the queue, add it to `shape`, and put its white neighbors that are not in `seen` in `seen` and at the back of the queue. Replace the second with the box.",
                      "nl": "Vervang de eerste `...` door één ronde van de loop: haal de oudste pixel uit de wachtrij, voeg hem toe aan `shape`, en zet zijn witte buren die nog niet in `seen` zitten in `seen` en achteraan de wachtrij. Vervang de tweede door de box."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "slow": {"en": "Your function is still not done after 2 seconds. Does every pass of the loop take a pixel from the queue, with `queue.popleft()`? And does a neighbor go into the queue only when it is not in `seen` yet, and into `seen` at the same time? Otherwise two white pixels keep putting each other back in the queue.",
                     "nl": "Je functie is na 2 seconden nog niet klaar. Haalt elke ronde van de loop een pixel uit de wachtrij, met `queue.popleft()`? En gaat een buur alleen de wachtrij in als hij nog niet in `seen` zit, en tegelijk in `seen`? Anders zetten twee witte pixels elkaar steeds weer in de wachtrij."},
            "arguments": {"en": "The check calls `{call}`, with an image and a start pixel, and the `def` line of your function has a different number of names between its parentheses. Keep `(img, start)`.",
                          "nl": "De check roept `{call}` aan, met een beeld en een startpixel, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(img, start)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. The box runs from the smallest to the largest row and from the smallest to the largest column of the pixels the search finds from `start`, also at a corner.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. De box loopt van de kleinste tot de grootste rij en van de kleinste tot de grootste kolom van de pixels die de zoektocht vanaf `start` vindt, ook via een hoek."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your search work for a start pixel in the middle of a shape, and for a shape at the border of the image?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je zoektocht voor een startpixel midden in een vorm, en voor een vorm aan de rand van het beeld?"},
            "correct": {"en": "Your function is right in all {n} cases. In the example, the search finds four pixels from `(1, 1)`: `(1, 2)` and `(2, 1)` in the first step, and `(3, 2)`, which touches `(2, 1)` only at a corner, in the second. The other shape, on the right, it never reaches.",
                        "nl": "Je functie klopt in alle {n} gevallen. In het voorbeeld vindt de zoektocht vanaf `(1, 1)` vier pixels: `(1, 2)` en `(2, 1)` in de eerste stap, en `(3, 2)`, dat `(2, 1)` alleen met een hoek raakt, in de tweede. De andere vorm, rechts, bereikt hij nooit."},
        },
    ),
    "smallest": Value(
        name="smallest",
        start=1,
        solution=5,            # the smallest stop that keeps only the four apples; the test of the lesson measures it again
        outcome=int,
        prose_numbers=True,
        mistakes=[
            Mistake("specks", matches=lambda got, n: int(n) < 5,
                    en="With {value}, some specks are still there: the sentence under the picture counts more than four shapes. Move the slider to the right.",
                    nl="Met {value} zijn er nog spikkels over: de zin onder het plaatje telt meer dan vier vormen. Schuif de slider naar rechts."),
            Mistake("apple-gone", matches=lambda got, n: int(n) > 2094,
                    en="With {value}, the small apple on the right is gone too: it has 2094 pixels. Look for the smallest size that keeps only the four apples.",
                    nl="Met {value} is de kleine appel rechts ook weg: die heeft 2094 pixels. Zoek de kleinste grootte die alleen de vier appels houdt."),
            Mistake("big", matches=lambda got, n: int(n) > 5,
                    en="With {value}, only the four apples are left, but a smaller size does that too. Look for the smallest: a filter by size also removes small shapes you may want to keep.",
                    nl="Met {value} zijn alleen de vier appels over, maar een kleinere grootte doet dat ook. Zoek de kleinste: een filter op grootte haalt ook kleine vormen weg die je misschien wilt houden."),
        ],
        texts={
            "missing": {"en": "I cannot see the slider. Run the cell with the slider again.",
                        "nl": "Ik zie de slider niet. Voer de cel met de slider opnieuw uit."},
            "empty": {"en": "The smallest size is still 1: every shape stays, all 1335 of them. Move the slider to the right.",
                      "nl": "De kleinste grootte is nog 1: elke vorm blijft, alle 1335. Schuif de slider naar rechts."},
            "wrong": {"en": "{value} is not it. Look for the smallest size after which only the four apples are left.",
                      "nl": "{value} is het niet. Zoek de kleinste grootte waarna alleen de vier appels over zijn."},
            "correct": {"en": "{value} is right: every speck has 1 to 4 pixels, so from 5 on only the four apples are left, with 146 pixels wrong: the 101 pixels of the holes of the shine, and 45 pixels of the tablecloth. Every size up to 2094, the size of the small apple, gives the same here; the smallest is the safest, because it keeps a small shape you may want, such as an apple further away.",
                        "nl": "{value} klopt: elke spikkel heeft 1 tot 4 pixels, dus vanaf 5 zijn alleen de vier appels over, met 146 pixels fout: de 101 pixels van de gaten van de glans, en 45 pixels van het tafelkleed. Elke grootte tot 2094, de grootte van de kleine appel, geeft hier hetzelfde; de kleinste is het veiligst, want die houdt een kleine vorm die je misschien wilt, zoals een appel verder weg."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("box", _START, "empty", None),
    ("box", _SOLUTION, "correct", None),
    ("box", _BY_HAND, "correct", None),
    # the search of the maze: a pixel that is done already is skipped when it comes out of the queue
    ("box", _HEAD + "        r, c = queue.popleft()\n        shape.append((r, c))\n"
     "        for nxt in white_neighbors(img, r, c):\n            if nxt not in shape:\n                queue.append(nxt)\n"
     + _ROWS + f"    return {_RETURN}\n", "correct", None),
    ("box", _with("        pixel = queue.popleft()\n        shape.append(pixel)\n"
                  "        for nxt in white_neighbors(img, pixel[0], pixel[1]):\n            if nxt not in seen:\n"
                  "                seen.add(nxt)\n                queue.append(nxt)\n"), "correct", None),
    ("box", _with(back="(rows.min(), cols.min(), rows.max(), cols.max())"), "correct", None),
    ("box", _with(back="[rows.min(), cols.min(), rows.max(), cols.max()]"), "correct", None),
    ("box", _with(back="min(rows), min(cols), max(rows), max(cols)"), "correct", None),
    ("box", _with(_STEP.replace("        shape.append((r, c))\n", "")), "error", "empty-shape"),
    ("box", _with("        r, c = queue.popleft()\n        shape.append((r, c))\n"
                  "        for nr in range(r - 1, r + 2):\n            for nc in range(c - 1, c + 2):\n"
                  "                if img[nr, nc]:\n                    if (nr, nc) not in seen:\n"
                  "                        seen.add((nr, nc))\n                        queue.append((nr, nc))\n"), "error", "border"),
    ("box", _with("        r, c = queue.popleft()\n        shape.append((r, c))\n"
                  "        for nr in range(r - 1, min(r + 2, len(img))):\n            for nc in range(c - 1, min(c + 2, len(img[0]))):\n"
                  "                if img[nr, nc]:\n                    if (nr, nc) not in seen:\n"
                  "                        seen.add((nr, nc))\n                        queue.append((nr, nc))\n"), "wrong", "negative"),
    ("box", _with(_STEP.replace("queue.popleft()", "queue.popleft(0)")), "error", "pop-arguments"),
    ("box", _with("        r, c = queue.popleft()\n        shape.append((r, c))\n"), "wrong", "alone"),
    ("box", _with("        r, c = queue.popleft()\n        shape.append((r, c))\n"
                  "        for nxt in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:\n"
                  "            if nxt in white_neighbors(img, r, c):\n                if nxt not in seen:\n"
                  "                    seen.add(nxt)\n                    queue.append(nxt)\n"), "wrong", "four"),
    ("box", _with("        r, c = queue.popleft()\n        shape.append((r, c))\n"
                  "        for nr in range(max(r - 1, 0), min(r + 2, len(img))):\n"
                  "            for nc in range(max(c - 1, 0), min(c + 2, len(img[0]))):\n"
                  "                if (nr, nc) not in seen:\n                    seen.add((nr, nc))\n"
                  "                    queue.append((nr, nc))\n"), "wrong", "black"),
    ("box", _with(back="rows.min(), rows.max(), cols.min(), cols.max()"), "wrong", "order"),
    ("box", _with(back="cols.min(), rows.min(), cols.max(), rows.max()"), "wrong", "xy"),
    ("box", _with(back="rows.min(), cols.min(), rows.max() + 1, cols.max() + 1"), "wrong", "plus-one"),
    ("box", _with(back="rows.min(), cols.min(), rows.max() - rows.min() + 1, cols.max() - cols.min() + 1"), "wrong", "size"),
    ("box", _HEAD + "        queue.popleft()\n    rows = np.nonzero(img)[0]\n    cols = np.nonzero(img)[1]\n    return "
     + _RETURN + "\n", "wrong", "whole"),
    ("box", _with(_STEP.replace("                seen.add(nxt)\n", "")), "slow", None),
    # seen.add without the if: every neighbor goes in the queue again
    ("box", _with("        r, c = queue.popleft()\n        shape.append((r, c))\n"
                  "        for nxt in white_neighbors(img, r, c):\n            seen.add(nxt)\n            queue.append(nxt)\n"),
     "slow", None),
    # the habit of the maze: seen checked when a pixel comes out of the queue; the start is skipped
    ("box", _with("        r, c = queue.popleft()\n        if (r, c) not in seen:\n            seen.add((r, c))\n"
                  "            shape.append((r, c))\n            for nxt in white_neighbors(img, r, c):\n"
                  "                queue.append(nxt)\n"), "error", "empty-shape"),
    ("box", _with("        r, c = queue.popleft()\n"
                  "        for nxt in white_neighbors(img, r, c):\n            if nxt not in seen:\n"
                  "                seen.add(nxt)\n                queue.append(nxt)\n                shape.append(nxt)\n"),
     "wrong", "start-missing"),
    # the four neighbors of the maze, without white_neighbors: the border makes rows of -1
    ("box", _with("        r, c = queue.popleft()\n        shape.append((r, c))\n"
                  "        for nxt in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:\n"
                  "            if nxt[0] < len(img):\n                if nxt[1] < len(img[0]):\n"
                  "                    if img[nxt[0], nxt[1]]:\n                        if nxt not in seen:\n"
                  "                            seen.add(nxt)\n                            queue.append(nxt)\n"), "wrong", "four"),
    ("box", _with("        r, c = queue[0]\n        shape.append((r, c))\n"), "slow", None),
    ("box", _with(back="rows.max(), cols.max(), rows.min(), cols.min()"), "wrong", None),
    ("box", _HEAD + "        ...\n" + _ROWS + f"    return {_RETURN}\n", "wrong", "dots-left"),
    ("smallest", 1, "empty", None),
    ("smallest", 5, "correct", None),
    ("smallest", 2, "wrong", "specks"),
    ("smallest", 4, "wrong", "specks"),
    ("smallest", 10, "wrong", "big"),
    ("smallest", 2000, "wrong", "big"),
    ("smallest", 3000, "wrong", "apple-gone"),
]
