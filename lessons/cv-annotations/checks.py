# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-annotations: which annotation a task needs, a box in the course's order written as COCO
writes it, the flag of a keypoint that is hidden, and why someone who builds a dataset for object detection draws
boxes and not masks. And the pictures: the four kinds of annotation, on the course's own scenes; and for the slider,
a square turned step by step in its box, next to how much of the box it fills at every angle."""
import functools
import math
import re

import numpy as np

from everything_ai import scenes
from everything_ai.check import FunctionBox, Mistake
from everything_ai import yours

SEED = 18                    # the scene of the worked example: two triangles, a square and a circle on top of it
ONE = 16                     # the scene of one shape, a triangle, for a label per image: scenes.scene(ONE, count=1)
MEASURED = 2000              # the numbers in the text are measured on the scenes of seeds 0 to 1999
ANGLES = list(range(0, 95, 5))   # the stops of the slider, in degrees
TURN_CENTRE = (63.5, 63.5)   # the turned square of the slider: in the middle of a scene
TURN_RADIUS = 60.0           # from its middle to a corner, in pixels: big enough for a smooth curve
TURN_COLOUR = scenes.COLOURS[1]

# The numbers in the text. `measure()` measures them again; the test of the lesson checks that both are the same.
REFERENCE = {
    # per kind, for shapes that nothing hides: how many, and how much of its box a shape fills (smallest, largest,
    # mean), from the visible pixels and the box with both ends included
    "fill": {"circle": (1815, 0.7335, 0.83, 0.7855), "square": (1940, 0.4876, 1.0, 0.6495),
             "triangle": (2069, 0.4486, 0.5725, 0.4973)},
    "shapes": 7945,          # every shape of the 2000 scenes
    "whole": 5824,           # of them, the shapes that nothing hides
    "other": 4325,           # the boxes that also hold visible pixels of another shape
    "corners": 18523,        # every corner of a square or a triangle: a keypoint
    "hidden": 1215,          # of them, the keypoints with v = 1
    # the turned square of the slider: how much of its box it fills at every stop, rounded to whole percents
    "turn": {0: 100, 5: 85, 10: 75, 15: 67, 20: 62, 25: 57, 30: 54, 35: 52, 40: 52, 45: 50, 50: 52, 55: 52, 60: 54,
             65: 57, 70: 62, 75: 67, 80: 75, 85: 85, 90: 100},
}
# What annotating costs, from the papers: "over 22 worker hours per 1,000 segmentations" in COCO (Lin et al. 2014,
# arXiv:1405.0312), so more than 79 seconds per mask; "35s for one highquality box" drawn the usual way, and "only 7s
# per box" with extreme clicking (Papadopoulos et al. 2017, arXiv:1708.02750).
COST = {"mask_hours_per_1000": 22, "box_slow": 35, "box_fast": 7}


def mask_seconds() -> int:
    """The seconds per mask in COCO from 22 worker hours per 1000 masks; the paper says "over", so the text says more
    than this."""
    return round(COST["mask_hours_per_1000"] * 3600 / 1000)


def box_of_mask(mask) -> tuple[int, int, int, int]:
    """The box of a mask, as the worked example derives it: the smallest and the largest row and column."""
    mask = np.asarray(mask, dtype=bool)
    rows = np.arange(len(mask))[mask.sum(axis=1) > 0]
    cols = np.arange(len(mask[0]))[mask.sum(axis=0) > 0]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def fill(mask, box) -> float:
    """How much of the box the pixels of the mask fill."""
    top, left, bottom, right = box
    return float(np.asarray(mask, dtype=bool).sum() / ((bottom - top + 1) * (right - left + 1)))


def measure() -> dict:
    """Every number of REFERENCE, measured again on the course's scenes."""
    fills = {kind: [] for kind in scenes.KINDS}
    shapes = whole = other = corners = hidden = 0
    for seed in range(MEASURED):
        s = scenes.scene(seed)
        for i, shape in enumerate(s.shapes):
            shapes += 1
            if shape.whole:
                whole += 1
                fills[shape.kind].append(fill(shape.mask, shape.box))
            top, left, bottom, right = shape.box
            if any(o.mask[top:bottom + 1, left:right + 1].any() for j, o in enumerate(s.shapes) if j != i):
                other += 1
            corners += len(shape.keypoints)
            hidden += sum(v == 1 for _, _, v in shape.keypoints)
    return {
        "fill": {k: (len(v), round(min(v), 4), round(max(v), 4), round(float(np.mean(v)), 4)) for k, v in fills.items()},
        "shapes": shapes, "whole": whole, "other": other, "corners": corners, "hidden": hidden,
        "turn": {a: turn_outcome(a)["fill"] for a in ANGLES},
    }


def percent(part: int, whole: int, digits: int = 0) -> str:
    """A share as the text writes it: 54, or 6.6 with one digit."""
    value = 100 * part / whole
    return f"{value:.{digits}f}"


@functools.lru_cache(maxsize=1)
def example() -> scenes.Scene:
    """The scene of the worked example."""
    return scenes.scene(SEED)


def square() -> scenes.Shape:
    """The square of the worked example, partly under the circle."""
    return next(s for s in example().shapes if s.kind == "square")


@functools.lru_cache(maxsize=None)
def _turned(angle: int) -> np.ndarray:
    """The pixels of the square of the slider, turned by `angle` degrees: upright at 0 and at 90."""
    mask, _ = scenes.shape_mask("square", TURN_CENTRE, TURN_RADIUS, math.radians(angle) + math.pi / 4)
    mask.flags.writeable = False
    return mask


def turn_outcome(angle: int) -> dict:
    """At this stop of the slider: the box of the turned square, and how much of it the square fills, in percents."""
    mask = _turned(int(angle))
    box = box_of_mask(mask)
    return {"box": box, "fill": round(100 * fill(mask, box))}


def _photo_axes(ax, img, title):
    if np.ndim(img) == 3:
        ax.imshow(img, interpolation="nearest")
    else:
        ax.imshow(np.asarray(img, dtype=float), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title, fontsize=9)


def _line(ax, xs, ys):
    """An annotation drawn on a scene: a line in the color of data, with a thin edge in the color of the page, so that
    it stands out on every shape, in light and in dark."""
    from everything_ai import plots
    ax.plot(xs, ys, color=plots.PAGE, linestyle="-", linewidth=3.2, solid_joinstyle="miter")
    ax.plot(xs, ys, color=plots.DATA, linestyle="-", linewidth=1.4, solid_joinstyle="miter")


def _draw_box(ax, b):
    """A box as a line around its pixels: from the outer edge of the top-left pixel to that of the bottom-right one."""
    top, left, bottom, right = b
    top, bottom, left, right = top - 0.5, bottom + 0.5, left - 0.5, right + 0.5
    _line(ax, [left, right, right, left, left], [top, top, bottom, bottom, top])


def _draw_keypoint(ax, r, c, v):
    """A keypoint: a filled dot when it can be seen (v = 2), an open circle when it is hidden (v = 1)."""
    from everything_ai import plots
    if v == 2:
        ax.plot([c], [r], linestyle="none", marker="o", markersize=5.5, color=plots.DATA, markeredgecolor=plots.PAGE, markeredgewidth=1.2)
    else:
        ax.plot([c], [r], linestyle="none", marker="o", markersize=7.5, markerfacecolor="none", markeredgecolor=plots.PAGE, markeredgewidth=3.0)
        ax.plot([c], [r], linestyle="none", marker="o", markersize=7.5, markerfacecolor="none", markeredgecolor=plots.DATA, markeredgewidth=1.4)


def annotations_picture(labels: dict[str, str]) -> str:
    """Four kinds of annotation: an image of one shape with its label; and the scene of the worked example with its
    boxes, the mask of the square, and the corners as keypoints."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    s = example()
    fig, axes = plt.subplots(2, 2, figsize=(5.4, 5.7))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.01, wspace=0.06, hspace=0.12)
    (one, boxes), (masks, points) = axes
    _photo_axes(one, scenes.scene(ONE, count=1).image, labels["label"])
    _photo_axes(boxes, s.image, labels["boxes"])
    for shape in s.shapes:
        _draw_box(boxes, shape.box)
    # the mask of the square in white; every other shape in dark gray, with a mask of its own
    tones = np.zeros(s.image.shape[:2])
    for shape in s.shapes:
        tones[shape.mask] = 0.3
    tones[square().mask] = 1.0
    _photo_axes(masks, tones, labels["mask"])
    _photo_axes(points, s.image, labels["keypoints"])
    for shape in s.shapes:
        for r, c, v in shape.keypoints:
            _draw_keypoint(points, r, c, v)
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


def turn_frame(angle: int, labels: dict[str, str]) -> tuple[str, dict]:
    """For the slider: the square turned by `angle` degrees in its box, and next to it how much of its box it fills at
    every stop, with this one marked. With what the sentence under it says."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    angle = int(angle)
    out = turn_outcome(angle)
    img = np.full((scenes.SIZE, scenes.SIZE, 3), 222, dtype=np.uint8)
    img[_turned(angle)] = TURN_COLOUR
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.28})
    _photo_axes(left, img, labels["title"].format(angle=angle))
    _draw_box(left, out["box"])
    curve = [turn_outcome(a)["fill"] for a in ANGLES]
    right.plot(ANGLES, curve, color=plots.INK, linestyle="-", linewidth=1.8, marker="o", markersize=3.2)
    right.plot([angle, angle], [40, out["fill"]], color=plots.INK_3, linestyle=":", linewidth=1.2)
    right.plot([angle], [out["fill"]], linestyle="none", marker="o", color=plots.INK, markersize=8)
    right.set_xlim(-4, 94)
    right.set_ylim(40, 104)
    right.set_xticks([0, 15, 30, 45, 60, 75, 90])
    right.set_yticks([40, 50, 60, 70, 80, 90, 100])
    right.set_yticklabels([f"{v}%" for v in (40, 50, 60, 70, 80, 90, 100)])
    right.set_xlabel(labels["angle"])
    right.set_title(labels["fill"], fontsize=10, loc="left")
    html = plots.render(fig, alt=labels["alt"].format(angle=angle, **out))
    plt.close(fig)
    return html, out


def _numbers(got):
    """What the learner's function gave back as an array of floats, or None if it is not numbers: text such as '39'
    or True and False do not count as numbers."""
    a = np.asarray(got, dtype=object)
    if any(isinstance(x, (str, bytes, bool, np.bool_)) for x in a.ravel()):
        return None
    try:
        return np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return None


def _shown(v) -> str:
    """A box the way COCO writes it: four numbers between square brackets."""
    a = _numbers(v)
    if a is None or a.ndim != 1:
        return str(v)
    return "[" + ", ".join(str(int(x)) if float(x).is_integer() else str(round(float(x), 2)) for x in a) + "]"


def _same(got, want) -> bool:
    """Right when the four numbers are the course's, in the same order; a list, a tuple or an array of them."""
    a = _numbers(got)
    return a is not None and a.shape == (4,) and bool(np.array_equal(a, np.asarray(want, dtype=float)))


# The lesson is done when these are correct. "which" and "hidden" are choice questions and "cost" an open question in
# the lesson.
REQUIRED = ["which", "coco", "hidden", "cost"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace each `...` with your own code.",
                nl="Er staat nog een `...` in je code. Vervang elke `...` door je eigen code.")

_SAME = {"en": "`{call}` gives `{got}`, and it should give `{want}`.", "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn."}


def _m(key, en, nl, **kw):
    """A mistake whose message starts with what the call gives and what it should give."""
    return Mistake(key, en=_SAME["en"] + " " + en, nl=_SAME["nl"] + " " + nl, **kw)


_HEAD = ("def to_coco(box):\n"
         "    # box: (top, left, bottom, right), both ends included\n"
         "    top, left, bottom, right = box\n")


def _with(x: str = "left", y: str = "top", width: str = "right - left + 1", height: str = "bottom - top + 1") -> str:
    """The course's code, with other lines for x, y, the width or the height."""
    return _HEAD + f"    x = {x}\n    y = {y}\n    width = {width}\n    height = {height}\n" + "    return [x, y, width, height]\n"


_START = _with("...", "...", "...", "...")
_SOLUTION = _with()

# The box of the square of the worked example, and the other boxes the check tries: a triangle, one pixel, the whole
# scene, a wide box and a tall one.
SQUARE_BOX = (42, 39, 73, 72)

EXERCISES = {
    "coco": FunctionBox(
        name="coco",
        picture=yours.steps(call=0),
        function="to_coco",
        start=_START,
        solution=_SOLUTION,
        visible=[(SQUARE_BOX,)],
        hidden=[((102, 8, 117, 23),), ((5, 5, 5, 5),), ((0, 0, 127, 127),), ((10, 20, 14, 29),), ((30, 3, 61, 11),)],
        show=_shown,
        compare=_same,
        first=[_DOTS],
        mistakes=[
            Mistake("unpack", error=lambda e: isinstance(e, ValueError) and "unpack" in str(e),
                    en="With `{call}`, your function stops with an error in the line that takes the box apart: a box holds four numbers, so that line needs four names, `top, left, bottom, right = box`.",
                    nl="Met `{call}` stopt je functie met een fout in de regel die de box uit elkaar haalt: een box heeft vier getallen, dus die regel heeft vier namen nodig, `top, left, bottom, right = box`."),
            _m("xy", "x is a column, as on a graph, where x runs to the right: the left side of the box, `left`. y is a row: the top, `top`.",
               "x is een kolom, zoals in een grafiek, waar x naar rechts loopt: de linkerkant van de box, `left`. y is een rij: de bovenkant, `top`.",
               wrong=_with("top", "left")),
            _m("rows-first", "Your box puts the rows first everywhere, as the course does. COCO puts the columns first: x is the left side, `left`, and the width counts the columns, from `left` to `right`; y is the top, and the height counts the rows.",
               "Je box zet overal de rijen eerst, zoals de course. COCO zet de kolommen eerst: x is de linkerkant, `left`, en de breedte telt de kolommen, van `left` tot `right`; y is de bovenkant, en de hoogte telt de rijen.",
               wrong=_with("top", "left", "bottom - top + 1", "right - left + 1")),
            _m("size-swapped", "The width and the height are swapped. The width runs across, from column `left` to column `right`; the height runs down, from row `top` to row `bottom`.",
               "De breedte en de hoogte zijn omgewisseld. De breedte loopt opzij, van kolom `left` tot kolom `right`; de hoogte loopt omlaag, van rij `top` tot rij `bottom`.",
               wrong=_with(width="bottom - top + 1", height="right - left + 1")),
            _m("corners", "The last two numbers are the bottom-right corner, `(x2, y2)`, as many other programs write a box. COCO gives the width and the height instead: how many columns and rows the box covers.",
               "De laatste twee getallen zijn de hoek rechtsonder, `(x2, y2)`, zoals veel andere programma's een box schrijven. COCO geeft in plaats daarvan de breedte en de hoogte: hoeveel kolommen en rijen de box beslaat.",
               wrong=_with(width="right", height="bottom")),
            _m("unchanged", "That is the box as the course writes it, `(top, left, bottom, right)`. COCO writes x, the left side, first, then y, the top, and then the width and the height.",
               "Dat is de box zoals de course hem schrijft, `(top, left, bottom, right)`. COCO schrijft eerst x, de linkerkant, dan y, de bovenkant, en dan de breedte en de hoogte.",
               wrong=_with("top", "left", "bottom", "right")),
            _m("centre", "x and y are the middle of the box, as some other formats write it. COCO gives the top-left corner: x is `left` and y is `top`.",
               "x en y zijn het midden van de box, zoals sommige andere formaten het schrijven. COCO geeft de hoek linksboven: x is `left` en y is `top`.",
               wrong=_with("(left + right) / 2", "(top + bottom) / 2")),
            _m("plus-one", "The width and the height are both one too small. Both ends are part of the box: from column 39 to column 72 are 72 - 39 + 1 = 34 columns, and from row 42 to row 73 are 32 rows.",
               "De breedte en de hoogte zijn er allebei één te klein. Beide uiteinden horen bij de box: van kolom 39 tot kolom 72 zijn 72 - 39 + 1 = 34 kolommen, en van rij 42 tot rij 73 zijn 32 rijen.",
               wrong=_with(width="right - left", height="bottom - top")),
            _m("plus-one-width", "The width is one too small. Both ends are part of the box: from column 39 to column 72 are 72 - 39 + 1 = 34 columns. The height has it right.",
               "De breedte is er één te klein. Beide uiteinden horen bij de box: van kolom 39 tot kolom 72 zijn 72 - 39 + 1 = 34 kolommen. De hoogte klopt wel.",
               wrong=_with(width="right - left")),
            _m("plus-one-height", "The height is one too small. Both ends are part of the box: from row 42 to row 73 are 73 - 42 + 1 = 32 rows. The width has it right.",
               "De hoogte is er één te klein. Beide uiteinden horen bij de box: van rij 42 tot rij 73 zijn 73 - 42 + 1 = 32 rijen. De breedte klopt wel.",
               wrong=_with(height="bottom - top")),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace each `...` with a number made from the box: `x` is a column, `y` a row, and `width` and `height` say how many columns and rows the box covers.",
                      "nl": "Vervang elke `...` door een getal uit de box: `x` is een kolom, `y` een rij, en `width` en `height` zeggen hoeveel kolommen en rijen de box beslaat."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with one box of four numbers, and the `def` line of your function has a different number of names between its parentheses. Keep `(box)`.",
                          "nl": "De check roept `{call}` aan, met één box van vier getallen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(box)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. x is the column of the left side, y the row of the top, and the width and the height count the columns and the rows of the box, both ends included.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. x is de kolom van de linkerkant, y de rij van de bovenkant, en de breedte en de hoogte tellen de kolommen en de rijen van de box, beide uiteinden meegeteld."},
            "almost": {"en": "Right for the square. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work with the numbers of the box it is given, for every box?",
                       "nl": "Goed voor het vierkant. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Rekent je functie met de getallen van de box die hij krijgt, voor elke box?"},
            "correct": {"en": "Your function is right in all {n} cases. The square's box covers columns 39 to 72, 34 columns, and rows 42 to 73, 32 rows: `[39, 42, 34, 32]`. A box of one pixel, such as `(5, 5, 5, 5)`, is `[5, 5, 1, 1]` in COCO: its bottom is its top, and its height is 1.",
                        "nl": "Je functie klopt in alle {n} gevallen. De box van het vierkant beslaat de kolommen 39 tot 72, 34 kolommen, en de rijen 42 tot 73, 32 rijen: `[39, 42, 34, 32]`. Een box van één pixel, zoals `(5, 5, 5, 5)`, is `[5, 5, 1, 1]` in COCO: zijn onderkant is zijn bovenkant, en zijn hoogte is 1."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("coco", _START, "empty", None),
    ("coco", _SOLUTION, "correct", None),
    ("coco", _SOLUTION.replace("return [x, y, width, height]", "return (x, y, width, height)"), "correct", None),
    ("coco", _HEAD + "    return [left, top, right - left + 1, bottom - top + 1]\n", "correct", None),
    ("coco", "def to_coco(box):\n    return [box[1], box[0], box[3] - box[1] + 1, box[2] - box[0] + 1]\n", "correct", None),
    ("coco", _with(width="1 + right - left", height="(bottom - top) + 1"), "correct", None),
    ("coco", _with("top", "left"), "wrong", "xy"),
    ("coco", _with("top", "left", "bottom - top + 1", "right - left + 1"), "wrong", "rows-first"),
    ("coco", _with(width="bottom - top + 1", height="right - left + 1"), "wrong", "size-swapped"),
    ("coco", _with(width="right", height="bottom"), "wrong", "corners"),
    ("coco", _with("top", "left", "bottom", "right"), "wrong", "unchanged"),
    ("coco", _HEAD + "    return box\n", "wrong", "unchanged"),
    ("coco", _with("(left + right) / 2", "(top + bottom) / 2"), "wrong", "centre"),
    ("coco", _with(width="right - left", height="bottom - top"), "wrong", "plus-one"),
    ("coco", "def to_coco(box):\n    t, l, b, r = box\n    return [l, t, r - l, b - t]\n", "wrong", "plus-one"),
    ("coco", "def to_coco(box):\n    y1, x1, y2, x2 = box\n    return [x1, y1, x2 - x1, y2 - y1 + 1]\n", "wrong", "plus-one-width"),
    ("coco", _with(width="(right - left)"), "wrong", "plus-one-width"),
    ("coco", _with(height="bottom - top"), "wrong", "plus-one-height"),
    ("coco", _with(width="right - left + 2", height="bottom - top + 2"), "wrong", None),
    ("coco", _with(width="right - left - 1"), "wrong", None),
    # a box written out by hand: right for the square only
    ("coco", "def to_coco(box):\n    return [39, 42, 34, 32]\n", "almost", None),
    # text instead of numbers
    ("coco", _SOLUTION.replace("return [x, y, width, height]", "return [str(x), str(y), str(width), str(height)]"), "wrong", None),
    ("coco", _SOLUTION.replace("return [x, y, width, height]", "return np.array([x, y, width, height])").replace(
        "def to_coco", "import numpy as np\n\n\ndef to_coco"), "correct", None),
    ("coco", _with("left", "top", "...", "bottom - top + 1"), "wrong", "dots-left"),
    ("coco", _HEAD.replace("top, left, bottom, right = box", "top, left, bottom = box") + "    return [left, top, 1, 1]\n",
     "error", "unpack"),
    ("coco", _with().replace("def to_coco(box):", "def to_coco(top, left, bottom, right):").replace(
        "    top, left, bottom, right = box\n", ""), "error", "arguments"),
    ("coco", _with().replace("return [x, y, width, height]", "print([x, y, width, height])"), "wrong", "prints"),
]
