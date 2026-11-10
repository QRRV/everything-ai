# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-anchors: match, which tells each anchor box what to learn from its IoU with the true boxes
(a box, background, or nothing), and boxes_from, which turns anchors and their box offsets back into boxes. Each is
saved in the learner's project (work/projects/cnn/match.py and offsets.py) when right, for the one-stage detector of the
next lesson. And the pictures: the anchors of one square of the feature map and the anchors that learn a shape, on the
scene of the worked example; and for the slider, the anchors of one size that reach an IoU of 0.5, next to the share of
the shapes of 500 scenes that get one. Everything is computed from the course's own scenes (scenes.py): no download."""
import functools
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import scenes
from everything_ai.check import Mistake, ScenarioBox

SEED = 18                    # the scene of the worked example, as in cv-annotations, cv-iou and cv-nms
CELLS, STRIDE = 16, 8        # a feature map of 16 by 16 squares, each for 8 by 8 pixels of the scene of 128 by 128
SIZES = ((16, 16), (24, 24), (32, 32))   # (height, width) of the anchors of every square
TALL = ((34, 17), (17, 34))  # the change in the worked example: a tall and a wide anchor with the area of 24 by 24
# RetinaNet (Lin and colleagues, 2017, section 4): "anchors are assigned to ground-truth object boxes using an
# intersection-over-union (IoU) threshold of 0.5; and to background if their IoU is in [0, 0.4)"; in between, ignored.
POSITIVE, BACKGROUND = 0.5, 0.4
IGNORED = -2                 # what match gives an anchor that is ignored; -1 is background
SCENES = 500                 # the scenes 0 to 499, for the measures over many scenes
SLIDER = list(range(8, 58, 2))   # the stops of the slider: one square anchor of 8 to 56 pixels per square
TRIANGLE, LARGE_TRIANGLE, SQUARE, CIRCLE = 0, 1, 2, 3   # the shapes of the scene, as s.shapes[k]

# The numbers in the text. `measure()` measures them again; the test of the lesson checks that both are the same.
REFERENCE = {
    "anchors": (768, (-4, -4, 11, 11), (108, 108, 139, 139)),   # how many, the first and the last
    "boxes": ((102, 8, 117, 23), (51, 7, 75, 29), (42, 39, 73, 72), (58, 50, 86, 78)),   # the true boxes of the scene
    "kinds": ("triangle", "triangle", "square", "circle"),
    "most": (0.4884, 0.6705, 0.7005, 0.6417),   # each shape's highest IoU with an anchor
    "above": (0, 4, 6, 8),                       # each shape's anchors with an IoU of 0.5 or more
    "learn": (2, 4, 6, 8),                       # the anchors that learn each shape, after the second rule
    "counts": (20, 13, 735),                     # anchors that learn a shape, that are ignored, that learn background
    # the circle's best anchor: its number, its square (row, column), its size, its box, its IoU, its offsets
    "circle": (458, (9, 8), (32, 32), (60, 52, 91, 83), 0.6417, (-0.1094, -0.1094, -0.0984, -0.0984)),
    "triangle": ((627, (100, 4, 115, 19)), (630, (100, 12, 115, 27))),   # the small triangle's two best anchors
    "rung": (20, 0.3483),                        # the anchors of the PyTorch example, and their largest offset
    # with the change in the worked example: anchors, learn a shape, ignored, background; share over 500 scenes
    "tall": (1280, 27, 28, 1225, 0.9886),
    "shapes": (2012, 9, 44, 0.88, 1.13),         # boxes of 500 scenes: smallest and largest side, height / width at 5% and 95%
    "found": 0.9876,                             # share of the shapes of 500 scenes with an anchor of 0.5 or more
    "found_07": 0.5497,                          # the same with Faster R-CNN's 0.7
    "rule": (25, 0.4007),                        # shapes whose best anchor is below 0.5, and the lowest best IoU
    "per_scene": (17.2, 6, 33, 0.0224),          # anchors that learn a shape per scene: mean, least, most; share of 768
    "grid8": 0.6963,                             # the three sizes on 8 by 8 squares of 16 pixels
    "nine": 0.9901,                              # 3 sizes times 3 shapes (1:1, 2:1, 1:2) on 16 by 16 squares
    "offsets": ((0.11, 0.11, 0.15, 0.15), (0.25, 0.25, 0.32, 0.32)),   # |t| of every learning anchor: mean, 95%
    # the slider: at every size, the share of the shapes of 500 scenes with an anchor of 0.5 or more
    "curve": {8: 0.0, 10: 0.0005, 12: 0.0149, 14: 0.0755, 16: 0.2018, 18: 0.3449, 20: 0.508, 22: 0.6277, 24: 0.7227,
              26: 0.7843, 28: 0.8101, 30: 0.7932, 32: 0.7545, 34: 0.6695, 36: 0.6223, 38: 0.5477, 40: 0.4682, 42: 0.42,
              44: 0.3504, 46: 0.2868, 48: 0.2416, 50: 0.1754, 52: 0.1322, 54: 0.0964, 56: 0.0706},
    # and on the scene: anchors with an IoU of 0.5 or more, and how many of the 4 shapes have one
    "slider": {8: (0, 0), 10: (0, 0), 12: (0, 0), 14: (0, 0), 16: (0, 0), 18: (4, 2), 20: (4, 2), 22: (5, 2), 24: (8, 3),
               26: (10, 3), 28: (10, 3), 30: (10, 3), 32: (10, 3), 34: (8, 2), 36: (8, 2), 38: (8, 2), 40: (8, 2),
               42: (4, 1), 44: (4, 1), 46: (4, 1), 48: (0, 0), 50: (0, 0), 52: (0, 0), 54: (0, 0), 56: (0, 0)},
    "best_size": (28, 0.8101),
    # the sizes at which the small triangle and the square of the scene have an anchor of 0.5 or more
    "sizes_found": ((18, 20), (24, 46)),
}


# --- the course's versions ----------------------------------------------------------------------------------------

def iou_matrix(anchors, boxes) -> np.ndarray:
    """The IoU of every anchor with every box, as in cv-iou, both ends included: shape (anchors, boxes)."""
    anchors = np.asarray(anchors, dtype=float).reshape(-1, 4)
    out = np.zeros((len(anchors), len(boxes)))
    area_a = (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    for k in range(len(boxes)):
        box = boxes[k]
        top, left = np.maximum(anchors[:, 0], box[0]), np.maximum(anchors[:, 1], box[1])
        bottom, right = np.minimum(anchors[:, 2], box[2]), np.minimum(anchors[:, 3], box[3])
        overlap = np.maximum(0, bottom - top + 1) * np.maximum(0, right - left + 1)
        area_b = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        out[:, k] = overlap / (area_a + area_b - overlap)
    return out


def match(ious) -> np.ndarray:
    """What each anchor learns: the number of the box it overlaps most when that IoU is POSITIVE or more, IGNORED (-2)
    from BACKGROUND up to POSITIVE, else -1, background. Then every box also gets its best anchors, every anchor with
    its highest IoU, also below POSITIVE: Faster R-CNN gives a positive label to "the anchor/anchors with the highest
    Intersection-over-Union (IoU) overlap with a ground-truth box", for "rare cases" where the threshold "may find no
    positive sample". Two anchors share the highest IoU when the middle of a box lies halfway between theirs."""
    ious = np.asarray(ious, dtype=float)
    best, most = ious.argmax(axis=1), ious.max(axis=1)
    label = np.zeros(len(ious), dtype=int) - 1
    label[most >= POSITIVE] = best[most >= POSITIVE]
    label[(most >= BACKGROUND) & (most < POSITIVE)] = IGNORED
    for k in range(ious.shape[1]):
        label[ious[:, k] == ious[:, k].max()] = k
    return label


def middle_size(boxes) -> tuple[np.ndarray, np.ndarray]:
    """The middle (y, x) and the size (h, w) of boxes of shape (N, 4), both ends included."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return (boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2] + 1


def offsets(anchors, boxes) -> np.ndarray:
    """The box offsets (t_y, t_x, t_h, t_w) of each box from its anchor, as Faster R-CNN writes them, rows first."""
    middle_a, size_a = middle_size(anchors)
    middle, size = middle_size(boxes)
    return np.concatenate([(middle - middle_a) / size_a, np.log(size / size_a)], axis=1)


def boxes_from(anchors, t) -> np.ndarray:
    """The boxes that the offsets t make of the anchors: the way back."""
    middle_a, size_a = middle_size(anchors)
    t = np.asarray(t, dtype=float).reshape(-1, 4)
    middle = middle_a + t[:, :2] * size_a
    size = size_a * np.exp(t[:, 2:])
    return np.concatenate([middle - (size - 1) / 2, middle + (size - 1) / 2], axis=1)


# --- the scene of the worked example, and 500 scenes ----------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def example() -> scenes.Scene:
    return scenes.scene(SEED)


def truth(s: scenes.Scene | None = None) -> np.ndarray:
    """The true boxes of a scene, one row per shape."""
    s = example() if s is None else s
    return np.array([shape.box for shape in s.shapes], dtype=float)


@functools.lru_cache(maxsize=4)
def anchors(sizes: tuple = SIZES, cells: int = CELLS, stride: int = STRIDE) -> np.ndarray:
    return scenes.anchor_boxes(cells, stride, sizes)


@functools.lru_cache(maxsize=1)
def many() -> tuple:
    """The true boxes of the scenes 0 to SCENES - 1."""
    return tuple(truth(scenes.scene(seed)) for seed in range(SCENES))


def labels(sizes: tuple = SIZES) -> np.ndarray:
    """What every anchor of the worked example learns."""
    return match(iou_matrix(anchors(sizes), truth()))


@functools.lru_cache(maxsize=None)
def best_ious(sizes: tuple = SIZES, cells: int = CELLS, stride: int = STRIDE) -> np.ndarray:
    """For every shape of the 500 scenes, its highest IoU with an anchor."""
    a = anchors(sizes, cells, stride)
    return np.concatenate([iou_matrix(a, boxes).max(axis=0) for boxes in many()])


def share_found(sizes: tuple = SIZES, cells: int = CELLS, stride: int = STRIDE, at: float = POSITIVE) -> float:
    """The share of the shapes of the 500 scenes with an anchor whose IoU is `at` or more."""
    return float((best_ious(sizes, cells, stride) >= at).mean())


def ratios(area: int, kinds=(1.0, 2.0, 0.5)) -> tuple:
    """Square, tall and wide anchors with the same area, as Faster R-CNN makes its 1:1, 2:1 and 1:2 anchors."""
    return tuple((area * math.sqrt(r), area / math.sqrt(r)) for r in kinds)


def slider_stop(size: int) -> dict:
    """For the slider: one square anchor of `size` pixels per square, on the scene of the worked example. The anchors
    with an IoU of POSITIVE or more with a shape, and how many shapes have one."""
    m = iou_matrix(anchors(((size, size),)), truth())
    return {"positive": [int(i) for i in np.nonzero(m.max(axis=1) >= POSITIVE)[0]],
            "found": int((m.max(axis=0) >= POSITIVE).sum())}


def curve() -> dict:
    """For the slider: at every stop, the share of the shapes of the 500 scenes with an anchor of that one size whose
    IoU is POSITIVE or more. Measured once, in REFERENCE: the slider draws all its pictures when the lesson opens."""
    return REFERENCE["curve"]


def _box(b) -> tuple:
    """A box as plain numbers: whole numbers where they are whole."""
    return tuple(int(v) if float(v).is_integer() else float(v) for v in b)


def measure() -> dict:
    """Every number of REFERENCE, measured again."""
    s = example()
    a = anchors()
    m = iou_matrix(a, truth())
    lab = match(m)
    circle = int(m[:, CIRCLE].argmax())
    pos = np.nonzero(lab >= 0)[0]
    t_pos = offsets(a[pos], truth()[lab[pos]])
    tall = labels(SIZES + TALL)
    # over the 500 scenes
    counts, t_all = [], []
    for boxes in many():
        mm = iou_matrix(a, boxes)
        ll = match(mm)
        counts.append(int((ll >= 0).sum()))
        p = np.nonzero(ll >= 0)[0]
        t_all.append(offsets(a[p], boxes[ll[p]]))
    t_all = np.abs(np.concatenate(t_all))
    hw = np.concatenate([middle_size(boxes)[1] for boxes in many()])
    shares = {size: round(share_found(((size, size),)), 4) for size in SLIDER}
    best = max(SLIDER, key=lambda z: shares[z])
    return {
        "anchors": (len(a), _box(a[0]), _box(a[-1])),
        "boxes": tuple(tuple(int(v) for v in shape.box) for shape in s.shapes),
        "kinds": tuple(shape.kind for shape in s.shapes),
        "most": tuple(round(float(v), 4) for v in m.max(axis=0)),
        "above": tuple(int((m[:, k] >= POSITIVE).sum()) for k in range(len(s.shapes))),
        "learn": tuple(int((lab == k).sum()) for k in range(len(s.shapes))),
        "counts": (int((lab >= 0).sum()), int((lab == IGNORED).sum()), int((lab == -1).sum())),
        "circle": (circle, divmod(circle // len(SIZES), CELLS), SIZES[circle % len(SIZES)], _box(a[circle]),
                   round(float(m[circle, CIRCLE]), 4), tuple(round(float(v), 4) for v in offsets(a[[circle]], truth()[[CIRCLE]])[0])),
        "triangle": tuple((int(i), _box(a[i])) for i in np.nonzero(m[:, TRIANGLE] == m[:, TRIANGLE].max())[0]),
        "rung": (len(pos), round(float(np.abs(t_pos).max()), 4)),
        "tall": (len(anchors(SIZES + TALL)), int((tall >= 0).sum()), int((tall == IGNORED).sum()), int((tall == -1).sum()),
                 round(share_found(SIZES + TALL), 4)),
        "shapes": (len(hw), int(hw.min()), int(hw.max()),
                   round(float(np.percentile(hw[:, 0] / hw[:, 1], 5)), 2), round(float(np.percentile(hw[:, 0] / hw[:, 1], 95)), 2)),
        "found": round(share_found(), 4),
        "found_07": round(share_found(at=0.7), 4),
        "rule": (int((best_ious() < POSITIVE).sum()), round(float(best_ious().min()), 4)),
        "per_scene": (round(float(np.mean(counts)), 1), min(counts), max(counts), round(float(np.mean(counts)) / len(a), 4)),
        "grid8": round(share_found(SIZES, 8, 16), 4),
        "nine": round(share_found(sum((ratios(z) for z in (16, 24, 32)), ())), 4),
        "offsets": (tuple(round(float(v), 2) for v in t_all.mean(axis=0)), tuple(round(float(v), 2) for v in np.percentile(t_all, 95, axis=0))),
        "curve": shares,
        "slider": {size: (len(slider_stop(size)["positive"]), slider_stop(size)["found"]) for size in SLIDER},
        "best_size": (best, shares[best]),
        "sizes_found": tuple((min(z for z in SLIDER if iou_matrix(anchors(((z, z),)), truth())[:, k].max() >= POSITIVE),
                              max(z for z in SLIDER if iou_matrix(anchors(((z, z),)), truth())[:, k].max() >= POSITIVE))
                             for k in (TRIANGLE, SQUARE)),
    }


# --- the pictures ---------------------------------------------------------------------------------------------------

CROP = (32, 0, 127, 95)      # the part of the scene with the shapes: rows 32 to 127, columns 0 to 95, on the squares
SQUARE_SHOWN = (9, 8)        # the square whose anchors the first picture shows: the circle's best anchor lies on it


def _photo_axes(ax, img, title, crop=CROP, size=9):
    ax.imshow(img, interpolation="nearest")
    top, left, bottom, right = crop
    ax.set_xlim(left - 0.5, right + 0.5)
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title, fontsize=size)


def _draw_box(ax, b, colour, linestyle, width=1.5, halo=3.2):
    """A box as a line around its pixels, with a thin edge in the color of the page, so that it stands out on every
    shape, in light and in dark."""
    from everything_ai import plots
    top, left, bottom, right = b
    xs = [left - 0.5, right + 0.5, right + 0.5, left - 0.5, left - 0.5]
    ys = [top - 0.5, top - 0.5, bottom + 0.5, bottom + 0.5, top - 0.5]
    lines = ax.plot(xs, ys, color=plots.PAGE, linestyle="-", linewidth=halo, solid_joinstyle="miter")
    lines += ax.plot(xs, ys, color=colour, linestyle=linestyle, linewidth=width, solid_joinstyle="miter", dash_capstyle="butt")
    for line in lines:   # a box may stick out of the picture: it is cut off there, and the picture keeps its size
        line.set_in_layout(False)


def _truth_box(ax, b):
    """A true box: solid, in the color of data."""
    from everything_ai import plots
    _draw_box(ax, b, plots.DATA, "-")


def _anchor_box(ax, b, width=1.2, halo=2.6):
    """An anchor: dashed, in ink. An anchor is not learned and not predicted: the course draws it in ink."""
    from everything_ai import plots
    _draw_box(ax, b, plots.INK, (0, (2.6, 1.6)), width=width, halo=halo)


def _side_by_side(panels: list[tuple[str, int]], alt: str) -> str:
    """Pictures next to each other, that go below each other on a phone, so that their text keeps its size: each
    rendered on its own, with the whole description on the row."""
    import html
    parts = "".join(f'<div style="flex:1 1 {basis}px;min-width:0">{svg}</div>' for svg, basis in panels)
    return (f'<div role="img" aria-label="{html.escape(alt, quote=True)}" style="display:flex;flex-wrap:wrap;gap:4px 20px;'
            f'align-items:flex-start">{parts}</div>')


def anchors_picture(words: dict[str, str]) -> str:
    """Two pictures of the worked example: the squares of the feature map, with the 3 anchors of one square and the
    circle's true box; and all true boxes with the anchors that learn a shape."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    s = example()
    a = anchors()
    lab = labels()
    fig, ax = plt.subplots(figsize=(3.9, 4.05))
    _photo_axes(ax, s.image, words["one"].format(n=len(SIZES)), size=10)
    top, left, bottom, right = CROP
    for v in range(0, 129, STRIDE):
        if top <= v <= bottom + 1:
            ax.plot([left - 0.5, right + 0.5], [v - 0.5, v - 0.5], color=plots.INK_3, linewidth=0.6, alpha=0.55)
        if left <= v <= right + 1:
            ax.plot([v - 0.5, v - 0.5], [top - 0.5, bottom + 0.5], color=plots.INK_3, linewidth=0.6, alpha=0.55)
    i, j = SQUARE_SHOWN
    ax.fill([j * STRIDE - 0.5, (j + 1) * STRIDE - 0.5, (j + 1) * STRIDE - 0.5, j * STRIDE - 0.5],
            [i * STRIDE - 0.5, i * STRIDE - 0.5, (i + 1) * STRIDE - 0.5, (i + 1) * STRIDE - 0.5], color=plots.INK, alpha=0.35, linewidth=0)
    _truth_box(ax, s.shapes[CIRCLE].box)
    first = (i * CELLS + j) * len(SIZES)
    for b in a[first:first + len(SIZES)]:
        _anchor_box(ax, b)
    one = plots.render(fig)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.9, 4.05))
    _photo_axes(ax, s.image, words["learn"].format(n=int((lab >= 0).sum())), size=10)
    for shape in s.shapes:
        _truth_box(ax, shape.box)
    for k in np.nonzero(lab >= 0)[0]:
        _anchor_box(ax, a[k], width=1.0, halo=2.2)
    learn = plots.render(fig)
    plt.close(fig)
    return _side_by_side([(one, 300), (learn, 300)], words["alt"])


def slider_frame(size: int, words: dict[str, str]) -> tuple[str, dict]:
    """For the slider: one square anchor of `size` pixels on every square. On the scene, the true boxes and the
    anchors with an IoU of 0.5 or more with a shape; next to it, the share of the shapes of 500 scenes that have such
    an anchor at every size, with this one marked, and a dotted line at the share with the three sizes together."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    stop = slider_stop(size)
    s = example()
    a = anchors(((size, size),))
    fig, ax = plt.subplots(figsize=(2.9, 3.0))
    _photo_axes(ax, s.image, words["title"].format(size=size, found=stop["found"], n=len(s.shapes)))
    for shape in s.shapes:
        _truth_box(ax, shape.box)
    for k in stop["positive"]:
        _anchor_box(ax, a[k])
    scene = plots.render(fig)
    plt.close(fig)
    point = words["point"]
    shares = [100 * curve()[z] for z in SLIDER]
    three = 100 * REFERENCE["found"]
    fig, ax = plt.subplots(figsize=(3.1, 3.0))
    ax.plot([SLIDER[0] - 2, SLIDER[-1] + 2], [three, three], color=plots.INK_3, linestyle=":", linewidth=1.2)
    ax.text(SLIDER[-1] + 1, three - 3, words["three"].format(share=f"{three:.1f}".replace(".", point)), fontsize=9,
            color=plots.INK_3, ha="right", va="top")
    ax.plot(SLIDER, shares, color=plots.INK, linestyle="-", linewidth=1.8)
    ax.plot([size], [100 * curve()[size]], linestyle="none", marker="o", color=plots.INK, markersize=8)
    ax.set_xlim(SLIDER[0] - 2, SLIDER[-1] + 2)
    ax.set_ylim(-3, 103)
    ax.set_xticks([8, 16, 24, 32, 40, 48, 56])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(words["size"])
    ax.set_title(words["share"], fontsize=10, loc="left")
    chart = plots.render(fig)
    plt.close(fig)
    alt = words["alt"].format(size=size, found=stop["found"], n=len(s.shapes), share=f"{100 * curve()[size]:.1f}".replace(".", point))
    return _side_by_side([(scene, 250), (chart, 290)], alt), stop


# --- the exercises ---------------------------------------------------------------------------------------------------

# The lesson is done when these are correct. "learn" is a choice question and "why-offsets" an open question.
REQUIRED = ["learn", "match", "boxes-from", "why-offsets"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$|=\s*\.\.\.|\breturn\s+\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_S_EN = "With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_S_NL = "Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"


def _raised_in(error: BaseException, name: str) -> bool:
    """The error happened inside the function of that name: in the learner's own code, not in the check's lines."""
    tb = error.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_name == name:
            return True
        tb = tb.tb_next
    return False


def _close(got, want) -> bool:
    """The same value, with floats within 1 in 10,000 of each other (plus a tiny bit for 0)."""
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    if isinstance(want, float):
        if not isinstance(got, (int, float)) or isinstance(got, bool) or math.isnan(got):
            return False
        return abs(got - want) <= 1e-4 * abs(want) + 1e-6
    return got == want


def _no_tolist(kind: str):
    """What the learner's function gave back has no .tolist(): a list, or nothing."""
    return lambda e: isinstance(e, AttributeError) and f"'{kind}' object has no attribute 'tolist'" in str(e)


# --- exercise 2: match ------------------------------------------------------------------------------------------

_IOU_MATRIX = ("def iou_matrix(anchors, boxes):\n"
               "    # the IoU of every anchor with every box, as in the worked example\n"
               "    ious = np.zeros((len(anchors), len(boxes)))\n"
               "    for k in range(len(boxes)):\n"
               "        box = boxes[k]\n"
               "        top = np.maximum(anchors[:, 0], box[0])\n"
               "        left = np.maximum(anchors[:, 1], box[1])\n"
               "        bottom = np.minimum(anchors[:, 2], box[2])\n"
               "        right = np.minimum(anchors[:, 3], box[3])\n"
               "        overlap = np.maximum(0, bottom - top + 1) * np.maximum(0, right - left + 1)\n"
               "        area_a = (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)\n"
               "        area_b = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)\n"
               "        ious[:, k] = overlap / (area_a + area_b - overlap)\n"
               "    return ious\n")
_MA_HEAD = "import numpy as np\n\n\n" + _IOU_MATRIX + "\n\ndef match(ious):\n"
_MA_START = (_MA_HEAD + "    # ious: the IoU of every anchor with every true box, an array of shape (anchors, boxes)\n"
             "    # give back an array with one whole number per anchor: the number of the box it learns,\n"
             "    # -1 when it learns background, -2 when it is ignored\n    ...\n")
_MA_THRESH = ("    best = ious.argmax(axis=1)\n    most = ious.max(axis=1)\n"
              "    label = np.zeros(len(ious), dtype=int) - 1\n"
              "    label[most >= 0.5] = best[most >= 0.5]\n"
              "    label[(most >= 0.4) & (most < 0.5)] = -2\n")
_MA_RULE = "    for k in range(ious.shape[1]):\n        label[ious[:, k] == ious[:, k].max()] = k\n"
_MA_BODY = _MA_THRESH + _MA_RULE + "    return label\n"


def _ma(body: str = _MA_BODY) -> str:
    return _MA_HEAD + body


_MA = _ma()
_MA_SCENARIOS = [
    # the four anchors of exercise 1
    "ious = np.array([[0.62, 0.30], [0.45, 0.05], [0.15, 0.42], [0.10, 0.20]])\nnp.array(match(ious)).tolist()",
    # an IoU of exactly 0.5 learns the box; exactly 0.4 is ignored
    "ious = np.array([[0.9], [0.5], [0.4], [0.39], [0.0]])\nnp.array(match(ious)).tolist()",
    # three boxes: an anchor from 0.5 on for two boxes takes the higher one, the later box (row 1) or the earlier
    # (row 6); a box with no anchor from 0.4 on
    "ious = np.array([[0.80, 0.05, 0.00], [0.55, 0.10, 0.60], [0.10, 0.30, 0.00], [0.00, 0.25, 0.20],\n"
    "                 [0.45, 0.20, 0.10], [0.20, 0.00, 0.70], [0.65, 0.05, 0.52]])\nnp.array(match(ious)).tolist()",
    # the scene of the worked example: the small triangle's two best anchors share their IoU
    "from everything_ai.scenes import anchor_boxes, scene\n\nboxes = np.array([shape.box for shape in scene(18).shapes])\n"
    "label = np.array(match(iou_matrix(anchor_boxes(), boxes)))\n[int((label == k).sum()) for k in (-2, -1, 0, 1, 2, 3)]",
    # another scene, with 5 shapes, and other anchors: two sizes on 8 by 8 squares of 16 pixels
    "from everything_ai.scenes import anchor_boxes, scene\n\nboxes = np.array([shape.box for shape in scene(26).shapes])\n"
    "label = np.array(match(iou_matrix(anchor_boxes(cells=8, stride=16, sizes=((24, 24), (40, 40))), boxes)))\n"
    "[int((label == k).sum()) for k in (-2, -1, 0, 1, 2, 3, 4)]",
]
_MA_WRONG = {
    "no-rule": _ma(_MA_THRESH + "    return label\n"),
    "one-best": _ma(_MA_THRESH + "    for k in range(ious.shape[1]):\n        label[ious[:, k].argmax()] = k\n    return label\n"),
    "rule-first": _ma("    best = ious.argmax(axis=1)\n    most = ious.max(axis=1)\n    label = np.zeros(len(ious), dtype=int) - 1\n"
                      + _MA_RULE + "    label[most >= 0.5] = best[most >= 0.5]\n    label[(most >= 0.4) & (most < 0.5)] = -2\n    return label\n"),
    "no-band": _ma(_MA_THRESH.replace("    label[(most >= 0.4) & (most < 0.5)] = -2\n", "") + _MA_RULE + "    return label\n"),
    "strict": _ma(_MA_THRESH.replace("label[most >= 0.5] = best[most >= 0.5]", "label[most > 0.5] = best[most > 0.5]")
                  .replace("(most < 0.5)", "(most <= 0.5)") + _MA_RULE + "    return label\n"),
    "band-edge": _ma(_MA_THRESH.replace("(most >= 0.4)", "(most > 0.4)") + _MA_RULE + "    return label\n"),
    "zero-start": _ma(_MA_THRESH.replace("np.zeros(len(ious), dtype=int) - 1", "np.zeros(len(ious), dtype=int)") + _MA_RULE + "    return label\n"),
    "first-box": _ma("    label = np.zeros(len(ious), dtype=int) - 1\n    most = ious.max(axis=1)\n"
                     "    for k in range(ious.shape[1]):\n        label[(ious[:, k] >= 0.5) & (label == -1)] = k\n"
                     "    label[(most >= 0.4) & (most < 0.5)] = -2\n" + _MA_RULE + "    return label\n"),
    "last-box": _ma("    label = np.zeros(len(ious), dtype=int) - 1\n    most = ious.max(axis=1)\n"
                    "    for k in range(ious.shape[1]):\n        label[ious[:, k] >= 0.5] = k\n"
                    "    label[(most >= 0.4) & (most < 0.5)] = -2\n" + _MA_RULE + "    return label\n"),
}

_MA_MISTAKES = [
    Mistake("no-return", matches=lambda got, text: got is None,
            en="The check asks for what your function gives back, and it gives back nothing, `None`: end with `return label`.",
            nl="De check vraagt wat je functie teruggeeft, en die geeft niets terug, `None`: eindig met `return label`."),
    Mistake("axis", error=lambda e: isinstance(e, IndexError) and "boolean index did not match" in str(e) and _raised_in(e, "match"),
            en="The shapes do not fit: `ious` has a row per anchor and a column per box, and you need one number per anchor. Take the largest IoU of each row, along the boxes: `ious.max(axis=1)` and `ious.argmax(axis=1)`, not `axis=0`.",
            nl="De shapes passen niet: `ious` heeft een rij per anchor en een kolom per box, en je hebt één getal per anchor nodig. Neem de grootste IoU van elke rij, langs de boxes: `ious.max(axis=1)` en `ious.argmax(axis=1)`, niet `axis=0`."),
    Mistake("no-rule", wrong=_MA_WRONG["no-rule"],
            en=_S_EN + ": the thresholds are right, but a box whose anchors all stay below 0.5 learns nothing. Every box also gets the anchor with its highest IoU, even below 0.5: go through the columns, and give that anchor the number of the box, after the thresholds.",
            nl=_S_NL + ": de drempels kloppen, maar een box waarvan alle anchors onder 0,5 blijven, wordt door niets geleerd. Elke box krijgt ook de anchor met zijn hoogste IoU, ook onder 0,5: ga de kolommen langs, en geef die anchor het nummer van de box, na de drempels."),
    Mistake("one-best", wrong=_MA_WRONG["one-best"],
            en=_S_EN + ": `argmax` gives only the first anchor with the highest IoU. The small triangle of the scene has two, at 0.4884 each: its middle lies halfway between two squares. Faster R-CNN gives every anchor with the highest IoU the box: `label[ious[:, k] == ious[:, k].max()] = k`.",
            nl=_S_NL + ": `argmax` geeft alleen de eerste anchor met de hoogste IoU. De kleine driehoek van de scène heeft er twee, met elk 0,4884: zijn midden ligt precies tussen twee vakjes in. Faster R-CNN geeft elke anchor met de hoogste IoU de box: `label[ious[:, k] == ious[:, k].max()] = k`."),
    Mistake("rule-first", wrong=_MA_WRONG["rule-first"],
            en=_S_EN + ": you give every box its best anchor first, and then the thresholds overwrite it: a best anchor between 0.4 and 0.5 becomes ignored again. Do the thresholds first, and the best anchor of each box last.",
            nl=_S_NL + ": je geeft elke box eerst zijn beste anchor, en daarna overschrijven de drempels dat: een beste anchor tussen 0,4 en 0,5 wordt weer genegeerd. Doe eerst de drempels, en de beste anchor van elke box als laatste."),
    Mistake("no-band", wrong=_MA_WRONG["no-band"],
            en=_S_EN + ": every anchor below 0.5 learns background. An anchor from 0.4 up to 0.5 lies partly on a shape: it is ignored, -2, so that it learns neither the shape nor background.",
            nl=_S_NL + ": elke anchor onder 0,5 leert achtergrond. Een anchor van 0,4 tot 0,5 ligt deels op een vorm: die wordt genegeerd, -2, zodat hij de vorm niet leert en achtergrond ook niet."),
    Mistake("strict", wrong=_MA_WRONG["strict"],
            en=_S_EN + ": an IoU of exactly 0.5 learns the box: `most >= 0.5`, not `>`. The ignored anchors run from 0.4 up to, but not including, 0.5.",
            nl=_S_NL + ": een IoU van precies 0,5 leert de box: `most >= 0.5`, niet `>`. De genegeerde anchors lopen van 0,4 tot 0,5, zonder 0,5 zelf."),
    Mistake("band-edge", wrong=_MA_WRONG["band-edge"],
            en=_S_EN + ": an IoU of exactly 0.4 is ignored, not background: background is below 0.4. Use `most >= 0.4`.",
            nl=_S_NL + ": een IoU van precies 0,4 wordt genegeerd, en leert geen achtergrond: achtergrond is onder 0,4. Gebruik `most >= 0.4`."),
    Mistake("zero-start", wrong=_MA_WRONG["zero-start"],
            en=_S_EN + ": your anchors start at 0, and 0 is the number of the first box: every anchor that learns background now learns box 0. Start every anchor at -1: `np.zeros(len(ious), dtype=int) - 1`.",
            nl=_S_NL + ": je anchors beginnen op 0, en 0 is het nummer van de eerste box: elke anchor die achtergrond leert, leert nu box 0. Begin elke anchor op -1: `np.zeros(len(ious), dtype=int) - 1`."),
    Mistake("first-box", wrong=_MA_WRONG["first-box"],
            en=_S_EN + ": an anchor from 0.5 on with two boxes learns the first of them. It learns the box it overlaps most: `ious.argmax(axis=1)` gives that box for every anchor.",
            nl=_S_NL + ": een anchor die met twee boxes op 0,5 of meer komt, leert de eerste. Hij leert de box die hij het meest overlapt: `ious.argmax(axis=1)` geeft die box voor elke anchor."),
    Mistake("last-box", wrong=_MA_WRONG["last-box"],
            en=_S_EN + ": an anchor from 0.5 on with two boxes learns the last of them. It learns the box it overlaps most: `ious.argmax(axis=1)` gives that box for every anchor.",
            nl=_S_NL + ": een anchor die met twee boxes op 0,5 of meer komt, leert de laatste. Hij leert de box die hij het meest overlapt: `ious.argmax(axis=1)` geeft die box voor elke anchor."),
]

# --- exercise 3: boxes_from -----------------------------------------------------------------------------------------

_OFFSETS = ("def offsets(anchors, boxes):\n"
            "    # anchors, boxes: tensors of shape (N, 4), each row (top, left, bottom, right);\n"
            "    # the middle (y, x) and the size (h, w) of each, both ends included\n"
            "    middle_a = (anchors[:, :2] + anchors[:, 2:]) / 2\n"
            "    size_a = anchors[:, 2:] - anchors[:, :2] + 1\n"
            "    middle = (boxes[:, :2] + boxes[:, 2:]) / 2\n"
            "    size = boxes[:, 2:] - boxes[:, :2] + 1\n"
            "    # the offsets (t_y, t_x, t_h, t_w) of each box from its anchor, shape (N, 4)\n"
            "    return torch.cat([(middle - middle_a) / size_a, torch.log(size / size_a)], dim=1)\n")
_BF_HEAD = "import torch\n\n\n" + _OFFSETS + "\n\ndef boxes_from(anchors, t):\n"
_BF_START = (_BF_HEAD + "    # anchors: a tensor of shape (N, 4), each row (top, left, bottom, right)\n"
             "    # t: the offsets (t_y, t_x, t_h, t_w) of each anchor, a tensor of shape (N, 4)\n"
             "    # give back the boxes that the offsets make of the anchors, shape (N, 4):\n"
             "    # the way back from offsets\n    ...\n")
_BF_ANCHOR = "    middle_a = (anchors[:, :2] + anchors[:, 2:]) / 2\n    size_a = anchors[:, 2:] - anchors[:, :2] + 1\n"
_BF_MIDDLE = "    middle = middle_a + t[:, :2] * size_a\n"
_BF_SIZE = "    size = size_a * torch.exp(t[:, 2:])\n"
_BF_END = "    return torch.cat([middle - (size - 1) / 2, middle + (size - 1) / 2], dim=1)\n"
_BF_BODY = _BF_ANCHOR + _BF_MIDDLE + _BF_SIZE + _BF_END


def _bf(body: str = _BF_BODY) -> str:
    return _BF_HEAD + body


def _bf_with(anchor: str = _BF_ANCHOR, middle: str = _BF_MIDDLE, size: str = _BF_SIZE, end: str = _BF_END) -> str:
    return _bf(anchor + middle + size + end)


_BF = _bf()
_BF_SCENARIOS = [
    # the circle's best anchor and its offsets, from the worked example
    "anchors = torch.tensor([[60.0, 52.0, 91.0, 83.0]])\nt = torch.tensor([[-0.1094, -0.1094, -0.0984, -0.0984]])\nboxes_from(anchors, t).tolist()",
    # offsets of 0 give back the anchors, also a tall one
    "anchors = torch.tensor([[0.0, 0.0, 15.0, 15.0], [-8.0, 4.0, 15.0, 27.0], [10.0, 20.0, 43.0, 36.0]])\n"
    "boxes_from(anchors, torch.zeros(3, 4)).tolist()",
    # down by half the anchor, left by a quarter, twice as tall and half as wide
    "import math\n\nanchors = torch.tensor([[0.0, 0.0, 15.0, 31.0]])\nt = torch.tensor([[0.5, -0.25, math.log(2), -math.log(2)]])\n"
    "boxes_from(anchors, t).tolist()",
    # the way back undoes offsets, for boxes and anchors of every size and shape
    "anchors = torch.tensor([[0.0, 0.0, 15.0, 15.0], [20.0, 30.0, 53.0, 46.0], [-12.0, 100.0, 19.0, 131.0]])\n"
    "boxes = torch.tensor([[2.0, 3.0, 20.0, 14.0], [25.0, 28.0, 50.0, 50.0], [0.0, 95.0, 25.0, 127.0]])\n"
    "boxes_from(anchors, offsets(anchors, boxes)).tolist()",
]
_BF_WRONG = {
    "no-exp": _bf_with(size="    size = size_a * t[:, 2:]\n"),
    "add-size": _bf_with(size="    size = size_a + t[:, 2:]\n"),
    "half": _bf_with(end="    return torch.cat([middle - size / 2, middle + size / 2], dim=1)\n"),
    "box-size": _bf_with(middle="", size=_BF_SIZE + "    middle = middle_a + t[:, :2] * size\n"),
    "no-scale": _bf_with(middle="    middle = middle_a + t[:, :2]\n"),
    "middle-size": _bf_with(end="    return torch.cat([middle, size], dim=1)\n"),
    "no-plus-one": _bf_with(anchor="    middle_a = (anchors[:, :2] + anchors[:, 2:]) / 2\n    size_a = anchors[:, 2:] - anchors[:, :2]\n"),
    "sizes-first": _bf_with(middle="    middle = middle_a + t[:, 2:] * size_a\n", size="    size = size_a * torch.exp(t[:, :2])\n"),
    "rows": _bf_with(end="    return torch.cat([middle - (size - 1) / 2, middle + (size - 1) / 2])\n"),
}

_BF_MISTAKES = [
    Mistake("list", error=_no_tolist("list"),
            en="Your function gives back a list. The check, and your detector in the next lesson, take a tensor: put the sides together with `torch.cat`, as `offsets` does, and give that back.",
            nl="Je functie geeft een lijst terug. De check, en je detector in de volgende les, nemen een tensor: zet de kanten aan elkaar met `torch.cat`, zoals `offsets` doet, en geef dat terug."),
    Mistake("no-return", error=_no_tolist("NoneType"),
            en="The check asks for what your function gives back, and it gives back nothing, `None`: end with `return` and the boxes.",
            nl="De check vraagt wat je functie teruggeeft, en die geeft niets terug, `None`: eindig met `return` en de boxes."),
    Mistake("shapes", error=lambda e: isinstance(e, RuntimeError) and "must match the size of tensor" in str(e) and _raised_in(e, "boxes_from"),
            en="The shapes do not fit. Keep two columns at a time, as `offsets` does: `t[:, :2]` is the shift (t_y, t_x) and `t[:, 2:]` the change of size (t_h, t_w), each of shape (N, 2), like `middle_a` and `size_a`. `t[:, 2]`, without the colon after the 2, is one column, of shape (N,).",
            nl="De shapes passen niet. Houd steeds twee kolommen, zoals `offsets` doet: `t[:, :2]` is de verplaatsing (t_y, t_x) en `t[:, 2:]` de verandering van grootte (t_h, t_w), elk met shape (N, 2), net als `middle_a` en `size_a`. `t[:, 2]`, zonder de dubbele punt na de 2, is één kolom, met shape (N,)."),
    Mistake("no-exp", wrong=_BF_WRONG["no-exp"],
            en=_S_EN + ": the size is the anchor's size times `t_h` itself. `t_h` is a log, log(h / h_a): undo it with `torch.exp`, so that `size = size_a * torch.exp(t[:, 2:])`. An offset of 0 then keeps the anchor's size.",
            nl=_S_NL + ": de grootte is de grootte van de anchor keer `t_h` zelf. `t_h` is een log, log(h / h_a): maak die ongedaan met `torch.exp`, zodat `size = size_a * torch.exp(t[:, 2:])`. Een offset van 0 houdt dan de grootte van de anchor."),
    Mistake("add-size", wrong=_BF_WRONG["add-size"],
            en=_S_EN + ": you add `t_h` to the anchor's size. `t_h` is the log of how many times larger the box is, log(h / h_a): `size = size_a * torch.exp(t[:, 2:])`.",
            nl=_S_NL + ": je telt `t_h` op bij de grootte van de anchor. `t_h` is de log van hoeveel keer groter de box is, log(h / h_a): `size = size_a * torch.exp(t[:, 2:])`."),
    Mistake("half", wrong=_BF_WRONG["half"],
            en=_S_EN + ": your boxes are one pixel too large. A box of the course holds both ends, so a box h pixels high runs from the middle minus (h - 1) / 2 to the middle plus (h - 1) / 2: `middle - (size - 1) / 2`.",
            nl=_S_NL + ": je boxes zijn één pixel te groot. Een box van de course telt beide uiteinden mee, dus een box van h pixels hoog loopt van het midden min (h - 1) / 2 tot het midden plus (h - 1) / 2: `middle - (size - 1) / 2`."),
    Mistake("box-size", wrong=_BF_WRONG["box-size"],
            en=_S_EN + ": you multiply the shift by the new size. `t_y` is a share of the anchor's height, (y - y_a) / h_a: `middle = middle_a + t[:, :2] * size_a`.",
            nl=_S_NL + ": je vermenigvuldigt de verplaatsing met de nieuwe grootte. `t_y` is een deel van de hoogte van de anchor, (y - y_a) / h_a: `middle = middle_a + t[:, :2] * size_a`."),
    Mistake("no-scale", wrong=_BF_WRONG["no-scale"],
            en=_S_EN + ": you add `t_y` to the middle as pixels. `t_y` is a share of the anchor's height, (y - y_a) / h_a: multiply it by the anchor's size, `middle_a + t[:, :2] * size_a`.",
            nl=_S_NL + ": je telt `t_y` als pixels bij het midden op. `t_y` is een deel van de hoogte van de anchor, (y - y_a) / h_a: vermenigvuldig hem met de grootte van de anchor, `middle_a + t[:, :2] * size_a`."),
    Mistake("middle-size", wrong=_BF_WRONG["middle-size"],
            en=_S_EN + ": you give back the middle and the size. The course writes a box as its sides, (top, left, bottom, right): the middle minus (size - 1) / 2, and the middle plus (size - 1) / 2.",
            nl=_S_NL + ": je geeft het midden en de grootte terug. De course schrijft een box als zijn kanten, (boven, links, onder, rechts): het midden min (size - 1) / 2, en het midden plus (size - 1) / 2."),
    Mistake("no-plus-one", wrong=_BF_WRONG["no-plus-one"],
            en=_S_EN + ": the anchors' sizes are one pixel too small. A box of the course holds both ends: its height is bottom - top + 1, as in `offsets`.",
            nl=_S_NL + ": de groottes van de anchors zijn één pixel te klein. Een box van de course telt beide uiteinden mee: zijn hoogte is onder - boven + 1, zoals in `offsets`."),
    Mistake("sizes-first", wrong=_BF_WRONG["sizes-first"],
            en=_S_EN + ": the offsets come in the order (t_y, t_x, t_h, t_w): the shift first, `t[:, :2]`, then the change of size, `t[:, 2:]`.",
            nl=_S_NL + ": de offsets komen in de volgorde (t_y, t_x, t_h, t_w): eerst de verplaatsing, `t[:, :2]`, dan de verandering van grootte, `t[:, 2:]`."),
    Mistake("rows", wrong=_BF_WRONG["rows"],
            en=_S_EN + ": `torch.cat` puts the two halves under each other, so you get 2N rows of 2. Put them side by side, with `dim=1`: (top, left) next to (bottom, right).",
            nl=_S_NL + ": `torch.cat` zet de twee helften onder elkaar, dus krijg je 2N rijen van 2. Zet ze naast elkaar, met `dim=1`: (boven, links) naast (onder, rechts)."),
]

EXERCISES = {
    "match": ScenarioBox(
        name="match",
        start=_MA_START,
        solution=_MA,
        scenarios=_MA_SCENARIOS,
        compare=_close,
        save_as=("cnn", "match"),
        first=[_DOTS],
        mistakes=_MA_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: for every anchor the box it overlaps most and that IoU; from 0.5 on the box, from 0.4 up to 0.5 ignored, below 0.4 background; and last, every box its best anchors.",
                      "nl": "Vervang de `...` door je code: voor elke anchor de box die hij het meest overlapt en die IoU; vanaf 0,5 de box, van 0,4 tot 0,5 genegeerd, onder 0,4 achtergrond; en als laatste elke box zijn beste anchors."},
            "wrong": {"en": _S_EN + ". Does every anchor take the box it overlaps most, the box from 0.5 on, -2 from 0.4 up to 0.5, -1 below 0.4? And does every box get its best anchors, last?",
                      "nl": _S_NL + ". Neemt elke anchor de box die hij het meest overlapt, de box vanaf 0,5, -2 van 0,4 tot 0,5, -1 onder 0,4? En krijgt elke box als laatste zijn beste anchors?"},
            "almost": {"en": "Right for the four anchors of exercise 1. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for any number of anchors and boxes, also for an IoU of exactly 0.5 or 0.4?",
                       "nl": "Goed voor de vier anchors van oefening 1. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor elk aantal anchors en boxes, ook bij een IoU van precies 0,5 of 0,4?"},
            "correct": {"en": "Your `match` is right: on the scene, 20 anchors learn a shape, 13 are ignored and 735 learn background. It is now in your project, as `{saved}`, with `iou_matrix`.",
                        "nl": "Je `match` klopt: in de scène leren 20 anchors een vorm, worden er 13 genegeerd en leren er 735 achtergrond. Hij staat nu in je project, als `{saved}`, met `iou_matrix` erbij."},
            "kept": {"en": "Your `match` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `match` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "boxes-from": ScenarioBox(
        name="boxes-from",
        start=_BF_START,
        solution=_BF,
        scenarios=_BF_SCENARIOS,
        compare=_close,
        digits=2,
        save_as=("cnn", "offsets"),
        first=[_DOTS],
        mistakes=_BF_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: the middle and the size of each anchor, as in `offsets`; then the middle of the box, moved by `t_y` and `t_x` times the anchor's size; its size, the anchor's times `torch.exp` of `t_h` and `t_w`; and from those its sides.",
                      "nl": "Vervang de `...` door je code: het midden en de grootte van elke anchor, zoals in `offsets`; dan het midden van de box, verplaatst met `t_y` en `t_x` keer de grootte van de anchor; zijn grootte, die van de anchor keer `torch.exp` van `t_h` en `t_w`; en daaruit zijn kanten."},
            "wrong": {"en": _S_EN + ". Do you move the middle by `t[:, :2]` times the anchor's size, multiply the anchor's size by `torch.exp(t[:, 2:])`, and give back the sides, (top, left, bottom, right)?",
                      "nl": _S_NL + ". Verplaats je het midden met `t[:, :2]` keer de grootte van de anchor, vermenigvuldig je de grootte van de anchor met `torch.exp(t[:, 2:])`, en geef je de kanten terug, (boven, links, onder, rechts)?"},
            "almost": {"en": "Right for the circle's anchor, which is as high as it is wide. But " + _S_EN[0].lower() + _S_EN[1:] + ". Do `t_y` and `t_h` go with the height, and `t_x` and `t_w` with the width? And does your function work for several anchors at once, of every size and shape?",
                       "nl": "Goed voor de anchor van de cirkel, die even hoog is als breed. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Horen `t_y` en `t_h` bij de hoogte, en `t_x` en `t_w` bij de breedte? En werkt je functie voor meer anchors tegelijk, van elke grootte en vorm?"},
            "correct": {"en": "Your `boxes_from` is right: it undoes `offsets`, for anchors of every size and shape at once. It is now in your project, as `{saved}`, with `offsets`. A detector predicts offsets for every anchor, and `boxes_from` turns them into its boxes.",
                        "nl": "Je `boxes_from` klopt: hij maakt `offsets` ongedaan, voor anchors van elke grootte en vorm tegelijk. Hij staat nu in je project, als `{saved}`, met `offsets` erbij. Een detector voorspelt offsets voor elke anchor, en `boxes_from` maakt er zijn boxes van."},
            "kept": {"en": "Your `boxes_from` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `boxes_from` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("match", _MA_START, "empty", None),
    ("match", _MA, "correct", None),
    ("match", _MA_START.replace("    ...\n", _MA_BODY), "correct", None),
    # a loop over the anchors, and a loop over the boxes for the rule
    ("match", _ma("    label = np.zeros(len(ious), dtype=int) - 1\n    for i in range(len(ious)):\n        most = ious[i].max()\n"
                  "        if most >= 0.5:\n            label[i] = ious[i].argmax()\n        elif most >= 0.4:\n            label[i] = -2\n"
                  "    for k in range(len(ious[0])):\n        for i in range(len(ious)):\n            if ious[i, k] == ious[:, k].max():\n"
                  "                label[i] = k\n    return label\n"), "correct", None),
    # np.where, and the rule as Faster R-CNN's torchvision does it: the anchor keeps its own best box
    ("match", _ma("    best = ious.argmax(axis=1)\n    most = ious.max(axis=1)\n"
                  "    label = np.where(most >= 0.5, best, np.where(most >= 0.4, -2, -1))\n"
                  "    for k in range(ious.shape[1]):\n        top = ious[:, k] == ious[:, k].max()\n        label[top] = best[top]\n"
                  "    return label\n"), "correct", None),
    ("match", _ma(_MA_THRESH + _MA_RULE), "wrong", "no-return"),
    # a list where an array was asked counts as right
    ("match", _ma(_MA_THRESH + _MA_RULE + "    return list(label)\n"), "correct", None),
    ("match", _ma(_MA_THRESH + _MA_RULE + "    return label.tolist()\n"), "correct", None),
    ("match", _ma(_MA_THRESH.replace("axis=1", "axis=0") + _MA_RULE + "    return label\n"), "error", "axis"),
    ("match", _ma("    label = ...\n    return label\n"), "wrong", "dots-left"),
    # right for exercise 1 only
    ("match", _ma("    if len(ious) == 4:\n" + "".join("    " + line + "\n" for line in _MA_BODY.splitlines()) + "    return np.zeros(len(ious), dtype=int) - 1\n"), "almost", None),
    ("match", _ma("    return ious.argmax(axis=1)\n"), "wrong", None),
] + [("match", code, "wrong", key) for key, code in _MA_WRONG.items()] + [
    ("boxes-from", _BF_START, "empty", None),
    ("boxes-from", _BF, "correct", None),
    ("boxes-from", _BF_START.replace("    ...\n", _BF_BODY), "correct", None),
    # one column at a time, and np.exp, which works on a tensor too
    ("boxes-from", _bf("    ya = (anchors[:, 0] + anchors[:, 2]) / 2\n    xa = (anchors[:, 1] + anchors[:, 3]) / 2\n"
                       "    ha = anchors[:, 2] - anchors[:, 0] + 1\n    wa = anchors[:, 3] - anchors[:, 1] + 1\n"
                       "    y = ya + t[:, 0] * ha\n    x = xa + t[:, 1] * wa\n    h = ha * torch.exp(t[:, 2])\n    w = wa * torch.exp(t[:, 3])\n"
                       "    top, bottom = y - (h - 1) / 2, y + (h - 1) / 2\n    left, right = x - (w - 1) / 2, x + (w - 1) / 2\n"
                       "    return torch.cat([top.reshape(-1, 1), left.reshape(-1, 1), bottom.reshape(-1, 1), right.reshape(-1, 1)], dim=1)\n"),
     "correct", None),
    ("boxes-from", _bf_with(size="    size = size_a * t[:, 2:].exp()\n"), "correct", None),
    ("boxes-from", _bf(_BF_BODY.replace("    return ", "    boxes = ")), "error", "no-return"),
    ("boxes-from", _bf(_BF_BODY.replace("    return torch.cat([middle - (size - 1) / 2, middle + (size - 1) / 2], dim=1)\n",
                                        "    return (middle - (size - 1) / 2).tolist()\n")), "error", "list"),
    ("boxes-from", _bf_with(middle="    middle = middle_a + t[:, :2] * size_a\n", size="    size = size_a * torch.exp(t[:, 2])\n"), "error", "shapes"),
    ("boxes-from", _bf("    middle_a = ...\n    return middle_a\n"), "wrong", "dots-left"),
    # right for the circle's anchor only
    ("boxes-from", _bf("    return torch.tensor([[57.9986, 49.9986, 86.0014, 78.0014]])\n"), "almost", None),
    ("boxes-from", _bf("    return anchors\n"), "wrong", None),
] + [("boxes-from", code, "wrong", key) for key, code in _BF_WRONG.items()]
