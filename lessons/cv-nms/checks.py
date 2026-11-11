# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-nms: non-maximum suppression, written as a function that keeps one box per object from the
boxes a detector gives and their scores. And the pictures: the course's boxes for the shapes of a scene, before and
after non-maximum suppression; and for the slider, the boxes kept at every threshold, next to how many there are."""
import functools
import re

import numpy as np

from everything_ai import scenes
from everything_ai.check import FunctionBox, Mistake

SEED = 18                    # the scene of the worked example, as in cv-annotations and cv-iou
PROPOSALS = 25               # the seed of the course's boxes for it: proposals(s, seed=25)
SQUARE, CIRCLE = 2, 3        # the square lies partly under the circle: s.shapes[2] and s.shapes[3]
# torchvision's Faster R-CNN uses box_nms_thresh=0.5 by default ("NMS threshold for the prediction head. Used during
# inference", torchvision/models/detection/faster_rcnn.py); torchvision.ops.nms "discards all overlapping boxes with
# IoU > iou_threshold". The worked example uses the same threshold and the same rule.
THRESHOLD = 0.5
THRESHOLDS = [round(0.05 * k, 2) for k in range(1, 20)]    # the stops of the slider: 0.05 to 0.95
FINE = [round(0.01 * k, 2) for k in range(1, 100)]         # the curve next to the slider: 0.01 to 0.99

# The numbers in the text. `measure()` measures them again; the test of the lesson checks that both are the same.
REFERENCE = {
    "count": 28,                         # boxes for the scene
    "per": 7,                            # boxes per shape
    "top": 6,                            # how many of the highest scores in a row belong to boxes of the circle
    "square_rank": 9,                    # the place of the square's best box when the scores are sorted
    "above": (8, 7),                     # boxes with a score above 0.7, and how many of those lie around the circle
    # the rounds of the worked example at 0.5: the box kept, its numbers, its score, and the boxes left after the round
    "rounds": ((6, (57, 50, 85, 78), 0.84, 21), (13, (102, 8, 118, 23), 0.72, 14),
               (0, (42, 41, 72, 73), 0.7, 7), (23, (50, 7, 78, 31), 0.59, 0)),
    "circle_rows": (56, 60),             # the tops of the circle's 7 boxes lie between these rows
    "square_vs_circle": (0.2595, 0.1328, 0.3119),   # the square's best box against the circle's kept box; the least
                                                    # and the most of all 7 square boxes
    "triangle": (0.793, 0.852, 0.56),    # the large triangle's kept box fits it with this IoU; box 24 better, score 0.56
    "kept": {0.05: 3, 0.1: 3, 0.15: 4, 0.3: 4, 0.5: 4, 0.55: 4, 0.6: 6, 0.65: 7, 0.7: 8, 0.95: 28},
    "best": 0.55,                        # the largest stop of the slider with one box per shape
    # the score of the square's kept box at these stops: none, its worst, a middle one, its best
    "square": {0.1: None, 0.15: 0.45, 0.2: 0.45, 0.25: 0.65, 0.3: 0.7},
    "square_worst": 0.45,                # the lowest score of the square's boxes
    "steps": (0.14, 0.58, 0.94),         # on the fine curve: 4 boxes from here, more than 4 from here, all 28 from here
}


def area(box) -> int:
    top, left, bottom, right = box
    return (bottom - top + 1) * (right - left + 1)


def iou(a, b) -> float:
    """The intersection over union of two boxes, as in cv-iou."""
    top, left = max(a[0], b[0]), max(a[1], b[1])
    bottom, right = min(a[2], b[2]), min(a[3], b[3])
    both = max(0, bottom - top + 1) * max(0, right - left + 1)
    return both / (area(a) + area(b) - both)


def nms(boxes, scores, threshold) -> list[int]:
    """The course's non-maximum suppression, as the worked example runs it: the numbers of the boxes kept, in the order
    they are kept, the highest score first."""
    rest = list(range(len(boxes)))
    kept = []
    while len(rest) > 0:
        best = rest[0]
        for i in rest:
            if scores[i] > scores[best]:
                best = i
        kept.append(best)
        rest = [i for i in rest if iou(boxes[i], boxes[best]) <= threshold]
    return kept


def rounds(boxes, scores, threshold) -> list[tuple]:
    """Every round of nms: the box kept, its numbers, its score, and how many boxes are left after the round."""
    rest, out = list(range(len(boxes))), []
    while rest:
        best = max(rest, key=lambda i: scores[i])
        rest = [i for i in rest if iou(boxes[i], boxes[best]) <= threshold]
        out.append((best, tuple(boxes[best]), scores[best], len(rest)))
    return out


@functools.lru_cache(maxsize=1)
def example() -> scenes.Scene:
    """The scene of the worked example."""
    return scenes.scene(SEED)


@functools.lru_cache(maxsize=1)
def proposed() -> tuple[tuple, tuple]:
    """The course's boxes for the scene and their scores, as the worked example gets them."""
    boxes, scores = scenes.proposals(example(), seed=PROPOSALS)
    return tuple(boxes), tuple(scores)


def owner(box) -> int:
    """The shape a box was made for: the shape whose box it overlaps most."""
    shapes = example().shapes
    return max(range(len(shapes)), key=lambda k: iou(box, shapes[k].box))


def kept_at(t: float) -> list[int]:
    boxes, scores = proposed()
    return nms(boxes, scores, t)


def one_per_shape(t: float) -> bool:
    boxes, _ = proposed()
    return sorted(owner(boxes[i]) for i in kept_at(t)) == list(range(len(example().shapes)))


def measure() -> dict:
    """Every number of REFERENCE, measured again."""
    s = example()
    boxes, scores = proposed()
    owners = [owner(b) for b in boxes]
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    top = next(k for k, i in enumerate(order) if owners[i] != CIRCLE)
    square = [i for i in range(len(boxes)) if owners[i] == SQUARE]
    best_square = max(square, key=lambda i: scores[i])
    circle_kept = order[0]
    against = [iou(boxes[i], boxes[circle_kept]) for i in square]
    above = [i for i in range(len(boxes)) if scores[i] > 0.7]
    found = rounds(boxes, scores, THRESHOLD)
    triangle = found[3][0]
    counts = {t: len(kept_at(t)) for t in FINE}

    def square_score(t):
        mine = [scores[i] for i in kept_at(t) if owners[i] == SQUARE]
        return mine[0] if len(mine) == 1 else None if not mine else tuple(mine)

    return {
        "count": len(boxes),
        "per": min(owners.count(k) for k in range(len(s.shapes))),
        "top": top,
        "square_rank": order.index(best_square) + 1,
        "above": (len(above), sum(1 for i in above if owners[i] == CIRCLE)),
        "rounds": tuple(found),
        "circle_rows": (min(boxes[i][0] for i in range(len(boxes)) if owners[i] == CIRCLE),
                        max(boxes[i][0] for i in range(len(boxes)) if owners[i] == CIRCLE)),
        "square_vs_circle": (round(iou(boxes[best_square], boxes[circle_kept]), 4), round(min(against), 4), round(max(against), 4)),
        "triangle": (round(iou(boxes[triangle], s.shapes[1].box), 3), round(iou(boxes[24], s.shapes[1].box), 3), scores[24]),
        "kept": {t: len(kept_at(t)) for t in (0.05, 0.1, 0.15, 0.3, 0.5, 0.55, 0.6, 0.65, 0.7, 0.95)},
        "best": max(t for t in THRESHOLDS if one_per_shape(t)),
        "square": {t: square_score(t) for t in (0.1, 0.15, 0.2, 0.25, 0.3)},
        "square_worst": min(scores[i] for i in square),
        "steps": (min(t for t in FINE if counts[t] == 4), min(t for t in FINE if counts[t] > 4),
                  min(t for t in FINE if counts[t] == len(boxes))),
    }


CROP = (34, 0, 123, 89)      # the part of the scene with the shapes: rows 34 to 123, columns 0 to 89


def _photo_axes(ax, img, title, crop=CROP):
    ax.imshow(img, interpolation="nearest")
    top, left, bottom, right = crop
    ax.set_xlim(left - 0.5, right + 0.5)
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title, fontsize=9)


def _draw_box(ax, b, width=1.5, halo=3.2):
    """A predicted box, dashed in the color of a prediction, with a thin edge in the color of the page, so that it
    stands out on every shape, in light and in dark."""
    from everything_ai import plots
    top, left, bottom, right = b
    xs = [left - 0.5, right + 0.5, right + 0.5, left - 0.5, left - 0.5]
    ys = [top - 0.5, top - 0.5, bottom + 0.5, bottom + 0.5, top - 0.5]
    ax.plot(xs, ys, color=plots.PAGE, linestyle="-", linewidth=halo, solid_joinstyle="miter")
    ax.plot(xs, ys, color=plots.PRED, linestyle=(0, (3.0, 1.6)), linewidth=width, solid_joinstyle="miter",
            dash_capstyle="butt")


def _score(ax, b, score, point):
    """The score of a box, just above its top left corner."""
    from everything_ai import plots
    ax.text(b[1] - 0.5, b[0] - 1.6, f"{score:.2f}".replace(".", point), fontsize=9, color=plots.PRED_TEXT,
            ha="left", va="bottom", bbox=dict(boxstyle="square,pad=0.15", facecolor=plots.PAGE, edgecolor="none"))


def proposals_picture(labels: dict[str, str]) -> str:
    """Two pictures of the worked example: all the course's boxes for the scene; the boxes NMS keeps at 0.5, with their
    scores. `labels["point"]` is the decimal mark."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    boxes, scores = proposed()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.75))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.05)
    _photo_axes(axes[0], example().image, labels["all"].format(n=len(boxes)))
    for b in boxes:
        _draw_box(axes[0], b, width=1.0, halo=2.6)
    kept = nms(boxes, scores, THRESHOLD)
    _photo_axes(axes[1], example().image, labels["kept"].format(n=len(kept)))
    for i in kept:
        _draw_box(axes[1], boxes[i])
    for i in kept:
        _score(axes[1], boxes[i], scores[i], labels["point"])
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


def threshold_frame(t: float, labels: dict[str, str]) -> tuple[str, int]:
    """For the slider: the boxes NMS keeps at threshold `t`, on the scene; and next to it how many it keeps at every
    threshold, with this one marked and a dotted line at one box per shape."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    boxes, scores = proposed()
    kept = kept_at(t)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.28})
    point = labels["point"]
    shown = f"{t:.2f}".replace(".", point)
    _photo_axes(left, example().image, labels["title"].format(t=shown, n=len(kept)))
    for i in kept:
        _draw_box(left, boxes[i])
    counts = [len(kept_at(x)) for x in FINE]
    n = len(example().shapes)
    right.plot([0, 1], [n, n], color=plots.INK_3, linestyle=":", linewidth=1.2)
    right.plot(FINE, counts, color=plots.INK, linestyle="-", linewidth=1.8, drawstyle="steps-post")
    right.plot([t], [len(kept)], linestyle="none", marker="o", color=plots.INK, markersize=8)
    right.set_xlim(-0.02, 1.02)
    right.set_ylim(-1, len(boxes) + 1)
    right.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    right.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    right.set_yticks([0, 4, 14, 28])
    right.set_xlabel(labels["threshold"])
    right.set_title(labels["count"], fontsize=10, loc="left")
    html = plots.render(fig, alt=labels["alt"].format(t=shown, n=len(kept)))
    plt.close(fig)
    return html, len(kept)


def _indices(got):
    """What the learner's function gave back as a list of box numbers, or None if it is not one: a list, tuple, set or
    array of whole numbers, not True or False."""
    if isinstance(got, (str, bytes, dict)) or got is None:
        return None
    try:
        items = list(got)
    except TypeError:
        return None
    out = []
    for x in items:
        if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
            return None
        out.append(int(x))
    return out


def _shown(v) -> str:
    """Box numbers as the function gave them: a list in its own order, a set from small to large."""
    got = _indices(v)
    if got is None:
        return repr(v) if isinstance(v, (str, bytes)) else str(v)
    if isinstance(v, (set, frozenset)):
        return "{" + ", ".join(str(i) for i in sorted(got)) + "}"
    return str(got)


def _same(got, want) -> bool:
    """Right when the function gives the numbers of the right boxes, each once, in any order. Two functions that both
    give nothing back give the same, when the check compares a known mistake with the learner's function."""
    if got is None and want is None:
        return True
    g = _indices(got)
    return g is not None and len(g) == len(set(g)) and sorted(g) == sorted(want)


def _boxes_given(got, text) -> bool:
    """The function gives boxes, four numbers each, instead of their numbers."""
    try:
        items = list(got)
    except TypeError:
        return False
    return len(items) > 0 and all(isinstance(x, (tuple, list)) and len(x) == 4 for x in items)


def _twice(got, text) -> bool:
    g = _indices(got)
    return g is not None and len(g) != len(set(g))


def _every(got, text) -> bool:
    """Every box kept: the numbers 0 to n - 1, with more than one box."""
    g = _indices(got)
    return g is not None and len(g) > 1 and sorted(g) == list(range(len(g)))


class _NmsBox(FunctionBox):
    """The check of nms: the scene's 28 boxes are too long to show, so a message names them as the worked example
    does, `boxes` and `scores`."""

    def _call_text(self, case) -> str:
        boxes, scores, threshold = case
        if tuple(boxes) == proposed()[0]:
            return f"nms(boxes, scores, {threshold})"
        return f"nms({list(boxes)!r}, {list(scores)!r}, {threshold})"


# The lesson is done when these are correct. "chain" and "cars" are choice questions and "kinds" an open question in
# the lesson.
REQUIRED = ["chain", "nms", "cars", "kinds"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME = {"en": "`{call}` gives `{got}`, and it should give `{want}`.", "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn."}


def _m(key, en, nl, **kw):
    """A mistake whose message starts with what the call gives and what it should give."""
    return Mistake(key, en=_SAME["en"] + " " + en, nl=_SAME["nl"] + " " + nl, **kw)


_IOU = ("def iou(a, b):\n"
        "    # the intersection over union of two boxes, as in the previous lesson\n"
        "    top, left = max(a[0], b[0]), max(a[1], b[1])\n"
        "    bottom, right = min(a[2], b[2]), min(a[3], b[3])\n"
        "    overlap = max(0, bottom - top + 1) * max(0, right - left + 1)\n"
        "    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)\n"
        "    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)\n"
        "    return overlap / (area_a + area_b - overlap)\n"
        "\n\n")
_HEAD = ("def nms(boxes, scores, threshold):\n"
         "    # boxes: a list of (top, left, bottom, right)\n"
         "    # scores: a list with the score of each box, scores[i] for boxes[i]\n"
         "    rest = list(range(len(boxes)))   # the numbers of the boxes still in the list\n"
         "    kept = []\n")


def _with(pick: str = "if scores[i] > scores[best]:", stay: str = "iou(boxes[i], boxes[best]) <= threshold",
          start: str = "rest[0]") -> str:
    """The course's code, with another test for the highest score, for the boxes that stay, or another first box."""
    return (_IOU + _HEAD + "    while len(rest) > 0:\n"
            f"        best = {start}\n"
            "        for i in rest:\n"
            f"            {pick}\n"
            "                best = i\n"
            "        kept.append(best)\n"
            f"        rest = [i for i in rest if {stay}]\n"
            "    return kept\n")


_START = _IOU + _HEAD + "    ...\n    return kept\n"
_SOLUTION = _with()
# the boxes in the order of the list, without looking for the highest score
_LIST_ORDER = (_IOU + _HEAD + "    while len(rest) > 0:\n        best = rest[0]\n        kept.append(best)\n"
               "        rest = [i for i in rest if iou(boxes[i], boxes[best]) <= threshold]\n    return kept\n")
# a box is dropped when any box with a higher score overlaps it too much, also a box that was dropped itself
_DROPPED_DROP = (_IOU + "def nms(boxes, scores, threshold):\n    kept = []\n    for i in range(len(boxes)):\n"
                 "        alone = True\n        for j in range(len(boxes)):\n"
                 "            if scores[j] > scores[i] and iou(boxes[i], boxes[j]) > threshold:\n"
                 "                alone = False\n        if alone:\n            kept.append(i)\n    return kept\n")
# each box compared only with the last box kept
_ONLY_LAST = (_IOU + "def nms(boxes, scores, threshold):\n"
              "    order = sorted(range(len(boxes)), key=lambda i: -scores[i])\n    kept = []\n    for i in order:\n"
              "        if len(kept) == 0 or iou(boxes[i], boxes[kept[-1]]) <= threshold:\n"
              "            kept.append(i)\n    return kept\n")
# only the first round: the box with the highest score
_FIRST_ONLY = (_IOU + _HEAD + "    best = rest[0]\n    for i in rest:\n        if scores[i] > scores[best]:\n"
               "            best = i\n    kept.append(best)\n    return kept\n")

# the 28 boxes of the scene, at 0.5 as in the worked example, and at a low and a high threshold; boxes with an IoU of
# exactly 0.5; a chain, in which a dropped box overlaps a later box; the highest score not first in the list; one
# box; boxes apart; two boxes the same; a box that overlaps the first kept box but not the last; no boxes at all
_SCENE = proposed()
_VISIBLE = [(list(_SCENE[0]), list(_SCENE[1]), 0.5)]
_HIDDEN = [
    (list(_SCENE[0]), list(_SCENE[1]), 0.1),
    (list(_SCENE[0]), list(_SCENE[1]), 0.7),
    ([(0, 0, 9, 9), (0, 0, 9, 19)], [0.9, 0.8], 0.5),
    ([(0, 0, 9, 9), (0, 4, 9, 13), (0, 8, 9, 17)], [0.9, 0.8, 0.7], 0.3),
    ([(0, 0, 9, 9), (0, 1, 9, 10), (30, 30, 39, 39)], [0.6, 0.9, 0.8], 0.5),
    ([(5, 5, 20, 20)], [0.3], 0.5),
    ([(0, 0, 9, 9), (0, 20, 9, 29), (20, 0, 29, 9)], [0.2, 0.6, 0.4], 0.5),
    ([(0, 0, 9, 9), (0, 0, 9, 9)], [0.4, 0.7], 0.5),
    ([(0, 0, 9, 9), (0, 50, 9, 59), (0, 1, 9, 10)], [0.9, 0.8, 0.7], 0.5),
    ([], [], 0.5),
]

def _yours(seen) -> str:
    """The boxes the learner's nms keeps at 0.5 on the scene of the worked example, as the lesson draws them, beside
    the course's where the learner's keeps others."""
    import matplotlib.pyplot as plt
    from everything_ai import check as checking, course, plots, yours
    mine = seen.mine
    fn = mine.names.get("nms") if mine else None
    if not callable(fn):
        return ""
    boxes, scores = proposed()
    try:
        kept = _indices(checking.with_time_limit(lambda: fn(list(boxes), list(scores), THRESHOLD)))
    except Exception:
        return ""
    if kept is None or any(not 0 <= i < len(boxes) for i in kept):
        return ""
    right = nms(boxes, scores, THRESHOLD)
    lang = seen.lang
    panels = [(course.phrase("yours-code", lang), kept)]
    if sorted(kept) != sorted(right):
        panels.append((course.phrase("yours-right", lang), right))
    word = {"en": "{who}: {n} boxes", "nl": "{who}: {n} boxes"}[lang]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.75), squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.05)
    for ax, (who, ks) in zip(axes[0], panels):
        _photo_axes(ax, example().image, word.format(who=who, n=len(ks)))
        for i in ks:
            _draw_box(ax, boxes[i])
    title = {"en": "The boxes your nms keeps at 0.5", "nl": "De boxes die jouw nms houdt bij 0,5"}[lang]
    alt = "; ".join(word.format(who=who, n=len(ks)) for who, ks in panels)
    out = plots.render(fig, alt=alt)
    plt.close(fig)
    return yours.figure(title, out, alt=alt)


EXERCISES = {
    "nms": _NmsBox(
        name="nms",
        picture=_yours,
        function="nms",
        start=_START,
        solution=_SOLUTION,
        visible=_VISIBLE,
        hidden=_HIDDEN,
        show=_shown,
        compare=_same,
        first=[_DOTS],
        mistakes=[
            _m("boxes", "Your function gives boxes. Give their numbers instead, their places in the list `boxes`: `kept.append(best)` with the number `best`, not `boxes[best]`.",
               "Je functie geeft boxes. Geef in plaats daarvan hun nummers, hun plaatsen in de lijst `boxes`: `kept.append(best)` met het nummer `best`, niet `boxes[best]`.",
               matches=_boxes_given),
            _m("remove", "Do you take boxes out of `rest` with `remove` while a `for` loop goes through `rest`? Then the loop skips the box right after each one it takes out. Build a new list instead, as the worked example does: `rest = [i for i in rest if ...]`.",
               "Haal je boxes uit `rest` met `remove` terwijl een `for`-loop door `rest` gaat? Dan slaat de loop de box over die direct na elke weggehaalde box komt. Maak in plaats daarvan een nieuwe lijst, zoals het uitgewerkte voorbeeld doet: `rest = [i for i in rest if ...]`.",
               matches=lambda got, text: re.search(r"\.remove\s*\(", text) is not None),
            _m("twice", "Your function keeps a box more than once. Is it `best`, the box with the highest score of the round, that you add to `kept`? And does every round build `rest` again, with only the boxes whose IoU with `best` is at most the threshold?",
               "Je functie houdt een box vaker dan één keer. Is het `best`, de box met de hoogste score van de ronde, die je aan `kept` toevoegt? En maakt elke ronde `rest` opnieuw, met alleen de boxes waarvan de IoU met `best` hoogstens de drempel is?",
               matches=_twice),
            _m("list-order", "Your function takes the boxes in the order of the list. Each round has to start with the box with the highest score of those still in the list: a box earlier in the list can have a lower score, and then it drops a better box.",
               "Je functie neemt de boxes in de volgorde van de lijst. Elke ronde moet beginnen met de box met de hoogste score van de boxes die nog in de lijst staan: een box eerder in de lijst kan een lagere score hebben, en dan gooit hij een betere box weg.",
               wrong=_LIST_ORDER),
            _m("lowest-first", "Your function starts with the lowest score. Each round keeps the box with the highest score: `>` in the test for `best`.",
               "Je functie begint bij de laagste score. Elke ronde houdt de box met de hoogste score: `>` in de test voor `best`.",
               wrong=_with(pick="if scores[i] < scores[best]:")),
            _m("dropped-drop", "Your function drops a box when its IoU with any box with a higher score is above the threshold, also when that box was dropped itself. Only kept boxes drop others: compare each box with the boxes you keep.",
               "Je functie gooit een box weg als zijn IoU met een box met een hogere score boven de drempel ligt, ook als die box zelf al weggegooid is. Alleen gehouden boxes gooien andere weg: vergelijk elke box met de boxes die je houdt.",
               wrong=_DROPPED_DROP),
            _m("only-last", "Your function compares each box with the last box kept only. Its IoU can be above the threshold with a box kept earlier: compare it with every box you keep.",
               "Je functie vergelijkt elke box alleen met de laatste box die gehouden is. Zijn IoU kan boven de drempel liggen met een box die eerder gehouden is: vergelijk hem met elke box die je houdt.",
               wrong=_ONLY_LAST),
            _m("equal-dropped", "A box whose IoU with the kept box is exactly the threshold has to stay: NMS drops only the boxes above it. Keep the boxes with `<=`, not `<`.",
               "Een box waarvan de IoU met de gehouden box precies de drempel is, moet blijven: NMS gooit alleen de boxes erboven weg. Houd de boxes met `<=`, niet met `<`.",
               wrong=_with(stay="iou(boxes[i], boxes[best]) < threshold")),
            _m("fixed-half", "Your function always works with 0.5. Use the threshold it gets: `threshold`.",
               "Je functie werkt altijd met 0,5. Gebruik de drempel die hij krijgt: `threshold`.",
               wrong=_with(stay="iou(boxes[i], boxes[best]) <= 0.5")),
            _m("first-only", "Your function keeps only the box with the highest score. That is one round: repeat it with the boxes that are left, until the list is empty. If your function already has a `while` loop, look at the spaces in front of `return kept`: it belongs after the loop, as far in as `while`, not inside it.",
               "Je functie houdt alleen de box met de hoogste score. Dat is één ronde: herhaal die met de boxes die over zijn, tot de lijst leeg is. Heeft je functie al een `while`-loop, kijk dan naar de spaties voor `return kept`: die hoort na de loop, even ver ingesprongen als `while`, niet erin.",
               wrong=_FIRST_ONLY),
            _m("first-only-b", "Your function keeps only the box with the highest score. That is one round: repeat it with the boxes that are left, until the list is empty. If your function already has a `while` loop, look at the spaces in front of `return kept`: it belongs after the loop, as far in as `while`, not inside it.",
               "Je functie houdt alleen de box met de hoogste score. Dat is één ronde: herhaal die met de boxes die over zijn, tot de lijst leeg is. Heeft je functie al een `while`-loop, kijk dan naar de spaties voor `return kept`: die hoort na de loop, even ver ingesprongen als `while`, niet erin.",
               wrong=_SOLUTION.replace("    return kept\n", "        return kept\n")),
            _m("keeps-all", "Your function keeps every box. After each box you keep, drop the boxes whose IoU with it is above the threshold.",
               "Je functie houdt elke box. Gooi na elke box die je houdt de boxes weg waarvan de IoU ermee boven de drempel ligt.",
               matches=_every),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with your own code: as long as there are boxes in `rest`, find the one with the highest score, keep it, and keep in `rest` only the boxes whose IoU with it is at most `threshold`. The recipe says how.",
                      "nl": "Vervang de `...` door je eigen code: zolang er boxes in `rest` staan, zoek je die met de hoogste score, houd je hem, en houd je in `rest` alleen de boxes waarvan de IoU ermee hoogstens `threshold` is. Het recept zegt hoe."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the boxes, the scores and a threshold, and the `def` line of your function has a different number of names between its parentheses. Keep `(boxes, scores, threshold)`.",
                          "nl": "De check roept `{call}` aan, met de boxes, de scores en een drempel, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(boxes, scores, threshold)`."},
            "slow": {"en": "Your function is still not done after 2 seconds. Does the `while` loop ever end? Every round must take boxes out of `rest`. Is the box you keep one of the boxes still in `rest`? Start with `best = rest[0]` and go through `rest`, not through all boxes. Then keep in `rest` only the boxes whose IoU with it is at most the threshold: the kept box goes too, as its IoU with itself is 1.",
                     "nl": "Je functie is na 2 seconden nog niet klaar. Houdt de `while`-loop ooit op? Elke ronde moet boxes uit `rest` halen. Staat de box die je houdt nog in `rest`? Begin met `best = rest[0]` en ga door `rest`, niet door alle boxes. Houd daarna in `rest` alleen de boxes waarvan de IoU ermee hoogstens de drempel is: de gehouden box gaat ook weg, want zijn IoU met zichzelf is 1."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Follow the recipe: as long as there are boxes in `rest`, take the one with the highest score and keep it; then keep in `rest` only the boxes whose IoU with it is at most `threshold`.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Volg het recept: zolang er boxes in `rest` staan, neem je die met de hoogste score en houd je hem; daarna houd je in `rest` alleen de boxes waarvan de IoU ermee hoogstens `threshold` is."},
            "almost": {"en": "Right for the scene at 0.5. But `{call}` gives `{got}`, and it should give `{want}`. Does a box drop others only once it is kept? Does a box whose IoU is exactly the threshold stay? Does each round start with the highest score of the boxes left?",
                       "nl": "Goed voor de scène bij 0,5. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gooit een box pas andere weg als hij gehouden is? Blijft een box waarvan de IoU precies de drempel is? Begint elke ronde bij de hoogste score van de boxes die over zijn?"},
            "correct": {"en": "Your function is right in all {n} cases. For the scene at 0.5 it keeps 4 boxes, one per shape, as in the worked example; at 0.1 it keeps 3, and the square under the circle loses its box. torchvision, PyTorch's library for images, has the same step as `torchvision.ops.nms`, with the same rule: it drops the boxes whose IoU is above the threshold.",
                        "nl": "Je functie klopt in alle {n} gevallen. Voor de scène bij 0,5 houdt hij 4 boxes, één per vorm, net als in het uitgewerkte voorbeeld; bij 0,1 houdt hij er 3, en verliest het vierkant onder de cirkel zijn box. torchvision, de bibliotheek van PyTorch voor beelden, heeft dezelfde stap als `torchvision.ops.nms`, met dezelfde regel: hij gooit de boxes weg waarvan de IoU boven de drempel ligt."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("nms", _START, "empty", None),
    ("nms", _SOLUTION, "correct", None),
    # other ways that are right too: the comparison turned round; boxes taken out one by one; kept boxes in
    # another order; a list of the boxes that stay built with a for loop
    ("nms", _with(stay="threshold >= iou(boxes[best], boxes[i])"), "correct", None),
    ("nms", _IOU + _HEAD + "    while len(rest) > 0:\n        best = rest[0]\n        for i in rest:\n"
            "            if scores[i] > scores[best]:\n                best = i\n        kept.append(best)\n"
            "        new = []\n        for i in rest:\n            if iou(boxes[i], boxes[best]) <= threshold:\n"
            "                new.append(i)\n        rest = new\n    return kept\n", "correct", None),
    ("nms", _SOLUTION.replace("    return kept\n", "    return sorted(kept)\n"), "correct", None),
    ("nms", _IOU + "def nms(boxes, scores, threshold):\n    order = sorted(range(len(boxes)), key=lambda i: -scores[i])\n"
            "    kept = []\n    for i in order:\n        alone = True\n        for k in kept:\n"
            "            if iou(boxes[i], boxes[k]) > threshold:\n                alone = False\n"
            "        if alone:\n            kept.append(i)\n    return kept\n", "correct", None),
    ("nms", _LIST_ORDER, "wrong", "list-order"),
    ("nms", _with(pick="if scores[i] < scores[best]:"), "wrong", "lowest-first"),
    ("nms", _DROPPED_DROP, "wrong", "dropped-drop"),
    ("nms", _ONLY_LAST, "wrong", "only-last"),
    ("nms", _with(stay="iou(boxes[i], boxes[best]) < threshold"), "wrong", "equal-dropped"),
    ("nms", _with(stay="iou(boxes[i], boxes[best]) <= 0.5"), "wrong", "fixed-half"),
    ("nms", _FIRST_ONLY, "wrong", "first-only"),
    ("nms", _SOLUTION.replace("kept.append(best)", "kept.append(boxes[best])"), "wrong", "boxes"),
    ("nms", _IOU + "def nms(boxes, scores, threshold):\n    return list(range(len(boxes)))\n", "wrong", "keeps-all"),
    ("nms", _IOU + "def nms(boxes, scores, threshold):\n    kept = []\n    for i in range(len(boxes)):\n"
            "        kept.append(i)\n        kept.append(i)\n    return kept\n", "wrong", "twice"),
    # the kept box left in the list, or a kept box that is no longer in it: the loop never ends
    ("nms", _with(stay="iou(boxes[i], boxes[best]) <= threshold or i == best"), "slow", None),
    ("nms", _with(start="0"), "slow", None),
    # a set of the right boxes counts
    ("nms", _SOLUTION.replace("    return kept\n", "    return set(kept)\n"), "correct", None),
    # return inside the loop: one round
    ("nms", _SOLUTION.replace("    return kept\n", "        return kept\n"), "wrong", "first-only-b"),
    # boxes taken out of the list while a for loop goes through it; the wrong box added to kept
    ("nms", _IOU + _HEAD + "    while len(rest) > 0:\n        best = rest[0]\n        for i in rest:\n"
            "            if scores[i] > scores[best]:\n                best = i\n        kept.append(best)\n"
            "        for i in rest:\n            if iou(boxes[i], boxes[best]) > threshold:\n                rest.remove(i)\n"
            "    return kept\n", "wrong", "remove"),
    ("nms", _SOLUTION.replace("kept.append(best)", "kept.append(i)"), "wrong", "twice"),
    # the scene's answer written out by hand: right for the visible case only
    ("nms", _IOU + "def nms(boxes, scores, threshold):\n    return [6, 13, 0, 23]\n", "almost", None),
    # text instead of numbers
    ("nms", _SOLUTION.replace("    return kept\n", "    return str(kept)\n"), "wrong", None),
    ("nms", _START.replace("    ...\n", "    best = ...\n"), "wrong", "dots-left"),
    ("nms", _SOLUTION.replace("def nms(boxes, scores, threshold):", "def nms(boxes, scores):"), "error", "arguments"),
    ("nms", _SOLUTION.replace("    return kept\n", "    print(kept)\n"), "wrong", "prints"),
]
