# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-map: average precision, written as a function of the true positives of one kind, from the
highest score down, and the number of true boxes. And the pictures: the course's detector boxes on 8 scenes, true and
false positives apart; and for the slider, the precision-recall curve of each kind and the mAP at every IoU threshold
of COCO."""
import functools
import re

import numpy as np

from everything_ai import scenes
from everything_ai.check import FunctionBox, Mistake

FIRST, COUNT = 36, 8         # the scenes of the lesson: scene(36) to scene(43)
DETECTOR = 17                # the seed of the course's detector boxes for them: detections(scenes, seed=17)
KIND = "circle"              # the kind of the worked example
# PASCAL VOC counts a detection as right when its overlap with a true box "exceeds 50%" (Everingham et al. 2010,
# section 4.4), and COCO computes AP at the IoU thresholds .50:.05:.95 and takes their mean as its main score, "AP"
# (cocodataset.org/#detection-eval). The lesson uses VOC's rule, IoU above the threshold; no IoU in the lesson's data
# lies exactly on one of these thresholds, so COCO's "at least" gives the same (the test checks that).
IOU = 0.5
STOPS = [round(0.5 + 0.05 * k, 2) for k in range(10)]   # COCO's IoU thresholds: the stops of the slider

# The numbers in the text. `measure()` measures them again; the test of the lesson checks that both are the same.
REFERENCE = {
    "shapes": (29, 11, 9, 9),            # true boxes in the 8 scenes: all, circles, squares, triangles
    "boxes": (32, 11, 8, 13),            # the detector's boxes: all, circles, squares, triangles
    "tp": (23, 9, 6),                    # at 0.5: true positives, false positives, true boxes never found
    "circle_tp": 8,                      # the circles found at 0.5
    "circle_first": (0.86, 0.837),       # the circles' highest score lies on a square, with this IoU with the square
    "circle_hits": (False, True, True, True, True, True, True, True, True, False, False),
    "circle_precision": (0.0, 0.5, 0.67, 0.75, 0.8, 0.83, 0.86, 0.88, 0.89, 0.8, 0.73),
    "circle_recall": (0.0, 0.09, 0.18, 0.27, 0.36, 0.45, 0.55, 0.64, 0.73, 0.73, 0.73),
    "ap": (0.56, 0.78, 0.73),            # AP at 0.5: circles, squares, triangles
    "map": 0.69,                         # their mean
    "without": (0.73, 0.56),             # the circles' AP without their highest false box (0.86), and without the
                                         # lowest (0.30)
    "smoothed": 0.65,                    # the circles' AP with interpolated precision, as VOC computes it
    "maps": {0.5: 0.69, 0.55: 0.69, 0.6: 0.63, 0.65: 0.6, 0.7: 0.54, 0.75: 0.44, 0.8: 0.2, 0.85: 0.07, 0.9: 0.0, 0.95: 0.0},
    "circle_ap": {0.6: 0.38, 0.7: 0.21},  # the circles' AP at these IoU thresholds
    "circle_loose": (0.571, 0.574),      # the IoUs of the two circle boxes that stop counting at 0.6
    "coco": 0.39,                        # the mean of the ten mAPs
    "best": 0.7,                         # the largest IoU threshold with an mAP of at least 0.5
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


@functools.lru_cache(maxsize=1)
def example() -> tuple:
    """The scenes of the lesson."""
    return tuple(scenes.scene(k) for k in range(FIRST, FIRST + COUNT))


@functools.lru_cache(maxsize=1)
def found() -> tuple:
    """The course's detector boxes for the scenes, as the worked example gets them: (image, kind, score, box), the
    highest score first."""
    return tuple(scenes.detections(list(example()), seed=DETECTOR))


def total(kind: str) -> int:
    """The true boxes of a kind in the scenes."""
    return sum(1 for s in example() for shape in s.shapes if shape.kind == kind)


@functools.lru_cache(maxsize=None)
def true_positives(kind: str, threshold: float = IOU) -> tuple:
    """For each box of a kind, the highest score first, whether it is a true positive, as the worked example decides:
    its IoU with a true box of its kind in the same image is above the threshold, and no box with a higher score found
    that true box first."""
    taken, hits = set(), []
    for image, k, score, box in found():
        if k != kind:
            continue
        truth = None
        for shape in example()[image].shapes:
            if shape.kind == kind and iou(box, shape.box) > threshold:
                truth = (image, shape.box)
        if truth is None or truth in taken:
            hits.append(False)
        else:
            taken.add(truth)
            hits.append(True)
    return tuple(hits)


def average_precision(hits, n) -> float:
    """The course's AP: the precision at each true positive, added up, divided by the number of true boxes."""
    tp, out = 0, 0.0
    for k, hit in enumerate(hits):
        if hit:
            tp += 1
            out += tp / (k + 1)
    return out / n


def curve(hits, n) -> tuple[list[float], list[float]]:
    """Precision and recall after each box."""
    tp = np.cumsum(np.array(hits, dtype=bool))
    return list(tp / np.arange(1, len(hits) + 1)), list(tp / n)


def smoothed(hits, n) -> float:
    """AP with interpolated precision, as PASCAL VOC computes it from 2010 on: the precision at each point becomes the
    highest precision at that recall or a higher one. COCO does the same, and then reads the curve at 101 recall
    values."""
    precision, _ = curve(hits, n)
    best = np.maximum.accumulate(np.array(precision)[::-1])[::-1]
    return float(best[np.array(hits, dtype=bool)].sum() / n)


def aps(threshold: float = IOU) -> dict[str, float]:
    return {kind: average_precision(true_positives(kind, threshold), total(kind)) for kind in scenes.KINDS}


def mean_ap(threshold: float = IOU) -> float:
    return float(np.mean(list(aps(threshold).values())))


def measure() -> dict:
    """Every number of REFERENCE, measured again."""
    boxes = found()
    hits = {kind: true_positives(kind) for kind in scenes.KINDS}
    tp = sum(sum(h) for h in hits.values())
    circle = hits[KIND]
    precision, recall = curve(circle, total(KIND))
    first = [d for d in boxes if d[1] == KIND][0]
    on = max(iou(first[3], shape.box) for shape in example()[first[0]].shapes)
    ap = aps()
    at = {t: mean_ap(t) for t in STOPS}
    # the circle boxes that count at 0.55 and not at 0.6
    mine = [d for d in boxes if d[1] == KIND]
    loose = sorted(round(max(iou(d[3], s.box) for s in example()[d[0]].shapes if s.kind == KIND), 3)
                   for d, a, b in zip(mine, true_positives(KIND, 0.55), true_positives(KIND, 0.6)) if a and not b)
    return {
        "shapes": (sum(len(s.shapes) for s in example()),) + tuple(total(k) for k in ("circle", "square", "triangle")),
        "boxes": (len(boxes),) + tuple(sum(1 for d in boxes if d[1] == k) for k in ("circle", "square", "triangle")),
        "tp": (tp, len(boxes) - tp, sum(len(s.shapes) for s in example()) - tp),
        "circle_tp": sum(circle),
        "circle_first": (first[2], round(on, 3)),
        "circle_hits": circle,
        "circle_precision": tuple(round(float(p), 2) for p in precision),
        "circle_recall": tuple(round(float(r), 2) for r in recall),
        "ap": tuple(round(ap[k], 2) for k in ("circle", "square", "triangle")),
        "map": round(mean_ap(), 2),
        "without": (round(average_precision(circle[1:], total(KIND)), 2), round(average_precision(circle[:-1], total(KIND)), 2)),
        "smoothed": round(smoothed(circle, total(KIND)), 2),
        "maps": {t: round(at[t], 2) for t in STOPS},
        "circle_ap": {t: round(aps(t)[KIND], 2) for t in (0.6, 0.7)},
        "circle_loose": tuple(loose),
        "coco": round(float(np.mean(list(at.values()))), 2),
        "best": max(t for t in STOPS if at[t] >= 0.5),
    }


# --- pictures --------------------------------------------------------------------------------------------------------

def _photo_axes(ax, img, title):
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title, fontsize=9)


def _outline(box):
    top, left, bottom, right = box
    return ([left - 0.5, right + 0.5, right + 0.5, left - 0.5, left - 0.5],
            [top - 0.5, top - 0.5, bottom + 0.5, bottom + 0.5, top - 0.5])


def _draw_true(ax, box):
    """A true box: a thin line in the data color, with a soft edge in the page color. The scenes are small and the
    same light grey in both themes: a full edge in the page color would frame every box in black in dark."""
    from everything_ai import plots
    xs, ys = _outline(box)
    ax.plot(xs, ys, color=plots.PAGE, alpha=0.5, linestyle="-", linewidth=2.2, solid_joinstyle="miter")
    ax.plot(xs, ys, color=plots.DATA, linestyle="-", linewidth=1.0, solid_joinstyle="miter")


def _draw_found(ax, box, right):
    """A box of the detector: a true positive dashed in the color of a prediction, a false positive dotted in the
    color of a mistake, so that the two differ by more than their color. Its edge is soft, as that of a true box."""
    from everything_ai import plots
    xs, ys = _outline(box)
    ax.plot(xs, ys, color=plots.PAGE, alpha=0.5, linestyle="-", linewidth=2.8, solid_joinstyle="miter")
    if right:
        ax.plot(xs, ys, color=plots.PRED, linestyle=(0, (3.0, 1.6)), linewidth=1.5, solid_joinstyle="miter",
                dash_capstyle="butt")
    else:
        ax.plot(xs, ys, color=plots.LOSS, linestyle=(0, (0.1, 2.0)), linewidth=2.0, solid_joinstyle="miter",
                dash_capstyle="round")


def _hit_of_box() -> dict:
    """For each box of the detector, whether it is a true positive at 0.5."""
    out = {}
    for kind in scenes.KINDS:
        mine = [d for d in found() if d[1] == kind]
        out.update({d: h for d, h in zip(mine, true_positives(kind))})
    return out


def scenes_picture(labels: dict[str, str]) -> str:
    """The 8 scenes, with the true boxes and the detector's boxes: true positives and false positives in their own
    color. `labels["image"]` is the title of a scene, with {i} its place in the list."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    hit = _hit_of_box()
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.95))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.06, hspace=0.16)
    for i, (ax, s) in enumerate(zip(axes.flat, example())):
        _photo_axes(ax, s.image, labels["image"].format(i=i))
        for shape in s.shapes:
            _draw_true(ax, shape.box)
        for d in found():
            if d[0] == i:
                _draw_found(ax, d[3], hit[d])
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


def iou_frame(t: float, labels: dict[str, str]) -> tuple[str, float]:
    """For the slider: the precision-recall curve of each kind at IoU threshold `t`, and next to it the mAP at every
    threshold of COCO, with this one marked and a dotted line at 0.5."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    point = labels["point"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.3})
    ap = aps(t)
    for kind, style in zip(("circle", "square", "triangle"), ("-", "--", ":")):
        hits = true_positives(kind, t)
        precision, recall = curve(hits, total(kind))
        name = labels[kind] + " " + f"{ap[kind]:.2f}".replace(".", point)
        # steps: over each step of recall, the precision at the box that makes it, so the area is the AP
        left.plot([0.0] + recall, [precision[0]] + precision, color=plots.INK, linestyle=style, linewidth=1.8,
                  drawstyle="steps-pre", label=name)
    left.set_xlim(-0.02, 1.02)
    left.set_ylim(-0.03, 1.05)
    left.set_xticks([0, 0.5, 1.0])
    left.set_xticklabels(["0", f"0{point}5", "1"])
    left.set_yticks([0, 0.5, 1.0])
    left.set_yticklabels(["0", f"0{point}5", "1"])
    left.set_xlabel(labels["recall"])
    left.set_ylabel(labels["precision"])
    left.set_title(labels["curves"].format(t=f"{t:.2f}".replace(".", point)), fontsize=10, loc="left")
    left.legend(loc="lower right", fontsize=8.5, handlelength=2.4)
    values = [mean_ap(x) for x in STOPS]
    right.plot([0.48, 0.97], [0.5, 0.5], color=plots.INK_3, linestyle=":", linewidth=1.2)
    right.plot(STOPS, values, color=plots.INK, linestyle="-", linewidth=1.8, marker="o", markersize=4)
    right.plot([t], [mean_ap(t)], linestyle="none", marker="o", color=plots.INK, markersize=9)
    right.set_xlim(0.48, 0.97)
    right.set_ylim(-0.03, 1.05)
    right.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    right.set_xticklabels([f"0{point}{d}" for d in (5, 6, 7, 8, 9)])
    right.set_yticks([0, 0.5, 1.0])
    right.set_yticklabels(["0", f"0{point}5", "1"])
    right.set_xlabel(labels["threshold"])
    right.set_title(labels["map"], fontsize=10, loc="left")
    m = mean_ap(t)
    html = plots.render(fig, alt=labels["alt"].format(t=f"{t:.2f}".replace(".", point), m=f"{m:.2f}".replace(".", point)))
    plt.close(fig)
    return html, m


# --- the exercise ----------------------------------------------------------------------------------------------------

# The lesson is done when these are correct. "hand" and "bikes" are choice questions and "extra" an open question in
# the lesson.
REQUIRED = ["hand", "ap", "bikes", "extra"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME = {"en": "`{call}` gives `{got}`, and it should give `{want}`.", "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn."}


def _m(key, en, nl, **kw):
    """A mistake whose message starts with what the call gives and what it should give."""
    return Mistake(key, en=_SAME["en"] + " " + en, nl=_SAME["nl"] + " " + nl, **kw)


_HEAD = ("def average_precision(hits, total):\n"
         "    # hits: for each box, from the highest score down, True for a true positive\n"
         "    # total: the number of true boxes\n")


def _with(add: str = "found / (k + 1)", end: str = "sum_precision / total", order: str = "count first") -> str:
    """The course's code, with another precision added at a true positive, another end, or the precision taken before
    the true positive is counted."""
    count = "            found = found + 1\n"
    add_line = f"            sum_precision = sum_precision + {add}\n"
    return (_HEAD + "    found = 0            # true positives so far\n"
            "    sum_precision = 0\n"
            "    for k in range(len(hits)):\n"
            "        if hits[k]:\n"
            + (count + add_line if order == "count first" else add_line + count) +
            f"    return {end}\n")


_START = _HEAD + "    ...\n"
_SOLUTION = _with()
# the precision after every box, averaged over all boxes
_EVERY = (_HEAD + "    found = 0\n    sum_precision = 0\n    for k in range(len(hits)):\n        if hits[k]:\n"
          "            found = found + 1\n        sum_precision = sum_precision + found / (k + 1)\n"
          "    return sum_precision / len(hits)\n")
# the mean of the precision at the true positives, with NumPy as in the worked example: it divides by the true
# positives, and gives nan when there is none
_NP_MEAN = ("import numpy as np\n\n" + _HEAD + "    hits = np.array(hits, dtype=bool)\n"
            "    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)\n    return np.mean(precision[hits])\n")
# the precision interpolated first, as PASCAL VOC does
_SMOOTHED = (_HEAD + "    precision = []\n    found = 0\n    for k in range(len(hits)):\n        if hits[k]:\n"
             "            found = found + 1\n        precision.append(found / (k + 1))\n"
             "    sum_precision = 0\n    for k in range(len(hits)):\n        if hits[k]:\n"
             "            sum_precision = sum_precision + max(precision[k:])\n    return sum_precision / total\n")

_CIRCLE = list(true_positives("circle"))
_VISIBLE = [(_CIRCLE, total("circle"))]
# the squares and the triangles at 0.5, the circles at 0.7 and the triangles at 0.75; no true positive; every box a
# true positive; two true positives with a false one between them; fewer true positives than true boxes; true
# positives after false ones
_HIDDEN = [
    (list(true_positives("square")), total("square")),
    (list(true_positives("triangle")), total("triangle")),
    (list(true_positives("circle", 0.7)), total("circle")),
    (list(true_positives("triangle", 0.75)), total("triangle")),
    ([False, False], 2),
    ([True, True, True], 3),
    ([True, False, True], 2),
    ([True], 4),
    ([False, True, False, True], 2),
]


def _shown(v) -> str:
    """A number as the function gave it, to two decimals when it is a float with more."""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        r = round(float(v), 4)
        return str(r) if r == float(v) else f"{float(v):.4f}..."
    return repr(v) if isinstance(v, str) else str(v)


class _ApBox(FunctionBox):
    """The check of average_precision: the scene's list of 11 values is long, so a message names it as the worked
    example does, `hits`."""

    def _call_text(self, case) -> str:
        hits, n = case
        if list(hits) == _CIRCLE:
            return f"average_precision(hits, {n})"
        return f"average_precision({list(hits)!r}, {n})"


EXERCISES = {
    "ap": _ApBox(
        name="ap",
        function="average_precision",
        start=_START,
        solution=_SOLUTION,
        visible=_VISIBLE,
        hidden=_HIDDEN,
        show=_shown,
        first=[_DOTS],
        mistakes=[
            _m("found", "Your function divides by the number of true positives. Divide by `total`, the number of true boxes: a true box that is never found counts as 0. Otherwise a detector that finds only its surest object gets an AP of 1.",
               "Je functie deelt door het aantal true positives. Deel door `total`, het aantal echte boxes: een echte box die nooit gevonden wordt, telt als 0. Anders krijgt een detector die alleen zijn zekerste voorwerp vindt een AP van 1.",
               wrong=_with(end="sum_precision / found")),
            _m("found-mean", "Your function takes the mean of the precision at the true positives, so it divides by their number. Divide the sum by `total`, the number of true boxes: a true box that is never found counts as 0. Otherwise a detector that finds only its surest object gets an AP of 1.",
               "Je functie neemt het gemiddelde van de precision bij de true positives, dus deelt hij door hun aantal. Deel de som door `total`, het aantal echte boxes: een echte box die nooit gevonden wordt, telt als 0. Anders krijgt een detector die alleen zijn zekerste voorwerp vindt een AP van 1.",
               wrong=_NP_MEAN),
            _m("boxes", "Your function divides by the number of boxes. Divide by `total`, the number of true boxes: AP is the mean over the true boxes, with 0 for every one that is never found.",
               "Je functie deelt door het aantal boxes. Deel door `total`, het aantal echte boxes: AP is het gemiddelde over de echte boxes, met 0 voor elke box die nooit gevonden wordt.",
               wrong=_with(end="sum_precision / len(hits)")),
            _m("every", "Your function averages the precision after every box. AP only adds the precision at each true positive, and divides by the number of true boxes: a false positive counts only by lowering the precision at the true positives after it.",
               "Je functie neemt het gemiddelde van de precision na elke box. AP telt alleen de precision bij elke true positive op, en deelt door het aantal echte boxes: een false positive telt alleen mee doordat hij de precision bij de true positives erna verlaagt.",
               wrong=_EVERY),
            _m("every-total", "Your function adds the precision after every box, also after a false positive. Add it only at a true positive: does the line that adds it stand inside `if hits[k]:`, indented as far as the line that adds 1 to `found`?",
               "Je functie telt de precision na elke box op, ook na een false positive. Tel hem alleen op bij een true positive: staat de regel die hem optelt binnen `if hits[k]:`, even ver ingesprongen als de regel die `found` met 1 verhoogt?",
               wrong=_EVERY.replace("    return sum_precision / len(hits)\n", "    return sum_precision / total\n")),
            _m("return-inside", "Your function stops after the first box. Look at the spaces in front of `return`: it belongs after the loop, as far in as `for`, so that the loop goes through every box first.",
               "Je functie stopt na de eerste box. Kijk naar de spaties voor `return`: die hoort na de loop, even ver ingesprongen als `for`, zodat de loop eerst door elke box gaat.",
               wrong=_SOLUTION.replace("    return sum_precision / total\n", "        return sum_precision / total\n")),
            _m("recall", "That is the recall at the end of the list: how many of the true boxes are found. AP also looks at the order: at each true positive, add the precision so far, `found / (k + 1)`.",
               "Dat is de recall aan het eind van de lijst: hoeveel van de echte boxes gevonden zijn. AP kijkt ook naar de volgorde: tel bij elke true positive de precision tot dan toe op, `found / (k + 1)`.",
               wrong=_with(end="found / total")),
            _m("precision", "That is the precision at the end of the list. AP adds the precision at each true positive, from the highest score down, and divides by the number of true boxes.",
               "Dat is de precision aan het eind van de lijst. AP telt de precision bij elke true positive op, van de hoogste score omlaag, en deelt door het aantal echte boxes.",
               wrong=_with(end="found / len(hits)")),
            _m("before", "Your function takes the precision before it counts the true positive. At box `k`, the true positives so far include that box itself: add 1 to `found` first, then add `found / (k + 1)`.",
               "Je functie neemt de precision voordat hij de true positive meetelt. Bij box `k` horen de true positives tot dan toe ook die box zelf: verhoog `found` eerst met 1, en tel dan `found / (k + 1)` op.",
               wrong=_with(order="add first")),
            _m("index", "The precision after box `k` divides by the number of boxes so far, and with `range` that is `k + 1`: `k` starts at 0.",
               "De precision na box `k` deelt door het aantal boxes tot dan toe, en met `range` is dat `k + 1`: `k` begint bij 0.",
               wrong=_with(add="found / k")),
            _m("recall-added", "At each true positive your function adds the recall so far. AP adds the precision so far, `found / (k + 1)`: the share of the boxes so far that are true positives.",
               "Bij elke true positive telt je functie de recall tot dan toe op. AP telt de precision tot dan toe op, `found / (k + 1)`: het deel van de boxes tot dan toe dat een true positive is.",
               wrong=_with(add="found / total")),
            _m("smoothed", "That is the AP with interpolated precision, as PASCAL VOC computes it: at each true positive the highest precision at that recall or a higher one. This lesson leaves that step out: add the precision at each true positive itself.",
               "Dat is de AP met geïnterpoleerde precision, zoals PASCAL VOC hem uitrekent: bij elke true positive de hoogste precision bij die recall of een hogere. Deze les laat die stap weg: tel de precision bij elke true positive zelf op.",
               wrong=_SMOOTHED),
            _m("rounded", "Your function rounds. Give the AP itself back, without `round`: the check compares it with more decimals.",
               "Je functie rondt af. Geef de AP zelf terug, zonder `round`: de check vergelijkt hem met meer decimalen.",
               wrong=_with(end="round(sum_precision / total, 2)")),
            _m("percent", "Your function gives a percentage. AP is a number from 0 to 1: leave out the `* 100`.",
               "Je functie geeft een percentage. AP is een getal van 0 tot 1: laat de `* 100` weg.",
               wrong=_with(end="sum_precision / total * 100")),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with your own code: go through `hits`, keep count of the true positives so far, add the precision at each true positive, and divide by `total` at the end. The recipe says how.",
                      "nl": "Vervang de `...` door je eigen code: ga door `hits`, houd bij hoeveel true positives je al hebt gezien, tel bij elke true positive de precision op, en deel aan het eind door `total`. Het recept zegt hoe."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the list and the number of true boxes, and the `def` line of your function has a different number of names between its parentheses. Keep `(hits, total)`.",
                          "nl": "De check roept `{call}` aan, met de lijst en het aantal echte boxes, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(hits, total)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Follow the recipe: go through `hits` from the start; at each true positive, add 1 to `found`, and add the precision so far, the true positives so far divided by the boxes so far. At the end, divide the sum by `total`.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Volg het recept: ga door `hits` vanaf het begin; verhoog bij elke true positive `found` met 1, en tel de precision tot dan toe op, de true positives tot dan toe gedeeld door de boxes tot dan toe. Deel de som aan het eind door `total`."},
            "almost": {"en": "Right for the circles. But `{call}` gives `{got}`, and it should give `{want}`. Do you divide by `total`, the number of true boxes, also when not every true box is found? Do you count the boxes from 1, so that the precision after the first box divides by 1?",
                       "nl": "Goed voor de cirkels. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Deel je door `total`, het aantal echte boxes, ook als niet elke echte box gevonden is? Tel je de boxes vanaf 1, zodat de precision na de eerste box door 1 deelt?"},
            "correct": {"en": "Your function is right in all {n} cases. For the circles it gives 0.56, as in the worked example. scikit-learn's `average_precision_score` uses the same formula, without interpolation, but it only sees the boxes you give it: a true box that no box finds is missing, so on a detector's boxes it divides by the true positives. Detection libraries such as torchmetrics count the misses, and compute COCO's interpolated version.",
                        "nl": "Je functie klopt in alle {n} gevallen. Voor de cirkels geeft hij 0,56, net als in het uitgewerkte voorbeeld. `average_precision_score` van scikit-learn gebruikt dezelfde formule, zonder interpolatie, maar ziet alleen de boxes die je hem geeft: een echte box die door geen enkele box gevonden wordt, ontbreekt, dus op de boxes van een detector deelt hij door de true positives. Bibliotheken voor detectie zoals torchmetrics tellen de gemiste boxes mee, en rekenen de geïnterpoleerde versie van COCO uit."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("ap", _START, "empty", None),
    ("ap", _SOLUTION, "correct", None),
    # other ways that are right too: with NumPy, as in the worked example; with enumerate; with the precision of
    # every box in a list first
    ("ap", "import numpy as np\n\n" + _HEAD + "    hits = np.array(hits, dtype=bool)\n"
           "    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)\n"
           "    return np.sum(precision[hits]) / total\n", "correct", None),
    ("ap", _HEAD + "    found = 0\n    out = 0.0\n    for k, hit in enumerate(hits):\n        if hit:\n"
           "            found += 1\n            out += found / (k + 1)\n    return out / total\n", "correct", None),
    ("ap", _HEAD + "    precision = []\n    found = 0\n    for k in range(len(hits)):\n        if hits[k]:\n"
           "            found = found + 1\n        precision.append(found / (k + 1))\n    sum_precision = 0\n"
           "    for k in range(len(hits)):\n        if hits[k]:\n            sum_precision = sum_precision + precision[k]\n"
           "    return sum_precision / total\n", "correct", None),
    ("ap", _with(end="sum_precision / found"), "wrong", "found"),
    ("ap", _with(end="sum_precision / len(hits)"), "wrong", "boxes"),
    ("ap", _EVERY, "wrong", "every"),
    ("ap", _with(end="found / total"), "wrong", "recall"),
    ("ap", _with(end="found / len(hits)"), "wrong", "precision"),
    ("ap", _with(order="add first"), "wrong", "before"),
    ("ap", _with(add="found / k"), "wrong", "index"),
    ("ap", _NP_MEAN, "wrong", "found-mean"),
    ("ap", _EVERY.replace("    return sum_precision / len(hits)\n", "    return sum_precision / total\n"), "wrong", "every-total"),
    ("ap", _SOLUTION.replace("    return sum_precision / total\n", "        return sum_precision / total\n"), "wrong", "return-inside"),
    ("ap", _with(add="found / total"), "wrong", "recall-added"),
    ("ap", _SMOOTHED, "wrong", "smoothed"),
    ("ap", _with(end="round(sum_precision / total, 2)"), "wrong", "rounded"),
    ("ap", _with(end="sum_precision / total * 100"), "wrong", "percent"),
    # the circles' AP written out by hand: right for the visible case only
    ("ap", _HEAD + "    return 0.561002886002886\n", "almost", None),
    ("ap", _START.replace("    ...\n", "    found = ...\n"), "wrong", "dots-left"),
    ("ap", _SOLUTION.replace("def average_precision(hits, total):", "def average_precision(hits):"), "error", "arguments"),
    ("ap", _SOLUTION.replace("    return sum_precision / total\n", "    print(sum_precision / total)\n"), "wrong", "prints"),
]
