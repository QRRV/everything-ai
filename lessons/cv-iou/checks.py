# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-iou: the intersection over union of two boxes, written as a function that works from the
four numbers of each box. And the pictures: a true box and a predicted box with their overlap and their union, on
the course's own scene; and for the slider, the predicted box shifted step by step, next to its IoU at every step."""
import functools
import re

import numpy as np

from everything_ai import scenes
from everything_ai.check import FunctionBox, Mistake

SEED = 18                    # the scene of the worked example, as in cv-annotations
CIRCLE = 3                   # the circle is the last shape of that scene, on top: s.shapes[3]
PRED = (62, 46, 91, 77)      # the box that a model could predict for the circle, in the worked example
SHIFTS = list(range(0, 30))  # the stops of the slider: the circle's box moved 0 to 29 pixels to the right
# PASCAL VOC counts a detection as correct when "the area of overlap ... must exceed 50%", with the overlap measured as
# the intersection over union (Everingham et al. 2010, "The PASCAL Visual Object Classes (VOC) Challenge", IJCV).
CORRECT_ABOVE = 0.5

# The numbers in the text. `measure()` measures them again; the test of the lesson checks that both are the same.
REFERENCE = {
    "truth": (58, 50, 86, 78),           # the box of the circle
    "areas": (841, 960),                 # the pixels of the circle's box and of the predicted box
    "overlap": 700,                      # their overlap
    "union": 1101,                       # their union
    "iou": 0.6358,                       # 700 / 1101
    "others": (0.0, 0.0, 0.1879, 0.6358),   # the predicted box against the box of every shape of the scene, in order
    # the third cell of the worked example: against the box of the square and of the large triangle, the rows and
    # the columns of the overlap before they are made at least 0, the overlap, the union and the IoU
    "steps": ((12, 27, 324, 1724, 0.1879), (14, -16, 0, 1535, 0.0)),
    "whole": 0.0513,                     # a box over the whole scene against the circle's box
    "best": 9,                           # the largest shift of the slider with an IoU above 0.5
    "shift": {9: 0.5263, 10: 0.4872, 29: 0.0},
}


def area(box) -> int:
    """The pixels of a box, both ends included."""
    top, left, bottom, right = box
    return (bottom - top + 1) * (right - left + 1)


def overlap_box(a, b) -> tuple[int, int, int, int]:
    """Step 1 of the recipe: the largest top and left, the smallest bottom and right. When the boxes do not overlap,
    its bottom lies above its top or its right side left of its left side."""
    return max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])


def overlap(a, b) -> int:
    """Step 2: the pixels of the overlap, a count below 0 made 0."""
    top, left, bottom, right = overlap_box(a, b)
    return max(0, bottom - top + 1) * max(0, right - left + 1)


def union(a, b) -> int:
    """Step 3: the pixels of both boxes, the overlap counted once."""
    return area(a) + area(b) - overlap(a, b)


def iou(a, b) -> float:
    """Step 4: the intersection over union of two boxes."""
    return overlap(a, b) / union(a, b)


def box_mask(box, size: int = scenes.SIZE) -> np.ndarray:
    """A box as a mask as large as the scene, as the worked example makes it."""
    top, left, bottom, right = box
    mask = np.zeros((size, size), dtype=bool)
    mask[top:bottom + 1, left:right + 1] = True
    return mask


def counted(a, b) -> tuple[int, int, float]:
    """The overlap, the union and the IoU of two boxes from their pixels, as the worked example counts them."""
    ma, mb = box_mask(a), box_mask(b)
    o, u = int((ma & mb).sum()), int((ma | mb).sum())
    return o, u, o / u


@functools.lru_cache(maxsize=1)
def example() -> scenes.Scene:
    """The scene of the worked example."""
    return scenes.scene(SEED)


def truth() -> tuple[int, int, int, int]:
    """The box of the circle of the worked example."""
    return example().shapes[CIRCLE].box


def shifted(d: int) -> tuple[int, int, int, int]:
    """The circle's box, moved `d` pixels to the right: the predicted box of the slider."""
    top, left, bottom, right = truth()
    return top, left + d, bottom, right + d


def shift_outcome(d: int) -> dict:
    """At this stop of the slider: the predicted box, its overlap and union with the true box, and its IoU."""
    b = shifted(int(d))
    return {"box": b, "overlap": overlap(truth(), b), "union": union(truth(), b), "iou": iou(truth(), b)}


def steps(t) -> tuple:
    """Steps 1 to 4 for the predicted box and the box `t`, as the third cell of the worked example prints them: the
    rows and the columns of the overlap before they are made at least 0, the overlap, the union and the IoU."""
    top, left, bottom, right = overlap_box(t, PRED)
    return (bottom - top + 1, right - left + 1, overlap(t, PRED), union(t, PRED), round(iou(t, PRED), 4))


def sized(side: int, more: int) -> float:
    """The IoU of a square box of `side` pixels and a box with the same middle, `more` pixels larger on every side."""
    return side * side / ((side + 2 * more) ** 2)


def measure() -> dict:
    """Every number of REFERENCE, measured again: the box numbers by counting pixels, the others by the arithmetic."""
    t, s = truth(), example()
    o, u, value = counted(t, PRED)
    ious = [iou(t, shifted(d)) for d in SHIFTS]
    return {
        "truth": t, "areas": (int(box_mask(t).sum()), int(box_mask(PRED).sum())), "overlap": o, "union": u,
        "iou": round(value, 4),
        "others": tuple(round(counted(shape.box, PRED)[2], 4) for shape in s.shapes),
        "whole": round(counted(t, (0, 0, scenes.SIZE - 1, scenes.SIZE - 1))[2], 4),
        "steps": tuple(steps(shape.box) for shape in (s.shapes[2], s.shapes[1])),
        "best": max(d for d, v in zip(SHIFTS, ious) if v > CORRECT_ABOVE),
        "shift": {d: round(ious[d], 4) for d in (9, 10, 29)},
    }


def _photo_axes(ax, img, title, crop):
    if np.ndim(img) == 3:
        ax.imshow(img, interpolation="nearest")
    else:
        ax.imshow(np.asarray(img, dtype=float), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    top, left, bottom, right = crop
    ax.set_xlim(left - 0.5, right + 0.5)
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title, fontsize=9)


def _draw_box(ax, b, colour, linestyle):
    """A box as a line around its pixels, from the outer edge of the top-left pixel to that of the bottom-right one,
    with a thin edge in the color of the page, so that it stands out on every shape, in light and in dark."""
    from everything_ai import plots
    top, left, bottom, right = b
    xs = [left - 0.5, right + 0.5, right + 0.5, left - 0.5, left - 0.5]
    ys = [top - 0.5, top - 0.5, bottom + 0.5, bottom + 0.5, top - 0.5]
    ax.plot(xs, ys, color=plots.PAGE, linestyle="-", linewidth=3.2, solid_joinstyle="miter")
    ax.plot(xs, ys, color=colour, linestyle=linestyle, linewidth=1.5, solid_joinstyle="miter", dash_capstyle="butt")


def _both(ax, b):
    """The true box in the color of data, a solid line; the predicted box in the color of a prediction, dashed."""
    from everything_ai import plots
    _draw_box(ax, truth(), plots.DATA, "-")
    _draw_box(ax, b, plots.PRED, (0, (3.0, 1.6)))


def overlap_picture(labels: dict[str, str]) -> str:
    """Three pictures of the worked example: the circle's box and the predicted box on the scene; their overlap; their
    union.
    The pixels counted are white, as in the mask of cv-annotations."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    t, g = truth(), PRED
    a, b = box_mask(t), box_mask(g)
    crop = (45, 32, 104, 91)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.06)
    _photo_axes(axes[0], example().image, labels["boxes"], crop)
    inside = np.where(a & b, 1.0, np.where(a | b, 0.3, 0.0))
    _photo_axes(axes[1], inside, labels["overlap"].format(overlap=int((a & b).sum())), crop)
    _photo_axes(axes[2], (a | b).astype(float), labels["union"].format(union=int((a | b).sum())), crop)
    for ax in axes:
        _both(ax, g)
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


def shift_frame(d: int, labels: dict[str, str]) -> tuple[str, dict]:
    """For the slider: the true box and the predicted box moved `d` pixels to the right, on the scene; and next to it
    the IoU at every stop, with this one marked and a dotted line at 0.5. `labels["point"]` is the decimal mark of the
    description."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    d = int(d)
    out = shift_outcome(d)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.28})
    _photo_axes(left, example().image, labels["title"].format(shift=d), (28, 35, 115, 122))
    _both(left, out["box"])
    curve = [shift_outcome(s)["iou"] for s in SHIFTS]
    right.plot([SHIFTS[0], SHIFTS[-1]], [CORRECT_ABOVE, CORRECT_ABOVE], color=plots.INK_3, linestyle=":", linewidth=1.2)
    right.plot(SHIFTS, curve, color=plots.INK, linestyle="-", linewidth=1.8)
    right.plot([d], [out["iou"]], linestyle="none", marker="o", color=plots.INK, markersize=8)
    right.set_xlim(-1.2, 30.2)
    right.set_ylim(-0.04, 1.04)
    right.set_xticks([0, 5, 10, 15, 20, 25, 29])
    right.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    right.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    right.set_xlabel(labels["shift"])
    right.set_title(labels["iou"], fontsize=10, loc="left")
    html = plots.render(fig, alt=labels["alt"].format(shift=d, iou=f"{out['iou']:.2f}".replace(".", labels["point"])))
    plt.close(fig)
    return html, out


def _number(got):
    """What the learner's function gave back as a float, or None if it is not one number: text such as '0.5', True
    and False, and a list do not count."""
    if isinstance(got, (str, bytes, bool, np.bool_)):
        return None
    try:
        a = np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return None
    return float(a) if a.ndim == 0 else None


def _shown(v) -> str:
    """A share the way the lesson writes it: 0.6358 for four decimals, and whole numbers as they are. Text keeps its
    quotes, so that '0.64' does not look like a number."""
    x = _number(v)
    if x is None:
        return repr(v) if isinstance(v, (str, bytes)) else str(v)
    return str(int(x)) if x.is_integer() else f"{x:.4f}".rstrip("0")


def _same(got, want) -> bool:
    """Right when the function gives one number, the course's IoU, up to rounding."""
    x = _number(got)
    return x is not None and abs(x - want) < 1e-9


# The lesson is done when these are correct. "overlap" and "middle" are choice questions and "big" an open question
# in the lesson.
REQUIRED = ["overlap", "iou", "middle", "big"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace each `...` with your own code.",
                nl="Er staat nog een `...` in je code. Vervang elke `...` door je eigen code.")

_SAME = {"en": "`{call}` gives `{got}`, and it should give `{want}`.", "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn."}


def _m(key, en, nl, **kw):
    """A mistake whose message starts with what the call gives and what it should give."""
    return Mistake(key, en=_SAME["en"] + " " + en, nl=_SAME["nl"] + " " + nl, **kw)


_AREA = ("def area(box):\n"
         "    top, left, bottom, right = box\n"
         "    return (bottom - top + 1) * (right - left + 1)\n"
         "\n\n")
_HEAD = ("def iou(a, b):\n"
         "    # a and b: (top, left, bottom, right), both ends included\n")
_OVERLAP = "max(0, bottom - top + 1) * max(0, right - left + 1)"


def _with(top: str = "max(a[0], b[0])", left: str = "max(a[1], b[1])", bottom: str = "min(a[2], b[2])",
          right: str = "min(a[3], b[3])", overlap: str = _OVERLAP, union: str = "area(a) + area(b) - overlap") -> str:
    """The course's code, with other lines for the overlap's sides, the overlap or the union."""
    return (_AREA + _HEAD + f"    top = {top}\n    left = {left}\n    bottom = {bottom}\n    right = {right}\n"
            f"    overlap = {overlap}\n    union = {union}\n" + "    return overlap / union\n")


_START = _with("...", "...", "...", "...", "...", "...")
_SOLUTION = _with()
_SWAPPED = dict(top="min(a[0], b[0])", left="min(a[1], b[1])", bottom="max(a[2], b[2])", right="max(a[3], b[3])")

EXERCISES = {
    "iou": FunctionBox(
        name="iou",
        function="iou",
        start=_START,
        solution=_SOLUTION,
        visible=[(REFERENCE["truth"], PRED)],
        # the same box; side by side; apart in both directions; touching; one pixel in common; a small box in a
        # large one, both ways round; two boxes of one pixel; the two triangles of the scene
        hidden=[((0, 0, 9, 9), (0, 0, 9, 9)), ((0, 0, 9, 9), (0, 20, 9, 29)), ((0, 0, 9, 9), (20, 20, 29, 29)),
                ((0, 0, 9, 9), (10, 0, 19, 9)), ((0, 0, 9, 9), (9, 9, 18, 18)), ((10, 10, 19, 19), (0, 0, 39, 39)),
                ((0, 0, 39, 39), (10, 10, 19, 19)), ((5, 5, 5, 5), (5, 5, 5, 5)), ((102, 8, 117, 23), (51, 7, 75, 29))],
        show=_shown,
        compare=_same,
        first=[_DOTS],
        mistakes=[
            _m("swapped", "An IoU is never above 1, and never below 0. Your overlap is the smallest box around both boxes: the smallest top and the largest bottom. The overlap starts where both boxes have begun, at the larger of the two tops and lefts, and ends where the first one ends, at the smaller of the two bottoms and rights.",
               "Een IoU is nooit groter dan 1, en nooit kleiner dan 0. Jouw overlap is de kleinste box om beide boxes heen: de kleinste bovenkant en de grootste onderkant. De overlap begint waar beide boxes begonnen zijn, bij de grootste van de twee bovenkanten en linkerkanten, en eindigt waar de eerste ophoudt, bij de kleinste van de twee onderkanten en rechterkanten.",
               wrong=_with(**_SWAPPED)),
            _m("all-max", "Your overlap takes the larger number of each pair, also for the bottom and the right. The overlap ends where the first box ends: at the smaller of the two bottoms and the smaller of the two rights.",
               "Je overlap neemt het grootste getal van elk paar, ook voor de onderkant en de rechterkant. De overlap houdt op waar de eerste box ophoudt: bij de kleinste van de twee onderkanten en de kleinste van de twee rechterkanten.",
               wrong=_with(bottom="max(a[2], b[2])", right="max(a[3], b[3])")),
            _m("all-min", "Your overlap takes the smaller number of each pair, also for the top and the left. The overlap begins where both boxes have begun: at the larger of the two tops and the larger of the two lefts.",
               "Je overlap neemt het kleinste getal van elk paar, ook voor de bovenkant en de linkerkant. De overlap begint waar beide boxes begonnen zijn: bij de grootste van de twee bovenkanten en de grootste van de twee linkerkanten.",
               wrong=_with(top="min(a[0], b[0])", left="min(a[1], b[1])")),
            _m("union-sum", "The pixels of the overlap lie in both boxes, so adding the two areas counts them twice. Subtract the overlap once: two boxes that are the same must give 1, not 0.5.",
               "De pixels van de overlap liggen in beide boxes, dus als je de twee oppervlaktes optelt, tel je ze dubbel. Trek de overlap er één keer af: twee boxes die gelijk zijn, moeten 1 geven, niet 0,5.",
               wrong=_with(union="area(a) + area(b)")),
            _m("one-box", "You divide by the area of one box. Then a box far too large that holds all of the other one gets 1. The union is what the two boxes cover together: both areas added, minus the overlap.",
               "Je deelt door de oppervlakte van één box. Dan krijgt een veel te grote box waar de andere helemaal in past een 1. De vereniging is wat de twee boxes samen bedekken: beide oppervlaktes opgeteld, min de overlap.",
               wrong=_with(union="area(a)")),
            _m("one-box-b", "You divide by the area of one box. Then a box far too large that holds all of the other one gets 1. The union is what the two boxes cover together: both areas added, minus the overlap.",
               "Je deelt door de oppervlakte van één box. Dan krijgt een veel te grote box waar de andere helemaal in past een 1. De vereniging is wat de twee boxes samen bedekken: beide oppervlaktes opgeteld, min de overlap.",
               wrong=_with(union="area(b)")),
            _m("no-clamp", "These two boxes do not overlap. Then the bottom of the overlap lies above its top, or its right side left of its left side, and a count of rows or columns comes out below 0: a negative overlap, or with two negative counts even a positive one. Make each count 0 when it comes out below 0, for instance with `max(0, ...)`. With `area` of the overlap box, the same happens.",
               "Deze twee boxes overlappen niet. Dan ligt de onderkant van de overlap boven zijn bovenkant, of zijn rechterkant links van zijn linkerkant, en komt een aantal rijen of kolommen onder 0 uit: een negatieve overlap, of met twee negatieve aantallen zelfs een positieve. Maak elk aantal 0 als het onder 0 uitkomt, bijvoorbeeld met `max(0, ...)`. Met `area` van de overlap-box gebeurt hetzelfde.",
               wrong=_with(overlap="(bottom - top + 1) * (right - left + 1)")),
            _m("clamp-product", "These two boxes lie apart both down and across: the rows and the columns of the overlap both come out below 0, and two negative numbers multiplied give a positive one. Make each of the two counts at least 0 on its own, before you multiply.",
               "Deze twee boxes liggen zowel omlaag als opzij uit elkaar: de rijen en de kolommen van de overlap komen allebei onder 0 uit, en twee negatieve getallen vermenigvuldigd geven een positief getal. Maak elk van de twee aantallen apart minstens 0, voordat je vermenigvuldigt.",
               wrong=_with(overlap="max(0, (bottom - top + 1) * (right - left + 1))")),
            _m("clamp-inside", "Your overlap gets at least one row and one column, also when the boxes do not overlap at all: the + 1 comes after `max(0, ...)`. Put it inside: `max(0, bottom - top + 1)`, so that a count below 0 becomes 0, not 1.",
               "Je overlap krijgt minstens één rij en één kolom, ook als de boxes helemaal niet overlappen: de + 1 komt na `max(0, ...)`. Zet hem erin: `max(0, bottom - top + 1)`, dan wordt een aantal onder 0 een 0, en geen 1.",
               wrong=_with(overlap="(max(0, bottom - top) + 1) * (max(0, right - left) + 1)")),
            _m("plus-one", "The overlap is too small. Both ends are part of it: from row 62 to row 86 are 86 - 62 + 1 = 25 rows, and from column 50 to column 77 are 28 columns.",
               "De overlap is te klein. Beide uiteinden horen erbij: van rij 62 tot rij 86 zijn 86 - 62 + 1 = 25 rijen, en van kolom 50 tot kolom 77 zijn 28 kolommen.",
               wrong=_with(overlap="max(0, bottom - top) * max(0, right - left)")),
            Mistake("rounds", matches=lambda got, text: re.search(r"\bround\s*\(", text) is not None,
                    en="`{call}` gives `{got}`, and it should give `{want}`. Your function rounds. Give back the share itself: 0.64 is only how the lesson writes it, and a rounded IoU can make two different boxes look the same.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je functie rondt af. Geef de verhouding zelf terug: 0,64 is alleen hoe de les hem schrijft, en een afgeronde IoU kan twee verschillende boxes gelijk laten lijken."),
            Mistake("out-of-range", matches=lambda got, text: (x := _number(got)) is not None and not 0 <= x <= 1,
                    en="`{call}` gives `{got}`, and it should give `{want}`. An IoU always lies between 0 and 1: the overlap is never below 0, and never larger than the union. Below 0, a count of rows or columns came out below 0 and was not made at least 0; above 1, the overlap holds more than the pixels in both boxes, or you divide the wrong way round.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Een IoU ligt altijd tussen 0 en 1: de overlap is nooit kleiner dan 0, en nooit groter dan de vereniging. Onder 0 kwam een aantal rijen of kolommen onder 0 uit en is het niet minstens 0 gemaakt; boven 1 bevat de overlap meer dan de pixels in beide boxes, of deel je verkeerd om."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace each `...` with your own code: the four sides of the overlap, its pixels, and the pixels of the union. Steps 1 to 3 of the recipe say how.",
                      "nl": "Vervang elke `...` door je eigen code: de vier zijden van de overlap, zijn pixels, en de pixels van de vereniging. Stap 1 tot en met 3 van het recept zeggen hoe."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with two boxes, and the `def` line of your function has a different number of names between its parentheses. Keep `(a, b)`.",
                          "nl": "De check roept `{call}` aan, met twee boxes, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(a, b)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Follow the recipe: the overlap runs from the larger of the two tops and lefts to the smaller of the two bottoms and rights; count its rows and its columns with both ends included, each at least 0; the union is both areas added, minus the overlap.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Volg het recept: de overlap loopt van de grootste van de twee bovenkanten en linkerkanten tot de kleinste van de twee onderkanten en rechterkanten; tel zijn rijen en zijn kolommen met beide uiteinden erbij, elk minstens 0; de vereniging is beide oppervlaktes opgeteld, min de overlap."},
            "almost": {"en": "Right for the circle and the predicted box, which overlap. But `{call}` gives `{got}`, and it should give `{want}`. Do these two boxes overlap, touch or lie apart? Count the rows and the columns of the overlap with both ends included, each at least 0, and work with the numbers of the boxes your function gets.",
                       "nl": "Goed voor de cirkel en de voorspelde box, die overlappen. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Overlappen deze twee boxes, raken ze elkaar of liggen ze uit elkaar? Tel de rijen en de kolommen van de overlap met beide uiteinden meegeteld, elk minstens 0, en reken met de getallen van de boxes die je functie krijgt."},
            "correct": {"en": "Your function is right in all {n} cases. For the circle and the predicted box it gives 700 / 1101 = 0.64, as counting the pixels did in the worked example, without making a mask. Two boxes that only touch give 0: a box from row 0 to row 9 and one from row 10 to row 19 have no row in common.",
                        "nl": "Je functie klopt in alle {n} gevallen. Voor de cirkel en de voorspelde box geeft hij 700 / 1101 = 0,64, net als het tellen van de pixels in het uitgewerkte voorbeeld, zonder een mask te maken. Twee boxes die elkaar alleen raken, geven 0: een box van rij 0 tot rij 9 en een van rij 10 tot rij 19 hebben geen rij gemeen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("iou", _START, "empty", None),
    ("iou", _SOLUTION, "correct", None),
    ("iou", _with(overlap="area((top, left, bottom, right)) if bottom >= top and right >= left else 0"), "correct", None),
    ("iou", _with(overlap="max(bottom - top + 1, 0) * max(0, 1 + right - left)"), "correct", None),
    ("iou", "def iou(a, b):\n    t, l = max(a[0], b[0]), max(a[1], b[1])\n    bo, r = min(a[2], b[2]), min(a[3], b[3])\n"
            "    o = max(0, bo - t + 1) * max(0, r - l + 1)\n"
            "    return o / ((a[2] - a[0] + 1) * (a[3] - a[1] + 1) + (b[2] - b[0] + 1) * (b[3] - b[1] + 1) - o)\n", "correct", None),
    ("iou", _with(union="area(b) + area(a) - overlap"), "correct", None),
    ("iou", _with(**_SWAPPED), "wrong", "swapped"),
    ("iou", _with(union="area(a) + area(b)"), "wrong", "union-sum"),
    ("iou", _with(union="area(a)"), "wrong", "one-box"),
    ("iou", _with(union="area(b)"), "wrong", "one-box-b"),
    ("iou", _with(overlap="(bottom - top + 1) * (right - left + 1)"), "wrong", "no-clamp"),
    ("iou", _with(overlap="area((top, left, bottom, right))"), "wrong", "no-clamp"),
    ("iou", _with(overlap="max(0, (bottom - top + 1) * (right - left + 1))"), "wrong", "clamp-product"),
    ("iou", _with(overlap="(max(0, bottom - top) + 1) * (max(0, right - left) + 1)"), "wrong", "clamp-inside"),
    ("iou", _with(overlap="max(0, bottom - top) * max(0, right - left)"), "wrong", "plus-one"),
    ("iou", _with(bottom="max(a[2], b[2])", right="max(a[3], b[3])"), "wrong", "all-max"),
    ("iou", _with(top="min(a[0], b[0])", left="min(a[1], b[1])"), "wrong", "all-min"),
    ("iou", _SOLUTION.replace("return overlap / union", "return round(overlap / union, 2)"), "wrong", "rounds"),
    ("iou", _SOLUTION.replace("return overlap / union", "return union / overlap"), "wrong", "out-of-range"),
    ("iou", _with(top="min(a[0], b[0])"), "wrong", None),
    # only the rows made at least 0; abs instead of max; boxes that only overlap in one pixel taken as apart
    ("iou", _with(overlap="max(0, bottom - top + 1) * (right - left + 1)"), "wrong", "out-of-range"),
    ("iou", _with(overlap="abs(bottom - top + 1) * abs(right - left + 1)"), "almost", None),
    ("iou", _AREA + _HEAD + "    if a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]:\n        return 0\n"
     "    top, left = max(a[0], b[0]), max(a[1], b[1])\n    bottom, right = min(a[2], b[2]), min(a[3], b[3])\n"
     "    overlap = (bottom - top + 1) * (right - left + 1)\n    return overlap / (area(a) + area(b) - overlap)\n", "almost", None),
    ("iou", _with(overlap="max(0, bottom - top) * max(0, right - left + 1)"), "wrong", None),
    ("iou", _with(top="max(a[0], b[0])", left="max(a[0], b[0])"), "wrong", None),
    # a share written out by hand: right for the circle and the predicted box only
    ("iou", _AREA + "def iou(a, b):\n    return 700 / 1101\n", "almost", None),
    # text instead of a number
    ("iou", _SOLUTION.replace("return overlap / union", "return str(overlap / union)"), "wrong", None),
    ("iou", _with(top="...", left="max(a[1], b[1])"), "wrong", "dots-left"),
    ("iou", _SOLUTION.replace("def iou(a, b):", "def iou(a):"), "error", "arguments"),
    ("iou", _SOLUTION.replace("return overlap / union", "print(overlap / union)"), "wrong", "prints"),
]
