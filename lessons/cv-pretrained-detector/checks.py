# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-pretrained-detector: to_pixels, which turns a box of DETR into pixels. It is checked on
boxes the course gives, so it works without DETR. DETR with ResNet-50 (167 MB, from Hugging Face) and Caltech-101
(137 MB, from CaltechDATA) the lesson downloads once, when the learner asks for them (everything_ai.hub,
everything_ai.caltech): the worked example and the demo need them, the exercises do not. The picture and the slider
use what the course measured with them (REFERENCE)."""
import errno
import functools
import html
import re
import urllib.error

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import caltech, chart, detr, hub, paths
from everything_ai.lesson import download_card
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit

DOWNLOAD_LINE = {
    "en": "Works on every computer. DETR, 167 MB, is downloaded once from Hugging Face, and Caltech-101, 137 MB of photos, from Caltech, when you click their button in the lesson; after that, the lesson works without a connection. The exercises work without them too.",
    "nl": "Werkt op elke computer. DETR, 167 MB, wordt één keer gedownload van Hugging Face, en Caltech-101, 137 MB aan foto's, van Caltech, als je in de les op de knop ervoor klikt; daarna werkt de les zonder verbinding. De oefeningen werken ook zonder.",
}

PHOTOS = ("coffee", "chelsea", "fruit", "cell")   # the course's own photos in the picture (data/photos)
CHAIRS, STARFISH = "windsor_chair", "starfish"    # a kind DETR learned, and one it did not
COUNT = 30                                        # the photos of each, as in the lessons before
CHAIR = detr.KINDS.index("chair")
THRESHOLD = 0.9                                   # the score threshold of the worked example
LOW = 0.05                                        # the lowest score the course keeps in REFERENCE
STOPS = [round(0.05 * k, 2) for k in range(1, 20)] + [0.99]   # the score thresholds of the slider
TASK = 10                                         # the task: at most this many starfish photos with a box
DEMO = {CHAIRS: (0, 2, 13, 20), STARFISH: (1, 4, 12, 21)}   # the photos of the demo: places in photos(name)
DEMO_LOW = 0.3                                    # the lowest threshold of the demo

# The numbers in the text. `measure(model)` measures them again with DETR and Caltech-101; the test of the lesson
# checks that both are the same.
REFERENCE = {
    # the course's photos: their size, the size DETR gets them in (rows, columns), and every query with a score above
    # LOW: (score, kind, box)
    "photos": {
        "coffee": {"size": (600, 400), "model": (800, 1200), "queries": (
            (0.997, 'cup', (0.486, 0.381, 0.397, 0.674)),
            (0.993, 'dining table', (0.5, 0.496, 1.0, 0.986)),
            (0.98, 'spoon', (0.626, 0.492, 0.168, 0.657)),
            (0.293, 'spoon', (0.608, 0.679, 0.135, 0.287)),
        )},
        "chelsea": {"size": (451, 300), "model": (800, 1202), "queries": (
            (0.996, 'cat', (0.473, 0.5, 0.945, 1.0)),
            (0.861, 'carrot', (0.592, 0.823, 0.113, 0.123)),
            (0.188, 'cat', (0.5, 0.5, 1.0, 1.0)),
        )},
        "fruit": {"size": (360, 240), "model": (800, 1200), "queries": (
            (0.999, 'sports ball', (0.42, 0.668, 0.2, 0.306)),
            (0.998, 'sports ball', (0.168, 0.733, 0.166, 0.251)),
            (0.998, 'sports ball', (0.197, 0.29, 0.187, 0.285)),
            (0.998, 'sports ball', (0.726, 0.732, 0.195, 0.295)),
            (0.997, 'sports ball', (0.487, 0.291, 0.18, 0.274)),
            (0.996, 'sports ball', (0.921, 0.689, 0.146, 0.221)),
            (0.989, 'sports ball', (0.796, 0.314, 0.154, 0.242)),
        )},
        "cell": {"size": (550, 660), "model": (960, 800), "queries": (
            (0.997, 'clock', (0.775, 0.574, 0.258, 0.208)),
            (0.138, 'bed', (0.5, 0.5, 1.0, 1.0)),
        )},
    },
    # of the 30 chair photos, those with a box of the kind chair; of the 30 starfish photos, those with any box
    "found": {0.05: 27, 0.1: 26, 0.15: 26, 0.2: 25, 0.25: 25, 0.3: 25, 0.35: 25, 0.4: 25, 0.45: 24, 0.5: 23, 0.55: 23, 0.6: 23, 0.65: 23, 0.7: 23, 0.75: 22, 0.8: 22, 0.85: 21, 0.9: 20, 0.95: 19, 0.99: 12},
    "starfish": {0.05: 30, 0.1: 30, 0.15: 30, 0.2: 30, 0.25: 30, 0.3: 30, 0.35: 29, 0.4: 28, 0.45: 25, 0.5: 23, 0.55: 22, 0.6: 21, 0.65: 21, 0.7: 21, 0.75: 18, 0.8: 16, 0.85: 15, 0.9: 13, 0.95: 9, 0.99: 4},
    # at 0.5 and 0.9: all boxes on the starfish photos
    "boxes": {0.5: 39, 0.9: 14},
    "never": 3,   # chair photos without a chair box, even at LOW
    # the boxes above 0.99 on starfish photos: (score, kind, place in photos("starfish"))
    "surest": ((0.999, 'snowboard', 4), (0.996, 'teddy bear', 19), (0.996, 'bird', 26), (0.993, 'bird', 8)),
    "task": 0.95,   # the lowest stop with at most TASK starfish photos with a box
    "margin": True,   # no score that decides a number lies within 0.001 of its threshold
    # the worked example: the queries it keeps, by number and kind; the queries that give no object the highest
    # probability, and the lowest such probability
    "example": ((59, 'dining table'), (71, 'cup'), (92, 'spoon')),
    "nothing": (97, 0.707),
    # the photos of the demo, by "class/place": every query with a score above DEMO_LOW
    "demo": {
        "windsor_chair/0": ((0.978, 'cat', (0.49, 0.566, 0.61, 0.153)), (0.976, 'chair', (0.482, 0.499, 0.953, 0.929)), (0.777, 'chair', (0.484, 0.637, 0.946, 0.64)),),
        "windsor_chair/2": ((0.997, 'chair', (0.517, 0.519, 0.697, 0.908)),),
        "windsor_chair/13": ((0.885, 'chair', (0.489, 0.511, 0.93, 0.973)), (0.81, 'chair', (0.564, 0.507, 0.785, 0.979)), (0.363, 'chair', (0.569, 0.447, 0.78, 0.858)), (0.322, 'bench', (0.07, 0.675, 0.14, 0.617)),),
        "windsor_chair/20": (),
        "starfish/1": ((0.709, 'airplane', (0.157, 0.307, 0.313, 0.16)), (0.7, 'airplane', (0.275, 0.302, 0.541, 0.17)), (0.499, 'airplane', (0.279, 0.241, 0.542, 0.279)), (0.437, 'chair', (0.881, 0.851, 0.239, 0.195)), (0.426, 'airplane', (0.428, 0.149, 0.248, 0.101)),),
        "starfish/4": ((0.999, 'snowboard', (0.5, 0.502, 0.969, 0.99)),),
        "starfish/12": ((0.837, 'teddy bear', (0.5, 0.5, 0.994, 1.0)), (0.521, 'suitcase', (0.5, 0.5, 1.0, 1.0)),),
        "starfish/21": ((0.969, 'cat', (0.339, 0.689, 0.677, 0.615)), (0.817, 'cat', (0.638, 0.681, 0.713, 0.6)), (0.722, 'cat', (0.5, 0.5, 1.0, 1.0)), (0.395, 'cat', (0.5, 0.685, 0.997, 0.623)),),
    },
}


# --- DETR on a photo, as the lesson does it ----------------------------------------------------------------------------

def photo(name: str):
    """One of the course's own photos, opened with Pillow."""
    from PIL import Image
    return Image.open(paths.ROOT / "data" / "photos" / f"{name}.png")


def queries(model, img) -> list[tuple[float, int, tuple]]:
    """What DETR says about a photo: for each of its 100 queries the score, the kind and the box, as the lesson works
    them out. The score is the highest probability softmax gives a kind, without the last, "no object"."""
    with torch.no_grad():
        logits, boxes = model(detr.prepare(img))
    p = torch.softmax(logits[0], dim=-1)[:, :-1]
    scores, kinds = p.max(dim=-1)
    return [(float(scores[i]), int(kinds[i]), tuple(float(v) for v in boxes[0, i])) for i in range(detr.QUERIES)]


def kept(found, threshold: float) -> list:
    """The queries whose score is above the threshold, the highest score first."""
    return sorted((q for q in found if q[0] > threshold), key=lambda q: -q[0])


def to_pixels(box, width: float, height: float) -> tuple[float, float, float, float]:
    """A box of DETR, (center x, center y, width, height) as fractions of the photo, as (top, left, bottom, right) in
    pixels: the lines where the box starts and ends, measured from the photo's top-left corner."""
    cx, cy, w, h = box
    return ((cy - h / 2) * height, (cx - w / 2) * width, (cy + h / 2) * height, (cx + w / 2) * width)


def photos(name: str) -> list:
    """The 30 photos of a class the lesson uses, without people: everything_ai.caltech.checked."""
    return caltech.checked(name, COUNT)


def _rounded(found, low: float = LOW) -> tuple:
    """Queries with a score above `low`, rounded as REFERENCE keeps them: (score, kind name, box)."""
    return tuple((round(s, 3), detr.KINDS[k], tuple(round(v, 3) for v in b)) for s, k, b in kept(found, low))


def best(found, kind: int | None = None) -> float:
    """The highest score among a photo's queries, of one kind or of any: the photo has a box above every threshold
    below it."""
    return max((q[0] for q in found if kind is None or q[1] == kind), default=0.0)


def measure(model) -> dict:
    """Every number of REFERENCE, measured with DETR on the course's photos and on 30 photos each of Windsor chairs
    and starfish from Caltech-101."""
    from PIL import Image
    out = {"photos": {}}
    course = {}
    for name in PHOTOS:
        img = photo(name)
        course[name] = queries(model, img)
        out["photos"][name] = {"size": img.size, "model": tuple(detr.prepare(img).shape[2:]),
                               "queries": _rounded(course[name])}
    found = {c: [queries(model, Image.open(p)) for p in photos(c)] for c in (CHAIRS, STARFISH)}
    chairs = [best(f, CHAIR) for f in found[CHAIRS]]      # each chair photo's surest chair box
    starfish = [best(f) for f in found[STARFISH]]         # each starfish photo's surest box, of any kind
    out["found"] = {t: sum(s > t for s in chairs) for t in STOPS}
    out["starfish"] = {t: sum(s > t for s in starfish) for t in STOPS}
    out["boxes"] = {t: sum(len(kept(f, t)) for f in found[STARFISH]) for t in (0.5, THRESHOLD)}
    out["never"] = sum(s <= LOW for s in chairs)
    out["surest"] = tuple(sorted(((round(q[0], 3), detr.KINDS[q[1]], i) for i, f in enumerate(found[STARFISH])
                                  for q in kept(f, 0.99)), reverse=True))
    out["task"] = min(t for t in STOPS if out["starfish"][t] <= TASK)
    # how near a score that decides a number comes to its threshold, so that the test can see that no number hangs on
    # a last decimal: the surest box of each photo against every stop, every box against the thresholds whose boxes
    # the text counts, and the boxes on the course's photos against the thresholds the lesson shows them at
    near = [abs(s - t) for s in chairs + starfish for t in STOPS]
    near += [abs(q[0] - t) for c in found.values() for f in c for q in f for t in (0.5, THRESHOLD)]
    near += [abs(q[0] - t) for f in course.values() for q in f for t in (0.2, 0.5, THRESHOLD)]
    out["margin"] = min(near) > 1e-3
    out["demo"] = {f"{c}/{i}": _rounded(found[c][i], DEMO_LOW) for c in DEMO for i in DEMO[c]}
    # the worked example: the queries it keeps on the coffee photo, by number; and the queries whose highest
    # probability, among all 92, is no object, with the lowest such probability
    out["example"] = tuple((i, detr.KINDS[q[1]]) for i, q in enumerate(course["coffee"]) if q[0] > THRESHOLD)
    with torch.no_grad():
        p = torch.softmax(model(detr.prepare(photo("coffee")))[0][0], dim=-1)
    most = p.argmax(dim=-1) == len(detr.KINDS) - 1
    out["nothing"] = (int(most.sum()), round(float(p[most, -1].min()), 3))
    return out


# --- DETR and Caltech-101 in the lesson -------------------------------------------------------------------------------

def ready() -> bool:
    return detr.downloaded() and caltech.unpacked()


@functools.lru_cache(maxsize=1)
def _detr():
    return detr.load()


def detr_or_stop(button, texts: dict):
    """DETR for the lesson's cells, with Caltech-101 unpacked. When either is not there yet, the cell stops with
    `texts["ask"]` and the button; a click downloads what is missing and unpacks the photos, with the progress in the
    cell. While it downloads, the lesson's other cells wait. `texts` has "ask", "busy" ({name}, {done} and {total} in
    MB), "unpacking" ({done} photos), "no_net", "not_pinned", "no_room" and "failed" ({why}): why it did not work;
    and "here"."""
    import marimo as mo
    failed = None
    if not ready() and button.value:
        try:
            for name, pin in (("DETR", detr.FILE), ("Caltech-101", caltech.ZIP)):
                if hub.pinned(*pin) is not None:
                    continue
                total = round(hub.PINNED[pin]["size"] / 1e6)
                last = [-1]

                def progress(done, size, name=name, total=total, last=last):
                    mb = int(done / 1e6)
                    if mb // 5 != last[0] // 5:   # every 5 MB
                        last[0] = mb
                        mo.output.replace(mo.md(texts["busy"].format(name=name, done=mb, total=total)))
                hub.download(*pin, progress=progress)
            caltech.unpack(progress=lambda done: mo.output.replace(mo.md(texts["unpacking"].format(done=done))))
        except hub.NotPinned:
            failed = texts["not_pinned"]
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            failed = texts["no_net"]
        except OSError as e:
            failed = texts["no_room"] if e.errno == errno.ENOSPC else texts["failed"].format(why=html.escape(str(e)))
    if not ready():
        mo.stop(True, download_card(texts["ask"], button, failed))
    model = _detr()
    mo.output.replace(mo.md(texts["here"]))
    return model


@functools.lru_cache(maxsize=1)
def demo_queries(model) -> dict:
    """What DETR says about the 8 photos of the demo, computed once: their queries, by "class/place"."""
    from PIL import Image
    return {f"{c}/{i}": queries(model, Image.open(photos(c)[i])) for c in DEMO for i in DEMO[c]}


# --- the pictures ------------------------------------------------------------------------------------------------------

def number(v: float, lang: str, digits: int = 2) -> str:
    s = f"{v:.{digits}f}"
    return s.replace(".", ",") if lang == "nl" else s


def _line(ax, box):
    """A box of DETR on a photo, as the lessons before draw a predicted box: dashed in the color of a prediction, with
    an edge in the color of the page, so that it stands out on every photo, in light and in dark."""
    from everything_ai import plots
    top, left, bottom, right = box
    xs, ys = [left, right, right, left, left], [top, top, bottom, bottom, top]
    ax.plot(xs, ys, color=plots.PAGE, linestyle="-", linewidth=3.4, solid_joinstyle="miter")
    ax.plot(xs, ys, color=plots.PRED, linestyle=(0, (3.0, 1.6)), linewidth=1.6, solid_joinstyle="miter",
            dash_capstyle="butt")


def _label(ax, box, words: str, placed: list, renderer):
    """The kind and score of a box in a label at its top-left corner, over every line. A label that would cover one
    in `placed` moves down until it is free, and joins them."""
    from everything_ai import plots
    top, left = box[0], box[1]
    t = ax.text(left, top, words, fontsize=7, color=plots.PRED_TEXT, va="top", ha="left", linespacing=1.1, zorder=5,
                bbox={"boxstyle": "square,pad=0.2", "facecolor": plots.PAGE, "edgecolor": "none"})
    for _ in range(len(placed) + 1):
        here = t.get_window_extent(renderer).expanded(1.0, 1.15)
        hit = next((b for b in placed if here.overlaps(b)), None)
        if hit is None:
            break
        x, y = ax.transData.transform(t.get_position())
        t.set_position(ax.transData.inverted().transform((x, y - (here.y1 - hit.y0) - 1)))
    placed.append(t.get_window_extent(renderer).expanded(1.0, 1.15))


def boxes_picture(images: list, found: list, threshold: float, labels: dict, lang: str, fn=to_pixels,
                  cols: int = 4, height: float = 2.2, width: float = 7.2) -> str:
    """Photos side by side, each with the boxes of DETR whose score is above the threshold, drawn with `fn` (the
    learner's to_pixels, or the course's). `found[i]` holds the queries of photo i as (score, kind name, box), with
    COCO's English names as DETR gives them, in both languages; labels has "titles" (one per photo) and "alt". The
    box with the highest score is drawn last, on top; its label, with its kind above its score, keeps its place, and
    labels of lower scores move down where they would cover it."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(width, height * rows), squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.05, hspace=0.18)
    renderer = fig.canvas.get_renderer()
    for ax, img, qs, title in zip(axes.flat, images, found, labels["titles"]):
        # the photo's edges at 0 and at its width and height, as DETR measures a box: pixel k spans k to k + 1
        ax.imshow(np.asarray(img.convert("RGB")), interpolation="antialiased", extent=(0, img.width, img.height, 0))
        ax.set_xlim(0, img.width)
        ax.set_ylim(img.height, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9)
        ax.apply_aspect()   # the photo's place in the figure as it is drawn, so that the labels can be measured
        shown = [(score, kind, fn(box, img.width, img.height)) for score, kind, box in qs if score > threshold]
        for score, kind, box in sorted(shown, key=lambda q: q[0]):
            _line(ax, box)
        placed = []
        for score, kind, box in sorted(shown, key=lambda q: -q[0]):
            _label(ax, box, f"{kind}\n{number(score, lang)}", placed, renderer)
    for ax in list(axes.flat)[len(images):]:
        ax.axis("off")
    out = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return out


def course_picture(labels: dict, lang: str, threshold: float = 0.5) -> str:
    """The course's four photos with DETR's boxes above `threshold`, from what the course measured."""
    return boxes_picture([photo(n) for n in PHOTOS], [REFERENCE["photos"][n]["queries"] for n in PHOTOS], threshold,
                         labels, lang, cols=2, height=3.0, width=6.4)


def demo(model, text, threshold: float, labels: dict, lang: str) -> dict:
    """The 8 photos of the demo with DETR's boxes above `threshold`, drawn with the learner's to_pixels once it is
    right, else with the course's: the picture, whose function it was, and of the 4 chair photos those with a chair
    box and of the 4 starfish photos those with any box. If the learner's function fails here after all, the course's
    draws instead. The cell passes the model it got from detr_or_stop, so that it runs only once the downloads are
    there."""
    from PIL import Image
    ran = run_code(text) if text else None
    fn = ran.names.get("to_pixels") if ran is not None and ran.error is None else None
    places = [(c, i) for c in DEMO for i in DEMO[c]]
    found = demo_queries(model)
    images = [Image.open(photos(c)[i]) for c, i in places]
    qs = [[(s, detr.KINDS[k], b) for s, k, b in found[f"{c}/{i}"]] for c, i in places]
    out = {"chairs": sum(best(found[f"{CHAIRS}/{i}"], CHAIR) > threshold for i in DEMO[CHAIRS]),
           "starfish": sum(best(found[f"{STARFISH}/{i}"]) > threshold for i in DEMO[STARFISH])}
    if callable(fn):
        try:
            return out | {"picture": with_time_limit(lambda: boxes_picture(images, qs, threshold, labels, lang, fn=fn), 30),
                          "yours": True}
        except Exception:
            pass
    return out | {"picture": boxes_picture(images, qs, threshold, labels, lang), "yours": False}


def threshold_frame(t: float, labels: dict, lang: str) -> tuple[str, int, int]:
    """For the slider: of the 30 chair photos, those with a box of the kind chair, and of the 30 starfish photos, those
    with any box, against the score threshold; the threshold t marked. Gives the chart and both counts at t. `labels`
    has "x", "y", "chairs", "starfish" and "alt" ({t}, {found} and {starfish})."""
    found, star = REFERENCE["found"], REFERENCE["starfish"]
    panel = chart.Panel((0, 1), (0, 30), xlabel=labels["x"], ylabel=labels["y"], xticks=[0, 0.25, 0.5, 0.75, 1],
                        yticks=[0, 10, 20, 30])
    panel.line([t, t], [0, 30], role="hyper", width=1.4, dashed=True)
    panel.line(STOPS, [found[x] for x in STOPS], role="ink")
    panel.line(STOPS, [star[x] for x in STOPS], role="loss", dashed=True)
    panel.points([t], [found[t]], role="ink", r=5)
    panel.points([t], [star[t]], role="loss", r=5)
    # the label of the chairs above the lines on the right, where both stay at 23 or below; the label of the starfish
    # under the end of its line
    panel.text(0.98, 26.5, labels["chairs"], role="ink", anchor="end")
    panel.text(0.98, 1.0, labels["starfish"], role="loss", anchor="end")
    alt = labels["alt"].format(t=number(t, lang), found=found[t], starfish=star[t])
    svg = chart.figure([panel], alt=alt, width=440, height=270, drag=0)
    return svg, found[t], star[t]


# The lesson is done when these are correct. "score" and "apps" are choice questions, "starfish" the open one.
REQUIRED = ["score", "pixels", "apps", "starfish"]

# --- Exercise 2: to_pixels ---------------------------------------------------------------------------------------------

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace each `...` with your own code.",
                nl="Er staat nog een `...` in je code. Vervang elke `...` door je eigen code.")

_SAME = {"en": "`{call}` gives `{got}`, and it should give `{want}`.", "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn."}


def _m(key, en, nl, **kw):
    """A mistake whose message starts with what the call gives and what it should give."""
    return Mistake(key, en=_SAME["en"] + " " + en, nl=_SAME["nl"] + " " + nl, **kw)


_HEAD = ("def to_pixels(box, width, height):\n"
         "    # box: (center x, center y, width, height), each a fraction of the photo\n"
         "    cx, cy, w, h = box\n")


def _with(top: str = "(cy - h / 2) * height", left: str = "(cx - w / 2) * width",
          bottom: str = "(cy + h / 2) * height", right: str = "(cx + w / 2) * width",
          order: str = "top, left, bottom, right") -> str:
    """The course's code, with other lines for the sides of the box, or another order."""
    return (_HEAD + f"    top = {top}\n    left = {left}\n    bottom = {bottom}\n    right = {right}\n"
            + f"    return ({order})\n")


_START = _with("...", "...", "...", "...")


def _rounded_cup(got, text) -> bool:
    """Whole numbers, each within a pixel of the cup's box: the course's answer, rounded. A rounded answer always
    fails on the cup first, whose box has decimals."""
    a = _numbers(got)
    cup = REFERENCE["photos"]["coffee"]["queries"][0][2]
    return (a is not None and a.shape == (4,) and bool(np.all(np.mod(a, 1) == 0))
            and bool(np.all(np.abs(a - np.asarray(to_pixels(cup, 600, 400))) <= 1)))
_SOLUTION = _with()


def _numbers(got):
    """What the learner's function gave back as an array of floats, or None if it is not numbers."""
    a = np.asarray(got, dtype=object)
    if any(isinstance(x, (str, bytes, bool, np.bool_)) for x in a.ravel()):
        return None
    try:
        return np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return None


def _shown(v) -> str:
    """Four numbers between parentheses, to two decimals."""
    a = _numbers(v)
    if a is None or a.ndim != 1:
        return str(v)
    return "(" + ", ".join(f"{float(x):.2f}" for x in a) + ")"


def _same(got, want) -> bool:
    """Right when the four numbers are the course's, in the same order, to a thousandth of a pixel: also when they are
    computed in float32."""
    a = _numbers(got)
    return a is not None and a.shape == (4,) and bool(np.allclose(a, np.asarray(want, dtype=float), rtol=0, atol=1e-3))


def _cases():
    """The cup on the coffee photo (600 by 400), and the other boxes the check tries: the spoon, Chelsea and the box
    on her nose, the cell on its tall photo, a box over the whole photo, and a small one in a corner."""
    p = REFERENCE["photos"]
    cup, _, spoon = (q[2] for q in p["coffee"]["queries"][:3])
    cat, nose = (q[2] for q in p["chelsea"]["queries"][:2])
    cell = p["cell"]["queries"][0][2]
    return ([(cup, 600, 400)],
            [(spoon, 600, 400), (cat, 451, 300), (nose, 451, 300), (cell, 550, 660), ((0.5, 0.5, 1.0, 1.0), 640, 480),
             ((0.05, 0.9, 0.1, 0.2), 200, 300)])


def _exercise():
    visible, hidden = _cases()
    return FunctionBox(
        name="pixels",
        function="to_pixels",
        start=_START,
        solution=_SOLUTION,
        visible=visible,
        hidden=hidden,
        show=_shown,
        compare=_same,
        first=[_DOTS],
        mistakes=[
            Mistake("unpack", error=lambda e: isinstance(e, ValueError) and "unpack" in str(e),
                    en="With `{call}`, your function stops with an error in the line that takes the box apart: a box of DETR holds four numbers, so that line needs four names, `cx, cy, w, h = box`.",
                    nl="Met `{call}` stopt je functie met een fout in de regel die de box uit elkaar haalt: een box van DETR heeft vier getallen, dus die regel heeft vier namen nodig, `cx, cy, w, h = box`."),
            _m("xyxy", "Those are the right four numbers, in the order of PyTorch's torchvision and Hugging Face: (left, top, right, bottom), x before y. The course writes a box as (top, left, bottom, right), rows first.",
               "Dat zijn de goede vier getallen, in de volgorde van torchvision en Hugging Face: (left, top, right, bottom), x voor y. De course schrijft een box als (top, left, bottom, right), de rijen eerst.",
               wrong=_with(order="left, top, right, bottom")),
            _m("fractions", "Those are fractions of the photo, between 0 and 1. Multiply what runs down, the top and the bottom, by the height of the photo, and what runs across by its width.",
               "Dat zijn delen van de foto, tussen 0 en 1. Vermenigvuldig wat omlaag loopt, de boven- en de onderkant, met de hoogte van de foto, en wat opzij loopt met zijn breedte.",
               wrong=_with("cy - h / 2", "cx - w / 2", "cy + h / 2", "cx + w / 2")),
            _m("whole", "Your box is twice as large as DETR's: from the center, the box reaches half its width to each side, `w / 2`, and half its height up and down.",
               "Je box is twee keer zo groot als die van DETR: vanuit het midden reikt de box de helft van zijn breedte naar elke kant, `w / 2`, en de helft van zijn hoogte omhoog en omlaag.",
               wrong=_with("(cy - h) * height", "(cx - w) * width", "(cy + h) * height", "(cx + w) * width")),
            _m("corner", "That treats `cx` and `cy` as the top-left corner, as COCO's own files write a box. DETR gives the center: the box starts half its width to the left of `cx`, and half its height above `cy`.",
               "Zo behandel je `cx` en `cy` als de hoek linksboven, zoals de bestanden van COCO zelf een box schrijven. DETR geeft het midden: de box begint de helft van zijn breedte links van `cx`, en de helft van zijn hoogte boven `cy`.",
               wrong=_with("cy * height", "cx * width", "(cy + h) * height", "(cx + w) * width")),
            _m("sizes-swapped", "The width and the height of the photo are swapped. What runs across, `cx` and `w`, goes with the photo's width; what runs down, `cy` and `h`, with its height.",
               "De breedte en de hoogte van de foto zijn omgewisseld. Wat opzij loopt, `cx` en `w`, hoort bij de breedte van de foto; wat omlaag loopt, `cy` en `h`, bij de hoogte.",
               wrong=_with("(cy - h / 2) * width", "(cx - w / 2) * height", "(cy + h / 2) * width", "(cx + w / 2) * height")),
            _m("xy-swapped", "x and y are swapped: `cx` and `w` run across, so they give the left and the right side; `cy` and `h` run down, so they give the top and the bottom.",
               "x en y zijn omgewisseld: `cx` en `w` lopen opzij, dus die geven de linker- en de rechterkant; `cy` en `h` lopen omlaag, dus die geven de boven- en de onderkant.",
               wrong=_with("(cx - w / 2) * height", "(cy - h / 2) * width", "(cx + w / 2) * height", "(cy + h / 2) * width")),
            _m("size", "The last two numbers are the size of the box in pixels. The course wants where the box ends: the bottom, the center plus half the height, and the right side.",
               "De laatste twee getallen zijn de grootte van de box in pixels. De course wil waar de box eindigt: de onderkant, het midden plus de helft van de hoogte, en de rechterkant.",
               wrong=_with(bottom="h * height", right="w * width")),
            _m("rounded", "Your numbers are rounded to whole pixels. DETR's box can start halfway through a pixel; keep the decimals, so that the box is DETR's own.",
               "Je getallen zijn afgerond op hele pixels. De box van DETR kan halverwege een pixel beginnen; rond niet af, dan is de box die van DETR zelf.",
               matches=_rounded_cup),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace each `...` with a number of pixels made from the box and the size of the photo: from the center, the box reaches half its width to each side and half its height up and down.",
                      "nl": "Vervang elke `...` door een aantal pixels uit de box en de grootte van de foto: vanuit het midden reikt de box de helft van zijn breedte naar elke kant en de helft van zijn hoogte omhoog en omlaag."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with a box, the width and the height of the photo, and the `def` line of your function has a different number of names between its parentheses. Keep `(box, width, height)`.",
                          "nl": "De check roept `{call}` aan, met een box, de breedte en de hoogte van de foto, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(box, width, height)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. The top is the center minus half the height, times the height of the photo; the left side the center minus half the width, times its width; the bottom and the right side the same with plus.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. De bovenkant is het midden min de helft van de hoogte, keer de hoogte van de foto; de linkerkant het midden min de helft van de breedte, keer zijn breedte; de onderkant en de rechterkant hetzelfde met plus."},
            "almost": {"en": "Right for the cup. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work with the box and the size it is given, for every photo?",
                       "nl": "Goed voor het kopje. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Rekent je functie met de box en de grootte die hij krijgt, voor elke foto?"},
            "correct": {"en": "Your function is right in all {n} cases. The cup's box runs from row 17.6 to row 287.2 and from column 172.5 to column 410.7: `(17.60, 172.50, 287.20, 410.70)`. A box over the whole photo, `(0.5, 0.5, 1.0, 1.0)`, runs from the top-left corner, `(0, 0)`, to the photo's height and width: its edges are the photo's own, so its right side is 640 on a photo 640 pixels wide, where the last column of pixels is number 639.",
                        "nl": "Je functie klopt in alle {n} gevallen. De box van het kopje loopt van rij 17,6 tot rij 287,2 en van kolom 172,5 tot kolom 410,7: `(17.60, 172.50, 287.20, 410.70)`. Een box over de hele foto, `(0.5, 0.5, 1.0, 1.0)`, loopt van de hoek linksboven, `(0, 0)`, tot de hoogte en de breedte van de foto: zijn randen zijn die van de foto zelf, dus zijn rechterkant is 640 op een foto van 640 pixels breed, waar de laatste kolom pixels nummer 639 is."},
        },
    )


EXERCISES = {"pixels": _exercise()}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("pixels", _START, "empty", None),
    ("pixels", _SOLUTION, "correct", None),
    ("pixels", _SOLUTION.replace("return (top, left, bottom, right)", "return [top, left, bottom, right]"), "correct", None),
    ("pixels", _with("cy * height - h * height / 2", "cx * width - w * width / 2", "cy * height + h * height / 2",
                     "cx * width + w * width / 2"), "correct", None),
    ("pixels", "def to_pixels(box, width, height):\n    return ((box[1] - box[3] / 2) * height, (box[0] - box[2] / 2) * width,\n"
               "            (box[1] + box[3] / 2) * height, (box[0] + box[2] / 2) * width)\n", "correct", None),
    ("pixels", _HEAD + "    half_w, half_h = w / 2, h / 2\n    return ((cy - half_h) * height, (cx - half_w) * width, (cy + half_h) * height, (cx + half_w) * width)\n",
     "correct", None),
    ("pixels", _with(order="left, top, right, bottom"), "wrong", "xyxy"),
    ("pixels", _with("cy - h / 2", "cx - w / 2", "cy + h / 2", "cx + w / 2"), "wrong", "fractions"),
    ("pixels", _with("(cy - h) * height", "(cx - w) * width", "(cy + h) * height", "(cx + w) * width"), "wrong", "whole"),
    ("pixels", _with("cy * height", "cx * width", "(cy + h) * height", "(cx + w) * width"), "wrong", "corner"),
    ("pixels", _with("(cy - h / 2) * width", "(cx - w / 2) * height", "(cy + h / 2) * width", "(cx + w / 2) * height"),
     "wrong", "sizes-swapped"),
    ("pixels", _with("(cx - w / 2) * height", "(cy - h / 2) * width", "(cx + w / 2) * height", "(cy + h / 2) * width"),
     "wrong", "xy-swapped"),
    ("pixels", _with(bottom="h * height", right="w * width"), "wrong", "size"),
    ("pixels", _SOLUTION.replace("return (top, left, bottom, right)", "return (round(top), round(left), round(bottom), round(right))"),
     "wrong", "rounded"),
    ("pixels", _SOLUTION.replace("return (top, left, bottom, right)", "return (int(top), int(left), int(bottom), int(right))"),
     "wrong", "rounded"),
    ("pixels", _START.replace("top = ...", "top = (cy - h / 2) * height"), "wrong", "dots-left"),
    ("pixels", _SOLUTION.replace("cx, cy, w, h = box", "cx, cy, w = box"), "error", "unpack"),
    ("pixels", _SOLUTION.replace("def to_pixels(box, width, height):", "def to_pixels(box):"), "error", "arguments"),
    ("pixels", _SOLUTION.replace("    return (top, left, bottom, right)\n", "    print(top, left, bottom, right)\n"), "wrong", "prints"),
    ("pixels", _SOLUTION.replace("    return (top, left, bottom, right)\n", ""), "wrong", "no-return"),
    ("pixels", _with(top="(cy - h / 2) * 400", left="(cx - w / 2) * 600", bottom="(cy + h / 2) * 400", right="(cx + w / 2) * 600"),
     "almost", None),
    ("pixels", "", "empty", None),
    # whole numbers that are not the course's answer rounded: a plain wrong answer, or the mistake it rounds
    ("pixels", _HEAD + "    return (0, 0, height, width)\n", "wrong", None),
    ("pixels", _with("int(cy - h / 2)", "int(cx - w / 2)", "int(cy + h / 2)", "int(cx + w / 2)"), "wrong", None),
    ("pixels", _with("round((cy - h / 2) * height)", "round((cx - w / 2) * width)", "round((cy + h / 2) * height)",
                     "round((cx + w / 2) * width)", order="left, top, right, bottom"), "wrong", None),
    ("pixels", _with("round(cy * height)", "round(cx * width)", "round((cy + h) * height)", "round((cx + w) * width)"),
     "wrong", None),
    # computed in float32: right all the same
    ("pixels", "import numpy as np\n" + _SOLUTION.replace("cx, cy, w, h = box", "cx, cy, w, h = np.float32(box)")
     .replace("    return (top, left, bottom, right)", "    return (np.float32(top), np.float32(left), np.float32(bottom), np.float32(right))"),
     "correct", None),
]
