# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-sam: choose, which picks the mask of SAM 2.1 that a prompt asks for, and next_click, which
says where to click next. Both are checked on masks the course gives, so they work without SAM. SAM 2.1 tiny (156 MB,
from Hugging Face) and Caltech-101 (137 MB, from CaltechDATA) the lesson downloads once, when the learner asks for
them (everything_ai.hub, everything_ai.caltech): the worked example and the demo need them, the exercises do not. The
picture, the slider and the checks use what the course measured with them: the numbers in REFERENCE, and SAM's masks
on the coffee photo in coffee-masks.png, next to this file (MASKS says which bit of a pixel is which mask)."""
import copy
import errno
import functools
import html
import re
import urllib.error

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import caltech, chart, course, hub, paths, sam2, yours
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit
from everything_ai.lesson import download_card

DOWNLOAD_LINE = {
    "en": "Works on every computer. SAM 2.1, 156 MB, is downloaded once from Hugging Face, and Caltech-101, 137 MB of photos, from Caltech, when you click their button in the lesson; after that, the lesson works without a connection. The exercises work without them too.",
    "nl": "Werkt op elke computer. SAM 2.1, 156 MB, wordt één keer gedownload van Hugging Face, en Caltech-101, 137 MB aan foto's, van Caltech, als je in de les op de knop ervoor klikt; daarna werkt de les zonder verbinding. De oefeningen werken ook zonder.",
}

PHOTO = "coffee"                   # the course's own photo of the picture, the worked example and the checks
SPOON = (360, 277)                 # a click on the bowl of the spoon, (x, y) in pixels of the coffee photo
HANDLE = (217, 262)                # a click on the handle of the cup
CLASSES = list(caltech.CHECKED)    # the 10 classes of Caltech-101 the lesson uses, in alphabetical order
NAMES = {
    "en": {"chandelier": "chandelier", "dragonfly": "dragonfly", "ewer": "ewer", "lotus": "lotus", "nautilus": "nautilus",
           "rooster": "rooster", "starfish": "starfish", "sunflower": "sunflower", "water_lilly": "water lily",
           "windsor_chair": "Windsor chair"},
    "nl": {"chandelier": "kroonluchter", "dragonfly": "libel", "ewer": "schenkkan", "lotus": "lotus", "nautilus": "nautilus",
           "rooster": "haan", "starfish": "zeester", "sunflower": "zonnebloem", "water_lilly": "waterlelie",
           "windsor_chair": "windsorstoel"},
}
COUNT = 30                         # the photos of each class, as in the lessons before
CLICKS = 5                         # the clicks the course gives each photo
WHOLE = 0.8                        # the IoU at which a mask counts as the whole thing, in the guess
DEMO = [("sunflower", 0), ("nautilus", 0), ("starfish", 4), ("dragonfly", 5), ("chandelier", 1), ("windsor_chair", 2)]
# Which bit of a pixel of coffee-masks.png holds which of SAM's masks on the coffee photo: the four masks of a click
# on the spoon and of a click on the handle of the cup, and mask 0 after the second click that next_click gives.
MASKS = {**{f"spoon{k}": k for k in range(4)}, **{f"handle{k}": 4 + k for k in range(4)}, "spoon-two": 8}

# The numbers in the text. `measure(model)` measures them again with SAM 2.1 and Caltech-101; the test of the lesson
# checks that both are the same.
REFERENCE = {
    "spoon": {"scores": (0.901, 0.645, 0.059, 0.941), "areas": (5516, 6762, 228, 5575)},
    "handle": {"scores": (0.919, 0.122, 0.719, 0.894), "areas": (3550, 20247, 2590, 3894)},
    "second": {"click": (411, 83, 1), "missed": 1257, "extra": 70, "before": 0.806, "after": 0.941, "score": 0.917},
    "classes": {
        "chandelier": {"clicks": (0.078, 0.579, 0.608, 0.664, 0.678), "best": 0.616, "box": 0.761},
        "dragonfly": {"clicks": (0.643, 0.861, 0.892, 0.896, 0.904), "best": 0.857, "box": 0.912},
        "ewer": {"clicks": (0.82, 0.894, 0.914, 0.854, 0.924), "best": 0.837, "box": 0.856},
        "lotus": {"clicks": (0.821, 0.891, 0.909, 0.92, 0.912), "best": 0.899, "box": 0.932},
        "nautilus": {"clicks": (0.535, 0.872, 0.94, 0.963, 0.957), "best": 0.912, "box": 0.947},
        "rooster": {"clicks": (0.832, 0.883, 0.898, 0.914, 0.922), "best": 0.879, "box": 0.919},
        "starfish": {"clicks": (0.919, 0.929, 0.921, 0.918, 0.914), "best": 0.919, "box": 0.916},
        "sunflower": {"clicks": (0.18, 0.913, 0.907, 0.921, 0.921), "best": 0.926, "box": 0.935},
        "water_lilly": {"clicks": (0.811, 0.913, 0.921, 0.927, 0.917), "best": 0.916, "box": 0.927},
        "windsor_chair": {"clicks": (0.419, 0.78, 0.788, 0.832, 0.754), "best": 0.726, "box": 0.678},
    },
    "sunflower": {"whole": 1, "among": 25},
    "demo": {"sunflower/0": (0.042, 0.956, 0.956, 0.965, 0.963), "nautilus/0": (0.044, 0.966, 0.944, 0.976, 0.974), "starfish/4": (0.943, 0.947, 0.95, 0.95, 0.945), "dragonfly/5": (0.167, 0.904, 0.908, 0.91, 0.892), "chandelier/1": (0.158, 0.432, 0.759, 0.744, 0.787), "windsor_chair/2": (0.123, 0.772, 0.916, 0.833, 0.732)},
}


# --- what SAM does, as the lesson works it out --------------------------------------------------------------------------

def photo(name: str = PHOTO):
    """One of the course's own photos, opened with Pillow, in RGB."""
    from PIL import Image
    return Image.open(paths.ROOT / "data" / "photos" / f"{name}.png").convert("RGB")


def points(clicks, width: int, height: int) -> torch.Tensor:
    """Clicks (x, y) in pixels of a photo as points on the photo of 1024 by 1024 that SAM sees, as a batch of one
    prompt: (1, k, 2)."""
    return torch.tensor([[[x * sam2.SIZE / width, y * sam2.SIZE / height] for x, y in clicks]], dtype=torch.float32)


def masks_of(model, features, clicks, labels, width: int, height: int):
    """SAM's four masks for a prompt, as large as the photo, (4, height, width) logits, and their four scores."""
    with torch.no_grad():
        logits, scores = model.decode(features, points(clicks, width, height), torch.tensor([list(labels)]))
    return sam2.to_photo(logits[0], width, height), scores[0].numpy()


def choose(logits, scores, clicks: int):
    """The mask for a prompt: with one click, the mask with the highest score among masks 1 to 3; with more clicks,
    mask 0. A pixel belongs to it where its logit is above 0."""
    if clicks == 1:
        k = 1 + int(np.argmax(scores[1:]))
    else:
        k = 0
    return logits[k] > 0


def next_click(mask, truth):
    """Where to click next: in the deepest pixel of what is most wrong, the pixels of the truth the mask missed or
    the pixels of the mask that are not in the truth, whichever are more, as (x, y, label): label 1 for missed
    pixels, 0 for extra ones. None when the mask is the truth."""
    missed = truth & ~mask
    extra = mask & ~truth
    if not missed.any() and not extra.any():
        return None
    if missed.sum() >= extra.sum():
        x, y = sam2.deepest(missed)
        return (x, y, 1)
    x, y = sam2.deepest(extra)
    return (x, y, 0)


def iou(a, b) -> float:
    """The IoU of two masks: the pixels in both, divided by the pixels in either."""
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def truth_of(outline, width: int, height: int) -> np.ndarray:
    """The mask inside an outline of Caltech-101, points [x, y] in pixels, as large as its photo."""
    from PIL import Image, ImageDraw
    img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(img).polygon([tuple(p) for p in outline], fill=1, outline=1)
    return np.array(img).astype(bool)


def clicking(model, features, truth, clicks: int = CLICKS, choose_fn=choose, next_fn=next_click) -> list[dict]:
    """A person who clicks on a photo, as the course plays one: the first click in the deepest pixel of what they
    mean, then each next click where next_fn says, `clicks` in all, each time with all clicks so far. Gives every
    step: the clicks, their labels, the mask choose_fn picks, and its IoU with the truth. When the mask is right, the
    person stops clicking."""
    height, width = truth.shape
    x, y = sam2.deepest(truth)
    at, labels, steps = [(x, y)], [1], []
    while True:
        logits, scores = masks_of(model, features, at, labels, width, height)
        mask = np.asarray(choose_fn(logits, scores, len(at)), dtype=bool)
        steps.append({"clicks": list(at), "labels": list(labels), "mask": mask, "iou": iou(mask, truth)})
        if len(at) == clicks:
            return steps
        nxt = next_fn(mask, truth)
        if nxt is None:
            return steps
        at.append((int(nxt[0]), int(nxt[1])))
        labels.append(int(nxt[2]))


def photos(name: str) -> list:
    """The 30 photos of a class the lesson uses, without people: everything_ai.caltech.checked."""
    return caltech.checked(name, COUNT)


def _median(values) -> float:
    return round(float(np.median(values)), 3)


def measure_coffee(model) -> tuple[dict, dict]:
    """The numbers of REFERENCE about the coffee photo, and the masks of coffee-masks.png, by their names in MASKS."""
    out, masks = {}, {}
    img = photo()
    w, h = img.size
    with torch.no_grad():
        features = model.features(sam2.prepare(img))
    for name, click in (("spoon", SPOON), ("handle", HANDLE)):
        logits, scores = masks_of(model, features, [click], [1], w, h)
        for k in range(4):
            masks[f"{name}{k}"] = logits[k] > 0
        out[name] = {"scores": tuple(round(float(s), 3) for s in scores),
                     "areas": tuple(int((logits[k] > 0).sum()) for k in range(4))}
    # the whole spoon is what you meant; SAM chose the bowl: where next_click clicks, and what SAM does with it
    bowl, spoon = choose(*masks_of(model, features, [SPOON], [1], w, h), 1), masks["spoon1"]
    x, y, label = next_click(bowl, spoon)
    logits, scores = masks_of(model, features, [SPOON, (x, y)], [1, label], w, h)
    masks["spoon-two"] = logits[0] > 0
    out["second"] = {"click": (x, y, label), "missed": int((spoon & ~bowl).sum()), "extra": int((bowl & ~spoon).sum()),
                     "before": round(iou(bowl, spoon), 3), "after": round(iou(masks["spoon-two"], spoon), 3),
                     "score": round(float(scores[0]), 3)}
    return out, masks


def measure(model) -> tuple[dict, dict]:
    """Every number of REFERENCE, measured with SAM on the coffee photo and on 30 photos each of the 10 classes of
    Caltech-101, and the masks of coffee-masks.png, by their names in MASKS."""
    from PIL import Image
    out, masks = measure_coffee(model)
    rows = {}
    for name in CLASSES:
        outlines = caltech.outlines(name)
        rows[name] = []
        for path in photos(name):
            img = Image.open(path).convert("RGB")
            w, h = img.size
            truth = truth_of(outlines[path.name], w, h)
            with torch.no_grad():
                features = model.features(sam2.prepare(img))
            steps = clicking(model, features, truth)
            ious = [s["iou"] for s in steps] + [steps[-1]["iou"]] * (CLICKS - len(steps))
            logits, scores = masks_of(model, features, steps[0]["clicks"], [1], w, h)
            ys, xs = np.nonzero(truth)
            box, _ = masks_of(model, features, [(xs.min(), ys.min()), (xs.max(), ys.max())], [sam2.TOP_LEFT, sam2.BOTTOM_RIGHT], w, h)
            rows[name].append({"ious": ious, "best": max(iou(logits[k] > 0, truth) for k in (1, 2, 3)),
                               "box": iou(box[0] > 0, truth)})
    out["classes"] = {name: {"clicks": tuple(_median([r["ious"][k] for r in rows[name]]) for k in range(CLICKS)),
                             "best": _median([r["best"] for r in rows[name]]),
                             "box": _median([r["box"] for r in rows[name]])} for name in CLASSES}
    sun = rows["sunflower"]
    out["sunflower"] = {"whole": sum(r["ious"][0] >= WHOLE for r in sun), "among": sum(r["best"] >= WHOLE for r in sun)}
    out["demo"] = {f"{c}/{i}": tuple(round(v, 3) for v in rows[c][i]["ious"]) for c, i in DEMO}
    return out, masks


def save_masks(masks: dict, path=None) -> None:
    """The masks of measure as coffee-masks.png: bit MASKS[name] of each pixel says whether it is in that mask."""
    from PIL import Image
    bits = sum(masks[name].astype(np.uint32) << MASKS[name] for name in MASKS)
    rgb = np.stack([(bits >> s) & 255 for s in (0, 8, 16)], axis=-1).astype(np.uint8)
    Image.fromarray(rgb, "RGB").save(path or course.lesson_dir("cv-sam") / "coffee-masks.png", optimize=True)


@functools.cache
def _bits() -> np.ndarray:
    from PIL import Image
    rgb = np.asarray(Image.open(course.lesson_dir("cv-sam") / "coffee-masks.png").convert("RGB")).astype(np.uint32)
    return rgb[..., 0] | rgb[..., 1] << 8 | rgb[..., 2] << 16


def mask(name: str) -> np.ndarray:
    """One of SAM's masks on the coffee photo, as the course measured it: an array of True and False, 400 by 600."""
    return (_bits() >> MASKS[name]) & 1 == 1


def as_logits(masks) -> np.ndarray:
    """Masks as logits, for the checks: 4 inside a mask, -4 outside."""
    return np.where(np.asarray(masks), 4.0, -4.0).astype(np.float32)


# --- SAM and Caltech-101 in the lesson ----------------------------------------------------------------------------------

def ready() -> bool:
    """SAM, the unpacked photos of Caltech-101, and their outlines or the zip file to read them from."""
    return hub.pinned(*sam2.FILE) is not None and caltech.unpacked() and caltech.outlines_ready()


@functools.lru_cache(maxsize=1)
def _sam():
    return sam2.load()


def sam_or_stop(button, texts: dict):
    """SAM for the lesson's cells, with Caltech-101 and its outlines unpacked. When either is not there yet, the cell
    stops with `texts["ask"]` and the button; a click downloads what is missing and unpacks the photos, with the
    progress in the cell. While it downloads, the lesson's other cells wait. `texts` has "ask", "busy" ({name},
    {done} and {total} in MB), "unpacking" ({done} photos), "no_net", "not_pinned", "no_room" and "failed" ({why}):
    why it did not work; and "here"."""
    import marimo as mo
    failed = None
    if not ready() and button.value:
        try:
            for name, pin in (("SAM 2.1", sam2.FILE), ("Caltech-101", caltech.ZIP)):
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
            caltech.unpack_outlines()
        except hub.NotPinned:
            failed = texts["not_pinned"]
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            failed = texts["no_net"]
        except OSError as e:
            failed = texts["no_room"] if e.errno == errno.ENOSPC else texts["failed"].format(why=html.escape(str(e)))
    if not ready():
        mo.stop(True, download_card(texts["ask"], button, failed))
    try:
        caltech.unpack_outlines()   # once: from the zip file that is there, also without a click
    except OSError as e:
        why = texts["no_room"] if e.errno == errno.ENOSPC else texts["failed"].format(why=html.escape(str(e)))
        mo.stop(True, download_card(texts["ask"], button, why))
    model = _sam()
    mo.output.replace(mo.md(texts["here"]))
    return model


@functools.lru_cache(maxsize=1)
def _demo_photos(model) -> tuple:
    """The photos of the demo, computed once: for each, the photo, the mask of its outline and SAM's features."""
    from PIL import Image
    out = []
    for name, i in DEMO:
        path = photos(name)[i]
        img = Image.open(path).convert("RGB")
        truth = truth_of(caltech.outlines(name)[path.name], *img.size)
        with torch.no_grad():
            features = model.features(sam2.prepare(img))
        out.append((img, truth, features))
    return tuple(out)


@functools.lru_cache(maxsize=4)
def _demo_steps(model, choose_text: str | None, next_text: str | None) -> tuple:
    """Every step of the demo, with the learner's functions where they are given and work, else the course's: the
    steps of each photo, and whose functions they were. If a learner's function fails here after all, the course's
    takes over, for every photo."""
    def made(text, name):
        ran = run_code(text) if text else None
        fn = ran.names.get(name) if ran is not None and ran.error is None else None
        return fn if callable(fn) else None
    mine = (made(choose_text, "choose"), made(next_text, "next_click"))
    fns = (mine[0] or choose, mine[1] or next_click)
    shot = _demo_photos(model)
    if mine != (None, None):
        try:
            return tuple(with_time_limit(lambda: clicking(model, f, t, choose_fn=fns[0], next_fn=fns[1]), 30)
                         for _, t, f in shot), mine[0] is not None, mine[1] is not None
        except Exception:
            pass
    return tuple(clicking(model, f, t) for _, t, f in shot), False, False


# --- the pictures -------------------------------------------------------------------------------------------------------

def number(v: float, lang: str, digits: int = 2) -> str:
    s = f"{v:.{digits}f}"
    return s.replace(".", ",") if lang == "nl" else s


def _faded(img) -> np.ndarray:
    """A photo in light gray, so that masks in the color of a prediction stand out on every photo."""
    gray = np.asarray(img, dtype=float).mean(axis=2, keepdims=True)
    return np.repeat(255 - (255 - gray) * 0.6, 3, axis=2).astype(np.uint8)


def _crop(*masks, margin: int = 24, shape=None) -> tuple[int, int, int, int]:
    """The part of a photo around masks: (top, left, bottom, right), with a margin."""
    ys, xs = np.nonzero(np.any(np.stack(masks), axis=0))
    h, w = shape or masks[0].shape
    return (max(int(ys.min()) - margin, 0), max(int(xs.min()) - margin, 0), min(int(ys.max()) + margin + 1, h),
            min(int(xs.max()) + margin + 1, w))


def _axes(ax, img, title: str, crop=None) -> None:
    """A photo, faded, on its axes, from pixel edge to pixel edge, cut to `crop`."""
    h, w = np.asarray(img).shape[:2]
    ax.imshow(_faded(img), interpolation="antialiased", extent=(-0.5, w - 0.5, h - 0.5, -0.5))
    top, left, bottom, right = crop or (0, 0, h, w)
    ax.set_xlim(left - 0.5, right - 0.5)
    ax.set_ylim(bottom - 0.5, top - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def _mask(ax, m, dashed: bool = False) -> None:
    """A mask of SAM: filled in the color of a prediction, with a line around it; or only a dashed line."""
    from everything_ai import plots
    m = np.asarray(m, dtype=float)
    if not m.any():
        return
    if dashed:
        ax.contour(m, levels=[0.5], colors=[plots.PAGE], linewidths=3.0)
        ax.contour(m, levels=[0.5], colors=[plots.PRED], linewidths=1.5, linestyles=[(0, (3, 1.6))])
    else:
        ax.contourf(m, levels=[0.5, 1.5], colors=[plots.PRED], alpha=0.5)
        ax.contour(m, levels=[0.5], colors=[plots.PRED], linewidths=1.2)


def _truth(ax, m) -> None:
    """The mask of what you meant, as its edge in the color of data, with a thin edge in the color of the page."""
    from everything_ai import plots
    m = np.asarray(m, dtype=float)
    if m.any():
        ax.contour(m, levels=[0.5], colors=[plots.PAGE], linewidths=2.6)
        ax.contour(m, levels=[0.5], colors=[plots.DATA], linewidths=1.1)


def _wrong(ax, m) -> None:
    """Pixels the mask gets wrong, filled in the color of mistakes."""
    from everything_ai import plots
    m = np.asarray(m, dtype=float)
    if m.any():
        ax.contourf(m, levels=[0.5, 1.5], colors=[plots.LOSS], alpha=0.6)


def _clicks(ax, at, labels) -> None:
    """Clicks: filled for a click on the thing (label 1), open for a click on what is not (label 0)."""
    from everything_ai import plots
    for (x, y), label in zip(at, labels):
        ax.plot([x], [y], "o", markersize=7, markeredgewidth=1.6, markeredgecolor=plots.PAGE if label else plots.INK,
                markerfacecolor=plots.INK if label else plots.PAGE, zorder=6)


def key_html(lang: str, entries: tuple = ("mask", "truth", "click", "negative")) -> str:
    """The key under the pictures of masks, in the lesson's words."""
    words = {
        "en": {"mask": "SAM's mask", "truth": "the outline of what you meant", "wrong": "pixels the mask gets wrong",
               "click": "a click on the thing, label 1", "negative": "a click on what is not part of it, label 0"},
        "nl": {"mask": "het mask van SAM", "truth": "de omtrek van wat je bedoelt", "wrong": "pixels die het mask fout heeft",
               "click": "een klik op het ding, label 1", "negative": "een klik op wat er niet bij hoort, label 0"},
    }[lang]
    signs = {
        "mask": '<span style="display:inline-block;width:14px;height:10px;background:var(--ea-pred);opacity:0.6;border:1px solid var(--ea-pred)"></span>',
        "truth": '<span style="display:inline-block;width:16px;height:0;border-top:2px solid var(--ea-data)"></span>',
        "wrong": '<span style="display:inline-block;width:14px;height:10px;background:var(--ea-loss);opacity:0.7"></span>',
        "click": '<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--ea-ink)"></span>',
        "negative": '<span style="display:inline-block;width:7px;height:7px;border-radius:50%;border:2px solid var(--ea-ink)"></span>',
    }
    items = "".join(f'<span style="display:inline-flex;align-items:center;gap:6px">{signs[e]}{html.escape(words[e])}</span>'
                    for e in entries)
    return (f'<div style="display:flex;flex-wrap:wrap;gap:4px 18px;font-family:var(--ea-font-ui);font-size:13px;'
            f'color:var(--ea-ink-2);margin:0.2rem 0 0.6rem">{items}</div>')


def _render(fig, alt: str) -> str:
    import matplotlib.pyplot as plt
    from everything_ai import plots
    out = plots.render(fig, alt=alt)
    plt.close(fig)
    return out


def spoon_picture(words: dict, lang: str) -> str:
    """The three masks of one click on the bowl of the spoon, each with its score, from coffee-masks.png and
    REFERENCE. `words` has "title" ({k} and {score}) and "alt"."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    plots.style()
    img = photo()
    crop = _crop(mask("spoon1"), mask("spoon3"), margin=40)
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 3.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.05)
    for ax, k in zip(axes, (1, 2, 3)):
        _axes(ax, img, words["title"].format(k=k, score=number(REFERENCE["spoon"]["scores"][k], lang, 3)), crop)
        _mask(ax, mask(f"spoon{k}"))
        _clicks(ax, [SPOON], [1])
    return _render(fig, words["alt"]) + key_html(lang, ("mask", "click"))


def clicks_frame(k: int, labels: dict, lang: str) -> str:
    """For the slider: per class the median IoU after k clicks, as a dot, with a line back to the median after one
    click, an open dot; the median of a box prompt as a short upright line. `labels` has "x" and "alt" ({k}); the
    names of the classes come from NAMES."""
    data = REFERENCE["classes"]
    rows = list(reversed(CLASSES))   # the first class at the top
    panel = chart.Panel((0, 1), (-0.6, len(rows) - 0.2), xlabel=labels["x"], xticks=[0, 0.25, 0.5, 0.75, 1], yticks=[])
    for y, name in enumerate(rows):
        one, now, box = data[name]["clicks"][0], data[name]["clicks"][k - 1], data[name]["box"]
        panel.text(0.01, y + 0.42, NAMES[lang][name], role="ink-2", anchor="start")
        panel.line([box, box], [y - 0.22, y + 0.22], role="ink-3", width=2.0)
        panel.line([one, now], [y, y], role="ink-3", width=1.6)
        panel.points([one], [y], role="pred", r=4, hollow=True)
        panel.points([now], [y], role="pred", r=5)
    return chart.figure([panel], alt=labels["alt"].format(k=k), width=440, height=420, drag=None)


TASK = 0.85   # the task of the slider: the fewest clicks after which most kinds reach this median IoU


def reached(k: int) -> int:
    """How many of the 10 kinds have a median IoU of TASK or more after k clicks."""
    return sum(REFERENCE["classes"][name]["clicks"][k - 1] >= TASK for name in CLASSES)


def clicks_line(k: int, lang: str) -> str:
    """The sentence under the slider's chart: how many kinds reach TASK after k clicks, and three of them."""
    data = REFERENCE["classes"]
    a, b, c = (number(data[name]["clicks"][k - 1], lang) for name in ("sunflower", "chandelier", "windsor_chair"))
    if lang == "nl":
        return (f"Na {k} {'klik' if k == 1 else 'klikken'}: een mediane IoU van {number(TASK, lang)} of meer bij "
                f"{reached(k)} van de 10 soorten. Zonnebloemen {a}, kroonluchters {b}, windsorstoelen {c}.")
    return (f"After {k} {'click' if k == 1 else 'clicks'}: a median IoU of {number(TASK, lang)} or more for {reached(k)} "
            f"of the 10 kinds. Sunflowers {a}, chandeliers {b}, Windsor chairs {c}.")


def demo(model, choose_text, next_text, k: int, labels: dict, lang: str) -> dict:
    """The photos of the demo after k clicks: the outline of what you meant, SAM's mask and the clicks, chosen and
    placed with the learner's choose and next_click once they are right, else with the course's. Gives the picture,
    whose functions they were, the IoU of each photo, those as a line in the lesson's language ("listed": with a
    semicolon between them in Dutch, where a comma is the decimal point), and "clicks", such as "1 click". `labels` has "titles" ({iou}) and "alt". The
    cell passes the model it got from sam_or_stop, so that it runs only once the downloads are there."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    plots.style()
    steps, mine_choose, mine_next = _demo_steps(model, choose_text, next_text)
    shot = _demo_photos(model)
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.9))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.05, hspace=0.16)
    ious = []
    for ax, (img, truth, _), title, run in zip(axes.flat, shot, labels["titles"], steps):
        step = run[min(k, len(run)) - 1]
        ious.append(step["iou"])
        _axes(ax, img, title.format(iou=number(step["iou"], lang)))
        _mask(ax, step["mask"])
        _truth(ax, truth)
        _clicks(ax, step["clicks"], step["labels"])
    listed = ("; " if lang == "nl" else ", ").join(number(v, lang) for v in ious)
    clicks = {"en": f"{k} click" if k == 1 else f"{k} clicks", "nl": f"{k} klik" if k == 1 else f"{k} klikken"}[lang]
    return {"picture": _render(fig, labels["alt"]) + key_html(lang), "choose": mine_choose, "next": mine_next,
            "ious": ious, "listed": listed, "clicks": clicks}


# The lesson is done when these are correct. "seeds" is a choice question, "count" the open one.
REQUIRED = ["choose", "seeds", "next", "name"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace each `...` with your own code.",
                nl="Er staat nog een `...` in je code. Vervang elke `...` door je eigen code.")


class _Named:
    """An argument of a case, written in the check's messages by its name, such as `logits`; the function gets a copy
    of its value."""

    def __init__(self, shown: str, value):
        self.shown, self.value = shown, value

    def __repr__(self):
        return self.shown

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


def _array(got):
    """What a function gave back as a NumPy array, or None."""
    if isinstance(got, torch.Tensor):
        got = got.detach().cpu().numpy()
    try:
        return np.asarray(got)
    except Exception:
        return None


# --- Exercise 1: choose -------------------------------------------------------------------------------------------------

_C_HEAD = ("import numpy as np\n\n\n"
           "def choose(logits, scores, clicks):\n"
           "    # logits: SAM's four masks of logits; scores: their four scores; clicks: how many clicks the prompt has\n")


def _choose_with(one: str = "1 + np.argmax(scores[1:])", more: str = "0", back: str = "logits[k] > 0") -> str:
    """The course's code, with other lines for the mask of one click, of more clicks, or what it gives back."""
    return _C_HEAD + f"    if clicks == 1:\n        k = {one}\n    else:\n        k = {more}\n    return {back}\n"


_C_START = _choose_with("...", "...", "...")
_C_SOLUTION = _choose_with()


def _blob(h: int, w: int, cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
    """Logits of an ellipse: above 0 inside, below 0 outside, rising by about 1 per pixel toward its middle."""
    y, x = np.mgrid[0:h, 0:w]
    d = np.sqrt(((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2)
    return ((1 - d) * min(ry, rx)).astype(np.float32)


def _made_up(h: int = 36, w: int = 52) -> np.ndarray:
    """Four masks of logits the course makes up: a part, a larger part and the whole of a thing, and mask 0."""
    return np.stack([_blob(h, w, 18, 26, 9.5, 12.5), _blob(h, w, 18, 26, 4, 5), _blob(h, w, 18, 26, 9, 12),
                     _blob(h, w, 18, 26, 15, 22)])


def _choose_cases():
    """The click on the spoon of the worked example; then the click on the handle of the cup, whose mask 0 has the
    highest score of all four; the click on the spoon with a second click; and logits the course makes up, whose
    edges lie between 0 and 0.5, with one click and with three."""
    spoon = as_logits([mask(f"spoon{k}") for k in range(4)])
    handle = as_logits([mask(f"handle{k}") for k in range(4)])
    s, hs = REFERENCE["spoon"]["scores"], REFERENCE["handle"]["scores"]
    made = _made_up()
    return ([(_Named("logits", spoon), _Named(f"np.array({list(s)})", np.array(s)), 1)],
            [(_Named("handle_logits", handle), _Named(f"np.array({list(hs)})", np.array(hs)), 1),
             (_Named("logits", spoon), _Named(f"np.array({list(s)})", np.array(s)), 2),
             (_Named("made_up", made), _Named("np.array([0.95, 0.3, 0.85, 0.6])", np.array([0.95, 0.3, 0.85, 0.6])), 1),
             (_Named("made_up", made), _Named("np.array([0.2, 0.9, 0.5, 0.4])", np.array([0.2, 0.9, 0.5, 0.4])), 1),
             (_Named("made_up", made), _Named("np.array([0.4, 0.3, 0.9, 0.8])", np.array([0.4, 0.3, 0.9, 0.8])), 3)])


def _mask_text(v) -> dict | str:
    """What choose gave back, in words: a mask by its number of pixels; numbers that are not True and False; a number."""
    a = _array(v)
    if a is not None and a.dtype == bool and a.ndim == 2:
        n = int(a.sum())
        return {"en": f"a mask of {n:,} pixels", "nl": f"een mask van {n:,} pixels".replace(",", ".")}
    if a is not None and a.dtype == bool and a.ndim == 3:
        sizes = [f"{int(m.sum()):,}" for m in a]
        return {"en": f"{len(a)} masks, of {', '.join(sizes)} pixels",
                "nl": f"{len(a)} masks, van {'; '.join(s.replace(',', '.') for s in sizes)} pixels"}
    if a is not None and a.ndim == 2 and a.dtype.kind in "iuf":
        return {"en": f"an array of {a.dtype} numbers, from {a.min():.2f} to {a.max():.2f}",
                "nl": f"een array met getallen van het type {a.dtype}, van {a.min():.2f} tot {a.max():.2f}".replace(".", ",")}
    return str(v)


def _same_mask(got, want) -> bool:
    a = _array(got)
    return a is not None and a.dtype == bool and a.shape == np.shape(want) and bool((a == want).all())


def _choose_picture(seen) -> str:
    """The mask the learner's choose picks for one click on the spoon, on the coffee photo, beside the course's
    where the learner's is another."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    mine = seen.mine
    fn = mine.names.get("choose") if mine else None
    if not callable(fn):
        return ""
    case = EXERCISES["choose"].visible[0]
    try:
        got = _array(with_time_limit(lambda: fn(*(copy.deepcopy(a) for a in case))))
    except Exception:
        return ""
    right = choose(case[0].value, case[1].value, 1)
    if got is None or got.shape != right.shape:
        return ""
    got = got > 0 if got.dtype != bool else got
    lang = seen.lang
    panels = [(course.phrase("yours-code", lang), got)]
    if not (got == right).all():
        panels.append((course.phrase("yours-right", lang), right))
    plots.style()
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.0), squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.05)
    crop = _crop(mask("spoon1"), mask("spoon3"), margin=40)
    word = {"en": "{who}: {n} pixels", "nl": "{who}: {n} pixels"}[lang]
    for ax, (who, m) in zip(axes[0], panels):
        _axes(ax, photo(), word.format(who=who, n=f"{int(m.sum()):,}".replace(",", "." if lang == "nl" else ",")), crop)
        _mask(ax, m)
        _clicks(ax, [SPOON], [1])
    title = {"en": "The mask your choose picks for one click on the spoon",
             "nl": "Het mask dat jouw choose kiest bij één klik op de lepel"}[lang]
    alt = "; ".join(word.format(who=who, n=int(m.sum())) for who, m in panels)
    return yours.figure(title, _render(fig, alt) + key_html(lang, ("mask", "click")), alt=alt)


def _cm(key, en, nl, **kw):
    return Mistake(key, en="`{call}` gives `{got}`, and it should give `{want}`. " + en,
                   nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. " + nl, **kw)


def _choose_exercise():
    visible, hidden = _choose_cases()
    return FunctionBox(
        name="choose",
        picture=_choose_picture,
        function="choose",
        start=_C_START,
        solution=_C_SOLUTION,
        visible=visible,
        hidden=hidden,
        show=_mask_text,
        compare=_same_mask,
        first=[_DOTS],
        mistakes=[
            _cm("all-four", "With one click, mask 0 takes no part: it is the answer to a prompt that says exactly what it means. Take the highest score among masks 1 to 3, `scores[1:]`.",
                "Bij één klik doet mask 0 niet mee: dat is het antwoord op een prompt die precies zegt wat hij bedoelt. Neem de hoogste score van masks 1 tot en met 3, `scores[1:]`.",
                wrong=_choose_with(one="np.argmax(scores)")),
            _cm("shifted", "`np.argmax(scores[1:])` counts from the start of `scores[1:]`: its 0 is mask 1. Add 1 to get the number of the mask in `logits`.",
                "`np.argmax(scores[1:])` telt vanaf het begin van `scores[1:]`: zijn 0 is mask 1. Tel er 1 bij op, dan heb je het nummer van het mask in `logits`.",
                wrong=_choose_with(one="np.argmax(scores[1:])")),
            _cm("lowest", "That is the mask with the lowest score. The score is the IoU SAM expects the mask to have with what you meant: the higher, the better.",
                "Dat is het mask met de laagste score. De score is de IoU die SAM verwacht tussen het mask en wat je bedoelt: hoe hoger, hoe beter.",
                wrong=_choose_with(one="1 + np.argmin(scores[1:])")),
            _cm("largest", "That is the largest of masks 1 to 3. One click can mean a part or the whole, and the largest mask is not always what SAM expects to fit best: take the highest score.",
                "Dat is het grootste van masks 1 tot en met 3. Eén klik kan een deel of het geheel bedoelen, en het grootste mask is niet altijd wat volgens SAM het best past: neem de hoogste score.",
                wrong=_choose_with(one="1 + np.argmax((logits[1:] > 0).sum(axis=(1, 2)))")),
            _cm("always-zero", "With one click, the prompt can mean a part or the whole, and SAM gives three masks for it: masks 1 to 3. Mask 0 is for a prompt of more clicks.",
                "Bij één klik kan de prompt een deel of het geheel bedoelen, en daar geeft SAM drie masks voor: masks 1 tot en met 3. Mask 0 is voor een prompt van meer klikken.",
                wrong=_choose_with(one="0")),
            _cm("swapped", "The two cases are swapped: with one click, choose among masks 1 to 3 by their score; with more clicks, the prompt says what it means, and mask 0 is the answer.",
                "De twee gevallen zijn omgewisseld: bij één klik kies je uit masks 1 tot en met 3 op hun score; bij meer klikken zegt de prompt wat hij bedoelt, en is mask 0 het antwoord.",
                wrong=_choose_with(one="0", more="1 + np.argmax(scores[1:])")),
            _cm("always-best", "With more than one click, the prompt says what it means: take mask 0, whatever the scores of the others.",
                "Bij meer dan één klik zegt de prompt wat hij bedoelt: neem mask 0, wat de scores van de andere ook zijn.",
                wrong=_choose_with(more="1 + np.argmax(scores[1:])")),
            _cm("probability", "A logit is not a probability: a pixel belongs to the mask where its logit is above 0, where the probability, after sigmoid, is above 0.5.",
                "Een logit is geen kans: een pixel hoort bij het mask waar zijn logit boven 0 ligt, waar de kans, na sigmoid, boven 0,5 ligt.",
                wrong=_choose_with(back="logits[k] > 0.5")),
            _cm("logits", "Those are the logits of the mask, numbers. The mask itself is True where a logit is above 0, and False elsewhere: `logits[k] > 0`.",
                "Dat zijn de logits van het mask, getallen. Het mask zelf is True waar een logit boven 0 ligt, en elders False: `logits[k] > 0`.",
                matches=lambda got, text: (a := _array(got)) is not None and a.ndim == 2 and a.dtype.kind == "f"),
            _cm("all-masks", "Those are all four masks at once. Choose one of them: `logits[k] > 0`, with `k` the number of the mask.",
                "Dat zijn alle vier de masks tegelijk. Kies er een: `logits[k] > 0`, met `k` het nummer van het mask.",
                matches=lambda got, text: (a := _array(got)) is not None and a.ndim == 3),
            _cm("index", "That is the number of the mask. Give back the mask itself: True where its logit is above 0, `logits[k] > 0`.",
                "Dat is het nummer van het mask. Geef het mask zelf terug: True waar zijn logit boven 0 ligt, `logits[k] > 0`.",
                matches=lambda got, text: isinstance(got, (int, np.integer)) and not isinstance(got, bool)),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace each `...`: the number `k` of the mask with one click and with more, and what the function gives back, the mask: True where its logit is above 0.",
                      "nl": "Vervang elke `...`: het nummer `k` van het mask bij één klik en bij meer, en wat de functie teruggeeft, het mask: True waar zijn logit boven 0 ligt."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the logits, the scores and the number of clicks, and the `def` line of your function has a different number of names between its parentheses. Keep `(logits, scores, clicks)`.",
                          "nl": "De check roept `{call}` aan, met de logits, de scores en het aantal klikken, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(logits, scores, clicks)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. With one click: the mask with the highest score among masks 1 to 3; with more clicks: mask 0. Then True where its logit is above 0.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Bij één klik: het mask met de hoogste score van masks 1 tot en met 3; bij meer klikken: mask 0. Dan True waar zijn logit boven 0 ligt."},
            "almost": {"en": "Right for the spoon. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use the scores and the number of clicks it is given?",
                       "nl": "Goed voor de lepel. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Rekent je functie met de scores en het aantal klikken die hij krijgt?"},
            "correct": {"en": "Your function is right in all {n} cases. For the click on the spoon, it picks mask 3, the bowl, with a score of 0.941. For the click on the handle of the cup, mask 0 has the highest score of all four, 0.919, but with one click it takes no part: your function picks mask 3, the handle, with 0.894.",
                        "nl": "Je functie klopt in alle {n} gevallen. Bij de klik op de lepel kiest hij mask 3, het kommetje, met een score van 0,941. Bij de klik op het oor van het kopje heeft mask 0 de hoogste score van alle vier, 0,919, maar bij één klik doet dat niet mee: je functie kiest mask 3, het oor, met 0,894."},
        },
    )


# --- Exercise 3: next_click ---------------------------------------------------------------------------------------------

_N_HEAD = ("from everything_ai.sam2 import deepest\n\n\n"
           "def next_click(mask, truth):\n"
           "    # mask: the mask SAM gave; truth: the mask of what you meant; both arrays of True and False\n")


def _next_with(missed: str = "truth & ~mask", extra: str = "mask & ~truth", which: str = "missed.sum() >= extra.sum()",
               first: str = "(x, y, 1)", second: str = "(x, y, 0)", where: tuple = ("missed", "extra"),
               none: bool = True) -> str:
    """The course's code, with other lines for the two regions, the choice between them, where it looks for the
    deepest pixel or what it gives back."""
    stop = "    if not missed.any() and not extra.any():\n        return None\n" if none else ""
    return (_N_HEAD + f"    missed = {missed}\n    extra = {extra}\n" + stop
            + f"    if {which}:\n        x, y = deepest({where[0]})\n        return {first}\n"
            + f"    x, y = deepest({where[1]})\n    return {second}\n")


_N_START = _next_with("...", "...", "...", "...", "...")
_N_SOLUTION = _next_with()


def _disk(h: int, w: int, cy: float, cx: float, r: float) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    return (y - cy) ** 2 + (x - cx) ** 2 <= r * r


def _next_cases():
    """The whole spoon, which SAM's mask of the bowl misses part of: its handle. Then masks the course makes up, on
    images wider than high: a mask too small, one too large, one that misses a second piece, one that is right, no
    mask at all, and a mask beside the thing. The missed and the extra pixels are never as many."""
    h, w = 40, 64
    two = _disk(h, w, 12, 14, 7) | _disk(h, w, 28, 48, 9)
    return ([(_Named("bowl", mask("spoon3")), _Named("spoon", mask("spoon1")))],
            [(_Named("small", _disk(h, w, 20, 30, 6)), _Named("disk", _disk(h, w, 20, 30, 12))),
             (_Named("large", _disk(h, w, 20, 30, 14)), _Named("disk", _disk(h, w, 22, 36, 7))),
             (_Named("one", _disk(h, w, 12, 14, 7)), _Named("two", two)),
             (_Named("disk", _disk(h, w, 20, 30, 12)), _Named("disk", _disk(h, w, 20, 30, 12))),
             (_Named("nothing", np.zeros((h, w), dtype=bool)), _Named("disk", _disk(h, w, 22, 40, 10))),
             (_Named("beside", _disk(h, w, 20, 16, 11)), _Named("disk", _disk(h, w, 20, 48, 8)))])


def _click_text(v) -> str:
    """What next_click gave back: a click (x, y, label) as plain whole numbers, or None."""
    if v is None:
        return "None"
    try:
        return "(" + ", ".join(str(int(x)) if float(x) == int(x) else str(x) for x in v) + ")"
    except (TypeError, ValueError):
        return str(v)


def _same_click(got, want) -> bool:
    if want is None or got is None:
        return got is None and want is None
    try:
        return len(got) == 3 and all(float(a) == float(b) for a, b in zip(got, want))
    except (TypeError, ValueError):
        return False


def _click_picture(seen) -> str:
    """Where the learner's next_click clicks on the spoon: SAM's mask of the bowl, the outline of the whole spoon,
    the pixels the mask gets wrong and the click, beside the course's click where the learner's is another."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    mine = seen.mine
    fn = mine.names.get("next_click") if mine else None
    if not callable(fn):
        return ""
    bowl, spoon = mask("spoon3"), mask("spoon1")
    try:
        got = with_time_limit(lambda: fn(bowl.copy(), spoon.copy()))
        got = (int(got[0]), int(got[1]), int(got[2]))
    except Exception:
        return ""
    right = next_click(bowl, spoon)
    lang = seen.lang
    panels = [(course.phrase("yours-code", lang), got)]
    if got != right:
        panels.append((course.phrase("yours-right", lang), right))
    plots.style()
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.0), squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.05)
    crop = _crop(spoon, bowl, margin=40)
    word = {"en": "{who}: ({x}, {y}), label {label}", "nl": "{who}: ({x}, {y}), label {label}"}[lang]
    for ax, (who, (x, y, label)) in zip(axes[0], panels):
        _axes(ax, photo(), word.format(who=who, x=x, y=y, label=label), crop)
        _mask(ax, bowl)
        _wrong(ax, bowl ^ spoon)
        _truth(ax, spoon)
        inside = (min(max(x, 0), bowl.shape[1] - 1), min(max(y, 0), bowl.shape[0] - 1))   # x and y swapped can lie outside
        _clicks(ax, [SPOON, inside], [1, label])
    title = {"en": "Where your next_click clicks after the click on the bowl",
             "nl": "Waar jouw next_click klikt na de klik op het kommetje"}[lang]
    alt = "; ".join(word.format(who=who, x=c[0], y=c[1], label=c[2]) for who, c in panels)
    return yours.figure(title, _render(fig, alt) + key_html(lang, ("mask", "truth", "wrong", "click", "negative")),
                        alt=alt)


def _nm(key, en, nl, **kw):
    return Mistake(key, en="`{call}` gives `{got}`, and it should give `{want}`. " + en,
                   nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. " + nl, **kw)


def _next_exercise():
    second = REFERENCE["second"]
    visible, hidden = _next_cases()
    return FunctionBox(
        name="next",
        picture=_click_picture,
        function="next_click",
        start=_N_START,
        solution=_N_SOLUTION,
        visible=visible,
        hidden=hidden,
        show=_click_text,
        compare=_same_click,
        first=[_DOTS],
        mistakes=[
            Mistake("unpack-none", error=lambda e: isinstance(e, TypeError) and "NoneType" in str(e),
                    en="With `{call}`, your function stops with an error: `deepest` gave None, because the region you gave it has no pixels. When the mask is right, both regions are empty: give back None before you look for a click.",
                    nl="Met `{call}` stopt je functie met een fout: `deepest` gaf None, omdat het gebied dat je hem gaf geen pixels heeft. Als het mask goed is, zijn beide gebieden leeg: geef dan None terug, voordat je een klik zoekt."),
            _nm("labels-swapped", "The labels are the other way around: a click on pixels the mask missed says \"this belongs to it too\", label 1; a click on pixels the mask has too many says \"this is not it\", label 0. Look at `missed` and `extra` too: the pixels the mask missed are in `truth` and not in the mask, `truth & ~mask`.",
                "De labels zijn andersom: een klik op pixels die het mask mist zegt \"dit hoort er ook bij\", label 1; een klik op pixels die het mask te veel heeft zegt \"dit is het niet\", label 0. Kijk ook naar `missed` en `extra`: de pixels die het mask mist, zitten in `truth` en niet in het mask, `truth & ~mask`.",
                wrong=_next_with(first="(x, y, 0)", second="(x, y, 1)")),
            _nm("fewer", "That clicks in the smaller of the two regions. Click where the mask is most wrong: in the region with the most pixels.",
                "Zo klik je in het kleinste van de twee gebieden. Klik waar het mask het meest fout is: in het gebied met de meeste pixels.",
                wrong=_next_with(which="missed.sum() <= extra.sum()")),
            _nm("only-missed", "Your function always clicks in the pixels the mask missed. When the mask has more pixels too many, click in those, with label 0.",
                "Je functie klikt altijd in de pixels die het mask mist. Als het mask meer pixels te veel heeft, klik dan daarin, met label 0.",
                wrong=_next_with(which="missed.any()")),
            _nm("xy", "x and y are swapped: `deepest` gives (x, y), x across and y down, and the click is (x, y, label) too.",
                "x en y zijn omgewisseld: `deepest` geeft (x, y), x opzij en y omlaag, en de klik is ook (x, y, label).",
                wrong=_next_with(first="(y, x, 1)", second="(y, x, 0)")),
            _nm("whole", "`missed` and `extra` are the whole of `truth` and the whole mask. Take only the pixels one has and the other has not: those of `truth` the mask missed are `truth & ~mask`.",
                "`missed` en `extra` zijn heel `truth` en het hele mask. Neem alleen de pixels die het een heeft en het ander niet: die van `truth` die het mask mist, zijn `truth & ~mask`.",
                wrong=_next_with(missed="truth", extra="mask")),
            Mistake("compare-arrays", error=lambda e: isinstance(e, ValueError) and "ambiguous" in str(e),
                    en="With `{call}`, your function stops with an error: comparing `missed` and `extra` themselves compares them pixel by pixel, and gives an array, not one True or False. Compare how many pixels each has: `missed.sum() >= extra.sum()`.",
                    nl="Met `{call}` stopt je functie met een fout: `missed` en `extra` zelf vergelijken gaat pixel voor pixel, en geeft een array, niet één True of False. Vergelijk hoeveel pixels elk heeft: `missed.sum() >= extra.sum()`."),
            Mistake("minus", error=lambda e: isinstance(e, TypeError) and "subtract" in str(e),
                    en="With `{call}`, your function stops with an error: `-` does not work on arrays of True and False. The pixels in `truth` and not in `mask` are `truth & ~mask`.",
                    nl="Met `{call}` stopt je functie met een fout: `-` werkt niet op arrays van True en False. De pixels in `truth` en niet in `mask` zijn `truth & ~mask`."),
            _nm("no-label", "A click needs its label too: give back (x, y, label), with 1 for missed pixels and 0 for pixels too many.",
                "Een klik heeft ook zijn label nodig: geef (x, y, label) terug, met 1 voor gemiste pixels en 0 voor pixels te veel.",
                matches=lambda got, text: isinstance(got, (tuple, list)) and len(got) == 2),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace each `...`: the pixels the mask missed, the pixels it has too many, which of the two is larger, and the click in each, (x, y, label).",
                      "nl": "Vervang elke `...`: de pixels die het mask mist, de pixels die het te veel heeft, welke van de twee groter is, en de klik in elk, (x, y, label)."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with SAM's mask and the mask of what you meant, and the `def` line of your function has a different number of names between its parentheses. Keep `(mask, truth)`.",
                          "nl": "De check roept `{call}` aan, met het mask van SAM en het mask van wat je bedoelt, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(mask, truth)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Take the pixels of `truth` the mask missed and the pixels of the mask that are not in `truth`; click in the deepest pixel of the larger of the two, with label 1 for missed pixels and 0 for pixels too many.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Neem de pixels van `truth` die het mask mist en de pixels van het mask die niet in `truth` zitten; klik in de diepste pixel van de grootste van de twee, met label 1 voor gemiste pixels en 0 voor pixels te veel."},
            "almost": {"en": "Right for the spoon. But `{call}` gives `{got}`, and it should give `{want}`. Does your function also click on pixels the mask has too many, with label 0, and give back None when the mask is right?",
                       "nl": "Goed voor de lepel. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Klikt je functie ook op pixels die het mask te veel heeft, met label 0, en geeft hij None terug als het mask goed is?"},
            "correct": {"en": "Your function is right in all {{n}} cases. On the spoon, it clicks on the end of the handle, ({x}, {y}): the mask of the bowl misses {missed:,} pixels of the spoon, and has only {extra} too many. With that second click, SAM's mask 0 is the bowl with the end of the handle: its IoU with the spoon of mask 1 goes from {before} to {after}.".format(
                            x=second["click"][0], y=second["click"][1], missed=second["missed"], extra=second["extra"],
                            before=second["before"], after=second["after"]),
                        "nl": "Je functie klopt in alle {{n}} gevallen. Op de lepel klikt hij op het eind van de steel, ({x}, {y}): het mask van het kommetje mist daar {missed} pixels van de lepel, en heeft er maar {extra} te veel. Met die tweede klik is mask 0 van SAM het kommetje met het eind van de steel: zijn IoU met de lepel van mask 1 gaat van {before} naar {after}.".format(
                            x=second["click"][0], y=second["click"][1], missed=f"{second['missed']:,}".replace(",", "."),
                            extra=second["extra"], before=str(second["before"]).replace(".", ","), after=str(second["after"]).replace(".", ","))},
        },
    )


EXERCISES = {"choose": _choose_exercise(), "next": _next_exercise()}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("choose", _C_START, "empty", None),
    ("choose", _C_SOLUTION, "correct", None),
    ("choose", _C_HEAD + "    if clicks > 1:\n        return logits[0] > 0\n    best = np.argmax(scores[1:]) + 1\n    return logits[best] > 0\n",
     "correct", None),
    ("choose", _C_HEAD + "    k = 0\n    if clicks == 1:\n        for i in (1, 2, 3):\n            if k == 0 or scores[i] > scores[k]:\n                k = i\n"
     "    return logits[k] > 0\n", "correct", None),
    ("choose", _choose_with(one="np.argmax(scores)"), "wrong", "all-four"),
    ("choose", _choose_with(one="np.argmax(scores[1:])"), "wrong", "shifted"),
    ("choose", _choose_with(one="1 + np.argmin(scores[1:])"), "wrong", "lowest"),
    ("choose", _choose_with(one="1 + np.argmax((logits[1:] > 0).sum(axis=(1, 2)))"), "wrong", "largest"),
    ("choose", _choose_with(one="0"), "wrong", "always-zero"),
    ("choose", _choose_with(one="0", more="1 + np.argmax(scores[1:])"), "wrong", "swapped"),
    ("choose", _choose_with(more="1 + np.argmax(scores[1:])"), "wrong", "always-best"),
    ("choose", _choose_with(back="logits[k] > 0.5"), "wrong", "probability"),
    ("choose", _choose_with(back="logits[k]"), "wrong", "logits"),
    ("choose", _choose_with(back="k"), "wrong", "index"),
    ("choose", _choose_with(back="logits > 0"), "wrong", "all-masks"),
    ("choose", _C_START.replace("k = ...\n    else", "k = 1 + np.argmax(scores[1:])\n    else", 1), "wrong", "dots-left"),
    ("choose", _C_SOLUTION.replace("def choose(logits, scores, clicks):", "def choose(logits, scores):"), "error", "arguments"),
    ("choose", _C_SOLUTION.replace("    return logits[k] > 0\n", "    print(logits[k] > 0)\n"), "wrong", "prints"),
    ("choose", _C_SOLUTION.replace("    return logits[k] > 0\n", ""), "wrong", "no-return"),
    ("choose", "", "empty", None),
    ("next", _N_START, "empty", None),
    ("next", _N_SOLUTION, "correct", None),
    ("next", _next_with(first="[x, y, 1]", second="[x, y, 0]"), "correct", None),
    ("next", _N_HEAD + "    missed = truth & ~mask\n    extra = mask & ~truth\n    if missed.sum() == 0 and extra.sum() == 0:\n"
     "        return None\n    if missed.sum() >= extra.sum():\n        return (*deepest(missed), 1)\n"
     "    return (*deepest(extra), 0)\n", "correct", None),
    ("next", _N_HEAD + "    missed = truth & (mask == False)\n    extra = mask & (truth == False)\n"
     "    if missed.sum() + extra.sum() == 0:\n        return None\n    if missed.sum() > extra.sum():\n"
     "        x, y = deepest(missed)\n        return (x, y, 1)\n    x, y = deepest(extra)\n    return (x, y, 0)\n", "correct", None),
    ("next", _next_with(first="(x, y, 0)", second="(x, y, 1)"), "wrong", "labels-swapped"),
    ("next", _next_with(missed="mask & ~truth", extra="truth & ~mask"), "wrong", "labels-swapped"),
    ("next", _next_with(which="missed.sum() <= extra.sum()"), "wrong", "fewer"),
    ("next", _next_with(which="missed.any()"), "wrong", "only-missed"),
    ("next", _next_with(first="(y, x, 1)", second="(y, x, 0)"), "wrong", "xy"),
    ("next", _next_with(missed="truth", extra="mask"), "wrong", "whole"),
    ("next", _next_with(which="missed >= extra"), "error", "compare-arrays"),
    ("next", _next_with(missed="truth - mask", extra="mask - truth"), "error", "minus"),
    ("next", _next_with(first="(x, y)", second="(x, y)"), "wrong", "no-label"),
    ("next", _next_with(none=False), "error", "unpack-none"),
    ("next", _N_START.replace("missed = ...", "missed = truth & ~mask"), "wrong", "dots-left"),
    ("next", _N_SOLUTION.replace("def next_click(mask, truth):", "def next_click(mask):"), "error", "arguments"),
    ("next", "", "empty", None),
]
