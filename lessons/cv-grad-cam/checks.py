# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-grad-cam: class_map, the class activation map of feature maps for one class (the maps, each
times its weight, added up), and grad_cam, Grad-CAM (the feature maps of a layer, weighted by the mean slope of a
class's logit over each map, added up, and only the positive part); each saved in the learner's project
(work/projects/cnn/cam.py and gradcam.py) when right. Both are checked on made-up maps, on a ResNet-18 with random
weights and on the course's DigitCNN with random weights, a CNN without global average pooling, so they work without
the download. ResNet-18 itself, with torchvision's weights trained on ImageNet, 47 MB, the lesson downloads once when
the learner asks for it, as cv-try-classifier does: the worked example, the maps of the collage and the demo need it.
The slider in *The idea* needs no download: it shows Grad-CAMs the course measured, in REFERENCE.
tests/test_cv_grad_cam.py measures them all again when EAI_RESNET18_FILE names the downloaded file."""
import base64
import copy
import html
import importlib.util
import io
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import models, paths, projects
from everything_ai.check import Mistake, ScenarioBox, run_code, with_time_limit
from everything_ai import yours


def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_grad_cam",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SAL = _lesson_checks("cv-saliency")   # the photos as ResNet-18 sees them, saliency maps, and shared check texts
_TRY = _SAL._TRY                       # the download of ResNet-18, and its button's texts
_INSIDE = _SAL._INSIDE                 # how a map is drawn: _slim and _bare
resnet_or_stop = _TRY.resnet_or_stop
crop, as_input, prepared, best = _SAL.crop, _SAL.as_input, _SAL.prepared, _SAL.best
course_saliency = _SAL.course_saliency   # the saliency map of cv-saliency, for the maps of the collage

DOWNLOADS = {"ResNet-18": models.RESNET18}   # what the lesson downloads, for the line under its header

# The classes the lesson names, by their place among ImageNet's 1,000 logits (data/imagenet-classes.txt, from 0).
EGYPTIAN, ESPRESSO = 285, 967
SIDE = 224            # the photo as ResNet-18 sees it: 224 by 224 pixels
HALF = SIDE // 2      # the collage: the cat in columns 0 to 111, the coffee in columns 112 to 223
GRID = 7              # a map of layer4 for a photo of 224 by 224: 7 by 7, each number for 32 by 32 pixels
RANKS = list(range(1, 9))   # the stops of the slider: ResNet-18's 8 best guesses for the collage
VEIL = 0.85           # how much of the page a cell's veil shows where a map is 0: none where the map is largest
DEMO_PHOTOS = ["collage", "cat", "coffee", "fruit"]
LAYERS = ["layer4", "layer3"]


def percent(share: float) -> int:
    return round(100 * share)


# --- the course's versions of the exercises ------------------------------------------------------------------------

def course_class_map(maps, weights):
    """The class activation map of feature maps of shape (K, H, W) with one weight per map, shape (K,): shape (H, W)."""
    return (maps * weights.reshape(-1, 1, 1)).sum(dim=0)


def course_grad_cam(model, layer, x, label):
    """Grad-CAM of photo x, of shape (1, channels, H, W), for class `label`, from the output of `layer`: a tensor of the shape
    of one of its maps, without recording. x itself is left as it was."""
    maps = []
    handle = layer.register_forward_hook(lambda module, inputs, output: maps.append(output))
    x = x.clone()
    x.requires_grad = True
    try:
        score = model(x)[0, label]
    finally:
        handle.remove()
    A = maps[0]
    A.retain_grad()
    score.backward()
    alpha = A.grad[0].mean(dim=(1, 2))
    with torch.no_grad():
        return torch.relu(course_class_map(A[0], alpha))


# --- the photos, the maps and what the lesson measures on them -----------------------------------------------------

def collage() -> np.ndarray:
    """The middle half of the cat beside the middle half of the coffee: 224 by 224 by 3, colors of 0 to 255."""
    return np.concatenate([crop("cat")[:, HALF // 2:HALF // 2 + HALF], crop("coffee")[:, HALF // 2:HALF // 2 + HALF]], axis=1)


def photo_of(name: str) -> np.ndarray:
    """A photo of DEMO_PHOTOS as ResNet-18 sees it: 224 by 224 by 3, colors of 0 to 255."""
    return collage() if name == "collage" else crop(name)


def input_of(name: str) -> torch.Tensor:
    return as_input(photo_of(name))


def feature_maps(model, layer, x):
    """The output of `layer` for photo x, of shape (1, K, h, w), and the logits, without recording."""
    maps = []
    handle = layer.register_forward_hook(lambda module, inputs, output: maps.append(output))
    try:
        with torch.no_grad():
            logits = model(x)
    finally:
        handle.remove()
    return maps[0], logits


def enlarged(m) -> np.ndarray:
    """A map of h by w as 224 by 224: each number a block of 224 / h by 224 / w pixels. Only for maps whose sides
    divide 224, as those of layer4 and layer3 do: 7 and 14."""
    m = np.asarray(m, dtype=float)
    if SIDE % m.shape[0] or SIDE % m.shape[1]:
        raise ValueError(f"a map of {m.shape[0]} by {m.shape[1]} does not divide {SIDE} by {SIDE} into blocks")
    return np.kron(m, np.ones((SIDE // m.shape[0], SIDE // m.shape[1])))


def cat_share(m) -> float:
    """The share of a map, enlarged to the collage, that lies on its cat half: columns 0 to 111."""
    big = enlarged(m)
    return float(big[:, :HALF].sum() / big.sum())


def measure(model) -> dict:
    """Everything REFERENCE holds, measured again with the course's code."""
    names = models.imagenet_classes()
    out = {}
    x = prepared("cat")
    A, logits = feature_maps(model, model.layer4, x)
    out["cat"] = {}
    for key, label in (("egyptian", EGYPTIAN), ("espresso", ESPRESSO)):
        cam = course_class_map(A[0], model.fc.weight[label])
        r, c = divmod(int(cam.argmax()), GRID)
        out["cat"][key] = {"logit": round(float(logits[0, label]), 4), "bias": round(float(model.fc.bias[label]), 4),
                           "mean": round(float(cam.mean()), 4), "largest": (round(float(cam.max()), 2), r, c),
                           "smallest": round(float(cam.min()), 2)}
    x = input_of("collage")
    A, logits = feature_maps(model, model.layer4, x)
    p = torch.softmax(logits[0], 0)
    top = [int(i) for i in p.argsort(descending=True)[:len(RANKS)]]
    out["collage"], out["layer3"], out["maps"] = [], [], []
    g4s = {}
    for label in top:
        g4s[label] = g4 = course_grad_cam(model, model.layer4, x, label).numpy()
        g3 = course_grad_cam(model, model.layer3, x, label).numpy()
        out["collage"].append((names[label], round(float(p[label]), 4), round(cat_share(g4), 4)))
        out["layer3"].append(round(cat_share(g3), 4))
        out["maps"].append(tuple(tuple(round(float(v), 2) for v in row) for row in g4 / g4.max()))
    out["logits"] = {key: round(float(logits[0, label]), 4) for key, label in (("egyptian", EGYPTIAN), ("espresso", ESPRESSO))}
    g = g4s[EGYPTIAN]
    cam = course_class_map(A[0], model.fc.weight[EGYPTIAN])
    out["zeros"] = {"cells": int((g == 0).sum()), "first_column": int(np.nonzero(g == 0)[1].min()), "smallest": round(float(cam.min()), 2),
                    "smallest_at": divmod(int(cam.argmin()), GRID)}
    out["saliency"], out["saliency_white"] = {}, {}
    for key, label in (("egyptian", EGYPTIAN), ("espresso", ESPRESSO)):
        sal = course_saliency(model, x, label)
        white = sal >= np.quantile(sal, _SAL.WHITE)
        out["saliency"][key] = round(cat_share(sal), 4)
        out["saliency_white"][key] = round(float(white[:, :HALF].sum() / white.sum()), 4)
    out["top"] = {}
    for photo in ("cat", "coffee", "fruit"):
        with torch.no_grad():
            out["top"][photo] = best(torch.softmax(model(prepared(photo))[0], 0))
    return out


# What the course measured with its code (course_class_map and course_grad_cam, everything_ai.models.resnet18 and its
# pinned file, the five steps of cv-prepare-photo, one thread). Probabilities are softmax over the 1,000 logits.
# "cat": on the cat, for Egyptian cat and for espresso, the logit, the bias of the class in fc, the mean of the class
# activation map from layer4 (logit = mean + bias), its largest number with its row and column, and its smallest.
# "collage": ResNet-18's 8 best guesses for the collage (the middle half of the cat beside the middle half of the
# coffee), each with its probability and the share of its Grad-CAM from layer4 that lies on the cat half, as the
# map enlarged to 224 by 224 (each number a block of 32 by 32) adds it up; "layer3": the same share from layer3, a
# map of 14 by 14. "maps": the 8 Grad-CAMs from layer4, each divided by its largest number, with 2 decimals: the
# slider draws them. "logits": on the collage, the logits of Egyptian cat and of espresso. "zeros": of Egyptian cat on
# the collage, the cells of its Grad-CAM that are 0, the first column with such a cell (from column 4 on, a cell lies
# wholly on the coffee half), and the smallest number of its class activation map with its row and column.
# "saliency": the share of the saliency map of cv-saliency of Egyptian cat and of espresso on the collage that lies on
# the cat half; "saliency_white": the share of its 1% largest numbers, which the course draws white, that lies there.
# "top": each photo's three best guesses.
REFERENCE = {
    "cat": {"egyptian": {'logit': 19.6482, 'bias': 0.0092, 'mean': 19.639, 'largest': (33.57, 5, 1), 'smallest': 11.32},
            "espresso": {'logit': 0.0781, 'bias': 0.0229, 'mean': 0.0552, 'largest': (2.93, 3, 0), 'smallest': -5.42}},
    "collage": [('Egyptian cat', 0.3979, 0.9518), ('tiger cat', 0.2571, 0.9575), ('tabby', 0.2068, 0.9514), ('lynx', 0.0225, 0.9634),
                ('espresso', 0.0195, 0.1305), ('bucket', 0.0106, 0.444), ('coffee mug', 0.0067, 0.2078), ('cup', 0.006, 0.2103)],
    "layer3": [0.3746, 0.5185, 0.4653, 0.4358, 0.4742, 0.6479, 0.4757, 0.4077],
    "maps": [
        ((0.47, 0.68, 0.5, 0.24, 0.02, 0.0, 0.0), (0.7, 1.0, 0.71, 0.32, 0.01, 0.0, 0.0), (0.73, 0.96, 0.66, 0.25, 0.0, 0.0, 0.0), (0.63, 0.78, 0.52, 0.2, 0.0, 0.0, 0.0), (0.56, 0.61, 0.35, 0.06, 0.0, 0.0, 0.01), (0.56, 0.62, 0.35, 0.09, 0.0, 0.0, 0.0), (0.48, 0.52, 0.31, 0.09, 0.0, 0.0, 0.0)),
        ((0.49, 0.61, 0.4, 0.18, 0.0, 0.01, 0.02), (0.71, 0.93, 0.58, 0.25, 0.0, 0.0, 0.0), (0.75, 1.0, 0.63, 0.26, 0.0, 0.0, 0.0), (0.71, 0.87, 0.54, 0.2, 0.0, 0.0, 0.0), (0.67, 0.72, 0.39, 0.1, 0.0, 0.0, 0.0), (0.7, 0.75, 0.45, 0.11, 0.0, 0.0, 0.0), (0.6, 0.62, 0.39, 0.09, 0.0, 0.0, 0.0)),
        ((0.49, 0.68, 0.46, 0.2, 0.0, 0.01, 0.02), (0.72, 0.98, 0.66, 0.28, 0.0, 0.01, 0.02), (0.77, 1.0, 0.66, 0.25, 0.0, 0.0, 0.0), (0.67, 0.83, 0.57, 0.18, 0.0, 0.0, 0.0), (0.59, 0.66, 0.41, 0.11, 0.0, 0.0, 0.0), (0.57, 0.64, 0.42, 0.13, 0.01, 0.01, 0.0), (0.49, 0.53, 0.34, 0.09, 0.0, 0.0, 0.0)),
        ((0.46, 0.68, 0.5, 0.26, 0.02, 0.0, 0.0), (0.7, 1.0, 0.72, 0.31, 0.0, 0.0, 0.0), (0.67, 0.92, 0.61, 0.19, 0.0, 0.0, 0.0), (0.71, 0.85, 0.49, 0.13, 0.0, 0.0, 0.0), (0.73, 0.76, 0.4, 0.03, 0.0, 0.0, 0.0), (0.68, 0.73, 0.39, 0.06, 0.0, 0.0, 0.0), (0.55, 0.57, 0.32, 0.04, 0.0, 0.0, 0.0)),
        ((0.03, 0.05, 0.15, 0.33, 0.48, 0.54, 0.4), (0.0, 0.03, 0.11, 0.38, 0.65, 0.84, 0.69), (0.0, 0.0, 0.0, 0.37, 0.71, 1.0, 0.82), (0.02, 0.03, 0.06, 0.35, 0.67, 0.93, 0.75), (0.0, 0.04, 0.1, 0.28, 0.42, 0.52, 0.37), (0.02, 0.05, 0.07, 0.16, 0.22, 0.23, 0.24), (0.0, 0.0, 0.02, 0.13, 0.14, 0.17, 0.2)),
        ((0.27, 0.31, 0.33, 0.58, 0.67, 0.76, 0.55), (0.46, 0.54, 0.48, 0.64, 0.8, 1.0, 0.77), (0.59, 0.6, 0.44, 0.46, 0.61, 0.81, 0.62), (0.59, 0.57, 0.44, 0.45, 0.54, 0.77, 0.57), (0.49, 0.48, 0.34, 0.26, 0.24, 0.3, 0.16), (0.24, 0.21, 0.17, 0.06, 0.18, 0.19, 0.21), (0.13, 0.1, 0.1, 0.0, 0.1, 0.12, 0.21)),
        ((0.13, 0.14, 0.24, 0.51, 0.65, 0.6, 0.43), (0.19, 0.18, 0.21, 0.54, 0.78, 0.95, 0.74), (0.11, 0.07, 0.13, 0.46, 0.76, 1.0, 0.77), (0.12, 0.12, 0.12, 0.35, 0.65, 0.89, 0.7), (0.07, 0.05, 0.06, 0.18, 0.31, 0.37, 0.22), (0.01, 0.0, 0.0, 0.05, 0.17, 0.16, 0.14), (0.03, 0.0, 0.01, 0.06, 0.13, 0.08, 0.11)),
        ((0.09, 0.13, 0.23, 0.48, 0.65, 0.72, 0.55), (0.11, 0.13, 0.19, 0.5, 0.74, 1.0, 0.83), (0.13, 0.13, 0.11, 0.42, 0.69, 0.99, 0.83), (0.15, 0.16, 0.08, 0.35, 0.62, 0.95, 0.79), (0.11, 0.1, 0.06, 0.19, 0.29, 0.46, 0.34), (0.1, 0.07, 0.01, 0.1, 0.21, 0.26, 0.33), (0.1, 0.06, 0.06, 0.09, 0.12, 0.11, 0.19)),
    ],
    "logits": {'egyptian': 11.1965, 'espresso': 8.1797},
    "zeros": {'cells': 15, 'first_column': 4, 'smallest': -2.54, 'smallest_at': (3, 5)},
    "saliency": {'egyptian': 0.745, 'espresso': 0.4409},
    "saliency_white": {'egyptian': 0.9821, 'espresso': 0.1135},
    "top": {'cat': [('Egyptian cat', 0.3914), ('tiger cat', 0.3861), ('tabby', 0.2176)], 'coffee': [('espresso', 0.8517), ('chocolate sauce', 0.0403), ('consomme', 0.0391)], 'fruit': [('balloon', 0.7349), ('ping-pong ball', 0.1971), ('Granny Smith', 0.0277)]},
}


# --- the pictures and the sentences under them ---------------------------------------------------------------------

_JPEGS = {}


def _jpeg(name: str) -> str:
    """A photo of DEMO_PHOTOS as a JPEG in base64, to draw inside an SVG: much smaller than a PNG of a photo."""
    if name not in _JPEGS:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(np.asarray(photo_of(name), dtype=np.uint8)).save(buf, format="JPEG", quality=90)
        _JPEGS[name] = base64.b64encode(buf.getvalue()).decode("ascii")
    return _JPEGS[name]


def _panel(name: str, m, x0: float, y0: float, side: int, title: str, under: list) -> list:
    """One photo with a map over it, in SVG: the photo, and over every cell of the map a veil in the page's color,
    as opaque as VEIL where the map is 0 and clear where it is largest. A title above it; `under` gives texts under
    it, each (x as a share of the side, text)."""
    m = np.asarray(m, dtype=float)
    top = float(m.max())
    h, w = m.shape
    parts = [f'<text x="{x0}" y="{y0 - 10}" style="font-size:14px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>',
             f'<image href="data:image/jpeg;base64,{_jpeg(name)}" x="{x0}" y="{y0}" width="{side}" height="{side}" preserveAspectRatio="none"/>',
             '<g shape-rendering="crispEdges">']
    for i in range(h):
        for j in range(w):
            a = VEIL * (1 - m[i, j] / top) if top > 0 else VEIL
            if a >= 0.005:
                xa, xb = x0 + round(j * side / w), x0 + round((j + 1) * side / w)
                ya, yb = y0 + round(i * side / h), y0 + round((i + 1) * side / h)
                parts.append(f'<rect x="{xa}" y="{ya}" width="{xb - xa}" height="{yb - ya}" style="fill:var(--ea-page);opacity:{a:.2f}"/>')
    parts.append("</g>")
    for share, text in under:
        parts.append(f'<text x="{x0 + share * side:.1f}" y="{y0 + side + 18}" style="font-size:13px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(text)}</text>')
    return parts


def _svg(parts: list, width: int, height: int, alt: str, max_width: int) -> str:
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(alt)}" style="width:100%;max-width:{max_width}px;height:auto;'
            f'font-family:var(--ea-font-ui)"><title>{html.escape(alt)}</title>' + "".join(parts) + "</svg>")


def halves(share: float) -> tuple:
    """The percentages of a map on the cat half and on the coffee half, adding up to 100."""
    cat = percent(share)
    return cat, 100 - cat


def guess_frames(labels: dict, lang: str) -> dict:
    """For the slider in *The idea*: for each of ResNet-18's 8 best guesses for the collage, the collage with that
    guess's Grad-CAM from layer4 over it, from REFERENCE, and the sentence under it. Drawn 330 wide, for a phone.
    `labels` gives "title" ({rank}, {name} and {p}), "cat" and "coffee" ({share}), and "alt" ({name}, {cat} and
    {coffee})."""
    sep = "," if lang == "nl" else "."
    W, SIDE_SHOWN, TOP = 330, GRID * 42, 30
    frames = {}
    for rank in RANKS:
        name, p, share = REFERENCE["collage"][rank - 1]
        cat, coffee = halves(share)
        parts = _panel("collage", REFERENCE["maps"][rank - 1], 18, TOP, SIDE_SHOWN,
                       labels["title"].format(rank=rank, name=name, p=_TRY.shown(p, sep)),
                       [(0.25, labels["cat"].format(share=cat)), (0.75, labels["coffee"].format(share=coffee))])
        svg = _svg(parts, W, TOP + SIDE_SHOWN + 28, labels["alt"].format(name=name, cat=cat, coffee=coffee), W)
        frames[rank] = (svg, guess_sentence(rank, lang))
    return frames


def guess_sentence(rank: int, lang: str) -> str:
    sep = "," if lang == "nl" else "."
    name, p, share = REFERENCE["collage"][rank - 1]
    cat, coffee = halves(share)
    if lang == "nl":
        return (f"Gok {rank} van ResNet-18 voor de collage is {name}, met {_TRY.shown(p, sep)}. Van zijn Grad-CAM ligt "
                f"{cat}% op de helft met de kat en {coffee}% op de helft met de koffie.")
    return (f"ResNet-18's guess {rank} for the collage is {name}, at {_TRY.shown(p, sep)}. Of its Grad-CAM, {cat}% lies "
            f"on the cat half and {coffee}% on the coffee half.")


def mixed_rank() -> int:
    """The guess whose map lies least on one half: the share on the cat half closest to a half."""
    return min(RANKS, key=lambda r: abs(REFERENCE["collage"][r - 1][2] - 0.5))


def compare_picture(maps: dict, labels: dict) -> str:
    """On the collage, the saliency map and the Grad-CAM from layer4 of Egyptian cat and of espresso, as gray maps:
    a row per class. `maps` gives for "saliency" and "gradcam" a map per class key ("egyptian", "espresso");
    `labels` gives "saliency" and "gradcam" ({label} and {share}) and "alt"."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    names = models.imagenet_classes()
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.9), gridspec_kw={"wspace": 0.06, "hspace": 0.22})
    for row, (key, label) in enumerate((("egyptian", EGYPTIAN), ("espresso", ESPRESSO))):
        for col, kind in enumerate(("saliency", "gradcam")):
            m = np.asarray(maps[kind][key], dtype=float)
            ax = axes[row, col]
            top = float(np.quantile(m, _SAL.WHITE)) if kind == "saliency" else float(m.max())
            ax.imshow(m, cmap="gray", vmin=0, vmax=top, interpolation="none", extent=(0, SIDE, SIDE, 0))
            _INSIDE._bare(ax)
            ax.set_title(labels[kind].format(label=names[label], share=percent(cat_share(m))), fontsize=10)
    for ax in axes.ravel():
        for side in ax.spines.values():
            side.set_color("none")
    out = _INSIDE._slim(plots.render(fig, alt=labels["alt"]))
    plt.close(fig)
    return out


def demo_picture(result: dict, labels: dict, lang: str) -> str:
    """The demo: the photo three times, each with the learner's Grad-CAM of one of the three best guesses over it,
    side by side where there is room and below each other on a phone, so that the texts keep their size. `labels`
    gives "title" ({rank}, {name} and {p}), "cat" and "coffee" ({share}, under a collage) and "alt" (the whole)."""
    sep = "," if lang == "nl" else "."
    side, top = SIDE, 30
    height = top + side + (28 if result["shares"] is not None else 6)
    panels = []
    for i, (m, name, p) in enumerate(zip(result["maps"], result["names"], result["p"])):
        under = []
        if result["shares"] is not None:
            cat, coffee = halves(result["shares"][i])
            under = [(0.25, labels["cat"].format(share=cat)), (0.75, labels["coffee"].format(share=coffee))]
        title = labels["title"].format(rank=i + 1, name=_TRY._short(name, 18), p=_TRY.shown(p, sep))
        panels.append(f'<div style="flex:0 1 {side}px">'
                      + _svg(_panel(result["photo"], m, 0, top, side, title, under), side, height, title, side) + "</div>")
    return (f'<div role="img" aria-label="{html.escape(labels["alt"])}" style="display:flex;flex-wrap:wrap;gap:18px">'
            + "".join(panels) + "</div>")


def demo_sentence(result: dict, lang: str) -> str:
    """What the demo measured, with the learner's grad_cam and class_map."""
    sep = "," if lang == "nl" else "."
    said = [f"{name} ({_TRY.shown(p, sep)})" for name, p in zip(result["names"], result["p"])]
    layer = f"`{result['layer']}`"
    if lang == "nl":
        out = f"Je `grad_cam` uit {layer}, voor de drie beste gokken van ResNet-18: {said[0]}, {said[1]} en {said[2]}."
        if result["shares"] is not None:
            cat = [f"{percent(s)}%" for s in result["shares"]]
            out += f" Op de helft met de kat ligt van de drie maps {cat[0]}, {cat[1]} en {cat[2]}."
        if result["same"] is not None:
            out += (f" En je `class_map` van {result['names'][0]}, met de weights van `fc`: zijn positieve deel, gedeeld door 49, "
                    + ("is dezelfde map als je Grad-CAM, op 0,00001 na." if result["same"] <= 1e-5
                       else f"wijkt tot {result['same']:.4f} af van je Grad-CAM.".replace(".", ",", 1)))
        return out
    out = f"Your `grad_cam` from {layer}, for ResNet-18's three best guesses: {said[0]}, {said[1]} and {said[2]}."
    if result["shares"] is not None:
        cat = [f"{percent(s)}%" for s in result["shares"]]
        out += f" Of each map, on the cat half: {cat[0]}, {cat[1]} and {cat[2]}."
    if result["same"] is not None:
        out += (f" And your `class_map` of {result['names'][0]}, with the weights of `fc`: its positive part, divided by 49, "
                + ("is the same map as your Grad-CAM, to within 0.00001." if result["same"] <= 1e-5
                   else f"differs from your Grad-CAM by up to {result['same']:.4f}."))
    return out


# --- the exercises ---------------------------------------------------------------------------------------------------

# The lesson is done when these are correct. "logit" and "evidence" are the choice questions in the lesson.
REQUIRED = ["logit", "class-map", "grad-cam", "evidence"]

_DOTS, _TEXTS, _S_EN, _S_NL, _close, _raised_in = _SAL._DOTS, _SAL._TEXTS, _SAL._S_EN, _SAL._S_NL, _SAL._close, _SAL._raised_in


def _broadcast(name: str):
    return lambda e: isinstance(e, RuntimeError) and "must match the size of tensor" in str(e) and _raised_in(e, name)


def _code(text: str) -> str:
    """The learner's text without its comments, such as the starter's, which name `model.layer4`."""
    return re.sub(r"#[^\n]*", "", text)


# --- exercise 2: the class activation map --------------------------------------------------------------------------

_CM_HEAD = "import torch\n\n\ndef class_map(maps, weights):\n"
_CM_START = (_CM_HEAD + "    # maps: the feature maps of one photo, a tensor of shape (K, H, W)\n"
             "    # weights: one weight per map, a tensor of shape (K,)\n"
             "    # give back the class activation map, a tensor of shape (H, W):\n"
             "    # each map times its weight, all added up\n    ...\n")
_CM_BODY = "    return (maps * weights.reshape(-1, 1, 1)).sum(dim=0)\n"


def _cm(body: str = _CM_BODY) -> str:
    return _CM_HEAD + body


_CM = _cm()
_CM_SCENARIOS = [
    # the two maps of exercise 1
    "maps = torch.tensor([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 2.0]]])\nweights = torch.tensor([3.0, -1.0])\nclass_map(maps, weights).tolist()",
    # three maps of 2 by 4: a number of maps that is not the number of columns
    "torch.manual_seed(0)\nmaps = torch.randn(3, 2, 4)\nweights = torch.tensor([1.0, -2.0, 0.5])\ncam = class_map(maps, weights)\n"
    "[list(cam.shape), cam.sum().item(), cam[1, 3].item()]",
    # 512 maps of 7 by 7, as layer4 of ResNet-18 gives for one photo
    "torch.manual_seed(1)\nmaps = torch.relu(torch.randn(512, 7, 7))\nweights = torch.randn(512) / 100\ncam = class_map(maps, weights)\n"
    "[list(cam.shape), cam.mean().item(), cam.max().item(), cam.min().item()]",
]
_CM_WRONG = {
    "mean": _cm("    return (maps * weights.reshape(-1, 1, 1)).mean(dim=0)\n"),
    "no-weights": _cm("    return maps.sum(dim=0)\n"),
    "relu": _cm("    return torch.relu((maps * weights.reshape(-1, 1, 1)).sum(dim=0))\n"),
    "pooled": _cm("    return maps.mean(dim=(1, 2)) * weights\n"),
    "logit": _cm("    return (maps.mean(dim=(1, 2)) * weights).sum()\n"),
    "sum-all": _cm("    return (maps * weights.reshape(-1, 1, 1)).sum()\n"),
    "rows": _cm("    return (maps * weights.reshape(-1, 1, 1)).sum(dim=1)\n"),
}

# --- exercise 3: Grad-CAM ------------------------------------------------------------------------------------------

_GC_HEAD = "import torch\n\n\ndef grad_cam(model, layer, x, label):\n"
_GC_KEEP = "    maps = []\n\n    def keep(module, inputs, output):\n        maps.append(output)\n\n"
_GC_START = (_GC_HEAD + "    # model: a CNN in eval mode; layer: one of its layers, such as model.layer4\n"
             "    # x: one photo, a tensor of shape (1, channels, H, W)\n"
             "    # give back the Grad-CAM of x for the class label, a tensor of shape (h, w):\n"
             "    # the maps of layer, each times the mean slope of the logit over that map,\n"
             "    # added up, and only the positive part\n" + _GC_KEEP + "    ...\n")
_GC_FORWARD = ("    handle = layer.register_forward_hook(keep)\n    x.requires_grad = True\n    score = model(x)[0, label]\n"
               "    handle.remove()\n    A = maps[0]\n    A.retain_grad()\n    score.backward()\n")
_GC_BODY = _GC_FORWARD + ("    alpha = A.grad[0].mean(dim=(1, 2))\n    cam = (A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0)\n"
                          "    return torch.relu(cam)\n")


def _gc(body: str = _GC_BODY) -> str:
    return _GC_HEAD + _GC_KEEP + body


def _gc_end(end: str, score: str = "model(x)[0, label]") -> str:
    """The solution with another score, or another end after `score.backward()`."""
    return _gc(_GC_FORWARD.replace("model(x)[0, label]", score) + end)


_GC = _gc()
_GC_MODEL = ("from everything_ai.models import ResNet18\n\ntorch.manual_seed(0)\n"
             "model = ResNet18().eval()     # random weights, not trained: the check needs no download\n"
             "model.requires_grad_(False)   # as in the course's ResNet-18: the weights keep no slopes\n")
_GC_SCENARIOS = [
    # a photo of 128 by 128: from layer4, a map of 4 by 4
    _GC_MODEL + "x = torch.randn(1, 3, 128, 128)\ncam = grad_cam(model, model.layer4, x, 5)\n"
    "[list(cam.shape), cam.sum().item(), cam.max().item(), int(cam.argmax())]",
    # another layer, another photo and another class: from layer3, a map of 4 by 4 for a photo of 64 by 64
    _GC_MODEL + "x = torch.randn(1, 3, 64, 64)\ncam = grad_cam(model, model.layer3, x, 4)\n[list(cam.shape), cam.sum().item(), cam.max().item()]",
    # a CNN without global average pooling, on a gray image: the maps of 14 by 14 of the digit reader's conv2 go
    # through a ReLU and max pooling to 7 by 7, and then, number by number, into a linear layer
    "from everything_ai.models import DigitCNN\n\ntorch.manual_seed(0)\ndigits = DigitCNN().eval()   # random weights\n"
    "digits.requires_grad_(False)\nx = torch.randn(1, 1, 28, 28)\ncam = grad_cam(digits, digits.conv2, x, 4)\n"
    "[list(cam.shape), cam.sum().item(), cam.max().item()]",
]
_GC_WRONG = {
    "no-relu": _gc_end("    alpha = A.grad[0].mean(dim=(1, 2))\n    return (A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0)\n"),
    "alpha-sum": _gc_end("    alpha = A.grad[0].sum(dim=(1, 2))\n    return torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0))\n"),
    "per-cell": _gc_end("    return torch.relu((A.grad[0] * A[0]).sum(dim=0))\n"),
    "keeps-batch": _gc_end("    alpha = A.grad[0].mean(dim=(1, 2))\n    return torch.relu((A * alpha.reshape(-1, 1, 1)).sum(dim=1))\n"),
    "relu-first": _gc_end("    alpha = A.grad[0].mean(dim=(1, 2))\n    return (torch.relu(A[0] * alpha.reshape(-1, 1, 1))).sum(dim=0)\n"),
    "probability": _gc_end("    alpha = A.grad[0].mean(dim=(1, 2))\n    return torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0))\n",
                           score="torch.softmax(model(x), dim=1)[0, label]"),
    "top-class": _gc_end("    alpha = A.grad[0].mean(dim=(1, 2))\n    return torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0))\n",
                         score="model(x).max()"),
    "normalized": _gc_end("    alpha = A.grad[0].mean(dim=(1, 2))\n    cam = torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0))\n"
                          "    return cam / cam.max()\n"),
    "relu-alpha": _gc_end("    alpha = torch.relu(A.grad[0].mean(dim=(1, 2)))\n    return torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0))\n"),
}


def _hook_left(got, text) -> bool:
    code = _code(text)
    return "register_forward_hook" in code and ".remove()" not in code


# A part of the maps taken after the forward pass, such as maps[0][0], with retain_grad on that part: the logit does
# not come from it, so it gets no slopes, while maps[0] itself never asked to keep them.
_SLICE_LATE = (r"^[^#\n]*\bmaps\[0\]\[0\]\.retain_grad\(\)"
               r"|^[ \t]*(\w+)[ \t]*=[ \t]*maps\[0\]\[0\][ \t]*(?:#[^\n]*)?$(?=[\s\S]*^[^#\n]*\b\1\.retain_grad\(\))")
_USES_FC = Mistake("uses-fc", matches=lambda got, text: re.search(r"\.fc\b", _code(text)) is not None,
                   en="Your function takes the weights from `fc`, the last layer of ResNet-18: that is the class activation map, and it works only for a CNN that ends with global average pooling and one linear layer, and only for its last maps. The check also tries `layer3`, and a CNN without `fc`. Find the weights from the slopes of the logit instead, for any layer: `A.grad[0].mean(dim=(1, 2))`.",
                   nl="Je functie neemt de weights uit `fc`, de laatste layer van ResNet-18: dat is de class activation map, en die werkt alleen bij een CNN dat eindigt met global average pooling en één linear layer, en alleen voor zijn laatste maps. De check probeert ook `layer3`, en een CNN zonder `fc`. Vind de weights in plaats daarvan uit de hellingen van de logit, voor elke layer: `A.grad[0].mean(dim=(1, 2))`.")


_CM_MISTAKES = [
    Mistake("weights-columns", error=_broadcast("class_map"),
            en="`maps * weights` stops: PyTorch lines a tensor of shape (K,) up with the last axis of `maps`, the columns, not with the maps, and these maps do not have K columns. Give the weights the shape (K, 1, 1) first, `weights.reshape(-1, 1, 1)`: then each weight multiplies a whole map, by broadcasting.",
            nl="`maps * weights` stopt: PyTorch zet een tensor met shape (K,) langs de laatste axis van `maps`, de kolommen, niet langs de maps, en deze maps hebben geen K kolommen. Geef de weights eerst de shape (K, 1, 1), `weights.reshape(-1, 1, 1)`: dan vermenigvuldigt elke weight een hele map, door broadcasting."),
    Mistake("no-return", error=lambda e: isinstance(e, AttributeError) and "'NoneType' object has no attribute" in str(e) and not _raised_in(e, "class_map"),
            en="The check asks for what your function gives back, and it gives back nothing, `None`: end with `return` and the map.",
            nl="De check vraagt wat je functie teruggeeft, en die geeft niets terug, `None`: eindig met `return` en de map."),
    Mistake("relu", wrong=_CM_WRONG["relu"],
            en=_S_EN + ": you keep only the positive part. That is the last step of Grad-CAM; a class activation map keeps its negative numbers, the places that speak against the class: with them, its mean plus the bias is the logit. Leave out the `relu`.",
            nl=_S_NL + ": je houdt alleen het positieve deel. Dat is de laatste stap van Grad-CAM; een class activation map houdt zijn negatieve getallen, de plekken die tegen de klasse pleiten: mét die getallen is zijn gemiddelde plus de bias de logit. Laat de `relu` weg."),
    Mistake("mean", wrong=_CM_WRONG["mean"],
            en=_S_EN + ": you take the mean of the weighted maps, so your map is K times too small. The linear layer adds up its K inputs, each times its weight: add up the weighted maps, `.sum(dim=0)`.",
            nl=_S_NL + ": je neemt het gemiddelde van de gewogen maps, dus is je map K keer te klein. De linear layer telt zijn K inputs op, elk keer zijn weight: tel de gewogen maps op, `.sum(dim=0)`."),
    Mistake("no-weights", wrong=_CM_WRONG["no-weights"],
            en=_S_EN + ": you add up the maps without their weights, so the map is the same for every class. Multiply each map by its weight first.",
            nl=_S_NL + ": je telt de maps op zonder hun weights, dus is de map voor elke klasse dezelfde. Vermenigvuldig eerst elke map met zijn weight."),
    Mistake("pooled", wrong=_CM_WRONG["pooled"],
            en=_S_EN + ": you take the mean of each map first, global average pooling, and get one number per map: what the linear layer gets, not a map. A class activation map keeps the places: weight the whole maps, and add them up.",
            nl=_S_NL + ": je neemt eerst het gemiddelde van elke map, global average pooling, en krijgt één getal per map: wat de linear layer krijgt, geen map. Een class activation map houdt de plekken: weeg de hele maps, en tel ze op."),
    Mistake("logit", wrong=_CM_WRONG["logit"],
            en=_S_EN + ": that is one number, the logit without its bias: global average pooling and the linear layer. The map comes before the mean: weight the whole maps, and add them up along axis 0 only.",
            nl=_S_NL + ": dat is één getal, de logit zonder zijn bias: global average pooling en de linear layer. De map komt vóór het gemiddelde: weeg de hele maps, en tel ze alleen langs axis 0 op."),
    Mistake("sum-all", wrong=_CM_WRONG["sum-all"],
            en=_S_EN + ": `.sum()` without an axis adds up everything, to one number. Add up along axis 0 only, the maps: `.sum(dim=0)` keeps the rows and the columns.",
            nl=_S_NL + ": `.sum()` zonder axis telt alles op, tot één getal. Tel alleen langs axis 0 op, de maps: `.sum(dim=0)` houdt de rijen en de kolommen."),
    Mistake("rows", wrong=_CM_WRONG["rows"],
            en=_S_EN + ": the weighted maps have shape (K, H, W), and you add up along axis 1, the rows. Add up along axis 0, the maps: `.sum(dim=0)`.",
            nl=_S_NL + ": de gewogen maps hebben shape (K, H, W), en je telt op langs axis 1, de rijen. Tel op langs axis 0, de maps: `.sum(dim=0)`."),
]

_GC_MISTAKES = [
    Mistake("not-recorded", error=lambda e: isinstance(e, RuntimeError) and ("can't retain_grad on Tensor that has requires_grad=False" in str(e) or "does not require grad" in str(e)),
            en="PyTorch recorded nothing on the way from the photo through `layer` to the score, so the maps get no slopes. The weights of the model keep no slopes, so the photo must: set `x.requires_grad = True` before you take it through the model. And not inside `with torch.no_grad():`, where PyTorch records nothing.",
            nl="PyTorch heeft niets opgenomen op de weg van de foto via `layer` naar de score, dus krijgen de maps geen hellingen. De weights van het model houden geen hellingen bij, dus moet de foto dat doen: zet `x.requires_grad = True` voordat je hem door het model haalt. En niet binnen `with torch.no_grad():`, waar PyTorch niets opneemt."),
    Mistake("hook-late", error=lambda e: isinstance(e, IndexError) and "list index out of range" in str(e) and _raised_in(e, "grad_cam"),
            en="`maps` is still empty: `keep` never ran. A forward hook runs while the model computes `layer`, so hang it on before you take the photo through the model: `layer.register_forward_hook(keep)` first, then `model(x)`.",
            nl="`maps` is nog leeg: `keep` is nooit gedraaid. Een forward hook draait terwijl het model `layer` berekent, dus hang hem eraan voordat je de foto door het model haalt: eerst `layer.register_forward_hook(keep)`, dan `model(x)`."),
    Mistake("slice-late", syntax=_SLICE_LATE,
            en="`A.grad` is still `None`, though you call `retain_grad` and `backward`. `retain_grad` works only on a tensor the logit was computed from: `maps[0]` itself, which the hook got during the forward pass. A part of it that you take afterwards, such as `maps[0][0]`, is a new tensor that the logit does not come from, so it gets no slopes. Call `retain_grad()` on `maps[0]`, and take photo 0 later, from `A.grad[0]` and `A[0]`.",
            nl="`A.grad` is nog `None`, hoewel je `retain_grad` en `backward` aanroept. `retain_grad` werkt alleen op een tensor waaruit de logit is berekend: `maps[0]` zelf, die de hook tijdens de forward pass kreeg. Een deel ervan dat je daarna neemt, zoals `maps[0][0]`, is een nieuwe tensor waar de logit niet uit voortkomt, dus krijgt hij geen hellingen. Roep `retain_grad()` aan op `maps[0]`, en neem foto 0 later, uit `A.grad[0]` en `A[0]`."),
    Mistake("no-grad-yet", error=lambda e: isinstance(e, (TypeError, AttributeError)) and "'NoneType'" in str(e) and _raised_in(e, "grad_cam"),
            en="`A.grad` is still `None`. Two things fill it: `A.retain_grad()` before `backward`, because `A` is not a leaf tensor but computed from the photo, and PyTorch keeps the slopes of such tensors only when you ask; and `score.backward()` itself.",
            nl="`A.grad` is nog `None`. Twee dingen vullen hem: `A.retain_grad()` vóór `backward`, omdat `A` geen leaf tensor is maar berekend uit de foto, en PyTorch de hellingen van zulke tensors alleen bewaart als je erom vraagt; en `score.backward()` zelf."),
    Mistake("one-number", error=lambda e: isinstance(e, RuntimeError) and "only for scalar outputs" in str(e),
            en="`backward` starts from one number, and you call it on more: `model(x)` gives a logit for every class, shape (1, classes). Take the logit of `label` for photo 0: `model(x)[0, label]`.",
            nl="`backward` begint bij één getal, en je roept hem aan op meer: `model(x)` geeft een logit voor elke klasse, shape (1, klassen). Neem de logit van `label` voor foto 0: `model(x)[0, label]`."),
    Mistake("batch-index", error=lambda e: isinstance(e, IndexError) and "dimension 0 with size 1" in str(e) and _raised_in(e, "grad_cam"),
            en="`model(x)` has shape (1, classes): one row, for the one photo of the batch. `model(x)[label]` asks for row `label`, which is not there. Take row 0 and then the class: `model(x)[0, label]`.",
            nl="`model(x)` heeft shape (1, klassen): één rij, voor de ene foto van de batch. `model(x)[label]` vraagt rij `label`, en die is er niet. Neem rij 0 en dan de klasse: `model(x)[0, label]`."),
    Mistake("shapes", error=_broadcast("grad_cam"),
            en="The shapes do not fit. `A` and `A.grad` have the batch first, shape (1, K, h, w). `alpha` needs one number per map, shape (K,): the mean of `A.grad[0]` over its rows and columns, `dim=(1, 2)`. Then `alpha.reshape(-1, 1, 1)` lines it up with the maps of `A[0]`, shape (K, h, w).",
            nl="De shapes passen niet. `A` en `A.grad` hebben de batch vooraan, shape (1, K, h, w). `alpha` heeft één getal per map nodig, shape (K,): het gemiddelde van `A.grad[0]` over zijn rijen en kolommen, `dim=(1, 2)`. Dan zet `alpha.reshape(-1, 1, 1)` het langs de maps van `A[0]`, shape (K, h, w)."),
    Mistake("numpy-recorded", error=lambda e: isinstance(e, RuntimeError) and "requires grad" in str(e) and "numpy" in str(e),
            en="`.numpy()` does not work on a tensor that PyTorch records, and the map is computed from `A`, which it records. Give back the tensor itself: the check and the demo take it as it is.",
            nl="`.numpy()` werkt niet op een tensor die PyTorch opneemt, en de map is berekend uit `A`, die het opneemt. Geef de tensor zelf terug: de check en de demo nemen hem zoals hij is."),
    Mistake("no-return", error=lambda e: isinstance(e, AttributeError) and "'NoneType' object has no attribute" in str(e) and not _raised_in(e, "grad_cam"),
            en="The check asks for the `.shape` of what your function gives back, and it gives back nothing, `None`: end with `return` and the map.",
            nl="De check vraagt de `.shape` van wat je functie teruggeeft, en die geeft niets terug, `None`: eindig met `return` en de map."),
    Mistake("fixed-layer", matches=lambda got, text: re.search(r"model\.layer\d", _code(text)) is not None,
            en=_S_EN + ": your function takes its maps from a layer of its own choosing, such as `model.layer4`, not from the `layer` it gets. Hang the hook on `layer`: then the same function works for any layer of any CNN.",
            nl=_S_NL + ": je functie neemt zijn maps uit een layer die hij zelf kiest, zoals `model.layer4`, niet uit de `layer` die hij krijgt. Hang de hook aan `layer`: dan werkt dezelfde functie voor elke layer van elk CNN."),
    Mistake("no-relu", wrong=_GC_WRONG["no-relu"],
            en=_S_EN + ": your map keeps its negative numbers, the places that speak against the class. Grad-CAM keeps only the positive part, the places that speak for it: `torch.relu` of the sum.",
            nl=_S_NL + ": je map houdt zijn negatieve getallen, de plekken die tegen de klasse pleiten. Grad-CAM houdt alleen het positieve deel, de plekken die ervoor pleiten: `torch.relu` van de som."),
    Mistake("alpha-sum", wrong=_GC_WRONG["alpha-sum"],
            en=_S_EN + ": you add up the slopes of each map instead of taking their mean, so every number is h × w times too large: 16 times for these maps of 4 by 4, 49 times for 7 by 7. Selvaraju and colleagues take the mean: `A.grad[0].mean(dim=(1, 2))`.",
            nl=_S_NL + ": je telt de hellingen van elke map op in plaats van hun gemiddelde te nemen, dus is elk getal h × w keer te groot: 16 keer bij deze maps van 4 bij 4, 49 keer bij 7 bij 7. Selvaraju en zijn collega's nemen het gemiddelde: `A.grad[0].mean(dim=(1, 2))`."),
    Mistake("per-cell", wrong=_GC_WRONG["per-cell"],
            en=_S_EN + ": you weight each number of a map by its own slope. Grad-CAM gives each whole map one weight, the mean of its slopes, `A.grad[0].mean(dim=(1, 2))`, as the class activation map gives each map one weight of the linear layer.",
            nl=_S_NL + ": je weegt elk getal van een map met zijn eigen helling. Grad-CAM geeft elke hele map één weight, het gemiddelde van zijn hellingen, `A.grad[0].mean(dim=(1, 2))`, zoals de class activation map elke map één weight van de linear layer geeft."),
    Mistake("keeps-batch", wrong=_GC_WRONG["keeps-batch"],
            en=_S_EN + ": the numbers are right, but your map keeps the batch: shape (1, h, w). Take the maps of photo 0 first, `A[0]`, of shape (K, h, w), and add up along axis 0.",
            nl=_S_NL + ": de getallen kloppen, maar je map houdt de batch: shape (1, h, w). Neem eerst de maps van foto 0, `A[0]`, met shape (K, h, w), en tel op langs axis 0."),
    Mistake("relu-first", wrong=_GC_WRONG["relu-first"],
            en=_S_EN + ": you keep the positive part of each weighted map, and then add them up. Grad-CAM adds them up first, so a map that speaks against the class can cancel one that speaks for it, and keeps the positive part of the sum.",
            nl=_S_NL + ": je houdt het positieve deel van elke gewogen map, en telt ze daarna op. Grad-CAM telt ze eerst op, zodat een map die tegen de klasse pleit er een kan opheffen die ervoor pleit, en houdt het positieve deel van de som."),
    Mistake("probability", wrong=_GC_WRONG["probability"],
            en=_S_EN + ": you take the slopes of the probability. A probability also rises when other classes fall, so its slopes mix them all in. Selvaraju and colleagues take the score before softmax, the logit: `model(x)[0, label]`.",
            nl=_S_NL + ": je neemt de hellingen van de kans. Een kans stijgt ook als andere klassen dalen, dus mengen zijn hellingen ze er allemaal in. Selvaraju en zijn collega's nemen de score vóór softmax, de logit: `model(x)[0, label]`."),
    Mistake("top-class", wrong=_GC_WRONG["top-class"],
            en=_S_EN + ": `model(x).max()` is the largest logit, the class the model chose, whatever `label` is. Take the logit of `label`: `model(x)[0, label]`.",
            nl=_S_NL + ": `model(x).max()` is de grootste logit, de klasse die het model koos, wat `label` ook is. Neem de logit van `label`: `model(x)[0, label]`."),
    Mistake("normalized", wrong=_GC_WRONG["normalized"],
            en=_S_EN + ": you divide the map by its largest number. Programs that draw a Grad-CAM often do, but then the largest number of every map is 1, and you no longer see that one class has much more evidence than another. Give back the map itself; the course scales it when it draws it.",
            nl=_S_NL + ": je deelt de map door zijn grootste getal. Programma's die een Grad-CAM tekenen doen dat vaak, maar dan is het grootste getal van elke map 1, en zie je niet meer dat de ene klasse veel meer bewijs heeft dan de andere. Geef de map zelf terug; de course schaalt hem als hij hem tekent."),
    Mistake("relu-alpha", wrong=_GC_WRONG["relu-alpha"],
            en=_S_EN + ": you keep only the positive weights. Grad-CAM keeps a map with a negative weight too: it speaks against the class where it is high, and can cancel a map that speaks for it. The ReLU comes once, at the end, on the sum.",
            nl=_S_NL + ": je houdt alleen de positieve weights. Grad-CAM houdt ook een map met een negatieve weight: die pleit tegen de klasse waar hij hoog is, en kan een map opheffen die ervoor pleit. De ReLU komt één keer, aan het eind, op de som."),
]

EXERCISES = {
    "class-map": ScenarioBox(
        name="class-map",
        picture=yours.arrays(call=0, show=("weights",), result="class_map(maps, weights)", shade=True),
        start=_CM_START,
        solution=_CM,
        scenarios=_CM_SCENARIOS,
        compare=_close,
        digits=5,
        save_as=("cnn", "cam"),
        first=[_DOTS],
        mistakes=_CM_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: multiply each map by its weight, and add the maps up.",
                      "nl": "Vervang de `...` door je code: vermenigvuldig elke map met zijn weight, en tel de maps op."},
            "wrong": {"en": _S_EN + ". Does each map get its own weight, and do you add up the weighted maps, keeping the rows and the columns?",
                      "nl": _S_NL + ". Krijgt elke map zijn eigen weight, en tel je de gewogen maps op, met de rijen en de kolommen erbij?"},
            "almost": {"en": "Right for the two maps of exercise 1. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for any number of maps, of any size?",
                       "nl": "Goed voor de twee maps van oefening 1. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor elk aantal maps, van elke maat?"},
            "correct": {"en": "Your `class_map` is right: each map times its weight, added up, a map with the places of the logit. It is now in your project, as `{saved}`.",
                        "nl": "Je `class_map` klopt: elke map keer zijn weight, opgeteld, een map met de plekken van de logit. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `class_map` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `class_map` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "grad-cam": ScenarioBox(
        name="grad-cam",
        picture=yours.arrays(call=0, result="cam", shade=True),
        start=_GC_START,
        solution=_GC,
        scenarios=_GC_SCENARIOS,
        compare=_close,
        digits=5,
        save_as=("cnn", "gradcam"),
        first=[_DOTS, _USES_FC],
        mistakes=_GC_MISTAKES,
        form=[Mistake("hook-left", state="almost", matches=_hook_left,
                      en="Your map is right, but the hook stays on `layer`: every later pass through the model calls `keep` again, and each call of `grad_cam` hangs on one more. Take it off after the forward pass: `handle = layer.register_forward_hook(keep)`, and later `handle.remove()`.",
                      nl="Je map klopt, maar de hook blijft aan `layer` hangen: bij elke volgende keer door het model wordt `keep` weer aangeroepen, en elke aanroep van `grad_cam` hangt er nog een bij. Haal hem eraf na de forward pass: `handle = layer.register_forward_hook(keep)`, en later `handle.remove()`.")],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: hang `keep` on `layer`, record the photo, take the logit of `label`, go back from it, and weight the maps by the mean of their slopes.",
                      "nl": "Vervang de `...` door je code: hang `keep` aan `layer`, zet het opnemen aan voor de foto, neem de logit van `label`, ga ervandaan terug, en weeg de maps met het gemiddelde van hun hellingen."},
            "wrong": {"en": _S_EN + ". Is each map weighted by the mean of its slopes, the weighted maps added up, and only the positive part kept?",
                      "nl": _S_NL + ". Weeg je elke map met het gemiddelde van zijn hellingen, tel je de gewogen maps op, en houd je alleen het positieve deel?"},
            "almost": {"en": "Right for `layer4` of the first photo. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for any layer, and for a CNN without global average pooling?",
                       "nl": "Goed voor `layer4` bij de eerste foto. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor elke layer, en voor een CNN zonder global average pooling?"},
            "correct": {"en": "Your `grad_cam` is right: the slopes of one logit give each map a weight, for any layer of any CNN. It is now in your project, as `{saved}`.",
                        "nl": "Je `grad_cam` klopt: de hellingen van één logit geven elke map een weight, voor elke layer van elk CNN. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `grad_cam` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `grad_cam` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("class-map", _CM_START, "empty", None),
    ("class-map", _CM, "correct", None),
    # a loop over the maps
    ("class-map", _cm("    cam = 0\n    for k in range(len(weights)):\n        cam = cam + weights[k] * maps[k]\n    return cam\n"), "correct", None),
    # the weights first, with permute: (H, W, K) times (K,) by broadcasting
    ("class-map", _cm("    return (maps.permute(1, 2, 0) * weights).sum(dim=2)\n"), "correct", None),
    # the weights along the columns: right by chance for the 2 maps of 2 by 2, and stops at the 3 maps of 2 by 4
    ("class-map", _cm("    return (maps * weights).sum(dim=0)\n"), "error", "weights-columns"),
    ("class-map", _cm("    cam = (maps * weights.reshape(-1, 1, 1)).sum(dim=0)\n"), "error", "no-return"),
    ("class-map", _cm("    cam = ...\n    return cam\n"), "wrong", "dots-left"),
    # right for the two maps of 2 by 2 only
    ("class-map", _cm("    if maps.shape[0] == 2:\n        return (maps * weights.reshape(-1, 1, 1)).sum(dim=0)\n    return maps.sum(dim=0)\n"), "almost", None),
    ("class-map", _cm("    return 2 * (maps * weights.reshape(-1, 1, 1)).sum(dim=0)\n"), "wrong", None),
] + [("class-map", code, "wrong", key) for key, code in _CM_WRONG.items()] + [
    ("grad-cam", _GC_START, "empty", None),
    ("grad-cam", _GC, "correct", None),
    # the slopes of A kept inside the hook, and the map made without recording
    ("grad-cam", _gc("    handle = layer.register_forward_hook(keep)\n    x.requires_grad = True\n    score = model(x)[0, label]\n    handle.remove()\n"
                     "    maps[0].retain_grad()\n    score.backward()\n    A, G = maps[0][0], maps[0].grad[0]\n"
                     "    with torch.no_grad():\n        alpha = G.mean(dim=(1, 2))\n        return torch.relu((A * alpha.reshape(-1, 1, 1)).sum(dim=0))\n"), "correct", None),
    # a copy of the photo, and a loop over the maps
    ("grad-cam", _gc("    handle = layer.register_forward_hook(keep)\n    x = x.clone()\n    x.requires_grad = True\n    score = model(x)[0, label]\n"
                     "    handle.remove()\n    A = maps[0]\n    A.retain_grad()\n    score.backward()\n    cam = 0\n"
                     "    for k in range(A.shape[1]):\n        cam = cam + A.grad[0, k].mean() * A[0, k]\n    return torch.relu(cam)\n"), "correct", None),
    # with torch.autograd.grad: the slopes of the maps without .grad at all
    ("grad-cam", _gc("    handle = layer.register_forward_hook(keep)\n    x.requires_grad = True\n    score = model(x)[0, label]\n    handle.remove()\n"
                     "    (G,) = torch.autograd.grad(score, maps[0])\n    alpha = G[0].mean(dim=(1, 2))\n"
                     "    return torch.relu((maps[0][0] * alpha.reshape(-1, 1, 1)).sum(dim=0))\n"), "correct", None),
    ("grad-cam", _gc(_GC_FORWARD.replace("    x.requires_grad = True\n", "") + "    alpha = A.grad[0].mean(dim=(1, 2))\n"
                     "    return torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0))\n"), "error", "not-recorded"),
    ("grad-cam", _gc(_GC_FORWARD.replace("    score = model(x)[0, label]\n", "    with torch.no_grad():\n        score = model(x)[0, label]\n")
                     + "    alpha = A.grad[0].mean(dim=(1, 2))\n    return torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0))\n"), "error", "not-recorded"),
    ("grad-cam", _gc("    x.requires_grad = True\n    score = model(x)[0, label]\n    handle = layer.register_forward_hook(keep)\n    handle.remove()\n"
                     "    A = maps[0]\n    A.retain_grad()\n    score.backward()\n    alpha = A.grad[0].mean(dim=(1, 2))\n"
                     "    return torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0))\n"), "error", "hook-late"),
    ("grad-cam", _gc(_GC_FORWARD.replace("    A.retain_grad()\n", "") + "    alpha = A.grad[0].mean(dim=(1, 2))\n"
                     "    return torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0))\n"), "error", "no-grad-yet"),
    ("grad-cam", _gc(_GC_FORWARD.replace("    score.backward()\n", "") + "    alpha = A.grad[0].mean(dim=(1, 2))\n"
                     "    return torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0))\n"), "error", "no-grad-yet"),
    ("grad-cam", _gc_end("    alpha = A.grad[0].mean(dim=(1, 2))\n    return torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0))\n",
                         score="model(x)"), "error", "one-number"),
    ("grad-cam", _gc_end("    alpha = A.grad[0].mean(dim=(1, 2))\n    return torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0))\n",
                         score="model(x)[label]"), "error", "batch-index"),
    # the batch left in alpha: shape (1, w), which does not fit the K maps
    ("grad-cam", _gc_end("    alpha = A.grad.mean(dim=(1, 2))\n    return torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0))\n"), "error", "shapes"),
    ("grad-cam", _gc_end("    alpha = A.grad[0].mean(dim=(1, 2))\n    return torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0)).numpy()\n"), "error", "numpy-recorded"),
    ("grad-cam", _gc_end("    alpha = A.grad[0].mean(dim=(1, 2))\n    cam = torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0))\n"), "error", "no-return"),
    ("grad-cam", _gc("    handle = model.layer4.register_forward_hook(keep)\n" + _GC_FORWARD.split("\n", 1)[1]
                     + "    alpha = A.grad[0].mean(dim=(1, 2))\n    return torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0))\n"), "wrong", "fixed-layer"),
    ("grad-cam", _gc(_GC_FORWARD.replace("    handle.remove()\n", "") + "    alpha = A.grad[0].mean(dim=(1, 2))\n"
                     "    return torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0))\n"), "almost", "hook-left"),
    ("grad-cam", _gc("    handle = layer.register_forward_hook(keep)\n    ...\n"), "wrong", "dots-left"),
    ("grad-cam", _gc_end("    alpha = A.grad[0].mean(dim=(1, 2))\n    return 2 * torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0))\n"), "wrong", None),
    # photo 0 taken from the maps before retain_grad, as in the worked example of an older version
    ("grad-cam", _gc("    handle = layer.register_forward_hook(keep)\n    x.requires_grad = True\n    score = model(x)[0, label]\n    handle.remove()\n"
                     "    A = maps[0][0]\n    A.retain_grad()\n    score.backward()\n    alpha = A.grad.mean(dim=(1, 2))\n"
                     "    return torch.relu((A * alpha.reshape(-1, 1, 1)).sum(dim=0))\n"), "error", "slice-late"),
    ("grad-cam", _gc("    handle = layer.register_forward_hook(keep)\n    x.requires_grad = True\n    score = model(x)[0, label]\n    handle.remove()\n"
                     "    maps[0][0].retain_grad()\n    score.backward()\n    alpha = maps[0][0].grad.mean(dim=(1, 2))\n"
                     "    return torch.relu((maps[0][0] * alpha.reshape(-1, 1, 1)).sum(dim=0))\n"), "error", "slice-late"),
    # the class activation map's shortcut: the weights of fc, divided by the number of places
    ("grad-cam", _gc("    handle = layer.register_forward_hook(keep)\n    model(x)\n    handle.remove()\n    A = maps[0][0]\n"
                     "    alpha = model.fc.weight[label] / (A.shape[1] * A.shape[2])\n"
                     "    return torch.relu((A * alpha.reshape(-1, 1, 1)).sum(dim=0))\n"), "wrong", "uses-fc"),
] + [("grad-cam", code, "wrong", key) for key, code in _GC_WRONG.items()] + [
    # with the starter's comments, which name model.layer4 and say what the function gives back
    ("class-map", _CM_START.replace("    ...\n", _CM_BODY), "correct", None),
    ("grad-cam", _GC_START.replace("    ...\n", _GC_BODY), "correct", None),
    ("grad-cam", _GC_START.replace("    ...\n", _GC_WRONG["no-relu"][len(_GC_HEAD + _GC_KEEP):]), "wrong", "no-relu"),
    ("grad-cam", _GC_START.replace("    ...\n", _GC_WRONG["normalized"][len(_GC_HEAD + _GC_KEEP):]), "wrong", "normalized"),
    ("grad-cam", _GC_START.replace("    ...\n", "    handle = model.layer4.register_forward_hook(keep)\n" + _GC_BODY.split("\n", 1)[1]), "wrong", "fixed-layer"),
    # the hook left on, with a remove() only in a comment
    ("grad-cam", _GC_START.replace("    ...\n", _GC_BODY.replace("    handle.remove()\n", "    # handle.remove() comes later\n")), "almost", "hook-left"),
]


# --- the demo: the learner's functions on ResNet-18 ------------------------------------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def _as_map(m, shape, grad_cam: bool = False) -> np.ndarray | None:
    """What the learner's function gave back as a NumPy array of floats, or None when it is not a finite map of that
    shape; for a Grad-CAM also when it has a negative number or is 0 everywhere, which the veil and the shares of the
    halves cannot draw."""
    if isinstance(m, torch.Tensor):
        m = m.detach().numpy()
    m = np.asarray(m, dtype=float)
    if m.shape != shape or not np.isfinite(m).all():
        return None
    if grad_cam and (m.min() < 0 or not m.max() > 0):
        return None
    return m


def demo(photo: str, layer: str):
    """The demo's result for a photo of DEMO_PHOTOS and a layer of LAYERS, worked out anew each time, in a few
    seconds: None when cam.py or gradcam.py is not in the project yet; "class_map" or "grad_cam" when the learner's
    function stops with an error or gives something of another shape (or, for grad_cam, a map with a negative number
    or of zeros); else a dict with the photo's name ("photo"),
    the layer ("layer"), the learner's Grad-CAMs of ResNet-18's three best guesses ("maps"), their names ("names")
    and probabilities ("p"), on the collage the share of each map on the cat half ("shares", else None), and from
    layer4 the largest difference between the Grad-CAM of the best guess and the positive part of the learner's
    class_map with fc's weights, divided by 49 ("same", else None)."""
    if not (_saved("cam") and _saved("gradcam")):
        return None
    run = run_code("from cnn.cam import class_map")
    if run.error is not None:
        return "class_map"
    class_map = run.names["class_map"]
    run = run_code("from cnn.gradcam import grad_cam")
    if run.error is not None:
        return "grad_cam"
    grad_cam = run.names["grad_cam"]
    model = _TRY._resnet18()
    mine = copy.deepcopy(model)   # the learner's function gets a copy: what it does to its model never reaches the course's
    names = models.imagenet_classes()
    x = input_of(photo)
    A, logits = feature_maps(model, getattr(model, layer), x)
    p = torch.softmax(logits[0], 0)
    top = [int(i) for i in p.argsort(descending=True)[:3]]
    maps = []
    try:
        for label in top:
            m = with_time_limit(lambda: _as_map(grad_cam(mine, getattr(mine, layer), x.clone(), label), tuple(A.shape[2:]), grad_cam=True), 30)
            if m is None:
                return "grad_cam"
            maps.append(m)
    except Exception:
        return "grad_cam"
    same = None
    if layer == "layer4":
        try:
            cam = with_time_limit(lambda: _as_map(class_map(A[0].clone(), model.fc.weight[top[0]].clone()), (GRID, GRID)), 10)
        except Exception:
            return "class_map"
        if cam is None:
            return "class_map"
        same = float(np.abs(np.maximum(cam, 0) / (GRID * GRID) - maps[0]).max())
    return {"photo": photo, "layer": layer, "maps": maps, "names": [names[c] for c in top], "p": [float(p[c]) for c in top],
            "shares": [cat_share(m) for m in maps] if photo == "collage" else None, "same": same}
