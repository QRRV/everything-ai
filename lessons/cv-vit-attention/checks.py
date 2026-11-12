# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-vit-attention: class_attention, the class token's attention to the patches in one layer
as a map per head; rollout, attention rollout over a list of layers (the heads averaged, the identity added for the
residual connection, the layers multiplied); and keep, the tokens a ViT gets when the patches a map ranks highest are
left out. Each is checked on made-up attention weights, so it works without DINOv2. DINOv2-small and Caltech-101 the
lesson downloads once, when the learner asks for them, as cv-pretrained-features does: the worked example, the two
play blocks and the measurement on the 100 test photos need them, the exercises do not. The pictures in *The idea*
and after the play blocks show what the course measured with them (REFERENCE, COFFEE).
tests/test_cv_vit_attention.py measures them all again when EAI_DINOV2_FILE and EAI_CALTECH_DIR are set."""
import base64
import errno
import functools
import html
import importlib.util
import io
import re
import urllib.error

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import caltech, dinov2, hub, paths
from everything_ai.check import Mistake, ScenarioBox, run_code, with_time_limit
from everything_ai.lesson import download_card
from everything_ai import yours


def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_vit_attention",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PF = _lesson_checks("cv-pretrained-features")   # DINOv2 and Caltech-101: the downloads, the photos and the probe
DOWNLOAD_LINE = _PF.DOWNLOAD_LINE
CLASSES, NAMES, TRAIN, TEST = _PF.CLASSES, _PF.NAMES, _PF.TRAIN, _PF.TEST
photos = _PF.photos

ROWS = COLS = 16          # a photo of 224 by 224 pixels in patches of 14 by 14: 16 by 16 patches
TOKENS = 1 + ROWS * COLS  # the class token and 256 patches
LAYERS, HEADS = dinov2.LAYERS, dinov2.HEADS
SIDE = 224
KS = [0, 32, 64, 96, 128, 160, 192, 224]   # the stops of the slider: patches left out, of 256
MAPS = ["top", "low", "random", "rollout"]  # which map ranks the patches to leave out
VEIL = 0.85               # how much of the page a patch's veil shows where a map is 0
CLEAR = 3                 # the patches with the 3 largest numbers of a map get no veil, and the rest one by their number
COFFEE_PHOTO = "data/photos/coffee.png"   # 600 by 400 pixels, RGB; CC0 (data/README.md)
SHOWN = ["coffee"] + CLASSES        # the photos of the play blocks: the coffee, and the first test photo of each class


# --- DINOv2 and Caltech-101 in the lesson ------------------------------------------------------------------------------

def ready() -> bool:
    """DINOv2-small, the unpacked photos of Caltech-101, and their outlines or the zip file to read them from."""
    return _PF.ready() and caltech.outlines_ready()


def dino_or_stop(button, texts: dict):
    """DINOv2-small for the lesson's cells, with Caltech-101 and its outlines unpacked. When either is not there yet,
    the cell stops with `texts["ask"]` and the button; a click downloads what is missing and unpacks the photos and
    the outlines, with the progress in the cell. While it downloads, the lesson's other cells wait. `texts` has "ask",
    "busy" ({name}, {done} and {total} in MB), "unpacking" ({done} photos), "no_net", "not_pinned", "no_room" and
    "failed" ({why}): why it did not work; and "here"."""
    import marimo as mo
    failed = None
    if not ready() and button.value:
        try:
            for name, pin in (("DINOv2-small", dinov2.FILE), ("Caltech-101", caltech.ZIP)):
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
    model = _PF._dino()
    mo.output.replace(mo.md(texts["here"]))
    return model


# --- the photos as DINOv2 sees them ------------------------------------------------------------------------------------

def path_of(key: str):
    """The file of a photo of the play blocks: the coffee, or the first test photo (photo 20) of a class."""
    return paths.ROOT / COFFEE_PHOTO if key == "coffee" else photos(key)[TEST[0]]


def _geometry(w: int, h: int) -> tuple:
    """How dinov2.prepare resizes and crops a photo of w by h pixels: the size it resizes to, and the left and top of
    the crop of 224 by 224."""
    size = (int(256 * w / h), 256) if w >= h else (256, int(256 * h / w))
    return size, int((size[0] - SIDE) / 2), int((size[1] - SIDE) / 2)


def seen(path) -> np.ndarray:
    """The photo as DINOv2 sees it, before its colors are scaled: 224 by 224 pixels, RGB, as numbers from 0 to 255."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    size, left, top = _geometry(*img.size)
    return np.array(img.resize(size, Image.BICUBIC))[top:top + SIDE, left:left + SIDE]


def outline_of(path) -> list | None:
    """The outline the makers of Caltech-101 drew around the object of a photo, as points (x, y) in the 224 by 224
    pixels DINOv2 sees; None for a photo without one, such as the coffee."""
    if path.parent.name not in caltech.CHECKED:
        return None
    from PIL import Image
    points = caltech.outlines(path.parent.name).get(path.name)
    if points is None:
        return None
    w, h = Image.open(path).size
    size, left, top = _geometry(w, h)
    return [(x * size[0] / w - left, y * size[1] / h - top) for x, y in points]


def inside(path) -> torch.Tensor | None:
    """For every patch, the part of it that lies inside the outline of the object, from 0 to 1: shape (16, 16).
    The outline is filled in at the size of the photo, made smaller and cropped as the photo is, and averaged over
    each patch of 14 by 14 pixels. None for a photo without an outline."""
    if path.parent.name not in caltech.CHECKED or path.name not in caltech.outlines(path.parent.name):
        return None
    from PIL import Image, ImageDraw
    img = Image.open(path)
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in caltech.outlines(path.parent.name)[path.name]], fill=255)
    size, left, top = _geometry(*img.size)
    small = np.array(mask.resize(size, Image.BILINEAR), dtype=float)[top:top + SIDE, left:left + SIDE] / 255
    return torch.tensor(small.reshape(ROWS, 14, COLS, 14).mean(axis=(1, 3))).float()


def share(m, inner) -> float:
    """The share of a map that lies inside the outline: every patch's number times the part of it inside, added up,
    divided by the sum of the map."""
    m = torch.as_tensor(m, dtype=torch.float64)
    return float((m * torch.as_tensor(inner, dtype=torch.float64)).sum() / m.sum())


# --- the course's own functions, from the solutions of the exercises --------------------------------------------------

_CA = "import torch\n\n\ndef class_attention(weights, rows, cols):\n    return weights[:, 0, 1:].reshape(-1, rows, cols)\n"
_RO_HEAD = "import torch\n\n\ndef rollout(layers, rows, cols):\n"
_RO_GIVEN = "    t = layers[0].shape[-1]\n    I = torch.eye(t)   # the identity matrix of t by t, as np.eye\n    R = I\n"
_RO_BODY = (_RO_GIVEN + "    for weights in layers:\n        W = 0.5 * weights.mean(dim=0) + 0.5 * I\n        R = W @ R\n"
            "    return R[0, 1:].reshape(rows, cols)\n")
_KE_HEAD = "import torch\n\n\ndef keep(m, k):\n"
_KE_BODY = ("    order = torch.sort(m.reshape(-1), descending=True).indices\n"
            "    return torch.cat([torch.tensor([0]), order[k:] + 1])\n")


@functools.cache
def course(name: str):
    """The course's class_attention, rollout or keep: the solution of its exercise."""
    code = {"class_attention": _CA, "rollout": _RO_HEAD + _RO_BODY, "keep": _KE_HEAD + _KE_BODY}[name]
    return run_code(code).names[name]


def yours_or_course(text: str | None, name: str, right: bool):
    """The learner's function from what is in its box when its check said it is right, else the course's: the
    function, and whether it is the learner's."""
    if right and text:
        ran = run_code(text)
        fn = ran.names.get(name) if ran.error is None else None
        if callable(fn):
            return fn, True
    return course(name), False


# --- DINOv2's attention ------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=16)
def _looks(dino, key: str):
    with torch.no_grad():
        return [w[0] for w in dino.looks(dinov2.prepare(path_of(key)))]


def looks(dino, key: str) -> list:
    """The attention weights of every block of DINOv2 for a photo of the play blocks: 12 tensors of shape
    (6, 257, 257), the 6 heads of a block stacked. The class token is token 0, the patch in row r and column c
    token 1 + 16 · r + c. Computed once per photo."""
    return _looks(dino, key)


def view(weights: list, layer: int, which: str, class_attention=None, rollout=None) -> torch.Tensor:
    """The map of a play block: in layer `layer` (1 to 12), head 1 to 6 ("1" to "6"), the average of the heads
    ("average"), or the rollout of layers 1 to `layer` ("rollout"); shape (16, 16)."""
    class_attention = class_attention or course("class_attention")
    rollout = rollout or course("rollout")
    if which == "rollout":
        return torch.as_tensor(rollout(weights[:layer], ROWS, COLS)).float()
    maps = torch.as_tensor(class_attention(weights[layer - 1], ROWS, COLS)).float()
    return maps.mean(dim=0) if which == "average" else maps[int(which) - 1]


@functools.lru_cache(maxsize=1)
def probe(dino):
    """The linear probe of cv-pretrained-features, trained as its worked example trains it (torch.manual_seed(0),
    SGD with a learning rate of 0.1, 100 epochs) on DINOv2's features of the first 20 photos of each class."""
    X, y, _, _ = _PF.split(_PF.features(dino), TRAIN)
    return _PF.course_train_probe()(X, y, len(CLASSES))


def ranking(which: str, top_map, roll_map, i: int) -> torch.Tensor:
    """The map whose largest numbers are left out first: the class token's attention in the last layer, the average
    of its heads ("top"); the same with its sign turned, so that the least attended go first ("low"); random
    numbers, the same for photo i every time ("random"); or the rollout of all 12 layers ("rollout")."""
    if which == "top":
        return top_map
    if which == "low":
        return -top_map
    if which == "random":
        return torch.rand(ROWS, COLS, generator=torch.Generator().manual_seed(i))
    return roll_map


def answer(dino, x, kept) -> torch.Tensor:
    """DINOv2's class token for a prepared photo x when only the tokens in `kept` go into its blocks: (384,). It
    raises ValueError when `kept` holds no class token, or a number that is no token."""
    kept = torch.as_tensor(kept, dtype=torch.long).reshape(-1)
    if not bool((kept == 0).any()) or int(kept.min()) < 0 or int(kept.max()) >= TOKENS:
        raise ValueError("the tokens to keep need the class token, 0, and numbers from 0 to 256")
    with torch.no_grad():
        out = dino.tokens(x, kept)[0]
    return out[int((kept == 0).nonzero()[0, 0])]


def percent(part: float) -> int:
    return round(100 * part)


def seed_of(key: str) -> int:
    """The number of a photo of the play blocks among the 100 test photos, for its random map: the first test photo
    of class c is number 10 · c, as in right_without."""
    return (TEST[1] - TEST[0]) * CLASSES.index(key)


def leave_one(dino, key: str, which: str, k: int, keep) -> dict:
    """The second play block: what `keep` leaves of a photo when the k patches `which` map ranks highest go, and what
    the probe makes of it. "kept": the tokens kept; "now" and "p_now": the class the probe gives and its probability;
    "all" and "p_all": the same with all patches; "stood_in": whether the course's keep stood in, because the
    learner's stopped with an error or gave no class token."""
    w = looks(dino, key)
    m = ranking(which, view(w, LAYERS, "average"), view(w, LAYERS, "rollout"), seed_of(key))
    x, p = dinov2.prepare(path_of(key)), probe(dino)
    try:
        kept = torch.as_tensor(with_time_limit(lambda: keep(m.clone(), k), 2)).reshape(-1).long()
        now = answer(dino, x, kept)
        stood_in = False
    except Exception:
        kept = course("keep")(m, k)
        now, stood_in = answer(dino, x, kept), True
    with torch.no_grad():
        p_now = torch.softmax(p(now), dim=0)
        p_all = torch.softmax(p(answer(dino, x, torch.arange(TOKENS))), dim=0)
    return {"kept": kept, "now": CLASSES[int(p_now.argmax())], "p_now": float(p_now.max()),
            "all": CLASSES[int(p_all.argmax())], "p_all": float(p_all.max()), "stood_in": stood_in}


def test_maps(dino) -> list:
    """For each of the 100 test photos: its prepared photo, its label, and the two maps the leave-out uses, the
    average of the heads in the last layer and the rollout of all layers. Computed once."""
    return _test_maps(dino)


@functools.lru_cache(maxsize=1)
def _test_maps(dino):
    out = []
    with torch.no_grad():
        for label, name in enumerate(CLASSES):
            for path in photos(name)[TEST[0]:TEST[1]]:
                x = dinov2.prepare(path)
                weights = [w[0] for w in dino.looks(x)]
                out.append((x, label, view(weights, LAYERS, "average"), view(weights, LAYERS, "rollout")))
    return out


def right_without(dino, keep, which: str, k: int, progress=None) -> int:
    """How many of the 100 test photos the probe gets right when `keep` leaves out the k patches `which` map ranks
    highest. `progress(done)` hears after every photo."""
    p = probe(dino)
    right = 0
    for i, (x, label, top_map, roll_map) in enumerate(test_maps(dino)):
        kept = with_time_limit(lambda: keep(ranking(which, top_map, roll_map, i).clone(), k), 2)
        with torch.no_grad():
            right += int(p(answer(dino, x, kept)).argmax()) == label
        if progress is not None:
            progress(i + 1)
    return right


# --- what the course measured ------------------------------------------------------------------------------------------

# Measured with the functions above and DINOv2-small from its pinned file, on the 30 photos of each of the 10 classes
# (photos), filled in by the course's fill script: "layers" is the share of the class token's attention, the average
# of its heads, that lies inside the outline of the object, in layers 1 to 12, averaged over the 300 photos; "heads"
# the same for each head of layer 12; "rollout" for the rollout of all 12 layers; "area" the part of the photo the
# object covers, which is also what a map spread evenly would give; "above" on how many of the 300 photos the average of
# layer 12 and the rollout lie more inside the outline than that. "left_out": how many of the 100 test photos the probe
# gets right when keep leaves out the k patches of KS that each map of MAPS ranks highest (k = 0: all 256 patches);
# "random_draws": the same for 5 draws of random maps (seeds i + 1000 · d for photo i and draw d), at k = 64, 128, 192.
# Why rollout spreads, over the 300 photos: "rollout_from" the share inside the outline of the rollout of the layers from
# layer 11 or 9 to the last; "no_identity" of the rollout of all 12 layers with W the average of the heads alone;
# "with_layer_1" on how many photos at least 2 of the rollout's 3 largest numbers are among the 3 largest of layer 1.
REFERENCE = {
    "layers": [0.5509, 0.5015, 0.5511, 0.5526, 0.5323, 0.5519, 0.5505, 0.5, 0.5617, 0.5836, 0.743, 0.7088],
    "heads": [0.7251, 0.7647, 0.8382, 0.387, 0.7807, 0.6858],
    "rollout": 0.5686, "area": 0.5124,
    "above": {"top": 277, "rollout": 228}, "photos": 300,
    "left_out": {
        "top": [98, 87, 84, 74, 67, 58, 40, 29],
        "low": [98, 97, 97, 96, 96, 96, 94, 85],
        "random": [98, 98, 97, 97, 97, 93, 81, 59],
        "rollout": [98, 94, 88, 84, 81, 70, 54, 42],
    },
    "random_draws": {64: [97, 97, 97, 97, 97], 128: [97, 93, 97, 97, 96], 192: [81, 81, 83, 82, 86]},
    "rollout_from": {11: 0.7156, 9: 0.6498}, "no_identity": 0.5552, "with_layer_1": 270,
}
# The coffee: in layer 12, the class token's row of each head, its attention to itself first, then to the 256 patches,
# and the rollout of all 12 layers, in ten-thousandths.
COFFEE = {
    "row": [
        [1763,   # itself, then the patches row by row
         29, 8, 7, 4, 3, 6, 14, 44, 9, 5, 2, 3, 5, 7, 10, 6, 6, 17, 2, 15, 15, 29, 71, 53, 33, 34, 14, 3, 1, 2, 18, 15,
         8, 5, 5, 24, 82, 74, 326, 54, 63, 39, 52, 37, 127, 166, 2, 5, 6, 20, 7, 26, 184, 366, 247, 179, 139, 39, 40, 37, 3, 106, 3, 2,
         10, 14, 13, 24, 311, 94, 63, 89, 78, 96, 35, 26, 23, 4, 4, 7, 13, 2, 24, 17, 46, 143, 71, 274, 145, 81, 115, 38, 10, 4, 8, 5,
         8, 6, 49, 18, 25, 81, 105, 201, 76, 39, 54, 25, 37, 5, 17, 1, 20, 7, 16, 63, 21, 10, 21, 63, 74, 26, 24, 39, 18, 4, 8, 2,
         14, 8, 4, 107, 14, 13, 22, 23, 14, 27, 26, 49, 3, 5, 5, 3, 3, 16, 17, 31, 56, 32, 11, 31, 36, 14, 14, 13, 4, 4, 8, 3,
         7, 6, 7, 32, 42, 32, 36, 59, 206, 13, 41, 20, 11, 2, 2, 3, 3, 3, 3, 23, 37, 16, 14, 44, 21, 22, 84, 25, 14, 4, 6, 2,
         24, 42, 4, 18, 25, 19, 14, 15, 14, 18, 136, 14, 3, 7, 12, 5, 13, 33, 3, 4, 5, 17, 11, 9, 5, 9, 11, 5, 2, 7, 5, 4,
         8, 13, 20, 4, 2, 2, 19, 4, 6, 4, 2, 4, 7, 6, 4, 16, 22, 7, 5, 7, 3, 4, 4, 2, 2, 6, 8, 8, 2, 4, 10, 23,
        ],
        [431,   # itself, then the patches row by row
         18, 5, 2, 1, 5, 46, 102, 207, 51, 20, 2, 2, 3, 2, 4, 3, 3, 4, 2, 121, 122, 96, 39, 33, 20, 133, 37, 18, 1, 2, 3, 4,
         3, 6, 6, 177, 29, 125, 32, 139, 62, 38, 179, 245, 151, 169, 1, 1, 9, 15, 104, 106, 35, 35, 48, 55, 48, 33, 73, 137, 11, 199, 1, 1,
         7, 9, 125, 101, 66, 57, 21, 25, 23, 42, 32, 81, 248, 6, 5, 3, 3, 2, 133, 159, 34, 38, 8, 16, 14, 23, 75, 149, 118, 5, 7, 2,
         5, 3, 136, 153, 148, 23, 7, 9, 4, 4, 45, 217, 96, 2, 10, 2, 9, 6, 75, 61, 96, 46, 167, 28, 6, 198, 187, 75, 41, 3, 4, 2,
         8, 12, 5, 196, 25, 25, 93, 34, 89, 84, 40, 155, 2, 1, 1, 2, 3, 13, 19, 91, 179, 124, 34, 24, 59, 20, 56, 20, 4, 2, 2, 1,
         4, 6, 3, 69, 40, 117, 61, 69, 97, 30, 13, 8, 8, 1, 1, 3, 1, 3, 2, 14, 22, 6, 52, 89, 62, 7, 17, 3, 6, 2, 3, 2,
         12, 10, 4, 19, 60, 26, 9, 11, 14, 7, 8, 3, 2, 4, 8, 4, 11, 11, 1, 2, 5, 9, 3, 5, 2, 3, 4, 2, 1, 4, 5, 3,
         8, 13, 9, 5, 4, 2, 5, 3, 15, 2, 1, 2, 5, 6, 3, 5, 14, 4, 3, 7, 4, 2, 1, 2, 1, 3, 9, 8, 2, 1, 5, 8,
        ],
        [209,   # itself, then the patches row by row
         2, 0, 0, 0, 1, 1, 6, 25, 3, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 5, 11, 22, 33, 14, 7, 14, 4, 0, 0, 1, 1, 0,
         0, 0, 0, 16, 14, 40, 69, 92, 63, 25, 33, 19, 61, 119, 0, 0, 1, 2, 2, 30, 82, 199, 484, 517, 326, 153, 101, 33, 1, 72, 0, 0,
         1, 7, 5, 60, 573, 357, 112, 251, 115, 427, 205, 47, 19, 1, 0, 0, 1, 1, 8, 28, 212, 468, 78, 205, 243, 177, 477, 105, 6, 1, 1, 0,
         2, 1, 12, 12, 55, 211, 97, 56, 54, 65, 233, 35, 7, 1, 2, 0, 3, 3, 7, 15, 10, 4, 31, 179, 82, 37, 22, 13, 5, 1, 1, 0,
         4, 3, 1, 55, 7, 7, 13, 4, 7, 14, 14, 23, 1, 0, 0, 0, 1, 5, 7, 18, 103, 49, 11, 20, 32, 13, 25, 16, 3, 1, 1, 0,
         1, 1, 2, 91, 27, 70, 66, 18, 132, 28, 45, 29, 19, 1, 0, 0, 0, 1, 2, 21, 34, 32, 11, 25, 13, 22, 81, 29, 18, 1, 1, 0,
         2, 3, 1, 42, 66, 24, 3, 4, 27, 23, 44, 17, 4, 1, 1, 0, 1, 3, 0, 1, 3, 7, 2, 3, 1, 10, 10, 5, 1, 1, 0, 0,
         1, 1, 2, 3, 2, 1, 3, 1, 5, 1, 1, 1, 1, 0, 0, 1, 2, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1,
        ],
        [1725,   # itself, then the patches row by row
         146, 95, 40, 39, 10, 8, 19, 34, 21, 14, 12, 25, 58, 18, 65, 48, 52, 67, 12, 15, 18, 19, 60, 31, 16, 23, 43, 6, 6, 7, 57, 60,
         46, 27, 23, 23, 64, 35, 128, 32, 40, 25, 34, 31, 74, 103, 15, 18, 26, 55, 12, 32, 59, 107, 99, 131, 103, 43, 46, 35, 5, 101, 34, 23,
         23, 22, 16, 32, 88, 95, 77, 94, 84, 84, 69, 24, 25, 19, 22, 61, 14, 6, 10, 20, 69, 139, 33, 99, 77, 77, 109, 38, 12, 11, 32, 52,
         16, 10, 19, 12, 24, 65, 49, 106, 66, 30, 58, 22, 15, 19, 34, 9, 21, 20, 13, 17, 12, 32, 21, 47, 37, 22, 21, 7, 29, 12, 14, 6,
         44, 27, 15, 40, 15, 17, 10, 39, 10, 9, 15, 37, 19, 11, 8, 8, 4, 24, 43, 31, 28, 31, 8, 26, 13, 11, 11, 52, 15, 8, 11, 6,
         10, 13, 27, 23, 21, 15, 25, 30, 117, 19, 21, 14, 13, 5, 7, 9, 7, 10, 9, 14, 16, 7, 14, 37, 28, 14, 36, 10, 11, 12, 9, 6,
         43, 65, 12, 18, 15, 49, 39, 44, 17, 15, 31, 9, 5, 13, 22, 21, 38, 54, 8, 17, 9, 74, 15, 23, 14, 11, 10, 7, 7, 15, 16, 25,
         34, 44, 46, 11, 9, 8, 21, 12, 13, 10, 9, 12, 23, 24, 17, 123, 94, 27, 19, 26, 12, 6, 6, 7, 4, 13, 34, 28, 14, 18, 78, 167,
        ],
        [1533,   # itself, then the patches row by row
         19, 6, 4, 2, 6, 12, 21, 42, 16, 12, 3, 2, 4, 4, 6, 5, 4, 7, 2, 26, 23, 28, 54, 30, 19, 29, 19, 10, 1, 1, 7, 7,
         3, 8, 6, 29, 33, 53, 98, 69, 55, 33, 51, 44, 45, 67, 1, 2, 9, 17, 12, 38, 46, 88, 162, 263, 150, 59, 61, 50, 6, 43, 2, 2,
         10, 12, 15, 39, 148, 152, 83, 159, 97, 165, 102, 44, 34, 6, 7, 4, 7, 3, 37, 25, 148, 326, 93, 212, 163, 149, 335, 66, 23, 6, 9, 3,
         8, 5, 80, 25, 41, 128, 105, 78, 59, 57, 123, 36, 47, 4, 14, 2, 13, 10, 48, 68, 30, 11, 28, 82, 57, 33, 34, 37, 42, 4, 5, 2,
         14, 15, 7, 189, 30, 18, 39, 17, 21, 40, 47, 91, 4, 3, 3, 3, 3, 16, 27, 81, 81, 28, 22, 42, 72, 21, 24, 14, 6, 2, 4, 2,
         5, 6, 6, 79, 57, 43, 60, 78, 77, 14, 28, 21, 15, 1, 2, 4, 2, 4, 5, 18, 28, 9, 40, 136, 63, 13, 46, 14, 14, 4, 4, 3,
         14, 17, 5, 26, 40, 16, 15, 21, 19, 13, 29, 10, 4, 4, 9, 4, 11, 16, 2, 4, 7, 13, 18, 11, 5, 8, 6, 4, 2, 5, 5, 3,
         8, 9, 11, 6, 4, 3, 23, 6, 16, 3, 2, 2, 5, 6, 4, 8, 12, 5, 2, 5, 3, 3, 2, 2, 1, 3, 7, 6, 2, 2, 5, 10,
        ],
        [811,   # itself, then the patches row by row
         27, 20, 12, 7, 2, 7, 22, 87, 14, 8, 3, 5, 12, 5, 24, 10, 10, 21, 2, 13, 29, 43, 33, 29, 18, 51, 32, 3, 2, 4, 23, 23,
         10, 9, 12, 45, 48, 88, 100, 98, 49, 26, 93, 49, 133, 201, 4, 4, 7, 42, 10, 66, 64, 85, 107, 149, 85, 54, 57, 54, 2, 222, 8, 4,
         16, 19, 22, 59, 106, 101, 67, 120, 85, 80, 89, 50, 37, 13, 12, 10, 12, 3, 30, 38, 88, 148, 29, 101, 92, 93, 158, 58, 25, 10, 25, 8,
         12, 6, 38, 51, 34, 77, 39, 82, 37, 25, 65, 56, 45, 11, 28, 4, 22, 21, 24, 45, 39, 28, 40, 58, 38, 43, 64, 41, 23, 8, 15, 5,
         25, 16, 7, 92, 34, 30, 47, 27, 34, 38, 35, 69, 6, 5, 6, 5, 5, 22, 22, 38, 194, 109, 24, 43, 30, 25, 29, 38, 6, 3, 11, 5,
         9, 7, 11, 150, 87, 152, 107, 50, 139, 57, 61, 44, 36, 3, 4, 8, 3, 6, 5, 45, 76, 24, 22, 52, 39, 49, 132, 40, 34, 7, 11, 2,
         26, 35, 8, 46, 112, 54, 17, 25, 34, 51, 52, 25, 9, 9, 21, 7, 29, 31, 3, 8, 9, 32, 8, 15, 7, 22, 22, 11, 6, 12, 13, 9,
         8, 29, 22, 11, 12, 4, 11, 6, 12, 6, 6, 6, 16, 13, 11, 25, 20, 7, 7, 12, 5, 3, 2, 3, 1, 5, 20, 19, 5, 8, 16, 25,
        ],
    ],
    "rollout": [
        21, 20, 14, 21, 21, 43, 20, 28, 23, 22, 20, 18, 24, 19, 25, 24, 23, 19, 15, 26, 21, 29, 18, 33, 29, 23, 30, 22, 20, 15, 20, 21,
        16, 17, 22, 23, 19, 26, 19, 23, 20, 18, 33, 20, 25, 19, 18, 19, 19, 32, 17, 25, 17, 21, 20, 25, 20, 19, 24, 22, 38, 17, 25, 22,
        20, 19, 25, 18, 22, 25, 19, 25, 20, 17, 22, 16, 25, 20, 16, 16, 24, 28, 17, 31, 25, 42, 25, 38, 33, 19, 38, 22, 25, 20, 19, 18,
        22, 23, 16, 22, 15, 37, 24, 33, 27, 26, 21, 21, 24, 17, 25, 21, 27, 29, 19, 28, 23, 42, 15, 17, 17, 22, 31, 24, 28, 19, 31, 24,
        18, 21, 17, 19, 18, 30, 28, 15, 17, 25, 24, 25, 23, 18, 25, 21, 17, 21, 20, 26, 20, 20, 17, 21, 23, 18, 22, 23, 26, 16, 23, 19,
        25, 32, 16, 22, 48, 27, 24, 36, 21, 24, 22, 20, 17, 20, 30, 24, 21, 23, 19, 18, 19, 20, 18, 31, 29, 38, 101, 21, 22, 15, 21, 18,
        24, 31, 18, 18, 21, 19, 20, 32, 29, 23, 853, 19, 17, 19, 32, 18, 20, 23, 16, 25, 19, 22, 18, 27, 25, 20, 20, 24, 18, 16, 22, 18,
        18, 23, 16, 30, 20, 38, 20, 36, 27, 19, 30, 18, 31, 17, 27, 19, 32, 37, 19, 27, 20, 29, 17, 26, 22, 24, 23, 21, 52, 25, 20, 27,
    ],
}

_JPEGS = {}


def _jpeg(key: str) -> str:
    """A photo of the play blocks as DINOv2 sees it, as a JPEG in base64, to draw inside an SVG."""
    if key not in _JPEGS:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(seen(path_of(key))).save(buf, format="JPEG", quality=88)
        _JPEGS[key] = base64.b64encode(buf.getvalue()).decode("ascii")
    return _JPEGS[key]


def coffee_heads() -> torch.Tensor:
    """The class token's attention to the patches in layer 12 of the coffee, per head: (6, 16, 16), from COFFEE."""
    return torch.tensor(COFFEE["row"], dtype=torch.float64)[:, 1:].reshape(HEADS, ROWS, COLS) / 10000


def coffee_weights() -> torch.Tensor:
    """Attention weights of shape (6, 257, 257) for the picture of exercise 1: the class token's row is that of the
    coffee in layer 12, and every other token divides its attention evenly."""
    w = torch.full((HEADS, TOKENS, TOKENS), 1 / TOKENS)
    row = torch.tensor(COFFEE["row"], dtype=torch.float32)
    w[:, 0] = row / row.sum(dim=1, keepdim=True)
    return w


def _panel(key: str, m, x0: float, y0: float, side: float, title: str = "", under: str = "", covered=None,
           outline=None) -> list:
    """One photo with a map over it, in SVG: over every patch a veil in the page's color, as opaque as VEIL where the
    map is 0 and clear at its 3 largest numbers (CLEAR), so that one patch far above the rest does not hide the
    others. With `covered`, a (16, 16) tensor of booleans, those patches are covered instead, almost all the way, and
    the others left clear. `outline` gives points (x, y) in the 224 pixels of the
    photo, drawn as a dashed line. A title above, a text under it."""
    parts = []
    if title:
        parts.append(f'<text x="{x0 + side / 2:.1f}" y="{y0 - 7:.1f}" style="font-size:12.5px;font-weight:600;fill:var(--ea-ink);text-anchor:middle">{html.escape(title)}</text>')
    parts.append(f'<image href="data:image/jpeg;base64,{_jpeg(key)}" x="{x0:.1f}" y="{y0:.1f}" width="{side:.1f}" height="{side:.1f}" preserveAspectRatio="none"/>')
    parts.append('<g shape-rendering="crispEdges">')
    if covered is not None:
        alpha = np.where(np.asarray(covered, dtype=bool), 0.93, 0.0)
    else:
        m = np.clip(np.asarray(m, dtype=float), 0, None)
        top = float(np.sort(m.ravel())[-CLEAR])
        alpha = VEIL * (1 - np.minimum(m / top, 1)) if top > 0 else np.full(m.shape, VEIL)
    cell = side / COLS
    for i in range(ROWS):
        for j in range(COLS):
            if alpha[i, j] >= 0.005:
                parts.append(f'<rect x="{x0 + j * cell:.2f}" y="{y0 + i * cell:.2f}" width="{cell:.2f}" height="{cell:.2f}" style="fill:var(--ea-page);opacity:{alpha[i, j]:.2f}"/>')
    parts.append("</g>")
    if outline:
        f = side / SIDE
        pts = " ".join(f"{x0 + x * f:.1f},{y0 + y * f:.1f}" for x, y in outline)
        parts.append(f'<polygon points="{pts}" style="fill:none;stroke:var(--ea-ink);stroke-width:1.4;stroke-dasharray:4 3;stroke-linejoin:round"/>')
    parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{side:.1f}" height="{side:.1f}" style="fill:none;stroke:var(--ea-line)"/>')
    if under:
        parts.append(f'<text x="{x0 + side / 2:.1f}" y="{y0 + side + 16:.1f}" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(under)}</text>')
    return parts


def _svg(parts: list, width: float, height: float, alt: str) -> str:
    return (f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="{html.escape(alt)}" style="width:100%;max-width:{width:.0f}px;height:auto;'
            f'font-family:var(--ea-font-ui)"><title>{html.escape(alt)}</title>' + "".join(parts) + "</svg>")


def maps_picture(maps: list, titles: list, alt: str, key: str = "coffee", per_row: int = 4, side: int = 128) -> str:
    """Maps over one photo, in rows of `per_row`, each with its title: such as the six heads of layer 12 and their
    average over the coffee."""
    gap, top = 12, 22
    rows = (len(maps) + per_row - 1) // per_row
    parts = []
    for n, (m, title) in enumerate(zip(maps, titles)):
        r, c = divmod(n, per_row)
        parts += _panel(key, m, c * (side + gap), top + r * (side + top + 6), side, title)
    width = per_row * side + (per_row - 1) * gap
    return _svg(parts, width, top + rows * (side + top + 6) - 4, alt)


def heads_picture(labels: dict) -> str:
    """The coffee in layer 12: the class token's attention in each of the 6 heads, their average, and the rollout of
    all 12 layers. `labels` has "head" ({n}), "average", "rollout" and "alt"."""
    heads = coffee_heads()
    roll = torch.tensor(COFFEE["rollout"], dtype=torch.float64).reshape(ROWS, COLS)
    maps = [heads[h] for h in range(HEADS)] + [heads.mean(dim=0), roll]
    titles = [labels["head"].format(n=h + 1) for h in range(HEADS)] + [labels["average"], labels["rollout"]]
    return maps_picture(maps, titles, labels["alt"])


def layers_picture(labels: dict, lang: str) -> str:
    """The share of the class token's attention, the average of its heads, that lies inside the outline of the
    object, in layers 1 to 12 and for the rollout of all 12 layers, over the 300 photos; a dashed line at the part of
    the photo the object covers. `labels` has "title", "x", "y", "rollout", "area" and "alt"."""
    from everything_ai import chart
    shares = [100 * s for s in REFERENCE["layers"]]
    area, roll = 100 * REFERENCE["area"], 100 * REFERENCE["rollout"]
    panel = chart.Panel((0.3, 15.2), (0, 100), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                        xticks=list(range(1, 13)) + [14.3], xtick_labels=[str(n) for n in range(1, 13)] + [labels["rollout"]],
                        yticks=[0, 25, 50, 75, 100])
    for n, s in enumerate(shares, start=1):
        panel.rect(n - 0.36, 0, n + 0.36, s, role="hidden")
    panel.rect(14.3 - 0.6, 0, 14.3 + 0.6, roll, role="hidden")
    panel.line([0.3, 15.2], [area, area], role="ink-3", width=1.4, dashed=True)
    panel.line([0.6, 1.5], [92, 92], role="ink-3", width=1.4, dashed=True)
    panel.text(1.8, 92, labels["area"], role="ink-2", anchor="start")
    svg = chart.figure([panel], alt=labels["alt"], width=470, height=260)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def look_picture(key: str, m, labels: dict) -> str:
    """One photo of a play block with a map over it, large, and the outline of its object where it has one.
    `labels` has "title" and "alt"."""
    outline = outline_of(path_of(key))
    side = 300
    return _svg(_panel(key, m, 0, 22, side, labels["title"], outline=outline), side, 22 + side + 4, labels["alt"])


def leave_picture(key: str, kept, labels: dict) -> str:
    """One photo of a play block with the patches that `kept` leaves out covered: kept holds token numbers, the
    class token 0 and the patches 1 + j. `labels` has "title" and "alt"."""
    covered = torch.ones(ROWS * COLS, dtype=torch.bool)
    kept = torch.as_tensor(kept, dtype=torch.long).reshape(-1)
    covered[kept[kept > 0] - 1] = False
    side = 300
    return _svg(_panel(key, None, 0, 22, side, labels["title"], covered=covered.reshape(ROWS, COLS)), side, 22 + side + 4, labels["alt"])


def left_out_chart(k: int, labels: dict, lang: str) -> str:
    """How many of the 100 test photos the probe gets right against the patches left out, for each map of MAPS, from
    REFERENCE, with the stop k marked, and a key under it. `labels` has "title", "x", "y", one name per map in MAPS,
    and "alt"."""
    from everything_ai import chart
    style = {"top": ("ink", False), "low": ("ink", True), "random": ("ink-3", False), "rollout": ("ink-2", True)}   # a metric: ink
    panel = chart.Panel((0, 224), (0, 100), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                        xticks=[0, 32, 64, 96, 128, 160, 192, 224], yticks=[0, 25, 50, 75, 100])
    for which in MAPS:
        role, dashed = style[which]
        ys = REFERENCE["left_out"][which]
        panel.line(KS, ys, role=role, dashed=dashed)
        panel.points([k], [ys[KS.index(k)]], role=role, r=4.5, hollow=dashed)
    svg = chart.figure([panel], alt=labels["alt"], width=420, height=260)
    key = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:0.4rem"><svg width="28" height="10" aria-hidden="true">'
        f'<line x1="1" y1="5" x2="27" y2="5" style="stroke:var(--ea-{style[w][0]});stroke-width:2.2'
        f'{";stroke-dasharray:5 4" if style[w][1] else ""}"/></svg>{html.escape(labels[w])}</span>' for w in MAPS)
    out = (svg + f'<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);display:flex;'
           f'flex-wrap:wrap;gap:0.3rem 1.2rem;margin:-0.4rem 0 0.6rem 0.5rem">{key}</div>')
    return chart.with_decimal_comma(out) if lang == "nl" else out


# --- the pictures under the checks ---------------------------------------------------------------------------------------

_YOURS_TITLE = {"en": "Your class_attention on the coffee, layer 12", "nl": "Je class_attention op de koffie, layer 12"}
_HEAD = {"en": "head {n}", "nl": "head {n}"}


def _class_picture(s) -> str:
    """Exercise 1: the learner's class_attention on the coffee's weights of layer 12 (coffee_weights): the six maps it
    gives, over the photo. Nothing when it does not give 6 maps of 16 by 16."""
    mine = s.mine
    fn = mine.names.get("class_attention") if mine is not None else None
    if not callable(fn):
        return ""
    try:
        got = with_time_limit(lambda: torch.as_tensor(fn(coffee_weights(), ROWS, COLS)).float(), 2)
    except Exception:
        return ""
    if tuple(got.shape) != (HEADS, ROWS, COLS) or not torch.isfinite(got).all():
        return ""
    alt = {"en": "The coffee six times, each with the map your function gives for one head of layer 12 over it: the clearer a patch, the more the class token takes from it.",
           "nl": "De koffie zes keer, elk met de map die je functie geeft voor één head van layer 12 eroverheen: hoe helderder een patch, hoe meer het class token ervan neemt."}
    body = maps_picture([got[h] for h in range(HEADS)], [s.word(_HEAD).format(n=h + 1) for h in range(HEADS)], s.word(alt),
                        per_row=3, side=112)
    return yours.figure(s.word(_YOURS_TITLE), body, alt=s.word(alt))


def _keep_picture(s) -> str:
    """Exercise 4: what the learner's keep leaves of the coffee with the average map of layer 12 and k = 128: the
    patches it leaves out covered."""
    mine = s.mine
    fn = mine.names.get("keep") if mine is not None else None
    if not callable(fn):
        return ""
    m = coffee_heads().mean(dim=0).float()
    try:
        kept = with_time_limit(lambda: torch.as_tensor(fn(m.clone(), 128)).reshape(-1).long(), 2)
    except Exception:
        return ""
    if kept.numel() == 0 or int(kept.min()) < 0 or int(kept.max()) > ROWS * COLS:
        return ""
    gone = ROWS * COLS - len(set((kept[kept > 0] - 1).tolist()))
    title = {"en": "Your keep(m, 128) on the coffee: {gone} patches left out", "nl": "Je keep(m, 128) op de koffie: {gone} patches weggelaten"}
    alt = {"en": "The coffee with the patches your keep leaves out covered; the map is the average of the heads in layer 12.",
           "nl": "De koffie met de patches die je keep weglaat afgedekt; de map is het gemiddelde van de heads in layer 12."}
    body = leave_picture("coffee", kept, {"title": "", "alt": s.word(alt)})
    return yours.figure(s.word(title).format(gone=gone), body, alt=s.word(alt))


# --- the exercises ---------------------------------------------------------------------------------------------------------

# The lesson is done when these are correct. "spread" is the choice question in the lesson, "what-it-shows" the open one.
REQUIRED = ["class-attention", "rollout", "spread", "keep", "what-it-shows"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$|=\s*\.\.\.|\breturn\s+\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _close(got, want) -> bool:
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    if isinstance(want, float):
        return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - want) <= 1e-4 * max(1.0, abs(want))
    return got == want



def _reshaped(e, fits) -> bool:
    """Whether e is a reshape whose numbers did not fit, and fits(p, dims, size) says why: p is rows · cols, the last two
    sizes asked for, dims all sizes asked for, size how many numbers there were."""
    found = re.search(r"shape '\[([-0-9, ]+)\]' is invalid for input of size (\d+)", str(e)) if isinstance(e, RuntimeError) else None
    if found is None:
        return False
    dims = [int(v) for v in found.group(1).split(",")]
    return len(dims) >= 2 and fits(dims[-2] * dims[-1], dims, int(found.group(2)))


_S_EN = "With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_S_NL = "Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"

# Made-up attention weights: every row a softmax of random numbers, so that it adds up to 1, as in a ViT.
_W = "import torch\n\ntorch.manual_seed({seed})\nweights = torch.softmax(torch.randn({heads}, {t}, {t}), dim=-1)\n"
_LW = ("import torch\n\ntorch.manual_seed({seed})\nlayers = [torch.softmax(torch.randn({heads}, {t}, {t}), dim=-1) for _ in range({n})]\n")

# --- exercise 1: class_attention ---------------------------------------------------------------------------------------

_CA_HEAD = "import torch\n\n\ndef class_attention(weights, rows, cols):\n"
_CA_START = (_CA_HEAD + "    # weights: the attention weights of one layer, shape (heads, T, T), with T = 1 + rows * cols\n"
             "    # token 0 is the class token; the patch in row r and column c is token 1 + cols * r + c\n"
             "    # give back shape (heads, rows, cols): for every head, how much the class token takes from each patch\n"
             "    ...\n")


def _ca(body: str) -> str:
    return _CA_HEAD + body


_CA_SCENARIOS = [
    # 2 heads, 2 by 2 patches: what head 1 of the class token gives the four patches
    _W.format(seed=0, heads=2, t=5) + "out = class_attention(weights, 2, 2)\n[list(out.shape), out[1].tolist()]",
    # 3 heads, 2 rows of 3 patches: the shape, and in the last head the patch in row 1 and column 0, token 4
    _W.format(seed=1, heads=3, t=7) + "out = class_attention(weights, 2, 3)\n[list(out.shape), out[-1, 1, 0].item(), out[0].tolist()]",
    # 6 heads, 3 rows of 2 patches: the bottom right patch of every head, token 6
    _W.format(seed=2, heads=6, t=7) + "out = class_attention(weights, 3, 2)\n[list(out.shape), out[:, 2, 1].tolist()]",
]
_CA_WRONG = {
    "column": _ca("    return weights[:, 1:, 0].reshape(-1, rows, cols)\n"),
    "one-head": _ca("    return weights[0, 0, 1:].reshape(rows, cols)\n"),
    "average": _ca("    return weights[:, 0, 1:].mean(dim=0).reshape(rows, cols)\n"),
    "last-dropped": _ca("    return weights[:, 0, :-1].reshape(-1, rows, cols)\n"),
    "cols-rows": _ca("    return weights[:, 0, 1:].reshape(-1, cols, rows)\n"),
}

# --- exercise 2: rollout -------------------------------------------------------------------------------------------------

_RO_START = (_RO_HEAD + "    # layers: the attention weights of every layer, in order, each of shape (heads, T, T)\n"
             + _RO_GIVEN + "    for weights in layers:\n"
             "        # the average of the heads, and the residual connection as attention to itself, half and half\n"
             "        W = ...\n        # this layer after the layers before it\n        R = ...\n"
             "    # the class token's row, without itself, as a map of rows by cols\n    return ...\n")


def _ro(w: str = "0.5 * weights.mean(dim=0) + 0.5 * I", r: str = "W @ R", out: str = "R[0, 1:].reshape(rows, cols)") -> str:
    return _RO_HEAD + _RO_GIVEN + f"    for weights in layers:\n        W = {w}\n        R = {r}\n    return {out}\n"


_RO_SCENARIOS = [
    # one layer of 2 heads, 2 by 2 patches
    _LW.format(seed=0, heads=2, t=5, n=1) + "rollout(layers, 2, 2).tolist()",
    # two layers of 2 heads, 2 rows of 3 patches
    _LW.format(seed=1, heads=2, t=7, n=2) + "rollout(layers, 2, 3).tolist()",
    # three layers of 3 heads, 3 rows of 2 patches; with the class token's part of itself, the row adds up to 1
    _LW.format(seed=2, heads=3, t=7, n=3) + "m = rollout(layers, 3, 2)\n[m.tolist(), round(m.sum().item(), 4)]",
]
_RO_WRONG = {
    "no-identity": _ro(w="weights.mean(dim=0)"),
    "no-half": _ro(w="weights.mean(dim=0) + I"),
    "order": _ro(r="R @ W"),
    "max": _ro(w="0.5 * weights.max(dim=0).values + 0.5 * I"),
    "last-only": _ro(r="W"),
    "column": _ro(out="R[1:, 0].reshape(rows, cols)"),
}

# --- exercise 4: keep ----------------------------------------------------------------------------------------------------

_KE_START = (_KE_HEAD + "    # m: a map of shape (rows, cols); k: how many patches to leave out\n"
             "    # give back the numbers of the tokens to keep, as one tensor: the class token, 0,\n"
             "    # and every patch except the k with the largest numbers in m (patch j is token 1 + j)\n"
             "    ...\n")


def _ke(body: str) -> str:
    return _KE_HEAD + body


_M = "import torch\n\nm = torch.tensor([[0.1, 0.5, 0.2], [0.9, 0.3, 0.0]])\n"
_KE_SCENARIOS = [
    # 2 rows of 3 patches; leave out the 2 with the largest numbers, 0.9 and 0.5: patches 3 and 1, tokens 4 and 2
    _M + "sorted(int(t) for t in keep(m, 2))",
    # leave out none, and all 6
    _M + "[sorted(int(t) for t in keep(m, 0)), sorted(int(t) for t in keep(m, 6))]",
    # a map of 4 by 4 with the numbers in another order, 5 left out
    "import torch\n\ntorch.manual_seed(3)\nm = torch.rand(4, 4)\nsorted(int(t) for t in keep(m, 5))",
]
_KE_WRONG = {
    "no-shift": _ke("    order = torch.sort(m.reshape(-1), descending=True).indices\n    return torch.cat([torch.tensor([0]), order[k:]])\n"),
    "no-class": _ke("    order = torch.sort(m.reshape(-1), descending=True).indices\n    return order[k:] + 1\n"),
    "kept-top": _ke("    order = torch.sort(m.reshape(-1), descending=True).indices\n    return torch.cat([torch.tensor([0]), order[:k] + 1])\n"),
    "ascending": _ke("    order = torch.sort(m.reshape(-1)).indices\n    return torch.cat([torch.tensor([0]), order[k:] + 1])\n"),
}

EXERCISES = {
    "class-attention": ScenarioBox(
        name="class-attention",
        picture=_class_picture,
        start=_CA_START,
        solution=_CA,
        scenarios=_CA_SCENARIOS,
        compare=_close,
        digits=4,
        first=[_DOTS],
        mistakes=[
            Mistake("no-return", error=lambda e: isinstance(e, (AttributeError, TypeError)) and "NoneType" in str(e),
                    en="The check asks for the shape and the numbers of what your function gives back, and it gives back nothing, `None`: end with `return` and the maps.",
                    nl="De check vraagt de shape en de getallen van wat je functie teruggeeft, en die geeft niets terug, `None`: eindig met `return` en de maps."),
            Mistake("with-class", error=lambda e: _reshaped(e, lambda p, dims, size: size % (p + 1) == 0 and size % p != 0),
                    en="Your numbers do not fit in `rows` by `cols` per head. A row of the weights has T = 1 + rows · cols numbers, and the first is how much the class token takes from itself: leave that one out, with `1:` on the last axis, and `rows * cols` are left.",
                    nl="Je getallen passen niet in `rows` bij `cols` per head. Een rij van de weights heeft T = 1 + rows · cols getallen, en de eerste is hoeveel het class token van zichzelf neemt: laat die weg, met `1:` op de laatste axis, en er blijven `rows * cols` over."),
            Mistake("no-heads", error=lambda e: _reshaped(e, lambda p, dims, size: len(dims) == 2 and size % p == 0 and size > p),
                    en="Your numbers do not fit in one map of `rows` by `cols`: there is a map for every head. Keep the axis of the heads in front, `reshape(-1, rows, cols)`, and give back shape `(heads, rows, cols)`.",
                    nl="Je getallen passen niet in één map van `rows` bij `cols`: er is een map voor elke head. Houd de axis van de heads vooraan, `reshape(-1, rows, cols)`, en geef shape `(heads, rows, cols)` terug."),
            Mistake("column", wrong=_CA_WRONG["column"],
                    en=_S_EN + ": you take column 0, how much every patch takes from the class token. Row 0 is the class token's own row: how it divides its attention over all tokens, `weights[:, 0, 1:]`.",
                    nl=_S_NL + ": je neemt kolom 0, hoeveel elke patch van het class token neemt. Rij 0 is de eigen rij van het class token: hoe het zijn attention over alle tokens verdeelt, `weights[:, 0, 1:]`."),
            Mistake("one-head", wrong=_CA_WRONG["one-head"],
                    en=_S_EN + ": you give the map of head 0 alone. Keep the axis of the heads, `weights[:, 0, 1:]`, and give back a map per head: shape `(heads, rows, cols)`.",
                    nl=_S_NL + ": je geeft alleen de map van head 0. Houd de axis van de heads, `weights[:, 0, 1:]`, en geef een map per head terug: shape `(heads, rows, cols)`."),
            Mistake("average", wrong=_CA_WRONG["average"],
                    en=_S_EN + ": you give the average of the heads. That map comes later, from the maps of the heads: give back each head's own map, shape `(heads, rows, cols)`.",
                    nl=_S_NL + ": je geeft het gemiddelde van de heads. Die map komt later, uit de maps van de heads: geef de eigen map van elke head terug, shape `(heads, rows, cols)`."),
            Mistake("last-dropped", wrong=_CA_WRONG["last-dropped"],
                    en=_S_EN + ": you leave out the last token instead of the first. The class token is token 0, and the patches are tokens 1 to T − 1: `1:` on the last axis.",
                    nl=_S_NL + ": je laat het laatste token weg in plaats van het eerste. Het class token is token 0, en de patches zijn de tokens 1 tot en met T − 1: `1:` op de laatste axis."),
            Mistake("cols-rows", wrong=_CA_WRONG["cols-rows"],
                    en=_S_EN + ": your map has `cols` rows and `rows` columns. The patches come row by row, so a row of the photo is `cols` patches long: `reshape(-1, rows, cols)`.",
                    nl=_S_NL + ": je map heeft `cols` rijen en `rows` kolommen. De patches komen rij voor rij, dus een rij van de foto is `cols` patches lang: `reshape(-1, rows, cols)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: take the class token's row of every head without its first number, and give it the shape of the photo.",
                      "nl": "Vervang de `...` door je code: neem de rij van het class token in elke head zonder zijn eerste getal, en geef die de shape van de foto."},
            "wrong": {"en": _S_EN + ". Row 0 of every head, without column 0, as `rows` by `cols`: step 2 of the recipe?",
                      "nl": _S_NL + ". Rij 0 van elke head, zonder kolom 0, als `rows` bij `cols`: stap 2 van het recept?"},
            "almost": {"en": "Right for 2 by 2 patches. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function use `rows` and `cols`, and no fixed numbers?",
                       "nl": "Goed voor 2 bij 2 patches. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Gebruikt je functie `rows` en `cols`, en geen vaste getallen?"},
            "correct": {"en": "Your `class_attention` is right: a map per head of where the class token looks. Under this message it reads the coffee's weights of layer 12.",
                        "nl": "Je `class_attention` klopt: een map per head van waar het class token naar kijkt. Onder dit bericht leest hij de weights van de koffie in layer 12."},
        },
    ),
    "rollout": ScenarioBox(
        name="rollout",
        picture=yours.arrays(call=1, result="rollout(layers, 2, 3)", shade=True),
        start=_RO_START,
        solution=_RO_HEAD + _RO_BODY,
        scenarios=_RO_SCENARIOS,
        compare=_close,
        digits=4,
        first=[_DOTS],
        mistakes=[
            Mistake("no-return", error=lambda e: isinstance(e, (AttributeError, TypeError)) and "NoneType" in str(e),
                    en="The check asks for the numbers of what your function gives back, and it gives back nothing, `None`: end with `return` and the map.",
                    nl="De check vraagt de getallen van wat je functie teruggeeft, en die geeft niets terug, `None`: eindig met `return` en de map."),
            Mistake("with-class", error=lambda e: _reshaped(e, lambda p, dims, size: size == p + 1),
                    en="Your numbers do not fit in `rows` by `cols`. Row 0 of `R` has T = 1 + rows · cols numbers, and the first is the class token's part of itself: leave that one out, `R[0, 1:]`.",
                    nl="Je getallen passen niet in `rows` bij `cols`. Rij 0 van `R` heeft T = 1 + rows · cols getallen, en de eerste is het deel van het class token uit zichzelf: laat die weg, `R[0, 1:]`."),
            Mistake("no-mean", error=lambda e: _reshaped(e, lambda p, dims, size: size == p * (p + 1)),
                    en="Your `W` still has an axis for the heads, so `R` gets one too, and row 0 holds too many numbers. Average the heads first, `weights.mean(dim=0)`: then `W` has shape (T, T).",
                    nl="Je `W` heeft nog een axis voor de heads, dus `R` krijgt er ook een, en rij 0 heeft te veel getallen. Middel eerst de heads, `weights.mean(dim=0)`: dan heeft `W` shape (T, T)."),
            Mistake("heads-axis", error=lambda e: isinstance(e, RuntimeError) and "must match the size of tensor" in str(e),
                    en="Your average has another shape than `I`. The heads are the first axis of `weights`, axis 0: their average, `weights.mean(dim=0)`, has shape (T, T).",
                    nl="Je gemiddelde heeft een andere shape dan `I`. De heads zijn de eerste axis van `weights`, axis 0: hun gemiddelde, `weights.mean(dim=0)`, heeft shape (T, T)."),
            Mistake("no-identity", wrong=_RO_WRONG["no-identity"],
                    en=_S_EN + ": you multiply the attention alone. Every block adds what attention brings to the token itself, the residual connection: half the identity matrix, `0.5 * I`, beside half the average of the heads.",
                    nl=_S_NL + ": je vermenigvuldigt alleen de attention. Elk block telt de uitkomst van attention op bij het token zelf, de residual connection: de helft van de eenheidsmatrix, `0.5 * I`, naast de helft van het gemiddelde van de heads."),
            Mistake("no-half", wrong=_RO_WRONG["no-half"],
                    en=_S_EN + ": a row of your `W` adds up to 2, so every layer doubles the numbers. Take half of each, `0.5 * ... + 0.5 * I`: then a row adds up to 1 again, a share of where a token's numbers come from.",
                    nl=_S_NL + ": een rij van je `W` telt op tot 2, dus elke layer verdubbelt de getallen. Neem van elk de helft, `0.5 * ... + 0.5 * I`: dan telt een rij weer op tot 1, een deel van waar de getallen van een token vandaan komen."),
            Mistake("order", wrong=_RO_WRONG["order"],
                    en=_S_EN + ": you multiply the other way round, `R @ W`. Row 0 of `W @ R` says where the class token's numbers come from: from the tokens of the layer before, `W`, which got theirs from the layers before that, `R`. With `R @ W` the first layer counts last.",
                    nl=_S_NL + ": je vermenigvuldigt andersom, `R @ W`. Rij 0 van `W @ R` zegt waar de getallen van het class token vandaan komen: van de tokens van de layer ervoor, `W`, die de hunne kregen van de layers daarvoor, `R`. Met `R @ W` telt de eerste layer als laatste."),
            Mistake("max", wrong=_RO_WRONG["max"],
                    en=_S_EN + ": you take the largest of the heads, not their average. Then a row no longer adds up to 1. The recipe averages the heads: `weights.mean(dim=0)`.",
                    nl=_S_NL + ": je neemt de grootste van de heads, niet hun gemiddelde. Dan telt een rij niet meer op tot 1. Het recept middelt de heads: `weights.mean(dim=0)`."),
            Mistake("last-only", wrong=_RO_WRONG["last-only"],
                    en=_S_EN + ": `R = W` forgets the layers before: what is left is the last layer alone. Multiply: `R = W @ R`.",
                    nl=_S_NL + ": `R = W` vergeet de layers ervoor: wat overblijft is alleen de laatste layer. Vermenigvuldig: `R = W @ R`."),
            Mistake("column", wrong=_RO_WRONG["column"],
                    en=_S_EN + ": you take column 0 of `R`, how much of every patch comes from the class token. The map is row 0, where the class token's numbers come from: `R[0, 1:]`.",
                    nl=_S_NL + ": je neemt kolom 0 van `R`, hoeveel van elke patch uit het class token komt. De map is rij 0, waar de getallen van het class token vandaan komen: `R[0, 1:]`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...` with your code: `W` from the average of the heads and the identity matrix, `R` this layer after the ones before, and at the end the class token's row as a map.",
                      "nl": "Vervang de drie `...` door je code: `W` uit het gemiddelde van de heads en de eenheidsmatrix, `R` deze layer na de layers ervoor, en aan het eind de rij van het class token als map."},
            "wrong": {"en": _S_EN + ". Half the average of the heads plus half the identity, `W @ R` for every layer, and at the end `R[0, 1:]` as `rows` by `cols`: step 3 of the recipe?",
                      "nl": _S_NL + ". De helft van het gemiddelde van de heads plus de helft van de eenheidsmatrix, `W @ R` voor elke layer, en aan het eind `R[0, 1:]` als `rows` bij `cols`: stap 3 van het recept?"},
            "almost": {"en": "Right for one layer. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function go through every layer of `layers`, and use `rows` and `cols`?",
                       "nl": "Goed voor één layer. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Gaat je functie door elke layer van `layers`, en gebruikt hij `rows` en `cols`?"},
            "correct": {"en": "Your `rollout` is right: it follows the class token back through every layer to the patches. Under this message: your map of the second case, two layers of 2 heads.",
                        "nl": "Je `rollout` klopt: hij volgt het class token terug door elke layer naar de patches. Onder dit bericht: je map van het tweede geval, twee layers van 2 heads."},
        },
    ),
    "keep": ScenarioBox(
        name="keep",
        picture=_keep_picture,
        start=_KE_START,
        solution=_KE_HEAD + _KE_BODY,
        scenarios=_KE_SCENARIOS,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("no-return", error=lambda e: isinstance(e, TypeError) and "NoneType" in str(e),
                    en="The check asks for the tokens your function gives back, and it gives back nothing, `None`: end with `return` and the tokens.",
                    nl="De check vraagt de tokens die je functie teruggeeft, en die geeft niets terug, `None`: eindig met `return` en de tokens."),
            Mistake("rows-apart", error=lambda e: isinstance(e, RuntimeError) and "same number of dimensions" in str(e),
                    en="`torch.sort` of a map of `rows` by `cols` sorts every row apart, and gives places of that shape. Make the map one row first, `m.reshape(-1)`: then patch j is number j in it, as in the token numbers.",
                    nl="`torch.sort` van een map van `rows` bij `cols` sorteert elke rij apart, en geeft plekken met die shape. Maak de map eerst één rij, `m.reshape(-1)`: dan is patch j daarin nummer j, zoals in de nummers van de tokens."),
            Mistake("no-shift", wrong=_KE_WRONG["no-shift"],
                    en=_S_EN + ": you give the patch numbers themselves. Token 0 is the class token, so patch j is token 1 + j: add 1 to the patches you keep.",
                    nl=_S_NL + ": je geeft de nummers van de patches zelf. Token 0 is het class token, dus patch j is token 1 + j: tel 1 op bij de patches die je houdt."),
            Mistake("no-class", wrong=_KE_WRONG["no-class"],
                    en=_S_EN + ": the class token is missing. DINOv2's features are what the class token holds after the last block, so it always stays: put `torch.tensor([0])` in front, with `torch.cat`.",
                    nl=_S_NL + ": het class token ontbreekt. De features van DINOv2 zijn wat het class token na het laatste block bevat, dus het blijft altijd: zet `torch.tensor([0])` ervoor, met `torch.cat`."),
            Mistake("kept-top", wrong=_KE_WRONG["kept-top"],
                    en=_S_EN + ": you keep the k patches with the largest numbers, and leave out the rest. It is the other way round: those k go, the rest stays, `order[k:]`.",
                    nl=_S_NL + ": je houdt de k patches met de grootste getallen, en laat de rest weg. Het is andersom: die k gaan weg, de rest blijft, `order[k:]`."),
            Mistake("ascending", wrong=_KE_WRONG["ascending"],
                    en=_S_EN + ": you leave out the k patches with the smallest numbers. `torch.sort` sorts from small to large; with `descending=True` the largest come first.",
                    nl=_S_NL + ": je laat de k patches met de kleinste getallen weg. `torch.sort` sorteert van klein naar groot; met `descending=True` komen de grootste eerst."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: sort the patches from the largest number in `m` to the smallest, skip the first k, and put the class token in front.",
                      "nl": "Vervang de `...` door je code: sorteer de patches van het grootste getal in `m` naar het kleinste, sla de eerste k over, en zet het class token ervoor."},
            "wrong": {"en": _S_EN + ". The class token, 0, and the patches after the first k of `torch.sort(m.reshape(-1), descending=True).indices`, each plus 1: step 4 of the recipe?",
                      "nl": _S_NL + ". Het class token, 0, en de patches na de eerste k van `torch.sort(m.reshape(-1), descending=True).indices`, elk plus 1: stap 4 van het recept?"},
            "almost": {"en": "Right for 2 left out. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function use `k`, also when it is 0, or every patch?",
                       "nl": "Goed voor 2 weggelaten. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Gebruikt je functie `k`, ook als die 0 is, of elke patch?"},
            "correct": {"en": "Your `keep` is right: the class token and every patch but the k a map ranks highest. Under this message it leaves out half of the coffee's patches, those the class token looks at most in layer 12.",
                        "nl": "Je `keep` klopt: het class token en elke patch behalve de k die een map het hoogst zet. Onder dit bericht laat hij de helft van de patches van de koffie weg, die waar het class token in layer 12 het meest naar kijkt."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("class-attention", _CA_START, "empty", None),
    ("class-attention", _CA, "correct", None),
    # the number of heads from the shape, and a loop over the heads: other code, the same maps
    ("class-attention", _ca("    return weights[:, 0, 1:].reshape(weights.shape[0], rows, cols)\n"), "correct", None),
    ("class-attention", _ca("    maps = []\n    for h in range(len(weights)):\n        maps.append(weights[h, 0, 1:].reshape(1, rows, cols))\n"
                            "    return torch.cat(maps)\n"), "correct", None),
    ("class-attention", _ca("    weights[:, 0, 1:].reshape(-1, rows, cols)\n"), "error", "no-return"),
    ("class-attention", _ca("    return weights[:, 0].reshape(-1, rows, cols)\n"), "error", "with-class"),
    ("class-attention", _ca("    return weights[:, 0, 1:].reshape(rows, cols)\n"), "error", "no-heads"),
    ("class-attention", _ca("    maps = weights[:, 0, 1:]\n    return ...\n"), "wrong", "dots-left"),
    # only the first two heads: right for the first case, which has two
    ("class-attention", _ca("    return weights[:2, 0, 1:].reshape(-1, rows, cols)\n"), "almost", None),
] + [("class-attention", code, "wrong", key) for key, code in _CA_WRONG.items()] + [
    ("rollout", _RO_START, "empty", None),
    ("rollout", _RO_HEAD + _RO_BODY, "correct", None),
    ("rollout", _ro(w="(weights.mean(dim=0) + I) / 2", r="torch.matmul(W, R)"), "correct", None),
    ("rollout", _ro(out="R[0][1:].reshape(rows, cols)"), "correct", None),
    ("rollout", _ro(out="R[0].reshape(rows, cols)"), "error", "with-class"),
    ("rollout", _ro(w="0.5 * weights + 0.5 * I"), "error", "no-mean"),
    ("rollout", _ro(w="0.5 * weights.mean(dim=1) + 0.5 * I"), "error", "heads-axis"),
    ("rollout", _ro(out="R[0, 1:].reshape(rows, cols)").replace("    return R[0, 1:].reshape(rows, cols)\n", "    R[0, 1:].reshape(rows, cols)\n"), "error", "no-return"),
    ("rollout", _ro(r="..."), "wrong", "dots-left"),
    # only the first layer: right for the first case, which has one
    ("rollout", _ro().replace("for weights in layers:", "for weights in layers[:1]:"), "almost", None),
] + [("rollout", code, "wrong", key) for key, code in _RO_WRONG.items()] + [
    ("keep", _KE_START, "empty", None),
    ("keep", _KE_HEAD + _KE_BODY, "correct", None),
    # a list instead of a tensor, and the class token at the end: the same tokens
    ("keep", _ke("    order = torch.sort(m.reshape(-1), descending=True).indices\n    return [0] + [int(j) + 1 for j in order[k:]]\n"), "correct", None),
    ("keep", _ke("    order = torch.sort(m.reshape(-1), descending=True).indices\n    return torch.cat([order[k:] + 1, torch.tensor([0])])\n"), "correct", None),
    ("keep", _ke("    order = torch.sort(m, descending=True).indices\n    return torch.cat([torch.tensor([0]), order[k:] + 1])\n"), "error", "rows-apart"),
    ("keep", _ke("    order = torch.sort(m.reshape(-1), descending=True).indices\n    torch.cat([torch.tensor([0]), order[k:] + 1])\n"), "error", "no-return"),
    ("keep", _ke("    order = ...\n    return torch.cat([torch.tensor([0]), order[k:] + 1])\n"), "wrong", "dots-left"),
    # always 2 left out: right for the first case
    ("keep", _ke("    order = torch.sort(m.reshape(-1), descending=True).indices\n    return torch.cat([torch.tensor([0]), order[2:] + 1])\n"), "almost", None),
] + [("keep", code, "wrong", key) for key, code in _KE_WRONG.items()]
