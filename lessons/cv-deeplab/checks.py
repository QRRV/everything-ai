# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-deeplab: field, how many pixels one place of a stack of layers sees; ASPP, atrous spatial
pyramid pooling, 3×3 convolutions with several dilations side by side on one map; and DeepLab, the course's FCN with
stride 16 whose last two poolings are gone and whose layers after them are dilated, with an ASPP at the end. ASPP and
DeepLab are saved in the learner's project (work/projects/cnn/aspp.py and deeplab.py) when right. The demo trains the
learner's DeepLab on the class maps of their class_map of cv-fcn, and measures it with their kind_iou. And the
pictures: the taps of a dilated convolution, and for the slider, the pixels one place of two stacked convolutions
sees. Everything is computed from the course's own scenes (scenes.py): no download."""
import functools
import html
import importlib.util
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn

from everything_ai import netview, paths, projects, scenes, yours
from everything_ai.check import ClassBox, Mistake, ScenarioBox, Timeout, run_code, with_time_limit


def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_deeplab",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The scenes, their class maps, the training of cv-fcn, its FCN, its upsampler and its pictures of class maps
_FCN = _lesson_checks("cv-fcn")

# The demo makes the class maps with the learner's class_map of cv-fcn and measures with their kind_iou; their DeepLab
# builds on their upsampler of cv-fcn and their ASPP of this lesson. Where one is missing or broken, the course's
# version stands in, and the lesson says so.
BUILDS_ON = (("cnn", "classes", "cv-fcn", "class-map"), ("cnn", "kind_iou", "cv-fcn", "kind-iou"),
             ("cnn", "upsample", "cv-fcn", "upsampler"), ("cnn", "aspp", "cv-deeplab", "aspp"))
projects.builds_on(*BUILDS_ON)

SEED = _FCN.SEED              # the scene of the worked example: scenes.scene(18), as in cv-fcn
KINDS, CLASSES = _FCN.KINDS, _FCN.CLASSES
N_TRAIN, N_TEST, TRAIN, TEST = _FCN.N_TRAIN, _FCN.N_TEST, _FCN.TRAIN, _FCN.TEST
EPOCHS, LR, BATCH = _FCN.EPOCHS, _FCN.LR, _FCN.BATCH   # trained exactly as the course's FCN
SHOWN = _FCN.SHOWN            # the test scenes of the rewind
STRIDE = 4                    # the output stride of the course's DeepLab: two poolings, as the FCN with stride 4
RATES = (2, 4, 8)             # the dilations of the 3×3 branches of the course's ASPP, next to its 1×1 branch
TRAIN_LIMIT = 1200            # seconds the demo gives the training: the note says twenty minutes

# The networks the course compares, each trained with the seeds 0 to 4: the layers of the course's FCN with stride 16
# with its last two poolings gone, and then: whether the layers after them are dilated (conv4 by 2, the end by 4),
# and at the end an ASPP with these rates, or one 3×3 convolution (None).
VARIANTS = {
    "deeplab": (True, RATES),
    "same-rates": (True, (1, 1, 1)),
    "dilated": (True, None),
    "plain": (False, None),
}


# --- the course's versions ------------------------------------------------------------------------------------------

def field(layers) -> int:
    """How many pixels one place after a stack of layers sees, along one side, as the learner's: every layer is
    (kernel, stride, dilation); a layer with kernel k and dilation d adds (k - 1) * d times the step between neighbors
    of its input, and its stride multiplies that step."""
    seen, step = 1, 1
    for k, stride, d in layers:
        seen += (k - 1) * d * step
        step *= stride
    return seen


class ASPP(nn.Module):
    """Atrous spatial pyramid pooling, as the learner's: on one map, a 1×1 convolution and a 3×3 convolution for every
    dilation in `rates`, side by side, each followed by ReLU; their maps stacked along the channels, and a 1×1
    convolution with ReLU that makes them `c` channels again."""

    def __init__(self, c_in, c, rates):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv2d(c_in, c, 1)] +
                                      [nn.Conv2d(c_in, c, 3, padding=r, dilation=r) for r in rates])
        self.project = nn.Conv2d(c * (len(rates) + 1), c, 1)

    def forward(self, x):
        return torch.relu(self.project(torch.cat([torch.relu(b(x)) for b in self.branches], dim=1)))


class _ConvEnd(nn.Module):
    """The end of the networks without ASPP: one 3×3 convolution with ReLU, dilated by `d`."""

    def __init__(self, c, d):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=d, dilation=d)

    def forward(self, x):
        return torch.relu(self.conv(x))


class DeepLab(nn.Module):
    """The course's DeepLab, as the learner's: the layers of the course's FCN with stride 16 (3×3 convolutions with 16,
    32, 64 and 64 channels, each with ReLU), with a pooling of 2 after the first two only. conv4 would have come after
    the third pooling, so it is dilated by 2; the end, after the fourth, by 4: an ASPP with RATES around that 4. Then
    a 1×1 convolution to a score for every class at every place, and the upsampler of cv-fcn with stride 4.
    `dilate=False` keeps every dilation at 1, `rates=None` ends with one 3×3 convolution instead of an ASPP."""

    def __init__(self, rates=RATES, dilate: bool = True, upsampler_fn=None, aspp=ASPP):
        super().__init__()
        d = 2 if dilate else 1
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=d, dilation=d)
        self.pool = nn.MaxPool2d(2)
        self.aspp = aspp(64, 64, rates) if rates else _ConvEnd(64, 2 * d if dilate else 1)
        self.score = nn.Conv2d(64, CLASSES, 1)
        self.up = (upsampler_fn or _FCN.upsampler)(CLASSES, STRIDE)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        return self.up(self.score(self.aspp(x)))


# --- training and measuring -----------------------------------------------------------------------------------------

def make(variant: str) -> nn.Module:
    dilate, rates = VARIANTS[variant]
    return DeepLab(rates, dilate)


def train_course(variant: str = "deeplab", seed: int = 0, heard=None) -> tuple:
    """A network of the course, trained as the course's FCN is, with `seed` on one thread, as a check runs: (model,
    history)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(seed)
        model = make(variant)
        _, X, C = _FCN.scene_set("train")
        _, Xt, Ct = _FCN.scene_set("test")
        history = _FCN.train(model, (X, C), (Xt, Ct), seed, EPOCHS, heard)
    finally:
        torch.set_num_threads(threads)
    return model, history


def measure_variant(variant: str, seed: int) -> dict:
    """Train one network and measure what TRAINED holds for it: on the test scenes the share of pixels right, the IoU
    of every kind and the loss; with seed 0, the class map it gives the scene of the worked example, and for the
    course's DeepLab its loss curves."""
    model, history = train_course(variant, seed)
    _, Xt, Ct = _FCN.scene_set("test")
    pred = _FCN.outputs(model, Xt).argmax(1)
    acc, ious = _FCN.scores(pred, Ct)
    r = {"acc": round(acc, 4), "ious": [round(v, 4) for v in ious], "val": round(history[-1]["val_loss"], 4)}
    if seed == 0:
        r["map"] = _FCN.pack(pred[TEST.index(SEED)])
        if variant == "deeplab":
            r["curves"] = {"train": [round(h["train_loss"], 4) for h in history],
                           "val": [round(h["val_loss"], 4) for h in history]}
    return r


def measure_trained(variants=None, seeds=range(5)) -> dict:
    """Everything TRAINED holds, measured again: every variant with every seed. Takes three to six minutes per
    training on one thread, about an hour and a half in all."""
    out = {}
    for variant in variants or VARIANTS:
        runs = [measure_variant(variant, seed) for seed in seeds]
        out[variant] = {key: [r[key] for r in runs] for key in ("acc", "ious", "val")}
        for key in ("map", "curves"):
            if key in runs[0]:
                out[variant][key] = runs[0][key]
    return out


# What the course's networks did, measured by `measure_trained()` with the code above: for each, over the seeds 0 to 4,
# the share of the pixels of the test scenes right, the IoU of every kind and the loss on the test scenes; with seed 0
# the class map it gives the scene of the worked example, packed, and for the course's DeepLab its loss curves.
TRAINED = {
    "deeplab": {
        "acc": [0.9883, 0.9898, 0.9889, 0.9898, 0.9879],
        "ious": [[0.9134, 0.8102, 0.8153], [0.9283, 0.8346, 0.8352], [0.9202, 0.8186, 0.8221], [0.9273, 0.8316, 0.835], [0.912, 0.7963, 0.8122]],
        "val": [0.0341, 0.0297, 0.0326, 0.0305, 0.0361],
        "map": "eNrt1T0OgjAUB/AiYTEmnVw8gmcwgSMwWAaP4CmId8DB3YGegiNxgCa1HySS8D4GQhzsf4Dhl/faRwkIkZKSkpKyLnvdMq4151o/WtrxHufJ2x87ZBfGLe056wPp0vsJd0t7HvyGehm8CdorqP2sXlVA+5l3FdDe+2vyejFcdGdP/VbqinvjXQHto/cqeIX7vXNeg/13OhQv3Y9vx5lDG6RccO4aGOe9mgIcAO3SuZtPIf3dCRl/Q13I0V+PqMcUG3vW0b6Dz+cbzuOAFeP4J64gt+cGoJePC1Df0IwuT/nHGMYtzeHnQvq4yqU1jFvOyQVK3s0at3z9QDv9/A/pDd4wH6JNE3Y=",
        "curves": {"train": [0.7069, 0.1611, 0.1004, 0.0671, 0.0499, 0.0417, 0.0374, 0.0362], "val": [0.2837, 0.1223, 0.0779, 0.0547, 0.0424, 0.0365, 0.0346, 0.0341]},
    },
    "same-rates": {
        "acc": [0.9879, 0.988, 0.9886, 0.9873, 0.9881],
        "ious": [[0.9002, 0.796, 0.8285], [0.9049, 0.7986, 0.8196], [0.9109, 0.8074, 0.8312], [0.9019, 0.7888, 0.8156], [0.9074, 0.8031, 0.8225]],
        "val": [0.034, 0.0346, 0.0329, 0.0362, 0.0353],
        "map": "eNrt1TEOgjAUBuCHxMWYdPIILk4ewASOwNAuPQ2XcHchkZ7CI3EAkkoLaBPa9wyNMTH9B5Yvfe+1JQCQkpKSkhKXnaoJV4pype6Eh2ucJq9/7LDCL4Rr3HP9QJ0RrnHPtemvgl44fpW+8m9vpGc623/0Vtae8k5/eVtON3sjVSO4p/xU37rwlB988/JyMT3q7nohB68883VhtwO6zj3nhzqbXdhwzwX0Zv9tyBnhYBwg7Hlnnsegj9lGevaZV2sdKD9YLwHfAODOw57h7ccBSuwjii9P+cecCe8J1zjbn0uUd6gzrSl/RDo6QEF7H7ke3V+BHw/s0xv+xTwBYEAUbQ==",
    },
    "dilated": {
        "acc": [0.9813, 0.9785, 0.9835, 0.9786, 0.9822],
        "ious": [[0.859, 0.6935, 0.7391], [0.8291, 0.6485, 0.717], [0.8817, 0.7351, 0.7531], [0.8362, 0.661, 0.7129], [0.8706, 0.7148, 0.7382]],
        "val": [0.057, 0.0654, 0.0496, 0.0639, 0.0545],
        "map": "eNrt1r1twzAQBWBaQpo0rLKEKm9Aj+DCbDKNl6ER3xLRSBxAwIU/BiLEvHdAiDQBX/uRx9NBImTMyMjIyEhfJsL+SqQ50VVxog/Jl+rc6Z/K+ZrTb9wpzit0m32R5+M4Ip8Zu2Vc3xUnyVP5nVOr/N6vje3IbfWpYHgPF8X9c/c7p+BPwNMKf0b1vf/pJnusfvcNt9+et/tLYwDIc4Hkt+S+6amDrczvLrjNvkyim+xpRZB8juUNF/ebY/0CRH/cIKHLTfVzp59EXxR/K88nX2EvuH1zwO3VA9AdekDdjfzPHBXfMM+s+QrdcoTueFOcNV8Vj52+dbpSH8/P4fGVf4ORv8oXlgwd6w==",
    },
    "plain": {
        "acc": [0.9846, 0.9741, 0.9794, 0.9839, 0.9871],
        "ious": [[0.878, 0.7456, 0.7894], [0.7901, 0.5831, 0.6636], [0.8299, 0.6651, 0.7338], [0.871, 0.7391, 0.7723], [0.8986, 0.7865, 0.8156]],
        "val": [0.0444, 0.0713, 0.0607, 0.0469, 0.0388],
        "map": "eNrt1TEOgjAUBuAKcXFhcvEIHMGQlCN0sCZyGi7h7sLSQa/AkTgASaUFEiJ9f0ka42DfwPLlvf4QeDAWK1asWLECS9WQE6UU8oOyVSNvKnpGPvb/zE8BXnhc4/Mz3Y7eSKdzj2vsqcf57I2UlXK1G7ft8qoc6Rcuq9ox3tz/7A/H+EW/vDjGL12W6/GDJ5TrTxeg/7YOYPN1i35XQOQmQU/PtydAZ8aZelLzhwSD5y/SU+MsJZ1lnbmeSZ++4Dv2JNB3cpMLcgVt85L0o3V6xe1hvMkF2JH4eDtAxB/Rn1XhcR3mqVme0DvodrlA1z6HAbjGATZ4H+La757z8fPjbXzFv1dvsxQUWA==",
    },
}

# --- the numbers ----------------------------------------------------------------------------------------------------

_START = ((3, 1, 1), (2, 2, 1), (3, 1, 1), (2, 2, 1), (3, 1, 1))   # conv1, a pooling, conv2, a pooling, conv3


def layers_of(variant: str, branch: int | None = None) -> list:
    """The layers of a network of the course from the scene up, as (kernel, stride, dilation): up to its last 3×3
    convolution, or for an ASPP up to the branch with that dilation (1: its 1×1 branch). "fcn-16" and "fcn-8" are the
    FCNs of cv-fcn with that output stride."""
    if variant.startswith("fcn-"):
        return [(3, 1, 1), (2, 2, 1)] * int(math.log2(int(variant[4:]))) + [(3, 1, 1)]
    dilate, rates = VARIANTS[variant]
    d = 2 if dilate else 1
    if rates:
        r = max(rates) if branch is None else branch
        end = (1, 1, 1) if r == 1 and branch == 1 else (3, 1, r)
    else:
        end = (3, 1, 2 * d if dilate else 1)
    return list(_START) + [(3, 1, d), end]


def sees(variant: str) -> tuple[int, int]:
    """How many pixels one place of the last map of a network sees, along one side: the least and the most, which
    differ for the branches of an ASPP."""
    if variant in VARIANTS and VARIANTS[variant][1]:
        return field(layers_of(variant, 1)), field(layers_of(variant))
    return field(layers_of(variant)), field(layers_of(variant))


def params(variant: str) -> int:
    model = _FCN.FCN(int(variant[4:])) if variant.startswith("fcn-") else make(variant)
    return sum(p.numel() for p in model.parameters())


def _trained(variant: str) -> dict:
    """What TRAINED holds for one of the course's networks; "fcn-16" and "fcn-8" from cv-fcn's TRAINED."""
    return _FCN.TRAINED["s" + variant[4:]] if variant.startswith("fcn-") else TRAINED[variant]


def last_side(variant: str) -> int:
    """The side of the last map of a network, before its transposed convolution."""
    return scenes.SIZE // (int(variant[4:]) if variant.startswith("fcn-") else STRIDE)


# The worked example's numbers, measured again by `worked()`: the spread-out kernel of the first cell, the shapes of its
# two maps and their largest difference; for every dilation of the second cell, the pixels of the middle row and of
# the whole picture the 1 reaches; and the shapes of the FCN's map and DeepLab's of the third cell, and the largest
# difference at every second place.
REFERENCE = {
    "kernel": [[-0.0025, 0.0, 0.1788, 0.0, -0.2743], [0.0, 0.0, 0.0, 0.0, 0.0], [-0.2453, 0.0, -0.1284, 0.0, 0.0894],
               [0.0, 0.0, 0.0, 0.0, 0.0], [-0.0066, 0.0, 0.2643, 0.0, -0.0296]],
    "spread_shapes": [[1, 1, 8, 8], [1, 1, 8, 8]],
    "spread_diff": 0.0,
    "seen": [[1, 3, 9], [2, 7, 49], [4, 15, 225], [8, 31, 961]],
    "shapes": [[1, 64, 16, 16], [1, 64, 31, 31]],
    "deeplab_diff": 0.0,
}


# --- the worked example -----------------------------------------------------------------------------------------------

def worked() -> dict:
    """The numbers the cells of the worked example print, worked out as they do."""
    torch.manual_seed(0)
    dilated = nn.Conv2d(1, 1, 3, padding=2, dilation=2, bias=False)
    spread = nn.Conv2d(1, 1, 5, padding=2, bias=False)
    kernel = torch.zeros(5, 5)
    with torch.no_grad():
        for i in range(3):
            for j in range(3):
                kernel[2 * i, 2 * j] = dilated.weight[0, 0, i, j]
        spread.weight[0, 0] = kernel
    kernel = [[round(float(v), 4) for v in row] for row in kernel]
    x = torch.rand(1, 1, 8, 8)
    with torch.no_grad():
        a, b = dilated(x), spread(x)
    spread_diff = round(float((a - b).abs().max()), 5)

    x = torch.zeros(1, 1, 41, 41)
    x[0, 0, 20, 20] = 1
    seen = []
    for d in [1, 2, 4, 8]:
        conv = nn.Conv2d(1, 1, 3, padding=d, dilation=d, bias=False)
        nn.init.ones_(conv.weight)
        with torch.no_grad():
            x = conv(x)
        seen.append([d, int((x[0, 0, 20] > 0).sum()), int((x[0, 0] > 0).sum())])

    torch.manual_seed(0)
    conv1 = nn.Conv2d(3, 16, 3, padding=1)
    conv2 = nn.Conv2d(16, 32, 3, padding=1)
    conv3 = nn.Conv2d(32, 64, 3, padding=1)
    conv4 = nn.Conv2d(64, 64, 3, padding=1)
    pool = nn.MaxPool2d(2)
    x = torch.tensor(_FCN.example().image).permute(2, 0, 1).unsqueeze(0).float() / 255
    with torch.no_grad():
        y = pool(torch.relu(conv2(pool(torch.relu(conv1(x))))))
        y = torch.relu(conv3(y))
        fcn = torch.relu(conv4(pool(y)))
        conv4d = nn.Conv2d(64, 64, 3, padding=2, dilation=2)
        conv4d.load_state_dict(conv4.state_dict())
        deeplab = torch.relu(conv4d(nn.MaxPool2d(2, stride=1)(y)))
    return {"kernel": kernel, "spread_shapes": [list(a.shape), list(b.shape)], "spread_diff": spread_diff, "seen": seen,
            "shapes": [list(fcn.shape), list(deeplab.shape)],
            "deeplab_diff": round(float((deeplab[:, :, ::2, ::2] - fcn).abs().max()), 5)}


# --- the pictures -------------------------------------------------------------------------------------------------

def _escape(text) -> str:
    return html.escape(str(text), quote=False)


def _caption(text: str, x: float, y: float) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:middle">'
            f'{_escape(text)}</text>')


def taps_picture(words: dict) -> str:
    """Three 3×3 kernels on a grid of pixels, with dilation 1, 2 and 4: the 9 pixels each multiplies by a weight (its
    taps, in the color of what a model learns), and dashed around them the part of the picture it sees."""
    n, cell = 9, 13
    side = n * cell
    mid = n // 2
    panels = []
    for d in (1, 2, 4):
        parts = [_FCN._title(words["title"].format(d=d), side)]
        for i in range(n):
            for j in range(n):
                tap = abs(i - mid) in (0, d) and abs(j - mid) in (0, d)
                style = "fill:var(--ea-param);stroke:var(--ea-page)" if tap else "fill:var(--ea-page);stroke:var(--ea-line)"
                parts.append(f'<rect x="{j * cell}" y="{_FCN._TOP + i * cell}" width="{cell}" height="{cell}" style="{style}"/>')
        lo = (mid - d) * cell
        w = (2 * d + 1) * cell
        parts.append(f'<rect x="{lo + 1}" y="{_FCN._TOP + lo + 1}" width="{w - 2}" height="{w - 2}" '
                     'style="fill:none;stroke:var(--ea-ink);stroke-width:1.5;stroke-dasharray:4 3"/>')
        parts.append(_caption(words["sees"].format(n=2 * d + 1), side / 2, _FCN._TOP + side + 16))
        panels.append((_FCN._svg(parts, side, _FCN._TOP + side + 22), side))
    return _FCN.side_by_side(panels, words["alt"])


def paths_along(dilations) -> dict:
    """For a stack of 3×3 convolutions with these dilations: for every offset from a place along one side, along how
    many paths through the stack that pixel reaches the place (0 for a pixel it does not see)."""
    counts = {0: 1}
    for d in dilations:
        new: dict = {}
        for offset, c in counts.items():
            for t in (-d, 0, d):
                new[offset + t] = new.get(offset + t, 0) + c
        counts = new
    return counts


GAPS = (1, 2, 3, 4, 5, 6)   # the dilations of the slider: a 3×3 convolution with dilation 1, then one with this one


def gaps_frame(d: int, words: dict) -> tuple[str, dict]:
    """For the slider: the pixels around one place that a 3×3 convolution with dilation 1 followed by one with dilation
    d sees, on a grid of 15 by 15 pixels: a pixel seen along more paths is fuller, a pixel it skips is left empty, and
    the place itself has a frame. Next to it, how wide the part it sees is, how many pixels in it it sees, and how
    many it skips. The color is as full for the same number of paths in every frame: 9, the most, with dilation 1."""
    along = paths_along([1, d])
    n, cell = 15, 14
    side = n * cell
    mid = n // 2
    most = 9
    width = max(along) - min(along) + 1
    parts = [_FCN._title(words["title"].format(d=d), side)]
    for i in range(n):
        for j in range(n):
            paths = along.get(i - mid, 0) * along.get(j - mid, 0)
            if paths:
                style = f"fill:var(--ea-data);fill-opacity:{0.3 + 0.7 * paths / most:.2f};stroke:var(--ea-page)"
            else:
                style = "fill:var(--ea-page);stroke:var(--ea-line)"
            parts.append(f'<rect x="{j * cell}" y="{_FCN._TOP + i * cell}" width="{cell}" height="{cell}" style="{style}"/>')
    lo = (mid + min(along)) * cell
    parts.append(f'<rect x="{lo + 1}" y="{_FCN._TOP + lo + 1}" width="{width * cell - 2}" height="{width * cell - 2}" '
                 'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.5;stroke-dasharray:4 3"/>')
    parts.append(f'<rect x="{mid * cell + 1}" y="{_FCN._TOP + mid * cell + 1}" width="{cell - 2}" height="{cell - 2}" '
                 'style="fill:none;stroke:var(--ea-ink);stroke-width:2"/>')
    parts.append(_caption(words["key"], side / 2, _FCN._TOP + side + 16))
    svg = _FCN._svg(parts, side, _FCN._TOP + side + 22, words["alt"].format(d=d))
    seen = sum(1 for a in along.values() if a) ** 2
    return svg, {"width": width, "seen": seen, "gaps": width * width - seen}


def maps_picture(words: dict) -> str:
    """The scene of the worked example; the class maps that the course's FCN with stride 16 of cv-fcn and the course's
    DeepLab give it (seed 0), with the same layers and the same first two poolings; and the true class map."""
    panels = [(_FCN.photo_svg(SEED, words["scene"]), 150),
              (_FCN.map_svg(_FCN.unpack(_FCN.TRAINED["s16"]["map"], scenes.SIZE), words["fcn"]), 150),
              (_FCN.map_svg(_FCN.unpack(TRAINED["deeplab"]["map"], scenes.SIZE), words["deeplab"]), 150),
              (_FCN.map_svg(_FCN.class_map(_FCN.example()), words["truth"]), 150)]
    return _FCN.side_by_side(panels, words["alt"]) + _FCN.key_html(words["kinds"])


ROWS = ("deeplab", "same-rates", "dilated", "plain", "fcn-4", "fcn-8", "fcn-16")


def variants_table(words: dict, lang: str, rows=ROWS) -> str:
    """The course's networks side by side, as a Markdown table: for each, the side of its last map, what one place of
    it sees, and the IoU of every kind on the test scenes, from the lowest to the highest over the seeds 0 to 4."""
    lines = ["| " + " | ".join(words["header"]) + " |", "|" + "---|" * len(words["header"])]
    for variant in rows:
        r = _trained(variant)
        side = last_side(variant)
        lo, hi = sees(variant)
        cells = [words["names"][variant], f"{side} × {side}", str(lo) if lo == hi else f"{lo}–{hi}"] + \
            [_FCN.span_text([run[k] for run in r["ious"]], lang) for k in range(len(KINDS))]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def middle(variant: str, kind: int) -> float:
    """The median IoU of one kind (0 circles, 1 squares, 2 triangles) over the seeds of a network."""
    return float(np.median([run[kind] for run in _trained(variant)["ious"]]))


# --- the exercises: shared --------------------------------------------------------------------------------------------

_DOTS, _TEXTS, _close, _kept = _FCN._DOTS, _FCN._TEXTS, _FCN._close, _FCN._kept
_S_EN, _S_NL = _FCN._S_EN, _FCN._S_NL
_K_EN = "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_K_NL = "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"


def _channels(e: BaseException, want: int | None = None, got: int | None = None) -> bool:
    """A convolution stopped because it got another number of channels than it was made for: `want` the channels it
    expected, `got` those it got, each when given."""
    if not isinstance(e, RuntimeError):
        return False
    m = re.search(r"to have (\d+) channels, but got (\d+) channels", str(e))
    return m is not None and (want is None or int(m.group(1)) == want) and (got is None or int(m.group(2)) == got)


def _sizes(e: BaseException) -> bool:
    """torch.cat stopped because its maps are not the same size."""
    return isinstance(e, RuntimeError) and "Sizes of tensors must match" in str(e)


# --- exercise 1: field ------------------------------------------------------------------------------------------------

_F_HEAD = ("def field(layers):\n"
           "    # layers: one (kernel, stride, dilation) for every layer, from the scene up; a pooling of 2\n"
           "    # is (2, 2, 1), a 3x3 convolution with dilation 2 is (3, 1, 2)\n"
           "    # give back how many pixels of the scene one place after the last layer sees, along one side\n")
_F_START = _F_HEAD + "    ...\n"
_F_SEEN = "    seen = 1\n    step = 1\n"
_F_LOOP = "    for k, stride, d in layers:\n"
_F_ADD = "        seen += (k - 1) * d * step\n"
_F_MUL = "        step *= stride\n"
_F_RETURN = "    return seen\n"


def _f(seen=_F_SEEN, loop=_F_LOOP, add=_F_ADD, mul=_F_MUL, ret=_F_RETURN) -> str:
    return _F_HEAD + seen + loop + add + mul + ret


_F = _f()
_F_SCENARIOS = [
    # the course's FCN with stride 8: three times a 3x3 convolution and a pooling of 2, then one more 3x3 convolution
    "field([(3, 1, 1), (2, 2, 1), (3, 1, 1), (2, 2, 1), (3, 1, 1), (2, 2, 1), (3, 1, 1)])",
    # three 3x3 convolutions with dilation 1, 2 and 4
    "field([(3, 1, 1), (3, 1, 2), (3, 1, 4)])",
    # the course's DeepLab up to conv4, then each branch of its ASPP
    "start = [(3, 1, 1), (2, 2, 1), (3, 1, 1), (2, 2, 1), (3, 1, 1), (3, 1, 2)]\n"
    "[field(start + [(1, 1, 1)]), field(start + [(3, 1, 2)]), field(start + [(3, 1, 4)]), field(start + [(3, 1, 8)])]",
    # a 7x7 convolution with stride 2 and a 3x3 one with stride 2, and no layers at all
    "[field([(7, 2, 1), (3, 2, 1)]), field([])]",
]
_F_WRONG = {
    "no-step": _f(add="        seen += (k - 1) * d\n"),
    "no-dilation": _f(add="        seen += (k - 1) * step\n"),
    "kernel": _f(add="        seen += k * d * step\n"),
    "step-first": _f(add=_F_MUL, mul=_F_ADD),
    "step-set": _f(mul="        step = stride\n"),
    "start-0": _f(seen="    seen = 0\n    step = 1\n"),
}
_F_MISTAKES = [
    Mistake("no-step", wrong=_F_WRONG["no-step"],
            en=_S_EN + ": after a pooling of 2, neighboring places of the map are 2 pixels of the scene apart, and after two poolings 4. A layer after them reaches that much further: it adds `(k - 1) * d` times the step between neighbors, `step`, and every stride multiplies that step.",
            nl=_S_NL + ": na een pooling van 2 liggen buurplaatsen van de map 2 pixels van de scène uit elkaar, en na twee poolings 4. Een layer daarna reikt zoveel verder: hij telt `(k - 1) * d` keer de stap tussen buren op, `step`, en elke stride vermenigvuldigt die stap."),
    Mistake("no-dilation", wrong=_F_WRONG["no-dilation"],
            en=_S_EN + ": your function leaves the dilation out. A 3×3 convolution with dilation 2 has its taps 2 places apart, so it spans 5 places of its map instead of 3: it adds `(k - 1) * d` places, times the step.",
            nl=_S_NL + ": je functie laat de dilation weg. Een 3×3-convolutie met dilation 2 heeft zijn taps 2 plaatsen uit elkaar, dus hij beslaat 5 plaatsen van zijn map in plaats van 3: hij telt `(k - 1) * d` plaatsen op, keer de stap."),
    Mistake("kernel", wrong=_F_WRONG["kernel"],
            en=_S_EN + ": your function adds a whole kernel for every layer. The place in the middle was already counted: a 3×3 convolution adds 1 place on each side, 2 in all, so `(k - 1) * d`, times the step.",
            nl=_S_NL + ": je functie telt voor elke layer een hele kernel op. De plaats in het midden was al geteld: een 3×3-convolutie voegt aan elke kant 1 plaats toe, 2 in totaal, dus `(k - 1) * d`, keer de stap."),
    Mistake("step-first", wrong=_F_WRONG["step-first"],
            en=_S_EN + ": your function multiplies the step before the layer adds what it sees. A layer with stride 2 itself still works on its input, whose neighbors are `step` apart; its stride only counts for the layers after it. So first `seen += ...`, then `step *= stride`.",
            nl=_S_NL + ": je functie vermenigvuldigt de stap voordat de layer optelt wat hij ziet. Een layer met stride 2 werkt zelf nog op zijn input, waarvan de buren `step` uit elkaar liggen; zijn stride telt pas voor de layers erna. Dus eerst `seen += ...`, dan `step *= stride`."),
    Mistake("step-set", wrong=_F_WRONG["step-set"],
            en=_S_EN + ": your function sets the step to the stride of the last layer. Strides multiply: after two poolings of 2, neighbors are 2 · 2 = 4 pixels apart. So `step *= stride`.",
            nl=_S_NL + ": je functie zet de stap op de stride van de laatste layer. Strides vermenigvuldigen: na twee poolings van 2 liggen buren 2 · 2 = 4 pixels uit elkaar. Dus `step *= stride`."),
    Mistake("start-0", wrong=_F_WRONG["start-0"],
            en=_S_EN + ": without any layer, a place is one pixel of the scene, and sees just that pixel: `field([])` is 1. Start at `seen = 1`.",
            nl=_S_NL + ": zonder layers is een plaats één pixel van de scène, en ziet hij alleen die pixel: `field([])` is 1. Begin bij `seen = 1`."),
]


# --- exercise 2: ASPP -------------------------------------------------------------------------------------------------

_A_HEAD = ("import torch\nfrom torch import nn\n\n\nclass ASPP(nn.Module):\n"
           "    # side by side on one map: a 1x1 convolution, and a 3x3 convolution for every dilation in\n"
           "    # rates, each from c_in channels to c and followed by ReLU; their maps stacked along the\n"
           "    # channels, and a 1x1 convolution with ReLU that makes them c channels again\n"
           "    def __init__(self, c_in, c, rates):\n        super().__init__()\n")
_A_BRANCHES = ("        self.branches = nn.ModuleList([nn.Conv2d(c_in, c, 1)] +\n"
               "                                      [nn.Conv2d(c_in, c, 3, padding=r, dilation=r) for r in rates])\n")
_A_PROJECT = "        self.project = nn.Conv2d(c * (len(rates) + 1), c, 1)\n"
_A_FORWARD = "\n    def forward(self, x):\n"
_A_BODY = ("        maps = [torch.relu(branch(x)) for branch in self.branches]\n"
           "        return torch.relu(self.project(torch.cat(maps, dim=1)))\n")
_A_START = _A_HEAD + "        self.branches = ...\n        self.project = ...\n" + _A_FORWARD + "        ...\n"


def _a(branches=_A_BRANCHES, project=_A_PROJECT, body=_A_BODY) -> str:
    return _A_HEAD + branches + project + _A_FORWARD + body


_A = _a()
_A_SCENARIOS = [
    # 8 channels in, 4 out, with the dilations 2 and 4
    "torch.manual_seed(0)\naspp = ASPP(8, 4, [2, 4])\nx = torch.rand(2, 8, 12, 12)\nwith torch.no_grad():\n    y = aspp(x)\n"
    "[list(y.shape), round(float(y.mean()), 4), round(float(y[1, :, 3, 5].sum()), 4), sum(p.numel() for p in aspp.parameters())]",
    # other channels, three dilations, a map that is not square, and the share of values ReLU makes 0
    "torch.manual_seed(1)\naspp = ASPP(3, 5, [1, 3, 6])\nx = torch.randn(1, 3, 7, 9)\nwith torch.no_grad():\n    y = aspp(x)\n"
    "[list(y.shape), round(float(y[0, 4].sum()), 4), round(float((y == 0).float().mean()), 4)]",
]
_A_WRONG = {
    "no-relu": _a(body="        maps = [branch(x) for branch in self.branches]\n"
                       "        return torch.relu(self.project(torch.cat(maps, dim=1)))\n"),
    "no-last-relu": _a(body="        maps = [torch.relu(branch(x)) for branch in self.branches]\n"
                            "        return self.project(torch.cat(maps, dim=1))\n"),
    "list": _a(branches="        self.branches = [nn.Conv2d(c_in, c, 1)]\n"
                        "        self.branches += [nn.Conv2d(c_in, c, 3, padding=r, dilation=r) for r in rates]\n"),
    "one-last": _a(branches="        self.branches = nn.ModuleList([nn.Conv2d(c_in, c, 3, padding=r, dilation=r) for r in rates] +\n"
                            "                                      [nn.Conv2d(c_in, c, 1)])\n"),
    "added": _a(project="        self.project = nn.Conv2d(c, c, 1)\n",
                body="        total = 0\n        for branch in self.branches:\n            total = total + torch.relu(branch(x))\n"
                     "        return torch.relu(self.project(total))\n"),
}
_A_MISTAKES = [
    Mistake("padding", error=_sizes,
            en="Python stops at `torch.cat`: the maps of the branches are not the same size. A 3×3 convolution with dilation `r` spans 2r + 1 pixels, and `padding=r` keeps its map as large as it was; without the dilation, `padding=r` makes the map larger. Give every 3×3 branch both: `nn.Conv2d(c_in, c, 3, padding=r, dilation=r)`. The 1×1 convolution needs no padding.",
            nl="Python stopt bij `torch.cat`: de maps van de takken zijn niet even groot. Een 3×3-convolutie met dilation `r` beslaat 2r + 1 pixels, en `padding=r` houdt zijn map zo groot als hij was; zonder de dilation maakt `padding=r` de map groter. Geef elke 3×3-tak allebei: `nn.Conv2d(c_in, c, 3, padding=r, dilation=r)`. De 1×1-convolutie heeft geen padding nodig."),
    Mistake("stack", error=lambda e: _channels(e, want=12, got=4),
            en="Python stops in `self.project`: it was made for 12 channels, and gets 4. Stack the maps of the branches along the channels, `torch.cat(maps, dim=1)`, so that the 4 channels of each of the 3 branches come in side by side. Adding them keeps 4; so does stacking along `dim=0`, which stacks the scenes of the batch.",
            nl="Python stopt in `self.project`: die is gemaakt voor 12 channels, en krijgt er 4. Stapel de maps van de takken langs de channels, `torch.cat(maps, dim=1)`, zodat de 4 channels van elk van de 3 takken naast elkaar binnenkomen. Optellen houdt het op 4; stapelen langs `dim=0` ook, want dat stapelt de scènes van de batch."),
    Mistake("no-1x1", error=lambda e: _channels(e, want=12, got=8),
            en="Python stops in `self.project`: it was made for 12 channels, and gets 8. With the dilations 2 and 4, an ASPP has three branches: the 1×1 convolution, and a 3×3 convolution for every dilation. Is the 1×1 convolution, `nn.Conv2d(c_in, c, 1)`, the first of `self.branches`?",
            nl="Python stopt in `self.project`: die is gemaakt voor 12 channels, en krijgt er 8. Met de dilations 2 en 4 heeft een ASPP drie takken: de 1×1-convolutie, en een 3×3-convolutie voor elke dilation. Is de 1×1-convolutie, `nn.Conv2d(c_in, c, 1)`, de eerste van `self.branches`?"),
    Mistake("branch-in", error=lambda e: _channels(e, want=4, got=8),
            en="Python stops in a branch: it was made for 4 channels, and gets the 8 of the map. Every branch gets the map `x` itself, with `c_in` channels, and gives `c`: `nn.Conv2d(c_in, c, ...)`.",
            nl="Python stopt in een tak: die is gemaakt voor 4 channels, en krijgt de 8 van de map. Elke tak krijgt de map `x` zelf, met `c_in` channels, en geeft er `c`: `nn.Conv2d(c_in, c, ...)`."),
    Mistake("project-channels", error=_channels,
            en="Python stops in a convolution: it gets another number of channels than it was made for. Every branch gets the `c_in` channels of `x` and gives `c`; `self.project` gets the `c` channels of every branch, of the 1×1 convolution and of one 3×3 convolution for every dilation: `nn.Conv2d(c * (len(rates) + 1), c, 1)`.",
            nl="Python stopt in een convolutie: die krijgt een ander aantal channels dan waarvoor hij gemaakt is. Elke tak krijgt de `c_in` channels van `x` en geeft er `c`; `self.project` krijgt de `c` channels van elke tak, van de 1×1-convolutie en van één 3×3-convolutie voor elke dilation: `nn.Conv2d(c * (len(rates) + 1), c, 1)`."),
    Mistake("no-relu", wrong=_A_WRONG["no-relu"],
            en=_K_EN + ": the shape comes first, then a mean, a sum, and the number of parameters. Every branch is followed by ReLU before the stack: `torch.relu(branch(x))`.",
            nl=_K_NL + ": de shape staat voorop, dan een gemiddelde, een som, en het aantal parameters. Na elke tak komt ReLU, voor het stapelen: `torch.relu(branch(x))`."),
    Mistake("no-last-relu", wrong=_A_WRONG["no-last-relu"],
            en=_K_EN + ": your ASPP gives negative numbers, because `self.project` has no ReLU after it. The 1×1 convolution at the end is followed by ReLU too.",
            nl=_K_NL + ": je ASPP geeft negatieve getallen, omdat `self.project` geen ReLU na zich heeft. De 1×1-convolutie aan het eind krijgt ook ReLU na zich."),
    Mistake("list", wrong=_A_WRONG["list"],
            en=_K_EN + ": the last number is the parameters, and yours has far fewer. A plain Python list hides its convolutions from PyTorch: `parameters()` does not see them, so training would never change them. Put them in a `nn.ModuleList`, as in [VGG: deeper with small filters](/course/open/cnn-vgg-blocks?lang=en).",
            nl=_K_NL + ": het laatste getal is het aantal parameters, en het jouwe is veel kleiner. Een gewone Python-lijst verbergt zijn convoluties voor PyTorch: `parameters()` ziet ze niet, dus training zou ze nooit veranderen. Zet ze in een `nn.ModuleList`, zoals in [VGG: dieper met kleine filters](/course/open/cnn-vgg-blocks?lang=nl)."),
    Mistake("one-last", wrong=_A_WRONG["one-last"],
            en=_K_EN + ": your ASPP has the 1×1 convolution last. The course puts it first, before the 3×3 convolutions: `nn.ModuleList([nn.Conv2d(c_in, c, 1)] + [...])`. The order decides which weights of `self.project` see which channels, and which random weights each convolution starts with, so the numbers of the check change with it.",
            nl=_K_NL + ": je ASPP heeft de 1×1-convolutie als laatste. De course zet hem voorop, voor de 3×3-convoluties: `nn.ModuleList([nn.Conv2d(c_in, c, 1)] + [...])`. De volgorde bepaalt welke weights van `self.project` welke channels zien, en met welke willekeurige weights elke convolutie begint, dus de getallen van de check veranderen mee."),
    Mistake("added", wrong=_A_WRONG["added"],
            en=_K_EN + ": your ASPP adds the maps of the branches up. It stacks them along the channels instead, `torch.cat(maps, dim=1)`, so that `self.project` gets every branch whole and learns itself how to weigh them: `nn.Conv2d(c * (len(rates) + 1), c, 1)`.",
            nl=_K_NL + ": je ASPP telt de maps van de takken op. Hij stapelt ze in plaats daarvan langs de channels, `torch.cat(maps, dim=1)`, zodat `self.project` elke tak heel krijgt en zelf leert hoe zwaar elke tak telt: `nn.Conv2d(c * (len(rates) + 1), c, 1)`."),
]


# --- exercise 3: DeepLab ----------------------------------------------------------------------------------------------

_D_HEAD = ("import torch\nfrom torch import nn\nfrom cnn.aspp import ASPP\nfrom cnn.upsample import upsampler\n\n\n"
           "class DeepLab(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
           "        # the layers of the course's FCN with stride 16, 3x3 convolutions with 16, 32, 64 and 64\n"
           "        # channels, of which only the first two are followed by a pooling of 2\n"
           "        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)\n"
           "        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)\n"
           "        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)\n"
           "        # conv4 came after the third pooling, which is gone: dilated by 2\n")
_D_CONV4 = "        self.conv4 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)\n"
_D_REST = ("        self.pool = nn.MaxPool2d(2)\n"
           "        # the end came after the fourth pooling, also gone: an ASPP with dilations around 4\n"
           "        self.aspp = ASPP(64, 64, [2, 4, 8])\n"
           "        # a score for every class at every place, then 4 times larger, as in your FCN\n"
           "        self.score = nn.Conv2d(64, 4, 1)\n"
           "        self.up = upsampler(4, 4)\n")
_D_FORWARD = ("\n    def forward(self, x):\n"
              "        # x: (N, 3, H, W), H and W multiples of 4; give back the scores, (N, 4, H, W)\n")
_D_BODY = ("        x = self.pool(torch.relu(self.conv1(x)))\n"
           "        x = self.pool(torch.relu(self.conv2(x)))\n"
           "        x = torch.relu(self.conv3(x))\n"
           "        x = torch.relu(self.conv4(x))\n"
           "        return self.up(self.score(self.aspp(x)))\n")
_D_START = _D_HEAD + "        self.conv4 = ...\n" + _D_REST + _D_FORWARD + "        ...\n"


def _d(conv4=_D_CONV4, body=_D_BODY) -> str:
    return _D_HEAD + conv4 + _D_REST + _D_FORWARD + body


_D = _d()
_D_SCENARIOS = [
    # one pixel of a scene of 128 by 128 changed: the shape of the change, and the rows of the scores it reaches
    "torch.manual_seed(2)\nnet = DeepLab()\nx = torch.zeros(1, 3, 128, 128)\ny = torch.zeros(1, 3, 128, 128)\ny[0, :, 64, 64] = 1\n"
    "with torch.no_grad():\n    change = (net(y) - net(x)).abs().sum(1)[0]\n[list(change.shape), int((change.sum(1) > 0).sum())]",
    # 2 scenes of 32 by 32
    "torch.manual_seed(0)\nnet = DeepLab()\nx = torch.rand(2, 3, 32, 32)\nwith torch.no_grad():\n    out = net(x)\n"
    "[list(out.shape), round(float(out[0, :, 9, 13].sum()), 4), round(float(out[1].mean()), 4), sum(p.numel() for p in net.parameters())]",
    # 1 scene of 24 by 40
    "torch.manual_seed(1)\nnet = DeepLab()\nx = torch.rand(1, 3, 24, 40)\nwith torch.no_grad():\n    out = net(x)\n"
    "[list(out.shape), round(float(out[0, 2, 23, 39]), 4), round(float(out[0, :, 0, 0].sum()), 4)]",
]
_D_WRONG = {
    "not-dilated": _d(conv4="        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)\n"),
    "padding-1": _d(conv4="        self.conv4 = nn.Conv2d(64, 64, 3, padding=1, dilation=2)\n"),
    "third-pool": _d(body=_D_BODY.replace("        x = torch.relu(self.conv3(x))\n", "        x = self.pool(torch.relu(self.conv3(x)))\n")),
    "no-aspp": _d(body=_D_BODY.replace("self.score(self.aspp(x))", "self.score(x)")),
}
_D_MISTAKES = [
    Mistake("no-score", error=lambda e: _channels(e, want=4, got=64),
            en="Python stops in `self.up`: it was made for the 4 scores, one map for every class, and gets the 64 channels of the ASPP. First the 1×1 convolution to the scores, then 4 times larger: `self.up(self.score(self.aspp(x)))`.",
            nl="Python stopt in `self.up`: die is gemaakt voor de 4 scores, één map voor elke klasse, en krijgt de 64 channels van de ASPP. Eerst de 1×1-convolutie naar de scores, dan 4 keer zo groot: `self.up(self.score(self.aspp(x)))`."),
    Mistake("conv-channels", error=_channels,
            en="Python stops in a convolution: it gets another number of channels than it was made for. Every layer gets the channels of the layer before it: conv4 gets the 64 of conv3 and gives 64, `nn.Conv2d(64, 64, 3, ...)`, and the layers go in the order conv1, conv2, conv3, conv4, ASPP, score, up.",
            nl="Python stopt in een convolutie: die krijgt een ander aantal channels dan waarvoor hij gemaakt is. Elke layer krijgt de channels van de layer ervoor: conv4 krijgt de 64 van conv3 en geeft er 64, `nn.Conv2d(64, 64, 3, ...)`, en de layers komen in de volgorde conv1, conv2, conv3, conv4, ASPP, score, up."),
    Mistake("not-dilated", wrong=_D_WRONG["not-dilated"],
            en=_K_EN + ": the shape of the scores is right, but one changed pixel of the scene reaches fewer rows of them. Your conv4 has no dilation. It came after the third pooling, which is gone, so its neighbors are now twice as close: with `dilation=2` its taps are 2 places apart, and it reaches about as far as it did in the FCN. Give it `padding=2` too.",
            nl=_K_NL + ": de shape van de scores klopt, maar één veranderde pixel van de scène bereikt minder rijen ervan. Je conv4 heeft geen dilation. Hij kwam na de derde pooling, die weg is, dus zijn buren liggen nu twee keer zo dicht bij elkaar: met `dilation=2` liggen zijn taps 2 plaatsen uit elkaar, en reikt hij ongeveer zo ver als in het FCN. Geef hem ook `padding=2`."),
    Mistake("padding-1", wrong=_D_WRONG["padding-1"],
            en=_K_EN + ": the shape comes first, and your scores are smaller than the scene. A 3×3 convolution with dilation 2 spans 5 places, so with `padding=1` it loses a place on every side. `padding=2` keeps the map as large as it was.",
            nl=_K_NL + ": de shape staat voorop, en je scores zijn kleiner dan de scène. Een 3×3-convolutie met dilation 2 beslaat 5 plaatsen, dus met `padding=1` verliest hij aan elke kant een plaats. `padding=2` houdt de map zo groot als hij was."),
    Mistake("third-pool", wrong=_D_WRONG["third-pool"],
            en=_K_EN + ": the shape comes first, and your scores are smaller than the scene. Only conv1 and conv2 are followed by a pooling; that is the point of DeepLab. After conv3 comes conv4 at once, on a map 4 times smaller than the scene, and the upsampler makes it 4 times larger again.",
            nl=_K_NL + ": de shape staat voorop, en je scores zijn kleiner dan de scène. Alleen na conv1 en conv2 komt een pooling; daar draait DeepLab om. Na conv3 komt meteen conv4, op een map die 4 keer kleiner is dan de scène, en de upsampler maakt hem weer 4 keer zo groot."),
    Mistake("no-aspp", wrong=_D_WRONG["no-aspp"],
            en=_K_EN + ": one changed pixel of the scene reaches far fewer rows of your scores, because your network skips the ASPP. After conv4 comes `self.aspp`, then the scores: `self.up(self.score(self.aspp(x)))`.",
            nl=_K_NL + ": één veranderde pixel van de scène bereikt veel minder rijen van je scores, omdat je netwerk de ASPP overslaat. Na conv4 komt `self.aspp`, dan de scores: `self.up(self.score(self.aspp(x)))`."),
]


# --- the exercises ----------------------------------------------------------------------------------------------------

def _scene_batch() -> torch.Tensor:
    """The scene of the worked example as a batch of one: what the pictures under exercises 2 and 3 send through the
    learner's ASPP or network, redrawn as they type (netview.network)."""
    return torch.tensor(scenes.scene(SEED).image).permute(2, 0, 1).unsqueeze(0).float() / 255


EXERCISES = {
    "field": ScenarioBox(
        name="field",
        start=_F_START,
        solution=_F,
        scenarios=_F_SCENARIOS,
        compare=_close,
        picture=yours.steps(scenario=1),
        first=[_DOTS],
        mistakes=_F_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: start at one pixel, and let every layer add what it sees, with the step between neighbors of its input.",
                      "nl": "Vervang de `...` door je code: begin bij één pixel, en laat elke layer optellen wat hij ziet, met de stap tussen buren van zijn input."},
            "wrong": {"en": _S_EN + ". Do you start at `seen = 1` and `step = 1`, and for every layer add `(k - 1) * d * step` to `seen`, and then multiply `step` by its stride?",
                      "nl": _S_NL + ". Begin je bij `seen = 1` en `step = 1`, en tel je voor elke layer `(k - 1) * d * step` op bij `seen`, en vermenigvuldig je daarna `step` met zijn stride?"},
            "almost": {"en": "Right for the course's FCN. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function count the dilation, and any stride?",
                       "nl": "Goed voor het FCN van de course. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Telt je functie de dilation mee, en elke stride?"},
            "correct": {"en": "Your `field` is right: one place of the FCN with stride 8 sees 38 pixels, the stack with dilation 1, 2 and 4 sees 15, and the branches of the course's DeepLab see 34, 50, 66 and 98.",
                        "nl": "Je `field` klopt: één plaats van het FCN met stride 8 ziet 38 pixels, de stapel met dilation 1, 2 en 4 ziet er 15, en de takken van de DeepLab van de course zien er 34, 50, 66 en 98."},
        },
    ),
    "aspp": ClassBox(
        name="aspp",
        start=_A_START,
        solution=_A,
        scenarios=_A_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "aspp"),
        picture=netview.network("ASPP", _scene_batch, 3, 8, (2, 4, 8), chosen=3, scores=False),
        first=[_DOTS],
        mistakes=_A_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...`: `self.branches`, the 1×1 convolution and a 3×3 convolution for every dilation; `self.project`, the 1×1 convolution at the end; and `forward`, every branch with ReLU, stacked, and `self.project` with ReLU.",
                      "nl": "Vervang de drie `...`: `self.branches`, de 1×1-convolutie en een 3×3-convolutie voor elke dilation; `self.project`, de 1×1-convolutie aan het eind; en `forward`, elke tak met ReLU, gestapeld, en `self.project` met ReLU."},
            "wrong": {"en": _K_EN + ": the shape of what your ASPP gives, then a mean, a sum, and the number of parameters. Is the 1×1 convolution the first branch, does every 3×3 branch have `padding=r` and `dilation=r`, and are the maps stacked along the channels before `self.project`?",
                      "nl": _K_NL + ": de shape van wat je ASPP geeft, dan een gemiddelde, een som, en het aantal parameters. Is de 1×1-convolutie de eerste tak, heeft elke 3×3-tak `padding=r` en `dilation=r`, en worden de maps langs de channels gestapeld voor `self.project`?"},
            "almost": {"en": "Right for the dilations 2 and 4. But " + _K_EN[0].lower() + _K_EN[1:] + ". Does your ASPP work for any number of dilations, any channels, and maps that are not square?",
                       "nl": "Goed voor de dilations 2 en 4. Maar " + _K_NL[0].lower() + _K_NL[1:] + ". Werkt je ASPP voor elk aantal dilations, elk aantal channels, en maps die niet vierkant zijn?"},
            "correct": {"en": "Your `ASPP` is right: a 1×1 convolution and a 3×3 convolution for every dilation, side by side on the same map, stacked and made `c` channels again. It is now in your project, as `{saved}`.",
                        "nl": "Je `ASPP` klopt: een 1×1-convolutie en een 3×3-convolutie voor elke dilation, naast elkaar op dezelfde map, gestapeld en weer `c` channels gemaakt. Hij staat nu in je project, als `{saved}`."},
            "kept": _kept("ASPP"),
        },
    ),
    "deeplab": ClassBox(
        name="deeplab",
        start=_D_START,
        solution=_D,
        scenarios=_D_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "deeplab"),
        picture=netview.network("DeepLab", _scene_batch),
        first=[_DOTS],
        mistakes=_D_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: `self.conv4`, a 3×3 convolution from 64 channels to 64 with dilation 2, and `forward`, the layers in order with a pooling after only the first two.",
                      "nl": "Vervang de twee `...`: `self.conv4`, een 3×3-convolutie van 64 channels naar 64 met dilation 2, en `forward`, de layers op volgorde met alleen na de eerste twee een pooling."},
            "wrong": {"en": _K_EN + ": the shape of the scores, then a few of them, and the number of parameters. Is conv4 dilated by 2 with `padding=2`, are only conv1 and conv2 followed by a pooling, and do the ASPP, the scores and the upsampler come after conv4 in that order?",
                      "nl": _K_NL + ": de shape van de scores, dan een paar ervan, en het aantal parameters. Heeft conv4 dilation 2 met `padding=2`, komt alleen na conv1 en conv2 een pooling, en komen de ASPP, de scores en de upsampler in die volgorde na conv4?"},
            "almost": {"en": "Right for 2 scenes of 32 by 32. But " + _K_EN[0].lower() + _K_EN[1:] + ". Does your network work for one scene, and for scenes that are not square?",
                       "nl": "Goed voor 2 scènes van 32 bij 32. Maar " + _K_NL[0].lower() + _K_NL[1:] + ". Werkt je netwerk voor één scène, en voor scènes die niet vierkant zijn?"},
            "correct": {"en": "Your `DeepLab` is right: two poolings instead of four, conv4 and the ASPP dilated, and 4 scores for every pixel, with 193,188 parameters. It is now in your project, as `{saved}`.",
                        "nl": "Je `DeepLab` klopt: twee poolings in plaats van vier, conv4 en de ASPP met dilation, en 4 scores voor elke pixel, met 193.188 parameters. Hij staat nu in je project, als `{saved}`."},
            "kept": _kept("DeepLab"),
        },
    ),
}

# The lesson is done when these are correct. "why-dilate" is the open question.
REQUIRED = ["field", "aspp", "deeplab", "why-dilate"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("field", _F_START, "empty", None),
    ("field", _F, "correct", None),
    # the step first, and the stride applied to the next layer's step in another way
    ("field", _F_HEAD + "    seen, step = 1, 1\n    for layer in layers:\n        k, stride, d = layer\n"
              "        seen = seen + (k - 1) * d * step\n        step = step * stride\n    return seen\n", "correct", None),
    ("field", _f(ret="    ...\n"), "wrong", "dots-left"),
] + [("field", code, "wrong", key) for key, code in _F_WRONG.items()] + [
    ("aspp", _A_START, "empty", None),
    ("aspp", _A, "correct", None),
    # the branches made in a loop, and forward as a loop that collects the maps
    ("aspp", _a(branches="        self.branches = nn.ModuleList([nn.Conv2d(c_in, c, kernel_size=1)])\n"
                         "        for r in rates:\n            self.branches.append(nn.Conv2d(c_in, c, 3, padding=r, dilation=r))\n",
                body="        maps = []\n        for branch in self.branches:\n            maps.append(torch.relu(branch(x)))\n"
                     "        stacked = torch.cat(maps, dim=1)\n        return torch.relu(self.project(stacked))\n"), "correct", None),
    ("aspp", _a(branches=_A_BRANCHES.replace("padding=r, ", "padding=1, ")), "error", "padding"),
    ("aspp", _a(branches=_A_BRANCHES.replace("padding=r, ", "")), "error", "padding"),
    ("aspp", _a(branches=_A_BRANCHES.replace(", dilation=r", "")), "error", "padding"),   # padding=r without the dilation
    ("aspp", _a(body=_A_BODY.replace("dim=1", "dim=0")), "error", "stack"),
    ("aspp", _a(body="        return torch.relu(self.project(sum(torch.relu(branch(x)) for branch in self.branches)))\n"), "error", "stack"),
    ("aspp", _a(branches="        self.branches = nn.ModuleList([nn.Conv2d(c_in, c, 3, padding=r, dilation=r) for r in rates])\n"), "error", "no-1x1"),
    ("aspp", _a(branches=_A_BRANCHES.replace("nn.Conv2d(c_in, c, 3", "nn.Conv2d(c, c, 3")), "error", "branch-in"),
    ("aspp", _a(project="        self.project = nn.Conv2d(c * len(rates), c, 1)\n"), "error", "project-channels"),
    ("aspp", _a(body="        ...\n"), "wrong", "dots-left"),
] + [("aspp", code, "wrong", key) for key, code in _A_WRONG.items()] + [
    ("deeplab", _D_START, "empty", None),
    ("deeplab", _D, "correct", None),
    # the dilation with keywords, and forward in fewer lines
    ("deeplab", _d(conv4="        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, dilation=2, padding=2)\n",
                   body="        x = self.pool(torch.relu(self.conv1(x)))\n        x = self.pool(torch.relu(self.conv2(x)))\n"
                        "        x = torch.relu(self.conv4(torch.relu(self.conv3(x))))\n        x = self.aspp(x)\n"
                        "        return self.up(self.score(x))\n"), "correct", None),
    ("deeplab", _d(body=_D_BODY.replace("self.score(self.aspp(x))", "self.aspp(x)")), "error", "no-score"),
    ("deeplab", _d(conv4="        self.conv4 = nn.Conv2d(32, 64, 3, padding=2, dilation=2)\n"), "error", "conv-channels"),
    ("deeplab", _d(body="        ...\n"), "wrong", "dots-left"),
] + [("deeplab", code, "wrong", key) for key, code in _D_WRONG.items()]


# --- the demo: the learner's DeepLab, trained on the class maps of their class_map ----------------------------------

_Stopped, _named = _FCN._Stopped, _FCN._named


def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def _demo(show=None):
    """For the demo: None when deeplab.py is not in the project yet; "classes" when the learner's class_map stops or
    does not give a class map for every scene; "deeplab" when their DeepLab, or the ASPP or the upsampler it uses,
    stops, does not give 4 scores for every pixel of 2 scenes, or trains to a loss that is not a number; "kind_iou"
    when their kind_iou stops or does not give 1 for a class map compared with itself; "slow" when the training takes
    too long; else a dict with the loss after every epoch ("curves"), the class maps of the SHOWN scenes after every
    epoch ("moments"), on the test scenes after the last epoch the share of pixels right and the IoU of every kind by
    their kind_iou, the parameters of their DeepLab, the modules the course's versions stood in for, and their trained
    DeepLab layer by layer on the scene of the worked example ("net", a netview.Trace). `show` hears the run while it
    trains."""
    if not _saved("deeplab"):
        return None
    names = {}
    for module, name, key in (("classes", "class_map", "classes"), ("kind_iou", "kind_iou", "kind_iou"),
                              ("deeplab", "DeepLab", "deeplab")):
        run = run_code(f"from cnn.{module} import {name}")
        if run.error is not None:
            return key
        names[name] = run.names[name]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    ss, X, _ = _FCN.scene_set("train")
    test_ss, Xt, _ = _FCN.scene_set("test")
    try:   # the learner's class maps: the targets of every training and test scene
        C = _FCN._their_classes(names["class_map"], ss)
        Ct = _FCN._their_classes(names["class_map"], test_ss)
    except Exception:
        return "classes"
    if C is None or Ct is None:
        return "classes"
    try:   # the learner's DeepLab on 2 scenes: 4 scores for every pixel
        model = names["DeepLab"]()
        with torch.no_grad():
            if not (isinstance(model, nn.Module)
                    and tuple(model(X[:2].float() / 255).shape) == (2, CLASSES, scenes.SIZE, scenes.SIZE)):
                return "deeplab"
    except Exception:
        return "deeplab"
    try:   # the learner's IoU of every kind for a class map against itself: 1
        ones = [float(names["kind_iou"](Ct[:4], Ct[:4], k)) for k in range(1, CLASSES)]
        if not all(math.isfinite(v) and abs(v - 1) < 1e-6 for v in ones):
            return "kind_iou"
    except Exception:
        return "kind_iou"
    torch.manual_seed(0)   # the same start as the course's run
    try:
        model = names["DeepLab"]()
    except Exception:
        return "deeplab"
    model.forward = _named(model.forward, "deeplab")   # an error in their network is theirs; one in the course's code is not
    iou_of = _named(names["kind_iou"], "kind_iou")
    shown_at = [TEST.index(seed) for seed in SHOWN]
    moments = []

    def heard(p):
        if "pred" in p:
            moments.append([p["pred"][i].numpy() for i in shown_at])
            p = {k: v for k, v in p.items() if k != "pred"}
        if show is not None:
            try:
                show(p)
            except Exception:   # the live chart is the course's own: it never stops the learner's run
                pass

    try:   # with_time_limit trains on one thread, as the course's run
        history = with_time_limit(lambda: _FCN.train(model, (X, C), (Xt, Ct), 0, EPOCHS, heard, iou_of), TRAIN_LIMIT)
    except Timeout:
        return "slow"
    except _Stopped as e:
        return e.key
    if not all(math.isfinite(h["train_loss"]) and math.isfinite(h["val_loss"]) for h in history):
        return "deeplab"
    last = history[-1]
    curves = {"train": [float(h["train_loss"]) for h in history], "val": [float(h["val_loss"]) for h in history]}
    net = netview.trace(model, Xt[TEST.index(SEED):TEST.index(SEED) + 1].float() / 255)   # the scene of the example
    return {"curves": curves, "moments": moments, "acc": last["acc"], "ious": last["ious"],
            "params": sum(p.numel() for p in model.parameters()), "stand_ins": stand_ins, "net": net}


def layer_frame(net, k, lang: str) -> str:
    """The slider over the layers of the learner's trained DeepLab: the network, with layer k (from 1) chosen and its
    feature maps under it."""
    return netview.picture(net, lang, chosen=int(k) - 1, uid=f"nv-deeplab-{int(k)}")


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file)."""
    texts = []
    for module in ("classes", "kind_iou", "upsample", "aspp", "deeplab"):
        path = projects.folder() / "cnn" / f"{module}.py"
        texts.append(path.read_text(encoding="utf-8") if path.exists() else None)
    return tuple(texts)


def demo(show=None):
    """The demo's result, worked out once for the project files as they are; `show` hears the run when it trains."""
    key = _project_texts()
    if key not in _DEMO_RUNS:
        _DEMO_RUNS.clear()
        _DEMO_RUNS[key] = _demo(show)
    return _DEMO_RUNS[key]
