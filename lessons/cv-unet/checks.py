# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-unet: DoubleConv, the block of a U-Net (two 3×3 convolutions, each with batch
normalization and ReLU); UNet, the course's U-Net, whose way up stacks every map with the map of the same size from
the way down; and mean_iou, the IoU of every class averaged over the classes. All three are saved in the learner's
project (work/projects/cnn/double_conv.py, unet.py and mean_iou.py) when right. The demo trains the learner's U-Net
on the class maps of their class_map of cv-fcn, and measures it with their mean_iou on new scenes. And the pictures:
the U-Net itself, and for the slider, what the course's U-Net gives the scene of the worked example with fewer skip
connections. Everything is computed from the course's own scenes (scenes.py): no download."""
import functools
import importlib.util
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
import torch.nn.functional as F
from torch import nn

from everything_ai import netview, paths, projects, scenes
from everything_ai.check import ClassBox, Mistake, ScenarioBox, Timeout, run_code, with_time_limit


def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_unet",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The scenes, their class maps, the training of cv-fcn, its FCN and its pictures of class maps
_FCN = _lesson_checks("cv-fcn")

# The demo makes the class maps with the learner's class_map of cv-fcn, and their mean_iou measures with their
# kind_iou; their UNet builds on their DoubleConv of this lesson. Where one is missing or broken, the course's version
# stands in, and the lesson says so.
BUILDS_ON = (("cnn", "classes", "cv-fcn", "class-map"), ("cnn", "kind_iou", "cv-fcn", "kind-iou"),
             ("cnn", "double_conv", "cv-unet", "double-conv"))
projects.builds_on(*BUILDS_ON)

SEED = _FCN.SEED              # the scene of the worked example: scenes.scene(18), as in cv-fcn
KINDS, CLASSES = _FCN.KINDS, _FCN.CLASSES
N_TRAIN, N_TEST, TRAIN, TEST = _FCN.N_TRAIN, _FCN.N_TEST, _FCN.TRAIN, _FCN.TEST
EPOCHS, LR, BATCH = _FCN.EPOCHS, _FCN.LR, _FCN.BATCH   # trained exactly as the course's FCN
CHANNELS = (8, 16, 32, 64)    # the channels of the blocks of the way down: at 128, 64, 32 and 16 pixels
SHOWN = _FCN.SHOWN            # the test scenes of the rewind
TRAIN_LIMIT = 1800            # seconds the demo gives the training: the note says half an hour

# The U-Nets the course compares, each trained with the seeds 0 to 4: how many skip connections it keeps, from the
# coarsest join up (3: all, the course's U-Net; 0: none), and whether its blocks have batch normalization.
VARIANTS = {
    "unet": (3, True),
    "skips2": (2, True),
    "skips1": (1, True),
    "skips0": (0, True),
    "plain": (3, False),
}
BY_SKIPS = {0: "skips0", 1: "skips1", 2: "skips2", 3: "unet"}


# --- the course's versions ------------------------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """The block of the course's U-Net, as the learner's: two 3×3 convolutions with padding 1, each followed by batch
    normalization and ReLU."""

    def __init__(self, a, b):
        super().__init__()
        self.conv1 = nn.Conv2d(a, b, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(b)
        self.conv2 = nn.Conv2d(b, b, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(b)

    def forward(self, x):
        x = torch.relu(self.norm1(self.conv1(x)))
        return torch.relu(self.norm2(self.conv2(x)))


class _Plain(nn.Module):
    """The block without batch normalization, as the paper's: two 3×3 convolutions, each followed by ReLU."""

    def __init__(self, a, b):
        super().__init__()
        self.conv1 = nn.Conv2d(a, b, 3, padding=1)
        self.conv2 = nn.Conv2d(b, b, 3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv2(torch.relu(self.conv1(x))))


class UNet(nn.Module):
    """The course's U-Net, as the learner's: the way down, blocks at 128, 64 and 32 pixels, each followed by a pooling
    of 2, and the bottom block at 16; the way up, a transposed convolution of 2 by 2 with stride 2 that makes the map
    twice as large with half the channels, the map of the same size from the way down stacked before it, and a block;
    a 1×1 convolution to a score for every class at every pixel. `skips` keeps the skip connections of the coarsest
    joins only (3: all of them), `block` makes the blocks."""

    def __init__(self, skips: int = 3, block=DoubleConv):
        super().__init__()
        self.skips = skips
        self.down = nn.ModuleList([block(3, 8), block(8, 16), block(16, 32), block(32, 64)])
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList([nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ConvTranspose2d(32, 16, 2, stride=2),
                                  nn.ConvTranspose2d(16, 8, 2, stride=2)])
        self.convs = nn.ModuleList([block(64 if skips > 0 else 32, 32), block(32 if skips > 1 else 16, 16),
                                    block(16 if skips > 2 else 8, 8)])
        self.score = nn.Conv2d(8, CLASSES, 1)

    def forward(self, x):
        kept = []
        for block in self.down[:3]:
            x = block(x)
            kept.append(x)
            x = self.pool(x)
        x = self.down[3](x)
        for i, (up, conv) in enumerate(zip(self.ups, self.convs)):
            skip = kept.pop()
            x = conv(torch.cat([skip, up(x)], dim=1) if i < self.skips else up(x))
        return self.score(x)


def mean_iou(pred, classes, n: int) -> float:
    """The mean IoU, as the learner's: the IoU of the pixels of every class from 0 to n - 1, background included,
    averaged over the classes."""
    return sum(_FCN.kind_iou(pred, classes, c) for c in range(n)) / n


# --- training and measuring -----------------------------------------------------------------------------------------

def make(variant: str) -> nn.Module:
    skips, norm = VARIANTS[variant]
    return UNet(skips, DoubleConv if norm else _Plain)


def train_course(variant: str = "unet", seed: int = 0, heard=None, epochs: int = EPOCHS) -> tuple:
    """A U-Net of the course, trained as the course's FCN is, with `seed` on one thread, as a check runs: (model,
    history)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(seed)
        model = make(variant)
        _, X, C = _FCN.scene_set("train")
        _, Xt, Ct = _FCN.scene_set("test")
        history = _FCN.train(model, (X, C), (Xt, Ct), seed, epochs, heard)
    finally:
        torch.set_num_threads(threads)
    return model, history


def class_ious(pred, C) -> list[float]:
    """The IoU of every class, background first, over all pixels of all scenes."""
    return [float(_FCN.kind_iou(pred, C, k)) for k in range(CLASSES)]


def measure_variant(variant: str, seed: int) -> dict:
    """Train one network and measure what TRAINED holds for it: on the test scenes the share of pixels right, the IoU
    of every class, background first, and the loss; with seed 0, the class map it gives the scene of the worked
    example, and for the course's U-Net its loss curves. "fcn" is the course's FCN of cv-fcn, trained again."""
    if variant == "fcn":
        model, history = _FCN.train_course("s8", seed)
    else:
        model, history = train_course(variant, seed)
    _, Xt, Ct = _FCN.scene_set("test")
    out = _FCN.outputs(model, Xt)
    pred = out.argmax(1)
    r = {"acc": round(float((pred == Ct).float().mean()), 4), "ious": [round(v, 4) for v in class_ious(pred, Ct)],
         "val": round(history[-1]["val_loss"], 4)}
    if seed == 0:
        r["map"] = _FCN.pack(pred[TEST.index(SEED)])
        if variant == "unet":
            r["curves"] = {"train": [round(h["train_loss"], 4) for h in history],
                           "val": [round(h["val_loss"], 4) for h in history]}
    return r


LONG = (3, 16)   # the seed whose U-Net mixes up two kinds after 8 epochs, and how many epochs it trains for "long"


def measure_long() -> dict:
    """The course's U-Net with seed LONG[0], trained for LONG[1] epochs: on the test scenes the share of pixels right
    and the IoU of every class, background first."""
    model, _ = train_course("unet", LONG[0], epochs=LONG[1])
    _, Xt, Ct = _FCN.scene_set("test")
    pred = _FCN.outputs(model, Xt).argmax(1)
    return {"acc": round(float((pred == Ct).float().mean()), 4), "ious": [round(v, 4) for v in class_ious(pred, Ct)]}


def confusion(variant: str, seed: int) -> list[list[int]]:
    """How many test pixels of every true class (rows, background first) the network gives every class (columns)."""
    model, _ = train_course(variant, seed)
    _, Xt, Ct = _FCN.scene_set("test")
    pred = _FCN.outputs(model, Xt).argmax(1)
    return [[int(((Ct == a) & (pred == b)).sum()) for b in range(CLASSES)] for a in range(CLASSES)]


def measure_trained(variants=None, seeds=range(5)) -> dict:
    """Everything TRAINED holds, measured again: every variant with every seed, the course's FCN, the U-Net of seed
    LONG[0] trained longer, and what it mixes up after 8 epochs. Takes three to ten minutes per training on one thread,
    about four hours in all."""
    out = {}
    for variant in variants or (*VARIANTS, "fcn"):
        runs = [measure_variant(variant, seed) for seed in seeds]
        out[variant] = {key: [r[key] for r in runs] for key in ("acc", "ious", "val")}
        for key in ("map", "curves"):
            if key in runs[0]:
                out[variant][key] = runs[0][key]
    if variants is None:
        out["long"] = measure_long()
        out["confusion"] = confusion("unet", LONG[0])
    return out


# --- the numbers ----------------------------------------------------------------------------------------------------

# The worked example's numbers, measured again by `worked()`; "background" is the share of background pixels of the
# test scenes, as cv-fcn measured it, so a network that says background everywhere reaches a mean IoU of a quarter of it.
REFERENCE = {
    "kept": [[1, 8, 128, 128], [1, 16, 64, 64], [1, 32, 32, 32]],
    "bottom": [1, 64, 16, 16],
    "up": [1, 32, 32, 32],
    "stacked": [1, 64, 32, 32],
    "nothing": [0.8925, 0.0, 0.0, 0.0],
    "nothing_mean": 0.2231,
    "mixed_acc": 0.9601,
    "mixed": [1.0, 0.5152, 0.0, 1.0],
    "mixed_mean": 0.6288,
    "background": _FCN.REFERENCE["background"],
}

# The course's U-Nets and its FCN, each trained with the seeds 0 to 4 by `measure_trained()` as the demo trains:
# on the 200 test scenes the share of pixels right, the IoU of every class (background, circles, squares,
# triangles) and the loss; with seed 0 the class map of the scene of the worked example, and the loss curves of the
# course's U-Net. "long": the course's U-Net with seed 3 trained for 16 epochs; "confusion": what the one of seed 3
# gives the test pixels of every class after 8 epochs (rows the true class, columns its answer, background first).
TRAINED = {
    "unet": {
        "acc": [0.9962, 0.9959, 0.9962, 0.9734, 0.9961],
        "ious": [[0.9999, 0.9639, 0.9253, 0.9215], [0.9999, 0.9651, 0.9156, 0.9142], [1.0, 0.9635, 0.9258, 0.9213], [0.9999, 0.9662, 0.6003, 0.0427], [1.0, 0.9648, 0.9231, 0.917]],
        "val": [0.0149, 0.0156, 0.0141, 0.0416, 0.0142],
        "map": "eNrt1TGOwjAQBdABi2YbV1yCihuEI6TI3IdLpN8mBUaCngK4xB6BhioNzSpab2yMQDCeifB269++ePQTSxOAnJycnJy0fJil4IZ9wrkxn4LHZ6T67I+cMiW4FtzyrgTXglt75Fx5D7x69eLBm4oaf/O1QSTa3c8j4oIYH3xrasSKGB980/TnkRgf/Jvy4u5nx08FlA1+Ntud95KcP+7rNZT7fq3zw8l7RRT0vkfS4eY/EdfBLxGH4F8x11evYw5F674ERl05h0nUYS44DPHRMC/fdUDi/h8zHeTxFTdh64cCJeNTdrx/wzL/if5ZWsE7wS3Pyi0/1tsk17YT3Ep+FJwt0O//TnCb4gKDwOL15aTkF+wrFLk=",
        "curves": {"train": [0.8022, 0.1006, 0.0451, 0.0318, 0.0252, 0.0198, 0.017, 0.0158], "val": [0.237, 0.0787, 0.0329, 0.0256, 0.0193, 0.018, 0.0153, 0.0149]},
    },
    "skips2": {
        "acc": [0.995, 0.9953, 0.9947, 0.9944, 0.9949],
        "ious": [[0.9991, 0.9587, 0.9082, 0.9058], [0.9991, 0.9615, 0.916, 0.9035], [0.999, 0.9561, 0.9069, 0.8923], [0.999, 0.9555, 0.8971, 0.8899], [0.9992, 0.9586, 0.9023, 0.904]],
        "val": [0.0185, 0.0174, 0.0186, 0.0211, 0.0198],
        "map": "eNrt1LENgzAQBdAjiCYNVZbIEmYECrxPlqBPQ4GnYCQGQLpgDMEi57sIp4r8C5rHHR+QDJCSkpKSEpereQhu2DusG/MUPLwj1u8/cjjhSnDkPRe8FBxx4Dxf3ARdCY6+90Q733VFrN+91w2xfvdOa2L97vrDle/z+LEA+j6zrsn5yzZ+9GXB6Ly13hAFV9eUw9s72svNW9phcx3w0nkXclCjvYY9X/wWdJci0rPvvD7r4LwKuntBOO0FW38tUAP/AO6MzfjxlH/MKPgkOPKc28OP9THKS5wER8kHwdkC8/k9Cc4WQNn5z49xvy8lJi8HTBk+",
    },
    "skips1": {
        "acc": [0.971, 0.9932, 0.9937, 0.9945, 0.9935],
        "ious": [[0.9967, 0.9584, 0.5937, 0.0], [0.9973, 0.9508, 0.8892, 0.8836], [0.9973, 0.9579, 0.8967, 0.8861], [0.9974, 0.9601, 0.9173, 0.8978], [0.9973, 0.956, 0.8963, 0.8763]],
        "val": [0.057, 0.0231, 0.0228, 0.02, 0.0232],
        "map": "eNrt1jEOgzAMBdCA1CU7e+ecgiMw1PfhEuwsSOBTNiQgEHX8kdJOzR8YeMQ4BBGMKSkpKSnJS8296paZkbNSI3r6CuTuS25+4Ba4A866W+BO9zqc5KTb8NA2noXuzj5J5Q+fqJUWdveFqBPKn/0lvRe7D0Qkrcvmk+drA3z4sg6/NnCqv5Dg4QZ9mGUYLjaoudk9lv90C9wXGFU33n2bQ9LduB6bpNfBH0mPyfWKxOd72809b5PeAI8TSH/iKnV6WwOd4o1aPtygKzvRn+UJfAbOeY72/ZpH4DNwRv8VyNUGHOsN5DpgAxguX0lO3r+e610=",
    },
    "skips0": {
        "acc": [0.9914, 0.9906, 0.9912, 0.9898, 0.9905],
        "ious": [[0.9956, 0.9466, 0.8688, 0.8479], [0.995, 0.9425, 0.8604, 0.8365], [0.9954, 0.9459, 0.8652, 0.8496], [0.9949, 0.9352, 0.8445, 0.8223], [0.9942, 0.9486, 0.8724, 0.8127]],
        "val": [0.0291, 0.0305, 0.0334, 0.0367, 0.0514],
        "map": "eNrt1TEOgzAMBVC3iKVLJi7RS4QjZKjvwyXYuzA0p+BIHADJhQQVpCR2RZZKzV8fdpwIJQAlJSUlJXm52o71m7VPwZd0gqe/+HW/bx4zLTjxXgmuaASwaSfeK8E170v54levPbbhdHv9C0PXh/oB8RFOv9dHXNOhvkfESDlN3pflgwHoUL+Wo0n3x4jD7kPU1cdd++gGnbvpIq4Eh82HlG//R9JhdYAm6UpwnzrTL9+5OevgvT3t/gDSV1zNjr8NaJg7smHbuwamvER/ljnP3ePB+ij4lOWKZsFJ8lFwdoH1/s5xkp2dT9g+CMdfkpU3RuoZhA==",
    },
    "plain": {
        "acc": [0.9394, 0.9336, 0.9785, 0.9396, 0.9336],
        "ious": [[0.9994, 0.5533, 0.0946, 0.0767], [0.9999, 0.4888, 0.0417, 0.0324], [0.9993, 0.8196, 0.6183, 0.6493], [0.9983, 0.5764, 0.2193, 0.0179], [0.9997, 0.491, 0.0235, 0.0378]],
        "val": [0.118, 0.1344, 0.0545, 0.1198, 0.1346],
        "map": "eNrt1TFOwzAUBuAXoyIlUrt1YWIBCY7AxBE64IAysTCQU+QIXRKhTl1SkYdEj9CrZMUD7AyYpEpEQvDviooF+R/zxfZzZL8Qubi4uLjsl7FMoI+khG/ULuWtxc1ztH75Sz+x+HRH/8mExUcWf8HuxdgnBXaVJ8hFnFebbjgc+sVKfvnV0Dnv+HD+Sd8H3/dcF10fLLDYqK5/XyB40u8p8Cnrj6Lr/QIO5FprVT27b33WH+9Hz4oTEianbK1VQh6bNuBHzEvglPFbidwP6/HS6JRyCT1IlySA09m8PmV3ucnFds9BYXI63tZpHN90EIt7ciefWdzcYvAFb6+IzS0t7hr0QFh+swDqsQLtzuV/prT4jcUji8f4v+5VzWcfH/Mj9MMNY2eGBQb6FRZwyriAI60y6JyukC9yyBRipgd3wv8wnzoMtzs=",
    },
    "fcn": {
        "acc": [0.9802, 0.9739, 0.9744, 0.9684, 0.9704],
        "ious": [[0.9892, 0.8725, 0.7125, 0.7261], [0.988, 0.8139, 0.6192, 0.6732], [0.9878, 0.8328, 0.6262, 0.652], [0.9883, 0.7788, 0.5268, 0.5995], [0.9877, 0.7892, 0.5652, 0.6282]],
        "val": [0.0604, 0.0771, 0.076, 0.0838, 0.084],
        "map": "eNrt1T0SgjAQBeAolTapPIKNl4AjUBAKTuBwCo5iyeQUnMcqB2AmxgVpzL51zFiZ134kbHbyo1ROTk5OTmLumAtrr8j3lsL6cfFB8JvgY6KzBV6AF4J7B10LXgru01wL7skt50XwCXgZXK1uui42/TY++BBzt3k7Qu9bi930Ffa2xm6gW2OayPoXN5Tmvf1r/xc38f7zHj6Yaf9yrp8eJuCc9mc4oKxrRyeAdTXRCYqvb8su0dVnXrN+EvxMXrF+4Jb/uuJweWEPwN8vHYaXLKou5z/jhHfZT9Dp8UnyGXpJt//37mWffuxJ9ZVCf6T+5qTkAX2NHLo=",
    },
    "long": {"acc": 0.9976, "ious": [1.0, 0.9766, 0.9538, 0.9452]},
    "confusion": [[2849451, 0, 4, 0], [29, 208927, 1808, 60], [18, 3842, 127730, 55], [367, 1569, 79313, 3627]],
}


def bottom_field() -> int:
    """How many pixels of the scene one place of the bottom map sees, along one side: every 3×3 convolution widens
    what a place sees by 2 places of its map, every pooling of 2 by 1, and each pooling makes a place of the map
    stand for twice as many pixels."""
    field, step = 1, 1
    for _ in range(3):
        field += 2 * 2 * step + step   # the two convolutions of a block, and the pooling after it
        step *= 2
    return field + 2 * 2 * step        # the bottom block


def mean_of(ious) -> float:
    return sum(ious) / len(ious)


def mious(variant: str) -> list[float]:
    """The mean IoU of every seed of a variant."""
    return [mean_of(run) for run in TRAINED[variant]["ious"]]


def middle(variant: str) -> list[float]:
    """The median over the seeds of a variant, the middle one of the five: of the mean IoU, and of the IoU of every
    kind, circles first."""
    runs = TRAINED[variant]["ious"]
    return [float(np.median(mious(variant)))] + [float(np.median([r[k] for r in runs])) for k in range(1, CLASSES)]


# --- the worked example -------------------------------------------------------------------------------------------

def worked() -> dict:
    """The numbers the cells of the worked example print, worked out as they do."""
    s = _FCN.example()
    torch.manual_seed(0)
    x = torch.tensor(s.image).permute(2, 0, 1).unsqueeze(0).float() / 255
    pool = nn.MaxPool2d(2)
    kept_maps = []
    for a, b in [(3, 8), (8, 16), (16, 32)]:
        conv1, norm1 = nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b)
        conv2, norm2 = nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b)
        x = torch.relu(norm2(conv2(torch.relu(norm1(conv1(x))))))
        kept_maps.append(x)
        x = pool(x)
    conv1, norm1 = nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64)
    conv2, norm2 = nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64)
    bottom = torch.relu(norm2(conv2(torch.relu(norm1(conv1(x))))))
    up = nn.ConvTranspose2d(64, 32, 2, stride=2)
    y = up(bottom)
    stacked = torch.cat([kept_maps[2], y], dim=1)
    classes = _FCN.class_map(s)

    def ious(pred):
        return [float(_FCN.kind_iou(pred, classes, c)) for c in range(CLASSES)]

    nothing = np.zeros_like(classes)
    mixed = classes.copy()
    mixed[classes == 2] = 1
    return {"kept": [list(m.shape) for m in kept_maps], "bottom": list(bottom.shape), "up": list(y.shape),
            "stacked": list(stacked.shape),
            "nothing": [round(v, 4) for v in ious(nothing)], "nothing_mean": round(mean_of(ious(nothing)), 4),
            "mixed_acc": round(float((mixed == classes).mean()), 4), "mixed": [round(v, 4) for v in ious(mixed)],
            "mixed_mean": round(mean_of(ious(mixed)), 4)}


# --- the pictures -------------------------------------------------------------------------------------------------

def _escape(text) -> str:
    import html
    return html.escape(str(text), quote=False)


def unet_picture(words: dict) -> str:
    """The course's U-Net, drawn as a U: on the left the way down, three blocks each followed by a pooling of 2; the
    bottom block; on the right the way up, each step a transposed convolution that makes the map twice as large, the
    map of the same size from the way down stacked before it (dashed), and a block; at the end a 1×1 convolution to
    the scores of the 4 classes. Every box is a stack of feature maps: its height grows with the side of the map, its
    width with the channels, and the number above it is its channels."""
    sides = (128, 64, 32, 16)
    ys = (66, 146, 210, 258)                       # the middle of each level
    heights = {128: 44, 64: 30, 32: 20, 16: 14}
    widths = {4: 7, 8: 9, 16: 12, 32: 16, 64: 22}
    lx, rx, bx, sx = 112, 236, 174, 298             # the middles of the left boxes, the right ones, the bottom, the scores
    ink, ink2, ink3 = "var(--ea-ink)", "var(--ea-ink-2)", "var(--ea-ink-3)"
    fill = "fill:var(--ea-hidden-wash);stroke:var(--ea-hidden);stroke-width:1.5"
    copied = "fill:var(--ea-page);stroke:var(--ea-hidden);stroke-width:1.5;stroke-dasharray:3 2"
    text = "font-size:14px;fill:" + ink
    small = "font-size:13px;fill:" + ink2
    parts = []

    def arrow(x0, y0, x1, y1, dash=False):
        d = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / d, (y1 - y0) / d
        hx, hy = x1 - 6 * ux, y1 - 6 * uy
        style = f"stroke:{ink3};stroke-width:1.5" + (";stroke-dasharray:4 3" if dash else "")
        parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{hx:.1f}" y2="{hy:.1f}" style="{style}"/>'
                     f'<path d="M{hx - 4 * uy:.1f} {hy + 4 * ux:.1f} L{x1:.1f} {y1:.1f} L{hx + 4 * uy:.1f} {hy - 4 * ux:.1f} Z" style="fill:{ink3}"/>')

    def box(x, y, c, side, style=fill):
        w, h = widths[c], heights[side]
        parts.append(f'<rect x="{x - w / 2:.1f}" y="{y - h / 2:.1f}" width="{w}" height="{h}" rx="2" style="{style}"/>')

    def label(x, y, words_, anchor="middle", style=small):
        parts.append(f'<text x="{x:.1f}" y="{y:.1f}" style="{style};text-anchor:{anchor}">{_escape(words_)}</text>')

    label(lx, 16, words["down"], style=text + ";font-weight:600")
    label(rx, 16, words["up"], style=text + ";font-weight:600")
    for i, (side, y) in enumerate(zip(sides, ys)):
        label(4, y + 5, f"{side} × {side}", "start")
        h = heights[side]
        if i == 3:   # the bottom block, in the middle
            w = widths[64]
            box(bx, y, 64, side)
            label(bx, y - h / 2 - 6, "64")
            top32 = ys[2] + heights[32] / 2
            arrow(lx, top32 + 3, bx - w / 2 - 3, y - 2)                 # pooling into the bottom
            up32 = rx + widths[32] / 2 + 1                               # the map from below, at 32 by 32
            arrow(bx + w / 2 + 3, y - 2, up32, top32 + 3)               # twice as large out of the bottom
            mid = (top32 + 3 + y - 2) / 2
            label((lx + bx - w / 2) / 2 - 7, mid + 5, words["pool"], "end")
            label((up32 + bx + w / 2) / 2 + 8, mid + 5, "×2", "start")
            continue
        c = CHANNELS[i]
        w = widths[c]
        box(lx, y, c, side)
        label(lx, y - h / 2 - 6, str(c))
        box(rx - w / 2 - 1, y, c, side, copied)                         # the copy from the way down, stacked first
        box(rx + w / 2 + 1, y, c, side)                                  # the map from below, twice as large
        label(rx, y - h / 2 - 6, f"{2 * c} → {c}")
        arrow(lx + w / 2 + 3, y, rx - w - 4, y, dash=True)              # the skip connection
        label((lx + rx - w) / 2, y - 6, words["stack"])
        if i < 2:
            nxt = ys[i + 1] - heights[sides[i + 1]] / 2 - 20             # above the number of the box below
            arrow(lx, y + h / 2 + 3, lx, nxt)                            # pooling
            arrow(rx + w / 2 + 1, nxt, rx + w / 2 + 1, y + h / 2 + 3)    # twice as large, into the map from below
            mid = (y + h / 2 + nxt) / 2
            label(lx - 7, mid + 5, words["pool"], "end")
            label(rx + w / 2 + 9, mid + 5, "×2", "start")
    arrow(rx + widths[8] + 4, ys[0], sx - widths[4] / 2 - 3, ys[0])    # the 1×1 convolution to the scores
    label((rx + widths[8] + sx) / 2, ys[0] - 6, "1×1")
    box(sx, ys[0], 4, 128)
    label(sx, ys[0] - heights[128] / 2 - 6, "4")
    width, height = 310, ys[3] + 26
    for j, line in enumerate(words["key"]):
        label(4, height + 18 * j, line, "start")
    height += 18 * (len(words["key"]) - 1) + 8
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:min(40rem,{width}px)">'
            f'<svg viewBox="0 0 {width} {height}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{_escape(words["alt"])}</title>{"".join(parts)}</svg></figure>')


def skips_frame(skips: int, words: dict) -> tuple[str, dict]:
    """For the slider: the class map that the course's U-Net with that many skip connections (seed 0) gives the
    scene of the worked example, next to the true class map; and on the test scenes the mean IoU and the IoU of every
    kind of seed 0, and the median mean IoU over the seeds."""
    variant = BY_SKIPS[skips]
    pred = _FCN.map_svg(_FCN.unpack(TRAINED[variant]["map"], scenes.SIZE), words["title"].format(skips=skips), 200)
    truth = _FCN.map_svg(_FCN.class_map(_FCN.example()), words["truth"], 200)
    stop = {"miou": mious(variant)[0], "ious": TRAINED[variant]["ious"][0][1:], "middle": middle(variant)[0]}
    return _FCN.side_by_side([(pred, 200), (truth, 200)], words["alt"].format(skips=skips)) + _FCN.key_html(words["kinds"]), stop


def variants_table(words: dict, lang: str, rows=("unet", "skips2", "skips1", "skips0", "plain", "fcn")) -> str:
    """The course's networks side by side, as a Markdown table: for each, on the test scenes over the seeds 0 to 4,
    the median mean IoU, the lowest mean IoU, and the median IoU of every kind."""
    lines = ["| " + " | ".join(words["header"]) + " |", "|" + "---|" * len(words["header"])]
    for variant in rows:
        mid = middle(variant)
        cells = [words["names"][variant], _FCN.number(mid[0], 2, lang), _FCN.number(min(mious(variant)), 2, lang)] + \
            [_FCN.number(v, 2, lang) for v in mid[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def params(variant: str) -> int:
    return sum(p.numel() for p in (_FCN.FCN(8) if variant == "fcn" else make(variant)).parameters())


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


# --- exercise 2: DoubleConv -------------------------------------------------------------------------------------------

_B_HEAD = ("import torch\nfrom torch import nn\n\n\nclass DoubleConv(nn.Module):\n"
           "    # two 3x3 convolutions with padding 1, each followed by batch normalization and ReLU:\n"
           "    # a channels in, b channels out, and the map keeps its size\n"
           "    def __init__(self, a, b):\n        super().__init__()\n")
_B_INIT = ("        self.conv1 = nn.Conv2d(a, b, 3, padding=1)\n        self.norm1 = nn.BatchNorm2d(b)\n"
           "        self.conv2 = nn.Conv2d(b, b, 3, padding=1)\n        self.norm2 = nn.BatchNorm2d(b)\n")
_B_FORWARD = "\n    def forward(self, x):\n"
_B_BODY = "        x = torch.relu(self.norm1(self.conv1(x)))\n        return torch.relu(self.norm2(self.conv2(x)))\n"
_B_START = _B_HEAD + "        ...\n" + _B_FORWARD + "        ...\n"


def _b(init=_B_INIT, body=_B_BODY) -> str:
    return _B_HEAD + init + _B_FORWARD + body


_B = _b()
_B_SCENARIOS = [
    # 3 channels in, 8 out, as the first block of the course's U-Net
    "torch.manual_seed(0)\nblock = DoubleConv(3, 8)\nx = torch.rand(2, 3, 16, 16)\nwith torch.no_grad():\n    y = block(x)\n"
    "[list(y.shape), round(float(y.mean()), 4), round(float(y[1, :, 4, 7].sum()), 4), sum(p.numel() for p in block.parameters())]",
    # other channels, a map that is not square, and the share of values ReLU makes 0
    "torch.manual_seed(1)\nblock = DoubleConv(5, 4)\nx = torch.randn(3, 5, 7, 9)\nwith torch.no_grad():\n    y = block(x)\n"
    "[list(y.shape), round(float(y[2, 3].sum()), 4), round(float((y == 0).float().mean()), 4)]",
]
_B_WRONG = {
    "no-padding": _b(init=_B_INIT.replace(", padding=1", "")),
    "no-norm": _b(body="        x = torch.relu(self.conv1(x))\n        return torch.relu(self.conv2(x))\n"),
    "relu-first": _b(body="        x = self.norm1(torch.relu(self.conv1(x)))\n        return self.norm2(torch.relu(self.conv2(x)))\n"),
    "no-last-relu": _b(body="        x = torch.relu(self.norm1(self.conv1(x)))\n        return self.norm2(self.conv2(x))\n"),
    "one-conv": _b(body="        return torch.relu(self.norm1(self.conv1(x)))\n"),
}
_B_MISTAKES = [
    Mistake("norm-channels", error=lambda e: isinstance(e, RuntimeError) and "running_mean should contain" in str(e),
            en="Python stops in a batch normalization: it was made for another number of channels than the convolution before it gives. Both convolutions give `b` channels, so both get `nn.BatchNorm2d(b)`.",
            nl="Python stopt in een batch normalization: die is gemaakt voor een ander aantal channels dan de convolutie ervoor geeft. Allebei de convoluties geven `b` channels, dus allebei krijgen ze `nn.BatchNorm2d(b)`."),
    Mistake("conv2-in", error=_channels,
            en="Python stops in a convolution: it gets another number of channels than it was made for. The first convolution makes `a` channels into `b`; the second gets those `b`, and gives `b` again: `nn.Conv2d(b, b, 3, padding=1)`.",
            nl="Python stopt in een convolutie: die krijgt een ander aantal channels dan waarvoor hij gemaakt is. De eerste convolutie maakt van `a` channels er `b`; de tweede krijgt die `b`, en geeft weer `b`: `nn.Conv2d(b, b, 3, padding=1)`."),
    Mistake("no-padding", wrong=_B_WRONG["no-padding"],
            en=_K_EN + ": the shape comes first, and your map gets smaller. A 3×3 convolution without padding loses a pixel on every side; `padding=1` keeps the map as large as it was, so that the map of the way down is as large as the one it is stacked with later.",
            nl=_K_NL + ": de shape staat voorop, en je map wordt kleiner. Een 3×3-convolutie zonder padding verliest aan elke kant een pixel; `padding=1` houdt de map zo groot als hij was, zodat de map van de weg omlaag straks even groot is als die waarmee hij gestapeld wordt."),
    Mistake("no-norm", wrong=_B_WRONG["no-norm"],
            en=_K_EN + ": your block skips the batch normalization. After each convolution comes `self.norm1` or `self.norm2`, and then ReLU: `torch.relu(self.norm1(self.conv1(x)))`.",
            nl=_K_NL + ": je blok slaat de batch normalization over. Na elke convolutie komt `self.norm1` of `self.norm2`, en dan ReLU: `torch.relu(self.norm1(self.conv1(x)))`."),
    Mistake("relu-first", wrong=_B_WRONG["relu-first"],
            en=_K_EN + ": your ReLU comes before the batch normalization. As in [Batch normalization](/course/open/nn-batchnorm?lang=en), it goes between the convolution and the ReLU: convolution, batch normalization, ReLU.",
            nl=_K_NL + ": je ReLU komt voor de batch normalization. Zoals in [Batch normalization](/course/open/nn-batchnorm?lang=nl) zit die tussen de convolutie en de ReLU: convolutie, batch normalization, ReLU."),
    Mistake("no-last-relu", wrong=_B_WRONG["no-last-relu"],
            en=_K_EN + ": your block gives negative numbers, because the second convolution has no ReLU after it. Each convolution is followed by batch normalization and ReLU, the second one too.",
            nl=_K_NL + ": je blok geeft negatieve getallen, omdat de tweede convolutie geen ReLU na zich heeft. Elke convolutie krijgt batch normalization en ReLU na zich, de tweede ook."),
    Mistake("one-conv", wrong=_B_WRONG["one-conv"],
            en=_K_EN + ": `forward` uses only the first convolution. A block has two: the output of the first goes into the second.",
            nl=_K_NL + ": `forward` gebruikt alleen de eerste convolutie. Een blok heeft er twee: de output van de eerste gaat de tweede in."),
]


# --- exercise 3: UNet -------------------------------------------------------------------------------------------------

_N_HEAD = ("import torch\nfrom torch import nn\nfrom cnn.double_conv import DoubleConv\n\n\nclass UNet(nn.Module):\n"
           "    def __init__(self):\n        super().__init__()\n"
           "        # the way down: a block at 128, 64 and 32 pixels, each followed by a pooling of 2,\n"
           "        # and the bottom block at 16\n"
           "        self.down = nn.ModuleList([DoubleConv(3, 8), DoubleConv(8, 16), DoubleConv(16, 32),\n"
           "                                   DoubleConv(32, 64)])\n"
           "        self.pool = nn.MaxPool2d(2)\n"
           "        # the way up: each a map twice as large with half the channels, at 32, 64 and 128 pixels\n"
           "        self.ups = nn.ModuleList([nn.ConvTranspose2d(64, 32, 2, stride=2),\n"
           "                                  nn.ConvTranspose2d(32, 16, 2, stride=2),\n"
           "                                  nn.ConvTranspose2d(16, 8, 2, stride=2)])\n"
           "        # after every skip connection, a block\n")
_N_CONVS = "        self.convs = nn.ModuleList([DoubleConv(64, 32), DoubleConv(32, 16), DoubleConv(16, 8)])\n"
_N_SCORE = "        # a score for every class at every pixel\n        self.score = nn.Conv2d(8, 4, 1)\n"
_N_FORWARD = ("\n    def forward(self, x):\n"
              "        # x: (N, 3, H, W), H and W multiples of 8; give back the scores, (N, 4, H, W)\n")
_N_DOWN = ("        kept = []\n        for i in range(3):\n            x = self.down[i](x)\n            kept.append(x)\n"
           "            x = self.pool(x)\n        x = self.down[3](x)\n")
_N_UP = "        for i in range(3):\n            x = self.convs[i](torch.cat([kept[2 - i], self.ups[i](x)], dim=1))\n"
_N_RETURN = "        return self.score(x)\n"
_N_START = _N_HEAD + "        self.convs = ...\n" + _N_SCORE + _N_FORWARD + "        ...\n"


def _n(convs=_N_CONVS, down=_N_DOWN, up=_N_UP, ret=_N_RETURN) -> str:
    return _N_HEAD + convs + _N_SCORE + _N_FORWARD + down + up + ret


_N = _n()
_N_SCENARIOS = [
    # 2 scenes of 32 by 32
    "torch.manual_seed(0)\nnet = UNet()\nx = torch.rand(2, 3, 32, 32)\nwith torch.no_grad():\n    out = net(x)\n"
    "[list(out.shape), round(float(out[0, :, 9, 21].sum()), 4), round(float(out[1].mean()), 4), sum(p.numel() for p in net.parameters())]",
    # 1 scene of 48 by 64
    "torch.manual_seed(1)\nnet = UNet()\nx = torch.rand(1, 3, 48, 64)\nwith torch.no_grad():\n    out = net(x)\n"
    "[list(out.shape), round(float(out[0, 2, 47, 63]), 4), round(float(out[0, :, 0, 0].sum()), 4)]",
]
_N_WRONG = {
    "order": _n(up="        for i in range(3):\n            x = self.convs[i](torch.cat([self.ups[i](x), kept[2 - i]], dim=1))\n"),
    "no-score": _n(ret="        return x\n"),
}


def _sizes(e: BaseException) -> tuple[int, int] | None:
    """torch.cat stopped because its maps are not the same size: the size of the first map, and of the one that
    differs; None for any other error."""
    m = re.search(r"Sizes of tensors must match except in dimension \d+\. Expected size (\d+) but got size (\d+)", str(e))
    return (int(m.group(1)), int(m.group(2))) if isinstance(e, RuntimeError) and m else None


def _half_from_below(e: BaseException) -> bool:
    """The map from below is half as large as the kept one: it was not upsampled first, or the bottom block was
    pooled too."""
    sizes = _sizes(e)
    return sizes is not None and sizes[0] == 2 * sizes[1]


def _kept_half(e: BaseException) -> bool:
    """The kept map is half as large as the one from below: it was kept after its pooling, or nothing was pooled."""
    sizes = _sizes(e)
    return sizes is not None and sizes[1] == 2 * sizes[0]


_N_MISTAKES = [
    Mistake("not-stacked", error=lambda e: _channels(e, want=64, got=32),
            en="Python stops in the first block of the way up: it was made for 64 channels, and gets 32. Stack the map of the way down with the upsampled one along the channels, `torch.cat([kept[2 - i], self.ups[i](x)], dim=1)`, so that 32 + 32 = 64 channels come in. Adding them, as a residual connection does, keeps 32; so does stacking along `dim=0`, which stacks the scenes of the batch.",
            nl="Python stopt in het eerste blok van de weg omhoog: dat is gemaakt voor 64 channels, en krijgt er 32. Stapel de map van de weg omlaag met de opgeschaalde langs de channels, `torch.cat([kept[2 - i], self.ups[i](x)], dim=1)`, zodat er 32 + 32 = 64 channels binnenkomen. Optellen, zoals een residual connection doet, houdt het op 32; stapelen langs `dim=0` ook, want dat stapelt de scènes van de batch."),
    Mistake("pool-bottom", error=lambda e: _channels(e, want=64, got=96) or _half_from_below(e),
            en="Python stops on the way up: the map that comes up from below is half as large as the kept map it is stacked with. Each step up first makes it twice as large, `self.ups[i](x)`, and then stacks it. And the bottom block has no pooling after it, and its map is not kept: the loop goes over the first three blocks, `range(3)`, and then `x = self.down[3](x)`.",
            nl="Python stopt op de weg omhoog: de map die van onderen komt, is half zo groot als de bewaarde map waarmee hij gestapeld wordt. Elke stap omhoog maakt hem eerst twee keer zo groot, `self.ups[i](x)`, en stapelt hem dan. En het onderste blok heeft geen pooling na zich, en zijn map wordt niet bewaard: de loop gaat over de eerste drie blokken, `range(3)`, en daarna `x = self.down[3](x)`."),
    Mistake("kept-after-pool", error=_kept_half,
            en="Python stops at `torch.cat`: the kept map is half as large as the upsampled one. Keep each map before its pooling: `kept.append(x)` comes before `x = self.pool(x)`, and each of the first three blocks is followed by `self.pool`.",
            nl="Python stopt bij `torch.cat`: de bewaarde map is half zo groot als de opgeschaalde. Bewaar elke map vóór zijn pooling: `kept.append(x)` komt voor `x = self.pool(x)`, en na elk van de eerste drie blokken komt `self.pool`."),
    Mistake("convs-channels", error=_channels,
            en="Python stops in a block of the way up: it gets another number of channels than it was made for. After a skip connection, the blocks get 32 + 32 = 64, 16 + 16 = 32 and 8 + 8 = 16 channels, and give 32, 16 and 8: `DoubleConv(64, 32), DoubleConv(32, 16), DoubleConv(16, 8)`.",
            nl="Python stopt in een blok van de weg omhoog: dat krijgt een ander aantal channels dan waarvoor het gemaakt is. Na een skip connection krijgen de blokken 32 + 32 = 64, 16 + 16 = 32 en 8 + 8 = 16 channels, en geven ze 32, 16 en 8: `DoubleConv(64, 32), DoubleConv(32, 16), DoubleConv(16, 8)`."),
    Mistake("wrong-kept", error=lambda e: _sizes(e) is not None,
            en="Python stops at `torch.cat`: the two maps are not the same size. Each step of the way up needs the map of the way down that is as large as the upsampled one: the last one kept first, `kept[2 - i]`, or `kept[-1 - i]`.",
            nl="Python stopt bij `torch.cat`: de twee maps zijn niet even groot. Elke stap van de weg omhoog heeft de map van de weg omlaag nodig die even groot is als de opgeschaalde: de laatst bewaarde eerst, `kept[2 - i]`, of `kept[-1 - i]`."),
    Mistake("order", wrong=_N_WRONG["order"],
            en=_K_EN + ": your network stacks the upsampled map first. The course stacks the map of the way down first, as the paper's figure draws it: `torch.cat([kept[2 - i], self.ups[i](x)], dim=1)`. The order decides which weights of the block see which channels, so the numbers of the check change with it.",
            nl=_K_NL + ": je netwerk stapelt de opgeschaalde map eerst. De course stapelt de map van de weg omlaag eerst, zoals de figuur van de paper hem tekent: `torch.cat([kept[2 - i], self.ups[i](x)], dim=1)`. De volgorde bepaalt welke weights van het blok welke channels zien, dus de getallen van de check veranderen mee."),
    Mistake("no-score", wrong=_N_WRONG["no-score"],
            en=_K_EN + ": your network gives the 8 channels of the last block. The scores come from the 1×1 convolution after it: `return self.score(x)`.",
            nl=_K_NL + ": je netwerk geeft de 8 channels van het laatste blok. De scores komen uit de 1×1-convolutie erna: `return self.score(x)`."),
]


# --- exercise 4: mean_iou ---------------------------------------------------------------------------------------------

_M_HEAD = ("from cnn.kind_iou import kind_iou\n\n\ndef mean_iou(pred, classes, n):\n"
           "    # pred and classes: the class of every pixel, of one scene or of many; n: the number of classes\n"
           "    # give back the IoU of every class from 0 to n - 1, background included, averaged over the n classes\n")
_M_START = _M_HEAD + "    ...\n"
_M_ZERO = "    total = 0\n"
_M_LOOP = "    for c in range(n):\n"
_M_ADD = "        total += kind_iou(pred, classes, c)\n"
_M_RETURN = "    return total / n\n"


def _m(zero=_M_ZERO, loop=_M_LOOP, add=_M_ADD, ret=_M_RETURN) -> str:
    return _M_HEAD + zero + loop + add + ret


_M = _m()
_MIXED = ("import numpy as np\nfrom everything_ai.scenes import scene\n\ns = scene(18)\n"
          "classes = np.zeros((128, 128), dtype=int)\nfor shape in s.shapes:\n    classes[shape.mask] = shape.label + 1\n")
_M_SCENARIOS = [
    # the scene of the worked example with every square called a circle
    _MIXED + "mixed = classes.copy()\nmixed[classes == 2] = 1\nround(float(mean_iou(mixed, classes, 4)), 4)",
    # a network that says background everywhere
    _MIXED + "nothing = np.zeros((128, 128), dtype=int)\nround(float(mean_iou(nothing, classes, 4)), 4)",
    # 3 classes at random in tensors of 2 scenes, a third of every scene changed
    "import torch\n\ntorch.manual_seed(0)\nclasses = torch.randint(0, 3, (2, 20, 30))\npred = classes.clone()\n"
    "pred[:, :, :10] = torch.randint(0, 3, (2, 20, 10))\nround(float(mean_iou(pred, classes, 3)), 4)",
]
_M_WRONG = {
    "kinds-only": _m(loop="    for c in range(1, n):\n", ret="    return total / (n - 1)\n"),
    "no-background": _m(loop="    for c in range(1, n):\n"),
    "fixed-4": _m(ret="    return total / 4\n"),
    "pooled": _m(zero="    overlap = 0\n    union = 0\n",
                 add="        overlap += ((pred == c) & (classes == c)).sum()\n        union += ((pred == c) | (classes == c)).sum()\n",
                 ret="    return overlap / union\n"),
}
_M_MISTAKES = [
    Mistake("kinds-only", wrong=_M_WRONG["kinds-only"],
            en=_S_EN + ": you average over the kinds only. Mean IoU counts background as a class too: the IoU of every class from 0 to n − 1, `range(n)`, divided by `n`.",
            nl=_S_NL + ": je neemt het gemiddelde over alleen de soorten. Mean IoU telt achtergrond ook als klasse: de IoU van elke klasse van 0 tot en met n − 1, `range(n)`, gedeeld door `n`."),
    Mistake("no-background", wrong=_M_WRONG["no-background"],
            en=_S_EN + ": your loop starts at class 1, so the IoU of background is missing from the sum, though you divide by all `n` classes. Start at 0: `range(n)`.",
            nl=_S_NL + ": je loop begint bij klasse 1, dus de IoU van achtergrond ontbreekt in de som, terwijl je door alle `n` klassen deelt. Begin bij 0: `range(n)`."),
    Mistake("fixed-4", wrong=_M_WRONG["fixed-4"],
            en=_S_EN + ": you divide by 4, the classes of the course's scenes. The check also gives 3 classes: divide by `n`.",
            nl=_S_NL + ": je deelt door 4, de klassen van de scènes van de course. De check geeft ook 3 klassen: deel door `n`."),
    Mistake("pooled", wrong=_M_WRONG["pooled"],
            en=_S_EN + ": you add the overlaps and the unions of all classes up before you divide, so the large class, background, decides almost everything. Mean IoU divides for every class on its own, `kind_iou(pred, classes, c)`, and then takes the mean of those n numbers.",
            nl=_S_NL + ": je telt de overlappen en de unions van alle klassen op voordat je deelt, dus de grote klasse, achtergrond, beslist bijna alles. Mean IoU deelt voor elke klasse apart, `kind_iou(pred, classes, c)`, en neemt dan het gemiddelde van die n getallen."),
]


# --- the exercises ----------------------------------------------------------------------------------------------------

def _scene_batch() -> torch.Tensor:
    """The scene of the worked example as a batch of one: what the picture under exercises 2 and 3 sends through the
    learner's block or network, redrawn as they type (netview.network)."""
    return torch.tensor(scenes.scene(SEED).image).permute(2, 0, 1).unsqueeze(0).float() / 255


EXERCISES = {
    "double-conv": ClassBox(
        name="double-conv",
        start=_B_START,
        solution=_B,
        scenarios=_B_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "double_conv"),
        picture=netview.network("DoubleConv", _scene_batch, 3, 8, chosen=1, scores=False),
        first=[_DOTS],
        mistakes=_B_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: in `__init__` the two convolutions and their batch normalizations, and in `forward` convolution, batch normalization and ReLU, twice.",
                      "nl": "Vervang de twee `...`: in `__init__` de twee convoluties en hun batch normalizations, en in `forward` convolutie, batch normalization en ReLU, twee keer."},
            "wrong": {"en": _K_EN + ": the shape of what your block gives, then a mean, a sum, and the number of parameters. Does each convolution have a kernel of 3 and `padding=1`, the first from `a` channels to `b` and the second from `b` to `b`, each followed by its batch normalization and ReLU?",
                      "nl": _K_NL + ": de shape van wat je blok geeft, dan een gemiddelde, een som, en het aantal parameters. Heeft elke convolutie een kernel van 3 en `padding=1`, de eerste van `a` channels naar `b` en de tweede van `b` naar `b`, elk gevolgd door zijn batch normalization en ReLU?"},
            "almost": {"en": "Right for 3 channels in and 8 out. But " + _K_EN[0].lower() + _K_EN[1:] + ". Does your block work for any `a` and `b`, and for maps that are not square?",
                       "nl": "Goed voor 3 channels erin en 8 eruit. Maar " + _K_NL[0].lower() + _K_NL[1:] + ". Werkt je blok voor elke `a` en `b`, en voor maps die niet vierkant zijn?"},
            "correct": {"en": "Your `DoubleConv` is right: two convolutions, each with batch normalization and ReLU, and the map keeps its size. It is now in your project, as `{saved}`.",
                        "nl": "Je `DoubleConv` klopt: twee convoluties, elk met batch normalization en ReLU, en de map houdt zijn grootte. Hij staat nu in je project, als `{saved}`."},
            "kept": _kept("DoubleConv"),
        },
    ),
    "unet": ClassBox(
        name="unet",
        start=_N_START,
        solution=_N,
        scenarios=_N_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "unet"),
        picture=netview.network("UNet", _scene_batch),
        first=[_DOTS],
        mistakes=_N_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: `self.convs`, the three blocks after the skip connections, and `forward`, the way down with every map kept, and the way up with every map stacked.",
                      "nl": "Vervang de twee `...`: `self.convs`, de drie blokken na de skip connections, en `forward`, de weg omlaag met elke map bewaard, en de weg omhoog met elke map gestapeld."},
            "wrong": {"en": _K_EN + ": the shape of the scores, then a few of them, and the number of parameters. Does the way down keep the map of each of the first three blocks before its pooling, and does each step of the way up stack the map of the way down of its size first, then the upsampled one, before its block?",
                      "nl": _K_NL + ": de shape van de scores, dan een paar ervan, en het aantal parameters. Bewaart de weg omlaag de map van elk van de eerste drie blokken vóór zijn pooling, en stapelt elke stap van de weg omhoog eerst de map van de weg omlaag van zijn grootte, dan de opgeschaalde, voor zijn blok?"},
            "almost": {"en": "Right for 2 scenes of 32 by 32. But " + _K_EN[0].lower() + _K_EN[1:] + ". Does your network work for one scene, and for scenes that are not square?",
                       "nl": "Goed voor 2 scènes van 32 bij 32. Maar " + _K_NL[0].lower() + _K_NL[1:] + ". Werkt je netwerk voor één scène, en voor scènes die niet vierkant zijn?"},
            "correct": {"en": "Your `UNet` is right: down to 16 by 16 and back up, with a skip connection at every size, and 4 scores for every pixel, with 121,556 parameters. It is now in your project, as `{saved}`.",
                        "nl": "Je `UNet` klopt: omlaag naar 16 bij 16 en weer omhoog, met een skip connection bij elke grootte, en 4 scores voor elke pixel, met 121.556 parameters. Hij staat nu in je project, als `{saved}`."},
            "kept": _kept("UNet"),
        },
    ),
    "mean-iou": ScenarioBox(
        name="mean-iou",
        start=_M_START,
        solution=_M,
        scenarios=_M_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "mean_iou"),
        first=[_DOTS],
        mistakes=_M_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: the IoU of every class from 0 to n − 1 with `kind_iou`, added up, and divided by `n`.",
                      "nl": "Vervang de `...` door je code: de IoU van elke klasse van 0 tot en met n − 1 met `kind_iou`, opgeteld, en gedeeld door `n`."},
            "wrong": {"en": _S_EN + ". Do you add `kind_iou(pred, classes, c)` for every class `c` from 0 to n − 1, background included, and divide by `n`?",
                      "nl": _S_NL + ". Tel je `kind_iou(pred, classes, c)` op voor elke klasse `c` van 0 tot en met n − 1, achtergrond meegeteld, en deel je door `n`?"},
            "almost": {"en": "Right for the 4 classes of the course's scenes. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your function work for any number of classes?",
                       "nl": "Goed voor de 4 klassen van de scènes van de course. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je functie voor elk aantal klassen?"},
            "correct": {"en": "Your `mean_iou` is right: every class counts the same, however few pixels it has. It is now in your project, as `{saved}`.",
                        "nl": "Je `mean_iou` klopt: elke klasse telt even zwaar, hoe weinig pixels ze ook heeft. Hij staat nu in je project, als `{saved}`."},
            "kept": _kept("mean_iou"),
        },
    ),
}

# The lesson is done when these are correct. "stack-channels" is the first choice, "why-skips" the open question.
REQUIRED = ["stack-channels", "double-conv", "unet", "mean-iou", "why-skips"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("double-conv", _B_START, "empty", None),
    ("double-conv", _B, "correct", None),
    # the layers in another order of lines, and forward in one line
    ("double-conv", _b(init="        self.conv1 = nn.Conv2d(a, b, kernel_size=3, padding=1)\n        self.conv2 = nn.Conv2d(b, b, kernel_size=3, padding=1)\n"
                            "        self.norm1 = nn.BatchNorm2d(b)\n        self.norm2 = nn.BatchNorm2d(b)\n",
                       body="        return torch.relu(self.norm2(self.conv2(torch.relu(self.norm1(self.conv1(x))))))\n"), "correct", None),
    ("double-conv", _b(init=_B_INIT.replace("self.norm1 = nn.BatchNorm2d(b)", "self.norm1 = nn.BatchNorm2d(a)")), "error", "norm-channels"),
    ("double-conv", _b(init=_B_INIT.replace("self.conv2 = nn.Conv2d(b, b", "self.conv2 = nn.Conv2d(a, b")), "error", "conv2-in"),
    ("double-conv", _b(body="        ...\n"), "wrong", "dots-left"),
] + [("double-conv", code, "wrong", key) for key, code in _B_WRONG.items()] + [
    ("unet", _N_START, "empty", None),
    ("unet", _N, "correct", None),
    # the way up with the kept maps taken from the end, and the blocks as a loop over the way down
    ("unet", _n(down="        kept = []\n        for block in self.down[:3]:\n            x = block(x)\n            kept.append(x)\n"
                     "            x = self.pool(x)\n        x = self.down[3](x)\n",
                up="        for i in range(3):\n            up = self.ups[i](x)\n            x = self.convs[i](torch.cat([kept[-1 - i], up], dim=1))\n"),
     "correct", None),
    # added, as a residual connection
    ("unet", _n(up="        for i in range(3):\n            x = self.convs[i](kept[2 - i] + self.ups[i](x))\n"), "error", "not-stacked"),
    # stacked along the scenes
    ("unet", _n(up="        for i in range(3):\n            x = self.convs[i](torch.cat([kept[2 - i], self.ups[i](x)], dim=0))\n"), "error", "not-stacked"),
    # no skip connections at all
    ("unet", _n(up="        for i in range(3):\n            x = self.convs[i](self.ups[i](x))\n"), "error", "not-stacked"),
    ("unet", _n(down="        kept = []\n        for i in range(4):\n            x = self.down[i](x)\n            kept.append(x)\n"
                     "            x = self.pool(x)\n"), "error", "pool-bottom"),
    # the blank of the last hint filled with x: the map from below is not upsampled
    ("unet", _n(up="        for i in range(3):\n            x = self.convs[i](torch.cat([kept[2 - i], x], dim=1))\n"), "error", "pool-bottom"),
    # upsampled after the stack
    ("unet", _n(up="        for i in range(3):\n            x = self.convs[i](self.ups[i](torch.cat([kept[2 - i], x], dim=1)))\n"), "error", "pool-bottom"),
    # kept after the pooling
    ("unet", _n(down="        kept = []\n        for i in range(3):\n            x = self.down[i](x)\n            x = self.pool(x)\n"
                     "            kept.append(x)\n        x = self.down[3](x)\n"), "error", "kept-after-pool"),
    # no pooling at all
    ("unet", _n(down="        kept = []\n        for i in range(3):\n            x = self.down[i](x)\n            kept.append(x)\n"
                     "        x = self.down[3](x)\n"), "error", "kept-after-pool"),
    ("unet", _n(convs="        self.convs = nn.ModuleList([DoubleConv(32, 32), DoubleConv(16, 16), DoubleConv(8, 8)])\n"), "error", "convs-channels"),
    ("unet", _n(up="        for i in range(3):\n            x = self.convs[i](torch.cat([kept[i], self.ups[i](x)], dim=1))\n"), "error", "wrong-kept"),
    ("unet", _n(ret="        ...\n"), "wrong", "dots-left"),
] + [("unet", code, "wrong", key) for key, code in _N_WRONG.items()] + [
    ("mean-iou", _M_START, "empty", None),
    ("mean-iou", _M, "correct", None),
    # in one line, with a list of the IoUs
    ("mean-iou", _M_HEAD + "    ious = [kind_iou(pred, classes, c) for c in range(n)]\n    return sum(ious) / len(ious)\n", "correct", None),
    ("mean-iou", _m(ret="    ...\n"), "wrong", "dots-left"),
] + [("mean-iou", code, "wrong", key) for key, code in _M_WRONG.items()]


# --- the demo: the learner's U-Net, trained on the class maps of their class_map ------------------------------------

_Stopped, _named = _FCN._Stopped, _FCN._named


def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def _demo(show=None):
    """For the demo: None when unet.py or mean_iou.py is not in the project yet; "classes" when the learner's
    class_map stops or does not give a class map for every scene; "unet" when their UNet stops, does not give 4
    scores for every pixel of 2 scenes, or trains to a loss that is not a number; "mean_iou" when their mean_iou stops
    or does not give 1 for class maps compared with themselves; "slow" when the training takes too long; else a dict
    with the loss after every epoch ("curves"), the class maps of the SHOWN scenes after every epoch ("moments"), and
    on the test scenes after the last epoch the share of pixels right, the IoU of every class (background first), the
    mean IoU by their mean_iou, the parameters of their U-Net, the modules the course's versions stood in for, and
    their trained U-Net layer by layer on the scene of the worked example ("net", a netview.Trace). `show` hears the
    run while it trains."""
    if not (_saved("unet") and _saved("mean_iou")):
        return None
    names = {}
    for module, name, key in (("classes", "class_map", "classes"), ("unet", "UNet", "unet"),
                              ("mean_iou", "mean_iou", "mean_iou")):
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
    try:   # the learner's U-Net on 2 scenes: 4 scores for every pixel
        model = names["UNet"]()
        with torch.no_grad():
            if not (isinstance(model, nn.Module)
                    and tuple(model(X[:2].float() / 255).shape) == (2, CLASSES, scenes.SIZE, scenes.SIZE)):
                return "unet"
    except Exception:
        return "unet"
    try:   # the learner's mean IoU of class maps compared with themselves: 1
        one = float(names["mean_iou"](Ct[:4], Ct[:4], CLASSES))
        if not (math.isfinite(one) and abs(one - 1) < 1e-6):
            return "mean_iou"
    except Exception:
        return "mean_iou"
    torch.manual_seed(0)   # the same start as the course's run, and batch normalization that saw no scene yet
    try:
        model = names["UNet"]()
    except Exception:
        return "unet"
    model.forward = _named(model.forward, "unet")   # an error in their U-Net is theirs; one in the course's code is not
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
        history = with_time_limit(lambda: _FCN.train(model, (X, C), (Xt, Ct), 0, EPOCHS, heard), TRAIN_LIMIT)
    except Timeout:
        return "slow"
    except _Stopped as e:
        return e.key
    if not all(math.isfinite(h["train_loss"]) and math.isfinite(h["val_loss"]) for h in history):
        return "unet"
    try:
        pred = _FCN.outputs(model, Xt).argmax(1)
        miou = float(_named(names["mean_iou"], "mean_iou")(pred, Ct, CLASSES))
    except _Stopped as e:
        return e.key
    curves = {"train": [float(h["train_loss"]) for h in history], "val": [float(h["val_loss"]) for h in history]}
    net = netview.trace(model, Xt[TEST.index(SEED):TEST.index(SEED) + 1].float() / 255)   # the scene of the example
    return {"curves": curves, "moments": moments, "acc": float((pred == Ct).float().mean()), "ious": class_ious(pred, Ct),
            "miou": miou, "params": sum(p.numel() for p in model.parameters()), "stand_ins": stand_ins, "net": net}


def layer_frame(net, k, lang: str) -> str:
    """The slider over the layers of the learner's trained U-Net: the network, with layer k (from 1) chosen and its
    feature maps under it."""
    return netview.picture(net, lang, chosen=int(k) - 1, uid=f"nv-unet-{int(k)}")


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file)."""
    texts = []
    for module in ("classes", "kind_iou", "double_conv", "unet", "mean_iou"):
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
