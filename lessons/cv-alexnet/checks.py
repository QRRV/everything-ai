# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-alexnet: SmallAlexNet, AlexNet's eight layers with weights at the size of CIFAR-10, saved in
the learner's project (work/projects/cnn/alexnet.py) when right, and checked on made-up photos, so it works without
CIFAR-10. AlexNet itself the lesson only counts and draws, layer by layer, without its weights: its numbers come from
the paper (Krizhevsky, Sutskever and Hinton, 2012) and from building its layers here (alexnet_layers). The demo trains
the learner's SmallAlexNet with their Trainer on the first 6,000 training photos of CIFAR-10, with or without dropout,
and with or without the changes of cv-augment-photos (their crop and jitter), and tests it on the first 1,000 test
photos; after every epoch it keeps the network's guess for one checked test photo of each kind, for rewinding."""
import importlib.util
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import chart, cifar, live, paths, projects
from everything_ai.check import Mistake, ScenarioBox, Sort, Timeout, run_code, with_time_limit
from everything_ai import netview, yours


def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_alexnet",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_AUG = _lesson_checks("cv-augment-photos")   # CIFAR-10 and its button, the changed photos, the rewind's pictures

# The demo uses the learner's channel_stats of cv-color-cnn, their crop and jitter of cv-augment-photos, their Trainer
# of nn-trainer and their count_params of cnn-pytorch; where one is missing or broken, the course's version stands in
# (template 1.3).
BUILDS_ON = (("nn", "train_epoch", "nn-trainer", "train-epoch"), ("nn", "trainer", "nn-trainer", "trainer"),
             ("cnn", "count", "cnn-pytorch", "count-params"), ("cnn", "channels", "cv-color-cnn", "channel-stats"),
             ("cnn", "crop", "cv-augment-photos", "crop"), ("cnn", "color", "cv-augment-photos", "color"))
projects.builds_on(*BUILDS_ON)

DOWNLOAD_LINE = {
    "en": "Works on every computer. CIFAR-10, 170 MB of photos, is downloaded once from the University of Toronto when you click its button in the lesson, unless you did so in an earlier lesson; after that, the lesson works without a connection. The exercises work without it too.",
    "nl": "Werkt op elke computer. CIFAR-10, 170 MB aan foto's, wordt één keer gedownload van de University of Toronto als je in de les op de knop ervoor klikt, tenzij je dat al in een eerdere les deed; daarna werkt de les zonder verbinding. De oefeningen werken ook zonder.",
}

cifar_or_stop = _AUG.cifar_or_stop
ready = _AUG.ready
photos = _AUG.photos                   # the first N_TRAIN training photos and N_TEST test photos, from 0 to 1
grouped, number, percent, low_high = _AUG.grouped, _AUG.number, _AUG.percent, _AUG.low_high
N_TRAIN, N_TEST = _AUG.N_TRAIN, _AUG.N_TEST
SHOWN, MOMENTS, NAMES = _AUG.SHOWN, _AUG.MOMENTS, _AUG.NAMES
EVERY = _AUG.EVERY   # the live chart hears the training loss of every EVERY batches
moments, best_epoch, guesses_picture = _AUG.moments, _AUG.best_epoch, _AUG.guesses_picture

EPOCHS, LR = 40, 0.001   # the demo: as long as in cv-augment-photos; at 0.01 or 0.003 SmallAlexNet hardly learns its training photos
P = 0.5                  # dropout, as in AlexNet
# The demo's four runs: (dropout, the changes of cv-augment-photos)
SETTINGS = {"none": (0.0, False), "dropout": (P, False), "changes": (0.0, True), "both": (P, True)}

# --- AlexNet -------------------------------------------------------------------------------------------------------

# AlexNet as in the paper, section 3.5: its eight layers with weights, each with its number of filters or outputs,
# and for a convolution its filter size, stride, padding and how many maps of the layer before each filter sees: the
# kernels of conv2, conv4 and conv5 see only the maps on their own GPU, half of them (groups=2). The paper gives the
# input as 224 by 224 and the maps of conv1 as 55 by 55; with filters of 11 by 11 and stride 4, that needs 2 pixels of
# padding, as torchvision's AlexNet has. Max pooling of 3 by 3 with stride 2 follows conv1, conv2 and conv5.
ALEXNET = (("conv1", 96, 11, 4, 2, 3), ("conv2", 256, 5, 1, 2, 48), ("conv3", 384, 3, 1, 1, 256),
           ("conv4", 384, 3, 1, 1, 192), ("conv5", 256, 3, 1, 1, 192), ("fc6", 4096), ("fc7", 4096), ("fc8", 1000))
POOLED_AFTER = ("conv1", "conv2", "conv5")
SIDE = 224


def alexnet_layers() -> list[dict]:
    """AlexNet's layers, built in PyTorch on the "meta" device, which knows shapes but holds no numbers: for each its
    name, its number of parameters, the shape of what it gives for one photo of SIDE by SIDE, before and after its
    pooling, and its number of multiplications for one photo: one per weight of a filter, for every number it gives."""
    x = torch.zeros(1, 3, SIDE, SIDE, device="meta")
    pool = nn.MaxPool2d(3, stride=2)
    out, maps = [], 3
    for name, n, *conv in ALEXNET:
        if conv:
            k, stride, padding, sees = conv
            layer = nn.Conv2d(maps, n, k, stride=stride, padding=padding, groups=maps // sees, device="meta")
        else:
            x = x.reshape(1, -1)
            layer = nn.Linear(x.shape[1], n, device="meta")
        x = layer(x)
        shape = tuple(x.shape[1:])
        if name in POOLED_AFTER:
            x = pool(x)
        out.append({"name": name, "params": sum(q.numel() for q in layer.parameters()), "shape": shape,
                    "pooled": tuple(x.shape[1:]) if name in POOLED_AFTER else None, "inputs": None if conv else layer.in_features,
                    "mults": layer.weight[0].numel() * int(np.prod(shape))})
        maps = n
    return out


LAYERS = alexnet_layers()
TOTAL = sum(layer["params"] for layer in LAYERS)                          # 60,965,224
CONV = sum(layer["params"] for layer in LAYERS if layer["name"].startswith("conv"))   # 2,334,080
MULTS = sum(layer["mults"] for layer in LAYERS)                                         # 724,406,816
CONV_MULTS = sum(layer["mults"] for layer in LAYERS if layer["name"].startswith("conv"))   # 665,784,864


def share(n: int, of: int = TOTAL) -> float:
    """n as a percentage of `of`, with one decimal."""
    return round(100 * n / of, 1)


_WHAT = {
    "en": {"conv1": "96 filters of 11 by 11 on the 3 colors, with stride 4", "conv2": "256 filters of 5 by 5, each on the 48 maps on its own GPU",
           "conv3": "384 filters of 3 by 3 on all 256 maps", "conv4": "384 filters of 3 by 3, each on the 192 maps on its own GPU",
           "conv5": "256 filters of 3 by 3, each on the 192 maps on its own GPU", "fc6": "a linear layer from 256 × 6 × 6 = 9,216 numbers to 4,096",
           "fc7": "a linear layer from 4,096 numbers to 4,096", "fc8": "a linear layer from 4,096 numbers to 1,000, a logit for each class"},
    "nl": {"conv1": "96 filters van 11 bij 11 op de 3 kleuren, met stride 4", "conv2": "256 filters van 5 bij 5, elk op de 48 maps op zijn eigen GPU",
           "conv3": "384 filters van 3 bij 3 op alle 256 maps", "conv4": "384 filters van 3 bij 3, elk op de 192 maps op zijn eigen GPU",
           "conv5": "256 filters van 3 bij 3, elk op de 192 maps op zijn eigen GPU", "fc6": "een linear layer van 256 × 6 × 6 = 9.216 getallen naar 4.096",
           "fc7": "een linear layer van 4.096 getallen naar 4.096", "fc8": "een linear layer van 4.096 getallen naar 1.000, een logit per klasse"},
}


def layer_sentence(k: int, lang: str) -> str:
    """The sentence under the slider over AlexNet's layers, at layer k (1 to 8)."""
    layer = LAYERS[k - 1]
    so_far = sum(L["params"] for L in LAYERS[:k])
    n, s, c, cs = grouped(layer["params"], lang), number(share(layer["params"]), 1, lang), grouped(so_far, lang), number(share(so_far), 1, lang)
    maps, side = layer["shape"][0], layer["shape"][-1] if len(layer["shape"]) > 1 else None
    if lang == "nl":
        after = (f" Daarna: {maps} maps van {side} bij {side}" + (f", en {layer['pooled'][-1]} bij {layer['pooled'][-1]} na de pooling." if layer["pooled"] else ".")) if side else ""
        return f"`{layer['name']}`, {_WHAT['nl'][layer['name']]}: {n} parameters, {s}% van alle {grouped(TOTAL, lang)}.{after} Tot hier samen: {c}, {cs}%."
    after = (f" After it: {maps} maps of {side} by {side}" + (f", and {layer['pooled'][-1]} by {layer['pooled'][-1]} after the pooling." if layer["pooled"] else ".")) if side else ""
    return f"`{layer['name']}`, {_WHAT['en'][layer['name']]}: {n} parameters, {s}% of all {grouped(TOTAL, lang)}.{after} Up to here together: {c}, {cs}%."


def layers_picture(k: int, labels: dict, lang: str) -> str:
    """AlexNet's parameters, layer by layer, as bars in millions, with its share of all above each: the layers up to
    layer k in the color of what a model learns, the ones after it in grey; under them, a bar of the layers up to k
    together, and a dashed line at 90% of all. `labels` gives "title", "x", "so_far", "line" and "alt"."""
    rows = len(LAYERS) + 1.5
    p = chart.Panel((-19, 64), (-0.9, rows - 0.6), title=labels["title"], xlabel=labels["x"], yticks=[], xticks=[0, 10, 20, 30, 40, 50, 60])
    for i, layer in enumerate(LAYERS):
        y = rows - 1 - i
        m = layer["params"] / 1e6
        p.rect(0, y - 0.32, m, y + 0.32, "param" if i < k else "ink-3")
        p.text(-18.5, y - 0.04, layer["name"], "ink" if i == k - 1 else "ink-3", "start")
        p.text(m + 0.8, y - 0.04, number(share(layer["params"]), 1, lang) + "%", "ink-2" if i < k else "ink-3", "start")
    so_far = sum(L["params"] for L in LAYERS[:k]) / 1e6
    p.rect(0, -0.32, so_far, 0.32, "param")
    p.text(-18.5, -0.04, labels["so_far"], "ink", "start")
    line = 0.9 * TOTAL / 1e6
    p.line([line, line], [-0.6, rows - 0.6], "ink-2", 1.2, dashed=True)
    p.text(line + 0.8, rows - 1.0, labels["line"], "ink-2", "start")
    return chart.figure([p], alt=labels["alt"], width=400, height=290)


# --- ILSVRC --------------------------------------------------------------------------------------------------------

# The lowest top-5 error on the test photos of the classification contest of ILSVRC, each year from 2010 to 2015, in
# percent: Russakovsky et al. (2015), tables 5 to 7, for 2010 to 2014 (NEC, XRCE, SuperVision with the seven CNNs of
# which two were also trained on ImageNet's Fall 2011 release, Clarifai also with extra photos, GoogLeNet), and the
# ResNet paper for 2015 (He et al., 2015: 3.57%). In 2012, ISI came second with 26.2%.
WINNERS = ((2010, 28.2), (2011, 25.8), (2012, 15.3), (2013, 11.2), (2014, 6.7), (2015, 3.6))
SECOND_2012 = 26.2
# The best entries of 2012 and 2013 that trained on ILSVRC's photos only: SuperVision's five CNNs (the paper's table 2)
# and Clarifai without extra photos (Russakovsky et al., table 6: 11.7%).
WITHOUT_EXTRA = {2012: 16.4, 2013: 11.7}


def ilsvrc_picture(labels: dict, lang: str) -> str:
    """The lowest top-5 error of each year's contest as bars, in the color of mistakes, with its number above it; an
    open dot for the second place of 2012. `labels` gives "title", "y", "second", "alt", and a name above the bar of
    a year, such as {2012: "AlexNet"}."""
    p = chart.Panel((2009.4, 2015.6), (0, 35), title=labels["title"], ylabel=labels["y"], xticks=[y for y, _ in WINNERS],
                    yticks=[0, 10, 20, 30])
    for year, error in WINNERS:
        p.rect(year - 0.3, 0, year + 0.3, error, "loss")
        p.text(year, error + 1.5, number(error, 1, lang), "ink", "middle")
        if year in labels:
            p.text(year, error + 4.6, labels[year], "ink-2", "middle")
    p.points([2012.42], [SECOND_2012], "ink-3", r=3.5, hollow=True)
    p.text(2012.55, SECOND_2012 - 0.4, labels["second"], "ink-3", "start")
    return chart.figure([p], alt=labels["alt"], width=380, height=250)


# --- what the course measured ---------------------------------------------------------------------------------------

# What the course measured, on one thread, with the course's code, through run(): SmallAlexNet of exercise 2 after
# torch.manual_seed(seed), with dropout P ("dropout", "both") or 0 ("none", "changes"), trained with the Trainer of
# nn-trainer (AdamW, a learning rate of LR, a warmup of one epoch, cosine decay, max_norm 1.0, mini-batches of 64) on
# the first N_TRAIN training photos, shuffled after torch.manual_seed(seed + 1), for EPOCHS epochs and the seeds 0 to 4.
# With "changes" and "both", the photos come from the Dataset of cv-augment-photos: mirrored half of the time, cut out
# at a random place with crop, and changed in color with jitter, anew every time; without, they are used as they are.
# Both are normalized with channel_stats of the training photos as they are; the test photos are only normalized.
# "train" and "test" are how many of the training photos as they are and of the first N_TEST test photos are read right
# after the last epoch; "best" is the first epoch with the lowest validation loss, rounded to 3 decimals; "last_val"
# and "last_train" are the validation and the training loss of the last epoch. CURVES are seed 0, after every epoch:
# the validation loss of "none", "changes" and "both", and the training loss of "both". tests/test_cv_alexnet.py
# measures seed 0 of "both" again when EAI_CIFAR_FILE names the unpacked cifar-10.npz.
REFERENCE = {
    "none": {"train": [5907, 5965, 5956, 5943, 5977], "test": [554, 570, 576, 541, 550], "best": [16, 11, 13, 13, 13],
            "last_val": [2.612, 3.119, 2.864, 3.2, 3.448], "last_train": [0.081, 0.042, 0.051, 0.055, 0.031]},
    "dropout": {"train": [5482, 5603, 5572, 5568, 5557], "test": [550, 542, 575, 557, 562], "best": [17, 16, 14, 15, 16],
               "last_val": [1.896, 2.273, 1.869, 2.178, 1.949], "last_train": [0.351, 0.252, 0.296, 0.266, 0.311]},
    "changes": {"train": [4076, 4221, 4181, 4140, 4106], "test": [595, 626, 632, 636, 594], "best": [33, 33, 29, 36, 32],
               "last_val": [1.157, 1.095, 1.087, 1.118, 1.168], "last_train": [0.937, 0.883, 0.91, 0.911, 0.943]},
    "both": {"train": [3871, 3945, 3953, 3893, 3921], "test": [585, 593, 604, 581, 581], "best": [39, 30, 37, 36, 33],
            "last_val": [1.196, 1.203, 1.156, 1.197, 1.21], "last_train": [1.066, 1.053, 1.022, 1.088, 1.081]},
}
CURVES = {
    "none_val": [2.039, 1.92, 1.878, 1.655, 1.584, 1.572, 1.451, 1.369, 1.446, 1.308, 1.432, 1.29, 1.283, 1.257, 1.277, 1.232,
        1.274, 1.294, 1.322, 1.341, 1.405, 1.499, 1.487, 1.591, 1.677, 1.706, 1.791, 1.89, 2.055, 2.093,
        2.172, 2.283, 2.35, 2.436, 2.518, 2.547, 2.581, 2.601, 2.61, 2.612],
    "changes_val": [2.037, 1.899, 1.779, 1.749, 1.805, 1.57, 1.538, 1.488, 1.454, 1.448, 1.393, 1.402, 1.401, 1.372, 1.301,
        1.355, 1.262, 1.282, 1.244, 1.276, 1.243, 1.228, 1.238, 1.204, 1.21, 1.166, 1.21, 1.217, 1.172, 1.174,
        1.154, 1.168, 1.151, 1.159, 1.165, 1.159, 1.162, 1.155, 1.157, 1.157],
    "both_val": [2.126, 1.948, 1.814, 1.755, 1.687, 1.664, 1.555, 1.539, 1.518, 1.45, 1.425, 1.388, 1.367, 1.384, 1.357,
        1.348, 1.468, 1.305, 1.299, 1.303, 1.266, 1.287, 1.272, 1.244, 1.258, 1.24, 1.234, 1.274, 1.226,
        1.202, 1.22, 1.206, 1.207, 1.2, 1.217, 1.202, 1.198, 1.198, 1.195, 1.196],
    "both_train": [2.268, 2.048, 1.954, 1.859, 1.809, 1.739, 1.695, 1.668, 1.609, 1.586, 1.528, 1.494, 1.476, 1.447, 1.407,
        1.382, 1.379, 1.352, 1.331, 1.312, 1.285, 1.264, 1.238, 1.215, 1.212, 1.186, 1.167, 1.155, 1.156,
        1.122, 1.129, 1.107, 1.113, 1.098, 1.083, 1.072, 1.075, 1.089, 1.072, 1.066],
}
PARAMS = 201_402   # of SmallAlexNet: 59,056 in its convolutions, 142,346 in its linear layers
COLOR_CNN = {key: _AUG.REFERENCE[(key, 40)] for key in ("none", "all")}   # ColorCNN after 40 epochs, cv-augment-photos


# --- the pictures of the runs ----------------------------------------------------------------------------------------

ROWS = ("none", "dropout", "changes", "both")


def runs_picture(labels: dict, lang: str, mine: dict | None = None) -> str:
    """How many photos SmallAlexNet reads right after 40 epochs, of its training photos as they are and of the test
    photos, in each of the four runs, with a dot for every seed, filled for the training photos and open for the test
    photos, and a line between their means; under them, ColorCNN of cv-augment-photos the same way, in grey. With
    `mine` ({"variant", "train", "test"}), the learner's run as a row of its own at the top. `labels` gives "title",
    "x", a name for each row ("{epochs}" may be in it), "color_none", "color_all", "mine", "train", "test" and "alt"."""
    rows = [(key, REFERENCE[key], "ink", labels[key]) for key in ROWS]
    rows += [(key, COLOR_CNN[key], "ink-3", labels[f"color_{key}"]) for key in ("none", "all")]
    if mine is not None:
        rows.insert(0, ("mine", {"train": [mine["train"]], "test": [mine["test"]]}, "ink", labels["mine"]))
    p = chart.Panel((40, 100), (-0.8, len(rows) - 0.25), title=labels["title"], xlabel=labels["x"], yticks=[],
                    xticks=[40, 50, 60, 70, 80, 90, 100])
    for k, (key, run, role, name) in enumerate(rows):
        y = len(rows) - 1 - k
        train = [100 * v / N_TRAIN for v in run["train"]]
        test = [100 * v / N_TEST for v in run["test"]]
        middle = sum(test) / len(test), sum(train) / len(train)
        p.line(list(middle), [y, y], "ink-3", 1.6)
        p.points(train, [y] * len(train), role, r=3.5)
        p.points(test, [y] * len(test), role, r=3.5, hollow=True)
        p.text(41, y + 0.38, name.format(epochs=EPOCHS), "ink-2" if role != "ink-3" else "ink-3", "start")
        if k == len(rows) - 1:   # under the last row, each label away from the other, toward its own end of the line
            p.text(middle[0], y - 0.45, labels["test"], "ink-3", "end")
            p.text(middle[1], y - 0.45, labels["train"], "ink-3", "start")
    svg = chart.figure([p], alt=labels["alt"], width=420, height=60 * len(rows) + 60)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


TOP = 2.8   # the top of the curves picture: above the loss of guessing, ln 10 ≈ 2.30, and the end of CURVES["none_val"]


def curves_picture(curves: dict, labels: dict, lang: str, at: int | None = None, other: str = "none") -> str:
    """The loss after every epoch of the demo's run: on the training photos solid and on the test photos dashed, with
    a dot where the validation loss is lowest; in grey the validation loss of the course's run `other` (seed 0), up to
    where it leaves the top of the picture. With `at`, a line at that epoch, for rewinding. `labels` gives "title",
    "x", "y", "train", "val", the name of `other` and "alt"."""
    train, val = curves.get("train", []), curves.get("val", [])
    mine = [v for v in train + val if np.isfinite(v)]
    top = max([TOP] + [1.1 * v for v in mine])
    before = CURVES[f"{other}_val"]
    p = chart.Panel((0, max(len(val), len(train), len(before), 1) + 0.5), (0, top), title=labels["title"],
                    xlabel=labels["x"], ylabel=labels["y"])
    out = next((k for k, v in enumerate(before) if v > top), len(before))
    xs, ys = list(range(1, out + 1)), before[:out]
    if out < len(before):   # where it crosses the top, between epoch `out` and `out + 1`
        xs.append(out + (top - before[out - 1]) / (before[out] - before[out - 1]))
        ys.append(top)
    p.line(xs, ys, "ink-3", 1.6, dashed=True)
    p.text(xs[-1], ys[-1] + 0.04 * top if out < len(before) else ys[-1] + 0.06 * top, labels[other], "ink-3", "end")
    train_above = bool(train and val and train[-1] >= val[-1])
    for ys, dashed, name, above in ((train, False, labels["train"], train_above), (val, True, labels["val"], not train_above)):
        if ys:
            xs = list(range(1, len(ys) + 1))
            p.line(xs, ys, "loss", 2.2, dashed=dashed)
            below = ys[-1] - 0.08 * top
            p.text(len(ys), ys[-1] + 0.07 * top if above or below < 0.04 * top else below, name, "loss", "end")
    if val:
        best = best_epoch(val)
        p.points([best], [val[best - 1]], "loss", r=4)
    if at is not None:
        p.line([at, at], [0, top], "ink-2", 1.2)
    svg = chart.figure([p], alt=labels["alt"], width=380, height=260)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def now_pictures(seen: dict, title: str, lang: str, uid: str) -> list:
    """What the network makes at one moment of its run: its guesses for the photos SHOWN, and its first layer on the
    car (cv-augment-photos)."""
    return _AUG.now_pictures(seen, title, lang, uid)


def epoch_frame(result: dict, e: int, labels: dict, title: str, lang: str, **kw) -> str:
    """The rewind at epoch e (from 1): the loss after every epoch with a line at e, then the network's guesses for the
    photos SHOWN after it, and under them its first layer on the car, as it was then (cv-augment-photos)."""
    return _AUG.rewind_frame(curves_picture, result, e, labels, title, lang, "alexnet", **kw)


# The lesson is done when these are correct. "dropout-here" is the choice question in the lesson.
REQUIRED = ["alexnet-today", "small-alexnet", "dropout-here"]

# --- exercise 1: what ResNet-18 still does ------------------------------------------------------------------------

_SORT = Sort(
    name="alexnet-today",
    items={
        "relu": {"en": "**ReLU** after the layers with weights, but not after the last.", "nl": "**ReLU** na de layers met weights, behalve na de laatste."},
        "pool": {"en": "**Max pooling of 3 by 3 with stride 2**: squares that overlap.", "nl": "**Max pooling van 3 bij 3 met stride 2**: vierkanten die overlappen."},
        "crops": {"en": "**Mirror images and random crops** of the training photos.", "nl": "**Spiegelbeelden en random crops** van de trainingsfoto's."},
        "big": {"en": "**Filters of 11 by 11 with stride 4** in the first layer.", "nl": "**Filters van 11 bij 11 met stride 4** in de eerste layer."},
        "lrn": {"en": "**Local response normalization** after the first two layers.", "nl": "**Local response normalization** na de eerste twee layers."},
        "dense": {"en": "**Three linear layers with dropout** at the end, with 96% of the parameters.", "nl": "**Drie linear layers met dropout** aan het eind, met 96% van de parameters."},
    },
    groups={"same": {"en": "ResNet-18 does this too", "nl": "ResNet-18 doet dit ook"},
            "other": {"en": "ResNet-18 does this differently", "nl": "ResNet-18 doet dit anders"}},
    solution={"relu": "same", "pool": "same", "crops": "same", "big": "other", "lrn": "other", "dense": "other"},
    nudge={
        "relu": {"en": "What comes after each convolution and its batch normalization in a block of ResNet-18?",
                 "nl": "Wat komt er in een blok van ResNet-18 na elke convolutie en zijn batch normalization?"},
        "pool": {"en": "In *Look inside a CNN*, a photo went through `conv1`, `bn1`, a ReLU, and then what, before `layer1`?",
                 "nl": "In *In een CNN kijken* ging een foto door `conv1`, `bn1`, een ReLU, en dan wat, vóór `layer1`?"},
        "crops": {"en": "How did the ResNet paper train on CIFAR-10, and how did your ColorCNN learn less by heart?",
                  "nl": "Hoe trainde het ResNet-paper op CIFAR-10, en hoe leerde je ColorCNN minder uit zijn hoofd?"},
        "big": {"en": "How large are the filters of `conv1` of ResNet-18, and what is its stride?",
                "nl": "Hoe groot zijn de filters van `conv1` van ResNet-18, en wat is zijn stride?"},
        "lrn": {"en": "What does ResNet-18 put after each convolution, to keep its numbers in range?",
                "nl": "Wat zet ResNet-18 na elke convolutie, om zijn getallen binnen de perken te houden?"},
        "dense": {"en": "What does ResNet-18 do with the 512 maps of `layer4`, and how many linear layers come after that?",
                  "nl": "Wat doet ResNet-18 met de 512 maps van `layer4`, en hoeveel linear layers komen daarna?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the ideas again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de ideeën opnieuw uit."},
        "empty": {"en": "Put each idea of AlexNet in a group: ResNet-18 does it too, or ResNet-18 does it differently.",
                  "nl": "Zet elk idee van AlexNet in een groep: ResNet-18 doet het ook, of ResNet-18 doet het anders."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "{item} belongs in the other group.",
                  "nl": "{item} hoort in de andere groep."},
        "correct": {"en": "All {n} are right. ReLU, the overlapping pooling and training on mirrored and cut-out photos stayed. The large first filters became 7 by 7 with stride 2, the size with which Zeiler and Fergus made AlexNet better; batch normalization took the place of local response normalization; and the three large linear layers became the mean of each map and one linear layer of 513,000 parameters.",
                    "nl": "Alle {n} kloppen. ReLU, de pooling die overlapt en trainen op gespiegelde en uitgesneden foto's bleven. De grote eerste filters werden 7 bij 7 met stride 2, de grootte waarmee Zeiler en Fergus AlexNet beter maakten; batch normalization nam de plaats in van local response normalization; en de drie grote linear layers werden het gemiddelde van elke map en één linear layer van 513.000 parameters."},
    },
)

# --- exercise 2: SmallAlexNet --------------------------------------------------------------------------------------

_A_HEAD = ("import torch\nfrom torch import nn\n\n\nclass SmallAlexNet(nn.Module):\n    def __init__(self, p=0.5):\n        super().__init__()\n"
           "        self.conv1 = nn.Conv2d(3, 12, 5, padding=2)\n        self.conv2 = nn.Conv2d(12, 32, 5, padding=2)\n"
           "        self.conv3 = nn.Conv2d(32, 48, 3, padding=1)\n        self.conv4 = nn.Conv2d(48, 48, 3, padding=1)\n"
           "        self.conv5 = nn.Conv2d(48, 32, 3, padding=1)\n        self.pool = nn.MaxPool2d(3, stride=2)\n"
           "        self.drop = nn.Dropout(p)\n")
_A_FORWARD = ("\n    def forward(self, x):\n        x = self.pool(torch.relu(self.conv1(x)))\n        x = self.pool(torch.relu(self.conv2(x)))\n"
              "        x = torch.relu(self.conv3(x))\n        x = torch.relu(self.conv4(x))\n        x = self.pool(torch.relu(self.conv5(x)))\n"
              "        x = x.reshape(len(x), -1)\n")
_A_DENSE = "        self.fc7 = nn.Linear(256, 256)\n        self.fc8 = nn.Linear(256, 10)\n"
_A_BODY = ("        x = self.drop(torch.relu(self.fc6(x)))\n        x = self.drop(torch.relu(self.fc7(x)))\n        return self.fc8(x)\n")


def _a(inputs: str = "32 * 3 * 3", body: str = _A_BODY, dense: str = _A_DENSE, head: str = _A_HEAD, forward: str = _A_FORWARD) -> str:
    return head + f"        self.fc6 = nn.Linear({inputs}, 256)\n" + dense + forward + body


_A = _a()
_A_START = (_A_HEAD + "        # fc6: from all numbers after the last pooling to 256\n        self.fc6 = nn.Linear(..., 256)\n" + _A_DENSE + _A_FORWARD
            + "        # fc6 and fc7, each with a ReLU and then dropout; fc8 gives the 10 logits\n        ...\n")
_A_DATA = "import torch\n\ntorch.manual_seed(0)\nmodel = SmallAlexNet()\n"
_A_SCENARIOS = [
    # two made-up photos, in eval mode: the shape of the logits, the first four of the first photo, the number of
    # inputs of fc6 and the number of parameters
    _A_DATA + "model.eval()\nx = torch.rand(2, 3, 32, 32)\nlogits = model(x)\n"
    "[list(logits.shape), logits[0, :4].tolist(), model.fc6.in_features, sum(q.numel() for q in model.parameters())]",
    # dropout: with p=0 training and eval give the same; with p=0.5 two runs in training differ and two in eval do
    # not; no logit is set to 0 by dropout; and with a seed, the first four logits in training mode are the course's,
    # which they are only with one dropout after the ReLU of fc6 and one after that of fc7 (or before those ReLUs,
    # which drops the same numbers)
    "import torch\n\ntorch.manual_seed(0)\nx = torch.rand(64, 3, 32, 32)\nplain = SmallAlexNet(p=0.0)\nplain.train()\na = plain(x)\nplain.eval()\nb = plain(x)\n"
    "model = SmallAlexNet(p=0.5)\nmodel.train()\ntorch.manual_seed(1)\ng = model(x)\nc, d = model(x), model(x)\nmodel.eval()\ne, f = model(x), model(x)\n"
    "[torch.equal(a, b), torch.equal(c, d), torch.equal(e, f), (c == 0).float().mean().item(), g[0, :4].tolist()]",
    # a batch of one photo
    _A_DATA + "model.eval()\nlogits = model(torch.rand(1, 3, 32, 32))\n[list(logits.shape), logits[0, 7:].tolist()]",
]
_A_WRONG = {
    "no-dropout": _a(body="        x = torch.relu(self.fc6(x))\n        x = torch.relu(self.fc7(x))\n        return self.fc8(x)\n"),
    "drop-logits": _a(body="        x = self.drop(torch.relu(self.fc6(x)))\n        x = self.drop(torch.relu(self.fc7(x)))\n        return self.drop(self.fc8(x))\n"),
    "relu-logits": _a(body="        x = self.drop(torch.relu(self.fc6(x)))\n        x = self.drop(torch.relu(self.fc7(x)))\n        return torch.relu(self.fc8(x))\n"),
    "no-relu": _a(body="        x = self.drop(self.fc6(x))\n        x = self.drop(self.fc7(x))\n        return self.fc8(x)\n"),
    "fixed-p": _a(head=_A_HEAD.replace("self.drop = nn.Dropout(p)", "self.drop = nn.Dropout(0.5)")),
    "pool-2": _a(inputs="32 * 4 * 4", head=_A_HEAD.replace("nn.MaxPool2d(3, stride=2)", "nn.MaxPool2d(2)")),
    # dropout before fc6 and fc7, as torchvision's AlexNet has it: right in eval mode, and random in training mode
    "drop-before": _a(body="        x = torch.relu(self.fc6(self.drop(x)))\n        x = torch.relu(self.fc7(self.drop(x)))\n        return self.fc8(x)\n"),
}
# `...` on a line of its own, after `=` or `return`, or as the first value in brackets: nn.Linear(..., 256)
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$|=\s*\.\.\.|\breturn\s+\.\.\.|\(\s*\.\.\.\s*,", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code: the number of inputs of `fc6`, and the end of `forward`.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code: het aantal inputs van `fc6`, en het eind van `forward`.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_S_EN = "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_S_NL = "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"


def _close(got, want) -> bool:
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    if isinstance(want, bool):
        return got is want or (isinstance(got, (bool, np.bool_)) and bool(got) == want)
    if isinstance(want, float):
        return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - want) <= 1e-4 * max(1.0, abs(want))
    return got == want


EXERCISES = {
    "alexnet-today": _SORT,
    "small-alexnet": ScenarioBox(
        name="small-alexnet",
        picture=netview.network("SmallAlexNet", yours.scene_batch, chosen=0),
        start=_A_START,
        solution=_A,
        scenarios=_A_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "alexnet"),
        first=[_DOTS],
        mistakes=[
            Mistake("flat-size", error=lambda e: isinstance(e, RuntimeError) and "cannot be multiplied" in str(e),
                    en="A linear layer gets another number of inputs than it was made for. Does `forward` use `fc6`, `fc7` and `fc8` once each, in that order? Then it is `fc6`: it must get as many inputs as each photo has numbers after the last pooling. Each convolution keeps the size it gets, thanks to its padding, and each pooling of 3 with stride 2 turns n rows into (n − 3) // 2 + 1. How many rows are left after three poolings, and how many maps does `conv5` give?",
                    nl="Een linear layer krijgt een ander aantal inputs dan waarvoor hij gemaakt is. Gebruikt `forward` `fc6`, `fc7` en `fc8` elk één keer, in die volgorde? Dan is het `fc6`: die moet zoveel inputs krijgen als elke foto getallen heeft na de laatste pooling. Elke convolutie houdt de grootte die hij krijgt, dankzij zijn padding, en elke pooling van 3 met stride 2 maakt van n rijen (n − 3) // 2 + 1. Hoeveel rijen zijn er over na drie poolings, en hoeveel maps geeft `conv5`?"),
            Mistake("no-return", error=lambda e: isinstance(e, AttributeError) and "'NoneType' object has no attribute" in str(e),
                    en="The check asks for the shape of what your network gives, and `forward` gives back nothing, `None`: end it with `return self.fc8(x)`, the 10 logits.",
                    nl="De check vraagt de shape van wat je netwerk geeft, en `forward` geeft niets terug, `None`: eindig hem met `return self.fc8(x)`, de 10 logits."),
            Mistake("no-dropout", wrong=_A_WRONG["no-dropout"],
                    en=_S_EN + ": two runs in training mode give the same logits, so nothing in your network is random: there is no dropout. Put `self.drop` after the ReLU of `fc6` and after the ReLU of `fc7`.",
                    nl=_S_NL + ": twee runs in training mode geven dezelfde logits, dus niets in je netwerk is willekeurig: er is geen dropout. Zet `self.drop` na de ReLU van `fc6` en na de ReLU van `fc7`."),
            Mistake("drop-logits", wrong=_A_WRONG["drop-logits"],
                    en=_S_EN + ": in training mode, about half of the logits are exactly 0, so dropout is on the logits themselves. Dropout switches off neurons of a hidden layer; a logit set to 0 would say nothing about its class. AlexNet put it on `fc6` and `fc7` only.",
                    nl=_S_NL + ": in training mode is ongeveer de helft van de logits precies 0, dus zit er dropout op de logits zelf. Dropout zet neuronen van een hidden layer uit; een logit die op 0 is gezet, zegt niets meer over zijn klasse. AlexNet zette het alleen op `fc6` en `fc7`."),
            Mistake("relu-logits", wrong=_A_WRONG["relu-logits"],
                    en=_S_EN + ": your logits are never below 0, because of a ReLU after `fc8`. Logits may be negative: softmax makes probabilities of them. `fc8` comes last, without a ReLU.",
                    nl=_S_NL + ": je logits zijn nooit kleiner dan 0, door een ReLU na `fc8`. Logits mogen negatief zijn: softmax maakt er kansen van. `fc8` komt als laatste, zonder ReLU."),
            Mistake("no-relu", wrong=_A_WRONG["no-relu"],
                    en=_S_EN + ": without a ReLU after `fc6` and `fc7`, the three linear layers together are no more than one linear layer. Put `torch.relu` around `self.fc6(x)` and `self.fc7(x)`, before the dropout.",
                    nl=_S_NL + ": zonder ReLU na `fc6` en `fc7` zijn de drie linear layers samen niet meer dan één linear layer. Zet `torch.relu` om `self.fc6(x)` en `self.fc7(x)`, vóór de dropout."),
            Mistake("fixed-p", wrong=_A_WRONG["fixed-p"],
                    en=_S_EN + ": with `p=0.0`, your network still drops neurons in training mode, so it does not use the `p` it gets. Give `nn.Dropout` the `p` of `__init__`: the demo trains with and without dropout that way.",
                    nl=_S_NL + ": met `p=0.0` laat je netwerk in training mode nog steeds neuronen vallen, dus gebruikt het de `p` die het krijgt niet. Geef `nn.Dropout` de `p` van `__init__`: zo traint de demo met en zonder dropout."),
            Mistake("always-on", matches=lambda got, text: re.search(r"(?:\bF|functional)\.dropout\(", text) is not None and "training" not in text,
                    en=_S_EN + ": in eval mode, your logits differ from the course's, so your dropout still drops numbers there. `nn.functional.dropout` drops neurons also in eval mode, unless you give it `training=self.training`. `self.drop`, the `nn.Dropout` of `__init__`, does that by itself.",
                    nl=_S_NL + ": in eval mode verschillen je logits van die van de course, dus laat je dropout daar nog steeds getallen vallen. `nn.functional.dropout` laat ook in eval mode neuronen vallen, tenzij je hem `training=self.training` geeft. `self.drop`, de `nn.Dropout` van `__init__`, doet dat vanzelf."),
            Mistake("pool-2", wrong=_A_WRONG["pool-2"],
                    en=_S_EN + ": your pooling takes squares of 2 by 2 with stride 2, so they do not overlap, and `fc6` gets 512 numbers instead of 288. AlexNet's pooling takes squares of 3 by 3 with stride 2: `nn.MaxPool2d(3, stride=2)`, as the box had it.",
                    nl=_S_NL + ": je pooling neemt vierkanten van 2 bij 2 met stride 2, dus overlappen ze niet, en krijgt `fc6` 512 getallen in plaats van 288. De pooling van AlexNet neemt vierkanten van 3 bij 3 met stride 2: `nn.MaxPool2d(3, stride=2)`, zoals het in het vak stond."),
            Mistake("drop-before", wrong=_A_WRONG["drop-before"],
                    en=_S_EN + ": in eval mode your logits are right, and in training mode they are not: your dropout drops numbers that go into `fc6` and `fc7`, as torchvision's AlexNet does. Here, as in the paper, dropout comes after the ReLU of `fc6` and after the ReLU of `fc7`, on what comes out of them.",
                    nl=_S_NL + ": in eval mode kloppen je logits, en in training mode niet: je dropout laat getallen vallen die `fc6` en `fc7` in gaan, zoals de AlexNet van torchvision doet. Hier komt dropout, net als in het paper, na de ReLU van `fc6` en na de ReLU van `fc7`, op wat eruit komt."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...` with your code: the number of inputs of `fc6`, and the end of `forward`.",
                      "nl": "Vervang de twee `...` door je code: het aantal inputs van `fc6`, en het eind van `forward`."},
            "wrong": {"en": _S_EN + ". Does `fc6` get 32 × 3 × 3 numbers, and do `fc6` and `fc7` each have a ReLU and then dropout, with `fc8` last and on its own?",
                      "nl": _S_NL + ". Krijgt `fc6` 32 × 3 × 3 getallen, en hebben `fc6` en `fc7` elk een ReLU en dan dropout, met `fc8` als laatste en zonder iets erna?"},
            "almost": {"en": "Right in eval mode. But " + _S_EN[0].lower() + _S_EN[1:] + ". In training mode, do `fc6` and `fc7` each get a ReLU and then dropout, with the `p` your network gets, and nothing else?",
                       "nl": "Goed in eval mode. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Krijgen `fc6` en `fc7` in training mode elk een ReLU en dan dropout, met de `p` die je netwerk krijgt, en verder niets?"},
            "correct": {"en": "Your `SmallAlexNet` is right: five convolutions with three overlapping poolings, and three linear layers with dropout after the first two, 201,402 parameters. It is now in your project, as `{saved}`.",
                        "nl": "Je `SmallAlexNet` klopt: vijf convoluties met drie poolings die overlappen, en drie linear layers met dropout na de eerste twee, 201.402 parameters. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `SmallAlexNet` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `SmallAlexNet` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("small-alexnet", _A_START, "empty", None),
    ("small-alexnet", _A, "correct", None),
    ("small-alexnet", _a("288"), "correct", None),
    # dropout before the ReLU drops the same numbers: ReLU(2z) = 2 ReLU(z), and ReLU(0) = 0
    ("small-alexnet", _a(body="        x = torch.relu(self.drop(self.fc6(x)))\n        x = torch.relu(self.drop(self.fc7(x)))\n        return self.fc8(x)\n"), "correct", None),
    # the ReLU after the pooling: max pooling and a ReLU may change places
    ("small-alexnet", _a(forward=_A_FORWARD.replace("self.pool(torch.relu(self.conv1(x)))", "torch.relu(self.pool(self.conv1(x)))")), "correct", None),
    # nn.functional.dropout told when the network trains
    ("small-alexnet", _a(body="        x = nn.functional.dropout(torch.relu(self.fc6(x)), self.drop.p, training=self.training)\n"
                              "        x = nn.functional.dropout(torch.relu(self.fc7(x)), self.drop.p, training=self.training)\n        return self.fc8(x)\n"), "correct", None),
    ("small-alexnet", _a(body="        x = self.drop(torch.relu(self.fc6(x)))\n        x = self.drop(torch.relu(self.fc7(x)))\n        self.fc8(x)\n"), "error", "no-return"),
    ("small-alexnet", _a("32 * 4 * 4"), "error", "flat-size"),
    ("small-alexnet", _a("32 * 8 * 8"), "error", "flat-size"),
    ("small-alexnet", _a("12 * 3 * 3"), "error", "flat-size"),
    ("small-alexnet", _a(body="        x = nn.functional.dropout(torch.relu(self.fc6(x)), 0.5)\n        x = nn.functional.dropout(torch.relu(self.fc7(x)), 0.5)\n        return self.fc8(x)\n"), "wrong", "always-on"),
    ("small-alexnet", _A_START.replace("nn.Linear(..., 256)", "nn.Linear(288, 256)"), "wrong", "dots-left"),
    # fc7 left out: fc8 also takes 256 numbers, so nothing stops, but the logits differ
    ("small-alexnet", _a(body="        x = self.drop(torch.relu(self.fc6(x)))\n        return self.fc8(x)\n"), "wrong", None),
    # only one dropout, after fc6 or after fc7, or one on what goes into fc6: right in eval mode, and random in
    # training mode, but not the course's logits there
    ("small-alexnet", _a(body="        x = self.drop(torch.relu(self.fc6(x)))\n        x = torch.relu(self.fc7(x))\n        return self.fc8(x)\n"), "almost", None),
    ("small-alexnet", _a(body="        x = torch.relu(self.fc6(x))\n        x = self.drop(torch.relu(self.fc7(x)))\n        return self.fc8(x)\n"), "almost", None),
    ("small-alexnet", _a(body="        x = torch.relu(self.fc6(self.drop(x)))\n        x = torch.relu(self.fc7(x))\n        return self.fc8(x)\n"), "almost", None),
    # fc7 before fc6: a linear layer gets another number of inputs
    ("small-alexnet", _a(body="        x = self.drop(torch.relu(self.fc7(x)))\n        x = self.drop(torch.relu(self.fc6(x)))\n        return self.fc8(x)\n"), "error", "flat-size"),
] + [("small-alexnet", code, "wrong", key) for key, code in _A_WRONG.items()]


# --- the demo: the learner's SmallAlexNet on CIFAR-10 ---------------------------------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def _accuracy(model, X, y) -> int:
    model.eval()
    with torch.no_grad():
        return int(sum((model(X[i:i + 500]).argmax(dim=1) == y[i:i + 500]).sum() for i in range(0, len(X), 500)))


def run(variant: str, seed: int, SmallAlexNet, crop, jitter, channel_stats, Trainer, count_params, show=None):
    """One run of SmallAlexNet, as REFERENCE says: with dropout and the changes of `variant` (SETTINGS), the network
    made after torch.manual_seed(seed) and the photos shuffled after torch.manual_seed(seed + 1). Gives back a dict with
    the number of parameters, how many of the training photos as they are and of the test photos it reads right, the
    loss after every epoch ("curves"), the epoch with the lowest validation loss, and the network's guess for the photos
    SHOWN after every epoch ("guesses") with their right kinds ("truth"), and its first layer of feature maps on the car
    ("layers"). Errors go to the caller, which says whose code made them. `show` hears the run while it trains
    (live.follow, every 10 batches), with what the network makes then."""
    p, changed = SETTINGS[variant]
    X, y, X_test, y_test = photos()
    mean, std = channel_stats(X.clone())
    mean = torch.as_tensor(mean, dtype=torch.float32).reshape(3)
    std = torch.as_tensor(std, dtype=torch.float32).reshape(3)
    X_norm = (X - mean[:, None, None]) / std[:, None, None]
    X_test = (X_test - mean[:, None, None]) / std[:, None, None]
    torch.manual_seed(seed)
    model = SmallAlexNet(p=p)
    params = int(count_params(model))
    truth, guesses, layers = [int(y_test[place]) for place in SHOWN], [], []   # what the rewind keeps of every epoch
    heard = _AUG.hearing(model, X_test[list(SHOWN)], truth, X_test[_AUG.PHOTO:_AUG.PHOTO + 1], show, guesses, layers)

    torch.manual_seed(seed + 1)
    if changed:
        loader = DataLoader(_AUG._Augmented(X, y, mean, std, crop, jitter), batch_size=64, shuffle=True)
    else:
        loader = DataLoader(TensorDataset(X_norm, y), batch_size=64, shuffle=True)
    val = DataLoader(TensorDataset(X_test, y_test), batch_size=500)
    trainer = Trainer(model, loader, val, epochs=EPOCHS, lr=LR, warmup=len(loader), max_norm=1.0)
    live.follow(trainer, heard, "alexnet", every=EVERY)
    history = trainer.fit() or []
    curves = {"train": [float(h["train_loss"]) for h in history if isinstance(h, dict) and "train_loss" in h],
              "val": [float(h["val_loss"]) for h in history if isinstance(h, dict) and "val_loss" in h]}
    return {"variant": variant, "params": params, "train": _accuracy(model, X_norm, y), "test": _accuracy(model, X_test, y_test),
            "curves": curves, "best": best_epoch(curves["val"]), "guesses": guesses[:len(curves["val"])], "layers": layers[:len(curves["val"])],
            "truth": truth}


def _demo(variant: str, show=None):
    """For the demo: None when alexnet.py is not in the project yet; "data" when CIFAR-10 is not unpacked; "alexnet"
    when the learner's SmallAlexNet stops with an error or does not give 10 logits a photo; "changes" when their crop
    or jitter stops with an error or does not give a photo of (3, 32, 32), for a run with changes; "cnn" when
    channel_stats does not give what it should; "trainer" when the Trainer or count_params stops with an error; "slow"
    when the training takes too long; else the dict of run(), with the modules the course's versions stood in for."""
    if not _saved("alexnet"):
        return None
    if not ready():
        return "data"
    loaded = run_code("from cnn.alexnet import SmallAlexNet")
    if loaded.error is not None:
        return "alexnet"
    SmallAlexNet = loaded.names["SmallAlexNet"]
    crop = jitter = None   # only a run with changes uses them
    if SETTINGS[variant][1]:
        loaded = run_code("from cnn.crop import crop\nfrom cnn.color import jitter")
        if loaded.error is not None:
            return "changes"
        crop, jitter = loaded.names["crop"], loaded.names["jitter"]
    loaded = run_code("from cnn.channels import channel_stats")
    if loaded.error is not None:
        return "cnn"
    channel_stats = loaded.names["channel_stats"]
    loaded = run_code("from nn.trainer import Trainer\nfrom cnn.count import count_params")
    if loaded.error is not None:
        return "trainer"
    Trainer, count_params = loaded.names["Trainer"], loaded.names["count_params"]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    X, _, _, _ = photos()
    try:
        def tried():
            torch.manual_seed(0)
            model = SmallAlexNet(p=SETTINGS[variant][0])
            model.eval()
            with torch.no_grad():
                return tuple(model(X[:2]).shape)
        if with_time_limit(tried, 10) != (2, 10):
            return "alexnet"
    except Exception:
        return "alexnet"
    if SETTINGS[variant][1]:
        for change in (lambda x: crop(x, 0, 8), lambda x: jitter(x, 1.1, 0.9, 1.2)):
            try:
                out = with_time_limit(lambda: torch.as_tensor(change(X[0].clone())), 10)
                if tuple(out.shape) != (3, 32, 32) or not torch.isfinite(out).all():
                    return "changes"
            except Exception:
                return "changes"
    try:
        mean, std = with_time_limit(lambda: channel_stats(X.clone()), 60)
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(3)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(3)
        if not (torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
            return "cnn"
    except Exception:
        return "cnn"
    try:
        result = with_time_limit(lambda: run(variant, 0, SmallAlexNet, crop, jitter, channel_stats, Trainer, count_params, show), 600)
    except Timeout:
        return "slow"
    except Exception:
        return "trainer"
    return result | {"stand_ins": stand_ins}


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file), and whether CIFAR-10
    is there."""
    texts = []
    for package, module in (("cnn", "alexnet"), ("cnn", "crop"), ("cnn", "color"), ("cnn", "channels"),
                            ("cnn", "count"), ("nn", "train_epoch"), ("nn", "trainer")):
        path = projects.folder() / package / f"{module}.py"
        texts.append(path.read_text(encoding="utf-8") if path.exists() else None)
    return tuple(texts) + (ready(),)


def demo(variant: str = "both", show=None):
    """The demo's result for `variant`, worked out once for the project files as they are; `show` hears the run when it
    trains."""
    key = (variant,) + _project_texts()
    if key not in _DEMO_RUNS:
        _DEMO_RUNS.clear()
        _DEMO_RUNS[key] = _demo(variant, show)
    return _DEMO_RUNS[key]
