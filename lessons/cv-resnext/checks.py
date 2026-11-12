# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-resnext: a grouped convolution splits the channels into groups that each get filters of
their own, and a ResNeXt block puts one between a 1×1 convolution to `width` channels and one back, with the input
added at the end. The learner's ResNeXtBlock is saved in their project (work/projects/cnn/resnext.py) when right, and
checked on made-up numbers, so it works without CIFAR-10. The demo trains the CNN of cv-1x1-convolution with the
learner's block in its middle, in 1, 2, 4 or 32 groups with about the same number of weights, with their Trainer on
the first 6,000 training photos of CIFAR-10, and tests it on the first 1,000 test photos; after every epoch it keeps
the network's guess for one checked test photo of each kind, for rewinding."""
import html
import importlib.util
import re
from pathlib import Path

import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import chart, draw, live, paths, projects
from everything_ai.check import Mistake, ScenarioBox, Sort, Timeout, run_code, with_time_limit
from everything_ai import netview, yours


def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_resnext",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ONE = _lesson_checks("cv-1x1-convolution")   # the demo's network, CIFAR-10 and its button, the photos, the rewind

# The demo uses the learner's channel_stats of cv-color-cnn, their Trainer of nn-trainer and their count_params of
# cnn-pytorch; where one is missing or broken, the course's version stands in (template 1.3).
BUILDS_ON = _ONE.BUILDS_ON
projects.builds_on(*BUILDS_ON)
DOWNLOAD_LINE = _ONE.DOWNLOAD_LINE

cifar_or_stop, ready, photos = _ONE.cifar_or_stop, _ONE.ready, _ONE.photos
grouped, number, percent, low_high = _ONE.grouped, _ONE.number, _ONE.percent, _ONE.low_high
guesses_picture, moments, best_epoch = _ONE.guesses_picture, _ONE.moments, _ONE.best_epoch
SHOWN, N_TRAIN, N_TEST = _ONE.SHOWN, _ONE.N_TRAIN, _ONE.N_TEST
EVERY = _ONE.EVERY   # the live chart hears the training loss of every EVERY batches
EPOCHS, LR, CHANNELS, SIDE = _ONE.EPOCHS, _ONE.LR, _ONE.CHANNELS, _ONE.SIDE
conv_params = _ONE.conv_params
WideCNN, Plain, AROUND, PLAIN_BLOCK = _ONE.WideCNN, _ONE.Plain, _ONE.AROUND, _ONE.PLAIN_BLOCK


# --- counting -------------------------------------------------------------------------------------------------------

def grouped_weights(c_in: int, c_out: int, k: int, groups: int) -> int:
    """The weights of nn.Conv2d(c_in, c_out, k, groups=groups): c_out filters, each of c_in / groups × k × k."""
    return c_out * (c_in // groups) * k * k


def block_weights(channels: int, width: int, groups: int) -> int:
    """The weights of a ResNeXt block: a 1×1 convolution from `channels` to `width`, a 3×3 one from `width` to `width`
    in `groups` groups, and a 1×1 one back to `channels`. With C groups of d channels, the paper's C·(256·d + 3·3·d·d +
    d·256) for 256 channels."""
    return channels * width + grouped_weights(width, width, 3, groups) + width * channels


def block_params(channels: int, width: int, groups: int) -> int:
    """The parameters of a ResNeXt block, as nn.Conv2d makes its three convolutions: with a bias for every filter."""
    return block_weights(channels, width, groups) + width + width + channels


# The demo's blocks on the 64 maps of 8 by 8 in the middle of WideCNN: C groups of d channels, the settings of the
# paper's table 2 for 256 channels (1×64d, 2×40d, 4×24d, 32×4d) divided by 4, so that they have about the same number
# of weights: 4,352, 4,360, 4,368 and 4,384. Its 8×14d would give 8 groups of 3.5 channels, so it is left out.
VARIANTS = {"1x16": (1, 16), "2x10": (2, 10), "4x6": (4, 6), "32x1": (32, 1)}


def width_groups(variant: str) -> tuple[int, int]:
    """The width of the middle and the number of groups of `variant`."""
    groups, d = VARIANTS[variant]
    return groups * d, groups


# --- the block, as the course writes it -----------------------------------------------------------------------------

_R_HEAD = ("import torch\nfrom torch import nn\n\n\nclass ResNeXtBlock(nn.Module):\n"
           "    def __init__(self, channels, width, groups):\n        super().__init__()\n"
           "        self.reduce = nn.Conv2d(channels, width, 1)\n")
_R_EXPAND = "        self.expand = nn.Conv2d(width, channels, 1)\n"
_R_FORWARD = ("\n    def forward(self, x):\n        y = torch.relu(self.reduce(x))\n        y = torch.relu(self.conv(y))\n"
              "        y = self.expand(y)\n")


def _r(conv: str = "nn.Conv2d(width, width, 3, padding=1, groups=groups)", back: str = "x + y") -> str:
    return _R_HEAD + f"        self.conv = {conv}\n" + _R_EXPAND + _R_FORWARD + f"        return {back}\n"


_R = _r()


# --- the demo: the learner's ResNeXtBlock in the middle of WideCNN, on CIFAR-10 -------------------------------------

def _accuracy(model, X, y) -> int:
    model.eval()
    with torch.no_grad():
        return int(sum((model(X[i:i + 500]).argmax(dim=1) == y[i:i + 500]).sum() for i in range(0, len(X), 500)))


def run(variant: str, seed: int, ResNeXtBlock, channel_stats, Trainer, count_params, show=None):
    """One run of WideCNN with ResNeXtBlock(CHANNELS, width, groups) of `variant` in its middle, as REFERENCE says: the
    block made after torch.manual_seed(seed), then the rest of the network, and the photos shuffled after
    torch.manual_seed(seed + 1). Gives back a dict with the parameters of the network and of its block, how many of the
    training and test photos it reads right, the loss after every epoch ("curves"), the epoch with the lowest validation
    loss, and the network's guess for the photos SHOWN after every epoch ("guesses") with their right kinds ("truth"),
    and the feature maps of its middle on the car ("layers"). Errors go to the caller, which says whose code made them.
    `show` hears the run while it trains (live.follow, every 10 batches), with what the network makes then."""
    X, y, X_test, y_test = photos()
    mean_, std = channel_stats(X.clone())
    mean_ = torch.as_tensor(mean_, dtype=torch.float32).reshape(3)
    std = torch.as_tensor(std, dtype=torch.float32).reshape(3)
    X = (X - mean_[:, None, None]) / std[:, None, None]
    X_test = (X_test - mean_[:, None, None]) / std[:, None, None]
    torch.manual_seed(seed)
    block = ResNeXtBlock(CHANNELS, *width_groups(variant))
    model = WideCNN(block)
    params, block_params_ = int(count_params(model)), int(count_params(block))
    truth, guesses, layers = [int(y_test[place]) for place in SHOWN], [], []   # what the rewind keeps of every epoch
    heard = _ONE.hearing(model, X_test[list(SHOWN)], truth, X_test[_ONE.PHOTO:_ONE.PHOTO + 1], show, guesses, layers, "block")

    torch.manual_seed(seed + 1)
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    val = DataLoader(TensorDataset(X_test, y_test), batch_size=500)
    trainer = Trainer(model, loader, val, epochs=EPOCHS, lr=LR, warmup=len(loader), max_norm=1.0)
    live.follow(trainer, heard, "resnext", every=EVERY)
    history = trainer.fit() or []
    curves = {"train": [float(h["train_loss"]) for h in history if isinstance(h, dict) and "train_loss" in h],
              "val": [float(h["val_loss"]) for h in history if isinstance(h, dict) and "val_loss" in h]}
    return {"variant": variant, "params": params, "block_params": block_params_, "train": _accuracy(model, X, y),
            "test": _accuracy(model, X_test, y_test), "curves": curves, "best": best_epoch(curves["val"]),
            "guesses": guesses[:len(curves["val"])], "layers": layers[:len(curves["val"])], "truth": truth}


# A block of ResNeXt-50 with 256 channels in and out (Xie et al., 2016, figure 1, and table 1, conv2, 32×4d): a 1×1 convolution to 128, a 3×3
# one from 128 to 128 in 32 groups of 4, and a 1×1 one back to 256. The paper's equation 4 counts the weights:
# 32·(256·4 + 3·3·4·4 + 4·256) = 70,144, "about 70k", as for ResNet-50's bottleneck of cv-1x1-convolution, 69,632
# weights (70,016 parameters with biases). Its table 3 gives the top-1 error on ImageNet of ResNet-50 (1×64d) and of
# ResNeXt-50 with the same cost: 2×40d, 4×24d, 8×14d and 32×4d; table 1 the parameters and multiply-adds.
R50 = (256, 128, 32)
R50_WEIGHTS = block_weights(*R50)                 # 70,144
R50_PARAMS = block_params(*R50)                   # 70,656
R50_ONE_GROUP = block_params(256, 128, 1)         # 213,504: the same block with one group
RESNET_BOTTLENECK = _ONE.RESNET["bottleneck"]     # 70,016
PAPER_ERROR = {"1×64d": 23.9, "2×40d": 23.0, "4×24d": 22.6, "8×14d": 22.3, "32×4d": 22.2}
# The worked example: 128 channels in 32 groups of 4.
EXAMPLE_WEIGHTS = grouped_weights(128, 128, 3, 32)   # 4,608
EXAMPLE_ONE_GROUP = grouped_weights(128, 128, 3, 1)  # 147,456


# --- what the course measured ---------------------------------------------------------------------------------------

# What the course measured, on one thread, with the course's code, through run(): ResNeXtBlock(CHANNELS, width, groups)
# of each variant made after torch.manual_seed(seed), then the rest of WideCNN, trained as in cv-1x1-convolution (the
# Trainer of nn-trainer with AdamW, a learning rate of LR, a warmup of one epoch, cosine decay, max_norm 1.0,
# mini-batches of 64) on the first N_TRAIN training photos, shuffled after torch.manual_seed(seed + 1), for EPOCHS
# epochs and the seeds 0 to 4, with the photos normalized by channel_stats of the training photos. "train" and "test"
# are how many of the training photos and of the first N_TEST test photos are read right after the last epoch; "best"
# is the first epoch with the lowest validation loss, rounded to 3 decimals, and "lowest" that loss; "last_val" and
# "last_train" are the validation and the training loss of the last epoch. CURVES are seed 0. The runs with one 3×3
# convolution ("plain") and with the bottleneck with 16 channels in the middle ("b16": the convolutions of "1x16",
# without the input added) are those of cv-1x1-convolution, which builds them the same way.
# tests/test_cv_resnext.py measures seed 0 of DEFAULT again when EAI_CIFAR_FILE names the unpacked cifar-10.npz.
REFERENCE = {
    "plain": _ONE.REFERENCE["plain"],
    "b16": _ONE.REFERENCE["b16"],
    "1x16": {"train": [5997, 5997, 5977, 5999, 5987], "test": [598, 585, 562, 591, 590], "best": [6, 8, 4, 4, 7],
             "lowest": [1.347, 1.311, 1.313, 1.273, 1.341], "last_val": [2.38, 2.228, 2.202, 2.273, 2.175],
             "last_train": [0.024, 0.032, 0.055, 0.018, 0.037]},
    "2x10": {"train": [5993, 5977, 5996, 5993, 5984], "test": [565, 569, 580, 569, 565], "best": [6, 5, 4, 4, 3],
             "lowest": [1.322, 1.295, 1.365, 1.339, 1.307], "last_val": [2.376, 2.185, 2.233, 2.21, 2.15],
             "last_train": [0.028, 0.05, 0.029, 0.033, 0.042]},
    "4x6": {"train": [5995, 5980, 5938, 5999, 5999], "test": [599, 576, 572, 594, 581], "best": [6, 3, 7, 6, 3],
            "lowest": [1.305, 1.284, 1.288, 1.253, 1.309], "last_val": [2.336, 2.202, 1.963, 2.16, 2.239],
            "last_train": [0.024, 0.046, 0.089, 0.021, 0.024]},
    "32x1": {"train": [5998, 5992, 5995, 5996, 5996], "test": [574, 580, 584, 582, 580], "best": [4, 8, 4, 4, 4],
             "lowest": [1.304, 1.305, 1.33, 1.269, 1.313], "last_val": [2.363, 2.071, 2.22, 2.172, 2.334],
             "last_train": [0.022, 0.032, 0.026, 0.026, 0.019]},
}
CURVES = {
    "1x16_val": [1.493, 1.416, 1.43, 1.356, 1.436, 1.347, 1.421, 1.58, 1.518, 1.608, 1.743, 1.877, 2.045, 2.073, 2.157,
                2.247, 2.322, 2.368, 2.378, 2.38],
    "1x16_train": [1.824, 1.461, 1.304, 1.158, 1.057, 0.96, 0.844, 0.721, 0.638, 0.537, 0.429, 0.351, 0.243, 0.17, 0.11,
                  0.068, 0.042, 0.03, 0.025, 0.024],
    "2x10_val": [1.578, 1.379, 1.422, 1.387, 1.394, 1.322, 1.48, 1.421, 1.481, 1.645, 1.658, 1.742, 1.925, 2.068, 2.117,
                2.208, 2.311, 2.358, 2.375, 2.376],
    "2x10_train": [1.831, 1.508, 1.316, 1.18, 1.076, 0.964, 0.854, 0.768, 0.664, 0.582, 0.466, 0.372, 0.278, 0.194,
                  0.129, 0.082, 0.049, 0.035, 0.029, 0.028],
    "4x6_val": [1.558, 1.547, 1.406, 1.418, 1.405, 1.305, 1.388, 1.424, 1.441, 1.56, 1.621, 1.823, 1.796, 1.994, 2.115,
               2.249, 2.287, 2.317, 2.333, 2.336],
    "4x6_train": [1.834, 1.507, 1.325, 1.193, 1.056, 0.967, 0.884, 0.773, 0.677, 0.566, 0.453, 0.356, 0.264, 0.178,
                 0.114, 0.069, 0.042, 0.03, 0.025, 0.024],
    "32x1_val": [1.628, 1.534, 1.442, 1.304, 1.325, 1.381, 1.392, 1.401, 1.432, 1.533, 1.675, 1.824, 1.995, 2.118,
                2.146, 2.264, 2.324, 2.342, 2.36, 2.363],
    "32x1_train": [1.859, 1.507, 1.312, 1.157, 1.046, 0.951, 0.832, 0.772, 0.664, 0.549, 0.416, 0.337, 0.245, 0.161,
                  0.108, 0.061, 0.037, 0.027, 0.023, 0.022],
}
DEFAULT = "32x1"   # the demo's block when the learner chooses none: as many groups as ResNeXt-50 has
RESIDUAL = ("1x16", "2x10", "4x6", "32x1")


def mean(values) -> float:
    return sum(values) / len(values)


def all_runs(key: str) -> list[int]:
    """The test photos read right by all the runs of the four ResNeXt blocks together, or by the runs of `key`."""
    return [v for k in RESIDUAL for v in REFERENCE[k][key]]


# --- the picture of the runs ----------------------------------------------------------------------------------------

ROWS = ("plain", "b16", "1x16", "2x10", "4x6", "32x1")


def runs_picture(labels: dict, lang: str, mine: dict | None = None) -> str:
    """How many of the N_TEST test photos the network reads right after EPOCHS epochs, for each block in its middle:
    one 3×3 convolution and the bottleneck with 16 channels in the middle of cv-1x1-convolution, both without the input
    added, and the four ResNeXt blocks; an open dot for every seed of the course and a line at their mean. With `mine`
    ({"variant", "test"}), the learner's run as a filled dot in the row of its block. `labels` gives "title", "x", a
    name for each row, "mine", "course" and "alt"."""
    yours = {} if mine is None else {mine["variant"]: mine["test"]}
    # from 500 to 620, wider where the learner's run lies outside, so that every dot is where its number is
    low, high = min([500] + [v - 15 for v in yours.values()]), max([620] + [v + 15 for v in yours.values()])
    tick = 20 if high - low <= 160 else 50 if high - low <= 400 else 100
    low, high = int(low // tick * tick), int(-(-high // tick) * tick)
    p = chart.Panel((low, high), (-1.0, len(ROWS) - 0.3), title=labels["title"], xlabel=labels["x"], yticks=[],
                    xticks=list(range(low, high + 1, tick)))
    step = (high - low) / 220
    for k, key in enumerate(ROWS):
        y = len(ROWS) - 1 - k
        values = REFERENCE[key]["test"]
        m = mean(values)
        p.line([m, m], [y - 0.22, y + 0.22], "ink-3", 1.6)
        p.points(values, [y] * len(values), "ink", r=3.5, hollow=True)
        p.text(low + step, y + 0.45, labels[key], "ink-2", "start")
        if key in yours:
            p.points([yours[key]], [y], "ink", r=4.5)
    if mine is not None:
        p.points([low + 6 * step], [-0.75], "ink", r=4.5)
        p.text(low + 12 * step, -0.75, labels["mine"], "ink-2", "start")
        p.points([low + 106 * step], [-0.75], "ink", r=3.5, hollow=True)
        p.text(low + 112 * step, -0.75, labels["course"], "ink-2", "start")
    svg = chart.figure([p], alt=labels["alt"], width=440, height=60 + 44 * len(ROWS) + (24 if mine is not None else 0))
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


curves_picture = _ONE.curves_picture   # its gray line: the course's run with one 3×3 convolution, seed 0


def now_pictures(seen: dict, title: str, lang: str, uid: str) -> list:
    """What the network makes at one moment of its run: its guesses for the photos SHOWN, and the feature maps of its
    middle on the car (cv-augment-photos)."""
    return _ONE.now_pictures(seen, title, lang, uid)


def epoch_frame(result: dict, e: int, labels: dict, title: str, lang: str) -> str:
    """The rewind at epoch e (from 1): the loss after every epoch with a line at e, then the network's guesses for the
    photos SHOWN after it, and under them the feature maps of its middle on the car, as it was then (cv-augment-photos)."""
    return _ONE.rewind_frame(curves_picture, result, e, labels, title, lang, "resnext")


# The lesson is done when these are correct. "groups-runs" is the choice question in the lesson.
REQUIRED = ["groups", "resnext", "groups-runs"]


# --- the picture of the groups --------------------------------------------------------------------------------------

def groups_picture(labels: dict, channels: int = 8, groups: tuple = (1, 4)):
    """Which channels each filter sees, for a convolution from `channels` to `channels`: a square for every filter (a
    row) and every channel that comes in (a column), in the colour of what a model learns where the filter has 3 × 3
    weights for that channel, empty where it has none; one grid for each number of groups. `labels` gives "title",
    "in", "out", "sub" (with {g} for the groups and {n} for the orange squares) and "alt"."""
    CELL, GAP, TOP, LEFT = 18, 70, 52, 30
    side = channels * CELL
    W = LEFT + len(groups) * side + (len(groups) - 1) * GAP + 4
    small = "font-size:12.5px;fill:var(--ea-ink-2)"
    parts = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    for i, g in enumerate(groups):
        x0 = LEFT + i * (side + GAP)
        size = channels // g
        squares = 0
        for row in range(channels):
            for col in range(channels):
                seen = row // size == col // size
                squares += seen
                style = ("fill:var(--ea-param-wash);stroke:var(--ea-param);stroke-width:1" if seen
                         else "fill:none;stroke:var(--ea-line);stroke-width:0.8")
                parts.append(f'<rect x="{x0 + col * CELL + 1}" y="{TOP + row * CELL + 1}" width="{CELL - 2}" height="{CELL - 2}" rx="2" style="{style}"/>')
        parts.append(f'<text x="{x0 + side / 2}" y="{TOP - 10}" style="{small};text-anchor:middle">{html.escape(labels["in"])}</text>')
        parts.append(f'<text x="{x0 - 8}" y="{TOP + side / 2}" transform="rotate(-90 {x0 - 8} {TOP + side / 2})" style="{small};text-anchor:middle">{html.escape(labels["out"])}</text>')
        parts.append(f'<text x="{x0 + side / 2}" y="{TOP + side + 22}" style="font-size:13.5px;font-weight:600;fill:var(--ea-ink);text-anchor:middle">'
                     f'{html.escape(labels["sub"].format(g=g, n=squares))}</text>')
    return draw._svg(W, TOP + side + 32, labels["alt"], "".join(parts), natural=True)


# --- the slider: how many groups the 3×3 convolution of a block of ResNeXt-50 has -----------------------------------

GROUPS = [1, 2, 4, 8, 16, 32, 64, 128]


def layer_params(groups: int) -> tuple[int, int, int]:
    """The parameters of the three convolutions of a block of ResNeXt-50 with 256 channels, with its 3×3 convolution
    in `groups` groups: the 1×1 convolution from 256 to 128, the 3×3 one, and the 1×1 one back to 256."""
    return conv_params(256, 128), grouped_weights(128, 128, 3, groups) + 128, conv_params(128, 256)


def first_below() -> int:
    """The fewest groups for which the 3×3 convolution has fewer parameters than the 1×1 convolution in front of it."""
    return next(g for g in GROUPS if layer_params(g)[1] < layer_params(g)[0])


def groups_sentence(g: int, lang: str) -> str:
    """The sentence under the slider, for `g` groups."""
    reduce, conv, expand = layer_params(g)
    r, c, e, t = (grouped(n, lang) for n in (reduce, conv, expand, reduce + conv + expand))
    size = 128 // g
    if lang == "nl":
        each = "1 channel" if size == 1 else f"{size} channels"
        return (f"Met {g} {'groep' if g == 1 else 'groepen'}: elk filter van de 3×3-convolutie ziet {each}. Die convolutie heeft {c} parameters, "
                f"de 1×1-convolutie ervoor {r} en die erna {e}: samen {t}.")
    each = "1 channel" if size == 1 else f"{size} channels"
    return (f"With {g} {'group' if g == 1 else 'groups'}: each filter of the 3×3 convolution sees {each}. That convolution has {c} parameters, "
            f"the 1×1 convolution in front of it {r} and the one after it {e}: {t} together.")


def groups_bars(g: int, labels: dict, lang: str) -> str:
    """Three bars of parameters, in thousands, on an axis that stays the same for every number of groups: the 1×1
    convolution in front, the 3×3 convolution in `g` groups, and the 1×1 convolution after it, each in the colour of
    what a model learns, with its name above it. `labels` gives "title", "x", "reduce", "conv", "expand" and "alt"."""
    values = layer_params(g)
    p = chart.Panel((0, 200), (-0.6, 3.1), title=labels["title"], xlabel=labels["x"], yticks=[], xticks=[0, 50, 100, 150, 200])
    for k, (key, v) in enumerate(zip(("reduce", "conv", "expand"), values)):
        y = 2 - k
        p.text(0, y + 0.5, labels[key], "ink-2", "start")
        p.rect(0, y - 0.2, v / 1000, y + 0.2, "param")
        p.text(v / 1000 + 2, y - 0.02, grouped(v, lang), "ink", "start")
    svg = chart.figure([p], alt=labels["alt"], width=440, height=230)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


# --- exercise 1: which groups PyTorch accepts -----------------------------------------------------------------------

_SORT = Sort(
    name="groups",
    items={
        "four": {"en": "**`nn.Conv2d(128, 128, 3, groups=32)`**", "nl": "**`nn.Conv2d(128, 128, 3, groups=32)`**"},
        "hundred": {"en": "**`nn.Conv2d(128, 100, 3, groups=32)`**", "nl": "**`nn.Conv2d(128, 100, 3, groups=32)`**"},
        "three": {"en": "**`nn.Conv2d(128, 128, 3, groups=3)`**", "nl": "**`nn.Conv2d(128, 128, 3, groups=3)`**"},
        "each": {"en": "**`nn.Conv2d(64, 64, 3, groups=64)`**", "nl": "**`nn.Conv2d(64, 64, 3, groups=64)`**"},
        "eight": {"en": "**`nn.Conv2d(96, 64, 1, groups=8)`**", "nl": "**`nn.Conv2d(96, 64, 1, groups=8)`**"},
        "half": {"en": "**`nn.Conv2d(64, 32, 3, groups=64)`**", "nl": "**`nn.Conv2d(64, 32, 3, groups=64)`**"},
    },
    groups={"works": {"en": "works: channels and filters split evenly", "nl": "werkt: channels en filters zijn eerlijk te verdelen"},
            "stops": {"en": "PyTorch stops", "nl": "PyTorch stopt"}},
    solution={"four": "works", "hundred": "stops", "three": "stops", "each": "works", "eight": "works", "half": "stops"},
    nudge={
        "four": {"en": "How many of the 128 channels does each of the 32 groups get, and how many of the 128 filters?",
                 "nl": "Hoeveel van de 128 channels krijgt elk van de 32 groepen, en hoeveel van de 128 filters?"},
        "hundred": {"en": "The 100 filters are split over 32 groups too. Is 100 / 32 a whole number?",
                    "nl": "De 100 filters worden ook over 32 groepen verdeeld. Is 100 / 32 een heel getal?"},
        "three": {"en": "Can 128 channels be split into 3 groups of the same size?",
                  "nl": "Kun je 128 channels verdelen in 3 even grote groepen?"},
        "each": {"en": "64 channels in 64 groups: how many channels does each group get? Is that a whole number?",
                 "nl": "64 channels in 64 groepen: hoeveel channels krijgt elke groep? Is dat een heel getal?"},
        "eight": {"en": "Work out 96 / 8 and 64 / 8. Are both whole numbers?",
                  "nl": "Reken 96 / 8 en 64 / 8 uit. Zijn het allebei hele getallen?"},
        "half": {"en": "The 32 filters are split over 64 groups. How many filters does each group get?",
                 "nl": "De 32 filters worden over 64 groepen verdeeld. Hoeveel filters krijgt elke groep?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the six convolutions again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de zes convoluties opnieuw uit."},
        "empty": {"en": "Sort each convolution: it works, or PyTorch stops.",
                  "nl": "Deel elke convolutie in: hij werkt, of PyTorch stopt."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "{item} belongs in the other group.", "nl": "{item} hoort in de andere groep."},
        "correct": {"en": "All {n} are right. The channels that come in and the filters are both split into the groups, so both numbers must be divisible by `groups`: 128 / 32 = 4 channels and 4 filters per group, 64 / 64 = 1, and 96 / 8 = 12 channels with 64 / 8 = 8 filters. 100 / 32 and 32 / 64 are not whole numbers, and 128 / 3 neither: PyTorch stops with `out_channels must be divisible by groups` or `in_channels must be divisible by groups`.",
                    "nl": "Alle {n} kloppen. De channels die binnenkomen en de filters worden allebei over de groepen verdeeld, dus moeten beide getallen deelbaar zijn door `groups`: 128 / 32 = 4 channels en 4 filters per groep, 64 / 64 = 1, en 96 / 8 = 12 channels met 64 / 8 = 8 filters. 100 / 32 en 32 / 64 zijn geen hele getallen, en 128 / 3 ook niet: PyTorch stopt met `out_channels must be divisible by groups` of `in_channels must be divisible by groups`."},
    },
)

# --- exercise 2: a ResNeXt block ------------------------------------------------------------------------------------

_R_START = (_R_HEAD + "        # conv: 3×3, width to width in `groups` groups, keeps the size\n        self.conv = ...\n"
            + _R_EXPAND + _R_FORWARD + "        # add the input x\n        return ...\n")
# The scenarios give the three convolutions their weights by name, in this order, so that the order in which __init__
# makes them does not change the numbers.
_R_WEIGHTS = ("with torch.no_grad():\n    for layer in (block.reduce, block.conv, block.expand):\n"
              "        for q in layer.parameters():\n            q.copy_(torch.randn_like(q) / 4)\n")
_R_SCENARIOS = [
    # a block of ResNeXt-50 with 256 channels, on made-up maps of 8 by 8: the shape, the weights of conv, and the number of parameters
    "import torch\n\nblock = ResNeXtBlock(256, 128, 32)\nout = block(torch.rand(1, 256, 8, 8))\n"
    "[list(out.shape), list(block.conv.weight.shape), sum(q.numel() for q in block.parameters())]",
    # a small block: the numbers of every channel at a pixel in the middle and at a corner
    "import torch\n\nblock = ResNeXtBlock(6, 4, 2)\ntorch.manual_seed(0)\n" + _R_WEIGHTS
    + "out = block(torch.randn(1, 6, 5, 5))\n[out[0, :, 2, 2].tolist(), out[0, :, 0, 0].tolist()]",
    # another size: the weights of conv and the shape
    "import torch\n\ntorch.manual_seed(1)\nblock = ResNeXtBlock(8, 6, 3)\nout = block(torch.randn(2, 8, 7, 7))\n"
    "[list(block.conv.weight.shape), list(out.shape)]",
]
_R_WRONG = {
    # the 3×3 convolution over all channels: one group
    "no-groups": _r(conv="nn.Conv2d(width, width, 3, padding=1)"),
    "no-residual": _r(back="y"),
    # a ReLU after the addition, as the first ResNet of 2015 and ResNeXt's paper have
    "relu-after": _r(back="torch.relu(x + y)"),
    # a ReLU at the end of f, on y before the input is added
    "relu-end": _r(back="x + torch.relu(y)"),
}
_NO_GROUPS = block_params(256, 128, 1)   # 213,504
_FIXED = Mistake("fixed", matches=lambda got, text: re.search(r"self\.conv\s*=.*\bgroups\s*=\s*\d", text) is not None
                 or re.search(r"self\.conv\s*=\s*nn\.Conv2d\(\s*(\d|[A-Za-z_]\w*\s*,\s*\d+\s*,)|self\.conv\s*=\s*nn\.Conv2d\([^)\n]*\b(in_channels|out_channels)\s*=\s*\d", text) is not None,
                 en="`conv` has a fixed number where `width` or `groups` belongs. The block must work for every size: use the names `width` and `groups`, as `reduce` and `expand` use `channels` and `width`.",
                 nl="`conv` heeft een vast getal waar `width` of `groups` hoort. Het block moet voor elke grootte werken: gebruik de namen `width` en `groups`, zoals `reduce` en `expand` `channels` en `width` gebruiken.")
# `...` on a line of its own, or after `=` or `return`
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$|=\s*\.\.\.|\breturn\s+\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace both with your own code: `conv`, and what `forward` gives back.",
                nl="Er staat nog een `...` in je code. Vervang ze allebei door je eigen code: `conv`, en wat `forward` teruggeeft.")
_TEXTS = _ONE._TEXTS
_S_EN, _S_NL = _ONE._S_EN, _ONE._S_NL
_close = _ONE._close


EXERCISES = {
    "groups": _SORT,
    "resnext": ScenarioBox(
        name="resnext",
        picture=netview.network("ResNeXtBlock", lambda: torch.rand(1, 256, 8, 8), 256, 128, 32, scores=False),
        start=_R_START,
        solution=_R,
        scenarios=_R_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "resnext"),
        first=[_DOTS, _FIXED],
        mistakes=[
            Mistake("padding", error=lambda e: isinstance(e, RuntimeError) and "must match the size of tensor" in str(e),
                    en="`x + y` stops: `y` has other heights and widths than `x`. A 3×3 convolution without padding makes the maps 2 smaller; with `padding=1` it keeps them, and then `y` fits `x`.",
                    nl="`x + y` stopt: `y` heeft andere hoogtes en breedtes dan `x`. Een 3×3-convolutie zonder padding maakt de maps 2 kleiner; met `padding=1` houdt hij ze, en dan past `y` bij `x`."),
            Mistake("divisible", error=lambda e: isinstance(e, ValueError) and "divisible by groups" in str(e),
                    en="PyTorch stops: the channels of `conv` cannot be split into groups of the same size. Does `conv` go from `width` to `width`, with `groups=groups`?",
                    nl="PyTorch stopt: de channels van `conv` zijn niet te verdelen in even grote groepen. Gaat `conv` van `width` naar `width`, met `groups=groups`?"),
            Mistake("channels", error=lambda e: isinstance(e, RuntimeError) and "channels" in str(e) and "expected" in str(e),
                    en="`conv` gets another number of channels than it was made for. `reduce` gives `width` channels, so `conv` goes from `width` to `width`, and `expand` takes them back to `channels`.",
                    nl="`conv` krijgt een ander aantal channels dan waarvoor hij gemaakt is. `reduce` geeft `width` channels, dus `conv` gaat van `width` naar `width`, en `expand` brengt ze terug naar `channels`."),
            Mistake("no-return", error=lambda e: isinstance(e, AttributeError) and "'NoneType' object has no attribute" in str(e),
                    en="The check asks for the shape of what your block gives, and `forward` gives back nothing, `None`: end it with `return`.",
                    nl="De check vraagt de shape van wat je block geeft, en `forward` geeft niets terug, `None`: eindig hem met `return`."),
            Mistake("no-groups", wrong=_R_WRONG["no-groups"],
                    en=_S_EN + f": your block, with the numbers of a block of ResNeXt-50, has {grouped(_NO_GROUPS, 'en')} parameters instead of {grouped(R50_PARAMS, 'en')}, and `conv.weight` has the shape (128, 128, 3, 3): every filter sees all 128 channels. Give `conv` the argument `groups=groups`.",
                    nl=_S_NL + f": je block, met de getallen van een block van ResNeXt-50, heeft {grouped(_NO_GROUPS, 'nl')} parameters in plaats van {grouped(R50_PARAMS, 'nl')}, en `conv.weight` heeft de shape (128, 128, 3, 3): elk filter ziet alle 128 channels. Geef `conv` het argument `groups=groups`."),
            Mistake("no-residual", wrong=_R_WRONG["no-residual"],
                    en=_S_EN + ": your block gives only `y`, as a plain network does. A ResNeXt block is a residual block: it adds its input, `return x + y`.",
                    nl=_S_NL + ": je block geeft alleen `y`, zoals een gewoon netwerk. Een ResNeXt block is een residual block: het telt zijn input erbij op, `return x + y`."),
            Mistake("relu-after", wrong=_R_WRONG["relu-after"],
                    en=_S_EN + ": you put a ReLU after the addition, like ResNeXt's paper and the first ResNet of 2015. Where `x + y` is below 0, it then gives 0, and the direct path is cut. As in your block of [ResNet](/course/open/cnn-resnet?lang=en), give back `x + y` as it is.",
                    nl=_S_NL + ": je zet een ReLU na het optellen, net als de paper van ResNeXt en het eerste ResNet van 2015. Waar `x + y` onder 0 is, geeft die dan 0, en is het directe pad doorgeknipt. Geef, zoals in je block uit [ResNet](/course/open/cnn-resnet?lang=nl), `x + y` terug zoals het is."),
            Mistake("relu-end", wrong=_R_WRONG["relu-end"],
                    en=_S_EN + ": you put a ReLU on `y` before adding it, so the block can only add to `x`, never take away. The ReLUs go after `reduce` and after `conv` only, as the box has them.",
                    nl=_S_NL + ": je zet een ReLU op `y` voordat je het optelt, dus kan het block alleen bij `x` optellen, nooit eraf halen. De ReLU's staan alleen na `reduce` en na `conv`, zoals in het vak."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...` with your code: `conv`, and what `forward` gives back.",
                      "nl": "Vervang de twee `...` door je code: `conv`, en wat `forward` teruggeeft."},
            "wrong": {"en": _S_EN + ". Is `conv` a 3×3 convolution from `width` to `width` with `padding=1` and `groups=groups`, and does `forward` give back `x + y`?",
                      "nl": _S_NL + ". Is `conv` een 3×3-convolutie van `width` naar `width` met `padding=1` en `groups=groups`, en geeft `forward` `x + y` terug?"},
            "almost": {"en": "The shape and the number of parameters are right, but " + _S_EN[0].lower() + _S_EN[1:] + ". Did you change the lines of `forward` that were in the box? It sends `x` through `reduce`, a ReLU, `conv`, a ReLU and `expand`, and gives back `x + y`.",
                       "nl": "De shape en het aantal parameters kloppen, maar " + _S_NL[0].lower() + _S_NL[1:] + ". Heb je de regels van `forward` veranderd die in het vak stonden? Het stuurt `x` door `reduce`, een ReLU, `conv`, een ReLU en `expand`, en geeft `x + y` terug."},
            "correct": {"en": f"Your `ResNeXtBlock` is right: with the numbers of a block of ResNeXt-50, it has {grouped(R50_PARAMS, 'en')} parameters, and each of its 128 filters of 3 by 3 sees 4 channels. It is now in your project, as `{{saved}}`.",
                        "nl": f"Je `ResNeXtBlock` klopt: met de getallen van een block van ResNeXt-50 heeft hij {grouped(R50_PARAMS, 'nl')} parameters, en elk van zijn 128 filters van 3 bij 3 ziet 4 channels. Hij staat nu in je project, als `{{saved}}`."},
            "kept": {"en": "Your `ResNeXtBlock` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `ResNeXtBlock` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_R_ORDER = _R.replace("        self.reduce = nn.Conv2d(channels, width, 1)\n", "").replace(
    _R_EXPAND, _R_EXPAND + "        self.reduce = nn.Conv2d(channels, width, 1)\n")
CASES = [
    ("groups", {}, "empty", None),
    ("groups", _SORT.solution, "correct", None),
    ("groups", {"four": "works", "hundred": "stops"}, "wrong", "unfinished"),
    ("groups", {**_SORT.solution, "each": "stops"}, "wrong", "wrong:each"),
    ("groups", {**_SORT.solution, "three": "works"}, "wrong", "wrong:three"),
    ("resnext", _R_START, "empty", None),
    ("resnext", _R, "correct", None),
    # the layers made in another order: the scenarios give them their weights by name
    ("resnext", _R_ORDER, "correct", None),
    # other ways to write the same: padding "same", arguments by name, y + x, torch.add
    ("resnext", _r(conv="nn.Conv2d(width, width, 3, padding=\"same\", groups=groups)"), "correct", None),
    ("resnext", _r(conv="nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, padding=1, groups=groups)"), "correct", None),
    ("resnext", _r(back="y + x"), "correct", None),
    ("resnext", _r(back="torch.add(x, y)"), "correct", None),
    # a ReLU after the addition, written another way: known by what comes out
    ("resnext", _r(back="nn.functional.relu(y + x)"), "wrong", "relu-after"),
    ("resnext", _r(conv="nn.Conv2d(width, width, 3, padding=1, groups=32)"), "wrong", "fixed"),
    ("resnext", _r(conv="nn.Conv2d(128, 128, 3, padding=1, groups=groups)"), "wrong", "fixed"),
    ("resnext", _R_START.replace("self.conv = ...", "self.conv = nn.Conv2d(width, width, 3, padding=1, groups=groups)"), "wrong", "dots-left"),
    ("resnext", _r(conv="nn.Conv2d(width, width, 3, groups=groups)"), "error", "padding"),
    ("resnext", _r(conv="nn.Conv2d(channels, width, 3, padding=1, groups=groups)"), "error", "channels"),
    ("resnext", _r(conv="nn.Conv2d(width, width, 3, padding=1, groups=width // 3)"), "error", "divisible"),
    ("resnext", _R.replace("        return x + y\n", "        x + y\n"), "error", "no-return"),
    # every filter on one channel of its own: other numbers of parameters
    ("resnext", _r(conv="nn.Conv2d(width, width, 3, padding=1, groups=width)"), "wrong", None),
    # the right shape and parameters, but other numbers that no known mistake explains
    ("resnext", _R.replace("y = torch.relu(self.reduce(x))", "y = torch.sigmoid(self.reduce(x))"), "almost", None),
] + [("resnext", code, "wrong", key) for key, code in _R_WRONG.items()]


# --- the demo: the learner's ResNeXtBlock in the middle of WideCNN, on CIFAR-10 -------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def _from_block(e: BaseException) -> bool:
    """The error came from the learner's ResNeXtBlock: a line of their resnext.py is in its traceback."""
    mine = (projects.folder() / "cnn" / "resnext.py").resolve()
    tb = e.__traceback__
    while tb is not None:
        if Path(tb.tb_frame.f_code.co_filename).resolve() == mine:
            return True
        tb = tb.tb_next
    return False


def _demo(variant: str, show=None):
    """For the demo: None when resnext.py is not in the project yet; "data" when CIFAR-10 is not unpacked; "resnext"
    when the learner's ResNeXtBlock stops with an error, also during the training, or does not give back 64 maps of 8
    by 8; "cnn" when channel_stats does not give what it should; "trainer" when the Trainer or count_params stops with
    an error; "slow" when the training takes too long; else the dict of run(), with the modules the course's versions
    stood in for."""
    if not _saved("resnext"):
        return None
    if not ready():
        return "data"
    loaded = run_code("from cnn.resnext import ResNeXtBlock")
    if loaded.error is not None:
        return "resnext"
    ResNeXtBlock = loaded.names["ResNeXtBlock"]
    try:
        def tried():   # as in training: a mini-batch, in training mode, with a backward pass
            torch.manual_seed(0)
            block = ResNeXtBlock(CHANNELS, *width_groups(variant))
            out = block(torch.rand(4, CHANNELS, SIDE, SIDE))
            out.sum().backward()
            return tuple(out.shape)
        if with_time_limit(tried, 10) != (4, CHANNELS, SIDE, SIDE):
            return "resnext"
    except Exception:
        return "resnext"
    loaded = run_code("from cnn.channels import channel_stats")
    if loaded.error is not None:
        return "cnn"
    channel_stats = loaded.names["channel_stats"]
    loaded = run_code("from nn.trainer import Trainer\nfrom cnn.count import count_params")
    if loaded.error is not None:
        return "trainer"
    Trainer, count_params = loaded.names["Trainer"], loaded.names["count_params"]
    try:
        int(with_time_limit(lambda: count_params(WideCNN(Plain(CHANNELS))), 10))
    except Exception:
        return "trainer"
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    X, _, _, _ = photos()
    try:
        mean_, std = with_time_limit(lambda: channel_stats(X.clone()), 60)
        mean_ = torch.as_tensor(mean_, dtype=torch.float32).reshape(3)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(3)
        if not (torch.isfinite(mean_).all() and torch.isfinite(std).all() and (std > 0).all()):
            return "cnn"
    except Exception:
        return "cnn"
    try:
        result = with_time_limit(lambda: run(variant, 0, ResNeXtBlock, channel_stats, Trainer, count_params, show), 600)
    except Timeout:
        return "slow"
    except Exception as e:
        return "resnext" if _from_block(e) else "trainer"
    return result | {"stand_ins": stand_ins}


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file), and whether CIFAR-10
    is there."""
    texts = []
    for package, module in (("cnn", "resnext"), ("cnn", "channels"), ("cnn", "count"), ("nn", "train_epoch"), ("nn", "trainer")):
        path = projects.folder() / package / f"{module}.py"
        texts.append(path.read_text(encoding="utf-8") if path.exists() else None)
    return tuple(texts) + (ready(),)


def demo(variant: str = DEFAULT, show=None):
    """The demo's result for `variant`, worked out once for the project files as they are; `show` hears the run when it
    trains."""
    key = (variant,) + _project_texts()
    if key not in _DEMO_RUNS:
        _DEMO_RUNS.clear()
        _DEMO_RUNS[key] = _demo(variant, show)
    return _DEMO_RUNS[key]
