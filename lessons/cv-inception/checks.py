# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-inception: an Inception block puts branches of 1×1, 3×3 and 5×5 convolutions and of max
pooling side by side on the same maps, the 3×3 and 5×5 behind a 1×1 convolution to fewer channels and the pooling
before one, and stacks what they give along the channels with torch.cat. The learner's Inception is saved in their
project (work/projects/cnn/inception.py) when right, and checked on made-up numbers, so it works without CIFAR-10.
The demo trains the CNN of cv-1x1-convolution with the learner's Inception in its middle, with their Trainer on the
first 6,000 training photos of CIFAR-10, tests it on the first 1,000 test photos, and then switches off one branch at
a time, by setting its channels to 0, and counts again; after every epoch it keeps the network's guess for one checked
test photo of each kind, for rewinding."""
import html
import importlib.util
import re
from pathlib import Path

import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import chart, draw, live, paths, projects
from everything_ai.check import Mistake, ScenarioBox, Sort, Timeout, run_code, with_time_limit


def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_inception",
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
curves_picture = _ONE.curves_picture   # the demo's loss after every epoch, with the course's 3×3 run in grey
SHOWN, N_TRAIN, N_TEST = _ONE.SHOWN, _ONE.N_TRAIN, _ONE.N_TEST
EPOCHS, LR, CHANNELS, SIDE = _ONE.EPOCHS, _ONE.LR, _ONE.CHANNELS, _ONE.SIDE
conv_params, conv_mults = _ONE.conv_params, _ONE.conv_mults
WideCNN, Plain, AROUND, PLAIN_BLOCK = _ONE.WideCNN, _ONE.Plain, _ONE.AROUND, _ONE.PLAIN_BLOCK


# --- counting -------------------------------------------------------------------------------------------------------

def inception_params(c_in: int, c1: int, c3_reduce: int, c3: int, c5_reduce: int, c5: int, pool_proj: int) -> int:
    """The parameters of an Inception block, as nn.Conv2d makes its six convolutions, each with a bias per filter."""
    return (conv_params(c_in, c1) + conv_params(c_in, c3_reduce) + conv_params(c3_reduce, c3, 3)
            + conv_params(c_in, c5_reduce) + conv_params(c5_reduce, c5, 5) + conv_params(c_in, pool_proj))


def inception_mults(c_in: int, c1: int, c3_reduce: int, c3: int, c5_reduce: int, c5: int, pool_proj: int) -> int:
    """The multiplications of an Inception block for one pixel of what it gives."""
    return (conv_mults(c_in, c1) + conv_mults(c_in, c3_reduce) + conv_mults(c3_reduce, c3, 3)
            + conv_mults(c_in, c5_reduce) + conv_mults(c5_reduce, c5, 5) + conv_mults(c_in, pool_proj))


def naive_params(c_in: int, c1: int, c3: int, c5: int) -> int:
    """The naive block of the paper, without the 1×1 convolutions before the 3×3 and 5×5 filters and after the
    pooling: 1×1, 3×3 and 5×5 straight on the c_in channels, and pooling that keeps all c_in channels."""
    return conv_params(c_in, c1) + conv_params(c_in, c3, 3) + conv_params(c_in, c5, 5)


def naive_mults(c_in: int, c1: int, c3: int, c5: int) -> int:
    return conv_mults(c_in, c1) + conv_mults(c_in, c3, 3) + conv_mults(c_in, c5, 5)


# Inception (3a) of GoogLeNet (Szegedy et al., 2014, table 1): 192 channels of 28 by 28 in; 64 filters of 1 by 1, 96
# before 128 of 3 by 3, 16 before 32 of 5 by 5, and 32 after the pooling. Counted with biases, as nn.Conv2d makes them.
# Its table gives 128M operations: the 163,328 multiplications for each of the 28 × 28 pixels, 128,049,152. Its 159K
# parameters are the 163,328 weights without the biases, in thousands of 1,024.
G3A = (192, 64, 96, 128, 16, 32, 32)
G3A_SIDE = 28
G3A_NAIVE = (192, 64, 128, 32)
G3A_OUT = 64 + 128 + 32 + 32                 # 256 channels
G3A_NAIVE_OUT = 64 + 128 + 32 + 192          # 416: the naive block's pooling keeps all 192 channels
G3A_PARAMS = inception_params(*G3A)          # 163,696
G3A_WEIGHTS = G3A_PARAMS - (64 + 96 + 128 + 16 + 32 + 32)   # 163,328: without the biases, the table's 159K
G3A_NAIVE_PARAMS = naive_params(*G3A_NAIVE)  # 387,296
G3A_MULTS = inception_mults(*G3A) * G3A_SIDE ** 2              # 128,049,152 for one photo
G3A_NAIVE_MULTS = naive_mults(*G3A_NAIVE) * G3A_SIDE ** 2      # 303,464,448
# The six convolutions of inception (3a) in the order of the worked example, with their parameters.
G3A_PARTS = (conv_params(192, 64), conv_params(192, 96), conv_params(96, 128, 3), conv_params(192, 16),
             conv_params(16, 32, 5), conv_params(192, 32))

# The demo's block on the 64 maps of 8 by 8 in the middle of WideCNN: 16 filters of 1 by 1, 32 before 32 of 3 by 3,
# 8 before 8 of 5 by 5, and 8 after the pooling: 64 channels, the outputs of inception (3a) divided by 4.
DEMO_BLOCK = (CHANNELS, 16, 32, 32, 8, 8, 8)
BLOCK_PARAMS = inception_params(*DEMO_BLOCK)    # 15,016
NETWORK_PARAMS = AROUND + BLOCK_PARAMS         # 44,658
PLAIN_PARAMS = AROUND + PLAIN_BLOCK            # 66,570
# The branches in the order of torch.cat, with the channels each one gives in the demo's block.
BRANCHES = ("b1", "b3", "b5", "pool")
WIDTHS = {"b1": DEMO_BLOCK[1], "b3": DEMO_BLOCK[3], "b5": DEMO_BLOCK[5], "pool": DEMO_BLOCK[6]}


def branch_channels(branch: str) -> tuple[int, int]:
    """Where the channels of `branch` are in what the demo's block gives: from the first to just before the second."""
    start = 0
    for name in BRANCHES:
        if name == branch:
            return start, start + WIDTHS[name]
        start += WIDTHS[name]
    raise KeyError(branch)


# --- the picture of the block ---------------------------------------------------------------------------------------

def block_picture(block: tuple, side: int, labels: dict):
    """An Inception block from top to bottom: the maps that come in, the four branches side by side, each convolution
    a box with its kernel size and its filters, in the colour of what a model learns, the pooling in grey, and the
    maps that come out, side by side along the channels. `labels` gives "title", "input" and "output" (with {c} for
    the channels and {s} for the side), "filters" (with {n}), "conv1" for a 1×1 convolution and "conv" (with {k}) for
    a larger one, "pool" and "alt"."""
    c_in, c1, c3r, c3, c5r, c5, pp = block
    W, BW, BH, GAP = 590, 132, 46, 20
    xs = [2 + i * (BW + GAP) for i in range(4)]
    top = 32
    rows = [top, top + 82, top + 82 + BH + 26]   # the input, the first box of each branch, the second
    out_y = rows[2] + BH + 34
    ui = "font-size:13px;fill:var(--ea-ink);text-anchor:middle"
    small = "font-size:12.5px;fill:var(--ea-ink-2);text-anchor:middle"
    line = "stroke:var(--ea-ink-3);stroke-width:1.5;fill:none"
    parts = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']

    def box(x, y, w, first, second, role):
        fill = f"fill:var(--ea-{role}-wash);stroke:var(--ea-{role});stroke-width:1.5" if role else "fill:var(--ea-inset);stroke:var(--ea-line);stroke-width:1.2"
        tone = f"fill:var(--ea-{role}-text)" if role == "param" else "fill:var(--ea-ink)"
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{BH}" rx="7" style="{fill}"/>'
                     f'<text x="{x + w / 2}" y="{y + 19}" style="font-size:13.5px;font-weight:600;{tone};text-anchor:middle">{html.escape(first)}</text>'
                     f'<text x="{x + w / 2}" y="{y + 36}" style="{small}">{html.escape(second)}</text>')

    def arrow(x1, y1, x2, y2):
        parts.append(f'<path d="M{x1} {y1} L{x2} {y2 - 3}" style="{line}"/>'
                     f'<path d="M{x2 - 4} {y2 - 9} L{x2} {y2 - 3} L{x2 + 4} {y2 - 9}" style="{line};stroke-linecap:round;stroke-linejoin:round"/>')

    whole = xs[3] + BW - xs[0]
    box(xs[0], rows[0], whole, labels["input"].format(c=c_in, s=side), "", "hidden")
    conv = lambda k, n: (labels["conv1"] if k == 1 else labels["conv"].format(k=k), labels["filters"].format(n=n))
    firsts = [conv(1, c1), conv(1, c3r), conv(1, c5r), (labels["pool"], "")]
    seconds = [None, conv(3, c3), conv(5, c5), conv(1, pp)]
    for i, x in enumerate(xs):
        mid = x + BW / 2
        arrow(mid, rows[0] + BH, mid, rows[1])
        first = firsts[i]
        box(x, rows[1], BW, first[0], first[1], "param" if i < 3 else None)
        if seconds[i] is None:   # the 1×1 branch has one convolution: its line runs on past the second row
            parts.append(f'<path d="M{mid} {rows[1] + BH} L{mid} {rows[2] + BH}" style="{line}"/>')
        else:
            arrow(mid, rows[1] + BH, mid, rows[2])
            box(x, rows[2], BW, seconds[i][0], seconds[i][1], "param")
        arrow(mid, rows[2] + BH, mid, out_y)
    total = c1 + c3 + c5 + pp
    box(xs[0], out_y, whole, labels["output"].format(c=total, s=side), f"{c1} + {c3} + {c5} + {pp} = {total}", "hidden")
    return draw._svg(W, int(out_y + BH + 4), labels["alt"], "".join(parts), natural=True)


# --- the slider: how large the filter of a branch is ----------------------------------------------------------------

KERNELS = [1, 3, 5, 7, 9, 11]
WIDE, NARROW, OUT = 192, 16, 32   # a branch on 192 channels that gives 32: straight, or behind a 1×1 convolution to 16


def branch_params(k: int) -> tuple[int, int, int]:
    """A branch with filters of k by k from WIDE channels to OUT: straight on the WIDE channels, and behind a 1×1
    convolution to NARROW, as its two convolutions."""
    return conv_params(WIDE, OUT, k), conv_params(WIDE, NARROW), conv_params(NARROW, OUT, k)


def ratio(k: int) -> float:
    straight, reduce, conv = branch_params(k)
    return straight / (reduce + conv)


def first_tenfold() -> int:
    """The smallest kernel size for which the straight branch has more than 10 times the parameters."""
    return next(k for k in KERNELS if ratio(k) > 10)


def kernel_sentence(k: int, lang: str) -> str:
    """The sentence under the slider, for filters of k by k."""
    straight, reduce, conv = branch_params(k)
    s, r, c, t = (grouped(n, lang) for n in (straight, reduce, conv, reduce + conv))
    x = number(round(ratio(k), 1), 1, lang)
    if lang == "nl":
        name = "1×1 convolution" if k == 1 else f"{k}×{k}-convolutie"
        return (f"Met filters van {k} bij {k}: {s} parameters rechtstreeks op de {WIDE} channels, tegen {r} in de 1×1 convolution naar {NARROW} "
                f"en {c} in de {name} erachter, samen {t}: {x} keer zo weinig.")
    return (f"With filters of {k} by {k}: {s} parameters straight on the {WIDE} channels, against {r} in the 1×1 convolution to {NARROW} "
            f"and {c} in the {k}×{k} convolution behind it, {t} together: {x} times fewer.")


RIGHT = 900   # the right end of the slider's picture, in thousands: room for the number after the longest bar, 743,456


def kernel_picture(k: int, labels: dict, lang: str) -> str:
    """Two bars of parameters, in thousands, on an axis that stays the same for every k, each with its name above it:
    the branch straight on the WIDE channels in grey, and behind a 1×1 convolution to NARROW, its two convolutions one
    after the other, in the colour of what a model learns. `labels` gives "title", "x", "straight", "reduced" and
    "alt"."""
    straight, reduce, conv = branch_params(k)
    p = chart.Panel((0, RIGHT), (-0.55, 2.0), title=labels["title"], xlabel=labels["x"], yticks=[],
                    xticks=[0, 200, 400, 600, 800])
    p.text(0, 1.62, labels["straight"], "ink-2", "start")
    p.rect(0, 0.95, straight / 1000, 1.35, "ink-3")
    p.text(straight / 1000 + 10, 1.12, grouped(straight, lang), "ink-2", "start")
    p.text(0, 0.42, labels["reduced"], "ink", "start")
    p.rect(0, -0.25, reduce / 1000, 0.15, "param")
    p.rect(reduce / 1000, -0.25, (reduce + conv) / 1000, 0.15, "param")
    p.text((reduce + conv) / 1000 + 10, -0.08, grouped(reduce + conv, lang), "ink", "start")
    svg = chart.figure([p], alt=labels["alt"], width=440, height=190)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


# --- what the course measured ---------------------------------------------------------------------------------------

# What the course measured, on one thread, with the course's code, through run(): Inception(*DEMO_BLOCK) made after
# torch.manual_seed(seed), then the rest of WideCNN, trained as in cv-1x1-convolution (the Trainer of nn-trainer with
# AdamW, a learning rate of LR, a warmup of one epoch, cosine decay, max_norm 1.0, mini-batches of 64) on the first
# N_TRAIN training photos, shuffled after torch.manual_seed(seed + 1), for EPOCHS epochs and the seeds 0 to 4, with
# the photos normalized by channel_stats of the training photos. "train" and "test" are how many of the training
# photos and of the first N_TEST test photos are read right after the last epoch; "off" how many test photos are read
# right with the channels of one branch set to 0 (branch_channels); "best" is the first epoch with the lowest
# validation loss, rounded to 3 decimals, and "lowest" that loss; "last_val" and "last_train" are the validation and
# the training loss of the last epoch. CURVES are seed 0. The runs with one 3×3 convolution in the middle are those
# of cv-1x1-convolution, which builds them the same way. tests/test_cv_inception.py measures seed 0 again when
# EAI_CIFAR_FILE names the unpacked cifar-10.npz.
REFERENCE = {
    "plain": _ONE.REFERENCE["plain"],
    "inception": {"train": [5989, 5896, 5931, 5972, 5997], "test": [584, 559, 553, 555, 585],
                  "off": {"b1": [484, 455, 409, 418, 443], "b3": [477, 472, 453, 504, 497],
                          "b5": [560, 561, 553, 521, 567], "pool": [530, 491, 508, 468, 541]},
                  "best": [7, 8, 10, 6, 4], "lowest": [1.29, 1.32, 1.361, 1.383, 1.311],
                  "last_val": [2.46, 2.088, 2.21, 2.295, 2.324], "last_train": [0.033, 0.114, 0.092, 0.056, 0.023]},
}
CURVES = {
    "inception_val": [1.624, 1.52, 1.427, 1.381, 1.292, 1.358, 1.29, 1.412, 1.396, 1.445, 1.581, 1.63, 1.835, 1.891,
                      2.075, 2.275, 2.383, 2.436, 2.457, 2.46],
    "inception_train": [1.934, 1.58, 1.432, 1.315, 1.213, 1.097, 1.031, 0.922, 0.831, 0.708, 0.596, 0.479, 0.368, 0.27,
                        0.181, 0.114, 0.071, 0.047, 0.037, 0.033],
}


def mean(values) -> float:
    return sum(values) / len(values)


def drops(branch: str) -> list[int]:
    """How many fewer test photos each of the course's runs reads right with `branch` switched off."""
    run = REFERENCE["inception"]
    return [all_on - off for all_on, off in zip(run["test"], run["off"][branch])]


# --- the picture of the runs, with each branch switched off ---------------------------------------------------------

ROWS = ("plain", "inception", "b1", "b3", "b5", "pool")


def off_picture(labels: dict, lang: str, mine: dict | None = None) -> str:
    """How many of the N_TEST test photos the network reads right after EPOCHS epochs: with one 3×3 convolution in the
    middle, with the Inception block, and with the block with one branch switched off, an open dot for every seed of
    the course and a line at their mean. With `mine` ({"test", "off"}), the learner's run as a filled dot in each row
    of the Inception block. `labels` gives "title", "x", a name for each row, "mine", "course" and "alt"."""
    run = REFERENCE["inception"]
    values = {"plain": REFERENCE["plain"]["test"], "inception": run["test"], **run["off"]}
    yours = {} if mine is None else {"inception": mine["test"], **mine["off"]}
    # from 380 to 600, wider where the learner's run lies outside, so that every dot is where its number is
    low = min([380] + [int(v - 15) // 50 * 50 for v in yours.values()])
    high = max([600] + [-(-int(v + 15) // 50) * 50 for v in yours.values()])
    p = chart.Panel((low, high), (-1.0, len(ROWS) - 0.3), title=labels["title"], xlabel=labels["x"], yticks=[],
                    xticks=list(range(-(-(low + 1) // 50) * 50, high + 1, 50)))
    step = (high - low) / 220   # the space of one photo on the axis of 380 to 600
    for k, key in enumerate(ROWS):
        y = len(ROWS) - 1 - k
        m = mean(values[key])
        p.line([m, m], [y - 0.22, y + 0.22], "ink-3", 1.6)
        p.points(values[key], [y] * len(values[key]), "ink", r=3.5, hollow=True)
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


# The lesson is done when these are correct. "branches" is the choice question in the lesson.
REQUIRED = ["same-size", "inception", "branches"]

# --- exercise 1: what keeps 28 by 28 --------------------------------------------------------------------------------

_SORT = Sort(
    name="same-size",
    items={
        "one": {"en": "**1×1 convolution, no padding**", "nl": "**1×1 convolution, zonder padding**"},
        "three": {"en": "**3×3 convolution, padding 1**", "nl": "**3×3-convolutie, padding 1**"},
        "five_two": {"en": "**5×5 convolution, padding 2**", "nl": "**5×5-convolutie, padding 2**"},
        "five_one": {"en": "**5×5 convolution, padding 1**", "nl": "**5×5-convolutie, padding 1**"},
        "pool_one": {"en": "**3×3 max pooling, stride 1, padding 1**", "nl": "**3×3 max pooling, stride 1, padding 1**"},
        "pool_two": {"en": "**2×2 max pooling.** As after the convolutions of your ColorCNN.",
                     "nl": "**2×2 max pooling.** Zoals na de convoluties van je ColorCNN."},
    },
    groups={"keeps": {"en": "keeps 28 by 28: it fits in the block", "nl": "houdt 28 bij 28: past in het block"},
            "changes": {"en": "changes the size: torch.cat stops", "nl": "verandert de grootte: torch.cat stopt"}},
    solution={"one": "keeps", "three": "keeps", "five_two": "keeps", "five_one": "changes", "pool_one": "keeps", "pool_two": "changes"},
    nudge={
        "one": {"en": "Where does a filter of 1 by 1 fit: on every pixel, or only away from the edge?",
                "nl": "Waar past een filter van 1 bij 1: op elke pixel, of alleen weg van de rand?"},
        "three": {"en": "Without padding, a filter of 3 by 3 fits 26 times across 28 pixels. How many more places does a border of 1 give it?",
                  "nl": "Zonder padding past een filter van 3 bij 3 op 28 pixels 26 keer. Hoeveel plekken meer geeft een rand van 1?"},
        "five_two": {"en": "Without padding, a filter of 5 by 5 fits 24 times across 28 pixels. How many more places does a border of 2 give it?",
                     "nl": "Zonder padding past een filter van 5 bij 5 op 28 pixels 24 keer. Hoeveel plekken meer geeft een rand van 2?"},
        "five_one": {"en": "Work it out: 28 + 2 × 1 − 5 + 1. Is that 28?", "nl": "Reken het uit: 28 + 2 × 1 − 5 + 1. Is dat 28?"},
        "pool_one": {"en": "Pooling moves like a filter: with stride 1, a window of 3 by 3 and a border of 1, how many places across 28?",
                     "nl": "Pooling schuift zoals een filter: met stride 1, een venster van 3 bij 3 en een rand van 1, hoeveel plekken op 28?"},
        "pool_two": {"en": "What did the 2×2 max pooling do to your maps of 32 by 32 in your ColorCNN?",
                     "nl": "Wat deed de 2×2 max pooling met je maps van 32 bij 32 in je ColorCNN?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the six steps again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de zes stappen opnieuw uit."},
        "empty": {"en": "Put each step in a group: it keeps 28 by 28, or it changes the size.",
                  "nl": "Zet elke stap in een groep: hij houdt 28 bij 28, of hij verandert de grootte."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "{item} belongs in the other group.", "nl": "{item} hoort in de andere groep."},
        "correct": {"en": "All {n} are right. With stride 1, a window of k by k fits n + 2p − k + 1 times across n pixels with padding p: a padding of (k − 1) / 2 keeps the size. So 0 for 1 by 1, 1 for 3 by 3 and 2 for 5 by 5, and 3×3 max pooling with stride 1 and padding 1 keeps it too. A 5×5 convolution with padding 1 gives 26 by 26, and 2×2 max pooling, with its stride of 2, 14 by 14: `torch.cat` then stops.",
                    "nl": "Alle {n} kloppen. Met stride 1 past een venster van k bij k op n pixels met padding p precies n + 2p − k + 1 keer: een padding van (k − 1) / 2 houdt de grootte. Dus 0 voor 1 bij 1, 1 voor 3 bij 3 en 2 voor 5 bij 5, en 3×3 max pooling met stride 1 en padding 1 houdt hem ook. Een 5×5-convolutie met padding 1 geeft 26 bij 26, en 2×2 max pooling, met zijn stride van 2, 14 bij 14: `torch.cat` stopt dan."},
    },
)

# --- exercise 2: an Inception block ---------------------------------------------------------------------------------

_I_HEAD = ("import torch\nfrom torch import nn\n\n\nclass Inception(nn.Module):\n"
           "    def __init__(self, c_in, c1, c3_reduce, c3, c5_reduce, c5, pool_proj):\n        super().__init__()\n"
           "        self.b1 = nn.Conv2d(c_in, c1, 1)\n        self.b3_reduce = nn.Conv2d(c_in, c3_reduce, 1)\n"
           "        self.b3 = nn.Conv2d(c3_reduce, c3, 3, padding=1)\n")
_I_POOL = "        self.pool = nn.MaxPool2d(3, stride=1, padding=1)\n        self.pool_proj = nn.Conv2d(c_in, pool_proj, 1)\n"
_I_FORWARD = ("\n    def forward(self, x):\n        y1 = torch.relu(self.b1(x))\n"
              "        y3 = torch.relu(self.b3(torch.relu(self.b3_reduce(x))))\n")
_I_YP = "        yp = torch.relu(self.pool_proj(self.pool(x)))\n"
_I_Y5 = "torch.relu(self.b5(torch.relu(self.b5_reduce(x))))"
_I_CAT = "torch.cat([y1, y3, y5, yp], dim=1)"


def _i(reduce: str = "nn.Conv2d(c_in, c5_reduce, 1)", conv: str = "nn.Conv2d(c5_reduce, c5, 5, padding=2)",
       y5: str = _I_Y5, back: str = _I_CAT) -> str:
    return (_I_HEAD + f"        self.b5_reduce = {reduce}\n        self.b5 = {conv}\n" + _I_POOL + _I_FORWARD
            + f"        y5 = {y5}\n" + _I_YP + f"        return {back}\n")


_I = _i()
_I_START = (_I_HEAD + "        # b5_reduce: a 1×1 convolution from c_in to c5_reduce\n        self.b5_reduce = ...\n"
            "        # b5: a 5×5 convolution from c5_reduce to c5, same height and width\n        self.b5 = ...\n"
            + _I_POOL + _I_FORWARD + "        # y5: b5_reduce and b5, each with a ReLU after it\n        y5 = ...\n" + _I_YP
            + "        # side by side along the channels, in this order: y1, y3, y5, yp\n        return ...\n")
# The scenarios give the six convolutions their weights by name, in this order, so that the order in which __init__
# makes them does not change the numbers.
_I_WEIGHTS = ("with torch.no_grad():\n    for layer in (block.b1, block.b3_reduce, block.b3, block.b5_reduce, block.b5, block.pool_proj):\n"
              "        for q in layer.parameters():\n            q.copy_(torch.randn_like(q) / 4)\n")
_I_SCENARIOS = [
    # inception (3a) of GoogLeNet on one made-up photo's maps: the shape, and the number of parameters
    "import torch\n\nblock = Inception(192, 64, 96, 128, 16, 32, 32)\nout = block(torch.rand(1, 192, 28, 28))\n"
    "[list(out.shape), sum(q.numel() for q in block.parameters())]",
    # a small block: the numbers of every channel at a pixel in the middle and at a corner
    "import torch\n\nblock = Inception(6, 2, 3, 2, 2, 3, 1)\ntorch.manual_seed(0)\n" + _I_WEIGHTS
    + "out = block(torch.randn(1, 6, 7, 7))\n[out[0, :, 3, 3].tolist(), out[0, :, 0, 0].tolist()]",
    # another size: the shapes of the weights of b5_reduce and b5, and whether no number below 0 comes out
    "import torch\n\ntorch.manual_seed(1)\nblock = Inception(8, 2, 4, 3, 2, 5, 1)\nout = block(torch.randn(2, 8, 6, 6))\n"
    "[list(block.b5_reduce.weight.shape), list(block.b5.weight.shape), list(out.shape), bool((out >= 0).all())]",
]
_I_WRONG = {
    # the branches in another order
    "order": _i(back="torch.cat([y1, y5, y3, yp], dim=1)"),
    "no-relu": _i(y5="self.b5(self.b5_reduce(x))"),
    "reduce-relu": _i(y5="torch.relu(self.b5(self.b5_reduce(x)))"),
    # a 3×3 convolution where the 5×5 one should be
    "kernel-3": _i(conv="nn.Conv2d(c5_reduce, c5, 3, padding=1)"),
}
_KERNEL_3 = G3A_PARAMS - conv_params(16, 32, 5) + conv_params(16, 32, 3)   # 155,504
_FIXED = Mistake("fixed", matches=lambda got, text: re.search(r"self\.(b5_reduce|b5)\s*=\s*nn\.Conv2d\(\s*(\d|[A-Za-z_]\w*\s*,\s*\d+\s*,"
                                                               r"|[^)\n]*\b(in_channels|out_channels)\s*=\s*\d)", text) is not None,
                 en="`b5_reduce` or `b5` has a fixed number of channels. The block must work for every size: use the names `c_in`, `c5_reduce` and `c5`, as `b3_reduce` and `b3` do with theirs.",
                 nl="`b5_reduce` of `b5` heeft een vast aantal channels. Het block moet voor elke grootte werken: gebruik de namen `c_in`, `c5_reduce` en `c5`, zoals `b3_reduce` en `b3` dat met de hunne doen.")
# `...` on a line of its own, or after `=` or `return`
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$|=\s*\.\.\.|\breturn\s+\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace each with your own code: `b5_reduce`, `b5`, `y5`, and what `forward` gives back.",
                nl="Er staat nog een `...` in je code. Vervang elk door je eigen code: `b5_reduce`, `b5`, `y5`, en wat `forward` teruggeeft.")
_TEXTS = _ONE._TEXTS
_S_EN, _S_NL = _ONE._S_EN, _ONE._S_NL
_close = _ONE._close


def _below_zero(got) -> bool:
    """Numbers below 0 in what a scenario gives: the numbers of the second, or the last answer of the third, False.
    A ReLU never gives those."""
    if isinstance(got, list) and len(got) == 4 and got[-1] is False:
        return True
    if isinstance(got, (list, tuple)):
        return any(_below_zero(v) for v in got)
    return isinstance(got, float) and got < 0


def _other_order(text: str) -> bool:
    """torch.cat gets y1, y3, y5 and yp, but in another order."""
    found = re.search(r"torch\.cat\(\s*[\[(]([^\])]*)[\])]", text)
    names = re.findall(r"\w+", found.group(1)) if found else []
    return sorted(names) == ["y1", "y3", "y5", "yp"] and names != ["y1", "y3", "y5", "yp"]


def _cat_dim(e) -> bool:
    """torch.cat along another axis than the channels: its error names the axis it puts together."""
    return isinstance(e, RuntimeError) and re.search(r"Sizes of tensors must match except in dimension (0|2|3|-)", str(e)) is not None


EXERCISES = {
    "same-size": _SORT,
    "inception": ScenarioBox(
        name="inception",
        start=_I_START,
        solution=_I,
        scenarios=_I_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "inception"),
        first=[_DOTS, _FIXED],
        mistakes=[
            Mistake("sizes", error=lambda e: isinstance(e, RuntimeError) and "Sizes of tensors must match except in dimension 1" in str(e),
                    en="Your branches give maps of different heights and widths, and `torch.cat` stops: `Sizes of tensors must match except in dimension 1`. A 1×1 convolution needs no padding. A filter of 5 by 5 with stride 1 fits n + 2p − 4 times across n pixels with padding p: which padding keeps n?",
                    nl="Je takken geven maps met verschillende hoogtes en breedtes, en `torch.cat` stopt: `Sizes of tensors must match except in dimension 1`. Een 1×1 convolution heeft geen padding nodig. Een filter van 5 bij 5 met stride 1 past op n pixels met padding p precies n + 2p − 4 keer: welke padding houdt n?"),
            Mistake("cat-dim", error=_cat_dim,
                    en="`torch.cat` puts your branches together along another axis than the channels, and stops: their maps have different numbers of channels, and on every other axis they must match. A batch of maps has the shape (photos, channels, rows, columns): the channels are `dim=1`.",
                    nl="`torch.cat` zet je takken aan elkaar langs een andere axis dan de channels, en stopt: hun maps hebben verschillende aantallen channels, en op elke andere axis moeten ze passen. Een batch maps heeft de shape (foto's, channels, rijen, kolommen): de channels zijn `dim=1`."),
            Mistake("stack", error=lambda e: isinstance(e, RuntimeError) and "stack expects each tensor to be equal size" in str(e),
                    en="`torch.stack` makes a new axis, and needs tensors of exactly the same shape; your branches give different numbers of channels. `torch.cat` puts them together along an axis they already have: `dim=1`, the channels.",
                    nl="`torch.stack` maakt een nieuwe axis, en heeft tensors van precies dezelfde shape nodig; je takken geven verschillende aantallen channels. `torch.cat` zet ze aan elkaar langs een axis die ze al hebben: `dim=1`, de channels."),
            Mistake("added", error=lambda e: isinstance(e, RuntimeError) and "must match the size of tensor" in str(e),
                    en="Your block adds its branches up, and their channels do not match. An Inception block keeps what every branch found: it puts their maps side by side with `torch.cat`, so that the channels add up instead of the numbers.",
                    nl="Je block telt zijn takken op, en hun channels passen niet. Een Inception block houdt wat elke tak vond: het zet hun maps naast elkaar met `torch.cat`, zodat de channels optellen in plaats van de getallen."),
            Mistake("channels", error=lambda e: isinstance(e, RuntimeError) and "channels" in str(e) and "expected" in str(e),
                    en="A convolution gets another number of channels than it was made for. `nn.Conv2d(a, b, k)` takes `a` channels and gives `b`: `b5_reduce` goes from `c_in` to `c5_reduce`, and `b5` from `c5_reduce` to `c5`. Does `y5` send `x` through `b5_reduce` first, and then through `b5`?",
                    nl="Een convolutie krijgt een ander aantal channels dan waarvoor hij gemaakt is. `nn.Conv2d(a, b, k)` neemt `a` channels en geeft er `b`: `b5_reduce` gaat van `c_in` naar `c5_reduce`, en `b5` van `c5_reduce` naar `c5`. Stuurt `y5` `x` eerst door `b5_reduce`, en dan door `b5`?"),
            Mistake("no-return", error=lambda e: isinstance(e, AttributeError) and "'NoneType' object has no attribute" in str(e),
                    en="The check asks for the shape of what your block gives, and `forward` gives back nothing, `None`: end it with `return`, and the four branches side by side.",
                    nl="De check vraagt de shape van wat je block geeft, en `forward` geeft niets terug, `None`: eindig hem met `return`, en de vier takken naast elkaar."),
            Mistake("list", error=lambda e: isinstance(e, AttributeError) and re.search(r"'(list|tuple)' object has no attribute", str(e)) is not None,
                    en="`forward` gives back the four branches as separate tensors, not as one. `torch.cat` puts a list of tensors together into one: `torch.cat([y1, y3, y5, yp], dim=1)`.",
                    nl="`forward` geeft de vier takken terug als losse tensors, niet als één. `torch.cat` zet een lijst tensors aan elkaar tot één: `torch.cat([y1, y3, y5, yp], dim=1)`."),
            Mistake("order", matches=lambda got, text: _other_order(text),
                    en=_S_EN + ": your `torch.cat` puts the branches side by side in another order, so the channels come out in another order. Put them as y1, y3, y5, yp, the order of GoogLeNet's table: the demo switches each branch off by its place among the channels.",
                    nl=_S_NL + ": je `torch.cat` zet de takken in een andere volgorde naast elkaar, dus komen de channels in een andere volgorde. Zet ze als y1, y3, y5, yp, de volgorde van de tabel van GoogLeNet: de demo zet elke tak uit op zijn plek tussen de channels."),
            Mistake("no-relu", matches=lambda got, text: _below_zero(got),
                    en=_S_EN + ": numbers below 0 come out of your block, and a ReLU never gives those, so a branch has no ReLU at its end. Every convolution of the block has a ReLU after it: in `y5`, after `b5_reduce` and after `b5`.",
                    nl=_S_NL + ": er komen getallen onder 0 uit je block, en een ReLU geeft die nooit, dus heeft een tak geen ReLU aan zijn eind. Elke convolutie van het block heeft een ReLU erna: in `y5` na `b5_reduce` en na `b5`."),
            Mistake("reduce-relu", wrong=_I_WRONG["reduce-relu"],
                    en=_S_EN + ": `b5_reduce` has no ReLU after it, so `b5` gets numbers below 0 too. Without a ReLU between them, two convolutions in a row are no more than one: put `torch.relu` around `self.b5_reduce(x)` too, as in `y3`.",
                    nl=_S_NL + ": `b5_reduce` heeft geen ReLU erna, dus krijgt `b5` ook getallen onder 0. Zonder ReLU ertussen zijn twee convoluties achter elkaar niet meer dan één: zet `torch.relu` ook om `self.b5_reduce(x)`, zoals in `y3`."),
            Mistake("kernel-3", wrong=_I_WRONG["kernel-3"],
                    en=_S_EN + f": the shape is right, but inception (3a) has {grouped(_KERNEL_3, 'en')} parameters instead of {grouped(G3A_PARAMS, 'en')}, because `b5` has filters of 3 by 3. The third branch sees further: `nn.Conv2d(c5_reduce, c5, 5, padding=2)`.",
                    nl=_S_NL + f": de shape klopt, maar inception (3a) heeft {grouped(_KERNEL_3, 'nl')} parameters in plaats van {grouped(G3A_PARAMS, 'nl')}, omdat `b5` filters van 3 bij 3 heeft. De derde tak kijkt verder: `nn.Conv2d(c5_reduce, c5, 5, padding=2)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the four `...` with your code: `b5_reduce`, `b5`, `y5`, and what `forward` gives back.",
                      "nl": "Vervang de vier `...` door je code: `b5_reduce`, `b5`, `y5`, en wat `forward` teruggeeft."},
            "wrong": {"en": _S_EN + ". Is `b5_reduce` a 1×1 convolution from `c_in` to `c5_reduce`, and `b5` a 5×5 one from `c5_reduce` to `c5` with padding 2, each with a ReLU after it in `y5`, and does `forward` give back `y1`, `y3`, `y5` and `yp` side by side along the channels?",
                      "nl": _S_NL + ". Is `b5_reduce` een 1×1 convolution van `c_in` naar `c5_reduce`, en `b5` een 5×5-convolutie van `c5_reduce` naar `c5` met padding 2, elk met een ReLU erna in `y5`, en geeft `forward` `y1`, `y3`, `y5` en `yp` naast elkaar terug langs de channels?"},
            "almost": {"en": "The shape and the number of parameters are right, but " + _S_EN[0].lower() + _S_EN[1:] + ". Does `y5` put a ReLU after `b5_reduce` and after `b5`, and are the branches in the order y1, y3, y5, yp?",
                       "nl": "De shape en het aantal parameters kloppen, maar " + _S_NL[0].lower() + _S_NL[1:] + ". Zet `y5` een ReLU na `b5_reduce` en na `b5`, en staan de takken in de volgorde y1, y3, y5, yp?"},
            "correct": {"en": f"Your `Inception` is right: inception (3a) gives 256 channels of 28 by 28, with {grouped(G3A_PARAMS, 'en')} parameters. It is now in your project, as `{{saved}}`.",
                        "nl": f"Je `Inception` klopt: inception (3a) geeft 256 channels van 28 bij 28, met {grouped(G3A_PARAMS, 'nl')} parameters. Hij staat nu in je project, als `{{saved}}`."},
            "kept": {"en": "Your `Inception` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `Inception` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_I_ORDER = _I.replace("        self.b1 = nn.Conv2d(c_in, c1, 1)\n", "").replace(
    _I_POOL, _I_POOL + "        self.b1 = nn.Conv2d(c_in, c1, 1)\n")
CASES = [
    ("same-size", {}, "empty", None),
    ("same-size", _SORT.solution, "correct", None),
    ("same-size", {"one": "keeps", "three": "keeps"}, "wrong", "unfinished"),
    ("same-size", {**_SORT.solution, "five_one": "keeps"}, "wrong", "wrong:five_one"),
    ("same-size", {**_SORT.solution, "pool_one": "changes"}, "wrong", "wrong:pool_one"),
    ("inception", _I_START, "empty", None),
    ("inception", _I, "correct", None),
    # the layers made in another order: the scenarios give them their weights by name
    ("inception", _I_ORDER, "correct", None),
    # other ways to write the same: padding "same", kernel_size by name, a tuple and dim by place, nn.functional
    ("inception", _i(conv="nn.Conv2d(c5_reduce, c5, 5, padding=\"same\")"), "correct", None),
    ("inception", _i(conv="nn.Conv2d(c5_reduce, c5, kernel_size=5, padding=2)"), "correct", None),
    ("inception", _i(back="torch.cat((y1, y3, y5, yp), 1)"), "correct", None),
    ("inception", _i(y5="nn.functional.relu(self.b5(nn.functional.relu(self.b5_reduce(x))))"), "correct", None),
    ("inception", _i(reduce="nn.Conv2d(192, 16, 1)"), "wrong", "fixed"),
    ("inception", _i(conv="nn.Conv2d(c5_reduce, 32, 5, padding=2)"), "wrong", "fixed"),
    ("inception", _I_START.replace("self.b5_reduce = ...", "self.b5_reduce = nn.Conv2d(c_in, c5_reduce, 1)"), "wrong", "dots-left"),
    ("inception", _i(conv="nn.Conv2d(c5_reduce, c5, 5, padding=1)"), "error", "sizes"),
    ("inception", _i(conv="nn.Conv2d(c5_reduce, c5, 5)"), "error", "sizes"),
    ("inception", _i(back="torch.cat([y1, y3, y5, yp])"), "error", "cat-dim"),
    ("inception", _i(back="torch.cat([y1, y3, y5, yp], dim=-1)"), "error", "cat-dim"),
    ("inception", _i(back="torch.stack([y1, y3, y5, yp], dim=1)"), "error", "stack"),
    ("inception", _i(back="y1 + y3 + y5 + yp"), "error", "added"),
    # the 5×5 convolution straight on x, without b5_reduce
    ("inception", _i(y5="torch.relu(self.b5(x))"), "error", "channels"),
    ("inception", _i(reduce="nn.Conv2d(c5_reduce, c_in, 1)"), "error", "channels"),
    ("inception", _i(back="[y1, y3, y5, yp]"), "error", "list"),
    ("inception", _I.replace("        return torch.cat([y1, y3, y5, yp], dim=1)\n", "        torch.cat([y1, y3, y5, yp], dim=1)\n"), "error", "no-return"),
    # b5_reduce to c5 instead of c5_reduce: inception (3a) then has other numbers
    ("inception", _i(reduce="nn.Conv2d(c_in, c5, 1)", conv="nn.Conv2d(c5, c5, 5, padding=2)"), "wrong", None),
    ("inception", _i(reduce="nn.Conv2d(in_channels=c_in, out_channels=16, kernel_size=1)"), "wrong", "fixed"),
    ("inception", _i(back="torch.cat([y3, y1, y5, yp], dim=1)"), "wrong", "order"),
    # a ReLU after b5_reduce, but none after b5
    ("inception", _i(y5="self.b5(torch.relu(self.b5_reduce(x)))"), "wrong", "no-relu"),
    # the right shape and parameters, but other numbers that no known mistake explains
    ("inception", _i(y5="torch.relu(self.b5(torch.sigmoid(self.b5_reduce(x))))"), "almost", None),
] + [("inception", code, "wrong", key) for key, code in _I_WRONG.items()]


# --- the demo: the learner's Inception in the middle of WideCNN, on CIFAR-10 ----------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def _accuracy(model, X, y) -> int:
    model.eval()
    with torch.no_grad():
        return int(sum((model(X[i:i + 500]).argmax(dim=1) == y[i:i + 500]).sum() for i in range(0, len(X), 500)))


def switched_off(model, block, X, y) -> dict:
    """How many photos the trained model reads right with the channels of each branch of its block set to 0."""
    off = {}
    for branch in BRANCHES:
        a, b = branch_channels(branch)

        def zero(module, inputs, out, a=a, b=b):
            out = out.clone()
            out[:, a:b] = 0
            return out

        hook = block.register_forward_hook(zero)
        try:
            off[branch] = _accuracy(model, X, y)
        finally:
            hook.remove()
    return off


def run(seed: int, Inception, channel_stats, Trainer, count_params, show=None):
    """One run of WideCNN with Inception(*DEMO_BLOCK) in its middle, as REFERENCE says: the block made after
    torch.manual_seed(seed), then the rest of the network, and the photos shuffled after torch.manual_seed(seed + 1).
    Gives back a dict with the parameters of the network and of its block, how many of the training and test photos it
    reads right, how many test photos with each branch switched off ("off"), the loss after every epoch ("curves"),
    the epoch with the lowest validation loss, and the network's guess for the photos SHOWN after every epoch
    ("guesses") with their right kinds ("truth"). Errors go to the caller, which says whose code made them. `show`
    hears the run while it trains."""
    X, y, X_test, y_test = photos()
    mean_, std = channel_stats(X.clone())
    mean_ = torch.as_tensor(mean_, dtype=torch.float32).reshape(3)
    std = torch.as_tensor(std, dtype=torch.float32).reshape(3)
    X = (X - mean_[:, None, None]) / std[:, None, None]
    X_test = (X_test - mean_[:, None, None]) / std[:, None, None]
    torch.manual_seed(seed)
    block = Inception(*DEMO_BLOCK)
    model = WideCNN(block)
    params, block_params = int(count_params(model)), int(count_params(block))
    shown, guesses = X_test[list(SHOWN)], []

    def heard(e):
        if "val_loss" in e:   # after every epoch: the guesses for the rewind, in eval mode, so the run stays the same
            was = model.training
            model.eval()
            with torch.no_grad():
                guesses.append([int(g) for g in model(shown).argmax(dim=1)])
            model.train(was)
        if show is not None:
            show(e)

    torch.manual_seed(seed + 1)
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    val = DataLoader(TensorDataset(X_test, y_test), batch_size=500)
    trainer = Trainer(model, loader, val, epochs=EPOCHS, lr=LR, warmup=len(loader), max_norm=1.0)
    live.follow(trainer, heard, "inception")
    history = trainer.fit() or []
    curves = {"train": [float(h["train_loss"]) for h in history if isinstance(h, dict) and "train_loss" in h],
              "val": [float(h["val_loss"]) for h in history if isinstance(h, dict) and "val_loss" in h]}
    return {"params": params, "block_params": block_params, "train": _accuracy(model, X, y),
            "test": _accuracy(model, X_test, y_test), "off": switched_off(model, block, X_test, y_test),
            "curves": curves, "best": best_epoch(curves["val"]), "guesses": guesses[:len(curves["val"])],
            "truth": [int(y_test[place]) for place in SHOWN]}


def _from_block(e: BaseException) -> bool:
    """The error came from the learner's Inception: a line of their inception.py is in its traceback."""
    mine = (projects.folder() / "cnn" / "inception.py").resolve()
    tb = e.__traceback__
    while tb is not None:
        if Path(tb.tb_frame.f_code.co_filename).resolve() == mine:
            return True
        tb = tb.tb_next
    return False


def _demo(show=None):
    """For the demo: None when inception.py is not in the project yet; "data" when CIFAR-10 is not unpacked;
    "inception" when the learner's Inception stops with an error, also during the training, or does not give back 64
    maps of 8 by 8; "cnn" when channel_stats does not give what it should; "trainer" when the Trainer or
    count_params stops with an error; "slow" when the training takes too long; else the dict of run(), with the
    modules the course's versions stood in for."""
    if not _saved("inception"):
        return None
    if not ready():
        return "data"
    loaded = run_code("from cnn.inception import Inception")
    if loaded.error is not None:
        return "inception"
    Inception = loaded.names["Inception"]
    try:
        def tried():   # as in training: a mini-batch, in training mode, with a backward pass
            torch.manual_seed(0)
            block = Inception(*DEMO_BLOCK)
            out = block(torch.rand(4, CHANNELS, SIDE, SIDE))
            out.sum().backward()
            return tuple(out.shape)
        if with_time_limit(tried, 10) != (4, CHANNELS, SIDE, SIDE):
            return "inception"
    except Exception:
        return "inception"
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
        result = with_time_limit(lambda: run(0, Inception, channel_stats, Trainer, count_params, show), 600)
    except Timeout:
        return "slow"
    except Exception as e:
        return "inception" if _from_block(e) else "trainer"
    return result | {"stand_ins": stand_ins}


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file), and whether CIFAR-10
    is there."""
    texts = []
    for package, module in (("cnn", "inception"), ("cnn", "channels"), ("cnn", "count"), ("nn", "train_epoch"), ("nn", "trainer")):
        path = projects.folder() / package / f"{module}.py"
        texts.append(path.read_text(encoding="utf-8") if path.exists() else None)
    return tuple(texts) + (ready(),)


def demo(show=None):
    """The demo's result, worked out once for the project files as they are; `show` hears the run when it trains."""
    key = _project_texts()
    if key not in _DEMO_RUNS:
        _DEMO_RUNS.clear()
        _DEMO_RUNS[key] = _demo(show)
    return _DEMO_RUNS[key]
