# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-augment-photos: crop, a photo shifted by cutting 32 by 32 out of it with a black border of
4 pixels, and jitter, a change of its brightness, contrast and saturation, each saved in the learner's project
(work/projects/cnn/crop.py and color.py) when right. Both are checked on made-up photos, so they work without
CIFAR-10. The demo trains the learner's ColorCNN of cv-color-cnn again on the first 6,000 training photos, now from a
Dataset that mirrors half of them and crops and changes the color of every one, anew every time, and tests it on the
first 1,000 test photos, as they are; after every epoch it keeps the network's guess for one checked test photo of each
kind, for rewinding through the run."""
import errno
import functools
import html
import re
import urllib.error

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch.utils.data import DataLoader, Dataset, TensorDataset

from everything_ai import chart, cifar, hub, live, projects
from everything_ai.lesson import download_card
from everything_ai.check import Mistake, ScenarioBox, Sort, Timeout, run_code, with_time_limit

# The demo uses the learner's ColorCNN and channel_stats of cv-color-cnn, their Trainer of nn-trainer and their
# count_params of cnn-pytorch; where one is missing or broken, the course's version stands in (template 1.3).
BUILDS_ON = (("nn", "train_epoch", "nn-trainer", "train-epoch"), ("nn", "trainer", "nn-trainer", "trainer"),
             ("cnn", "count", "cnn-pytorch", "count-params"), ("cnn", "color_cnn", "cv-color-cnn", "color-cnn"),
             ("cnn", "channels", "cv-color-cnn", "channel-stats"))
projects.builds_on(*BUILDS_ON)

DOWNLOAD_LINE = {
    "en": "Works on every computer. CIFAR-10, 170 MB of photos, is downloaded once from the University of Toronto when you click its button in the lesson, unless you did so in Project: your CNN on color photos; after that, the lesson works without a connection. The exercises work without it too.",
    "nl": "Werkt op elke computer. CIFAR-10, 170 MB aan foto's, wordt één keer gedownload van de University of Toronto als je in de les op de knop ervoor klikt, tenzij je dat al deed in Project: je CNN op kleurenfoto's; daarna werkt de les zonder verbinding. De oefeningen werken ook zonder.",
}
N_TRAIN, N_TEST = 6_000, 1_000   # the first photos of each set, as in cv-color-cnn
EPOCHS, LR = 40, 0.01            # the demo
PAD = 4                          # black pixels around the photo before a crop: (2 · 4 + 1)² = 81 places
LOW, HIGH = 0.8, 1.2             # the range of the factors of brightness, contrast and saturation
PHOTO = 82                       # the test photo of the pictures and the worked example: a car, in cifar.CHECKED
SHIFTS = list(range(-PAD, PAD + 1))   # how far the car moves in the plane, each way: from row 4 + up, column 4 - right
SHOWN = tuple(cifar.checked(kind, 1)[0] for kind in cifar.KINDS)   # one checked test photo of each kind, for the rewind
MOMENTS = (1, 2, 5, 10, 20, 30, 40)   # the epochs to rewind to after the demo, with its last epoch
NAMES = {
    "en": dict(zip(cifar.KINDS, ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"))),
    "nl": dict(zip(cifar.KINDS, ("vliegtuig", "auto", "vogel", "kat", "hert", "hond", "kikker", "paard", "schip", "vrachtwagen"))),
}

# What the course measured, on one thread, with the course's code: ColorCNN of cv-color-cnn after
# torch.manual_seed(seed), trained with the Trainer of nn-trainer (a learning rate of 0.01, a warmup of one epoch,
# max_norm 1.0, mini-batches of 64) on the first N_TRAIN training photos, after torch.manual_seed(seed + 1), for the
# seeds 0 to 4. The photos come from the Dataset of the demo (_Augmented): for every photo, 6 numbers from torch.rand;
# the first below 0.5 mirrors it ("flip") or turns it upside down ("upside"), the next two give the place of the crop
# ("crop", int(u · 9)), the last three the factors of the color ("color", LOW + (HIGH − LOW) · u), and then every
# number is kept between 0 and 1; "all" is flip, crop and color, "none" no changes (with 20 epochs, cv-color-cnn's
# run). The photos are then normalized with the mean and the standard deviation of each color of the training photos
# as they are; the test photos are only normalized. "train" and "test" are how many of the training photos as they
# are and of the first N_TEST test photos are read right after the last epoch; "best" is the first epoch with the
# lowest validation loss, rounded to 3 decimals, as the demo takes it too, "last_val" and "last_train" the validation and training loss of the last
# epoch. "curves" are seed 0 with "all", and the validation loss of seed 0 with "none", both after 40 epochs.
# tests/test_cv_augment_photos.py measures seed 0 of "all" again when EAI_CIFAR_FILE names the unpacked cifar-10.npz.
REFERENCE = {
    ("none", 20): {"train": [5710, 5514, 5785, 5752, 5800], "test": [519, 515, 523, 543, 538], "best": [3, 4, 4, 6, 5],
                   "last_val": [2.369, 2.188, 2.349, 2.179, 2.386], "last_train": [0.212, 0.305, 0.183, 0.186, 0.164]},
    ("none", 40): {"train": [5992, 5960, 6000, 5998, 6000], "test": [498, 460, 502, 545, 511], "best": [6, 4, 4, 5, 3],
                   "last_val": [4.425, 4.118, 4.467, 4.193, 4.478], "last_train": [0.026, 0.065, 0.014, 0.014, 0.01]},
    ("flip", 40): {"train": [5114, 5122, 5120, 5266, 5109], "test": [571, 572, 541, 591, 559], "best": [11, 8, 6, 6, 9],
                   "last_val": [1.685, 1.636, 1.621, 1.697, 1.697], "last_train": [0.437, 0.454, 0.465, 0.404, 0.463]},
    ("flip+crop", 40): {"train": [3969, 3875, 3881, 4029, 4004], "test": [601, 592, 590, 607, 599], "best": [38, 40, 37, 36, 39],
                        "last_val": [1.131, 1.179, 1.181, 1.095, 1.187], "last_train": [1.08, 1.129, 1.137, 1.062, 1.078]},
    ("all", 40): {"train": [4015, 3917, 3967, 4024, 3986], "test": [617, 593, 583, 619, 598], "best": [39, 33, 39, 36, 39],
                  "last_val": [1.072, 1.176, 1.167, 1.079, 1.162], "last_train": [1.122, 1.136, 1.122, 1.1, 1.111]},
    ("upside", 40): {"train": [4603, 4569, 4793, 4816, 4718], "test": [492, 482, 490, 485, 479], "best": [12, 6, 6, 13, 9],
                     "last_val": [1.698, 1.781, 1.792, 1.898, 1.864], "last_train": [0.721, 0.716, 0.632, 0.598, 0.628]},
}
CURVES = {
    "train": [2.015, 1.778, 1.717, 1.657, 1.595, 1.579, 1.555, 1.536, 1.487, 1.486, 1.449, 1.436, 1.444, 1.429, 1.39,
              1.378, 1.347, 1.32, 1.314, 1.317, 1.291, 1.275, 1.267, 1.246, 1.238, 1.229, 1.208, 1.186, 1.178, 1.167,
              1.159, 1.149, 1.146, 1.115, 1.13, 1.115, 1.108, 1.103, 1.097, 1.122],
    "val": [1.77, 1.626, 1.527, 1.546, 1.475, 1.416, 1.468, 1.457, 1.454, 1.357, 1.371, 1.332, 1.369, 1.338, 1.283, 1.27,
            1.262, 1.256, 1.224, 1.217, 1.21, 1.186, 1.146, 1.151, 1.195, 1.152, 1.148, 1.179, 1.111, 1.097, 1.112, 1.117,
            1.084, 1.088, 1.079, 1.074, 1.077, 1.074, 1.072, 1.072],
    "none_val": [1.699, 1.484, 1.433, 1.433, 1.499, 1.385, 1.506, 1.555, 1.584, 1.697, 1.728, 1.827, 1.952, 2.028, 2.148,
                 2.174, 2.391, 2.425, 2.61, 2.734, 2.968, 2.916, 3.084, 3.286, 3.314, 3.554, 3.646, 3.816, 3.852, 3.926,
                 4.086, 4.183, 4.286, 4.306, 4.369, 4.386, 4.415, 4.422, 4.425, 4.425],
}


def low_high(values) -> tuple:
    return min(values), max(values)


def grouped(n: int, lang: str) -> str:
    """A whole number with its thousands marked, as the language writes it: 5,514 or 5.514."""
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def percent(right: int, of: int) -> int:
    return round(100 * right / of)


# --- the changes, as the course makes them -------------------------------------------------------------------------

def course_crop(x, top: int, left: int):
    padded = torch.zeros(x.shape[0], x.shape[1] + 2 * PAD, x.shape[2] + 2 * PAD)
    padded[:, PAD:PAD + x.shape[1], PAD:PAD + x.shape[2]] = x
    return padded[:, top:top + x.shape[1], left:left + x.shape[2]]


def course_jitter(x, b: float, c: float, s: float):
    x = x * b
    m = x.mean()
    x = (x - m) * c + m
    g = x.mean(dim=0)
    return (x - g) * s + g


# --- CIFAR-10 on the learner's computer ----------------------------------------------------------------------------

def ready() -> bool:
    return cifar.unpacked()


def cifar_or_stop(button, texts: dict):
    """everything_ai.cifar for the lesson's cells, once CIFAR-10 is downloaded and unpacked. When it is not there
    yet, the cell stops with `texts["ask"]` and the button; a click downloads the archive, with the progress in the
    cell, and unpacks it. While it downloads, the lesson's other cells wait. `texts` has "ask", "busy" ({done} and
    {total} in MB), "unpacking", "no_net", "not_pinned", "damaged", "no_room" and "failed" ({why}): why it did not
    work; and "here"."""
    import marimo as mo
    failed = None
    if not ready() and button.value:
        try:
            if hub.pinned(*cifar.TAR) is None:
                total = round(hub.PINNED[cifar.TAR]["size"] / 1e6)
                last = [-1]

                def progress(done, size):
                    mb = int(done / 1e6)
                    if mb // 5 != last[0] // 5:   # every 5 MB
                        last[0] = mb
                        mo.output.replace(mo.md(texts["busy"].format(done=mb, total=total)))
                hub.download(*cifar.TAR, progress=progress)
            mo.output.replace(mo.md(texts["unpacking"]))
            cifar.unpack()
        except hub.NotPinned:
            failed = texts["not_pinned"]
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            failed = texts["no_net"]
        except cifar.Damaged:
            failed = texts["damaged"]
        except OSError as e:
            failed = texts["no_room"] if e.errno == errno.ENOSPC else texts["failed"].format(why=html.escape(str(e)))
    if not ready():
        mo.stop(True, download_card(texts["ask"], button, failed))
    mo.output.replace(mo.md(texts["here"]))
    return cifar


@functools.lru_cache(maxsize=1)
def photos():
    """The first N_TRAIN training photos and the first N_TEST test photos of CIFAR-10 as tensors of 0 to 1, of shape
    (N, 3, 32, 32), and their kinds."""
    X, y, X_test, y_test = cifar.load()
    as_floats = lambda a: torch.tensor(a, dtype=torch.float32) / 255
    return (as_floats(X[:N_TRAIN]), torch.tensor(y[:N_TRAIN], dtype=torch.int64),
            as_floats(X_test[:N_TEST]), torch.tensor(y_test[:N_TEST], dtype=torch.int64))


# --- the pictures --------------------------------------------------------------------------------------------------

# The changes the picture shows, in its order: (key, mirrored, upside down, color (b, c, s)). The crops are in the
# plane (crop_frames).
VARIANTS = [("as_is", False, False, None), ("mirrored", True, False, None), ("brighter", False, False, (1.2, 1.2, 1.0)),
            ("duller", False, False, (0.8, 1.0, 0.8)), ("upside", False, True, None)]


def variant(x, mirrored: bool, upside: bool, factors):
    """A photo of 0 to 1 of shape (3, 32, 32), changed as one entry of VARIANTS says."""
    if mirrored:
        x = x.flip(dims=[2])
    if upside:
        x = x.flip(dims=[1])
    if factors is not None:
        x = course_jitter(x, *factors).clamp(0, 1)
    return x


def variants_picture(labels: dict, lang: str) -> str:
    """Test photo PHOTO, a car, as it is and changed four ways, each with its label."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    _, _, X_test, _ = cifar.load()
    x = torch.tensor(X_test[PHOTO], dtype=torch.float32) / 255
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(5.4, 1.5))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.76, bottom=0.02, wspace=0.08)
    for ax, (key, *how) in zip(axes, VARIANTS):
        ax.imshow(np.transpose(variant(x, *how).numpy(), (1, 2, 0)), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(labels[key], fontsize=8, loc="center")
    out = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return out


def crop_frames(labels: dict) -> dict:
    """For the plane: for every shift (right, up) of the car of PHOTO, each from -4 to 4 pixels, the car in its black
    photo of 40 by 40 with the 32 by 32 that is cut out marked, and next to it that cut, as HTML. The car moves `up`
    pixels up and `right` pixels to the right in the cut from row 4 + up and column 4 - right. `labels` gives "whole",
    and "cut" and "alt" with {top} and {left}. One figure, drawn again for every shift."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    _, _, X_test, _ = cifar.load()
    x = torch.tensor(X_test[PHOTO], dtype=torch.float32) / 255
    side = x.shape[1] + 2 * PAD
    padded = torch.zeros(3, side, side)
    padded[:, PAD:PAD + x.shape[1], PAD:PAD + x.shape[2]] = x
    as_image = lambda t: np.transpose(t.numpy(), (1, 2, 0))
    fig, (whole, cut) = plt.subplots(1, 2, figsize=(4.6, 2.6), gridspec_kw={"width_ratios": [side, x.shape[1]], "wspace": 0.12})
    whole.imshow(as_image(padded), interpolation="nearest")
    shown = cut.imshow(as_image(x), interpolation="nearest")
    edge, = whole.plot([], [], color=plots.PAGE, linestyle="-", linewidth=3.2, solid_joinstyle="miter")   # stands out on black and on the car
    line, = whole.plot([], [], color=plots.INK, linestyle="-", linewidth=1.5, solid_joinstyle="miter")
    whole.set_xlim(-0.5, side - 0.5)
    whole.set_ylim(side - 0.5, -0.5)
    for ax in (whole, cut):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    whole.set_title(labels["whole"], fontsize=9)
    title = cut.set_title("", fontsize=9)
    frames = {}
    for up in SHIFTS:
        for right in SHIFTS:
            top, left = PAD + up, PAD - right
            shown.set_data(as_image(course_crop(x, top, left)))
            xs = [left - 0.5, left + 31.5, left + 31.5, left - 0.5, left - 0.5]
            ys = [top - 0.5, top - 0.5, top + 31.5, top + 31.5, top - 0.5]
            edge.set_data(xs, ys)
            line.set_data(xs, ys)
            title.set_text(labels["cut"].format(top=top, left=left))
            frames[(right, up)] = plots.render(fig, alt=labels["alt"].format(top=top, left=left))
    plt.close(fig)
    return frames


def guesses_picture(guesses: list, truth: list, lang: str) -> str:
    """The checked test photos of the rewind, one of each kind, each with the kind the network guessed at that moment:
    in ink when it is right, in the color of mistakes and with the right kind under it when it is not."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    _, _, X_test, _ = cifar.load()
    fig, axes = plt.subplots(1, len(SHOWN), figsize=(6.8, 1.25))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.62, bottom=0.02, wspace=0.12)
    said = []
    for ax, place, guess, kind in zip(axes, SHOWN, guesses, truth):
        right = guess == kind
        ax.imshow(np.transpose(X_test[place], (1, 2, 0)), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        name, real = NAMES[lang][cifar.KINDS[guess]], NAMES[lang][cifar.KINDS[kind]]
        ax.set_title(name if right else f"{name}\n({real})", fontsize=7.5, loc="center", **({} if right else {"color": plots.LOSS_TEXT}))
        said.append(f"{real}: {name}")
    alt = {"en": "One test photo of each kind, each with the kind your network guessed then, and in brackets the right kind where it is wrong: ",
           "nl": "Eén testfoto van elke soort, elk met de soort die je netwerk toen gokte, en tussen haakjes de goede soort waar het fout is: "}
    out = plots.render(fig, alt=alt[lang] + "; ".join(said) + ".")
    plt.close(fig)
    return out


def moments(n: int) -> list:
    """The epochs to rewind to in a run of n epochs: those of MOMENTS before its last, and its last."""
    return [m for m in MOMENTS if m < n] + ([n] if n else [])


# The rows of the picture of how many photos the network reads right, from top to bottom.
ROWS = [("none", 20), ("none", 40), ("flip", 40), ("flip+crop", 40), ("all", 40), ("upside", 40)]


def gap_picture(labels: dict) -> str:
    """How many photos ColorCNN reads right, of its training photos as they are and of the test photos, with each
    set of changes: one row each, with a dot for every seed, filled for the training photos and open for the test
    photos, and a line between their means; the last row names both."""
    p = chart.Panel((40, 100), (-0.8, len(ROWS) - 0.25), title=labels["title"], xlabel=labels["x"], yticks=[],
                    xticks=[40, 50, 60, 70, 80, 90, 100])
    for k, (changes, epochs) in enumerate(ROWS):
        y = len(ROWS) - 1 - k
        run = REFERENCE[(changes, epochs)]
        train = [100 * v / N_TRAIN for v in run["train"]]
        test = [100 * v / N_TEST for v in run["test"]]
        middle = sum(test) / len(test), sum(train) / len(train)
        p.line(list(middle), [y, y], "ink-3", 1.6)
        p.points(train, [y] * len(train), "ink", r=3.5)
        p.points(test, [y] * len(test), "ink", r=3.5, hollow=True)
        p.text(41, y + 0.38, labels[changes].format(epochs=epochs), "ink-2", "start")
        if k == len(ROWS) - 1:
            p.text(middle[0], y - 0.45, labels["test"], "ink-2")
            p.text(middle[1], y - 0.45, labels["train"], "ink-2")
    svg = chart.figure([p], alt=labels["alt"], width=420, height=360)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


TOP = 2.6   # the top of the curves picture: above the loss of guessing, ln 10 ≈ 2.30


def best_epoch(val: list):
    """The first epoch with the lowest validation loss, rounded to 3 decimals as in REFERENCE: epochs 39 and 40 of
    seed 0 both have 1.072, and the lesson says 39."""
    return int(np.argmin(np.round(val, 3))) + 1 if len(val) else None


def curves_picture(curves: dict, labels: dict, lang: str, at: int | None = None) -> str:
    """The loss after every epoch of the demo's run: on the training photos solid and on the test photos dashed, as in
    torch-loss-curves, with a dot where the validation loss is lowest; and in grey the validation loss of the course's
    run without changes (one run, seed 0), up to where it leaves the top of the picture, on its way to 4.4. With `at`,
    a line at that epoch, for rewinding."""
    train, val = curves.get("train", []), curves.get("val", [])
    mine = [v for v in train + val if np.isfinite(v)]
    top = max([TOP] + [1.1 * v for v in mine])
    p = chart.Panel((0, max(len(val), len(train), len(CURVES["none_val"]), 1) + 0.5), (0, top), title=labels["title"],
                    xlabel=labels["x"], ylabel=labels["y"])
    before = CURVES["none_val"]
    out = next((k for k, v in enumerate(before) if v > top), len(before))
    xs, ys = list(range(1, out + 1)), before[:out]
    if out < len(before):   # where it crosses the top, between epoch `out` and `out + 1`
        xs.append(out + (top - before[out - 1]) / (before[out] - before[out - 1]))
        ys.append(top)
    p.line(xs, ys, "ink-3", 1.6, dashed=True)
    h = 0.9 * top   # the label's height, right of where the line climbs past it: the line does not cross it there
    up = next((k for k, v in enumerate(before) if v > h), 0)
    if up:
        p.text(up + (h - before[up - 1]) / (before[up] - before[up - 1]) + 1.0, h, labels["none"], "ink-3", "start")
    else:
        p.text(xs[-1], ys[-1] + 0.04 * top, labels["none"], "ink-3", "end")
    train_above = bool(train and val and train[-1] >= val[-1])   # with changed photos, the training loss can end higher
    for ys, dashed, name, above in ((train, False, labels["train"], train_above), (val, True, labels["val"], not train_above)):
        if ys:
            xs = list(range(1, len(ys) + 1))
            p.line(xs, ys, "loss", 2.2, dashed=dashed)
            p.text(len(ys), ys[-1] + (0.07 if above else -0.08) * top, name, "loss", "end")
    if val:
        best = best_epoch(val)
        p.points([best], [val[best - 1]], "loss", r=4)
    if at is not None:
        p.line([at, at], [0, top], "ink-2", 1.2)
    svg = chart.figure([p], alt=labels["alt"], width=380, height=260)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


# The lesson is done when these are correct. "augment-gap" is the choice question in the lesson.
REQUIRED = ["photo-changes", "crop", "color", "augment-gap"]

# a `...` on a line of its own, or after `=` or `return`: not `padded[..., 4:36, 4:36]`, which is correct
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


_S_EN = "With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_S_NL = "Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"

# --- exercise 1: which changes help --------------------------------------------------------------------------------

_SORT = Sort(
    name="photo-changes",
    items={
        "mirror": {"en": "**Mirror image.** Left and right swapped.", "nl": "**Spiegelbeeld.** Links en rechts omgewisseld."},
        "shift": {"en": "**Shifted 4 pixels.** The photo itself unchanged, with a thin black band along two edges, as in a random crop.", "nl": "**4 pixels verschoven.** De foto zelf onveranderd, met een smalle zwarte rand langs twee kanten, zoals bij een random crop."},
        "brighter": {"en": "**A little brighter.** Every color times 1.2.", "nl": "**Iets lichter.** Elke kleur keer 1,2."},
        "upside": {"en": "**Upside down.** Top and bottom swapped.", "nl": "**Ondersteboven.** Boven en onder omgewisseld."},
        "turned": {"en": "**Turned a quarter.** Lying on its side.", "nl": "**Een kwartslag gedraaid.** Op zijn kant."},
        "corner": {"en": "**One corner, enlarged.** The top left 8 by 8, blown up to 32 by 32.", "nl": "**Eén hoek, vergroot.** De 8 bij 8 linksboven, opgeblazen tot 32 bij 32."},
    },
    groups={"keeps": {"en": "Keeps the kind, and looks like a real photo", "nl": "Blijft dezelfde soort, en lijkt op een echte foto"},
            "breaks": {"en": "Can lose what it shows, or looks unlike a real photo", "nl": "Kan kwijtraken wat erop staat, of lijkt niet op een echte foto"}},
    solution={"mirror": "keeps", "shift": "keeps", "brighter": "keeps", "upside": "breaks", "turned": "breaks", "corner": "breaks"},
    nudge={
        "mirror": {"en": "Do real photos show cars driving to the left as well as to the right?",
                   "nl": "Laten echte foto's auto's zien die naar links rijden, en ook naar rechts?"},
        "shift": {"en": "Do real photos all put the horse in exactly the same place? And does a band of 4 pixels, of 32, change what the photo shows?",
                  "nl": "Zetten echte foto's het paard allemaal op precies dezelfde plek? En verandert een rand van 4 pixels, van de 32, wat er op de foto staat?"},
        "brighter": {"en": "Do real photos all have the same light, or is one taken in the sun and another at dusk?",
                     "nl": "Hebben echte foto's allemaal hetzelfde licht, of is de ene in de zon genomen en de andere in de schemering?"},
        "upside": {"en": "How often does a photo show a car on its roof, or a ship with the sea above it?",
                   "nl": "Hoe vaak laat een foto een auto op zijn dak zien, of een schip met de zee erboven?"},
        "turned": {"en": "How many of the test photos show a horse standing on the side of the photo?",
                   "nl": "Hoeveel testfoto's laten een paard zien dat op de zijkant van de foto staat?"},
        "corner": {"en": "What if the bird sits in the bottom right?",
                   "nl": "Wat als de vogel rechtsonder zit?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the changes again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de veranderingen opnieuw uit."},
        "empty": {"en": "Put each change in a group: keeps the kind and looks like a real photo, or can lose what it shows or looks unlike a real photo.",
                  "nl": "Zet elke verandering in een groep: blijft dezelfde soort en lijkt op een echte foto, of kan kwijtraken wat erop staat of lijkt niet op een echte foto."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "{item} belongs in the other group.",
                  "nl": "{item} hoort in de andere groep."},
        "correct": {"en": "All {n} are right. A mirror image, a small shift and a little more light keep what a photo shows, and real photos differ from each other in those ways too. Upside down or on its side, a photo shows the network something the test photos never show; a small corner can lose the car or the bird altogether.",
                    "nl": "Alle {n} kloppen. Een spiegelbeeld, een paar pixels opzij en iets meer licht veranderen niets aan wat er op een foto staat, en echte foto's verschillen ook zo van elkaar. Ondersteboven of op zijn kant laat een foto het netwerk iets zien wat de testfoto's nooit laten zien; een kleine hoek kan de auto of de vogel helemaal kwijtraken."},
    },
)

# --- exercise 2: crop ----------------------------------------------------------------------------------------------

_C_HEAD = "import torch\n\n\ndef crop(x, top, left):\n"
_C_START = (_C_HEAD + "    # x: a photo of shape (3, 32, 32), floats from 0 to 1\n"
            "    # put it in a black photo of 40 by 40, 4 pixels from every edge, and give\n"
            "    # back the 32 by 32 of that black photo that starts at row top, column left\n    ...\n")
_C_BODY = ("    padded = torch.zeros(3, 40, 40)\n    padded[:, 4:36, 4:36] = x\n"
           "    return padded[:, top:top + 32, left:left + 32]\n")


def _c(body: str = _C_BODY) -> str:
    return _C_HEAD + body


_C = _c()
_C_X = "import torch\n\ntorch.manual_seed(0)\nx = torch.rand(3, 32, 32)\n"
_C_SCENARIOS = [
    # cut out from row 0 and column 8: the photo 4 pixels down and 4 to the left, black above and on the right; the
    # shape, the sums of the black rows and columns, and a pixel of the photo, x[0, 0, 4], which lands at row 4, column 0
    _C_X + "out = crop(x, 0, 8)\n[list(out.shape), out[:, :4].sum().item(), out[:, :, 28:].sum().item(), out[0, 4, 0].item(), x[0, 0, 4].item()]",
    # from row 8 and column 1: 4 pixels up and 3 to the right, black below and on the left
    _C_X + "out = crop(x, 8, 1)\n[list(out.shape), out[:, 28:].sum().item(), out[:, :, :3].sum().item(), out[2, 0, 3].item(), x[2, 4, 0].item()]",
    # from row 4 and column 4: the photo itself, so nothing differs; and the two opposite corners
    _C_X + "[(crop(x, 4, 4) - x).abs().sum().item(), crop(x, 8, 8)[1, 27, 27].item(), crop(x, 0, 0)[1, 4, 4].item()]",
]
_C_WRONG = {
    "no-border": _c("    return x[:, top:top + 32, left:left + 32]\n"),
    "roll": _c("    return torch.roll(x, shifts=(4 - top, 4 - left), dims=(1, 2))\n"),
    "swapped": _c("    padded = torch.zeros(3, 40, 40)\n    padded[:, 4:36, 4:36] = x\n    return padded[:, left:left + 32, top:top + 32]\n"),
    "corner": _c("    padded = torch.zeros(3, 40, 40)\n    padded[:, :32, :32] = x\n    return padded[:, top:top + 32, left:left + 32]\n"),
}

# --- exercise 3: jitter --------------------------------------------------------------------------------------------

_J_HEAD = "import torch\n\n\ndef jitter(x, b, c, s):\n"
_J_START = (_J_HEAD + "    # x: a photo of shape (3, H, W); b, c and s: numbers close to 1\n"
            "    # brightness: every number times b\n"
            "    # contrast: the distance to the mean of the whole photo, times c\n"
            "    # saturation: the distance to the gray of each pixel, the mean of its 3 colors, times s\n    ...\n")
_J_BODY = ("    x = x * b\n    m = x.mean()\n    x = (x - m) * c + m\n    g = x.mean(dim=0)\n    x = (x - g) * s + g\n"
           "    return x\n")


def _j(body: str = _J_BODY) -> str:
    return _J_HEAD + body


_J = _j()
# Every number stays between 0 and 1 in these photos, so a jitter that keeps them there, as the Dataset of the demo does
# after it, gives the same.
_J_SCENARIOS = [
    # a made-up photo of 4 by 4, brighter, with more contrast and less color: the shape, the mean of each color, and
    # the darkest and the lightest number
    "import torch\n\ntorch.manual_seed(0)\nx = 0.2 + 0.6 * torch.rand(3, 4, 4)\nout = jitter(x, 1.1, 1.1, 0.9)\n"
    "[list(out.shape), out.mean(dim=(1, 2)).tolist(), out.min().item(), out.max().item()]",
    # a photo of 2 by 2 with one red pixel on gray, darker, with less contrast and more color: that pixel, and a gray one
    "import torch\n\nx = torch.tensor([[[0.9, 0.5], [0.5, 0.5]], [[0.2, 0.5], [0.5, 0.5]], [[0.1, 0.5], [0.5, 0.5]]])\n"
    "out = jitter(x, 0.8, 0.9, 1.2)\n[out[:, 0, 0].tolist(), out[:, 1, 1].tolist()]",
    # one change at a time: each factor on its own
    # (a new photo for every call, the same each time, so that a jitter that changes its x itself counts too)
    "import torch\n\n\ndef photo():\n    torch.manual_seed(1)\n    return 0.2 + 0.6 * torch.rand(3, 3, 3)\n\n\n"
    "[jitter(photo(), 1.1, 1, 1)[0, 0, 0].item(), jitter(photo(), 1, 1.2, 1)[1, 1, 1].item(), jitter(photo(), 1, 1, 0.5)[2, 2, 2].item()]",
]
_J_WRONG = {
    "contrast-half": _j("    x = x * b\n    x = (x - 0.5) * c + 0.5\n    g = x.mean(dim=0)\n    x = (x - g) * s + g\n    return x\n"),
    "contrast-color": _j("    x = x * b\n    m = x.mean(dim=(1, 2), keepdim=True)\n    x = (x - m) * c + m\n    g = x.mean(dim=0)\n    x = (x - g) * s + g\n    return x\n"),
    "gray-photo": _j("    x = x * b\n    m = x.mean()\n    x = (x - m) * c + m\n    g = x.mean()\n    x = (x - g) * s + g\n    return x\n"),
    "gray-place": _j("    x = x * b\n    m = x.mean()\n    x = (x - m) * c + m\n    g = x.mean(dim=(1, 2), keepdim=True)\n    x = (x - g) * s + g\n    return x\n"),
    "plus": _j("    x = x + b - 1\n    m = x.mean()\n    x = (x - m) * c + m\n    g = x.mean(dim=0)\n    x = (x - g) * s + g\n    return x\n"),
}


EXERCISES = {
    "photo-changes": _SORT,
    "crop": ScenarioBox(
        name="crop",
        start=_C_START,
        solution=_C,
        scenarios=_C_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "crop"),
        first=[_DOTS],
        mistakes=[
            Mistake("no-return", error=lambda e: isinstance(e, AttributeError) and "'NoneType' object has no attribute" in str(e),
                    en="The check asks for the `.shape` of what your function gives back, and it gives back nothing, `None`: end with `return` and the 32 by 32 you cut out.",
                    nl="De check vraagt de `.shape` van wat je functie teruggeeft, en die geeft niets terug, `None`: eindig met `return` en de 32 bij 32 die je uitsnijdt."),
            Mistake("no-border", wrong=_C_WRONG["no-border"],
                    en=_S_EN + ": you cut out of the photo itself, so from row 0 and column 8 there are only 24 columns left, and nothing is black. First put the photo in a black photo of 40 by 40, `torch.zeros(3, 40, 40)`, at rows and columns 4 to 35.",
                    nl=_S_NL + ": je snijdt uit de foto zelf, dus vanaf rij 0 en kolom 8 blijven er maar 24 kolommen over, en is er niets zwart. Zet de foto eerst in een zwarte foto van 40 bij 40, `torch.zeros(3, 40, 40)`, op de rijen en kolommen 4 tot en met 35."),
            Mistake("roll", wrong=_C_WRONG["roll"],
                    en=_S_EN + ": the photo is shifted the right way, but with `torch.roll`, what falls off one side comes back on the other, where the crop has black. Around a digit, that was empty paper; around a photo, it is a piece of the other side of the photo. Put the photo in a black photo of 40 by 40, and cut out of that.",
                    nl=_S_NL + ": de foto is de goede kant op verschoven, maar met `torch.roll` komt wat er aan de ene kant afvalt aan de andere kant terug, waar de crop zwart heeft. Rond een cijfer was dat leeg papier; rond een foto is het een stuk van de andere kant van de foto. Zet de foto in een zwarte foto van 40 bij 40, en snijd daaruit."),
            Mistake("swapped", wrong=_C_WRONG["swapped"],
                    en=_S_EN + ": the black is on the wrong sides, because `top` and `left` changed places. In a photo of shape (3, 32, 32), axis 1 holds the rows, counted from the top, and axis 2 the columns, counted from the left: rows first, then columns.",
                    nl=_S_NL + ": het zwart zit aan de verkeerde kanten, want `top` en `left` zijn van plaats gewisseld. In een foto met shape (3, 32, 32) staan op axis 1 de rijen, geteld vanaf boven, en op axis 2 de kolommen, geteld vanaf links: eerst de rijen, dan de kolommen."),
            Mistake("corner", wrong=_C_WRONG["corner"],
                    en=_S_EN + ": the photo sits in the top left corner of the black photo, so all the black is below it and on its right. Put it 4 pixels from every edge, at rows and columns 4 to 35: `padded[:, 4:36, 4:36] = x`. From row 4 and column 4, the crop then gives the photo itself.",
                    nl=_S_NL + ": de foto zit in de hoek linksboven van de zwarte foto, dus zit al het zwart eronder en rechts ervan. Zet hem 4 pixels van elke rand, op de rijen en kolommen 4 tot en met 35: `padded[:, 4:36, 4:36] = x`. Vanaf rij 4 en kolom 4 geeft de crop dan de foto zelf."),
            Mistake("sizes", error=lambda e: isinstance(e, RuntimeError) and re.search(r"must match|expanded size|shape mismatch", str(e)) is not None,
                    en="The sizes do not fit where you put the photo in the black one: the photo is 32 by 32, so it needs a place of 32 rows and 32 columns, such as `[:, 4:36, 4:36]`, in a black photo of 40 by 40.",
                    nl="De maten passen niet waar je de foto in de zwarte zet: de foto is 32 bij 32, dus hij heeft een plek van 32 rijen en 32 kolommen nodig, zoals `[:, 4:36, 4:36]`, in een zwarte foto van 40 bij 40."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: a black photo of 40 by 40 with the photo in the middle, and the 32 by 32 from row `top` and column `left`.",
                      "nl": "Vervang de `...` door je code: een zwarte foto van 40 bij 40 met de foto in het midden, en de 32 bij 32 vanaf rij `top` en kolom `left`."},
            "wrong": {"en": _S_EN + ": the shape, the sums of the rows and columns that should be black, the pixel where a pixel of the photo should land, and that pixel of the photo. Is the photo at rows and columns 4 to 35 of the black photo, and do you cut 32 rows from `top` and 32 columns from `left`?",
                      "nl": _S_NL + ": de shape, de sommen van de rijen en kolommen die zwart moeten zijn, de pixel waar een pixel van de foto moet landen, en die pixel van de foto. Staat de foto op de rijen en kolommen 4 tot en met 35 van de zwarte foto, en snijd je 32 rijen vanaf `top` en 32 kolommen vanaf `left`?"},
            "almost": {"en": "Right for the first crop. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does it work for every `top` and `left` from 0 to 8?",
                       "nl": "Goed voor de eerste crop. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt het voor elke `top` en `left` van 0 tot en met 8?"},
            "correct": {"en": "Your `crop` is right: from row 4 and column 4 it gives the photo itself, and from any other of the 81 places the photo shifted by up to 4 pixels, with black where it moved away. It is now in your project, as `{saved}`.",
                        "nl": "Je `crop` klopt: vanaf rij 4 en kolom 4 geeft hij de foto zelf, en vanaf elke andere plek van de 81 de foto tot 4 pixels verschoven, met zwart waar hij wegschoof. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `crop` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `crop` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "color": ScenarioBox(
        name="color",
        start=_J_START,
        solution=_J,
        scenarios=_J_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "color"),
        first=[_DOTS],
        mistakes=[
            Mistake("no-return", error=lambda e: isinstance(e, AttributeError) and "'NoneType' object has no attribute" in str(e),
                    en="The check asks for the shape and the numbers of what your function gives back, and it gives back nothing, `None`: end with `return x`, the changed photo.",
                    nl="De check vraagt de shape en de getallen van wat je functie teruggeeft, en die geeft niets terug, `None`: eindig met `return x`, de veranderde foto."),
            Mistake("contrast-half", wrong=_J_WRONG["contrast-half"],
                    en=_S_EN + ": your contrast is around 0.5, the middle between black and white. It goes around the mean of the photo itself, `x.mean()`: a dark photo then stays dark, with its light and dark parts further apart.",
                    nl=_S_NL + ": je contrast gaat rond 0,5, het midden tussen zwart en wit. Het gaat rond het gemiddelde van de foto zelf, `x.mean()`: een donkere foto blijft dan donker, met zijn lichte en donkere delen verder uit elkaar."),
            Mistake("contrast-color", wrong=_J_WRONG["contrast-color"],
                    en=_S_EN + ": your contrast is around the mean of each color, so a gray pixel no longer stays gray: in a red photo, less contrast makes it reddish. Around one number, the mean of the whole photo, `x.mean()`, the same for all three colors, gray stays gray.",
                    nl=_S_NL + ": je contrast gaat rond het gemiddelde van elke kleur, dus blijft een grijze pixel niet grijs: in een rode foto wordt hij bij minder contrast roodachtig. Rond één getal, het gemiddelde van de hele foto, `x.mean()`, hetzelfde voor alle drie de kleuren, blijft grijs grijs."),
            Mistake("gray-photo", wrong=_J_WRONG["gray-photo"],
                    en=_S_EN + ": your gray is the mean of the whole photo, one number, so your saturation changes the contrast a second time. The gray of a pixel is the mean of its own 3 colors, over axis 0: `x.mean(dim=0)`, a gray for every pixel.",
                    nl=_S_NL + ": je grijs is het gemiddelde van de hele foto, één getal, dus verandert je verzadiging het contrast nog een keer. Het grijs van een pixel is het gemiddelde van zijn eigen 3 kleuren, over axis 0: `x.mean(dim=0)`, een grijs voor elke pixel."),
            Mistake("gray-place", wrong=_J_WRONG["gray-place"],
                    en=_S_EN + ": your gray is the mean of each color over the whole photo. The gray of a pixel is the mean of its own 3 colors, over axis 0, the colors: `x.mean(dim=0)`, a gray for every pixel.",
                    nl=_S_NL + ": je grijs is het gemiddelde van elke kleur over de hele foto. Het grijs van een pixel is het gemiddelde van zijn eigen 3 kleuren, over axis 0, de kleuren: `x.mean(dim=0)`, een grijs voor elke pixel."),
            Mistake("plus", wrong=_J_WRONG["plus"],
                    en=_S_EN + ": your brightness adds to every number. Brighter light makes every color stronger by the same factor, so black stays black: every number times `b`.",
                    nl=_S_NL + ": je helderheid telt iets bij elk getal op. Feller licht maakt elke kleur met dezelfde factor sterker, dus zwart blijft zwart: elk getal keer `b`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: the brightness, the contrast and the saturation, one after the other, and give back the photo.",
                      "nl": "Vervang de `...` door je code: de helderheid, het contrast en de verzadiging, na elkaar, en geef de foto terug."},
            "wrong": {"en": _S_EN + ". Brightness: every number times `b`. Contrast: around one number, the mean of the whole photo. Saturation: around the gray of each pixel, the mean of its 3 colors. Which of the three is off?",
                      "nl": _S_NL + ". Helderheid: elk getal keer `b`. Contrast: rond één getal, het gemiddelde van de hele foto. Verzadiging: rond het grijs van elke pixel, het gemiddelde van zijn 3 kleuren. Welke van de drie klopt niet?"},
            "almost": {"en": "Right for the first photo. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does each of the three changes use its own factor?",
                       "nl": "Goed voor de eerste foto. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Gebruikt elk van de drie veranderingen zijn eigen factor?"},
            "correct": {"en": "Your `jitter` is right: brighter or darker with `b`, more or less contrast with `c`, more or less color with `s`. It is now in your project, as `{saved}`.",
                        "nl": "Je `jitter` klopt: lichter of donkerder met `b`, meer of minder contrast met `c`, meer of minder kleur met `s`. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `jitter` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `jitter` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("crop", _C_START, "empty", None),
    ("crop", _C, "correct", None),
    ("crop", _c("    out = torch.zeros(3, 40, 40)\n    out[:, 4:-4, 4:-4] = x\n    return out[:, top:top + x.shape[1], left:left + x.shape[2]]\n"), "correct", None),
    ("crop", _c("    padded = torch.zeros(3, 40, 40)\n    padded[:, 4:36, 4:36] = x\n    return padded[:, top:][:, :32, left:left + 32]\n"), "correct", None),
    ("crop", _c("    padded = torch.zeros(3, 40, 40)\n    padded[:, 4:36, 4:36] = x\n    return padded[:, top:top + 32, left:left + 32].clone()\n"), "correct", None),
    # `...` in an index is Python's Ellipsis, not a dot left over
    ("crop", _c("    padded = torch.zeros(3, 40, 40)\n    padded[..., 4:36, 4:36] = x\n    return padded[..., top:top + 32, left:left + 32]\n"), "correct", None),
    ("crop", _c("    padded = torch.zeros(3, 40, 40)\n    padded[:, 4:36, 4:36] = x\n    padded[:, top:top + 32, left:left + 32]\n"), "error", "no-return"),
    ("crop", _c("    padded = torch.zeros(3, 40, 40)\n    padded[:, 4:40, 4:40] = x\n    return padded[:, top:top + 32, left:left + 32]\n"), "error", "sizes"),
    ("crop", _c("    padded = torch.zeros(3, 40, 40)\n    padded[:, 4:36, 4:36] = x\n    return padded[:, top:top + 32, left:left + 31]\n"), "wrong", None),
    ("crop", _c("    padded = torch.zeros(3, 40, 40)\n    padded[:, 4:36, 4:36] = x\n    return ...\n"), "wrong", "dots-left"),
    ("crop", _c("    padded = torch.zeros(3, 40, 40)\n    padded[:, 4:36, 4:36] = x\n    if top == 0:\n        return padded[:, :32, left:left + 32]\n    return x\n"), "almost", None),
] + [("crop", code, "wrong", key) for key, code in _C_WRONG.items()] + [
    ("color", _J_START, "empty", None),
    ("color", _J, "correct", None),
    ("color", _j("    x = b * x\n    x = c * (x - x.mean()) + x.mean()\n    return s * (x - x.mean(dim=0)) + x.mean(dim=0)\n"), "correct", None),
    # the three changes in another order give the same photo
    ("color", _j("    g = x.mean(dim=0, keepdim=True)\n    x = (x - g) * s + g\n    m = x.mean()\n    x = (x - m) * c + m\n    return x * b\n"), "correct", None),
    # changing x itself: every call of the check gets a photo of its own
    ("color", _j("    x *= b\n    m = x.mean()\n    x -= m\n    x *= c\n    x += m\n    g = x.mean(dim=0)\n    return (x - g) * s + g\n"), "correct", None),
    # keeping every number between 0 and 1 changes nothing in these photos
    ("color", _j("    x = x * b\n    m = x.mean()\n    x = (x - m) * c + m\n    g = x.mean(dim=0)\n    return ((x - g) * s + g).clamp(0, 1)\n"), "correct", None),
    ("color", _j("    x = x * b\n    m = x.mean()\n    x = (x - m) * c + m\n    g = x.mean(dim=0)\n    x = (x - g) * s + g\n"), "error", "no-return"),
    ("color", _j("    x = x * b\n    m = x.mean()\n    x = (x - m) * c + m\n    g = x.mean(dim=0)\n    x = (x - g) * c + g\n    return x\n"), "wrong", None),
    ("color", _j("    x = x * b\n    m = x.mean()\n    x = (x - m) * c + m\n    g = x.max(dim=0).values\n    x = (x - g) * s + g\n    return x\n"), "wrong", None),
    ("color", _j("    x = x * b\n    ...\n    return x\n"), "wrong", "dots-left"),
] + [("color", code, "wrong", key) for key, code in _J_WRONG.items()]


# --- the demo: the learner's ColorCNN on changed photos ------------------------------------------------------------

class _Augmented(Dataset):
    """The first N_TRAIN training photos, each changed anew every time it is asked for: mirrored when the first of 6
    numbers from torch.rand is below 0.5, cut out at the place the next two give, and changed in color with the
    factors of the last three; then every number is kept between 0 and 1, and the photo normalized."""

    def __init__(self, X, y, mean, std, crop, jitter):
        self.X, self.y, self.crop, self.jitter = X, y, crop, jitter
        self.mean, self.std = mean[:, None, None], std[:, None, None]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].clone()   # a jitter that changes its x itself then leaves the training photo as it is
        u = torch.rand(6)
        if u[0] < 0.5:
            x = x.flip(dims=[2])
        x = self.crop(x, int(u[1] * (2 * PAD + 1)), int(u[2] * (2 * PAD + 1)))
        x = torch.as_tensor(self.jitter(x, *(LOW + (HIGH - LOW) * u[3:6]).tolist())).clamp(0, 1)
        return (x - self.mean) / self.std, self.y[i]


def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def _accuracy(model, X, y) -> int:
    model.eval()
    with torch.no_grad():
        return int(sum((model(X[i:i + 500]).argmax(dim=1) == y[i:i + 500]).sum() for i in range(0, len(X), 500)))


def _demo(show=None):
    """For the demo: None when crop.py or color.py is not in the project yet; "data" when CIFAR-10 is not unpacked;
    "crop" or "color" when the learner's crop or jitter stops with an error or gives no photo of shape (3, 32, 32);
    "cnn" when ColorCNN or channel_stats does; "trainer" when the Trainer or count_params does; "slow" when the
    training takes too long; else a dict with how many of the training photos as they are and of the test photos it
    reads right, the loss after every epoch on both ("curves"), the epoch with the lowest validation loss (best_epoch),
    the network's guess for the photos SHOWN after every epoch ("guesses") and their right kinds ("truth"), and the
    modules the course's versions stood in for. `show` hears the run while it trains (live.follow)."""
    if not (_saved("crop") and _saved("color")):
        return None
    if not ready():
        return "data"
    run = run_code("from cnn.crop import crop")
    if run.error is not None:
        return "crop"
    crop = run.names["crop"]
    run = run_code("from cnn.color import jitter")
    if run.error is not None:
        return "color"
    jitter = run.names["jitter"]
    run = run_code("from cnn.color_cnn import ColorCNN\nfrom cnn.channels import channel_stats")
    if run.error is not None:
        return "cnn"
    ColorCNN, channel_stats = run.names["ColorCNN"], run.names["channel_stats"]
    run = run_code("from nn.trainer import Trainer\nfrom cnn.count import count_params")
    if run.error is not None:
        return "trainer"
    Trainer, count_params = run.names["Trainer"], run.names["count_params"]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    X, y, X_test, y_test = photos()
    for name, change in (("crop", lambda x: crop(x, 0, 8)), ("color", lambda x: jitter(x, 1.1, 0.9, 1.2))):
        try:
            out = with_time_limit(lambda: torch.as_tensor(change(X[0].clone())), 10)
            if tuple(out.shape) != (3, 32, 32) or not torch.isfinite(out).all():
                return name
        except Exception:
            return name
    try:
        mean, std = with_time_limit(lambda: channel_stats(X.clone()), 60)
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(3)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(3)
        torch.manual_seed(0)
        model = ColorCNN()
        with torch.no_grad():
            if tuple(model(X[:2]).shape) != (2, 10):
                return "cnn"
    except Exception:
        return "cnn"
    if not (torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
        return "cnn"
    X_norm = (X - mean[:, None, None]) / std[:, None, None]
    X_test = (X_test - mean[:, None, None]) / std[:, None, None]
    try:
        params = int(count_params(model))
    except Exception:
        return "trainer"

    shown, guesses = X_test[list(SHOWN)], []

    def heard(p):
        if "val_loss" in p:   # after every epoch: the guesses for the rewind, without randomness, so the run stays the same
            was = model.training
            model.eval()
            with torch.no_grad():
                guesses.append([int(g) for g in model(shown).argmax(dim=1)])
            model.train(was)
        if show is not None:
            show(p)

    def train():
        torch.manual_seed(1)
        loader = DataLoader(_Augmented(X, y, mean, std, crop, jitter), batch_size=64, shuffle=True)
        val = DataLoader(TensorDataset(X_test, y_test), batch_size=500)
        trainer = Trainer(model, loader, val, epochs=EPOCHS, lr=LR, warmup=len(loader), max_norm=1.0)
        live.follow(trainer, heard, "augment")
        return trainer.fit() or []

    try:
        history = with_time_limit(train, 600)
    except Timeout:
        return "slow"
    except Exception:
        return "trainer"
    curves = {"train": [float(h["train_loss"]) for h in history if isinstance(h, dict) and "train_loss" in h],
              "val": [float(h["val_loss"]) for h in history if isinstance(h, dict) and "val_loss" in h]}
    return {"params": params, "train": _accuracy(model, X_norm, y), "test": _accuracy(model, X_test, y_test),
            "curves": curves, "best": best_epoch(curves["val"]), "guesses": guesses[:len(curves["val"])],
            "truth": [int(y_test[place]) for place in SHOWN], "stand_ins": stand_ins}


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file), and whether CIFAR-10
    is there."""
    texts = []
    for package, module in (("cnn", "crop"), ("cnn", "color"), ("cnn", "color_cnn"), ("cnn", "channels"),
                            ("cnn", "count"), ("nn", "train_epoch"), ("nn", "trainer")):
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
