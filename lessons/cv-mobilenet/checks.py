# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-mobilenet: a depthwise convolution gives every channel a 3×3 filter of its own, a grouped
convolution with as many groups as channels, and a 1×1 convolution after it mixes the channels; together, a depthwise
separable convolution, they do the work of a 3×3 convolution with far fewer multiplications and weights. The learner's
SeparableConv is saved in their project (work/projects/cnn/separable.py) when right, and checked on made-up numbers, so
it works without CIFAR-10. The demo trains the CNN of cv-1x1-convolution with a middle of the learner's choice on its 64
maps of 8 by 8: their SeparableConv, two of them in a row, or, from the course, only the depthwise convolution or only
the 1×1 convolution. It trains with the learner's Trainer on the first 6,000 training photos of CIFAR-10, tests on the
first 1,000 test photos, and keeps the network's guess for one checked test photo of each kind after every epoch, for
rewinding."""
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
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_mobilenet",
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

def full_mults(c_in: int, c_out: int, side: int, k: int = 3) -> int:
    """The multiplications of a k×k convolution from c_in to c_out channels on maps of side by side, with stride 1 and
    the padding that keeps the size: each of the c_out · side² numbers it gives needs k · k · c_in, the paper's
    D_K · D_K · M · N · D_F · D_F."""
    return k * k * c_in * c_out * side * side


def depthwise_mults(c_in: int, side: int, k: int = 3) -> int:
    """A depthwise k×k convolution: one filter of k × k for each of the c_in channels, D_K · D_K · M · D_F · D_F."""
    return k * k * c_in * side * side


def pointwise_mults(c_in: int, c_out: int, side: int) -> int:
    """A 1×1 convolution from c_in to c_out channels, M · N · D_F · D_F."""
    return c_in * c_out * side * side


def separable_mults(c_in: int, c_out: int, side: int, k: int = 3) -> int:
    """A depthwise separable convolution: the depthwise convolution, then the 1×1 convolution."""
    return depthwise_mults(c_in, side, k) + pointwise_mults(c_in, c_out, side)


def full_weights(c_in: int, c_out: int, k: int = 3) -> int:
    """The weights of a k×k convolution, without its biases: with stride 1, each is used once for every pixel it gives."""
    return k * k * c_in * c_out


def separable_weights(c_in: int, c_out: int, k: int = 3) -> int:
    """The weights of a depthwise separable convolution, without biases."""
    return k * k * c_in + c_in * c_out


def separable_params(c_in: int, c_out: int, k: int = 3) -> int:
    """The parameters of a depthwise separable convolution as nn.Conv2d makes it: with a bias for every filter."""
    return c_in * (k * k + 1) + conv_params(c_in, c_out)


def times_fewer(c_out: int, k: int = 3) -> float:
    """How many times fewer multiplications a depthwise separable convolution needs than a full one: the full one over
    the separable one, k²·N / (k² + N); the channels that come in drop out of it."""
    return full_mults(1, c_out, 1, k) / separable_mults(1, c_out, 1, k)


def share(c_out: int, k: int = 3) -> tuple[float, float]:
    """The multiplications of the depthwise and of the 1×1 convolution as shares of the full convolution's: the paper's
    1/N and 1/D_K²."""
    return 1 / c_out, 1 / (k * k)


# The demo's middles on the 64 maps of 8 by 8 in the middle of WideCNN.
VARIANTS = ("separable", "two", "depthwise", "pointwise")


def middle_params(variant: str) -> int:
    """The parameters of the demo's middle for `variant`."""
    return {"plain": PLAIN_BLOCK, "separable": separable_params(CHANNELS, CHANNELS),
            "two": 2 * separable_params(CHANNELS, CHANNELS), "depthwise": CHANNELS * (9 + 1),
            "pointwise": conv_params(CHANNELS, CHANNELS)}[variant]


def middle_mults(variant: str) -> int:
    """The multiplications of the demo's middle for `variant`, for one photo: on the 64 maps of 8 by 8."""
    one = separable_mults(CHANNELS, CHANNELS, SIDE)
    return {"plain": full_mults(CHANNELS, CHANNELS, SIDE), "separable": one, "two": 2 * one,
            "depthwise": depthwise_mults(CHANNELS, SIDE), "pointwise": pointwise_mults(CHANNELS, CHANNELS, SIDE)}[variant]


# --- the block, as the course writes it -----------------------------------------------------------------------------

_S_HEAD = ("import torch\nfrom torch import nn\n\n\nclass SeparableConv(nn.Module):\n"
           "    def __init__(self, channels_in, channels_out):\n        super().__init__()\n")
_S_FORWARD = ("\n    def forward(self, x):\n        x = torch.relu(self.depthwise(x))\n"
              "        return torch.relu(self.pointwise(x))\n")
_DW = "nn.Conv2d(channels_in, channels_in, 3, padding=1, groups=channels_in)"
_PW = "nn.Conv2d(channels_in, channels_out, 1)"


def _s(dw: str = _DW, pw: str = _PW) -> str:
    return _S_HEAD + f"        self.depthwise = {dw}\n        self.pointwise = {pw}\n" + _S_FORWARD


_S = _s()


class Depthwise(nn.Module):
    """The course's middle with only a depthwise convolution: a 3×3 filter for each channel, with a ReLU."""

    def __init__(self, channels: int = CHANNELS):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, x):
        return torch.relu(self.conv(x))


class Pointwise(nn.Module):
    """The course's middle with only a 1×1 convolution, with a ReLU."""

    def __init__(self, channels: int = CHANNELS):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return torch.relu(self.conv(x))


def make_middle(variant: str, SeparableConv):
    """The demo's middle for `variant`: the learner's SeparableConv, two of them in a row, or the course's depthwise or
    1×1 convolution alone."""
    if variant == "depthwise":
        return Depthwise()
    if variant == "pointwise":
        return Pointwise()
    if variant == "two":
        return nn.Sequential(SeparableConv(CHANNELS, CHANNELS), SeparableConv(CHANNELS, CHANNELS))
    return SeparableConv(CHANNELS, CHANNELS)


# --- the demo: the middle of WideCNN, on CIFAR-10 -------------------------------------------------------------------

def _accuracy(model, X, y) -> int:
    model.eval()
    with torch.no_grad():
        return int(sum((model(X[i:i + 500]).argmax(dim=1) == y[i:i + 500]).sum() for i in range(0, len(X), 500)))


def run(variant: str, seed: int, SeparableConv, channel_stats, Trainer, count_params, show=None):
    """One run of WideCNN with the middle of `variant` (make_middle), as REFERENCE says: the middle made after
    torch.manual_seed(seed), then the rest of the network, and the photos shuffled after torch.manual_seed(seed + 1).
    Gives back a dict with the parameters of the network and of its middle, how many of the training and test photos it
    reads right, the loss after every epoch ("curves"), the epoch with the lowest validation loss, and the network's
    guess for the photos SHOWN after every epoch ("guesses") with their right kinds ("truth"), and the feature maps of
    its middle on the car ("layers"). Errors go to the caller, which says whose code made them. `show` hears the run
    while it trains (live.follow, every 10 batches), with what the network makes then."""
    X, y, X_test, y_test = photos()
    mean_, std = channel_stats(X.clone())
    mean_ = torch.as_tensor(mean_, dtype=torch.float32).reshape(3)
    std = torch.as_tensor(std, dtype=torch.float32).reshape(3)
    X = (X - mean_[:, None, None]) / std[:, None, None]
    X_test = (X_test - mean_[:, None, None]) / std[:, None, None]
    torch.manual_seed(seed)
    middle = make_middle(variant, SeparableConv)
    model = WideCNN(middle)
    params, middle_params_ = int(count_params(model)), int(count_params(middle))
    truth, guesses, layers = [int(y_test[place]) for place in SHOWN], [], []   # what the rewind keeps of every epoch
    heard = _ONE.hearing(model, X_test[list(SHOWN)], truth, X_test[_ONE.PHOTO:_ONE.PHOTO + 1], show, guesses, layers, "block")

    torch.manual_seed(seed + 1)
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    val = DataLoader(TensorDataset(X_test, y_test), batch_size=500)
    trainer = Trainer(model, loader, val, epochs=EPOCHS, lr=LR, warmup=len(loader), max_norm=1.0)
    live.follow(trainer, heard, "mobilenet", every=EVERY)
    history = trainer.fit() or []
    curves = {"train": [float(h["train_loss"]) for h in history if isinstance(h, dict) and "train_loss" in h],
              "val": [float(h["val_loss"]) for h in history if isinstance(h, dict) and "val_loss" in h]}
    return {"variant": variant, "params": params, "middle_params": middle_params_, "train": _accuracy(model, X, y),
            "test": _accuracy(model, X_test, y_test), "curves": curves, "best": best_epoch(curves["val"]),
            "guesses": guesses[:len(curves["val"])], "layers": layers[:len(curves["val"])], "truth": truth}


# --- the picture of the filters -------------------------------------------------------------------------------------

def filters_picture(labels: dict, c_in: int = 4, c_out: int = 3):
    """The weights of three convolutions as grids, a column for each channel that comes in and a row for each filter:
    a 3×3 convolution from c_in to c_out channels (a 3×3 grid of weights in every place), the depthwise convolution
    (one filter for each channel, only on the diagonal; the other places are empty), and the 1×1 convolution after it
    (one weight in every place). Weights are in the colour of what a model learns. Under each, how many weights it has.
    `labels` gives "full", "depthwise", "pointwise", "channels" (the columns), "filters" (the rows), "weights" (with
    {n}) and "alt"."""
    CELL, GAP, PAD = 7, 7, 3
    BOX = 3 * CELL + 2 * PAD
    STEP = BOX + GAP
    TOP, LEFT, BETWEEN = 58, 34, 44
    width = c_in * STEP - GAP
    weight = "fill:var(--ea-param-wash);stroke:var(--ea-param);stroke-width:1"
    small = "font-size:13px;fill:var(--ea-ink-2)"
    parts = []

    def matrix(x0, rows, kind, title, n):
        parts.append(f'<text x="{x0}" y="16" style="font-size:14px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
        parts.append(f'<text x="{x0}" y="{TOP - 24}" style="{small}">{html.escape(labels["channels"].format(n=c_in))}</text>')
        for c in range(c_in):
            parts.append(f'<text x="{x0 + c * STEP + BOX / 2}" y="{TOP - 7}" style="{small};text-anchor:middle">{c + 1}</text>')
        for r in range(rows):
            y0 = TOP + r * STEP
            parts.append(f'<text x="{x0 - 8}" y="{y0 + BOX / 2 + 4}" style="{small};text-anchor:end">{r + 1}</text>')
            for c in range(c_in):
                x = x0 + c * STEP
                if kind == "depthwise" and c != r:   # this filter sees only its own channel: no weights here
                    parts.append(f'<rect x="{x}" y="{y0}" width="{BOX}" height="{BOX}" rx="3" '
                                 f'style="fill:none;stroke:var(--ea-ink-3);stroke-width:1;stroke-dasharray:3 3"/>')
                    continue
                parts.append(f'<rect x="{x}" y="{y0}" width="{BOX}" height="{BOX}" rx="3" '
                             f'style="fill:none;stroke:var(--ea-ink-3);stroke-width:1"/>')
                if kind == "pointwise":
                    parts.append(f'<rect x="{x + PAD + CELL}" y="{y0 + PAD + CELL}" width="{CELL}" height="{CELL}" style="{weight}"/>')
                else:
                    for i in range(3):
                        for j in range(3):
                            parts.append(f'<rect x="{x + PAD + j * CELL}" y="{y0 + PAD + i * CELL}" width="{CELL}" '
                                         f'height="{CELL}" style="{weight}"/>')
        parts.append(f'<text x="{x0}" y="{TOP + rows * STEP + 14}" style="font-size:13px;fill:var(--ea-param-text)">'
                     f'{html.escape(labels["weights"].format(n=n))}</text>')

    x_full = LEFT
    x_dw = x_full + width + BETWEEN + LEFT
    x_pw = x_dw + width + BETWEEN
    matrix(x_full, c_out, "full", labels["full"], full_weights(c_in, c_out))
    matrix(x_dw, c_in, "depthwise", labels["depthwise"], 9 * c_in)
    matrix(x_pw, c_out, "pointwise", labels["pointwise"], c_in * c_out)
    rows = max(c_in, c_out)
    H = TOP + rows * STEP + 24
    W = x_pw + max(width, 165) + 6   # room for the title over the last grid
    parts.append(f'<text x="{x_full - 22}" y="{TOP + c_out * STEP / 2}" style="{small};text-anchor:middle" '
                 f'transform="rotate(-90 {x_full - 22} {TOP + c_out * STEP / 2})">{html.escape(labels["filters"])}</text>')
    parts.append(f'<text x="{x_dw - 22}" y="{TOP + c_in * STEP / 2}" style="{small};text-anchor:middle" '
                 f'transform="rotate(-90 {x_dw - 22} {TOP + c_in * STEP / 2})">{html.escape(labels["filters"])}</text>')
    return draw._svg(W, H, labels["alt"], "".join(parts), natural=True)


# --- the slider: the share of the multiplications, for more and more channels ---------------------------------------

CHANNELS_OUT = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def share_picture(c_out: int, labels: dict, lang: str) -> str:
    """The multiplications of a depthwise separable convolution as a share of a 3×3 convolution's, with `c_out` channels
    coming out: a bar of 100% for the 3×3 convolution, and under it the depthwise separable convolution's: the 1×1
    convolution's 1/9, light, and after it the depthwise convolution's 1/c_out, dark, with a dashed line at 1/9,
    which the bar never gets below. The axis stays the same for every number of channels. `labels` gives "title", "x",
    "full", "separable", "floor" and "alt"."""
    d, p = (100 * s for s in share(c_out))
    panel = chart.Panel((0, 100), (-0.3, 2.4), title=labels["title"], xlabel=labels["x"], yticks=[],
                        xticks=[0, 25, 50, 75, 100])
    panel.rect(0, 1.45, 100, 1.75, "ink-3")
    panel.text(1, 1.98, labels["full"], "ink-2", "start")
    panel.rect(0, 0.35, p, 0.65, "ink-3")
    for _ in range(3):   # a wash three times over: dark against the 1×1 convolution's light part
        panel.rect(p, 0.35, p + d, 0.65, "ink")
    panel.text(1, 0.88, labels["separable"].format(t=number(d + p, 1, lang)), "ink-2", "start")
    panel.line([p, p], [0.15, 0.7], "ink-2", 1.2, dashed=True)   # below the label above the bar
    panel.text(p + 1.5, 0.05, labels["floor"], "ink-2", "start")
    svg = chart.figure([panel], alt=labels["alt"], width=440, height=230)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def share_sentence(c_out: int, lang: str) -> str:
    """The sentence under the slider, for `c_out` channels coming out."""
    t = number(100 * sum(share(c_out)), 1, lang)
    x = number(times_fewer(c_out), 2, lang)
    if lang == "nl":
        return (f"Met {grouped(c_out, lang)} channels eruit doet de depthwise convolution 1/{grouped(c_out, lang)} van de "
                f"vermenigvuldigingen van de 3×3-convolutie, en de 1×1-convolutie 1/9: samen {t}%, {x} keer zo weinig.")
    return (f"With {grouped(c_out, lang)} channels coming out, the depthwise convolution does 1/{grouped(c_out, lang)} of the "
            f"multiplications of the 3×3 convolution, and the 1×1 convolution 1/9: {t}% together, {x} times fewer.")


# --- exercise 1: which step does what -------------------------------------------------------------------------------

_SORT = Sort(
    name="steps",
    items={
        "neighbors": {"en": "**Neighbors.** It looks at the 3×3 neighbors of every pixel.",
                      "nl": "**Buren.** Hij kijkt naar de 3×3 buren van elke pixel."},
        "mixes": {"en": "**Mixing.** It mixes the channels of every pixel.",
                  "nl": "**Mengen.** Hij mengt de channels van elke pixel."},
        "own": {"en": "**One channel each.** Each of its filters sees only one channel.",
                "nl": "**Eén channel per filter.** Elk van zijn filters ziet maar één channel."},
        "out": {"en": "**Channels out.** It decides how many channels come out.",
                "nl": "**Channels eruit.** Hij bepaalt hoeveel channels eruit komen."},
        "work": {"en": "**Most work.** With 64 channels in and out, it does most of the multiplications.",
                 "nl": "**Meeste werk.** Met 64 channels erin en eruit doet hij de meeste vermenigvuldigingen."},
        "groups": {"en": "**Groups.** In PyTorch, its `groups` is the number of channels it gets.",
                   "nl": "**Groepen.** In PyTorch is zijn `groups` het aantal channels dat hij krijgt."},
    },
    groups={"depthwise": {"en": "the depthwise convolution", "nl": "de depthwise convolution"},
            "pointwise": {"en": "the 1×1 convolution after it", "nl": "de 1×1-convolutie erna"}},
    solution={"neighbors": "depthwise", "mixes": "pointwise", "own": "depthwise", "out": "pointwise", "work": "pointwise",
              "groups": "depthwise"},
    nudge={
        "neighbors": {"en": "Which of the two has filters of 3 by 3, and which of 1 by 1?",
                      "nl": "Welke van de twee heeft filters van 3 bij 3, en welke van 1 bij 1?"},
        "mixes": {"en": "A filter of the depthwise convolution sees one channel. Which convolution sees all the channels of a pixel at once?",
                  "nl": "Een filter van de depthwise convolution ziet één channel. Welke convolutie ziet alle channels van een pixel tegelijk?"},
        "own": {"en": "Which convolution gives every channel a filter of its own?",
                "nl": "Welke convolutie geeft elk channel een eigen filter?"},
        "out": {"en": "The depthwise convolution gives one map for each channel it gets. Which convolution can make another number?",
                "nl": "De depthwise convolution geeft één map voor elk channel dat hij krijgt. Welke convolutie kan een ander aantal maken?"},
        "work": {"en": "For one pixel, the depthwise convolution needs 9 × 64 multiplications, and the 1×1 convolution 64 × 64. Which is more?",
                 "nl": "Voor één pixel heeft de depthwise convolution 9 × 64 vermenigvuldigingen nodig, en de 1×1-convolutie 64 × 64. Wat is meer?"},
        "groups": {"en": "With as many groups as channels, each group is one channel. Which convolution is that?",
                   "nl": "Met evenveel groepen als channels is elke groep één channel. Welke convolutie is dat?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the six sentences again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de zes zinnen opnieuw uit."},
        "empty": {"en": "Choose for each sentence: the depthwise convolution or the 1×1 convolution after it.",
                  "nl": "Kies bij elke zin: de depthwise convolution of de 1×1-convolutie erna."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "{item} belongs to the other convolution.", "nl": "{item} hoort bij de andere convolutie."},
        "correct": {"en": "All {n} are right. The depthwise convolution filters: every channel gets a 3×3 filter of its own, which looks at the neighbors but at no other channel, so `groups` is its number of channels. The 1×1 convolution mixes: it looks at all the channels of one pixel, and makes as many maps as you ask. With 64 channels in and out, it does 64 × 64 = 4,096 multiplications for a pixel, and the depthwise convolution 9 × 64 = 576.",
                    "nl": "Alle {n} kloppen. De depthwise convolution filtert: elk channel krijgt een eigen 3×3-filter, dat naar de buren kijkt maar naar geen ander channel, dus is `groups` zijn aantal channels. De 1×1-convolutie mengt: hij kijkt naar alle channels van één pixel, en maakt zoveel maps als je vraagt. Met 64 channels erin en eruit doet hij 64 × 64 = 4.096 vermenigvuldigingen voor een pixel, en de depthwise convolution 9 × 64 = 576."},
    },
)

# --- exercise 2: a depthwise separable convolution ------------------------------------------------------------------

_S_START = (_S_HEAD + "        # depthwise: a 3×3 filter for each channel on its own, same size\n        self.depthwise = ...\n"
            + "        # pointwise: 1×1 from channels_in to channels_out\n        self.pointwise = ...\n" + _S_FORWARD)
# The scenarios give the weights by name, so that the order of the two lines in __init__ does not matter.
_S_WEIGHTS = ("with torch.no_grad():\n    for layer in (block.depthwise, block.pointwise):\n"
              "        for q in (layer.weight, layer.bias):\n            q.copy_(torch.randn_like(q) / 2)\n")
_S_SCENARIOS = [
    # the demo's block on made-up maps of 8 by 8: shapes and parameters
    "import torch\n\nblock = SeparableConv(64, 64)\nout = block(torch.rand(1, 64, 8, 8))\n"
    "[list(out.shape), list(block.depthwise.weight.shape), list(block.pointwise.weight.shape), sum(q.numel() for q in block.parameters())]",
    # a small block from 3 channels to 5: the numbers of every channel at a pixel in the middle and at a corner
    "import torch\n\nblock = SeparableConv(3, 5)\ntorch.manual_seed(0)\n" + _S_WEIGHTS
    + "out = block(torch.randn(1, 3, 5, 5))\n[out[0, :, 2, 2].tolist(), out[0, :, 0, 0].tolist()]",
    # from 4 channels to 8: the weights and the shape
    "import torch\n\ntorch.manual_seed(1)\nblock = SeparableConv(4, 8)\nout = block(torch.randn(2, 4, 6, 6))\n"
    "[list(block.depthwise.weight.shape), list(block.pointwise.weight.shape), list(out.shape)]",
]
_S_WRONG = {
    # no groups: a full 3×3 convolution, then the 1×1
    "no-groups": _s(dw="nn.Conv2d(channels_in, channels_in, 3, padding=1)"),
    # the second step as a 3×3 convolution
    "kernel": _s(pw="nn.Conv2d(channels_in, channels_out, 3, padding=1)"),
}


def _conv_args(text: str) -> str:
    """The arguments of the two convolutions the learner wrote, together, without their comments."""
    text = re.sub(r"#[^\n]*", "", text)
    return " ".join(re.findall(r"self\.(?:depthwise|pointwise)\s*=\s*nn\.Conv2d\(([^\n]*)\)", text))


def _first(got):
    """The first scenario's value, [output shape, depthwise weight shape, 1×1 weight shape, parameters]; None when
    `got` is another scenario's."""
    ok = (isinstance(got, list) and len(got) == 4 and isinstance(got[3], int)
          and all(isinstance(shape, list) and len(shape) == 4 for shape in got[:3]))
    return got if ok else None


def _other_size(got, text) -> bool:
    """The first scenario's block gives back maps of another height and width than the 8 by 8 it gets."""
    g = _first(got)
    return g is not None and g[0][2:] != [8, 8]


def _no_groups(got, text) -> bool:
    """The first scenario's depthwise convolution has 3×3 filters that see all 64 channels."""
    g = _first(got)
    return g is not None and g[1][1:] == [64, 3, 3]


def _swapped(got, text) -> bool:
    """The first scenario's block has a 1×1 convolution as self.depthwise and a 3×3 one as self.pointwise."""
    g = _first(got)
    return g is not None and g[1][2:] == [1, 1] and g[2][2:] == [3, 3]


def _pointwise_groups(got, text) -> bool:
    """The 1×1 convolution has `groups`, so that each of its filters sees only part of the channels."""
    g = _first(got)
    return (g is not None and g[2][2:] == [1, 1]
            and re.search(r"self\.pointwise\s*=\s*nn\.Conv2d\([^\n]*groups", re.sub(r"#[^\n]*", "", text)) is not None)


_FIXED = Mistake("fixed", matches=lambda got, text: re.search(r"\b\d{2,}\b", _conv_args(text)) is not None,
                 en="A convolution has a fixed number in it. The block must work for every number of channels: use the names `channels_in` and `channels_out`.",
                 nl="Een convolutie heeft een vast getal erin. Het block moet voor elk aantal channels werken: gebruik de namen `channels_in` en `channels_out`.")
# `...` on a line of its own, or after `=`
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$|=\s*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace both with a convolution: the depthwise one, and the 1×1 one.",
                nl="Er staat nog een `...` in je code. Vervang ze allebei door een convolutie: de depthwise, en de 1×1.")
_TEXTS = _ONE._TEXTS
_S_EN, _S_NL = _ONE._S_EN, _ONE._S_NL
_close = _ONE._close


EXERCISES = {
    "steps": _SORT,
    "separable": ScenarioBox(
        name="separable",
        picture=netview.network("SeparableConv", lambda: torch.rand(1, 64, 8, 8), 64, 64, scores=False),
        start=_S_START,
        solution=_S,
        scenarios=_S_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "separable"),
        first=[_DOTS, _FIXED],
        mistakes=[
            Mistake("groups", error=lambda e: isinstance(e, ValueError) and "in_channels must be divisible by groups" in str(e),
                    en="PyTorch stops: the channels that come in must split into `groups` groups of the same size. A depthwise convolution gives every channel a group of its own: `groups=channels_in`, as many groups as channels.",
                    nl="PyTorch stopt: de channels die binnenkomen, moeten in `groups` even grote groepen te verdelen zijn. Een depthwise convolution geeft elk channel een eigen groep: `groups=channels_in`, evenveel groepen als channels."),
            Mistake("depthwise-out", error=lambda e: isinstance(e, ValueError) and "out_channels must be divisible by groups" in str(e),
                    en="PyTorch stops: the filters of the depthwise convolution must split into its groups, and you gave it `channels_out` filters. A depthwise convolution has one filter for each channel it gets, so it makes `channels_in` maps: `nn.Conv2d(channels_in, channels_in, ...)`. The 1×1 convolution after it makes `channels_out`.",
                    nl="PyTorch stopt: de filters van de depthwise convolution moeten in zijn groepen te verdelen zijn, en je gaf hem `channels_out` filters. Een depthwise convolution heeft één filter voor elk channel dat hij krijgt, dus maakt hij `channels_in` maps: `nn.Conv2d(channels_in, channels_in, ...)`. De 1×1-convolutie erna maakt er `channels_out`."),
            Mistake("channels", error=lambda e: isinstance(e, RuntimeError) and "channels, but got" in str(e),
                    en="A convolution gets another number of channels than it was made for. The first number of `nn.Conv2d` is what it gets, the second what it makes. The depthwise convolution gets `channels_in` and makes one map for each, `channels_in`; the 1×1 convolution gets those `channels_in` and makes `channels_out`.",
                    nl="Een convolutie krijgt een ander aantal channels dan waarvoor hij gemaakt is. Het eerste getal van `nn.Conv2d` is wat hij krijgt, het tweede wat hij maakt. De depthwise convolution krijgt `channels_in` en maakt er voor elk één map, `channels_in`; de 1×1-convolutie krijgt die `channels_in` en maakt er `channels_out`."),
            Mistake("no-groups", matches=_no_groups,
                    en=_S_EN + ": your depthwise convolution has no `groups`, so it is an ordinary 3×3 convolution. Each of its filters then sees all `channels_in` channels, with 9 × `channels_in` weights instead of 9. Give every channel a filter of its own: `groups=channels_in`.",
                    nl=_S_NL + ": je depthwise convolution heeft geen `groups`, dus is hij een gewone 3×3-convolutie. Elk van zijn filters ziet dan alle `channels_in` channels, met 9 × `channels_in` weights in plaats van 9. Geef elk channel een eigen filter: `groups=channels_in`."),
            Mistake("kernel", wrong=_S_WRONG["kernel"],
                    en=_S_EN + ": your second convolution is a 3×3 one. It looks at the neighbors again, which the depthwise convolution already did, and has 9 times as many weights. The second step only mixes the channels: kernel size 1.",
                    nl=_S_NL + ": je tweede convolutie is een 3×3-convolutie. Hij kijkt opnieuw naar de buren, wat de depthwise convolution al deed, en heeft 9 keer zoveel weights. De tweede stap mengt alleen de channels: kernel size 1."),
            Mistake("swapped", matches=_swapped,
                    en=_S_EN + ": your `self.depthwise` is a 1×1 convolution and your `self.pointwise` a 3×3 one. The depthwise convolution comes first, with a 3×3 filter for each channel and `groups=channels_in`; the 1×1 convolution after it mixes the channels, from `channels_in` to `channels_out`.",
                    nl=_S_NL + ": je `self.depthwise` is een 1×1-convolutie en je `self.pointwise` een 3×3-convolutie. De depthwise convolution komt eerst, met een 3×3-filter voor elk channel en `groups=channels_in`; de 1×1-convolutie erna mengt de channels, van `channels_in` naar `channels_out`."),
            Mistake("pointwise-groups", matches=_pointwise_groups,
                    en=_S_EN + ": your 1×1 convolution has `groups` too. Then each of its filters sees only part of the channels, and with `groups=channels_in` only one: it mixes nothing. The 1×1 convolution is the step that mixes all the channels: leave `groups` out there.",
                    nl=_S_NL + ": je 1×1-convolutie heeft ook `groups`. Dan ziet elk van zijn filters maar een deel van de channels, en met `groups=channels_in` maar één: hij mengt niets. De 1×1-convolutie is de stap die alle channels mengt: laat `groups` daar weg."),
            Mistake("pointwise-out", wrong=_s(pw="nn.Conv2d(channels_in, channels_in, 1)"),
                    en=_S_EN + ": your 1×1 convolution makes `channels_in` maps. With 64 channels in and out you cannot see the difference, but from 3 channels to 5 it must make 5: the 1×1 convolution goes from `channels_in` to `channels_out`.",
                    nl=_S_NL + ": je 1×1-convolutie maakt `channels_in` maps. Met 64 channels erin en eruit zie je het verschil niet, maar van 3 channels naar 5 moet hij er 5 maken: de 1×1-convolutie gaat van `channels_in` naar `channels_out`."),
            Mistake("padding", matches=_other_size,
                    en=_S_EN + ": your block gives back maps of another size than it gets, 8 by 8. A 3×3 convolution keeps the height and width with `padding=1`, and a 1×1 convolution without padding.",
                    nl=_S_NL + ": je block geeft maps van een andere grootte terug dan het krijgt, 8 bij 8. Een 3×3-convolutie houdt de hoogte en breedte met `padding=1`, en een 1×1-convolutie zonder padding."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...` with a convolution: the depthwise one, and the 1×1 one.",
                      "nl": "Vervang de twee `...` door een convolutie: de depthwise, en de 1×1."},
            "wrong": {"en": _S_EN + ". Does the depthwise convolution give every channel a 3×3 filter of its own, with `padding=1` and `groups=channels_in`, and does the 1×1 convolution go from `channels_in` to `channels_out`?",
                      "nl": _S_NL + ". Geeft de depthwise convolution elk channel een eigen 3×3-filter, met `padding=1` en `groups=channels_in`, en gaat de 1×1-convolutie van `channels_in` naar `channels_out`?"},
            "almost": {"en": "With 64 channels in and out your block is right, but " + _S_EN[0].lower() + _S_EN[1:] + ". Did you change the lines that were in the box? `forward` sends `x` through the depthwise convolution and a ReLU, then through the 1×1 convolution and a ReLU.",
                       "nl": "Met 64 channels erin en eruit klopt je block, maar " + _S_NL[0].lower() + _S_NL[1:] + ". Heb je de regels veranderd die in het vak stonden? `forward` stuurt `x` door de depthwise convolution en een ReLU, en dan door de 1×1-convolutie en een ReLU."},
            "correct": {"en": f"Your `SeparableConv` is right: with 64 channels in and out, it has {grouped(separable_params(64, 64), 'en')} parameters, where a 3×3 convolution has {grouped(conv_params(64, 64, 3), 'en')}. It is now in your project, as `{{saved}}`.",
                        "nl": f"Je `SeparableConv` klopt: met 64 channels erin en eruit heeft hij {grouped(separable_params(64, 64), 'nl')} parameters, waar een 3×3-convolutie er {grouped(conv_params(64, 64, 3), 'nl')} heeft. Hij staat nu in je project, als `{{saved}}`."},
            "kept": {"en": "Your `SeparableConv` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `SeparableConv` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("steps", {}, "empty", None),
    ("steps", _SORT.solution, "correct", None),
    ("steps", {"neighbors": "depthwise", "mixes": "pointwise"}, "wrong", "unfinished"),
    ("steps", {**_SORT.solution, "work": "depthwise"}, "wrong", "wrong:work"),
    ("steps", {**_SORT.solution, "groups": "pointwise"}, "wrong", "wrong:groups"),
    ("separable", _S_START, "empty", None),
    ("separable", _S, "correct", None),
    # other ways to write the same: padding "same", arguments by name, the two lines the other way round
    ("separable", _s(dw="nn.Conv2d(channels_in, channels_in, 3, padding=\"same\", groups=channels_in)"), "correct", None),
    ("separable", _s(dw="nn.Conv2d(in_channels=channels_in, out_channels=channels_in, kernel_size=3, padding=1, groups=channels_in)",
                     pw="nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=1)"), "correct", None),
    ("separable", _S_HEAD + f"        self.pointwise = {_PW}\n        self.depthwise = {_DW}\n" + _S_FORWARD, "correct", None),
    ("separable", _s(dw="nn.Conv2d(channels_in, channels_in, 3, padding=1, groups=64)"), "wrong", "fixed"),
    ("separable", _s(pw="nn.Conv2d(channels_in, 64, 1)"), "wrong", "fixed"),
    ("separable", _S_START.replace("self.depthwise = ...", f"self.depthwise = {_DW}"), "wrong", "dots-left"),
    ("separable", _s(pw="nn.Conv2d(channels_in, channels_out, 1, groups=channels_in)"), "wrong", "pointwise-groups"),
    ("separable", _s(dw="nn.Conv2d(channels_in, channels_in, 3, padding=1, groups=channels_out)"), "error", "groups"),
    ("separable", _s(dw="nn.Conv2d(channels_in, channels_in, 3, padding=1, groups=3)"), "error", "groups"),
    ("separable", _s(dw="nn.Conv2d(channels_in, channels_out, 3, padding=1, groups=channels_in)"), "error", "depthwise-out"),
    ("separable", _s(pw="nn.Conv2d(channels_out, channels_out, 1)"), "error", "channels"),
    ("separable", _s(dw="nn.Conv2d(channels_in, channels_in, 3, padding=1, groups=1)"), "wrong", "no-groups"),
    ("separable", _s(dw="nn.Conv2d(channels_in, channels_in, 3, groups=channels_in)"), "wrong", "padding"),
    ("separable", _s(pw="nn.Conv2d(channels_in, channels_out, 1, padding=1)"), "wrong", "padding"),
    # a number in a comment is no fixed number
    ("separable", _s(dw="nn.Conv2d(channels_in, channels_in, 3, padding=1, groups=channels_in)  # one filter per channel (64 here)"), "correct", None),
    # no groups, and the depthwise convolution already to channels_out
    ("separable", _s(dw="nn.Conv2d(channels_in, channels_out, 3, padding=1)", pw="nn.Conv2d(channels_out, channels_out, 1)"), "wrong", "no-groups"),
    # the two steps the other way round
    ("separable", _s(dw="nn.Conv2d(channels_in, channels_out, 1)", pw="nn.Conv2d(channels_out, channels_out, 3, padding=1, groups=channels_out)"), "wrong", "swapped"),
    ("separable", _s(dw="nn.Conv2d(channels_in, channels_in, 1)", pw="nn.Conv2d(channels_in, channels_out, 3, padding=1, groups=channels_in)"), "wrong", "swapped"),
    # the 1×1 convolution to channels_in: right with 64 in and out, not from 3 channels to 5
    ("separable", _s(pw="nn.Conv2d(channels_in, channels_in, 1)"), "wrong", "pointwise-out"),
    # the right shapes and parameters, but other numbers that no known mistake explains
    ("separable", _S.replace("return torch.relu(self.pointwise(x))", "return torch.sigmoid(self.pointwise(x))"), "almost", None),
] + [("separable", code, "wrong", key) for key, code in _S_WRONG.items()]


# --- the paper ------------------------------------------------------------------------------------------------------

# Howard et al. (2017). Table 3: one layer of MobileNet with 512 channels in and out on maps of 14 by 14, as a 3×3
# convolution and as a depthwise separable convolution: million multiply-adds and million weights. Table 4: MobileNet
# with full 3×3 convolutions against MobileNet: ImageNet accuracy in %, million multiply-adds and million parameters.
# Table 8: VGG 16 and GoogleNet. Table 2: the share of MobileNet's multiply-adds and parameters in its 1×1
# convolutions, in %.
PAPER_LAYER = {"full": (462, 2.36), "separable": (52.3, 0.27)}
PAPER_NETS = {"conv-mobilenet": (71.7, 4866, 29.3), "mobilenet": (70.6, 569, 4.2), "vgg16": (71.5, 15300, 138),
              "googlenet": (69.8, 1550, 6.8)}
PAPER_POINTWISE = (94.86, 74.59)
# The worked example's 3×3 convolution and depthwise separable convolution from 64 channels to 64 and to 512, on one
# photo's maps of 8 by 8: their multiplications, which are their weights times the 64 pixels they give.
EXAMPLE = {n: (full_weights(CHANNELS, n) * SIDE * SIDE, separable_weights(CHANNELS, n) * SIDE * SIDE) for n in (64, 512)}


# --- what the course measured ---------------------------------------------------------------------------------------

# What the course measured, on one thread, with the course's code, through run(): the middle of each variant
# (make_middle, with the course's SeparableConv) made after torch.manual_seed(seed), then the rest of WideCNN, trained as
# in cv-1x1-convolution (the Trainer of nn-trainer with AdamW, a learning rate of LR, a warmup of one epoch, cosine
# decay, max_norm 1.0, mini-batches of 64) on the first N_TRAIN training photos, shuffled after
# torch.manual_seed(seed + 1), for EPOCHS epochs and the seeds 0 to 4, with the photos normalized by channel_stats of
# the training photos. "train" and "test" are how many of the training photos and of the first N_TEST test photos are
# read right after the last epoch; "best" is the first epoch with the lowest validation loss, rounded to 3 decimals,
# and "lowest" that loss; "last_val" and "last_train" are the validation and the training loss of the last epoch.
# CURVES are seed 0. The runs with one 3×3 convolution in the middle ("plain") are those of cv-1x1-convolution, which
# builds them the same way. tests/test_cv_mobilenet.py measures seed 0 of every middle again when EAI_CIFAR_FILE names
# the unpacked cifar-10.npz.
REFERENCE = {
    "plain": _ONE.REFERENCE["plain"],
    "separable": {"train": [5880, 5899, 5919, 5982, 5829], "test": [548, 550, 553, 581, 552], "best": [6, 5, 8, 8, 7],
                  "lowest": [1.297, 1.322, 1.347, 1.35, 1.339], "last_val": [2.14, 2.071, 2.217, 2.3, 2.035],
                  "last_train": [0.125, 0.109, 0.092, 0.042, 0.145]},
    "two": {"train": [5735, 5535, 5436, 5569, 5614], "test": [534, 541, 530, 527, 536], "best": [8, 9, 10, 9, 10],
            "lowest": [1.401, 1.386, 1.36, 1.38, 1.349], "last_val": [1.945, 1.878, 1.74, 1.848, 1.846],
            "last_train": [0.199, 0.304, 0.346, 0.278, 0.246]},
    "depthwise": {"train": [5807, 5810, 5854, 5840, 5741], "test": [563, 557, 573, 585, 578], "best": [6, 4, 4, 7, 6],
                  "lowest": [1.338, 1.327, 1.362, 1.374, 1.284], "last_val": [2.025, 1.979, 1.993, 2.12, 1.919],
                  "last_train": [0.152, 0.153, 0.137, 0.137, 0.187]},
    "pointwise": {"train": [5952, 5951, 5924, 5968, 5943], "test": [574, 557, 565, 576, 571], "best": [7, 6, 6, 6, 6],
                  "lowest": [1.297, 1.344, 1.357, 1.287, 1.291], "last_val": [2.056, 2.127, 2.064, 1.994, 2.074],
                  "last_train": [0.08, 0.084, 0.105, 0.057, 0.088]},
}
CURVES = {
    "separable_val": [1.776, 1.501, 1.448, 1.347, 1.347, 1.297, 1.353, 1.393, 1.356, 1.401, 1.507, 1.597, 1.721,
                      1.768, 1.881, 1.99, 2.054, 2.107, 2.134, 2.14],
    "separable_train": [1.979, 1.609, 1.424, 1.334, 1.249, 1.135, 1.075, 0.977, 0.884, 0.799, 0.689, 0.588, 0.488,
                        0.39, 0.311, 0.246, 0.189, 0.154, 0.134, 0.125],
    "two_val": [1.734, 1.54, 1.464, 1.418, 1.481, 1.407, 1.404, 1.401, 1.44, 1.446, 1.475, 1.487, 1.558, 1.666,
                1.741, 1.82, 1.869, 1.927, 1.941, 1.945],
    "two_train": [2.056, 1.656, 1.501, 1.398, 1.316, 1.257, 1.169, 1.101, 1.016, 0.939, 0.833, 0.733, 0.619, 0.516,
                  0.428, 0.35, 0.281, 0.239, 0.211, 0.199],
    "depthwise_val": [1.607, 1.446, 1.456, 1.359, 1.361, 1.338, 1.413, 1.442, 1.562, 1.452, 1.556, 1.577, 1.679,
                      1.807, 1.849, 1.919, 1.977, 2.015, 2.023, 2.025],
    "depthwise_train": [1.926, 1.521, 1.358, 1.239, 1.12, 1.032, 0.958, 0.856, 0.784, 0.698, 0.602, 0.525, 0.443,
                        0.361, 0.301, 0.249, 0.204, 0.177, 0.16, 0.152],
    "pointwise_val": [1.586, 1.55, 1.413, 1.344, 1.346, 1.361, 1.297, 1.342, 1.374, 1.387, 1.428, 1.534, 1.677,
                      1.763, 1.816, 1.935, 1.997, 2.035, 2.053, 2.056],
    "pointwise_train": [1.929, 1.573, 1.408, 1.282, 1.167, 1.067, 1.005, 0.908, 0.805, 0.709, 0.619, 0.503, 0.405,
                        0.314, 0.228, 0.175, 0.129, 0.102, 0.087, 0.08],
}
DEFAULT = "separable"   # the demo's middle when the learner chooses none: the block of exercise 2
RANGE = (500, 600)      # the axis of the picture of the runs, which every run of the course fits on


def mean(values) -> float:
    return sum(values) / len(values)


# --- the picture of the runs ----------------------------------------------------------------------------------------

ROWS = ("plain", "separable", "two", "depthwise", "pointwise")


def runs_picture(labels: dict, lang: str, mine: dict | None = None) -> str:
    """How many of the N_TEST test photos the network reads right after EPOCHS epochs, for each middle: one 3×3
    convolution of cv-1x1-convolution, the depthwise separable convolution, two of them, and the depthwise or the 1×1
    convolution alone; an open dot for every seed of the course and a line at their mean. With `mine` ({"variant",
    "test"}), the learner's run as a filled dot in the row of its middle. `labels` gives "title", "x", a name for each
    row, "mine", "course" and "alt"."""
    yours = {} if mine is None else {mine["variant"]: mine["test"]}
    # from RANGE, wider where the learner's run lies outside, so that every dot is where its number is
    low, high = min([RANGE[0]] + [v - 15 for v in yours.values()]), max([RANGE[1]] + [v + 15 for v in yours.values()])
    tick = 20 if high - low <= 200 else 50 if high - low <= 400 else 100
    low, high = int(low // tick * tick), int(-(-high // tick) * tick)
    p = chart.Panel((low, high), (-1.0, len(ROWS) - 0.3), title=labels["title"], xlabel=labels["x"], yticks=[],
                    xticks=list(range(low, high + 1, tick)))
    step = (high - low) / 220
    for k, key in enumerate(ROWS):
        y = len(ROWS) - 1 - k
        values = REFERENCE[key]["test"]
        m = mean(values)
        p.line([m, m], [y - 0.18, y + 0.18], "ink-3", 1.6)
        p.points(values, [y] * len(values), "ink", r=3.5, hollow=True)
        p.text(low + step, y + 0.5, labels[key], "ink-2", "start")   # clear of the line at the mean
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
    return _ONE.rewind_frame(curves_picture, result, e, labels, title, lang, "mobilenet")


# The lesson is done when these are correct. "separable-runs" is the choice question in the lesson.
REQUIRED = ["steps", "separable", "separable-runs"]


# --- the demo: the middle of WideCNN, on CIFAR-10 -------------------------------------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def _from_block(e: BaseException) -> bool:
    """The error came from the learner's SeparableConv: a line of their separable.py is in its traceback."""
    mine = (projects.folder() / "cnn" / "separable.py").resolve()
    tb = e.__traceback__
    while tb is not None:
        if Path(tb.tb_frame.f_code.co_filename).resolve() == mine:
            return True
        tb = tb.tb_next
    return False


def _demo(variant: str, show=None):
    """For the demo: None when separable.py is not in the project yet and the variant needs it (the depthwise or the
    1×1 convolution alone does not); "data" when CIFAR-10 is not unpacked; "separable" when the learner's SeparableConv
    stops with an error, also during the training, or does not give back 64 maps of 8 by 8; "cnn" when channel_stats
    does not give what it should; "trainer" when the Trainer or count_params stops with an error; "slow" when the
    training takes too long; else the dict of run(), with the modules the course's versions stood in for."""
    theirs = variant in ("separable", "two")
    if theirs and not _saved("separable"):
        return None
    if not ready():
        return "data"
    SeparableConv = None
    if theirs:
        loaded = run_code("from cnn.separable import SeparableConv")
        if loaded.error is not None:
            return "separable"
        SeparableConv = loaded.names["SeparableConv"]
        try:
            def tried():   # as in training: a mini-batch, in training mode, with a backward pass
                torch.manual_seed(0)
                block = SeparableConv(CHANNELS, CHANNELS)
                out = block(torch.rand(4, CHANNELS, SIDE, SIDE))
                out.sum().backward()
                return tuple(out.shape)
            if with_time_limit(tried, 10) != (4, CHANNELS, SIDE, SIDE):
                return "separable"
        except Exception:
            return "separable"
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
        result = with_time_limit(lambda: run(variant, 0, SeparableConv, channel_stats, Trainer, count_params, show), 600)
    except Timeout:
        return "slow"
    except Exception as e:
        return "separable" if theirs and _from_block(e) else "trainer"
    return result | {"stand_ins": stand_ins}


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file), and whether CIFAR-10
    is there."""
    texts = []
    for package, module in (("cnn", "separable"), ("cnn", "channels"), ("cnn", "count"), ("nn", "train_epoch"), ("nn", "trainer")):
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
