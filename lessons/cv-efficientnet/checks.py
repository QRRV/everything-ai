# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-efficientnet: a CNN can grow in depth (more blocks), in width (more channels) or in the
resolution of its photos (more pixels); a convolution's multiplications grow with the depth, the square of the width
and the square of the resolution, and EfficientNet grows all three together in a fixed ratio, compound scaling. The
learner's ScaledCNN, a 3×3 convolution with stride 2 followed by `depth` of their depthwise separable convolutions of
cv-mobilenet with residual connections, the mean of each map and a linear layer, is saved in their project
(work/projects/cnn/scaled.py) when right, and checked on made-up numbers, so it works without CIFAR-10. The demo trains
it on photos of CIFAR-10 made smaller by the course, as a base network and grown in four ways to about four times its
multiplications: deeper, wider, sharper, or all three together. It trains with the learner's Trainer on the first 6,000
training photos, tests on the first 1,000 test photos, and keeps the network's guess for one checked test photo of each
kind after every epoch, for rewinding."""
import html
import importlib.util
import math
import re
from pathlib import Path

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import chart, draw, live, paths, projects
from everything_ai.check import Mistake, ScenarioBox, Sort, Timeout, run_code, with_time_limit
from everything_ai import netview, yours


def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_efficientnet",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MOB = _lesson_checks("cv-mobilenet")   # the depthwise separable convolution, and through it CIFAR-10 and the rewind
_ONE = _MOB._ONE

# The demo uses the learner's SeparableConv of cv-mobilenet, their channel_stats of cv-color-cnn, their Trainer of
# nn-trainer and their count_params of cnn-pytorch; where one is missing or broken, the course's version stands in
# (template 1.3).
BUILDS_ON = _MOB.BUILDS_ON + (("cnn", "separable", "cv-mobilenet", "separable"),)
projects.builds_on(*BUILDS_ON)
DOWNLOAD_LINE = _ONE.DOWNLOAD_LINE

cifar_or_stop, ready, photos = _ONE.cifar_or_stop, _ONE.ready, _ONE.photos
grouped, number, percent, low_high = _ONE.grouped, _ONE.number, _ONE.percent, _ONE.low_high
guesses_picture, moments, best_epoch = _ONE.guesses_picture, _ONE.moments, _ONE.best_epoch
SHOWN, N_TRAIN, N_TEST, EPOCHS, LR = _ONE.SHOWN, _ONE.N_TRAIN, _ONE.N_TEST, _ONE.EPOCHS, _ONE.LR
EVERY = _ONE.EVERY   # the live chart hears the training loss of every EVERY batches
conv_params = _ONE.conv_params


# --- counting -------------------------------------------------------------------------------------------------------

def multiplications(depth: int, width: int, side: int) -> int:
    """The multiplications of ScaledCNN(depth, width) for one photo of side by side pixels (an even side). The first
    convolution, 3×3 from 3 colors to `width` channels with stride 2, gives maps of side // 2 by side // 2, and needs
    3 · 3 · 3 for each number it gives; each block, a depthwise 3×3 convolution and a 1×1 convolution from `width`
    channels to `width`, needs 9 · width + width · width for each pixel of its maps; the linear layer at the end,
    from `width` means to 10 logits, width · 10. The means themselves are additions."""
    half = side // 2
    stem = 3 * 3 * 3 * width * half * half
    block = (3 * 3 * width + width * width) * half * half
    return stem + depth * block + width * 10


def parameters(depth: int, width: int) -> int:
    """The parameters of ScaledCNN(depth, width): the same for photos of every size, because the same filters slide
    over more pixels."""
    stem = conv_params(3, width, 3)
    block = width * (3 * 3 + 1) + conv_params(width, width)
    return stem + depth * block + width * 10 + 10


def rule(depth_times: float, width_times: float, side_times: float) -> float:
    """How many times as many multiplications by the paper's rule: the multiplications of a convolution grow with the
    depth, the square of the width and the square of the resolution."""
    return depth_times * width_times ** 2 * side_times ** 2


# The demo's base network, and four ways to grow it to about four times its multiplications, counted exactly. By the
# rule, 4 times as many blocks would be 8 and twice the channels 128; counted, those cost only 3.53 and 3.48 times as
# much, because the first convolution does not grow with the depth, and the first convolution and the depthwise
# convolutions grow only with the width, not its square.
# The compound network takes the paper's α, β and γ with φ = 2: depth 2 · 1.2² = 2.88, rounded up to 3 blocks; width
# 64 · 1.1² = 77.4, to a multiple of 8, 80; side 16 · 1.15² = 21.2, to an even number, 22.
VARIANTS = {"base": (2, 64, 16), "deeper": (9, 64, 16), "wider": (2, 136, 16), "sharper": (2, 64, 32),
            "compound": (3, 80, 22)}
BASE = VARIANTS["base"]
ALPHA, BETA, GAMMA = 1.2, 1.1, 1.15   # EfficientNet-B0's, from the paper's small grid search with φ = 1
PHIS = list(range(6))                  # φ on the slider: at 5 the photos are as large as CIFAR-10's, 32 by 32


def grown(phi: int) -> tuple[int, int, int]:
    """The base network grown with compound scaling: depth · α^φ rounded up to whole blocks, width · β^φ to a multiple of
    8, side · γ^φ to an even number."""
    depth = math.ceil(round(BASE[0] * ALPHA ** phi, 9))
    width = 8 * round(BASE[1] * BETA ** phi / 8)
    side = 2 * round(BASE[2] * GAMMA ** phi / 2)
    return depth, width, side


def times(variant_or_net) -> float:
    """How many times the base network's multiplications a variant, or a (depth, width, side), needs."""
    net = VARIANTS[variant_or_net] if isinstance(variant_or_net, str) else variant_or_net
    return multiplications(*net) / multiplications(*BASE)


# --- the network, as the course writes it ---------------------------------------------------------------------------

_N_HEAD = ("import torch\nfrom torch import nn\n\nfrom cnn.separable import SeparableConv\n\n\nclass ScaledCNN(nn.Module):\n"
           "    def __init__(self, depth, width):\n        super().__init__()\n"
           "        self.stem = nn.Conv2d(3, width, 3, stride=2, padding=1)\n")
_N_MIDDLE = ("        self.fc = nn.Linear(width, 10)\n\n    def forward(self, x):\n"
             "        x = torch.relu(self.stem(x))\n")
_N_END = "        return self.fc(x.mean(dim=(2, 3)))\n"
_BLOCKS = "nn.ModuleList([SeparableConv(width, width) for _ in range(depth)])"
_LOOP = "        for block in self.blocks:\n            x = x + block(x)\n"


def _n(blocks: str = _BLOCKS, loop: str = _LOOP) -> str:
    return _N_HEAD + f"        self.blocks = {blocks}\n" + _N_MIDDLE + loop + _N_END


_N = _n()


# --- the photos, made smaller ---------------------------------------------------------------------------------------

def smaller(X: torch.Tensor, side: int) -> torch.Tensor:
    """Photos of 32 by 32 made side by side, with bilinear interpolation that mixes every old pixel under a new one and
    those around it, with less weight the farther they lie: what Pillow does when a photo gets much smaller."""
    if side == X.shape[-1]:
        return X
    return F.interpolate(X, size=(side, side), mode="bilinear", antialias=True, align_corners=False)


def photos_at(side: int):
    """photos() made side by side."""
    X, y, X_test, y_test = photos()
    return smaller(X, side), y, smaller(X_test, side), y_test


# --- the demo: ScaledCNN on CIFAR-10 --------------------------------------------------------------------------------

def _accuracy(model, X, y) -> int:
    model.eval()
    with torch.no_grad():
        return int(sum((model(X[i:i + 500]).argmax(dim=1) == y[i:i + 500]).sum() for i in range(0, len(X), 500)))


def run(variant: str, seed: int, ScaledCNN, channel_stats, Trainer, count_params, show=None):
    """One run of ScaledCNN as VARIANTS says for `variant`, as REFERENCE says: the photos made side by side (photos_at),
    normalized with channel_stats of the training photos, the network made after torch.manual_seed(seed), and the photos
    shuffled after torch.manual_seed(seed + 1). Gives back a dict with the parameters of the network, its
    multiplications for one photo, how many of the training and test photos it reads right, the loss after every epoch
    ("curves"), the epoch with the lowest validation loss, and the network's guess for the photos SHOWN after every
    epoch ("guesses") with their right kinds ("truth"), and its first layer of feature maps on the car ("layers").
    Errors go to the caller, which says whose code made them. `show` hears the run while it trains (live.follow, every
    10 batches), with what the network makes then."""
    depth, width, side = VARIANTS[variant]
    X, y, X_test, y_test = photos_at(side)
    mean_, std = channel_stats(X.clone())
    mean_ = torch.as_tensor(mean_, dtype=torch.float32).reshape(3)
    std = torch.as_tensor(std, dtype=torch.float32).reshape(3)
    X = (X - mean_[:, None, None]) / std[:, None, None]
    X_test = (X_test - mean_[:, None, None]) / std[:, None, None]
    torch.manual_seed(seed)
    model = ScaledCNN(depth, width)
    params = int(count_params(model))
    truth, guesses, layers = [int(y_test[place]) for place in SHOWN], [], []   # what the rewind keeps of every epoch
    heard = _ONE.hearing(model, X_test[list(SHOWN)], truth, X_test[_ONE.PHOTO:_ONE.PHOTO + 1], show, guesses, layers)

    torch.manual_seed(seed + 1)
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    val = DataLoader(TensorDataset(X_test, y_test), batch_size=500)
    trainer = Trainer(model, loader, val, epochs=EPOCHS, lr=LR, warmup=len(loader), max_norm=1.0)
    live.follow(trainer, heard, "efficientnet", every=EVERY)
    history = trainer.fit() or []
    curves = {"train": [float(h["train_loss"]) for h in history if isinstance(h, dict) and "train_loss" in h],
              "val": [float(h["val_loss"]) for h in history if isinstance(h, dict) and "val_loss" in h]}
    return {"variant": variant, "params": params, "mults": multiplications(depth, width, side),
            "train": _accuracy(model, X, y), "test": _accuracy(model, X_test, y_test), "curves": curves,
            "best": best_epoch(curves["val"]), "guesses": guesses[:len(curves["val"])], "layers": layers[:len(curves["val"])],
            "truth": truth}


# --- the picture of the five networks -------------------------------------------------------------------------------

ORDER = ("base", "deeper", "wider", "sharper", "compound")


def networks_picture(labels: dict, lang: str):
    """The demo's five networks side by side, as in the paper's figure 2: at the bottom of each, the photo it gets, a
    gray square as large as its pixels; above it a bar for each block, as wide as its channels, in the color of what a
    model learns. Under each, its name, its blocks, its channels, its photos, and how many times the base network's
    multiplications it needs. `labels` gives a name for each of ORDER, "blocks" ({d}), "channels" ({w}), "photo" ({s}),
    "times" ({t}) and "alt"."""
    COL, GAP, PX, BAR, STEP, WIDE = 116, 10, 2.0, 9, 13, 0.6
    tallest = max(d for d, _, _ in VARIANTS.values()) * STEP
    biggest = max(s for _, _, s in VARIANTS.values()) * PX
    BOTTOM = 6 + tallest + 8 + biggest   # where the photos stand
    small = "font-size:13px;fill:var(--ea-ink-2);text-anchor:middle"
    parts = []
    for k, key in enumerate(ORDER):
        d, w, s = VARIANTS[key]
        mid = k * (COL + GAP) + COL / 2
        side = s * PX
        parts.append(f'<rect x="{mid - side / 2:.1f}" y="{BOTTOM - side:.1f}" width="{side:.1f}" height="{side:.1f}" rx="2" '
                     'style="fill:var(--ea-data-wash);stroke:var(--ea-data);stroke-width:1.2"/>')
        top = BOTTOM - side - 8
        for b in range(d):
            parts.append(f'<rect x="{mid - w * WIDE / 2:.1f}" y="{top - (b + 1) * STEP + (STEP - BAR):.1f}" width="{w * WIDE:.1f}" '
                         f'height="{BAR}" rx="1.5" style="fill:var(--ea-param-wash);stroke:var(--ea-param);stroke-width:1.2"/>')
        y = BOTTOM + 22
        parts.append(f'<text x="{mid}" y="{y}" style="font-size:14px;font-weight:600;fill:var(--ea-ink);text-anchor:middle">'
                     f'{html.escape(labels[key])}</text>')
        lines = (labels["blocks"].format(d=d), labels["channels"].format(w=w), labels["photo"].format(s=s),
                 labels["times"].format(t=number(times(key), 2, lang)))
        for n, line in enumerate(lines):
            parts.append(f'<text x="{mid}" y="{y + 18 * (n + 1)}" style="{small}">{html.escape(line)}</text>')
    W = len(ORDER) * (COL + GAP) - GAP
    return draw._svg(W, int(BOTTOM + 22 + 18 * 4 + 8), labels["alt"], "".join(parts), natural=True)


# --- the slider: the base network grown with compound scaling -------------------------------------------------------

PER_STEP = ALPHA * BETA ** 2 * GAMMA ** 2   # 1.92: what each step of φ multiplies the multiplications by, by the rule


def family_picture(phi: int, labels: dict, lang: str) -> str:
    """The base network grown with compound scaling, for every φ on the slider: a bar of its multiplications as a
    multiple of the base network's, counted exactly, dark at the chosen φ; and what the rule says, PER_STEP to the power
    φ, as a dashed line with open dots. On a log scale, where the rule is a straight line and the gap between a bar and
    its dot shows at every φ. The axis stays the same for every φ. `labels` gives "title", "x", "y", "rule" and
    "alt"."""
    said = [PER_STEP ** f for f in PHIS]
    counted = [times(grown(f)) for f in PHIS]
    low, top = 0.7, 1.4 * max(said + counted)
    p = chart.Panel((-0.6, PHIS[-1] + 0.6), (low, top), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=PHIS, yticks=[1, 2, 5, 10, 20], ylog=True)
    for f, t in zip(PHIS, counted):
        p.rect(f - 0.32, low, f + 0.32, t, "ink-3")
        if f == phi:
            for _ in range(2):   # an ink wash twice over the gray one: dark against the other bars
                p.rect(f - 0.32, low, f + 0.32, t, "ink")
    p.line(PHIS, said, "ink-2", 1.4, dashed=True)
    p.points(PHIS, said, "ink-2", r=3.5, hollow=True)
    p.text(-0.4, PER_STEP ** 3.2, labels["rule"], "ink-2", "start")   # above the line, clear of the bars
    svg = chart.figure([p], alt=labels["alt"], width=440, height=260)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def family_sentence(phi: int, lang: str) -> str:
    """The sentence under the slider, for `phi`."""
    d, w, s = grown(phi)
    m = grouped(multiplications(d, w, s), lang)
    t, r = number(times((d, w, s)), 2, lang), number(PER_STEP ** phi, 2, lang)
    if lang == "nl":
        return (f"Met φ = {phi}: {d} blocks, {w} channels en foto's van {s} bij {s} pixels: {m} vermenigvuldigingen per "
                f"foto, {t} keer zoveel als het basisnetwerk. De regel zegt {r} keer.")
    return (f"With φ = {phi}: {d} blocks, {w} channels and photos of {s} by {s} pixels: {m} multiplications for a photo, "
            f"{t} times the base network's. The rule says {r} times.")


# --- exercise 1: what each choice costs -----------------------------------------------------------------------------

_SORT = Sort(
    name="costs",
    items={
        "depth2": {"en": "**Depth × 2.** Twice as many blocks.",
                   "nl": "**Diepte × 2.** Twee keer zoveel blocks."},
        "width2": {"en": "**Width × 2.** Twice as many channels in every layer.",
                   "nl": "**Breedte × 2.** Twee keer zoveel channels in elke layer."},
        "side2": {"en": "**Resolution × 2.** Photos twice as wide and twice as high.",
                  "nl": "**Resolutie × 2.** Foto's twee keer zo breed en twee keer zo hoog."},
        "depth4": {"en": "**Depth × 4.** Four times as many blocks.",
                   "nl": "**Diepte × 4.** Vier keer zoveel blocks."},
        "width141": {"en": "**Width × 1.41.** 1.41 times as many channels in every layer, the square root of 2.",
                     "nl": "**Breedte × 1,41.** 1,41 keer zoveel channels in elke layer, de wortel uit 2."},
        "side141": {"en": "**Resolution × 1.41.** Photos 1.41 times as wide and 1.41 times as high.",
                    "nl": "**Resolutie × 1,41.** Foto's 1,41 keer zo breed en 1,41 keer zo hoog."},
        "phi1": {"en": "**φ one higher.** With EfficientNet's α = 1.2, β = 1.1 and γ = 1.15.",
                 "nl": "**φ één hoger.** Met de α = 1,2, β = 1,1 en γ = 1,15 van EfficientNet."},
        "phi2": {"en": "**φ two higher.** With the same α, β and γ.",
                 "nl": "**φ twee hoger.** Met dezelfde α, β en γ."},
    },
    groups={"two": {"en": "about 2 times as many multiplications", "nl": "ongeveer 2 keer zoveel vermenigvuldigingen"},
            "four": {"en": "about 4 times as many", "nl": "ongeveer 4 keer zoveel"}},
    solution={"depth2": "two", "width2": "four", "side2": "four", "depth4": "four", "width141": "two", "side141": "two",
              "phi1": "two", "phi2": "four"},
    nudge={
        "depth2": {"en": "Each block costs the same: what do twice as many blocks cost?",
                   "nl": "Elk block kost evenveel: wat kosten twee keer zoveel blocks?"},
        "width2": {"en": "With twice the channels, a filter sees twice as many channels, and there are twice as many filters.",
                   "nl": "Met twee keer zoveel channels ziet een filter twee keer zoveel channels, en zijn er twee keer zoveel filters."},
        "side2": {"en": "Twice as wide and twice as high: how many times as many pixels is that?",
                  "nl": "Twee keer zo breed en twee keer zo hoog: hoeveel keer zoveel pixels is dat?"},
        "depth4": {"en": "The multiplications grow as fast as the blocks.",
                   "nl": "De vermenigvuldigingen groeien even snel als de blocks."},
        "width141": {"en": "The width counts squared: 1.41 × 1.41 is about what?",
                     "nl": "De breedte telt in het kwadraat: 1,41 × 1,41 is ongeveer wat?"},
        "side141": {"en": "The pixels grow with the width of the photo times its height: 1.41 × 1.41.",
                    "nl": "De pixels groeien met de breedte van de foto keer zijn hoogte: 1,41 × 1,41."},
        "phi1": {"en": "Each step of φ multiplies the cost by α · β² · γ²: 1.2 × 1.21 × 1.32.",
                 "nl": "Elke stap van φ vermenigvuldigt de kosten met α · β² · γ²: 1,2 × 1,21 × 1,32."},
        "phi2": {"en": "Two steps of φ: twice a factor of about 2.",
                 "nl": "Twee stappen van φ: twee keer een factor van ongeveer 2."},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the eight changes again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de acht veranderingen opnieuw uit."},
        "empty": {"en": "Choose for each change: about 2 or about 4 times as many multiplications.",
                  "nl": "Kies bij elke verandering: ongeveer 2 of ongeveer 4 keer zoveel vermenigvuldigingen."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "{item} belongs to the other group.", "nl": "{item} hoort bij de andere groep."},
        "correct": {"en": "All {n} are right. The multiplications grow with the depth, with the square of the width and with the square of the resolution: twice the blocks cost 2 times as much, twice the channels or twice the pixels on each side 4 times. Width and resolution therefore need only 1.41 times, the square root of 2, to cost twice as much. And each step of φ multiplies the cost by α · β² · γ² = 1.92, about 2.",
                    "nl": "Alle {n} kloppen. De vermenigvuldigingen groeien met de diepte, met het kwadraat van de breedte en met het kwadraat van de resolutie: twee keer zoveel blocks kosten 2 keer zoveel, twee keer zoveel channels of twee keer zoveel pixels langs elke kant 4 keer. Breedte en resolutie hebben dus maar 1,41 keer nodig, de wortel uit 2, om twee keer zoveel te kosten. En elke stap van φ vermenigvuldigt de kosten met α · β² · γ² = 1,92, ongeveer 2."},
    },
)


# --- exercise 2: a network that grows ------------------------------------------------------------------------------

_N_START = (_N_HEAD + "        # depth blocks, each a SeparableConv from width channels to width\n        self.blocks = ...\n"
            + _N_MIDDLE + "        # each block in turn, with a residual connection\n        ...\n" + _N_END)
_N_SCENARIOS = [
    # the base network on a made-up photo of 16 by 16: what comes out, how many blocks, how many parameters
    "import torch\n\nmodel = ScaledCNN(2, 64)\nout = model(torch.rand(1, 3, 16, 16))\n"
    "[list(out.shape), len(model.blocks), sum(q.numel() for q in model.parameters())]",
    # a small network with weights the check chooses, by name, on two made-up photos of 10 by 10: its first four logits
    "import torch\n\nmodel = ScaledCNN(3, 8)\ntorch.manual_seed(0)\nwith torch.no_grad():\n"
    "    for name, q in sorted(model.named_parameters()):\n        q.copy_(torch.randn_like(q) / 2)\n"
    "x = torch.randn(2, 3, 10, 10)\nmodel(x)[:, :4].tolist()",
    # one step of training: every parameter gets a gradient
    "import torch\n\ntorch.manual_seed(1)\nmodel = ScaledCNN(2, 4)\nloss = (model(torch.randn(3, 3, 8, 8)) ** 2).mean()\n"
    "loss.backward()\n[q.grad is not None for q in model.parameters()]",
]
_AROUND = parameters(0, 64)              # the first convolution and the linear layer of ScaledCNN(2, 64): 2,442
_ONE_BLOCK = parameters(1, 64) - _AROUND   # one block of 64 channels: 4,800
_N_WRONG = {
    # no residual connections
    "no-residual": _n(loop="        for block in self.blocks:\n            x = block(x)\n"),
    # one residual connection around all the blocks together
    "one-residual": _n(blocks="nn.Sequential(*[SeparableConv(width, width) for _ in range(depth)])",
                       loop="        x = x + self.blocks(x)\n"),
    # the sum of each block worked out, but not kept: the network without its blocks
    "no-assign": _n(loop="        for block in self.blocks:\n            x + block(x)\n"),
}


def _base_value(got):
    """The first scenario's value, [output shape, blocks, parameters]; None when `got` is another scenario's."""
    ok = isinstance(got, list) and len(got) == 3 and isinstance(got[0], list) and all(isinstance(v, int) for v in got[1:])
    return got if ok else None


def _plain_list(got, text) -> bool:
    """ScaledCNN(2, 64) has 2 blocks, but PyTorch counts no parameters in them: they are in a plain list."""
    g = _base_value(got)
    return g is not None and g[1] == 2 and g[2] == _AROUND


def _shared(got, text) -> bool:
    """ScaledCNN(2, 64) has 2 blocks, with the parameters of one: the same block twice."""
    g = _base_value(got)
    return g is not None and g[1] == 2 and g[2] == _AROUND + _ONE_BLOCK


def _depth(got, text) -> bool:
    """ScaledCNN(2, 64) has another number of blocks than 2."""
    g = _base_value(got)
    return g is not None and g[1] != 2


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$|=\s*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace both: the blocks in `__init__`, and the loop over them in `forward`.",
                nl="Er staat nog een `...` in je code. Vervang ze allebei: de blocks in `__init__`, en de loop erover in `forward`.")
_TEXTS = _ONE._TEXTS
_S_EN, _S_NL = _ONE._S_EN, _ONE._S_NL
_close = _ONE._close


EXERCISES = {
    "costs": _SORT,
    "scaled": ScenarioBox(
        name="scaled",
        picture=netview.network("ScaledCNN", yours.scene_batch, 2, 64, chosen=1),
        start=_N_START,
        solution=_N,
        scenarios=_N_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "scaled"),
        first=[_DOTS],
        mistakes=[
            Mistake("inplace", error=lambda e: isinstance(e, RuntimeError) and "inplace operation" in str(e),
                    en="PyTorch stops when it works out the gradients: `x += block(x)` changes `x` in place, and PyTorch still needed the old `x` to work out the gradients of the layer before. Write it as a new `x`: `x = x + block(x)`.",
                    nl="PyTorch stopt als het de gradients uitrekent: `x += block(x)` verandert `x` ter plekke, en PyTorch had de oude `x` nog nodig om de gradients van de layer ervoor uit te rekenen. Schrijf het als een nieuwe `x`: `x = x + block(x)`."),
            Mistake("call-list", error=lambda e: (isinstance(e, NotImplementedError) and "forward" in str(e))
                    or (isinstance(e, TypeError) and "'list' object is not callable" in str(e)),
                    en="You call `self.blocks(x)`, but an `nn.ModuleList` is a list of layers, without a `forward` of its own: it does not know in which order, or how, to use them. Go through it with a `for` loop, and give each block its residual connection: `x = x + block(x)`.",
                    nl="Je roept `self.blocks(x)` aan, maar een `nn.ModuleList` is een lijst van layers, zonder eigen `forward`: hij weet niet in welke volgorde, of hoe, hij ze moet gebruiken. Loop er met een `for`-loop doorheen, en geef elk block zijn residual connection: `x = x + block(x)`."),
            Mistake("block-channels", error=lambda e: isinstance(e, RuntimeError) and ("must match the size of tensor" in str(e) or "channels, but got" in str(e)),
                    en="A block gets or gives another number of channels than `width`. Every block gets the `width` channels of the first convolution, and must give back as many, so that `x + block(x)` can add them up: `SeparableConv(width, width)`.",
                    nl="Een block krijgt of geeft een ander aantal channels dan `width`. Elk block krijgt de `width` channels van de eerste convolutie, en moet er evenveel teruggeven, zodat `x + block(x)` ze kan optellen: `SeparableConv(width, width)`."),
            Mistake("range", error=lambda e: isinstance(e, TypeError) and "'int' object is not iterable" in str(e),
                    en="Python cannot go through a number: `for _ in depth` does not work. `range(depth)` gives the numbers 0 to `depth - 1`: `for _ in range(depth)`.",
                    nl="Python kan niet door een getal heen lopen: `for _ in depth` werkt niet. `range(depth)` geeft de getallen 0 tot en met `depth - 1`: `for _ in range(depth)`."),
            Mistake("depth-in-forward", error=lambda e: isinstance(e, NameError) and "'depth'" in str(e),
                    en="`forward` does not know `depth`: it is an argument of `__init__`, and gone when `__init__` is done. What `forward` needs, `__init__` keeps on `self`: go through the blocks with `for block in self.blocks:`.",
                    nl="`forward` kent `depth` niet: het is een argument van `__init__`, en weg als `__init__` klaar is. Wat `forward` nodig heeft, bewaart `__init__` op `self`: loop door de blocks met `for block in self.blocks:`."),
            Mistake("one-block", error=lambda e: isinstance(e, TypeError) and "iterable" in str(e) and "ModuleList" in str(e),
                    en="`nn.ModuleList` gets a list of layers, and you gave it something else, such as one layer. Make the list with square brackets: `nn.ModuleList([SeparableConv(width, width) for _ in range(depth)])`.",
                    nl="`nn.ModuleList` krijgt een lijst van layers, en je gaf hem iets anders, zoals één layer. Maak de lijst met vierkante haken: `nn.ModuleList([SeparableConv(width, width) for _ in range(depth)])`."),
            Mistake("fixed-depth", matches=lambda got, text: re.search(r"range\(\s*\d+\s*\)", text) is not None,
                    en=_S_EN + ": `range` gets a fixed number in your code, so your network makes as many blocks whatever `depth` is. The check also builds `ScaledCNN(3, 8)`, with 3 blocks: use `range(depth)`.",
                    nl=_S_NL + ": `range` krijgt in je code een vast getal, dus maakt je netwerk evenveel blocks wat `depth` ook is. De check bouwt ook `ScaledCNN(3, 8)`, met 3 blocks: gebruik `range(depth)`."),
            Mistake("plain-list", matches=_plain_list,
                    en=_S_EN + ": your network has 2 blocks, but PyTorch counts none of their parameters, only those of the first convolution and the linear layer. In a plain Python list, PyTorch does not see the blocks: `count_params` misses them, and the optimizer never trains them. Put them in an `nn.ModuleList`.",
                    nl=_S_NL + ": je netwerk heeft 2 blocks, maar PyTorch telt geen van hun parameters mee, alleen die van de eerste convolutie en de linear layer. In een gewone Python-lijst ziet PyTorch de blocks niet: `count_params` slaat ze over, en de optimizer traint ze nooit. Zet ze in een `nn.ModuleList`."),
            Mistake("shared", matches=_shared,
                    en=_S_EN + ": your network has 2 blocks with the parameters of one. `[SeparableConv(width, width)] * depth` makes one block and puts it in the list `depth` times, so every place uses the same weights. Make a new block for every place: `[SeparableConv(width, width) for _ in range(depth)]`.",
                    nl=_S_NL + ": je netwerk heeft 2 blocks met de parameters van één. `[SeparableConv(width, width)] * depth` maakt één block en zet het `depth` keer in de lijst, dus gebruikt elke plek dezelfde weights. Maak voor elke plek een nieuw block: `[SeparableConv(width, width) for _ in range(depth)]`."),
            Mistake("depth", matches=_depth,
                    en=_S_EN + ": `ScaledCNN(2, 64)` must have 2 blocks, one for each count of `depth`. `range(depth)` gives exactly `depth` numbers.",
                    nl=_S_NL + ": `ScaledCNN(2, 64)` moet 2 blocks hebben, één voor elke tel van `depth`. `range(depth)` geeft precies `depth` getallen."),
            Mistake("no-assign", wrong=_N_WRONG["no-assign"],
                    en=_S_EN + ": your network gives what it would give without its blocks. `x + block(x)` on a line of its own works out the sum and throws it away; keep it: `x = x + block(x)`.",
                    nl=_S_NL + ": je netwerk geeft wat het zonder zijn blocks zou geven. `x + block(x)` op een eigen regel rekent de som uit en gooit hem weg; bewaar hem: `x = x + block(x)`."),
            Mistake("no-residual", wrong=_N_WRONG["no-residual"],
                    en=_S_EN + ": your blocks have no residual connection. With `x = block(x)`, each block replaces `x`; as in ResNet, each block should add what it finds to `x`: `x = x + block(x)`.",
                    nl=_S_NL + ": je blocks hebben geen residual connection. Met `x = block(x)` vervangt elk block `x`; zoals in ResNet hoort elk block wat het vindt bij `x` op te tellen: `x = x + block(x)`."),
            Mistake("one-residual", wrong=_N_WRONG["one-residual"],
                    en=_S_EN + ": your network has one residual connection around all its blocks together. As in ResNet, give every block its own: `x = x + block(x)` inside the loop, for each block in turn.",
                    nl=_S_NL + ": je netwerk heeft één residual connection om al zijn blocks samen. Geef zoals in ResNet elk block een eigen: `x = x + block(x)` in de loop, voor elk block om de beurt."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the blocks in `__init__`, and the loop over them in `forward`.",
                      "nl": "Vervang de twee `...`: de blocks in `__init__`, en de loop erover in `forward`."},
            "wrong": {"en": _S_EN + ". Does `self.blocks` hold `depth` blocks `SeparableConv(width, width)` in an `nn.ModuleList`, and does `forward` add the output of each block to `x`?",
                      "nl": _S_NL + ". Bevat `self.blocks` `depth` blocks `SeparableConv(width, width)` in een `nn.ModuleList`, en telt `forward` de uitkomst van elk block bij `x` op?"},
            "almost": {"en": "With `ScaledCNN(2, 64)` the shape of what comes out, the blocks and the parameters are right, but " + _S_EN[0].lower() + _S_EN[1:] + ". Does `forward` send `x` through the first convolution and a ReLU, then through each block with `x = x + block(x)`, and give the mean of each map to the linear layer, as in the box?",
                       "nl": "Met `ScaledCNN(2, 64)` kloppen de shape van wat eruit komt, de blocks en de parameters, maar " + _S_NL[0].lower() + _S_NL[1:] + ". Stuurt `forward` `x` door de eerste convolutie en een ReLU, dan door elk block met `x = x + block(x)`, en geeft het het gemiddelde van elke map aan de linear layer, zoals in het vak?"},
            "correct": {"en": f"Your `ScaledCNN` is right: `ScaledCNN(2, 64)` has {grouped(parameters(*BASE[:2]), 'en')} parameters, for photos of every size, and needs {grouped(multiplications(*BASE), 'en')} multiplications for a photo of 16 by 16. It is now in your project, as `{{saved}}`.",
                        "nl": f"Je `ScaledCNN` klopt: `ScaledCNN(2, 64)` heeft {grouped(parameters(*BASE[:2]), 'nl')} parameters, voor foto's van elke grootte, en heeft {grouped(multiplications(*BASE), 'nl')} vermenigvuldigingen nodig voor een foto van 16 bij 16. Hij staat nu in je project, als `{{saved}}`."},
            "kept": {"en": "Your `ScaledCNN` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `ScaledCNN` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_SEQUENTIAL = "nn.Sequential(*[SeparableConv(width, width) for _ in range(depth)])"
CASES = [
    ("costs", {}, "empty", None),
    ("costs", _SORT.solution, "correct", None),
    ("costs", {"depth2": "two", "width2": "four"}, "wrong", "unfinished"),
    ("costs", {**_SORT.solution, "width2": "two"}, "wrong", "wrong:width2"),
    ("costs", {**_SORT.solution, "phi1": "four"}, "wrong", "wrong:phi1"),
    ("scaled", _N_START, "empty", None),
    ("scaled", _N, "correct", None),
    # other ways to write the same: the blocks one by one, a loop over the places, an nn.Sequential gone through by hand
    ("scaled", _N.replace(f"        self.blocks = {_BLOCKS}\n",
                          "        self.blocks = nn.ModuleList()\n        for _ in range(depth):\n"
                          "            self.blocks.append(SeparableConv(width, width))\n"), "correct", None),
    ("scaled", _n(loop="        for i in range(len(self.blocks)):\n            x = x + self.blocks[i](x)\n"), "correct", None),
    ("scaled", _n(blocks=_SEQUENTIAL), "correct", None),
    ("scaled", _n(loop="        for block in self.blocks:\n            x = block(x) + x\n"), "correct", None),
    ("scaled", _N_START.replace("self.blocks = ...", f"self.blocks = {_BLOCKS}"), "wrong", "dots-left"),
    ("scaled", _n(loop="        for block in self.blocks:\n            x += block(x)\n"), "error", "inplace"),
    ("scaled", _n(loop="        x = x + self.blocks(x)\n"), "error", "call-list"),
    ("scaled", _n(blocks="[SeparableConv(width, width) for _ in range(depth)]", loop="        x = x + self.blocks(x)\n"), "error", "call-list"),
    ("scaled", _n(blocks="nn.ModuleList([SeparableConv(width, 2 * width) for _ in range(depth)])"), "error", "block-channels"),
    ("scaled", _n(blocks="nn.ModuleList([SeparableConv(3, width) for _ in range(depth)])"), "error", "block-channels"),
    ("scaled", _n(blocks="nn.ModuleList([SeparableConv(width, width) for _ in depth])"), "error", "range"),
    ("scaled", _n(blocks="nn.ModuleList(SeparableConv(width, width))"), "error", "one-block"),
    ("scaled", _n(blocks="[SeparableConv(width, width) for _ in range(depth)]"), "wrong", "plain-list"),
    ("scaled", _n(blocks="nn.ModuleList([SeparableConv(width, width)] * depth)"), "wrong", "shared"),
    ("scaled", _n(blocks="nn.ModuleList([SeparableConv(width, width) for _ in range(depth + 1)])"), "wrong", "depth"),
    ("scaled", _n(blocks="nn.ModuleList([SeparableConv(width, width) for _ in range(depth - 1)])"), "wrong", "depth"),
    ("scaled", _n(blocks="nn.ModuleList([SeparableConv(width, width) for _ in range(2)])"), "wrong", "fixed-depth"),
    ("scaled", _n(loop="        for block in self.blocks:\n            pass\n"), "wrong", "no-assign"),
    ("scaled", _n(loop="        for i in range(depth):\n            x = x + self.blocks[i](x)\n"), "error", "depth-in-forward"),
    # the right shapes and parameters, but other numbers that no known mistake explains
    ("scaled", _N.replace("self.fc(x.mean(dim=(2, 3)))", "self.fc(x.amax(dim=(2, 3)))"), "almost", None),
] + [("scaled", code, "wrong", key) for key, code in _N_WRONG.items()]


# --- the paper ------------------------------------------------------------------------------------------------------

# Tan and Le (2019), with the numbers of version 5 on arXiv (1905.11946v5, 2020), which differ a little from those of
# the ICML paper of 2019. Table 7: EfficientNet-B0 grown four ways to about 4.5 times its FLOPS (0.4 billion to 1.8
# or 1.9), a multiply-add counted as one: billion FLOPS and ImageNet top-1 accuracy in %. By the rule, d = 4, w = 2
# and r = 2 are 4 times; the compound network has d = 1.4, w = 1.2 and r = 1.3.
PAPER_GROWN = {"base": (0.4, 77.3), "deeper": (1.8, 79.0), "wider": (1.8, 78.9), "sharper": (1.9, 79.1),
               "compound": (1.8, 81.1)}
PAPER_COMPOUND = (1.4, 1.2, 1.3)
# Table 3: MobileNetV1, MobileNetV2 and ResNet-50 grown the same ways: billion FLOPS and top-1 accuracy in %.
PAPER_OTHERS = {
    "mobilenet-v1": {"base": (0.6, 70.6), "wider": (2.2, 74.2), "sharper": (2.2, 72.7), "compound": (2.3, 75.6)},
    "mobilenet-v2": {"base": (0.3, 72.0), "deeper": (1.2, 76.8), "wider": (1.1, 76.4), "sharper": (1.2, 74.8),
                     "compound": (1.3, 77.4)},
    "resnet-50": {"base": (4.1, 76.0), "deeper": (16.2, 78.1), "wider": (14.7, 77.7), "sharper": (16.4, 77.5),
                  "compound": (16.7, 78.8)},
}
# Table 2: top-1 accuracy in %, million parameters and billion FLOPS of the members of the family the lesson shows,
# and of networks with about the same accuracy. GPipe's FLOPS are not in the table.
PAPER_FAMILY = {"B0": (77.1, 5.3, 0.39), "B1": (79.1, 7.8, 0.70), "B4": (82.9, 19, 4.2), "B7": (84.3, 66, 37)}
PAPER_NETS = {"resnet-50": (76.0, 26, 4.1), "resnet-152": (77.8, 60, 11), "gpipe": (84.3, 557, None)}


# --- what the course measured ---------------------------------------------------------------------------------------

# What the course measured, on one thread, with the course's code, through run(): ScaledCNN of each variant (VARIANTS,
# with the course's SeparableConv) made after torch.manual_seed(seed), trained as in cv-1x1-convolution (the Trainer of
# nn-trainer with AdamW, a learning rate of LR, a warmup of one epoch, cosine decay, max_norm 1.0, mini-batches of 64) on
# the first N_TRAIN training photos made side by side, shuffled after torch.manual_seed(seed + 1), for EPOCHS epochs and
# the seeds 0 to 4, with the photos normalized by channel_stats of the training photos. "train" and "test" are how many
# of the training photos and of the first N_TEST test photos are read right after the last epoch; "best" is the first
# epoch with the lowest validation loss, rounded to 3 decimals, and "lowest" that loss; "last_val" and "last_train" are
# the validation and the training loss of the last epoch. CURVES are seed 0. tests/test_cv_efficientnet.py measures
# seed 0 of every variant again when EAI_CIFAR_FILE names the unpacked cifar-10.npz.
REFERENCE = {
    "base": {"train": [3680, 3614, 3590, 3686, 3613], "test": [523, 516, 506, 524, 517],
           "best": [18, 19, 16, 18, 19], "lowest": [1.363, 1.364, 1.352, 1.314, 1.357],
           "last_val": [1.363, 1.364, 1.353, 1.314, 1.358], "last_train": [1.087, 1.114, 1.119, 1.082, 1.106]},
    "deeper": {"train": [4979, 4962, 4802, 4877, 4865], "test": [536, 526, 520, 513, 529],
             "best": [13, 10, 12, 13, 12], "lowest": [1.316, 1.351, 1.37, 1.399, 1.339],
             "last_val": [1.48, 1.511, 1.532, 1.588, 1.517], "last_train": [0.53, 0.534, 0.599, 0.563, 0.58]},
    "wider": {"train": [4428, 4335, 4404, 4312, 4262], "test": [539, 544, 539, 555, 536],
            "best": [15, 17, 16, 18, 19], "lowest": [1.319, 1.32, 1.305, 1.289, 1.33],
            "last_val": [1.34, 1.326, 1.31, 1.29, 1.33], "last_train": [0.77, 0.81, 0.769, 0.799, 0.826]},
    "sharper": {"train": [3693, 3607, 3592, 3667, 3720], "test": [558, 528, 514, 539, 530],
              "best": [20, 19, 19, 18, 19], "lowest": [1.226, 1.277, 1.295, 1.265, 1.267],
              "last_val": [1.226, 1.277, 1.296, 1.267, 1.268], "last_train": [1.086, 1.127, 1.129, 1.09, 1.068]},
    "compound": {"train": [4040, 4131, 3975, 4027, 4060], "test": [559, 548, 543, 571, 571],
               "best": [18, 20, 16, 20, 19], "lowest": [1.274, 1.263, 1.266, 1.241, 1.229],
               "last_val": [1.275, 1.263, 1.27, 1.241, 1.231], "last_train": [0.912, 0.9, 0.949, 0.931, 0.916]},
}
CURVES = {
    "base_val": [1.978, 1.806, 1.715, 1.652, 1.594, 1.551, 1.542, 1.501, 1.478, 1.438, 1.426, 1.397, 1.417, 1.405,
                 1.373, 1.383, 1.368, 1.363, 1.363, 1.363],
    "base_train": [2.139, 1.855, 1.765, 1.685, 1.618, 1.547, 1.485, 1.441, 1.394, 1.356, 1.299, 1.268, 1.233, 1.199,
                   1.164, 1.139, 1.117, 1.102, 1.091, 1.087],
    "deeper_val": [1.897, 1.755, 1.635, 1.505, 1.519, 1.395, 1.41, 1.343, 1.378, 1.372, 1.405, 1.358, 1.316, 1.351,
                   1.384, 1.412, 1.453, 1.477, 1.482, 1.48],
    "deeper_train": [2.103, 1.817, 1.711, 1.596, 1.501, 1.444, 1.37, 1.309, 1.238, 1.184, 1.099, 1.03, 0.942, 0.848,
                     0.772, 0.695, 0.625, 0.573, 0.545, 0.53],
    "wider_val": [1.936, 1.763, 1.666, 1.542, 1.553, 1.491, 1.463, 1.448, 1.44, 1.394, 1.397, 1.334, 1.32, 1.336,
                  1.319, 1.34, 1.353, 1.346, 1.342, 1.34],
    "wider_train": [2.088, 1.808, 1.709, 1.618, 1.534, 1.476, 1.413, 1.355, 1.284, 1.235, 1.143, 1.094, 1.032, 0.963,
                    0.915, 0.869, 0.825, 0.796, 0.778, 0.77],
    "sharper_val": [2.026, 1.799, 1.707, 1.645, 1.596, 1.543, 1.514, 1.454, 1.444, 1.376, 1.365, 1.339, 1.334, 1.288,
                    1.25, 1.257, 1.239, 1.231, 1.227, 1.226],
    "sharper_train": [2.151, 1.88, 1.762, 1.684, 1.627, 1.549, 1.48, 1.427, 1.372, 1.346, 1.297, 1.262, 1.222, 1.19,
                      1.16, 1.136, 1.115, 1.101, 1.09, 1.086],
    "compound_val": [1.946, 1.809, 1.662, 1.593, 1.579, 1.502, 1.456, 1.421, 1.424, 1.403, 1.438, 1.318, 1.318, 1.299,
                     1.281, 1.298, 1.288, 1.274, 1.274, 1.275],
    "compound_train": [2.129, 1.836, 1.727, 1.649, 1.585, 1.508, 1.44, 1.381, 1.318, 1.291, 1.218, 1.185, 1.127,
                       1.068, 1.032, 0.993, 0.961, 0.935, 0.918, 0.912],
}
DEFAULT = "compound"   # the demo's network when the learner chooses none
RANGE = (500, 600)     # the axis of the picture of the runs, which every run of the course fits on


def mean(values) -> float:
    return sum(values) / len(values)


# --- the picture of the runs ----------------------------------------------------------------------------------------

def runs_picture(labels: dict, lang: str, mine: dict | None = None) -> str:
    """How many of the N_TEST test photos each network reads right after EPOCHS epochs: an open dot for every seed of
    the course and a line at their mean, a row for each of ORDER. With `mine` ({"variant", "test"}), the learner's run
    as a filled dot in the row of its network. `labels` gives "title", "x", a name for each row, "mine", "course" and
    "alt"."""
    yours = {} if mine is None else {mine["variant"]: mine["test"]}
    # from RANGE, wider where the learner's run lies outside, so that every dot is where its number is
    low, high = min([RANGE[0]] + [v - 15 for v in yours.values()]), max([RANGE[1]] + [v + 15 for v in yours.values()])
    tick = 20 if high - low <= 200 else 50 if high - low <= 400 else 100
    low, high = int(low // tick * tick), int(-(-high // tick) * tick)
    p = chart.Panel((low, high), (-1.0, len(ORDER) - 0.3), title=labels["title"], xlabel=labels["x"], yticks=[],
                    xticks=list(range(low, high + 1, tick)))
    step = (high - low) / 220
    for k, key in enumerate(ORDER):
        y = len(ORDER) - 1 - k
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
    svg = chart.figure([p], alt=labels["alt"], width=440, height=60 + 44 * len(ORDER) + (24 if mine is not None else 0))
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


TOP = 2.6   # the top of the curves picture: above the loss of guessing, ln 10 ≈ 2.30


def curves_picture(curves: dict, labels: dict, lang: str, at: int | None = None) -> str:
    """The loss after every epoch of the demo's run: on the training photos solid and on the test photos dashed, with
    a dot where the validation loss is lowest; in gray the validation loss of the course's base network (seed 0). With
    `at`, a line at that epoch, for rewinding. `labels` gives "title", "x", "y", "train", "val", "base" and "alt"."""
    train, val = curves.get("train", []), curves.get("val", [])
    mine = [v for v in train + val if np.isfinite(v)]
    top = max([TOP] + [1.1 * v for v in mine])
    before = CURVES["base_val"]
    right = max(len(val), len(train), len(before), 1) + 0.5
    p = chart.Panel((0, right), (0, top), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    p.line(list(range(1, len(before) + 1)), before, "ink-3", 1.6, dashed=True)
    # its name in a key at the top right, above every line: at the end of the line it would lie on the names of the
    # learner's lines, which end close to it (on top of it, for the base network)
    p.line([right - 9.6, right - 8.1], [0.92 * top] * 2, "ink-3", 1.6, dashed=True)
    p.text(right - 7.7, 0.92 * top, labels["base"], "ink-3", "start")
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
    return _ONE.now_pictures(seen, title, lang, uid)


def epoch_frame(result: dict, e: int, labels: dict, title: str, lang: str) -> str:
    """The rewind at epoch e (from 1): the loss after every epoch with a line at e, then the network's guesses for the
    photos SHOWN after it, and under them its first layer on the car, as it was then (cv-augment-photos)."""
    return _ONE.rewind_frame(curves_picture, result, e, labels, title, lang, "efficientnet")


# The lesson is done when these are correct. "scaling-runs" is the choice question in the lesson.
REQUIRED = ["costs", "scaled", "scaling-runs"]

DEMO_LIMIT = 600   # seconds: the deeper network trains in about 100 on one thread, the lesson says "ten minutes"


# --- the demo: ScaledCNN on CIFAR-10 --------------------------------------------------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def _from_mine(e: BaseException) -> bool:
    """The error came from the learner's network: a line of their scaled.py or separable.py is in its traceback."""
    mine = {(projects.folder() / "cnn" / f"{m}.py").resolve() for m in ("scaled", "separable")}
    tb = e.__traceback__
    while tb is not None:
        if Path(tb.tb_frame.f_code.co_filename).resolve() in mine:
            return True
        tb = tb.tb_next
    return False


def _demo(variant: str, show=None):
    """For the demo: None when scaled.py is not in the project yet; "data" when CIFAR-10 is not unpacked; "scaled" when
    the learner's ScaledCNN stops with an error, also during the training, or does not give back 10 logits for each
    photo; "cnn" when channel_stats does not give what it should; "trainer" when the Trainer or count_params stops with
    an error; "slow" when the training takes too long; else the dict of run(), with the modules the course's versions
    stood in for."""
    if not _saved("scaled"):
        return None
    if not ready():
        return "data"
    loaded = run_code("from cnn.scaled import ScaledCNN")
    if loaded.error is not None:
        return "scaled"
    ScaledCNN = loaded.names["ScaledCNN"]
    try:
        def tried():   # as in training: a mini-batch, in training mode, with a backward pass
            torch.manual_seed(0)
            model = ScaledCNN(2, 8)
            out = model(torch.rand(4, 3, 16, 16))
            out.sum().backward()
            return tuple(out.shape)
        if with_time_limit(tried, 10) != (4, 10):
            return "scaled"
    except Exception:
        return "scaled"
    loaded = run_code("from cnn.channels import channel_stats")
    if loaded.error is not None:
        return "cnn"
    channel_stats = loaded.names["channel_stats"]
    loaded = run_code("from nn.trainer import Trainer\nfrom cnn.count import count_params")
    if loaded.error is not None:
        return "trainer"
    Trainer, count_params = loaded.names["Trainer"], loaded.names["count_params"]
    try:
        int(with_time_limit(lambda: count_params(nn.Linear(2, 2)), 10))
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
        result = with_time_limit(lambda: run(variant, 0, ScaledCNN, channel_stats, Trainer, count_params, show), DEMO_LIMIT)
    except Timeout:
        return "slow"
    except Exception as e:
        return "scaled" if _from_mine(e) else "trainer"
    return result | {"stand_ins": stand_ins}


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file), and whether CIFAR-10
    is there."""
    texts = []
    for package, module in (("cnn", "scaled"), ("cnn", "separable"), ("cnn", "channels"), ("cnn", "count"),
                            ("nn", "train_epoch"), ("nn", "trainer")):
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
