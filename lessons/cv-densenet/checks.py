# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-densenet: in a dense block every layer gets the maps of the block's input and of all the
layers before it, stacked along the channels with torch.cat, and adds `growth` maps of its own. The learner's DenseBlock
is saved in their project (work/projects/cnn/dense.py) when right, and checked on made-up numbers, so it works without
CIFAR-10. The demo trains the CNN of cv-1x1-convolution with the learner's dense block in its middle, followed by a
transition, a 1×1 convolution back to 64 channels: growth 24 with 2 layers, 12 with 4 or 6 with 8, all ending with 112
channels and about as many weights; or the course's chain of 4 layers that each see only the layer before. It trains
with the learner's Trainer on the first 6,000 training photos of CIFAR-10, tests on the first 1,000 test photos, and
keeps the network's guess for one checked test photo of each kind after every epoch, for rewinding."""
import html
import importlib.util
import re
from pathlib import Path

import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import chart, draw, live, paths, projects
from everything_ai.check import Mistake, ScenarioBox, Sort, Timeout, run_code, with_time_limit


def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_densenet",
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
EPOCHS, LR, CHANNELS, SIDE = _ONE.EPOCHS, _ONE.LR, _ONE.CHANNELS, _ONE.SIDE
conv_params = _ONE.conv_params
WideCNN, Plain, AROUND, PLAIN_BLOCK = _ONE.WideCNN, _ONE.Plain, _ONE.AROUND, _ONE.PLAIN_BLOCK


# --- counting -------------------------------------------------------------------------------------------------------

def channels_in(channels: int, growth: int, layer: int) -> int:
    """The channels layer `layer` (1, 2, ...) of a dense block gets: the block's input and the maps of the layers before
    it, the paper's k0 + k·(l − 1)."""
    return channels + growth * (layer - 1)


def channels_out(channels: int, growth: int, layers: int) -> int:
    """The channels a dense block gives back: its input and the maps of all its layers."""
    return channels + growth * layers


def layer_params(channels: int, growth: int, layer: int) -> int:
    """The parameters of layer `layer` of a dense block: a 3×3 convolution to `growth` maps, with a bias for each."""
    return conv_params(channels_in(channels, growth, layer), growth, 3)


def block_params(channels: int, growth: int, layers: int) -> int:
    """The parameters of a dense block, as nn.Conv2d makes its convolutions: with a bias for every filter."""
    return sum(layer_params(channels, growth, l) for l in range(1, layers + 1))


def connections(layers: int) -> int:
    """The direct connections in a dense block of `layers` layers: layer l gets l earlier outputs, the input included."""
    return layers * (layers + 1) // 2


# The demo's dense blocks on the 64 maps of 8 by 8 in the middle of WideCNN, each followed by a transition back to 64
# channels: (growth, layers). All three end with 64 + 48 = 112 channels.
VARIANTS = {"k24": (24, 2), "k12": (12, 4), "k6": (6, 8)}
CHAIN = 30   # the channels of each layer of the course's chain: about as many weights as the dense block "k12"


def chain_params(width: int = CHAIN, layers: int = 4) -> int:
    """The course's chain: 3×3 convolutions from CHANNELS to `width` and from `width` to `width`, each seeing only the
    layer before, and the 1×1 transition from `width` back to CHANNELS."""
    return conv_params(CHANNELS, width, 3) + (layers - 1) * conv_params(width, width, 3) + conv_params(width, CHANNELS)


def middle_params(variant: str) -> int:
    """The parameters of the demo's middle for `variant`: the block and its transition, or the chain."""
    if variant == "chain":
        return chain_params()
    if variant == "plain":
        return PLAIN_BLOCK
    growth, layers = VARIANTS[variant]
    return block_params(CHANNELS, growth, layers) + conv_params(channels_out(CHANNELS, growth, layers), CHANNELS)


# --- the block, as the course writes it -----------------------------------------------------------------------------

_D_HEAD = ("import torch\nfrom torch import nn\n\n\nclass DenseBlock(nn.Module):\n"
           "    def __init__(self, channels, growth, layers):\n        super().__init__()\n"
           "        self.layers = nn.ModuleList()\n        for i in range(layers):\n")
_D_FORWARD = "\n    def forward(self, x):\n        for layer in self.layers:\n            y = torch.relu(layer(x))\n"


def _d(conv: str = "nn.Conv2d(channels + i * growth, growth, 3, padding=1)", join: str = "torch.cat([x, y], dim=1)") -> str:
    return (_D_HEAD + f"            self.layers.append({conv})\n" + _D_FORWARD + f"            x = {join}\n"
            + "        return x\n")


_D = _d()


class Middle(nn.Module):
    """The demo's middle: a dense block, then the transition, a 1×1 convolution from all its channels back to
    `channels`, with a ReLU."""

    def __init__(self, block, channels_after: int, channels: int = CHANNELS):
        super().__init__()
        self.block = block
        self.transition = nn.Conv2d(channels_after, channels, 1)

    def forward(self, x):
        return torch.relu(self.transition(self.block(x)))


class Chain(nn.Module):
    """The course's chain for the demo: `layers` 3×3 convolutions with a ReLU that each see only the layer before, from
    `channels` to `width` and on, then the 1×1 transition back to `channels`, with a ReLU."""

    def __init__(self, channels: int = CHANNELS, width: int = CHAIN, layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(nn.Conv2d(channels if i == 0 else width, width, 3, padding=1) for i in range(layers))
        self.transition = nn.Conv2d(width, channels, 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return torch.relu(self.transition(x))


def make_middle(variant: str, DenseBlock):
    """The demo's middle for `variant`: the chain, or the learner's DenseBlock with its transition."""
    if variant == "chain":
        return Chain()
    growth, layers = VARIANTS[variant]
    return Middle(DenseBlock(CHANNELS, growth, layers), channels_out(CHANNELS, growth, layers))


# --- the demo: the learner's DenseBlock in the middle of WideCNN, on CIFAR-10 ---------------------------------------

def _accuracy(model, X, y) -> int:
    model.eval()
    with torch.no_grad():
        return int(sum((model(X[i:i + 500]).argmax(dim=1) == y[i:i + 500]).sum() for i in range(0, len(X), 500)))


def run(variant: str, seed: int, DenseBlock, channel_stats, Trainer, count_params, show=None):
    """One run of WideCNN with the middle of `variant` (make_middle), as REFERENCE says: the middle made after
    torch.manual_seed(seed), then the rest of the network, and the photos shuffled after torch.manual_seed(seed + 1).
    Gives back a dict with the parameters of the network and of its middle, how many of the training and test photos it
    reads right, the loss after every epoch ("curves"), the epoch with the lowest validation loss, and the network's
    guess for the photos SHOWN after every epoch ("guesses") with their right kinds ("truth"). Errors go to the caller,
    which says whose code made them. `show` hears the run while it trains."""
    X, y, X_test, y_test = photos()
    mean_, std = channel_stats(X.clone())
    mean_ = torch.as_tensor(mean_, dtype=torch.float32).reshape(3)
    std = torch.as_tensor(std, dtype=torch.float32).reshape(3)
    X = (X - mean_[:, None, None]) / std[:, None, None]
    X_test = (X_test - mean_[:, None, None]) / std[:, None, None]
    torch.manual_seed(seed)
    middle = make_middle(variant, DenseBlock)
    model = WideCNN(middle)
    params, middle_params_ = int(count_params(model)), int(count_params(middle))
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
    live.follow(trainer, heard, "densenet")
    history = trainer.fit() or []
    curves = {"train": [float(h["train_loss"]) for h in history if isinstance(h, dict) and "train_loss" in h],
              "val": [float(h["val_loss"]) for h in history if isinstance(h, dict) and "val_loss" in h]}
    return {"variant": variant, "params": params, "middle_params": middle_params_, "train": _accuracy(model, X, y),
            "test": _accuracy(model, X_test, y_test), "curves": curves, "best": best_epoch(curves["val"]),
            "guesses": guesses[:len(curves["val"])], "truth": [int(y_test[place]) for place in SHOWN]}


# --- the picture of the connections ---------------------------------------------------------------------------------

def connections_picture(labels: dict, layers: int = 4, growth: int = 12, channels: int = CHANNELS):
    """A dense block as a row of boxes, its input x and its layers, with an arrow from every box to every later layer:
    layer l gets the maps of x and of the l − 1 layers before it, so layers·(layers + 1) / 2 arrows. Arrows go out of
    the right half of a box and come in on the left half of a later one; the arrows into the last layer are darker.
    Under each box what it gets and what it gives, the maps a layer makes in the colour of values inside the model.
    `labels` gives "title", "input", "layer" (with {l}), "gets" and "gives" (with {n}), and "alt"."""
    BOX_W, BOX_H, GAP, LEFT = 82, 40, 30, 10
    n = layers + 1
    top = 34 + 20 * layers + 14
    W = LEFT * 2 + n * BOX_W + (n - 1) * GAP
    small = "font-size:12.5px;fill:var(--ea-ink-2);text-anchor:middle"
    parts = [f'<defs><marker id="ea-dense-head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">'
             f'<path d="M0,0 L8,4 L0,8 z" style="fill:var(--ea-ink-3)"/></marker>'
             f'<marker id="ea-dense-head-last" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">'
             f'<path d="M0,0 L8,4 L0,8 z" style="fill:var(--ea-ink-2)"/></marker></defs>',
             f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']

    half = BOX_W / 2
    step = (half - 12) / max(layers - 1, 1)
    for j in range(1, n):            # the arrows into layer j, from x (0) and the layers before it
        for i in range(j):
            last = j == n - 1
            x0 = LEFT + i * (BOX_W + GAP) + half + 6 + (j - i - 1) * step   # out of the right half of box i
            x1 = LEFT + j * (BOX_W + GAP) + 6 + i * step                    # into the left half of box j
            h = top - 16 - 19 * (j - i)
            style = ("stroke:var(--ea-ink-2);stroke-width:1.6" if last else "stroke:var(--ea-ink-3);stroke-width:1.1")
            head = "ea-dense-head-last" if last else "ea-dense-head"
            parts.append(f'<path d="M{x0:.1f},{top} C{x0:.1f},{h} {x1:.1f},{h} {x1:.1f},{top - 2}" '
                         f'style="fill:none;{style}" marker-end="url(#{head})"/>')
    for i in range(n):
        x = LEFT + i * (BOX_W + GAP)
        name = labels["input"] if i == 0 else labels["layer"].format(l=i)
        parts.append(f'<rect x="{x}" y="{top}" width="{BOX_W}" height="{BOX_H}" rx="5" '
                     f'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.2"/>')
        parts.append(f'<text x="{x + BOX_W / 2}" y="{top + 25}" style="font-size:13.5px;font-weight:600;fill:var(--ea-ink);text-anchor:middle">{html.escape(name)}</text>')
        if i == 0:
            parts.append(f'<text x="{x + BOX_W / 2}" y="{top + BOX_H + 20}" style="{small}">{html.escape(labels["gives"].format(n=channels))}</text>')
        else:
            parts.append(f'<text x="{x + BOX_W / 2}" y="{top + BOX_H + 20}" style="{small}">{html.escape(labels["gets"].format(n=channels_in(channels, growth, i)))}</text>')
            parts.append(f'<text x="{x + BOX_W / 2}" y="{top + BOX_H + 37}" style="font-size:12.5px;fill:var(--ea-hidden-text);text-anchor:middle">{html.escape(labels["gives"].format(n=growth))}</text>')
    return draw._svg(W, top + BOX_H + 46, labels["alt"], "".join(parts), natural=True)


# --- the slider: the channels each layer gets, and its parameters ---------------------------------------------------

LAYERS = list(range(1, 9))
GROWTH = 12   # the slider's growth rate, on the 64 channels of the demo's middle


def first_more() -> int:
    """The fewest layers for which the layers of a dense block with GROWTH on CHANNELS have more parameters together
    than one 3×3 convolution from CHANNELS to CHANNELS."""
    return next(n for n in LAYERS if block_params(CHANNELS, GROWTH, n) > PLAIN_BLOCK)


def stack_picture(layers: int, labels: dict, lang: str) -> str:
    """For a dense block with GROWTH on CHANNELS: a bar for each layer, the channels it gets, the block's input in
    gray and the maps of each layer before it in the colour of values inside the model, with its parameters
    in the colour of what a model learns; and a last bar for what the block gives back. The axis and the room for 8 layers stay the same for every
    number of layers. `labels` gives "title", "x", "layer" (with {l}), "out" and "alt"."""
    rows = len(LAYERS) + 1
    p = chart.Panel((0, 175), (-0.7, rows - 0.2), title=labels["title"], xlabel=labels["x"], yticks=[],
                    xticks=[0, 25, 50, 75, 100, 125, 150, 175])

    def bar(y, upto):
        p.rect(0, y - 0.17, CHANNELS, y + 0.17, "ink-3")
        for j in range(1, (upto - CHANNELS) // GROWTH + 1):
            p.rect(CHANNELS + (j - 1) * GROWTH, y - 0.17, CHANNELS + j * GROWTH, y + 0.17, "hidden")

    for l in range(1, layers + 1):
        y = rows - l
        n = channels_in(CHANNELS, GROWTH, l)
        bar(y, n)
        p.text(1, y + 0.45, labels["layer"].format(l=l, n=n), "ink-2", "start")
        p.text(n + 2, y - 0.08, grouped(layer_params(CHANNELS, GROWTH, l), lang), "param", "start")
    y = rows - layers - 1
    out = channels_out(CHANNELS, GROWTH, layers)
    bar(y, out)
    p.text(1, y + 0.45, labels["out"].format(n=out), "ink", "start")
    svg = chart.figure([p], alt=labels["alt"], width=440, height=420)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def stack_sentence(layers: int, lang: str) -> str:
    """The sentence under the slider, for `layers` layers."""
    out, params, links = channels_out(CHANNELS, GROWTH, layers), block_params(CHANNELS, GROWTH, layers), connections(layers)
    p, plain = grouped(params, lang), grouped(PLAIN_BLOCK, lang)
    if lang == "nl":
        word = "layer" if layers == 1 else "layers"
        return (f"Met {layers} {word} geeft het block {out} channels terug. Zijn layers hebben samen {p} parameters; "
                f"één 3×3-convolutie van 64 channels naar 64 heeft er {plain}. Het block heeft {links} {'verbinding' if links == 1 else 'verbindingen'}.")
    word = "layer" if layers == 1 else "layers"
    return (f"With {layers} {word}, the block gives back {out} channels. Its layers have {p} parameters together; "
            f"one 3×3 convolution from 64 channels to 64 has {plain}. The block has {links} {'connection' if links == 1 else 'connections'}.")


# --- exercise 1: a residual block or a dense block ------------------------------------------------------------------

_SORT = Sort(
    name="blocks",
    items={
        "keeps": {"en": "**As many channels.** It gives back as many channels as it gets.", "nl": "**Evenveel channels.** Het geeft evenveel channels terug als het krijgt."},
        "more": {"en": "**More channels.** It gives back more channels than it gets.", "nl": "**Meer channels.** Het geeft meer channels terug dan het krijgt."},
        "adds": {"en": "**Adds up.** It adds what its layers make to its input, number by number.", "nl": "**Optellen.** Het telt wat zijn layers maken getal voor getal bij zijn input op."},
        "after": {"en": "**Puts after.** It puts the maps of each layer after the maps it got, with `torch.cat`.", "nl": "**Erachter zetten.** Het zet de maps van elke layer achter de maps die het kreeg, met `torch.cat`."},
        "shape": {"en": "**Same shape.** What its layers make must have the shape of its input, channels included.", "nl": "**Dezelfde shape.** Wat zijn layers maken, moet de shape van zijn input hebben, ook het aantal channels."},
        "sees": {"en": "**Sees everything.** Its last layer gets the block's input and the maps of every layer before it, unchanged.", "nl": "**Ziet alles.** Zijn laatste layer krijgt de input van het block en de maps van elke layer ervoor, onveranderd."},
    },
    groups={"residual": {"en": "a residual block (ResNet)", "nl": "een residual block (ResNet)"},
            "dense": {"en": "a dense block (DenseNet)", "nl": "een dense block (DenseNet)"}},
    solution={"keeps": "residual", "more": "dense", "adds": "residual", "after": "dense", "shape": "residual", "sees": "dense"},
    nudge={
        "keeps": {"en": "`x + f(x)` has the shape of `x`. And `torch.cat` along the channels?",
                  "nl": "`x + f(x)` heeft de shape van `x`. En `torch.cat` langs de channels?"},
        "more": {"en": "Which block keeps its input and puts new maps after it?",
                 "nl": "Welk block houdt zijn input en zet er nieuwe maps achter?"},
        "adds": {"en": "Which block gives back `x + f(x)`?", "nl": "Welk block geeft `x + f(x)` terug?"},
        "after": {"en": "Which block makes its output longer along the channels with every layer?",
                  "nl": "Welk block maakt zijn output met elke layer langer langs de channels?"},
        "shape": {"en": "`+` adds number by number. `torch.cat` along the channels needs the same height and width, but not the same number of channels.",
                  "nl": "`+` telt getal voor getal op. `torch.cat` langs de channels heeft dezelfde hoogte en breedte nodig, maar niet hetzelfde aantal channels."},
        "sees": {"en": "In which block are the maps of the earlier layers still there, one after the other?",
                 "nl": "In welk block staan de maps van de eerdere layers er nog, na elkaar?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the six sentences again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de zes zinnen opnieuw uit."},
        "empty": {"en": "Choose for each sentence: a residual block or a dense block.",
                  "nl": "Kies bij elke zin: een residual block of een dense block."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "{item} belongs to the other block.", "nl": "{item} hoort bij het andere block."},
        "correct": {"en": "All {n} are right. A residual block adds: `x + f(x)` keeps the shape of `x`, so `f(x)` must have it too. A dense block keeps: `torch.cat([x, y], dim=1)` puts the new maps after the old ones, so the channels grow with every layer, and the last layer still gets the block's input and the maps of every layer before it.",
                    "nl": "Alle {n} kloppen. Een residual block telt op: `x + f(x)` houdt de shape van `x`, dus moet `f(x)` die ook hebben. Een dense block houdt: `torch.cat([x, y], dim=1)` zet de nieuwe maps achter de oude, dus groeien de channels met elke layer, en krijgt de laatste layer nog de input van het block en de maps van elke layer ervoor."},
    },
)

# --- exercise 2: a dense block --------------------------------------------------------------------------------------

_D_START = (_D_HEAD + "            # layer i: 3×3 from all the maps so far to growth, same size\n            self.layers.append(...)\n"
            + _D_FORWARD + "            # put y after x along the channels\n            x = ...\n        return x\n")
# The scenarios give the layers their weights in their order in self.layers.
_D_WEIGHTS = ("with torch.no_grad():\n    for layer in block.layers:\n"
              "        for q in layer.parameters():\n            q.copy_(torch.randn_like(q) / 4)\n")
_D_SCENARIOS = [
    # the demo's block with growth 12 and 4 layers on made-up maps of 8 by 8: shapes and parameters
    "import torch\n\nblock = DenseBlock(64, 12, 4)\nout = block(torch.rand(1, 64, 8, 8))\n"
    "[list(out.shape), [list(layer.weight.shape) for layer in block.layers], sum(q.numel() for q in block.parameters())]",
    # a small block: the numbers of every channel at a pixel in the middle and at a corner
    "import torch\n\nblock = DenseBlock(3, 2, 3)\ntorch.manual_seed(0)\n" + _D_WEIGHTS
    + "out = block(torch.randn(1, 3, 5, 5))\n[out[0, :, 2, 2].tolist(), out[0, :, 0, 0].tolist()]",
    # another size: the weights of the layers and the shape
    "import torch\n\ntorch.manual_seed(1)\nblock = DenseBlock(5, 4, 2)\nout = block(torch.randn(2, 5, 6, 6))\n"
    "[[list(layer.weight.shape) for layer in block.layers], list(out.shape)]",
]
_D_WRONG = {
    # the new maps in front of x: the same channels in another order
    "order": _d(join="torch.cat([y, x], dim=1)"),
}


def _conv_args(text: str) -> str:
    found = re.search(r"self\.layers\.append\(\s*nn\.Conv2d\(([^\n]*)\)\s*\)", text)
    return found.group(1) if found else ""


_FIXED = Mistake("fixed", matches=lambda got, text: re.search(r"\b\d{2,}\b", _conv_args(text)) is not None,
                 en="The convolution of a layer has a fixed number in it. The block must work for every size: use the names `channels`, `growth` and `i`.",
                 nl="De convolutie van een layer heeft een vast getal erin. Het block moet voor elke grootte werken: gebruik de namen `channels`, `growth` en `i`.")
# `...` on a line of its own, or after `=` or `append(`
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$|=\s*\.\.\.|append\(\s*\.\.\.\s*\)", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace both with your own code: the convolution of each layer, and the new `x`.",
                nl="Er staat nog een `...` in je code. Vervang ze allebei door je eigen code: de convolutie van elke layer, en de nieuwe `x`.")
_TEXTS = _ONE._TEXTS
_S_EN, _S_NL = _ONE._S_EN, _ONE._S_NL
_close = _ONE._close


EXERCISES = {
    "blocks": _SORT,
    "dense": ScenarioBox(
        name="dense",
        start=_D_START,
        solution=_D,
        scenarios=_D_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "dense"),
        first=[_DOTS, _FIXED],
        mistakes=[
            Mistake("no-cat", syntax=r"^\s*x\s*=\s*y\s*(#.*)?$",
                    en="`x = y` keeps only the new maps of a layer, and throws away the maps so far: the next layer then gets `growth` channels, where it was made for all of them. Put `y` after `x`: `torch.cat([x, y], dim=1)`.",
                    nl="`x = y` houdt alleen de nieuwe maps van een layer, en gooit de maps tot nu toe weg: de volgende layer krijgt dan `growth` channels, terwijl hij voor allemaal gemaakt is. Zet `y` achter `x`: `torch.cat([x, y], dim=1)`."),
            Mistake("plus", error=lambda e: isinstance(e, RuntimeError) and "must match the size of tensor" in str(e),
                    en="`+` adds number by number, and stops: `x` has more channels than `y`, which has `growth`. A dense block does not add, it keeps: put `y` after `x` with `torch.cat([x, y], dim=1)`.",
                    nl="`+` telt getal voor getal op, en stopt: `x` heeft meer channels dan `y`, dat er `growth` heeft. Een dense block telt niet op, het houdt: zet `y` achter `x` met `torch.cat([x, y], dim=1)`."),
            Mistake("stack", error=lambda e: isinstance(e, RuntimeError) and "stack expects each tensor to be equal size" in str(e),
                    en="`torch.stack` makes a new axis, and needs tensors of the same shape. `torch.cat` puts them after each other along an axis they have: `torch.cat([x, y], dim=1)`, along the channels.",
                    nl="`torch.stack` maakt een nieuwe axis, en heeft tensors van dezelfde shape nodig. `torch.cat` zet ze na elkaar langs een axis die ze hebben: `torch.cat([x, y], dim=1)`, langs de channels."),
            Mistake("dim", error=lambda e: isinstance(e, RuntimeError) and re.search(r"must match except in dimension [023]\b", str(e)) is not None,
                    en="`torch.cat` puts the tensors after each other along the `dim` you give it, and needs the same size in every other dimension. Along any `dim` but 1, that includes the channels, and `x` has more than `y`; without a `dim`, it uses `dim=0`, the photos of the batch. The maps go after each other along the channels: `dim=1`.",
                    nl="`torch.cat` zet de tensors na elkaar langs de `dim` die je hem geeft, en heeft in elke andere dimensie dezelfde grootte nodig. Langs elke `dim` behalve 1 zijn dat ook de channels, en `x` heeft er meer dan `y`; zonder `dim` gebruikt hij `dim=0`, de foto's van de batch. De maps komen na elkaar langs de channels: `dim=1`."),
            Mistake("padding", error=lambda e: isinstance(e, RuntimeError) and "must match except in dimension 1" in str(e),
                    en="`torch.cat` stops: `y` has other heights and widths than `x`. A 3×3 convolution keeps them only with `padding=1` and stride 1: without padding they get 2 smaller, with `padding=2` 2 larger.",
                    nl="`torch.cat` stopt: `y` heeft andere hoogtes en breedtes dan `x`. Een 3×3-convolutie houdt ze alleen met `padding=1` en stride 1: zonder padding worden ze 2 kleiner, met `padding=2` 2 groter."),
            Mistake("channels", error=lambda e: isinstance(e, RuntimeError) and "channels, but got" in str(e),
                    en="A layer gets another number of channels than it was made for. The first number of `nn.Conv2d` is what a layer gets, the second what it makes. Layer `i` gets the block's input and the maps of the `i` layers before it, `channels + i * growth` channels, and makes `growth` maps, no more. And does `forward` put `y` after `x` for every layer?",
                    nl="Een layer krijgt een ander aantal channels dan waarvoor hij gemaakt is. Het eerste getal van `nn.Conv2d` is wat een layer krijgt, het tweede wat hij maakt. Layer `i` krijgt de input van het block en de maps van de `i` layers ervoor, `channels + i * growth` channels, en maakt `growth` maps, niet meer. En zet `forward` voor elke layer `y` achter `x`?"),
            Mistake("order", wrong=_D_WRONG["order"],
                    en=_S_EN + ": you put the new maps in front of `x`. The block then has the right channels, in another order, and the check compares them channel by channel. DenseNet keeps the input first, then the maps of each layer in their order: `torch.cat([x, y], dim=1)`.",
                    nl=_S_NL + ": je zet de nieuwe maps vóór `x`. Het block heeft dan de goede channels, in een andere volgorde, en de check vergelijkt ze channel voor channel. DenseNet houdt de input vooraan, en daarna de maps van elke layer in hun volgorde: `torch.cat([x, y], dim=1)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...` with your code: the convolution of each layer, and the new `x`.",
                      "nl": "Vervang de twee `...` door je code: de convolutie van elke layer, en de nieuwe `x`."},
            "wrong": {"en": _S_EN + ". Does layer `i` get `channels + i * growth` channels and give `growth`, with `padding=1`, and does `forward` put `y` after `x` with `torch.cat([x, y], dim=1)`?",
                      "nl": _S_NL + ". Krijgt layer `i` `channels + i * growth` channels en geeft hij er `growth`, met `padding=1`, en zet `forward` `y` achter `x` met `torch.cat([x, y], dim=1)`?"},
            "almost": {"en": "With growth 12 and 4 layers on 64 channels your block is right, but " + _S_EN[0].lower() + _S_EN[1:] + ". Did you change the lines that were in the box? Each layer sends `x` through its convolution and a ReLU, and puts what comes out after `x`.",
                       "nl": "Met groei 12 en 4 layers op 64 channels klopt je block, maar " + _S_NL[0].lower() + _S_NL[1:] + ". Heb je de regels veranderd die in het vak stonden? Elke layer stuurt `x` door zijn convolutie en een ReLU, en zet wat eruit komt achter `x`."},
            "correct": {"en": f"Your `DenseBlock` is right: with growth 12 and 4 layers on 64 channels, it gives back {channels_out(64, 12, 4)} channels, and its layers have {grouped(block_params(64, 12, 4), 'en')} parameters. It is now in your project, as `{{saved}}`.",
                        "nl": f"Je `DenseBlock` klopt: met groei 12 en 4 layers op 64 channels geeft hij {channels_out(64, 12, 4)} channels terug, en hebben zijn layers {grouped(block_params(64, 12, 4), 'nl')} parameters. Hij staat nu in je project, als `{{saved}}`."},
            "kept": {"en": "Your `DenseBlock` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `DenseBlock` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("blocks", {}, "empty", None),
    ("blocks", _SORT.solution, "correct", None),
    ("blocks", {"keeps": "residual", "more": "dense"}, "wrong", "unfinished"),
    ("blocks", {**_SORT.solution, "shape": "dense"}, "wrong", "wrong:shape"),
    ("blocks", {**_SORT.solution, "sees": "residual"}, "wrong", "wrong:sees"),
    ("dense", _D_START, "empty", None),
    ("dense", _D, "correct", None),
    # other ways to write the same: padding "same", arguments by name, the channels worked out another way, torch.concat
    ("dense", _d(conv="nn.Conv2d(channels + i * growth, growth, 3, padding=\"same\")"), "correct", None),
    ("dense", _d(conv="nn.Conv2d(in_channels=channels + growth * i, out_channels=growth, kernel_size=3, padding=1)"), "correct", None),
    ("dense", _d(join="torch.concat([x, y], dim=1)"), "correct", None),
    ("dense", _d(join="torch.cat((x, y), 1)"), "correct", None),
    ("dense", _d(conv="nn.Conv2d(64 + i * 12, 12, 3, padding=1)"), "wrong", "fixed"),
    ("dense", _d(conv="nn.Conv2d(channels + i * growth, 12, 3, padding=1)"), "wrong", "fixed"),
    ("dense", _D_START.replace("self.layers.append(...)", "self.layers.append(nn.Conv2d(channels + i * growth, growth, 3, padding=1))"), "wrong", "dots-left"),
    ("dense", _d(join="y"), "error", "no-cat"),
    ("dense", _d(join="x + y"), "error", "plus"),
    ("dense", _d(join="torch.stack([x, y], dim=1)"), "error", "stack"),
    ("dense", _d(join="torch.cat([x, y], dim=0)"), "error", "dim"),
    ("dense", _d(join="torch.cat([x, y])"), "error", "dim"),
    ("dense", _d(join="torch.cat([x, y], dim=-1)"), "error", "dim"),
    ("dense", _d(join="torch.cat([x, y], dim=2)"), "error", "dim"),
    ("dense", _d(conv="nn.Conv2d(channels + i * growth, growth, 3)"), "error", "padding"),
    ("dense", _d(conv="nn.Conv2d(channels + i * growth, growth, 3, padding=2)"), "error", "padding"),
    ("dense", _d(conv="nn.Conv2d(channels, growth, 3, padding=1)"), "error", "channels"),
    ("dense", _d(conv="nn.Conv2d(channels + (i + 1) * growth, growth, 3, padding=1)"), "error", "channels"),
    ("dense", _d(conv="nn.Conv2d(channels + i * growth, channels, 3, padding=1)"), "error", "channels"),
    # the right shapes and parameters, but other numbers that no known mistake explains
    ("dense", _D.replace("y = torch.relu(layer(x))", "y = torch.sigmoid(layer(x))"), "almost", None),
] + [("dense", code, "wrong", key) for key, code in _D_WRONG.items()]


# --- the paper ------------------------------------------------------------------------------------------------------

# Huang, Liu, van der Maaten and Weinberger (2016), table 2: the error in % on CIFAR-10 with augmentation (C10+) and
# the parameters in millions, for the networks the lesson shows; the 0.8M DenseNet-BC matches the 1001-layer
# pre-activation ResNet "using 90% fewer parameters".
PAPER_C10 = {"preact-164": (1.7, 5.46), "preact-1001": (10.2, 4.62), "dense-40": (1.0, 5.24), "bc-100": (0.8, 4.51),
             "bc-190": (25.6, 3.46)}
# The slider's block with 4 layers, as the worked example makes it: the weights of its convolutions, without biases.
EXAMPLE_WEIGHTS = block_params(CHANNELS, GROWTH, 4) - 4 * GROWTH   # 35,424
EXAMPLE_WEIGHTS_24 = block_params(CHANNELS, 24, 4) - 4 * 24        # 86,400


# --- what the course measured ---------------------------------------------------------------------------------------

# What the course measured, on one thread, with the course's code, through run(): the middle of each variant
# (make_middle, with the course's DenseBlock) made after torch.manual_seed(seed), then the rest of WideCNN, trained as in
# cv-1x1-convolution (the Trainer of nn-trainer with AdamW, a learning rate of LR, a warmup of one epoch, cosine decay,
# max_norm 1.0, mini-batches of 64) on the first N_TRAIN training photos, shuffled after torch.manual_seed(seed + 1),
# for EPOCHS epochs and the seeds 0 to 4, with the photos normalized by channel_stats of the training photos. "train"
# and "test" are how many of the training photos and of the first N_TEST test photos are read right after the last
# epoch; "best" is the first epoch with the lowest validation loss, rounded to 3 decimals, and "lowest" that loss;
# "last_val" and "last_train" are the validation and the training loss of the last epoch. CURVES are seed 0. The runs
# with one 3×3 convolution in the middle ("plain") are those of cv-1x1-convolution, which builds them the same way.
# tests/test_cv_densenet.py measures seed 0 of DEFAULT again when EAI_CIFAR_FILE names the unpacked cifar-10.npz.
REFERENCE = {
    "plain": _ONE.REFERENCE["plain"],
    "chain": {"train": [3951, 4727, 3908, 4084, 4414], "test": [468, 515, 475, 459, 452], "best": [13, 14, 15, 12, 13],
              "lowest": [1.466, 1.408, 1.483, 1.474, 1.469], "last_val": [1.539, 1.582, 1.54, 1.579, 1.598],
              "last_train": [0.957, 0.635, 0.98, 0.918, 0.787]},
    "k24": {"train": [5975, 5990, 5918, 5986, 5983], "test": [584, 564, 553, 577, 574], "best": [7, 8, 7, 6, 7],
            "lowest": [1.286, 1.307, 1.334, 1.319, 1.318], "last_val": [2.051, 2.296, 2.049, 2.208, 2.143],
            "last_train": [0.06, 0.034, 0.102, 0.04, 0.037]},
    "k12": {"train": [5987, 5982, 5925, 5925, 5978], "test": [561, 573, 558, 569, 546], "best": [8, 7, 6, 6, 7],
            "lowest": [1.337, 1.291, 1.336, 1.358, 1.302], "last_val": [2.172, 2.259, 2.092, 2.124, 2.263],
            "last_train": [0.044, 0.048, 0.099, 0.097, 0.051]},
    "k6": {"train": [5990, 5949, 5955, 5985, 5957], "test": [557, 600, 559, 576, 570], "best": [3, 4, 8, 6, 6],
           "lowest": [1.378, 1.345, 1.297, 1.289, 1.295], "last_val": [2.481, 1.982, 2.194, 2.127, 2.14],
           "last_train": [0.033, 0.075, 0.071, 0.053, 0.068]},
}
CURVES = {
    "chain_val": [2.133, 1.879, 1.73, 1.673, 1.632, 1.62, 1.542, 1.549, 1.5, 1.499, 1.483, 1.472, 1.466, 1.475, 1.509,
                  1.524, 1.515, 1.539, 1.54, 1.539],
    "chain_train": [2.145, 1.881, 1.778, 1.707, 1.626, 1.585, 1.546, 1.494, 1.447, 1.409, 1.355, 1.304, 1.262, 1.2,
                    1.147, 1.093, 1.037, 0.997, 0.97, 0.957],
    "k24_val": [1.736, 1.5, 1.389, 1.412, 1.349, 1.337, 1.286, 1.352, 1.389, 1.349, 1.476, 1.504, 1.624, 1.715, 1.781,
                1.922, 1.996, 2.024, 2.045, 2.051],
    "k24_train": [1.925, 1.595, 1.429, 1.327, 1.233, 1.136, 1.061, 0.963, 0.875, 0.76, 0.647, 0.531, 0.426, 0.321,
                  0.227, 0.157, 0.111, 0.081, 0.065, 0.06],
    "k12_val": [1.736, 1.549, 1.458, 1.414, 1.521, 1.362, 1.39, 1.337, 1.338, 1.423, 1.502, 1.533, 1.625, 1.779, 1.848,
                1.95, 2.085, 2.145, 2.168, 2.172],
    "k12_train": [1.96, 1.608, 1.462, 1.338, 1.231, 1.131, 1.048, 0.961, 0.859, 0.771, 0.657, 0.53, 0.408, 0.294,
                  0.203, 0.136, 0.088, 0.06, 0.048, 0.044],
    "k6_val": [1.616, 1.538, 1.378, 1.48, 1.397, 1.46, 1.399, 1.4, 1.469, 1.503, 1.583, 1.703, 1.857, 1.955, 2.112,
               2.276, 2.421, 2.452, 2.475, 2.481],
    "k6_train": [1.936, 1.557, 1.403, 1.29, 1.199, 1.081, 1.022, 0.935, 0.803, 0.711, 0.588, 0.492, 0.345, 0.259,
                 0.179, 0.107, 0.066, 0.044, 0.036, 0.033],
}
DEFAULT = "k12"   # the demo's middle when the learner chooses none: the dense block of the slider and exercise 2
DENSE = ("k24", "k12", "k6")


def mean(values) -> float:
    return sum(values) / len(values)


def all_runs(key: str) -> list[int]:
    """What all the runs of the three dense blocks together read right: of the training photos or of the test photos."""
    return [v for k in DENSE for v in REFERENCE[k][key]]


# --- the picture of the runs ----------------------------------------------------------------------------------------

ROWS = ("plain", "chain", "k24", "k12", "k6")


def runs_picture(labels: dict, lang: str, mine: dict | None = None) -> str:
    """How many of the N_TEST test photos the network reads right after EPOCHS epochs, for each middle: one 3×3
    convolution of cv-1x1-convolution, the course's chain, and the three dense blocks with their transition; an open dot
    for every seed of the course and a line at their mean. With `mine` ({"variant", "test"}), the learner's run as a
    filled dot in the row of its middle. `labels` gives "title", "x", a name for each row, "mine", "course" and "alt"."""
    yours = {} if mine is None else {mine["variant"]: mine["test"]}
    # from 440 to 620, wider where the learner's run lies outside, so that every dot is where its number is
    low, high = min([440] + [v - 15 for v in yours.values()]), max([620] + [v + 15 for v in yours.values()])
    tick = 20 if high - low <= 200 else 50 if high - low <= 400 else 100
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

# The lesson is done when these are correct. "dense-runs" is the choice question in the lesson.
REQUIRED = ["blocks", "dense", "dense-runs"]


# --- the demo: the learner's DenseBlock in the middle of WideCNN, on CIFAR-10 ---------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def _from_block(e: BaseException) -> bool:
    """The error came from the learner's DenseBlock: a line of their dense.py is in its traceback."""
    mine = (projects.folder() / "cnn" / "dense.py").resolve()
    tb = e.__traceback__
    while tb is not None:
        if Path(tb.tb_frame.f_code.co_filename).resolve() == mine:
            return True
        tb = tb.tb_next
    return False


def _demo(variant: str, show=None):
    """For the demo: None when dense.py is not in the project yet and the variant needs it (the chain does not); "data"
    when CIFAR-10 is not unpacked; "dense" when the learner's DenseBlock stops with an error, also during the training,
    or does not give back the channels it should, of 8 by 8; "cnn" when channel_stats does not give what it should;
    "trainer" when the Trainer or count_params stops with an error; "slow" when the training takes too long; else the
    dict of run(), with the modules the course's versions stood in for."""
    dense = variant != "chain"
    if dense and not _saved("dense"):
        return None
    if not ready():
        return "data"
    DenseBlock = None
    if dense:
        loaded = run_code("from cnn.dense import DenseBlock")
        if loaded.error is not None:
            return "dense"
        DenseBlock = loaded.names["DenseBlock"]
        growth, layers = VARIANTS[variant]
        try:
            def tried():   # as in training: a mini-batch, in training mode, with a backward pass
                torch.manual_seed(0)
                block = DenseBlock(CHANNELS, growth, layers)
                out = block(torch.rand(4, CHANNELS, SIDE, SIDE))
                out.sum().backward()
                return tuple(out.shape)
            if with_time_limit(tried, 10) != (4, channels_out(CHANNELS, growth, layers), SIDE, SIDE):
                return "dense"
        except Exception:
            return "dense"
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
        result = with_time_limit(lambda: run(variant, 0, DenseBlock, channel_stats, Trainer, count_params, show), 600)
    except Timeout:
        return "slow"
    except Exception as e:
        return "dense" if dense and _from_block(e) else "trainer"
    return result | {"stand_ins": stand_ins}


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file), and whether CIFAR-10
    is there."""
    texts = []
    for package, module in (("cnn", "dense"), ("cnn", "channels"), ("cnn", "count"), ("nn", "train_epoch"), ("nn", "trainer")):
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
