# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the final project lesson cv-project-resnet-grad-cam: Block, the residual block of ResNet with batch
normalization, and a convolution of 1 by 1 on the residual connection where the shape changes; ResNet, a whole network
of the learner's Block in three stages; changed, a random crop with a black border and a mirror image for an image of
any size; and sure_mistakes, the wrong guesses a network was most sure of. Each is saved in the learner's project
(work/projects/cnn/block.py, deep_resnet.py, changes.py and mistakes.py) when right. The demo trains the learner's
ResNet on the first 5,000 training images of Fashion-MNIST, changed by their `changed` anew every time, with their
Trainer, watches the last 1,000 as its validation set, and tests it once on the 1,000 test images; the error analysis
after it counts the mistakes in a confusion matrix, and draws the learner's Grad-CAM of cv-grad-cam over the test images
their network was most sure of and got wrong."""
import copy
import html
import importlib.util
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from everything_ai import live, netview, paths, projects, yours
from everything_ai.check import ClassBox, FunctionBox, Mistake, Timeout, run_code, with_time_limit


def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_project",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_RES = _lesson_checks("cnn-resnet")   # Fashion-MNIST, the names of its kinds, and how numbers are written

# The demo trains with the learner's Trainer of nn-trainer, counts with their count_params of cnn-pytorch, builds their
# ResNet from their Block of this lesson, and explains it with their grad_cam of cv-grad-cam; where one is missing or
# broken, the course's version stands in (template 1.3).
BUILDS_ON = (("nn", "train_epoch", "nn-trainer", "train-epoch"), ("nn", "trainer", "nn-trainer", "trainer"),
             ("cnn", "count", "cnn-pytorch", "count-params"), ("cnn", "block", "cv-project-resnet-grad-cam", "block"),
             ("cnn", "gradcam", "cv-grad-cam", "grad-cam"))
projects.builds_on(*BUILDS_ON)

fashion = _RES.fashion        # the 6,000 training and 1,000 test images of Fashion-MNIST, from 0 to 1, as (N, 1, 28, 28)
KINDS = _RES.NAMES            # the 10 kinds, label 0 to 9, in both languages
grouped, number = _RES.grouped, _RES.number
N_TRAIN, N_TEST = 6_000, 1_000
WIDTH, BLOCKS = 16, 3         # the demo's ResNet: 16, 32 and 64 channels, 3 blocks per stage, 6 · 3 + 2 = 20 layers with weights
EPOCHS, LR = 20, 0.01         # as the course set them; with the crop, the validation loss is lowest after epoch 16 to 20 (REFERENCE)
PAD = 2                       # black pixels around an image of clothes before a crop: (2 · 2 + 1)² = 25 places
MIRROR = False                # no mirror images: chosen on the validation set, see REFERENCE
EVERY = 10                    # the live chart hears the training loss of every 10 batches (live.follow)
WATCHED = f"blocks.{3 * BLOCKS - 1}"   # the layer the demo shows while it trains, and Grad-CAM's layer: the last block
SURE = 8                      # the error analysis draws the 8 mistakes the network was most sure of
TOPS = (0, 2, 4, 6)           # T-shirt, pullover, coat and shirt: the clothes for the top of the body
SHOES = (5, 7, 9)             # sandal, sneaker and ankle boot


def one_each() -> list:
    """The first test image of each kind, in the order of the labels: what the demo shows its network guess."""
    return _RES.one_each()


# --- the course's versions of the exercises ------------------------------------------------------------------------

class CourseBlock(nn.Module):
    """The residual block of ResNet (He and others, 2015): f(x) is a convolution of 3 by 3 with `stride`, batch
    normalization, a ReLU, a convolution of 3 by 3 and batch normalization; the residual connection is x itself, or a
    convolution of 1 by 1 with `stride` and batch normalization where the shape changes; the block gives the ReLU of the
    sum."""
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1)
        self.norm1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(c_out)
        self.skip = None
        if stride != 1 or c_in != c_out:
            self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride)
            self.skip_norm = nn.BatchNorm2d(c_out)

    def forward(self, x):
        f = self.norm2(self.conv2(torch.relu(self.norm1(self.conv1(x)))))
        if self.skip is not None:
            x = self.skip_norm(self.skip(x))
        return torch.relu(x + f)


def course_resnet(channels: int, width: int, blocks: int, kinds: int, block=CourseBlock) -> nn.Module:
    """The course's ResNet, made in the order of the exercise's: a stem from `channels` to `width` with batch
    normalization, three stages of `blocks` blocks with width, 2 · width and 4 · width channels, the first block of
    stage 2 and 3 with stride 2, global average pooling and a linear layer to `kinds` logits. `block` is the block it is
    made of."""
    class ResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(channels, width, 3, padding=1)
            self.stem_norm = nn.BatchNorm2d(width)
            layers = []
            for stage in range(3):
                c = width * 2 ** stage
                for b in range(blocks):
                    layers.append(block(c // 2, c, 2) if stage > 0 and b == 0 else block(c, c, 1))
            self.blocks = nn.ModuleList(layers)
            self.out = nn.Linear(4 * width, kinds)

        def forward(self, x):
            x = torch.relu(self.stem_norm(self.stem(x)))
            for b in self.blocks:
                x = b(x)
            return self.out(x.mean(dim=(2, 3)))
    return ResNet()


def course_changed(x, u, pad, mirror):
    """Image x of shape (C, H, W), mirrored when `mirror` and u[0] < 0.5, then cut out at the place u[1] and u[2] give
    from the image with a black border of `pad` pixels: a new tensor of shape (C, H, W)."""
    if mirror and u[0] < 0.5:
        x = x.flip(dims=[2])
    c, h, w = x.shape
    padded = torch.zeros(c, h + 2 * pad, w + 2 * pad)
    padded[:, pad:pad + h, pad:pad + w] = x
    top, left = int(u[1] * (2 * pad + 1)), int(u[2] * (2 * pad + 1))
    return padded[:, top:top + h, left:left + w]


def course_sure_mistakes(logits, labels, n):
    """The places of the n wrong guesses with the highest probability, most sure first: a list of whole numbers."""
    p = torch.softmax(logits, dim=1)
    guess = p.argmax(dim=1)
    pairs = [(p[i, guess[i]].item(), i) for i in range(len(labels)) if guess[i] != labels[i]]
    return [pair[1] for pair in sorted(pairs, reverse=True)[:n]]


class Changed(Dataset):
    """The training images, each changed anew every time it is asked for: three numbers from torch.rand, then the
    learner's `changed` with them, and normalized with the mean and the standard deviation of the training images as
    they are."""

    def __init__(self, X, y, mean, std, changed, pad=PAD, mirror=False):
        self.X, self.y, self.mean, self.std = X, y, mean, std
        self.changed, self.pad, self.mirror = changed, pad, mirror

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.changed(self.X[i].clone(), torch.rand(3), self.pad, self.mirror)
        return (torch.as_tensor(x, dtype=torch.float32) - self.mean) / self.std, self.y[i]


N_FIT = 5_000   # the network learns from training images 0 to 4,999; 5,000 to 5,999 are its validation set


def splits() -> tuple:
    """The images the network learns from, its validation set and the test images, each with its labels: the first
    5,000 training images, the last 1,000, and the 1,000 test images."""
    X, y, X_test, y_test = fashion()
    return X[:N_FIT], y[:N_FIT], X[N_FIT:], y[N_FIT:], X_test, y_test


def mean_std() -> tuple:
    """The mean and the standard deviation of all pixels of the images the network learns from, as they are."""
    X = splits()[0]
    return float(X.mean()), float(X.std())


# The lesson is done when these are correct. "why-wrong" is the choice question in the lesson, "report" the open one.
REQUIRED = ["block", "resnet", "changes", "sure-mistakes", "why-wrong", "report"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$|=\s*\.\.\.|\breturn\s+\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_C_EN = "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_C_NL = "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"
_F_EN = "With `{call}`, your function gives `{got}`, and it should be `{want}`"
_F_NL = "Met `{call}` geeft je functie `{got}`, en dat moet `{want}` zijn"


def _close(got, want) -> bool:
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    if isinstance(want, float):
        return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - want) <= 1e-4 * max(1.0, abs(want))
    return got == want


def _sizes(e) -> bool:
    return isinstance(e, RuntimeError) and "must match the size" in str(e)


# --- exercise 1: the block of ResNet ---------------------------------------------------------------------------------

_B_HEAD = ("import torch\nfrom torch import nn\n\n\nclass Block(nn.Module):\n    def __init__(self, c_in, c_out, stride):\n"
           "        super().__init__()\n")
_B_INIT = ("        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1)\n"
           "        self.norm1 = nn.BatchNorm2d(c_out)\n"
           "        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)\n"
           "        self.norm2 = nn.BatchNorm2d(c_out)\n")
_B_SKIP_HEAD = "        self.skip = None\n        if stride != 1 or c_in != c_out:\n"
_B_SKIP = ("            self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride)\n"
           "            self.skip_norm = nn.BatchNorm2d(c_out)\n")
_B_FORWARD = ("        f = self.norm2(self.conv2(torch.relu(self.norm1(self.conv1(x)))))\n"
              "        if self.skip is not None:\n            x = self.skip_norm(self.skip(x))\n"
              "        return torch.relu(x + f)\n")
_B_START = (_B_HEAD
            + "        # conv1: 3 by 3 from c_in to c_out channels, this stride, padding 1\n"
            + "        # norm1: batch normalization of c_out\n"
            + "        # conv2: 3 by 3 from c_out to c_out channels, padding 1; norm2: batch normalization of c_out\n"
            + "        ...\n" + _B_SKIP_HEAD
            + "            # skip: 1 by 1 from c_in to c_out channels, this stride; skip_norm: batch normalization of c_out\n"
            + "            ...\n\n    def forward(self, x):\n"
            + "        # f(x): conv1, norm1, ReLU, conv2, norm2\n"
            + "        # the residual connection: x itself, or skip_norm(skip(x)) when the block has a skip\n"
            + "        # the block gives the ReLU of the sum\n        ...\n")


def _b(init: str = _B_INIT, skip: str = _B_SKIP, forward: str = _B_FORWARD) -> str:
    return _B_HEAD + init + _B_SKIP_HEAD + skip + "\n    def forward(self, x):\n" + forward


_B = _b()
_B_SCENARIOS = [
    # a block where a stage changes, from 4 to 8 channels with stride 2, on 2 feature maps of 6 by 6
    "import torch\n\ntorch.manual_seed(0)\nblock = Block(4, 8, 2)\ny = block(torch.randn(2, 4, 6, 6))\n"
    "[list(y.shape), y[0, :4, 0, 0].tolist()]",
    # a block inside a stage: the same channels and size, and x itself on the residual connection
    "import torch\n\ntorch.manual_seed(0)\nblock = Block(4, 4, 1)\ny = block(torch.randn(2, 4, 5, 5))\n"
    "[list(y.shape), y[1, :4, 2, 2].tolist()]",
    # an odd size: 7 by 7 becomes 4 by 4 on both paths
    "import torch\n\ntorch.manual_seed(0)\nblock = Block(3, 6, 2)\nlist(block(torch.randn(2, 3, 7, 7)).shape)",
    # which blocks get a convolution on the residual connection
    "[Block(8, 8, 1).skip is None, Block(8, 16, 2).skip is None, Block(8, 8, 2).skip is None, Block(8, 16, 1).skip is None]",
    # the gradient goes back through both paths
    "import torch\n\ntorch.manual_seed(0)\nblock = Block(3, 3, 1)\na = torch.randn(2, 3, 6, 6, requires_grad=True)\n"
    "block(torch.relu(a)).sum().backward()\nround(a.grad.abs().sum().item(), 4)",
    # what PyTorch sees of a block from 16 to 32 channels and of a block of 32, as in the demo
    "block = Block(16, 32, 2)\nother = Block(32, 32, 1)\n[len(list(block.parameters())), sum(p.numel() for p in block.parameters()), "
    "len(list(other.parameters())), sum(p.numel() for p in other.parameters())]",
]
_B_WRONG = {
    "stride-second": _b(init=_B_INIT.replace("3, stride=stride, padding=1)", "3, padding=1)").replace(
        "nn.Conv2d(c_out, c_out, 3, padding=1)", "nn.Conv2d(c_out, c_out, 3, stride=stride, padding=1)")),
    "no-relu-end": _b(forward=_B_FORWARD.replace("return torch.relu(x + f)", "return x + f")),
    "relu-in-f": _b(forward=_B_FORWARD.replace("f = self.norm2(", "f = torch.relu(self.norm2(").replace("(x)))))\n", "(x))))))\n", 1)),
    "no-norm": _b(forward=_B_FORWARD.replace("self.norm2(self.conv2(torch.relu(self.norm1(self.conv1(x)))))",
                                             "self.conv2(torch.relu(self.conv1(x)))")),
    "norm-after-relu": _b(forward=_B_FORWARD.replace("torch.relu(self.norm1(self.conv1(x)))", "self.norm1(torch.relu(self.conv1(x)))")),
    "skip-no-norm": _b(forward=_B_FORWARD.replace("x = self.skip_norm(self.skip(x))", "x = self.skip(x)")),
}
_B_MISTAKES = [
    Mistake("sizes", error=_sizes,
            en="`x` and `f(x)` have different shapes, so `+` cannot add them. Where the block changes the channels or the size, the residual connection gets the same change: `skip`, a convolution of 1 by 1 from `c_in` to `c_out` channels with this `stride`, then `skip_norm`; and `conv1` needs `padding=1`, so that both halve an odd size in the same way.",
            nl="`x` en `f(x)` hebben verschillende shapes, dus `+` kan ze niet optellen. Waar het block de channels of de maat verandert, krijgt de residual connection dezelfde verandering: `skip`, een convolutie van 1 bij 1 van `c_in` naar `c_out` channels met deze `stride`, en dan `skip_norm`; en `conv1` heeft `padding=1` nodig, zodat ze allebei een oneven maat op dezelfde manier halveren."),
    Mistake("channels", error=lambda e: isinstance(e, RuntimeError) and re.search(r"to have \d+ channels, but got \d+ channels", str(e)) is not None,
            en="One of your convolutions gets another number of channels than it was made for. `conv1` and `skip` get `x`, with `c_in` channels; `conv2` gets what comes out of `conv1`, with `c_out`.",
            nl="Een van je convoluties krijgt een ander aantal channels dan waarvoor hij gemaakt is. `conv1` en `skip` krijgen `x`, met `c_in` channels; `conv2` krijgt wat uit `conv1` komt, met `c_out`."),
    Mistake("norm-size", error=lambda e: isinstance(e, RuntimeError) and "running_mean should contain" in str(e),
            en="One of your batch normalizations gets another number of channels than it was made for. Each comes after a convolution to `c_out` channels, so each is `nn.BatchNorm2d(c_out)`.",
            nl="Een van je batch normalizations krijgt een ander aantal channels dan waarvoor hij gemaakt is. Elk komt na een convolutie naar `c_out` channels, dus elk is `nn.BatchNorm2d(c_out)`."),
    Mistake("in-place", error=lambda e: isinstance(e, RuntimeError) and "inplace operation" in str(e),
            en="The backward pass stops: your block changes a tensor itself, as `+=` does, and PyTorch still needs it as it was on the way back. Make a new tensor: `return torch.relu(x + f)`.",
            nl="De backward pass stopt: je block verandert een tensor zelf, zoals `+=` doet, en PyTorch heeft hem op de terugweg nog nodig zoals hij was. Maak een nieuwe tensor: `return torch.relu(x + f)`."),
    Mistake("stride-second", wrong=_B_WRONG["stride-second"],
            en=_C_EN + ": your `conv2` halves the size, and `conv1` keeps it. The shapes fit, but ResNet halves in `conv1`: then `conv1` already computes only a quarter of the places, and `conv2` works on the small map.",
            nl=_C_NL + ": je `conv2` halveert de maat, en `conv1` houdt hem gelijk. De shapes passen, maar ResNet halveert in `conv1`: dan rekent `conv1` al maar een kwart van de plaatsen uit, en werkt `conv2` op de kleine map."),
    Mistake("no-relu-end", wrong=_B_WRONG["no-relu-end"],
            en=_C_EN + ": the ReLU after the addition is missing. In ResNet, the block gives the ReLU of the sum: `torch.relu(x + f)`.",
            nl=_C_NL + ": de ReLU na het optellen ontbreekt. In ResNet geeft het block de ReLU van de som: `torch.relu(x + f)`."),
    Mistake("relu-in-f", wrong=_B_WRONG["relu-in-f"],
            en=_C_EN + ": you put a ReLU at the end of `f`, before the addition. `f` ends with `norm2`, so that it can also make numbers below 0 and take something away from `x`; the one ReLU comes after the sum.",
            nl=_C_NL + ": je zet een ReLU aan het eind van `f`, vóór het optellen. `f` eindigt met `norm2`, zodat het ook getallen onder 0 kan maken en iets van `x` kan afhalen; de ene ReLU komt na de som."),
    Mistake("no-norm", wrong=_B_WRONG["no-norm"],
            en=_C_EN + ": your `f` does not use `norm1` and `norm2`. Each convolution is followed by its batch normalization: `self.norm1(self.conv1(x))`, and the same with `conv2` and `norm2`.",
            nl=_C_NL + ": je `f` gebruikt `norm1` en `norm2` niet. Na elke convolutie komt zijn batch normalization: `self.norm1(self.conv1(x))`, en net zo met `conv2` en `norm2`."),
    Mistake("norm-after-relu", wrong=_B_WRONG["norm-after-relu"],
            en=_C_EN + ": your `norm1` comes after the ReLU. ResNet puts batch normalization right after each convolution, before the ReLU: `torch.relu(self.norm1(self.conv1(x)))`.",
            nl=_C_NL + ": je `norm1` komt na de ReLU. ResNet zet batch normalization direct na elke convolutie, vóór de ReLU: `torch.relu(self.norm1(self.conv1(x)))`."),
    Mistake("skip-no-norm", wrong=_B_WRONG["skip-no-norm"],
            en=_C_EN + ": on the residual connection, `skip_norm` is missing. The convolution of 1 by 1 is a convolution too, so batch normalization comes right after it: `self.skip_norm(self.skip(x))`.",
            nl=_C_NL + ": op de residual connection ontbreekt `skip_norm`. De convolutie van 1 bij 1 is ook een convolutie, dus direct erna komt batch normalization: `self.skip_norm(self.skip(x))`."),
]


# --- exercise 2: the whole ResNet ------------------------------------------------------------------------------------

_N_HEAD = ("import torch\nfrom torch import nn\nfrom cnn.block import Block\n\n\n"
           "class ResNet(nn.Module):\n    def __init__(self, channels, width, blocks, kinds):\n        super().__init__()\n"
           "        self.stem = nn.Conv2d(channels, width, 3, padding=1)\n        self.stem_norm = nn.BatchNorm2d(width)\n"
           "        layers = []\n")
_N_STAGES = ("        for stage in range(3):\n            c = width * 2 ** stage\n            for b in range(blocks):\n"
             "                if stage > 0 and b == 0:\n                    layers.append(Block(c // 2, c, 2))\n"
             "                else:\n                    layers.append(Block(c, c, 1))\n")
_N_TAIL = "        self.blocks = nn.ModuleList(layers)\n        self.out = nn.Linear(4 * width, kinds)\n"
_N_FORWARD = ("        x = torch.relu(self.stem_norm(self.stem(x)))\n        for block in self.blocks:\n            x = block(x)\n"
              "        return self.out(x.mean(dim=(2, 3)))\n")
_N_START = (_N_HEAD + "        # three stages of `blocks` blocks: width, 2 * width and 4 * width channels;\n"
            "        # the first block of stage 2 and of stage 3 comes from the channels before, with stride 2\n"
            "        ...\n" + _N_TAIL + "\n    def forward(self, x):\n"
            "        # the stem, stem_norm and a ReLU, every block, global average pooling, the output layer\n        ...\n")


def _n(stages: str = _N_STAGES, tail: str = _N_TAIL, forward: str = _N_FORWARD) -> str:
    return _N_HEAD + stages + tail + "\n    def forward(self, x):\n" + forward


_N = _n()
_N_SCENARIOS = [
    # a small ResNet on 2 images of clothes: the shape of the logits, and 4 of the first
    "import torch\n\ntorch.manual_seed(0)\nnet = ResNet(1, 4, 2, 10)\ny = net(torch.randn(2, 1, 28, 28))\n[list(y.shape), y[0, :4].tolist()]",
    # which block stands where, with 3 blocks per stage: its channels in and out, and its stride
    "net = ResNet(1, 4, 3, 10)\n[[block.conv1.in_channels, block.conv1.out_channels, block.conv1.stride[0]] for block in net.blocks]",
    # the maps of the last block, the layer of Grad-CAM: for an image of 28 by 28, 7 by 7
    "import torch\n\nmaps = []\nnet = ResNet(1, 8, 2, 10)\nnet.blocks[-1].register_forward_hook(lambda module, inputs, output: maps.append(output))\n"
    "with torch.no_grad():\n    net(torch.randn(2, 1, 28, 28))\nlist(maps[0].shape)",
    # color photos of 32 by 32 and 5 kinds
    "import torch\n\ntorch.manual_seed(0)\nnet = ResNet(3, 4, 1, 5)\ny = net(torch.randn(2, 3, 32, 32))\n[list(y.shape), y[0].tolist()]",
    # what PyTorch sees of the demo's ResNet
    "net = ResNet(1, 16, 3, 10)\n[len(list(net.parameters())), sum(p.numel() for p in net.parameters())]",
]
_N_WRONG = {
    "no-stride": _n(stages=_N_STAGES.replace("Block(c // 2, c, 2)", "Block(c // 2, c, 1)")),
    "max-pool": _n(forward=_N_FORWARD.replace("x.mean(dim=(2, 3))", "x.amax(dim=(2, 3))")),
    "no-stem-relu": _n(forward=_N_FORWARD.replace("x = torch.relu(self.stem_norm(self.stem(x)))", "x = self.stem_norm(self.stem(x))")),
    "no-stem-norm": _n(forward=_N_FORWARD.replace("x = torch.relu(self.stem_norm(self.stem(x)))", "x = torch.relu(self.stem(x))")),
    "plain-list": _n(tail="        self.blocks = layers\n        self.out = nn.Linear(4 * width, kinds)\n"),
}
_N_MISTAKES = [
    Mistake("flatten", error=lambda e: isinstance(e, RuntimeError) and "cannot be multiplied" in str(e),
            en="The sizes do not fit: your output layer gets another number of inputs per image than it was made for, `4 * width`. Global average pooling makes one number of each map: the mean over the height and the width, `x.mean(dim=(2, 3))`, shape (N, 4 · width), whatever the size of the image.",
            nl="De maten passen niet: je output layer krijgt per afbeelding een ander aantal inputs dan waarvoor hij gemaakt is, `4 * width`. Global average pooling maakt van elke map één getal: het gemiddelde over de hoogte en de breedte, `x.mean(dim=(2, 3))`, shape (N, 4 · width), wat de maat van de afbeelding ook is."),
    Mistake("channels", error=lambda e: isinstance(e, RuntimeError) and re.search(r"to have \d+ channels, but got \d+ channels", str(e)) is not None,
            en="A block gets another number of channels than it was made for. Stage 1 has `width` channels, stage 2 `2 * width` and stage 3 `4 * width`: the first block of a stage goes from the channels of the stage before, `c // 2`, to those of its own, `c`.",
            nl="Een block krijgt een ander aantal channels dan waarvoor het gemaakt is. Stage 1 heeft `width` channels, stage 2 `2 * width` en stage 3 `4 * width`: het eerste block van een stage gaat van de channels van de stage ervoor, `c // 2`, naar die van zijn eigen, `c`."),
    Mistake("no-stride", wrong=_N_WRONG["no-stride"],
            en=_C_EN + ": your first block of stage 2 and of stage 3 keeps the size of the map. It halves the height and the width, with stride 2: `Block(c // 2, c, 2)`. Then every stage works on a map half as high and half as wide as the one before, and the last maps are 7 by 7.",
            nl=_C_NL + ": je eerste block van stage 2 en van stage 3 houdt de maat van de map gelijk. Het halveert de hoogte en de breedte, met stride 2: `Block(c // 2, c, 2)`. Dan werkt elke stage op een map die half zo hoog en half zo breed is als die ervoor, en zijn de laatste maps 7 bij 7."),
    Mistake("max-pool", wrong=_N_WRONG["max-pool"],
            en=_C_EN + ": you take the largest number of each map. Global average pooling takes the mean: `.mean(dim=(2, 3))`, as at the end of ResNet-18 in [Grad-CAM: where a CNN looked](/course/open/cv-grad-cam?lang=en).",
            nl=_C_NL + ": je neemt het grootste getal van elke map. Global average pooling neemt het gemiddelde: `.mean(dim=(2, 3))`, zoals aan het eind van ResNet-18 in [Grad-CAM: waar een CNN keek](/course/open/cv-grad-cam?lang=nl)."),
    Mistake("no-stem-relu", wrong=_N_WRONG["no-stem-relu"],
            en=_C_EN + ": the ReLU after the stem is missing. Put `torch.relu` around `self.stem_norm(self.stem(x))`.",
            nl=_C_NL + ": de ReLU na de stem ontbreekt. Zet `torch.relu` om `self.stem_norm(self.stem(x))`."),
    Mistake("no-stem-norm", wrong=_N_WRONG["no-stem-norm"],
            en=_C_EN + ": your stem does not use `stem_norm`. Like every convolution of ResNet, the stem is followed by batch normalization, then the ReLU: `torch.relu(self.stem_norm(self.stem(x)))`.",
            nl=_C_NL + ": je stem gebruikt `stem_norm` niet. Zoals na elke convolutie van ResNet komt na de stem batch normalization, en dan de ReLU: `torch.relu(self.stem_norm(self.stem(x)))`."),
    Mistake("plain-list", wrong=_N_WRONG["plain-list"],
            en=_C_EN + ": PyTorch does not see the blocks in a plain list, so their parameters are not counted, and an optimizer would never train them. Put them in an `nn.ModuleList(layers)`.",
            nl=_C_NL + ": PyTorch ziet de blocks in een gewone list niet, dus hun parameters worden niet geteld, en een optimizer zou ze nooit trainen. Zet ze in een `nn.ModuleList(layers)`."),
]


# --- exercise 3: changes for any image -------------------------------------------------------------------------------

_X_HEAD = "import torch\n\n\ndef changed(x, u, pad, mirror):\n"
_X_START = (_X_HEAD + "    # x: an image of shape (C, H, W); u: three random numbers from 0 to 1\n"
            "    # pad: the width of the black border; mirror: whether these images may be mirrored\n"
            "    # 1. mirror when mirror is True and u[0] < 0.5\n"
            "    # 2. x in a black image with pad pixels on every side; top from u[1], left from u[2]\n"
            "    # 3. give back the H by W that starts at row top and column left\n    ...\n")
_X_MIRROR = "    if mirror and u[0] < 0.5:\n        x = x.flip(dims=[2])\n"
_X_PAD = ("    c, h, w = x.shape\n    padded = torch.zeros(c, h + 2 * pad, w + 2 * pad)\n"
          "    padded[:, pad:pad + h, pad:pad + w] = x\n")
_X_PLACE = "    top, left = int(u[1] * (2 * pad + 1)), int(u[2] * (2 * pad + 1))\n"
_X_CUT = "    return padded[:, top:top + h, left:left + w]\n"


def _x(mirror: str = _X_MIRROR, pad: str = _X_PAD, place: str = _X_PLACE, cut: str = _X_CUT) -> str:
    return _X_HEAD + mirror + pad + place + cut


_X = _x()
_SMALL = torch.arange(1.0, 17.0).reshape(1, 4, 4) / 16            # a made-up gray image of 4 by 4: 1/16 to 16/16
_COLOR = torch.arange(1.0, 91.0).reshape(3, 5, 6) / 90            # a made-up color image of 5 by 6


def _u(a: float, b: float, c: float) -> torch.Tensor:
    return torch.tensor([a, b, c])


_X_VISIBLE = [(_SMALL, _u(0.2, 0.1, 0.9), 1, True)]
_X_HIDDEN = [
    (_SMALL, _u(0.7, 0.5, 0.5), 1, True),      # not mirrored, and cut from the middle: the image itself
    (_SMALL, _u(0.2, 0.5, 0.5), 1, False),     # mirror off: the image itself, whatever u[0] is
    (_COLOR, _u(0.3, 0.99, 0.0), 2, True),     # color, of 5 by 6, mirrored, from row 4 and column 0
    (_COLOR, _u(0.9, 0.3, 0.7), 2, False),     # from row 1 and column 3
    (_SMALL, _u(0.1, 0.4, 0.8), 0, True),      # no border: only the mirror image
]
_X_WRONG = {
    "always-mirror": _x(mirror="    if u[0] < 0.5:\n        x = x.flip(dims=[2])\n"),
    "upside-down": _x(mirror=_X_MIRROR.replace("dims=[2]", "dims=[1]")),
    "swapped": _x(place="    top, left = int(u[2] * (2 * pad + 1)), int(u[1] * (2 * pad + 1))\n"),
    "places": _x(place="    top, left = int(u[1] * 2 * pad), int(u[2] * 2 * pad)\n"),
    "no-mirror": _x(mirror=""),
}
_FIXED_EN = ("With `{call}`, the image does not fit your black image. Make that for any image: `c, h, w = x.shape`, and "
             "`torch.zeros(c, h + 2 * pad, w + 2 * pad)`; the image goes in at rows `pad` to `pad + h` and columns `pad` to `pad + w`.")
_FIXED_NL = ("Met `{call}` past de afbeelding niet bij je zwarte afbeelding. Maak die voor elke afbeelding: `c, h, w = x.shape`, en "
             "`torch.zeros(c, h + 2 * pad, w + 2 * pad)`; de afbeelding komt erin bij rij `pad` tot `pad + h` en kolom `pad` tot `pad + w`.")
_FIXED = re.compile(r"torch\.zeros\(\s*\(?\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)?\s*\)")   # a black image of one size, such as (3, 40, 40)
_X_MISTAKES = [
    Mistake("fixed-size", error=lambda e: isinstance(e, RuntimeError) and ("must match" in str(e) or "shape mismatch" in str(e)),
            en=_FIXED_EN, nl=_FIXED_NL),
    Mistake("fixed-size", matches=lambda got, text: _FIXED.search(text) is not None, en=_FIXED_EN, nl=_FIXED_NL),
    Mistake("always-mirror", wrong=_X_WRONG["always-mirror"],
            en=_F_EN + ": your function mirrors also when `mirror` is False. Mirror only when both hold: `mirror and u[0] < 0.5`.",
            nl=_F_NL + ": je functie spiegelt ook als `mirror` False is. Spiegel alleen als allebei gelden: `mirror and u[0] < 0.5`."),
    Mistake("no-mirror", wrong=_X_WRONG["no-mirror"],
            en=_F_EN + ": your function never mirrors. When `mirror` is True and `u[0]` is below 0.5, it mirrors the image first: `x.flip(dims=[2])`.",
            nl=_F_NL + ": je functie spiegelt nooit. Als `mirror` True is en `u[0]` onder 0,5, spiegelt hij de afbeelding eerst: `x.flip(dims=[2])`."),
    Mistake("upside-down", wrong=_X_WRONG["upside-down"],
            en=_F_EN + ": your function turns the image upside down: `dims=[1]` puts the rows in the other order. A mirror image puts the columns in the other order, axis 2 of (C, H, W): `x.flip(dims=[2])`.",
            nl=_F_NL + ": je functie zet de afbeelding op zijn kop: `dims=[1]` zet de rijen in de andere volgorde. Een spiegelbeeld zet de kolommen in de andere volgorde, axis 2 van (C, H, W): `x.flip(dims=[2])`."),
    Mistake("swapped", wrong=_X_WRONG["swapped"],
            en=_F_EN + ": your rows and columns are swapped. `top`, the row where the cut starts, comes from `u[1]`, and `left`, its column, from `u[2]`.",
            nl=_F_NL + ": je rijen en kolommen zijn omgewisseld. `top`, de rij waar de uitsnede begint, komt uit `u[1]`, en `left`, zijn kolom, uit `u[2]`."),
    Mistake("places", wrong=_X_WRONG["places"],
            en=_F_EN + ": the cut can start at rows 0 to 2 · pad, which are 2 · pad + 1 places, and so can its column. `int(u[1] * 2 * pad)` gives one fewer: the last row, and the last column, never come.",
            nl=_F_NL + ": de uitsnede kan beginnen bij rij 0 tot en met 2 · pad, dat zijn 2 · pad + 1 plekken, en zijn kolom ook. `int(u[1] * 2 * pad)` geeft er één minder: de laatste rij, en de laatste kolom, komen nooit."),
]


# --- exercise 4: the mistakes the network was most sure of -----------------------------------------------------------

_M_HEAD = "import torch\n\n\ndef sure_mistakes(logits, labels, n):\n"
_M_START = (_M_HEAD + "    # logits: shape (N, K), a row per image; labels: the right kind of each image, shape (N,)\n"
            "    # give back the places of the n wrong guesses with the highest probability, most sure first\n    ...\n")
_M_PROBS = "    p = torch.softmax(logits, dim=1)\n    guess = p.argmax(dim=1)\n"
_M_PAIRS = "    pairs = [(p[i, guess[i]].item(), i) for i in range(len(labels)) if guess[i] != labels[i]]\n"
_M_END = "    return [pair[1] for pair in sorted(pairs, reverse=True)[:n]]\n"


def _m(probs: str = _M_PROBS, pairs: str = _M_PAIRS, end: str = _M_END) -> str:
    return _M_HEAD + probs + pairs + end


_M = _m()
_L1 = torch.tensor([[2.0, 0.5, 0.1], [0.2, 3.0, 0.1], [1.0, 1.2, 0.9], [0.1, 0.3, 4.0], [2.5, 2.4, 0.0], [0.0, 0.2, 1.5]])
_Y1 = torch.tensor([0, 0, 2, 1, 1, 2])
# image 0 has the largest logit of the wrong ones, 3.0, with a close second; image 2 a smaller logit, 2.0, and is surer
_L2 = torch.tensor([[3.0, 2.9, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [2.0, -1.0, -1.0, -1.0], [0.0, 0.0, 0.5, 0.4], [0.0, 0.0, 0.0, 5.0]])
_Y2 = torch.tensor([1, 1, 3, 1, 3])
_M_VISIBLE = [(_L1, _Y1, 2)]
_M_HIDDEN = [
    (_L1, _Y1, 10),           # more asked for than there are mistakes: all 4, most sure first
    (_L2, _Y2, 3),            # a large logit with a close second is less sure than a smaller one alone
    (_L2, _Y2, 0),            # none asked for
    (_L1[[0, 5]], _Y1[[0, 5]], 3),   # no mistakes at all
]
_M_WRONG = {
    "by-logit": _m(pairs=_M_PAIRS.replace("p[i, guess[i]]", "logits[i, guess[i]]")),
    "least-first": _m(end=_M_END.replace(", reverse=True", "")),
    "with-right": _m(pairs="    pairs = [(p[i, guess[i]].item(), i) for i in range(len(labels))]\n"),
    "by-right": _m(pairs=_M_PAIRS.replace("(p[i, guess[i]].item(), i)", "(-p[i, labels[i]].item(), i)")),
    "all": _m(end="    return [pair[1] for pair in sorted(pairs, reverse=True)]\n"),
}
_M_MISTAKES = [
    Mistake("by-logit", wrong=_M_WRONG["by-logit"],
            en=_F_EN + ": you sort by the largest logit. How sure the network is, is the probability of its guess, after softmax: a large logit with a second one close to it gives a probability of about one half, and a smaller logit far above all others gives more.",
            nl=_F_NL + ": je sorteert op de grootste logit. Hoe zeker het netwerk is, is de kans van zijn gok, na softmax: een grote logit met een tweede er vlak bij geeft een kans van ongeveer een half, en een kleinere logit ver boven alle andere geeft meer."),
    Mistake("least-first", wrong=_M_WRONG["least-first"],
            en=_F_EN + ": your list starts with the mistake the network was least sure of. The most sure comes first: sort the pairs from the highest probability down, with `sorted(pairs, reverse=True)`.",
            nl=_F_NL + ": je lijst begint met de fout waarvan het netwerk het minst zeker was. De zekerste komt eerst: sorteer de paren van de hoogste kans naar beneden, met `sorted(pairs, reverse=True)`."),
    Mistake("with-right", wrong=_M_WRONG["with-right"],
            en=_F_EN + ": your list holds images the network got right too. Keep only the places where its guess, the kind with the highest probability, is not the label.",
            nl=_F_NL + ": je lijst bevat ook afbeeldingen die het netwerk goed had. Houd alleen de plekken waar zijn gok, de soort met de hoogste kans, niet het label is."),
    Mistake("by-right", wrong=_M_WRONG["by-right"],
            en=_F_EN + ": you sort by how little probability the right kind got. That is another way to look at mistakes; this function sorts by how sure the network was of its own guess, the highest probability in the row.",
            nl=_F_NL + ": je sorteert op hoe weinig kans de goede soort kreeg. Dat is een andere manier om naar fouten te kijken; deze functie sorteert op hoe zeker het netwerk van zijn eigen gok was, de hoogste kans in de rij."),
    Mistake("all", wrong=_M_WRONG["all"],
            en=_F_EN + ": your list holds all mistakes. Give back only the first `n`: `[:n]` after sorting.",
            nl=_F_NL + ": je lijst bevat alle fouten. Geef alleen de eerste `n` terug: `[:n]` na het sorteren."),
]


def _places(got, want) -> bool:
    """The same places in the same order: a list, a tuple or a tensor of whole numbers."""
    if isinstance(got, torch.Tensor):
        if got.dim() != 1 or got.dtype.is_floating_point:
            return False
        got = got.tolist()
    if not isinstance(got, (list, tuple)) or len(got) != len(want):
        return False
    out = []
    for g in got:
        if isinstance(g, torch.Tensor) and g.dim() == 0 and not g.dtype.is_floating_point:
            g = int(g)
        if isinstance(g, bool) or not isinstance(g, (int, np.integer)):
            return False
        out.append(int(g))
    return out == list(want)


def _same_image(got, want) -> bool:
    """The same tensor of numbers, within float rounding."""
    if not isinstance(got, torch.Tensor):
        return False
    return tuple(got.shape) == tuple(want.shape) and bool(torch.allclose(got.float(), want.float(), atol=1e-6))


# --- the pictures under the checks -----------------------------------------------------------------------------------

def _sneaker() -> torch.Tensor:
    """The first test image of a sneaker, (1, 28, 28): the toe points left, as in all of them."""
    _, _, X_test, _ = fashion()
    return X_test[one_each()[7]].clone()


# (u, mirror): four with mirror True, and one with mirror False and u[0] below 0.5, as the demo trains
_SHOWN_CHANGES = [((0.2, 0.0, 0.0), True), ((0.2, 1.0 - 1e-6, 1.0 - 1e-6), True), ((0.7, 0.0, 1.0 - 1e-6), True),
                  ((0.7, 0.5, 0.5), True), ((0.2, 0.5, 0.5), False)]


def changes_picture(seen) -> str:
    """What the learner's `changed` makes of a sneaker of the test images: the sneaker, then four versions with `pad` 2
    and `mirror` True and one with `mirror` False, with the course's version under any that differs. Nothing when their
    function does not run."""
    mine = seen.mine
    fn = mine.names.get("changed") if mine else None
    if not callable(fn):
        return ""
    x = _sneaker()
    rows, other = [], []
    for u, mirror in _SHOWN_CHANGES:
        try:
            got = with_time_limit(lambda: fn(x.clone(), torch.tensor(u), PAD, mirror), 5)
            got = torch.as_tensor(got, dtype=torch.float32)
        except Exception:
            return ""
        if got.dim() != 3 or got.shape[0] != 1:
            return ""
        want = course_changed(x.clone(), torch.tensor(u), PAD, mirror)
        rows.append(got[0].numpy())
        other.append(None if _same_image(got, want) else want[0].numpy())
    words = {"en": ("as it is", "right"), "nl": ("zoals hij is", "goed")}[seen.lang]
    cells = [_cell(x[0].numpy(), [words[0]])]
    for ((a, b, c), mirror), got, want in zip(_SHOWN_CHANGES, rows, other):
        label = [f"u = ({a:.1f}, {b:.1f}, {c:.1f})"] + ([] if mirror else ["mirror False"])
        cells.append(_cell(got, label, None if want is None else (want, words[1])))
    title = {"en": "Your `changed` on a sneaker of the test images, with pad 2: mirror True, and in the last one mirror False",
             "nl": "Je `changed` op een sneaker uit de testafbeeldingen, met pad 2: mirror True, en in de laatste mirror False"}[seen.lang]
    body = '<div style="display:flex;flex-wrap:wrap;gap:10px 14px;align-items:flex-start">' + "".join(cells) + "</div>"
    return yours.figure(title, body, alt=title)


def _cell(image, label: list, right: tuple | None = None) -> str:
    """One image of 28 by 28 drawn at 84 pixels, with the lines of its label above it, and the right one under it if
    given."""
    parts = [f'<span style="{yours._MONO};font-size:12px;color:var(--ea-ink-2)">{html.escape(line)}</span>'
             for line in label] + [yours._png(np.asarray(image), 84) or ""]
    if right is not None:
        parts += [f'<span style="{yours._MONO};font-size:12px;color:var(--ea-ink-2)">{html.escape(right[1])}</span>',
                  yours._png(np.asarray(right[0]), 84) or ""]
    return '<div style="display:flex;flex-direction:column;gap:3px">' + "".join(parts) + "</div>"


EXERCISES = {
    "block": ClassBox(
        name="block",
        picture=netview.network("Block", lambda: torch.randn(2, 4, 8, 8), 4, 8, 2, scores=False),
        start=_B_START,
        solution=_B,
        scenarios=_B_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "block"),
        first=[_DOTS],
        mistakes=_B_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...`: in `__init__` the two convolutions with their batch normalizations, and the convolution of 1 by 1 with its batch normalization for the residual connection; in `forward`, `f(x)`, the residual connection and the ReLU of the sum.",
                      "nl": "Vervang de drie `...`: in `__init__` de twee convoluties met hun batch normalizations, en de convolutie van 1 bij 1 met zijn batch normalization voor de residual connection; in `forward` `f(x)`, de residual connection en de ReLU van de som."},
            "wrong": {"en": _C_EN + ". `f(x)` is `conv1`, `norm1`, a ReLU, `conv2` and `norm2`; the residual connection is `x`, or `skip_norm(skip(x))` when the block has a skip; the block gives `torch.relu` of their sum. The layers are made in the order `conv1`, `norm1`, `conv2`, `norm2`, `skip`, `skip_norm`, so that they get the same random weights as the course's.",
                      "nl": _C_NL + ". `f(x)` is `conv1`, `norm1`, een ReLU, `conv2` en `norm2`; de residual connection is `x`, of `skip_norm(skip(x))` als het block een skip heeft; het block geeft `torch.relu` van hun som. De layers worden gemaakt in de volgorde `conv1`, `norm1`, `conv2`, `norm2`, `skip`, `skip_norm`, zodat ze dezelfde willekeurige weights krijgen als bij de course."},
            "almost": {"en": "Right for the first block. But " + _C_EN[0].lower() + _C_EN[1:] + ". Does your block keep `x` itself where nothing changes, halve an odd size on both paths, get a skip only where the shape changes, let the gradient back through both paths, and does PyTorch see every layer?",
                       "nl": "Goed voor het eerste block. Maar " + _C_NL[0].lower() + _C_NL[1:] + ". Houdt je block `x` zelf waar niets verandert, halveert het een oneven maat op beide paden, krijgt het alleen een skip waar de shape verandert, laat het de gradient over beide paden terug, en ziet PyTorch elke layer?"},
            "correct": {"en": "Your `Block` is right: `f` with a batch normalization after each convolution, the residual connection with a convolution of 1 by 1 only where the shape changes, and the ReLU after the sum, as in ResNet. It is now in your project, as `{saved}`.",
                        "nl": "Je `Block` klopt: `f` met een batch normalization na elke convolutie, de residual connection met een convolutie van 1 bij 1 alleen waar de shape verandert, en de ReLU na de som, zoals in ResNet. Het staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `Block` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `Block` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "resnet": ClassBox(
        name="resnet",
        picture=netview.network("ResNet", lambda: _sneaker()[None], 1, 8, 2, 10),
        start=_N_START,
        solution=_N,
        scenarios=_N_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "deep_resnet"),
        first=[_DOTS],
        mistakes=_N_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: in `__init__` the loop that puts the blocks of the three stages in `layers`, and in `forward` the way through the network.",
                      "nl": "Vervang de twee `...`: in `__init__` de loop die de blocks van de drie stages in `layers` zet, en in `forward` de weg door het netwerk."},
            "wrong": {"en": _C_EN + ". Stage 1 has `blocks` times a `Block(width, width, 1)`; stage 2 and 3 start with a `Block` from the channels of the stage before, with stride 2, then `blocks - 1` times a `Block(c, c, 1)`, with `2 * width` and `4 * width` channels. The blocks are made in that order, so that they get the same random weights as the course's. In `forward`: the stem, `stem_norm`, a ReLU, every block, the mean of every map, and `self.out`.",
                      "nl": _C_NL + ". Stage 1 heeft `blocks` keer een `Block(width, width, 1)`; stage 2 en 3 beginnen met een `Block` van de channels van de stage ervoor, met stride 2, dan `blocks - 1` keer een `Block(c, c, 1)`, met `2 * width` en `4 * width` channels. De blocks worden in die volgorde gemaakt, zodat ze dezelfde willekeurige weights krijgen als bij de course. In `forward`: de stem, `stem_norm`, een ReLU, elk block, het gemiddelde van elke map, en `self.out`."},
            "almost": {"en": "Right for the first network. But " + _C_EN[0].lower() + _C_EN[1:] + ". Does every block stand in its place, do the last maps have 7 by 7 for an image of 28 by 28, does it work for color photos and another number of kinds, and does PyTorch see every block?",
                       "nl": "Goed voor het eerste netwerk. Maar " + _C_NL[0].lower() + _C_NL[1:] + ". Staat elk block op zijn plek, hebben de laatste maps 7 bij 7 voor een afbeelding van 28 bij 28, werkt het voor kleurenfoto's en een ander aantal soorten, en ziet PyTorch elk block?"},
            "correct": {"en": "Your `ResNet` is right: a stem, three stages of your blocks, each stage with half the height and width and twice the channels of the one before, global average pooling and a linear layer. With `ResNet(1, 16, 3, 10)` it has 20 layers with weights in a row, as ResNet-20 of the paper. It is now in your project, as `{saved}`.",
                        "nl": "Je `ResNet` klopt: een stem, drie stages van je blocks, elke stage met de helft van de hoogte en breedte en twee keer de channels van de vorige, global average pooling en een linear layer. Met `ResNet(1, 16, 3, 10)` heeft het 20 layers met weights achter elkaar, zoals ResNet-20 uit de paper. Het staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `ResNet` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `ResNet` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "changes": FunctionBox(
        name="changes",
        picture=changes_picture,
        function="changed",
        start=_X_START,
        solution=_X,
        visible=_X_VISIBLE,
        hidden=_X_HIDDEN,
        compare=_same_image,
        save_as=("cnn", "changes"),
        first=[_DOTS],
        mistakes=_X_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: the mirror image, the black image with the image in it, and the cut.",
                      "nl": "Vervang de `...` door je code: het spiegelbeeld, de zwarte afbeelding met de afbeelding erin, en de uitsnede."},
            "wrong": {"en": _F_EN + ". Mirror only when `mirror` is True and `u[0]` is below 0.5; the cut starts at row `int(u[1] * (2 * pad + 1))` and column `int(u[2] * (2 * pad + 1))` of the black image, and is as large as `x`.",
                      "nl": _F_NL + ". Spiegel alleen als `mirror` True is en `u[0]` onder 0,5; de uitsnede begint bij rij `int(u[1] * (2 * pad + 1))` en kolom `int(u[2] * (2 * pad + 1))` van de zwarte afbeelding, en is zo groot als `x`."},
            "almost": {"en": "Right for the image of the lesson. But " + _F_EN[0].lower() + _F_EN[1:] + ". Does your function work for any size and any number of channels, with mirror on and off, and with any border?",
                       "nl": "Goed voor de afbeelding uit de les. Maar " + _F_NL[0].lower() + _F_NL[1:] + ". Werkt je functie voor elke maat en elk aantal channels, met spiegelen aan en uit, en met elke rand?"},
            "correct": {"en": "Your `changed` is right: a mirror image when it may, and a random crop with a black border, for images of any size. It is now in your project, as `{saved}`.",
                        "nl": "Je `changed` klopt: een spiegelbeeld als dat mag, en een random crop met een zwarte rand, voor afbeeldingen van elke maat. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `changed` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `changed` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "sure-mistakes": FunctionBox(
        name="sure-mistakes",
        picture=yours.arrays(call=0),
        function="sure_mistakes",
        start=_M_START,
        solution=_M,
        visible=_M_VISIBLE,
        hidden=_M_HIDDEN,
        compare=_places,
        save_as=("cnn", "mistakes"),
        first=[_DOTS],
        mistakes=_M_MISTAKES,
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with your code: the probabilities, the guesses, the places of the wrong ones, sorted, and the first `n`.",
                      "nl": "Vervang de `...` door je code: de kansen, de gokken, de plekken van de foute, gesorteerd, en de eerste `n`."},
            "wrong": {"en": _F_EN + ". The guess of a row is the kind with the highest probability, after softmax over the row; keep the places where the guess is not the label, sort them from the highest probability of their guess down, and give back the first `n`.",
                      "nl": _F_NL + ". De gok van een rij is de soort met de hoogste kans, na softmax over de rij; houd de plekken waar de gok niet het label is, sorteer ze van de hoogste kans van hun gok naar beneden, en geef de eerste `n` terug."},
            "almost": {"en": "Right for the example of the lesson. But " + _F_EN[0].lower() + _F_EN[1:] + ". Does your function work when `n` is larger than the number of mistakes, or 0, and when there are no mistakes?",
                       "nl": "Goed voor het voorbeeld uit de les. Maar " + _F_NL[0].lower() + _F_NL[1:] + ". Werkt je functie als `n` groter is dan het aantal fouten, of 0, en als er geen fouten zijn?"},
            "correct": {"en": "Your `sure_mistakes` is right: the wrong guesses, from the one the network was most sure of down. It is now in your project, as `{saved}`.",
                        "nl": "Je `sure_mistakes` klopt: de foute gokken, vanaf die waarvan het netwerk het zekerst was. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `sure_mistakes` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `sure_mistakes` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("block", _B_START, "empty", None),
    ("block", _B, "correct", None),
    ("block", _b(forward="        f = self.conv1(x)\n        f = torch.relu(self.norm1(f))\n        f = self.norm2(self.conv2(f))\n"
                         "        s = x\n        if self.skip is not None:\n            s = self.skip_norm(self.skip(x))\n"
                         "        return torch.relu(s + f)\n"), "correct", None),
    ("block", _b(forward="        f = self.norm2(self.conv2(torch.relu(self.norm1(self.conv1(x)))))\n        return torch.relu(x + f)\n"), "error", "sizes"),
    ("block", _b(init=_B_INIT.replace("3, stride=stride, padding=1)", "3, stride=stride)")), "error", "sizes"),
    ("block", _b(skip=_B_SKIP.replace("1, stride=stride)", "1)")), "error", "sizes"),
    ("block", _b(init=_B_INIT.replace("nn.Conv2d(c_out, c_out, 3", "nn.Conv2d(c_in, c_out, 3")), "error", "channels"),
    ("block", _b(init=_B_INIT.replace("self.norm1 = nn.BatchNorm2d(c_out)", "self.norm1 = nn.BatchNorm2d(c_in)")), "error", "norm-size"),
    ("block", _b(forward="        f = self.norm2(self.conv2(torch.relu(self.norm1(self.conv1(x)))))\n"
                         "        if self.skip is not None:\n            x = self.skip_norm(self.skip(x))\n"
                         "        x += f\n        return torch.relu(x)\n"), "error", "in-place"),
    ("block", _B_HEAD + _B_SKIP_HEAD + _B_SKIP + _B_INIT + "\n    def forward(self, x):\n" + _B_FORWARD, "wrong", None),
    ("block", _B_START.replace("        ...\n        self.skip", _B_INIT + "        self.skip"), "wrong", "dots-left"),
] + [("block", code, "wrong", key) for key, code in _B_WRONG.items()] + [
    ("resnet", _N_START, "empty", None),
    ("resnet", _N, "correct", None),
    ("resnet", _n(stages="        for stage in range(3):\n            c = width * 2 ** stage\n"
                         "            if stage > 0:\n                layers.append(Block(c // 2, c, 2))\n"
                         "            for b in range(blocks - 1 if stage > 0 else blocks):\n                layers.append(Block(c, c, 1))\n"), "correct", None),
    ("resnet", _n(forward="        x = torch.relu(self.stem_norm(self.stem(x)))\n        for block in self.blocks:\n            x = block(x)\n"
                          "        x = torch.relu(x)\n        x = x.mean(dim=3).mean(dim=2)\n        return self.out(x)\n"), "correct", None),
    ("resnet", _n(forward="        x = torch.relu(self.stem_norm(self.stem(x)))\n        for block in self.blocks:\n            x = block(x)\n"
                          "        return self.out(x.reshape(len(x), -1))\n"), "error", "flatten"),
    ("resnet", _n(stages=_N_STAGES.replace("stage > 0 and b == 0", "b == 0 and stage > 1")), "error", "channels"),
    ("resnet", _n(stages=_N_STAGES.replace("if stage > 0 and b == 0", "if stage > 0 and b == blocks - 1")), "error", "channels"),
] + [("resnet", code, "wrong", key) for key, code in _N_WRONG.items()] + [
    ("changes", _X_START, "empty", None),
    ("changes", _X, "correct", None),
    ("changes", _x(pad="    c, h, w = x.shape\n    padded = torch.zeros(c, h + 2 * pad, w + 2 * pad)\n"
                   "    padded[:, pad:pad + h, pad:pad + w] = x\n", cut="    return padded[:, top:top + h, left:left + w].clone()\n"), "correct", None),
    ("changes", _x(mirror="    if mirror and u[0] < 0.5:\n        x = torch.flip(x, [2])\n"), "correct", None),
    ("changes", _x(pad="    c, h, w = x.shape\n    padded = torch.zeros(3, 40, 40)\n    padded[:, pad:pad + h, pad:pad + w] = x\n"), "wrong", "fixed-size"),
    ("changes", _x(pad="    c, h, w = x.shape\n    padded = torch.zeros(1, 32, 32)\n    padded[:, pad:pad + h, pad:pad + w] = x\n"), "error", "fixed-size"),
    ("changes", _x(pad="    c, h, w = x.shape\n    padded = torch.zeros(c, h + 2, w + 2)\n    padded[:, pad:pad + h, pad:pad + w] = x\n"), "almost", None),
    ("changes", _x(cut="    return padded[:, top:top + 28, left:left + 28]\n"), "wrong", None),
] + [("changes", code, "wrong", key) for key, code in _X_WRONG.items()] + [
    ("sure-mistakes", _M_START, "empty", None),
    ("sure-mistakes", _M, "correct", None),
    ("sure-mistakes", _m(pairs="    wrong = [i for i in range(len(labels)) if guess[i] != labels[i]]\n",
                         end="    return sorted(wrong, key=lambda i: -p[i].max().item())[:n]\n"), "correct", None),
    ("sure-mistakes", _m(probs="    p = torch.softmax(logits, dim=1)\n    guess = logits.argmax(dim=1)\n",
                         end="    return torch.tensor([pair[1] for pair in sorted(pairs, reverse=True)[:n]], dtype=torch.int64)\n"), "correct", None),
    ("sure-mistakes", _m(end="    return [float(pair[1]) for pair in sorted(pairs, reverse=True)[:n]]\n"), "wrong", None),
] + [("sure-mistakes", code, "wrong", key) for key, code in _M_WRONG.items()]


# --- the demo: the learner's ResNet on changed clothes ---------------------------------------------------------------

LIMIT = 1200                  # seconds the demo gives the training: the note says twenty minutes
SHIRT = one_each()[6]         # the first shirt of the test images: the demo shows the last block's maps on it


def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def _logits(model, X) -> torch.Tensor:
    """The logits of `model` for the images X, in eval mode, without a graph, 500 at a time."""
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + 500]) for i in range(0, len(X), 500)])


def best_epoch(val: list):
    """The first epoch with the lowest validation loss, rounded to 3 decimals as in REFERENCE."""
    return int(np.argmin(np.round(val, 3))) + 1 if len(val) else None


def _train(model, changed, Trainer, seed: int, mirror: bool, epochs: int, heard=None) -> list:
    """Train `model` as the demo does: on the first 5,000 training images, each changed anew by `changed` with PAD and
    `mirror` (none when changed is None), with the validation set as the Trainer's, after torch.manual_seed(seed + 1)."""
    X, y, X_val, y_val, _, _ = splits()
    mean, std = mean_std()
    torch.manual_seed(seed + 1)
    data = Changed(X, y, mean, std, changed, PAD, mirror) if changed is not None else TensorDataset((X - mean) / std, y)
    loader = DataLoader(data, batch_size=64, shuffle=True)
    val = DataLoader(TensorDataset((X_val - mean) / std, y_val), batch_size=500)
    trainer = Trainer(model, loader, val, epochs=epochs, lr=LR, warmup=len(loader), max_norm=1.0)
    if heard is not None:
        live.follow(trainer, heard, "project", every=EVERY)
    return trainer.fit() or []


def course_trainer():
    """The course's Trainer, from the solutions of nn-trainer. Its checks say what nn-trainer builds on when they load;
    what this lesson builds on is put back after it."""
    trainer = _lesson_checks("nn-trainer")._course_trainer()
    projects.builds_on(*BUILDS_ON)
    return trainer


def course_run(seed: int, changes: bool = True, mirror: bool = False, epochs: int = EPOCHS) -> dict:
    """One run of the demo with the course's code, as the runs of REFERENCE were made: the ResNet after
    torch.manual_seed(seed), trained on changed images (or as they are), and then measured once. Gives what it read
    right of the images it learned from, of the validation set and of the test images, the loss after every epoch,
    the logits of the test images and the model; REFERENCE's "sure", "why" and "confusion" come from the logits."""
    torch.manual_seed(seed)
    model = course_resnet(1, WIDTH, BLOCKS, 10)
    history = _train(model, course_changed if changes else None, course_trainer(), seed, mirror, epochs)
    X, y, X_val, y_val, X_test, y_test = splits()
    mean, std = mean_std()
    logits = _logits(model, (X_test - mean) / std)
    return {"seed": seed, "changes": changes, "mirror": mirror, "epochs": epochs,
            "fit": int((_logits(model, (X - mean) / std).argmax(1) == y).sum()),
            "val": int((_logits(model, (X_val - mean) / std).argmax(1) == y_val).sum()),
            "test": int((logits.argmax(1) == y_test).sum()),
            "val_loss": [float(h["val_loss"]) for h in history], "train_loss": [float(h["train_loss"]) for h in history],
            "logits": logits, "model": model}


def now_pictures(seen: dict, heads: dict, lang: str, uid: str) -> list:
    """What the network of the demo makes at one moment: its guess for the first test image of each kind, under
    heads["guesses"], and under them its last block on the shirt, under heads["layer"] with {name} the layer's name.
    `uid` keeps the masks of the layer apart from those of another picture on the page."""
    X_test = splits()[4]
    said = [KINDS[lang][g] for g in seen["guesses"]]
    right = [KINDS[lang][k] for k in seen["truth"]]
    alt = {"en": "The first test image of each kind, each with the kind the network guessed, and in brackets the right kind where it is wrong: ",
           "nl": "De eerste testafbeelding van elke soort, elk met de soort die het netwerk gokte, en tussen haakjes de goede soort waar het fout is: "}
    picture = yours.guesses([X_test[place][0].numpy() for place in one_each()], said, right,
                            alt[lang] + "; ".join(f"{r}: {g}" for r, g in zip(right, said)) + ".", title=heads["guesses"])
    return netview.moment(picture, seen.get("layer"), heads["layer"], lang, uid)


# The top of the curves pictures and of the live chart: above the loss of guessing, ln 10 ≈ 2.30. After the first
# epoch, the validation loss is above it (8.59 with the crop and seed 0, see EPOCH_1); the line comes in from the top.
TOP_LOSS = 2.6


def curves_picture(curves: dict, labels: dict, lang: str, at: int | None = None) -> str:
    """The loss after every epoch of the demo's run: on the changed training images solid and on the validation set
    dashed, as in torch-loss-curves, with a dot where the validation loss is lowest. With `at`, a line at that epoch,
    for rewinding."""
    from everything_ai import chart
    train, val = curves.get("train", []), curves.get("val", [])
    top = TOP_LOSS
    p = chart.Panel((0, max(len(val), len(train), EPOCHS) + 0.5), (0, top), title=labels["title"],
                    xlabel=labels["x"], ylabel=labels["y"])
    for ys, dashed in ((train, False), (val, True)):
        if ys:
            p.line(list(range(1, len(ys) + 1)), ys, "loss", 2.2, dashed=dashed)
    drawn = sorted([(ys, name) for ys, name in ((train, labels["train"]), (val, labels["val"])) if ys], key=lambda d: d[0][-1], reverse=True)
    for (ys, name), y in zip(drawn, chart.name_heights([min(ys[-1], top) for ys, _ in drawn], top)):
        p.text(len(ys) - 0.3 if len(ys) > 1 else 1, y, name, "loss", "end")   # clear of the line at the last epoch
    if val:
        best = best_epoch(val)
        p.points([best], [val[best - 1]], "loss", r=4)
    if at is not None:
        p.line([at, at], [0, top], "ink-2", 1.2)
    svg = chart.figure([p], alt=labels["alt"], width=276, height=200)   # 11 px text in a play box on a phone
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def epoch_frame(result: dict, e: int, labels: dict, heads: dict, lang: str) -> str:
    """The rewind at epoch e (from 1): the loss curves with a line at e, beside the network's guesses as they were
    after that epoch, and its last block on the shirt under them (decision 1.17)."""
    layers = result["layers"]
    seen = {"guesses": result["guesses"][e - 1], "truth": result["truth"], "layer": layers[e - 1] if e <= len(layers) else None}
    return live.rewound(curves_picture(result["curves"], labels, lang, at=e), now_pictures(seen, heads, lang, f"nv-project-epoch-{e}"))


def layer_frame(net, k, lang: str) -> str:
    """The slider over the layers of the learner's trained ResNet: the network, with layer k (from 1) chosen and its
    feature maps under it."""
    return netview.picture(net, lang, chosen=int(k) - 1, uid=f"nv-project-{int(k)}")


def network_picture(lang: str) -> str:
    """The course's ResNet-20 before it learned anything, after torch.manual_seed(0), in eval mode, on the sneaker of
    the test images, normalized as the demo does: a box for every layer with its shape and parameters."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = course_resnet(1, WIDTH, BLOCKS, 10).eval()
    mean, std = mean_std()
    return netview.picture(netview.trace(model, (_sneaker()[None] - mean) / std), lang, uid="nv-project-idea")


# What the course measured without batch normalization, kept as numbers: ResNet-20 made of the blocks of cnn-resnet,
# f(x) = conv2(relu(conv1(x))) and x + f(x), with a 1x1 convolution on the residual connection where the shape
# changes and a ReLU after the last block; trained as the demo, with the crop, with seeds 0 and 1 on two threads.
# "val": what it read right of the validation set after 20 epochs. "long": seed 0 trained for 40 epochs on one thread:
# the validation loss after every epoch, rounded to 3 decimals, and what it read right of the validation set after it.
NO_NORM = {"val": [822, 834],
           "long": {"epochs": 40, "right": 87,
                    "val_loss": [1.239, 1.271, 0.85, 0.713, 2.315, 2.303, 2.304, 2.305, 2.305, 2.306, 2.305, 2.305, 2.305,
                                 2.306, 2.306, 2.305, 2.305, 2.305, 2.305, 2.305, 2.305, 2.305, 2.305, 2.305, 2.305, 2.305,
                                 2.305, 2.305, 2.305, 2.305, 2.305, 2.305, 2.305, 2.305, 2.305, 2.305, 2.305, 2.305, 2.305,
                                 2.305]}}

# What the course measured with the three settings, five seeds each (0 to 4), trained as the demo for 20 epochs, each
# on one thread: what each read right of the 5,000 images it learned from, unchanged ("fit"), of the validation set
# ("val") and of the test images ("test"), the epoch with the lowest validation loss ("best"), the loss after every
# epoch with seed 0, rounded to 3 decimals, the seconds each run took ("secs", three at a time beside other work), the 8
# surest mistakes on the test images of each seed as (place, probability) ("sure"), the guess of each seed for test
# image 147 with its probability ("why"), and the confusion matrix of the test images of each seed. The settings were
# chosen on "val" alone; the demo is the run with the crop and seed 0, which tests/test_cv_project_resnet_grad_cam.py
# trains again.
REFERENCE = {
    "none": {"seeds": [0, 1, 2, 3, 4], "fit": [4999, 4993, 4998, 4997, 4995], "val": [868, 878, 873, 871, 864], "test": [897, 891, 895, 899, 894], "best": [8, 14, 11, 10, 11],
             "train_loss": [1.014, 0.649, 0.513, 0.447, 0.413, 0.374, 0.35, 0.301, 0.27, 0.239, 0.225, 0.174, 0.13, 0.092, 0.064, 0.045, 0.025, 0.017, 0.014, 0.013],
             "val_loss": [9.219, 0.898, 0.691, 0.579, 0.484, 0.551, 0.434, 0.42, 0.467, 0.525, 0.458, 0.543, 0.427, 0.486, 0.483, 0.476, 0.506, 0.522, 0.517, 0.517],
             "secs": [428.9, 432.9, 340.0, 335.2, 330.5],
             "sure": [[(787, 0.9999), (147, 0.9999), (396, 0.9999), (732, 0.9999), (49, 0.9998), (248, 0.9998), (711, 0.9989), (979, 0.9987)], [(147, 1.0), (23, 0.9999), (244, 0.9999), (68, 0.9998), (787, 0.9992), (49, 0.9958), (585, 0.9951), (732, 0.9937)], [(23, 1.0), (244, 0.9998), (701, 0.9998), (577, 0.9998), (787, 0.9998), (979, 0.9994), (313, 0.9991), (444, 0.999)], [(787, 1.0), (244, 0.9998), (23, 0.9995), (732, 0.9982), (674, 0.9965), (324, 0.9955), (722, 0.9953), (382, 0.9951)], [(979, 0.9999), (23, 0.9997), (147, 0.9995), (117, 0.9994), (722, 0.9994), (787, 0.9993), (732, 0.9989), (49, 0.9987)]],
             "why": [(3, 0.9999), (3, 1.0), (3, 0.9809), (3, 0.9939), (3, 0.9995)],
             "confusion": [[[96, 0, 2, 0, 0, 0, 8, 0, 1, 0], [0, 102, 1, 2, 0, 0, 0, 0, 0, 0], [3, 0, 96, 1, 5, 0, 6, 0, 0, 0], [4, 0, 0, 86, 0, 0, 3, 0, 0, 0], [0, 0, 8, 4, 94, 0, 9, 0, 0, 0], [0, 0, 0, 0, 0, 84, 0, 3, 0, 0], [10, 0, 5, 5, 9, 0, 67, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 90, 0, 4], [1, 1, 0, 0, 0, 0, 0, 0, 92, 1], [0, 0, 0, 0, 0, 1, 0, 4, 0, 90]], [[89, 0, 3, 2, 2, 0, 11, 0, 0, 0], [0, 103, 0, 2, 0, 0, 0, 0, 0, 0], [2, 0, 98, 0, 5, 0, 6, 0, 0, 0], [2, 0, 1, 84, 2, 0, 4, 0, 0, 0], [0, 1, 9, 3, 92, 0, 10, 0, 0, 0], [0, 0, 0, 0, 0, 85, 0, 1, 0, 1], [9, 0, 7, 4, 10, 0, 67, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 91, 0, 4], [1, 0, 1, 0, 0, 1, 0, 0, 92, 0], [0, 0, 0, 0, 0, 1, 0, 4, 0, 90]], [[90, 0, 2, 3, 0, 0, 12, 0, 0, 0], [0, 102, 1, 2, 0, 0, 0, 0, 0, 0], [4, 0, 100, 0, 5, 0, 2, 0, 0, 0], [3, 2, 1, 80, 1, 0, 6, 0, 0, 0], [0, 0, 6, 6, 94, 0, 9, 0, 0, 0], [0, 0, 0, 0, 0, 82, 0, 3, 0, 2], [8, 1, 5, 3, 8, 0, 72, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 94, 0, 1], [0, 0, 0, 0, 0, 1, 1, 0, 93, 0], [0, 0, 0, 1, 0, 1, 0, 5, 0, 88]], [[93, 0, 3, 2, 0, 0, 9, 0, 0, 0], [0, 102, 1, 2, 0, 0, 0, 0, 0, 0], [2, 0, 100, 0, 3, 0, 6, 0, 0, 0], [3, 0, 0, 83, 1, 0, 6, 0, 0, 0], [0, 0, 7, 4, 92, 0, 11, 0, 1, 0], [0, 0, 0, 0, 0, 82, 0, 4, 0, 1], [6, 0, 6, 4, 7, 0, 74, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 92, 0, 3], [1, 0, 1, 0, 0, 0, 2, 0, 91, 0], [0, 0, 0, 0, 0, 1, 0, 4, 0, 90]], [[93, 0, 2, 0, 0, 0, 11, 0, 1, 0], [0, 103, 1, 1, 0, 0, 0, 0, 0, 0], [2, 1, 100, 0, 5, 0, 3, 0, 0, 0], [1, 0, 1, 83, 2, 0, 5, 0, 1, 0], [0, 0, 13, 3, 91, 0, 8, 0, 0, 0], [0, 0, 0, 0, 0, 83, 0, 4, 0, 0], [7, 0, 7, 7, 9, 0, 67, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 91, 0, 4], [1, 0, 0, 0, 0, 0, 1, 0, 93, 0], [0, 0, 0, 0, 0, 1, 0, 4, 0, 90]]]},
    "crop": {"seeds": [0, 1, 2, 3, 4], "fit": [4708, 4708, 4689, 4711, 4696], "val": [892, 891, 884, 894, 894], "test": [909, 914, 910, 903, 905], "best": [16, 18, 18, 18, 20],
             "train_loss": [1.123, 0.706, 0.586, 0.516, 0.499, 0.456, 0.432, 0.391, 0.346, 0.326, 0.32, 0.292, 0.274, 0.247, 0.237, 0.221, 0.206, 0.19, 0.188, 0.183],
             "val_loss": [8.588, 1.206, 1.382, 1.689, 0.563, 0.619, 0.513, 0.739, 0.411, 0.4, 0.387, 0.409, 0.404, 0.348, 0.349, 0.324, 0.338, 0.327, 0.331, 0.328],
             "secs": [447.0, 449.9, 435.1, 355.3, 355.2],
             "sure": [[(147, 0.9995), (787, 0.9987), (23, 0.998), (938, 0.9958), (244, 0.9852), (441, 0.9777), (44, 0.9774), (444, 0.9712)], [(23, 0.9994), (147, 0.9972), (787, 0.9969), (441, 0.9897), (68, 0.988), (938, 0.9858), (444, 0.9706), (979, 0.9601)], [(787, 0.9996), (23, 0.9995), (147, 0.9991), (938, 0.9987), (68, 0.9958), (722, 0.9911), (444, 0.991), (893, 0.9889)], [(68, 0.9993), (147, 0.9988), (938, 0.9974), (787, 0.9962), (23, 0.9944), (722, 0.9815), (244, 0.9783), (444, 0.9745)], [(147, 0.9997), (23, 0.9995), (787, 0.9995), (938, 0.9954), (244, 0.9894), (722, 0.9874), (29, 0.979), (747, 0.9776)]],
             "why": [(3, 0.9995), (3, 0.9972), (3, 0.9991), (3, 0.9988), (3, 0.9997)],
             "confusion": [[[90, 0, 1, 0, 0, 0, 16, 0, 0, 0], [0, 104, 0, 1, 0, 0, 0, 0, 0, 0], [2, 0, 99, 0, 3, 0, 7, 0, 0, 0], [1, 0, 1, 82, 2, 0, 7, 0, 0, 0], [0, 0, 4, 2, 98, 0, 11, 0, 0, 0], [0, 0, 0, 0, 0, 86, 0, 1, 0, 0], [4, 0, 4, 3, 10, 0, 76, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 89, 0, 5], [0, 1, 0, 0, 0, 0, 0, 0, 94, 0], [0, 0, 0, 0, 0, 1, 0, 3, 0, 91]], [[92, 0, 0, 0, 0, 0, 15, 0, 0, 0], [0, 102, 1, 2, 0, 0, 0, 0, 0, 0], [2, 0, 104, 1, 3, 0, 1, 0, 0, 0], [3, 0, 1, 81, 2, 0, 5, 0, 1, 0], [0, 0, 2, 2, 104, 0, 7, 0, 0, 0], [0, 0, 0, 0, 0, 85, 0, 1, 0, 1], [7, 0, 5, 4, 13, 0, 68, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 94, 0, 1], [0, 0, 2, 0, 0, 0, 0, 0, 92, 1], [0, 0, 0, 0, 0, 1, 0, 2, 0, 92]], [[90, 0, 0, 0, 0, 0, 17, 0, 0, 0], [0, 103, 0, 2, 0, 0, 0, 0, 0, 0], [3, 0, 99, 0, 6, 0, 3, 0, 0, 0], [1, 0, 1, 82, 2, 0, 6, 0, 1, 0], [0, 0, 1, 3, 103, 0, 8, 0, 0, 0], [0, 0, 0, 0, 0, 82, 0, 4, 0, 1], [2, 0, 3, 3, 14, 0, 75, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 92, 0, 2], [1, 0, 0, 0, 0, 0, 0, 0, 93, 1], [0, 0, 0, 0, 0, 1, 0, 3, 0, 91]], [[96, 0, 1, 2, 0, 0, 8, 0, 0, 0], [0, 104, 0, 1, 0, 0, 0, 0, 0, 0], [2, 0, 96, 1, 6, 0, 6, 0, 0, 0], [1, 0, 2, 82, 3, 0, 4, 0, 1, 0], [0, 0, 7, 6, 94, 0, 8, 0, 0, 0], [0, 0, 0, 0, 0, 84, 0, 2, 0, 1], [6, 0, 5, 4, 11, 0, 71, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 92, 0, 2], [1, 0, 0, 0, 0, 1, 0, 0, 93, 0], [0, 0, 0, 0, 0, 1, 0, 3, 0, 91]], [[93, 0, 1, 0, 0, 0, 13, 0, 0, 0], [0, 103, 0, 2, 0, 0, 0, 0, 0, 0], [3, 0, 100, 0, 5, 0, 3, 0, 0, 0], [3, 0, 1, 78, 2, 0, 7, 0, 2, 0], [0, 0, 6, 4, 96, 0, 9, 0, 0, 0], [0, 0, 0, 0, 0, 85, 0, 2, 0, 0], [7, 0, 4, 2, 9, 0, 75, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 91, 0, 3], [0, 0, 0, 0, 0, 0, 1, 0, 94, 0], [0, 0, 0, 0, 0, 1, 0, 4, 0, 90]]]},
    "mirror": {"seeds": [0, 1, 2, 3, 4], "fit": [4648, 4629, 4626, 4627, 4616], "val": [900, 890, 884, 886, 889], "test": [907, 912, 913, 898, 895], "best": [17, 17, 20, 18, 19],
             "train_loss": [1.171, 0.757, 0.613, 0.561, 0.531, 0.49, 0.461, 0.415, 0.371, 0.358, 0.335, 0.316, 0.303, 0.278, 0.263, 0.251, 0.234, 0.227, 0.217, 0.214],
             "val_loss": [3.689, 0.925, 1.399, 1.888, 0.627, 0.821, 0.628, 1.013, 0.466, 0.483, 0.393, 0.4, 0.416, 0.345, 0.378, 0.338, 0.327, 0.336, 0.331, 0.33],
             "secs": [368.2, 333.2, 351.4, 342.9, 449.8],
             "sure": [[(787, 0.9989), (23, 0.9983), (147, 0.9981), (938, 0.9877), (29, 0.9758), (444, 0.9735), (441, 0.9627), (722, 0.9599)], [(147, 0.9985), (787, 0.9969), (23, 0.9963), (444, 0.9933), (722, 0.9919), (938, 0.9898), (244, 0.9895), (29, 0.9837)], [(23, 0.9992), (787, 0.9988), (147, 0.995), (68, 0.9946), (979, 0.9932), (244, 0.9824), (938, 0.9814), (722, 0.9711)], [(147, 0.9982), (23, 0.9981), (787, 0.996), (938, 0.9925), (628, 0.9914), (244, 0.9863), (68, 0.9758), (722, 0.9721)], [(787, 0.9996), (147, 0.9992), (23, 0.9989), (938, 0.9941), (441, 0.9847), (713, 0.9798), (722, 0.9669), (510, 0.9611)]],
             "why": [(3, 0.9981), (3, 0.9985), (3, 0.995), (3, 0.9982), (3, 0.9992)],
             "confusion": [[[96, 0, 1, 0, 0, 0, 10, 0, 0, 0], [0, 104, 0, 1, 0, 0, 0, 0, 0, 0], [2, 0, 95, 1, 7, 0, 6, 0, 0, 0], [1, 1, 1, 78, 2, 1, 9, 0, 0, 0], [0, 0, 4, 2, 102, 0, 7, 0, 0, 0], [0, 0, 0, 0, 0, 82, 0, 3, 0, 2], [6, 0, 6, 3, 9, 0, 73, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 91, 0, 3], [0, 0, 0, 0, 0, 1, 0, 0, 94, 0], [0, 0, 0, 0, 0, 1, 0, 2, 0, 92]], [[93, 0, 0, 2, 0, 0, 12, 0, 0, 0], [0, 103, 1, 1, 0, 0, 0, 0, 0, 0], [3, 0, 100, 1, 4, 0, 3, 0, 0, 0], [3, 0, 1, 83, 1, 0, 5, 0, 0, 0], [0, 0, 1, 5, 101, 0, 8, 0, 0, 0], [0, 0, 0, 0, 0, 82, 0, 3, 0, 2], [4, 0, 5, 3, 11, 0, 74, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 93, 0, 1], [0, 0, 0, 0, 0, 2, 1, 0, 92, 0], [0, 0, 0, 0, 0, 1, 0, 3, 0, 91]], [[94, 0, 2, 1, 0, 0, 10, 0, 0, 0], [0, 104, 0, 1, 0, 0, 0, 0, 0, 0], [3, 0, 98, 0, 8, 0, 2, 0, 0, 0], [1, 0, 1, 82, 2, 0, 7, 0, 0, 0], [0, 0, 1, 3, 103, 0, 8, 0, 0, 0], [0, 0, 0, 0, 0, 81, 0, 5, 1, 0], [6, 0, 3, 4, 9, 0, 75, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 92, 0, 2], [0, 0, 1, 0, 0, 0, 1, 0, 93, 0], [0, 0, 0, 0, 0, 1, 0, 3, 0, 91]], [[94, 0, 1, 1, 0, 0, 11, 0, 0, 0], [0, 104, 0, 1, 0, 0, 0, 0, 0, 0], [4, 0, 97, 1, 5, 0, 4, 0, 0, 0], [1, 2, 1, 82, 1, 0, 6, 0, 0, 0], [0, 0, 4, 4, 97, 0, 10, 0, 0, 0], [0, 0, 0, 0, 0, 80, 0, 4, 1, 2], [7, 0, 5, 4, 11, 0, 70, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 91, 0, 3], [0, 0, 0, 0, 0, 1, 1, 0, 92, 1], [0, 0, 0, 0, 0, 1, 0, 3, 0, 91]], [[89, 0, 1, 0, 0, 0, 17, 0, 0, 0], [0, 104, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 101, 0, 2, 0, 6, 0, 0, 0], [4, 0, 1, 79, 1, 0, 6, 0, 2, 0], [0, 0, 4, 4, 96, 0, 11, 0, 0, 0], [0, 0, 0, 0, 0, 82, 0, 5, 0, 0], [9, 0, 3, 5, 9, 0, 71, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 90, 0, 4], [0, 0, 0, 0, 0, 1, 1, 0, 93, 0], [0, 0, 0, 0, 0, 1, 0, 4, 0, 90]]]},
}

# The validation loss after the first epoch with the crop and seed 0, in eval mode, with the running averages of batch
# normalization, and with the validation images' own mean and variance (train mode, 500 at a time), with what the
# network read right of the 1,000: the running averages still lag behind the weights.
EPOCH_1 = {"eval": {"loss": 8.588, "right": 357}, "batch": {"loss": 0.882, "right": 643}}

SETTINGS = ("none", "crop", "mirror")   # the settings the course chose between: no changes, the crop, crop and mirror


def settings_frame(k: int, labels: dict, lang: str) -> tuple:
    """The slider over the settings: the loss curves of the course's run with seed 0 for SETTINGS[k], and what the
    five seeds read right of the validation set and of the images they learned from, from REFERENCE."""
    ref = REFERENCE[SETTINGS[k]]
    picture = curves_picture({"train": ref["train_loss"], "val": ref["val_loss"]}, labels, lang)
    val, fit = sorted(ref["val"]), sorted(ref["fit"])
    return picture, {"low": val[0], "high": val[-1], "middle": val[len(val) // 2],
                     "fit_low": grouped(fit[0], lang), "fit_high": grouped(fit[-1], lang)}


def moments(n: int) -> list:
    """The epochs to rewind to in a run of n epochs: 1, 2, 5, then every fifth, and the last."""
    return sorted({m for m in (1, 2, 5, 10, 15, 20, 25, 30, 35, 40) if m < n} | ({n} if n else set()))


def _demo(show=None):
    """For the demo: None when deep_resnet.py or changes.py is not in the project yet; "resnet" when the learner's
    ResNet stops with an error or gives no 10 logits per image; "changes" when their `changed` stops with an error or
    gives no image of shape (1, 28, 28); "trainer" when the Trainer or count_params does; "slow" when the training
    takes longer than LIMIT; else a dict with the parameters, how many of the images it learned from, of the
    validation set and of the test images it reads right, the loss after every epoch ("curves"), the epoch with the
    lowest validation loss ("best"), the network's guess for the first test image of each kind after every epoch
    ("guesses", with their right kinds in "truth") and its last block on the shirt ("layers"), the logits of the test
    images and the trained model, for the error analysis, and the modules the course's versions stood in for. `show`
    hears the run while it trains (live.follow, every 10 batches), with what the network makes then. It trains with
    PAD and MIRROR, the changes the course chose on the validation set, for EPOCHS epochs with LR."""
    if not (_saved("deep_resnet") and _saved("changes")):
        return None
    run = run_code("from cnn.deep_resnet import ResNet")
    if run.error is not None:
        return "resnet"
    ResNet = run.names["ResNet"]
    run = run_code("from cnn.changes import changed")
    if run.error is not None:
        return "changes"
    changed = run.names["changed"]
    run = run_code("from nn.trainer import Trainer\nfrom cnn.count import count_params")
    if run.error is not None:
        return "trainer"
    Trainer, count_params = run.names["Trainer"], run.names["count_params"]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    X, y, X_val, y_val, X_test, y_test = splits()
    try:
        out = with_time_limit(lambda: torch.as_tensor(changed(X[0].clone(), torch.tensor([0.2, 0.0, 1.0 - 1e-6]), PAD, MIRROR)), 10)
        if tuple(out.shape) != (1, 28, 28) or not torch.isfinite(out).all():
            return "changes"
    except Exception:
        return "changes"
    try:
        torch.manual_seed(0)
        model = ResNet(1, WIDTH, BLOCKS, 10)
        with torch.no_grad():   # on a copy: in train mode, batch normalization would change the running averages
            if tuple(copy.deepcopy(model)(X[:2]).shape) != (2, 10):
                return "resnet"
    except Exception:
        return "resnet"
    try:
        params = int(count_params(model))
    except Exception:
        return "trainer"
    mean, std = mean_std()
    shown = one_each()
    truth, guesses, layers = [int(y_test[place]) for place in shown], [], []
    test_n = (X_test - mean) / std
    heard = netview.hearing(model, test_n[shown], truth, test_n[SHIRT:SHIRT + 1], show, guesses, layers, WATCHED)
    try:
        history = with_time_limit(lambda: _train(model, changed, Trainer, 0, MIRROR, EPOCHS, heard), LIMIT)
    except Timeout:
        return "slow"
    except Exception:
        return "trainer"
    curves = {"train": [float(h["train_loss"]) for h in history if isinstance(h, dict) and "train_loss" in h],
              "val": [float(h["val_loss"]) for h in history if isinstance(h, dict) and "val_loss" in h]}
    logits = _logits(model, test_n)   # in eval mode from here on
    net = netview.trace(model, test_n[SHIRT:SHIRT + 1])   # the trained network layer by layer, on the shirt
    return {"params": params, "fit": int((_logits(model, (X - mean) / std).argmax(1) == y).sum()),
            "val": int((_logits(model, (X_val - mean) / std).argmax(1) == y_val).sum()),
            "test": int((logits.argmax(1) == y_test).sum()), "curves": curves, "best": best_epoch(curves["val"]),
            "guesses": guesses[:len(curves["val"])], "layers": layers[:len(curves["val"])], "truth": truth,
            "logits": logits, "model": model, "stand_ins": stand_ins, "net": net}


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file)."""
    texts = []
    for package, module in (("cnn", "deep_resnet"), ("cnn", "block"), ("cnn", "changes"),
                            ("cnn", "count"), ("nn", "train_epoch"), ("nn", "trainer")):
        path = projects.folder() / package / f"{module}.py"
        texts.append(path.read_text(encoding="utf-8") if path.exists() else None)
    return tuple(texts)


def demo(show=None):
    """The demo's result, worked out once for the project files as they are; `show` hears the run when it trains."""
    key = _project_texts()
    if key not in _DEMO_RUNS:
        _DEMO_RUNS.clear()
        _DEMO_RUNS[key] = _demo(show)
    return _DEMO_RUNS[key]


def last_run():
    """The result of the last run of the demo, when it trained a network; else None."""
    for result in _DEMO_RUNS.values():
        if isinstance(result, dict):
            return result
    return None


# --- the error analysis: the confusion matrix, and the learner's Grad-CAM over the surest mistakes --------------------

def confusion(logits, labels) -> np.ndarray:
    """The confusion matrix of the test images: rows the right kind, columns the guess."""
    cm = np.zeros((10, 10), dtype=int)
    for k, g in zip(labels.tolist(), logits.argmax(dim=1).tolist()):
        cm[k, g] += 1
    return cm


def _as_map(m) -> np.ndarray | None:
    """What the learner's grad_cam gave back as a NumPy array of 7 by 7, or None when it is not a finite map of that
    shape without negative numbers."""
    if isinstance(m, torch.Tensor):
        m = m.detach().numpy()
    try:
        m = np.asarray(m, dtype=float)
    except (TypeError, ValueError):
        return None
    if m.shape != (7, 7) or not np.isfinite(m).all() or m.min() < 0:
        return None
    return m


def analysis(result: dict):
    """The error analysis of the demo's network: None when mistakes.py is not in the project yet; "mistakes" when the
    learner's sure_mistakes stops with an error or gives no list of places of wrong guesses; "grad_cam" when their
    grad_cam stops with an error or gives no map of 7 by 7 without negative numbers; else a dict with the confusion
    matrix ("confusion"), and for each of the SURE mistakes their network was most sure of: its place, the right kind,
    the guess and its probability, and the learner's Grad-CAM of the guess and of the right kind from the last block
    ("mistakes"), and the modules the course's versions stood in for."""
    if not _saved("mistakes"):
        return None
    run = run_code("from cnn.mistakes import sure_mistakes")
    if run.error is not None:
        return "mistakes"
    sure_mistakes = run.names["sure_mistakes"]
    run = run_code("from cnn.gradcam import grad_cam")
    if run.error is not None:
        return "grad_cam"
    grad_cam = run.names["grad_cam"]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    logits, labels = result["logits"], splits()[5]
    guess = logits.argmax(dim=1)
    p = torch.softmax(logits, dim=1)
    try:
        places = with_time_limit(lambda: sure_mistakes(logits.clone(), labels.clone(), SURE), 10)
        if isinstance(places, torch.Tensor):
            places = places.tolist()
        places = [int(i) for i in places]
    except Exception:
        return "mistakes"
    if len(places) > SURE or len(set(places)) != len(places) or any(not 0 <= i < len(labels) or guess[i] == labels[i] for i in places):
        return "mistakes"
    model = copy.deepcopy(result["model"]).eval()   # their grad_cam gets a copy: what it does to it stays there
    mean, std = mean_std()
    X_test = splits()[4]
    shown = []
    try:
        for i in places:
            x = ((X_test[i:i + 1] - mean) / std).clone()
            maps = []
            for label in (int(guess[i]), int(labels[i])):
                m = with_time_limit(lambda: _as_map(grad_cam(model, model.blocks[-1], x.clone(), label)), 20)
                if m is None:
                    return "grad_cam"
                maps.append(m)
            shown.append({"place": i, "right": int(labels[i]), "guess": int(guess[i]), "p": float(p[i, guess[i]]),
                          "maps": maps})
    except Exception:
        return "grad_cam"
    return {"confusion": confusion(logits, labels), "mistakes": shown, "stand_ins": stand_ins}


def confusion_picture(cm, labels: dict, lang: str) -> str:
    """The confusion matrix as a grid of counts: a row for each right kind, named on the left, a column for each guess,
    numbered at the top. The right guesses on the diagonal in ink; a cell with mistakes tinted in the color of
    mistakes, the more the darker."""
    cm = np.asarray(cm)
    names = KINDS[lang]
    cell, left, top = 34, 112, 58
    off = cm.copy()
    np.fill_diagonal(off, 0)
    most = max(int(off.max()), 1)
    width, height = left + 10 * cell + 6, top + 10 * cell + 6
    parts = [f'<text x="{left + 5 * cell}" y="16" style="font-size:13px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(labels["guess"])}</text>',
             f'<text x="{left - 8}" y="{top - 10}" style="font-size:13px;fill:var(--ea-ink-2);text-anchor:end">{html.escape(labels["right"])}</text>']
    for j in range(10):
        parts.append(f'<text x="{left + j * cell + cell / 2}" y="{top - 10}" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:middle">{j}</text>')
    for i in range(10):
        y0 = top + i * cell
        parts.append(f'<text x="{left - 8}" y="{y0 + cell / 2 + 4}" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:end">{i} {html.escape(names[i])}</text>')
        for j in range(10):
            x0 = left + j * cell
            v = int(cm[i, j])
            if i != j and v:
                parts.append(f'<rect x="{x0 + 1}" y="{y0 + 1}" width="{cell - 2}" height="{cell - 2}" rx="3" '
                             f'style="fill:var(--ea-loss);fill-opacity:{0.08 + 0.42 * v / most:.2f}"/>')
            elif i == j:
                parts.append(f'<rect x="{x0 + 1}" y="{y0 + 1}" width="{cell - 2}" height="{cell - 2}" rx="3" style="fill:var(--ea-inset)"/>')
            if v or i == j:
                weight = "font-weight:600;" if i == j else ""
                parts.append(f'<text x="{x0 + cell / 2}" y="{y0 + cell / 2 + 4}" style="font-size:12px;{weight}fill:var(--ea-ink);text-anchor:middle">{v}</text>')
    alt = html.escape(labels["alt"], quote=True)
    # drawn at its own size, so that its text stays 12 px: on a narrow screen, the grid scrolls sideways
    return (f'<div style="overflow-x:auto;max-width:100%"><svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'role="img" aria-label="{alt}" style="display:block;width:{width}px;height:{height}px;max-width:none;'
            f'font-family:var(--ea-font-ui)"><title>{alt}</title>' + "".join(parts) + "</svg></div>")


VEIL = 0.85   # how much of the page a cell's veil shows where a map is 0: none where the map is largest


def _png64(image) -> str:
    """A gray image of 0 to 1 as a PNG in base64."""
    import base64
    import io
    from PIL import Image
    a = np.clip(np.asarray(image, dtype=float) * 255, 0, 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _veiled(image, m, top: float, x0: float, y0: float, side: int, title: str, role: str) -> list:
    """An image with a map over it, in SVG: over every cell of the map a veil in the page's color, as opaque as VEIL
    where the map is 0 and clear where it reaches `top`, as in cv-grad-cam; its title above it, in the color of `role`."""
    m = np.asarray(m, dtype=float)
    h, w = m.shape
    color = {"loss": "var(--ea-loss-text)", "ink": "var(--ea-ink)"}[role]
    parts = [f'<text x="{x0}" y="{y0 - 6}" style="font-size:12px;fill:{color}">{html.escape(title)}</text>',
             f'<image href="data:image/png;base64,{_png64(image)}" x="{x0}" y="{y0}" width="{side}" height="{side}" '
             f'preserveAspectRatio="none" style="image-rendering:pixelated"/>', '<g shape-rendering="crispEdges">']
    for i in range(h):
        for j in range(w):
            a = VEIL * (1 - m[i, j] / top) if top > 0 else VEIL
            if a >= 0.005:
                xa, xb = x0 + round(j * side / w), x0 + round((j + 1) * side / w)
                ya, yb = y0 + round(i * side / h), y0 + round((i + 1) * side / h)
                parts.append(f'<rect x="{xa}" y="{ya}" width="{xb - xa}" height="{yb - ya}" style="fill:var(--ea-page);opacity:{a:.2f}"/>')
    parts.append("</g>")
    parts.append(f'<rect x="{x0}" y="{y0}" width="{side}" height="{side}" style="fill:none;stroke:var(--ea-line)"/>')
    return parts


def mistakes_picture(found: dict, labels: dict, lang: str) -> str:
    """The mistakes of the error analysis, one card each: what the image is and what the network read, how sure it
    was, and the image twice, veiled where the Grad-CAM of the guess is low and where that of the right kind is low,
    both on the scale of the larger of the two maps, so that a weak map stays veiled."""
    X_test = splits()[4]
    names = KINDS[lang]
    side, gap = 98, 14
    cards = []
    for k in found["mistakes"]:
        image = X_test[k["place"]][0].numpy()
        width, height = 2 * side + gap, side + 64
        values = {"guess": names[k["guess"]], "right": names[k["right"]], "p": number(k["p"], 2, lang), "place": k["place"]}
        parts = [f'<text x="0" y="14" style="font-size:13px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["read"].format(**values))}</text>',
                 f'<text x="0" y="31" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(labels["sure"].format(**values))}</text>']
        top = max(float(np.max(k["maps"][0])), float(np.max(k["maps"][1])))   # both maps of a card on one scale
        parts += _veiled(image, k["maps"][0], top, 0, 58, side, values["guess"], "loss")
        parts += _veiled(image, k["maps"][1], top, side + gap, 58, side, values["right"], "ink")
        alt = labels["card"].format(**values)
        cards.append(f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(alt, quote=True)}" '
                     f'style="width:{width}px;max-width:100%;height:auto;font-family:var(--ea-font-ui)"><title>{html.escape(alt)}</title>'
                     + "".join(parts) + "</svg>")
    body = '<div style="display:flex;flex-wrap:wrap;gap:14px 26px">' + "".join(cards) + "</div>"
    return yours.figure(labels["title"], body, alt=labels["title"])
