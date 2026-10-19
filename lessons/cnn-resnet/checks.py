# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson cnn-resnet: the residual block of He and others (2015) for a CNN, two convolutions of 3
by 3 with the block's input added to their output, saved in the learner's project (work/projects/cnn/resnet.py) when
right. The picture shows how much the logits of a fresh network still differ between images, plain and residual, as it
gets deeper; the demo trains a plain network and one of the learner's residual blocks, of the same depth, with their
Trainer."""
import functools
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import paths, projects
from everything_ai.check import ClassBox, Mistake, Timeout, run_code, with_time_limit

# The demo trains with the learner's Trainer of nn-trainer and counts with their count_params of cnn-pytorch; where one
# is missing or broken, the course's version stands in (template 1.3).
projects.builds_on(("nn", "train_epoch", "nn-trainer", "train-epoch"), ("nn", "trainer", "nn-trainer", "trainer"),
                   ("cnn", "count", "cnn-pytorch", "count-params"))

DATA = "data/fashion-mnist-small.npz"   # 6,000 training and 1,000 test images of Fashion-MNIST: see data/README.md
EPOCHS, LR = 5, 0.01
BLOCKS = 4   # residual blocks per stage in the demo: 2 + 4 × 4 convolutions and the output head, 19 layers with weights

# How much the logits of a fresh network differ between images: the standard deviation over the first 256 test images,
# averaged over the 10 logits, after torch.manual_seed(0), for 0 to 8 blocks per stage (_network below), on one thread.
# Plain blocks wash the differences out; below about 10⁻⁸ only float32 rounding is left.
SPREAD_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]
SPREAD_PLAIN = [0.0184, 0.000137, 1.43e-06, 1.78e-08, 2.93e-09, 2.58e-09, 1.96e-09, 2.74e-09, 1.91e-09]
SPREAD_RESIDUAL = [0.0184, 0.0174, 0.0218, 0.0183, 0.0224, 0.0244, 0.0213, 0.0243, 0.0240]


@functools.lru_cache(maxsize=1)
def fashion():
    """The training and test images as tensors of 0 to 1, each image as one channel of 28 by 28, and their labels."""
    d = np.load(paths.ROOT / DATA)
    X = torch.tensor(d["train_images"], dtype=torch.float32).reshape(-1, 1, 28, 28) / 255
    y = torch.tensor(d["train_labels"], dtype=torch.int64)
    X_test = torch.tensor(d["test_images"], dtype=torch.float32).reshape(-1, 1, 28, 28) / 255
    y_test = torch.tensor(d["test_labels"], dtype=torch.int64)
    return X, y, X_test, y_test


def spread_picture(labels: dict) -> str:
    """The spread of the logits over images at the start, on a log scale, against the layers with weights: plain
    (grey, dashed) falls to rounding noise, residual (ink) stays."""
    from everything_ai import chart
    p = chart.Panel((0, 37), (1e-9, 0.3), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=[3, 11, 19, 27, 35], ylog=True)
    p.line(SPREAD_LAYERS, SPREAD_PLAIN, "ink-3", 2.2, dashed=True)
    p.points(SPREAD_LAYERS, SPREAD_PLAIN, "ink-3")
    p.line(SPREAD_LAYERS, SPREAD_RESIDUAL, "ink", 2.2)
    p.points(SPREAD_LAYERS, SPREAD_RESIDUAL, "ink")
    p.text(36, 0.003, labels["residual"], "ink", "end")
    p.text(36, 3e-8, labels["plain"], "ink-2", "end")
    svg = chart.figure([p], alt=labels["alt"], width=330, height=260)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


def grouped(n: int, lang: str) -> str:
    """A whole number with its thousands marked, as the language writes it: 32,330 or 32.330."""
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


# The lesson is done when these are correct. "res-shape" and "res-projection" are the choice questions in the lesson.
REQUIRED = ["res-shape", "res-block", "res-projection"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
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


def _close(got, want) -> bool:
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    if isinstance(want, float):
        return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - want) <= 1e-4 * max(1.0, abs(want))
    return got == want


# --- exercise 2: the residual block -----------------------------------------------------------------------------------------

_R_HEAD = "import torch\nfrom torch import nn\n\n\nclass ResBlock(nn.Module):\n    def __init__(self, c):\n        super().__init__()\n"
_R_INIT = "        self.conv1 = nn.Conv2d(c, c, 3, padding=1)\n        self.conv2 = nn.Conv2d(c, c, 3, padding=1)\n"
_R_FORWARD = "        return x + self.conv2(torch.relu(self.conv1(x)))\n"
_R_START = (_R_HEAD + "        # two convolutions of 3 by 3 from c to c channels, with padding 1, so that f(x) has the shape of x\n"
            "        ...\n\n    def forward(self, x):\n        # f(x): conv1, ReLU, conv2; the block gives x + f(x)\n        ...\n")


def _r(init: str = _R_INIT, forward: str = _R_FORWARD) -> str:
    return _R_HEAD + init + "\n    def forward(self, x):\n" + forward


_R = _r()
_R_SCENARIOS = [
    # a block of 4 channels on 2 feature maps of 6 by 6, with numbers below 0 as after an earlier block: the shape, and
    # 4 numbers of the first
    "import torch\n\ntorch.manual_seed(0)\nblock = ResBlock(4)\ny = block(torch.randn(2, 4, 6, 6))\n"
    "[list(y.shape), y[0, :, 0, 0].tolist()]",
    # with f(x) made 0, the block passes x on unchanged
    "import torch\n\ntorch.manual_seed(0)\nblock = ResBlock(3)\nwith torch.no_grad():\n    block.conv2.weight.zero_()\n"
    "    block.conv2.bias.zero_()\nx = torch.randn(1, 3, 5, 5)\n(block(x) - x).abs().max().item()",
    # the gradient goes back through the block, after a ReLU as in a network
    "import torch\n\ntorch.manual_seed(0)\nblock = ResBlock(3)\na = torch.randn(1, 3, 5, 5, requires_grad=True)\n"
    "block(torch.relu(a)).sum().backward()\nround(a.grad.abs().sum().item(), 4)",
    # what PyTorch sees of a block of 5 channels: tensors of parameters, and numbers in them
    "block = ResBlock(5)\n[len(list(block.parameters())), sum(p.numel() for p in block.parameters())]",
]
_R_WRONG = {
    "no-skip": _r(forward="        return self.conv2(torch.relu(self.conv1(x)))\n"),
    "relu-after": _r(forward="        return torch.relu(x + self.conv2(torch.relu(self.conv1(x))))\n"),
    "relu-end": _r(forward="        return x + torch.relu(self.conv2(torch.relu(self.conv1(x))))\n"),
    "no-inner-relu": _r(forward="        return x + self.conv2(self.conv1(x))\n"),
    "one-conv": _r(forward="        return x + self.conv1(x)\n"),
    "relu-x": _r(forward="        return torch.relu(x) + self.conv2(torch.relu(self.conv1(x)))\n"),
}


EXERCISES = {
    "res-block": ClassBox(
        name="res-block",
        start=_R_START,
        solution=_R,
        scenarios=_R_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "resnet"),
        first=[_DOTS],
        mistakes=[
            Mistake("sizes", error=lambda e: isinstance(e, RuntimeError) and "must match the size" in str(e),
                    en="`x` and `f(x)` have different shapes, so `+` cannot add them. `f(x)` keeps the shape of `x` when both convolutions go from `c` to `c` channels, with a filter of 3 by 3 and `padding=1`: without padding, every convolution of 3 by 3 takes 2 off the height and width.",
                    nl="`x` en `f(x)` hebben verschillende shapes, dus `+` kan ze niet optellen. `f(x)` houdt de shape van `x` als beide convoluties van `c` naar `c` channels gaan, met een filter van 3 bij 3 en `padding=1`: zonder padding haalt elke convolutie van 3 bij 3 er 2 af van de hoogte en breedte."),
            Mistake("in-place", error=lambda e: isinstance(e, RuntimeError) and "inplace operation" in str(e),
                    en="The backward pass stops: your block changes `x` itself, as `x += ...` does, and PyTorch still needs `x` as it was on the way back, for instance for the gradient of the weights of `conv1`. Make a new tensor: `return x + ...`.",
                    nl="De backward pass stopt: je block verandert `x` zelf, zoals `x += ...` doet, en PyTorch heeft `x` op de terugweg nog nodig zoals hij was, bijvoorbeeld voor de gradient van de weights van `conv1`. Maak een nieuwe tensor: `return x + ...`."),
            Mistake("no-skip", wrong=_R_WRONG["no-skip"],
                    en=_C_EN + ": your block gives only `f(x)`, as a plain network does. A residual block adds its input: `x + f(x)`.",
                    nl=_C_NL + ": je blok geeft alleen `f(x)`, zoals een gewoon netwerk. Een residual block telt zijn input erbij op: `x + f(x)`."),
            Mistake("relu-after", wrong=_R_WRONG["relu-after"],
                    en=_C_EN + ": you put a ReLU after the addition, as the first ResNet of 2015 did. Where `x + f(x)` is below 0, it then gives 0, and the direct path is cut. As in your block of [Residual connections](/course/open/nn-residual?lang=en), give back `x + f(x)` as it is.",
                    nl=_C_NL + ": je zet een ReLU na het optellen, zoals het eerste ResNet van 2015 deed. Waar `x + f(x)` onder 0 is, geeft die dan 0, en is het directe pad doorgeknipt. Geef, zoals in je block uit [Residual connections](/course/open/nn-residual?lang=nl), `x + f(x)` terug zoals het is."),
            Mistake("relu-end", wrong=_R_WRONG["relu-end"],
                    en=_C_EN + ": your `f` ends with a ReLU, so it can only add to `x`, never take away. The ReLU goes between the two convolutions only.",
                    nl=_C_NL + ": je `f` eindigt met een ReLU, dus hij kan alleen bij `x` optellen, nooit eraf halen. De ReLU staat alleen tussen de twee convoluties."),
            Mistake("no-inner-relu", wrong=_R_WRONG["no-inner-relu"],
                    en=_C_EN + ": your `f` has no ReLU between the two convolutions, so together they are one linear step. Put `torch.relu` around `self.conv1(x)`.",
                    nl=_C_NL + ": je `f` heeft geen ReLU tussen de twee convoluties, dus samen zijn ze één lineaire stap. Zet `torch.relu` om `self.conv1(x)`."),
            Mistake("one-conv", wrong=_R_WRONG["one-conv"],
                    en=_C_EN + ": your `f` is one convolution. In a residual block of ResNet, `f` is two: `conv1`, a ReLU, and `conv2`.",
                    nl=_C_NL + ": je `f` is één convolutie. In een residual block van ResNet is `f` er twee: `conv1`, een ReLU, en `conv2`."),
            Mistake("relu-x", wrong=_R_WRONG["relu-x"],
                    en=_C_EN + ": you put a ReLU on `x` before adding it, so the numbers of `x` below 0 do not get through. Add `x` as it is: `x + f(x)`.",
                    nl=_C_NL + ": je zet een ReLU op `x` voordat je hem optelt, dus de getallen van `x` onder 0 komen er niet door. Tel `x` op zoals hij is: `x + f(x)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: in `__init__` the two convolutions, and in `forward` the input plus `f(x)`.",
                      "nl": "Vervang de twee `...`: in `__init__` de twee convoluties, en in `forward` de input plus `f(x)`."},
            "wrong": {"en": _C_EN + ". `f(x)` is `conv1`, a ReLU, and `conv2`, each convolution of 3 by 3 from `c` to `c` channels with padding 1, made in that order so that they get the same random weights as the course's; the block gives `x + f(x)`.",
                      "nl": _C_NL + ". `f(x)` is `conv1`, een ReLU, en `conv2`, elke convolutie van 3 bij 3 van `c` naar `c` channels met padding 1, in die volgorde gemaakt, zodat ze dezelfde willekeurige weights krijgen als bij de course; het block geeft `x + f(x)`."},
            "almost": {"en": "Right for the first block. But " + _C_EN[0].lower() + _C_EN[1:] + ". Does your block pass `x` on when `f(x)` is 0, does the gradient get back through it, and does PyTorch see both convolutions?",
                       "nl": "Goed voor het eerste blok. Maar " + _C_NL[0].lower() + _C_NL[1:] + ". Geeft je blok `x` door als `f(x)` 0 is, komt de gradient erdoorheen terug, en ziet PyTorch beide convoluties?"},
            "correct": {"en": "Your residual block is right: `x + f(x)`, with `f` two convolutions of 3 by 3 and a ReLU between them. Your `ResBlock` is now in your project, as `{saved}`.",
                        "nl": "Je residual block klopt: `x + f(x)`, met `f` twee convoluties van 3 bij 3 en een ReLU ertussen. Je `ResBlock` staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `ResBlock` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `ResBlock` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("res-block", _R_START, "empty", None),
    ("res-block", _R, "correct", None),
    ("res-block", _r(forward="        f = self.conv2(torch.relu(self.conv1(x)))\n        return f + x\n"), "correct", None),
    ("res-block", _r(init=_R_INIT.replace(", padding=1)", ")")), "error", "sizes"),
    ("res-block", _r(forward="        h = self.conv2(torch.relu(self.conv1(x)))\n        x += h\n        return x\n"), "error", "in-place"),
    ("res-block", _r(forward="        return x + self.conv2(torch.relu(self.conv1(torch.relu(x))))\n"), "wrong", None),
    ("res-block", _r(init=_R_INIT.replace("padding=1)", "padding=1, stride=2)", 1)), "error", "sizes"),
    ("res-block", _r(forward="        x.add_(self.conv2(torch.relu(self.conv1(x))))\n        return x\n"), "error", "in-place"),
] + [("res-block", code, "wrong", key) for key, code in _R_WRONG.items()]


# --- the demo: a plain network and a residual one of the same depth -------------------------------------------------------------

class _PlainBlock(torch.nn.Module):
    """The learner's block without its residual connection: two convolutions, each with a ReLU after it. Its layers are
    made in the same order as in ResBlock, so that both networks start from the same weights."""
    def __init__(self, c):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv2(torch.relu(self.conv1(x))))


def _network(Block, m: int = BLOCKS):
    """A stem from 1 to 8 channels, m blocks of 8 channels, pooling, a convolution from 8 to 16 channels, m blocks of 16
    channels, pooling, and the output head: 2 + 4m convolutions."""
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = torch.nn.Conv2d(1, 8, 3, padding=1)
            self.stage1 = torch.nn.ModuleList([Block(8) for _ in range(m)])
            self.wider = torch.nn.Conv2d(8, 16, 3, padding=1)
            self.stage2 = torch.nn.ModuleList([Block(16) for _ in range(m)])
            self.pool = torch.nn.MaxPool2d(2)
            self.out = torch.nn.Linear(16 * 7 * 7, 10)

        def forward(self, x):
            x = torch.relu(self.stem(x))
            for block in self.stage1:
                x = block(x)
            x = torch.relu(self.wider(self.pool(x)))
            for block in self.stage2:
                x = block(x)
            return self.out(self.pool(x).reshape(len(x), -1))
    return Net()


def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def _accuracy(model, X, y) -> int:
    model.eval()
    with torch.no_grad():
        return int((model(X).argmax(dim=1) == y).sum())


def _demo():
    """For the demo: None when resnet.py is not in the project yet; "resnet" when a network of the learner's ResBlock
    stops with an error; "trainer" when the Trainer or count_params does; "slow" when a training run takes too long;
    else a dict with the parameters and the test images read right of the plain and the residual network, one run each,
    seed 0, and the modules the course's versions stood in for."""
    if not _saved("resnet"):
        return None
    run = run_code("from cnn.resnet import ResBlock")
    if run.error is not None:
        return "resnet"
    ResBlock = run.names["ResBlock"]
    run = run_code("from nn.trainer import Trainer\nfrom cnn.count import count_params")
    if run.error is not None:
        return "trainer"
    Trainer, count_params = run.names["Trainer"], run.names["count_params"]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    X, y, X_test, y_test = fashion()

    def train(model):
        torch.manual_seed(1)
        loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
        val = DataLoader(TensorDataset(X_test, y_test), batch_size=500)
        Trainer(model, loader, val, epochs=EPOCHS, lr=LR, warmup=len(loader), max_norm=1.0).fit()
        return _accuracy(model, X_test, y_test)

    out = {}
    for name, Block in (("plain", _PlainBlock), ("residual", ResBlock)):
        try:
            torch.manual_seed(0)
            model = _network(Block)
        except Exception:
            return "resnet"
        try:
            out["params_" + name] = int(count_params(model))
        except Exception:
            return "trainer"
        try:
            out[name] = with_time_limit(lambda: train(model), 240)
        except Timeout:
            return "slow"
        except Exception:
            return "resnet" if name == "residual" else "trainer"
    return {**out, "stand_ins": stand_ins}


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file)."""
    texts = []
    for package, module in (("cnn", "resnet"), ("cnn", "count"), ("nn", "train_epoch"), ("nn", "trainer")):
        path = projects.folder() / package / f"{module}.py"
        texts.append(path.read_text(encoding="utf-8") if path.exists() else None)
    return tuple(texts)


def demo():
    """The demo's result, worked out once for the project files as they are."""
    key = _project_texts()
    if key not in _DEMO_RUNS:
        _DEMO_RUNS.clear()
        _DEMO_RUNS[key] = _demo()
    return _DEMO_RUNS[key]
