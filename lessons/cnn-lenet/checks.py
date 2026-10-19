# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson cnn-lenet: LeNet5, the network of LeCun and others (1998) rebuilt in PyTorch for the
28 by 28 images of Fashion-MNIST, saved in the learner's project (work/projects/cnn/lenet.py) when right. The picture
follows one image through it; the demo trains it and the learner's SmallCNN of cnn-pytorch with their Trainer, and
compares their parameters and the test images they read right."""
import functools
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import draw, paths, projects
from everything_ai.check import ClassBox, Mistake, Timeout, run_code, with_time_limit

# The demo trains with the learner's Trainer of nn-trainer, and compares with their SmallCNN and counts with their
# count_params of cnn-pytorch; where one is missing or broken, the course's version stands in (template 1.3).
projects.builds_on(("nn", "train_epoch", "nn-trainer", "train-epoch"), ("nn", "trainer", "nn-trainer", "trainer"),
                   ("cnn", "small_cnn", "cnn-pytorch", "small-cnn"), ("cnn", "count", "cnn-pytorch", "count-params"))

DATA = "data/fashion-mnist-small.npz"   # 6,000 training and 1,000 test images of Fashion-MNIST: see data/README.md
EPOCHS, LR = 5, 0.01


@functools.lru_cache(maxsize=1)
def fashion():
    """The training and test images as tensors of 0 to 1, each image as one channel of 28 by 28, and their labels."""
    d = np.load(paths.ROOT / DATA)
    X = torch.tensor(d["train_images"], dtype=torch.float32).reshape(-1, 1, 28, 28) / 255
    y = torch.tensor(d["train_labels"], dtype=torch.int64)
    X_test = torch.tensor(d["test_images"], dtype=torch.float32).reshape(-1, 1, 28, 28) / 255
    y_test = torch.tensor(d["test_labels"], dtype=torch.int64)
    return X, y, X_test, y_test


def shapes_picture(labels: dict):
    """One image on its way through LeNet5: its shape after every step, and the steps between."""
    maps = labels["maps"]
    stages = [("(1, 28, 28)", labels["input"], 1, 28, "data"),
              ("(6, 28, 28)", maps.format(6, 28), 6, 28, "hidden"),
              ("(6, 14, 14)", maps.format(6, 14), 6, 14, "hidden"),
              ("(16, 10, 10)", maps.format(16, 10), 16, 10, "hidden"),
              ("(16, 5, 5)", maps.format(16, 5), 16, 5, "hidden"),
              ("(120,)", labels["fc"].format(120), 120, None, "hidden"),
              ("(84,)", labels["fc"].format(84), 84, None, "hidden"),
              ("(10,)", labels["logits"], 10, None, "pred")]
    steps = [labels["conv1"], labels["pool"], labels["conv2"], labels["pool"],
             labels["flat"], labels["linear"], labels["head"]]
    return draw.feature_maps(stages, steps, title=labels["title"], alt=labels["alt"])


def grouped(n: int, lang: str) -> str:
    """A whole number with its thousands marked, as the language writes it: 61,706 or 61.706."""
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


# The lesson is done when these are correct. "lenet-shape" and "where-params" are the choice questions in the lesson.
REQUIRED = ["lenet-shape", "lenet5", "where-params"]

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


# --- exercise 2: LeNet5 ----------------------------------------------------------------------------------------------------------

_L_HEAD = "import torch\nfrom torch import nn\n\n\nclass LeNet5(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
_L_INIT = ("        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)\n        self.conv2 = nn.Conv2d(6, 16, 5)\n"
           "        self.pool = nn.AvgPool2d(2)\n        self.fc1 = nn.Linear(16 * 5 * 5, 120)\n"
           "        self.fc2 = nn.Linear(120, 84)\n        self.out = nn.Linear(84, 10)\n")
_L_FORWARD = ("        x = self.pool(torch.tanh(self.conv1(x)))\n        x = self.pool(torch.tanh(self.conv2(x)))\n"
              "        x = x.reshape(len(x), -1)\n        x = torch.tanh(self.fc1(x))\n        x = torch.tanh(self.fc2(x))\n"
              "        return self.out(x)\n")
_L_START = (_L_HEAD + "        # in the order the data passes them: conv1, 5 by 5 from 1 to 6 channels with padding 2;\n"
            "        # conv2, 5 by 5 from 6 to 16 without padding; average pooling of 2 by 2;\n"
            "        # fc1 from 16 * 5 * 5 to 120, fc2 from 120 to 84, and the output head from 84 to 10\n        ...\n\n"
            "    def forward(self, x):\n"
            "        # two blocks of convolution, tanh and pooling; one row per image; fc1 and fc2 with tanh; the head\n        ...\n")


def _l(init: str = _L_INIT, forward: str = _L_FORWARD) -> str:
    return _L_HEAD + init + "\n    def forward(self, x):\n" + forward


_L = _l()
_L_DATA = "import torch\n\ntorch.manual_seed(0)\nmodel = LeNet5()\n"
_L_SCENARIOS = [
    # a mini-batch of 2 images: the shape of the logits and the first four of the first image
    _L_DATA + "x = torch.rand(2, 1, 28, 28)\nlogits = model(x)\n[list(logits.shape), logits[0, :4].tolist()]",
    # a mini-batch of 3: does the network work for any number of images?
    _L_DATA + "x = torch.rand(3, 1, 28, 28)\nlogits = model(x)\n[list(logits.shape), logits[2, :4].tolist()]",
    # how many tensors of parameters PyTorch sees: two convolutions and three linear layers, each a weight and a bias
    _L_DATA + "len(list(model.parameters()))",
    # all parameters 5 times as large: with the small starting weights tanh is almost a straight line, and a tanh
    # after the output head changes the logits by less than the check can see; here they go beyond -1 and 1
    (_L_DATA + "with torch.no_grad():\n    for p in model.parameters():\n        p *= 5\n"
     "x = torch.rand(2, 1, 28, 28)\nlogits = model(x)\n[list(logits.shape), logits[0, :4].tolist()]"),
]

_L_WRONG = {
    "relu": _l(forward=_L_FORWARD.replace("torch.tanh", "torch.relu")),
    "max-pool": _l(init=_L_INIT.replace("nn.AvgPool2d(2)", "nn.MaxPool2d(2)")),
    "no-tanh-fc": _l(forward=_L_FORWARD.replace("        x = torch.tanh(self.fc1(x))\n        x = torch.tanh(self.fc2(x))\n",
                                                "        x = self.fc1(x)\n        x = self.fc2(x)\n")),
    "tanh-out": _l(forward=_L_FORWARD.replace("return self.out(x)", "return torch.tanh(self.out(x))")),
}


def _out_of_order(got, text) -> bool:
    """The layers with weights made in another order than the data passes them, and logits of the right shape:
    other random starting values."""
    fc1 = text.find("nn.Linear(16") if "nn.Linear(16" in text else text.find("nn.Linear(400")
    at = [text.find("nn.Conv2d(1,"), text.find("nn.Conv2d(6,"), fc1, text.find("nn.Linear(120"), text.find("nn.Linear(84")]
    shape = got[0] if isinstance(got, list) and got and isinstance(got[0], list) else None
    return shape is not None and len(shape) == 2 and shape[1] == 10 and min(at) >= 0 and at != sorted(at)


EXERCISES = {
    "lenet5": ClassBox(
        name="lenet5",
        start=_L_START,
        solution=_L,
        scenarios=_L_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "lenet"),
        first=[_DOTS],
        mistakes=[
            Mistake("one-row", syntax=r"^[ \t]*\w+ = .*(\.reshape\(\s*-1\s*\)|\.view\(\s*-1\s*\)|\.flatten\(\s*\)|torch\.flatten\(\s*\w+\s*\))",
                    en="`{line}` lays out the whole mini-batch as one long row, so `fc1` gets the numbers of all images at once. Keep one row per image: `x.reshape(len(x), -1)`.",
                    nl="`{line}` maakt van de hele mini-batch één lange rij, dus `fc1` krijgt de getallen van alle afbeeldingen tegelijk. Houd één rij per afbeelding: `x.reshape(len(x), -1)`."),
            Mistake("batch-size", syntax=r"^[ \t]*\w+ = .*\.(?:reshape|view)\(\s*\d+\s*,",
                    en="`{line}` always makes the same number of rows, but a mini-batch can hold any number of images: the check gives 2, then 3. Take the number from `x` itself: `x.reshape(len(x), -1)`.",
                    nl="`{line}` maakt altijd hetzelfde aantal rijen, maar een mini-batch kan elk aantal afbeeldingen bevatten: de check geeft er 2, dan 3. Neem het aantal uit `x` zelf: `x.reshape(len(x), -1)`."),
            Mistake("channels", error=lambda e: isinstance(e, RuntimeError) and re.search(r"to have \d+ channels, but got \d+ channels", str(e)) is not None,
                    en="A convolutional layer gets another number of channels than it was made for. The first number of `nn.Conv2d` is the number of channels that come in: 1 for the gray image, and 6 for the feature maps of `conv1`.",
                    nl="Een convolutional layer krijgt een ander aantal channels dan waarvoor hij gemaakt is. Het eerste getal van `nn.Conv2d` is het aantal channels dat binnenkomt: 1 voor de grijze afbeelding, en 6 voor de feature maps van `conv1`."),
            Mistake("sizes", error=lambda e: isinstance(e, RuntimeError) and "cannot be multiplied" in str(e),
                    en="The sizes do not fit: a linear layer gets another number of inputs per image than it was made for. With padding 2, `conv1` keeps 28 by 28; pooling makes 14; `conv2` without padding makes 10; pooling makes 5. So 16 channels of 5 by 5 come out: one row of 400 numbers per image, with `x.reshape(len(x), -1)`. Then `fc1` goes from 400 to 120, `fc2` from 120 to 84, and the head from 84 to 10.",
                    nl="De maten passen niet: een linear layer krijgt per afbeelding een ander aantal inputs dan waarvoor hij gemaakt is. Met padding 2 houdt `conv1` 28 bij 28; pooling maakt er 14 van; `conv2` zonder padding 10; pooling 5. Er komen dus 16 channels van 5 bij 5 uit: één rij van 400 getallen per afbeelding, met `x.reshape(len(x), -1)`. Dan gaat `fc1` van 400 naar 120, `fc2` van 120 naar 84, en de head van 84 naar 10."),
            Mistake("relu", wrong=_L_WRONG["relu"],
                    en=_C_EN + ": your network uses ReLU. LeNet-5 used tanh, which squeezes every number between −1 and 1: `torch.tanh`, after both convolutions and after `fc1` and `fc2`.",
                    nl=_C_NL + ": je netwerk gebruikt ReLU. LeNet-5 gebruikte tanh, die elk getal tussen −1 en 1 perst: `torch.tanh`, na beide convoluties en na `fc1` en `fc2`."),
            Mistake("max-pool", wrong=_L_WRONG["max-pool"],
                    en=_C_EN + ": your network takes the largest number of every block of 2 by 2, and LeNet-5 took their average: `nn.AvgPool2d(2)`.",
                    nl=_C_NL + ": je netwerk neemt het grootste getal van elk blok van 2 bij 2, en LeNet-5 nam hun gemiddelde: `nn.AvgPool2d(2)`."),
            Mistake("no-tanh-fc", wrong=_L_WRONG["no-tanh-fc"],
                    en=_C_EN + ": `fc1` and `fc2` have no tanh after them, so the two layers together are one linear layer. Put `torch.tanh` around both.",
                    nl=_C_NL + ": na `fc1` en `fc2` komt geen tanh, dus samen zijn de twee layers één linear layer. Zet `torch.tanh` om allebei."),
            Mistake("tanh-out", wrong=_L_WRONG["tanh-out"],
                    en=_C_EN + ": your logits go through tanh too, so they stay between −1 and 1. The output head gives the logits as they are: the cross-entropy loss does the softmax itself.",
                    nl=_C_NL + ": je logits gaan ook door tanh, dus ze blijven tussen −1 en 1. De output head geeft de logits zoals ze zijn: de cross-entropy loss doet de softmax zelf."),
            Mistake("order", matches=_out_of_order,
                    en=_C_EN + ": the right shape, other numbers. Make the layers in the order in which the data passes them, `conv1`, `conv2`, `fc1`, `fc2`, `out`, so that they get the same random starting values as in the course's solution.",
                    nl=_C_NL + ": de goede shape, andere getallen. Maak de layers in de volgorde waarin de data erdoor gaat, `conv1`, `conv2`, `fc1`, `fc2`, `out`, zodat ze dezelfde willekeurige startwaarden krijgen als in de oplossing van de course."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the layers in `__init__`, and in `forward` the two blocks, one row per image, `fc1` and `fc2` with tanh, and the head.",
                      "nl": "Vervang de twee `...`: de layers in `__init__`, en in `forward` de twee blokken, één rij per afbeelding, `fc1` en `fc2` met tanh, en de head."},
            "wrong": {"en": _C_EN + ". Two blocks of convolution, tanh and average pooling, one row per image, `fc1` and `fc2` each with tanh, and the head: steps 1 to 4 of the recipe?",
                      "nl": _C_NL + ". Twee blokken van convolutie, tanh en average pooling, één rij per afbeelding, `fc1` en `fc2` elk met tanh, en de head: stap 1 tot en met 4 van het recept?"},
            "almost": {"en": "Right for 2 images. But " + _C_EN[0].lower() + _C_EN[1:] + ". Does your network work for any number of images, with `len(x)`, does it make each layer once, and is tanh after every layer but the output head?",
                       "nl": "Goed voor 2 afbeeldingen. Maar " + _C_NL[0].lower() + _C_NL[1:] + ". Werkt je netwerk voor elk aantal afbeeldingen, met `len(x)`, maakt het elke layer één keer, en staat tanh na elke layer behalve de output head?"},
            "correct": {"en": "Your LeNet-5 gives 10 logits per image, from two blocks of convolution, tanh and average pooling, and three linear layers. Your `LeNet5` is now in your project, as `{saved}`.",
                        "nl": "Je LeNet-5 geeft 10 logits per afbeelding, uit twee blokken van convolutie, tanh en average pooling, en drie linear layers. Je `LeNet5` staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `LeNet5` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `LeNet5` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("lenet5", _L_START, "empty", None),
    ("lenet5", _L, "correct", None),
    ("lenet5", _l(forward=("        x = self.pool(torch.tanh(self.conv1(x)))\n        x = self.pool(torch.tanh(self.conv2(x)))\n"
                           "        x = x.reshape(x.shape[0], 400)\n        return self.out(torch.tanh(self.fc2(torch.tanh(self.fc1(x)))))\n")), "correct", None),
    ("lenet5", _l(forward=_L_FORWARD.replace("x.reshape(len(x), -1)", "x.reshape(-1)")), "error", "one-row"),
    ("lenet5", _l(init=_L_INIT.replace("nn.Conv2d(6, 16, 5)", "nn.Conv2d(16, 16, 5)")), "error", "channels"),
    ("lenet5", _l(init=_L_INIT.replace(", padding=2)", ")")), "error", "sizes"),
    ("lenet5", _L_WRONG["relu"], "wrong", "relu"),
    ("lenet5", _L_WRONG["max-pool"], "wrong", "max-pool"),
    ("lenet5", _L_WRONG["no-tanh-fc"], "wrong", "no-tanh-fc"),
    ("lenet5", _l(init=("        self.out = nn.Linear(84, 10)\n        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)\n"
                        "        self.conv2 = nn.Conv2d(6, 16, 5)\n        self.pool = nn.AvgPool2d(2)\n"
                        "        self.fc1 = nn.Linear(16 * 5 * 5, 120)\n        self.fc2 = nn.Linear(120, 84)\n")), "wrong", "order"),
    ("lenet5", _l(init=("        self.out = nn.Linear(84, 10)\n        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)\n"
                        "        self.conv2 = nn.Conv2d(6, 16, 5)\n        self.pool = nn.AvgPool2d(2)\n"
                        "        self.fc1 = nn.Linear(400, 120)\n        self.fc2 = nn.Linear(120, 84)\n")), "wrong", "order"),
    ("lenet5", _L_WRONG["tanh-out"], "wrong", "tanh-out"),
    ("lenet5", _l(forward=_L_FORWARD.replace("x.reshape(len(x), -1)", "x.reshape(2, -1)")), "error", "batch-size"),
    ("lenet5", _l(init=_L_INIT.replace("nn.Linear(84, 10)", "nn.Linear(120, 10)")), "error", "sizes"),
    ("lenet5", _l(forward=_L_FORWARD.replace("        x = x.reshape(len(x), -1)\n", "")), "error", "sizes"),
    ("lenet5", _l(forward=_L_FORWARD.replace("torch.tanh(self.fc2(x))", "self.fc2(x)")), "wrong", None),
]


# --- the demo: LeNet-5 against the learner's SmallCNN -----------------------------------------------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def _accuracy(model, X, y) -> int:
    model.eval()
    with torch.no_grad():
        return int((model(X).argmax(dim=1) == y).sum())


def _demo():
    """For the demo: None when lenet.py is not in the project yet; "lenet" when LeNet5 stops with an error; "trainer"
    when the Trainer or SmallCNN stops with an error; "slow" when a training run takes too long; else a dict with the
    test images read right and the parameters of both networks, one run each, seed 0, and the modules the course's
    versions stood in for."""
    if not _saved("lenet"):
        return None
    run = run_code("from cnn.lenet import LeNet5")
    if run.error is not None:
        return "lenet"
    LeNet5 = run.names["LeNet5"]
    run = run_code("from nn.trainer import Trainer\nfrom cnn.small_cnn import SmallCNN\nfrom cnn.count import count_params")
    if run.error is not None:
        return "trainer"
    Trainer, SmallCNN, count_params = run.names["Trainer"], run.names["SmallCNN"], run.names["count_params"]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    X, y, X_test, y_test = fashion()

    def train(model):
        torch.manual_seed(1)
        loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
        val = DataLoader(TensorDataset(X_test, y_test), batch_size=500)
        Trainer(model, loader, val, epochs=EPOCHS, lr=LR, warmup=len(loader), max_norm=1.0).fit()
        return _accuracy(model, X_test, y_test)

    out = {}
    for name, cls in (("lenet", LeNet5), ("small", SmallCNN)):
        try:
            torch.manual_seed(0)
            model = cls()
            out["params_" + name] = int(count_params(model))
            out[name] = with_time_limit(lambda: train(model), 120)
        except Timeout:
            return "slow"
        except Exception:
            return "lenet" if name == "lenet" else "trainer"
    return {**out, "stand_ins": stand_ins}


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file)."""
    texts = []
    for package, module in (("cnn", "lenet"), ("cnn", "small_cnn"), ("cnn", "count"), ("nn", "train_epoch"), ("nn", "trainer")):
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
