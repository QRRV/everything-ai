# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson cnn-pytorch: SmallCNN, a small convolutional network for the 28 by 28 images of
Fashion-MNIST, built with nn.Conv2d and nn.MaxPool2d; and count_params, how many numbers a model learns. Both are saved
in the learner's project (work/projects/cnn/) when right. The picture follows one image through the network; the demo
runs the learner's overfit test on their CNN, then trains it and their deep MLP with their Trainer, and compares."""
import functools
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import draw, paths, projects
from everything_ai.check import ClassBox, Mistake, Timeout, run_code, with_time_limit

# The demo trains with the learner's Trainer of nn-trainer and runs their overfit test and DeepMLP of
# nn-deep-net-project, or the course's versions where theirs are missing or broken. The Trainer imports train_epoch.
projects.builds_on(("nn", "train_epoch", "nn-trainer", "train-epoch"), ("nn", "trainer", "nn-trainer", "trainer"),
                   ("nn", "overfit", "nn-deep-net-project", "overfit"), ("nn", "deep_mlp", "nn-deep-net-project", "deep-mlp"))

DATA = "data/fashion-mnist-small.npz"   # 6,000 training and 1,000 test images of Fashion-MNIST: see data/README.md
EPOCHS, LR = 5, 0.01
OVERFIT_STEPS, OVERFIT_LR = 300, 0.1


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
    """One image on its way through SmallCNN: its shape after every step, and the steps between."""
    maps = labels["maps"]
    stages = [("(1, 28, 28)", labels["input"], 1, 28, "data"),
              ("(8, 28, 28)", maps.format(8, 28), 8, 28, "hidden"),
              ("(8, 14, 14)", maps.format(8, 14), 8, 14, "hidden"),
              ("(16, 14, 14)", maps.format(16, 14), 16, 14, "hidden"),
              ("(16, 7, 7)", maps.format(16, 7), 16, 7, "hidden"),
              ("(10,)", labels["logits"], 10, None, "pred")]
    steps = [labels["conv"].format(1, 8), labels["pool"], labels["conv"].format(8, 16), labels["pool"], labels["head"]]
    return draw.feature_maps(stages, steps, title=labels["title"], alt=labels["alt"])


def grouped(n: int, lang: str) -> str:
    """A whole number with its thousands marked, as the language writes it: 9,098 or 9.098."""
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


# The lesson is done when these are correct. "shape-after" is the choice question in the lesson.
REQUIRED = ["shape-after", "small-cnn", "count-params"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_SCENARIO = "With these lines after your {what}:\n\n```python\n{{scenario}}\n```\n\nyou get `{{got}}`, and it should be `{{want}}`"
_SCENARIO_NL = "Met deze regels na je {what}:\n\n```python\n{{scenario}}\n```\n\nkrijg je `{{got}}`, en dat moet `{{want}}` zijn"
_F_EN, _F_NL = _SCENARIO.format(what="function"), _SCENARIO_NL.format(what="functie")
_C_EN, _C_NL = _SCENARIO.format(what="class"), _SCENARIO_NL.format(what="class")


def _close(got, want) -> bool:
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    try:
        return bool(np.allclose(float(got), float(want), rtol=1e-4, atol=1e-4))
    except (TypeError, ValueError):
        return got == want


# --- exercise 2: SmallCNN -------------------------------------------------------------------------------------------------

_S_HEAD = "import torch\nfrom torch import nn\n\n\nclass SmallCNN(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
_S_INIT = ("        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)\n        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)\n"
           "        self.pool = nn.MaxPool2d(2)\n        self.out = nn.Linear(16 * 7 * 7, 10)\n")
_S_FORWARD = ("        x = self.pool(torch.relu(self.conv1(x)))\n        x = self.pool(torch.relu(self.conv2(x)))\n"
              "        x = x.reshape(len(x), -1)\n        return self.out(x)\n")
_S_START = (_S_HEAD + "        # two convolutional layers of 3 by 3 with padding 1: from 1 to 8 channels, and from 8 to 16;\n"
            "        # max pooling of 2 by 2; and the output head\n        ...\n\n    def forward(self, x):\n"
            "        # two blocks of convolution, ReLU and pooling; then one row per image; then the output head\n        ...\n")


def _s(init: str = _S_INIT, forward: str = _S_FORWARD) -> str:
    return _S_HEAD + init + "\n    def forward(self, x):\n" + forward


_S = _s()
_S_DATA = "import torch\n\ntorch.manual_seed(0)\nmodel = SmallCNN()\n"
_S_SCENARIOS = [
    # a mini-batch of 2 images: the shape of the logits and the first four of the first image
    _S_DATA + "x = torch.rand(2, 1, 28, 28)\nlogits = model(x)\n[list(logits.shape), logits[0, :4].tolist()]",
    # a mini-batch of 3: does the network work for any number of images?
    _S_DATA + "x = torch.rand(3, 1, 28, 28)\nlogits = model(x)\n[list(logits.shape), logits[2, :4].tolist()]",
]


def _out_of_order(got, text) -> bool:
    """The three layers with weights made in another order than the data passes them, and logits of the right shape:
    other random starting values."""
    at = [text.find(made) for made in ("nn.Conv2d(1,", "nn.Conv2d(8,", "nn.Linear(")]
    shape = got[0] if isinstance(got, list) and got and isinstance(got[0], list) else None
    return shape is not None and len(shape) == 2 and shape[1] == 10 and min(at) >= 0 and at != sorted(at)


# --- exercise 3: count_params ---------------------------------------------------------------------------------------------

_P_START = "def count_params(model):\n    # how many numbers all parameters of the model hold together\n    ...\n"
_P_HEAD = "def count_params(model):\n    total = 0\n    for p in model.parameters():\n"


def _p(step: str = "        total = total + p.numel()\n") -> str:
    return _P_HEAD + step + "    return total\n"


_P = _p()
_P_SCENARIOS = [
    "from torch import nn\n\ncount_params(nn.Linear(3, 2))",
    "from torch import nn\n\ncount_params(nn.Conv2d(8, 16, 3, padding=1))",
    "from torch import nn\n\ncount_params(nn.MaxPool2d(2))",
]

EXERCISES = {
    "small-cnn": ClassBox(
        name="small-cnn",
        start=_S_START,
        solution=_S,
        scenarios=_S_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "small_cnn"),
        first=[_DOTS],
        mistakes=[
            Mistake("one-row", syntax=r"^[ \t]*\w+ = .*(\.reshape\(\s*-1\s*\)|\.view\(\s*-1\s*\)|\.flatten\(\s*\)|torch\.flatten\(\s*\w+\s*\))",
                    en="`{line}` lays out the whole mini-batch as one long row, so the output head gets the numbers of all images at once. Keep one row per image: `x.reshape(len(x), -1)`.",
                    nl="`{line}` maakt van de hele mini-batch één lange rij, dus de output head krijgt de getallen van alle afbeeldingen tegelijk. Houd één rij per afbeelding: `x.reshape(len(x), -1)`."),
            Mistake("batch-size", syntax=r"^[ \t]*\w+ = .*\.(?:reshape|view)\(\s*\d+\s*,",
                    en="`{line}` always makes the same number of rows, but a mini-batch can hold any number of images: the check gives 2, then 3. Take the number from `x` itself: `x.reshape(len(x), -1)`.",
                    nl="`{line}` maakt altijd hetzelfde aantal rijen, maar een mini-batch kan elk aantal afbeeldingen bevatten: de check geeft er 2, dan 3. Neem het aantal uit `x` zelf: `x.reshape(len(x), -1)`."),
            Mistake("channels", error=lambda e: isinstance(e, RuntimeError) and re.search(r"to have \d+ channels, but got \d+ channels", str(e)) is not None,
                    en="A convolutional layer gets another number of channels than it was made for. The first number of `nn.Conv2d` is the number of channels that come in: 1 for the gray image, and 8 for the feature maps of the first layer.",
                    nl="Een convolutional layer krijgt een ander aantal channels dan waarvoor hij gemaakt is. Het eerste getal van `nn.Conv2d` is het aantal channels dat binnenkomt: 1 voor de grijze afbeelding, en 8 voor de feature maps van de eerste layer."),
            Mistake("sizes", error=lambda e: isinstance(e, RuntimeError) and "cannot be multiplied" in str(e),
                    en="The sizes do not fit: your output head gets another number of inputs per image than it was made for. With padding 1, each convolution keeps the height and width, and each pooling of 2 halves them: 28, 14, 7. So 16 channels of 7 by 7 come out, 784 numbers per image, for `nn.Linear(16 * 7 * 7, 10)`, with one row per image: `x.reshape(len(x), -1)`.",
                    nl="De maten passen niet: je output head krijgt per afbeelding een ander aantal inputs dan waarvoor hij gemaakt is. Met padding 1 houdt elke convolutie de hoogte en breedte, en elke pooling van 2 halveert ze: 28, 14, 7. Er komen dus 16 channels van 7 bij 7 uit, 784 getallen per afbeelding, voor `nn.Linear(16 * 7 * 7, 10)`, met één rij per afbeelding: `x.reshape(len(x), -1)`."),
            Mistake("no-relu", wrong=_s(forward=_S_FORWARD.replace("torch.relu(self.conv1(x))", "self.conv1(x)").replace("torch.relu(self.conv2(x))", "self.conv2(x)")),
                    en=_C_EN + ": your blocks have no ReLU. In each block, ReLU comes between the convolution and the pooling, as in step 2 of the recipe: `self.pool(torch.relu(self.conv1(x)))`.",
                    nl=_C_NL + ": je blokken hebben geen ReLU. In elk blok komt ReLU tussen de convolutie en de pooling, zoals in stap 2 van het recept: `self.pool(torch.relu(self.conv1(x)))`."),
            Mistake("softmax", wrong=_s(forward=_S_FORWARD.replace("return self.out(x)", "return torch.softmax(self.out(x), dim=1)")),
                    en=_C_EN + ": your network gives probabilities. It should give the logits: `F.cross_entropy` takes the softmax itself, so a softmax before it would be taken twice.",
                    nl=_C_NL + ": je netwerk geeft kansen. Het moet de logits geven: `F.cross_entropy` neemt zelf de softmax, dus een softmax ervoor zou twee keer genomen worden."),
            Mistake("order", matches=_out_of_order,
                    en=_C_EN + ": the right shape, other numbers. Make the layers in the order in which the data passes them, the convolution from 1 channel, the one from 8, then the output head, so that they get the same random starting values as in the course's solution.",
                    nl=_C_NL + ": de goede shape, andere getallen. Maak de layers in de volgorde waarin de data erdoor gaat, de convolutie vanaf 1 channel, die vanaf 8, dan de output head, zodat ze dezelfde willekeurige startwaarden krijgen als in de oplossing van de course."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the layers in `__init__`, and in `forward` the two blocks, one row per image and the output head.",
                      "nl": "Vervang de twee `...`: de layers in `__init__`, en in `forward` de twee blokken, één rij per afbeelding en de output head."},
            "wrong": {"en": _C_EN + ": the shape of the logits and the first four of one image. Two blocks of convolution, ReLU and pooling, one row per image, the output head: steps 2 to 4 of the recipe?",
                      "nl": _C_NL + ": de shape van de logits en de eerste vier van één afbeelding. Twee blokken van convolutie, ReLU en pooling, één rij per afbeelding, de output head: stap 2 tot en met 4 van het recept?"},
            "almost": {"en": "Right for 2 images. But " + _C_EN[0].lower() + _C_EN[1:] + ". Does your network work for any number of images, with `len(x)` and no fixed batch size?",
                       "nl": "Goed voor 2 afbeeldingen. Maar " + _C_NL[0].lower() + _C_NL[1:] + ". Werkt je netwerk voor elk aantal afbeeldingen, met `len(x)` en geen vaste batchgrootte?"},
            "correct": {"en": "Your CNN gives 10 logits per image, from two blocks of convolution, ReLU and pooling. Your `SmallCNN` is now in your project, as `work/projects/cnn/small_cnn.py`.",
                        "nl": "Je CNN geeft 10 logits per afbeelding, uit twee blokken van convolutie, ReLU en pooling. Je `SmallCNN` staat nu in je project, als `work/projects/cnn/small_cnn.py`."},
            "kept": {"en": "Your `SmallCNN` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `SmallCNN` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "count-params": ClassBox(
        name="count-params",
        start=_P_START,
        solution=_P,
        scenarios=_P_SCENARIOS,
        compare=_close,
        save_as=("cnn", "count"),
        first=[_DOTS],
        mistakes=[
            Mistake("tensors", wrong=_p("        total = total + 1\n"),
                    en=_F_EN + ": you count the parameters, the tensors, not the numbers in them. `p.numel()` gives how many numbers one parameter holds.",
                    nl=_F_NL + ": je telt de parameters, de tensors, niet de getallen erin. `p.numel()` geeft hoeveel getallen één parameter bevat."),
            Mistake("first-axis", wrong=_p("        total = total + len(p)\n"),
                    en=_F_EN + ": you count only the length of the first axis, as `len(p)` gives it: the 2 rows of a weight matrix of shape (2, 3). `p.numel()` counts all numbers: 6.",
                    nl=_F_NL + ": je telt alleen de lengte van de eerste axis, zoals `len(p)` die geeft: de 2 rijen van een weight-matrix met shape (2, 3). `p.numel()` telt alle getallen: 6."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: go over `model.parameters()` and add up how many numbers each one holds.",
                      "nl": "Vervang de `...`: loop over `model.parameters()` en tel op hoeveel getallen elk bevat."},
            "wrong": {"en": _F_EN + ". Does your function go over all of `model.parameters()`, and add `p.numel()` for each?",
                      "nl": _F_NL + ". Loopt je functie over heel `model.parameters()`, en telt hij `p.numel()` op voor elk?"},
            "almost": {"en": "Right for `nn.Linear`. But " + _F_EN[0].lower() + _F_EN[1:] + ". Does your function start at 0, and count every parameter, of any layer?",
                       "nl": "Goed voor `nn.Linear`. Maar " + _F_NL[0].lower() + _F_NL[1:] + ". Begint je functie bij 0, en telt hij elke parameter, van elke layer?"},
            "correct": {"en": "Your function counts what a model learns: 8 numbers for `nn.Linear(3, 2)`, and 0 for pooling, which learns nothing. Your `count_params` is now in your project, as `work/projects/cnn/count.py`.",
                        "nl": "Je functie telt wat een model leert: 8 getallen voor `nn.Linear(3, 2)`, en 0 voor pooling, dat niets leert. Je `count_params` staat nu in je project, als `work/projects/cnn/count.py`."},
            "kept": {"en": "Your `count_params` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `count_params` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}


# --- the demo: the learner's overfit test, CNN, deep MLP and Trainer on Fashion-MNIST ---------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def _accuracy(model, X, y) -> int:
    model.eval()
    with torch.no_grad():
        return int((model(X).argmax(dim=1) == y).sum())


def _demo():
    """For the demo: None when small_cnn.py or count.py is not in the project yet; "cnn" when SmallCNN or count_params
    stops with an error, also on a mini-batch of the training; "overfit" or "trainer" when the learner's module of nn-deep-net-project or nn-trainer
    does; "slow" when the CNN, the overfit test or a training run takes too long; else a dict with the last loss of the overfit test, the test images read
    right by the CNN and by the deep MLP (one run each, seed 0), their numbers of parameters, and the modules the
    course's versions stood in for."""
    if not (_saved("small_cnn") and _saved("count")):
        return None
    run = run_code("from cnn.small_cnn import SmallCNN\nfrom cnn.count import count_params")
    if run.error is not None:
        return "cnn"
    SmallCNN, count_params = run.names["SmallCNN"], run.names["count_params"]
    run = run_code("from nn.overfit import overfit_one_batch\nfrom nn.trainer import Trainer\nfrom nn.deep_mlp import DeepMLP")
    if run.error is not None:
        return "trainer"
    overfit_one_batch, Trainer, DeepMLP = run.names["overfit_one_batch"], run.names["Trainer"], run.names["DeepMLP"]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    X, y, X_test, y_test = fashion()

    def train(model, images, test_images):
        torch.manual_seed(1)
        loader = DataLoader(TensorDataset(images, y), batch_size=64, shuffle=True)
        val = DataLoader(TensorDataset(test_images, y_test), batch_size=500)
        Trainer(model, loader, val, epochs=EPOCHS, lr=LR, warmup=len(loader), max_norm=1.0).fit()
        return _accuracy(model, test_images, y_test)

    def probe():   # every batch size the demo gives the CNN: 64, the last 48 of an epoch, 500, and all 1,000 test images
        with torch.no_grad():
            for images in (X[:64], X[:len(X) % 64], X_test[:500], X_test):
                cnn(images)

    try:
        torch.manual_seed(0)
        cnn = SmallCNN()
        params_cnn = int(count_params(cnn))
        with_time_limit(probe, 60)
    except Timeout:
        return "slow"
    except Exception:
        return "cnn"
    try:
        loss = float(with_time_limit(lambda: overfit_one_batch(cnn, X[:64], y[:64], OVERFIT_STEPS, OVERFIT_LR), 60))
    except Timeout:
        return "slow"
    except Exception:
        return "overfit"
    try:
        torch.manual_seed(0)
        cnn = SmallCNN()
        read_cnn = with_time_limit(lambda: train(cnn, X, X_test), 120)
        torch.manual_seed(0)
        mlp = DeepMLP(10, 64, he=True, norm=True, residual=True)
        params_mlp = int(count_params(mlp))
        read_mlp = with_time_limit(lambda: train(mlp, X.reshape(len(X), -1), X_test.reshape(len(X_test), -1)), 120)
    except Timeout:
        return "slow"
    except Exception:
        return "trainer"
    return {"loss": loss, "cnn": read_cnn, "mlp": read_mlp, "params_cnn": params_cnn, "params_mlp": params_mlp,
            "stand_ins": stand_ins}


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file)."""
    texts = []
    for package, module in (("cnn", "small_cnn"), ("cnn", "count"), ("nn", "train_epoch"), ("nn", "trainer"),
                            ("nn", "overfit"), ("nn", "deep_mlp")):
        path = projects.folder() / package / f"{module}.py"
        texts.append(path.read_text(encoding="utf-8") if path.exists() else None)
    return tuple(texts)


def demo():
    """The demo's result, worked out once for the project files as they are: it is the same for the same files, so a
    check that saves the same code again does not train for ten seconds more."""
    key = _project_texts()
    if key not in _DEMO_RUNS:
        _DEMO_RUNS.clear()
        _DEMO_RUNS[key] = _demo()
    return _DEMO_RUNS[key]


# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("small-cnn", _S_START, "empty", None),
    ("small-cnn", _s(forward=("        x = self.pool(torch.relu(self.conv1(x)))\n        x = self.pool(torch.relu(self.conv2(x)))\n"
                              "        return self.out(x.reshape(x.shape[0], 16 * 7 * 7))\n")), "correct", None),
    ("small-cnn", _s(forward=_S_FORWARD.replace("x.reshape(len(x), -1)", "x.reshape(-1)")), "error", "one-row"),
    ("small-cnn", _s(init=_S_INIT.replace("nn.Conv2d(8, 16, 3, padding=1)", "nn.Conv2d(16, 16, 3, padding=1)")), "error", "channels"),
    ("small-cnn", _s(init=_S_INIT.replace(", padding=1)", ")")), "error", "sizes"),
    ("small-cnn", _s(forward=_S_FORWARD.replace("torch.relu(self.conv1(x))", "self.conv1(x)").replace("torch.relu(self.conv2(x))", "self.conv2(x)")), "wrong", "no-relu"),
    ("small-cnn", _s(forward=_S_FORWARD.replace("return self.out(x)", "return torch.softmax(self.out(x), dim=1)")), "wrong", "softmax"),
    ("small-cnn", _s(init=("        self.out = nn.Linear(16 * 7 * 7, 10)\n        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)\n"
                           "        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)\n        self.pool = nn.MaxPool2d(2)\n")), "wrong", "order"),
    ("small-cnn", _s(forward=_S_FORWARD.replace("x.reshape(len(x), -1)", "x.reshape(2, -1)")), "error", "batch-size"),
    ("small-cnn", _s(forward=_S_FORWARD.replace("x.reshape(len(x), -1)", "x.view(64, 784)")), "error", "batch-size"),
    ("small-cnn", _s(init=("        self.out = nn.Linear(16 * 7 * 7, 5)\n        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)\n"
                           "        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)\n        self.pool = nn.MaxPool2d(2)\n")), "wrong", None),
    ("count-params", _P_START, "empty", None),
    ("count-params", "def count_params(model):\n    n = 0\n    for weights in model.parameters():\n        n += weights.numel()\n    return n\n", "correct", None),
    ("count-params", _p("        total = total + 1\n"), "wrong", "tensors"),
    ("count-params", _p("        total = total + len(p)\n"), "wrong", "first-axis"),
]
