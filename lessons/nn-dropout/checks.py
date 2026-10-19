# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-dropout: Dropout, a layer of one's own that sets numbers to 0 in train mode and scales the
rest, and DropoutMLP, the network of the picture with nn.Dropout after each hidden layer. And the picture: that MLP
trained on 500 handwritten digits of QMNIST for 100 epochs, without and with dropout, and its loss on the 1,000 test
digits after each epoch."""

import functools
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import paths
from everything_ai.check import ClassBox, Mistake

DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md
N_TRAIN, HIDDEN, LR, EPOCHS = 500, 256, 0.1, 100


class DropoutMLP(nn.Module):
    """The network of the picture: two hidden layers of `hidden` neurons, each followed by dropout."""

    def __init__(self, hidden, p):
        super().__init__()
        self.hidden1 = nn.Linear(784, hidden)
        self.hidden2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 10)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = self.drop(torch.relu(self.hidden1(x)))
        h = self.drop(torch.relu(self.hidden2(h)))
        return self.out(h)


@functools.lru_cache(maxsize=2)
def run(p: float):
    """100 epochs of SGD on the first 500 training digits. Gives the loss on the 1,000 test digits after each epoch,
    in eval mode, and the number of test digits read right at the end."""
    d = np.load(paths.ROOT / DATA)
    X = torch.tensor(d["train_images"][:N_TRAIN], dtype=torch.float32) / 255
    y = torch.tensor(d["train_labels"][:N_TRAIN], dtype=torch.int64)
    X_test = torch.tensor(d["test_images"], dtype=torch.float32) / 255
    y_test = torch.tensor(d["test_labels"], dtype=torch.int64)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)   # faster on these small layers, and the numbers no longer depend on the number of cores
    try:
        torch.manual_seed(0)
        model = DropoutMLP(HIDDEN, p)
        opt = torch.optim.SGD(model.parameters(), lr=LR)
        loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
        loss_fn = nn.CrossEntropyLoss()
        torch.manual_seed(1)
        test_losses = []
        for epoch in range(EPOCHS):
            model.train()
            for xb, yb in loader:
                loss = loss_fn(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                test_losses.append(loss_fn(model(X_test), y_test).item())
        with torch.no_grad():
            right = int((model(X_test).argmax(dim=1) == y_test).sum().item())
    finally:
        torch.set_num_threads(threads)
    return test_losses, right


def dropout_picture(labels: dict) -> str:
    """The test loss after each epoch, without dropout (dashed) and with dropout 0.5 (solid)."""
    from everything_ai import chart
    plain, dropped = run(0.0)[0], run(0.5)[0]
    epochs = list(range(1, EPOCHS + 1))
    panel = chart.Panel((0, EPOCHS + 1), (0.4, 1.2), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    panel.line(epochs, plain, "loss", 1.6, dashed=True)
    panel.line(epochs, dropped, "loss", 2.4)
    panel.text(EPOCHS - 1, plain[-1] + 0.05, labels["without"], "loss", "end")
    panel.text(EPOCHS - 1, dropped[-1] - 0.1, labels["with"], "loss", "end")
    return chart.figure([panel], alt=labels["alt"], width=330, height=250)


# The lesson is done when these are correct. "why-it-helps" is the choice question in the lesson.
REQUIRED = ["why-it-helps", "own-dropout", "dropout-mlp"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_NO_SUPER = Mistake("no-super", syntax=r"\A(?:(?![\s\S]*\bsuper\s*\()(?![\s\S]*Module\s*\.\s*__init__\s*\()|[\s\S]*?^[ \t]+self\.\w+[ \t]*=[^=][\s\S]*?\bsuper\s*\()",
                    en="Your `__init__` does not start with `super().__init__()`. `nn.Module` has to set itself up before your class can keep anything: write `super().__init__()` as the first line of `__init__`.",
                    nl="Je `__init__` begint niet met `super().__init__()`. `nn.Module` moet zichzelf eerst opzetten voordat je class iets kan bewaren: schrijf `super().__init__()` als eerste regel van `__init__`.")
_NO_FORWARD = Mistake("no-forward", syntax=r"\A(?![\s\S]*^[ \t]+def[ \t]+forward[ \t]*\()",
                      en="Your class has no method `forward`: PyTorch calls it when you write `layer(x)`. Write `def forward(self, x):`, with exactly that name.",
                      nl="Je class heeft geen method `forward`: PyTorch roept die aan als je `layer(x)` schrijft. Schrijf `def forward(self, x):`, met precies die naam.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _close_list(got, want, tol: float = 0.02) -> bool:
    """Lists of numbers that may differ by `tol`: shares and means of random masks, which differ a little with the way
    the mask is drawn. With 100,000 ones and p = 0.5, the mean of another draw differs from the course's by more than
    0.01 in about 1 of 170 draws; by more than 0.02 practically never. The known mistakes differ by 0.2 or more."""
    try:
        g, w = [float(x) for x in got], [float(x) for x in want]
    except (TypeError, ValueError):
        return False
    return len(g) == len(w) and all(abs(a - b) <= tol for a, b in zip(g, w))


def _close_outputs(got, want) -> bool:
    try:
        return len(got) == len(want) and all(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=1e-4, atol=1e-6)
                                             for a, b in zip(got, want))
    except (TypeError, ValueError):
        return False


# --- exercise 2: a dropout layer of your own ---------------------------------------------------------------------------------

_D_HEAD = "import torch\nfrom torch import nn\n\n\nclass Dropout(nn.Module):\n"
_D_START = (_D_HEAD + "    def __init__(self, p):\n        super().__init__()\n        # keep p\n        ...\n\n"
            "    def forward(self, x):\n        # in train mode: each number to 0 with chance p, the rest divided by 1 - p; in eval mode: x as it is\n"
            "        ...\n")


def _d(keep: str = "torch.rand_like(x) >= self.p", out: str = "x * keep / (1 - self.p)", test: str = "        if not self.training:\n            return x\n") -> str:
    return (_D_HEAD + "    def __init__(self, p):\n        super().__init__()\n        self.p = p\n\n    def forward(self, x):\n"
            f"{test}        keep = {keep}\n        return {out}\n")


# the share of zeros and the mean in train mode, the largest number, and the share of zeros and the mean in eval mode
_STATS = ("torch.manual_seed(0)\nlayer = Dropout({p})\nx = torch.ones(100000)\nlayer.train()\ny = layer(x)\n"
          "train = [(y == 0).float().mean().item(), y.mean().item(), y.max().item()]\nlayer.eval()\ny = layer(x)\n"
          "train + [(y == 0).float().mean().item(), y.mean().item()]")
_D_SCENARIOS = [
    _STATS.format(p=0.2),
    _STATS.format(p=0.5),
    "torch.manual_seed(1)\nlayer = Dropout(0.0)\nx = torch.tensor([[1.0, -2.0], [3.0, 0.5]])\nlayer(x).tolist()",
    # two calls in train mode: a new mask each time, so about half of the numbers differ
    "torch.manual_seed(2)\nlayer = Dropout(0.5)\nx = torch.ones(100000)\na, b = layer(x), layer(x)\n[(a == b).float().mean().item()]",
]

# --- exercise 3: the network of the picture -----------------------------------------------------------------------------------

_M_HEAD = "import torch\nfrom torch import nn\n\n\nclass DropoutMLP(nn.Module):\n"
_M_START = (_M_HEAD + "    def __init__(self, hidden, p):\n        super().__init__()\n"
            "        self.hidden1 = nn.Linear(784, hidden)\n        self.hidden2 = nn.Linear(hidden, hidden)\n        self.out = nn.Linear(hidden, 10)\n"
            "        # and a dropout layer with chance p\n        ...\n\n"
            "    def forward(self, x):\n        # each hidden layer: ReLU, then dropout; the output layer without\n        ...\n")
_M_INIT = ("    def __init__(self, hidden, p):\n        super().__init__()\n"
           "        self.hidden1 = nn.Linear(784, hidden)\n        self.hidden2 = nn.Linear(hidden, hidden)\n        self.out = nn.Linear(hidden, 10)\n"
           "        self.drop = nn.Dropout(p)\n\n")


def _m(body: str = "        h = self.drop(torch.relu(self.hidden1(x)))\n        h = self.drop(torch.relu(self.hidden2(h)))\n        return self.out(h)\n") -> str:
    return _M_HEAD + _M_INIT + "    def forward(self, x):\n" + body


_M_SETUP = "torch.manual_seed(0)\nmodel = DropoutMLP(8, 0.5)\nx = torch.rand(1, 784)\n"
_M_SCENARIOS = [
    # eval mode: the same as without dropout
    _M_SETUP + "model.eval()\nmodel(x).tolist()",
    # train mode, with a fixed seed: where the numbers are dropped
    _M_SETUP + "model.train()\ntorch.manual_seed(5)\nmodel(x).tolist()",
    # train mode twice, eval mode twice: does the same input give the same output?
    _M_SETUP + "model.train()\na, b = model(x), model(x)\nmodel.eval()\nc, d = model(x), model(x)\n"
               "[a.tolist() == b.tolist(), c.tolist() == d.tolist()]",
    # another p: does the network use the p it gets?
    "torch.manual_seed(0)\nmodel = DropoutMLP(8, 0.2)\nx = torch.rand(1, 784)\nmodel.train()\ntorch.manual_seed(5)\nmodel(x).tolist()",
]

EXERCISES = {
    "own-dropout": ClassBox(
        name="own-dropout",
        start=_D_START,
        solution=_d(),
        scenarios=_D_SCENARIOS,
        compare=lambda got, want: (_close_list(got, want) if isinstance(want, list) and want and not isinstance(want[0], list)
                                   else _close_outputs(got, want)),
        digits=2,
        first=[_DOTS],
        mistakes=[
            _NO_SUPER,
            _NO_FORWARD,
            # a forward that gives back no tensor in one mode, often no `return x` after `if self.training:`: then `y == 0` is
            # one bool, and the scenario's `.float()` fails on it
            Mistake("no-tensor", error=lambda e: isinstance(e, AttributeError) and "'bool' object has no attribute 'float'" in str(e),
                    en="Your `forward` gives back no tensor in one of the two modes, so the check cannot count its zeros. Does each mode end in a `return`: in train mode the numbers dropped and scaled, in eval mode `x` as it is?",
                    nl="Je `forward` geeft in een van de twee modes geen tensor terug, dus de check kan zijn nullen niet tellen. Eindigt elke mode met een `return`: in train mode de getallen weggelaten en geschaald, in eval mode `x` zoals hij is?"),
            Mistake("no-scale", wrong=_d(out="x * keep"),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: in train mode, the mean of your numbers is smaller than in eval mode. Divide the numbers you keep by `1 - self.p`, so that on average they stay as large as `x`.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: in train mode is het gemiddelde van je getallen kleiner dan in eval mode. Deel de getallen die je houdt door `1 - self.p`, zodat ze gemiddeld net zo groot blijven als `x`."),
            Mistake("keep-p", wrong=_d(keep="torch.rand_like(x) < self.p"),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: you keep each number with chance `p`, so with `p = 0.2` you drop 80% instead of 20%. `p` is the chance to drop a number: keep it where `torch.rand_like(x) >= self.p`.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: je houdt elk getal met kans `p`, dus met `p = 0.2` laat je 80% vallen in plaats van 20%. `p` is de kans om een getal te laten vallen: houd het waar `torch.rand_like(x) >= self.p`."),
            Mistake("always", wrong=_d(test=""),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: in eval mode, your layer still sets numbers to 0. When `self.training` is `False`, give back `x` as it is.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: in eval mode zet je layer nog steeds getallen op 0. Geef `x` terug zoals hij is als `self.training` `False` is."),
            Mistake("scale-in-eval", wrong=_d(test="        if not self.training:\n            return x / (1 - self.p)\n"),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: in eval mode, your numbers are larger than `x`. The dividing by `1 - p` belongs to train mode only; in eval mode, give back `x` as it is.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: in eval mode zijn je getallen groter dan `x`. Het delen door `1 - p` hoort alleen bij train mode; geef in eval mode `x` terug zoals hij is."),
            Mistake("times-keep-share", wrong=_d(out="x * keep * (1 - self.p)"),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: you multiply the numbers you keep by `1 - p`, which makes them smaller still. Divide them by `1 - p`, so that the ones that stay make up for the ones that drop.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: je vermenigvuldigt de getallen die je houdt met `1 - p`, en dan worden ze nog kleiner. Deel ze door `1 - p`, zodat de getallen die blijven goedmaken wat er wegvalt."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: in `__init__`, keep `p`; in `forward`, the numbers dropped and scaled in train mode, and `x` as it is in eval mode.",
                      "nl": "Vervang de twee `...`: bewaar in `__init__` `p`; in `forward` de getallen weggelaten en geschaald in train mode, en `x` zoals hij is in eval mode."},
            "wrong": {"en": "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: in train mode the share of zeros, the mean and the largest number, then in eval mode the share of zeros and the mean. Does each number drop with chance `p`, and are the others divided by `1 - p`, only in train mode?",
                      "nl": "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: in train mode het deel nullen, het gemiddelde en het grootste getal, daarna in eval mode het deel nullen en het gemiddelde. Valt elk getal weg met kans `p`, en worden de andere gedeeld door `1 - p`, alleen in train mode?"},
            "almost": {"en": "Right with `p = 0.2`. But with these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does your layer use `self.p` instead of a fixed number, and the numbers in `x` themselves, so that it works for every `p` and every `x`, with a new mask at every call?",
                       "nl": "Goed met `p = 0.2`. Maar met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Gebruikt je layer `self.p` in plaats van een vast getal, en de getallen in `x` zelf, zodat hij werkt voor elke `p` en elke `x`, met een nieuw mask bij elke aanroep?"},
            "correct": {"en": "Your layer drops each number with chance p in train mode, scales the rest so the mean stays the same, and leaves everything alone in eval mode, as nn.Dropout does.",
                        "nl": "Je layer laat in train mode elk getal vallen met kans p, schaalt de rest zodat het gemiddelde gelijk blijft, en laat in eval mode alles met rust, zoals nn.Dropout doet."},
        },
    ),
    "dropout-mlp": ClassBox(
        name="dropout-mlp",
        start=_M_START,
        solution=_m(),
        scenarios=_M_SCENARIOS,
        compare=lambda got, want: (got == want if want and isinstance(want[0], bool) else _close_outputs(got, want)),
        digits=4,
        first=[_DOTS],
        mistakes=[
            _NO_SUPER,
            _NO_FORWARD,
            Mistake("on-input", wrong=_m("        h = torch.relu(self.hidden1(self.drop(x)))\n        h = self.drop(torch.relu(self.hidden2(h)))\n        return self.out(h)\n"),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: you drop pixels of the input instead of neurons of the first hidden layer. Put dropout after each hidden layer's ReLU: `self.drop(torch.relu(self.hidden1(x)))`.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: je laat pixels van de input vallen in plaats van neuronen van de eerste hidden layer. Zet dropout na de ReLU van elke hidden layer: `self.drop(torch.relu(self.hidden1(x)))`."),
            Mistake("on-output", wrong=_m("        h = self.drop(torch.relu(self.hidden1(x)))\n        h = torch.relu(self.hidden2(h))\n        return self.drop(self.out(h))\n"),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: you drop logits of the output instead of neurons of the second hidden layer. The output layer stays whole; put dropout after the ReLU of `hidden2`.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: je laat logits van de output vallen in plaats van neuronen van de tweede hidden layer. De output layer blijft heel; zet dropout na de ReLU van `hidden2`."),
            Mistake("one-layer", wrong=_m("        h = self.drop(torch.relu(self.hidden1(x)))\n        h = torch.relu(self.hidden2(h))\n        return self.out(h)\n"),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: only your first hidden layer has dropout. Put it after the ReLU of `hidden2` as well.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: alleen je eerste hidden layer heeft dropout. Zet het ook na de ReLU van `hidden2`."),
            Mistake("functional", wrong=_m("        h = nn.functional.dropout(torch.relu(self.hidden1(x)), 0.5, training=True)\n"
                                           "        h = nn.functional.dropout(torch.relu(self.hidden2(h)), 0.5, training=True)\n        return self.out(h)\n"),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: in eval mode, your network still drops numbers, so the same input gives another output each time. Use the layer `self.drop`, made in `__init__` with `nn.Dropout(p)`: it reads train and eval mode itself.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: in eval mode laat je netwerk nog steeds getallen vallen, dus dezelfde input geeft elke keer een andere output. Gebruik de layer `self.drop`, gemaakt in `__init__` met `nn.Dropout(p)`: die leest zelf of het train of eval mode is."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: in `__init__`, a dropout layer `nn.Dropout(p)`; in `forward`, each hidden layer with ReLU and then dropout, and the output layer.",
                      "nl": "Vervang de twee `...`: in `__init__` een dropout layer `nn.Dropout(p)`; in `forward` elke hidden layer met ReLU en dan dropout, en de output layer."},
            "wrong": {"en": "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does `forward` go through `hidden1`, ReLU, dropout, `hidden2`, ReLU, dropout and `out`, in that order, with the layer `nn.Dropout(p)` from `__init__`?",
                      "nl": "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Gaat `forward` door `hidden1`, ReLU, dropout, `hidden2`, ReLU, dropout en `out`, in die volgorde, met de layer `nn.Dropout(p)` uit `__init__`?"},
            "almost": {"en": "Right in eval mode. But with these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. In train mode, does your network drop neurons after the ReLU of both hidden layers, and nowhere else, with the layer `nn.Dropout(p)` made from the `p` it gets?",
                       "nl": "Goed in eval mode. Maar met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Laat je netwerk in train mode neuronen vallen na de ReLU van beide hidden layers, en nergens anders, met de layer `nn.Dropout(p)` gemaakt van de `p` die het krijgt?"},
            "correct": {"en": "Your network drops neurons of both hidden layers in train mode, and uses all of them in eval mode.",
                        "nl": "Je netwerk laat in train mode neuronen van beide hidden layers vallen, en gebruikt ze in eval mode allemaal."},
        },
    ),
}

CASES = [
    ("own-dropout", _D_START, "empty", None),
    ("own-dropout", _d(), "correct", None),
    ("own-dropout", _d(keep="torch.rand(x.shape) > self.p"), "correct", None),
    ("own-dropout", _d(keep="(torch.rand_like(x) < 1 - self.p).float()"), "correct", None),
    ("own-dropout", _d(keep="torch.bernoulli(torch.full_like(x, 1 - self.p))", out="x * keep / (1 - self.p)"), "correct", None),
    ("own-dropout", _d(out="x * keep"), "wrong", "no-scale"),
    ("own-dropout", _d(keep="torch.rand_like(x) < self.p"), "wrong", "keep-p"),
    ("own-dropout", _d(test=""), "wrong", "always"),
    ("own-dropout", _d(test="        if not self.training:\n            return x / (1 - self.p)\n"), "wrong", "scale-in-eval"),
    ("own-dropout", _d(out="x * keep * (1 - self.p)"), "wrong", "times-keep-share"),
    ("own-dropout", _d().replace("        super().__init__()\n", ""), "error", "no-super"),
    ("own-dropout", _d(test="", keep="torch.rand_like(x) >= self.p", out="x * keep / (1 - self.p) if self.training else None"), "error", "no-tensor"),
    ("own-dropout", _d(keep="torch.rand_like(x) >= 0.2", out="x * keep / 0.8"), "almost", None),
    ("own-dropout", _d(out="keep / (1 - self.p)"), "almost", None),
    ("dropout-mlp", _M_START, "empty", None),
    ("dropout-mlp", _m(), "correct", None),
    ("dropout-mlp", _m("        h = torch.relu(self.drop(self.hidden1(x)))\n        h = torch.relu(self.drop(self.hidden2(h)))\n        return self.out(h)\n"), "correct", None),
    ("dropout-mlp", _m("        h = torch.relu(self.hidden1(self.drop(x)))\n        h = self.drop(torch.relu(self.hidden2(h)))\n        return self.out(h)\n"), "wrong", "on-input"),
    ("dropout-mlp", _m("        h = self.drop(torch.relu(self.hidden1(x)))\n        h = torch.relu(self.hidden2(h))\n        return self.drop(self.out(h))\n"), "wrong", "on-output"),
    ("dropout-mlp", _m("        h = self.drop(torch.relu(self.hidden1(x)))\n        h = torch.relu(self.hidden2(h))\n        return self.out(h)\n"), "wrong", "one-layer"),
    ("dropout-mlp", _m("        h = nn.functional.dropout(torch.relu(self.hidden1(x)), 0.5, training=True)\n"
                       "        h = nn.functional.dropout(torch.relu(self.hidden2(h)), 0.5, training=True)\n        return self.out(h)\n"), "wrong", "functional"),
    ("dropout-mlp", _m("        h = torch.relu(self.hidden1(x))\n        h = torch.relu(self.hidden2(h))\n        return self.out(h)\n"), "almost", None),
    ("own-dropout", _D_HEAD + "    def __init__(self, p):\n        super().__init__()\n        self.p = p\n        self.keep = None\n\n"
                    "    def forward(self, x):\n        if not self.training:\n            return x\n        if self.keep is None:\n"
                    "            self.keep = torch.rand_like(x) >= self.p\n        return x * self.keep / (1 - self.p)\n", "almost", None),
    ("dropout-mlp", _M_HEAD + _M_INIT.replace("nn.Dropout(p)", "nn.Dropout(0.5)") + "    def forward(self, x):\n"
                    "        h = self.drop(torch.relu(self.hidden1(x)))\n        h = self.drop(torch.relu(self.hidden2(h)))\n        return self.out(h)\n", "almost", None),
]
