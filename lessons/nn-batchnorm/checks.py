# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-batchnorm: batch_norm, batch normalization of a batch of z's in train mode, and BNMLP, an
MLP with an nn.BatchNorm1d after each hidden layer. And the data of the picture: an MLP with 10 hidden layers and He
starting weights for the handwritten digits of QMNIST, trained for one epoch with and without batch normalization."""
import functools
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import paths
from everything_ai.check import ClassBox, FunctionBox, Mistake

DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md
DEPTH = 10      # hidden layers in the picture
LR = 0.3


@functools.lru_cache(maxsize=1)
def tensors():
    d = np.load(paths.ROOT / DATA)
    return (torch.tensor(d["train_images"], dtype=torch.float32) / 255, torch.tensor(d["train_labels"], dtype=torch.int64),
            torch.tensor(d["test_images"], dtype=torch.float32) / 255, torch.tensor(d["test_labels"], dtype=torch.int64))


class DeepNet(nn.Module):
    """DEPTH hidden layers of 128 neurons with He starting weights and ReLU, with or without an nn.BatchNorm1d
    between each hidden layer and its ReLU, and 10 logits."""

    def __init__(self, batchnorm: bool):
        super().__init__()
        self.hidden = nn.ModuleList()
        for i in range(DEPTH):
            layer = nn.Linear(784 if i == 0 else 128, 128)
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
            self.hidden.append(layer)
        self.norms = nn.ModuleList([nn.BatchNorm1d(128) for _ in range(DEPTH)]) if batchnorm else None
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.hidden):
            z = layer(h)
            if self.norms is not None:
                z = self.norms[i](z)
            h = torch.relu(z)
        return self.out(h)


@functools.lru_cache(maxsize=2)
def run(batchnorm: bool):
    """One epoch of SGD with learning rate LR on the 6,000 training digits: the loss of each mini-batch, and the
    accuracy on the 1,000 test digits afterwards, in eval mode."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)   # faster on these small layers, and the numbers no longer depend on the number of cores
    try:
        X, y, X_test, y_test = tensors()
        torch.manual_seed(0)
        model = DeepNet(batchnorm)
        loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
        opt = torch.optim.SGD(model.parameters(), lr=LR)
        loss_fn = nn.CrossEntropyLoss()
        torch.manual_seed(2)
        losses = []
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            accuracy = (model(X_test).argmax(dim=1) == y_test).float().mean().item()
        return losses, accuracy
    finally:
        torch.set_num_threads(threads)


def _smooth(values, k=10):
    """The mean of the last k values at each step (fewer at the start)."""
    return [float(np.mean(values[max(0, i + 1 - k):i + 1])) for i in range(len(values))]


def loss_picture(labels: dict) -> str:
    """The training loss of the deep net during one epoch, with and without batch normalization."""
    from everything_ai import chart
    plain, _ = run(False)
    normed, _ = run(True)
    steps = list(range(1, len(plain) + 1))
    p = chart.Panel((0, len(plain) + 1), (0, 2.6), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    p.line(steps, _smooth(plain), "loss", 2.2, dashed=True)
    p.line(steps, _smooth(normed), "loss", 2.2)
    p.text(len(plain) - 2, _smooth(plain)[-1] + 0.17, labels["plain"], "ink-2", "end")
    p.text(len(plain) - 2, _smooth(normed)[-1] + 0.2, labels["normed"], "ink", "end")
    return chart.figure([p], alt=labels["alt"], width=560, height=300)


# The lesson is done when these are correct. "eval-mode" is the choice question in the lesson.
REQUIRED = ["eval-mode", "batch-norm", "bn-mlp"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _round(x: float) -> float | str:
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return round(x, 4) + 0.0


def _rows(v):
    """A tensor, array or list as a list of rows of plain numbers, or None when it is none of those."""
    try:
        t = torch.as_tensor(v, dtype=torch.float32) if not isinstance(v, torch.Tensor) else v.detach().float()
    except (TypeError, ValueError, RuntimeError):
        return None
    return t.tolist()


def _show(v) -> str:
    rows = _rows(v)
    if rows is None:
        return repr(v)
    if isinstance(rows, list) and rows and isinstance(rows[0], list):
        return "[" + ", ".join("[" + ", ".join(str(_round(x)) for x in row) + "]" for row in rows) + "]"
    if isinstance(rows, list):
        return "[" + ", ".join(str(_round(x)) for x in rows) + "]"
    return str(_round(rows))


def _close(got, want) -> bool:
    g, w = _rows(got), _rows(want)
    if g is None or w is None:
        return False
    try:
        g, w = np.asarray(g, dtype=float), np.asarray(w, dtype=float)
    except (TypeError, ValueError):
        return False
    return g.shape == w.shape and bool(np.allclose(g, w, atol=1e-3, equal_nan=True))


class _T:
    """A tensor for a check call, shown the way the lesson writes one: torch.tensor([...]). The learner's function
    gets a plain tensor."""

    def __init__(self, rows):
        self.rows = rows

    def __repr__(self):
        return f"torch.tensor({self.rows!r})"

    def __deepcopy__(self, memo):
        return torch.tensor(self.rows, dtype=torch.float32)


# --- exercise 2: batch normalization in train mode ----------------------------------------------------------------------

_B_HEAD = "import torch\n\n\ndef batch_norm(Z, gamma, beta):\n"
_B_START = _B_HEAD + "    # standardize each column of Z over the batch (the rows), with eps = 0.00001; then times gamma, plus beta\n    ...\n"


def _b(mean: str = "Z.mean(dim=0)", var: str = "((Z - mean) ** 2).mean(dim=0)", out: str = "gamma * (Z - mean) / torch.sqrt(var + 1e-5) + beta") -> str:
    return _B_HEAD + f"    mean = {mean}\n    var = {var}\n    return {out}\n"


# --- exercise 3: an MLP with batch normalization -------------------------------------------------------------------------

_M_HEAD = "import torch\nfrom torch import nn\n\n\nclass BNMLP(nn.Module):\n"
_M_START = (_M_HEAD + "    def __init__(self, n_in, n_hidden, n_out):\n        super().__init__()\n"
            "        # h1, then a BatchNorm1d, h2, then a BatchNorm1d, and out\n        ...\n\n"
            "    def forward(self, x):\n        # each hidden layer: Linear, then BatchNorm1d, then ReLU\n        ...\n")


def _m(init_bn2: str = "self.bn2 = nn.BatchNorm1d(n_hidden)\n", first: str = "torch.relu(self.bn1(self.h1(x)))",
       second: str = "torch.relu(self.bn2(self.h2(h)))") -> str:
    return (_M_HEAD + "    def __init__(self, n_in, n_hidden, n_out):\n        super().__init__()\n"
            "        self.h1 = nn.Linear(n_in, n_hidden)\n        self.bn1 = nn.BatchNorm1d(n_hidden)\n"
            "        self.h2 = nn.Linear(n_hidden, n_hidden)\n" + (f"        {init_bn2}" if init_bn2 else "")
            + "        self.out = nn.Linear(n_hidden, n_out)\n\n"
            "    def forward(self, x):\n" + f"        h = {first}\n        h = {second}\n        return self.out(h)\n")


_M_SCENARIOS = [
    "torch.manual_seed(0)\nmodel = BNMLP(4, 6, 3)\nx = torch.randn(8, 4)\nmodel(x)[:2].tolist()",
    "torch.manual_seed(0)\nmodel = BNMLP(4, 6, 3)\nx = torch.randn(8, 4)\nmodel(x)\nmodel.eval()\nmodel(x)[:2].tolist()",
    "torch.manual_seed(1)\nmodel = BNMLP(5, 4, 2)\nx = torch.randn(6, 5)\nmodel(x)\nmodel(2 * x)\nmodel.eval()\nmodel(x[:3]).tolist()",
]

# A learner's own normalization, as in exercise 2, instead of nn.BatchNorm1d: no running averages.
_M_OWN = ("import torch\nfrom torch import nn\n\n\ndef batch_norm(Z, gamma, beta):\n    mean = Z.mean(dim=0)\n"
          "    var = ((Z - mean) ** 2).mean(dim=0)\n    return gamma * (Z - mean) / torch.sqrt(var + 1e-5) + beta\n\n\n"
          "class BNMLP(nn.Module):\n    def __init__(self, n_in, n_hidden, n_out):\n        super().__init__()\n"
          "        self.h1 = nn.Linear(n_in, n_hidden)\n        self.h2 = nn.Linear(n_hidden, n_hidden)\n"
          "        self.out = nn.Linear(n_hidden, n_out)\n        self.gamma = torch.ones(n_hidden)\n        self.beta = torch.zeros(n_hidden)\n\n"
          "    def forward(self, x):\n        h = torch.relu(batch_norm(self.h1(x), self.gamma, self.beta))\n"
          "        h = torch.relu(batch_norm(self.h2(h), self.gamma, self.beta))\n        return self.out(h)\n")


EXERCISES = {
    "batch-norm": FunctionBox(
        name="batch-norm",
        function="batch_norm",
        start=_B_START,
        solution=_b(),
        visible=[(_T([[1.0, 10.0], [2.0, 20.0], [6.0, 30.0]]), _T([1.0, 2.0]), _T([0.0, 1.0]))],
        hidden=[(_T([[3.0, -1.0, 5.0], [3.0, 1.0, 1.0], [3.0, 0.0, 0.0], [3.0, 4.0, 2.0]]), _T([2.0, 1.0, 0.5]), _T([1.0, 0.0, -1.0])),
                (_T([[-2.0], [2.0]]), _T([3.0]), _T([0.5])),
                (_T([[0.5, 4.0], [1.5, 2.0], [-1.0, 0.0], [3.0, 2.0], [1.0, 7.0]]), _T([0.5, 1.0]), _T([2.0, -2.0]))],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("math-sqrt", syntax=r"\bmath\.sqrt\(",
                    en="`math.sqrt` takes one number, and `var + 1e-5` is a tensor with one number per column. `torch.sqrt` takes the square root of every number in a tensor, or use `** 0.5`.",
                    nl="`math.sqrt` neemt één getal, en `var + 1e-5` is een tensor met één getal per kolom. `torch.sqrt` neemt de wortel van elk getal in een tensor, of gebruik `** 0.5`."),
            Mistake("unbiased", wrong=_b(var="Z.var(dim=0)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: in PyTorch, `Z.var(dim=0)` and `Z.std(dim=0)` divide by the number of rows minus 1, a correction for estimating the variance of all data from a sample. Batch normalization divides by the number of rows itself: `((Z - mean) ** 2).mean(dim=0)`, or `Z.var(dim=0, unbiased=False)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: in PyTorch delen `Z.var(dim=0)` en `Z.std(dim=0)` door het aantal rijen min 1, een correctie om de variantie van alle data te schatten uit een steekproef. Batch normalization deelt door het aantal rijen zelf: `((Z - mean) ** 2).mean(dim=0)`, of `Z.var(dim=0, unbiased=False)`."),
            Mistake("torch-std", wrong=_B_HEAD + "    return gamma * (Z - Z.mean(dim=0)) / Z.std(dim=0) + beta\n",
                    en="`{call}` gives `{got}`, and it should give `{want}`: in PyTorch, `Z.std(dim=0)` and `Z.var(dim=0)` divide by the number of rows minus 1, unlike `.std()` in NumPy. Batch normalization divides by the number of rows itself, and adds the small `1e-5` under the square root, so that a column with the same number in every row does not give `nan`: `torch.sqrt(var + 1e-5)`, with `var = ((Z - mean) ** 2).mean(dim=0)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: in PyTorch delen `Z.std(dim=0)` en `Z.var(dim=0)` door het aantal rijen min 1, anders dan `.std()` in NumPy. Batch normalization deelt door het aantal rijen zelf, en telt de kleine `1e-5` op onder de wortel, zodat een kolom met in elke rij hetzelfde getal geen `nan` geeft: `torch.sqrt(var + 1e-5)`, met `var = ((Z - mean) ** 2).mean(dim=0)`."),
            Mistake("rows", wrong=_b(mean="Z.mean(dim=1, keepdim=True)", var="((Z - mean) ** 2).mean(dim=1, keepdim=True)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you standardize each row, one example, over its features. Batch normalization standardizes each column, one feature, over the examples in the batch: `dim=0`. Per example is layer normalization, the next lesson.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je standaardiseert elke rij, één voorbeeld, over zijn features. Batch normalization standaardiseert elke kolom, één feature, over de voorbeelden in de batch: `dim=0`. Per voorbeeld is layer normalization, de volgende les."),
            Mistake("no-sqrt", wrong=_b(out="gamma * (Z - mean) / (var + 1e-5) + beta"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you divide by the variance. Standardizing divides by the standard deviation, its square root: `torch.sqrt(var + 1e-5)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je deelt door de variantie. Standaardiseren deelt door de standaardafwijking, de wortel daaruit: `torch.sqrt(var + 1e-5)`."),
            Mistake("no-scale", wrong=_b(out="(Z - mean) / torch.sqrt(var + 1e-5)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: each column is standardized, but not yet rescaled. Multiply by `gamma` and add `beta`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elke kolom is gestandaardiseerd, maar nog niet herschaald. Vermenigvuldig met `gamma` en tel `beta` erbij op."),
            Mistake("swapped", wrong=_b(out="beta * (Z - mean) / torch.sqrt(var + 1e-5) + gamma"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `gamma` and `beta` have swapped places. `gamma` multiplies, `beta` is added.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `gamma` en `beta` zijn van plaats gewisseld. `gamma` vermenigvuldigt, `beta` tel je op."),
            Mistake("no-eps", wrong=_b(out="gamma * (Z - mean) / torch.sqrt(var) + beta"), state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: a column with the same number in every row has a variance of 0, and 0 divided by 0 is `nan`. Add the small `1e-5` under the square root: `torch.sqrt(var + 1e-5)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: een kolom met in elke rij hetzelfde getal heeft een variantie van 0, en 0 gedeeld door 0 is `nan`. Tel de kleine `1e-5` op onder de wortel: `torch.sqrt(var + 1e-5)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: the mean and the variance of each column, the standardized `Z`, and then times `gamma`, plus `beta`.",
                      "nl": "Vervang de `...` door de body van de functie: het gemiddelde en de variantie van elke kolom, de gestandaardiseerde `Z`, en dan keer `gamma`, plus `beta`."},
            "arguments": {"en": "The check calls `{call}`, with three tensors, and the `def` line of your function has another number of names between its parentheses. Keep `(Z, gamma, beta)`.",
                          "nl": "De check roept `{call}` aan, met drie tensors, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(Z, gamma, beta)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Work out the first column by hand: its mean, its variance, dividing by the number of rows, and then each number standardized, times `gamma`, plus `beta`.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken de eerste kolom met de hand uit: zijn gemiddelde, zijn variantie, gedeeld door het aantal rijen, en dan elk getal gestandaardiseerd, keer `gamma`, plus `beta`."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any number of rows and columns, also for a column with the same number in every row?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elk aantal rijen en kolommen, ook voor een kolom met in elke rij hetzelfde getal?"},
            "correct": {"en": "Your function is right in all {n} cases: every column standardized over the batch, then rescaled with gamma and beta.",
                        "nl": "Je functie klopt in alle {n} gevallen: elke kolom gestandaardiseerd over de batch, en dan herschaald met gamma en beta."},
        },
    ),
    "bn-mlp": ClassBox(
        name="bn-mlp",
        start=_M_START,
        solution=_m(),
        scenarios=_M_SCENARIOS,
        compare=_close,
        digits=4,
        first=[_DOTS],
        mistakes=[
            Mistake("no-super", syntax=r"\A(?:(?![\s\S]*\bsuper\s*\()(?![\s\S]*Module\s*\.\s*__init__\s*\()|[\s\S]*?^[ \t]+self\.\w+[ \t]*=[^=][\s\S]*?\bsuper\s*\()",
                    en="Your `__init__` does not start with `super().__init__()`. `nn.Module` has to set itself up before your class can give it a layer: write `super().__init__()` as the first line of `__init__`, and only then make the layers.",
                    nl="Je `__init__` begint niet met `super().__init__()`. `nn.Module` moet zichzelf eerst opzetten voordat je class er een layer aan kan geven: schrijf `super().__init__()` als eerste regel van `__init__`, en maak pas daarna de layers."),
            Mistake("no-forward", syntax=r"\A(?![\s\S]*^[ \t]+def[ \t]+forward[ \t]*\()",
                    en="Your class has no method `forward`: PyTorch calls it when you write `model(x)`. Write `def forward(self, x):`, with exactly that name.",
                    nl="Je class heeft geen method `forward`: PyTorch roept die aan als je `model(x)` schrijft. Schrijf `def forward(self, x):`, met precies die naam."),
            Mistake("bn-n-in", syntax=r"BatchNorm1d\(\s*n_in\s*\)",
                    en="An `nn.BatchNorm1d` standardizes the z's that the `nn.Linear` before it gives: `n_hidden` of them, not `n_in`. Use `nn.BatchNorm1d(n_hidden)`.",
                    nl="Een `nn.BatchNorm1d` standaardiseert de z's die de `nn.Linear` ervoor geeft: `n_hidden` stuks, niet `n_in`. Gebruik `nn.BatchNorm1d(n_hidden)`."),
            Mistake("after-relu", wrong=_m(first="self.bn1(torch.relu(self.h1(x)))", second="self.bn2(torch.relu(self.h2(h)))"), state="almost",
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: your BatchNorm1d comes after the ReLU. That is used too, but this exercise follows the original paper: Linear, then BatchNorm1d, then ReLU, so that the ReLU gets standardized z's.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: je BatchNorm1d komt na de ReLU. Dat wordt ook gebruikt, maar deze oefening volgt de oorspronkelijke paper: Linear, dan BatchNorm1d, dan ReLU, zodat de ReLU gestandaardiseerde z's krijgt."),
            Mistake("shared", wrong=_m(init_bn2="", second="torch.relu(self.bn1(self.h2(h)))"), state="almost",
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: in train mode it is right, but in eval mode not. Both hidden layers use the same BatchNorm1d, so they share its running averages, and its gamma and beta. Give each hidden layer its own.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: in train mode klopt het, maar in eval mode niet. Beide hidden layers gebruiken dezelfde BatchNorm1d, dus ze delen zijn voortschrijdende gemiddelden, en zijn gamma en beta. Geef elke hidden layer zijn eigen."),
            Mistake("own-norm", wrong=_M_OWN, state="almost",
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: in train mode it is right, but in eval mode not. Your normalization uses the batch it gets, also in eval mode; `nn.BatchNorm1d` keeps running averages and uses those in eval mode, and its gamma and beta are learned. Use `nn.BatchNorm1d(n_hidden)`.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: in train mode klopt het, maar in eval mode niet. Je normalisatie gebruikt de batch die hij krijgt, ook in eval mode; `nn.BatchNorm1d` houdt voortschrijdende gemiddelden bij en gebruikt die in eval mode, en zijn gamma en beta worden geleerd. Gebruik `nn.BatchNorm1d(n_hidden)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: in `__init__` the layers `h1`, `bn1`, `h2`, `bn2` and `out`, and in `forward` each hidden layer as Linear, then BatchNorm1d, then ReLU, and `out` at the end.",
                      "nl": "Vervang de twee `...`: in `__init__` de layers `h1`, `bn1`, `h2`, `bn2` en `out`, en in `forward` elke hidden layer als Linear, dan BatchNorm1d, dan ReLU, en `out` aan het eind."},
            "wrong": {"en": "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Are the three `nn.Linear` layers made in the order `h1`, `h2`, `out`, so that they get the same random weights, and does each hidden layer go Linear, then BatchNorm1d, then ReLU?",
                      "nl": "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Maak je de drie `nn.Linear`-layers in de volgorde `h1`, `h2`, `out`, zodat ze dezelfde willekeurige weights krijgen, en gaat elke hidden layer Linear, dan BatchNorm1d, dan ReLU?"},
            "almost": {"en": "Right in train mode. But with these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. In eval mode, does each hidden layer use its own `nn.BatchNorm1d`, with its own running averages?",
                       "nl": "Goed in train mode. Maar met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Gebruikt in eval mode elke hidden layer zijn eigen `nn.BatchNorm1d`, met zijn eigen voortschrijdende gemiddelden?"},
            "correct": {"en": "Your MLP standardizes the z's of each hidden layer over the batch in train mode, and uses its running averages in eval mode.",
                        "nl": "Je MLP standaardiseert de z's van elke hidden layer over de batch in train mode, en gebruikt zijn voortschrijdende gemiddelden in eval mode."},
        },
    ),
}

CASES = [
    ("batch-norm", _B_START, "empty", None),
    ("batch-norm", "", "empty", None),
    ("batch-norm", _b(), "correct", None),
    ("batch-norm", _b(var="Z.var(dim=0, unbiased=False)"), "correct", None),
    ("batch-norm", _b(var="Z.std(dim=0, unbiased=False) ** 2"), "correct", None),
    ("batch-norm", _B_HEAD + "    Z_hat = (Z - Z.mean(dim=0)) / (((Z - Z.mean(dim=0)) ** 2).mean(dim=0) + 1e-5) ** 0.5\n    return gamma * Z_hat + beta\n", "correct", None),
    ("batch-norm", _B_HEAD + "    bn = torch.nn.BatchNorm1d(Z.shape[1])\n    with torch.no_grad():\n        bn.weight.copy_(gamma)\n        bn.bias.copy_(beta)\n    return bn(Z).detach()\n", "correct", None),
    ("batch-norm", _b(var="Z.var(dim=0)"), "wrong", "unbiased"),
    ("batch-norm", "import math\n\n\n" + _b(out="gamma * (Z - mean) / math.sqrt(var + 1e-5) + beta"), "error", "math-sqrt"),
    ("batch-norm", _b(var="Z.std(dim=0) ** 2"), "wrong", "unbiased"),
    ("batch-norm", _B_HEAD + "    return gamma * (Z - Z.mean(dim=0)) / Z.std(dim=0) + beta\n", "wrong", "torch-std"),
    ("batch-norm", _b(mean="Z.mean(dim=1, keepdim=True)", var="((Z - mean) ** 2).mean(dim=1, keepdim=True)"), "wrong", "rows"),
    ("batch-norm", _b(out="gamma * (Z - mean) / (var + 1e-5) + beta"), "wrong", "no-sqrt"),
    ("batch-norm", _b(out="(Z - mean) / torch.sqrt(var + 1e-5)"), "wrong", "no-scale"),
    ("batch-norm", _b(out="beta * (Z - mean) / torch.sqrt(var + 1e-5) + gamma"), "wrong", "swapped"),
    ("batch-norm", _b(out="gamma * (Z - mean) / torch.sqrt(var) + beta"), "almost", "no-eps"),
    ("batch-norm", _b(out="gamma * (Z - mean) / torch.sqrt(var + 1e-5)"), "wrong", None),
    ("batch-norm", "def batch_norm(Z):\n    return Z\n", "error", "arguments"),
    ("bn-mlp", _M_START, "empty", None),
    ("bn-mlp", "", "empty", None),
    ("bn-mlp", _m(), "correct", None),
    ("bn-mlp", _m(first="torch.relu(self.bn1(self.h1(x)))", second="self.bn2(self.h2(h)).relu()"), "correct", None),
    ("bn-mlp", _m(first="self.bn1(torch.relu(self.h1(x)))", second="self.bn2(torch.relu(self.h2(h)))"), "almost", "after-relu"),
    ("bn-mlp", _m(init_bn2="", second="torch.relu(self.bn1(self.h2(h)))"), "almost", "shared"),
    ("bn-mlp", _M_OWN, "almost", "own-norm"),
    ("bn-mlp", _m().replace("self.bn1 = nn.BatchNorm1d(n_hidden)", "self.bn1 = nn.BatchNorm1d(n_in)"), "error", "bn-n-in"),
    ("bn-mlp", _m(second="torch.relu(self.h2(h))"), "wrong", None),
    ("bn-mlp", _M_HEAD + "    def __init__(self, n_in, n_hidden, n_out):\n        self.h1 = nn.Linear(n_in, n_hidden)\n", "error", "no-super"),
    ("bn-mlp", _m().replace("        super().__init__()\n", ""), "error", "no-super"),
    ("bn-mlp", _m().replace("def forward", "def foward"), "error", "no-forward"),
    ("bn-mlp", _m().replace("        super().__init__()\n", "").replace("        self.out = nn.Linear(n_hidden, n_out)\n", "        self.out = nn.Linear(n_hidden, n_out)\n        super().__init__()\n"), "error", "no-super"),
]
