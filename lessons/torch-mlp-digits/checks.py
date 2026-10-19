# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson torch-mlp-digits: DigitsMLP, an MLP for the handwritten digits as a PyTorch module,
saved as nn/torch_mlp.py, and fit, the training loop with nn.CrossEntropyLoss, saved as nn/torch_train.py. And the
demo: it trains the learner's own MLP with their own fit on 6,000 digits of QMNIST, measures it on 1,000 others, and
draws the loss of each epoch."""
import functools
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import paths, projects
from everything_ai.check import ClassBox, Mistake, Timeout, run_code, with_time_limit

DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md
HIDDEN, LR, EPOCHS, BATCH = 64, 0.5, 10, 64   # the demo: 64 hidden neurons, a learning rate of 0.5, 10 epochs of 64
torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)   # the first optimizer in a process takes a second: not in a check
LINEAR = 890   # test digits that the linear classifier of ml-digits-linear reads right (its demo, 100 steps, lr 1)


@functools.lru_cache(maxsize=1)
def tensors():
    """The training images and labels and the test images and labels as tensors; every pixel divided by 255."""
    d = np.load(paths.ROOT / DATA)
    return (torch.tensor(d["train_images"], dtype=torch.float32) / 255, torch.tensor(d["train_labels"], dtype=torch.int64),
            torch.tensor(d["test_images"], dtype=torch.float32) / 255, torch.tensor(d["test_labels"], dtype=torch.int64))


class DigitsMLP(nn.Module):
    """The course's own MLP: the solution of exercise 2."""
    def __init__(self, hidden):
        super().__init__()
        self.hidden = nn.Linear(784, hidden)
        self.out = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.out(torch.relu(self.hidden(x)))


def _fit(model, loader, epochs, lr):
    """The course's own fit: the solution of exercise 3."""
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    losses = []
    for _ in range(epochs):
        total = 0
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total = total + loss.item()
        losses.append(total / len(loader))
    return losses


def accuracy(model) -> float:
    """The share of the 1,000 test digits the model reads right, in eval mode and without a graph."""
    _, _, Xt, yt = tensors()
    model.eval()
    with torch.no_grad():
        pred = model(Xt).argmax(dim=1)
    model.train()
    return (pred == yt).float().mean().item()


def baseline() -> tuple[int, float]:
    """The most common digit of the training set, and the share of the test digits that always saying it gets right."""
    _, y, _, yt = tensors()
    digit = int(torch.bincount(y).argmax())
    return digit, (yt == digit).float().mean().item()


def train_demo(make_model, fit) -> tuple:
    """A model from `make_model` trained with `fit` as in the demo: the same seeds, so the same run every time.
    Gives back the model and the mean loss of each epoch."""
    X, y, _, _ = tensors()
    torch.manual_seed(0)
    model = make_model(HIDDEN)
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH, shuffle=True)
    torch.manual_seed(2)
    return model, fit(model, loader, EPOCHS, LR)


# --- the demo's picture ------------------------------------------------------------------------------------------

def losses_picture(losses: list[float], labels: dict[str, str]) -> str:
    """The mean loss of each epoch of the learner's own training, as a line with a dot per epoch."""
    from everything_ai import chart
    epochs = list(range(1, len(losses) + 1))
    p = chart.Panel((0.5, len(losses) + 0.5), (0, max(1.0, max(losses) * 1.1)), title=labels["title"],
                    xlabel=labels["epoch"], ylabel=labels["loss"])
    p.line(epochs, losses, "loss", 2.5)
    p.points(epochs, losses, "loss", 4)
    return chart.figure([p], alt=labels["alt"], width=520, height=300)


# --- the demo: the learner's own DigitsMLP, trained with their own fit ----------------------------------------------

# the learner's two modules, then PyTorch's nn: the project is called nn too, and an import finds it by its full name
_MINE = "from nn.torch_mlp import DigitsMLP\nfrom nn.torch_train import fit\nimport torch\nfrom torch import nn\n"


def _saved(module: str) -> bool:
    return (projects.folder() / "nn" / f"{module}.py").exists()


def trained():
    """For the demo: None when torch_mlp.py or torch_train.py is not in the project yet; "mlp" or "fit" when that saved
    module no longer gives what the course's gives on a small case, or stops with an error; "slow" when the training
    takes more than a minute; else (accuracy, losses) of the learner's MLP after the learner's fit."""
    if not (_saved("torch_mlp") and _saved("torch_train")):
        return None
    run = run_code(_MINE + _M_SCENARIOS[1])
    if run.timeout:
        return "slow"
    if run.error is not None or not _same_logits(run.value, run_code(_M + "\n\n" + _M_SCENARIOS[1]).value):
        return "mlp"
    run = run_code(_MINE + _F_SCENARIOS[0])
    if run.timeout:
        return "slow"
    if run.error is not None or not _close_list(run.value, run_code(_F + "\n\n" + _F_SCENARIOS[0]).value):
        return "fit"
    make, fit = run.names["DigitsMLP"], run.names["fit"]
    try:
        model, losses = with_time_limit(lambda: train_demo(make, fit), 60)
    except Timeout:
        return "slow"
    except Exception:
        return "fit"
    return accuracy(model), [float(v) for v in losses]


def _same_logits(got, want) -> bool:
    try:
        return torch.allclose(torch.as_tensor(got, dtype=torch.float32), torch.as_tensor(want, dtype=torch.float32), rtol=1e-4, atol=1e-5)
    except (TypeError, ValueError, RuntimeError):
        return False


def _close_list(got, want) -> bool:
    try:
        return len(got) == len(want) and all(abs(float(a) - float(b)) <= 1e-4 * max(1.0, abs(float(b))) for a, b in zip(got, want))
    except (TypeError, ValueError):
        return False


# The lesson is done when these are correct. "loss-input" is the choice question in the lesson.
REQUIRED = ["loss-input", "digits-mlp", "fit-torch"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# --- exercise 2: DigitsMLP -------------------------------------------------------------------------------------------

_M_HEAD = ("import torch\nfrom torch import nn\n\n\nclass DigitsMLP(nn.Module):\n    def __init__(self, hidden):\n"
           "        super().__init__()\n        self.hidden = nn.Linear(784, hidden)\n")
_M_OUT = "        self.out = nn.Linear(hidden, 10)\n"
_M_FORWARD = "        return self.out(torch.relu(self.hidden(x)))\n"
_M_START = _M_HEAD + "        self.out = ...\n\n    def forward(self, x):\n        ...\n"


def _m(out: str = _M_OUT, forward: str = _M_FORWARD) -> str:
    return _M_HEAD + out + "\n    def forward(self, x):\n" + forward


_M = _m()
_DIGITS = ("import numpy as np\ndata = np.load(\"data/qmnist-small.npz\")\n"
           "x = torch.tensor(data[\"test_images\"][:3], dtype=torch.float32) / 255\n")
# the logits of the first image, for 64 and for 16 hidden neurons; without a graph, so they show as plain numbers
_M_SCENARIOS = [
    _DIGITS + f"torch.manual_seed(0)\nmodel = DigitsMLP({h})\nwith torch.no_grad():\n    logits = model(x)\nlogits[0]" for h in (64, 16)
]


def _sums_to_one(got) -> bool:
    try:
        t = torch.as_tensor(got, dtype=torch.float32)
        return t.numel() == 10 and bool((t >= 0).all()) and abs(t.sum().item() - 1) < 1e-4
    except (TypeError, ValueError, RuntimeError):
        return False


# --- exercise 3: fit ---------------------------------------------------------------------------------------------------

_F_HEAD = ("import torch\nfrom torch import nn\n\n\ndef fit(model, loader, epochs, lr):\n    loss_fn = nn.CrossEntropyLoss()\n"
           "    opt = torch.optim.SGD(model.parameters(), lr=lr)\n    model.train()\n    losses = []\n"
           "    for epoch in range(epochs):\n        total = 0\n        for xb, yb in loader:\n")
_F_BODY = "            loss = loss_fn(model(xb), yb)\n            opt.zero_grad()\n            loss.backward()\n            opt.step()\n"
_F_TOTAL = "            total = total + loss.item()\n"
_F_END = "        losses.append(total / len(loader))\n    return losses"
_F_START = _F_HEAD + "            ...\n" + _F_END + "\n"


def _f(body: str = _F_BODY, total: str = _F_TOTAL) -> str:
    return _F_HEAD + body + total + _F_END


_F = _f()
_SMALL = ("from torch.utils.data import DataLoader, TensorDataset\n"
          "X = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])\ny = torch.tensor([1, 0, 2, 1])\n"
          "torch.manual_seed(0)\nmodel = nn.Linear(2, 3)\n")
_F_SCENARIOS = [
    _SMALL + "loader = DataLoader(TensorDataset(X, y), batch_size=2)\nfit(model, loader, 3, 0.5)",
    _SMALL + "loader = DataLoader(TensorDataset(X, y), batch_size=4)\nfit(model, loader, 2, 0.1)",
]

EXERCISES = {
    "digits-mlp": ClassBox(
        name="digits-mlp",
        start=_M_START,
        solution=_M,
        scenarios=_M_SCENARIOS,
        digits=4,
        save_as=("nn", "torch_mlp"),
        first=[_DOTS],
        mistakes=[
            Mistake("fixed-size", syntax=r"nn\.Linear\(\s*64\s*,\s*10\s*\)",
                    en="Python stops: with `DigitsMLP(16)` the hidden layer has 16 neurons, and your output layer expects 64 inputs. Give it `hidden` inputs: `nn.Linear(hidden, 10)`.",
                    nl="Python stopt: met `DigitsMLP(16)` heeft de hidden layer 16 neuronen, en je output layer verwacht 64 inputs. Geef hem `hidden` inputs: `nn.Linear(hidden, 10)`."),
            Mistake("order", matches=lambda got, text: re.search(r"nn\.Linear\(\s*hidden\s*,[\s\S]*nn\.Linear\(\s*784\s*,", text) is not None,
                    en="Your class may be right, but it makes the output layer before the hidden layer. PyTorch draws the starting weights in the order the layers are made, so your logits differ from those of the course's model. Keep the line with the hidden layer first, as it was.",
                    nl="Je class kan goed zijn, maar hij maakt de output layer vóór de hidden layer. PyTorch trekt de beginweights in de volgorde waarin de layers gemaakt worden, dus je logits verschillen van die van het model van de course. Houd de regel met de hidden layer voorop, zoals hij was."),
            Mistake("softmax", matches=lambda got, text: _sums_to_one(got),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyour model gives `{got}`: ten numbers from 0 to 1 that add up to 1, so your `forward` ends with softmax. `nn.CrossEntropyLoss` does the softmax itself. Give back the logits of the output layer.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\ngeeft je model `{got}`: tien getallen van 0 tot 1 die samen 1 zijn, dus je `forward` eindigt met softmax. `nn.CrossEntropyLoss` doet de softmax zelf. Geef de logits van de output layer terug."),
            Mistake("no-relu", wrong=_m(forward="        return self.out(self.hidden(x))\n"),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyour model gives `{got}`, and it should be `{want}`: your `forward` has no ReLU between the two layers. Without it, two layers together are just one weighted sum per digit. Put `torch.relu` around the hidden layer.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\ngeeft je model `{got}`, en dat moet `{want}` zijn: je `forward` heeft geen ReLU tussen de twee layers. Zonder ReLU zijn twee layers samen gewoon één gewogen som per cijfer. Zet `torch.relu` om de hidden layer."),
            Mistake("relu-out", wrong=_m(forward="        return torch.relu(self.out(torch.relu(self.hidden(x))))\n"),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyour model gives `{got}`, and it should be `{want}`: your logits are never below 0, so there is a ReLU after the output layer too. Logits may be negative: ReLU belongs only after the hidden layer.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\ngeeft je model `{got}`, en dat moet `{want}` zijn: je logits zijn nooit kleiner dan 0, dus staat er ook een ReLU na de output layer. Logits mogen negatief zijn: ReLU hoort alleen na de hidden layer."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "Replace the two `...`: the output layer, from the hidden neurons to 10 logits, and in `forward` the image through both layers, with ReLU in between.",
                      "nl": "Vervang de twee `...`: de output layer, van de hidden neuronen naar 10 logits, en in `forward` de afbeelding door beide layers, met ReLU ertussen."},
            "wrong": {"en": "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyour model gives `{got}`, and it should be `{want}`. Does your output layer turn `hidden` numbers into 10, and does `forward` take the image through the hidden layer, ReLU and the output layer, in that order?",
                      "nl": "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\ngeeft je model `{got}`, en dat moet `{want}` zijn. Maakt je output layer van `hidden` getallen er 10, en stuurt `forward` de afbeelding door de hidden layer, ReLU en de output layer, in die volgorde?"},
            "almost": {"en": "Right for 64 hidden neurons. But with these lines after your class:\n\n```python\n{scenario}\n```\n\nyour model gives `{got}`, and it should be `{want}`. Does your class use `hidden` everywhere, rather than the number 64?",
                       "nl": "Goed voor 64 hidden neuronen. Maar met deze regels na je class:\n\n```python\n{scenario}\n```\n\ngeeft je model `{got}`, en dat moet `{want}` zijn. Gebruikt je class overal `hidden`, en niet het getal 64?"},
            "correct": {"en": "Ten logits for each image, for 64 and for 16 hidden neurons. Your `DigitsMLP` is now in your project, as `work/projects/nn/torch_mlp.py`.",
                        "nl": "Tien logits voor elke afbeelding, voor 64 en voor 16 hidden neuronen. Je `DigitsMLP` staat nu in je project, als `work/projects/nn/torch_mlp.py`."},
            "kept": {"en": "Ten logits for each image, for 64 and for 16 hidden neurons. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Tien logits voor elke afbeelding, voor 64 en voor 16 hidden neuronen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
        },
    ),
    "fit-torch": ClassBox(
        name="fit-torch",
        start=_F_START,
        solution=_F,
        scenarios=_F_SCENARIOS,
        digits=4,
        save_as=("nn", "torch_train"),
        first=[_DOTS],
        mistakes=[
            Mistake("swapped", syntax=r"loss_fn\(\s*yb\s*,",
                    en="Python stops: `loss_fn` gets the labels first. `nn.CrossEntropyLoss` wants the logits first, then the labels: `loss_fn(model(xb), yb)`.",
                    nl="Python stopt: `loss_fn` krijgt eerst de labels. `nn.CrossEntropyLoss` wil eerst de logits, dan de labels: `loss_fn(model(xb), yb)`."),
            Mistake("no-zero", wrong=_f("            loss = loss_fn(model(xb), yb)\n            loss.backward()\n            opt.step()\n"),
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nyour `fit` gives `{got}`, and it should be `{want}`: the slopes of each mini-batch are added to those of the ones before, because nothing clears them. Call `opt.zero_grad()` before `loss.backward()`.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\ngeeft je `fit` `{got}`, en dat moet `{want}` zijn: de hellingen van elke mini-batch komen bij die van de vorige, omdat niets ze wist. Roep `opt.zero_grad()` aan vóór `loss.backward()`."),
            Mistake("double-softmax", wrong=_f("            loss = loss_fn(torch.softmax(model(xb), dim=1), yb)\n            opt.zero_grad()\n            loss.backward()\n            opt.step()\n"),
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nyour `fit` gives `{got}`, and it should be `{want}`: your loss gets probabilities after softmax, and `nn.CrossEntropyLoss` does softmax again. Give it the logits: `loss_fn(model(xb), yb)`.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\ngeeft je `fit` `{got}`, en dat moet `{want}` zijn: je loss krijgt kansen na softmax, en `nn.CrossEntropyLoss` doet nog een keer softmax. Geef hem de logits: `loss_fn(model(xb), yb)`."),
            Mistake("item-early", syntax=r"^\s*loss\s*=.*\.item\(\)\s*$",
                    en="Python stops: after `.item()`, `loss` is a plain number, not a tensor, and a plain number has no `.backward()` or `.item()`. Keep `loss` the tensor that `loss_fn` gives, and take `.item()` only where you add it to `total`.",
                    nl="Python stopt: na `.item()` is `loss` een gewoon getal, geen tensor, en een gewoon getal heeft geen `.backward()` of `.item()`. Laat `loss` de tensor die `loss_fn` geeft, en neem `.item()` pas waar je hem bij `total` optelt."),
            Mistake("float-labels", syntax=r"yb\.float\(\)",
                    en="Python stops: `nn.CrossEntropyLoss` wants the labels as whole numbers, int64, one digit per image, and `yb.float()` makes them floats. Give it `yb` as it comes from the loader.",
                    nl="Python stopt: `nn.CrossEntropyLoss` wil de labels als gehele getallen, int64, één cijfer per afbeelding, en `yb.float()` maakt er floats van. Geef hem `yb` zoals die uit de loader komt."),
            Mistake("no-total", matches=lambda got, text: isinstance(got, list) and len(got) > 0 and all(float(v) == 0 for v in got),
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nyour `fit` gives `{got}`: every epoch has a mean loss of 0, so `total` stays 0. Add the loss of each mini-batch to it: `total = total + loss.item()`.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\ngeeft je `fit` `{got}`: elke epoch heeft een gemiddelde loss van 0, dus `total` blijft 0. Tel de loss van elke mini-batch erbij op: `total = total + loss.item()`."),
            Mistake("no-step", matches=lambda got, text: isinstance(got, list) and len(got) > 1 and len(set(round(float(v), 6) for v in got)) == 1,
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nyour `fit` gives `{got}`: the same loss every epoch, so no parameter moved. Each mini-batch needs `opt.zero_grad()`, then `loss.backward()`, then `opt.step()`, in that order.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\ngeeft je `fit` `{got}`: elke epoch dezelfde loss, dus er is geen parameter veranderd. Elke mini-batch heeft `opt.zero_grad()`, dan `loss.backward()`, dan `opt.step()` nodig, in die volgorde."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "Replace the `...` with one update step for the mini-batch: its loss, then clear the slopes, work them out and take the step; then add its loss to `total`.",
                      "nl": "Vervang de `...` door één update-stap voor de mini-batch: zijn loss, dan de hellingen wissen, ze uitrekenen en de stap zetten; tel daarna zijn loss op bij `total`."},
            "wrong": {"en": "With these lines after your function:\n\n```python\n{scenario}\n```\n\nyour `fit` gives `{got}`, and it should be `{want}`. Does each mini-batch get its loss from `loss_fn`, and then the three steps of the optimizer, in the order of the recipe?",
                      "nl": "Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\ngeeft je `fit` `{got}`, en dat moet `{want}` zijn. Krijgt elke mini-batch zijn loss van `loss_fn`, en daarna de drie stappen van de optimizer, in de volgorde van het recept?"},
            "almost": {"en": "Right for the first call. But with these lines after your function:\n\n```python\n{scenario}\n```\n\nyour `fit` gives `{got}`, and it should be `{want}`. Does your step use `opt`, made with `lr`, and the mini-batches of `loader`?",
                       "nl": "Goed voor de eerste aanroep. Maar met deze regels na je functie:\n\n```python\n{scenario}\n```\n\ngeeft je `fit` `{got}`, en dat moet `{want}` zijn. Gebruikt je stap `opt`, gemaakt met `lr`, en de mini-batches van `loader`?"},
            "correct": {"en": "The loss goes down every epoch, for both calls. Your `fit` is now in your project, as `work/projects/nn/torch_train.py`.",
                        "nl": "De loss daalt elke epoch, bij beide aanroepen. Je `fit` staat nu in je project, als `work/projects/nn/torch_train.py`."},
            "kept": {"en": "The loss goes down every epoch, for both calls. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "De loss daalt elke epoch, bij beide aanroepen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("digits-mlp", _M_START, "empty", None),
    ("digits-mlp", _m(forward="        h = torch.relu(self.hidden(x))\n        return self.out(h)\n"), "correct", None),
    ("digits-mlp", _m(out="        self.out = nn.Linear(64, 10)\n"), "error", "fixed-size"),
    ("digits-mlp", _m(forward="        return torch.softmax(self.out(torch.relu(self.hidden(x))), dim=1)\n"), "wrong", "softmax"),
    ("digits-mlp", _m(forward="        return self.out(self.hidden(x))\n"), "wrong", "no-relu"),
    ("digits-mlp", _m(forward="        return torch.relu(self.out(torch.relu(self.hidden(x))))\n"), "wrong", "relu-out"),
    ("digits-mlp", _m(forward="        return self.out(torch.sigmoid(self.hidden(x)))\n"), "wrong", None),
    ("digits-mlp", _m(out="        self.out = nn.Linear(hidden, 1)\n"), "wrong", None),
    ("fit-torch", _F_START, "empty", None),
    ("fit-torch", _f("            logits = model(xb)\n            loss = loss_fn(logits, yb)\n            opt.zero_grad()\n            loss.backward()\n            opt.step()\n"), "correct", None),
    ("fit-torch", _f("            opt.zero_grad()\n            loss = loss_fn(model(xb), yb)\n            loss.backward()\n            opt.step()\n"), "correct", None),
    ("fit-torch", _f("            loss = loss_fn(yb, model(xb))\n            opt.zero_grad()\n            loss.backward()\n            opt.step()\n"), "error", "swapped"),
    ("fit-torch", _f("            loss = loss_fn(model(xb), yb)\n            loss.backward()\n            opt.step()\n"), "wrong", "no-zero"),
    ("fit-torch", _f("            loss = loss_fn(torch.softmax(model(xb), dim=1), yb)\n            opt.zero_grad()\n            loss.backward()\n            opt.step()\n"), "wrong", "double-softmax"),
    ("fit-torch", _f("            loss = loss_fn(model(xb), yb)\n            opt.zero_grad()\n            opt.step()\n            loss.backward()\n"), "wrong", "no-step"),
    ("fit-torch", _f(total=""), "wrong", "no-total"),
    ("fit-torch", _f("            loss = loss_fn(model(xb), yb).item()\n            opt.zero_grad()\n            loss.backward()\n            opt.step()\n", total="            total = total + loss\n"), "error", "item-early"),
    ("fit-torch", _f("            loss = loss_fn(model(xb), yb.float())\n            opt.zero_grad()\n            loss.backward()\n            opt.step()\n"), "error", "float-labels"),
    ("digits-mlp", _M_HEAD.replace("        self.hidden = nn.Linear(784, hidden)\n", "        self.out = nn.Linear(hidden, 10)\n        self.hidden = nn.Linear(784, hidden)\n") + "\n    def forward(self, x):\n" + _M_FORWARD, "wrong", "order"),
]
