# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson torch-loss-curves: the loss on the training digits and on a validation set after every epoch,
with the learner's own DigitsMLP and fit from torch-mlp-digits, or the course's where theirs is missing or broken;
and which remedy goes with which curve. And the pictures: three runs side by side, still learning, overfitting and
stuck, and one run to diagnose."""
import functools
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import paths, projects
from everything_ai.check import CodeBox, Mistake, Sort

# The modules of the lesson before: yours when they are there and give the course's results, else the course's
# (template 1.3).
projects.builds_on(("nn", "torch_mlp", "torch-mlp-digits", "digits-mlp"), ("nn", "torch_train", "torch-mlp-digits", "fit-torch"))

DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md
VAL = 500                        # the validation set: training images 5000 to 5499, which no run trains on


@functools.lru_cache(maxsize=1)
def tensors():
    """The training images, every pixel divided by 255, and their labels."""
    d = np.load(paths.ROOT / DATA)
    return torch.tensor(d["train_images"], dtype=torch.float32) / 255, torch.tensor(d["train_labels"], dtype=torch.int64)


class _MLP(nn.Module):
    """The course's own DigitsMLP, for the pictures."""
    def __init__(self, hidden):
        super().__init__()
        self.hidden = nn.Linear(784, hidden)
        self.out = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.out(torch.relu(self.hidden(x)))


@functools.lru_cache(maxsize=None)
def curves(n: int, lr: float, epochs: int) -> tuple[list[float], list[float]]:
    """One run, as in the worked example: an MLP of 64 hidden neurons trained on the first `n` training digits, in
    mini-batches of 50; after every epoch the loss on those digits and on the validation set. Worked out once."""
    X, y = tensors()
    X_train, y_train, X_val, y_val = X[:n], y[:n], X[5000:5000 + VAL], y[5000:5000 + VAL]
    loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(0)
    model = _MLP(64)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=50, shuffle=True)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    torch.manual_seed(1)
    train_losses, val_losses = [], []
    for _ in range(epochs):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            train_losses.append(loss_fn(model(X_train), y_train).item())
            val_losses.append(loss_fn(model(X_val), y_val).item())
        model.train()
    return train_losses, val_losses


RUNS = {"learning": (1000, 0.05, 30), "overfitting": (1000, 0.5, 30), "stuck": (1000, 0.001, 30)}   # the three panels
DIAGNOSE = (300, 0.5, 30)   # exercise 1 and 2: 300 training digits

# The run of exercise 1, worked out once when the lesson opens, for its picture: this also gets PyTorch's first
# training steps, which take about a second, out of the way before a check times the learner's code.
curves(*DIAGNOSE)


def best(key_or_run) -> tuple[int, float]:
    """The epoch, counted from 1, with the lowest validation loss, and that loss."""
    _, val = curves(*(RUNS[key_or_run] if isinstance(key_or_run, str) else key_or_run))
    k = int(np.argmin(val))
    return k + 1, val[k]


def _panel(run, title: str, labels: dict[str, str], mark: bool = False):
    from everything_ai import chart
    train, val = curves(*run)
    epochs = list(range(1, len(train) + 1))
    p = chart.Panel((0, len(train) + 1), (0.003, 3), title=title, xlabel=labels["epoch"], ylabel=labels["loss"], ylog=True)
    p.line(epochs, train, "loss", 2.2)
    p.line(epochs, val, "loss", 2.2, dashed=True)
    if mark:   # the lowest validation loss: on a log scale, the rise after it is small to see
        k = int(np.argmin(val))
        p.points([k + 1], [val[k]], "loss", 4.5)
        p.text(k + 2.5, val[k] * 0.55, labels["lowest"], "ink", "start")
    return p


def picture(labels: dict[str, str]) -> str:
    """Three runs side by side, each with the training loss solid and the validation loss dashed, on a log scale."""
    from everything_ai import chart
    panels = [_panel(RUNS[key], labels[key], labels, mark=key == "overfitting") for key in ("learning", "overfitting", "stuck")]
    return chart.figure(panels, alt=labels["alt"], width=250, height=260)


def diagnose_picture(labels: dict[str, str]) -> str:
    """The run of exercise 1, drawn the same way, with a key."""
    from everything_ai import chart
    p = _panel(DIAGNOSE, labels["title"], labels)
    p.text(1, 0.0045, labels["key"], "ink", "start")
    return chart.figure([p], alt=labels["alt"], width=520, height=300)


# The lesson is done when these are correct. "diagnose" is the choice question in the lesson.
REQUIRED = ["diagnose", "curves", "remedy"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# --- exercise 2: the two curves ---------------------------------------------------------------------------------------

_C_HEAD = ("import numpy as np\nimport torch\nfrom torch import nn\nfrom torch.utils.data import DataLoader, TensorDataset\n"
           "from nn.torch_mlp import DigitsMLP\nfrom nn.torch_train import fit\n\n"
           "data = np.load(\"data/qmnist-small.npz\")\nX = torch.tensor(data[\"train_images\"], dtype=torch.float32) / 255\n"
           "y = torch.tensor(data[\"train_labels\"], dtype=torch.int64)\nX_train, y_train = X[:300], y[:300]\n"
           "X_val, y_val = X[5000:5500], y[5000:5500]\nloss_fn = nn.CrossEntropyLoss()\n\n"
           "torch.manual_seed(0)\nmodel = DigitsMLP(64)\nloader = DataLoader(TensorDataset(X_train, y_train), batch_size=50, shuffle=True)\n"
           "torch.manual_seed(1)\ntrain_losses = []\nval_losses = []\nfor epoch in range(30):\n    fit(model, loader, 1, 0.5)\n")
_C_BODY = ("    model.eval()\n    with torch.no_grad():\n        train_losses.append(loss_fn(model(X_train), y_train).item())\n"
           "        val_losses.append(loss_fn(model(X_val), y_val).item())\n    model.train()\n")
_C_END = "[train_losses[-1], val_losses[-1], min(val_losses)]"
_C_START = _C_HEAD + "    ...\n" + _C_END + "\n"


def _c(body: str = _C_BODY) -> str:
    return _C_HEAD + body + _C_END


_C = _c()


def _three(got) -> bool:
    return isinstance(got, list) and len(got) == 3


EXERCISES = {
    "curves": CodeBox(
        name="curves",
        start=_C_START,
        solution=_C,
        keep=["for epoch in range(30):", "    fit(model, loader, 1, 0.5)", _C_END],
        first=[_DOTS],
        mistakes=[
            Mistake("empty-list", error=lambda e: isinstance(e, IndexError) or (isinstance(e, ValueError) and "empty" in str(e)),
                    en="Python stops on the last line: `train_losses` or `val_losses` is still empty. After each epoch, append the loss on `X_train` and `y_train` to the one, and the loss on `X_val` and `y_val` to the other.",
                    nl="Python stopt op de laatste regel: `train_losses` of `val_losses` is nog leeg. Voeg na elke epoch de loss op `X_train` en `y_train` aan de ene toe, en de loss op `X_val` en `y_val` aan de andere."),
            Mistake("swapped", wrong=_c("    model.eval()\n    with torch.no_grad():\n        train_losses.append(loss_fn(model(X_val), y_val).item())\n"
                                        "        val_losses.append(loss_fn(model(X_train), y_train).item())\n    model.train()\n"),
                    en="Your last line gives `{got}`: the training loss is high and the validation loss close to 0, so the two lists are swapped. `train_losses` gets the loss on `X_train`, `val_losses` the loss on `X_val`.",
                    nl="Je laatste regel geeft `{got}`: de training loss is hoog en de validation loss bijna 0, dus de twee lijsten zijn verwisseld. `train_losses` krijgt de loss op `X_train`, `val_losses` de loss op `X_val`."),
            Mistake("train-twice", wrong=_c("    model.eval()\n    with torch.no_grad():\n        train_losses.append(loss_fn(model(X_train), y_train).item())\n"
                                            "        val_losses.append(loss_fn(model(X_train), y_train).item())\n    model.train()\n"),
                    en="Your last line gives `{got}`: the validation loss is the same as the training loss, so both measure the training digits. The validation loss uses `X_val` and `y_val`, digits the model never trains on.",
                    nl="Je laatste regel geeft `{got}`: de validation loss is gelijk aan de training loss, dus beide meten de trainingscijfers. De validation loss gebruikt `X_val` en `y_val`, cijfers waarop het model nooit traint."),
            Mistake("before-fit", wrong=_C_HEAD.replace("    fit(model, loader, 1, 0.5)\n", "") + _C_BODY + "    fit(model, loader, 1, 0.5)\n" + _C_END,
                    en="Your last line gives `{got}`: each epoch measures the losses before it trains, so the last numbers belong to the model after 29 epochs, not 30. Measure after the line with `fit`.",
                    nl="Je laatste regel geeft `{got}`: elke epoch meet de losses voordat hij traint, dus de laatste getallen horen bij het model na 29 epochs, niet 30. Meet na de regel met `fit`."),
            Mistake("once", matches=lambda got, text: _three(got) and abs(float(got[1]) - float(got[2])) < 1e-9 and float(got[0]) < 0.05 < float(got[1]),
                    en="Your last line gives `{got}`: the lowest validation loss is the last one, and the training loss is close to 0. Your lists probably hold one number each, from after the loop. Measure inside the loop over the epochs, indented under `for epoch in range(30):`.",
                    nl="Je laatste regel geeft `{got}`: de laagste validation loss is de laatste, en de training loss is bijna 0. Je lijsten hebben waarschijnlijk elk één getal, van na de loop. Meet binnen de loop over de epochs, ingesprongen onder `for epoch in range(30):`."),
        ],
        digits=4,
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
            "empty": {"en": "Replace the `...` with step 2 of the recipe: after each epoch, measure the loss on all training digits and on the validation set, and append each to its list.",
                      "nl": "Vervang de `...` door stap 2 van het recept: meet na elke epoch de loss op alle trainingscijfers en op de validation set, en voeg ze elk aan hun lijst toe."},
            "no_value": {"en": "The last line of the box shows nothing. Put the list back on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet de lijst terug op de laatste regel, zodat de check hem kan zien."},
            "wrong": {"en": "Your last line gives `{got}`: the last training loss, the last validation loss and the lowest validation loss should be about 0.0098, 0.7589 and 0.6424. Does each epoch append one loss over all of `X_train` and one over all of `X_val`?",
                      "nl": "Je laatste regel geeft `{got}`: de laatste training loss, de laatste validation loss en de laagste validation loss moeten ongeveer 0,0098, 0,7589 en 0,6424 zijn. Voegt elke epoch één loss over heel `X_train` en één over heel `X_val` toe?"},
            "correct": {"en": "The training loss ends close to 0, at 0.0098, and the validation loss at 0.7589, above its lowest point, 0.6424: these are the curves of exercise 1. The model learned its 300 digits by heart.",
                        "nl": "De training loss eindigt bijna op 0, op 0,0098, en de validation loss op 0,7589, boven zijn laagste punt, 0,6424: dit zijn de curves van oefening 1. Het model heeft zijn 300 cijfers uit zijn hoofd geleerd."},
        },
    ),
    "remedy": Sort(
        name="remedy",
        groups={
            "learning": {"en": "Still learning", "nl": "Leert nog"},
            "overfitting": {"en": "Overfitting", "nl": "Overfitting"},
            "stuck": {"en": "Stuck", "nl": "Zit vast"},
        },
        items={
            "longer": {"en": "Train for more epochs", "nl": "Train meer epochs"},
            "keep-best": {"en": "Keep the model of the epoch with the lowest validation loss", "nl": "Bewaar het model van de epoch met de laagste validation loss"},
            "more-data": {"en": "Train on more examples", "nl": "Train op meer voorbeelden"},
            "raise-lr": {"en": "Raise the learning rate", "nl": "Verhoog de learning rate"},
            "check-step": {"en": "Check that each update step changes the parameters", "nl": "Controleer of elke update-stap de parameters verandert"},
        },
        solution={"longer": "learning", "keep-best": "overfitting", "more-data": "overfitting", "raise-lr": "stuck", "check-step": "stuck"},
        nudge={
            "longer": {"en": "Which curves had come down a long way, and were still coming down, when training stopped?",
                       "nl": "Welke curves waren al een eind gedaald, en daalden nog, toen het trainen stopte?"},
            "keep-best": {"en": "In which picture is a later epoch worse on new digits than an earlier one?",
                          "nl": "In welke figuur is een latere epoch slechter op nieuwe cijfers dan een eerdere?"},
            "more-data": {"en": "When does a model learn its examples by heart, and when do more examples make that harder?",
                          "nl": "Wanneer leert een model zijn voorbeelden uit zijn hoofd, en wanneer maken meer voorbeelden dat moeilijker?"},
            "raise-lr": {"en": "With which curves does every step move the parameters too little?",
                         "nl": "Bij welke curves verplaatst elke stap de parameters te weinig?"},
            "check-step": {"en": "Which curves would a forgotten `opt.step()` give?",
                           "nl": "Welke curves zou een vergeten `opt.step()` geven?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the remedies again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de oplossingen opnieuw uit."},
            "empty": {"en": "Put each remedy with the curves whose problem it fixes: still learning, overfitting or stuck.",
                      "nl": "Zet elke oplossing bij de curves met het probleem dat ze verhelpt: leert nog, overfitting of zit vast."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs with other curves.",
                      "nl": "{item} hoort bij andere curves."},
            "correct": {"en": "All {n} are right. Still learning: train longer. Overfitting: keep the best epoch, or give the model more examples. Stuck: a larger step, or find out why the step does nothing.",
                        "nl": "Alle {n} kloppen. Leert nog: train langer. Overfitting: bewaar de beste epoch, of geef het model meer voorbeelden. Zit vast: een grotere stap, of zoek uit waarom de stap niets doet."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("curves", _C_START, "empty", None),
    ("curves", _c("    model.eval()\n    with torch.no_grad():\n        loss_train = loss_fn(model(X_train), y_train)\n        loss_val = loss_fn(model(X_val), y_val)\n"
                  "    model.train()\n    train_losses.append(loss_train.item())\n    val_losses.append(loss_val.item())\n"), "correct", None),
    ("curves", _c("    with torch.no_grad():\n        train_losses.append(loss_fn(model(X_train), y_train).item())\n"
                  "        val_losses.append(loss_fn(model(X_val), y_val).item())\n"), "correct", None),
    ("curves", _c("    model.eval()\n    with torch.no_grad():\n        train_losses.append(loss_fn(model(X_train), y_train).item())\n    model.train()\n"), "error", "empty-list"),
    ("curves", _c("    model.eval()\n    with torch.no_grad():\n        train_losses.append(loss_fn(model(X_val), y_val).item())\n"
                  "        val_losses.append(loss_fn(model(X_train), y_train).item())\n    model.train()\n"), "wrong", "swapped"),
    ("curves", _c("    model.eval()\n    with torch.no_grad():\n        train_losses.append(loss_fn(model(X_train), y_train).item())\n"
                  "        val_losses.append(loss_fn(model(X_train), y_train).item())\n    model.train()\n"), "wrong", "train-twice"),
    ("curves", _c("    pass\nmodel.eval()\nwith torch.no_grad():\n    train_losses.append(loss_fn(model(X_train), y_train).item())\n"
                  "    val_losses.append(loss_fn(model(X_val), y_val).item())\nmodel.train()\n"), "wrong", "once"),
    ("curves", _C_HEAD.replace("    fit(model, loader, 1, 0.5)\n", "") + _C_BODY + "    fit(model, loader, 1, 0.5)\n" + _C_END, "wrong", "before-fit"),
]
