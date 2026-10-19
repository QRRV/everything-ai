# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-early-stopping: a training loop that saves a checkpoint at every new lowest validation
loss and stops after `patience` epochs without one, with the learner's own DigitsMLP and fit from torch-mlp-digits, or
the course's where theirs is missing or broken. And the picture: where early stopping stops on the run of
torch-loss-curves, and which epoch it keeps."""
import functools
import re
from pathlib import Path

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import paths, projects
from everything_ai.check import CodeBox, Mistake

# The modules of the lesson before: yours when they are there and give the course's results, else the course's
# (template 1.3).
projects.builds_on(("nn", "torch_mlp", "torch-mlp-digits", "digits-mlp"), ("nn", "torch_train", "torch-mlp-digits", "fit-torch"))


def _warm_up() -> None:
    """The first optimizer step in a process loads more of PyTorch, about a second. Take it here, when the lesson
    opens, so it never counts against the time limit of the check."""
    w = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.1)
    (w * w).sum().backward()
    opt.step()


_warm_up()

DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md


@functools.lru_cache(maxsize=1)
def _tensors():
    d = np.load(paths.ROOT / DATA)
    return torch.tensor(d["train_images"], dtype=torch.float32) / 255, torch.tensor(d["train_labels"], dtype=torch.int64)


class _MLP(nn.Module):
    """The course's own DigitsMLP, for the picture."""
    def __init__(self, hidden):
        super().__init__()
        self.hidden = nn.Linear(784, hidden)
        self.out = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.out(torch.relu(self.hidden(x)))


@functools.lru_cache(maxsize=None)
def curves(n: int = 1000, epochs: int = 30) -> tuple[list[float], list[float]]:
    """The run of the worked example of torch-loss-curves: 64 hidden neurons, the first `n` training digits, mini-batches
    of 50, a learning rate of 0.5; after every epoch the loss on those digits and on the 500 validation digits."""
    X, y = _tensors()
    X_train, y_train, X_val, y_val = X[:n], y[:n], X[5000:5500], y[5000:5500]
    loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(0)
    model = _MLP(64)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=50, shuffle=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
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


def stop(patience: int, n: int = 1000) -> tuple[int, int, float]:
    """Early stopping on that run: the epoch after which it stops, the epoch it keeps, and that validation loss."""
    _, val = curves(n)
    best, best_epoch, waited = float("inf"), 0, 0
    for epoch, v in enumerate(val, 1):
        if v < best:
            best, best_epoch, waited = v, epoch, 0
        else:
            waited += 1
            if waited == patience:
                return epoch, best_epoch, best
    return len(val), best_epoch, best


def picture(patience: int, labels: dict[str, str]) -> str:
    """The training loss solid and the validation loss dashed, on a log scale, up to the epoch where early stopping
    stops; a dot at the epoch it keeps, and a thin line at the stop."""
    from everything_ai import chart
    train, val = curves()
    end, best_epoch, best = stop(patience)
    epochs = list(range(1, end + 1))
    p = chart.Panel((0, 31), (0.01, 3), title=labels["title"], xlabel=labels["epoch"], ylabel=labels["loss"], ylog=True)
    p.line(epochs, train[:end], "loss", 2.2)
    p.line(epochs, val[:end], "loss", 2.2, dashed=True)
    p.points([best_epoch], [best], "loss", 4.5)
    p.text(best_epoch, best * 1.9, labels["kept"], "ink", "middle")
    p.line([end, end], [0.01, 3], "ink", 1, dashed=True)
    p.text(end + 0.5, 1.6, labels["stop"], "ink", "start")
    return chart.figure([p], alt=labels["alt"], width=520, height=300)


# The lesson is done when these are correct. "which-model" and "patience" are choice questions in the lesson.
REQUIRED = ["which-model", "early-stop", "patience"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

def checkpoint_path(name: str) -> str:
    """For the worked example: the path of a file in the learner's work folder, work/checkpoints/<name>."""
    folder = paths.WORK / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / name)


def fresh_path(name: str) -> str:
    """For the exercise: a path in the check's own folder, work/checkpoints/check/<name>, with the file of an earlier
    run removed first, so an old file never gets in the way. One fixed folder, so nothing piles up. In the box it is
    called checkpoint_path, as in torch-checkpoints."""
    folder = Path(checkpoint_path("check"))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.unlink(missing_ok=True)
    return str(path)


# --- exercise 2: early stopping ----------------------------------------------------------------------------------------

_E_HEAD = ("import numpy as np\nimport torch\nfrom torch import nn\nfrom torch.utils.data import DataLoader, TensorDataset\n"
           "from nn.torch_mlp import DigitsMLP\nfrom nn.torch_train import fit\n\n"
           "data = np.load(\"data/qmnist-small.npz\")\nX = torch.tensor(data[\"train_images\"], dtype=torch.float32) / 255\n"
           "y = torch.tensor(data[\"train_labels\"], dtype=torch.int64)\nX_train, y_train = X[:300], y[:300]\n"
           "X_val, y_val = X[5000:5500], y[5000:5500]\nloss_fn = nn.CrossEntropyLoss()\npath = checkpoint_path(\"best.pt\")\n\n\n"
           "def val_loss(model):\n    model.eval()\n    with torch.no_grad():\n        loss = loss_fn(model(X_val), y_val).item()\n"
           "    model.train()\n    return loss\n\n\n"
           "torch.manual_seed(0)\nmodel = DigitsMLP(64)\nloader = DataLoader(TensorDataset(X_train, y_train), batch_size=50, shuffle=True)\n"
           "torch.manual_seed(1)\npatience = 5\nval_losses = []\nbest_epoch = 0\nwaited = 0\nfor epoch in range(100):\n"
           "    fit(model, loader, 1, 0.5)\n    val_losses.append(val_loss(model))\n")
_E_BODY = ("    if val_losses[-1] == min(val_losses):\n        torch.save(model.state_dict(), path)\n        best_epoch = epoch + 1\n"
           "        waited = 0\n    else:\n        waited = waited + 1\n        if waited == patience:\n            break\n")
_E_END = "\nbest = DigitsMLP(64)\nbest.load_state_dict(torch.load(path))\n[epoch + 1, best_epoch, val_loss(best)]"
_E_START = _E_HEAD + "    ...\n" + _E_END + "\n"


def _e(body: str = _E_BODY) -> str:
    return _E_HEAD + body + _E_END


_E = _e()


def _got3(got) -> bool:
    return isinstance(got, list) and len(got) == 3


EXERCISES = {
    "early-stop": CodeBox(
        name="early-stop",
        start=_E_START,
        solution=_E,
        keep=["path = checkpoint_path(\"best.pt\")", "patience = 5", "for epoch in range(100):", "best = DigitsMLP(64)",
              "best.load_state_dict(torch.load(path))", "[epoch + 1, best_epoch, val_loss(best)]"],
        names={"checkpoint_path": fresh_path},
        first=[_DOTS],
        mistakes=[
            Mistake("never-saved", error=lambda e: isinstance(e, FileNotFoundError) and "checkpoints" in str(getattr(e, "filename", "")),
                    en="Python stops at `torch.load(path)`: there is no file at `path`, so nothing was saved. At a new lowest validation loss, save the parameters: `torch.save(model.state_dict(), path)`.",
                    nl="Python stopt bij `torch.load(path)`: er staat geen bestand op `path`, dus er is niets opgeslagen. Sla bij een nieuwe laagste validation loss de parameters op: `torch.save(model.state_dict(), path)`."),
            Mistake("whole-model", error=lambda e: "Weights only load failed" in str(e) or "weights_only" in str(e),
                    en="Python stops at `torch.load(path)`: the file holds the whole model object, and `torch.load` reads back only tensors and simple values. Save `model.state_dict()`, the parameters by name, not `model` itself.",
                    nl="Python stopt bij `torch.load(path)`: het bestand bevat het hele model-object, en `torch.load` leest alleen tensors en eenvoudige waarden terug. Sla `model.state_dict()` op, de parameters per naam, niet `model` zelf."),
            Mistake("no-stop", matches=lambda got, text: _got3(got) and got[0] == 100,
                    en="Your last line gives `{got}`: training ran all 100 epochs, so it never stopped. Does `waited` go up by 1 at every epoch without a new lowest, and does Python reach a `break` once it equals `patience`?",
                    nl="Je laatste regel geeft `{got}`: het trainen liep alle 100 epochs, dus het stopte nooit. Gaat `waited` omhoog met 1 bij elke epoch zonder nieuwe laagste, en komt Python bij een `break` zodra die gelijk is aan `patience`?"),
            Mistake("no-reset", wrong=_e(_E_BODY.replace("        waited = 0\n", "")),
                    en="Your last line gives `{got}`: training stops an epoch too early, after only 4 epochs without a new lowest since the best one. `waited` should count only those epochs: add 1 only in the `else`, and set it back to 0 at every new lowest.",
                    nl="Je laatste regel geeft `{got}`: het trainen stopt een epoch te vroeg, na maar 4 epochs zonder nieuwe laagste sinds de beste. `waited` hoort alleen die epochs te tellen: tel er alleen in de `else` 1 bij op, en zet hem bij elke nieuwe laagste terug op 0."),
            Mistake("one-late", wrong=_e(_E_BODY.replace("if waited == patience:", "if waited > patience:")),
                    en="Your last line gives `{got}`: training stops one epoch late, after 6 epochs without a new lowest instead of 5. Stop as soon as `waited` reaches `patience`: first add 1, then test `waited == patience`.",
                    nl="Je laatste regel geeft `{got}`: het trainen stopt een epoch te laat, na 6 epochs zonder nieuwe laagste in plaats van 5. Stop zodra `waited` gelijk is aan `patience`: tel eerst 1 op, en test dan `waited == patience`."),
            Mistake("saved-every-epoch", wrong=_e(_E_BODY.replace("        torch.save(model.state_dict(), path)\n", "").replace("    if val_losses[-1]", "    torch.save(model.state_dict(), path)\n    if val_losses[-1]")),
                    en="Your last line gives `{got}`: the loaded model has the validation loss of the epoch you stop at, not that of epoch 9, so the file is also written at epochs without a new lowest. Save only at a new lowest validation loss, inside the `if`.",
                    nl="Je laatste regel geeft `{got}`: het geladen model heeft de validation loss van de epoch waarop je stopt, niet die van epoch 9, dus het bestand wordt ook geschreven bij epochs zonder nieuwe laagste. Sla alleen op bij een nieuwe laagste validation loss, binnen de `if`."),
            Mistake("epoch-from-0", wrong=_e(_E_BODY.replace("best_epoch = epoch + 1", "best_epoch = epoch")),
                    en="Your last line gives `{got}`: the epoch you keep is one too low. `epoch` counts from 0, so after epoch 9 it is 8: remember `epoch + 1` in `best_epoch`.",
                    nl="Je laatste regel geeft `{got}`: de epoch die je bewaart, is er één te laag. `epoch` telt vanaf 0, dus na epoch 9 is `epoch` gelijk aan 8: onthoud `epoch + 1` in `best_epoch`."),
        ],
        digits=4,
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
            "empty": {"en": "Replace the `...` with the rest of the epoch: at a new lowest validation loss, save and remember the epoch; otherwise count the epoch, and stop when the count reaches `patience`.",
                      "nl": "Vervang de `...` door de rest van de epoch: sla bij een nieuwe laagste validation loss op en onthoud de epoch; tel anders de epoch, en stop als de telling `patience` bereikt."},
            "no_value": {"en": "The last line of the box shows nothing. Put the list back on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet de lijst terug op de laatste regel, zodat de check het kan zien."},
            "slow": {"en": "Your code is still not done after 2 seconds. Does each epoch call `fit` once, and does the loop stop when `waited` reaches `patience`?",
                     "nl": "Je code is na 2 seconden nog niet klaar. Roept elke epoch `fit` één keer aan, en stopt de loop als `waited` `patience` bereikt?"},
            "wrong": {"en": "Your last line gives `{got}`: the epoch after which training stops, the epoch it keeps and that epoch's validation loss should be 14, 9 and about 0.6424. Which of the three is different? Compare that part of your code with steps 3 and 4 of the recipe.",
                      "nl": "Je laatste regel geeft `{got}`: de epoch waarna het trainen stopt, de epoch die het bewaart en de validation loss daarvan moeten 14, 9 en ongeveer 0,6424 zijn. Welke van de drie wijkt af? Vergelijk dat deel van je code met stap 3 en 4 van het recept."},
            "correct": {"en": "Training stops after epoch 14, five epochs after the lowest validation loss, and keeps epoch 9, with 0.6424. The 30 epochs of the lesson before were 16 more than needed.",
                        "nl": "Het trainen stopt na epoch 14, vijf epochs na de laagste validation loss, en bewaart epoch 9, met 0,6424. De 30 epochs uit de vorige les waren er 16 meer dan nodig."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("early-stop", _E_START, "empty", None),
    ("early-stop", _e("    if val_losses[-1] == min(val_losses):\n        torch.save(model.state_dict(), path)\n        best_epoch = epoch + 1\n"
                      "        waited = 0\n    else:\n        waited += 1\n    if waited == patience:\n        break\n"), "correct", None),
    ("early-stop", _e(_E_BODY.replace("        torch.save(model.state_dict(), path)\n", "")), "error", "never-saved"),
    ("early-stop", _e(_E_BODY.replace("        if waited == patience:\n            break\n", "")), "wrong", "no-stop"),
    ("early-stop", _e(_E_BODY.replace("        waited = 0\n", "")), "wrong", "no-reset"),
    ("early-stop", _e(_E_BODY.replace("if waited == patience:", "if waited > patience:")), "wrong", "one-late"),
    ("early-stop", _e(_E_BODY.replace("        torch.save(model.state_dict(), path)\n", "").replace("    if val_losses[-1]", "    torch.save(model.state_dict(), path)\n    if val_losses[-1]")), "wrong", "saved-every-epoch"),
    # other right answers: <= instead of ==; the test turned round, with >= and len(val_losses)
    ("early-stop", _e(_E_BODY.replace("if val_losses[-1] == min(val_losses):", "if val_losses[-1] <= min(val_losses):")), "correct", None),
    ("early-stop", _e("    if val_losses[-1] > min(val_losses):\n        waited = waited + 1\n        if waited >= patience:\n            break\n"
                      "    else:\n        torch.save(model.state_dict(), path)\n        best_epoch = len(val_losses)\n        waited = 0\n"), "correct", None),
    # the same outcome from other slips, with a message that is true for them too
    ("early-stop", _e("    if val_losses[-1] == min(val_losses):\n        torch.save(model.state_dict(), path)\n        best_epoch = epoch + 1\n"
                      "        waited = 0\n    waited = waited + 1\n    if waited == patience:\n        break\n"), "wrong", "no-reset"),
    ("early-stop", _e(_E_BODY.replace("        torch.save(model.state_dict(), path)\n", "").replace("        waited = waited + 1\n", "        waited = waited + 1\n        torch.save(model.state_dict(), path)\n")), "wrong", "saved-every-epoch"),
    ("early-stop", _e(_E_BODY.replace("        waited = waited + 1\n        if waited == patience:\n            break\n", "        if waited == patience:\n            break\n        waited = waited + 1\n")), "wrong", "one-late"),
    ("early-stop", _e(_E_BODY.replace("best_epoch = epoch + 1", "best_epoch = epoch")), "wrong", "epoch-from-0"),
    ("early-stop", _e(_E_BODY.replace("torch.save(model.state_dict(), path)", "torch.save(model, path)")), "error", "whole-model"),
    ("early-stop", _e(_E_BODY.replace("        best_epoch = epoch + 1\n", "")), "wrong", None),
]
