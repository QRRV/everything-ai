# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson torch-checkpoints: keeping the model of the best epoch in a file with torch.save and
loading it back with load_state_dict, with the learner's own DigitsMLP and fit from torch-mlp-digits, or the course's
where theirs is missing or broken; and what goes into a checkpoint. And the helpers for the worked example: where the
learner's checkpoints go, and the state_dict of DigitsMLP(64) for the picture."""
import re
from pathlib import Path

import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import paths, projects
from everything_ai.check import CodeBox, Mistake, Sort

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


def checkpoint_path(name: str) -> str:
    """For the worked example: the path of a file in the learner's work folder, work/checkpoints/<name>."""
    folder = paths.WORK / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / name)


def fresh_path(name: str) -> str:
    """For the exercise: a path in the check's own folder, work/checkpoints/check/<name>, with the file of an earlier
    run removed first, so an old file never gets in the way. One fixed folder, so nothing piles up. In the box it is
    called checkpoint_path, as in the worked example."""
    folder = Path(checkpoint_path("check"))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.unlink(missing_ok=True)
    return str(path)


def state_pairs() -> list[tuple[str, str]]:
    """The names and shapes in the state_dict of DigitsMLP(64), for the picture."""
    return [("hidden.weight", "tensor (64, 784)"), ("hidden.bias", "tensor (64,)"),
            ("out.weight", "tensor (10, 64)"), ("out.bias", "tensor (10,)")]


# The lesson is done when these are correct. "what-saved" is the choice question in the lesson.
REQUIRED = ["what-saved", "best-epoch", "checkpoint"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# --- exercise 2: keep the best epoch ----------------------------------------------------------------------------------

_B_HEAD = ("import numpy as np\nimport torch\nfrom torch import nn\nfrom torch.utils.data import DataLoader, TensorDataset\n"
           "from nn.torch_mlp import DigitsMLP\nfrom nn.torch_train import fit\n\n"
           "data = np.load(\"data/qmnist-small.npz\")\nX = torch.tensor(data[\"train_images\"], dtype=torch.float32) / 255\n"
           "y = torch.tensor(data[\"train_labels\"], dtype=torch.int64)\nX_train, y_train = X[:300], y[:300]\n"
           "X_val, y_val = X[5000:5500], y[5000:5500]\nloss_fn = nn.CrossEntropyLoss()\npath = checkpoint_path(\"best.pt\")\n\n\n"
           "def val_loss(model):\n    model.eval()\n    with torch.no_grad():\n        loss = loss_fn(model(X_val), y_val).item()\n"
           "    model.train()\n    return loss\n\n\n"
           "torch.manual_seed(0)\nmodel = DigitsMLP(64)\nloader = DataLoader(TensorDataset(X_train, y_train), batch_size=50, shuffle=True)\n"
           "torch.manual_seed(1)\nval_losses = []\nfor epoch in range(30):\n    fit(model, loader, 1, 0.5)\n"
           "    val_losses.append(val_loss(model))\n    if val_losses[-1] == min(val_losses):\n")
_B_SAVE = "        torch.save(model.state_dict(), path)\n"
_B_MID = "\nbest = DigitsMLP(64)\n"
_B_LOAD = "best.load_state_dict(torch.load(path))\n"
_B_END = "[val_loss(model), val_loss(best)]"
_B_START = _B_HEAD + "        ...\n" + _B_MID + "...\n" + _B_END + "\n"


def _b(save: str = _B_SAVE, load: str = _B_LOAD) -> str:
    return _B_HEAD + save + _B_MID + load + _B_END


_B = _b()


def _same_two(got) -> bool:
    return isinstance(got, list) and len(got) == 2 and abs(float(got[0]) - float(got[1])) < 1e-9


EXERCISES = {
    "best-epoch": CodeBox(
        name="best-epoch",
        start=_B_START,
        solution=_B,
        keep=["path = checkpoint_path(\"best.pt\")", "for epoch in range(30):", "    if val_losses[-1] == min(val_losses):",
              "best = DigitsMLP(64)", _B_END],
        names={"checkpoint_path": fresh_path},
        first=[_DOTS],
        mistakes=[
            Mistake("never-saved", error=lambda e: isinstance(e, FileNotFoundError) and "checkpoints" in str(getattr(e, "filename", "")),
                    en="Python stops at `torch.load(path)`: there is no file at `path`, so nothing was saved. Inside the `if`, save the parameters of the model: `torch.save(model.state_dict(), path)`.",
                    nl="Python stopt bij `torch.load(path)`: er staat geen bestand op `path`, dus er is niets opgeslagen. Sla binnen de `if` de parameters van het model op: `torch.save(model.state_dict(), path)`."),
            Mistake("whole-model", error=lambda e: "Weights only load failed" in str(e) or "weights_only" in str(e),
                    en="Python stops at `torch.load(path)`: the file holds the whole model object, and `torch.load` reads back only tensors and simple values. Save `model.state_dict()`, the parameters by name, not `model` itself.",
                    nl="Python stopt bij `torch.load(path)`: het bestand bevat het hele model-object, en `torch.load` leest alleen tensors en eenvoudige waarden terug. Sla `model.state_dict()` op, de parameters per naam, niet `model` zelf."),
            Mistake("save-order", error=lambda e: isinstance(e, AttributeError) and "expected 'f'" in str(e),
                    en="Python stops at `torch.save`: its second value is not a path. `torch.save` takes what to save first, and then the path: `torch.save(model.state_dict(), path)`.",
                    nl="Python stopt bij `torch.save`: de tweede waarde is geen pad. `torch.save` krijgt eerst wat je opslaat, en dan het pad: `torch.save(model.state_dict(), path)`."),
            Mistake("dict-not-model", error=lambda e: isinstance(e, AttributeError) and "OrderedDict" in str(e),
                    en="Python stops in `val_loss(best)`: `best` is a dictionary of tensors, not a model. `torch.load` gives back the state_dict; give it to the model that is there, with `best.load_state_dict(torch.load(path))`.",
                    nl="Python stopt in `val_loss(best)`: `best` is een dictionary met tensors, geen model. `torch.load` geeft de state_dict terug; geef die aan het model dat er al is, met `best.load_state_dict(torch.load(path))`."),
            Mistake("report", error=lambda e: isinstance(e, AttributeError) and "_IncompatibleKeys" in str(e),
                    en="Python stops in `val_loss(best)`: `best` is no longer a model. `load_state_dict` changes the model itself and gives back only a report of the names it matched. Call it without `best =` in front.",
                    nl="Python stopt in `val_loss(best)`: `best` is geen model meer. `load_state_dict` verandert het model zelf en geeft alleen een verslag terug van de namen die klopten. Roep hem aan zonder `best =` ervoor."),
            Mistake("last-model", matches=lambda got, text: _same_two(got),
                    en="Your last line gives `{got}`: the loaded model has the same validation loss as the model after epoch 30, so it is that model. Is the file written every epoch, outside the `if`? Or do you keep `model.state_dict()` in a name? That dictionary holds the model's own tensors, which training keeps changing. Save to the file inside the `if`, and load the file after the loop.",
                    nl="Je laatste regel geeft `{got}`: het geladen model heeft dezelfde validation loss als het model na epoch 30, dus het is dat model. Wordt het bestand elke epoch geschreven, buiten de `if`? Of bewaar je `model.state_dict()` onder een naam? Die dictionary bevat de eigen tensors van het model, en die blijft het trainen veranderen. Sla binnen de `if` op in het bestand, en laad het bestand na de loop."),
            Mistake("untrained-best", matches=lambda got, text: isinstance(got, list) and len(got) == 2 and float(got[1]) > 2.0,
                    en="Your last line gives `{got}`: `best` has a validation loss of about 2.30, the loss of guessing, as a model that never trained. After the loop, does `best` itself get the parameters from the file?",
                    nl="Je laatste regel geeft `{got}`: `best` heeft een validation loss van ongeveer 2,30, de loss van gokken, zoals een model dat nooit getraind heeft. Krijgt `best` zelf na de loop de parameters uit het bestand?"),
        ],
        digits=4,
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
            "empty": {"en": "Replace the two `...`: inside the `if`, save the parameters of the model to `path`; after the loop, load the file into `best`.",
                      "nl": "Vervang de twee `...`: sla binnen de `if` de parameters van het model op in `path`; laad na de loop het bestand in `best`."},
            "no_value": {"en": "The last line of the box shows nothing. Put the list back on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet de lijst terug op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`: the validation loss of the model after epoch 30 and of the loaded model should be about 0.7589 and 0.6424. Does the `if` save the parameters, and does `best` get them from the file?",
                      "nl": "Je laatste regel geeft `{got}`: de validation loss van het model na epoch 30 en van het geladen model moeten ongeveer 0,7589 en 0,6424 zijn. Slaat de `if` de parameters op, en krijgt `best` ze uit het bestand?"},
            "correct": {"en": "The model after epoch 30 has a validation loss of 0.7589; the loaded one has 0.6424, the lowest of all 30 epochs. The file kept the parameters of epoch 9, while training went on changing the model.",
                        "nl": "Het model na epoch 30 heeft een validation loss van 0,7589; het geladen model 0,6424, de laagste van alle 30 epochs. Het bestand hield de parameters van epoch 9 vast, terwijl het trainen het model bleef veranderen."},
        },
    ),
    "checkpoint": Sort(
        name="checkpoint",
        groups={
            "always": {"en": "In every checkpoint", "nl": "In elk checkpoint"},
            "resume": {"en": "Only to train on later", "nl": "Alleen om later verder te trainen"},
            "none": {"en": "Not in the checkpoint", "nl": "Niet in het checkpoint"},
        },
        items={
            "model": {"en": "`model.state_dict()`", "nl": "`model.state_dict()`"},
            "opt": {"en": "`opt.state_dict()`", "nl": "`opt.state_dict()`"},
            "epoch": {"en": "The number of the epoch", "nl": "Het nummer van de epoch"},
            "class": {"en": "The code of the class `DigitsMLP`", "nl": "De code van de class `DigitsMLP`"},
            "data": {"en": "The training digits", "nl": "De trainingscijfers"},
        },
        solution={"model": "always", "opt": "resume", "epoch": "resume", "class": "none", "data": "none"},
        nudge={
            "model": {"en": "Without what can you not use the trained model at all?",
                      "nl": "Zonder wat kun je het getrainde model helemaal niet gebruiken?"},
            "opt": {"en": "Does anyone who only predicts with the model still take update steps?",
                    "nl": "Doet iemand die alleen met het model voorspelt nog update-stappen?"},
            "epoch": {"en": "When does it matter how many epochs a model has had?",
                      "nl": "Wanneer maakt het uit hoeveel epochs een model al heeft gehad?"},
            "class": {"en": "Where does the code of your model live, and how does a new program get it?",
                      "nl": "Waar staat de code van je model, en hoe krijgt een nieuw programma die?"},
            "data": {"en": "Does the data change when the model trains, and where is it kept?",
                     "nl": "Verandert de data als het model traint, en waar wordt die bewaard?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the lines again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de regels opnieuw uit."},
            "empty": {"en": "Put each thing in a group: in every checkpoint, only to train on later, or not in the checkpoint.",
                      "nl": "Zet elk ding in een groep: in elk checkpoint, alleen om later verder te trainen, of niet in het checkpoint."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs in another group.",
                      "nl": "{item} hoort in een andere groep."},
            "correct": {"en": "All {n} are right. The parameters are always in it; the optimizer and the epoch only when you want to train on; the code and the data stay where they are, in your files.",
                        "nl": "Alle {n} kloppen. De parameters zitten er altijd in; de optimizer en de epoch alleen als je verder wilt trainen; de code en de data blijven waar ze zijn, in je bestanden."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("best-epoch", _B_START, "empty", None),
    ("best-epoch", _b(load="state = torch.load(path)\nbest.load_state_dict(state)\n"), "correct", None),
    ("best-epoch", _b(save="        pass\n"), "error", "never-saved"),
    ("best-epoch", _b(save="        torch.save(model, path)\n"), "error", "whole-model"),
    ("best-epoch", _b(save="        torch.save(path, model.state_dict())\n"), "error", "save-order"),
    ("best-epoch", _b(load="best = torch.load(path)\n"), "error", "dict-not-model"),
    ("best-epoch", _b(load="best = best.load_state_dict(torch.load(path))\n"), "error", "report"),
    ("best-epoch", _b(save="        state = model.state_dict()\n", load="best.load_state_dict(state)\n"), "wrong", "last-model"),
    ("best-epoch", _b(save="        pass\n    torch.save(model.state_dict(), path)\n"), "wrong", "last-model"),
    ("best-epoch", _b(load="model.load_state_dict(torch.load(path))\n"), "wrong", "untrained-best"),
]
