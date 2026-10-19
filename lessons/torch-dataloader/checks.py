# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson torch-dataloader: a `Dataset` for the handwritten digits of QMNIST, with `__len__` and
`__getitem__`, and the training loop of torch-batches with a `DataLoader` that shuffles the examples every epoch.
And the picture: the labels of the first mini-batch of the digits sorted by label, without shuffling and in two
shuffled epochs."""
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch.utils.data import DataLoader, Dataset

from everything_ai.check import CodeBox, Mistake

DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md


class Digits(Dataset):
    """The course's own Dataset of the lesson: each example an image of 784 numbers from 0 to 1, and its digit."""
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32) / 255
        self.labels = torch.tensor(labels, dtype=torch.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.images[i], self.labels[i]


def data():
    """The 200 examples of torch-batches: x between -2 and 2 and y = 2x + 1 plus noise, both of shape (200, 1)."""
    torch.manual_seed(0)
    X = torch.rand(200, 1) * 4 - 2
    y = 2 * X + 1 + 0.5 * torch.randn(200, 1)
    return X, y


def first_labels() -> dict[str, list[int]]:
    """The first 8 labels of the first mini-batch of 64, with the digits sorted by label: without shuffling, and in
    two shuffled epochs after seed 0."""
    data = np.load(DATA)
    order = np.argsort(data["train_labels"], kind="stable")
    ds = Digits(data["train_images"][order], data["train_labels"][order])
    out = {"order": next(iter(DataLoader(ds, batch_size=64)))[1][:8].tolist()}
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    torch.manual_seed(0)
    out["epoch1"] = next(iter(loader))[1][:8].tolist()
    out["epoch2"] = next(iter(loader))[1][:8].tolist()
    return out


def picture(titles: list[str], notes: list[str]) -> list[tuple]:
    """The grids for draw.array_grids: the first 8 labels of the first mini-batch of the sorted digits, without
    shuffling and in two shuffled epochs. `titles` and `notes` are in the lesson's language."""
    labels = first_labels()
    code = ["shuffle=False", "shuffle=True", "shuffle=True"]
    out = []
    for k, key in enumerate(("order", "epoch1", "epoch2")):
        out.append((titles[k], [[str(v) for v in labels[key]]], [code[k], notes[k]]))
    return out


# The lesson is done when these are correct. "reshuffle" is the choice question in the lesson.
REQUIRED = ["reshuffle", "digits", "shuffled"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
    "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
             "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
}

# --- exercise 2: a Dataset for the digits ----------------------------------------------------------------------------

_D_HEAD = ("import numpy as np\nimport torch\nfrom torch.utils.data import Dataset\n\ndata = np.load(\"data/qmnist-small.npz\")\n\n\n"
           "class Digits(Dataset):\n    def __init__(self, images, labels):\n"
           "        self.images = torch.tensor(images, dtype=torch.float32) / 255\n"
           "        self.labels = torch.tensor(labels, dtype=torch.int64)\n\n")
_D_END = ("\n\nds = Digits(data[\"train_images\"], data[\"train_labels\"])\nx, y = ds[7]\n"
          "[len(ds), len(x), y.item(), x.sum().item()]")
_D_START = _D_HEAD + "    def __len__(self):\n        ...\n\n    def __getitem__(self, i):\n        ...\n" + _D_END + "\n"


def _d(length: str = "        return len(self.labels)\n", item: str = "        return self.images[i], self.labels[i]\n") -> str:
    return _D_HEAD + "    def __len__(self):\n" + length + "\n    def __getitem__(self, i):\n" + item + _D_END


_D = _d()

# --- exercise 3: training with a shuffling DataLoader ------------------------------------------------------------------

_S_HEAD = ("import torch\nfrom torch import nn\nfrom torch.utils.data import TensorDataset, DataLoader\n\n"
           "# the data of the lesson before: y = 2x + 1 plus random scatter, shapes (200, 1) and (200, 1)\nX, y = data()\n"
           "torch.manual_seed(1)\nmodel = nn.Linear(1, 1)\nlr = 0.1\nopt = torch.optim.SGD(model.parameters(), lr=lr)\n")
_S_END = "((model(X) - y) ** 2).mean().item()"
_S_START = _S_HEAD + "loader = ...\n\ntorch.manual_seed(2)\nfor epoch in range(3):\n    ...\n" + _S_END + "\n"
_S_LOOP = ("    for xb, yb in loader:\n        loss = ((model(xb) - yb) ** 2).mean()\n        opt.zero_grad()\n"
           "        loss.backward()\n        opt.step()\n")


def _s(loader: str = "DataLoader(TensorDataset(X, y), batch_size=20, shuffle=True)", loop: str = _S_LOOP) -> str:
    return _S_HEAD + f"loader = {loader}\n\ntorch.manual_seed(2)\nfor epoch in range(3):\n" + loop + _S_END


_S = _s()

EXERCISES = {
    "digits": CodeBox(
        name="digits",
        start=_D_START,
        solution=_D,
        keep=["class Digits(Dataset):", "x, y = ds[7]", "[len(ds), len(x), y.item(), x.sum().item()]"],
        first=[_DOTS],
        mistakes=[
            Mistake("one-thing", error=lambda e: (isinstance(e, ValueError) and "too many values to unpack" in str(e))
                    or (isinstance(e, TypeError) and "iteration over a 0-d tensor" in str(e)),
                    en="Python stops at `x, y = ds[7]`: `__getitem__` gives back one thing, and the line expects two, the image and its label. Give back both: `return self.images[i], self.labels[i]`.",
                    nl="Python stopt bij `x, y = ds[7]`: `__getitem__` geeft één ding terug, en de regel verwacht er twee, de afbeelding en zijn label. Geef ze allebei terug: `return self.images[i], self.labels[i]`."),
            Mistake("swapped", error=lambda e: isinstance(e, TypeError) and "len() of a 0-d tensor" in str(e),
                    en="Python stops at `len(x)`: `x` is one number, so `__getitem__` gives back the label first. Give back the image first and then the label: `return self.images[i], self.labels[i]`.",
                    nl="Python stopt bij `len(x)`: `x` is één getal, dus `__getitem__` geeft eerst het label terug. Geef eerst de afbeelding terug en dan het label: `return self.images[i], self.labels[i]`."),
            Mistake("all-labels", error=lambda e: isinstance(e, RuntimeError) and "6000 elements" in str(e),
                    en="Python stops at `y.item()`: `y` holds all 6,000 labels, not the label of example 7. `__getitem__` gives back example `i` only: take `[i]` of the images and of the labels.",
                    nl="Python stopt bij `y.item()`: `y` bevat alle 6.000 labels, niet het label van voorbeeld 7. `__getitem__` geeft alleen voorbeeld `i` terug: neem `[i]` van de afbeeldingen en van de labels."),
            Mistake("no-len", error=lambda e: isinstance(e, TypeError) and "NoneType" in str(e) and "integer" in str(e),
                    en="Python stops at `len(ds)`: `__len__` gives back nothing. Let it give back the number of examples: `return len(self.labels)`.",
                    nl="Python stopt bij `len(ds)`: `__len__` geeft niets terug. Laat hem het aantal voorbeelden teruggeven: `return len(self.labels)`."),
            Mistake("pixels", matches=lambda got, text: isinstance(got, list) and len(got) == 4 and got[0] == 784,
                    en="Your last line gives `{got}`: `len(ds)` is 784, the number of pixels of one image. `__len__` gives the number of examples: the length of `self.labels`, or of `self.images`.",
                    nl="Je laatste regel geeft `{got}`: `len(ds)` is 784, het aantal pixels van één afbeelding. `__len__` geeft het aantal voorbeelden: de lengte van `self.labels`, of van `self.images`."),
        ],
        digits=4,
        texts={
            **_COMMON,
            "empty": {"en": "Replace the two `...`: `__len__` gives back the number of examples, `__getitem__` the image and the label of example `i`.",
                      "nl": "Vervang de twee `...`: `__len__` geeft het aantal voorbeelden terug, `__getitem__` de afbeelding en het label van voorbeeld `i`."},
            "no_value": {"en": "The last line of the box shows nothing. Put the list back on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet de lijst terug op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`: the number of examples, the length of one image, its digit and the sum of its pixels should be 6,000, 784, 3 and about 140.67. Does `__getitem__` give back image `i` and label `i`, in that order?",
                      "nl": "Je laatste regel geeft `{got}`: het aantal voorbeelden, de lengte van één afbeelding, zijn cijfer en de som van zijn pixels moeten 6.000, 784, 3 en ongeveer 140,67 zijn. Geeft `__getitem__` afbeelding `i` en label `i` terug, in die volgorde?"},
            "correct": {"en": "6,000 examples; example 7 is an image of 784 numbers from 0 to 1, of the digit 3. Your `Digits` hands out one example at a time, as a DataLoader asks for them.",
                        "nl": "6.000 voorbeelden; voorbeeld 7 is een afbeelding van 784 getallen van 0 tot 1, van het cijfer 3. Je `Digits` geeft één voorbeeld tegelijk, zoals een DataLoader erom vraagt."},
        },
    ),
    "shuffled": CodeBox(
        name="shuffled",
        start=_S_START,
        solution=_S,
        keep=["X, y = data()", "torch.manual_seed(2)", "for epoch in range(3):", _S_END],
        names={"data": data},   # the lesson's own data, as in the worked example: torch.rand and torch.randn are not in the chain
        others=[{"lr": 0.05}],
        first=[_DOTS],
        mistakes=[
            Mistake("in-order", wrong=_s(loader="DataLoader(TensorDataset(X, y), batch_size=20)"),
                    en="Your last line gives `{got}`: that is the loss with the mini-batches in the same order every epoch, as in the lesson before. Let the DataLoader shuffle: `shuffle=True`.",
                    nl="Je laatste regel geeft `{got}`: dat is de loss met de mini-batches elke epoch in dezelfde volgorde, zoals in de vorige les. Laat de DataLoader shufflen: `shuffle=True`."),
            Mistake("size-one", wrong=_s(loader="DataLoader(TensorDataset(X, y), shuffle=True)"),
                    en="Your last line gives `{got}`: without `batch_size`, a DataLoader makes mini-batches of 1 example, so each step follows one example. Give it `batch_size=20`.",
                    nl="Je laatste regel geeft `{got}`: zonder `batch_size` maakt een DataLoader mini-batches van 1 voorbeeld, dus volgt elke stap één voorbeeld. Geef hem `batch_size=20`."),
            Mistake("full-data", wrong=_s(loop="    for xb, yb in loader:\n        loss = ((model(X) - y) ** 2).mean()\n        opt.zero_grad()\n        loss.backward()\n        opt.step()\n"),
                    en="Your last line gives `{got}`: each step works out the loss over all 200 examples, not over its mini-batch, so the DataLoader only counts the steps. Use `xb` and `yb` in the loss.",
                    nl="Je laatste regel geeft `{got}`: elke stap rekent de loss uit over alle 200 voorbeelden, niet over zijn mini-batch, dus de DataLoader telt alleen de stappen. Gebruik `xb` en `yb` in de loss."),
        ],
        digits=4,
        texts={
            **_COMMON,
            "empty": {"en": "Replace the two `...`: a DataLoader that shuffles, over `TensorDataset(X, y)`, with batches of 20, and in each epoch a loop over its mini-batches with an update step for each.",
                      "nl": "Vervang de twee `...`: een DataLoader die shufflet, over `TensorDataset(X, y)`, met batches van 20, en in elke epoch een loop over zijn mini-batches met een update-stap voor elke."},
            "no_value": {"en": "The last line of the box shows nothing. Put the loss over all data back on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet de loss over alle data terug op de laatste regel, zodat de check het kan zien."},
            "slow": {"en": "Your code is still not done after 2 seconds. Does each epoch go once over the mini-batches of the DataLoader?",
                     "nl": "Je code is na 2 seconden nog niet klaar. Gaat elke epoch één keer over de mini-batches van de DataLoader?"},
            "wrong": {"en": "Your last line gives `{got}`, and it should be about 0.279. Does the DataLoader make mini-batches of 20 and shuffle, and does each mini-batch get its own update step?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet ongeveer 0,279 zijn. Maakt de DataLoader mini-batches van 20 en shufflet hij, en krijgt elke mini-batch zijn eigen update-stap?"},
            "other_error": {"en": "Right for this learning rate. But with {where}, Python stops with an error. Does your code use `opt` itself?",
                            "nl": "Goed voor deze learning rate. Maar met {where} stopt Python met een fout. Gebruikt je code `opt` zelf?"},
            "other": {"en": "Right for this learning rate. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does your step go through `opt`, made with `lr`?",
                      "nl": "Goed voor deze learning rate. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gaat je stap via `opt`, gemaakt met `lr`?"},
            "correct": {"en": "About 0.279 after 3 shuffled epochs, a little lower than the 0.2821 of the batches in order: each epoch mixed the examples differently. Seed 2 makes the shuffles the same on every run.",
                        "nl": "Ongeveer 0,279 na 3 epochs met shuffling, iets lager dan de 0,2821 van de batches op volgorde: elke epoch mengde de voorbeelden anders. Door seed 2 shufflet de DataLoader bij elke run hetzelfde."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("digits", _D_START, "empty", None),
    ("digits", _d(length="        return len(self.images)\n"), "correct", None),
    ("digits", _d(item="        image = self.images[i]\n        label = self.labels[i]\n        return image, label\n"), "correct", None),
    ("digits", _d(item="        return self.images[i]\n"), "error", "one-thing"),
    ("digits", _d(item="        return self.labels[i]\n"), "error", "one-thing"),
    ("digits", _d(item="        return self.labels[i], self.images[i]\n"), "error", "swapped"),
    ("digits", _d(item="        return self.images, self.labels\n"), "error", "all-labels"),
    ("digits", _d(length="        len(self.labels)\n"), "error", "no-len"),
    ("digits", _d(length="        return len(self.images[0])\n"), "wrong", "pixels"),
    ("shuffled", _S_START, "empty", None),
    ("shuffled", _s(loader="DataLoader(TensorDataset(X, y), shuffle=True, batch_size=20)"), "correct", None),
    ("shuffled", _s(loader="DataLoader(TensorDataset(X, y), batch_size=20)"), "wrong", "in-order"),
    ("shuffled", _s(loader="DataLoader(TensorDataset(X, y), shuffle=True)"), "wrong", "size-one"),
    ("shuffled", _s(loop="    for xb, yb in loader:\n        loss = ((model(X) - y) ** 2).mean()\n        opt.zero_grad()\n        loss.backward()\n        opt.step()\n"), "wrong", "full-data"),
]
