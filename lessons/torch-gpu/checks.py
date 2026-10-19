# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson torch-gpu: a predict function that moves the model and the data to a device, tried on the CPU
and on PyTorch's "meta" device, which stands in for a GPU: it has no numbers, only shapes, and stops just like a GPU
when a tensor is on another device. With the learner's own DigitsMLP from torch-mlp-digits, or the course's where
theirs is missing or broken. And which things move to the device."""
import re

import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn

from everything_ai import projects
from everything_ai.check import ClassBox, Mistake, Sort

# The module of the lesson before: yours when it is there and gives the course's results, else the course's
# (template 1.3).
projects.builds_on(("nn", "torch_mlp", "torch-mlp-digits", "digits-mlp"))


# The lesson is done when these are correct. "no-gpu" is the choice question in the lesson.
REQUIRED = ["no-gpu", "predict-device", "what-moves"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# --- exercise 2: predict on a device ------------------------------------------------------------------------------------

_P_HEAD = "import torch\nfrom nn.torch_mlp import DigitsMLP\n\n\ndef predict(model, X, device):\n"
_P_MODEL = "    model.to(device)\n"
_P_MID = "    model.eval()\n    with torch.no_grad():\n"
_P_PRED = "        pred = model(X.to(device)).argmax(dim=1)\n"
_P_END = "    model.train()\n    return pred\n"
_P_START = _P_HEAD + "    ...\n" + _P_MID + "        pred = model(...).argmax(dim=1)\n" + _P_END


def _p(model: str = _P_MODEL, pred: str = _P_PRED) -> str:
    return _P_HEAD + model + _P_MID + pred + _P_END


_P = _p()
_DIGITS = ("import numpy as np\ndata = np.load(\"data/qmnist-small.npz\")\n"
           "X = torch.tensor(data[\"test_images\"][:5], dtype=torch.float32) / 255\ntorch.manual_seed(0)\nmodel = DigitsMLP(16)\n")
_P_SCENARIOS = [
    _DIGITS + "predict(model, X, \"cpu\").tolist()",
    _DIGITS + "pred = predict(model, X, \"meta\")\n[pred.device.type, list(pred.shape), model.hidden.weight.device.type]",
]

EXERCISES = {
    "predict-device": ClassBox(
        name="predict-device",
        start=_P_START,
        solution=_P,
        scenarios=_P_SCENARIOS,
        digits=2,   # logits, when the argmax is gone: no float noise in the message
        first=[_DOTS],
        mistakes=[
            Mistake("data-stays", syntax=r"\A(?![\s\S]*\bX\s*\.\s*to\s*\()",
                    en="Python stops with an error, and your code does not move `X` with `X.to(device)`. `X` starts on the CPU, and the tensors of one calculation must be on the same device: once the model is on `\"meta\"`, `X` must go there too. Move the data too: `X.to(device)`.",
                    nl="Python stopt met een fout, en je code verplaatst `X` niet met `X.to(device)`. `X` begint op de CPU, en de tensors van één berekening moeten op hetzelfde device staan: staat het model op `\"meta\"`, dan moet `X` daar ook heen. Verplaats de data ook: `X.to(device)`."),
            Mistake("copy-dropped", syntax=r"^[ \t]*X\s*\.\s*to\s*\(.*\)[ \t]*$",
                    en="Python stops with an error. Your line `{line}` makes a moved copy of `X` and throws it away: `.to(device)` gives a copy back, and leaves `X` itself where it was. Keep the copy: `X = X.to(device)`, or give the model `X.to(device)` directly.",
                    nl="Python stopt met een fout. Je regel `{line}` maakt een verplaatste kopie van `X` en gooit die weg: `.to(device)` geeft een kopie terug, en laat `X` zelf staan waar hij stond. Bewaar de kopie: `X = X.to(device)`, of geef het model meteen `X.to(device)`."),
            Mistake("model-stays", syntax=r"\A(?![\s\S]*\bmodel\s*\.\s*to\s*\()",
                    en="Python stops with an error, and your code does not move the model with `model.to(device)`. Its parameters start on the CPU, and the tensors of one calculation must be on the same device: once `X` is on `\"meta\"`, the parameters must go there too. Move the model too: `model.to(device)`.",
                    nl="Python stopt met een fout, en je code verplaatst het model niet met `model.to(device)`. Zijn parameters beginnen op de CPU, en de tensors van één berekening moeten op hetzelfde device staan: staat `X` op `\"meta\"`, dan moeten de parameters daar ook heen. Verplaats het model ook: `model.to(device)`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the two `...`: first move the model to `device`, then give it `X`, moved to `device` as well.",
                      "nl": "Vervang de twee `...`: verplaats eerst het model naar `device`, en geef het dan `X`, ook verplaatst naar `device`."},
            "wrong": {"en": "With these lines after your function:\n\n```python\n{scenario}\n```\n\nyour `predict` gives `{got}`, and it should be `{want}`. Does it give back the digit with the largest logit for every image, as the lines after `with` do?",
                      "nl": "Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\ngeeft je `predict` `{got}`, en dat moet `{want}` zijn. Geeft hij voor elke afbeelding het cijfer met de grootste logit terug, zoals de regels na `with` doen?"},
            "almost": {"en": "Right on the CPU. But with these lines after your function:\n\n```python\n{scenario}\n```\n\nyour `predict` gives `{got}`, and it should be `{want}`. Do the model and the data both go to `device`, and does the prediction stay there?",
                       "nl": "Goed op de CPU. Maar met deze regels na je functie:\n\n```python\n{scenario}\n```\n\ngeeft je `predict` `{got}`, en dat moet `{want}` zijn. Gaan het model en de data allebei naar `device`, en blijft de voorspelling daar?"},
            "correct": {"en": "On the CPU, the same five digits as the course's model gives, and on the \"meta\" device five predictions that stay there, without an error: the model and the data were on the same device. On a GPU, `\"cuda\"` or `\"mps\"`, the same function works.",
                        "nl": "Op de CPU dezelfde vijf cijfers als het model van de course geeft, en op het \"meta\"-device vijf voorspellingen die daar blijven, zonder fout: het model en de data stonden op hetzelfde device. Op een GPU, `\"cuda\"` of `\"mps\"`, werkt dezelfde functie."},
        },
    ),
    "what-moves": Sort(
        name="what-moves",
        groups={
            "moves": {"en": "Moves to the device", "nl": "Gaat naar het device"},
            "stays": {"en": "Needs no move", "nl": "Hoeft niet te verhuizen"},
        },
        items={
            "model": {"en": "The model, `model`", "nl": "Het model, `model`"},
            "batch": {"en": "Each mini-batch, `xb` and `yb`", "nl": "Elke mini-batch, `xb` en `yb`"},
            "lr": {"en": "The learning rate, `lr = 0.5`", "nl": "De learning rate, `lr = 0.5`"},
            "item": {"en": "The number `loss.item()`", "nl": "Het getal `loss.item()`"},
            "epochs": {"en": "The number of epochs", "nl": "Het aantal epochs"},
        },
        solution={"model": "moves", "batch": "moves", "lr": "stays", "item": "stays", "epochs": "stays"},
        nudge={
            "model": {"en": "Where must the parameters be to be multiplied with the data?",
                      "nl": "Waar moeten de parameters staan om met de data vermenigvuldigd te worden?"},
            "batch": {"en": "The model is on the device. Where must the numbers it gets be?",
                      "nl": "Het model staat op het device. Waar moeten de getallen staan die het krijgt?"},
            "lr": {"en": "Is a learning rate a tensor, or a plain Python number?",
                   "nl": "Is een learning rate een tensor, of een gewoon getal van Python?"},
            "item": {"en": "What does `.item()` give back: a tensor, or a plain Python number?",
                     "nl": "Wat geeft `.item()` terug: een tensor, of een gewoon getal van Python?"},
            "epochs": {"en": "Is the number of epochs a tensor that the GPU calculates with, or a plain Python number that only counts the rounds of the loop?",
                       "nl": "Is het aantal epochs een tensor waarmee de GPU rekent, of een gewoon getal van Python dat alleen de rondes van de loop telt?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the things again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de dingen opnieuw uit."},
            "empty": {"en": "Put each thing in a group: moves to the device, or needs no move.",
                      "nl": "Zet elk ding in een groep: gaat naar het device, of hoeft niet te verhuizen."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs in the other group.",
                      "nl": "{item} hoort in de andere groep."},
            "correct": {"en": "All {n} are right. The tensors that are calculated with, the parameters and the data, go to the device; plain Python numbers stay where they are, and `.item()` already brings a result back.",
                        "nl": "Alle {n} kloppen. De tensors waarmee gerekend wordt, de parameters en de data, gaan naar het device; gewone getallen van Python blijven waar ze zijn, en `.item()` haalt een resultaat al terug."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("predict-device", _P_START, "empty", None),
    ("predict-device", _p(model="    model = model.to(device)\n", pred="        X = X.to(device)\n        pred = model(X).argmax(dim=1)\n"), "correct", None),
    ("predict-device", _p(pred="        pred = model(X).argmax(dim=1)\n"), "error", "data-stays"),
    ("predict-device", _p(model="    pass\n"), "error", "model-stays"),
    ("predict-device", _p(model="    model.to(device)\n    X.to(device)\n", pred="        pred = model(X).argmax(dim=1)\n"), "error", "copy-dropped"),
    # moving only the result: on the "meta" device the model stayed on the CPU
    ("predict-device", _p(model="    pass\n", pred="        pred = model(X).argmax(dim=1).to(device)\n"), "almost", None),
    ("predict-device", _p(model="    pass\n", pred="        pred = model(X).argmax(dim=1)\n"), "almost", None),
]
