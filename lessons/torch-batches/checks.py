# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson torch-batches: a generator `batches(X, y, size)` that hands out the data in mini-batches,
and a training loop over epochs and mini-batches with an optimizer. And the picture: a line through 200 noisy
points, trained with mini-batches of 200, 20 and 2 examples."""
import functools
import re

import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn

from everything_ai.check import CodeBox, Code, Mistake

# --- the data and the picture: y = 2x + 1 with noise, trained with three batch sizes ------------------------------

N, EPOCHS, LR = 200, 5, 0.1
SIZES = (200, 20, 2)


def data():
    """200 examples with x between -2 and 2 and y = 2x + 1 plus noise, both of shape (200, 1); the same every time."""
    torch.manual_seed(0)
    X = torch.rand(N, 1) * 4 - 2
    y = 2 * X + 1 + 0.5 * torch.randn(N, 1)
    return X, y


def batches(X, y, size):
    """The course's own generator: the examples in order, `size` at a time; the last batch may be smaller."""
    for start in range(0, len(X), size):
        yield X[start:start + size], y[start:start + size]


@functools.lru_cache(maxsize=1)
def runs():
    """For each batch size: the loss over all data before training and after every step, and the loss of every
    mini-batch. Worked out once. After every step, not only after each epoch: with the batches always in the same
    order, the loss after an epoch of batches of 2 is the same every epoch, and hides how it jumps in between."""
    X, y = data()
    out = {}
    for size in SIZES:
        torch.manual_seed(1)
        model = nn.Linear(1, 1)
        opt = torch.optim.SGD(model.parameters(), lr=LR)
        with torch.no_grad():
            full = [((model(X) - y) ** 2).mean().item()]
        steps = []
        for _ in range(EPOCHS):
            for xb, yb in batches(X, y, size):
                loss = ((model(xb) - yb) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                steps.append(loss.item())
                with torch.no_grad():
                    full.append(((model(X) - y) ** 2).mean().item())
        out[size] = (full, steps)
    return out


def picture(labels: dict[str, str]) -> str:
    """Left: the loss over all data after every step, for each batch size, with the epochs on the x-axis; all three
    start at 4.9, above the top. Right: the loss of each of the first 30 mini-batches of 20 examples, three epochs."""
    from everything_ai import chart
    r = runs()
    # a plain axis from 0 to 2 (a log scale is not in the chain), so that the jumps of B = 2 between 0.28 and 0.52 show
    left = chart.Panel((0, EPOCHS), (0, 2), title=labels["epochs"], xlabel=labels["epoch"], ylabel=labels["loss"])
    for size, width, dashed in ((2, 1.0, False), (20, 2.6, False), (200, 2, True)):
        full = r[size][0]
        per_epoch = len(range(0, N, size))
        left.line([k / per_epoch for k in range(len(full))], full, "loss", width, dashed)
    left.points(range(1, EPOCHS + 1), r[200][0][1:], "loss", 3.2)   # the one step per epoch of the full batch
    left.text(4.25, 1.32, "B = 200", "ink")
    left.text(0.6, 1.32, "B = 20", "ink", anchor="start")
    left.text(2.0, 0.8, "B = 2", "ink")
    steps20 = r[20][1][:30]
    right = chart.Panel((0, 30), (0, 5), title=labels["steps"], xlabel=labels["step"], ylabel=labels["loss"])
    right.line(range(1, 31), steps20, "loss", 1.6)
    right.points(range(1, 31), steps20, "loss", 3)
    return chart.figure([left, right], alt=labels["alt"], width=330, height=260)


# The lesson is done when these are correct. "last-batch" is the choice question in the lesson.
REQUIRED = ["last-batch", "batches", "epochs"]

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

# --- exercise 2: the generator ------------------------------------------------------------------------------------------

_B_DEF = "import torch\n\n\ndef batches(X, y, size):\n"
_B_END = ("\n\nX = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]])\n"
          "y = torch.tensor([[0.0], [10.0], [20.0], [30.0], [40.0], [50.0], [60.0], [70.0], [80.0], [90.0]])\n"
          "size = 4\nseen = []\nfor xb, yb in batches(X, y, size):\n    seen.append([len(xb), xb[0].item(), yb[0].item()])\nseen")
_B_START = _B_DEF + "    ...\n" + _B_END + "\n"


def _b(body: str = "    for start in range(0, len(X), size):\n        yield X[start:start + size], y[start:start + size]\n") -> str:
    return _B_DEF + body + _B_END


_B = _b()

# --- exercise 3: epochs and mini-batches ------------------------------------------------------------------------------------

_E_HEAD = ("import torch\nfrom torch import nn\n\n\ndef batches(X, y, size):\n    for start in range(0, len(X), size):\n"
           "        yield X[start:start + size], y[start:start + size]\n\n\n"
           "# the data of the picture: y = 2x + 1 plus random scatter, shapes (200, 1) and (200, 1)\nX, y = data()\n"
           "torch.manual_seed(1)\nmodel = nn.Linear(1, 1)\nlr = 0.1\nopt = torch.optim.SGD(model.parameters(), lr=lr)\n\n"
           "for epoch in range(3):\n    for xb, yb in batches(X, y, 20):\n")
_E_END = "((model(X) - y) ** 2).mean().item()"
_E_START = _E_HEAD + "        ...\n" + _E_END + "\n"


def _e(body: str = "        loss = ((model(xb) - yb) ** 2).mean()\n        opt.zero_grad()\n        loss.backward()\n        opt.step()\n") -> str:
    return _E_HEAD + body + _E_END


_E = _e()

EXERCISES = {
    "batches": CodeBox(
        name="batches",
        start=_B_START,
        solution=_B,
        keep=["def batches(X, y, size):", "for xb, yb in batches(X, y, size):"],
        others=[{"size": 3}],
        form=[Mistake("no-yield", state="almost", matches=lambda got, text: re.search(r"\byield\b", text) is None,
                      en="Right batches, but your function builds them all first and gives them back at once. A generator hands them out one at a time, with `yield`: then the training loop can use each batch as soon as it is ready.",
                      nl="Goede batches, maar je functie maakt ze eerst allemaal en geeft ze in één keer terug. Een generator geeft ze één voor één af, met `yield`: dan kan de training loop elke batch gebruiken zodra hij klaar is.")],
        first=[_DOTS],
        mistakes=[
            Mistake("one-tensor", syntax=r"^\s*yield\s+X\[[^\]]*\]\s*$",
                    en="Python stops: `for xb, yb in ...` expects two things from each batch, the inputs and the labels, and `batches` hands out only `X`. Hand out both: `yield X[...], y[...]`.",
                    nl="Python stopt: `for xb, yb in ...` verwacht van elke batch twee dingen, de inputs en de labels, en `batches` geeft alleen `X`. Geef ze allebei: `yield X[...], y[...]`."),
            Mistake("return", syntax=r"^\s*return\b",
                    en="Python stops: `return` gives back one batch and ends the function, so the `for` loop gets the two halves of one batch instead of batch after batch. Hand out each batch with `yield`.",
                    nl="Python stopt: `return` geeft één batch terug en beëindigt de functie, dus krijgt de `for`-loop de twee helften van één batch in plaats van batch na batch. Geef elke batch met `yield`."),
            Mistake("no-last", wrong=_b("    for start in range(0, len(X) - size + 1, size):\n        yield X[start:start + size], y[start:start + size]\n"),
                    en="Your last line gives `{got}`: the last examples are missing. 10 examples in batches of 4 make two full batches and a last one of 2. A slice past the end simply stops at the end, so let `range` run to `len(X)`.",
                    nl="Je laatste regel geeft `{got}`: de laatste voorbeelden ontbreken. 10 voorbeelden in batches van 4 zijn twee volle batches en een laatste van 2. Een slice voorbij het eind stopt gewoon bij het eind, dus laat `range` lopen tot `len(X)`."),
            Mistake("every-start", wrong=_b("    for start in range(len(X)):\n        yield X[start:start + size], y[start:start + size]\n"),
                    en="Your last line gives `{got}`: a batch starts at every example, so the batches overlap and each example comes back several times. Let the start jump by the batch size: `range(0, len(X), size)`.",
                    nl="Je laatste regel geeft `{got}`: bij elk voorbeeld begint een batch, dus de batches overlappen en elk voorbeeld komt meerdere keren terug. Laat het begin springen met de batch size: `range(0, len(X), size)`."),
            Mistake("same-y", wrong=_b("    for start in range(0, len(X), size):\n        yield X[start:start + size], y\n"),
                    en="Your last line gives `{got}`: every batch gets all the labels, starting at the first. Slice `y` the same way as `X`, so that each batch keeps its own labels.",
                    nl="Je laatste regel geeft `{got}`: elke batch krijgt alle labels, vanaf het eerste. Snij `y` op dezelfde manier als `X`, zodat elke batch zijn eigen labels houdt."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "Replace the `...` with a loop over the start of each batch, which hands out a slice of `X` and the same slice of `y` with `yield`.",
                      "nl": "Vervang de `...` door een loop over het begin van elke batch, die met `yield` een slice van `X` en dezelfde slice van `y` geeft."},
            "no_value": {"en": "The last line of the box shows nothing. Put `seen` back on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `seen` terug op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be `[[4, 0.0, 0.0], [4, 4.0, 40.0], [2, 8.0, 80.0]]`: three batches, of 4, 4 and 2 examples, each starting with its own first example and label. Which of your batches differs from these, and in what?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet `[[4, 0.0, 0.0], [4, 4.0, 40.0], [2, 8.0, 80.0]]` zijn: drie batches, van 4, 4 en 2 voorbeelden, die elk met hun eigen eerste voorbeeld en label beginnen. Welke van je batches wijkt hiervan af, en waarin?"},
            "other_error": {"en": "Right for batches of 4. But with {where}, Python stops with an error. Does `batches` use `size` itself?",
                            "nl": "Goed voor batches van 4. Maar met {where} stopt Python met een fout. Gebruikt `batches` `size` zelf?"},
            "other": {"en": "Right for batches of 4. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does `batches` use `size` itself, rather than the number 4?",
                      "nl": "Goed voor batches van 4. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruikt `batches` `size` zelf, en niet het getal 4?"},
            "correct": {"en": "Batches of 4, 4 and 2, each with its own labels; with batches of 3, the check got 3, 3, 3 and 1. Every example comes out once.",
                        "nl": "Batches van 4, 4 en 2, elk met zijn eigen labels; met batches van 3 kreeg de check 3, 3, 3 en 1. Elk voorbeeld komt één keer langs."},
        },
    ),
    "epochs": CodeBox(
        name="epochs",
        start=_E_START,
        solution=_E,
        keep=["X, y = data()", "torch.manual_seed(1)", "for epoch in range(3):", "    for xb, yb in batches(X, y, 20):", _E_END],
        names={"data": data},   # the lesson's own data, as in the worked example: torch.rand and torch.randn are not in the chain
        others=[{"lr": 0.05}],
        first=[_DOTS],
        mistakes=[
            Mistake("full-data", wrong=_e("        loss = ((model(X) - y) ** 2).mean()\n        opt.zero_grad()\n        loss.backward()\n        opt.step()\n"),
                    en="Your last line gives `{got}`: each step works out the loss over all 200 examples, not over its mini-batch. Here that ends even a little lower than with mini-batches, because it takes 30 steps over all the data: 10 times the work. Use `xb` and `yb` in the loss.",
                    nl="Je laatste regel geeft `{got}`: elke stap rekent de loss uit over alle 200 voorbeelden, niet over zijn mini-batch. Hier eindigt dat zelfs iets lager dan met mini-batches, omdat het 30 stappen over alle data zet: 10 keer zoveel werk. Gebruik `xb` en `yb` in de loss."),
            Mistake("no-zero", wrong=_e("        loss = ((model(xb) - yb) ** 2).mean()\n        loss.backward()\n        opt.step()\n"),
                    en="Your last line gives `{got}`: the slopes are never thrown away, so each step also moves by the slopes of all the batches before it. Call `opt.zero_grad()` before `loss.backward()`.",
                    nl="Je laatste regel geeft `{got}`: de hellingen worden nooit weggegooid, dus elke stap schuift ook met de hellingen van alle batches ervoor. Roep `opt.zero_grad()` aan vóór `loss.backward()`."),
            # zero_grad between backward and step, no backward, or no step: nothing moves, the loss stays where it began
            Mistake("nothing-moved", wrong=_e("        loss = ((model(xb) - yb) ** 2).mean()\n        loss.backward()\n        opt.zero_grad()\n        opt.step()\n"),
                    en="Your last line gives `{got}`, the same as before training: no parameter moved. Each mini-batch needs `opt.zero_grad()`, then `loss.backward()`, then `opt.step()`, in that order: `opt.step()` only moves what `loss.backward()` has just given a slope.",
                    nl="Je laatste regel geeft `{got}`, hetzelfde als vóór het trainen: geen enkele parameter is verschoven. Elke mini-batch heeft `opt.zero_grad()`, dan `loss.backward()` en dan `opt.step()` nodig, in die volgorde: `opt.step()` schuift alleen wat `loss.backward()` net een helling heeft gegeven."),
            # the predictions of one set of examples against the labels of another: 200 against 20 rows
            Mistake("mixed-batch", error=lambda e: isinstance(e, RuntimeError) and "must match the size" in str(e),
                    en="Python stops: the loss puts the predictions for one group of examples against the labels of another, and the number of rows does not match. Use the mini-batch on both sides: `model(xb)` and `yb`.",
                    nl="Python stopt: de loss zet de voorspellingen voor de ene groep voorbeelden tegenover de labels van een andere, en het aantal rijen klopt niet. Gebruik aan beide kanten de mini-batch: `model(xb)` en `yb`."),
        ],
        digits=4,
        texts={
            **_COMMON,
            "empty": {"en": "Replace the `...` with one step for each mini-batch: its loss, `opt.zero_grad()`, `loss.backward()` and `opt.step()`.",
                      "nl": "Vervang de `...` door één stap voor elke mini-batch: zijn loss, `opt.zero_grad()`, `loss.backward()` en `opt.step()`."},
            "no_value": {"en": "The last line of the box shows nothing. Put the loss over all data back on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet de loss over alle data terug op de laatste regel, zodat de check het kan zien."},
            "slow": {"en": "Your code is still not done after 2 seconds. Does each mini-batch take one step, 30 steps in all?",
                     "nl": "Je code is na 2 seconden nog niet klaar. Zet elke mini-batch één stap, 30 stappen in totaal?"},
            "wrong": {"en": "Your last line gives `{got}`, and it should be about 0.2821. Does each mini-batch get its own step: the loss of `xb` and `yb`, then the three calls of the optimizer?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet ongeveer 0,2821 zijn. Krijgt elke mini-batch zijn eigen stap: de loss van `xb` en `yb`, en dan de drie aanroepen van de optimizer?"},
            "other_error": {"en": "Right for this learning rate. But with {where}, Python stops with an error. Does your code use `opt` itself?",
                            "nl": "Goed voor deze learning rate. Maar met {where} stopt Python met een fout. Gebruikt je code `opt` zelf?"},
            "other": {"en": "Right for this learning rate. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does your step go through `opt`, made with `lr`?",
                      "nl": "Goed voor deze learning rate. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gaat je stap via `opt`, gemaakt met `lr`?"},
            "correct": {"en": "About 0.2821 after 3 epochs of 10 mini-batches each: 30 steps, and the line is nearly as close as the random scatter in the data allows. With all 200 examples in each step, as in the lesson before, 3 epochs are 3 steps, and the loss would still be about 1.33.",
                        "nl": "Ongeveer 0,2821 na 3 epochs van elk 10 mini-batches: 30 stappen, en de lijn zit bijna zo dichtbij als de willekeurige spreiding in de data toelaat. Met alle 200 voorbeelden in elke stap, zoals in de vorige les, zijn 3 epochs 3 stappen, en zou de loss nog ongeveer 1,33 zijn."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("batches", _B_START, "empty", None),
    ("batches", _b("    out = []\n    for start in range(0, len(X), size):\n        out.append((X[start:start + size], y[start:start + size]))\n    return out\n"), "almost", "no-yield"),
    ("batches", _b("    start = 0\n    while start < len(X):\n        yield X[start:start + size], y[start:start + size]\n        start = start + size\n"), "correct", None),
    ("batches", _b("    for start in range(0, len(X), size):\n        end = start + size\n        yield X[start:end], y[start:end]\n"), "correct", None),
    ("batches", _b("    for start in range(0, len(X), size):\n        yield X[start:start + size]\n"), "error", "one-tensor"),
    ("batches", _b("    for start in range(0, len(X), size):\n        return X[start:start + size], y[start:start + size]\n"), "error", "return"),
    ("batches", _b("    for start in range(0, len(X) - size + 1, size):\n        yield X[start:start + size], y[start:start + size]\n"), "wrong", "no-last"),
    ("batches", _b("    for start in range(len(X)):\n        yield X[start:start + size], y[start:start + size]\n"), "wrong", "every-start"),
    ("batches", _b("    for start in range(0, len(X), size):\n        yield X[start:start + size], y\n"), "wrong", "same-y"),
    ("batches", _b("    for start in range(0, len(X), 4):\n        yield X[start:start + 4], y[start:start + 4]\n"), "almost", "other-value"),
    ("epochs", _E_START, "empty", None),
    ("epochs", _e("        opt.zero_grad()\n        loss = ((model(xb) - yb) ** 2).mean()\n        loss.backward()\n        opt.step()\n"), "correct", None),
    ("epochs", _e("        loss = ((model(X) - y) ** 2).mean()\n        opt.zero_grad()\n        loss.backward()\n        opt.step()\n"), "wrong", "full-data"),
    ("epochs", _e("        loss = ((model(xb) - yb) ** 2).mean()\n        loss.backward()\n        opt.step()\n"), "wrong", "no-zero"),
    ("epochs", _e("        loss = ((model(xb) - yb) ** 2).mean()\n        opt.zero_grad()\n        loss.backward()\n"), "wrong", "nothing-moved"),
    ("epochs", _e("        loss = ((model(X) - yb) ** 2).mean()\n        opt.zero_grad()\n        loss.backward()\n        opt.step()\n"), "error", "mixed-batch"),
    ("epochs", _e("        pred = model(xb)\n        loss = torch.mean((pred - yb) ** 2)\n        opt.zero_grad()\n        loss.backward()\n        opt.step()\n"), "correct", None),
]
