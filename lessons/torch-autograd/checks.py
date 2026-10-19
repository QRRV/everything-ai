# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson torch-autograd: the slopes of the neuron of nn-micrograd-backward, now with PyTorch's
autograd, and a training loop for a linear model with `.grad = None`, `loss.backward()` and a step inside
`torch.no_grad()`. And the picture: the graph PyTorch records for d = a · b + a, with its slopes."""
import re

import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import draw
from everything_ai.check import CodeBox, Code, Mistake

# --- the picture: d = a * b + a, as PyTorch records it ------------------------------------------------------------


def picture(labels: dict[str, str]):
    """The graph of d = a · b + a with a = 2 and b = 3, with the local gradient on each arrow and the slope of d with
    respect to every value, as `d.backward()` works them out. Only a and b keep theirs in `.grad`."""
    nodes = [("a", 0, 0, "a", "2", "4", "param"), ("b", 0, 1.25, "b", "3", "2", "param"),
             ("c", 1, 1, "c = a · b", "6", "1", None), ("d", 2, 0, "d = c + a", "8", "1", None)]
    edges = [("a", "c", "3"), ("b", "c", "2"), ("c", "d", "1"), ("a", "d", "1")]
    return draw.computational_graph(nodes, edges, title=labels["title"], alt=labels["alt"])


# The lesson is done when these are correct. "step" is the choice question in the lesson.
REQUIRED = ["neuron", "step", "train"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TWICE = Mistake("twice", error=lambda e: isinstance(e, RuntimeError) and "backward through the graph a second time" in str(e),
                 en="PyTorch stops: `backward()` ran twice on the same graph. After a backward pass, PyTorch throws away what it remembered of the forward pass, to save memory. Call `loss.backward()` once, after a new forward pass.",
                 nl="PyTorch stopt: `backward()` liep twee keer over dezelfde graaf. Na een backward pass gooit PyTorch weg wat hij van de forward pass onthield, om geheugen te sparen. Roep `loss.backward()` één keer aan, na een nieuwe forward pass.")
_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
    "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
             "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
}

# --- exercise 1: the neuron of nn-micrograd-backward, with autograd -------------------------------------------------

_N_HEAD = ("import torch\n\nx = torch.tensor([2.0, 1.0])\ny = torch.tensor(3.0)\n"
           "w = torch.tensor([0.5, -1.0], requires_grad=True)\nb = torch.tensor(1.0, requires_grad=True)\n\n")
_N_END = "[w.grad, b.grad]"
_N_START = _N_HEAD + "...\n" + _N_END + "\n"


def _n(body: str) -> str:
    return _N_HEAD + body + _N_END


_N = _n("z = w @ x + b\ny_pred = torch.relu(z)\nloss = (y_pred - y) ** 2\nloss.backward()\n")

# --- exercise 3: a training loop with autograd ------------------------------------------------------------------------

_T_HEAD = ("import torch\n\nX = torch.tensor([[1.0, 2.0], [3.0, 1.0], [0.0, 2.0], [2.0, 2.0]])\n"
           "y = torch.tensor([5.0, 4.0, 3.0, 6.0])\nw = torch.tensor([0.0, 0.0], requires_grad=True)\n"
           "b = torch.tensor(0.0, requires_grad=True)\nlr = 0.05\n\n"
           "for step in range(100):\n    loss = ((X @ w + b - y) ** 2).mean()\n")
_T_END = "loss.item()"
_T_ZERO = "    w.grad = None\n    b.grad = None\n"
_T_BACK = "    loss.backward()\n"
_T_STEP = "    with torch.no_grad():\n        w -= lr * w.grad\n        b -= lr * b.grad\n"
_T_START = _T_HEAD + "    ...\n" + _T_END + "\n"


def _t(zero: str = _T_ZERO, back: str = _T_BACK, step: str = _T_STEP) -> str:
    return _T_HEAD + zero + back + step + _T_END


_T = _t()

EXERCISES = {
    "neuron": CodeBox(
        name="neuron",
        start=_N_START,
        solution=_N,
        keep=["x = torch.tensor([2.0, 1.0])", "y = torch.tensor(3.0)", _N_END],
        # z = -0.5, below 0, where ReLU gives 0 and every slope is 0; and another bias term, so a typed number shows
        others=[{"w": Code("torch.tensor([-1.0, 0.5], requires_grad=True)")}, {"b": Code("torch.tensor(2.0, requires_grad=True)")}],
        first=[_DOTS],
        mistakes=[
            _TWICE,
            Mistake("not-one-number", error=lambda e: isinstance(e, RuntimeError) and "only for scalar outputs" in str(e),
                    en="PyTorch stops at `loss.backward()`: it needs a loss with one number in it, and yours holds more than one. Does your forward pass use the dot product `w @ x`? `w * x` multiplies number by number, without adding up.",
                    nl="PyTorch stopt bij `loss.backward()`: die heeft een loss met één getal erin nodig, en de jouwe bevat er meer. Gebruikt je forward pass het inproduct `w @ x`? `w * x` vermenigvuldigt getal voor getal, zonder op te tellen."),
            Mistake("no-backward", matches=lambda got, text: isinstance(got, list) and all(g is None for g in got),
                    en="Your last line gives `{got}`: there is no slope yet. `.grad` stays `None` until `loss.backward()` has run. Call it after the forward pass.",
                    nl="Je laatste regel geeft `{got}`: er is nog geen helling. `.grad` blijft `None` tot `loss.backward()` gelopen heeft. Roep hem aan na de forward pass."),
            Mistake("no-bias", matches=lambda got, text: isinstance(got, list) and len(got) == 2 and got[0] is not None and got[1] is None,
                    en="Your last line gives `{got}`: `b.grad` is `None`, so `b` is not in the graph of your loss. Add the bias term: `z = w @ x + b`.",
                    nl="Je laatste regel geeft `{got}`: `b.grad` is `None`, dus `b` zit niet in de graaf van je loss. Tel de bias term erbij op: `z = w @ x + b`."),
            Mistake("no-square", wrong=_n("z = w @ x + b\ny_pred = torch.relu(z)\nloss = y_pred - y\nloss.backward()\n"),
                    en="Your last line gives `{got}`: the slopes of ŷ itself, not of the loss. `backward()` ran on ŷ − y, or on ŷ, not on the square of ŷ − y. Make `loss = (y_pred - y) ** 2`, and call `loss.backward()` on that.",
                    nl="Je laatste regel geeft `{got}`: de hellingen van ŷ zelf, niet van de loss. `backward()` liep over ŷ − y, of over ŷ, niet over het kwadraat van ŷ − y. Maak `loss = (y_pred - y) ** 2`, en roep daarop `loss.backward()` aan."),
        ],
        digits=4,
        texts={
            **_COMMON,
            "empty": {"en": "Replace the `...` with the forward pass, ending in `loss`, and then `loss.backward()`.",
                      "nl": "Vervang de `...` door de forward pass, die eindigt in `loss`, en dan `loss.backward()`."},
            "no_value": {"en": "The last line of the box shows nothing. Put `[w.grad, b.grad]` back on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `[w.grad, b.grad]` terug op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be `[tensor([-8., -4.]), tensor(-4.)]`. Does your forward pass follow the formula: `w @ x + b`, ReLU with `torch.relu`, and the square of ŷ − y? And do you call `loss.backward()` once?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet `[tensor([-8., -4.]), tensor(-4.)]` zijn. Volgt je forward pass de formule: `w @ x + b`, ReLU met `torch.relu`, en het kwadraat van ŷ − y? En roep je `loss.backward()` één keer aan?"},
            "other_error": {"en": "Right for these numbers. But with {where}, PyTorch stops with an error. Does your code work for other weights and bias terms?",
                            "nl": "Goed voor deze getallen. Maar met {where} stopt PyTorch met een fout. Werkt je code ook voor andere weights en bias terms?"},
            "other": {"en": "Right for these numbers. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does your forward pass use ReLU, with `torch.relu`, and `w` and `b` themselves?",
                      "nl": "Goed voor deze getallen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruikt je forward pass ReLU, met `torch.relu`, en `w` en `b` zelf?"},
            "correct": {"en": "−8 and −4 for the weights, and −4 for the bias term: the same slopes as your own `backward` gave for this neuron. The check also tried a `z` below 0, where every slope is 0.",
                        "nl": "−8 en −4 voor de weights, en −4 voor de bias term: dezelfde hellingen als je eigen `backward` voor dit neuron gaf. De check probeerde ook een `z` onder 0, waar elke helling 0 is."},
        },
    ),
    "train": CodeBox(
        name="train",
        start=_T_START,
        solution=_T,
        keep=["for step in range(100):", "    loss = ((X @ w + b - y) ** 2).mean()"],
        others=[{"lr": 0.02}],
        first=[_DOTS],
        mistakes=[
            _TWICE,
            Mistake("in-place", error=lambda e: isinstance(e, RuntimeError) and "leaf Variable that requires grad" in str(e),
                    en="PyTorch stops: `w -= lr * w.grad` changes a parameter in place while PyTorch is still recording, and it refuses that. Put the step inside `with torch.no_grad():`.",
                    nl="PyTorch stopt: `w -= lr * w.grad` verandert een parameter ter plekke terwijl PyTorch nog opneemt, en dat weigert hij. Zet de stap binnen `with torch.no_grad():`."),
            Mistake("rebind", syntax=r"^\s*(w|b)\s*=\s*\1\s*[-+]",
                    en="Your code stops at the second step. `w = w - lr * w.grad` makes a new tensor and gives it the name `w`: it is not the tensor you made with `requires_grad=True`, so the next `loss.backward()` gives it no `.grad`. Change the parameter itself: `w -= lr * w.grad`, inside `with torch.no_grad():`.",
                    nl="Je code stopt bij de tweede stap. `w = w - lr * w.grad` maakt een nieuwe tensor en geeft die de naam `w`: dat is niet de tensor die je met `requires_grad=True` maakte, dus de volgende `loss.backward()` geeft hem geen `.grad`. Verander de parameter zelf: `w -= lr * w.grad`, binnen `with torch.no_grad():`."),
            Mistake("grad-zero", error=lambda e: isinstance(e, TypeError) and "assigned grad expected to be a Tensor or None" in str(e),
                    en="PyTorch stops: `.grad` holds a tensor in the shape of its parameter, or `None`, never a plain number such as 0. Zero the gradients the PyTorch way: `w.grad = None` and `b.grad = None`, before `loss.backward()`.",
                    nl="PyTorch stopt: `.grad` bevat een tensor in de shape van zijn parameter, of `None`, nooit een los getal zoals 0. Zet de gradients op nul zoals PyTorch dat doet: `w.grad = None` en `b.grad = None`, vóór `loss.backward()`."),
            Mistake("no-backward", error=lambda e: isinstance(e, TypeError) and "NoneType" in str(e),
                    en="Python stops: `w.grad` is `None` when the step comes. Either no `loss.backward()` filled it, or the `.grad`s were set to `None` after it. Keep this order: both `.grad`s to `None`, then `loss.backward()`, then the step.",
                    nl="Python stopt: `w.grad` is `None` als de stap komt. Ofwel heeft geen `loss.backward()` hem gevuld, ofwel zijn de `.grad`s daarna op `None` gezet. Houd deze volgorde aan: beide `.grad`s op `None`, dan `loss.backward()`, dan de stap."),
            Mistake("no-zero", wrong=_t(zero=""),
                    en="Your last line gives `{got}`: the slopes are never thrown away. PyTorch adds every new slope to `.grad`, as your own `Value` did, so every step also moves by the slopes of all the steps before it. Set `w.grad = None` and `b.grad = None` before `loss.backward()`.",
                    nl="Je laatste regel geeft `{got}`: de hellingen worden nooit weggegooid. PyTorch telt elke nieuwe helling op bij `.grad`, zoals je eigen `Value` deed, dus elke stap schuift ook met de hellingen van alle stappen ervoor. Zet `w.grad = None` en `b.grad = None` vóór `loss.backward()`."),
            Mistake("uphill", wrong=_t(step="    with torch.no_grad():\n        w += lr * w.grad\n        b += lr * b.grad\n"),
                    en="Your last line gives `{got}`: the loss runs away. Each step moves with the slope, uphill. Move against it: `w -= lr * w.grad`.",
                    nl="Je laatste regel geeft `{got}`: de loss loopt weg. Elke stap schuift met de helling mee, bergop. Schuif ertegenin: `w -= lr * w.grad`."),
        ],
        digits=4,
        texts={
            **_COMMON,
            "empty": {"en": "Replace the `...` with three parts of the loop: both `.grad`s to `None`, `loss.backward()`, and a step against the slope inside `with torch.no_grad():`.",
                      "nl": "Vervang de `...` door drie delen van de loop: beide `.grad`s op `None`, `loss.backward()`, en een stap tegen de helling in binnen `with torch.no_grad():`."},
            "no_value": {"en": "The last line of the box shows nothing. Put `loss.item()` back on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `loss.item()` terug op de laatste regel, zodat de check het kan zien."},
            "slow": {"en": "Your code is still not done after 2 seconds. Does the loop run 100 times, with one `loss.backward()` per step?",
                     "nl": "Je code is na 2 seconden nog niet klaar. Draait de loop 100 keer, met één `loss.backward()` per stap?"},
            "wrong": {"en": "Your last line gives `{got}`, and it should be about 0.3261. Does each step set both `.grad`s to `None`, then call `loss.backward()`, and then do `w -= lr * w.grad` and `b -= lr * b.grad` inside `with torch.no_grad():`?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet ongeveer 0,3261 zijn. Zet elke stap beide `.grad`s op `None`, roept hij dan `loss.backward()` aan, en doet hij daarna `w -= lr * w.grad` en `b -= lr * b.grad` binnen `with torch.no_grad():`?"},
            "other_error": {"en": "Right for this learning rate. But with {where}, Python stops with an error. Does your loop use `lr` itself?",
                            "nl": "Goed voor deze learning rate. Maar met {where} stopt Python met een fout. Gebruikt je loop `lr` zelf?"},
            "other": {"en": "Right for this learning rate. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does your step use `lr` itself, rather than a number you typed?",
                      "nl": "Goed voor deze learning rate. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruikt je stap `lr` zelf, en geen getal dat je hebt ingetypt?"},
            "correct": {"en": "The loss went from 21.5 to about 0.33 in 100 steps, and every slope came from autograd. The check also tried another learning rate.",
                        "nl": "De loss ging in 100 stappen van 21,5 naar ongeveer 0,33, en elke helling kwam van autograd. De check probeerde ook een andere learning rate."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("neuron", _N_START, "empty", None),
    ("neuron", _n("loss = (torch.relu(w @ x + b) - y) ** 2\nloss.backward()\n"), "correct", None),
    ("neuron", _n("z = (w * x).sum() + b\ny_pred = z.clamp(min=0)\nr = y_pred - y\nloss = r * r\nloss.backward()\n"), "correct", None),
    ("neuron", _n("z = w @ x + b\ny_pred = torch.relu(z)\nloss = (y_pred - y) ** 2\n"), "wrong", "no-backward"),
    ("neuron", _n("z = w @ x + b\ny_pred = torch.relu(z)\nloss = y_pred - y\nloss.backward()\n"), "wrong", "no-square"),
    ("neuron", _n("z = w @ x + b\ny_pred = torch.relu(z)\nloss = (y_pred - y) ** 2\nloss.backward()\nloss.backward()\n"), "error", "twice"),
    ("neuron", _n("z = w @ x + b\nloss = (z - y) ** 2\nloss.backward()\n"), "almost", "other-value"),
    ("neuron", _n("z = w @ x\ny_pred = torch.relu(z)\nloss = (y_pred - y) ** 2\nloss.backward()\n"), "wrong", "no-bias"),
    ("neuron", _n("z = w * x + b\ny_pred = torch.relu(z)\nloss = (y_pred - y) ** 2\nloss.backward()\n"), "error", "not-one-number"),
    ("neuron", _n("z = w @ x + b\ny_pred = torch.relu(z)\nloss = (y_pred - y) ** 2\ny_pred.backward()\n"), "wrong", "no-square"),
    ("train", _T_START, "empty", None),
    ("train", _t(zero="    w.grad = None\n    b.grad = None\n"), "correct", None),
    ("train", _t(step="    with torch.no_grad():\n        w.sub_(lr * w.grad)\n        b.sub_(lr * b.grad)\n"), "correct", None),
    ("train", _t(zero="", step="    with torch.no_grad():\n        w -= lr * w.grad\n        b -= lr * b.grad\n        w.grad = None\n        b.grad = None\n"), "correct", None),
    ("train", _t(zero=""), "wrong", "no-zero"),
    ("train", _t(step="    with torch.no_grad():\n        w += lr * w.grad\n        b += lr * b.grad\n"), "wrong", "uphill"),
    ("train", _t(step="    w -= lr * w.grad\n    b -= lr * b.grad\n"), "error", "in-place"),
    ("train", _t(step="    w = w - lr * w.grad\n    b = b - lr * b.grad\n"), "error", "rebind"),
    ("train", _t(step="    with torch.no_grad():\n        w = w - lr * w.grad\n        b = b - lr * b.grad\n"), "error", "rebind"),
    ("train", _t(zero="    w.grad = 0\n    b.grad = 0\n"), "error", "grad-zero"),
    ("train", _t(back=""), "error", "no-backward"),
    ("train", _t(zero="", back=_T_BACK + _T_ZERO), "error", "no-backward"),
    ("train", _t(step="    with torch.no_grad():\n        w -= 0.05 * w.grad\n        b -= 0.05 * b.grad\n"), "almost", "other-value"),
]
