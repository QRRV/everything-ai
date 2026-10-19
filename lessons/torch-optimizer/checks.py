# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson torch-optimizer: the training loop of torch-nn-module with `torch.optim.SGD` doing the
zeroing and the step, and a function `train` that makes its own optimizer and gives back the loss of every step as
Python numbers."""
import re

import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import draw
from everything_ai.check import CodeBox, Mistake, run_code, same_value


def picture(labels: dict[str, str], title: str):
    """One step of the training loop twice: with the two loops of torch-nn-module, and with an optimizer. The
    numbers mark which lines each call of the optimizer takes over."""
    return draw.marked_lines([
        (labels["own"], [("loss = ((net(X) - y) ** 2).mean()", ""), ("for p in net.parameters():", "1"),
                         ("    p.grad = None", "1"), ("loss.backward()", ""), ("with torch.no_grad():", "2"),
                         ("    for p in net.parameters():", "2"), ("        p -= lr * p.grad", "2")]),
        (labels["opt"], [("loss = ((net(X) - y) ** 2).mean()", ""), ("opt.zero_grad()", "1"), ("loss.backward()", ""),
                         ("opt.step()", "2")]),
    ], {"1": labels["1"], "2": labels["2"]}, title=title)


def _close_loss(got, want) -> bool:
    """The loss after training: the course's, or, when the course's is almost 0, almost 0 too. Shown with 4 digits,
    0.00001 and 0.000002 both read 0, and a learner who trains just as well must not hear that 0 should be almost 0."""
    if isinstance(got, float) and isinstance(want, float) and want < 1e-3:
        return 0 <= got < 1e-3
    return same_value(got, want)


def _numbers(got, want) -> bool:
    """The losses of `train`, as Python numbers: a list of tensors, even with the right values in it, is not that yet.
    That is what `.item()` is for, so this exercise does not count tensors as their numbers."""
    return isinstance(got, list) and not any(isinstance(v, torch.Tensor) for v in got) and same_value(got, want)


# The lesson is done when these are correct. "stale" is the choice question in the lesson.
REQUIRED = ["stale", "loop", "train"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_MODULE = Mistake("module", error=lambda e: isinstance(e, TypeError) and "'Net' object is not iterable" in str(e),
                  en="Python stops: the optimizer needs the parameters of the network, not the network itself. Give it `net.parameters()`.",
                  nl="Python stopt: de optimizer heeft de parameters van het netwerk nodig, niet het netwerk zelf. Geef hem `net.parameters()`.")
_METHOD = Mistake("method", error=lambda e: isinstance(e, TypeError) and "'method' object is not iterable" in str(e),
                  en="Python stops: `net.parameters` without brackets is the method itself, not the parameters it gives. Call it: `net.parameters()`.",
                  nl="Python stopt: `net.parameters` zonder haakjes is de method zelf, niet de parameters die hij geeft. Roep hem aan: `net.parameters()`.")
_STEP_FIRST = Mistake("step-first", error=lambda e: isinstance(e, RuntimeError) and "modified by an inplace operation" in str(e),
                      en="PyTorch stops at `loss.backward()`: `opt.step()` has already moved the parameters that this step's loss was worked out with, and the backward pass needs them as they were. Take the step after the backward pass: `opt.zero_grad()`, `loss.backward()`, then `opt.step()`.",
                      nl="PyTorch stopt bij `loss.backward()`: `opt.step()` heeft de parameters waarmee de loss van deze stap is uitgerekend al verschoven, en de backward pass heeft ze nodig zoals ze waren. Zet de stap na de backward pass: `opt.zero_grad()`, `loss.backward()`, dan `opt.step()`.")
_NO_MOVE_EN = ("the loss never changes, so no parameter of `net` moved. `opt.step()` only moves the parameters `opt` was made "
               "with, and of those only the ones that have a `.grad`. Is `opt` made from `net.parameters()`, and does each step "
               "call `opt.zero_grad()`, then `loss.backward()`, then `opt.step()`, in that order?")
_NO_MOVE_NL = ("de loss verandert nooit, dus geen enkele parameter van `net` is verschoven. `opt.step()` verschuift alleen de "
               "parameters waarmee `opt` gemaakt is, en daarvan alleen die met een `.grad`. Is `opt` gemaakt van "
               "`net.parameters()`, en roept elke stap `opt.zero_grad()` aan, dan `loss.backward()`, dan `opt.step()`, in die volgorde?")
_NO_ZERO_EN = ("the slopes are never thrown away. `loss.backward()` adds to every `.grad`, so every step also moves by the "
               "slopes of all the steps before it. Call `opt.zero_grad()` before `loss.backward()`.")
_NO_ZERO_NL = ("de hellingen worden nooit weggegooid. `loss.backward()` telt op bij elke `.grad`, dus elke stap schuift ook "
               "met de hellingen van alle stappen ervoor. Roep `opt.zero_grad()` aan vóór `loss.backward()`.")
_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
    "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
             "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
}

_CLASS = ("import torch\nfrom torch import nn\n\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
          "        self.hidden = nn.Linear(2, 3)\n        self.out = nn.Linear(3, 1)\n\n    def forward(self, x):\n"
          "        return self.out(torch.relu(self.hidden(x)))\n\n\n")
_DATA = ("X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])\n"
         "y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])\n")

# --- exercise 2: the loop with an optimizer ---------------------------------------------------------------------------

_O_HEAD = (_CLASS + "torch.manual_seed(0)\nnet = Net()\n" + _DATA + "lr = 0.1\nopt = torch.optim.SGD(net.parameters(), lr=lr)\n\n"
           "for step in range(500):\n    loss = ((net(X) - y) ** 2).mean()\n")
_O_START = _O_HEAD + "    ...\nloss.item()\n"


def _o(body: str = "    opt.zero_grad()\n    loss.backward()\n    opt.step()\n") -> str:
    return _O_HEAD + body + "loss.item()"


_O = _o()

# --- exercise 3: a function train ---------------------------------------------------------------------------------------

_T_DEF = "def train(net, X, y, lr, steps):\n"
_T_END = "\n\ntorch.manual_seed(0)\nnet = Net()\n" + _DATA + "train(net, X, y, 0.1, 5)"
_T_START = (_CLASS + _T_DEF + "    opt = ...\n    losses = []\n    for step in range(steps):\n        loss = ((net(X) - y) ** 2).mean()\n"
            "        ...\n    return losses\n" + _T_END + "\n")


def _t(opt: str = "    opt = torch.optim.SGD(net.parameters(), lr=lr)\n",
       body: str = "        opt.zero_grad()\n        loss.backward()\n        opt.step()\n        losses.append(loss.item())\n",
       end: str = "    return losses\n") -> str:
    return (_CLASS + _T_DEF + opt + "    losses = []\n    for step in range(steps):\n        loss = ((net(X) - y) ** 2).mean()\n"
            + body + end + _T_END)


_T = _t()
_T_OTHER = "\ntorch.manual_seed(1)\ntrain(Net(), X, y, 0.05, 3)"


def _typed(got, text) -> bool:
    """The losses are right for `train(net, X, y, 0.1, 5)`, but not for a new network with another learning rate
    and number of steps: `train` uses a number typed in, such as `lr=0.1` or `range(5)`, instead of `lr` and `steps`."""
    mine = run_code(text + _T_OTHER)
    return mine.error is not None or mine.timeout or not _numbers(mine.value, run_code(_T + _T_OTHER).value)

EXERCISES = {
    "loop": CodeBox(
        name="loop",
        compare=_close_loss,
        start=_O_START,
        solution=_O,
        keep=["torch.manual_seed(0)", "for step in range(500):", "    loss = ((net(X) - y) ** 2).mean()"],
        others=[{"lr": 0.05}],
        first=[_DOTS],
        mistakes=[
            _MODULE,
            _METHOD,
            _STEP_FIRST,
            Mistake("no-zero", wrong=_o("    loss.backward()\n    opt.step()\n"),
                    en="Your last line gives `{got}`: " + _NO_ZERO_EN, nl="Je laatste regel geeft `{got}`: " + _NO_ZERO_NL),
            # also a loop without backward, without the step, or with zero_grad between backward and the step
            Mistake("no-move", wrong=_o("    loss.backward()\n    opt.zero_grad()\n    opt.step()\n"),
                    en="Your last line gives `{got}`: " + _NO_MOVE_EN, nl="Je laatste regel geeft `{got}`: " + _NO_MOVE_NL),
        ],
        digits=4,
        texts={
            **_COMMON,
            "empty": {"en": "Replace the `...` with the rest of each step: steps 3 and 4 of the recipe, in that order.",
                      "nl": "Vervang de `...` door de rest van elke stap: stap 3 en 4 van het recept, in die volgorde."},
            "no_value": {"en": "The last line of the box shows nothing. Put `loss.item()` back on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `loss.item()` terug op de laatste regel, zodat de check het kan zien."},
            "slow": {"en": "Your code is still not done after 2 seconds. Does the loop run 500 times, with one `loss.backward()` per step?",
                     "nl": "Je code is na 2 seconden nog niet klaar. Draait de loop 500 keer, met één `loss.backward()` per stap?"},
            "wrong": {"en": "Your last line gives `{got}`, and it should be almost 0. Does your loop do steps 3 and 4 of the recipe every time, in that order, as in the worked example?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet bijna 0 zijn. Doet je loop elke keer stap 3 en 4 van het recept, in die volgorde, zoals in het uitgewerkte voorbeeld?"},
            "other_error": {"en": "Right for this learning rate. But with {where}, Python stops with an error. Does your code use `lr` itself?",
                            "nl": "Goed voor deze learning rate. Maar met {where} stopt Python met een fout. Gebruikt je code `lr` zelf?"},
            "other": {"en": "Right for this learning rate. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does the optimizer get `lr=lr`, rather than a number you typed?",
                      "nl": "Goed voor deze learning rate. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Krijgt de optimizer `lr=lr`, en geen getal dat je hebt ingetypt?"},
            "correct": {"en": "The loss is almost 0, the same as with your own two loops in the lesson before: `torch.optim.SGD` takes the same step. The check also tried another learning rate.",
                        "nl": "De loss is bijna 0, net als met je eigen twee loops in de vorige les: `torch.optim.SGD` zet dezelfde stap. De check probeerde ook een andere learning rate."},
        },
    ),
    "train": CodeBox(
        name="train",
        compare=_numbers,
        start=_T_START,
        solution=_T,
        keep=["def train(net, X, y, lr, steps):", "torch.manual_seed(0)", "train(net, X, y, 0.1, 5)"],
        first=[_DOTS],
        mistakes=[
            _MODULE,
            _METHOD,
            _STEP_FIRST,
            Mistake("tensors", matches=lambda got, text: isinstance(got, list) and len(got) > 0 and isinstance(got[0], torch.Tensor),
                    en="Your last line gives a list of tensors, `{got}`. Keep the losses as Python numbers: `losses.append(loss.item())`.",
                    nl="Je laatste regel geeft een lijst met tensors, `{got}`. Bewaar de losses als Python-getallen: `losses.append(loss.item())`."),
            Mistake("early-return", wrong=_t(body="        opt.zero_grad()\n        loss.backward()\n        opt.step()\n        losses.append(loss.item())\n        return losses\n", end=""),
                    en="Your last line gives `{got}`: one loss, not five. Is `return losses` inside the loop? Then `train` stops after the first step. Put it after the loop, at the indentation of `for`.",
                    nl="Je laatste regel geeft `{got}`: één loss, geen vijf. Staat `return losses` in de loop? Dan stopt `train` na de eerste stap. Zet hem na de loop, op het inspringniveau van `for`."),
            Mistake("append-after", wrong=_t(body="        opt.zero_grad()\n        loss.backward()\n        opt.step()\n", end="    losses.append(loss.item())\n    return losses\n"),
                    en="Your last line gives `{got}`: one loss, the last one, not five. Is `losses.append(loss.item())` after the loop? Then it runs only once, after the last step. Put it inside the loop, at the indentation of the three calls.",
                    nl="Je laatste regel geeft `{got}`: één loss, de laatste, geen vijf. Staat `losses.append(loss.item())` na de loop? Dan draait hij maar één keer, na de laatste stap. Zet hem in de loop, op het inspringniveau van de drie aanroepen."),
            Mistake("no-zero", wrong=_t(body="        loss.backward()\n        opt.step()\n        losses.append(loss.item())\n"),
                    en="Your last line gives `{got}`: " + _NO_ZERO_EN, nl="Je laatste regel geeft `{got}`: " + _NO_ZERO_NL),
            Mistake("no-move", wrong=_t(body="        loss.backward()\n        opt.zero_grad()\n        opt.step()\n        losses.append(loss.item())\n"),
                    en="Your last line gives `{got}`: " + _NO_MOVE_EN, nl="Je laatste regel geeft `{got}`: " + _NO_MOVE_NL),
        ],
        form=[
            Mistake("typed", matches=_typed, state="almost",
                    en="Right for `train(net, X, y, 0.1, 5)`. But a new network with another learning rate and number of steps, `train(Net(), X, y, 0.05, 3)`, does not give the losses it should. Does `train` use its own `lr` and `steps`, rather than numbers you typed?",
                    nl="Goed voor `train(net, X, y, 0.1, 5)`. Maar een nieuw netwerk met een andere learning rate en een ander aantal stappen, `train(Net(), X, y, 0.05, 3)`, geeft niet de losses die het moet geven. Gebruikt `train` zijn eigen `lr` en `steps`, en geen getallen die je hebt ingetypt?"),
        ],
        digits=4,
        texts={
            **_COMMON,
            "empty": {"en": "Replace the two `...`: before the loop, the optimizer, made from the parameters of `net` and the learning rate `lr`; in the loop, steps 3 and 4 of the recipe, with the loss kept as a Python number.",
                      "nl": "Vervang de twee `...`: vóór de loop de optimizer, gemaakt van de parameters van `net` en de learning rate `lr`; in de loop stap 3 en 4 van het recept, waarbij je de loss als Python-getal bewaart."},
            "no_value": {"en": "The last line of the box shows nothing. Put `train(net, X, y, 0.1, 5)` back on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `train(net, X, y, 0.1, 5)` terug op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be about `[1.3444, 0.9206, 0.6565, 0.4894, 0.3874]`. Does `train` make its optimizer from the parameters of `net` with `lr`, do steps 3 and 4 of the recipe every time, and keep each loss as a Python number?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet ongeveer `[1.3444, 0.9206, 0.6565, 0.4894, 0.3874]` zijn. Maakt `train` zijn optimizer van de parameters van `net` met `lr`, doet hij elke keer stap 3 en 4 van het recept, en bewaart hij elke loss als Python-getal?"},
            "correct": {"en": "Five losses, each lower than the one before: 1.3444, 0.9206, 0.6565, 0.4894 and 0.3874. Your `train` works for any network and any learning rate: the check also tried a new network with `lr=0.05` and 3 steps.",
                        "nl": "Vijf losses, elke lager dan de vorige: 1,3444; 0,9206; 0,6565; 0,4894 en 0,3874. Je `train` werkt voor elk netwerk en elke learning rate: de check probeerde ook een nieuw netwerk met `lr=0.05` en 3 stappen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("loop", _O_START, "empty", None),
    ("loop", _o("    loss.backward()\n    opt.step()\n    opt.zero_grad()\n"), "correct", None),
    ("loop", _o("    loss.backward()\n    opt.step()\n"), "wrong", "no-zero"),
    ("loop", _o("    opt.zero_grad()\n    loss.backward()\n"), "wrong", "no-move"),
    ("loop", _o("    opt.zero_grad()\n    opt.step()\n"), "wrong", "no-move"),
    ("loop", _o("    opt.zero_grad()\n    opt.step()\n    loss.backward()\n"), "wrong", "no-move"),
    ("loop", _o("    opt.step()\n    loss.backward()\n"), "error", "step-first"),
    ("loop", _o("    opt.step()\n    opt.zero_grad()\n    loss.backward()\n"), "error", "step-first"),
    ("loop", _O.replace("opt = torch.optim.SGD(net.parameters(), lr=lr)", "opt = torch.optim.SGD(net, lr=lr)"), "error", "module"),
    ("loop", _O.replace("opt = torch.optim.SGD(net.parameters(), lr=lr)", "opt = torch.optim.SGD(net.parameters, lr=lr)"), "error", "method"),
    ("loop", _O.replace("opt = torch.optim.SGD(net.parameters(), lr=lr)", "opt = torch.optim.SGD(net.parameters(), lr=0.1)"), "almost", "other-value"),
    ("train", _T_START, "empty", None),
    ("train", _t(body="        opt.zero_grad()\n        loss.backward()\n        losses.append(loss.item())\n        opt.step()\n"), "correct", None),
    ("train", _t(body="        opt.zero_grad()\n        loss.backward()\n        opt.step()\n        losses.append(loss)\n"), "wrong", "tensors"),
    ("train", _t(body="        opt.zero_grad()\n        loss.backward()\n        opt.step()\n        losses.append(loss.item())\n        return losses\n", end=""), "wrong", "early-return"),
    ("train", _t(body="        loss.backward()\n        opt.step()\n        losses.append(loss.item())\n"), "wrong", "no-zero"),
    ("train", _t(body="        opt.zero_grad()\n        opt.step()\n        losses.append(loss.item())\n"), "wrong", "no-move"),
    ("train", _t(opt="    opt = torch.optim.SGD(net, lr=lr)\n"), "error", "module"),
    ("train", _t(body="        opt.step()\n        loss.backward()\n        losses.append(loss.item())\n"), "error", "step-first"),
    ("train", _t(body="        opt.zero_grad()\n        loss.backward()\n        opt.step()\n        losses.append(float(loss))\n"), "correct", None),
    ("train", _t(opt="    opt = torch.optim.SGD(net.parameters(), lr=0.1)\n"), "almost", "typed"),
    ("train", _t().replace("range(steps)", "range(5)"), "almost", "typed"),
]
