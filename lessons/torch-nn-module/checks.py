# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson torch-nn-module: a network of two `nn.Linear` layers with ReLU between them, written as a
class on `nn.Module`, and a training loop over `net.parameters()` that teaches it XOR. And the picture: the
network, with the layer each `nn.Linear` is."""
import re

import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import draw
from everything_ai.check import CodeBox, Mistake, run_code, same_value


def picture(captions: list[str], title: str, alt: str):
    """The network of the lesson: 2 inputs, a hidden layer of 3 neurons and 1 output neuron."""
    return draw.network_diagram([2, 3, 1], captions, title=title, alt=alt)


def _close_loss(got, want) -> bool:
    """The loss after training: the course's, or, when the course's is almost 0, almost 0 too. Shown with 4 digits,
    0.00001 and 0.000002 both read 0, and a learner who trains just as well must not hear that 0 should be almost 0."""
    if isinstance(got, float) and isinstance(want, float) and want < 1e-3:
        return 0 <= got < 1e-3
    return same_value(got, want)


# The lesson is done when these are correct. "output-shape" is the choice question in the lesson.
REQUIRED = ["output-shape", "net", "train"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_NO_SUPER = Mistake("no-super", error=lambda e: isinstance(e, AttributeError) and "before Module.__init__() call" in str(e),
                    en="Python stops: `nn.Module` has to set itself up before your class can give it a layer. Start `__init__` with `super().__init__()`, and only then make the layers.",
                    nl="Python stopt: `nn.Module` moet zichzelf eerst opzetten voordat je class er een layer aan kan geven. Begin `__init__` met `super().__init__()`, en maak pas daarna de layers.")
_NO_FORWARD = Mistake("no-forward", error=lambda e: isinstance(e, NotImplementedError) and "forward" in str(e),
                      en="PyTorch stops: your class has no method `forward`, so `net(X)` does not know what to calculate. Write `def forward(self, x):`, with exactly that name.",
                      nl="PyTorch stopt: je class heeft geen method `forward`, dus `net(X)` weet niet wat het moet uitrekenen. Schrijf `def forward(self, x):`, met precies die naam.")
_SIZES = Mistake("sizes", error=lambda e: isinstance(e, RuntimeError) and "cannot be multiplied" in str(e),
                 en="PyTorch stops: the sizes of the layers do not fit. `nn.Linear(n_inputs, n_outputs)`: the hidden layer gets the 2 inputs and has 3 neurons, `nn.Linear(2, 3)`, and the output layer gets those 3, `nn.Linear(3, 1)`.",
                 nl="PyTorch stopt: de groottes van de layers passen niet. `nn.Linear(n_inputs, n_outputs)`: de hidden layer krijgt de 2 inputs en heeft 3 neuronen, `nn.Linear(2, 3)`, en de output-layer krijgt die 3, `nn.Linear(3, 1)`.")
_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
    "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
             "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
}

# --- exercise 2: the class ------------------------------------------------------------------------------------------

_IMPORTS = "import torch\nfrom torch import nn\n\n\n"
_DATA = "X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])\n"
_N_END = "\n\ntorch.manual_seed(0)\nnet = Net()\n" + _DATA + "net(X)"
_N_START = (_IMPORTS + "class Net(nn.Module):\n    def __init__(self):\n        ...\n\n    def forward(self, x):\n        ...\n"
            + _N_END + "\n")


def _net(init: str = "        super().__init__()\n        self.hidden = nn.Linear(2, 3)\n        self.out = nn.Linear(3, 1)\n",
         forward: str = "        return self.out(torch.relu(self.hidden(x)))\n") -> str:
    return "class Net(nn.Module):\n    def __init__(self):\n" + init + "\n    def forward(self, x):\n" + forward


_CLASS = _net()
_N = _IMPORTS + _CLASS + _N_END


def _n(**parts) -> str:
    return _IMPORTS + _net(**parts) + _N_END


def _unfound(got, text) -> bool:
    """The outcome is right, but `net.parameters()` does not find the 4 tensors of the two layers, such as when the
    layers are kept in a plain list: training would then change nothing."""
    run = run_code(text + "\nlen(list(net.parameters()))")
    return run.error is None and run.value != 4


_LIST = _n(init="        super().__init__()\n        self.layers = [nn.Linear(2, 3), nn.Linear(3, 1)]\n",
           forward="        return self.layers[1](torch.relu(self.layers[0](x)))\n")
_CALL = (_IMPORTS + "class Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.hidden = nn.Linear(2, 3)\n"
         "        self.out = nn.Linear(3, 1)\n\n    def __call__(self, x):\n        return self.out(torch.relu(self.hidden(x)))\n" + _N_END)


# --- exercise 3: training it on XOR -----------------------------------------------------------------------------------

_T_HEAD = (_IMPORTS + _CLASS + "\n\ntorch.manual_seed(0)\nnet = Net()\n" + _DATA
           + "y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])\nlr = 0.1\n\n"
           "for step in range(500):\n    loss = ((net(X) - y) ** 2).mean()\n")
_T_END = "loss.item()"
_T_ZERO = "    for p in net.parameters():\n        p.grad = None\n"
_T_BACK = "    loss.backward()\n"
_T_STEP = "    with torch.no_grad():\n        for p in net.parameters():\n            p -= lr * p.grad\n"
_T_START = _T_HEAD + "    ...\n" + _T_END + "\n"


def _t(zero: str = _T_ZERO, back: str = _T_BACK, step: str = _T_STEP) -> str:
    return _T_HEAD + zero + back + step + _T_END


_T = _t()

EXERCISES = {
    "net": CodeBox(
        name="net",
        start=_N_START,
        solution=_N,
        keep=["torch.manual_seed(0)", "net = Net()", "net(X)"],
        first=[_DOTS],
        mistakes=[
            _NO_SUPER,
            _NO_FORWARD,
            _SIZES,
            Mistake("order", wrong=_n(init="        super().__init__()\n        self.out = nn.Linear(3, 1)\n        self.hidden = nn.Linear(2, 3)\n"),
                    en="Your last line gives `{got}`. Your network may be right, but its starting weights differ: PyTorch draws the random weights of each layer when you make it, so the order counts. Make `self.hidden` first and `self.out` second, as the network goes.",
                    nl="Je laatste regel geeft `{got}`. Je netwerk kan goed zijn, maar zijn beginweights verschillen: PyTorch trekt de willekeurige weights van elke layer als je hem maakt, dus de volgorde telt. Maak eerst `self.hidden` en dan `self.out`, in de volgorde van het netwerk."),
            Mistake("no-relu", wrong=_n(forward="        return self.out(self.hidden(x))\n"),
                    en="Your last line gives `{got}`: there is no ReLU between the layers. Two `nn.Linear` layers in a row act as one linear layer, which cannot learn XOR. Put `torch.relu` around the output of `self.hidden`.",
                    nl="Je laatste regel geeft `{got}`: er zit geen ReLU tussen de layers. Twee `nn.Linear`-layers achter elkaar werken samen als één linear layer, en die kan XOR niet leren. Zet `torch.relu` om de output van `self.hidden`."),
            Mistake("hidden-only", matches=lambda got, text: isinstance(got, torch.Tensor) and tuple(got.shape) == (4, 3),
                    en="Your last line gives `{got}`: 3 numbers per example, the hidden layer. `forward` should also send them through `self.out`, so that each example gets one output.",
                    nl="Je laatste regel geeft `{got}`: 3 getallen per voorbeeld, de hidden layer. `forward` moet ze ook door `self.out` sturen, zodat elk voorbeeld één output krijgt."),
            Mistake("relu-out", wrong=_n(forward="        return torch.relu(self.out(torch.relu(self.hidden(x))))\n"),
                    en="Your last line gives `{got}`: there is a ReLU after `self.out`, and it turns every output below 0 into 0. The output layer has no ReLU, so that it can give any number: ReLU goes only between the layers.",
                    nl="Je laatste regel geeft `{got}`: er staat een ReLU na `self.out`, en die maakt elke output onder 0 tot 0. De output-layer heeft geen ReLU, zodat hij elk getal kan geven: ReLU hoort alleen tussen de layers."),
            Mistake("no-return", matches=lambda got, text: got is None,
                    en="Your last line gives `None`: your `forward` gives nothing back. Start its last line with `return`.",
                    nl="Je laatste regel geeft `None`: je `forward` geeft niets terug. Begin zijn laatste regel met `return`."),
        ],
        form=[
            Mistake("unfound", matches=_unfound,
                    en="Your last line gives the right numbers, but `net.parameters()` does not find your layers: a module only finds layers that are its own attributes, such as `self.hidden = nn.Linear(2, 3)`, not layers inside a list or a dictionary. Training would then change nothing. Make each layer an attribute.",
                    nl="Je laatste regel geeft de goede getallen, maar `net.parameters()` vindt je layers niet: een module vindt alleen layers die zijn eigen attributen zijn, zoals `self.hidden = nn.Linear(2, 3)`, geen layers in een lijst of een dictionary. Trainen zou dan niets veranderen. Maak van elke layer een attribuut."),
            Mistake("own-call", matches=lambda got, text: re.search(r"^\s*def __call__\(", text, re.MULTILINE) is not None,
                    en="Your last line gives the right numbers, but your class writes its own `__call__`. On an `nn.Module` you write `forward`: `nn.Module` already has a `__call__`, which runs your `forward` and does more that PyTorch needs later. Rename the method to `forward`.",
                    nl="Je laatste regel geeft de goede getallen, maar je class schrijft een eigen `__call__`. Op een `nn.Module` schrijf je `forward`: `nn.Module` heeft al een `__call__`, die je `forward` uitvoert en meer doet wat PyTorch later nodig heeft. Noem de method `forward`."),
        ],
        digits=4,
        texts={
            **_COMMON,
            "empty": {"en": "Replace the two `...`: in `__init__`, `super().__init__()` and the two layers; in `forward`, the hidden layer, ReLU and the output layer.",
                      "nl": "Vervang de twee `...`: in `__init__` `super().__init__()` en de twee layers; in `forward` de hidden layer, ReLU en de output-layer."},
            "no_value": {"en": "The last line of the box shows nothing. Put `net(X)` back on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `net(X)` terug op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be about `[[-0.6494], [-0.5172], [-0.5516], [-0.4966]]`. Compare with the picture: does `self.hidden` get the 2 inputs and have 3 neurons, does `self.out` get those 3 and have 1, and does `forward` send `x` through `self.hidden`, ReLU and `self.out`, in that order?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet ongeveer `[[-0.6494], [-0.5172], [-0.5516], [-0.4966]]` zijn. Vergelijk met de figuur: krijgt `self.hidden` de 2 inputs en heeft hij 3 neuronen, krijgt `self.out` die 3 en heeft hij er 1, en stuurt `forward` `x` door `self.hidden`, ReLU en `self.out`, in die volgorde?"},
            "correct": {"en": "One output per example, from a network of 13 parameters, all random for now. The next exercise trains it.",
                        "nl": "Eén output per voorbeeld, van een netwerk met 13 parameters, voorlopig allemaal willekeurig. De volgende oefening traint het."},
        },
    ),
    "train": CodeBox(
        name="train",
        compare=_close_loss,
        start=_T_START,
        solution=_T,
        keep=["torch.manual_seed(0)", "for step in range(500):", "    loss = ((net(X) - y) ** 2).mean()"],
        others=[{"lr": 0.05}],
        first=[_DOTS],
        mistakes=[
            Mistake("in-place", error=lambda e: isinstance(e, RuntimeError) and "leaf Variable that requires grad" in str(e),
                    en="PyTorch stops: the step changes a parameter in place while PyTorch is still recording. Put the loop with the step inside `with torch.no_grad():`.",
                    nl="PyTorch stopt: de stap verandert een parameter ter plekke terwijl PyTorch nog opneemt. Zet de loop met de stap binnen `with torch.no_grad():`."),
            Mistake("zero-none", error=lambda e: isinstance(e, AttributeError) and "'NoneType' object has no attribute 'zero_'" in str(e),
                    en="Python stops in the first step: no `loss.backward()` has run yet, so `p.grad` is still `None`, and `None` has nothing to set to zero. Set `p.grad = None` instead, as in the recipe.",
                    nl="Python stopt in de eerste stap: er is nog geen `loss.backward()` geweest, dus `p.grad` is nog `None`, en bij `None` valt niets op nul te zetten. Zet in plaats daarvan `p.grad = None`, zoals in het recept."),
            Mistake("backward-twice", error=lambda e: isinstance(e, RuntimeError) and "backward through the graph a second time" in str(e),
                    en="PyTorch stops: `loss.backward()` runs more than once in a step, because it is inside the loop over the parameters. Take it out of that loop: one backward pass per step.",
                    nl="PyTorch stopt: `loss.backward()` draait meer dan één keer per stap, omdat hij in de loop over de parameters staat. Haal hem uit die loop: één backward pass per stap."),
            Mistake("no-backward", error=lambda e: isinstance(e, TypeError) and "NoneType" in str(e),
                    en="Python stops: a `p.grad` is `None` when the step comes. Either no `loss.backward()` filled it, or the `.grad`s were set to `None` after it. Keep this order: every `.grad` to `None`, then `loss.backward()`, then the step.",
                    nl="Python stopt: een `p.grad` is `None` als de stap komt. Ofwel heeft geen `loss.backward()` hem gevuld, ofwel zijn de `.grad`s daarna op `None` gezet. Houd deze volgorde aan: elke `.grad` op `None`, dan `loss.backward()`, dan de stap."),
            Mistake("rebind", wrong=_t(step="    with torch.no_grad():\n        for p in net.parameters():\n            p = p - lr * p.grad\n"),
                    en="Your last line gives `{got}`: the loss never changes. `p = p - lr * p.grad` makes a new tensor and calls it `p`; the parameter inside the network stays as it was. Change the parameter itself: `p -= lr * p.grad`.",
                    nl="Je laatste regel geeft `{got}`: de loss verandert nooit. `p = p - lr * p.grad` maakt een nieuwe tensor en noemt die `p`; de parameter in het netwerk blijft zoals hij was. Verander de parameter zelf: `p -= lr * p.grad`."),
            Mistake("no-zero", wrong=_t(zero=""),
                    en="Your last line gives `{got}`: the slopes are never thrown away, so every step also moves by the slopes of all the steps before it. Set every `p.grad = None` before `loss.backward()`.",
                    nl="Je laatste regel geeft `{got}`: de hellingen worden nooit weggegooid, dus elke stap schuift ook met de hellingen van alle stappen ervoor. Zet elke `p.grad = None` vóór `loss.backward()`."),
            Mistake("uphill", wrong=_t(step="    with torch.no_grad():\n        for p in net.parameters():\n            p += lr * p.grad\n"),
                    en="Your last line gives `{got}`: the loss runs away until it is no number any more. Each step moves with the slope, uphill. Move against it: `p -= lr * p.grad`.",
                    nl="Je laatste regel geeft `{got}`: de loss loopt weg tot hij geen getal meer is. Elke stap schuift met de helling mee, bergop. Schuif ertegenin: `p -= lr * p.grad`."),
        ],
        digits=4,
        texts={
            **_COMMON,
            "empty": {"en": "Replace the `...` with three parts of the loop: every `p.grad` of `net.parameters()` to `None`, `loss.backward()`, and the step for every parameter inside `with torch.no_grad():`.",
                      "nl": "Vervang de `...` door drie delen van de loop: elke `p.grad` van `net.parameters()` op `None`, `loss.backward()`, en de stap voor elke parameter binnen `with torch.no_grad():`."},
            "no_value": {"en": "The last line of the box shows nothing. Put `loss.item()` back on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `loss.item()` terug op de laatste regel, zodat de check het kan zien."},
            "slow": {"en": "Your code is still not done after 2 seconds. Does the loop run 500 times, with one `loss.backward()` per step?",
                     "nl": "Je code is na 2 seconden nog niet klaar. Draait de loop 500 keer, met één `loss.backward()` per stap?"},
            "wrong": {"en": "Your last line gives `{got}`, and it should be almost 0. Does each step set every `.grad` of `net.parameters()` to `None`, then run `loss.backward()`, and then move every parameter against its slope, inside `with torch.no_grad():`?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet bijna 0 zijn. Zet je loop in elke stap elke `.grad` van `net.parameters()` op `None`, draait hij dan `loss.backward()`, en schuift hij daarna elke parameter tegen zijn helling in, binnen `with torch.no_grad():`?"},
            "other_error": {"en": "Right for this learning rate. But with {where}, Python stops with an error. Does your loop use `lr` itself?",
                            "nl": "Goed voor deze learning rate. Maar met {where} stopt Python met een fout. Gebruikt je loop `lr` zelf?"},
            "other": {"en": "Right for this learning rate. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does your step use `lr` itself, rather than a number you typed?",
                      "nl": "Goed voor deze learning rate. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruikt je stap `lr` zelf, en geen getal dat je hebt ingetypt?"},
            "correct": {"en": "The loss is almost 0: your network gives about 0, 1, 1 and 0 for the four corners. XOR, learned by two `nn.Linear` layers. The check also tried another learning rate.",
                        "nl": "De loss is bijna 0: je netwerk geeft ongeveer 0, 1, 1 en 0 voor de vier hoeken. XOR, geleerd door twee `nn.Linear`-layers. De check probeerde ook een andere learning rate."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("net", _N_START, "empty", None),
    ("net", _n(forward="        h = torch.relu(self.hidden(x))\n        return self.out(h)\n"), "correct", None),
    ("net", _n(forward="        h = self.hidden(x).clamp(min=0)\n        return self.out(h)\n"), "correct", None),
    ("net", _n(init="        self.hidden = nn.Linear(2, 3)\n        self.out = nn.Linear(3, 1)\n"), "error", "no-super"),
    ("net", _IMPORTS + "class Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.hidden = nn.Linear(2, 3)\n        self.out = nn.Linear(3, 1)\n\n"
            "    def predict(self, x):\n        return self.out(torch.relu(self.hidden(x)))\n" + _N_END, "error", "no-forward"),
    ("net", _n(init="        super().__init__()\n        self.hidden = nn.Linear(3, 2)\n        self.out = nn.Linear(2, 1)\n"), "error", "sizes"),
    ("net", _n(init="        super().__init__()\n        self.out = nn.Linear(3, 1)\n        self.hidden = nn.Linear(2, 3)\n"), "wrong", "order"),
    ("net", _n(forward="        return self.out(self.hidden(x))\n"), "wrong", "no-relu"),
    ("net", _n(forward="        return torch.relu(self.hidden(x))\n"), "wrong", "hidden-only"),
    ("net", _n(forward="        return torch.relu(self.out(torch.relu(self.hidden(x))))\n"), "wrong", "relu-out"),
    ("net", _n(forward="        return torch.relu(self.out(self.hidden(x)))\n"), "wrong", "relu-out"),
    ("net", _n(forward="        self.out(torch.relu(self.hidden(x)))\n"), "wrong", "no-return"),
    ("net", _LIST, "wrong", "unfound"),
    ("net", _n(init="        super().__init__()\n        self.hidden = nn.Linear(2, 3)\n        self.out = nn.Linear(3, 1)\n        self.layers = [self.hidden, self.out]\n"), "correct", None),
    ("net", _n(init="        super().__init__()\n        self.hidden = nn.Linear(2, 3)\n        self.relu = nn.ReLU()\n        self.out = nn.Linear(3, 1)\n",
               forward="        return self.out(self.relu(self.hidden(x)))\n"), "correct", None),
    ("net", _CALL, "wrong", "own-call"),
    ("train", _T_START, "empty", None),
    ("train", _t(zero="    net.zero_grad()\n"), "correct", None),
    ("train", _t(step="    with torch.no_grad():\n        for p in net.parameters():\n            p.sub_(lr * p.grad)\n"), "correct", None),
    ("train", _t(zero=""), "wrong", "no-zero"),
    ("train", _t(step="    with torch.no_grad():\n        for p in net.parameters():\n            p += lr * p.grad\n"), "wrong", "uphill"),
    ("train", _t(step="    with torch.no_grad():\n        for p in net.parameters():\n            p = p - lr * p.grad\n"), "wrong", "rebind"),
    ("train", _t(step="    for p in net.parameters():\n        p -= lr * p.grad\n"), "error", "in-place"),
    ("train", _t(back=""), "error", "no-backward"),
    ("train", _t(zero="    for p in net.parameters():\n        p.grad.zero_()\n"), "error", "zero-none"),
    ("train", _t(zero="    for p in net.parameters():\n        p.grad = None\n        loss.backward()\n", back=""), "error", "backward-twice"),
    ("train", _t(step="    with torch.no_grad():\n        for p in net.parameters():\n            p -= 0.1 * p.grad\n"), "almost", "other-value"),
]
