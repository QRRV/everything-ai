# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-activation-stats: dead_share, the share of the neurons of a layer whose z is 0 or below
for every example, and dead_per_layer, the same for a list of layers, measured with forward hooks. And the data of
the picture: an MLP with three hidden layers for the handwritten digits of QMNIST, trained for one epoch with a good
and with a far too large learning rate, as in the worked example."""
import functools
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import paths
from everything_ai.check import ClassBox, FunctionBox, Mistake, run_code

DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md


@functools.lru_cache(maxsize=1)
def tensors():
    d = np.load(paths.ROOT / DATA)
    return (torch.tensor(d["train_images"], dtype=torch.float32) / 255, torch.tensor(d["train_labels"], dtype=torch.int64),
            torch.tensor(d["test_images"], dtype=torch.float32) / 255, torch.tensor(d["test_labels"], dtype=torch.int64))


class DeepMLP(nn.Module):
    """The model of the worked example: three hidden layers of 128 neurons with ReLU, and 10 logits."""

    def __init__(self):
        super().__init__()
        self.h1 = nn.Linear(784, 128)
        self.h2 = nn.Linear(128, 128)
        self.h3 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        return self.out(torch.relu(self.h3(torch.relu(self.h2(torch.relu(self.h1(x)))))))


@functools.lru_cache(maxsize=2)
def trained(lr: float) -> DeepMLP:
    """The model after one epoch of SGD on the 6,000 training digits, with the seeds of the worked example."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)   # faster on these small layers, and the numbers no longer depend on the number of cores
    try:
        X, y, _, _ = tensors()
        torch.manual_seed(0)
        model = DeepMLP()
        loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        torch.manual_seed(2)
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return model
    finally:
        torch.set_num_threads(threads)


def third_layer_z(lr: float) -> torch.Tensor:
    """The z's of the third hidden layer for the 1,000 test digits, one row per digit, after training with `lr`."""
    model = trained(lr)
    _, _, X_test, _ = tensors()
    with torch.no_grad():
        return model.h3(torch.relu(model.h2(torch.relu(model.h1(X_test)))))


def z_picture(labels: dict) -> str:
    """Two activation histograms: the z's of the third hidden layer for the 1,000 test digits, after one epoch with
    a learning rate of 0.3 and of 2.0. Each bar is the share of all z's in a bin 0.25 wide."""
    from everything_ai import chart
    panels = []
    edges = np.arange(-2.5, 4.51, 0.25)
    for lr, key in ((0.3, "good"), (2.0, "large")):
        z = third_layer_z(lr)
        dead = int((z <= 0).all(dim=0).sum())
        counts, _ = np.histogram(np.clip(z.flatten().numpy(), -2.49, 4.49), bins=edges)
        p = chart.Panel((-2.5, 4.5), (0, 0.8), title=labels[key].format(dead=dead), xlabel=labels["x"],
                        ylabel=labels["y"] if not panels else "")
        for a, c in zip(edges[:-1], counts / z.numel()):
            if c > 0:
                p.rect(float(a) + 0.02, 0, float(a) + 0.23, float(c), "hidden")
        p.line([0, 0], [0, 0.78], "ink-3", 1.2, dashed=True)
        p.text(0.2, 0.62, labels["zero"], "ink-2", "start")
        panels.append(p)
    return chart.figure(panels, alt=labels["alt"], width=330, height=290)


# The lesson is done when these are correct. "what-dead-means" is the choice question in the lesson.
REQUIRED = ["what-dead-means", "dead-share", "dead-per-layer"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _num(v) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError, RuntimeError):
        return repr(v)
    text = f"{round(x, 4) + 0.0:.4f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _close(got, want) -> bool:
    try:
        return abs(float(got) - float(want)) < 1e-6
    except (TypeError, ValueError, RuntimeError):
        return False


class _Z:
    """A tensor for a check call, shown the way the lesson writes one: torch.tensor([...]). The learner's function
    gets a plain tensor."""

    def __init__(self, rows):
        self.rows = rows

    def __repr__(self):
        return f"torch.tensor({self.rows!r})"

    def __deepcopy__(self, memo):
        return torch.tensor(self.rows, dtype=torch.float32)


# --- exercise 2: the share of dead neurons -----------------------------------------------------------------------------------

_D_HEAD = "import torch\n\n\ndef dead_share(Z):\n"
_D_START = _D_HEAD + "    # the share of the neurons (columns) whose z is 0 or below for every example (row)\n    ...\n"


def _d(body: str = "(Z <= 0).all(dim=0).float().mean().item()") -> str:
    return _D_HEAD + f"    return {body}\n"


# --- exercise 3: per layer, with forward hooks ---------------------------------------------------------------------------------

_P_HEAD = "import torch\n\n\ndef dead_per_layer(model, layers, x):\n"
_P_START = _P_HEAD + "    # for each layer in layers, in that order: the share of its neurons whose z is 0 or below for every example of x\n    ...\n"


def _p(keep: str = "zs[module] = output", remove: bool = True, share: str = "(zs[layer] <= 0).all(dim=0).float().mean().item()") -> str:
    return (_P_HEAD + "    zs = {}\n\n    def keep(module, inputs, output):\n" + f"        {keep}\n\n"
            "    handles = []\n    for layer in layers:\n        handles.append(layer.register_forward_hook(keep))\n"
            "    with torch.no_grad():\n        model(x)\n"
            + ("    for handle in handles:\n        handle.remove()\n" if remove else "")
            + "    shares = []\n    for layer in layers:\n" + f"        shares.append({share})\n    return shares\n")


# The hooks run in the order the model runs its layers: a list keeps that order, not the one of `layers`.
_P_LIST = (_P_HEAD + "    zs = []\n\n    def keep(module, inputs, output):\n        zs.append(output)\n\n"
           "    handles = []\n    for layer in layers:\n        handles.append(layer.register_forward_hook(keep))\n"
           "    with torch.no_grad():\n        model(x)\n    for handle in handles:\n        handle.remove()\n"
           "    shares = []\n    for z in zs:\n        shares.append((z <= 0).all(dim=0).float().mean().item())\n    return shares\n")

# A model as in the worked example, an nn.Module, so that code which treats it as one (model.eval(), model.zero_grad()) runs.
_NET = ("from torch import nn\n\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
        "        self.h1 = nn.Linear(4, 6)\n        self.h2 = nn.Linear(6, 5)\n\n    def forward(self, x):\n"
        "        return self.h2(torch.relu(self.h1(x)))\n\n\ntorch.manual_seed({seed})\nmodel = Net()\nx = torch.randn({n}, 4)\n")
_SCENARIOS = [
    _NET.format(seed=1, n=4) + "dead_per_layer(model, [model.h1, model.h2], x)",
    _NET.format(seed=6, n=4) + "dead_per_layer(model, [model.h2, model.h1], x)",
    _NET.format(seed=11, n=3) + "dead_per_layer(model, [model.h2], x)",
]


def _hooks_left(got, text) -> bool:
    """After the first scenario, a layer still has hooks on it."""
    run = run_code(text + "\n\n" + _SCENARIOS[0] + "\nlen(model.h1._forward_hooks) + len(model.h2._forward_hooks)")
    return run.error is None and run.value != 0


def _same_list(got, want) -> bool:
    try:
        return len(got) == len(want) and all(abs(float(g) - float(w)) < 1e-6 for g, w in zip(got, want))
    except (TypeError, ValueError, RuntimeError):
        return False


EXERCISES = {
    "dead-share": FunctionBox(
        name="dead-share",
        function="dead_share",
        start=_D_START,
        solution=_d(),
        visible=[(_Z([[1.0, -2.0, 0.0], [-1.0, -0.5, 2.0]]),)],
        hidden=[(_Z([[0.0, -1.0], [0.0, 3.0], [0.0, -2.0]]),), (_Z([[-1.0, -1.0, -1.0, 2.0]]),), (_Z([[0.5, 1.0], [2.0, 0.1]]),)],
        show=_num,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("bool-mean", syntax=r"(?:\.all\([^)]*\)|<=?\s*0\))\s*\.mean\(",
                    en="`.mean()` needs numbers, not `True` and `False`. Turn them into 1.0 and 0.0 first with `.float()`, as with the accuracy.",
                    nl="`.mean()` heeft getallen nodig, geen `True` en `False`. Maak er eerst 1.0 en 0.0 van met `.float()`, zoals bij de accuracy."),
            Mistake("any", wrong=_d("(Z <= 0).any(dim=0).float().mean().item()"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you count a neuron that is 0 or below for some example. A neuron is dead only when it is 0 or below for every example: `.all(dim=0)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt een neuron dat bij een of ander voorbeeld 0 of lager is. Een neuron is pas dead als het bij elk voorbeeld 0 of lager is: `.all(dim=0)`."),
            Mistake("all-zs", wrong=_d("(Z <= 0).float().mean().item()"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the share of all z's that are 0 or below, over examples and neurons together. The question is per neuron: in which columns is every number 0 or below?",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is het deel van alle z's dat 0 of lager is, over voorbeelden en neuronen samen. De vraag is per neuron: in welke kolommen is elk getal 0 of lager?"),
            Mistake("rows", wrong=_d("(Z <= 0).all(dim=1).float().mean().item()"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `dim=1` goes along each row, one example. A neuron is a column: go down each column with `dim=0`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `dim=1` gaat langs elke rij, één voorbeeld. Een neuron is een kolom: loop elke kolom af met `dim=0`."),
            Mistake("strict", wrong=_d("(Z < 0).all(dim=0).float().mean().item()"), state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: a z of exactly 0 also gives 0 after ReLU, and no gradient. Use `<= 0`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: een z van precies 0 geeft na ReLU ook 0, en geen gradient. Gebruik `<= 0`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: per column, whether every z in it is 0 or below, and the share of those columns.",
                      "nl": "Vervang de `...` door de body van de functie: per kolom of elke z erin 0 of lager is, en het deel van die kolommen."},
            "arguments": {"en": "The check calls `{call}`, with one tensor, and the `def` line of your function has another number of names between its parentheses. Keep `(Z)`.",
                          "nl": "De check roept `{call}` aan, met één tensor, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(Z)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Go down each column by hand: is every number in it 0 or below? Then: how many columns is that, out of all of them?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Loop elke kolom met de hand af: is elk getal erin 0 of lager? En dan: hoeveel kolommen zijn dat, van allemaal?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any number of examples and neurons, also with z's of exactly 0?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elk aantal voorbeelden en neuronen, ook met z's van precies 0?"},
            "correct": {"en": "Your function is right in all {n} cases: the share of the neurons that give 0 for every example.",
                        "nl": "Je functie klopt in alle {n} gevallen: het deel van de neuronen dat bij elk voorbeeld 0 geeft."},
        },
    ),
    "dead-per-layer": ClassBox(
        name="dead-per-layer",
        start=_P_START,
        solution=_p(),
        scenarios=_SCENARIOS,
        compare=_same_list,
        digits=4,
        first=[_DOTS],
        form=[Mistake("hooks-left", matches=_hooks_left, state="almost",
                      en="Your shares are right, but your hooks stay on the layers: every later run of the model calls them again, also during training. Keep what `register_forward_hook` gives back, and call `.remove()` on it once the model has run.",
                      nl="Je delen kloppen, maar je hooks blijven aan de layers hangen: elke latere run van het model roept ze opnieuw aan, ook tijdens het trainen. Bewaar wat `register_forward_hook` teruggeeft, en roep er `.remove()` op aan als het model gedraaid heeft.")],
        mistakes=[
            Mistake("removed-early", syntax=r"\.remove\(\)(?=[\s\S]*\bmodel\()",
                    en="Your hooks come off before the model runs, so they keep nothing. Run the model on `x` first, then call `.remove()` on each handle.",
                    nl="Je hooks gaan eraf voordat het model draait, dus ze bewaren niets. Draai eerst het model op `x`, en roep daarna `.remove()` aan op elke handle."),
            Mistake("model-order", wrong=_P_LIST, state="almost",
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: the shares are in the order the model runs its layers, `model.h1` before `model.h2`, and not in the order of `layers`. The hooks run in the model's order. Keep each output under its own layer, in a dict, and read them in the order of `layers`.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: de delen staan in de volgorde waarin het model zijn layers draait, `model.h1` voor `model.h2`, en niet in de volgorde van `layers`. De hooks draaien in de volgorde van het model. Bewaar elke output onder zijn eigen layer, in een dict, en lees ze in de volgorde van `layers`."),
            Mistake("inputs", wrong=_p(keep="zs[module] = inputs[0]"),
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: you keep what goes into each layer, `inputs`. The z's of a layer are what comes out of it: `output`.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: je bewaart wat elke layer in gaat, `inputs`. De z's van een layer zijn wat eruit komt: `output`."),
            Mistake("dict", matches=lambda got, text: isinstance(got, dict),
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get a dict, and it should be the list `{want}`: one share for each layer, in the order of `layers`. Go through `layers` and append the share of each layer to a list.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je een dict, en dat moet de lijst `{want}` zijn: één deel voor elke layer, in de volgorde van `layers`. Ga door `layers` en voeg het deel van elke layer toe aan een lijst."),
            Mistake("any", wrong=_p(share="(zs[layer] <= 0).any(dim=0).float().mean().item()"),
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: you count a neuron that is 0 or below for some example. Dead means 0 or below for every example: `.all(dim=0)`, as in exercise 2.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: je telt een neuron dat bij een of ander voorbeeld 0 of lager is. Dead betekent 0 of lager bij elk voorbeeld: `.all(dim=0)`, zoals in oefening 2."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: a hook on each layer that keeps its output, one run of the model on `x`, the hooks removed, and the share of dead neurons of each layer, in the order of `layers`.",
                      "nl": "Vervang de `...`: een hook op elke layer die zijn output bewaart, één run van het model op `x`, de hooks verwijderd, en het deel dead neurons van elke layer, in de volgorde van `layers`."},
            "wrong": {"en": "With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does every layer in `layers` get one hook that keeps its output, does the model run once on `x`, and is the share per layer the one of exercise 2?",
                      "nl": "Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Krijgt elke layer in `layers` één hook die zijn output bewaart, draait het model één keer op `x`, en is het deel per layer dat uit oefening 2?"},
            "almost": {"en": "Right for the first network. But with these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Are the shares in the order of `layers`, whatever that order is, and one for each layer in it?",
                       "nl": "Goed voor het eerste netwerk. Maar met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Staan de delen in de volgorde van `layers`, welke volgorde dat ook is, en één voor elke layer daarin?"},
            "correct": {"en": "The share of dead neurons of every layer, in the order you asked for, measured with forward hooks that you take off again.",
                        "nl": "Het deel dead neurons van elke layer, in de volgorde die je vroeg, gemeten met forward hooks die je er weer af haalt."},
        },
    ),
}

CASES = [
    ("dead-share", _D_START, "empty", None),
    ("dead-share", "", "empty", None),
    ("dead-share", _d(), "correct", None),
    ("dead-share", _D_HEAD + "    dead = (Z <= 0).all(dim=0)\n    return dead.sum().item() / Z.shape[1]\n", "correct", None),
    ("dead-share", _D_HEAD + "    return (Z.max(dim=0).values <= 0).float().mean()\n", "correct", None),
    ("dead-share", _D_HEAD + "    count = 0\n    for j in range(Z.shape[1]):\n        if (Z[:, j] <= 0).all():\n            count += 1\n    return count / Z.shape[1]\n", "correct", None),
    ("dead-share", _D_HEAD + "    return torch.relu(Z).sum(dim=0).eq(0).float().mean().item()\n", "correct", None),
    ("dead-share", _d("(Z <= 0).any(dim=0).float().mean().item()"), "wrong", "any"),
    ("dead-share", _d("(Z <= 0).float().mean().item()"), "wrong", "all-zs"),
    ("dead-share", _d("(Z <= 0).all(dim=1).float().mean().item()"), "wrong", "rows"),
    ("dead-share", _d("(Z < 0).all(dim=0).float().mean().item()"), "almost", "strict"),
    ("dead-share", _d("(Z <= 0).all(dim=0).mean().item()"), "error", "bool-mean"),
    ("dead-share", _d("(Z <= 0).all(dim=0).sum().item()"), "wrong", None),
    ("dead-share", _D_HEAD + "    ...\n    return 0\n", "wrong", "dots-left"),
    ("dead-share", "def dead_share(Z, layer):\n    return 0\n", "error", "arguments"),
    ("dead-share", _D_HEAD + "    return (Z <= 0).all(dim=0).float().mean().item(\n", "error", None),
    ("dead-per-layer", _P_START, "empty", None),
    ("dead-per-layer", "", "empty", None),
    ("dead-per-layer", _p(), "correct", None),
    ("dead-per-layer", _p(keep="zs[module] = torch.relu(output)"), "correct", None),
    # without a dict: one run per layer, each with only its own hook
    ("dead-per-layer", _P_HEAD + "    shares = []\n    for layer in layers:\n        zs = []\n\n        def keep(module, inputs, output):\n            zs.append(output)\n\n"
     "        handle = layer.register_forward_hook(keep)\n        model(x)\n        handle.remove()\n"
     "        shares.append((zs[0] <= 0).all(dim=0).float().mean().item())\n    return shares\n", "correct", None),
    ("dead-per-layer", _p(remove=False), "almost", "hooks-left"),
    ("dead-per-layer", _P_HEAD + "    zs = {}\n\n    def keep(module, inputs, output):\n        zs[module] = output\n\n"
     "    handles = []\n    for layer in layers:\n        handles.append(layer.register_forward_hook(keep))\n"
     "    for handle in handles:\n        handle.remove()\n    with torch.no_grad():\n        model(x)\n"
     "    shares = []\n    for layer in layers:\n        shares.append((zs[layer] <= 0).all(dim=0).float().mean().item())\n    return shares\n", "error", "removed-early"),
    ("dead-per-layer", _P_LIST, "almost", "model-order"),
    ("dead-per-layer", _p(keep="zs[module] = inputs[0]"), "wrong", "inputs"),
    ("dead-per-layer", _p(share="(zs[layer] <= 0).any(dim=0).float().mean().item()"), "wrong", "any"),
    ("dead-per-layer", _p(share="(zs[layer] <= 0).float().mean().item()"), "wrong", None),
    ("dead-per-layer", _p().replace("    shares = []\n    for layer in layers:\n        shares.append(", "    shares = {}\n    for layer in layers:\n        shares[layer] = ("), "wrong", "dict"),
    # code that treats the model as an nn.Module, as in Reading loss curves: eval mode, and back
    ("dead-per-layer", _p().replace("    with torch.no_grad():\n        model(x)\n", "    model.eval()\n    with torch.no_grad():\n        model(x)\n    model.train()\n"), "correct", None),
    ("dead-per-layer", _P_HEAD + "    return [0.0 for layer in layers]\n", "wrong", None),
    ("dead-per-layer", _P_HEAD + "    handles = layer.register_forward_hook(keep)\n", "error", None),
]
