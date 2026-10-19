# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson opt-adam: adam_steps, the steps that Adam takes for a list of gradients of one parameter, and
AdamOptimizer, Adam for PyTorch tensors. And the data of the picture: one step of SGD and one of Adam for the weights
of the hidden layer of an MLP for the handwritten digits of QMNIST, step size against gradient size."""
import functools
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import paths
from everything_ai.check import ClassBox, FunctionBox, Mistake

DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md
AT_STEP = 51    # the step of the picture
SAMPLE = 1500   # weights shown in the picture


class Net(nn.Module):
    """The MLP of the digits: 784 pixels, 64 hidden neurons with ReLU, 10 logits."""

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        return self.out(torch.relu(self.hidden(x)))


@functools.lru_cache(maxsize=2)
def one_step(optimizer: str):
    """The gradient and the step of every weight of the hidden layer at step AT_STEP of training, with SGD (lr 0.1)
    or Adam (lr 0.001), as two arrays of sizes."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)   # faster on these small layers, and the numbers no longer depend on the number of cores
    try:
        d = np.load(paths.ROOT / DATA)
        X = torch.tensor(d["train_images"], dtype=torch.float32) / 255
        y = torch.tensor(d["train_labels"], dtype=torch.int64)
        torch.manual_seed(0)
        model = Net()
        if optimizer == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
        loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
        loss_fn = nn.CrossEntropyLoss()
        torch.manual_seed(2)
        for k, (xb, yb) in enumerate(loader, start=1):
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            if k == AT_STEP:
                before = model.hidden.weight.detach().clone()
                grad = model.hidden.weight.grad.detach().abs().flatten().numpy()
                opt.step()
                step = (model.hidden.weight.detach() - before).abs().flatten().numpy()
                return grad, step
            opt.step()
        raise ValueError("too few mini-batches")
    finally:
        torch.set_num_threads(threads)


def step_picture(labels: dict) -> str:
    """Step size against gradient size for a sample of the weights of the hidden layer, with SGD and with Adam."""
    from everything_ai import chart
    rng = np.random.default_rng(0)
    panels = []
    for key in ("sgd", "adam"):
        grad, step = one_step(key)
        keep = np.flatnonzero(grad > 1e-6)
        pick = rng.choice(keep, size=min(SAMPLE, len(keep)), replace=False)
        p = chart.Panel((1e-6, 1e-1), (1e-7, 1e-2), title=labels[key], xlabel=labels["x"], ylabel=labels["y"] if not panels else "",
                        xlog=True, ylog=True)
        p.points(grad[pick].tolist(), step[pick].tolist(), "param", 1.6)
        panels.append(p)
    return chart.figure(panels, alt=labels["alt"], width=330, height=300)


# The lesson is done when these are correct. "first-step" is the choice question in the lesson.
REQUIRED = ["first-step", "adam-steps", "adam-optimizer"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _num(x) -> str:
    x = float(x)
    if not math.isfinite(x):
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    if x == 0:
        return "0"
    return np.format_float_positional(x, precision=4, unique=False, fractional=False, trim="-").rstrip(".")


def _show(v) -> str:
    try:
        values = [float(x) for x in v]
    except (TypeError, ValueError):
        return repr(v)
    return "[" + ", ".join(_num(x) for x in values) + "]"


def _close(got, want) -> bool:
    try:
        g, w = [float(x) for x in got], [float(x) for x in want]
    except (TypeError, ValueError):
        return False
    return len(g) == len(w) and all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-9) or (math.isnan(a) and math.isnan(b)) for a, b in zip(g, w))


# --- exercise 2: the steps of Adam ----------------------------------------------------------------------------------------

_S_HEAD = "def adam_steps(gs, lr, beta1, beta2):\n"
_S_START = (_S_HEAD + "    # for each gradient g in gs, at step t = 1, 2, ...: m and s, both corrected, and the step\n"
            "    # lr * m_hat / (s_hat ** 0.5 + 1e-8); give back the list of steps\n    ...\n")


def _s(m: str = "beta1 * m + (1 - beta1) * g", s: str = "beta2 * s + (1 - beta2) * g ** 2", m_hat: str = "m / (1 - beta1 ** t)",
       s_hat: str = "s / (1 - beta2 ** t)", step: str = "lr * m_hat / (s_hat ** 0.5 + 1e-8)") -> str:
    return (_S_HEAD + "    m = 0.0\n    s = 0.0\n    steps = []\n    t = 0\n    for g in gs:\n        t = t + 1\n"
            f"        m = {m}\n        s = {s}\n        m_hat = {m_hat}\n        s_hat = {s_hat}\n        steps.append({step})\n    return steps\n")


# --- exercise 3: Adam for PyTorch ------------------------------------------------------------------------------------------

_A_HEAD = "import torch\n\n\nclass AdamOptimizer:\n"
_A_START = (_A_HEAD + "    def __init__(self, params, lr, beta1, beta2):\n"
            "        # keep the parameters and the numbers, a step counter t = 0, and one m and one s of zeros for each parameter\n        ...\n\n"
            "    def zero_grad(self):\n        for p in self.params:\n            p.grad = None\n\n"
            "    def step(self):\n        # t one up; for each parameter: m, s, both corrected, and the parameter minus the step\n        ...\n")


def _a(m: str = "self.beta1 * self.ms[i] + (1 - self.beta1) * p.grad", s: str = "self.beta2 * self.ss[i] + (1 - self.beta2) * p.grad ** 2",
       correct: bool = True, counter: str = "        self.t = self.t + 1\n") -> str:
    hats = ("                m_hat = self.ms[i] / (1 - self.beta1 ** self.t)\n                s_hat = self.ss[i] / (1 - self.beta2 ** self.t)\n"
            if correct else "                m_hat = self.ms[i]\n                s_hat = self.ss[i]\n")
    return (_A_HEAD + "    def __init__(self, params, lr, beta1, beta2):\n        self.params = params\n        self.lr = lr\n"
            "        self.beta1 = beta1\n        self.beta2 = beta2\n        self.t = 0\n        self.ms = []\n        self.ss = []\n"
            "        for p in params:\n            self.ms.append(torch.zeros_like(p))\n            self.ss.append(torch.zeros_like(p))\n\n"
            "    def zero_grad(self):\n        for p in self.params:\n            p.grad = None\n\n"
            "    def step(self):\n" + counter + "        with torch.no_grad():\n            for i in range(len(self.params)):\n"
            "                p = self.params[i]\n" + f"                self.ms[i] = {m}\n                self.ss[i] = {s}\n" + hats
            + "                p -= self.lr * m_hat / (s_hat ** 0.5 + 1e-8)\n")


_TRAIN = ("w = torch.tensor({w}, requires_grad=True)\nb = torch.tensor([0.0], requires_grad=True)\n"
          "x = torch.tensor({x})\ny = torch.tensor({y})\nopt = AdamOptimizer([w, b], lr={lr}, beta1={b1}, beta2={b2})\n"
          "for _ in range({steps}):\n    loss = ((x @ w + b - y) ** 2).mean()\n    opt.zero_grad()\n    loss.backward()\n    opt.step()\n"
          "w.tolist() + b.tolist()")
_X3 = [[1.0, 0.5, -1.0], [0.0, 2.0, 1.0], [-1.0, 1.0, 0.5], [2.0, -0.5, 0.0]]
_A_SCENARIOS = [
    _TRAIN.format(w=[0.5, -0.5, 1.0], x=_X3, y=[1.0, -1.0, 0.5, 2.0], lr=0.1, b1=0.9, b2=0.999, steps=1),
    _TRAIN.format(w=[0.5, -0.5, 1.0], x=_X3, y=[1.0, -1.0, 0.5, 2.0], lr=0.1, b1=0.9, b2=0.999, steps=5),
    _TRAIN.format(w=[1.0, 2.0], x=[[1.0, 0.0], [0.5, 1.0], [-1.0, 2.0]], y=[0.0, 1.0, 3.0], lr=0.05, b1=0.8, b2=0.99, steps=8),
]

EXERCISES = {
    "adam-steps": FunctionBox(
        name="adam-steps",
        function="adam_steps",
        start=_S_START,
        solution=_s(),
        visible=[([4.0, 1.0, -2.0], 0.1, 0.9, 0.999)],
        hidden=[([0.001, 0.001, 0.001], 0.1, 0.9, 0.999), ([1.0, -1.0, 1.0, -1.0], 0.01, 0.5, 0.9), ([100.0], 0.001, 0.9, 0.999),
                ([0.5, 2.0, 0.0, 1.0, 3.0], 0.2, 0.8, 0.99), ([0.0, 0.5, -0.5], 0.1, 0.9, 0.999)],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("no-correction", wrong=_s(m_hat="m", s_hat="s"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: without the correction, `m` and `s` start far too small, because they start at 0. Divide them by `1 - beta1 ** t` and `1 - beta2 ** t`, with `t` the number of the step.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: zonder de correctie beginnen `m` en `s` veel te klein, omdat ze op 0 beginnen. Deel ze door `1 - beta1 ** t` en `1 - beta2 ** t`, met `t` het nummer van de stap."),
            Mistake("no-root", wrong=_s(step="lr * m_hat / (s_hat + 1e-8)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you divide by `s_hat`, the mean of the squared gradient. Divide by its square root, `s_hat ** 0.5`, which has the size of the gradient itself.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je deelt door `s_hat`, het gemiddelde van de gekwadrateerde gradient. Deel door de wortel daaruit, `s_hat ** 0.5`, die de grootte van de gradient zelf heeft."),
            Mistake("momentum-sum", wrong=_s(m="beta1 * m + g"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your `m` adds the whole gradient, as momentum does. Adam keeps an average: `beta1 * m + (1 - beta1) * g`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je `m` telt de hele gradient erbij op, zoals momentum doet. Adam houdt een gemiddelde bij: `beta1 * m + (1 - beta1) * g`."),
            Mistake("square-of-mean", wrong=_s(s="beta2 * s + (1 - beta2) * g", step="lr * m_hat / (abs(s_hat) + 1e-8)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your `s` averages the gradient itself, not its square. Square it first: `(1 - beta2) * g ** 2`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je `s` middelt de gradient zelf, niet zijn kwadraat. Kwadrateer hem eerst: `(1 - beta2) * g ** 2`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: `m` and `s` start at 0, and for each gradient, the new `m` and `s`, their corrections and the step into a list.",
                      "nl": "Vervang de `...` door de body van de functie: `m` en `s` beginnen op 0, en voor elke gradient de nieuwe `m` en `s`, hun correcties en de stap in een lijst."},
            "arguments": {"en": "The check calls `{call}`, with a list of gradients, a learning rate and two betas, and the `def` line of your function has another number of names between its parentheses. Keep `(gs, lr, beta1, beta2)`.",
                          "nl": "De check roept `{call}` aan, met een lijst gradients, een learning rate en twee bèta's, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(gs, lr, beta1, beta2)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Follow the recipe one gradient at a time: the first step is `lr` times `m_hat` divided by the root of `s_hat`; after that, `m`, `s` and `t` carry on from the step before.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Volg het recept één gradient tegelijk: de eerste stap is `lr` keer `m_hat` gedeeld door de wortel uit `s_hat`; daarna gaan `m`, `s` en `t` verder waar de vorige stap ophield."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for tiny and huge gradients, for a gradient of 0, and for other betas?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor piepkleine en enorme gradients, voor een gradient van 0, en voor andere bèta's?"},
            "correct": {"en": "Your function is right in all {n} cases: steps of about the learning rate, whether the gradient is 0.001 or 100.",
                        "nl": "Je functie klopt in alle {n} gevallen: stappen van ongeveer de learning rate, of de gradient nu 0,001 is of 100."},
        },
    ),
    "adam-optimizer": ClassBox(
        name="adam-optimizer",
        start=_A_START,
        solution=_a(),
        scenarios=_A_SCENARIOS,
        compare=_close,
        digits=4,
        first=[_DOTS],
        mistakes=[
            Mistake("no-grad-missing", error=lambda e: isinstance(e, RuntimeError) and "leaf Variable that requires grad" in str(e),
                    en="PyTorch stops because you change a parameter that requires a gradient while it is recording. Put the updates of `step` inside `with torch.no_grad():`, as in your `MomentumSGD`.",
                    nl="PyTorch stopt omdat je een parameter verandert die een gradient nodig heeft, terwijl PyTorch opneemt. Zet de updates van `step` binnen `with torch.no_grad():`, zoals in je `MomentumSGD`."),
            Mistake("no-correction", wrong=_a(correct=False),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: without the correction, `m` and `s` start far too small. Divide them by `1 - self.beta1 ** self.t` and `1 - self.beta2 ** self.t`.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: zonder de correctie beginnen `m` en `s` veel te klein. Deel ze door `1 - self.beta1 ** self.t` en `1 - self.beta2 ** self.t`."),
            Mistake("no-counter", wrong=_a(counter=""),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does `self.t` go up by one in every step, before the corrections use it?",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Gaat `self.t` bij elke stap één omhoog, voordat de correcties hem gebruiken?"),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: in `__init__` the parameters, the numbers, `self.t = 0` and the lists `self.ms` and `self.ss`; in `step`, `self.t` one up and, inside `torch.no_grad()`, the update of each parameter.",
                      "nl": "Vervang de twee `...`: in `__init__` de parameters, de getallen, `self.t = 0` en de lijsten `self.ms` en `self.ss`; in `step` `self.t` één omhoog en, binnen `torch.no_grad()`, de update van elke parameter."},
            "wrong": {"en": "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. After one step, each parameter should have moved by about `lr`, against the sign of its gradient: are `m` and `s` corrected with `self.t = 1`?",
                      "nl": "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Na één stap moet elke parameter ongeveer `lr` verschoven zijn, tegen het teken van zijn gradient in: worden `m` en `s` gecorrigeerd met `self.t = 1`?"},
            "almost": {"en": "Right after one step. But with these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Are `m` and `s` kept between steps, each as an average with its own beta?",
                       "nl": "Goed na één stap. Maar met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Worden `m` en `s` tussen de stappen bewaard, elk als gemiddelde met zijn eigen bèta?"},
            "correct": {"en": "Your optimizer takes the same steps as `torch.optim.Adam`, in all {n} runs.",
                        "nl": "Je optimizer zet dezelfde stappen als `torch.optim.Adam`, in alle {n} runs."},
        },
    ),
}

CASES = [
    ("adam-steps", _S_START, "empty", None),
    ("adam-steps", _s(), "correct", None),
    ("adam-steps", _s(step="lr * m_hat / (s_hat ** 0.5 + 0.00000001)"), "correct", None),
    ("adam-steps", _s(m_hat="m", s_hat="s"), "wrong", "no-correction"),
    ("adam-steps", _s(step="lr * m_hat / (s_hat + 1e-8)"), "wrong", "no-root"),
    ("adam-steps", _s(m="beta1 * m + g"), "wrong", "momentum-sum"),
    ("adam-steps", _s(s="beta2 * s + (1 - beta2) * g", step="lr * m_hat / (abs(s_hat) + 1e-8)"), "wrong", "square-of-mean"),
    ("adam-steps", "def adam_steps(gs, lr):\n    return []\n", "error", "arguments"),
    ("adam-optimizer", _A_START, "empty", None),
    ("adam-optimizer", _a(), "correct", None),
    ("adam-optimizer", _a(correct=False), "wrong", "no-correction"),
    ("adam-optimizer", _a(counter=""), "wrong", "no-counter"),
    ("adam-optimizer", _a().replace("        with torch.no_grad():\n", "        if True:\n"), "error", "no-grad-missing"),
]
