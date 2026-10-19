# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson opt-momentum: momentum_steps, the steps that momentum takes for a list of gradients of one
parameter, and MomentumSGD, an optimizer with momentum for PyTorch tensors. And the picture: gradient descent with
and without momentum in a narrow valley, the loss 0.5 (w1² + 100 w2²)."""
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai.check import ClassBox, FunctionBox, Mistake

START = (10.0, 1.0)
STEEP = 100.0   # the valley is 100 times as steep across (w2) as along (w1)


def valley_loss(w) -> float:
    return 0.5 * (w[0] ** 2 + STEEP * w[1] ** 2)


def valley_path(lr: float, beta: float, steps: int) -> list[tuple[float, float]]:
    """Where (w1, w2) is after each step, from START, with v = beta * v + g and w = w - lr * v."""
    w = np.array(START)
    v = np.zeros(2)
    out = [tuple(w)]
    for _ in range(steps):
        g = np.array([w[0], STEEP * w[1]])
        v = beta * v + g
        w = w - lr * v
        out.append((float(w[0]), float(w[1])))
    return out


def steps_needed(lr: float, beta: float, below: float = 0.001, limit: int = 1000) -> int:
    path = valley_path(lr, beta, limit)
    return next(i for i, w in enumerate(path) if valley_loss(w) < below)


def path_picture(labels: dict) -> str:
    """Two panels: 40 steps of plain gradient descent (lr 0.019) and of momentum (lr 0.01, beta 0.9) over the
    contour lines of the valley."""
    from everything_ai import chart
    panels = []
    t = np.linspace(0, 2 * math.pi, 120)
    for lr, beta, key in ((0.019, 0.0, "plain"), (0.01, 0.9, "momentum")):
        p = chart.Panel((-11, 11), (-1.25, 1.25), title=labels[key], xlabel="w₁", ylabel="w₂" if not panels else "")
        for level in (0.5, 2, 8, 20, 50):
            r = math.sqrt(2 * level)
            p.line((r * np.cos(t)).tolist(), (r / math.sqrt(STEEP) * np.sin(t)).tolist(), "loss", 1.0)
        path = valley_path(lr, beta, 40)
        p.line([w[0] for w in path], [w[1] for w in path], "ink-3", 1.0)
        p.points([w[0] for w in path], [w[1] for w in path], "param", 2.6)
        p.points([0], [0], "loss", 3.5, hollow=True)
        end = path[-1]
        p.text(0, -1.1, labels["after"].format(loss=labels["number"](valley_loss(end), 2)), "loss", "middle")
        panels.append(p)
    return chart.figure(panels, alt=labels["alt"], width=330, height=280)


# The lesson is done when these are correct. "zigzag" is the choice question in the lesson.
REQUIRED = ["zigzag", "momentum-steps", "momentum-sgd"]

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
    text = f"{round(x, 4) + 0.0:.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _show(v) -> str:
    """A list of numbers as [0.1, 0.19], and one number, the learner's answer instead of a list, as 0.3439."""
    try:
        return _num(v) if np.ndim(v) == 0 else "[" + ", ".join(_num(x) for x in v) + "]"
    except (TypeError, ValueError):
        return repr(v)


def _close(got, want, rel_tol: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    try:
        g, w = [float(x) for x in got], [float(x) for x in want]
    except (TypeError, ValueError):
        return False
    return len(g) == len(w) and all(math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) for a, b in zip(g, w))


def _close32(got, want) -> bool:
    """For the optimizer: PyTorch calculates in float32, and `v = beta * v + lr * g` with `p -= v`, or the average
    with `lr / (1 - beta)`, takes the same steps with other roundings, about 1e-7 apart after five steps."""
    return _close(got, want, rel_tol=1e-5, abs_tol=1e-6)


# --- exercise 2: the steps of momentum ------------------------------------------------------------------------------------

_S_HEAD = "def momentum_steps(gs, lr, beta):\n"
_S_START = _S_HEAD + "    # for each gradient in gs: v = beta * v + g, and the step lr * v; give back the list of steps\n    ...\n"


def _s(update: str = "beta * v + g", step: str = "lr * v", order: str = "update-first") -> str:
    body = f"        v = {update}\n        steps.append({step})\n"
    if order == "step-first":
        body = f"        steps.append({step})\n        v = {update}\n"
    return _S_HEAD + "    v = 0.0\n    steps = []\n    for g in gs:\n" + body + "    return steps\n"


# --- exercise 3: an optimizer with momentum --------------------------------------------------------------------------------

_M_HEAD = "import torch\n\n\nclass MomentumSGD:\n"
_M_START = (_M_HEAD + "    def __init__(self, params, lr, beta):\n        # keep the parameters, lr and beta, and one v of zeros for each parameter\n        ...\n\n"
            "    def zero_grad(self):\n        for p in self.params:\n            p.grad = None\n\n"
            "    def step(self):\n        # for each parameter: v = beta * v + its gradient, then the parameter minus lr * v\n        ...\n")


def _m(update: str = "self.beta * self.vs[i] + p.grad", init_v: str = "torch.zeros_like(p)", keep: bool = True, old_first: bool = False) -> str:
    store = f"            self.vs[i] = {update}\n            p -= self.lr * self.vs[i]\n" if keep else f"            v = {update}\n            p -= self.lr * v\n"
    if old_first:
        store = f"            p -= self.lr * self.vs[i]\n            self.vs[i] = {update}\n"
    return (_M_HEAD + "    def __init__(self, params, lr, beta):\n        self.params = params\n        self.lr = lr\n        self.beta = beta\n"
            f"        self.vs = []\n        for p in params:\n            self.vs.append({init_v})\n\n"
            "    def zero_grad(self):\n        for p in self.params:\n            p.grad = None\n\n"
            "    def step(self):\n        with torch.no_grad():\n            for i in range(len(self.params)):\n                p = self.params[i]\n"
            + store.replace("            ", "                ") )


_TRAIN = ("w = torch.tensor({w}, requires_grad=True)\nb = torch.tensor([0.0], requires_grad=True)\n"
          "x = torch.tensor({x})\ny = torch.tensor({y})\nopt = MomentumSGD([w, b], lr={lr}, beta={beta})\n"
          "for _ in range({steps}):\n    loss = ((x @ w + b - y) ** 2).mean()\n    opt.zero_grad()\n    loss.backward()\n    opt.step()\n"
          "w.tolist() + b.tolist()")
_X3 = [[1.0, 0.5, -1.0], [0.0, 2.0, 1.0], [-1.0, 1.0, 0.5], [2.0, -0.5, 0.0]]
_M_SCENARIOS = [
    _TRAIN.format(w=[0.5, -0.5, 1.0], x=_X3, y=[1.0, -1.0, 0.5, 2.0], lr=0.1, beta=0.9, steps=1),
    _TRAIN.format(w=[0.5, -0.5, 1.0], x=_X3, y=[1.0, -1.0, 0.5, 2.0], lr=0.1, beta=0.9, steps=5),
    _TRAIN.format(w=[1.0, 2.0], x=[[1.0, 0.0], [0.5, 1.0], [-1.0, 2.0]], y=[0.0, 1.0, 3.0], lr=0.05, beta=0.5, steps=8),
]

EXERCISES = {
    "momentum-steps": FunctionBox(
        name="momentum-steps",
        function="momentum_steps",
        start=_S_START,
        solution=_s(),
        visible=[([1.0, 1.0, 1.0, 1.0], 0.1, 0.9)],
        hidden=[([1.0, -1.0, 1.0, -1.0, 1.0], 0.1, 0.9), ([2.0, 0.0, 0.0], 0.5, 0.5), ([0.5, 1.5, -2.0], 0.2, 0.0), ([3.0], 1.0, 0.9)],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("average", wrong=_s(update="beta * v + (1 - beta) * g"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you mix in only `1 - beta` of each gradient, as an average does. Momentum as PyTorch does it adds the whole gradient: `v = beta * v + g`, so that on a steady slope the steps grow to `lr / (1 - beta)` times the gradient.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je mengt maar `1 - beta` van elke gradient erbij, zoals een gemiddelde doet. Momentum zoals PyTorch het doet, telt de hele gradient erbij op: `v = beta * v + g`, zodat de stappen op een gelijkmatige helling groeien tot `lr / (1 - beta)` keer de gradient."),
            Mistake("old-v", wrong=_s(order="step-first"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you take the step with the old `v`, before you add the new gradient, so the first step is 0. Update `v` first, then step with it.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je zet de stap met de oude `v`, voordat je de nieuwe gradient erbij optelt, dus de eerste stap is 0. Werk eerst `v` bij, en zet dan de stap ermee."),
            Mistake("minus", wrong=_s(step="-lr * v"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your steps have a minus sign. The step is `lr * v`; the minus comes only when you take it from `w`: `w - lr * v`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je stappen hebben een minteken. De stap is `lr * v`; het min komt pas als je hem van `w` aftrekt: `w - lr * v`."),
            Mistake("beta-on-g", wrong=_s(update="beta * (v + g)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `beta` multiplies the new gradient too. Only the old `v` is multiplied by `beta`: `v = beta * v + g`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `beta` vermenigvuldigt de nieuwe gradient ook. Alleen de oude `v` wordt met `beta` vermenigvuldigd: `v = beta * v + g`."),
            Mistake("swapped", wrong=_s(update="v + beta * g"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `beta` is on the new gradient instead of on the old `v`, so nothing is ever forgotten. `v = beta * v + g`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `beta` staat bij de nieuwe gradient in plaats van bij de oude `v`, dus er wordt nooit iets vergeten. `v = beta * v + g`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: `v` starts at 0, and for each gradient, first the new `v`, then the step `lr * v` into a list.",
                      "nl": "Vervang de `...` door de body van de functie: `v` begint op 0, en voor elke gradient eerst de nieuwe `v`, dan de stap `lr * v` in een lijst."},
            "arguments": {"en": "The check calls `{call}`, with a list of gradients, a learning rate and a beta, and the `def` line of your function has another number of names between its parentheses. Keep `(gs, lr, beta)`.",
                          "nl": "De check roept `{call}` aan, met een lijst gradients, een learning rate en een beta, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(gs, lr, beta)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Work out the first two steps by hand: `v` starts at 0, then `beta * v + g` for the first gradient, times `lr`; then the same for the second.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken de eerste twee stappen met de hand uit: `v` begint op 0, dan `beta * v + g` voor de eerste gradient, keer `lr`; dan hetzelfde voor de tweede."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for gradients that change sign, for any `beta`, also 0, and for a single gradient?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor gradients die van teken wisselen, voor elke `beta`, ook 0, en voor één gradient?"},
            "correct": {"en": "Your function is right in all {n} cases: steps that grow on a steady slope and shrink when the gradient keeps changing sign.",
                        "nl": "Je functie klopt in alle {n} gevallen: stappen die groeien op een gelijkmatige helling en krimpen als de gradient steeds van teken wisselt."},
        },
    ),
    "momentum-sgd": ClassBox(
        name="momentum-sgd",
        start=_M_START,
        solution=_m(),
        scenarios=_M_SCENARIOS,
        compare=_close32,
        digits=4,
        first=[_DOTS],
        mistakes=[
            Mistake("no-grad-missing", error=lambda e: isinstance(e, RuntimeError) and "leaf Variable that requires grad" in str(e),
                    en="PyTorch stops because you change a parameter that requires a gradient while it is recording. Put the updates of `step` inside `with torch.no_grad():`, as in your training loops before you had an optimizer.",
                    nl="PyTorch stopt omdat je een parameter die een gradient nodig heeft, verandert terwijl PyTorch opneemt. Zet de updates van `step` binnen `with torch.no_grad():`, zoals in je training loops voordat je een optimizer had."),
            Mistake("not-kept", wrong=_m(keep=False), state="almost",
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: the first step is right, but after that your `v` starts from zero again in every step, which is plain gradient descent. Keep each `v` in `self.vs` between steps.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: de eerste stap klopt, maar daarna begint je `v` bij elke stap weer bij nul, en dat is gewone gradient descent. Bewaar elke `v` tussen de stappen in `self.vs`."),
            Mistake("old-v", wrong=_m(old_first=True),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: each parameter steps with its old `v`, before the new gradient is added, so the first step does nothing. Update `self.vs[i]` first, then step with it.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: elke parameter stapt met zijn oude `v`, voordat de nieuwe gradient erbij komt, dus de eerste stap doet niets. Werk eerst `self.vs[i]` bij, en stap dan ermee."),
            Mistake("average", wrong=_m(update="self.beta * self.vs[i] + (1 - self.beta) * p.grad"),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: you mix in only `1 - beta` of each gradient. As in exercise 2 and in PyTorch's SGD, add the whole gradient: `self.beta * self.vs[i] + p.grad`.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: je mengt maar `1 - beta` van elke gradient erbij. Tel, zoals in oefening 2 en in de SGD van PyTorch, de hele gradient erbij op: `self.beta * self.vs[i] + p.grad`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: in `__init__` the parameters, `lr`, `beta` and a list `self.vs` with `torch.zeros_like(p)` for each parameter; in `step`, inside `torch.no_grad()`, the new `v` and the update of each parameter.",
                      "nl": "Vervang de twee `...`: in `__init__` de parameters, `lr`, `beta` en een lijst `self.vs` met `torch.zeros_like(p)` voor elke parameter; in `step`, binnen `torch.no_grad()`, de nieuwe `v` en de update van elke parameter."},
            "wrong": {"en": "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. After one step, `v` is the gradient itself, so the step should be that of plain gradient descent: is each parameter changed by `self.lr` times its own `v`?",
                      "nl": "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Na één stap is `v` de gradient zelf, dus de stap moet die van gewone gradient descent zijn: verandert elke parameter met `self.lr` keer zijn eigen `v`?"},
            "almost": {"en": "Right after one step. But with these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Is each `v` kept between steps, multiplied by `beta`, with the new gradient added?",
                       "nl": "Goed na één stap. Maar met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Wordt elke `v` tussen de stappen bewaard, keer `beta`, met de nieuwe gradient erbij?"},
            "correct": {"en": "Your optimizer takes the same steps as `torch.optim.SGD` with `momentum`, in all {n} runs.",
                        "nl": "Je optimizer zet dezelfde stappen als `torch.optim.SGD` met `momentum`, in alle {n} runs."},
        },
    ),
}

CASES = [
    ("momentum-steps", _S_START, "empty", None),
    ("momentum-steps", _s(), "correct", None),
    ("momentum-steps", _S_HEAD + "    v = 0\n    steps = []\n    for g in gs:\n        v = beta * v + lr * g\n        steps.append(v)\n    return steps\n", "correct", None),
    ("momentum-steps", _s(update="beta * v + (1 - beta) * g"), "wrong", "average"),
    ("momentum-steps", _s(order="step-first"), "wrong", "old-v"),
    ("momentum-steps", _s(update="beta * (v + g)"), "wrong", "beta-on-g"),
    ("momentum-steps", _s(update="v + beta * g"), "wrong", "swapped"),
    ("momentum-steps", _s(update="g"), "wrong", None),
    ("momentum-steps", "def momentum_steps(gs, lr):\n    return []\n", "error", "arguments"),
    ("momentum-sgd", _M_START, "empty", None),
    ("momentum-sgd", _m(), "correct", None),
    ("momentum-sgd", _m(init_v="torch.zeros(p.shape)"), "correct", None),
    ("momentum-sgd", _m(keep=False), "almost", "not-kept"),
    ("momentum-sgd", _m(update="self.beta * self.vs[i] + (1 - self.beta) * p.grad"), "wrong", "average"),
    ("momentum-sgd", _m().replace("        with torch.no_grad():\n", "        if True:\n"), "error", "no-grad-missing"),
    ("momentum-sgd", _m(update="self.beta * self.vs[i] + self.lr * p.grad").replace("p -= self.lr * self.vs[i]", "p -= self.vs[i]"), "correct", None),
    ("momentum-sgd", _m(update="self.beta * self.vs[i] + (1 - self.beta) * p.grad").replace("p -= self.lr * self.vs[i]", "p -= self.lr / (1 - self.beta) * self.vs[i]"), "correct", None),
    ("momentum-sgd", _m().replace("        with torch.no_grad():\n", "        if True:\n").replace("p -= self.lr", "p.data -= self.lr"), "correct", None),
    ("momentum-sgd", _m().replace("        with torch.no_grad():\n", "        if True:\n").replace("        self.vs = []\n", ""), "error", None),
    ("momentum-steps", _S_HEAD + "    v = 0.0\n    for g in gs:\n        v = beta * v + g\n    return lr * v\n", "wrong", None),
    ("momentum-steps", _s(step="-lr * v"), "wrong", "minus"),
    ("momentum-sgd", _m(old_first=True), "wrong", "old-v"),
]
