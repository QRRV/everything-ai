# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson opt-adamw: adamw_path, where one parameter goes under AdamW, and param_groups, the two
groups of parameters for AdamW: the weight matrices with weight decay, the rest without. And the picture: a weight
without a gradient of its own, under an L2 penalty inside Adam and under AdamW, for two values of lambda."""
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai.check import ClassBox, FunctionBox, Mistake

LR = 0.01
STEPS = 1000


def quiet_weight(decoupled: bool, lam: float, steps: int = STEPS) -> list[float]:
    """A weight that starts at 1 and gets no gradient from the data: Adam with the penalty lam * w added to its
    gradient (an L2 penalty), or AdamW with weight decay lam. Its value after each step, starting with 1."""
    w, m, s = 1.0, 0.0, 0.0
    out = [w]
    for t in range(1, steps + 1):
        g = 0.0 if decoupled else lam * w
        m = 0.9 * m + 0.1 * g
        s = 0.999 * s + 0.001 * g * g
        m_hat, s_hat = m / (1 - 0.9 ** t), s / (1 - 0.999 ** t)
        if decoupled:
            w = w - LR * lam * w
        w = w - LR * m_hat / (math.sqrt(s_hat) + 1e-8)
        out.append(w)
    return out


def quiet_picture(labels: dict) -> str:
    """Two panels: the quiet weight over STEPS steps with an L2 penalty inside Adam and with AdamW, for lambda 0.01
    and 0.0001."""
    from everything_ai import chart
    steps = list(range(STEPS + 1))
    panels = []
    for decoupled, key in ((False, "l2"), (True, "adamw")):
        p = chart.Panel((0, STEPS), (-0.05, 1.1), title=labels[key], xlabel=labels["x"], ylabel=labels["y"] if not panels else "")
        for lam, dashed, place in ((0.01, False, "big"), (0.0001, True, "small")):
            path = quiet_weight(decoupled, lam)
            p.line(steps[::5], path[::5], "param", 2.2 if not dashed else 2.0, dashed=dashed)
        if decoupled:
            p.text(STEPS - 20, quiet_weight(True, 0.01)[-1] - 0.08, labels["big"], "param", "end")
            p.text(STEPS - 20, 1.04, labels["small"], "param", "end")
        else:
            p.text(260, 0.12, labels["both"], "param", "start")
        panels.append(p)
    return chart.figure(panels, alt=labels["alt"], width=330, height=290)


# The lesson is done when these are correct. "weight-decay-arg" is the choice question in the lesson.
REQUIRED = ["weight-decay-arg", "adamw-path", "param-groups"]

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
    return np.format_float_positional(x, precision=5, unique=False, fractional=False, trim="-").rstrip(".")


def _show(v) -> str:
    if np.ndim(v) == 0:
        try:
            return _num(v)
        except (TypeError, ValueError):
            return repr(v)
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
    return len(g) == len(w) and all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-9) for a, b in zip(g, w))


# --- exercise 2: one parameter under AdamW ----------------------------------------------------------------------------------

_P_HEAD = "def adamw_path(gs, w, lr, wd):\n"
_P_START = (_P_HEAD + "    # for each gradient g in gs: first w shrinks by the fraction lr * wd, then Adam's step with\n"
            "    # beta1 = 0.9, beta2 = 0.999 and eps = 1e-8; give back the list of w after each step\n    ...\n")


def _p(decay: str = "w = w - lr * wd * w\n        ", grad: str = "g", tail: str = "") -> str:
    return (_P_HEAD + "    m = 0.0\n    s = 0.0\n    t = 0\n    ws = []\n    for g in gs:\n        t = t + 1\n" + f"        g = {grad}\n"
            "        m = 0.9 * m + 0.1 * g\n        s = 0.999 * s + 0.001 * g ** 2\n        m_hat = m / (1 - 0.9 ** t)\n        s_hat = s / (1 - 0.999 ** t)\n"
            f"        {decay}w = w - lr * m_hat / (s_hat ** 0.5 + 1e-8)\n{tail}        ws.append(w)\n    return ws\n")


# --- exercise 3: two groups of parameters -------------------------------------------------------------------------------------

_G_HEAD = "def param_groups(model, wd):\n"
_G_START = (_G_HEAD + "    # the weight matrices (2 or more dimensions) with weight_decay wd, the rest (bias terms, norms) with 0\n    ...\n")


def _g(test: str = "p.dim() >= 2", first: str = "wd", second: str = "0.0") -> str:
    return (_G_HEAD + "    decay = []\n    no_decay = []\n    for p in model.parameters():\n" + f"        if {test}:\n"
            "            decay.append(p)\n        else:\n            no_decay.append(p)\n"
            f"    return [{{\"params\": decay, \"weight_decay\": {first}}}, {{\"params\": no_decay, \"weight_decay\": {second}}}]\n")


_NET = ("import torch\nfrom torch import nn\n\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
        "        self.hidden = nn.Linear({a}, {b})\n        self.norm = nn.LayerNorm({b})\n        self.out = nn.Linear({b}, {c})\n\n\n"
        "model = Net()\ngroups = param_groups(model, {wd})\nopt = torch.optim.AdamW(groups, lr=0.001)\n")
_COUNT = ("sizes = []\nfor group in groups:\n    n = 0\n    for p in group[\"params\"]:\n        n = n + p.numel()\n    sizes.append([n, group[\"weight_decay\"]])\nsizes")
_G_SCENARIOS = [
    _NET.format(a=4, b=3, c=2, wd=0.1) + _COUNT,
    _NET.format(a=10, b=6, c=5, wd=0.05) + _COUNT,   # not 0.01: that is AdamW's default
]

EXERCISES = {
    "adamw-path": FunctionBox(
        name="adamw-path",
        function="adamw_path",
        start=_P_START,
        solution=_p(),
        visible=[([0.3, 0.1, 0.0], 1.0, 0.1, 0.5)],
        hidden=[([0.0, 0.0, 0.0, 0.0], -2.0, 0.1, 0.5), ([1.0, -1.0, 2.0], 0.5, 0.01, 0.0), ([0.02, 0.02, 0.05, -0.01, 0.0], 3.0, 0.05, 0.1)],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("l2", wrong=_p(decay="", grad="g + wd * w"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you add `wd * w` to the gradient, so Adam divides it by the root of `s` like the rest: that is an L2 penalty inside Adam. AdamW shrinks `w` itself, apart from the gradient: `w = w - lr * wd * w`, before Adam's step.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt `wd * w` op bij de gradient, dus Adam deelt hem door de wortel uit `s`, net als de rest: dat is een L2-straf binnen Adam. AdamW laat `w` zelf krimpen, los van de gradient: `w = w - lr * wd * w`, vóór de stap van Adam."),
            Mistake("after", wrong=_p(decay="", tail="        w = w - lr * wd * w\n"), state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: you shrink `w` after Adam's step, so the step is shrunk too. As PyTorch's AdamW does, shrink `w` first, then take the step.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je laat `w` krimpen na de stap van Adam, dus de stap krimpt mee. Laat, zoals AdamW in PyTorch, eerst `w` krimpen en zet dan de stap."),
            Mistake("no-decay", wrong=_p(decay=""),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your weight moves as with Adam alone, without weight decay. AdamW first shrinks the weight, `w = w - lr * wd * w`, and then takes Adam's step.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je weight beweegt zoals met alleen Adam, zonder weight decay. AdamW laat de weight eerst krimpen, `w = w - lr * wd * w`, en zet dan de stap van Adam."),
            Mistake("no-lr", wrong=_p(decay="w = w - wd * w\n        "),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `w` shrinks by the fraction `wd` per step. In AdamW the fraction is `lr * wd`: the learning rate times the weight decay.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `w` krimpt per stap met de fractie `wd`. In AdamW is de fractie `lr * wd`: de learning rate keer de weight decay."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: `m`, `s` and `t` start at 0; for each gradient, first `w` shrinks, then Adam's step, and `w` into a list.",
                      "nl": "Vervang de `...` door de body van de functie: `m`, `s` en `t` beginnen op 0; voor elke gradient eerst `w` laten krimpen, dan de stap van Adam, en `w` in een lijst."},
            "arguments": {"en": "The check calls `{call}`, with a list of gradients, a starting `w`, a learning rate and a weight decay, and the `def` line of your function has another number of names between its parentheses. Keep `(gs, w, lr, wd)`.",
                          "nl": "De check roept `{call}` aan, met een lijst gradients, een begin-`w`, een learning rate en een weight decay, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(gs, w, lr, wd)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Work out the first step by hand: `w` times `1 - lr * wd`, and then Adam's first step, `lr` against the sign of the gradient.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken de eerste stap met de hand uit: `w` keer `1 - lr * wd`, en dan de eerste stap van Adam, `lr` tegen het teken van de gradient in."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work when every gradient is 0, for a negative `w`, and with a weight decay of 0?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie als elke gradient 0 is, voor een negatieve `w`, en met een weight decay van 0?"},
            "correct": {"en": "Your function is right in all {n} cases: every weight shrinks by the same fraction, apart from Adam's step.",
                        "nl": "Je functie klopt in alle {n} gevallen: elke weight krimpt met dezelfde fractie, los van de stap van Adam."},
        },
    ),
    "param-groups": ClassBox(
        name="param-groups",
        start=_G_START,
        solution=_g(),
        scenarios=_G_SCENARIOS,
        compare=lambda got, want: got == want,
        first=[_DOTS],
        mistakes=[
            Mistake("both-groups", error=lambda e: isinstance(e, ValueError) and "more than one parameter group" in str(e),
                    en="`torch.optim.AdamW` stops with an error: some parameters are in both of your groups. Put each parameter in one list only, with `if` and `else`.",
                    nl="`torch.optim.AdamW` stopt met een fout: sommige parameters staan in allebei je groepen. Zet elke parameter in maar één lijst, met `if` en `else`."),
            Mistake("set", error=lambda e: isinstance(e, TypeError) and "ordered collections" in str(e),
                    en="`torch.optim.AdamW` stops with an error: a group holds its parameters in a set, and a set has no fixed order. Use a list for each group: `decay = []` and `decay.append(p)`.",
                    nl="`torch.optim.AdamW` stopt met een fout: een groep bewaart zijn parameters in een set, en een set heeft geen vaste volgorde. Gebruik voor elke groep een lijst: `decay = []` en `decay.append(p)`."),
            Mistake("not-tensors", error=lambda e: isinstance(e, TypeError) and "can only optimize Tensors" in str(e),
                    en="`torch.optim.AdamW` stops with an error: where it looks for parameters, it finds something that is not a tensor. Give back a list of two dicts, each with a list of parameters under `\"params\"`.",
                    nl="`torch.optim.AdamW` stopt met een fout: waar hij parameters zoekt, vindt hij iets wat geen tensor is. Geef een lijst van twee dicts terug, elk met een lijst parameters onder `\"params\"`."),
            Mistake("no-key", matches=lambda got, text: "0.01" not in text and isinstance(got, list)
                    and any(isinstance(g, list) and len(g) == 2 and g[1] == 0.01 for g in got),
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: a group without the key `\"weight_decay\"` gets AdamW's default, 0.01. Give each group its own: `wd` for the weight matrices, `0.0` for the rest.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: een groep zonder de key `\"weight_decay\"` krijgt de standaardwaarde van AdamW, 0,01. Geef elke groep een eigen waarde: `wd` voor de weight-matrices, `0.0` voor de rest."),
            Mistake("swapped", wrong=_g(first="0.0", second="wd"),
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: the weight decay is on the bias terms and norms, and not on the weight matrices. It belongs on the weight matrices, the parameters with 2 or more dimensions.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: de weight decay staat op de bias terms en norms, en niet op de weight-matrices. Hij hoort op de weight-matrices, de parameters met 2 of meer dimensies."),
            Mistake("one-dim", wrong=_g(test="p.dim() >= 1"),
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: every parameter has 1 or more dimensions, so everything gets weight decay. The weight matrices have 2: `p.dim() >= 2`.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: elke parameter heeft 1 of meer dimensies, dus alles krijgt weight decay. De weight-matrices hebben er 2: `p.dim() >= 2`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: go over `model.parameters()`, put the ones with 2 or more dimensions in one list and the others in another, and give back the two groups.",
                      "nl": "Vervang de `...`: ga langs `model.parameters()`, zet die met 2 of meer dimensies in één lijst en de andere in een andere, en geef de twee groepen terug."},
            "wrong": {"en": "With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: for each group, the number of numbers in its parameters and its weight decay. Are the weight matrices in the first group, with `wd`, and the bias terms and norms in the second, with 0.0?",
                      "nl": "Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: per groep het aantal getallen in zijn parameters en zijn weight decay. Staan de weight-matrices in de eerste groep, met `wd`, en de bias terms en norms in de tweede, met 0.0?"},
            "almost": {"en": "Right for the first network. But with these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does your function use `wd` itself, and work for any sizes?",
                       "nl": "Goed voor het eerste netwerk. Maar met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Gebruikt je functie `wd` zelf, en werkt ze voor elke grootte?"},
            "correct": {"en": "Your groups give weight decay to the weight matrices only, as most transformers are trained.",
                        "nl": "Je groepen geven alleen de weight-matrices weight decay, zoals de meeste transformers getraind worden."},
        },
    ),
}

CASES = [
    ("adamw-path", _P_START, "empty", None),
    ("adamw-path", _p(), "correct", None),
    ("adamw-path", _p(decay="w = w * (1 - lr * wd)\n        "), "correct", None),
    ("adamw-path", _p(decay=""), "wrong", "no-decay"),
    ("adamw-path", _p(decay="", grad="g + wd * w"), "wrong", "l2"),
    ("adamw-path", _p(decay="", tail="        w = w - lr * wd * w\n"), "almost", "after"),
    ("adamw-path", _p(decay="w = w - wd * w\n        "), "wrong", "no-lr"),
    ("adamw-path", "def adamw_path(gs, w):\n    return []\n", "error", "arguments"),
    ("param-groups", _G_START, "empty", None),
    ("param-groups", _g(), "correct", None),
    ("param-groups", _g(test="len(p.shape) >= 2"), "correct", None),
    ("param-groups", _g(test="p.ndim > 1"), "correct", None),
    ("param-groups", _g(first="0.0", second="wd"), "wrong", "swapped"),
    ("param-groups", _g(test="p.dim() >= 1"), "wrong", "one-dim"),
    ("param-groups", _g(first="0.1"), "almost", None),
    ("param-groups", _g().replace(', {"params": no_decay, "weight_decay": 0.0}]', ', {"params": no_decay}]'), "wrong", "no-key"),
    ("param-groups", _g().replace('"weight_decay": wd}, {', '"wd": wd}, {'), "wrong", "no-key"),
    ("param-groups", _g().replace("        else:\n            no_decay.append(p)", "        no_decay.append(p)"), "error", "both-groups"),
    ("param-groups", _g().replace("[]", "set()").replace(".append(", ".add("), "error", "set"),
    ("param-groups", _g().replace('return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]', 'return {"params": decay, "weight_decay": wd}'), "error", "not-tensors"),
    ("adamw-path", _p().replace("(s_hat ** 0.5 + 1e-8)", "(s_hat + 1e-8) ** 0.5"), "correct", None),
]
