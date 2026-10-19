# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-swiglu: SwiGLU, the gated MLP of modern LLMs, with the layers gate, up and down."""
import re

import numpy as np

from everything_ai.check import ClassBox, Mistake


def activations_picture(labels: dict) -> str:
    """ReLU, GELU and SiLU from -4 to 3. Activation functions are not learned: ink, told apart by their names, a dashed
    line for ReLU, a thin one for GELU and a thick one for SiLU."""
    import torch
    import torch.nn.functional as F
    from everything_ai import chart
    xs = torch.linspace(-4, 3, 141)
    p = chart.Panel((-4, 4.7), (-1, 3), title=labels["title"], xlabel="x", ylabel=labels["y"],
                    xticks=[-4, -2, 0, 2], yticks=[-1, 0, 1, 2, 3])
    p.line([-4, 3], [0, 0], "ink-3", 0.8)
    p.line(xs.tolist(), F.relu(xs).tolist(), "ink-3", 2.0, dashed=True)
    p.line(xs.tolist(), F.gelu(xs).tolist(), "ink-2", 1.4)
    p.line(xs.tolist(), F.silu(xs).tolist(), "ink", 3.0)
    p.text(3.3, 2.9, "ReLU", "ink-3", "start")
    p.text(3.3, 2.45, "GELU", "ink-2", "start")
    p.text(3.3, 2.0, "SiLU", "ink", "start")
    p.text(-2.6, -0.62, labels["dip"], "ink-2", "middle")
    svg = chart.figure([p], alt=labels["alt"], width=330, height=270)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# --- exercise 1: SwiGLU -------------------------------------------------------------------------------------------------------

_HEAD = ("import torch\nimport torch.nn.functional as F\nfrom torch import nn\n\n\nclass SwiGLU(nn.Module):\n"
         "    def __init__(self, d, hidden):\n        super().__init__()\n")
_INIT = ("        self.gate = nn.Linear(d, hidden, bias=False)\n        self.up = nn.Linear(d, hidden, bias=False)\n"
         "        self.down = nn.Linear(hidden, d, bias=False)\n")
_FWD = "        return self.down(F.silu(self.gate(x)) * self.up(x))\n"
_START = (_HEAD + "        # three linear layers without bias: gate and up from d to hidden, down from hidden back to d\n        ...\n\n"
          "    def forward(self, x):\n        # SiLU of the gate, times up, then down\n        ...\n")


def _s(init: str = _INIT, fwd: str = _FWD) -> str:
    return _HEAD + init + "\n    def forward(self, x):\n" + fwd


_SET = ("with torch.no_grad():\n    mlp.gate.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))\n"
        "    mlp.up.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, -1.0]]))\n"
        "    mlp.down.weight.copy_(torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]]))\n")

_SCENARIOS = [
    # d = 2, hidden = 3, with weights set by hand: one token
    "import torch\n\nmlp = SwiGLU(2, 3)\n" + _SET + "mlp(torch.tensor([[1.0, -1.0]])).tolist()",
    # the parameters of SwiGLU(8, 12): their shapes, in the order gate, up, down
    "mlp = SwiGLU(8, 12)\n[list(mlp.gate.weight.shape), list(mlp.up.weight.shape), list(mlp.down.weight.shape), sum(p.numel() for p in mlp.parameters())]",
    # a mini-batch of 2 texts of 3 tokens with 4 numbers, hidden 6, random weights: the last token of the second text
    "import torch\n\ntorch.manual_seed(0)\nmlp = SwiGLU(4, 6)\nwith torch.no_grad():\n    mlp.gate.weight.copy_(torch.randn(6, 4))\n"
    "    mlp.up.weight.copy_(torch.randn(6, 4))\n    mlp.down.weight.copy_(torch.randn(4, 6))\nx = torch.randn(2, 3, 4)\nmlp(x)[1, 2].tolist()",
]

_SCENARIO = "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_SCENARIO_NL = "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"


def _close(got, want) -> bool:
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    try:
        return bool(np.allclose(float(got), float(want), rtol=1e-4, atol=1e-4))
    except (TypeError, ValueError):
        return got == want


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

def _has_bias(text: str) -> bool:
    """Some nn.Linear without bias=False, in the code itself (comments left out); a positional False counts as no bias."""
    code = re.sub(r"#[^\n]*", "", text)
    return re.search(r"nn\.Linear\((?![^)]*(?:bias\s*=\s*False|,\s*False\s*\)))[^)]*\)", code) is not None


_SIZES = Mistake("sizes", error=lambda e: isinstance(e, RuntimeError) and ("must match the size" in str(e) or "cannot be multiplied" in str(e)),
                 en="The sizes do not fit: one of your layers is made for other sizes, or gets another number of numbers per token than it was made for, "
                    "or the product is not `*` of two vectors of `hidden` numbers. `gate` and `up` take d numbers to `hidden`, `down` takes `hidden` back to d.",
                 nl="De maten passen niet: een van je layers is gemaakt voor andere maten, of krijgt een ander aantal getallen per token dan waarvoor hij gemaakt is, "
                    "of het product is geen `*` van twee vectoren van `hidden` getallen. `gate` en `up` brengen d getallen naar `hidden`, `down` brengt `hidden` terug naar d.")

_NAMES = Mistake("names", error=lambda e: isinstance(e, AttributeError) and any(f"'{n}'" in str(e) for n in ("gate", "up", "down")),
                 en="The check sets the weights of `mlp.gate`, `mlp.up` and `mlp.down`, and your class has no layer by one of these names. Call the three layers `self.gate`, `self.up` and `self.down`.",
                 nl="De check zet de weights van `mlp.gate`, `mlp.up` en `mlp.down`, en je class heeft geen layer met een van die namen. Noem de drie layers `self.gate`, `self.up` en `self.down`.")

EXERCISES = {
    "swiglu": ClassBox(
        name="swiglu",
        start=_START,
        solution=_s(),
        scenarios=_SCENARIOS,
        compare=_close,
        digits=4,
        first=[_DOTS],
        mistakes=[
            _NAMES,
            _SIZES,
            Mistake("bias", matches=lambda got, text: _has_bias(text),
                    en=_SCENARIO + ": a layer of yours has a bias. The gated MLPs of Llama and its kin have none: give every `nn.Linear` `bias=False`.",
                    nl=_SCENARIO_NL + ": een layer van je heeft een bias. De gated MLP's van Llama en verwanten hebben er geen: geef elke `nn.Linear` `bias=False`."),
            Mistake("no-gate", wrong=_s(fwd="        return self.down(F.silu(self.gate(x)))\n"),
                    en=_SCENARIO + ": you leave `self.up` out, so there is nothing to gate: this is a plain MLP with SiLU. Multiply SiLU of the gate by `self.up(x)`, number by number.",
                    nl=_SCENARIO_NL + ": je laat `self.up` weg, dus de gate heeft niets om te vermenigvuldigen: dit is een gewoon MLP met SiLU. Vermenigvuldig SiLU van de gate getal voor getal met `self.up(x)`."),
            Mistake("no-down", wrong=_s(fwd="        return F.silu(self.gate(x)) * self.up(x)\n"),
                    en=_SCENARIO + ": you leave `self.down` out, so every token keeps `hidden` numbers instead of d. Send the product through `self.down`.",
                    nl=_SCENARIO_NL + ": je laat `self.down` weg, dus elk token houdt `hidden` getallen in plaats van d. Stuur het product door `self.down`."),
            Mistake("silu-on-up", wrong=_s(fwd="        return self.down(self.gate(x) * F.silu(self.up(x)))\n"),
                    en=_SCENARIO + ": you take SiLU of `self.up`. The gate is the one that goes through SiLU: `F.silu(self.gate(x)) * self.up(x)`.",
                    nl=_SCENARIO_NL + ": je neemt SiLU van `self.up`. De gate is degene die door SiLU gaat: `F.silu(self.gate(x)) * self.up(x)`."),
            Mistake("silu-after", wrong=_s(fwd="        return self.down(F.silu(self.gate(x) * self.up(x)))\n"),
                    en=_SCENARIO + ": you take SiLU of the product. SiLU goes on the gate alone, before you multiply by `self.up(x)`.",
                    nl=_SCENARIO_NL + ": je neemt SiLU van het product. SiLU hoort alleen op de gate, voordat je met `self.up(x)` vermenigvuldigt."),
            Mistake("sigmoid", wrong=_s(fwd="        return self.down(torch.sigmoid(self.gate(x)) * self.up(x))\n"),
                    en=_SCENARIO + ": with the sigmoid itself as the gate, this is the original GLU. SwiGLU uses SiLU, $x$ times its sigmoid: `F.silu`.".replace("$x$", "x"),
                    nl=_SCENARIO_NL + ": met de sigmoid zelf als gate is dit de oorspronkelijke GLU. SwiGLU gebruikt SiLU, x maal zijn sigmoid: `F.silu`."),
            Mistake("relu", wrong=_s(fwd="        return self.down(torch.relu(self.gate(x)) * self.up(x))\n"),
                    en=_SCENARIO + ": with ReLU as the gate, this is ReGLU. SwiGLU uses SiLU: `F.silu`.",
                    nl=_SCENARIO_NL + ": met ReLU als gate is dit ReGLU. SwiGLU gebruikt SiLU: `F.silu`."),
            Mistake("gelu", wrong=_s(fwd="        return self.down(F.gelu(self.gate(x)) * self.up(x))\n"),
                    en=_SCENARIO + ": with GELU as the gate, this is GEGLU, a close cousin. SwiGLU uses SiLU: `F.silu`.",
                    nl=_SCENARIO_NL + ": met GELU als gate is dit GEGLU, een naaste verwant. SwiGLU gebruikt SiLU: `F.silu`."),
            Mistake("added", wrong=_s(fwd="        return self.down(F.silu(self.gate(x)) + self.up(x))\n"),
                    en=_SCENARIO + ": you add the two. A gate multiplies, number by number: `*`.",
                    nl=_SCENARIO_NL + ": je telt de twee op. Een gate vermenigvuldigt, getal voor getal: `*`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: in `__init__` the three layers, in `forward` the gated product and the way back to d.",
                      "nl": "Vervang de twee `...`: in `__init__` de drie layers, in `forward` het product met de gate en de weg terug naar d."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "wrong": {"en": _SCENARIO + ". Do `gate` and `up` take d numbers to `hidden`, `down` `hidden` back to d, all without bias, and is the hidden vector SiLU of the gate times up?",
                      "nl": _SCENARIO_NL + ". Brengen `gate` en `up` d getallen naar `hidden`, `down` `hidden` terug naar d, allemaal zonder bias, en is de hidden vector SiLU van de gate maal up?"},
            "almost": {"en": "Right for the first example. But " + _SCENARIO[0].lower() + _SCENARIO[1:] + ". Does your class use `d` and `hidden`, and work on a mini-batch of tokens?",
                       "nl": "Goed voor het eerste voorbeeld. Maar " + _SCENARIO_NL[0].lower() + _SCENARIO_NL[1:] + ". Gebruikt je class `d` en `hidden`, en werkt hij op een mini-batch tokens?"},
            "correct": {"en": "Your SwiGLU gates every hidden number: SiLU of the gate times up, then down back to d, with three matrices and no bias.",
                        "nl": "Bij je SwiGLU gaat elk hidden getal door een gate: SiLU van de gate maal up, dan down terug naar d, met drie matrices en geen bias."},
        },
    ),
}

# The lesson is done when these are correct. "hidden-size" is the choice question, "what-the-gate-adds" the open one.
REQUIRED = ["swiglu", "hidden-size", "what-the-gate-adds"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("swiglu", _START, "empty", None),
    ("swiglu", _s(), "correct", None),
    ("swiglu", _s(fwd="        g = self.gate(x)\n        h = g * torch.sigmoid(g) * self.up(x)\n        return self.down(h)\n"), "correct", None),
    ("swiglu", _s(init=_INIT.replace("self.gate", "self.w1").replace("self.up", "self.w3").replace("self.down", "self.w2"),
                  fwd="        return self.w2(F.silu(self.w1(x)) * self.w3(x))\n"), "error", "names"),
    ("swiglu", _s(init=_INIT.replace(", bias=False", "")), "wrong", "bias"),
    ("swiglu", _s(fwd="        return self.down(F.silu(self.gate(x)))\n"), "wrong", "no-gate"),
    ("swiglu", _s(fwd="        return self.down(self.gate(x) * F.silu(self.up(x)))\n"), "wrong", "silu-on-up"),
    ("swiglu", _s(fwd="        return self.down(F.silu(self.gate(x) * self.up(x)))\n"), "wrong", "silu-after"),
    ("swiglu", _s(fwd="        return self.down(torch.sigmoid(self.gate(x)) * self.up(x))\n"), "wrong", "sigmoid"),
    ("swiglu", _s(fwd="        return self.down(torch.relu(self.gate(x)) * self.up(x))\n"), "wrong", "relu"),
    ("swiglu", _s(fwd="        return self.down(F.gelu(self.gate(x)) * self.up(x))\n"), "wrong", "gelu"),
    ("swiglu", _s(fwd="        return self.down(F.silu(self.gate(x)) + self.up(x))\n"), "wrong", "added"),
    ("swiglu", _s(fwd="        return F.silu(self.gate(x)) * self.up(x)\n"), "wrong", "no-down"),
    ("swiglu", _s(init=_INIT.replace("nn.Linear(hidden, d, bias=False)", "nn.Linear(d, hidden, bias=False)")), "error", "sizes"),
    ("swiglu", _s(fwd="        return self.down(F.silu(self.gate(x))) * self.up(x)\n"), "error", "sizes"),
    ("swiglu", _s(fwd="        return self.down(F.silu(self.gate(x)) @ self.up(x))\n"), "error", "sizes"),
    ("swiglu", _s(init=_INIT.replace(", bias=False", ", False"), fwd="        return self.down(self.gate(x) * F.silu(self.up(x)))\n"), "wrong", "silu-on-up"),
    ("swiglu", _s(init="        # nn.Linear(d, hidden), but without a bias\n" + _INIT, fwd="        return self.down(self.gate(x) * F.silu(self.up(x)))\n"), "wrong", "silu-on-up"),
]
