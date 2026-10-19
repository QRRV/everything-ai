# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-residual: residual_chain_gradient, the gradient that comes back through a chain of sigmoid
layers with residual connections, and ResidualBlock, a block y = x + f(x) in PyTorch. And the picture: the size of
the gradient in each block of a deep MLP for the handwritten digits of QMNIST, with and without residual connections."""
import functools
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn

from everything_ai import paths
from everything_ai.check import ClassBox, FunctionBox, Mistake

DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md
BLOCKS = 10
WIDTH = 64


class DeepNet(nn.Module):
    """An input layer, BLOCKS blocks of two layers, f(h) = second(relu(first(h))), and 10 logits. With residual
    connections, each block gives h + f(h); without, relu(f(h))."""

    def __init__(self, residual: bool):
        super().__init__()
        self.inp = nn.Linear(784, WIDTH)
        self.first = nn.ModuleList([nn.Linear(WIDTH, WIDTH) for _ in range(BLOCKS)])
        self.second = nn.ModuleList([nn.Linear(WIDTH, WIDTH) for _ in range(BLOCKS)])
        self.out = nn.Linear(WIDTH, 10)
        self.residual = residual

    def forward(self, x):
        h = torch.relu(self.inp(x))
        for a, b in zip(self.first, self.second):
            f = b(torch.relu(a(h)))
            h = h + f if self.residual else torch.relu(f)
        return self.out(h)


@functools.lru_cache(maxsize=2)
def block_gradients(residual: bool) -> list[float]:
    """The mean size of the gradient of the first weights of each block, for one backward pass on 256 training
    digits, from the first block to the last."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)   # faster on these small layers, and the numbers no longer depend on the number of cores
    try:
        d = np.load(paths.ROOT / DATA)
        X = torch.tensor(d["train_images"][:256], dtype=torch.float32) / 255
        y = torch.tensor(d["train_labels"][:256], dtype=torch.int64)
        torch.manual_seed(0)
        model = DeepNet(residual)
        loss = nn.CrossEntropyLoss()(model(X), y)
        loss.backward()
        return [layer.weight.grad.abs().mean().item() for layer in model.first]
    finally:
        torch.set_num_threads(threads)


def grad_picture(labels: dict) -> str:
    """The size of the gradient per block, on a log scale, with and without residual connections."""
    from everything_ai import chart
    blocks = list(range(1, BLOCKS + 1))
    plain, res = block_gradients(False), block_gradients(True)
    p = chart.Panel((0.5, BLOCKS + 0.5), (1e-12, 1e-2), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], ylog=True,
                    xticks=[1, 2, 4, 6, 8, 10])
    p.line(blocks, plain, "grad", 2.0, dashed=True)
    p.line(blocks, res, "grad", 2.4)
    p.text(6, plain[5] * 30, labels["plain"], "grad", "end")
    p.text(5.5, res[4] * 12, labels["res"], "grad", "middle")
    return chart.figure([p], alt=labels["alt"], width=560, height=310)


# The lesson is done when these are correct. "why-harmless" is the choice question in the lesson.
REQUIRED = ["why-harmless", "residual-chain", "residual-block"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _plain(x) -> str:
    """A number with 4 significant digits, written out, never in e-notation."""
    x = float(x)
    if not math.isfinite(x):
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    if x == 0:
        return "0"
    return np.format_float_positional(x, precision=4, unique=False, fractional=False, trim="-").rstrip(".")


def _show(v) -> str:
    """A number, or every number of a list or array, with _plain; anything else as Python shows it."""
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return repr(v)
    if a.ndim == 0:
        return _plain(a)
    return "[" + ", ".join(_plain(x) for x in a.ravel()) + "]"


def _close(got, want) -> bool:
    try:
        return math.isclose(float(got), float(want), rel_tol=1e-6)
    except (TypeError, ValueError):
        return False


def _lists_close(got, want) -> bool:
    try:
        g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    except (TypeError, ValueError):
        return False
    return g.shape == w.shape and bool(np.allclose(g, w, atol=1e-4))


# --- exercise 2: a chain of sigmoid layers with residual connections ------------------------------------------------------

_C_HEAD = "import numpy as np\n\n\ndef sigmoid(z):\n    return 1 / (1 + np.exp(-z))\n\n\ndef residual_chain_gradient(ws, x):\n"
_C_START = _C_HEAD + "    # the gradient that comes back to x through one layer for each weight in ws,\n    # where each layer gives h + sigmoid(w * h)\n    ...\n"


def _c(factor: str = "(1 + w * sigmoid(z) * (1 - sigmoid(z)))", forward: str = "h + sigmoid(z)", step: str = "g * {factor}") -> str:
    return (_C_HEAD + "    h = x\n    zs = []\n    for w in ws:\n        z = w * h\n        zs.append(z)\n" + f"        h = {forward}\n"
            "    g = 1.0\n    for k in reversed(range(len(ws))):\n        w = ws[k]\n        z = zs[k]\n" + f"        g = {step.format(factor=factor)}\n" + "    return g\n")


# --- exercise 3: a residual block in PyTorch ------------------------------------------------------------------------------

_B_HEAD = "import torch\nfrom torch import nn\n\n\nclass ResidualBlock(nn.Module):\n"
_B_START = (_B_HEAD + "    def __init__(self, d, hidden):\n        super().__init__()\n        # lin1 from d to hidden, then lin2 from hidden back to d\n        ...\n\n"
            "    def forward(self, x):\n        # x plus f(x), with f: lin1, ReLU, lin2\n        ...\n")


def _b(out: str = "x + self.lin2(torch.relu(self.lin1(x)))") -> str:
    return (_B_HEAD + "    def __init__(self, d, hidden):\n        super().__init__()\n        self.lin1 = nn.Linear(d, hidden)\n"
            "        self.lin2 = nn.Linear(hidden, d)\n\n    def forward(self, x):\n" + f"        return {out}\n")


_B_IN_PLACE = _b("x").replace("        return x\n", "        x += self.lin2(torch.relu(self.lin1(x)))\n        return x\n")


_B_SCENARIOS = [
    "torch.manual_seed(0)\nblock = ResidualBlock(4, 8)\nx = torch.randn(3, 4)\nblock(x).tolist()",
    "torch.manual_seed(1)\nblock = ResidualBlock(3, 5)\nx = torch.randn(2, 3)\nblock(block(x)).tolist()",
    "torch.manual_seed(2)\nblock = ResidualBlock(6, 2)\nx = torch.randn(4, 6)\nblock(x).tolist()",
    "torch.manual_seed(3)\nblock = ResidualBlock(3, 4)\nx = torch.randn(2, 3)\ny = block(x)\nx.tolist()",
]

EXERCISES = {
    "residual-chain": FunctionBox(
        name="residual-chain",
        function="residual_chain_gradient",
        start=_C_START,
        solution=_c(),
        visible=[([2.0, 1.0, 1.5], 0.5)],
        hidden=[([1.0] * 10, 0.5), ([4.0] * 6, 0.5), ([-1.0, 2.0, 0.5, 3.0], 1.0), ([0.5], -2.0)],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("rounded", matches=lambda got, text: "round(" in text, state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: they may look the same, but you round the answer, and the check compares more decimals than it shows. Give back the number itself, without `round`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: ze lijken misschien gelijk, maar je rondt het antwoord af, en de check vergelijkt meer decimalen dan hij laat zien. Geef het getal zelf terug, zonder `round`."),
            Mistake("no-one", wrong=_c(factor="w * sigmoid(z) * (1 - sigmoid(z))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: each factor is only the weight times the slope of the sigmoid, as without residual connections. A layer `h + sigmoid(w * h)` has the derivative 1 plus that: `1 + w * sigmoid(z) * (1 - sigmoid(z))`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elke factor is alleen de weight keer de helling van de sigmoid, zoals zonder residual connections. Een layer `h + sigmoid(w * h)` heeft als afgeleide 1 plus dat: `1 + w * sigmoid(z) * (1 - sigmoid(z))`."),
            Mistake("plain-forward", wrong=_c(forward="sigmoid(z)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your way back is right, but in the forward pass each layer gives only `sigmoid(z)`. With a residual connection it gives its input plus that: `h = h + sigmoid(z)`, and the next `z` is then `w` times that.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je weg terug klopt, maar in de forward pass geeft elke layer alleen `sigmoid(z)`. Met een residual connection geeft hij zijn input plus dat: `h = h + sigmoid(z)`, en de volgende `z` is dan `w` keer dat."),
            Mistake("no-weight", wrong=_c(factor="(1 + sigmoid(z) * (1 - sigmoid(z)))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: inside the factor, the slope of the sigmoid is also multiplied by the weight, because `z = w * h`: `1 + w * sigmoid(z) * (1 - sigmoid(z))`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: binnen de factor wordt de helling van de sigmoid ook met de weight vermenigvuldigd, omdat `z = w * h`: `1 + w * sigmoid(z) * (1 - sigmoid(z))`."),
            Mistake("added", wrong=_c(step="g + w * sigmoid(z) * (1 - sigmoid(z))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you add each layer's slope to the gradient. The forward pass adds, but the chain rule still multiplies: each layer multiplies the gradient by its derivative, `1 + w * sigmoid(z) * (1 - sigmoid(z))`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt de helling van elke layer bij de gradient op. De forward pass telt op, maar de kettingregel vermenigvuldigt nog steeds: elke layer vermenigvuldigt de gradient met zijn afgeleide, `1 + w * sigmoid(z) * (1 - sigmoid(z))`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: the forward pass, keeping each `z`, and then the way back, multiplying the gradient by each layer's factor.",
                      "nl": "Vervang de `...` door de body van de functie: de forward pass, waarbij je elke `z` bewaart, en dan de weg terug, waarbij je de gradient met de factor van elke layer vermenigvuldigt."},
            "arguments": {"en": "The check calls `{call}`, with a list of weights and a number, and the `def` line of your function has another number of names between its parentheses. Keep `(ws, x)`.",
                          "nl": "De check roept `{call}` aan, met een lijst weights en een getal, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(ws, x)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Work out the last layer by hand: its `z`, and its factor, 1 plus the weight times the slope of the sigmoid at that `z`.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken de laatste layer met de hand uit: zijn `z`, en zijn factor, 1 plus de weight keer de helling van de sigmoid bij die `z`."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any number of layers, with a different weight in each, also a negative one?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elk aantal layers, met in elke layer een andere weight, ook een negatieve?"},
            "correct": {"en": "Your function is right in all {n} cases. Ten layers with residual connections and weight 1 let 1.79 come back, where ten plain ones let 0.00000035 through.",
                        "nl": "Je functie klopt in alle {n} gevallen. Tien layers met residual connections en weight 1 laten 1,79 terugkomen, waar tien gewone er 0,00000035 doorlaten."},
        },
    ),
    "residual-block": ClassBox(
        name="residual-block",
        start=_B_START,
        solution=_b(),
        scenarios=_B_SCENARIOS,
        compare=_lists_close,
        digits=4,
        first=[_DOTS],
        mistakes=[
            Mistake("no-super", syntax=r"\A(?:(?![\s\S]*\bsuper\s*\()(?![\s\S]*Module\s*\.\s*__init__\s*\()|[\s\S]*?^[ \t]+self\.\w+[ \t]*=[^=][\s\S]*?\bsuper\s*\()",
                    en="Your `__init__` does not start with `super().__init__()`. `nn.Module` has to set itself up before your class can give it a layer: write `super().__init__()` as the first line of `__init__`, and only then make the layers.",
                    nl="Je `__init__` begint niet met `super().__init__()`. `nn.Module` moet zichzelf eerst opzetten voordat je class er een layer aan kan geven: schrijf `super().__init__()` als eerste regel van `__init__`, en maak pas daarna de layers."),
            Mistake("no-forward", syntax=r"\A(?![\s\S]*^[ \t]+def[ \t]+forward[ \t]*\()",
                    en="Your class has no method `forward`: PyTorch calls it when you write `model(x)`. Write `def forward(self, x):`, with exactly that name.",
                    nl="Je class heeft geen method `forward`: PyTorch roept die aan als je `model(x)` schrijft. Schrijf `def forward(self, x):`, met precies die naam."),
            Mistake("no-skip", wrong=_b("self.lin2(torch.relu(self.lin1(x)))"),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: your block gives only `f(x)`. A residual block adds its input: `x + self.lin2(torch.relu(self.lin1(x)))`.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: je block geeft alleen `f(x)`. Een residual block telt zijn input erbij op: `x + self.lin2(torch.relu(self.lin1(x)))`."),
            Mistake("relu-after", wrong=_b("torch.relu(x + self.lin2(torch.relu(self.lin1(x))))"),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: you put a ReLU after the addition. The first ResNets did that, but it cuts the direct path: wherever `x + f(x)` is below 0, the ReLU gives 0, and no gradient gets back to `x` there. Most transformers today keep the path clean, as this exercise asks: `x + f(x)`.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: je zet een ReLU na het optellen. De eerste ResNets deden dat, maar het onderbreekt de directe weg: waar `x + f(x)` onder 0 ligt, geeft de ReLU 0, en daar komt er geen gradient terug bij `x`. De meeste transformers van nu houden de weg schoon, zoals deze oefening vraagt: `x + f(x)`."),
            Mistake("relu-end", wrong=_b("x + torch.relu(self.lin2(torch.relu(self.lin1(x))))"),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: your `f` ends with a ReLU, so it can only add, never take away. `f` ends with `lin2`, without ReLU.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: je `f` eindigt met een ReLU, dus hij kan alleen optellen, nooit aftrekken. `f` eindigt met `lin2`, zonder ReLU."),
            Mistake("in-place", wrong=_B_IN_PLACE,
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: your block changes `x` itself, as `x += ...` does. PyTorch needs that `x` on the way back, so in a network `backward()` stops with an error. Write `return x + ...`: that makes a new tensor and leaves `x` as it was.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: je block verandert `x` zelf, zoals `x += ...` doet. PyTorch heeft die `x` nodig op de terugweg, dus in een netwerk stopt `backward()` met een fout. Schrijf `return x + ...`: dat maakt een nieuwe tensor en laat `x` zoals hij was."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: in `__init__` the layers `lin1` and `lin2`, and in `forward` the input plus `f(x)`.",
                      "nl": "Vervang de twee `...`: in `__init__` de layers `lin1` en `lin2`, en in `forward` de input plus `f(x)`."},
            "wrong": {"en": "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Do you make `lin1` before `lin2`, so that they get the same random weights, and is `f` lin1, ReLU, lin2, added to `x`?",
                      "nl": "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Maak je `lin1` vóór `lin2`, zodat ze dezelfde willekeurige weights krijgen, en is `f` lin1, ReLU, lin2, opgeteld bij `x`?"},
            "almost": {"en": "Right for the example. But with these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does your block work for any `d` and `hidden`, also twice in a row?",
                       "nl": "Goed voor het voorbeeld. Maar met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Werkt je block voor elke `d` en `hidden`, ook twee keer na elkaar?"},
            "correct": {"en": "Your block adds f(x) to its input: the input always has a direct path through, forward and back.",
                        "nl": "Je block telt f(x) op bij zijn input: de input heeft altijd een directe weg erdoorheen, heen en terug."},
        },
    ),
}

CASES = [
    ("residual-chain", _C_START, "empty", None),
    ("residual-chain", _c(), "correct", None),
    ("residual-chain", _C_HEAD + "    h = x\n    factors = []\n    for w in ws:\n        s = sigmoid(w * h)\n        factors.append(1 + w * s * (1 - s))\n        h = h + s\n    return float(np.prod(factors))\n", "correct", None),
    ("residual-chain", _c(factor="w * sigmoid(z) * (1 - sigmoid(z))"), "wrong", "no-one"),
    ("residual-chain", _c().replace("    return g\n", "    return round(g, 3)\n"), "almost", "rounded"),
    ("residual-chain", _c(forward="sigmoid(z)"), "wrong", "plain-forward"),
    ("residual-chain", _c(factor="(1 + sigmoid(z) * (1 - sigmoid(z)))"), "wrong", "no-weight"),
    ("residual-chain", _c(step="g + w * sigmoid(z) * (1 - sigmoid(z))"), "wrong", "added"),
    ("residual-chain", "def residual_chain_gradient(ws):\n    return 1.0\n", "error", "arguments"),
    ("residual-chain", _c().replace("    return g\n", "    return [g]\n"), "wrong", None),
    ("residual-block", _B_START, "empty", None),
    ("residual-block", _b(), "correct", None),
    ("residual-block", _b("self.lin2(self.lin1(x).relu()) + x"), "correct", None),
    ("residual-block", _b("self.lin2(torch.relu(self.lin1(x)))"), "wrong", "no-skip"),
    ("residual-block", _b("torch.relu(x + self.lin2(torch.relu(self.lin1(x))))"), "wrong", "relu-after"),
    ("residual-block", _b("x + torch.relu(self.lin2(torch.relu(self.lin1(x))))"), "wrong", "relu-end"),
    ("residual-block", _b("x + self.lin2(self.lin1(x))"), "wrong", None),
    ("residual-block", _B_IN_PLACE, "wrong", "in-place"),
    ("residual-block", _b("x + self.f(x)").replace("self.lin1 = nn.Linear(d, hidden)\n        self.lin2 = nn.Linear(hidden, d)\n", "self.f = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, d))\n"), "correct", None),
    ("residual-block", _b().replace("        super().__init__()\n", ""), "error", "no-super"),
    ("residual-block", _b().replace("def forward(self, x):", "def forwrad(self, x):"), "error", "no-forward"),
]
