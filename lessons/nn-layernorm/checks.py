# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-layernorm: layer_norm, layer normalization of a batch, row by row, and PreNormBlock, a
residual block with a LayerNorm before its f, as in a transformer. And the data of the picture: a deep MLP of
residual blocks for the handwritten digits of QMNIST, trained without normalization, with post-norm and with
pre-norm."""
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
BLOCKS = 30
WIDTH = 64
LR = 0.3

class DeepNet(nn.Module):
    """An input layer, BLOCKS residual blocks with f(h) = second(relu(first(h))), and 10 logits. `norm` says where
    each block's LayerNorm goes: "pre" gives h + f(norm(h)), with one more LayerNorm before the output; "post" gives
    norm(h + f(h)); "none" gives h + f(h)."""

    def __init__(self, norm: str):
        super().__init__()
        self.inp = nn.Linear(784, WIDTH)
        self.first = nn.ModuleList([nn.Linear(WIDTH, WIDTH) for _ in range(BLOCKS)])
        self.second = nn.ModuleList([nn.Linear(WIDTH, WIDTH) for _ in range(BLOCKS)])
        self.norms = nn.ModuleList([nn.LayerNorm(WIDTH) for _ in range(BLOCKS)])
        self.final = nn.LayerNorm(WIDTH)
        self.out = nn.Linear(WIDTH, 10)
        self.norm = norm

    def forward(self, x):
        h = torch.relu(self.inp(x))
        for a, b, n in zip(self.first, self.second, self.norms):
            if self.norm == "pre":
                h = h + b(torch.relu(a(n(h))))
            elif self.norm == "post":
                h = n(h + b(torch.relu(a(h))))
            else:
                h = h + b(torch.relu(a(h)))
        if self.norm == "pre":
            h = self.final(h)
        return self.out(h)


@functools.lru_cache(maxsize=3)
def run(norm: str):
    """One epoch of SGD with learning rate LR on the 6,000 training digits: the loss of each mini-batch, and the
    accuracy on the 1,000 test digits afterwards."""
    d = np.load(paths.ROOT / DATA)
    X = torch.tensor(d["train_images"], dtype=torch.float32) / 255
    y = torch.tensor(d["train_labels"], dtype=torch.int64)
    X_test = torch.tensor(d["test_images"], dtype=torch.float32) / 255
    y_test = torch.tensor(d["test_labels"], dtype=torch.int64)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)   # faster on these small layers, and the numbers no longer depend on the number of cores
    try:
        torch.manual_seed(0)
        model = DeepNet(norm)
        loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
        opt = torch.optim.SGD(model.parameters(), lr=LR)
        loss_fn = nn.CrossEntropyLoss()
        torch.manual_seed(2)
        losses = []
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        with torch.no_grad():
            accuracy = (model(X_test).argmax(dim=1) == y_test).float().mean().item()
    finally:
        torch.set_num_threads(threads)
    return losses, accuracy


def _smooth(values, k=10):
    """The mean of the last k values at each step (fewer at the start)."""
    return [float(np.mean(values[max(0, i + 1 - k):i + 1])) for i in range(len(values))]


def loss_picture(labels: dict) -> str:
    """The training loss of the 30-block network during one epoch, with post-norm and with pre-norm."""
    from everything_ai import chart
    post, _ = run("post")
    pre, _ = run("pre")
    steps = list(range(1, len(pre) + 1))
    p = chart.Panel((0, len(pre) + 1), (0, 3.5), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    p.line(steps, _smooth(post), "loss", 2.2, dashed=True)
    p.line(steps, _smooth(pre), "loss", 2.2)
    p.text(len(pre) - 2, _smooth(post)[-1] + 0.17, labels["post"], "ink-2", "end")
    p.text(len(pre) - 2, _smooth(pre)[-1] - 0.25, labels["pre"], "ink", "end")
    return chart.figure([p], alt=labels["alt"], width=560, height=300)

# The lesson is done when these are correct. "batch-of-one" is the choice question in the lesson.
REQUIRED = ["batch-of-one", "layer-norm", "pre-norm-block"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _round(x: float) -> float | str:
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return round(x, 4) + 0.0


def _rows(v):
    """A tensor, array or list as a list of rows of plain numbers, or None when it is none of those."""
    try:
        t = v.detach().float() if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError):
        return None
    return t.tolist()


def _show(v) -> str:
    rows = _rows(v)
    if rows is None:
        return repr(v)
    if isinstance(rows, list) and rows and isinstance(rows[0], list):
        return "[" + ", ".join("[" + ", ".join(str(_round(x)) for x in row) + "]" for row in rows) + "]"
    if isinstance(rows, list):
        return "[" + ", ".join(str(_round(x)) for x in rows) + "]"
    return str(_round(rows))


def _close(got, want) -> bool:
    g, w = _rows(got), _rows(want)
    if g is None or w is None:
        return False
    try:
        g, w = np.asarray(g, dtype=float), np.asarray(w, dtype=float)
    except (TypeError, ValueError):
        return False
    return g.shape == w.shape and bool(np.allclose(g, w, atol=1e-3, equal_nan=True))


class _T:
    """A tensor for a check call, shown the way the lesson writes one: torch.tensor([...]). The learner's function
    gets a plain tensor."""

    def __init__(self, rows):
        self.rows = rows

    def __repr__(self):
        return f"torch.tensor({self.rows!r})"

    def __deepcopy__(self, memo):
        return torch.tensor(self.rows, dtype=torch.float32)


# --- exercise 2: layer normalization by hand ----------------------------------------------------------------------------

_L_HEAD = "import torch\n\n\ndef layer_norm(X, gamma, beta):\n"
_L_START = _L_HEAD + "    # standardize each row of X over its own features (the columns), with eps = 0.00001; then times gamma, plus beta\n    ...\n"


def _l(mean: str = "X.mean(dim=1, keepdim=True)", var: str = "((X - mean) ** 2).mean(dim=1, keepdim=True)",
       out: str = "gamma * (X - mean) / torch.sqrt(var + 1e-5) + beta") -> str:
    return _L_HEAD + f"    mean = {mean}\n    var = {var}\n    return {out}\n"


# --- exercise 3: a pre-norm residual block ---------------------------------------------------------------------------------

_B_HEAD = "import torch\nfrom torch import nn\n\n\nclass PreNormBlock(nn.Module):\n"
_B_START = (_B_HEAD + "    def __init__(self, d, hidden):\n        super().__init__()\n        # norm, then lin1 from d to hidden, then lin2 from hidden back to d\n        ...\n\n"
            "    def forward(self, x):\n        # x plus f of the normalized x, with f: lin1, ReLU, lin2\n        ...\n")


def _b(out: str = "x + self.lin2(torch.relu(self.lin1(self.norm(x))))") -> str:
    return (_B_HEAD + "    def __init__(self, d, hidden):\n        super().__init__()\n        self.norm = nn.LayerNorm(d)\n"
            "        self.lin1 = nn.Linear(d, hidden)\n        self.lin2 = nn.Linear(hidden, d)\n\n    def forward(self, x):\n" + f"        return {out}\n")


_B_SCENARIOS = [
    "torch.manual_seed(0)\nblock = PreNormBlock(4, 8)\nx = 5 * torch.randn(3, 4)\nblock(x).tolist()",
    "torch.manual_seed(1)\nblock = PreNormBlock(3, 5)\nx = torch.randn(2, 3) + 10\nblock(block(x)).tolist()",
    "torch.manual_seed(2)\nblock = PreNormBlock(6, 2)\nx = torch.randn(4, 6)\nblock(x[:1]).tolist()",
]

EXERCISES = {
    "layer-norm": FunctionBox(
        name="layer-norm",
        function="layer_norm",
        start=_L_START,
        solution=_l(),
        visible=[(_T([[1.0, 2.0, 6.0], [10.0, 20.0, 30.0]]), _T([1.0, 2.0, 0.5]), _T([0.0, 1.0, -1.0]))],
        hidden=[(_T([[3.0, -1.0, 5.0, 1.0], [2.0, 2.0, 2.0, 2.0], [0.0, 4.0, 1.0, 1.0]]), _T([2.0, 1.0, 0.5, 1.0]), _T([1.0, 0.0, -1.0, 0.0])),
                (_T([[-2.0, 2.0]]), _T([3.0, 1.0]), _T([0.5, 0.0])),
                (_T([[0.5, 4.0], [1.5, 2.0], [-1.0, 0.0], [3.0, 2.0]]), _T([0.5, 1.0]), _T([2.0, -2.0]))],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("no-keepdim", syntax=r"\b(?:mean|var|std)\((?![^()]*keepdim)(?:[^()]*,\s*)?(?:(?:dim|axis)\s*=\s*)?-?1\s*\)",
                    en="`X.mean(dim=1)` gives one number per row, as a flat list: PyTorch cannot take that from every row of `X`. Keep it as a column with `keepdim=True`: `X.mean(dim=1, keepdim=True)`, and the same for the variance.",
                    nl="`X.mean(dim=1)` geeft één getal per rij, als platte lijst: PyTorch kan dat niet van elke rij van `X` aftrekken. Houd het als kolom met `keepdim=True`: `X.mean(dim=1, keepdim=True)`, en net zo voor de variantie."),
            Mistake("columns", wrong=_l(mean="X.mean(dim=0)", var="((X - mean) ** 2).mean(dim=0)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you standardize each column over the examples in the batch. That is batch normalization. Layer normalization standardizes each row, one example, over its own features: `dim=1`, with `keepdim=True`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je standaardiseert elke kolom over de voorbeelden in de batch. Dat is batch normalization. Layer normalization standaardiseert elke rij, één voorbeeld, over zijn eigen features: `dim=1`, met `keepdim=True`."),
            Mistake("unbiased", wrong=_l(var="X.var(dim=1, keepdim=True)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `X.var` and `X.std` divide by the number of features minus 1. Layer normalization divides by the number itself: `((X - mean) ** 2).mean(dim=1, keepdim=True)`, or `unbiased=False`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `X.var` en `X.std` delen door het aantal features min 1. Layer normalization deelt door het aantal zelf: `((X - mean) ** 2).mean(dim=1, keepdim=True)`, of `unbiased=False`."),
            Mistake("no-scale", wrong=_l(out="(X - mean) / torch.sqrt(var + 1e-5)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: each row is standardized, but not yet rescaled. Multiply by `gamma` and add `beta`: one number per feature, the same for every example.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elke rij is gestandaardiseerd, maar nog niet herschaald. Vermenigvuldig met `gamma` en tel `beta` erbij op: één getal per feature, hetzelfde voor elk voorbeeld."),
            Mistake("no-eps", wrong=_l(out="gamma * (X - mean) / torch.sqrt(var) + beta"), state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: a row with the same number everywhere has a variance of 0, and 0 divided by 0 is `nan`. Add the small `1e-5` under the square root.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: een rij met overal hetzelfde getal heeft een variantie van 0, en 0 gedeeld door 0 is `nan`. Tel de kleine `1e-5` op onder de wortel."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: the mean and the variance of each row, the standardized `X`, and then times `gamma`, plus `beta`.",
                      "nl": "Vervang de `...` door de body van de functie: het gemiddelde en de variantie van elke rij, de gestandaardiseerde `X`, en dan keer `gamma`, plus `beta`."},
            "arguments": {"en": "The check calls `{call}`, with three tensors, and the `def` line of your function has another number of names between its parentheses. Keep `(X, gamma, beta)`.",
                          "nl": "De check roept `{call}` aan, met drie tensors, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X, gamma, beta)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Work out the first row by hand: its mean, its variance, dividing by the number of features, and then each number standardized, times its `gamma`, plus its `beta`.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken de eerste rij met de hand uit: zijn gemiddelde, zijn variantie, gedeeld door het aantal features, en dan elk getal gestandaardiseerd, keer zijn `gamma`, plus zijn `beta`."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any number of rows and features, also for one row, and for a row with the same number everywhere?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elk aantal rijen en features, ook voor één rij, en voor een rij met overal hetzelfde getal?"},
            "correct": {"en": "Your function is right in all {n} cases: every example standardized over its own features, whatever else is in the batch.",
                        "nl": "Je functie klopt in alle {n} gevallen: elk voorbeeld gestandaardiseerd over zijn eigen features, wat er verder ook in de batch zit."},
        },
    ),
    "pre-norm-block": ClassBox(
        name="pre-norm-block",
        start=_B_START,
        solution=_b(),
        scenarios=_B_SCENARIOS,
        compare=_close,
        digits=4,
        first=[_DOTS],
        mistakes=[
            Mistake("no-super", syntax=r"\A(?:(?![\s\S]*\bsuper\s*\()(?![\s\S]*Module\s*\.\s*__init__\s*\()|[\s\S]*?^[ \t]+self\.\w+[ \t]*=[^=][\s\S]*?\bsuper\s*\()",
                    en="Your `__init__` does not start with `super().__init__()`. `nn.Module` has to set itself up before your class can give it a layer: write `super().__init__()` as the first line of `__init__`, and only then make the layers.",
                    nl="Je `__init__` begint niet met `super().__init__()`. `nn.Module` moet zichzelf eerst opzetten voordat je class er een layer aan kan geven: schrijf `super().__init__()` als eerste regel van `__init__`, en maak pas daarna de layers."),
            Mistake("no-forward", syntax=r"\A(?![\s\S]*^[ \t]+def[ \t]+forward[ \t]*\()",
                    en="Your class has no method `forward`: PyTorch calls it when you write `model(x)`. Write `def forward(self, x):`, with exactly that name.",
                    nl="Je class heeft geen method `forward`: PyTorch roept die aan als je `model(x)` schrijft. Schrijf `def forward(self, x):`, met precies die naam."),
            Mistake("batchnorm", wrong=_b().replace("nn.LayerNorm(d)", "nn.BatchNorm1d(d)"),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: your `norm` is a BatchNorm, which standardizes over the examples in the batch. A pre-norm block in a transformer uses `nn.LayerNorm(d)`, over the features of each example.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: je `norm` is een BatchNorm, die standaardiseert over de voorbeelden in de batch. Een pre-norm block in een transformer gebruikt `nn.LayerNorm(d)`, over de features van elk voorbeeld."),
            Mistake("post-norm", wrong=_b("self.norm(x + self.lin2(torch.relu(self.lin1(x))))"),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: that is post-norm, the LayerNorm after the addition, as in the first transformer. This exercise asks for pre-norm: the LayerNorm on the way into `f`, and the input added unchanged: `x + f(norm(x))`.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: dat is post-norm, de LayerNorm na het optellen, zoals in de eerste transformer. Deze oefening vraagt om pre-norm: de LayerNorm op weg naar `f`, en de input ongewijzigd erbij opgeteld: `x + f(norm(x))`."),
            Mistake("norm-added", wrong=_b("self.norm(x) + self.lin2(torch.relu(self.lin1(self.norm(x))))"),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: you add the normalized `x`. The direct path carries `x` itself, unchanged; only `f` gets the normalized `x`.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: je telt de genormaliseerde `x` erbij op. De directe weg draagt `x` zelf, ongewijzigd; alleen `f` krijgt de genormaliseerde `x`."),
            Mistake("no-norm", wrong=_b("x + self.lin2(torch.relu(self.lin1(x)))"),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: `f` gets `x` itself. In a pre-norm block, `f` gets `self.norm(x)`.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: `f` krijgt `x` zelf. In een pre-norm block krijgt `f` `self.norm(x)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: in `__init__` the layers `norm`, `lin1` and `lin2`, and in `forward` the input plus `f` of the normalized input.",
                      "nl": "Vervang de twee `...`: in `__init__` de layers `norm`, `lin1` en `lin2`, en in `forward` de input plus `f` van de genormaliseerde input."},
            "wrong": {"en": "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Do you make `lin1` before `lin2`, so that they get the same random weights, does `f` get `self.norm(x)`, and is `x` itself added?",
                      "nl": "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Maak je `lin1` vóór `lin2`, zodat ze dezelfde willekeurige weights krijgen, krijgt `f` `self.norm(x)`, en wordt `x` zelf erbij opgeteld?"},
            "almost": {"en": "Right for the first block. But with these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does your block work for any `d` and `hidden`, twice in a row, and for one example on its own?",
                       "nl": "Goed voor het eerste block. Maar met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Werkt je block voor elke `d` en `hidden`, twee keer na elkaar, en voor één voorbeeld alleen?"},
            "correct": {"en": "Your block normalizes on the way into f and adds the input unchanged: a pre-norm block, as in GPT-2 and most transformers since.",
                        "nl": "Je block normaliseert op weg naar f en telt de input ongewijzigd erbij op: een pre-norm block, zoals in GPT-2 en de meeste transformers daarna."},
        },
    ),
}

CASES = [
    ("layer-norm", _L_START, "empty", None),
    ("layer-norm", _l(), "correct", None),
    ("layer-norm", _l(var="X.var(dim=1, keepdim=True, unbiased=False)"), "correct", None),
    ("layer-norm", _L_HEAD + "    return torch.nn.functional.layer_norm(X, (X.shape[1],), gamma, beta, 1e-5)\n", "correct", None),
    ("layer-norm", _L_HEAD + "    rows = []\n    for row in X:\n        m = row.mean()\n        v = ((row - m) ** 2).mean()\n        rows.append(gamma * (row - m) / torch.sqrt(v + 1e-5) + beta)\n    return torch.stack(rows)\n", "correct", None),
    ("layer-norm", _l(mean="X.mean(dim=0)", var="((X - mean) ** 2).mean(dim=0)"), "wrong", "columns"),
    ("layer-norm", _l(var="X.var(dim=1, keepdim=True)"), "wrong", "unbiased"),
    ("layer-norm", _l(out="(X - mean) / torch.sqrt(var + 1e-5)"), "wrong", "no-scale"),
    ("layer-norm", _l(out="gamma * (X - mean) / torch.sqrt(var) + beta"), "almost", "no-eps"),
    ("layer-norm", _l(mean="X.mean(dim=1)", var="((X - mean) ** 2).mean(dim=1)"), "error", "no-keepdim"),
    ("layer-norm", _l(mean="X.mean(axis=1)", var="((X - mean) ** 2).mean(axis=1)"), "error", "no-keepdim"),
    ("layer-norm", _l(mean="torch.mean(X, dim=1)"), "error", "no-keepdim"),
    ("layer-norm", "def layer_norm(X):\n    return X\n", "error", "arguments"),
    ("pre-norm-block", _B_START, "empty", None),
    ("pre-norm-block", _b(), "correct", None),
    ("pre-norm-block", _b("self.lin2(self.lin1(self.norm(x)).relu()) + x"), "correct", None),
    ("pre-norm-block", _b("self.norm(x + self.lin2(torch.relu(self.lin1(x))))"), "wrong", "post-norm"),
    ("pre-norm-block", _b("self.norm(x) + self.lin2(torch.relu(self.lin1(self.norm(x))))"), "wrong", "norm-added"),
    ("pre-norm-block", _b("x + self.lin2(torch.relu(self.lin1(x)))"), "wrong", "no-norm"),
    ("pre-norm-block", _b("self.lin2(torch.relu(self.lin1(self.norm(x))))"), "wrong", None),
    ("pre-norm-block", _b().replace("        super().__init__()\n", ""), "error", "no-super"),
    ("pre-norm-block", _b().replace("def forward(self, x):", "def forwrad(self, x):"), "error", "no-forward"),
    ("pre-norm-block", _b().replace("nn.LayerNorm(d)", "nn.BatchNorm1d(d)"), "wrong", "batchnorm"),
]
