# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-embedding-layer: sgd_step, one step of gradient descent on an embedding table, where only
the rows that were looked up change and a row looked up twice gets both gradients; and CharBigram, the bigram model
with nn.Embedding and nn.Linear. And the picture: that model trained on all character pairs of Tiny Shakespeare with
embeddings of 1 to 32 numbers, against the table of 65 by 65 of the bigram project."""

import functools
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn

from everything_ai import paths
from everything_ai.check import ClassBox, FunctionBox, Mistake

DATA = "data/tiny-shakespeare.txt"   # about 1.1 MB of plays by Shakespeare: see data/README.md
DIMS = [1, 2, 4, 8, 16, 32]
STEPS, LR = 1500, 0.05


@functools.lru_cache(maxsize=1)
def _counts():
    text = (paths.ROOT / DATA).read_text(encoding="utf-8")
    chars = sorted(set(text))
    index = {c: i for i, c in enumerate(chars)}
    ids = np.array([index[c] for c in text])
    counts = np.zeros((len(chars), len(chars)))
    np.add.at(counts, (ids[:-1], ids[1:]), 1)
    return chars, counts


def table_loss() -> float:
    """The NLL of the counted bigram table on all pairs: the best a bigram model can do."""
    _, counts = _counts()
    rows = counts.sum(axis=1, keepdims=True)
    p = np.divide(counts, rows, out=np.zeros_like(counts), where=rows > 0)
    return float(-(counts[counts > 0] * np.log(p[counts > 0])).sum() / counts.sum())


@functools.lru_cache(maxsize=len(DIMS))
def train(dim: int):
    """The bigram model with an embedding layer of `dim` numbers, trained with Adam on all character pairs at once
    (the pairs counted, so one step sees them all). Gives its NLL and its embedding table."""
    chars, counts = _counts()
    C = torch.tensor(counts, dtype=torch.float32)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)   # the same numbers on every computer
    try:
        torch.manual_seed(0)
        emb = nn.Embedding(len(chars), dim)
        out = nn.Linear(dim, len(chars))
        opt = torch.optim.Adam(list(emb.parameters()) + list(out.parameters()), lr=LR)
        every = torch.arange(len(chars))
        for _ in range(STEPS):
            loss = -(C * torch.log_softmax(out(emb(every)), dim=1)).sum() / C.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    finally:
        torch.set_num_threads(threads)
    return loss.item(), emb.weight.detach().numpy().copy()


def neighbours(dim: int, char: str, k: int = 4) -> str:
    """The k characters whose embeddings have the largest cosine similarity with that of `char`."""
    chars, _ = _counts()
    _, E = train(dim)
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    sims = En @ En[chars.index(char)]
    order = [i for i in np.argsort(-sims) if chars[i] != char][:k]
    return " ".join(repr(chars[i])[1:-1] for i in order)


def dims_picture(labels: dict) -> str:
    """The loss after training against the number of numbers per character, with the table's loss as a line."""
    from everything_ai import chart
    losses = [train(d)[0] for d in DIMS]
    best = table_loss()
    panel = chart.Panel((0.8, 40), (2.4, 3.0), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], xlog=True)
    panel.xticks = DIMS
    panel.line([0.8, 40], [best, best], "loss", 1.4, dashed=True)
    panel.text(1.0, best - 0.035, labels["table"], "loss", "start")
    panel.line(DIMS, losses, "loss", 2.4)
    panel.points(DIMS, losses, "loss")
    return chart.figure([panel], alt=labels["alt"], width=330, height=260)


# The lesson is done when these are correct. "why-lookup" is the choice question in the lesson.
REQUIRED = ["why-lookup", "sgd-step", "char-bigram"]

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
    if abs(x) < 1e-12:
        return "0"
    return np.format_float_positional(x, precision=4, unique=False, fractional=False, trim="-").rstrip(".")


def _nested(v) -> str:
    if np.ndim(v) == 0:
        return _num(v)
    return "[" + ", ".join(_nested(x) for x in v) + "]"


class _Array(np.ndarray):
    """A table or a list of ids that a check message writes the way the lesson's code does: np.array([...]). The
    learner's code gets a plain copy."""

    def __repr__(self):
        return f"np.array({_nested(np.asarray(self).tolist())})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _a(values, dtype=float) -> np.ndarray:
    return np.array(values, dtype=dtype).view(_Array)


def _show(v) -> str:
    try:
        return "np.array(" + _nested(np.asarray(v, dtype=float).tolist()) + ")"
    except (TypeError, ValueError):
        return repr(v)


def _close(got, want) -> bool:
    try:
        g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    except (TypeError, ValueError):
        return False
    return g.shape == w.shape and bool(np.allclose(g, w, rtol=1e-6, atol=1e-9))


# --- exercise 2: one step of gradient descent on the table -------------------------------------------------------------------

_S_HEAD = "import numpy as np\n\n\ndef sgd_step(E, ids, grad_out, lr):\n"
_S_START = (_S_HEAD + "    # the gradient of the table: row ids[k] gets grad_out[k], added up for ids that come more than once\n"
            "    grad_E = np.zeros(E.shape)\n    ...\n    # one step of gradient descent on the whole table\n    return ...\n")


def _s(add: str = "grad_E[ids[k]] = grad_E[ids[k]] + grad_out[k]", step: str = "E - lr * grad_E") -> str:
    return (_S_HEAD + "    grad_E = np.zeros(E.shape)\n    for k in range(len(ids)):\n" + f"        {add}\n" + f"    return {step}\n")


_VISIBLE = (_a([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), _a([0, 0], int), _a([[1.0, 0.0], [0.0, 1.0]]), 0.5)
_HIDDEN = [
    (_a([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), _a([2, 0], int), _a([[1.0, 0.0], [0.0, 1.0]]), 0.5),
    (_a([[1.0, -1.0, 0.5], [0.0, 2.0, 1.0], [3.0, 0.0, -2.0], [1.0, 1.0, 1.0]]), _a([3, 1, 3, 3], int),
     _a([[1.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.0, 2.0, 0.0], [-1.0, 0.0, 1.0]]), 0.1),
    (_a([[0.5], [1.5]]), _a([1], int), _a([[2.0]]), 1.0),
    (_a([[0.0, 1.0], [1.0, 0.0]]), _a([0, 1, 0, 1, 0], int), _a([[1.0, 1.0]] * 5), 0.01),
]

# --- exercise 3: the bigram model with an embedding layer --------------------------------------------------------------------

_B_HEAD = "import torch\nfrom torch import nn\n\n\nclass CharBigram(nn.Module):\n"
_B_START = (_B_HEAD + "    def __init__(self, vocab, dim):\n        super().__init__()\n"
            "        # first the embedding layer: vocab rows of dim numbers; then a linear layer from dim to vocab logits\n        ...\n\n"
            "    def forward(self, x):\n        # x holds ids; give back one row of vocab logits per id\n        ...\n")


def _b(init: str = "        self.emb = nn.Embedding(vocab, dim)\n        self.out = nn.Linear(dim, vocab)\n",
       forward: str = "        return self.out(self.emb(x))\n") -> str:
    return (_B_HEAD + "    def __init__(self, vocab, dim):\n        super().__init__()\n" + init + "\n    def forward(self, x):\n" + forward)


_B_SCENARIOS = [
    "torch.manual_seed(0)\nmodel = CharBigram(5, 2)\nmodel(torch.tensor([0, 3, 3])).tolist()",
    "torch.manual_seed(1)\nmodel = CharBigram(65, 8)\n[[list(p.shape) for p in model.parameters()], list(model(torch.tensor([[1, 2], [64, 0]])).shape)]",
]

_NO_SUPER = Mistake("no-super", syntax=r"\A(?:(?![\s\S]*\bsuper\s*\()(?![\s\S]*Module\s*\.\s*__init__\s*\()|[\s\S]*?^[ \t]+self\.\w+[ \t]*=[^=][\s\S]*?\bsuper\s*\()",
                    en="Your `__init__` does not start with `super().__init__()`. `nn.Module` has to set itself up before your class can give it a layer: write `super().__init__()` as the first line of `__init__`, and only then make the layers.",
                    nl="Je `__init__` begint niet met `super().__init__()`. `nn.Module` moet zichzelf eerst opzetten voordat je class er een layer aan kan geven: schrijf `super().__init__()` als eerste regel van `__init__`, en maak pas daarna de layers.")
_NO_FORWARD = Mistake("no-forward", syntax=r"\A(?![\s\S]*^[ \t]+def[ \t]+forward[ \t]*\()",
                      en="Your class has no method `forward`: PyTorch calls it when you write `model(x)`. Write `def forward(self, x):`, with exactly that name.",
                      nl="Je class heeft geen method `forward`: PyTorch roept die aan als je `model(x)` schrijft. Schrijf `def forward(self, x):`, met precies die naam.")

EXERCISES = {
    "sgd-step": FunctionBox(
        name="sgd-step",
        function="sgd_step",
        start=_S_START,
        solution=_s(),
        visible=[_VISIBLE],
        hidden=_HIDDEN,
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("overwrite", wrong=_s(add="grad_E[ids[k]] = grad_out[k]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: a row that is looked up twice only gets its last gradient. `+=` on several rows at once, as in `grad_E[ids] += grad_out`, does the same. Add the gradients up: `grad_E[ids[k]] = grad_E[ids[k]] + grad_out[k]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: een rij die twee keer wordt opgezocht, krijgt alleen zijn laatste gradient. `+=` op een paar rijen tegelijk, zoals in `grad_E[ids] += grad_out`, doet hetzelfde. Tel de gradients op: `grad_E[ids[k]] = grad_E[ids[k]] + grad_out[k]`."),
            Mistake("uphill", wrong=_s(step="E + lr * grad_E"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you step with the gradient, uphill. Gradient descent goes against it: `E - lr * grad_E`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je stapt met de gradient mee, bergop. Gradient descent gaat ertegenin: `E - lr * grad_E`."),
            Mistake("no-lr", wrong=_s(step="E - grad_E"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your step leaves out the learning rate. Multiply the gradient by `lr`: `E - lr * grad_E`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je stap laat de learning rate weg. Vermenigvuldig de gradient met `lr`: `E - lr * grad_E`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: a loop that adds `grad_out[k]` to row `ids[k]` of `grad_E`, and the table after one step of gradient descent.",
                      "nl": "Vervang de twee `...`: een loop die `grad_out[k]` optelt bij rij `ids[k]` van `grad_E`, en de tabel na één stap gradient descent."},
            "arguments": {"en": "The check calls `{call}`, with a table, the ids, their gradients and a learning rate, and the `def` line of your function has another number of names between its parentheses. Keep `(E, ids, grad_out, lr)`.",
                          "nl": "De check roept `{call}` aan, met een tabel, de ids, hun gradients en een learning rate, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(E, ids, grad_out, lr)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does row `ids[k]` of `grad_E` get `grad_out[k]` for every `k`, and do the rows that were not looked up stay as they are?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Krijgt rij `ids[k]` van `grad_E` voor elke `k` de `grad_out[k]`, en blijven de rijen die niet werden opgezocht zoals ze zijn?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any ids, in any order, and any number of numbers per row?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor alle ids, in elke volgorde, en elk aantal getallen per rij?"},
            "correct": {"en": "Your step is right in all {n} cases: only the rows that were looked up change, and a row looked up more than once gets all its gradients.",
                        "nl": "Je stap klopt in alle {n} gevallen: alleen de rijen die werden opgezocht veranderen, en een rij die vaker werd opgezocht krijgt al zijn gradients."},
        },
    ),
    "char-bigram": ClassBox(
        name="char-bigram",
        start=_B_START,
        solution=_b(),
        scenarios=_B_SCENARIOS,
        digits=4,
        first=[_DOTS],
        mistakes=[
            _NO_SUPER,
            _NO_FORWARD,
            Mistake("table-too-small", error=lambda e: isinstance(e, IndexError) and "index out of range" in str(e),
                    en="Python stops with an error: an id has no row in your embedding table. The table needs one row for every id, from 0 to `vocab - 1`: `nn.Embedding(vocab, dim)`, `vocab` rows of `dim` numbers, with the `vocab` the model gets.",
                    nl="Python stopt met een fout: een id heeft geen rij in je embedding-tabel. De tabel heeft één rij nodig voor elke id, van 0 tot en met `vocab - 1`: `nn.Embedding(vocab, dim)`, `vocab` rijen van `dim` getallen, met de `vocab` die het model krijgt."),
            Mistake("sizes", error=lambda e: isinstance(e, RuntimeError) and "cannot be multiplied" in str(e),
                    en="PyTorch stops: the sizes of the layers do not fit. Each embedding has `dim` numbers, and the linear layer goes from those `dim` numbers to `vocab` logits: `nn.Embedding(vocab, dim)`, then `nn.Linear(dim, vocab)`.",
                    nl="PyTorch stopt: de groottes van de layers passen niet. Elke embedding heeft `dim` getallen, en de linear layer gaat van die `dim` getallen naar `vocab` logits: `nn.Embedding(vocab, dim)`, dan `nn.Linear(dim, vocab)`."),
            Mistake("not-ids", error=lambda e: isinstance(e, RuntimeError) and "'indices'" in str(e),
                    en="PyTorch stops: `nn.Embedding` gets numbers that are not ids. It looks up whole numbers, the ids themselves, not floats or one-hot vectors: give it `x` as it comes in.",
                    nl="PyTorch stopt: `nn.Embedding` krijgt getallen die geen ids zijn. Hij zoekt hele getallen op, de ids zelf, geen kommagetallen of one-hot-vectoren: geef hem `x` zoals die binnenkomt."),
            Mistake("out-first", wrong=_b(init="        self.out = nn.Linear(dim, vocab)\n        self.emb = nn.Embedding(vocab, dim)\n"),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: the layers are right, but you make the linear layer first, so it draws other random starting numbers than the check's model. Make the embedding layer first, as the task says.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: de layers kloppen, maar je maakt de linear layer eerst, dus die trekt andere willekeurige startgetallen dan het model van de check. Maak eerst de embedding layer, zoals de opdracht zegt."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: in `__init__` the embedding layer and then the linear layer; in `forward` the logits for the ids in `x`.",
                      "nl": "Vervang de twee `...`: in `__init__` de embedding layer en dan de linear layer; in `forward` de logits voor de ids in `x`."},
            "wrong": {"en": "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does `__init__` make `nn.Embedding(vocab, dim)` and then `nn.Linear(dim, vocab)`, and does `forward` look up the ids and then give the rows to the linear layer?",
                      "nl": "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Maakt `__init__` `nn.Embedding(vocab, dim)` en dan `nn.Linear(dim, vocab)`, en zoekt `forward` de ids op en geeft hij de rijen door aan de linear layer?"},
            "almost": {"en": "Right for the first model. But with these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: the shapes of its parameters, and the shape of the logits for ids in a table of 2 by 2. Does your model use `vocab` and `dim` instead of fixed numbers, and work for ids of any shape?",
                       "nl": "Goed voor het eerste model. Maar met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: de shapes van zijn parameters, en de shape van de logits voor ids in een tabel van 2 bij 2. Gebruikt je model `vocab` en `dim` in plaats van vaste getallen, en werkt het voor ids van elke shape?"},
            "correct": {"en": "Your model looks up a row of dim numbers for every id and turns it into vocab logits: the bigram model with an embedding layer.",
                        "nl": "Je model zoekt voor elke id een rij van dim getallen op en maakt daar vocab logits van: het bigram-model met een embedding layer."},
        },
    ),
}

CASES = [
    ("sgd-step", _S_START, "empty", None),
    ("sgd-step", _s(), "correct", None),
    ("sgd-step", _S_HEAD + "    grad_E = np.zeros(E.shape)\n    np.add.at(grad_E, ids, grad_out)\n    return E - lr * grad_E\n", "correct", None),
    ("sgd-step", _S_HEAD + "    E = E.copy()\n    for k in range(len(ids)):\n        E[ids[k]] -= lr * grad_out[k]\n    return E\n", "correct", None),
    ("sgd-step", _s(add="grad_E[ids[k]] = grad_out[k]"), "wrong", "overwrite"),
    ("sgd-step", _s(step="E + lr * grad_E"), "wrong", "uphill"),
    ("sgd-step", _s(step="E - grad_E"), "wrong", "no-lr"),
    ("sgd-step", "def sgd_step(E, ids):\n    return E\n", "error", "arguments"),
    ("char-bigram", _B_START, "empty", None),
    ("char-bigram", _b(), "correct", None),
    ("char-bigram", _b(forward="        e = self.emb(x)\n        return self.out(e)\n"), "correct", None),
    ("char-bigram", _b(init="        self.emb = nn.Embedding(dim, vocab)\n        self.out = nn.Linear(vocab, vocab)\n"), "error", "table-too-small"),
    ("char-bigram", _b(init="        self.out = nn.Linear(dim, vocab)\n        self.emb = nn.Embedding(vocab, dim)\n"), "wrong", "out-first"),
    ("char-bigram", _b(init="        self.emb = nn.Embedding(5, 2)\n        self.out = nn.Linear(2, 5)\n"), "error", "table-too-small"),
    ("char-bigram", _b(init="        self.emb = nn.Embedding(vocab, 2)\n        self.out = nn.Linear(2, vocab)\n"), "almost", None),
    ("char-bigram", _b().replace("        super().__init__()\n", ""), "error", "no-super"),
    ("char-bigram", _b(init="        self.emb = nn.Embedding(vocab, dim)\n        self.out = nn.Linear(vocab, dim)\n"), "error", "sizes"),
    ("char-bigram", _b(forward="        return self.out(self.emb(x.float()))\n"), "error", "not-ids"),
]
