# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson attn-multi-head: multi_head, causal attention in h heads that each take their own columns of
Q, K and V, with their outputs side by side; and MultiHeadAttention, the same as a PyTorch module with learned
matrices and an output projection."""
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import draw
from everything_ai.check import ClassBox, FunctionBox, Mistake

SENTENCE = ["the", "cat", "sat", "down"]


def _causal(q, k, v):
    n, d_k = q.shape
    S = q @ k.T / math.sqrt(d_k)
    S = S.masked_fill(torch.tril(torch.ones(n, n)) == 0, float("-inf"))
    return torch.softmax(S, dim=1) @ v


def head_weights() -> list:
    """The causal attention weights of two heads of "the cat sat down", with made-up queries and keys of one number:
    every query is 3, and the key is 1 for the word a head looks for, "cat" (a thing) or "sat" (an action), else 0."""
    out = []
    for target in (1, 2):
        q = torch.full((4, 1), 3.0)
        k = torch.zeros(4, 1)
        k[target, 0] = 1.0
        S = (q @ k.T).masked_fill(torch.tril(torch.ones(4, 4)) == 0, float("-inf"))
        out.append(torch.softmax(S, dim=1).numpy())
    return out


def heads_pictures(labels: dict):
    """The attention weights of the two heads, one grid each."""
    common = {"row_label": labels["rows"], "col_label": labels["cols"], "decimal": labels["decimal"]}
    one, two = head_weights()
    return [draw.weight_grid(SENTENCE, one, title=labels["one"], alt=labels["one_alt"], **common),
            draw.weight_grid(SENTENCE, two, title=labels["two"], alt=labels["two_alt"], **common)]


# The lesson is done when these are correct. "head-weights" is the choice question in the lesson.
REQUIRED = ["head-weights", "multi-head", "multi-head-module"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_SCENARIO = "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_SCENARIO_NL = "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"


def _array(v):
    if isinstance(v, torch.Tensor):
        return np.asarray(v.detach(), dtype=float)
    return np.asarray(v, dtype=float)


def _grid(v):
    try:
        a = _array(v)
    except (TypeError, ValueError):
        return repr(v)
    if np.all(np.isfinite(a)) and np.all(a == np.round(a)):
        return repr(a.astype(int).tolist())
    return repr((np.round(a, 4) + 0.0).tolist())


def _same(got, want) -> bool:
    try:
        g, w = _array(got), _array(want)
    except (TypeError, ValueError):
        return False
    return g.shape == w.shape and bool(np.allclose(g, w, rtol=1e-4, atol=1e-4, equal_nan=True))


def _close(got, want) -> bool:
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    try:
        return bool(np.allclose(float(got), float(want), rtol=1e-4, atol=1e-4))
    except (TypeError, ValueError):
        return got == want


_HELPER = ("import math\nimport torch\n\n\ndef causal_attention(q, k, v):\n    n, d_k = q.shape\n    S = q @ k.T / math.sqrt(d_k)\n"
           "    S = S.masked_fill(torch.tril(torch.ones(n, n)) == 0, float(\"-inf\"))\n    return torch.softmax(S, dim=1) @ v\n\n\n")
_LOOP = ("        a, b = i * d_head, (i + 1) * d_head\n"
         "        outputs.append(causal_attention(Q[:, a:b], K[:, a:b], V[:, a:b]))\n")

# --- exercise 2: multi_head -----------------------------------------------------------------------------------------------

_H_HEAD = (_HELPER + "def multi_head(Q, K, V, h):\n"
           "    Q, K, V = torch.tensor(Q, dtype=torch.float32), torch.tensor(K, dtype=torch.float32), torch.tensor(V, dtype=torch.float32)\n"
           "    d_head = int(Q.shape[1] / h)\n    outputs = []\n    for i in range(h):\n")
_H_START = (_H_HEAD + "        # head i: its own columns of Q, K and V, through causal attention\n        ...\n"
            "    # the outputs of all heads side by side\n    return ...\n")


def _h(loop: str = _LOOP, out: str = "torch.cat(outputs, dim=1)") -> str:
    return _H_HEAD + loop + f"    return {out}\n"


_QKV = [[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 2.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
_V3 = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]

# --- exercise 3: MultiHeadAttention ---------------------------------------------------------------------------------------

_M_HEAD = (_HELPER.replace("import math\nimport torch\n", "import math\nimport torch\nfrom torch import nn\n")
           + "class MultiHeadAttention(nn.Module):\n    def __init__(self, d, h):\n        super().__init__()\n"
           "        self.h = h\n        self.d_head = int(d / h)\n")
_M_INIT = ("        self.query = nn.Linear(d, d, bias=False)\n        self.key = nn.Linear(d, d, bias=False)\n"
           "        self.value = nn.Linear(d, d, bias=False)\n        self.out = nn.Linear(d, d)\n")
_M_FORWARD = ("        q, k, v = self.query(x), self.key(x), self.value(x)\n        outputs = []\n        for i in range(self.h):\n"
              "            a, b = i * self.d_head, (i + 1) * self.d_head\n"
              "            outputs.append(causal_attention(q[:, a:b], k[:, a:b], v[:, a:b]))\n"
              "        return self.out(torch.cat(outputs, dim=1))\n")
_M_START = (_M_HEAD + "        # queries, keys and values for all heads at once, d to d numbers without bias terms,\n"
            "        # and the output projection, d to d: query, key, value and out, in that order\n        ...\n\n"
            "    def forward(self, x):\n        q, k, v = self.query(x), self.key(x), self.value(x)\n"
            "        # every head on its own columns, the outputs side by side, then the output projection\n        ...\n")


def _m(init: str = _M_INIT, forward: str = _M_FORWARD) -> str:
    return _M_HEAD + init + "\n    def forward(self, x):\n" + forward


_M_SCENARIOS = [
    # 4 positions with 6 numbers each, in 2 heads of 3: the shape, and the new vector of the last position
    "import torch\n\ntorch.manual_seed(0)\nattention = MultiHeadAttention(6, 2)\nx = torch.rand(4, 6)\ny = attention(x)\n[list(y.shape), y[3].tolist()]",
    # 3 heads of 2, and the second position: the first only takes from itself, the same with any number of heads.
    # Larger inputs make sharper weights, so that the number of heads shows in the first decimals.
    "import torch\n\ntorch.manual_seed(1)\nattention = MultiHeadAttention(6, 3)\nx = 4 * torch.rand(2, 6)\ny = attention(x)\ny[1].tolist()",
    # the shapes of the parameters: four matrices of 6 by 6, and the bias terms of the output projection
    "torch.manual_seed(0)\nattention = MultiHeadAttention(6, 2)\n[list(p.shape) for p in attention.parameters()]",
]


def _m_order(got, text) -> bool:
    """The four layers made in another order than query, key, value, out, and nothing else wrong: with the lines
    that make them put in that order, the class gives what the course's solution gives in every scenario."""
    lines = text.split("\n")
    rows = []
    for name in ("query", "key", "value", "out"):
        found = [i for i, line in enumerate(lines) if line.strip().startswith(f"self.{name} =")]
        if len(found) != 1:
            return False
        rows.append(found[0])
    if rows == sorted(rows):
        return False
    fixed = list(lines)
    for row, i in zip(sorted(rows), rows):
        fixed[row] = lines[i]
    return EXERCISES["multi-head-module"]._same_everywhere("\n".join(fixed), _m())


EXERCISES = {
    "multi-head": FunctionBox(
        name="multi-head",
        function="multi_head",
        start=_H_START,
        solution=_h(),
        visible=[(_QKV, _QKV, _V3, 2)],
        hidden=[(_QKV, _QKV, _V3, 1), (_QKV, _QKV, _V3, 4),
                ([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]], [[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]],
                 [[2.0, 4.0, 6.0, 8.0, 10.0, 12.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], 3),
                ([[0.5, -1.0], [2.0, 1.0]], [[1.0, 1.0], [0.0, 3.0]], [[1.0, 0.0], [0.0, 1.0]], 2)],
        compare=_same,
        show=_grid,
        first=[_DOTS],
        mistakes=[
            Mistake("one-head", wrong=_h(loop="        outputs.append(causal_attention(Q, K, V))\n", out="outputs[0]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: one attention over all columns at once. Every head takes its own block of `d_head` columns: head `i` the columns from `i * d_head` up to `(i + 1) * d_head`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: één attention over alle kolommen tegelijk. Elke head neemt zijn eigen blok van `d_head` kolommen: head `i` de kolommen van `i * d_head` tot `(i + 1) * d_head`."),
            Mistake("first-head", wrong=_h(loop="        outputs.append(causal_attention(Q[:, :d_head], K[:, :d_head], V[:, :d_head]))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every head takes the columns of the first head. The block of head `i` starts at column `i * d_head`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elke head neemt de kolommen van de eerste head. Het blok van head `i` begint bij kolom `i * d_head`."),
            Mistake("stacked", wrong=_h(out="torch.cat(outputs, dim=0)"),
                    en="`{call}` gives a tensor of shape (6, 2): you put the outputs of the heads under each other. Every position needs the outputs of all heads side by side, in one row: `torch.cat(outputs, dim=1)`.",
                    nl="`{call}` geeft een tensor met shape (6, 2): je zet de outputs van de heads onder elkaar. Elke positie heeft de outputs van alle heads naast elkaar nodig, in één rij: `torch.cat(outputs, dim=1)`."),
            Mistake("rows", syntax=r"^[^#\n]*\b[QKV]\[\s*a\s*:\s*b\s*(?:,\s*:\s*)?\]",
                    en="`{line}` cuts rows, the positions, and not columns. A head keeps every position and takes its own columns: `Q[:, a:b]`.",
                    nl="`{line}` knipt rijen, de posities, en geen kolommen. Een head houdt elke positie en neemt zijn eigen kolommen: `Q[:, a:b]`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the attention of head `i` on its own columns, added to `outputs`, and the outputs side by side.",
                      "nl": "Vervang de twee `...`: de attention van head `i` op zijn eigen kolommen, toegevoegd aan `outputs`, en de outputs naast elkaar."},
            "arguments": {"en": "The check calls `{call}`, with queries, keys, values and a number of heads, and the `def` line of your function has another number of names between its parentheses. Keep `(Q, K, V, h)`.",
                          "nl": "De check roept `{call}` aan, met queries, keys, values en een aantal heads, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(Q, K, V, h)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your lines with steps 2 and 3 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je regels met stap 2 en 3 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any number of heads, with `i` and `d_head`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elk aantal heads, met `i` en `d_head`?"},
            "correct": {"en": "Your function runs h attentions side by side, each on its own columns. With 1 head it is the attention of the last lesson; with 4 heads of 1 number, every head can only compare one number.",
                        "nl": "Je functie draait h attentions naast elkaar, elk op zijn eigen kolommen. Met 1 head is het de attention van de vorige les; met 4 heads van 1 getal kan elke head maar één getal vergelijken."},
        },
    ),
    "multi-head-module": ClassBox(
        name="multi-head-module",
        start=_M_START,
        solution=_m(),
        scenarios=_M_SCENARIOS,
        compare=_close,
        digits=4,
        first=[_DOTS],
        mistakes=[
            Mistake("no-out", wrong=_m(forward=_M_FORWARD.replace("return self.out(torch.cat(outputs, dim=1))", "return torch.cat(outputs, dim=1)")),
                    en=_SCENARIO + ": the outputs of the heads come out side by side, without the output projection. Send them through `self.out`, which mixes the heads: `self.out(torch.cat(outputs, dim=1))`.",
                    nl=_SCENARIO_NL + ": de outputs van de heads komen naast elkaar naar buiten, zonder de output-projectie. Stuur ze door `self.out`, die de heads mengt: `self.out(torch.cat(outputs, dim=1))`."),
            Mistake("order", matches=_m_order,
                    en=_SCENARIO + ": the right shape, other numbers. Make the four layers in the order query, key, value, out, so that they get the same random starting values as in the course's solution; the rest of your class is right.",
                    nl=_SCENARIO_NL + ": de goede shape, andere getallen. Maak de vier layers in de volgorde query, key, value, out, zodat ze dezelfde willekeurige startwaarden krijgen als in de oplossing van de course; de rest van je class klopt."),
            Mistake("one-head", wrong=_m(forward=("        q, k, v = self.query(x), self.key(x), self.value(x)\n"
                                                  "        return self.out(causal_attention(q, k, v))\n")),
                    en=_SCENARIO + ": one attention over all columns at once. Every head takes its own block of `self.d_head` columns, in a loop over `range(self.h)`.",
                    nl=_SCENARIO_NL + ": één attention over alle kolommen tegelijk. Elke head neemt zijn eigen blok van `self.d_head` kolommen, in een loop over `range(self.h)`."),
            Mistake("bias", wrong=_m(init=_M_INIT.replace(", bias=False)", ")")),
                    en=_SCENARIO + ": your query, key and value layers have bias terms. Make them with `bias=False`, as in the comment; only the output projection keeps its bias terms.",
                    nl=_SCENARIO_NL + ": je query-, key- en value-layers hebben bias terms. Maak ze met `bias=False`, zoals het commentaar zegt; alleen de output-projectie houdt zijn bias terms."),
            Mistake("rows", syntax=r"^[^#\n]*\b[qkv]\[\s*a\s*:\s*b\s*(?:,\s*:\s*)?\]",
                    en="`{line}` cuts rows, the positions, and not columns. A head keeps every position and takes its own columns: `q[:, a:b]`.",
                    nl="`{line}` knipt rijen, de posities, en geen kolommen. Een head houdt elke positie en neemt zijn eigen kolommen: `q[:, a:b]`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the four layers in `__init__`, and in `forward` the heads, side by side, through the output projection.",
                      "nl": "Vervang de twee `...`: de vier layers in `__init__`, en in `forward` de heads, naast elkaar, door de output-projectie."},
            "wrong": {"en": _SCENARIO + ". The four layers, then steps 2 to 4 of the recipe: a loop over the heads on their own columns, and the output projection of all heads side by side?",
                      "nl": _SCENARIO_NL + ". De vier layers, en dan stap 2 tot en met 4 van het recept: een loop over de heads op hun eigen kolommen, en de output-projectie van alle heads naast elkaar?"},
            "almost": {"en": "Right for 2 heads. But " + _SCENARIO[0].lower() + _SCENARIO[1:] + ". Does your class use `self.h` and `self.d_head`, no fixed numbers, and only the four layers?",
                       "nl": "Goed voor 2 heads. Maar " + _SCENARIO_NL[0].lower() + _SCENARIO_NL[1:] + ". Gebruikt je class `self.h` en `self.d_head`, geen vaste getallen, en alleen de vier layers?"},
            "correct": {"en": "Your module is multi-head attention with a causal mask: 4 matrices of d by d and d bias terms, 150 parameters for d = 6, whatever the number of heads.",
                        "nl": "Je module is multi-head attention met een causal mask: 4 matrices van d bij d en d bias terms, 150 parameters bij d = 6, wat het aantal heads ook is."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("multi-head", _H_START, "empty", None),
    ("multi-head", _h(loop="        outputs.append(causal_attention(Q[:, i * d_head:(i + 1) * d_head], K[:, i * d_head:(i + 1) * d_head], V[:, i * d_head:(i + 1) * d_head]))\n"), "correct", None),
    ("multi-head", _h(loop="        outputs.append(causal_attention(Q, K, V))\n", out="outputs[0]"), "wrong", "one-head"),
    ("multi-head", _h(loop="        outputs.append(causal_attention(Q[:, :d_head], K[:, :d_head], V[:, :d_head]))\n"), "wrong", "first-head"),
    ("multi-head", _h(out="torch.cat(outputs, dim=0)"), "wrong", "stacked"),
    ("multi-head", _h(loop="        a, b = i * d_head, (i + 1) * d_head\n        outputs.append(causal_attention(Q[a:b], K[a:b], V[a:b]))\n"), "error", "rows"),
    ("multi-head-module", _M_START, "empty", None),
    ("multi-head-module", _m(forward=("        q, k, v = self.query(x), self.key(x), self.value(x)\n"
                                       "        heads = [causal_attention(q[:, i * self.d_head:(i + 1) * self.d_head], k[:, i * self.d_head:(i + 1) * self.d_head], v[:, i * self.d_head:(i + 1) * self.d_head]) for i in range(self.h)]\n"
                                       "        return self.out(torch.cat(heads, dim=1))\n")), "correct", None),
    ("multi-head-module", _m(forward=_M_FORWARD.replace("return self.out(torch.cat(outputs, dim=1))", "return torch.cat(outputs, dim=1)")), "wrong", "no-out"),
    ("multi-head-module", _m(init=("        self.out = nn.Linear(d, d)\n        self.query = nn.Linear(d, d, bias=False)\n"
                                   "        self.key = nn.Linear(d, d, bias=False)\n        self.value = nn.Linear(d, d, bias=False)\n")), "wrong", "order"),
    ("multi-head-module", _m(init=("        self.out = nn.Linear(d, d)\n        self.query = nn.Linear(d, d, bias=False)\n"
                                   "        self.key = nn.Linear(d, d, bias=False)\n        self.value = nn.Linear(d, d, bias=False)\n"),
                             forward=_M_FORWARD.replace("return self.out(torch.cat(outputs, dim=1))", "return sum(outputs)")), "wrong", None),
    ("multi-head-module", _m(forward=_M_FORWARD.replace("range(self.h)", "range(2)").replace("i * self.d_head, (i + 1) * self.d_head", "i * 3, (i + 1) * 3")), "almost", None),
    ("multi-head-module", _m(forward=("        q, k, v = self.query(x), self.key(x), self.value(x)\n"
                                       "        return self.out(causal_attention(q, k, v))\n")), "wrong", "one-head"),
    ("multi-head-module", _m(init=_M_INIT.replace(", bias=False)", ")")), "wrong", "bias"),
    ("multi-head-module", _m(forward=_M_FORWARD.replace("q[:, a:b], k[:, a:b], v[:, a:b]", "q[a:b], k[a:b], v[a:b]")), "error", "rows"),
]
