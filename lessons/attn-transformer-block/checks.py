# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson attn-transformer-block: CausalSelfAttention, multi-head causal attention for a mini-batch of
texts of shape (B, n, d); and Block, a transformer block of attention and an MLP, each with layer normalization before
it and a residual connection around it. Both are saved in the learner's project llm, as attention.py and block.py."""
import html
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import projects
from everything_ai.check import ClassBox, Mistake, Timeout, run_code, with_time_limit

# The block imports the attention of exercise 2; until that is saved, the course's version stands in (template 1.3).
projects.builds_on(("llm", "attention", "attn-transformer-block", "causal-self-attention"))

GPT2_D, GPT2_H, GPT2_BLOCKS = 768, 12, 12


# --- the picture: one transformer block, with the residual stream running straight through ---------------------------------

def block_picture(labels: dict) -> str:
    """The residual stream as a straight line from top to bottom (the hidden role: the vectors of the tokens), with two
    branches beside it, each a layer normalization and then attention or the MLP, whose result is added back."""
    W, SX, BX, BW, BH = 340, 46, 118, 206, 30
    cx = BX + BW / 2
    ink, line = "var(--ea-ink)", "var(--ea-ink-3)"
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:{ink}">{html.escape(labels["title"])}</text>']

    def box(y, words):
        p.append(f'<rect x="{BX}" y="{y}" width="{BW}" height="{BH}" rx="6" style="fill:var(--ea-inset);stroke:var(--ea-line)"/>'
                 f'<text x="{cx}" y="{y + BH / 2 + 4.5}" style="font-size:13px;fill:{ink};text-anchor:middle">{html.escape(words)}</text>')

    def down(x, y0, y1):
        p.append(f'<path d="M{x} {y0} V{y1 - 1} M{x - 3.5} {y1 - 5} L{x} {y1 - 1} L{x + 3.5} {y1 - 5}" '
                 f'style="fill:none;stroke:{line};stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"/>')

    def branch(y_from, y_box, words, y_add):
        p.append(f'<path d="M{SX} {y_from} H{cx} V{y_box - 1}" style="fill:none;stroke:{line};stroke-width:1.5"/>')
        down(cx, y_box - 12, y_box)
        box(y_box, labels["norm"])
        down(cx, y_box + BH, y_box + BH + 14)
        box(y_box + BH + 14, words)
        y_out = y_box + 2 * BH + 14
        p.append(f'<path d="M{cx} {y_out} V{y_add} H{SX + 10}" style="fill:none;stroke:{line};stroke-width:1.5"/>'
                 f'<path d="M{SX + 15} {y_add - 3.5} L{SX + 11} {y_add} L{SX + 15} {y_add + 3.5}" style="fill:none;stroke:{line};stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"/>'
                 f'<circle cx="{SX}" cy="{y_add}" r="9" style="fill:var(--ea-page);stroke:{ink};stroke-width:1.5"/>'
                 f'<path d="M{SX - 5} {y_add} H{SX + 5} M{SX} {y_add - 5} V{y_add + 5}" style="stroke:{ink};stroke-width:1.5"/>')

    top, bottom = 54, 356
    p.append(f'<text x="2" y="{top - 10}" style="font-size:13px;fill:var(--ea-hidden-text)">{html.escape(labels["input"])}</text>')
    p.append(f'<path d="M{SX} {top} V{bottom - 2}" style="stroke:var(--ea-hidden);stroke-width:3"/>'
             f'<path d="M{SX - 6} {bottom - 9} L{SX} {bottom - 1} L{SX + 6} {bottom - 9}" style="fill:none;stroke:var(--ea-hidden);stroke-width:3;stroke-linecap:round;stroke-linejoin:round"/>')
    branch(top + 16, top + 40, labels["attention"], top + 140)
    branch(top + 160, top + 184, labels["mlp"], top + 284)
    p.append(f'<text x="2" y="{bottom + 16}" style="font-size:13px;fill:var(--ea-hidden-text)">{html.escape(labels["output"])}</text>')
    p.append(f'<text transform="translate({SX - 16} {(top + bottom) / 2}) rotate(-90)" style="font-size:13px;fill:var(--ea-hidden-text);text-anchor:middle">{html.escape(labels["stream"])}</text>')
    H = bottom + 24
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


def grouped(n: int, lang: str) -> str:
    """A whole number with its thousands marked, as the language writes it: 7,085,568 or 7.085.568."""
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


# The lesson is done when these are correct. "no-attention" is the choice question in the lesson.
REQUIRED = ["no-attention", "causal-self-attention", "block"]

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


def _close(got, want) -> bool:
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    try:
        return bool(np.allclose(float(got), float(want), rtol=1e-4, atol=1e-4))
    except (TypeError, ValueError):
        return got == want


def _reordered(text: str, names: tuple, exercise: str, solution: str) -> bool:
    """The layers made in another order than `names`, and nothing else wrong: with the lines that make them put in
    that order, the class gives what the course's solution gives in every scenario."""
    lines = text.split("\n")
    rows = []
    for name in names:
        found = [i for i, line in enumerate(lines) if line.strip().startswith(f"self.{name} =")]
        if len(found) != 1:
            return False
        rows.append(found[0])
    if rows == sorted(rows):
        return False
    fixed = list(lines)
    for row, i in zip(sorted(rows), rows):
        fixed[row] = lines[i]
    return EXERCISES[exercise]._same_everywhere("\n".join(fixed), solution)


# --- exercise 2: CausalSelfAttention, for a mini-batch -----------------------------------------------------------------------

_A_HEAD = ("import math\nimport torch\nfrom torch import nn\n\n\nclass CausalSelfAttention(nn.Module):\n"
           "    def __init__(self, d, h):\n        super().__init__()\n        self.h = h\n        self.d_head = int(d / h)\n"
           "        self.query = nn.Linear(d, d, bias=False)\n        self.key = nn.Linear(d, d, bias=False)\n"
           "        self.value = nn.Linear(d, d, bias=False)\n        self.out = nn.Linear(d, d)\n\n"
           "    def forward(self, x):\n        n = x.shape[1]\n        future = torch.tril(torch.ones(n, n)) == 0\n"
           "        q, k, v = self.query(x), self.key(x), self.value(x)\n        outputs = []\n        for i in range(self.h):\n"
           "            a, b = i * self.d_head, (i + 1) * self.d_head\n")
_A_START = (_A_HEAD + "            # head i, for every text at once: the scaled scores, the mask, the softmax, times the values\n"
            "            ...\n        # the outputs of the heads side by side, then the output projection\n        ...\n")
_A_SCORES = "            S = q[:, :, a:b] @ k[:, :, a:b].transpose(1, 2) / math.sqrt(self.d_head)\n"
_A_MASK = "            S = S.masked_fill(future, float(\"-inf\"))\n"
_A_WEIGHTS = "            outputs.append(torch.softmax(S, dim=2) @ v[:, :, a:b])\n"
_A_OUT = "        return self.out(torch.cat(outputs, dim=2))\n"


def _a(scores: str = _A_SCORES, mask: str = _A_MASK, weights: str = _A_WEIGHTS, out: str = _A_OUT) -> str:
    return _A_HEAD + scores + mask + weights + out


_A_SCENARIOS = [
    # 2 texts of 3 tokens, 6 numbers each, in 2 heads of 3: the shape, and the new vector of the second token of text 2
    "import torch\n\ntorch.manual_seed(0)\nattention = CausalSelfAttention(6, 2)\nx = 4 * torch.rand(2, 3, 6)\ny = attention(x)\n[list(y.shape), y[1, 1].tolist()]",
    # 3 heads of 2, one text of 4 tokens, and its second token
    "import torch\n\ntorch.manual_seed(1)\nattention = CausalSelfAttention(6, 3)\nx = 4 * torch.rand(1, 4, 6)\ny = attention(x)\ny[0, 1].tolist()",
    # 3 texts of 5 tokens, 4 numbers in 2 heads: the last token of the first text
    "import torch\n\ntorch.manual_seed(2)\nattention = CausalSelfAttention(4, 2)\nx = 4 * torch.rand(3, 5, 4)\ny = attention(x)\ny[0, 4].tolist()",
]

# --- exercise 3: Block --------------------------------------------------------------------------------------------------------

_B_HEAD = ("import torch\nfrom torch import nn\nfrom llm.attention import CausalSelfAttention\n\n\nclass Block(nn.Module):\n"
           "    def __init__(self, d, h):\n        super().__init__()\n")
_B_INIT = ("        self.norm1 = nn.LayerNorm(d)\n        self.attention = CausalSelfAttention(d, h)\n"
           "        self.norm2 = nn.LayerNorm(d)\n        self.lin1 = nn.Linear(d, 4 * d)\n        self.lin2 = nn.Linear(4 * d, d)\n")
_B_FORWARD = ("        x = x + self.attention(self.norm1(x))\n        x = x + self.lin2(torch.relu(self.lin1(self.norm2(x))))\n"
              "        return x\n")
_B_START = (_B_HEAD + "        # in the order the data passes them: norm1, attention, norm2,\n"
            "        # then the MLP: lin1 from d to 4 * d numbers, and lin2 back to d\n        ...\n\n"
            "    def forward(self, x):\n        # x plus the attention of the normalized x; then x plus the MLP of the normalized x\n        ...\n")
_MLP = "self.lin2(torch.relu(self.lin1(self.norm2(x))))"


def _b(init: str = _B_INIT, forward: str = _B_FORWARD) -> str:
    return _B_HEAD + init + "\n    def forward(self, x):\n" + forward


_B_SCENARIOS = [
    # 2 texts of 3 tokens with 8 numbers each, in 2 heads: the shape, and the first 4 numbers of the last token of text 2
    "import torch\n\ntorch.manual_seed(0)\nblock = Block(8, 2)\nx = torch.randn(2, 3, 8)\ny = block(x)\n[list(y.shape), y[1, 2, :4].tolist()]",
    # two blocks in a row, on numbers far from mean 0 and variance 1
    "import torch\n\ntorch.manual_seed(1)\nblock = Block(6, 3)\nx = 3 * torch.randn(1, 4, 6) + 2\ny = block(block(x))\ny[0, 3, :4].tolist()",
    # the shapes of all parameters, in the order they were made
    "import torch\n\ntorch.manual_seed(2)\nblock = Block(4, 1)\n[list(p.shape) for p in block.parameters()]",
]


def _a_order(got, text) -> bool:
    return _reordered(text, ("query", "key", "value", "out"), "causal-self-attention", _a())


# the parameters of Block(4, 1) with one LayerNorm instead of two, in any order
_ONE_NORM = sorted([[4], [4], [4, 4], [4, 4], [4, 4], [4, 4], [4], [16, 4], [16], [4, 16], [4]])


def _one_norm(got, text) -> bool:
    return isinstance(got, list) and all(isinstance(g, list) for g in got) and sorted(got) == _ONE_NORM


def _b_order(got, text) -> bool:
    return _reordered(text, ("norm1", "attention", "norm2", "lin1", "lin2"), "block", _b())


EXERCISES = {
    "causal-self-attention": ClassBox(
        name="causal-self-attention",
        start=_A_START,
        solution=_a(),
        scenarios=_A_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("llm", "attention"),
        first=[_DOTS],
        mistakes=[
            Mistake("dot-T", syntax=r"^[^#\n]*(?:\[\s*:\s*,\s*:\s*,[^\]\n]*\]|\b[qkv])\.T\b",
                    en="`{line}` uses `.T`, which turns a tensor of 3 axes around completely: (B, n, d_head) becomes (d_head, n, B). In a mini-batch, swap only the last two axes, so that every text gets its own transpose: `.transpose(1, 2)`.",
                    nl="`{line}` gebruikt `.T`, dat een tensor met 3 axes helemaal omdraait: (B, n, d_head) wordt (d_head, n, B). Wissel in een mini-batch alleen de laatste twee axes, zodat elke tekst zijn eigen getransponeerde krijgt: `.transpose(1, 2)`."),
            Mistake("tokens", syntax=r"^[^#\n]*\b[qkv]\[\s*:\s*,\s*a\s*:\s*b\s*\]",
                    en="`{line}` cuts axis 1, the tokens, as `[:, a:b]` did for one text. In a mini-batch the columns are axis 2: `q[:, :, a:b]`.",
                    nl="`{line}` knipt in axis 1, de tokens, zoals `[:, a:b]` deed bij één tekst. In een mini-batch zijn de kolommen axis 2: `q[:, :, a:b]`."),
            Mistake("cat-axis", syntax=r"^[^#\n]*torch\.cat\(\s*outputs\s*,\s*(?:dim\s*=\s*)?[01]\s*\)",
                    en="`{line}` puts the outputs of the heads together along another axis than the columns, so `self.out` does not get d numbers per token. In a mini-batch of shape (B, n, d), the columns are axis 2: `dim=2`.",
                    nl="`{line}` zet de outputs van de heads aan elkaar langs een andere axis dan de kolommen, dus `self.out` krijgt geen d getallen per token. In een mini-batch met shape (B, n, d) zijn de kolommen axis 2: `dim=2`."),
            Mistake("softmax-axis", wrong=_a(weights=_A_WEIGHTS.replace("dim=2", "dim=1")),
                    en=_SCENARIO + ": your softmax goes along axis 1, down the queries of a text, so the weights of one token do not add up to 1. In a mini-batch, the scores of one token are a row along axis 2: `torch.softmax(S, dim=2)`.",
                    nl=_SCENARIO_NL + ": je softmax gaat langs axis 1, langs de queries van een tekst, dus de weights van één token tellen niet op tot 1. In een mini-batch zijn de scores van één token een rij langs axis 2: `torch.softmax(S, dim=2)`."),
            Mistake("no-mask", wrong=_a(mask=""),
                    en=_SCENARIO + ": every token also takes from the tokens after it. Set the scores of the future to minus infinity before the softmax: `S = S.masked_fill(future, float(\"-inf\"))`.",
                    nl=_SCENARIO_NL + ": elk token neemt ook van de tokens erna. Zet de scores van de toekomst op min oneindig vóór de softmax: `S = S.masked_fill(future, float(\"-inf\"))`."),
            Mistake("zero", wrong=_a(mask=_A_MASK.replace('float("-inf")', "0.0")),
                    en=_SCENARIO + ": the scores of the future are 0, and the softmax still gives them a weight, so every token also takes from the tokens after it. Use minus infinity: `float(\"-inf\")`.",
                    nl=_SCENARIO_NL + ": de scores van de toekomst zijn 0, en de softmax geeft ze nog steeds een weight, dus elk token neemt ook van de tokens erna. Gebruik min oneindig: `float(\"-inf\")`."),
            Mistake("no-scale", wrong=_a(scores=_A_SCORES.replace(" / math.sqrt(self.d_head)", "")),
                    en=_SCENARIO + ": your scores are not scaled. Divide them by the square root of the width of a head: `/ math.sqrt(self.d_head)`.",
                    nl=_SCENARIO_NL + ": je scores zijn niet geschaald. Deel ze door de wortel uit de breedte van een head: `/ math.sqrt(self.d_head)`."),
            Mistake("no-out", wrong=_a(out="        return torch.cat(outputs, dim=2)\n"),
                    en=_SCENARIO + ": the outputs of the heads come out side by side, without the output projection. Send them through `self.out`: `self.out(torch.cat(outputs, dim=2))`.",
                    nl=_SCENARIO_NL + ": de outputs van de heads komen naast elkaar naar buiten, zonder de output-projectie. Stuur ze door `self.out`: `self.out(torch.cat(outputs, dim=2))`."),
            Mistake("order", matches=_a_order,
                    en=_SCENARIO + ": the right shape, other numbers. Keep the four layers in the order query, key, value, out, as they were given, so that they get the same random starting values as in the course's solution; the rest of your class is right.",
                    nl=_SCENARIO_NL + ": de goede shape, andere getallen. Houd de vier layers in de volgorde query, key, value, out, zoals ze gegeven waren, zodat ze dezelfde willekeurige startwaarden krijgen als in de oplossing van de course; de rest van je class klopt."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: in the loop, the causal attention of head `i` on its own columns, for every text at once, added to `outputs`; after the loop, the outputs side by side through the output projection.",
                      "nl": "Vervang de twee `...`: in de loop de causale attention van head `i` op zijn eigen kolommen, voor elke tekst tegelijk, toegevoegd aan `outputs`; na de loop de outputs naast elkaar door de output-projectie."},
            "wrong": {"en": _SCENARIO + ". Per head: the scores of its columns with `transpose(1, 2)`, scaled, masked, the softmax along axis 2, times its values; then the outputs side by side along axis 2, through `self.out`: steps 2 and 3 of the recipe?",
                      "nl": _SCENARIO_NL + ". Per head: de scores van zijn kolommen met `transpose(1, 2)`, geschaald, gemaskeerd, de softmax langs axis 2, maal zijn values; dan de outputs naast elkaar langs axis 2, door `self.out`: stap 2 en 3 van het recept?"},
            "almost": {"en": "Right for the first example. But " + _SCENARIO[0].lower() + _SCENARIO[1:] + ". Does your class use `self.h`, `self.d_head` and `n`, and no fixed numbers?",
                       "nl": "Goed voor het eerste voorbeeld. Maar " + _SCENARIO_NL[0].lower() + _SCENARIO_NL[1:] + ". Gebruikt je class `self.h`, `self.d_head` en `n`, en geen vaste getallen?"},
            "correct": {"en": "Your attention works on a whole mini-batch at once: every text compares only its own tokens, and every token only those before it. Your `CausalSelfAttention` is now in your project, as `work/projects/llm/attention.py`.",
                        "nl": "Je attention werkt op een hele mini-batch tegelijk: elke tekst vergelijkt alleen zijn eigen tokens, en elk token alleen die ervoor. Je `CausalSelfAttention` staat nu in je project, als `work/projects/llm/attention.py`."},
            "kept": {"en": "Your `CausalSelfAttention` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `CausalSelfAttention` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "block": ClassBox(
        name="block",
        start=_B_START,
        solution=_b(),
        scenarios=_B_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("llm", "block"),
        first=[_DOTS],
        mistakes=[
            Mistake("sizes", error=lambda e: isinstance(e, RuntimeError) and ("cannot be multiplied" in str(e) or "normalized_shape" in str(e)),
                    en="The sizes do not fit: one of your layers gets another number of numbers per token than it was made for. Every layer gets d numbers per token, except `lin2`, which gets the 4 * d of `lin1`: `nn.Linear(d, 4 * d)` and `nn.Linear(4 * d, d)`. And `CausalSelfAttention(d, h)` takes d first.",
                    nl="De maten passen niet: een van je layers krijgt een ander aantal getallen per token dan waarvoor hij gemaakt is. Elke layer krijgt d getallen per token, behalve `lin2`, die de 4 * d van `lin1` krijgt: `nn.Linear(d, 4 * d)` en `nn.Linear(4 * d, d)`. En `CausalSelfAttention(d, h)` krijgt d eerst."),
            Mistake("no-residual", wrong=_b(forward=f"        x = self.attention(self.norm1(x))\n        x = {_MLP}\n        return x\n"),
                    en=_SCENARIO + ": the attention and the MLP replace x instead of adding to it, so there is no residual stream: every block starts over. Add their results to x: `x = x + ...`.",
                    nl=_SCENARIO_NL + ": de attention en het MLP vervangen x in plaats van erbij op te tellen, dus er is geen residual stream: elk block begint opnieuw. Tel hun uitkomsten op bij x: `x = x + ...`."),
            Mistake("post-norm", wrong=_b(forward=("        x = self.norm1(x + self.attention(x))\n"
                                                   "        x = self.norm2(x + self.lin2(torch.relu(self.lin1(x))))\n        return x\n")),
                    en=_SCENARIO + ": you normalize after adding, as the transformer of 2017 did: post-norm. GPT-2 normalizes a copy before each part, and leaves the residual stream itself alone: `x = x + self.attention(self.norm1(x))`.",
                    nl=_SCENARIO_NL + ": je normaliseert na het optellen, zoals de transformer van 2017 deed: post-norm. GPT-2 normaliseert een kopie vóór elk deel, en laat de residual stream zelf met rust: `x = x + self.attention(self.norm1(x))`."),
            Mistake("normalized-stream", wrong=_b(forward=("        x = self.norm1(x)\n        x = x + self.attention(x)\n"
                                                           "        x = self.norm2(x)\n        x = x + self.lin2(torch.relu(self.lin1(x)))\n        return x\n")),
                    en=_SCENARIO + ": you normalize x itself, so the residual stream is normalized twice in every block, and what comes in no longer goes straight through. Normalize a copy, inside the call: `x = x + self.attention(self.norm1(x))`.",
                    nl=_SCENARIO_NL + ": je normaliseert x zelf, dus de residual stream wordt in elk block twee keer genormaliseerd, en wat binnenkomt gaat er niet meer rechtdoor doorheen. Normaliseer een kopie, binnen de aanroep: `x = x + self.attention(self.norm1(x))`."),
            Mistake("parallel", wrong=_b(forward=f"        return x + self.attention(self.norm1(x)) + {_MLP}\n"),
                    en=_SCENARIO + ": the attention and the MLP both read the same x, side by side. In the block of GPT-2, the MLP comes after the attention: it reads x after the attention has added to it. Two lines, one after the other: first `x = x + ...` with the attention, then with the MLP.",
                    nl=_SCENARIO_NL + ": de attention en het MLP lezen allebei dezelfde x, naast elkaar. In het block van GPT-2 komt het MLP na de attention: het leest x nadat de attention erbij heeft opgeteld. Twee regels na elkaar: eerst `x = x + ...` met de attention, dan met het MLP."),
            Mistake("no-relu", wrong=_b(forward=_B_FORWARD.replace("torch.relu(self.lin1(self.norm2(x)))", "self.lin1(self.norm2(x))")),
                    en=_SCENARIO + ": your MLP has no ReLU, so `lin1` and `lin2` together are one linear function. `self.lin2(torch.relu(self.lin1(...)))`.",
                    nl=_SCENARIO_NL + ": je MLP heeft geen ReLU, dus `lin1` en `lin2` samen zijn één lineaire functie. `self.lin2(torch.relu(self.lin1(...)))`."),
            Mistake("narrow", wrong=_b(init=_B_INIT.replace("nn.Linear(d, 4 * d)", "nn.Linear(d, d)").replace("nn.Linear(4 * d, d)", "nn.Linear(d, d)")),
                    en=_SCENARIO + ": your MLP keeps d numbers per token. In GPT-2 it widens every vector to 4 * d numbers and then back: `nn.Linear(d, 4 * d)` and `nn.Linear(4 * d, d)`.",
                    nl=_SCENARIO_NL + ": je MLP houdt d getallen per token. In GPT-2 maakt het elke vector 4 * d getallen breed en dan weer smal: `nn.Linear(d, 4 * d)` en `nn.Linear(4 * d, d)`."),
            Mistake("one-norm", matches=_one_norm,
                    en=_SCENARIO + ": two shapes of 4 numbers are missing, the parameters of a layer normalization. Your block uses one LayerNorm for both parts, so the attention and the MLP share its parameters. Give each part one of its own: `self.norm1` and `self.norm2`, each `nn.LayerNorm(d)`.",
                    nl=_SCENARIO_NL + ": er ontbreken twee shapes van 4 getallen, de parameters van een layer normalization. Je block gebruikt één LayerNorm voor beide delen, dus de attention en het MLP delen zijn parameters. Geef elk deel een eigen: `self.norm1` en `self.norm2`, elk `nn.LayerNorm(d)`."),
            Mistake("order", matches=_b_order,
                    en=_SCENARIO + ": the right shape, other numbers. Make the layers in the order the data passes them, `norm1`, `attention`, `norm2`, `lin1`, `lin2`, so that they get the same random starting values as in the course's solution; the rest of your class is right.",
                    nl=_SCENARIO_NL + ": de goede shape, andere getallen. Maak de layers in de volgorde waarin de data erdoor gaat, `norm1`, `attention`, `norm2`, `lin1`, `lin2`, zodat ze dezelfde willekeurige startwaarden krijgen als in de oplossing van de course; de rest van je class klopt."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the five layers in `__init__`, and in `forward` the two steps, each added to `x`.",
                      "nl": "Vervang de twee `...`: de vijf layers in `__init__`, en in `forward` de twee stappen, elk opgeteld bij `x`."},
            "wrong": {"en": _SCENARIO + ". The five layers, then step 4 of the recipe: x plus the attention of `self.norm1(x)`, then x plus the MLP of `self.norm2(x)`, with a ReLU between `lin1` and `lin2`?",
                      "nl": _SCENARIO_NL + ". De vijf layers, en dan stap 4 van het recept: x plus de attention van `self.norm1(x)`, dan x plus het MLP van `self.norm2(x)`, met een ReLU tussen `lin1` en `lin2`?"},
            "almost": {"en": "Right for one block of 8 numbers. But " + _SCENARIO[0].lower() + _SCENARIO[1:] + ". Does your class use `d` and `h`, and no fixed numbers?",
                       "nl": "Goed voor één block van 8 getallen. Maar " + _SCENARIO_NL[0].lower() + _SCENARIO_NL[1:] + ". Gebruikt je class `d` en `h`, en geen vaste getallen?"},
            "correct": {"en": "Your block gives back the shape it gets, (B, n, d): attention and an MLP, each with layer normalization before it, added to the residual stream. Your `Block` is now in your project, as `work/projects/llm/block.py`.",
                        "nl": "Je block geeft de shape terug die het krijgt, (B, n, d): attention en een MLP, elk met layer normalization ervoor, opgeteld bij de residual stream. Je `Block` staat nu in je project, als `work/projects/llm/block.py`."},
            "kept": {"en": "Your `Block` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `Block` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}


# --- the demo: a stack of the learner's blocks ---------------------------------------------------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "llm" / f"{module}.py").exists()


def demo():
    """For the demo: None when attention.py or block.py is not in the project yet; "block" when Block stops with an
    error; "slow" when it takes too long; else a dict: the shape out of a stack of 4 blocks, how much the outputs of
    the first 7 tokens of text 1, of its last token, and of text 2 change when the last token of text 1 changes, the
    parameters of one block of GPT-2's size, and the modules the course's versions stood in for."""
    if not (_saved("attention") and _saved("block")):
        return None
    run = run_code("from llm.block import Block")
    if run.error is not None:
        return "block"
    Block = run.names["Block"]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]

    def stack(x, blocks):
        with torch.no_grad():
            for block in blocks:
                x = block(x)
        return x

    def work():
        torch.manual_seed(0)
        blocks = [Block(32, 4) for _ in range(4)]
        x = torch.randn(2, 8, 32)
        y = stack(x, blocks)
        changed = x.clone()
        changed[0, 7] = torch.randn(32)
        z = stack(changed, blocks)
        big = Block(GPT2_D, GPT2_H)
        return {"shape": list(y.shape), "before": float((z[0, :7] - y[0, :7]).abs().max()),
                "last": float((z[0, 7] - y[0, 7]).abs().max()), "other": float((z[1] - y[1]).abs().max()),
                "params": int(sum(p.numel() for p in big.parameters()))}

    try:
        out = with_time_limit(work, 30)
    except Timeout:
        return "slow"
    except Exception:
        return "block"
    out["stand_ins"] = stand_ins
    return out


# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_A_LOOP = _A_SCORES + _A_MASK + _A_WEIGHTS
CASES = [
    ("causal-self-attention", _A_START, "empty", None),
    ("causal-self-attention", _a(), "correct", None),
    ("causal-self-attention", _a(scores="            S = q[:, :, a:b] @ k[:, :, a:b].mT / self.d_head ** 0.5\n",
                                 weights="            A = torch.softmax(S, dim=-1)\n            outputs.append(A @ v[:, :, a:b])\n",
                                 out="        y = torch.cat(outputs, dim=-1)\n        return self.out(y)\n"), "correct", None),
    ("causal-self-attention", _a(scores=_A_SCORES.replace(".transpose(1, 2)", ".T")), "error", "dot-T"),
    ("causal-self-attention", _a(scores="            S = q[:, a:b] @ k[:, a:b].transpose(1, 2) / math.sqrt(self.d_head)\n",
                                 weights="            outputs.append(torch.softmax(S, dim=2) @ v[:, a:b])\n"), "error", "tokens"),
    ("causal-self-attention", _a(out="        return self.out(torch.cat(outputs, dim=1))\n"), "error", "cat-axis"),
    ("causal-self-attention", _a(weights=_A_WEIGHTS.replace("dim=2", "dim=1")), "wrong", "softmax-axis"),
    ("causal-self-attention", _a(mask=""), "wrong", "no-mask"),
    ("causal-self-attention", _a(scores=_A_SCORES.replace(" / math.sqrt(self.d_head)", "")), "wrong", "no-scale"),
    ("causal-self-attention", _a(mask=_A_MASK.replace('float("-inf")', "0.0")), "wrong", "zero"),
    ("causal-self-attention", _a(mask="", weights="            outputs.append(torch.softmax(S, dim=2).masked_fill(future, 0.0) @ v[:, :, a:b])\n"), "wrong", None),
    ("causal-self-attention", _a(out="        return self.out(torch.cat(outputs, 2))\n"), "correct", None),
    ("causal-self-attention", _a(out="        return self.out(torch.cat(outputs, 1))\n"), "error", "cat-axis"),
    ("causal-self-attention", _a(out="        return torch.cat(outputs, dim=2)\n"), "wrong", "no-out"),
    ("causal-self-attention", _a().replace("        self.query = nn.Linear(d, d, bias=False)\n        self.key = nn.Linear(d, d, bias=False)\n",
                                           "        self.key = nn.Linear(d, d, bias=False)\n        self.query = nn.Linear(d, d, bias=False)\n"), "wrong", "order"),
    ("causal-self-attention", _a(scores=_A_SCORES.replace("math.sqrt(self.d_head)", "math.sqrt(3)")), "almost", None),
    ("block", _B_START, "empty", None),
    ("block", _b(), "correct", None),
    ("block", _b(forward=("        h = self.norm1(x)\n        x = x + self.attention(h)\n        h = torch.relu(self.lin1(self.norm2(x)))\n"
                          "        return x + self.lin2(h)\n")), "correct", None),
    ("block", _b(init=_B_INIT.replace("nn.Linear(4 * d, d)", "nn.Linear(d, d)")), "error", "sizes"),
    ("block", _b(forward=f"        x = self.attention(self.norm1(x))\n        x = {_MLP}\n        return x\n"), "wrong", "no-residual"),
    ("block", _b(forward=("        x = self.norm1(x + self.attention(x))\n"
                          "        x = self.norm2(x + self.lin2(torch.relu(self.lin1(x))))\n        return x\n")), "wrong", "post-norm"),
    ("block", _b(forward=("        x = self.norm1(x)\n        x = x + self.attention(x)\n"
                          "        x = self.norm2(x)\n        x = x + self.lin2(torch.relu(self.lin1(x)))\n        return x\n")), "wrong", "normalized-stream"),
    ("block", _b(forward=f"        return x + self.attention(self.norm1(x)) + {_MLP}\n"), "wrong", "parallel"),
    ("block", _b(init=_B_INIT.replace("self.norm2 = nn.LayerNorm(d)", "self.norm2 = self.norm1")), "wrong", "one-norm"),
    ("block", _b(init=_B_INIT.replace("nn.LayerNorm(d)", "nn.LayerNorm(8)")), "error", "sizes"),
    ("block", _b(forward=_B_FORWARD.replace("torch.relu(self.lin1(self.norm2(x)))", "self.lin1(self.norm2(x))")), "wrong", "no-relu"),
    ("block", _b(init=_B_INIT.replace("nn.Linear(d, 4 * d)", "nn.Linear(d, d)").replace("nn.Linear(4 * d, d)", "nn.Linear(d, d)")), "wrong", "narrow"),
    ("block", _b(init=("        self.norm1 = nn.LayerNorm(d)\n        self.norm2 = nn.LayerNorm(d)\n        self.lin1 = nn.Linear(d, 4 * d)\n"
                       "        self.lin2 = nn.Linear(4 * d, d)\n        self.attention = CausalSelfAttention(d, h)\n")), "wrong", "order"),
    ("block", _b(init=("        self.norm1 = nn.LayerNorm(d)\n        self.norm2 = nn.LayerNorm(d)\n        self.lin1 = nn.Linear(d, 4 * d)\n"
                       "        self.lin2 = nn.Linear(4 * d, d)\n        self.attention = CausalSelfAttention(d, h)\n"),
                 forward=_B_FORWARD.replace("torch.relu(self.lin1(self.norm2(x)))", "self.lin1(self.norm2(x))")), "wrong", None),
    ("block", _b(init=_B_INIT.replace("nn.Linear(d, 4 * d)", "nn.Linear(d, 32)").replace("nn.Linear(4 * d, d)", "nn.Linear(32, d)")), "almost", None),
]
