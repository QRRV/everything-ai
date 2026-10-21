# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson llm-modern-gpt: the parts of the five lessons before it in the learner's project
(work/projects/llm/modern_parts.py), with a place to start from for RoPE; ModernAttention, grouped-query attention with
RoPE on every head and a KV cache (modern_attention.py); ModernBlock and ModernGPT, with RMSNorm and SwiGLU and without
a table of places (modern_gpt.py); and generate, which writes greedily with a KV cache in every block
(modern_generate.py). The demo trains the learner's ModernGPT as gpt-train-bpe trains their GPT, on the same tokens of
the same stories, and holds it against the course's runs of every change on its own; the second demo times the cache."""
import html
import importlib.util
import json
import math
import re
import time

import numpy as np
import torch

from everything_ai import paths, projects
from everything_ai.check import ClassBox, Mistake, run_code


def _train_bpe():
    """The checks of gpt-train-bpe, whose run this lesson repeats with a ModernGPT: its plans, tokens and helpers.
    Loading them registers that lesson's project modules; this lesson's own list follows below."""
    spec = importlib.util.spec_from_file_location("everything_ai_checks_gpt_train_bpe_for_modern",
                                                  paths.LESSONS / "gpt-train-bpe" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _train_bpe()
number, grouped = B.number, B.grouped


def block_picture(labels: dict) -> str:
    """The modern block, as the block of attn-transformer-block: the residual stream as a straight line (the hidden
    role), with two branches, each a normalization and then attention or the MLP, added back. Under every part, in
    grey, what it replaces in the learner's GPT."""
    W, SX, BX, BW, BH = 340, 46, 112, 220, 40
    cx = BX + BW / 2
    ink, line = "var(--ea-ink)", "var(--ea-ink-3)"
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:{ink}">{html.escape(labels["title"])}</text>']

    def box(y, words, was):
        p.append(f'<rect x="{BX}" y="{y}" width="{BW}" height="{BH}" rx="6" style="fill:var(--ea-inset);stroke:var(--ea-line)"/>'
                 f'<text x="{cx}" y="{y + 17}" style="font-size:13px;fill:{ink};text-anchor:middle">{html.escape(words)}</text>'
                 f'<text x="{cx}" y="{y + 32}" style="font-size:11.5px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(was)}</text>')

    def down(x, y0, y1):
        p.append(f'<path d="M{x} {y0} V{y1 - 1} M{x - 3.5} {y1 - 5} L{x} {y1 - 1} L{x + 3.5} {y1 - 5}" '
                 f'style="fill:none;stroke:{line};stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"/>')

    def branch(y_from, y_box, words, was, y_add):
        p.append(f'<path d="M{SX} {y_from} H{cx} V{y_box - 1}" style="fill:none;stroke:{line};stroke-width:1.5"/>')
        down(cx, y_box - 12, y_box)
        box(y_box, labels["norm"], labels["norm_was"])
        down(cx, y_box + BH, y_box + BH + 14)
        box(y_box + BH + 14, words, was)
        y_out = y_box + 2 * BH + 14
        p.append(f'<path d="M{cx} {y_out} V{y_add} H{SX + 10}" style="fill:none;stroke:{line};stroke-width:1.5"/>'
                 f'<path d="M{SX + 15} {y_add - 3.5} L{SX + 11} {y_add} L{SX + 15} {y_add + 3.5}" style="fill:none;stroke:{line};stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"/>'
                 f'<circle cx="{SX}" cy="{y_add}" r="9" style="fill:var(--ea-page);stroke:{ink};stroke-width:1.5"/>'
                 f'<path d="M{SX - 5} {y_add} H{SX + 5} M{SX} {y_add - 5} V{y_add + 5}" style="stroke:{ink};stroke-width:1.5"/>')

    top, bottom = 70, 392
    p.append(f'<text x="2" y="{top - 26}" style="font-size:13px;fill:var(--ea-hidden-text)">{html.escape(labels["input"])}</text>'
             f'<text x="2" y="{top - 10}" style="font-size:11.5px;fill:var(--ea-ink-2)">{html.escape(labels["input_was"])}</text>')
    p.append(f'<path d="M{SX} {top} V{bottom - 2}" style="stroke:var(--ea-hidden);stroke-width:3"/>'
             f'<path d="M{SX - 6} {bottom - 9} L{SX} {bottom - 1} L{SX + 6} {bottom - 9}" style="fill:none;stroke:var(--ea-hidden);stroke-width:3;stroke-linecap:round;stroke-linejoin:round"/>')
    branch(top + 16, top + 40, labels["attention"], labels["attention_was"], top + 156)
    branch(top + 176, top + 200, labels["mlp"], labels["mlp_was"], top + 312)
    p.append(f'<text x="2" y="{bottom + 16}" style="font-size:13px;fill:var(--ea-hidden-text)">{html.escape(labels["output"])}</text>')
    p.append(f'<text transform="translate({SX - 16} {(top + bottom) / 2}) rotate(-90)" style="font-size:13px;fill:var(--ea-hidden-text);text-anchor:middle">{html.escape(labels["stream"])}</text>')
    H = bottom + 24
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


def changes_picture(rows: list, labels: dict) -> str:
    """The best validation loss per token of the course's runs, one row per GPT and a dot per seed (the loss role), on
    one scale: so you see a change next to the spread between seeds. `rows`: (label, [losses])."""
    W, LEFT, ROW, TOP = 340, 0, 40, 46
    every = [x for _, xs in rows for x in xs]
    lo, hi = math.floor(min(every) * 20 - 0.5) / 20, math.ceil(max(every) * 20 + 0.5) / 20
    x0, x1 = 12, W - 12

    def X(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    for k, (label, xs) in enumerate(rows):
        y = TOP + k * ROW
        p.append(f'<text x="{LEFT}" y="{y}" style="font-size:13px;fill:var(--ea-ink-2)">{html.escape(label)}</text>'
                 f'<path d="M{x0} {y + 16} H{x1}" style="stroke:var(--ea-line);stroke-width:1"/>')
        for v in xs:
            p.append(f'<circle cx="{X(v):.1f}" cy="{y + 16}" r="4.5" style="fill:var(--ea-loss);fill-opacity:0.75"/>')
    y = TOP + len(rows) * ROW
    v = lo
    while v <= hi + 1e-9:
        label = f"{v:.2f}".replace(".", "," if labels.get("lang") == "nl" else ".")
        p.append(f'<path d="M{X(v):.1f} {y - 6} V{y - 2}" style="stroke:var(--ea-ink-3);stroke-width:1"/>'
                 f'<text x="{X(v):.1f}" y="{y + 10}" style="font-size:11.5px;fill:var(--ea-ink-3);text-anchor:middle">{label}</text>')
        v = round(v + 0.05, 2)
    p.append(f'<text x="{W / 2}" y="{y + 28}" style="font-size:12.5px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(labels["x"])}</text>')
    H = y + 36
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["title"] + ". " + labels["alt"])}</title>{"".join(p)}</svg></figure>')


history_picture = B.history_picture

# The lesson is done when these are correct. "gqa-why" is the choice question in the lesson.
REQUIRED = ["parts", "modern-attention", "modern-gpt", "generate", "gqa-why"]

# a `...` of the start: alone on its line, or after `=`; not the `...` of `out[..., 0::2]` in rope
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*$|=\s*\.\.\.\s*$", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_SCENARIO = "With these lines after your {what}:\n\n```python\n{{scenario}}\n```\n\nyou get `{{got}}`, and it should be `{{want}}`"
_SCENARIO_NL = "Met deze regels na je {what}:\n\n```python\n{{scenario}}\n```\n\nkrijg je `{{got}}`, en dat moet `{{want}}` zijn"
_S_CLASS, _S_CLASS_NL = _SCENARIO.format(what="class"), _SCENARIO_NL.format(what="class")
_S_CODE, _S_CODE_NL = _SCENARIO.format(what="code"), _SCENARIO_NL.format(what="code")


def _close(got, want) -> bool:
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    try:
        return bool(np.allclose(float(got), float(want), rtol=1e-4, atol=1e-4))
    except (TypeError, ValueError):
        return got == want


def _reordered(text: str, names: tuple, exercise: str, solution: str) -> bool:
    """The layers made in another order than `names`, and nothing else wrong: with the lines that make them put in
    that order, the code gives what the course's solution gives in every scenario."""
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

# --- the course's solutions -----------------------------------------------------------------------------------------

_P_HEAD = "import math\nimport torch\nimport torch.nn.functional as F\nfrom torch import nn\n\n\n"
_P_RMS = ("class RMSNorm(nn.Module):\n    def __init__(self, d, eps=1e-6):\n        super().__init__()\n"
          "        self.weight = nn.Parameter(torch.ones(d))\n        self.eps = eps\n\n    def forward(self, x):\n"
          "        rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)\n        return x / rms * self.weight\n")
_P_SWIGLU = ("class SwiGLU(nn.Module):\n    def __init__(self, d, hidden):\n        super().__init__()\n"
             "        self.gate = nn.Linear(d, hidden, bias=False)\n        self.up = nn.Linear(d, hidden, bias=False)\n"
             "        self.down = nn.Linear(hidden, d, bias=False)\n\n    def forward(self, x):\n"
             "        return self.down(F.silu(self.gate(x)) * self.up(x))\n")
_P_ANGLES = "    angles = (start + torch.arange(n))[:, None] * theta[None, :]\n"
_P_ROPE = ("def rope(x, base=10000.0, start=0):\n    n, d = x.shape[-2], x.shape[-1]\n"
           "    theta = base ** (-torch.arange(0, d, 2) / d)\n" + _P_ANGLES +
           "    cos, sin = torch.cos(angles), torch.sin(angles)\n    out = torch.zeros_like(x)\n"
           "    out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin\n"
           "    out[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos\n    return out\n")
_P_GQA = ("def gqa(q, k, v, n_heads, n_kv_heads):\n    n, d_head = q.shape[1], int(q.shape[2] / n_heads)\n"
          "    future = torch.tril(torch.ones(n, n)) == 0\n    outputs = []\n    for i in range(n_heads):\n"
          "        a, b = i * d_head, (i + 1) * d_head\n        j = int(i / (n_heads / n_kv_heads))\n"
          "        c, e = j * d_head, (j + 1) * d_head\n"
          "        S = q[:, :, a:b] @ k[:, :, c:e].transpose(1, 2) / math.sqrt(d_head)\n"
          "        S = S.masked_fill(future, float(\"-inf\"))\n        outputs.append(torch.softmax(S, dim=2) @ v[:, :, c:e])\n"
          "    return torch.cat(outputs, dim=2)\n")


def _p(rms: str = _P_RMS, swiglu: str = _P_SWIGLU, rope: str = _P_ROPE, gqa: str = _P_GQA) -> str:
    return _P_HEAD + rms + "\n\n" + swiglu + "\n\n" + rope + "\n\n" + gqa


_A_HEAD = "import torch\nfrom torch import nn\nfrom llm.modern_parts import gqa, rope\n\n\n"
_A_INIT = ("    def __init__(self, d, h, kv_heads):\n        super().__init__()\n        self.h = h\n        self.kv_heads = kv_heads\n"
           "        self.d_head = int(d / h)\n        self.query = nn.Linear(d, d, bias=False)\n"
           "        self.key = nn.Linear(d, kv_heads * self.d_head, bias=False)\n"
           "        self.value = nn.Linear(d, kv_heads * self.d_head, bias=False)\n        self.out = nn.Linear(d, d)\n")
_A_TURN = ("    def turn(self, x, n_heads, start):\n        parts = []\n        for i in range(n_heads):\n"
           "            parts.append(rope(x[:, :, i * self.d_head:(i + 1) * self.d_head], start=start))\n"
           "        return torch.cat(parts, dim=2)\n")
_A_FORWARD = ("    def forward(self, x, cache=None):\n        start = 0\n        if cache is not None and \"k\" in cache:\n"
              "            start = cache[\"k\"].shape[1]\n"
              "        q = self.turn(self.query(x), self.h, start)\n        k = self.turn(self.key(x), self.kv_heads, start)\n"
              "        v = self.value(x)\n        if start > 0:\n            k = torch.cat([cache[\"k\"], k], dim=1)\n"
              "            v = torch.cat([cache[\"v\"], v], dim=1)\n        if cache is not None:\n"
              "            cache[\"k\"] = k\n            cache[\"v\"] = v\n"
              "        return self.out(gqa(q, k, v, self.h, self.kv_heads))\n")


def _a(init: str = _A_INIT, turn: str = _A_TURN, forward: str = _A_FORWARD) -> str:
    return _A_HEAD + "class ModernAttention(nn.Module):\n" + init + "\n" + turn + "\n" + forward


_G_HEAD = "from torch import nn\nfrom llm.modern_attention import ModernAttention\nfrom llm.modern_parts import RMSNorm, SwiGLU\n\n\n"
_G_BLOCK_INIT = ("    def __init__(self, d, h, kv_heads):\n        super().__init__()\n        self.norm1 = RMSNorm(d)\n"
                 "        self.attention = ModernAttention(d, h, kv_heads)\n        self.norm2 = RMSNorm(d)\n"
                 "        self.mlp = SwiGLU(d, int(8 * d / 3))\n")
_G_BLOCK_FORWARD = ("    def forward(self, x, cache=None):\n        x = x + self.attention(self.norm1(x), cache)\n"
                    "        x = x + self.mlp(self.norm2(x))\n        return x\n")
_G_INIT = ("    def __init__(self, vocab, d, h, kv_heads, n_layers):\n        super().__init__()\n"
           "        self.tok = nn.Embedding(vocab, d)\n"
           "        self.blocks = nn.ModuleList([ModernBlock(d, h, kv_heads) for _ in range(n_layers)])\n"
           "        self.norm = RMSNorm(d)\n        self.head = nn.Linear(d, vocab)\n")
_G_FORWARD = ("    def forward(self, ids, caches=None):\n        x = self.tok(ids)\n        for i in range(len(self.blocks)):\n"
              "            if caches is None:\n                x = self.blocks[i](x)\n            else:\n"
              "                x = self.blocks[i](x, caches[i])\n        return self.head(self.norm(x))\n")


def _g(block_init: str = _G_BLOCK_INIT, block_forward: str = _G_BLOCK_FORWARD, init: str = _G_INIT,
       forward: str = _G_FORWARD) -> str:
    return (_G_HEAD + "class ModernBlock(nn.Module):\n" + block_init + "\n" + block_forward + "\n\n"
            "class ModernGPT(nn.Module):\n" + init + "\n" + forward)


_X_HEAD = "import torch\n\n\n"
_X_CACHES = "    caches = []\n    for _ in model.blocks:\n        caches.append({})\n"
_X_LOOP = ("        for _ in range(n):\n            j = int(logits[0, -1].argmax())\n            out.append(j)\n"
           "            logits = model(torch.tensor([[j]]), caches)\n")


def _x(caches: str = _X_CACHES, loop: str = _X_LOOP) -> str:
    return (_X_HEAD + "def generate(model, ids, n):\n" + caches + "    out = list(ids)\n    model.eval()\n"
            "    with torch.no_grad():\n        logits = model(torch.tensor([ids]), caches)\n" + loop + "    return out\n")


# --- exercise 1: the parts, in your project ------------------------------------------------------------------------------

_P_START = (_P_HEAD + "# your RMSNorm(d, eps=1e-6) of the lesson RMSNorm\n...\n\n\n"
            "# your SwiGLU(d, hidden) of the lesson SwiGLU\n...\n\n\n"
            "# your rope(x, base=10000.0) of the lesson RoPE, with one more argument, start=0:\n"
            "# the place of the first row, so that the places run from start to start + n - 1\n...\n\n\n"
            "# your gqa(q, k, v, n_heads, n_kv_heads) of the lesson grouped-query attention\n...\n")
_P_SCENARIOS = [
    # RMSNorm: two rows of two numbers
    "norm = RMSNorm(2)\nnorm(torch.tensor([[3.0, 4.0], [1.0, 1.0]])).tolist()",
    # SwiGLU with weights of its own
    "mlp = SwiGLU(2, 3)\nwith torch.no_grad():\n    mlp.gate.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))\n"
    "    mlp.up.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, -1.0]]))\n"
    "    mlp.down.weight.copy_(torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]]))\nmlp(torch.tensor([[1.0, -1.0]])).tolist()",
    # rope as before: places 0 and 1
    "x = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]])\nrope(x).tolist()",
    # rope from place 3: the same two rows, at places 3 and 4
    "x = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]])\nrope(x, start=3).tolist()",
    # the last two rows of six, turned on their own from place 4
    "torch.manual_seed(0)\nx = torch.randn(2, 6, 8)\nrope(x[:, 4:], start=4)[1].tolist()",
    # gqa: 4 query heads, 2 key and value heads
    "torch.manual_seed(1)\nq, k, v = torch.randn(1, 3, 8), torch.randn(1, 3, 4), torch.randn(1, 3, 4)\ngqa(q, k, v, 4, 2)[0, 2].tolist()",
    # gqa with one query and five keys, as with a KV cache
    "torch.manual_seed(2)\nq = torch.randn(2, 1, 8)\nk, v = torch.randn(2, 5, 4), torch.randn(2, 5, 4)\ngqa(q, k, v, 4, 2)[1, 0].tolist()",
]


def _missing(name: str, lesson: str, title_en: str, title_nl: str) -> Mistake:
    return Mistake(f"no-{name.lower()}", error=lambda e: isinstance(e, NameError) and f"'{name}'" in str(e),
                   en=f"Your box has no `{name}` yet, and the box needs all four parts. Copy your `{name}` from [{title_en}](/course/open/{lesson}?lang=en), or write it again.",
                   nl=f"Je vak heeft nog geen `{name}`, en het vak heeft alle vier de onderdelen nodig. Kopieer je `{name}` uit [{title_nl}](/course/open/{lesson}?lang=nl), of schrijf hem opnieuw.")


# --- exercise 2: ModernAttention ------------------------------------------------------------------------------------------

_A_START = (_A_HEAD + "class ModernAttention(nn.Module):\n    def __init__(self, d, h, kv_heads):\n        super().__init__()\n"
            "        self.h = h\n        self.kv_heads = kv_heads\n        self.d_head = int(d / h)\n"
            "        # query, key, value and out, in that order: queries for h heads, keys and values for only kv_heads heads\n"
            "        ...\n\n    def turn(self, x, n_heads, start):\n"
            "        # rope on the d_head numbers of every head on its own, from place start; the heads side by side again\n"
            "        ...\n\n    def forward(self, x, cache=None):\n"
            "        # the place of the first token: 0, or the number of keys already in the cache\n        start = 0\n"
            "        if cache is not None and \"k\" in cache:\n            start = cache[\"k\"].shape[1]\n"
            "        # the queries and keys turned, the values not\n        ...\n"
            "        # with keys in the cache: those first; with a cache: keep all keys and values in it\n        ...\n"
            "        return self.out(gqa(q, k, v, self.h, self.kv_heads))\n")
_A_SCENARIOS = [
    # the shapes of the four weights: d = 8, 4 heads, 2 key and value heads
    "attention = ModernAttention(8, 4, 2)\n[list(attention.query.weight.shape), list(attention.key.weight.shape), "
    "list(attention.value.weight.shape), list(attention.out.weight.shape)]",
    # turn: 2 heads of 4 numbers, two tokens at places 5 and 6
    "attention = ModernAttention(8, 2, 1)\nattention.turn(torch.ones(1, 2, 8), 2, 5).tolist()",
    # 2 texts of 5 tokens: the output of the last token of text 2
    "torch.manual_seed(0)\nattention = ModernAttention(8, 4, 2)\nx = torch.randn(2, 5, 8)\nattention(x)[1, 4].tolist()",
    # 4 tokens with an empty cache, then 1: the keys and values the cache holds
    "torch.manual_seed(1)\nattention = ModernAttention(8, 4, 2)\nx = torch.randn(1, 5, 8)\ncache = {}\n"
    "attention(x[:, :4], cache)\nattention(x[:, 4:], cache)\n[list(cache[\"k\"].shape), list(cache[\"v\"].shape)]",
    # the last token with the cache, and the same token with all 5 at once, without a cache
    "torch.manual_seed(1)\nattention = ModernAttention(8, 4, 2)\nx = torch.randn(1, 5, 8)\ncache = {}\n"
    "attention(x[:, :4], cache)\nlast = attention(x[:, 4:], cache)\n[last[0, 0].tolist(), attention(x)[0, 4].tolist()]",
]
_A_CAT = ("        if start > 0:\n            k = torch.cat([cache[\"k\"], k], dim=1)\n"
          "            v = torch.cat([cache[\"v\"], v], dim=1)\n")


def _a_order(got, text) -> bool:
    return _reordered(text, ("query", "key", "value", "out"), "modern-attention", _a())


# --- exercise 3: ModernBlock and ModernGPT --------------------------------------------------------------------------------

_G_START = (_G_HEAD + "class ModernBlock(nn.Module):\n    def __init__(self, d, h, kv_heads):\n        super().__init__()\n"
            "        # norm1, attention, norm2 and mlp, as in your Block: RMSNorm, ModernAttention, RMSNorm,\n"
            "        # and SwiGLU with a hidden size of int(8 * d / 3)\n        ...\n\n"
            "    def forward(self, x, cache=None):\n"
            "        # pre-norm and the residual stream, as in your Block; the cache goes to the attention\n        ...\n\n\n"
            "class ModernGPT(nn.Module):\n    def __init__(self, vocab, d, h, kv_heads, n_layers):\n        super().__init__()\n"
            "        # tok, blocks, norm and head, as in your GPT, without a table of places: RoPE gives them\n        ...\n\n"
            "    def forward(self, ids, caches=None):\n"
            "        # every block in turn, with its own cache, caches[i], when there are caches; then norm and head\n        ...\n")
_G_SCENARIOS = [
    # a vocabulary of 20, d = 16, 4 heads, 2 key and value heads, 3 blocks: the shape of the logits, and the parameters
    "import torch\n\nmodel = ModernGPT(20, 16, 4, 2, 3)\nids = torch.tensor([[1, 2, 3]])\n"
    "[list(model(ids).shape), sum(p.numel() for p in model.parameters())]",
    # one block on its own: the output of its last token
    "import torch\n\ntorch.manual_seed(0)\nblock = ModernBlock(8, 4, 2)\nblock(torch.randn(1, 3, 8))[0, 2].tolist()",
    # 2 texts of 4 tokens: 4 logits of the last token of text 2
    "import torch\n\ntorch.manual_seed(1)\nmodel = ModernGPT(10, 8, 2, 1, 2)\n"
    "model(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]))[1, 3, :4].tolist()",
    # 4 tokens with empty caches, then 1; and the same token with all 5 at once, without caches
    "import torch\n\ntorch.manual_seed(2)\nmodel = ModernGPT(10, 8, 2, 1, 2)\nids = torch.tensor([[1, 2, 3, 4, 5]])\n"
    "caches = [{}, {}]\nmodel(ids[:, :4], caches)\nlast = model(ids[:, 4:], caches)\n"
    "[last[0, 0, :4].tolist(), model(ids)[0, 4, :4].tolist()]",
]
_G_PLAIN_FORWARD = ("    def forward(self, ids, caches=None):\n        x = self.tok(ids)\n        for b in self.blocks:\n"
                    "            x = b(x)\n        return self.head(self.norm(x))\n")


def _g_order(got, text) -> bool:
    return (_reordered(text, ("tok", "blocks", "norm", "head"), "modern-gpt", _g())
            or _reordered(text, ("norm1", "attention", "norm2", "mlp"), "modern-gpt", _g()))


# --- exercise 4: generate with a KV cache ---------------------------------------------------------------------------------

_X_START = (_X_HEAD + "def generate(model, ids, n):\n    # an empty cache, a dict, for every block of the model\n    ...\n"
            "    out = list(ids)\n    model.eval()\n    with torch.no_grad():\n"
            "        # the whole start once: it fills the caches\n        logits = ...\n        for _ in range(n):\n"
            "            # the likeliest next id after the last token, and then only that id through the model\n"
            "            ...\n    return out\n")
_X_SCENARIOS = [
    # a ModernGPT with a vocabulary of 20: 5 ids after the start 1, 2, 3
    "from llm.modern_gpt import ModernGPT\n\ntorch.manual_seed(0)\nmodel = ModernGPT(20, 16, 4, 2, 2)\ngenerate(model, [1, 2, 3], 5)",
    # 10 ids with the cache, and the same 10 the slow way: every step all ids through the model, without a cache
    "from llm.modern_gpt import ModernGPT\n\ntorch.manual_seed(1)\nmodel = ModernGPT(30, 16, 4, 1, 3)\nslow = [4, 5]\n"
    "for _ in range(10):\n    slow.append(int(model(torch.tensor([slow]))[0, -1].argmax()))\n[generate(model, [4, 5], 10), slow]",
    # no new ids
    "from llm.modern_gpt import ModernGPT\n\ntorch.manual_seed(0)\nmodel = ModernGPT(20, 16, 4, 2, 2)\ngenerate(model, [7, 8], 0)",
    # how many tokens the model gets: the first time, and the most after that
    "from llm.modern_gpt import ModernGPT\n\n\nclass Counted(ModernGPT):\n    def forward(self, ids, caches=None):\n"
    "        seen.append(ids.shape[1])\n        return super().forward(ids, caches)\n\n\nseen = []\ntorch.manual_seed(2)\n"
    "model = Counted(20, 16, 4, 2, 2)\ngenerate(model, [1, 2, 3], 4)\n[seen[0], max(seen[1:])]",
]


EXERCISES = {
    "parts": ClassBox(
        name="parts",
        start=_P_START,
        solution=_p(),
        scenarios=_P_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("llm", "modern_parts"),
        first=[_DOTS],
        mistakes=[
            _missing("RMSNorm", "llm-rmsnorm", "RMSNorm", "RMSNorm"),
            _missing("SwiGLU", "llm-swiglu", "Modern MLP blocks: GELU, SiLU and SwiGLU", "Moderne MLP's: GELU, SiLU en SwiGLU"),
            _missing("rope", "llm-rope", "Rotary position embeddings (RoPE)", "Rotary position embeddings (RoPE)"),
            _missing("gqa", "llm-gqa", "Grouped-query attention", "Grouped-query attention"),
            Mistake("no-start", error=lambda e: isinstance(e, TypeError) and "unexpected keyword argument 'start'" in str(e),
                    en="Your `rope` has no argument `start` yet. Give it one more, which is 0 when it is not given: `def rope(x, base=10000.0, start=0)`. Then let the places run from `start`, with brackets around them before `[:, None]`: `(start + torch.arange(n))`.",
                    nl="Je `rope` heeft nog geen argument `start`. Geef hem er één bij, die 0 is als hij niet wordt meegegeven: `def rope(x, base=10000.0, start=0)`. Laat de plekken dan bij `start` beginnen, met haakjes eromheen vóór `[:, None]`: `(start + torch.arange(n))`."),
            Mistake("start-required", error=lambda e: isinstance(e, TypeError) and "rope() missing 1 required positional argument: 'start'" in str(e),
                    en="Your `rope` needs a `start` in every call, and the check also calls it without one, `rope(x)`, as the lesson RoPE did. Give `start` a value for when it is not given: `def rope(x, base=10000.0, start=0)`.",
                    nl="Je `rope` heeft in elke aanroep een `start` nodig, en de check roept hem ook zonder aan, `rope(x)`, zoals de les RoPE deed. Geef `start` een waarde voor als hij niet wordt meegegeven: `def rope(x, base=10000.0, start=0)`."),
            Mistake("start-unused", wrong=_p(rope=_P_ROPE.replace(_P_ANGLES, "    angles = torch.arange(n)[:, None] * theta[None, :]\n")),
                    en=_S_CODE + ": your `rope` turns the rows as if they stood at places 0, 1, …, whatever `start` is. Let the places run from `start`: `(start + torch.arange(n))[:, None] * theta[None, :]`.",
                    nl=_S_CODE_NL + ": je `rope` draait de rijen alsof ze op plek 0, 1, … staan, wat `start` ook is. Laat de plekken bij `start` beginnen: `(start + torch.arange(n))[:, None] * theta[None, :]`."),
            Mistake("start-angle", wrong=_p(rope=_P_ROPE.replace(_P_ANGLES, "    angles = torch.arange(n)[:, None] * theta[None, :] + start\n")),
                    en=_S_CODE + ": your `rope` adds `start` to the angles, so every pair turns by `start` radians more. `start` is a place: add it to the places, before you multiply by `theta`: `(start + torch.arange(n))[:, None] * theta[None, :]`.",
                    nl=_S_CODE_NL + ": je `rope` telt `start` bij de hoeken op, dus elk paar draait `start` radialen verder. `start` is een plek: tel hem bij de plekken op, vóór je met `theta` vermenigvuldigt: `(start + torch.arange(n))[:, None] * theta[None, :]`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the four `...` with your four parts: copy them from the lessons where you wrote them, and give `rope` the argument `start`.",
                      "nl": "Vervang de vier `...` door je vier onderdelen: kopieer ze uit de lessen waarin je ze schreef, en geef `rope` het argument `start`."},
            "wrong": {"en": _S_CODE + ". Is your RMSNorm the one of [RMSNorm](/course/open/llm-rmsnorm?lang=en): each row divided by the root of the mean of its squares, then scaled?",
                      "nl": _S_CODE_NL + ". Is je RMSNorm die van [RMSNorm](/course/open/llm-rmsnorm?lang=nl): elke rij gedeeld door de wortel van het gemiddelde van zijn kwadraten, en dan geschaald?"},
            "almost": {"en": "Right for the first example. But " + _S_CODE[0].lower() + _S_CODE[1:] + ". Each of the four still does what it did in its own lesson, and `rope` lets its places run from `start`?",
                       "nl": "Goed voor het eerste voorbeeld. Maar " + _S_CODE_NL[0].lower() + _S_CODE_NL[1:] + ". Doet elk van de vier nog wat hij in zijn eigen les deed, en laat `rope` zijn plekken bij `start` beginnen?"},
            "correct": {"en": "Your four parts work as in their own lessons, and `rope(x, start=t)` turns the rows as if they stood at places t, t + 1, …: what the KV cache needs in exercise 2. They are in your project now, as `{saved}`.",
                        "nl": "Je vier onderdelen werken zoals in hun eigen lessen, en `rope(x, start=t)` draait de rijen alsof ze op plek t, t + 1, … staan: wat de KV cache in oefening 2 nodig heeft. Ze staan nu in je project, als `{saved}`."},
            "kept": {"en": "Your four parts are right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je vier onderdelen zijn goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "modern-attention": ClassBox(
        name="modern-attention",
        start=_A_START,
        solution=_a(),
        scenarios=_A_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("llm", "modern_attention"),
        first=[_DOTS],
        mistakes=[
            Mistake("cache-not-kept", error=lambda e: isinstance(e, KeyError) and e.args == ("k",),
                    en="After a call with a cache, the cache has no `\"k\"`: your `forward` does not keep the keys and values in it. With a cache, also the empty one of the first call, put all keys and values in it before the attention: `cache[\"k\"] = k` and `cache[\"v\"] = v`. An empty dict counts as false in an `if`, so test `if cache is not None:`.",
                    nl="Na een aanroep met een cache heeft de cache geen `\"k\"`: je `forward` bewaart de keys en values er niet in. Zet met een cache, ook de lege van de eerste aanroep, alle keys en values erin vóór de attention: `cache[\"k\"] = k` en `cache[\"v\"] = v`. Een lege dict telt in een `if` als onwaar, dus test `if cache is not None:`."),
            Mistake("kv-full", wrong=_a(init=_A_INIT.replace("nn.Linear(d, kv_heads * self.d_head", "nn.Linear(d, d")),
                    en=_S_CLASS + ": your keys and values have as many heads as your queries, so `gqa` uses only the first `kv_heads` of them, and the rest is weight that never helps. Make keys and values for `kv_heads` heads: `nn.Linear(d, kv_heads * self.d_head, bias=False)`.",
                    nl=_S_CLASS_NL + ": je keys en values hebben net zoveel heads als je queries, dus `gqa` gebruikt alleen de eerste `kv_heads` ervan, en de rest is weight die nooit helpt. Maak keys en values voor `kv_heads` heads: `nn.Linear(d, kv_heads * self.d_head, bias=False)`."),
            Mistake("out-no-bias", wrong=_a(init=_A_INIT.replace("self.out = nn.Linear(d, d)", "self.out = nn.Linear(d, d, bias=False)")),
                    en=_S_CLASS + ": your `out` has no bias terms. This lesson changes four parts and keeps the rest of your GPT as it was, so that each change can be measured: `out` stays as in your CausalSelfAttention, `nn.Linear(d, d)`.",
                    nl=_S_CLASS_NL + ": je `out` heeft geen bias terms. Deze les verandert vier onderdelen en houdt de rest van je GPT zoals hij was, zodat elke verandering te meten is: `out` blijft zoals in je CausalSelfAttention, `nn.Linear(d, d)`."),
            Mistake("rope-whole", wrong=_a(turn="    def turn(self, x, n_heads, start):\n        return rope(x, start=start)\n"),
                    en=_S_CLASS + ": your `turn` turns all heads at once, as one vector of `n_heads * d_head` numbers, so the heads get other angles than RoPE gives a head. Cut `x` into heads, `x[:, :, i * self.d_head:(i + 1) * self.d_head]`, turn each with `rope`, and put them side by side with `torch.cat(..., dim=2)`.",
                    nl=_S_CLASS_NL + ": je `turn` draait alle heads tegelijk, als één vector van `n_heads * d_head` getallen, dus de heads krijgen andere hoeken dan RoPE een head geeft. Knip `x` in heads, `x[:, :, i * self.d_head:(i + 1) * self.d_head]`, draai elk met `rope`, en zet ze naast elkaar met `torch.cat(..., dim=2)`."),
            Mistake("turn-values", wrong=_a(forward=_A_FORWARD.replace("        v = self.value(x)\n", "        v = self.turn(self.value(x), self.kv_heads, start)\n")),
                    en=_S_CLASS + ": you turn the values too. RoPE turns only the queries and keys: their dot product, the score, is what should depend on how far apart two tokens are. Leave the values as they are: `v = self.value(x)`.",
                    nl=_S_CLASS_NL + ": je draait de values ook. RoPE draait alleen de queries en keys: hun dot product, de score, moet afhangen van hoe ver twee tokens uit elkaar staan. Laat de values zoals ze zijn: `v = self.value(x)`."),
            Mistake("no-turn-k", wrong=_a(forward=_A_FORWARD.replace("k = self.turn(self.key(x), self.kv_heads, start)", "k = self.key(x)")),
                    en=_S_CLASS + ": you turn the queries but not the keys. A score depends on the distance between two tokens only when both are turned: turn the keys too, with `kv_heads` heads: `self.turn(self.key(x), self.kv_heads, start)`.",
                    nl=_S_CLASS_NL + ": je draait de queries maar niet de keys. Een score hangt alleen van de afstand tussen twee tokens af als ze allebei gedraaid zijn: draai de keys ook, met `kv_heads` heads: `self.turn(self.key(x), self.kv_heads, start)`."),
            Mistake("no-turn-q", wrong=_a(forward=_A_FORWARD.replace("q = self.turn(self.query(x), self.h, start)", "q = self.query(x)")),
                    en=_S_CLASS + ": you turn the keys but not the queries. A score depends on the distance between two tokens only when both are turned: turn the queries too, with `h` heads: `self.turn(self.query(x), self.h, start)`.",
                    nl=_S_CLASS_NL + ": je draait de keys maar niet de queries. Een score hangt alleen van de afstand tussen twee tokens af als ze allebei gedraaid zijn: draai de queries ook, met `h` heads: `self.turn(self.query(x), self.h, start)`."),
            Mistake("start-unused", wrong=_a(turn=_A_TURN.replace("start=start", "start=0")),
                    en=_S_CLASS + ": with a cache, your new token is turned as if it stood at place 0. It stands at place `start`, after the tokens in the cache: give `start` to `rope`, `rope(..., start=start)`.",
                    nl=_S_CLASS_NL + ": met een cache wordt je nieuwe token gedraaid alsof het op plek 0 staat. Het staat op plek `start`, na de tokens in de cache: geef `start` aan `rope`, `rope(..., start=start)`."),
            Mistake("start-as-base", matches=lambda got, text: re.search(r"\brope\((?:[^()]|\([^()]*\))*,\s*start\s*\)", text) is not None,
                    en=_S_CLASS + ": your `turn` gives `start` to `rope` as its second value, and the second value of `rope` is `base`: the heads then turn with `start` as their base, not as RoPE turns them. Name it: `rope(..., start=start)`.",
                    nl=_S_CLASS_NL + ": je `turn` geeft `start` aan `rope` als tweede waarde, en de tweede waarde van `rope` is `base`: de heads draaien dan met `start` als base, niet zoals RoPE ze draait. Noem hem bij naam: `rope(..., start=start)`."),
            Mistake("no-cat", wrong=_a(forward=_A_FORWARD.replace(_A_CAT, "")),
                    en=_S_CLASS + ": the cache keeps only the keys and values of the last call, so a new token sees only itself. Put the keys and values of the cache first, `torch.cat([cache[\"k\"], k], dim=1)`, and keep all of them.",
                    nl=_S_CLASS_NL + ": de cache bewaart alleen de keys en values van de laatste aanroep, dus een nieuw token ziet alleen zichzelf. Zet de keys en values uit de cache ervoor, `torch.cat([cache[\"k\"], k], dim=1)`, en bewaar ze allemaal."),
            Mistake("order", matches=_a_order,
                    en=_S_CLASS + ": the right shapes, other numbers. Make the layers in the order `query`, `key`, `value`, `out`, so that they get the same random starting values as in the course's solution; the rest of your class is right.",
                    nl=_S_CLASS_NL + ": de goede shapes, andere getallen. Maak de layers in de volgorde `query`, `key`, `value`, `out`, zodat ze dezelfde willekeurige startwaarden krijgen als in de oplossing van de course; de rest van je class klopt."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the four `...`: the four layers, `turn`, the turned queries and keys and the values, and what happens with a cache.",
                      "nl": "Vervang de vier `...`: de vier layers, `turn`, de gedraaide queries en keys en de values, en wat er met een cache gebeurt."},
            "wrong": {"en": _S_CLASS + ". Queries for `h` heads, keys and values for `kv_heads` heads of `self.d_head` numbers, all without bias, and `out` as in your CausalSelfAttention?",
                      "nl": _S_CLASS_NL + ". Queries voor `h` heads, keys en values voor `kv_heads` heads van `self.d_head` getallen, allemaal zonder bias, en `out` zoals in je CausalSelfAttention?"},
            "almost": {"en": "Right for the shapes. But " + _S_CLASS[0].lower() + _S_CLASS[1:] + ". Every head turned on its own from `start`, only queries and keys, and with a cache its keys and values first?",
                       "nl": "Goed voor de shapes. Maar " + _S_CLASS_NL[0].lower() + _S_CLASS_NL[1:] + ". Elke head apart gedraaid vanaf `start`, alleen queries en keys, en met een cache eerst de keys en values daaruit?"},
            "correct": {"en": "Your attention turns the queries and keys of every head with RoPE, shares each key and value head among `h / kv_heads` query heads, and with a cache gives the new token what it would get with all tokens at once, up to rounding. It is in your project now, as `{saved}`.",
                        "nl": "Je attention draait de queries en keys van elke head met RoPE, deelt elke key head en value head met `h / kv_heads` query heads, en geeft met een cache het nieuwe token wat het met alle tokens tegelijk zou krijgen, op afronding na. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `ModernAttention` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `ModernAttention` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "modern-gpt": ClassBox(
        name="modern-gpt",
        start=_G_START,
        solution=_g(),
        scenarios=_G_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("llm", "modern_gpt"),
        first=[_DOTS],
        mistakes=[
            Mistake("caches-list", error=lambda e: isinstance(e, TypeError) and "list indices must be integers or slices, not str" in str(e),
                    en="A block gets the whole list of caches, and its attention needs one dict. Give every block its own: `self.blocks[i](x, caches[i])`.",
                    nl="Een block krijgt de hele lijst caches, en zijn attention heeft één dict nodig. Geef elk block zijn eigen: `self.blocks[i](x, caches[i])`."),
            Mistake("first-cache", syntax=r"caches\[0\]",
                    en="Every block gets `caches[0]`, the cache of the first block, so the second block finds the keys and values of the first in it, and its keys no longer fit its queries. Give block `i` its own: `caches[i]`.",
                    nl="Elk block krijgt `caches[0]`, de cache van het eerste block, dus het tweede block vindt daarin de keys en values van het eerste, en zijn keys passen niet meer bij zijn queries. Geef block `i` zijn eigen: `caches[i]`."),
            Mistake("caches-none", error=lambda e: isinstance(e, TypeError) and str(e) in ("'NoneType' object is not iterable", "'NoneType' object is not subscriptable"),
                    en="Without caches, as while training, `caches` is None, and Python cannot take a cache out of None. Give block `i` its cache, `caches[i]`, only when `caches` is not None; else call the block without one.",
                    nl="Zonder caches, zoals tijdens het trainen, is `caches` None, en Python kan geen cache uit None halen. Geef block `i` zijn cache, `caches[i]`, alleen als `caches` niet None is; roep het block anders aan zonder cache."),
            Mistake("hidden-4d", wrong=_g(block_init=_G_BLOCK_INIT.replace("int(8 * d / 3)", "4 * d")),
                    en=_S_CLASS + ": your SwiGLU has a hidden size of `4 * d`. It has three weight matrices where the MLP of your Block had two, so at `4 * d` it has about one and a half times as many parameters. With `int(8 * d / 3)` it has about as many as before, and the change can be measured fairly.",
                    nl=_S_CLASS_NL + ": je SwiGLU heeft een hidden size van `4 * d`. Hij heeft drie weight-matrices waar de MLP van je Block er twee had, dus bij `4 * d` heeft hij er ongeveer anderhalf keer zoveel. Met `int(8 * d / 3)` heeft hij er ongeveer evenveel als eerst, en is de verandering eerlijk te meten."),
            Mistake("layernorm", wrong=_g(block_init=_G_BLOCK_INIT.replace("RMSNorm(d)", "nn.LayerNorm(d)"), init=_G_INIT.replace("RMSNorm(d)", "nn.LayerNorm(d)")),
                    en=_S_CLASS + ": your GPT still normalizes with `nn.LayerNorm`. Use your `RMSNorm(d)`, from `llm.modern_parts`, in all three places: `norm1`, `norm2` and `norm`.",
                    nl=_S_CLASS_NL + ": je GPT normaliseert nog met `nn.LayerNorm`. Gebruik je `RMSNorm(d)`, uit `llm.modern_parts`, op alle drie de plekken: `norm1`, `norm2` en `norm`."),
            Mistake("no-norm", wrong=_g(forward=_G_FORWARD.replace("self.head(self.norm(x))", "self.head(x)")),
                    en=_S_CLASS + ": the residual stream goes into the output head without the last normalization. Every pre-norm block normalizes only a copy, so the stream itself still needs it: `self.head(self.norm(x))`.",
                    nl=_S_CLASS_NL + ": de residual stream gaat de output head in zonder de laatste normalisatie. Elk pre-norm block normaliseert alleen een kopie, dus de stream zelf heeft hem nog nodig: `self.head(self.norm(x))`."),
            Mistake("plain-list", wrong=_g(init=_G_INIT.replace("nn.ModuleList([ModernBlock(d, h, kv_heads) for _ in range(n_layers)])", "[ModernBlock(d, h, kv_heads) for _ in range(n_layers)]")),
                    en=_S_CLASS + ": PyTorch does not see the parameters of your blocks, as they sit in a plain list; an optimizer would never train them. Put them in an `nn.ModuleList`, as in your GPT.",
                    nl=_S_CLASS_NL + ": PyTorch ziet de parameters van je blocks niet, omdat ze in een gewone lijst staan; een optimizer zou ze nooit trainen. Zet ze in een `nn.ModuleList`, zoals in je GPT."),
            Mistake("one-block", wrong=_g(init=_G_INIT.replace("nn.ModuleList([ModernBlock(d, h, kv_heads) for _ in range(n_layers)])", "nn.ModuleList([ModernBlock(d, h, kv_heads)] * n_layers)")),
                    en=_S_CLASS + ": your blocks are one and the same block, n_layers times, with the same weights. Make a new `ModernBlock(d, h, kv_heads)` for every place in the list.",
                    nl=_S_CLASS_NL + ": je blocks zijn één en hetzelfde block, n_layers keer, met dezelfde weights. Maak een nieuw `ModernBlock(d, h, kv_heads)` voor elke plek in de lijst."),
            Mistake("residual-twice", wrong=_g(forward=_G_FORWARD.replace("x = self.blocks[i](x", "x = x + self.blocks[i](x")),
                    en=_S_CLASS + ": your block already adds its result to the residual stream, so `x + self.blocks[i](...)` counts the stream twice. Let every block give the new stream: `x = self.blocks[i](...)`.",
                    nl=_S_CLASS_NL + ": je block telt zijn uitkomst al op bij de residual stream, dus `x + self.blocks[i](...)` telt de stream twee keer. Laat elk block de nieuwe stream geven: `x = self.blocks[i](...)`."),
            Mistake("cache-not-passed", wrong=_g(block_forward=_G_BLOCK_FORWARD.replace("self.attention(self.norm1(x), cache)", "self.attention(self.norm1(x))")),
                    en=_S_CLASS + ": with caches, the new token sees only itself, at place 0: the keys and values of the first call do not reach its attention. Your GPT gives block `i` its own, `self.blocks[i](x, caches[i])`, and no cache when `caches` is None; your block passes it on, `self.attention(self.norm1(x), cache)`.",
                    nl=_S_CLASS_NL + ": met caches ziet het nieuwe token alleen zichzelf, op plek 0: de keys en values van de eerste aanroep komen niet bij zijn attention aan. Je GPT geeft block `i` zijn eigen, `self.blocks[i](x, caches[i])`, en geen cache als `caches` None is; je block geeft hem door, `self.attention(self.norm1(x), cache)`."),
            Mistake("order", matches=_g_order,
                    en=_S_CLASS + ": the right shape, other numbers. Make the layers in the order the data passes them, `norm1`, `attention`, `norm2`, `mlp` in a block and `tok`, `blocks`, `norm`, `head` in the GPT, so that they get the same random starting values as in the course's solution; the rest of your classes is right.",
                    nl=_S_CLASS_NL + ": de goede shape, andere getallen. Maak de layers in de volgorde waarin de data erdoor gaat, `norm1`, `attention`, `norm2`, `mlp` in een block en `tok`, `blocks`, `norm`, `head` in de GPT, zodat ze dezelfde willekeurige startwaarden krijgen als in de oplossing van de course; de rest van je classes klopt."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the four `...`: the parts of a block and its recipe, then the parts of the GPT and its recipe.",
                      "nl": "Vervang de vier `...`: de onderdelen van een block en zijn recept, dan de onderdelen van de GPT en zijn recept."},
            "wrong": {"en": _S_CLASS + ". A block of RMSNorm, ModernAttention, RMSNorm and SwiGLU with `int(8 * d / 3)`, and a GPT of tokens without places, the blocks, RMSNorm and the head?",
                      "nl": _S_CLASS_NL + ". Een block van RMSNorm, ModernAttention, RMSNorm en SwiGLU met `int(8 * d / 3)`, en een GPT van tokens zonder plekken, de blocks, RMSNorm en de head?"},
            "almost": {"en": "Right for the first example. But " + _S_CLASS[0].lower() + _S_CLASS[1:] + ". Pre-norm and the residual stream as in your Block, and every block with its own cache?",
                       "nl": "Goed voor het eerste voorbeeld. Maar " + _S_CLASS_NL[0].lower() + _S_CLASS_NL[1:] + ". Pre-norm en de residual stream zoals in je Block, en elk block met zijn eigen cache?"},
            "correct": {"en": "Your ModernGPT has no table of places, and no window size among its sizes: RoPE gives every token its place in every attention. With caches, a new token gets the logits it would get with all tokens at once, up to rounding. It is in your project now, as `{saved}`.",
                        "nl": "Je ModernGPT heeft geen tabel van plekken, en bij zijn maten hoort geen venstergrootte: RoPE geeft elk token zijn plek in elke attention. Met caches krijgt een nieuw token de logits die het met alle tokens tegelijk zou krijgen, op afronding na. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `ModernGPT` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `ModernGPT` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "generate": ClassBox(
        name="generate",
        start=_X_START,
        solution=_x(),
        scenarios=_X_SCENARIOS,
        save_as=("llm", "modern_generate"),
        first=[_DOTS],
        mistakes=[
            Mistake("one-dict", error=lambda e: isinstance(e, KeyError) and e.args == (0,),
                    en="Your GPT looks up `caches[0]`, the cache of its first block, and your `caches` has none: it is one dict. Make a list with an empty dict for every block: `caches.append({})` in a loop over `model.blocks`.",
                    nl="Je GPT zoekt `caches[0]` op, de cache van zijn eerste block, en je `caches` heeft die niet: het is één dict. Maak een lijst met een lege dict voor elk block: `caches.append({})` in een loop over `model.blocks`."),
            Mistake("no-cache", wrong=_x(loop=_X_LOOP.replace("logits = model(torch.tensor([[j]]), caches)", "logits = model(torch.tensor([out]))")),
                    en=_S_CODE + ": your `generate` sends all ids through the model at every step. The ids are right, but that is writing without a cache: every step does all the work again. Send the start once, with the caches, and then only the new id: `model(torch.tensor([[j]]), caches)`.",
                    nl=_S_CODE_NL + ": je `generate` stuurt bij elke stap alle ids door het model. De ids kloppen, maar zo schrijf je zonder cache: elke stap doet al het werk opnieuw. Stuur de start één keer, met de caches, en daarna alleen het nieuwe id: `model(torch.tensor([[j]]), caches)`."),
            Mistake("no-caches", wrong=_x(loop=_X_LOOP.replace("logits = model(torch.tensor([[j]]), caches)", "logits = model(torch.tensor([[j]]))")),
                    en=_S_CODE + ": the new id goes through the model without the caches, so it sees only itself, at place 0. Give the caches every time: `model(torch.tensor([[j]]), caches)`.",
                    nl=_S_CODE_NL + ": het nieuwe id gaat zonder de caches door het model, dus het ziet alleen zichzelf, op plek 0. Geef de caches elke keer mee: `model(torch.tensor([[j]]), caches)`."),
            Mistake("same-dict", syntax=r"\[\s*\{\s*\}\s*\]\s*\*",
                    en="`[{}] * len(model.blocks)` is a list with the same dict in every place, so all blocks keep their keys and values in one cache, and each overwrites the others. Make a new `{}` for every block, in a loop.",
                    nl="`[{}] * len(model.blocks)` is een lijst met dezelfde dict op elke plek, dus alle blocks bewaren hun keys en values in één cache, en elk overschrijft die van de andere. Maak een nieuwe `{}` voor elk block, in een loop."),
            Mistake("first-place", wrong=_x(loop=_X_LOOP.replace("logits[0, -1]", "logits[0, 0]")),
                    en=_S_CODE + ": the next id comes from the logits of the first place of the start, and should come from those of the last: `logits[0, -1]`.",
                    nl=_S_CODE_NL + ": het volgende id komt uit de logits van de eerste plek van de start, en moet uit die van de laatste komen: `logits[0, -1]`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...`: the empty caches, the logits of the whole start, and in the loop the next id and the step with only that id.",
                      "nl": "Vervang de drie `...`: de lege caches, de logits van de hele start, en in de loop het volgende id en de stap met alleen dat id."},
            "wrong": {"en": _S_CODE + ". An empty dict for every block, the start once through the model with the caches, then every step the likeliest id after the last place, and only that id with the caches?",
                      "nl": _S_CODE_NL + ". Een lege dict voor elk block, de start één keer door het model met de caches, en dan elke stap het waarschijnlijkste id na de laatste plek, en alleen dat id met de caches?"},
            "almost": {"en": "Right for the first example. But " + _S_CODE[0].lower() + _S_CODE[1:] + ". Does your `generate` use `n` and `ids`, and send only the new id with the caches?",
                       "nl": "Goed voor het eerste voorbeeld. Maar " + _S_CODE_NL[0].lower() + _S_CODE_NL[1:] + ". Gebruikt je `generate` `n` en `ids`, en stuurt hij alleen het nieuwe id met de caches?"},
            "correct": {"en": "Your `generate` writes the same ids as the slow way, which sends all ids through the model at every step, and after the start it sends one token per step. It is in your project now, as `{saved}`.",
                        "nl": "Je `generate` schrijft dezelfde ids als de langzame manier, die bij elke stap alle ids door het model stuurt, en na de start stuurt hij één token per stap. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `generate` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `generate` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# The run uses the learner's tokenizer and run of gpt-train-bpe, and this lesson's modules; where one is missing or
# broken, the course's version stands in (template 1.3). The GPT of gpt-assemble is not needed: ModernGPT replaces it.
# Not modern_generate: the run does not use it, and the second demo waits for the learner's own (exercise 4, below).
BUILDS_ON = tuple(m for m in B.BUILDS_ON if m[1] not in ("attention", "block", "gpt")) + (
    ("llm", "modern_parts", "llm-modern-gpt", "parts"), ("llm", "modern_attention", "llm-modern-gpt", "modern-attention"),
    ("llm", "modern_gpt", "llm-modern-gpt", "modern-gpt"))
projects.builds_on(*BUILDS_ON)


# --- the demo: your ModernGPT on the tokens of gpt-train-bpe ---------------------------------------------------------------

# The changes of the course's runs, in the order the lesson adds them, each on top of the ones before: the GPT of
# gpt-train-bpe, then RMSNorm, SwiGLU, RoPE in place of the table of places, and grouped-query attention, the ModernGPT.
CHANGES = ["gpt", "rms", "swiglu", "rope", "gqa"]
# --- measured: the course's runs (made by the course's scripts; see the notes) ---
# The course's runs of the demo's train with its own solutions, one thread, PyTorch 2.14.1: every change on the CPU plan
# with seeds 0, 1 and 2, and the ModernGPT on the GPU plan with seed 0, which ran on the course's CPU with 3 threads.
# Per run its seed, minutes, parameters, best validation loss per token and per character, and the loss after every
# epoch (epoch, training, validation). "gpt" is the GPT of gpt-train-bpe: its runs are in that lesson.
# The ModernGPT of seed 0 on the CPU plan (gqa) was recorded again with what it wrote after every epoch (snapper of
# gpt-train-bpe), with the same losses: its snapshots.
REFERENCE = {
    "cpu": {
        "rms": [
            {"seed": 0, "minutes": 44.2, "params": 1327721, "best": 2.4818, "best_epoch": 10, "best_per_char": 0.7374,
             "history": [(1, 4.7391, 3.6654), (2, 3.3947, 3.1979), (3, 3.0498, 2.9569), (4, 2.8318, 2.7941), (5, 2.6789, 2.6782), (6, 2.5556, 2.5962), (7, 2.4721, 2.536), (8, 2.4068, 2.5003), (9, 2.3686, 2.4841), (10, 2.3488, 2.4818)]},
            {"seed": 1, "minutes": 45.0, "params": 1327721, "best": 2.485, "best_epoch": 10, "best_per_char": 0.7383,
             "history": [(1, 4.7329, 3.6747), (2, 3.4085, 3.2088), (3, 3.0543, 2.9597), (4, 2.8386, 2.7949), (5, 2.6752, 2.6819), (6, 2.5614, 2.5962), (7, 2.4704, 2.5398), (8, 2.4085, 2.5027), (9, 2.3665, 2.4881), (10, 2.3486, 2.485)]},
            {"seed": 2, "minutes": 39.2, "params": 1327721, "best": 2.4736, "best_epoch": 10, "best_per_char": 0.7349,
             "history": [(1, 4.7653, 3.6759), (2, 3.3996, 3.1997), (3, 3.0512, 2.9564), (4, 2.829, 2.7918), (5, 2.6697, 2.6678), (6, 2.5462, 2.5796), (7, 2.4277, 2.5303), (8, 2.3921, 2.4933), (9, 2.3544, 2.4764), (10, 2.3335, 2.4736)]},
        ],
        "swiglu": [
            {"seed": 0, "minutes": 45.0, "params": 1323881, "best": 2.4126, "best_epoch": 10, "best_per_char": 0.7168,
             "history": [(1, 4.6567, 3.5639), (2, 3.3134, 3.1202), (3, 2.9633, 2.8718), (4, 2.7334, 2.7111), (5, 2.5777, 2.5916), (6, 2.4514, 2.5099), (7, 2.3705, 2.4592), (8, 2.308, 2.4283), (9, 2.2716, 2.4144), (10, 2.2522, 2.4126)]},
            {"seed": 1, "minutes": 46.0, "params": 1323881, "best": 2.4079, "best_epoch": 10, "best_per_char": 0.7154,
             "history": [(1, 4.6484, 3.5823), (2, 3.3168, 3.1259), (3, 2.9629, 2.8768), (4, 2.7349, 2.7016), (5, 2.5629, 2.5821), (6, 2.4493, 2.5024), (7, 2.3607, 2.4525), (8, 2.303, 2.4231), (9, 2.262, 2.4099), (10, 2.2445, 2.4079)]},
            {"seed": 2, "minutes": 39.6, "params": 1323881, "best": 2.4031, "best_epoch": 10, "best_per_char": 0.714,
             "history": [(1, 4.6565, 3.5676), (2, 3.3094, 3.1151), (3, 2.9493, 2.856), (4, 2.7148, 2.6874), (5, 2.5503, 2.5697), (6, 2.4327, 2.493), (7, 2.3132, 2.4495), (8, 2.286, 2.4189), (9, 2.2499, 2.4052), (10, 2.2288, 2.4031)]},
        ],
        "rope": [
            {"seed": 0, "minutes": 53.6, "params": 1307497, "best": 2.2267, "best_epoch": 10, "best_per_char": 0.6616,
             "history": [(1, 4.2616, 3.1399), (2, 2.8922, 2.7319), (3, 2.5517, 2.507), (4, 2.3462, 2.3925), (5, 2.2311, 2.3216), (6, 2.1345, 2.2776), (7, 2.0665, 2.2497), (8, 2.0076, 2.2333), (9, 1.9683, 2.2269), (10, 1.9461, 2.2267)]},
            {"seed": 1, "minutes": 51.7, "params": 1307497, "best": 2.2247, "best_epoch": 10, "best_per_char": 0.661,
             "history": [(1, 4.2524, 3.1209), (2, 2.8696, 2.7056), (3, 2.5379, 2.4939), (4, 2.3496, 2.3894), (5, 2.2241, 2.3185), (6, 2.135, 2.2759), (7, 2.0607, 2.2457), (8, 2.0062, 2.23), (9, 1.9603, 2.2248), (10, 1.9412, 2.2247)]},
            {"seed": 2, "minutes": 45.8, "params": 1307497, "best": 2.2273, "best_epoch": 10, "best_per_char": 0.6618,
             "history": [(1, 4.2556, 3.1428), (2, 2.8783, 2.7092), (3, 2.5422, 2.4952), (4, 2.3513, 2.3855), (5, 2.2283, 2.3191), (6, 2.1367, 2.2741), (7, 2.0484, 2.2497), (8, 2.0005, 2.2338), (9, 1.9599, 2.2279), (10, 1.9388, 2.2273)]},
        ],
        "gqa": [
            {"seed": 0, "minutes": 49.7, "params": 1225577, "best": 2.2415, "best_epoch": 10, "best_per_char": 0.666,
             "history": [(1, 4.2851, 3.183), (2, 2.9237, 2.7532), (3, 2.5787, 2.521), (4, 2.3754, 2.4085), (5, 2.2601, 2.341), (6, 2.1636, 2.2976), (7, 2.0969, 2.266), (8, 2.0397, 2.2494), (9, 2.0018, 2.242), (10, 1.9795, 2.2415)],
             "snapshots": [
              {"epoch": 1, "text": "Once upon a time, there was a boy named Tim. Tim liked to turten time, but he was not a fight. It was con what he wanted to do. He wanted to help himself.\nTim went to the warm and came to make"},
              {"epoch": 2, "text": "Once upon a time, there was a boy named Tim. Tim liked to a long. He had no friends. He was wise for his mom where how he could make it. They all played with his boat and had fun.\nOne day, Tim and Sue"},
              {"epoch": 3, "text": "Once upon a time, there was a smelly boy named Tom.\nTom and Sam were playing with the big, scary dog. They shared even more knocked in the clumser. Tom and Sue were very surprised. They had fun"},
              {"epoch": 4, "text": "Once upon a time, there was a smart dog named Spot. Spot was very selfish. He loved to play with his friends. From that day on, Spot and Max felt very happy to laugh.\nOne day, Spot was playing in the"},
              {"epoch": 5, "text": "Once upon a time, there was a bald man who loved to eat velir. One day, he noticed a little bird where people came to life! They were very sad and angry.\nThe spider thought, \"I know"},
              {"epoch": 6, "text": "Once upon a time, there was a little boy named Tim. Tim loved to wear an air in the front of paper until him happened. One day, Tim saw a big rock on the ground. He wanted to slide down the rock and"},
              {"epoch": 7, "text": "Once upon a time, there was a kitten named Tom. Tom and Sam big went to play with all the his friends. They liked to swing high in the sky.\nOne day, Tom found a big box in the park. He was very happy and"},
              {"epoch": 8, "text": "Once upon a time, there was a little boy named Tim. Tim loved to read books. One day, he saw a big red balloon near his house. He was very sad.\nTim decided to help his mom. \"What's wr"},
              {"epoch": 9, "text": "Once upon a time, there was a little boy named Tim. Tim loved to go to the park with his mom.\nOne day, Tim was going for a walk in the park. He saw a girl named Sue. Sue was sad and asked, \"Did"},
              {"epoch": 10, "text": "Once upon a time, there was a little boy named Tim. Tim loved to go to the park with his mom.\nOne day, Tim was going for a walk in the park. He saw a toy snake with a cabin on it. He thought"}]},
            {"seed": 1, "minutes": 49.0, "params": 1225577, "best": 2.2456, "best_epoch": 10, "best_per_char": 0.6672,
             "history": [(1, 4.2776, 3.1658), (2, 2.9067, 2.7426), (3, 2.5711, 2.5209), (4, 2.3784, 2.4097), (5, 2.2515, 2.3414), (6, 2.1646, 2.2959), (7, 2.0925, 2.2687), (8, 2.039, 2.2514), (9, 1.9942, 2.246), (10, 1.9762, 2.2456)]},
            {"seed": 2, "minutes": 42.7, "params": 1225577, "best": 2.2397, "best_epoch": 10, "best_per_char": 0.6654,
             "history": [(1, 4.2741, 3.1695), (2, 2.91, 2.7391), (3, 2.5659, 2.5166), (4, 2.3735, 2.4106), (5, 2.252, 2.3407), (6, 2.1619, 2.2908), (7, 2.0764, 2.2621), (8, 2.0293, 2.2457), (9, 1.99, 2.2401), (10, 1.9695, 2.2397)]},
        ],
    },
    "gpu": [
        {"seed": 0, "minutes": 190.2, "params": 10411113, "best": 2.0445, "best_epoch": 8, "best_per_char": 0.6074,
         "history": [(1, 5.019, 3.5597), (2, 3.158, 2.82), (3, 2.6144, 2.4696), (4, 2.2563, 2.2249), (5, 2.0173, 2.1145), (6, 1.8283, 2.0569), (7, 1.6701, 2.0475), (8, 1.5508, 2.0445), (9, 1.3968, 2.0829), (10, 1.2339, 2.1449), (11, 1.112, 2.2109), (12, 0.9687, 2.2968), (13, 0.7799, 2.4154), (14, 0.6999, 2.4847), (15, 0.6203, 2.5551), (16, 0.5322, 2.6191), (17, 0.4989, 2.6525), (18, 0.4241, 2.6814), (19, 0.4239, 2.6894), (20, 0.4215, 2.69)]},
    ],
}
# Seconds per training step of every version on the CPU plan, one thread, with nothing else running on the course's
# computer: the runs above ran four at a time, so their own minutes say less about the time a run takes alone.
STEP_SECONDS = {"gpt": 0.632, "rms": 0.73, "swiglu": 0.699, "rope": 0.873, "gqa": 0.788}
# A run that ends at or below this validation loss per token has learned as the course's runs did: the highest end
# of those runs, plus a margin for another computer, another PyTorch, and float rounding on a GPU.
THRESHOLD = {"cpu": 2.31, "gpu": 2.2}
# --- end of measured ---


def runs(change: str, where: str = "cpu") -> list:
    """The course's runs of one change on a plan; the GPT of gpt-train-bpe for "gpt"."""
    if change == "gpt":
        return B.REFERENCE[where]
    return REFERENCE["cpu"][change] if where == "cpu" else (REFERENCE["gpu"] if change == "gqa" else [])


def minutes(change: str) -> float:
    """The minutes a run of one version takes on the CPU plan alone, one thread: the GPT on tokens took the minutes of its
    runs in gpt-train-bpe, and every other version as many times longer as its steps take."""
    base = sum(r["minutes"] for r in B.REFERENCE["cpu"]) / len(B.REFERENCE["cpu"])
    return base * STEP_SECONDS[change] / STEP_SECONDS["gpt"]


def spread(change: str, where: str = "cpu") -> float:
    """The highest minus the lowest best validation loss of the course's runs of one version."""
    found = [r["best"] for r in runs(change, where)]
    return max(found) - min(found)


def mean_best(change: str, where: str = "cpu") -> float:
    found = runs(change, where)
    return sum(r["best"] for r in found) / len(found)

# --- the demo: the learner's modern GPT on the tokens of gpt-train-bpe -------------------------------------------------

KV_HEADS = {"gpu": 2, "cpu": 2}   # key and value heads of the plan: 6 and 4 query heads share them


def n_params(vocab: int, d: int, h: int, kv_heads: int, n_layers: int) -> int:
    """The parameters of the course's ModernGPT."""
    d_head = int(d / h)
    hidden = int(8 * d / 3)
    block = 2 * d + d * d + 2 * d * kv_heads * d_head + d * d + d + 3 * d * hidden
    return vocab * d + n_layers * block + d + d * vocab + vocab


def plan_for(name: str) -> dict:
    """A plan of gpt-train-bpe by its name, "cpu" or "gpu", with the key and value heads and the parameters of a
    ModernGPT, and those of the GPT on tokens."""
    p = B.PLANS[name]
    d, h, n_layers, block = p["sizes"]
    return {**p, "kv_heads": KV_HEADS[name], "params": n_params(B.VOCAB, d, h, KV_HEADS[name], n_layers),
            "params_gpt": B.n_params(B.VOCAB, d, n_layers, block)}


def on_device(device: str):
    """For a forward pass of a ModernGPT outside the trainer: the tensors it makes itself, such as its causal mask and
    its places, go to the device of the run, as in the trainer of gpt-train-chars. On the CPU nothing changes."""
    import contextlib
    return contextlib.nullcontext() if device == "cpu" else torch.device(device)


def plan() -> dict:
    """What the demo trains on this computer: the device of gpt-train-bpe, and its plan for a ModernGPT."""
    p = B.plan()
    return {**p, **plan_for("cpu" if p["device"] == "cpu" else "gpu")}


def signed(x: float, digits: int, lang: str) -> str:
    """A change with its sign: +0.01 or −0.03, with a real minus sign."""
    text = number(abs(x), digits, lang)
    return ("−" if x < 0 else "+") + text


def _saved(module: str) -> bool:
    return (projects.folder() / "llm" / f"{module}.py").exists()


def _tools() -> dict | str:
    """The learner's modules the run needs, by name; "run" when one of them cannot be imported."""
    run = run_code("from llm.fast_bpe import train_bpe_fast\nfrom llm.fast_encode import encode_fast\nfrom llm.decode import decode\n"
                   "from llm.encode import encode\nfrom llm.modern_gpt import ModernGPT\nfrom llm.gpt_loaders import gpt_loaders\n"
                   "from nn.trainer import Trainer")
    if run.error is not None:
        return "run"
    return run.names


def _make(tools: dict, p: dict):
    d, h, n_layers, block = p["sizes"]
    return tools["ModernGPT"](B.VOCAB, d, h, p["kv_heads"], n_layers)


def train(show, seed: int = 0, epochs: int | None = None, make=None) -> dict | str | None:
    """Train the learner's ModernGPT as gpt-train-bpe trains their GPT: their tokenizer on the first 90% of the stories,
    then the GPT on the tokens of all of them, with the same plan, mini-batches, learning rate and seed. `show` gets a dict
    with how far the run is. None when modern_gpt.py is not in the project yet; "run" when the run stops with an error;
    "memory" when the GPU runs out of memory; "stopped" when the stop button ended the run before its first epoch did;
    else the run, saved as work/projects/llm/gpt_modern.pt and gpt_modern.json, unless an earlier run had a lower
    validation loss. The course's own runs pass `make`, which builds another GPT from the tools and the plan: each change
    on its own (see REFERENCE)."""
    if not _saved("modern_gpt"):
        return None
    tools = _tools()
    if isinstance(tools, str):
        return tools
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    make = make or _make
    p = plan()
    where = p["device"]
    epochs = p["epochs"] if epochs is None else epochs
    d, h, n_layers, block = p["sizes"]
    if where == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    checkpoint, info = _paths()
    fresh = checkpoint.with_name("gpt_modern_new.pt")
    fresh.unlink(missing_ok=True)
    start = time.time()
    try:
        show({"phase": "tokenizer"})
        merges = tools["train_bpe_fast"](B.split_text()[0], B.MERGES)
        show({"phase": "encode"})
        ids = tools["encode_fast"](B.text(), merges)
        if len(merges) != B.MERGES or max(ids) >= B.VOCAB:
            return "run"
        torch.manual_seed(seed)
        train_loader, val_loader = tools["gpt_loaders"](ids, block, p["batch"], where)
        model = make(tools, p).to(where)
        trainer = tools["Trainer"](model, train_loader, val_loader, epochs, lr=0.001, warmup=100, max_norm=1.0, path=str(fresh))
        trainer.model = B.T._OnDevice(trainer.model, where)
    except Exception:
        return "run"
    steps = len(trainer.train_loader)
    state = {"step": 0, "shown": 0.0, "start": time.time()}
    snap = B.snapper(trainer, model, merges, block, seed, tools["decode"], tools["encode"])

    def step():
        snap()   # the epoch before has ended: what the GPT writes now
        state["step"] += 1
        now = time.time()
        if now - state["shown"] >= 2 or state["step"] == 1:
            state["shown"] = now
            done = state["step"] - 1
            seconds = now - state["start"]
            left = seconds / done * (epochs * steps - done) if done >= 10 else None
            show({"phase": "train", "epoch": done // steps + 1, "epochs": epochs, "step": done, "steps": epochs * steps,
                  "seconds": seconds, "left": left, "history": list(trainer.history)})

    trainer.train_loader = B.T._Counted(trainer.train_loader, step,
                                        B._shifted(tools["gpt_loaders"], ids, block, p["batch"], where, epochs, seed))
    stopped = False
    try:
        trainer.fit()
        B.T._sync(where)
    except KeyboardInterrupt:   # the stop button of marimo
        stopped = True
    except RuntimeError as e:
        return "memory" if "out of memory" in str(e).lower() else "run"
    except Exception:
        return "run"
    snap()   # after the last epoch, or after a stop: an epoch in the history without a story has had no step since it
    #          ended, so the model is still the one at its end
    history = [{k: int(v) if k == "epoch" else float(v) for k, v in hist.items()} for hist in trainer.history]
    if not history or not fresh.exists():
        fresh.unlink(missing_ok=True)
        return "stopped"
    cut = int(0.9 * len(ids))
    val_tokens, val_chars = len(ids) - cut, len(tools["decode"](ids[cut:], merges))
    best = min(history, key=lambda hist: hist["val_loss"])
    result = {"device": where, "name": p["name"], "sizes": list(p["sizes"]), "kv_heads": p["kv_heads"], "batch": p["batch"],
              "epochs": epochs, "params": p["params"], "seed": seed, "stopped": stopped, "minutes": (time.time() - start) / 60,
              "history": history, "merges": [[a, b] for (a, b) in merges], "stand_ins": stand_ins,
              "tokens": len(ids), "chars": len(B.text()), "val_tokens": val_tokens, "val_chars": val_chars,
              "best": best["val_loss"], "best_epoch": best["epoch"], "snapshots": snap.kept,
              "best_per_char": B.per_char(best["val_loss"], val_tokens, val_chars)}
    try:
        model = make(tools, p)
        model.load_state_dict(torch.load(fresh, map_location="cpu", weights_only=True))
        result["story"] = B.write(model, merges, block, seed, tools["decode"], tools["encode"])
    except Exception:
        result["story"] = None
    old = last_run()
    if old is None or not checkpoint.exists() or result["best"] <= old["best"]:
        fresh.replace(checkpoint)
        info.write_text(json.dumps(result, indent=1), encoding="utf-8")
        result["kept"] = None
    else:
        fresh.unlink()
        result["kept"] = old
    return result


def _paths(name: str = "gpt_modern") -> tuple:
    folder = projects.folder() / "llm"
    return folder / f"{name}.pt", folder / f"{name}.json"


def last_run(name: str = "gpt_modern") -> dict | None:
    """The last run the demo saved, or None."""
    _, info = _paths(name)
    try:
        return json.loads(info.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


COURSE_GPT = "data/course-modern-gpt"   # .pt and .json: the course's ModernGPT, trained on Cosmopedia, see data/README.md


def course_bpe() -> dict:
    """What the JSON of the course's GPT on tokens of gpt-train-bpe holds: trained on the same stories as the course's
    ModernGPT, so their losses compare."""
    return json.loads((paths.ROOT / B.COURSE_GPT).with_suffix(".json").read_text(encoding="utf-8"))


def trained() -> dict:
    """Which trained ModernGPT the lesson uses: the learner's of the first demo ("yours"), else the course's ("course"),
    with what its JSON holds (sizes, key and value heads, merge rules, losses, story) and its checkpoint."""
    folder = projects.folder() / "llm"
    for source, base in (("yours", folder / "gpt_modern"), ("course", paths.ROOT / COURSE_GPT)):
        pt, info = base.with_suffix(".pt"), base.with_suffix(".json")
        if pt.exists() and info.exists():
            try:
                meta = json.loads(info.read_text(encoding="utf-8"))
            except ValueError:
                continue
            return {**meta, "source": source, "path": pt}
    return {"source": None}


def speed() -> dict | str | None:
    """For the second demo: a trained ModernGPT (the learner's of the first demo, else the course's) writes greedily
    from START, as many ids as fit in the window of its plan, once with the learner's generate and its KV cache, and
    once the slow way, every step all ids through the model. None when modern_generate.py is not in the project yet;
    "run" when it stops with an error; else the seconds of both, whether they wrote the same ids, and the numbers the
    caches hold, with grouped-query attention and with as many key and value heads as query heads."""
    if not _saved("modern_generate"):
        return None
    g = trained()
    if g["source"] is None:
        return None
    run = run_code("from llm.modern_generate import generate\nfrom llm.modern_gpt import ModernGPT\nfrom llm.encode import encode")
    if run.error is not None:
        return "run"
    d, h, n_layers, block = g["sizes"]
    merges = {(a, b): 256 + i for i, (a, b) in enumerate(g["merges"])}
    try:
        model = run.names["ModernGPT"](256 + len(merges) + 1, d, h, g["kv_heads"], n_layers)
        state = torch.load(g["path"], map_location="cpu", weights_only=True)
        model.load_state_dict({k: v.float() for k, v in state.items()})
        model.eval()
        ids = run.names["encode"](B.START, merges)
        n = block - len(ids)
        cached, full = [], []
        for _ in range(3):   # the best of three: a computer that is busy for a moment makes one of them slower
            start = time.perf_counter()
            fast = run.names["generate"](model, list(ids), n)
            cached.append(time.perf_counter() - start)
            start = time.perf_counter()
            slow = list(ids)
            with torch.no_grad():
                for _ in range(n):
                    slow.append(int(model(torch.tensor([slow]))[0, -1].argmax()))
            full.append(time.perf_counter() - start)
        cached, full = min(cached), min(full)
    except Exception:
        return "run"
    d_head = int(d / h)
    return {"source": g["source"], "tokens": n, "start": len(ids), "cache": cached, "full": full, "same": fast == slow,
            "numbers": n_layers * 2 * g["kv_heads"] * d_head * block, "numbers_mha": n_layers * 2 * h * d_head * block,
            "layers": n_layers, "h": h, "kv_heads": g["kv_heads"], "d_head": d_head, "block": block}


# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_P_LOOP_ROPE = _P_ROPE.replace("    angles = (start + torch.arange(n))[:, None] * theta[None, :]\n",
                               "    angles = torch.zeros(n, int(d / 2))\n    for i in range(n):\n"
                               "        angles[i] = (start + i) * theta\n")
CASES = [
    ("parts", _P_START, "empty", None),
    ("parts", "", "empty", None),
    ("parts", _p(rope=_P_ROPE.replace("def rope(x, base=10000.0, start=0):", "def rope(x, start=0, base=10000.0):")), "correct", None),   # start before base: the checks name it; before _p(), which then saves the course's order
    ("parts", _p(), "correct", None),
    ("parts", _p(rope=_P_LOOP_ROPE), "correct", None),   # the angles row by row: another way, the same outcome
    ("parts", _P_START.replace("# your RMSNorm(d, eps=1e-6) of the lesson RMSNorm\n...", _P_RMS.strip()), "wrong", "dots-left"),
    ("parts", _p(swiglu=""), "error", "no-swiglu"),
    ("parts", _p(rms=""), "error", "no-rmsnorm"),
    ("parts", _p(rope=""), "error", "no-rope"),
    ("parts", _p(gqa=""), "error", "no-gqa"),
    ("parts", _p(rope=_P_ROPE.replace("def rope(x, base=10000.0, start=0):", "def rope(x, base=10000.0):").replace(_P_ANGLES, "    angles = torch.arange(n)[:, None] * theta[None, :]\n")), "error", "no-start"),
    ("parts", _p(rope=_P_ROPE.replace(_P_ANGLES, "    angles = torch.arange(n)[:, None] * theta[None, :]\n")), "wrong", "start-unused"),
    ("parts", _p(rope=_P_ROPE.replace("def rope(x, base=10000.0, start=0):", "def rope(x, start, base=10000.0):")), "error", "start-required"),
    ("parts", _p(rope=_P_ROPE.replace(_P_ANGLES, "    angles = torch.arange(n)[:, None] * theta[None, :] + start\n")), "wrong", "start-angle"),
    ("parts", _p(rms=_P_RMS.replace("(x * x).mean", "(x * x).sum")), "wrong", None),
    ("modern-attention", _A_START, "empty", None),
    ("modern-attention", _a(), "correct", None),
    ("modern-attention", _a(forward=_A_FORWARD.replace("        if cache is not None:\n            cache[\"k\"] = k\n            cache[\"v\"] = v\n", "")), "error", "cache-not-kept"),
    ("modern-attention", _a(init=_A_INIT.replace("nn.Linear(d, kv_heads * self.d_head", "nn.Linear(d, d")), "wrong", "kv-full"),
    ("modern-attention", _a(init=_A_INIT.replace("self.out = nn.Linear(d, d)", "self.out = nn.Linear(d, d, bias=False)")), "wrong", "out-no-bias"),
    ("modern-attention", _a(turn="    def turn(self, x, n_heads, start):\n        return rope(x, start=start)\n"), "wrong", "rope-whole"),
    ("modern-attention", _a(forward=_A_FORWARD.replace("        v = self.value(x)\n", "        v = self.turn(self.value(x), self.kv_heads, start)\n")), "wrong", "turn-values"),
    ("modern-attention", _a(forward=_A_FORWARD.replace("k = self.turn(self.key(x), self.kv_heads, start)", "k = self.key(x)")), "wrong", "no-turn-k"),
    ("modern-attention", _a(forward=_A_FORWARD.replace("q = self.turn(self.query(x), self.h, start)", "q = self.query(x)")), "wrong", "no-turn-q"),
    ("modern-attention", _a(turn=_A_TURN.replace("start=start", "start=0")), "wrong", "start-unused"),
    ("modern-attention", _a(forward=_A_FORWARD.replace(_A_CAT, "")), "wrong", "no-cat"),
    ("modern-attention", _a(turn=_A_TURN.replace("start=start", "start")), "wrong", "start-as-base"),
    # the heads by view and transpose, rope on all of them at once: another way, the same outcome
    ("modern-attention", _a(turn="    def turn(self, x, n_heads, start):\n        B, n, _ = x.shape\n"
                              "        heads = x.view(B, n, n_heads, self.d_head).transpose(1, 2)\n"
                              "        return rope(heads, start=start).transpose(1, 2).reshape(B, n, n_heads * self.d_head)\n"), "correct", None),
    # `if cache:` is false for the empty dict of the first call, so nothing is kept
    ("modern-attention", _a(forward=_A_FORWARD.replace("        if cache is not None:\n            cache[", "        if cache:\n            cache[")), "error", "cache-not-kept"),
    ("modern-attention", _a(init=_A_INIT.replace("        self.query = nn.Linear(d, d, bias=False)\n", "").replace(
        "        self.out = nn.Linear(d, d)\n", "        self.out = nn.Linear(d, d)\n        self.query = nn.Linear(d, d, bias=False)\n")), "wrong", "order"),
    ("modern-gpt", _G_START, "empty", None),
    ("modern-gpt", _g(), "correct", None),
    ("modern-gpt", _g(forward=_G_FORWARD.replace("x = self.blocks[i](x, caches[i])", "x = self.blocks[i](x, caches)")), "error", "caches-list"),
    ("modern-gpt", _g(forward="    def forward(self, ids, caches=None):\n        x = self.tok(ids)\n        for b, c in zip(self.blocks, caches):\n"
                       "            x = b(x, c)\n        return self.head(self.norm(x))\n"), "error", "caches-none"),
    ("modern-gpt", _g(forward=_G_FORWARD.replace("            if caches is None:\n                x = self.blocks[i](x)\n            else:\n    ", "")), "error", "caches-none"),
    ("modern-gpt", _g(forward=_G_FORWARD.replace("x = self.blocks[i](x, caches[i])", "x = self.blocks[i](x, caches[0])")), "error", "first-cache"),
    ("modern-gpt", _g(block_init=_G_BLOCK_INIT.replace("int(8 * d / 3)", "4 * d")), "wrong", "hidden-4d"),
    ("modern-gpt", _g(block_init=_G_BLOCK_INIT.replace("RMSNorm(d)", "nn.LayerNorm(d)"), init=_G_INIT.replace("RMSNorm(d)", "nn.LayerNorm(d)")), "wrong", "layernorm"),
    ("modern-gpt", _g(forward=_G_FORWARD.replace("self.head(self.norm(x))", "self.head(x)")), "wrong", "no-norm"),
    ("modern-gpt", _g(init=_G_INIT.replace("nn.ModuleList([ModernBlock(d, h, kv_heads) for _ in range(n_layers)])", "[ModernBlock(d, h, kv_heads) for _ in range(n_layers)]")), "wrong", "plain-list"),
    ("modern-gpt", _g(init=_G_INIT.replace("nn.ModuleList([ModernBlock(d, h, kv_heads) for _ in range(n_layers)])", "nn.ModuleList([ModernBlock(d, h, kv_heads)] * n_layers)")), "wrong", "one-block"),
    ("modern-gpt", _g(forward=_G_FORWARD.replace("x = self.blocks[i](x", "x = x + self.blocks[i](x")), "wrong", "residual-twice"),
    ("modern-gpt", _g(block_forward=_G_BLOCK_FORWARD.replace("self.attention(self.norm1(x), cache)", "self.attention(self.norm1(x))")), "wrong", "cache-not-passed"),
    ("modern-gpt", _g(forward=_G_PLAIN_FORWARD), "wrong", "cache-not-passed"),
    ("modern-gpt", _g(init=_G_INIT.replace("        self.tok = nn.Embedding(vocab, d)\n", "").replace(
        "        self.head = nn.Linear(d, vocab)\n", "        self.head = nn.Linear(d, vocab)\n        self.tok = nn.Embedding(vocab, d)\n")), "wrong", "order"),
    ("modern-gpt", _g(block_init=_G_BLOCK_INIT.replace("        self.mlp = SwiGLU(d, int(8 * d / 3))\n", "").replace(
        "        self.attention = ModernAttention(d, h, kv_heads)\n", "        self.mlp = SwiGLU(d, int(8 * d / 3))\n        self.attention = ModernAttention(d, h, kv_heads)\n")), "wrong", "order"),
    # RMSNorm has no random starting values, so where its lines stand makes no difference
    ("modern-gpt", _g(block_init=_G_BLOCK_INIT.replace("        self.norm1 = RMSNorm(d)\n", "").replace(
        "        self.mlp = SwiGLU(d, int(8 * d / 3))\n", "        self.mlp = SwiGLU(d, int(8 * d / 3))\n        self.norm1 = RMSNorm(d)\n")), "correct", None),
    ("generate", _X_START, "empty", None),
    ("generate", _x(), "correct", None),
    # the loop the other way round: the model first, then the next id; one call fewer at the end
    ("generate", _X_HEAD + "def generate(model, ids, n):\n    caches = [{} for _ in model.blocks]\n    out = list(ids)\n    step = list(ids)\n"
     "    with torch.no_grad():\n        for _ in range(n):\n            logits = model(torch.tensor([step]), caches)\n"
     "            step = [int(logits[0, -1].argmax())]\n            out.append(step[0])\n    return out\n", "correct", None),
    ("generate", _x(caches="    caches = {}\n"), "error", "one-dict"),
    ("generate", _x(caches="    caches = [{}] * len(model.blocks)\n"), "error", "same-dict"),
    ("generate", _x(loop=_X_LOOP.replace("logits = model(torch.tensor([[j]]), caches)", "logits = model(torch.tensor([out]))")), "wrong", "no-cache"),
    ("generate", _x(loop=_X_LOOP.replace("logits = model(torch.tensor([[j]]), caches)", "logits = model(torch.tensor([[j]]))")), "wrong", "no-caches"),
    ("generate", _x(loop=_X_LOOP.replace("logits[0, -1]", "logits[0, 0]")), "wrong", "first-place"),
]
