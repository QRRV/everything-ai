# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson gpt-assemble: GPT, embeddings of tokens and places, a stack of the learner's transformer
blocks, a final layer normalization and an output head; and gpt_examples, the training examples of a GPT, windows of
token ids with the same windows shifted by one as targets. Both are saved in the learner's project
(work/projects/llm/gpt.py and gpt_data.py) when right; the demo builds the learner's GPT for Tiny Shakespeare."""
import html
import math
import re
from functools import cache

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
import torch.nn.functional as F

from everything_ai import paths, projects
from everything_ai.check import ClassBox, FunctionBox, Mistake, Timeout, run_code, with_time_limit

# GPT imports Block of attn-transformer-block, which imports CausalSelfAttention; where those are missing or broken,
# the course's versions stand in (template 1.3).
projects.builds_on(("llm", "attention", "attn-transformer-block", "causal-self-attention"),
                   ("llm", "block", "attn-transformer-block", "block"))

DATA = "data/tiny-shakespeare.txt"   # about 1.1 MB of plays by Shakespeare: see data/README.md
D, H, LAYERS, BLOCK = 64, 4, 4, 64


@cache
def text() -> str:
    return (paths.ROOT / DATA).read_text(encoding="utf-8")


@cache
def chars() -> tuple:
    """The characters of Tiny Shakespeare, sorted: character i has id i."""
    return tuple(sorted(set(text())))


def char_ids(text_: str) -> list:
    index = {ch: i for i, ch in enumerate(chars())}
    return [index[ch] for ch in text_]


def gpt_picture(labels: dict) -> str:
    """The GPT from top to bottom: the ids (data), the residual stream through the embeddings and the blocks (hidden),
    and the logits (pred), each box with the shape of what comes out of it."""
    W, BW, BH, GAP = 340, 290, 34, 16
    x0 = (W - BW) / 2
    steps = [(labels["ids"], "(B, n)", "data"), (labels["embed"], "(B, n, d)", "hidden"), (labels["block1"], "(B, n, d)", "hidden"),
             ("…", "", "hidden"), (labels["blockL"], "(B, n, d)", "hidden"), (labels["norm"], "(B, n, d)", "hidden"),
             (labels["head"], "(B, n, vocab)", "pred")]
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    y = 30
    for k, (words, shape, role) in enumerate(steps):
        if words == "…":
            p.append(f'<text x="{W / 2}" y="{y + 16}" style="font-size:16px;fill:var(--ea-{role}-text);text-anchor:middle">⋮</text>')
            y += 26
            continue
        p.append(f'<rect x="{x0}" y="{y}" width="{BW}" height="{BH}" rx="6" style="fill:var(--ea-{role}-wash);stroke:var(--ea-{role})"/>'
                 f'<text x="{x0 + 12}" y="{y + BH / 2 + 4.5}" style="font-size:13px;fill:var(--ea-ink)">{html.escape(words)}</text>'
                 f'<text x="{x0 + BW - 12}" y="{y + BH / 2 + 4.5}" style="font-size:12px;fill:var(--ea-{role}-text);text-anchor:end;font-family:var(--ea-font-code)">{html.escape(shape)}</text>')
        y += BH
        if k < len(steps) - 1:
            cx = W / 2
            p.append(f'<path d="M{cx} {y + 2} V{y + GAP - 2} M{cx - 3.5} {y + GAP - 6} L{cx} {y + GAP - 2} L{cx + 3.5} {y + GAP - 6}" '
                     f'style="fill:none;stroke:var(--ea-ink-3);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"/>')
            y += GAP
    H = y + 6
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


def grouped(n: int, lang: str) -> str:
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


def number(x: float, digits: int, lang: str) -> str:
    text_ = f"{x:.{digits}f}"
    return text_.replace(".", ",") if lang == "nl" else text_


# The lesson is done when these are correct. "logit-shape" is the choice question in the lesson.
REQUIRED = ["logit-shape", "gpt", "gpt-examples"]

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
_CALL = "`{call}` gives `{got}`, and it should give `{want}`"
_CALL_NL = "`{call}` geeft `{got}`, en dat moet `{want}` zijn"


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


# --- exercise 2: GPT ----------------------------------------------------------------------------------------------------------

_G_HEAD = ("import torch\nfrom torch import nn\nfrom llm.block import Block\n\n\nclass GPT(nn.Module):\n"
           "    def __init__(self, vocab, d, h, n_layers, block):\n        super().__init__()\n")
_G_INIT = ("        self.tok = nn.Embedding(vocab, d)\n        self.pos = nn.Embedding(block, d)\n"
           "        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_layers)])\n"
           "        self.norm = nn.LayerNorm(d)\n        self.head = nn.Linear(d, vocab)\n")
_G_FORWARD = ("        n = ids.shape[1]\n        x = self.tok(ids) + self.pos(torch.arange(n))\n        for b in self.blocks:\n"
              "            x = b(x)\n        return self.head(self.norm(x))\n")
_G_START = (_G_HEAD + "        # in the order the data passes them: tok and pos, the tables of tokens and places;\n"
            "        # blocks, n_layers of your Block in an nn.ModuleList; norm; head, from d to vocab\n        ...\n\n"
            "    def forward(self, ids):\n        n = ids.shape[1]\n"
            "        # the tokens and their places, through every block, then norm and head: the logits\n        ...\n")


def _g(init: str = _G_INIT, forward: str = _G_FORWARD) -> str:
    return _G_HEAD + init + "\n    def forward(self, ids):\n" + forward


_G_SCENARIOS = [
    # 2 texts of 3 tokens, a vocabulary of 10, d = 8 in 2 heads, 2 blocks: the shape, and 4 logits of the last token of text 2
    "import torch\n\ntorch.manual_seed(0)\nmodel = GPT(10, 8, 2, 2, 6)\nids = torch.tensor([[1, 2, 3], [4, 5, 6]])\nlogits = model(ids)\n[list(logits.shape), logits[1, 2, :4].tolist()]",
    # one text of 4 tokens, as long as the table of places, one block
    "import torch\n\ntorch.manual_seed(1)\nmodel = GPT(7, 6, 3, 1, 4)\nids = torch.tensor([[0, 1, 2, 3]])\nmodel(ids)[0, 3, :4].tolist()",
    # how many tensors of parameters PyTorch sees in a GPT of 3 blocks
    "import torch\n\ntorch.manual_seed(2)\nmodel = GPT(5, 4, 1, 3, 8)\nlen(list(model.parameters()))",
]


def _g_order(got, text) -> bool:
    return _reordered(text, ("tok", "pos", "blocks", "norm", "head"), "gpt", _g())


# --- exercise 3: gpt_examples -------------------------------------------------------------------------------------------------

_X_HEAD = "def gpt_examples(ids, block):\n    X = []\n    Y = []\n"
_X_LOOP = "    for i in range(0, len(ids) - block, block):\n        X.append(ids[i:i + block])\n        Y.append(ids[i + 1:i + block + 1])\n"
_X_START = (_X_HEAD + "    # windows of block ids, side by side without overlap, as long as one more id follows\n    for i in range(...):\n"
            "        X.append(...)\n        # the same window, one id further: the next id for every place\n        Y.append(...)\n    return X, Y\n")


def _x(loop: str = _X_LOOP) -> str:
    return _X_HEAD + loop + "    return X, Y\n"


EXERCISES = {
    "gpt": ClassBox(
        name="gpt",
        start=_G_START,
        solution=_g(),
        scenarios=_G_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("llm", "gpt"),
        first=[_DOTS],
        mistakes=[
            Mistake("places", error=lambda e: isinstance(e, RuntimeError) and "must match the size" in str(e),
                    en="The sizes do not fit: the embeddings of the tokens and those of the places have different shapes. Look up exactly the places of this text, `self.pos(torch.arange(n))` with `n = ids.shape[1]`, the number of tokens, in tables of `d` numbers per row: `nn.Embedding(vocab, d)` for the tokens and `nn.Embedding(block, d)` for the places.",
                    nl="De maten passen niet: de embeddings van de tokens en die van de plekken hebben een andere shape. Zoek precies de plekken van deze tekst op, `self.pos(torch.arange(n))` met `n = ids.shape[1]`, het aantal tokens, in tabellen met `d` getallen per rij: `nn.Embedding(vocab, d)` voor de tokens en `nn.Embedding(block, d)` voor de plekken."),
            Mistake("norm-size", error=lambda e: isinstance(e, RuntimeError) and "normalized_shape" in str(e),
                    en="The layer normalization gets vectors of another length than it was made for. `nn.LayerNorm(d)` normalizes the d numbers of the residual stream, so it comes before the output head: `self.head(self.norm(x))`.",
                    nl="De layer normalization krijgt vectoren van een andere lengte dan waarvoor hij gemaakt is. `nn.LayerNorm(d)` normaliseert de d getallen van de residual stream, dus hij komt vóór de output head: `self.head(self.norm(x))`."),
            Mistake("no-places", wrong=_g(forward=_G_FORWARD.replace(" + self.pos(torch.arange(n))", "")),
                    en=_SCENARIO + ": your GPT does not know where its tokens stand, so attention sees every text as a bag of tokens. Add the places: `self.tok(ids) + self.pos(torch.arange(n))`.",
                    nl=_SCENARIO_NL + ": je GPT weet niet waar zijn tokens staan, dus attention ziet elke tekst als een zak tokens. Tel de plekken erbij op: `self.tok(ids) + self.pos(torch.arange(n))`."),
            Mistake("no-norm", wrong=_g(forward=_G_FORWARD.replace("self.head(self.norm(x))", "self.head(x)")),
                    en=_SCENARIO + ": the residual stream goes into the output head without the last layer normalization. Every pre-norm block normalizes only a copy, so the stream itself still needs it: `self.head(self.norm(x))`.",
                    nl=_SCENARIO_NL + ": de residual stream gaat de output head in zonder de laatste layer normalization. Elk pre-norm block normaliseert alleen een kopie, dus de stream zelf heeft hem nog nodig: `self.head(self.norm(x))`."),
            Mistake("no-bias", wrong=_g(init=_G_INIT.replace("nn.Linear(d, vocab)", "nn.Linear(d, vocab, bias=False)")),
                    en=_SCENARIO + ": your output head has no bias terms, and in the formula the logits are LN(X) W + b. Leave out `bias=False`: the head needs a bias term for every token.",
                    nl=_SCENARIO_NL + ": je output head heeft geen bias terms, en in de formule zijn de logits LN(X) W + b. Laat `bias=False` weg: de head heeft een bias term voor elk token nodig."),
            Mistake("no-blocks", wrong=_g(forward=_G_FORWARD.replace("        for b in self.blocks:\n            x = b(x)\n", "")),
                    en=_SCENARIO + ": the tokens never go through your blocks, so no token sees another. Send `x` through every block in `self.blocks`, one after the other: `x = b(x)`.",
                    nl=_SCENARIO_NL + ": de tokens gaan nooit door je blocks, dus geen token ziet een ander. Stuur `x` door elk block in `self.blocks`, na elkaar: `x = b(x)`."),
            Mistake("plain-list", wrong=_g(init=_G_INIT.replace("nn.ModuleList([Block(d, h) for _ in range(n_layers)])", "[Block(d, h) for _ in range(n_layers)]")),
                    en=_SCENARIO + ": PyTorch does not see the parameters of your blocks, as they sit in a plain list; an optimizer would never train them. Put them in an `nn.ModuleList`, as in [Project: make a deep network train](/course/open/nn-deep-net-project?lang=en).",
                    nl=_SCENARIO_NL + ": PyTorch ziet de parameters van je blocks niet, omdat ze in een gewone lijst staan; een optimizer zou ze nooit trainen. Zet ze in een `nn.ModuleList`, zoals in [Project: een diep netwerk laten trainen](/course/open/nn-deep-net-project?lang=nl)."),
            Mistake("one-block", wrong=_g(init=_G_INIT.replace("nn.ModuleList([Block(d, h) for _ in range(n_layers)])", "nn.ModuleList([Block(d, h)] * n_layers)")),
                    en=_SCENARIO + ": your blocks are one and the same block, n_layers times, with the same weights. Every block needs weights of its own: make a new `Block(d, h)` for every place in the list, `[Block(d, h) for _ in range(n_layers)]`.",
                    nl=_SCENARIO_NL + ": je blocks zijn één en hetzelfde block, n_layers keer, met dezelfde weights. Elk block heeft eigen weights nodig: maak een nieuw `Block(d, h)` voor elke plek in de lijst, `[Block(d, h) for _ in range(n_layers)]`."),
            Mistake("residual-twice", wrong=_g(forward=_G_FORWARD.replace("x = b(x)", "x = x + b(x)")),
                    en=_SCENARIO + ": your block already adds its result to the residual stream, so `x + b(x)` counts the stream twice. Let every block give the new stream: `x = b(x)`.",
                    nl=_SCENARIO_NL + ": je block telt zijn uitkomst al op bij de residual stream, dus `x + b(x)` telt de stream twee keer. Laat elk block de nieuwe stream geven: `x = b(x)`."),
            Mistake("order", matches=_g_order,
                    en=_SCENARIO + ": the right shape, other numbers. Make the layers in the order the data passes them, `tok`, `pos`, `blocks`, `norm`, `head`, so that they get the same random starting values as in the course's solution; the rest of your class is right.",
                    nl=_SCENARIO_NL + ": de goede shape, andere getallen. Maak de layers in de volgorde waarin de data erdoor gaat, `tok`, `pos`, `blocks`, `norm`, `head`, zodat ze dezelfde willekeurige startwaarden krijgen als in de oplossing van de course; de rest van je class klopt."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the five parts in `__init__`, and in `forward` the embeddings of tokens and places, every block, and then `norm` and `head`.",
                      "nl": "Vervang de twee `...`: de vijf onderdelen in `__init__`, en in `forward` de embeddings van tokens en plekken, elk block, en dan `norm` en `head`."},
            "wrong": {"en": _SCENARIO + ". The five parts, then the recipe: tokens plus places, every block in turn, layer normalization, output head?",
                      "nl": _SCENARIO_NL + ". De vijf onderdelen, en dan het recept: tokens plus plekken, elk block om de beurt, layer normalization, output head?"},
            "almost": {"en": "Right for the first example. But " + _SCENARIO[0].lower() + _SCENARIO[1:] + ". Does your class use `vocab`, `d`, `h`, `n_layers`, `block` and `n`, and no fixed numbers?",
                       "nl": "Goed voor het eerste voorbeeld. Maar " + _SCENARIO_NL[0].lower() + _SCENARIO_NL[1:] + ". Gebruikt je class `vocab`, `d`, `h`, `n_layers`, `block` en `n`, en geen vaste getallen?"},
            "correct": {"en": "Your GPT gives one logit for every token of the vocabulary, at every place of every text: shape (B, n, vocab). It is in your project now, as `{saved}`.",
                        "nl": "Je GPT geeft één logit voor elk token van de vocabulary, op elke plek van elke tekst: shape (B, n, vocab). Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `GPT` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `GPT` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "gpt-examples": FunctionBox(
        name="gpt-examples",
        function="gpt_examples",
        start=_X_START,
        solution=_x(),
        visible=[([5, 6, 7, 8, 9, 10, 11], 3)],
        hidden=[([1, 2, 3], 2), ([1, 2], 2), (list(range(10)), 4), (list(range(9)), 1), (list(range(13)), 4)],
        python_values=True,
        save_as=("llm", "gpt_data"),
        first=[_DOTS],
        mistakes=[
            Mistake("not-shifted", wrong=_x(loop="    for i in range(0, len(ids) - block, block):\n        X.append(ids[i:i + block])\n        Y.append(ids[i:i + block])\n"),
                    en=_CALL + ": your targets are the windows themselves, so every place would learn to copy its own token. The target of a place is the id after it: the same window, one further, `ids[i + 1:i + block + 1]`.",
                    nl=_CALL_NL + ": je targets zijn de vensters zelf, dus elke plek zou leren zijn eigen token over te nemen. Het target van een plek is het id erna: hetzelfde venster, één verder, `ids[i + 1:i + block + 1]`."),
            Mistake("next-only", wrong=_x(loop="    for i in range(0, len(ids) - block, block):\n        X.append(ids[i:i + block])\n        Y.append(ids[i + block])\n"),
                    en=_CALL + ": every window gets one target, the id after the window, as in your `contexts`. A GPT predicts at every place at once: the target is the whole window, one further, `ids[i + 1:i + block + 1]`.",
                    nl=_CALL_NL + ": elk venster krijgt één target, het id na het venster, zoals in je `contexts`. Een GPT voorspelt op elke plek tegelijk: het target is het hele venster, één verder, `ids[i + 1:i + block + 1]`."),
            Mistake("overlap", wrong=_x(loop="    for i in range(len(ids) - block):\n        X.append(ids[i:i + block])\n        Y.append(ids[i + 1:i + block + 1])\n"),
                    en=_CALL + ": your windows start at every id, so they overlap. Every window already gives `block` examples, one at each place, so overlapping windows repeat the same targets. Let them stand side by side: steps of `block`, `range(0, len(ids) - block, block)`.",
                    nl=_CALL_NL + ": je vensters beginnen bij elk id, dus ze overlappen. Elk venster geeft al `block` voorbeelden, één op elke plek, dus overlappende vensters herhalen dezelfde targets. Zet ze naast elkaar: stappen van `block`, `range(0, len(ids) - block, block)`."),
            Mistake("one-short", wrong=_x(loop="    for i in range(0, len(ids) - block + 1, block):\n        X.append(ids[i:i + block])\n        Y.append(ids[i + 1:i + block + 1])\n"),
                    en=_CALL + ": your last window has no full target, as the text ends right after it. A window can only start where one more id follows its end: `range(0, len(ids) - block, block)`.",
                    nl=_CALL_NL + ": je laatste venster heeft geen volledig target, want de tekst houdt er direct na op. Een venster kan alleen beginnen waar er na zijn eind nog één id volgt: `range(0, len(ids) - block, block)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...`: the start, end and step of the windows; then a window of `block` ids, and the same window one further.",
                      "nl": "Vervang de drie `...`: het begin, eind en de stap van de vensters; dan een venster van `block` ids, en hetzelfde venster één verder."},
            "wrong": {"en": _CALL + ". Windows of `block` ids side by side, as long as one more id follows, and as target every window one id further: step 4 of the recipe?",
                      "nl": _CALL_NL + ". Vensters van `block` ids naast elkaar, zolang er nog één id volgt, en als target elk venster één id verder: stap 4 van het recept?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use `block` and `len(ids)`, and no fixed numbers?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie `block` en `len(ids)`, en geen vaste getallen?"},
            "correct": {"en": "Your function makes the examples of a GPT: every window of `block` ids, with the same window one further as targets, the next id for every place. It is in your project now, as `{saved}`.",
                        "nl": "Je functie maakt de voorbeelden van een GPT: elk venster van `block` ids, met hetzelfde venster één verder als targets, het volgende id voor elke plek. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `gpt_examples` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `gpt_examples` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}


# --- the demo: the learner's GPT for Tiny Shakespeare, before any training -----------------------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "llm" / f"{module}.py").exists()


def demo():
    """For the demo: None when gpt.py or gpt_data.py is not in the project yet; "gpt" when they stop with an error;
    "slow" when it takes too long; else a dict with the characters, the number of windows, the parameters of the GPT,
    the shape of the logits of 8 examples, its loss on them before any training, and the modules the course's versions
    stood in for."""
    if not (_saved("gpt") and _saved("gpt_data")):
        return None
    run = run_code("from llm.gpt import GPT\nfrom llm.gpt_data import gpt_examples")
    if run.error is not None:
        return "gpt"
    GPT, gpt_examples = run.names["GPT"], run.names["gpt_examples"]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]

    def work():
        ids = char_ids(text())
        X, Y = gpt_examples(ids, BLOCK)
        torch.manual_seed(0)
        model = GPT(len(chars()), D, H, LAYERS, BLOCK)
        with torch.no_grad():
            logits = model(torch.tensor(X[:8]))
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), torch.tensor(Y[:8]).reshape(-1))
        return {"chars": len(chars()), "windows": len(X), "params": int(sum(p.numel() for p in model.parameters())),
                "shape": list(logits.shape), "loss": float(loss)}

    try:
        out = with_time_limit(work, 60)
    except Timeout:
        return "slow"
    except Exception:
        return "gpt"
    out["stand_ins"] = stand_ins
    out["guess_loss"] = math.log(out["chars"])
    return out


# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("gpt", _G_START, "empty", None),
    ("gpt", _g(), "correct", None),
    ("gpt", _g(init=_G_INIT.replace("nn.ModuleList([Block(d, h) for _ in range(n_layers)])",
                                    "nn.ModuleList()\n        for _ in range(n_layers):\n            self.blocks.append(Block(d, h))"),
               forward=("        n = ids.shape[1]\n        places = torch.arange(n)\n        x = self.tok(ids) + self.pos(places)\n"
                        "        for block in self.blocks:\n            x = block(x)\n        x = self.norm(x)\n        return self.head(x)\n")), "correct", None),
    ("gpt", _g(forward=_G_FORWARD.replace("torch.arange(n)", "torch.arange(6)")), "error", "places"),
    ("gpt", _g(forward=_G_FORWARD.replace(" + self.pos(torch.arange(n))", "")), "wrong", "no-places"),
    ("gpt", _g(forward=_G_FORWARD.replace("self.head(self.norm(x))", "self.head(x)")), "wrong", "no-norm"),
    ("gpt", _g(forward=_G_FORWARD.replace("        for b in self.blocks:\n            x = b(x)\n", "")), "wrong", "no-blocks"),
    ("gpt", _g(init=_G_INIT.replace("nn.Linear(d, vocab)", "nn.Linear(d, vocab, bias=False)")), "wrong", "no-bias"),
    ("gpt", _g(init=_G_INIT.replace("nn.ModuleList([Block(d, h) for _ in range(n_layers)])", "[Block(d, h) for _ in range(n_layers)]")), "wrong", "plain-list"),
    ("gpt", _g(init=_G_INIT.replace("nn.ModuleList([Block(d, h) for _ in range(n_layers)])", "nn.ModuleList([Block(d, h)] * n_layers)")), "wrong", "one-block"),
    ("gpt", _g(init=("        self.pos = nn.Embedding(block, d)\n        self.tok = nn.Embedding(vocab, d)\n"
                     "        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_layers)])\n"
                     "        self.norm = nn.LayerNorm(d)\n        self.head = nn.Linear(d, vocab)\n")), "wrong", "order"),
    ("gpt", _g(init=_G_INIT.replace("range(n_layers)", "range(2)")), "almost", None),
    ("gpt", _g(forward=_G_FORWARD.replace("x = b(x)", "x = x + b(x)")), "wrong", "residual-twice"),
    ("gpt", _g(forward=_G_FORWARD.replace("self.head(self.norm(x))", "self.norm(self.head(x))")), "error", "norm-size"),
    ("gpt", _g(init=_G_INIT.replace("nn.Embedding(block, d)", "nn.Embedding(block, vocab)")), "error", "places"),
    ("gpt-examples", _X_START, "empty", None),
    ("gpt-examples", _x(), "correct", None),
    ("gpt-examples", _x(loop="    i = 0\n    while i + block < len(ids):\n        X.append(ids[i:i + block])\n        Y.append(ids[i + 1:i + block + 1])\n        i = i + block\n"), "correct", None),
    ("gpt-examples", _x(loop="    for i in range(0, len(ids) - block, block):\n        X.append(ids[i:i + block])\n        Y.append(ids[i:i + block])\n"), "wrong", "not-shifted"),
    ("gpt-examples", _x(loop="    for i in range(0, len(ids) - block, block):\n        X.append(ids[i:i + block])\n        Y.append(ids[i + block])\n"), "wrong", "next-only"),
    ("gpt-examples", _x(loop="    for i in range(len(ids) - block):\n        X.append(ids[i:i + block])\n        Y.append(ids[i + 1:i + block + 1])\n"), "wrong", "overlap"),
    ("gpt-examples", _x(loop="    for i in range(0, len(ids) - block + 1, block):\n        X.append(ids[i:i + block])\n        Y.append(ids[i + 1:i + block + 1])\n"), "wrong", "one-short"),
    ("gpt-examples", _x(loop="    for i in range(0, len(ids) - 3, 3):\n        X.append(ids[i:i + 3])\n        Y.append(ids[i + 1:i + 4])\n"), "almost", None),
]
