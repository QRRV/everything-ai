# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson attn-positions: sinusoids, the fixed positional encoding of the original transformer; and
TokensAndPositions, a token embedding plus a learned position embedding, as GPT-2 has."""
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import chart
from everything_ai.check import ClassBox, FunctionBox, Mistake, run_code

N, D = 64, 16
SHOWN = [0, 4, 8]   # the columns in the picture: fast, middle and slow


def _sinusoids(n: int, d: int) -> np.ndarray:
    pe = np.zeros((n, d))
    for t in range(n):
        for i in range(0, d, 2):
            angle = t / 10000 ** (i / d)
            pe[t, i] = math.sin(angle)
            pe[t, i + 1] = math.cos(angle)
    return pe


def waves_picture(labels: dict) -> str:
    """Three columns of the sinusoidal encoding for 64 positions and d = 16: each a sine wave, slower to the right."""
    pe = _sinusoids(N, D)
    panel = chart.Panel((0, N - 1), (-1.25, 1.6), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    styles = [(2.2, False), (2.2, True), (1.3, False)]
    for col, (width, dashed) in zip(SHOWN, styles):
        panel.line(list(range(N)), pe[:, col].tolist(), "hidden", width, dashed)
    panel.text(1, 1.32, labels["fast"], "hidden", "start")
    panel.text(N - 2, 1.32, labels["slow"], "hidden", "end")
    svg = chart.figure([panel], alt=labels["alt"], width=330, height=250)
    return chart.with_decimal_comma(svg) if labels.get("decimal") == "," else svg


# The lesson is done when these are correct. "too-long" is the choice question in the lesson.
REQUIRED = ["too-long", "sinusoids", "tokens-and-positions"]

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


def _grid(v):
    try:
        a = np.asarray(v.detach() if isinstance(v, torch.Tensor) else v, dtype=float)
    except (TypeError, ValueError, AttributeError):
        return repr(v)
    if np.all(np.isfinite(a)) and np.all(a == np.round(a)):
        return repr(a.astype(int).tolist())
    return repr((np.round(a, 4) + 0.0).tolist())


def _same(got, want) -> bool:
    try:
        g = np.asarray(got.detach() if isinstance(got, torch.Tensor) else got, dtype=float)
        w = np.asarray(want.detach() if isinstance(want, torch.Tensor) else want, dtype=float)
    except (TypeError, ValueError, AttributeError):
        return False
    return g.shape == w.shape and bool(np.allclose(g, w, rtol=1e-4, atol=1e-4))


def _close(got, want) -> bool:
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    if isinstance(want, bool) or isinstance(got, bool):
        return got == want
    try:
        return bool(np.allclose(float(got), float(want), rtol=1e-4, atol=1e-4))
    except (TypeError, ValueError):
        return got == want


# --- exercise 2: sinusoids ------------------------------------------------------------------------------------------------

_S_HEAD = ("import math\nimport numpy as np\n\n\ndef sinusoids(n, d):\n    pe = np.zeros((n, d))\n    for t in range(n):\n"
           "        for i in range(0, d, 2):\n")
_S_ANGLE = "            angle = t / 10000 ** (i / d)\n"
_S_FILL = "            pe[t, i] = math.sin(angle)\n            pe[t, i + 1] = math.cos(angle)\n"
_S_START = (_S_HEAD + "            # the angle of place t in columns i and i + 1: slower for later columns\n            angle = ...\n"
            "            # sin in column i, cos in column i + 1\n            ...\n    return pe\n")


def _s(angle: str = _S_ANGLE, fill: str = _S_FILL) -> str:
    return _S_HEAD + angle + fill + "    return pe\n"


# --- exercise 3: TokensAndPositions ---------------------------------------------------------------------------------------

_T_HEAD = ("import torch\nfrom torch import nn\n\n\nclass TokensAndPositions(nn.Module):\n    def __init__(self, vocab, d, block):\n"
           "        super().__init__()\n")
_T_INIT = "        self.tok = nn.Embedding(vocab, d)\n        self.pos = nn.Embedding(block, d)\n"
_T_FORWARD = "        return self.tok(ids) + self.pos(torch.arange(len(ids)))\n"
_T_START = (_T_HEAD + "        # an embedding for every token of the vocabulary, then a learned one for every place up to block\n        ...\n\n"
            "    def forward(self, ids):\n        # every token: the embedding of the token plus the embedding of its place, 0, 1, 2, ...\n        ...\n")


def _t(init: str = _T_INIT, forward: str = _T_FORWARD) -> str:
    return _T_HEAD + init + "\n    def forward(self, ids):\n" + forward


_T_SCENARIOS = [
    # the same token, 3, at places 0 and 2; another seed than the worked example, so the tables differ from its own
    "import torch\n\ntorch.manual_seed(2)\nembed = TokensAndPositions(10, 4, 8)\nx = embed(torch.tensor([3, 1, 3]))\n[list(x.shape), x[0].tolist(), x[2].tolist()]",
    # another size, a text as long as the block, and more places than tokens in the vocabulary: a table of places
    # with `vocab` rows stops at place 3
    "import torch\n\ntorch.manual_seed(1)\nembed = TokensAndPositions(3, 2, 6)\nx = embed(torch.tensor([0, 1, 2, 2, 1, 0]))\nx[5].tolist()",
]


def _t_order(got, text) -> bool:
    """The two tables are made in the order pos, tok, and that is all: made as tok, pos, the class gives what the
    course's solution gives in every scenario."""
    lines = text.split("\n")
    at = [next((i for i, line in enumerate(lines) if line.strip().startswith(f"self.{name} =")), -1) for name in ("tok", "pos")]
    if min(at) < 0 or at == sorted(at):
        return False
    made = [lines[i] for i in at]
    for i, line in zip(sorted(at), made):
        lines[i] = line
    fixed = "\n".join(lines)
    for scenario in _T_SCENARIOS:
        mine, want = run_code(fixed + "\n\n" + scenario), run_code(_t() + "\n\n" + scenario)
        if mine.error is not None or not _close(mine.value, want.value):
            return False
    return True


EXERCISES = {
    "sinusoids": FunctionBox(
        name="sinusoids",
        function="sinusoids",
        start=_S_START,
        solution=_s(),
        visible=[(3, 4)],
        hidden=[(1, 2), (5, 6), (4, 8)],
        compare=_same,
        show=_grid,
        first=[_DOTS],
        mistakes=[
            Mistake("swapped", wrong=_s(fill="            pe[t, i] = math.cos(angle)\n            pe[t, i + 1] = math.sin(angle)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your sin and cos have swapped places. The sine goes in column `i`, which starts at 0 for place 0, and the cosine in column `i + 1`, which starts at 1.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je sin en cos staan op elkaars plek. De sinus hoort in kolom `i`, die bij plek 0 op 0 begint, en de cosinus in kolom `i + 1`, die op 1 begint."),
            Mistake("one-speed", wrong=_s(angle="            angle = t\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every pair of columns turns at the same speed, so they all repeat each other. Divide by `10000 ** (i / d)`: the later the column, the slower the wave.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elk paar kolommen draait even snel, dus ze herhalen elkaar allemaal. Deel door `10000 ** (i / d)`: hoe later de kolom, hoe trager de golf."),
            Mistake("two-j", wrong=_s(angle="            angle = t / 10000 ** (2 * i / d)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your later columns turn too slowly. The `i` of the loop already counts the columns 0, 2, 4, ...: it is the 2j of the formula. Divide by `10000 ** (i / d)`, without the extra 2.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je latere kolommen draaien te traag. De `i` van de loop telt de kolommen 0, 2, 4, ... al: hij is de 2j van de formule. Deel door `10000 ** (i / d)`, zonder de extra 2."),
            Mistake("times", wrong=_s(angle="            angle = t * 10000 ** (i / d)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you multiply by `10000 ** (i / d)`, so later columns turn faster instead of slower. Divide: `t / 10000 ** (i / d)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je vermenigvuldigt met `10000 ** (i / d)`, dus latere kolommen draaien sneller in plaats van trager. Deel: `t / 10000 ** (i / d)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the angle of place `t` for columns `i` and `i + 1`, and the sine and cosine of it in those columns.",
                      "nl": "Vervang de twee `...`: de hoek van plek `t` voor kolom `i` en `i + 1`, en daarvan de sinus en de cosinus in die kolommen."},
            "arguments": {"en": "The check calls `{call}`, with a number of positions and a number of columns, and the `def` line of your function has another number of names between its parentheses. Keep `(n, d)`.",
                          "nl": "De check roept `{call}` aan, met een aantal posities en een aantal kolommen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(n, d)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your lines with steps 1 and 2 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je regels met stap 1 en 2 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use `n` and `d`, and no fixed numbers?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie `n` en `d`, en geen vaste getallen?"},
            "correct": {"en": "Your function gives every place its own row: place 0 is 0, 1, 0, 1, ..., and every later one a different mix of fast and slow waves, for any number of places.",
                        "nl": "Je functie geeft elke plek een eigen rij: plek 0 is 0, 1, 0, 1, ..., en elke latere een andere mengeling van snelle en trage golven, voor elk aantal plekken."},
        },
    ),
    "tokens-and-positions": ClassBox(
        name="tokens-and-positions",
        start=_T_START,
        solution=_t(),
        scenarios=_T_SCENARIOS,
        compare=_close,
        digits=4,
        first=[_DOTS],
        mistakes=[
            Mistake("no-places", wrong=_t(forward="        return self.tok(ids) + self.pos(ids)\n"),
                    en=_SCENARIO + ": you look up the token ids in the table of places, so the same token gets the same vector wherever it stands. Look up the places themselves, 0, 1, 2, ...: `self.pos(torch.arange(len(ids)))`.",
                    nl=_SCENARIO_NL + ": je zoekt de token-ids op in de tabel van plekken, dus hetzelfde token krijgt dezelfde vector, waar het ook staat. Zoek de plekken zelf op, 0, 1, 2, ...: `self.pos(torch.arange(len(ids)))`."),
            Mistake("tokens-only", wrong=_t(forward="        return self.tok(ids)\n"),
                    en=_SCENARIO + ": only the token embeddings, so token 3 at place 0 and at place 2 gets the same vector. Add the embedding of every place: `+ self.pos(torch.arange(len(ids)))`.",
                    nl=_SCENARIO_NL + ": alleen de token-embeddings, dus token 3 op plek 0 en op plek 2 krijgt dezelfde vector. Tel de embedding van elke plek erbij op: `+ self.pos(torch.arange(len(ids)))`."),
            Mistake("side-by-side", wrong=_t(forward="        return torch.cat([self.tok(ids), self.pos(torch.arange(len(ids)))], dim=1)\n"),
                    en=_SCENARIO + ": you put the two embeddings side by side, so every token gets twice `d` numbers. The place is added to the token, number by number: `self.tok(ids) + self.pos(...)`, d numbers.",
                    nl=_SCENARIO_NL + ": je zet de twee embeddings naast elkaar, dus elk token krijgt twee keer `d` getallen. De plek wordt bij het token opgeteld, getal voor getal: `self.tok(ids) + self.pos(...)`, d getallen."),
            Mistake("range", error=lambda e: isinstance(e, TypeError) and re.search(r"must be Tensor, not (range|list)", str(e)) is not None,
                    en="An embedding layer looks up a tensor of ids, and a `range` or a list is not a tensor. Make the places with `torch.arange(len(ids))`: the same numbers, 0, 1, 2, ..., as a tensor.",
                    nl="Een embedding layer zoekt een tensor van ids op, en een `range` of een lijst is geen tensor. Maak de plekken met `torch.arange(len(ids))`: dezelfde getallen, 0, 1, 2, ..., als tensor."),
            Mistake("too-few-rows", error=lambda e: isinstance(e, IndexError) and "index out of range in self" in str(e),
                    en="One of your tables is asked for a row it does not have. `self.tok` needs a row for every token of the vocabulary, `vocab` rows, and `self.pos` a row for every place up to `block`, `block` rows.",
                    nl="Een van je tabellen moet een rij geven die hij niet heeft. `self.tok` heeft een rij nodig voor elk token van de vocabulary, `vocab` rijen, en `self.pos` een rij voor elke plek tot aan `block`, `block` rijen."),
            Mistake("rows-differ", error=lambda e: isinstance(e, RuntimeError) and "must match the size" in str(e) and "dimension 0" in str(e),
                    en="The token embeddings and the place embeddings have another number of rows, so Python cannot add them. Look up one place for every token: `torch.arange(len(ids))`.",
                    nl="De token-embeddings en de plek-embeddings hebben een ander aantal rijen, dus Python kan ze niet optellen. Zoek één plek op voor elk token: `torch.arange(len(ids))`."),
            Mistake("width-differs", error=lambda e: isinstance(e, RuntimeError) and "must match the size" in str(e) and "dimension 1" in str(e),
                    en="The token embeddings and the place embeddings have another number of numbers per token, so Python cannot add them. Both tables need rows of `d` numbers: `nn.Embedding(vocab, d)` and `nn.Embedding(block, d)`.",
                    nl="De token-embeddings en de plek-embeddings hebben een ander aantal getallen per token, dus Python kan ze niet optellen. Beide tabellen hebben rijen van `d` getallen nodig: `nn.Embedding(vocab, d)` en `nn.Embedding(block, d)`."),
            Mistake("order", matches=_t_order,
                    en=_SCENARIO + ": the right shape, other numbers. Make the two tables in the order tok, pos, so that they get the same random starting values as in the course's solution.",
                    nl=_SCENARIO_NL + ": de goede shape, andere getallen. Maak de twee tabellen in de volgorde tok, pos, zodat ze dezelfde willekeurige startwaarden krijgen als in de oplossing van de course."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the two tables in `__init__`, and in `forward` the token embeddings plus the place embeddings.",
                      "nl": "Vervang de twee `...`: de twee tabellen in `__init__`, en in `forward` de token-embeddings plus de plek-embeddings."},
            "wrong": {"en": _SCENARIO + ". Two `nn.Embedding` tables, of the vocabulary and of the places, and their embeddings added up: steps 3 and 4 of the recipe?",
                      "nl": _SCENARIO_NL + ". Twee `nn.Embedding`-tabellen, van de vocabulary en van de plekken, en hun embeddings opgeteld: stap 3 en 4 van het recept?"},
            "almost": {"en": "Right for the first example. But " + _SCENARIO[0].lower() + _SCENARIO[1:] + ". Does your class use `vocab`, `d`, `block` and `len(ids)`, and no fixed numbers?",
                       "nl": "Goed voor het eerste voorbeeld. Maar " + _SCENARIO_NL[0].lower() + _SCENARIO_NL[1:] + ". Gebruikt je class `vocab`, `d`, `block` en `len(ids)`, en geen vaste getallen?"},
            "correct": {"en": "Your module gives every token the embedding of what it is plus the embedding of where it is: token 3 at place 0 and at place 2 now gets two different vectors, and attention can tell them apart.",
                        "nl": "Je module geeft elk token de embedding van wat het is plus de embedding van waar het staat: token 3 op plek 0 en op plek 2 krijgt nu twee verschillende vectoren, en attention kan ze uit elkaar houden."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("sinusoids", _S_START, "empty", None),
    ("sinusoids", _s(angle="            angle = t / math.pow(10000, i / d)\n", fill="            pe[t, i], pe[t, i + 1] = math.sin(angle), math.cos(angle)\n"), "correct", None),
    ("sinusoids", _s(fill="            pe[t, i] = math.cos(angle)\n            pe[t, i + 1] = math.sin(angle)\n"), "wrong", "swapped"),
    ("sinusoids", _s(angle="            angle = t\n"), "wrong", "one-speed"),
    ("sinusoids", _s(angle="            angle = t * 10000 ** (i / d)\n"), "wrong", "times"),
    ("sinusoids", _s(angle="            angle = t / 10000 ** (2 * i / d)\n"), "wrong", "two-j"),
    ("sinusoids", _s(angle="            angle = t / 10000 ** (i / 4)\n"), "almost", None),
    ("tokens-and-positions", _T_START, "empty", None),
    ("tokens-and-positions", _t(forward="        places = torch.arange(len(ids))\n        return self.pos(places) + self.tok(ids)\n"), "correct", None),
    ("tokens-and-positions", _t(forward="        return self.tok(ids) + self.pos(ids)\n"), "wrong", "no-places"),
    ("tokens-and-positions", _t(forward="        return self.tok(ids)\n"), "wrong", "tokens-only"),
    ("tokens-and-positions", _t(forward="        return torch.cat([self.tok(ids), self.pos(torch.arange(len(ids)))], dim=1)\n"), "wrong", "side-by-side"),
    ("tokens-and-positions", _t(init="        self.pos = nn.Embedding(block, d)\n        self.tok = nn.Embedding(vocab, d)\n"), "wrong", "order"),
    # the order and another fault: not the order message, which says the shape is right
    ("tokens-and-positions", _t(init="        self.pos = nn.Embedding(block, d)\n        self.tok = nn.Embedding(vocab, d)\n",
                                forward="        return torch.cat([self.tok(ids), self.pos(torch.arange(len(ids)))], dim=1)\n"), "wrong", None),
    ("tokens-and-positions", _t(init="        self.pos = nn.Embedding(block, d)\n        self.tok = nn.Embedding(vocab, d)\n", forward="        return self.tok(ids)\n"), "wrong", None),
    ("tokens-and-positions", _t(init="        self.tok = nn.Embedding(vocab, d)\n        self.pos = nn.Embedding(vocab, d)\n"), "error", "too-few-rows"),
    ("tokens-and-positions", _t(forward="        return self.tok(ids) + self.pos(range(len(ids)))\n"), "error", "range"),
    ("tokens-and-positions", _t(forward="        return self.tok(ids) + self.pos(torch.arange(3))\n"), "error", "rows-differ"),
    ("tokens-and-positions", _t(init="        self.tok = nn.Embedding(d, vocab)\n        self.pos = nn.Embedding(d, block)\n"), "error", "width-differs"),
]
