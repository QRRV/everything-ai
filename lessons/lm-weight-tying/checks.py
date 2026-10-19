# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson lm-weight-tying: tied_logits, the logits of an output head that is the embedding table, and
TiedLM, a small model whose head uses the weight of its embedding. And the picture: the share of the parameters of the
GPT-2 models that tying saves."""
import re

import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai.check import ClassBox, FunctionBox, Mistake, run_code

# GPT-2's four sizes: (name, d, layers). All have a vocabulary of 50,257 tokens and 1,024 places.
GPT2 = [("small", 768, 12), ("medium", 1024, 24), ("large", 1280, 36), ("XL", 1600, 48)]
VOCAB, PLACES = 50257, 1024


def gpt2_params(d: int, layers: int, tied: bool = True) -> int:
    """The parameters of a GPT-2 model: the tables of tokens and places, per block two layer normalizations, attention
    (4 d² + 4 d) and the MLP (8 d² + 5 d), the last layer normalization, and a head without bias terms when untied."""
    blocks = layers * (12 * d * d + 13 * d)
    return VOCAB * d + PLACES * d + blocks + 2 * d + (0 if tied else VOCAB * d)


def grouped(n: int, lang: str) -> str:
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


def millions(n: int, lang: str) -> str:
    return grouped(round(n / 1e6), lang)


def saved_share(d: int, layers: int) -> float:
    """The share of an untied GPT-2's parameters that tying saves: its head."""
    return VOCAB * d / gpt2_params(d, layers, tied=False)


def saved_picture(labels: dict) -> str:
    """For each size of GPT-2, the share of the untied model's parameters that tying saves, as a bar: weights a model
    learns, so the parameter colour."""
    from everything_ai import chart
    lang = labels.get("lang", "en")
    p = chart.Panel((0, 30), (0.2, 4.6), title=labels["title"], xlabel=labels["x"], ylabel="", xticks=[0, 10, 20, 30], yticks=[])
    for k, (name, d, layers) in enumerate(GPT2):
        y = 3.6 - k
        share = 100 * saved_share(d, layers)
        for _ in range(3):   # three washes on top of each other: a bar, not a band
            p.rect(0, y - 0.16, share, y + 0.16, "param")
        p.text(share + 0.6, y, f"{share:.0f}%", "ink", "start")
        p.text(0.2, y + 0.42, labels["row"].format(name=name, n=millions(gpt2_params(d, layers, tied=False), lang)), "ink-2", "start")
    return chart.figure([p], alt=labels["alt"], width=330, height=250)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}
_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}


def _close(got, want) -> bool:
    try:
        a = torch.as_tensor(got, dtype=torch.float64)
        b = torch.as_tensor(want, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError):
        return False
    return a.shape == b.shape and bool(torch.allclose(a, b, rtol=1e-5, atol=1e-7))


def _show_t(v) -> str:
    """A tensor as the list it holds, rounded to 4 decimals, for the check's messages."""
    try:
        t = torch.as_tensor(v, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError):
        return repr(v)
    return repr((torch.round(t, decimals=4) + 0.0).tolist())


# --- exercise 1: tied_logits ---------------------------------------------------------------------------------------------------

_L_HEAD = ("import torch\n\n\ndef tied_logits(h, E):\n    h, E = torch.tensor(h), torch.tensor(E)\n"
           "    # one logit for every token: the dot product of every vector in h with every row of E\n")
_L_START = _L_HEAD + "    return ...\n"


def _l(body: str = "    return h @ E.T\n") -> str:
    return _L_HEAD + body


_L_WRONG = {"turned": _l("    return E @ h.T\n")}

# --- exercise 2: TiedLM --------------------------------------------------------------------------------------------------------

_T_HEAD = ("import torch\nfrom torch import nn\n\n\nclass TiedLM(nn.Module):\n    def __init__(self, vocab, d):\n"
           "        super().__init__()\n")
_T_LAYERS = "        self.tok = nn.Embedding(vocab, d)\n        self.head = nn.Linear(d, vocab)\n"
_T_TAIL = "\n    def forward(self, ids):\n        return self.head(self.tok(ids))\n"
_T_START = (_T_HEAD + "        # the embedding tok and the head, from d to vocab, as in your GPT;\n"
            "        # then one table at both ends: the head uses the weight of the embedding\n        ...\n" + _T_TAIL)


def _t(tie: str = "        self.head.weight = self.tok.weight\n", layers: str = _T_LAYERS) -> str:
    return _T_HEAD + layers + tie + _T_TAIL


_T_SCENARIOS = [
    # the head's weight is the embedding's weight: one object
    "model = TiedLM(10, 4)\nmodel.head.weight is model.tok.weight",
    # how many tensors of parameters PyTorch sees: the table once, and the head's bias terms
    "model = TiedLM(10, 4)\nlen(list(model.parameters()))",
    # a training step changes the one table, so the head changes with the embedding
    ("import torch\n\ntorch.manual_seed(0)\nmodel = TiedLM(5, 3)\nopt = torch.optim.SGD(model.parameters(), lr=0.5)\n"
     "loss = model(torch.tensor([1, 2])).sum()\nloss.backward()\nopt.step()\n"
     "torch.equal(model.head.weight, model.tok.weight)"),
]

def _copied(got, text) -> bool:
    """A line that gives the head (or the embedding) a weight of its own, made from the other one: a new
    nn.Parameter, a clone, or the same memory through .data."""
    for line in re.findall(r"^\s*self\.(?:head|tok)\.weight\b.*=.*$", text, re.MULTILINE):
        if re.search(r"nn\.Parameter\(|\.clone\(|\.data\b|\.detach\(", line):
            return True
    return False


_T_WRONG = {
    "copy": _t("        self.head.weight = nn.Parameter(self.tok.weight.clone())\n"),
    "none": _t("        pass\n"),
}

EXERCISES = {
    "tied-logits": FunctionBox(
        name="tied-logits",
        function="tied_logits",
        start=_L_START,
        solution=_l(),
        visible=[([[1.0, 0.0, 2.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.5, -1.0, 2.0]])],
        hidden=[
            ([[1.0, 2.0], [0.0, -1.0]], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            ([[0.5, -0.5, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, -1.0]]),
            ([[3.0]], [[1.0], [-2.0], [0.5], [0.0], [4.0]]),
        ],
        compare=_close,
        show=_show_t,
        first=[_DOTS],
        mistakes=[
            Mistake("turned", wrong=_L_WRONG["turned"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: the same dot products, turned over, with a row for every token and a column for every vector of `h`. Every vector of `h` gets a row of logits, one for every token: start with `h`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dezelfde inproducten, omgedraaid, met een rij voor elk token en een kolom voor elke vector van `h`. Elke vector van `h` krijgt een rij logits, één voor elk token: begin met `h`."),
            Mistake("no-transpose", error=lambda e: isinstance(e, RuntimeError) and "cannot be multiplied" in str(e),
                    en="The shapes do not fit: `h` has shape (n, d), and `E` has a row of d numbers for every token, shape (vocab, d). A dot product with every row of `E` needs its rows as columns: `E.T`, of shape (d, vocab).",
                    nl="De shapes passen niet: `h` heeft shape (n, d), en `E` heeft een rij van d getallen voor elk token, shape (vocab, d). Een inproduct met elke rij van `E` heeft zijn rijen als kolommen nodig: `E.T`, met shape (d, vocab)."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "Replace the `...` with the logits: one row for every vector of `h`, one number for every row of `E`.",
                      "nl": "Vervang de `...` door de logits: één rij voor elke vector van `h`, één getal voor elke rij van `E`."},
            "arguments": {"en": "The check calls `{call}`, with the vectors and the table, and the `def` line of your function has another number of names between its parentheses. Keep `(h, E)`.",
                          "nl": "De check roept `{call}` aan, met de vectoren en de tabel, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(h, E)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. The logit of token v for a vector of `h` is the dot product of that vector with row v of `E`.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. De logit van token v voor een vector van `h` is het inproduct van die vector met rij v van `E`."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for more vectors in `h`, and for other sizes of `E`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt het ook voor meer vectoren in `h`, en voor andere maten van `E`?"},
            "correct": {"en": "Your tied head gives a logit for every token in all {n} cases: the dot product with that token's own embedding.",
                        "nl": "Je gekoppelde head geeft een logit voor elk token in alle {n} gevallen: het inproduct met de eigen embedding van dat token."},
        },
    ),
    "tied-lm": ClassBox(
        name="tied-lm",
        start=_T_START,
        solution=_t(),
        scenarios=_T_SCENARIOS,
        first=[_DOTS],
        mistakes=[
            Mistake("transpose", error=lambda e: isinstance(e, TypeError) and "cannot assign" in str(e),
                    en="PyTorch stops: the weight of a layer only takes a parameter, and what you assign is not one. A layer such as `self.tok` is not a parameter, and something made from one, such as `self.tok.weight.T` or a `.clone()`, is a plain tensor. No transpose is needed either: `nn.Linear(d, vocab)` keeps its weight with a row for every token, shape (vocab, d), the same as the embedding. Assign the parameter itself: `self.head.weight = self.tok.weight`.",
                    nl="PyTorch stopt: de weight van een layer neemt alleen een parameter aan, en wat je toewijst is er geen. Een layer zoals `self.tok` is geen parameter, en iets wat je van een parameter maakt, zoals `self.tok.weight.T` of een `.clone()`, is een gewone tensor. Een getransponeerde is ook niet nodig: `nn.Linear(d, vocab)` bewaart zijn weight met een rij voor elk token, shape (vocab, d), net als de embedding. Wijs de parameter zelf toe: `self.head.weight = self.tok.weight`."),
            Mistake("copy", matches=_copied,
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: your head has a parameter of its own, with the same numbers as the embedding or even the same memory, but PyTorch counts it and trains it as a second table. Make the head's weight the embedding's parameter itself: `self.head.weight = self.tok.weight`.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: je head heeft een eigen parameter, met dezelfde getallen als de embedding of zelfs hetzelfde geheugen, maar PyTorch telt hem en traint hem als een tweede tabel. Maak de weight van de head de parameter van de embedding zelf: `self.head.weight = self.tok.weight`."),
            Mistake("none", wrong=_T_WRONG["none"],
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: the head still has its own table. Tie it in `__init__`, after both are made: `self.head.weight = self.tok.weight`.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: de head heeft nog een eigen tabel. Koppel hem in `__init__`, nadat ze allebei gemaakt zijn: `self.head.weight = self.tok.weight`."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "Replace the `...` with three lines: the embedding, the head, and the line that gives the head the weight of the embedding.",
                      "nl": "Vervang de `...` door drie regels: de embedding, de head, en de regel die de head de weight van de embedding geeft."},
            "wrong": {"en": "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Is the head's weight the embedding's parameter itself, one object at both ends?",
                      "nl": "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Is de weight van de head de parameter van de embedding zelf, één object aan beide kanten?"},
            "almost": {"en": "Right for the first example: the head's weight is the embedding's. But with these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Tied, `TiedLM` holds two tensors of parameters: the table, once, and the bias terms of its head. Make the head as in your GPT, `nn.Linear(d, vocab)` with its bias terms, and tie only its weight.",
                       "nl": "Goed voor het eerste voorbeeld: de weight van de head is die van de embedding. Maar met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Gekoppeld heeft `TiedLM` twee tensors met parameters: de tabel, één keer, en de bias terms van zijn head. Maak de head zoals in je GPT, `nn.Linear(d, vocab)` met zijn bias terms, en koppel alleen zijn weight."},
            "correct": {"en": "Your `TiedLM` has one table for both ends: PyTorch counts it once, and a training step changes the embedding and the head together.",
                        "nl": "Je `TiedLM` heeft één tabel voor beide kanten: PyTorch telt hem één keer, en een trainingsstap verandert de embedding en de head samen."},
        },
    ),
}

# The lesson is done when these are correct. "rows-grad" is the choice question.
REQUIRED = ["tied-logits", "tied-lm", "rows-grad"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("tied-logits", _L_START, "empty", None),
    ("tied-logits", _l(), "correct", None),
    ("tied-logits", _l("    return torch.matmul(h, E.transpose(0, 1))\n"), "correct", None),
    ("tied-logits", _L_WRONG["turned"], "wrong", "turned"),
    ("tied-logits", _l("    return h @ E\n"), "error", "no-transpose"),
    ("tied-lm", _T_START, "empty", None),
    ("tied-lm", _t(), "correct", None),
    ("tied-lm", _t(layers="        self.head = nn.Linear(d, vocab)\n        self.tok = nn.Embedding(vocab, d)\n"), "correct", None),
    ("tied-lm", _t("        self.tok.weight = self.head.weight\n"), "correct", None),
    ("tied-lm", _t("        self.head.weight = self.tok.weight.T\n"), "error", "transpose"),
    ("tied-lm", _t("        self.head.weight = self.tok\n"), "error", "transpose"),
    ("tied-lm", _t("        self.head = self.tok\n"), "almost", None),
    ("tied-lm", _t("        self.head = nn.Linear(d, vocab, bias=False)\n        self.head.weight = self.tok.weight\n"), "almost", None),
    ("tied-lm", _T_WRONG["copy"], "wrong", "copy"),
    ("tied-lm", _t("        self.head.weight.data = self.tok.weight.data\n"), "wrong", "copy"),
    ("tied-lm", _t("        self.head.weight = nn.Parameter(self.tok.weight)\n"), "wrong", "copy"),
    ("tied-lm", _T_WRONG["none"], "wrong", "none"),
]


def counts(text) -> tuple | None:
    """The parameters of the learner's TiedLM with GPT-2's vocabulary and d = 768, and of the same model untied:
    (tied, untied), each counted once per parameter, as an optimizer sees them."""
    ran = run_code(text) if text else None
    cls = ran.names.get("TiedLM") if ran is not None and ran.error is None else None
    if cls is None:
        return None
    with torch.device("meta"):   # only the shapes: no memory for 77 million numbers
        model = cls(VOCAB, 768)
    tied = sum(p.numel() for p in model.parameters())
    return tied, tied + VOCAB * 768
