# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson attn-qkv: qkv_attention, self-attention in NumPy with given matrices for the queries, keys and
values; and SelfAttention, the same as a PyTorch module whose three matrices are learned."""
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import draw
from everything_ai.check import ClassBox, FunctionBox, Mistake, run_code

# "a red car" with made-up embeddings of 2 numbers: how much a word describes, and how much it is a thing.
SENTENCE = ["a", "red", "car"]
X = [[0.3, 0.3], [2.0, 0.0], [0.0, 2.0]]
W_Q = [[0.0, 1.0], [1.0, 0.0]]   # a query swaps the two numbers: a word that describes looks for a thing, and back
W_K = [[1.0, 0.0], [0.0, 1.0]]   # the keys are the embeddings themselves


def _softmax_rows(S: np.ndarray) -> np.ndarray:
    S = S - S.max(axis=1, keepdims=True)
    return np.exp(S) / np.sum(np.exp(S), axis=1, keepdims=True)


def weights_pictures(labels: dict):
    """The attention weights of "a red car": with the scores X Xᵀ of attn-scores, and with queries and keys."""
    x = np.array(X)
    plain = _softmax_rows(x @ x.T)
    learned = _softmax_rows((x @ np.array(W_Q)) @ (x @ np.array(W_K)).T)
    common = {"row_label": labels["rows"], "col_label": labels["cols"], "decimal": labels["decimal"]}
    return [draw.weight_grid(SENTENCE, plain, title=labels["plain"], alt=labels["plain_alt"], **common),
            draw.weight_grid(SENTENCE, learned, title=labels["learned"], alt=labels["learned_alt"], **common)]


# The lesson is done when these are correct. "value-shape" is the choice question in the lesson.
REQUIRED = ["value-shape", "qkv-attention", "self-attention-module"]

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
    """An array for the messages: nested lists rounded to 4 decimals, whole numbers without .0."""
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return repr(v)
    if np.all(np.isfinite(a)) and np.all(a == np.round(a)):
        return repr(a.astype(int).tolist())
    return repr(np.round(a, 4).tolist())


def _same_grid(got, want) -> bool:
    try:
        g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
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


def _matmul_error(e) -> bool:
    return (isinstance(e, ValueError) and "matmul" in str(e)) or (isinstance(e, RuntimeError) and "cannot be multiplied" in str(e))


# --- exercise 2: qkv_attention --------------------------------------------------------------------------------------------

_N_HEAD = ("import numpy as np\n\n\ndef qkv_attention(X, Wq, Wk, Wv):\n"
           "    X, Wq, Wk, Wv = np.array(X, dtype=float), np.array(Wq, dtype=float), np.array(Wk, dtype=float), np.array(Wv, dtype=float)\n")
_N_SOFTMAX = ("    # 3. the attention weights: the softmax of every row\n    S = S - S.max(axis=1, keepdims=True)\n"
              "    A = np.exp(S) / np.sum(np.exp(S), axis=1, keepdims=True)\n")
_N_START = (_N_HEAD + "    # 1. the queries, keys and values of all words\n    Q = ...\n    K = ...\n    V = ...\n"
            "    # 2. the score of every query with every key\n    S = ...\n" + _N_SOFTMAX
            + "    # 4. the new vectors: the values, weighted\n    return ...\n")


def _n(q: str = "X @ Wq", k: str = "X @ Wk", v: str = "X @ Wv", s: str = "Q @ K.T", out: str = "A @ V") -> str:
    return _N_HEAD + f"    Q = {q}\n    K = {k}\n    V = {v}\n    S = {s}\n" + _N_SOFTMAX + f"    return {out}\n"


_SWAP = [[0, 1], [1, 0]]
_EYE = [[1, 0], [0, 1]]

# --- exercise 3: SelfAttention --------------------------------------------------------------------------------------------

_M_HEAD = ("import torch\nfrom torch import nn\n\n\nclass SelfAttention(nn.Module):\n    def __init__(self, d, d_k):\n"
           "        super().__init__()\n")
_M_INIT = ("        self.query = nn.Linear(d, d_k, bias=False)\n        self.key = nn.Linear(d, d_k, bias=False)\n"
           "        self.value = nn.Linear(d, d_k, bias=False)\n")
_M_FORWARD = ("        q = self.query(x)\n        k = self.key(x)\n        v = self.value(x)\n"
              "        A = torch.softmax(q @ k.T, dim=1)\n        return A @ v\n")
_M_START = (_M_HEAD + "        # three learned matrices from d to d_k numbers, without bias terms: query, key and value, in that order\n"
            "        ...\n\n    def forward(self, x):\n"
            "        # x holds one embedding per row: queries, keys and values, then the weights, then the new vectors\n        ...\n")


def _m(init: str = _M_INIT, forward: str = _M_FORWARD) -> str:
    return _M_HEAD + init + "\n    def forward(self, x):\n" + forward


_M_SCENARIOS = [
    # 5 words of 4 numbers: the shape of the new vectors, and the new vector of the first word
    "import torch\n\ntorch.manual_seed(0)\nattention = SelfAttention(4, 3)\nx = torch.rand(5, 4)\nh = attention(x)\n[list(h.shape), h[0].tolist()]",
    # other sizes, and the last word
    "import torch\n\ntorch.manual_seed(1)\nattention = SelfAttention(6, 2)\nx = torch.rand(3, 6)\nh = attention(x)\n[list(h.shape), h[2].tolist()]",
    # embeddings of zeros: without bias terms, every query, key and value is 0
    "import torch\n\ntorch.manual_seed(2)\nattention = SelfAttention(3, 2)\nh = attention(torch.zeros(2, 3))\nh.tolist()",
]


def _out_of_order(got, text) -> bool:
    """The three layers are made in another order, and that is all: made as query, key, value, the class gives what
    the course's solution gives in every scenario."""
    lines = text.split("\n")
    at = [next((i for i, line in enumerate(lines) if line.strip().startswith(f"self.{name} =")), -1) for name in ("query", "key", "value")]
    if min(at) < 0 or at == sorted(at):
        return False
    made = [lines[i] for i in at]
    for i, line in zip(sorted(at), made):
        lines[i] = line
    fixed = "\n".join(lines)
    for scenario in _M_SCENARIOS:
        mine, want = run_code(fixed + "\n\n" + scenario), run_code(_m() + "\n\n" + scenario)
        if mine.error is not None or not _close(mine.value, want.value):
            return False
    return True


EXERCISES = {
    "qkv-attention": FunctionBox(
        name="qkv-attention",
        function="qkv_attention",
        start=_N_START,
        solution=_n(),
        visible=[([[0.3, 0.3], [0, 2], [0.3, 0.3], [2, 0]], _SWAP, _EYE, [[1, 0], [0, 0.5]])],
        hidden=[([[1, 0], [0, 1], [1, 1]], _EYE, _EYE, _EYE), ([[1, 2, 0], [0, 1, 1]], [[1, 0], [0, 1], [1, 1]], [[0, 1], [1, 0], [1, 0]], [[1], [2], [0]]),
                ([[3, 4]], _SWAP, _EYE, [[1, 1], [0, 1]]), ([[1, 0], [0, 1], [1, 1], [2, 0]], [[1, 1], [0, 1]], [[0, 1], [1, 1]], [[2, 0, 1], [0, 1, 1]])],
        compare=_same_grid,
        show=_grid,
        first=[_DOTS],
        mistakes=[
            Mistake("weights-first", syntax=r"^[ \t]*[QKV] = .*\bW[qkv]\s*@\s*X\b",
                    en="`{line}` puts the matrix before `X`. `X` has one row per word, so `X` comes first: `X @ Wq`, `X @ Wk` and `X @ Wv` give a query, a key and a value per word, a row each, as in recipe step 1.",
                    nl="`{line}` zet de matrix vóór `X`. `X` heeft één rij per woord, dus `X` komt eerst: `X @ Wq`, `X @ Wk` en `X @ Wv` geven per woord een query, een key en een value, elk een rij, zoals in stap 1 van het recept."),
            Mistake("sizes", error=_matmul_error,
                    en="The sizes of a matrix multiplication do not fit. Check them step by step: `X @ Wq` is (n, d) times (d, d_k), the scores `Q @ K.T` are (n, d_k) times (d_k, n), and the new vectors `A @ V` are (n, n) times the values, one row per word.",
                    nl="De maten van een matrixvermenigvuldiging passen niet. Loop ze stap voor stap na: `X @ Wq` is (n, d) keer (d, d_k), de scores `Q @ K.T` zijn (n, d_k) keer (d_k, n), en de nieuwe vectoren `A @ V` zijn (n, n) keer de values, één rij per woord."),
            Mistake("keys-first", wrong=_n(s="K @ Q.T"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your scores are the other way round. Row `i` holds what word `i` looks for, its query, against every key: `Q @ K.T`. `K @ Q.T` is its transpose.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je scores staan andersom. Rij `i` bevat waar woord `i` naar zoekt, zijn query, tegen elke key: `Q @ K.T`. `K @ Q.T` is de getransponeerde daarvan."),
            Mistake("embeddings", wrong=_n(out="A @ X"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you average the embeddings. What a word passes on is its value: `A @ V`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je middelt de embeddings. Wat een woord doorgeeft, is zijn value: `A @ V`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the five `...`: the queries, keys and values, the scores, and the new vectors.",
                      "nl": "Vervang de vijf `...`: de queries, keys en values, de scores, en de nieuwe vectoren."},
            "arguments": {"en": "The check calls `{call}`, with embeddings and three matrices, and the `def` line of your function has another number of names between its parentheses. Keep `(X, Wq, Wk, Wv)`.",
                          "nl": "De check roept `{call}` aan, met embeddings en drie matrices, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X, Wq, Wk, Wv)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your lines with steps 1, 2 and 4 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je regels met stap 1, 2 en 4 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any number of words, and for values of another length than the queries?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elk aantal woorden, en voor values met een andere lengte dan de queries?"},
            "correct": {"en": "Your function is self-attention: in \"a car is red\", \"red\" takes most from \"car\", and \"car\" from \"red\", though they are not next to each other, and every word passes on its value, with the second number halved.",
                        "nl": "Je functie is self-attention: in \"a car is red\" neemt \"red\" het meest van \"car\", en \"car\" van \"red\", al staan ze niet naast elkaar, en elk woord geeft zijn value door, met het tweede getal gehalveerd."},
        },
    ),
    "self-attention-module": ClassBox(
        name="self-attention-module",
        start=_M_START,
        solution=_m(),
        scenarios=_M_SCENARIOS,
        compare=_close,
        digits=4,
        first=[_DOTS],
        mistakes=[
            Mistake("sizes", error=_matmul_error,
                    en="The sizes of a matrix multiplication do not fit. Check them step by step: each layer is `nn.Linear(d, d_k, bias=False)` and gets `x` itself, (n, d); the scores `q @ k.T` are (n, d_k) times (d_k, n); and the new vectors `A @ v` are (n, n) times (n, d_k).",
                    nl="De maten van een matrixvermenigvuldiging passen niet. Loop ze stap voor stap na: elke layer is `nn.Linear(d, d_k, bias=False)` en krijgt `x` zelf, (n, d); de scores `q @ k.T` zijn (n, d_k) keer (d_k, n); en de nieuwe vectoren `A @ v` zijn (n, n) keer (n, d_k)."),
            Mistake("no-dim", error=lambda e: isinstance(e, TypeError) and "softmax()" in str(e) and "dim" in str(e),
                    en="`torch.softmax` needs a `dim`: the dimension along which it takes the softmax. With `dim=1` it takes the softmax of every row, so that the weights of every word add up to 1.",
                    nl="`torch.softmax` heeft een `dim` nodig: de dimensie waarlangs hij de softmax neemt. Met `dim=1` neemt hij de softmax van elke rij, zodat de weights van elk woord optellen tot 1."),
            Mistake("bias", wrong=_m(init=_M_INIT.replace(", bias=False)", ")")),
                    en=_SCENARIO + ": your layers have bias terms, so even embeddings of zeros give queries, keys and values that are not 0. Make them with `bias=False`: `nn.Linear(d, d_k, bias=False)`.",
                    nl=_SCENARIO_NL + ": je layers hebben bias terms, dus zelfs embeddings van nullen geven queries, keys en values die niet 0 zijn. Maak ze met `bias=False`: `nn.Linear(d, d_k, bias=False)`."),
            Mistake("order", matches=_out_of_order,
                    en=_SCENARIO + ": the right shape, other numbers. Make the three layers in the order query, key, value, so that they get the same random starting values as in the course's solution.",
                    nl=_SCENARIO_NL + ": de goede shape, andere getallen. Maak de drie layers in de volgorde query, key, value, zodat ze dezelfde willekeurige startwaarden krijgen als in de oplossing van de course."),
            Mistake("keys-first", wrong=_m(forward=_M_FORWARD.replace("q @ k.T", "k @ q.T")),
                    en=_SCENARIO + ": your scores are the other way round. Row `i` holds the query of word `i` against every key: `q @ k.T`.",
                    nl=_SCENARIO_NL + ": je scores staan andersom. Rij `i` bevat de query van woord `i` tegen elke key: `q @ k.T`."),
            Mistake("columns", wrong=_m(forward=_M_FORWARD.replace("dim=1", "dim=0")),
                    en=_SCENARIO + ": with `dim=0`, every column of weights adds up to 1, not every row. The weights of word `i` are row `i`: `torch.softmax(..., dim=1)`.",
                    nl=_SCENARIO_NL + ": met `dim=0` telt elke kolom weights op tot 1, niet elke rij. De weights van woord `i` zijn rij `i`: `torch.softmax(..., dim=1)`."),
            Mistake("embeddings", wrong=_m(forward=_M_FORWARD.replace("return A @ v", "return A @ x")),
                    en=_SCENARIO + ": you average the embeddings. What a word passes on is its value: `A @ v`.",
                    nl=_SCENARIO_NL + ": je middelt de embeddings. Wat een woord doorgeeft, is zijn value: `A @ v`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the three layers in `__init__`, and in `forward` the queries, keys and values, the weights and the new vectors.",
                      "nl": "Vervang de twee `...`: de drie layers in `__init__`, en in `forward` de queries, keys en values, de weights en de nieuwe vectoren."},
            "wrong": {"en": _SCENARIO + ". Queries, keys and values from the three layers, the softmax of `q @ k.T` along every row, and the weighted values: steps 1 to 4 of the recipe?",
                      "nl": _SCENARIO_NL + ". Queries, keys en values uit de drie layers, de softmax van `q @ k.T` langs elke rij, en de gewogen values: stap 1 tot en met 4 van het recept?"},
            "almost": {"en": "Right for 5 words. But " + _SCENARIO[0].lower() + _SCENARIO[1:] + ". Does your class use `d` and `d_k`, and no fixed sizes?",
                       "nl": "Goed voor 5 woorden. Maar " + _SCENARIO_NL[0].lower() + _SCENARIO_NL[1:] + ". Gebruikt je class `d` en `d_k`, en geen vaste maten?"},
            "correct": {"en": "Your module is self-attention with learned queries, keys and values: training changes its three matrices, so it learns what each word looks for, offers and passes on.",
                        "nl": "Je module is self-attention met geleerde queries, keys en values: trainen verandert zijn drie matrices, dus hij leert waar elk woord naar zoekt, wat het aanbiedt en wat het doorgeeft."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("qkv-attention", _N_START, "empty", None),
    ("qkv-attention", _n(q="np.dot(X, Wq)", k="np.dot(X, Wk)", v="np.dot(X, Wv)", s="np.dot(Q, K.T)", out="np.dot(A, V)"), "correct", None),
    ("qkv-attention", _n(q="Wq @ X"), "error", "weights-first"),
    ("qkv-attention", _n(s="Q @ K"), "error", "sizes"),
    ("qkv-attention", _n(s="K @ Q.T"), "wrong", "keys-first"),
    ("qkv-attention", _n(out="A @ X"), "wrong", "embeddings"),
    ("qkv-attention", _n(out="V @ A"), "error", "sizes"),
    ("self-attention-module", _M_START, "empty", None),
    ("self-attention-module", _m(forward=("        q, k, v = self.query(x), self.key(x), self.value(x)\n"
                                          "        weights = torch.softmax(q @ k.transpose(0, 1), dim=-1)\n        return weights @ v\n")), "correct", None),
    ("self-attention-module", _m(init=_M_INIT.replace(", bias=False)", ")")), "wrong", "bias"),
    ("self-attention-module", _m(init=("        self.value = nn.Linear(d, d_k, bias=False)\n        self.query = nn.Linear(d, d_k, bias=False)\n"
                                       "        self.key = nn.Linear(d, d_k, bias=False)\n")), "wrong", "order"),
    ("self-attention-module", _m(forward=_M_FORWARD.replace("q @ k.T", "k @ q.T")), "wrong", "keys-first"),
    ("self-attention-module", _m(forward=_M_FORWARD.replace("dim=1", "dim=0")), "wrong", "columns"),
    ("self-attention-module", _m(forward=_M_FORWARD.replace("q @ k.T", "q @ k")), "error", "sizes"),
    ("self-attention-module", _m(init=_M_INIT.replace("nn.Linear(d, d_k", "nn.Linear(d_k, d")), "error", "sizes"),
    ("self-attention-module", _m(forward=_M_FORWARD.replace(", dim=1)", ")")), "error", "no-dim"),
    ("self-attention-module", _m(forward=_M_FORWARD.replace("return A @ v", "return A @ x")), "wrong", "embeddings"),
    ("self-attention-module", _m(init=("        self.key = nn.Linear(d, d_k, bias=False)\n        self.query = nn.Linear(d, d_k, bias=False)\n"
                                       "        self.value = nn.Linear(d, d_k, bias=False)\n"), forward=_M_FORWARD.replace("dim=1", "dim=0")), "wrong", None),
]
