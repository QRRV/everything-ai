# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rnn-lstm: lstm_step, one step of an LSTM in NumPy with PyTorch's order of the gates, compared
with PyTorch's nn.LSTMCell. And the picture: how far back the gradient reaches in a plain recurrent network and in an
LSTM, measured with PyTorch."""
import functools
import math
import re

import numpy as np

from everything_ai.check import FunctionBox, Mistake

STEPS, HIDDEN, INPUTS, SEEDS, FORGET_BIAS = 60, 16, 4, 8, 3.0


def _reach_one(kind: str, seed: int) -> list:
    """log10 of how much input x_t still changes the last hidden state, relative to the last input, for every step back
    from 0 to STEPS - 1: a plain recurrent network ("rnn"), an LSTM as PyTorch starts it ("lstm"), or that LSTM with
    the bias of its forget gate raised by FORGET_BIAS ("keep"). Weights from seed `seed`, inputs from seed 100 + seed."""
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    m = torch.nn.RNN(INPUTS, HIDDEN) if kind == "rnn" else torch.nn.LSTM(INPUTS, HIDDEN)
    if kind == "keep":
        with torch.no_grad():
            m.bias_ih_l0[HIDDEN:2 * HIDDEN] += FORGET_BIAS
    g = torch.Generator().manual_seed(100 + seed)
    x = torch.randn(STEPS, 1, INPUTS, generator=g).requires_grad_(True)
    out, _ = m(x)
    out[-1].sum().backward()
    size = x.grad[:, 0, :].norm(dim=1)
    return [math.log10(float(size[STEPS - 1 - k] / size[-1])) for k in range(STEPS)]


@functools.lru_cache(maxsize=4)
def reach(kind: str) -> tuple:
    """How much input x_t still changes the last hidden state, relative to the last input, for every step back: the
    geometric mean over SEEDS networks and inputs."""
    runs = [_reach_one(kind, s) for s in range(SEEDS)]
    return tuple(10 ** (sum(r[k] for r in runs) / SEEDS) for k in range(STEPS))


def reach_picture(labels: dict) -> str:
    """How far back the gradient reaches, on a log scale: a plain recurrent network (grey, dashed), an LSTM as PyTorch
    starts it (gradient colour, dashed) and an LSTM that keeps its cell state (gradient colour, solid)."""
    from everything_ai import chart
    p = chart.Panel((0, STEPS - 1), (1e-16, 10), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    ylog=True, xticks=[0, 20, 40, 59])
    ks = list(range(STEPS))
    for kind, role, dashed, (tx, ty, anchor) in (("rnn", "ink-3", True, (2, 1e-14, "start")),
                                                 ("lstm", "grad", True, (58, 1e-4, "end")),
                                                 ("keep", "grad", False, (58, 0.01, "end"))):
        p.line(ks, list(reach(kind)), role, 2.2, dashed=dashed)
        p.text(tx, ty, labels[kind], role, anchor)
    svg = chart.figure([p], alt=labels["alt"], width=330, height=260)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# --- exercise 1: lstm_step -----------------------------------------------------------------------------------------------------

_HEAD = ("import numpy as np\n\n\ndef sigmoid(z):\n    return 1 / (1 + np.exp(-z))\n\n\n"
         "def lstm_step(x, h, c, W_x, W_h, b):\n"
         "    x, h, c = np.array(x), np.array(h), np.array(c)\n"
         "    z = np.array(W_x) @ x + np.array(W_h) @ h + np.array(b)\n"
         "    n = len(h)\n"
         "    # the four parts of z, in PyTorch's order: input gate, forget gate, candidate, output gate\n"
         "    i = sigmoid(z[0:n])\n"
         "    f = sigmoid(z[n:2 * n])\n")
_G = "    g = np.tanh(z[2 * n:3 * n])\n"
_O = "    o = sigmoid(z[3 * n:4 * n])\n"
_MID = "    # the new cell state, and the new hidden state\n"
_C = "    c_new = f * c + i * g\n"
_H = "    h_new = o * np.tanh(c_new)\n"
_TAIL = "    return h_new, c_new\n"
_START = _HEAD + "    g = ...\n    o = ...\n" + _MID + "    c_new = ...\n    h_new = ...\n" + _TAIL


def _b(g: str = _G, o: str = _O, c: str = _C, h: str = _H) -> str:
    return _HEAD + g + o + _MID + c + h + _TAIL


def _pair(v):
    """The outcome as two lists of floats, (h_new, c_new), or None when it is not two vectors."""
    try:
        if isinstance(v, (str, bytes)) or len(v) != 2:
            return None
        return [np.asarray(part, dtype=float).reshape(-1) for part in v]
    except (TypeError, ValueError):
        return None


def _close(got, want) -> bool:
    a, b = _pair(got), _pair(want)
    return a is not None and b is not None and all(x.shape == y.shape and np.allclose(x, y, rtol=1e-6, atol=1e-9)
                                                   for x, y in zip(a, b))


def _show(v):
    """(h_new, c_new) as two lists rounded to 4 decimals, on one line."""
    p = _pair(v)
    if p is None:
        try:   # one array or number instead of two: rounded too, without float noise
            return repr((np.round(np.asarray(v, dtype=float), 4) + 0.0).tolist())
        except (TypeError, ValueError):
            return repr(v)
    return "(" + ", ".join(repr((np.round(part, 4) + 0.0).tolist()) for part in p) + ")"


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_WRONG = {
    "g-sigmoid": _b(g="    g = sigmoid(z[2 * n:3 * n])\n"),
    "order": _b(g="    g = np.tanh(z[3 * n:4 * n])\n", o="    o = sigmoid(z[2 * n:3 * n])\n"),
    "no-forget": _b(c="    c_new = c + i * g\n"),
    "no-old": _b(c="    c_new = i * g\n"),
    "h-no-tanh": _b(h="    h_new = o * c_new\n"),
    "old-c": _b(h="    h_new = o * np.tanh(c)\n"),
}

_VISIBLE = ([1.0], [0.5], [2.0], [[1.0], [2.0], [0.5], [-1.0]], [[0.5], [1.0], [1.0], [2.0]], [0.0, 1.0, 0.0, 0.5])

EXERCISES = {
    "lstm-step": FunctionBox(
        name="lstm-step",
        function="lstm_step",
        start=_START,
        solution=_b(),
        visible=[_VISIBLE],
        hidden=[
            ([2.0], [-1.0], [-0.5], [[0.5], [-1.0], [1.0], [0.25]], [[1.0], [0.5], [-2.0], [1.0]], [1.0, 0.0, -0.5, 2.0]),
            ([1.0], [0.3, -0.6], [1.5, -1.0],
             [[1.0], [-1.0], [0.5], [0.5], [2.0], [0.0], [1.0], [-1.0]],
             [[0.5, 0.0], [0.0, 0.5], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [-1.0, 1.0]],
             [0.0, 0.5, -1.0, 2.0, 0.5, -0.5, 0.0, 1.0]),
            ([0.0, 0.0], [0.0, 0.0], [1.0, -2.0], [[0.0, 0.0]] * 8, [[0.0, 0.0]] * 8, [0.0] * 8),
            ([1], [0], [0], [[1], [1], [1], [1]], [[0], [0], [0], [0]], [0, 0, 0, 0]),
        ],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("g-sigmoid", wrong=_WRONG["g-sigmoid"],
                    en="`{call}` gives {got}, and it should give {want}: your candidate `g` comes from the sigmoid, so it can only add to the cell state. It is made with tanh, between −1 and 1, so that it can also take away.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je kandidaat `g` komt uit de sigmoid, dus hij kan alleen iets bij de cell state optellen. Hij wordt gemaakt met tanh, tussen −1 en 1, zodat hij er ook iets af kan halen."),
            Mistake("order", wrong=_WRONG["order"],
                    en="`{call}` gives {got}, and it should give {want}: your `g` and `o` come from each other's part of `z`. PyTorch's order is input gate, forget gate, candidate, output gate: `g` is the third part, `o` the fourth.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `g` en `o` komen uit elkaars deel van `z`. De volgorde van PyTorch is input gate, forget gate, kandidaat, output gate: `g` is het derde deel, `o` het vierde."),
            Mistake("no-forget", wrong=_WRONG["no-forget"],
                    en="`{call}` gives {got}, and it should give {want}: your new cell state keeps all of the old one. The forget gate decides how much of it stays: multiply `c` by `f`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je nieuwe cell state houdt de oude helemaal. De forget gate beslist hoeveel ervan blijft: vermenigvuldig `c` met `f`."),
            Mistake("no-old", wrong=_WRONG["no-old"],
                    en="`{call}` gives {got}, and it should give {want}: your new cell state has only the new part, `i * g`. The old cell state stays as far as the forget gate lets it: add `f * c`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je nieuwe cell state heeft alleen het nieuwe deel, `i * g`. De oude cell state blijft voor zover de forget gate hem laat: tel `f * c` erbij op."),
            Mistake("h-no-tanh", wrong=_WRONG["h-no-tanh"],
                    en="`{call}` gives {got}, and it should give {want}: your hidden state is the output gate times the cell state itself. The cell state can grow beyond 1; it goes through tanh first: `o * np.tanh(c_new)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je hidden state is de output gate maal de cell state zelf. De cell state kan boven 1 uitkomen; hij gaat eerst door tanh: `o * np.tanh(c_new)`."),
            Mistake("old-c", wrong=_WRONG["old-c"],
                    en="`{call}` gives {got}, and it should give {want}: your hidden state is made from the old cell state, `c`. It shows the new one: use `c_new`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je hidden state wordt gemaakt uit de oude cell state, `c`. Hij laat de nieuwe zien: gebruik `c_new`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace each `...` with your code: the candidate `g` and the output gate `o` from their parts of `z`, then the new cell state and the new hidden state.",
                      "nl": "Vervang elke `...` door je code: de kandidaat `g` en de output gate `o` uit hun delen van `z`, dan de nieuwe cell state en de nieuwe hidden state."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the input, the hidden state, the cell state, the weights and the bias, and the `def` line of your function has another number of names between its parentheses. Keep `(x, h, c, W_x, W_h, b)`.",
                          "nl": "De check roept `{call}` aan, met de input, de hidden state, de cell state, de weights en de bias, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x, h, c, W_x, W_h, b)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Compare each line with the formulas in *The idea*: which part of `z` each gate comes from, sigmoid or tanh, and what goes into the new cell state and the new hidden state.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Vergelijk elke regel met de formules in *Het idee*: uit welk deel van `z` elke gate komt, sigmoid of tanh, en wat er in de nieuwe cell state en de nieuwe hidden state gaat."},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does it work for a hidden state of more than one number, for weights of 0, and for whole numbers?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt het ook voor een hidden state van meer dan één getal, voor weights van 0, en voor gehele getallen?"},
            "correct": {"en": "Your LSTM step gives the right hidden state and cell state in all {n} cases.",
                        "nl": "Je LSTM-stap geeft de goede hidden state en cell state in alle {n} gevallen."},
        },
    ),
}

# The lesson is done when these are correct. "forget-zero" and "gru-copy" are the choice questions.
REQUIRED = ["lstm-step", "forget-zero", "gru-copy"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("lstm-step", _START, "empty", None),
    ("lstm-step", _b(), "correct", None),
    ("lstm-step", _b(c="    c_new = i * g + c * f\n", h="    h_new = np.tanh(c_new) * o\n"), "correct", None),
] + [("lstm-step", code, "wrong", key) for key, code in _WRONG.items()]


def against_torch(text) -> tuple | None:
    """The learner's lstm_step against PyTorch's nn.LSTMCell with the same weights, for an input of 3 numbers and a
    hidden state of 2: (their h_new, PyTorch's h_new), each a list rounded to 4 decimals."""
    import torch
    from everything_ai.check import run_code
    ran = run_code(text) if text else None
    fn = ran.names.get("lstm_step") if ran is not None and ran.error is None else None
    if not callable(fn):
        return None
    torch.manual_seed(0)
    cell = torch.nn.LSTMCell(3, 2)
    x, h, c = torch.tensor([1.0, -0.5, 0.2]), torch.tensor([0.1, -0.3]), torch.tensor([0.8, -0.4])
    with torch.no_grad():
        h_torch, _ = cell(x, (h, c))
        W_x, W_h, b = cell.weight_ih.tolist(), cell.weight_hh.tolist(), (cell.bias_ih + cell.bias_hh).tolist()
    try:
        theirs = fn(x.tolist(), h.tolist(), c.tolist(), W_x, W_h, b)
        mine = [round(float(v), 4) for v in np.asarray(theirs[0], dtype=float).reshape(-1)]
    except Exception:
        return None
    return mine, [round(float(v), 4) for v in h_torch.tolist()]
