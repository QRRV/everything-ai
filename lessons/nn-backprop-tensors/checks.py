# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson nn-backprop-tensors: linear_backward, the backward pass of one linear layer, saved as
nn/layer_backward.py, and mlp_gradients, the backward pass of a two-layer MLP with softmax and cross-entropy, in
NumPy, saved as nn/mlp_backward.py. And the demo: it compares the learner's own gradients with those of PyTorch on 64
digits of QMNIST, then trains the MLP with them, in NumPy only, and measures it on 1,000 test digits."""
import functools
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import paths, projects
from everything_ai.check import ClassBox, FunctionBox, Mistake, Timeout, run_code, with_time_limit

DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md
HIDDEN, LR, EPOCHS, BATCH = 64, 0.5, 5, 64   # the demo: 64 hidden neurons, a learning rate of 0.5, 5 epochs of 64


@functools.lru_cache(maxsize=1)
def digits():
    """The training images and labels and the test images and labels as NumPy arrays; every pixel divided by 255."""
    d = np.load(paths.ROOT / DATA)
    return (d["train_images"].astype(np.float64) / 255, d["train_labels"].astype(np.int64),
            d["test_images"].astype(np.float64) / 255, d["test_labels"].astype(np.int64))


def start(hidden: int = HIDDEN, seed: int = 0):
    """Starting weights for the digits MLP: small random numbers, scaled by the number of inputs, and bias terms of 0."""
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 1 / math.sqrt(784), (hidden, 784)), np.zeros(hidden),
            rng.normal(0, 1 / math.sqrt(hidden), (10, hidden)), np.zeros(10))


def forward(X, W1, b1, W2, b2):
    """The forward pass of the course: the inputs of the ReLU, what comes out of it, and the probabilities."""
    Z1 = X @ W1.T + b1
    H = np.maximum(Z1, 0)
    Z2 = H @ W2.T + b2
    E = np.exp(Z2 - Z2.max(axis=1, keepdims=True))
    return Z1, H, E / E.sum(axis=1, keepdims=True)


def loss(X, y, W1, b1, W2, b2) -> float:
    _, _, P = forward(X, W1, b1, W2, b2)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def torch_gradients(X, y, W1, b1, W2, b2):
    """The same gradients from PyTorch's autograd, in float64: the reference the demo compares with."""
    params = [torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in (W1, b1, W2, b2)]
    tW1, tb1, tW2, tb2 = params
    Z2 = torch.relu(torch.tensor(X) @ tW1.T + tb1) @ tW2.T + tb2
    P = torch.softmax(Z2, dim=1)
    loss = -torch.log(P[torch.arange(len(y)), torch.tensor(y)]).mean()
    loss.backward()
    return [p.grad.numpy() for p in params]


def power(x: float) -> int:
    """The power of 10 just above x: 10 to the power -11 for 5.3 · 10 to the power -12."""
    return math.ceil(math.log10(x)) if x > 0 else -16


def relative_error(a, n) -> float:
    """The largest relative error between two arrays, as in nn-gradient-check; where both are 0, they agree."""
    a, n = np.asarray(a, dtype=float), np.asarray(n, dtype=float)
    size = np.maximum(np.abs(a), np.abs(n))
    return float(np.max(np.where(size > 0, np.abs(a - n) / np.where(size > 0, size, 1), 0)))


# --- the demo: the learner's own gradients, checked against PyTorch and used to train ------------------------------------

_MINE = "from nn.mlp_backward import mlp_gradients\nimport numpy as np\n"


def _saved(module: str) -> bool:
    return (projects.folder() / "nn" / f"{module}.py").exists()


def train(gradients, epochs: int = EPOCHS):
    """The digits MLP trained with `gradients`, mini-batch by mini-batch, in NumPy: the same seeds every time.
    Gives back the weights and the loss over all training digits after each epoch."""
    X, y, _, _ = digits()
    W1, b1, W2, b2 = start()
    rng = np.random.default_rng(1)
    losses = []
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for s in range(0, len(y), BATCH):
            idx = order[s:s + BATCH]
            dW1, db1, dW2, db2 = (np.asarray(g, dtype=float) for g in gradients(X[idx], y[idx], W1, b1, W2, b2))
            W1, b1, W2, b2 = W1 - LR * dW1, b1 - LR * db1, W2 - LR * dW2, b2 - LR * db2
        losses.append(loss(X, y, W1, b1, W2, b2))
    return (W1, b1, W2, b2), losses


def accuracy(W1, b1, W2, b2) -> float:
    _, _, Xt, yt = digits()
    _, _, P = forward(Xt, W1, b1, W2, b2)
    return float(np.mean(P.argmax(axis=1) == yt))


def demo():
    """For the demo: None when mlp_backward.py is not in the project yet; "broken" when it stops with an error or gives
    other gradients than PyTorch on 64 digits; "slow" when training takes more than a minute; else (the largest
    relative error of each of the four gradients, the accuracy, the losses)."""
    if not _saved("mlp_backward"):
        return None
    run = run_code(_MINE)
    if run.error is not None or "mlp_gradients" not in run.names:
        return "broken"
    mine = run.names["mlp_gradients"]
    X, y, _, _ = digits()
    W1, b1, W2, b2 = start()
    try:
        got = with_time_limit(lambda: mine(X[:64], y[:64], W1, b1, W2, b2), 10)
        errors = [relative_error(g, t) for g, t in zip(got, torch_gradients(X[:64], y[:64], W1, b1, W2, b2))]
    except Exception:   # also Timeout, and a wrong shape or number of gradients
        return "broken"
    if len(errors) != 4 or max(errors) > 1e-6:
        return "broken"
    try:
        weights, losses = with_time_limit(lambda: train(mine), 60)
    except Timeout:
        return "slow"
    except Exception:
        return "broken"
    return errors, accuracy(*weights), losses


def losses_picture(losses: list[float], labels: dict) -> str:
    """The loss over all training digits after each epoch of the learner's own training, a dot per epoch."""
    from everything_ai import chart
    epochs = list(range(1, len(losses) + 1))
    p = chart.Panel((0.5, len(losses) + 0.5), (0, max(0.5, max(losses) * 1.2)), title=labels["title"],
                    xlabel=labels["epoch"], ylabel=labels["loss"], xticks=epochs)
    p.line(epochs, losses, "loss", 2.5)
    p.points(epochs, losses, "loss", 4)
    return chart.figure([p], alt=labels["alt"], width=520, height=290)


# The lesson is done when these are correct. "bias-sum" is the choice question in the lesson.
REQUIRED = ["bias-sum", "linear-backward", "mlp-gradients"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


class _Matrix(np.ndarray):
    """An array that a check message writes the way the lesson's code does: np.array([[...], ...]). The learner's
    code gets a plain copy."""

    def __repr__(self):
        return f"np.array({np.asarray(self).tolist()!r})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _m(rows) -> np.ndarray:
    return np.array(rows, dtype=float).view(_Matrix)


def _round(v):
    """Arrays and numbers rounded to 4 decimals, as plain lists, for a check message without float noise."""
    if isinstance(v, (list, tuple)):
        return type(v)(_round(x) for x in v)
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return v
    out = np.round(a, 4) + 0.0
    return out.tolist() if a.ndim else float(out)


def _show(v) -> str:
    v = _round(v)
    if isinstance(v, tuple):
        return "(" + ", ".join(f"np.array({x!r})" if isinstance(x, list) else repr(x) for x in v) + ")"
    return repr(v)


def _same_grads(got, want) -> bool:
    try:
        if len(got) != len(want):
            return False
        return all(np.asarray(g, dtype=float).shape == np.asarray(w).shape and np.allclose(np.asarray(g, dtype=float), w, rtol=1e-6, atol=1e-9)
                   for g, w in zip(got, want))
    except (TypeError, ValueError):
        return False


def _shape_off(got, want) -> bool:
    try:
        return len(got) == len(want) and any(np.shape(g) != np.shape(w) and sorted(np.shape(g)) == sorted(np.shape(w)) for g, w in zip(got, want))
    except TypeError:
        return False


# --- exercise 2: one linear layer --------------------------------------------------------------------------------------------

_L_HEAD = "import numpy as np\n\n\ndef linear_backward(G, X, W):\n"
_L_START = _L_HEAD + "    # the gradients for the input X, the weights W and the bias terms b of the layer X @ W.T + b,\n    # from G, the gradient for its output\n    ...\n"


def _l(dX: str = "G @ W", dW: str = "G.T @ X", db: str = "G.sum(axis=0)") -> str:
    return _L_HEAD + f"    dX = {dX}\n    dW = {dW}\n    db = {db}\n    return dX, dW, db\n"


_G = _m([[0.5, -1.0], [0.0, 2.0], [1.0, 1.0]])
_X = _m([[1.0, 2.0, 0.0, -1.0], [0.0, 1.0, 1.0, 2.0], [2.0, 0.0, 1.0, 1.0]])
_W = _m([[1.0, 0.0, -1.0, 2.0], [0.5, 1.0, 0.0, -1.0]])

# --- exercise 3: the whole MLP -----------------------------------------------------------------------------------------------

_B_HEAD = "import numpy as np\n\n\ndef mlp_gradients(X, y, W1, b1, W2, b2):\n"
_B_START = (_B_HEAD + "    # the forward pass, keeping what the backward pass needs\n    ...\n"
            "    # the backward pass: the gradients for W1, b1, W2 and b2, in that order\n    ...\n")
_FORWARD = ("    n = len(y)\n    Z1 = X @ W1.T + b1\n    H = np.maximum(Z1, 0)\n    Z2 = H @ W2.T + b2\n"
            "    E = np.exp(Z2 - Z2.max(axis=1, keepdims=True))\n    P = E / E.sum(axis=1, keepdims=True)\n")


def _b(G2: str = "(P - np.eye(W2.shape[0])[y]) / n", G1: str = "dH * (Z1 > 0)", db: str = "sum(axis=0)") -> str:
    return (_B_HEAD + _FORWARD + f"    G2 = {G2}\n    dW2 = G2.T @ H\n    db2 = G2.{db}\n    dH = G2 @ W2\n"
            f"    G1 = {G1}\n    dW1 = G1.T @ X\n    db1 = G1.{db}\n    return dW1, db1, dW2, db2\n")


_SMALL = ("X = np.array([[1.0, 2.0], [0.0, 1.0], [2.0, -1.0], [1.0, 1.0]])\ny = np.array([0, 1, 1, 0])\n"
          "W1 = np.array([[0.5, -0.3], [0.2, 0.4], [-0.6, 0.3]])\nb1 = np.array([0.2, -0.1, -0.1])\n"
          "W2 = np.array([[0.3, -0.2, 0.5], [-0.4, 0.1, 0.2]])\nb2 = np.array([0.0, 0.1])\n"
          "dW1, db1, dW2, db2 = mlp_gradients(X, y, W1, b1, W2, b2)\n")
_LARGER = ("rng = np.random.default_rng(3)\nX = rng.normal(size=(6, 5))\ny = np.array([0, 2, 1, 2, 0, 1])\n"
           "W1 = rng.normal(size=(4, 5))\nb1 = rng.normal(size=4)\nW2 = rng.normal(size=(3, 4))\nb2 = rng.normal(size=3)\n"
           "dW1, db1, dW2, db2 = mlp_gradients(X, y, W1, b1, W2, b2)\n")
_SCENARIOS = [_SMALL + "np.asarray(dW2).tolist()", _SMALL + "np.asarray(db2).tolist()", _SMALL + "np.asarray(dW1).tolist()",
              _SMALL + "np.asarray(db1).tolist()", _LARGER + "[np.asarray(g).tolist() for g in (dW1, db1, dW2, db2)]"]

EXERCISES = {
    "linear-backward": FunctionBox(
        name="linear-backward",
        function="linear_backward",
        start=_L_START,
        solution=_l(),
        visible=[(_G, _X, _W)],
        hidden=[(_m([[1.0], [-2.0]]), _m([[3.0, 1.0], [0.5, -1.0]]), _m([[2.0, -1.0]])),
                (_m([[0.1, 0.2, 0.3]]), _m([[1.0, -1.0]]), _m([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))],
        show=_show,
        compare=_same_grads,
        save_as=("nn", "layer_backward"),
        first=[_DOTS],
        mistakes=[
            Mistake("db-mean", wrong=_l(db="G.mean(axis=0)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your `db` is the mean of `G` over the examples. `b` is added to every example, so it gets the gradient of each, added up: `G.sum(axis=0)`. The mean of the loss is already in `G`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je `db` is het gemiddelde van `G` over de voorbeelden. `b` wordt bij elk voorbeeld opgeteld, dus hij krijgt de gradient van elk, bij elkaar opgeteld: `G.sum(axis=0)`. Het gemiddelde van de loss zit al in `G`."),
            Mistake("db-axis", wrong=_l(db="G.sum(axis=1)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your `db` has one number per example. It needs one per output, the shape of `b`: add up over the examples, `axis=0`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je `db` heeft één getal per voorbeeld. Er hoort één per output, de shape van `b`: tel op over de voorbeelden, `axis=0`."),
            Mistake("shapes", error=lambda e: isinstance(e, ValueError) and "matmul" in str(e),
                    en="With `{call}`, your function stops: in one of your products the sizes in the middle differ, and `@` needs them equal. Check each product against the shapes: `X` is (examples, inputs), `W` is (outputs, inputs) and `G` is (examples, outputs).",
                    nl="Met `{call}` stopt je functie: in een van je producten verschillen de maten in het midden, en `@` wil ze gelijk. Leg elk product naast de shapes: `X` is (voorbeelden, inputs), `W` is (outputs, inputs) en `G` is (voorbeelden, outputs)."),
            Mistake("transposed", wrong=_l(dW="X.T @ G"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your `dW` is turned over, with one row per input. `dW` has the shape of `W`, one row per output: `G.T @ X`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je `dW` is omgedraaid, met één rij per input. `dW` heeft de shape van `W`, één rij per output: `G.T @ X`."),
            Mistake("order", wrong=_L_HEAD + "    return G.T @ X, G @ W, G.sum(axis=0)\n",
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function gives `dW` first and `dX` second. The check takes them in the order of the task: `return dX, dW, db`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je functie geeft eerst `dW` en dan `dX`. De check neemt ze in de volgorde van de opdracht: `return dX, dW, db`."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "Replace the `...` with the three gradients of the layer and give them back: for `X` and `W` a product with `G`, for `b` a sum.",
                      "nl": "Vervang de `...` door de drie gradients van de layer en geef ze terug: voor `X` en `W` een product met `G`, voor `b` een som."},
            "arguments": {"en": "The check calls `{call}`, with three arrays, and the `def` line of your function has another number of names between its parentheses. Keep `(G, X, W)`.",
                          "nl": "De check roept `{call}` aan, met drie arrays, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(G, X, W)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does each gradient have the shape of what it belongs to: `dX` that of `X`, `dW` that of `W`, `db` one number per output?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Heeft elke gradient de shape van waar hij bij hoort: `dX` die van `X`, `dW` die van `W`, `db` één getal per output?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every number of examples, inputs and outputs?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elk aantal voorbeelden, inputs en outputs?"},
            "correct": {"en": "The three gradients of a linear layer are right in all {n} cases. Your `linear_backward` is now in your project, as `{saved}`.",
                        "nl": "De drie gradients van een lineaire layer kloppen in alle {n} gevallen. Je `linear_backward` staat nu in je project, als `{saved}`."},
            "kept": {"en": "The three gradients of a linear layer are right in all {n} cases. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "De drie gradients van een lineaire layer kloppen in alle {n} gevallen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
        },
    ),
    "mlp-gradients": ClassBox(
        name="mlp-gradients",
        start=_B_START,
        solution=_b(),
        scenarios=_SCENARIOS,
        digits=4,
        save_as=("nn", "mlp_backward"),
        first=[_DOTS],
        mistakes=[
            Mistake("no-n", wrong=_b(G2="P - np.eye(W2.shape[0])[y]"),
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: every gradient is `n` times too large. The loss is the mean over the examples, so the gradient for the logits is `(P - Y) / n`.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: elke gradient is `n` keer te groot. De loss is het gemiddelde over de voorbeelden, dus de gradient voor de logits is `(P - Y) / n`."),
            Mistake("no-relu", wrong=_b(G1="dH"), state="almost",
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: the gradient goes back through the ReLU unchanged. Where the input of the ReLU was 0 or below, its output did not move, so the gradient there is 0: `dH * (Z1 > 0)`.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: de gradient gaat ongewijzigd terug door de ReLU. Waar de input van de ReLU 0 of lager was, bewoog de output niet, dus daar is de gradient 0: `dH * (Z1 > 0)`."),
            Mistake("times-h", wrong=_b(G1="dH * H"), state="almost",
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: you multiply by the output of the ReLU. The slope of ReLU is 1 where its input is above 0 and 0 elsewhere: multiply by `(Z1 > 0)`.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: je vermenigvuldigt met de output van de ReLU. De helling van ReLU is 1 waar de input boven 0 ligt en elders 0: vermenigvuldig met `(Z1 > 0)`."),
            Mistake("db-mean", wrong=_b(db="mean(axis=0)"), state="almost",
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: the gradient of a bias term is the sum of `G` over the examples, not the mean; the mean of the loss is already in `G`.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: de gradient van een bias term is de som van `G` over de voorbeelden, niet het gemiddelde; het gemiddelde van de loss zit al in `G`."),
            Mistake("no-onehot", wrong=_b(G2="(P - y.reshape(-1, 1)) / n"),
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: from `P` you take off the labels as numbers. Take off the one-hot rows: `np.eye(W2.shape[0])[y]`, a 1 at the right class of each example.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: je haalt de labels als getallen van `P` af. Haal de one-hot-rijen eraf: `np.eye(W2.shape[0])[y]`, een 1 bij de goede klasse van elk voorbeeld."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "Replace the two `...`: first the forward pass, keeping `Z1`, `H` and `P`; then the gradients, from the loss back to the first layer.",
                      "nl": "Vervang de twee `...`: eerst de forward pass, en bewaar `Z1`, `H` en `P`; dan de gradients, van de loss terug naar de eerste layer."},
            "wrong": {"en": "With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Work back from the loss, one step at a time, as in the recipe: does each gradient have the shape of what it belongs to?",
                      "nl": "Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Werk terug vanaf de loss, stap voor stap, zoals in het recept: heeft elke gradient de shape van waar hij bij hoort?"},
            "almost": {"en": "The gradients checked before this one are right. But with these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Look at the step that makes this gradient: back through `W2`, back through the ReLU, and for any number of examples, hidden neurons and classes.",
                       "nl": "De gradients die eerder gecontroleerd zijn, kloppen. Maar met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Kijk naar de stap die deze gradient maakt: terug door `W2`, terug door de ReLU, en voor elk aantal voorbeelden, hidden neuronen en klassen."},
            "correct": {"en": "All four gradients are right, also for a larger network with three classes. Your `mlp_gradients` is now in your project, as `{saved}`.",
                        "nl": "Alle vier gradients kloppen, ook voor een groter netwerk met drie klassen. Je `mlp_gradients` staat nu in je project, als `{saved}`."},
            "kept": {"en": "All four gradients are right, also for a larger network with three classes. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Alle vier gradients kloppen, ook voor een groter netwerk met drie klassen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("linear-backward", _L_START, "empty", None),
    ("linear-backward", _l(), "correct", None),
    ("linear-backward", _l(dW="(X.T @ G).T", db="np.sum(G, axis=0)"), "correct", None),
    ("linear-backward", _l(db="G.mean(axis=0)"), "wrong", "db-mean"),
    ("linear-backward", _l(db="G.sum(axis=1)"), "wrong", "db-axis"),
    ("linear-backward", _l(dW="X.T @ G"), "wrong", "transposed"),
    ("linear-backward", _l(dX="G @ W.T"), "error", "shapes"),
    ("linear-backward", _L_HEAD + "    return G.T @ X, G @ W, G.sum(axis=0)\n", "wrong", "order"),
    ("mlp-gradients", _B_START, "empty", None),
    ("mlp-gradients", _b(), "correct", None),
    ("mlp-gradients", _b(G1="dH * (H > 0)"), "correct", None),
    ("mlp-gradients", _b(G1="dH * (Z1 >= 0)"), "correct", None),   # no example sits on the kink of a ReLU
    ("mlp-gradients", _B_HEAD + _FORWARD + "    Y = np.zeros_like(P)\n    Y[np.arange(n), y] = 1\n    G2 = (P - Y) / n\n    G1 = (G2 @ W2) * (Z1 > 0)\n"
     "    return G1.T @ X, G1.sum(axis=0), G2.T @ H, G2.sum(axis=0)\n", "correct", None),
    ("mlp-gradients", _b(G2="P - np.eye(W2.shape[0])[y]"), "wrong", "no-n"),
    ("mlp-gradients", _b(G1="dH"), "almost", "no-relu"),
    ("mlp-gradients", _b(G1="dH * H"), "almost", "times-h"),
    ("mlp-gradients", _b(db="mean(axis=0)"), "almost", "db-mean"),
    ("mlp-gradients", _b(G2="(P - y.reshape(-1, 1)) / n"), "wrong", "no-onehot"),
]
