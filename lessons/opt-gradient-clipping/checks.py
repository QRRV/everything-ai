# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson opt-gradient-clipping: clip_grads, which scales a list of gradients down to a largest norm,
and train_step, one training step with torch.nn.utils.clip_grad_norm_ between backward() and step(). And the picture:
the MLP for the handwritten digits of QMNIST, one epoch of SGD, with one mini-batch whose pixels were not divided by
255, without and with gradient clipping."""

import functools
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import paths
from everything_ai.check import ClassBox, FunctionBox, Mistake

DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md
LR, HIDDEN, BAD, CLIP = 0.5, 64, 40, 5.0   # the MLP of torch-mlp-digits; the bad mini-batch is step 40 of 94


class DigitsMLP(nn.Module):
    """The MLP of torch-mlp-digits."""

    def __init__(self, hidden):
        super().__init__()
        self.hidden = nn.Linear(784, hidden)
        self.out = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.out(torch.relu(self.hidden(x)))


@functools.lru_cache(maxsize=3)
def run(clip: float | None, bad: bool = True):
    """One epoch of SGD; with `bad`, mini-batch BAD has pixels from 0 to 255. Gives the norm of the gradient of each
    mini-batch before clipping, the loss of each mini-batch, and the number of the 1,000 test digits read right."""
    d = np.load(paths.ROOT / DATA)
    X = torch.tensor(d["train_images"], dtype=torch.float32) / 255
    y = torch.tensor(d["train_labels"], dtype=torch.int64)
    X_test = torch.tensor(d["test_images"], dtype=torch.float32) / 255
    y_test = torch.tensor(d["test_labels"], dtype=torch.int64)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)   # faster on these small layers, and the numbers no longer depend on the number of cores
    try:
        torch.manual_seed(0)
        model = DigitsMLP(HIDDEN)
        opt = torch.optim.SGD(model.parameters(), lr=LR)
        loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
        loss_fn = nn.CrossEntropyLoss()
        torch.manual_seed(1)
        norms, losses = [], []
        for step, (xb, yb) in enumerate(loader):
            if bad and step == BAD:
                xb = xb * 255
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip else math.inf)
            opt.step()
            norms.append(norm.item())
            losses.append(loss.item())
        with torch.no_grad():
            right = int((model(X_test).argmax(dim=1) == y_test).sum().item())
    finally:
        torch.set_num_threads(threads)
    return norms, losses, right


def clip_picture(labels: dict) -> str:
    """Left: the norm of the gradient at each mini-batch, before clipping, and the limit. Right: the loss of each
    mini-batch, without and with clipping. Both on a log scale."""
    from everything_ai import chart
    plain, clipped = run(None), run(CLIP)
    total = len(plain[0])
    steps = list(range(1, total + 1))
    left = chart.Panel((0, total + 1), (0.1, 2000), title=labels["norm_title"], xlabel=labels["x"], ylabel=labels["norm"], ylog=True)
    right = chart.Panel((0, total + 1), (0.1, 2000), title=labels["loss_title"], xlabel=labels["x"], ylabel=labels["loss"], ylog=True)
    left.line(steps, plain[0], "grad", 1.4, dashed=True)
    left.line(steps, clipped[0], "grad", 2.4)
    left.line([0, total + 1], [CLIP, CLIP], "hyper", 1.4, dashed=True)
    left.text(total, CLIP * 1.6, labels["limit"], "hyper", "end")
    right.line(steps, plain[1], "loss", 1.4, dashed=True)
    right.line(steps, clipped[1], "loss", 2.4)
    right.text(BAD + 6, 300, labels["without"], "loss", "start")
    right.text(BAD + 6, 0.15, labels["with"], "loss", "start")
    return chart.figure([left, right], alt=labels["alt"], width=330, height=290)


# The lesson is done when these are correct. "keeps-direction" is the choice question in the lesson.
REQUIRED = ["keeps-direction", "clip-grads", "train-step"]

def _close32(got, want) -> bool:
    """For a training step: PyTorch calculates in float32, and clipping by hand, or clip_grad_norm_, which divides by
    the norm plus 0.000001, gives the same weights with other roundings, about 1e-9 apart."""
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close32(g, w) for g, w in zip(got, want))
    try:
        return math.isclose(float(got), float(want), rel_tol=1e-5, abs_tol=1e-6)
    except (TypeError, ValueError):
        return False


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
    if x == 0:
        return "0"
    return np.format_float_positional(x, precision=5, unique=False, fractional=False, trim="-").rstrip(".")


def _nested(v) -> str:
    if np.ndim(v) == 0:
        return _num(v)
    return "[" + ", ".join(_nested(x) for x in v) + "]"


class _Array(np.ndarray):
    """A gradient that a check message writes the way the lesson's code does: np.array([...]). The learner's code
    gets a plain copy."""

    def __repr__(self):
        return f"np.array({_nested(np.asarray(self).tolist())})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _a(values) -> np.ndarray:
    return np.array(values, dtype=float).view(_Array)


def _show(v) -> str:
    """A list of gradients as code: [np.array([...]), ...]; a number as a number; anything else the way Python shows it."""
    if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
        return _num(v)
    if isinstance(v, (list, tuple)) or isinstance(v, np.ndarray) and v.dtype == object:
        try:
            return "[" + ", ".join(f"np.array({_nested(np.asarray(x, dtype=float).tolist())})" for x in v) + "]"
        except (TypeError, ValueError):
            return repr(v)
    return repr(v)


def _close_grads(got, want) -> bool:
    if isinstance(got, np.ndarray) and got.dtype == object and got.ndim == 1:   # a NumPy array that holds the arrays
        got = list(got)
    if not isinstance(got, (list, tuple)) or len(got) != len(want):
        return False
    try:
        pairs = [(np.asarray(g, dtype=float), np.asarray(w, dtype=float)) for g, w in zip(got, want)]
    except (TypeError, ValueError):
        return False
    return all(g.shape == w.shape and np.allclose(g, w, rtol=1e-6, atol=1e-12) for g, w in pairs)


# --- exercise 2: clipping a list of gradients ------------------------------------------------------------------------------

_C_HEAD = "import numpy as np\n\n\ndef clip_grads(grads, max_norm):\n"
_C_START = (_C_HEAD + "    # the norm of all numbers of all arrays together; above max_norm, every array times max_norm / norm\n"
            "    ...\n")


def _c(total: str = "np.sqrt(sum(np.sum(g ** 2) for g in grads))", test: str = "norm > max_norm",
       scaled: str = "[g * (max_norm / norm) for g in grads]") -> str:
    return (_C_HEAD + f"    norm = {total}\n    if {test}:\n        return {scaled}\n    return grads\n")


_C_VISIBLE = ([_a([1.0, 2.0]), _a([[2.0], [4.0]])], 2.5)
_C_HIDDEN = [
    ([_a([0.3, 0.4]), _a([[0.0], [1.2]])], 2.0),                  # norm 1.3: below the limit, unchanged
    ([_a([6.0, 8.0])], 5.0),                                      # one array
    ([_a([0.0, 0.0]), _a([0.0])], 1.0),                           # norm 0
    ([_a([2.0]), _a([[1.0, 2.0], [0.0, 4.0]]), _a([-2.0])], 1.0),  # norm 5.385, three arrays
    ([_a([3.0, 4.0])], 5.0),                                      # exactly at the limit
]
_C_CASES = [_C_VISIBLE] + _C_HIDDEN

# --- exercise 3: one training step with clipping -----------------------------------------------------------------------------

_T_HEAD = "import torch\nfrom torch import nn\n\n\ndef train_step(model, opt, xb, yb, max_norm):\n    loss_fn = nn.CrossEntropyLoss()\n"
_T_START = (_T_HEAD + "    # gradients to zero, the loss and backward(), clip to max_norm, the step;\n"
            "    # give back the norm of the gradient before clipping, as a number\n    ...\n")
_ZERO = "    opt.zero_grad()\n"
_LOSS = "    loss = loss_fn(model(xb), yb)\n    loss.backward()\n"
_CLIP = "    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)\n"
_STEP = "    opt.step()\n"
_RETURN = "    return norm.item()\n"


def _t(*parts: str) -> str:
    return _T_HEAD + "".join(parts or (_ZERO, _LOSS, _CLIP, _STEP, _RETURN))


_T_SETUP = ("torch.manual_seed(0)\nmodel = nn.Linear(3, 2)\nopt = torch.optim.SGD(model.parameters(), lr=0.1)\n"
            "xb = torch.tensor([[1.0, 2.0, -1.0], [0.0, -3.0, 2.0]])\nyb = torch.tensor([1, 0])\n")
_T_SCENARIOS = [
    _T_SETUP + "norm = train_step(model, opt, xb, yb, 1.0)\n[round(float(norm), 4), model.weight.tolist()]",
    _T_SETUP + "first = train_step(model, opt, xb, yb, 100.0)\nsecond = train_step(model, opt, xb, yb, 2.0)\n"
               "[round(float(first), 4), round(float(second), 4), model.bias.tolist()]",
]

_BY_VALUE = Mistake("by-value", matches=lambda got, text: re.search(r"clip_grad_value_", text) is not None,
                   en="You use `clip_grad_value_`, which cuts off every number on its own, changes the direction of the step, and gives back nothing. Use `torch.nn.utils.clip_grad_norm_`: it scales the whole gradient down to `max_norm` and gives back the norm before clipping.",
                   nl="Je gebruikt `clip_grad_value_`, die elk getal apart afkapt, de richting van de stap verandert, en niets teruggeeft. Gebruik `torch.nn.utils.clip_grad_norm_`: die schaalt de hele gradient terug tot `max_norm` en geeft de norm vóór het clippen terug.")

EXERCISES = {
    "clip-grads": FunctionBox(
        name="clip-grads",
        function="clip_grads",
        start=_C_START,
        solution=_c(),
        visible=[_C_VISIBLE],
        hidden=_C_HIDDEN,
        show=_show,
        compare=_close_grads,
        first=[_DOTS],
        mistakes=[
            Mistake("one-array", error=lambda e: isinstance(e, ValueError) and "inhomogeneous" in str(e),
                    en="NumPy cannot make one array out of arrays with different shapes, such as the `(2,)` and the `(2, 1)` of the example, so `np.linalg.norm(grads)` or `np.array(grads)` stops with an error. Work out `np.sum(g ** 2)` for each array on its own, and add those up.",
                    nl="NumPy kan van arrays met verschillende vormen, zoals de `(2,)` en de `(2, 1)` uit het voorbeeld, geen één array maken, dus `np.linalg.norm(grads)` of `np.array(grads)` stopt met een fout. Reken `np.sum(g ** 2)` voor elke array apart uit, en tel die op."),
            Mistake("each-own", wrong=_c(total="None", test="True", scaled="[g * min(1, max_norm / np.sqrt(np.sum(g ** 2))) for g in grads]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you clip every array by its own norm, so each array gets another factor and the direction of the whole gradient changes. Work out one norm over the numbers of all arrays together, and scale every array by the same factor.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je clipt elke array met zijn eigen norm, dus elke array krijgt een andere factor en de richting van de hele gradient verandert. Reken één norm uit over de getallen van alle arrays samen, en schaal elke array met dezelfde factor."),
            Mistake("no-root", wrong=_c(total="sum(np.sum(g ** 2) for g in grads)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your norm is the sum of the squares, without the square root, so the gradient becomes much smaller than `max_norm`. The norm is the square root of that sum.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je norm is de som van de kwadraten, zonder de wortel, dus de gradient wordt veel kleiner dan `max_norm`. De norm is de wortel uit die som."),
            Mistake("by-value", wrong=_c(test="True", scaled="[np.clip(g, -max_norm, max_norm) for g in grads]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you cut off every number on its own at `max_norm`, which changes the direction of the gradient. Scale all numbers by the same factor, `max_norm / norm`, when the norm of all of them together is above `max_norm`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je kapt elk getal apart af op `max_norm`, en dat verandert de richting van de gradient. Schaal alle getallen met dezelfde factor, `max_norm / norm`, als de norm van allemaal samen boven `max_norm` ligt."),
            Mistake("always", wrong=_c(test="norm > 0"), state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: you scale also when the norm is below `max_norm`, and then a small gradient becomes larger. Clipping only ever makes the gradient smaller: scale only when the norm is above `max_norm`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je schaalt ook als de norm onder `max_norm` ligt, en dan wordt een kleine gradient groter. Clippen maakt de gradient alleen kleiner: schaal alleen als de norm boven `max_norm` ligt."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: the norm of all numbers together, and the arrays scaled down when it is above `max_norm`.",
                      "nl": "Vervang de `...` door de body van de functie: de norm van alle getallen samen, en de arrays kleiner geschaald als die boven `max_norm` ligt."},
            "arguments": {"en": "The check calls `{call}`, with a list of arrays and a number, and the `def` line of your function has another number of names between its parentheses. Keep `(grads, max_norm)`.",
                          "nl": "De check roept `{call}` aan, met een lijst arrays en een getal, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(grads, max_norm)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Is your norm the square root of the sum of the squares of all numbers of all arrays, and do you give back a list with every array times `max_norm / norm`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Is je norm de wortel uit de som van de kwadraten van alle getallen van alle arrays, en geef je een lijst terug met elke array keer `max_norm / norm`?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every list: below the limit, at the limit, with one array or with three, and when the norm is 0?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke lijst: onder de grens, op de grens, met één array of met drie, en als de norm 0 is?"},
            "correct": {"en": "Your function is right in all {n} cases: one norm over all arrays, and only scaled down above the limit.",
                        "nl": "Je functie klopt in alle {n} gevallen: één norm over alle arrays, en alleen kleiner geschaald boven de grens."},
        },
    ),
    "train-step": ClassBox(
        name="train-step",
        start=_T_START,
        solution=_t(),
        scenarios=_T_SCENARIOS,
        digits=4,
        compare=_close32,
        first=[_DOTS, _BY_VALUE],
        mistakes=[
            Mistake("after-step", wrong=_t(_ZERO, _LOSS, _STEP, _CLIP, _RETURN),
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: your norm is right, but the weights moved as without clipping: `opt.step()` used the gradient before it was clipped. Clip between `loss.backward()` and `opt.step()`.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: je norm klopt, maar de weights zijn verschoven zoals zonder clippen: `opt.step()` gebruikte de gradient voordat hij geclipt was. Clip tussen `loss.backward()` en `opt.step()`."),
            Mistake("before-backward", wrong=_t(_ZERO, _CLIP, _LOSS, _STEP, _RETURN),
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: your norm is 0. You clip before `loss.backward()`, when the gradients are still empty. Clip between `loss.backward()` and `opt.step()`.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: je norm is 0. Je clipt vóór `loss.backward()`, als de gradients nog leeg zijn. Clip tussen `loss.backward()` en `opt.step()`."),
            Mistake("norm-after", wrong=_t(_ZERO, _LOSS, _CLIP, _STEP, "    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()\n"),
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: the weights are right, but the norm you give back is the one after clipping. Give back what `clip_grad_norm_` gives back the first time: the norm before clipping, which shows how large the gradient really was.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: de weights kloppen, maar de norm die je teruggeeft is die na het clippen. Geef terug wat `clip_grad_norm_` de eerste keer teruggeeft: de norm vóór het clippen, die laat zien hoe groot de gradient echt was."),
            Mistake("zero-late", wrong=_t(_LOSS, _ZERO, _CLIP, _STEP, _RETURN),
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: your norm is 0 and the weights did not move. `opt.zero_grad()` comes after `loss.backward()`, and sets the gradients it just made to zero. Set them to zero before `loss.backward()`.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: je norm is 0 en de weights zijn niet verschoven. `opt.zero_grad()` komt na `loss.backward()`, en zet de gradients die die net maakte op nul. Zet ze vóór `loss.backward()` op nul."),
            Mistake("no-zero", wrong=_t(_LOSS, _CLIP, _STEP, _RETURN),
                    en="With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: the second step is wrong. Without `opt.zero_grad()`, PyTorch adds the new gradients to those of the step before. Set them to zero first.",
                    nl="Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: de tweede stap klopt niet. Zonder `opt.zero_grad()` telt PyTorch de nieuwe gradients op bij die van de stap ervoor. Zet ze eerst op nul."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: the gradients to zero, the loss and `backward()`, clipping with `torch.nn.utils.clip_grad_norm_`, the step, and give back the norm.",
                      "nl": "Vervang de `...`: de gradients op nul, de loss en `backward()`, clippen met `torch.nn.utils.clip_grad_norm_`, de stap, en geef de norm terug."},
            "wrong": {"en": "With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: the norm before clipping and the weights after the step. Do you clip with `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)` between `loss.backward()` and `opt.step()`?",
                      "nl": "Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: de norm vóór het clippen en de weights na de stap. Clip je met `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)` tussen `loss.backward()` en `opt.step()`?"},
            "almost": {"en": "Right for one step. But with these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: the two norms and the bias terms after two steps. Do you clip to `max_norm` itself, not to a fixed number, and set the gradients to zero in every step?",
                       "nl": "Goed voor één stap. Maar met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: de twee normen en de bias terms na twee stappen. Clip je op `max_norm` zelf, niet op een vast getal, en zet je de gradients in elke stap op nul?"},
            "correct": {"en": "Your training step clips at the right moment, and gives back the norm you can log to see a bad mini-batch coming.",
                        "nl": "Je trainingsstap clipt op het goede moment, en geeft de norm terug die je kunt bijhouden om een slechte mini-batch te zien."},
        },
    ),
}

CASES = [
    ("clip-grads", _C_START, "empty", None),
    ("clip-grads", _c(), "correct", None),
    ("clip-grads", _C_HEAD + "    norm = np.sqrt(sum(np.sum(g * g) for g in grads))\n    scale = min(1.0, max_norm / norm) if norm > 0 else 1.0\n"
                   "    return [g * scale for g in grads]\n", "correct", None),
    ("clip-grads", _C_HEAD + "    norm = np.linalg.norm(np.concatenate([g.ravel() for g in grads]))\n    if norm <= max_norm:\n        return grads\n"
                   "    return [g / norm * max_norm for g in grads]\n", "correct", None),
    ("clip-grads", _c(total="None", test="True", scaled="[g * min(1, max_norm / np.sqrt(np.sum(g ** 2))) for g in grads]"), "wrong", "each-own"),
    ("clip-grads", _c(total="sum(np.sum(g ** 2) for g in grads)"), "wrong", "no-root"),
    ("clip-grads", _c(test="True", scaled="[np.clip(g, -max_norm, max_norm) for g in grads]"), "wrong", "by-value"),
    ("clip-grads", _c(test="norm > 0"), "almost", "always"),
    ("clip-grads", "def clip_grads(grads):\n    return grads\n", "error", "arguments"),
    ("clip-grads", _c(total="np.linalg.norm(grads)"), "error", "one-array"),
    ("clip-grads", _C_HEAD + "    total = 0\n    for g in grads:\n        total = total + np.sum(g ** 2)\n    norm = total ** 0.5\n    if norm > max_norm:\n"
                   "        out = []\n        for g in grads:\n            out.append(g * (max_norm / norm))\n        return out\n    return grads\n", "correct", None),
    ("train-step", _T_START, "empty", None),
    ("train-step", _t(), "correct", None),
    ("train-step", _T_HEAD + "    opt.zero_grad()\n    loss_fn(model(xb), yb).backward()\n"
                   "    norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)\n    opt.step()\n    return float(norm)\n", "correct", None),
    ("train-step", _t(_ZERO, _LOSS, _STEP, _CLIP, _RETURN), "wrong", "after-step"),
    ("train-step", _t(_ZERO, _CLIP, _LOSS, _STEP, _RETURN), "wrong", "before-backward"),
    ("train-step", _t(_ZERO, _LOSS, _CLIP, _STEP, "    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()\n"), "wrong", "norm-after"),
    ("train-step", _t(_LOSS, _CLIP, _STEP, _RETURN), "wrong", "no-zero"),
    ("train-step", _t(_ZERO, _LOSS, "    torch.nn.utils.clip_grad_value_(model.parameters(), max_norm)\n", _STEP, "    return max_norm\n"), "wrong", "by-value"),
    ("train-step", _t(_LOSS, _ZERO, _CLIP, _STEP, _RETURN), "wrong", "zero-late"),
    ("train-step", _t(_ZERO, _LOSS, _CLIP.replace("max_norm)", "1.0)"), _STEP, _RETURN), "almost", None),
    ("train-step", _t(_ZERO, _LOSS, "    total = 0\n    for p in model.parameters():\n        total = total + (p.grad ** 2).sum()\n    norm = total ** 0.5\n"
                                    "    if norm > max_norm:\n        for p in model.parameters():\n            p.grad = p.grad * (max_norm / norm)\n",
                      _STEP, _RETURN), "correct", None),
]
