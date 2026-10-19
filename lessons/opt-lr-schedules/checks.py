# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson opt-lr-schedules: lr_at, the learning rate of a warmup and cosine schedule at a given step,
and WarmupCosine, a scheduler that sets it in an optimizer. And the picture: an MLP of 10 residual blocks for the
handwritten digits of QMNIST, one epoch of AdamW at a peak learning rate of 0.03, without warmup, with warmup, and with
warmup and cosine decay."""
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
BLOCKS = 10
WIDTH = 64
PEAK = 0.03     # 30 times Adam's usual 0.001: large enough that the first steps matter
WARMUP = 30


def lr_at(step: int, peak: float, warmup: int, total: int, minimum: float = 0.0) -> float:
    """The course's schedule: linear warmup over the first `warmup` steps, then cosine decay to `minimum` at `total`."""
    if step < warmup:
        return peak * (step + 1) / warmup
    if step >= total:
        return minimum
    r = (step - warmup) / (total - warmup)
    return minimum + 0.5 * (1 + math.cos(math.pi * r)) * (peak - minimum)


class DeepNet(nn.Module):
    """The network of nn-residual: an input layer, BLOCKS blocks h + second(relu(first(h))), and 10 logits."""

    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(784, WIDTH)
        self.first = nn.ModuleList([nn.Linear(WIDTH, WIDTH) for _ in range(BLOCKS)])
        self.second = nn.ModuleList([nn.Linear(WIDTH, WIDTH) for _ in range(BLOCKS)])
        self.out = nn.Linear(WIDTH, 10)

    def forward(self, x):
        h = torch.relu(self.inp(x))
        for a, b in zip(self.first, self.second):
            h = h + b(torch.relu(a(h)))
        return self.out(h)


def schedule(kind: str, total: int) -> list[float]:
    """The learning rate at each step: "constant" at PEAK, "warmup" then PEAK, or "cosine", warmup then cosine decay."""
    out = []
    for s in range(total):
        if kind == "constant":
            out.append(PEAK)
        elif kind == "warmup":
            out.append(PEAK * min(1.0, (s + 1) / WARMUP))
        else:
            out.append(lr_at(s, PEAK, WARMUP, total))
    return out


@functools.lru_cache(maxsize=3)
def run(kind: str):
    """One epoch of AdamW, with its default weight decay, with the schedule `kind`: the loss of each mini-batch, and
    the accuracy on the 1,000 test digits."""
    d = np.load(paths.ROOT / DATA)
    X = torch.tensor(d["train_images"], dtype=torch.float32) / 255
    y = torch.tensor(d["train_labels"], dtype=torch.int64)
    X_test = torch.tensor(d["test_images"], dtype=torch.float32) / 255
    y_test = torch.tensor(d["test_labels"], dtype=torch.int64)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)   # faster on these small layers, and the numbers no longer depend on the number of cores
    try:
        torch.manual_seed(0)
        model = DeepNet()
        opt = torch.optim.AdamW(model.parameters(), lr=PEAK)
        loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
        lrs = schedule(kind, len(loader))
        loss_fn = nn.CrossEntropyLoss()
        torch.manual_seed(2)
        losses = []
        for step, (xb, yb) in enumerate(loader):
            for group in opt.param_groups:
                group["lr"] = lrs[step]
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        with torch.no_grad():
            accuracy = (model(X_test).argmax(dim=1) == y_test).float().mean().item()
    finally:
        torch.set_num_threads(threads)
    return losses, accuracy


def _smooth(values, k=10):
    return [float(np.mean(values[max(0, i + 1 - k):i + 1])) for i in range(len(values))]


def schedule_picture(labels: dict) -> str:
    """Left: the three schedules. Right: the training loss with each, the mean of the last 10 mini-batches, on a log
    scale, because with the constant learning rate it shoots up."""
    from everything_ai import chart
    total = len(run("constant")[0])
    steps = list(range(1, total + 1))
    left = chart.Panel((0, total + 1), (0, PEAK * 1.12), title=labels["lr_title"], xlabel=labels["x"], ylabel=labels["lr"])
    right = chart.Panel((0, total + 1), (0.15, 100), title=labels["loss_title"], xlabel=labels["x"], ylabel=labels["loss"], ylog=True)
    styles = {"constant": (1.4, True), "warmup": (2.2, True), "cosine": (2.4, False)}
    for kind in ("constant", "warmup", "cosine"):
        width, dashed = styles[kind]
        left.line(steps, schedule(kind, total), "hyper", width, dashed=dashed)
        right.line(steps, _smooth(run(kind)[0]), "loss", width, dashed=dashed)
    left.text(2, PEAK * 1.05, labels["constant"], "hyper", "start")
    left.text(WARMUP / 2 + 2, PEAK * 0.45, labels["warmup"], "hyper", "start")
    left.text(total - 22, PEAK * 0.15, labels["cosine"], "hyper", "end")
    peak = max(range(total), key=lambda i: _smooth(run("constant")[0])[i])
    right.text(peak + 4, 40, labels["constant"], "loss", "start")
    right.text(total - 1, 0.2, labels["both"], "loss", "end")
    return chart.figure([left, right], alt=labels["alt"], width=330, height=290)


# The lesson is done when these are correct. "why-decay" is the choice question in the lesson.
REQUIRED = ["why-decay", "lr-at", "warmup-cosine"]

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
    if abs(x) < 1e-12:   # cos(pi / 2) is 6e-17 in floats, not 0: show it as 0, without float noise
        return "0"
    return np.format_float_positional(x, precision=5, unique=False, fractional=False, trim="-").rstrip(".")


def _show(v) -> str:
    if np.ndim(v) == 0:
        try:
            return _num(v)
        except (TypeError, ValueError):
            return repr(v)
    try:
        return "[" + ", ".join(_num(x) for x in v) + "]"
    except (TypeError, ValueError):
        return repr(v)


def _close(got, want) -> bool:
    try:
        return math.isclose(float(got), float(want), rel_tol=1e-6, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _close_list(got, want) -> bool:
    try:
        g, w = [float(x) for x in got], [float(x) for x in want]
    except (TypeError, ValueError):
        return False
    return len(g) == len(w) and all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12) for a, b in zip(g, w))


# --- exercise 2: the learning rate at a step --------------------------------------------------------------------------------

_L_HEAD = "import math\n\n\ndef lr_at(step, peak, warmup, total, minimum):\n"
_L_START = (_L_HEAD + "    # warmup: peak * (step + 1) / warmup for the first warmup steps; from total on: minimum;\n"
            "    # in between, cosine decay from peak to minimum\n    ...\n")


def _l(warm: str = "peak * (step + 1) / warmup", r: str = "(step - warmup) / (total - warmup)",
       decay: str = "minimum + 0.5 * (1 + math.cos(math.pi * r)) * (peak - minimum)", end: str = "    if step >= total:\n        return minimum\n") -> str:
    return (_L_HEAD + f"    if step < warmup:\n        return {warm}\n" + end + f"    r = {r}\n    return {decay}\n")


_L_CASES = [(0, 0.3, 50, 94, 0.0), (49, 0.3, 50, 94, 0.0), (50, 0.3, 50, 94, 0.0), (72, 0.3, 50, 94, 0.0), (93, 0.3, 50, 94, 0.0),
            (100, 0.3, 50, 94, 0.0), (5, 0.001, 10, 110, 0.0001), (60, 0.001, 10, 110, 0.0001), (500, 0.001, 10, 110, 0.0001)]

# --- exercise 3: a scheduler for an optimizer ---------------------------------------------------------------------------------

_W_HEAD = "import math\n\n\nclass WarmupCosine:\n"
_W_START = (_W_HEAD + "    def __init__(self, opt, peak, warmup, total):\n"
            "        # keep the numbers and a step counter at 0, and set the learning rate of step 0\n        ...\n\n"
            "    def step(self):\n        # the counter one up, and set the learning rate of that step in every group of the optimizer\n        ...\n")


def _w(warm: str = "self.peak * (s + 1) / self.warmup", count_first: bool = True, set_in_init: bool = True) -> str:
    body = ("    def lr(self, s):\n" + f"        if s < self.warmup:\n            return {warm}\n"
            "        if s >= self.total:\n            return 0.0\n        r = (s - self.warmup) / (self.total - self.warmup)\n"
            "        return 0.5 * (1 + math.cos(math.pi * r)) * self.peak\n\n"
            "    def set(self):\n        for group in self.opt.param_groups:\n            group[\"lr\"] = self.lr(self.t)\n\n")
    init = ("    def __init__(self, opt, peak, warmup, total):\n        self.opt = opt\n        self.peak = peak\n        self.warmup = warmup\n"
            "        self.total = total\n        self.t = 0\n" + ("        self.set()\n" if set_in_init else "") + "\n")
    step = ("    def step(self):\n        self.t = self.t + 1\n        self.set()\n" if count_first
            else "    def step(self):\n        self.set()\n        self.t = self.t + 1\n")
    return _W_HEAD + init + body + step


_RUN = ("import torch\n\nw = torch.zeros(3, requires_grad=True)\nb = torch.zeros(1, requires_grad=True)\n"
        "{opt} = torch.optim.SGD([{{\"params\": [w]}}, {{\"params\": [b]}}], lr=1.0)\nsched = WarmupCosine({opt}, {peak}, {warmup}, {total})\n"
        "lrs = []\nfor _ in range({steps}):\n    lrs.append({opt}.param_groups[{group}][\"lr\"])\n    {opt}.step()\n    sched.step()\nlrs")
_W_SCENARIOS = [
    _RUN.format(opt="opt", peak=0.1, warmup=2, total=6, steps=8, group=0),
    _RUN.format(opt="sgd", peak=0.5, warmup=4, total=12, steps=14, group=1),   # another name: a class that uses a global opt fails here
]

EXERCISES = {
    "lr-at": FunctionBox(
        name="lr-at",
        function="lr_at",
        start=_L_START,
        solution=_l(),
        visible=[(30, 0.002, 10, 50, 0.0)],
        hidden=_L_CASES,
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("warm-from-zero", wrong=_l(warm="peak * step / warmup"), state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: your warmup starts at 0, so step 0 changes nothing. Some libraries do it that way; this course counts the step itself, `peak * (step + 1) / warmup`, so that the last warmup step reaches `peak`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je warmup begint op 0, dus stap 0 verandert niets. Sommige libraries doen het zo; deze course telt de stap zelf mee, `peak * (step + 1) / warmup`, zodat de laatste warmup-stap op `peak` uitkomt."),
            Mistake("from-zero", wrong=_l(r="step / total"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your cosine counts from step 0 and over `total` steps. It starts after the warmup: `r = (step - warmup) / (total - warmup)`, 0 at the end of the warmup and 1 at `total`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je cosinus telt vanaf stap 0 en over `total` stappen. Hij begint na de warmup: `r = (step - warmup) / (total - warmup)`, 0 aan het eind van de warmup en 1 bij `total`."),
            Mistake("no-0.5", wrong=_l(decay="minimum + (1 + math.cos(math.pi * r)) * (peak - minimum)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `1 + math.cos(math.pi * r)` runs from 2 to 0, so your decay starts at twice the peak. Halve it: `0.5 * (1 + math.cos(math.pi * r))` runs from 1 to 0.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `1 + math.cos(math.pi * r)` loopt van 2 naar 0, dus je daling begint op twee keer de piek. Halveer het: `0.5 * (1 + math.cos(math.pi * r))` loopt van 1 naar 0."),
            Mistake("no-half", wrong=_l(decay="minimum + math.cos(math.pi * r) * (peak - minimum)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `math.cos(math.pi * r)` goes from 1 to −1, so your learning rate goes below `minimum`, even below 0. `0.5 * (1 + math.cos(math.pi * r))` goes from 1 to 0.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `math.cos(math.pi * r)` gaat van 1 naar −1, dus je learning rate zakt onder `minimum`, zelfs onder 0. `0.5 * (1 + math.cos(math.pi * r))` gaat van 1 naar 0."),
            Mistake("no-minimum", wrong=_l(decay="0.5 * (1 + math.cos(math.pi * r)) * peak", end="    if step >= total:\n        return 0.0\n"), state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: your decay ends at 0, not at `minimum`. Decay the part above the minimum: `minimum + 0.5 * (1 + math.cos(math.pi * r)) * (peak - minimum)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je daling eindigt op 0, niet op `minimum`. Laat het deel boven het minimum dalen: `minimum + 0.5 * (1 + math.cos(math.pi * r)) * (peak - minimum)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: the warmup for the first `warmup` steps, `minimum` from `total` on, and the cosine decay in between.",
                      "nl": "Vervang de `...` door de body van de functie: de warmup voor de eerste `warmup` stappen, `minimum` vanaf `total`, en de cosine decay daartussen."},
            "arguments": {"en": "The check calls `{call}`, with five numbers, and the `def` line of your function has another number of names between its parentheses. Keep `(step, peak, warmup, total, minimum)`.",
                          "nl": "De check roept `{call}` aan, met vijf getallen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(step, peak, warmup, total, minimum)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Which of the three parts does this step fall in? For the decay, work out `r` first: how far the step is between the end of the warmup and `total`, from 0 to 1.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. In welk van de drie delen valt deze stap? Reken voor de daling eerst `r` uit: hoe ver de stap is tussen het eind van de warmup en `total`, van 0 tot 1."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function handle the warmup steps, the steps from `total` on, and a `minimum` above 0?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gaat je functie goed met de warmup-stappen, de stappen vanaf `total`, en een `minimum` boven 0?"},
            "correct": {"en": "Your function is right in all {n} cases: warmup, cosine decay, and the minimum after the end.",
                        "nl": "Je functie klopt in alle {n} gevallen: warmup, cosine decay, en het minimum na het eind."},
        },
    ),
    "warmup-cosine": ClassBox(
        name="warmup-cosine",
        start=_W_START,
        solution=_w(),
        scenarios=_W_SCENARIOS,
        compare=_close_list,
        digits=5,
        first=[_DOTS],
        mistakes=[
            Mistake("counter-is-step", error=lambda e: isinstance(e, TypeError) and "'int' object is not callable" in str(e),
                    en="Python stops with an error: `sched.step()` calls a number. Is your counter called `self.step`? Then that number takes the place of the method `step`. Give the counter another name, such as `self.t`.",
                    nl="Python stopt met een fout: `sched.step()` roept een getal aan. Heet je teller `self.step`? Dan neemt dat getal de plaats in van de method `step`. Geef de teller een andere naam, zoals `self.t`."),
            Mistake("not-set-at-start", wrong=_w(set_in_init=False),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: the first step still uses the learning rate the optimizer was made with. Set the learning rate of step 0 already in `__init__`.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: de eerste stap gebruikt nog de learning rate waarmee de optimizer gemaakt is. Zet de learning rate van stap 0 al in `__init__`."),
            Mistake("count-after", wrong=_w(count_first=False),
                    en="With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: each learning rate comes one step late. `step()` is called after a step: count one up first, then set the learning rate of the next step.",
                    nl="Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: elke learning rate komt een stap te laat. `step()` wordt na een stap aangeroepen: tel eerst één op, en zet dan de learning rate van de volgende stap."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: in `__init__` the numbers, a counter at 0 and the learning rate of step 0; in `step` the counter one up and the learning rate of that step in every group.",
                      "nl": "Vervang de twee `...`: in `__init__` de getallen, een teller op 0 en de learning rate van stap 0; in `step` de teller één omhoog en de learning rate van die stap in elke groep."},
            "wrong": {"en": "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: the learning rate before each step. Is it `lr_at` of your counter, set as `group[\"lr\"]` in every group of `opt.param_groups`?",
                      "nl": "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: de learning rate vóór elke stap. Is het `lr_at` van je teller, gezet als `group[\"lr\"]` in elke groep van `opt.param_groups`?"},
            "almost": {"en": "Right for the first group. But with these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does every group of the optimizer get the learning rate, not only the first?",
                       "nl": "Goed voor de eerste groep. Maar met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Krijgt elke groep van de optimizer de learning rate, niet alleen de eerste?"},
            "correct": {"en": "Your scheduler sets the warmup and cosine learning rate before every step, in every group, as PyTorch's schedulers do.",
                        "nl": "Je scheduler zet vóór elke stap de learning rate van warmup en cosinus, in elke groep, zoals de schedulers van PyTorch doen."},
        },
    ),
}

CASES = [
    ("lr-at", _L_START, "empty", None),
    ("lr-at", _l(), "correct", None),
    ("lr-at", _l(warm="peak * (step + 1) / warmup", decay="minimum + (peak - minimum) * (1 + math.cos(r * math.pi)) / 2"), "correct", None),
    ("lr-at", _l(warm="peak * step / warmup"), "almost", "warm-from-zero"),
    ("lr-at", _l(r="step / total"), "wrong", "from-zero"),
    ("lr-at", _l(decay="minimum + math.cos(math.pi * r) * (peak - minimum)"), "wrong", "no-half"),
    ("lr-at", _l(decay="minimum + (1 + math.cos(math.pi * r)) * (peak - minimum)"), "wrong", "no-0.5"),
    ("lr-at", _l(decay="0.5 * (1 + math.cos(math.pi * r)) * peak", end="    if step >= total:\n        return 0.0\n"), "almost", "no-minimum"),
    ("lr-at", "def lr_at(step, peak):\n    return peak\n", "error", "arguments"),
    ("warmup-cosine", _W_START, "empty", None),
    ("warmup-cosine", _w(), "correct", None),
    ("warmup-cosine", _w(set_in_init=False), "wrong", "not-set-at-start"),
    ("warmup-cosine", _w(count_first=False), "wrong", "count-after"),
    ("warmup-cosine", re.sub(r"self\.t\b", "self.step", _w()), "error", "counter-is-step"),
]
