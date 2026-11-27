# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-reproducibility: same run, same result. A training run uses random numbers for the weights
at the start and for the order of the examples; with seeds, the same versions and the same hardware it comes out the
same again, bit for bit. Runs with other seeds still land apart, so two settings are compared by the spread of several
seeds each. The learner draws a new order every epoch from a generator of its own, as a DataLoader does, and decides
from ten seeds each whether one setting beats another; the demo goes through the ten runs of the digits MLP.

The runs are the course's own: the MLP of torch-mlp-digits on data/qmnist-small.npz, trained on a CPU with PyTorch
2.14.1 (see TORCH and train)."""
import ast
import html

import torch

from everything_ai import check, paths, yours
from everything_ai.check import FunctionBox, Mistake

LESSON = "sys-reproducibility"
LANGS = ("en", "nl")

# ======== the course's runs ===============================================================================================
# The digits MLP of torch-mlp-digits (784 pixels, a hidden layer, 10 logits), 10 epochs of SGD with learning rate 0.5 on
# mini-batches of 64 from a DataLoader that shuffles, after torch.manual_seed(seed); what `train` below does. Measured on
# a CPU with PyTorch 2.14.1, the version in the course's uv.lock; the test trains a few of them again.
DATA = "data/qmnist-small.npz"
TORCH = "2.14.1"
SEEDS = list(range(10))
RUNS_64 = [932, 935, 929, 926, 923, 933, 931, 920, 919, 930]     # test digits right of 1,000, 64 hidden neurons
RUNS_128 = [925, 932, 932, 935, 920, 928, 938, 930, 931, 927]    # the same with 128 hidden neurons
RUNS_LINEAR = [894, 891, 896, 896, 890, 884, 896, 893, 885, 890]  # a linear model, nn.Linear(784, 10)
# The mean loss over the mini-batches of each epoch, 64 hidden neurons, one row per seed.
LOSSES_64 = [
    [0.7501, 0.2931, 0.2105, 0.1595, 0.1199, 0.0932, 0.0879, 0.0608, 0.0475, 0.0388],
    [0.7543, 0.3032, 0.2149, 0.1598, 0.1262, 0.1015, 0.0797, 0.0631, 0.0562, 0.0418],
    [0.7899, 0.2944, 0.2133, 0.1698, 0.1276, 0.1081, 0.084, 0.0668, 0.0515, 0.0411],
    [0.7696, 0.3104, 0.2172, 0.1645, 0.1324, 0.1046, 0.0808, 0.0667, 0.053, 0.0419],
    [0.7412, 0.3125, 0.2087, 0.1635, 0.127, 0.0936, 0.077, 0.0619, 0.0499, 0.0386],
    [0.7645, 0.3014, 0.2119, 0.1617, 0.1263, 0.0963, 0.0793, 0.0632, 0.05, 0.0381],
    [0.7775, 0.2959, 0.2113, 0.1606, 0.1349, 0.1004, 0.0817, 0.0654, 0.051, 0.0394],
    [0.7392, 0.3096, 0.2078, 0.1564, 0.1229, 0.0976, 0.079, 0.0608, 0.0521, 0.038],
    [0.7353, 0.309, 0.213, 0.1672, 0.1299, 0.105, 0.0857, 0.0661, 0.0496, 0.0416],
    [0.7791, 0.3126, 0.2139, 0.1584, 0.129, 0.0994, 0.0795, 0.0634, 0.0513, 0.0371],
]
# The labels of the first 8 digits in the first mini-batch of each seed.
FIRST_LABELS = [[1, 7, 5, 6, 5, 1, 2, 5], [3, 1, 6, 0, 8, 4, 7, 7], [8, 8, 7, 7, 3, 1, 7, 1], [7, 7, 8, 0, 6, 6, 5, 2],
                [2, 2, 3, 8, 0, 1, 1, 9], [1, 6, 4, 8, 9, 4, 5, 7], [1, 0, 0, 9, 5, 9, 3, 7], [4, 0, 8, 3, 4, 3, 3, 6],
                [4, 7, 8, 5, 5, 2, 5, 5], [4, 1, 1, 6, 7, 4, 5, 4]]
# Two generators, one per source: torch.manual_seed for the weights, a torch.Generator for the order. 64 hidden neurons.
INIT_ONLY = [923, 931, 919, 923, 918, 920, 923, 923, 919, 929]    # weights with seeds 0 to 9, order always seed 0
ORDER_ONLY = [923, 929, 929, 930, 924, 924, 927, 932, 922, 920]   # weights always seed 0, order with seeds 0 to 9


def train(seed: int, hidden: int = 64, order_seed: int | None = None, epochs: int = 10):
    """One run as the course measured it: test digits right of 1,000, the mean loss of each epoch (4 decimals), and the
    labels of the first 8 digits of the first mini-batch. hidden=0 is the linear model. Without `order_seed`, the
    DataLoader takes its order from PyTorch's global generator; with one, from a generator of its own."""
    import numpy as np
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    d = np.load(paths.ROOT / DATA)
    x = torch.tensor(d["train_images"], dtype=torch.float32) / 255
    y = torch.tensor(d["train_labels"], dtype=torch.int64)
    x_test = torch.tensor(d["test_images"], dtype=torch.float32) / 255
    y_test = torch.tensor(d["test_labels"], dtype=torch.int64)

    class DigitsMLP(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden = nn.Linear(784, hidden)
            self.out = nn.Linear(hidden, 10)

        def forward(self, x):
            return self.out(torch.relu(self.hidden(x)))

    torch.manual_seed(seed)
    model = DigitsMLP(hidden) if hidden else nn.Linear(784, 10)
    rng = None
    if order_seed is not None:
        rng = torch.Generator()
        rng.manual_seed(order_seed)
    loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True, generator=rng)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    loss_fn = nn.CrossEntropyLoss()
    losses, first = [], None
    for _ in range(epochs):
        total = 0.0
        for xb, yb in loader:
            if first is None:
                first = yb[:8].tolist()
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        losses.append(round(total / len(loader), 4))
    model.eval()
    with torch.no_grad():
        right = int((model(x_test).argmax(dim=1) == y_test).sum())
    return right, losses, first


def mean(xs: list) -> float:
    return sum(xs) / len(xs)


SPREAD_64 = max(RUNS_64) - min(RUNS_64)                     # 16: from 919 to 935
WINS_64 = sum(a > b for a, b in zip(RUNS_64, RUNS_128))      # the seeds where 64 neurons beat 128: five of ten


# ======== the exercises' functions ========================================================================================

def orders(n, epochs, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    found = []
    for epoch in range(epochs):
        order = torch.randperm(n, generator=g)
        found.append(order.tolist())
    return found


def better(a, b):
    return min(b) > max(a)


# ======== words and drawing ===============================================================================================

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    out = f"{x:,.{digits}f}"
    out = out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out
    return out.replace("-", "−")


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text_).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _note(text_: str, style: str = _UI) -> str:
    return f'<div style="{style}">{_marked(text_)}</div>'


def _t(x: float, y: float, text_: str, size: float = 12.5, weight: int = 400, fill: str = "var(--ea-ink)",
       anchor: str = "start") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="font-size:{size}px;font-weight:{weight};'
            f'fill:{fill}">{html.escape(text_)}</text>')


def _mono(x: float, y: float, text_: str, size: float = 10.5, fill: str = "var(--ea-ink)", anchor: str = "start",
          weight: int = 400) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="font-family:var(--ea-font-code);'
            f'font-variant-ligatures:none;font-size:{size}px;font-weight:{weight};fill:{fill}">{html.escape(text_)}</text>')


def _line(x1: float, y1: float, x2: float, y2: float, ink: str = "var(--ea-line)", width: float = 1.0) -> str:
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'style="stroke:{ink};stroke-width:{width}"/>')


def _rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "") -> str:
    edge = f";stroke:{stroke};stroke-width:1" if stroke else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" style="fill:{fill}{edge}"/>'


def _box(x: float, y: float, w: float, h: float, stroke: str = "var(--ea-ink-3)", width: float = 1.2) -> str:
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" '
            f'style="fill:var(--ea-page);stroke:{stroke};stroke-width:{width}"/>')


def _dot(x: float, y: float, r: float, fill: str, stroke: str = "var(--ea-page)") -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" style="fill:{fill};stroke:{stroke};stroke-width:1"/>'


def _arrow(x1: float, y1: float, x2: float, y2: float, ink: str = "var(--ea-ink-2)", width: float = 1.3) -> str:
    """A line from (x1, y1) to (x2, y2) with a small head at its end."""
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    h1 = (x2 - 6 * math.cos(a - 0.45), y2 - 6 * math.sin(a - 0.45))
    h2 = (x2 - 6 * math.cos(a + 0.45), y2 - 6 * math.sin(a + 0.45))
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" style="stroke:{ink};stroke-width:{width}"/>'
            f'<path d="M{h1[0]:.1f},{h1[1]:.1f} L{x2:.1f},{y2:.1f} L{h2[0]:.1f},{h2[1]:.1f}" '
            f'style="fill:none;stroke:{ink};stroke-width:{width}"/>')


def _svg(height: float, alt: str, parts: list) -> str:
    return (f'<svg viewBox="0 0 330 {height:.0f}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">'
            f'{"".join(parts)}</svg>')


# ======== the picture of the idea =========================================================================================

IDEA_TEXT = {
    "en": {"title": "What a training run depends on",
           "rows": [("Seed of the weights", "random numbers before training"),
                    ("Seed of the order", "a new order of examples each epoch"),
                    ("Versions", "the same code runs: uv.lock"),
                    ("Hardware", "CPU or GPU: the order of additions")],
           "model": ("trained", "model"),
           "same": "On a CPU, the same four give the same model, bit for bit.",
           "other": "Change the seed, and another model comes out.",
           "alt": "Four boxes with arrows to one trained model: the seed of the weights, the seed of the order of the examples, the versions in uv.lock, and the hardware, which sets the order of additions. On a CPU, the same four give the same model, bit for bit; change the seed, and another model comes out."},
    "nl": {"title": "Waar een training-run van afhangt",
           "rows": [("Seed van de weights", "willekeurige getallen aan het begin"),
                    ("Seed van de volgorde", "elke epoch een nieuwe volgorde"),
                    ("Versies", "dezelfde code draait: uv.lock"),
                    ("Hardware", "CPU of GPU: de volgorde van optellen")],
           "model": ("getraind", "model"),
           "same": "Op een CPU geven dezelfde vier bit voor bit hetzelfde model.",
           "other": "Verander de seed, en er komt een ander model uit.",
           "alt": "Vier vakken met pijlen naar één getraind model: de seed van de weights, de seed van de volgorde van de voorbeelden, de versies in uv.lock, en de hardware, met de volgorde van optellen. Op een CPU geven dezelfde vier bit voor bit hetzelfde model; verander de seed, en er komt een ander model uit."},
}


def idea_picture(lang: str) -> str:
    """Four things a run depends on, each with an arrow to the trained model."""
    w = IDEA_TEXT[lang]
    parts = [_t(0, 14, w["title"], 12.5, 600)]
    top, gap, h, bw = 30.0, 46.0, 38.0, 214.0
    mx, mw = 250.0, 78.0
    c = top + (3 * gap + h) / 2   # the middle of the four rows
    for k, (label, detail) in enumerate(w["rows"]):
        y = top + k * gap
        parts.append(_box(0, y, bw, h))
        parts.append(_t(8, y + 15, label, 11, 600))
        parts.append(_t(8, y + 30, detail, 9.5, 400, "var(--ea-ink-2)"))
        parts.append(_arrow(bw + 2, y + h / 2, mx - 2, c + (k - 1.5) * 10))
    parts.append(_box(mx, c - 32, mw, 64, "var(--ea-param)", 1.8))
    parts.append(_t(mx + mw / 2, c - 3, w["model"][0], 11, 600, "var(--ea-param-text)", "middle"))
    parts.append(_t(mx + mw / 2, c + 12, w["model"][1], 11, 600, "var(--ea-param-text)", "middle"))
    y = top + 4 * gap + 10
    parts.append(_t(0, y, w["same"], 10, 400, "var(--ea-ink)"))
    parts.append(_t(0, y + 16, w["other"], 10, 400, "var(--ea-ink-2)"))
    height = y + 24
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(w["alt"])}" style="margin:0.25rem 0 0.75rem;'
            f'max-width:min(100%,330px)"><svg viewBox="0 0 330 {height:.0f}" aria-hidden="true" style="{_SVG}">'
            f'{"".join(parts)}</svg></figure>')


# ======== the pictures of the learner's work ==============================================================================

PICTURE_SECONDS = 2   # what the learner's function gets for all the calls of one picture together
_STOPPED = object()   # what _call gives for a function that stopped with an error


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _call(make_):
    """make_() within a time limit; _STOPPED when it stops with an error or runs out of time."""
    try:
        return check.with_time_limit(make_, PICTURE_SECONDS)
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _rows(v) -> list | None:
    """v as a list of orders, each a list of whole numbers, or None: never anything else."""
    if not isinstance(v, (list, tuple)) or len(v) > 6:
        return None
    out = []
    for row in v:
        if not isinstance(row, (list, tuple)) or len(row) > 12:
            return None
        if not all(isinstance(x, int) and not isinstance(x, bool) and -99 <= x <= 999 for x in row):
            return None
        out.append(list(row))
    return out


def _repeats(rows: list) -> bool:
    """True when some epoch has the same order as the epoch before it."""
    return any(rows[i] == rows[i - 1] for i in range(1, len(rows)))


def orders_svg(runs: list, lang: str, alt: str) -> str:
    """Three calls of the learner's function, each a block of epochs with a cell per example, and what each call
    should show: a new order every epoch, the same orders again with the same seed, other orders with another seed."""
    w = _YOURS_TEXT[lang]["orders"]
    first = runs[0]
    statuses = [(w["repeat"], False) if _repeats(first) else (w["new"], True),
                (w["same"], True) if runs[1] == first else (w["differs"], False),
                (w["copy"], False) if runs[2] == first else (w["other"], True)]
    parts = []
    y = 4.0
    for k, (rows, (status, good)) in enumerate(zip(runs, statuses)):
        parts.append(_t(0, y + 12, w["runs"][k], 11, 600))
        parts.append(_t(328, y + 12, status, 9.5, 600, "var(--ea-ink-2)" if good else "var(--ea-loss-text)", "end"))
        y += 20
        for e, row in enumerate(rows):
            parts.append(_t(0, y + 13, w["epoch"].format(e=e + 1), 9.5, 400, "var(--ea-ink-2)"))
            step = 30.0 if len(row) <= 8 else 264.0 / len(row)
            differs = k == 1 and (e >= len(first) or first[e] != row)
            for i, x in enumerate(row):
                bad = differs and (e >= len(first) or i >= len(first[e]) or first[e][i] != x)
                parts.append(_rect(62 + i * step, y + 1, step - 4, 17, "var(--ea-inset)",
                                   "var(--ea-loss)" if bad else "var(--ea-line)"))
                parts.append(_mono(62 + i * step + (step - 4) / 2, y + 13.5, str(x), 10.5,
                                   "var(--ea-loss-text)" if bad else "var(--ea-ink)", "middle"))
            y += 21
        if not rows:
            parts.append(_t(62, y + 13, w["empty"], 9.5, 400, "var(--ea-loss-text)"))
            y += 21
        y += 12
    return _svg(y, alt, parts)


BETTER_PAIRS = [("pair_64_128", "n64", RUNS_64, "n128", RUNS_128), ("pair_linear_64", "linear", RUNS_LINEAR, "n64", RUNS_64)]


def better_svg(rows: list, lang: str, alt: str) -> str:
    """For each pair of settings, the ten runs of each as dots on an axis of its own, with a band from the worst to
    the best, and what the learner's function says, in the color of mistakes when it is not right."""
    w = _YOURS_TEXT[lang]["better"]
    X0, X1 = 100.0, 322.0
    parts = []
    y = 4.0
    for title, (name_a, runs_a), (name_b, runs_b), said in rows:
        LOW = (min(runs_a + runs_b) - 2) // 5 * 5           # the axis runs over whole fives around both settings
        HIGH = -(-(max(runs_a + runs_b) + 2) // 5) * 5
        tick = 5 if HIGH - LOW <= 30 else 10

        def x_of(v: float, low=LOW, high=HIGH) -> float:
            return X0 + (min(max(v, low), high) - low) / (high - low) * (X1 - X0)

        parts.append(_t(0, y + 12, title, 11, 600))
        y += 24
        for letter, name_, runs in (("A", name_a, runs_a), ("B", name_b, runs_b)):
            parts.append(_t(0, y + 4, f"{letter}: {name_}", 9.5, 400, "var(--ea-ink-2)"))
            parts.append(_rect(x_of(min(runs)) - 4, y - 6, x_of(max(runs)) - x_of(min(runs)) + 8, 12, "var(--ea-inset)",
                               "var(--ea-line)"))
            for v in runs:
                parts.append(_dot(x_of(v), y, 2.8, "var(--ea-ink-2)"))
            y += 20
        for v in range(LOW, HIGH + 1, tick):
            parts.append(_line(x_of(v), y - 8, x_of(v), y - 4, "var(--ea-ink-3)"))
            parts.append(_t(x_of(v), y + 6, number(v, 0, lang), 8.5, 400, "var(--ea-ink-2)", "middle"))
        right = better(runs_a, runs_b)
        ink = "var(--ea-ink)" if said is right else "var(--ea-loss-text)"
        parts.append(_t(1, y + 26, w["said"].format(v=repr(said)) if isinstance(said, bool) else w["odd"], 11, 600, ink))
        y += 46
    return _svg(y, alt, parts)


_YOURS_TEXT = {
    "en": {"orders": {"title": "Your orders: seed 0 twice, then seed 1",
                      "intro": "Three calls of your `orders(8, 3, seed)`. Each row is one epoch: an order of the examples 0 to 7.",
                      "runs": ["call 1: seed 0", "call 2: seed 0 again", "call 3: seed 1"], "epoch": "epoch {e}",
                      "new": "a new order every epoch", "repeat": "the same order again",
                      "same": "the same as call 1", "differs": "differs from call 1",
                      "other": "other orders than seed 0", "copy": "the same as seed 0", "empty": "no orders",
                      "alt": "Three blocks of orders of the examples 0 to 7, one row per epoch: your orders with seed 0, with seed 0 again, and with seed 1.",
                      "right": "In these three calls, the second repeats the first exactly, every epoch has an order of its own, and seed 1 gives other orders: a run you can repeat, with a new mix of examples each epoch.",
                      "wrong": "Not every call is right yet: look at the labels and cells in the color of mistakes.",
                      "other_orders": "Every call behaves as it should, but the orders differ from the course's: the check above shows where.",
                      "fail": "Your `orders` stops with an error, takes too long, or does not give a list of orders, so nothing can be drawn yet."},
           "better": {"title": "Your better on two pairs of settings",
                      "intro": "Each dot is one run, one of ten seeds: test digits right of 1,000. The band runs from the worst run to the best. Your `better(a, b)` says whether B beats A.",
                      "pair_64_128": "Are 128 neurons better than 64?", "pair_linear_64": "Are 64 neurons better than a linear model?",
                      "n64": "64 neurons", "n128": "128 neurons", "linear": "linear model",
                      "said": "your function: {v}", "odd": "your function: not True or False",
                      "alt": "Two pairs of settings, each with ten runs as dots on an axis of test digits right, a band from the worst run to the best, and what your function says.",
                      "right": "On these two pairs, your function says `True` only where the bands do not touch: 64 neurons against the linear model. For 64 against 128 neurons, the bands overlap, so ten seeds each do not settle it.",
                      "wrong": "What your function says is not yet right for every pair: compare it with where the two bands lie.",
                      "fail": "Your `better` stops with an error or takes too long, so nothing can be drawn yet."}},
    "nl": {"orders": {"title": "Je volgordes: twee keer seed 0, dan seed 1",
                      "intro": "Drie aanroepen van je `orders(8, 3, seed)`. Elke regel is één epoch: een volgorde van de voorbeelden 0 tot en met 7.",
                      "runs": ["aanroep 1: seed 0", "aanroep 2: weer seed 0", "aanroep 3: seed 1"], "epoch": "epoch {e}",
                      "new": "elke epoch een nieuwe volgorde", "repeat": "weer dezelfde volgorde",
                      "same": "hetzelfde als aanroep 1", "differs": "anders dan aanroep 1",
                      "other": "andere volgordes dan seed 0", "copy": "hetzelfde als seed 0", "empty": "geen volgordes",
                      "alt": "Drie blokken met volgordes van de voorbeelden 0 tot en met 7, één regel per epoch: je volgordes met seed 0, nog een keer met seed 0, en met seed 1.",
                      "right": "In deze drie aanroepen herhaalt de tweede de eerste precies, heeft elke epoch een eigen volgorde, en geeft seed 1 andere volgordes: een run die je kunt herhalen, met elke epoch een nieuwe mix van voorbeelden.",
                      "wrong": "Nog niet elke aanroep klopt: kijk naar de labels en vakjes in de kleur van fouten.",
                      "other_orders": "Elke aanroep gedraagt zich zoals het hoort, toch wijken je volgordes af van de volgordes van de course: de check hierboven laat zien waar.",
                      "fail": "Je `orders` stopt met een fout, duurt te lang of geeft geen lijst met volgordes, dus is er nog geen afbeelding."},
           "better": {"title": "Je better bij twee paren instellingen",
                      "intro": "Elke stip is één run, één van tien seeds: goed gelezen testcijfers van de 1.000. De band loopt van de slechtste run tot de beste. Je `better(a, b)` zegt of B beter is dan A.",
                      "pair_64_128": "Zijn 128 neuronen beter dan 64?", "pair_linear_64": "Zijn 64 neuronen beter dan een lineair model?",
                      "n64": "64 neuronen", "n128": "128 neuronen", "linear": "lineair model",
                      "said": "je functie: {v}", "odd": "je functie: niet True of False",
                      "alt": "Twee paren instellingen, elk met tien runs als stippen op een as met goed gelezen testcijfers, een band van de slechtste run tot de beste, en wat je functie zegt.",
                      "right": "Bij deze twee paren zegt je functie alleen `True` waar de banden elkaar niet raken: 64 neuronen tegen het lineaire model. Bij 64 tegen 128 neuronen overlappen de banden, dus beslissen tien seeds per instelling het niet.",
                      "wrong": "Wat je functie zegt klopt nog niet voor elk paar: vergelijk het met waar de twee banden liggen.",
                      "fail": "Je `better` stopt met een fout of duurt te lang, dus is er nog geen afbeelding."}},
}


def orders_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["orders"]
    fn = _fn(seen)
    runs = []
    for seed in (0, 0, 1):
        v = _call(lambda s=seed: fn(8, 3, s)) if fn is not None else _STOPPED
        rows = _rows(v) if v is not _STOPPED else None
        if rows is None:
            return yours.figure(w["title"], _note(w["intro"]) + _note(w["fail"], _LOSS_UI), alt=w["title"])
        runs.append(rows)
    right = runs[0] == runs[1] == orders(8, 3, 0) and runs[2] == orders(8, 3, 1)
    behaves = not _repeats(runs[0]) and runs[1] == runs[0] and runs[2] != runs[0]   # every status in ink
    note = w["right"] if right else w["other_orders"] if behaves else w["wrong"]
    return yours.figure(w["title"], _note(w["intro"]) + orders_svg(runs, lang, w["alt"]) +
                        _note(note, _UI if right else _LOSS_UI), alt=w["title"])


def better_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["better"]
    fn = _fn(seen)
    rows = []
    for key, name_a, runs_a, name_b, runs_b in BETTER_PAIRS:
        v = _call(lambda a=runs_a, b=runs_b: fn(list(a), list(b))) if fn is not None else _STOPPED
        if v is _STOPPED:
            return yours.figure(w["title"], _note(w["intro"]) + _note(w["fail"], _LOSS_UI), alt=w["title"])
        said = bool(v) if type(v).__name__ == "bool_" else v   # a NumPy bool counts as True or False
        rows.append((w[key], (w[name_a], runs_a), (w[name_b], runs_b), said))
    right = all(said is better(a[1], b[1]) for _, a, b, said in rows)
    return yours.figure(w["title"], _note(w["intro"]) + better_svg(rows, lang, w["alt"]) +
                        _note(w["right"] if right else w["wrong"], _UI if right else _LOSS_UI), alt=w["title"])


# ======== the demo: ten seeds of the digits MLP ===========================================================================

DEMO_TEXT = {
    "en": {"title": "Seed {s}", "loss": "loss per epoch, 64 neurons, all ten seeds", "epoch": "epoch",
           "close": "the ten curves lie almost on top of each other",
           "right": "test digits right, of 1,000, all ten seeds", "n64": "64 neurons: {v}", "n128": "128 neurons: {v}",
           "first": "the first mini-batch starts with the digits",
           "alt": "Ten loss curves of the digits MLP with 64 neurons, one per seed, with seed {s} drawn thick. Below, the test digits right of the ten runs with 64 and with 128 neurons: with seed {s}, {a} and {b}. The first mini-batch starts with the digits {first}.",
           "says": {
               "win": "With seed **{s}**, the MLP with 64 neurons reads **{a}** of the 1,000 test digits right, and the MLP with 128 neurons **{b}**. With this seed alone, {w} neurons would look better, by {d} digits.",
               "tie": "With seed **{s}**, the MLP with 64 neurons and the MLP with 128 neurons both read **{a}** of the 1,000 test digits right. With this seed alone, neither would look better.",
           }},
    "nl": {"title": "Seed {s}", "loss": "loss per epoch, 64 neuronen, alle tien seeds", "epoch": "epoch",
           "close": "de tien curves liggen bijna op elkaar",
           "right": "goed gelezen testcijfers van de 1.000, alle tien seeds", "n64": "64 neuronen: {v}", "n128": "128 neuronen: {v}",
           "first": "de eerste mini-batch begint met de cijfers",
           "alt": "Tien loss-curves van het cijfer-MLP met 64 neuronen, één per seed, met seed {s} dik getekend. Daaronder de goed gelezen testcijfers van de tien runs met 64 en met 128 neuronen: met seed {s} zijn dat er {a} en {b}. De eerste mini-batch begint met de cijfers {first}.",
           "says": {
               "win": "Met seed **{s}** leest het MLP met 64 neuronen **{a}** van de 1.000 testcijfers goed, en het MLP met 128 neuronen **{b}**. Met deze seed alleen zouden {w} neuronen beter lijken, met {d} cijfers verschil.",
               "tie": "Met seed **{s}** lezen het MLP met 64 neuronen en het MLP met 128 neuronen allebei **{a}** van de 1.000 testcijfers goed. Met deze seed alleen lijkt geen van beide beter.",
           }},
}


def say(seed: int, lang: str) -> str:
    w = DEMO_TEXT[lang]["says"]
    a, b = RUNS_64[seed], RUNS_128[seed]
    if a == b:
        return w["tie"].format(s=seed, a=a)
    return w["win"].format(s=seed, a=a, b=b, w=64 if a > b else 128, d=abs(a - b))


def frame(value: float, lang: str) -> tuple[str, str]:
    """The loss curves of the ten seeds with the chosen one drawn thick, the test digits right of the ten runs with 64
    and with 128 neurons with the chosen seed's run filled in, and the first labels of its first mini-batch."""
    s = min(max(int(round(value)), 0), len(SEEDS) - 1)
    w = DEMO_TEXT[lang]
    parts = [_t(0, 14, w["title"].format(s=s), 12, 600)]
    # the loss curves
    parts.append(_t(0, 36, w["loss"], 9.5, 400, "var(--ea-ink-2)"))
    X0, X1, Y0, Y1, TOP = 30.0, 322.0, 136.0, 48.0, 0.8

    def xy(e: int, v: float) -> tuple[float, float]:
        return X0 + e * (X1 - X0) / 9, Y0 - min(v, TOP) / TOP * (Y0 - Y1)

    for v in (0.0, 0.4, 0.8):
        _, y = xy(0, v)
        parts.append(_line(X0, y, X1, y, "var(--ea-line)", 0.8))
        parts.append(_t(X0 - 6, y + 3, number(v, 1, lang), 8.5, 400, "var(--ea-ink-2)", "end"))
    for k in [s] + [i for i in SEEDS if i != s]:   # the chosen seed thick below, the other nine thin on top of it
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(e, v) for e, v in enumerate(LOSSES_64[k])))
        ink, width = ("var(--ea-loss)", 3.4) if k == s else ("var(--ea-ink-3)", 0.7)
        parts.append(f'<polyline points="{pts}" style="fill:none;stroke:{ink};stroke-width:{width};stroke-linejoin:round"/>')
    parts.append(_t(X1, Y1 + 14, w["close"], 9, 400, "var(--ea-ink-2)", "end"))
    for e in range(10):
        x, _ = xy(e, 0)
        parts.append(_t(x, Y0 + 13, str(e + 1), 8.5, 400, "var(--ea-ink-2)", "middle"))
    parts.append(_t(0, Y0 + 13, w["epoch"], 8.5, 400, "var(--ea-ink-2)"))
    # the test digits right
    y = Y0 + 40
    parts.append(_t(0, y, w["right"], 9.5, 400, "var(--ea-ink-2)"))
    A0, A1, LO, HI = 110.0, 322.0, 915, 940

    def ax(v: float) -> float:
        return A0 + (v - LO) / (HI - LO) * (A1 - A0)

    y += 22
    for key, runs in (("n64", RUNS_64), ("n128", RUNS_128)):
        parts.append(_t(0, y + 4, w[key].format(v=runs[s]), 9.5, 600))
        parts.append(_rect(ax(min(runs)) - 4, y - 6, ax(max(runs)) - ax(min(runs)) + 8, 12, "var(--ea-inset)", "var(--ea-line)"))
        for k, v in enumerate(runs):
            if k != s:
                parts.append(_dot(ax(v), y, 2.8, "var(--ea-ink-3)"))
        parts.append(_dot(ax(runs[s]), y, 4.4, "var(--ea-ink)"))
        y += 22
    for v in range(LO, HI + 1, 5):
        parts.append(_line(ax(v), y - 10, ax(v), y - 6, "var(--ea-ink-3)"))
        parts.append(_t(ax(v), y + 4, str(v), 8.5, 400, "var(--ea-ink-2)", "middle"))
    # the first labels of the first mini-batch
    y += 28
    parts.append(_t(0, y, w["first"], 9.5, 400, "var(--ea-ink-2)"))
    first = "  ".join(str(d) for d in FIRST_LABELS[s])
    parts.append(_mono(0, y + 18, first, 12, "var(--ea-ink)", weight=600))
    alt = w["alt"].format(s=s, a=RUNS_64[s], b=RUNS_128[s], first=", ".join(str(d) for d in FIRST_LABELS[s]))
    return _svg(y + 26, alt, parts), say(s, lang)


# ======== the exercises ===================================================================================================

def _dots_left(got, text_) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Constant) and node.value is Ellipsis for node in ast.walk(tree))


def _global(got, text_) -> bool:
    """`torch.randperm` without a generator of its own: it draws from PyTorch's global generator."""
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name == "randperm" and not any(k.arg == "generator" for k in node.keywords):
                return True
    return False


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Can you replace it with your own code?",
                nl="Er staat nog een `...` in je code. Kun je de `...` vervangen door je eigen code?")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then begins with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


class _Listed(list):
    """A list of the lesson that the check shows by its name: C.RUNS_64."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self._name = name

    def __repr__(self) -> str:
        return self._name

    def __deepcopy__(self, memo):
        return list(self)


RUNS_64_CASE = _Listed(RUNS_64, "C.RUNS_64")
RUNS_128_CASE = _Listed(RUNS_128, "C.RUNS_128")

# exercise 1: orders
_O_HEAD = ("import torch\n\n\n"
           "def orders(n, epochs, seed):\n"
           "    g = torch.Generator()\n"
           "    g.manual_seed(seed)\n"
           "    found = []\n"
           "    for epoch in range(epochs):\n")
_O_END = ("        found.append(order.tolist())\n"
          "    return found\n")


def _o(line: str) -> str:
    return _O_HEAD + f"        order = {line}\n" + _O_END


_O_START = _o("...")
_O = _o("torch.randperm(n, generator=g)")
_O_GLOBAL = _o("torch.randperm(n)")
_O_RESEED = _o("torch.randperm(n, generator=torch.Generator().manual_seed(seed))")
_O_EIGHT = _o("torch.randperm(8, generator=g)")

# exercise 2: better
_B_START = "def better(a, b):\n    return ...\n"
_B = "def better(a, b):\n    return min(b) > max(a)\n"
_B_MEANS = "def better(a, b):\n    return sum(b) / len(b) > sum(a) / len(a)\n"
_B_BEST = "def better(a, b):\n    return max(b) > max(a)\n"
_B_WORST = "def better(a, b):\n    return min(b) > min(a)\n"
_B_TIE = "def better(a, b):\n    return min(b) >= max(a)\n"
_B_SWAPPED = "def better(a, b):\n    return min(a) > max(b)\n"

EXERCISES = {
    "orders": FunctionBox(
        name="orders",
        function="orders",
        start=_O_START,
        solution=_O,
        visible=[(8, 2, 0)],
        hidden=[(8, 3, 1), (5, 2, 7), (1, 2, 0), (0, 2, 3), (6, 0, 5), (10, 1, 2026)],
        first=[_DOTS],
        picture=orders_picture,
        mistakes=[
            Mistake("reseed", wrong=_O_RESEED,
                    en="`{call}` gives {got}, and it should give {want}: every epoch gets the same order, because each epoch starts a generator over at the same seed. Which generator has already gone on after the first epoch?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke epoch krijgt dezelfde volgorde, omdat elke epoch een generator opnieuw bij dezelfde seed laat beginnen. Welke generator is na de eerste epoch al verder gegaan?"),
            Mistake("positional", error=lambda e: isinstance(e, TypeError) and "randperm()" in str(e) and "Generator" in str(e),
                    en="`torch.randperm` takes a generator only as a value with a name, not on its own after `n`. Which name goes in front of `g`, as with a DataLoader?",
                    nl="`torch.randperm` neemt een generator alleen aan als waarde met een naam, niet los na `n`. Welke naam komt voor `g`, net als bij een DataLoader?"),
            Mistake("global", matches=_global,
                    en="`{call}` gives {got}, and it should give {want}: `torch.randperm` draws from PyTorch's global generator here, so `g` and its seed are never used, and the next call gives other orders. How does `torch.randperm` get a generator of your choice?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `torch.randperm` neemt hier getallen van de globale generator van PyTorch, dus worden `g` en de seed nooit gebruikt, en geeft de volgende aanroep andere volgordes. Hoe geef je `torch.randperm` een generator naar keuze?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with a random order of the numbers 0 to `n` − 1, drawn from the generator `g`.",
                      "nl": "Vervang de `...` door een willekeurige volgorde van de getallen 0 tot en met `n` − 1, getrokken uit de generator `g`."},
            "arguments": {"en": "The check calls `{call}`, with a number of examples, a number of epochs and a seed, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(n, epochs, seed)`?",
                          "nl": "De check roept `{call}` aan, met een aantal voorbeelden, een aantal epochs en een seed, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(n, epochs, seed)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does every epoch get a random order of all `n` numbers, drawn from `g`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Krijgt elke epoch een willekeurige volgorde van alle `n` getallen, getrokken uit `g`?"},
            "almost": {"en": "Right for `orders(8, 2, 0)`. But `{call}` gives {got}, and it should give {want}. Does your order use `n`, whatever its value?",
                       "nl": "Goed voor `orders(8, 2, 0)`. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je volgorde `n`, welke waarde `n` ook heeft?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return found`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return found`?"},
            "correct": {"en": "Right in all {n} cases. Every epoch gets a new order, and the same seed gives the same orders on every run. Which other random numbers of a training run need a seed?",
                        "nl": "Goed in alle {n} gevallen. Elke epoch krijgt een nieuwe volgorde, en dezelfde seed geeft bij elke run dezelfde volgordes. Welke andere willekeurige getallen van een training-run hebben een seed nodig?"},
        },
    ),
    "better": FunctionBox(
        name="better",
        function="better",
        start=_B_START,
        solution=_B,
        visible=[(RUNS_64_CASE, RUNS_128_CASE)],
        hidden=[(list(RUNS_LINEAR), list(RUNS_64)), (list(RUNS_64), list(RUNS_LINEAR)), ([900, 940], [935, 936]),
                ([920, 930], [910, 960]), ([900, 910], [910, 920]), ([926], [927])],
        first=[_DOTS],
        picture=better_picture,
        mistakes=[
            Mistake("means", wrong=_B_MEANS,
                    en="`{call}` gives {got}, and it should give {want}: your function compares the means, and a mean can come out higher through a few lucky runs while the two spreads overlap. Which run of `b` has to beat which run of `a`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie vergelijkt de gemiddelden, en een gemiddelde kan hoger uitkomen door een paar runs met geluk terwijl de twee spreidingen overlappen. Welke run van `b` moet welke run van `a` verslaan?"),
            Mistake("best", wrong=_B_BEST,
                    en="`{call}` gives {got}, and it should give {want}: your function compares the best run of each, and a best run can be one lucky seed. Which run of `b` has to beat the best run of `a`, so that every run of `b` does?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie vergelijkt van elk de beste run, en een beste run kan één seed met geluk zijn. Welke run van `b` moet de beste run van `a` verslaan, zodat elke run van `b` dat doet?"),
            Mistake("swapped", wrong=_B_SWAPPED,
                    en="`{call}` gives {got}, and it should give {want}: your function says `True` when every run of `a` beats every run of `b`, the other way around. Which list holds the runs that have to win?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie zegt `True` als elke run van `a` elke run van `b` verslaat, precies andersom. In welke lijst staan de runs die moeten winnen?"),
            Mistake("tie", wrong=_B_TIE, state="almost",
                    en="Right for `C.RUNS_64` and `C.RUNS_128`. But `{call}` gives {got}, and it should give {want}: the worst run of `b` is exactly as good as the best run of `a`. Does an equal run beat the other?",
                    nl="Goed voor `C.RUNS_64` en `C.RUNS_128`. Maar `{call}` geeft {got}, en het moet {want} zijn: de slechtste run van `b` is precies even goed als de beste run van `a`. Verslaat een even goede run de andere?"),
            Mistake("worst", wrong=_B_WORST,
                    en="`{call}` gives {got}, and it should give {want}: your function compares the worst run of each. The worst run of `b` beats the worst of `a`, but does it beat the other runs of `a` too?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie vergelijkt van elk de slechtste run. De slechtste run van `b` verslaat de slechtste van `a`, maar verslaat deze run ook de andere runs van `a`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with one comparison that is `True` only when every run in `b` beats every run in `a`.",
                      "nl": "Vervang de `...` door één vergelijking die alleen `True` is als elke run in `b` elke run in `a` verslaat."},
            "arguments": {"en": "The check calls `{call}`, with two lists of results, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(a, b)`?",
                          "nl": "De check roept `{call}` aan, met twee lijsten met uitkomsten, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(a, b)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is your answer `True` only when every run of `b` beats every run of `a`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is je antwoord alleen `True` als elke run van `b` elke run van `a` verslaat?"},
            "almost": {"en": "Right for `C.RUNS_64` and `C.RUNS_128`. But `{call}` gives {got}, and it should give {want}. Does your answer use the runs of both lists, and is it `True` exactly when every run of `b` beats every run of `a`?",
                       "nl": "Goed voor `C.RUNS_64` en `C.RUNS_128`. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je antwoord de runs van beide lijsten, en is het precies `True` als elke run van `b` elke run van `a` verslaat?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still begin with `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Begint de regel in je functie nog met `return`?"},
            "correct": {"en": "Right in all {n} cases. For 64 and 128 neurons your function says `False`, though the mean and the best run both favor 128 neurons. What would you need before you call 128 neurons better?",
                        "nl": "Goed in alle {n} gevallen. Bij 64 en 128 neuronen zegt je functie `False`, terwijl het gemiddelde en de beste run allebei gunstiger zijn voor 128 neuronen. Wat zou je nodig hebben om 128 neuronen beter te noemen?"},
        },
    ),
}

# The lesson is done when these are correct. "gpu" and "one-seed" are the choice questions at the end.
REQUIRED = ["orders", "better", "gpu", "one-seed"]

CASES = [
    ("orders", _O_START, "empty", None),
    ("orders", _O, "correct", None),
    ("orders", _o("torch.randperm(n, ...)"), "wrong", "dots-left"),
    ("orders", _O_GLOBAL, "wrong", "global"),
    ("orders", _O_RESEED, "wrong", "reseed"),
    ("orders", _O_EIGHT, "almost", None),
    ("orders", _o("torch.randperm(n, g)"), "error", "positional"),
    ("orders", _o("torch.randperm(n, generator=g, dtype=torch.int64)"), "correct", None),
    ("orders", _o("torch.randperm(n, generator=g).clone()"), "correct", None),
    ("orders", _O.replace("    return found\n", "    print(found)\n"), "wrong", "prints"),
    ("better", _B_START, "empty", None),
    ("better", _B, "correct", None),
    ("better", "def better(a, b):\n    return min(b) > ...\n", "wrong", "dots-left"),
    ("better", _B_MEANS, "wrong", "means"),
    ("better", _B_BEST, "wrong", "best"),
    ("better", _B_WORST, "wrong", "worst"),
    ("better", _B_TIE, "almost", "tie"),
    ("better", "def better(a, b):\n    return False\n", "almost", None),
    ("better", _B_SWAPPED, "wrong", "swapped"),
    ("better", "def better(a, b):\n    return max(b) < min(a)\n", "wrong", "swapped"),
    ("better", "def better(a, b):\n    return all(x > y for x in b for y in a)\n", "correct", None),
    ("better", "def better(a, b):\n    return sorted(b)[0] > sorted(a)[-1]\n", "correct", None),
    ("better", "def better(a, b):\n    print(min(b) > max(a))\n", "wrong", "prints"),
]
