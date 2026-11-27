# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-experiment-tracking: keeping track of your runs. Every run writes its settings (params),
its measurements after every epoch (metrics, each with its step) and its result to a log, one JSON line per run, so
runs can be compared later. The learner writes `log_metric` of a small tracker, and finds the best run in a log by its
lowest validation loss; the demo goes through the course's own log of twelve runs.

The runs are the course's own: the digits MLP of torch-loss-curves on the first 1,000 training digits of
data/qmnist-small.npz, with the validation set of that lesson, trained on a CPU with PyTorch 2.14.1 (see `train`)."""
import html
import json

import torch

from everything_ai import check, paths, yours
from everything_ai.check import ClassBox, FunctionBox, Mistake

LESSON = "sys-experiment-tracking"
LANGS = ("en", "nl")

# ======== the course's log ================================================================================================
# The MLP of torch-loss-curves (784 pixels, a hidden layer, 10 logits), 20 epochs of SGD on mini-batches of 50 of the
# first 1,000 training digits, after torch.manual_seed(seed), with a torch.Generator seeded the same for the order; after
# every epoch the loss on those 1,000 digits and on the validation set, in eval mode. Three learning rates, two sizes of
# the hidden layer, two seeds: twelve runs, measured on a CPU with PyTorch 2.14.1; the test trains one of them again.
DATA = "data/qmnist-small.npz"
TORCH = "2.14.1"
TRAIN_DIGITS, VAL_FIRST, VAL_DIGITS, EPOCHS, BATCH = 1000, 5000, 500, 20, 50
LRS, HIDDENS, SEEDS = (0.05, 0.2, 0.5), (32, 128), (0, 1)

# The loss on the 1,000 training digits after every epoch, 4 decimals, one run per name.
TRAIN_LOSS = {
    "lr0.05-h32-s0": [2.1643, 1.9372, 1.6368, 1.3253, 1.0811, 0.9, 0.7746, 0.686, 0.6165, 0.5668,
                      0.5229, 0.4879, 0.4576, 0.4289, 0.4077, 0.389, 0.3712, 0.355, 0.342, 0.3299],
    "lr0.05-h32-s1": [2.1545, 1.923, 1.6262, 1.3205, 1.0706, 0.8919, 0.7703, 0.6811, 0.6144, 0.5584,
                      0.5153, 0.4838, 0.452, 0.4295, 0.4056, 0.3852, 0.3661, 0.35, 0.3361, 0.3257],
    "lr0.05-h128-s0": [2.067, 1.7513, 1.4051, 1.115, 0.9186, 0.7759, 0.6762, 0.6068, 0.5496, 0.5077,
                       0.4721, 0.4426, 0.4154, 0.3912, 0.3728, 0.3565, 0.3412, 0.3266, 0.3153, 0.3052],
    "lr0.05-h128-s1": [2.0731, 1.7666, 1.4202, 1.1241, 0.913, 0.7712, 0.6773, 0.6043, 0.5502, 0.5032,
                       0.4673, 0.441, 0.4142, 0.3958, 0.3751, 0.3564, 0.3392, 0.3252, 0.3127, 0.3031],
    "lr0.2-h32-s0": [1.4375, 0.7359, 0.5361, 0.4293, 0.3698, 0.3194, 0.269, 0.236, 0.2155, 0.1932,
                     0.1691, 0.1801, 0.1489, 0.1172, 0.1068, 0.0992, 0.0878, 0.0815, 0.0706, 0.0679],
    "lr0.2-h32-s1": [1.4562, 0.7335, 0.5358, 0.4096, 0.3454, 0.2938, 0.2608, 0.2382, 0.2119, 0.1831,
                     0.1828, 0.154, 0.1462, 0.1258, 0.1352, 0.1022, 0.088, 0.0807, 0.0772, 0.0719],
    "lr0.2-h128-s0": [1.1839, 0.6348, 0.4776, 0.396, 0.3378, 0.2925, 0.2485, 0.2146, 0.1948, 0.1743,
                      0.1563, 0.171, 0.1374, 0.107, 0.0966, 0.0899, 0.08, 0.0729, 0.0644, 0.0624],
    "lr0.2-h128-s1": [1.2143, 0.6354, 0.46, 0.3679, 0.3245, 0.2743, 0.2412, 0.2193, 0.1919, 0.1679,
                      0.1716, 0.1408, 0.1458, 0.1133, 0.1238, 0.093, 0.0776, 0.0702, 0.0667, 0.0619],
    "lr0.5-h32-s0": [0.801, 0.4277, 0.3256, 0.502, 0.1982, 0.1841, 0.112, 0.0869, 0.0649, 0.0802,
                     0.0401, 0.035, 0.0267, 0.0211, 0.0189, 0.0167, 0.0148, 0.014, 0.0122, 0.0114],
    "lr0.5-h32-s1": [1.0383, 0.4197, 0.5082, 0.2719, 0.1807, 0.1413, 0.1348, 0.1067, 0.0744, 0.054,
                     0.0453, 0.0398, 0.0326, 0.026, 0.0244, 0.0191, 0.0168, 0.0148, 0.0139, 0.0125],
    "lr0.5-h128-s0": [0.7828, 0.3956, 0.2848, 0.3348, 0.1688, 0.1537, 0.101, 0.0694, 0.0501, 0.0537,
                      0.0338, 0.031, 0.0224, 0.0186, 0.0164, 0.0148, 0.0132, 0.0123, 0.0109, 0.0103],
    "lr0.5-h128-s1": [0.8915, 0.4214, 0.3353, 0.2199, 0.1624, 0.1165, 0.1096, 0.0815, 0.0556, 0.0417,
                      0.0406, 0.0285, 0.025, 0.0205, 0.0191, 0.0155, 0.0136, 0.0122, 0.0115, 0.0104],
}
# The loss on the validation set (training images 5000 to 5499) after every epoch.
VAL_LOSS = {
    "lr0.05-h32-s0": [2.1719, 1.9432, 1.6397, 1.3304, 1.1002, 0.9435, 0.8345, 0.7638, 0.7081, 0.6571,
                      0.6333, 0.5962, 0.5821, 0.5571, 0.5466, 0.5279, 0.5207, 0.5098, 0.4923, 0.4929],
    "lr0.05-h32-s1": [2.1451, 1.9085, 1.6127, 1.3179, 1.086, 0.9291, 0.8238, 0.7613, 0.7048, 0.654,
                      0.6184, 0.5952, 0.5698, 0.5625, 0.5382, 0.5307, 0.5177, 0.5065, 0.499, 0.4903],
    "lr0.05-h128-s0": [2.0576, 1.7318, 1.3925, 1.1153, 0.938, 0.8243, 0.7395, 0.6894, 0.6461, 0.6037,
                       0.5875, 0.5569, 0.5441, 0.5253, 0.5158, 0.4994, 0.4953, 0.4848, 0.4711, 0.4721],
    "lr0.05-h128-s1": [2.0603, 1.7509, 1.4124, 1.1294, 0.9378, 0.8167, 0.7337, 0.6879, 0.6433, 0.6012,
                       0.5709, 0.5517, 0.5309, 0.5283, 0.5049, 0.4997, 0.4885, 0.479, 0.4709, 0.4649],
    "lr0.2-h32-s0": [1.453, 0.8025, 0.6656, 0.5867, 0.5141, 0.5264, 0.4518, 0.4554, 0.4595, 0.4297,
                     0.4551, 0.4603, 0.4858, 0.4498, 0.4495, 0.4581, 0.4669, 0.4798, 0.4622, 0.4788],
    "lr0.2-h32-s1": [1.4543, 0.7925, 0.6292, 0.541, 0.4981, 0.4965, 0.4701, 0.4601, 0.4702, 0.4429,
                     0.4805, 0.452, 0.4737, 0.4721, 0.4975, 0.478, 0.4869, 0.4999, 0.5006, 0.4929],
    "lr0.2-h128-s0": [1.2009, 0.7171, 0.6177, 0.5602, 0.4899, 0.5047, 0.4491, 0.439, 0.4495, 0.4231,
                      0.446, 0.4594, 0.4724, 0.444, 0.4413, 0.4443, 0.4523, 0.4567, 0.448, 0.462],
    "lr0.2-h128-s1": [1.2207, 0.701, 0.5554, 0.4985, 0.4745, 0.479, 0.4465, 0.4333, 0.4371, 0.4179,
                      0.459, 0.421, 0.452, 0.4373, 0.4637, 0.4434, 0.4477, 0.4533, 0.4496, 0.4474],
    "lr0.5-h32-s0": [0.9231, 0.5938, 0.5558, 0.7768, 0.456, 0.5028, 0.4269, 0.4292, 0.4406, 0.4844,
                     0.4628, 0.4554, 0.473, 0.4779, 0.4877, 0.4895, 0.4976, 0.5068, 0.5053, 0.5184],
    "lr0.5-h32-s1": [1.1087, 0.5536, 0.6943, 0.4615, 0.4427, 0.4623, 0.4842, 0.4406, 0.4668, 0.4452,
                     0.4585, 0.4518, 0.4705, 0.4743, 0.4909, 0.4863, 0.4972, 0.5053, 0.5016, 0.5046],
    "lr0.5-h128-s0": [0.8899, 0.5835, 0.5259, 0.6218, 0.4268, 0.4703, 0.4097, 0.3965, 0.4093, 0.419,
                      0.423, 0.4109, 0.4234, 0.4288, 0.4343, 0.4325, 0.4396, 0.4422, 0.4419, 0.4503],
    "lr0.5-h128-s1": [0.9492, 0.5798, 0.526, 0.4183, 0.4298, 0.4309, 0.4447, 0.4229, 0.4244, 0.4204,
                      0.4528, 0.4188, 0.4348, 0.4375, 0.4459, 0.4479, 0.4558, 0.4581, 0.4572, 0.4572],
}
# Test digits read right of 1,000, after the last epoch.
RIGHT = {"lr0.05-h32-s0": 817, "lr0.05-h32-s1": 821, "lr0.05-h128-s0": 828, "lr0.05-h128-s1": 830,
         "lr0.2-h32-s0": 833, "lr0.2-h32-s1": 852, "lr0.2-h128-s0": 845, "lr0.2-h128-s1": 849, "lr0.5-h32-s0": 850, "lr0.5-h32-s1": 870, "lr0.5-h128-s0": 873, "lr0.5-h128-s1": 866}


def run_name(lr: float, hidden: int, seed: int) -> str:
    return f"lr{lr}-h{hidden}-s{seed}"


NAMES = [run_name(lr, h, s) for lr in LRS for h in HIDDENS for s in SEEDS]


def make_run(name: str) -> dict:
    """One run of the course's log, as the tracker of the lesson keeps it: a new dict every time."""
    lr, hidden, seed = name[2:].split("-")
    return {"name": name,
            "params": {"lr": float(lr), "hidden": int(hidden[1:]), "seed": int(seed[1:]), "epochs": EPOCHS},
            "metrics": {"train_loss": [[e + 1, v] for e, v in enumerate(TRAIN_LOSS[name])],
                        "val_loss": [[e + 1, v] for e, v in enumerate(VAL_LOSS[name])],
                        "test_right": [[EPOCHS, RIGHT[name]]]}}


LOG = "".join(json.dumps(make_run(n)) + "\n" for n in NAMES)   # the course's log, one JSON line per run


def train(name: str) -> tuple[list[float], list[float], int]:
    """One run as the course measured it: the training loss and the validation loss after every epoch (4 decimals),
    and the test digits read right after the last epoch."""
    import numpy as np
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    lr, hidden, seed = (make_run(name)["params"][k] for k in ("lr", "hidden", "seed"))
    d = np.load(paths.ROOT / DATA)
    x = torch.tensor(d["train_images"], dtype=torch.float32) / 255
    y = torch.tensor(d["train_labels"], dtype=torch.int64)
    x_tr, y_tr = x[:TRAIN_DIGITS], y[:TRAIN_DIGITS]
    x_val, y_val = x[VAL_FIRST:VAL_FIRST + VAL_DIGITS], y[VAL_FIRST:VAL_FIRST + VAL_DIGITS]
    x_test = torch.tensor(d["test_images"], dtype=torch.float32) / 255
    y_test = torch.tensor(d["test_labels"], dtype=torch.int64)

    class DigitsMLP(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden = nn.Linear(784, hidden)
            self.out = nn.Linear(hidden, 10)

        def forward(self, x):
            return self.out(torch.relu(self.hidden(x)))

    loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(seed)
    model = DigitsMLP(hidden)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=BATCH, shuffle=True, generator=g)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    for _ in range(EPOCHS):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            train_losses.append(round(loss_fn(model(x_tr), y_tr).item(), 4))
            val_losses.append(round(loss_fn(model(x_val), y_val).item(), 4))
        model.train()
    model.eval()
    with torch.no_grad():
        right = int((model(x_test).argmax(dim=1) == y_test).sum())
    return train_losses, val_losses, right


def log_path(name: str) -> str:
    """For the worked example: a file in the learner's work folder, work/runs/<name>."""
    folder = paths.WORK / "runs"
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / name)


# ======== the course's solutions ==========================================================================================

def log_metric(run, key, value, step):
    if key not in run["metrics"]:
        run["metrics"][key] = []
    run["metrics"][key].append([step, value])


def lowest(run, key):
    return min([value for step, value in run["metrics"][key]])


def best_run(runs, key):
    best = runs[0]
    for run in runs:
        if lowest(run, key) < lowest(best, key):
            best = run
    return best["name"]


def best_epoch(name: str) -> tuple[int, float]:
    """The epoch, counted from 1, with the lowest validation loss of a run of the log, and that loss."""
    val = VAL_LOSS[name]
    k = val.index(min(val))
    return k + 1, val[k]


BEST = best_run([make_run(n) for n in NAMES], "val_loss")                       # by the lowest validation loss
BEST_LAST = min(NAMES, key=lambda n: VAL_LOSS[n][-1])                            # by the loss after the last epoch
STILL_FALLING = [n for n in NAMES if best_epoch(n)[0] == EPOCHS]                 # lowest after the last epoch


# ======== drawing =========================================================================================================

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
            out.append(f'<code style="white-space:nowrap">{part}</code>')
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


def _line(x1: float, y1: float, x2: float, y2: float, ink: str = "var(--ea-line)", width: float = 1.0,
          dash: str = "") -> str:
    d = f";stroke-dasharray:{dash}" if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'style="stroke:{ink};stroke-width:{width}{d}"/>')


def _rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "") -> str:
    edge = f";stroke:{stroke};stroke-width:1" if stroke else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" style="fill:{fill}{edge}"/>'


def _box(x: float, y: float, w: float, h: float, stroke: str = "var(--ea-ink-3)", width: float = 1.2) -> str:
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" '
            f'style="fill:var(--ea-page);stroke:{stroke};stroke-width:{width}"/>')


def _dot(x: float, y: float, r: float, fill: str, stroke: str = "var(--ea-page)") -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" style="fill:{fill};stroke:{stroke};stroke-width:1"/>'


def _ring(x: float, y: float, r: float, ink: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" style="fill:var(--ea-page);stroke:{ink};stroke-width:1.3"/>'


def _arrow(x1: float, y1: float, x2: float, y2: float, ink: str = "var(--ea-ink-2)", width: float = 1.3) -> str:
    """A line from (x1, y1) to (x2, y2) with a small head at its end."""
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    h1 = (x2 - 6 * math.cos(a - 0.45), y2 - 6 * math.sin(a - 0.45))
    h2 = (x2 - 6 * math.cos(a + 0.45), y2 - 6 * math.sin(a + 0.45))
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" style="stroke:{ink};stroke-width:{width}"/>'
            f'<path d="M{h1[0]:.1f},{h1[1]:.1f} L{x2:.1f},{y2:.1f} L{h2[0]:.1f},{h2[1]:.1f}" '
            f'style="fill:none;stroke:{ink};stroke-width:{width}"/>')


def _poly(points: list, ink: str, width: float, dash: str = "") -> str:
    d = f";stroke-dasharray:{dash}" if dash else ""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{pts}" style="fill:none;stroke:{ink};stroke-width:{width};stroke-linejoin:round{d}"/>'


def _svg(height: float, alt: str, parts: list) -> str:
    return (f'<svg viewBox="0 0 330 {height:.0f}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">'
            f'{"".join(parts)}</svg>')


def _short(text_: str, n: int) -> str:
    return text_ if len(text_) <= n else text_[:n - 1] + "…"


# A loss chart on a log scale, as in torch-loss-curves: training solid, validation dashed, both in the color of the loss.
_LOG_TICKS = (0.01, 0.1, 1.0)


def _loss_chart(parts: list, top: float, train: list, val: list, lang: str, mark: bool = True,
                height: float = 96.0) -> float:
    """Draws the two curves of one run from `top` down; gives the y under the chart."""
    import math
    X0, X1 = 34.0, 322.0
    Y1, Y0 = top, top + height
    lo, hi = math.log10(0.008), math.log10(3.0)
    n = max(len(train), len(val), 2)

    def xy(e: int, v: float) -> tuple[float, float]:
        v = min(max(v, 0.008), 3.0)
        return X0 + e * (X1 - X0) / (n - 1), Y0 - (math.log10(v) - lo) / (hi - lo) * (Y0 - Y1)

    for v in _LOG_TICKS:
        _, y = xy(0, v)
        parts.append(_line(X0, y, X1, y, "var(--ea-line)", 0.8))
        parts.append(_t(X0 - 5, y + 3, number(v, 2 if v < 0.1 else 1, lang), 8.5, 400, "var(--ea-ink-2)", "end"))
    if train:
        parts.append(_poly([xy(e, v) for e, v in enumerate(train)], "var(--ea-loss)", 1.8))
    if val:
        parts.append(_poly([xy(e, v) for e, v in enumerate(val)], "var(--ea-loss)", 1.8, "5 3"))
        if mark:
            k = val.index(min(val))
            x, y = xy(k, val[k])
            parts.append(_dot(x, y, 4.0, "var(--ea-loss)"))
    for e in range(n):
        if e == 0 or (e + 1) % 5 == 0:
            x, _ = xy(e, 1.0)
            parts.append(_t(x, Y0 + 13, str(e + 1), 8.5, 400, "var(--ea-ink-2)", "middle"))
    return Y0 + 18


def _legend(parts: list, y: float, lang: str) -> None:
    w = {"en": ("training loss", "validation loss", "lowest"), "nl": ("training loss", "validation loss", "laagste")}[lang]
    parts.append(_line(0, y - 3, 18, y - 3, "var(--ea-loss)", 1.8))
    parts.append(_t(22, y, w[0], 9, 400, "var(--ea-ink-2)"))
    parts.append(_line(104, y - 3, 122, y - 3, "var(--ea-loss)", 1.8, "5 3"))
    parts.append(_t(126, y, w[1], 9, 400, "var(--ea-ink-2)"))
    parts.append(_dot(222, y - 3, 3.5, "var(--ea-loss)"))
    parts.append(_t(229, y, w[2], 9, 400, "var(--ea-ink-2)"))


# ======== the picture of the idea =========================================================================================

IDEA_TEXT = {
    "en": {"title": "One run, one line in the log",
           "before": "Before training: log_param, once per setting",
           "every": "After every epoch: log_metric, with the step",
           "end": "At the end: the run as one line, added to the log",
           "new": "new",
           "alt": "Three steps of one run. Before training, log_param records the learning rate, the size and the seed. After every epoch, log_metric records the validation loss with its step. At the end, the run is added as one JSON line at the end of runs.jsonl, below the lines of earlier runs, which stay as they are."},
    "nl": {"title": "Eén run, één regel in het log",
           "before": "Voordat de training begint: log_param, één keer per instelling",
           "every": "Na elke epoch: log_metric, met de step",
           "end": "Aan het eind: de run als één regel, onderaan het log",
           "new": "nieuw",
           "alt": "Drie stappen van één run. Voordat de training begint, legt log_param de learning rate, de grootte en de seed vast. Na elke epoch legt log_metric de validation loss vast met de step. Aan het eind komt de run als één JSON-regel onderaan runs.jsonl, onder de regels van eerdere runs; deze regels blijven ongemoeid."},
}


def idea_picture(lang: str) -> str:
    """A run logs its params once, its metrics after every epoch, and ends as one new line at the end of the log."""
    w = IDEA_TEXT[lang]
    show = BEST
    run = make_run(show)
    p = run["params"]
    val = VAL_LOSS[show]
    parts = [_t(0, 14, w["title"], 12.5, 600)]
    y = 28.0
    parts.append(_box(0, y, 330, 44))
    parts.append(_t(8, y + 15, w["before"], 10, 600))
    parts.append(_mono(8, y + 33, f'lr {p["lr"]}   hidden {p["hidden"]}   seed {p["seed"]}', 10))
    parts.append(_arrow(20, y + 46, 20, y + 60))
    y += 62
    parts.append(_box(0, y, 330, 58))
    parts.append(_t(8, y + 15, w["every"], 10, 600))
    for k in range(2):
        parts.append(_mono(8, y + 32 + 14 * k, f"step {k + 1}   val_loss", 10))
        parts.append(_mono(140, y + 32 + 14 * k, f"{val[k]}", 10, "var(--ea-loss-text)"))
    parts.append(_mono(200, y + 46, f"… step {EPOCHS}", 10, "var(--ea-ink-2)"))
    parts.append(_arrow(20, y + 60, 20, y + 74))
    y += 76
    parts.append(_t(0, y + 12, w["end"], 10, 600))
    lines = [json.dumps(make_run(n)) for n in (NAMES[0], NAMES[4], show)]
    y += 22
    parts.append(_mono(0, y, "runs.jsonl", 9.5, "var(--ea-ink-2)", weight=600))
    for k, line in enumerate(lines):
        yy = y + 8 + 16 * k
        new = k == len(lines) - 1
        parts.append(_rect(0, yy, 330, 14, "var(--ea-inset)", "var(--ea-ink-2)" if new else "var(--ea-line)"))
        parts.append(_mono(4, yy + 10, _short(line, 47), 9, "var(--ea-ink)" if new else "var(--ea-ink-2)"))
        if new:
            parts.append(_t(326, yy + 26, w["new"], 9, 600, "var(--ea-ink)", "end"))
    height = y + 8 + 16 * len(lines) + 16
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(w["alt"])}" style="margin:0.25rem 0 0.75rem;'
            f'max-width:min(100%,330px)"><svg viewBox="0 0 330 {height:.0f}" aria-hidden="true" style="{_SVG}">'
            f'{"".join(parts)}</svg></figure>')


# ======== the pictures of the learner's work ==============================================================================

PICTURE_SECONDS = 2   # what the learner's function gets for all the calls of one picture together
_STOPPED = object()   # what _call gives for a function that stopped with an error


def _fn(seen, name: str):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(name)
    return fn if callable(fn) else None


def _call(make_):
    """make_() within a time limit; _STOPPED when it stops with an error or runs out of time."""
    try:
        return check.with_time_limit(make_, PICTURE_SECONDS)
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _pairs(v) -> list | None:
    """v as a list of [step, value] pairs with a whole step from 1 to 99 and a number, or None."""
    if not isinstance(v, (list, tuple)) or len(v) > 200:
        return None
    out = []
    for item in v:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        s, x = item
        if not (isinstance(s, int) and not isinstance(s, bool) and 1 <= s <= 99):
            return None
        if not (isinstance(x, (int, float)) and not isinstance(x, bool) and 0 < x < 100):
            return None
        out.append([s, float(x)])
    return out


def _shown(v, n: int = 46) -> str:
    """What a value of the learner's log looks like as JSON, shortened."""
    try:
        return _short(json.dumps(v), n)
    except (TypeError, ValueError):
        return _short(repr(v), n)


_YOURS_TEXT = {
    "en": {
        "metric": {"title": "Your log_metric on one run",
                   "intro": "The picture calls your `log_metric` after every epoch of the run `{name}`, with its training loss and its validation loss, and draws what your log kept.",
                   "kept": "len(run[\"metrics\"][\"val_loss\"]) == {n}",
                   "fail": "Your `log_metric` stops before the log is full, so there is nothing to draw yet. The check above says why.",
                   "right": "Every epoch is in the log, with its step: from the log alone, you can draw the two curves again and see that the validation loss is lowest after epoch {e}.",
                   "wrong": "The chart can only draw what your log kept: an item for every epoch, with the step first and the value second. Compare your items with the course's text above.",
                   "alt": "What your log_metric kept of the run {name}: the items of the validation loss, and the two loss curves drawn from them."},
        "best": {"title": "Your best_run on the course's log",
                 "intro": "Each row is a run of the course's log: a dot at its lowest validation loss, a ring at its loss after the last epoch. The bold row is the run your `best_run(C.RUNS, \"val_loss\")` gives.",
                 "lowest": "lowest", "last": "after the last epoch", "yours": "yours",
                 "none": "Your `best_run` gives {got}, which is not a name of a run in the log.",
                 "fail": "Your `best_run` stops with an error on the course's log, so it marks no run yet. The check above says why.",
                 "right": "Your function picks the run with the lowest dot: `{name}`. Sorted by the rings, the losses after the last epoch, another run would lead.",
                 "wrong": "The run your function picks does not have the lowest dot. Which run reaches the lowest validation loss, in any epoch?",
                 "lucky": "On the course's log, your function picks the run with the lowest dot, `{name}`, but on other logs the check above gets another run. Which value of each run does your comparison look at?",
                 "alt": "Twelve runs, each with its lowest validation loss and its loss after the last epoch on one axis; the run your best_run picks is bold."},
    },
    "nl": {
        "metric": {"title": "Je log_metric bij één run",
                   "intro": "De afbeelding roept je `log_metric` na elke epoch van de run `{name}` aan, met de training loss en de validation loss, en tekent wat je log bewaarde.",
                   "kept": "len(run[\"metrics\"][\"val_loss\"]) == {n}",
                   "fail": "Je `log_metric` stopt voordat het log vol is, dus valt er nog niets te tekenen. De check hierboven zegt waarom.",
                   "right": "Elke epoch staat in het log, met de step: uit het log alleen kun je de twee curves opnieuw tekenen en zien dat de validation loss na epoch {e} het laagst is.",
                   "wrong": "De grafiek kan alleen tekenen wat je log bewaarde: een item voor elke epoch, met eerst de step en dan de waarde. Vergelijk je items met de tekst van de course hierboven.",
                   "alt": "Wat je log_metric van de run {name} bewaarde: de items van de validation loss, en de twee loss-curves die daaruit getekend zijn."},
        "best": {"title": "Je best_run op het log van de course",
                 "intro": "Elke rij is een run uit het log van de course: een stip bij de laagste validation loss, een ring bij de loss na de laatste epoch. De vette rij is de run die je `best_run(C.RUNS, \"val_loss\")` geeft.",
                 "lowest": "laagste", "last": "na de laatste epoch", "yours": "jouw keus",
                 "none": "Je `best_run` geeft {got}, en dat is geen naam van een run in het log.",
                 "fail": "Je `best_run` stopt met een fout op het log van de course, dus wijst je functie nog geen run aan. De check hierboven zegt waarom.",
                 "right": "Je functie kiest de run met de laagste stip: `{name}`. Gesorteerd op de ringen, de loss na de laatste epoch, zou een andere run bovenaan staan.",
                 "wrong": "Je functie kiest een run zonder de laagste stip. Welke run haalt de laagste validation loss, in welke epoch dan ook?",
                 "lucky": "Op het log van de course kiest je functie de run met de laagste stip, `{name}`, maar op andere logs krijgt de check hierboven een andere run. Naar welke waarde van elke run kijkt je vergelijking?",
                 "alt": "Twaalf runs, elk met de laagste validation loss en de loss na de laatste epoch op één as; je best_run kiest de vette run."},
    },
}


def metric_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["metric"]
    name = BEST
    intro = _note(w["intro"].format(name=name))
    fn = _fn(seen, "log_metric")
    run = {"name": name, "params": {}, "metrics": {}}

    def fill():
        for e in range(EPOCHS):
            fn(run, "train_loss", TRAIN_LOSS[name][e], e + 1)
            fn(run, "val_loss", VAL_LOSS[name][e], e + 1)
        return run

    v = _call(fill) if fn is not None else _STOPPED
    if v is _STOPPED or not isinstance(run.get("metrics"), dict):
        return yours.figure(w["title"], intro + _note(w["fail"], _LOSS_UI), alt=w["alt"].format(name=name))
    metrics = run["metrics"]
    kept_val = metrics.get("val_loss")
    val_pairs = _pairs(kept_val)
    train_pairs = _pairs(metrics.get("train_loss"))
    right = (val_pairs == [[e + 1, x] for e, x in enumerate(VAL_LOSS[name])]
             and train_pairs == [[e + 1, x] for e, x in enumerate(TRAIN_LOSS[name])])
    n = len(kept_val) if isinstance(kept_val, (list, tuple)) else 0
    parts = [_mono(0, 12, w["kept"].format(n=n), 9.5, "var(--ea-ink)" if n == EPOCHS else "var(--ea-loss-text)", weight=600)]
    parts.append(_mono(0, 28, _shown(kept_val), 9.5, "var(--ea-ink-2)"))
    # the curves, drawn only from the items that are [step, value] pairs, in the order of their steps
    train = [x for _, x in sorted(train_pairs)] if train_pairs else []
    val = [x for _, x in sorted(val_pairs)] if val_pairs else []
    y = _loss_chart(parts, 44, train, val, lang, mark=bool(val) and right)
    _legend(parts, y + 10, lang)
    svg = _svg(y + 16, w["alt"].format(name=name), parts)
    note = w["right"].format(e=best_epoch(name)[0]) if right else w["wrong"]
    return yours.figure(w["title"], intro + svg + _note(note, _UI if right else _LOSS_UI), alt=w["alt"].format(name=name))


def best_svg(picked: str | None, lang: str, alt: str) -> str:
    """The twelve runs of the log on one axis of validation loss: a dot at the lowest, a ring at the last epoch."""
    w = _YOURS_TEXT[lang]["best"]
    A0, A1, LO, HI = 102.0, 316.0, 0.38, 0.53
    parts = []

    def ax(v: float) -> float:
        return A0 + (min(max(v, LO), HI) - LO) / (HI - LO) * (A1 - A0)

    y = 14.0
    parts.append(_dot(A0, y - 3, 3.2, "var(--ea-ink-2)"))
    parts.append(_t(A0 + 7, y, w["lowest"], 9, 400, "var(--ea-ink-2)"))
    parts.append(_ring(A0 + 62, y - 3, 3.0, "var(--ea-ink-2)"))
    parts.append(_t(A0 + 69, y, w["last"], 9, 400, "var(--ea-ink-2)"))
    y += 18
    for name in NAMES:
        mine = name == picked
        low, last = min(VAL_LOSS[name]), VAL_LOSS[name][-1]
        ink = "var(--ea-ink)" if mine else "var(--ea-ink-2)"
        parts.append(_mono(0, y + 3, name, 9.5, ink, weight=700 if mine else 400))
        parts.append(_line(ax(low), y, ax(last), y, "var(--ea-line)", 1.2))
        parts.append(_ring(ax(last), y, 3.0, "var(--ea-ink-3)"))
        parts.append(_dot(ax(low), y, 4.4 if mine else 3.2, "var(--ea-ink)" if mine else "var(--ea-ink-3)"))
        y += 15
    for v in (0.40, 0.45, 0.50):
        parts.append(_line(ax(v), y - 6, ax(v), y - 2, "var(--ea-ink-3)"))
        parts.append(_t(ax(v), y + 9, number(v, 2, lang), 8.5, 400, "var(--ea-ink-2)", "middle"))
    parts.append(_t(A1, y + 22, "validation loss", 8.5, 400, "var(--ea-ink-2)", "end"))
    return _svg(y + 28, alt, parts)


def best_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["best"]
    fn = _fn(seen, "best_run")
    v = _call(lambda: fn([make_run(n) for n in NAMES], "val_loss")) if fn is not None else _STOPPED
    if v is _STOPPED:
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["fail"], _LOSS_UI), alt=w["alt"])
    picked = v if isinstance(v, str) and v in NAMES else None
    svg = best_svg(picked, lang, w["alt"])
    if picked is None:
        note = _note(w["none"].format(got=_shown(v, 40)), _LOSS_UI)
    elif picked == BEST and all(_call(lambda c=c: fn(json.loads(json.dumps(c[0])), c[1])) == best_run(*c) for c in _HIDDEN_BEST):
        note = _note(w["right"].format(name=BEST))
    elif picked == BEST:   # right on the course's log by luck, as the check's other logs show
        note = _note(w["lucky"].format(name=BEST), _LOSS_UI)
    else:
        note = _note(w["wrong"], _LOSS_UI)
    return yours.figure(w["title"], _note(w["intro"]) + svg + note, alt=w["alt"])


# ======== the demo: the course's log ======================================================================================

DEMO_TEXT = {
    "en": {"params": "learning rate {lr}, {h} neurons, seed {s}", "loss": "loss after every epoch, log scale",
           "all": "lowest validation loss of all 12 runs", "line": "its line in the log",
           "alt": "The run {name}: its training loss and validation loss after every epoch, with the lowest validation loss, {low} after epoch {e}, marked; below, the lowest validation loss of all twelve runs, with this run drawn larger, and the start of its line in the log.",
           "says": {
               "dip": "Run **`{name}`**: learning rate {lr}, {h} neurons, seed {s}. Its validation loss is lowest after epoch **{e}**, at {low}, and {last} after the last epoch. After the last epoch it reads **{right}** of the 1,000 test digits right.",
               "falling": "Run **`{name}`**: learning rate {lr}, {h} neurons, seed {s}. Its validation loss is still falling: lowest after the last epoch, epoch **{e}**, at {low}. After the last epoch it reads **{right}** of the 1,000 test digits right.",
           }},
    "nl": {"params": "learning rate {lr}, {h} neuronen, seed {s}", "loss": "loss na elke epoch, logaritmische schaal",
           "all": "laagste validation loss van alle 12 runs", "line": "de regel in het log",
           "alt": "De run {name}: de training loss en de validation loss na elke epoch, met de laagste validation loss, {low} na epoch {e}, gemarkeerd; daaronder de laagste validation loss van alle twaalf runs, met deze run groter getekend, en het begin van de regel in het log.",
           "says": {
               "dip": "Run **`{name}`**: learning rate {lr}, {h} neuronen, seed {s}. De validation loss is na epoch **{e}** het laagst, {low}, en na de laatste epoch {last}. Na de laatste epoch leest het model **{right}** van de 1.000 testcijfers goed.",
               "falling": "Run **`{name}`**: learning rate {lr}, {h} neuronen, seed {s}. De validation loss daalt nog: het laagst na de laatste epoch, epoch **{e}**, met {low}. Na de laatste epoch leest het model **{right}** van de 1.000 testcijfers goed.",
           }},
}


def say(k: int, lang: str) -> str:
    name = NAMES[k]
    p = make_run(name)["params"]
    e, low = best_epoch(name)
    w = DEMO_TEXT[lang]["says"]["falling" if e == EPOCHS else "dip"]
    return w.format(name=name, lr=number(p["lr"], 2 if p["lr"] < 0.1 else 1, lang), h=p["hidden"], s=p["seed"], e=e,
                    low=number(low, 4, lang), last=number(VAL_LOSS[name][-1], 4, lang), right=RIGHT[name])


def frame(value: float, lang: str) -> tuple[str, str]:
    """The chosen run of the log: its two loss curves, its place among the lowest validation losses of all twelve runs,
    and the start of its line in the log."""
    k = min(max(int(round(value)), 0), len(NAMES) - 1)
    name = NAMES[k]
    w = DEMO_TEXT[lang]
    p = make_run(name)["params"]
    lr = number(p["lr"], 2 if p["lr"] < 0.1 else 1, lang)
    parts = [_mono(0, 14, name, 12, "var(--ea-ink)", weight=600),
             _t(0, 30, w["params"].format(lr=lr, h=p["hidden"], s=p["seed"]), 9.5, 400, "var(--ea-ink-2)"),
             _t(0, 50, w["loss"], 9.5, 400, "var(--ea-ink-2)")]
    y = _loss_chart(parts, 58, TRAIN_LOSS[name], VAL_LOSS[name], lang, height=104)
    _legend(parts, y + 10, lang)
    # the lowest validation loss of every run, on one axis
    y += 34
    parts.append(_t(0, y, w["all"], 9.5, 400, "var(--ea-ink-2)"))
    A0, A1, LO, HI = 34.0, 312.0, 0.38, 0.50

    def ax(v: float) -> float:
        return A0 + (min(max(v, LO), HI) - LO) / (HI - LO) * (A1 - A0)

    y += 18
    parts.append(_line(A0, y, A1, y, "var(--ea-line)", 1.0))
    for n in NAMES:
        if n != name:
            parts.append(_dot(ax(min(VAL_LOSS[n])), y, 3.0, "var(--ea-ink-3)"))
    parts.append(_dot(ax(min(VAL_LOSS[name])), y, 4.8, "var(--ea-ink)"))
    for v in (0.40, 0.45, 0.50):
        parts.append(_line(ax(v), y + 6, ax(v), y + 10, "var(--ea-ink-3)"))
        parts.append(_t(ax(v), y + 21, number(v, 2, lang), 8.5, 400, "var(--ea-ink-2)", "middle"))
    # the start of its line in the log
    y += 44
    parts.append(_t(0, y, w["line"], 9.5, 400, "var(--ea-ink-2)"))
    parts.append(_rect(0, y + 6, 330, 15, "var(--ea-inset)", "var(--ea-line)"))
    parts.append(_mono(4, y + 17, _short(json.dumps(make_run(name)), 47), 9, "var(--ea-ink)"))
    e, low = best_epoch(name)
    alt = w["alt"].format(name=name, low=number(low, 4, lang), e=e)
    return _svg(y + 26, alt, parts), say(k, lang)


# ======== the exercises ===================================================================================================

def _dots_left(got, text_) -> bool:
    import ast
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return "..." in text_
    return any(isinstance(n, ast.Constant) and n.value is Ellipsis for n in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Can you replace it with your own code?",
                nl="Er staat nog een `...` in je code. Kun je de `...` vervangen door je eigen code?")


def _plain(v):
    """A value as it lands in the log: tuples become lists, as json.dumps writes them."""
    try:
        return json.loads(json.dumps(v))
    except (TypeError, ValueError):
        return v


def _same_logged(got, want) -> bool:
    return check.same_value(_plain(got), _plain(want))


_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then begins with the course's lines again.",
                "nl": "Alle code in het vak is weg. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# log_metric: the box holds the function; each scenario makes a run, logs into it and ends with what the log holds.
_M_HEAD = ("def log_metric(run, key, value, step):\n"
           "    if key not in run[\"metrics\"]:\n"
           "        run[\"metrics\"][key] = []\n")


def _m(line: str) -> str:
    return _M_HEAD + "    " + line + "\n"


_M_START = _m("...")
_M = _m("run[\"metrics\"][key].append([step, value])")
_M_OVERWRITE = _m("run[\"metrics\"][key] = [[step, value]]")
_M_VALUE = _m("run[\"metrics\"][key].append(value)")
_M_SWAPPED = _m("run[\"metrics\"][key].append([value, step])")
_M_DICT = _m("run[\"metrics\"][key].append({\"step\": step, \"value\": value})")
_M_FIXED = _m("run[\"metrics\"][\"val_loss\"].append([step, value])")
_M_TWO = _m("run[\"metrics\"][key].append(step, value)")

_M_SCENARIOS = [
    'run = {"name": "first-try", "params": {}, "metrics": {}}\n'
    'log_metric(run, "val_loss", 0.61, 1)\n'
    'log_metric(run, "val_loss", 0.48, 2)\n'
    'run["metrics"]',
    'run = {"name": "two-keys", "params": {"lr": 0.5}, "metrics": {}}\n'
    'log_metric(run, "train_loss", 0.7, 1)\n'
    'log_metric(run, "val_loss", 0.8, 1)\n'
    'log_metric(run, "train_loss", 0.4, 2)\n'
    'log_metric(run, "val_loss", 0.6, 2)\n'
    'run',
    'run = {"name": "from-the-log", "params": {}, "metrics": {"val_loss": [[1, 0.9]]}}\n'
    'log_metric(run, "val_loss", 0.5, 2)\n'
    'log_metric(run, "test_right", 873, 20)\n'
    'run["metrics"]',
    'run = {"name": "one-step", "params": {}, "metrics": {}}\n'
    'log_metric(run, "accuracy", 0.92, 3)\n'
    'run["metrics"]["accuracy"]',
]

_F_EN = "With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives {got}, and it should give {want}"
_F_NL = "Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel {got}, en het moet {want} zijn"


class _Runs(list):
    """The course's log as a list of runs, which the check shows by its name: C.RUNS."""

    def __repr__(self) -> str:
        return "C.RUNS"

    def __deepcopy__(self, memo):
        return [json.loads(json.dumps(r)) for r in self]


RUNS = _Runs(make_run(n) for n in NAMES)   # the course's log, read back: what `json.loads` gives for each line of LOG

# best_run: the box holds `lowest` and `best_run`.
_B_HEAD = ("def lowest(run, key):\n"
           "    return min([value for step, value in run[\"metrics\"][key]])\n"
           "\n\n"
           "def best_run(runs, key):\n"
           "    best = runs[0]\n"
           "    for run in runs:\n")
_B_END = ("            best = run\n"
          "    return best[\"name\"]\n")


def _b(condition: str) -> str:
    return _B_HEAD + f"        if {condition}:\n" + _B_END


_B_START = _b("...")
_B = _b("lowest(run, key) < lowest(best, key)")
_B_LAST = _b("run[\"metrics\"][key][-1][1] < best[\"metrics\"][key][-1][1]")
_B_WORST = _b("lowest(run, key) > lowest(best, key)")
_B_FIRST = _b("lowest(run, key) < lowest(runs[0], key)")
_B_FIXED = _b("lowest(run, \"val_loss\") < lowest(best, \"val_loss\")")
_B_LISTS = _b("run[\"metrics\"][key] < best[\"metrics\"][key]")

_A = {"name": "a", "metrics": {"val_loss": [[1, 0.5], [2, 0.4]], "train_loss": [[1, 0.3], [2, 0.1]]}}
_Bb = {"name": "b", "metrics": {"val_loss": [[1, 0.45], [2, 0.42]], "train_loss": [[1, 0.2], [2, 0.05]]}}
_DIP = {"name": "dip", "metrics": {"loss": [[1, 0.6], [2, 0.3], [3, 0.5]]}}
_FLAT = {"name": "flat", "metrics": {"loss": [[1, 0.5], [2, 0.45], [3, 0.4]]}}
_HIDDEN_BEST = [([_A, _Bb], "train_loss"), ([_A, _Bb], "val_loss"), ([_DIP, _FLAT], "loss"), ([_FLAT, _DIP], "loss"),
                ([{"name": "only", "metrics": {"loss": [[1, 0.7]]}}], "loss")]

EXERCISES = {
    "log-metric": ClassBox(
        name="log-metric",
        start=_M_START,
        solution=_M,
        scenarios=_M_SCENARIOS,
        compare=_same_logged,
        first=[_DOTS],
        picture=metric_picture,
        mistakes=[
            Mistake("overwrite", wrong=_M_OVERWRITE,
                    en=_F_EN + ": only the last step stays, because each call puts a new list in the place of the old one. How do you add one item to the end of the list that is there?",
                    nl=_F_NL + ": alleen de laatste step blijft over, omdat elke aanroep een nieuwe lijst op de plek van de oude zet. Hoe voeg je één item toe aan het eind van de lijst die er al is?"),
            Mistake("value-only", wrong=_M_VALUE,
                    en=_F_EN + ": your log keeps the values without their steps, so it no longer says after which epoch the loss was 0.48. What else goes in each item?",
                    nl=_F_NL + ": je log bewaart de waarden zonder steps, dus staat er niet meer na welke epoch de loss 0,48 was. Wat hoort er nog meer in elk item?"),
            Mistake("swapped", wrong=_M_SWAPPED,
                    en=_F_EN + ": each item has the value first and the step second, the other way around. Which comes first in `[2, 0.48]`?",
                    nl=_F_NL + ": elk item heeft eerst de waarde en dan de step, precies andersom. Wat komt eerst in `[2, 0.48]`?"),
            Mistake("dict-item", wrong=_M_DICT,
                    en=_F_EN + ": your items are dicts. A dict works too, but the course's log keeps a list of two, the step first, so that `lowest` in exercise 2 can read it. Can you add a list of two instead?",
                    nl=_F_NL + ": je items zijn dicts. Een dict werkt ook, maar het log van de course bewaart een lijst van twee, met eerst de step, zodat `lowest` in oefening 2 het kan lezen. Kun je in plaats daarvan een lijst van twee toevoegen?"),
            Mistake("two-items", error=lambda e: isinstance(e, TypeError) and "append" in str(e),
                    en="Python stops with an error: `append` adds one item to a list, and your line gives it two. How do you make the step and the value one item?",
                    nl="Python stopt met een fout: `append` voegt één item toe aan een lijst, en je regel geeft `append` er twee. Hoe maak je van de step en de waarde één item?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with one line that adds the step and the value to the end of the list of `key`.",
                      "nl": "Vervang de `...` door één regel die de step en de waarde toevoegt aan het eind van de lijst van `key`."},
            "wrong": {"en": _F_EN + ". Does each call add one item to the end of the list of `key`, with the step first and the value second?",
                      "nl": _F_NL + ". Voegt elke aanroep één item toe aan het eind van de lijst van `key`, met eerst de step en dan de waarde?"},
            "almost": {"en": "Right for the run of the example. But " + _F_EN[0].lower() + _F_EN[1:] + ". Does your line use `key`, whatever metric it names?",
                       "nl": "Goed voor de run van het voorbeeld. Maar " + _F_NL[0].lower() + _F_NL[1:] + ". Gebruikt je regel `key`, welke metric `key` ook noemt?"},
            "correct": {"en": "Right in all {n} cases. Each call adds one item and keeps the earlier ones, so after 20 epochs the list holds 20 items, one per step. Which metric would you log once, at the end of a run?",
                        "nl": "Goed in alle {n} gevallen. Elke aanroep voegt één item toe en bewaart de eerdere, dus na 20 epochs bevat de lijst 20 items, één per step. Welke metric zou je één keer loggen, aan het eind van een run?"},
        },
    ),
    "best-run": FunctionBox(
        name="best-run",
        function="best_run",
        start=_B_START,
        solution=_B,
        visible=[(RUNS, "val_loss")],
        hidden=_HIDDEN_BEST,
        first=[_DOTS],
        picture=best_picture,
        mistakes=[
            Mistake("last", wrong=_B_LAST,
                    en="`{call}` gives {got}, and it should give {want}: your function compares the loss after the last epoch. With overfitting, the best model is from the epoch with the lowest validation loss, wherever it lies. Which function gives that value?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie vergelijkt de loss na de laatste epoch. Bij overfitting komt het beste model uit de epoch met de laagste validation loss, waar deze epoch ook ligt. Welke functie geeft deze waarde?"),
            Mistake("worst", wrong=_B_WORST,
                    en="`{call}` gives {got}, and it should give {want}: your function keeps the run with the highest lowest loss, the worst run. Which way should the comparison point for a loss?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie houdt de run met de hoogste laagste loss, de slechtste run. Welke kant moet de vergelijking op wijzen bij een loss?"),
            Mistake("first", wrong=_B_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your function compares every run with the first run of the list, so it keeps the last run that beats the first one. Which run should each run have to beat?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie vergelijkt elke run met de eerste run van de lijst, dus houdt je functie de laatste run die de eerste verslaat. Welke run zou elke run moeten verslaan?"),
            Mistake("fixed-key", wrong=_B_FIXED, state="almost",
                    en="Right for the course's log. But `{call}` gives {got}, and it should give {want}: your function looks at `\"val_loss\"`, whatever `key` names. Can you compare by `key`?",
                    nl="Goed voor het log van de course. Maar `{call}` geeft {got}, en het moet {want} zijn: je functie kijkt naar `\"val_loss\"`, wat `key` ook noemt. Kun je vergelijken op `key`?"),
            Mistake("first-epoch", wrong=_B_LISTS, state="almost",
                    en="Right for the course's log. But `{call}` gives {got}, and it should give {want}: your comparison ends up comparing the items `[1, …]`, the value after step 1. Python compares lists item by item, and every list of `key` starts with step 1. Which function gives the lowest value over all steps?",
                    nl="Goed voor het log van de course. Maar `{call}` geeft {got}, en het moet {want} zijn: je vergelijking vergelijkt uiteindelijk de items `[1, …]`, de waarde na step 1. Python vergelijkt lijsten item voor item, en elke lijst van `key` begint met step 1. Welke functie geeft de laagste waarde over alle steps?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with a comparison that is `True` when `run` reaches a lower value of `key` than the best run so far.",
                      "nl": "Vervang de `...` door een vergelijking die `True` is als `run` een lagere waarde van `key` haalt dan de beste run tot nu toe."},
            "arguments": {"en": "The check calls `{call}`, with a list of runs and the name of a metric, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(runs, key)`?",
                          "nl": "De check roept `{call}` aan, met een lijst runs en de naam van een metric, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(runs, key)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your comparison keep the run whose lowest value of `key` is the lowest so far?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Houdt je vergelijking de run waarvan de laagste waarde van `key` tot nu toe het laagst is?"},
            "almost": {"en": "Right for the course's log. But `{call}` gives {got}, and it should give {want}. Does your comparison use `run`, `best` and `key`, for any list of runs?",
                       "nl": "Goed voor het log van de course. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je vergelijking `run`, `best` en `key`, bij elke lijst runs?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return best[\"name\"]`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return best[\"name\"]`?"},
            "correct": {"en": "Right in all {n} cases. In the course's log, your function picks `" + BEST + "`, whose validation loss is lowest after epoch " + str(best_epoch(BEST)[0]) + ". Which run would you pick if you only looked at the last epoch?",
                        "nl": "Goed in alle {n} gevallen. In het log van de course kiest je functie `" + BEST + "`, met de laagste validation loss na epoch " + str(best_epoch(BEST)[0]) + ". Welke run zou je kiezen als je alleen naar de laatste epoch keek?"},
        },
    ),
}

# The lesson is done when these are correct. "lines" and "repeat" are the choice questions at the end.
REQUIRED = ["log-metric", "best-run", "lines", "repeat"]

CASES = [
    ("log-metric", _M_START, "empty", None),
    ("log-metric", _M, "correct", None),
    ("log-metric", _m("run[\"metrics\"][key].append(...)"), "wrong", "dots-left"),
    ("log-metric", _M_OVERWRITE, "wrong", "overwrite"),
    ("log-metric", _M_HEAD.replace("    if key not in run[\"metrics\"]:\n        ", "    ") + "    run[\"metrics\"][key].append([step, value])\n", "wrong", "overwrite"),
    ("log-metric", _M_VALUE, "wrong", "value-only"),
    ("log-metric", _M_SWAPPED, "wrong", "swapped"),
    ("log-metric", _M_DICT, "wrong", "dict-item"),
    ("log-metric", _m("run[\"metrics\"][key].append((step, value))"), "correct", None),
    ("log-metric", _m("run[\"metrics\"][key] = run[\"metrics\"][key] + [[step, value]]"), "correct", None),
    ("log-metric", _M_FIXED, "error", None),
    ("log-metric", _M_TWO, "error", "two-items"),
    ("log-metric", _m("run[\"metrics\"][key].append([step, value, value])"), "wrong", None),
    ("log-metric", _m("run[\"metrics\"].setdefault(\"val_loss\", []).append([step, value])"), "almost", None),
    ("best-run", _B_START, "empty", None),
    ("best-run", _B, "correct", None),
    ("best-run", _b("lowest(run, key) < ..."), "wrong", "dots-left"),
    ("best-run", _B_LAST, "wrong", "last"),
    ("best-run", _B_WORST, "wrong", "worst"),
    ("best-run", _B_FIRST, "wrong", "first"),
    ("best-run", _B_FIXED, "almost", "fixed-key"),
    ("best-run", _B_LISTS, "almost", "first-epoch"),
    ("best-run", _b("min(run[\"metrics\"][key]) < min(best[\"metrics\"][key])"), "almost", "first-epoch"),
    ("best-run", _b("lowest(best, key) > lowest(run, key)"), "correct", None),
    ("best-run", _b("min([v for s, v in run[\"metrics\"][key]]) < lowest(best, key)"), "correct", None),
    ("best-run", _b("run[\"name\"] < best[\"name\"]"), "wrong", None),
    ("best-run", _b("lowest(run, key) <= lowest(best, key)"), "correct", None),
    ("best-run", _B.replace("    return best[\"name\"]\n", "    print(best[\"name\"])\n"), "wrong", "prints"),
]
