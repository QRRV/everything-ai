# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-tuning-playbook: lr_range, learning rates an equal factor apart, and range_pick, the learning
rate at the lowest smoothed loss of a range test. The picture and the demo run a real range test and a small sweep on
the handwritten digits, with an MLP and SGD with momentum, on one thread."""
import functools
import math
import re

from everything_ai import paths
from everything_ai.check import FunctionBox, Mistake, run_code

DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md
LO, HI, STEPS, BATCH, BETA = 1e-4, 10.0, 100, 64, 0.9


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def rate(x: float, lang: str) -> str:
    """A learning rate as the lesson writes it: 0.0001, 0.03, 0.3, 3, without e-notation and without noise."""
    digits = max(0, -math.floor(math.log10(x)) + 1) if x < 1 else 0
    text = f"{round(x, digits):.{digits}f}".rstrip("0").rstrip(".") if digits else f"{x:.0f}"
    return text.replace(".", ",") if lang == "nl" else text


@functools.lru_cache(maxsize=1)
def _data():
    import numpy as np
    import torch
    d = np.load(paths.ROOT / DATA)
    return (torch.tensor(d["train_images"], dtype=torch.float32) / 255, torch.tensor(d["train_labels"], dtype=torch.int64),
            torch.tensor(d["test_images"], dtype=torch.float32) / 255, torch.tensor(d["test_labels"], dtype=torch.int64))


def _model(seed: int):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))


def range_test(lrs: list, seed: int = 0) -> list:
    """One run of a learning-rate range test: one mini-batch of 64 at every learning rate of `lrs`, in order, with SGD
    and momentum 0.9. Gives back the loss of every step, and stops once the loss is 4 times its start or no number."""
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    X, y, _, _ = _data()
    model = _model(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lrs[0], momentum=0.9)
    g = torch.Generator().manual_seed(seed)
    losses = []
    for lr in lrs:
        for group in opt.param_groups:
            group["lr"] = lr
        idx = torch.randint(0, len(X), (BATCH,), generator=g)
        loss = F.cross_entropy(model(X[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if not math.isfinite(losses[-1]) or losses[-1] > 4 * losses[0]:
            break
    return losses


def train(lr: float, seed: int = 0) -> float:
    """One epoch at one learning rate, with SGD and momentum 0.9: the validation loss after it."""
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    X, y, Xv, yv = _data()
    model = _model(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    order = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
    for i in range(0, len(X), BATCH):
        idx = order[i:i + BATCH]
        loss = F.cross_entropy(model(X[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        v = F.cross_entropy(model(Xv), yv).item()
    return v if math.isfinite(v) else float("inf")


def _course_range(lr_min, lr_max, steps):
    return [lr_min * (lr_max / lr_min) ** (i / (steps - 1)) for i in range(steps)]


def _course_pick(lrs, losses, beta):
    avg, best, best_lr = 0.0, None, None
    for i in range(len(losses)):
        avg = beta * avg + (1 - beta) * losses[i]
        smooth = avg / (1 - beta ** (i + 1))
        if best is None or smooth < best:
            best, best_lr = smooth, lrs[i]
    return best_lr


def smoothed(losses: list, beta: float = BETA) -> list:
    avg, out = 0.0, []
    for i, x in enumerate(losses):
        avg = beta * avg + (1 - beta) * x
        out.append(avg / (1 - beta ** (i + 1)))
    return out


@functools.lru_cache(maxsize=1)
def picture_run():
    lrs = _course_range(LO, HI, STEPS)
    losses = range_test(lrs, 0)
    return lrs[:len(losses)], losses


def range_picture(labels: dict) -> str:
    """The range test of seed 0: the loss of every step, small dots, and its smoothed loss, a line, against the
    learning rate on a log scale. Both are loss: purple. The lowest smoothed loss is marked, and the range a sweep
    looks at, from a tenth of it up to it, is a light band."""
    from everything_ai import chart
    lrs, losses = picture_run()
    sm = smoothed(losses)
    k = min(range(len(sm)), key=lambda i: sm[i])
    p = chart.Panel((LO, HI), (0, 4), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], xlog=True, yticks=[0, 1, 2, 3, 4])
    p.rect(lrs[k] / 10, 0, lrs[k], 4, "ink-3")
    p.points(lrs, [min(x, 3.95) for x in losses], "loss", r=2.2, hollow=True)
    p.line(lrs, [min(x, 3.95) for x in sm], "loss", 2.4)
    p.points([lrs[k]], [sm[k]], "ink", r=4.5)
    p.text(1.5e-4, 3.55, labels["lowest"].format(lr=rate(lrs[k], labels.get("lang", "en"))), "ink", "start")
    p.text(lrs[k] / 9, 2.95, labels["band"], "ink-2", "start")
    svg = chart.figure([p], alt=labels["alt"], width=330, height=260)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# --- exercise 1: lr_range -------------------------------------------------------------------------------------------------------

_R_HEAD = "def lr_range(lr_min, lr_max, steps):\n"
_R_START = (_R_HEAD + "    # steps learning rates from lr_min to lr_max, each the same factor larger than the one before\n"
            "    lrs = []\n    for i in range(steps):\n        ...\n    return lrs\n")


def _r(line: str = "        lrs.append(lr_min * (lr_max / lr_min) ** (i / (steps - 1)))\n") -> str:
    return _R_HEAD + "    lrs = []\n    for i in range(steps):\n" + line + "    return lrs\n"


def _r_values(v):
    try:
        if isinstance(v, (str, bytes)):
            return None
        return [float(x) for x in v]
    except (TypeError, ValueError):
        return None


def _r_close(got, want) -> bool:
    """Equal to 1 in 100,000: what _r_show tells apart with 6 significant digits, so a wrong list never looks right."""
    g, w = _r_values(got), _r_values(want)
    return g is not None and len(g) == len(w) and all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(g, w))


def _sig(x: float, n: int = 4) -> str:
    """At most n significant digits, never in e-notation, without float noise."""
    if x == 0:
        return "0"
    if not math.isfinite(x):
        return str(x)
    digits = max(0, n - 1 - math.floor(math.log10(abs(x))))
    return f"{round(x, digits):.{digits}f}".rstrip("0").rstrip(".") if digits else f"{x:.0f}"


def _r_show(v):
    vals = _r_values(v)
    if vals is None:
        return repr(v)
    return "[" + ", ".join(_sig(x, 6) for x in vals[:9]) + (", …]" if len(vals) > 9 else "]")


# --- exercise 2: range_pick -----------------------------------------------------------------------------------------------------

_P_HEAD = "def range_pick(lrs, losses, beta):\n"
_P_START = (_P_HEAD + "    # the learning rate at the lowest loss, after smoothing the losses as Adam smooths its gradients\n"
            "    avg = 0.0\n    best, best_lr = None, None\n    for i in range(len(losses)):\n        ...\n    return best_lr\n")
_SMOOTH = "        avg = beta * avg + (1 - beta) * losses[i]\n        smooth = avg / (1 - beta ** (i + 1))\n"
_KEEP = "        if best is None or smooth < best:\n            best, best_lr = smooth, lrs[i]\n"


def _p(smooth: str = _SMOOTH, keep: str = _KEEP) -> str:
    return _P_HEAD + "    avg = 0.0\n    best, best_lr = None, None\n    for i in range(len(losses)):\n" + smooth + keep + "    return best_lr\n"


def _p_close(got, want) -> bool:
    try:
        return not isinstance(got, (str, bytes, bool)) and math.isclose(float(got), float(want), rel_tol=1e-6)
    except (TypeError, ValueError):
        return False


def _p_show(v):
    try:
        if isinstance(v, (str, bytes, bool)) or v is None:
            return repr(v)
        return {"en": _sig(float(v)), "nl": _sig(float(v)).replace(".", ",")}
    except (TypeError, ValueError):
        return repr(v)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_R_WRONG = {
    "linear": _r("        lrs.append(lr_min + (lr_max - lr_min) * i / (steps - 1))\n"),
    "steps": _r("        lrs.append(lr_min * (lr_max / lr_min) ** (i / steps))\n"),
    "down": _r("        lrs.append(lr_max * (lr_min / lr_max) ** (i / (steps - 1)))\n"),
}
_P_WRONG = {
    "no-correction": _p(smooth="        avg = beta * avg + (1 - beta) * losses[i]\n        smooth = avg\n"),
    "raw": _p(smooth="        smooth = losses[i]\n"),
    "loss": _p(keep="        if best is None or smooth < best:\n            best, best_lr = smooth, smooth\n"),
    "highest": _p(keep="        if best is None or smooth > best:\n            best, best_lr = smooth, lrs[i]\n"),
}

EXERCISES = {
    "lr-range": FunctionBox(
        name="lr-range",
        function="lr_range",
        start=_R_START,
        solution=_r(),
        visible=[(0.001, 1.0, 4)],
        hidden=[(1e-4, 10.0, 6), (0.01, 0.1, 3), (0.5, 0.5 * 2 ** 8, 9)],
        show=_r_show,
        compare=_r_close,
        first=[_DOTS],
        mistakes=[
            Mistake("linear", wrong=_R_WRONG["linear"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: your learning rates are an equal step apart, not an equal factor. Most of them land near `lr_max`; on a log scale, equal factors spread them evenly. Multiply `lr_min` by `(lr_max / lr_min)` to the power `i / (steps - 1)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je learning rates liggen een gelijke stap uit elkaar, geen gelijke factor. De meeste komen dicht bij `lr_max`; op een logaritmische schaal spreiden gelijke factoren ze gelijkmatig. Vermenigvuldig `lr_min` met `(lr_max / lr_min)` tot de macht `i / (steps - 1)`."),
            Mistake("steps", wrong=_R_WRONG["steps"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: your last learning rate stops short of `lr_max`. With `steps` rates there are `steps - 1` factors between the first and the last: divide by `steps - 1`, not by `steps`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je laatste learning rate komt niet tot `lr_max`. Met `steps` rates zitten er `steps - 1` factoren tussen de eerste en de laatste: deel door `steps - 1`, niet door `steps`."),
            Mistake("down", wrong=_R_WRONG["down"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: your learning rates go down, from `lr_max` to `lr_min`. A range test goes up, from small to large, until the loss blows up.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je learning rates gaan omlaag, van `lr_max` naar `lr_min`. Een range test gaat omhoog, van klein naar groot, tot de loss ontploft."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with one line that adds learning rate `i` to `lrs`.",
                      "nl": "Vervang de `...` door één regel die learning rate `i` aan `lrs` toevoegt."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the smallest and the largest learning rate and how many, and the `def` line of your function has another number of names between its parentheses. Keep `(lr_min, lr_max, steps)`.",
                          "nl": "De check roept `{call}` aan, met de kleinste en de grootste learning rate en hoeveel, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(lr_min, lr_max, steps)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does every learning rate come out the same factor larger than the one before, from `lr_min` to `lr_max`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Wordt elke learning rate dezelfde factor groter dan de vorige, van `lr_min` tot `lr_max`?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for other ranges and other numbers of steps?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt het ook voor andere bereiken en andere aantallen stappen?"},
            "correct": {"en": "Your learning rates are an equal factor apart in all {n} cases, from `lr_min` up to `lr_max`: evenly spread on a log scale.",
                        "nl": "Je learning rates liggen in alle {n} gevallen een gelijke factor uit elkaar, van `lr_min` tot `lr_max`: gelijkmatig verdeeld op een logaritmische schaal."},
        },
    ),
    "range-pick": FunctionBox(
        name="range-pick",
        function="range_pick",
        start=_P_START,
        solution=_p(),
        visible=[([0.001, 0.01, 0.1, 1.0], [2.3, 2.0, 1.2, 5.0], 0.5)],
        hidden=[([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0], [2.3, 1.2, 2.2, 1.6, 1.4, 1.3, 5.0], 0.5),
                ([0.01, 0.03, 0.1, 0.3, 1.0], [2.3, 1.8, 1.1, 0.8, 0.9], 0.9),
                ([0.1, 0.2, 0.4, 0.8], [2.0, 1.6, 2.4, 9.0], 0.0)],
        show=_p_show,
        compare=_p_close,
        first=[_DOTS],
        mistakes=[
            Mistake("no-correction", wrong=_P_WRONG["no-correction"],
                    en="`{call}` gives {got}, and it should give {want}: the first learning rate wins, because your average is not corrected for its start at 0 the way Adam does it. Divide `avg` by `1 - beta ** (i + 1)`: a different number at every step, small at the first and close to 1 later. Dividing by the same number at every step does not change which one is lowest.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de eerste learning rate wint, omdat je gemiddelde niet gecorrigeerd is voor zijn start bij 0 zoals Adam dat doet. Deel `avg` door `1 - beta ** (i + 1)`: bij elke stap een ander getal, klein bij de eerste en dicht bij 1 later. Elke stap door hetzelfde getal delen verandert niet welke het laagst is."),
            Mistake("raw", wrong=_P_WRONG["raw"],
                    en="`{call}` gives {got}, and it should give {want}: you take the lowest loss of one step. One mini-batch can be lucky; smooth the losses first, with the running average `avg`, and take the lowest smoothed loss.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je neemt de laagste loss van één stap. Eén mini-batch kan geluk hebben; strijk de losses eerst glad, met het lopende gemiddelde `avg`, en neem de laagste gladgestreken loss."),
            Mistake("loss", wrong=_P_WRONG["loss"],
                    en="`{call}` gives {got}, and it should give {want}: that is the lowest smoothed loss, not the learning rate where it was. Keep `lrs[i]` in `best_lr`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is de laagste gladgestreken loss, niet de learning rate waar hij lag. Bewaar `lrs[i]` in `best_lr`."),
            Mistake("highest", wrong=_P_WRONG["highest"],
                    en="`{call}` gives {got}, and it should give {want}: you keep the highest smoothed loss, where the loss blew up. Keep the lowest: `smooth < best`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je bewaart de hoogste gladgestreken loss, waar de loss ontplofte. Bewaar de laagste: `smooth < best`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the body of the loop: the running average, its correction, and the best so far.",
                      "nl": "Vervang de `...` door het lijf van de loop: het lopende gemiddelde, de correctie ervan, en de beste tot nu toe."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the learning rates, the loss at each and beta, and the `def` line of your function has another number of names between its parentheses. Keep `(lrs, losses, beta)`.",
                          "nl": "De check roept `{call}` aan, met de learning rates, de loss bij elk en beta, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(lrs, losses, beta)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Do you keep a running average of the losses, correct it for its start at 0, and remember the learning rate where the corrected average was lowest?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Houd je een lopend gemiddelde van de losses bij, corrigeer je het voor zijn start bij 0, en onthoud je de learning rate waar het gecorrigeerde gemiddelde het laagst was?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does it work for other beta, also 0, and for losses with a lucky dip?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt het ook voor een andere beta, ook 0, en voor losses met een toevallige dip?"},
            "correct": {"en": "Your pick is right in all {n} cases: the learning rate at the lowest smoothed loss, not fooled by one lucky mini-batch, nor by the low start of an average.",
                        "nl": "Je keuze klopt in alle {n} gevallen: de learning rate bij de laagste gladgestreken loss, niet beetgenomen door één gelukkige mini-batch, en ook niet door de lage start van een gemiddelde."},
        },
    ),
}


def functions_of(range_text, pick_text):
    """The learner's lr_range and range_pick when their checks are right, else the course's; and whether each is theirs."""
    def get(text, name, course):
        if text:
            ran = run_code(text)
            if ran.error is None and callable(ran.names.get(name)):
                return ran.names[name], True
        return course, False
    r, r_yours = get(range_text, "lr_range", _course_range)
    p, p_yours = get(pick_text, "range_pick", _course_pick)
    return r, p, r_yours, p_yours


@functools.lru_cache(maxsize=4)
def sweep(values: tuple, seed: int = 0) -> tuple:
    return tuple(train(v, seed) for v in values)


# The lesson is done when these are correct. "sweep-values" is the choice question.
REQUIRED = ["lr-range", "range-pick", "sweep-values"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("lr-range", _R_START, "empty", None),
    ("lr-range", _r(), "correct", None),
    ("lr-range", _r("        lrs.append(lr_min * (lr_max / lr_min) ** (i / (steps - 1)) * 1.0)\n"), "correct", None),
    ("lr-range", _R_HEAD + "    import numpy as np\n    return list(np.geomspace(lr_min, lr_max, steps))\n", "correct", None),
    ("lr-range", _R_WRONG["linear"], "wrong", "linear"),
    ("lr-range", _R_WRONG["steps"], "wrong", "steps"),
    ("lr-range", _R_WRONG["down"], "wrong", "down"),
    ("range-pick", _P_START, "empty", None),
    ("range-pick", _p(), "correct", None),
] + [("range-pick", code, "wrong", key) for key, code in _P_WRONG.items()]
