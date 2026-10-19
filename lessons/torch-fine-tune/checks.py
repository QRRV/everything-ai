# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson torch-fine-tune: fine_tune, which trains the digit reader as the previous lesson leaves it
(frozen, with a new last layer from 64 to 10) in two phases: first only the new last layer, then every layer with a
learning rate ten times smaller. The check gives it a `train` that only notes which parameters each call would
change, with which learning rate and for how many epochs; the demo gives it the course's real training loop on the
100 labeled digits of the old scanner of the previous lesson, and tests on 1,000 such digits."""
import functools
import html
import re
import threading

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn

from everything_ai import paths
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit

DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md
LABELED = 100                    # the labeled digits of the new task: the first 100 of the training images
SPECKLE = 0.5                    # the old scanner adds to every pixel a random amount from 0 to 0.5
HEAD_LR, HEAD_EPOCHS = 0.01, 30  # phase 1: only the new last layer
TUNE_LR, TUNE_EPOCHS = 0.001, 30 # phase 2: every layer, with a learning rate ten times smaller
BATCH = 32
# The course's runs with the course's code, on one thread, for seeds 0 to 4: test images read right of 1,000, lowest
# and highest. "frozen" is phase 1 alone, "tuned" both phases, "big-lr" phase 2 with the learning rate of phase 1,
# "no-head" every layer from the start with the learning rate of phase 2, 30 epochs ("no-head-60": 60 epochs).
# "clothes-*" is the same on the first 100 images of Fashion-MNIST, "scratch" from the previous lesson: a DigitCNN
# from scratch, 60 epochs with a learning rate of 0.003. "1000-*" is the same with 1,000 labeled digits of the scanner.
# "gain" is how many more test digits both phases read right than phase 1 alone, per seed: the fewest and the most.
REFERENCE = {"frozen": (860, 883), "tuned": (879, 893), "big-lr": (776, 883), "no-head": (747, 844), "no-head-60": (761, 850),
             "clothes-frozen": (506, 539), "clothes-tuned": (566, 593), "clothes-scratch": (617, 713),
             "1000-scratch": (900, 920), "1000-frozen": (925, 938), "1000-tuned": (940, 951),
             "scratch": (629, 669), "gain": (3, 26)}


def _speckled(images: np.ndarray, seed: int) -> torch.Tensor:
    X = torch.tensor(images, dtype=torch.float32) / 255
    noise = SPECKLE * torch.rand(X.shape, generator=torch.Generator().manual_seed(seed))
    return (X + noise).clamp(0, 1)


@functools.lru_cache(maxsize=1)
def scanner_digits():
    """The digits as the old scanner gives them, as in the previous lesson: the 100 labeled training images and
    their digits, and the 1,000 test images and theirs. The same speckles every time."""
    d = np.load(paths.ROOT / DATA)
    X = _speckled(d["train_images"][:LABELED], seed=7)
    X_test = _speckled(d["test_images"], seed=8)
    return X, torch.tensor(d["train_labels"][:LABELED], dtype=torch.int64), X_test, torch.tensor(d["test_labels"], dtype=torch.int64)


def train(model, params, X, y, lr: float, epochs: int, seed: int = 0) -> None:
    """The course's training loop: mini-batches of 32 in a new order every epoch, cross-entropy, and the optimizer
    Adam, which a later lesson explains, on `params`, the parameters that should learn."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(X))
        for i in range(0, len(X), BATCH):
            batch = order[i:i + BATCH]
            loss = loss_fn(model(X[batch]), y[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()


def right(model, X, y) -> int:
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).sum().item()


def frozen_reader():
    """The digit reader as the previous lesson leaves it: every parameter frozen, embed passing the 64 hidden
    features on, and a new last layer from 64 to 10."""
    from everything_ai.models import digit_cnn
    model = digit_cnn()
    for p in model.parameters():
        p.requires_grad = False
    model.embed = nn.Identity()
    model.out = nn.Linear(64, 10)
    return model


def phases_picture(labels: dict) -> str:
    """The two phases side by side, the layers of the digit reader top to bottom: in phase 1 conv1, conv2 and hidden
    are frozen (grey) and only out learns (param); in phase 2 every layer learns, with a smaller learning rate."""
    W, COL, GAP = 340, 160, 20
    mono = "font-family:var(--ea-font-code)"
    frozen = "fill:var(--ea-ink);fill-opacity:0.06;stroke:var(--ea-ink-3);stroke-width:1.2"
    learns = "fill:var(--ea-param-wash);stroke:var(--ea-param);stroke-width:1.5"
    arrow = 'style="fill:none;stroke:var(--ea-ink-3);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"'
    parts = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    names = ["conv1", "conv2", "hidden", "out"]
    for col, (phase, rate, learning) in enumerate([(labels["phase1"], labels["rate1"], {"out"}),
                                                   (labels["phase2"], labels["rate2"], set(names))]):
        x0 = col * (COL + GAP)
        parts.append(f'<text x="{x0}" y="44" style="font-size:13px;font-weight:600;fill:var(--ea-ink)">{html.escape(phase)}</text>'
                     f'<text x="{x0}" y="61" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(rate)}</text>'
                     f'<text x="{x0}" y="76" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(labels["epochs"])}</text>')
        y = 88
        for k, name in enumerate(names):
            on = name in learning
            parts.append(f'<rect x="{x0 + 1}" y="{y}" width="{COL - 2}" height="34" rx="6" style="{learns if on else frozen}"/>'
                         f'<text x="{x0 + 10}" y="{y + 21}" style="{mono};font-size:13px;fill:var(--ea-ink)">{name}</text>'
                         f'<text x="{x0 + COL - 9}" y="{y + 21}" style="font-size:12px;font-style:italic;fill:var(--ea-ink-2);text-anchor:end">'
                         f'{html.escape(labels["learns"] if on else labels["frozen"])}</text>')
            y += 34
            if k < len(names) - 1:
                parts.append(f'<path d="M{x0 + COL / 2} {y + 2} V{y + 12} M{x0 + COL / 2 - 4} {y + 8} L{x0 + COL / 2} {y + 12} '
                             f'L{x0 + COL / 2 + 4} {y + 8}" {arrow}/>')
                y += 14
    H = 88 + 4 * 34 + 3 * 14 + 4
    text = labels["title"] + ". " + labels["alt"]
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:min(100%,{W}px)">'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(text)}" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(text)}</title>' + "".join(parts) + "</svg></figure>")


# The lesson is done when these are correct. "big-lr" and "more-digits" are the choice questions in the lesson.
REQUIRED = ["fine-tune", "big-lr", "more-digits"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code left in the box. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts again with the course's lines.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_CALL = "With `{call}`, your function trains `{got}`, and it should train `{want}`"
_CALL_NL = "Met `{call}` traint je functie `{got}`, en dat moet `{want}` zijn"

# What one check notes, per thread: the model the check made for this call of fine_tune, and the calls of its train.
_STATE = threading.local()


class _Reader:
    """The digit reader as the previous lesson leaves it, as a test case: written `reader` in the check's messages;
    every call gets a new copy, which the `train` of the same call looks at."""

    def __repr__(self):
        return "reader"

    def __deepcopy__(self, memo):
        torch.manual_seed(0)
        model = frozen_reader()
        _STATE.model, _STATE.calls = model, []
        return model


class _Train:
    """`train(params, lr, epochs)` as a test case: it trains nothing, but notes which parameters of the model the call
    would change (those it gets that still learn), with which learning rate, and for how many epochs."""

    def __repr__(self):
        return "train"

    def __deepcopy__(self, memo):
        model, calls = _STATE.model, _STATE.calls

        def train(params, lr, epochs):
            given = list(params)
            changes = [name for name, p in model.named_parameters() if p.requires_grad and any(p is q for q in given)]
            range(epochs)   # a number of epochs that is not a whole number stops here, as in the real training loop
            calls.append([changes, round(float(lr), 8), int(epochs)])
        return train


def _summary(value):
    """What the check compares: the calls of train, each [the parameters it would change, learning rate, epochs]."""
    return [list(call) for call in _STATE.calls]


_ALL = ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias", "hidden.weight", "hidden.bias", "out.weight", "out.bias"]
_HEAD = ["out.weight", "out.bias"]


def _rate(lr: float, lang: str) -> str:
    text = f"{lr:.8f}".rstrip("0").rstrip(".")
    return text.replace(".", ",") if lang == "nl" else text


def _shown(v):
    if not (isinstance(v, list) and all(isinstance(c, list) and len(c) == 3 for c in v)):
        return v
    if not v:
        return {"en": "nothing", "nl": "niets"}
    out = {}
    for lang, every, nothing in [("en", "all 8 parameters", "nothing"), ("nl", "alle 8 parameters", "niets")]:
        steps = []
        for k, (changes, lr, epochs) in enumerate(v):
            what = every if changes == _ALL else ", ".join(changes) if changes else nothing
            steps.append(f"{k + 1}. {what}: learning rate {_rate(lr, lang)}, {epochs} epochs")
        out[lang] = "; ".join(steps)
    return out


_F_HEAD = "def fine_tune(model, train):\n"
_F_ONE = "    train(model.out.parameters(), 0.01, 30)\n"
_F_THAW = "    for p in model.parameters():\n        p.requires_grad = True\n"
_F_TWO = "    train(model.parameters(), 0.001, 30)\n"
_F_START = (_F_HEAD + "    # phase 1: only the new last layer learns: 30 epochs, learning rate 0.01\n    ...\n"
            "    # unfreeze every parameter\n    for p in model.parameters():\n        ...\n"
            "    # phase 2: every parameter learns: 30 epochs, a learning rate ten times smaller\n    ...\n")


def _f(one: str = _F_ONE, thaw: str = _F_THAW, two: str = _F_TWO) -> str:
    return _F_HEAD + one + thaw + two


_F_WRONG = {
    "no-head-phase": _f(one=""),
    "still-frozen": _f(thaw=""),
    "same-lr": _f(two="    train(model.parameters(), 0.01, 30)\n"),
    "one-phase": _f(thaw="", two=""),
}
_F_CASES = [(_Reader(), _Train())]

EXERCISES = {
    "fine-tune": FunctionBox(
        name="fine-tune",
        function="fine_tune",
        start=_F_START,
        solution=_f(),
        visible=[_F_CASES[0]],
        collect=_summary,
        show=_shown,
        first=[_DOTS],
        mistakes=[
            Mistake("no-head-phase", matches=lambda got, text: isinstance(got, list) and bool(got) and got[0][0] == _ALL,
                    en=_CALL + ": your first training already changes every layer, while the new last layer is still random. Its first errors say nothing about digits, and they pull the borrowed layers along. Phase 1 first: only `model.out.parameters()`.",
                    nl=_CALL_NL + ": je eerste training verandert al elke layer, terwijl de nieuwe laatste layer nog willekeurig is. Zijn eerste fouten zeggen niets over cijfers, en ze trekken de geleende layers mee. Eerst fase 1: alleen `model.out.parameters()`."),
            Mistake("one-phase", matches=lambda got, text: isinstance(got, list) and len(got) == 1 and got[0][0] == _HEAD,
                    en=_CALL + ": only phase 1, as in the previous lesson. Phase 2 is missing: a second `train`, after the loop that unfreezes, with every parameter and a learning rate ten times smaller.",
                    nl=_CALL_NL + ": alleen fase 1, zoals in de vorige les. Fase 2 ontbreekt: een tweede `train`, na de loop die ontdooit, met elke parameter en een learning rate die tien keer zo klein is."),
            Mistake("still-frozen", matches=lambda got, text: isinstance(got, list) and len(got) == 2 and got[1][0] == _HEAD,
                    en=_CALL + ": your second training changes only the new last layer again. For every layer to learn, each parameter needs `p.requires_grad = True`, and `train` needs all of them: `model.parameters()`.",
                    nl=_CALL_NL + ": je tweede training verandert weer alleen de nieuwe laatste layer. Om elke layer te laten leren, heeft elke parameter `p.requires_grad = True` nodig, en heeft `train` ze allemaal nodig: `model.parameters()`."),
            Mistake("same-lr", matches=lambda got, text: isinstance(got, list) and len(got) >= 2 and got[0][1] == HEAD_LR and got[1][0] == _ALL and got[1][1] >= HEAD_LR,
                    en=_CALL + ": in phase 2 your learning rate is as large as in phase 1, or larger. Steps that large carry the borrowed features far from what they knew. Ten times smaller: 0.001.",
                    nl=_CALL_NL + ": in fase 2 is je learning rate even groot als in fase 1, of groter. Zulke grote stappen brengen de geleende features ver weg van wat ze wisten. Tien keer zo klein: 0,001."),
            Mistake("train-values", error=lambda e: isinstance(e, TypeError) and "train()" in str(e) and "positional argument" in str(e),
                    en="With `{call}`, your function stops with an error: `train` gets another number of values than it takes. The `train` of this exercise already knows the model and the 100 digits, so it takes three values: `train(params, lr, epochs)`, without `model`.",
                    nl="Met `{call}` stopt je functie met een fout: `train` krijgt een ander aantal waarden dan hij aanneemt. De `train` van deze oefening kent het model en de 100 cijfers al, dus hij krijgt drie waarden: `train(params, lr, epochs)`, zonder `model`."),
            Mistake("no-call", error=lambda e: isinstance(e, TypeError) and "'method' object is not iterable" in str(e),
                    en="With `{call}`, your function stops with an error: `train` gets `parameters` without `()`. Without the parentheses, Python does not call the method, and `train` gets the method itself instead of the parameters. Write `model.parameters()` or `model.out.parameters()`.",
                    nl="Met `{call}` stopt je functie met een fout: `train` krijgt `parameters` zonder `()`. Zonder de haakjes roept Python de methode niet aan, en krijgt `train` de methode zelf in plaats van de parameters. Schrijf `model.parameters()` of `model.out.parameters()`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...` with your own code: phase 1, a call of `train`; the line in the loop that unfreezes; and phase 2, another call of `train`.",
                      "nl": "Vervang de drie `...` door je eigen code: fase 1, een aanroep van `train`; de regel in de loop die ontdooit; en fase 2, nog een aanroep van `train`."},
            "arguments": {"en": "The check calls `{call}`, and the `def` line of your function has another number of names between its parentheses. Keep `(model, train)`.",
                          "nl": "De check roept `{call}` aan, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(model, train)`."},
            "wrong": {"en": _CALL + ". The check shows, per call of `train`, which parameters it would change, with which learning rate and for how many epochs. Compare yours with the comments in the box.",
                      "nl": _CALL_NL + ". De check laat per aanroep van `train` zien welke parameters die zou veranderen, met welke learning rate en hoeveel epochs. Vergelijk de jouwe met het commentaar in het vak."},
            "correct": {"en": "First only the new last layer, then every layer with a learning rate ten times smaller: the two phases of fine-tuning.",
                        "nl": "Eerst alleen de nieuwe laatste layer, dan elke layer met een learning rate die tien keer zo klein is: de twee fases van fine-tuning."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("fine-tune", _F_START, "empty", None),
    ("fine-tune", _f(), "correct", None),
    ("fine-tune", _f(thaw="    model.requires_grad_(True)\n", two="    train(list(model.parameters()), 0.01 / 10, 30)\n"), "correct", None),
    ("fine-tune", _f(one="    train(model.parameters(), 0.01, 30)\n"), "correct", None),   # the frozen layers do not change
    ("fine-tune", _f(two="    train(model.parameters(), 0.0005, 30)\n"), "wrong", None),
    ("fine-tune", _f(one="    train(model.out.parameters(), 0.001, 30)\n"), "wrong", None),   # phase 2 is right: not same-lr
    ("fine-tune", _f(one=_F_ONE * 2), "wrong", None),   # phase 1 twice, then phase 2: not still-frozen
    ("fine-tune", _f(thaw=_F_THAW + "    pass\n", two=""), "wrong", "one-phase"),
    ("fine-tune", _f(one="    train(model, model.out.parameters(), 0.01, 30)\n"), "error", "train-values"),
    ("fine-tune", _f(two="    train(model.parameters, 0.001, 30)\n"), "error", "no-call"),
] + [("fine-tune", code, "wrong", key) for key, code in _F_WRONG.items()]


# --- the demo: phase 1 alone against both phases, on the 100 labeled digits of the old scanner ------------------------

def _runs(fine_tune, seed: int) -> dict:
    """One run of each: the frozen digit reader with only its new last layer trained (phase 1), and the digit reader
    through fine_tune with the course's training loop on the 100 labeled digits."""
    X, y, X_test, y_test = scanner_digits()
    torch.manual_seed(seed)
    frozen = frozen_reader()
    train(frozen, frozen.out.parameters(), X, y, HEAD_LR, HEAD_EPOCHS, seed)
    out = {"frozen": right(frozen, X_test, y_test)}
    torch.manual_seed(seed)
    model = frozen_reader()
    fine_tune(model, lambda params, lr, epochs: train(model, params, X, y, lr, epochs, seed))
    out["tuned"] = right(model, X_test, y_test)
    return out


def course_fine_tune():
    return run_code(_f()).names["fine_tune"]


def demo(text) -> dict:
    """The demo with the learner's fine_tune once it is right, else the course's, seed 0, on one thread: the test
    digits read right after phase 1 and after both phases, and whose fine_tune it was."""
    ran = run_code(text) if text else None
    fn = ran.names.get("fine_tune") if ran is not None and ran.error is None else None
    yours = callable(fn)
    try:
        return {**with_time_limit(lambda: _runs(fn if yours else course_fine_tune(), 0), 60), "yours": yours}
    except Exception:
        return {**with_time_limit(lambda: _runs(course_fine_tune(), 0), 60), "yours": False}
