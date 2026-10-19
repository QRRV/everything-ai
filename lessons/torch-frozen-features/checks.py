# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson torch-frozen-features: new_head, which freezes every layer of the course's digit reader and
gives it a new last layer on its 64 hidden features. The new task: digits from an old scanner that puts grey speckles
on every image (QMNIST with noise), with only 100 labeled ones. The demo trains the new last layer of the learner's
model on them, and a DigitCNN from scratch, and tests both and the digit reader as it is on 1,000 such digits."""
import copy
import functools
import html
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn

from everything_ai import paths
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit

DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md
LABELED = 100                    # the labeled digits of the new task: the first 100 of the training images
SPECKLE = 0.5                     # the scanner adds to every pixel a random amount from 0 to 0.5
HEAD_LR, HEAD_EPOCHS = 0.01, 30   # the new last layer, on frozen features
SCRATCH_LR, SCRATCH_EPOCHS = 0.003, 60
BATCH = 32
# The course's runs with the course's code, on one thread, for seeds 0 to 4: test images read right of 1,000, lowest and
# highest. "clothes" is the same pair of runs on the first 100 images of Fashion-MNIST, tested on its 1,000 test images.
REFERENCE = {"as-is": (729, 729), "scratch": (629, 669), "frozen": (860, 883),
             "clothes-scratch": (617, 713), "clothes-frozen": (506, 539)}


def _speckled(images: np.ndarray, seed: int) -> torch.Tensor:
    X = torch.tensor(images, dtype=torch.float32) / 255
    noise = SPECKLE * torch.rand(X.shape, generator=torch.Generator().manual_seed(seed))
    return (X + noise).clamp(0, 1)


@functools.lru_cache(maxsize=1)
def scanner_digits():
    """The digits as the old scanner gives them, pixels from 0 to 1 with speckles: the 100 labeled training images
    and their digits, and the 1,000 test images and theirs. The same speckles every time."""
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


def speckle_picture(labels: dict) -> str:
    """Three test digits as the course's set has them, and as the old scanner gives them: grey speckles everywhere."""
    d = np.load(paths.ROOT / DATA)
    clean = torch.tensor(d["test_images"][:3], dtype=torch.float32) / 255
    noisy = scanner_digits()[2][:3]
    S, GAP, TOP = 3, 12, 30
    parts = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    for row, (images, name) in enumerate([(clean, labels["clean"]), (noisy, labels["scanner"])]):
        y0 = TOP + row * (28 * S + 34)
        parts.append(f'<text x="0" y="{y0 + 12}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(name)}</text>')
        for k in range(3):
            x0 = k * (28 * S + GAP)
            img = images[k].reshape(28, 28)
            rects = []
            for i in range(28):
                for j in range(28):
                    v = float(img[i, j])
                    if v > 0.02:
                        rects.append(f'<rect x="{x0 + j * S}" y="{y0 + 18 + i * S}" width="{S}" height="{S}" style="fill:var(--ea-ink);fill-opacity:{v:.2f}"/>')
            parts.append(f'<rect x="{x0}" y="{y0 + 18}" width="{28 * S}" height="{28 * S}" style="fill:var(--ea-page);stroke:var(--ea-ink-3);stroke-width:1"/>' + "".join(rects))
    W, H = 3 * 28 * S + 2 * GAP, TOP + 2 * (28 * S + 34)
    text = labels["title"] + ". " + labels["alt"]
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:min(100%,{W}px)">'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(text)}" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(text)}</title>' + "".join(parts) + "</svg></figure>")


def layers_picture(labels: dict) -> str:
    """The layers of the digit reader, top to bottom: conv1, conv2 and hidden frozen (grey, with their numbers kept),
    and the old embed and out replaced by one new layer from 64 to 10 (param: the only part that learns)."""
    W = 330
    mono = "font-family:var(--ea-font-code)"
    parts = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    frozen = "fill:var(--ea-ink);fill-opacity:0.06;stroke:var(--ea-ink-3);stroke-width:1.2"
    learns = "fill:var(--ea-param-wash);stroke:var(--ea-param);stroke-width:1.5"
    arrow = 'style="fill:none;stroke:var(--ea-ink-3);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"'
    rows = [("conv1", labels["conv1"], frozen, labels["frozen"]), ("conv2", labels["conv2"], frozen, labels["frozen"]),
            ("hidden", labels["hidden"], frozen, labels["frozen"]),
            ("out", labels["out"], learns, labels["learns"])]
    y = 28
    for k, (name, note, style, state) in enumerate(rows):
        parts.append(f'<rect x="1" y="{y}" width="{W - 2}" height="40" rx="6" style="{style}"/>'
                     f'<text x="12" y="{y + 17}" style="{mono};font-size:13px;fill:var(--ea-ink)">{name}</text>'
                     f'<text x="12" y="{y + 33}" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(note)}</text>'
                     f'<text x="{W - 10}" y="{y + 17}" style="font-size:12px;font-style:italic;fill:var(--ea-ink-2);text-anchor:end">{html.escape(state)}</text>')
        y += 40
        if k < len(rows) - 1:
            parts.append(f'<path d="M{W / 2} {y + 3} V{y + 17} M{W / 2 - 5} {y + 12} L{W / 2} {y + 17} L{W / 2 + 5} {y + 12}" {arrow}/>')
            y += 20
    parts.append(f'<text x="0" y="{y + 20}" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(labels["gone"])}</text>')
    y += 28
    text = labels["title"] + ". " + labels["alt"]
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:min(100%,{W}px)">'
            f'<svg viewBox="0 0 {W} {y}" role="img" aria-label="{html.escape(text)}" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(text)}</title>' + "".join(parts) + "</svg></figure>")


# The lesson is done when these are correct. "same-optimizer" and "clothes" are the choice questions in the lesson.
REQUIRED = ["new-head", "same-optimizer", "clothes"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_CALL = "`{call}` gives `{got}`, and it should give `{want}`"
_CALL_NL = "`{call}` geeft `{got}`, en dat moet `{want}` zijn"


class _Reader:
    """The digit reader as a test case: written `reader` in the check's messages; every call gets a new copy of the
    course's digit reader, after torch.manual_seed(0), so that a new layer starts the same in every run."""

    def __init__(self, shown: str, seed: int | None = None):
        self.shown, self.seed = shown, seed

    def __repr__(self):
        return self.shown

    def __deepcopy__(self, memo):
        from everything_ai.models import DigitCNN, digit_cnn
        if self.seed is None:
            model = digit_cnn()
        else:   # another DigitCNN, with other numbers: a function that ignores its model gives itself away
            torch.manual_seed(self.seed)
            model = DigitCNN()
        torch.manual_seed(0)
        return model


def _summary(model):
    """What the check compares: the parameters that still learn, the shape of the weight of `out`, and the 10 logits
    of the first test digit, rounded; anything that is not a model, as it is."""
    if not isinstance(model, nn.Module):
        return model
    learning = [name for name, p in model.named_parameters() if p.requires_grad]
    out = getattr(model, "out", None)
    shape = list(out.weight.shape) if isinstance(out, nn.Linear) else None
    try:
        with torch.no_grad():
            logits = [round(v, 3) for v in model(scanner_digits()[2][:1])[0].tolist()]
    except Exception:   # "error-embed": the old embed, from 64 to 2, still in front of a new layer made for 64
        embed = getattr(model, "embed", None)
        old = isinstance(embed, nn.Linear) and embed.out_features == 2 and shape is not None and shape[1] == 64
        logits = "error-embed" if old else "error"
    return [learning, shape, logits]


def _shown(v):
    if isinstance(v, list) and len(v) == 3 and isinstance(v[0], list):
        learning = ", ".join(v[0]) if v[0] else "nothing"
        learning_nl = ", ".join(v[0]) if v[0] else "niets"
        shape = "×".join(str(s) for s in v[1]) if v[1] else "?"
        if v[2] in ("error", "error-embed"):
            return {"en": f"learning: {learning}; out.weight {shape}; the model stops with an error",
                    "nl": f"leert: {learning_nl}; out.weight {shape}; het model stopt met een fout"}
        first = [f"{x:.3f}" for x in v[2][:2]]
        return {"en": f"learning: {learning}; out.weight {shape}; first logits {', '.join(first)}",
                "nl": f"leert: {learning_nl}; out.weight {shape}; eerste logits {' en '.join(x.replace('.', ',') for x in first)}"}
    return v


_N_HEAD = "from torch import nn\n\n\ndef new_head(model):\n"
_N_FREEZE = "    for p in model.parameters():\n        p.requires_grad = False\n"
_N_HEADS = "    model.embed = nn.Identity()\n    model.out = nn.Linear(64, 10)\n"
_N_START = (_N_HEAD + "    # 2. freeze every parameter: the optimizer may not change it\n    for p in model.parameters():\n        ...\n"
            "    # 3. pass the 64 hidden numbers on as they are, and a new last layer from 64 to 10\n    model.embed = ...\n"
            "    model.out = ...\n    return model\n")


def _n(freeze: str = _N_FREEZE, heads: str = _N_HEADS) -> str:
    return _N_HEAD + freeze + heads + "    return model\n"


_N_WRONG = {
    "not-frozen": _n(freeze=""),
    "head-frozen": _n(freeze="", heads=_N_HEADS + _N_FREEZE),
    "two-features": _n(heads="    model.out = nn.Linear(2, 10)\n"),
}
_N_CASES = [(_Reader("reader"),), (_Reader("other", seed=3),)]

EXERCISES = {
    "new-head": FunctionBox(
        name="new-head",
        function="new_head",
        start=_N_START,
        solution=_n(),
        visible=[_N_CASES[0]],
        hidden=[_N_CASES[1]],
        collect=_summary,
        show=_shown,
        first=[_DOTS],
        mistakes=[
            Mistake("no-identity", matches=lambda got, text: isinstance(got, list) and len(got) == 3 and got[2] == "error-embed",
                    en="Your model stops at your new last layer: it gets the 2 numbers of `embed`, and it was made for 64. Pass the 64 hidden numbers on as they are: `model.embed = nn.Identity()`.",
                    nl="Je model stopt bij je nieuwe laatste layer: die krijgt de 2 getallen van `embed`, en is gemaakt voor 64. Geef de 64 verborgen getallen door zoals ze zijn: `model.embed = nn.Identity()`."),
            Mistake("swapped", matches=lambda got, text: isinstance(got, list) and len(got) == 3 and got[1] == [64, 10],
                    en=_CALL + ": `out.weight` has the shape 64×10, so your new layer goes from 10 numbers to 64. `nn.Linear(64, 10)` takes the 64 hidden numbers and gives 10 logits.",
                    nl=_CALL_NL + ": `out.weight` heeft de shape 64×10, dus je nieuwe layer gaat van 10 getallen naar 64. `nn.Linear(64, 10)` neemt de 64 verborgen getallen en geeft 10 logits."),
            Mistake("other-random", matches=lambda got, text: isinstance(got, list) and len(got) == 3 and got[0] == ["out.weight", "out.bias"]
                    and got[1] == [10, 64] and isinstance(got[2], list),
                    en=_CALL + ": the right parameters learn, and `out` has the right shape, but the logits differ. They depend on the random start of the new layer: make only the one new `nn.Linear(64, 10)`, and nothing else that is random, such as a new model.",
                    nl=_CALL_NL + ": de goede parameters leren, en `out` heeft de goede shape, maar de logits verschillen. Die hangen af van het willekeurige begin van de nieuwe layer: maak alleen die ene nieuwe `nn.Linear(64, 10)`, en verder niets willekeurigs, zoals een nieuw model."),
            Mistake("not-frozen", wrong=_N_WRONG["not-frozen"],
                    en=_CALL + ": every layer still learns, so training changes the borrowed features too. Freeze every parameter first: `p.requires_grad = False` in the loop.",
                    nl=_CALL_NL + ": elke layer leert nog, dus het trainen verandert ook de geleende features. Bevries eerst elke parameter: `p.requires_grad = False` in de loop."),
            Mistake("head-frozen", wrong=_N_WRONG["head-frozen"],
                    en=_CALL + ": nothing learns, also not the new last layer: you freeze after you made it. Freeze first, then make the new layer: a new `nn.Linear` learns.",
                    nl=_CALL_NL + ": niets leert, ook de nieuwe laatste layer niet: je bevriest nadat je hem gemaakt hebt. Bevries eerst, maak dan de nieuwe layer: een nieuwe `nn.Linear` leert."),
            Mistake("two-features", wrong=_N_WRONG["two-features"],
                    en=_CALL + ": your new last layer gets the 2 numbers of `embed`, a part of the old last layers. Let it get the 64 hidden numbers: `model.embed = nn.Identity()`, and `nn.Linear(64, 10)`.",
                    nl=_CALL_NL + ": je nieuwe laatste layer krijgt de 2 getallen van `embed`, een deel van de oude laatste layers. Laat hem de 64 verborgen getallen krijgen: `model.embed = nn.Identity()`, en `nn.Linear(64, 10)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: in the loop, every parameter frozen; then `embed` as an `nn.Identity()`, and a new `out` from 64 to 10.",
                      "nl": "Vervang de `...`: in de loop elke parameter bevroren; dan `embed` als een `nn.Identity()`, en een nieuwe `out` van 64 naar 10."},
            "arguments": {"en": "The check calls `{call}`, with the digit reader, and the `def` line of your function has another number of names between its parentheses. Keep `(model)`.",
                          "nl": "De check roept `{call}` aan, met de cijferlezer, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(model)`."},
            "wrong": {"en": _CALL + ": the check shows which parameters still learn, and the shape of the weight of `out`. Steps 2 and 3 of the recipe, in that order?",
                      "nl": _CALL_NL + ": de check laat zien welke parameters nog leren, en de shape van de weight van `out`. Stap 2 en 3 van het recept, in die volgorde?"},
            "almost": {"en": "Right for the digit reader. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function change the model it gets, `model`, and give that back?",
                       "nl": "Goed voor de cijferlezer. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Verandert je functie het model dat hij krijgt, `model`, en geeft hij dat terug?"},
            "correct": {"en": "Your function freezes the digit reader and gives it a new last layer on its 64 hidden numbers: only `out.weight` and `out.bias`, 650 numbers, still learn.",
                        "nl": "Je functie bevriest de cijferlezer en geeft hem een nieuwe laatste layer op zijn 64 verborgen getallen: alleen `out.weight` en `out.bias`, 650 getallen, leren nog."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("new-head", _N_START, "empty", None),
    ("new-head", _n(), "correct", None),
    ("new-head", _n(freeze="    model.requires_grad_(False)\n"), "correct", None),
    ("new-head", _n(heads="    model.out = nn.Linear(64, 10)\n"), "wrong", "no-identity"),
    ("new-head", _n(heads="    model.embed = nn.Identity()\n    model.out = nn.Linear(10, 64)\n"), "wrong", "swapped"),
    ("new-head", _n(heads="    model.embed = nn.Identity()\n    nn.Linear(64, 10)\n    model.out = nn.Linear(64, 10)\n"), "wrong", "other-random"),
] + [("new-head", code, "wrong", key) for key, code in _N_WRONG.items()]


# --- the demo: the new last layer on 100 labeled digits of the scanner, against a DigitCNN from scratch -------------------

def _runs(new_head, seed: int) -> dict:
    """One run of each: the digit reader as it is, a new DigitCNN with a new last layer trained from scratch on the 100
    labeled digits, and the digit reader through new_head with only its new last layer trained on them."""
    from everything_ai.models import DigitCNN, digit_cnn
    X, y, X_test, y_test = scanner_digits()
    out = {"as-is": right(digit_cnn(), X_test, y_test)}
    torch.manual_seed(seed)
    scratch = DigitCNN()
    scratch.embed, scratch.out = nn.Identity(), nn.Linear(64, 10)
    train(scratch, list(scratch.parameters()), X, y, SCRATCH_LR, SCRATCH_EPOCHS, seed)
    out["scratch"] = right(scratch, X_test, y_test)
    torch.manual_seed(seed)
    frozen = new_head(digit_cnn())
    train(frozen, [p for p in frozen.parameters() if p.requires_grad], X, y, HEAD_LR, HEAD_EPOCHS, seed)
    out["frozen"] = right(frozen, X_test, y_test)
    return out


def course_new_head():
    return run_code(_n()).names["new_head"]


def demo(text) -> dict | str:
    """The demo with the learner's new_head once it is right, else the course's, seed 0, on one thread: the test
    digits read right of each way, and whose new_head it was. If the learner's new_head fails here after all, the
    course's runs instead."""
    ran = run_code(text) if text else None
    fn = ran.names.get("new_head") if ran is not None and ran.error is None else None
    yours = callable(fn)
    fn = fn if yours else course_new_head()
    try:
        return {**with_time_limit(lambda: _runs(fn, 0), 60), "yours": yours}
    except Exception:
        return {**with_time_limit(lambda: _runs(course_new_head(), 0), 60), "yours": False}
