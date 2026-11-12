# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-fine-tune-photos: phase_two, phase 2 of fine-tuning in a training loop of its own, which
unfreezes every parameter of a model and trains them all further, in mini-batches of 20. It is checked on a tiny
made-up model, so it works without ResNet-18. ResNet-18 (47 MB, from Hugging Face) and Caltech-101 (137 MB, from
CaltechDATA) the lesson downloads once, when the learner asks for them (everything_ai.hub, everything_ai.caltech): the
worked example and the demo need them, the exercises do not. The slider uses what the course measured with them
(REFERENCE)."""
import errno
import functools
import html
import re
import urllib.error

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn

from everything_ai import caltech, chart, hub, live, models, netview, yours
from everything_ai.lesson import download_card
from everything_ai.check import FunctionBox, Mistake, run_code

DOWNLOAD_LINE = {
    "en": "Works on every computer. ResNet-18, 47 MB, is downloaded once from Hugging Face, and Caltech-101, 137 MB of photos, from Caltech, when you click their button in the lesson; after that, the lesson works without a connection. The exercises work without them too.",
    "nl": "Werkt op elke computer. ResNet-18, 47 MB, wordt één keer gedownload van Hugging Face, en Caltech-101, 137 MB aan foto's, van Caltech, als je in de les op de knop ervoor klikt; daarna werkt de les zonder verbinding. De oefeningen werken ook zonder.",
}
CLASSES = list(caltech.CHECKED)   # the 10 classes of Caltech-101 the lesson uses, in alphabetical order: label 0 to 9
NAMES = {
    "en": {"chandelier": "chandelier", "dragonfly": "dragonfly", "ewer": "ewer", "lotus": "lotus", "nautilus": "nautilus",
           "rooster": "rooster", "starfish": "starfish", "sunflower": "sunflower", "water_lilly": "water lily",
           "windsor_chair": "Windsor chair"},
    "nl": {"chandelier": "kroonluchter", "dragonfly": "libel", "ewer": "schenkkan", "lotus": "lotus", "nautilus": "nautilus",
           "rooster": "haan", "starfish": "zeester", "sunflower": "zonnebloem", "water_lilly": "waterlelie",
           "windsor_chair": "windsorstoel"},
}
TRAIN, TEST = 20, (20, 30)        # of each class's 30 photos: the first 20 to learn from, photos 20 to 29 to test on
HEAD_LR, HEAD_EPOCHS = 0.1, 100   # phase 1: the new last layer, on the features of all 200 training photos at once
TUNE_LR, TUNE_EPOCHS, BATCH = 0.003, 2, 20   # phase 2: every parameter, in mini-batches of 20 photos
RATES = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]   # the stops of the slider: learning rates of phase 2
SEEDS = 5
STEPS = 10 * TRAIN // BATCH   # the steps of phase 2 in one epoch: 200 training photos in mini-batches of 20
PAIR = ("lotus", "water_lilly")   # the two classes that every mistake after phase 1 mixes up
SHOWN = [10 * CLASSES.index(name) + k for name in PAIR for k in range(10)]   # their 20 test photos, in the order of split()
EXAMPLE = SHOWN[0]    # the test photo the demo shows the feature maps of: the first lotus
WATCHED = "layer4"    # the layer it shows: ResNet-18's last block of convolutions, which phase 2 changes most
MOMENTS = (0, 1, 2, 5, 10, 15, 20)   # the steps of phase 2 to rewind to: 0 is after phase 1, before phase 2 changes anything

# What the course measured: everything_ai.models.resnet18 with its pinned file, the 30 photos of each class
# (everything_ai.caltech.checked), prepared with `prepare`; phase 1 with phase_one(seed) and phase 2 with the course's
# phase_two, torch.manual_seed(seed) as its first line, for the seeds 0 to 4, on 4 threads. "phase1" is how many of
# the 100 test photos each seed gets right after phase 1; "rates" the same after TUNE_EPOCHS epochs of phase 2, per
# learning rate of RATES. "classes" is how many classes the test photos get after phase 2, per learning rate, on the
# seed that gives the most. "train" is how many of the 200 training photos phase 1 gets right; "mixed1" how many of
# its mistakes on the test photos take a lotus for a water lily or the other way around, and "mixed" the same after
# phase 2 with TUNE_LR. "ten" is phase 2 with TUNE_LR for 10 epochs instead of TUNE_EPOCHS. "train-mode" is phase 2 with TUNE_LR and model.train() before it, so that batch normalization
# uses the numbers of each batch. "scratch" is a new ResNet-18 with random weights and 10 classes, in train mode, trained
# like phase 2 for 10 epochs with a learning rate of 0.01, for the seeds 0 to 2. "runaway" is seed 0 with a learning
# rate of 0.03: the loss of the batch at step 7 of phase 2, the first step whose loss is nan, how many of the 200
# training photos are right after 2 epochs, and how many test photos after 6. tests/test_cv_fine_tune_photos.py
# measures them again when EAI_RESNET18_FILE names the downloaded file and EAI_CALTECH_DIR the unpacked photos.
# A DataLoader draws a number from torch's random generator each time it starts going through its data, even without
# shuffle=True; a test between two epochs would change the order of the next one. That is why the worked example and
# the demo test after phase 2, not after each epoch.
REFERENCE = {
    "phase1": [90, 95, 95, 95, 93],
    "rates": {0.0003: [90, 96, 95, 95, 93], 0.001: [90, 96, 96, 96, 93], 0.003: [92, 96, 95, 96, 94],
              0.01: [93, 95, 94, 96, 93], 0.03: [10, 10, 10, 10, 10], 0.1: [10, 10, 10, 10, 10]},
    "classes": {0.0003: 10, 0.001: 10, 0.003: 10, 0.01: 10, 0.03: 1, 0.1: 1},
    "train": [200, 200, 200, 200, 200],
    "ten": [93, 97, 95, 96, 92],
    "train-mode": [92, 94, 92, 92, 94],
    "mixed1": [10, 4, 5, 5, 6],
    "mixed": [8, 3, 5, 4, 6],
    "scratch": [30, 38, 29],
    "runaway": {"loss7": 49.48, "nan-at": 9, "train": 20, "epochs6": 10},
}
# How long the worked example took on the course's computer, with 4 cores, in seconds, rounded: phase 1 with the
# features of the 200 training photos (2.5 to 3.7 over 5 seeds), and one epoch of phase 2 (12.4 to 15.0 for 2 epochs).
TIMES = {"phase1": 3, "epoch": 7}
PARAMS = 11_181_642   # the parameters of ResNet-18 with a last layer for 10 classes: what learns in phase 2
HEAD = 5_130          # those of the last layer alone, 512 · 10 + 10: what learns in phase 1


def number(x: int, lang: str) -> str:
    """A whole number with the thousands mark of the language: 11,181,642, or 11.181.642 in Dutch."""
    text = f"{x:,}"
    return text.replace(",", ".") if lang == "nl" else text


def rate(lr: float, lang: str) -> str:
    """A learning rate as the lesson writes it: 0.003, or 0,003 in Dutch."""
    text = f"{lr:g}"
    return text.replace(".", ",") if lang == "nl" else text


def low_high(values) -> tuple:
    return min(values), max(values)


def after(lr: float) -> list:
    """Test photos right after phase 2 with learning rate lr, per seed."""
    return REFERENCE["rates"][lr]


def gains(lr: float = TUNE_LR) -> list:
    """How many more test photos each seed gets right after phase 2 than after phase 1; fewer is below 0."""
    return [a - b for a, b in zip(after(lr), REFERENCE["phase1"])]


def gain_words(lo: int, hi: int, lang: str) -> str:
    """The gains of gains(), lowest and highest, in words: "0 to 2 more", "between 1 fewer and 2 more", or "1 to 3
    fewer"."""
    if hi < 0:
        if lo == hi:
            return f"{-hi} fewer" if lang == "en" else f"{-hi} minder"
        return f"{-hi} to {-lo} fewer" if lang == "en" else f"{-hi} tot {-lo} minder"
    if lo < 0 <= hi:
        return f"between {-lo} fewer and {hi} more" if lang == "en" else f"tussen {-lo} minder en {hi} meer"
    if lo == hi:
        return f"{lo} more" if lang == "en" else f"{lo} meer"
    return f"{lo} to {hi} more" if lang == "en" else f"{lo} tot {hi} meer"


# --- the picture: the two phases, with the layers of ResNet-18 -------------------------------------------------------

def phases_picture(labels: dict) -> str:
    """The two phases side by side, the parts of ResNet-18 from top to bottom: in phase 1 everything is frozen (grey)
    but the new last layer, fc (param); in phase 2 every part learns. `labels` has "title", "phase1", "rate1",
    "count1", "phase2", "rate2", "count2", "learns", "frozen", "fc" and "alt"."""
    W, COL, GAP = 380, 180, 20
    mono = "font-family:var(--ea-font-code)"
    frozen = "fill:var(--ea-ink);fill-opacity:0.06;stroke:var(--ea-ink-3);stroke-width:1.2"
    learns = "fill:var(--ea-param-wash);stroke:var(--ea-param);stroke-width:1.5"
    arrow = 'style="fill:none;stroke:var(--ea-ink-3);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"'
    parts = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    names = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]
    BOX, STEP = 28, 12
    for col, (phase, rate_, count, learning) in enumerate([(labels["phase1"], labels["rate1"], labels["count1"], {"fc"}),
                                                           (labels["phase2"], labels["rate2"], labels["count2"], set(names))]):
        x0 = col * (COL + GAP)
        parts.append(f'<text x="{x0}" y="44" style="font-size:13px;font-weight:600;fill:var(--ea-ink)">{html.escape(phase)}</text>'
                     f'<text x="{x0}" y="61" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(rate_)}</text>'
                     f'<text x="{x0}" y="76" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(count)}</text>')
        y = 88
        for k, name in enumerate(names):
            on = name in learning
            shown = labels["fc"] if name == "fc" else name
            parts.append(f'<rect x="{x0 + 1}" y="{y}" width="{COL - 2}" height="{BOX}" rx="6" style="{learns if on else frozen}"/>'
                         f'<text x="{x0 + 10}" y="{y + 18.5}" style="{mono};font-size:12.5px;fill:var(--ea-ink)">{html.escape(shown)}</text>'
                         f'<text x="{x0 + COL - 9}" y="{y + 18.5}" style="font-size:12px;font-style:italic;fill:var(--ea-ink-2);text-anchor:end">'
                         f'{html.escape(labels["learns"] if on else labels["frozen"])}</text>')
            y += BOX
            if k < len(names) - 1:
                parts.append(f'<path d="M{x0 + COL / 2} {y + 2} V{y + STEP - 2} M{x0 + COL / 2 - 4} {y + STEP - 6} L{x0 + COL / 2} {y + STEP - 2} '
                             f'L{x0 + COL / 2 + 4} {y + STEP - 6}" {arrow}/>')
                y += STEP
    H = 88 + len(names) * BOX + (len(names) - 1) * STEP + 4
    text = labels["title"] + ". " + labels["alt"]
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:min(100%,{W}px)">'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(text)}" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(text)}</title>' + "".join(parts) + "</svg></figure>")


# --- the slider: the learning rate of phase 2 against test photos right ----------------------------------------------

def curve(lr: float, labels: dict, lang: str) -> str:
    """The test photos right after phase 2, of 100, against its learning rate: for each rate a bar from the lowest to
    the highest of the 5 seeds, with a line through the middles; phase 1 alone as a band, and guessing at 10. The
    stop lr is marked. All in ink: test photos right is an accuracy, which training does not optimise (the colour
    dictionary). `labels` has "title", "x", "y", "phase1", "phase2", "guess" and "alt"."""
    lo, hi = low_high(REFERENCE["phase1"])
    panel = chart.Panel((0.0002, 0.15), (0, 100), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                        yticks=[0, 25, 50, 75, 100], xlog=True)
    panel.rect(0.0002, lo, 0.15, hi, role="ink-3")
    panel.line([0.0002, 0.15], [10, 10], role="ink-3", width=1.2, dashed=True)
    middles = []
    for r in RATES:
        a, b = low_high(after(r))
        middles.append((a + b) / 2)
        panel.line([r, r], [a, b], role="ink", width=3 if r == lr else 1.6)
    panel.line(RATES, middles, role="ink", width=1.4, dashed=True)
    a, b = low_high(after(lr))
    panel.points([lr, lr], [a, b], role="ink", r=5)
    panel.text(0.00025, lo - 9, labels["phase1"], role="ink-2", anchor="start")
    panel.text(0.009, 55, labels["phase2"], role="ink", anchor="end")
    panel.text(0.00025, 16, labels["guess"], role="ink-3", anchor="start")
    svg = chart.figure([panel], alt=labels["alt"], width=460, height=290)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


# --- ResNet-18 and Caltech-101 in the lesson -------------------------------------------------------------------------

def photos(name: str) -> list:
    """The 30 photos of a class the lesson uses, without people: everything_ai.caltech.checked."""
    return caltech.checked(name, 30)


def ready() -> bool:
    return hub.pinned(*models.RESNET18) is not None and caltech.unpacked()


def downloads_or_stop(button, texts: dict):
    """everything_ai.models.resnet18, which gives a new ResNet-18 with ImageNet's weights each time it is called, for
    the lesson's cells, once ResNet-18 is downloaded and Caltech-101 unpacked. When either is not there yet, the cell
    stops with `texts["ask"]` and the button; a click downloads what is missing and unpacks the photos, with the
    progress in the cell. While it downloads, the lesson's other cells wait. `texts` has "ask", "busy" ({name},
    {done} and {total} in MB), "unpacking" ({done} photos), "no_net", "not_pinned", "no_room" and "failed" ({why}):
    why it did not work; and "here"."""
    import marimo as mo
    failed = None
    if not ready() and button.value:
        try:
            for name, pin in (("ResNet-18", models.RESNET18), ("Caltech-101", caltech.ZIP)):
                if hub.pinned(*pin) is not None:
                    continue
                total = round(hub.PINNED[pin]["size"] / 1e6)
                last = [-1]

                def progress(done, size, name=name, total=total, last=last):
                    mb = int(done / 1e6)
                    if mb // 5 != last[0] // 5:   # every 5 MB
                        last[0] = mb
                        mo.output.replace(mo.md(texts["busy"].format(name=name, done=mb, total=total)))
                hub.download(*pin, progress=progress)
            caltech.unpack(progress=lambda done: mo.output.replace(mo.md(texts["unpacking"].format(done=done))))
        except hub.NotPinned:
            failed = texts["not_pinned"]
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            failed = texts["no_net"]
        except OSError as e:
            failed = texts["no_room"] if e.errno == errno.ENOSPC else texts["failed"].format(why=html.escape(str(e)))
    if not ready():
        mo.stop(True, download_card(texts["ask"], button, failed))
    mo.output.replace(mo.md(texts["here"]))
    return models.resnet18


MEAN = np.array([0.485, 0.456, 0.406])   # ImageNet's mean and standard deviation per channel, as in cv-prepare-photo
STD = np.array([0.229, 0.224, 0.225])


def prepare(photo) -> np.ndarray:
    """The five steps of cv-prepare-photo, for a photo of any size: the shorter side to 256 pixels (bilinear), the
    middle 224 by 224, divided by 255, ImageNet's mean and standard deviation per channel, and the channels first: an
    array of shape (3, 224, 224), as 32-bit floats."""
    from PIL import Image
    with Image.open(photo) as image:
        image = image.convert("RGB")
        w, h = image.size
        s = 256 / min(w, h)
        small = np.asarray(image.resize((round(w * s), round(h * s)), Image.BILINEAR))
    top, left = (small.shape[0] - 224) // 2, (small.shape[1] - 224) // 2
    crop = small[top:top + 224, left:left + 224]
    return ((crop / 255 - MEAN) / STD).astype(np.float32).transpose(2, 0, 1)


_PREPARED = {}


def split():
    """The prepared photos and labels, as the worked example makes them: the first 20 photos of each class to learn
    from, and photos 20 to 29 of each class to test on. Prepared once, the first time."""
    if not _PREPARED:
        train, y_train, test, y_test = [], [], [], []
        for label, name in enumerate(CLASSES):
            ps = photos(name)
            for p in ps[:TRAIN]:
                train.append(prepare(p))
                y_train.append(label)
            for p in ps[TEST[0]:TEST[1]]:
                test.append(prepare(p))
                y_test.append(label)
        _PREPARED["data"] = (torch.tensor(np.array(train)), torch.tensor(y_train), torch.tensor(np.array(test)), torch.tensor(y_test))
    return _PREPARED["data"]


def guesses(model, X) -> torch.Tensor:
    """The class the model gives each photo, 50 photos at a time, as the worked example tests."""
    from torch.utils.data import DataLoader, TensorDataset
    out = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(X), batch_size=50):
            out += model(xb).argmax(dim=1).tolist()
    return torch.tensor(out)


def right(model, X, y) -> int:
    return (guesses(model, X) == y).sum().item()


def phase_one(model, X, y, seed: int = 0):
    """Phase 1, as in the worked example: the 512 features of every photo through the frozen model, once, 50 photos
    at a time; then a new last layer from 512 to 10, trained on them for HEAD_EPOCHS epochs with SGD."""
    from torch.utils.data import DataLoader, TensorDataset
    model.fc = nn.Identity()
    rows = []
    with torch.no_grad():
        for xb, yb in DataLoader(TensorDataset(X, y), batch_size=50):
            for row in model(xb):
                rows.append(row.numpy())
    F = torch.tensor(np.array(rows))
    torch.manual_seed(seed)
    model.fc = nn.Linear(512, len(CLASSES))
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.fc.parameters(), lr=HEAD_LR)
    for epoch in range(HEAD_EPOCHS):
        loss = loss_fn(model.fc(F), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


# The lesson is done when these are correct. "params" and "too-big" are the choice questions, "worth-it" the open one.
REQUIRED = ["params", "phase-two", "too-big", "worth-it"]

# --- Exercise 2: phase_two -------------------------------------------------------------------------------------------

_HEAD = ("import torch\nfrom torch import nn\nfrom torch.utils.data import DataLoader, TensorDataset\n\n\n"
         "def phase_two(model, X, y, lr, epochs):\n    torch.manual_seed({seed})\n")


def _phase_two(thaw: str = "        p.requires_grad = True\n", opt: str = "torch.optim.SGD(model.parameters(), lr=lr)",
               loss: str = "loss_fn(model(xb), yb)", comments: bool = False, seed: int = 0) -> str:
    first = "    # unfreeze every parameter\n" if comments else ""
    second = "    # SGD on every parameter, with learning rate lr\n" if comments else ""
    third = "            # the loss of the model's logits for this batch, against its labels\n" if comments else ""
    return (_HEAD.format(seed=seed) + first + "    for p in model.parameters():\n" + thaw + second + f"    opt = {opt}\n"
            "    loss_fn = nn.CrossEntropyLoss()\n    loader = DataLoader(TensorDataset(X, y), batch_size=20, shuffle=True)\n"
            "    for epoch in range(epochs):\n        for xb, yb in loader:\n" + third + f"            loss = {loss}\n"
            "            opt.zero_grad()\n            loss.backward()\n            opt.step()\n    return model\n")


_START = _phase_two(thaw="        ...\n", opt="...", loss="...", comments=True)
_SOLUTION = _phase_two()


def course_phase_two(seed: int = 0):
    """The course's phase_two, with torch.manual_seed(seed) as its first line: what REFERENCE measured."""
    return run_code(_phase_two(seed=seed)).names["phase_two"]


class _Tiny(nn.Module):
    """A tiny made-up model as phase 1 leaves it: a first layer `body` from 4 numbers to 6, frozen, dropout, and a last
    layer `fc` from 6 numbers to 3 classes, which learns; in eval mode, as `pretrained()` gives ResNet-18. The dropout
    does nothing in eval mode, and lets the check see a `model.train()`, which phase 2 leaves out. The same weights for
    the same seed."""

    def __init__(self, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(100 + seed)
            self.body = nn.Linear(4, 6)
            self.fc = nn.Linear(6, 3)
        self.drop = nn.Dropout(0.5)
        for p in self.body.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        return self.fc(self.drop(torch.relu(self.body(x))))


def _made_up(n: int, seed: int):
    """n made-up examples of 3 classes, 4 numbers each: a centre per class, and noise around it."""
    g = torch.Generator().manual_seed(seed)
    centres = 2 * torch.randn(3, 4, generator=g)
    y = torch.arange(n) % 3
    return centres[y] + torch.randn(n, 4, generator=g), y


class _Model:
    """The tiny model as a test case: written `model` in the check's messages; every call gets a new one, which also
    keeps its examples, so that the check can work out its loss on them afterwards."""

    def __init__(self, seed: int, n: int):
        self.seed, self.n = seed, n

    def __repr__(self):
        return "model"

    def __deepcopy__(self, memo):
        model = _Tiny(self.seed)
        model._ea_seed, model._ea_case = self.seed, _made_up(self.n, self.seed)
        return model


class _Data:
    """The examples or the labels of a case, written as their name in the check's messages."""

    def __init__(self, case: _Model, which: int, shown: str):
        self.case, self.which, self.shown = case, which, shown

    def __repr__(self):
        return self.shown

    def __deepcopy__(self, memo):
        return _made_up(self.case.n, self.case.seed)[self.which].clone()


def _case(seed: int, n: int, lr: float, epochs: int):
    m = _Model(seed, n)
    return (m, _Data(m, 0, "X"), _Data(m, 1, "y"), lr, epochs)


def _summary(model):
    """What the check compares: whether the first layer and the last layer changed, and the loss of the model on its
    examples afterwards, to 3 decimals; anything that is not the tiny model, as it is."""
    if not isinstance(model, _Tiny) or not hasattr(model, "_ea_case"):
        return model
    start = _Tiny(model._ea_seed)
    X, y = model._ea_case
    with torch.no_grad():
        loss = nn.functional.cross_entropy(model(X), y).item()
    return [not torch.equal(model.body.weight, start.body.weight), not torch.equal(model.fc.weight, start.fc.weight),
            round(loss, 3)]


_LAYERS = {(True, True): {"en": "in which both layers learned", "nl": "waarin beide layers leerden"},
           (True, False): {"en": "in which the first layer learned and the last layer did not change",
                           "nl": "waarin de eerste layer leerde en de laatste niet veranderde"},
           (False, True): {"en": "in which the first layer did not change and the last layer learned",
                           "nl": "waarin de eerste layer niet veranderde en de laatste leerde"},
           (False, False): {"en": "in which neither layer changed", "nl": "waarin geen van beide layers veranderde"}}


def _shown(v):
    if isinstance(v, list) and len(v) == 3 and isinstance(v[0], bool):
        body, head, loss = v
        layers = _LAYERS[(body, head)]
        comma = f"{loss:.3f}".replace(".", ",")
        return {"en": f"a model with a loss of {loss:.3f} on its examples, {layers['en']}",
                "nl": f"een model met een loss van {comma} op zijn voorbeelden, {layers['nl']}"}
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            one = f"{v.item():.3f}"
            return {"en": f"a tensor with one number, {one}", "nl": f"een tensor met één getal, {one.replace('.', ',')}"}
        return {"en": f"a tensor of shape {tuple(v.shape)}", "nl": f"een tensor met shape {tuple(v.shape)}"}
    return repr(v)


def _is_model(got) -> bool:
    return isinstance(got, list) and len(got) == 3 and isinstance(got[0], bool)


def _all_frozen(e: BaseException):
    """Whether every parameter of the model in the learner's phase_two was frozen when it stopped with error e: True
    or False, read from the function's own `model` in the traceback; None when it is not there."""
    tb = e.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        model = frame.f_locals.get("model")
        if frame.f_code.co_name == "phase_two" and isinstance(model, nn.Module):
            return not any(p.requires_grad for p in model.parameters())
        tb = tb.tb_next
    return None


def _no_grad_error(e: BaseException) -> bool:
    return isinstance(e, RuntimeError) and "does not require grad" in str(e)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_CALL = "`{call}` gives {got}, and it should give {want}"
_CALL_NL = "`{call}` geeft {got}, maar de goede uitkomst is {want}"
_CASES = [_case(0, 50, 0.1, 2), _case(1, 60, 0.05, 3), _case(2, 45, 0.2, 1)]

EXERCISES = {
    "phase-two": FunctionBox(
        name="phase-two",
        function="phase_two",
        start=_START,
        solution=_SOLUTION,
        visible=_CASES[:1],
        hidden=_CASES[1:],
        collect=_summary,
        show=_shown,
        first=[_DOTS],
        mistakes=[
            Mistake("batch-labels", syntax=r"loss_fn\(\s*model\(\s*xb\s*\)\s*,\s*y\s*\)",
                    en="Your code stops with an error at `{line}`: the logits are those of the 20 photos of the batch, `xb`, and `y` holds the labels of all photos. Give the loss the labels of the batch: `yb`.",
                    nl="Je code stopt met een fout bij `{line}`: de logits zijn die van de 20 foto's van de batch, `xb`, en `y` bevat de labels van alle foto's. Geef de loss de labels van de batch: `yb`."),
            Mistake("loss-order", syntax=r"loss_fn\(\s*yb?\s*,",
                    en="Your code stops with an error at `{line}`: `nn.CrossEntropyLoss` gets the logits first and the labels second: `loss_fn(model(xb), yb)`.",
                    nl="Je code stopt met een fout bij `{line}`: `nn.CrossEntropyLoss` krijgt eerst de logits en dan de labels: `loss_fn(model(xb), yb)`."),
            Mistake("all-frozen", error=lambda e: _no_grad_error(e) and _all_frozen(e) is True,
                    en="With `{call}`, your function stops with an error at `loss.backward()`: no parameter of the model learns, so there is no gradient to compute. Your loop freezes every parameter; unfreeze them instead: `p.requires_grad = True`.",
                    nl="Met `{call}` stopt je functie met een fout bij `loss.backward()`: geen enkele parameter van het model leert, dus er valt geen gradient te berekenen. Je loop bevriest elke parameter; ontdooi ze in plaats daarvan: `p.requires_grad = True`."),
            Mistake("no-way-back", error=lambda e: _no_grad_error(e) and _all_frozen(e) is False,
                    en="With `{call}`, your function stops with an error at `loss.backward()`: your `loss` has no way back to the parameters, so there is no gradient to compute. Compute it straight from the logits of the model, outside `torch.no_grad()` and without `.detach()`: `loss_fn(model(xb), yb)`.",
                    nl="Met `{call}` stopt je functie met een fout bij `loss.backward()`: je `loss` heeft geen weg terug naar de parameters, dus er valt geen gradient te berekenen. Bereken hem rechtstreeks uit de logits van het model, buiten `torch.no_grad()` en zonder `.detach()`: `loss_fn(model(xb), yb)`."),
            Mistake("head-only", matches=lambda got, text: _is_model(got) and not got[0] and re.search(r"fc\.parameters\(", text) is not None,
                    en=_CALL + ". Your optimizer gets only the parameters of the last layer, `model.fc.parameters()`, so the rest of the model stays as it was: that is phase 1 again. In phase 2 every parameter learns: `model.parameters()`.",
                    nl=_CALL_NL + ". Je optimizer krijgt alleen de parameters van de laatste layer, `model.fc.parameters()`, dus de rest van het model blijft zoals het was: dat is weer fase 1. In fase 2 leert elke parameter: `model.parameters()`."),
            Mistake("still-frozen", matches=lambda got, text: _is_model(got) and not got[0],
                    en=_CALL + ". The first layer of the model is still frozen: a parameter with `requires_grad` off gets no gradient, so SGD skips it. In the loop, unfreeze every parameter: `p.requires_grad = True`.",
                    nl=_CALL_NL + ". De eerste layer van het model is nog bevroren: een parameter met `requires_grad` uit krijgt geen gradient, dus SGD slaat hem over. Ontdooi in de loop elke parameter: `p.requires_grad = True`."),
            Mistake("fixed-lr", matches=lambda got, text: _is_model(got) and re.search(r"lr\s*=\s*[0-9.]", text) is not None,
                    en=_CALL + ". Your optimizer has a learning rate of its own, written as a number. The check tries other learning rates too, such as 0.05; take the one `phase_two` gets: `lr=lr`.",
                    nl=_CALL_NL + ". Je optimizer heeft een eigen learning rate, als getal geschreven. De check probeert ook andere learning rates, zoals 0,05; neem die die `phase_two` krijgt: `lr=lr`."),
            Mistake("head-left-out", matches=lambda got, text: _is_model(got) and got[0] and not got[1],
                    en=_CALL + ". The first layer learns, but the last layer stays as it was: your optimizer leaves it out. Give the optimizer every parameter of the model: `model.parameters()`.",
                    nl=_CALL_NL + ". De eerste layer leert, maar de laatste blijft zoals hij was: je optimizer laat hem weg. Geef de optimizer elke parameter van het model: `model.parameters()`."),
            Mistake("train-mode", matches=lambda got, text: _is_model(got) and re.search(r"\.train\(\s*(True)?\s*\)", text) is not None,
                    en=_CALL + ". Your function puts the model in train mode, with `.train()`. Phase 2 keeps it in eval mode, as phase 1 left it, so that batch normalization keeps the averages it measured on ImageNet: leave out `.train()`.",
                    nl=_CALL_NL + ". Je functie zet het model in train mode, met `.train()`. Fase 2 houdt het in eval mode, zoals fase 1 het achterliet, zodat batch normalization de gemiddelden houdt die hij op ImageNet mat: laat `.train()` weg."),
            Mistake("all-photos", wrong=_phase_two(loss="loss_fn(model(X), y)"),
                    en=_CALL + ". Every step computes the loss on all examples, `X` and `y`, instead of on the batch: the loader goes round, but its batches are not used. Use the batch: `loss_fn(model(xb), yb)`.",
                    nl=_CALL_NL + ". Elke stap berekent de loss op alle voorbeelden, `X` en `y`, in plaats van op de batch: de loader gaat rond, maar zijn batches worden niet gebruikt. Gebruik de batch: `loss_fn(model(xb), yb)`."),
            Mistake("not-a-model", matches=lambda got, text: not _is_model(got),
                    en="`{call}` gives {got}, and it should give the model itself, trained further: keep `return model` as the last line.",
                    nl="`{call}` geeft {got}, en dat moet het model zelf zijn, verder getraind: houd `return model` als laatste regel."),
            Mistake("other-random", matches=lambda got, text: _is_model(got) and got[0] and got[1],
                    en=_CALL + ". Both layers learn, but the model ends up elsewhere. Keep the course's lines as they are, `torch.manual_seed(0)` first, and give the optimizer every parameter with learning rate `lr`, and the loss the logits and labels of the batch.",
                    nl=_CALL_NL + ". Beide layers leren, maar het model komt ergens anders uit. Laat de regels van de course zoals ze zijn, met `torch.manual_seed(0)` eerst, en geef de optimizer elke parameter met learning rate `lr`, en de loss de logits en labels van de batch."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the three `...` in `phase_two`: step 3 of the recipe.",
                      "nl": "Vervang de drie `...` in `phase_two`: stap 3 van het recept."},
            "arguments": {"en": "The check calls `{call}`, with the model, the examples, the labels, the learning rate and the number of epochs, and the `def` line of your function has another number of names between its parentheses. Keep `(model, X, y, lr, epochs)`.",
                          "nl": "De check roept `{call}` aan, met het model, de voorbeelden, de labels, de learning rate en het aantal epochs, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(model, X, y, lr, epochs)`."},
            "wrong": {"en": _CALL + ": both layers learn, for 2 epochs with learning rate 0.1. Step 3 of the recipe?",
                      "nl": _CALL_NL + ": beide layers leren, 2 epochs met learning rate 0,1. Stap 3 van het recept?"},
            "almost": {"en": "Right for the example. But " + _CALL + ". Does your function take the learning rate from `lr`, and the batches from the loader?",
                       "nl": "Goed voor het voorbeeld. Maar " + _CALL_NL + ". Haalt je functie de learning rate uit `lr`, en de batches uit de loader?"},
            "correct": {"en": "Your function is right in all {n} cases: every parameter unfrozen, SGD on all of them, and a step per batch. The demo below gives it ResNet-18 after phase 1.",
                        "nl": "Je functie klopt in alle {n} gevallen: elke parameter ontdooid, SGD op allemaal, en een stap per batch. De demo hieronder geeft hem ResNet-18 na fase 1."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("phase-two", _START, "empty", None),
    ("phase-two", _SOLUTION, "correct", None),
    ("phase-two", _phase_two(comments=True), "correct", None),
    ("phase-two", _phase_two(thaw="        p.requires_grad_(True)\n", opt="torch.optim.SGD(params=model.parameters(), lr=lr)",
                             loss="nn.functional.cross_entropy(model(xb), yb)"), "correct", None),
    ("phase-two", _phase_two(loss="loss_fn(model(xb), y)"), "error", "batch-labels"),
    ("phase-two", _phase_two(loss="loss_fn(yb, model(xb))"), "error", "loss-order"),
    ("phase-two", _phase_two(opt="torch.optim.SGD(model.fc.parameters(), lr=lr)"), "wrong", "head-only"),
    ("phase-two", _phase_two(thaw="        pass\n"), "wrong", "still-frozen"),
    ("phase-two", _phase_two(thaw="        p.requires_grad = False\n"), "error", "all-frozen"),
    ("phase-two", _phase_two(loss="loss_fn(model(xb), yb).detach()"), "error", "no-way-back"),
    ("phase-two", _phase_two(opt="torch.optim.SGD(model.body.parameters(), lr=lr)"), "wrong", "head-left-out"),
    ("phase-two", _SOLUTION.replace("    for epoch in range(epochs):\n", "    model.train()\n    for epoch in range(epochs):\n"), "wrong", "train-mode"),
    ("phase-two", _phase_two(opt="torch.optim.SGD(model.parameters(), lr=0.003)"), "wrong", "fixed-lr"),
    ("phase-two", _phase_two(opt="torch.optim.SGD(model.parameters(), lr=0.1)"), "wrong", "fixed-lr"),
    ("phase-two", _phase_two(loss="loss_fn(model(X), y)"), "wrong", "all-photos"),
    ("phase-two", _SOLUTION.replace("    return model\n", "    return loss\n"), "wrong", "not-a-model"),
    ("phase-two", _SOLUTION.replace("    torch.manual_seed(0)\n", "    torch.manual_seed(1)\n"), "wrong", "other-random"),
    ("phase-two", _phase_two(loss="..."), "wrong", "dots-left"),
]


# --- the demo: the learner's phase_two on ResNet-18 after phase 1 ----------------------------------------------------

def per_class(model, X, y) -> list:
    """How many of the 10 test photos of each class the model gets right, in the order of CLASSES."""
    hits = guesses(model, X) == y
    return [int(hits[y == k].sum()) for k in range(len(CLASSES))]


def mixed(model, X, y) -> int:
    """The test photos the model takes for a lotus when they are a water lily, or the other way around."""
    lotus, lily = CLASSES.index("lotus"), CLASSES.index("water_lilly")
    g = guesses(model, X)
    return int(((y == lotus) & (g == lily)).sum() + ((y == lily) & (g == lotus)).sum())


def loss_text(loss: float, lang: str) -> str:
    """A loss as the demo writes it, to 3 decimals: 0.014, or 0,014 in Dutch."""
    text = f"{loss:.3f}"
    return text.replace(".", ",") if lang == "nl" else text


def _key(photo) -> bytes:
    """A prepared photo, recognised by 31 of its numbers: how the demo finds the label of each photo in a batch."""
    return photo.flatten()[::4999].numpy().tobytes()


@functools.lru_cache(maxsize=1)
def thumbnails() -> tuple:
    """The test photos SHOWN, small: the middle square of each, 64 by 64 pixels, as the demo shows them with its
    guesses."""
    from PIL import Image
    out = []
    for photo in [p for name in PAIR for p in photos(name)[TEST[0]:TEST[1]]]:
        with Image.open(photo) as image:
            image = image.convert("RGB")
            w, h = image.size
            s = min(w, h)
            square = image.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
            out.append(np.asarray(square.resize((64, 64), Image.BILINEAR)))
    return tuple(out)


def now_pictures(seen: dict, title: str, lang: str, uid: str) -> list:
    """What ResNet-18 makes at one moment of phase 2, as `look` or the rewind gives it: the class it guesses for each
    of the 20 test photos of lotus and water lily, and how sure it is of that guess, as softmax gives it; and under
    them its layer WATCHED on the first lotus (layer4.1, the last part of layer4), under `title` with {name} the
    layer's name. `uid` keeps the masks of the layer apart from those of another picture on the page."""
    said = [NAMES[lang][CLASSES[g]] for g in seen["guesses"]]
    right = [NAMES[lang][CLASSES[k]] for k in seen["truth"]]
    heads = {"en": "ResNet-18's guess for each test photo of lotus and water lily, and how sure it is",
             "nl": "Wat ResNet-18 gokt voor elke testfoto van lotus en waterlelie, en hoe zeker het is"}
    alt = {"en": "The 10 test photos of lotus and the 10 of water lily, each with the class ResNet-18 guesses, how sure it is, and in brackets the right class where it is wrong: ",
           "nl": "De 10 testfoto's van lotus en de 10 van waterlelie, elk met de klasse die ResNet-18 gokt, hoe zeker het is, en tussen haakjes de goede klasse waar het fout is: "}
    sure = [f"{round(100 * s)}%" for s in seen["sure"]]
    picture = yours.guesses(list(thumbnails()), said, right, alt[lang] + "; ".join(
        f"{r}: {g}, {s}" for r, g, s in zip(right, said, sure)) + ".", size=48, title=heads[lang], notes=sure)
    return netview.moment(picture, seen.get("layer"), title, lang, uid)


def losses_picture(losses: list, labels: dict, lang: str, at: int | None = None) -> str:
    """The loss of each batch of phase 2, as the live chart drew it, with a line at step `at` for rewinding."""
    run = live.Live(labels, max(len(losses), TUNE_EPOCHS * STEPS), lang=lang, low=0.0)
    run.xs, run.ys = list(range(1, len(losses) + 1)), list(losses)
    return run.picture(at=at)


def step_frame(result: dict, k: int, labels: dict, title: str, lang: str) -> str:
    """The rewind at step k of phase 2 (0: after phase 1): the loss of each batch with a line at k, then what ResNet-18
    guessed after that step, how sure, and its layer layer4, as they were then (decision 1.17)."""
    return live.rewound(losses_picture(result["losses"], labels, lang, at=k),
                        now_pictures(result["looked"][k], title, lang, f"nv-tune-step-{k}"))


def demo(pretrained, text, progress=None) -> dict:
    """The learner's phase_two once it is right, else the course's, on ResNet-18 after phase 1 (seed 0), with TUNE_LR
    for TUNE_EPOCHS epochs: how many test photos of each class it gets right after phase 1 and after phase 2, how
    many mistakes mix up lotus and water lily, and whose function it was; and for the rewind, the loss of each batch
    of phase 2 ("losses") and at the steps of MOMENTS what the model makes ("looked": its guesses for the photos SHOWN,
    how sure it is of each, and its layer WATCHED on EXAMPLE). `progress` hears {"phase": 1} when phase 1 starts, and
    {"step": k, "loss": v, "look": f} after each forward pass of phase 2 with gradients: the loss of that batch,
    worked out from the logits by a hook on the model, since the learner's loop keeps its own, and f() what the model
    makes before that step changes it, as now_pictures draws it (decision 1.17). It looks in eval mode without a graph and without drawing a random
    number, and puts the model back in its own mode, so phase 2 stays the same. If the learner's
    function fails here after all, the course's runs instead, and progress hears {"phase": 1} again. It runs without
    the time limit of a check, which would make PyTorch calculate with one thread: phase 2 takes long enough as it is.
    The cell passes `pretrained`, which it got from downloads_or_stop, so that it runs only once the downloads are
    there."""
    ran = run_code(text) if text else None
    fn = ran.names.get("phase_two") if ran is not None and ran.error is None else None
    yours = callable(fn)
    X, y, X_test, y_test = split()
    labels = {_key(photo): int(label) for photo, label in zip(X, y)}
    tell = progress or (lambda report: None)
    truth = [int(y_test[place]) for place in SHOWN]
    kept = {}

    def run(f):
        tell({"phase": 1})
        model = phase_one(pretrained(), X, y)
        before = per_class(model, X_test, y_test)
        steps, losses = [0], []
        kept.clear()
        kept["losses"], kept["looked"] = losses, {}

        def look(m) -> dict:
            with netview.looking(m):
                p = torch.softmax(m(X_test[SHOWN]), dim=1)
            return {"guesses": p.argmax(dim=1).tolist(), "sure": [float(v) for v in p.max(dim=1).values], "truth": truth,
                    "layer": netview.watched(m, X_test[EXAMPLE:EXAMPLE + 1], WATCHED)}

        now = {}   # the model the hook fired on, and what it makes before this step changes it, once something looks

        def look_now() -> dict:   # the chart looks at most every 2 seconds (Live.watch), not at every step
            if "seen" not in now:
                now["seen"] = look(now["model"])
            return now["seen"]

        def hook(module, inputs, output):
            if not torch.is_grad_enabled():   # also while the hook looks: no graph there
                return
            known = [labels.get(_key(photo)) for photo in inputs[0]]
            if None in known:
                return
            now.clear()
            now["model"] = module   # also when the learner's loop trains a copy of the model
            try:   # the course's pictures never stop the learner's phase 2
                if steps[0] in MOMENTS:
                    kept["looked"][steps[0]] = look_now()
            except Exception:
                pass
            steps[0] += 1
            loss = nn.functional.cross_entropy(output.detach(), torch.tensor(known)).item()
            losses.append(loss)
            try:
                tell({"step": steps[0], "loss": loss, "look": look_now})
            except Exception:
                pass
        handle = model.register_forward_hook(hook)
        try:
            model = f(model, X, y, TUNE_LR, TUNE_EPOCHS)
        finally:
            handle.remove()
        model.eval()
        if steps[0] in MOMENTS:   # after the last step
            try:
                kept["looked"][steps[0]] = look(model)
            except Exception:
                pass
        return before, per_class(model, X_test, y_test), mixed(model, X_test, y_test)
    try:
        before, after_, lotus = run(fn if yours else course_phase_two())
    except Exception:
        yours = False
        before, after_, lotus = run(course_phase_two())
    return {"before": before, "after": after_, "mixed": lotus, "yours": yours, "truth": truth, **kept}
