# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson cv-color-cnn: channel_stats, the mean and the standard deviation of each color channel
of a set of photos, and ColorCNN, the SmallCNN of cnn-pytorch for color photos of 32 by 32, each saved in the
learner's project (work/projects/cnn/channels.py and color_cnn.py) when right. Both are checked on made-up tensors, so
they work without CIFAR-10. CIFAR-10 (170 MB, from the University of Toronto) the lesson downloads once, when the
learner asks for it (everything_ai.hub, everything_ai.cifar): the worked example, the picture of its photos and the
demo need it. The demo trains the learner's ColorCNN with their Trainer on the first 6,000 training photos, normalized
with their channel_stats, and tests it on the first 1,000 test photos, as the clothes of cnn-pytorch."""
import errno
import functools
import html
import re
import urllib.error

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import chart, cifar, hub, live, projects
from everything_ai.lesson import download_card
from everything_ai.check import ClassBox, Mistake, ScenarioBox, Timeout, run_code, with_time_limit
from everything_ai import netview, yours

# The demo trains with the learner's Trainer of nn-trainer and counts with their count_params of cnn-pytorch; where one
# is missing or broken, the course's version stands in (template 1.3).
BUILDS_ON = (("nn", "train_epoch", "nn-trainer", "train-epoch"), ("nn", "trainer", "nn-trainer", "trainer"),
             ("cnn", "count", "cnn-pytorch", "count-params"))
projects.builds_on(*BUILDS_ON)

DOWNLOAD_LINE = {
    "en": "Works on every computer. CIFAR-10, 170 MB of photos, is downloaded once from the University of Toronto when you click its button in the lesson; after that, the lesson works without a connection. The exercises work without it too.",
    "nl": "Werkt op elke computer. CIFAR-10, 170 MB aan foto's, wordt één keer gedownload van de University of Toronto als je in de les op de knop ervoor klikt; daarna werkt de les zonder verbinding. De oefeningen werken ook zonder.",
}
NAMES = {
    "en": dict(zip(cifar.KINDS, ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"))),
    "nl": dict(zip(cifar.KINDS, ("vliegtuig", "auto", "vogel", "kat", "hert", "hond", "kikker", "paard", "schip", "vrachtwagen"))),
}
N_TRAIN, N_TEST = 6_000, 1_000   # the first photos of each set: as many as the clothes of data/fashion-mnist-small.npz
EPOCHS, LR = 20, 0.01            # the demo; the guess and the picture also have the 5 epochs of cnn-pytorch
PARAMS = 11_642                  # of ColorCNN: 3·8·9 + 8, 8·16·9 + 16 and 1,024·10 + 10
SHOWN = 2                        # checked test photos of each kind in the demo, with the network's guess
ONE_EACH = tuple(cifar.checked(kind, 1)[0] for kind in cifar.KINDS)   # the first of those of each kind, for the run
PHOTO = 82                       # the test photo the run shows its first layer on: a car, in cifar.CHECKED
EVERY = 10                       # the live chart hears the training loss of every 10 batches (live.follow)
MOMENTS = (1, 2, 5, 10, 20)      # the epochs to rewind to after the demo, with its last epoch

# What the course measured, on one thread, with the course's code: the first N_TRAIN training photos and N_TEST test
# photos, normalized with the mean and the standard deviation of each channel of those training photos (MEAN, STD,
# as torch's std gives them); ColorCNN after torch.manual_seed(seed), the training photos shuffled after
# torch.manual_seed(seed + 1), in mini-batches of 64, with the Trainer of nn-trainer, a learning rate of 0.01, a warmup
# of one epoch and max_norm 1.0, for the seeds 0 to 4. The clothes are the same with SmallCNN of cnn-pytorch on the
# images of data/fashion-mnist-small.npz, divided by 255 and not normalized, as in cnn-pytorch. "train" and "test" are
# how many photos are read right after the last epoch, of 6,000 and of 1,000; "best" is the epoch with the lowest
# validation loss, "last_val" and "last_train" the validation and the training loss after the last epoch, rounded to 3
# decimals. "curves" is seed 0 of the color photos after 20 epochs, the loss after every epoch. "mixed" is how
# often two kinds were taken for each other in the test photos, both ways, summed over the 5 seeds after 20 epochs;
# "kinds" how many test photos of each kind were read right, summed the same way, and "per_kind" how many test photos
# each kind has. tests/test_cv_color_cnn.py measures seed 0 again when EAI_CIFAR_FILE names the unpacked cifar-10.npz.
MEAN = (0.4920, 0.4819, 0.4453)
STD = (0.2479, 0.2449, 0.2631)
REFERENCE = {
    ("clothes", 5): {"train": [5430, 5391, 5459, 5420, 5451], "test": [875, 864, 879, 877, 866]},
    ("clothes", 20): {"train": [5858, 5846, 5898, 5867, 5892], "test": [882, 877, 878, 875, 856], "best": [9, 8, 5, 9, 9]},
    ("photos", 5): {"train": [4214, 3972, 4170, 4129, 4209], "test": [561, 548, 559, 556, 553]},
    ("photos", 20): {"train": [5710, 5514, 5785, 5752, 5800], "test": [519, 515, 523, 543, 538], "best": [3, 4, 4, 6, 5],
                     "last_val": [2.369, 2.188, 2.349, 2.179, 2.386], "last_train": [0.212, 0.305, 0.183, 0.186, 0.164]},
}
CURVES = {
    "train": [1.913, 1.544, 1.342, 1.212, 1.085, 1.004, 0.914, 0.821, 0.73, 0.666, 0.582, 0.52, 0.447, 0.393, 0.341,
              0.293, 0.26, 0.237, 0.221, 0.212],
    "val": [1.699, 1.496, 1.38, 1.422, 1.434, 1.401, 1.513, 1.572, 1.595, 1.697, 1.789, 1.865, 2.035, 2.061, 2.198,
            2.279, 2.323, 2.356, 2.365, 2.369],
}
MIXED = [("bird", "deer", 167), ("airplane", "ship", 158), ("cat", "dog", 154), ("automobile", "truck", 126)]
KINDS_RIGHT = {"airplane": 282, "automobile": 300, "bird": 190, "cat": 179, "deer": 157, "dog": 166, "frog": 363,
               "horse": 302, "ship": 354, "truck": 345}
PER_KIND = {"airplane": 103, "automobile": 89, "bird": 100, "cat": 103, "deer": 90, "dog": 86, "frog": 112,
            "horse": 102, "ship": 106, "truck": 109}


def low_high(values) -> tuple:
    return min(values), max(values)


def grouped(n: int, lang: str) -> str:
    """A whole number with its thousands marked, as the language writes it: 11,642 or 11.642."""
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def percent(right: int, of: int) -> int:
    return round(100 * right / of)


# --- CIFAR-10 on the learner's computer ----------------------------------------------------------------------------------

def ready() -> bool:
    return cifar.unpacked()


def cifar_or_stop(button, texts: dict):
    """everything_ai.cifar for the lesson's cells, once CIFAR-10 is downloaded and unpacked. When it is not there
    yet, the cell stops with `texts["ask"]` and the button; a click downloads the archive, with the progress in the
    cell, and unpacks it. While it downloads, the lesson's other cells wait. `texts` has "ask", "busy" ({done} and
    {total} in MB), "unpacking", "no_net", "not_pinned", "damaged", "no_room" and "failed" ({why}): why it did not
    work; and "here"."""
    import marimo as mo
    failed = None
    if not ready() and button.value:
        try:
            if hub.pinned(*cifar.TAR) is None:
                total = round(hub.PINNED[cifar.TAR]["size"] / 1e6)
                last = [-1]

                def progress(done, size):
                    mb = int(done / 1e6)
                    if mb // 5 != last[0] // 5:   # every 5 MB
                        last[0] = mb
                        mo.output.replace(mo.md(texts["busy"].format(done=mb, total=total)))
                hub.download(*cifar.TAR, progress=progress)
            mo.output.replace(mo.md(texts["unpacking"]))
            cifar.unpack()
        except hub.NotPinned:
            failed = texts["not_pinned"]
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            failed = texts["no_net"]
        except cifar.Damaged:
            failed = texts["damaged"]
        except OSError as e:
            failed = texts["no_room"] if e.errno == errno.ENOSPC else texts["failed"].format(why=html.escape(str(e)))
    if not ready():
        mo.stop(True, download_card(texts["ask"], button, failed))
    mo.output.replace(mo.md(texts["here"]))
    return cifar


@functools.lru_cache(maxsize=1)
def photos():
    """The first N_TRAIN training photos and the first N_TEST test photos of CIFAR-10 as tensors of 0 to 1, of shape
    (N, 3, 32, 32), and their kinds."""
    X, y, X_test, y_test = cifar.load()
    as_floats = lambda a: torch.tensor(a, dtype=torch.float32) / 255
    return (as_floats(X[:N_TRAIN]), torch.tensor(y[:N_TRAIN], dtype=torch.int64),
            as_floats(X_test[:N_TEST]), torch.tensor(y_test[:N_TEST], dtype=torch.int64))


# --- the pictures --------------------------------------------------------------------------------------------------------

def gap_picture(labels: dict) -> str:
    """How many images SmallCNN reads right, of its training images and of the test images, for clothes and for color
    photos, after 5 and after 20 epochs: one row each, with a dot for every seed, filled for the training images and
    open for the test images, and a line between their means; the last row names both."""
    rows = [("clothes", 5), ("clothes", 20), ("photos", 5), ("photos", 20)]
    p = chart.Panel((40, 100), (-0.8, len(rows) - 0.25), title=labels["title"], xlabel=labels["x"], yticks=[],
                    xticks=[40, 50, 60, 70, 80, 90, 100])
    for k, (data, epochs) in enumerate(rows):
        y = len(rows) - 1 - k
        run = REFERENCE[(data, epochs)]
        train = [100 * v / N_TRAIN for v in run["train"]]
        test = [100 * v / N_TEST for v in run["test"]]
        middle = sum(test) / len(test), sum(train) / len(train)
        p.line(list(middle), [y, y], "ink-3", 1.6)
        p.points(train, [y] * len(train), "ink", r=3.5)
        p.points(test, [y] * len(test), "ink", r=3.5, hollow=True)
        p.text(41, y + 0.38, labels[data].format(epochs=epochs), "ink-2", "start")
        if k == len(rows) - 1:
            p.text(middle[0], y - 0.45, labels["test"], "ink-2")
            p.text(middle[1], y - 0.45, labels["train"], "ink-2")
    svg = chart.figure([p], alt=labels["alt"], width=420, height=300)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


def _photo_axes(ax, image, title: str, color=None):
    ax.imshow(np.transpose(image, (1, 2, 0)), interpolation="nearest")   # 32 by 32, as the network sees it
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if color:
        ax.set_title(title, fontsize=8, loc="center", color=color)
    else:
        ax.set_title(title, fontsize=8, loc="center")


def photos_picture(lang: str, n: int = 4) -> str:
    """The first n checked test photos of every kind, a column per kind, enlarged so that every pixel shows."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    _, _, X_test, _ = cifar.load()
    fig, axes = plt.subplots(n, len(cifar.KINDS), figsize=(6.8, 0.74 * n + 0.3))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.08, hspace=0.08)
    for j, kind in enumerate(cifar.KINDS):
        for i, place in enumerate(cifar.checked(kind, n)):
            _photo_axes(axes[i, j], X_test[place], NAMES[lang][kind] if i == 0 else "")
    alt = {"en": f"{n} photos of each of the 10 kinds of CIFAR-10, 32 by 32 pixels, enlarged so that every pixel shows: ",
           "nl": f"{n} foto's van elk van de 10 soorten van CIFAR-10, 32 bij 32 pixels, vergroot zodat elke pixel te zien is: "}
    out = plots.render(fig, alt=alt[lang] + ", ".join(NAMES[lang][k] for k in cifar.KINDS) + ".")
    plt.close(fig)
    return out


def guesses_picture(guesses: list, lang: str) -> str:
    """The demo's checked test photos, each with the kind the network guessed: in ink when it is right, in the color
    of mistakes and with the right kind under it when it is not."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    _, _, X_test, _ = cifar.load()
    cols = len(cifar.KINDS)
    rows = -(-len(guesses) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.8, 0.95 * rows + 0.2), squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.01, wspace=0.08, hspace=0.62)
    said = []
    for ax, (place, kind, guess) in zip(axes.T.flat, guesses):   # a column per kind
        right = guess == kind
        title = NAMES[lang][guess] if right else f"{NAMES[lang][guess]}\n({NAMES[lang][kind]})"
        _photo_axes(ax, X_test[place], title, None if right else plots.LOSS_TEXT)
        said.append(f"{NAMES[lang][kind]}: {NAMES[lang][guess]}")
    alt = {"en": "Photos of CIFAR-10, each with the kind your network guessed, and in brackets the right kind where it is wrong: ",
           "nl": "Foto's van CIFAR-10, elk met de soort die je netwerk gokte, en tussen haakjes de goede soort waar het fout is: "}
    out = plots.render(fig, alt=alt[lang] + "; ".join(said) + ".")
    plt.close(fig)
    return out


def curves_picture(curves: dict, labels: dict, lang: str, at: int | None = None) -> str:
    """The loss after every epoch of the demo's run: on the training photos solid and on the test photos dashed, as in
    torch-loss-curves, with a dot where the validation loss is lowest. With `at`, a line at that epoch, for rewinding."""
    train, val = curves.get("train", []), curves.get("val", [])
    every = [v for v in train + val if np.isfinite(v)] or [0.0, 1.0]
    p = chart.Panel((0, max(len(val), len(train), 1) + 0.5), (0, max(every) * 1.12), title=labels["title"],
                    xlabel=labels["x"], ylabel=labels["y"])
    for ys, dashed, name in ((train, False, labels["train"]), (val, True, labels["val"])):
        if ys:
            xs = list(range(1, len(ys) + 1))
            p.line(xs, ys, "loss", 2.2, dashed=dashed)
            under = ys[-1] - 0.1 * max(every)   # the training loss under its line, unless that is under the axis
            p.text(len(ys), ys[-1] + 0.07 * max(every) if dashed or under < 0.06 * max(every) else under, name, "loss", "end")
    if val:
        best = int(np.argmin(val))
        p.points([best + 1], [val[best]], "loss", r=4)
    if at is not None:
        p.line([at, at], [0, max(every) * 1.12], "ink-2", 1.2)
    svg = chart.figure([p], alt=labels["alt"], width=360, height=250)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def now_guesses(guesses: list, truth: list, lang: str) -> str:
    """The photos ONE_EACH, one checked test photo of each kind, each with the kind the network guessed at one moment
    of its run under it: in ink when it is right, in the color of mistakes and with the right kind under it when it is
    not (yours.guesses: quick to draw again while the network trains)."""
    _, _, X_test, _ = cifar.load()
    said = [NAMES[lang][cifar.KINDS[g]] for g in guesses]
    right = [NAMES[lang][cifar.KINDS[k]] for k in truth]
    alt = {"en": "One test photo of each kind, each with the kind your network guessed then, and in brackets the right kind where it is wrong: ",
           "nl": "Eén testfoto van elke soort, elk met de soort die je netwerk toen gokte, en tussen haakjes de goede soort waar het fout is: "}
    return yours.guesses([np.transpose(X_test[place], (1, 2, 0)) for place in ONE_EACH], said, right,
                         alt[lang] + "; ".join(f"{r}: {g}" for r, g in zip(right, said)) + ".")


def now_pictures(seen: dict, title: str, lang: str, uid: str) -> list:
    """What the learner's network makes at one moment of its run, as `look` or the rewind gives it: its guesses for the
    photos ONE_EACH, and under them its first layer on the car, under `title` with {name} the layer's name. `uid` keeps
    the masks of the layer apart from those of another picture on the page."""
    return netview.moment(now_guesses(seen["guesses"], seen["truth"], lang), seen.get("layer"), title, lang, uid)


def epoch_frame(result: dict, e: int, labels: dict, title: str, lang: str) -> str:
    """The rewind at epoch e (from 1): the loss after every epoch with a line at e, then the network's guesses for the
    photos ONE_EACH after that epoch, and under them its first layer on the car, as it was then (decision 1.17)."""
    layers = result.get("layers", [])
    seen = {"guesses": result["looked"][e - 1], "truth": result["truth"], "layer": layers[e - 1] if e <= len(layers) else None}
    return live.rewound(curves_picture(result["curves"], labels, lang, at=e), now_pictures(seen, title, lang, f"nv-color-epoch-{e}"))


def moments(n: int) -> list:
    """The epochs to rewind to in a run of n epochs: those of MOMENTS before its last, and its last."""
    return [m for m in MOMENTS if m < n] + ([n] if n else [])


# The lesson is done when these are correct. "color-weights" and "color-gap" are the choice questions in the lesson.
REQUIRED = ["color-weights", "channel-stats", "color-cnn", "color-gap"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _close(got, want) -> bool:
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    if isinstance(want, float):
        return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - want) <= 1e-4 * max(1.0, abs(want))
    return got == want


# --- exercise 2: channel_stats -------------------------------------------------------------------------------------------

_S_EN = "With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_S_NL = "Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"
_C_HEAD = "import torch\n\n\ndef channel_stats(X):\n"
_C_START = (_C_HEAD + "    # X: photos as floats from 0 to 1, of shape (N, 3, H, W)\n"
            "    # per color, the mean and the standard deviation over all photos,\n"
            "    # rows and columns: give back both, each a tensor of shape (3,)\n    ...\n")


def _c(body: str = "    return X.mean(dim=(0, 2, 3)), X.std(dim=(0, 2, 3))\n") -> str:
    return _C_HEAD + body


_C = _c()
_C_SCENARIOS = [
    # 50 made-up photos of 16 by 16 whose red runs from 0 to 1, their green to 0.6 and their blue to 0.3: the shapes,
    # and the first numbers, at most 6 of each, so that a wrong shape does not fill the message
    "import torch\n\ntorch.manual_seed(0)\nX = torch.rand(50, 3, 16, 16) * torch.tensor([1.0, 0.6, 0.3]).reshape(1, 3, 1, 1)\n"
    "mean, std = channel_stats(X)\n[list(mean.shape), list(std.shape), mean.reshape(-1)[:6].tolist(), std.reshape(-1)[:6].tolist()]",
    # 200 photos of one color each: red in half of them, blue of 0.5 in a quarter, green in none; the numbers must
    # come from all photos together
    "import torch\n\nX = torch.zeros(200, 3, 8, 8)\nX[:100, 0] = 1.0\nX[:50, 2] = 0.5\n"
    "mean, std = channel_stats(X)\n[list(mean.shape), mean.reshape(-1)[:6].tolist(), std.reshape(-1)[:6].tolist()]",
]
_C_WRONG = {
    "one-number": _c("    return X.mean(), X.std()\n"),
    "over-colors": _c("    return X.mean(dim=1), X.std(dim=1)\n"),
    "per-photo": _c("    return X.mean(dim=(2, 3)), X.std(dim=(2, 3))\n"),
    "per-place": _c("    return X.mean(dim=0), X.std(dim=0)\n"),
    "variance": _c("    return X.mean(dim=(0, 2, 3)), X.var(dim=(0, 2, 3))\n"),
    "photo-means": _c("    m = X.mean(dim=(2, 3))\n    return m.mean(dim=0), m.std(dim=0)\n"),
    "keepdim": _c("    return X.mean(dim=(0, 2, 3), keepdim=True), X.std(dim=(0, 2, 3), keepdim=True)\n"),
}

# --- exercise 3: ColorCNN ------------------------------------------------------------------------------------------------

_K_EN = "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_K_NL = "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"
_K_HEAD = "import torch\nfrom torch import nn\n\n\nclass ColorCNN(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
_K_FORWARD = ("\n    def forward(self, x):\n        x = self.pool(torch.relu(self.conv1(x)))\n"
              "        x = self.pool(torch.relu(self.conv2(x)))\n        x = x.reshape(len(x), -1)\n        return self.out(x)\n")


def _k(channels: str = "3", inputs: str = "16 * 8 * 8", comment: bool = False) -> str:
    note = "        # your SmallCNN, for color photos of 32 by 32: fill in the two ...\n" if comment else ""
    return (_K_HEAD + note + f"        self.conv1 = nn.Conv2d({channels}, 8, 3, padding=1)\n"
            "        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)\n        self.pool = nn.MaxPool2d(2)\n"
            f"        self.out = nn.Linear({inputs}, 10)\n" + _K_FORWARD)


_K = _k()
_K_START = _k("...", "...", comment=True)
_K_DATA = "import torch\n\ntorch.manual_seed(0)\nmodel = ColorCNN()\n"
_K_SCENARIOS = [
    # a mini-batch of 2 color photos: the shape of the logits and the first four of the first photo
    _K_DATA + "x = torch.rand(2, 3, 32, 32)\nlogits = model(x)\n[list(logits.shape), logits[0, :4].tolist()]",
    # a mini-batch of 3, and how many numbers its parameters hold
    _K_DATA + "x = torch.rand(3, 3, 32, 32)\nlogits = model(x)\n"
    "[list(logits.shape), logits[2, :4].tolist(), sum(p.numel() for p in model.parameters())]",
]


EXERCISES = {
    "channel-stats": ScenarioBox(
        name="channel-stats",
        start=_C_START,
        solution=_C,
        scenarios=_C_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "channels"),
        first=[_DOTS],
        mistakes=[
            Mistake("unpack", error=lambda e: isinstance(e, (ValueError, TypeError)) and re.search(r"values to unpack|cannot unpack", str(e)) is not None,
                    en="The check takes two things from your function, `mean, std = channel_stats(X)`, and your function gives back something else. Give back both, with a comma between them: `return mean, std`.",
                    nl="De check haalt twee dingen uit je functie, `mean, std = channel_stats(X)`, en je functie geeft iets anders terug. Geef ze allebei terug, met een komma ertussen: `return mean, std`."),
            Mistake("no-tensor", error=lambda e: isinstance(e, AttributeError) and "has no attribute 'shape'" in str(e),
                    en="The check asks your `mean` and `std` for their `.shape`, and they have none: your function gives back no tensors, but for instance lists or numbers. Give back the tensors that `.mean(...)` and `.std(...)` make, without `.tolist()` or `.item()`.",
                    nl="De check vraagt je `mean` en `std` om hun `.shape`, en die hebben ze niet: je functie geeft geen tensors terug, maar bijvoorbeeld lists of getallen. Geef de tensors terug die `.mean(...)` en `.std(...)` maken, zonder `.tolist()` of `.item()`."),
            Mistake("one-number", wrong=_C_WRONG["one-number"],
                    en=_S_EN + ": one mean and one standard deviation for all three colors together. Every color channel needs its own, so that red, green and blue each get mean 0 and standard deviation 1: keep the channel axis, and take the mean over the other three.",
                    nl=_S_NL + ": één gemiddelde en één standaardafwijking voor alle drie de kleuren samen. Elke kleur heeft er een eigen nodig, zodat rood, groen en blauw elk gemiddelde 0 en standaardafwijking 1 krijgen: houd de axis van de kleuren, en neem het gemiddelde over de andere drie."),
            Mistake("over-colors", wrong=_C_WRONG["over-colors"],
                    en=_S_EN + ": you take the mean over the colors, so you get one number per pixel, with red, green and blue mixed. It is the other way around: keep axis 1, the colors, and take the mean over the photos, the rows and the columns.",
                    nl=_S_NL + ": je neemt het gemiddelde over de kleuren, dus je krijgt één getal per pixel, met rood, groen en blauw door elkaar. Het is andersom: houd axis 1, de kleuren, en neem het gemiddelde over de foto's, de rijen en de kolommen."),
            Mistake("per-photo", wrong=_C_WRONG["per-photo"],
                    en=_S_EN + ": a mean per photo and per color, of shape (N, 3). The numbers must hold for every photo, the training photos and every new one: take axis 0, the photos, along as well.",
                    nl=_S_NL + ": een gemiddelde per foto en per kleur, met shape (N, 3). De getallen moeten voor elke foto gelden, de trainingsfoto's en elke nieuwe: neem axis 0, de foto's, ook mee."),
            Mistake("per-place", wrong=_C_WRONG["per-place"],
                    en=_S_EN + ": a mean for every place in the photo, of shape (3, H, W), over the photos only. Per color, one number is enough: take the rows and the columns, axes 2 and 3, along as well.",
                    nl=_S_NL + ": een gemiddelde voor elke plek in de foto, met shape (3, H, W), alleen over de foto's. Per kleur is één getal genoeg: neem de rijen en de kolommen, axis 2 en 3, ook mee."),
            Mistake("variance", wrong=_C_WRONG["variance"],
                    en=_S_EN + ": your means are right, and your second tensor is the variance, the square of the standard deviation. Use `.std(...)`, or the square root of the variance.",
                    nl=_S_NL + ": je gemiddelden kloppen, en je tweede tensor is de variantie, het kwadraat van de standaardafwijking. Gebruik `.std(...)`, of de wortel uit de variantie."),
            Mistake("photo-means", wrong=_C_WRONG["photo-means"],
                    en=_S_EN + ": your means are right, and your standard deviation is how much the mean color differs from photo to photo. Normalizing needs how much the pixels themselves spread: take the standard deviation over the photos, the rows and the columns at once.",
                    nl=_S_NL + ": je gemiddelden kloppen, en je standaardafwijking is hoeveel de gemiddelde kleur van foto tot foto verschilt. Om te normaliseren heb je de spreiding van de pixels zelf nodig: neem de standaardafwijking over de foto's, de rijen en de kolommen tegelijk."),
            Mistake("keepdim", wrong=_C_WRONG["keepdim"],
                    en=_S_EN + ": your numbers are right, with shape (1, 3, 1, 1): `keepdim=True` keeps the axes you take the mean over, each with length 1. Leave it out, and you get shape (3,).",
                    nl=_S_NL + ": je getallen kloppen, met shape (1, 3, 1, 1): `keepdim=True` houdt de axes waarover je het gemiddelde neemt, elk met lengte 1. Laat het weg, dan krijg je shape (3,)."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with a line that gives back the mean and the standard deviation of each color.",
                      "nl": "Vervang de `...` door een regel die het gemiddelde en de standaardafwijking van elke kleur teruggeeft."},
            "wrong": {"en": _S_EN + ": the shapes of `mean` and `std`, then their first numbers, at most 6 of each. Per color channel, one mean and one standard deviation, over all photos, rows and columns: which axes of (N, 3, H, W) are those?",
                      "nl": _S_NL + ": de shapes van `mean` en `std`, dan hun eerste getallen, hoogstens 6 van elk. Per kleur één gemiddelde en één standaardafwijking, over alle foto's, rijen en kolommen: welke axes van (N, 3, H, W) zijn dat?"},
            "almost": {"en": "Right for the first made-up photos. But " + _S_EN[0].lower() + _S_EN[1:] + ". Do your numbers come from all photos together, and not from some of them?",
                       "nl": "Goed voor de eerste verzonnen foto's. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Komen je getallen van alle foto's samen, en niet van een deel ervan?"},
            "correct": {"en": "Your `channel_stats` is right: one mean and one standard deviation per color, over all photos, rows and columns. It is now in your project, as `{saved}`.",
                        "nl": "Je `channel_stats` klopt: één gemiddelde en één standaardafwijking per kleur, over alle foto's, rijen en kolommen. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `channel_stats` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `channel_stats` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "color-cnn": ClassBox(
        name="color-cnn",
        picture=netview.network("ColorCNN", yours.scene_batch, chosen=0),
        start=_K_START,
        solution=_K,
        scenarios=_K_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "color_cnn"),
        first=[_DOTS],
        mistakes=[
            Mistake("channels", error=lambda e: isinstance(e, RuntimeError) and re.search(r"to have \d+ channels, but got \d+ channels", str(e)) is not None,
                    en="Your first convolution gets another number of channels than it was made for. A color photo has 3: red, green and blue. The first number of `nn.Conv2d` is the number of channels that come in.",
                    nl="Je eerste convolutie krijgt een ander aantal channels dan waarvoor hij gemaakt is. Een kleurenfoto heeft er 3: rood, groen en blauw. Het eerste getal van `nn.Conv2d` is het aantal channels dat binnenkomt."),
            Mistake("sizes", error=lambda e: isinstance(e, RuntimeError) and "cannot be multiplied" in str(e),
                    en="The sizes do not fit: your output head gets another number of inputs per photo than it was made for. With padding 1, each convolution keeps the height and width, and each pooling of 2 halves them: 32, 16, 8. How many numbers do 16 channels of 8 by 8 make?",
                    nl="De maten passen niet: je output head krijgt per foto een ander aantal inputs dan waarvoor hij gemaakt is. Met padding 1 behoudt elke convolutie de hoogte en breedte, en elke pooling van 2 halveert ze: 32, 16, 8. Hoeveel getallen maken 16 channels van 8 bij 8?"),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the channels that come into the first convolution, and the numbers per photo that come into the output head.",
                      "nl": "Vervang de twee `...`: de channels die de eerste convolutie ingaan, en de getallen per foto die de output head ingaan."},
            "wrong": {"en": _K_EN + ": the shape of the logits and the first four of one photo. Did you change more than the two `...`? The layers are made in the order of the course, so that they get the same random weights.",
                      "nl": _K_NL + ": de shape van de logits en de eerste vier van één foto. Heb je meer veranderd dan de twee `...`? De layers worden in de volgorde van de course gemaakt, zodat ze dezelfde willekeurige weights krijgen."},
            "almost": {"en": "Right for 2 photos. But " + _K_EN[0].lower() + _K_EN[1:] + ": the shape of the logits, the first four of the last photo, and the numbers in the parameters.",
                       "nl": "Goed voor 2 foto's. Maar " + _K_NL[0].lower() + _K_NL[1:] + ": de shape van de logits, de eerste vier van de laatste foto, en de getallen in de parameters."},
            "correct": {"en": "Your CNN takes color photos of 32 by 32 as input: 3 channels go into the first convolution, 1,024 numbers per photo into the output head. Your `ColorCNN` is now in your project, as `{saved}`.",
                        "nl": "Je CNN krijgt kleurenfoto's van 32 bij 32 als input: 3 channels gaan de eerste convolutie in, 1.024 getallen per foto de output head. Je `ColorCNN` staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `ColorCNN` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `ColorCNN` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("channel-stats", _C_START, "empty", None),
    ("channel-stats", _C, "correct", None),
    ("channel-stats", _c("    return X.mean(dim=(0, 2, 3)), X.std(dim=(0, 2, 3), unbiased=False)\n"), "correct", None),
    ("channel-stats", _c("    per = X.transpose(0, 1).reshape(3, -1)\n    return per.mean(dim=1), per.std(dim=1)\n"), "correct", None),
    ("channel-stats", _c("    return X.mean(axis=(0, 2, 3)), X.var(axis=(0, 2, 3)).sqrt()\n"), "correct", None),
    ("channel-stats", _c("    return X.mean(dim=(0, 2, 3))\n"), "error", "unpack"),
    ("channel-stats", _c("    return torch.stack([X.mean(dim=(0, 2, 3)), X.std(dim=(0, 2, 3)), X.amax(dim=(0, 2, 3))])\n"), "error", "unpack"),
    ("channel-stats", _c("    return X.mean(dim=(1, 2, 3)), X.std(dim=(1, 2, 3))\n"), "wrong", None),
    ("channel-stats", _c("    return X.mean(1), X.std(1)\n"), "wrong", "over-colors"),
    ("channel-stats", _c("    return X[:50].mean(dim=(0, 2, 3)), X[:50].std(dim=(0, 2, 3))\n"), "almost", None),
    ("channel-stats", _c("    return X.mean(dim=(0, 2, 3)), ...\n"), "wrong", "dots-left"),
    ("channel-stats", _c("    return X.mean(dim=(0, 2, 3)).tolist(), X.std(dim=(0, 2, 3)).tolist()\n"), "error", "no-tensor"),
    ("channel-stats", _c("    return X.mean().item(), X.std().item()\n"), "error", "no-tensor"),
] + [("channel-stats", code, "wrong", key) for key, code in _C_WRONG.items()] + [
    ("color-cnn", _K_START, "empty", None),
    ("color-cnn", _K, "correct", None),
    ("color-cnn", _k("3", "1024"), "correct", None),
    ("color-cnn", _k("3", "16 * 8 ** 2"), "correct", None),
    ("color-cnn", _k("1", "16 * 8 * 8"), "error", "channels"),
    ("color-cnn", _k("3 * 3", "16 * 8 * 8"), "error", "channels"),
    ("color-cnn", _k("3", "16 * 7 * 7"), "error", "sizes"),
    ("color-cnn", _k("3", "16 * 32 * 32"), "error", "sizes"),
    ("color-cnn", _k("3", "3 * 8 * 8"), "error", "sizes"),
    ("color-cnn", _k("3", "..."), "wrong", "dots-left"),
    ("color-cnn", _k().replace("nn.Linear(16 * 8 * 8, 10)\n", "nn.Linear(16 * 8 * 8, 10)\n        self.scale = nn.Parameter(torch.ones(10))\n"), "almost", None),
    ("color-cnn", _k().replace("        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)\n        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)\n",
                               "        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)\n        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)\n"), "wrong", None),
]


# --- the demo: the learner's ColorCNN on color photos ----------------------------------------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def _accuracy(model, X, y) -> int:
    model.eval()
    with torch.no_grad():
        return int(sum((model(X[i:i + 500]).argmax(dim=1) == y[i:i + 500]).sum() for i in range(0, len(X), 500)))


def _demo(show=None):
    """For the demo: None when color_cnn.py or channels.py is not in the project yet; "data" when CIFAR-10 is not
    unpacked; "stats" when the learner's channel_stats stops with an error or gives no 3 numbers twice; "cnn" when
    their ColorCNN does, on 2 photos; "trainer" when the Trainer or count_params does; "slow" when the training takes
    too long; else a dict with the mean and standard deviation of their channel_stats, the parameters, how many of the
    training and test photos it reads right, the loss after every epoch on both ("curves"), the epoch with the lowest
    validation loss, its guesses on the checked photos (place, kind, guess), and the modules the course's versions
    stood in for; and for the rewind, after every epoch, its guesses for the photos ONE_EACH ("looked") with their
    right kinds ("truth"), and its first layer on the car ("layers"). `show` hears the run while it trains
    (live.follow, every 10 batches), with what the network makes then (netview.hearing)."""
    if not (_saved("color_cnn") and _saved("channels")):
        return None
    if not ready():
        return "data"
    run = run_code("from cnn.color_cnn import ColorCNN")
    if run.error is not None:
        return "cnn"
    ColorCNN = run.names["ColorCNN"]
    run = run_code("from cnn.channels import channel_stats")
    if run.error is not None:
        return "stats"
    channel_stats = run.names["channel_stats"]
    run = run_code("from nn.trainer import Trainer\nfrom cnn.count import count_params")
    if run.error is not None:
        return "trainer"
    Trainer, count_params = run.names["Trainer"], run.names["count_params"]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    X, y, X_test, y_test = photos()
    try:
        mean, std = with_time_limit(lambda: channel_stats(X.clone()), 60)
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(3)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(3)
    except Exception:
        return "stats"
    if not (torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
        return "stats"
    X = (X - mean[:, None, None]) / std[:, None, None]
    X_test = (X_test - mean[:, None, None]) / std[:, None, None]
    try:
        torch.manual_seed(0)
        model = ColorCNN()
        with torch.no_grad():
            if tuple(model(X[:2]).shape) != (2, 10):
                return "cnn"
    except Exception:
        return "cnn"
    try:
        params = int(count_params(model))
    except Exception:
        return "trainer"

    truth, looked, layers = [int(y_test[place]) for place in ONE_EACH], [], []   # what the rewind keeps of every epoch
    heard = netview.hearing(model, X_test[list(ONE_EACH)], truth, X_test[PHOTO:PHOTO + 1], show, looked, layers)

    def train():
        torch.manual_seed(1)
        loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
        val = DataLoader(TensorDataset(X_test, y_test), batch_size=500)
        trainer = Trainer(model, loader, val, epochs=EPOCHS, lr=LR, warmup=len(loader), max_norm=1.0)
        live.follow(trainer, heard, "color", every=EVERY)
        return trainer.fit() or []

    try:
        history = with_time_limit(train, 300)
    except Timeout:
        return "slow"
    except Exception:
        return "trainer"
    curves = {"train": [float(h["train_loss"]) for h in history if isinstance(h, dict) and "train_loss" in h],
              "val": [float(h["val_loss"]) for h in history if isinstance(h, dict) and "val_loss" in h]}
    places = [place for kind in cifar.KINDS for place in cifar.checked(kind, SHOWN)]
    model.eval()
    with torch.no_grad():
        said = model(X_test[places]).argmax(dim=1).tolist()
    guesses = [(place, cifar.KINDS[int(y_test[place])], cifar.KINDS[g]) for place, g in zip(places, said)]
    return {"mean": mean.tolist(), "std": std.tolist(), "params": params, "train": _accuracy(model, X, y),
            "test": _accuracy(model, X_test, y_test), "curves": curves,
            "best": int(np.argmin(curves["val"])) + 1 if curves["val"] else None, "guesses": guesses,
            "looked": looked, "truth": truth, "layers": layers, "stand_ins": stand_ins}


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file), and whether CIFAR-10
    is there."""
    texts = []
    for package, module in (("cnn", "color_cnn"), ("cnn", "channels"), ("cnn", "count"), ("nn", "train_epoch"), ("nn", "trainer")):
        path = projects.folder() / package / f"{module}.py"
        texts.append(path.read_text(encoding="utf-8") if path.exists() else None)
    return tuple(texts) + (ready(),)


def demo(show=None):
    """The demo's result, worked out once for the project files as they are; `show` hears the run when it trains."""
    key = _project_texts()
    if key not in _DEMO_RUNS:
        _DEMO_RUNS.clear()
        _DEMO_RUNS[key] = _demo(show)
    return _DEMO_RUNS[key]
