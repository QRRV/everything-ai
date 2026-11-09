# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-1x1-convolution: a 1×1 convolution mixes the channels of each pixel, and a bottleneck puts
the expensive 3×3 convolution between a 1×1 convolution to fewer channels and one back. The learner's Bottleneck is
saved in their project (work/projects/cnn/bottleneck.py) when right, and checked on made-up numbers, so it works
without CIFAR-10. The pictures mix the colors of the course's landscape of data-images-as-arrays, and count the
parameters of a bottleneck for every width of its middle. The demo trains a CNN with a 3×3 convolution from 64 to 64
channels in its middle, or with the learner's Bottleneck there, with their Trainer on the first 6,000 training photos
of CIFAR-10, and tests it on the first 1,000 test photos; after every epoch it keeps the network's guess for one
checked test photo of each kind, for rewinding."""
import importlib.util
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import chart, live, paths, projects
from everything_ai.check import Mistake, ScenarioBox, Sort, Timeout, run_code, with_time_limit


def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_1x1",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_AUG = _lesson_checks("cv-augment-photos")   # CIFAR-10 and its button, the photos, the rewind's pictures

# The demo uses the learner's channel_stats of cv-color-cnn, their Trainer of nn-trainer and their count_params of
# cnn-pytorch; where one is missing or broken, the course's version stands in (template 1.3).
BUILDS_ON = (("nn", "train_epoch", "nn-trainer", "train-epoch"), ("nn", "trainer", "nn-trainer", "trainer"),
             ("cnn", "count", "cnn-pytorch", "count-params"), ("cnn", "channels", "cv-color-cnn", "channel-stats"))
projects.builds_on(*BUILDS_ON)

DOWNLOAD_LINE = {
    "en": "Works on every computer. CIFAR-10, 170 MB of photos, is downloaded once from the University of Toronto when you click its button in the lesson, unless you did so in Project: your CNN on color photos; after that, the lesson works without a connection. The exercises work without it too.",
    "nl": "Werkt op elke computer. CIFAR-10, 170 MB aan foto's, wordt één keer gedownload van de University of Toronto als je in de les op de knop ervoor klikt, tenzij je dat al deed in Project: je CNN op kleurenfoto's; daarna werkt de les zonder verbinding. De oefeningen werken ook zonder.",
}

cifar_or_stop = _AUG.cifar_or_stop
ready = _AUG.ready
photos = _AUG.photos                   # the first N_TRAIN training photos and N_TEST test photos, from 0 to 1
grouped, number, percent, low_high = _AUG.grouped, _AUG.number, _AUG.percent, _AUG.low_high
guesses_picture, moments, best_epoch = _AUG.guesses_picture, _AUG.moments, _AUG.best_epoch
SHOWN = _AUG.SHOWN                     # one checked test photo of each kind, for the rewind
N_TRAIN, N_TEST = _AUG.N_TRAIN, _AUG.N_TEST
EPOCHS, LR = 20, 0.01                  # the demo: as in cv-color-cnn
CHANNELS = 64                          # the channels around the block in the middle of the demo's network
MIDDLE = {"plain": None, "b32": 32, "b16": 16}   # the block: a 3×3 convolution, or a bottleneck with this many in the middle


# --- counting -------------------------------------------------------------------------------------------------------

def conv_params(c_in: int, c_out: int, k: int = 1) -> int:
    """The parameters of nn.Conv2d(c_in, c_out, k): a filter of c_in × k × k weights and a bias for each output channel."""
    return c_out * (c_in * k * k + 1)


def conv_mults(c_in: int, c_out: int, k: int = 1) -> int:
    """The multiplications of nn.Conv2d(c_in, c_out, k) for one pixel of what it gives."""
    return c_out * c_in * k * k


def bottleneck_params(c: int, m: int, k: int = 3) -> int:
    """A 1×1 convolution from c to m channels, a k×k one from m to m, and a 1×1 one from m back to c."""
    return conv_params(c, m) + conv_params(m, m, k) + conv_params(m, c)


def bottleneck_mults(c: int, m: int, k: int = 3) -> int:
    return conv_mults(c, m) + conv_mults(m, m, k) + conv_mults(m, c)


# ResNet-50's bottleneck (He et al., 2015, section 4.1 and figure 5): 256 channels, 64 in the middle; counted with
# biases, as nn.Conv2d makes them (ResNet itself leaves them out, before batch normalization). Against it, one 3×3
# convolution from 256 to 256, and the block of two 3×3 convolutions of 64 channels of the smaller ResNets.
RESNET = {"plain": conv_params(256, 256, 3), "bottleneck": bottleneck_params(256, 64),             # 590,080 and 70,016
          "plain_mults": conv_mults(256, 256, 3), "bottleneck_mults": bottleneck_mults(256, 64),   # 589,824 and 69,632
          "basic_mults": 2 * conv_mults(64, 64, 3)}                                                 # 73,728
PLAIN_BLOCK = conv_params(CHANNELS, CHANNELS, 3)   # 36,928: the 3×3 convolution in the middle of the demo's network


def share(n: int, of: int) -> float:
    """n as a percentage of `of`, with one decimal."""
    return round(100 * n / of, 1)


def first_wider() -> int:
    """The fewest channels in the middle for which a bottleneck around CHANNELS has more parameters than PLAIN_BLOCK."""
    return next(m for m in range(1, CHANNELS + 1) if bottleneck_params(CHANNELS, m) > PLAIN_BLOCK)


# --- mixing the colors of the landscape -----------------------------------------------------------------------------

LANDSCAPE = "lessons/data-images-as-arrays/landscape.png"   # 24 by 40 pixels, made by the course
SUN = (5, 32)                       # a pixel in the sun: row 5, column 32
GRAY = (0.299, 0.587, 0.114)        # the weights of a gray value: how bright each color looks to us (ITU-R BT.601)
RED_BLUE = (1.0, 0.0, -1.0)         # red minus blue: high where there is more red than blue


def landscape():
    """The course's landscape as a tensor of shape (1, 3, 24, 40), from 0 to 1."""
    import matplotlib.pyplot as plt
    img = plt.imread(str(paths.ROOT / LANDSCAPE))
    return torch.tensor(img[:, :, :3]).permute(2, 0, 1).unsqueeze(0)


def mixed(weights, bias: float = 0.0):
    """What a 1×1 convolution from 3 channels to 1 with these weights makes of the landscape: shape (24, 40)."""
    mix = nn.Conv2d(3, 1, 1)
    with torch.no_grad():
        mix.weight[0, :, 0, 0] = torch.tensor(weights)
        mix.bias[0] = bias
        return mix(landscape())[0, 0]


def mix_picture(labels: dict) -> str:
    """The landscape, and what two 1×1 convolutions from 3 channels to 1 make of it, each drawn with its lowest number
    black and its highest white, as imshow draws one channel. `labels` gives "photo", "gray", "red_blue" and "alt"."""
    import matplotlib.pyplot as plt
    from everything_ai import plots
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 1.75))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.8, bottom=0.02, wspace=0.08)
    images = ((landscape()[0].permute(1, 2, 0).numpy(), labels["photo"], None),
              (mixed(GRAY).numpy(), labels["gray"], "gray"), (mixed(RED_BLUE).numpy(), labels["red_blue"], "gray"))
    for ax, (image, title, cmap) in zip(axes, images):
        ax.imshow(image, cmap=cmap, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(title, fontsize=9, loc="center")
    out = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return out


# --- the slider: how wide the middle of a bottleneck is -------------------------------------------------------------

def bottleneck_sentence(m: int, lang: str) -> str:
    """The sentence under the slider, with m channels in the middle of a bottleneck around CHANNELS channels."""
    parts = [grouped(n, lang) for n in (conv_params(CHANNELS, m), conv_params(m, m, 3), conv_params(m, CHANNELS))]
    total = bottleneck_params(CHANNELS, m)
    s, t, p = grouped(total, lang), grouped(PLAIN_BLOCK, lang), number(share(total, PLAIN_BLOCK), 1, lang)
    if lang == "nl":
        return (f"Met {m} in het midden: {parts[0]} parameters in de 1×1 convolution van {CHANNELS} naar {m}, {parts[1]} in de 3×3 van {m} naar {m}, "
                f"en {parts[2]} in de 1×1 van {m} terug naar {CHANNELS}. Samen {s}, {p}% van de {t} van één 3×3-convolutie van {CHANNELS} naar {CHANNELS}.")
    return (f"With {m} in the middle: {parts[0]} parameters in the 1×1 convolution from {CHANNELS} to {m}, {parts[1]} in the 3×3 from {m} to {m}, "
            f"and {parts[2]} in the 1×1 from {m} back to {CHANNELS}. Together {s}, {p}% of the {t} of one 3×3 convolution from {CHANNELS} to {CHANNELS}.")


RIGHT = 58   # the right end of the slider's picture, in thousands: room for the total after the longest bar, 45,248


def bottleneck_picture(m: int, labels: dict, lang: str) -> str:
    """Two bars of parameters: one 3×3 convolution from CHANNELS to CHANNELS, and a bottleneck with m in the middle,
    its three convolutions one after the other; a dashed line where the first bar ends. `labels` gives "title", "x",
    "plain", "bottleneck" and "alt"."""
    top = bottleneck_params(CHANNELS, CHANNELS) / 1000
    p = chart.Panel((-17, RIGHT), (-0.7, 1.7), title=labels["title"], xlabel=labels["x"], yticks=[],
                    xticks=list(range(0, int(top) + 1, 10)))
    plain = PLAIN_BLOCK / 1000
    p.rect(0, 0.75, plain, 1.25, "ink-3")
    p.text(-16.5, 0.96, labels["plain"], "ink-2", "start")
    p.text(plain + 0.8, 0.96, grouped(PLAIN_BLOCK, lang), "ink-2", "start")
    left = 0.0
    for n, role in ((conv_params(CHANNELS, m), "param"), (conv_params(m, m, 3), "param"), (conv_params(m, CHANNELS), "param")):
        p.rect(left, -0.25, left + n / 1000, 0.25, role)
        left += n / 1000
    p.text(-16.5, -0.04, labels["bottleneck"], "ink", "start")
    p.text(left + 0.8, -0.04, grouped(bottleneck_params(CHANNELS, m), lang), "ink", "start")
    p.line([plain, plain], [-0.55, 1.45], "ink-2", 1.2, dashed=True)
    svg = chart.figure([p], alt=labels["alt"], width=420, height=170)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


# --- the networks of the demo ---------------------------------------------------------------------------------------

class Plain(nn.Module):
    """The block in the middle without a bottleneck: one 3×3 convolution from `channels` to `channels`, with a ReLU."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv(x))


class WideCNN(nn.Module):
    """ColorCNN of cv-color-cnn made wider: 32 and 64 filters of 3 by 3, each with max pooling of 2 by 2, then `block`
    on 64 maps of 8 by 8, pooling to 4 by 4, and one linear layer to the 10 logits."""

    def __init__(self, block):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, CHANNELS, 3, padding=1)
        self.block = block
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(CHANNELS * 4 * 4, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(self.block(x))
        return self.fc(x.reshape(len(x), -1))


AROUND = conv_params(3, 32, 3) + conv_params(32, CHANNELS, 3) + CHANNELS * 4 * 4 * 10 + 10   # 29,642: all but the block
SIDE = 8   # the rows and columns of the maps the block gets


def network_params(variant: str) -> int:
    m = MIDDLE[variant]
    return AROUND + (PLAIN_BLOCK if m is None else bottleneck_params(CHANNELS, m))


# --- what the course measured ---------------------------------------------------------------------------------------

# What the course measured, on one thread, with the course's code, through run(): the block of `variant` (MIDDLE) made
# after torch.manual_seed(seed), then the rest of WideCNN, trained with the Trainer of nn-trainer (AdamW, a learning
# rate of LR, a warmup of one epoch, cosine decay, max_norm 1.0, mini-batches of 64) on the first N_TRAIN training
# photos, shuffled after torch.manual_seed(seed + 1), for EPOCHS epochs and the seeds 0 to 4. The photos are normalized
# with channel_stats of the training photos. "train" and "test" are how many of the training photos and of the first
# N_TEST test photos are read right after the last epoch; "best" is the first epoch with the lowest validation loss,
# rounded to 3 decimals, and "lowest" that loss; "last_val" and "last_train" are the validation and the training loss
# of the last epoch. CURVES are seed 0, the validation loss after every epoch, and the training loss of "b16".
# tests/test_cv_1x1_convolution.py measures seed 0 of "b16" again when EAI_CIFAR_FILE names the unpacked cifar-10.npz.
REFERENCE = {
    "plain": {"train": [6000, 6000, 5999, 5999, 6000], "test": [539, 560, 544, 532, 572], "best": [6, 5, 4, 5, 5],
              "lowest": [1.398, 1.416, 1.371, 1.41, 1.402], "last_val": [3.055, 3.115, 3.007, 2.814, 2.998],
              "last_train": [0.006, 0.007, 0.01, 0.018, 0.007]},
    "b32": {"train": [5445, 5673, 5333, 5604, 5559], "test": [547, 553, 535, 527, 535], "best": [8, 8, 8, 9, 7],
            "lowest": [1.32, 1.284, 1.359, 1.297, 1.403], "last_val": [1.727, 1.857, 1.754, 1.807, 1.952],
            "last_train": [0.339, 0.23, 0.377, 0.264, 0.275]},
    "b16": {"train": [5310, 5077, 5175, 5507, 5510], "test": [537, 535, 550, 544, 521], "best": [9, 10, 12, 10, 7],
            "lowest": [1.305, 1.309, 1.333, 1.303, 1.351], "last_val": [1.641, 1.557, 1.639, 1.718, 1.864],
            "last_train": [0.387, 0.493, 0.441, 0.311, 0.298]},
}
CURVES = {
    "plain_val": [1.691, 1.584, 1.552, 1.446, 1.399, 1.398, 1.434, 1.565, 1.643, 1.798, 1.941, 2.125, 2.393, 2.553, 2.819,
                  2.935, 3.003, 3.04, 3.053, 3.055],
    "b16_val": [1.804, 1.712, 1.521, 1.43, 1.401, 1.398, 1.328, 1.309, 1.305, 1.35, 1.373, 1.375, 1.359, 1.432, 1.483, 1.565,
                1.584, 1.621, 1.637, 1.641],
    "b16_train": [2.013, 1.678, 1.567, 1.445, 1.388, 1.287, 1.214, 1.144, 1.077, 1.003, 0.931, 0.848, 0.762, 0.68, 0.597,
                  0.532, 0.474, 0.429, 0.4, 0.387],
}


# --- the pictures of the runs ----------------------------------------------------------------------------------------

ROWS = ("plain", "b32", "b16")


def runs_picture(labels: dict, lang: str, mine: dict | None = None) -> str:
    """How many photos the network reads right after EPOCHS epochs, of its training photos and of the test photos, for
    each block in the middle, with a dot for every seed, filled for the training photos and open for the test photos,
    and a line between their means. With `mine` ({"variant", "train", "test"}), the learner's run as a row of its own
    at the top. `labels` gives "title", "x", a name for each row, "mine", "train", "test" and "alt"."""
    rows = [(key, REFERENCE[key], labels[key]) for key in ROWS]
    if mine is not None:
        rows.insert(0, ("mine", {"train": [mine["train"]], "test": [mine["test"]]}, labels["mine"]))
    p = chart.Panel((40, 100), (-0.8, len(rows) - 0.25), title=labels["title"], xlabel=labels["x"], yticks=[],
                    xticks=[40, 50, 60, 70, 80, 90, 100])
    for k, (key, run, name) in enumerate(rows):
        y = len(rows) - 1 - k
        train = [100 * v / N_TRAIN for v in run["train"]]
        test = [100 * v / N_TEST for v in run["test"]]
        middle = sum(test) / len(test), sum(train) / len(train)
        p.line(list(middle), [y, y], "ink-3", 1.6)
        p.points(train, [y] * len(train), "ink", r=3.5)
        p.points(test, [y] * len(test), "ink", r=3.5, hollow=True)
        p.text(41, y + 0.38, name, "ink-2", "start")
        if k == len(rows) - 1:   # under the last row, each label away from the other, toward its own end of the line
            p.text(middle[0], y - 0.45, labels["test"], "ink-3", "end")
            p.text(middle[1], y - 0.45, labels["train"], "ink-3", "start")
    svg = chart.figure([p], alt=labels["alt"], width=420, height=60 * len(rows) + 60)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


TOP = 3.4   # the top of the curves picture: above the end of CURVES["plain_val"]


def curves_picture(curves: dict, labels: dict, lang: str, at: int | None = None, other: str = "plain") -> str:
    """The loss after every epoch of the demo's run: on the training photos solid and on the test photos dashed, with
    a dot where the validation loss is lowest; in grey the validation loss of the course's run `other` (seed 0). With
    `at`, a line at that epoch, for rewinding. `labels` gives "title", "x", "y", "train", "val", the name of `other`
    and "alt"."""
    train, val = curves.get("train", []), curves.get("val", [])
    mine = [v for v in train + val if np.isfinite(v)]
    top = max([TOP] + [1.1 * v for v in mine])
    before = CURVES[f"{other}_val"]
    p = chart.Panel((0, max(len(val), len(train), len(before), 1) + 0.5), (0, top), title=labels["title"],
                    xlabel=labels["x"], ylabel=labels["y"])
    p.line(list(range(1, len(before) + 1)), before, "ink-3", 1.6, dashed=True)
    p.text(len(before), before[-1] + 0.06 * top, labels[other], "ink-3", "end")
    train_above = bool(train and val and train[-1] >= val[-1])
    for ys, dashed, name, above in ((train, False, labels["train"], train_above), (val, True, labels["val"], not train_above)):
        if ys:
            xs = list(range(1, len(ys) + 1))
            p.line(xs, ys, "loss", 2.2, dashed=dashed)
            below = ys[-1] - 0.08 * top
            p.text(len(ys), ys[-1] + 0.07 * top if above or below < 0.04 * top else below, name, "loss", "end")
    if val:
        best = best_epoch(val)
        p.points([best], [val[best - 1]], "loss", r=4)
    if at is not None:
        p.line([at, at], [0, top], "ink-2", 1.2)
    svg = chart.figure([p], alt=labels["alt"], width=380, height=260)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


# The lesson is done when these are correct. "bottleneck-runs" is the choice question in the lesson.
REQUIRED = ["one-by-one", "bottleneck", "bottleneck-runs"]

# --- exercise 1: what a 1×1 convolution can do ------------------------------------------------------------------------

_SORT = Sort(
    name="one-by-one",
    items={
        "gray": {"en": "**Gray**: make a color photo gray, with 0.299 of its red, 0.587 of its green and 0.114 of its blue.",
                 "nl": "**Grijs**: een kleurenfoto grijs maken, met 0,299 van zijn rood, 0,587 van zijn groen en 0,114 van zijn blauw."},
        "edge": {"en": "**Edges**: find the places where a photo gets brighter from left to right.",
                 "nl": "**Randen**: de plekken vinden waar een foto van links naar rechts lichter wordt."},
        "fewer": {"en": "**Fewer channels**: turn 64 channels into 16.", "nl": "**Minder channels**: van 64 channels er 16 maken."},
        "blur": {"en": "**Blur**: make each pixel the mean of itself and its 8 neighbors.",
                 "nl": "**Vervagen**: elke pixel het gemiddelde maken van zichzelf en zijn 8 buren."},
        "swap": {"en": "**Swap colors**: make the red channel blue and the blue channel red.",
                 "nl": "**Kleuren wisselen**: van het rode channel het blauwe maken, en van het blauwe het rode."},
        "shift": {"en": "**Shift**: move a photo one pixel to the right.", "nl": "**Verschuiven**: een foto één pixel naar rechts schuiven."},
    },
    groups={"can": {"en": "a 1×1 convolution can do it", "nl": "een 1×1 convolution kan het"},
            "larger": {"en": "it needs a larger filter", "nl": "daar is een groter filter voor nodig"}},
    solution={"gray": "can", "edge": "larger", "fewer": "can", "blur": "larger", "swap": "can", "shift": "larger"},
    nudge={
        "gray": {"en": "What does the gray value of a pixel need: the colors of that pixel, or of its neighbors too?",
                 "nl": "Wat heeft de grijswaarde van een pixel nodig: de kleuren van die pixel, of ook die van zijn buren?"},
        "edge": {"en": "Can one pixel on its own tell whether its neighbor to the right is brighter?",
                 "nl": "Kan één pixel in zijn eentje zeggen of zijn buurman rechts lichter is?"},
        "fewer": {"en": "How many filters does a convolution need to give 16 channels, and how large must they be?",
                  "nl": "Hoeveel filters heeft een convolutie nodig om 16 channels te geven, en hoe groot moeten die zijn?"},
        "blur": {"en": "Which pixels does the new number of a pixel come from?",
                 "nl": "Uit welke pixels komt het nieuwe getal van een pixel?"},
        "swap": {"en": "Does the new red of a pixel need anything but the blue of that same pixel?",
                 "nl": "Heeft het nieuwe rood van een pixel iets anders nodig dan het blauw van diezelfde pixel?"},
        "shift": {"en": "Where does the new number at a place come from: that place, or the place to its left?",
                  "nl": "Waar komt het nieuwe getal op een plek vandaan: van die plek, of van de plek links ervan?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the six things again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de zes dingen opnieuw uit."},
        "empty": {"en": "Put each one in a group: a 1×1 convolution can do it, or it needs a larger filter.",
                  "nl": "Zet elk ding in een groep: een 1×1 convolution kan het, of daar is een groter filter voor nodig."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "{item} belongs in the other group.", "nl": "{item} hoort in de andere groep."},
        "correct": {"en": "All {n} are right. A 1×1 convolution makes the new numbers of a pixel from the old numbers of that same pixel: a gray value, fewer channels and swapped colors all are that, with the right weights (for the swap: 1 where the new red takes the old blue, and the other way around). An edge, a blur and a shift need the neighbors of a pixel, so a filter larger than 1 by 1: two pixels side by side for the edge and the shift, 3 by 3 for the blur.",
                    "nl": "Alle {n} kloppen. Een 1×1 convolution maakt de nieuwe getallen van een pixel uit de oude getallen van diezelfde pixel: een grijswaarde, minder channels en verwisselde kleuren zijn dat allemaal, met de goede weights (voor het wisselen: 1 waar het nieuwe rood het oude blauw neemt, en andersom). Een rand, vervagen en verschuiven hebben de buren van een pixel nodig, dus een filter groter dan 1 bij 1: twee pixels naast elkaar voor de rand en het verschuiven, 3 bij 3 voor het vervagen."},
    },
)

# --- exercise 2: a bottleneck ---------------------------------------------------------------------------------------

_B_HEAD = ("import torch\nfrom torch import nn\n\n\nclass Bottleneck(nn.Module):\n    def __init__(self, channels, middle):\n"
           "        super().__init__()\n")
_B_CONV = "        self.conv = nn.Conv2d(middle, middle, 3, padding=1)\n"
_B_FORWARD = "\n    def forward(self, x):\n"
_B_BODY = "        x = torch.relu(self.reduce(x))\n        x = torch.relu(self.conv(x))\n        return torch.relu(self.expand(x))\n"


def _b(reduce: str = "nn.Conv2d(channels, middle, 1)", expand: str = "nn.Conv2d(middle, channels, 1)", body: str = _B_BODY) -> str:
    return _B_HEAD + f"        self.reduce = {reduce}\n" + _B_CONV + f"        self.expand = {expand}\n" + _B_FORWARD + body


_B = _b()
_B_START = (_B_HEAD + "        # reduce: a 1×1 convolution from channels to middle\n        self.reduce = ...\n" + _B_CONV
            + "        # expand: a 1×1 convolution from middle back to channels\n        self.expand = ...\n" + _B_FORWARD
            + "        # reduce, conv and expand, each with a ReLU after it\n        ...\n")
# The scenarios give reduce, conv and expand their weights by name, in that order, so that the order in which
# __init__ makes them does not change the numbers.
_B_WEIGHTS = ("with torch.no_grad():\n    for layer in (block.reduce, block.conv, block.expand):\n        for q in layer.parameters():\n"
              "            q.copy_(torch.randn_like(q) / 8)\n")
_B_SCENARIOS = [
    # 64 channels with 16 in the middle, as in the demo, on two made-up maps of 8 by 8: the shape, four numbers of
    # the first pixel, and the number of parameters
    "import torch\n\nblock = Bottleneck(64, 16)\ntorch.manual_seed(0)\n" + _B_WEIGHTS + "x = torch.rand(2, 64, 8, 8)\nout = block(x)\n"
    "[list(out.shape), out[0, :4, 0, 0].tolist(), sum(q.numel() for q in block.parameters())]",
    # another size: the shapes of the weights of reduce and expand, and whether no number below 0 comes out
    "import torch\n\ntorch.manual_seed(0)\nblock = Bottleneck(8, 2)\nout = block(torch.randn(1, 8, 5, 5))\n"
    "[list(block.reduce.weight.shape), list(block.expand.weight.shape), list(out.shape), bool((out >= 0).all())]",
    # the numbers of one pixel in the middle of the maps
    "import torch\n\nblock = Bottleneck(8, 2)\ntorch.manual_seed(1)\n" + _B_WEIGHTS + "out = block(torch.randn(1, 8, 5, 5))\nout[0, :, 2, 2].tolist()",
]
_B_WRONG = {
    "kernel-3": _b(reduce="nn.Conv2d(channels, middle, 3, padding=1)", expand="nn.Conv2d(middle, channels, 3, padding=1)"),
    "no-relu": _b(body="        x = self.reduce(x)\n        x = self.conv(x)\n        return self.expand(x)\n"),
    "last-relu": _b(body="        x = torch.relu(self.reduce(x))\n        x = torch.relu(self.conv(x))\n        return self.expand(x)\n"),
    "no-expand": _b(body="        x = torch.relu(self.reduce(x))\n        return torch.relu(self.conv(x))\n"),
    # the input added, as in ResNet-50's bottleneck block
    "added": _b(body="        y = torch.relu(self.reduce(x))\n        y = torch.relu(self.conv(y))\n        return torch.relu(self.expand(y) + x)\n"),
    "no-bias": _b(reduce="nn.Conv2d(channels, middle, 1, bias=False)", expand="nn.Conv2d(middle, channels, 1, bias=False)"),
    "padding": _b(reduce="nn.Conv2d(channels, middle, 1, padding=1)", expand="nn.Conv2d(middle, channels, 1, padding=1)"),
}
_KERNEL_3 = conv_params(64, 16, 3) + conv_params(16, 16, 3) + conv_params(16, 64, 3)   # 20,832: reduce and expand of 3 by 3
# `...` on a line of its own, or after `=` or `return`
# a number in place of channels or middle, in reduce or expand: right for 64 and 16 only
_FIXED = Mistake("fixed", matches=lambda got, text: re.search(r"self\.(reduce|expand)\s*=\s*nn\.Conv2d\(\s*(\d|[A-Za-z_]\w*\s*,\s*\d+\s*,)", text) is not None,
                 en="`reduce` or `expand` has a fixed number of channels. The block must work for every number of channels and every middle: use the names `channels` and `middle`, as `conv` does with `middle`.",
                 nl="`reduce` of `expand` heeft een vast aantal channels. Het block moet werken voor elk aantal channels en elk midden: gebruik de namen `channels` en `middle`, zoals `conv` dat doet met `middle`.")
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$|=\s*\.\.\.|\breturn\s+\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace each with your own code: `reduce`, `expand`, and the body of `forward`.",
                nl="Er staat nog een `...` in je code. Vervang elk door je eigen code: `reduce`, `expand`, en wat `forward` doet.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_S_EN = "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_S_NL = "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"


def _close(got, want) -> bool:
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    if isinstance(want, bool):
        return got is want or (isinstance(got, (bool, np.bool_)) and bool(got) == want)
    if isinstance(want, float):
        return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - want) <= 1e-4 * max(1.0, abs(want))
    return got == want


EXERCISES = {
    "one-by-one": _SORT,
    "bottleneck": ScenarioBox(
        name="bottleneck",
        start=_B_START,
        solution=_B,
        scenarios=_B_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "bottleneck"),
        first=[_DOTS, _FIXED],
        mistakes=[
            Mistake("channels", error=lambda e: isinstance(e, RuntimeError) and "channels" in str(e) and "expected" in str(e),
                    en="A convolution gets another number of channels than it was made for. `nn.Conv2d(a, b, 1)` takes `a` channels and gives `b`: `reduce` goes from `channels` to `middle`, and `expand` from `middle` back to `channels`. Does `forward` use them in the order `reduce`, `conv`, `expand`?",
                    nl="Een convolutie krijgt een ander aantal channels dan waarvoor hij gemaakt is. `nn.Conv2d(a, b, 1)` neemt `a` channels en geeft er `b`: `reduce` gaat van `channels` naar `middle`, en `expand` van `middle` terug naar `channels`. Gebruikt `forward` ze in de volgorde `reduce`, `conv`, `expand`?"),
            Mistake("add-sizes", error=lambda e: isinstance(e, RuntimeError) and "must match the size of tensor" in str(e),
                    en="Your block adds two things whose channels do not match. The block stands in for one 3×3 convolution, without the addition of ResNet-50's bottleneck block: give back what `expand` and its ReLU make.",
                    nl="Je block telt twee dingen op waarvan de channels niet passen. Het block staat op de plek van één 3×3-convolutie, zonder de optelling van het bottleneck block van ResNet-50: geef terug wat `expand` en zijn ReLU maken."),
            Mistake("no-return", error=lambda e: isinstance(e, AttributeError) and "'NoneType' object has no attribute" in str(e),
                    en="The check asks for the shape of what your block gives, and `forward` gives back nothing, `None`: end it with `return`, and what `expand` and its ReLU make.",
                    nl="De check vraagt de shape van wat je block geeft, en `forward` geeft niets terug, `None`: eindig hem met `return`, en wat `expand` en zijn ReLU maken."),
            Mistake("kernel-3", wrong=_B_WRONG["kernel-3"],
                    en=_S_EN + f": the shape is right, but your block has {grouped(_KERNEL_3, 'en')} parameters instead of {grouped(bottleneck_params(64, 16), 'en')}, because `reduce` and `expand` have filters of 3 by 3. A 1×1 convolution has a kernel size of 1: `nn.Conv2d(channels, middle, 1)`. It needs no padding, because it does not look past its own pixel.",
                    nl=_S_NL + f": de shape klopt, maar je block heeft {grouped(_KERNEL_3, 'nl')} parameters in plaats van {grouped(bottleneck_params(64, 16), 'nl')}, omdat `reduce` en `expand` filters van 3 bij 3 hebben. Een 1×1 convolution heeft een kernel size van 1: `nn.Conv2d(channels, middle, 1)`. Hij heeft geen padding nodig, omdat hij niet verder kijkt dan zijn eigen pixel."),
            Mistake("no-relu", wrong=_B_WRONG["no-relu"],
                    en=_S_EN + ": without a ReLU between them, three convolutions in a row are no more than one convolution. Put `torch.relu` around `self.reduce(x)`, `self.conv(x)` and `self.expand(x)`.",
                    nl=_S_NL + ": zonder ReLU ertussen zijn drie convoluties achter elkaar niet meer dan één convolutie. Zet `torch.relu` om `self.reduce(x)`, `self.conv(x)` en `self.expand(x)`."),
            Mistake("last-relu", wrong=_B_WRONG["last-relu"],
                    en=_S_EN + ": numbers below 0 come out of your block, so `expand` has no ReLU after it. The block stands in for a 3×3 convolution with its ReLU: give `expand` one too.",
                    nl=_S_NL + ": er komen getallen onder 0 uit je block, dus heeft `expand` geen ReLU erna. Het block staat op de plek van een 3×3-convolutie met zijn ReLU: geef `expand` er ook een."),
            Mistake("added", wrong=_B_WRONG["added"],
                    en=_S_EN + ": your block adds its input to what `expand` gives, as ResNet-50's bottleneck block does. Here the block stands in for one 3×3 convolution, without that addition: give back `torch.relu(self.expand(x))`.",
                    nl=_S_NL + ": je block telt zijn input op bij wat `expand` geeft, zoals het bottleneck block van ResNet-50 doet. Hier staat het block op de plek van één 3×3-convolutie, zonder die optelling: geef `torch.relu(self.expand(x))` terug."),
            Mistake("no-bias", wrong=_B_WRONG["no-bias"],
                    en=_S_EN + f": your block has {grouped(bottleneck_params(64, 16) - 16 - 64, 'en')} parameters instead of {grouped(bottleneck_params(64, 16), 'en')}, because `reduce` and `expand` have no bias. ResNet leaves them out because batch normalization follows; here nothing follows, so leave out `bias=False`.",
                    nl=_S_NL + f": je block heeft {grouped(bottleneck_params(64, 16) - 16 - 64, 'nl')} parameters in plaats van {grouped(bottleneck_params(64, 16), 'nl')}, omdat `reduce` en `expand` geen bias hebben. ResNet laat ze weg omdat er batch normalization volgt; hier volgt niets, dus laat `bias=False` weg."),
            Mistake("padding", wrong=_B_WRONG["padding"],
                    en=_S_EN + ": your block gives maps of 12 by 12 instead of 8 by 8, because `reduce` and `expand` have `padding=1`. A 1×1 convolution fits on every pixel, so it needs no padding; with it, each adds a border of zeros, 2 more rows and columns.",
                    nl=_S_NL + ": je block geeft maps van 12 bij 12 in plaats van 8 bij 8, omdat `reduce` en `expand` `padding=1` hebben. Een 1×1 convolution past op elke pixel, dus hij heeft geen padding nodig; daarmee zet elk er een rand van nullen omheen, 2 rijen en kolommen meer."),
            Mistake("no-expand", wrong=_B_WRONG["no-expand"],
                    en=_S_EN + ": your block gives `middle` channels, 16 instead of 64, so `expand` is missing. After `conv` comes `expand`, back to `channels`: then the block fits where the 3×3 convolution was.",
                    nl=_S_NL + ": je block geeft `middle` channels, 16 in plaats van 64, dus ontbreekt `expand`. Na `conv` komt `expand`, terug naar `channels`: dan past het block waar de 3×3-convolutie zat."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...` with your code: `reduce`, `expand`, and the body of `forward`.",
                      "nl": "Vervang de drie `...` door je code: `reduce`, `expand`, en wat `forward` doet."},
            "wrong": {"en": _S_EN + ". Are `reduce` and `expand` convolutions with a kernel size of 1, from `channels` to `middle` and back, and does `forward` use `reduce`, `conv` and `expand` in that order, each with a ReLU after it?",
                      "nl": _S_NL + ". Zijn `reduce` en `expand` convoluties met een kernel size van 1, van `channels` naar `middle` en terug, en gebruikt `forward` `reduce`, `conv` en `expand` in die volgorde, elk met een ReLU erna?"},
            "almost": {"en": "Right for the first lines. But " + _S_EN[0].lower() + _S_EN[1:] + ". Does your block work for every number of channels, and every size of the maps?",
                       "nl": "Goed voor de eerste regels. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je block voor elk aantal channels, en elke grootte van de maps?"},
            "correct": {"en": "Your `Bottleneck` is right: from 64 channels to 16, a 3×3 convolution on those 16, and back to 64, with 4,448 parameters, where one 3×3 convolution from 64 to 64 has 36,928. It is now in your project, as `{saved}`.",
                        "nl": "Je `Bottleneck` klopt: van 64 channels naar 16, een 3×3-convolutie op die 16, en terug naar 64, met 4.448 parameters, waar één 3×3-convolutie van 64 naar 64 er 36.928 heeft. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `Bottleneck` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `Bottleneck` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_B_ORDER = _B.replace("        self.conv = nn.Conv2d(middle, middle, 3, padding=1)\n", "").replace(
    "        self.expand = nn.Conv2d(middle, channels, 1)\n",
    "        self.expand = nn.Conv2d(middle, channels, 1)\n        self.conv = nn.Conv2d(middle, middle, 3, padding=1)\n")
CASES = [
    ("one-by-one", {}, "empty", None),
    ("one-by-one", _SORT.solution, "correct", None),
    ("one-by-one", {"gray": "can", "edge": "larger"}, "wrong", "unfinished"),
    ("one-by-one", {**_SORT.solution, "shift": "can"}, "wrong", "wrong:shift"),
    ("one-by-one", {**_SORT.solution, "fewer": "larger"}, "wrong", "wrong:fewer"),
    ("bottleneck", _B_START, "empty", None),
    ("bottleneck", _B, "correct", None),
    # the layers made in another order: the scenarios give them their weights by name
    ("bottleneck", _B_ORDER, "correct", None),
    # a number in place of channels or middle
    ("bottleneck", _b(reduce="nn.Conv2d(64, 16, 1)", expand="nn.Conv2d(16, 64, 1)"), "wrong", "fixed"),
    ("bottleneck", _b(reduce="nn.Conv2d(channels, 16, 1)"), "wrong", "fixed"),
    # the input added after x was made smaller: the channels do not match
    ("bottleneck", _b(body="        x = torch.relu(self.reduce(x))\n        x = torch.relu(self.conv(x))\n        return torch.relu(self.expand(x) + x)\n"), "error", "add-sizes"),
    # kernel_size by name, and the ReLU with nn.functional
    ("bottleneck", _b(reduce="nn.Conv2d(channels, middle, kernel_size=1)", expand="nn.Conv2d(middle, channels, kernel_size=1)"), "correct", None),
    ("bottleneck", _b(body="        x = nn.functional.relu(self.reduce(x))\n        x = nn.functional.relu(self.conv(x))\n        return nn.functional.relu(self.expand(x))\n"), "correct", None),
    # reduce and expand the other way around
    ("bottleneck", _b(reduce="nn.Conv2d(middle, channels, 1)", expand="nn.Conv2d(channels, middle, 1)"), "error", "channels"),
    # expand before conv
    ("bottleneck", _b(body="        x = torch.relu(self.reduce(x))\n        x = torch.relu(self.expand(x))\n        return torch.relu(self.conv(x))\n"), "error", "channels"),
    ("bottleneck", _b(body="        x = torch.relu(self.reduce(x))\n        x = torch.relu(self.conv(x))\n        torch.relu(self.expand(x))\n"), "error", "no-return"),
    ("bottleneck", _B_START.replace("self.reduce = ...", "self.reduce = nn.Conv2d(channels, middle, 1)"), "wrong", "dots-left"),
    # middle in place of channels in expand: the block gives 16 channels
    ("bottleneck", _b(expand="nn.Conv2d(middle, middle, 1)"), "wrong", None),
] + [("bottleneck", code, "wrong", key) for key, code in _B_WRONG.items()]


# --- the demo: the learner's Bottleneck in the middle of WideCNN, on CIFAR-10 --------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def _accuracy(model, X, y) -> int:
    model.eval()
    with torch.no_grad():
        return int(sum((model(X[i:i + 500]).argmax(dim=1) == y[i:i + 500]).sum() for i in range(0, len(X), 500)))


def run(variant: str, seed: int, Bottleneck, channel_stats, Trainer, count_params, show=None):
    """One run of WideCNN, as REFERENCE says: the block of `variant` (a Plain 3×3 convolution, or Bottleneck with MIDDLE
    in the middle) made after torch.manual_seed(seed), then the rest of the network, and the photos shuffled after
    torch.manual_seed(seed + 1). Gives back a dict with the parameters of the network and of its block, how many of
    the training and test photos it reads right, the loss after every epoch ("curves"), the epoch with the lowest
    validation loss, and the network's guess for the photos SHOWN after every epoch ("guesses") with their right kinds
    ("truth"). Errors go to the caller, which says whose code made them. `show` hears the run while it trains."""
    X, y, X_test, y_test = photos()
    mean, std = channel_stats(X.clone())
    mean = torch.as_tensor(mean, dtype=torch.float32).reshape(3)
    std = torch.as_tensor(std, dtype=torch.float32).reshape(3)
    X = (X - mean[:, None, None]) / std[:, None, None]
    X_test = (X_test - mean[:, None, None]) / std[:, None, None]
    torch.manual_seed(seed)
    middle = MIDDLE[variant]
    block = Plain(CHANNELS) if middle is None else Bottleneck(CHANNELS, middle)
    model = WideCNN(block)
    params, block_params = int(count_params(model)), int(count_params(block))
    shown, guesses = X_test[list(SHOWN)], []

    def heard(e):
        if "val_loss" in e:   # after every epoch: the guesses for the rewind, in eval mode, so the run stays the same
            was = model.training
            model.eval()
            with torch.no_grad():
                guesses.append([int(g) for g in model(shown).argmax(dim=1)])
            model.train(was)
        if show is not None:
            show(e)

    torch.manual_seed(seed + 1)
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    val = DataLoader(TensorDataset(X_test, y_test), batch_size=500)
    trainer = Trainer(model, loader, val, epochs=EPOCHS, lr=LR, warmup=len(loader), max_norm=1.0)
    live.follow(trainer, heard, "bottleneck")
    history = trainer.fit() or []
    curves = {"train": [float(h["train_loss"]) for h in history if isinstance(h, dict) and "train_loss" in h],
              "val": [float(h["val_loss"]) for h in history if isinstance(h, dict) and "val_loss" in h]}
    return {"variant": variant, "params": params, "block_params": block_params, "train": _accuracy(model, X, y),
            "test": _accuracy(model, X_test, y_test), "curves": curves, "best": best_epoch(curves["val"]),
            "guesses": guesses[:len(curves["val"])], "truth": [int(y_test[place]) for place in SHOWN]}


def _demo(variant: str, show=None):
    """For the demo: None when bottleneck.py is not in the project yet, for a run with a bottleneck; "data" when
    CIFAR-10 is not unpacked; "bottleneck" when the learner's Bottleneck stops with an error or does not give back 64
    maps of 8 by 8; "cnn" when channel_stats does not give what it should; "trainer" when the Trainer or count_params
    stops with an error; "slow" when the training takes too long; else the dict of run(), with the modules the course's
    versions stood in for."""
    Bottleneck = None   # only a run with a bottleneck uses it
    if MIDDLE[variant] is not None:
        if not _saved("bottleneck"):
            return None
    if not ready():
        return "data"
    if MIDDLE[variant] is not None:
        loaded = run_code("from cnn.bottleneck import Bottleneck")
        if loaded.error is not None:
            return "bottleneck"
        Bottleneck = loaded.names["Bottleneck"]
        try:
            def tried():   # as in training: a mini-batch, in training mode, with a backward pass
                torch.manual_seed(0)
                block = Bottleneck(CHANNELS, MIDDLE[variant])
                out = block(torch.rand(4, CHANNELS, SIDE, SIDE))
                out.sum().backward()
                return tuple(out.shape)
            if with_time_limit(tried, 10) != (4, CHANNELS, SIDE, SIDE):
                return "bottleneck"
        except Exception:
            return "bottleneck"
    loaded = run_code("from cnn.channels import channel_stats")
    if loaded.error is not None:
        return "cnn"
    channel_stats = loaded.names["channel_stats"]
    loaded = run_code("from nn.trainer import Trainer\nfrom cnn.count import count_params")
    if loaded.error is not None:
        return "trainer"
    Trainer, count_params = loaded.names["Trainer"], loaded.names["count_params"]
    try:
        int(with_time_limit(lambda: count_params(WideCNN(Plain(CHANNELS))), 10))
    except Exception:
        return "trainer"
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    X, _, _, _ = photos()
    try:
        mean, std = with_time_limit(lambda: channel_stats(X.clone()), 60)
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(3)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(3)
        if not (torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
            return "cnn"
    except Exception:
        return "cnn"
    try:
        result = with_time_limit(lambda: run(variant, 0, Bottleneck, channel_stats, Trainer, count_params, show), 600)
    except Timeout:
        return "slow"
    except Exception:
        return "trainer"
    return result | {"stand_ins": stand_ins}


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file), and whether CIFAR-10
    is there."""
    texts = []
    for package, module in (("cnn", "bottleneck"), ("cnn", "channels"), ("cnn", "count"), ("nn", "train_epoch"), ("nn", "trainer")):
        path = projects.folder() / package / f"{module}.py"
        texts.append(path.read_text(encoding="utf-8") if path.exists() else None)
    return tuple(texts) + (ready(),)


def demo(variant: str = "b16", show=None):
    """The demo's result for `variant`, worked out once for the project files as they are; `show` hears the run when it
    trains."""
    key = (variant,) + _project_texts()
    if key not in _DEMO_RUNS:
        _DEMO_RUNS.clear()
        _DEMO_RUNS[key] = _demo(variant, show)
    return _DEMO_RUNS[key]
