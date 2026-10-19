# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-data-augmentation: a Dataset that shifts every digit by a random pixel or so each time it
is asked for one, and which changes keep the label of a digit. And the pictures: one digit with shifted variants, and
the validation loss of the same MLP trained with and without shifts."""
import functools
import html
import random
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from everything_ai import paths
from everything_ai.check import CodeBox, Mistake, Sort

# termen:allow verschuiving, verschuivingen
# (the shift of an image here, not the bias term)
DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md
EPOCHS = 40


@functools.lru_cache(maxsize=1)
def tensors():
    """The training images, every pixel divided by 255, and their labels."""
    d = np.load(paths.ROOT / DATA)
    return torch.tensor(d["train_images"], dtype=torch.float32) / 255, torch.tensor(d["train_labels"], dtype=torch.int64)


class ShiftedDigits(Dataset):
    """The course's own Dataset of the lesson: every time it gives an image, shifted by a random number of pixels
    from -k to k, down and across."""
    def __init__(self, images, labels, k):
        self.images = images
        self.labels = labels
        self.k = k

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i].reshape(28, 28)
        dy = random.randint(-self.k, self.k)
        dx = random.randint(-self.k, self.k)
        img = torch.roll(img, shifts=(dy, dx), dims=(0, 1))
        return img.reshape(784), self.labels[i]


def shift_of(x, original) -> list[int] | None:
    """By how many pixels `x` is `original` shifted, down and across, as [dy, dx]; None when it is no shift of it."""
    x = torch.as_tensor(x).reshape(28, 28)
    original = torch.as_tensor(original).reshape(28, 28)
    steps = sorted(range(-14, 14), key=abs)   # every roll of 28 pixels, the smallest shifts first
    for dy in steps:
        for dx in steps:
            if torch.equal(torch.roll(original, shifts=(dy, dx), dims=(0, 1)), x):
                return [dy, dx]
    return None


class _MLP(nn.Module):
    """The course's own DigitsMLP, for the picture."""
    def __init__(self, hidden):
        super().__init__()
        self.hidden = nn.Linear(784, hidden)
        self.out = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.out(torch.relu(self.hidden(x)))


@functools.lru_cache(maxsize=None)
def val_curve(k: int) -> list[float]:
    """The validation loss after each epoch of the MLP on the first 1,000 training digits, as in torch-loss-curves,
    with shifts of up to k pixels (k = 0: none). Worked out once."""
    X, y = tensors()
    X_val, y_val = X[5000:5500], y[5000:5500]
    loss_fn = nn.CrossEntropyLoss()
    data = TensorDataset(X[:1000], y[:1000]) if k == 0 else ShiftedDigits(X[:1000], y[:1000], k)
    torch.manual_seed(0)
    random.seed(0)
    model = _MLP(64)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    loader = DataLoader(data, batch_size=50, shuffle=True)
    torch.manual_seed(1)
    out = []
    for _ in range(EPOCHS):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            out.append(loss_fn(model(X_val), y_val).item())
        model.train()
    return out


def best(k: int) -> tuple[int, float]:
    """The epoch, counted from 1, with the lowest validation loss, and that loss."""
    curve = val_curve(k)
    i = int(np.argmin(curve))
    return i + 1, curve[i]


def curves_picture(labels: dict[str, str]) -> str:
    """The validation loss per epoch without shifts (thin) and with shifts of up to 1 pixel (thick), both dashed: both
    are validation losses."""
    from everything_ai import chart
    plain, shifted = val_curve(0), val_curve(1)
    epochs = list(range(1, EPOCHS + 1))
    p = chart.Panel((0, EPOCHS + 1), (0, 1.2), title=labels["title"], xlabel=labels["epoch"], ylabel=labels["loss"])
    p.line(epochs, plain, "loss", 1.4, dashed=True)
    p.line(epochs, shifted, "loss", 3, dashed=True)
    p.text(EPOCHS, plain[-1] + 0.08, labels["plain"], "ink", "end")
    p.text(EPOCHS, shifted[-1] - 0.11, labels["shifted"], "ink", "end")
    return chart.figure([p], alt=labels["alt"], width=520, height=300)


# --- the picture of shifted digits: images of 28 by 28 as squares in shades of the data role, so they follow the theme ---

_SHADES = [0.3, 0.55, 0.8, 1.0]


def _image(img, x0: float, y0: float, size: float) -> str:
    img = np.asarray(img, dtype=float).reshape(28, 28)
    parts = {}
    for r in range(28):
        for c in range(28):
            v = img[r, c]
            if v < 0.1:
                continue
            parts.setdefault(min(3, int(v * 4)), []).append(f"M{x0 + c * size:g} {y0 + r * size:g}h{size:g}v{size:g}h{-size:g}z")
    out = [f'<rect x="{x0 - 0.5:g}" y="{y0 - 0.5:g}" width="{28 * size + 1:g}" height="{28 * size + 1:g}" '
           f'style="fill:none;stroke:var(--ea-line);stroke-width:1"/>']
    for level, d in sorted(parts.items()):
        out.append(f'<path d="{"".join(d)}" style="fill:var(--ea-data);fill-opacity:{_SHADES[level]}"/>')
    return "".join(out)


def variants_picture(labels: dict[str, str]) -> str:
    """Training digit 0, a 5, as it is and shifted four ways by one pixel, each with its shift below."""
    X, _ = tensors()
    img = X[0].reshape(28, 28)
    shifts = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    size, gap, top_y = 3, 18, 30
    box = 28 * size
    parts = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    for i, (dy, dx) in enumerate(shifts):
        x = 1 + i * (box + gap)
        parts.append(_image(torch.roll(img, shifts=(dy, dx), dims=(0, 1)), x, top_y, size))
        caption = labels["original"] if (dy, dx) == (0, 0) else f"({dy}, {dx})"
        parts.append(f'<text x="{x + box / 2:g}" y="{top_y + box + 20:g}" style="font-size:13px;fill:var(--ea-ink-2);text-anchor:middle">'
                     f'{html.escape(caption)}</text>')
    width, height = 5 * box + 4 * gap + 2, top_y + box + 34
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:min(40rem,{width}px)">'
            f'<svg viewBox="0 0 {width} {height}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(parts)}</svg></figure>')


# The lesson is done when these are correct. "where" is the choice question in the lesson.
REQUIRED = ["where", "shifted", "keeps-label"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# --- exercise 2: a Dataset that shifts ---------------------------------------------------------------------------------

_S_HEAD = ("import random\nimport numpy as np\nimport torch\nfrom torch.utils.data import Dataset\n\n"
           "data = np.load(\"data/qmnist-small.npz\")\nX = torch.tensor(data[\"train_images\"], dtype=torch.float32) / 255\n"
           "y = torch.tensor(data[\"train_labels\"], dtype=torch.int64)\n\n\n"
           "class ShiftedDigits(Dataset):\n    def __init__(self, images, labels, k):\n        self.images = images\n"
           "        self.labels = labels\n        self.k = k\n\n    def __len__(self):\n        return len(self.labels)\n\n"
           "    def __getitem__(self, i):\n        img = self.images[i].reshape(28, 28)\n        dy = random.randint(-self.k, self.k)\n")
_S_BODY = "        dx = random.randint(-self.k, self.k)\n        img = torch.roll(img, shifts=(dy, dx), dims=(0, 1))\n"
_S_END = ("        return img.reshape(784), self.labels[i]\n\n\n"
          "random.seed(3)\nds = ShiftedDigits(X[:10], y[:10], 2)\nx1, label = ds[2]\nx2, label = ds[2]\n"
          "[label.item(), shift_of(x1, X[2]), shift_of(x2, X[2])]")
_S_START = _S_HEAD + "        ...\n" + _S_END + "\n"


def _s(body: str = _S_BODY) -> str:
    return _S_HEAD + body + _S_END


_S = _s()

EXERCISES = {
    "shifted": CodeBox(
        name="shifted",
        start=_S_START,
        solution=_S,
        keep=["        dy = random.randint(-self.k, self.k)", "        return img.reshape(784), self.labels[i]", "random.seed(3)",
              "ds = ShiftedDigits(X[:10], y[:10], 2)", "x1, label = ds[2]", "x2, label = ds[2]",
              "[label.item(), shift_of(x1, X[2]), shift_of(x2, X[2])]"],
        names={"shift_of": shift_of},
        first=[_DOTS],
        mistakes=[
            Mistake("no-shift", matches=lambda got, text: isinstance(got, list) and len(got) == 3 and got[1] == [0, 0] and got[2] == [0, 0],
                    en="Your last line gives `{got}`: both times the image comes back unshifted. After drawing `dx`, move the image with `torch.roll`, and give that result the name `img`.",
                    nl="Je laatste regel geeft `{got}`: beide keren komt de afbeelding onverschoven terug. Verschuif de afbeelding na het trekken van `dx` met `torch.roll`, en geef dat resultaat de naam `img`."),
            Mistake("not-a-shift", matches=lambda got, text: isinstance(got, list) and len(got) == 3 and (got[1] is None or got[2] is None),
                    en="Your last line gives `{got}`: `None` means the image that comes back is no shift of the original, but another change of it. Move the 28 by 28 image with `torch.roll`, along both axes, as in the worked example.",
                    nl="Je laatste regel geeft `{got}`: `None` betekent dat de afbeelding die terugkomt geen verschuiving van het origineel is, maar een andere verandering ervan. Verschuif de afbeelding van 28 bij 28 met `torch.roll`, langs beide axes, zoals in het uitgewerkte voorbeeld."),
            Mistake("no-dims", error=lambda e: isinstance(e, RuntimeError) and "must align" in str(e),
                    en="Python stops at `torch.roll`: it has two shifts, but no axes to move them along. Give the axes too, with `dims`, one for each number in `shifts`.",
                    nl="Python stopt bij `torch.roll`: die krijgt twee verschuivingen, maar geen axes om ze langs te doen. Geef ook de axes, met `dims`, één voor elk getal in `shifts`."),
            Mistake("flat-image", error=lambda e: isinstance(e, IndexError) and "Dimension out of range" in str(e),
                    en="Python stops at `torch.roll`: the tensor you move has only one axis, 784 numbers in a row, so there is no axis 1. Move `img`, the image of 28 by 28 from the first line of `__getitem__`.",
                    nl="Python stopt bij `torch.roll`: de tensor die je verschuift heeft maar één axis, 784 getallen op een rij, dus er is geen axis 1. Verschuif `img`, de afbeelding van 28 bij 28 uit de eerste regel van `__getitem__`."),
            Mistake("swapped", wrong=_s("        dx = random.randint(-self.k, self.k)\n        img = torch.roll(img, shifts=(dx, dy), dims=(0, 1))\n"),
                    en="Your last line gives `{got}`: the right shifts, but down and across are swapped. `dy` is drawn first and is the shift down, along axis 0, the rows; `dx` comes next and is the shift across, along axis 1, the columns. In `torch.roll`, each number in `shifts` goes with the axis at the same place in `dims`.",
                    nl="Je laatste regel geeft `{got}`: de goede verschuivingen, maar omlaag en opzij zijn verwisseld. `dy` wordt eerst getrokken en is de verschuiving omlaag, langs axis 0, de rijen; `dx` komt daarna en is de verschuiving opzij, langs axis 1, de kolommen. In `torch.roll` hoort elk getal in `shifts` bij de axis op dezelfde plek in `dims`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
            "empty": {"en": "Replace the `...`: draw `dx` the same way as `dy`, then shift `img` by `dy` down and `dx` across.",
                      "nl": "Vervang de `...`: trek `dx` op dezelfde manier als `dy`, en verschuif `img` dan over `dy` rijen omlaag en `dx` kolommen opzij."},
            "no_value": {"en": "The last line of the box shows nothing. Put the list back on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet de lijst terug op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be `[4, [-1, 2], [2, -1]]`: the label of example 2, and its two shifts, down and across. Is `dx` drawn right after `dy`, in the same way, and is the image shifted by both?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet `[4, [-1, 2], [2, -1]]` zijn: het label van voorbeeld 2, en zijn twee verschuivingen, omlaag en opzij. Wordt `dx` direct na `dy` getrokken, op dezelfde manier, en wordt de afbeelding met allebei verschoven?"},
            "correct": {"en": "The label stays a 4, and the image comes back with a newly drawn shift each time it is asked for: {got}. So every epoch, the model sees a new variant of each digit.",
                        "nl": "Het label blijft een 4, en de afbeelding komt elke keer dat erom gevraagd wordt met een andere verschuiving terug: {got}. Zo ziet het model elke epoch een nieuwe variant van elk cijfer."},
        },
    ),
    "keeps-label": Sort(
        name="keeps-label",
        groups={
            "keeps": {"en": "Keeps the digit", "nl": "Blijft hetzelfde cijfer"},
            "breaks": {"en": "Can change the digit or make it unreadable", "nl": "Kan het cijfer veranderen of onleesbaar maken"},
        },
        items={
            "shift": {"en": "Shift it one pixel to the right", "nl": "Eén pixel naar rechts schuiven"},
            "mirror": {"en": "Mirror it, left to right", "nl": "Spiegelen, van links naar rechts"},
            "upside": {"en": "Turn it upside down", "nl": "Ondersteboven draaien"},
            "rotate": {"en": "Turn it 10 degrees", "nl": "Tien graden draaien"},
            "half": {"en": "Cut off its bottom half", "nl": "De onderste helft wegsnijden"},
            "lighter": {"en": "Make the ink a little lighter", "nl": "De inkt een beetje lichter maken"},
        },
        solution={"shift": "keeps", "mirror": "breaks", "upside": "breaks", "rotate": "keeps", "half": "breaks", "lighter": "keeps"},
        nudge={
            "shift": {"en": "Does a digit that is written a little further to the right stay the same digit?",
                      "nl": "Blijft een cijfer dat iets verder naar rechts geschreven is, hetzelfde cijfer?"},
            "rotate": {"en": "Do people write their digits perfectly straight?",
                       "nl": "Schrijven mensen hun cijfers precies recht?"},
            "lighter": {"en": "Does a pen that writes lighter change which digit it is?",
                        "nl": "Verandert een pen die lichter schrijft welk cijfer het is?"},
            "mirror": {"en": "What does a 2 or a 3 look like in a mirror?",
                       "nl": "Hoe ziet een 2 of een 3 eruit in een spiegel?"},
            "upside": {"en": "What does a 6 look like upside down?",
                       "nl": "Hoe ziet een 6 er ondersteboven uit?"},
            "half": {"en": "Can you still tell an 8 from a 9 by their top halves alone?",
                     "nl": "Kun je een 8 nog van een 9 onderscheiden aan alleen hun bovenste helft?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the changes again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de veranderingen opnieuw uit."},
            "empty": {"en": "Put each change in a group: keeps the digit, or can change it or make it unreadable.",
                      "nl": "Zet elke verandering in een groep: blijft hetzelfde cijfer, of kan het veranderen of onleesbaar maken."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs in the other group.",
                      "nl": "{item} hoort in de andere groep."},
            "correct": {"en": "All {n} are right. A shift, a small turn or lighter ink is how real handwriting varies; a mirror, upside down or half a digit can turn it into another digit, or into none.",
                        "nl": "Alle {n} kloppen. Een verschuiving, een kleine draai of lichtere inkt is hoe echt handschrift verschilt; spiegelen, ondersteboven of een half cijfer kan er een ander cijfer van maken, of helemaal geen cijfer."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("shifted", _S_START, "empty", None),
    ("shifted", _s("        dx = random.randint(-self.k, self.k)\n        shifted = torch.roll(img, (dy, dx), (0, 1))\n        img = shifted\n"), "correct", None),
    ("shifted", _s("        dx = random.randint(-self.k, self.k)\n        torch.roll(img, shifts=(dy, dx), dims=(0, 1))\n"), "wrong", "no-shift"),
    ("shifted", _s("        dx = random.randint(-self.k, self.k)\n        img = img.T\n"), "wrong", "not-a-shift"),
    ("shifted", _s("        dx = random.randint(-self.k, self.k)\n        img = torch.roll(img, shifts=(dx, dy), dims=(0, 1))\n"), "wrong", "swapped"),
    ("shifted", _s("        dx = random.randint(-self.k, self.k)\n        img = torch.roll(img, dy, 0)\n        img = torch.roll(img, dx, 1)\n"), "correct", None),
    ("shifted", _s("        dx = random.randint(-self.k, self.k)\n        img = torch.roll(img, shifts=(dy, dx), dims=(1, 0))\n"), "wrong", "swapped"),
    ("shifted", _S.replace("        dy = random.randint(-self.k, self.k)\n", "        dx = random.randint(-self.k, self.k)\n        dy = random.randint(-self.k, self.k)\n", 1)
                  .replace("        dx = random.randint(-self.k, self.k)\n        img = torch.roll", "        img = torch.roll", 1), "wrong", "swapped"),
    ("shifted", _s("        dx = random.randint(-self.k, self.k)\n        img = torch.flip(img, dims=(1,))\n"), "wrong", "not-a-shift"),
    ("shifted", _S.replace("ShiftedDigits(X[:10], y[:10], 2)", "ShiftedDigits(X[:10], y[:10], 0)"), "wrong", "changed-line"),
    ("shifted", _s("        dx = random.randint(-self.k, self.k)\n        img = torch.roll(img, shifts=(dy, dx))\n"), "error", "no-dims"),
    ("shifted", _s("        dx = random.randint(-self.k, self.k)\n        img = torch.roll(self.images[i], shifts=(dy, dx), dims=(0, 1))\n"), "error", "flat-image"),
    ("shifted", _s("        dx = random.randint(-self.k, self.k)\n        img = torch.roll(torch.roll(img, shifts=(dy, dx), dims=(0, 1)), shifts=(dy, dx), dims=(0, 1))\n"), "wrong", None),
]
