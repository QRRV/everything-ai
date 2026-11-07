# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson cv-prepare-photo: the middle square of an image, and a photo made into the input of a pretrained
image model, with every color scaled with the mean and the standard deviation of ImageNet, the channels first and a
batch of one. And the pictures the lesson uses. The choice about the shape after step 1 and the open question about a
photo scaled with its own numbers are in the lesson itself."""
import functools

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai.check import CodeBox, FunctionBox, Mistake

PHOTO = "data/photos/coffee.png"    # 600 by 400 pixels: the worked example and the pictures
CAT = "data/photos/chelsea.png"     # 451 by 300 pixels: exercise 3 uses an upright piece of it
PIECE = (100, 300)                  # the columns of that piece: 200 wide and 300 high
SHORT, SIZE = 256, 224              # the shorter side after step 1, and the side of the middle square
MEAN = (0.485, 0.456, 0.406)        # per color, the mean of ImageNet's training photos, on a scale from 0 to 1
STD = (0.229, 0.224, 0.225)         # and their standard deviation
SIDES = [224, 256, 320, 384, 448]   # the stops of the slider: the shorter side after step 1


def read(path: str) -> np.ndarray:
    """An image file as an array, as the lesson reads it."""
    from PIL import Image
    return np.array(Image.open(path))


def grouped(n: int, sep: str) -> str:
    """A whole number with a separator between groups of three digits: 2,585 and 15,544; Dutch writes 15.544, but a
    number of four digits without a dot, 2585."""
    if sep == "." and abs(int(n)) < 10_000:
        return str(int(n))
    return f"{int(n):,}".replace(",", sep)


def percent(x: float, sep: str) -> str:
    """A percentage with at most one decimal: 87.5, 70, 58.3; in Dutch with a comma, 87,5."""
    return f"{round(x, 1):g}".replace(".", sep)


def size_after(width: int, height: int, short: int = SHORT) -> tuple[int, int]:
    """Step 1: the size, as Pillow writes it (width, height), with the shorter side `short` and the ratio kept; what
    comes after the point is dropped, as PyTorch's own tools for images do."""
    if width <= height:
        return short, int(short * height / width)
    return int(short * width / height), short


def upright():
    """The piece of the cat photo that exercise 3 uses, as a photo taken upright: 200 pixels wide and 300 high."""
    from PIL import Image
    return Image.fromarray(read(CAT)[:, PIECE[0]:PIECE[1]])


def resized(photo, short: int = SHORT) -> np.ndarray:
    """Step 1 on a photo (a path, or an image opened with Pillow): the shorter side `short`, the ratio kept, with
    Pillow's bilinear interpolation."""
    from PIL import Image
    if isinstance(photo, str):
        photo = Image.open(photo)
    return np.array(photo.resize(size_after(*photo.size, short), Image.BILINEAR))


def center(img: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Step 2: the middle square of `size` by `size` pixels."""
    top, left = int((img.shape[0] - size) / 2), int((img.shape[1] - size) / 2)
    return img[top:top + size, left:left + size]


def scaled(crop: np.ndarray) -> np.ndarray:
    """Steps 3 and 4: from 0 to 1, then per color the mean and the standard deviation of ImageNet."""
    return (crop / 255 - np.array(MEAN)) / np.array(STD)


def prepared(photo):
    """All five steps: the tensor a pretrained image model takes, of shape (1, 3, 224, 224) and float32."""
    return torch.tensor(scaled(center(resized(photo)))).float().permute(2, 0, 1).unsqueeze(0)


def _panel(ax, img, title, **kw):
    ax.imshow(img, **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title, fontsize=10)


def _dimmed(img: np.ndarray, size: int = SIZE) -> np.ndarray:
    """The image with everything outside its middle square darker: what step 2 cuts away."""
    out = (img * 0.35).astype(np.uint8)
    top, left = int((img.shape[0] - size) / 2), int((img.shape[1] - size) / 2)
    out[top:top + size, left:left + size] = img[top:top + size, left:left + size]
    return out


def picture(labels: dict[str, str]) -> str:
    """The coffee photo squashed to 224 by 224; made 384 by 256, with what step 2 cuts away darker; and the middle
    square that the model gets, all at the same scale."""
    import matplotlib.pyplot as plt
    from PIL import Image
    from everything_ai import plots
    squashed = np.array(Image.open(PHOTO).resize((SIZE, SIZE), Image.BILINEAR))
    small = resized(PHOTO)
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 1.95), gridspec_kw={"width_ratios": [224, 384, 224]})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.01, wspace=0.06)
    for ax, img, title in zip(axes, (squashed, _dimmed(small), center(small)),
                              (labels["squashed"], labels["resized"], labels["crop"])):
        _panel(ax, img, title)
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


def side_frame(short: int, labels: dict[str, str]) -> str:
    """For the slider: the coffee photo with its shorter side `short`, with what step 2 cuts away darker, drawn at
    the same height at every stop, and the middle square of 224 by 224 that the model gets."""
    import matplotlib.pyplot as plt
    from PIL import Image
    from everything_ai import plots
    short = int(short)
    small = resized(PHOTO, short)
    shown = np.array(Image.fromarray(_dimmed(small)).resize((300, 200), Image.BILINEAR))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), gridspec_kw={"width_ratios": [300, 200]})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, wspace=0.05)
    _panel(axes[0], shown, labels["resized"].format(w=small.shape[1], h=small.shape[0]))
    _panel(axes[1], center(small), labels["crop"])
    html = plots.render(fig, alt=labels["alt"].format(w=small.shape[1], h=small.shape[0]))
    plt.close(fig)
    return html


REQUIRED = ["shape", "crop", "prepare", "own"]

_SAME = {"en": "`{call}` gives `{got}`, and it should give `{want}`.", "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn."}


def _m(key_, en, nl, **kw):
    """A mistake whose message starts with what the call gives and what it should give."""
    return Mistake(key_, en=_SAME["en"] + " " + en, nl=_SAME["nl"] + " " + nl, **kw)


class _Arr(np.ndarray):
    """An image of a case, shown in the check's messages as the code that makes it."""

    def __repr__(self):
        shape = ", ".join(str(n) for n in self.shape)
        return f"np.arange({self.size}).reshape({shape})"


def _arr(*shape) -> np.ndarray:
    return np.arange(int(np.prod(shape))).reshape(shape).view(_Arr)


def _shown(v):
    """What the learner's function gave: the numbers of an array, or only its shape when it holds many (in words that
    are the same in both languages)."""
    if isinstance(v, np.ndarray):
        arr = np.asarray(v)
        if arr.size <= 48:
            return str(arr.tolist())
        return f"array, shape {arr.shape}"
    return repr(v)


# Exercise 2: the middle square
_CROP_HEAD = "def center_crop(img, size):\n"
_CROP_START = _CROP_HEAD + "    top = ...\n    left = ...\n    return ...\n"


def _crop(top: str = "int((img.shape[0] - size) / 2)", left: str = "int((img.shape[1] - size) / 2)",
          back: str = "img[top:top + size, left:left + size]") -> str:
    """The course's function with another top, left or slice."""
    return _CROP_HEAD + f"    top = {top}\n    left = {left}\n    return {back}\n"


_CROP = _crop()

# Exercise 3: an upright piece of the cat photo made into the input of the model
_PREP_HEAD = ("import numpy as np\n"
              "import torch\n"
              "from PIL import Image\n\n"
              "cat = np.array(Image.open(\"data/photos/chelsea.png\"))\n"
              "# an upright piece, 200 wide and 300 high: as a photo taken upright\n"
              "piece = Image.fromarray(cat[:, 100:300])\n"
              "# 1. the shorter side 256, the ratio kept: 300 × 256 / 200 = 384\n"
              "small = np.array(piece.resize((256, 384), Image.BILINEAR))\n"
              "mean = np.array([0.485, 0.456, 0.406])\n"
              "std = np.array([0.229, 0.224, 0.225])\n")
_PREP_START = _PREP_HEAD + "crop = ...\nx = ...\nx\n"


def _prep(line: str = "torch.tensor((crop / 255 - mean) / std).float().permute(2, 0, 1).unsqueeze(0)",
          crop: str = "small[80:304, 16:240]") -> str:
    """The box of exercise 3 with these lines for `crop` and `x`."""
    return _PREP_HEAD + f"crop = {crop}\nx = {line}\nx\n"


_PREP = _prep()
_MEAN, _STD = np.array(MEAN), np.array(STD)


@functools.cache
def _crop_cat() -> np.ndarray:
    """Steps 1 and 2 on the upright piece: its middle 224 by 224 at 256 by 384, as floats."""
    return center(resized(upright())).astype(float)


@functools.cache
def _want():
    return prepared(upright())


def _tensor(got) -> bool:
    return isinstance(got, torch.Tensor)


def _prep_right(got, want) -> bool:
    """The same tensor: the same shape, the same dtype, and the same numbers up to float32's last digits."""
    if not (_tensor(got) and _tensor(want)) or got.shape != want.shape or got.dtype != want.dtype:
        return False
    return bool(torch.allclose(got.detach().double(), want.detach().double(), atol=1e-4, rtol=0))


def _close(got, other) -> bool:
    """The learner's tensor has the numbers of `other`, whatever its dtype."""
    return (_tensor(got) and got.shape == other.shape
            and bool(torch.allclose(got.detach().double(), other.double(), atol=1e-4, rtol=0)))


def _has(*shapes):
    """`x` is a tensor of one of these shapes."""
    return lambda got, text: _tensor(got) and tuple(got.shape) in shapes


def _with(n: int):
    """`x` is a tensor or an array with an axis of length `n`."""
    return lambda got, text: (_tensor(got) or isinstance(got, np.ndarray)) and n in tuple(got.shape)


def _shape(got, text) -> dict:
    return {"shape": str(tuple(got.shape))}


def _form(got, text) -> dict:
    return {"shape": str(tuple(got.shape)), "dtype": str(got.dtype).replace("torch.", "")}


def _order(got, text) -> bool:
    """The right axes in another order: the colors not right after the batch, or the rows and columns swapped."""
    return (_tensor(got) and sorted(got.shape) in ([3, 224, 224], [1, 3, 224, 224])
            and tuple(got.shape) not in ((1, 3, 224, 224), (3, 224, 224)))


def _like(scale):
    """A mistake in steps 3 and 4: `x` holds `scale(crop)`, with the colors first and a batch of one, whatever its dtype."""
    return lambda got, text: _close(got, torch.tensor(np.asarray(scale(_crop_cat()))).permute(2, 0, 1).unsqueeze(0))


def _reshaped(got, text) -> bool:
    """The right shape and the right numbers, but in other places: `reshape` instead of `permute`."""
    want = _want()
    if not _tensor(got) or got.shape != want.shape or _close(got, want):
        return False
    a, b = torch.sort(got.detach().double().flatten()).values, torch.sort(want.double().flatten()).values
    return bool(torch.allclose(a, b, atol=1e-4, rtol=0))


def _own(got, text) -> bool:
    """Every color scaled with the photo's own mean and standard deviation: per color, mean 0 and spread 1."""
    if not _tensor(got) or tuple(got.shape) != (1, 3, 224, 224):
        return False
    g = got.detach().double()[0].reshape(3, -1)
    return bool(torch.allclose(g.mean(dim=1), torch.zeros(3, dtype=torch.float64), atol=1e-3)
                and torch.allclose(g.std(dim=1, correction=0), torch.ones(3, dtype=torch.float64), atol=1e-2))


EXERCISES = {
    "crop": FunctionBox(
        name="crop",
        function="center_crop",
        start=_CROP_START,
        solution=_CROP,
        visible=[(_arr(6, 8), 4)],
        hidden=[(_arr(4, 10, 3), 2), (_arr(10, 4), 2), (_arr(6, 6), 6), (_arr(8, 6, 3), 2)],
        show=_shown,
        mistakes=[
            Mistake("float", error=lambda e: isinstance(e, TypeError) and "slice indices must be integers" in str(e),
                    en="With `{call}`, your function stops with an error: `(6 - 4) / 2` gives `1.0`, a float, and Python only cuts a slice at whole numbers. Make it whole with `int(...)` around it, as in [Bigger and smaller: interpolation](/course/open/cv-resize-interpolation?lang=en).",
                    nl="Met `{call}` stopt je functie met een fout: `(6 - 4) / 2` geeft `1.0`, een float, en Python knipt een slice alleen bij hele getallen. Maak er een heel getal van met `int(...)` eromheen, zoals in [Groter en kleiner: interpolatie](/course/open/cv-resize-interpolation?lang=nl)."),
            Mistake("unpack", error=lambda e: isinstance(e, ValueError) and "too many values to unpack" in str(e),
                    en="With `{call}`, your function stops with an error: an image with colors has three numbers in its shape, rows, columns and colors, and two names before `= img.shape` have room for only two of them. Take the first two with a slice, `img.shape[:2]`, or use `img.shape[0]` and `img.shape[1]`.",
                    nl="Met `{call}` stopt je functie met een fout: een beeld met kleuren heeft drie getallen in zijn shape, rijen, kolommen en kleuren, en twee namen voor `= img.shape` hebben maar plek voor twee ervan. Neem de eerste twee met een slice, `img.shape[:2]`, of gebruik `img.shape[0]` en `img.shape[1]`."),
            _m("swap", "`top` counts rows, and `img.shape[0]` is the number of rows; `left` counts columns, `img.shape[1]`. Yours are the other way around, so the square is off the middle when the image is not square.",
               "`top` telt rijen, en `img.shape[0]` is het aantal rijen; `left` telt kolommen, `img.shape[1]`. Bij jou staan ze omgekeerd, dus staat het vierkant naast het midden als het beeld niet vierkant is.",
               wrong=_crop(top="int((img.shape[1] - size) / 2)", left="int((img.shape[0] - size) / 2)")),
            _m("no-half", "`img.shape[0] - size` is the whole margin, the rows that must go. Half of it goes above the square and half below: divide it by 2.",
               "`img.shape[0] - size` is de hele marge, de rijen die weg moeten. De helft gaat boven het vierkant weg en de helft eronder: deel hem door 2.",
               wrong=_crop(top="img.shape[0] - size", left="img.shape[1] - size")),
            _m("middle", "`img.shape[0] / 2` is the middle row, and your square starts there: it takes the bottom right quarter. The square starts half a margin from the top: `(img.shape[0] - size) / 2`.",
               "`img.shape[0] / 2` is de middelste rij, en je vierkant begint daar: het pakt het kwart rechtsonder. Het vierkant begint een halve marge van boven: `(img.shape[0] - size) / 2`.",
               wrong=_crop(top="int(img.shape[0] / 2)", left="int(img.shape[1] / 2)")),
            _m("half-size", "`size / 2` is half of the square, not half of the margin around it. The square starts at `(img.shape[0] - size) / 2`.",
               "`size / 2` is de helft van het vierkant, niet de helft van de marge eromheen. Het vierkant begint bij `(img.shape[0] - size) / 2`.",
               wrong=_crop(top="int(size / 2)", left="int(size / 2)")),
            _m("end", "A slice stops at its second number: `img[top:size]` stops at row `size`, not `size` rows after `top`. Write `img[top:top + size, left:left + size]`.",
               "Een slice stopt bij zijn tweede getal: `img[top:size]` stopt bij rij `size`, niet `size` rijen na `top`. Schrijf `img[top:top + size, left:left + size]`.",
               wrong=_crop(back="img[top:size, left:size]")),
            _m("corner", "Your square starts at the top left corner. Start it at `top` and `left`: `img[top:top + size, left:left + size]`.",
               "Je vierkant begint in de hoek linksboven. Laat het bij `top` en `left` beginnen: `img[top:top + size, left:left + size]`.",
               wrong=_crop(back="img[:size, :size]")),
            _m("rows-only", "Your function cuts only rows, so every column stays. Cut the columns too, after a comma: `img[top:top + size, left:left + size]`.",
               "Je functie knipt alleen rijen, dus blijft elke kolom staan. Knip ook de kolommen, na een komma: `img[top:top + size, left:left + size]`.",
               wrong=_crop(back="img[top:top + size]")),
            _m("brackets", "`img[top:top + size]` cuts the rows, and the second pair of brackets, `[left:left + size]`, cuts the rows of that again, not the columns. Put both slices in one pair of brackets, with a comma: `img[top:top + size, left:left + size]`.",
               "`img[top:top + size]` knipt de rijen, en het tweede paar haken, `[left:left + size]`, knipt daarvan opnieuw de rijen, niet de kolommen. Zet beide slices tussen één paar haken, met een komma: `img[top:top + size, left:left + size]`.",
               wrong=_crop(back="img[top:top + size][left:left + size]")),
            _m("minus", "`img[top:-top]` stops `top` rows before the end. That goes well as long as `top` is not 0, but `-0` is just 0: for an image as big as the square, `img[0:0]` is empty. And when the margin is odd, the square gets one row too many. Count from where it starts: `img[top:top + size, left:left + size]`.",
               "`img[top:-top]` stopt `top` rijen voor het eind. Dat gaat goed zolang `top` niet 0 is, maar `-0` is gewoon 0: bij een beeld dat even groot is als het vierkant, is `img[0:0]` leeg. En als de marge oneven is, krijgt het vierkant een rij te veel. Tel vanaf waar het begint: `img[top:top + size, left:left + size]`.",
               wrong=_crop(back="img[top:-top, left:-left]")),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace each `...`: `top` is half of the rows that do not fit, `left` half of the columns that do not fit, and the function gives back the square of `size` by `size` that starts there.",
                      "nl": "Vervang elke `...`: `top` is de helft van de rijen die niet passen, `left` de helft van de kolommen die niet passen, en de functie geeft het vierkant van `size` bij `size` terug dat daar begint."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with an image and a size, and the `def` line of your function has a different number of names between its parentheses. Keep `(img, size)`.",
                          "nl": "De check roept `{call}` aan, met een beeld en een maat, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(img, size)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Of the 6 rows, 2 do not fit: 1 goes above the square and 1 below, so it starts at row 1. Of the 8 columns, 4 do not fit, so it starts at column 2.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Van de 6 rijen passen er 2 niet: 1 gaat boven het vierkant weg en 1 eronder, dus het begint bij rij 1. Van de 8 kolommen passen er 4 niet, dus het begint bij kolom 2."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for an image that is taller than wide, for one with three colors, and for one exactly as big as the square?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor een beeld dat hoger is dan breed, voor een beeld met drie kleuren, en voor een beeld dat precies even groot is als het vierkant?"},
            "correct": {"en": "Your function is right in all {n} cases. On the coffee photo of 384 by 256 it starts at row 16 and column 80, as in the worked example.",
                        "nl": "Je functie klopt in alle {n} gevallen. Op de koffiefoto van 384 bij 256 begint hij bij rij 16 en kolom 80, zoals in het uitgewerkte voorbeeld."},
        },
    ),
    "prepare": CodeBox(
        name="prepare",
        start=_PREP_START,
        solution=_PREP,
        compare=_prep_right,
        mistakes=[
            Mistake("dots", error=lambda e: isinstance(e, TypeError) and "'ellipsis'" in str(e),
                    en="Python stops with an error: a line still has its `...`. First make `crop` the middle square of `small`, and then `x` from `crop`.",
                    nl="Python stopt met een fout: een regel heeft nog zijn `...`. Maak eerst van `crop` het middelste vierkant van `small`, en dan `x` van `crop`."),
            Mistake("permute-axes", error=lambda e: isinstance(e, RuntimeError) and "permute" in str(e) and "number of dimensions" in str(e),
                    en="Python stops at `.permute`: it needs one number for every axis of the tensor, and yours has another number of axes. After `.unsqueeze(0)` the tensor has four axes: put the colors first while it still has three, so first `.permute(2, 0, 1)`, and then `.unsqueeze(0)`.",
                    nl="Python stopt bij `.permute`: die heeft één getal nodig voor elke as van de tensor, en de jouwe heeft een ander aantal assen. Na `.unsqueeze(0)` heeft de tensor vier assen: zet de kleuren vooraan zolang hij er nog drie heeft, dus eerst `.permute(2, 0, 1)`, en dan `.unsqueeze(0)`."),
            Mistake("scale-late", error=lambda e: (isinstance(e, RuntimeError) and "(3) at non-singleton dimension" in str(e))
                    or (isinstance(e, ValueError) and "could not be broadcast" in str(e) and "(3,)" in str(e)),
                    en="Python stops with an error: `mean` and `std` have one number per color, and they are stretched over the last axis. Once the colors are moved to the front, that last axis holds the columns. Scale while the colors are still last: first `(crop / 255 - mean) / std`, then the tensor and `.permute`.",
                    nl="Python stopt met een fout: `mean` en `std` hebben één getal per kleur, en ze worden uitgerekt over de laatste as. Zodra de kleuren vooraan staan, staan op die laatste as de kolommen. Schaal zolang de kleuren nog achteraan staan: eerst `(crop / 255 - mean) / std`, dan de tensor en `.permute`."),
            Mistake("x-dots", matches=lambda got, text: got is Ellipsis,
                    en="`x` is still `...`: replace it with the tensor, made from `crop` with steps 3 to 5.",
                    nl="`x` is nog `...`: vervang het door de tensor, gemaakt van `crop` met stap 3 tot en met 5."),
            Mistake("numpy", matches=lambda got, text: isinstance(got, np.ndarray),
                    en="`x` is still a NumPy array. The model takes a tensor: make one with `torch.tensor(...)`, as in [PyTorch tensors](/course/open/torch-tensors?lang=en).",
                    nl="`x` is nog een NumPy-array. Het model verwacht een tensor: maak er een met `torch.tensor(...)`, zoals in [Tensors in PyTorch](/course/open/torch-tensors?lang=nl)."),
            Mistake("no-crop", matches=_with(384), detail=_shape,
                    en="`x` has shape `{shape}`: the whole photo after step 1, 384 rows by 256 columns. First cut the middle square out of it, step 2, and make `x` from `crop`.",
                    nl="`x` heeft shape `{shape}`: de hele foto na stap 1, 384 rijen bij 256 kolommen. Knip er eerst het middelste vierkant uit, stap 2, en maak `x` van `crop`."),
            Mistake("coffee-crop", matches=_with(176), detail=_shape,
                    en="`x` has shape `{shape}`: your square is 224 rows by 176 columns. `small` has only 256 columns, so a slice such as `80:304` stops at the last one. This photo is upright, 384 rows by 256 columns: of its rows, 160 do not fit, and of its columns 32. Where does the square start?",
                    nl="`x` heeft shape `{shape}`: je vierkant is 224 rijen bij 176 kolommen. `small` heeft maar 256 kolommen, dus een slice als `80:304` stopt bij de laatste. Deze foto staat rechtop, 384 rijen bij 256 kolommen: van zijn rijen passen er 160 niet, en van zijn kolommen 32. Waar begint het vierkant?"),
            Mistake("crop-size", matches=lambda got, text: _tensor(got) and got.dim() == 4 and tuple(got.shape[:2]) == (1, 3) and tuple(got.shape[2:]) != (224, 224),
                    detail=_shape,
                    en="`x` has shape `{shape}`: `crop` is not 224 by 224. Cut 224 rows and 224 columns out of the middle of `small`: `small[top:top + 224, left:left + 224]`.",
                    nl="`x` heeft shape `{shape}`: `crop` is niet 224 bij 224. Knip 224 rijen en 224 kolommen uit het midden van `small`: `small[top:top + 224, left:left + 224]`."),
            Mistake("channels-last", matches=_has((1, 224, 224, 3), (224, 224, 3)), detail=_shape,
                    en="`x` has shape `{shape}`: the colors are still last, as NumPy keeps them. The model wants them first: `.permute(2, 0, 1)` puts axis 2, the colors, in front.",
                    nl="`x` heeft shape `{shape}`: de kleuren staan nog achteraan, zoals NumPy ze houdt. Het model wil ze vooraan: `.permute(2, 0, 1)` zet as 2, de kleuren, vooraan."),
            Mistake("batch", matches=_has((3, 224, 224)),
                    en="`x` has shape `(3, 224, 224)`: one photo, but the model takes a batch. Put an axis of length 1 in front with `.unsqueeze(0)`: a batch of one, `(1, 3, 224, 224)`.",
                    nl="`x` heeft shape `(3, 224, 224)`: één foto, maar het model verwacht een batch. Zet er een as van lengte 1 voor met `.unsqueeze(0)`: een batch van één, `(1, 3, 224, 224)`."),
            Mistake("batch-place", matches=_has((3, 1, 224, 224), (3, 224, 1, 224), (3, 224, 224, 1)), detail=_shape,
                    en="`x` has shape `{shape}`: the axis of length 1 is not in front. The number in `.unsqueeze(...)` is the place of the new axis: `.unsqueeze(0)` puts it in front, `(1, 3, 224, 224)`.",
                    nl="`x` heeft shape `{shape}`: de as van lengte 1 staat niet vooraan. Het getal in `.unsqueeze(...)` is de plek van de nieuwe as: `.unsqueeze(0)` zet hem vooraan, `(1, 3, 224, 224)`."),
            Mistake("order", matches=_order, detail=_shape,
                    en="`x` has shape `{shape}`: the axes are in another order than `(1, 3, 224, 224)`. `.permute(...)` lists the old axes in their new order: the colors are axis 2 of the photo and come first, then the rows, axis 0, and the columns, axis 1: `.permute(2, 0, 1)`.",
                    nl="`x` heeft shape `{shape}`: de assen staan in een andere volgorde dan `(1, 3, 224, 224)`. `.permute(...)` noemt de oude assen in hun nieuwe volgorde: de kleuren zijn as 2 van de foto en komen eerst, dan de rijen, as 0, en de kolommen, as 1: `.permute(2, 0, 1)`."),
            Mistake("sideways", matches=lambda got, text: _close(got, _want().transpose(2, 3)),
                    en="`.permute(2, 1, 0)` also puts the colors first, but swaps the rows and the columns: the cat lies on its side. `.permute(2, 0, 1)` keeps rows before columns.",
                    nl="`.permute(2, 1, 0)` zet de kleuren ook vooraan, maar verwisselt de rijen en de kolommen: de kat ligt op zijn zij. `.permute(2, 0, 1)` houdt rijen vóór kolommen."),
            Mistake("reshape", matches=_reshaped,
                    en="The shape is right, but the numbers are in the wrong places. `reshape` keeps the numbers in their order and only cuts them anew: the first row of `x` holds red, green, blue, red, green, blue... of the first pixels. `.permute(2, 0, 1)` moves the axes themselves.",
                    nl="De shape klopt, maar de getallen staan op de verkeerde plekken. `reshape` houdt de getallen in hun volgorde en knipt ze alleen opnieuw: de eerste rij van `x` bevat rood, groen, blauw, rood, groen, blauw... van de eerste pixels. `.permute(2, 0, 1)` verplaatst de assen zelf."),
            Mistake("float64", matches=lambda got, text: _tensor(got) and str(got.dtype) == "torch.float64" and _close(got, _want()),
                    en="The numbers are right, but `x` is float64: a tensor made from a NumPy array keeps NumPy's float64, and the weights of the model are float32. Add `.float()`.",
                    nl="De getallen kloppen, maar `x` is float64: een tensor die van een NumPy-array is gemaakt, houdt NumPy's float64, en de weights van het model zijn float32. Zet er `.float()` achter."),
            Mistake("raw", matches=_like(lambda c: c),
                    en="`x` still holds the colors from 0 to 255, as the photo stores them. Steps 3 and 4 come first: `(crop / 255 - mean) / std`.",
                    nl="`x` heeft nog de kleuren van 0 tot en met 255, zoals de foto ze opslaat. Eerst komen stap 3 en 4: `(crop / 255 - mean) / std`."),
            Mistake("no-255", matches=_like(lambda c: (c - _MEAN) / _STD),
                    en="`crop` still runs from 0 to 255, and the mean and standard deviation of ImageNet are on a scale from 0 to 1. Divide by 255 first: `crop / 255`.",
                    nl="`crop` loopt nog van 0 tot en met 255, en het gemiddelde en de standaardafwijking van ImageNet staan op een schaal van 0 tot en met 1. Deel eerst door 255: `crop / 255`."),
            Mistake("after", matches=_like(lambda c: (c - _MEAN) / _STD / 255),
                    en="You divide by 255 after the scaling, so the mean is taken off numbers from 0 to 255. Divide by 255 first: `(crop / 255 - mean) / std`.",
                    nl="Je deelt door 255 na het schalen, dus gaat het gemiddelde af van getallen van 0 tot en met 255. Deel eerst door 255: `(crop / 255 - mean) / std`."),
            Mistake("swap", matches=_like(lambda c: (c / 255 - _STD) / _MEAN),
                    en="`mean` and `std` are swapped: take off the mean, and divide by the standard deviation, `(crop / 255 - mean) / std`.",
                    nl="`mean` en `std` zijn omgewisseld: haal het gemiddelde eraf, en deel door de standaardafwijking, `(crop / 255 - mean) / std`."),
            Mistake("mean-only", matches=_like(lambda c: c / 255 - _MEAN),
                    en="You take off the mean, but do not divide by the standard deviation: `(crop / 255 - mean) / std`.",
                    nl="Je haalt het gemiddelde eraf, maar deelt niet door de standaardafwijking: `(crop / 255 - mean) / std`."),
            Mistake("only-255", matches=_like(lambda c: c / 255),
                    en="The colors run from 0 to 1, but the model learned from photos scaled per color with the mean and the standard deviation of ImageNet: `(crop / 255 - mean) / std`.",
                    nl="De kleuren lopen van 0 tot en met 1, maar het model leerde van foto's die per kleur geschaald zijn met het gemiddelde en de standaardafwijking van ImageNet: `(crop / 255 - mean) / std`."),
            Mistake("brackets", matches=_like(lambda c: c / 255 - _MEAN / _STD),
                    en="Python divides before it subtracts, so `crop / 255 - mean / std` divides only `mean` by `std`. Put brackets around the subtraction: `(crop / 255 - mean) / std`.",
                    nl="Python deelt voordat het aftrekt, dus `crop / 255 - mean / std` deelt alleen `mean` door `std`. Zet haakjes om het aftrekken: `(crop / 255 - mean) / std`."),
            Mistake("own", matches=_own,
                    en="Every color of `x` has mean 0 and standard deviation 1: you scaled the photo with its own numbers. Scale it with those of ImageNet, `mean` and `std`, the numbers the model learned with.",
                    nl="Elke kleur van `x` heeft gemiddelde 0 en standaardafwijking 1: je hebt de foto met zijn eigen getallen geschaald. Schaal hem met die van ImageNet, `mean` en `std`, de getallen waarmee het model leerde."),
            Mistake("form", matches=lambda got, text: _tensor(got) and (tuple(got.shape) != (1, 3, 224, 224) or got.dtype != torch.float32),
                    detail=_form,
                    en="`x` is a tensor of shape `{shape}` with dtype {dtype}, and the model expects shape `(1, 3, 224, 224)` and float32: the colors first, a batch of one, and `.float()`.",
                    nl="`x` is een tensor met shape `{shape}` en dtype {dtype}, en het model verwacht shape `(1, 3, 224, 224)` en float32: de kleuren vooraan, een batch van één, en `.float()`."),
            Mistake("numbers", matches=lambda got, text: _tensor(got),
                    en="`x` has the right shape and dtype, but other numbers. Does the square start at the right place, and does every color go through `(crop / 255 - mean) / std`?",
                    nl="`x` heeft de goede shape en het goede dtype, maar andere getallen. Begint het vierkant op de goede plek, en gaat elke kleur door `(crop / 255 - mean) / std`?"),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: make `crop` the middle square of 224 by 224 of `small`, and `x` the tensor the model expects: `crop` scaled to 0 to 1 and then with `mean` and `std`, float32, the colors first and a batch axis in front.",
                      "nl": "Vervang de twee `...`: maak van `crop` het middelste vierkant van 224 bij 224 van `small`, en van `x` de tensor die het model verwacht: `crop` geschaald naar 0 tot en met 1 en daarna met `mean` en `std`, float32, de kleuren vooraan en een batch-as ervoor."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "no_value": {"en": "The last line of the box should be `x`, so the check sees it. Put it back.",
                         "nl": "De laatste regel van het vak moet `x` zijn, zodat de check hem ziet. Zet hem terug."},
            "wrong": {"en": "That is not yet the input of the model. `x` should be a float32 tensor of shape `(1, 3, 224, 224)`: the middle square of `small`, divided by 255, per color the mean taken off and divided by the standard deviation, the colors first, and a batch of one.",
                      "nl": "Dat is nog niet de input van het model. `x` moet een tensor van float32 zijn met shape `(1, 3, 224, 224)`: het middelste vierkant van `small`, gedeeld door 255, per kleur het gemiddelde eraf en gedeeld door de standaardafwijking, de kleuren vooraan, en een batch van één."},
            "correct": {"en": "Right: `x` is the cat as the model expects it, a float32 tensor of shape `(1, 3, 224, 224)`. The photo is upright, so the square starts at row 80 and column 16. Its red has mean 0.43 and its blue −0.61: the cat photo is warmer than the average ImageNet photo, and the scaling keeps that.",
                        "nl": "Goed: `x` is de kat zoals het model hem verwacht, een tensor van float32 met shape `(1, 3, 224, 224)`. De foto staat rechtop, dus het vierkant begint bij rij 80 en kolom 16. Zijn rood heeft gemiddelde 0,43 en zijn blauw −0,61: de kattenfoto is warmer dan de gemiddelde foto van ImageNet, en dat blijft na het schalen zo."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("crop", _CROP_START, "empty", None),
    ("crop", _CROP, "correct", None),
    ("crop", _crop(top="(img.shape[0] - size) // 2", left="(img.shape[1] - size) // 2"), "correct", None),
    ("crop", _crop(top="round((len(img) - size) / 2)", left="round((len(img[0]) - size) / 2)"), "correct", None),
    ("crop", _crop(back="img[top:top + size, left:left + size, ...]"), "correct", None),
    ("crop", _crop(top="(img.shape[0] - size) / 2", left="(img.shape[1] - size) / 2"), "error", "float"),
    ("crop", _crop(top="int((img.shape[1] - size) / 2)", left="int((img.shape[0] - size) / 2)"), "wrong", "swap"),
    ("crop", _crop(top="img.shape[0] - size", left="img.shape[1] - size"), "wrong", "no-half"),
    ("crop", _crop(top="int(img.shape[0] / 2)", left="int(img.shape[1] / 2)"), "wrong", "middle"),
    ("crop", _crop(top="int(size / 2)", left="int(size / 2)"), "wrong", "half-size"),
    ("crop", _crop(back="img[top:size, left:size]"), "wrong", "end"),
    ("crop", _crop(back="img[:size, :size]"), "wrong", "corner"),
    ("crop", _crop(back="img[top:top + size]"), "wrong", "rows-only"),
    ("crop", _crop(back="img[top:top + size][left:left + size]"), "wrong", "brackets"),
    ("crop", _crop(back="img[top:-top, left:-left]"), "wrong", "minus"),
    ("crop", _CROP_HEAD + "    h, w = img.shape\n    top = int((h - size) / 2)\n    left = int((w - size) / 2)\n"
             "    return img[top:top + size, left:left + size]\n", "error", "unpack"),
    ("crop", _CROP_HEAD + "    h, w = img.shape[:2]\n    top = (h - size) // 2\n    left = (w - size) // 2\n"
             "    return img[top:top + size, left:left + size]\n", "correct", None),
    ("prepare", _PREP_START, "empty", None),
    ("prepare", _PREP, "correct", None),
    ("prepare", _prep("torch.tensor((crop / 255 - mean) / std, dtype=torch.float32).permute(2, 0, 1)[None]"), "correct", None),
    ("prepare", _prep("torch.from_numpy((crop / 255 - mean) / std).float().unsqueeze(0).permute(0, 3, 1, 2)"), "correct", None),
    ("prepare", _PREP_HEAD + "top = int((small.shape[0] - 224) / 2)\nleft = int((small.shape[1] - 224) / 2)\n"
                "crop = small[top:top + 224, left:left + 224]\n"
                "x = torch.tensor((crop / 255 - mean) / std).float().permute(2, 0, 1).unsqueeze(0)\nx\n", "correct", None),
    ("prepare", _PREP_HEAD + "crop = small[80:304, 16:240]\nt = torch.tensor(crop / 255).float()\n"
                "t = (t - torch.tensor(mean).float()) / torch.tensor(std).float()\nx = t.permute(2, 0, 1).unsqueeze(0)\nx\n", "correct", None),
    ("prepare", _PREP_HEAD + "crop = ...\nx = torch.tensor((crop / 255 - mean) / std).float().permute(2, 0, 1).unsqueeze(0)\nx\n", "error", "dots"),
    ("prepare", _prep("torch.tensor((crop / 255 - mean) / std).float().unsqueeze(0).permute(2, 0, 1)"), "error", "permute-axes"),
    ("prepare", _PREP_HEAD + "crop = small[80:304, 16:240]\nt = torch.tensor(crop / 255).float().permute(2, 0, 1).unsqueeze(0)\n"
                "x = (t - torch.tensor(mean).float()) / torch.tensor(std).float()\nx\n", "error", "scale-late"),
    ("prepare", _PREP_HEAD + "crop = small[80:304, 16:240]\nt = np.transpose(crop / 255, (2, 0, 1))\n"
                "x = torch.tensor((t - mean) / std).float().unsqueeze(0)\nx\n", "error", "scale-late"),
    ("prepare", _prep("..."), "wrong", "x-dots"),
    ("prepare", _prep("(crop / 255 - mean) / std"), "wrong", "numpy"),
    ("prepare", _prep(crop="small"), "wrong", "no-crop"),
    ("prepare", _prep(crop="small[16:240, 80:304]"), "wrong", "coffee-crop"),
    ("prepare", _prep(crop="small[80:304]"), "wrong", "crop-size"),
    ("prepare", _prep("torch.tensor((crop / 255 - mean) / std).float().unsqueeze(0)"), "wrong", "channels-last"),
    ("prepare", _prep("torch.tensor((crop / 255 - mean) / std).float()"), "wrong", "channels-last"),
    ("prepare", _prep("torch.tensor((crop / 255 - mean) / std).float().permute(2, 0, 1)"), "wrong", "batch"),
    ("prepare", _prep("torch.tensor((crop / 255 - mean) / std).float().permute(2, 0, 1).unsqueeze(1)"), "wrong", "batch-place"),
    ("prepare", _prep("torch.tensor((crop / 255 - mean) / std).float().permute(1, 2, 0).unsqueeze(0)"), "wrong", "order"),
    ("prepare", _prep("torch.tensor((crop / 255 - mean) / std).float().permute(2, 1, 0).unsqueeze(0)"), "wrong", "sideways"),
    ("prepare", _prep("torch.tensor((crop / 255 - mean) / std).float().reshape(1, 3, 224, 224)"), "wrong", "reshape"),
    ("prepare", _prep("torch.tensor((crop / 255 - mean) / std).permute(2, 0, 1).unsqueeze(0)"), "wrong", "float64"),
    ("prepare", _prep("torch.tensor(crop).permute(2, 0, 1).unsqueeze(0)"), "wrong", "raw"),
    ("prepare", _prep("torch.tensor((crop - mean) / std).float().permute(2, 0, 1).unsqueeze(0)"), "wrong", "no-255"),
    ("prepare", _prep("torch.tensor((crop - mean) / std / 255).float().permute(2, 0, 1).unsqueeze(0)"), "wrong", "after"),
    ("prepare", _prep("torch.tensor((crop / 255 - std) / mean).float().permute(2, 0, 1).unsqueeze(0)"), "wrong", "swap"),
    ("prepare", _prep("torch.tensor(crop / 255 - mean).float().permute(2, 0, 1).unsqueeze(0)"), "wrong", "mean-only"),
    ("prepare", _prep("torch.tensor(crop / 255).float().permute(2, 0, 1).unsqueeze(0)"), "wrong", "only-255"),
    ("prepare", _prep("torch.tensor(crop / 255).permute(2, 0, 1).unsqueeze(0)"), "wrong", "only-255"),
    ("prepare", _prep("torch.tensor(crop / 255 - mean / std).float().permute(2, 0, 1).unsqueeze(0)"), "wrong", "brackets"),
    ("prepare", _PREP_HEAD + "crop = small[80:304, 16:240]\nc = crop / 255\n"
                "c = (c - c.mean(axis=0).mean(axis=0)) / np.array([c[:, :, 0].std(), c[:, :, 1].std(), c[:, :, 2].std()])\n"
                "x = torch.tensor(c).float().permute(2, 0, 1).unsqueeze(0)\nx\n", "wrong", "own"),
    ("prepare", _prep("torch.tensor((crop / 255 - mean) / std).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0)"), "wrong", "form"),
    ("prepare", _prep(crop="small[:224, :224]"), "wrong", "numbers"),
    ("prepare", _PREP_HEAD + "crop = small[80:304, 16:240]\nx = torch.tensor((crop / 255 - mean) / std).float().permute(2, 0, 1).unsqueeze(0)\n",
     "wrong", "no-value"),
]
